"""Document analysis service — runs all checkers and produces structured reports.

Provides:
  - run_analysis():      runs all available checkers, returns example-structured report
  - build_analysis_docx(): generates .docx following the bid-rigging clue analysis format
"""
import logging
from datetime import datetime, timezone
from io import BytesIO

from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from celery_app import celery as celery_app

from app.services.batch_orchestrator import (
    compute_all_pairs, build_key_info_matches, build_attr_details,
    RiskScorer, truncate_filename,
)
from app.services.indicator_defs import INDICATOR_DEFS, rule_text

logger = logging.getLogger(__name__)


def _run_checker(name, file_data, user_id, thread_id):
    """Run a single checker with try-except, returns dict or None."""
    try:
        if name == 'skip':
            # Network/data-source dependent indicator — placeholder per design.
            return {'skipped': True, 'error': '需外部数据源（交易平台/评标系统数据）'}

        if name == 'text_sim':
            pairs, risk_matrix = compute_all_pairs(file_data, {
                'text_sim': True, 'key_info': False,
                'file_attr': False, 'image_sim': False
            })
            max_risk = max(p['risk'] for p in pairs) if pairs else 0
            max_sim = max(p['sim'] for p in pairs) if pairs else 0
            return {
                'pairs': pairs, 'risk_matrix': risk_matrix,
                'max_risk': max_risk, 'max_sim': max_sim,
            }

        elif name == 'key_info':
            pairs, risk_matrix = compute_all_pairs(file_data, {
                'text_sim': False, 'key_info': True,
                'file_attr': False, 'image_sim': False
            })
            key_info_matches = build_key_info_matches(pairs)
            return {'key_info_matches': key_info_matches}

        elif name == 'file_attr':
            attr_details = build_attr_details(file_data)
            author_groups = {}
            for ad in attr_details:
                a = ad.get('author', '')
                if a:
                    author_groups[a] = author_groups.get(a, []) + [ad['filename']]
            return {'attr_details': attr_details, 'author_groups': author_groups}

        elif name == 'typo':
            from app.services.typo_detector import detect_typos_batch
            results = detect_typos_batch(file_data)
            return {'results': results}

        elif name == 'relationship':
            from app.services.relationship_extractor import extract_relationships
            report = extract_relationships(file_data)
            return {'report': report}

        elif name == 'quote':
            from app.services.quote_anomaly import compare_bidders_quotes
            result = compare_bidders_quotes(file_data)
            return {'result': result}

    except Exception as e:
        logger.warning(f"Checker {name} failed: {e}")
        return {'error': str(e), 'skipped': True}


def run_analysis(file_data, user_id=None, thread_id=None):
    """Run all checkers, build a structured report matching the example format.

    Args:
        file_data: list of dicts with {'filename', 'text', 'metadata', 'images'}
        user_id: optional user identifier
        thread_id: optional thread identifier

    Returns:
        dict with basic_info, suspected_units, indicators, personnel_summary
    """
    n = len(file_data)
    _ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    # Run all checkers (each in try-except)
    checker_data = {}
    for ind in INDICATOR_DEFS:
        checker_data[ind['checker']] = _run_checker(
            ind['checker'], file_data, user_id, thread_id
        )

    # Build pairs and risk matrix from text_sim
    text_data = checker_data.get('text_sim', {})
    pairs = text_data.get('pairs', [])
    risk_matrix = text_data.get('risk_matrix', [[0]*n for _ in range(n)])
    max_risk = text_data.get('max_risk', 0)

    # Compute risk scores per file
    file_scores = {i: 0.0 for i in range(n)}
    for p in pairs:
        file_scores[p['i']] += p['risk'] * 0.5
        file_scores[p['j']] += p['risk'] * 0.5

    # Add scores from other checkers
    ki_data = checker_data.get('key_info', {})
    if ki_data.get('key_info_matches'):
        for ki in ki_data['key_info_matches']:
            for i in range(n):
                if file_data[i]['filename'] in (ki['name1'], ki['name2']):
                    file_scores[i] += min(len(ki.get('common_keywords', [])), 10) * 0.5

    attr_data = checker_data.get('file_attr', {})
    author_groups = attr_data.get('author_groups', {})
    for author, files in author_groups.items():
        if len(files) >= 2:
            for i in range(n):
                if file_data[i]['filename'] in files:
                    file_scores[i] += 5.0

    typo_data = checker_data.get('typo', {})
    if typo_data.get('results'):
        for name, report in typo_data['results'].items():
            for i in range(n):
                if file_data[i]['filename'] == name:
                    file_scores[i] += min(report.total_suspects * 0.5, 10)

    rel_data = checker_data.get('relationship', {})
    if rel_data.get('report') and hasattr(rel_data['report'], 'risk_score'):
        file_scores_boost = rel_data['report'].risk_score * 0.1
        for i in range(n):
            file_scores[i] += file_scores_boost

    quote_data = checker_data.get('quote', {})
    if quote_data.get('result'):
        qr = quote_data['result']
        for pb in qr.get('per_bidder', []):
            for i in range(n):
                if file_data[i]['filename'] == pb.get('filename'):
                    file_scores[i] += pb.get('risk_score', 0) * 0.1

    # Build suspected units
    suspected_units = []
    triggered_count = 0
    for i in range(n):
        score = round(file_scores[i], 1)
        if score > 0:
            triggered_count += 1
            suspected_units.append({
                'name': file_data[i]['filename'],
                'indicators_triggered': int(triggered_count > 0),
                'score': score,
            })

    # Build indicator results
    indicators = []
    for ind in INDICATOR_DEFS:
        cd = checker_data.get(ind['checker'], {})
        skipped = cd.get('skipped', False)
        error_msg = cd.get('error', '')

        score = 0.0
        result_text = ''
        details = []

        if ind['checker'] == 'text_sim':
            score = max_risk
            if max_risk > 10:
                high_pairs = [p for p in pairs if p['risk'] > 10]
                bidder_names = set()
                for p in high_pairs:
                    bidder_names.add(p['name1'])
                    bidder_names.add(p['name2'])
                result_text = f"⚠️ 发现异常：{', '.join(list(bidder_names)[:3])} 等{len(high_pairs)}对组合存在高风险文本相似（最高{max_risk:.1f}），存在围串标嫌疑。"
                details = [{'pair': f'{p["name1"]} ↔ {p["name2"]}', 'risk': f'{p["risk"]:.1f}', 'sim': f'{p["sim"]:.1f}%'}
                          for p in high_pairs[:10]]
            else:
                result_text = "✅ 未发现显著文本相似异常。"
            if skipped:
                result_text = f"⏭️ 跳过（{error_msg}）"

        elif ind['checker'] == 'key_info':
            ki_matches = ki_data.get('key_info_matches', [])
            if ki_matches:
                total = sum(len(ki.get('common_keywords', [])) for ki in ki_matches)
                score = min(total * 0.5, 20)
                result_text = f"⚠️ 发现{len(ki_matches)}组共享关键信息，共{total}个共同关键词。"
                details = [{'pair': f'{ki["name1"]} ↔ {ki["name2"]}',
                           'keywords': ', '.join(ki.get('common_keywords', [])[:10])}
                          for ki in ki_matches[:10]]
            else:
                result_text = "✅ 未发现重点信息雷同。"
            if skipped:
                result_text = f"⏭️ 跳过（{error_msg}）"

        elif ind['checker'] == 'file_attr':
            group_count = sum(1 for f in author_groups.values() if len(f) >= 2)
            score = group_count * 5.0
            if group_count > 0:
                result_text = f"⚠️ 发现{group_count}组文件属性雷同。"
                details = [{'author': a, 'files': ', '.join(fs), 'count': len(fs)}
                          for a, fs in author_groups.items() if len(fs) >= 2]
            else:
                result_text = "✅ 未发现文件属性雷同。"
            if skipped:
                result_text = f"⏭️ 跳过（{error_msg}）"

        elif ind['checker'] == 'typo':
            typos = typo_data.get('results', {})
            if typos:
                total = sum(r.total_suspects for r in typos.values())
                critical = sum(r.critical_count for r in typos.values())
                if total > 0:
                    score = min(total * 0.5 + critical * 2, 15)
                    result_text = f"⚠️ 发现{total}处疑似错别字（其中严重{critical}处）。"
                    details = [{'file': n, 'total': r.total_suspects, 'critical': r.critical_count}
                              for n, r in typos.items() if r.total_suspects > 0][:10]
                else:
                    result_text = "✅ 未发现文本质量问题。"
            else:
                result_text = "✅ 文本质量检测未发现异常。"
            if skipped:
                result_text = f"⏭️ 跳过（{error_msg}）"

        elif ind['checker'] == 'relationship':
            report = rel_data.get('report')
            if report:
                rs = report.risk_score if hasattr(report, 'risk_score') else 0
                red = len(report.red_flags) if hasattr(report, 'red_flags') else 0
                ents = len(report.entities) if hasattr(report, 'entities') else 0
                rels = len(report.relationships) if hasattr(report, 'relationships') else 0
                score = min(rs, 30)
                if red > 0:
                    result_text = f"🔴 发现{red}个红色预警，{ents}个实体，{rels}个关系，风险评分{rs:.1f}。"
                    details = [{'flag': f} for f in (report.red_flags if hasattr(report, 'red_flags') else [])[:10]]
                else:
                    result_text = f"✅ 未发现高危关联，{ents}个实体，{rels}个关系，风险评分{rs:.1f}。"
            else:
                result_text = "✅ 未发现关联关系。"
            if skipped:
                result_text = f"⏭️ 跳过（{error_msg}）"

        elif ind['checker'] == 'quote':
            qr = quote_data.get('result', {})
            per_bidder = qr.get('per_bidder', [])
            if per_bidder:
                max_qr = qr.get('max_risk_score', 0)
                suspicious = [pb for pb in per_bidder if pb.get('risk_score', 0) > 10]
                score = min(max_qr, 20)
                if suspicious:
                    result_text = f"⚠️ 发现{suspicious}个投标单位报价疑义。最高风险评分{max_qr:.1f}。"
                    details = [{'bidder': pb.get('filename', ''), 'risk': f'{pb.get("risk_score", 0):.1f}',
                               'cv': f'{pb.get("cv", 0):.4f}',
                               'flags': ', '.join([k for k in ['same_rate_flag', 'abnormal_drop_flag', 'clustering_flag']
                                                   if pb.get(k)] or ['正常'])}
                              for pb in per_bidder]
                else:
                    result_text = f"✅ 报价分析未发现异常。平均CV: {qr.get('avg_cv', 0):.4f}。"
            else:
                result_text = "✅ 报价分析未发现异常。"
            if skipped:
                result_text = f"⏭️ 跳过（{error_msg}）"

        else:
            # Data-source dependent indicator — skipped placeholder
            skipped = True
            result_text = '⏭️ 需外部数据源（交易平台/评标系统数据）'

        rr = ind.get('rule_ref', {})
        indicators.append({
            'id': ind['id'],
            'name': ind['name'],
            'category': ind['category'],
            'score': round(score, 1),
            'problem': ind['problem'],
            'rule': ind['rule'],
            'rule_ref': rr,
            'rule_text': rule_text(rr.get('law', ''), rr.get('article', '')),
            'result': result_text,
            'details': details,
            'skipped': skipped,
        })

    # Build personnel summary
    personnel = []
    rel_data = checker_data.get('relationship', {})
    report = rel_data.get('report')
    if report and hasattr(report, 'company_personnel_map'):
        cpm = report.company_personnel_map
        for comp in cpm.get('companies', [])[:20]:
            for p in comp.get('personnel', []):
                personnel.append({
                    'company': comp.get('name', ''),
                    'person': p.get('name', ''),
                    'title': p.get('title', ''),
                })

    total_score = round(sum(ind['score'] for ind in indicators), 1)

    return {
        '_pairs': pairs,
        '_files': [fd.get('filename', '') for fd in file_data],
        'basic_info': {
            'project_name': '用户自定义',
            'bidder_count': n,
            'analysis_date': _ts,
            'total_score': total_score,
            'warning_level': '🟠 中等预警' if total_score > 20 else (
                '🔴 高度预警' if total_score > 50 else '🟢 正常'
            ),
        },
        'suspected_units': suspected_units,
        'indicators': indicators,
        'personnel_summary': {
            'total': len(personnel),
            'bidders': sum(1 for p in personnel if '投标' in p.get('title', '')),
            'agents': sum(1 for p in personnel if '代理' in p.get('title', '')),
            'list': personnel[:30],
        },
    }


def _safe(s):
    return str(s) if s else ''


def _add_heading(doc, text, size, bold=True):
    h = doc.add_paragraph()
    run = h.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    return h


def _add_table(doc, rows, cols):
    t = doc.add_table(rows=rows, cols=cols)
    t.style = 'Table Grid'
    return t


def _merge_row(t, row, start, end):
    merged = t.cell(row, start)
    for ci in range(start + 1, end + 1):
        merged = merged.merge(t.cell(row, ci))
    return merged


def _set_cell(cell, text, bold=False, size=9):
    cell.text = ''
    p = cell.paragraphs[0]
    run = p.add_run(_safe(text))
    run.bold = bold
    run.font.size = Pt(size)
    return run


def _build_standard_sections(doc, report):
    """Build cover + sections 一~五 (shared by analysis & clearance docx)."""
    info = report['basic_info']

    # ── Cover page ──
    for _ in range(6):
        doc.add_paragraph()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run('串通投标线索分析报告')
    run.bold = True
    run.font.size = Pt(40)

    for _ in range(4):
        doc.add_paragraph()

    doc.add_page_break()

    # ── Section 1: 分析报告综述 ──
    _add_heading(doc, '一、分析报告综述', 26)

    info_table = _add_table(doc, 8, 4)
    bidder_count = info.get('bidder_count', 0)
    total_score = info.get('total_score', 0)
    rows_data = [
        ('标段名称', [info.get('project_name', '用户自定义')] * 3),
        ('标段编号', '—', '开标时间', '—'),
        ('招标单位', '—', '招标代理', '—'),
        ('评标办法', '—', '中标公告发布时间', '—'),
        ('中标单位', '—', '中标金额', '—'),
        ('冒烟指数', f'{total_score} 分', '预警级别', info.get('warning_level', '')),
        ('地区', '—', '监督部门', '—'),
        ('真实交易平台', ['—'] * 2),
    ]
    for ri, row in enumerate(rows_data):
        if ri in (0, 7):
            merged = _merge_row(info_table, ri, 1, 3)
            _set_cell(info_table.cell(ri, 0), row[0], bold=True)
            _set_cell(merged, row[1][0])
        else:
            for ci in range(4):
                _set_cell(info_table.cell(ri, ci), row[ci], bold=(ci % 2 == 0))

    doc.add_paragraph()

    # ── Section 2: 预警单位汇总 ──
    _add_heading(doc, '二、预警单位汇总', 26)
    suspected = report.get('suspected_units', [])
    p = doc.add_paragraph()
    run = p.add_run(f'经过对上述所有指标的深度分析，筛选嫌疑单位{len(suspected)}家(不含招标人与招标代理),详情如下：')
    run.bold = True
    run.font.size = Pt(18.5)

    su_table = _add_table(doc, len(suspected) + 1, 2)
    _set_cell(su_table.cell(0, 0), '单位名称', bold=True)
    _set_cell(su_table.cell(0, 1), '涉及指标数量', bold=True)
    for si, su in enumerate(suspected):
        _set_cell(su_table.cell(si + 1, 0), (('★ ' if su['score'] > 10 else '  ') + _safe(su['name'])))
        _set_cell(su_table.cell(si + 1, 1), str(su.get('indicators_triggered', 0)))

    doc.add_paragraph()

    # ── Section 3: 指标详情分析 ──
    _add_heading(doc, '三、指标详情分析', 26)
    p = doc.add_paragraph()
    run = p.add_run(
        '分析指标指的是经过大量串通投标历史案例的调研，分析此类犯罪行为特征、提炼数据规则，形成的'
        '衡量串通投标风险的依据。根据发生串通投标行为的概率为指标进行了归类，分为触发指标、核心指标、'
        '基础指标、扩展指标。'
    )
    run.bold = True
    run.font.size = Pt(17)

    cat_defs = [
        '触发指标：能够直接作为串通投标线索的指标。',
        '核心指标：能够深层次挖掘围串标主体之间潜在联系的指标。',
        '基础指标：能够从一定程度上反映出串通投标线索的指标。',
        '扩展指标：不影响冒烟指数得分但会引申出其他非主要可疑线索的指标。',
    ]
    for cd in cat_defs:
        p = doc.add_paragraph()
        run = p.add_run(cd)
        run.bold = True
        run.font.size = Pt(17)

    indicators = report.get('indicators', [])
    group_meta = [
        ('触发指标', '3.1、'),
        ('核心指标', '3.2、'),
        ('基础指标', '3.3、'),
        ('扩展指标', '3.4、'),
    ]
    for gi, (cat, prefix) in enumerate(group_meta):
        group = [ind for ind in indicators if ind['category'] == cat]
        if not group:
            continue
        _add_heading(doc, f'{prefix}{cat}', 20.5)
        for si, ind in enumerate(group):
            sub_no = f'3.{gi + 1}.{si + 1}'
            _add_heading(doc, f'{sub_no}、{ind["name"]}', 17.5)

            t = _add_table(doc, 6, 4)
            _set_cell(t.cell(0, 0), '指标名称', bold=True)
            _merge_row(t, 0, 1, 3)
            _set_cell(t.cell(0, 1), ind['name'])

            _set_cell(t.cell(1, 0), '指标类别', bold=True)
            _set_cell(t.cell(1, 1), ind['category'])
            _set_cell(t.cell(1, 2), '指标得分', bold=True)
            _set_cell(t.cell(1, 3), f"{ind.get('score', 0)} 分")

            _set_cell(t.cell(2, 0), '针对问题', bold=True)
            _merge_row(t, 2, 1, 3)
            _set_cell(t.cell(2, 1), ind.get('problem', ''))

            _set_cell(t.cell(3, 0), '分析规则', bold=True)
            _merge_row(t, 3, 1, 3)
            rule_text_val = ind.get('rule', '')
            rule_ref = ind.get('rule_ref', {})
            law_text = ind.get('rule_text', '')
            if law_text:
                rule_text_val += f"\n规则依据：{rule_ref.get('label', '')} {law_text}"
            _set_cell(t.cell(3, 1), rule_text_val)

            _set_cell(t.cell(4, 0), '分析结果', bold=True)
            _merge_row(t, 4, 1, 3)
            result_run = _set_cell(t.cell(4, 1), ind.get('result', ''))
            result_text = _safe(ind.get('result', ''))
            if '⚠️' in result_text or '🔴' in result_text:
                result_run.font.color.rgb = RGBColor(0xDC, 0x26, 0x26)
            elif '✅' in result_text:
                result_run.font.color.rgb = RGBColor(0x16, 0xA3, 0x4A)
            elif '⏭️' in result_text:
                result_run.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

            _set_cell(t.cell(5, 0), '指标详情', bold=True)
            details = ind.get('details', [])
            if details and not ind.get('skipped', False):
                _merge_row(t, 5, 1, 3)
                detail_cell = t.cell(5, 1)
                detail_cell.text = ''
                dp = detail_cell.paragraphs[0]
                if isinstance(details[0], dict):
                    for d in details[:10]:
                        rp = dp
                        for k, v in list(d.items())[:4]:
                            rp = detail_cell.add_paragraph()
                            rn = rp.add_run(f'{k}: {_safe(v)}')
                            rn.font.size = Pt(8)
                        if len(details) > 10:
                            rp = detail_cell.add_paragraph()
                            rn = rp.add_run(f'…（共{len(details)}项）')
                            rn.font.size = Pt(8)
                            break
                else:
                    rn = dp.add_run(_safe(details[0]))
                    rn.font.size = Pt(8)
            else:
                _merge_row(t, 5, 1, 3)

    # ── Section 4: 关系人员汇总 ──
    doc.add_page_break()
    _add_heading(doc, '四、关系人员汇总', 26, bold=True)
    personnel = report.get('personnel_summary', {})
    personnel_list = personnel.get('list', [])
    p = doc.add_paragraph()
    run = p.add_run(
        f'以冒烟嫌疑单位为分析对象，筛选单位项目所属关系人员共计{len(personnel_list)}人，'
        f'其中所属投标人{personnel.get("bidders", 0)}人，所属招标人0人，所属招标代理0人。详情如下：'
    )
    run.bold = True
    run.font.size = Pt(18.5)

    pt = _add_table(doc, len(personnel_list) + 1, 6)
    for j, hdr in enumerate(['单位名称', '单位类型', '姓名', '身份证号', '联系电话', '人员类型']):
        _set_cell(pt.cell(0, j), hdr, bold=True)
    for pi, pe in enumerate(personnel_list):
        _set_cell(pt.cell(pi + 1, 0), _safe(pe.get('company', '')))
        _set_cell(pt.cell(pi + 1, 1), '投标人')
        _set_cell(pt.cell(pi + 1, 2), _safe(pe.get('person', '')))
        _set_cell(pt.cell(pi + 1, 3), '')
        _set_cell(pt.cell(pi + 1, 4), '')
        _set_cell(pt.cell(pi + 1, 5), _safe(pe.get('title', '')))

    doc.add_paragraph()

    # ── Section 5: 开标信息表 ──
    _add_heading(doc, '五、开标信息表', 26, bold=True)
    files = report.get('_files', [])
    bid_table = _add_table(doc, len(files) + 1, 15)
    headers = ['序号', '开标时间', '投标单位', '联系人', '联系电话', '文件下载IP', '标书上传时间',
               '标书上传IP', '解密状态', '解密IP', '文件码', '加密锁', '报价方式', '投标报价', '备注']
    for j, hdr in enumerate(headers):
        _set_cell(bid_table.cell(0, j), hdr, bold=True, size=8)
    for fi, fname in enumerate(files):
        _set_cell(bid_table.cell(fi + 1, 0), str(fi + 1), size=8)
        _set_cell(bid_table.cell(fi + 1, 2), _safe(fname), size=8)
        for j in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14):
            _set_cell(bid_table.cell(fi + 1, j), '', size=8)

    doc.add_paragraph()


def _add_date_line(doc, report):
    """Append the closing date line (ends the document)."""
    info = report.get('basic_info', {})
    p = doc.add_paragraph()
    run = p.add_run(_safe(info.get('analysis_date', '')))
    run.font.size = Pt(18.5)


def _new_docx():
    from docx import Document as DocxDocument
    doc = DocxDocument()
    style = doc.styles['Normal']
    style.font.name = 'Microsoft YaHei'
    style.font.size = Pt(10)
    return doc


def build_analysis_docx(report):
    """Generate .docx strictly replicating the reference report layout.

    Reference structure (串通投标线索分析报告.docx):
      1. Cover page: centered title 40pt
      2. 一、分析报告综述 (26pt) + basic info table (8 rows x 4 cols)
      3. 二、预警单位汇总 (26pt) + intro 18.5pt + suspected units table (2 cols)
      4. 三、指标详情分析 (26pt) + category definitions 17pt
         - 3.1、触发指标 (20.5pt) / 3.2、核心指标 / 3.3、基础指标 / 3.4、扩展指标
         - per-indicator subsection + 6-row table
      5. 四、关系人员汇总 (26pt bold) + personnel table (6 cols)
      6. 五、开标信息表 (26pt bold) + bid-opening table (15 cols)
      7. Closing date line

    Args:
        report: dict from run_analysis()

    Returns:
        bytes of the .docx file
    """
    doc = _new_docx()
    _build_standard_sections(doc, report)
    _add_date_line(doc, report)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ── 清标 docx：标准五章 + 六横向对比 / 七合规 / 八 AI 评审 ──

def build_clearance_docx(report):
    """Generate the unified 清标 .docx.

    Sections 一~五 follow the reference report exactly; sections 六/七/八
    hold the cross-comparison / compliance / AI-review dimensions that do
    not fit the reference layout, appended as their own tables at the end.

    report: dict from clearance_engine.run_clearance()
    """
    doc = _new_docx()
    _build_standard_sections(doc, report)

    # ── Section 6: 横向对比分析 ──
    cross = report.get('cross_comparison')
    if cross:
        pairs = cross.get('pairs', [])
        risk_matrix = cross.get('risk_matrix', [])
        key_info = cross.get('key_info_matches', [])
        attr = cross.get('attr_details', [])
        files = report.get('_files', [])

        _add_heading(doc, '六、横向对比分析', 26, bold=True)

        # 6a. 风险矩阵表（N+1 x N+1）
        _add_heading(doc, '6.1、风险矩阵', 17.5)
        n = len(files)
        mt = _add_table(doc, n + 1, n + 1)
        _set_cell(mt.cell(0, 0), '投标单位', bold=True, size=8)
        for j, fname in enumerate(files):
            _set_cell(mt.cell(0, j + 1), _safe(truncate_filename(fname, 8)), bold=True, size=8)
        for i in range(n):
            _set_cell(mt.cell(i + 1, 0), _safe(truncate_filename(files[i], 8)), bold=True, size=8)
            for j in range(n):
                val = '--' if i == j else (
                    f"{risk_matrix[i][j]:.1f}" if i < len(risk_matrix) and j < len(risk_matrix[i]) else '--')
                _set_cell(mt.cell(i + 1, j + 1), val, size=8)
        doc.add_paragraph()

        # 6b. 高风险对详情表
        high_pairs = [p for p in pairs if p.get('risk', 0) > 5]
        if high_pairs:
            _add_heading(doc, '6.2、高风险组合详情', 17.5)
            pt = _add_table(doc, len(high_pairs) + 1, 5)
            for j, hdr in enumerate(['投标单位1', '投标单位2', '风险度', '文本相似度', '文件属性雷同']):
                _set_cell(pt.cell(0, j), hdr, bold=True, size=8)
            for ri, p in enumerate(high_pairs):
                _set_cell(pt.cell(ri + 1, 0), _safe(p.get('name1', '')), size=8)
                _set_cell(pt.cell(ri + 1, 1), _safe(p.get('name2', '')), size=8)
                _set_cell(pt.cell(ri + 1, 2), f"{p.get('risk', 0):.1f}", size=8)
                _set_cell(pt.cell(ri + 1, 3), f"{p.get('sim', 0):.1f}%", size=8)
                _set_cell(pt.cell(ri + 1, 4), '是' if p.get('attr_same') else '否', size=8)
            doc.add_paragraph()

        # 6c. 关键信息匹配表
        if key_info:
            _add_heading(doc, '6.3、重点信息雷同', 17.5)
            kt = _add_table(doc, len(key_info) + 1, 3)
            for j, hdr in enumerate(['投标单位1', '投标单位2', '共同关键词']):
                _set_cell(kt.cell(0, j), hdr, bold=True, size=8)
            for ri, ki in enumerate(key_info):
                _set_cell(kt.cell(ri + 1, 0), _safe(ki.get('name1', '')), size=8)
                _set_cell(kt.cell(ri + 1, 1), _safe(ki.get('name2', '')), size=8)
                _set_cell(kt.cell(ri + 1, 2), ', '.join(ki.get('common_keywords', [])[:10]), size=8)
            doc.add_paragraph()

    # ── Section 7: 合规审查结果 ──
    comp = report.get('compliance')
    if comp and not comp.get('skipped'):
        _add_heading(doc, '七、合规审查结果', 26, bold=True)
        p = doc.add_paragraph()
        run = p.add_run(f'基于招标文件《{_safe(comp.get("tender_name", ""))}》提取的 {len(comp.get("rules", []))} 条规则，对各单位投标文件逐一审查。')
        run.bold = True
        run.font.size = Pt(18.5)

        for pf in comp.get('per_file', []):
            _add_heading(doc, f'7.{comp.get("per_file", []).index(pf) + 1}、{_safe(pf.get("filename", ""))}', 17.5)
            s = pf.get('summary', {})
            st = _add_table(doc, 2, 4)
            for j, hdr in enumerate(['通过', '警告', '违规', '严重']):
                _set_cell(st.cell(0, j), hdr, bold=True, size=9)
            for j, key in enumerate(['pass', 'warning', 'violation', 'critical']):
                _set_cell(st.cell(1, j), str(s.get(key, 0)), size=9)
            doc.add_paragraph()

            results = pf.get('results', [])
            if results:
                rule_map = {r.get('rule_id'): r.get('description', '') for r in comp.get('rules', [])}
                rt = _add_table(doc, len(results) + 1, 5)
                for j, hdr in enumerate(['规则编号', '规则描述', '审查结论', '证据', '建议']):
                    _set_cell(rt.cell(0, j), hdr, bold=True, size=8)
                for ri, res in enumerate(results):
                    _set_cell(rt.cell(ri + 1, 0), _safe(res.get('rule_id', '')), size=8)
                    _set_cell(rt.cell(ri + 1, 1), _safe(rule_map.get(res.get('rule_id', ''), ''))[:80], size=8)
                    _set_cell(rt.cell(ri + 1, 2), _safe(res.get('verdict', '')), size=8)
                    _set_cell(rt.cell(ri + 1, 3), _safe(res.get('evidence', ''))[:80], size=8)
                    _set_cell(rt.cell(ri + 1, 4), _safe(res.get('suggestion', ''))[:80], size=8)
                doc.add_paragraph()

    # ── Section 8: AI 综合评审 ──
    ai = report.get('ai_review')
    if ai and ai.get('per_file'):
        _add_heading(doc, '八、AI 综合评审', 26, bold=True)
        for pi, pf in enumerate(ai.get('per_file', [])):
            review = pf.get('review', {})
            _add_heading(doc, f'8.{pi + 1}、{_safe(pf.get("filename", ""))}', 17.5)

            scores = review.get('scores') or {}
            if scores:
                st = _add_table(doc, 2, len(scores) + 2)
                _set_cell(st.cell(0, 0), '综合评分', bold=True, size=9)
                _set_cell(st.cell(1, 0), _safe(review.get('overall', '')), size=9)
                for j, (k, v) in enumerate(scores.items(), start=1):
                    _set_cell(st.cell(0, j), _safe(k), bold=True, size=9)
                    _set_cell(st.cell(1, j), _safe(v), size=9)
                _set_cell(st.cell(0, len(scores) + 1), '结论', bold=True, size=9)
                _set_cell(st.cell(1, len(scores) + 1), _safe(review.get('verdict', '')), size=9)
                doc.add_paragraph()

            issues = review.get('issues') or []
            if issues:
                it = _add_table(doc, len(issues) + 1, 4)
                for j, hdr in enumerate(['维度', '严重度', '问题', '建议']):
                    _set_cell(it.cell(0, j), hdr, bold=True, size=8)
                for ri, iss in enumerate(issues):
                    _set_cell(it.cell(ri + 1, 0), _safe(iss.get('axis', '')), size=8)
                    _set_cell(it.cell(ri + 1, 1), _safe(iss.get('severity', '')), size=8)
                    _set_cell(it.cell(ri + 1, 2), _safe(iss.get('finding', ''))[:100], size=8)
                    _set_cell(it.cell(ri + 1, 3), _safe(iss.get('suggestion', ''))[:100], size=8)
                doc.add_paragraph()

            if review.get('summary'):
                p = doc.add_paragraph()
                run = p.add_run(f"AI 综合意见：{_safe(review.get('summary', ''))}")
                run.font.size = Pt(10)

    _add_date_line(doc, report)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ── LibreOffice PDF conversion ──

def convert_docx_to_pdf(docx_bytes, task_id=''):
    """Convert .docx bytes to .pdf bytes via headless LibreOffice.

    Uses LIBREOFFICE_BIN env (default: 'soffice'). Runs in a temp dir so
    --convert-to keeps a clean output filename. Returns None if the
    soffice binary is unavailable (caller falls back to DOCX-only ZIP).

    Args:
        docx_bytes: bytes of the .docx document
        task_id: optional id used in the temp file name (avoid collisions)

    Returns:
        bytes of the converted .pdf, or None if LibreOffice unavailable
    """
    import os as _os, shutil, subprocess, tempfile, time as _time

    soffice = _os.environ.get('LIBREOFFICE_BIN', 'soffice')
    if not shutil.which(soffice):
        return None

    with tempfile.TemporaryDirectory(prefix='docx2pdf_') as tmp:
        src_name = f"report_{task_id or int(_time.time())}.docx"
        src = _os.path.join(tmp, src_name)
        with open(src, 'wb') as f:
            f.write(docx_bytes)
        result = subprocess.run(
            [soffice, '--headless', '--convert-to', 'pdf', '--outdir', tmp, src],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0:
            logger.warning('soffice convert failed: %s', result.stderr.decode('utf-8', 'replace')[:500])
            return None
        pdf_path = _os.path.join(tmp, src_name.replace('.docx', '.pdf'))
        if not _os.path.exists(pdf_path):
            return None
        with open(pdf_path, 'rb') as f:
            return f.read()


# ── Celery async task ──

@celery_app.task(bind=True, name='document_analysis_task', max_retries=1)
def run_analysis_async(self, file_data, user_id, thread_id, task_id, project_id=None):
    """Celery task: runs full analysis + generates DOCX + stores ZIP in DB."""
    from app.services.task_bus import TaskBus
    import os as _os, zipfile, hashlib, json

    bus = TaskBus(task_id, 'doc_analysis', '投标文档深度分析')
    bus.start()

    try:
        # Step 1: Run analysis (25%)
        bus.progress(5, '正在提取文件内容...')
        n = len(file_data)

        bus.progress(15, f'开始分析 {n} 个文件...')

        # Step 2: Run all checkers
        bus.progress(25, '正在进行文本相似度分析...')
        report = run_analysis(file_data, user_id, thread_id)

        bus.progress(40, '正在进行重点信息雷同分析...')
        bus.progress(55, '正在进行文件属性对比...')
        bus.progress(65, '正在进行文本质量检测...')
        bus.progress(75, '正在进行关联关系分析...')
        bus.progress(85, '正在进行报价异常检测...')

        # Step 3: Generate DOCX + PDF (90%)
        bus.progress(90, '正在生成分析报告...')

        # Initialize Flask context for DB access
        from celery_app import init_flask_context
        init_flask_context()

        from app.config import DATA_DIR, to_rel_path
        docx_bytes = build_analysis_docx(report)

        # Step 4: Convert to PDF via LibreOffice (headless, Docker). Falls back to DOCX-only.
        pdf_bytes = None
        try:
            pdf_bytes = convert_docx_to_pdf(docx_bytes, task_id)
        except Exception as _e:
            logger.warning('PDF conversion failed, falling back to DOCX only: %s', _e)
            bus.progress(95, 'PDF 转换不可用，仅输出 DOCX')
            pdf_bytes = None

        # Store ZIP in DB
        batch_dir = _os.path.join(DATA_DIR, 'batch_results')
        _os.makedirs(batch_dir, exist_ok=True)
        zip_name = f"doc_analysis_{task_id}.zip"
        zip_path = _os.path.join(batch_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"串通投标线索分析报告_{task_id[:8]}.docx", docx_bytes)
            if pdf_bytes:
                zf.writestr(f"串通投标线索分析报告_{task_id[:8]}.pdf", pdf_bytes)

        file_names = [fd['filename'] for fd in file_data]
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO batch_comparison_results (user_id, task_id, project_id, file_count, pair_count, max_risk, file_names, zip_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id, task_id, project_id, len(file_data), 0,
                    report['basic_info'].get('total_score', 0),
                    json.dumps(file_names, ensure_ascii=False),
                    to_rel_path(zip_path),
                ))

                pairs = report.get('_pairs', [])
                if pairs:
                    ranked = sorted(pairs, key=lambda p: p.get('risk', 0), reverse=True)
                    for rank, p in enumerate(ranked):
                        risk_scores = {
                            'text_sim': round(p.get('sim', 0), 2),
                            'attr_same': p.get('attr_same', 0),
                            'risk': round(p.get('risk', 0), 2),
                        }
                        cur.execute("""
                            INSERT INTO batch_pair_results (task_id, file_a, file_b, similarity, max_risk, risk_scores, pair_rank)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (task_id, file_a, file_b) DO UPDATE SET
                                similarity = EXCLUDED.similarity,
                                max_risk = EXCLUDED.max_risk,
                                risk_scores = EXCLUDED.risk_scores,
                                pair_rank = EXCLUDED.pair_rank
                        """, (
                            task_id,
                            p.get('name1', ''),
                            p.get('name2', ''),
                            round(p.get('sim', 0), 2),
                            round(p.get('risk', 0), 2),
                            json.dumps(risk_scores, ensure_ascii=False),
                            rank + 1,
                        ))

                conn.commit()

        from flask import url_for
        try:
            download_url = url_for('batch.download_batch_result', task_id=task_id, _external=True)
        except Exception:
            from app.config import BASE_DIR
            download_url = f'/batch_result/{task_id}'

        bus.progress(100, '分析完成')
        bus.complete({
            'report': report,
            'download_url': download_url,
            'file_count': len(file_data),
        })

    except Exception as e:
        logger.error(f"Analysis task failed: {e}", exc_info=True)
        bus.fail(str(e)[:500])
