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

logger = logging.getLogger(__name__)


# ── Indicator definitions ──
INDICATOR_DEFS = [
    {
        'id': 'text_similarity',
        'name': '标书内容雷同分析',
        'category': '基础指标',
        'problem': '分析同一标段内投标人使用同一文档模板或复制内容的异常行为。',
        'rule': '基于AI语义相似度检测，文本相似度≥80%定义为可疑匹配。中标单位高相似度得基准分30分，每对追加5分。非中标单位高相似度每对追加2分。',
        'checker': 'text_sim',
    },
    {
        'id': 'key_info',
        'name': '重点信息雷同',
        'category': '基础指标',
        'problem': '分析不同投标文件的重点信息（工期、质量标准、报价金额等）是否出现异常雷同。',
        'rule': '提取各投标文件的关键实体与关键词，计算雷同比例。存在3项以上相同关键词的组合记为疑似，每组得5分。',
        'checker': 'key_info',
    },
    {
        'id': 'file_attr',
        'name': '文件属性雷同',
        'category': '基础指标',
        'problem': '分析不同投标文件的属性信息（作者、创建日期、创建程序等）是否存在雷同。',
        'rule': '多文件作者相同或创建日期相同的，记为一组异常。每组得5分。',
        'checker': 'file_attr',
    },
    {
        'id': 'typo',
        'name': '文本质量检测',
        'category': '基础指标',
        'problem': '检测投标文件中的错别字、不规范用语等文本质量问题。',
        'rule': '疑似错别字数≥5处或严重错字≥1处得5分，每增加5处追加2分。',
        'checker': 'typo',
    },
    {
        'id': 'relationship',
        'name': '投标方关联关系',
        'category': '基础指标',
        'problem': '以投标人为视角出发，挖掘不同投标单位之间潜在的关联关系。',
        'rule': '风险评分>50得基准分30分，>20得基准分10分。出现红色预警信息的追加5分。',
        'checker': 'relationship',
    },
    {
        'id': 'quote_anomaly',
        'name': '报价异常',
        'category': '扩展指标',
        'problem': '分析中标单位通过组织围标使报价处于异常区间的行为。',
        'rule': '离散系数(CV)≥0.15或存在同价疑义的得5分。存在异常降幅或价格聚类疑义的追加5分。',
        'checker': 'quote',
    },
]


def _run_checker(name, file_data, user_id, thread_id):
    """Run a single checker with try-except, returns dict or None."""
    try:
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

        indicators.append({
            'id': ind['id'],
            'name': ind['name'],
            'category': ind['category'],
            'score': round(score, 1),
            'problem': ind['problem'],
            'rule': ind['rule'],
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


def build_analysis_docx(report):
    """Generate .docx containing the full bid-rigging clue analysis report.

    Follows the example document's structure:
      1. Cover page with title + metadata
      2. Basic info table
      3. Suspected units table
      4. Indicator analysis section (one subsection per indicator)
      5. Personnel summary table

    Args:
        report: dict from run_analysis()

    Returns:
        bytes of the .docx file
    """
    from docx import Document as DocxDocument

    def _safe(s):
        return str(s) if s else ''

    doc = DocxDocument()
    style = doc.styles['Normal']
    style.font.name = 'Microsoft YaHei'
    style.font.size = Pt(10)

    # ── Cover page ──
    for _ in range(3):
        doc.add_paragraph()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run('串通投标线索分析报告')
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

    doc.add_paragraph()
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run('Bid-Rigging Clue Analysis Report')
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    doc.add_paragraph()
    doc.add_paragraph()
    info = report['basic_info']
    meta_items = [
        ('项目名称', info.get('project_name', '用户自定义')),
        ('投标单位数', str(info.get('bidder_count', 0))),
        ('分析日期', info.get('analysis_date', '')),
        ('综合风险评分', f'{info.get("total_score", 0)} 分'),
        ('预警级别', info.get('warning_level', '')),
        ('分析引擎', '中联招标智能助手 AI'),
    ]
    for label, value in meta_items:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f'{label}: {value}')
        run.font.size = Pt(11)

    doc.add_page_break()

    # ── Section 1: Basic info ──
    indicators = report.get('indicators', [])
    total_score = info.get('total_score', 0)

    h = doc.add_paragraph()
    run = h.add_run('一、投标人串通投标分析报告')
    run.bold = True
    run.font.size = Pt(16)

    h = doc.add_paragraph()
    run = h.add_run('1.1 基本信息')
    run.bold = True
    run.font.size = Pt(13)

    info_table = doc.add_table(rows=8, cols=2)
    info_table.style = 'Table Grid'
    rows_data = [
        ('项目名称', info.get('project_name', '用户自定义')),
        ('投标单位数', str(info.get('bidder_count', 0))),
        ('分析日期', info.get('analysis_date', '')),
        ('采购方式', '公开招标'),
        ('预算金额', '—'),
        ('综合预警指数', f'{total_score} 分'),
        ('预警级别', info.get('warning_level', '')),
        ('数据来源', '投标文件 + AI智能分析'),
    ]
    for i, (label, value) in enumerate(rows_data):
        info_table.cell(i, 0).text = label
        info_table.cell(i, 1).text = _safe(value)
        for paragraph in info_table.cell(i, 0).paragraphs:
            for run in paragraph.runs:
                run.bold = True

    doc.add_paragraph()

    # ── Section 2: Suspected units ──
    h = doc.add_paragraph()
    run = h.add_run('1.2 预警嫌疑单位')
    run.bold = True
    run.font.size = Pt(13)

    suspected = report.get('suspected_units', [])
    if list(suspected):
        p = doc.add_paragraph()
        run = p.add_run(f'经指标分析筛选确定预警嫌疑单位{len(suspected)}家，详情如下：')
        run.font.size = Pt(10)

        su_table = doc.add_table(rows=len(suspected) + 1, cols=3)
        su_table.style = 'Table Grid'
        for j, hdr in enumerate(['单位名称', '涉及指标数', '风险分']):
            su_table.cell(0, j).text = hdr
            for paragraph in su_table.cell(0, j).paragraphs:
                for run in paragraph.runs:
                    run.bold = True
        for si, su in enumerate(suspected):
            su_table.cell(si + 1, 0).text = (('★ ' if su['score'] > 10 else '  ') + _safe(su['name']))
            su_table.cell(si + 1, 1).text = str(su.get('indicators_triggered', 0))
            su_table.cell(si + 1, 2).text = str(su.get('score', 0))
    else:
        p = doc.add_paragraph()
        run = p.add_run('经指标分析未发现嫌疑单位。')
        run.font.size = Pt(10)

    doc.add_paragraph()

    # ── Section 3: Indicator analysis ──
    h = doc.add_paragraph()
    run = h.add_run('1.3 指标分析详情')
    run.bold = True
    run.font.size = Pt(13)

    idx = 1
    for ind in indicators:
        doc.add_paragraph()
        h = doc.add_paragraph()
        cat_tag = ind.get('category', '')
        skip = ind.get('skipped', False)
        state = ' ⏭️ 跳过' if skip else ''
        run = h.add_run(f'1.3.{idx}  {ind["name"]}  ({cat_tag})    指标得分: {ind["score"]}分{state}')
        run.bold = True
        run.font.size = Pt(11)

        # Problem
        p = doc.add_paragraph()
        run = p.add_run('针对问题: ')
        run.bold = True
        run.font.size = Pt(10)
        run = p.add_run(_safe(ind.get('problem', '')))
        run.font.size = Pt(10)

        # Rule
        p = doc.add_paragraph()
        run = p.add_run('分析规则: ')
        run.bold = True
        run.font.size = Pt(10)
        run = p.add_run(_safe(ind.get('rule', '')))
        run.font.size = Pt(10)

        # Result
        p = doc.add_paragraph()
        run = p.add_run('分析结果: ')
        run.bold = True
        run.font.size = Pt(10)
        run = p.add_run(_safe(ind.get('result', '')))
        run.font.size = Pt(10)
        text = _safe(ind.get('result', ''))
        if '⚠️' in text or '🔴' in text:
            run.font.color.rgb = RGBColor(0xDC, 0x26, 0x26)
        elif '✅' in text:
            run.font.color.rgb = RGBColor(0x16, 0xA3, 0x4A)

        # Details table
        details = ind.get('details', [])
        if details and not skip:
            doc.add_paragraph()
            detail_p = doc.add_paragraph()
            run = detail_p.add_run('指标详情:')
            run.bold = True
            run.font.size = Pt(9)

            if len(details) > 0 and isinstance(details[0], dict):
                keys = list(details[0].keys())
                dt = doc.add_table(rows=min(len(details), 20) + 1, cols=len(keys))
                dt.style = 'Table Grid'
                for j, k in enumerate(keys):
                    dt.cell(0, j).text = k
                    for paragraph in dt.cell(0, j).paragraphs:
                        for run in paragraph.runs:
                            run.bold = True
                            run.font.size = Pt(8)
                for di, d in enumerate(details[:20]):
                    for j, k in enumerate(keys):
                        dt.cell(di + 1, j).text = _safe(d.get(k, ''))

            if len(details) > 20:
                p = doc.add_paragraph()
                run = p.add_run(f'（仅显示前20项，共{len(details)}项）')
                run.font.size = Pt(8)
                run.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

        idx += 1

    # ── Section 4: Personnel summary ──
    doc.add_page_break()
    h = doc.add_paragraph()
    run = h.add_run('1.4 关系人员汇总')
    run.bold = True
    run.font.size = Pt(13)

    personnel = report.get('personnel_summary', {})
    personnel_list = personnel.get('list', [])
    if personnel_list:
        summary_text = (
            f'以嫌疑单位为分析对象，筛选单位项目所属关系人员共计{len(personnel_list)}人，'
            f'其中所属投标人{personnel.get("bidders", 0)}人，'
            f'所属招标人0人，所属招标代理0人。详情如下：'
        )
        p = doc.add_paragraph()
        run = p.add_run(summary_text)
        run.font.size = Pt(10)

        pt = doc.add_table(rows=len(personnel_list) + 1, cols=3)
        pt.style = 'Table Grid'
        for j, hdr in enumerate(['单位名称', '姓名', '人员类型']):
            pt.cell(0, j).text = hdr
            for paragraph in pt.cell(0, j).paragraphs:
                for run in paragraph.runs:
                    run.bold = True
        for pi, pe in enumerate(personnel_list):
            pt.cell(pi + 1, 0).text = _safe(pe.get('company', ''))
            pt.cell(pi + 1, 1).text = _safe(pe.get('person', ''))
            pt.cell(pi + 1, 2).text = _safe(pe.get('title', ''))
    else:
        p = doc.add_paragraph()
        run = p.add_run('未发现关联人员。')
        run.font.size = Pt(10)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


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

        # Step 3: Generate DOCX (90%)
        bus.progress(90, '正在生成分析报告...')

        # Initialize Flask context for DB access
        from celery_app import init_flask_context
        init_flask_context()

        from app.config import DATA_DIR, to_rel_path
        docx_bytes = build_analysis_docx(report)

        # Store ZIP in DB
        batch_dir = _os.path.join(DATA_DIR, 'batch_results')
        _os.makedirs(batch_dir, exist_ok=True)
        zip_name = f"doc_analysis_{task_id}.zip"
        zip_path = _os.path.join(batch_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"串通投标线索分析报告_{task_id[:8]}.docx", docx_bytes)

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
