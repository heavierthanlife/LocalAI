"""清标编排引擎 — 统一调度 4 个分析维度，合并为单一报告。

维度：
  1. 指标分析（纵向）：复用 run_analysis() → 45 项指标，填充报告一~五章
  2. 横向对比（横向）：复用 compute_all_pairs / key_info / attr → 第六章
  3. 合规审查（一步到位）：RuleExtractor + ComplianceChecker → 第七章（需招标文件）
  4. AI 评审（五轴）：agent + AI_DOC_REVIEW_PROMPT → 第八章（无 LLM key 时跳过）

输出：merged_report dict → build_clearance_docx() → DOCX + PDF → ZIP。
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── 是否有可用 LLM（AI 评审跳过条件）──────────────────────────────
_LLM_ENV_KEYS = ('DEEPSEEK_API_KEY', 'ZHIPU_API_KEY', 'QWEN_API_KEY', 'SILICONFLOW_API_KEY')


def has_llm():
    return any(os.getenv(k) for k in _LLM_ENV_KEYS)


def _now_str():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


# ── 维度 1: 指标分析（纵向）────────────────────────────────────────
def _run_indicator_analysis(file_data, user_id, thread_id):
    from app.services.document_analysis_svc import run_analysis
    report = run_analysis(file_data, user_id, thread_id)
    return {
        'basic_info': report['basic_info'],
        'suspected_units': report['suspected_units'],
        'indicators': report['indicators'],
        'personnel_summary': report['personnel_summary'],
        '_files': report['_files'],
        '_pairs': report.get('_pairs', []),
    }


# ── 维度 2: 横向对比（横向）────────────────────────────────────────
def _run_cross_comparison(file_data):
    from app.services.batch_orchestrator import (
        compute_all_pairs, build_key_info_matches, build_attr_details,
    )
    check_items = {
        'text_sim': True,
        'key_info': True,
        'file_attr': True,
        'image_sim': False,
    }
    pairs, risk_matrix = compute_all_pairs(file_data, check_items)
    return {
        'pairs': pairs,
        'risk_matrix': risk_matrix,
        'key_info_matches': build_key_info_matches(pairs),
        'attr_details': build_attr_details(file_data),
    }


# ── 维度 3: 合规审查（一步到位：提取规则 + 逐文件审查）────────────
def _run_compliance_check(tender_text, tender_name, file_data, use_ai=True):
    from app.services.rule_extractor import RuleExtractor
    from app.services.compliance_checker import ComplianceChecker

    if not tender_text:
        return None

    # 1) 提取招标文件规则
    try:
        extractor = RuleExtractor()
        rules_result = extractor.extract(tender_text, tender_name or '招标文件', use_ai=use_ai)
        rules = rules_result.get('rules', [])
    except Exception as e:
        logger.warning(f"Compliance rule extraction failed: {e}")
        rules = []
    if not rules:
        return {
            'rules': [],
            'per_file': [],
            'summary': {'pass': 0, 'warning': 0, 'violation': 0, 'critical': 0},
            'tender_name': tender_name,
            'skipped': True,
            'error': '未能从招标文件提取到规则',
        }

    # 2) 逐投标文件审查
    checker = ComplianceChecker()
    per_file = []
    overall = {'pass': 0, 'warning': 0, 'violation': 0, 'critical': 0}
    laws_applied = []
    for fd in file_data:
        try:
            result = checker.check(fd.get('text', ''), rules, fd.get('filename', ''), use_ai=use_ai)
        except Exception as e:
            logger.warning(f"Compliance check failed for {fd.get('filename')}: {e}")
            result = {
                'results': [], 'summary': {'pass': 0, 'warning': 0, 'violation': 0, 'critical': 0},
                'laws_applied': [], 'ai_used': False,
            }
        per_file.append({
            'filename': fd.get('filename', ''),
            'results': result.get('results', []),
            'summary': result.get('summary', {'pass': 0, 'warning': 0, 'violation': 0, 'critical': 0}),
        })
        s = result.get('summary', {})
        for k in overall:
            overall[k] += s.get(k, 0)
        for la in result.get('laws_applied', []):
            if la not in laws_applied:
                laws_applied.append(la)

    return {
        'rules': rules,
        'per_file': per_file,
        'summary': overall,
        'laws_applied': laws_applied[:20],
        'tender_name': tender_name,
        'ai_used': use_ai,
        'skipped': False,
    }


# ── 维度 4: AI 评审（五轴，逐文件）────────────────────────────────
def _run_ai_review(file_data):
    """五轴审查每个投标文件。无 LLM 或全部失败时返回 None。

    Uses create_chat_model directly (stateless, no Flask request context
    needed) so it can run inside a Celery worker thread.
    """
    if not has_llm():
        return None
    try:
        from app.routes.admin import AI_DOC_REVIEW_PROMPT
        from app.services.llm_provider import create_chat_model
        import json as _json
        import re as _re
        from langchain_core.messages import HumanMessage

        llm = create_chat_model(
            streaming=False, temperature=0.3, max_tokens=1500,
            timeout=int(os.getenv("LLM_TIMEOUT", "90")),
        )
    except Exception as e:
        logger.warning(f"AI review unavailable: {e}")
        return None

    per_file = []
    for fd in file_data:
        text = (fd.get('text') or '')[:12000]
        if len(text) < 50:
            continue
        try:
            resp = llm.invoke([HumanMessage(content=f"{AI_DOC_REVIEW_PROMPT}\n\n=== 待审查文档 ===\n{text}")])
            raw = resp.content if hasattr(resp, 'content') else str(resp)
            m = _re.search(r'\{[\s\S]*\}', raw)
            result = _json.loads(m.group(0)) if m else {"raw_analysis": raw, "parse_error": True}
        except Exception as e:
            logger.warning(f"AI review failed for {fd.get('filename')}: {e}")
            continue
        per_file.append({
            'filename': fd.get('filename', ''),
            'review': result,
        })
        if len(per_file) >= 5:
            break  # 控制成本：最多审查 5 份

    if not per_file:
        return None
    return {'per_file': per_file}


# ── 主编排入口 ─────────────────────────────────────────────────────
def run_clearance(file_data, tender_text, tender_name, options, user_id=None, thread_id=None):
    """执行清标全维度分析，返回合并后的报告 dict。

    options: {
        'indicator_analysis': bool,
        'cross_comparison': bool,
        'compliance_check': bool,
        'ai_review': bool,
    }
    """
    n = len(file_data)
    results = {}

    futures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        if options.get('indicator_analysis', True):
            futures['indicators'] = pool.submit(_run_indicator_analysis, file_data, user_id, thread_id)
        if options.get('cross_comparison', True):
            futures['cross'] = pool.submit(_run_cross_comparison, file_data)
        if options.get('compliance_check', False) and tender_text:
            futures['compliance'] = pool.submit(_run_compliance_check, tender_text, tender_name, file_data)
        if options.get('ai_review', True):
            futures['ai'] = pool.submit(_run_ai_review, file_data)

        name_by_future = {fut: name for name, fut in futures.items()}
        for fut in as_completed(futures.values()):
            name = name_by_future[fut]
            try:
                results[name] = fut.result()
            except Exception as e:
                logger.error(f"Clearance dimension {name} failed: {e}", exc_info=True)
                results[name] = None

    # ── 合并报告 ──
    indicator = results.get('indicators') or {}
    basic_info = indicator.get('basic_info') or {
        'project_name': '用户自定义',
        'bidder_count': n,
        'analysis_date': _now_str(),
        'total_score': 0,
        'warning_level': '🟢 正常',
    }
    # 冒烟指数叠加合规/AI 风险提示（保持 0-100 量纲）
    total_score = basic_info.get('total_score', 0)

    merged = {
        'basic_info': basic_info,
        'suspected_units': indicator.get('suspected_units', []),
        'indicators': indicator.get('indicators', []),
        'personnel_summary': indicator.get('personnel_summary', {'total': 0, 'bidders': 0, 'agents': 0, 'list': []}),
        '_files': indicator.get('_files', [fd.get('filename', '') for fd in file_data]),
        '_pairs': indicator.get('_pairs', []),
        'cross_comparison': results.get('cross'),
        'compliance': results.get('compliance'),
        'ai_review': results.get('ai'),
    }

    # 预警级别按最终总分（含合规严重违规加权）
    comp = merged['compliance']
    if comp and not comp.get('skipped') and comp.get('per_file'):
        criticals = sum(p['summary'].get('critical', 0) for p in comp['per_file'])
        violations = sum(p['summary'].get('violation', 0) for p in comp['per_file'])
        total_score += criticals * 5 + violations * 2
    merged['basic_info']['total_score'] = round(total_score, 1)
    merged['basic_info']['warning_level'] = (
        '🔴 高度预警' if total_score > 50 else ('🟠 中等预警' if total_score > 20 else '🟢 正常')
    )

    return merged


# ── Celery async task ───────────────────────────────────────────────
from celery_app import celery as _celery_app


@_celery_app.task(bind=True, name='clearance_task', max_retries=1)
def run_clearance_async(self, file_data, tender_text, tender_name, options,
                        user_id, thread_id, task_id, project_id=None):
    """Celery task: full clearance analysis → DOCX + PDF → ZIP → DB."""
    from app.services.task_bus import TaskBus
    from app.services.document_analysis_svc import build_clearance_docx, convert_docx_to_pdf
    import os as _os, zipfile, json

    bus = TaskBus(task_id, 'clearance', '清标分析')
    bus.start()

    try:
        n = len(file_data)
        bus.progress(5, f'开始清标，共 {n} 个投标文件...')

        bus.progress(15, '正在执行指标分析与横向对比...')
        report = run_clearance(file_data, tender_text, tender_name, options,
                               user_id=user_id, thread_id=thread_id)

        bus.progress(70, '正在生成清标报告...')

        from celery_app import init_flask_context
        init_flask_context()
        from app.config import DATA_DIR, to_rel_path

        docx_bytes = build_clearance_docx(report)

        pdf_bytes = None
        try:
            pdf_bytes = convert_docx_to_pdf(docx_bytes, task_id)
        except Exception as _e:
            logger.warning('Clearance PDF conversion failed, DOCX only: %s', _e)
            bus.progress(90, 'PDF 转换不可用，仅输出 DOCX')

        batch_dir = _os.path.join(DATA_DIR, 'batch_results')
        _os.makedirs(batch_dir, exist_ok=True)
        zip_name = f"clearance_{task_id}.zip"
        zip_path = _os.path.join(batch_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("串通投标线索分析报告.docx", docx_bytes)
            if pdf_bytes:
                zf.writestr("串通投标线索分析报告.pdf", pdf_bytes)

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
                            task_id, p.get('name1', ''), p.get('name2', ''),
                            round(p.get('sim', 0), 2), round(p.get('risk', 0), 2),
                            json.dumps(risk_scores, ensure_ascii=False), rank + 1,
                        ))
                conn.commit()

        from flask import url_for
        try:
            download_url = url_for('batch.download_batch_result', task_id=task_id, _external=True)
        except Exception:
            download_url = f'/batch_result/{task_id}'

        bus.progress(100, '清标完成')
        bus.complete({
            'report': report,
            'download_url': download_url,
            'file_count': len(file_data),
        })

    except Exception as e:
        logger.error(f"Clearance task failed: {e}", exc_info=True)
        bus.fail(str(e)[:500])
