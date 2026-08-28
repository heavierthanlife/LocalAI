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
def _run_indicator_analysis(file_data, user_id, thread_id, tender_text=None,
                            open_info=None, eval_criteria=None):
    from app.services.document_analysis_svc import run_analysis
    report = run_analysis(file_data, user_id, thread_id,
                          tender_text=tender_text, open_info=open_info,
                          eval_criteria=eval_criteria)
    return {
        'basic_info': report['basic_info'],
        'suspected_units': report['suspected_units'],
        'indicators': report['indicators'],
        'personnel_summary': report['personnel_summary'],
        '_files': report['_files'],
        '_pairs': report.get('_pairs', []),
    }


# ── 维度 2: 横向对比（横向）────────────────────────────────────────
def _run_cross_comparison(file_data, tender_text=None):
    from app.services.batch_orchestrator import (
        compute_all_pairs, build_key_info_matches, build_attr_details,
        cluster_order_by_risk, detect_gangs,
    )
    from app.services.batch_compare_svc import _precompute_tfidf_for_files
    check_items = {
        'text_sim': True,
        'key_info': True,
        'file_attr': True,
        'image_sim': False,
    }
    # Precompute TF-IDF once so text similarity actually runs (was silently 0).
    # Pass the tender doc as template_text so its boilerplate is stripped first.
    tfidf_matrix = None
    try:
        _vec, tfidf_matrix = _precompute_tfidf_for_files(file_data, template_text=tender_text)
    except Exception as e:
        logger.warning(f"TF-IDF precompute failed, text sim will be 0: {e}")
    pairs, risk_matrix = compute_all_pairs(
        file_data, check_items, tfidf_matrix=tfidf_matrix, template_text=tender_text)
    filenames = [fd['filename'] for fd in file_data]
    n = len(filenames)

    # E1: per-dimension matrices (text / key-info / attr)
    text_matrix = [[0.0] * n for _ in range(n)]
    key_matrix = [[0.0] * n for _ in range(n)]
    attr_matrix = [[0.0] * n for _ in range(n)]
    for p in pairs:
        i, j = p['i'], p['j']
        text_matrix[i][j] = text_matrix[j][i] = p.get('sim', 0.0)
        key_matrix[i][j] = key_matrix[j][i] = p.get('key_sim', 0.0)
        attr_matrix[i][j] = attr_matrix[j][i] = float(p.get('attr_same', 0) or 0)

    result = {
        'pairs': pairs,
        'risk_matrix': risk_matrix,
        'text_matrix': text_matrix,
        'key_matrix': key_matrix,
        'attr_matrix': attr_matrix,
        'key_info_matches': build_key_info_matches(pairs),
        'attr_details': build_attr_details(file_data),
        'files': filenames,
    }
    # E2: gang detection (cluster of mutually high-risk companies)
    try:
        gangs = detect_gangs(risk_matrix, filenames, threshold=10.0)
        if gangs:
            result['gangs'] = gangs
    except Exception as e:
        logger.warning(f"Gang detection failed: {e}")
    # E3: cluster ordering (visual reorder)
    try:
        result['cluster_order'] = cluster_order_by_risk(risk_matrix, filenames)
    except Exception as e:
        logger.warning(f"Cluster ordering failed: {e}")
    return result


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


# ── 维度 5: 全量审计补充检查（风格/规则提取/时间线占位）────────────
def _run_audit_supplement(file_data):
    """Merged from 全量审计's unique functions (方案 A):
      - style_analysis:     per-file text formality/consistency
      - rule_extraction:    extract rules from the bid doc itself (no tender)
      - timeline_compliance: placeholder-skip (upload flow has no project timeline)
    """
    from app.services import audit_engine
    from app.services.rule_extractor import RuleExtractor

    per_file = []
    for fd in file_data:
        text = (fd.get('text') or '')
        fname = fd.get('filename', '')
        entry = {'filename': fname, 'timeline': {'skipped': True,
                 'note': '上传场景无项目时间线数据，跳过'}}
        # style_analysis
        try:
            style_findings = audit_engine._run_style_analysis(text)
            style_score = audit_engine._score_style_analysis(style_findings)
            entry['style'] = {'score': round(style_score, 1), 'findings': style_findings}
        except Exception as e:
            logger.warning(f"Style analysis failed for {fname}: {e}")
            entry['style'] = {'score': 0.0, 'findings': {'error': str(e)}}
        # rule_extraction (self rules, no tender)
        try:
            extractor = RuleExtractor()
            rules_result = extractor.extract(text, fname, use_ai=True)
            rules = rules_result.get('rules', [])
            rules_score = audit_engine._score_rule_extraction({'rules': rules})
            entry['rules'] = {
                'score': round(rules_score, 1),
                'count': len(rules),
                'findings': {'rules': rules[:20]},
            }
        except Exception as e:
            logger.warning(f"Rule extraction failed for {fname}: {e}")
            entry['rules'] = {'score': 0.0, 'count': 0, 'findings': {'error': str(e)}}
        per_file.append(entry)
    return {'per_file': per_file}


# ── 主编排入口 ─────────────────────────────────────────────────────
def run_clearance(file_data, tender_text, tender_name, options, user_id=None, thread_id=None, info_overrides=None,
                  progress_cb=None, open_info=None, eval_criteria=None):
    """执行清标全维度分析，返回合并后的报告 dict。

    info_overrides: dict with keys like bid_number, bid_open_time, etc.
    open_info:      structured 开标信息表 (parsed) or None.
    eval_criteria:  structured 评审标准 (extracted from tender) or None.
    progress_cb: optional callable(percent:int, message:str) — heartbeat per
                 finished dimension so long stalls are visible in the sidebar.
    """
    n = len(file_data)
    results = {}

    dim_labels = {
        'indicators': '指标分析',
        'cross': '横向对比',
        'compliance': '合规审查',
        'ai': 'AI 评审',
        'audit': '全量审计补充检查',
    }

    futures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        if options.get('indicator_analysis', True):
            futures['indicators'] = pool.submit(
                _run_indicator_analysis, file_data, user_id, thread_id,
                tender_text, open_info, eval_criteria)
        if options.get('cross_comparison', True):
            futures['cross'] = pool.submit(_run_cross_comparison, file_data, tender_text)
        if options.get('compliance_check', False) and tender_text:
            futures['compliance'] = pool.submit(_run_compliance_check, tender_text, tender_name, file_data)
        if options.get('ai_review', True):
            futures['ai'] = pool.submit(_run_ai_review, file_data)
        # 维度5：全量审计补充检查（always on — style + rules, no tender needed）
        futures['audit'] = pool.submit(_run_audit_supplement, file_data)

        total_dims = len(futures)
        done_count = 0
        name_by_future = {fut: name for name, fut in futures.items()}
        for fut in as_completed(futures.values()):
            name = name_by_future[fut]
            try:
                results[name] = fut.result()
            except Exception as e:
                logger.error(f"Clearance dimension {name} failed: {e}", exc_info=True)
                results[name] = None
            done_count += 1
            if progress_cb:
                pct = 65 + int(20 * done_count / max(1, total_dims))  # 65→85
                progress_cb(pct, f"{dim_labels.get(name, name)}完成 ({done_count}/{total_dims})")

    # ── 合并报告 ──
    indicator = results.get('indicators') or {}
    basic_info = indicator.get('basic_info') or {
        'project_name': '用户自定义',
        'bidder_count': n,
        'analysis_date': _now_str(),
        'total_score': 0,
        'warning_level': '◇ 正常',
    }
    # 冒烟指数叠加合规/AI 风险提示（保持 0-100 量纲）
    total_score = basic_info.get('total_score', 0)

    merged = {
        'basic_info': {**basic_info, **(info_overrides or {})},
        'suspected_units': indicator.get('suspected_units', []),
        'indicators': indicator.get('indicators', []),
        'personnel_summary': indicator.get('personnel_summary', {'total': 0, 'bidders': 0, 'agents': 0, 'list': []}),
        '_files': indicator.get('_files', [fd.get('filename', '') for fd in file_data]),
        '_pairs': indicator.get('_pairs', []),
        'cross_comparison': results.get('cross'),
        'compliance': results.get('compliance'),
        'ai_review': results.get('ai'),
        'audit_supplement': results.get('audit'),
        'open_info': open_info,
        'eval_criteria': eval_criteria,
    }

    # 预警级别按最终总分（含合规严重违规加权）
    comp = merged['compliance']
    if comp and not comp.get('skipped') and comp.get('per_file'):
        criticals = sum(p['summary'].get('critical', 0) for p in comp['per_file'])
        violations = sum(p['summary'].get('violation', 0) for p in comp['per_file'])
        total_score += criticals * 5 + violations * 2
    merged['basic_info']['total_score'] = round(total_score, 1)
    merged['basic_info']['warning_level'] = (
        '● 高度预警' if total_score > 50 else ('◆ 中等预警' if total_score > 20 else '◇ 正常')
    )

    return merged


# ── Celery async task ───────────────────────────────────────────────
from celery_app import celery as _celery_app


@_celery_app.task(bind=True, name='clearance_task', max_retries=1,
                  soft_time_limit=2400, time_limit=2700)
def run_clearance_async(self, file_data, file_specs, tender_text, tender_name, tender_spec,
                        options, user_id, thread_id, task_id, project_id=None, info_overrides=None,
                        open_info=None, eval_criteria=None):
    """Celery task: full clearance analysis → DOCX + PDF → ZIP → DB.

    file_data:  pre-extracted dicts (legacy small uploads)
    file_specs: [{'abs_path','filename'}] — extracted here in the worker,
                page-by-page, so Flask never touches the big files.
    open_info:      structured 开标信息表 dict (from clearance_openinfo).
    eval_criteria:  structured 评审标准 dict (from clearance_openinfo).
    """
    from app.services.task_bus import TaskBus
    from app.services.document_analysis_svc import build_clearance_docx, convert_docx_to_pdf
    from app.services.file_processing import (
        extract_text_from_path, extract_metadata_from_path, take_image_sampling_log,
    )
    import os as _os, zipfile, json

    # Extraction helpers touch flask.session (analyze_images pref) which
    # needs a *request* context — test_request_context provides one for the
    # whole task.
    from celery_app import init_flask_context
    _flask_app = init_flask_context()
    _req_ctx = _flask_app.test_request_context()
    _req_ctx.push()

    bus = TaskBus(task_id, 'clearance', '清标分析')
    bus.start(extra={'thread_id': thread_id or ''})

    try:
        n = len(file_data) + len(file_specs)

        # ── Worker-side extraction (bounded memory: one file at a time) ──
        all_file_data = list(file_data or [])
        for i, spec in enumerate(file_specs or []):
            fname = spec.get('filename', '')
            bus.progress(5 + int(50 * i / max(1, len(file_specs))),
                         f'正在提取文件内容 ({i + 1}/{len(file_specs)}): {fname}...')
            text, _ = extract_text_from_path(spec['abs_path'], fname)
            if not text or text.startswith("["):
                logger.warning(f"Skipping unreadable file: {fname}")
                continue
            meta = extract_metadata_from_path(spec['abs_path'], fname)
            all_file_data.append({
                'filename': fname,
                'text': text,
                'metadata': meta or {},
                'images': [],
            })

        if tender_spec and not tender_text:
            bus.progress(60, f"正在提取招标文件: {tender_spec.get('filename', '')}...")
            ttext, _ = extract_text_from_path(tender_spec['abs_path'], tender_spec.get('filename'))
            if ttext and not ttext.startswith("["):
                tender_text = ttext
                tender_name = tender_spec.get('filename')
            else:
                tender_text = None

        if len(all_file_data) < 2:
            bus.fail('可提取文本的投标文件不足 2 份，任务终止')
            return

        bus.progress(65, '正在执行指标分析与横向对比...')
        report = run_clearance(all_file_data, tender_text, tender_name, options,
                               user_id=user_id, thread_id=thread_id, info_overrides=info_overrides,
                               progress_cb=lambda pct, msg: bus.progress(pct, msg),
                               open_info=open_info, eval_criteria=eval_criteria)

        # ── 图片随机抽检说明（九章）──
        sampling = take_image_sampling_log()
        if sampling:
            report['image_sampling'] = sampling

        bus.progress(70, '正在生成清标报告...')

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

        file_names = [fd['filename'] for fd in all_file_data]
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO batch_comparison_results (user_id, task_id, project_id, file_count, pair_count, max_risk, file_names, zip_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id, task_id, project_id, len(all_file_data), 0,
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

                # ── Persist clearance result as an assistant chat message ──
                # Store the full report JSON (frontend renders it into a chat
                # bubble via the CLEARANCE_REPORT marker on both live-complete
                # and session reload). Reuses this same transaction/connection.
                if thread_id:
                    try:
                        from flask import url_for as _url_for
                        _dl = _url_for('batch.download_batch_result', task_id=task_id)
                    except Exception:
                        _dl = f'/batch_result/{task_id}'
                    chat_payload = json.dumps({
                        'report': report,
                        'download_url': _dl,
                        'file_count': len(all_file_data),
                    }, ensure_ascii=False)
                    chat_content = '<!-- CLEARANCE_REPORT -->' + chat_payload
                    try:
                        cur.execute(
                            "INSERT INTO chat_messages (thread_id, role, content, thinking, timestamp) "
                            "VALUES (%s, 'assistant', %s, NULL, NOW())",
                            (thread_id, chat_content))
                    except Exception as _ce:
                        logger.warning(f"Failed to persist clearance chat message: {_ce}")
                conn.commit()

        from flask import url_for
        try:
            download_url = url_for('batch.download_batch_result', task_id=task_id)
        except Exception:
            download_url = f'/batch_result/{task_id}'

        bus.progress(100, '清标完成')
        bus.complete({
            'report': report,
            'download_url': download_url,
            'file_count': len(all_file_data),
        })

    except Exception as e:
        logger.error(f"Clearance task failed: {e}", exc_info=True)
        import traceback as _tb
        _frames = _tb.format_exc().strip().split('\n')
        bus.fail(f"{e} | at: {_frames[-1].strip()}"[:500])
