"""Blueprint: batch routes (auto-extracted)."""
import os, json, uuid, time, logging, hashlib, io, html
from datetime import datetime, timezone
from io import BytesIO
from flask import Blueprint, request, jsonify, session, send_file, render_template, url_for

from app.config import BASE_DIR, DATA_DIR, TEMP_ROOT, TEMP_DIR, USER_FILES_ORIGINAL_ROOT, to_rel_path, resolve_path, is_valid_extracted_text, allowed_file
from app.database import get_db_connection, db_transaction
from app.utils.helpers import ok, err, utc_now, beijing_now, safe_error_response, split_thinking_answer
from psycopg2.extras import RealDictCursor
import app.globals as g
from app.services.file_cache import file_cache_manager, add_to_cache, load_cache_from_db

logger = logging.getLogger(__name__)
from app.services.session_manager import get_user_id, get_or_create_session, ensure_user_exists, store_message, record_file_usage
from app.services.task_locking import acquire_task_lock, release_task_lock

import secrets, os as _os
from openpyxl import Workbook
from sklearn.metrics.pairwise import cosine_similarity
from app.services.file_processing import (
    extract_text_from_file, compute_similarity_with_numbers,
    compute_batch_semantic_similarity, extract_metadata, file_attr_similarity,
    extract_images_from_file, image_similarity, preprocess_text_for_similarity,
    remove_template_content, keyword_overlap_similarity, extract_keywords,
    truncate_filename, get_or_extract_file_analysis,
)
from app.services.batch_compare_svc import (
    _precompute_tfidf_for_files, _compute_pair_similarity_from_matrix,
    store_batch_comparison_temp, load_batch_comparison_temp
)

from app.services.batch_orchestrator import (
    RiskScorer, compute_all_pairs, build_key_info_matches, build_attr_details,
    build_excel_workbook, build_summary_html, build_pair_report_html,
    build_full_report_html, run_all_sub_checkers, build_report_docx,
)

batch_bp = Blueprint('batch', __name__, template_folder=str(BASE_DIR / 'templates'), static_folder=str(BASE_DIR / 'static'))

@batch_bp.route('/compare_batch', methods=['POST'])
def compare_batch():
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    if 'files' not in request.files:
        return err("No files uploaded", "VALIDATION_ERROR", 400)
    files = request.files.getlist('files')
    if len(files) < 2:
        return err("Need at least 2 files for comparison", "VALIDATION_ERROR", 400)
    if len(files) > 10:
        return err("Maximum 10 files allowed", "VALIDATION_ERROR", 400)
    for f in files:
        if not allowed_file(f.filename):
            return err(f"不支持的文件类型: {f.filename}", "VALIDATION_ERROR", 400)

    user_id = get_user_id()
    thread_id = session['thread_id']
    success, busy_thread, busy_name = acquire_task_lock(user_id, thread_id, 'batch_compare')
    get_or_create_session(thread_id)
    if not success:
        return ok({
            "error": "resource_busy",
            "busy_chat": busy_name,
            "message": f"另一个资源密集型任务正在聊天“{busy_name}”中进行，请稍后再试。"
        }), 409

    try:
        template_file = request.files.get('template_file')
        template_text = None
        if template_file and template_file.filename:
            if not allowed_file(template_file.filename):
                return err(f"不支持的文件类型: {template_file.filename}", "VALIDATION_ERROR", 400)
            if session.get('consent_value', 0) == 1:
                template_text = get_or_extract_file_analysis(template_file, 'chat', user_id, thread_id=thread_id)
            else:
                template_text, _ = extract_text_from_file(template_file)
            if template_text and not template_text.startswith("["):
                if not is_valid_extracted_text(template_text):
                    template_text = None   # invalid template, ignore
                else:
                    record_file_usage(thread_id, template_file.filename, 'template_upload', "上传模板文件用于对比")

        check_items_json = request.form.get('check_items', '{}')
        try:
            check_items = json.loads(check_items_json)
        except (json.JSONDecodeError, TypeError):
            check_items = {}

        defaults = {
            'text_sim': True,
            'key_info': True,
            'file_attr': True,
            'image_sim': True,
            'semantic': False
        }
        for k, v in defaults.items():
            if k not in check_items:
                check_items[k] = v

        project_id = request.form.get('project_id', type=int)

        # ── Audit: detailed component participation log ──
        from app.services.audit_logger import AuditLogger
        _audit = AuditLogger("batch_compare", thread_id[:12] if thread_id else "")
        _audit.component("init", status="OK",
                         file_count=len(files), check_items=str(check_items),
                         has_template=bool(template_text))

        if len(files) > 10 and check_items.get('semantic'):
            check_items['semantic'] = False
            logger.info("Semantic analysis disabled because number of files exceeds 10.")
            _audit.component("semantic", status="SKIPPED", reason="files > 10")

        file_data = []
        for f in files:
            if not f.filename:
                continue
            if session.get('consent_value', 0) == 1:
                f.seek(0)
                file_bytes = f.read()
                file_hash = hashlib.sha256(file_bytes).hexdigest()
                file_size = len(file_bytes)
                f.seek(0)
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT id FROM user_files WHERE user_id = %s AND file_hash = %s", (user_id, file_hash))
                        existing = cur.fetchone()
                        if not existing:
                            ext = os.path.splitext(f.filename)[1]
                            unique_name = f"{file_hash}_{int(time.time())}{ext}"
                            original_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)
                            os.makedirs(original_dir, exist_ok=True)
                            original_path = os.path.join(original_dir, unique_name)
                            original_rel = to_rel_path(original_path)
                            f.seek(0)
                            f.save(original_path)
                            f.seek(0)  # save() consumes stream, reset for later reads
                            ensure_user_exists(user_id)
                            cur.execute("""
                                INSERT INTO user_files (user_id, thread_id, filename, size_bytes, expires_at, original_stored_path, file_hash, original_expires_at, original_name)
                                VALUES (%s, %s, %s, %s, NULL, %s, %s, NOW() + INTERVAL '3 days', %s)
                                ON CONFLICT (thread_id, filename) DO UPDATE SET
                                    size_bytes = EXCLUDED.size_bytes,
                                    original_stored_path = EXCLUDED.original_stored_path,
                                    file_hash = EXCLUDED.file_hash,
                                    original_expires_at = EXCLUDED.original_expires_at,
                                    original_name = EXCLUDED.original_name
                            """, (user_id, thread_id, f.filename, file_size, original_rel, file_hash, f.filename))
                            conn.commit()
                text = get_or_extract_file_analysis(f, 'chat', user_id, thread_id=thread_id)
            else:
                text, _ = extract_text_from_file(f)

            if text and not text.startswith("[") and is_valid_extracted_text(text):
                record_file_usage(thread_id, f.filename, 'compare_batch', "批量对比")
                f.seek(0)
                meta = extract_metadata(f)
                images = extract_images_from_file(f)
                file_data.append({
                    'filename': f.filename,
                    'text': text,
                    'metadata': meta,
                    'images': images
                })
            else:
                logger.warning(f"Skipping file {f.filename}: extraction failed or invalid (text='{text}')")
                continue

        if len(file_data) < 2:
            return err("Could not extract valid text from at least two files", "VALIDATION_ERROR", 400)

        n = len(file_data)
        if check_items.get('text_sim', True) or check_items.get('key_info', True):
            vectorizer, tfidf_matrix = _precompute_tfidf_for_files(file_data, template_text)
        else:
            vectorizer = tfidf_matrix = None

        semantic_sim_matrix = None
        if check_items.get('semantic', False):
            all_texts = [fd['text'] for fd in file_data]
            semantic_sim_matrix = compute_batch_semantic_similarity(all_texts)
            logger.info("Semantic similarity matrix computed.")
            _audit.component("semantic", status="OK",
                             model=getattr(semantic_sim_matrix, '__class__', 'computed') if semantic_sim_matrix is not None else 'failed',
                             file_count=n)
        if semantic_sim_matrix is None and check_items.get('semantic', False):
            check_items['semantic'] = False
            logger.warning("Semantic analysis disabled due to model load failure.")
            _audit.component("semantic", status="FAILED", reason="model_load_failure")

        pairs, risk_matrix = compute_all_pairs(file_data, check_items, tfidf_matrix, template_text)
        _pair_audit_summary = {"total": len(pairs)}

        key_info_matches = build_key_info_matches(pairs)
        attr_details = build_attr_details(file_data)

        batch_data = {
            'file_data': [{'filename': fd['filename'], 'metadata': fd['metadata']} for fd in file_data],
            'pairs': pairs,
            'check_items': check_items,
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'key_info_matches': key_info_matches,
            'attr_details': attr_details,
            'semantic_sim_matrix': semantic_sim_matrix,
        }

        temp_path = store_batch_comparison_temp(batch_data)
        session['batch_comparison_path'] = temp_path

        short_names = [truncate_filename(fd['filename'], 20) for fd in file_data]

        summary_html = build_summary_html(file_data, pairs, risk_matrix)
        main_report = build_pair_report_html(file_data, pairs, risk_matrix)

        # ── Sub-checkers (non-blocking, each handles its own audit logging) ──
        sub_results = run_all_sub_checkers(file_data, user_id, thread_id, project_id=project_id, audit=_audit)
        typo_results = sub_results.get('typo')
        rel_report = sub_results.get('relationship')
        quote_result = sub_results.get('quote')

        # ── AI professional analysis of the highest-risk pair ──
        _ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        ai_analysis_html = ''
        top_pair = max(pairs, key=lambda p: p['risk']) if pairs else None
        if top_pair and top_pair['risk'] > 5:
            try:
                from app.services.analysis_prompts import build_bid_analysis_prompt, BID_COMPARISON_SYSTEM
                from app.services.llm_provider import create_chat_model
                from langchain_core.messages import SystemMessage, HumanMessage
                high_pairs = sorted(
                    [(p['name1'], p['name2'], p['risk'], p['sim']) for p in pairs if p['risk'] > 10],
                    key=lambda x: x[2], reverse=True
                )
                short_names = [truncate_filename(fd['filename'], 20) for fd in file_data]
                prompt = build_bid_analysis_prompt(
                    risk_matrix=risk_matrix,
                    file_names=short_names,
                    high_risk_pairs=high_pairs[:5],
                    top_pair_text1=top_pair['text1'],
                    top_pair_text2=top_pair['text2'],
                    top_pair_name1=top_pair['name1'],
                    top_pair_name2=top_pair['name2'],
                    top_pair_risk=top_pair['risk'],
                )
                llm = create_chat_model(streaming=False, temperature=0.3, max_tokens=1500,
                                        timeout=int(os.getenv("LLM_TIMEOUT", "90")))
                from app.services.prompt_safety import sanitize_for_prompt
                ai_resp = llm.invoke([SystemMessage(content=BID_COMPARISON_SYSTEM),
                                     HumanMessage(content=sanitize_for_prompt(prompt, 'bid_analysis'))])
                ai_text = ai_resp.content if hasattr(ai_resp, 'content') else str(ai_resp)
                if ai_text and len(ai_text) > 15:
                    ai_analysis_html = f'<div class="card" style="background:#eff6ff;border-color:#bfdbfe;margin-top:16px;"><h2 style="color:#1e40af;">🤖 AI专业分析</h2>{ai_text}</div>'
            except Exception as e:
                logger.warning(f"AI bid analysis failed: {e}")
                _audit.component("ai_analysis", status="FAILED", error=str(e)[:100])
            else:
                _audit.component("ai_analysis", status="OK",
                                 top_risk=top_pair['risk'], model="deepseek-v4-pro",
                                 response_chars=len(ai_text) if ai_text else 0)

        # ── Build full HTML report (orchestrator) ──
        ai_html = build_full_report_html(file_data, pairs, risk_matrix,
                                         typo_results=typo_results,
                                         rel_report=rel_report,
                                         quote_result=quote_result,
                                         ai_analysis_html=ai_analysis_html)

        # ── Final audit: summarize all components ──
        max_risk_val = max(p['risk'] for p in pairs) if pairs else 0
        _audit.result(
            file_count=n, pair_count=len(pairs), max_risk=round(max_risk_val, 2),
            components=str(_pair_audit_summary),
            template_used=bool(template_text),
            ai_analysis=bool(ai_analysis_html),
        )

        # ── ZIP: HTML + DOCX ──
        import zipfile
        batch_task_id = str(uuid.uuid4())
        batch_dir = os.path.join(DATA_DIR, 'batch_results')
        os.makedirs(batch_dir, exist_ok=True)
        zip_name = f"batch_{batch_task_id}.zip"
        zip_path = os.path.join(batch_dir, zip_name)
        zip_rel = to_rel_path(zip_path)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("AI分析报告.html", ai_html.encode('utf-8'))
            docx_bytes = build_report_docx(file_data, pairs, key_info_matches, attr_details)
            zf.writestr("对比分析报告.docx", docx_bytes)
        # Insert into DB for permanent access
        file_names_list = [fd['filename'] for fd in file_data]
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO batch_comparison_results (user_id, task_id, project_id, file_count, pair_count, max_risk, file_names, zip_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (user_id, batch_task_id, project_id, len(file_data), len(pairs), float(round(max_risk, 2)),
                      json.dumps(file_names_list, ensure_ascii=False), zip_rel))
                conn.commit()

        batch_download_url = url_for('batch.download_batch_result', task_id=batch_task_id, _external=True)
        export_html = f'<p style="margin-top:12px;"><a href="{batch_download_url}" target="_blank" style="background:#27ae60; color:white; text-decoration:none; border-radius:8px; padding:8px 16px; display:inline-block;">📦 下载完整报告 (HTML+DOCX，永久有效)</a></p>'
        full_message = f"<!--COMPARE_REPORT--><div style='font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, sans-serif; line-height:1.5; max-width:100%; overflow-x:auto;'><h4>📁 批量对比结果（{len(file_data)}个文件）</h4>{summary_html}{main_report}{export_html}</div>"

        if session.get('consent_value', 0) == 1:
            ensure_user_exists(user_id)
            report_filename = f"批量对比_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.html"
            report_bytes = full_message.encode('utf-8')
            report_hash = hashlib.sha256(report_bytes).hexdigest()
            ext = '.html'
            unique_name = f"{report_hash}_{int(time.time())}{ext}"
            original_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)
            os.makedirs(original_dir, exist_ok=True)
            report_path = os.path.join(original_dir, unique_name)
            report_rel = to_rel_path(report_path)
            with open(report_path, 'wb') as f:
                f.write(report_bytes)
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO user_files (user_id, thread_id, filename, size_bytes, expires_at, original_stored_path, file_hash, original_expires_at, original_name)
                        VALUES (%s, %s, %s, %s, NULL, %s, %s, NOW() + INTERVAL '3 days', %s)
                        ON CONFLICT (thread_id, filename) DO NOTHING
                    """, (user_id, thread_id, report_filename, len(report_bytes), report_rel, report_hash, report_filename))
                    conn.commit()
            record_file_usage(thread_id, report_filename, 'compare_batch_report', "批量对比生成的报告")

        store_message(thread_id, 'assistant', full_message, thinking="")
        session.setdefault('chat_history', []).append({
            "role": "assistant",
            "content": full_message,
            "thinking": ""
        })
        return ok({
            "success": True,
            "pair_count": len(pairs),
            "download_url": batch_download_url,
            "file_count": len(file_data),
        })
    finally:
        release_task_lock(user_id)

@batch_bp.route('/export_batch_docx_download/<token>', methods=['GET'])
def export_batch_docx_download(token):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    remaining = g.download_tokens.get(token, 0)
    if remaining <= 0:
        return err("Download link has expired or already used the maximum number of times.", "SERVER_ERROR", 410)
    temp_path = session.get(f'download_path_{token}')
    if not temp_path or not os.path.exists(temp_path):
        return err("Comparison data not found.", "NOT_FOUND", 404)
    try:
        batch_data = load_batch_comparison_temp(temp_path)
    except Exception as e:
        logger.error(f"Failed to load batch data: {e}")
        return err("Comparison data corrupted.", "VALIDATION_ERROR", 400)

    file_data = batch_data['file_data']
    pairs = batch_data['pairs']
    key_info_matches = batch_data.get('key_info_matches', [])
    attr_details = batch_data.get('attr_details', [])

    docx_bytes = build_report_docx(file_data, pairs, key_info_matches, attr_details)
    output = BytesIO(docx_bytes)
    output.seek(0)
    filename = f"对比分析报告_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.docx"
    with g.download_tokens_lock:
        g.download_tokens[token] -= 1
        if g.download_tokens[token] <= 0:
            del g.download_tokens[token]
            session.pop(f'download_path_{token}', None)
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    return send_file(output, as_attachment=True, download_name=filename,
                     mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')


@batch_bp.route('/batch_result/<task_id>', methods=['GET'])
def download_batch_result(task_id):
    """Download a permanently-stored batch result ZIP (HTML + Excel)."""
    if session.get('consent_value', 0) != 1:
        return err("请先登录", "AUTH_REQUIRED", 401)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT zip_path FROM batch_comparison_results WHERE task_id = %s", (task_id,))
            row = cur.fetchone()
            if not row or not os.path.exists(resolve_path(row[0])):
                return err("Report not found", "NOT_FOUND", 404)
    return send_file(resolve_path(row[0]), as_attachment=True, download_name=f"batch_report_{task_id}.zip",
                    mimetype='application/zip')


@batch_bp.route('/list_batch_results', methods=['GET'])
def list_batch_results():
    """Return ALL batch comparison results visible to all registered users."""
    if session.get('consent_value', 0) != 1:
        return err("请先登录", "AUTH_REQUIRED", 401)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT r.id, r.task_id, r.file_count, r.pair_count, r.max_risk, r.file_names, r.created_at,
                       u.username as created_by_name
                FROM batch_comparison_results r
                JOIN users u ON r.user_id = u.user_id
                ORDER BY r.created_at DESC LIMIT 30
            """)
            results = cur.fetchall()
    return ok({"results": [dict(r) for r in results]})


@batch_bp.route('/delete_batch_result/<int:id>', methods=['POST'])
def delete_batch_result(id):
    """Admin only: delete a batch result."""
    if session.get('consent_value', 0) != 1:
        return err("请先登录", "AUTH_REQUIRED", 401)
    if session.get('role') != 'admin':
        return err("仅管理员可删除", "FORBIDDEN", 403)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT zip_path FROM batch_comparison_results WHERE id = %s", (id,))
            row = cur.fetchone()
            if not row:
                return err("Not found", "NOT_FOUND", 404)
            if os.path.exists(resolve_path(row[0])):
                os.remove(resolve_path(row[0]))
            cur.execute("DELETE FROM batch_comparison_results WHERE id = %s", (id,))
            conn.commit()
    return ok(message="ok")


# ── Standalone quote anomaly endpoints ──

@batch_bp.route('/check_quote_anomaly', methods=['POST'])
def check_quote_anomaly_standalone():
    """Standalone endpoint: detect quote anomalies in a single bid document."""
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)

    if 'file' not in request.files:
        return err("No file uploaded", "VALIDATION_ERROR", 400)
    f = request.files['file']
    if not f.filename or not allowed_file(f.filename):
        return err(f"Unsupported file type: {f.filename}", "VALIDATION_ERROR", 400)

    from app.services.quote_anomaly import check_quote_anomaly as run_qa
    from app.services.audit_logger import AuditLogger

    text, _ = extract_text_from_file(f)
    if not text or text.startswith("["):
        return err("Could not extract text from file", "VALIDATION_ERROR", 400)

    _audit = AuditLogger("quote_anomaly_standalone", f.filename)
    result = run_qa(text, doc_name=f.filename, audit=_audit)
    _audit.result(risk_score=round(result.risk_score, 1), doc_name=f.filename)

    return ok({
        "doc_name": result.doc_name,
        "prices": result.prices[:20],
        "percentages": result.percentages[:20],
        "cv": result.cv,
        "same_rate_flag": result.same_rate_flag,
        "abnormal_drop_flag": result.abnormal_drop_flag,
        "clustering_flag": result.clustering_flag,
        "benford_deviation": result.benford_deviation,
        "risk_score": result.risk_score,
        "details": result.details,
        "daxie_mismatches": result.daxie_mismatches,
    })


@batch_bp.route('/compare_bidders_quotes', methods=['POST'])
def compare_bidders_quotes_endpoint():
    """Standalone endpoint: cross-bidder quote comparison without full batch compare."""
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)

    if 'files' not in request.files:
        return err("No files uploaded", "VALIDATION_ERROR", 400)
    files = request.files.getlist('files')
    if len(files) < 2:
        return err("Need at least 2 files", "VALIDATION_ERROR", 400)
    if len(files) > 10:
        return err("Maximum 10 files allowed", "VALIDATION_ERROR", 400)

    from app.services.quote_anomaly import compare_bidders_quotes as run_cbq, save_quote_anomaly_results
    from app.services.audit_logger import AuditLogger

    file_data = []
    for f in files:
        if not f.filename or not allowed_file(f.filename):
            continue
        text, _ = extract_text_from_file(f)
        if text and not text.startswith("["):
            file_data.append({'filename': f.filename, 'text': text})

    if len(file_data) < 2:
        return err("Could not extract valid text from at least 2 files", "VALIDATION_ERROR", 400)

    thread_id = session.get('thread_id', '')
    project_id = request.form.get('project_id', type=int)
    _audit = AuditLogger("compare_bidders_quotes", thread_id[:12] if thread_id else "")
    result = run_cbq(file_data, audit=_audit)
    save_quote_anomaly_results(user_id, thread_id or str(uuid.uuid4()),
                               result['per_bidder'], result, project_id=project_id)
    _audit.result(max_risk=round(result.get('max_risk_score', 0), 1),
                  bidders=len(file_data))

    return ok(result)


# ── Standalone relationship extraction endpoints ──

@batch_bp.route('/check_quote_anomaly/feedback', methods=['POST'])
def quote_anomaly_feedback():
    """Submit feedback on a quote anomaly analysis result."""
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    data = request.get_json() or {}
    doc_name = data.get('doc_name', '')
    rating = data.get('rating')
    if not doc_name or rating is None:
        return err("Missing doc_name or rating", "VALIDATION_ERROR", 400)
    if rating not in (1, 5):
        return err("rating must be 1 or 5", "VALIDATION_ERROR", 400)
    try:
        from app.services.training_logger import log_interaction
        log_interaction(
            thread_id=f"quote_analysis_{doc_name[:40]}_{session.get('thread_id', 'anon')[:8]}",
            user_msg=f"报价分析文档: {doc_name}",
            assistant_response=f"用户评分: {rating}/5",
            rating=rating,
            source='quote_analysis',
        )
        return ok(message="反馈已保存")
    except Exception as e:
        from app.config import logger
        logger.error(f"Quote anomaly feedback error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@batch_bp.route('/extract_relationships', methods=['POST'])
def extract_relationships_endpoint():
    """Standalone endpoint: extract entity relationships from bid documents."""
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)

    if 'files' not in request.files:
        return err("No files uploaded", "VALIDATION_ERROR", 400)
    files = request.files.getlist('files')
    if len(files) < 1:
        return err("Need at least 1 file", "VALIDATION_ERROR", 400)
    if len(files) > 10:
        return err("Maximum 10 files allowed", "VALIDATION_ERROR", 400)

    from app.services.relationship_extractor import extract_relationships, save_relationship_results
    from app.services.audit_logger import AuditLogger

    file_data = []
    for f in files:
        if not f.filename or not allowed_file(f.filename):
            continue
        text, _ = extract_text_from_file(f)
        if text and not text.startswith("["):
            f.seek(0)
            meta = extract_metadata(f)
            file_data.append({'filename': f.filename, 'text': text, 'metadata': meta})

    if not file_data:
        return err("Could not extract valid text from any file", "VALIDATION_ERROR", 400)

    thread_id = session.get('thread_id', '')
    task_id = thread_id or str(uuid.uuid4())
    project_id = request.form.get('project_id', type=int)
    _audit = AuditLogger("relationship_extraction", task_id[:12] if task_id else "")

    report = extract_relationships(file_data, audit=_audit)
    save_relationship_results(user_id, task_id, report, project_id=project_id)
    _audit.result(risk_score=report.risk_score, entities=len(report.entities),
                  relations=len(report.relationships), flags=len(report.red_flags))

    return ok({
        "entities": [{'text': e.text, 'type': e.entity_type, 'confidence': e.confidence}
                     for e in report.entities[:50]],
        "relationships": [{
            'source': r.source_entity, 'target': r.target_entity,
            'type': r.relation_type, 'subtype': r.relation_subtype,
            'confidence': r.confidence, 'evidence': r.evidence[:200],
            'module': r.module, 'risk_flag': r.risk_flag,
            'risk_reason': r.risk_reason,
        } for r in report.relationships],
        "red_flags": report.red_flags,
        "risk_score": report.risk_score,
        "modules_run": report.modules_run,
        "company_personnel_map": report.company_personnel_map,
    })


# ── Standalone typo detection endpoint ──

@batch_bp.route('/check_typos', methods=['POST'])
def check_typos_endpoint():
    """Standalone endpoint: detect typos in a single document."""
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)

    diff_mode = request.form.get('diff_mode', 'false').lower() == 'true'

    if 'file' not in request.files:
        return err("No file uploaded", "VALIDATION_ERROR", 400)
    f = request.files['file']
    if not f.filename or not allowed_file(f.filename):
        return err(f"Unsupported file type: {f.filename}", "VALIDATION_ERROR", 400)

    from app.services.typo_detector import detect_typos, save_typo_results
    from app.services.audit_logger import AuditLogger

    text, _ = extract_text_from_file(f)
    if not text or text.startswith("["):
        return err("Could not extract text from file", "VALIDATION_ERROR", 400)

    _audit = AuditLogger("typo_detection", f.filename)
    report = detect_typos(text, doc_name=f.filename, audit=_audit)

    thread_id = session.get('thread_id', '')
    save_typo_results(user_id, thread_id or str(uuid.uuid4()), {f.filename: report})
    _audit.result(total=report.total_suspects, critical=report.critical_count,
                  layers=','.join(report.layers_run))

    result = {
        "doc_name": f.filename,
        "total_suspects": report.total_suspects,
        "critical_count": report.critical_count,
        "layers_run": report.layers_run,
        "findings": [{
            'layer': f.layer,
            'suspect_text': f.suspect_text,
            'suggestions': f.suggestions,
            'confidence': f.confidence,
            'context_snippet': f.context_snippet,
            'severity': f.severity,
            'is_daxie_error': f.is_daxie_error,
            'daxie_expected': f.daxie_expected,
            'daxie_actual': f.daxie_actual,
        } for f in report.findings],
    }

    if diff_mode and report.diff_text:
        result['diff_text'] = report.diff_text[:5000]

    return ok(result)
