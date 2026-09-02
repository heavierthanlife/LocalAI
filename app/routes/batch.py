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


@batch_bp.route('/batch_result/<task_id>', methods=['GET'])
def download_batch_result(task_id):
    """Download a permanently-stored batch result ZIP (HTML + Excel)."""
    if session.get('consent_value', 0) != 1:
        return err("请先登录", "AUTH_REQUIRED", 401)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT zip_path FROM batch_comparison_results WHERE task_id = %s", (task_id,))
            row = cur.fetchone()
            if not row:
                return err("任务记录不存在", "NOT_FOUND", 404)
            zip_abs = resolve_path(row[0])
            if not os.path.exists(zip_abs):
                # Stale record: the ZIP was written by a worker in a different
                # filesystem (e.g. Docker volume vs local dev host). Clean up
                # so the sidebar stops offering a dead download.
                cur.execute("DELETE FROM batch_comparison_results WHERE task_id = %s", (task_id,))
                conn.commit()
                return err("报告文件已过期（由其他环境生成），记录已清理", "FILE_GONE", 410)
    return send_file(zip_abs, as_attachment=True, download_name=f"batch_report_{task_id}.zip",
                    mimetype='application/zip')


@batch_bp.route('/list_batch_results', methods=['GET'])
def list_batch_results():
    """Return ALL batch comparison results visible to all registered users.

    Rows whose ZIP file is missing (written by a worker in another filesystem,
    e.g. Docker volume vs local dev host) are pruned so the sidebar never
    offers a dead download.
    """
    if session.get('consent_value', 0) != 1:
        return err("请先登录", "AUTH_REQUIRED", 401)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT r.id, r.task_id, r.zip_path, r.file_count, r.pair_count, r.max_risk, r.file_names, r.created_at,
                       u.username as created_by_name
                FROM batch_comparison_results r
                JOIN users u ON r.user_id = u.user_id
                ORDER BY r.created_at DESC LIMIT 30
            """)
            results = cur.fetchall()
    live = []
    stale_ids = []
    for r in results:
        if os.path.exists(resolve_path(r.pop('zip_path', '') or '')):
            live.append(dict(r))
        else:
            stale_ids.append(r['id'])
    if stale_ids:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM batch_comparison_results WHERE id = ANY(%s)", (stale_ids,))
                conn.commit()
    return ok({"results": live})


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


@batch_bp.route('/plagiarism/compare', methods=['POST'])
def plagiarism_compare():
    """Plagiarism mode: two-file comparison (FIX-016 后续).

    Reuses compute_similarity_with_numbers + paragraph aggregation. Distinct from
    清标 mode — outputs pairwise verdict + paragraph heatmap, NOT composite score.
    """
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    if not session.get('user_id'):
        return err("Not logged in", "AUTH_REQUIRED", 401)

    files = request.files.getlist('files') or []
    if len(files) < 2:
        return err("需要上传 2 个文件", "VALIDATION_ERROR", 400)

    texts = []
    names = []
    for f in files[:2]:
        if not f.filename or not allowed_file(f.filename):
            return err(f"不支持的格式: {f.filename}", "VALIDATION_ERROR", 400)
        text, _ = extract_text_from_file(f)
        if not text or not is_valid_extracted_text(text):
            return err(f"无法提取文本: {f.filename}", "VALIDATION_ERROR", 400)
        texts.append(text)
        names.append(f.filename)

    # optional template (招标文件) for boilerplate removal
    template_text = None
    tf = request.files.get('template')
    if tf and tf.filename and allowed_file(tf.filename):
        template_text, _ = extract_text_from_file(tf)

    from app.services.plagiarism_detector import detect_plagiarism
    report = detect_plagiarism(
        texts[0], texts[1],
        template_text=template_text,
        filename_a=names[0], filename_b=names[1],
    )
    return ok(report)
