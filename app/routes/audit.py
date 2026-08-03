"""Unified Bid Audit API routes."""
import json
import logging
import os
import queue
import threading
import uuid
from datetime import datetime, timezone

from flask import Blueprint, request, session, Response, stream_with_context, send_file
from app.database import get_db_connection
from app.utils.helpers import ok, err
from app.config import resolve_path

logger = logging.getLogger(__name__)

audit_bp = Blueprint('audit', __name__, url_prefix='/audit')


def _admin_required(f):
    """Same logic as admin.is_admin() — checks session['role']."""
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        role = session.get('role', 'user')
        if role != 'admin':
            return err("需要管理员权限", "FORBIDDEN", 403)
        return f(*args, **kwargs)
    return wrapper


# ── Preflight ──

@audit_bp.route('/preflight', methods=['POST'])
@_admin_required
def preflight():
    """Check text extraction status for files in selected folders."""
    data = request.get_json(silent=True) or {}
    folder_ids = data.get('folder_ids', [])
    if not folder_ids or not isinstance(folder_ids, list):
        return err("请提供 folder_ids 数组", "VALIDATION_ERROR", 400)

    from app.services.audit_engine import run_preflight
    result = run_preflight(folder_ids)
    return ok(result)


# ── Start Audit ──

@audit_bp.route('/start', methods=['POST'])
@_admin_required
def start_audit():
    """Launch a full audit in the background."""
    data = request.get_json(silent=True) or {}
    folder_ids = data.get('folder_ids', [])
    file_ids = data.get('file_ids')  # optional: if provided, only audit these files
    enabled_functions = data.get('enabled_functions')
    extract_on_demand = data.get('extract_on_demand', False)
    project_id = data.get('project_id')
    project_level_functions = data.get('project_level_functions')

    if not folder_ids or not isinstance(folder_ids, list):
        return err("请提供 folder_ids 数组", "VALIDATION_ERROR", 400)
    if not project_id:
        return err("请提供 project_id", "VALIDATION_ERROR", 400)

    user_id = session.get('user_id')

    # Validate functions
    all_funcs = ['rule_extraction', 'compliance_check', 'typo_detection', 'quote_anomaly',
                 'relationship_extraction', 'ai_doc_review', 'style_analysis']
    if not enabled_functions:
        # Load defaults from config
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT function_name FROM audit_config WHERE enabled_by_default = TRUE")
                enabled_functions = [row[0] for row in cur.fetchall()]
    else:
        enabled_functions = [f for f in enabled_functions if f in all_funcs]

    if not enabled_functions:
        return err("没有启用的审计功能", "VALIDATION_ERROR", 400)

    # Check for existing running audit
    from app.services.audit_engine import get_running_audit
    existing = get_running_audit(project_id)
    if existing:
        return err("该项目已有正在运行的审计", "AUDIT_RUNNING", 409)

    # Create run record
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_runs (project_id, user_id, status) VALUES (%s, %s, 'running') RETURNING id",
                (project_id, user_id))
            run_id = cur.fetchone()[0]
            thread_id = session.get('thread_id', '')
            if thread_id:
                cur.execute("UPDATE audit_runs SET task_id = %s WHERE id = %s",
                            (str(thread_id), run_id))
            conn.commit()

    # Launch background thread
    thread = threading.Thread(
        target=_run_audit_thread,
        args=(run_id, folder_ids, enabled_functions, extract_on_demand, user_id, file_ids, project_level_functions),
        daemon=True
    )
    thread.start()

    return ok({'run_id': run_id, 'enabled_functions': enabled_functions}, "审计已启动")


def _run_audit_thread(run_id, folder_ids, enabled_functions, extract_on_demand, user_id, file_ids, project_level_functions):
    """Thin wrapper to push app context into the background thread."""
    from flask import current_app
    try:
        from app.services.audit_engine import run_audit
        run_audit(run_id, folder_ids, enabled_functions, extract_on_demand, user_id, file_ids, project_level_functions)
    except Exception as e:
        logger.error(f"Audit thread crashed: {e}", exc_info=True)


# ── SSE Progress ──

@audit_bp.route('/progress/<int:run_id>', methods=['GET'])
@_admin_required
def progress(run_id):
    """SSE endpoint for live audit progress."""
    from app.services.audit_engine import register_progress_queue, unregister_progress_queue

    q = queue.Queue(maxsize=200)
    register_progress_queue(run_id, q)

    def generate():
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connected', 'run_id': run_id})}\n\n"

            while True:
                try:
                    msg = q.get(timeout=30)
                    yield f"data: {msg}\n\n"
                    # Check for terminal events
                    data = json.loads(msg)
                    if data.get('type') in ('complete', 'error'):
                        break
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except GeneratorExit:
            pass
        finally:
            unregister_progress_queue(run_id)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


# ── Results ──

@audit_bp.route('/result/<int:run_id>', methods=['GET'])
@_admin_required
def get_result(run_id):
    """Get full audit results."""
    from app.services.audit_engine import get_run_results
    data = get_run_results(run_id)
    if not data:
        return err("审计记录未找到", "NOT_FOUND", 404)
    return ok(data)


# ── History ──

@audit_bp.route('/history/<int:project_id>', methods=['GET'])
@_admin_required
def history(project_id):
    """List past audit runs for a project."""
    from app.services.audit_engine import get_project_history
    results = get_project_history(project_id)
    return ok(results)


# ── Running check ──

@audit_bp.route('/running/<int:project_id>', methods=['GET'])
@_admin_required
def running_check(project_id):
    """Check if a project has a running audit."""
    from app.services.audit_engine import get_running_audit
    result = get_running_audit(project_id)
    return ok(result)


# ── Downloads ──

@audit_bp.route('/download/<int:run_id>/docx', methods=['GET'])
@_admin_required
def download_docx(run_id):
    """Download the DOCX report."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT docx_path, overall_status FROM audit_runs WHERE id = %s", (run_id,))
            row = cur.fetchone()
    if not row or not row[0]:
        return err("报告文件不存在", "NOT_FOUND", 404)
    path = resolve_path(row[0])
    if not os.path.exists(path):
        return err("报告文件已被删除", "NOT_FOUND", 404)
    return send_file(
        path,
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        as_attachment=True,
        download_name=f"bid_audit_{run_id}.docx"
    )


@audit_bp.route('/download/<int:run_id>/xlsx', methods=['GET'])
@_admin_required
def download_xlsx(run_id):
    """Download the XLSX report."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT xlsx_path, overall_status FROM audit_runs WHERE id = %s", (run_id,))
            row = cur.fetchone()
    if not row or not row[0]:
        return err("报告文件不存在", "NOT_FOUND", 404)
    path = resolve_path(row[0])
    if not os.path.exists(path):
        return err("报告文件已被删除", "NOT_FOUND", 404)
    return send_file(
        path,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f"bid_audit_{run_id}.xlsx"
    )


# ── Config CRUD ──

@audit_bp.route('/config', methods=['GET'])
@_admin_required
def get_config():
    """Get the global audit configuration."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT function_name, enabled_by_default, fail_threshold, weight, severity_thresholds FROM audit_config ORDER BY function_name")
            configs = []
            for row in cur.fetchall():
                configs.append({
                    'function_name': row[0],
                    'enabled_by_default': row[1],
                    'fail_threshold': row[2],
                    'weight': row[3],
                    'severity_thresholds': row[4] or {},
                })
    return ok(configs)


@audit_bp.route('/config', methods=['PUT'])
@_admin_required
def update_config():
    """Update the global audit configuration."""
    data = request.get_json(silent=True) or {}
    configs = data.get('configs', [])
    if not configs or not isinstance(configs, list):
        return err("请提供 configs 数组", "VALIDATION_ERROR", 400)

    # Normalize weights
    total_weight = sum(c.get('weight', 0) for c in configs)
    if total_weight > 0:
        for c in configs:
            c['weight'] = c.get('weight', 0) / total_weight

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for cfg in configs:
                func_name = cfg.get('function_name')
                if not func_name:
                    continue
                cur.execute("""
                    UPDATE audit_config
                    SET enabled_by_default = %s,
                        fail_threshold = %s,
                        weight = %s,
                        severity_thresholds = %s::jsonb,
                        updated_at = NOW()
                    WHERE function_name = %s
                """, (
                    cfg.get('enabled_by_default', True),
                    cfg.get('fail_threshold', 50),
                    cfg.get('weight', 14.28),
                    json.dumps(cfg.get('severity_thresholds', {}), ensure_ascii=False),
                    func_name,
                ))
            conn.commit()

    return ok({"updated": len(configs)}, "配置已更新")