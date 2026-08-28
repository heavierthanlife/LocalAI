"""Blueprint: document analysis — deep analysis of bidding documents."""
import json
import logging
import uuid
from datetime import datetime, timezone

from flask import Blueprint, request, Response, session

from app.utils.helpers import ok, err
from app.database import get_db_connection
from app.services.session_manager import get_user_id

logger = logging.getLogger(__name__)

document_analysis_bp = Blueprint('document_analysis', __name__, url_prefix='/document_analysis')


@document_analysis_bp.before_request
def _increase_body_limit():
    # 深度分析需上传多份投标文件（PDF 扫描件），全局 50MB 不够。提高到 200MB。
    request.max_content_length = 200 * 1024 * 1024


@document_analysis_bp.route('/analyze', methods=['POST'])
def start_analysis():
    """Submit files for deep analysis. Launches Celery task, returns task_id for SSE.

    Accepts either pre-uploaded `file_ids` (large-file friendly — the worker
    extracts text from disk) or direct multipart `files` (small uploads).
    """
    if session.get('consent_value', 0) != 1:
        return err("请先登录", "AUTH_REQUIRED", 401)
    user_id = get_user_id()
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)

    file_ids = request.form.getlist('file_ids')
    uploaded_files = request.files.getlist('files') if 'files' in request.files else []
    if not file_ids and not uploaded_files:
        return err("No files uploaded", "VALIDATION_ERROR", 400)

    from app.config import allowed_file
    if len(file_ids) + len(uploaded_files) < 1:
        return err("Need at least 1 file", "VALIDATION_ERROR", 400)
    if len(file_ids) + len(uploaded_files) > 10:
        return err("Maximum 10 files allowed", "VALIDATION_ERROR", 400)

    # Pre-uploaded path specs — extracted in the worker, page-by-page
    from app.services.file_store import resolve as resolve_file
    file_specs = []
    for fid in file_ids:
        info = resolve_file(fid, user_id=user_id)
        if not info:
            return err(f"文件不存在或无权访问: file_id={fid}", "NOT_FOUND", 404)
        if not allowed_file(info['filename']):
            continue
        file_specs.append({'abs_path': info['abs_path'], 'filename': info['filename']})

    # Legacy direct upload (small files) — extract inline
    from app.services.file_processing import extract_text_from_file, extract_metadata
    file_data = []
    for f in uploaded_files:
        if not f.filename or not allowed_file(f.filename):
            continue
        text, _ = extract_text_from_file(f)
        if not text or text.startswith("["):
            continue
        meta = extract_metadata(f)
        file_data.append({
            'filename': f.filename,
            'text': text,
            'metadata': meta or {},
            'images': [],
        })

    if not file_data and not file_specs:
        return err("Could not find any valid files", "VALIDATION_ERROR", 400)

    task_id = str(uuid.uuid4())
    thread_id = session.get('thread_id', '')
    project_id = request.form.get('project_id', type=int)

    # Pre-register the task as 'queued' so /document_analysis/status never 404s
    # before the Celery worker calls bus.start() (which upgrades it).
    from app.services.task_bus import TaskBus
    TaskBus(task_id, 'document_analysis', '深度分析').register_queued(extra={'thread_id': thread_id or ''})

    from celery_app import celery
    celery.send_task(
        'document_analysis_task',
        args=[file_data, file_specs, user_id, thread_id, task_id, project_id],
        task_id=task_id,
    )

    return ok({'task_id': task_id, 'file_count': len(file_data) + len(file_specs)})


@document_analysis_bp.route('/status/<task_id>', methods=['GET'])
def analysis_status(task_id):
    """Check if task has already completed (for SSE reconnection edge case)."""
    from app.services.task_bus import TaskBus
    meta = TaskBus.get(task_id)
    if not meta:
        return err("Task not found", "NOT_FOUND", 404)
    completed = meta.get('status') == 'completed'
    from app.utils.helpers import ok as _ok
    return _ok({
        'completed': completed,
        'status': meta.get('status', ''),
        'progress': meta.get('progress', 0),
        'message': meta.get('message', ''),
        'result': json.loads(meta.get('result', 'null')) if completed and meta.get('result') else None,
    })


@document_analysis_bp.route('/stream/<task_id>', methods=['GET'])
def stream_progress(task_id):
    """SSE endpoint for analysis progress."""
    from app.services.task_bus import TaskBus
    return Response(
        TaskBus.subscribe(task_id),
        mimetype='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        }
    )
