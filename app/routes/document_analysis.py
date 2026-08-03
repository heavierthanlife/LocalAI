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


@document_analysis_bp.route('/analyze', methods=['POST'])
def start_analysis():
    """Submit files for deep analysis. Launches Celery task, returns task_id for SSE."""
    if session.get('consent_value', 0) != 1:
        return err("请先登录", "AUTH_REQUIRED", 401)
    user_id = get_user_id()
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)

    if 'files' not in request.files:
        return err("No files uploaded", "VALIDATION_ERROR", 400)

    uploaded_files = request.files.getlist('files')
    from app.config import allowed_file
    if len(uploaded_files) < 1:
        return err("Need at least 1 file", "VALIDATION_ERROR", 400)
    if len(uploaded_files) > 10:
        return err("Maximum 10 files allowed", "VALIDATION_ERROR", 400)

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

    if not file_data:
        return err("Could not extract valid text from any file", "VALIDATION_ERROR", 400)

    task_id = str(uuid.uuid4())
    thread_id = session.get('thread_id', '')
    project_id = request.form.get('project_id', type=int)

    from celery_app import celery
    celery.send_task(
        'document_analysis_task',
        args=[file_data, user_id, thread_id, task_id, project_id],
        task_id=task_id,
    )

    return ok({'task_id': task_id, 'file_count': len(file_data)})


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
