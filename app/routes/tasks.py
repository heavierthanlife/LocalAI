"""Async task progress endpoints: polling + SSE streaming.

Blueprint: tasks_bp
- GET  /tasks                  List recent tasks (sidebar)
- GET  /tasks/<task_id>        Get task status (polling)
- GET  /tasks/<task_id>/stream SSE progress stream
- POST /tasks/<task_id>/delete Delete task from registry
- POST /tasks/<task_id>/cancel Cancel a running task (revoke + mark failed)
"""
import json
import logging

from flask import Blueprint, request, jsonify, Response, session

from app.services.task_bus import TaskBus, META_TTL
from app.utils.helpers import task_owner_ok
from app.services.session_manager import get_user_id

logger = logging.getLogger(__name__)

tasks_bp = Blueprint('tasks', __name__, url_prefix='/tasks')


def _auth_guard():
    """Return (user_id, error_response). error_response is None when allowed."""
    if session.get('consent_value', 0) != 1:
        return None, (jsonify({'error': 'Please login'}), 401)
    user_id = get_user_id()
    if not user_id:
        return None, (jsonify({'error': 'Please login'}), 401)
    return user_id, None


@tasks_bp.route('', methods=['GET'])
def list_tasks():
    """Return recent tasks for the Background Tasks sidebar.

    Anonymous callers get an empty list. Logged-in callers see only their own
    tasks (ownership is recorded in task meta at register time; legacy tasks
    without user_id are excluded from the list).
    """
    user_id, auth_err = _auth_guard()
    if auth_err:
        return jsonify({'tasks': [], 'count': 0})
    limit = request.args.get('limit', 50, type=int)
    tasks = TaskBus.list_tasks(limit=min(limit, 200))
    tasks = [t for t in tasks if str(t.get('user_id') or '') == str(user_id)]
    return jsonify({'tasks': tasks, 'count': len(tasks)})


@tasks_bp.route('/<task_id>', methods=['GET'])
def get_task(task_id: str):
    """Get a single task's current status (for polling fallback)."""
    user_id, auth_err = _auth_guard()
    if auth_err:
        return auth_err
    meta = TaskBus.get(task_id)
    if not meta:
        return jsonify({'error': 'Task not found'}), 404
    if not task_owner_ok(meta, user_id):
        return jsonify({'error': 'Forbidden'}), 403
    meta['task_id'] = task_id
    # Parse result JSON string → dict so the frontend doesn't need double-parsing
    if isinstance(meta.get('result'), str) and meta['result']:
        try:
            meta['result'] = json.loads(meta['result'])
        except (json.JSONDecodeError, TypeError):
            pass
    return jsonify(meta)


@tasks_bp.route('/<task_id>/delete', methods=['POST'])
def delete_task(task_id: str):
    """Delete a task from the registry (log-out removes from sidebar)."""
    user_id, auth_err = _auth_guard()
    if auth_err:
        return auth_err
    meta = TaskBus.get(task_id)
    if meta and not task_owner_ok(meta, user_id):
        return jsonify({'error': 'Forbidden'}), 403
    TaskBus.delete(task_id)
    return jsonify({'success': True})


@tasks_bp.route('/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id: str):
    """Cancel a running task: revoke the celery task + mark as failed so the
    sidebar's delete flow becomes available."""
    user_id, auth_err = _auth_guard()
    if auth_err:
        return auth_err
    meta = TaskBus.get(task_id)
    if not meta:
        return jsonify({'error': 'Task not found'}), 404
    if not task_owner_ok(meta, user_id):
        return jsonify({'error': 'Forbidden'}), 403
    if meta.get('status') not in ('running', 'queued', 'pending'):
        return jsonify({'error': f"Task already {meta.get('status')}"}), 409

    try:
        from celery.result import AsyncResult
        from celery_app import celery as _celery
        result = AsyncResult(task_id, app=_celery)
        result.revoke(terminate=True, signal='SIGTERM')
    except Exception:
        logger.warning(f"Revoke failed for {task_id} (task may have finished)", exc_info=True)

    from app.services.task_bus import TaskBus as _TB
    _TB(task_id).fail('已手动取消')
    return jsonify({'success': True})


@tasks_bp.route('/<task_id>/stream', methods=['GET'])
def stream_task(task_id: str):
    """SSE endpoint: stream task progress events to browser.
    
    Browser: new EventSource('/tasks/abc123/stream')
    """
    user_id, auth_err = _auth_guard()
    if auth_err:
        return auth_err
    meta = TaskBus.get(task_id)
    if not meta:
        return jsonify({'error': 'Task not found'}), 404
    if not task_owner_ok(meta, user_id):
        return jsonify({'error': 'Forbidden'}), 403

    timeout = request.args.get('timeout', 300, type=int)

    def generate():
        yield ": ok\n\n"  # SSE handshake
        for event in TaskBus.subscribe(task_id, timeout=min(timeout, 600)):
            yield event

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        }
    )
