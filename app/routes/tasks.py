"""Async task progress endpoints: polling + SSE streaming.

Blueprint: tasks_bp
- GET  /tasks                  List recent tasks (sidebar)
- GET  /tasks/<task_id>        Get task status (polling)
- GET  /tasks/<task_id>/stream SSE progress stream
"""
import json
import logging

from flask import Blueprint, request, jsonify, Response, session

from app.services.task_bus import TaskBus, META_TTL

logger = logging.getLogger(__name__)

tasks_bp = Blueprint('tasks', __name__, url_prefix='/tasks')


@tasks_bp.route('', methods=['GET'])
def list_tasks():
    """Return recent tasks for the Background Tasks sidebar.
    
    Filters by current user_id if logged in, otherwise shows public tasks.
    """
    user_id = session.get('user_id')
    limit = request.args.get('limit', 50, type=int)
    tasks = TaskBus.list_tasks(limit=min(limit, 200))

    # Filter to user's own tasks if logged in
    if user_id:
        tasks = [t for t in tasks if t.get('user_id') == str(user_id) or not t.get('user_id')]

    return jsonify({'tasks': tasks, 'count': len(tasks)})


@tasks_bp.route('/<task_id>', methods=['GET'])
def get_task(task_id: str):
    """Get a single task's current status (for polling fallback)."""
    meta = TaskBus.get(task_id)
    if not meta:
        return jsonify({'error': 'Task not found'}), 404
    meta['task_id'] = task_id
    return jsonify(meta)


@tasks_bp.route('/<task_id>/stream', methods=['GET'])
def stream_task(task_id: str):
    """SSE endpoint: stream task progress events to browser.
    
    Browser: new EventSource('/tasks/abc123/stream')
    """
    meta = TaskBus.get(task_id)
    if not meta:
        return jsonify({'error': 'Task not found'}), 404

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
