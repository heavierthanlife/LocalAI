"""Task locking and concurrency control (auto-extracted)."""
import logging
from datetime import datetime, timezone
from flask import session
import app.globals as g
from app.services.session_manager import get_chat_short_name

logger = logging.getLogger(__name__)

def cleanup_stale_tasks():
    with g.user_task_lock:
        now = datetime.now(timezone.utc)
        stale = [uid for uid, info in g.user_active_tasks.items() if
                 (now - info['start_time']).total_seconds() > g.TASK_TIMEOUT_SECONDS]
        for uid in stale:
            logger.warning(f"Cleaning stale task lock for user {uid}")
            del g.user_active_tasks[uid]

def acquire_task_lock(user_id, thread_id, task_type):
    with g.user_task_lock:
        cleanup_stale_tasks()
        if user_id in g.user_active_tasks:
            busy = g.user_active_tasks[user_id]
            return False, busy['thread_id'], get_chat_short_name(busy['thread_id'])
        else:
            g.user_active_tasks[user_id] = {'thread_id': thread_id, 'task_type': task_type, 'start_time': datetime.now(timezone.utc)}
            return True, None, None

def release_task_lock(user_id):
    with g.user_task_lock:
        if user_id in g.user_active_tasks:
            del g.user_active_tasks[user_id]
