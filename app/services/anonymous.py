"""Anonymous user session helpers (auto-extracted)."""
import os, json, logging, atexit, uuid
import shutil
from flask import session
from app.config import TEMP_ROOT, logger
from app.utils.helpers import beijing_now

def get_anon_temp_dir(anon_id):
    path = os.path.join(TEMP_ROOT, anon_id)
    os.makedirs(path, exist_ok=True)
    return path

def cleanup_anon_temp(anon_id):
    path = os.path.join(TEMP_ROOT, anon_id)
    if os.path.exists(path):
        shutil.rmtree(path)
        logger.info(f"Cleaned up temp files for anon user {anon_id}")

def cleanup_all_temp_on_exit():
    # Concurrent gunicorn workers exiting together can race on rmtree
    # (one worker deletes the dir while another is mid-removal). Wrap in
    # try/except so an already-gone dir never aborts process teardown.
    try:
        if os.path.exists(TEMP_ROOT):
            shutil.rmtree(TEMP_ROOT, ignore_errors=True)
            logger.info("Cleaned up all temp files on exit.")
    except Exception:
        pass

atexit.register(cleanup_all_temp_on_exit)

# Anonymous session storage
def get_anon_history_path(thread_id):
    user_id = session.get('user_id') or str(uuid.uuid4())
    temp_dir = get_anon_temp_dir(user_id)
    return os.path.join(temp_dir, f"{thread_id}_history.json")

def get_session_messages_anon(thread_id):
    path = get_anon_history_path(thread_id)
    if not os.path.exists(path):
        return []
    try:
        from filelock import FileLock
        lock_path = path + ".lock"
        with FileLock(lock_path, timeout=5):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except ImportError:
        logger.warning("filelock not installed. Anonymous file reads may have race conditions.")
        # No lock
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read anon history {thread_id}: {e}")
            return []
    except Exception as e:
        logger.error(f"Failed to read anon history {thread_id}: {e}")
        return []

def store_message_anon(thread_id, role, content, thinking=None):
    path = get_anon_history_path(thread_id)
    try:
        from filelock import FileLock
        lock_path = path + ".lock"
        with FileLock(lock_path, timeout=5):
            history = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            history.append({
                "role": role,
                "content": content,
                "thinking": thinking,
                "timestamp": beijing_now()
            })
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
    except ImportError:
        logger.warning("filelock not installed. Anonymous file writes may have race conditions.")
        # No lock
        history = []
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception as e:
                logger.debug(f"Failed to read anon history during store (fallback path, {thread_id}): {e}")
        history.append({
            "role": role,
            "content": content,
            "thinking": thinking,
            "timestamp": beijing_now()
        })
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to write anon history {thread_id}: {e}")

