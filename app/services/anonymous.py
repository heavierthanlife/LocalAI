"""Anonymous user session helpers (auto-extracted)."""
import os, json, logging, atexit, uuid
import shutil
from flask import session
from app.config import TEMP_ROOT, logger
from app.utils.helpers import beijing_now
from app.database import get_db_connection

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

# Anonymous session storage — PostgreSQL-backed (anon_chat_messages), atomic
# UPSERT per thread. Previously each thread was a JSON file; files are subject
# to read-modify-write races and loss on crash. See FIX-2026-08-28-004.

def get_anon_history_path(thread_id):
    """Compat shim — history now lives in PG (anon_chat_messages)."""
    user_id = session.get('user_id') or str(uuid.uuid4())
    temp_dir = get_anon_temp_dir(user_id)
    return os.path.join(temp_dir, f"{thread_id}_history.json")

def get_session_messages_anon(thread_id):
    if not thread_id:
        return []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT messages FROM anon_chat_messages WHERE thread_id = %s",
                    (thread_id,))
                row = cur.fetchone()
                if row is None:
                    return []
                messages = row[0]
                return messages if isinstance(messages, list) else []
    except Exception as e:
        logger.error(f"Failed to read anon history {thread_id}: {e}")
        return []

def store_message_anon(thread_id, role, content, thinking=None):
    if not thread_id:
        logger.error("store_message_anon: missing thread_id")
        return
    message = {
        "role": role,
        "content": content,
        "thinking": thinking,
        "timestamp": beijing_now()
    }
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Atomic: append to existing JSONB array, or create if absent.
                cur.execute("""
                    INSERT INTO anon_chat_messages (thread_id, messages, updated_at)
                    VALUES (%s, %s::jsonb, NOW())
                    ON CONFLICT (thread_id) DO UPDATE
                      SET messages = anon_chat_messages.messages || EXCLUDED.messages,
                          updated_at = NOW()
                """, (thread_id, json.dumps([message], ensure_ascii=False)))
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to write anon history {thread_id}: {e}")

