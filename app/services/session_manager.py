"""Session and chat management helpers (auto-extracted)."""
import os, json, uuid, logging
from datetime import datetime, timezone, timedelta
from psycopg2.extras import RealDictCursor
from flask import session
from app.config import DUMP_DIR, logger
from app.database import get_db_connection, db_transaction
from app.utils.helpers import utc_now, beijing_now
from app.services.file_cache import file_cache_manager

BEIJING_TZ = timezone(timedelta(hours=8))

from app.services.anonymous import get_session_messages_anon, store_message_anon, get_anon_temp_dir, get_anon_history_path

def db_execute_readonly(cur):
    cur.execute("SET TRANSACTION READ ONLY")

# ── User ID helpers ──
def get_user_id():
    if session.get('consent_value', 0) == 1:
        if 'user_id' in session:
            return session['user_id']
        session['user_id'] = str(uuid.uuid4())
        return session['user_id']
    else:
        if 'temp_user_id' not in session:
            session['temp_user_id'] = str(uuid.uuid4())
        return session['temp_user_id']

def ensure_user_exists(user_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (user_id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
            conn.commit()

def get_or_create_session(thread_id, title=None):
    if session.get('consent_value', 0) != 1:
        return
    user_id = get_user_id()
    ensure_user_exists(user_id)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM chat_sessions WHERE thread_id = %s", (thread_id,))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO chat_sessions (user_id, thread_id, title, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, thread_id, title or "新对话", utc_now(), utc_now())
                )
                conn.commit()

def generate_session_title(messages, max_len=20):
    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.get('content', '').strip()
            if content:
                title = content[:max_len]
                if len(content) > max_len:
                    title += '...'
                return title
    return '新对话'

def update_session_title(thread_id, title):
    if session.get('consent_value', 0) != 1:
        return
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET title = %s, updated_at = %s WHERE thread_id = %s",
                (title, utc_now(), thread_id)
            )
            conn.commit()

def store_message(thread_id, role, content, thinking=None):
    if session.get('consent_value', 0) != 1:
        store_message_anon(thread_id, role, content, thinking)
        return None
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_messages (thread_id, role, content, thinking, timestamp) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (thread_id, role, content, thinking, utc_now())
                )
                msg_id = cur.fetchone()[0]
                cur.execute(
                    "UPDATE chat_sessions SET updated_at = %s WHERE thread_id = %s",
                    (utc_now(), thread_id)
                )
    messages = get_session_messages(thread_id)
    if len(messages) == 2:
        new_title = generate_session_title(messages)
        update_session_title(thread_id, new_title)
    return msg_id

def get_session_messages(thread_id):
    if session.get('consent_value', 0) != 1:
        return get_session_messages_anon(thread_id)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            db_execute_readonly(cur)
            cur.execute(
                "SELECT role, content, thinking, timestamp FROM chat_messages WHERE thread_id = %s ORDER BY id ASC",
                (thread_id,)
            )
            rows = cur.fetchall()
            messages = []
            for row in rows:
                ts_utc = row['timestamp']
                ts_beijing = ts_utc.astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S') if ts_utc else None
                messages.append({
                    "role": row['role'],
                    "content": row['content'],
                    "thinking": row['thinking'],
                    "timestamp": ts_beijing
                })
            return messages

def get_user_sessions():
    if session.get('consent_value', 0) != 1:
        return []
    user_id = get_user_id()
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            db_execute_readonly(cur)
            cur.execute("""
                (SELECT cs.thread_id, cs.title, cs.created_at, cs.updated_at, cs.project_id, cs.is_grilling
                 FROM chat_sessions cs
                 WHERE cs.user_id = %s)
                UNION
                (SELECT cs.thread_id, cs.title, cs.created_at, cs.updated_at, cs.project_id, cs.is_grilling
                 FROM chat_sessions cs
                 JOIN project_members pm ON cs.project_id = pm.project_id
                 WHERE pm.user_id = %s AND pm.status = 'active')
                ORDER BY updated_at DESC
            """, (user_id, user_id))
            rows = cur.fetchall()
            sessions = []
            for row in rows:
                sessions.append({
                    "thread_id": row['thread_id'],
                    "title": row['title'],
                    "created_at": row['created_at'].astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S') if row['created_at'] else None,
                    "updated_at": row['updated_at'].astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S') if row['updated_at'] else None,
                    "project_id": row.get('project_id'),
                    "is_grilling": row.get('is_grilling', False)
                })
            return sessions

def delete_session(thread_id):
    try:
        with get_db_connection() as conn:
            with db_transaction(conn):
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT user_id FROM chat_sessions WHERE thread_id = %s", (thread_id,))
                    row = cur.fetchone()
                    if not row:
                        return
                    user_id = row['user_id']

                    cur.execute("SELECT id, filename, content, size_bytes, original_stored_path, file_hash, thread_id FROM user_files WHERE thread_id = %s", (thread_id,))
                    files = cur.fetchall()
                    for f in files:
                        cur.execute("""
                            INSERT INTO recycle_bin 
                            (original_table, original_id, user_id, file_name, file_content, file_size, original_stored_path, file_hash, thread_id, original_thread_id, deletion_reason, deleted_at, expires_at, uploaded_by, deleted_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '3 days', %s, %s)
                        """, ('user_files', f['id'], user_id, f['filename'], f['content'], f['size_bytes'], f['original_stored_path'], f['file_hash'], f['thread_id'], thread_id, 'chat_deleted', user_id, user_id))

                    cur.execute("DELETE FROM user_files WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM chat_messages WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM file_usage WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM feedback WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM consent WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM chat_sessions WHERE thread_id = %s", (thread_id,))
        logger.info(f"Deleted session {thread_id} and moved {len(files)} files to recycle bin")
        file_cache_manager.clear_thread(thread_id)
    except Exception as e:
        logger.error(f"Failed to delete session {thread_id}: {e}", exc_info=True)
        raise

def archive_session(thread_id, user_id, reason="manual"):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT archive_path FROM archived_sessions WHERE thread_id = %s", (thread_id,))
            if cur.fetchone():
                return None
            cur.execute("SELECT title, created_at, updated_at FROM chat_sessions WHERE thread_id = %s", (thread_id,))
            sess_row = cur.fetchone()
            if not sess_row:
                return None
            title = sess_row['title']
            created_at = sess_row['created_at']
            updated_at = sess_row['updated_at']
            cur.execute("SELECT role, content, thinking, timestamp FROM chat_messages WHERE thread_id = %s ORDER BY timestamp", (thread_id,))
            messages = []
            for row in cur.fetchall():
                messages.append({
                    "role": row['role'],
                    "content": row['content'],
                    "thinking": row['thinking'],
                    "timestamp": row['timestamp']
                })
            cur.execute("SELECT user_message, assistant_response, rating, comment, timestamp FROM feedback WHERE thread_id = %s", (thread_id,))
            feedbacks = []
            for row in cur.fetchall():
                feedbacks.append({
                    "user_message": row['user_message'],
                    "assistant_response": row['assistant_response'],
                    "rating": row['rating'],
                    "comment": row['comment'],
                    "timestamp": row['timestamp']
                })
            cur.execute("SELECT consent_given, timestamp FROM consent WHERE thread_id = %s", (thread_id,))
            consent_row = cur.fetchone()
            consent = {"consent_given": consent_row['consent_given'], "timestamp": consent_row['timestamp']} if consent_row else None
            archive_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            dump_dir = str(DUMP_DIR / user_id / f"{user_id}_{archive_date}")
            os.makedirs(dump_dir, exist_ok=True)
            if not os.path.isdir(dump_dir):
                logger.error(f"Failed to create archive directory: {dump_dir}")
                return None
            session_info = {
                "thread_id": thread_id,
                "user_id": user_id,
                "title": title,
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "reason": reason
            }
            try:
                with open(os.path.join(dump_dir, f"{thread_id}_session.json"), "w", encoding="utf-8") as f:
                    json.dump(session_info, f, ensure_ascii=False, indent=2, default=str)
                with open(os.path.join(dump_dir, f"{thread_id}_messages.json"), "w", encoding="utf-8") as f:
                    json.dump(messages, f, ensure_ascii=False, indent=2, default=str)
                if feedbacks:
                    with open(os.path.join(dump_dir, f"{thread_id}_feedback.json"), "w", encoding="utf-8") as f:
                        json.dump(feedbacks, f, ensure_ascii=False, indent=2, default=str)
                if consent:
                    with open(os.path.join(dump_dir, f"{thread_id}_consent.json"), "w", encoding="utf-8") as f:
                        json.dump(consent, f, ensure_ascii=False, indent=2, default=str)
            except Exception as e:
                logger.error(f"Failed to write archive files for thread {thread_id}: {e}")
                return None
            archive_path = os.path.join(dump_dir, f"{thread_id}_session.json")
            cur.execute("INSERT INTO archived_sessions (thread_id, user_id, archive_path) VALUES (%s, %s, %s)", (thread_id, user_id, archive_path))
            conn.commit()
            logger.info(f"Archived session {thread_id} for user {user_id} to {dump_dir}")
            return dump_dir

def cleanup_old_sessions(days=15):
    cutoff = utc_now() - timedelta(days=days)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT thread_id, user_id FROM chat_sessions WHERE updated_at < %s", (cutoff,))
            old = cur.fetchall()
            for thread_id, user_id in old:
                archive_session(thread_id, user_id, reason="auto_15days")
                delete_session(thread_id)

def cleanup_stale_message_responses(hours=1):
    cutoff = utc_now() - timedelta(hours=hours)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM message_responses WHERE created_at < %s AND (assistant_response = '' OR assistant_response IS NULL)", (cutoff,))
            conn.commit()
            logger.info(f"Deleted stale message response placeholders older than {hours} hours.")


def get_cached_image_description(file_hash):
    if session.get('consent_value', 0) != 1:
        return None
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT description FROM image_description_cache WHERE file_hash = %s", (file_hash,))
            row = cur.fetchone()
            if row:
                return row[0]
    return None

def cache_image_description(file_hash, description):
    if session.get('consent_value', 0) != 1:
        return
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO image_description_cache (file_hash, description)
                VALUES (%s, %s)
                ON CONFLICT (file_hash) DO UPDATE
                SET description = EXCLUDED.description, created_at = NOW()
            """, (file_hash, description))
            conn.commit()


def get_user_total_storage_size(user_id):
    if session.get('consent_value', 0) != 1:
        return 0
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM user_files WHERE user_id = %s AND (expires_at IS NULL OR expires_at > NOW())",
                (user_id,))
            return cur.fetchone()[0]

def record_file_usage(thread_id, filename, usage_type, question_text=None):
    if session.get('consent_value', 0) != 1:
        return
    user_id = get_user_id()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO file_usage (user_id, thread_id, filename, usage_type, question) VALUES (%s, %s, %s, %s, %s)",
                (user_id, thread_id, filename, usage_type, question_text)
            )
            conn.commit()

def get_chat_short_name(thread_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT title FROM chat_sessions WHERE thread_id = %s", (thread_id,))
            row = cur.fetchone()
            if row and row[0]:
                name = row[0]
                return name if len(name) <= 20 else name[:17] + '...'
            return "新对话"

