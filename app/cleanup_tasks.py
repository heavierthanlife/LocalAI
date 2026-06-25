"""Background cleanup tasks run by the scheduler."""
import os
import shutil
import logging
from datetime import datetime, timezone, timedelta

from .config import TEMP_ROOT
from .database import get_db_connection

logger = logging.getLogger(__name__)


def cleanup_old_sessions(days=15):
    """Clean up old Flask session files."""
    try:
        from .config import SESSION_DIR
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
        count = 0
        if os.path.exists(SESSION_DIR):
            for f in os.listdir(SESSION_DIR):
                fp = os.path.join(SESSION_DIR, f)
                if os.path.isfile(fp) and os.path.getmtime(fp) < cutoff:
                    os.remove(fp)
                    count += 1
        if count:
            logger.info(f"Cleaned up {count} old session files.")
    except Exception as e:
        logger.error(f"Failed to cleanup old sessions: {e}")


def delete_expired_original_files():
    """Delete original files whose expiration has passed."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, original_stored_path FROM user_files
                    WHERE original_expires_at IS NOT NULL AND original_expires_at < NOW()
                """)
                expired = cur.fetchall()
                for file_id, original_path in expired:
                    if original_path and os.path.exists(original_path):
                        try:
                            os.remove(original_path)
                        except Exception:
                            pass
                    cur.execute("UPDATE user_files SET original_stored_path = NULL WHERE id = %s", (file_id,))
                if expired:
                    cur.execute("DELETE FROM user_files WHERE original_expires_at IS NOT NULL AND original_expires_at < NOW()")
                    conn.commit()
                    logger.info(f"Cleaned up {len(expired)} expired original files.")
    except Exception as e:
        logger.error(f"Failed to delete expired original files: {e}")


def cleanup_stale_tasks():
    """Release stale task locks."""
    try:
        from .globals import user_active_tasks, user_task_lock, TASK_TIMEOUT_SECONDS
        with user_task_lock:
            now = datetime.now().timestamp()
            stale = [uid for uid, v in user_active_tasks.items() if now - v.get('start', 0) > TASK_TIMEOUT_SECONDS]
            for uid in stale:
                del user_active_tasks[uid]
            if stale:
                logger.info(f"Released {len(stale)} stale task locks.")
    except Exception as e:
        logger.error(f"Failed to cleanup stale tasks: {e}")


def cleanup_stale_message_responses():
    """Clean up stale message response placeholders."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
                cur.execute(
                    "DELETE FROM message_responses WHERE created_at < %s AND (assistant_response = '' OR assistant_response IS NULL)",
                    (cutoff,)
                )
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to cleanup stale message responses: {e}")


def cleanup_old_anon_temp_files(days=1):
    """Clean up anonymous user temp files."""
    try:
        cutoff = datetime.now().timestamp() - days * 86400
        count = 0
        if os.path.exists(TEMP_ROOT):
            for d in os.listdir(TEMP_ROOT):
                dp = os.path.join(TEMP_ROOT, d)
                if os.path.isdir(dp) and os.path.getmtime(dp) < cutoff:
                    shutil.rmtree(dp)
                    count += 1
        if count:
            logger.info(f"Cleaned up {count} old anonymous temp dirs.")
    except Exception as e:
        logger.error(f"Failed to cleanup old anon temp files: {e}")


def schedule_project_deletion_cleanup():
    """Delete projects that have been scheduled for deletion more than 30 days ago."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                cur.execute(
                    "SELECT id FROM projects WHERE deletion_scheduled_at IS NOT NULL AND deletion_scheduled_at < %s",
                    (cutoff,)
                )
                to_delete = cur.fetchall()
                for (project_id,) in to_delete:
                    cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
                conn.commit()
                if to_delete:
                    logger.info(f"Deleted {len(to_delete)} scheduled-for-deletion projects.")
    except Exception as e:
        logger.error(f"Failed to cleanup scheduled project deletions: {e}")


def cleanup_expired_recycle_bin():
    """Clean up expired items from recycle bins."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT stored_path FROM recycle_bin WHERE expires_at < NOW()")
                paths = cur.fetchall()
                for (sp,) in paths:
                    if sp and os.path.exists(sp):
                        try:
                            os.remove(sp)
                        except Exception:
                            pass
                cur.execute("DELETE FROM recycle_bin WHERE expires_at < NOW()")

                cur.execute("SELECT stored_path FROM project_recycle_bin WHERE expires_at < NOW()")
                paths = cur.fetchall()
                for (sp,) in paths:
                    if sp and os.path.exists(sp):
                        try:
                            os.remove(sp)
                        except Exception:
                            pass
                cur.execute("DELETE FROM project_recycle_bin WHERE expires_at < NOW()")

                cur.execute("DELETE FROM project_folders_recycle_bin WHERE expires_at < NOW()")
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to cleanup expired recycle bin: {e}")


def cleanup_orphan_users():
    """Remove users that have no data associated."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM users WHERE user_id NOT IN (
                        SELECT DISTINCT user_id FROM chat_sessions WHERE user_id IS NOT NULL
                        UNION
                        SELECT DISTINCT user_id FROM user_files WHERE user_id IS NOT NULL
                        UNION
                        SELECT DISTINCT user_id FROM project_members WHERE user_id IS NOT NULL
                    ) AND role IS DISTINCT FROM 'admin'
                """)
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to cleanup orphan users: {e}")
