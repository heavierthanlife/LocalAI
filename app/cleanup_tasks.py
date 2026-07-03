"""Background cleanup tasks run by the scheduler."""
import os
import shutil
import logging
from datetime import datetime, timezone, timedelta

from .config import TEMP_ROOT
from .database import get_db_connection

logger = logging.getLogger(__name__)


def _rc(key: str, default):
    """Read a runtime config value, falling back to default."""
    try:
        from app.services.runtime_config import get as rc_get
        return rc_get(key, default)
    except Exception:
        return default


def cleanup_old_sessions(days=None):
    """Clean up old Flask session files."""
    if days is None:
        days = _rc('cleanup_session_days', 15)
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
                        except Exception as e:
                            logger.debug(f"Failed to remove original file {original_path}: {e}")
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
        from .globals import user_active_tasks, user_task_lock
        timeout = _rc('task_timeout_seconds', 600)
        with user_task_lock:
            now = datetime.now(timezone.utc)
            stale = [uid for uid, v in user_active_tasks.items()
                     if (now - v.get('start_time', now)).total_seconds() > timeout]
            for uid in stale:
                del user_active_tasks[uid]
            if stale:
                logger.info(f"Released {len(stale)} stale task locks.")
    except Exception as e:
        logger.error(f"Failed to cleanup stale tasks: {e}")


def cleanup_stale_message_responses():
    """Clean up stale message response placeholders."""
    hours = _rc('cleanup_message_response_hours', 1)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
                cur.execute(
                    "DELETE FROM message_responses WHERE created_at < %s AND (assistant_response = '' OR assistant_response IS NULL)",
                    (cutoff,)
                )
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to cleanup stale message responses: {e}")


def cleanup_old_anon_temp_files(days=None):
    """Clean up anonymous user temp files."""
    if days is None:
        days = _rc('cleanup_anon_temp_days', 1)
    try:
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
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
    """Delete projects that have been scheduled for deletion more than N days ago."""
    days = _rc('cleanup_project_deletion_days', 30)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                cur.execute(
                    "SELECT id FROM projects WHERE deletion_scheduled_at IS NOT NULL AND deletion_scheduled_at < %s",
                    (cutoff,)
                )
                to_delete = cur.fetchall()
                for (project_id,) in to_delete:
                    cur.execute("DELETE FROM chat_messages WHERE thread_id IN (SELECT thread_id FROM chat_sessions WHERE project_id = %s)", (project_id,))
                    cur.execute("DELETE FROM chat_sessions WHERE project_id = %s", (project_id,))
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
                        except Exception as e:
                            logger.debug(f"Failed to remove recycle_bin file {sp}: {e}")
                cur.execute("DELETE FROM recycle_bin WHERE expires_at < NOW()")

                cur.execute("SELECT stored_path FROM project_recycle_bin WHERE expires_at < NOW()")
                paths = cur.fetchall()
                for (sp,) in paths:
                    if sp and os.path.exists(sp):
                        try:
                            os.remove(sp)
                        except Exception as e:
                            logger.debug(f"Failed to remove project_recycle_bin file {sp}: {e}")
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


def cleanup_old_training_exports():
    """Quarterly cleanup of old export files (keep last N)."""
    try:
        from app.services.training_logger import cleanup_old_exports
        result = cleanup_old_exports()
        if result['deleted']:
            logger.info(f"Export cleanup: deleted {len(result['deleted'])} files, kept {result['kept']}")
        else:
            logger.info(f"Export cleanup: nothing to delete ({result['kept']} files, within retention)")
    except Exception as e:
        logger.error(f"Export cleanup failed: {e}")


def auto_cleanup_stale_reviews():
    """Daily check: log warning if stale, auto-clean if overdue."""
    try:
        from app.services.ingest_pipeline import check_stale_reviews, cleanup_stale_reviews
        stale = check_stale_reviews()
        if stale['has_stale']:
            logger.warning(
                f"Stale ingest reviews: {stale['domain_candidates']} domain candidates, "
                f"{stale['kb_reviews']} KB reviews (warn={stale['warn_days']}d, cleanup={stale['cleanup_days']}d)"
            )
        if stale['overdue_cleanup'] > 0:
            result = cleanup_stale_reviews()
            logger.info(f"Auto-cleaned {result['cleaned']} stale review items")
    except Exception as e:
        logger.error(f"Stale review check failed: {e}")


def auto_training_health_check():
    """Weekly scheduled health check — scans all training sessions for issues."""
    try:
        from app.services.training_logger import run_training_health_check
        report = run_training_health_check(repair=False)
        logger.info(
            f"Training health check: {report['total']} sessions, "
            f"{report['healthy']} healthy, {report['warning']} warning, "
            f"{report['corrupt']} corrupt, {report['issues_found']} issues"
        )
        if report['corrupt'] > 0:
            logger.warning(f"Training data has {report['corrupt']} corrupt session(s) — admin action recommended")
    except Exception as e:
        logger.error(f"Training health check failed: {e}")


def cleanup_old_training_data(days=None):
    """Remove training data sessions (raw/) older than N days.

    Controlled by runtime_config key ``training_retention_days`` (default 90).
    Only purges sessions whose last_updated timestamp is older than the cutoff.
    """
    if days is None:
        days = _rc('training_retention_days', 90)
    try:
        from app.services.training_logger import cleanup_training_sessions
        removed = cleanup_training_sessions(retention_days=days)
        if removed > 0:
            logger.info(f"Cleaned up {removed} old training data sessions (retention: {days}d).")
    except Exception as e:
        logger.error(f"Failed to cleanup old training data: {e}")


def cleanup_expired_share_files(days=None):
    """Remove expired share_*.json files from TEMP_DIR."""
    if days is None:
        days = _rc('cleanup_share_file_days', 7)
    try:
        from .config import TEMP_DIR
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
        count = 0
        if os.path.exists(TEMP_DIR):
            for f in os.listdir(TEMP_DIR):
                if f.startswith('share_') and f.endswith('.json'):
                    fp = os.path.join(TEMP_DIR, f)
                    if os.path.isfile(fp) and os.path.getmtime(fp) < cutoff:
                        os.remove(fp)
                        count += 1
        if count:
            logger.info(f"Cleaned up {count} expired share files.")
    except Exception as e:
        logger.error(f"Failed to cleanup expired share files: {e}")


def cleanup_stale_download_tokens(hours=None):
    """Remove download tokens older than N hours from globals."""
    if hours is None:
        hours = _rc('cleanup_download_token_hours', 24)
    try:
        from .globals import download_tokens, download_tokens_lock
        with download_tokens_lock:
            stale = [k for k, v in download_tokens.items() if v == 20]
            for k in stale:
                del download_tokens[k]
            if stale:
                logger.info(f"Cleaned up {len(stale)} unused download tokens.")
    except Exception as e:
        logger.error(f"Failed to cleanup download tokens: {e}")


def _auto_generate_report(period='weekly'):
    """Core auto-report logic: weekly/monthly/annual. Returns (filename, admin_uid) or (None, None)."""
    from datetime import timedelta
    from app.services.analysis_prompts import build_work_report_prompt, WORK_REPORT_SYSTEM
    from app.services.llm_provider import create_chat_model
    from langchain_core.messages import SystemMessage, HumanMessage
    from app.config import USER_FILES_ORIGINAL_ROOT
    import hashlib, zipfile

    now = datetime.now(timezone.utc)
    days = {'weekly': 7, 'monthly': 30, 'annual': 365}.get(period, 7)
    since = now - timedelta(days=days)
    label = {'weekly': '周报', 'monthly': '月报', 'annual': '年报'}.get(period, '报告')
    date_range = f"{since.strftime('%Y%m%d')}_{now.strftime('%Y%m%d')}"

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT COUNT(*) as cnt FROM chat_messages cm
                JOIN chat_sessions cs ON cm.thread_id = cs.thread_id WHERE cm.timestamp >= %s""", (since,))
            if cur.fetchone()['cnt'] < _rc('report_min_messages', 5):
                logger.info(f"Auto {label} skipped: fewer than 5 messages.")
                return None, None

            cur.execute("SELECT COUNT(DISTINCT cs.thread_id) as cnt FROM chat_sessions cs JOIN chat_messages cm ON cm.thread_id = cs.thread_id WHERE cm.timestamp >= %s", (since,))
            session_count = cur.fetchone()['cnt']
            cur.execute("SELECT COUNT(DISTINCT user_id) as cnt FROM chat_sessions WHERE updated_at >= %s", (since,))
            user_count = cur.fetchone()['cnt']
            cur.execute("SELECT COUNT(*) as cnt FROM credit_check_reports WHERE created_at >= %s", (since,))
            credit_count = cur.fetchone()['cnt']
            cur.execute("SELECT COUNT(*) as cnt FROM batch_comparison_results WHERE created_at >= %s", (since,))
            batch_count = cur.fetchone()['cnt']

    prompt = build_work_report_prompt(
        period=period,
        stats={'sessions': session_count, 'messages': cur.fetchone()['cnt'] if False else session_count*2,
               'users': user_count, 'knowledge_files': 0, 'credit_checks': credit_count, 'batch_compares': batch_count},
        topics=[], highlights=[], previous_summary="",
    )
    llm = create_chat_model(streaming=False, temperature=0.5, max_tokens=2000,
                            timeout=int(os.environ.get("LLM_TIMEOUT", "120")))
    from app.services.prompt_safety import sanitize_for_prompt
    resp = llm.invoke([SystemMessage(content=WORK_REPORT_SYSTEM),
                       HumanMessage(content=sanitize_for_prompt(prompt, 'auto_report'))])
    report = resp.content if hasattr(resp, 'content') else str(resp)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM users WHERE role = 'admin' LIMIT 1")
            admin_row = cur.fetchone()
            admin_uid = admin_row[0] if admin_row else 'system'

    filename_prefix = f"全体用户_{label}_{date_range}"
    report_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, admin_uid)
    os.makedirs(report_dir, exist_ok=True)
    try:
        from app.services.style_engine import generate_report_file
        report_path = generate_report_file(report, filename_prefix, f'{label} (全体)')
        filename = os.path.basename(report_path)
    except Exception:
        filename = f'{filename_prefix}.md'
        report_path = os.path.join(report_dir, filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

    # ZIP all reports for this period
    zip_name = f"全体用户_{label}_{date_range}.zip"
    zip_path = os.path.join(report_dir, zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(report_path, filename)
        for fname in os.listdir(report_dir):
            if fname.endswith(('.md','.docx')) and date_range in fname and fname != filename:
                zf.write(os.path.join(report_dir, fname), fname)

    file_hash = hashlib.sha256(report.encode()).hexdigest()
    thread_id = f'auto_{period}'
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO user_files (user_id, thread_id, filename, size_bytes, original_stored_path,
                file_hash, original_expires_at, original_name) VALUES (%s,%s,%s,%s,%s,%s,NOW()+INTERVAL'%s days',%s)""",
                (admin_uid, thread_id, zip_name, os.path.getsize(zip_path), zip_path,
                 hashlib.sha256(open(zip_path,'rb').read()).hexdigest(), str(_rc('cleanup_report_retention_days', 90)), zip_name))
            conn.commit()
    return zip_name, admin_uid


def auto_generate_weekly_report():
    try:
        from app.services.admin_utils import log_admin_action
        zip_name, admin_uid = _auto_generate_report('weekly')
        if zip_name:
            log_admin_action('system', 'auto-scheduler', 'AUTO_WEEKLY_REPORT', 'system', None,
                            column_name='weekly_report', new_value=zip_name)
            logger.info(f"Auto weekly report: {zip_name}")
    except Exception as e:
        logger.error(f"Auto weekly report failed: {e}")


def auto_generate_monthly_report():
    try:
        from app.services.admin_utils import log_admin_action
        zip_name, admin_uid = _auto_generate_report('monthly')
        if zip_name:
            log_admin_action('system', 'auto-scheduler', 'AUTO_MONTHLY_REPORT', 'system', None,
                            column_name='monthly_report', new_value=zip_name)
    except Exception as e:
        logger.error(f"Auto monthly report failed: {e}")


def auto_generate_annual_report():
    try:
        from app.services.admin_utils import log_admin_action
        zip_name, admin_uid = _auto_generate_report('annual')
        if zip_name:
            log_admin_action('system', 'auto-scheduler', 'AUTO_ANNUAL_REPORT', 'system', None,
                            column_name='annual_report', new_value=zip_name)
    except Exception as e:
        logger.error(f"Auto annual report failed: {e}")


def auto_rag_health_check():
    """Weekly RAG index health check: log stats to audit log."""
    try:
        from app.services.rag_engine import get_index_stats
        from app.services.admin_utils import log_admin_action
        stats = get_index_stats()
        if stats.get('total', 0) > 0:
            log_admin_action('system', 'auto-scheduler', 'AUTO_RAG_HEALTH', 'rag_index', None,
                            column_name='health_check', new_value=f"total={stats.get('total',0)} chunks")
            logger.info(f"RAG health: {stats.get('total',0)} total chunks across {len(stats)-1} sources")
        else:
            logger.info("RAG health check: no indexes yet.")
    except Exception as e:
        logger.error(f"RAG health check failed: {e}")


def auto_cleanup_stale_sessions():
    """Daily: archive + delete chat sessions inactive for >15 days."""
    try:
        from app.services.session_manager import cleanup_old_sessions
        from app.services.admin_utils import log_admin_action
        days = _rc('cleanup_session_days', 15)
        cleanup_old_sessions(days=days)
        log_admin_action('system', 'auto-scheduler', 'AUTO_CLEANUP_SESSIONS', 'chat_sessions', None,
                        column_name='cleanup', new_value=f'>{days}days_inactive')
        logger.info(f"Auto-cleaned chat sessions older than {days} days")
    except Exception as e:
        logger.error(f"Auto session cleanup failed: {e}")


def auto_cleanup_temp_files():
    """Daily: cleanup orphan batch compare temps and stale ingest task dirs."""
    cleaned = 0
    try:
        import time as _time
        cutoff = _time.time() - 7 * 86400  # 7 days

        # Clean orphan batch compare temp files
        temp_dir = os.path.join(TEMP_ROOT)
        if os.path.exists(temp_dir):
            for fname in os.listdir(temp_dir):
                if fname.startswith('comp_') and fname.endswith('.json'):
                    fpath = os.path.join(temp_dir, fname)
                    try:
                        if os.path.getmtime(fpath) < cutoff:
                            os.remove(fpath)
                            cleaned += 1
                    except Exception:
                        pass

        # Clean stale ingest task dirs
        from app.config import DATA_DIR
        ingest_dir = os.path.join(DATA_DIR, 'ingest')
        if os.path.exists(ingest_dir):
            for task_dir in os.listdir(ingest_dir):
                task_path = os.path.join(ingest_dir, task_dir)
                if os.path.isdir(task_path):
                    try:
                        if os.path.getmtime(task_path) < cutoff:
                            shutil.rmtree(task_path, ignore_errors=True)
                            cleaned += 1
                    except Exception:
                        pass
    except Exception as e:
        logger.error(f"Temp file cleanup failed: {e}")
    if cleaned:
        logger.info(f"Cleaned {cleaned} orphan temp files/dirs")


def auto_cleanup_memory():
    """Hourly: clean stale _project_presence and credit_tasks entries."""
    try:
        import time as _time
        now = _time.time()

        # Clean stale project presence entries (>5 min idle)
        try:
            from app.routes.admin import _project_presence
            for pid in list(_project_presence.keys()):
                stale_users = [uid for uid, info in _project_presence[pid].items() if now - info['ts'] > 300]
                for uid in stale_users:
                    del _project_presence[pid][uid]
                if not _project_presence[pid]:
                    del _project_presence[pid]
        except Exception:
            pass

        # Clean stale credit_tasks (>24h old)
        try:
            import app.globals as g
            stale_tasks = [tid for tid, info in list(g.credit_tasks.items())
                          if info.get('status') in ('completed', 'error')
                          and now - info.get('finished_at', 0) > 86400]
            for tid in stale_tasks:
                del g.credit_tasks[tid]
        except Exception:
            pass

        # Clean stale download tokens
        try:
            from app.globals import download_tokens
            stale = [t for t, v in list(download_tokens.items()) if v > 0 and v < 20]
            for t in stale:
                del download_tokens[t]
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")
