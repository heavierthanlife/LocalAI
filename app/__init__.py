"""
AI_Services - Modular Flask Application Factory.

Usage:
    from app import create_app
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
"""
import os
import sys
import logging
import atexit
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

# Load env before anything else
load_dotenv()

# Package config
from . import config
from . import globals as g
from . import database as db

# ============================================================
# Environment Validation
# ============================================================

def _validate_env():
    """Check all required environment variables are set. Raises RuntimeError if missing."""
    required = {
        ('SECRET_KEY', 'FLASK_SECRET_KEY'): 'Flask secret key for sessions and CSRF',
    }
    missing = []
    for keys, desc in required.items():
        if isinstance(keys, tuple):
            if not any(os.getenv(k) for k in keys):
                missing.append(f'  {keys[0]} — {desc}')
        elif not os.getenv(keys):
            missing.append(f'  {keys} — {desc}')
    if missing:
        msg = 'Missing required environment variables:\n' + '\n'.join(missing)
        msg += '\n\nCopy .env.example to .env and fill in the values.'
        raise RuntimeError(msg)

# ============================================================
# App Factory
# ============================================================

def create_app():
    """Create and configure the Flask application."""
    from flask import Flask
    from flask_session import Session
    from flask_wtf.csrf import CSRFProtect, generate_csrf
    from werkzeug.security import generate_password_hash, check_password_hash

    app = Flask(__name__, template_folder=str(config.BASE_DIR / 'templates'),
                static_folder=str(config.BASE_DIR / 'static'))

    # Validate required environment variables
    _validate_env()

    # Session config
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = str(config.SESSION_DIR)
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = True

    secret_key = os.getenv('SECRET_KEY') or os.getenv('FLASK_SECRET_KEY')
    if not secret_key:
        raise RuntimeError(
            "SECRET_KEY or FLASK_SECRET_KEY is not set. Set it in .env or environment."
        )
    app.config['SECRET_KEY'] = secret_key
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
    Session(app)

    # CSRF — opt-in only (all API routes use JSON, safe from CSRF via CORS + Content-Type)
    app.config['WTF_CSRF_CHECK_DEFAULT'] = False
    csrf = CSRFProtect()
    csrf.init_app(app)

    # Admin password — auto-generate from ADMIN_PIN if not explicitly set
    ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")
    if not ADMIN_PASSWORD_HASH:
        from werkzeug.security import generate_password_hash
        admin_pin = os.getenv("ADMIN_PIN", "123456")
        ADMIN_PASSWORD_HASH = generate_password_hash(admin_pin)
    app.config['ADMIN_PASSWORD_HASH'] = ADMIN_PASSWORD_HASH

    # Session lifetime
    app.config['SESSION_PERMANENT'] = True
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

    # Pass config and db to app ctx
    app.config['BASE_DIR'] = config.BASE_DIR
    app.config['DATA_DIR'] = config.DATA_DIR
    app.config['ALLOWED_EXTENSIONS'] = config.ALLOWED_EXTENSIONS

    # ── Swagger/OpenAPI docs (available at /apidocs) ──
    try:
        from flasgger import Swagger
        Swagger(app, template={
            "swagger": "2.0",
            "info": {
                "title": "中联招标智能助手 API",
                "description": "AI-powered bidding agency platform — REST API documentation",
                "version": "2026.06",
            },
            "securityDefinitions": {
                "Bearer": {"type": "apiKey", "name": "Authorization", "in": "header"},
                "Session": {"type": "apiKey", "name": "Cookie", "in": "header"},
            },
        })
    except ImportError:
        pass  # flasgger optional

    # Register routes from the monolithic app module
    from . import routes
    routes.register_all(app)

    # Setup scheduler (only in main process)
    _setup_scheduler(app)

    # Cleanup hooks
    atexit.register(_cleanup)

    return app


def _setup_scheduler(app):
    """Setup background scheduler for periodic tasks.
    
    Gated by ENABLE_SCHEDULER env var (default: true).
    Set to 'false' when running multiple workers to avoid duplicate jobs.
    """
    if os.environ.get('ENABLE_SCHEDULER', 'true').lower() != 'true':
        logging.getLogger(__name__).info("Scheduler disabled (ENABLE_SCHEDULER != true)")
        return

    from apscheduler.schedulers.background import BackgroundScheduler
    from . import cleanup_tasks

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=cleanup_tasks.cleanup_old_sessions, trigger="interval", days=1, args=[15])
    scheduler.add_job(func=cleanup_tasks.delete_expired_original_files, trigger="interval", hours=6)
    scheduler.add_job(func=cleanup_tasks.cleanup_stale_tasks, trigger="interval", minutes=5)
    scheduler.add_job(func=cleanup_tasks.cleanup_stale_message_responses, trigger="interval", hours=1)
    scheduler.add_job(func=cleanup_tasks.cleanup_old_anon_temp_files, trigger="interval", days=1, args=[1])
    scheduler.add_job(func=cleanup_tasks.schedule_project_deletion_cleanup, trigger="interval", days=1)
    scheduler.add_job(func=cleanup_tasks.cleanup_expired_recycle_bin, trigger="interval", days=3)
    scheduler.add_job(func=cleanup_tasks.cleanup_expired_share_files, trigger="interval", days=1, args=[7])
    scheduler.add_job(func=cleanup_tasks.cleanup_stale_download_tokens, trigger="interval", hours=6, args=[24])
    scheduler.add_job(func=cleanup_tasks.cleanup_orphan_users, trigger="interval", days=3)
    # Training data lifecycle — quarterly cleanup (raw + exports)
    scheduler.add_job(func=cleanup_tasks.cleanup_old_training_data,
                      trigger="cron", month="1,4,7,10", day="1", hour=4, minute=0,
                      id="quarterly_training_cleanup")
    scheduler.add_job(func=cleanup_tasks.cleanup_old_training_exports,
                      trigger="cron", month="1,4,7,10", day="1", hour=4, minute=30,
                      id="quarterly_export_cleanup")
    # ── Auto-workflow tasks ──
    scheduler.add_job(func=cleanup_tasks.auto_generate_weekly_report,
                      trigger="cron", day_of_week="mon", hour=2, minute=0,
                      id="auto_weekly_report")
    scheduler.add_job(func=cleanup_tasks.auto_generate_monthly_report,
                      trigger="cron", day="1", hour=2, minute=30,
                      id="auto_monthly_report")
    scheduler.add_job(func=cleanup_tasks.auto_generate_annual_report,
                      trigger="cron", month="1", day="1", hour=3, minute=0,
                      id="auto_annual_report")
    scheduler.add_job(func=cleanup_tasks.auto_rag_health_check,
                      trigger="cron", day_of_week="sun", hour=3, minute=0,
                      id="auto_rag_health")
    # Daily: archive+delete stale chat sessions
    scheduler.add_job(func=cleanup_tasks.auto_cleanup_stale_sessions, trigger="interval", hours=24,
                      id="auto_cleanup_sessions")
    # Daily: cleanup orphan batch compare temps + stale ingest dirs
    scheduler.add_job(func=cleanup_tasks.auto_cleanup_temp_files, trigger="interval", hours=24,
                      id="auto_cleanup_temps")
    # Hourly: clean memory leaks (presence, credit_tasks, download_tokens)
    scheduler.add_job(func=cleanup_tasks.auto_cleanup_memory, trigger="interval", hours=1,
                      id="auto_cleanup_memory")
    # Weekly training data health check
    scheduler.add_job(func=cleanup_tasks.auto_training_health_check,
                      trigger="cron", day_of_week="sun", hour=3, minute=30,
                      id="auto_training_health")
    # Daily stale review check + cleanup
    scheduler.add_job(func=cleanup_tasks.auto_cleanup_stale_reviews,
                      trigger="interval", hours=24,
                      id="auto_stale_review_cleanup")
    # Weekly skill audit: analyze duplicates, unused, promote candidates
    try:
        from app.services.skill_auditor import analyze_all_skills as weekly_skill_audit
        def run_skill_audit():
            import logging
            log = logging.getLogger(__name__)
            try:
                from app.database import get_db_connection
                with get_db_connection() as conn:
                    result = weekly_skill_audit(conn)
                log.info(f"Weekly skill audit: {result['total_skills']} skills, {result['duplicate_pairs']} duplicates, {result['unused_count']} unused, {len(result['promote_candidates'])} promote")
            except Exception as e:
                log.warning(f"Weekly skill audit skipped: {e}")
        scheduler.add_job(func=run_skill_audit, trigger="interval", days=7)
    except ImportError:
        pass
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())


def _cleanup():
    """Cleanup on shutdown."""
    from .globals import _agent

    # Shutdown agent
    if _agent is not None:
        _agent = None

    # Close DB pool
    if db.db_pool:
        db.db_pool.closeall()

    logging.getLogger(__name__).info("Application shutdown complete.")


def init_services():
    """Initialize background services. Called in run.py before app.run."""
    import logging
    log = logging.getLogger(__name__)

    log.info("Initializing services (fast-path)...")

    # Step 1: Database tables (non-blocking)
    log.info("  [1/4] Initializing PostgreSQL tables...")
    db.init_postgres_tables()

    # Step 2: EdgeDriver — skip at boot, load on first use
    log.info("  [2/4] Edge WebDriver deferred to first use.")

    # Step 3: ChromeDriver — skip at boot, load on first use
    log.info("  [3/4] ChromeDriver deferred to first use.")

    # Step 4: Async checkpointer
    log.info("  [4/4] Initializing async checkpointer...")
    _init_async_checkpointer()

    log.info("Services initialization complete (drivers lazy-loaded).")


def _init_async_checkpointer():
    """Initialize the SQLite checkpointer for LangGraph agent."""
    from .globals import _async_checkpointer
    from .config import DATA_DIR, logger
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = str(DATA_DIR / "checkpoints.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    _async_checkpointer = SqliteSaver(conn)
    logger.info("SqliteSaver initialized.")
