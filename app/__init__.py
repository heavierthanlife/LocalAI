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
import asyncio
import threading
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

    # Session config
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = str(config.SESSION_DIR)
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = True

    secret_key = os.getenv('SECRET_KEY')
    if not secret_key:
        import secrets
        secret_key = secrets.token_hex(32)
        logging.getLogger(__name__).warning(
            "SECRET_KEY not set in environment. Using a random temporary key."
        )
    app.config['SECRET_KEY'] = secret_key
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
    Session(app)

    # CSRF
    csrf = CSRFProtect()
    csrf.init_app(app)

    # Admin password
    global ADMIN_PASSWORD_HASH
    ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")
    if not ADMIN_PASSWORD_HASH and os.getenv("ADMIN_PSWD"):
        ADMIN_PASSWORD_HASH = generate_password_hash(os.getenv("ADMIN_PSWD"))
        logging.getLogger(__name__).warning(
            "ADMIN_PASSWORD_HASH not set, using plaintext ADMIN_PSWD."
        )
    app.config['ADMIN_PASSWORD_HASH'] = ADMIN_PASSWORD_HASH

    # Session lifetime
    app.config['SESSION_PERMANENT'] = True
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

    # Pass config and db to app ctx
    app.config['BASE_DIR'] = config.BASE_DIR
    app.config['DATA_DIR'] = config.DATA_DIR
    app.config['ALLOWED_EXTENSIONS'] = config.ALLOWED_EXTENSIONS

    # Register routes from the monolithic app module
    from . import routes
    routes.register_all(app)

    # Setup scheduler (only in main process)
    _setup_scheduler(app)

    # Cleanup hooks
    atexit.register(_cleanup)

    return app


def _setup_scheduler(app):
    """Setup background scheduler for periodic tasks."""
    if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
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
        scheduler.add_job(func=cleanup_tasks.cleanup_orphan_users, trigger="interval", days=3)
        scheduler.start()
        atexit.register(lambda: scheduler.shutdown())


def _cleanup():
    """Cleanup on shutdown."""
    from .globals import _agent, _async_loop, _async_conn
    import time

    # Shutdown agent
    _agent_global = _agent
    if _agent_global is not None:
        _agent_global = None

    # Shutdown async checkpointer
    if _async_loop is not None and _async_loop.is_running():
        async def _shutdown():
            if _async_conn:
                await _async_conn.close()
            _async_loop.stop()
        asyncio.run_coroutine_threadsafe(_shutdown(), _async_loop)
        time.sleep(0.5)

    # Close DB pool
    if db.db_pool:
        db.db_pool.closeall()

    logging.getLogger(__name__).info("Application shutdown complete.")


def init_services():
    """Initialize background services. Called in run.py before app.run."""
    import logging
    log = logging.getLogger(__name__)

    db.init_postgres_tables()
    config.preinstall_edgedriver()

    # Try pre-installing ChromeDriver
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        ChromeDriverManager().install()
        log.info("ChromeDriver pre-installed successfully")
    except Exception as e:
        log.warning(f"Failed to pre-install ChromeDriver: {e}")

    # Init async checkpointer
    _init_async_checkpointer()

    # Load semantic model
    try:
        from .services.semantic import get_semantic_model
        get_semantic_model()
        log.info("Semantic model loaded successfully.")
    except Exception as e:
        log.warning(f"Failed to load semantic model: {e}")


def _init_async_checkpointer():
    """Initialize the async SQLite checkpointer for LangGraph agent."""
    from .globals import _async_loop, _async_checkpointer, _async_conn
    from .config import DATA_DIR, logger
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    _async_loop = asyncio.new_event_loop()

    async def _create():
        global _async_conn
        _async_conn = await aiosqlite.connect(str(DATA_DIR / "checkpoints.db"))
        return AsyncSqliteSaver(_async_conn)

    _async_checkpointer = _async_loop.run_until_complete(_create())

    def _run_loop():
        asyncio.set_event_loop(_async_loop)
        _async_loop.run_forever()

    t = threading.Thread(target=_run_loop, daemon=True)
    t.start()
    logger.info("AsyncSqliteSaver initialized.")
