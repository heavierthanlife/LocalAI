"""Celery application for background task processing.

Start worker:  celery -A celery_app worker -l info -c 2
Start beat:    celery -A celery_app beat -l info

The worker uses Redis as both broker and result backend.
Heavy tasks (OCR, skill extraction, RAG indexing) run asynchronously
so they never block the Flask HTTP request/response cycle.
"""
import os
from celery import Celery
from celery.schedules import crontab

# ── Redis URL ──
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# ── Celery app ──
celery = Celery(
    'local_ai',
    broker=REDIS_URL,
    backend=REDIS_URL,  # Store task results
    include=[
        'app.services.ingest_pipeline',
        'app.services.skill_auditor',
        'app.services.nightly_trainer',
        'cleanup_tasks',
    ]
)

# ── Celery config ──
celery.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,

    # Retry / timeout
    task_acks_late=True,  # Re-deliver if worker crashes mid-task
    task_reject_on_worker_lost=True,
    task_soft_time_limit=600,  # 10 min soft limit
    task_time_limit=900,       # 15 min hard limit

    # Result expiry (keep results for 24h so frontend can poll)
    result_expires=86400,

    # Beat schedule — replaces APScheduler time triggers
    beat_schedule={
        'cleanup-stale-sessions': {
            'task': 'cleanup_tasks.cleanup_stale_sessions',
            'schedule': 3600.0,  # every hour
        },
        'cleanup-temp-files': {
            'task': 'cleanup_tasks.cleanup_temp_files',
            'schedule': 3600.0,
        },
        'skill-audit-weekly': {
            'task': 'app.services.skill_auditor.run_skill_audit',
            'schedule': 604800.0,  # weekly (7 days)
        },
        'generate-weekly-report': {
            'task': 'cleanup_tasks.generate_weekly_report',
            'schedule': 604800.0,  # weekly
            'kwargs': {'period': 'weekly'},
        },
        'nightly-lora-training': {
            'task': 'app.services.nightly_trainer.run_nightly_training',
            'schedule': crontab(hour=2, minute=0),  # 2:00 AM daily (off-work hours)
        },
    },
)


# ── Flask app context helper ──
# Celery tasks that need DB access or Flask config can call this.
def init_flask_context():
    """Provide Flask app context inside a celery task (call once per worker)."""
    from app import create_app
    app = create_app()
    app.app_context().push()
    return app
