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
        'app.services.clearance_engine',
        'app.services.plagiarism_task',
        'app.cleanup_tasks',
        'app.services.wiki_ingest',
        'app.services.compliance_checker',
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
            'task': 'app.cleanup_tasks.auto_cleanup_stale_sessions',
            'schedule': 3600.0,
        },
        'cleanup-temp-files': {
            'task': 'app.cleanup_tasks.cleanup_old_anon_temp_files',
            'schedule': 3600.0,
        },
        'skill-audit-weekly': {
            'task': 'app.cleanup_tasks.auto_skill_audit_weekly',
            'schedule': 604800.0,
        },
        'skill-compile-weekly': {
            'task': 'app.cleanup_tasks.auto_skill_compile',
            'schedule': 604800.0,
        },
        'generate-weekly-report': {
            'task': 'app.cleanup_tasks.auto_generate_weekly_report',
            'schedule': crontab(day_of_week='mon', hour=2, minute=0),
        },
        'nightly-lora-training': {
            'task': 'app.services.nightly_trainer.run_nightly_training',
            'schedule': crontab(hour=2, minute=0),  # 2:00 AM daily (off-work hours)
        },
        'refresh-llm-catalog': {
            'task': 'app.cleanup_tasks.refresh_llm_catalog_task',
            'schedule': crontab(hour=2, minute=30),  # 2:30 AM daily
        },
        # ── FIX-061: mirror the APScheduler maintenance jobs so they also run in
        # Docker (ENABLE_SCHEDULER=false). Schedules/args mirror app/__init__.py. ──
        'cleanup-old-sessions': {
            'task': 'app.cleanup_tasks.cleanup_old_sessions',
            'schedule': crontab(hour=3, minute=0), 'args': (15,),
        },
        'delete-expired-original-files': {
            'task': 'app.cleanup_tasks.delete_expired_original_files',
            'schedule': crontab(minute=0, hour='*/6'),
        },
        'cleanup-stale-tasks': {
            'task': 'app.cleanup_tasks.cleanup_stale_tasks',
            'schedule': crontab(minute='*/5'),
        },
        'cleanup-stale-message-responses': {
            'task': 'app.cleanup_tasks.cleanup_stale_message_responses',
            'schedule': crontab(minute=0),
        },
        'schedule-project-deletion-cleanup': {
            'task': 'app.cleanup_tasks.schedule_project_deletion_cleanup',
            'schedule': crontab(hour=4, minute=0),
        },
        'cleanup-expired-recycle-bin': {
            'task': 'app.cleanup_tasks.cleanup_expired_recycle_bin',
            'schedule': crontab(hour=5, minute=0, day_of_month='*/3'),
        },
        'cleanup-expired-share-files': {
            'task': 'app.cleanup_tasks.cleanup_expired_share_files',
            'schedule': crontab(hour=6, minute=0), 'args': (7,),
        },
        'cleanup-stale-download-tokens': {
            'task': 'app.cleanup_tasks.cleanup_stale_download_tokens',
            'schedule': crontab(minute=0, hour='*/6'), 'args': (24,),
        },
        'cleanup-orphan-users': {
            'task': 'app.cleanup_tasks.cleanup_orphan_users',
            'schedule': crontab(hour=7, minute=0, day_of_month='*/3'),
        },
        'cleanup-old-training-data': {
            'task': 'app.cleanup_tasks.cleanup_old_training_data',
            'schedule': crontab(month_of_year='1,4,7,10', day_of_month=1, hour=4, minute=0),
        },
        'cleanup-old-training-exports': {
            'task': 'app.cleanup_tasks.cleanup_old_training_exports',
            'schedule': crontab(month_of_year='1,4,7,10', day_of_month=1, hour=4, minute=30),
        },
        'generate-monthly-report': {
            'task': 'app.cleanup_tasks.auto_generate_monthly_report',
            'schedule': crontab(day_of_month=1, hour=2, minute=30),
        },
        'generate-annual-report': {
            'task': 'app.cleanup_tasks.auto_generate_annual_report',
            'schedule': crontab(month_of_year=1, day_of_month=1, hour=3, minute=0),
        },
        'auto-rag-health-check': {
            'task': 'app.cleanup_tasks.auto_rag_health_check',
            'schedule': crontab(day_of_week='sun', hour=3, minute=0),
        },
        'auto-cleanup-temp-files': {
            'task': 'app.cleanup_tasks.auto_cleanup_temp_files',
            'schedule': crontab(hour=1, minute=0),
        },
        'auto-cleanup-memory': {
            'task': 'app.cleanup_tasks.auto_cleanup_memory',
            'schedule': crontab(minute=0),
        },
        'auto-training-health-check': {
            'task': 'app.cleanup_tasks.auto_training_health_check',
            'schedule': crontab(day_of_week='sun', hour=3, minute=30),
        },
        'auto-cleanup-stale-reviews': {
            'task': 'app.cleanup_tasks.auto_cleanup_stale_reviews',
            'schedule': crontab(hour=2, minute=0),
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
