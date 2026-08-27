"""Route registration — all blueprints eager-loaded at startup."""
import logging

logger = logging.getLogger(__name__)


def register_all(flask_app):
    """Register bootstrap routes + all Blueprints at startup."""
    import os
    from flask import render_template, request, session, send_from_directory
    from flask_wtf.csrf import generate_csrf
    from app.utils.helpers import ok

    logger.info("Boot-routing (no ML imports yet)...")

    # ── Bootstrap endpoints (no Blueprint needed) ──
    @flask_app.route('/')
    def index():
        from flask import session
        consent_given = session.get('consent_value', 0) == 1
        chat_history = session.get('chat_history', [])
        return render_template('index.html', consent_given=consent_given, chat_history=chat_history)

    @flask_app.route('/get_csrf_token')
    def get_csrf_token_route():
        return ok({"csrf_token": generate_csrf()})

    @flask_app.route('/favicon.ico')
    def favicon():
        path = os.path.join(str(flask_app.static_folder), 'favicon.ico')
        if not os.path.exists(path):
            from flask import Response
            return Response(status=204)
        return send_from_directory(str(flask_app.static_folder), 'favicon.ico',
                                   mimetype='image/vnd.microsoft.icon')

    @flask_app.route('/.well-known/appspecific/com.chrome.devtools.json')
    def chrome_devtools_discovery():
        # Chrome/Edge DevTools protocol probe — silence with 204 instead of 404 noise
        from flask import Response
        return Response(status=204)

    @flask_app.route('/health')
    def health():
        try:
            from app.database import get_db_connection
            with get_db_connection() as conn:
                conn.cursor().execute("SELECT 1")
            return {"status": "healthy", "database": "connected"}, 200
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)[:200]}, 503

    # /check_auth is handled by auth_bp (eager-loaded below) —
    # it includes DB-fallback logic for recovering session state

    @flask_app.route('/sw.js')
    def service_worker():
        from app.config import BASE_DIR
        return send_from_directory(str(BASE_DIR / 'static'), 'sw.js',
                                   mimetype='application/javascript')

    # ── Core Blueprints: eager loading (chat + auth, ~8s on this laptop) ──
    import time
    t0 = time.time()
    from app.routes.chat import chat_bp
    flask_app.register_blueprint(chat_bp)
    logger.info(f"  OK chat_bp ({time.time()-t0:.0f}s)")
    t0 = time.time()
    from app.routes.auth import auth_bp
    flask_app.register_blueprint(auth_bp)
    logger.info(f"  OK auth_bp ({time.time()-t0:.0f}s)")

    # ── Admin Blueprint: eager (needed early for frontend) ──
    t0 = time.time()
    from app.routes.admin import admin_bp
    flask_app.register_blueprint(admin_bp)
    logger.info(f"  OK admin_bp ({time.time()-t0:.0f}s)")

    # ── Knowledge + Projects: also eager (needed for tabs) ──
    t0 = time.time()
    from app.routes.knowledge import knowledge_bp
    flask_app.register_blueprint(knowledge_bp)
    logger.info(f"  OK knowledge_bp ({time.time()-t0:.0f}s)")
    t0 = time.time()
    from app.routes.projects import projects_bp
    flask_app.register_blueprint(projects_bp)
    logger.info(f"  OK projects_bp ({time.time()-t0:.0f}s)")

    # ── Tasks: eager (lightweight, needed for async progress sidebar) ──
    t0 = time.time()
    from app.routes.tasks import tasks_bp
    flask_app.register_blueprint(tasks_bp)
    logger.info(f"  OK tasks_bp ({time.time()-t0:.0f}s)")

    # ── Batch: eager (Flask locks blueprint registration after first request)
    t0 = time.time()
    from app.routes.batch import batch_bp
    flask_app.register_blueprint(batch_bp)
    logger.info(f"  OK batch_bp ({time.time()-t0:.0f}s)")

    # ── Compliance: eager (lightweight, needed for bid compliance checks) ──
    t0 = time.time()
    from app.routes.compliance import compliance_bp
    flask_app.register_blueprint(compliance_bp)
    logger.info(f"  OK compliance_bp ({time.time()-t0:.0f}s)")

    # ── Templates: eager (bid template CRUD + .docx import) ──
    t0 = time.time()
    from app.routes.templates import templates_bp
    flask_app.register_blueprint(templates_bp)
    logger.info(f"  OK templates_bp ({time.time()-t0:.0f}s)")

    # ── Wiki: eager (needed for wiki tab) ──
    t0 = time.time()
    from app.routes.wiki import wiki_bp
    flask_app.register_blueprint(wiki_bp)
    logger.info(f"  OK wiki_bp ({time.time()-t0:.0f}s)")

    # ── Timeline: eager (needed for bidding timeline tab) ──
    t0 = time.time()
    from app.routes.timeline import timeline_bp
    flask_app.register_blueprint(timeline_bp)
    logger.info(f"  OK timeline_bp ({time.time()-t0:.0f}s)")

    # ── Cases: eager (case library from audit findings) ──
    t0 = time.time()
    from app.routes.cases import cases_bp
    flask_app.register_blueprint(cases_bp)
    logger.info(f"  OK cases_bp ({time.time()-t0:.0f}s)")

    # ── Document Analysis: eager (deep analysis of bidding documents) ──
    t0 = time.time()
    from app.routes.document_analysis import document_analysis_bp
    flask_app.register_blueprint(document_analysis_bp)
    logger.info(f"  OK document_analysis_bp ({time.time()-t0:.0f}s)")

    # ── Clearance: eager (unified 清标 entry, merges compare/analysis/compliance/AI review) ──
    t0 = time.time()
    from app.routes.clearance import clearance_bp
    flask_app.register_blueprint(clearance_bp)
    logger.info(f"  OK clearance_bp ({time.time()-t0:.0f}s)")

    # ── Upload: eager (stream large-file uploads to disk) ──
    t0 = time.time()
    from app.routes.upload import upload_bp
    flask_app.register_blueprint(upload_bp)
    logger.info(f"  OK upload_bp ({time.time()-t0:.0f}s)")

    # ── Graph: eager (spider-web knowledge graphs) ──
    t0 = time.time()
    from app.routes.graph import graph_bp
    flask_app.register_blueprint(graph_bp)
    logger.info(f"  OK graph_bp ({time.time()-t0:.0f}s)")

    # ── Credit Check: eager (enterprise credit checking) ──
    t0 = time.time()
    from app.routes.credit import credit_bp
    flask_app.register_blueprint(credit_bp)
    logger.info(f"  OK credit_bp ({time.time()-t0:.0f}s)")

    # ── All blueprints loaded eagerly — no lazy-load needed ──

    logger.info("App ready (all blueprints eager).")
