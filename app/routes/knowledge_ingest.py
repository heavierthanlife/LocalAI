"""Admin ingest-pipeline routes for the knowledge blueprint family.

Registered on the shared ``knowledge_bp`` Blueprint object from
app/routes/knowledge.py. Covers /admin/ingest/* (ZIP upload, domain review,
KB chunk review/approve/reject, stale status, workload, structured list).
"""
import uuid

from flask import request, jsonify, session

from app.routes.knowledge import knowledge_bp


# ── Batch Document Ingestion Pipeline ──

@knowledge_bp.route('/admin/ingest/upload', methods=['POST'])
def admin_ingest_upload():
    """Upload a ZIP of scanned pages, start ingestion pipeline."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    if not f.filename or not f.filename.lower().endswith('.zip'):
        return jsonify({"error": "Only .zip files accepted"}), 400

    targets = request.form.get('targets', 'domain,knowledge,skills').split(',')
    task_id = str(uuid.uuid4())[:12]

    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    f.save(tmp.name)
    tmp.close()

    from app.services.ingest_pipeline import start_ingestion_async
    start_ingestion_async(task_id, tmp.name, targets)
    return jsonify({"status": "ok", "task_id": task_id, "targets": targets,
                    "message": "Ingestion pipeline started"})


@knowledge_bp.route('/admin/ingest/status/<task_id>', methods=['GET'])
def admin_ingest_status(task_id):
    """Get ingestion pipeline progress."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.ingest_pipeline import get_task_status
    status = get_task_status(task_id)
    if not status:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({"status": "ok", "task": status})


# ── Domain Words Review ──

@knowledge_bp.route('/admin/ingest/domain_review', methods=['GET'])
def admin_ingest_domain_review():
    """Get pending domain word candidates for review (admin/auditor)."""
    from app.routes.admin import is_auditor_or_admin
    if not is_auditor_or_admin():
        return jsonify({"error": "Admin or Auditor required"}), 403
    from app.services.ingest_pipeline import get_domain_review_candidates
    return jsonify({"status": "ok", "candidates": get_domain_review_candidates()})


@knowledge_bp.route('/admin/ingest/domain_approve', methods=['POST'])
def admin_ingest_domain_approve():
    """Approve selected domain words (admin/auditor)."""
    from app.routes.admin import is_auditor_or_admin
    if not is_auditor_or_admin():
        return jsonify({"error": "Admin or Auditor required"}), 403
    data = request.get_json(silent=True) or {}
    words = data.get('words', [])
    if not words:
        return jsonify({"error": "No words provided"}), 400
    from app.services.ingest_pipeline import approve_domain_words
    from app.services.admin_utils import log_admin_action
    from app.services.review_logger import log_review_action
    count = approve_domain_words(words)
    log_admin_action(session.get('user_id', ''), session.get('username', ''),
                    'DOMAIN_APPROVE', 'system', None,
                    column_name='domain_words', new_value=f'approved:{count} words')
    log_review_action(session.get('user_id',''), session.get('username',''),
                     session.get('role',''), 'domain_approve', 'domain_dict', count)
    return jsonify({"status": "ok", "approved": count, "message": f"Approved {count} words"})


@knowledge_bp.route('/admin/ingest/domain_reject', methods=['POST'])
def admin_ingest_domain_reject():
    """Reject domain word candidates (admin/auditor)."""
    from app.routes.admin import is_auditor_or_admin
    if not is_auditor_or_admin():
        return jsonify({"error": "Admin or Auditor required"}), 403
    data = request.get_json(silent=True) or {}
    words = data.get('words', [])
    if not words:
        return jsonify({"error": "No words provided"}), 400
    from app.services.ingest_pipeline import reject_domain_words
    from app.services.review_logger import log_review_action
    count = reject_domain_words(words)
    log_review_action(session.get('user_id',''), session.get('username',''),
                     session.get('role',''), 'domain_reject', 'domain_dict', count)
    return jsonify({"status": "ok", "rejected": count, "message": f"Rejected {count} words"})


# ── KB Chunk Review ──

@knowledge_bp.route('/admin/ingest/kb_review/<task_id>', methods=['GET'])
def admin_ingest_kb_review(task_id):
    """Get sample chunks for KB review (admin/auditor)."""
    from app.routes.admin import is_auditor_or_admin
    if not is_auditor_or_admin():
        return jsonify({"error": "Admin or Auditor required"}), 403
    from app.services.ingest_pipeline import get_kb_review_sample
    sample = get_kb_review_sample(task_id)
    if not sample:
        return jsonify({"error": "Review not found"}), 404
    return jsonify({"status": "ok", "review": sample})


@knowledge_bp.route('/admin/ingest/kb_chunk/<task_id>/<int:chunk_index>', methods=['GET'])
def admin_ingest_kb_chunk(task_id, chunk_index):
    """Get a specific chunk for editing (admin/auditor)."""
    from app.routes.admin import is_auditor_or_admin
    if not is_auditor_or_admin():
        return jsonify({"error": "Admin or Auditor required"}), 403
    from app.services.ingest_pipeline import get_kb_review_chunk
    chunk = get_kb_review_chunk(task_id, chunk_index)
    if not chunk:
        return jsonify({"error": "Chunk not found"}), 404
    return jsonify({"status": "ok", "chunk": chunk})


@knowledge_bp.route('/admin/ingest/kb_chunk/<task_id>/<int:chunk_index>', methods=['POST'])
def admin_ingest_kb_chunk_update(task_id, chunk_index):
    """Correct a KB chunk (admin/auditor)."""
    from app.routes.admin import is_auditor_or_admin
    if not is_auditor_or_admin():
        return jsonify({"error": "Admin or Auditor required"}), 403
    data = request.get_json(silent=True) or {}
    new_text = data.get('text', '').strip()
    if not new_text:
        return jsonify({"error": "Text cannot be empty"}), 400
    from app.services.ingest_pipeline import update_kb_review_chunk
    from app.services.review_logger import log_review_action
    if update_kb_review_chunk(task_id, chunk_index, new_text):
        log_review_action(session.get('user_id',''), session.get('username',''),
                         session.get('role',''), 'kb_chunk_edit', task_id, 1)
        return jsonify({"status": "ok", "message": "Chunk updated"})
    return jsonify({"error": "Chunk not found"}), 404


@knowledge_bp.route('/admin/ingest/kb_approve/<task_id>', methods=['POST'])
def admin_ingest_kb_approve(task_id):
    """Approve KB review (admin/auditor)."""
    from app.routes.admin import is_auditor_or_admin
    if not is_auditor_or_admin():
        return jsonify({"error": "Admin or Auditor required"}), 403
    try:
        from app.services.ingest_pipeline import approve_kb_review
        from app.services.admin_utils import log_admin_action
        from app.services.review_logger import log_review_action
        count = approve_kb_review(task_id)
        log_admin_action(session.get('user_id', ''), session.get('username', ''),
                        'KB_APPROVE', 'system', None,
                        column_name='kb_ingest', new_value=f'approved:{count} chunks')
        log_review_action(session.get('user_id',''), session.get('username',''),
                         session.get('role',''), 'kb_approve', task_id, count)
        return jsonify({"status": "ok", "indexed": count, "message": f"Indexed {count} chunks to knowledge base"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/ingest/kb_reject/<task_id>/<int:chunk_index>', methods=['POST'])
def admin_ingest_kb_reject_chunk(task_id, chunk_index):
    """Reject a KB chunk (admin/auditor)."""
    from app.routes.admin import is_auditor_or_admin
    if not is_auditor_or_admin():
        return jsonify({"error": "Admin or Auditor required"}), 403
    from app.services.ingest_pipeline import reject_kb_chunk
    from app.services.review_logger import log_review_action
    if reject_kb_chunk(task_id, chunk_index):
        log_review_action(session.get('user_id',''), session.get('username',''),
                         session.get('role',''), 'kb_reject', task_id, 1)
        return jsonify({"status": "ok", "message": f"Chunk {chunk_index} rejected"})
    return jsonify({"error": "Chunk not found"}), 404


@knowledge_bp.route('/admin/ingest/stale_status', methods=['GET'])
def admin_ingest_stale_status():
    """Return stale review status for warning banner (admin/auditor)."""
    from app.routes.admin import is_auditor_or_admin
    if not is_auditor_or_admin():
        return jsonify({"error": "Admin or Auditor required"}), 403
    from app.services.ingest_pipeline import check_stale_reviews
    return jsonify({"status": "ok", "stale": check_stale_reviews()})


@knowledge_bp.route('/admin/ingest/review_workload', methods=['GET'])
def admin_ingest_review_workload():
    """Return reviewer workload stats."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.review_logger import get_reviewer_workload, get_review_log
    return jsonify({"status": "ok", "workload": get_reviewer_workload(),
                    "recent_log": get_review_log(30)})


@knowledge_bp.route('/admin/ingest/structured', methods=['GET'])
def admin_ingest_structured_list():
    """List all AI-extracted structured documents."""
    from app.routes.admin import is_auditor_or_admin
    if not is_auditor_or_admin():
        return jsonify({"error": "Admin or Auditor required"}), 403
    from app.services.ingest_pipeline import get_structured_documents
    return jsonify({"status": "ok", "documents": get_structured_documents()})


# ── Personal Notebook ──
