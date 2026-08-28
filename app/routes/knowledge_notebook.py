"""Notebook routes for the knowledge blueprint family.

Registered on the shared ``knowledge_bp`` Blueprint object from
app/routes/knowledge.py. Keeps the monolith smaller while preserving exact URLs.
"""
from flask import request, jsonify, session

from app.routes.knowledge import knowledge_bp


@knowledge_bp.route('/notebook', methods=['GET'])
def notebook_list():
    """List current user's notes."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Login required"}), 401
    from app.services.notebook import list_notes
    return jsonify({"status": "ok", "notes": list_notes(user_id)})


@knowledge_bp.route('/notebook/<note_id>', methods=['GET'])
def notebook_get(note_id):
    """Get a single note."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Login required"}), 401
    from app.services.notebook import get_note
    note = get_note(user_id, note_id)
    if not note:
        return jsonify({"error": "Note not found"}), 404
    return jsonify({"status": "ok", "note": note})


@knowledge_bp.route('/notebook/<note_id>', methods=['POST'])
def notebook_save(note_id):
    """Create or update a note."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Login required"}), 401
    data = request.get_json(silent=True) or {}
    content = data.get('content', '').strip()
    if not content:
        return jsonify({"error": "Content required"}), 400
    from app.services.notebook import save_note
    result = save_note(user_id, note_id, content)
    return jsonify({"status": "ok", "note": result})


@knowledge_bp.route('/notebook/<note_id>', methods=['DELETE'])
def notebook_delete(note_id):
    """Delete a note."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Login required"}), 401
    from app.services.notebook import delete_note
    if delete_note(user_id, note_id):
        return jsonify({"status": "ok", "message": "Deleted"})
    return jsonify({"error": "Not found"}), 404


@knowledge_bp.route('/notebook/<note_id>/summarize', methods=['POST'])
def notebook_summarize(note_id):
    """AI-summarize a note."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Login required"}), 401
    from app.services.notebook import ai_summarize_note
    summary = ai_summarize_note(user_id, note_id)
    if summary:
        return jsonify({"status": "ok", "summary": summary})
    return jsonify({"error": "Summarization failed"}), 500


@knowledge_bp.route('/notebook/search', methods=['POST'])
def notebook_search():
    """Semantic search across user notes."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Login required"}), 401
    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"error": "Query required"}), 400
    from app.services.notebook import search_notes
    return jsonify({"status": "ok", "results": search_notes(user_id, query)})
