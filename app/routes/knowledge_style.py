"""Writing-style routes for the knowledge blueprint family.

Registered on the shared ``knowledge_bp`` Blueprint object from
app/routes/knowledge.py. Covers /my_writing_style and /admin/user_styles.
"""
from flask import request, jsonify, session

from app.routes.knowledge import knowledge_bp


@knowledge_bp.route('/my_writing_style', methods=['GET'])
def my_writing_style():
    """Get the current user's writing style profile."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Login required"}), 401
    from app.services.style_engine import get_user_style
    return jsonify({"status": "ok", "style": get_user_style(user_id)})


@knowledge_bp.route('/my_writing_style', methods=['POST'])
def update_my_writing_style():
    """Update the current user's style preferences (label, description)."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Login required"}), 401
    data = request.get_json(silent=True) or {}
    from app.services.style_engine import update_user_style
    result = update_user_style(user_id, data)
    return jsonify({"status": "ok", "style": result})


@knowledge_bp.route('/my_writing_style/analyze', methods=['POST'])
def analyze_my_writing_style():
    """Trigger style analysis for the current user."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Login required"}), 401
    from app.services.style_engine import analyze_user_style
    result = analyze_user_style(user_id)
    return jsonify({"status": "ok", "style": result})


# ── Admin Style Management ──

@knowledge_bp.route('/admin/user_styles', methods=['GET'])
def admin_user_styles():
    """List all user style profiles (admin only)."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.style_engine import get_all_style_profiles
    return jsonify({"status": "ok", "styles": get_all_style_profiles()})


@knowledge_bp.route('/admin/user_styles/<user_id>', methods=['GET'])
def admin_user_style_detail(user_id):
    """Get a specific user's style profile (admin only)."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.style_engine import get_user_style
    return jsonify({"status": "ok", "style": get_user_style(user_id)})


@knowledge_bp.route('/admin/user_styles/<user_id>', methods=['POST'])
def admin_user_style_update(user_id):
    """Admin edit a user's style profile."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json(silent=True) or {}
    from app.services.style_engine import update_user_style
    result = update_user_style(user_id, data)
    return jsonify({"status": "ok", "style": result})


@knowledge_bp.route('/admin/user_styles/<user_id>/analyze', methods=['POST'])
def admin_user_style_analyze(user_id):
    """Admin trigger style analysis for a specific user."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.style_engine import analyze_user_style
    result = analyze_user_style(user_id)
    return jsonify({"status": "ok", "style": result})


@knowledge_bp.route('/admin/user_styles/<user_id>/delete', methods=['POST'])
def admin_user_style_delete(user_id):
    """Admin delete a user's style profile."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.style_engine import delete_user_style
    from app.services.admin_utils import log_admin_action
    if delete_user_style(user_id):
        log_admin_action(session.get('user_id', ''), session.get('username', ''),
                        'STYLE_DELETE', 'users', user_id,
                        column_name='writing_style', new_value='deleted')
        return jsonify({"status": "ok", "message": "Style profile deleted"})
    return jsonify({"error": "Profile not found"}), 404


@knowledge_bp.route('/admin/user_styles/analyze_all', methods=['POST'])
def admin_user_styles_analyze_all():
    """Admin batch analyze all users' writing styles."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    import asyncio
    from app.database import get_db_connection
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT user_id FROM chat_sessions WHERE user_id IS NOT NULL")
            user_ids = [r[0] for r in cur.fetchall()]
    from app.services.style_engine import analyze_user_style
    results = []
    for uid in user_ids:
        profile = analyze_user_style(uid)
        results.append({'user_id': uid, 'style_label': profile.get('style_label', 'N/A'),
                       'message_count': profile.get('message_count', 0)})
    return jsonify({"status": "ok", "analyzed": len(results), "results": results})
