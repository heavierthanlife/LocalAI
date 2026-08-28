"""LLM-configuration and misc chat routes for the chat blueprint family.

Registered on the shared ``chat_bp`` Blueprint object from
app/routes/chat.py. Covers /set_max_tokens, /llm_providers, /llm_providers/set,
/feedback, /get_recent_files, and /load_cached_file.
"""
import os
import re

from flask import request, jsonify, session

from app.database import get_db_connection
from app.utils.helpers import ok, err, utc_now
import app.globals as g
from app.services.session_manager import get_user_id
from app.services.file_cache import file_cache_manager, add_to_cache
from app.services.anonymous import get_anon_temp_dir
from app.routes.chat import chat_bp, BEIJING_TZ

from psycopg2.extras import RealDictCursor


@chat_bp.route('/set_max_tokens', methods=['POST'])
def set_max_tokens():
    data = request.get_json()
    tokens = data.get('max_tokens', 4800)
    tokens = max(100, min(4800, tokens))
    session['max_tokens'] = tokens
    with g._agent_lock:
        g._agent = None
    return jsonify({"success": True, "max_tokens": tokens})

# ── LLM Provider / Model selection ──

@chat_bp.route('/llm_providers', methods=['GET'])
def list_llm_providers():
    """Return available LLM providers and the currently active one."""
    try:
        from app.services.llm_provider import get_available_providers, get_active_provider
    except ImportError:
        return jsonify({"available": [], "active": None, "error": "llm_provider module not loaded"})
    active = get_active_provider()
    return jsonify({
        "available": get_available_providers(),
        "active": active,
    })

@chat_bp.route('/llm_providers/set', methods=['POST'])
def set_llm_provider():
    """Set the active LLM provider and model via session."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Login required"}), 403
    data = request.get_json()
    provider = data.get('provider', '').strip()
    model = data.get('model', '').strip()
    if provider:
        session['llm_provider'] = provider
    if model:
        session['llm_model'] = model
    # Invalidate agent cache so next request picks up new model
    with g._agent_lock:
        g._agent = None
    return jsonify({"success": True, "provider": provider, "model": model})

@chat_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Feedback not allowed – no consent"}), 403
    data = request.get_json()
    rating = data.get('rating')
    comment = data.get('comment', '')
    user_message = data.get('user_message')
    assistant_response = data.get('assistant_response')
    if not user_message or not assistant_response:
        user_message = session.get('last_user_msg', '')
        assistant_response = session.get('last_assistant_msg', '')
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (thread_id, user_message, assistant_response, rating, comment, timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
                (session['thread_id'], user_message, assistant_response, rating, comment, utc_now())
            )
            conn.commit()
    # Log to training data pipeline
    try:
        from app.services.training_logger import log_interaction
        log_interaction(session['thread_id'], user_message, assistant_response,
                       rating=rating, rating_comment=comment)
    except Exception:
        pass
    return jsonify({"status": "ok"})

@chat_bp.route('/get_recent_files', methods=['GET'])
def get_recent_files():
    thread_id = session.get('thread_id')
    if not thread_id:
        return jsonify({"recent_files": []})
    recent = file_cache_manager.get_recent_with_lock(thread_id)
    files_with_usage = []
    if session.get('consent_value', 0) == 1:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for filename in recent:
                    cur.execute(
                        """SELECT usage_type, question, timestamp
                           FROM file_usage
                           WHERE thread_id = %s
                             AND filename = %s
                           ORDER BY timestamp DESC
                           LIMIT 5""",
                        (thread_id, filename)
                    )
                    usage_records = []
                    for row in cur.fetchall():
                        ts_utc = row['timestamp']
                        if ts_utc:
                            ts_beijing = ts_utc.astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            ts_beijing = None
                        usage_records.append({
                            "type": row['usage_type'],
                            "question": row['question'],
                            "time": ts_beijing
                        })
                    files_with_usage.append({
                        "filename": filename,
                        "usage": usage_records
                    })
    else:
        for filename in recent:
            files_with_usage.append({"filename": filename, "usage": []})
    return jsonify({"recent_files": files_with_usage})

@chat_bp.route('/load_cached_file', methods=['POST'])
def load_cached_file():
    data = request.get_json()
    filename = data.get('filename')
    thread_id = session.get('thread_id')
    if not thread_id:
        return jsonify({"error": "Session expired"}), 401
    content = file_cache_manager.get_content(thread_id, filename)
    if content:
        return jsonify({"content": content})
    if session.get('consent_value', 0) != 1:
        user_id = get_user_id()
        temp_dir = get_anon_temp_dir(user_id)
        safe_name = re.sub(r'[^\w\-_\. ]', '_', filename) + '.txt'
        fpath = os.path.join(temp_dir, safe_name)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            add_to_cache(thread_id, filename, content, user_id)
            return jsonify({"content": content})
        else:
            return jsonify({"error": "File not found"}), 404
    user_id = get_user_id()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content FROM user_files WHERE user_id = %s AND filename = %s AND (expires_at IS NULL OR expires_at > NOW())",
                (user_id, filename)
            )
            row = cur.fetchone()
            if row:
                content = row[0] or ''
                add_to_cache(thread_id, filename, content, user_id)
                return jsonify({"content": content})
    return jsonify({"error": "File not found"}), 404
