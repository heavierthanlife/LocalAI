"""Session-management routes for the chat blueprint family.

Registered on the shared ``chat_bp`` Blueprint object from
app/routes/chat.py. Covers /new_chat, /api/login, /get_sessions,
/load_session, /delete_session, /update_session_title, /archive_session,
/restore_session, /list_archived_sessions, and /regenerate.
"""
import hashlib
import json
import logging
import os
import uuid

from flask import request, jsonify, session

from app.database import get_db_connection, db_transaction
from app.utils.helpers import ok, err, split_thinking_answer
import app.globals as g
from app.services.session_manager import (
    get_user_id, get_or_create_session, get_session_messages, get_user_sessions,
    store_message, delete_session, archive_session, db_execute_readonly,
)
from app.services.anonymous import get_session_messages_anon
from app.services.file_cache import load_cache_from_db
from app.services.task_locking import cleanup_stale_tasks
from app.services.agent import get_agent
from app.services.redteam_agent import get_redteam_agent
from app.routes.chat import chat_bp, BEIJING_TZ

from psycopg2.extras import RealDictCursor


logger = logging.getLogger(__name__)


@chat_bp.route('/new_chat', methods=['POST'])
def new_chat():
    new_thread_id = str(uuid.uuid4())
    session['thread_id'] = new_thread_id
    session['chat_history'] = []
    get_or_create_session(new_thread_id)
    return jsonify({"thread_id": new_thread_id})

@chat_bp.route('/api/login', methods=['POST'])
def api_login():
    """JWT login for external API access (WeChat Enterprise, CLI tools, etc.).

    POST body: {"username": "...", "pin": "1234"}
    Returns: {"access_token": "eyJ...", "user_id": "...", "username": "...", "role": "..."}
    """
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get('username') or '').strip()
    pin = (data.get('pin') or '').strip()
    if not username or not pin:
        return jsonify({"error": "username and pin required"}), 400
    try:
        from app.services.auth_jwt import create_token
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id, username, role, pin_hash FROM users WHERE username = %s AND is_active = TRUE",
                    (username,))
                row = cur.fetchone()
                if not row:
                    return jsonify({"error": "Invalid credentials"}), 401
                user_id, uname, role, pin_hash = row
                import hashlib
                if hashlib.sha256(pin.encode()).hexdigest() != pin_hash:
                    return jsonify({"error": "Invalid credentials"}), 401
                token = create_token(user_id, uname, role)
                return jsonify({
                    "access_token": token,
                    "user_id": user_id,
                    "username": uname,
                    "role": role,
                    "expires_in_hours": 24,
                })
    except Exception as e:
        logger.error(f"API login failed: {e}")
        return jsonify({"error": "Login failed"}), 500

@chat_bp.route('/get_sessions', methods=['GET'])
def get_sessions():
    sessions = get_user_sessions()
    return jsonify({"sessions": sessions})

@chat_bp.route('/load_session/<thread_id>', methods=['GET'])
def load_session(thread_id):
    if session.get('consent_value', 0) != 1:
        messages = get_session_messages_anon(thread_id)
        session['thread_id'] = thread_id
        session['chat_history'] = messages
        return jsonify({"messages": messages, "thread_id": thread_id})

    user_sessions = get_user_sessions()
    if not any(s['thread_id'] == thread_id for s in user_sessions):
        return jsonify({"error": "Session not found"}), 404

    with get_db_connection() as conn:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                db_execute_readonly(cur)
                cur.execute(
                    "SELECT id, role, content, thinking, timestamp FROM chat_messages WHERE thread_id = %s ORDER BY id ASC",
                    (thread_id,)
                )
                rows = cur.fetchall()
                messages = []
                for row in rows:
                    ts_utc = row['timestamp']
                    ts_beijing = ts_utc.astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S') if ts_utc else None
                    messages.append({
                        "id": row['id'],
                        "role": row['role'],
                        "content": row['content'],
                        "thinking": row['thinking'],
                        "timestamp": ts_beijing
                    })
            session['thread_id'] = thread_id
            session['chat_history'] = messages
            user_id = get_user_id()
            load_cache_from_db(thread_id, user_id)
            # Include sessions so frontend can determine project chat context
            return jsonify({"messages": messages, "thread_id": thread_id, "sessions": user_sessions})
        except Exception as e:
            logger.error(f"load_session failed for {thread_id}: {e}", exc_info=True)
            return jsonify({"error": f"Failed to load session: {e}"}), 500

@chat_bp.route('/delete_session/<thread_id>', methods=['POST'])
def delete_session_route(thread_id):
    user_sessions = get_user_sessions()
    if not any(s['thread_id'] == thread_id for s in user_sessions):
        return jsonify({"error": "Session not found"}), 404
    # Block deletion of project chats — they're archived with the project
    target = next((s for s in user_sessions if s['thread_id'] == thread_id), None)
    if target and target.get('project_id'):
        return jsonify({"error": "项目对话不能单独删除，请通过归档/删除项目来管理"}), 403
    user_id = get_user_id()
    with g.user_task_lock:
        cleanup_stale_tasks()
        if user_id in g.user_active_tasks and g.user_active_tasks[user_id]['thread_id'] == thread_id:
            return jsonify({
                "error": "task_running",
                "message": "无法删除：该聊天正在进行资源密集型任务，请等待任务完成后再试。"
            }), 409
    try:
        archive_session(thread_id, user_id, reason="manual")
    except Exception as e:
        logger.error(f"Archive session failed for {thread_id}: {e}", exc_info=True)
    try:
        delete_session(thread_id)
        logger.info(f"Session {thread_id} deleted successfully for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to delete session {thread_id}: {e}", exc_info=True)
        return jsonify({"error": "删除失败，请稍后重试"}), 500
    new_thread_id = None
    if session.get('thread_id') == thread_id:
        new_thread_id = str(uuid.uuid4())
        session['thread_id'] = new_thread_id
        session['chat_history'] = []
        get_or_create_session(new_thread_id)
        load_cache_from_db(new_thread_id, get_user_id())
    return jsonify({
        "status": "ok",
        "new_thread_id": new_thread_id,
        "messages": []
    })

@chat_bp.route('/update_session_title', methods=['POST'])
def update_session_title():
    """Allow users to rename their chat sessions."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json(silent=True) or {}
    thread_id = data.get('thread_id', '').strip()
    title = data.get('title', '').strip()
    if not thread_id or not title:
        return jsonify({"error": "Missing thread_id or title"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET title = %s WHERE thread_id = %s AND user_id = %s",
                (title, thread_id, user_id)
            )
            if cur.rowcount == 0:
                return jsonify({"error": "Session not found or access denied"}), 404
    return jsonify({"status": "ok", "title": title})

@chat_bp.route('/archive_session/<thread_id>', methods=['POST'])
def archive_session_route(thread_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    try:
        archive_path = archive_session(thread_id, user_id, reason="manual")
        if archive_path:
            delete_session(thread_id)   # remove from active sessions
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Archive failed"}), 500
    except Exception as e:
        logger.error(f"Archive session error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@chat_bp.route('/restore_session/<thread_id>', methods=['POST'])
def restore_session_route(thread_id):
    """Restore an archived session back to active chat."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT archive_path FROM archived_sessions WHERE thread_id = %s AND user_id = %s", (thread_id, user_id))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Archived session not found"}), 404
            archive_file = os.path.join(row['archive_path'], f"{thread_id}_session.json")
            if not os.path.exists(archive_file):
                return jsonify({"error": "Archive file missing"}), 404
            try:
                with open(archive_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                return jsonify({"error": "Archive data corrupted"}), 500
            title = data.get('session', {}).get('title', 'Restored Chat')
            cur.execute("INSERT INTO chat_sessions (thread_id, user_id, title, created_at, updated_at) VALUES (%s, %s, %s, %s, NOW()) ON CONFLICT (thread_id) DO UPDATE SET updated_at = NOW()", (thread_id, user_id, title, data.get('session', {}).get('created_at')))
            for msg in data.get('messages', []):
                content = msg.get('content', '')
                thinking = msg.get('thinking', '')
                role = msg.get('role', 'user')
                store_message(thread_id, role, content, thinking if role == 'assistant' else None)
            cur.execute("DELETE FROM archived_sessions WHERE thread_id = %s", (thread_id,))
            conn.commit()
    return jsonify({"success": True, "title": title})


@chat_bp.route('/list_archived_sessions', methods=['GET'])
def list_archived_sessions():
    """Admin only: list all archived sessions across all users."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    if session.get('role') != 'admin':
        return jsonify({"error": "Admin only"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT a.thread_id, a.archived_at, a.user_id, u.username
                FROM archived_sessions a LEFT JOIN users u ON a.user_id = u.user_id
                ORDER BY a.archived_at DESC LIMIT 30""")
            sessions = cur.fetchall()
    return jsonify({"sessions": [{
        'thread_id': s['thread_id'],
        'archived_at': s['archived_at'].isoformat() if s['archived_at'] else None,
        'username': s.get('username', '?'),
    } for s in sessions]})

@chat_bp.route('/regenerate', methods=['POST'])
def regenerate():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    data = request.get_json()
    user_message = data.get('user_message')
    if not user_message:
        return jsonify({"error": "Missing user_message"}), 400
    thread_id = session['thread_id']
    get_or_create_session(thread_id)

    is_grilling = False
    try:
        from app.database import get_db_connection
        from psycopg2.extras import RealDictCursor
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT is_grilling FROM chat_sessions WHERE thread_id = %s", (thread_id,))
                row = cur.fetchone()
                if row and row.get('is_grilling'):
                    is_grilling = True
    except Exception as e:
        logger.warning(f"Failed to check is_grilling: {e}")

    agent = get_redteam_agent() if is_grilling else get_agent()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        response = agent.invoke({"messages": [{"role": "user", "content": user_message}]}, config)
    except Exception as e:
        logger.error(f"Regenerate invoke failed: {e}", exc_info=True)
        return jsonify({"error": "AI 服务暂时不可用"}), 500
    assistant_message = response["messages"][-1]
    raw_response = assistant_message.content
    reasoning = assistant_message.additional_kwargs.get('reasoning_content', '')
    if reasoning and reasoning.strip():
        thinking = reasoning.strip()
        answer = raw_response.strip() if raw_response else ''
    else:
        thinking, answer = split_thinking_answer(raw_response)
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute("""
                            DELETE
                            FROM chat_messages
                            WHERE id IN (SELECT id
                                         FROM chat_messages
                                         WHERE thread_id = %s
                                         ORDER BY timestamp DESC
                                         LIMIT 2)
                            """, (thread_id,))
                conn.commit()
    store_message(thread_id, 'user', user_message)
    store_message(thread_id, 'assistant', answer if answer else raw_response, thinking if thinking else "")
    new_messages = get_session_messages(thread_id)
    session['chat_history'] = new_messages
    return jsonify({
        "assistant_message": answer if answer else raw_response,
        "thinking": thinking if thinking else ""
    })
