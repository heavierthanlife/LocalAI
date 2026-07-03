"""Blueprint: admin routes (auto-extracted)."""
import os, json, uuid, time, logging, hashlib, io, re, zipfile, shutil, random
from datetime import datetime, timezone, timedelta
from functools import wraps
from io import BytesIO
from flask import Blueprint, request, jsonify, session, send_file, render_template, url_for, current_app
from werkzeug.datastructures import FileStorage

from app.config import BASE_DIR, DATA_DIR, TEMP_ROOT, TEMP_DIR, USER_FILES_ORIGINAL_ROOT, PROJECT_FILES_ROOT, logger
from app.database import get_db_connection, db_transaction
from app.utils.helpers import utc_now, beijing_now, safe_error_response, split_thinking_answer
import app.globals as g
from app.services.file_cache import file_cache_manager, add_to_cache, load_cache_from_db
from app.services.file_processing import extract_text_from_file

from psycopg2.extras import RealDictCursor
from psycopg2 import sql
from werkzeug.security import check_password_hash
from app.services.admin_utils import admin_rate_limiter, log_admin_action
from app.routes.auth import validate_table_column

# can_* helpers live in projects.py; imported lazily in functions to avoid circular import
def _can_manage_files(project_id, user_id):
    from app.routes.projects import can_manage_files as _fn
    return _fn(project_id, user_id)

def _can_edit_file(project_id, file_id, user_id):
    from app.routes.projects import can_edit_file as _cef
    return _cef(project_id, file_id, user_id)

def _can_move_file(project_id, file_id, user_id):
    from app.routes.projects import can_move_file as _cmf
    return _cmf(project_id, file_id, user_id)

def _can_edit_folder(project_id, folder_id, user_id):
    from app.routes.projects import can_edit_folder as _fn
    return _fn(project_id, folder_id, user_id)

def _can_manage_members(project_id, user_id):
    from app.routes.projects import can_manage_members as _fn
    return _fn(project_id, user_id)

def _can_access_project(project_id, user_id):
    from app.routes.projects import can_access_project as _fn
    return _fn(project_id, user_id)

admin_bp = Blueprint('admin', __name__, template_folder=str(BASE_DIR / 'templates'), static_folder=str(BASE_DIR / 'static'))

@admin_bp.route('/admin/task_deposit', methods=['GET'])
def get_task_deposit():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    is_admin_user = session.get('role') == 'admin'
    if not is_admin_user:
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT id,
                               original_user_id,
                               original_username,
                               project_id,
                               project_name,
                               item_type,
                               item_data,
                               stored_path,
                               transferred_to_user_id,
                               transferred_at,
                               created_at
                        FROM task_deposit_items
                        WHERE deleted_at IS NULL
                        ORDER BY created_at DESC
                        """)
            items = cur.fetchall()
            return jsonify({"items": items})

@admin_bp.route('/admin/task_deposit/transfer/<int:item_id>', methods=['POST'])
def transfer_task_deposit_item(item_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if session.get('role') != 'admin':
        return jsonify({"error": "Only admin can transfer deposit items"}), 403
    data = request.get_json()
    target_user_id = data.get('target_user_id')
    if not target_user_id:
        return jsonify({"error": "Missing target_user_id"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE user_id = %s", (target_user_id,))
            if not cur.fetchone():
                return jsonify({"error": "Target user not found"}), 404
            cur.execute("""
                        UPDATE task_deposit_items
                        SET transferred_to_user_id = %s,
                            transferred_at         = NOW()
                        WHERE id = %s
                          AND deleted_at IS NULL
                        RETURNING id, item_type, item_data, stored_path
                        """, (target_user_id, item_id))
            item = cur.fetchone()
            if not item:
                return jsonify({"error": "Item not found or already deleted"}), 404
            conn.commit()
            return jsonify({"success": True, "item": dict(item)})

# Permission helpers for projects
ROLE_HIERARCHY = {'admin': 4, 'manager': 3, 'editor': 2, 'viewer': 1, 'user': 0}

def get_role_level():
    role = session.get('role', 'user')
    return ROLE_HIERARCHY.get(role, 0)

def is_admin():
    return get_role_level() >= ROLE_HIERARCHY['admin']

def is_manager():
    return get_role_level() >= ROLE_HIERARCHY['manager']

def is_editor():
    return get_role_level() >= ROLE_HIERARCHY['editor']

def require_role(min_role):
    """Decorator: require minimum role level to access."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if get_role_level() < ROLE_HIERARCHY.get(min_role, 0):
                return jsonify({"error": f"Requires {min_role} role or higher"}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_admin():
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated_function


def is_auditor_or_admin():
    """Check if current session is admin or auditor (review permissions)."""
    if is_admin():
        return True
    return session.get('is_auditor', False)


def auditor_required(f):
    """Decorator: admin or auditor required (review/audit operations only)."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_auditor_or_admin():
            return jsonify({"error": "Admin or Auditor access required"}), 403
        return f(*args, **kwargs)
    return decorated_function

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('consent_value', 0) != 1:
            return jsonify({"error": "Consent not given"}), 403
        if not session.get('user_id'):
            return jsonify({"error": "Not logged in"}), 401
        return f(*args, **kwargs)
    return decorated_function

def get_user_role_in_project(project_id, user_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT role FROM project_members WHERE project_id = %s AND user_id = %s",
                        (project_id, user_id))
            row = cur.fetchone()
            return row[0] if row else None

@admin_bp.route('/admin/projects', methods=['POST'])
@admin_required
def create_project():
    data = request.get_json()
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()
    industry = data.get('industry', 'general').strip()
    manager_id = data.get('manager_id', '').strip()
    if industry not in ('bidding_agency', 'engineering_cost', 'engineering_audit', 'general'):
        industry = 'general'
    if not name:
        return jsonify({"error": "Project name required"}), 400
    user_id = session.get('user_id')
    import uuid
    chat_thread_id = str(uuid.uuid4())
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO projects (name, description, created_by, status, industry) VALUES (%s, %s, %s, 'active', %s) RETURNING id",
                    (name, description, user_id, industry))
                project_id = cur.fetchone()[0]
                cur.execute(
                    "INSERT INTO project_members (project_id, user_id, role, added_by) VALUES (%s, %s, 'admin', %s)",
                    (project_id, user_id, user_id))
                # Add manager if specified and different from creator
                if manager_id and manager_id != user_id:
                    cur.execute(
                        "INSERT INTO project_members (project_id, user_id, role, added_by) VALUES (%s, %s, 'manager', %s) ON CONFLICT DO NOTHING",
                        (project_id, manager_id, user_id))
                cur.execute(
                    "INSERT INTO project_folders (project_id, parent_folder_id, name, created_by) VALUES (%s, NULL, %s, %s)",
                    (project_id, name, user_id))
                # Auto-create shared project chat
                cur.execute(
                    "INSERT INTO chat_sessions (user_id, thread_id, title, project_id) VALUES (%s, %s, %s, %s)",
                    (user_id, chat_thread_id, name, project_id))
                conn.commit()
                return jsonify({"success": True, "id": project_id, "chat_thread_id": chat_thread_id})

@admin_bp.route('/admin/projects/<int:project_id>/backfill_chat', methods=['POST'])
def backfill_project_chat(project_id):
    """Ensure a shared project chat session exists for this project.
    
    Called by frontend when openProject() detects the project chat is missing
    from the sidebar (e.g. legacy projects created before auto-chat was added).
    """
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if project exists and user is a member
                cur.execute("""
                    SELECT p.id, p.name FROM projects p
                    JOIN project_members pm ON p.id = pm.project_id
                    WHERE p.id = %s AND pm.user_id = %s AND pm.status = 'active'
                """, (project_id, user_id))
                proj = cur.fetchone()
                if not proj:
                    return jsonify({"error": "Project not found or access denied"}), 404
                
                # Check if chat already exists
                cur.execute("""
                    SELECT thread_id FROM chat_sessions WHERE project_id = %s LIMIT 1
                """, (project_id,))
                existing = cur.fetchone()
                if existing:
                    return jsonify({"success": True, "thread_id": existing['thread_id'], "existed": True})
                
                # Create new shared chat
                import uuid
                thread_id = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO chat_sessions (user_id, thread_id, title, project_id) VALUES (%s, %s, %s, %s)",
                    (user_id, thread_id, proj['name'], project_id))
                conn.commit()
                logger.info(f"Backfilled project chat: {proj['name']} (project_id={project_id}, thread={thread_id})")
                return jsonify({"success": True, "thread_id": thread_id, "existed": False})
    except Exception as e:
        logger.error(f"Backfill project chat failed: {e}")
        return jsonify({"error": str(e)[:200]}), 500

@admin_bp.route('/admin/projects', methods=['GET'])
def get_projects():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"projects": [], "has_projects": False})
    if is_admin():
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, name, description, created_at, updated_at, status, archived_at, archive_filename, deletion_scheduled_at FROM projects ORDER BY CASE status WHEN 'active' THEN 1 WHEN 'archived' THEN 2 WHEN 'aborted' THEN 3 END, created_at DESC")
                projects = cur.fetchall()
                has_projects = len(projects) > 0
                return jsonify({"projects": projects, "has_projects": has_projects})
    else:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Active memberships first, then quitted (dimmed in frontend)
                cur.execute("""
                            SELECT p.id,
                                   p.name,
                                   p.description,
                                   p.created_at,
                                   p.updated_at,
                                   p.status,
                                   p.archived_at,
                                   p.deletion_scheduled_at,
                                   pm.role as member_role,
                                   pm.status as member_status
                            FROM projects p
                                     JOIN project_members pm ON p.id = pm.project_id
                            WHERE pm.user_id = %s
                            ORDER BY CASE pm.status WHEN 'active' THEN 1 WHEN 'quitted' THEN 2 END,
                                     CASE p.status WHEN 'active' THEN 1 WHEN 'archived' THEN 2 WHEN 'aborted' THEN 3 END,
                                     p.created_at DESC
                            """, (user_id,))
                projects = cur.fetchall()
                has_projects = len(projects) > 0
                return jsonify({"projects": projects, "has_projects": has_projects})

@admin_bp.route('/admin/projects/<int:project_id>', methods=['PUT'])
def update_project(project_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    if not is_admin() and not _can_manage_files(project_id, user_id):
        return jsonify({"error": "Permission denied"}), 403

    data = request.get_json()
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()
    if not name:
        return jsonify({"error": "Project name required"}), 400

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE projects
                SET name = %s, description = %s, updated_at = NOW()
                WHERE id = %s
                RETURNING id
            """, (name, description, project_id))
            if cur.fetchone():
                # Sync project chat title
                cur.execute(
                    "UPDATE chat_sessions SET title = %s WHERE project_id = %s",
                    (name, project_id)
                )
                conn.commit()
                return jsonify({"success": True})
            else:
                return jsonify({"error": "Project not found"}), 404

@admin_bp.route('/admin/projects/<int:project_id>', methods=['DELETE'])
@admin_required
def delete_project(project_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Project not found"}), 404
            status = row[0]
            if status not in ('archived', 'aborted'):
                return jsonify({"error": "Only archived or aborted projects can be deleted"}), 400

            # Archive project chat sessions before deletion (full content to disk JSON)
            from app.services.session_manager import archive_session
            cur.execute("SELECT thread_id, user_id, title FROM chat_sessions WHERE project_id = %s", (project_id,))
            project_chats = cur.fetchall()
            for thread_id, uid, title in project_chats:
                archive_session(thread_id, uid, reason=f"project_{project_id}_deleted")
            # Delete dependent records first (FK constraints)
            cur.execute("DELETE FROM message_quotes WHERE project_id = %s", (project_id,))
            cur.execute("DELETE FROM project_todos WHERE project_id = %s", (project_id,))
            cur.execute("DELETE FROM regen_vote_ballots WHERE vote_id IN (SELECT id FROM regen_votes WHERE project_id = %s)", (project_id,))
            cur.execute("DELETE FROM regen_votes WHERE project_id = %s", (project_id,))
            cur.execute("DELETE FROM project_ai_memory WHERE project_id = %s", (project_id,))
            cur.execute("DELETE FROM chat_messages WHERE thread_id IN (SELECT thread_id FROM chat_sessions WHERE project_id = %s)", (project_id,))
            cur.execute("DELETE FROM chat_sessions WHERE project_id = %s", (project_id,))
            cur.execute("DELETE FROM project_members WHERE project_id = %s", (project_id,))

            cur.execute("SELECT stored_path FROM project_files WHERE project_id = %s", (project_id,))
            for (stored_path,) in cur.fetchall():
                _safe_delete_file(stored_path, f'project_{project_id}_file')

            cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
            conn.commit()
            return jsonify({"success": True})

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>', methods=['DELETE'])
def delete_project_file(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not _can_edit_file(project_id, file_id, user_id):
        return jsonify({"error": "Permission denied"}), 403

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, original_name, file_size, stored_path, uploaded_by, folder_id, filename, version, file_hash, project_id
                    FROM project_files
                    WHERE id = %s AND project_id = %s
                """, (file_id, project_id))
                file_record = cur.fetchone()
                if not file_record:
                    return jsonify({"error": "File not found"}), 404

                cur.execute("""
                    INSERT INTO project_recycle_bin 
                    (original_table, original_id, project_id, folder_id, file_name, original_name, file_size, stored_path, file_hash, version, uploaded_by, deleted_at, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '3 days')
                """, (
                    'project_files', file_record['id'], file_record['project_id'], file_record['folder_id'],
                    file_record['original_name'],
                    file_record['original_name'],
                    file_record['file_size'], file_record['stored_path'], file_record['file_hash'],
                    file_record['version'], file_record['uploaded_by']
                ))

                cur.execute("DELETE FROM project_files WHERE id = %s AND project_id = %s", (file_id, project_id))
                conn.commit()
                return jsonify({"success": True, "moved_to_recycle_bin": True})

@admin_bp.route('/admin/projects/<int:project_id>/abort', methods=['POST'])
@admin_required
def abort_project(project_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE projects SET status = 'aborted', archived_at = NOW() WHERE id = %s RETURNING id",
                        (project_id,))
            if cur.fetchone():
                conn.commit()
                return jsonify({"success": True})
            return jsonify({"error": "Project not found"}), 404

@admin_bp.route('/admin/projects/<int:project_id>/finish', methods=['POST'])
def finish_project(project_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not _can_manage_files(project_id, user_id):
        return jsonify({"error": "Only admin or project manager can finish a project"}), 403
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM projects WHERE id = %s AND status = 'active'", (project_id,))
            project = cur.fetchone()
            if not project:
                return jsonify({"error": "Project not found or already finished/aborted"}), 404
            project_name = project[0]
            cur.execute("SELECT stored_path, original_name FROM project_files WHERE project_id = %s", (project_id,))
            files = cur.fetchall()
            if not files:
                return jsonify({"error": "No files to archive"}), 400
            zip_dir = os.path.join(PROJECT_FILES_ROOT, 'archives')
            os.makedirs(zip_dir, exist_ok=True)
            safe_name = re.sub(r'[^\w\-_\.]', '_', project_name)
            zip_filename = f"project_{project_id}_{safe_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = os.path.join(zip_dir, zip_filename)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for stored_path, original_name in files:
                    zipf.write(stored_path, original_name)
            cur.execute("UPDATE projects SET status = 'archived', archived_at = NOW(), archive_filename = %s WHERE id = %s",
                        (zip_filename, project_id))
            conn.commit()
            return jsonify({
                "success": True,
                "download_url": f"/admin/projects/{project_id}/download_archive/{zip_filename}",
                "zip_filename": zip_filename
            })

@admin_bp.route('/admin/projects/<int:project_id>/download_archive/<zip_filename>', methods=['GET'])
def download_archive(project_id, zip_filename):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    zip_dir = os.path.join(PROJECT_FILES_ROOT, 'archives')
    zip_path = os.path.join(zip_dir, zip_filename)
    if not os.path.exists(zip_path):
        return jsonify({"error": "Archive not found"}), 404
    return send_file(zip_path, as_attachment=True, download_name=zip_filename)

# Project members routes
@admin_bp.route('/admin/projects/<int:project_id>/members', methods=['GET'])
def get_project_members(project_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT u.user_id, u.username, pm.role, pm.added_at
                FROM project_members pm
                JOIN users u ON pm.user_id = u.user_id
                WHERE pm.project_id = %s AND u.username IS NOT NULL AND u.username != ''
                ORDER BY pm.role, u.username
            """, (project_id,))
            members = cur.fetchall()
            return jsonify({"members": members})

@admin_bp.route('/admin/projects/<int:project_id>/members/search', methods=['GET'])
def search_users_to_add(project_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_manage_members(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403

    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return jsonify({"users": []})
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT user_id, username
                FROM users
                WHERE username ILIKE %s
                  AND user_id NOT IN (SELECT user_id FROM project_members WHERE project_id = %s)
                  AND user_id != %s
                  AND role != 'admin'
                LIMIT 20
            """, (f'%{query}%', project_id, user_id))
            users = cur.fetchall()
            return jsonify({"users": users})

@admin_bp.route('/admin/users', methods=['GET'])
def list_users():
    """Return all active users (for searchable dropdowns, member pickers)."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    if not is_admin():
        return jsonify({"error": "Admin only"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT user_id, username, role
                FROM users
                WHERE is_active = TRUE AND username IS NOT NULL AND username != ''
                ORDER BY username
            """)
            return jsonify({"users": cur.fetchall()})

@admin_bp.route('/admin/projects/<int:project_id>/all_users', methods=['GET'])
def get_all_users_for_project(project_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    current_user_id = session.get('user_id')
    if not current_user_id:
        return jsonify({"error": "Not logged in"}), 401
    if not is_admin() and not _can_manage_members(project_id, current_user_id):
        return jsonify({"error": "Access denied"}), 403

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT user_id, username
                FROM users
                WHERE username IS NOT NULL AND username != ''
                  AND user_id NOT IN (SELECT user_id FROM project_members WHERE project_id = %s)
                  AND user_id != %s
                  AND role != 'admin'
                ORDER BY username
                LIMIT 100
            """, (project_id, current_user_id))
            users = cur.fetchall()
            return jsonify({"users": users})

@admin_bp.route('/admin/projects/<int:project_id>/members', methods=['POST'])
def add_project_member(project_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not _can_manage_members(project_id, user_id):
        return jsonify({"error": "Only admin or project manager can add members"}), 403

    data = request.get_json()
    new_user_id = data.get('user_id')
    role = data.get('role', 'member')
    if role == 'manager' and not is_admin():
        return jsonify({"error": "Only admin can add managers"}), 403
    if role not in ('member', 'manager'):
        return jsonify({"error": "Invalid role"}), 400

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT role FROM users WHERE user_id = %s", (new_user_id,))
            row = cur.fetchone()
            if row and row[0] == 'admin':
                return jsonify({"error": "Cannot add a global admin as a project member"}), 403

            if not row:
                return jsonify({"error": "User not found"}), 404

            cur.execute("SELECT 1 FROM project_members WHERE project_id = %s AND user_id = %s",
                        (project_id, new_user_id))
            if cur.fetchone():
                return jsonify({"error": "User already a member"}), 409

            cur.execute("""
                INSERT INTO project_members (project_id, user_id, role, added_by)
                VALUES (%s, %s, %s, %s)
            """, (project_id, new_user_id, role, user_id))
            # Auto-backfill project chat if none exists
            cur.execute("SELECT 1 FROM chat_sessions WHERE project_id = %s LIMIT 1", (project_id,))
            if not cur.fetchone():
                import uuid as _uuid
                cur.execute("SELECT name FROM projects WHERE id = %s", (project_id,))
                proj_row = cur.fetchone()
                proj_name = proj_row[0] if proj_row else f"项目#{project_id}"
                cur.execute(
                    "INSERT INTO chat_sessions (user_id, thread_id, title, project_id) VALUES (%s,%s,%s,%s)",
                    (user_id, f"proj_{project_id}_{_uuid.uuid4().hex[:8]}", proj_name, project_id)
                )
            conn.commit()
            return jsonify({"success": True})

@admin_bp.route('/admin/projects/<int:project_id>/members/<user_id>', methods=['PUT'])
@admin_required
def update_member_role(project_id, user_id):
    data = request.get_json()
    new_role = data.get('role')
    if new_role not in ('member', 'manager'):
        return jsonify({"error": "Invalid role"}), 400

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT role FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row and row[0] == 'admin':
                return jsonify({"error": "Cannot modify global admin's role"}), 403

            cur.execute("""
                UPDATE project_members
                SET role = %s
                WHERE project_id = %s AND user_id = %s
                RETURNING user_id
            """, (new_role, project_id, user_id))
            if cur.rowcount == 0:
                return jsonify({"error": "Member not found"}), 404
            conn.commit()
            return jsonify({"success": True})

@admin_bp.route('/admin/projects/<int:project_id>/members/<user_id>', methods=['DELETE'])
def remove_project_member(project_id, user_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    current_user_id = session.get('user_id')
    if not _can_manage_members(project_id, current_user_id):
        return jsonify({"error": "Only admin or project manager can remove members"}), 403

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT role FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row and row[0] == 'admin':
                return jsonify({"error": "Cannot remove a global admin"}), 403

            cur.execute("""
                SELECT role FROM project_members
                WHERE project_id = %s AND user_id = %s
            """, (project_id, user_id))
            target_member = cur.fetchone()
            if not target_member:
                return jsonify({"error": "Member not found"}), 404

            target_role = target_member[0]
            if target_role == 'admin':
                return jsonify({"error": "Cannot remove the project admin"}), 403
            if target_role == 'manager' and not is_admin():
                return jsonify({"error": "Only admin can remove managers"}), 403

            cur.execute("""
                UPDATE project_members SET status = 'quitted'
                WHERE project_id = %s AND user_id = %s
            """, (project_id, user_id))
            if cur.rowcount == 0:
                return jsonify({"error": "Member not found"}), 404
            conn.commit()
            return jsonify({"success": True, "quitted": True})
            conn.commit()
            return jsonify({"success": True})

@admin_bp.route('/admin/projects/<int:project_id>/transfer_manager/<user_id>', methods=['POST'])
def transfer_manager_role(project_id, user_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    current_user_id = session.get('user_id')
    current_role = get_user_role_in_project(project_id, current_user_id)
    if current_role != 'manager':
        return jsonify({"error": "Only a manager can transfer manager rights"}), 403
    target_role = get_user_role_in_project(project_id, user_id)
    if target_role != 'member':
        return jsonify({"error": "Target user must be a member"}), 400
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute("UPDATE project_members SET role = 'member' WHERE project_id = %s AND user_id = %s",
                            (project_id, current_user_id))
                cur.execute("UPDATE project_members SET role = 'manager' WHERE project_id = %s AND user_id = %s",
                            (project_id, user_id))
                conn.commit()
    return jsonify({"success": True})

# Project folders and files
def ensure_root_folder(project_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM project_folders WHERE project_id = %s AND parent_folder_id IS NULL",
                        (project_id,))
            if not cur.fetchone():
                cur.execute("SELECT name FROM projects WHERE id = %s", (project_id,))
                row = cur.fetchone()
                if row:
                    project_name = row[0]
                    cur.execute(
                        "INSERT INTO project_folders (project_id, parent_folder_id, name, created_by) VALUES (%s, NULL, %s, %s)",
                        (project_id, project_name, session.get('user_id')))
                    conn.commit()
                    logger.info(f"Created missing root folder for project {project_id}")

def build_folder_path(folder_id, folder_dict):
    parts = []
    current_id = folder_id
    while current_id:
        folder = folder_dict.get(current_id)
        if not folder:
            break
        parts.insert(0, folder['name'])
        current_id = folder['parent_folder_id']
    return '/' + '/'.join(parts) if parts else '/'

@admin_bp.route('/admin/projects/<int:project_id>/folders', methods=['GET'])
def get_folders(project_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    ensure_root_folder(project_id)
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, parent_folder_id, name FROM project_folders WHERE project_id = %s ORDER BY parent_folder_id, name",
                    (project_id,))
                folders = cur.fetchall()
                if not folders:
                    return jsonify({"folders": []})
                folder_dict = {f['id']: f for f in folders}
                for f in folder_dict.values():
                    f['children'] = []
                root_folders = []
                for f in folder_dict.values():
                    if f['parent_folder_id'] is None:
                        root_folders.append(f)
                    else:
                        parent = folder_dict.get(f['parent_folder_id'])
                        if parent:
                            parent['children'].append(f)
                        else:
                            root_folders.append(f)
                for f in folder_dict.values():
                    f['path'] = build_folder_path(f['id'], folder_dict)
                return jsonify({"folders": root_folders})
    except Exception as e:
        logger.error(f"Error in get_folders: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@admin_bp.route('/admin/projects/<int:project_id>/folders', methods=['POST'])
def create_folder(project_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    data = request.get_json()
    name = data.get('name', '').strip()
    parent_folder_id = data.get('parent_folder_id')
    if not name:
        return jsonify({"error": "Folder name required"}), 400
    if parent_folder_id is None:
        return jsonify({"error": "Cannot create root folder. Only one root folder exists per project."}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s",
                        (parent_folder_id, project_id))
            if not cur.fetchone():
                return jsonify({"error": "Parent folder not found"}), 404
            cur.execute(
                "INSERT INTO project_folders (project_id, parent_folder_id, name, created_by) VALUES (%s, %s, %s, %s) RETURNING id",
                (project_id, parent_folder_id, name, user_id))
            new_id = cur.fetchone()[0]
            conn.commit()
            return jsonify({"success": True, "id": new_id})

@admin_bp.route('/admin/projects/<int:project_id>/folders/<int:folder_id>', methods=['DELETE'])
def delete_folder(project_id, folder_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not _can_edit_folder(project_id, folder_id, user_id):
        return jsonify({"error": "Permission denied"}), 403

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    WITH RECURSIVE folder_tree AS (
                        SELECT id, name, parent_folder_id, created_at, created_by
                        FROM project_folders
                        WHERE id = %s AND project_id = %s
                        UNION ALL
                        SELECT pf.id, pf.name, pf.parent_folder_id, pf.created_at, pf.created_by
                        FROM project_folders pf
                        INNER JOIN folder_tree ft ON pf.parent_folder_id = ft.id
                    )
                    SELECT * FROM folder_tree
                """, (folder_id, project_id))
                folders = cur.fetchall()

                folder_ids = [f['id'] for f in folders]
                for f in folders:
                    cur.execute("""
                        INSERT INTO project_folders_recycle_bin
                        (original_id, project_id, name, parent_folder_id, original_parent_id, created_at, created_by, deleted_at, expires_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '3 days')
                    """, (
                        f['id'], project_id, f['name'], f['parent_folder_id'], f['parent_folder_id'],
                        f['created_at'], f['created_by']
                    ))

                if folder_ids:
                    placeholders = ','.join(['%s'] * len(folder_ids))
                    cur.execute(f"""
                        SELECT id, original_name, file_size, stored_path, file_hash, version, uploaded_by, folder_id
                        FROM project_files
                        WHERE project_id = %s AND folder_id IN ({placeholders})
                    """, [project_id] + folder_ids)
                    files = cur.fetchall()
                    for f in files:
                        cur.execute("""
                            INSERT INTO project_recycle_bin 
                            (original_table, original_id, project_id, folder_id, file_name, original_name, file_size, stored_path, file_hash, version, uploaded_by, deleted_at, expires_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '3 days')
                        """, (
                            'project_files', f['id'], project_id, f['folder_id'],
                            f['original_name'], f['original_name'], f['file_size'],
                            f['stored_path'], f['file_hash'], f['version'],
                            f['uploaded_by']
                        ))
                    cur.execute(f"""
                        DELETE FROM project_files
                        WHERE project_id = %s AND folder_id IN ({placeholders})
                    """, [project_id] + folder_ids)

                cur.execute(f"""
                    DELETE FROM project_folders
                    WHERE project_id = %s AND id IN ({','.join(['%s']*len(folder_ids))})
                """, [project_id] + folder_ids)

                conn.commit()
                return jsonify({
                    "success": True,
                    "folders_moved": len(folders),
                    "files_moved": len(files) if files else 0
                })

@admin_bp.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/rename', methods=['PUT'])
def rename_folder(project_id, folder_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not _can_edit_folder(project_id, folder_id, user_id):
        return jsonify({"error": "Permission denied"}), 403
    data = request.get_json()
    new_name = data.get('name', '').strip()
    if not new_name:
        return jsonify({"error": "Folder name required"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT parent_folder_id FROM project_folders WHERE id = %s AND project_id = %s",
                        (folder_id, project_id))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Folder not found"}), 404
            parent_id = row[0]
            cur.execute(
                "SELECT id FROM project_folders WHERE project_id = %s AND parent_folder_id = %s AND name = %s AND id != %s",
                (project_id, parent_id, new_name, folder_id))
            if cur.fetchone():
                return jsonify({"error": "A folder with this name already exists in this location"}), 400
            cur.execute("UPDATE project_folders SET name = %s WHERE id = %s", (new_name, folder_id))
            conn.commit()
            return jsonify({"success": True})

# Project files management
os.makedirs(PROJECT_FILES_ROOT, exist_ok=True)

def get_project_file_path(project_id, unique_filename):
    project_dir = os.path.join(PROJECT_FILES_ROOT, str(project_id))
    os.makedirs(project_dir, exist_ok=True)
    return os.path.join(project_dir, unique_filename)

@admin_bp.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/upload', methods=['POST'])
def upload_project_file(project_id, folder_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
            if not row or row[0] != 'active':
                return jsonify({"error": "Project is not active. Cannot upload."}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s", (folder_id, project_id))
            if not cur.fetchone():
                return jsonify({"error": "Folder not found"}), 404

    original_name = file.filename
    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file.seek(0)

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, original_name, stored_path, version, folder_id FROM project_files WHERE project_id = %s AND file_hash = %s", (project_id, file_hash))
            duplicate = cur.fetchone()
            if duplicate:
                return jsonify({
                    "duplicate": True,
                    "existing_file": {
                        "id": duplicate['id'],
                        "original_name": duplicate['original_name'],
                        "folder_id": duplicate['folder_id'],
                        "version": duplicate['version']
                    },
                    "new_filename": original_name
                })

    ext = os.path.splitext(original_name)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = get_project_file_path(project_id, unique_name)
    # Save the binary file
    file.save(stored_path)
    file_size = os.path.getsize(stored_path)

    # Extract text content from the file for knowledge base and search
    fake_file = FileStorage(BytesIO(file_bytes), filename=original_name)
    text_content, _ = extract_text_from_file(fake_file)
    if not text_content or text_content.startswith("["):
        text_content = ""  # fallback

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO project_files (project_id, folder_id, filename, original_name, file_size,
                                               stored_path, uploaded_by, file_hash, content)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (project_id, folder_id, unique_name, original_name, file_size, stored_path, user_id, file_hash, text_content))
                file_id = cur.fetchone()[0]
                conn.commit()
                return jsonify({"success": True, "file_id": file_id, "original_name": original_name, "version": 1})

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/new_version', methods=['POST'])
def new_file_version(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT stored_path, version, original_name, folder_id, filename
                        FROM project_files
                        WHERE id = %s
                          AND project_id = %s
                        """, (file_id, project_id))
            existing = cur.fetchone()
            if not existing:
                return jsonify({"error": "File not found"}), 404

            original_name = file.filename
            file_bytes = file.read()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            file.seek(0)

            # Only extract text for supported text/office files; for images, just store.
            ext = os.path.splitext(original_name)[1].lower()
            text_extensions = {'.txt', '.md', '.text', '.csv', '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt'}
            if ext in text_extensions:
                # Only attempt text extraction for office documents, not for images
                if file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')):
                    file_content = "[Image file – no text extracted]"
                else:
                    file_content, _ = extract_text_from_file(file)
                    if not file_content or file_content.startswith("["):
                        return jsonify({"error": "Could not extract text from new version"}), 400
            else:
                # For images, audio, video, etc., just store empty content
                file_content = ""

            ext = os.path.splitext(original_name)[1]
            unique_name = f"{uuid.uuid4().hex}{ext}"
            stored_path = get_project_file_path(project_id, unique_name)
            file.save(stored_path)
            file_size = os.path.getsize(stored_path)
            new_version = existing['version'] + 1

            cur.execute("""
                        INSERT INTO project_file_versions (file_id, version, stored_path, file_size, uploaded_by)
                        VALUES (%s, %s, %s, %s, %s)
                        """, (file_id, existing['version'], existing['stored_path'], file_size, user_id))

            cur.execute("""
                        UPDATE project_files
                        SET version       = %s,
                            stored_path   = %s,
                            file_size     = %s,
                            uploaded_at   = NOW(),
                            uploaded_by   = %s,
                            file_hash     = %s,
                            original_name = %s,
                            content       = %s
                        WHERE id = %s
                        """,
                        (new_version, stored_path, file_size, user_id, file_hash, original_name, file_content, file_id))

            cur.execute("""
                        INSERT INTO project_file_usage (file_id, user_id, action, details)
                        VALUES (%s, %s, 'new_version', %s)
                        """, (file_id, user_id, json.dumps({'version': new_version, 'size': file_size})))

            conn.commit()
            return jsonify({"success": True, "file_id": file_id, "original_name": original_name, "version": new_version})

@admin_bp.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/files', methods=['GET'])
def list_project_files(project_id, folder_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403

    # Pre-compute permissions once to avoid N+1 DB round-trips per file
    _is_admin = is_admin()
    _role = get_user_role_in_project(project_id, user_id) if not _is_admin else 'admin'
    _is_manager = _role == 'manager'
    _is_member = _role == 'member'
    _can_move_all = _is_admin or _is_manager or _is_member
    _can_edit_all = _is_admin or _is_manager  # member needs per-file uploaded_by check

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT f.id,
                               f.original_name,
                               f.file_size,
                               f.version,
                               f.uploaded_at,
                               f.uploaded_by,
                               (SELECT username FROM users WHERE user_id = f.uploaded_by)        as uploaded_by_name,
                               (SELECT COUNT(*) FROM project_file_versions WHERE file_id = f.id) as version_count
                        FROM project_files f
                        WHERE f.project_id = %s
                          AND f.folder_id = %s
                        ORDER BY f.uploaded_at DESC
                        """, (project_id, folder_id))
            files = cur.fetchall()
            result = []
            for f in files:
                f['has_versions'] = f['version_count'] > 0
                f['can_move'] = _can_move_all
                f['can_delete'] = _can_edit_all or (_is_member and f['uploaded_by'] == user_id)
                f['can_rename'] = _can_edit_all or (_is_member and f['uploaded_by'] == user_id)
                f['can_download'] = True
                f['file_size_kb'] = round(f['file_size'] / 1024, 1)
                f['uploaded_at_str'] = f['uploaded_at'].strftime('%Y-%m-%d %H:%M:%S')
                result.append(f)
            return jsonify({"files": result})

@admin_bp.route('/admin/projects/<int:project_id>/files', methods=['GET'])
def list_root_files(project_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403

    # Pre-compute permissions once to avoid N+1 DB round-trips per file
    _is_admin = is_admin()
    _role = get_user_role_in_project(project_id, user_id) if not _is_admin else 'admin'
    _is_manager = _role == 'manager'
    _is_member = _role == 'member'
    _can_move_all = _is_admin or _is_manager or _is_member
    _can_edit_all = _is_admin or _is_manager  # member needs per-file uploaded_by check

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT f.id,
                               f.original_name,
                               f.file_size,
                               f.version,
                               f.uploaded_at,
                               f.uploaded_by,
                               (SELECT username FROM users WHERE user_id = f.uploaded_by)        as uploaded_by_name,
                               (SELECT COUNT(*) FROM project_file_versions WHERE file_id = f.id) as version_count
                        FROM project_files f
                        WHERE f.project_id = %s
                          AND f.folder_id IS NULL
                        ORDER BY f.uploaded_at DESC
                        """, (project_id,))
            files = cur.fetchall()
            result = []
            for f in files:
                f['has_versions'] = f['version_count'] > 0
                f['can_move'] = _can_move_all
                f['can_delete'] = _can_edit_all or (_is_member and f['uploaded_by'] == user_id)
                f['can_rename'] = _can_edit_all or (_is_member and f['uploaded_by'] == user_id)
                f['can_download'] = True
                f['file_size_kb'] = round(f['file_size'] / 1024, 1)
                f['uploaded_at_str'] = f['uploaded_at'].strftime('%Y-%m-%d %H:%M:%S')
                result.append(f)
            return jsonify({"files": result})

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/versions', methods=['GET'])
def get_file_versions(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT version,
                               file_size,
                               uploaded_at,
                               uploaded_by,
                               (SELECT username FROM users WHERE user_id = fv.uploaded_by) as uploaded_by_name
                        FROM project_file_versions fv
                        WHERE file_id = %s
                        ORDER BY version DESC
                        """, (file_id,))
            versions = cur.fetchall()
            return jsonify({"versions": versions})

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/download', methods=['GET'])
def download_project_file(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    version = request.args.get('version', type=int)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if version:
                cur.execute(
                    "SELECT stored_path, original_name FROM project_file_versions WHERE file_id = %s AND version = %s",
                    (file_id, version))
            else:
                cur.execute("SELECT stored_path, original_name FROM project_files WHERE id = %s", (file_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "File not found"}), 404
            stored_path, original_name = row
    if not os.path.exists(stored_path):
        return jsonify({"error": "文件已被清理，无法下载"}), 410
    return send_file(stored_path, as_attachment=True, download_name=original_name)

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/comments', methods=['GET'])
def get_file_comments(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT c.id, c.comment, c.created_at, u.username
                        FROM project_file_comments c
                                 JOIN users u ON c.user_id = u.user_id
                        WHERE c.file_id = %s
                        ORDER BY c.created_at ASC
                        """, (file_id,))
            comments = cur.fetchall()
            return jsonify({"comments": comments})

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/comments', methods=['POST'])
def add_file_comment(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    data = request.get_json()
    comment = data.get('comment', '').strip()
    if not comment:
        return jsonify({"error": "Comment cannot be empty"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO project_file_comments (file_id, user_id, comment) VALUES (%s, %s, %s)",
                        (file_id, user_id, comment))
            conn.commit()
            return jsonify({"success": True})

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/move', methods=['POST'])
def move_file(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not _can_move_file(project_id, file_id, user_id):
        return jsonify({"error": "Permission denied"}), 403
    data = request.get_json()
    target_folder_id = data.get('folder_id')
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if target_folder_id:
                cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s", (target_folder_id, project_id))
                if not cur.fetchone():
                    return jsonify({"error": "Target folder not found in this project"}), 404
            cur.execute("UPDATE project_files SET folder_id = %s WHERE id = %s AND project_id = %s", (target_folder_id, file_id, project_id))
            if cur.rowcount == 0:
                return jsonify({"error": "File not found"}), 404
            conn.commit()
            return jsonify({"success": True})

@admin_bp.route('/admin/projects/<int:project_id>/files/batch_move', methods=['POST'])
def batch_move_files(project_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    file_ids = data.get('file_ids', [])
    target_folder_id = data.get('folder_id')
    if not file_ids:
        return jsonify({"error": "No files selected"}), 400
    if not target_folder_id:
        return jsonify({"error": "Target folder required"}), 400

    role = get_user_role_in_project(project_id, user_id)
    if not role and not is_admin():
        return jsonify({"error": "You are not a member of this project"}), 403

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s", (target_folder_id, project_id))
            if not cur.fetchone():
                return jsonify({"error": "Target folder not found in this project"}), 404

            placeholders = ','.join(['%s'] * len(file_ids))
            cur.execute(f"""
                SELECT id FROM project_files 
                WHERE id IN ({placeholders}) AND project_id = %s
            """, file_ids + [project_id])
            found = cur.fetchall()
            if len(found) != len(file_ids):
                return jsonify({"error": "Some files not found in this project"}), 404

            cur.execute(f"""
                UPDATE project_files SET folder_id = %s 
                WHERE id IN ({placeholders}) AND project_id = %s
            """, [target_folder_id] + file_ids + [project_id])
            conn.commit()
            return jsonify({"success": True, "moved_count": len(file_ids)})

@admin_bp.route('/admin/projects/<int:project_id>/batch_download', methods=['POST'])
def batch_download_files(project_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    data = request.get_json()
    file_ids = data.get('file_ids', [])
    if not file_ids:
        return jsonify({"error": "No files selected"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            placeholders = ','.join(['%s'] * len(file_ids))
            cur.execute(
                f"SELECT stored_path, original_name FROM project_files WHERE id IN ({placeholders}) AND project_id = %s",
                file_ids + [project_id])
            files = cur.fetchall()
            if not files:
                return jsonify({"error": "No valid files found"}), 404
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for stored_path, original_name in files:
                    zipf.write(stored_path, original_name)
            zip_buffer.seek(0)
            return send_file(zip_buffer, as_attachment=True, download_name=f"project_{project_id}_files.zip",
                             mimetype='application/zip')

@admin_bp.route('/admin/projects/<int:project_id>/files/search', methods=['GET'])
def search_project_files(project_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return jsonify({"files": []})
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT f.id, f.original_name, f.file_size, f.uploaded_at, fo.name as folder_name
                        FROM project_files f
                                 LEFT JOIN project_folders fo ON f.folder_id = fo.id
                        WHERE f.project_id = %s
                          AND f.original_name ILIKE %s
                        ORDER BY f.uploaded_at DESC
                        LIMIT 50
                        """, (project_id, f'%{query}%'))
            files = cur.fetchall()
            for f in files:
                f['file_size_kb'] = round(f['file_size'] / 1024, 1)
            return jsonify({"files": files})

@admin_bp.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/comments', methods=['GET'])
def get_folder_comments(project_id, folder_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT c.id, c.comment, c.created_at, u.username
                        FROM project_folder_comments c
                                 JOIN users u ON c.user_id = u.user_id
                        WHERE c.folder_id = %s
                        ORDER BY c.created_at ASC
                        """, (folder_id,))
            comments = cur.fetchall()
            return jsonify({"comments": comments})

@admin_bp.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/comments', methods=['POST'])
def add_folder_comment(project_id, folder_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    data = request.get_json()
    comment = data.get('comment', '').strip()
    if not comment:
        return jsonify({"error": "Comment cannot be empty"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO project_folder_comments (folder_id, user_id, comment) VALUES (%s, %s, %s)",
                        (folder_id, user_id, comment))
            conn.commit()
            return jsonify({"success": True})

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/rename', methods=['PUT'])
def rename_project_file(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not _can_edit_file(project_id, file_id, user_id):
        return jsonify({"error": "Permission denied"}), 403
    data = request.get_json()
    new_name = data.get('original_name', '').strip()
    if not new_name:
        return jsonify({"error": "New name required"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE project_files SET original_name = %s WHERE id = %s AND project_id = %s",
                        (new_name, file_id, project_id))
            if cur.rowcount == 0:
                return jsonify({"error": "File not found"}), 404
            conn.commit()
    return jsonify({"success": True})

# ========== Knowledge Lab Routes ==========
@admin_bp.route('/admin/db_tables', methods=['GET'])
@admin_required
def admin_db_tables():
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT tablename
                        FROM pg_tables
                        WHERE schemaname = 'public'
                        ORDER BY tablename
                        """)
            tables = [row['tablename'] for row in cur.fetchall()]
            return jsonify({"tables": tables})

def _query_db_table(table, page, per_page, search, search_column):
    """Shared helper: run a paginated, searchable SELECT against any public table."""
    if not table:
        return jsonify({"error": "Table name required"}), 400

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = %s", (table,))
            if not cur.fetchone():
                return jsonify({"error": "Invalid table name"}), 400

    offset = (page - 1) * per_page
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = %s
                        ORDER BY ordinal_position
                        """, (table,))
            columns = cur.fetchall()
            col_names = [col['column_name'] for col in columns]

            if 'id' in col_names: order_col = 'id'
            else: order_col = col_names[0] if col_names else '1'

            where_clause = sql.SQL("")
            params = []
            if search and search_column and search_column in col_names:
                where_clause = sql.SQL(" WHERE {col}::text ILIKE %s").format(col=sql.Identifier(search_column))
                params.append(f"%{search}%")
            elif search:
                text_cols = [col['column_name'] for col in columns if
                             col['data_type'] in ('text', 'varchar', 'character varying', 'char', 'name')]
                if text_cols:
                    conditions = sql.SQL(" OR ").join([
                        sql.SQL("{c}::text ILIKE %s").format(c=sql.Identifier(c)) for c in text_cols
                    ])
                    where_clause = sql.SQL(" WHERE {conditions}").format(conditions=conditions)
                    params.extend([f"%{search}%"] * len(text_cols))

            table_ident = sql.Identifier(table)
            order_ident = sql.Identifier(order_col)
            count_query = sql.SQL("SELECT COUNT(*) as total FROM {table} {where}").format(
                table=table_ident, where=where_clause
            ).as_string(conn)
            cur.execute(count_query, params)
            total = cur.fetchone()['total']

            query = sql.SQL("SELECT * FROM {table} {where} ORDER BY {order_col} DESC LIMIT %s OFFSET %s").format(
                table=table_ident, where=where_clause, order_col=order_ident
            ).as_string(conn)
            cur.execute(query, params + [per_page, offset])
            rows = cur.fetchall()

            return jsonify({
                "columns": col_names,
                "rows": rows,
                "total": total,
                "page": page,
                "per_page": per_page
            })

@admin_bp.route('/admin/db_data', methods=['GET'])
@admin_required
def admin_db_data_get():
    """GET version used by sidebar DB overview and exports."""
    table = request.args.get('table', '')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    search = request.args.get('search', '').strip()
    search_column = request.args.get('search_column', '')
    return _query_db_table(table, page, per_page, search, search_column)

@admin_bp.route('/admin/db_schema', methods=['GET'])
@admin_required
def admin_db_schema():
    """Return column info for a given table (used by sidebar 'view schema' button)."""
    table = request.args.get('table', '')
    if not table:
        return jsonify({"error": "Table name required"}), 400
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename=%s", (table,))
            if not cur.fetchone():
                return jsonify({"error": "Invalid table"}), 400
            cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name=%s ORDER BY ordinal_position", (table,))
            cols = [dict(r) for r in cur.fetchall()]
    return jsonify({"columns": cols})


@admin_bp.route('/admin/db_table_data', methods=['POST'])
@admin_required
def admin_db_table_data():
    data = request.get_json()
    table = data.get('table')
    page = int(data.get('page', 1))
    per_page = int(data.get('per_page', 50))
    search = data.get('search', '').strip()
    search_column = data.get('search_column', '')
    return _query_db_table(table, page, per_page, search, search_column)

IMMUTABLE_TABLES = {'admin_audit_log'}

@admin_bp.route('/admin/db_update_row', methods=['POST'])
@admin_required
@admin_rate_limiter
def admin_db_update_row():
    data = request.get_json()
    table = data.get('table')
    if table in IMMUTABLE_TABLES:
        return jsonify({"error": "审计日志不可修改，仅可查看和导出"}), 403
    row_id = data.get('row_id')
    column = data.get('column')
    new_value = data.get('value')
    pin = data.get('pin', '').strip()
    admin_user_id = session.get('user_id')
    admin_username = session.get('username', 'admin')

    admin_hash = current_app.config.get('ADMIN_PASSWORD_HASH')
    if not admin_hash:
        logger.error("Admin password hash not configured")
        log_admin_action(admin_user_id, admin_username, 'UPDATE', table, row_id,
                         column, None, new_value, success=False,
                         error_message="Admin password hash not configured")
        return jsonify({"error": "Admin authentication not configured"}), 500

    if not check_password_hash(admin_hash, pin):
        log_admin_action(admin_user_id, admin_username, 'UPDATE', table, row_id,
                         column, None, new_value, success=False,
                         error_message="Invalid admin PIN")
        return jsonify({"error": "Invalid admin PIN"}), 403

    # Validate table exists
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = %s", (table,))
            if not cur.fetchone():
                log_admin_action(admin_user_id, admin_username, 'UPDATE', table, row_id,
                                 column, None, new_value, success=False,
                                 error_message=f"Invalid table name: {table}")
                return jsonify({"error": "Invalid table name"}), 400

    # Validate column exists
    if not validate_table_column(table, column):
        log_admin_action(admin_user_id, admin_username, 'UPDATE', table, row_id,
                         column, None, new_value, success=False,
                         error_message=f"Invalid column: {column}")
        return jsonify({"error": f"Column '{column}' does not exist"}), 400

    # Determine primary key column
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s AND column_name IN ('id', 'thread_id')
            """, (table,))
            pk_col = cur.fetchone()
            if not pk_col:
                log_admin_action(admin_user_id, admin_username, 'UPDATE', table, row_id,
                                 column, None, new_value, success=False,
                                 error_message="No primary key column found")
                return jsonify({"error": f"Table '{table}' has no known primary key"}), 400
            pk_col = pk_col[0]

    if pk_col == 'id':
        try:
            row_id_val = int(row_id)
        except ValueError:
            log_admin_action(admin_user_id, admin_username, 'UPDATE', table, row_id,
                             column, None, new_value, success=False,
                             error_message="Invalid row_id (not an integer)")
            return jsonify({"error": "Row ID must be an integer"}), 400
    else:
        row_id_val = row_id

    old_value = None
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            select_q = sql.SQL("SELECT {col} FROM {table} WHERE {pk} = %s").format(
                col=sql.Identifier(column), table=sql.Identifier(table), pk=sql.Identifier(pk_col)
            ).as_string(conn)
            cur.execute(select_q, (row_id_val,))
            row = cur.fetchone()
            if row:
                old_value = str(row[0]) if row[0] is not None else None
            else:
                log_admin_action(admin_user_id, admin_username, 'UPDATE', table, str(row_id_val),
                                 column, None, new_value, success=False,
                                 error_message="Row not found")
                return jsonify({"error": "Row not found"}), 404

    # Execute update
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            update_q = sql.SQL("UPDATE {table} SET {col} = %s WHERE {pk} = %s").format(
                table=sql.Identifier(table), col=sql.Identifier(column), pk=sql.Identifier(pk_col)
            ).as_string(conn)
            cur.execute(update_q, (new_value, row_id_val))
            conn.commit()

    # Log successful update
    log_admin_action(admin_user_id, admin_username, 'UPDATE', table, str(row_id_val),
                     column, old_value, new_value, success=True)

    return jsonify({"success": True})

@admin_bp.route('/admin/db_delete_row', methods=['POST'])
@admin_required
@admin_rate_limiter
def admin_db_delete_row():
    data = request.get_json()
    table = data.get('table')
    if table in IMMUTABLE_TABLES:
        return jsonify({"error": "审计日志不可删除，仅可查看和导出"}), 403
    row_id = data.get('row_id')
    pin = data.get('pin', '').strip()
    admin_user_id = session.get('user_id')
    admin_username = session.get('username', 'admin')

    if not table or not row_id or row_id == 'undefined':
        log_admin_action(admin_user_id, admin_username, 'DELETE', table, row_id,
                         success=False, error_message="Missing table or row_id")
        return jsonify({"error": "Missing table or valid row_id"}), 400

    admin_hash = current_app.config.get('ADMIN_PASSWORD_HASH')
    if not admin_hash:
        log_admin_action(admin_user_id, admin_username, 'DELETE', table, row_id,
                         success=False, error_message="Admin password hash not configured")
        return jsonify({"error": "Admin authentication not configured"}), 500

    if not check_password_hash(admin_hash, pin):
        log_admin_action(admin_user_id, admin_username, 'DELETE', table, row_id,
                         success=False, error_message="Invalid admin PIN")
        return jsonify({"error": "Invalid admin PIN"}), 403

    # Validate table exists
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = %s", (table,))
            if not cur.fetchone():
                log_admin_action(admin_user_id, admin_username, 'DELETE', table, row_id,
                                 success=False, error_message=f"Invalid table: {table}")
                return jsonify({"error": "Invalid table name"}), 400

    # Determine primary key column
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s AND column_name IN ('id', 'thread_id')
            """, (table,))
            pk_col = cur.fetchone()
            if not pk_col:
                log_admin_action(admin_user_id, admin_username, 'DELETE', table, row_id,
                                 success=False, error_message="No primary key column found")
                return jsonify({"error": f"Table '{table}' has no known primary key"}), 400
            pk_col = pk_col[0]

    if pk_col == 'id':
        try:
            row_id_val = int(row_id)
        except ValueError:
            log_admin_action(admin_user_id, admin_username, 'DELETE', table, row_id,
                             success=False, error_message="Invalid row_id (not integer)")
            return jsonify({"error": "Row ID must be an integer"}), 400
    else:
        row_id_val = row_id

    row_snapshot = None
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            select_q = sql.SQL("SELECT * FROM {table} WHERE {pk} = %s").format(
                table=sql.Identifier(table), pk=sql.Identifier(pk_col)
            ).as_string(conn)
            cur.execute(select_q, (row_id_val,))
            row = cur.fetchone()
            if row:
                # Convert to dict for logging
                columns = [desc[0] for desc in cur.description]
                row_snapshot = dict(zip(columns, row))
            else:
                log_admin_action(admin_user_id, admin_username, 'DELETE', table, str(row_id_val),
                                 success=False, error_message="Row not found")
                return jsonify({"error": "Row not found"}), 404

    # Execute deletion
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            delete_q = sql.SQL("DELETE FROM {table} WHERE {pk} = %s").format(
                table=sql.Identifier(table), pk=sql.Identifier(pk_col)
            ).as_string(conn)
            cur.execute(delete_q, (row_id_val,))
            conn.commit()

    # Log successful deletion (store row snapshot as old_value JSON)
    import json as json_module
    log_admin_action(admin_user_id, admin_username, 'DELETE', table, str(row_id_val),
                     old_value=json_module.dumps(row_snapshot, default=str) if row_snapshot else None,
                     success=True)

    return jsonify({"success": True})

@admin_bp.route('/admin/archived_sessions', methods=['GET'])
@admin_required
def admin_archived_sessions():
    """List archived chat sessions — reads from DB + falls back to disk JSON."""
    import os, json
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT a.thread_id, a.archived_at, a.user_id, a.archive_path,
                       cs.title, cs.updated_at, cs.project_id
                FROM archived_sessions a
                LEFT JOIN chat_sessions cs ON a.thread_id = cs.thread_id
                ORDER BY a.archived_at DESC LIMIT 50
            """)
            rows = cur.fetchall()
    result = []
    for r in rows:
        entry = dict(r)
        # If DB row is gone (project deleted), read title from disk JSON
        if not entry.get('title') and entry.get('archive_path'):
            try:
                json_path = os.path.join(os.path.dirname(__file__), '..', '..', entry['archive_path'])
                msgs_path = json_path.replace('_session.json', '_messages.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        entry['title'] = data.get('title', '(已归档)')
                if os.path.exists(msgs_path):
                    with open(msgs_path, 'r', encoding='utf-8') as f:
                        entry['message_count'] = len(json.load(f))
            except Exception:
                entry['title'] = '(已归档)'
        if not entry.get('title'):
            entry['title'] = '(已归档)'
        result.append(entry)
    return jsonify({"sessions": result})

@admin_bp.route('/admin/archived_sessions/<thread_id>', methods=['DELETE'])
@admin_required
def delete_archived_session(thread_id):
    """Delete a single archived session (DB row + disk files)."""
    import os
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT archive_path FROM archived_sessions WHERE thread_id = %s", (thread_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Not found"}), 404
            # Remove disk files
            base = row['archive_path']
            if base:
                root = os.path.join(os.path.dirname(__file__), '..', '..')
                for suffix in ['_session.json', '_messages.json', '_feedback.json']:
                    p = os.path.join(root, base.replace('_session.json', suffix))
                    if os.path.exists(p):
                        os.remove(p)
            cur.execute("DELETE FROM archived_sessions WHERE thread_id = %s", (thread_id,))
            conn.commit()
    return jsonify({"success": True})

@admin_bp.route('/admin/archived_sessions', methods=['DELETE'])
@admin_required
def delete_selected_archived_sessions():
    """Delete selected archived sessions. Body: {"thread_ids": ["id1","id2"]}"""
    import os
    data = request.get_json() or {}
    thread_ids = data.get('thread_ids', [])
    if not thread_ids:
        return jsonify({"error": "No thread_ids provided"}), 400
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for tid in thread_ids:
                cur.execute("SELECT archive_path FROM archived_sessions WHERE thread_id = %s", (tid,))
                row = cur.fetchone()
                if row and row['archive_path']:
                    root = os.path.join(os.path.dirname(__file__), '..', '..')
                    base = row['archive_path']
                    for suffix in ['_session.json', '_messages.json', '_feedback.json']:
                        p = os.path.join(root, base.replace('_session.json', suffix))
                        if os.path.exists(p):
                            os.remove(p)
            cur.execute("DELETE FROM archived_sessions WHERE thread_id = ANY(%s)", (thread_ids,))
            conn.commit()
    return jsonify({"success": True, "deleted": len(thread_ids)})

@admin_bp.route('/admin/archived_sessions/all', methods=['DELETE'])
@admin_required
def delete_all_archived_sessions():
    """Delete ALL archived sessions (DB rows + disk files)."""
    import os, glob
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dump')
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT archive_path FROM archived_sessions")
            for row in cur.fetchall():
                base = row['archive_path']
                if base:
                    root = os.path.join(os.path.dirname(__file__), '..', '..')
                    for suffix in ['_session.json', '_messages.json', '_feedback.json']:
                        p = os.path.join(root, base.replace('_session.json', suffix))
                        if os.path.exists(p):
                            os.remove(p)
            cur.execute("DELETE FROM archived_sessions")
            conn.commit()
    return jsonify({"success": True})

@admin_bp.route('/admin/clear_file_cache', methods=['POST'])
@admin_required
def clear_file_cache():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE file_text_cache")
            conn.commit()
    return jsonify({"success": True})

# Project presence tracking (lightweight in-memory)
_project_presence = {}

@admin_bp.route('/admin/projects/<int:project_id>/ping', methods=['POST'])
def project_ping(project_id):
    """Update user's last-active timestamp for a project."""
    user_id = session.get('user_id')
    username = session.get('username', 'unknown')
    if not user_id: return jsonify({"error": "Not logged in"}), 401
    _project_presence.setdefault(project_id, {})[user_id] = {
        'username': username, 'ts': time.time()
    }
    return jsonify({"status": "ok"})

@admin_bp.route('/admin/projects/<int:project_id>/presence', methods=['GET'])
def project_presence(project_id):
    """Get currently active users in a project (active in last 60s)."""
    now = time.time()
    active = {}
    for uid, info in _project_presence.get(project_id, {}).items():
        if now - info['ts'] < 60:
            active[uid] = info['username']
    return jsonify({"active_users": active})

@admin_bp.route('/admin/projects/<int:project_id>/ai_assist/stream', methods=['POST'])
def project_ai_assist_stream(project_id):
    """Streaming SSE version of ai_assist — timer + typewriter effect in frontend."""
    from app.routes.projects import can_access_project
    import hashlib
    user_id = session.get('user_id')
    username = session.get('username', user_id)
    if not user_id:
        return jsonify({"error": "未登录"}), 401
    if not can_access_project(project_id, user_id):
        return jsonify({"error": "无权访问此项目"}), 403

    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    if not query or len(query) < 3:
        return jsonify({"error": "请用一句话描述您想生成的内容"}), 400
    quoted_message_id = data.get('quoted_message_id')

    # Build context (reuse same context gathering logic)
    proj_industry = 'general'
    system_prompt = ""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Project files
                cur.execute("""
                    SELECT id, original_name, content, skill_summary, file_hash
                    FROM project_files
                    WHERE project_id = %s AND content IS NOT NULL AND content != ''
                    ORDER BY uploaded_at DESC LIMIT 15
                """, (project_id,))
                project_files = cur.fetchall()
                from app.services.context_utils import deduplicate_names
                display_names = deduplicate_names(project_files, name_key='original_name', id_key='id', hash_key='file_hash')
                project_texts = []
                for i, f in enumerate(project_files):
                    text = (f.get('skill_summary') or '') + '\n' + (f.get('content') or '')
                    if text.strip():
                        project_texts.append(f"--- 项目文件: {display_names[i]} ---\n{text[:2000]}")
                project_context = '\n'.join(project_texts[:10]) if project_texts else '(本项目暂无文件内容)'

                # RAG
                rag_context = ''
                try:
                    from app.services.rag_engine import build_rag_context
                    rag_context = build_rag_context(query, ['company_kb', 'knowledge_lab'], top_k=8, max_chars=4000)
                except Exception:
                    pass

                # Memory
                memory_context = ''
                cur.execute("""
                    SELECT pam.user_id, u.username, pam.role, pam.content, pam.created_at
                    FROM project_ai_memory pam
                    LEFT JOIN users u ON pam.user_id = u.user_id
                    WHERE pam.project_id = %s
                    ORDER BY pam.created_at DESC LIMIT 40
                """, (project_id,))
                memory_rows = cur.fetchall()
                if memory_rows:
                    query_keywords = set(query.lower().split())
                    scored = []
                    for r in memory_rows:
                        content_words = set((r['content'] or '').lower().split())
                        overlap = len(query_keywords & content_words) if query_keywords and content_words else 0
                        score = overlap / max(len(query_keywords), 1)
                        scored.append((score, r))
                    scored.sort(key=lambda x: -x[0])
                    selected = []
                    per_user = {}
                    for score, r in scored:
                        if score <= 0 and len(selected) >= 3: break
                        uid = r['user_id']
                        if per_user.get(uid, 0) >= 2: continue
                        selected.append(r)
                        per_user[uid] = per_user.get(uid, 0) + 1
                        if len(selected) >= 10: break
                    selected.sort(key=lambda r: r.get('created_at') or '', reverse=False)
                    memory_lines = []
                    for r in selected:
                        who = r.get('username') or r['user_id'] or '?'
                        memory_lines.append(f"{'@'+who if r['role']=='user' else 'AI→'+who}: {r['content'][:300]}")
                    memory_context = '\n'.join(memory_lines)

                # Industry + workflow
                cur.execute("SELECT industry FROM projects WHERE id = %s", (project_id,))
                industry_row = cur.fetchone()
                if industry_row:
                    proj_industry = industry_row.get('industry') or 'general'
                workflow_prompt = ''
                workflow_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'workflows', f'{proj_industry}.md')
                try:
                    if os.path.exists(workflow_path):
                        with open(workflow_path, 'r', encoding='utf-8') as wf:
                            workflow_prompt = wf.read()[:3000]
                except Exception:
                    pass

                # Quote context
                quote_context = ''
                if quoted_message_id:
                    try:
                        cur.execute("SELECT id, role, content FROM chat_messages WHERE id = %s", (quoted_message_id,))
                        quoted_msg = cur.fetchone()
                        if quoted_msg:
                            quoted_full = (quoted_msg['content'] or '')[:3000]
                            role_label = '用户' if quoted_msg['role'] == 'user' else 'AI'
                            quote_context = f"\n=== [QUOTED] ({role_label}) ===\n{quoted_full}\n=== END OF QUOTE ===\n"
                    except Exception:
                        pass

                rag_section = f"\n=== 知识库检索结果 ===\n{rag_context}" if rag_context else ''
                memory_section = f"\n=== 项目协作历史 ===\n{memory_context}" if memory_context else ''
                workflow_section = f"\n=== 行业标准工作流 ===\n{workflow_prompt}" if workflow_prompt else ''

                system_prompt = f"""你是投标服务机构的AI协作助手。当前请求来自 @{username}。
所有项目成员共享你的上下文，请根据身份标签引用历史对话。

=== 当前项目文件内容 ===
{project_context}
{rag_section}
{memory_section}
{workflow_section}

请严格按照行业标准工作流，根据 @{username} 的需求，结合以上项目文件、知识库和所有成员的对话历史，生成专业内容。
{quote_context}
【引用规则】只引用对话历史中确实存在的内容，不得编造。输出格式专业清晰。"""
    except Exception as e:
        return jsonify({"error": f"上下文构建失败: {str(e)[:200]}"}), 500

    # Stream LLM response
    def generate():
        try:
            from app.services.llm_provider import create_chat_model, PROVIDER_CONFIG
            import re

            # Use admin-selected provider/model if set, else auto-detect
            provider_id = None
            model_id = None
            try:
                from app.services.runtime_config import get as rc_get
                rc_provider = rc_get('active_llm_provider', '')
                rc_model = rc_get('active_llm_model', '')
                if rc_provider and rc_provider != 'auto':
                    provider_id = rc_provider
                if rc_model and rc_model != 'auto':
                    model_id = rc_model
            except Exception:
                pass

            llm = create_chat_model(
                provider_id=provider_id, model=model_id,
                streaming=True, temperature=0.5, max_tokens=3200,
                timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            )

            from langchain_core.messages import SystemMessage, HumanMessage
            from app.services.prompt_safety import sanitize_for_prompt

            safe_query = sanitize_for_prompt(query, 'project_ai')
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"@{username}: {safe_query}"),
            ]

            full_text = ''
            for chunk in llm.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    text = chunk.content
                    full_text += text
                    yield f"data: {json.dumps({'text': text})}\n\n"

            # Save to project_ai_memory after streaming completes
            try:
                answer = re.sub(r'【思考】.*?【回答】', '【回答】', full_text, flags=re.DOTALL)
                with get_db_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute("""
                            INSERT INTO project_ai_memory (project_id, user_id, role, content)
                            VALUES (%s, %s, 'assistant', %s) RETURNING id
                        """, (project_id, user_id, answer[:5000]))
                        mid = cur.fetchone()['id']
                        conn.commit()
                yield f"event: done\ndata: {json.dumps({'memory_id': mid})}\n\n"
            except Exception as e:
                logger.warning(f"Failed to save streaming ai_assist memory: {e}")

        except Exception as e:
            logger.error(f"Project AI assist stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)[:200]})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        }
    )


@admin_bp.route('/admin/projects/<int:project_id>/ai_assist', methods=['POST'])
def project_ai_assist(project_id):
    """AI-assisted content generation — shared project context with identity tags.
    
    Accepts: { query, output_format? }
    One shared chat per project. All members' AI memory visible with @username labels.
    Duplicate detection warns if same question/file already processed.
    """
    from app.routes.projects import can_access_project
    import hashlib
    user_id = session.get('user_id')
    username = session.get('username', user_id)
    if not user_id:
        return jsonify({"error": "未登录"}), 401
    if not can_access_project(project_id, user_id):
        return jsonify({"error": "无权访问此项目"}), 403

    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    output_fmt = data.get('output_format', '').strip().lower()
    quoted_message_id = data.get('quoted_message_id')
    if output_fmt not in ('docx', 'xlsx', ''):
        output_fmt = ''
    if not query or len(query) < 3:
        return jsonify({"error": "请用一句话描述您想生成的内容"}), 400

    warnings = []
    question_hash = hashlib.sha256(query[:200].encode()).hexdigest()[:16]

    # Get project name + backfill chat session for old projects
    project_name = f"项目#{project_id}"
    chat_thread_id = None
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT name FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
            if row:
                project_name = row['name']
            # Backfill: ensure shared project chat exists
            cur.execute(
                "SELECT thread_id FROM chat_sessions WHERE project_id = %s LIMIT 1",
                (project_id,)
            )
            chat_row = cur.fetchone()
            if not chat_row:
                import uuid as _uuid
                chat_thread_id = f"proj_{project_id}_{_uuid.uuid4().hex[:8]}"
                try:
                    cur.execute(
                        "INSERT INTO chat_sessions (user_id, thread_id, title, project_id) VALUES (%s,%s,%s,%s)",
                        (user_id, chat_thread_id, project_name, project_id)
                    )
                    conn.commit()
                    logger.info(f"Backfilled project chat: {project_name} (thread={chat_thread_id})")
                except Exception as e:
                    logger.warning(f"Backfill chat session failed: {e}")
                    chat_thread_id = None
            else:
                chat_thread_id = chat_row['thread_id']

    # Use shared context gatherer (available for chat.py and other callers too)
    from app.services.context_utils import gather_project_context as _gather_ctx
    proj_industry = 'general'
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Gather project file texts
            cur.execute("""
                SELECT id, original_name, content, skill_summary, file_hash
                FROM project_files
                WHERE project_id = %s AND content IS NOT NULL AND content != ''
                ORDER BY uploaded_at DESC LIMIT 15
            """, (project_id,))
            project_files = cur.fetchall()

            # Disambiguate duplicate file names for LLM
            from app.services.context_utils import deduplicate_names
            display_names = deduplicate_names(project_files, name_key='original_name', id_key='id', hash_key='file_hash')

            project_texts = []
            project_file_hashes = set()
            for i, f in enumerate(project_files):
                text = (f.get('skill_summary') or '') + '\n' + (f.get('content') or '')
                if text.strip():
                    project_texts.append(f"--- 项目文件: {display_names[i]} ---\n{text[:2000]}")
                if f.get('file_hash'):
                    project_file_hashes.add(f['file_hash'])

            # 2. RAG context: company_kb + knowledge_lab
            rag_context = ''
            try:
                from app.services.rag_engine import build_rag_context
                rag_context = build_rag_context(
                    query, ['company_kb', 'knowledge_lab'],
                    top_k=10, max_chars=6000
                )
            except Exception:
                pass

            # 3. Find matching skills
            skill_hints = []
            for table in ('knowledge_lab_files', 'company_knowledge_base', 'project_files'):
                try:
                    cur.execute(f"""
                        SELECT skill_summary FROM {table}
                        WHERE skill_summary IS NOT NULL AND skill_summary != ''
                        ORDER BY skill_generated_at DESC NULLS LAST LIMIT 10
                    """)
                    for row in cur.fetchall():
                        s = row.get('skill_summary', '')
                        if s and len(s) > 10:
                            skill_hints.append(s[:300])
                except Exception:
                    pass

            # 4. DUPLICATE DETECTION: check if same question was asked before
            cur.execute("""
                SELECT pam.user_id, u.username, pam.content, pam.created_at
                FROM project_ai_memory pam
                LEFT JOIN users u ON pam.user_id = u.user_id
                WHERE pam.project_id = %s AND pam.role = 'user' AND pam.question_hash = %s
                ORDER BY pam.created_at DESC LIMIT 1
            """, (project_id, question_hash))
            dup_row = cur.fetchone()
            if dup_row:
                prev_user = dup_row.get('username') or dup_row['user_id']
                prev_time = dup_row['created_at'].strftime('%m/%d %H:%M') if dup_row.get('created_at') else ''
                if prev_user != username:
                    warnings.append(f"⚠️ @{prev_user} 在 {prev_time} 问过相同问题，建议先查看TA的结果避免重复工作")
                else:
                    warnings.append(f"⚠️ 你在 {prev_time} 问过相同问题，要重新生成吗？")

            # 5. Load ALL members' AI memory, semantically filtered by query relevance
            memory_context = ''
            cur.execute("""
                SELECT pam.user_id, u.username, pam.role, pam.content, pam.created_at
                FROM project_ai_memory pam
                LEFT JOIN users u ON pam.user_id = u.user_id
                WHERE pam.project_id = %s
                ORDER BY pam.created_at DESC LIMIT 60
            """, (project_id,))
            memory_rows = cur.fetchall()
            if memory_rows:
                # Semantic scoring: keyword overlap between query and memory
                query_keywords = set(query.lower().split())
                scored = []
                for r in memory_rows:
                    content_lower = (r['content'] or '').lower()
                    # Jaccard-like: |query_words ∩ content_words|
                    content_words = set(content_lower.split())
                    if query_keywords and content_words:
                        overlap = len(query_keywords & content_words)
                        score = overlap / max(len(query_keywords), 1)
                    else:
                        score = 0
                    scored.append((score, r))
                scored.sort(key=lambda x: (-x[0], x[1].get('created_at') or ''), reverse=False)
                # Take top 15 by relevance, ensure identity diversity (at most 3 per user)
                selected = []
                per_user_count = {}
                for score, r in scored:
                    if score <= 0 and len(selected) >= 5:
                        break  # stop when relevance drops to zero and we have enough
                    uid = r['user_id']
                    if per_user_count.get(uid, 0) >= 3:
                        continue
                    selected.append(r)
                    per_user_count[uid] = per_user_count.get(uid, 0) + 1
                    if len(selected) >= 15:
                        break
                # Sort chronologically for coherent conversation flow
                selected.sort(key=lambda r: r.get('created_at') or '', reverse=False)
                memory_lines = []
                total_chars = 0
                for r in selected:
                    who = r.get('username') or r['user_id'] or '?'
                    label = f"@{who}" if r['role'] == 'user' else f"AI→{who}"
                    line = f"{label}: {r['content'][:400]}"
                    if total_chars + len(line) > 5000:
                        break
                    memory_lines.append(line)
                    total_chars += len(line)
                if memory_lines:
                    memory_context = '\n'.join(memory_lines)

            # 6. Check for concurrent activity
            cur.execute("""
                SELECT COUNT(*) as cnt FROM project_ai_memory
                WHERE project_id = %s AND role = 'user' AND created_at > NOW() - INTERVAL '5 minutes'
            """, (project_id,))
            recent = cur.fetchone()
            if recent and recent['cnt'] > 2:
                warnings.append(f"💡 近5分钟内有 {recent['cnt']} 次AI助手使用，注意协作避免重复工作")

            # 7. Build prompt with identity-tagged context
            project_context_raw = '\n'.join(project_texts[:10]) if project_texts else '(本项目暂无文件内容)'
            rag_context_raw = rag_context or ''
            skills_raw = '\n---\n'.join(skill_hints[:8]) if skill_hints else ''
            memory_raw = memory_context or ''

            # Load industry workflow prompt BEFORE budget (so it can be trimmed)
            cur.execute("SELECT industry FROM projects WHERE id = %s", (project_id,))
            industry_row = cur.fetchone()
            if industry_row:
                proj_industry = industry_row.get('industry') or 'general'
            workflow_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'workflows', f'{proj_industry}.md')
            try:
                if os.path.exists(workflow_path):
                    with open(workflow_path, 'r', encoding='utf-8') as wf:
                        workflow_prompt = wf.read()
            except Exception:
                pass

            # ── Token budget: trim ALL sections proportionally ──
            from app.services.prompt_safety import budget_sections
            budgeted = budget_sections({
                'project_files': project_context_raw,
                'rag': rag_context_raw,
                'skills': skills_raw,
                'memory': memory_raw,
                'workflow': workflow_prompt or '',
            }, max_total_tokens=7000)

            project_context = budgeted.get('project_files', project_context_raw)
            rag_section = f"\n=== 知识库检索结果 ===\n{budgeted.get('rag', '')}" if budgeted.get('rag') else ''
            skills_section = '\n=== 可用AI技能 ===\n' + budgeted.get('skills', '') if budgeted.get('skills') else ''
            memory_section = f"\n=== 项目协作对话历史（所有成员共享） ===\n{budgeted.get('memory', '')}" if budgeted.get('memory') else ''
            workflow_prompt = budgeted.get('workflow', workflow_prompt or '')
            workflow_section = f"\n=== 行业标准工作流 ===\n{workflow_prompt}" if workflow_prompt else ''

            # ── 8. Quote chain context (for project chat quotes) ──
            quote_context = ''
            if quoted_message_id:
                try:
                    cur.execute("SELECT id, role, content FROM chat_messages WHERE id = %s", (quoted_message_id,))
                    quoted_msg = cur.fetchone()
                    if quoted_msg:
                        quoted_full = (quoted_msg['content'] or '')[:4000]
                        role_label = '用户' if quoted_msg['role'] == 'user' else 'AI'
                        quote_context = f"\n=== [QUOTED] ({role_label}) ===\n{quoted_full}\n=== END OF QUOTE ===\n"
                except Exception as e:
                    logger.warning(f"Quote context build failed in ai_assist: {e}")

            system_prompt = f"""你是投标服务机构的AI协作助手。当前请求来自 @{username}。
所有项目成员共享你的上下文，请根据身份标签引用历史对话。

=== 当前项目文件内容 ===
{project_context}
{rag_section}
{skills_section}
{memory_section}
{workflow_section}

请严格按照行业标准工作流，根据 @{username} 的需求，结合以上项目文件、知识库、技能和所有成员的对话历史，生成专业内容。
{quote_context}
【引用规则 — 严格禁止虚构引用】
- 只有在「项目协作对话历史」中明确出现过 @某人 的内容时，才可以引用「@某人 之前分析过...」
- 不得为不存在的对话创造引用标注
- 如果某成员的对话记录中没有相关内容，就不要引用他
- 对文件内容的引用必须用文件名称（如「根据 技术方案.docx #a3f2 第X节...」）
如果某些信息不充分，请诚实说明并建议补充。输出格式专业清晰。"""

    # 8. Call LLM
    try:
        from app.services.llm_provider import call_llm
        result_text = call_llm(system_prompt, f"@{username}: {query}",
                              temperature=0.5, max_tokens=3200, industry=proj_industry)
    except Exception as e:
        logger.error(f"Project AI assist LLM error: {e}")
        return jsonify({"error": f"AI生成失败: {str(e)[:200]}"}), 500

    # 9. Save to project_ai_memory
    memory_id = None
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "INSERT INTO project_ai_memory (project_id, user_id, role, content, question_hash) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                    (project_id, user_id, 'user', query, question_hash)
                )
                cur.execute(
                    "INSERT INTO project_ai_memory (project_id, user_id, role, content, content_md) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                    (project_id, user_id, 'assistant', f"@{username}: {result_text[:2000]}", result_text)
                )
                memory_id = cur.fetchone()['id']
                conn.commit()
    except Exception as e:
        logger.warning(f"Failed to save AI memory: {e}")

    # 10. Save to shared project chat
    try:
        if not chat_thread_id:
            # Fallback: re-fetch the session if backfill failed to set chat_thread_id
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT thread_id FROM chat_sessions WHERE project_id = %s LIMIT 1", (project_id,))
                    row = cur.fetchone()
                    if row:
                        chat_thread_id = row['thread_id']
        if not chat_thread_id:
            logger.warning(f"No project chat session for project {project_id}, cannot save messages")
        else:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO chat_messages (thread_id, role, content) VALUES (%s,%s,%s)",
                        (chat_thread_id, 'user', f"@{username}: {query}")
                    )
                    cur.execute(
                        "INSERT INTO chat_messages (thread_id, role, content) VALUES (%s,%s,%s)",
                        (chat_thread_id, 'assistant', f"@{os.getenv('AI_NAME', '中联助手')}对@{username}说：\n{result_text[:4000]}")
                    )
                    cur.execute(
                        "UPDATE chat_sessions SET updated_at = NOW() WHERE thread_id = %s",
                        (chat_thread_id,)
                    )
                    conn.commit()
    except Exception as e:
        logger.warning(f"Failed to save to project chat: {e}")

    # 11. Return result with warnings
    resp = {"status": "ok", "result": result_text, "by": username}
    if warnings:
        resp["warnings"] = warnings
    if memory_id:
        resp["memory_id"] = memory_id
        resp["download_formats"] = ["docx", "xlsx"]
    return jsonify(resp)

@admin_bp.route('/admin/projects/<int:project_id>/ai_activity', methods=['GET'])
def project_ai_activity(project_id):
    """Polling endpoint: returns recent AI activity AND chat messages since ?since=ISO timestamp."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"items": []})

    since = request.args.get('since', '')
    items = []
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # AI memory entries
            if since:
                cur.execute("""
                    SELECT pam.id, pam.user_id, u.username, pam.role, pam.content,
                           pam.content_md, pam.created_at
                    FROM project_ai_memory pam
                    LEFT JOIN users u ON pam.user_id = u.user_id
                    WHERE pam.project_id = %s AND pam.created_at > %s
                    ORDER BY pam.created_at ASC LIMIT 30
                """, (project_id, since))
            else:
                cur.execute("""
                    SELECT pam.id, pam.user_id, u.username, pam.role, pam.content,
                           pam.content_md, pam.created_at
                    FROM project_ai_memory pam
                    LEFT JOIN users u ON pam.user_id = u.user_id
                    WHERE pam.project_id = %s
                    ORDER BY pam.created_at DESC LIMIT 20
                """, (project_id,))
            for r in cur.fetchall():
                items.append({
                    "type": "ai_memory",
                    "id": r['id'],
                    "user_id": r['user_id'],
                    "username": r['username'],
                    "role": r['role'],
                    "content": (r['content'] or '')[:500],
                    "content_md": r.get('content_md'),
                    "created_at": r['created_at'].isoformat() if r['created_at'] else '',
                })

            # Chat messages from project's shared chat session
            if since:
                cur.execute("""
                    SELECT cm.id, cm.thread_id, cm.role, cm.content, cm.thinking, cm.timestamp
                    FROM chat_messages cm
                    JOIN chat_sessions cs ON cm.thread_id = cs.thread_id
                    WHERE cs.project_id = %s AND cm.timestamp > %s
                    ORDER BY cm.timestamp ASC LIMIT 50
                """, (project_id, since))
            else:
                cur.execute("""
                    SELECT cm.id, cm.thread_id, cm.role, cm.content, cm.thinking, cm.timestamp
                    FROM chat_messages cm
                    JOIN chat_sessions cs ON cm.thread_id = cs.thread_id
                    WHERE cs.project_id = %s
                    ORDER BY cm.timestamp DESC LIMIT 50
                """, (project_id,))
            for r in cur.fetchall():
                items.append({
                    "type": "chat_message",
                    "id": r['id'],
                    "thread_id": r['thread_id'],
                    "role": r['role'],
                    "content": (r['content'] or '')[:1000],
                    "thinking": r.get('thinking'),
                    "created_at": r['timestamp'].isoformat() if r['timestamp'] else '',
                })

    # Sort merged
    items.sort(key=lambda x: x.get('created_at', ''))
    return jsonify({"items": items, "now": datetime.now(timezone.utc).isoformat()})

@admin_bp.route('/admin/projects/<int:project_id>/unread_count', methods=['GET'])
def project_unread_count(project_id):
    """Return unread chat messages count since user's last read position."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"count": 0})

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get user's last read timestamp
            cur.execute(
                "SELECT last_read_at FROM project_members WHERE project_id = %s AND user_id = %s",
                (project_id, user_id)
            )
            row = cur.fetchone()
            since = row['last_read_at'].isoformat() if row and row['last_read_at'] else '1970-01-01'

            # Count messages since last read
            cur.execute("""
                SELECT COUNT(*) as cnt FROM chat_messages cm
                JOIN chat_sessions cs ON cm.thread_id = cs.thread_id
                WHERE cs.project_id = %s AND cm.role = 'assistant'
                  AND cm.timestamp > %s
            """, (project_id, since))
            count = cur.fetchone()['cnt']
    return jsonify({"count": count, "since": since})

@admin_bp.route('/admin/projects/<int:project_id>/mark_read', methods=['POST'])
def project_mark_read(project_id):
    """Update user's last_read_at for this project."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # UPSERT: ensure user has a row in project_members, then update last_read_at
            cur.execute(
                "INSERT INTO project_members (project_id, user_id, role, last_read_at) "
                "VALUES (%s, %s, 'member', NOW()) "
                "ON CONFLICT (project_id, user_id) DO UPDATE SET last_read_at = NOW()",
                (project_id, user_id)
            )
            conn.commit()
    return jsonify({"success": True})


# ======================== Project Todo System ========================

@admin_bp.route('/admin/projects/<int:project_id>/todos', methods=['GET'])
def project_todos_list(project_id):
    """Get current user's pending todos for this project."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, message_id, content_copy, original_role, original_author, status, created_at
                FROM project_todos
                WHERE project_id = %s AND user_id = %s AND status = 'pending'
                ORDER BY created_at ASC
            """, (project_id, user_id))
            todos = cur.fetchall()
    return jsonify({"todos": [{
        "id": t["id"],
        "message_id": t["message_id"],
        "content_copy": t["content_copy"],
        "original_role": t["original_role"],
        "original_author": t["original_author"],
        "created_at": t["created_at"].isoformat() if t["created_at"] else None,
    } for t in todos]})


@admin_bp.route('/admin/projects/<int:project_id>/todos', methods=['POST'])
def project_todos_add(project_id):
    """Add a message to user's todo list."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    data = request.get_json() or {}
    message_id = data.get('message_id')
    content_copy = (data.get('content_copy') or '').strip()
    if not content_copy:
        return jsonify({"error": "内容不能为空"}), 400
    original_role = data.get('original_role', '')
    original_author = data.get('original_author', '')
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM project_todos WHERE project_id = %s AND user_id = %s AND status = 'pending'",
                (project_id, user_id)
            )
            count = cur.fetchone()[0]
            if count >= 5:
                return jsonify({"error": "待办最多5条，请先完成或删除现有待办"}), 400
            cur.execute("""
                INSERT INTO project_todos (project_id, user_id, message_id, content_copy, original_role, original_author)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
            """, (project_id, user_id, message_id, content_copy[:2000], original_role, original_author))
            todo_id = cur.fetchone()[0]
            conn.commit()
    return jsonify({"success": True, "todo_id": todo_id})


@admin_bp.route('/admin/projects/<int:project_id>/todos/<int:todo_id>/done', methods=['POST'])
def project_todos_done(project_id, todo_id):
    """Mark a todo as done — records to project log, visible to admin only."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    username = session.get('username', user_id)
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE project_todos SET status = 'done', done_at = NOW()
                WHERE id = %s AND project_id = %s AND user_id = %s AND status = 'pending'
                RETURNING content_copy, original_author
            """, (todo_id, project_id, user_id))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "待办不存在或已完成"}), 404
            content_copy, original_author = row
            cur.execute("""
                INSERT INTO project_ai_memory (project_id, user_id, role, content)
                VALUES (%s, %s, 'system', %s)
            """, (project_id, user_id, f"[TODO DONE] by @{username}: original from @{original_author}: {content_copy[:200]}"))
            conn.commit()
    return jsonify({"success": True})


@admin_bp.route('/admin/projects/<int:project_id>/todos/<int:todo_id>/remove', methods=['POST'])
def project_todos_remove(project_id, todo_id):
    """Completely remove a todo — no trace left."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM project_todos WHERE id = %s AND project_id = %s AND user_id = %s AND status = 'pending'",
                (todo_id, project_id, user_id)
            )
            conn.commit()
    return jsonify({"success": True})


@admin_bp.route('/admin/projects/<int:project_id>/todos/done_log', methods=['GET'])
def project_todos_done_log(project_id):
    """Admin-only: view completed todo records."""
    from app.routes.projects import is_admin
    if not is_admin():
        return jsonify({"error": "Admin only"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT t.id, t.content_copy, t.original_role, t.original_author,
                       t.done_at, t.created_at, u.username as done_by_username
                FROM project_todos t
                LEFT JOIN users u ON t.user_id = u.user_id
                WHERE t.project_id = %s AND t.status = 'done'
                ORDER BY t.done_at DESC
            """, (project_id,))
            logs = cur.fetchall()
    return jsonify({"logs": [{
        "id": l["id"],
        "content_copy": l["content_copy"],
        "original_role": l["original_role"],
        "original_author": l["original_author"],
        "done_by": l.get("done_by_username") or "unknown",
        "done_at": l["done_at"].isoformat() if l["done_at"] else None,
        "created_at": l["created_at"].isoformat() if l["created_at"] else None,
    } for l in logs]})


# ======================== Quote Tree System ========================

@admin_bp.route('/admin/projects/<int:project_id>/quote', methods=['POST'])
def project_quote_create(project_id):
    """Create a quote association (tree node)."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    data = request.get_json() or {}
    quoted_message_id = data.get('quoted_message_id')
    parent_quote_id = data.get('parent_quote_id')
    thread_id = data.get('thread_id')
    if not quoted_message_id:
        return jsonify({"error": "quoted_message_id required"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO message_quotes (project_id, quoted_message_id, parent_quote_id, thread_id)
                VALUES (%s, %s, %s, %s) RETURNING id
            """, (project_id, quoted_message_id, parent_quote_id, thread_id))
            quote_id = cur.fetchone()[0]
            conn.commit()
    return jsonify({"success": True, "quote_id": quote_id})


@admin_bp.route('/admin/projects/<int:project_id>/quote_tree/<int:message_id>', methods=['GET'])
def project_quote_tree(project_id, message_id):
    """Get the quote tree for a message — returns full ancestry chain."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            nodes = []
            current_msg_id = message_id
            visited = set()
            while current_msg_id and current_msg_id not in visited:
                visited.add(current_msg_id)
                cur.execute("""
                    SELECT mq.id, mq.quoted_message_id, mq.quoting_message_id, mq.parent_quote_id,
                           mq.thread_id, mq.created_at,
                           cm.role, cm.content, cm.timestamp
                    FROM message_quotes mq
                    LEFT JOIN chat_messages cm ON mq.quoted_message_id = cm.id
                    WHERE mq.project_id = %s AND mq.quoted_message_id = %s
                    ORDER BY mq.created_at ASC
                """, (project_id, current_msg_id))
                rows = cur.fetchall()
                if not rows:
                    break
                for r in rows:
                    nodes.append({
                        "quote_id": r["id"],
                        "quoted_message_id": r["quoted_message_id"],
                        "quoting_message_id": r["quoting_message_id"],
                        "parent_quote_id": r["parent_quote_id"],
                        "role": r["role"],
                        "content": r["content"],
                        "timestamp": r["timestamp"].isoformat() if r["timestamp"] else None,
                    })
                    if r["parent_quote_id"]:
                        cur.execute("SELECT quoted_message_id FROM message_quotes WHERE id = %s", (r["parent_quote_id"],))
                        parent = cur.fetchone()
                        if parent:
                            current_msg_id = parent["quoted_message_id"]
                            break
                    else:
                        break
                else:
                    break
    return jsonify({"nodes": nodes})


# ======================== Regeneration Vote System ========================

import re

# Number extraction pattern: integers, decimals, percentages, Chinese numbers
_NUM_PATTERN = re.compile(
    r'\d+(?:\.\d+)?(?:%|万|亿|k|K|m|M|g|G|t|T|b|B|ms|s|h)?'
    r'|[零一二三四五六七八九十百千万亿]+'
)
# Negation pattern for Chinese
_NEG_PATTERN = re.compile(
    r'(?:不|没|无|未|非|别|莫|勿|禁|否|无[法可]|不可|没有|绝不)'
)


def _extract_numbers(text: str):
    """Extract all numeric values from a Chinese/text string."""
    return _NUM_PATTERN.findall(text)


def _detect_number_change(text1: str, text2: str) -> bool:
    """Return True if the two texts have significantly different numeric content.
    
    A change is significant if:
    - different count of numbers in the text, OR
    - the same position has a different value (e.g., 3→6, 80%→60%)
    """
    nums1 = _extract_numbers(text1)
    nums2 = _extract_numbers(text2)
    
    # If neither has numbers, no change
    if not nums1 and not nums2:
        return False
    
    # Different count → very likely a structural change
    if len(nums1) != len(nums2):
        return True
    
    # Same count: compare values
    for n1, n2 in zip(nums1, nums2):
        if n1 != n2:
            return True
    
    return False


def _detect_negation_change(text1: str, text2: str) -> bool:
    """Return True if one text has negation and the other doesn't (or negation differs)."""
    neg1 = set(_NEG_PATTERN.findall(text1))
    neg2 = set(_NEG_PATTERN.findall(text2))
    
    # If neither has negation, no change
    if not neg1 and not neg2:
        return False
    
    # If negation words differ between the two texts
    return neg1 != neg2


def _compute_semantic_diff(text1, text2):
    """Compute semantic similarity between two texts using 4-signal fusion.
    
    Signals:
    1. jieba TF-IDF text similarity (word-level structural overlap)
    2. jieba keyword overlap similarity (key concept coverage)
    3. Sentence-transformers semantic embedding similarity (deep meaning)
    4. Number & negation change penalties (surface-level critical differences)
    
    Returns weighted score 0-1 (1 = identical meaning, lower = more different).
    """
    if not text1 or not text2:
        return 0.0
    
    # 1. jieba TF-IDF similarity
    tfidf_sim = 0.0
    try:
        from app.services.file_processing import _make_vectorizer, preprocess_text_for_similarity
        clean1 = preprocess_text_for_similarity(text1)
        clean2 = preprocess_text_for_similarity(text2)
        if clean1.strip() and clean2.strip():
            vectorizer = _make_vectorizer(stop_words=None)
            tfidf_matrix = vectorizer.fit_transform([clean1, clean2])
            from sklearn.metrics.pairwise import cosine_similarity
            tfidf_sim = float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])
    except Exception as e:
        logger.warning(f"TF-IDF diff failed: {e}")
    
    # 2. jieba keyword overlap similarity
    kw_sim = 0.0
    try:
        from app.services.file_processing import keyword_overlap_similarity
        kw_sim = float(keyword_overlap_similarity(text1, text2))
    except Exception as e:
        logger.warning(f"Keyword overlap diff failed: {e}")
    
    # 3. Semantic embedding similarity (language-aware model switching)
    sem_sim = 0.5  # neutral fallback
    model_used = 'none'
    try:
        from app.services.semantic import get_model_for_texts
        model, lang = get_model_for_texts(text1, text2)
        if model:
            model_used = lang
            from sklearn.metrics.pairwise import cosine_similarity
            embeddings = model.encode([text1, text2], show_progress_bar=False)
            sem_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        else:
            # Legacy fallback
            from app.services.file_processing import compute_batch_semantic_similarity as _legacy_sem
            matrix = _legacy_sem([text1, text2])
            if matrix and len(matrix) >= 2:
                sem_sim = float(matrix[0][1])
                model_used = 'legacy'
    except Exception as e:
        logger.warning(f"Semantic embedding diff failed: {e}")
    
    # Weighted fusion: Plan B weights (higher semantic, lower surface)
    # semantic 0.65 + keyword 0.2 + tfidf 0.15 = 1.0
    fused = 0.65 * sem_sim + 0.2 * kw_sim + 0.15 * tfidf_sim
    
    # 4. Number & negation change penalties
    # These catch critical differences that semantic models miss:
    # "3个月" vs "6个月", "可行" vs "不可行" — identical structure, opposite meaning
    flags = []
    num_changed = _detect_number_change(text1, text2)
    neg_changed = _detect_negation_change(text1, text2)
    
    if num_changed:
        # Penalty: subtract 0.45 from fused score
        # A 0.9 paraphrase with different numbers → 0.45 (below threshold)
        old_fused = fused
        fused = max(0.0, fused - 0.45)
        flags.append(f"num_penalty({old_fused:.3f}→{fused:.3f})")
    
    if neg_changed:
        # Penalty: subtract 0.40 from fused score
        # A 0.9 paraphrase with negation flipped → 0.50 (below threshold)
        old_fused = fused
        fused = max(0.0, fused - 0.40)
        flags.append(f"neg_penalty({old_fused:.3f}→{fused:.3f})")
    
    flag_str = ' ' + ' '.join(flags) if flags else ''
    logger.info(f"Semantic diff [{model_used}]: tfidf={tfidf_sim:.3f}, kw={kw_sim:.3f}, sem={sem_sim:.3f} → fused={fused:.3f}{flag_str}")
    return fused


def _get_involved_users(project_id, message_id):
    """Get all users involved with a message: quote tree users + todo users."""
    user_ids = set()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Users who quoted this message or its descendants
            cur.execute("""
                SELECT DISTINCT user_id FROM message_quotes
                WHERE project_id = %s AND quoted_message_id = %s
            """, (project_id, message_id))
            for row in cur.fetchall():
                user_ids.add(row[0])
            # Users who todo-ed this message
            cur.execute("""
                SELECT DISTINCT user_id FROM project_todos
                WHERE project_id = %s AND message_id = %s
            """, (project_id, message_id))
            for row in cur.fetchall():
                user_ids.add(row[0])
    return list(user_ids)


@admin_bp.route('/admin/projects/<int:project_id>/regen_votes', methods=['GET'])
def project_regen_votes_list(project_id):
    """Get active votes for this project."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT rv.id, rv.message_id, rv.original_content, rv.new_content,
                       rv.status, rv.round, rv.expires_at, rv.created_at,
                       (SELECT COUNT(*) FROM regen_vote_ballots WHERE vote_id = rv.id AND vote = 'keep_original') as keep_count,
                       (SELECT COUNT(*) FROM regen_vote_ballots WHERE vote_id = rv.id AND vote = 'replace') as replace_count,
                       (SELECT COUNT(*) FROM regen_vote_ballots WHERE vote_id = rv.id AND voter_id = %s) as my_vote_count,
                       (SELECT vote FROM regen_vote_ballots WHERE vote_id = rv.id AND voter_id = %s) as my_vote
                FROM regen_votes rv
                WHERE rv.project_id = %s AND rv.status = 'active'
                ORDER BY rv.created_at DESC
            """, (user_id, user_id, project_id))
            votes = cur.fetchall()
    return jsonify({"votes": [{
        "id": v["id"],
        "message_id": v["message_id"],
        "original_content": (v["original_content"] or '')[:200],
        "new_content": (v["new_content"] or '')[:200],
        "round": v["round"],
        "expires_at": v["expires_at"].isoformat() if v["expires_at"] else None,
        "keep_count": v["keep_count"],
        "replace_count": v["replace_count"],
        "my_vote": v["my_vote"],
        "created_at": v["created_at"].isoformat() if v["created_at"] else None,
    } for v in votes]})


@admin_bp.route('/admin/projects/<int:project_id>/regen_votes/<int:vote_id>/cast', methods=['POST'])
def project_regen_vote_cast(project_id, vote_id):
    """Cast a vote on a regeneration proposal."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403
    data = request.get_json() or {}
    vote_choice = data.get('vote')  # 'keep_original' or 'replace'
    if vote_choice not in ('keep_original', 'replace'):
        return jsonify({"error": "Invalid vote"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check vote is still active
            cur.execute("SELECT status, expires_at FROM regen_votes WHERE id = %s AND project_id = %s", (vote_id, project_id))
            row = cur.fetchone()
            if not row or row[0] != 'active':
                return jsonify({"error": "投票已结束"}), 400
            # Upsert ballot
            cur.execute("""
                INSERT INTO regen_vote_ballots (vote_id, voter_id, vote)
                VALUES (%s, %s, %s)
                ON CONFLICT (vote_id, voter_id) DO UPDATE SET vote = EXCLUDED.vote
            """, (vote_id, user_id, vote_choice))
            conn.commit()
    return jsonify({"success": True})


@admin_bp.route('/admin/projects/<int:project_id>/regen_votes/<int:vote_id>/resolve', methods=['POST'])
def project_regen_vote_resolve(project_id, vote_id):
    """Resolve a vote — manager decides on draw, or auto-resolve on timeout."""
    from app.routes.projects import can_manage_members
    user_id = session.get('user_id')
    if not user_id or not can_manage_members(project_id, user_id):
        return jsonify({"error": "仅项目经理可裁决"}), 403
    data = request.get_json() or {}
    decision = data.get('decision')  # 'keep_original' or 'replace'
    if decision not in ('keep_original', 'replace'):
        return jsonify({"error": "Invalid decision"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if decision == 'replace':
                cur.execute("""
                    UPDATE regen_votes SET status = 'resolved_replace', resolved_at = NOW()
                    WHERE id = %s AND project_id = %s
                """, (vote_id, project_id))
            else:
                cur.execute("""
                    UPDATE regen_votes SET status = 'resolved_keep', resolved_at = NOW()
                    WHERE id = %s AND project_id = %s
                """, (vote_id, project_id))
            conn.commit()
    return jsonify({"success": True})


def _check_and_create_regen_vote(project_id, message_id, new_content):
    """Check if new content is semantically very different from original. If so, create a vote.
    Called internally when a new AI response is generated in a project chat.
    Returns True if a vote was created."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get the original message content
            cur.execute("SELECT content FROM chat_messages WHERE id = %s", (message_id,))
            row = cur.fetchone()
            if not row:
                return False
            original_content = row['content']
            # Semantic similarity check (Plan B threshold: 0.55)
            similarity = _compute_semantic_diff(original_content, new_content)
            if similarity > 0.55:
                # Not different enough — ignore
                return False
            # Check if there's already an active vote for this message
            cur.execute("SELECT id FROM regen_votes WHERE message_id = %s AND status = 'active'", (message_id,))
            if cur.fetchone():
                return False
            # Create a new vote — 24h expiry
            cur.execute("""
                INSERT INTO regen_votes (project_id, message_id, original_content, new_content, status, round, expires_at)
                VALUES (%s, %s, %s, %s, 'active', 1, NOW() + INTERVAL '24 hours')
                RETURNING id
            """, (project_id, message_id, original_content, new_content))
            vote_id = cur.fetchone()['id']
            conn.commit()
            logger.info(f"Created regen vote {vote_id} for message {message_id} (similarity={similarity:.2f})")
            return True



def project_ai_download(project_id, memory_id):
    """Download AI-generated content as .docx or .xlsx.
    Query param: format=docx (default) or xlsx
    """
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "未登录"}), 401
    if not can_access_project(project_id, user_id):
        return jsonify({"error": "无权访问"}), 403

    fmt = request.args.get('format', 'docx').strip().lower()
    if fmt not in ('docx', 'xlsx'):
        return jsonify({"error": "格式仅支持 docx / xlsx"}), 400

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT content, content_md FROM project_ai_memory WHERE id = %s AND project_id = %s",
                (memory_id, project_id)
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "记录不存在"}), 404

    md_text = row.get('content_md') or row.get('content') or ''
    if not md_text.strip():
        return jsonify({"error": "内容为空"}), 400

    try:
        from app.services.file_generator import generate_file
        file_data, filename, mime_type = generate_file(md_text, fmt, f"项目{project_id}_AI生成")
        from flask import send_file
        import io as io_module
        return send_file(
            io_module.BytesIO(file_data),
            mimetype=mime_type,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"File generation failed: {e}")
        return jsonify({"error": f"文件生成失败: {str(e)[:200]}"}), 500

@admin_bp.route('/admin/projects/<int:project_id>/my_workflow', methods=['GET'])
def get_my_workflow(project_id):
    """Get current member's custom workflow for this project."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "无权访问"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM member_workflows WHERE project_id=%s AND user_id=%s",
                (project_id, user_id))
            row = cur.fetchone()
    return jsonify({"workflow": dict(row) if row else None, "needs_setup": row is None})

@admin_bp.route('/admin/projects/<int:project_id>/my_workflow', methods=['POST'])
def save_my_workflow(project_id):
    """Save/update member's custom workflow steps."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "无权访问"}), 403
    data = request.get_json(silent=True) or {}
    steps = data.get('steps', [])
    name = data.get('name', '默认工作流').strip() or '默认工作流'
    if not steps or not isinstance(steps, list):
        return jsonify({"error": "请至少定义一个步骤"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO member_workflows (project_id, user_id, workflow_name, steps, updated_at)
                VALUES (%s,%s,%s,%s,NOW())
                ON CONFLICT (project_id, user_id)
                DO UPDATE SET workflow_name=%s, steps=%s, updated_at=NOW()
            """, (project_id, user_id, name, json.dumps(steps), name, json.dumps(steps)))
            conn.commit()
    return jsonify({"status": "ok"})

@admin_bp.route('/admin/projects/<int:project_id>/ai_workflow_step', methods=['POST'])
def project_ai_workflow_step(project_id):
    """Execute one step of the member's workflow interactively.
    Accepts: { query, step_index, step_action? }
    step_action: 'execute' | 'revise' (with revised_query) | 'approve'
    """
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    username = session.get('username', user_id)
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "无权访问"}), 403

    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    step_index = data.get('step_index', 0)
    step_action = data.get('step_action', 'execute')
    revised_query = data.get('revised_query', '').strip()

    # Load member's workflow
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT steps, workflow_name FROM member_workflows WHERE project_id=%s AND user_id=%s",
                (project_id, user_id))
            wf = cur.fetchone()
    if not wf:
        return jsonify({"error": "请先设置工作流"}), 400

    steps = wf['steps'] if isinstance(wf['steps'], list) else json.loads(wf['steps'])
    if step_index >= len(steps):
        return jsonify({"done": True, "message": "所有步骤已完成"})

    current_step = steps[step_index]
    step_name = current_step.get('step', f'步骤{step_index+1}')
    step_desc = current_step.get('desc', '')

    # Gather context
    from app.services.context_utils import gather_project_context as _gctx
    with get_db_connection() as conn:
        ctx = _gctx(conn, project_id, query or step_desc, user_id, username)

    workflow_section = f"\n=== 行业标准工作流 ===\n{ctx['workflow_section']}" if ctx['workflow_section'] else ''
    # Build steps context
    steps_context = '\n'.join([f"{i+1}. {s.get('step','?')}: {s.get('desc','')}" for i, s in enumerate(steps)])
    
    try:
        from app.services.llm_provider import call_llm

        if step_action == 'revise' and revised_query:
            prompt = f"""当前步骤: {step_name} - {step_desc}
工作流: {wf['workflow_name']}
完整步骤: {steps_context}
{workflow_section}

用户要求修改: {revised_query}
请根据修改意见重新生成当前步骤的内容。"""
            result = call_llm(prompt, revised_query, temperature=0.5, max_tokens=3200,
                             industry=ctx['proj_industry'])
        else:
            prompt = f"""当前步骤: {step_name} - {step_desc}
工作流: {wf['workflow_name']}
完整步骤: {steps_context}
{workflow_section}

请根据当前步骤的要求和项目上下文，生成该步骤的专业内容。
如果是第一步，这是用户初始需求: {query or step_desc}"""
            result = call_llm(prompt, f"执行步骤: {step_name}", temperature=0.5, max_tokens=3200,
                             industry=ctx['proj_industry'])

        # KPI: increment generation count
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO workflow_kpi (project_id, user_id, generations, output_chars, last_active)
                        VALUES (%s,%s,1,%s,NOW())
                        ON CONFLICT DO NOTHING
                    """, (project_id, user_id, len(result)))
                    cur.execute("""
                        UPDATE workflow_kpi SET generations=generations+1, output_chars=output_chars+%s,
                        last_active=NOW() WHERE project_id=%s AND user_id=%s
                    """, (len(result), project_id, user_id))
                    conn.commit()
        except Exception:
            pass

        # Check for overlap warnings
        overlap_warn = None
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT u.username, COUNT(*) as cnt FROM project_ai_memory pam
                        LEFT JOIN users u ON pam.user_id = u.user_id
                        WHERE pam.project_id=%s AND pam.user_id != %s AND pam.role='assistant'
                        AND pam.created_at > NOW() - INTERVAL '1 hour'
                        GROUP BY u.username
                    """, (project_id, user_id))
                    recent = cur.fetchall()
                    if recent:
                        overlap_warn = '⚠️ 近1小时同事也生成了内容: ' + ', '.join(
                            [f"@{r['username']}({r['cnt']}次)" for r in recent[:3]])
        except Exception:
            pass

        resp = {
            "status": "ok",
            "step_index": step_index,
            "step_name": step_name,
            "result": result,
            "total_steps": len(steps),
            "next_step": step_index + 1 if step_index + 1 < len(steps) else None,
        }
        if overlap_warn:
            resp["warning"] = overlap_warn
        return jsonify(resp)

    except Exception as e:
        logger.error(f"Workflow step error: {e}")
        return jsonify({"error": f"执行失败: {str(e)[:200]}"}), 500

@admin_bp.route('/admin/projects/<int:project_id>/workflow_kpi', methods=['GET'])
@admin_required
def project_workflow_kpi(project_id):
    """Get KPI stats for all project members."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "无权访问"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT wk.*, u.username FROM workflow_kpi wk
                LEFT JOIN users u ON wk.user_id = u.user_id
                WHERE wk.project_id = %s ORDER BY wk.generations DESC
            """, (project_id,))
            rows = cur.fetchall()
    return jsonify({"kpi": [dict(r) for r in rows]})

@admin_bp.route('/admin/projects/<int:project_id>/ai_workflow', methods=['POST'])
def project_ai_workflow(project_id):
    """Multi-step document workflow: draft → review → revise → finalize."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "无权访问"}), 403

    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    if not query or len(query) < 3:
        return jsonify({"error": "请描述您需要起草的文档"}), 400

    # Get project industry
    proj_industry = 'general'
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT industry FROM projects WHERE id = %s", (project_id,))
                row = cur.fetchone()
                if row:
                    proj_industry = row[0] or 'general'
    except Exception:
        pass

    try:
        from app.services.workflow_engine import run_document_workflow
        result = run_document_workflow(query, industry=proj_industry)
        return jsonify({"status": "ok", **result})
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return jsonify({"error": f"工作流执行失败: {str(e)[:200]}"}), 500

@admin_bp.route('/admin/projects/<int:project_id>/ai_analyze', methods=['POST'])
def project_ai_analyze(project_id):
    """Data analysis: upload Excel/CSV → pandas analysis → comparison matrix + anomaly report."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return jsonify({"error": "无权访问"}), 403

    file = request.files.get('file')
    if not file:
        return jsonify({"error": "请上传Excel或CSV文件"}), 400

    try:
        import pandas as pd
        import numpy as np
        from io import BytesIO

        filename = file.filename.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(file.read()))
        else:
            df = pd.read_excel(BytesIO(file.read()))

        if df.empty or len(df.columns) < 2:
            return jsonify({"error": "文件需要包含至少两列数据"}), 400

        result = {
            "rows": len(df),
            "columns": len(df.columns),
            "columns_list": list(df.columns),
        }

        # Detect numeric columns for comparison
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()

        if len(numeric_cols) >= 1 and len(text_cols) >= 1:
            # Comparison matrix: text_col as index, first numeric as values
            pivot_col = text_cols[0]
            val_col = numeric_cols[0]
            top_n = min(20, len(df))
            top_rows = df.nlargest(top_n, val_col) if val_col in df else df.head(top_n)
            comparison = [{"name": str(r[pivot_col])[:60], "value": float(r[val_col]) if pd.notna(r[val_col]) else 0}
                         for _, r in top_rows.iterrows()]
            result["comparison"] = comparison
            result["comparison_key"] = pivot_col
            result["comparison_value"] = val_col

        # Anomaly detection on numeric columns
        if len(numeric_cols) >= 1:
            from sklearn.ensemble import IsolationForest
            num_data = df[numeric_cols].fillna(0)
            if len(num_data) >= 3:
                clf = IsolationForest(contamination=0.1, random_state=42)
                preds = clf.fit_predict(num_data)
                anomalies = [i for i, p in enumerate(preds) if p == -1]
                result["anomalies_count"] = len(anomalies)
                if anomalies and text_cols:
                    result["anomalies"] = [
                        {"row": i, "label": str(df.iloc[i][text_cols[0]])[:60]}
                        for i in anomalies[:15]
                    ]

        # Basic stats
        if numeric_cols:
            result["stats"] = {}
            for col in numeric_cols[:5]:
                result["stats"][col] = {
                    "mean": round(float(df[col].mean()), 2),
                    "max": round(float(df[col].max()), 2),
                    "min": round(float(df[col].min()), 2),
                    "sum": round(float(df[col].sum()), 2),
                }

        return jsonify({"status": "ok", "analysis": result})
    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
        return jsonify({"error": f"分析失败: {str(e)[:200]}"}), 500

@admin_bp.route('/admin/projects/<int:project_id>/ai_memory', methods=['POST'])
def project_ai_sync_chat(project_id):
    """Sync a chat message from the project chat tab into AI memory.
    Called when user sends a message in a project-scoped chat session.
    Accepts: { role, content }
    """
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "未登录"}), 401
    
    data = request.get_json(silent=True) or {}
    role = data.get('role', 'user')
    content = data.get('content', '').strip()
    if not content or len(content) < 2:
        return jsonify({"status": "skipped"})

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO project_ai_memory (project_id, user_id, role, content) VALUES (%s,%s,%s,%s)",
                    (project_id, user_id, role, content[:2000])
                )
                conn.commit()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500

@admin_bp.route('/admin/analytics', methods=['GET'])
def admin_analytics():
    """Return usage statistics — admin sees all users, regular users see own stats."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    admin_view = is_admin()
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            stats = {}

            if admin_view:
                cur.execute("SELECT COUNT(*) as cnt FROM users")
                stats['total_users'] = cur.fetchone()['cnt']
                cur.execute("SELECT COUNT(DISTINCT cs.user_id) as cnt FROM chat_messages cm JOIN chat_sessions cs ON cm.thread_id = cs.thread_id WHERE cm.timestamp > NOW() - INTERVAL '24 hours'")
                stats['active_users_24h'] = cur.fetchone()['cnt']
            else:
                stats['total_users'] = 1
                stats['active_users_24h'] = 1

            if admin_view:
                cur.execute("SELECT COUNT(*) as cnt FROM chat_sessions cs")
                stats['total_sessions'] = cur.fetchone()['cnt']
                cur.execute("SELECT COUNT(*) as cnt FROM chat_messages cm JOIN chat_sessions cs ON cm.thread_id = cs.thread_id")
                stats['total_messages'] = cur.fetchone()['cnt']
                cur.execute("SELECT COUNT(*) as cnt FROM chat_messages cm JOIN chat_sessions cs ON cm.thread_id = cs.thread_id WHERE cm.timestamp > NOW() - INTERVAL '24 hours'")
                stats['messages_today'] = cur.fetchone()['cnt']
                cur.execute("SELECT COUNT(*) as cnt FROM user_files")
                stats['total_files'] = cur.fetchone()['cnt']
                cur.execute("SELECT COALESCE(SUM(size_bytes), 0) as total FROM user_files")
                stats['storage_mb'] = round(cur.fetchone()['total'] / (1024 * 1024), 1)
                cur.execute("SELECT COUNT(*) as cnt FROM credit_check_reports")
                stats['credit_checks'] = cur.fetchone()['cnt']
                cur.execute("""
                    SELECT DATE(cm.timestamp) as day, COUNT(*) as cnt
                    FROM chat_messages cm JOIN chat_sessions cs ON cm.thread_id = cs.thread_id
                    WHERE cm.timestamp > NOW() - INTERVAL '7 days'
                    GROUP BY DATE(cm.timestamp) ORDER BY day
                """)
                stats['messages_per_day'] = [{'day': str(r['day']), 'count': r['cnt']} for r in cur.fetchall()]
                cur.execute("SELECT COUNT(*) as cnt FROM projects p JOIN project_members pm ON p.id = pm.project_id WHERE p.status = 'active'")
                stats['active_projects'] = cur.fetchone()['cnt']
            else:
                cur.execute("SELECT COUNT(*) as cnt FROM chat_sessions cs WHERE cs.user_id = %s", (user_id,))
                stats['total_sessions'] = cur.fetchone()['cnt']
                cur.execute("SELECT COUNT(*) as cnt FROM chat_messages cm JOIN chat_sessions cs ON cm.thread_id = cs.thread_id WHERE cs.user_id = %s", (user_id,))
                stats['total_messages'] = cur.fetchone()['cnt']
                cur.execute("SELECT COUNT(*) as cnt FROM chat_messages cm JOIN chat_sessions cs ON cm.thread_id = cs.thread_id WHERE cm.timestamp > NOW() - INTERVAL '24 hours' AND cs.user_id = %s", (user_id,))
                stats['messages_today'] = cur.fetchone()['cnt']
                cur.execute("SELECT COUNT(*) as cnt FROM user_files WHERE user_id = %s", (user_id,))
                stats['total_files'] = cur.fetchone()['cnt']
                cur.execute("SELECT COALESCE(SUM(size_bytes), 0) as total FROM user_files WHERE user_id = %s", (user_id,))
                stats['storage_mb'] = round(cur.fetchone()['total'] / (1024 * 1024), 1)
                cur.execute("SELECT COUNT(*) as cnt FROM credit_check_reports WHERE user_id = %s", (user_id,))
                stats['credit_checks'] = cur.fetchone()['cnt']
                cur.execute("""
                    SELECT DATE(cm.timestamp) as day, COUNT(*) as cnt
                    FROM chat_messages cm JOIN chat_sessions cs ON cm.thread_id = cs.thread_id
                    WHERE cm.timestamp > NOW() - INTERVAL '7 days' AND cs.user_id = %s
                    GROUP BY DATE(cm.timestamp) ORDER BY day
                """, (user_id,))
                stats['messages_per_day'] = [{'day': str(r['day']), 'count': r['cnt']} for r in cur.fetchall()]
                cur.execute("SELECT COUNT(*) as cnt FROM projects p JOIN project_members pm ON p.id = pm.project_id WHERE p.status = 'active' AND pm.user_id = %s", (user_id,))
                stats['active_projects'] = cur.fetchone()['cnt']

            # Admin-only: storage breakdown + top users
            if admin_view:
                breakdown = {}
                for label, query in [
                    ('聊天文件', "SELECT COALESCE(SUM(size_bytes),0)::float FROM user_files"),
                    ('知识库', "SELECT COALESCE(SUM(file_size),0)::float FROM knowledge_lab_files"),
                    ('公司库', "SELECT COALESCE(SUM(file_size),0)::float FROM company_knowledge_base"),
                    ('项目文件', "SELECT COALESCE(SUM(file_size),0)::float FROM project_files"),
                ]:
                    cur.execute(query)
                    val = cur.fetchone()
                    breakdown[label] = round(float(list(val.values())[0]) / (1024 * 1024), 1) if val else 0
                stats['storage_breakdown'] = breakdown

                cur.execute("""
                    SELECT u.username, COUNT(uf.id) as file_count,
                           COALESCE(SUM(uf.size_bytes),0) as total_bytes
                    FROM users u LEFT JOIN user_files uf ON u.user_id = uf.user_id
                    WHERE u.is_active = TRUE
                    GROUP BY u.username ORDER BY total_bytes DESC LIMIT 10
                """)
                top = []
                for r in cur.fetchall():
                    top.append({'username': r['username'], 'files': r['file_count'],
                                'storage_mb': round(r['total_bytes']/(1024*1024), 1)})
                stats['top_users'] = top

                try:
                    from app.services.rag_engine import get_index_stats
                    stats['rag_stats'] = get_index_stats()
                except Exception:
                    stats['rag_stats'] = {}

            stats['is_admin_view'] = admin_view
    return jsonify(stats)

@admin_bp.route('/admin/audit_log', methods=['GET'])
@admin_required
def admin_audit_log():
    page = request.args.get('page', 1, type=int)
    per_page = 50
    offset = (page - 1) * per_page
    search = request.args.get('search', '').strip()
    action_filter = request.args.get('action', '').strip()
    success_filter = request.args.get('success', '').strip()

    where_clauses = []
    params = []
    if search:
        where_clauses.append(
            "(action ILIKE %s OR table_name ILIKE %s OR admin_username ILIKE %s OR CAST(row_id AS TEXT) ILIKE %s)")
        params.extend([f"%{search}%"] * 4)
    if action_filter in ('UPDATE', 'DELETE'):
        where_clauses.append("action = %s")
        params.append(action_filter)
    if success_filter in ('true', 'false'):
        where_clauses.append("success = %s")
        params.append(success_filter == 'true')

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT * FROM admin_audit_log
                {where_sql}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, params + [per_page, offset])
            logs = cur.fetchall()
            cur.execute(f"SELECT COUNT(*) as total FROM admin_audit_log {where_sql}", params)
            total = cur.fetchone()['total']
    return jsonify({
        "logs": logs,
        "total": total,
        "page": page,
        "per_page": per_page
    })


@admin_bp.route('/admin/audit_note', methods=['POST'])
@admin_required
def admin_audit_note():
    """Add a manual note (e.g. hardware maintenance) to the audit log."""
    from app.services.admin_utils import log_admin_action
    data = request.get_json(silent=True) or {}
    note = (data.get('note', '') or '').strip()
    if not note:
        return jsonify({"error": "备注不能为空"}), 400
    log_admin_action(
        session.get('user_id', ''),
        session.get('username', ''),
        'ADMIN_NOTE', 'system', None,
        column_name='note', new_value=note[:500]
    )
    return jsonify({"status": "ok"})


@admin_bp.route('/admin/approve_delete/<username>', methods=['POST'])
@admin_required
def admin_approve_delete(username):
    """Admin approves a user's deletion request, sends 4-digit code to user."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT user_id, email, deletion_requested FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "用户不存在"}), 404
            if not row['deletion_requested']:
                return jsonify({"error": "该用户未申请删除"}), 400
            user_email = row.get('email', '')
            code = f"{random.randint(1000, 9999)}"
            cur.execute("UPDATE users SET deletion_code = %s WHERE username = %s", (code, username))
            conn.commit()
    from app.utils.mailer import send_email, is_configured
    from app.services.admin_utils import log_admin_action
    admin_uid = session.get('user_id', '')
    admin_uname = session.get('username', '')
    log_admin_action(admin_uid, admin_uname, 'DELETE_APPROVE', 'users', username,
                    column_name='deletion_requested', old_value='pending', new_value=f'code_sent_{code}')
    if is_configured() and user_email:
        send_email(user_email, "[中联AI] 账户删除验证码",
                   f"验证码: {code}\n有效5分钟。输入此码确认删除账户。", async_mode=True)
    return jsonify({"status": "ok", "hint": f"验证码{'已发送至 '+user_email if user_email else ': '+code}"})


@admin_bp.route('/admin/pending_deletions', methods=['GET'])
@admin_required
def admin_pending_deletions():
    """List users with pending deletion requests."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT username, email, deletion_requested FROM users WHERE deletion_requested = TRUE")
            users = cur.fetchall()
    return jsonify({"users": users})


# ── User Assets Overview ──

@admin_bp.route('/admin/user_assets', methods=['GET'])
@admin_required
def admin_user_assets():
    """Return all registered users with their digital asset inventory + deposit items."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT user_id, username, email, role, is_auditor, created_at FROM users WHERE is_active = TRUE AND role != 'admin' ORDER BY username")
            users = [dict(r) for r in cur.fetchall()]
            user_ids = [u['user_id'] for u in users]
            for u in users:
                uid = u['user_id']
                cur.execute("SELECT COUNT(*) as cnt FROM user_files WHERE user_id = %s", (uid,)); u['chat_files'] = cur.fetchone()['cnt']
                cur.execute("SELECT COALESCE(SUM(size_bytes),0) as s FROM user_files WHERE user_id = %s", (uid,)); u['chat_mb'] = round(cur.fetchone()['s']/(1024*1024),1)
                cur.execute("SELECT COUNT(*) as cnt FROM knowledge_lab_files WHERE user_id = %s", (uid,)); u['kb_files'] = cur.fetchone()['cnt']
                cur.execute("SELECT COUNT(*) as cnt FROM credit_check_reports WHERE user_id = %s", (uid,)); u['credit_reports'] = cur.fetchone()['cnt']
                cur.execute("SELECT COUNT(*) as cnt FROM batch_comparison_results WHERE user_id = %s", (uid,)); u['batch_results'] = cur.fetchone()['cnt']
                cur.execute("SELECT COUNT(*) as cnt FROM chat_sessions WHERE user_id = %s", (uid,)); u['sessions'] = cur.fetchone()['cnt']
                cur.execute("SELECT COUNT(*) as cnt FROM project_members WHERE user_id = %s AND status = 'active'", (uid,)); u['projects'] = cur.fetchone()['cnt']
                u['total'] = u['chat_files'] + u['kb_files'] + u['credit_reports'] + u['batch_results']
            # Deposit items
            cur.execute("""SELECT id, original_username, item_type, item_data, stored_path, created_at
                FROM task_deposit_items WHERE deleted_at IS NULL AND transferred_to_user_id IS NULL ORDER BY created_at DESC""")
            deposits = [dict(r) for r in cur.fetchall()]
    return jsonify({"users": users, "deposits": deposits})


@admin_bp.route('/admin/transfer_assets', methods=['POST'])
@admin_required
def admin_transfer_assets():
    """Bulk transfer assets from users/deposit to a target user."""
    from app.services.admin_utils import log_admin_action
    data = request.get_json(silent=True) or {}
    target_user_id = (data.get('target_user_id') or '').strip()
    source_user_ids = data.get('source_user_ids', [])
    deposit_ids = data.get('deposit_ids', [])
    types = data.get('types', ['all'])
    if not target_user_id:
        return jsonify({"error": "Missing target user"}), 400
    admin_uid = session.get('user_id', ''); admin_uname = session.get('username', '')
    transferred = 0
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users WHERE user_id = %s", (target_user_id,))
                if not cur.fetchone(): return jsonify({"error": "Target user not found"}), 404
                for src in source_user_ids:
                    if 'all' in types or 'chat_files' in types:
                        cur.execute("UPDATE user_files SET user_id = %s WHERE user_id = %s", (target_user_id, src)); transferred += cur.rowcount
                    if 'all' in types or 'kb_files' in types:
                        cur.execute("UPDATE knowledge_lab_files SET user_id = %s WHERE user_id = %s", (target_user_id, src)); transferred += cur.rowcount
                    if 'all' in types or 'credit_reports' in types:
                        cur.execute("UPDATE credit_check_reports SET user_id = %s WHERE user_id = %s", (target_user_id, src)); transferred += cur.rowcount
                    if 'all' in types or 'batch_results' in types:
                        cur.execute("UPDATE batch_comparison_results SET user_id = %s WHERE user_id = %s", (target_user_id, src)); transferred += cur.rowcount
                for did in deposit_ids:
                    cur.execute("UPDATE task_deposit_items SET transferred_to_user_id=%s, transferred_at=NOW() WHERE id=%s AND transferred_to_user_id IS NULL", (target_user_id, did)); transferred += cur.rowcount
                conn.commit()
    log_admin_action(admin_uid, admin_uname, 'ASSET_TRANSFER', 'users', target_user_id,
                    column_name='bulk_transfer', old_value=f'{len(source_user_ids)}src+{len(deposit_ids)}dep', new_value=f'{transferred}items')
    return jsonify({"status": "ok", "transferred": transferred})


# ── System Prompt Management ──

@admin_bp.route('/admin/system_prompt', methods=['GET'])
@admin_required
def get_system_prompt():
    """Return the current agent system prompt (admin only)."""
    from app.globals import AGENT_SYSTEM_PROMPT
    return jsonify({"prompt": AGENT_SYSTEM_PROMPT.strip()})


@admin_bp.route('/admin/system_prompt', methods=['POST'])
@admin_required
def set_system_prompt():
    """Update the agent system prompt and persist to disk."""
    from app.globals import save_prompt, AGENT_SYSTEM_PROMPT as _current
    from app.services.admin_utils import log_admin_action
    data = request.get_json(silent=True) or {}
    new_prompt = (data.get('prompt', '') or '').strip()
    if not new_prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400
    if new_prompt == _current.strip():
        return jsonify({"status": "ok", "message": "No changes"})
    save_prompt(new_prompt)
    log_admin_action(session.get('user_id', ''), session.get('username', ''),
                    'PROMPT_EDIT', 'system', None, column_name='agent_system_prompt',
                    old_value=_current[:80] + '...', new_value=new_prompt[:80] + '...')
    return jsonify({"status": "ok", "message": "System prompt updated"})


# ── Search Cache Config ──

@admin_bp.route('/admin/search_cache_config', methods=['GET'])
@admin_required
def get_search_cache_config():
    """Return search cache config and stats."""
    from app.services.agent import get_cache_stats
    return jsonify({"status": "ok", "config": get_cache_stats()})


@admin_bp.route('/admin/search_cache_config', methods=['POST'])
@admin_required
def set_search_cache_config():
    """Update search cache TTL or clear cache."""
    from app.services.agent import _set_cache_ttl, clear_search_cache, get_cache_ttl
    from app.services.admin_utils import log_admin_action
    data = request.get_json(silent=True) or {}
    action = data.get('action', 'set_ttl')

    if action == 'clear':
        clear_search_cache()
        log_admin_action(session.get('user_id', ''), session.get('username', ''),
                        'CACHE_CLEAR', 'system', None,
                        column_name='search_cache', new_value='cleared')
        return jsonify({"status": "ok", "message": "搜索缓存已清除"})

    # set_ttl
    ttl_hours = data.get('ttl_hours')
    if ttl_hours is None:
        return jsonify({"error": "缺少 ttl_hours 参数"}), 400
    try:
        ttl_hours = float(ttl_hours)
    except (ValueError, TypeError):
        return jsonify({"error": "ttl_hours 必须是数字"}), 400
    if ttl_hours < 0:
        return jsonify({"error": "TTL 不能为负数（设为0表示禁用缓存）"}), 400

    old_ttl_hours = get_cache_ttl() / 3600
    _set_cache_ttl(int(ttl_hours * 3600))
    log_admin_action(session.get('user_id', ''), session.get('username', ''),
                    'CACHE_TTL_CHANGE', 'system', None,
                    column_name='search_cache_ttl',
                    old_value=f'{old_ttl_hours}h', new_value=f'{ttl_hours}h')
    return jsonify({"status": "ok", "message": f"搜索缓存 TTL 已设为 {ttl_hours} 小时"})


# ── Unified Runtime Config ──

@admin_bp.route('/admin/runtime_config', methods=['GET'])
@admin_required
def get_runtime_config():
    """Return all runtime-adjustable config values."""
    from app.services.runtime_config import get_all
    return jsonify({"status": "ok", "config": get_all()})


@admin_bp.route('/admin/runtime_config', methods=['POST'])
@admin_required
def update_runtime_config():
    """Update one or more runtime config values."""
    from app.services.runtime_config import update, reset_to_defaults, save_factory_presets, restore_factory_presets, has_factory_presets
    from app.services.admin_utils import log_admin_action
    data = request.get_json(silent=True) or {}
    action = data.pop('_action', 'update')

    if action == 'reset':
        cfg = reset_to_defaults()
        log_admin_action(session.get('user_id', ''), session.get('username', ''),
                        'CONFIG_RESET', 'system', None,
                        column_name='runtime_config', new_value='reset_to_defaults')
        return jsonify({"status": "ok", "message": "Config reset to defaults", "config": cfg})

    if action == 'save_factory':
        if has_factory_presets():
            return jsonify({"error": "Factory presets already saved — cannot overwrite"}), 409
        factory = save_factory_presets()
        log_admin_action(session.get('user_id', ''), session.get('username', ''),
                        'CONFIG_FACTORY_SAVE', 'system', None,
                        column_name='runtime_config', new_value=f'factory_saved:{len(factory)}keys')
        return jsonify({"status": "ok", "message": f"Factory presets saved ({len(factory)} keys, read-only)", "factory": factory})

    if action == 'restore_factory':
        if not has_factory_presets():
            return jsonify({"error": "No factory presets exist — save factory first"}), 400
        cfg = restore_factory_presets()
        log_admin_action(session.get('user_id', ''), session.get('username', ''),
                        'CONFIG_RESTORE_FACTORY', 'system', None,
                        column_name='runtime_config', new_value='restored_to_factory')
        return jsonify({"status": "ok", "message": "Restored to factory presets", "config": cfg})

    if not data:
        return jsonify({"error": "No update parameters provided"}), 400

    # Validate types against defaults
    from app.services.runtime_config import DEFAULTS
    sanitized = {}
    for k, v in data.items():
        if k not in DEFAULTS:
            continue
        expected_type = type(DEFAULTS[k])
        try:
            if expected_type is int and isinstance(v, float):
                sanitized[k] = int(v)
            elif expected_type is bool:
                sanitized[k] = bool(v)
            else:
                sanitized[k] = expected_type(v)
        except (ValueError, TypeError):
            return jsonify({"error": f"Type mismatch for {k}: expected {expected_type.__name__}"}), 400

    cfg = update(sanitized)
    changed = ', '.join(f'{k}={v}' for k, v in sanitized.items())
    log_admin_action(session.get('user_id', ''), session.get('username', ''),
                    'CONFIG_UPDATE', 'system', None,
                    column_name='runtime_config', new_value=changed[:200])

    # Invalidate agent cache if LLM provider/model changed
    if 'active_llm_provider' in sanitized or 'active_llm_model' in sanitized:
        from app import globals as g
        with g._agent_lock:
            g._agent = None
            g._current_max_tokens = None
        logger.info("Agent cache invalidated due to LLM config change")

    return jsonify({"status": "ok", "message": f"Updated {len(sanitized)} config keys", "config": cfg})


@admin_bp.route('/admin/embedding_cache', methods=['GET'])
@admin_required
def get_embedding_cache():
    """Return embedding cache stats for admin monitoring."""
    try:
        from app.services.rag_engine import embedding_cache_stats
        return jsonify(embedding_cache_stats())
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500

@admin_bp.route('/admin/embedding_cache/clear', methods=['POST'])
@admin_required
def clear_embedding_cache_route():
    """Clear the embedding cache (useful after model update)."""
    try:
        from app.services.rag_engine import clear_embedding_cache
        clear_embedding_cache()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500

# ── DB Migration Management ──

@admin_bp.route('/admin/db_migrations', methods=['GET'])
@admin_required
def get_db_migrations():
    """Return pending migrations and history."""
    try:
        import subprocess, os, sys
        scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        result = subprocess.run(
            [sys.executable, os.path.join(scripts_dir, 'manage_db.py'), 'check'],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, 'PYTHONPATH': os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}
        )
        # Also get history
        hist_result = subprocess.run(
            [sys.executable, os.path.join(scripts_dir, 'manage_db.py'), 'history'],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, 'PYTHONPATH': os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}
        )
        return jsonify({
            "pending": result.stdout,
            "history": hist_result.stdout,
            "error": result.stderr if result.stderr else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500

@admin_bp.route('/admin/db_migrations/apply', methods=['POST'])
@admin_required
def apply_db_migrations():
    """Apply all safe pending migrations. Pass ?force=1 to apply risky ones too."""
    try:
        import subprocess, os, sys
        scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        force = request.args.get('force') == '1'
        cmd = [sys.executable, os.path.join(scripts_dir, 'manage_db.py'), 'migrate']
        if force:
            cmd.append('--yes')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
            env={**os.environ, 'PYTHONPATH': os.path.dirname(os.path.dirname(os.path.dirname(__file__)))})
        return jsonify({
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.stderr else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500

@admin_bp.route('/admin/db_migrations/rollback', methods=['POST'])
@admin_required
def rollback_db_migration():
    """Rollback the last migration."""
    try:
        import subprocess, os, sys
        scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        result = subprocess.run(
            [sys.executable, os.path.join(scripts_dir, 'manage_db.py'), 'rollback'],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, 'PYTHONPATH': os.path.dirname(os.path.dirname(os.path.dirname(__file__)))})
        return jsonify({
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.stderr else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500

@admin_bp.route('/admin/db_migrations/snapshot', methods=['POST'])
@admin_required
def snapshot_db_schema():
    """Capture current DB schema snapshot."""
    try:
        import subprocess, os, sys
        scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        result = subprocess.run(
            [sys.executable, os.path.join(scripts_dir, 'manage_db.py'), 'snapshot'],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, 'PYTHONPATH': os.path.dirname(os.path.dirname(os.path.dirname(__file__)))})
        return jsonify({
            "success": True,
            "output": result.stdout,
        })
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500

@admin_bp.route('/admin/runtime_config_schema', methods=['GET'])
@admin_required
def get_runtime_config_schema():
    """Return metadata about each config key (for building smart UI)."""
    # Build dynamic model list from all configured providers
    from app.services.llm_provider import PROVIDER_CONFIG
    all_models = ['auto']
    model_labels = {'auto': '自动(服务商默认)'}
    for pid, cfg in PROVIDER_CONFIG.items():
        for m in cfg.get('models', []):
            label = f"{m} ({cfg['name']})"
            all_models.append(m)
            model_labels[m] = label

    schema = {
        # ── LLM ──
        "active_llm_provider":        {"label": "LLM 服务商", "unit": "", "type": "select", "group": "LLM/AI Model", "is_llm": True, "is_not_factory": True,
                                         "options": ["auto", "deepseek", "zhipu", "qwen", "siliconflow"],
                                         "option_labels": {"auto": "自动检测", "deepseek": "DeepSeek", "zhipu": "智谱AI", "qwen": "Qwen", "siliconflow": "硅基流动"}},
        "active_llm_model":           {"label": "LLM 模型", "unit": "", "type": "select", "group": "LLM/AI Model", "is_llm": True, "is_not_factory": True,
                                         "options": all_models,
                                         "option_labels": model_labels},
        "llm_timeout_seconds":        {"label": "LLM 请求超时", "unit": "秒", "type": "int", "group": "LLM/AI Model", "min": 10, "max": 600},
        "llm_max_tokens":             {"label": "LLM 默认最大Token", "unit": "tokens", "type": "int", "group": "LLM/AI Model", "min": 50, "max": 8192},
        "llm_temperature":            {"label": "LLM 温度", "unit": "", "type": "float", "group": "LLM/AI Model", "min": 0, "max": 2, "step": 0.05},
        "llm_batch_timeout_seconds":  {"label": "批量对比超时", "unit": "秒", "type": "int", "group": "LLM/AI Model", "min": 10, "max": 600},
        "llm_max_tokens_min":         {"label": "用户最小Token", "unit": "tokens", "type": "int", "group": "LLM/AI Model", "min": 1, "max": 500},
        "llm_max_tokens_max":         {"label": "用户最大Token", "unit": "tokens", "type": "int", "group": "LLM/AI Model", "min": 500, "max": 16384},
        # ── Search cache ──
        "search_cache_ttl_hours":     {"label": "搜索缓存有效期", "unit": "小时", "type": "float", "group": "Search & Cache", "min": 0, "max": 720, "step": 0.5},
        "headroom_enabled":           {"label": "Headroom 压缩", "unit": "", "type": "bool", "group": "Search & Cache"},
        "judge_review_enabled":       {"label": "Judge 审查模型", "unit": "", "type": "bool", "group": "LLM/AI Model"},
        # ── VL model ──
        "vl_max_image_size":          {"label": "VL 最大图片尺寸", "unit": "px", "type": "int", "group": "LLM/AI Model", "min": 128, "max": 4096},
        "vl_jpeg_quality":            {"label": "JPEG 质量", "unit": "%", "type": "int", "group": "LLM/AI Model", "min": 10, "max": 100},
        "vl_max_tokens":              {"label": "VL 最大Token", "unit": "tokens", "type": "int", "group": "LLM/AI Model", "min": 50, "max": 4096},
        "vl_temperature":             {"label": "VL 温度", "unit": "", "type": "float", "group": "LLM/AI Model", "min": 0, "max": 2, "step": 0.05},
        # ── RAG ──
        "rag_chunk_size":             {"label": "RAG 分块大小", "unit": "字符", "type": "int", "group": "RAG Engine", "min": 50, "max": 5000},
        "rag_chunk_overlap":          {"label": "RAG 分块重叠", "unit": "字符", "type": "int", "group": "RAG Engine", "min": 0, "max": 1000},
        "rag_top_k_default":          {"label": "RAG 默认Top-K", "unit": "条", "type": "int", "group": "RAG Engine", "min": 1, "max": 50},
        "rag_max_context_chars":      {"label": "RAG 最大上下文", "unit": "字符", "type": "int", "group": "RAG Engine", "min": 500, "max": 50000},
        "rag_min_chunk_chars":        {"label": "RAG 最小块大小", "unit": "字符", "type": "int", "group": "RAG Engine", "min": 5, "max": 200},
        # ── File cache ──
        "file_cache_max_age_hours":   {"label": "文件缓存有效期", "unit": "小时", "type": "float", "group": "File Processing", "min": 0, "max": 720},
        "file_cache_max_cached_files":{"label": "最大缓存文件数", "unit": "个", "type": "int", "group": "File Processing", "min": 1, "max": 100},
        "file_cache_max_content_size":{"label": "缓存内容上限", "unit": "byte", "type": "int", "group": "File Processing", "min": 1024, "max": 1048576},
        # ── File processing ──
        "file_template_similarity_threshold": {"label": "模板相似度阈值", "unit": "", "type": "float", "group": "File Processing", "min": 0.1, "max": 1.0, "step": 0.01},
        "file_keywords_top_k":        {"label": "关键词数量", "unit": "个", "type": "int", "group": "File Processing", "min": 1, "max": 100},
        "file_semantic_batch_size":   {"label": "语义批处理大小", "unit": "条", "type": "int", "group": "File Processing", "min": 1, "max": 256},
        "file_ocr_zoom":              {"label": "OCR 渲染缩放", "unit": "倍", "type": "float", "group": "File Processing", "min": 0.5, "max": 5, "step": 0.1},
        "file_ocr_max_dim":           {"label": "OCR 最大图片尺寸", "unit": "px", "type": "int", "group": "File Processing", "min": 256, "max": 5000},
        "file_name_max_len":          {"label": "文件名截断长度", "unit": "字符", "type": "int", "group": "File Processing", "min": 5, "max": 200},
        # ── Session ──
        "session_title_max_len":      {"label": "会话标题长度", "unit": "字符", "type": "int", "group": "Session & Messages", "min": 5, "max": 100},
        # ── Cleanup ──
        "cleanup_session_days":       {"label": "会话保留天数", "unit": "天", "type": "int", "group": "Auto Cleanup", "min": 1, "max": 365},
        "cleanup_anon_temp_days":     {"label": "匿名临时文件保留", "unit": "天", "type": "int", "group": "Auto Cleanup", "min": 0, "max": 30},
        "cleanup_project_deletion_days": {"label": "项目删除宽限期", "unit": "天", "type": "int", "group": "Auto Cleanup", "min": 1, "max": 365},
        "cleanup_share_file_days":    {"label": "分享文件保留", "unit": "天", "type": "int", "group": "Auto Cleanup", "min": 1, "max": 365},
        "cleanup_download_token_hours":{"label": "下载Token有效期", "unit": "小时", "type": "int", "group": "Auto Cleanup", "min": 1, "max": 720},
        "cleanup_report_retention_days":{"label": "自动报告保留", "unit": "天", "type": "int", "group": "Auto Cleanup", "min": 1, "max": 365},
        "cleanup_recycle_bin_days":   {"label": "回收站保留", "unit": "天", "type": "int", "group": "Auto Cleanup", "min": 1, "max": 90},
        "cleanup_original_file_days": {"label": "原始文件保留", "unit": "天", "type": "int", "group": "Auto Cleanup", "min": 0, "max": 90},
        "cleanup_message_response_hours":{"label": "待响应超时", "unit": "小时", "type": "int", "group": "Auto Cleanup", "min": 0, "max": 72},
        # ── Rate limits ──
        "ratelimit_admin_max":        {"label": "管理员频率限制", "unit": "次", "type": "int", "group": "Rate Limits", "min": 1, "max": 100},
        "ratelimit_admin_window_seconds":{"label": "管理员频率窗口", "unit": "秒", "type": "int", "group": "Rate Limits", "min": 60, "max": 86400},
        "ratelimit_credit_max":       {"label": "征信查询频率限制", "unit": "次", "type": "int", "group": "Rate Limits", "min": 1, "max": 100},
        "ratelimit_credit_window_seconds":{"label": "征信查询频率窗口", "unit": "秒", "type": "int", "group": "Rate Limits", "min": 60, "max": 86400},
        # ── Anonymous ──
        "anon_max_files":             {"label": "匿名最大文件数", "unit": "个", "type": "int", "group": "Anonymous Limits", "min": 0, "max": 50},
        "anon_max_file_size_mb":      {"label": "匿名文件大小限制", "unit": "MB", "type": "float", "group": "Anonymous Limits", "min": 0.1, "max": 50, "step": 0.5},
        "anon_message_max_chars":     {"label": "最大消息长度", "unit": "字符", "type": "int", "group": "Anonymous Limits", "min": 100, "max": 100000},
        "storage_warn_threshold_mb":  {"label": "存储警告阈值", "unit": "MB", "type": "int", "group": "Anonymous Limits", "min": 10, "max": 10000},
        # ── Training ──
        "training_min_rating":        {"label": "训练最低评分", "unit": "星", "type": "int", "group": "Training Data", "min": 0, "max": 5},
        "training_min_length":        {"label": "训练最低长度", "unit": "字符", "type": "int", "group": "Training Data", "min": 10, "max": 10000},
        "training_retention_days":    {"label": "训练数据保留", "unit": "天", "type": "int", "group": "Training Data", "min": 7, "max": 730},
        "export_retention_count":     {"label": "导出文件保留数", "unit": "个", "type": "int", "group": "Training Data", "min": 3, "max": 200},
        # ── Report ──
        "report_min_messages":        {"label": "自动报告最低消息数", "unit": "条", "type": "int", "group": "Auto Reports", "min": 1, "max": 1000},
        # ── Web extractor ──
        "web_extract_retries":        {"label": "Web Extract Retries", "unit": "retries", "type": "int", "group": "File Processing", "min": 0, "max": 10},
        "web_extract_timeout_seconds":{"label": "Web Extract Timeout", "unit": "sec", "type": "int", "group": "File Processing", "min": 5, "max": 120},
        # ── Upload ──
        "max_upload_size_mb":         {"label": "Max Upload Size (info)", "unit": "MB", "type": "int", "group": "File Processing", "min": 1, "max": 500},
        # ── Task ──
        "task_timeout_seconds":       {"label": "Task Lock Timeout", "unit": "sec", "type": "int", "group": "Session & Messages", "min": 30, "max": 3600},
    }
    # Inject factory status
    from app.services.runtime_config import has_factory_presets, get_factory_presets, NON_FACTORY_KEYS
    return jsonify({
        "status": "ok",
        "schema": schema,
        "has_factory": has_factory_presets(),
        "non_factory_keys": list(NON_FACTORY_KEYS),
        "factory_presets": get_factory_presets(),
    })


# ── LLM Provider Management (admin-only, replaces user account modal selector) ──

@admin_bp.route('/admin/llm_providers', methods=['GET'])
@admin_required
def admin_llm_providers():
    """Return full provider info with model lists for admin config panel."""
    from app.services.llm_provider import PROVIDER_CONFIG
    from app.services.runtime_config import get as rc_get

    active_provider = rc_get('active_llm_provider', '') or 'auto'
    active_model = rc_get('active_llm_model', '') or 'auto'

    providers = {}
    for pid, cfg in PROVIDER_CONFIG.items():
        providers[pid] = {
            'name': cfg['name'],
            'models': cfg['models'],
            'default_model': cfg['default_model'],
        }

    # Build dynamic model list for active provider
    model_options = ['auto']
    if active_provider != 'auto' and active_provider in PROVIDER_CONFIG:
        model_options = ['auto'] + PROVIDER_CONFIG[active_provider]['models']

    # Also return what session currently has (live state, may differ if not yet applied)
    from flask import session as flask_session
    session_provider = flask_session.get('llm_provider', '')
    session_model = flask_session.get('llm_model', '')

    return jsonify({
        "status": "ok",
        "providers": providers,
        "active_provider": active_provider,
        "active_model": active_model,
        "model_options": model_options,
        "session_provider": session_provider,
        "session_model": session_model,
    })


# ── Mail: admin compose and send email ──
@admin_bp.route('/admin/send_mail', methods=['POST'])
@admin_required
def admin_send_mail():
    data = request.get_json(silent=True) or {}
    to_addr = data.get('to', '').strip()
    subject = data.get('subject', '').strip()
    body = data.get('body', '').strip()
    if not to_addr or not subject or not body:
        return jsonify({"error": "收件人、主题和正文不能为空"}), 400
    try:
        from app.utils.mailer import send_email, is_configured
        if not is_configured():
            return jsonify({"error": "SMTP未配置，请设置SMTP_HOST等环境变量"}), 503
        success = send_email(to_addr, subject, body, async_mode=True)
        if success:
            return jsonify({"status": "ok", "message": f"邮件已发送至 {to_addr}"})
        else:
            return jsonify({"error": "邮件发送失败"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Mail: get all user emails for autocomplete ──
@admin_bp.route('/admin/user_emails', methods=['GET'])
@admin_required
def admin_user_emails():
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT username, user_id, role, is_auditor FROM users WHERE is_active = TRUE ORDER BY username")
            users = cur.fetchall()
    return jsonify({"users": [
        {"username": u['username'], "user_id": u['user_id'],
         "role": u.get('role', 'user'), "is_auditor": bool(u.get('is_auditor', False))}
        for u in users
    ]})

# ---------- Helper functions for recycle bin folder restoration ----------
def restore_folder_recursive(folder_item, conn, cur, target_parent_id=None):
    parent_id = target_parent_id if target_parent_id is not None else folder_item['original_parent_id']
    cur.execute("""
        INSERT INTO project_folders (id, project_id, parent_folder_id, name, created_at, created_by)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """, (folder_item['original_id'], folder_item['project_id'], parent_id,
          folder_item['name'], folder_item['created_at'], folder_item['created_by']))
    cur.execute("""
        SELECT * FROM project_recycle_bin
        WHERE project_id = %s AND folder_id = %s
    """, (folder_item['project_id'], folder_item['original_id']))
    files = cur.fetchall()
    for f in files:
        cur.execute("""
            INSERT INTO project_files (project_id, folder_id, filename, original_name, file_size, stored_path, version, uploaded_by, file_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (f['project_id'], folder_item['original_id'], f['file_name'], f['original_name'],
              f['file_size'], f['stored_path'], f['version'], f['uploaded_by'], f['file_hash']))
        cur.execute("DELETE FROM project_recycle_bin WHERE id = %s", (f['id'],))
    cur.execute("""
        SELECT * FROM project_folders_recycle_bin
        WHERE project_id = %s AND original_parent_id = %s
    """, (folder_item['project_id'], folder_item['original_id']))
    subfolders = cur.fetchall()
    for sf in subfolders:
        restore_folder_recursive(sf, conn, cur, target_parent_id=folder_item['original_id'])
    cur.execute("DELETE FROM project_folders_recycle_bin WHERE id = %s", (folder_item['id'],))

def restore_folder_path_for_file(file_item, conn, cur):
    folder_id = file_item['folder_id']
    if folder_id is None:
        return
    cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s", (folder_id, file_item['project_id']))
    if cur.fetchone():
        return
    cur.execute("SELECT * FROM project_folders_recycle_bin WHERE original_id = %s AND project_id = %s", (folder_id, file_item['project_id']))
    folder = cur.fetchone()
    if not folder:
        return
    if folder['original_parent_id']:
        cur.execute("SELECT * FROM project_folders_recycle_bin WHERE original_id = %s AND project_id = %s", (folder['original_parent_id'], file_item['project_id']))
        parent = cur.fetchone()
        if parent:
            restore_folder_path_for_file(parent, conn, cur)
    cur.execute("""
        INSERT INTO project_folders (id, project_id, parent_folder_id, name, created_at, created_by)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """, (folder['original_id'], folder['project_id'], folder['original_parent_id'],
          folder['name'], folder['created_at'], folder['created_by']))
    cur.execute("DELETE FROM project_folders_recycle_bin WHERE id = %s", (folder['id'],))

# ---------- Scheduled jobs ----------
def delete_expired_original_files():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, original_stored_path
                FROM user_files
                WHERE original_expires_at IS NOT NULL AND original_expires_at <= NOW()
                  AND original_stored_path IS NOT NULL
            """)
            expired = cur.fetchall()
            for file_id, original_path in expired:
                if original_path and os.path.exists(original_path):
                    try:
                        os.remove(original_path)
                        logger.info(f"Deleted expired original file: {original_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete expired file {original_path}: {e}")
                cur.execute("UPDATE user_files SET original_stored_path = NULL WHERE id = %s", (file_id,))
            conn.commit()

def cleanup_old_anon_temp_files(days=1):
    now = time.time()
    for item in os.listdir(TEMP_ROOT):
        item_path = os.path.join(TEMP_ROOT, item)
        if os.path.isdir(item_path):
            if (now - os.path.getctime(item_path)) > days * 86400:
                shutil.rmtree(item_path)
                logger.info(f"Removed old anonymous temp dir: {item_path}")

def schedule_project_deletion_cleanup():
    cutoff = utc_now() - timedelta(days=3)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM projects WHERE status = 'archived' AND archived_at < %s", (cutoff,))
            to_delete = cur.fetchall()
            for (project_id,) in to_delete:
                logger.info(f"Auto-deleting archived project {project_id} after 3 days")
                # Delete physical files first (same as manual delete_project)
                cur.execute("SELECT stored_path FROM project_files WHERE project_id = %s", (project_id,))
                for (stored_path,) in cur.fetchall():
                    _safe_delete_file(stored_path, f'project_{project_id}_file')
                # Archive chat sessions before cascade delete
                cur.execute("SELECT thread_id, user_id FROM chat_sessions WHERE project_id = %s", (project_id,))
                for thread_id, uid in cur.fetchall():
                    cur.execute(
                        "INSERT INTO archived_sessions (thread_id, user_id, archive_path) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                        (thread_id, uid, f"data/dump/{thread_id}_session.json")
                    )
                cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
            conn.commit()

def cleanup_expired_recycle_bin():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT original_stored_path FROM recycle_bin WHERE expires_at <= NOW()")
            paths = cur.fetchall()
            for row in paths:
                if row[0] and os.path.exists(row[0]):
                    try:
                        os.remove(row[0])
                        logger.info(f"Deleted expired recycle file: {row[0]}")
                    except Exception as e:
                        logger.warning(f"Failed to delete expired file {row[0]}: {e}")
            cur.execute("DELETE FROM recycle_bin WHERE expires_at <= NOW()")
            cur.execute("SELECT stored_path FROM project_recycle_bin WHERE expires_at <= NOW()")
            paths = cur.fetchall()
            for row in paths:
                if row[0] and os.path.exists(row[0]):
                    try:
                        os.remove(row[0])
                        logger.info(f"Deleted expired project recycle file: {row[0]}")
                    except Exception as e:
                        logger.warning(f"Failed to delete expired file {row[0]}: {e}")
            cur.execute("DELETE FROM project_recycle_bin WHERE expires_at <= NOW()")
            # Also clean expired kb_recycle_bin (skills + KB files)
            cur.execute("SELECT stored_path FROM kb_recycle_bin WHERE expires_at <= NOW()")
            for row in cur.fetchall():
                if row[0] and os.path.exists(row[0]):
                    try: os.remove(row[0])
                    except: pass
            cur.execute("DELETE FROM kb_recycle_bin WHERE expires_at <= NOW()")
            conn.commit()
            logger.info("Cleaned up expired recycle bin items (all 3 bins)")

@admin_bp.route('/admin/system_cleanup', methods=['POST'])
@admin_required
def admin_system_cleanup():
    """Run all cleanup tasks and return a report."""
    results = {}
    # 1. Stale chat sessions
    try:
        from app.services.session_manager import cleanup_old_sessions
        cleanup_old_sessions(days=15)
        results['sessions'] = '已完成'
    except Exception as e:
        results['sessions'] = str(e)[:100]
    # 2. Temp files
    try:
        from app.cleanup_tasks import auto_cleanup_temp_files
        auto_cleanup_temp_files()
        results['temp_files'] = '已完成'
    except Exception as e:
        results['temp_files'] = str(e)[:100]
    # 3. Memory
    try:
        from app.cleanup_tasks import auto_cleanup_memory
        auto_cleanup_memory()
        results['memory'] = '已完成'
    except Exception as e:
        results['memory'] = str(e)[:100]
    # 4. File audit
    try:
        audit = admin_file_audit()
        audit_data = audit.get_json()
        results['file_audit'] = f"孤儿{audit_data.get('orphans_count',0)}个, 泄漏{audit_data.get('disk_leaks_count',0)}个"
    except Exception as e:
        results['file_audit'] = str(e)[:100]
    return jsonify({"status": "ok", "results": results})

@admin_bp.route('/admin/clear_all_data', methods=['POST'])
@admin_required
def admin_clear_all_data():
    """Wipe all uploaded files, generated skills, and their DB records.
    Keeps: users (including admin accounts), projects structure, chat sessions.
    Destroys: file content, skills, AI memory, RAG indexes, search cache.
    """
    import shutil as _shutil
    results = {}

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 1. Clear skill data from all file tables
            for table in ('knowledge_lab_files', 'company_knowledge_base', 'project_files'):
                cur.execute(f"UPDATE {table} SET skill_summary=NULL, skill_generated_at=NULL, skill_summary_hash=NULL")
                results[f'{table}_skills_cleared'] = cur.rowcount
            # 2. Clear file content
            for table in ('knowledge_lab_files', 'company_knowledge_base', 'project_files', 'user_files'):
                cur.execute(f"UPDATE {table} SET content=NULL")
                results[f'{table}_content_cleared'] = cur.rowcount
            # 3. Delete file records
            for table in ('knowledge_lab_files', 'company_knowledge_base', 'user_files'):
                cur.execute(f"DELETE FROM {table}")
                results[f'{table}_deleted'] = cur.rowcount
            # 4. Clear AI memory
            cur.execute("DELETE FROM project_ai_memory")
            results['ai_memory_deleted'] = cur.rowcount
            # 5. Clear caches
            cur.execute("DELETE FROM file_text_cache")
            cur.execute("DELETE FROM image_description_cache")
            results['caches_cleared'] = 'ok'
            # 6. Clear skill audit cache
            from app.services.skill_auditor import invalidate_audit_cache
            invalidate_audit_cache()
            conn.commit()

    # 8. Wipe physical files
    dirs_to_clean = [
        ('data/user_files', False),      # files only
        ('company_kb_files', False),
        ('knowledge_lab_files', False),
        ('data/project_files', True),    # recursive
        ('data/training/raw', True),
        ('data/training/exports', False),
        ('data/search_cache', False),
    ]
    base = os.path.dirname(__file__)  # app/routes/
    base = os.path.dirname(base)      # app/
    base = os.path.dirname(base)      # project root
    
    for rel_dir, recursive in dirs_to_clean:
        full = os.path.join(base, rel_dir)
        if not os.path.exists(full):
            continue
        count = 0
        if recursive:
            for root, dirs, files in os.walk(full, topdown=False):
                for f in files:
                    try: os.remove(os.path.join(root, f)); count += 1
                    except: pass
                for d in dirs:
                    try: _shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                    except: pass
        else:
            for f in os.listdir(full):
                fp = os.path.join(full, f)
                if os.path.isfile(fp):
                    try: os.remove(fp); count += 1
                    except: pass
        results[f'disk_{rel_dir}'] = count

    # 9. Delete skill audit cache file
    try:
        cache_file = os.path.join(base, 'data', 'skill_audit_cache.json')
        if os.path.exists(cache_file):
            os.remove(cache_file)
    except: pass

    return jsonify({"status": "ok", "results": results})

# ── Safe file deletion helper ──
def _safe_delete_file(filepath, label=''):
    """Delete a file and log failure. Returns True if deleted or didn't exist."""
    if not filepath:
        return True
    if not os.path.exists(filepath):
        return True
    try:
        os.remove(filepath)
        return True
    except Exception as e:
        logger.error(f"[FILE_LEAK] Cannot delete {label or filepath}: {e}")
        return False

@admin_bp.route('/admin/file_audit', methods=['GET'])
@admin_required
def admin_file_audit():
    """Audit: scan all stored_path references in DB, check disk existence.
    Returns orphans (DB path exists but file missing) and leaks (file on disk but no DB row).
    """
    tables_to_check = [
        ('user_files', 'original_stored_path'),
        ('user_files', 'stored_path'),
        ('knowledge_lab_files', 'stored_path'),
        ('company_knowledge_base', 'stored_path'),
        ('project_files', 'stored_path'),
        ('recycle_bin', 'original_stored_path'),
        ('project_recycle_bin', 'stored_path'),
        ('kb_recycle_bin', 'stored_path'),
    ]
    orphans = []
    total_checked = 0
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for table, col in tables_to_check:
                try:
                    cur.execute(f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL AND {col} != ''")
                    for (path,) in cur.fetchall():
                        total_checked += 1
                        if path and not os.path.exists(path):
                            orphans.append({'table': table, 'column': col, 'path': path})
                except Exception as e:
                    logger.warning(f"Audit skip {table}.{col}: {e}")

    # Scan data directories for files not referenced in DB
    scan_dirs = ['data/user_files', 'data/project_files', 'data/dump']
    leaks = []
    for scan_dir in scan_dirs:
        abs_dir = os.path.join(os.path.dirname(__file__), '..', '..', scan_dir)
        if not os.path.exists(abs_dir):
            continue
        for root, _, files in os.walk(abs_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                # Quick check: is this path referenced anywhere in DB?
                found = False
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        for table, col in tables_to_check:
                            cur.execute(f"SELECT 1 FROM {table} WHERE {col} = %s LIMIT 1", (full_path,))
                            if cur.fetchone():
                                found = True
                                break
                if not found:
                    leaks.append({'path': full_path, 'size': os.path.getsize(full_path)})

    return jsonify({
        'db_paths_checked': total_checked,
        'orphans': orphans[:100],
        'orphans_count': len(orphans),
        'disk_leaks': leaks[:100],
        'disk_leaks_count': len(leaks),
        'total_leak_bytes': sum(l['size'] for l in leaks),
    })

# ── Training Notifications ──

@admin_bp.route('/admin/notifications', methods=['GET'])
@admin_required
def admin_notifications():
    """Get training/system notifications for admin panel."""
    import os, json
    from app.config import DATA_DIR

    notify_path = os.path.join(str(DATA_DIR), 'ingest', 'training_notifications.json')
    notifications = []
    if os.path.exists(notify_path):
        try:
            with open(notify_path, 'r', encoding='utf-8') as f:
                notifications = json.load(f)
        except Exception:
            pass

    unread = len([n for n in notifications if not n.get('seen_by')])
    return jsonify({
        "notifications": notifications,
        "unread": unread,
        "total": len(notifications),
    })


@admin_bp.route('/admin/notifications/mark_read', methods=['POST'])
@admin_required
def admin_mark_notifications_read():
    """Mark notifications as seen by current admin."""
    import os, json
    from app.config import DATA_DIR

    user_id = session.get('user_id', '')
    notify_path = os.path.join(str(DATA_DIR), 'ingest', 'training_notifications.json')

    if not os.path.exists(notify_path):
        return jsonify({"success": True, "marked": 0})

    try:
        with open(notify_path, 'r', encoding='utf-8') as f:
            notifications = json.load(f)

        marked = 0
        for n in notifications:
            if user_id not in n.get('seen_by', []):
                n.setdefault('seen_by', []).append(user_id)
                marked += 1

        with open(notify_path, 'w', encoding='utf-8') as f:
            json.dump(notifications, f, ensure_ascii=False, default=str)

        return jsonify({"success": True, "marked": marked})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======================== Quote Anomaly Admin Routes ========================

@admin_bp.route('/admin/quote_anomaly_results', methods=['GET'])
@admin_required
def admin_quote_anomaly_results():
    """List all stored quote anomaly results (admin view)."""
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT q.id, q.task_id, q.doc_name, q.cv, q.same_rate_flag,
                       q.abnormal_drop_flag, q.clustering_flag, q.benford_deviation,
                       q.risk_score, q.details, q.checked_at, u.username
                FROM quote_anomaly_results q
                LEFT JOIN users u ON q.user_id = u.user_id
                ORDER BY q.checked_at DESC LIMIT %s OFFSET %s
            """, (limit, offset))
            results = cur.fetchall()
            cur.execute("SELECT COUNT(*) as total FROM quote_anomaly_results")
            total = cur.fetchone()['total']
    return jsonify({"results": [dict(r) for r in results], "total": total})


@admin_bp.route('/admin/quote_anomaly_results/<int:id>', methods=['GET'])
@admin_required
def admin_quote_anomaly_detail(id):
    """Get detailed quote anomaly result by ID."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT q.*, u.username FROM quote_anomaly_results q
                LEFT JOIN users u ON q.user_id = u.user_id WHERE q.id = %s
            """, (id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Not found"}), 404
    return jsonify(dict(row))


# ── Typo detection admin routes ──

@admin_bp.route('/admin/typo_results', methods=['GET'])
@admin_required
def admin_typo_results():
    """List stored typo detection results."""
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT t.*, u.username FROM typo_detection_results t
                LEFT JOIN users u ON t.user_id = u.user_id
                ORDER BY t.checked_at DESC LIMIT %s OFFSET %s
            """, (limit, offset))
            results = cur.fetchall()
            cur.execute("SELECT COUNT(*) as total FROM typo_detection_results")
            total = cur.fetchone()['total']
    return jsonify({"results": [dict(r) for r in results], "total": total})


# ── Relationship extraction admin routes ──

@admin_bp.route('/admin/relationship_results', methods=['GET'])
@admin_required
def admin_relationship_results():
    """List all stored relationship extraction results."""
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT r.*, u.username FROM relationship_risk_summary r
                LEFT JOIN users u ON r.user_id = u.user_id
                ORDER BY r.checked_at DESC LIMIT %s OFFSET %s
            """, (limit, offset))
            results = cur.fetchall()
            cur.execute("SELECT COUNT(*) as total FROM relationship_risk_summary")
            total = cur.fetchone()['total']
    return jsonify({"results": [dict(r) for r in results], "total": total})


@admin_bp.route('/admin/relationship_results/<task_id>', methods=['GET'])
@admin_required
def admin_relationship_detail(task_id):
    """Get detailed relationship results for a task, including all individual relations."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT r.*, u.username FROM relationship_risk_summary r
                LEFT JOIN users u ON r.user_id = u.user_id WHERE r.task_id = %s
            """, (task_id,))
            summary = cur.fetchone()
            if not summary:
                return jsonify({"error": "Not found"}), 404

            cur.execute("""
                SELECT * FROM entity_relationships WHERE task_id = %s
                ORDER BY module, confidence DESC
            """, (task_id,))
            relations = cur.fetchall()
    return jsonify({
        "summary": dict(summary),
        "relationships": [dict(r) for r in relations],
    })


# ======================== Credit Check Routes ========================
credit_tasks = {}  # in‑memory store for running tasks

