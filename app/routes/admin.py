"""Blueprint: admin routes (auto-extracted)."""
import os, json, uuid, time, logging, hashlib, io, re, zipfile, shutil, random
from datetime import datetime, timezone, timedelta
from functools import wraps
from io import BytesIO
from flask import Blueprint, request, jsonify, session, send_file, render_template, url_for, current_app
from werkzeug.datastructures import FileStorage

from app.config import BASE_DIR, DATA_DIR, TEMP_ROOT, TEMP_DIR, USER_FILES_ORIGINAL_ROOT, PROJECT_FILES_ROOT, to_rel_path, resolve_path, logger
from app.database import get_db_connection, db_transaction
from app.utils.helpers import utc_now, beijing_now, safe_error_response, split_thinking_answer, ok, err
import app.globals as g
from app.services.file_cache import file_cache_manager, add_to_cache, load_cache_from_db
from app.services.file_processing import extract_text_from_file
from app.services.document_classifier import classify_and_categorize

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

# Project presence tracking (lightweight in-memory). Imported by cleanup_tasks.py
# and admin_knowledge_lab.py — must stay on this module.
_project_presence = {}

@admin_bp.route('/admin/task_deposit', methods=['GET'])
def get_task_deposit():
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)
    is_admin_user = session.get('role') == 'admin'
    if not is_admin_user:
        return err("Access denied", "FORBIDDEN", 403)
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
            return ok({"items": items})

@admin_bp.route('/admin/task_deposit/transfer/<int:item_id>', methods=['POST'])
def transfer_task_deposit_item(item_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if session.get('role') != 'admin':
        return err("Only admin can transfer deposit items", "FORBIDDEN", 403)
    data = request.get_json()
    target_user_id = data.get('target_user_id')
    if not target_user_id:
        return err("Missing target_user_id", "VALIDATION_ERROR", 400)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE user_id = %s", (target_user_id,))
            if not cur.fetchone():
                return err("Target user not found", "NOT_FOUND", 404)
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
                return err("Item not found or already deleted", "NOT_FOUND", 404)
            conn.commit()
            return ok({"item": dict(item)})

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
                return err(f"Requires {min_role} role or higher", "FORBIDDEN", 403)
            return f(*args, **kwargs)
        return wrapper
    return decorator

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_admin():
            return err("Admin access required", "FORBIDDEN", 403)
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
            return err("Admin or Auditor access required", "FORBIDDEN", 403)
        return f(*args, **kwargs)
    return decorated_function

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('consent_value', 0) != 1:
            return err("Consent not given", "FORBIDDEN", 403)
        if not session.get('user_id'):
            return err("Not logged in", "AUTH_REQUIRED", 401)
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
    bidding_category = data.get('bidding_category', 'general').strip()
    bid_method = data.get('bid_method', 'open').strip()
    if industry not in ('bidding_agency', 'engineering_cost', 'engineering_audit', 'general'):
        industry = 'general'
    if not name:
        return err("Project name required", "VALIDATION_ERROR", 400)
    user_id = session.get('user_id')
    import uuid
    chat_thread_id = str(uuid.uuid4())
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO projects (name, description, created_by, status, industry, bidding_category, bid_method) VALUES (%s, %s, %s, 'active', %s, %s, %s) RETURNING id",
                    (name, description, user_id, industry, bidding_category, bid_method))
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

        # Auto-create a default project timeline (after commit, uses its own connection)
        try:
            from datetime import date as _dt
            from app.services.project_timeline_service import create_timeline as svc_create_tl
            tl_name_map = {'工程': '施工招标', '货物': '货物采购', '服务': '服务采购'}
            tl_name = tl_name_map.get(bidding_category, '主招标流程')
            svc_create_tl(
                project_id=project_id, name=f"{name} - {tl_name}",
                category_code=bidding_category, method_code=bid_method,
                planned_start_date=_dt.today(),
                created_by=user_id,
            )
        except Exception:
            logger.warning("Failed to auto-create project timeline", exc_info=True)

        return ok({"id": project_id, "chat_thread_id": chat_thread_id})

@admin_bp.route('/admin/projects/<int:project_id>/backfill_chat', methods=['POST'])
def backfill_project_chat(project_id):
    """Ensure a shared project chat session exists for this project.
    
    Called by frontend when openProject() detects the project chat is missing
    from the sidebar (e.g. legacy projects created before auto-chat was added).
    """
    user_id = session.get('user_id')
    if not user_id:
        return err("Not authenticated", "AUTH_REQUIRED", 401)
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
                    return err("Project not found or access denied", "NOT_FOUND", 404)
                
                # Check if chat already exists (exclude grilling threads)
                cur.execute("""
                    SELECT thread_id FROM chat_sessions
                    WHERE project_id = %s AND (is_grilling = FALSE OR is_grilling IS NULL)
                    LIMIT 1
                """, (project_id,))
                existing = cur.fetchone()
                if existing:
                    return ok({"thread_id": existing['thread_id'], "existed": True})
                
                # Create new shared chat
                import uuid
                thread_id = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO chat_sessions (user_id, thread_id, title, project_id) VALUES (%s, %s, %s, %s)",
                    (user_id, thread_id, proj['name'], project_id))
                conn.commit()
                logger.info(f"Backfilled project chat: {proj['name']} (project_id={project_id}, thread={thread_id})")
                return ok({"thread_id": thread_id, "existed": False})
    except Exception as e:
        logger.error(f"Backfill project chat failed: {e}")
        return err(str(e)[:200], "SERVER_ERROR", 500)

@admin_bp.route('/admin/projects', methods=['GET'])
def get_projects():
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not user_id:
        return ok({"projects": [], "has_projects": False})
    if is_admin():
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, name, description, created_at, updated_at, status, archived_at, archive_filename, deletion_scheduled_at FROM projects ORDER BY CASE status WHEN 'active' THEN 1 WHEN 'archived' THEN 2 WHEN 'aborted' THEN 3 END, created_at DESC")
                projects = cur.fetchall()
                has_projects = len(projects) > 0
                return ok({"projects": projects, "has_projects": has_projects})
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
                return ok({"projects": projects, "has_projects": has_projects})

@admin_bp.route('/admin/projects/<int:project_id>', methods=['PUT'])
def update_project(project_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)

    if not is_admin() and not _can_manage_files(project_id, user_id):
        return err("Permission denied", "FORBIDDEN", 403)

    data = request.get_json()
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()
    if not name:
        return err("Project name required", "VALIDATION_ERROR", 400)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE projects
                SET name = %s, description = %s, updated_at = NOW()
                WHERE id = %s
                RETURNING id
            """, (name, description, project_id))
            if cur.fetchone():
                # Sync project chat title (skip grilling threads)
                cur.execute(
                    "UPDATE chat_sessions SET title = %s WHERE project_id = %s AND (is_grilling = FALSE OR is_grilling IS NULL)",
                    (name, project_id)
                )
                conn.commit()
                return ok(message="ok")
            else:
                return err("Project not found", "NOT_FOUND", 404)

@admin_bp.route('/admin/projects/<int:project_id>', methods=['DELETE'])
@admin_required
def delete_project(project_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
            if not row:
                return err("Project not found", "NOT_FOUND", 404)
            status = row[0]
            if status not in ('archived', 'aborted'):
                return err("Only archived or aborted projects can be deleted", "VALIDATION_ERROR", 400)

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
                _safe_delete_file(resolve_path(stored_path), f'project_{project_id}_file')

            cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
            conn.commit()
            return ok(message="ok")

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>', methods=['DELETE'])
def delete_project_file(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not _can_edit_file(project_id, file_id, user_id):
        return err("Permission denied", "FORBIDDEN", 403)

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, original_name, file_size, stored_path, uploaded_by, folder_id, filename, version, file_hash, project_id, status
                    FROM project_files
                    WHERE id = %s AND project_id = %s
                """, (file_id, project_id))
                file_record = cur.fetchone()
                if not file_record:
                    return err("File not found", "NOT_FOUND", 404)

                if file_record.get('status') == 'final':
                    from app.services.project_wiki_publisher import unpublish_project_file
                    unpublish_project_file(file_id)

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
                return ok({"moved_to_recycle_bin": True})

@admin_bp.route('/admin/projects/<int:project_id>/abort', methods=['POST'])
@admin_required
def abort_project(project_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE projects SET status = 'aborted', archived_at = NOW() WHERE id = %s RETURNING id",
                        (project_id,))
            if cur.fetchone():
                conn.commit()
                return ok(message="ok")
            return err("Project not found", "NOT_FOUND", 404)

@admin_bp.route('/admin/projects/<int:project_id>/finish', methods=['POST'])
def finish_project(project_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not _can_manage_files(project_id, user_id):
        return err("Only admin or project manager can finish a project", "FORBIDDEN", 403)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM projects WHERE id = %s AND status = 'active'", (project_id,))
            project = cur.fetchone()
            if not project:
                return err("Project not found or already finished/aborted", "NOT_FOUND", 404)
            project_name = project[0]
            cur.execute("SELECT stored_path, original_name FROM project_files WHERE project_id = %s", (project_id,))
            files = cur.fetchall()
            if not files:
                return err("No files to archive", "VALIDATION_ERROR", 400)
            zip_dir = os.path.join(PROJECT_FILES_ROOT, 'archives')
            os.makedirs(zip_dir, exist_ok=True)
            safe_name = re.sub(r'[^\w\-_\.]', '_', project_name)
            zip_filename = f"project_{project_id}_{safe_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = os.path.join(zip_dir, zip_filename)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for stored_path, original_name in files:
                    zipf.write(resolve_path(stored_path), original_name)
            cur.execute("UPDATE projects SET status = 'archived', archived_at = NOW(), archive_filename = %s WHERE id = %s",
                        (zip_filename, project_id))
            conn.commit()
            return ok({
                "success": True,
                "download_url": f"/admin/projects/{project_id}/download_archive/{zip_filename}",
                "zip_filename": zip_filename
            })

@admin_bp.route('/admin/projects/<int:project_id>/download_archive/<zip_filename>', methods=['GET'])
def download_archive(project_id, zip_filename):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    zip_dir = os.path.join(PROJECT_FILES_ROOT, 'archives')
    zip_path = os.path.join(zip_dir, zip_filename)
    if not os.path.exists(zip_path):
        return err("Archive not found", "NOT_FOUND", 404)
    return send_file(zip_path, as_attachment=True, download_name=zip_filename)

# Project members routes
@admin_bp.route('/admin/projects/<int:project_id>/members', methods=['GET'])
def get_project_members(project_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
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
            return ok({"members": members})

@admin_bp.route('/admin/projects/<int:project_id>/members/search', methods=['GET'])
def search_users_to_add(project_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_manage_members(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)

    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return ok({"users": []})
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
            return ok({"users": users})

@admin_bp.route('/admin/users', methods=['GET'])
def list_users():
    """Return all active users (for searchable dropdowns, member pickers)."""
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    if not is_admin():
        return err("Admin only", "FORBIDDEN", 403)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT user_id, username, role
                FROM users
                WHERE is_active = TRUE AND username IS NOT NULL AND username != ''
                ORDER BY username
            """)
            return ok({"users": cur.fetchall()})

@admin_bp.route('/admin/projects/<int:project_id>/all_users', methods=['GET'])
def get_all_users_for_project(project_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    current_user_id = session.get('user_id')
    if not current_user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)
    if not is_admin() and not _can_manage_members(project_id, current_user_id):
        return err("Access denied", "FORBIDDEN", 403)

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
            return ok({"users": users})

@admin_bp.route('/admin/projects/<int:project_id>/members', methods=['POST'])
def add_project_member(project_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not _can_manage_members(project_id, user_id):
        return err("Only admin or project manager can add members", "FORBIDDEN", 403)

    data = request.get_json()
    new_user_id = data.get('user_id')
    role = data.get('role', 'member')
    if role == 'manager' and not is_admin():
        return err("Only admin can add managers", "FORBIDDEN", 403)
    if role not in ('member', 'manager'):
        return err("Invalid role", "VALIDATION_ERROR", 400)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT role FROM users WHERE user_id = %s", (new_user_id,))
            row = cur.fetchone()
            if row and row[0] == 'admin':
                return err("Cannot add a global admin as a project member", "FORBIDDEN", 403)

            if not row:
                return err("User not found", "NOT_FOUND", 404)

            cur.execute("SELECT 1 FROM project_members WHERE project_id = %s AND user_id = %s",
                        (project_id, new_user_id))
            if cur.fetchone():
                return err("User already a member", "RESOURCE_BUSY", 409)

            cur.execute("""
                INSERT INTO project_members (project_id, user_id, role, added_by)
                VALUES (%s, %s, %s, %s)
            """, (project_id, new_user_id, role, user_id))
            # Auto-backfill project chat if none exists
            cur.execute("SELECT 1 FROM chat_sessions WHERE project_id = %s AND (is_grilling = FALSE OR is_grilling IS NULL) LIMIT 1", (project_id,))
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
            return ok(message="ok")

@admin_bp.route('/admin/projects/<int:project_id>/members/<user_id>', methods=['PUT'])
@admin_required
def update_member_role(project_id, user_id):
    data = request.get_json()
    new_role = data.get('role')
    if new_role not in ('member', 'manager'):
        return err("Invalid role", "VALIDATION_ERROR", 400)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT role FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row and row[0] == 'admin':
                return err("Cannot modify global admin's role", "FORBIDDEN", 403)

            cur.execute("""
                UPDATE project_members
                SET role = %s
                WHERE project_id = %s AND user_id = %s
                RETURNING user_id
            """, (new_role, project_id, user_id))
            if cur.rowcount == 0:
                return err("Member not found", "NOT_FOUND", 404)
            conn.commit()
            return ok(message="ok")

@admin_bp.route('/admin/projects/<int:project_id>/members/<user_id>', methods=['DELETE'])
def remove_project_member(project_id, user_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    current_user_id = session.get('user_id')
    if not _can_manage_members(project_id, current_user_id):
        return err("Only admin or project manager can remove members", "FORBIDDEN", 403)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT role FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row and row[0] == 'admin':
                return err("Cannot remove a global admin", "FORBIDDEN", 403)

            cur.execute("""
                SELECT role FROM project_members
                WHERE project_id = %s AND user_id = %s
            """, (project_id, user_id))
            target_member = cur.fetchone()
            if not target_member:
                return err("Member not found", "NOT_FOUND", 404)

            target_role = target_member[0]
            if target_role == 'admin':
                return err("Cannot remove the project admin", "FORBIDDEN", 403)
            if target_role == 'manager' and not is_admin():
                return err("Only admin can remove managers", "FORBIDDEN", 403)

            cur.execute("""
                UPDATE project_members SET status = 'quitted'
                WHERE project_id = %s AND user_id = %s
            """, (project_id, user_id))
            if cur.rowcount == 0:
                return err("Member not found", "NOT_FOUND", 404)
            conn.commit()
            return ok({"quitted": True})

@admin_bp.route('/admin/projects/<int:project_id>/transfer_manager/<user_id>', methods=['POST'])
def transfer_manager_role(project_id, user_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    current_user_id = session.get('user_id')
    current_role = get_user_role_in_project(project_id, current_user_id)
    if current_role != 'manager':
        return err("Only a manager can transfer manager rights", "FORBIDDEN", 403)
    target_role = get_user_role_in_project(project_id, user_id)
    if target_role != 'member':
        return err("Target user must be a member", "VALIDATION_ERROR", 400)
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute("UPDATE project_members SET role = 'member' WHERE project_id = %s AND user_id = %s",
                            (project_id, current_user_id))
                cur.execute("UPDATE project_members SET role = 'manager' WHERE project_id = %s AND user_id = %s",
                            (project_id, user_id))
                conn.commit()
    return ok(message="ok")

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
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
    ensure_root_folder(project_id)
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, parent_folder_id, name FROM project_folders WHERE project_id = %s ORDER BY parent_folder_id, name",
                    (project_id,))
                folders = cur.fetchall()
                if not folders:
                    return ok({"folders": []})
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
                return ok({"folders": root_folders})
    except Exception as e:
        logger.error(f"Error in get_folders: {e}", exc_info=True)
        return err("Internal server error", "SERVER_ERROR", 500)

@admin_bp.route('/admin/projects/<int:project_id>/folders', methods=['POST'])
def create_folder(project_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
    data = request.get_json()
    name = data.get('name', '').strip()
    parent_folder_id = data.get('parent_folder_id')
    if not name:
        return err("Folder name required", "VALIDATION_ERROR", 400)
    if parent_folder_id is None:
        return err("Cannot create root folder. Only one root folder exists per project.", "VALIDATION_ERROR", 400)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s",
                        (parent_folder_id, project_id))
            if not cur.fetchone():
                return err("Parent folder not found", "NOT_FOUND", 404)
            cur.execute(
                "INSERT INTO project_folders (project_id, parent_folder_id, name, created_by) VALUES (%s, %s, %s, %s) RETURNING id",
                (project_id, parent_folder_id, name, user_id))
            new_id = cur.fetchone()[0]
            conn.commit()
            return ok({"id": new_id})

@admin_bp.route('/admin/projects/<int:project_id>/folders/<int:folder_id>', methods=['DELETE'])
def delete_folder(project_id, folder_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not _can_edit_folder(project_id, folder_id, user_id):
        return err("Permission denied", "FORBIDDEN", 403)

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
                return ok({
                    "success": True,
                    "folders_moved": len(folders),
                    "files_moved": len(files) if files else 0
                })

@admin_bp.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/rename', methods=['PUT'])
def rename_folder(project_id, folder_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not _can_edit_folder(project_id, folder_id, user_id):
        return err("Permission denied", "FORBIDDEN", 403)
    data = request.get_json()
    new_name = data.get('name', '').strip()
    if not new_name:
        return err("Folder name required", "VALIDATION_ERROR", 400)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT parent_folder_id FROM project_folders WHERE id = %s AND project_id = %s",
                        (folder_id, project_id))
            row = cur.fetchone()
            if not row:
                return err("Folder not found", "NOT_FOUND", 404)
            parent_id = row[0]
            cur.execute(
                "SELECT id FROM project_folders WHERE project_id = %s AND parent_folder_id = %s AND name = %s AND id != %s",
                (project_id, parent_id, new_name, folder_id))
            if cur.fetchone():
                return err("A folder with this name already exists in this location", "VALIDATION_ERROR", 400)
            cur.execute("UPDATE project_folders SET name = %s WHERE id = %s", (new_name, folder_id))
            conn.commit()
            return ok(message="ok")

# Project files management
os.makedirs(PROJECT_FILES_ROOT, exist_ok=True)

def get_project_file_path(project_id, unique_filename):
    project_dir = os.path.join(PROJECT_FILES_ROOT, str(project_id))
    os.makedirs(project_dir, exist_ok=True)
    return os.path.join(project_dir, unique_filename)

@admin_bp.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/upload', methods=['POST'])
def upload_project_file(project_id, folder_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
            if not row or row[0] != 'active':
                return err("Project is not active. Cannot upload.", "VALIDATION_ERROR", 400)

    if 'file' not in request.files:
        return err("No file", "VALIDATION_ERROR", 400)
    file = request.files['file']
    if file.filename == '':
        return err("Empty filename", "VALIDATION_ERROR", 400)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s", (folder_id, project_id))
            if not cur.fetchone():
                return err("Folder not found", "NOT_FOUND", 404)

    original_name = file.filename
    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file.seek(0)

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check hash duplicate (identical content)
            cur.execute("SELECT id, original_name, stored_path, version, folder_id, file_size FROM project_files WHERE project_id = %s AND file_hash = %s", (project_id, file_hash))
            hash_dup = cur.fetchone()
            if hash_dup:
                return ok({
                    "duplicate": True,
                    "conflict_type": "hash",
                    "existing_file": {
                        "id": hash_dup['id'],
                        "original_name": hash_dup['original_name'],
                        "folder_id": hash_dup['folder_id'],
                        "version": hash_dup['version'],
                        "file_size": hash_dup['file_size']
                    },
                    "new_filename": original_name
                })
            # Check name duplicate (same name, different content)
            cur.execute("SELECT id, original_name, stored_path, version, folder_id, file_size, file_hash FROM project_files WHERE project_id = %s AND original_name = %s", (project_id, original_name))
            name_dup = cur.fetchone()
            if name_dup:
                return ok({
                    "duplicate": True,
                    "conflict_type": "name",
                    "existing_file": {
                        "id": name_dup['id'],
                        "original_name": name_dup['original_name'],
                        "folder_id": name_dup['folder_id'],
                        "version": name_dup['version'],
                        "file_size": name_dup['file_size'],
                        "file_hash": name_dup['file_hash']
                    },
                    "new_filename": original_name
                })

    ext = os.path.splitext(original_name)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = get_project_file_path(project_id, unique_name)
    stored_rel = to_rel_path(stored_path)
    # Save the binary file
    file.save(stored_path)
    file_size = os.path.getsize(stored_path)

    # Extract text content from the file for knowledge base and search
    fake_file = FileStorage(BytesIO(file_bytes), filename=original_name)
    text_content, _ = extract_text_from_file(fake_file)
    if not text_content or text_content.startswith("["):
        text_content = ""  # fallback

    # Auto-categorize file content
    doc_type, category = classify_and_categorize(text_content, original_name, file_hash)

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO project_files (project_id, folder_id, filename, original_name, file_size,
                                               stored_path, uploaded_by, file_hash, content, category)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (project_id, folder_id, unique_name, original_name, file_size, stored_rel, user_id, file_hash, text_content, category))
                file_id = cur.fetchone()[0]
                conn.commit()
                from app.routes.knowledge import _try_index_file, _try_wiki_ingest, _try_entity_extract
                _try_index_file(file_id, text_content, 'project_files',
                               {'original_name': original_name, 'owner': user_id})
                _try_wiki_ingest(file_id, text_content, original_name, 'project_files',
                                {'original_name': original_name, 'owner': user_id})
                _try_entity_extract(file_id, text_content, original_name, 'project_files',
                                   doc_type, category,
                                   {'original_name': original_name, 'owner': user_id})
                return ok({"file_id": file_id, "original_name": original_name, "version": 1})

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/content', methods=['GET'])
def get_file_content(project_id, file_id):
    """Return extracted text content for a single project file. Used by conflict compare panel."""
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, original_name, content, file_size, version, stored_path
                FROM project_files
                WHERE id = %s AND project_id = %s
            """, (file_id, project_id))
            f = cur.fetchone()
            if not f:
                return err("File not found", "NOT_FOUND", 404)

            text = (f.get('content') or '').strip()
            if not text:
                stored = f.get('stored_path')
                if stored and os.path.exists(resolve_path(stored)):
                    try:
                        with open(resolve_path(stored), 'rb') as fh:
                            fake = FileStorage(fh, filename=f['original_name'])
                            text, _ = extract_text_from_file(fake)
                            text = text or ''
                    except Exception:
                        text = ''
            if not text:
                text = '[无法提取文本内容]'

            return ok({
                "id": f['id'],
                "name": f['original_name'],
                "text": text,
                "size": f['file_size'],
                "version": f['version']
            })

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/new_version', methods=['POST'])
def new_file_version(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
    if 'file' not in request.files:
        return err("No file", "VALIDATION_ERROR", 400)
    file = request.files['file']
    if file.filename == '':
        return err("Empty filename", "VALIDATION_ERROR", 400)

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
                return err("File not found", "NOT_FOUND", 404)

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
                        return err("Could not extract text from new version", "VALIDATION_ERROR", 400)
            else:
                # For images, audio, video, etc., just store empty content
                file_content = ""

            ext = os.path.splitext(original_name)[1]
            unique_name = f"{uuid.uuid4().hex}{ext}"
            stored_path = get_project_file_path(project_id, unique_name)
            stored_rel = to_rel_path(stored_path)
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
                        (new_version, stored_rel, file_size, user_id, file_hash, original_name, file_content, file_id))

            cur.execute("""
                        INSERT INTO project_file_usage (file_id, user_id, action, details)
                        VALUES (%s, %s, 'new_version', %s)
                        """, (file_id, user_id, json.dumps({'version': new_version, 'size': file_size})))

            conn.commit()
            return ok({"file_id": file_id, "original_name": original_name, "version": new_version})

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/status', methods=['POST'])
@admin_required
def set_project_file_status(project_id, file_id):
    data = request.get_json()
    status = data.get('status')
    if status not in ('draft', 'final'):
        return err("Invalid status, must be 'draft' or 'final'", "VALIDATION_ERROR", 400)
    from app.services.project_wiki_publisher import set_status_and_publish
    set_status_and_publish(file_id, status)
    return ok({"status": status})

@admin_bp.route('/admin/projects/<int:project_id>/wiki-tree', methods=['GET'])
@admin_required
def project_wiki_tree(project_id):
    from app.services.wiki_tree_service import get_merged_tree
    tree = get_merged_tree(project_id)
    return ok({"tree": tree})

@admin_bp.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/files', methods=['GET'])
def list_project_files(project_id, folder_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)

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
            return ok({"files": result})

@admin_bp.route('/admin/projects/<int:project_id>/files', methods=['GET'])
def list_root_files(project_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)

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
            return ok({"files": result})

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/versions', methods=['GET'])
def get_file_versions(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
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
            return ok({"versions": versions})

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/download', methods=['GET'])
def download_project_file(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
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
                return err("File not found", "NOT_FOUND", 404)
            stored_path, original_name = row
            stored_path = resolve_path(stored_path)
    if not os.path.exists(stored_path):
        return err("文件已被清理，无法下载", "SERVER_ERROR", 410)
    return send_file(stored_path, as_attachment=True, download_name=original_name)

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/comments', methods=['GET'])
def get_file_comments(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
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
            return ok({"comments": comments})

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/comments', methods=['POST'])
def add_file_comment(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
    data = request.get_json()
    comment = data.get('comment', '').strip()
    if not comment:
        return err("Comment cannot be empty", "VALIDATION_ERROR", 400)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO project_file_comments (file_id, user_id, comment) VALUES (%s, %s, %s)",
                        (file_id, user_id, comment))
            conn.commit()
            return ok(message="ok")

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/move', methods=['POST'])
def move_file(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not _can_move_file(project_id, file_id, user_id):
        return err("Permission denied", "FORBIDDEN", 403)
    data = request.get_json()
    target_folder_id = data.get('folder_id')
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if target_folder_id:
                cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s", (target_folder_id, project_id))
                if not cur.fetchone():
                    return err("Target folder not found in this project", "NOT_FOUND", 404)
            cur.execute("UPDATE project_files SET folder_id = %s WHERE id = %s AND project_id = %s", (target_folder_id, file_id, project_id))
            if cur.rowcount == 0:
                return err("File not found", "NOT_FOUND", 404)
            conn.commit()
            return ok(message="ok")

@admin_bp.route('/admin/projects/<int:project_id>/files/batch_move', methods=['POST'])
def batch_move_files(project_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)

    data = request.get_json()
    file_ids = data.get('file_ids', [])
    target_folder_id = data.get('folder_id')
    if not file_ids:
        return err("No files selected", "VALIDATION_ERROR", 400)
    if not target_folder_id:
        return err("Target folder required", "VALIDATION_ERROR", 400)

    role = get_user_role_in_project(project_id, user_id)
    if not role and not is_admin():
        return err("You are not a member of this project", "FORBIDDEN", 403)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s", (target_folder_id, project_id))
            if not cur.fetchone():
                return err("Target folder not found in this project", "NOT_FOUND", 404)

            placeholders = ','.join(['%s'] * len(file_ids))
            cur.execute(f"""
                SELECT id FROM project_files 
                WHERE id IN ({placeholders}) AND project_id = %s
            """, file_ids + [project_id])
            found = cur.fetchall()
            if len(found) != len(file_ids):
                return err("Some files not found in this project", "NOT_FOUND", 404)

            cur.execute(f"""
                UPDATE project_files SET folder_id = %s 
                WHERE id IN ({placeholders}) AND project_id = %s
            """, [target_folder_id] + file_ids + [project_id])
            conn.commit()
            return ok({"moved_count": len(file_ids)})

@admin_bp.route('/admin/projects/<int:project_id>/batch_download', methods=['POST'])
def batch_download_files(project_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
    data = request.get_json()
    file_ids = data.get('file_ids', [])
    if not file_ids:
        return err("No files selected", "VALIDATION_ERROR", 400)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            placeholders = ','.join(['%s'] * len(file_ids))
            cur.execute(
                f"SELECT stored_path, original_name FROM project_files WHERE id IN ({placeholders}) AND project_id = %s",
                file_ids + [project_id])
            files = cur.fetchall()
            if not files:
                return err("No valid files found", "NOT_FOUND", 404)
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for stored_path, original_name in files:
                    zipf.write(resolve_path(stored_path), original_name)
            zip_buffer.seek(0)
            return send_file(zip_buffer, as_attachment=True, download_name=f"project_{project_id}_files.zip",
                             mimetype='application/zip')

@admin_bp.route('/admin/projects/<int:project_id>/files/search', methods=['GET'])
def search_project_files(project_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return ok({"files": []})
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
            return ok({"files": files})

@admin_bp.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/comments', methods=['GET'])
def get_folder_comments(project_id, folder_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
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
            return ok({"comments": comments})

@admin_bp.route('/admin/projects/<int:project_id>/folders/<int:folder_id>/comments', methods=['POST'])
def add_folder_comment(project_id, folder_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
    data = request.get_json()
    comment = data.get('comment', '').strip()
    if not comment:
        return err("Comment cannot be empty", "VALIDATION_ERROR", 400)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO project_folder_comments (folder_id, user_id, comment) VALUES (%s, %s, %s)",
                        (folder_id, user_id, comment))
            conn.commit()
            return ok(message="ok")

@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/rename', methods=['PUT'])
def rename_project_file(project_id, file_id):
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not _can_edit_file(project_id, file_id, user_id):
        return err("Permission denied", "FORBIDDEN", 403)
    data = request.get_json()
    new_name = data.get('original_name', '').strip()
    if not new_name:
        return err("New name required", "VALIDATION_ERROR", 400)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE project_files SET original_name = %s WHERE id = %s AND project_id = %s",
                        (new_name, file_id, project_id))
            if cur.rowcount == 0:
                return err("File not found", "NOT_FOUND", 404)
            conn.commit()
    return ok(message="ok")





# ======================== AI Document Review ========================

AI_DOC_REVIEW_PROMPT = """你是一个资深的招标文件审查专家。请用以下五轴法审查这份文档，每个轴给1-10分，指出具体问题，最后给出综合评分和修改建议。

**审查五轴：**
1. **合规性** — 是否满足招标文件的所有硬性要求？有没有遗漏必需的资质、证书、签章？
2. **清晰度** — 语言是否清晰？结构和逻辑是否合理？数据和金额是否表述准确无歧义？
3. **完整性** — 所有必填部分是否齐全？技术方案、商务报价、资质证明是否完整？
4. **风险** — 是否存在不利条款、过高承诺、模糊免责声明、容易被质疑的计算或推理？
5. **专业性** — 格式、用词、排版是否专业？是否符合行业规范？

**输出格式（严格JSON，不要Markdown包裹）：**
{
  "scores": {"合规性": N, "清晰度": N, "完整性": N, "风险": N, "专业性": N},
  "overall": N,
  "verdict": "通过 / 需修改 / 不合格",
  "issues": [
    {"axis": "轴名", "severity": "高/中/低", "location": "段落或位置描述", "finding": "具体问题", "suggestion": "修改建议"}
  ],
  "summary": "一段中文总结（100字内），概括主要问题和整体评价"
}"""


@admin_bp.route('/admin/review/document', methods=['POST'])
@admin_required
def admin_review_document():
    """AI document review using five-axis methodology (code-review-and-quality skill)."""
    if 'file' not in request.files:
        return err("请上传要审查的文件", "NO_FILE", 400)

    file = request.files['file']
    if not file.filename:
        return err("文件名为空", "EMPTY_FILENAME", 400)

    try:
        text, _ = extract_text_from_file(file)
    except Exception as e:
        logger.error(f"Text extraction failed for review: {e}")
        return err("文件解析失败，请确认文件格式正确", "EXTRACT_FAILED", 400)

    if not text or len(text.strip()) < 50:
        return err("文件内容过短，无法审查（至少50字符）", "TOO_SHORT", 400)

    axes_param = request.form.get('axes', 'all')
    if axes_param != 'all':
        try:
            selected_axes = [a.strip() for a in axes_param.split(',')]
        except Exception:
            selected_axes = None
    else:
        selected_axes = None

    prompt = AI_DOC_REVIEW_PROMPT
    if selected_axes:
        axis_list = "、".join(selected_axes)
        prompt = prompt.replace(
            "**审查五轴：**",
            f"**审查五轴（用户仅选择了以下轴）：**\n仅审查用户选择的轴: {axis_list}"
        )

    try:
        from app.services.agent import get_agent
        agent = get_agent()
        config = {"configurable": {"thread_id": f"doc_review_{uuid.uuid4()}"}}
        response = agent.invoke(
            {"messages": [{"role": "user", "content": f"{prompt}\n\n=== 待审查文档 ===\n{text[:12000]}"}]},
            config
        )
        raw = response["messages"][-1].content
    except Exception as e:
        logger.error(f"AI review invoke failed: {e}", exc_info=True)
        return err("AI审查服务暂时不可用", "AI_UNAVAILABLE", 503)

    # Parse JSON from AI response
    try:
        import re as _re
        json_match = _re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            result = {"raw_analysis": raw, "parse_error": True}
    except json.JSONDecodeError:
        result = {"raw_analysis": raw, "parse_error": True}

    return ok(result, "审查完成")


@admin_bp.route('/admin/ingest/feedback', methods=['POST'])
def submit_ingest_feedback():
    """Submit feedback on an ingest/batch processing result."""
    data = request.get_json(silent=True) or {}
    task_id = data.get('task_id', '')
    rating = data.get('rating')

    if not task_id or rating is None:
        return jsonify({"success": False, "error": "缺少必填参数"}), 400
    if rating not in (-1, 1):
        return jsonify({"success": False, "error": "rating 必须为 -1 或 1"}), 400

    user_id = session.get('user_id', '')
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_feedback (user_id, source, target_id, rating)
                    VALUES (%s, %s, %s, %s)
                """, (user_id or '', 'ingest', task_id, rating))
                conn.commit()

        try:
            from app.services.training_logger import log_interaction
            _rating_map = {1: 5, -1: 1}
            log_interaction(
                thread_id=f"ingest_{task_id[:40]}",
                user_msg=f"文件处理反馈: task_id={task_id}",
                assistant_response=f"用户评分: {rating}",
                rating=_rating_map.get(rating, 3),
                source='ingest',
            )
        except Exception:
            logger.warning("Failed to log ingest feedback to training", exc_info=True)

        return jsonify({"success": True, "message": "感谢反馈!"})
    except Exception as e:
        logger.error(f"Ingest feedback error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ======================== Credit Check Routes ========================


# ── Sub-modules (route groups registered on the shared admin_bp) ──
from app.routes import admin_regeneration  # noqa: F401  (registers regeneration-vote routes)
from app.routes import admin_knowledge_lab  # noqa: F401  (registers admin knowledge-lab routes)
from app.routes import admin_ops  # noqa: F401  (registers todo/quote-tree/quote-anomaly routes)

