"""Blueprint: projects routes (auto-extracted)."""
import os, json, uuid, time, logging, hashlib, io
from flask import Blueprint, request, jsonify, session, send_file, render_template, url_for

from app.config import BASE_DIR, DATA_DIR, TEMP_ROOT, TEMP_DIR, USER_FILES_ORIGINAL_ROOT
from app.database import get_db_connection, db_transaction
from app.utils.helpers import utc_now, beijing_now, safe_error_response, split_thinking_answer
import app.globals as g
from app.services.file_cache import file_cache_manager, add_to_cache, load_cache_from_db

from psycopg2.extras import RealDictCursor
from app.routes.admin import is_admin, get_user_role_in_project

import zipfile, shutil, secrets as _secrets

projects_bp = Blueprint('projects', __name__, template_folder=str(BASE_DIR / 'templates'), static_folder=str(BASE_DIR / 'static'))

@projects_bp.route('/user_project_files', methods=['GET'])
def get_user_project_files():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT pf.id, pf.original_name, pf.file_size, p.name as project_name
                        FROM project_files pf
                                 JOIN projects p ON pf.project_id = p.id
                                 JOIN project_members pm ON p.id = pm.project_id
                        WHERE pm.user_id = %s
                        ORDER BY p.name, pf.uploaded_at DESC
                        """, (user_id,))
            files = cur.fetchall()
            return jsonify({"files": files})

def can_manage_files(project_id, user_id):
    if is_admin():
        return True
    role = get_user_role_in_project(project_id, user_id)
    return role in ('admin', 'manager')

def can_edit_file(project_id, file_id, user_id):
    if is_admin():
        return True
    role = get_user_role_in_project(project_id, user_id)
    if role == 'manager':
        return True
    if role == 'member':
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT uploaded_by FROM project_files WHERE id = %s AND project_id = %s",
                            (file_id, project_id))
                row = cur.fetchone()
                return row and row[0] == user_id
    return False

def can_move_file(project_id, file_id, user_id):
    if is_admin():
        return True
    role = get_user_role_in_project(project_id, user_id)
    if role in ('admin', 'manager', 'member'):
        return True
    return False

def can_edit_folder(project_id, folder_id, user_id):
    if is_admin():
        return True
    role = get_user_role_in_project(project_id, user_id)
    if role == 'manager':
        return True
    if role == 'member':
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT created_by FROM project_folders WHERE id = %s AND project_id = %s",
                            (folder_id, project_id))
                row = cur.fetchone()
                return row and row[0] == user_id
    return False

def can_manage_members(project_id, user_id):
    if is_admin():
        return True
    role = get_user_role_in_project(project_id, user_id)
    return role == 'manager'

def can_access_project(project_id, user_id):
    if is_admin():
        return True
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM project_members WHERE project_id = %s AND user_id = %s AND status = 'active'",
                        (project_id, user_id))
            return cur.fetchone() is not None

def user_has_any_project(user_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM project_members WHERE user_id = %s LIMIT 1", (user_id,))
            return cur.fetchone() is not None

# ---------- Project management routes ----------
