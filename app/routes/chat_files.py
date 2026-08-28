"""File-station and recycle-bin routes for the chat blueprint family.

Registered on the shared ``chat_bp`` Blueprint object from
app/routes/chat.py. Covers /upload_file, /download_original_file, /fetch_url,
file-station listing, project file loading, and the recycle bin (list/restore/
delete/empty).
"""
import hashlib
import os
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO

from flask import request, jsonify, session, send_file
from werkzeug.datastructures import FileStorage

from app.config import to_rel_path, resolve_path, USER_FILES_ORIGINAL_ROOT, allowed_file, logger
from app.database import get_db_connection, db_transaction
from app.utils.helpers import ok, err
from app.services.session_manager import get_user_id, ensure_user_exists, get_or_create_session, record_file_usage
from app.services.file_cache import add_to_cache
from app.routes.admin import is_admin
from app.routes.projects import can_access_project
from app.services.file_processing import extract_text_from_file
from app.routes.chat import chat_bp

from psycopg2.extras import RealDictCursor


@chat_bp.route('/upload_file', methods=['POST'])
def upload_file():
    """Upload a file — registered users get persistent storage, anonymous get temp storage."""
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"不支持的文件类型: {file.filename}"}), 400

    user_id = get_user_id()
    thread_id = session.get('thread_id')
    if not thread_id:
        thread_id = str(uuid.uuid4())
        session['thread_id'] = thread_id
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


    is_anon = session.get('consent_value', 0) != 1
    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Extract text
    fake_file = FileStorage(BytesIO(file_bytes), filename=file.filename)
    extracted_text, _ = extract_text_from_file(fake_file)
    if not extracted_text or extracted_text.startswith("["):
        extracted_text = ""

    # ── Anonymous: temp-only storage ──
    if is_anon:
        anon_files = session.get('anon_files', [])
        if len(anon_files) >= 5:
            return jsonify({"error": "匿名用户最多上传5个临时文件，请注册以解锁更多功能"}), 400
        if len(file_bytes) > 5 * 1024 * 1024:
            return jsonify({"error": "匿名用户单文件限制5MB，请注册以解锁"}), 400
        anon_files.append({
            'filename': file.filename,
            'hash': file_hash,
            'size': len(file_bytes),
            'text': extracted_text,
        })
        session['anon_files'] = anon_files
        session.modified = True
        add_to_cache(thread_id, file.filename, extracted_text, user_id)
        return jsonify({
            "success": True, "filename": file.filename, "is_anon": True,
            "anon_count": len(anon_files), "anon_max": 5
        })

    # Registered user: check for existing file by hash
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, filename, original_stored_path FROM user_files WHERE user_id = %s AND file_hash = %s", (user_id, file_hash))
            existing = cur.fetchone()
            if existing and request.form.get('force') != 'true':
                return jsonify({
                    "exists": True,
                    "file_id": existing[0],
                    "filename": existing[1],
                    "original_path": existing[2] if existing[2] else None
                })

    ext = os.path.splitext(file.filename)[1]
    unique_name = f"{file_hash}_{int(time.time())}{ext}"
    original_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)
    os.makedirs(original_dir, exist_ok=True)
    original_path = os.path.join(original_dir, unique_name)
    original_rel = to_rel_path(original_path)
    # Save original binary file
    with open(original_path, 'wb') as f:
        f.write(file_bytes)

    # Add to in‑memory cache
    add_to_cache(thread_id, file.filename, extracted_text, user_id)
    record_file_usage(thread_id, file.filename, 'standalone_upload', "上传文件供日后使用")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if existing and request.form.get('force') == 'true':
                old_path = resolve_path(existing[2])
                if old_path and os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass
                cur.execute("""
                    UPDATE user_files
                    SET filename = %s,
                        size_bytes = %s,
                        original_stored_path = %s,
                        file_hash = %s,
                        expires_at = NULL,
                        original_expires_at = NOW() + INTERVAL '3 days',
                        original_name = %s,
                        content = %s
                    WHERE id = %s
                """, (file.filename, len(file_bytes), original_rel, file_hash, file.filename, extracted_text, existing[0]))
            else:
                ensure_user_exists(user_id)
                cur.execute("""
                    INSERT INTO user_files (user_id, thread_id, filename, size_bytes, expires_at,
                                            original_stored_path, file_hash, original_expires_at, original_name, content)
                    VALUES (%s, %s, %s, %s, NULL, %s, %s, NOW() + INTERVAL '3 days', %s, %s)
                    ON CONFLICT (thread_id, filename) DO UPDATE SET
                        size_bytes = EXCLUDED.size_bytes,
                        original_stored_path = EXCLUDED.original_stored_path,
                        file_hash = EXCLUDED.file_hash,
                        original_expires_at = EXCLUDED.original_expires_at,
                        original_name = EXCLUDED.original_name,
                        content = EXCLUDED.content
                """, (user_id, thread_id, file.filename, len(file_bytes), original_rel, file_hash, file.filename, extracted_text))
            conn.commit()

    return jsonify({"success": True, "filename": file.filename})

@chat_bp.route('/download_original_file', methods=['POST'])
def download_original_file():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403

    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({"error": "Missing filename"}), 400

    user_id = get_user_id()
    thread_id = session.get('thread_id')
    if not thread_id:
        return jsonify({"error": "No active session"}), 400

    if session.get('consent_value', 0) != 1:
        return jsonify({
            "error": "anonymous_not_allowed",
            "message": "匿名用户无法下载原文件。请注册或登录账户后使用此功能。"
        }), 403

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT original_stored_path
                FROM user_files
                WHERE user_id = %s AND thread_id = %s AND filename = %s
                  AND (original_expires_at IS NULL OR original_expires_at > NOW())
            """, (user_id, thread_id, filename))
            row = cur.fetchone()
            if not row or not row[0]:
                return jsonify({"error": "Original file not found or expired"}), 404
            original_path = resolve_path(row[0])
            if not os.path.exists(original_path):
                return jsonify({"error": "File missing on server"}), 404
            return send_file(original_path, as_attachment=True, download_name=filename)

# ── Web page fetch (URL analysis) ──

@chat_bp.route('/fetch_url', methods=['POST'])
def fetch_url():
    """Fetch and extract text content from a web page URL."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Login required"}), 403
    data = request.get_json()
    url = data.get('url', '').strip()
    if not url:
        return jsonify({"error": "URL required"}), 400
    if not url.startswith(('http://', 'https://')):
        return jsonify({"error": "Invalid URL"}), 400

    try:
        from app.services.web_extractor import fetch_page, extract_text_from_html
        html, status = fetch_page(url, retries=2, timeout=15)
        text = extract_text_from_html(html)
        # Truncate to reasonable size
        if len(text) > 50000:
            text = text[:50000] + "\n\n[内容已截断，仅保留前50000字符]"
        return jsonify({"success": True, "text": text, "length": len(text), "url": url})
    except Exception as e:
        logger.error(f"fetch_url failed for {url}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch URL: {e}"}), 500

@chat_bp.route('/delete_file_station', methods=['POST'])
def delete_file_station():
    data = request.get_json()
    file_id = data.get('file_id')
    if not file_id:
        return jsonify({"error": "Missing file_id"}), 400

    user_id = get_user_id()
    is_anon = session.get('consent_value', 0) != 1

    if is_anon:
        anon_files = session.get('anon_files', [])
        idx = int(file_id.replace('anon_', '')) if file_id.startswith('anon_') else -1
        if 0 <= idx < len(anon_files):
            anon_files.pop(idx)
            session['anon_files'] = anon_files
            session.modified = True
            return jsonify({"success": True})
        return jsonify({"error": "File not found"}), 404

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, filename, original_name, content, size_bytes, original_stored_path, file_hash, thread_id, user_id
                    FROM user_files
                    WHERE id = %s AND user_id = %s
                """, (file_id, user_id))
                file_record = cur.fetchone()
                if not file_record:
                    return jsonify({"error": "File not found or not owned"}), 404

                cur.execute("""
                            INSERT INTO recycle_bin
                            (original_table, original_id, user_id, file_name, file_content, file_size,
                             original_stored_path, file_hash, thread_id, deleted_at, expires_at,
                             uploaded_by, deleted_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '3 days',
                                    %s, %s)
                            """, (
                                'user_files', file_record['id'], user_id, file_record['original_name'],
                                file_record['content'], file_record['size_bytes'], file_record['original_stored_path'],
                                file_record['file_hash'], file_record['thread_id'],
                                file_record['user_id'],
                                user_id
                            ))

                cur.execute("DELETE FROM user_files WHERE id = %s AND user_id = %s", (file_id, user_id))

                conn.commit()
                return jsonify({"success": True, "moved_to_recycle_bin": True})

@chat_bp.route('/get_file_station', methods=['GET'])
def get_file_station():
    user_id = get_user_id()
    is_admin_user = session.get('role') == 'admin'
    is_anon = session.get('consent_value', 0) != 1

    # ── Anonymous: return session-based temp files ──
    if is_anon:
        anon_files = session.get('anon_files', [])
        files = [{
            "id": f"anon_{i}",
            "filename": af.get('filename', ''),
            "size_bytes": af.get('size', 0),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": None,
            "is_anon": True,
            "uploaded_by_name": "匿名用户",
        } for i, af in enumerate(anon_files)]
        return jsonify({"files": files, "is_anon": True, "anon_note": "匿名用户文件仅本次会话有效，关闭页面后自动清除，不支持下载"})

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # User files
            cur.execute("""
                SELECT 
                    'user_file' as source,
                    uf.id::text as id,
                    uf.original_name as filename,
                    uf.size_bytes,
                    uf.created_at,
                    uf.expires_at,
                    uf.meta_data,
                    uf.user_id as owner_id,
                    (uf.user_id = %s) as can_delete,
                    NULL as project_name,
                    NULL as project_id,
                    NULL as folder_path,
                    (SELECT json_agg(
                        json_build_object(
                            'usage_type', fu.usage_type,
                            'question', fu.question,
                            'timestamp', fu.timestamp,
                            'thread_id', fu.thread_id
                        ) ORDER BY fu.timestamp DESC
                    ) FROM file_usage fu WHERE fu.user_id = uf.user_id AND fu.filename = uf.original_name LIMIT 10) as usage
                FROM user_files uf
                WHERE uf.user_id = %s AND (uf.expires_at IS NULL OR uf.expires_at > NOW())
                ORDER BY uf.created_at DESC
            """, (user_id, user_id))
            user_files = cur.fetchall()

            # Project files
            if is_admin_user:
                cur.execute("""
                    SELECT 
                        'project_file' as source,
                        pf.id::text as id,
                        pf.original_name as filename,
                        pf.file_size as size_bytes,
                        pf.uploaded_at as created_at,
                        NULL as expires_at,
                        p.name as project_name,
                        p.id as project_id,
                        (SELECT string_agg(f.name, '/') FROM project_folders f WHERE f.id = pf.folder_id) as folder_path,
                        NULL as usage
                    FROM project_files pf
                    JOIN projects p ON pf.project_id = p.id
                    ORDER BY pf.uploaded_at DESC
                """)
            else:
                cur.execute("""
                    SELECT 
                        'project_file' as source,
                        pf.id::text as id,
                        pf.original_name as filename,
                        pf.file_size as size_bytes,
                        pf.uploaded_at as created_at,
                        NULL as expires_at,
                        p.name as project_name,
                        p.id as project_id,
                        (SELECT string_agg(f.name, '/') FROM project_folders f WHERE f.id = pf.folder_id) as folder_path,
                        NULL as usage
                    FROM project_files pf
                    JOIN projects p ON pf.project_id = p.id
                    JOIN project_members pm ON p.id = pm.project_id
                    WHERE pm.user_id = %s
                    ORDER BY pf.uploaded_at DESC
                """, (user_id,))
            project_files = cur.fetchall()

    all_files = user_files + project_files
    return jsonify({"files": all_files, "is_admin": is_admin_user})

@chat_bp.route('/load_project_file', methods=['POST'])
def load_project_file():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = get_user_id()
    data = request.get_json()
    project_id = data.get('project_id')
    file_id = data.get('file_id')
    if not project_id or not file_id:
        return jsonify({"error": "Missing project_id or file_id"}), 400

    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT stored_path, original_name
                FROM project_files
                WHERE id = %s AND project_id = %s
            """, (file_id, project_id))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "File not found"}), 404
            stored_path, original_name = row
            stored_path = resolve_path(stored_path)
            if not os.path.exists(stored_path):
                return jsonify({"error": "File missing on server"}), 404

            with open(stored_path, 'rb') as f:
                file_bytes = f.read()

            fake_file = FileStorage(BytesIO(file_bytes), filename=original_name)
            text, _ = extract_text_from_file(fake_file)
            if not text or text.startswith("["):
                return jsonify({"error": "Could not extract text from file"}), 400

            return jsonify({"content": text, "filename": original_name})

# ---------- Batch compare endpoints ----------

@chat_bp.route('/get_recycle_bin', methods=['GET'])
def get_recycle_bin():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DO $$ 
                BEGIN
                    BEGIN
                        ALTER TABLE recycle_bin ADD COLUMN deletion_reason TEXT DEFAULT 'manual';
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                END $$;
            """)
            conn.commit()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            from app.services.recycle_bin_service import get_recycle_items
            return jsonify(get_recycle_items(user_id, cur))

@chat_bp.route('/restore_from_recycle_bin', methods=['POST'])
def restore_from_recycle_bin():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    item_id = data.get('item_id')
    source = data.get('source')
    section = data.get('section')
    restore_all = data.get('restore_all', False)

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if restore_all:
                    from app.services.recycle_bin_service import bulk_restore_all
                    restored_count = bulk_restore_all(section, user_id, conn, cur)
                    return jsonify({"success": True, "restored_count": restored_count})

                from app.services.recycle_bin_service import restore_recycle_item
                restore_recycle_item(item_id, source, conn, cur, user_id)
                conn.commit()
                return jsonify({"success": True})

@chat_bp.route('/delete_recycle_item', methods=['POST'])
def delete_recycle_item():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    item_id = data.get('item_id')
    source = data.get('source')

    if not item_id or not source:
        return jsonify({"error": "Missing item_id or source"}), 400

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                from app.services.recycle_bin_service import permanently_delete_item
                permanently_delete_item(item_id, source, cur, user_id)
                conn.commit()
                return jsonify({"success": True})

@chat_bp.route('/empty_recycle_bin', methods=['POST'])
def empty_recycle_bin():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    source = data.get('source')

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                from app.services.recycle_bin_service import empty_recycle_bin
                empty_recycle_bin(source, user_id, cur)
                conn.commit()
                return jsonify({"success": True})
