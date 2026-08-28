"""Company knowledge-base routes for the knowledge blueprint family.

Registered on the shared ``knowledge_bp`` Blueprint object from
app/routes/knowledge.py. Keeps the monolith smaller while preserving exact URLs.
"""
import hashlib
import os
import time

from flask import request, jsonify, session, Response

from app.config import BASE_DIR, to_rel_path, resolve_path, logger
from app.database import get_db_connection, db_transaction
from app.routes.knowledge import knowledge_bp
from app.routes.knowledge_shared import _try_index_file, _try_wiki_ingest, _try_entity_extract
from app.services.file_processing import extract_text_from_file
from app.services.document_classifier import classify_and_categorize
from app.services.kb_skill_engine import generate_skill_for_file

from psycopg2.extras import RealDictCursor


@knowledge_bp.route('/company_kb/upload', methods=['POST'])
def upload_company_kb_file():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id or session.get('role') != 'admin':
        return jsonify({"error": "Admin access required"}), 403

    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    category = request.form.get('category', '').strip()
    if not category:
        return jsonify({"error": "Category is required"}), 400

    # Read file bytes
    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Extract text
    from io import BytesIO
    from werkzeug.datastructures import FileStorage
    fake_file = FileStorage(BytesIO(file_bytes), filename=file.filename)
    text_content, _ = extract_text_from_file(fake_file)
    if not text_content or text_content.startswith("["):
        text_content = ""

    # Permanent storage
    COMPANY_KB_DIR = str(BASE_DIR / 'company_kb_files')
    os.makedirs(COMPANY_KB_DIR, exist_ok=True)
    unique_name = f"{file_hash}_{int(time.time())}_{file.filename}"
    stored_path = os.path.join(COMPANY_KB_DIR, unique_name)
    stored_rel = to_rel_path(stored_path)
    with open(stored_path, 'wb') as f:
        f.write(file_bytes)

    # Auto-generate skill summary
    co_skill_summary = generate_skill_for_file(text_content, file.filename, "company_kb") if text_content else None

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check for existing file by hash
            cur.execute("SELECT id, stored_path FROM company_knowledge_base WHERE file_hash = %s", (file_hash,))
            existing = cur.fetchone()
            if existing:
                # Overwrite: delete old physical file, update record
                old_path = resolve_path(existing[1])
                if old_path and os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except Exception as e:
                        logger.warning(f"Could not delete old file: {e}")
                co_skill_hash = hashlib.sha256(co_skill_summary[:2000].encode('utf-8', errors='replace')).hexdigest()[:16] if co_skill_summary else None
                cur.execute("""
                    UPDATE company_knowledge_base
                    SET filename = %s, original_name = %s, file_size = %s, content = %s,
                        stored_path = %s, category = %s, uploaded_by = %s, updated_at = NOW(),
                        skill_summary = %s, skill_generated_at = NOW(), skill_summary_hash = %s
                    WHERE id = %s
                """, (file.filename, file.filename, len(file_bytes), text_content, stored_rel, category, user_id,
                      co_skill_summary, co_skill_hash, existing[0]))
                conn.commit()
                _try_index_file(existing[0], text_content, 'company_kb',
                               {'original_name': file.filename, 'owner': user_id},
                               skill_summary=co_skill_summary)
                _try_wiki_ingest(existing[0], text_content, file.filename, 'company_kb',
                               {'original_name': file.filename, 'owner': user_id})
                co_doc_type, co_wiki_category = classify_and_categorize(text_content, file.filename, file_hash)
                _try_entity_extract(existing[0], text_content, file.filename, 'company_kb',
                                   co_doc_type, co_wiki_category,
                                   {'original_name': file.filename, 'owner': user_id})
                return jsonify({"success": True, "file_id": existing[0], "filename": file.filename, "category": category, "skill_generated": bool(co_skill_summary), "updated": True})
            else:
                # New file
                cur.execute("""
                    INSERT INTO company_knowledge_base (filename, original_name, file_size, content, file_hash, stored_path, category, uploaded_by, skill_summary, skill_generated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    RETURNING id
                """, (file.filename, file.filename, len(file_bytes), text_content, file_hash, stored_rel, category, user_id, co_skill_summary))
                new_id = cur.fetchone()[0]
                conn.commit()
                # Background: index for RAG (with skill summary as priority chunks)
                _try_index_file(new_id, text_content, 'company_kb',
                               {'original_name': file.filename, 'owner': user_id},
                               skill_summary=co_skill_summary)
                _try_wiki_ingest(new_id, text_content, file.filename, 'company_kb',
                               {'original_name': file.filename, 'owner': user_id})
                co_doc_type, co_wiki_category = classify_and_categorize(text_content, file.filename, file_hash)
                _try_entity_extract(new_id, text_content, file.filename, 'company_kb',
                                   co_doc_type, co_wiki_category,
                                   {'original_name': file.filename, 'owner': user_id})
                return jsonify({"success": True, "file_id": new_id, "filename": file.filename, "category": category})

@knowledge_bp.route('/company_kb/list', methods=['GET'])
def list_company_kb_files():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    category = request.args.get('category', '')
    search = request.args.get('search', '').strip()
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    offset = (page - 1) * per_page

    ts_config = 'simple'

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Build where clause and base parameters (for count query)
            where_parts = []
            count_params = []
            if category:
                where_parts.append("category = %s")
                count_params.append(category)
            if search:
                where_parts.append(f"to_tsvector('{ts_config}', content) @@ plainto_tsquery('{ts_config}', %s)")
                count_params.append(search)

            where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

            # Count total
            count_query = f"SELECT COUNT(*) as total FROM company_knowledge_base {where_clause}"
            cur.execute(count_query, count_params)
            total = cur.fetchone()['total']

            # Build main query parameters
            if search:
                # We need two copies of the search term: one for the rank function, one for the where clause
                main_params = []
                if category:
                    main_params.append(category)
                main_params.append(search)   # for rank
                main_params.append(search)   # for where
                # Add pagination
                main_params.extend([per_page, offset])

                query = f"""
                    SELECT id, original_name as filename, file_size, category, uploaded_at,
                           (SELECT username FROM users WHERE user_id = uploaded_by) as uploaded_by_name,
                           (skill_summary IS NOT NULL) as has_skill,
                           ts_rank(to_tsvector('{ts_config}', content), plainto_tsquery('{ts_config}', %s)) as rank
                    FROM company_knowledge_base
                    {where_clause}
                    ORDER BY rank DESC, uploaded_at DESC
                    LIMIT %s OFFSET %s
                """
            else:
                main_params = count_params.copy()
                main_params.extend([per_page, offset])
                query = f"""
                    SELECT id, original_name as filename, file_size, category, uploaded_at,
                           (SELECT username FROM users WHERE user_id = uploaded_by) as uploaded_by_name,
                           (skill_summary IS NOT NULL) as has_skill
                    FROM company_knowledge_base
                    {where_clause}
                    ORDER BY uploaded_at DESC
                    LIMIT %s OFFSET %s
                """

            cur.execute(query, main_params)
            files = cur.fetchall()

            return jsonify({
                "files": files,
                "total": total,
                "page": page,
                "per_page": per_page,
                "has_next": offset + per_page < total
            })

@knowledge_bp.route('/company_kb/search', methods=['GET'])
def search_company_kb():
    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return jsonify({"results": []})

    ts_config = 'simple'
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, original_name as filename, category,
                       ts_headline('{ts_config}', content, plainto_tsquery('{ts_config}', %s), 'MaxWords=30, MinWords=15') as snippet
                FROM company_knowledge_base
                WHERE to_tsvector('{ts_config}', content) @@ plainto_tsquery('{ts_config}', %s)
                ORDER BY ts_rank(to_tsvector('{ts_config}', content), plainto_tsquery('{ts_config}', %s)) DESC
                LIMIT 20
            """, (query, query, query))
            results = cur.fetchall()
            return jsonify({"results": results})

@knowledge_bp.route('/company_kb/content/<int:file_id>', methods=['GET'])
def get_company_kb_content(file_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT content, original_name FROM company_knowledge_base WHERE id = %s", (file_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "File not found"}), 404
            return jsonify({"content": row[0], "filename": row[1]})

@knowledge_bp.route('/company_kb/delete/<int:file_id>', methods=['POST'])
def delete_company_kb_file(file_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id or session.get('role') != 'admin':
        return jsonify({"error": "Admin access required"}), 403

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM company_knowledge_base WHERE id = %s", (file_id,))
                row = cur.fetchone()
                if not row:
                    return jsonify({"error": "File not found"}), 404
                # Soft-delete: move to recycle bin
                cur.execute("""
                    INSERT INTO kb_recycle_bin (original_table, original_id, user_id, filename, original_name,
                                                file_size, content, file_hash, stored_path, category, uploaded_by, deleted_by)
                    VALUES ('company_knowledge_base', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (row['id'], row.get('uploaded_by'), row['filename'], row['original_name'], row['file_size'],
                      row.get('content', ''), row['file_hash'], row.get('stored_path', ''),
                      row.get('category', ''), row.get('uploaded_by'), user_id))
                cur.execute("DELETE FROM company_knowledge_base WHERE id = %s", (file_id,))
                conn.commit()
                return jsonify({"success": True, "recycled": True})

@knowledge_bp.route('/company_kb/skill/<int:file_id>', methods=['GET'])
def get_company_kb_skill(file_id):
    """Download the auto-generated skill markdown for company KB file."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT skill_summary, original_name FROM company_knowledge_base WHERE id = %s", (file_id,))
            row = cur.fetchone()
            if not row or not row[0]:
                return jsonify({"error": "No skill file generated"}), 404
            from urllib.parse import quote
            safe_name = os.path.splitext(row[1])[0] + '_skill.md'
            encoded = quote(safe_name)
            return Response(row[0], mimetype='text/markdown',
                           headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}"})

@knowledge_bp.route('/company_kb/generate_skill/<int:file_id>', methods=['POST'])
def generate_company_kb_skill(file_id):
    """Generate skill summary for a company KB file on demand."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id or session.get('role') != 'admin':
        return jsonify({"error": "Admin access required"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT content, original_name FROM company_knowledge_base WHERE id = %s", (file_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "File not found"}), 404
            if not row['content']:
                return jsonify({"error": "No text content available"}), 400
            skill = generate_skill_for_file(row['content'], row['original_name'], "company_kb")
            if not skill:
                return jsonify({"error": "文件内容不足以提取技能框架"}), 400
            cur.execute("UPDATE company_knowledge_base SET skill_summary = %s, skill_generated_at = NOW() WHERE id = %s", (skill, file_id))
            conn.commit()
    return jsonify({"status": "ok", "skill_length": len(skill), "message": "技能已生成"})

@knowledge_bp.route('/company_kb/categories', methods=['GET'])
def get_company_kb_categories():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT category FROM company_knowledge_base WHERE category IS NOT NULL AND category != '' ORDER BY category")
            rows = cur.fetchall()
            categories = [row[0] for row in rows]
            return jsonify({"categories": categories})
