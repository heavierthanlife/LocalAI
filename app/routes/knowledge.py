"""Blueprint: knowledge routes (auto-extracted)."""
import os, json, uuid, time, logging, hashlib, io, threading
from datetime import datetime, timezone, timedelta
from flask import Blueprint, request, jsonify, session, send_file, render_template, url_for, Response

from app.config import BASE_DIR, DATA_DIR, TEMP_ROOT, TEMP_DIR, USER_FILES_ORIGINAL_ROOT, to_rel_path, resolve_path, logger
from app.database import get_db_connection, db_transaction
from app.utils.helpers import utc_now, beijing_now, safe_error_response, split_thinking_answer
import app.globals as g
from app.services.file_cache import file_cache_manager, add_to_cache, load_cache_from_db
from app.services.file_processing import extract_text_from_file
from app.services.document_classifier import classify_and_categorize

from psycopg2.extras import RealDictCursor

from app.config import is_valid_extracted_text, ALLOWED_EXTENSIONS
from app.routes.admin import login_required
from app.services.kb_skill_engine import generate_skill_for_file


def _try_index_file(file_id, content, source, metadata=None, skill_summary=None):
    """Fire-and-forget RAG index update in background thread."""
    if not content:
        return
    def _do():
        try:
            from app.services.rag_engine import index_file
            index_file(file_id, content, source, metadata, skill_summary=skill_summary)
        except Exception as e:
            logger.warning(f"Background index failed for {source}.{file_id}: {e}")
    t = threading.Thread(target=_do, daemon=True)
    t.start()


def _try_wiki_ingest(file_id, content, filename, source_type, metadata=None):
    """Dispatch wiki ingest Celery task (fire-and-forget).
    
    LLM-based wiki page generation from uploaded documents. Uses the existing
    wiki_ingest_task Celery task which calls ingest_file() with max_retries=2.
    Failures are logged but do not affect the upload flow.
    """
    if not content:
        return
    try:
        from celery_app import celery as celery_app
        celery_app.send_task('wiki_ingest_task', args=[file_id, content, filename, source_type, metadata or {}])
    except Exception as e:
        logger.warning(f"Wiki ingest dispatch failed for {source_type}.{file_id}: {e}")


def _try_entity_extract(file_id, content, filename, source_type, doc_type="general", wiki_category="general", metadata=None):
    """Fire-and-forget entity extraction in background thread.
    
    Runs LLM-based entity extraction, resolves against existing entity index,
    and creates/updates entity wiki pages. Failures are logged but do not affect
    the upload flow.
    """
    if not content or len(content) < 50:
        return
    def _do():
        try:
            from app.services.wiki_entity_service import process_upload_entity_extraction
            process_upload_entity_extraction(
                file_id, content, filename, source_type,
                doc_type, wiki_category, metadata or {}
            )
        except Exception as e:
            logger.warning(f"Entity extraction failed for {source_type}.{file_id}: {e}")
    t = threading.Thread(target=_do, daemon=True)
    t.start()


knowledge_bp = Blueprint('knowledge', __name__, template_folder=str(BASE_DIR / 'templates'), static_folder=str(BASE_DIR / 'static'))

@knowledge_bp.route('/knowledge_lab/upload', methods=['POST'])
@login_required  # you need to define this decorator or just check session
def upload_knowledge_lab_file():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Extract text
    file_bytes = file.read()
    file.seek(0)
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    from io import BytesIO
    from werkzeug.datastructures import FileStorage
    fake_file = FileStorage(BytesIO(file_bytes), filename=file.filename)
    text_content, _ = extract_text_from_file(fake_file)
    if not text_content or text_content.startswith("["):
        text_content = ""

    doc_type, wiki_category = classify_and_categorize(text_content, file.filename, file_hash)
    category = wiki_category

    # Save file permanently
    KNOWLEDGE_LAB_DIR = str(BASE_DIR / 'knowledge_lab_files')
    os.makedirs(KNOWLEDGE_LAB_DIR, exist_ok=True)
    unique_name = f"{file_hash}_{int(time.time())}_{file.filename}"
    stored_path = os.path.join(KNOWLEDGE_LAB_DIR, unique_name)
    stored_rel = to_rel_path(stored_path)
    with open(stored_path, 'wb') as f:
        f.write(file_bytes)

    # Auto-generate skill summary
    skill_summary = generate_skill_for_file(text_content, file.filename, "knowledge_lab") if text_content else None

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check for duplicate by hash
            cur.execute("SELECT id FROM knowledge_lab_files WHERE file_hash = %s", (file_hash,))
            existing = cur.fetchone()
            if existing:
                return jsonify({"error": "File already exists in knowledge lab", "file_id": existing[0]}), 409
            cur.execute("""
                        INSERT INTO knowledge_lab_files (user_id, filename, original_name, file_size, content,
                                                         file_hash, stored_path, skill_summary, skill_generated_at, category)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)
                        RETURNING id
                        """,
                        (user_id, file.filename, file.filename, len(file_bytes), text_content, file_hash, stored_rel, skill_summary, category))
            new_id = cur.fetchone()[0]
            conn.commit()
            # Background: index for RAG (with skill summary as priority chunks)
            _try_index_file(new_id, text_content, 'knowledge_lab',
                           {'original_name': file.filename, 'owner': user_id},
                           skill_summary=skill_summary)
            _try_wiki_ingest(new_id, text_content, file.filename, 'knowledge_lab',
                           {'original_name': file.filename, 'owner': user_id})
            _try_entity_extract(new_id, text_content, file.filename, 'knowledge_lab',
                               doc_type, wiki_category,
                               {'original_name': file.filename, 'owner': user_id})
            return jsonify({
                "success": True,
                "file_id": new_id,
                "filename": file.filename,
                "file_size": len(file_bytes),
                "category": category,
                "skill_generated": bool(skill_summary),
                "uploaded_at": datetime.now(timezone.utc).isoformat()
            })

@knowledge_bp.route('/knowledge_lab/list', methods=['GET'])
def list_knowledge_lab_files():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT id, filename, original_name, file_size, uploaded_at,
                               (skill_summary IS NOT NULL) as has_skill
                        FROM knowledge_lab_files
                        WHERE user_id = %s
                        ORDER BY uploaded_at DESC
                        """, (user_id,))
            files = cur.fetchall()
            return jsonify({"files": files})

@knowledge_bp.route('/knowledge_lab/skill/<int:file_id>', methods=['GET'])
def get_knowledge_lab_skill(file_id):
    """Download the auto-generated skill markdown file."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT skill_summary, original_name FROM knowledge_lab_files WHERE id = %s AND user_id = %s", (file_id, user_id))
            row = cur.fetchone()
            if not row or not row[0]:
                return jsonify({"error": "No skill file generated"}), 404
            from urllib.parse import quote
            safe_name = os.path.splitext(row[1])[0] + '_skill.md'
            encoded = quote(safe_name)
            return Response(row[0], mimetype='text/markdown',
                           headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}"})

@knowledge_bp.route('/knowledge_lab/content/<int:file_id>', methods=['GET'])
def get_knowledge_lab_content(file_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT content, original_name FROM knowledge_lab_files WHERE id = %s AND user_id = %s", (file_id, user_id))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "File not found"}), 404
            return jsonify({"content": row[0], "filename": row[1]})

@knowledge_bp.route('/knowledge_lab/rename/<int:file_id>', methods=['POST'])
def rename_knowledge_lab_file(file_id):
    """Rename a knowledge lab file (admin only) or its skill label."""
    if not session.get('user_id'):
        return jsonify({"error": "未登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"error": "仅管理员可重命名"}), 403
    data = request.get_json(silent=True) or {}
    new_name = data.get('name', '').strip()
    if not new_name:
        return jsonify({"error": "名称不能为空"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE knowledge_lab_files SET original_name = %s WHERE id = %s", (new_name, file_id))
            conn.commit()
    return jsonify({"status": "ok"})

@knowledge_bp.route('/knowledge_lab/rename_skill/<int:file_id>', methods=['POST'])
def rename_knowledge_skill(file_id):
    """Rename the skill_summary label for a knowledge lab file."""
    if not session.get('user_id'):
        return jsonify({"error": "未登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"error": "仅管理员可重命名"}), 403
    data = request.get_json(silent=True) or {}
    new_skill = data.get('skill', '').strip()
    if not new_skill:
        return jsonify({"error": "技能名称不能为空"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE knowledge_lab_files SET skill_summary = %s WHERE id = %s", (new_skill, file_id))
            conn.commit()
    from app.services.skill_auditor import invalidate_audit_cache
    invalidate_audit_cache()
    return jsonify({"status": "ok"})

@knowledge_bp.route('/company_kb/rename/<int:file_id>', methods=['POST'])
def rename_company_kb_file(file_id):
    """Rename a company knowledge base file (admin only)."""
    if not session.get('user_id'):
        return jsonify({"error": "未登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"error": "仅管理员可重命名"}), 403
    data = request.get_json(silent=True) or {}
    new_name = data.get('name', '').strip()
    if not new_name:
        return jsonify({"error": "名称不能为空"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE company_knowledge_base SET original_name = %s WHERE id = %s", (new_name, file_id))
            conn.commit()
    return jsonify({"status": "ok"})

@knowledge_bp.route('/knowledge_lab/delete/<int:file_id>', methods=['POST'])
def delete_knowledge_lab_file(file_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM knowledge_lab_files WHERE id = %s AND user_id = %s",
                            (file_id, user_id))
                row = cur.fetchone()
                if not row:
                    return jsonify({"error": "File not found"}), 404
                # Soft-delete: move to recycle bin
                cur.execute("""
                    INSERT INTO kb_recycle_bin (original_table, original_id, user_id, filename, original_name,
                                                file_size, content, file_hash, stored_path, uploaded_by, deleted_by)
                    VALUES ('knowledge_lab_files', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (row['id'], user_id, row['filename'], row['original_name'], row['file_size'],
                      row.get('content', ''), row['file_hash'], row.get('stored_path', ''), user_id, user_id))
                cur.execute("DELETE FROM knowledge_lab_files WHERE id = %s AND user_id = %s", (file_id, user_id))
                conn.commit()
                return jsonify({"success": True, "recycled": True})

# ── On-demand skill generation ──
@knowledge_bp.route('/knowledge_lab/generate_skill/<int:file_id>', methods=['POST'])
def generate_skill_on_demand(file_id):
    """Generate or regenerate skill for a specific KB file."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT content, original_name FROM knowledge_lab_files WHERE id = %s AND user_id = %s", (file_id, user_id))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "File not found"}), 404
            if not row['content']:
                return jsonify({"error": "No text content to extract skill from"}), 400
            skill = generate_skill_for_file(row['content'], row['original_name'], "knowledge_lab")
            if not skill:
                return jsonify({"error": "Insufficient content for skill extraction. 文件内容不足以提取技能框架。"}), 400
            skill_hash = hashlib.sha256(skill[:2000].encode('utf-8', errors='replace')).hexdigest()[:16]
            cur.execute("UPDATE knowledge_lab_files SET skill_summary = %s, skill_generated_at = NOW(), "
                        "skill_summary_hash = %s WHERE id = %s", (skill, skill_hash, file_id))
            conn.commit()
    return jsonify({"status": "ok", "skill_length": len(skill), "message": "技能已生成"})

@knowledge_bp.route('/project_files/<int:file_id>/generate_skill', methods=['POST'])
def generate_project_file_skill(file_id):
    """Generate skill for a project file."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT content, original_name FROM project_files WHERE id = %s", (file_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "File not found"}), 404
            if not row['content']:
                return jsonify({"error": "No text content available"}), 400
            skill = generate_skill_for_file(row['content'], row['original_name'], "project_file")
            if not skill:
                return jsonify({"error": "文件内容不足以提取技能框架", "hint": "需要至少200字的文本内容才能生成技能摘要"}), 400
            skill_hash = hashlib.sha256(skill[:2000].encode('utf-8', errors='replace')).hexdigest()[:16]
            cur.execute("UPDATE project_files SET skill_summary = %s, skill_generated_at = NOW(), "
                        "skill_summary_hash = %s WHERE id = %s", (skill, skill_hash, file_id))
            # Also save to user's personal KB permanently
            skill_name = os.path.splitext(row['original_name'])[0] + '_技能.md'
            skill_hash = hashlib.sha256(skill.encode()).hexdigest()
            cur.execute("""
                INSERT INTO knowledge_lab_files (user_id, filename, original_name, file_size, content, file_hash, stored_path, skill_summary, skill_generated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (file_hash) DO UPDATE SET skill_summary = EXCLUDED.skill_summary, skill_generated_at = NOW()
            """, (user_id, skill_name, row['original_name'], len(skill.encode()), skill, skill_hash, '', skill))
            conn.commit()
    return jsonify({"status": "ok", "skill_length": len(skill), "message": "技能已生成并保存到个人知识库"})

# ── Skill Auditor routes ──
def is_skill_auditor():
    """Check if current user has auditor privileges (admin or is_auditor flag)."""
    role = session.get('role', 'user')
    is_auditor = session.get('is_auditor', False)
    return role == 'admin' or is_auditor

@knowledge_bp.route('/admin/skill_audit', methods=['GET'])
def get_skill_audit():
    """Run AI skill audit and return suggestions.
    
    Query params:
        force=true  — bypass cache, run full re-analysis
        info        — return cache status only (no analysis)
    """
    if not is_skill_auditor():
        return jsonify({"error": "需要技能审核员权限"}), 403
    try:
        from app.services.skill_auditor import analyze_all_skills, _load_cache, _fetch_all_skill_fingerprints, _compute_usage_stats
        force = request.args.get('force', '').lower() == 'true'

        if request.args.get('info'):
            cache = _load_cache()
            return jsonify({
                "status": "ok",
                "cached_skills": len(cache.get('fingerprints', {})),
                "last_analysis": cache.get('last_full_analysis'),
                "has_results": cache.get('results') is not None,
            })

        with get_db_connection() as conn:
            # Always return basic stats immediately
            skills = _fetch_all_skill_fingerprints(conn)
            usage = _compute_usage_stats(skills)
            results = {
                'total_skills': len(skills),
                'unused_count': len(usage['unused']),
                'unused': usage['unused'],
                'promote_candidates': usage['promote_candidates'],
                'duplicate_pairs': 0,
                'duplicates': [],
                'audit_run_id': str(uuid.uuid4()),
            }
            # Try full analysis (may take time for model download)
            try:
                full = analyze_all_skills(conn, force=force)
                results.update(full)
            except Exception as e:
                logger.warning(f"Full skill analysis skipped (model unavailable?): {e}")
                results['analysis_skipped'] = True
                results['analysis_note'] = '相似度分析模型加载中，仅显示基础统计'
        return jsonify(results)
    except Exception as e:
        logger.error(f"Skill audit failed: {e}")
        return jsonify({"error": "审计分析失败，请稍后重试"}), 500

@knowledge_bp.route('/admin/skill_merge', methods=['POST'])
def merge_skills():
    """Merge two similar skills."""
    if not is_skill_auditor():
        return jsonify({"error": "需要技能审核员权限"}), 403
    data = request.get_json(silent=True) or {}
    keep_id = data.get('keep_id')
    merge_id = data.get('merge_id')
    source = data.get('source', 'knowledge_lab')
    if not keep_id or not merge_id:
        return jsonify({"error": "需要两个技能的ID"}), 400
    try:
        from app.services.skill_auditor import merge_skills as do_merge
        from app.services.admin_utils import log_admin_action
        with get_db_connection() as conn:
            ok = do_merge(conn, keep_id, merge_id, source)
        log_admin_action(session.get('user_id',''), session.get('username',''), 'SKILL_MERGE',
                        source, str(keep_id), column_name='merged_from', old_value=str(merge_id),
                        new_value='merged', success=ok)
        return jsonify({"status": "ok" if ok else "failed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@knowledge_bp.route('/admin/skill_archive/<int:skill_id>', methods=['POST'])
def archive_skill(skill_id):
    """Remove skill from a file (set skill_summary to NULL)."""
    if not is_skill_auditor():
        return jsonify({"error": "需要技能审核员权限"}), 403
    source = request.args.get('source', 'knowledge_lab')
    table_map = {'knowledge_lab': 'knowledge_lab_files', 'company': 'company_knowledge_base', 'project': 'project_files'}
    table = table_map.get(source, 'knowledge_lab_files')
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Save skill to recycle bin before NULLing
            cur.execute(f"SELECT skill_summary, original_name FROM {table} WHERE id = %s", (skill_id,))
            row = cur.fetchone()
            if row and row.get('skill_summary'):
                cur.execute("""
                    INSERT INTO kb_recycle_bin (original_table, original_id, user_id, filename,
                        content, skill_summary, deleted_by, deleted_at, expires_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,NOW(),NOW()+INTERVAL '3 days')
                """, (table, skill_id, session.get('user_id',''), 
                      row.get('original_name','') or f'skill_{skill_id}',
                      '', row['skill_summary'], session.get('user_id','')))
            cur.execute(f"UPDATE {table} SET skill_summary = NULL, skill_generated_at = NULL, "
                        f"skill_summary_hash = NULL WHERE id = %s", (skill_id,))
            conn.commit()
    from app.services.admin_utils import log_admin_action
    from app.services.skill_auditor import invalidate_audit_cache
    log_admin_action(session.get('user_id',''), session.get('username',''), 'SKILL_ARCHIVE',
                    table, str(skill_id), column_name='skill_summary',
                    old_value='present', new_value='archived')
    invalidate_audit_cache()
    return jsonify({"status": "ok"})

@knowledge_bp.route('/feedback', methods=['POST'])
def submit_knowledge_lab_feedback():
    """Submit feedback on a knowledge-lab skill extraction result."""
    data = request.get_json(silent=True) or {}
    file_id = data.get('file_id', '')
    source = data.get('source', '')
    rating = data.get('rating')

    if not file_id or rating is None:
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
                """, (user_id, 'knowledge_lab', file_id, rating))
                conn.commit()

        try:
            from app.services.training_logger import log_interaction
            _rating_map = {1: 5, -1: 1}
            log_interaction(
                thread_id=f"knowledge_lab_{file_id[:40]}",
                user_msg=f"知识库技能反馈: file_id={file_id} source={source}",
                assistant_response=f"用户评分: {rating}",
                rating=_rating_map.get(rating, 3),
                source='knowledge_lab',
            )
        except Exception:
            logger.warning("Failed to log knowledge_lab feedback to training", exc_info=True)

        return jsonify({"success": True, "message": "感谢反馈!"})
    except Exception as e:
        logger.error(f"Knowledge lab feedback error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@knowledge_bp.route('/admin/skill_audit/feedback', methods=['POST'])
def submit_skill_audit_feedback():
    """Submit feedback on a skill audit result."""
    data = request.get_json(silent=True) or {}
    audit_run_id = data.get('audit_run_id', '')
    rating = data.get('rating')

    if rating is None:
        return jsonify({"success": False, "error": "缺少 rating"}), 400
    if rating not in (-1, 1):
        return jsonify({"success": False, "error": "rating 必须为 -1 或 1"}), 400

    user_id = session.get('user_id', '')
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_feedback (user_id, source, target_id, rating)
                    VALUES (%s, %s, %s, %s)
                """, (user_id, 'skill_audit', audit_run_id or '', rating))
                conn.commit()

        try:
            from app.services.training_logger import log_interaction
            _rating_map = {1: 5, -1: 1}
            log_interaction(
                thread_id=f"skill_audit_{audit_run_id[:40] or 'unknown'}",
                user_msg=f"技能审计反馈: audit_run={audit_run_id}",
                assistant_response=f"用户评分: {rating}",
                rating=_rating_map.get(rating, 3),
                source='skill_audit',
            )
        except Exception:
            logger.warning("Failed to log skill_audit feedback to training", exc_info=True)

        return jsonify({"success": True, "message": "感谢反馈!"})
    except Exception as e:
        logger.error(f"Skill audit feedback error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@knowledge_bp.route('/admin/role', methods=['POST'])
def set_user_role():
    """Admin: toggle auditor flag (parallel role, does not change base role).
    Only accepts 'user' (remove auditor) or 'auditor' (grant auditor).
    Admin users cannot be edited — admin automatically includes auditor.
    """
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    role = data.get('role', 'user').strip()
    if role not in ('user', 'auditor'):
        return jsonify({"error": "无效角色，仅支持 user / auditor"}), 400
    is_auditor = (role == 'auditor')
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Refuse to modify admin users
            cur.execute("SELECT role FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "用户不存在"}), 404
            if row[0] == 'admin':
                return jsonify({"error": "不能修改管理员角色"}), 403
            cur.execute("UPDATE users SET is_auditor = %s WHERE username = %s", (is_auditor, username))
            conn.commit()
    # If admin is editing their own auditor flag, sync session
    current_user = session.get('username', '')
    if current_user == username:
        session['is_auditor'] = is_auditor
    return jsonify({"status": "ok", "username": username, "is_auditor": is_auditor})

# ── Admin: view all users' personal KB files ──
@knowledge_bp.route('/admin/all_user_kb', methods=['GET'])
# ── Admin: Training Data Pipeline ──
@knowledge_bp.route('/admin/training_stats', methods=['GET'])
def admin_training_stats():
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import get_training_stats
        return jsonify({"stats": get_training_stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_export', methods=['POST'])
def admin_training_export():
    """Export training data. mode: 'full' | 'incremental' | 'quality' (legacy full+quality).

    - 'full': export all data, update watermark
    - 'incremental': export only new data since last export (fast, production use)
    - 'quality': legacy alias for full export with quality filter
    """
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'incremental')  # default to incremental for safety

    try:
        from app.services.training_logger import (
            export_training_jsonl, export_training_jsonl_incremental,
            reset_export_watermark
        )

        if mode == 'reset_watermark':
            reset_export_watermark()
            return jsonify({"status": "ok", "message": "Export watermark reset — next export will be full"})

        if mode == 'incremental':
            # Uses quality filter from runtime_config (default ≥3 stars)
            path = export_training_jsonl_incremental()
            if path:
                basename = os.path.basename(path)
                return jsonify({
                    "status": "ok", "path": path, "mode": "incremental",
                    "message": f"Incremental export: {basename}"
                })
            return jsonify({
                "status": "ok", "message": "No new data to export (up to date with last export)",
                "mode": "incremental", "path": ""
            })

        # Full export
        min_rating = 0 if mode == 'all' else None  # None = auto from config (default 3)
        path = export_training_jsonl('manual', min_rating=min_rating)
        if path:
            mode_label = 'all data' if mode == 'all' else 'quality ≥3★'
            return jsonify({
                "status": "ok", "path": path, "mode": "full",
                "message": f"Full export ({mode_label}): {os.path.basename(path)}"
            })
        return jsonify({"error": "No training data to export"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_export_history', methods=['GET'])
def admin_training_export_history():
    """Return export history + watermark status for admin panel."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import get_export_history
        return jsonify({"status": "ok", "history": get_export_history()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_cleanup_stats', methods=['GET'])
def admin_training_cleanup_stats():
    """Return age distribution of training data for cleanup preview."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import get_training_cleanup_stats
        return jsonify({"status": "ok", "stats": get_training_cleanup_stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_cleanup', methods=['POST'])
def admin_training_cleanup():
    """Manually trigger training data cleanup (admin only)."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json(silent=True) or {}
    dry_run = data.get('dry_run', False)
    try:
        if dry_run:
            from app.services.training_logger import get_training_cleanup_stats
            return jsonify({"status": "ok", "dry_run": True, "stats": get_training_cleanup_stats()})
        from app.services.training_logger import cleanup_training_sessions
        retention = data.get('retention_days')
        if retention is not None:
            removed = cleanup_training_sessions(retention_days=int(retention))
        else:
            from app.services.runtime_config import get as rc_get
            removed = cleanup_training_sessions(retention_days=rc_get('training_retention_days', 90))
        return jsonify({"status": "ok", "removed": removed, "message": f"Purged {removed} old training sessions"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Training Data Health Check ──

@knowledge_bp.route('/admin/training_health', methods=['GET'])
def admin_training_health():
    """Run training data health check (scan + report, no repair)."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import run_training_health_check
        report = run_training_health_check(repair=False)
        return jsonify({"status": "ok", "report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_health', methods=['POST'])
def admin_training_health_repair():
    """Run health check with auto-repair enabled."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import run_training_health_check
        report = run_training_health_check(repair=True)
        return jsonify({"status": "ok", "report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_health_history', methods=['GET'])
def admin_training_health_history():
    """Return health check history and trend data."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import get_health_history
        return jsonify({"status": "ok", "history": get_health_history()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Export file management ──

@knowledge_bp.route('/admin/training_exports_cleanup', methods=['POST'])
def admin_training_exports_cleanup():
    """Clean up old export files (keep last N, configurable)."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json(silent=True) or {}
    try:
        from app.services.training_logger import cleanup_old_exports
        from app.services.admin_utils import log_admin_action
        keep = data.get('keep_count')
        result = cleanup_old_exports(keep_count=keep if keep else None)
        log_admin_action(session.get('user_id', ''), session.get('username', ''),
                        'EXPORT_CLEANUP', 'training', None,
                        column_name='export_files',
                        new_value=f"deleted:{len(result['deleted'])} kept:{result['kept']}")
        return jsonify({"status": "ok", "deleted": len(result['deleted']), "kept": result['kept'],
                        "deleted_files": result['deleted'],
                        "message": f"Deleted {len(result['deleted'])} old export files, kept {result['kept']}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_exports_list', methods=['GET'])
def admin_training_exports_list():
    """Return detailed list of export files with size and mtime."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import get_export_files_detail
        from app.services.runtime_config import get as rc_get
        return jsonify({
            "status": "ok",
            "files": get_export_files_detail(),
            "retention_count": rc_get('export_retention_count', 20),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_exports_delete/<filename>', methods=['POST'])
def admin_training_exports_delete(filename):
    """Delete a specific export file (admin only, logged)."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import delete_export_file
        from app.services.admin_utils import log_admin_action
        ok = delete_export_file(filename)
        if ok:
            log_admin_action(session.get('user_id', ''), session.get('username', ''),
                            'EXPORT_DELETE', 'training', None,
                            column_name='export_file', old_value=filename, new_value='deleted')
            return jsonify({"status": "ok", "message": f"Deleted {filename}"})
        return jsonify({"error": "File not found or invalid"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_exports_download/<filename>', methods=['GET'])
def admin_training_exports_download(filename):
    """Download a specific export file (admin only)."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    # Path traversal guard
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({"error": "Invalid filename"}), 400
    if not filename.endswith('.jsonl'):
        return jsonify({"error": "Only .jsonl files allowed"}), 400
    from app.services.training_logger import EXPORT_DIR
    filepath = os.path.join(EXPORT_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404
    from app.services.admin_utils import log_admin_action
    log_admin_action(session.get('user_id', ''), session.get('username', ''),
                    'EXPORT_DOWNLOAD', 'training', None,
                    column_name='export_file', new_value=filename)
    from flask import send_file
    return send_file(filepath, as_attachment=True, download_name=filename,
                     mimetype='application/x-ndjson')


# ── LoRA Fine-tuning Management ──

@knowledge_bp.route('/admin/training/lora/datasets', methods=['GET'])
def admin_lora_datasets():
    """List available datasets for LoRA training."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.lora_trainer import list_available_datasets
    return jsonify({"datasets": list_available_datasets()})


@knowledge_bp.route('/admin/training/lora/adapters', methods=['GET'])
def admin_lora_adapters():
    """List all registered LoRA adapters."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.lora_trainer import list_registered_adapters
    return jsonify({"adapters": list_registered_adapters()})


@knowledge_bp.route('/admin/training/run_lora', methods=['POST'])
def admin_run_lora():
    """Launch LoRA fine-tuning as a background subprocess.

    Body params:
        dataset: filename in exports/ or absolute path (required)
        base_model: HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)
        industry: industry label (default: bidding_agency)
        rank: LoRA rank (default: 16)
        epochs: training epochs (default: 3)
        learning_rate: peak LR (default: 2e-4)

    Returns:
        {task_id, pid, status, config}
    """
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json() or {}
    dataset = data.get('dataset', '').strip()
    if not dataset:
        return jsonify({"error": "dataset is required"}), 400
    from app.services.lora_trainer import launch_training
    try:
        result = launch_training(
            dataset=dataset,
            base_model=data.get('base_model'),
            industry=data.get('industry'),
            rank=data.get('rank'),
            epochs=data.get('epochs'),
            learning_rate=data.get('learning_rate'),
            max_seq_length=data.get('max_seq_length'),
            batch_size=data.get('batch_size'),
            gradient_accumulation=data.get('gradient_accumulation'),
        )
        from app.services.admin_utils import log_admin_action
        log_admin_action(session.get('user_id', ''), session.get('username', ''),
                        'LORA_TRAINING_LAUNCH', 'training', None,
                        column_name='task_id', new_value=result['task_id'])
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Launch failed: {str(e)[:200]}"}), 500


@knowledge_bp.route('/admin/training/lora/<industry>/activate', methods=['POST'])
def admin_lora_activate(industry):
    """Activate a LoRA adapter for an industry."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.lora_trainer import activate_adapter
    if activate_adapter(industry):
        return jsonify({"success": True, "industry": industry, "active": True})
    return jsonify({"error": "Adapter not found"}), 404


@knowledge_bp.route('/admin/training/lora/<industry>/deactivate', methods=['POST'])
def admin_lora_deactivate(industry):
    """Deactivate a LoRA adapter for an industry."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.lora_trainer import deactivate_adapter
    if deactivate_adapter(industry):
        return jsonify({"success": True, "industry": industry, "active": False})
    return jsonify({"error": "Adapter not found"}), 404


# ── Writing Style Profiles ──

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


# ── Work Report Generation ──

@knowledge_bp.route('/admin/generate_work_report', methods=['POST'])
def admin_generate_work_report():
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json(silent=True) or {}
    period = data.get('period', 'weekly')
    user_ids = data.get('user_ids', [])  # empty = all users
    if period not in ('daily', 'weekly', 'monthly', 'annual'):
        return jsonify({"error": "Invalid period, use daily/weekly/monthly/annual"}), 400

    # Calculate time range
    now = datetime.now(timezone.utc)
    if period == 'daily':
        since = now - timedelta(days=1)
    elif period == 'weekly':
        since = now - timedelta(days=7)
    elif period == 'monthly':
        since = now - timedelta(days=30)
    else:
        since = now - timedelta(days=365)
    end_date = now

    user_filter = ""
    user_params = [since]
    if user_ids:
        placeholders = ",".join(["%s"] * len(user_ids))
        user_filter = f" AND cs.user_id IN ({placeholders})"
        user_params.extend(user_ids)

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT COUNT(*) as cnt FROM chat_messages cm
                JOIN chat_sessions cs ON cm.thread_id = cs.thread_id
                WHERE cm.timestamp >= %s {user_filter}
            """, user_params)
            msg_count = cur.fetchone()['cnt']

            cur.execute(f"""
                SELECT COUNT(DISTINCT cs.thread_id) as cnt FROM chat_sessions cs
                JOIN chat_messages cm ON cm.thread_id = cs.thread_id
                WHERE cm.timestamp >= %s {user_filter}
            """, user_params)
            session_count = cur.fetchone()['cnt']

            u_params = [since]
            if user_ids: u_params.extend(user_ids)
            user_filter_no_alias = user_filter.replace('cs.user_id','user_id') if user_filter else ''
            cur.execute(f"SELECT COUNT(DISTINCT user_id) as cnt FROM chat_sessions WHERE updated_at >= %s {user_filter_no_alias}", u_params)
            user_count = cur.fetchone()['cnt']

            ur = [since]
            if user_ids:
                ur.extend(user_ids)
            cur.execute(f"SELECT COUNT(*) as cnt FROM file_usage WHERE usage_type IN ('knowledge_lab','company_kb') AND used_at >= %s {user_filter_no_alias}", ur)
            kb_usage = cur.fetchone()['cnt']

            cur.execute(f"SELECT COUNT(*) as cnt FROM credit_check_reports WHERE created_at >= %s {user_filter_no_alias}", ur)
            credit_count = cur.fetchone()['cnt']

            cur.execute(f"SELECT COUNT(*) as cnt FROM batch_comparison_results WHERE created_at >= %s {user_filter_no_alias}", ur)
            batch_count = cur.fetchone()['cnt']

            # Get selected user names for filename
            user_names = []
            if user_ids:
                cur.execute("SELECT username FROM users WHERE user_id IN %s", (tuple(user_ids),))
                user_names = [r['username'] for r in cur.fetchall()]
            username_tag = '_'.join(user_names) if user_names else '全体用户'

            # Topic categorization
            cur.execute(f"""
                SELECT cm.content FROM chat_messages cm
                JOIN chat_sessions cs ON cm.thread_id = cs.thread_id
                WHERE cm.role = 'user' AND cm.timestamp >= %s {user_filter}
                ORDER BY cm.timestamp DESC LIMIT 200
            """, user_params)
            user_msgs = [r['content'] or '' for r in cur.fetchall()]

            # Highlight sessions (pick 3 top by message count)
            cur.execute("""
                SELECT cs.thread_id, cs.title, u.username as owner, COUNT(*) as msg_cnt
                FROM chat_sessions cs
                JOIN chat_messages cm ON cm.thread_id = cs.thread_id
                LEFT JOIN users u ON cs.user_id = u.user_id
                WHERE cm.timestamp >= %s AND cm.role = 'user'
                GROUP BY cs.thread_id, cs.title, u.username
                ORDER BY msg_cnt DESC LIMIT 3
            """, (since,))
            highlights = [dict(r) for r in cur.fetchall()]

            # Previous period summary for trend comparison
            prev_since = since - (now - since)
            cur.execute("""
                SELECT cm.content FROM chat_messages cm
                WHERE cm.role = 'assistant' AND cm.content LIKE '%AI工作报告%'
                AND cm.timestamp >= %s AND cm.timestamp < %s
                ORDER BY cm.timestamp DESC LIMIT 1
            """, (prev_since, since))
            prev_row = cur.fetchone()
            prev_summary = prev_row['content'][:500] if prev_row else ""

    # Topic categorization from user messages
    topic_keywords = {
        '招投标': ['招标', '投标', '标书', '中标', '开标', '评标', '串标', '围标'],
        '征信查询': ['征信', '信用', '信用中国', '法院', '执行'],
        '文档对比': ['对比', '批量', '相似', '差异', '雷同'],
        '文件分析': ['分析', '提取', '识别', 'OCR', '解析'],
        '知识管理': ['知识库', '上传', '分类', '技能', '归档'],
        '项目管理': ['项目', '成员', '权限', '归档'],
        '通用咨询': ['问题', '帮助', '怎么', '如何', '什么是'],
    }
    topic_counts = {}
    for msg in user_msgs:
        for cat, kws in topic_keywords.items():
            if any(kw in msg for kw in kws):
                topic_counts[cat] = topic_counts.get(cat, 0) + 1
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

    # Build prompt and call LLM
    try:
        from app.services.analysis_prompts import build_work_report_prompt, WORK_REPORT_SYSTEM
        from app.services.llm_provider import create_chat_model
        from langchain_core.messages import SystemMessage, HumanMessage

        prompt = build_work_report_prompt(
            period=period,
            stats={'sessions': session_count, 'messages': msg_count, 'users': user_count,
                   'knowledge_files': kb_usage, 'credit_checks': credit_count, 'batch_compares': batch_count},
            topics=sorted_topics,
            highlights=[(r.get('title','') or f"会话{r['thread_id'][:8]}",
                        f"{r['msg_cnt']}条", r.get('owner','?'), str(since.date())) for r in highlights],
            previous_summary=prev_summary,
        )
        llm = create_chat_model(streaming=False, temperature=0.5, max_tokens=2000,
                                timeout=int(os.getenv("LLM_TIMEOUT", "120")))
        from app.services.prompt_safety import sanitize_for_prompt
        resp = llm.invoke([SystemMessage(content=WORK_REPORT_SYSTEM),
                          HumanMessage(content=sanitize_for_prompt(prompt, 'work_report'))])
        report = resp.content if hasattr(resp, 'content') else str(resp)
    except Exception as e:
        logger.error(f"Work report generation failed: {e}")
        return jsonify({"error": f"AI生成报告失败: {str(e)[:100]}"}), 500

    # Save report as Word (.docx) or .md file
    user_id = session.get('user_id')
    label = {'daily': '日报', 'weekly': '周报', 'monthly': '月报', 'annual': '年报'}.get(period, '报告')
    date_range = f"{since.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    filename_prefix = f"{username_tag}_{label}_{date_range}"
    # Use Word format with user's style profile
    try:
        from app.services.style_engine import get_user_style, generate_report_file
        style = get_user_style(user_id) if user_id else None
        report_path = generate_report_file(report, filename_prefix, f'{label} - {username_tag}', style)
        filename = os.path.basename(report_path)
    except Exception:
        report_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)
        report_path = os.path.join(report_dir, f'{filename_prefix}.md')
        filename = f'{filename_prefix}.md'
        os.makedirs(report_dir, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

    # ZIP: package all same-period report files together
    import zipfile
    zip_name = f"{username_tag}_{label}_{date_range}.zip"
    zip_path = os.path.join(report_dir, zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Include the just-generated report
        zf.write(report_path, filename)
        # Also include any other reports with matching date_range in same dir
        for fname in os.listdir(report_dir):
            if fname.endswith('.md') and date_range in fname and fname != filename:
                zf.write(os.path.join(report_dir, fname), fname)

    # Store both .md and .zip in DB
    file_hash = hashlib.sha256(report.encode()).hexdigest()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO user_files (user_id, thread_id, filename, size_bytes, original_stored_path,
                file_hash, original_expires_at, original_name) VALUES (%s, %s, %s, %s, %s, %s, NOW() + INTERVAL '90 days', %s)""",
                (user_id, session.get('thread_id', str(uuid.uuid4())), filename,
                 len(report.encode('utf-8')), to_rel_path(report_path), file_hash, filename))
            cur.execute("""INSERT INTO user_files (user_id, thread_id, filename, size_bytes, original_stored_path,
                file_hash, original_expires_at, original_name) VALUES (%s, %s, %s, %s, %s, %s, NOW() + INTERVAL '90 days', %s)""",
                (user_id, session.get('thread_id', str(uuid.uuid4())), zip_name,
                 os.path.getsize(zip_path), to_rel_path(zip_path), hashlib.sha256(open(zip_path,'rb').read()).hexdigest(), zip_name))
            conn.commit()

    return jsonify({
        "status": "ok",
        "filename": zip_name,
        "path": zip_path,
        "size_kb": round(os.path.getsize(zip_path) / 1024, 1),
        "period": period,
        "download_url": f"/download_original_file/{user_id}/{filename}",
    })


@knowledge_bp.route('/my_daily_report', methods=['POST'])
def my_daily_report():
    """Any registered user can generate their own daily work report (300-500 words)."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "请先登录"}), 401
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=1)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT COUNT(*) as cnt FROM chat_messages cm
                JOIN chat_sessions cs ON cm.thread_id = cs.thread_id
                WHERE cm.timestamp >= %s AND cs.user_id = %s""", (since, user_id))
            msg_count = cur.fetchone()['cnt']
            if msg_count < 2:
                return jsonify({"error": "今日对话不足，至少需要2条问答"}), 400

            cur.execute("""SELECT COUNT(DISTINCT cs.thread_id) as cnt FROM chat_sessions cs
                JOIN chat_messages cm ON cm.thread_id = cs.thread_id
                WHERE cm.timestamp >= %s AND cs.user_id = %s""", (since, user_id))
            session_count = cur.fetchone()['cnt']

            cur.execute("SELECT COUNT(*) as cnt FROM credit_check_reports WHERE created_at >= %s AND user_id = %s", (since, user_id))
            credit_count = cur.fetchone()['cnt']

            cur.execute("""SELECT cm.content FROM chat_messages cm
                JOIN chat_sessions cs ON cm.thread_id = cs.thread_id
                WHERE cm.role = 'user' AND cm.timestamp >= %s AND cs.user_id = %s
                ORDER BY cm.timestamp DESC LIMIT 80""", (since, user_id))
            user_msgs = [r['content'] or '' for r in cur.fetchall()]

    # Simple topic categorization
    topic_keywords = {
        '招投标': ['招标','投标','标书','中标','开标','评标'],
        '征信查询': ['征信','信用','信用中国','法院'],
        '文档对比': ['对比','批量','相似','雷同'],
        '文件分析': ['分析','提取','解析'],
    }
    topic_counts = {}
    for msg in user_msgs:
        for cat, kws in topic_keywords.items():
            if any(kw in msg for kw in kws):
                topic_counts[cat] = topic_counts.get(cat, 0) + 1
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

    try:
        from app.services.analysis_prompts import build_work_report_prompt, WORK_REPORT_SYSTEM
        from app.services.llm_provider import create_chat_model
        from langchain_core.messages import SystemMessage, HumanMessage

        prompt = build_work_report_prompt(
            period='daily',
            stats={'sessions': session_count, 'messages': msg_count, 'users': 1,
                   'knowledge_files': 0, 'credit_checks': credit_count, 'batch_compares': 0},
            topics=sorted_topics,
            highlights=[],
        )
        llm = create_chat_model(streaming=False, temperature=0.5, max_tokens=800,
                                timeout=int(os.getenv("LLM_TIMEOUT", "60")))
        from app.services.prompt_safety import sanitize_for_prompt
        resp = llm.invoke([SystemMessage(content=WORK_REPORT_SYSTEM),
                          HumanMessage(content=sanitize_for_prompt(prompt, 'work_summary'))])
        report = resp.content if hasattr(resp, 'content') else str(resp)
    except Exception as e:
        return jsonify({"error": f"AI生成失败: {str(e)[:100]}"}), 500

    username = session.get('username', user_id[:8])
    date_range = f"{since.strftime('%Y%m%d')}_{now.strftime('%Y%m%d')}"
    label = '日报'
    filename = f"{username}_{label}_{date_range}.md"
    report_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)
    report_path = os.path.join(report_dir, filename)
    os.makedirs(report_dir, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    file_hash = hashlib.sha256(report.encode()).hexdigest()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO user_files (user_id, thread_id, filename, size_bytes, original_stored_path,
                file_hash, original_expires_at, original_name) VALUES (%s, %s, %s, %s, %s, %s, NOW() + INTERVAL '90 days', %s)""",
                (user_id, session.get('thread_id', str(uuid.uuid4())), filename,
                 len(report.encode('utf-8')), to_rel_path(report_path), file_hash, filename))
            conn.commit()

    return jsonify({
        "status": "ok", "filename": filename, "size_kb": round(len(report.encode('utf-8'))/1024, 1),
        "download_url": f"/download_original_file/{user_id}/{filename}",
    })


# ── Admin: RAG index management ──
@knowledge_bp.route('/admin/rag_stats', methods=['GET'])
def admin_rag_stats():
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.rag_engine import get_index_stats
        return jsonify({"stats": get_index_stats()})
    except Exception as e:
        logger.warning(f"RAG stats failed: {e}")
        return jsonify({"stats": {"total": 0}})


@knowledge_bp.route('/admin/rag_rebuild', methods=['POST'])
def admin_rag_rebuild():
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.rag_engine import rebuild_all_indexes
        from app.services.admin_utils import log_admin_action
        admin_uid = session.get('user_id', '')
        admin_uname = session.get('username', '')
        with get_db_connection() as conn:
            total = rebuild_all_indexes(conn)
        log_admin_action(admin_uid, admin_uname, 'RAG_REBUILD', 'rag_index', None,
                        column_name='all', old_value=str(total), new_value='rebuild_complete')
        return jsonify({"status": "ok", "indexed": total})
    except Exception as e:
        logger.error(f"RAG rebuild failed: {e}", exc_info=True)
        return jsonify({"error": f"重建失败: {str(e)[:200]}"}), 500


@knowledge_bp.route('/admin/rag_delete/<source>/<int:file_id>', methods=['POST'])
def admin_rag_delete(source, file_id):
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.rag_engine import delete_file_index
        from app.services.admin_utils import log_admin_action
        admin_uid = session.get('user_id', '')
        admin_uname = session.get('username', '')
        delete_file_index(file_id, source)
        log_admin_action(admin_uid, admin_uname, 'RAG_DELETE', 'rag_index', str(file_id),
                        column_name='source', old_value=source, new_value='deleted')
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def admin_all_user_kb():
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    username_filter = request.args.get('username', '').strip()
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if username_filter:
                cur.execute("""
                    SELECT k.id, k.filename, k.original_name, k.file_size, k.uploaded_at,
                           u.username as owner,
                           (k.skill_summary IS NOT NULL) as is_skill,
                           (k.skill_summary IS NOT NULL AND k.skill_generated_at IS NOT NULL) as has_generated_skill
                    FROM knowledge_lab_files k JOIN users u ON k.user_id = u.user_id
                    WHERE u.username = %s
                    ORDER BY k.uploaded_at DESC
                """, (username_filter,))
            else:
                cur.execute("""
                    SELECT k.id, k.filename, k.original_name, k.file_size, k.uploaded_at,
                           u.username as owner,
                           (k.skill_summary IS NOT NULL) as is_skill,
                           (k.skill_summary IS NOT NULL AND k.skill_generated_at IS NOT NULL) as has_generated_skill
                    FROM knowledge_lab_files k JOIN users u ON k.user_id = u.user_id
                    ORDER BY u.username, k.uploaded_at DESC
                """)
            files = cur.fetchall()
    return jsonify({"files": files})

# ── Admin: promote personal KB file to company library ──
@knowledge_bp.route('/admin/promote_to_company/<int:file_id>', methods=['POST'])
def promote_to_company(file_id):
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Read the personal KB file
            cur.execute("""
                SELECT k.*, u.username FROM knowledge_lab_files k
                JOIN users u ON k.user_id = u.user_id
                WHERE k.id = %s
            """, (file_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "File not found"}), 404

            # Insert into company_knowledge_base with skill if available
            skill = row.get('skill_summary', '')
            cur.execute("""
                INSERT INTO company_knowledge_base (filename, original_name, file_size, content, stored_path, uploaded_by, uploaded_at, category, skill_summary, skill_generated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, NOW())
            """, (
                row['filename'],
                row['original_name'],
                row['file_size'],
                row.get('content', ''),
                row.get('stored_path', ''),
                session.get('user_id'),
                request.get_json(silent=True) and (request.get_json(silent=True) or {}).get('category', '技能库') or '技能库',
                skill
            ))
            conn.commit()
    return jsonify({"status": "ok", "message": f"已将 {row['original_name']}（来自 {row['username']}）加入公司知识库"})

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

# File station routes
