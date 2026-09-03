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
from app.routes.knowledge_shared import (
    _try_index_file, _try_wiki_ingest, _try_entity_extract, is_skill_auditor,
)


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
            cur.execute("SELECT content, original_name, project_id FROM project_files WHERE id = %s", (file_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "File not found"}), 404
            # QA-Loop C2: prevent IDOR — user must be admin or a member of the file's project
            from app.routes.admin import is_admin as check_admin
            if not check_admin():
                if not row['project_id']:
                    return jsonify({"error": "文件不属于任何项目"}), 403
                from app.routes.admin import get_user_role_in_project
                if get_user_role_in_project(row['project_id'], user_id) is None:
                    return jsonify({"error": "无权访问该项目"}), 403
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
            kb_file_hash = hashlib.sha256(skill.encode()).hexdigest()
            cur.execute("""
                INSERT INTO knowledge_lab_files (user_id, filename, original_name, file_size, content, file_hash, stored_path, skill_summary, skill_generated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (file_hash) DO UPDATE SET skill_summary = EXCLUDED.skill_summary, skill_generated_at = NOW()
            """, (user_id, skill_name, row['original_name'], len(skill.encode()), skill, kb_file_hash, '', skill))
            conn.commit()
    return jsonify({"status": "ok", "skill_length": len(skill), "message": "技能已生成并保存到个人知识库"})

# ── Skill Auditor routes ──
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
# (路由装饰器挂在下方 admin_all_user_kb 定义处，避免空装饰器误绑下一个函数)


# ── Writing Style Profiles ──





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
    user_filter_no_alias = ""
    user_params = [since]
    if user_ids:
        placeholders = ",".join(["%s"] * len(user_ids))
        # QA-Loop C7: 结构化构建带/不带表别名的两套 filter，
        # 不再用 str.replace('cs.user_id','user_id') 后处理（对空格/换行格式敏感会静默失败）
        user_filter = f" AND cs.user_id IN ({placeholders})"
        user_filter_no_alias = f" AND user_id IN ({placeholders})"
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
    # QA-Loop C5: 显式 with open 关闭句柄，避免匿名 open() 在异常/循环路径泄漏 fd
    with open(zip_path, 'rb') as _zf:
        zip_hash = hashlib.sha256(_zf.read()).hexdigest()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO user_files (user_id, thread_id, filename, size_bytes, original_stored_path,
                file_hash, original_expires_at, original_name) VALUES (%s, %s, %s, %s, %s, %s, NOW() + INTERVAL '90 days', %s)""",
                (user_id, session.get('thread_id', str(uuid.uuid4())), filename,
                 len(report.encode('utf-8')), to_rel_path(report_path), file_hash, filename))
            cur.execute("""INSERT INTO user_files (user_id, thread_id, filename, size_bytes, original_stored_path,
                file_hash, original_expires_at, original_name) VALUES (%s, %s, %s, %s, %s, %s, NOW() + INTERVAL '90 days', %s)""",
                (user_id, session.get('thread_id', str(uuid.uuid4())), zip_name,
                 os.path.getsize(zip_path), to_rel_path(zip_path), zip_hash, zip_name))
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


@knowledge_bp.route('/admin/all_user_kb', methods=['GET'])
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



# File station routes


# ── Sub-modules (route groups registered on the shared knowledge_bp) ──
from app.routes import knowledge_notebook  # noqa: F401  (registers /notebook/* routes)
from app.routes import knowledge_company_kb  # noqa: F401  (registers /company_kb/* routes)
from app.routes import knowledge_style  # noqa: F401  (registers /my_writing_style + /admin/user_styles)
from app.routes import knowledge_ingest  # noqa: F401  (registers /admin/ingest/*)
from app.routes import knowledge_training  # noqa: F401  (registers /admin/training* + lora)
