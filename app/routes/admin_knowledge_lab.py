"""Knowledge-lab admin routes for the admin blueprint family.

Registered on the shared ``admin_bp`` Blueprint object from
app/routes/admin.py. Covers admin knowledge-lab file management, ingest
oversight, and skill generation tasks under /admin/*.
"""
import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO

from flask import request, jsonify, session, current_app, Response
from werkzeug.security import check_password_hash

from app.config import resolve_path, logger
from app.database import get_db_connection, db_transaction
from app.utils.helpers import ok, err
import app.globals as g
from app.services.admin_utils import admin_rate_limiter, log_admin_action
from app.routes.auth import validate_table_column
from app.routes.admin import admin_bp, admin_required, _project_presence

from psycopg2.extras import RealDictCursor
from psycopg2 import sql


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
            return ok({"tables": tables})

@admin_bp.route('/admin/db_tables_overview', methods=['GET'])
@admin_required
def admin_db_tables_overview():
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                        SELECT tablename
                        FROM pg_tables
                        WHERE schemaname = 'public'
                        ORDER BY tablename
                        """)
            tables = [row['tablename'] for row in cur.fetchall()]
            result = []
            for t in tables:
                try:
                    cur.execute(f"SELECT count(*) AS cnt FROM {t}")
                    row = cur.fetchone()
                    result.append({"table_name": t, "row_count": row['cnt']})
                except Exception:
                    result.append({"table_name": t, "row_count": 0})
            return ok({"tables": result})

def _query_db_table(table, page, per_page, search, search_column):
    """Shared helper: run a paginated, searchable SELECT against any public table."""
    if not table:
        return err("Table name required", "VALIDATION_ERROR", 400)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = %s", (table,))
            if not cur.fetchone():
                return err("Invalid table name", "VALIDATION_ERROR", 400)

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

            return ok({
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
        return err("Table name required", "VALIDATION_ERROR", 400)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename=%s", (table,))
            if not cur.fetchone():
                return err("Invalid table", "VALIDATION_ERROR", 400)
            cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name=%s ORDER BY ordinal_position", (table,))
            cols = [dict(r) for r in cur.fetchall()]
    return ok({"columns": cols})


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
        return err("审计日志不可修改，仅可查看和导出", "FORBIDDEN", 403)
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
        return err("Admin authentication not configured", "SERVER_ERROR", 500)

    if not check_password_hash(admin_hash, pin):
        log_admin_action(admin_user_id, admin_username, 'UPDATE', table, row_id,
                         column, None, new_value, success=False,
                         error_message="Invalid admin PIN")
        return err("Invalid admin PIN", "FORBIDDEN", 403)

    # Validate table exists
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = %s", (table,))
            if not cur.fetchone():
                log_admin_action(admin_user_id, admin_username, 'UPDATE', table, row_id,
                                 column, None, new_value, success=False,
                                 error_message=f"Invalid table name: {table}")
                return err("Invalid table name", "VALIDATION_ERROR", 400)

    # Validate column exists
    if not validate_table_column(table, column):
        log_admin_action(admin_user_id, admin_username, 'UPDATE', table, row_id,
                         column, None, new_value, success=False,
                         error_message=f"Invalid column: {column}")
        return err(f"Column '{column}' does not exist", "VALIDATION_ERROR", 400)

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
                return err(f"Table '{table}' has no known primary key", "VALIDATION_ERROR", 400)
            pk_col = pk_col[0]

    if pk_col == 'id':
        try:
            row_id_val = int(row_id)
        except ValueError:
            log_admin_action(admin_user_id, admin_username, 'UPDATE', table, row_id,
                             column, None, new_value, success=False,
                             error_message="Invalid row_id (not an integer)")
            return err("Row ID must be an integer", "VALIDATION_ERROR", 400)
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
                return err("Row not found", "NOT_FOUND", 404)

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

    return ok(message="ok")

@admin_bp.route('/admin/db_delete_row', methods=['POST'])
@admin_required
@admin_rate_limiter
def admin_db_delete_row():
    data = request.get_json()
    table = data.get('table')
    if table in IMMUTABLE_TABLES:
        return err("审计日志不可删除，仅可查看和导出", "FORBIDDEN", 403)
    row_id = data.get('row_id')
    pin = data.get('pin', '').strip()
    admin_user_id = session.get('user_id')
    admin_username = session.get('username', 'admin')

    if not table or not row_id or row_id == 'undefined':
        log_admin_action(admin_user_id, admin_username, 'DELETE', table, row_id,
                         success=False, error_message="Missing table or row_id")
        return err("Missing table or valid row_id", "VALIDATION_ERROR", 400)

    admin_hash = current_app.config.get('ADMIN_PASSWORD_HASH')
    if not admin_hash:
        log_admin_action(admin_user_id, admin_username, 'DELETE', table, row_id,
                         success=False, error_message="Admin password hash not configured")
        return err("Admin authentication not configured", "SERVER_ERROR", 500)

    if not check_password_hash(admin_hash, pin):
        log_admin_action(admin_user_id, admin_username, 'DELETE', table, row_id,
                         success=False, error_message="Invalid admin PIN")
        return err("Invalid admin PIN", "FORBIDDEN", 403)

    # Validate table exists
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = %s", (table,))
            if not cur.fetchone():
                log_admin_action(admin_user_id, admin_username, 'DELETE', table, row_id,
                                 success=False, error_message=f"Invalid table: {table}")
                return err("Invalid table name", "VALIDATION_ERROR", 400)

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
                return err(f"Table '{table}' has no known primary key", "VALIDATION_ERROR", 400)
            pk_col = pk_col[0]

    if pk_col == 'id':
        try:
            row_id_val = int(row_id)
        except ValueError:
            log_admin_action(admin_user_id, admin_username, 'DELETE', table, row_id,
                             success=False, error_message="Invalid row_id (not integer)")
            return err("Row ID must be an integer", "VALIDATION_ERROR", 400)
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
                return err("Row not found", "NOT_FOUND", 404)

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

    return ok(message="ok")

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
                json_path = resolve_path(entry['archive_path'])
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
    return ok({"sessions": result})

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
                return err("Not found", "NOT_FOUND", 404)
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
    return ok(message="ok")

@admin_bp.route('/admin/archived_sessions', methods=['DELETE'])
@admin_required
def delete_selected_archived_sessions():
    """Delete selected archived sessions. Body: {"thread_ids": ["id1","id2"]}"""
    import os
    data = request.get_json() or {}
    thread_ids = data.get('thread_ids', [])
    if not thread_ids:
        return err("No thread_ids provided", "VALIDATION_ERROR", 400)
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
    return ok({"deleted": len(thread_ids)})

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
    return ok(message="ok")

@admin_bp.route('/admin/clear_file_cache', methods=['POST'])
@admin_required
def clear_file_cache():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE file_text_cache")
            conn.commit()
    return ok(message="ok")

@admin_bp.route('/admin/projects/<int:project_id>/ping', methods=['POST'])
def project_ping(project_id):
    """Update user's last-active timestamp for a project."""
    user_id = session.get('user_id')
    username = session.get('username', 'unknown')
    if not user_id: return err("Not logged in", "AUTH_REQUIRED", 401)
    _project_presence.setdefault(project_id, {})[user_id] = {
        'username': username, 'ts': time.time()
    }
    return ok(message="ok")

@admin_bp.route('/admin/projects/<int:project_id>/presence', methods=['GET'])
def project_presence(project_id):
    """Get currently active users in a project (active in last 60s)."""
    now = time.time()
    active = {}
    for uid, info in _project_presence.get(project_id, {}).items():
        if now - info['ts'] < 60:
            active[uid] = info['username']
    return ok({"active_users": active})

@admin_bp.route('/admin/projects/<int:project_id>/ai_assist/stream', methods=['POST'])
def project_ai_assist_stream(project_id):
    """Streaming SSE version of ai_assist — timer + typewriter effect in frontend."""
    from app.routes.projects import can_access_project
    import hashlib
    user_id = session.get('user_id')
    username = session.get('username', user_id)
    if not user_id:
        return err("未登录", "AUTH_REQUIRED", 401)
    if not can_access_project(project_id, user_id):
        return err("无权访问此项目", "FORBIDDEN", 403)

    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    if not query or len(query) < 3:
        return err("请用一句话描述您想生成的内容", "VALIDATION_ERROR", 400)
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
        return err(f"上下文构建失败: {str(e)[:200]}", "SERVER_ERROR", 500)

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
        return err("未登录", "AUTH_REQUIRED", 401)
    if not can_access_project(project_id, user_id):
        return err("无权访问此项目", "FORBIDDEN", 403)

    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    output_fmt = data.get('output_format', '').strip().lower()
    quoted_message_id = data.get('quoted_message_id')
    if output_fmt not in ('docx', 'xlsx', ''):
        output_fmt = ''
    if not query or len(query) < 3:
        return err("请用一句话描述您想生成的内容", "VALIDATION_ERROR", 400)

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
                "SELECT thread_id FROM chat_sessions WHERE project_id = %s AND (is_grilling = FALSE OR is_grilling IS NULL) LIMIT 1",
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
        return err(f"AI生成失败: {str(e)[:200]}", "SERVER_ERROR", 500)

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
                    cur.execute("SELECT thread_id FROM chat_sessions WHERE project_id = %s AND (is_grilling = FALSE OR is_grilling IS NULL) LIMIT 1", (project_id,))
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
    return ok(resp)

@admin_bp.route('/admin/projects/<int:project_id>/ai_activity', methods=['GET'])
def project_ai_activity(project_id):
    """Polling endpoint: returns recent AI activity AND chat messages since ?since=ISO timestamp."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return ok({"items": []})

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
                    AND (cs.is_grilling = FALSE OR cs.is_grilling IS NULL)
                    ORDER BY cm.timestamp ASC LIMIT 50
                """, (project_id, since))
            else:
                cur.execute("""
                    SELECT cm.id, cm.thread_id, cm.role, cm.content, cm.thinking, cm.timestamp
                    FROM chat_messages cm
                    JOIN chat_sessions cs ON cm.thread_id = cs.thread_id
                    WHERE cs.project_id = %s
                    AND (cs.is_grilling = FALSE OR cs.is_grilling IS NULL)
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
    return ok({"items": items, "now": datetime.now(timezone.utc).isoformat()})

@admin_bp.route('/admin/projects/<int:project_id>/unread_count', methods=['GET'])
def project_unread_count(project_id):
    """Return unread chat messages count since user's last read position."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return ok({"count": 0})

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
                  AND (cs.is_grilling = FALSE OR cs.is_grilling IS NULL)
                  AND cm.timestamp > %s
            """, (project_id, since))
            count = cur.fetchone()['cnt']
    return ok({"count": count, "since": since})

@admin_bp.route('/admin/projects/<int:project_id>/mark_read', methods=['POST'])
def project_mark_read(project_id):
    """Update user's last_read_at for this project."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)

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
    return ok(message="ok")
