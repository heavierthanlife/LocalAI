"""Admin operations routes for the admin blueprint family.

Registered on the shared ``admin_bp`` Blueprint object from
app/routes/admin.py. Covers project todo system, quote tree system,
and quote-anomaly admin routes.
"""
import json
import logging
import re
import uuid

from flask import request, jsonify, session

from app.config import logger
from app.database import get_db_connection
from app.utils.helpers import ok, err
from app.routes.admin import admin_bp, admin_required, is_admin

from psycopg2.extras import RealDictCursor


# ======================== Project Todo System ========================

@admin_bp.route('/admin/projects/<int:project_id>/todos', methods=['GET'])
def project_todos_list(project_id):
    """Get current user's pending todos for this project."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, message_id, content_copy, original_role, original_author, status, created_at
                FROM project_todos
                WHERE project_id = %s AND user_id = %s AND status = 'pending'
                ORDER BY created_at ASC
            """, (project_id, user_id))
            todos = cur.fetchall()
    return ok({"todos": [{
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
        return err("Access denied", "FORBIDDEN", 403)
    data = request.get_json() or {}
    message_id = data.get('message_id')
    content_copy = (data.get('content_copy') or '').strip()
    if not content_copy:
        return err("内容不能为空", "VALIDATION_ERROR", 400)
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
                return err("待办最多5条，请先完成或删除现有待办", "VALIDATION_ERROR", 400)
            cur.execute("""
                INSERT INTO project_todos (project_id, user_id, message_id, content_copy, original_role, original_author)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
            """, (project_id, user_id, message_id, content_copy[:2000], original_role, original_author))
            todo_id = cur.fetchone()[0]
            conn.commit()
    return ok({"todo_id": todo_id})


@admin_bp.route('/admin/projects/<int:project_id>/todos/<int:todo_id>/done', methods=['POST'])
def project_todos_done(project_id, todo_id):
    """Mark a todo as done — records to project log, visible to admin only."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    username = session.get('username', user_id)
    if not user_id or not can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE project_todos SET status = 'done', done_at = NOW()
                WHERE id = %s AND project_id = %s AND user_id = %s AND status = 'pending'
                RETURNING content_copy, original_author
            """, (todo_id, project_id, user_id))
            row = cur.fetchone()
            if not row:
                return err("待办不存在或已完成", "NOT_FOUND", 404)
            content_copy, original_author = row
            cur.execute("""
                INSERT INTO project_ai_memory (project_id, user_id, role, content)
                VALUES (%s, %s, 'system', %s)
            """, (project_id, user_id, f"[TODO DONE] by @{username}: original from @{original_author}: {content_copy[:200]}"))
            conn.commit()
    return ok(message="ok")


@admin_bp.route('/admin/projects/<int:project_id>/todos/<int:todo_id>/remove', methods=['POST'])
def project_todos_remove(project_id, todo_id):
    """Completely remove a todo — no trace left."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM project_todos WHERE id = %s AND project_id = %s AND user_id = %s AND status = 'pending'",
                (todo_id, project_id, user_id)
            )
            conn.commit()
    return ok(message="ok")


@admin_bp.route('/admin/projects/<int:project_id>/todos/done_log', methods=['GET'])
def project_todos_done_log(project_id):
    """Admin-only: view completed todo records."""
    from app.routes.projects import is_admin
    if not is_admin():
        return err("Admin only", "FORBIDDEN", 403)
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
    return ok({"logs": [{
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
        return err("Access denied", "FORBIDDEN", 403)
    data = request.get_json() or {}
    quoted_message_id = data.get('quoted_message_id')
    parent_quote_id = data.get('parent_quote_id')
    thread_id = data.get('thread_id')
    if not quoted_message_id:
        return err("quoted_message_id required", "VALIDATION_ERROR", 400)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO message_quotes (project_id, quoted_message_id, parent_quote_id, thread_id)
                VALUES (%s, %s, %s, %s) RETURNING id
            """, (project_id, quoted_message_id, parent_quote_id, thread_id))
            quote_id = cur.fetchone()[0]
            conn.commit()
    return ok({"quote_id": quote_id})


@admin_bp.route('/admin/projects/<int:project_id>/quote_tree/<int:message_id>', methods=['GET'])
def project_quote_tree(project_id, message_id):
    """Get the quote tree for a message — returns full ancestry chain."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)
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
    return ok({"nodes": nodes})




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
    return ok({"results": [dict(r) for r in results], "total": total})


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
                return err("Not found", "NOT_FOUND", 404)
    return ok(dict(row))


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
    return ok({"results": [dict(r) for r in results], "total": total})


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
    return ok({"results": [dict(r) for r in results], "total": total})


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
                return err("Not found", "NOT_FOUND", 404)

            cur.execute("""
                SELECT * FROM entity_relationships WHERE task_id = %s
                ORDER BY module, confidence DESC
            """, (task_id,))
            relations = cur.fetchall()
    return ok({
        "summary": dict(summary),
        "relationships": [dict(r) for r in relations],
    })
