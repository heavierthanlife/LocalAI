"""Regeneration-vote admin routes for the admin blueprint family.

Registered on the shared ``admin_bp`` Blueprint object from
app/routes/admin.py. Covers the AI-document regeneration vote system:
number/negation extraction helpers, vote records, evaluation, and review.
"""
import json
import logging
import os
import random
import re
import shutil
import time
import uuid
from io import BytesIO

from flask import request, jsonify, session, send_file

from app.config import resolve_path
from app.database import get_db_connection, db_transaction
from app.utils.helpers import ok, err
import app.globals as g
from app.services.admin_utils import log_admin_action
from app.routes.admin import admin_bp, admin_required, is_admin

from psycopg2.extras import RealDictCursor
from psycopg2 import sql


logger = logging.getLogger(__name__)


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
        return err("Access denied", "FORBIDDEN", 403)
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
    return ok({"votes": [{
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
        return err("Access denied", "FORBIDDEN", 403)
    data = request.get_json() or {}
    vote_choice = data.get('vote')  # 'keep_original' or 'replace'
    if vote_choice not in ('keep_original', 'replace'):
        return err("Invalid vote", "VALIDATION_ERROR", 400)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check vote is still active
            cur.execute("SELECT status, expires_at FROM regen_votes WHERE id = %s AND project_id = %s", (vote_id, project_id))
            row = cur.fetchone()
            if not row or row[0] != 'active':
                return err("投票已结束", "VALIDATION_ERROR", 400)
            # Upsert ballot
            cur.execute("""
                INSERT INTO regen_vote_ballots (vote_id, voter_id, vote)
                VALUES (%s, %s, %s)
                ON CONFLICT (vote_id, voter_id) DO UPDATE SET vote = EXCLUDED.vote
            """, (vote_id, user_id, vote_choice))
            conn.commit()
    return ok(message="ok")


@admin_bp.route('/admin/projects/<int:project_id>/regen_votes/<int:vote_id>/resolve', methods=['POST'])
def project_regen_vote_resolve(project_id, vote_id):
    """Resolve a vote — manager decides on draw, or auto-resolve on timeout."""
    from app.routes.projects import can_manage_members
    user_id = session.get('user_id')
    if not user_id or not can_manage_members(project_id, user_id):
        return err("仅项目经理可裁决", "FORBIDDEN", 403)
    data = request.get_json() or {}
    decision = data.get('decision')  # 'keep_original' or 'replace'
    if decision not in ('keep_original', 'replace'):
        return err("Invalid decision", "VALIDATION_ERROR", 400)
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
    return ok(message="ok")


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
        return err("未登录", "AUTH_REQUIRED", 401)
    if not can_access_project(project_id, user_id):
        return err("无权访问", "FORBIDDEN", 403)

    fmt = request.args.get('format', 'docx').strip().lower()
    if fmt not in ('docx', 'xlsx'):
        return err("格式仅支持 docx / xlsx", "VALIDATION_ERROR", 400)

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT content, content_md FROM project_ai_memory WHERE id = %s AND project_id = %s",
                (memory_id, project_id)
            )
            row = cur.fetchone()
            if not row:
                return err("记录不存在", "NOT_FOUND", 404)

    md_text = row.get('content_md') or row.get('content') or ''
    if not md_text.strip():
        return err("内容为空", "VALIDATION_ERROR", 400)

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
        return err(f"文件生成失败: {str(e)[:200]}", "SERVER_ERROR", 500)

@admin_bp.route('/admin/projects/<int:project_id>/my_workflow', methods=['GET'])
def get_my_workflow(project_id):
    """Get current member's custom workflow for this project."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return err("无权访问", "FORBIDDEN", 403)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM member_workflows WHERE project_id=%s AND user_id=%s",
                (project_id, user_id))
            row = cur.fetchone()
    return ok({"workflow": dict(row) if row else None, "needs_setup": row is None})

@admin_bp.route('/admin/projects/<int:project_id>/my_workflow', methods=['POST'])
def save_my_workflow(project_id):
    """Save/update member's custom workflow steps."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return err("无权访问", "FORBIDDEN", 403)
    data = request.get_json(silent=True) or {}
    steps = data.get('steps', [])
    name = data.get('name', '默认工作流').strip() or '默认工作流'
    if not steps or not isinstance(steps, list):
        return err("请至少定义一个步骤", "VALIDATION_ERROR", 400)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO member_workflows (project_id, user_id, workflow_name, steps, updated_at)
                VALUES (%s,%s,%s,%s,NOW())
                ON CONFLICT (project_id, user_id)
                DO UPDATE SET workflow_name=%s, steps=%s, updated_at=NOW()
            """, (project_id, user_id, name, json.dumps(steps), name, json.dumps(steps)))
            conn.commit()
    return ok(message="ok")

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
        return err("无权访问", "FORBIDDEN", 403)

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
        return err("请先设置工作流", "VALIDATION_ERROR", 400)

    steps = wf['steps'] if isinstance(wf['steps'], list) else json.loads(wf['steps'])
    if step_index >= len(steps):
        return ok({"message": "所有步骤已完成"})

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
        return ok(resp)

    except Exception as e:
        logger.error(f"Workflow step error: {e}")
        return err(f"执行失败: {str(e)[:200]}", "SERVER_ERROR", 500)

@admin_bp.route('/admin/projects/<int:project_id>/workflow_kpi', methods=['GET'])
@admin_required
def project_workflow_kpi(project_id):
    """Get KPI stats for all project members."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return err("无权访问", "FORBIDDEN", 403)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT wk.*, u.username FROM workflow_kpi wk
                LEFT JOIN users u ON wk.user_id = u.user_id
                WHERE wk.project_id = %s ORDER BY wk.generations DESC
            """, (project_id,))
            rows = cur.fetchall()
    return ok({"kpi": [dict(r) for r in rows]})

@admin_bp.route('/admin/projects/<int:project_id>/ai_workflow', methods=['POST'])
def project_ai_workflow(project_id):
    """Multi-step document workflow: draft → review → revise → finalize."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return err("无权访问", "FORBIDDEN", 403)

    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    if not query or len(query) < 3:
        return err("请描述您需要起草的文档", "VALIDATION_ERROR", 400)

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
        return ok({**result}, "ok")
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return err(f"工作流执行失败: {str(e)[:200]}", "SERVER_ERROR", 500)

@admin_bp.route('/admin/projects/<int:project_id>/ai_analyze', methods=['POST'])
def project_ai_analyze(project_id):
    """Data analysis: upload Excel/CSV → pandas analysis → comparison matrix + anomaly report."""
    from app.routes.projects import can_access_project
    user_id = session.get('user_id')
    if not user_id or not can_access_project(project_id, user_id):
        return err("无权访问", "FORBIDDEN", 403)

    file = request.files.get('file')
    if not file:
        return err("请上传Excel或CSV文件", "VALIDATION_ERROR", 400)

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
            return err("文件需要包含至少两列数据", "VALIDATION_ERROR", 400)

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

        return ok({"analysis": result}, "ok")
    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
        return err(f"分析失败: {str(e)[:200]}", "SERVER_ERROR", 500)

@admin_bp.route('/admin/projects/<int:project_id>/ai_memory', methods=['POST'])
def project_ai_sync_chat(project_id):
    """Sync a chat message from the project chat tab into AI memory.
    Called when user sends a message in a project-scoped chat session.
    Accepts: { role, content }
    """
    user_id = session.get('user_id')
    if not user_id:
        return err("未登录", "AUTH_REQUIRED", 401)
    
    data = request.get_json(silent=True) or {}
    role = data.get('role', 'user')
    content = data.get('content', '').strip()
    if not content or len(content) < 2:
        return ok({"status": "skipped"})

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO project_ai_memory (project_id, user_id, role, content) VALUES (%s,%s,%s,%s)",
                    (project_id, user_id, role, content[:2000])
                )
                conn.commit()
        return ok(message="ok")
    except Exception as e:
        return err(str(e)[:200], "SERVER_ERROR", 500)

@admin_bp.route('/admin/analytics', methods=['GET'])
def admin_analytics():
    """Return usage statistics — admin sees all users, regular users see own stats."""
    user_id = session.get('user_id')
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)

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
                try:
                    cur.execute("SELECT COUNT(*) as cnt FROM credit_check_reports")
                    stats['credit_checks'] = cur.fetchone()['cnt']
                except Exception:
                    conn.rollback()
                    stats['credit_checks'] = 0
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
                try:
                    cur.execute("SELECT COUNT(*) as cnt FROM credit_check_reports WHERE user_id = %s", (user_id,))
                    stats['credit_checks'] = cur.fetchone()['cnt']
                except Exception:
                    conn.rollback()
                    stats['credit_checks'] = 0
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
                try:
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
                except Exception:
                    conn.rollback()
                    stats['storage_breakdown'] = {}

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
    return ok(stats)

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
    return ok({
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
        return err("备注不能为空", "VALIDATION_ERROR", 400)
    log_admin_action(
        session.get('user_id', ''),
        session.get('username', ''),
        'ADMIN_NOTE', 'system', None,
        column_name='note', new_value=note[:500]
    )
    return ok(message="ok")


@admin_bp.route('/admin/approve_delete/<username>', methods=['POST'])
@admin_required
def admin_approve_delete(username):
    """Admin approves a user's deletion request, sends 4-digit code to user."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT user_id, email, deletion_requested FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            if not row:
                return err("用户不存在", "NOT_FOUND", 404)
            if not row['deletion_requested']:
                return err("该用户未申请删除", "VALIDATION_ERROR", 400)
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
    return ok({"hint": f"验证码{'已发送至 '+user_email if user_email else ': '+code}"}, "ok")


@admin_bp.route('/admin/pending_deletions', methods=['GET'])
@admin_required
def admin_pending_deletions():
    """List users with pending deletion requests."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT username, email, deletion_requested FROM users WHERE deletion_requested = TRUE")
            users = cur.fetchall()
    return ok({"users": users})


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
    return ok({"users": users, "deposits": deposits})


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
        return err("Missing target user", "VALIDATION_ERROR", 400)
    admin_uid = session.get('user_id', ''); admin_uname = session.get('username', '')
    transferred = 0
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users WHERE user_id = %s", (target_user_id,))
                if not cur.fetchone(): return err("Target user not found", "NOT_FOUND", 404)
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
    return ok({"transferred": transferred}, "ok")


# ── System Prompt Management ──

@admin_bp.route('/admin/system_prompt', methods=['GET'])
@admin_required
def get_system_prompt():
    """Return the current agent system prompt (admin only)."""
    from app.globals import AGENT_SYSTEM_PROMPT
    return ok({"prompt": AGENT_SYSTEM_PROMPT.strip()})


@admin_bp.route('/admin/system_prompt', methods=['POST'])
@admin_required
def set_system_prompt():
    """Update the agent system prompt and persist to disk."""
    from app.globals import save_prompt, AGENT_SYSTEM_PROMPT as _current
    from app.services.admin_utils import log_admin_action
    data = request.get_json(silent=True) or {}
    new_prompt = (data.get('prompt', '') or '').strip()
    if not new_prompt:
        return err("Prompt cannot be empty", "VALIDATION_ERROR", 400)
    if new_prompt == _current.strip():
        return ok({"message": "No changes"}, "ok")
    save_prompt(new_prompt)
    log_admin_action(session.get('user_id', ''), session.get('username', ''),
                    'PROMPT_EDIT', 'system', None, column_name='agent_system_prompt',
                    old_value=_current[:80] + '...', new_value=new_prompt[:80] + '...')
    return ok({"message": "System prompt updated"}, "ok")


# ── Search Cache Config ──

@admin_bp.route('/admin/search_cache_config', methods=['GET'])
@admin_required
def get_search_cache_config():
    """Return search cache config and stats."""
    from app.services.agent import get_cache_stats
    return ok({"config": get_cache_stats()}, "ok")


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
        return ok({"message": "搜索缓存已清除"}, "ok")

    # set_ttl
    ttl_hours = data.get('ttl_hours')
    if ttl_hours is None:
        return err("缺少 ttl_hours 参数", "VALIDATION_ERROR", 400)
    try:
        ttl_hours = float(ttl_hours)
    except (ValueError, TypeError):
        return err("ttl_hours 必须是数字", "VALIDATION_ERROR", 400)
    if ttl_hours < 0:
        return err("TTL 不能为负数（设为0表示禁用缓存）", "VALIDATION_ERROR", 400)

    old_ttl_hours = get_cache_ttl() / 3600
    _set_cache_ttl(int(ttl_hours * 3600))
    log_admin_action(session.get('user_id', ''), session.get('username', ''),
                    'CACHE_TTL_CHANGE', 'system', None,
                    column_name='search_cache_ttl',
                    old_value=f'{old_ttl_hours}h', new_value=f'{ttl_hours}h')
    return ok({"message": f"搜索缓存 TTL 已设为 {ttl_hours} 小时"}, "ok")


# ── Unified Runtime Config ──

@admin_bp.route('/admin/runtime_config', methods=['GET'])
@admin_required
def get_runtime_config():
    """Return all runtime-adjustable config values."""
    from app.services.runtime_config import get_all
    return ok({"config": get_all()}, "ok")


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
        return ok({"message": "Config reset to defaults", "config": cfg}, "ok")

    if action == 'save_factory':
        if has_factory_presets():
            return err("Factory presets already saved — cannot overwrite", "RESOURCE_BUSY", 409)
        factory = save_factory_presets()
        log_admin_action(session.get('user_id', ''), session.get('username', ''),
                        'CONFIG_FACTORY_SAVE', 'system', None,
                        column_name='runtime_config', new_value=f'factory_saved:{len(factory)}keys')
        return ok({"message": f"Factory presets saved ({len(factory)} keys, read-only)", "factory": factory}, "ok")

    if action == 'restore_factory':
        if not has_factory_presets():
            return err("No factory presets exist — save factory first", "VALIDATION_ERROR", 400)
        cfg = restore_factory_presets()
        log_admin_action(session.get('user_id', ''), session.get('username', ''),
                        'CONFIG_RESTORE_FACTORY', 'system', None,
                        column_name='runtime_config', new_value='restored_to_factory')
        return ok({"message": "Restored to factory presets", "config": cfg}, "ok")

    if not data:
        return err("No update parameters provided", "VALIDATION_ERROR", 400)

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
            return err(f"Type mismatch for {k}: expected {expected_type.__name__}", "VALIDATION_ERROR", 400)

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

    if 'active_vl_provider' in sanitized or 'active_vl_model' in sanitized:
        from app.services.vl_model import vl_model
        vl_model.reload()
        logger.info("VL model reloaded due to config change")

    return ok({"message": f"Updated {len(sanitized)} config keys", "config": cfg}, "ok")


@admin_bp.route('/admin/embedding_cache', methods=['GET'])
@admin_required
def get_embedding_cache():
    """Return embedding cache stats for admin monitoring."""
    try:
        from app.services.rag_engine import embedding_cache_stats
        return ok(embedding_cache_stats())
    except Exception as e:
        return err(str(e)[:200], "SERVER_ERROR", 500)

@admin_bp.route('/admin/embedding_cache/clear', methods=['POST'])
@admin_required
def clear_embedding_cache_route():
    """Clear the embedding cache (useful after model update)."""
    try:
        from app.services.rag_engine import clear_embedding_cache
        clear_embedding_cache()
        return ok(message="ok")
    except Exception as e:
        return err(str(e)[:200], "SERVER_ERROR", 500)

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
        return ok({
            "pending": result.stdout,
            "history": hist_result.stdout,
            "error": result.stderr if result.stderr else None,
        })
    except Exception as e:
        return err(str(e)[:200], "SERVER_ERROR", 500)

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
        return ok({
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.stderr else None,
        })
    except Exception as e:
        return err(str(e)[:200], "SERVER_ERROR", 500)

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
        return ok({
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.stderr else None,
        })
    except Exception as e:
        return err(str(e)[:200], "SERVER_ERROR", 500)

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
        return ok({
            "success": True,
            "output": result.stdout,
        })
    except Exception as e:
        return err(str(e)[:200], "SERVER_ERROR", 500)

@admin_bp.route('/admin/runtime_config_schema', methods=['GET'])
@admin_required
def get_runtime_config_schema():
    """Return metadata about each config key (for building smart UI)."""
    # Build dynamic model list from all configured providers
    from app.services.llm_provider import PROVIDER_CONFIG
    all_models = ['auto']
    model_labels = {'auto': '自动(服务商默认)'}
    # FIX-016: dynamic provider options from PROVIDER_CONFIG (openrouter/nvidia)
    provider_options = ['auto']
    provider_labels = {'auto': '自动检测'}
    for pid, cfg in PROVIDER_CONFIG.items():
        provider_options.append(pid)
        provider_labels[pid] = cfg['name']
        for m in cfg.get('models', []):
            label = f"{m} ({cfg['name']})"
            all_models.append(m)
            model_labels[m] = label

    schema = {
        # ── LLM ──
        "active_llm_provider":        {"label": "LLM 服务商", "unit": "", "type": "select", "group": "LLM/AI Model", "is_llm": True, "is_not_factory": True,
                                         "options": provider_options,
                                         "option_labels": provider_labels},
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
        "active_vl_provider":         {"label": "VL 服务商", "unit": "", "type": "select", "group": "VL Model", "is_not_factory": True,
                                         "options": ["auto", "nvidia", "mimo", "dashscope"],
                                         "option_labels": {"auto": "自动检测", "nvidia": "NVIDIA", "mimo": "Mimo", "dashscope": "阿里云DashScope"}},
        "active_vl_model":            {"label": "VL 模型", "unit": "", "type": "select", "group": "VL Model", "is_not_factory": True,
                                         "options": ["auto", "nvidia/nvlm-d-72b", "nvidia/llama-3.2-nv-vision-34b",
                                                     "mimo-v2.5-pro", "mimo-v2.5",
                                                     "qwen3-vl-plus-2025-12-19", "qwen-vl-max", "qwen-vl-plus"],
                                         "option_labels": {"auto": "自动(服务商默认)", "nvidia/nvlm-d-72b": "NVLM-D-72B (NVIDIA)", "nvidia/llama-3.2-nv-vision-34b": "Llama-3.2-NV (NVIDIA)", "mimo-v2.5-pro": "mimo-v2.5-pro (Mimo)", "mimo-v2.5": "mimo-v2.5 (Mimo)", "qwen3-vl-plus-2025-12-19": "qwen3-vl-plus (DashScope)", "qwen-vl-max": "qwen-vl-max (DashScope)", "qwen-vl-plus": "qwen-vl-plus (DashScope)"}},
        "vl_max_image_size":          {"label": "VL 最大图片尺寸", "unit": "px", "type": "int", "group": "VL Model", "min": 128, "max": 4096},
        "vl_jpeg_quality":            {"label": "JPEG 质量", "unit": "%", "type": "int", "group": "VL Model", "min": 10, "max": 100},
        "vl_max_tokens":              {"label": "VL 最大Token", "unit": "tokens", "type": "int", "group": "VL Model", "min": 50, "max": 4096},
        "vl_temperature":             {"label": "VL 温度", "unit": "", "type": "float", "group": "VL Model", "min": 0, "max": 2, "step": 0.05},
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
    return ok({
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

    return ok({
        "status": "ok",
        "providers": providers,
        "active_provider": active_provider,
        "active_model": active_model,
        "model_options": model_options,
        "session_provider": session_provider,
        "session_model": session_model,
    })


@admin_bp.route('/admin/vl_status', methods=['GET'])
@admin_required
def admin_vl_status():
    from app.services.vl_model import vl_model, VL_PROVIDER_CONFIG
    cfg = vl_model.provider_id
    provider_name = VL_PROVIDER_CONFIG.get(cfg, {}).get('name', cfg) if cfg != 'auto' else 'auto'
    return ok({
        "status": "ok",
        "available": vl_model.is_available(),
        "has_api_key": bool(vl_model.api_key),
        "model": vl_model.model_name,
        "provider": provider_name,
        "provider_id": vl_model.provider_id,
        "config": {
            "max_image_size": vl_model.max_image_size,
            "max_tokens": 800,
            "temperature": 0.7,
        }
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
        return err("收件人、主题和正文不能为空", "VALIDATION_ERROR", 400)
    try:
        from app.utils.mailer import send_email, is_configured
        if not is_configured():
            return err("SMTP未配置，请设置SMTP_HOST等环境变量", "SERVICE_UNAVAILABLE", 503)
        success = send_email(to_addr, subject, body, async_mode=True)
        if success:
            return ok({"message": f"邮件已发送至 {to_addr}"}, "ok")
        else:
            return err("邮件发送失败", "SERVER_ERROR", 500)
    except Exception as e:
        return err(str(e), "SERVER_ERROR", 500)

# ── Mail: get all user emails for autocomplete ──
@admin_bp.route('/admin/user_emails', methods=['GET'])
@admin_required
def admin_user_emails():
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT username, user_id, role, is_auditor FROM users WHERE is_active = TRUE ORDER BY username")
            users = cur.fetchall()
    return ok({"users": [
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
    return ok({"results": results}, "ok")

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
                tbl = sql.Identifier(table)
                cur.execute(sql.SQL("UPDATE {} SET skill_summary=NULL, skill_generated_at=NULL, skill_summary_hash=NULL").format(tbl))
                results[f'{table}_skills_cleared'] = cur.rowcount
            # 2. Clear file content
            for table in ('knowledge_lab_files', 'company_knowledge_base', 'project_files', 'user_files'):
                tbl = sql.Identifier(table)
                cur.execute(sql.SQL("UPDATE {} SET content=NULL").format(tbl))
                results[f'{table}_content_cleared'] = cur.rowcount
            # 3. Delete file records
            for table in ('knowledge_lab_files', 'company_knowledge_base', 'user_files'):
                tbl = sql.Identifier(table)
                cur.execute(sql.SQL("DELETE FROM {}").format(tbl))
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
                    except Exception: pass
                for d in dirs:
                    try: _shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                    except Exception: pass
        else:
            for f in os.listdir(full):
                fp = os.path.join(full, f)
                if os.path.isfile(fp):
                    try: os.remove(fp); count += 1
                    except Exception: pass
        results[f'disk_{rel_dir}'] = count

    # 9. Delete skill audit cache file
    try:
        cache_file = os.path.join(base, 'data', 'skill_audit_cache.json')
        if os.path.exists(cache_file):
            os.remove(cache_file)
    except Exception: pass

    return ok({"results": results}, "ok")

# ── Safe file deletion helper ──
def _safe_delete_file(filepath, label=''):
    """Delete a file and log failure. Returns True if deleted or didn't exist."""
    if not filepath:
        return True
    filepath = resolve_path(filepath)
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
                    tbl = sql.Identifier(table)
                    col_id = sql.Identifier(col)
                    cur.execute(sql.SQL("SELECT {} FROM {} WHERE {} IS NOT NULL AND {} != ''").format(col_id, tbl, col_id, col_id))
                    for (path,) in cur.fetchall():
                        total_checked += 1
                        if path and not os.path.exists(resolve_path(path)):
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
                found = False
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        for table, col in tables_to_check:
                            tbl = sql.Identifier(table)
                            col_id = sql.Identifier(col)
                            cur.execute(sql.SQL("SELECT 1 FROM {} WHERE {} = %s LIMIT 1").format(tbl, col_id), (full_path,))
                            if cur.fetchone():
                                found = True
                                break
                        if not found:
                            for table, col in tables_to_check:
                                tbl = sql.Identifier(table)
                                col_id = sql.Identifier(col)
                                cur.execute(sql.SQL("SELECT {} FROM {} WHERE {} IS NOT NULL AND {} != ''").format(col_id, tbl, col_id, col_id))
                                for (db_path,) in cur.fetchall():
                                    if db_path and os.path.normpath(resolve_path(db_path)) == os.path.normpath(full_path):
                                        found = True
                                        break
                                if found:
                                    break
                if not found:
                    leaks.append({'path': full_path, 'size': os.path.getsize(full_path)})

    return ok({
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
    return ok({
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
        return ok({"marked": 0})

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

        return ok({"marked": marked})
    except Exception as e:
        return err(str(e), "SERVER_ERROR", 500)
