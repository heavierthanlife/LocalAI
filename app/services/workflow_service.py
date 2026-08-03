"""Workflow service — per-milestone step tracking."""
import logging
from datetime import datetime
from typing import Dict, List, Optional

from app.database import get_db_connection

logger = logging.getLogger(__name__)

DEFAULT_STEPS = {
    'TENDER_NOTICE': [
        {'name': '编写招标公告', 'order': 1},
        {'name': '审核招标公告', 'order': 2},
        {'name': '发布招标公告', 'order': 3},
    ],
    'DOC_SALE_END': [
        {'name': '准备招标文件', 'order': 1},
        {'name': '发布招标文件', 'order': 2},
        {'name': '确认发售期结束', 'order': 3},
    ],
    'CLARIFY_DEADLINE': [
        {'name': '收集投标人问题', 'order': 1},
        {'name': '编制澄清文件', 'order': 2},
        {'name': '发出澄清通知', 'order': 3},
    ],
    'BID_SUBMIT_DEADLINE': [
        {'name': '准备投标接收场所', 'order': 1},
        {'name': '接收投标文件', 'order': 2},
        {'name': '确认投标截止', 'order': 3},
    ],
    'BID_OPENING': [
        {'name': '准备开标场地', 'order': 1},
        {'name': '开标并唱标', 'order': 2},
        {'name': '记录开标情况', 'order': 3},
    ],
    'EVALUATION': [
        {'name': '组建评标委员会', 'order': 1},
        {'name': '评标', 'order': 2},
        {'name': '编写评标报告', 'order': 3},
    ],
    'CANDIDATE_ANNOUNCE': [
        {'name': '编写中标候选人公示', 'order': 1},
        {'name': '发布公示', 'order': 2},
    ],
    'CANDIDATE_DISPLAY_END': [
        {'name': '收集公示期异议', 'order': 1},
        {'name': '处理异议', 'order': 2},
        {'name': '确认公示期结束', 'order': 3},
    ],
    'AWARD_NOTICE': [
        {'name': '编写中标通知书', 'order': 1},
        {'name': '审核中标通知书', 'order': 2},
        {'name': '发出中标通知书', 'order': 3},
    ],
    'RESULT_ANNOUNCE': [
        {'name': '编写中标结果公告', 'order': 1},
        {'name': '发布中标结果公告', 'order': 2},
    ],
    'CONTRACT_SIGN': [
        {'name': '准备合同文件', 'order': 1},
        {'name': '合同谈判', 'order': 2},
        {'name': '签订书面合同', 'order': 3},
    ],
    'INVITATION_SENT': [
        {'name': '确定受邀投标人名单', 'order': 1},
        {'name': '编写投标邀请书', 'order': 2},
        {'name': '发出投标邀请书', 'order': 3},
    ],
    'NEGOTIATION_SESSION': [
        {'name': '准备谈判文件', 'order': 1},
        {'name': '进行谈判', 'order': 2},
        {'name': '记录谈判情况', 'order': 3},
    ],
    'FINAL_OFFER_DEADLINE': [
        {'name': '发出最后报价通知', 'order': 1},
        {'name': '接收最后报价', 'order': 2},
    ],
    'WINNER_DECISION': [
        {'name': '编写评审报告', 'order': 1},
        {'name': '确定中标/成交供应商', 'order': 2},
    ],
}


def create_step(timeline_id: int, milestone_code: str, step_name: str,
                step_order: int = 0, assigned_to: Optional[str] = None) -> dict:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM project_timeline_milestones WHERE timeline_id = %s AND milestone_code = %s",
                (timeline_id, milestone_code))
            ms_row = cur.fetchone()
            if not ms_row:
                raise ValueError(f"Milestone not found: {milestone_code}")
            milestone_id = ms_row[0]

            if step_order == 0:
                cur.execute(
                    "SELECT COALESCE(MAX(step_order), 0) + 1 FROM project_workflow_steps WHERE milestone_id = %s",
                    (milestone_id,))
                step_order = cur.fetchone()[0]

            cur.execute("""
                INSERT INTO project_workflow_steps (milestone_id, step_name, step_order, assigned_to)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (milestone_id, step_name, step_order, assigned_to))
            step_id = cur.fetchone()[0]
            conn.commit()

            cur.execute(
                "SELECT id, milestone_id, step_name, step_order, assigned_to, completed, completed_at, notes, created_at FROM project_workflow_steps WHERE id = %s",
                (step_id,))
            row = cur.fetchone()
            return _row_to_dict(row)


def list_steps(timeline_id: int, milestone_code: Optional[str] = None) -> List[dict]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if milestone_code:
                cur.execute("""
                    SELECT ws.id, ws.milestone_id, ws.step_name, ws.step_order,
                           ws.assigned_to, ws.completed, ws.completed_at, ws.notes, ws.created_at
                    FROM project_workflow_steps ws
                    JOIN project_timeline_milestones ptm ON ws.milestone_id = ptm.id
                    WHERE ptm.timeline_id = %s AND ptm.milestone_code = %s
                    ORDER BY ws.step_order
                """, (timeline_id, milestone_code))
            else:
                cur.execute("""
                    SELECT ws.id, ws.milestone_id, ws.step_name, ws.step_order,
                           ws.assigned_to, ws.completed, ws.completed_at, ws.notes, ws.created_at,
                           ptm.milestone_code
                    FROM project_workflow_steps ws
                    JOIN project_timeline_milestones ptm ON ws.milestone_id = ptm.id
                    WHERE ptm.timeline_id = %s
                    ORDER BY ptm.sort_order, ws.step_order
                """, (timeline_id,))

            results = []
            for row in cur.fetchall():
                entry = _row_to_dict(row[:9])
                if len(row) > 9:
                    entry['milestone_code'] = row[9]
                results.append(entry)
            return results


def update_step(step_id: int, completed: Optional[bool] = None,
                notes: Optional[str] = None, assigned_to: Optional[str] = None) -> dict:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            updates = []
            params = []
            if completed is not None:
                updates.append("completed = %s")
                params.append(completed)
                if completed:
                    updates.append("completed_at = NOW()")
                else:
                    updates.append("completed_at = NULL")
            if notes is not None:
                updates.append("notes = %s")
                params.append(notes)
            if assigned_to is not None:
                updates.append("assigned_to = %s")
                params.append(assigned_to)

            if not updates:
                cur.execute(
                    "SELECT id, milestone_id, step_name, step_order, assigned_to, completed, completed_at, notes, created_at FROM project_workflow_steps WHERE id = %s",
                    (step_id,))
                return _row_to_dict(cur.fetchone())

            updates.append("updated_at = NOW()")
            params.append(step_id)
            cur.execute(
                f"UPDATE project_workflow_steps SET {', '.join(updates)} WHERE id = %s",
                params)
            conn.commit()

            cur.execute(
                "SELECT id, milestone_id, step_name, step_order, assigned_to, completed, completed_at, notes, created_at FROM project_workflow_steps WHERE id = %s",
                (step_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Step not found: {step_id}")
            result = _row_to_dict(row)

            if completed:
                _check_milestone_auto_complete(conn, row[1])

            return result


def batch_complete_steps(step_ids: List[int]) -> List[dict]:
    results = []
    for sid in step_ids:
        results.append(update_step(sid, completed=True))
    return results


def auto_generate_steps(timeline_id: int) -> List[dict]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, milestone_code FROM project_timeline_milestones WHERE timeline_id = %s ORDER BY sort_order",
                (timeline_id,))
            milestones = cur.fetchall()

            all_results = []
            for ms_id, ms_code in milestones:
                existing = list_steps(timeline_id, ms_code)
                if existing:
                    continue

                template = DEFAULT_STEPS.get(ms_code, [])
                for step in template:
                    result = create_step(timeline_id, ms_code, step['name'], step['order'])
                    all_results.append(result)

    return all_results


def delete_step(step_id: int) -> bool:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM project_workflow_steps WHERE id = %s", (step_id,))
            conn.commit()
            return cur.rowcount > 0


def get_progress(timeline_id: int) -> dict:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*), COUNT(*) FILTER (WHERE completed = TRUE)
                FROM project_workflow_steps ws
                JOIN project_timeline_milestones ptm ON ws.milestone_id = ptm.id
                WHERE ptm.timeline_id = %s
            """, (timeline_id,))
            total, done = cur.fetchone()
            return {
                'total_steps': total,
                'completed_steps': done,
                'percentage': round(done / max(total, 1) * 100, 1),
            }


def _check_milestone_auto_complete(conn, milestone_id: int):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE completed = TRUE) FROM project_workflow_steps WHERE milestone_id = %s",
            (milestone_id,))
        total, done = cur.fetchone()
        if total > 0 and done == total:
            cur.execute(
                "UPDATE project_timeline_milestones SET status = 'completed', actual_date = CURRENT_DATE, updated_at = NOW() WHERE id = %s",
                (milestone_id,))
            conn.commit()


def _row_to_dict(row) -> dict:
    if not row:
        return {}
    return {
        'id': row[0],
        'milestone_id': row[1],
        'step_name': row[2],
        'step_order': row[3],
        'assigned_to': row[4],
        'completed': row[5],
        'completed_at': row[6].isoformat() if row[6] else None,
        'notes': row[7],
        'created_at': row[8].isoformat() if row[8] else None,
    }
