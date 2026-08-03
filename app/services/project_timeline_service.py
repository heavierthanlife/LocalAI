"""Project timeline service — CRUD, milestone tracking, diff computation."""
import json
import logging
from datetime import date, datetime
from typing import Dict, List, Optional

from app.database import get_db_connection
from app.services.legal_schedule_service import get_schedule, compute_planned_dates

logger = logging.getLogger(__name__)

VALID_REASON_CATEGORIES = {'legal', 'administrative', 'technical', 'force_majeure', 'other'}


def create_timeline(
    project_id: int,
    category_code: str,
    method_code: str,
    planned_start_date: date,
    name: str = '主招标流程',
    planned_end_date: Optional[date] = None,
    created_by: Optional[str] = None,
) -> dict:
    schedule = get_schedule(category_code, method_code)
    if not schedule:
        raise ValueError(f"Invalid category/method: {category_code}/{method_code}")

    milestones_data = compute_planned_dates(schedule['milestones'], planned_start_date)
    status = 'active'

    with get_db_connection() as conn:
        if planned_end_date is None:
            last_m = milestones_data[-1]
            planned_end_date = last_m.get('_planned_date')

        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM project_timelines WHERE project_id = %s AND status = 'active'",
                (project_id,))
            existing = cur.fetchone()
            if existing:
                cur.execute(
                    "UPDATE project_timelines SET status = 'archived', updated_at = NOW() WHERE project_id = %s AND status = 'active'",
                    (project_id,))

            cur.execute("""
                INSERT INTO project_timelines
                    (project_id, name, category_code, method_code, planned_start_date, planned_end_date, status, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (project_id, name, category_code, method_code, planned_start_date, planned_end_date, status, created_by))
            timeline_id = cur.fetchone()[0]

            sort_order = 0
            for m in milestones_data:
                sort_order += 1
                cur.execute("""
                    INSERT INTO project_timeline_milestones
                        (timeline_id, milestone_code, milestone_name, planned_date, status, sort_order)
                    VALUES (%s, %s, %s, %s, 'pending', %s)
                """, (
                    timeline_id, m['code'], m['name'],
                    m.get('_planned_date'), sort_order,
                ))

            conn.commit()

    return get_timeline(project_id)


def get_timeline(project_id: int, timeline_id: Optional[int] = None) -> Optional[dict]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if timeline_id:
                cur.execute("""
                    SELECT id, project_id, name, category_code, method_code, planned_start_date,
                           planned_end_date, actual_start_date, actual_end_date, status,
                           created_by, created_at, updated_at
                    FROM project_timelines
                    WHERE id = %s AND status != 'deleted'
                """, (timeline_id,))
            else:
                cur.execute("""
                    SELECT id, project_id, name, category_code, method_code, planned_start_date,
                           planned_end_date, actual_start_date, actual_end_date, status,
                           created_by, created_at, updated_at
                    FROM project_timelines
                    WHERE project_id = %s AND status != 'deleted'
                    ORDER BY created_at DESC LIMIT 1
                """, (project_id,))
            row = cur.fetchone()
            if not row:
                return None

            timeline = {
                'id': row[0], 'project_id': row[1], 'name': row[2] or '',
                'category_code': row[3], 'method_code': row[4],
                'planned_start_date': row[5].isoformat() if row[5] else None,
                'planned_end_date': row[6].isoformat() if row[6] else None,
                'actual_start_date': row[7].isoformat() if row[7] else None,
                'actual_end_date': row[8].isoformat() if row[8] else None,
                'status': row[9], 'created_by': row[10],
                'created_at': row[11].isoformat() if row[11] else None,
                'updated_at': row[12].isoformat() if row[12] else None,
                'milestones': [],
                'diff_summary': {},
            }

            cur.execute("""
                SELECT id, milestone_code, milestone_name, planned_date, actual_date,
                       diff_days, diff_reason, reason_category, status, sort_order
                FROM project_timeline_milestones
                WHERE timeline_id = %s
                ORDER BY sort_order
            """, (timeline['id'],))
            for m in cur.fetchall():
                timeline['milestones'].append({
                    'id': m[0], 'code': m[1], 'name': m[2],
                    'planned_date': m[3].isoformat() if m[3] else None,
                    'actual_date': m[4].isoformat() if m[4] else None,
                    'diff_days': m[5], 'diff_reason': m[6],
                    'reason_category': m[7], 'status': m[8],
                    'sort_order': m[9],
                })

            timeline['diff_summary'] = _compute_diff_summary(timeline['milestones'])

    return timeline


def list_timelines(project_id: int) -> List[dict]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, category_code, method_code, status,
                       planned_start_date, planned_end_date,
                       actual_start_date, actual_end_date,
                       created_by, created_at, updated_at
                FROM project_timelines
                WHERE project_id = %s AND status != 'deleted'
                ORDER BY created_at DESC
            """, (project_id,))
            results = []
            for row in cur.fetchall():
                results.append({
                    'id': row[0], 'name': row[1] or '',
                    'category_code': row[2], 'method_code': row[3],
                    'status': row[4],
                    'planned_start_date': row[5].isoformat() if row[5] else None,
                    'planned_end_date': row[6].isoformat() if row[6] else None,
                    'actual_start_date': row[7].isoformat() if row[7] else None,
                    'actual_end_date': row[8].isoformat() if row[8] else None,
                    'created_by': row[9],
                    'created_at': row[10].isoformat() if row[10] else None,
                    'updated_at': row[11].isoformat() if row[11] else None,
                })
            return results


def update_milestone(
    timeline_id: int,
    milestone_code: str,
    actual_date: Optional[date] = None,
    diff_reason: Optional[str] = None,
    reason_category: Optional[str] = None,
    status: Optional[str] = None,
    created_by: Optional[str] = None,
) -> dict:
    if reason_category and reason_category not in VALID_REASON_CATEGORIES:
        raise ValueError(f"Invalid reason_category: {reason_category}. Must be one of {VALID_REASON_CATEGORIES}")

    milestone = None
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, planned_date, actual_date FROM project_timeline_milestones WHERE timeline_id = %s AND milestone_code = %s",
                (timeline_id, milestone_code))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Milestone not found: {milestone_code}")
            milestone_id = row[0]
            planned_date = row[1]
            old_actual_date = row[2]

            diff_days = None
            diff_type = None
            if actual_date and planned_date:
                diff_days = (actual_date - planned_date).days
                if diff_days > 0:
                    diff_type = 'delay'
                elif diff_days < 0:
                    diff_type = 'advance'
                else:
                    diff_type = 'on_time'

            updates = []
            params = []
            if actual_date is not None:
                updates.append("actual_date = %s")
                params.append(actual_date)
            if diff_days is not None:
                updates.append("diff_days = %s")
                params.append(diff_days)
            if diff_reason is not None:
                updates.append("diff_reason = %s")
                params.append(diff_reason)
            if reason_category is not None:
                updates.append("reason_category = %s")
                params.append(reason_category)
            if status is not None:
                updates.append("status = %s")
                params.append(status)
            updates.append("updated_at = NOW()")

            if updates:
                params.append(milestone_id)
                cur.execute(
                    f"UPDATE project_timeline_milestones SET {', '.join(updates)} WHERE id = %s",
                    params)

            if diff_days is not None and diff_type:
                cur.execute("""
                    INSERT INTO timeline_diff_log
                        (milestone_id, milestone_code, planned_date, actual_date,
                         diff_days, diff_type, reason_category, reason_detail, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    milestone_id, milestone_code, planned_date, actual_date,
                    diff_days, diff_type, reason_category, diff_reason, created_by,
                ))

            conn.commit()

            cur.execute(
                "SELECT planned_date, actual_date, diff_days, diff_reason, reason_category, status FROM project_timeline_milestones WHERE id = %s",
                (milestone_id,))
            row = cur.fetchone()
            milestone = {
                'id': milestone_id, 'code': milestone_code,
                'planned_date': row[0].isoformat() if row[0] else None,
                'actual_date': row[1].isoformat() if row[1] else None,
                'diff_days': row[2], 'diff_reason': row[3],
                'reason_category': row[4], 'status': row[5],
            }

    return milestone


def get_diff_report(timeline_id: int) -> dict:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, milestone_code, milestone_name, planned_date, actual_date,
                       diff_days, diff_reason, reason_category, status
                FROM project_timeline_milestones
                WHERE timeline_id = %s
                ORDER BY sort_order
            """, (timeline_id,))
            milestones = []
            for m in cur.fetchall():
                milestones.append({
                    'id': m[0], 'code': m[1], 'name': m[2],
                    'planned_date': m[3].isoformat() if m[3] else None,
                    'actual_date': m[4].isoformat() if m[4] else None,
                    'diff_days': m[5], 'diff_reason': m[6],
                    'reason_category': m[7], 'status': m[8],
                })

            cur.execute("""
                SELECT milestone_code, planned_date, actual_date, diff_days, diff_type,
                       reason_category, reason_detail, created_at
                FROM timeline_diff_log
                WHERE milestone_id IN (SELECT id FROM project_timeline_milestones WHERE timeline_id = %s)
                ORDER BY created_at DESC
            """, (timeline_id,))
            diff_log = []
            for d in cur.fetchall():
                diff_log.append({
                    'milestone_code': d[0],
                    'planned_date': d[1].isoformat() if d[1] else None,
                    'actual_date': d[2].isoformat() if d[2] else None,
                    'diff_days': d[3], 'diff_type': d[4],
                    'reason_category': d[5], 'reason_detail': d[6],
                    'created_at': d[7].isoformat() if d[7] else None,
                })

    return {
        'milestones': milestones,
        'diff_log': diff_log,
        'summary': _compute_diff_summary(milestones),
    }


def batch_update_milestones(timeline_id: int, updates: List[dict], created_by: Optional[str] = None) -> List[dict]:
    results = []
    for u in updates:
        code = u.get('code')
        try:
            r = update_milestone(
                timeline_id=timeline_id,
                milestone_code=code,
                actual_date=u.get('actual_date'),
                diff_reason=u.get('diff_reason'),
                reason_category=u.get('reason_category'),
                status=u.get('status'),
                created_by=created_by,
            )
            results.append({'code': code, 'success': True, 'milestone': r})
        except Exception as e:
            logger.warning(f"Batch milestone update failed for {code}: {e}")
            results.append({'code': code, 'success': False, 'error': str(e)})
    return results


def update_timeline_dates(timeline_id: int, planned_start_date: Optional[date] = None,
                          planned_end_date: Optional[date] = None, actual_start_date: Optional[date] = None,
                          actual_end_date: Optional[date] = None, status: Optional[str] = None) -> dict:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            updates = []
            params = []
            if planned_start_date is not None:
                updates.append("planned_start_date = %s")
                params.append(planned_start_date)
            if planned_end_date is not None:
                updates.append("planned_end_date = %s")
                params.append(planned_end_date)
            if actual_start_date is not None:
                updates.append("actual_start_date = %s")
                params.append(actual_start_date)
            if actual_end_date is not None:
                updates.append("actual_end_date = %s")
                params.append(actual_end_date)
            if status is not None:
                updates.append("status = %s")
                params.append(status)
            if not updates:
                return get_timeline_by_id(timeline_id)

            updates.append("updated_at = NOW()")
            params.append(timeline_id)
            cur.execute(
                f"UPDATE project_timelines SET {', '.join(updates)} WHERE id = %s",
                params)
            conn.commit()

            if planned_start_date is not None:
                cur.execute(
                    "SELECT category_code, method_code FROM project_timelines WHERE id = %s",
                    (timeline_id,))
                row = cur.fetchone()
                if row:
                    schedule = get_schedule(row[0], row[1])
                    if schedule:
                        new_milestones = compute_planned_dates(schedule['milestones'], planned_start_date)
                        for nm in new_milestones:
                            cur.execute("""
                                UPDATE project_timeline_milestones
                                SET planned_date = %s, updated_at = NOW()
                                WHERE timeline_id = %s AND milestone_code = %s
                            """, (nm.get('_planned_date'), timeline_id, nm['code']))
                        conn.commit()

    return get_timeline_by_id(timeline_id)


def get_timeline_by_id(timeline_id: int) -> Optional[dict]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT project_id FROM project_timelines WHERE id = %s", (timeline_id,))
            row = cur.fetchone()
            if not row:
                return None
            return get_timeline(row[0])


def delete_timeline(timeline_id: int) -> bool:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE project_timelines SET status = 'deleted', updated_at = NOW() WHERE id = %s",
                (timeline_id,))
            conn.commit()
            return cur.rowcount > 0


def _compute_diff_summary(milestones: List[dict]) -> dict:
    total = len(milestones)
    completed = 0
    delayed = 0
    on_time = 0
    advanced = 0
    pending = 0
    total_delay_days = 0

    for m in milestones:
        if m.get('status') == 'completed':
            completed += 1
        elif m.get('status') == 'pending':
            pending += 1

        diff = m.get('diff_days')
        if diff is not None:
            if diff > 0:
                delayed += 1
                total_delay_days += diff
            elif diff == 0:
                on_time += 1
            else:
                advanced += 1

    by_category = {}
    for m in milestones:
        cat = m.get('reason_category') or 'unknown'
        by_category[cat] = by_category.get(cat, 0) + 1

    return {
        'total_milestones': total,
        'completed': completed,
        'pending': pending,
        'delayed': delayed,
        'on_time': on_time,
        'advanced': advanced,
        'total_delay_days': total_delay_days,
        'by_reason_category': by_category,
    }
