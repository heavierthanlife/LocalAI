"""Timeline suggestion engine — rule-based warnings + AI risk assessment."""
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

from app.database import get_db_connection
from app.services.legal_schedule_service import get_deadline_warning

logger = logging.getLogger(__name__)

AI_REFRESH_COOLDOWN_HOURS = 1
MAX_AI_SUGGESTIONS = 5


def evaluate_timeline(timeline_id: int, project_id: int) -> List[dict]:
    timeline = _get_timeline_with_milestones(timeline_id)
    if not timeline:
        return []

    _cleanup_stale_suggestions(timeline_id)

    today = date.today()
    suggestions = []
    milestones = timeline.get('milestones', [])

    for m in milestones:
        planned = m.get('planned_date')
        if not planned:
            continue
        try:
            planned_date = date.fromisoformat(planned)
        except (ValueError, TypeError):
            continue
        actual_date = m.get('actual_date')
        try:
            actual = date.fromisoformat(actual_date) if actual_date else None
        except (ValueError, TypeError):
            actual = None
        diff_days = m.get('diff_days')
        status = m.get('status', 'pending')
        code = m.get('code', '')
        name = m.get('name', '')

        if status == 'completed':
            if diff_days is not None and diff_days > 0:
                suggestions.append(_make_suggestion(
                    timeline_id, code, 'info', 'delay_recorded',
                    f'"{name}"延迟了 {diff_days} 天完成，原因为: {m.get("diff_reason", "未记录")}"',
                    f"建议在项目总结中记录此延迟情况及原因，归档备查"
                ))
            continue

        days_until = (planned_date - today).days if actual is None else None

        if days_until is not None:
            if days_until < 0:
                overdue_days = abs(days_until)
                severity = 'critical' if m.get('mandatory') else 'high'
                suggestions.append(_make_suggestion(
                    timeline_id, code, severity, 'overdue',
                    f'"{name}"已超期 {overdue_days} 天"',
                    f"请立即采取措施推进该节点，并记录超期原因。{'法律依据: ' + m.get('law_ref', '') if m.get('law_ref') else ''}"
                ))
            elif days_until <= 3:
                suggestions.append(_make_suggestion(
                    timeline_id, code, 'high', 'approaching',
                    f'"{name}"距截止仅剩 {days_until} 天"',
                    f"请确保按时完成。最后可操作日期: {planned}"
                ))
            elif days_until <= 7:
                suggestions.append(_make_suggestion(
                    timeline_id, code, 'medium', 'upcoming',
                    f'"{name}"距截止还有 {days_until} 天"',
                    f"请提前准备，计划日期: {planned}"
                ))

    consecutive_delays = 0
    for m in milestones:
        if m.get('diff_days') and m['diff_days'] > 0:
            consecutive_delays += 1
            if consecutive_delays >= 2:
                suggestions.append(_make_suggestion(
                    timeline_id, m['code'], 'high', 'consecutive_delay',
                    f"已连续 {consecutive_delays} 个里程碑出现延期",
                    "建议重新评估项目计划，考虑调整后续节点时间"
                ))
                break
        else:
            consecutive_delays = 0

    total_delay = sum(m.get('diff_days', 0) or 0 for m in milestones)
    if total_delay > 15:
        suggestions.append(_make_suggestion(
            timeline_id, milestones[0]['code'] if milestones else '', 'critical',
            'total_delay_high',
            f"项目累计延期 {total_delay} 天",
            "总延迟严重，建议召开项目协调会议，评估是否需要启动应急预案"
        ))
    elif total_delay > 7:
        suggestions.append(_make_suggestion(
            timeline_id, milestones[0]['code'] if milestones else '', 'medium',
            'total_delay_moderate',
            f"项目累计延期 {total_delay} 天",
            "建议关注后续节点的进度安排"
        ))

    _persist_suggestions(suggestions)
    return suggestions


def generate_ai_suggestions(timeline_id: int, project_id: int) -> List[dict]:
    last_ai = _last_ai_generation(timeline_id)
    if last_ai:
        age = datetime.now(timezone.utc) - last_ai
        if age < timedelta(hours=AI_REFRESH_COOLDOWN_HOURS):
            logger.info(f"AI suggestions recently generated ({age.total_seconds()/60:.0f}m ago), returning cached")
            return _get_existing_ai_suggestions(timeline_id)

    timeline = _get_timeline_with_milestones(timeline_id)
    if not timeline:
        return []

    _cleanup_stale_suggestions(timeline_id)

    context = {
        'category': timeline.get('category_code', ''),
        'method': timeline.get('method_code', ''),
        'status': timeline.get('status', ''),
        'planned_start': timeline.get('planned_start_date', ''),
        'planned_end': timeline.get('planned_end_date', ''),
        'milestones': [],
    }

    for m in timeline.get('milestones', []):
        context['milestones'].append({
            'code': m.get('code', ''),
            'name': m.get('name', ''),
            'planned_date': m.get('planned_date', ''),
            'actual_date': m.get('actual_date', ''),
            'diff_days': m.get('diff_days'),
            'diff_reason': m.get('diff_reason', ''),
            'reason_category': m.get('reason_category', ''),
            'status': m.get('status', ''),
        })

    prompt = f"""你是中国招标投标领域的专家顾问。请基于以下项目时间线数据，提供精炼的风险评估和操作建议。

项目信息:
- 采购类别: {context['category']}
- 采购方式: {context['method']}
- 计划开始: {context['planned_start']}
- 计划结束: {context['planned_end']}

时间节点状态:
{json.dumps(context['milestones'], ensure_ascii=False, indent=2)}

请以JSON格式回复，包含以下字段:
- overall_risk: "low"/"medium"/"high"/"critical" — 项目总体风险评级
- risk_summary: 一段简短的风险总结 (100字以内)
- next_steps: 最重要的3-5条下一步操作建议 (每条20字以内)
- alerts: 需要特别注意的合规风险 (如有)

请确保只返回有效的JSON，不要包含其他内容。"""

    try:
        from app.services.llm_provider import call_llm
        raw = call_llm(
            system_prompt="你是一个专业的中国招标投标顾问。你提供的所有建议必须基于中国法律法规，准确、简洁、可直接操作。",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=1024,
            provider_id=os.getenv('AI_SUGGESTION_PROVIDER'),
            model=os.getenv('AI_SUGGESTION_MODEL'),
        )

        try:
            import re as _re
            json_match = _re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                ai_result = json.loads(json_match.group(0))
            else:
                ai_result = {'overall_risk': 'unknown', 'risk_summary': raw[:200], 'next_steps': [], 'alerts': []}
        except (json.JSONDecodeError, TypeError):
            ai_result = {'overall_risk': 'unknown', 'risk_summary': raw[:200] if raw else 'AI分析异常', 'next_steps': [], 'alerts': []}

        suggestions = []

        if ai_result.get('risk_summary'):
            risk_map = {'critical': 'critical', 'high': 'high', 'medium': 'medium', 'low': 'low'}
            ai_priority = risk_map.get(ai_result.get('overall_risk', 'medium'), 'medium')
            suggestions.append(_make_suggestion(
                timeline_id, 'AI_OVERALL', ai_priority, 'ai_risk_summary',
                f"AI风险评估: {ai_result['overall_risk']}",
                str(ai_result.get('risk_summary', ''))[:300]
            ))

        for step in ai_result.get('next_steps', [])[:MAX_AI_SUGGESTIONS]:
            suggestions.append(_make_suggestion(
                timeline_id, 'AI_STEP', 'medium', 'ai_next_step',
                "AI建议操作",
                str(step)[:200]
            ))

        for alert in ai_result.get('alerts', [])[:3]:
            suggestions.append(_make_suggestion(
                timeline_id, 'AI_ALERT', 'high', 'ai_alert',
                "AI合规提醒",
                str(alert)[:200]
            ))

        _persist_suggestions(suggestions)
        return suggestions

    except Exception as e:
        logger.error(f"AI suggestion generation failed: {e}")
        return []


def get_suggestions(timeline_id: int, include_read: bool = False) -> List[dict]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if include_read:
                cur.execute(
                    "SELECT id, timeline_id, milestone_code, type, priority, content, suggestion, is_read, is_actioned, created_at FROM timeline_suggestions WHERE timeline_id = %s ORDER BY priority_order() DESC, created_at DESC",
                    (timeline_id,))
            else:
                cur.execute(
                    "SELECT id, timeline_id, milestone_code, type, priority, content, suggestion, is_read, is_actioned, created_at FROM timeline_suggestions WHERE timeline_id = %s AND is_read = FALSE ORDER BY CASE priority WHEN 'critical' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 ELSE 4 END, created_at DESC",
                    (timeline_id,))
            results = []
            for row in cur.fetchall():
                results.append({
                    'id': row[0], 'timeline_id': row[1], 'milestone_code': row[2],
                    'type': row[3], 'priority': row[4], 'content': row[5],
                    'suggestion': row[6], 'is_read': row[7], 'is_actioned': row[8],
                    'created_at': row[9].isoformat() if row[9] else None,
                })
            return results


def dismiss_suggestion(suggestion_id: int) -> bool:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE timeline_suggestions SET is_read = TRUE WHERE id = %s",
                (suggestion_id,))
            if cur.rowcount == 0:
                cur.execute(
                    "UPDATE timeline_suggestions SET is_actioned = TRUE, is_read = TRUE WHERE id = %s",
                    (suggestion_id,))
            conn.commit()
            return cur.rowcount > 0


def _make_suggestion(timeline_id: int, milestone_code: str, priority: str,
                     suggestion_type: str, content: str, suggestion: str) -> dict:
    return {
        'timeline_id': timeline_id,
        'milestone_code': milestone_code,
        'type': suggestion_type,
        'priority': priority,
        'content': content[:200],
        'suggestion': suggestion[:300],
    }


def _cleanup_stale_suggestions(timeline_id: int):
    """Delete stale suggestions older than 3 days that haven't been actioned."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM timeline_suggestions WHERE timeline_id = %s"
                " AND created_at < NOW() - INTERVAL '3 days' AND is_actioned = FALSE",
                (timeline_id,))
            conn.commit()


def _persist_suggestions(suggestions: List[dict]):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for s in suggestions:
                cur.execute(
                    "SELECT 1 FROM timeline_suggestions WHERE timeline_id = %s AND milestone_code = %s AND type = %s AND content = %s AND created_at > NOW() - INTERVAL '24 hours'",
                    (s['timeline_id'], s['milestone_code'], s['type'], s['content']))
                if cur.fetchone():
                    continue
                cur.execute("""
                    INSERT INTO timeline_suggestions
                        (timeline_id, milestone_code, type, priority, content, suggestion)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    s['timeline_id'], s['milestone_code'], s['type'],
                    s['priority'], s['content'], s['suggestion'],
                ))
            conn.commit()


def _get_existing_ai_suggestions(timeline_id: int) -> List[dict]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, timeline_id, milestone_code, type, priority, content, suggestion, is_read, is_actioned, created_at FROM timeline_suggestions WHERE timeline_id = %s AND type LIKE 'ai_%%' AND created_at > NOW() - INTERVAL '24 hours' ORDER BY created_at DESC",
                (timeline_id,))
            return [
                {
                    'id': row[0], 'timeline_id': row[1], 'milestone_code': row[2],
                    'type': row[3], 'priority': row[4], 'content': row[5],
                    'suggestion': row[6], 'is_read': row[7], 'is_actioned': row[8],
                    'created_at': row[9].isoformat() if row[9] else None,
                }
                for row in cur.fetchall()
            ]


def _last_ai_generation(timeline_id: int) -> Optional[datetime]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MAX(created_at) FROM timeline_suggestions WHERE timeline_id = %s AND type LIKE 'ai_%%'",
                (timeline_id,))
            row = cur.fetchone()
            return row[0] if row and row[0] else None


def _get_timeline_with_milestones(timeline_id: int) -> Optional[dict]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, project_id, category_code, method_code, planned_start_date,
                       planned_end_date, actual_start_date, actual_end_date, status
                FROM project_timelines WHERE id = %s
            """, (timeline_id,))
            row = cur.fetchone()
            if not row:
                return None

            timeline = {
                'id': row[0], 'project_id': row[1], 'category_code': row[2],
                'method_code': row[3], 'planned_start_date': row[4].isoformat() if row[4] else None,
                'planned_end_date': row[5].isoformat() if row[5] else None,
                'actual_start_date': row[6].isoformat() if row[6] else None,
                'actual_end_date': row[7].isoformat() if row[7] else None,
                'status': row[8], 'milestones': [],
            }

            cur.execute("""
                SELECT milestone_code, milestone_name, planned_date, actual_date,
                       diff_days, diff_reason, reason_category, status, sort_order
                FROM project_timeline_milestones WHERE timeline_id = %s ORDER BY sort_order
            """, (timeline_id,))
            for m in cur.fetchall():
                if len(m) < 9:
                    logger.error(
                        f"Milestone schema mismatch timeline_id={timeline_id}: "
                        f"expected 9 columns, got {len(m)}. Returning empty milestones."
                    )
                    timeline['milestones'] = []
                    return timeline
                timeline['milestones'].append({
                    'code': m[0], 'name': m[1],
                    'planned_date': m[2].isoformat() if m[2] else None,
                    'actual_date': m[3].isoformat() if m[3] else None,
                    'diff_days': m[4], 'diff_reason': m[5],
                    'reason_category': m[6], 'status': m[7], 'sort_order': m[8],
                })
            return timeline
