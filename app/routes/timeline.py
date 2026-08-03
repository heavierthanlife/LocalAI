"""Blueprint: timeline — project bidding timeline management."""
import logging
from datetime import date

from flask import Blueprint, request, session

from app.utils.helpers import ok, err

logger = logging.getLogger(__name__)

timeline_bp = Blueprint('timeline', __name__, url_prefix='/timeline')


def _parse_date(val):
    if not val:
        return None
    try:
        return date.fromisoformat(val)
    except (ValueError, TypeError):
        return None


def _admin_or_project_member(project_id):
    role = session.get('role', 'user')
    if role == 'admin':
        return True
    user_id = session.get('user_id')
    if not user_id or not project_id:
        return False
    from app.database import get_db_connection
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM project_members WHERE project_id = %s AND user_id = %s",
                (project_id, user_id))
            return cur.fetchone() is not None


@timeline_bp.route('/<int:project_id>', methods=['POST'])
def create_timeline(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()
    category_code = data.get('category_code', '').strip()
    method_code = data.get('method_code', '').strip()
    planned_start = _parse_date(data.get('planned_start_date'))
    planned_end = _parse_date(data.get('planned_end_date'))

    if not category_code or not method_code:
        from app.services.project_timeline_service import get_timeline
        latest = get_timeline(project_id)
        if latest:
            category_code = category_code or latest.get('category_code', 'general')
            method_code = method_code or latest.get('method_code', 'open')
        if not category_code:
            category_code = 'general'
        if not method_code:
            method_code = 'open'

    if not planned_start:
        from datetime import date as _date
        planned_start = _date.today()

    if not name:
        name = '主招标流程'

    if not category_code or not method_code:
        return err("请提供 category_code 和 method_code", "VALIDATION_ERROR", 400)

    try:
        from app.services.project_timeline_service import create_timeline as svc_create
        timeline = svc_create(
            project_id=project_id, name=name,
            category_code=category_code, method_code=method_code,
            planned_start_date=planned_start, planned_end_date=planned_end,
            created_by=session.get('user_id'),
        )
        return ok(timeline, "时间线已创建")
    except ValueError as e:
        return err(str(e), "VALIDATION_ERROR", 400)
    except Exception as e:
        logger.error(f"create_timeline failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>', methods=['GET'])
def get_timeline(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    try:
        timeline_id = request.args.get('timeline_id', type=int)
        from app.services.project_timeline_service import get_timeline as svc_get
        timeline = svc_get(project_id, timeline_id=timeline_id)
        if not timeline:
            return err("未找到时间线", "NOT_FOUND", 404)
        return ok(timeline)
    except Exception as e:
        logger.error(f"get_timeline failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/list', methods=['GET'])
def list_timelines(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    try:
        from app.services.project_timeline_service import list_timelines as svc_list
        timelines = svc_list(project_id)
        return ok({'timelines': timelines})
    except Exception as e:
        logger.error(f"list_timelines failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>', methods=['PUT'])
def update_timeline(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    data = request.get_json(silent=True) or {}
    try:
        from app.services.project_timeline_service import get_timeline as svc_get
        timeline_id = request.args.get('timeline_id', type=int)
        tl = svc_get(project_id, timeline_id=timeline_id)
        if not tl:
            return err("未找到时间线", "NOT_FOUND", 404)

        from app.services.project_timeline_service import update_timeline_dates
        result = update_timeline_dates(
            timeline_id=tl['id'],
            planned_start_date=_parse_date(data.get('planned_start_date')),
            planned_end_date=_parse_date(data.get('planned_end_date')),
            actual_start_date=_parse_date(data.get('actual_start_date')),
            actual_end_date=_parse_date(data.get('actual_end_date')),
            status=data.get('status'),
        )
        return ok(result, "时间线已更新")
    except Exception as e:
        logger.error(f"update_timeline failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/milestones', methods=['GET'])
def list_milestones(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    try:
        from app.services.project_timeline_service import get_timeline as svc_get
        timeline_id = request.args.get('timeline_id', type=int)
        tl = svc_get(project_id, timeline_id=timeline_id)
        if not tl:
            return err("未找到时间线", "NOT_FOUND", 404)
        return ok({'milestones': tl['milestones'], 'diff_summary': tl['diff_summary']})
    except Exception as e:
        logger.error(f"list_milestones failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/milestones/<milestone_code>', methods=['PUT'])
def update_milestone(project_id, milestone_code):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    data = request.get_json(silent=True) or {}
    try:
        from app.services.project_timeline_service import get_timeline as svc_get
        timeline_id = request.args.get('timeline_id', type=int)
        tl = svc_get(project_id, timeline_id=timeline_id)
        if not tl:
            return err("未找到时间线", "NOT_FOUND", 404)

        from app.services.project_timeline_service import update_milestone as svc_update
        result = svc_update(
            timeline_id=tl['id'],
            milestone_code=milestone_code,
            actual_date=_parse_date(data.get('actual_date')),
            diff_reason=data.get('diff_reason'),
            reason_category=data.get('reason_category'),
            status=data.get('status'),
            created_by=session.get('user_id'),
        )
        return ok(result, "里程碑已更新")
    except ValueError as e:
        return err(str(e), "VALIDATION_ERROR", 400)
    except Exception as e:
        logger.error(f"update_milestone failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/milestones/batch', methods=['POST'])
def batch_update_milestones(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    data = request.get_json(silent=True) or {}
    updates = data.get('updates', [])
    if not updates or not isinstance(updates, list):
        return err("请提供 updates 数组", "VALIDATION_ERROR", 400)

    try:
        from app.services.project_timeline_service import get_timeline as svc_get
        timeline_id = request.args.get('timeline_id', type=int)
        tl = svc_get(project_id, timeline_id=timeline_id)
        if not tl:
            return err("未找到时间线", "NOT_FOUND", 404)

        parsed = []
        for u in updates:
            parsed.append({
                'code': u.get('code'),
                'actual_date': _parse_date(u.get('actual_date')),
                'diff_reason': u.get('diff_reason'),
                'reason_category': u.get('reason_category'),
                'status': u.get('status'),
            })

        from app.services.project_timeline_service import batch_update_milestones as svc_batch
        results = svc_batch(tl['id'], parsed, created_by=session.get('user_id'))
        return ok({'results': results})
    except Exception as e:
        logger.error(f"batch_update_milestones failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/diff', methods=['GET'])
def get_diff_report(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    try:
        from app.services.project_timeline_service import get_timeline as svc_get
        timeline_id = request.args.get('timeline_id', type=int)
        tl = svc_get(project_id, timeline_id=timeline_id)
        if not tl:
            return err("未找到时间线", "NOT_FOUND", 404)

        from app.services.project_timeline_service import get_diff_report as svc_diff
        report = svc_diff(tl['id'])
        return ok(report)
    except Exception as e:
        logger.error(f"get_diff_report failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/legal/categories', methods=['GET'])
def list_categories():
    try:
        from app.services.legal_schedule_service import get_categories, get_methods
        cats = []
        for c in get_categories():
            methods = get_methods(c['code'])
            cats.append({
                'code': c['code'],
                'name': c['name'],
                'regime': c.get('regime', ''),
                'methods': methods,
            })
        return ok({'categories': cats})
    except Exception as e:
        logger.error(f"list_categories failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/legal/schedule', methods=['GET'])
def get_schedule():
    category = request.args.get('category', '').strip()
    method = request.args.get('method', '').strip()
    if not category or not method:
        return err("请提供 category 和 method 参数", "VALIDATION_ERROR", 400)

    try:
        from app.services.legal_schedule_service import get_schedule as svc_schedule
        schedule = svc_schedule(category, method)
        if not schedule:
            return err("未找到对应的法律排期模板", "NOT_FOUND", 404)
        return ok(schedule)
    except Exception as e:
        logger.error(f"get_schedule failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/steps', methods=['GET'])
def list_steps(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)
    milestone_code = request.args.get('milestone', '').strip() or None

    try:
        from app.services.project_timeline_service import get_timeline as svc_get
        timeline_id = request.args.get('timeline_id', type=int)
        tl = svc_get(project_id, timeline_id=timeline_id)
        if not tl:
            return err("未找到时间线", "NOT_FOUND", 404)

        from app.services.workflow_service import list_steps as svc_list
        steps = svc_list(tl['id'], milestone_code)
        return ok({'steps': steps})
    except Exception as e:
        logger.error(f"list_steps failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/steps', methods=['POST'])
def create_step(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    data = request.get_json(silent=True) or {}
    milestone_code = data.get('milestone_code', '').strip()
    step_name = data.get('step_name', '').strip()
    if not milestone_code or not step_name:
        return err("请提供 milestone_code 和 step_name", "VALIDATION_ERROR", 400)

    try:
        from app.services.project_timeline_service import get_timeline as svc_get
        timeline_id = request.args.get('timeline_id', type=int)
        tl = svc_get(project_id, timeline_id=timeline_id)
        if not tl:
            return err("未找到时间线", "NOT_FOUND", 404)

        from app.services.workflow_service import create_step as svc_create
        result = svc_create(
            timeline_id=tl['id'],
            milestone_code=milestone_code,
            step_name=step_name,
            step_order=data.get('step_order', 0),
            assigned_to=data.get('assigned_to'),
        )
        return ok(result, "步骤已创建")
    except ValueError as e:
        return err(str(e), "VALIDATION_ERROR", 400)
    except Exception as e:
        logger.error(f"create_step failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/steps/<int:step_id>', methods=['PUT'])
def update_step(project_id, step_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    data = request.get_json(silent=True) or {}
    try:
        from app.services.workflow_service import update_step as svc_update
        result = svc_update(
            step_id=step_id,
            completed=data.get('completed'),
            notes=data.get('notes'),
            assigned_to=data.get('assigned_to'),
        )
        return ok(result, "步骤已更新")
    except ValueError as e:
        return err(str(e), "VALIDATION_ERROR", 400)
    except Exception as e:
        logger.error(f"update_step failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/steps/<int:step_id>', methods=['DELETE'])
def delete_step(project_id, step_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    try:
        from app.services.workflow_service import delete_step as svc_delete
        ok_result = svc_delete(step_id)
        if not ok_result:
            return err("步骤未找到", "NOT_FOUND", 404)
        return ok(None, "步骤已删除")
    except Exception as e:
        logger.error(f"delete_step failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/steps/batch', methods=['POST'])
def batch_complete_steps(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    data = request.get_json(silent=True) or {}
    step_ids = data.get('step_ids', [])
    if not step_ids or not isinstance(step_ids, list):
        return err("请提供 step_ids 数组", "VALIDATION_ERROR", 400)

    try:
        from app.services.workflow_service import batch_complete_steps as svc_batch
        results = svc_batch(step_ids)
        return ok({'results': results})
    except Exception as e:
        logger.error(f"batch_complete_steps failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/steps/generate', methods=['POST'])
def auto_generate_steps(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    try:
        from app.services.project_timeline_service import get_timeline as svc_get
        timeline_id = request.args.get('timeline_id', type=int)
        tl = svc_get(project_id, timeline_id=timeline_id)
        if not tl:
            return err("未找到时间线", "NOT_FOUND", 404)

        from app.services.workflow_service import auto_generate_steps as svc_gen
        results = svc_gen(tl['id'])
        return ok({'generated': len(results), 'steps': results}, f"已生成 {len(results)} 个步骤")
    except Exception as e:
        logger.error(f"auto_generate_steps failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/steps/progress', methods=['GET'])
def get_workflow_progress(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    try:
        from app.services.project_timeline_service import get_timeline as svc_get
        timeline_id = request.args.get('timeline_id', type=int)
        tl = svc_get(project_id, timeline_id=timeline_id)
        if not tl:
            return err("未找到时间线", "NOT_FOUND", 404)

        from app.services.workflow_service import get_progress as svc_prog
        progress = svc_prog(tl['id'])
        return ok(progress)
    except Exception as e:
        logger.error(f"get_workflow_progress failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/suggestions', methods=['GET'])
def get_suggestions(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    try:
        from app.services.project_timeline_service import get_timeline as svc_get
        timeline_id = request.args.get('timeline_id', type=int)
        tl = svc_get(project_id, timeline_id=timeline_id)
        if not tl:
            return err("未找到时间线", "NOT_FOUND", 404)

        from app.services.suggestion_engine import evaluate_timeline, get_suggestions as svc_list
        evaluate_timeline(tl['id'], project_id)
        suggestions = svc_list(tl['id'])
        return ok({'suggestions': suggestions})
    except Exception as e:
        logger.error(f"get_suggestions failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/suggestions/<int:suggestion_id>', methods=['POST'])
def dismiss_suggestion(project_id, suggestion_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    try:
        from app.services.suggestion_engine import dismiss_suggestion as svc_dismiss
        ok_result = svc_dismiss(suggestion_id)
        if not ok_result:
            return err("建议未找到", "NOT_FOUND", 404)
        return ok(None, "已忽略")
    except Exception as e:
        logger.error(f"dismiss_suggestion failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@timeline_bp.route('/<int:project_id>/suggestions/generate', methods=['POST'])
def generate_ai_suggestions(project_id):
    if not _admin_or_project_member(project_id):
        return err("无权操作此项目", "FORBIDDEN", 403)

    try:
        from app.services.project_timeline_service import get_timeline as svc_get
        timeline_id = request.args.get('timeline_id', type=int)
        tl = svc_get(project_id, timeline_id=timeline_id)
        if not tl:
            return err("未找到时间线", "NOT_FOUND", 404)

        from app.services.suggestion_engine import generate_ai_suggestions as svc_ai
        results = svc_ai(tl['id'], project_id)
        return ok({'suggestions': results}, f"AI生成了 {len(results)} 条建议")
    except Exception as e:
        logger.error(f"generate_ai_suggestions failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


