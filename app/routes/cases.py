"""Audit case library API routes (U13).

Endpoints:
    GET  /cases                    — Paginated case list
    GET  /cases/stats              — Case library statistics
    GET  /cases/<id>               — Case detail
    PUT  /cases/<id>               — Update case
    DELETE /cases/<id>             — Delete case
    POST /cases/<id>/tags          — Add tags
    DELETE /cases/<id>/tags        — Remove tags
    POST /cases/<id>/laws          — Link law article
    DELETE /cases/<id>/laws/<article_id>  — Unlink law
    POST /cases/<id>/templates     — Link template section
    DELETE /cases/<id>/templates/<link_id> — Unlink template
    POST /cases/auto-generate      — Auto-generate from audit run
"""

import logging
from functools import wraps

from flask import Blueprint, request, session
from app.utils.helpers import ok, err

logger = logging.getLogger(__name__)

cases_bp = Blueprint('cases', __name__, url_prefix='/cases')


def _login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            return err("\u8bf7\u5148\u767b\u5f55", "UNAUTHORIZED", 401)
        return f(*args, **kwargs)
    return decorated


def _admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'admin':
            return err("Admin access required", "FORBIDDEN", 403)
        return f(*args, **kwargs)
    return decorated


@cases_bp.route('', methods=['GET'])
@_login_required
def list_cases():
    try:
        from app.services.case_service import list_cases
        severity = request.args.get('severity')
        category = request.args.get('category')
        resolved = request.args.get('resolved')
        if resolved is not None:
            resolved = resolved.lower() == 'true'
        search = request.args.get('search')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        result = list_cases(severity=severity, category=category,
                            resolved=resolved, search=search,
                            page=page, per_page=per_page)
        return ok(result)
    except Exception as e:
        logger.error(f"list_cases error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@cases_bp.route('/stats', methods=['GET'])
@_login_required
def get_stats():
    try:
        from app.services.case_service import get_stats
        return ok(get_stats())
    except Exception as e:
        logger.error(f"get_stats error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@cases_bp.route('/<int:case_id>', methods=['GET'])
@_login_required
def get_case(case_id: int):
    try:
        from app.services.case_service import get_case
        result = get_case(case_id)
        if result is None:
            return err("\u6848\u4f8b\u4e0d\u5b58\u5728", "NOT_FOUND", 404)
        return ok(result)
    except Exception as e:
        logger.error(f"get_case error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@cases_bp.route('/<int:case_id>', methods=['PUT'])
@_login_required
def update_case(case_id: int):
    try:
        if not request.is_json:
            return err("\u8bf7\u6c42\u4f53\u9700\u4e3a JSON", "VALIDATION_ERROR", 400)
        data = request.get_json() or {}
        from app.services.case_service import update_case
        result = update_case(case_id, **data)
        if result is None:
            return err("\u6848\u4f8b\u4e0d\u5b58\u5728", "NOT_FOUND", 404)
        return ok(result, message="\u6848\u4f8b\u5df2\u66f4\u65b0")
    except Exception as e:
        logger.error(f"update_case error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@cases_bp.route('/<int:case_id>', methods=['DELETE'])
@_admin_required
def delete_case(case_id: int):
    try:
        from app.services.case_service import delete_case
        if not delete_case(case_id):
            return err("\u6848\u4f8b\u4e0d\u5b58\u5728", "NOT_FOUND", 404)
        return ok(message="\u6848\u4f8b\u5df2\u5220\u9664")
    except Exception as e:
        logger.error(f"delete_case error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@cases_bp.route('/<int:case_id>/tags', methods=['POST'])
@_login_required
def add_tags(case_id: int):
    try:
        if not request.is_json:
            return err("\u8bf7\u6c42\u4f53\u9700\u4e3a JSON", "VALIDATION_ERROR", 400)
        data = request.get_json() or {}
        tags = data.get('tags', [])
        if not tags:
            return err("\u7f3a\u5c11 tags", "VALIDATION_ERROR", 400)
        from app.services.case_service import add_tags
        result = add_tags(case_id, tags)
        return ok(result, message="\u6807\u7b7e\u5df2\u6dfb\u52a0")
    except Exception as e:
        logger.error(f"add_tags error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@cases_bp.route('/<int:case_id>/tags', methods=['DELETE'])
@_login_required
def remove_tags(case_id: int):
    try:
        if not request.is_json:
            return err("\u8bf7\u6c42\u4f53\u9700\u4e3a JSON", "VALIDATION_ERROR", 400)
        data = request.get_json() or {}
        tags = data.get('tags', [])
        if not tags:
            return err("\u7f3a\u5c11 tags", "VALIDATION_ERROR", 400)
        from app.services.case_service import remove_tags
        result = remove_tags(case_id, tags)
        return ok(result, message="\u6807\u7b7e\u5df2\u5220\u9664")
    except Exception as e:
        logger.error(f"remove_tags error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@cases_bp.route('/<int:case_id>/laws', methods=['POST'])
@_login_required
def link_law(case_id: int):
    try:
        if not request.is_json:
            return err("\u8bf7\u6c42\u4f53\u9700\u4e3a JSON", "VALIDATION_ERROR", 400)
        data = request.get_json() or {}
        article_id = data.get('article_id')
        relation = data.get('relation', 'cited')
        if not article_id:
            return err("\u7f3a\u5c11 article_id", "VALIDATION_ERROR", 400)
        from app.services.case_service import link_law
        result = link_law(case_id, int(article_id), relation)
        return ok(result, message="\u6cd5\u89c4\u5df2\u5173\u8054")
    except Exception as e:
        logger.error(f"link_law error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@cases_bp.route('/<int:case_id>/laws/<int:article_id>', methods=['DELETE'])
@_login_required
def unlink_law(case_id: int, article_id: int):
    try:
        from app.services.case_service import unlink_law
        result = unlink_law(case_id, article_id)
        return ok(result, message="\u6cd5\u89c4\u5df2\u53d6\u6d88\u5173\u8054")
    except Exception as e:
        logger.error(f"unlink_law error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@cases_bp.route('/<int:case_id>/templates', methods=['POST'])
@_login_required
def link_template(case_id: int):
    try:
        if not request.is_json:
            return err("\u8bf7\u6c42\u4f53\u9700\u4e3a JSON", "VALIDATION_ERROR", 400)
        data = request.get_json() or {}
        template_id = data.get('template_id')
        section_id = data.get('section_id')
        relation = data.get('relation', 'related')
        if not template_id:
            return err("\u7f3a\u5c11 template_id", "VALIDATION_ERROR", 400)
        from app.services.case_service import link_template
        result = link_template(case_id, int(template_id), section_id, relation)
        return ok(result, message="\u6a21\u677f\u5df2\u5173\u8054")
    except Exception as e:
        logger.error(f"link_template error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@cases_bp.route('/<int:case_id>/templates/<int:link_id>', methods=['DELETE'])
@_login_required
def unlink_template(case_id: int, link_id: int):
    try:
        from app.services.case_service import unlink_template
        result = unlink_template(case_id, link_id)
        return ok(result, message="\u6a21\u677f\u5df2\u53d6\u6d88\u5173\u8054")
    except Exception as e:
        logger.error(f"unlink_template error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@cases_bp.route('/auto-generate', methods=['POST'])
@_admin_required
def auto_generate_cases():
    try:
        if not request.is_json:
            return err("\u8bf7\u6c42\u4f53\u9700\u4e3a JSON", "VALIDATION_ERROR", 400)
        data = request.get_json() or {}
        run_id = data.get('run_id')
        if not run_id:
            return err("\u7f3a\u5c11 run_id", "VALIDATION_ERROR", 400)
        from app.services.case_service import auto_generate_from_run
        new_ids = auto_generate_from_run(int(run_id))
        return ok({
            "run_id": run_id,
            "generated": len(new_ids),
            "case_ids": new_ids,
        }, message=f"\u81ea\u52a8\u751f\u6210 {len(new_ids)} \u4e2a\u6848\u4f8b")
    except Exception as e:
        logger.error(f"auto_generate error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)
