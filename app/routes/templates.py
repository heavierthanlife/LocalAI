
import json
import logging
from functools import wraps

from flask import Blueprint, request, session, send_file
from app.utils.helpers import ok, err

logger = logging.getLogger(__name__)

templates_bp = Blueprint('templates', __name__, url_prefix='/templates')


def _login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            return err("请先登录", "UNAUTHORIZED", 401)
        return f(*args, **kwargs)
    return decorated


@templates_bp.route('', methods=['GET'])
@_login_required
def list_templates():
    try:
        from app.services.template_service import list_templates
        category = request.args.get('category')
        tag = request.args.get('tag')
        search = request.args.get('search')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        result = list_templates(category=category, tag=tag, search=search,
                                page=page, per_page=per_page)
        return ok(result)
    except Exception as e:
        logger.error(f"list_templates error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@templates_bp.route('/<int:template_id>', methods=['GET'])
@_login_required
def get_template(template_id: int):
    try:
        from app.services.template_service import get_template
        result = get_template(template_id)
        if result is None:
            return err("模板不存在", "NOT_FOUND", 404)
        return ok(result)
    except Exception as e:
        logger.error(f"get_template error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@templates_bp.route('', methods=['POST'])
@_login_required
def create_template():
    try:
        if not request.is_json:
            return err("请求体需为 JSON", "VALIDATION_ERROR", 400)
        data = request.get_json() or {}
        name = data.get('name')
        category = data.get('category', '工程')
        sections = data.get('sections', [])
        if not name:
            return err("缺少 name", "VALIDATION_ERROR", 400)
        if not sections:
            return err("缺少 sections", "VALIDATION_ERROR", 400)
        from app.services.template_service import create_template
        result = create_template(
            name=name, category=category, sections=sections,
            description=data.get('description'),
            tags=data.get('tags'),
            created_by=session.get('user_id'),
        )
        return ok(result, message="模板已创建")
    except ValueError as e:
        return err(str(e), "VALIDATION_ERROR", 400)
    except Exception as e:
        logger.error(f"create_template error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@templates_bp.route('/<int:template_id>', methods=['PUT'])
@_login_required
def update_template(template_id: int):
    try:
        if not request.is_json:
            return err("请求体需为 JSON", "VALIDATION_ERROR", 400)
        data = request.get_json() or {}
        from app.services.template_service import update_template
        result = update_template(
            template_id,
            **{k: v for k, v in data.items() if v is not None},
            created_by=session.get('user_id'),
        )
        if result is None:
            return err("模板不存在", "NOT_FOUND", 404)
        return ok(result, message="模板已更新")
    except ValueError as e:
        return err(str(e), "VALIDATION_ERROR", 400)
    except Exception as e:
        logger.error(f"update_template error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@templates_bp.route('/<int:template_id>', methods=['DELETE'])
@_login_required
def delete_template(template_id: int):
    try:
        from app.services.template_service import delete_template
        if not delete_template(template_id):
            return err("模板不存在", "NOT_FOUND", 404)
        return ok(message="模板已删除")
    except Exception as e:
        logger.error(f"delete_template error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@templates_bp.route('/import/preview', methods=['POST'])
@_login_required
def import_preview():
    try:
        if 'file' not in request.files:
            return err("缺少文件", "VALIDATION_ERROR", 400)
        f = request.files['file']
        if not f.filename:
            return err("文件名为空", "VALIDATION_ERROR", 400)
        if not f.filename.lower().endswith('.docx'):
            return err("仅支持 .docx 文件", "VALIDATION_ERROR", 400)
        from app.services.template_service import import_from_docx_preview
        result = import_from_docx_preview(f.read(), f.filename)
        return ok(result, message="文件解析完成，请确认后导入")
    except Exception as e:
        logger.error(f"import_preview error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@templates_bp.route('/import/confirm', methods=['POST'])
@_login_required
def import_confirm():
    try:
        if not request.is_json:
            return err("请求体需为 JSON", "VALIDATION_ERROR", 400)
        data = request.get_json() or {}
        name = data.get('name')
        category = data.get('category', '工程')
        sections = data.get('sections', [])
        if not name:
            return err("缺少 name", "VALIDATION_ERROR", 400)
        if not sections:
            return err("缺少 sections", "VALIDATION_ERROR", 400)
        from app.services.template_service import create_template
        result = create_template(
            name=name, category=category, sections=sections,
            description=data.get('description'),
            tags=data.get('tags'),
            created_by=session.get('user_id'),
        )
        return ok(result, message="模板导入成功")
    except ValueError as e:
        return err(str(e), "VALIDATION_ERROR", 400)
    except Exception as e:
        logger.error(f"import_confirm error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@templates_bp.route('/<int:template_id>/export', methods=['GET'])
@_login_required
def export_template(template_id: int):
    try:
        import io
        from app.services.template_service import get_template
        from app.services.template_renderer import render_template_to_docx
        template = get_template(template_id)
        if not template:
            return err("模板不存在", "NOT_FOUND", 404)

        md = _sections_to_markdown(template['name'], template['sections'])
        docx_bytes = render_template_to_docx(md, title=template['name'])

        return send_file(
            io.BytesIO(docx_bytes),
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            as_attachment=True,
            download_name=f"{template['name']}.docx",
        )
    except Exception as e:
        logger.error(f"export_template error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@templates_bp.route('/<int:template_id>/versions', methods=['GET'])
@_login_required
def list_template_versions(template_id: int):
    try:
        from app.services.template_service import list_versions
        versions = list_versions(template_id)
        return ok({"template_id": template_id, "versions": versions})
    except Exception as e:
        logger.error(f"list_template_versions error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@templates_bp.route('/<int:template_id>/versions/<int:version_id>', methods=['GET'])
@_login_required
def get_template_version(template_id: int, version_id: int):
    try:
        from app.services.template_service import get_version
        result = get_version(version_id)
        if result is None:
            return err("版本不存在", "NOT_FOUND", 404)
        return ok(result)
    except Exception as e:
        logger.error(f"get_template_version error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@templates_bp.route('/<int:template_id>/diff', methods=['GET'])
@_login_required
def get_template_diff_route(template_id: int):
    try:
        from_vid = request.args.get('from', type=int)
        to_vid = request.args.get('to', type=int)
        if not from_vid or not to_vid:
            return err("缺少 from 和 to 参数 (version_id)", "VALIDATION_ERROR", 400)
        from app.services.template_diff import compute_template_diff
        result = compute_template_diff(template_id, from_vid, to_vid)
        return ok(result)
    except Exception as e:
        logger.error(f"get_template_diff error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@templates_bp.route('/recommend', methods=['POST'])
@_login_required
def recommend():
    """AI recommend templates for a bid project (U8)."""
    try:
        if not request.is_json:
            return err("\u8bf7\u6c42\u4f53\u9700\u4e3a JSON", "VALIDATION_ERROR", 400)
        data = request.get_json() or {}

        if data.get('log_usage'):
            template_id = data.get('template_id')
            if template_id:
                from app.services.template_recommend import log_template_usage
                log_template_usage(int(template_id), session.get('user_id'))
            return ok(message="\u4f7f\u7528\u5df2\u8bb0\u5f55")

        project_type = data.get('project_type')
        bid_text = data.get('bid_text')
        category = data.get('category')
        top_k = data.get('top_k', 5, type=int)
        from app.services.template_recommend import recommend_templates
        result = recommend_templates(
            project_type=project_type, bid_text=bid_text,
            category=category, top_k=top_k,
        )
        return ok({'recommendations': result, 'total': len(result)})
    except Exception as e:
        logger.error(f"template_recommend error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


def _sections_to_markdown(title: str, sections: list[dict]) -> str:
    lines = [f"# {title}", ""]
    for sec in sections:
        prefix = '#' * min(sec.get('level', 1) + 1, 4)
        lines.append(f"{prefix} {sec.get('title', '')}")
        if sec.get('content'):
            lines.append("")
            lines.append(sec['content'])
        lines.append("")
    return "\n".join(lines)


@templates_bp.route('/<int:file_id>/generate_doc', methods=['POST'])
@_login_required
def generate_doc_from_skill(file_id):
    try:
        import io
        from app.services.template_renderer import render_template_to_docx
        from app.database import get_db_connection

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT skill_summary, filename
                    FROM knowledge_lab_files
                    WHERE id = %s AND skill_summary IS NOT NULL
                """, (file_id,))
                row = cur.fetchone()
                if not row:
                    cur.execute("""
                        SELECT skill_summary, filename
                        FROM company_knowledge_base
                        WHERE id = %s AND skill_summary IS NOT NULL
                    """, (file_id,))
                    row = cur.fetchone()

        if not row:
            return err("技能模板不存在或没有摘要内容", "NOT_FOUND", 404)

        skill_md, filename = row
        docx_bytes = render_template_to_docx(skill_md, title=filename)

        return send_file(
            io.BytesIO(docx_bytes),
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            as_attachment=True,
            download_name=f"{filename}.docx",
        )
    except Exception as e:
        logger.error(f"generate_doc_from_skill error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)
