"""清标 Blueprint — 统一入口：多投标文件 + 可选招标文件 → 全维度分析。

合并了 智能对比分析 / 投标文档深度分析 / 合规审查 / AI 文档审查 四个工具，
输出严格遵循《串通投标线索分析报告.docx》格式的 DOCX + PDF。

Endpoints:
    POST /clearance/run          — 启动清标（Celery 异步）
    GET  /clearance/status/<id>  — 轮询状态
    GET  /clearance/stream/<id>  — SSE 进度
"""
import json
import logging
import uuid

from flask import Blueprint, Response, request, session

from app.config import allowed_file
from app.utils.helpers import err, ok
from app.services.session_manager import get_user_id

logger = logging.getLogger(__name__)

clearance_bp = Blueprint('clearance', __name__, url_prefix='/clearance')


@clearance_bp.before_request
def _increase_body_limit():
    # 清标需上传多份投标文件（PDF 扫描件），全局 50MB 不够。提高到 200MB。
    request.max_content_length = 200 * 1024 * 1024


@clearance_bp.route('/run', methods=['POST'])
def run_clearance_route():
    """启动统一清标分析。

    Request: multipart/form-data
        files       — 投标文件（≥2，≤10）
        tender_file — 招标文件（可选，用于合规审查规则提取）
        options     — JSON 字符串: {
                          'indicator_analysis': bool,
                          'cross_comparison': bool,
                          'compliance_check': bool,
                          'ai_review': bool,
                      }
    """
    if session.get('consent_value', 0) != 1:
        return err("请先登录", "AUTH_REQUIRED", 401)
    user_id = get_user_id()
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)

    if 'files' not in request.files:
        return err("No files uploaded", "VALIDATION_ERROR", 400)
    uploaded_files = request.files.getlist('files')
    if len(uploaded_files) < 2:
        return err("清标至少需要 2 份投标文件", "VALIDATION_ERROR", 400)
    if len(uploaded_files) > 10:
        return err("最多支持 10 份投标文件", "VALIDATION_ERROR", 400)

    from app.services.file_processing import extract_text_from_file, extract_metadata
    file_data = []
    for f in uploaded_files:
        if not f.filename or not allowed_file(f.filename):
            continue
        text, _ = extract_text_from_file(f)
        if not text or text.startswith("["):
            continue
        meta = extract_metadata(f)
        file_data.append({
            'filename': f.filename,
            'text': text,
            'metadata': meta or {},
            'images': [],
        })
    if len(file_data) < 2:
        return err("至少需要 2 份可提取文本的投标文件", "VALIDATION_ERROR", 400)

    # 招标文件（可选）
    tender_text = None
    tender_name = None
    tender_file = request.files.get('tender_file')
    if tender_file and tender_file.filename and allowed_file(tender_file.filename):
        ttext, _ = extract_text_from_file(tender_file)
        if ttext and not ttext.startswith("["):
            tender_text = ttext
            tender_name = tender_file.filename

    # 分析维度选项
    try:
        options = json.loads(request.form.get('options', '{}') or '{}')
    except (json.JSONDecodeError, TypeError):
        options = {}
    options.setdefault('indicator_analysis', True)
    options.setdefault('cross_comparison', True)
    options.setdefault('compliance_check', bool(tender_text))
    options.setdefault('ai_review', True)
    # 未提供招标文件时强制关闭合规
    if not tender_text:
        options['compliance_check'] = False

    task_id = str(uuid.uuid4())
    thread_id = session.get('thread_id', '')
    project_id = request.form.get('project_id', type=int)

    from celery_app import celery
    celery.send_task(
        'clearance_task',
        args=[file_data, tender_text, tender_name, options, user_id, thread_id, task_id, project_id],
        task_id=task_id,
    )

    return ok({'task_id': task_id, 'file_count': len(file_data)})


@clearance_bp.route('/status/<task_id>', methods=['GET'])
def clearance_status(task_id):
    from app.services.task_bus import TaskBus
    meta = TaskBus.get(task_id)
    if not meta:
        return err("Task not found", "NOT_FOUND", 404)
    completed = meta.get('status') == 'completed'
    return ok({
        'completed': completed,
        'status': meta.get('status', ''),
        'progress': meta.get('progress', 0),
        'message': meta.get('message', ''),
        'result': json.loads(meta.get('result', 'null')) if completed and meta.get('result') else None,
    })


@clearance_bp.route('/stream/<task_id>', methods=['GET'])
def clearance_stream(task_id):
    from app.services.task_bus import TaskBus
    return Response(
        TaskBus.subscribe(task_id),
        mimetype='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        }
    )
