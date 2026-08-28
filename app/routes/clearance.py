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
import os
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

    # ── New flow: pre-uploaded file_ids (large-file friendly, zero body) ──
    from app.services.file_store import resolve as resolve_file
    from app.services.file_processing import extract_text_from_file, extract_metadata

    file_ids = request.form.getlist('file_ids')
    # Back-compat: direct multipart upload (small files)
    uploaded_files = request.files.getlist('files') if 'files' in request.files else []

    if not file_ids and not uploaded_files:
        return err("No files uploaded", "VALIDATION_ERROR", 400)
    if len(file_ids) + len(uploaded_files) > 10:
        return err("最多支持 10 份投标文件", "VALIDATION_ERROR", 400)

    resolved = []   # [{'abs_path','filename','size'}]
    for fid in file_ids:
        info = resolve_file(fid, user_id=user_id)
        if not info:
            return err(f"文件不存在或无权访问: file_id={fid}", "NOT_FOUND", 404)
        if not allowed_file(info['filename']):
            continue
        resolved.append({'abs_path': info['abs_path'], 'filename': info['filename']})

    # Legacy direct-upload path — small files, extract inline as before
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

    if len(file_data) + len(resolved) < 2:
        return err("至少需要 2 份投标文件", "VALIDATION_ERROR", 400)

    # 招标文件（可选）— pre-uploaded id or direct upload
    tender_text = None
    tender_name = None
    tender_spec = None
    tender_fid = request.form.get('tender_file_id')
    if tender_fid:
        tinfo = resolve_file(tender_fid, user_id=user_id)
        if tinfo and allowed_file(tinfo['filename']):
            tender_spec = {'abs_path': tinfo['abs_path'], 'filename': tinfo['filename']}
    tender_file = request.files.get('tender_file')
    if not tender_spec and tender_file and tender_file.filename and allowed_file(tender_file.filename):
        ttext, _ = extract_text_from_file(tender_file)
        if ttext and not ttext.startswith("["):
            tender_text = ttext
            tender_name = tender_file.filename

    # 开标信息表（可选）— pre-uploaded id; 结构化解析（Excel/CSV/JSON）
    open_info = None
    eval_criteria = None
    open_fid = request.form.get('open_info_file_id')
    if open_fid:
        oinfo = resolve_file(open_fid, user_id=user_id)
        if oinfo and os.path.exists(oinfo['abs_path']):
            from app.services.clearance_openinfo import parse_open_info_file, extract_eval_criteria
            open_info = parse_open_info_file(oinfo['abs_path'], oinfo['filename'])
            if not open_info.get('parsed'):
                logger.warning(f"开标信息表解析失败: {open_info.get('error')}")
                open_info = None
            else:
                logger.info(f"开标信息表解析: {len(open_info.get('rows', []))} 行")
    # 评审标准：从招标文件自动提取（供预览 + 人工确认）
    if tender_text:
        from app.services.clearance_openinfo import extract_eval_criteria
        eval_criteria = extract_eval_criteria(tender_text)

    # 分析维度选项
    try:
        options = json.loads(request.form.get('options', '{}') or '{}')
    except (json.JSONDecodeError, TypeError):
        options = {}
    has_tender = bool(tender_text or tender_spec)
    options.setdefault('indicator_analysis', True)
    options.setdefault('cross_comparison', True)
    options.setdefault('compliance_check', has_tender)
    options.setdefault('ai_review', True)
    # 未提供招标文件时强制关闭合规
    if not has_tender:
        options['compliance_check'] = False

    task_id = str(uuid.uuid4())
    thread_id = session.get('thread_id', '')
    project_id = request.form.get('project_id', type=int)

    # Pre-register the task as 'queued' so /clearance/status never 404s in the
    # window before the Celery worker calls bus.start() (which upgrades it).
    from app.services.task_bus import TaskBus
    TaskBus(task_id, 'clearance', '清标分析').register_queued(extra={'thread_id': thread_id or ''})

    info_overrides = {}
    for field in ('bid_number', 'bid_open_time', 'bidder_name', 'agent_name',
                   'eval_method', 'award_announce_time', 'winner', 'award_amount',
                   'region', 'regulator', 'platform'):
        val = request.form.get(field, '').strip()
        if val:
            info_overrides[field] = val

    from celery_app import celery
    celery.send_task(
        'clearance_task',
        args=[file_data, resolved, tender_text, tender_name, tender_spec,
              options, user_id, thread_id, task_id, project_id, info_overrides,
              open_info, eval_criteria],
        task_id=task_id,
    )

    return ok({'task_id': task_id, 'file_count': len(file_data) + len(resolved)})


@clearance_bp.route('/preview_criteria', methods=['POST'])
def preview_criteria():
    """从招标文件提取评审标准供前端预览（决定 2：预览不合格可手动编辑或上传表格）。

    Request: tender_file_id (pre-uploaded) or tender_file (direct upload).
    Returns: {budget_price, plan_open_time, eval_method, score_points,
              objective_rules, confidence, error}
    """
    if session.get('consent_value', 0) != 1:
        return err("请先登录", "AUTH_REQUIRED", 401)
    user_id = get_user_id()
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)

    tender_text = None
    tender_fid = request.form.get('tender_file_id')
    if tender_fid:
        from app.services.file_store import resolve as resolve_file
        tinfo = resolve_file(tender_fid, user_id=user_id)
        if tinfo and os.path.exists(tinfo['abs_path']):
            from app.services.file_processing import extract_text_from_path
            ttext, _ = extract_text_from_path(tinfo['abs_path'], tinfo['filename'])
            if ttext and not ttext.startswith("["):
                tender_text = ttext
    tender_file = request.files.get('tender_file')
    if not tender_text and tender_file and tender_file.filename:
        from app.services.file_processing import extract_text_from_file
        ttext, _ = extract_text_from_file(tender_file)
        if ttext and not ttext.startswith("["):
            tender_text = ttext

    if not tender_text:
        return ok({'error': '未提供招标文件，无法提取评审标准'})

    from app.services.clearance_openinfo import extract_eval_criteria
    criteria = extract_eval_criteria(tender_text)
    return ok(criteria)


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
