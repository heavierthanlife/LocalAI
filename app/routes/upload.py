"""Upload blueprint — stream files to disk (large-file friendly).

Endpoints:
    POST /stream_upload   — stream one file to user_files, return file_id

Note: /upload_file already exists in chat.py (chat attachments) — do not
reuse that path here.
"""
import logging

from flask import Blueprint, request, session

from app.config import allowed_file
from app.utils.helpers import err, ok
from app.services.session_manager import get_user_id
from app.services import file_store

logger = logging.getLogger(__name__)

upload_bp = Blueprint('upload', __name__)


@upload_bp.before_request
def _raise_body_limit():
    # 流式上传：单文件请求体上限对齐 10GB 配额（留余量）。
    # 全局 MAX_CONTENT_LENGTH 只有 50MB（聊天附件等旧路径用）；
    # 实际磁盘占用由 file_store.check_quota 在落盘后校验，内存恒定。
    request.max_content_length = 11 * 1024 * 1024 * 1024


@upload_bp.route('/stream_upload', methods=['POST'])
def upload_file():
    """Stream-upload a single file. Returns {file_id, filename, size}.

    The body is streamed to disk in 8MB chunks; memory stays constant no
    matter how large the file is.
    """
    if session.get('consent_value', 0) != 1:
        return err("请先登录", "AUTH_REQUIRED", 401)
    user_id = get_user_id()
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)

    f = request.files.get('file')
    if not f or not f.filename:
        return err("缺少文件", "VALIDATION_ERROR", 400)
    if not allowed_file(f.filename):
        return err(f"不支持的文件类型: {f.filename}", "VALIDATION_ERROR", 400)

    try:
        result = file_store.save_stream(f, user_id, thread_id=request.form.get('thread_id') or None)
    except file_store.QuotaExceeded as e:
        limit_gb = e.limit_bytes / (1024 * 1024 * 1024)
        used_gb = e.used_bytes / (1024 * 1024 * 1024)
        return err(f"存储空间不足：已用 {used_gb:.1f}GB / 上限 {limit_gb:.0f}GB，请清理旧文件后再上传",
                   "QUOTA_EXCEEDED", 413)
    except Exception as e:
        logger.error(f"upload_file failed: {e}", exc_info=True)
        return err("上传失败", "SERVER_ERROR", 500)

    return ok({'file_id': result['file_id'], 'filename': result['filename'], 'size': result['size']})
