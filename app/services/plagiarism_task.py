"""Async plagiarism comparison task (FIX-2026-09-09-025).

大文件（数百 MB）在 web worker 内同步对比会 OOM/超时（502）。改为：
预上传 file_ids → 解析磁盘路径 → Celery 任务分页提取 + 对比 → TaskBus 存结果。
web 立即返回 task_id，前端轮询状态。
"""
import logging

from celery_app import celery as _celery_app

logger = logging.getLogger(__name__)


@_celery_app.task(bind=True, name='plagiarism_task', max_retries=0,
                  soft_time_limit=2400, time_limit=2700)
def run_plagiarism_async(self, task_id, docs, template_path=None):
    """docs: [{'abs_path','filename'}, {'abs_path','filename'}]; template_path optional."""
    from app.services.task_bus import TaskBus
    from app.services.file_processing import extract_text_from_path
    from app.services.plagiarism_detector import detect_plagiarism

    bus = TaskBus(task_id, 'plagiarism', '两文件剽窃对比')
    bus.start()
    try:
        texts, names = [], []
        for i, d in enumerate(docs[:2]):
            bus.progress(10 + i * 30, f'正在提取文本 ({i + 1}/2): {d.get("filename", "")}')
            text, _ = extract_text_from_path(d['abs_path'], d.get('filename', ''))
            if not text or text.startswith('['):
                bus.fail(f'无法提取文本: {d.get("filename", "")}')
                return
            texts.append(text)
            names.append(d.get('filename', ''))

        template_text = None
        if template_path:
            try:
                template_text, _ = extract_text_from_path(template_path, '')
            except Exception:
                template_text = None

        bus.progress(75, '正在进行剽窃比对...')
        report = detect_plagiarism(
            texts[0], texts[1],
            template_text=template_text,
            filename_a=names[0], filename_b=names[1],
        )
        bus.complete(report)
    except Exception as e:
        logger.error(f"plagiarism_task failed: {e}", exc_info=True)
        bus.fail(str(e)[:200])
