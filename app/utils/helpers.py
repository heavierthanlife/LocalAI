"""Shared utility functions used across the application."""
import re
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

BEIJING_TZ = timezone(timedelta(hours=8))


def beijing_now() -> str:
    """Return current Beijing time as formatted string."""
    return datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def safe_error_response(user_message="处理文件时出错，请检查文件格式或稍后重试。", log_error=None):
    """Return a standardized error string for file processing failures."""
    if log_error:
        logger.error(log_error, exc_info=True)
    return f"[错误] {user_message}"


def split_thinking_answer(text: str) -> tuple:
    """Split AI response into thinking and answer parts using common delimiters."""
    patterns = [
        r'【思考】(.*?)【回答】',
        r'思考：(.*?)回答：',
        r'<思考>(.*?)</思考>',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            answer = re.sub(pat, '', text, flags=re.DOTALL).strip()
            return thinking, answer
    return None, text
