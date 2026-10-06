"""Shared utility functions used across the application."""
import re
import logging
from datetime import datetime, timezone, timedelta

from flask import jsonify

logger = logging.getLogger(__name__)

BEIJING_TZ = timezone(timedelta(hours=8))


def ok(data=None, message=None, status=200):
    """Return a standardized success JSON response.

    ok(data, message, status) → {success:true, message, ...data}
    If data is a dict, its keys are flat-merged into the response.
    """
    response = {"success": True}
    if message:
        response["message"] = message
    if data is not None:
        if isinstance(data, dict):
            response.update(data)
        else:
            response["data"] = data
    return jsonify(response), status


def err(error, code="ERROR", status=400):
    """Return a standardized error JSON response.

    err(error, code, status) → {success:false, error, code}
    """
    return jsonify({"success": False, "error": error, "code": code}), status


def beijing_now() -> str:
    """Return current Beijing time as formatted string."""
    return datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def task_owner_ok(meta, user_id) -> bool:
    """Return True if ``user_id`` may read the task described by ``meta``.

    Ownership is recorded in task meta at register time (``user_id``). Legacy
    or in-flight tasks created before ownership tracking carry no ``user_id``;
    those are allowed through with a warning (Redis TTL expires them within 7
    days). Malformed meta fails closed.
    """
    if not isinstance(meta, dict):
        logger.warning("task_owner_ok: malformed task meta; denying access")
        return False
    owner = meta.get('user_id')
    if 'user_id' not in meta:
        logger.warning("Task meta missing user_id; allowing access (legacy task)")
        return True
    if not owner:
        # FIX-2026-10-04-QA-030: an explicitly empty owner (register_queued with
        # user_id='') must NOT fall into the legacy-allow branch — that silently
        # bypassed FIX-080's fail-closed owner check. Only a truly absent key is
        # treated as a pre-ownership-tracked legacy task.
        logger.warning("Task meta has empty user_id; denying access (fail-closed)")
        return False
    return str(user_id) == str(owner)


def load_task_for(task_id, user_id):
    """Resolve a TaskBus task and check ownership in one place (FIX-066).

    Returns ``(meta, status)`` with ``status`` in ``{'ok', 'missing', 'forbidden'}``:

    - ``ok``        → ``meta`` is a dict owned by ``user_id``; caller may proceed.
    - ``missing``   → no meta (expired / legacy / Redis unavailable); each caller
                      decides 404 vs. allow. Note: compliance's
                      ``_task_forbidden`` maps *both* ``missing`` and
                      ``forbidden`` to deny (fail-closed, FIX-080), so disk-
                      persisted results do NOT outlive the Redis TTL either.
    - ``forbidden`` → a different owner, or the lookup raised → deny (fail-closed).

    Fail-closed by design: a TaskBus/Redis exception is treated as ``forbidden``
    so the owner-check path never turns into a 500.
    """
    try:
        from app.services.task_bus import TaskBus
        meta = TaskBus.get(task_id)
    except Exception as e:
        # Fail closed (FIX-063): if ownership cannot be determined, deny.
        logger.warning(f"load_task_for: ownership lookup failed, denying access: {e}")
        return None, 'forbidden'
    if not meta:
        return None, 'missing'
    if not task_owner_ok(meta, user_id):
        return meta, 'forbidden'
    return meta, 'ok'


def safe_error_response(user_message="处理文件时出错，请检查文件格式或稍后重试。", log_error=None):
    """Return a standardized error string for file processing failures."""
    if log_error:
        logger.error(log_error, exc_info=True)
    return f"[错误] {user_message}"


def split_thinking_answer(text: str) -> tuple:
    """Split AI response into thinking and answer parts.

    Supports:
      - MiMo dual blocks:  【思考】A...【思考】B...【回答】C...
      - Standard single:   【思考】A...【回答】C...
      - 段错位置暴露:      …text…【思考】A...【回答】C...
    The entire text before the last 【回答】 is treated as thinking, so
    multi-block MiMo output is correctly separated.
    """
    if not text:
        return None, sanitize_response(text)

    # ── MiMo / multi-block (≥2 【思考】) ──
    if text.count('【思考】') >= 2:
        idx = text.rfind('【回答】')
        if idx >= 0:
            thinking = text[:idx].replace('【思考】', '').strip()
            thinking = re.sub(r'\n{3,}', '\n\n', thinking)
            answer = text[idx + len('【回答】'):].strip()
            return sanitize_response(thinking), sanitize_response(answer)
        return None, sanitize_response(text)

    # ── Standard single-block formats ──
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
            return sanitize_response(thinking), sanitize_response(answer)
    return None, sanitize_response(text)


# ── Response sanitization: never surface internal tool/LLM artifacts to users ──
# NOTE: patterns are mirrored in static/js/chat.js `_sanitizeResponse()` so the
# live SSE stream is cleaned too (split_thinking_answer only covers the stored
# copy). Keep the two in sync — tests/test_regression.py asserts the two are
# behaviourally equivalent, not merely textually similar.
#
# The tool-name gap is `[^\n\r]*?` rather than `\S+`: real LangChain output names
# tools with spaces ("Error: get current date is not a valid tool, ...") and `\S+`
# cannot span that space, so the whole artifact used to survive. The gap is
# deliberately unbounded within the line — a bounded `{0,80}?` still let a long
# tool name through — and stays lazy so "is not a valid tool" anchors the match.
#
# The tail is line-greedy ON PURPOSE. Restricting it to the `, try one of [...]`
# continuation (an earlier attempt) made other forms survive instead, e.g.
# "Error: X is not a valid tool, try 'Y' instead" — trading a cosmetic over-strip
# for a real leak. Leaking internal tool errors is the bug this sanitizer exists to
# prevent, so the tail stays greedy: R1-F5 (legitimate same-line text after the
# artifact is dropped too) is accepted as the lesser cost, not fixed.
_INVALID_TOOL_RE = re.compile(
    r"Error:\s*[^\n\r]*?is not a valid tool[^\n\r]*",
    re.IGNORECASE,
)
_PROMPT_TEMPLATE_RE = re.compile(
    r"Here is the JSON for a function call with its proper arguments[^\n\r]*",
    re.IGNORECASE,
)
_KNOWN_LEAK_PATTERNS = (
    _INVALID_TOOL_RE,
    _PROMPT_TEMPLATE_RE,
)

# Explicit trim set identical to JavaScript's String.prototype.trim()
# (WhiteSpace + LineTerminator). Bare str.strip() additionally removes
# \x1c-\x1f and \x85, which made the two copies diverge on such payloads.
_TRIM_CHARS = (
    " \t\n\r\f\v\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007"
    "\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def sanitize_response(text: str) -> str:
    """Strip internal tool-calling artifacts and prompt-template echoes from
    AI responses before they reach the user or are persisted.

    Covers: 'Error: X is not a valid tool, try one of [...]' (LangChain tool
    executor output) and function-calling template text echoed by models that
    lack proper tool-calling support.
    """
    if not text:
        return text
    cleaned = text
    for pat in _KNOWN_LEAK_PATTERNS:
        cleaned = pat.sub("", cleaned)
    return cleaned.strip(_TRIM_CHARS)
