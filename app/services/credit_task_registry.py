"""Cross-worker credit-check task registry (Redis-backed).

Why: the in-memory ``credit_tasks`` dict in ``app.globals`` is process-local.
Under gunicorn's multiple gevent workers, ``start_credit_check`` may land on
worker A while ``credit_check_status``/``get_captcha_image``/``solve_captcha``
hit workers B/C/D — the task state would be invisible → spurious 404s and
CAPTCHA handshake deadlocks.

This module stores each task's mutable state in a Redis hash
(``credit_task:{task_id}``) so every worker reads the same state. The
``captcha_image`` bytes are base64-encoded inside Redis and decoded on read.

If Redis is unavailable the registry degrades to an in-process dict so the
feature keeps working (single-worker or dev setups).
"""
import base64
import json
import logging
import time

logger = logging.getLogger(__name__)

_PREFIX = 'credit_task:'
_TTL = 2 * 3600  # 2 hours — long enough for a multi-company check

_in_memory = {}
_lock = None  # lazily imported to avoid heavy module pulls at import time

# Fields whose value is binary (bytes) and must be base64-encoded in Redis.
_BINARY_FIELDS = ('captcha_image',)


def _get_redis():
    try:
        from app.services.redis_client import get_redis
        r = get_redis(decode_responses=True)
        return r if r else None
    except Exception:
        return None


def _get_lock():
    global _lock
    if _lock is None:
        import threading
        _lock = threading.RLock()
    return _lock


def _encode(data):
    """Encode a task dict for Redis storage (bytes → base64 str, scalars → str)."""
    out = {}
    for k, v in data.items():
        if isinstance(v, bytes):
            out[k] = 'B64:' + base64.b64encode(v).decode('ascii')
        elif isinstance(v, bool):
            out[k] = 'BOOL:' + ('1' if v else '0')
        elif v is None:
            out[k] = ''
        else:
            out[k] = str(v)
    return out


def _decode(raw):
    """Decode a Redis hash back into a task dict."""
    data = {}
    for k, v in raw.items():
        if k in _BINARY_FIELDS and isinstance(v, str) and v.startswith('B64:'):
            try:
                data[k] = base64.b64decode(v[4:])
            except Exception:
                data[k] = None
        elif isinstance(v, str) and v.startswith('BOOL:'):
            data[k] = v[5:] == '1'
        elif isinstance(v, str) and k == 'progress':
            try:
                data[k] = int(v)
            except (TypeError, ValueError):
                data[k] = 0
        else:
            data[k] = v
    return data


def set_task(task_id: str, data: dict):
    """Create or fully replace a task's state."""
    with _get_lock():
        r = _get_redis()
        if r is not None:
            pipe = r.pipeline()
            pipe.delete(_PREFIX + task_id)
            if data:
                pipe.hset(_PREFIX + task_id, mapping=_encode(data))
            pipe.expire(_PREFIX + task_id, _TTL)
            pipe.execute()
        else:
            _in_memory[task_id] = dict(data)


def get_task(task_id: str):
    """Return the task state dict, or None if the task doesn't exist."""
    with _get_lock():
        r = _get_redis()
        if r is not None:
            raw = r.hgetall(_PREFIX + task_id)
            if not raw:
                return None
            data = _decode(raw)
            data.setdefault('status', 'running')
            return data
        return _in_memory.get(task_id)


def patch_task(task_id: str, **fields):
    """Update one or more fields of a task's state. No-op if task missing."""
    with _get_lock():
        r = _get_redis()
        if r is not None:
            if r.exists(_PREFIX + task_id):
                r.hset(_PREFIX + task_id, mapping=_encode(fields))
                r.expire(_PREFIX + task_id, _TTL)
        else:
            if task_id in _in_memory:
                _in_memory[task_id].update(fields)


def task_exists(task_id: str) -> bool:
    with _get_lock():
        r = _get_redis()
        if r is not None:
            return bool(r.exists(_PREFIX + task_id))
        return task_id in _in_memory


def delete_task(task_id: str):
    with _get_lock():
        r = _get_redis()
        if r is not None:
            r.delete(_PREFIX + task_id)
        else:
            _in_memory.pop(task_id, None)


def list_task_ids() -> list:
    """Return all live credit task IDs (Redis keys or in-memory fallback)."""
    with _get_lock():
        r = _get_redis()
        if r is not None:
            keys = r.keys(_PREFIX + '*')
            return [k[len(_PREFIX):] for k in keys]
        return list(_in_memory.keys())
