"""Login brute-force guard (FIX-2026-09-28-068).

Three-layer defence for ``/login``:

  Layer 1  ``@limiter.limit("5/minute", key_func=login_rate_key_func)``  (username+IP)
  Layer 2  cooldown gate — non-blocking, returns ``Retry-After`` seconds; never sleeps
  Layer 3  ``username+IP`` hard lock for ``LOCK_TTL`` seconds

Redis-primary with an in-memory fallback (same shape as ``routes/credit.py``).
A Redis outage is a *degraded* (per-worker) mode and is logged as a warning — it is
NOT fail-closed and must never be described as such.

Keys: ``login_fail:{u}:{ip}`` / ``login_last:{u}:{ip}`` / ``login_lock:{u}:{ip}``,
all with TTL (``INCR`` + ``EXPIRE`` on first write) so nothing accumulates forever.
FIX-2026-10-04-QA-030: fail counters are scoped by IP too, so an attacker cannot
lock a victim account out across all IPs by spamming failures.
"""
import logging
import threading
import time

logger = logging.getLogger(__name__)

FAIL_TTL = 120          # seconds a failure counter / last-fail timestamp lives
LOCK_TTL = 900          # Layer 3 hard lock, 15 minutes
FAIL_THRESHOLD = 5      # username+IP failures before Layer 3 lock
COOLDOWN_START = 3      # failures before Layer 2 cooldown kicks in
COOLDOWN_CAP = 5        # exponent cap -> 2**5 = 32s

_mem = {}
_mem_lock = threading.Lock()


def login_rate_key_func():
    """flask-limiter key_func — MUST include the real client IP.

    Requires ``ProxyFix`` (``TRUST_PROXY=1``) to see the client IP rather than the
    nginx container IP. The request body may be unparsable; we then fall back to
    IP-only. Never returns an empty/constant key (that would share one bucket).
    """
    from flask import request
    from flask_limiter.util import get_remote_address
    try:
        data = request.get_json(silent=True) or {}
        username = (data.get('username') or '').strip().lower()
    except Exception as e:
        logger.debug(f"login rate key: unparsable body, IP-only fallback: {e}")
        username = ''
    ip = get_remote_address() or 'unknown'
    return f"login:{username}:{ip}" if username else f"login:{ip}"


def _u(username):
    return (username or '').strip().lower()


def _redis():
    try:
        from app.services.redis_client import get_redis
        r = get_redis(decode_responses=True)
        if r is None:
            logger.warning("login_guard: Redis unavailable — in-memory fallback (per-worker, degraded)")
        return r
    except Exception as e:
        logger.warning(f"login_guard: Redis error — in-memory fallback: {e}")
        return None


def _mem_purge(now):
    # FIX-2026-10-04-QA-030: honour each key's own TTL (fail counters 120s, locks
    # 900s) instead of a single max() horizon.
    for k in list(_mem.keys()):
        ttl = LOCK_TTL if k.startswith('login_lock:') else FAIL_TTL
        if _mem[k]['ts'] < now - ttl:
            del _mem[k]


def _cooldown_for(count):
    """Layer 2 delay for a given failure count: 0 before threshold, then 1,2,4,8,16,32."""
    if count < COOLDOWN_START:
        return 0
    n = min(count - COOLDOWN_START, COOLDOWN_CAP)
    return 2 ** n


def check_login_gate(username, ip):
    """Return seconds the caller must wait (0 = allow). Never sleeps."""
    u = _u(username)
    if not u:
        return 0
    now = time.time()
    lock_key = f"login_lock:{u}:{ip}"
    fail_key = f"login_fail:{u}:{ip}"
    last_key = f"login_last:{u}:{ip}"

    r = _redis()
    if r is not None:
        try:
            ttl = r.ttl(lock_key)
            if ttl and ttl > 0:
                return int(ttl)
            raw = r.get(fail_key)
            count = int(raw) if raw else 0
            raw_ts = r.get(last_key)
            last = float(raw_ts) if raw_ts else 0.0
            cd = _cooldown_for(count)
            if cd and (now - last) < cd:
                return int(cd - (now - last)) + 1
            return 0
        except Exception as e:
            logger.warning(f"login_guard: gate Redis failed, in-memory fallback: {e}")

    with _mem_lock:
        _mem_purge(now)
        lock = _mem.get(lock_key)
        if lock:
            return max(1, int(LOCK_TTL - (now - lock['ts'])))
        entry = _mem.get(fail_key)
        count = entry['count'] if entry else 0
        last = entry['last'] if entry else 0.0
        cd = _cooldown_for(count)
        if cd and (now - last) < cd:
            return int(cd - (now - last)) + 1
    return 0


def record_login_failure(username, ip):
    """Count a failed credential check; escalate to Layer 3 lock at FAIL_THRESHOLD."""
    u = _u(username)
    if not u:
        return
    now = time.time()
    fail_key = f"login_fail:{u}:{ip}"
    last_key = f"login_last:{u}:{ip}"
    lock_key = f"login_lock:{u}:{ip}"

    r = _redis()
    if r is not None:
        try:
            count = r.incr(fail_key)
            if count == 1:
                r.expire(fail_key, FAIL_TTL)
            r.set(last_key, now, ex=FAIL_TTL)
            if count >= FAIL_THRESHOLD:
                r.set(lock_key, '1', ex=LOCK_TTL)
                logger.warning(f"login_guard: locked {u}@{ip} for {LOCK_TTL}s after {count} failures")
            return
        except Exception as e:
            logger.warning(f"login_guard: failure record Redis failed, in-memory fallback: {e}")

    with _mem_lock:
        entry = _mem.get(fail_key)
        if entry is None:
            entry = {'count': 0, 'last': now, 'ts': now}
            _mem[fail_key] = entry
        entry['count'] += 1
        entry['last'] = now
        entry['ts'] = now
        if entry['count'] >= FAIL_THRESHOLD:
            _mem[lock_key] = {'ts': now}
            logger.warning(f"login_guard: (memory) locked {u}@{ip} after {entry['count']} failures")


def reset_login(username, ip):
    """Clear all counters for this (username, ip) on a successful login."""
    u = _u(username)
    if not u:
        return
    r = _redis()
    if r is not None:
        try:
            r.delete(f"login_fail:{u}:{ip}", f"login_last:{u}:{ip}", f"login_lock:{u}:{ip}")
            return
        except Exception as e:
            logger.warning(f"login_guard: reset Redis failed, in-memory fallback: {e}")
    with _mem_lock:
        _mem.pop(f"login_fail:{u}:{ip}", None)
        _mem.pop(f"login_last:{u}:{ip}", None)
        _mem.pop(f"login_lock:{u}:{ip}", None)
