"""Incremental compliance check service (U9a).

Only checks modified sections of a bid document.
Caching layer: Redis with memory dict fallback.
Reuses compliance_checker.py for actual checking logic.
Debounce (300ms) is handled on the frontend side.
"""

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# In-memory cache fallback (when Redis unavailable)
_memory_cache: dict[str, tuple[dict, float]] = {}
_memory_cache_lock = threading.Lock()
_CACHE_TTL_SECONDS = 300


def _redis_client():
    """Get Redis client if available."""
    try:
        from app.services.redis_client import get_redis
        return get_redis()
    except Exception:
        return None


def _make_cache_key(bid_doc_name: str, section_id: str, section_hash: str) -> str:
    return f"incr_check:{hashlib.md5(f'{bid_doc_name}:{section_id}:{section_hash}'.encode()).hexdigest()[:16]}"


def _cache_get(key: str) -> dict | None:
    """Try Redis first, fall back to memory cache."""
    redis = _redis_client()
    if redis:
        try:
            val = redis.get(key)
            if val:
                return json.loads(val)
        except Exception:
            pass

    with _memory_cache_lock:
        entry = _memory_cache.get(key)
        if entry:
            result, expiry = entry
            if expiry > datetime.now(timezone.utc).timestamp():
                return result
            del _memory_cache[key]
    return None


def _cache_set(key: str, value: dict, ttl: int = _CACHE_TTL_SECONDS):
    """Set cache (Redis + memory)."""
    as_json = json.dumps(value, ensure_ascii=False, default=str)
    redis = _redis_client()
    if redis:
        try:
            redis.setex(key, ttl, as_json)
        except Exception:
            pass

    with _memory_cache_lock:
        _memory_cache[key] = (value, datetime.now(timezone.utc).timestamp() + ttl)


def _hash_section(section: dict) -> str:
    """Compute stable hash for a section."""
    content = json.dumps({
        'id': section.get('id', ''),
        'title': section.get('title', ''),
        'content': section.get('content', ''),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()


def incremental_check(bid_text: str, bid_doc_name: str,
                      rules: list[dict], changed_sections: list[dict],
                      use_ai: bool = True,
                      region_code: str = None) -> dict:
    """Run compliance check only on changed sections.

    Sections whose content hasn't changed since last check are served from cache.
    Only newly modified sections are re-checked with the LLM.

    Args:
        bid_text: full bid document text (for context when checking sections)
        bid_doc_name: document name for cache key
        rules: list of rule dicts (already extracted)
        changed_sections: list of {id, title, content} dicts
        use_ai: use AI for checking
        region_code: optional region code for law filtering

    Returns:
        dict with:
            results: merged results (cached + fresh)
            summary: {pass, warning, violation, critical}
            cache_hits: number of cached results
            fresh_checks: number of re-checked sections
    """
    from app.services.compliance_checker import ComplianceChecker

    cached = []
    to_check = []
    cache_hits = 0

    for section in changed_sections:
        section_hash = _hash_section(section)
        cache_key = _make_cache_key(bid_doc_name, section.get('id', ''), section_hash)
        cached_result = _cache_get(cache_key)

        if cached_result:
            cached.append(cached_result)
            cache_hits += 1
        else:
            to_check.append({
                'section': section,
                'cache_key': cache_key,
            })

    fresh_results = []
    if to_check:
        checker = ComplianceChecker()
        for item in to_check:
            section_text = item['section'].get('content', '')
            result = checker.check(
                section_text, rules, bid_doc_name,
                use_ai=use_ai, region_code=region_code,
            )
            # Store a lightweight version
            cached_entry = {
                'section_id': item['section'].get('id', ''),
                'section_title': item['section'].get('title', ''),
                'summary': result.get('summary', {}),
                'results': result.get('results', []),
            }
            _cache_set(item['cache_key'], cached_entry)
            fresh_results.append({
                'filename': f"{bid_doc_name}/{item['section'].get('title', item['section'].get('id', ''))}",
                **result,
                **cached_entry,
            })

    all_results = cached + fresh_results

    summary = {'pass': 0, 'warning': 0, 'violation': 0, 'critical': 0}
    for r in all_results:
        s = r.get('summary', {})
        for k in summary:
            summary[k] += s.get(k, 0)

    return {
        'results': all_results,
        'summary': summary,
        'cache_hits': cache_hits,
        'fresh_checks': len(fresh_results),
        'total_sections': len(changed_sections),
        'ai_used': use_ai and len(fresh_results) > 0,
        'checked_at': datetime.now(timezone.utc).isoformat(),
    }
