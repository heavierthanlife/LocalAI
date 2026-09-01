"""LangGraph Agent and tools for the AI chat system."""
import os
import json
import hashlib
import asyncio
import threading
import logging
from datetime import datetime, timezone, timedelta

import requests
import aiosqlite
from flask import session

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..config import DATA_DIR, logger as base_logger
from .. import globals as g

logger = logging.getLogger(__name__)

BEIJING_TZ = timezone(timedelta(hours=8))

# System prompt
AGENT_SYSTEM_PROMPT = g.AGENT_SYSTEM_PROMPT

# ---------- Search Cache ----------
SEARCH_CACHE_DIR = os.path.join(DATA_DIR, "search_cache")
SEARCH_CACHE_FILE = os.path.join(SEARCH_CACHE_DIR, "search_cache.json")
os.makedirs(SEARCH_CACHE_DIR, exist_ok=True)

# Default TTL: 72 hours (in seconds). Admin can override via API.
DEFAULT_CACHE_TTL_SECONDS = 72 * 3600
_cache_ttl = DEFAULT_CACHE_TTL_SECONDS
_search_cache: dict = {}          # {norm_query: (timestamp, result_str)}
_cache_lock = threading.Lock()
_last_search_cache_hit: bool = False   # per-tool-call flag for training data


def _normalize_query(query: str) -> str:
    """Normalize query for cache key: lowercase + strip to catch near-duplicates."""
    return query.strip().lower()


def _load_cache_from_disk():
    """Load persisted cache from disk."""
    global _search_cache
    if os.path.exists(SEARCH_CACHE_FILE):
        try:
            with open(SEARCH_CACHE_FILE, 'r', encoding='utf-8') as f:
                _search_cache = json.load(f)
        except Exception:
            _search_cache = {}


def _save_cache_to_disk():
    """Persist cache to disk (called after writes)."""
    try:
        with open(SEARCH_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(_search_cache, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to persist search cache: {e}")


def _set_cache_ttl(ttl_seconds: int):
    """Set cache TTL in seconds. Called by admin API. Also syncs to runtime_config."""
    global _cache_ttl
    _cache_ttl = max(0, ttl_seconds)
    # Sync to runtime_config so the panel stays consistent
    try:
        from app.services.runtime_config import update as rc_update
        rc_update({"search_cache_ttl_hours": round(ttl_seconds / 3600, 1)})
    except Exception:
        pass


def _init_cache_ttl():
    """Initialize cache TTL from runtime config (with env override)."""
    global _cache_ttl
    env_val = os.getenv("SEARCH_CACHE_TTL_HOURS")
    if env_val:
        try:
            _cache_ttl = int(float(env_val) * 3600)
            return
        except ValueError:
            pass
    try:
        from app.services.runtime_config import get as rc_get
        hours = rc_get("search_cache_ttl_hours", 72)
        _cache_ttl = int(hours * 3600)
    except Exception:
        _cache_ttl = DEFAULT_CACHE_TTL_SECONDS


# Initialize TTL from config
_init_cache_ttl()


def get_cache_ttl() -> int:
    """Return current cache TTL in seconds."""
    return _cache_ttl


def get_cache_stats() -> dict:
    """Return cache stats for admin panel (reads from runtime_config for consistency)."""
    try:
        from app.services.runtime_config import get as rc_get
        ttl_hours = rc_get("search_cache_ttl_hours", round(_cache_ttl / 3600, 1))
    except Exception:
        ttl_hours = round(_cache_ttl / 3600, 1)
    return {
        'entries': len(_search_cache),
        'ttl_seconds': int(ttl_hours * 3600),
        'ttl_hours': ttl_hours,
        'cache_file': SEARCH_CACHE_FILE,
    }


def clear_search_cache():
    """Admin: wipe entire cache."""
    global _search_cache
    with _cache_lock:
        _search_cache = {}
        _save_cache_to_disk()


def get_last_search_cache_hit() -> bool:
    """Called by chat.py after agent invocation to see if the last bocha_search was a cache hit."""
    return _last_search_cache_hit


# Load persisted cache at startup
_load_cache_from_disk()


# ---------- Tools ----------

@tool(description="Get current date and time in Beijing time (UTC+8).")
def get_date() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")


BOCHA_API_KEY = os.getenv("BOCHA_API_KEY")
BOCHA_URL = "https://api.bochaai.com/v1/web-search"


@tool(description="Search the web using Bocha. Use for up-to-date information.")
def bocha_search(query: str) -> str:
    global _last_search_cache_hit

    # ── Cache lookup ──
    norm = _normalize_query(query)
    with _cache_lock:
        entry = _search_cache.get(norm)
        if entry and _cache_ttl > 0:
            ts, cached_result = entry
            if ts + _cache_ttl > int(datetime.now(timezone.utc).timestamp()):
                _last_search_cache_hit = True
                return cached_result
            # Expired — remove
            del _search_cache[norm]
            _save_cache_to_disk()
        _last_search_cache_hit = False

    # ── Live API call ──
    headers = {"Authorization": f"Bearer {BOCHA_API_KEY}", "Content-Type": "application/json"}
    payload = json.dumps({"query": query, "summary": True, "freshness": "noLimit", "count": 10})
    try:
        response = requests.post(BOCHA_URL, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        webpages = data.get('data', {}).get('webPages', {}).get('value', [])
        if not webpages:
            result = "No search results found."
        else:
            formatted = []
            for idx, page in enumerate(webpages[:10], 1):
                title = page.get('name', 'No title')
                snippet = page.get('snippet', 'No snippet')
                date_pub = page.get('datePublished', 'Unknown date')
                url = page.get('url', 'No URL')
                formatted.append(f"{idx}. **{title}**\n   Published: {date_pub}\n   Summary: {snippet}\n   Source: {url}\n")
            result = "\n".join(formatted)
    except Exception as e:
        return f"Search failed: {str(e)}"

    # ── Headroom compression on search results (saves cache space + tokens) ──
    try:
        from app.services.runtime_config import get as rc_get
        if rc_get("headroom_enabled", True) and len(result) > 500:
            from app.utils.headroom_utils import compress_search_results
            compressed = compress_search_results(result)
            if compressed and len(compressed) < len(result):
                logger.info(f"Headroom search: {len(result)} -> {len(compressed)} chars")
                result = compressed
    except Exception:
        pass

    # ── Cache write (miss → store) ──
    if _cache_ttl > 0:
        with _cache_lock:
            _search_cache[norm] = (int(datetime.now(timezone.utc).timestamp()), result)
            _save_cache_to_disk()

    return result


# ---------- Agent management ----------

def get_agent(max_tokens=None):
    from .. import globals as g

    if max_tokens is None:
        max_tokens = session.get('max_tokens', 1600)
    if g._agent is not None and g._current_max_tokens == max_tokens:
        return g._agent
    with g._agent_lock:
        if g._agent is not None and g._current_max_tokens == max_tokens:
            return g._agent

        # Check if admin selected a different model via runtime_config
        provider_id = None
        model_id = None  # FIX-016: default resolved from free-model catalog
        try:
            from app.services.runtime_config import get as rc_get
            rc_provider = rc_get('active_llm_provider', '')
            rc_model = rc_get('active_llm_model', '')
            if rc_provider and rc_provider != 'auto' and rc_model and rc_model != 'auto':
                provider_id = rc_provider
                model_id = rc_model
        except Exception:
            pass

        if provider_id:
            # Use admin-selected provider+model via create_chat_model
            from app.services.llm_provider import create_chat_model
            llm = create_chat_model(
                provider_id=provider_id,
                model=model_id,
                streaming=False,
                temperature=0.7,
                max_tokens=max_tokens,
                timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            )
            logger.info(f"Agent using admin-selected model: {provider_id}/{model_id}")
        else:
            # FIX-016: default to free-model catalog (OpenRouter :free → NVIDIA NIM)
            from app.services.llm_provider import create_chat_model
            from app.services.llm_provider import get_active_provider as _active_prov
            try:
                from app.services.llm_catalog import get_default_model
                auto_provider = _active_prov()
                auto_model = get_default_model(auto_provider) if auto_provider else None
            except Exception:
                auto_provider, auto_model = None, None
            try:
                llm = create_chat_model(
                    provider_id=auto_provider,
                    model=auto_model,
                    streaming=False,
                    temperature=0.7,
                    max_tokens=max_tokens,
                    timeout=int(os.getenv("LLM_TIMEOUT", "120")),
                )
            except Exception as e:
                logger.warning(f"Agent catalog default failed ({e}); using provider auto-detect")
                llm = create_chat_model(
                    streaming=False, temperature=0.7,
                    max_tokens=max_tokens,
                    timeout=int(os.getenv("LLM_TIMEOUT", "120")),
                )
            logger.info(f"Agent using catalog default: {auto_provider}/{auto_model}")
        if g._async_checkpointer is None:
            with g._async_checkpointer_lock:
                if g._async_checkpointer is None:
                    _init_async_checkpointer()
        g._agent = create_agent(
            model=llm,
            tools=[get_date, bocha_search],
            system_prompt=AGENT_SYSTEM_PROMPT,
            checkpointer=g._async_checkpointer
        )
        g._current_max_tokens = max_tokens
        logger.info(f"Agent reinitialized with DeepSeek model, max_tokens={max_tokens}")
        return g._agent


def _init_async_checkpointer():
    """Initialize async SQLite checkpointer for LangGraph."""
    from .. import globals as g

    g._async_loop = asyncio.new_event_loop()

    async def _create():
        g._async_conn = await aiosqlite.connect(str(DATA_DIR / "checkpoints.db"))
        return AsyncSqliteSaver(g._async_conn)

    g._async_checkpointer = g._async_loop.run_until_complete(_create())

    def _run_loop():
        asyncio.set_event_loop(g._async_loop)
        g._async_loop.run_forever()

    t = threading.Thread(target=_run_loop, daemon=True)
    t.start()
    logger.info("AsyncSqliteSaver initialized.")


def set_max_tokens(tokens):
    """Set max_tokens and invalidate agent cache."""
    from .. import globals as g
    tokens = max(100, min(4800, tokens))
    session['max_tokens'] = tokens
    with g._agent_lock:
        g._agent = None
    return tokens
