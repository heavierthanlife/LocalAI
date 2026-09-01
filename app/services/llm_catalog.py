"""LLM free-model catalog (FIX-016).

Daily-refreshed list of free models from OpenRouter (:free pool) and NVIDIA NIM.

Strategy:
  - OpenRouter: GET https://openrouter.ai/api/v1/models → keep models whose id
    ends with ':free' OR have zero prompt+completion pricing.
  - NVIDIA: GET https://integrate.api.nvidia.com/v1/models → keep all (NIM free
    tier serves the catalog; rate limits apply but models are usable).
  - Fixed whitelist + dynamic fallback: the app prefers whitelisted models
    (stable behavior); if a whitelisted model drops out of the live pool, fall
    back to the first free model returned by the refresh.

Persistence: data/llm_catalog.json  (refreshed daily by Celery beat /
APScheduler / startup catch-up).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Fixed whitelist — prefer these for stable behavior across daily refreshes.
# Keyed by provider. If a whitelisted model is missing from the live pool,
# get_free_models falls back to the pool's first free model.
WHITELIST = {
    'openrouter': [
        'nvidia/nemotron-3-ultra-550b-a55b:free',   # text mainstay (1M ctx)
        'z-ai/glm-5.2:free',                        # text backup
        'thinkingmachines/inkling:free',            # long-context backup
        'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',  # VL-capable
    ],
    'nvidia': [
        'nvidia/nemotron-3-ultra-550b-a55b',        # text mainstay
        'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',  # VL-capable
        'nvidia/nemotron-3-super-120b-a12b',
    ],
}

CATALOG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "llm_catalog.json"
CATALOG_TTL_SECONDS = 24 * 3600  # 1 day


def _utc() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _load_local() -> Optional[dict]:
    try:
        if CATALOG_PATH.exists():
            return json.loads(CATALOG_PATH.read_text(encoding='utf-8'))
    except Exception as e:
        logger.warning(f"llm_catalog local load failed: {e}")
    return None


def _save_local(catalog: dict):
    try:
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as e:
        logger.warning(f"llm_catalog save failed: {e}")


def _fetch_openrouter() -> list[dict]:
    import urllib.request
    url = 'https://openrouter.ai/api/v1/models'
    req = urllib.request.Request(url, headers={'Accept': 'application/json', 'User-Agent': 'localai-catalog/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode('utf-8'))
    models = []
    for m in payload.get('data', []):
        mid = m.get('id', '')
        pricing = m.get('pricing', {}) or {}
        def _z(v):
            try:
                return float(v) if v else 0.0
            except (TypeError, ValueError):
                return 0.0
        prompt_cost = _z(pricing.get('prompt'))
        comp_cost = _z(pricing.get('completion'))
        is_free = mid.endswith(':free') or (prompt_cost == 0.0 and comp_cost == 0.0)
        if is_free:
            models.append({
                'id': mid,
                'context_length': m.get('context_length'),
                'reasoning': bool(m.get('reasoning')),
                'provider': 'openrouter',
            })
    return models


def _fetch_nvidia() -> list[dict]:
    import urllib.request
    url = 'https://integrate.api.nvidia.com/v1/models'
    req = urllib.request.Request(url, headers={'Accept': 'application/json', 'User-Agent': 'localai-catalog/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        logger.warning(f"NVIDIA /models fetch failed: {e}")
        return []
    models = []
    for m in payload.get('data', []):
        mid = m.get('id', '')
        if mid:
            models.append({
                'id': mid,
                'context_length': None,
                'reasoning': False,
                'provider': 'nvidia',
            })
    return models


def refresh_catalog() -> dict:
    """Fetch free model lists from both providers and persist. Returns catalog dict."""
    catalog = {'updated_at': _utc(), 'models': [], 'whitelist': WHITELIST}
    try:
        catalog['models'].extend(_fetch_openrouter())
    except Exception as e:
        logger.warning(f"OpenRouter catalog refresh failed: {e}")
    try:
        catalog['models'].extend(_fetch_nvidia())
    except Exception as e:
        logger.warning(f"NVIDIA catalog refresh failed: {e}")
    _save_local(catalog)
    logger.info(f"LLM catalog refreshed: {len(catalog['models'])} free models")
    return catalog


def get_catalog(force_refresh: bool = False) -> dict:
    """Return the catalog, refreshing if stale (>TTL) or missing."""
    local = _load_local()
    if force_refresh or not local or _is_stale(local):
        local = refresh_catalog()
    return local or {'updated_at': _utc(), 'models': [], 'whitelist': WHITELIST}


def _is_stale(catalog: dict) -> bool:
    try:
        ts = catalog.get('updated_at', '')
        updated = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - updated).total_seconds() > CATALOG_TTL_SECONDS
    except Exception:
        return True


def get_free_models(provider: str, max_results: int = 20) -> list[dict]:
    """Return free models for a provider, preferring the whitelist then the pool."""
    catalog = get_catalog()
    pool = [m for m in catalog.get('models', []) if m.get('provider') == provider]
    if not pool:
        return []

    whitelisted = WHITELIST.get(provider, [])
    # 1. whitelist models still present in the live pool
    chosen = []
    for w in whitelisted:
        for m in pool:
            if m['id'] == w:
                chosen.append(m)
                break
    # 2. fill the rest from the pool (excluding already chosen)
    chosen_ids = {m['id'] for m in chosen}
    for m in pool:
        if len(chosen) >= max_results:
            break
        if m['id'] not in chosen_ids:
            chosen.append(m)
    return chosen[:max_results]


def get_default_model(provider: str) -> Optional[str]:
    """Return the primary model for a provider (whitelist first, else pool first)."""
    models = get_free_models(provider, max_results=1)
    if models:
        return models[0]['id']
    wl = WHITELIST.get(provider, [])
    return wl[0] if wl else None


def get_fallback_chain() -> list[dict]:
    """Build a provider:model fallback chain from the catalog (whitelist-first)."""
    chain = []
    for pid in ('openrouter', 'nvidia'):
        for m in get_free_models(pid, max_results=2):
            chain.append({'provider': pid, 'model': m['id']})
    if not chain:
        # Fallback to hardcoded whitelist if catalog unavailable
        for pid, wl in WHITELIST.items():
            if wl:
                chain.append({'provider': pid, 'model': wl[0]})
    return chain
