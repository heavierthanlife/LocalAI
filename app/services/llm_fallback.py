"""LLM provider fallback chain with circuit breaker.

Auto-fallback across ordered providers on failure.
Circuit breaker uses exponential backoff per (provider, model) pair.
"""
import os
import json
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CHAIN = [
    ("zhipu", "glm-4.5-air"),
    ("deepseek", "deepseek-v4-flash"),
    ("deepseek", "deepseek-v4-pro"),
]

_circuit_state: dict[str, dict] = {}
_cb_lock = threading.Lock()

# Thread-local to track which (provider, model) is currently active in this request
_active_local = threading.local()

_BASE_COOLDOWN = 30.0
_MAX_COOLDOWN = 300.0
_BACKOFF_MULTIPLIER = 3.0


def _circuit_key(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def is_degraded(provider: str, model: str) -> bool:
    with _cb_lock:
        state = _circuit_state.get(_circuit_key(provider, model))
        if state is None:
            return False
        if time.monotonic() >= state["cooldown_until"]:
            del _circuit_state[_circuit_key(provider, model)]
            return False
        return True


def mark_failure(provider: str, model: str):
    with _cb_lock:
        key = _circuit_key(provider, model)
        state = _circuit_state.get(key, {"failures": 0, "cooldown_until": 0})
        state["failures"] += 1
        wait = min(_BASE_COOLDOWN * (_BACKOFF_MULTIPLIER ** (state["failures"] - 1)), _MAX_COOLDOWN)
        state["cooldown_until"] = time.monotonic() + wait
        _circuit_state[key] = state
        logger.warning(
            "Circuit breaker: %s failed (%dx), cooling down for %.0fs",
            key, state["failures"], wait,
        )


def mark_success(provider: str, model: str):
    with _cb_lock:
        key = _circuit_key(provider, model)
        if key in _circuit_state:
            del _circuit_state[key]


def get_circuit_status() -> dict:
    with _cb_lock:
        now = time.monotonic()
        return {
            key: {
                "failures": s["failures"],
                "cooldown_remaining": max(0, round(s["cooldown_until"] - now, 1)),
            }
            for key, s in list(_circuit_state.items())
            if s["cooldown_until"] > now
        }


def get_fallback_chain() -> list[tuple[str, str]]:
    try:
        from app.services.runtime_config import get as rc_get
        chain = rc_get("llm_fallback_chain", None)
        if chain and isinstance(chain, list):
            validated = []
            for entry in chain:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    validated.append((str(entry[0]), str(entry[1])))
            if validated:
                return validated
    except Exception as e:
        logger.debug(f"Failed to read fallback chain from config: {e}")
    return list(DEFAULT_CHAIN)


def is_fallback_enabled() -> bool:
    try:
        from app.services.runtime_config import get as rc_get
        return bool(rc_get("llm_fallback_enabled", True))
    except Exception:
        return True


def get_cooldown_base() -> float:
    try:
        from app.services.runtime_config import get as rc_get
        return float(rc_get("llm_fallback_cooldown_seconds", _BASE_COOLDOWN))
    except Exception:
        return _BASE_COOLDOWN


def _set_active_provider(provider: str, model: str):
    _active_local.active = (provider, model)


def get_active_provider() -> Optional[tuple[str, str]]:
    return getattr(_active_local, 'active', None)


def classify_error(e: Exception) -> bool:
    """Return True if retriable (transient), False if non-retriable (permanent)."""
    msg = str(e).lower()
    exc_name = type(e).__name__.lower()
    non_retriable = (
        'authentication', 'auth_error', 'invalid_api_key', 'incorrect_api_key',
        'permission', 'not_found', 'invalid_model', 'bad_request',
        'invalidrequest', 'unauthorized', 'forbidden',
    )
    if any(s in msg for s in non_retriable) or any(s in exc_name for s in non_retriable):
        return False
    return True


def create_chat_model_with_fallback(
    provider_id: Optional[str] = None,
    model: Optional[str] = None,
    streaming: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1600,
    timeout: int = 120,
):
    global _BASE_COOLDOWN
    _BASE_COOLDOWN = get_cooldown_base()

    if not is_fallback_enabled():
        from app.services.llm_provider import _create_chat_model_direct
        return _create_chat_model_direct(
            provider_id=provider_id, model=model,
            streaming=streaming, temperature=temperature,
            max_tokens=max_tokens, timeout=timeout,
        )

    chain = get_fallback_chain()
    attempted: list[tuple[str, str]] = []

    if provider_id:
        from app.services.llm_provider import PROVIDER_CONFIG
        cfg = PROVIDER_CONFIG.get(provider_id)
        if cfg:
            resolved_model = model or cfg["default_model"]
            attempted.append((provider_id, resolved_model))

    for pid, mdl in chain:
        if (pid, mdl) not in attempted:
            attempted.append((pid, mdl))

    last_error: Optional[Exception] = None

    for pid, mdl in attempted:
        if is_degraded(pid, mdl):
            logger.info(f"Fallback skip degraded: {pid}/{mdl}")
            continue
        if not _has_api_key(pid):
            logger.info(f"Fallback skip missing API key: {pid}")
            continue

        try:
            from app.services.llm_provider import _create_chat_model_direct
            llm = _create_chat_model_direct(
                provider_id=pid, model=mdl,
                streaming=streaming, temperature=temperature,
                max_tokens=max_tokens, timeout=timeout,
            )
            mark_success(pid, mdl)
            logger.info(f"Fallback success: {pid}/{mdl}")
            _set_active_provider(pid, mdl)
            return llm
        except Exception as e:
            last_error = e
            logger.warning(f"Fallback failed: {pid}/{mdl}: {e}")
            if classify_error(e):
                mark_failure(pid, mdl)

    raise RuntimeError(
        f"All LLM providers in fallback chain failed. Last error: {last_error}"
    ) from last_error


def _has_api_key(provider_id: str) -> bool:
    from app.services.llm_provider import PROVIDER_CONFIG
    cfg = PROVIDER_CONFIG.get(provider_id)
    if not cfg:
        return False
    key = os.getenv(cfg["env_key"], "").strip()
    return bool(key)
