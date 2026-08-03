"""Unit tests for LLM fallback chain and circuit breaker (app/services/llm_fallback.py)."""
import time
import pytest
from unittest.mock import MagicMock


class TestCircuitBreaker:
    def test_degraded_returns_true_after_failure(self):
        from app.services.llm_fallback import mark_failure, is_degraded, _circuit_state
        _circuit_state.clear()
        mark_failure("zhipu", "glm-4.5-air")
        assert is_degraded("zhipu", "glm-4.5-air") is True

    def test_unknown_provider_not_degraded(self):
        from app.services.llm_fallback import is_degraded, _circuit_state
        _circuit_state.clear()
        assert is_degraded("nonexistent", "model") is False

    def test_cooldown_expires(self):
        from app.services.llm_fallback import mark_failure, is_degraded, _circuit_key, _circuit_state
        _circuit_state.clear()
        mark_failure("deepseek", "deepseek-v4-flash")
        assert is_degraded("deepseek", "deepseek-v4-flash") is True
        key = _circuit_key("deepseek", "deepseek-v4-flash")
        _circuit_state[key]["cooldown_until"] = time.monotonic() - 1
        assert is_degraded("deepseek", "deepseek-v4-flash") is False

    def test_success_clears_circuit(self):
        from app.services.llm_fallback import mark_failure, mark_success, is_degraded, _circuit_state
        _circuit_state.clear()
        mark_failure("zhipu", "glm-4.5-air")
        assert is_degraded("zhipu", "glm-4.5-air") is True
        mark_success("zhipu", "glm-4.5-air")
        assert is_degraded("zhipu", "glm-4.5-air") is False

    def test_get_circuit_status(self):
        from app.services.llm_fallback import mark_failure, get_circuit_status, _circuit_state
        _circuit_state.clear()
        mark_failure("deepseek", "deepseek-v4-pro")
        status = get_circuit_status()
        assert "deepseek:deepseek-v4-pro" in status
        assert status["deepseek:deepseek-v4-pro"]["failures"] == 1
        assert status["deepseek:deepseek-v4-pro"]["cooldown_remaining"] > 0

    def test_multiple_failures_exponential_backoff(self):
        from app.services.llm_fallback import mark_failure, _circuit_key, _circuit_state
        _circuit_state.clear()
        mark_failure("zhipu", "glm-4.5-air")
        key = _circuit_key("zhipu", "glm-4.5-air")
        cooldown1 = _circuit_state[key]["cooldown_until"] - time.monotonic()
        mark_failure("zhipu", "glm-4.5-air")
        cooldown2 = _circuit_state[key]["cooldown_until"] - time.monotonic()
        assert cooldown2 > cooldown1 * 2


class TestFallbackChain:
    def test_fallback_to_next_provider(self, monkeypatch):
        from app.services import llm_fallback
        calls = []
        def mock_create(provider_id=None, model=None, **kwargs):
            calls.append((provider_id, model))
            if len(calls) <= 1:
                raise Exception("Primary provider failed")
            return MagicMock()
        monkeypatch.setattr('app.services.llm_provider._create_chat_model_direct', mock_create)
        llm_fallback._circuit_state.clear()
        result = llm_fallback.create_chat_model_with_fallback(streaming=False)
        assert result is not None
        assert len(calls) >= 2

    def test_skip_degraded_provider(self, monkeypatch):
        from app.services import llm_fallback
        llm_fallback._circuit_state.clear()
        llm_fallback.mark_failure("zhipu", "glm-4.5-air")
        calls = []
        def mock_create(provider_id=None, model=None, **kwargs):
            calls.append((provider_id, model))
            return MagicMock()
        monkeypatch.setattr('app.services.llm_provider._create_chat_model_direct', mock_create)
        result = llm_fallback.create_chat_model_with_fallback(streaming=False)
        assert result is not None
        assert not any(p == 'zhipu' for p, _ in calls)

    def test_all_providers_fail_raises_error(self, monkeypatch):
        from app.services import llm_fallback
        call_count = [0]
        def mock_create(provider_id=None, model=None, **kwargs):
            call_count[0] += 1
            raise Exception(f"Provider {provider_id} failed")
        monkeypatch.setattr('app.services.llm_provider._create_chat_model_direct', mock_create)
        llm_fallback._circuit_state.clear()
        with pytest.raises(RuntimeError):
            llm_fallback.create_chat_model_with_fallback(streaming=False)
        assert call_count[0] > 1

    def test_fallback_disabled(self, monkeypatch):
        from app.services import llm_fallback
        monkeypatch.setattr(llm_fallback, 'is_fallback_enabled', lambda: False)
        calls = []
        def mock_create(provider_id=None, model=None, **kwargs):
            calls.append((provider_id, model))
            return MagicMock()
        monkeypatch.setattr('app.services.llm_provider._create_chat_model_direct', mock_create)
        llm_fallback._circuit_state.clear()
        result = llm_fallback.create_chat_model_with_fallback(
            provider_id='deepseek', model='deepseek-chat', streaming=False
        )
        assert result is not None
        assert len(calls) == 1
        assert calls[0] == ('deepseek', 'deepseek-chat')

    def test_active_provider_tracking(self, monkeypatch):
        from app.services import llm_fallback
        def mock_create(provider_id=None, model=None, **kwargs):
            return MagicMock()
        monkeypatch.setattr('app.services.llm_provider._create_chat_model_direct', mock_create)
        llm_fallback._circuit_state.clear()
        llm_fallback._active_local = type('obj', (object,), {'active': None})()
        result = llm_fallback.create_chat_model_with_fallback(streaming=False)
        assert result is not None
        active = llm_fallback.get_active_provider()
        assert active is not None
        assert len(active) == 2
        assert isinstance(active[0], str)
        assert isinstance(active[1], str)

    def test_missing_api_key_skips_provider(self, monkeypatch):
        from app.services import llm_fallback
        def mock_has_key(pid):
            return pid == 'deepseek'
        monkeypatch.setattr(llm_fallback, '_has_api_key', mock_has_key)
        calls = []
        def mock_create(provider_id=None, model=None, **kwargs):
            calls.append((provider_id, model))
            return MagicMock()
        monkeypatch.setattr('app.services.llm_provider._create_chat_model_direct', mock_create)
        llm_fallback._circuit_state.clear()
        result = llm_fallback.create_chat_model_with_fallback(streaming=False)
        assert result is not None
        assert any(p == 'deepseek' for p, _ in calls)
        assert not any(p == 'zhipu' for p, _ in calls)

    def test_circuit_breaker_marks_on_creation_failure(self, monkeypatch):
        from app.services import llm_fallback
        calls = [0]
        def mock_create(provider_id=None, model=None, **kwargs):
            calls[0] += 1
            raise Exception(f"fail {calls[0]}")
        monkeypatch.setattr('app.services.llm_provider._create_chat_model_direct', mock_create)
        llm_fallback._circuit_state.clear()
        with pytest.raises(RuntimeError):
            llm_fallback.create_chat_model_with_fallback(streaming=False)
        status = llm_fallback.get_circuit_status()
        assert len(status) > 0
        for key, info in status.items():
            assert info["failures"] >= 1

    def test_get_fallback_chain_default(self, monkeypatch):
        from app.services.llm_fallback import get_fallback_chain, DEFAULT_CHAIN
        monkeypatch.setattr('app.services.runtime_config.get', lambda k, d=None: d)
        chain = get_fallback_chain()
        assert chain == DEFAULT_CHAIN
