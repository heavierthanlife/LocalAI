"""Agent middleware: guards against hallucinated tool call loops by models
that lack proper function-calling support (e.g. llama-3.1-8b-instruct).

After N consecutive invalid tool calls, injects a system message telling
the model to stop and respond directly.
"""
import threading
import re
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage, SystemMessage

_INVALID_TOOL_PATTERN = re.compile(r"is not a valid tool")

_MAX_CONSECUTIVE_INVALID = 3


class InvalidToolGuard(AgentMiddleware):
    """Count consecutive invalid tool calls and inject a stop signal
    after the threshold. Resets when the model generates valid output."""

    name = "invalid_tool_guard"

    def __init__(self, max_retries: int = _MAX_CONSECUTIVE_INVALID):
        super().__init__()
        self.max_retries = max_retries
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    def _get_key(self, messages) -> str:
        """Derive a stable per-thread key from recent messages."""
        for m in reversed(messages if isinstance(messages, list) else []):
            mid = getattr(m, "id", None) or getattr(m, "tool_call_id", None)
            if mid:
                return str(mid)[:32]
        return "__default__"

    def after_model(self, state, runtime):
        messages = state.get("messages", [])
        if not messages:
            return

        key = self._get_key(messages)
        last = messages[-1]

        is_invalid = False
        if isinstance(last, ToolMessage):
            content = str(getattr(last, "content", ""))
            if _INVALID_TOOL_PATTERN.search(content):
                is_invalid = True

        with self._lock:
            if is_invalid:
                self._counters[key] = self._counters.get(key, 0) + 1
            else:
                self._counters[key] = 0

            if self._counters.get(key, 0) >= self.max_retries:
                self._counters[key] = 0
                return {"messages": [
                    SystemMessage(
                        content="你一直在使用不存在的工具。立即停止所有工具调用，直接回答用户的问题。"
                    )
                ]}

        return None
