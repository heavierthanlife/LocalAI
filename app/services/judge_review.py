"""Judge Model Review Pipeline — second-model quality check for AI outputs.

Pattern: Coder (main agent) → Reviewer (judge model) → final output.
Enabled via runtime_config ``judge_review_enabled`` (default false, saves API costs).
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """你是一名严格的中文质量审查员。请审查以下 AI 助手生成的回答，从四个维度评价：

1. 事实准确性 — 是否存在与常识或给定上下文相矛盾的陈述？
2. 完整性 — 是否充分回答了用户的问题？
3. 语气 — 是否专业、得体、符合招投标业务场景？
4. 格式 — 结构是否清晰、是否易于阅读？

用户问题：{user_query}
待审查的 AI 回答：{ai_response}

请严格按照以下格式输出（不要更改字段名）：
SCORE: [1-10]
VERDICT: [PASS / NEEDS_IMPROVEMENT / FAIL]
ISSUES: [列出具体问题，或 "None"]
CORRECTED_RESPONSE: [若 NEEDS_IMPROVEMENT 则给出改进版，否则 "N/A"]

不要凭空捏造不存在的问题。如果回答事实正确且完整，请如实说明。"""


def review_response(user_query: str, ai_response: str,
                    max_tokens: int = 600, timeout: int = 30) -> dict | None:
    """Run a judge model review on an AI response.

    Uses a different model than the primary agent for unbiased review.
    Falls back gracefully if judge model is unavailable.
    """
    try:
        from app.services.runtime_config import get as rc_get
        if not rc_get("judge_review_enabled", False):
            return None
    except Exception:
        return None

    # Skip very short responses
    if len(ai_response) < 50:
        return None

    try:
        from app.services.llm_provider import PROVIDER_CONFIG, get_available_providers
        import os

        available = get_available_providers()
        if len(available) < 2:
            logger.debug("Judge review skipped: only 1 LLM provider available")
            return None

        # Pick a different provider than the primary (DeepSeek)
        judge_provider = None
        for pid in available:
            if pid != 'deepseek':
                judge_provider = pid
                break
        if not judge_provider:
            judge_provider = available[0]  # fallback

        from app.services.llm_provider import create_chat_model
        from langchain_core.messages import SystemMessage, HumanMessage

        prompt = JUDGE_PROMPT.format(user_query=user_query[:2000], ai_response=ai_response[:3000])
        llm = create_chat_model(
            provider_id=judge_provider,
            streaming=False, temperature=0.1, max_tokens=max_tokens, timeout=timeout
        )
        resp = llm.invoke([SystemMessage(content="你是一名严格的质量审查员，请简洁、客观地评价。"),
                          HumanMessage(content=prompt)])
        review_text = resp.content if hasattr(resp, 'content') else str(resp)

        # Parse with lenient parser (survives LLM format drift)
        from app.services.prompt_safety import safe_judge_parse
        parsed = safe_judge_parse(review_text)

        result = {
            'score': parsed['score'],
            'verdict': parsed['verdict'],
            'issues': parsed['issues'] or review_text[:500],
            'corrected': parsed['corrected'],
            'judge_model': judge_provider,
            'reviewed_at': datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"Judge review: score={result['score']}, verdict={result['verdict']}, model={judge_provider}")
        return result
    except Exception as e:
        logger.warning(f"Judge review failed (graceful skip): {e}")
        return None
