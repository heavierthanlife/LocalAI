"""Judge Model Review Pipeline — second-model quality check for AI outputs.

Pattern: Coder (main agent) → Reviewer (judge model) → final output.
Enabled via runtime_config ``judge_review_enabled`` (default false, saves API costs).
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are a quality reviewer. Review the following AI assistant response for:

1. Factual accuracy — any statements that contradict common knowledge or provided context?
2. Completeness — does it fully answer the user's question?
3. Tone — is it professional and appropriate?
4. Format — is it well-structured and readable?

User question: {user_query}
AI response to review: {ai_response}

Provide your review in this format exactly (do not change field names):
SCORE: [1-10]
VERDICT: [PASS / NEEDS_IMPROVEMENT / FAIL]
ISSUES: [list specific issues or "None"]
CORRECTED_RESPONSE: [improved version if NEEDS_IMPROVEMENT, otherwise "N/A"]

Do NOT fabricate issues that don't exist. If the response is factually correct and complete, say so."""


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
        resp = llm.invoke([SystemMessage(content="You are a strict quality reviewer. Be concise."),
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
