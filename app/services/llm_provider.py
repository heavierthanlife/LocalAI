"""Multi-provider LLM routing (adapted from douban-ai-analyzer background.js).

Supports: DeepSeek (default), Zhipu, Qwen, SiliconFlow.
All providers use OpenAI-compatible chat completions API.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Provider definitions ──
PROVIDER_CONFIG = {
    'deepseek': {
        'name': 'DeepSeek',
        'env_key': 'DEEPSEEK_API_KEY',
        'base_url': 'https://api.deepseek.com',
        'default_model': 'deepseek-v4-pro',
        'models': ['deepseek-v4-pro', 'deepseek-v4-flash', 'deepseek-chat', 'deepseek-reasoner'],
    },
    'zhipu': {
        'name': '智谱AI',
        'env_key': 'ZHIPU_API_KEY',
        'base_url': 'https://open.bigmodel.cn/api/paas/v4',
        'default_model': 'glm-4.5-air',
        'models': ['glm-4.5-air', 'glm-4-flash', 'glm-4-plus', 'glm-4-air', 'glm-4-long'],
    },
    'qwen': {
        'name': '通义千问',
        'env_key': 'QWEN_API_KEY',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'default_model': 'qwen3.7-plus',
        'models': ['qwen3.7-plus', 'qwen-max', 'qwen-plus', 'qwen-turbo'],
    },
    'siliconflow': {
        'name': '硅基流动',
        'env_key': 'SILICONFLOW_API_KEY',
        'base_url': 'https://api.siliconflow.cn/v1',
        'default_model': 'Qwen/Qwen2.5-7B-Instruct',
        'models': ['Qwen/Qwen2.5-7B-Instruct', 'deepseek-ai/DeepSeek-V3', 'Qwen/Qwen2.5-72B-Instruct'],
    },
    'mimo': {
        'name': 'Mimo',
        'env_key': 'MIMO_API_KEY',
        'base_url': 'https://token-plan-cn.xiaomimimo.com/v1',
        'default_model': 'mimo-v2.5-pro',
        'models': ['mimo-v2.5-pro', 'mimo-v2.5'],
    },
}


def get_available_providers() -> list[str]:
    """Return list of provider IDs that have API keys configured."""
    available = []
    for pid, cfg in PROVIDER_CONFIG.items():
        key = os.getenv(cfg['env_key'], '').strip()
        if key:
            available.append(pid)
    return available


def get_active_provider() -> Optional[str]:
    """Return the first available provider, or None if none configured."""
    available = get_available_providers()
    return available[0] if available else None


def get_provider_config(provider_id: Optional[str] = None) -> dict:
    """Get config dict for a provider. Falls back to first available if none specified."""
    if provider_id and provider_id in PROVIDER_CONFIG:
        return PROVIDER_CONFIG[provider_id]
    active = get_active_provider()
    if active:
        return PROVIDER_CONFIG[active]
    raise RuntimeError(
        "No LLM provider configured. Set one of: "
        + ", ".join(cfg['env_key'] for cfg in PROVIDER_CONFIG.values())
        + " in your .env file."
    )


def create_chat_model(
    provider_id: Optional[str] = None,
    model: Optional[str] = None,
    streaming: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1600,
    timeout: int = 120,
):
    """Create a LangChain-compatible ChatModel for the given provider.

    Args:
        provider_id: One of 'deepseek','zhipu','qwen','siliconflow'. Auto-detect if None.
        model: Model name. Uses provider default if None.
        streaming: Enable token streaming.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
        timeout: Request timeout in seconds.
    """
    cfg = get_provider_config(provider_id)
    api_key = os.getenv(cfg['env_key'], '').strip()
    if not api_key:
        raise RuntimeError(f"API key for {cfg['name']} not set ({cfg['env_key']}).")

    final_model = model or cfg['default_model']
    base_url = cfg['base_url']

    logger.info(f"Creating LLM: provider={cfg['name']}, model={final_model}, streaming={streaming}")

    if cfg['name'] == 'DeepSeek':
        # Use langchain-deepseek's native class (better integration)
        from langchain_deepseek import ChatDeepSeek
        extra = {}
        if 'pro' in final_model.lower():
            extra['thinking'] = {'type': 'disabled'}
        return ChatDeepSeek(
            model=final_model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            request_timeout=timeout,
            extra_body=extra if extra else None,
        )
    else:
        # Use OpenAI-compatible ChatOpenAI for all other providers
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=final_model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            request_timeout=timeout,
        )


# ── Quick API key helpers (for /send endpoint fallback) ──

def get_any_api_key() -> Optional[str]:
    """Return the first available API key across all providers."""
    for cfg in PROVIDER_CONFIG.values():
        key = os.getenv(cfg['env_key'], '').strip()
        if key:
            return key
    return None


# ── Industry-to-model routing (for future LoRA fine-tuned models) ──

# Map industry to model override. Set via env vars or runtime config.
# When a LoRA adapter is trained for a domain, set INDUSTRY_MODEL_{domain} 
# to the fine-tuned model name/path. Leave unset to use default model.
# Example: INDUSTRY_MODEL_bidding_agency = "qwen2.5-7b-bidding-lora"
INDUSTRY_MODEL_MAP = {}

def _load_industry_models():
    """Load industry-specific model overrides from environment + adapter registry."""
    global INDUSTRY_MODEL_MAP
    # Layer 1: Environment variables (manual override)
    for domain in ('bidding_agency', 'engineering_cost', 'engineering_audit'):
        env_key = f'INDUSTRY_MODEL_{domain.upper()}'
        val = os.getenv(env_key, '').strip()
        if val:
            INDUSTRY_MODEL_MAP[domain] = val

    # Layer 2: Auto-discover trained adapters from registry
    # (LoRA adapters trained by scripts/run_lora_training.py)
    try:
        registry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'training', 'adapter_registry.json'
        )
        if os.path.exists(registry_path):
            import json as _json
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = _json.load(f)
            for industry, info in registry.items():
                if info.get('active', True):
                    adapter_path = info.get('adapter_path', '')
                    base_model = info.get('base_model', '')
                    # If adapter exists on disk, register as "ollama:{industry}"
                    # The actual serving is done by Ollama/vLLM (separate process)
                    # Here we just record the path — the provider must serve it
                    if adapter_path and os.path.isdir(adapter_path):
                        # Use Ollama model name convention: {industry}-lora
                        ollama_name = f"{industry}-lora"
                        INDUSTRY_MODEL_MAP[industry] = ollama_name
                        logger.info(f"Loaded LoRA adapter for '{industry}': {adapter_path}")
    except Exception as e:
        logger.debug(f"Adapter registry load skipped: {e}")

# Load on import
try:
    _load_industry_models()
except Exception:
    pass


def get_industry_model(industry: str) -> Optional[str]:
    """Return the fine-tuned model name for a given industry, or None if not configured."""
    return INDUSTRY_MODEL_MAP.get(industry)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.5,
    max_tokens: int = 2000,
    provider_id: Optional[str] = None,
    model: Optional[str] = None,
    industry: str = 'general',
) -> str:
    """Universal LLM invocation — all AI features use this single entry point.

    Automatically applies:
      - Prompt-injection sanitization on user_prompt
      - Anti-hallucination safety guard on system_prompt
      - User-content wrapping for injection defence

    Returns the response text string. Raises on failure.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # ── Safety layer: sanitize + wrap user content ──
    from app.services.prompt_safety import (
        sanitize_for_prompt, wrap_user_content, build_safe_system_guard,
    )
    safe_user = sanitize_for_prompt(user_prompt, 'user_query')
    safe_user = wrap_user_content(safe_user, 'USER_QUERY')
    safe_system = system_prompt + build_safe_system_guard()

    # Try industry-specific model first, fall back to default
    llm = None
    if industry != 'general':
        try:
            llm = create_chat_model_for_industry(
                industry=industry, provider_id=provider_id,
                streaming=False, temperature=temperature, max_tokens=max_tokens
            )
        except Exception:
            pass
    if llm is None:
        llm = create_chat_model(
            provider_id=provider_id, model=model,
            streaming=False, temperature=temperature, max_tokens=max_tokens
        )

    response = llm.invoke([
        SystemMessage(content=safe_system),
        HumanMessage(content=safe_user)
    ])
    return response.content if hasattr(response, 'content') else str(response)


def create_chat_model_for_industry(
    industry: str = 'general',
    provider_id: Optional[str] = None,
    streaming: bool = False,
    temperature: float = 0.5,
    max_tokens: int = 3200,
):
    """Create a ChatModel, optionally using an industry-specific fine-tuned model.
    
    When an industry model is configured (via env var), uses that model.
    Otherwise falls back to the default provider model.
    """
    industry_model = get_industry_model(industry)
    return create_chat_model(
        provider_id=provider_id,
        model=industry_model,
        streaming=streaming,
        temperature=temperature,
        max_tokens=max_tokens,
    )
