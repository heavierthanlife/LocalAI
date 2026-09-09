"""Multi-provider LLM routing.

Active providers (FIX-016): OpenRouter (:free model pool) + NVIDIA NIM.
Both use OpenAI-compatible chat completions API.

Legacy providers (deepseek/zhipu/qwen/siliconflow/mimo) are COMMENTED OUT
but preserved for reference — user decided to consolidate on the two free
sources. See `data/fix_registry.yaml` FIX-2026-09-01-016.

Custom providers (admin-configured via runtime_config `llm_custom_providers`)
are dynamically merged on top of PROVIDER_CONFIG; built-in providers are
never overridden. See `get_merged_provider_config()`.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Provider definitions ──
# Active: OpenRouter (:free pool) + NVIDIA NIM. Model lists are refreshed daily
# from each provider's /models endpoint (see app/services/llm_catalog.py).
PROVIDER_CONFIG = {
    'openrouter': {
        'name': 'OpenRouter',
        'env_key': 'OPENROUTER_API_KEY',
        'base_url': 'https://openrouter.ai/api/v1',
        'default_model': 'nvidia/nemotron-3-ultra-550b-a55b:free',
        'models': ['nvidia/nemotron-3-ultra-550b-a55b:free', 'z-ai/glm-5.2:free', 'thinkingmachines/inkling:free'],
    },
    'nvidia': {
        'name': 'NVIDIA NIM',
        'env_key': 'NVIDIA_API_KEY',
        'base_url': 'https://integrate.api.nvidia.com/v1',
        'default_model': 'nvidia/nemotron-3-ultra-550b-a55b',
        'models': ['nvidia/nemotron-3-ultra-550b-a55b', 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning', 'nvidia/nemotron-3-super-120b-a12b'],
    },
}

# ── Legacy providers (COMMENTED OUT — preserved for reference) ──
# _LEGACY_PROVIDER_CONFIG = {
#     'deepseek': {
#         'name': 'DeepSeek',
#         'env_key': 'DEEPSEEK_API_KEY',
#         'base_url': 'https://api.deepseek.com',
#         'default_model': 'deepseek-v4-pro',
#         'models': ['deepseek-v4-pro', 'deepseek-v4-flash', 'deepseek-chat', 'deepseek-reasoner'],
#     },
#     'zhipu': {
#         'name': '智谱AI',
#         'env_key': 'ZHIPU_API_KEY',
#         'base_url': 'https://open.bigmodel.cn/api/paas/v4',
#         'default_model': 'glm-4.5-air',
#         'models': ['glm-4.5-air', 'glm-4-flash', 'glm-4-plus', 'glm-4-air', 'glm-4-long'],
#     },
#     'qwen': {
#         'name': '通义千问',
#         'env_key': 'QWEN_API_KEY',
#         'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
#         'default_model': 'qwen3.7-plus',
#         'models': ['qwen3.7-plus', 'qwen-max', 'qwen-plus', 'qwen-turbo'],
#     },
#     'siliconflow': {
#         'name': '硅基流动',
#         'env_key': 'SILICONFLOW_API_KEY',
#         'base_url': 'https://api.siliconflow.cn/v1',
#         'default_model': 'Qwen/Qwen2.5-7B-Instruct',
#         'models': ['Qwen/Qwen2.5-7B-Instruct', 'deepseek-ai/DeepSeek-V3', 'Qwen/Qwen2.5-72B-Instruct'],
#     },
#     'mimo': {
#         'name': 'Mimo',
#         'env_key': 'MIMO_API_KEY',
#         'base_url': 'https://token-plan-cn.xiaomimimo.com/v1',
#         'default_model': 'mimo-v2.5-pro',
#         'models': ['mimo-v2.5-pro', 'mimo-v2.5'],
#     },
# }


def get_merged_provider_config() -> dict:
    """PROVIDER_CONFIG 防御性拷贝 + 自定义 provider 动态叠加（内置永不覆盖）。"""
    from app.services.runtime_config import get as rc_get
    merged = {k: dict(v) for k, v in PROVIDER_CONFIG.items()}
    try:
        customs = rc_get('llm_custom_providers', []) or []
    except Exception:
        customs = []
    for cp in customs:
        pid = (cp.get('id') or '').strip()
        if not pid or pid in merged:   # 防自定义覆盖内置（id='openrouter' 等）
            continue
        models = cp.get('models') or []
        merged[pid] = {
            'name': cp.get('name') or pid,
            'env_key': (cp.get('api_key_env') or '').strip(),
            'base_url': (cp.get('base_url') or '').strip(),
            'default_model': models[0] if models else '',
            'models': models,
            'custom': True,
        }
    return merged


def validate_custom_provider(entry) -> tuple:
    """校验自定义 provider 条目，返回 (ok, error_msg)。"""
    import re
    if not isinstance(entry, dict):
        return False, "条目必须是 dict"
    pid = (entry.get('id') or '').strip()
    if not pid:
        return False, "id 不能为空"
    if not re.fullmatch(r'[A-Za-z0-9_-]+', pid):
        return False, "id 只能包含字母/数字/下划线/中划线"
    if pid in PROVIDER_CONFIG:
        return False, f"id '{pid}' 与内置 provider 冲突"
    base_url = (entry.get('base_url') or '').strip()
    if not base_url.startswith('https://') and not (
        base_url.startswith('http://localhost') or base_url.startswith('http://127.0.0.1')
    ):
        return False, "base_url 必须以 https:// 开头（http://localhost 或 http://127.0.0.1 本地豁免）"
    api_key_env = (entry.get('api_key_env') or '').strip()
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', api_key_env):
        return False, "api_key_env 必须是合法环境变量名"
    return True, ''


def get_available_providers() -> list[str]:
    """Return list of provider IDs that have API keys configured."""
    merged = get_merged_provider_config()
    available = []
    for pid, cfg in merged.items():
        env_key = (cfg.get('env_key') or '').strip()
        if not env_key:
            continue
        if os.getenv(env_key, '').strip():
            available.append(pid)
    return available


def get_active_provider() -> Optional[str]:
    """Return the first available provider, or None if none configured."""
    available = get_available_providers()
    return available[0] if available else None


def get_provider_config(provider_id: Optional[str] = None) -> dict:
    """Get config dict for a provider. Falls back to first available if none specified."""
    merged = get_merged_provider_config()
    if provider_id:
        cfg = merged.get(provider_id)
        if cfg is not None:
            return cfg
    active = get_active_provider()
    if active:
        return merged[active]
    raise RuntimeError(
        "No LLM provider configured. Set one of: "
        + ", ".join(cfg['env_key'] for cfg in PROVIDER_CONFIG.values())
        + " in your .env file."
    )


def _create_chat_model_direct(
    provider_id: Optional[str] = None,
    model: Optional[str] = None,
    streaming: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1600,
    timeout: int = 120,
):
    """Build a LangChain ChatModel directly from a provider id + model.

    FIX-016: this function exists so llm_fallback can build models directly
    (previously referenced a non-existent symbol → runtime ImportError).
    Both active providers (openrouter/nvidia) use OpenAI-compatible ChatOpenAI.
    Custom providers (llm_custom_providers) are also supported via the merged view.
    """
    from app.services.runtime_config import get as rc_get
    cfg = get_provider_config(provider_id)
    api_key = os.getenv(cfg['env_key'], '').strip()
    if not api_key:
        raise RuntimeError(f"API key for {cfg['name']} not set ({cfg['env_key']}).")
    final_model = model or cfg['default_model']
    base_url = cfg['base_url']
    logger.info(f"Creating LLM: provider={cfg['name']}, model={final_model}, streaming={streaming}")
    # 运行时死配置覆盖默认温度/最大长度；无值则沿用函数参数
    rt_temp = rc_get('llm_temperature')
    if rt_temp is not None:
        temperature = rt_temp
    rt_max = rc_get('llm_max_tokens')
    if rt_max is not None:
        max_tokens = rt_max
    kwargs = dict(
        model=final_model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        request_timeout=timeout,
    )
    # 统一 high thinking：reasoning_effort 注入（自定义 provider 若 supports_reasoning=False 跳过）
    effort = rc_get('llm_reasoning_effort', 'high')
    if effort and cfg.get('supports_reasoning', True):
        kwargs['extra_body'] = {'reasoning_effort': effort}
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(**kwargs)


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
        provider_id: One of 'openrouter','nvidia'. Auto-detect if None.
        model: Model name. Uses provider default if None.
        streaming: Enable token streaming.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
        timeout: Request timeout in seconds.
    """
    return _create_chat_model_direct(
        provider_id=provider_id, model=model,
        streaming=streaming, temperature=temperature,
        max_tokens=max_tokens, timeout=timeout,
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
    guard = build_safe_system_guard()
    safe_system = system_prompt + guard if system_prompt and guard not in system_prompt else system_prompt

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
