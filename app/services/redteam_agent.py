"""LangGraph Agent for Red Teaming (Grilling Mode)."""
import os
import logging
from flask import session
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import create_react_agent

from .agent import get_date, bocha_search, _init_async_checkpointer
from .. import globals as g

logger = logging.getLogger(__name__)

REDTEAM_SYSTEM_PROMPT = """
【质问模式】你现在是一个极其挑剔、攻击性强、且极具洞察力的供应商代表（Red Team）。
你的任务是审查用户提供的招标/项目文件，寻找其中的漏洞、矛盾、模糊不清的要求、或不合理的技术壁垒。
**规则：**
1. 用词尖锐但专业，不要客套。
2. 每次只抛出一个最核心的致命问题，迫使招标方（用户）进行解释。
3. 如果用户的解释合理，你可以接受并转向下一个漏洞；如果解释不合理，继续深入追问。
4. 你的最终目标是帮助用户完善招标文件，消除所有潜在的答疑争议。
5. 你必须使用中文进行所有对话。
6. 对于需要实时信息的问题，你必须使用 bocha_search 工具。
7. 你必须在每个回答中明确包含【思考】和【回答】两个部分，使用中文双括号。
"""

_redteam_agent = None

def get_redteam_agent(max_tokens=None):
    global _redteam_agent
    if max_tokens is None:
        max_tokens = session.get('max_tokens', g._current_max_tokens)
    
    with g._agent_lock:
        if _redteam_agent is not None and getattr(g, '_current_redteam_max_tokens', None) == max_tokens:
            return _redteam_agent
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY or QWEN_API_KEY")
        os.environ["DASHSCOPE_API_KEY"] = api_key
        os.environ["DASHSCOPE_API_BASE"] = "https://api.deepseek.com/v1"
        
        llm = ChatDeepSeek(
            model="deepseek-v4-pro",
            api_key=api_key,
            temperature=0.7,
            max_tokens=max_tokens,
            streaming=False,
            request_timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            extra_body={"thinking": {"type": "disabled"}},
        )
        
        if g._async_checkpointer is None:
            with g._async_checkpointer_lock:
                if g._async_checkpointer is None:
                    _init_async_checkpointer()
                    
        # We use create_react_agent from langgraph.prebuilt
        _redteam_agent = create_react_agent(
            model=llm,
            tools=[get_date, bocha_search],
            state_modifier=REDTEAM_SYSTEM_PROMPT,
            checkpointer=g._async_checkpointer
        )
        g._current_redteam_max_tokens = max_tokens
        logger.info(f"Red Team Agent initialized with max_tokens={max_tokens}")
        return _redteam_agent
