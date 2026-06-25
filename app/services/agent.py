"""LangGraph Agent and tools for the AI chat system."""
import os
import json
import asyncio
import threading
import logging
from datetime import datetime, timezone, timedelta

import requests
import aiosqlite
from flask import session

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..config import DATA_DIR, logger as base_logger
from .. import globals as g

logger = logging.getLogger(__name__)

BEIJING_TZ = timezone(timedelta(hours=8))

# System prompt
AGENT_SYSTEM_PROMPT = g.AGENT_SYSTEM_PROMPT

# ---------- Tools ----------

@tool(description="Get current date and time in Beijing time (UTC+8).")
def get_date() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")


BOCHA_API_KEY = os.getenv("BOCHA_API_KEY")
BOCHA_URL = "https://api.bochaai.com/v1/web-search"


@tool(description="Search the web using Bocha. Use for up-to-date information.")
def bocha_search(query: str) -> str:
    headers = {"Authorization": f"Bearer {BOCHA_API_KEY}", "Content-Type": "application/json"}
    payload = json.dumps({"query": query, "summary": True, "freshness": "noLimit", "count": 10})
    try:
        response = requests.post(BOCHA_URL, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        webpages = data.get('data', {}).get('webPages', {}).get('value', [])
        if not webpages:
            return "No search results found."
        formatted = []
        for idx, page in enumerate(webpages[:10], 1):
            title = page.get('name', 'No title')
            snippet = page.get('snippet', 'No snippet')
            date_pub = page.get('datePublished', 'Unknown date')
            url = page.get('url', 'No URL')
            formatted.append(f"{idx}. **{title}**\n   Published: {date_pub}\n   Summary: {snippet}\n   Source: {url}\n")
        return "\n".join(formatted)
    except Exception as e:
        return f"Search failed: {str(e)}"


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
            extra_body={"thinking": {"type": "disabled"}},
        )
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
