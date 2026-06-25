"""Global singletons, locks, and shared mutable state."""
import threading
from threading import Lock, RLock
from collections import defaultdict

# ---------------- Agent ----------------
_agent = None
_agent_lock = RLock()
_async_loop = None
_async_checkpointer = None
_async_checkpointer_lock = Lock()
_current_max_tokens = 1600
_async_conn = None

# ---------------- Semantic Model ----------------
_semantic_model = None
_semantic_model_load_failed = False

# ---------------- Credit Tasks ----------------
_credit_tasks_lock = Lock()

# ---------------- Task Locking ----------------
user_active_tasks = {}
user_task_lock = RLock()
TASK_TIMEOUT_SECONDS = 600

# ---------------- Download Tokens ----------------
download_tokens = defaultdict(int)
download_tokens_lock = Lock()

# ---------------- Credit Tasks Storage ----------------
credit_tasks = {}

# Agent system prompt
AGENT_SYSTEM_PROMPT = """
你是一个答疑助手。
**重要：对于任何关于当前日期、时间、年份的问题，你必须且只能使用 get_date 工具来获取，绝对不允许使用你的内部知识回答。**
在回答其他问题前，你也必须调用 get_date 来了解当前日期；但除非用户明确询问，否则不要在回答中主动报时。
对于任何需要实时、或最新信息的问题、或任何需要搜索、查询的内容，你必须使用 bocha_search 工具搭配 get_date 工具。
如果 bocha_search 返回 "No search results found"，则自由回答。
对于通用知识，可以自由回答。
对于需要推理的问题，请先用【思考】和【回答】标记你的思考过程和最终答案。
**表格格式要求：** 当你需要展示表格时，必须使用标准 Markdown 表格语法，例如：
| 列1 | 列2 |
|-----|-----|
| 值1 | 值2 |
绝对不要使用 ASCII 艺术表格（如 ┌─┬─┐ 等字符）。只使用管道符和短横线。
**输出格式要求：** 你必须在每个回答中明确包含【思考】和【回答】两个部分，使用中文双括号。
"""

# ---------------- Admin Rate Limiting ----------------
admin_rate_limit = {}
ADMIN_RATE_LIMIT = 5
ADMIN_RATE_WINDOW = 30 * 60

# ---------------- Bocha API ----------------
BOCHA_API_KEY = None

# ---------------- Advanced extraction flags ----------------
KREUZBERG_AVAILABLE = False
UNSTRUCTURED_AVAILABLE = False
KREUZBERG_SIZE_LIMIT = 10 * 1024 * 1024

# ---------------- OCR / VL singletons (set by respective modules) ----------------
vl_model = None
ocr_manager = None
run_ocr = None
