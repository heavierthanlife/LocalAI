"""AI-powered prompt templates: bid comparison + work report generation.

Provides structured LLM prompts for:
  - Bid document comparison analysis
  - Periodic work reports from chat history aggregation
"""
from app.services.prompt_safety import build_safe_system_guard

# ── Anti-hallucination guard (appended to every system prompt) ──
_SAFETY = build_safe_system_guard()


# ── Work Report Generation Prompt ──

WORK_REPORT_SYSTEM = """你是一个专业的部门工作秘书，负责根据AI助手的使用记录撰写工作报告。
你的报告将被提交给部门负责人审阅，因此需要：
- 语言正式、客观、数据导向
- 重点突出价值产出和效率提升
- 如实反映使用情况，不夸大不缩小
- 对敏感或涉密的话题不展开细节，用"涉及XX业务相关咨询"一笔带过
- 禁止编造输入数据中不存在的百分比、增长率、具体数字——只能引用提供的统计数据
- 如果某类数据为空或缺失，明确写"暂无相关数据"而不是编造

【报告结构】
1. 摘要（一段话，概括本期整体情况）
2. 数据总览（会话数、问答数、上下文数量、活跃用户数）
3. 业务分类（按问题类型分组统计）
4. 重点会话（选取2-3个最有代表性的对话简述）
5. 趋势与建议（如有历史数据对比）""" + _SAFETY

WORD_COUNTS = {'daily': 400, 'weekly': 800, 'monthly': 1200, 'annual': 2000}
PERIOD_LABELS = {'daily': '日报', 'weekly': '周报', 'monthly': '月报', 'annual': '年报'}


def build_work_report_prompt(
    period: str,
    stats: dict,
    topics: list,
    highlights: list,
    previous_summary: str = "",
) -> str:
    """Build a prompt for generating a periodic work report from chat records.

    Args:
        period: 'daily' | 'weekly' | 'monthly' | 'annual'
        stats: dict with 'sessions', 'messages', 'users', 'knowledge_files', 'credit_checks', etc.
        topics: list of (topic_category, count) tuples
        highlights: list of (question_preview, answer_preview, username, date) for highlight sessions
        previous_summary: summary from last period's report (for trend comparison)

    Returns:
        User prompt string.
    """
    word_target = WORD_COUNTS.get(period, 800)
    label = PERIOD_LABELS.get(period, '报告')

    stats_text = f"""会话总数：{stats.get('sessions', 0)}
问答对数：{stats.get('messages', 0)}
活跃用户数：{stats.get('users', 0)}
知识库文件使用：{stats.get('knowledge_files', 0)} 次
征信查询：{stats.get('credit_checks', 0)} 次
批量对比：{stats.get('batch_compares', 0)} 次"""

    topics_text = ""
    if topics:
        topics_text = "业务分类：" + " · ".join(f"{cat}({cnt}次)" for cat, cnt in topics[:10])

    highlights_text = ""
    if highlights:
        highlights_text = "重点对话：\n" + "\n".join(
            f"- [{h[2]}] {h[3][:10]}: Q: {h[0][:120]} → A: {h[1][:120]}"
            for h in highlights[:3]
        )

    trend_text = ""
    if previous_summary:
        trend_text = f"\n【上期摘要参考】\n{previous_summary[:300]}\n请对比上期数据，指出变化趋势。"

    return f"""请根据以下AI助手的实际使用记录，撰写一份{label}（目标{word_target}字）。

{stats_text}

{topics_text}

{highlights_text}
{trend_text}

请严格按照【报告结构】输出 Markdown 格式的报告。重点放在业务价值上，语言正式简洁。"""
