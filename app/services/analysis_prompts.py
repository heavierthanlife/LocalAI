"""AI-powered prompt templates: bid comparison + work report generation.

Provides structured LLM prompts for:
  - Bid document comparison analysis
  - Periodic work reports from chat history aggregation
"""
from app.services.prompt_safety import build_safe_system_guard

# ── Anti-hallucination guard (appended to every system prompt) ──
_SAFETY = build_safe_system_guard()

# ── Bid Comparison Prompt ──

BID_COMPARISON_SYSTEM = """你是一个专业的招投标分析助手，专精于技术标规律性分析。请严格基于提供的数据进行判断。

【分析维度】
1. 风险等级判定：根据风险度矩阵，哪些文件对存在围标串标风险
2. 文本雷同分析：匹配段落的具体内容是否真的雷同，还是偶然的模板化表述
3. 文件属性分析：相同作者、相同创建时间的异常情况
4. 综合建议：是否建议人工复核、是否需要进一步取证

【输出要求】
- 用中文，语言简洁专业（招标代理/工程咨询行业风格）
- 输出为 HTML 片段（不用完整 HTML 文档，只需要 <div> 内容块）
- 如果有高风险对（风险度>20），用红色标记警告
- 如果全部正常，用绿色标记"未发现明显围串标风险"
- 不要输出 JSON，不要 markdown 代码块
- 所有风险数值和公司名称必须来自输入数据，不得编造
- 如果数据不足以判断，必须明确说「基于现有数据无法确定」""" + _SAFETY


def build_bid_analysis_prompt(
    risk_matrix: list,
    file_names: list,
    high_risk_pairs: list,
    top_pair_text1: str = "",
    top_pair_text2: str = "",
    top_pair_name1: str = "",
    top_pair_name2: str = "",
    top_pair_risk: float = 0,
) -> str:
    """Build a user prompt for AI to write a human-readable analysis of batch comparison results.

    Args:
        risk_matrix: n×n risk matrix (list of lists)
        file_names: list of n file names
        high_risk_pairs: list of (name1, name2, risk, sim_pct) for high-risk pairs
        top_pair_text1: content of file A in the highest-risk pair (first 3000 chars)
        top_pair_text2: content of file B in the highest-risk pair (first 3000 chars)
        top_pair_name1: name of file A
        top_pair_name2: name of file B
        top_pair_risk: risk score of the top pair

    Returns:
        User prompt string to send to the LLM.
    """
    n = len(file_names)

    # Risk matrix summary
    matrix_lines = ["风险度矩阵："]
    header = "         " + "".join(f"{name[:8]:>8}" for name in file_names)
    matrix_lines.append(header)
    for i in range(n):
        row = f"{file_names[i][:8]:>8} " + "".join(
            f"{'--':>8}" if i == j else f"{risk_matrix[i][j]:>7.1f} "
            for j in range(n)
        )
        matrix_lines.append(row)

    # High-risk pairs list
    if high_risk_pairs:
        risk_lines = ["高危文件对："]
        for n1, n2, r, s in high_risk_pairs[:5]:
            flag = "🚨" if r > 20 else "⚠️"
            risk_lines.append(f"  {flag} {n1} ↔ {n2} | 风险度 {r:.1f} | 文本相似度 {s:.1f}%")
    else:
        risk_lines = ["高危文件对：无"]

    # Sample text from the highest-risk pair
    sample_text = ""
    if top_pair_text1 or top_pair_text2:
        sample_text = f"""
最高风险文件对的内容样本（各取前 3000 字符）：

=== {top_pair_name1}（风险度 {top_pair_risk:.1f}）===
{top_pair_text1[:3000]}

=== {top_pair_name2}（风险度 {top_pair_risk:.1f}）===
{top_pair_text2[:3000]}
"""

    return f"""以下是一次技术标批量对比的统计学结果。请基于这些数据，用专业语言写一份简明分析报告（300-600字）。

{chr(10).join(matrix_lines)}

{chr(10).join(risk_lines)}
{sample_text}

请写出：
1. 总体风险判断
2. 最高风险对的具体分析（是否真的雷同，还是标准化表述）
3. 建议"""


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
