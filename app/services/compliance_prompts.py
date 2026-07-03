"""AI prompt templates for compliance checking — rule extraction + violation analysis.

Provides structured LLM prompts for:
  - Extracting rules/requirements from bidding documents
  - Checking bid documents against extracted rules
  - Generating graded violation reports
"""
from app.services.prompt_safety import build_safe_system_guard

_SAFETY = build_safe_system_guard()

# ── Rule Extraction Prompt ──

RULE_EXTRACTION_SYSTEM = """你是一个专业的招投标文件分析专家。你的任务是从招标文件中提取所有可检查的规则和要求。

【提取维度】
1. 资质要求：企业资质等级、注册资金、业绩要求、人员证书、信用要求等
2. 技术规范：技术参数、质量标准、施工工艺、验收标准、品牌要求等
3. 商务条款：报价方式、付款条件、保证金、履约保函、工期要求、质保期等
4. 废标条件：直接导致投标无效或废标的条件（投标有效期、签字盖章、格式要求等）
5. 禁止性条款：明确禁止的行为或情形（转包、分包限制、关联企业投标等）

【输出格式】
每条规则输出为 JSON 对象，包含以下字段：
{
  "category": "qualification|technical|commercial|rejection|prohibition",
  "rule_id": "短标识符如 REQ-01",
  "description": "可读的规则描述",
  "quantitative": true/false（是否可量化检查）,
  "threshold": "如果有数值阈值，写出如'≥1000万'、'≤30天'，否则null",
  "original_text": "原文引用",
  "severity_if_violated": "warning|violation|critical（违反此规则的严重程度）"
}

【要求】
- 返回纯 JSON 数组，不要 markdown 代码块
- 每条规则必须独立、可判定
- 如果找不到某类的规则，该类返回空数组
- 不编造规则，所有规则必须来自输入文本
- 原文引用必须准确""" + _SAFETY


# ── Compliance Check Prompt ──

COMPLIANCE_CHECK_SYSTEM = """你是一个专业的招投标合规审查专家。你的任务是逐条检查投标文件是否满足招标文件中的规则要求，并检查投标文件是否违反相关法律法规。

【审查维度】
1. 资质合规：投标人是否满足资质等级、业绩、人员等要求
2. 技术合规：技术方案是否满足技术规范和参数要求
3. 商务合规：报价、工期、付款等是否符合商务条款
4. 形式合规：投标文件格式、签字盖章、有效期等
5. 法律合规：是否存在法律法规禁止的行为（围标串标、虚假材料、低于成本报价等）

【判定标准】（4级）
- PASS：完全满足要求，无问题
- WARNING：存在轻微偏差，但不影响实质性响应，建议关注
- VIOLATION：明确违反规则要求，可能影响中标资格
- CRITICAL：严重违规（废标条件、法律禁止行为），直接导致投标无效

【输出格式】
对每条规则，返回 JSON 对象：
{
  "rule_id": "对应规则ID",
  "verdict": "PASS|WARNING|VIOLATION|CRITICAL",
  "evidence": "投标文件中找到的相关内容引用",
  "reasoning": "判定理由（1-2句）",
  "suggestion": "如果违规，给出建议措施；如果通过则为空"
}

如果是基于法律法规的检查（无对应招标文件规则），rule_id 使用"LAW-法律名-条款号"格式。

【要求】
- 返回纯 JSON 数组，不要 markdown 代码块
- 必须给出每条规则的明确判定，不得含糊
- evidence 必须来自投标文件原文，不得编造
- 如果投标文件中找不到相关信息，标注为"投标文件中未见相关内容"，verdict 可判定为 VIOLATION 或 WARNING
- 重点检查：资质证书有效期、业绩金额达标、签字盖章完整性、投标有效期""" + _SAFETY


# ── Build check prompt ──

def build_rule_extraction_prompt(doc_text: str, doc_name: str = "") -> str:
    """Build prompt for extracting rules from a bidding document."""
    truncated = doc_text[:12000] if len(doc_text) > 12000 else doc_text
    return f"""请从以下招标文件中提取所有可检查的规则和要求。

文件名称：{doc_name}

招标文件内容：
---
{truncated}
---

请按照系统提示的要求，提取资质要求、技术规范、商务条款、废标条件、禁止性条款，并以 JSON 数组格式输出。"""


def build_compliance_check_prompt(
    rules: list[dict],
    bid_doc_text: str,
    bid_doc_name: str = "",
    applicable_laws: list[dict] = None,
) -> str:
    """Build prompt for checking bid document compliance against rules + laws.

    Args:
        rules: list of extracted rule dicts from the bidding document
        bid_doc_text: full text of the bid document to check
        bid_doc_name: filename of the bid document
        applicable_laws: optional list of law article dicts relevant to this check
    """
    truncated_bid = bid_doc_text[:15000] if len(bid_doc_text) > 15000 else bid_doc_text

    rules_text = ""
    for r in rules:
        rules_text += f"\n  [{r.get('rule_id', '?')}] ({r.get('category', '?')}) {r.get('description', '')}"
        if r.get('threshold'):
            rules_text += f" [阈值: {r['threshold']}]"
        rules_text += f"\n    原文: {r.get('original_text', '')[:200]}"
        rules_text += f"\n    违规严重度: {r.get('severity_if_violated', 'violation')}\n"

    laws_text = ""
    if applicable_laws:
        laws_text = "\n【适用法律法规条款】"
        for law in applicable_laws:
            laws_text += f"\n  [{law.get('article', '?')}] {law.get('law_name', '')}: {law.get('text', '')[:300]}"

    return f"""请对以下投标文件进行逐条合规审查。

投标文件名称：{bid_doc_name}

【需检查的招标文件规则】
{rules_text}
{laws_text}

【投标文件内容（截取前15000字符）】
---
{truncated_bid}
---

请对照每条规则和法律条款，对投标文件进行逐条检查。返回 JSON 数组格式的检查结果。"""


# ── Summary Report Prompt ──

COMPLIANCE_SUMMARY_SYSTEM = """你是一个专业的招投标合规审查报告撰写人。根据逐条检查结果，撰写一份简洁的合规审查摘要报告。

【报告结构】
1. 总体判定（一段话，概述合规情况）
2. 关键发现（列出所有 VIOLATION 和 CRITICAL 项）
3. 风险提示（WARNING 项汇总）
4. 建议措施

【要求】
- 语言简洁专业（招标代理/工程咨询行业风格）
- 输出为 HTML 片段（<div> 内容块）
- CRITICAL 项用红色标记，VIOLATION 用橙色，WARNING 用黄色，PASS 用绿色
- 所有引用必须来自检查结果，不得编造""" + _SAFETY


def build_summary_prompt(check_results: list[dict], bid_doc_name: str, rule_count: int) -> str:
    """Build prompt for generating a compliance summary report."""
    pass_count = sum(1 for r in check_results if r.get('verdict') == 'PASS')
    warn_count = sum(1 for r in check_results if r.get('verdict') == 'WARNING')
    viol_count = sum(1 for r in check_results if r.get('verdict') == 'VIOLATION')
    crit_count = sum(1 for r in check_results if r.get('verdict') == 'CRITICAL')

    findings = ""
    for r in check_results:
        verdict = r.get('verdict', '?')
        if verdict in ('VIOLATION', 'CRITICAL'):
            findings += f"\n  [{verdict}] 规则 {r.get('rule_id', '?')}: {r.get('reasoning', '')}"
            findings += f"\n    证据: {r.get('evidence', '')[:200]}"
            findings += f"\n    建议: {r.get('suggestion', '')}\n"

    warnings = ""
    for r in check_results:
        if r.get('verdict') == 'WARNING':
            warnings += f"\n  [WARNING] 规则 {r.get('rule_id', '?')}: {r.get('reasoning', '')}"

    return f"""请撰写合规审查摘要报告。

投标文件：{bid_doc_name}
规则总数：{rule_count}
通过：{pass_count} | 警告：{warn_count} | 违规：{viol_count} | 严重违规：{crit_count}

【关键违规项】
{findings if findings else '无关键违规项'}

【警告项】
{warnings if warnings else '无警告项'}

请按照系统提示的报告结构输出 HTML 格式的摘要报告。"""
