"""Rule Extraction Engine — AI + regex dual-channel extraction from bidding documents.

Architecture:
    1. AI channel: LLM extracts structured rules (qualification, technical, commercial, rejection, prohibition)
    2. Regex channel: pattern-based fallback for common rule patterns the AI might miss
    3. Merge + dedup: combine both channels, remove duplicates, assign rule_ids

Usage:
    from app.services.rule_extractor import RuleExtractor
    extractor = RuleExtractor()
    rules = extractor.extract(doc_text, doc_name)
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ── Regex patterns for common rule types ──

_REGEX_PATTERNS = {
    "qualification": [
        # 资质等级
        (r'(?:具有|具备|持有).{0,10}(?:资质|资格|许可证).{0,20}(?:等级|证书).{0,10}([\u4e00-\u9fff]*级)', 'qualification'),
        (r'(?:注册资金|注册资本).{0,5}(?:不低于|不少于|≥|≥)\s*(\d+)\s*(?:万元|亿元)', 'qualification'),
        (r'(?:类似项目).{0,10}(?:业绩|经验).{0,20}(?:不少于|不低于|≥|≥)\s*(\d+)\s*(?:个|项)', 'qualification'),
        (r'(?:项目负责人|项目经理).{0,10}(?:具有|持有).{0,10}(?:证书|资格|职称)', 'qualification'),
    ],
    "technical": [
        # 技术参数
        (r'(?:不低于|不小于|≥|≥)\s*([\d.]+)\s*(?:%|％|mm|cm|m|kg|吨|天|日|月|年)', 'technical'),
        (r'(?:符合|满足).{0,5}(?:GB|GB/T|ISO|JT|CJJ|JGJ)\s*[\d.\-]+', 'technical'),
        (r'(?:质量标准|质量要求|验收标准).{0,5}(?:达到|符合|满足)', 'technical'),
    ],
    "commercial": [
        # 商务条款
        (r'(?:投标有效期).{0,10}(?:不少于|不低于|≥|≥)\s*(\d+)\s*(?:天|日)', 'commercial'),
        (r'(?:质保期|保修期|缺陷责任期).{0,10}(?:不少于|不低于|≥|≥)\s*(\d+)\s*(?:年|月|天)', 'commercial'),
        (r'(?:投标保证金|履约保证金|履约保函).{0,10}(?:不低于|不少于|≥|≥)\s*(\d+)\s*(?:万元|元|％|%)', 'commercial'),
        (r'(?:工期|交货期|服务期).{0,10}(?:不超过|≤|≤)\s*(\d+)\s*(?:天|日|月|年)', 'commercial'),
    ],
    "rejection": [
        # 废标条件
        (r'(?:有下列情形之一的，?)?(?:应当|予以|按).{0,5}(?:废标|否决|拒绝|无效)', 'rejection'),
        (r'(?:不.?接受|拒绝).{0,5}(?:联合体|替代方案|备选方案)', 'rejection'),
        (r'(?:未按要求).{0,10}(?:签字|盖章|密封|装订)', 'rejection'),
        (r'(?:投标报价).{0,10}(?:超过|高于).{0,5}(?:最高限价|预算|控制价)', 'rejection'),
    ],
    "prohibition": [
        # 禁止性条款
        (r'(?:不得|禁止|不允许|严禁).{0,30}(?:转包|分包|挂靠|转让)', 'prohibition'),
        (r'(?:不得|禁止|不允许|严禁).{0,20}(?:串通|围标|弄虚作假)', 'prohibition'),
        (r'(?:关联企业|同一集团|控股关系).{0,20}(?:不得|禁止|不允许)', 'prohibition'),
        (r'(?:低于成本).{0,10}(?:报价|竞标|投标)', 'prohibition'),
    ],
}


def _extract_rules_regex(text: str) -> list[dict]:
    """Regex-based rule extraction as fallback/validation channel.

    Returns list of rule dicts matching the same schema as AI extraction.
    """
    results = []
    seen_texts = set()

    for category, patterns in _REGEX_PATTERNS.items():
        for pattern, cat in patterns:
            for match in re.finditer(pattern, text):
                # Get surrounding context (~100 chars)
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 80)
                context = text[start:end].strip()

                # Simple dedup
                norm = re.sub(r'\s+', '', context)
                if norm in seen_texts:
                    continue
                seen_texts.add(norm)

                results.append({
                    "category": category,
                    "rule_id": f"REGEX-{len(results)+1:03d}",
                    "description": _summarize_rule(context, category),
                    "quantitative": True if re.search(r'\d+', match.group()) else False,
                    "threshold": _extract_threshold(context),
                    "original_text": context,
                    "severity_if_violated": _guess_severity(category, context),
                    "source": "regex",
                })

    return results


def _summarize_rule(context: str, category: str) -> str:
    """Generate a short description from regex-matched context."""
    category_labels = {
        "qualification": "资质要求",
        "technical": "技术规范",
        "commercial": "商务条款",
        "rejection": "废标条件",
        "prohibition": "禁止条款",
    }
    # Try to extract the key phrase
    cleaned = re.sub(r'\s+', ' ', context)
    return f"[{category_labels.get(category, category)}] {cleaned[:80]}"


def _extract_threshold(context: str) -> Optional[str]:
    """Extract numeric threshold from context."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:万元|亿元|元|天|日|月|年|个|项|％|%)', context)
    if m:
        return f"{m.group(1)}{m.group(0)[len(m.group(1)):]}"
    return None


def _guess_severity(category: str, context: str) -> str:
    """Guess violation severity based on category and keywords."""
    if category == "rejection":
        return "critical"
    if category == "prohibition":
        return "critical"
    if "必须" in context or "强制" in context:
        return "critical"
    if "应当" in context or "应" in context:
        return "violation"
    if "宜" in context or "可" in context or "建议" in context:
        return "warning"
    return "violation"


def _extract_rules_ai(doc_text: str, doc_name: str = "") -> list[dict]:
    """AI-based rule extraction using LLM."""
    try:
        from app.services.compliance_prompts import RULE_EXTRACTION_SYSTEM, build_rule_extraction_prompt
        from app.services.llm_provider import create_chat_model
        from app.services.prompt_safety import safe_json_parse
        from langchain_core.messages import SystemMessage, HumanMessage

        prompt = build_rule_extraction_prompt(doc_text, doc_name)
        llm = create_chat_model(streaming=False, temperature=0.2, max_tokens=3000, timeout=120)
        resp = llm.invoke([SystemMessage(content=RULE_EXTRACTION_SYSTEM),
                          HumanMessage(content=prompt)])
        raw = resp.content if hasattr(resp, 'content') else str(resp)

        parsed = safe_json_parse(raw)
        if isinstance(parsed, list):
            for i, rule in enumerate(parsed):
                if isinstance(rule, dict):
                    rule.setdefault("source", "ai")
                    rule.setdefault("rule_id", f"AI-{i+1:03d}")
                    rule.setdefault("category", "technical")
                    rule.setdefault("severity_if_violated", "violation")
            return [r for r in parsed if isinstance(r, dict)]
        elif isinstance(parsed, dict) and "rules" in parsed:
            rules = parsed["rules"]
            for i, rule in enumerate(rules):
                rule.setdefault("source", "ai")
                rule.setdefault("rule_id", f"AI-{i+1:03d}")
            return rules
        logger.warning(f"AI rule extraction returned non-list: {type(parsed)}")
        return []
    except Exception as e:
        logger.error(f"AI rule extraction failed: {e}")
        return []


def _merge_dedup_rules(ai_rules: list[dict], regex_rules: list[dict]) -> list[dict]:
    """Merge AI and regex rules, deduplicate by content similarity, assign final IDs."""
    from difflib import SequenceMatcher

    merged = []
    # Add all AI rules first (higher priority)
    for rule in ai_rules:
        merged.append(rule)

    # Add regex rules if not too similar to existing AI rules
    for r_rule in regex_rules:
        is_dup = False
        r_text = r_rule.get("original_text", "")
        for existing in merged:
            e_text = existing.get("original_text", "")
            if SequenceMatcher(None, r_text, e_text).ratio() > 0.7:
                is_dup = True
                break
        if not is_dup:
            merged.append(r_rule)

    # Assign final rule IDs
    for i, rule in enumerate(merged):
        cat_prefix = {
            "qualification": "Q",
            "technical": "T",
            "commercial": "C",
            "rejection": "R",
            "prohibition": "P",
        }.get(rule.get("category", ""), "X")
        rule["rule_id"] = f"{cat_prefix}-{i+1:03d}"

    return merged


class RuleExtractor:
    """Extract structured rules from bidding documents.

    Dual-channel: AI (primary) + regex (fallback).
    """

    def __init__(self):
        pass

    def extract(self, doc_text: str, doc_name: str = "", use_ai: bool = True) -> dict:
        """Extract rules from a bidding document.

        Args:
            doc_text: full text of the bidding document
            doc_name: filename for logging
            use_ai: whether to use AI extraction (set False for fast regex-only)

        Returns:
            dict with 'rules' (list of rule dicts), 'ai_count', 'regex_count', 'total'
        """
        if not doc_text or len(doc_text.strip()) < 100:
            return {"rules": [], "ai_count": 0, "regex_count": 0, "total": 0, "error": "文档内容过短"}

        ai_rules = []
        regex_rules = _extract_rules_regex(doc_text)

        if use_ai:
            try:
                ai_rules = _extract_rules_ai(doc_text, doc_name)
            except Exception as e:
                logger.warning(f"AI extraction failed, using regex-only: {e}")

        merged = _merge_dedup_rules(ai_rules, regex_rules)

        # Sort by severity then category
        severity_order = {"critical": 0, "violation": 1, "warning": 2, "pass": 3}
        merged.sort(key=lambda r: (
            severity_order.get(r.get("severity_if_violated", "violation"), 1),
            r.get("category", ""),
        ))

        logger.info(
            f"Rule extraction for '{doc_name}': {len(ai_rules)} AI + "
            f"{len(regex_rules)} regex → {len(merged)} merged"
        )

        return {
            "rules": merged,
            "ai_count": len(ai_rules),
            "regex_count": len(regex_rules),
            "total": len(merged),
            "doc_name": doc_name,
        }

    def extract_fast(self, doc_text: str, doc_name: str = "") -> dict:
        """Regex-only fast extraction, no AI call."""
        return self.extract(doc_text, doc_name, use_ai=False)
