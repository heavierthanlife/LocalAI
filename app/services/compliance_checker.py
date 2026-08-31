"""Compliance Checker — compares bid documents against extracted rules + applicable laws.

Pipeline:
    1. Load extracted rules (from RuleExtractor)
    2. Load applicable laws (from DB seed + user-uploaded)
    3. Semantic matching: find relevant bid passages for each rule
    4. AI judgment: LLM determines PASS/WARNING/VIOLATION/CRITICAL per rule
    5. Generate summary report

Usage:
    from app.services.compliance_checker import ComplianceChecker
    checker = ComplianceChecker()
    results = checker.check(bid_text, rules, doc_name, applicable_laws)
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

from app.config import DATA_DIR as _DATA_DIR
COMPLIANCE_DIR = os.path.join(str(_DATA_DIR), "compliance_results")
os.makedirs(COMPLIANCE_DIR, exist_ok=True)


def _load_seed_laws() -> list[dict]:
    """Load built-in core laws from seed data."""
    seed_path = os.path.join(str(_DATA_DIR), "laws", "seed_laws.json")
    if not os.path.exists(seed_path):
        logger.warning(f"Seed laws not found at {seed_path}")
        return []
    try:
        with open(seed_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        laws = []
        for law in data:
            for art in law.get("articles", []):
                laws.append({
                    "law_name": law["law_name"],
                    "short_name": law.get("short_name", law["law_name"]),
                    "category": law.get("category", ""),
                    "article": art["article"],
                    "text": art["text"],
                    "tags": art.get("tags", []),
                })
        logger.info(f"Loaded {len(laws)} law articles from seed data ({len(data)} laws)")
        return laws
    except Exception as e:
        logger.error(f"Failed to load seed laws: {e}")
        return []


# Cache seed laws in memory
_SEED_LAWS = None


def _get_seed_laws() -> list[dict]:
    global _SEED_LAWS
    if _SEED_LAWS is None:
        _SEED_LAWS = _load_seed_laws()
    return _SEED_LAWS


def _select_relevant_laws(rules: list[dict], max_laws: int = 15) -> list[dict]:
    """Select applicable law articles based on rule categories and keywords.

    Dual-channel: first try semantic search (ChromaDB + embedding) when available,
    then merge with keyword-based scoring. Falls back to keyword-only if semantic
    unavailable.
    """
    all_laws = _get_seed_laws()
    if not all_laws:
        return []

    # ── Channel 1: semantic law search (ChromaDB) ──
    semantic_hits = []
    try:
        from app.services.law_semantic import semantic_law_search
        # Build a composite query from rule keywords/categories
        cat_text = ' '.join(rule.get('category', '') for rule in rules)
        desc_text = ' '.join((rule.get('description', '') or '')[:100] for rule in rules[:5])
        query = (cat_text + ' ' + desc_text).strip()
        if query:
            semantic_hits = semantic_law_search(query, top_k=max_laws)
    except Exception as e:
        logger.debug(f"Semantic law search unavailable, falling back to keyword: {e}")

    # ── Channel 2: keyword-based scoring (existing) ──
    rule_keywords = set()
    category_to_law_map = {
        "qualification": ["资质", "资格", "条件", "能力"],
        "technical": ["标准", "技术", "质量", "规范"],
        "commercial": ["合同", "价款", "期限", "保证金"],
        "rejection": ["否决", "废标", "无效", "不通过"],
        "prohibition": ["禁止", "不得", "串通", "围标", "弄虚作假"],
    }
    for rule in rules:
        cat = rule.get("category", "")
        rule_keywords.update(category_to_law_map.get(cat, []))
        desc = rule.get("description", "") + rule.get("original_text", "")
        for kw in ["资质", "业绩", "证书", "标准", "质量", "报价", "保证金",
                    "串标", "围标", "虚假", "废标", "否决", "转包", "分包",
                    "签字", "盖章", "有效期", "工期", "质保"]:
            if kw in desc:
                rule_keywords.add(kw)

    scored = []
    for law in all_laws:
        score = 0
        law_text = law["text"]
        for tag in law.get("tags", []):
            for kw in rule_keywords:
                if kw in tag:
                    score += 3
        for kw in rule_keywords:
            if kw in law_text:
                score += 1
        if score > 0:
            scored.append((score, law))

    # ── Merge: semantic hits get a boost, keyword results kept ──
    # Map semantic hits (which have law_name/article/text) to full seed-law shape
    semantic_by_key = {}
    for hit in semantic_hits:
        key = (hit.get('short_name') or hit.get('law_name', ''), hit.get('article', ''))
        semantic_by_key[key] = hit
    scored_with_sem = []
    seen = set()
    for score, law in scored:
        key = (law.get('short_name') or law.get('law_name', ''), law.get('article', ''))
        seen.add(key)
        if key in semantic_by_key:
            # Semantic confirms → boost
            scored_with_sem.append((score + 5, law))
        else:
            scored_with_sem.append((score, law))
    # Add semantic-only hits not already in keyword results
    for key, hit in semantic_by_key.items():
        if key not in seen and hit.get('text'):
            law = {
                'law_name': hit.get('law_name', ''),
                'short_name': hit.get('short_name', ''),
                'category': hit.get('category', ''),
                'article': hit.get('article', ''),
                'text': hit.get('text', ''),
                'tags': [],
            }
            scored_with_sem.append((6, law))
            seen.add(key)

    scored_with_sem.sort(key=lambda x: x[0], reverse=True)
    return [law for _, law in scored_with_sem[:max_laws]]


def _match_rule_to_text(rule: dict, bid_text: str) -> Optional[str]:
    """Find the most relevant passage in bid text for a given rule.

    Uses keyword overlap for fast matching (no embedding needed for this step).
    """
    description = rule.get("description", "")
    original = rule.get("original_text", "")

    # Extract key terms from the rule
    key_terms = set()
    for term in re.findall(r'[\u4e00-\u9fff]{2,}', description + original):
        if len(term) >= 2:
            key_terms.add(term)

    # Common words to ignore
    stop_words = {"不得", "应当", "必须", "可以", "包括", "以及", "或者", "并且",
                  "按照", "根据", "关于", "对于", "进行", "使用", "需要", "一个",
                  "以下", "以上", "所有", "相关", "其他", "这个", "那个"}
    key_terms -= stop_words

    if not key_terms:
        return None

    # Find the best matching passage (~300 chars window)
    best_score = 0
    best_passage = None
    window_size = 300
    step = 100

    for i in range(0, len(bid_text) - window_size, step):
        window = bid_text[i:i + window_size]
        score = sum(1 for term in key_terms if term in window)
        if score > best_score:
            best_score = score
            best_passage = window

    if best_score >= 1:
        return best_passage.strip()
    return None


def _run_ai_compliance_check(
    bid_text: str,
    rules: list[dict],
    bid_doc_name: str,
    applicable_laws: list[dict] = None,
) -> list[dict]:
    """Use LLM to perform compliance check — core judgment engine."""
    try:
        from app.services.compliance_prompts import (
            COMPLIANCE_CHECK_SYSTEM, build_compliance_check_prompt
        )
        from app.services.llm_provider import create_chat_model
        from app.services.prompt_safety import safe_json_parse
        from langchain_core.messages import SystemMessage, HumanMessage

        prompt = build_compliance_check_prompt(rules, bid_text, bid_doc_name, applicable_laws)
        llm = create_chat_model(streaming=False, temperature=0.1, max_tokens=4000, timeout=180)
        resp = llm.invoke([SystemMessage(content=COMPLIANCE_CHECK_SYSTEM),
                          HumanMessage(content=prompt)])
        raw = resp.content if hasattr(resp, 'content') else str(resp)

        parsed = safe_json_parse(raw)
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, dict)]
        if isinstance(parsed, dict) and "results" in parsed:
            return parsed["results"]
        logger.warning(f"AI compliance check returned unexpected type: {type(parsed)}")
        return []
    except Exception as e:
        logger.error(f"AI compliance check failed: {e}", exc_info=True)
        return []


def _fallback_check(rules: list[dict], bid_text: str) -> list[dict]:
    """Regex/keyword-based fallback check when AI is unavailable.

    Returns basic pass/fail based on keyword presence.
    """
    results = []
    bid_lower = bid_text.lower()

    for rule in rules:
        rule_id = rule.get("rule_id", "?")
        desc = rule.get("description", "")

        # Simple keyword-based check
        found = False
        # Extract 2-3 char keywords from description
        keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', desc)
        key_terms = [kw for kw in keywords if kw not in
                     ("不得", "应当", "必须", "可以", "包括", "按照", "根据", "关于",
                      "进行", "需要", "以下", "以上", "所有", "相关", "其他")]
        matches = sum(1 for kw in key_terms if kw in bid_text)
        if key_terms and matches >= len(key_terms) * 0.3:
            found = True

        results.append({
            "rule_id": rule_id,
            "verdict": "PASS" if found else "WARNING",
            "evidence": "关键词匹配检查" if found else "投标文件中未见相关内容",
            "reasoning": f"检出{matches}/{len(key_terms)}个关键词" if found
                else f"未找到相关关键词，请人工复核",
            "suggestion": "" if found else "建议人工核实此项",
            "source": "fallback",
        })

    return results


class ComplianceChecker:
    """Check bid documents against extracted rules and applicable laws.

    Pipeline:
        rules + bid_text + laws → AI judgment → graded results → summary
    """

    def __init__(self):
        pass

    def check(
        self,
        bid_text: str,
        rules: list[dict],
        bid_doc_name: str = "",
        use_ai: bool = True,
        custom_laws: list[dict] = None,
    ) -> dict:
        """Run full compliance check on a bid document.

        Args:
            bid_text: full text of the bid document
            rules: list of rule dicts from RuleExtractor
            bid_doc_name: filename for logging
            use_ai: use AI for judgment (False = keyword fallback)
            custom_laws: user-supplied law articles to check against

        Returns:
            dict with:
                results: list of per-rule verdicts
                summary: {pass, warning, violation, critical}
                laws_applied: list of law articles used
                ai_used: whether AI was used
        """
        # ── Audit logger ──
        from app.services.audit_logger import AuditLogger
        _audit = AuditLogger("compliance_check", bid_doc_name[:40])
        _audit.component("init", status="OK",
                         rule_count=len(rules), bid_chars=len(bid_text),
                         use_ai=use_ai, custom_laws=bool(custom_laws))

        if not bid_text or not rules:
            _audit.result(status="SKIPPED", reason="empty_input")
            return {
                "results": [],
                "summary": {"pass": 0, "warning": 0, "violation": 0, "critical": 0},
                "laws_applied": [],
                "ai_used": False,
            }

        # Select relevant laws
        applicable_laws = _select_relevant_laws(rules)
        if custom_laws:
            applicable_laws.extend(custom_laws)
        _audit.component("law_selection", status="OK",
                         laws_matched=len(applicable_laws),
                         laws=str([l.get('article','?') for l in applicable_laws[:5]]))

        # Run AI check
        results = []
        if use_ai:
            try:
                results = _run_ai_compliance_check(bid_text, rules, bid_doc_name, applicable_laws)
            except Exception as e:
                logger.warning(f"AI check failed, using fallback: {e}")
                results = []

        # Fallback if AI failed or returned nothing
        if not results:
            results = _fallback_check(rules, bid_text)
            use_ai = False

        # Ensure all rules have a result
        rule_ids_checked = {r.get("rule_id") for r in results}
        for rule in rules:
            rid = rule.get("rule_id")
            if rid not in rule_ids_checked:
                # Try keyword match
                passage = _match_rule_to_text(rule, bid_text)
                results.append({
                    "rule_id": rid,
                    "verdict": "WARNING",
                    "evidence": passage or "投标文件中未见相关内容",
                    "reasoning": "AI未明确判定此项，建议人工复核",
                    "suggestion": "请人工确认此项合规性",
                    "source": "unchecked",
                })

        # Sort by severity
        severity_order = {"CRITICAL": 0, "VIOLATION": 1, "WARNING": 2, "PASS": 3}
        results.sort(key=lambda r: severity_order.get(r.get("verdict", "PASS"), 3))

        # Summary
        summary = {"pass": 0, "warning": 0, "violation": 0, "critical": 0}
        for r in results:
            v = r.get("verdict", "PASS").lower()
            if v in summary:
                summary[v] += 1

        logger.info(
            f"Compliance check for '{bid_doc_name}': "
            f"P:{summary['pass']} W:{summary['warning']} "
            f"V:{summary['violation']} C:{summary['critical']} "
            f"(AI={use_ai}, laws={len(applicable_laws)})"
        )

        _audit.result(
            status="OK",
            pass_c=summary['pass'], warn_c=summary['warning'],
            viol_c=summary['violation'], crit_c=summary['critical'],
            ai_used=use_ai, laws_applied=len(applicable_laws),
            total_rules=len(rules),
        )

        return {
            "results": results,
            "summary": summary,
            "laws_applied": [
                {"article": l["article"], "law_name": l["law_name"], "text": l["text"][:200]}
                for l in applicable_laws
            ],
            "ai_used": use_ai,
        }

    def generate_report(
        self,
        check_results: list[dict],
        bid_doc_name: str,
        rule_count: int,
        use_ai: bool = True,
    ) -> str:
        """Generate HTML summary report from check results."""
        try:
            from app.services.compliance_prompts import (
                COMPLIANCE_SUMMARY_SYSTEM, build_summary_prompt
            )
            from app.services.llm_provider import create_chat_model
            from langchain_core.messages import SystemMessage, HumanMessage

            if not use_ai:
                return _generate_fallback_report(check_results, bid_doc_name, rule_count)

            prompt = build_summary_prompt(check_results, bid_doc_name, rule_count)
            llm = create_chat_model(streaming=False, temperature=0.3, max_tokens=2000, timeout=120)
            resp = llm.invoke([SystemMessage(content=COMPLIANCE_SUMMARY_SYSTEM),
                              HumanMessage(content=prompt)])
            return resp.content if hasattr(resp, 'content') else str(resp)
        except Exception as e:
            logger.warning(f"Report generation failed, using fallback: {e}")
            return _generate_fallback_report(check_results, bid_doc_name, rule_count)


def _generate_fallback_report(results: list[dict], doc_name: str, rule_count: int) -> str:
    """Generate a simple HTML report without AI."""
    pass_c = sum(1 for r in results if r.get('verdict') == 'PASS')
    warn_c = sum(1 for r in results if r.get('verdict') == 'WARNING')
    viol_c = sum(1 for r in results if r.get('verdict') == 'VIOLATION')
    crit_c = sum(1 for r in results if r.get('verdict') == 'CRITICAL')

    status_color = "#27ae60" if crit_c == 0 and viol_c == 0 else "#e67e22" if crit_c == 0 else "#e74c3c"
    status_text = "合规" if crit_c == 0 and viol_c == 0 else "存在违规项" if crit_c == 0 else "存在严重违规"

    html = f"""<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.6">
<h3 style="color:{status_color}">合规审查报告 — {doc_name}</h3>
<div style="background:#f8f9fa;border-radius:8px;padding:16px;margin:12px 0">
  <strong>规则总数：</strong>{rule_count} &nbsp;|&nbsp;
  <span style="color:#27ae60">通过：{pass_c}</span> &nbsp;|&nbsp;
  <span style="color:#f39c12">警告：{warn_c}</span> &nbsp;|&nbsp;
  <span style="color:#e67e22">违规：{viol_c}</span> &nbsp;|&nbsp;
  <span style="color:#e74c3c">严重违规：{crit_c}</span>
</div>
<p><strong>总体判定：</strong><span style="color:{status_color}">{status_text}</span></p>"""

    if crit_c > 0 or viol_c > 0:
        html += "<h4 style='color:#e74c3c'>关键违规项：</h4><ul>"
        for r in results:
            if r.get('verdict') in ('VIOLATION', 'CRITICAL'):
                color = "#e74c3c" if r['verdict'] == 'CRITICAL' else "#e67e22"
                html += f"""<li style="color:{color};margin:8px 0">
  <strong>[{r['verdict']}] {r.get('rule_id', '?')}</strong>: {r.get('reasoning', '')}
  <br><small>证据: {r.get('evidence', '')[:200]}</small>
  <br><small>建议: {r.get('suggestion', '请人工复核')}</small>
</li>"""
        html += "</ul>"

    if warn_c > 0:
        html += "<h4 style='color:#f39c12'>警告项：</h4><ul>"
        for r in results:
            if r.get('verdict') == 'WARNING':
                html += f"<li style='color:#f39c12;margin:4px 0'>[{r.get('rule_id', '?')}] {r.get('reasoning', '')}</li>"
        html += "</ul>"

    html += """<p style="margin-top:16px;color:#7f8c8d;font-size:0.85em">
⚠️ 本报告由系统自动生成，仅供参考。建议人工复核所有违规项。</p></div>"""
    return html


# ── Celery Task ──

def _save_check_result(task_id: str, data: dict):
    path = os.path.join(COMPLIANCE_DIR, f"{task_id}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)


try:
    from celery_app import celery as celery_app

    @celery_app.task(bind=True, name='compliance_check_task', max_retries=1)
    def compliance_check_task(self, task_id: str, bid_text: str, rules: list,
                               bid_name: str, use_ai: bool = True,
                               include_laws: bool = True) -> dict:
        """Celery task: run full compliance check in background."""
        from app.services.task_bus import TaskBus

        bus = TaskBus()
        bus.start(task_id, 'compliance_check', f'合规检查: {bid_name}')

        try:
            bus.progress(10, '正在加载适用法规...')
            checker = ComplianceChecker()

            bus.progress(30, '正在逐条检查投标文件...')
            result = checker.check(bid_text, rules, bid_name, use_ai=use_ai)

            bus.progress(70, '正在生成审查报告...')
            report_html = checker.generate_report(
                result["results"], bid_name, len(rules), use_ai=use_ai
            )

            output = {
                "success": True,
                "bid_name": bid_name,
                "rule_count": len(rules),
                "summary": result["summary"],
                "results": result["results"],
                "laws_applied": result.get("laws_applied", []),
                "report_html": report_html,
                "ai_used": result.get("ai_used", False),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
            _save_check_result(task_id, output)

            bus.progress(100, '合规检查完成')
            bus.complete(result={
                "summary": result["summary"],
                "rule_count": len(rules),
            })
            return output

        except Exception as e:
            logger.error(f"Compliance check task failed: {e}", exc_info=True)
            bus.fail(str(e)[:200])
            _save_check_result(task_id, {
                "success": False,
                "error": str(e)[:500],
                "checked_at": datetime.now(timezone.utc).isoformat(),
            })
            raise

except ImportError:
    # Celery not available — sync fallback used instead
    pass
