"""Document type classifier.

Two-layer strategy:
  1. Heuristic rules (cheap) — covers ~70% of documents
  2. LLM fine-grained classification (expensive) — for ambiguous cases

Wiki directory taxonomy:
  laws/        — legislative text with chapter/article markers
  regulations/ — industry standards, technical specifications (GB/T, ISO, etc.)
  cases/       — bid cases (success/failure), legal cases
  templates/   — document templates with multi-level headings
  documents/   — general business documents, project files
  general/     — fallback for everything else

Bid priority rule: if both law and bid markers are present,
classify as bid (bid documents routinely cite laws, the reverse is rare).
"""

from __future__ import annotations

import re
import hashlib
import logging
from typing import Tuple

from .document_schema import DOC_LAW, DOC_BID, DOC_TEMPLATE, DOC_GENERAL

logger = logging.getLogger(__name__)

_LAW_PATTERN = re.compile(r"第[一二三四五六七八九十百千\d]+[条章]", re.UNICODE)
_BID_PATTERN = re.compile(
    r"招标|投标|开标|评标|中标|资质要求|技术规格|资格预审|投标人须知|合同条款",
    re.UNICODE,
)
_TEMPLATE_PATTERN = re.compile(
    r"^(?:第[一二三四五六七八九十百千\d]+[章节条]|"
    r"[一二三四五六七八九十]+[、，．.]|"
    r"\d+[\.\、]|"
    r"（[一二三四五六七八九十]+）)",
    re.MULTILINE | re.UNICODE,
)
_REGULATION_PATTERN = re.compile(
    r"GB/T\s*\d+|GB\s*\d+|ISO\s*\d+|IEC\s*\d+|"
    r"行业标准|国家标准|技术规范|技术标准|"
    r"标准号|标准编号|规范要求|操作规程",
    re.UNICODE,
)

_LLM_CLASSIFY_CATEGORY_CACHE: dict[str, str] = {}


def classify(text: str, filename: str = "") -> str:
    """Classify document type from text and optional filename.

    Returns one of: law, bid, template, general.
    """
    if not text:
        return DOC_GENERAL

    sample = text[:3000]

    law_match = bool(_LAW_PATTERN.search(sample))
    bid_match = bool(_BID_PATTERN.search(sample))
    template_match = bool(_TEMPLATE_PATTERN.search(sample))

    if law_match and bid_match:
        logger.debug(f"bid priority over law for: {filename}")
        return DOC_BID

    if bid_match:
        return DOC_BID

    if law_match:
        return DOC_LAW

    if template_match:
        return DOC_TEMPLATE

    return DOC_GENERAL


_WIKI_CATEGORY_MAP: dict[str, str] = {
    DOC_LAW: "laws",
    DOC_BID: "documents",
    DOC_TEMPLATE: "templates",
    DOC_GENERAL: "general",
}


def doc_to_wiki_category(doc_type: str) -> str:
    """Map internal document type to wiki directory name.

    Returns one of: laws, regulations, cases, templates, documents, general.
    """
    return _WIKI_CATEGORY_MAP.get(doc_type, "general")


def classify_and_categorize(text: str, filename: str = "", file_hash: str = "") -> Tuple[str, str]:
    """Two-layer classification: heuristic first, LLM refinement for ambiguous cases.

    Layer 1: heuristic classify() — instant, covers ~70% of uploads
    Layer 2: if DOC_GENERAL and text > 200 chars, call LLM for fine-grained category
             (laws/regulations/cases/templates/documents/general)

    Cache by file_hash in memory to avoid re-classifying the same content.

    Returns:
        (doc_type, wiki_category) — e.g. ("law", "laws") or ("general", "cases")
    """
    if not text:
        return DOC_GENERAL, "general"

    cache_key = file_hash or hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    if cache_key in _LLM_CLASSIFY_CATEGORY_CACHE:
        doc_type, wiki_cat = _LLM_CLASSIFY_CATEGORY_CACHE[cache_key]
        return doc_type, wiki_cat

    doc_type = classify(text, filename)

    if doc_type != DOC_GENERAL:
        wiki_bucket = doc_to_wiki_category(doc_type)
        _LLM_CLASSIFY_CATEGORY_CACHE[cache_key] = (doc_type, wiki_bucket)
        return doc_type, wiki_bucket

    wiki_category = "general"
    if text and len(text) > 200:
        try:
            wiki_category = _llm_classify_category(text[:2000], filename)
        except Exception:
            wiki_category = "general"

    if wiki_category == "general":
        sample = text[:3000]
        if _REGULATION_PATTERN.search(sample):
            wiki_category = "regulations"
        elif _BID_PATTERN.search(sample):
            wiki_category = "documents"

    _LLM_CLASSIFY_CATEGORY_CACHE[cache_key] = (doc_type, wiki_category)
    return doc_type, wiki_category


_WIKI_CATEGORY_PROMPT = """判断以下文档属于哪个类别，只输出类别名称：

- laws: 法律、法规、条例、办法、实施细则
- regulations: 行业标准、技术规范、GB/T、ISO、操作规程
- cases: 中标案例、废标案例、违规案例、诉讼案例
- templates: 招标文件模板、投标文件模板、合同模板、格式范本
- documents: 项目文档、技术方案、业务文件
- general: 其他无法归类的内容

文件名：{filename}
内容（前2000字）：
{content}

只输出一个类别名称（laws/regulations/cases/templates/documents/general），不要解释。"""


def _llm_classify_category(text: str, filename: str = "") -> str:
    from app.services.llm_provider import call_llm

    prompt = _WIKI_CATEGORY_PROMPT.format(filename=filename, content=text)
    result = call_llm(user_prompt=prompt, temperature=0, max_tokens=10).strip().lower()

    valid = {"laws", "regulations", "cases", "templates", "documents", "general"}
    if result not in valid:
        result = "general"
    return result
