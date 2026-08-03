"""Unified document parser — converts structured paragraphs into intermediate sections.

Pipeline: file bytes → structured paragraphs → sections[] → consumer adapters

Integrates law_parser patterns for chapter/article detection and docx heading levels.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.services.document_schema import (
    make_section, make_document, DOC_LAW, DOC_BID, DOC_TEMPLATE, DOC_GENERAL,
)
from app.services.document_classifier import classify
from app.services.law_parser import _is_chapter_line, _is_article_line

logger = logging.getLogger(__name__)


def build_document(
    paragraphs: List[dict],
    filename: str = "",
    doc_type: str = "",
) -> dict:
    """Convert structured paragraphs → intermediate document format.

    Args:
        paragraphs: List of {text, level, style} dicts from extract_structured_text()
        filename: Optional filename for classifier
        doc_type: Override classifier. If empty, auto-detect from text content.

    Returns:
        dict: {title, type, sections[], metadata}
    """
    if not paragraphs:
        return make_document()

    full_text = "\n".join(p["text"] for p in paragraphs)

    if not doc_type:
        doc_type = classify(full_text, filename)

    title = paragraphs[0]["text"] if paragraphs else ""

    if doc_type == DOC_LAW:
        sections = _build_law_sections(paragraphs)
    elif doc_type in (DOC_BID, DOC_TEMPLATE):
        sections = _build_heading_sections(paragraphs)
    else:
        sections = _build_general_sections(paragraphs)

    if not sections:
        sections = _build_general_sections(paragraphs)

    return make_document(
        title=title,
        doc_type=doc_type,
        sections=sections,
        metadata={"filename": filename, "paragraph_count": len(paragraphs)},
    )


def _build_law_sections(paragraphs: List[dict]) -> List[dict]:
    """Build chapter/article hierarchy from law text.

    Uses law_parser markers: 第X章, 第X条, 附则.
    """
    sections: List[dict] = []
    sec_counter = 0
    current_chapter = None
    current_article = None

    for para in paragraphs:
        text = para["text"]
        stripped = text.strip()

        if _is_chapter_line(stripped):
            sec_counter += 1
            current_chapter = make_section(
                f"s{sec_counter}",
                title=stripped,
                level=1,
            )
            sections.append(current_chapter)
            current_article = None
            continue

        if _is_article_line(stripped):
            sec_counter += 1
            current_article = make_section(
                f"s{sec_counter}",
                title=stripped,
                level=2,
                parent_id=current_chapter["id"] if current_chapter else None,
            )
            sections.append(current_article)
            continue

        if current_article:
            if current_article["content"]:
                current_article["content"] += "\n" + text
            else:
                current_article["content"] = text
        elif current_chapter:
            if current_chapter["content"]:
                current_chapter["content"] += "\n" + text
            else:
                current_chapter["content"] = text

    return sections


def _build_heading_sections(paragraphs: List[dict]) -> List[dict]:
    """Build sections from docx heading levels (bid/template).

    Uses para.level from structured extraction to create sections.
    Non-heading paragraphs are appended to the last section.
    """
    sections: List[dict] = []
    sec_counter = 0
    current_section = None
    parent_stack: List[dict] = []
    section_type_map = {1: "chapter", 2: "article", 3: "clause"}

    for para in paragraphs:
        text = para["text"]
        stripped = text.strip()
        level = para.get("level", 0)

        if para.get("style") == "heading" and level > 0:
            sec_counter += 1
            while parent_stack and parent_stack[-1]["level"] >= level:
                parent_stack.pop()
            parent_id = parent_stack[-1]["id"] if parent_stack else None

            current_section = make_section(
                f"s{sec_counter}",
                title=stripped,
                level=level,
                parent_id=parent_id,
                section_type=section_type_map.get(level, ""),
            )
            sections.append(current_section)
            parent_stack.append(current_section)
        elif current_section:
            if current_section["content"]:
                current_section["content"] += "\n" + text
            else:
                current_section["content"] = text
        else:
            # No heading yet — first paragraph as root
            sec_counter += 1
            current_section = make_section(
                f"s{sec_counter}",
                title=stripped[:60],
                level=1,
                content=text,
            )
            sections.append(current_section)
            parent_stack = [current_section]

    return sections


def _build_general_sections(paragraphs: List[dict]) -> List[dict]:
    """Build flat paragraph sections for general documents.

    Each paragraph becomes a section, no hierarchy.
    """
    sections: List[dict] = []
    for i, para in enumerate(paragraphs):
        sections.append(make_section(
            f"s{i + 1}",
            title=para["text"][:60],
            content=para["text"],
            level=1,
        ))
    return sections


# ── Consumer adapters ──


def to_wiki_markdown(doc: dict) -> str:
    """Convert intermediate document to wiki-compatible markdown.

    Uses ## headings for articles/sections, # for top-level chapters.
    """
    lines: list[str] = []
    title = doc.get("title", "")
    if title:
        lines.append(f"# {title}\n")

    for sec in doc.get("sections", []):
        level = sec.get("level", 1)
        stitle = sec.get("title", "")
        content = sec.get("content", "")

        if level == 1:
            lines.append(f"\n# {stitle}\n")
        else:
            lines.append(f"\n## {stitle}\n")

        if content:
            lines.append(content)

    return "\n".join(lines)


def to_template_sections(doc: dict) -> list[dict]:
    """Convert intermediate document to bid_templates.sections format.

    Matches the existing JSONB format: {id, title, content, level, order}.
    """
    result: list[dict] = []
    for i, sec in enumerate(doc.get("sections", [])):
        result.append({
            "id": sec.get("id", f"sec-{i + 1}"),
            "title": sec.get("title", ""),
            "content": sec.get("content", ""),
            "level": sec.get("level", 1),
            "order": i + 1,
        })
    return result


def build_section_id(level: int, chapter_order: int, article_order: int) -> str:
    if level == 1:
        return str(chapter_order)
    elif level == 2:
        return f"{chapter_order}-{article_order}"
    else:
        return f"0-{article_order}"


def to_articles(doc: dict) -> list[dict]:
    """Convert intermediate document to flat article list with section references.

    Section IDs use 3-15 format: level 1 → "3", level 2 → "3-15", other → "0-N".
    """
    articles: list[dict] = []
    ch_order = 0
    art_order = 0
    standalone = 0

    for i, sec in enumerate(doc.get("sections", [])):
        level = sec.get("level", 1)
        if level == 1:
            ch_order += 1
            art_order = 0
        elif level == 2:
            art_order += 1
        else:
            standalone += 1

        sec_id = build_section_id(level, ch_order, art_order if level <= 2 else standalone)

        articles.append({
            "id": sec_id,
            "title": sec.get("title", ""),
            "content": sec.get("content", ""),
            "level": level,
            "parent_id": sec.get("parent_id"),
            "order": i + 1,
        })
    return articles


def to_chunked_text(doc: dict, max_chars: int = 8000) -> str:
    """Convert intermediate document to markdown text with section headings.

    Truncates at max_chars with a trailing [...] indicator.
    Used by chat adapter to preserve heading hierarchy in LLM context.
    """
    sections = doc.get("sections", [])
    if not sections:
        return doc.get("title", "")

    total = len(sections)
    parts: list[str] = []
    for s in sections:
        lvl = min(s.get("level", 1), 6)
        prefix = "#" * lvl
        parts.append(f"{prefix} {s['title']}\n{s.get('content', '')}")

    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[... 已截断，共 {total} 节]"
    return text


def process_file(file_storage, filename: str = "") -> dict:
    """Full pipeline: file bytes → structured paragraphs → sections → intermediate doc.

    Convenience function that runs extraction + classification + parsing.
    """
    from app.services.file_processing import extract_structured_text

    paragraphs = extract_structured_text(file_storage)
    if not paragraphs:
        return make_document()

    return build_document(paragraphs, filename=filename or file_storage.filename)
