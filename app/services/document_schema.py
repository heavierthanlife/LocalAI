"""Intermediate document format schema.

Flat section array with parent_id linking — compatible with:
  - bid_templates.sections (JSONB, flat array)
  - markdown-it frontend rendering (flat ## heading stream)
  - law_parser output (chapter → ##, article → ##)

Each section:
  { id, title, content, level, parent_id?, type? }
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Document type constants ──
DOC_LAW = "law"
DOC_BID = "bid"
DOC_TEMPLATE = "template"
DOC_GENERAL = "general"


def make_section(
    section_id: str,
    title: str = "",
    content: str = "",
    level: int = 1,
    parent_id: Optional[str] = None,
    section_type: str = "",
) -> dict:
    """Create a normalized section dict."""
    return {
        "id": section_id,
        "title": title.strip(),
        "content": content.strip(),
        "level": level,
        "parent_id": parent_id,
        "type": section_type,
    }


def make_document(
    title: str = "",
    doc_type: str = DOC_GENERAL,
    sections: Optional[list[dict]] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Create the intermediate document envelope."""
    return {
        "title": title.strip(),
        "type": doc_type,
        "sections": sections or [],
        "metadata": metadata or {},
    }
