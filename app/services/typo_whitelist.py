"""Domain allow-list for typo detection (FIX-015 B4).

Loads `data/domain_words.txt` as a hard allow-list: any token found in the
domain dictionary is a known bidding term and must NEVER be flagged as a typo.

This is deliberately SEPARATE from the similarity stop-word list
(`app/services/stop_words.py`) and from jieba's userdict loading in
`text_utils.py` — three different concerns:
  - stop_words.py      → tokens to IGNORE for TF-IDF similarity
  - text_utils.py      → jieba.load_userdict() segmentation hints
  - typo_whitelist.py  → tokens to never flag as typos (this module)
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DOMAIN_DICT = Path(__file__).resolve().parent.parent.parent / "data" / "domain_words.txt"

_WHITELIST: frozenset[str] | None = None


def _load() -> frozenset[str]:
    global _WHITELIST
    if _WHITELIST is not None:
        return _WHITELIST
    words: set[str] = set()
    try:
        if DOMAIN_DICT.exists():
            for line in DOMAIN_DICT.read_text(encoding="utf-8").splitlines():
                w = line.strip()
                if w:
                    words.add(w)
            logger.info(f"typo whitelist loaded: {DOMAIN_DICT} ({len(words)} words)")
        else:
            logger.warning(f"typo whitelist missing: {DOMAIN_DICT}")
    except Exception as e:
        logger.warning(f"typo whitelist load failed: {e}")
    _WHITELIST = frozenset(words)
    return _WHITELIST


def is_allowed(word: str) -> bool:
    """Return True if the word is a known domain term that must not be flagged."""
    return word in _load()
