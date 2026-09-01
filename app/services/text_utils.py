"""Shared Chinese text tokenization utility using jieba with domain dictionary.

Provides a centralized tokenizer that:
- Loads the domain dictionary (data/domain_words.txt) once at import
- Offers both cut (generator) and lcut (list) for word segmentation
- Offers POS tagging via jieba.posseg
- Provides a custom tokenizer callable for TfidfVectorizer
"""

import os
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DOMAIN_DICT = DATA_DIR / "domain_words.txt"
_jieba_loaded = False
_posseg_available = False


def _init_jieba():
    """Lazy-init jieba with domain dictionary. Called once on first use."""
    global _jieba_loaded, _posseg_available
    if _jieba_loaded:
        return
    try:
        # Suppress pkg_resources deprecation warning (triggered at import time)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pkg_resources.*")
            import jieba
        if DOMAIN_DICT.exists():
            jieba.load_userdict(str(DOMAIN_DICT))
            logger.info(f"jieba domain dict loaded: {DOMAIN_DICT}")
        else:
            logger.info("jieba initialized (no domain dict)")
        _jieba_loaded = True
        try:
            import jieba.posseg as pseg
            _posseg_available = True
        except ImportError:
            logger.warning("jieba.posseg not available — POS tagging disabled")
    except ImportError:
        logger.error("jieba not installed — Chinese tokenization unavailable")


def cut(text: str):
    """Segment Chinese text into words (generator). Falls back to whitespace split."""
    _init_jieba()
    if not _jieba_loaded:
        for w in text.split():
            yield w
        return
    import jieba
    for w in jieba.cut(text):
        w = w.strip()
        if w:
            yield w


def lcut(text: str) -> list[str]:
    """Segment Chinese text into a list of words."""
    return list(cut(text))


def posseg(text: str) -> list[tuple[str, str]] | None:
    """POS-tag Chinese text. Returns list of (word, flag) tuples, or None if unavailable."""
    _init_jieba()
    if not _posseg_available:
        return None
    import jieba.posseg as pseg
    return [(w.word, w.flag) for w in pseg.cut(text) if w.word.strip()]


def tokenize_for_tfidf(text: str, min_len: int = 2, stop_words: set = None) -> str:
    """Segment text and return space-joined string for TfidfVectorizer.

    Filters out single-character tokens (mostly punctuation/determiners) and
    high-frequency stop words (FIX-013) so shared tender boilerplate doesn't
    inflate similarity. Defaults to the project's DEFAULT_STOP_WORDS.
    """
    if stop_words is None:
        from app.services.stop_words import DEFAULT_STOP_WORDS
        stop_words = DEFAULT_STOP_WORDS
    words = [w for w in lcut(text) if len(w) >= min_len]
    if stop_words:
        words = [w for w in words if w not in stop_words]
    return ' '.join(words)


CN_CHAR = re.compile(r'[\u4e00-\u9fff]')


def has_chinese(text: str) -> bool:
    """Check if text contains any Chinese characters."""
    return bool(CN_CHAR.search(text))


def is_mostly_chinese(text: str, threshold: float = 0.3) -> bool:
    """Check if a significant portion of text is Chinese characters."""
    if not text:
        return False
    cn_count = len(CN_CHAR.findall(text))
    total = len(text.replace(' ', '').replace('\n', ''))
    return total > 0 and cn_count / total >= threshold


def word_count(text: str) -> int:
    """Count Chinese words (using jieba) + English words (whitespace split)."""
    words = lcut(text)
    return len(words)


def top_keywords(text: str, top_k: int = 20, min_len: int = 2) -> list[tuple[str, int]]:
    """Extract top-k Chinese keywords with frequencies using jieba."""
    from collections import Counter
    words = [w for w in lcut(text) if CN_CHAR.search(w) and min_len <= len(w) <= 10]
    return Counter(words).most_common(top_k)
