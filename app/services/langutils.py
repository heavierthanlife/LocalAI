"""Language detection utilities for model switching in semantic comparison."""
import re
import logging

logger = logging.getLogger(__name__)

# Chinese character range
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
# Mostly ASCII letters (English/latin)
_LATIN_RE = re.compile(r'[a-zA-Z]')


def detect_language(text: str) -> str:
    """Detect the primary language of a text.
    
    Returns:
        'zh' — predominantly Chinese (simplified)
        'en' — predominantly English/Latin
        'mixed' — balanced mix, defaults to en/multilingual
    """
    if not text:
        return 'en'
    
    cjk_count = len(_CJK_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))
    total_chars = len(text.strip())
    
    if total_chars == 0:
        return 'en'
    
    cjk_ratio = cjk_count / total_chars
    latin_ratio = latin_count / total_chars
    
    # If > 30% of chars are CJK, treat as Chinese
    if cjk_ratio > 0.30:
        return 'zh'
    # If > 50% Latin and very little CJK, treat as English
    if latin_ratio > 0.50 and cjk_ratio < 0.10:
        return 'en'
    # Mixed or unknown — default to English/multilingual
    return 'en'


def detect_pair_language(text1: str, text2: str) -> str:
    """Detect the language for a pair of texts.
    
    If both are the same language, use that model.
    If mixed, use multilingual (en) model.
    """
    lang1 = detect_language(text1)
    lang2 = detect_language(text2)
    
    logger.debug(f"Language detection: text1={lang1}, text2={lang2}")
    
    if lang1 == lang2:
        return lang1
    # Mixed: one zh one en — use multilingual model
    return 'en'
