"""Plagiarism detection mode (FIX-016 后续).

Two-file comparison focused on "who copied whom" — distinct from the 清标
clearance mode which aggregates 3+ files into a composite score.

Reuses (NO changes to these):
  - _make_vectorizer / tokenize_for_tfidf / DEFAULT_STOP_WORDS
  - remove_template_content
  - compute_similarity_with_numbers (returns sim + blocks_detail + diff HTML)

Independent (this module):
  - paragraph-level aggregation of match blocks
  - dual-signal verdict (cosine + high-match-paragraph ratio)
  - structured JSON report + diff HTML

NOT reused: INDICATOR_WEIGHTS, RiskScorer, composite_score, _detect_component.

Thresholds (locked by regression tests):
  - cosine >= PLAGIARISM_COSINE (0.60)
  - high-match-paragraph ratio >= PLAGIARISM_PARA_RATIO (0.20)
  - a paragraph counts as "high match" if matched-char ratio >= PARA_MATCH_RATIO (0.50)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Thresholds (FIX-016 后续; locked by tests/test_regression.py) ──
PLAGIARISM_COSINE = 0.60
PLAGIARISM_PARA_RATIO = 0.20
PARA_MATCH_RATIO = 0.50
# Minimum paragraph length (chars) to be considered a meaningful unit for
# plagiarism matching. Very short paragraphs (e.g. a 2-char "p1") have inflated
# match ratios (1 matching char = 100%) — exclude them to avoid false positives.
MIN_PARA_CHARS = 20


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by newline. Empty/whitespace paragraphs dropped."""
    out = []
    for p in (text or '').split('\n'):
        if p.strip():
            out.append(p)
    return out


def detect_plagiarism(
    text_a: str,
    text_b: str,
    template_text: Optional[str] = None,
    filename_a: str = '文件A',
    filename_b: str = '文件B',
) -> dict:
    """Compare two documents and produce a plagiarism report.

    Args:
        text_a / text_b: raw document texts
        template_text: optional 招标文件 for template boilerplate removal
        filename_a / filename_b: display names

    Returns:
        dict with doc_a, doc_b, cosine_similarity, verdict, high_match_paragraphs,
        diff_html, thresholds_used. See module docstring.
    """
    from app.services.file_processing import compute_similarity_with_numbers

    # Whole-document cosine + char-level diff blocks
    sim, html_a, html_b, blocks = compute_similarity_with_numbers(text_a, text_b, template_text)

    # ── Paragraph-level aggregation of matched characters ──
    paras_a = _split_paragraphs(text_a)
    # For each paragraph, compute how many of its chars fall inside a match block.
    # blocks entries: {pos1, size, ...} where pos1 is a 0-based offset in text_a.
    para_stats = []
    char_idx = 0
    para_start = 0
    for pa in paras_a:
        para_len = len(pa)
        para_start_char = para_start
        para_end_char = para_start + para_len
        # compute matched chars in this paragraph range
        matched = 0
        for b in blocks:
            b_start = b.get('pos1', 0)
            b_end = b_start + b.get('size', 0)
            overlap = max(0, min(para_end_char, b_end) - max(para_start_char, b_start))
            matched += overlap
        ratio = matched / para_len if para_len else 0.0
        para_stats.append({
            'para_idx': len(para_stats),
            'match_ratio': round(ratio, 4),
            'matched_chars': matched,
            'total_chars': para_len,
        })
        para_start = para_end_char + 1  # +1 for the newline
        char_idx += para_len + 1

    # high-match paragraphs (>= PARA_MATCH_RATIO of chars matched AND long enough
    # to be a meaningful unit — very short paras inflate ratio)
    high_paras = [ps for ps in para_stats
                  if ps['match_ratio'] >= PARA_MATCH_RATIO and ps['total_chars'] >= MIN_PARA_CHARS]
    high_ratio = len(high_paras) / len(para_stats) if para_stats else 0.0

    # ── Dual-signal verdict ──
    cosine_ok = sim >= PLAGIARISM_COSINE
    para_ok = high_ratio >= PLAGIARISM_PARA_RATIO
    if cosine_ok and para_ok:
        verdict = '疑似剽窃'
    elif cosine_ok or (para_ok and len(high_paras) >= 2):
        verdict = '高度相似'
    else:
        verdict = '正常'

    # enrich high paragraphs with snippets from both docs
    high_detail = []
    for ps in high_paras:
        idx = ps['para_idx']
        high_detail.append({
            **ps,
            'snippet_a': paras_a[idx][:120],
            'snippet_b': _para_b_at(text_b, idx, blocks)[:120],
        })

    return {
        'doc_a': filename_a,
        'doc_b': filename_b,
        'cosine_similarity': round(sim, 4),
        'verdict': verdict,
        'high_match_paragraphs': high_detail[:30],
        'high_match_para_count': len(high_paras),
        'para_count': len(para_stats),
        'high_match_para_ratio': round(high_ratio, 4),
        'diff_html': {'a': html_a, 'b': html_b},
        'thresholds_used': {
            'cosine': PLAGIARISM_COSINE,
            'para_ratio': PLAGIARISM_PARA_RATIO,
            'para_match': PARA_MATCH_RATIO,
        },
    }


def _para_b_at(text_b: str, para_idx: int, blocks: list[dict]) -> str:
    """Best-effort: return a text_b paragraph overlapping the match for the given a-paragraph."""
    paras_b = _split_paragraphs(text_b)
    # find the block whose pos1 range intersects; map to b-side j offset
    if para_idx < len(paras_b):
        return paras_b[para_idx]
    return ''
