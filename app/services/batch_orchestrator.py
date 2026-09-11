"""Batch audit orchestrator — pure business logic extracted from routes.

Provides:
  - RiskScorer:  configurable risk formula (snapshot-locked via W1a tests)
  - Pair comparison / clustering / gang detection helpers used by 清标.
"""


import os
import html
import json
import logging
from datetime import datetime, timezone
from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

from docx import Document as DocxDocument
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from app.services.batch_compare_svc import (
    _precompute_tfidf_for_files,
    _compute_pair_similarity_from_matrix,
)
from app.services.file_processing import (
    preprocess_text_for_similarity,
    keyword_overlap_similarity,
    extract_keywords,
    compute_similarity_with_numbers,
    remove_template_content,
    truncate_filename,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Risk scoring
# ═══════════════════════════════════════════════════════════════════════════

class RiskScorer:
    """Configurable risk scoring formula.

    FIX-2026-09-04-QA-B1: paragraph-level substantive collusion (collusion_para)
    is now the primary text signal; whole-document text_sim is down-weighted to
    0.10 (it averages away the real near-verbatim service-commitment evidence) and
    is fully zeroed when the tender file is missing (template overlap ≠ collusion).
    Snapshot locked via tests/test_batch_orchestrator.py::test_snapshot_risk_formula.
    """

    WEIGHTS = {
        "key_info": 0.30,
        "file_attr": 0.30,
        "text_sim": 0.10,
        "collusion_para": 0.30,  # paragraph-level near-verbatim substantive segments
        "image_sim": 0.0,   # image similarity disabled in clearance (images=[])
    }

    # text_sim only contributes when cosine similarity ≥ this gate (avoids
    # template/boilerplate overlap being scored as collusion)
    TEXT_SIM_GATE = 0.80

    @classmethod
    def compute(cls, key_info_pct: float, file_attr_val: float,
                text_sim_pct: float, img_sim_val: float,
                collusion_para: float = 0.0, template_missing: bool = False) -> float:
        # text_sim only counts if it clears the ≥80% gate AND the tender file was
        # available for template removal — otherwise raw cosine is template overlap.
        text_eff = text_sim_pct if (not template_missing and text_sim_pct >= cls.TEXT_SIM_GATE * 100) else 0.0
        return (
            cls.WEIGHTS["key_info"] * key_info_pct +
            cls.WEIGHTS["file_attr"] * file_attr_val +
            cls.WEIGHTS["text_sim"] * text_eff +
            cls.WEIGHTS["collusion_para"] * collusion_para +
            cls.WEIGHTS["image_sim"] * img_sim_val
        )


# ═══════════════════════════════════════════════════════════════════════════
# Pairwise comparison core
# ═══════════════════════════════════════════════════════════════════════════

def _detect_component(filename: str, text: str = '') -> str:
    """Detect bid-document component type from filename + leading text (FIX-014).

    Returns 'price' / 'tech' / 'commercial' / 'unknown'.  Only used to guard
    cross-component comparisons (价格标 vs 技术标 are not comparable via text).
    """
    s = (str(filename or '') + ' ' + (text[:300] if text else '')).lower()
    if any(k in s for k in ('价格标', '报价', '商务报价', '报价表', '开标一览表')):
        return 'price'
    if any(k in s for k in ('技术标', '技术方案', '施工组织设计', '技术部分')):
        return 'tech'
    if any(k in s for k in ('商务标', '商务部分', '资格标', '资质文件')):
        return 'commercial'
    return 'unknown'


def compute_single_pair(file_data, i, j, check_items, tfidf_matrix=None,
                        template_text=None, collusion_para=0.0, template_missing=False,
                        extra_stop_words=None):
    """Compute a single file-pair's similarity metrics."""
    text1 = file_data[i]['text']
    text2 = file_data[j]['text']
    meta1 = file_data[i].get('metadata') or {}
    meta2 = file_data[j].get('metadata') or {}
    images1 = file_data[i].get('images') or []
    images2 = file_data[j].get('images') or []

    from app.services.file_processing import image_similarity, file_attr_similarity

    # Component guard (FIX-014): 异组件（价格标↔技术标）文本相似度无统计可比性 → 归零
    comp1 = _detect_component(file_data[i].get('filename', ''), text1)
    comp2 = _detect_component(file_data[j].get('filename', ''), text2)
    component_mismatch = (comp1 != comp2 and comp1 != 'unknown' and comp2 != 'unknown')

    # Image similarity
    img_sim = image_similarity(images1, images2) if check_items.get('image_sim', True) else 0.0

    # Text similarity (TF-IDF cosine)
    if check_items.get('text_sim', True) and tfidf_matrix is not None:
        sim = _compute_pair_similarity_from_matrix(tfidf_matrix, i, j)
    else:
        sim = 0.0

    # Key info overlap
    if check_items.get('key_info', True):
        t1 = preprocess_text_for_similarity(text1, template_text, extra_stop_words=extra_stop_words)
        t2 = preprocess_text_for_similarity(text2, template_text, extra_stop_words=extra_stop_words)
        if template_text:
            t1 = remove_template_content(t1, template_text)
            t2 = remove_template_content(t2, template_text)
        key_sim = keyword_overlap_similarity(t1, t2)
    else:
        key_sim = 0.0

    # File attribute similarity
    if check_items.get('file_attr', True) and meta1 and meta2:
        attr_sim = file_attr_similarity(meta1, meta2)
    else:
        attr_sim = 0.0

    text_sim_val = sim * 100
    key_info_val = key_sim * 100
    file_attr_val = attr_sim
    img_sim_val = img_sim

    if component_mismatch:
        # Z-1: 异组件文本/关键词重叠均无统计可比性 → 归零（结构性条款重叠不作串标依据）
        text_sim_val = 0.0
        key_info_val = 0.0

    risk = RiskScorer.compute(key_info_val, file_attr_val, text_sim_val, img_sim_val,
                              collusion_para=collusion_para, template_missing=template_missing)

    _, html1, html2, blocks = compute_similarity_with_numbers(text1, text2, template_text)

    return {
        'i': i, 'j': j,
        'name1': file_data[i]['filename'],
        'name2': file_data[j]['filename'],
        'text1': text1, 'text2': text2,
        'sim': sim * 100,
        'key_sim': key_info_val,
        'attr_sim': file_attr_val,
        'risk': risk,
        'blocks': blocks,
        'html1': html1, 'html2': html2,
        'used_weights': {},
        'attr_same': 1 if meta1.get('author') and meta1['author'] == meta2.get('author') else 0,
        'component_mismatch': component_mismatch,
        'components': [comp1, comp2],
        'collusion_para': collusion_para,
    }


def compute_all_pairs(file_data, check_items, tfidf_matrix=None, template_text=None,
                      collusion_para_map=None, template_missing=False,
                      extra_stop_words=None):
    """Run pairwise comparison for all file pairs.

    collusion_para_map: {(i,j): 0-100 paragraph-collusion score} — threaded into
    the RiskScorer (FIX-2026-09-04-QA-B1).  template_missing: when True the
    whole-document text_sim contributes 0 (no tender file → template overlap).
    extra_stop_words: industry word tables merged into key_info preprocessing
    (FIX-2026-09-07-QA-C2) so 行业通用词 don't inflate keyword overlap.
    """
    n = len(file_data)
    pairs = []
    risk_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cp = 0.0
            if collusion_para_map:
                cp = float(collusion_para_map.get((i, j), collusion_para_map.get((j, i), 0.0)) or 0.0)
            pair = compute_single_pair(file_data, i, j, check_items, tfidf_matrix, template_text,
                                       collusion_para=cp, template_missing=template_missing,
                                       extra_stop_words=extra_stop_words)
            pairs.append(pair)
            risk_matrix[i][j] = pair['risk']
            risk_matrix[j][i] = pair['risk']
    return pairs, risk_matrix


def build_key_info_matches(pairs, extra_stop_words=None):
    """Post-process key info matches from pairs.

    FIX-015 (D4): carry the pair matrix coordinates (i, j) through so the report
    shows real coordinates instead of (-,-) from a mismatched index lookup.
    FIX-2026-09-07-QA-C4: extra_stop_words (industry tables) filter 行业通用词
    from the common keywords.
    """
    matches = []
    for p in pairs:
        kw1 = set(extract_keywords(p['text1'], 20, extra_stop_words=extra_stop_words))
        kw2 = set(extract_keywords(p['text2'], 20, extra_stop_words=extra_stop_words))
        matches.append({
            'name1': p['name1'],
            'name2': p['name2'],
            'i': p.get('i', '-'),
            'j': p.get('j', '-'),
            'common_keywords': list(kw1 & kw2)[:10],
        })
    return matches


def build_attr_details(file_data):
    """Post-process file attribute details."""
    details = []
    for fd in file_data:
        meta = fd.get('metadata') or {}
        details.append({
            'filename': fd['filename'],
            'author': meta.get('author', ''),
            'creation_date': meta.get('creationDate', ''),
            'creator': meta.get('creator', ''),
            'producer': meta.get('producer', ''),
            'last_modified_by': meta.get('last_modified_by', ''),
            'modified': str(meta.get('modified', '')),
        })
    return details


def cluster_order_by_risk(risk_matrix, files):
    """E3 — heuristic reorder of rows/cols so mutually-high-risk companies
    cluster together (visual "gang block"). Greedy: start from the row with
    the largest total risk, then repeatedly pick the file most connected to
    the already-selected set.

    Returns list of indices (new order).
    """
    n = len(files)
    if n <= 2:
        return list(range(n))
    total = [sum(row) for row in risk_matrix]
    used = set()
    order = []
    # start with max total
    first = max(range(n), key=lambda i: total[i])
    used.add(first)
    order.append(first)
    while len(order) < n:
        best = None
        best_score = -1
        for i in range(n):
            if i in used:
                continue
            # connection = sum of risk to already-ordered files
            score = sum(risk_matrix[i][j] for j in used)
            # tie-break by own total risk
            score = (score, total[i])
            if best is None or score > best_score:
                best = i
                best_score = score
        used.add(best)
        order.append(best)
    return order


def detect_gangs(risk_matrix, files, threshold=15.0, min_members=2,
                 collusion_para_map=None, evidence_counts=None, min_evidence=2):
    """E2 — find connected groups where every internal pair exceeds
    `threshold` risk (a "gang"/疑似围标集团).

    FIX-2026-09-04-QA-B1: when `collusion_para_map` is provided, a gang is only
    reported if it contains at least one internal pair with paragraph-level
    substantive-collusion evidence — otherwise it's just shared template/industry
    overlap, not a collusion ring.
    FIX-2026-09-07-QA-C3: `evidence_counts` maps pair→#non-template substantive
    shared segments; a gang must contain ≥ `min_evidence` such segments to be
    reported (default 2, per user decision).

    Returns list of dicts: {members:[indices], files:[names],
                            internal_pairs:[(i,j)], max_risk, avg_risk}
    """
    n = len(files)
    groups = []

    def _is_connected(inds):
        for a in range(len(inds)):
            for b in range(a + 1, len(inds)):
                i, j = inds[a], inds[b]
                if risk_matrix[i][j] <= threshold:
                    return False
        return True

    def _grow(seed):
        members = set(seed)
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if i in members:
                    continue
                if all(risk_matrix[i][j] > threshold for j in members):
                    members.add(i)
                    changed = True
        return members

    def _has_evidence(inds):
        if not collusion_para_map and not evidence_counts:
            return True  # legacy callers without the new signal
        # 非模板实质段证据：组内所有内部对共享的实质段总数（跨对不去重，保守求和）
        total = 0
        for a in range(len(inds)):
            for b in range(a + 1, len(inds)):
                i, j = inds[a], inds[b]
                if evidence_counts:
                    total += evidence_counts.get((i, j), evidence_counts.get((j, i), 0))
                elif collusion_para_map.get((i, j), collusion_para_map.get((j, i), 0.0)) > 0:
                    return True
        return total >= min_evidence

    seen_groups = set()
    for i in range(n):
        for j in range(i + 1, n):
            if risk_matrix[i][j] <= threshold:
                continue
            g = _grow([i, j])
            if len(g) < min_members:
                continue
            key = tuple(sorted(g))
            if key in seen_groups:
                continue
            gl = sorted(g)
            if not _has_evidence(gl):
                continue
            seen_groups.add(key)
            pairs = []
            risks = []
            for a in range(len(gl)):
                for b in range(a + 1, len(gl)):
                    pairs.append((gl[a], gl[b]))
                    risks.append(risk_matrix[gl[a]][gl[b]])
            groups.append({
                'members': gl,
                'files': [files[x] for x in gl],
                'internal_pairs': pairs,
                'max_risk': max(risks) if risks else 0.0,
                'avg_risk': round(sum(risks) / len(risks), 2) if risks else 0.0,
            })
    groups.sort(key=lambda g: (-len(g['members']), -g['max_risk']))
    return groups
