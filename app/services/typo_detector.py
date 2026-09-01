"""Typo / misspelling detection for Chinese bid documents.

Three detection layers:
  1. Chinese — homophone/shape-similar character errors (pycorrector)
  2. English — embedded English terminology typos (symspellpy + bidding dictionary)
  3. Numeric/Reference — amount validation, daxie (大写) cross-check, document ref codes

Output modes:
  - Suggest-only (default): flag suspicious text with correction candidates
  - Diff review (opt-in): generate before/after diff for human approval

Config via runtime_config.py: typo_chinese_enabled, typo_english_enabled,
typo_numeric_enabled, typo_daxie_enabled, typo_auto_correct,
typo_diff_review_enabled, typo_min_confidence.
"""

from __future__ import annotations

import json as _json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# --- Config helper -----------------------------------------------------------

def _get_typo_config() -> dict:
    try:
        from app.services.runtime_config import get
        return {
            'chinese_enabled': get('typo_chinese_enabled', True),
            'english_enabled': get('typo_english_enabled', True),
            'numeric_enabled': get('typo_numeric_enabled', True),
            'daxie_enabled': get('typo_daxie_enabled', True),
            'auto_correct': get('typo_auto_correct', False),
            'diff_review': get('typo_diff_review_enabled', False),
            'min_confidence': get('typo_min_confidence', 0.70),
        }
    except Exception:
        return {
            'chinese_enabled': True, 'english_enabled': True,
            'numeric_enabled': True, 'daxie_enabled': True,
            'auto_correct': False, 'diff_review': False, 'min_confidence': 0.70,
        }


# --- Data classes ------------------------------------------------------------

@dataclass
class TypoFinding:
    layer: str                  # "chinese", "english", "numeric", "daxie"
    suspect_text: str
    suggestions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    context_snippet: str = ""   # surrounding text for context
    position_start: int = 0
    position_end: int = 0
    is_daxie_error: bool = False
    daxie_expected: str = ""
    daxie_actual: str = ""
    severity: str = "info"      # "info", "warning", "critical"


@dataclass
class TypoReport:
    findings: list[TypoFinding] = field(default_factory=list)
    layers_run: list[str] = field(default_factory=list)
    total_suspects: int = 0       # warning + critical (real suspects, score-bearing)
    info_count: int = 0           # info observations (ambiguous, non-scoring)
    critical_count: int = 0
    truncated: bool = False       # FIX-015 (B5): set when findings hit the cap
    diff_text: str = ""         # populated when diff_review mode is enabled


# --- Layer 1: Chinese character detection -----------------------------------

# Common Chinese homophone/confusion pairs in bidding context
_BIDDING_CONFUSION_PAIRS = {
    '权力': ['权利', '权益'],
    '制定': ['制订'],
    '必须': ['必需'],
    '交纳': ['缴纳'],
    '签定': ['签订'],
    '订金': ['定金'],
    '启事': ['启示'],
    '截止': ['截至'],
    '期间': ['其间'],
    '形式': ['形势'],
    '制订': ['制定'],
    '权利': ['权力', '权益'],
    '必需': ['必须'],
    '缴纳': ['交纳'],
    '签订': ['签定'],
    '定金': ['订金'],
    '截至': ['截止'],
    '其间': ['期间'],
}

# FIX-015 (B2): pairs that are near-synonyms in tender contexts — both spellings
# are often legitimate, so flagging them as "warning" floods reports. Downgrade
# to info (non-scoring).
_AMBIGUOUS_PAIRS = {'权力', '权利', '制订', '制定', '期间', '其间', '形式', '形势', '交纳', '缴纳'}

# FIX-015 (B2): context guards. When the word appears with these following
# tokens, it's very likely the CORRECT spelling — skip the finding.
_CONTEXT_GUARD_OK = {
    '必须': ('条件', '办理', '遵守', '满足', '达到', '符合', '持有', '取得', '提交', '具有', '具有', '具备', '由', '向'),
    '必需': ('品', '物资', '材料', '设备', '营养', '条件'),
    '交纳': ('保证金', '款项', '费用', '款项'),
    '缴纳': ('税款', '费用', '社保', '养老金'),
    '截止': ('日期', '时间', '至', '前'),
    '截至': ('至', '目前', '为止'),
    '签定': ('合同', '协议', '合同书'),
    '签订': ('合同', '协议', '意向书', '前', '后', '之日', '生效'),
    '订金': ('支付', '预', '合同'),
    '定金': ('支付', '收取', '合同'),
    '启事': ('寻', '招领'),
    '启示': ('启发', '给'),
}

# FIX-015 (B2): when the word is PRECEDED by these tokens, it is almost always
# correct usage (e.g. 合同签订前 → 签订 correct).
_CONTEXT_GUARD_BEFORE_OK = {
    '签订': ('合同', '协议'),
    '缴纳': ('税', '费'),
    '交纳': ('保证', '履约'),
    '截止': ('报名', '投标', '递交', '响应'),
}

# FIX-015 (B2): words that are NEVER typos in any tender context (field labels
# and document headers) — hard skip.
_HARD_SKIP_WORDS = {'以下', '内容', '职务', '身份证号码', '编号', '名称', '地址', '电话', '联系人'}


def _detect_chinese_typos(text: str, config: dict) -> list[TypoFinding]:
    """Detect Chinese character typos using pycorrector + domain dictionary."""
    findings = []
    from app.services.typo_whitelist import is_allowed

    # 1. Use pycorrector if available
    try:
        from pycorrector import MacBertCorrector
        corrector = MacBertCorrector()
        corrected = corrector.correct(text)
        if corrected and isinstance(corrected, list):
            for item in corrected:
                if isinstance(item, dict):
                    orig = item.get('source', item.get('wrong', ''))
                    corr = item.get('target', item.get('correct', ''))
                    score = item.get('score', item.get('confidence', 0.85))
                elif isinstance(item, tuple) and len(item) >= 2:
                    orig, corr = item[0], item[1]
                    score = item[2] if len(item) > 2 else 0.85
                else:
                    continue

                if orig and corr and orig != corr and score >= config['min_confidence']:
                    if is_allowed(orig) or is_allowed(corr):
                        continue  # FIX-015 (B4): domain term — never a typo
                    pos = text.find(orig)
                    findings.append(TypoFinding(
                        layer="chinese",
                        suspect_text=orig,
                        suggestions=[corr],
                        confidence=score,
                        context_snippet=_get_context(text, orig, 30),
                        position_start=pos if pos >= 0 else 0,
                        position_end=pos + len(orig) if pos >= 0 else 0,
                        severity="warning",
                    ))
    except ImportError:
        logger.info("pycorrector not installed — using domain dictionary only")
    except Exception as e:
        logger.warning(f"pycorrector failed (non-blocking): {e}")

    # 2. Domain dictionary check (always runs, catches bidding-specific errors)
    # FIX-015 (B2): apply context guards (skip correct usage), hard-skip field
    # labels, downgrade ambiguous near-synonyms to info, and dedup per doc.
    seen_positions = set()
    for wrong, correct_list in _BIDDING_CONFUSION_PAIRS.items():
        for m in re.finditer(re.escape(wrong), text):
            # Dedup: same doc + same position → single finding
            if (wrong, m.start()) in seen_positions:
                continue
            # Hard-skip field labels / headers
            if wrong in _HARD_SKIP_WORDS:
                continue
            # FIX-015 (B4): domain term — never a typo
            if is_allowed(wrong):
                continue
            # Context guard: following tokens indicate correct usage
            context = text[m.end():m.end() + 12]
            guarded = _CONTEXT_GUARD_OK.get(wrong)
            if guarded and any(context.startswith(g) for g in guarded):
                continue
            # Context guard: preceding tokens indicate correct usage
            before = text[max(0, m.start()-4):m.start()]
            guarded_before = _CONTEXT_GUARD_BEFORE_OK.get(wrong)
            if guarded_before and any(before.endswith(g) for g in guarded_before):
                continue
            seen_positions.add((wrong, m.start()))
            # Ambiguous near-synonyms → info (non-scoring), not warning
            severity = "info" if wrong in _AMBIGUOUS_PAIRS else "warning"
            findings.append(TypoFinding(
                layer="chinese",
                suspect_text=wrong,
                suggestions=correct_list,
                confidence=0.75,
                context_snippet=text[max(0, m.start()-10):m.end()+20],
                position_start=m.start(),
                position_end=m.end(),
                severity=severity,
            ))

    return findings


def _get_context(text: str, target: str, window: int = 30) -> str:
    """Get surrounding context for a target string."""
    pos = text.find(target)
    if pos < 0:
        return ""
    start = max(0, pos - window)
    end = min(len(text), pos + len(target) + window)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


# --- Layer 2: English/ASCII typo detection -----------------------------------

_BIDDING_ENGLISH_TERMS = {
    'certification', 'certificate', 'specification', 'standard', 'compliance',
    'procurement', 'bidding', 'tender', 'contract', 'guarantee', 'warranty',
    'insurance', 'performance', 'bond', 'payment', 'delivery', 'quality',
    'inspection', 'acceptance', 'testing', 'approval', 'evaluation',
    'qualification', 'registration', 'license', 'permit', 'audit', 'review',
    'document', 'report', 'analysis', 'assessment', 'management', 'system',
    'requirement', 'condition', 'provision', 'clause', 'schedule', 'appendix',
    'ISO', 'GB', 'GB/T', 'IEC', 'ANSI', 'ASTM', 'CE', 'UL', 'ROHS',
}


def _detect_english_typos(text: str, config: dict) -> list[TypoFinding]:
    """Detect English spelling errors embedded in Chinese text."""
    findings = []

    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
        # Add domain terms to dictionary
        spell.word_frequency.load_words(_BIDDING_ENGLISH_TERMS)
    except ImportError:
        try:
            from symspellpy import SymSpell, Verbosity
            spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            for term in _BIDDING_ENGLISH_TERMS:
                spell.create_dictionary_entry(term, 1)
        except ImportError:
            logger.info("No English spellchecker available — skipping English layer")
            return findings

    # Extract English words from Chinese text
    english_words_pattern = re.compile(r'\b[A-Za-z]{3,}\b')
    seen = set()
    for m in english_words_pattern.finditer(text):
        word = m.group()
        if word.lower() in seen:
            continue
        seen.add(word.lower())

        # Skip known terms and proper nouns (all caps)
        if word.lower() in _BIDDING_ENGLISH_TERMS:
            continue
        if word.isupper() and len(word) <= 5:  # likely acronym
            continue

        # Check spelling
        try:
            if hasattr(spell, 'unknown'):
                # pyspellchecker
                if word.lower() in spell.unknown([word.lower()]):
                    correction = spell.correction(word.lower())
                    if correction and correction != word.lower():
                        findings.append(TypoFinding(
                            layer="english",
                            suspect_text=word,
                            suggestions=[correction],
                            confidence=0.82,
                            context_snippet=_get_context(text, word, 25),
                            position_start=m.start(),
                            position_end=m.end(),
                            severity="info",
                        ))
            elif hasattr(spell, 'lookup'):
                # symspellpy
                suggestions = spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                if suggestions and suggestions[0].term != word:
                    findings.append(TypoFinding(
                        layer="english",
                        suspect_text=word,
                        suggestions=[s.term for s in suggestions[:3]],
                        confidence=0.80,
                        context_snippet=_get_context(text, word, 25),
                        position_start=m.start(),
                        position_end=m.end(),
                        severity="info",
                    ))
        except Exception:
            continue

    return findings


# --- Layer 3: Numeric / reference validation ---------------------------------

# Chinese daxie numerals (大写金额) regex — must cover amounts like "壹佰贰拾叁万肆仟伍佰陆拾柒元捌角玖分"
# FIX-015 (B1): the daxie must START with a digit numeral (壹-玖 / 零), so bare
# unit-only runs like "万元"/"元整" (column headers) are never matched as amounts.
_DAXIE_DIGITS = '零壹贰叁肆伍陆柒捌玖'
_DAXIE_UNITS = '拾佰仟万亿角分元整'
_DAXIE_AMOUNT_RE = re.compile(
    r'(?P<daxie>'
    r'[' + _DAXIE_DIGITS + ']'
    r'[' + _DAXIE_DIGITS + _DAXIE_UNITS + ']*'
    r')'
)

_ARABIC_AMOUNT_RE = re.compile(
    r'(?P<num>[\d,]+\.?\d*)\s*(?P<unit>万元|万|亿元|亿|元|万元整|元整|元人民币|人民币元)?'
)

# Document reference code patterns
# FIX-015 (B3): use \b word boundaries so we match the FULL code token, not
# substrings. Previously "COO" or "BOOK" inside text would trip the O/l check.
_REF_CODE_PATTERNS = {
    'bid_ref': re.compile(r'(?:招标编号|项目编号|采购编号|ZFCG|ZC|GC|SG|JZ|'
                          r'BID|TENDER)[：:\s]*\b([A-Z0-9][A-Z0-9\-_/]*[A-Z0-9])\b', re.IGNORECASE),
    'contract_ref': re.compile(r'(?:合同编号|合同号|CONTRACT)[：:\s]*\b([A-Z0-9][A-Z0-9\-_/]*[A-Z0-9])\b', re.IGNORECASE),
    'iso_ref': re.compile(r'\b(?:ISO|GB|GB/T|IEC)\s*[\d\-:]+', re.IGNORECASE),
    'money_upper': re.compile(r'[¥￥]\s*([\d,]+\.?\d*)'),
}


def _parse_daxie_to_number(daxie: str) -> Optional[float]:
    """Parse Chinese uppercase amount to numeric value.

    FIX-015 (B1): section-based parser. Handles 亿/万/仟/佰/拾 correctly:
      叁亿零贰佰万 = 3×1e8 + 200×1e4 = 302,000,000
      壹佰贰拾叁万肆仟伍佰陆拾柒元捌角玖分 = 1,234,567.89
    """
    digit_map = {'壹':1,'贰':2,'叁':3,'肆':4,'伍':5,'陆':6,'柒':7,'捌':8,'玖':9,'零':0}
    unit_map = {'拾':10,'佰':100,'仟':1000,'万':10000,'亿':100000000}

    total = 0.0      # folded result (亿/万 sections already folded)
    section = 0.0    # current section being built (below 万)
    num = 0.0        # current digit group within section
    has_unit = False
    for ch in daxie:
        if ch in digit_map:
            num = float(digit_map[ch])
        elif ch in unit_map:
            unit = unit_map[ch]
            if unit >= 10000:
                # 万 / 亿 boundary: fold the current section into total
                total += (section + num) * unit
                section = 0.0
                num = 0.0
            else:
                # 拾/佰/仟: fold current digit into section
                section += (num if num > 0 else 1) * unit
                num = 0.0
            has_unit = True
        elif ch == '元':
            section += num
            num = 0.0
        elif ch == '角':
            section += num * 0.1
            num = 0.0
        elif ch == '分':
            section += num * 0.01
            num = 0.0
        elif ch == '整':
            pass  # marker only
        else:
            continue
    total += section + num
    return total if has_unit or total > 0 else None


def _detect_numeric_typos(text: str, config: dict) -> list[TypoFinding]:
    """Detect numeric/reference errors and daxie mismatches."""
    findings = []

    # 1. Daxie amount cross-validation
    if config.get('daxie_enabled', True):
        daxie_matches = list(_DAXIE_AMOUNT_RE.finditer(text))
        arabic_matches = list(_ARABIC_AMOUNT_RE.finditer(text))

        for dm in daxie_matches:
            daxie_str = dm.group('daxie')
            daxie_val = _parse_daxie_to_number(daxie_str)
            if daxie_val is None:
                continue

            # Find nearest Arabic amount (within a reasonable window).
            # FIX-015 (B1): only flag a daxie as an error when an Arabic
            # counterpart exists nearby but does NOT match, OR when a daxie
            # appears with no nearby Arabic at all (both are real mismatch
            # signals). Previously every unpaired 大写 became a finding.
            nearest_arabic = None
            nearest_arabic_dist = 10 ** 9
            for am in arabic_matches:
                raw_num = am.group('num').replace(',', '')
                try:
                    arabic_val = float(raw_num)
                    unit = am.group('unit') or ''
                    if '万' in unit:
                        arabic_val *= 10000
                    elif '亿' in unit:
                        arabic_val *= 100000000
                    dist = min(abs(dm.start() - am.start()), abs(dm.end() - am.end()))
                    if dist < nearest_arabic_dist:
                        nearest_arabic_dist = dist
                        nearest_arabic = arabic_val
                except ValueError:
                    continue

            # Only meaningful if an Arabic amount is near (within 40 chars);
            # a standalone 大写 with no Arabic nearby is just the amount line
            # header, not a typo signal.
            if nearest_arabic is not None and nearest_arabic_dist <= 40:
                if abs(nearest_arabic - daxie_val) >= 0.01 * max(nearest_arabic, 1):
                    # Arabic present but mismatched → real daxie error
                    findings.append(TypoFinding(
                        layer="numeric",
                        suspect_text=daxie_str,
                        suggestions=[str(int(nearest_arabic)) if nearest_arabic == int(nearest_arabic) else f"{nearest_arabic:.2f}"],
                        confidence=0.75,
                        context_snippet=_get_context(text, daxie_str, 50),
                        position_start=dm.start(),
                        position_end=dm.end(),
                        is_daxie_error=True,
                        daxie_actual=daxie_str,
                        daxie_expected=str(int(nearest_arabic)) if nearest_arabic == int(nearest_arabic) else f"{nearest_arabic:.2f}",
                        severity="warning",
                    ))
            elif nearest_arabic is None:
                # No Arabic counterpart in the whole doc → informational only
                findings.append(TypoFinding(
                    layer="numeric",
                    suspect_text=daxie_str,
                    suggestions=[],
                    confidence=0.45,
                    context_snippet=_get_context(text, daxie_str, 50),
                    position_start=dm.start(),
                    position_end=dm.end(),
                    is_daxie_error=True,
                    daxie_actual=daxie_str,
                    daxie_expected="",
                    severity="info",
                ))

    # 2. Reference code format validation
    for ref_type, pattern in _REF_CODE_PATTERNS.items():
        for m in pattern.finditer(text):
            code = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group()
            if not code:
                continue
            # Check for common typos in reference codes.
            # FIX-015 (B3): only flag O/l when the code is clearly a mixed
            # alphanumeric code (has digits) — pure words like COO/BOOK or
            # hex-like tokens are not typos. ISO/IEC/GB standards are exempt
            # (their O/l are legitimate prefix letters, e.g. ISO9001).
            issues = []
            is_standard = ref_type == 'iso_ref'
            has_digit = re.search(r'[0-9]', code)
            if has_digit and not is_standard and re.search(r'[Oo]', code):  # Letter O vs digit 0
                issues.append("可能包含字母O代替数字0")
            if has_digit and not is_standard and re.search(r'[lL]', code):  # Letter l vs digit 1
                issues.append("可能包含字母l代替数字1")
            if re.search(r'[一-鿿]', code):  # Chinese character in alphanumeric code
                issues.append("引用编号中混合了中文字符")
            if re.search(r'\s', code):      # Unexpected whitespace
                issues.append("引用编号中含有多余空格")

            if issues:
                findings.append(TypoFinding(
                    layer="numeric",
                    suspect_text=code,
                    suggestions=[re.sub(r'[Oo]', '0', code).replace('l', '1').replace('L', '1')],
                    confidence=0.78,
                    context_snippet=_get_context(text, code, 20),
                    position_start=m.start(),
                    position_end=m.end(),
                    severity="critical" if ref_type in ('bid_ref', 'contract_ref') else "warning",
                ))

    # 3. Large amount sanity check (warn on amounts that look implausible)
    suspicious_amounts = re.findall(
        r'(?P<num>[\d,]{8,})(?:\s*(?:万元|元|万|亿))?', text)
    for amount_str in suspicious_amounts[:10]:
        try:
            val = float(amount_str.replace(',', ''))
            if val > 1e10:  # 10 billion
                findings.append(TypoFinding(
                    layer="numeric",
                    suspect_text=amount_str,
                    suggestions=[],
                    confidence=0.55,
                    context_snippet=_get_context(text, amount_str, 30),
                    severity="info",
                ))
        except ValueError:
            continue

    return findings


# --- Main public API ---------------------------------------------------------

def detect_typos(
    text: str,
    doc_name: str = "",
    audit = None,
) -> TypoReport:
    """Run all typo detection layers on a single document.

    Args:
        text: document text to analyze
        doc_name: document identifier
        audit: optional AuditLogger

    Returns:
        TypoReport with all findings organized by layer.
    """
    config = _get_typo_config()
    report = TypoReport()

    if audit:
        audit.component("typo_detect_start", input_chars=len(text), doc_name=doc_name)

    # Layer 1: Chinese
    if config['chinese_enabled']:
        try:
            cn_findings = _detect_chinese_typos(text, config)
            report.findings.extend(cn_findings)
            report.layers_run.append("chinese")
            if audit:
                audit.component("typo_layer", status="OK", layer="chinese",
                                findings=len(cn_findings))
        except Exception as e:
            logger.warning(f"Chinese typo detection failed: {e}")
            if audit:
                audit.component("typo_layer", status="FAILED", layer="chinese",
                                error=str(e)[:100])

    # Layer 2: English
    if config['english_enabled']:
        try:
            en_findings = _detect_english_typos(text, config)
            report.findings.extend(en_findings)
            report.layers_run.append("english")
            if audit:
                audit.component("typo_layer", status="OK", layer="english",
                                findings=len(en_findings))
        except Exception as e:
            logger.warning(f"English typo detection failed: {e}")
            if audit:
                audit.component("typo_layer", status="FAILED", layer="english",
                                error=str(e)[:100])

    # Layer 3: Numeric/reference
    if config['numeric_enabled'] or config['daxie_enabled']:
        try:
            num_findings = _detect_numeric_typos(text, config)
            report.findings.extend(num_findings)
            report.layers_run.append("numeric")
            if audit:
                audit.component("typo_layer", status="OK", layer="numeric",
                                findings=len(num_findings))
        except Exception as e:
            logger.warning(f"Numeric typo detection failed: {e}")
            if audit:
                audit.component("typo_layer", status="FAILED", layer="numeric",
                                error=str(e)[:100])

    # FIX-015 (B5): circuit breaker — a garbage/OCR doc can produce thousands of
    # raw findings. Cap at MAX_TYPO_FINDINGS so the frontend/report never gets
    # flooded, and surface the truncation via report.truncated.
    MAX_TYPO_FINDINGS = 200
    if len(report.findings) > MAX_TYPO_FINDINGS:
        report.findings = report.findings[:MAX_TYPO_FINDINGS]
        report.truncated = True

    report.total_suspects = len([f for f in report.findings if f.severity in ("warning", "critical")])
    report.info_count = len([f for f in report.findings if f.severity == "info"])
    report.critical_count = len([f for f in report.findings if f.severity == "critical"])

    # Generate diff text if diff review mode is enabled
    if config['diff_review'] and report.findings:
        report.diff_text = _generate_diff(text, report.findings)

    if audit:
        audit.component("typo_summary", status="OK",
                        total=report.total_suspects,
                        critical=report.critical_count,
                        layers=','.join(report.layers_run))

    return report


def detect_typos_batch(
    file_data: list[dict],
    audit = None,
) -> dict[str, TypoReport]:
    """Run typo detection on multiple documents.

    Returns dict mapping filename → TypoReport.
    """
    results = {}
    for fd in file_data:
        text = fd.get('text', '')
        filename = fd.get('filename', '')
        if text:
            results[filename] = detect_typos(text, doc_name=filename, audit=audit)
    return results


def _generate_diff(text: str, findings: list[TypoFinding]) -> str:
    """Generate a simple before/after diff for review mode."""
    if not findings:
        return text

    # Sort by position descending so replacements don't shift indices
    sorted_findings = sorted(
        [f for f in findings if f.suggestions],
        key=lambda f: f.position_start, reverse=True
    )

    result = text
    for f in sorted_findings:
        if f.suggestions and f.position_start >= 0:
            replacement = f.suggestions[0]
            marked = f"[~~{f.suspect_text}~~→**{replacement}**]"
            result = result[:f.position_start] + marked + result[f.position_end:]

    return result


# --- DB persistence ----------------------------------------------------------

def save_typo_results(
    user_id: str,
    task_id: str,
    results: dict[str, TypoReport],
) -> int:
    """Persist typo detection results to DB. Returns count of rows saved."""
    saved = 0
    try:
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for doc_name, report in results.items():
                    for f in report.findings:
                        cur.execute("""
                            INSERT INTO typo_detection_results
                                (user_id, task_id, doc_name, layer,
                                 suspect_text, context_snippet,
                                 suggestions, confidence,
                                 position_start, position_end,
                                 is_daxie_error, daxie_expected, daxie_actual)
                            VALUES (%s,%s,%s,%s, %s,%s, %s,%s, %s,%s, %s,%s,%s)
                        """, (
                            user_id, task_id, doc_name, f.layer,
                            f.suspect_text, f.context_snippet[:500],
                            _json.dumps(f.suggestions, ensure_ascii=False),
                            f.confidence,
                            f.position_start, f.position_end,
                            f.is_daxie_error, f.daxie_expected, f.daxie_actual,
                        ))
                        saved += 1
                conn.commit()
        logger.info(f"Saved {saved} typo findings for task {task_id}")
    except Exception as e:
        logger.error(f"Failed to save typo results: {e}", exc_info=True)
    return saved
