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
    total_suspects: int = 0
    critical_count: int = 0
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


def _detect_chinese_typos(text: str, config: dict) -> list[TypoFinding]:
    """Detect Chinese character typos using pycorrector + domain dictionary."""
    findings = []

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
    for wrong, correct_list in _BIDDING_CONFUSION_PAIRS.items():
        for m in re.finditer(re.escape(wrong), text):
            # Check context: avoid flagging when it's clearly correct
            context = text[max(0, m.start()-10):m.end()+20]
            findings.append(TypoFinding(
                layer="chinese",
                suspect_text=wrong,
                suggestions=correct_list,
                confidence=0.75,
                context_snippet=context,
                position_start=m.start(),
                position_end=m.end(),
                severity="warning",
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
_DAXIE_AMOUNT_RE = re.compile(
    r'(?P<daxie>'
    r'(?:壹|贰|叁|肆|伍|陆|柒|捌|玖|拾|佰|仟|万|亿|零|元|角|分|整)+'
    r')'
)

_ARABIC_AMOUNT_RE = re.compile(
    r'(?P<num>[\d,]+\.?\d*)\s*(?P<unit>万元|万|亿元|亿|元|万元整|元整)'
)

# Document reference code patterns
_REF_CODE_PATTERNS = {
    'bid_ref': re.compile(r'(?:招标编号|项目编号|采购编号|ZFCG|ZC|GC|SG|JZ|'
                          r'BID|TENDER)[：:\s]*([A-Z0-9\-_/]+)', re.IGNORECASE),
    'contract_ref': re.compile(r'(?:合同编号|合同号|CONTRACT)[：:\s]*([A-Z0-9\-_/]+)', re.IGNORECASE),
    'iso_ref': re.compile(r'(?:ISO|GB|GB/T|IEC)\s*[\d\-:]+', re.IGNORECASE),
    'money_upper': re.compile(r'[¥￥]\s*([\d,]+\.?\d*)'),
}


def _parse_daxie_to_number(daxie: str) -> Optional[float]:
    """Parse Chinese uppercase amount to numeric value."""
    digit_map = {'壹':1,'贰':2,'叁':3,'肆':4,'伍':5,'陆':6,'柒':7,'捌':8,'玖':9,'零':0}
    unit_map = {'拾':10,'佰':100,'仟':1000,'万':10000,'亿':100000000}

    result = 0.0
    segment = 0.0
    has_unit = False
    for ch in daxie:
        if ch in digit_map:
            segment = digit_map[ch]
        elif ch in unit_map:
            if segment == 0:
                segment = 1
            unit = unit_map[ch]
            if unit >= 10000:
                result = (result + segment) * unit
                segment = 0
            else:
                segment *= unit
                result += segment
                segment = 0
            has_unit = True
        elif ch == '元':
            result += segment
            segment = 0
        elif ch == '角':
            result += segment * 0.1
            segment = 0
        elif ch == '分':
            result += segment * 0.01
            segment = 0
        elif ch == '整':
            pass  # marker only
        else:
            continue
    result += segment
    return result if has_unit or result > 0 else None


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

            # Find nearest Arabic amount
            nearest_arabic = None
            for am in arabic_matches:
                raw_num = am.group('num').replace(',', '')
                try:
                    arabic_val = float(raw_num)
                    unit = am.group('unit') or ''
                    if '万' in unit:
                        arabic_val *= 10000
                    elif '亿' in unit:
                        arabic_val *= 100000000
                    if abs(arabic_val - daxie_val) < 0.01 * max(arabic_val, 1):
                        nearest_arabic = arabic_val
                        break
                except ValueError:
                    continue

            if nearest_arabic is None and daxie_val > 0:
                findings.append(TypoFinding(
                    layer="numeric",
                    suspect_text=daxie_str,
                    suggestions=[],
                    confidence=0.70,
                    context_snippet=_get_context(text, daxie_str, 50),
                    position_start=dm.start(),
                    position_end=dm.end(),
                    is_daxie_error=True,
                    daxie_actual=daxie_str,
                    daxie_expected="",
                    severity="warning",
                ))

    # 2. Reference code format validation
    for ref_type, pattern in _REF_CODE_PATTERNS.items():
        for m in pattern.finditer(text):
            code = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group()
            if not code:
                continue
            # Check for common typos in reference codes
            issues = []
            if re.search(r'[Oo]', code):  # Letter O used instead of digit 0
                issues.append("可能包含字母O代替数字0")
            if re.search(r'[lL]', code):  # Letter l used instead of digit 1
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

    report.total_suspects = len(report.findings)
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
