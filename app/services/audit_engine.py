"""Audit scoring helpers — retained surface used by clearance.

C3-a (UNRESOLVED-027): the orchestrated full-audit flow (run_audit / run_preflight /
report generation / progress queues) was removed — after the ``audit_bp`` blueprint
was unregistered nothing called it. What remains is exactly what ``clearance_engine``
and the regression suite use: the ``_score_*`` functions, the scoring-function
registry, and ``_run_style_analysis``.
"""
import logging

from app.services.compliance_prompts import VERDICT_PASS

logger = logging.getLogger(__name__)


def _score_rule_extraction(findings: dict) -> float:
    rules = findings.get('rules', [])
    if not rules:
        return 0.0
    expected_min = 5
    try:
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT severity_thresholds FROM audit_config WHERE function_name = 'rule_extraction'")
                row = cur.fetchone()
                if row and row[0]:
                    expected_min = row[0].get('min_extracted_rules', 5)
    except Exception:
        pass
    return min(100.0, len(rules) / max(expected_min, 1) * 100)


def _score_compliance_check(findings: dict) -> float:
    results = findings.get('results', [])
    if not results:
        return 100.0
    passed = sum(1 for r in results if r.get('verdict') == VERDICT_PASS)
    return (passed / len(results)) * 100


def _score_quote_anomaly(findings: dict) -> float:
    severity = findings.get('severity_index', 50)
    return max(0.0, 100 - severity)


def _score_relationship_extraction(findings: dict) -> float:
    signals = findings.get('collusion_signals', [])
    red_flags = findings.get('red_flags', [])
    total_risks = len(signals) + len(red_flags)
    weight = 15
    try:
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT severity_thresholds FROM audit_config WHERE function_name = 'relationship_extraction'")
                row = cur.fetchone()
                if row and row[0]:
                    weight = row[0].get('risk_signal_weight', 15)
    except Exception:
        pass
    return max(0.0, 100 - total_risks * weight)


def _score_ai_review(findings: dict) -> float:
    axes = findings.get('axes', {})
    if not axes:
        return 50.0
    scores = []
    for axis_name, axis_data in axes.items():
        if isinstance(axis_data, dict):
            s = axis_data.get('score', 0)
        elif isinstance(axis_data, (int, float)):
            s = axis_data
        else:
            s = 0
        if isinstance(s, (int, float)) and s >= 0:
            scores.append(min(s, 10))
    if not scores:
        return 50.0
    return (sum(scores) / len(scores)) * 10


def _score_style_analysis(findings: dict) -> float:
    formality = findings.get('formality_level', 50)
    consistency = findings.get('consistency', 50)
    return (formality + consistency) / 2


def _score_timeline_compliance(findings: dict) -> float:
    if not findings or findings.get('error'):
        return 100.0
    delayed = findings.get('delayed_count', 0)
    total_delay = findings.get('total_delay_days', 0)
    return max(0.0, 100.0 - delayed * 10 - total_delay * 2)


SCORING_FUNCTIONS = {
    'rule_extraction': _score_rule_extraction,
    'compliance_check': _score_compliance_check,
    'quote_anomaly': _score_quote_anomaly,
    'relationship_extraction': _score_relationship_extraction,
    'ai_doc_review': _score_ai_review,
    'style_analysis': _score_style_analysis,
    'timeline_compliance': _score_timeline_compliance,
}


def _run_style_analysis(text: str) -> dict:
    if not text or len(text.strip()) < 100:
        return {'formality_level': 50, 'consistency': 50, 'error': 'doc too short'}
    try:
        from app.services.style_engine import _analyze_formality, _analyze_tone
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 20][:100]
        if not paragraphs:
            return {'formality_level': 50, 'consistency': 50}
        formality = _analyze_formality(paragraphs)
        tone = _analyze_tone(paragraphs)
        return {
            'formality_level': formality.get('formality_score', 50),
            'formality_label': formality.get('label', 'unknown'),
            'tone': tone,
            'avg_sentence_length': formality.get('avg_sentence_length', 0),
        }
    except Exception as e:
        logger.error(f"Style analysis failed: {e}")
        return {'formality_level': 50, 'consistency': 50, 'error': str(e)}
