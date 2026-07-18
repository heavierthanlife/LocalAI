"""Compliance history trend analysis service (U10).

Data sources:
  - Primary: audit_file_results.findings + audit_runs.overall_score
  - Accuracy: compliance_check_feedback (user-labeled verdicts)

Metrics:
  - Score trend over time
  - Violation type distribution
  - Most frequently violated rules
  - Category/severity breakdown
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from app.database import get_db_connection

logger = logging.getLogger(__name__)


def get_score_trend(days: int = 90) -> dict:
    """Compliance score trend over time (daily aggregation)."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DATE(started_at) AS d,
                       ROUND(AVG(overall_score)::numeric, 1) AS avg_score,
                       COUNT(*) AS run_count,
                       COUNT(CASE WHEN overall_status = 'PASS' THEN 1 END) AS pass_count,
                       COUNT(CASE WHEN overall_status = 'FAIL' THEN 1 END) AS fail_count
                FROM audit_runs
                WHERE started_at >= %s
                  AND overall_score IS NOT NULL
                GROUP BY DATE(started_at)
                ORDER BY d
            """, (since,))
            rows = cur.fetchall()

    dates = []
    scores = []
    pass_counts = []
    fail_counts = []
    for r in rows:
        dates.append(str(r[0]))
        scores.append(float(r[1]) if r[1] else 0)
        pass_counts.append(r[3] or 0)
        fail_counts.append(r[4] or 0)

    return {
        'period_days': days,
        'dates': dates,
        'scores': scores,
        'pass_counts': pass_counts,
        'fail_counts': fail_counts,
        'total_runs': sum(r[2] or 0 for r in rows),
    }


def get_violation_distribution(days: int = 90) -> dict:
    """Most frequently violated rules/types across audit runs."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT afr.function_name,
                       afr.status,
                       COUNT(*) AS cnt,
                       ROUND(AVG(afr.score)::numeric, 1) AS avg_score
                FROM audit_file_results afr
                JOIN audit_runs ar ON ar.id = afr.run_id
                WHERE ar.started_at >= %s
                  AND afr.status IN ('violation', 'critical', 'warning')
                GROUP BY afr.function_name, afr.status
                ORDER BY cnt DESC
                LIMIT 20
            """, (since,))
            rows = cur.fetchall()

            # Per-category breakdown
            cur.execute("""
                SELECT afr.function_name,
                       SUM(CASE WHEN afr.status = 'critical' THEN 1 ELSE 0 END) AS critical_count,
                       SUM(CASE WHEN afr.status = 'violation' THEN 1 ELSE 0 END) AS violation_count,
                       SUM(CASE WHEN afr.status = 'warning' THEN 1 ELSE 0 END) AS warning_count,
                       SUM(CASE WHEN afr.status = 'pass' THEN 1 ELSE 0 END) AS pass_count,
                       COUNT(*) AS total
                FROM audit_file_results afr
                JOIN audit_runs ar ON ar.id = afr.run_id
                WHERE ar.started_at >= %s
                GROUP BY afr.function_name
                ORDER BY total DESC
            """, (since,))
            cat_rows = cur.fetchall()

            cur.execute("""
                SELECT COUNT(*) AS total_checks,
                       COUNT(CASE WHEN status = 'pass' THEN 1 END) AS passed,
                       COUNT(CASE WHEN status = 'violation' THEN 1 END) AS violations,
                       COUNT(CASE WHEN status = 'critical' THEN 1 END) AS criticals
                FROM audit_file_results afr
                JOIN audit_runs ar ON ar.id = afr.run_id
                WHERE ar.started_at >= %s
            """, (since,))
            summary = cur.fetchone()

    distribution = []
    for r in rows:
        distribution.append({
            'function_name': r[0],
            'status': r[1],
            'count': r[2],
            'avg_score': float(r[3]) if r[3] else 0,
        })

    category_breakdown = []
    for r in cat_rows:
        category_breakdown.append({
            'function_name': r[0],
            'critical': r[1] or 0,
            'violation': r[2] or 0,
            'warning': r[3] or 0,
            'pass': r[4] or 0,
            'total': r[5],
        })

    return {
        'period_days': days,
        'distribution': distribution,
        'category_breakdown': category_breakdown,
        'overall': {
            'total_checks': summary[0] or 0,
            'passed': summary[1] or 0,
            'violations': summary[2] or 0,
            'criticals': summary[3] or 0,
        },
    }


def get_feedback_accuracy(days: int = 90) -> dict:
    """AI accuracy based on user feedback labels."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check if compliance_check_feedback table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'compliance_check_feedback'
                )
            """)
            if not cur.fetchone()[0]:
                return {'available': False, 'note': 'feedback table does not exist'}

            cur.execute("""
                SELECT user_verdict, COUNT(*) AS cnt
                FROM compliance_check_feedback
                WHERE created_at >= %s
                GROUP BY user_verdict
            """, (since,))
            verdicts = {r[0]: r[1] for r in cur.fetchall()}

            true_count = verdicts.get('true_violation', 0)
            false_count = verdicts.get('false_positive', 0)
            not_matter = verdicts.get('not_matter', 0)
            total = true_count + false_count + not_matter
            accuracy = round(true_count / total * 100, 1) if total > 0 else 0

            return {
                'available': True,
                'period_days': days,
                'total_feedback': total,
                'true_violation': true_count,
                'false_positive': false_count,
                'not_matter': not_matter,
                'ai_accuracy_pct': accuracy,
            }
