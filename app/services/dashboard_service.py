"""Compliance dashboard service (U11).

Aggregates compliance data for the dashboard:
  - Overall stats (total runs, pass rate, avg score)
  - Top 5 most-violated rules
  - Recent audit runs
  - Project-level breakdown
  - Critical alerts (recent CRITICAL findings)
"""

import logging
from datetime import datetime, timezone, timedelta
from app.database import get_db_connection

logger = logging.getLogger(__name__)


def get_dashboard_data(days: int = 30) -> dict:
    """Aggregated dashboard data for the compliance tab."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Overall stats
            cur.execute("""
                SELECT COUNT(*) AS total_runs,
                       COUNT(CASE WHEN overall_status = 'PASS' THEN 1 END) AS passed,
                       COUNT(CASE WHEN overall_status = 'FAIL' THEN 1 END) AS failed,
                       ROUND(AVG(overall_score)::numeric, 1) AS avg_score,
                       COUNT(CASE WHEN overall_score >= 80 THEN 1 END) AS high_score
                FROM audit_runs
                WHERE started_at >= %s
                  AND overall_score IS NOT NULL
            """, (since,))
            stats_row = cur.fetchone()
            total_runs = stats_row[0] or 0
            passed = stats_row[1] or 0
            pass_rate = round(passed / total_runs * 100, 1) if total_runs > 0 else 0

            # Total file checks
            cur.execute("""
                SELECT COUNT(*) FROM audit_file_results afr
                JOIN audit_runs ar ON ar.id = afr.run_id
                WHERE ar.started_at >= %s
            """, (since,))
            total_checks = cur.fetchone()[0] or 0

            # Top 5 violations by function_name + status
            cur.execute("""
                SELECT afr.function_name, afr.status, COUNT(*) AS cnt
                FROM audit_file_results afr
                JOIN audit_runs ar ON ar.id = afr.run_id
                WHERE ar.started_at >= %s
                  AND afr.status IN ('violation', 'critical')
                GROUP BY afr.function_name, afr.status
                ORDER BY cnt DESC
                LIMIT 5
            """, (since,))
            top_violations = [
                {'function': r[0], 'status': r[1], 'count': r[2]}
                for r in cur.fetchall()
            ]

            # Recent 10 audit runs
            cur.execute("""
                SELECT ar.id, ar.project_id, ar.overall_score, ar.overall_status,
                       ar.started_at, ar.completed_at,
                       (SELECT COUNT(*) FROM audit_file_results afr WHERE afr.run_id = ar.id AND afr.status = 'critical') AS critical_count,
                       (SELECT COUNT(*) FROM audit_file_results afr WHERE afr.run_id = ar.id AND afr.status = 'violation') AS violation_count,
                       (SELECT COUNT(*) FROM audit_file_results afr WHERE afr.run_id = ar.id) AS file_count
                FROM audit_runs ar
                WHERE ar.started_at >= %s
                ORDER BY ar.started_at DESC
                LIMIT 10
            """, (since,))
            recent_runs = [
                {
                    'id': r[0], 'project_id': r[1], 'score': float(r[2]) if r[2] else None,
                    'status': r[3], 'started_at': str(r[4]) if r[4] else None,
                    'completed_at': str(r[5]) if r[5] else None,
                    'critical_count': r[6] or 0, 'violation_count': r[7] or 0,
                    'file_count': r[8] or 0,
                }
                for r in cur.fetchall()
            ]

            # Project-level breakdown
            cur.execute("""
                SELECT ar.project_id,
                       COUNT(DISTINCT ar.id) AS runs,
                       ROUND(AVG(ar.overall_score)::numeric, 1) AS avg_score,
                       COUNT(CASE WHEN ar.overall_status = 'PASS' THEN 1 END) AS passed,
                       COUNT(CASE WHEN ar.overall_status = 'FAIL' THEN 1 END) AS failed
                FROM audit_runs ar
                WHERE ar.started_at >= %s
                  AND ar.project_id IS NOT NULL
                  AND ar.overall_score IS NOT NULL
                GROUP BY ar.project_id
                ORDER BY runs DESC
                LIMIT 20
            """, (since,))
            project_breakdown = [
                {
                    'project_id': r[0], 'runs': r[1],
                    'avg_score': float(r[2]) if r[2] else 0,
                    'passed': r[3] or 0, 'failed': r[4] or 0,
                }
                for r in cur.fetchall()
            ]

            # Critical alerts — recent CRITICAL findings
            cur.execute("""
                SELECT afr.function_name, afr.score, afr.filename,
                       ar.project_id, ar.started_at, afr.findings
                FROM audit_file_results afr
                JOIN audit_runs ar ON ar.id = afr.run_id
                WHERE ar.started_at >= %s
                  AND afr.status = 'critical'
                ORDER BY afr.score ASC
                LIMIT 10
            """, (since,))
            critical_alerts = [
                {
                    'function': r[0], 'score': float(r[1]) if r[1] else 0,
                    'filename': r[2], 'project_id': r[3],
                    'detected_at': str(r[4]) if r[4] else None,
                }
                for r in cur.fetchall()
            ]

            # Score distribution buckets
            cur.execute("""
                SELECT
                    CASE
                        WHEN overall_score >= 90 THEN '90-100'
                        WHEN overall_score >= 70 THEN '70-89'
                        WHEN overall_score >= 50 THEN '50-69'
                        WHEN overall_score >= 30 THEN '30-49'
                        ELSE '0-29'
                    END AS bucket,
                    COUNT(*) AS cnt
                FROM audit_runs
                WHERE started_at >= %s
                  AND overall_score IS NOT NULL
                GROUP BY bucket
                ORDER BY bucket DESC
            """, (since,))
            score_distribution = [
                {'bucket': r[0], 'count': r[1]} for r in cur.fetchall()
            ]

    return {
        'period_days': days,
        'overall': {
            'total_runs': total_runs,
            'passed': passed,
            'failed': stats_row[2] or 0,
            'pass_rate': pass_rate,
            'avg_score': float(stats_row[3]) if stats_row[3] else 0,
            'high_score_runs': stats_row[4] or 0,
            'total_file_checks': total_checks,
        },
        'top_violations': top_violations,
        'recent_runs': recent_runs,
        'project_breakdown': project_breakdown,
        'critical_alerts': critical_alerts,
        'score_distribution': score_distribution,
    }
