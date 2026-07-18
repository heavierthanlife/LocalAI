"""Multi-project compliance comparison service (U15).

Provides project×function matrix for cross-project analysis.
Features: heatmap data, pattern detection, aggregate comparisons.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from app.database import get_db_connection

logger = logging.getLogger(__name__)


def get_comparison_matrix(days: int = 90, project_ids: list[int] = None,
                          function_names: list[str] = None) -> dict:
    """Build a project×function compliance score matrix.

    Returns data suitable for a heatmap: rows=projects, columns=functions,
    values=average scores.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Build conditions
            conditions = ['ar.started_at >= %s']
            params = [since]
            if project_ids:
                conditions.append('ar.project_id = ANY(%s)')
                params.append(project_ids)
            if function_names:
                conditions.append('afr.function_name = ANY(%s)')
                params.append(function_names)

            where = ' AND '.join(conditions)

            cur.execute(f"""
                SELECT ar.project_id,
                       afr.function_name,
                       ROUND(AVG(afr.score)::numeric, 1) AS avg_score,
                       COUNT(*) AS check_count,
                       COUNT(CASE WHEN afr.status = 'pass' THEN 1 END) AS pass_count,
                       COUNT(CASE WHEN afr.status IN ('violation','critical') THEN 1 END) AS fail_count
                FROM audit_file_results afr
                JOIN audit_runs ar ON ar.id = afr.run_id
                WHERE {where}
                GROUP BY ar.project_id, afr.function_name
                ORDER BY ar.project_id, avg_score
            """, params)
            rows = cur.fetchall()

            # Get all distinct projects and functions
            cur.execute(f"""
                SELECT DISTINCT ar.project_id FROM audit_file_results afr
                JOIN audit_runs ar ON ar.id = afr.run_id
                WHERE {where} AND ar.project_id IS NOT NULL
                ORDER BY ar.project_id
            """, params)
            all_projects = [r[0] for r in cur.fetchall()]

            cur.execute(f"""
                SELECT DISTINCT afr.function_name FROM audit_file_results afr
                JOIN audit_runs ar ON ar.id = afr.run_id
                WHERE {where}
                ORDER BY afr.function_name
            """, params)
            all_functions = [r[0] for r in cur.fetchall()]

            # Build matrix: {project_id: {function_name: score}}
            matrix = {}
            for pid in all_projects:
                matrix[pid] = {fn: None for fn in all_functions}
            for r in rows:
                pid, fn, score, cnt, passes, fails = r[0], r[1], r[2], r[3], r[4], r[5]
                if pid not in matrix:
                    matrix[pid] = {fn: None for fn in all_functions}
                matrix[pid][fn] = {
                    'avg_score': float(score) if score is not None else None,
                    'check_count': cnt,
                    'pass_count': passes or 0,
                    'fail_count': fails or 0,
                }

            # Project summary rows
            cur.execute(f"""
                SELECT ar.project_id,
                       ROUND(AVG(ar.overall_score)::numeric, 1) AS avg_score,
                       COUNT(DISTINCT ar.id) AS runs,
                       COUNT(CASE WHEN ar.overall_status = 'PASS' THEN 1 END) AS passed_runs
                FROM audit_runs ar
                WHERE ar.started_at >= %s
                  AND ar.project_id IS NOT NULL
                  AND ar.overall_score IS NOT NULL
                  {'AND ar.project_id = ANY(%s)' if project_ids else ''}
                GROUP BY ar.project_id
                ORDER BY avg_score DESC
            """, [since] + ([project_ids] if project_ids else []))
            project_summaries = {
                r[0]: {
                    'avg_score': float(r[1]) if r[1] else 0,
                    'runs': r[2] or 0,
                    'passed_runs': r[3] or 0,
                    'pass_rate': round(r[3] / r[2] * 100, 1) if r[2] else 0,
                }
                for r in cur.fetchall()
            }

            # Function summary (most problematic functions across projects)
            cur.execute(f"""
                SELECT afr.function_name,
                       ROUND(AVG(afr.score)::numeric, 1) AS avg_score,
                       COUNT(*) AS total_checks,
                       COUNT(CASE WHEN afr.status IN ('violation','critical') THEN 1 END) AS problem_count
                FROM audit_file_results afr
                JOIN audit_runs ar ON ar.id = afr.run_id
                WHERE {where}
                GROUP BY afr.function_name
                ORDER BY problem_count DESC, avg_score ASC
            """, params)
            function_summaries = [
                {
                    'function': r[0],
                    'avg_score': float(r[1]) if r[1] else 0,
                    'total_checks': r[2] or 0,
                    'problem_count': r[3] or 0,
                    'problem_rate': round(r[3] / r[2] * 100, 1) if r[2] else 0,
                }
                for r in cur.fetchall()
            ]

    return {
        'period_days': days,
        'projects': all_projects,
        'functions': all_functions,
        'matrix': [
            {
                'project_id': pid,
                'summary': project_summaries.get(pid, {}),
                'function_scores': [
                    {'function': fn, **data} if data else {'function': fn, 'avg_score': None}
                    for fn, data in matrix.get(pid, {}).items()
                ],
            }
            for pid in all_projects
        ],
        'function_summaries': function_summaries,
        'project_summaries': [
            {'project_id': pid, **data}
            for pid, data in project_summaries.items()
        ],
    }
