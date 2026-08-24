"""Law change monitoring service (U14 MVP).

When a law version change is detected:
  1. Find articles affected by the law change
  2. Find cases citing those articles → impacted cases
  3. Find templates linked to those cases → impacted templates
  4. Return impact report

RSS/crawler deferred to optional phase.
"""

import json
import logging
from datetime import datetime, timezone
from app.database import get_db_connection

logger = logging.getLogger(__name__)


def create_change_event(law_id: int, from_version_id: int, to_version_id: int,
                         description: str, user_id: str = None) -> dict:
    """Record a law change event and compute impact."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO law_change_events
                    (law_id, from_version_id, to_version_id, description, submitted_by)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, created_at
            """, (law_id, from_version_id, to_version_id, description, user_id))
            event_id, created_at = cur.fetchone()
            conn.commit()

            impact = _compute_impact(cur, law_id)
            return {
                'event_id': event_id,
                'law_id': law_id,
                'from_version': from_version_id,
                'to_version': to_version_id,
                'description': description,
                'created_at': str(created_at) if created_at else None,
                'impact': impact,
            }


def compute_impact(law_id: int) -> dict:
    """Compute impact of a law change on the compliance ecosystem."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            return _compute_impact(cur, law_id)


def _compute_impact(cur, law_id: int) -> dict:
    """Find all affected cases and templates for a law change."""
    # Find articles under this law
    cur.execute(
        "SELECT id, article_label FROM law_articles WHERE law_id = %s",
        (law_id,)
    )
    articles = [(r[0], r[1]) for r in cur.fetchall()]
    if not articles:
        return {'affected_articles': 0, 'affected_cases': [], 'affected_templates': [],
                'total_affected_cases': 0, 'total_affected_templates': 0}

    article_ids = [a[0] for a in articles]

    # Find cases citing these articles
    cur.execute("""
        SELECT DISTINCT cl.case_id, ac.title, ac.severity, ac.is_resolved
        FROM case_law_links cl
        JOIN audit_cases ac ON ac.id = cl.case_id
        WHERE cl.article_id = ANY(%s)
        ORDER BY ac.severity DESC, ac.id
        LIMIT 30
    """, (article_ids,))
    affected_cases = [
        {
            'case_id': r[0], 'title': r[1], 'severity': r[2],
            'is_resolved': r[3],
        }
        for r in cur.fetchall()
    ]
    case_ids = [c['case_id'] for c in affected_cases]

    # Find templates linked to these cases
    affected_templates = []
    if case_ids:
        cur.execute("""
            SELECT DISTINCT ct.template_id, bt.name, bt.category
            FROM case_template_links ct
            JOIN bid_templates bt ON bt.id = ct.template_id
            WHERE ct.case_id = ANY(%s)
            ORDER BY bt.name
            LIMIT 20
        """, (case_ids,))
        affected_templates = [
            {'template_id': r[0], 'name': r[1], 'category': r[2]}
            for r in cur.fetchall()
        ]

    # Count all affected (not just limited)
    cur.execute(
        "SELECT COUNT(DISTINCT cl.case_id) FROM case_law_links cl WHERE cl.article_id = ANY(%s)",
        (article_ids,)
    )
    total_cases = cur.fetchone()[0] or 0

    total_templates = 0
    if total_cases:
        cur.execute("""
            SELECT COUNT(DISTINCT ct.template_id)
            FROM case_template_links ct
            JOIN case_law_links cl ON cl.case_id = ct.case_id
            WHERE cl.article_id = ANY(%s)
        """, (article_ids,))
        total_templates = cur.fetchone()[0] or 0

    return {
        'affected_articles': len(article_ids),
        'article_labels': [a[1] for a in articles[:5]],
        'affected_cases': affected_cases,
        'affected_templates': affected_templates,
        'total_affected_cases': total_cases,
        'total_affected_templates': total_templates,
        'severity_breakdown': _severity_breakdown(affected_cases),
    }


def _severity_breakdown(cases: list[dict]) -> dict:
    """Count cases by severity and resolution status."""
    result = {'open': 0, 'resolved': 0, 'critical': 0, 'violation': 0, 'warning': 0}
    for c in cases:
        if c['is_resolved']:
            result['resolved'] += 1
        else:
            result['open'] += 1
        severity = c.get('severity', '')
        if severity in result:
            result[severity] += 1
    return result


def get_change_history(law_id: int = None, limit: int = 20) -> list[dict]:
    """Get recent law change events."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if law_id:
                cur.execute("""
                    SELECT lce.id, lce.law_id, l.law_name, lce.from_version_id,
                           lce.to_version_id, lce.description, lce.submitted_by,
                           lce.created_at
                    FROM law_change_events lce
                    JOIN laws l ON l.id = lce.law_id
                    WHERE lce.law_id = %s
                    ORDER BY lce.created_at DESC LIMIT %s
                """, (law_id, limit))
            else:
                cur.execute("""
                    SELECT lce.id, lce.law_id, l.law_name, lce.from_version_id,
                           lce.to_version_id, lce.description, lce.submitted_by,
                           lce.created_at
                    FROM law_change_events lce
                    JOIN laws l ON l.id = lce.law_id
                    ORDER BY lce.created_at DESC LIMIT %s
                """, (limit,))
            return [
                {
                    'id': r[0], 'law_id': r[1], 'law_name': r[2],
                    'from_version_id': r[3], 'to_version_id': r[4],
                    'description': r[5], 'submitted_by': r[6],
                    'created_at': str(r[7]) if r[7] else None,
                }
                for r in cur.fetchall()
            ]
