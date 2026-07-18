"""Audit case library service (U13).

Auto-generates cases from VIOLATION/CRITICAL audit findings.
Supports CRUD, tagging, law linking, template linking, and statistics.
"""

import json
import logging
from app.database import get_db_connection

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = ('warning', 'violation', 'critical')


def create_case(title: str, severity: str = 'violation', category: str = '',
                tags: list[str] = None, description: str = '',
                resolution: str = '', project_id: int = None,
                user_id: str = None) -> dict:
    """Create a new audit case."""
    if severity not in SEVERITY_LEVELS:
        severity = 'violation'
    tags = tags or []

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO audit_cases (title, description, category, severity,
                    resolution, project_id, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                title[:200], description[:2000], category, severity,
                resolution, project_id, user_id,
            ))
            case_id = cur.fetchone()[0]

            for tag in tags:
                tag = tag.strip()
                if tag:
                    cur.execute("""
                        INSERT INTO case_tags (case_id, tag) VALUES (%s, %s)
                        ON CONFLICT (case_id, tag) DO NOTHING
                    """, (case_id, tag))

            conn.commit()

    return get_case(case_id)


def auto_generate_from_run(run_id: int) -> list[int]:
    """Scan audit_file_results for VIOLATION/CRITICAL findings and create cases.

    Returns list of newly created case IDs.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT afr.id, afr.run_id, afr.filename, afr.function_name,
                       afr.score, afr.status, afr.findings, afr.file_id,
                       ar.project_id, ar.user_id
                FROM audit_file_results afr
                JOIN audit_runs ar ON ar.id = afr.run_id
                WHERE afr.run_id = %s
                  AND afr.status IN ('violation', 'critical')
                  AND (afr.findings IS NOT NULL AND jsonb_array_length(afr.findings) > 0)
            """, (run_id,))
            rows = cur.fetchall()

        new_ids = []
        for row in rows:
            fid = row[0]
            cur.execute(
                "SELECT id FROM audit_cases WHERE source_finding_id = %s", (fid,)
            )
            if cur.fetchone():
                continue

            findings = row[6] if isinstance(row[6], list) else json.loads(row[6]) if row[6] else []
            if not findings:
                continue

            primary = findings[0]
            title = primary.get('rule_desc', primary.get('title', f'{row[2]} - {row[3]}'))
            description = primary.get('detail', primary.get('description', json.dumps(primary, ensure_ascii=False, default=str)))
            category = row[3] or 'general'
            severity = row[5]
            resolution = _suggest_resolution(primary, category)

            cur.execute("""
                INSERT INTO audit_cases (title, description, category, severity,
                    resolution, source_finding_id, source_run_id, project_id,
                    file_id, law_refs, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                title[:200], description[:2000], category, severity,
                resolution, fid, run_id, row[8], row[7],
                json.dumps(primary.get('law_refs', []), ensure_ascii=False),
                row[9],
            ))
            case_id = cur.fetchone()[0]

            tags = _extract_tags(primary, category)
            for tag in tags:
                cur.execute("""
                    INSERT INTO case_tags (case_id, tag) VALUES (%s, %s)
                    ON CONFLICT (case_id, tag) DO NOTHING
                """, (case_id, tag))

            new_ids.append(case_id)

        if new_ids:
            conn.commit()
            logger.info(f"Auto-generated {len(new_ids)} cases from run {run_id}")

        return new_ids


def list_cases(severity: str = None, category: str = None, resolved: bool = None,
               search: str = None, page: int = 1, per_page: int = 20) -> dict:
    """Paginated case list."""
    conditions = []
    params = []
    if severity:
        conditions.append('ac.severity = %s')
        params.append(severity)
    if category:
        conditions.append('ac.category = %s')
        params.append(category)
    if resolved is not None:
        conditions.append('ac.is_resolved = %s')
        params.append(resolved)
    if search:
        conditions.append('(ac.title ILIKE %s OR ac.description ILIKE %s)')
        params.extend([f'%{search}%', f'%{search}%'])

    where = ' AND '.join(conditions) if conditions else '1=1'
    offset = (page - 1) * per_page

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM audit_cases ac WHERE {where}", params)
            total = cur.fetchone()[0]

            cur.execute(f"""
                SELECT ac.id, ac.title, ac.category, ac.severity, ac.resolution,
                       ac.source_run_id, ac.project_id, ac.file_id,
                       ac.is_resolved, ac.created_by, ac.created_at, ac.updated_at,
                       COALESCE(jsonb_array_length(ac.law_refs), 0) AS law_count,
                       COALESCE(jsonb_array_length(ac.template_refs), 0) AS tpl_count,
                       (SELECT STRING_AGG(tag, ',') FROM case_tags WHERE case_id = ac.id) AS tags
                FROM audit_cases ac
                WHERE {where}
                ORDER BY ac.created_at DESC
                LIMIT %s OFFSET %s
            """, params + [per_page, offset])
            rows = cur.fetchall()

    items = []
    for r in rows:
        items.append({
            'id': r[0], 'title': r[1], 'category': r[2], 'severity': r[3],
            'resolution': r[4], 'source_run_id': r[5], 'project_id': r[6],
            'file_id': r[7], 'is_resolved': r[8], 'created_by': r[9],
            'created_at': str(r[10]) if r[10] else None,
            'updated_at': str(r[11]) if r[11] else None,
            'law_count': r[12], 'tpl_count': r[13],
            'tags': r[14].split(',') if r[14] else [],
        })

    return {
        'items': items, 'total': total, 'page': page,
        'per_page': per_page,
        'pages': max(1, (total + per_page - 1) // per_page),
    }


def get_case(case_id: int) -> dict | None:
    """Get a single case with full details."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, title, description, category, severity, resolution,
                       source_finding_id, source_run_id, project_id, file_id,
                       law_refs, template_refs, is_resolved, created_by,
                       created_at, updated_at
                FROM audit_cases WHERE id = %s
            """, (case_id,))
            row = cur.fetchone()
            if not row:
                return None
            result = _case_row(row)

            cur.execute("SELECT tag FROM case_tags WHERE case_id = %s", (case_id,))
            result['tags'] = [r[0] for r in cur.fetchall()]

            cur.execute("""
                SELECT cl.id, cl.article_id, la.article_label, la.article_text, cl.relation
                FROM case_law_links cl
                JOIN law_articles la ON la.id = cl.article_id
                WHERE cl.case_id = %s
            """, (case_id,))
            result['law_links'] = [_law_link_row(r) for r in cur.fetchall()]

            cur.execute("""
                SELECT ct.id, ct.template_id, bt.name AS tpl_name, ct.section_id, ct.relation
                FROM case_template_links ct
                JOIN bid_templates bt ON bt.id = ct.template_id
                WHERE ct.case_id = %s
            """, (case_id,))
            result['template_links'] = [_tpl_link_row(r) for r in cur.fetchall()]

            return result


def update_case(case_id: int, **fields) -> dict | None:
    """Update case fields (resolution, is_resolved, description)."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM audit_cases WHERE id = %s", (case_id,))
            if not cur.fetchone():
                return None
            updates = []
            params = []
            for field in ('title', 'description', 'resolution', 'is_resolved', 'severity', 'category'):
                if field in fields and fields[field] is not None:
                    updates.append(f'{field} = %s')
                    params.append(fields[field])
            if not updates:
                return get_case(case_id)
            updates.append('updated_at = NOW()')
            params.append(case_id)
            cur.execute(
                f"UPDATE audit_cases SET {', '.join(updates)} WHERE id = %s",
                params
            )
            conn.commit()
    return get_case(case_id)


def add_tags(case_id: int, tags: list[str]) -> dict:
    """Add tags to a case."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for tag in tags:
                cur.execute("""
                    INSERT INTO case_tags (case_id, tag) VALUES (%s, %s)
                    ON CONFLICT (case_id, tag) DO NOTHING
                """, (case_id, tag.strip()))
            conn.commit()
    return get_case(case_id)


def remove_tags(case_id: int, tags: list[str]) -> dict:
    """Remove tags from a case."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for tag in tags:
                cur.execute(
                    "DELETE FROM case_tags WHERE case_id = %s AND tag = %s",
                    (case_id, tag.strip())
                )
            conn.commit()
    return get_case(case_id)


def link_law(case_id: int, article_id: int, relation: str = 'cited') -> dict:
    """Link a law article to a case."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO case_law_links (case_id, article_id, relation)
                VALUES (%s, %s, %s)
                ON CONFLICT (case_id, article_id) DO UPDATE SET relation = EXCLUDED.relation
            """, (case_id, article_id, relation))
            conn.commit()
    return get_case(case_id)


def unlink_law(case_id: int, article_id: int) -> dict:
    """Unlink a law article from a case."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM case_law_links WHERE case_id = %s AND article_id = %s",
                (case_id, article_id)
            )
            conn.commit()
    return get_case(case_id)


def link_template(case_id: int, template_id: int, section_id: str = None, relation: str = 'related') -> dict:
    """Link a template section to a case."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO case_template_links (case_id, template_id, section_id, relation)
                VALUES (%s, %s, %s, %s)
            """, (case_id, template_id, section_id, relation))
            conn.commit()
    return get_case(case_id)


def unlink_template(case_id: int, link_id: int) -> dict:
    """Remove a template link from a case."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM case_template_links WHERE id = %s",
                (link_id,)
            )
            conn.commit()
    return get_case(case_id)


def get_stats() -> dict:
    """Get case library statistics."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT severity, COUNT(*) FROM audit_cases
                WHERE is_resolved = FALSE
                GROUP BY severity ORDER BY COUNT(*) DESC
            """)
            by_severity = {r[0]: r[1] for r in cur.fetchall()}

            cur.execute("""
                SELECT category, COUNT(*) FROM audit_cases
                WHERE is_resolved = FALSE
                GROUP BY category ORDER BY COUNT(*) DESC LIMIT 10
            """)
            by_category = [{'category': r[0], 'count': r[1]} for r in cur.fetchall()]

            cur.execute("""
                SELECT ct.tag, COUNT(*) AS cnt
                FROM case_tags ct
                JOIN audit_cases ac ON ac.id = ct.case_id
                WHERE ac.is_resolved = FALSE
                GROUP BY ct.tag ORDER BY cnt DESC LIMIT 10
            """)
            top_tags = [{'tag': r[0], 'count': r[1]} for r in cur.fetchall()]

            cur.execute("SELECT COUNT(*) FROM audit_cases WHERE is_resolved = FALSE")
            open_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM audit_cases WHERE is_resolved = TRUE")
            resolved_count = cur.fetchone()[0]

            # Top violation types linked to templates
            cur.execute("""
                SELECT bt.name, bt.id, COUNT(*) AS cnt
                FROM case_template_links ctl
                JOIN bid_templates bt ON bt.id = ctl.template_id
                GROUP BY bt.name, bt.id ORDER BY cnt DESC LIMIT 10
            """)
            top_template_issues = [
                {'template_name': r[0], 'template_id': r[1], 'case_count': r[2]}
                for r in cur.fetchall()
            ]

    return {
        'open': open_count,
        'resolved': resolved_count,
        'total': open_count + resolved_count,
        'by_severity': by_severity,
        'by_category': by_category,
        'top_tags': top_tags,
        'top_template_issues': top_template_issues,
    }


def delete_case(case_id: int) -> bool:
    """Delete a case with cascade (tags/links handled by FK)."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM audit_cases WHERE id = %s", (case_id,))
            conn.commit()
            return cur.rowcount > 0


def _suggest_resolution(finding: dict, category: str) -> str:
    """Suggest a resolution based on finding type."""
    rule_desc = finding.get('rule_desc', finding.get('title', ''))
    detail = finding.get('detail', finding.get('suggestion', ''))
    if rule_desc:
        return f'核查 "{rule_desc}" 相关条款，确保投标文件符合要求'
    if detail:
        return detail[:300]
    return f'审查 {category} 相关的投标文件内容，确认合规性'


def _extract_tags(finding: dict, category: str) -> list[str]:
    """Extract tags from finding metadata."""
    tags = set()
    tags.add(category)
    rule_desc = finding.get('rule_desc', finding.get('title', ''))
    for kw in ('资质', '资格', '业绩', '财务', '报价', '工期', '技术方案', '项目经理',
               '投标保证金', '联合体', '分包', '废标', '串通', '围标'):
        if kw in rule_desc:
            tags.add(kw)
    return list(tags)[:5]


def _case_row(row) -> dict:
    return {
        'id': row[0], 'title': row[1], 'description': row[2],
        'category': row[3], 'severity': row[4], 'resolution': row[5],
        'source_finding_id': row[6], 'source_run_id': row[7],
        'project_id': row[8], 'file_id': row[9],
        'law_refs': row[10] if isinstance(row[10], list) else json.loads(row[10]) if row[10] else [],
        'template_refs': row[11] if isinstance(row[11], list) else json.loads(row[11]) if row[11] else [],
        'is_resolved': row[12], 'created_by': row[13],
        'created_at': str(row[14]) if row[14] else None,
        'updated_at': str(row[15]) if row[15] else None,
    }


def _law_link_row(row) -> dict:
    return {
        'id': row[0], 'article_id': row[1], 'article_label': row[2],
        'article_text': row[3][:200] if row[3] else '', 'relation': row[4],
    }


def _tpl_link_row(row) -> dict:
    return {
        'id': row[0], 'template_id': row[1], 'template_name': row[2],
        'section_id': row[3], 'relation': row[4],
    }
