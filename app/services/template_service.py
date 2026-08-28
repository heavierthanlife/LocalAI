"""Bid template CRUD, version snapshot, and .docx import/export."""
# ── Bid Template Service ──
# CRUD + version snapshot + .docx import/export

import json
import logging
import difflib
from app.database import get_db_connection

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ('工程', '货物', '服务')


def list_templates(category: str = None, tag: str = None, search: str = None,
                   page: int = 1, per_page: int = 20) -> dict:
    """Paginated list of bid templates."""
    conditions = ['is_active = TRUE']
    params = []
    if category:
        conditions.append('category = %s')
        params.append(category)
    if tag:
        conditions.append('%s = ANY(tags)')
        params.append(tag)
    if search:
        conditions.append('(name ILIKE %s OR description ILIKE %s)')
        params.extend([f'%{search}%', f'%{search}%'])

    where = ' AND '.join(conditions)
    offset = (page - 1) * per_page

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM bid_templates WHERE {where}", params
            )
            total = cur.fetchone()[0]

            cur.execute(f"""
                SELECT id, name, category, description, tags, is_active, version,
                       created_at, updated_at,
                       (SELECT COUNT(*) FROM bid_template_versions
                        WHERE template_id = bid_templates.id) AS version_count
                FROM bid_templates
                WHERE {where}
                ORDER BY updated_at DESC
                LIMIT %s OFFSET %s
            """, params + [per_page, offset])
            rows = cur.fetchall()

    items = [_template_list_row(r) for r in rows]
    return {
        'items': items,
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': max(1, (total + per_page - 1) // per_page),
    }


def get_template(template_id: int) -> dict | None:
    """Get a single template with full sections."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, category, description, sections, tags,
                       is_active, version, created_by, created_at, updated_at
                FROM bid_templates WHERE id = %s
            """, (template_id,))
            row = cur.fetchone()
            if not row:
                return None
            return _template_full_row(row)


def create_template(name: str, category: str, sections: list[dict],
                    description: str = None, tags: list[str] = None,
                    created_by: str = None) -> dict:
    """Create a new bid template."""
    if category not in DEFAULT_CATEGORIES:
        raise ValueError(f"Invalid category: {category}. Must be one of {DEFAULT_CATEGORIES}")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bid_templates (name, category, description, sections,
                                            tags, created_by, version)
                VALUES (%s, %s, %s, %s, %s, %s, 1)
                RETURNING id
            """, (
                name, category, description,
                json.dumps(sections, ensure_ascii=False),
                tags or [], created_by,
            ))
            tid = cur.fetchone()[0]
            _create_version(cur, tid, 'v1 (初始创建)', sections, None, created_by)
            conn.commit()
    return get_template(tid)


def update_template(template_id: int, **fields) -> dict | None:
    """Update a template. Auto-creates a version snapshot if sections changed."""
    existing = get_template(template_id)
    if not existing:
        return None

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            updates = []
            params = []
            new_version_needed = False

            for field in ('name', 'category', 'description', 'tags', 'is_active'):
                if field in fields:
                    if field == 'category' and fields[field] not in DEFAULT_CATEGORIES:
                        raise ValueError(f"Invalid category: {fields[field]}")
                    updates.append(f'{field} = %s')
                    val = fields[field]
                    if field == 'tags' and not isinstance(val, list):
                        val = [val]
                    params.append(val)

            if 'sections' in fields:
                new_sections = fields['sections']
                old_sections = existing.get('sections', [])
                if json.dumps(new_sections, sort_keys=True, ensure_ascii=False) != json.dumps(old_sections, sort_keys=True, ensure_ascii=False):
                    new_version_needed = True
                    updates.append('sections = %s')
                    params.append(json.dumps(new_sections, ensure_ascii=False))
                    updates.append('version = version + 1')

            if not updates:
                return existing

            updates.append('updated_at = NOW()')
            params.append(template_id)

            cur.execute(
                f"UPDATE bid_templates SET {', '.join(updates)} WHERE id = %s",
                params
            )

            if new_version_needed:
                new_ver = existing['version'] + 1
                _create_version(
                    cur, template_id, f'v{new_ver}',
                    fields['sections'],
                    fields.get('change_summary'),
                    fields.get('created_by'),
                )

            conn.commit()
    return get_template(template_id)


def delete_template(template_id: int) -> bool:
    """Soft-delete a template."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE bid_templates SET is_active = FALSE, updated_at = NOW() WHERE id = %s",
                (template_id,)
            )
            conn.commit()
            return cur.rowcount > 0


def list_versions(template_id: int) -> list[dict]:
    """List all versions of a template."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, template_id, version_label, snapshot, change_summary,
                       created_by, created_at
                FROM bid_template_versions
                WHERE template_id = %s
                ORDER BY id DESC
            """, (template_id,))
            return [_version_row(r) for r in cur.fetchall()]


def get_version(version_id: int) -> dict | None:
    """Get a single template version."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, template_id, version_label, snapshot, change_summary,
                       created_by, created_at
                FROM bid_template_versions WHERE id = %s
            """, (version_id,))
            row = cur.fetchone()
            if not row:
                return None
            return _version_row(row)


def import_from_docx_preview(file_bytes: bytes, filename: str) -> dict:
    """Parse a .docx and return a preview for user confirmation."""
    from docx import Document as DocxDocument
    import io
    import re

    doc = DocxDocument(io.BytesIO(file_bytes))
    sections = []
    current_section = None
    chapter_pattern = re.compile(
        r'^(第[一二三四五六七八九十百千]+[章节条]|'
        r'[一二三四五六七八九十]+[、，．.]|'
        r'\d+[\.\、]|'
        r'（[一二三四五六七八九十]+）)'
    )

    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue
        style_name = para.style.name if para.style else ''

        is_heading = ('Heading' in style_name or
                      'heading' in style_name or
                      '标题' in style_name or
                      chapter_pattern.match(text))

        level = 1
        if 'Heading 2' in style_name or '标题 2' in style_name:
            level = 2
        elif 'Heading 3' in style_name or '标题 3' in style_name:
            level = 3

        if is_heading:
            if current_section:
                sections.append(current_section)
            current_section = {
                'id': f'sec-{len(sections)+1}',
                'title': text,
                'content': '',
                'level': level,
                'order': len(sections) + 1,
            }
        else:
            if current_section:
                current_section['content'] += text + '\n'
            else:
                current_section = {
                    'id': f'sec-{len(sections)+1}',
                    'title': f'段落 {len(sections)+1}',
                    'content': text + '\n',
                    'level': 1,
                    'order': len(sections) + 1,
                }

    if current_section:
        sections.append(current_section)

    for sec in sections:
        sec['content'] = sec['content'].strip()

    detected_title = filename.rsplit('.', 1)[0]
    title_guess = next((s['title'] for s in sections if s['level'] == 1), None)
    if title_guess and len(title_guess) < 50:
        detected_title = title_guess

    category_guess = _guess_category(sections)

    return {
        'detected_title': detected_title,
        'category_guess': category_guess,
        'sections': sections,
        'section_count': len(sections),
    }


def _create_version(cur, template_id, label, sections, change_summary, created_by):
    cur.execute("""
        INSERT INTO bid_template_versions
            (template_id, version_label, snapshot, change_summary, created_by)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        template_id, label,
        json.dumps({'sections': sections}, ensure_ascii=False),
        change_summary, created_by,
    ))


def _guess_category(sections):
    combined = ' '.join(s['title'] + ' ' + s['content'] for s in sections)
    if any(kw in combined for kw in ('工程', '施工', '建筑', '建设')):
        return '工程'
    if any(kw in combined for kw in ('货物', '设备', '产品', '材料', '采购')):
        return '货物'
    if any(kw in combined for kw in ('服务', '咨询', '维护', '外包')):
        return '服务'
    return '工程'


def _template_list_row(row) -> dict:
    return {
        'id': row[0],
        'name': row[1],
        'category': row[2],
        'description': row[3],
        'tags': row[4] or [],
        'is_active': row[5],
        'version': row[6],
        'created_at': str(row[7]) if row[7] else None,
        'updated_at': str(row[8]) if row[8] else None,
        'version_count': row[9] if len(row) > 9 else 0,
    }


def _template_full_row(row) -> dict:
    return {
        'id': row[0],
        'name': row[1],
        'category': row[2],
        'description': row[3],
        'sections': row[4] if isinstance(row[4], list) else json.loads(row[4]) if row[4] else [],
        'tags': row[5] or [],
        'is_active': row[6],
        'version': row[7],
        'created_by': row[8],
        'created_at': str(row[9]) if row[9] else None,
        'updated_at': str(row[10]) if row[10] else None,
    }


def _version_row(row) -> dict:
    return {
        'id': row[0],
        'template_id': row[1],
        'version_label': row[2],
        'snapshot': row[3] if isinstance(row[3], dict) else json.loads(row[3]) if row[3] else {},
        'change_summary': row[4],
        'created_by': row[5],
        'created_at': str(row[6]) if row[6] else None,
    }
