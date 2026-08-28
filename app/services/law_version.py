"""Law version management: listing, activation, and diff computation."""
# ── Law Version Service ──
# Handles law version management: listing, activation, diff computation

import json
import logging
import difflib
from app.database import get_db_connection

logger = logging.getLogger(__name__)


def list_versions(law_id: int) -> list[dict]:
    """List all versions of a law."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT lv.id, lv.version_label, lv.version_date,
                       lv.is_current, lv.change_summary, lv.created_at,
                       (SELECT COUNT(*) FROM law_articles la
                        WHERE la.version_id = lv.id) AS article_count
                FROM law_versions lv
                WHERE lv.law_id = %s
                ORDER BY lv.version_date DESC, lv.id DESC
            """, (law_id,))
            rows = cur.fetchall()
            return [_version_row(row) for row in rows]


def get_version(version_id: int) -> dict | None:
    """Get a single version with its articles."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT lv.id, lv.version_label, lv.version_date,
                       lv.is_current, lv.change_summary, lv.created_at
                FROM law_versions lv
                WHERE lv.id = %s
            """, (version_id,))
            row = cur.fetchone()
            if not row:
                return None
            result = _version_row(row)
            cur.execute("""
                SELECT id, article_label, article_text, tags, sort_order
                FROM law_articles
                WHERE version_id = %s
                ORDER BY sort_order
            """, (version_id,))
            result['articles'] = [_article_row(r) for r in cur.fetchall()]
            return result


def activate_version(law_id: int, version_id: int) -> dict | None:
    """Set a version as current (deactivates all others for this law)."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM law_versions WHERE id = %s AND law_id = %s",
                (version_id, law_id)
            )
            if not cur.fetchone():
                return None
            cur.execute(
                "UPDATE law_versions SET is_current = FALSE WHERE law_id = %s",
                (law_id,)
            )
            cur.execute(
                "UPDATE law_versions SET is_current = TRUE WHERE id = %s",
                (version_id,)
            )
            conn.commit()
            return get_version(version_id)


def create_version(law_id: int, version_label: str, articles: list[dict],
                   version_date: str = None, change_summary: str = None) -> dict:
    """Create a new version for a law with its articles."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM law_masters WHERE id = %s", (law_id,)
            )
            if not cur.fetchone():
                raise ValueError(f"Law {law_id} not found")
            cur.execute("""
                INSERT INTO law_versions (law_id, version_label, version_date,
                                          is_current, change_summary)
                VALUES (%s, %s, %s, FALSE, %s)
                RETURNING id
            """, (law_id, version_label, version_date, change_summary))
            new_vid = cur.fetchone()[0]
            for i, art in enumerate(articles):
                cur.execute("""
                    INSERT INTO law_articles (version_id, article_label,
                                              article_text, tags, sort_order)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    new_vid,
                    art.get('article_label', art.get('article', '')),
                    art.get('article_text', art.get('text', '')),
                    art.get('tags', []),
                    i
                ))
            conn.commit()
            return get_version(new_vid)


def compute_diff(law_id: int, from_version_id: int, to_version_id: int) -> dict:
    """Compute a structured diff between two versions of a law."""
    from_arts = _load_articles(from_version_id)
    to_arts = _load_articles(to_version_id)
    from_map = {a['article_label']: a for a in from_arts}
    to_map = {a['article_label']: a for a in to_arts}
    all_labels = sorted(set(list(from_map.keys()) + list(to_map.keys())))

    diffs = []
    for label in all_labels:
        old_a = from_map.get(label)
        new_a = to_map.get(label)
        if old_a and not new_a:
            diffs.append({
                'article_label': label,
                'status': 'removed',
                'old_text': old_a['article_text'],
                'new_text': '',
                'diff_html': _make_diff_html(old_a['article_text'], ''),
            })
        elif new_a and not old_a:
            diffs.append({
                'article_label': label,
                'status': 'added',
                'old_text': '',
                'new_text': new_a['article_text'],
                'diff_html': _make_diff_html('', new_a['article_text']),
            })
        elif old_a['article_text'] != new_a['article_text']:
            diffs.append({
                'article_label': label,
                'status': 'changed',
                'old_text': old_a['article_text'],
                'new_text': new_a['article_text'],
                'diff_html': _make_diff_html(old_a['article_text'], new_a['article_text']),
            })

    diff_data = {
        'law_id': law_id,
        'from_version_id': from_version_id,
        'to_version_id': to_version_id,
        'changes': diffs,
        'summary': {
            'added': sum(1 for d in diffs if d['status'] == 'added'),
            'removed': sum(1 for d in diffs if d['status'] == 'removed'),
            'changed': sum(1 for d in diffs if d['status'] == 'changed'),
            'total': len(diffs),
        },
    }

    _save_diff(law_id, from_version_id, to_version_id, diff_data)
    return diff_data


def get_diff(law_id: int, from_version_id: int, to_version_id: int) -> dict | None:
    """Retrieve a cached diff, or compute if not cached."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT diff_data FROM law_version_diffs
                WHERE law_id = %s AND from_version_id = %s AND to_version_id = %s
            """, (law_id, from_version_id, to_version_id))
            row = cur.fetchone()
            if row:
                return row[0]
    return compute_diff(law_id, from_version_id, to_version_id)


def _load_articles(version_id: int) -> list[dict]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, article_label, article_text, tags, sort_order
                FROM law_articles
                WHERE version_id = %s
                ORDER BY sort_order
            """, (version_id,))
            return [_article_row(r) for r in cur.fetchall()]


def _save_diff(law_id, from_vid, to_vid, diff_data):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO law_version_diffs (law_id, from_version_id, to_version_id, diff_data)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (law_id, from_version_id, to_version_id)
                DO UPDATE SET diff_data = EXCLUDED.diff_data, created_at = NOW()
            """, (law_id, from_vid, to_vid, json.dumps(diff_data, ensure_ascii=False)))
            conn.commit()


def _make_diff_html(old_text: str, new_text: str) -> str:
    """Generate inline HTML diff using difflib.HtmlDiff."""
    d = difflib.HtmlDiff(wrapcolumn=80)
    return d.make_table(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromdesc='旧版本',
        todesc='新版本',
        context=True,
        numlines=3,
    )


def _version_row(row) -> dict:
    return {
        'id': row[0],
        'version_label': row[1],
        'version_date': str(row[2]) if row[2] else None,
        'is_current': row[3],
        'change_summary': row[4],
        'created_at': str(row[5]) if row[5] else None,
        'article_count': row[6] if len(row) > 6 else 0,
    }


def _article_row(row) -> dict:
    return {
        'id': row[0],
        'article_label': row[1],
        'article_text': row[2],
        'tags': row[3] or [],
        'sort_order': row[4],
    }
