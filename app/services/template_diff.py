# ── Template Diff Service ──
# Computes structured diffs between bid_template_versions

import json
import difflib
import logging
from app.database import get_db_connection

logger = logging.getLogger(__name__)


def compute_template_diff(template_id: int, from_version_id: int, to_version_id: int) -> dict:
    """Compute a structured diff between two template version snapshots."""
    from_snapshot = _load_snapshot(from_version_id)
    to_snapshot = _load_snapshot(to_version_id)
    if not from_snapshot or not to_snapshot:
        raise ValueError("Version not found")

    from_sections = _build_section_map(from_snapshot.get('sections', []))
    to_sections = _build_section_map(to_snapshot.get('sections', []))
    all_ids = sorted(set(list(from_sections.keys()) + list(to_sections.keys())))

    changes = []
    for sec_id in all_ids:
        old_sec = from_sections.get(sec_id)
        new_sec = to_sections.get(sec_id)

        if old_sec and not new_sec:
            changes.append({
                'id': sec_id,
                'title': old_sec.get('title', ''),
                'status': 'removed',
                'old_content': old_sec.get('content', ''),
                'new_content': '',
                'diff_html': _make_diff_html(old_sec.get('content', ''), ''),
            })
        elif new_sec and not old_sec:
            changes.append({
                'id': sec_id,
                'title': new_sec.get('title', ''),
                'status': 'added',
                'old_content': '',
                'new_content': new_sec.get('content', ''),
                'diff_html': _make_diff_html('', new_sec.get('content', '')),
            })
        else:
            old_title = old_sec.get('title', '')
            new_title = new_sec.get('title', '')
            old_content = old_sec.get('content', '')
            new_content = new_sec.get('content', '')
            old_level = old_sec.get('level', 1)
            new_level = new_sec.get('level', 1)

            title_changed = old_title != new_title
            content_changed = old_content != new_content
            level_changed = old_level != new_level

            if title_changed or content_changed or level_changed:
                changes.append({
                    'id': sec_id,
                    'title': new_title or old_title,
                    'status': 'changed',
                    'old_content': json.dumps({'title': old_title, 'content': old_content, 'level': old_level}, ensure_ascii=False),
                    'new_content': json.dumps({'title': new_title, 'content': new_content, 'level': new_level}, ensure_ascii=False),
                    'diff_html': _make_diff_html(
                        f"[{old_title}]\n{old_content}",
                        f"[{new_title}]\n{new_content}",
                    ),
                })

    return {
        'template_id': template_id,
        'from_version_id': from_version_id,
        'to_version_id': to_version_id,
        'changes': changes,
        'summary': {
            'added': sum(1 for c in changes if c['status'] == 'added'),
            'removed': sum(1 for c in changes if c['status'] == 'removed'),
            'changed': sum(1 for c in changes if c['status'] == 'changed'),
            'total': len(changes),
        },
    }


def _load_snapshot(version_id: int) -> dict | None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT snapshot FROM bid_template_versions WHERE id = %s",
                (version_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            snapshot = row[0]
            if isinstance(snapshot, str):
                return json.loads(snapshot)
            return snapshot


def _build_section_map(sections: list[dict]) -> dict:
    """Build a map of section ID → section dict."""
    result = {}
    for sec in sections:
        sec_id = sec.get('id') or sec.get('title', '')
        result[sec_id] = sec
    return result


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
