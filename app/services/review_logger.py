"""Reviewer workload logger — tracks all admin/auditor review actions.

Stores to data/ingest/review_log.json for lightweight persistence.
Each entry records who did what, when, and the count of items affected.
"""

import os, json, logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
INGEST_DIR = DATA_DIR / "ingest"
REVIEW_LOG_PATH = INGEST_DIR / "review_log.json"
_lock = Lock()


def _ensure_dir():
    os.makedirs(INGEST_DIR, exist_ok=True)


def log_review_action(user_id: str, username: str, role: str,
                      action_type: str, target: str, count: int = 0,
                      detail: str = ""):
    """Record a review action (domain approve, KB edit, etc.).

    Args:
        user_id: UUID of the reviewer
        username: display name
        role: 'admin' or 'auditor'
        action_type: 'domain_approve' | 'domain_reject' | 'kb_chunk_edit' |
                     'kb_approve' | 'kb_reject' | 'kb_cleanup'
        target: what was reviewed (task_id, word_list, etc.)
        count: number of items affected
        detail: extra context
    """
    _ensure_dir()
    entry = {
        'user_id': user_id,
        'username': username,
        'role': role,
        'action_type': action_type,
        'target': str(target)[:120],
        'count': count,
        'detail': detail[:200],
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        logs = []
        if REVIEW_LOG_PATH.exists():
            try:
                with open(REVIEW_LOG_PATH, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except Exception:
                pass
        logs.append(entry)
        # Keep last 500 entries
        if len(logs) > 500:
            logs = logs[-500:]
        with open(REVIEW_LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)


def get_review_log(limit: int = 50) -> list[dict]:
    """Return recent review log entries."""
    if not REVIEW_LOG_PATH.exists():
        return []
    try:
        with open(REVIEW_LOG_PATH, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        return logs[-limit:]
    except Exception:
        return []


def get_reviewer_workload() -> dict:
    """Summarize workload per reviewer (admin/auditor)."""
    logs = get_review_log(500)
    workload = {}
    for entry in logs:
        uid = entry.get('user_id', 'unknown')
        if uid not in workload:
            workload[uid] = {
                'username': entry.get('username', '?'),
                'role': entry.get('role', '?'),
                'total_actions': 0,
                'total_items': 0,
                'by_type': {},
                'last_action': entry['timestamp'],
            }
        w = workload[uid]
        w['total_actions'] += 1
        w['total_items'] += entry.get('count', 0)
        at = entry.get('action_type', 'unknown')
        w['by_type'][at] = w['by_type'].get(at, 0) + 1
    # Sort by total items desc
    return dict(sorted(workload.items(), key=lambda x: x[1]['total_items'], reverse=True))
