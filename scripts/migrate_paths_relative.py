#!/usr/bin/env python
"""Migrate absolute file paths in DB to BASE_DIR-relative paths (portable).

Converts stored_path / archive_path / docx_path / xlsx_path / zip_path /
file_path / original_stored_path values from e.g.
    D:\\PyCharm\\Local_AI\\data\\project_files\\2\\xxx.docx
to
    data/project_files/2/xxx.docx

Only values rooted at BASE_DIR are rewritten; others are left untouched
(already relative, temp paths, or foreign). Idempotent — safe to re-run.

Usage:
    python scripts/migrate_paths_relative.py --check   # dry-run (default)
    python scripts/migrate_paths_relative.py --apply   # write changes
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent.absolute()
BASE_PREFIX = str(BASE_DIR).replace('\\', '/').rstrip('/') + '/'

TARGETS = [
    ('project_files', 'stored_path', 'id'),
    ('project_file_versions', 'stored_path', 'id'),
    ('knowledge_lab_files', 'stored_path', 'id'),
    ('company_knowledge_base', 'stored_path', 'id'),
    ('archived_sessions', 'archive_path', 'thread_id'),
    ('user_files', 'original_stored_path', 'id'),
    ('recycle_bin', 'original_stored_path', 'id'),
    ('kb_recycle_bin', 'stored_path', 'id'),
    ('project_recycle_bin', 'stored_path', 'id'),
    ('audit_runs', 'docx_path', 'id'),
    ('audit_runs', 'xlsx_path', 'id'),
    ('batch_comparison_results', 'zip_path', 'id'),
    ('task_deposit_items', 'stored_path', 'id'),
]


def get_db_url():
    """Read DATABASE_URL from env or construct from .env."""
    db_url = os.getenv('DATABASE_URL', '')
    if db_url:
        return db_url
    host = os.getenv('PG_HOST', 'localhost')
    port = os.getenv('PG_PORT', '5433')
    user = os.getenv('PG_USER', 'localai')
    pwd = os.getenv('PG_PASSWORD', '')
    db = os.getenv('PG_DB', 'localai')
    return f'postgresql://{user}:{pwd}@{host}:{port}/{db}'


def to_rel(path: str) -> str:
    """Convert an absolute path under BASE_DIR to a forward-slash relative one."""
    if not path:
        return path
    p = str(path).replace('\\', '/')
    if p.startswith(BASE_PREFIX):
        return p[len(BASE_PREFIX):]
    return path


def main():
    apply = '--apply' in sys.argv
    import psycopg2
    conn = psycopg2.connect(get_db_url())
    total_changed = 0
    try:
        cur = conn.cursor()
        for table, column, pk in TARGETS:
            try:
                cur.execute(
                    f"SELECT {pk}, {column} FROM {table} "
                    f"WHERE {column} IS NOT NULL AND {column} != ''"
                )
                rows = cur.fetchall()
            except Exception as e:
                conn.rollback()
                print(f"  [skip] {table}.{column}: {e}")
                continue
            changed = 0
            for row_id, value in rows:
                rel = to_rel(value)
                if rel != value:
                    if apply:
                        cur.execute(
                            f"UPDATE {table} SET {column} = %s WHERE {pk} = %s",
                            (rel, row_id)
                        )
                    changed += 1
            total_changed += changed
            verb = 'UPDATE' if apply else 'WOULD UPDATE'
            print(f"  {table}.{column}: {len(rows)} rows, {changed} {verb}")
        if apply:
            conn.commit()
            print(f"\nApplied: {total_changed} paths converted to relative.")
        else:
            conn.rollback()
            print(f"\nDry-run: {total_changed} paths would be converted. "
                  f"Re-run with --apply to write.")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
