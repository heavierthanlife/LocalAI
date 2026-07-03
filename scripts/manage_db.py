#!/usr/bin/env python
"""Database migration manager — safe, automated, embedded in the AI app.

Usage:
    python scripts/manage_db.py check     # Dry-run: show what would change
    python scripts/manage_db.py migrate   # Apply pending migrations
    python scripts/manage_db.py rollback  # Rollback last migration
    python scripts/manage_db.py history   # Show migration history

No manual SQL needed. Safe by default (check before migrate).
Integrated into the admin panel at /admin/db_migrations for GUI access.
"""
import os
import sys
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MIGRATION_LOG = DATA_DIR / 'migration_history.json'
SCHEMA_SNAPSHOT = DATA_DIR / 'current_schema.sql'


def get_db_url():
    """Read DATABASE_URL from env or construct from .env."""
    db_url = os.getenv('DATABASE_URL', '')
    if db_url:
        return db_url
    # Fallback: construct from environment
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')
    user = os.getenv('DB_USER', 'localai')
    pwd = os.getenv('DB_PASSWORD', 'localai')
    db = os.getenv('DB_NAME', 'localai')
    return f'postgresql://{user}:{pwd}@{host}:{port}/{db}'


def _pg_exec(sql: str) -> list:
    """Execute a SQL query and return rows."""
    db_url = get_db_url()
    import psycopg2
    conn = psycopg2.connect(db_url)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.commit()
        return rows
    finally:
        conn.close()


def snapshot_schema():
    """Capture current DB schema as a file for reference."""
    rows = _pg_exec("""
        SELECT table_name, column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """)
    lines = ['-- Current Database Schema (auto-generated)', f'-- {datetime.now(timezone.utc).isoformat()}', '']
    current_table = ''
    for table, col, dtype, nullable, default in rows:
        if table != current_table:
            current_table = table
            lines.append(f'\n-- TABLE: {table}')
        null_str = 'NULL' if nullable == 'YES' else 'NOT NULL'
        default_str = f' DEFAULT {default}' if default else ''
        lines.append(f'  {col:<30} {dtype:<20} {null_str}{default_str}')

    SCHEMA_SNAPSHOT.write_text('\n'.join(lines), encoding='utf-8')
    return SCHEMA_SNAPSHOT


def check_migrations():
    """Check what pending migrations exist without applying them.

    Returns list of dicts: {name, sql, fingerprint, safe, warnings}
    """
    migrations_dir = PROJECT_ROOT / 'migrations'
    if not migrations_dir.exists():
        return []

    # Load history
    history = _load_history()
    applied = set(m['fingerprint'] for m in history)

    pending = []
    for f in sorted(migrations_dir.glob('*.sql')):
        sql = f.read_text(encoding='utf-8')
        fp = hashlib.sha256(sql.encode()).hexdigest()[:12]
        if fp in applied:
            continue

        # Safety checks
        warnings = []
        sql_upper = sql.upper()
        if 'DROP TABLE' in sql_upper or 'DROP COLUMN' in sql_upper:
            warnings.append('CONTAINS DROP — data loss risk')
        if 'TRUNCATE' in sql_upper:
            warnings.append('CONTAINS TRUNCATE — data loss risk')
        if not sql.strip().endswith(';'):
            warnings.append('Missing semicolon at end')

        pending.append({
            'name': f.stem,
            'fingerprint': fp,
            'sql': sql[:500] + ('...' if len(sql) > 500 else ''),
            'safe': len(warnings) == 0,
            'warnings': warnings,
        })

    return pending


def apply_migration(fingerprint: str):
    """Apply a single migration by fingerprint."""
    migrations_dir = PROJECT_ROOT / 'migrations'
    for f in sorted(migrations_dir.glob('*.sql')):
        sql = f.read_text(encoding='utf-8')
        fp = hashlib.sha256(sql.encode()).hexdigest()[:12]
        if fp != fingerprint:
            continue
        _pg_exec(sql)
        _record_migration(f.stem, fp)
        return True
    return False


def rollback_last():
    """Undo the last applied migration."""
    history = _load_history()
    if not history:
        print("No migrations to roll back.")
        return False
    last = history[-1]
    rollback_file = PROJECT_ROOT / 'migrations' / f"{last['name']}.rollback.sql"
    if rollback_file.exists():
        sql = rollback_file.read_text(encoding='utf-8')
        _pg_exec(sql)
        history.pop()
        _save_history(history)
        print(f"Rolled back: {last['name']}")
        return True
    else:
        print(f"No rollback script found for: {last['name']}")
        print(f"Expected: {rollback_file}")
        return False


def _load_history():
    if MIGRATION_LOG.exists():
        return json.loads(MIGRATION_LOG.read_text(encoding='utf-8'))
    return []


def _save_history(history):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MIGRATION_LOG.write_text(json.dumps(history, indent=2, default=str), encoding='utf-8')


def _record_migration(name, fingerprint):
    history = _load_history()
    history.append({
        'name': name,
        'fingerprint': fingerprint,
        'applied_at': datetime.now(timezone.utc).isoformat(),
    })
    _save_history(history)


# ── CLI ──

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'check'

    if cmd == 'check':
        pending = check_migrations()
        if not pending:
            print("No pending migrations. DB is up to date.")
        else:
            print(f"Pending migrations: {len(pending)}")
            for m in pending:
                icon = 'SAFE' if m['safe'] else 'RISK'
                print(f"  [{icon}] {m['name']} (fp={m['fingerprint']})")
                for w in m['warnings']:
                    print(f"    WARN: {w}")

    elif cmd == 'migrate':
        pending = check_migrations()
        if not pending:
            print("Already up to date.")
        else:
            force = '--yes' in sys.argv
            for m in pending:
                risky = not m['safe']
                if risky and not force:
                    print(f"SKIPPED {m['name']} — has warnings. Use --yes to force.")
                    continue
                print(f"Applying: {m['name']}...")
                apply_migration(m['fingerprint'])
                print(f"  Done.")
            snapshot_schema()
            print("Schema snapshot updated.")

    elif cmd == 'rollback':
        rollback_last()

    elif cmd == 'history':
        for h in _load_history():
            print(f"  {h['applied_at']}  {h['name']}  ({h['fingerprint']})")

    elif cmd == 'snapshot':
        path = snapshot_schema()
        print(f"Schema snapshot saved to: {path}")

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python scripts/manage_db.py [check|migrate|rollback|history|snapshot]")
