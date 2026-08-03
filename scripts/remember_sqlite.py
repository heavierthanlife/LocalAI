"""
SQLite + FTS5 backend for .remember/ session continuity.

Commands:
  sync            Sync .remember/ markdown files into SQLite
  search <q>      FTS5 full-text search across all remembered content
  status          Show database stats
  sessions        List non-archived sessions, latest first
  archive <id>    Mark a session as archived

Usage:
  python scripts/remember_sqlite.py sync
  python scripts/remember_sqlite.py search "race condition"
  python scripts/remember_sqlite.py status
  python scripts/remember_sqlite.py sessions
  python scripts/remember_sqlite.py archive 3
"""

import argparse
import glob
import os
import re
import sqlite3
import sys
from datetime import datetime

REMEMBER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".remember")
DB_PATH = os.path.join(REMEMBER_DIR, "remember.db")


def get_db():
    os.makedirs(REMEMBER_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            branch TEXT,
            saved_at TEXT,
            handoff_text TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS today_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_date TEXT UNIQUE,
            content TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE,
            severity TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'unresolved',
            content TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
            title, content, category,
            tokenize='unicode61'
        );
    """)
    for col, dtype in [
        ("branch", "TEXT DEFAULT 'unknown'"),
        ("saved_at", "TEXT"),
        ("handoff_text", "TEXT"),
        ("archived", "INTEGER DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass
    conn.commit()


def parse_frontmatter(text):
    m = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', text, re.DOTALL)
    if m:
        fm = {}
        for line in m.group(1).strip().split('\n'):
            if ':' in line:
                k, v = line.split(':', 1)
                fm[k.strip()] = v.strip()
        return fm, m.group(2).strip()
    return {}, text.strip()


def sync_sessions(conn):
    conn.execute("DELETE FROM docs_fts WHERE category = 'handoff'")

    handoff_path = os.path.join(REMEMBER_DIR, "handoff.md")
    if not os.path.exists(handoff_path):
        return

    with open(handoff_path, encoding="utf-8") as f:
        text = f.read()

    fm, body = parse_frontmatter(text)
    saved_at = fm.get("saved_at", datetime.now().isoformat())
    branch = fm.get("branch", "unknown")

    cur = conn.execute("SELECT id FROM sessions WHERE saved_at = ? AND branch = ?", (saved_at, branch))
    if not cur.fetchone():
        conn.execute(
            "INSERT INTO sessions (branch, saved_at, handoff_text, handoff_content) VALUES (?, ?, ?, ?)",
            (branch, saved_at, body, body),
        )

    title = f"Handoff: {branch} @ {saved_at}"
    conn.execute(
        "INSERT INTO docs_fts (title, content, category) VALUES (?, ?, ?)",
        (title, body, "handoff"),
    )


def sync_today_logs(conn):
    conn.execute("DELETE FROM docs_fts WHERE category = 'today_log'")

    for path in sorted(glob.glob(os.path.join(REMEMBER_DIR, "today-*.md"))):
        basename = os.path.basename(path)
        log_date = basename.replace("today-", "").replace(".md", "").replace(".done", "")

        with open(path, encoding="utf-8") as f:
            content = f.read()

        conn.execute(
            "INSERT OR REPLACE INTO today_logs (log_date, content) VALUES (?, ?)",
            (log_date, content),
        )

        conn.execute(
            "INSERT INTO docs_fts (title, content, category) VALUES (?, ?, ?)",
            (f"Log: {log_date}", content, "today_log"),
        )


def sync_findings(conn):
    conn.execute("DELETE FROM docs_fts WHERE category = 'finding'")

    findings_dir = os.path.join(REMEMBER_DIR, "findings")
    if not os.path.isdir(findings_dir):
        return

    for path in sorted(glob.glob(os.path.join(findings_dir, "*.md"))):
        slug = os.path.basename(path).replace(".md", "")

        cur = conn.execute("SELECT id FROM findings WHERE slug = ?", (slug,))
        if not cur.fetchone():
            with open(path, encoding="utf-8") as f:
                text = f.read()

            fm, body = parse_frontmatter(text)
            severity = fm.get("severity", "medium")
            status = fm.get("status", "unresolved")
            created_at = fm.get("found", datetime.now().isoformat())

            conn.execute(
                "INSERT INTO findings (slug, severity, status, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (slug, severity, status, body, created_at),
            )

        cur = conn.execute("SELECT severity, content FROM findings WHERE slug = ?", (slug,))
        row = cur.fetchone()
        conn.execute(
            "INSERT INTO docs_fts (title, content, category) VALUES (?, ?, ?)",
            (f"Finding: {slug}", f"[{row['severity']}] {row['content']}", "finding"),
        )


def cmd_sync():
    conn = get_db()
    try:
        init_db(conn)
        sync_sessions(conn)
        sync_today_logs(conn)
        sync_findings(conn)
        conn.commit()

        cur = conn.execute("SELECT COUNT(*) FROM docs_fts")
        count = cur.fetchone()[0]
        print(f"Synced {count} documents to {DB_PATH}")
    finally:
        conn.close()


def cmd_search(query):
    conn = get_db()
    try:
        sanitized = re.sub(r'[^\w\u4e00-\u9fff\s-]', ' ', query).strip()
        if not sanitized:
            print("No valid search terms.")
            return

        cur = conn.execute(
            "SELECT title, category, snippet(docs_fts, 1, '>>>', '<<<', '...', 64) AS snippet "
            "FROM docs_fts WHERE docs_fts MATCH ? ORDER BY rank LIMIT 20",
            (sanitized,),
        )
        rows = cur.fetchall()
        if not rows:
            print(f"No results for '{query}'.")
            return

        print(f"Results for '{query}':\n")
        for r in rows:
            print(f"  [{r['category']}] {r['title']}")
            print(f"    {r['snippet']}\n")
    finally:
        conn.close()


def cmd_sessions():
    conn = get_db()
    try:
        init_db(conn)
        cur = conn.execute(
            "SELECT id, branch, saved_at, length(handoff_text) as len "
            "FROM sessions WHERE archived != 1 "
            "ORDER BY saved_at DESC LIMIT 10"
        )
        rows = cur.fetchall()
        if not rows:
            print("No sessions found.")
            return

        print(f"Sessions ({len(rows)}):\n")
        for r in rows:
            print(f"  #{r['id']} | {r['branch']} | {r['saved_at']} | {r['len']} bytes")
        print("\n  Run sessions <id> to view, or sessions <id> resume to load.")
    finally:
        conn.close()


def cmd_session_detail(session_id, resume=False):
    conn = get_db()
    try:
        init_db(conn)
        cur = conn.execute(
            "SELECT id, branch, saved_at, handoff_text FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            print(f"Session #{session_id} not found.")
            return

        if resume:
            handoff_path = os.path.join(REMEMBER_DIR, "handoff.md")
            with open(handoff_path, "w", encoding="utf-8") as f:
                f.write(f"---\nsaved_at: {row['saved_at']}\nbranch: {row['branch']}\n---\n\n")
                f.write(row["handoff_text"])
            print(f"Session #{session_id} loaded as current handoff ({row['saved_at']}).")
            print("Run /resume to view the full context.")
        else:
            print(f"Session #{session_id} | {row['branch']} @ {row['saved_at']}")
            print("─" * 60)
            print(row["handoff_text"])
            print(f"\nRun sessions {session_id} resume to load this session.")
    finally:
        conn.close()


def cmd_status():
    conn = get_db()
    try:
        init_db(conn)
        cur = conn.execute("SELECT COUNT(*) FROM sessions WHERE archived != 1")
        active = cur.fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) FROM sessions WHERE archived = 1")
        archived = cur.fetchone()[0]
        print(f"  sessions: {active} active, {archived} archived")
        for table in ("today_logs", "findings", "docs_fts"):
            cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"  {table}: {count} rows")

        cur = conn.execute(
            "SELECT branch, saved_at FROM sessions WHERE archived != 1 ORDER BY saved_at DESC LIMIT 3"
        )
        print("\n  Recent sessions:")
        for r in cur.fetchall():
            print(f"    - {r['branch']} @ {r['saved_at']}")

        cur = conn.execute(
            "SELECT slug, severity, status FROM findings WHERE status = 'unresolved' LIMIT 5"
        )
        print("\n  Unresolved findings:")
        for r in cur.fetchall():
            print(f"    [{r['severity']}] {r['slug']} ({r['status']})")
    finally:
        conn.close()


def cmd_archive(session_id):
    conn = get_db()
    try:
        init_db(conn)
        cur = conn.execute("SELECT id, branch, saved_at FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        if not row:
            print(f"Session #{session_id} not found.")
            return
        cur = conn.execute("SELECT archived FROM sessions WHERE id = ?", (session_id,))
        if cur.fetchone()["archived"]:
            print(f"Session #{session_id} is already archived.")
            return
        conn.execute("UPDATE sessions SET archived = 1 WHERE id = ?", (session_id,))
        conn.commit()
        print(f"Session #{session_id} ({row['branch']} @ {row['saved_at']}) archived.")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="SQLite+FTS5 backend for .remember/")
    parser.add_argument("command", choices=["sync", "search", "status", "sessions", "archive"])
    parser.add_argument("arg", nargs="?", help="Query (for 'search'), session ID (for 'archive'/'sessions'), or omitted")
    parser.add_argument("sub", nargs="?", help="'resume' for sessions <id> resume")
    args = parser.parse_args()

    if args.command == "sync":
        cmd_sync()
    elif args.command == "search":
        if not args.arg:
            print("Usage: python scripts/remember_sqlite.py search <query>")
            sys.exit(1)
        cmd_search(args.arg)
    elif args.command == "status":
        cmd_status()
    elif args.command == "sessions":
        if args.arg is not None:
            try:
                sid = int(args.arg)
            except ValueError:
                print(f"Invalid session ID: {args.arg}")
                sys.exit(1)
            cmd_session_detail(sid, resume=(args.sub == "resume"))
        else:
            cmd_sessions()
    elif args.command == "archive":
        if args.arg is None:
            print("Usage: python scripts/remember_sqlite.py archive <session_id>")
            sys.exit(1)
        try:
            sid = int(args.arg)
        except ValueError:
            print(f"Invalid session ID: {args.arg}")
            sys.exit(1)
        cmd_archive(sid)


if __name__ == "__main__":
    main()
