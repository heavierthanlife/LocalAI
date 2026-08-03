"""Recover data from filesystem to new PostgreSQL database (port 5433).

Usage:
    python scripts/recover_all.py

Requires:
    - Docker PostgreSQL running on localhost:5433 (user=localai, pass=Seven0413)
    - psycopg2
"""
import hashlib
import json
import mimetypes
import os
import sys
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import execute_values

# ── Config ──
NEW_DB = dict(dbname=os.environ.get("PG_DB", "localai"),
              user=os.environ.get("PG_USER", "localai"),
              password=os.environ.get("PG_PASSWORD", "localai"),
              host=os.environ.get("PG_HOST", "localhost"),
              port=int(os.environ.get("PG_PORT", "5433")))
OLD_DB = dict(dbname=os.environ.get("OLD_PG_DB", "test_chatbot"),
              user=os.environ.get("OLD_PG_USER", "postgres"),
              password=os.environ.get("OLD_PG_PASSWORD") or os.environ.get("PG_PASSWORD", "postgres"),
              host=os.environ.get("OLD_PG_HOST", "localhost"),
              port=int(os.environ.get("OLD_PG_PORT", "5432")))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADMIN_UUID = "934bb7fe-294c-4da7-bc9e-a5cf041ffc82"
RECOVERY_THREAD_ID = "recovery-session-001"
RECOVERY_SESSION_TITLE = "Recovered-Files-Session"

log = lambda msg: print(f"[{datetime.now():%H:%M:%S}] {msg}")


def connect(db_conf, name="DB"):
    try:
        conn = psycopg2.connect(**db_conf)
        log(f"Connected to {name}")
        return conn
    except Exception as e:
        log(f"ERROR connecting to {name}: {e}")
        sys.exit(1)


# ════════════════════════════════════════════════════════════
# Step 1: Import old users (CEO/COO) from old DB
# ════════════════════════════════════════════════════════════
def step1_import_old_users():
    log("─" * 50)
    log("Step 1: Import old CEO/COO users")
    try:
        old_conn = connect(OLD_DB, "Old DB (5432)")
        cur = old_conn.cursor()
        cur.execute("SELECT user_id, username, pin_hash, role FROM users")
        old_users = cur.fetchall()
        old_conn.close()
    except Exception as e:
        log(f"WARNING: Cannot connect to old DB: {e}")
        log("Skipping user import")
        return []

    new_conn = connect(NEW_DB, "New DB (5433)")
    cur = new_conn.cursor()
    cur.execute("SELECT username FROM users")
    existing = {r[0] for r in cur.fetchall()}

    imported = []
    for user_id, username, pin_hash, role in old_users:
        suffix = 1
        safe_username = username
        while safe_username in existing:
            safe_username = f"{username}_{suffix}"
            suffix += 1
        try:
            cur.execute(
                "INSERT INTO users (user_id, username, pin_hash, role) VALUES (%s, %s, %s, %s)",
                (user_id, safe_username, pin_hash, role or "user")
            )
            existing.add(safe_username)
            imported.append((user_id, username, safe_username))
            log(f"  Imported user {username} -> {safe_username} ({user_id})")
        except Exception as e:
            log(f"  SKIP {username}: {e}")
    new_conn.commit()
    new_conn.close()
    log(f"  Done: {len(imported)} users imported")
    return imported


# ════════════════════════════════════════════════════════════
# Step 2: Create project 'KekKekKek233'
# ════════════════════════════════════════════════════════════
def step2_create_project():
    log("─" * 50)
    log("Step 2: Create project 'KekKekKek233'")
    conn = connect(NEW_DB, "New DB")
    cur = conn.cursor()
    cur.execute("SELECT id FROM projects WHERE name = 'KekKekKek233'")
    row = cur.fetchone()
    if row:
        project_id = row[0]
        log(f"  Project already exists (id={project_id})")
    else:
        cur.execute(
            "INSERT INTO projects (name, description, created_by, status, industry) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            ("KekKekKek233", "Data recovered from database reset", ADMIN_UUID,
             "active", "general")
        )
        project_id = cur.fetchone()[0]
        conn.commit()
        log(f"  Created project id={project_id}")
    conn.close()
    return project_id


# ════════════════════════════════════════════════════════════
# Step 3: Create recovery session for user_files
# ════════════════════════════════════════════════════════════
def step3_create_recovery_session():
    log("─" * 50)
    log("Step 3: Create recovery chat session")
    conn = connect(NEW_DB, "New DB")
    cur = conn.cursor()
    cur.execute("SELECT thread_id FROM chat_sessions WHERE thread_id = %s",
                (RECOVERY_THREAD_ID,))
    if cur.fetchone():
        log(f"  Recovery session already exists")
    else:
        cur.execute(
            "INSERT INTO chat_sessions (user_id, thread_id, title, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (ADMIN_UUID, RECOVERY_THREAD_ID, RECOVERY_SESSION_TITLE,
             datetime.now(timezone.utc), datetime.now(timezone.utc))
        )
        conn.commit()
        log(f"  Created recovery session {RECOVERY_THREAD_ID}")
    conn.close()
    return RECOVERY_THREAD_ID


# ════════════════════════════════════════════════════════════
# Step 4: Restore chat sessions from data/dump/
# ════════════════════════════════════════════════════════════
def step4_restore_chats():
    log("─" * 50)
    log("Step 4: Restore chat sessions from data/dump/")
    dump_dir = os.path.join(BASE_DIR, "data", "dump")
    if not os.path.isdir(dump_dir):
        log("  No dump directory found")
        return 0, 0

    conn = connect(NEW_DB, "New DB")
    cur = conn.cursor()

    # Get existing thread_ids to avoid duplicates
    cur.execute("SELECT thread_id FROM chat_sessions")
    existing = {r[0] for r in cur.fetchall()}

    sessions_restored = 0
    messages_restored = 0

    for user_dir in os.listdir(dump_dir):
        user_path = os.path.join(dump_dir, user_dir)
        if not os.path.isdir(user_path):
            continue
        for date_dir in sorted(os.listdir(user_path)):
            date_path = os.path.join(user_path, date_dir)
            if not os.path.isdir(date_path):
                continue

            session_files = {}
            for fname in os.listdir(date_path):
                tid = fname.replace("_session.json", "") if "_session.json" in fname else fname.replace("_messages.json", "")
                if tid not in session_files:
                    session_files[tid] = {}
                if fname.endswith("_session.json"):
                    session_files[tid]["session"] = os.path.join(date_path, fname)
                elif fname.endswith("_messages.json"):
                    session_files[tid]["messages"] = os.path.join(date_path, fname)

            for tid, files in session_files.items():
                if tid in existing:
                    continue
                session_path = files.get("session")
                messages_path = files.get("messages")

                if not session_path:
                    continue

                try:
                    with open(session_path, "r", encoding="utf-8") as f:
                        sdata = json.load(f)
                except Exception as e:
                    log(f"  SKIP session {tid}: {e}")
                    continue

                try:
                    cur.execute(
                        "INSERT INTO chat_sessions "
                        "(user_id, thread_id, title, created_at, updated_at) "
                        "VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (thread_id) DO NOTHING",
                        (ADMIN_UUID, tid,
                         sdata.get("title", "Recovered Chat"),
                         sdata.get("created_at", datetime.now(timezone.utc).isoformat()),
                         sdata.get("updated_at", datetime.now(timezone.utc).isoformat()))
                    )
                    sessions_restored += 1
                    existing.add(tid)
                except Exception as e:
                    log(f"  SKIP session insert {tid}: {e}")
                    continue

                if not messages_path:
                    continue
                try:
                    with open(messages_path, "r", encoding="utf-8") as f:
                        msgs = json.load(f)
                except Exception as e:
                    log(f"  SKIP messages for {tid}: {e}")
                    continue

                for msg in msgs:
                    try:
                        cur.execute(
                            "INSERT INTO chat_messages "
                            "(thread_id, role, content, thinking, timestamp) "
                            "VALUES (%s, %s, %s, %s, %s)",
                            (tid, msg.get("role"), msg.get("content"),
                             msg.get("thinking"),
                             msg.get("timestamp", datetime.now(timezone.utc).isoformat()))
                        )
                        messages_restored += 1
                    except Exception as e:
                        log(f"  SKIP message in {tid}: {e}")

                if sessions_restored % 5 == 0:
                    conn.commit()

    conn.commit()
    conn.close()
    log(f"  Done: {sessions_restored} sessions, {messages_restored} messages restored")
    return sessions_restored, messages_restored


# ════════════════════════════════════════════════════════════
# Step 5: Restore project files from data/project_files/
# ════════════════════════════════════════════════════════════
def step5_restore_project_files(project_id):
    log("─" * 50)
    log("Step 5: Restore project files into project 'KekKekKek233'")
    proj_dir = os.path.join(BASE_DIR, "data", "project_files")
    if not os.path.isdir(proj_dir):
        log("  No project_files directory found")
        return 0, 0

    conn = connect(NEW_DB, "New DB")
    cur = conn.cursor()

    # Get existing folders
    cur.execute("SELECT id, name FROM project_folders WHERE project_id = %s",
                (project_id,))
    existing_folders = {r[1]: r[0] for r in cur.fetchall()}

    files_restored = 0
    folders_created = 0

    for item in sorted(os.listdir(proj_dir), key=lambda x: int(x) if x.isdigit() else 99999):
        item_path = os.path.join(proj_dir, item)
        if not os.path.isdir(item_path):
            continue

        folder_name = f"Project_{item}"
        if folder_name not in existing_folders:
            try:
                cur.execute(
                    "INSERT INTO project_folders (project_id, name, created_by) "
                    "VALUES (%s, %s, %s) RETURNING id",
                    (project_id, folder_name, ADMIN_UUID)
                )
                folder_id = cur.fetchone()[0]
                existing_folders[folder_name] = folder_id
                folders_created += 1
            except Exception as e:
                log(f"  SKIP folder creation for {item}: {e}")
                continue
        else:
            folder_id = existing_folders[folder_name]

        for fname in os.listdir(item_path):
            file_path = os.path.join(item_path, fname)
            if not os.path.isfile(file_path):
                continue

            try:
                with open(file_path, "rb") as f:
                    raw = f.read()
            except Exception:
                continue

            file_hash = hashlib.sha256(raw).hexdigest()
            size = len(raw)
            mime_type = mimetypes.guess_type(fname)[0] or "application/octet-stream"
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                content = None

            stored_path = os.path.relpath(file_path, BASE_DIR)

            try:
                cur.execute(
                    "INSERT INTO project_files "
                    "(project_id, folder_id, filename, original_name, file_size, "
                    " mime_type, stored_path, uploaded_by, file_hash, content, category) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (project_id, folder_id, fname, fname, size,
                     mime_type, stored_path, ADMIN_UUID, file_hash, content, "Recovered")
                )
                files_restored += 1
            except Exception as e:
                pass

            if files_restored % 50 == 0:
                conn.commit()

    conn.commit()
    conn.close()
    log(f"  Done: {folders_created} folders, {files_restored} files restored")
    return folders_created, files_restored


# ════════════════════════════════════════════════════════════
# Step 6: Restore user files (batch/comparison results)
# ════════════════════════════════════════════════════════════
def step6_restore_user_files():
    log("─" * 50)
    log("Step 6: Restore user files to recovery session")
    uf_dir = os.path.join(BASE_DIR, "data", "user_files")
    if not os.path.isdir(uf_dir):
        log("  No user_files directory found")
        return 0

    conn = connect(NEW_DB, "New DB")
    cur = conn.cursor()
    cur.execute("SELECT file_hash FROM user_files")
    existing_hashes = {r[0] for r in cur.fetchall() if r[0]}

    files_restored = 0

    for user_uuid in sorted(os.listdir(uf_dir)):
        user_path = os.path.join(uf_dir, user_uuid)
        if not os.path.isdir(user_path):
            continue
        for fname in os.listdir(user_path):
            file_path = os.path.join(user_path, fname)
            if not os.path.isfile(file_path):
                continue
            try:
                with open(file_path, "rb") as f:
                    raw = f.read()
            except Exception:
                continue

            file_hash = hashlib.sha256(raw).hexdigest()
            if file_hash in existing_hashes:
                continue

            size = len(raw)
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                content = None

            original_stored_path = os.path.relpath(file_path, BASE_DIR)
            meta = json.dumps({"original_user_uuid": user_uuid})

            try:
                cur.execute(
                    "INSERT INTO user_files "
                    "(user_id, thread_id, filename, content, size_bytes, "
                    " original_stored_path, file_hash, original_name, meta_data) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (ADMIN_UUID, RECOVERY_THREAD_ID, fname, content, size,
                     original_stored_path, file_hash, fname, meta)
                )
                files_restored += 1
                existing_hashes.add(file_hash)
            except Exception as e:
                pass

    conn.commit()
    conn.close()
    log(f"  Done: {files_restored} user files restored")
    return files_restored


# ════════════════════════════════════════════════════════════
# Step 7: Restore knowledge base from data/ingest/
# ════════════════════════════════════════════════════════════
def step7_restore_knowledge_base():
    log("─" * 50)
    log("Step 7: Restore knowledge base files from data/ingest/")
    ingest_dir = os.path.join(BASE_DIR, "data", "ingest")
    if not os.path.isdir(ingest_dir):
        log("  No ingest directory found")
        return 0

    conn = connect(NEW_DB, "New DB")
    cur = conn.cursor()
    cur.execute("SELECT file_hash FROM knowledge_lab_files")
    existing_hashes = {r[0] for r in cur.fetchall() if r[0]}

    files_restored = 0

    for root, dirs, files in os.walk(ingest_dir):
        for fname in files:
            file_path = os.path.join(root, fname)
            if not os.path.isfile(file_path):
                continue
            # Skip chunked images (from document parsing)
            if fname.startswith("_page_") and fname.endswith(".png"):
                continue

            try:
                with open(file_path, "rb") as f:
                    raw = f.read()
            except Exception:
                continue

            file_hash = hashlib.sha256(raw).hexdigest()
            if file_hash in existing_hashes:
                continue

            size = len(raw)
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                content = None

            stored_path = os.path.relpath(file_path, BASE_DIR)
            try:
                cur.execute(
                    "INSERT INTO knowledge_lab_files "
                    "(user_id, filename, original_name, file_size, content, "
                    " file_hash, stored_path, category) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (ADMIN_UUID, fname, fname, size, content,
                     file_hash, stored_path, "Recovered")
                )
                files_restored += 1
                existing_hashes.add(file_hash)
            except Exception as e:
                pass

            if files_restored % 20 == 0:
                conn.commit()

    conn.commit()
    conn.close()
    log(f"  Done: {files_restored} knowledge base files restored")
    return files_restored


# ════════════════════════════════════════════════════════════
# Step 8: Create backup script
# ════════════════════════════════════════════════════════════
def step8_create_backup_script():
    log("─" * 50)
    log("Step 8: Create backup script")
    scripts_dir = os.path.join(BASE_DIR, "scripts")
    backup_path = os.path.join(scripts_dir, "daily_backup.sh")

    content = """#!/bin/bash
# Daily PostgreSQL backup - keep last 7 days
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$(dirname "$0")/../backups"
mkdir -p "$BACKUP_DIR"
docker exec localai-postgres pg_dump -U localai localai > "$BACKUP_DIR/$TIMESTAMP.sql"
find "$BACKUP_DIR" -name "*.sql" -mtime +7 -delete
echo "Backup saved: $BACKUP_DIR/$TIMESTAMP.sql"
"""
    try:
        with open(backup_path, "w") as f:
            f.write(content)
        log(f"  Created {backup_path}")
    except Exception as e:
        log(f"  SKIP backup script: {e}")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
def main():
    log("=" * 50)
    log("DATA RECOVERY START")
    log(f"Target: New DB (localhost:5433/localai)")
    log(f"Admin UUID: {ADMIN_UUID}")
    log("=" * 50)

    step1_import_old_users()
    project_id = step2_create_project()
    step3_create_recovery_session()
    step4_restore_chats()
    step5_restore_project_files(project_id)
    step6_restore_user_files()
    step7_restore_knowledge_base()
    step8_create_backup_script()

    log("=" * 50)
    log("RECOVERY COMPLETE")
    log("=" * 50)


if __name__ == "__main__":
    main()
