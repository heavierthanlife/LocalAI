"""Seed the throwaway e2e stack with deterministic users/project for T1 UI audit.
Run inside localai-e2e-app:  docker exec localai-e2e-app python /tmp/e2e_seed.py
Idempotent. Creates: admin (CEO role=admin), normal user, one personal thread.
"""
import sys, io
sys.path.insert(0, "/app")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import uuid
from werkzeug.security import generate_password_hash
from app.database import get_db_connection

ADMIN_UNAME = "CEO"
USER_UNAME = "e2euser"
PIN = "123456"

def upsert_user(cur, uname, role):
    uid = str(uuid.uuid4())
    cur.execute(
        """INSERT INTO users (user_id, username, pin_hash, pin_length, role, is_active)
           VALUES (%s,%s,%s,%s,%s,TRUE)
           ON CONFLICT (username) DO UPDATE SET role=EXCLUDED.role, is_active=TRUE
           RETURNING user_id""", (uid, uname, generate_password_hash(PIN), 6, role))
    return cur.fetchone()[0]

with get_db_connection() as conn:
    with conn.cursor() as cur:
        aid = upsert_user(cur, ADMIN_UNAME, "admin")
        uid = upsert_user(cur, USER_UNAME, "user")
        conn.commit()

print("admin CEO:", aid)
print("user e2euser:", uid)
print("E2E SEED OK")
