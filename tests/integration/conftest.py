"""Integration-test fixtures: real DB, mock LLM HTTP, fakeredis.

Requires PostgreSQL with a 'test_chatbot_test' database.
Auto-creates the DB if missing.
"""
import os
import re
import json
from pathlib import Path

import httpx
import pytest

MOCK_LLM_DIR = Path(__file__).parent.parent / "mock_data" / "llm"

# ── Auto-create test_chatbot_test if missing ──
def _ensure_test_db():
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    conn = psycopg2.connect(
        dbname="postgres",
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
        host=os.environ.get("PG_HOST", "localhost"),
        port=os.environ.get("PG_PORT", "5432"),
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    db_name = os.environ.get("PG_DATABASE", "test_chatbot_test")
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{db_name}"')
    conn.close()

_ensure_test_db()


# ── Mock LLM HTTP: intercept all provider API calls ──
def _load_mock(name: str) -> dict:
    path = MOCK_LLM_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def mock_llm_http(httpx_mock):
    """Intercept all LLM /v1/chat/completions calls with canned JSON."""
    providers = {
        "api.deepseek.com": "deepseek_chat.json",
        "open.bigmodel.cn": "zhipu_chat.json",
        "dashscope.aliyuncs.com": "qwen_chat.json",
        "api.siliconflow.cn": "siliconflow_chat.json",
        "integrate.api.nvidia.com": "nvidia_chat.json",
    }
    for base_url, file_name in providers.items():
        data = _load_mock(file_name)
        httpx_mock.add_response(
            url=re.compile(f".*{re.escape(base_url)}.*/chat/completions"),
            json=data,
            is_optional=True,
            is_reusable=True,
        )
    return httpx_mock


# ── Database ──
@pytest.fixture(scope="session")
def db(app):
    """Initialize test schema once per session, return the pool."""
    from app.database import init_postgres_tables, get_db_pool
    init_postgres_tables()
    yield get_db_pool()


@pytest.fixture(autouse=True)
def clean_tables(db):
    """Truncate all public tables between tests, then re-seed data."""
    conn = db.getconn()
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute("SET session_replication_role = 'replica'")
        cur.execute(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' AND tableowner = CURRENT_USER"
        )
        for row in cur.fetchall():
            cur.execute(f'TRUNCATE TABLE "{row[0]}" CASCADE')
        cur.execute("SET session_replication_role = 'origin'")
    finally:
        cur.close()
        db.putconn(conn)
    import io, sys
    _old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from app.database import init_postgres_tables
        init_postgres_tables()
    finally:
        sys.stdout = _old


# ── HTTP client ──
@pytest.fixture
def client(app, clean_tables):
    """Flask test client with clean tables per test."""
    with app.test_client() as c:
        yield c


@pytest.fixture
def auth_client(client):
    """Client with a registered + logged-in user session — session manipulation to avoid flakiness."""
    import uuid
    user_id = str(uuid.uuid4())
    # Also create the user record in DB for routes that do lookups
    from app.database import get_db_connection
    from werkzeug.security import generate_password_hash
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (user_id, username, pin_hash, pin_length, role, is_active) "
                "VALUES (%s, %s, %s, %s, 'user', TRUE) ON CONFLICT (username) DO NOTHING",
                (user_id, "testuser", generate_password_hash("123456"), 6)
            )
        conn.commit()
    with client.session_transaction() as sess:
        sess['user_id'] = user_id
        sess['username'] = 'testuser'
        sess['role'] = 'user'
        sess['consent_value'] = 1
        sess['thread_id'] = 'test-thread-001'
        sess['is_auditor'] = False
    return client


@pytest.fixture
def admin_client(client):
    """Client with admin session (CEO) — look up real user_id from DB to satisfy FK constraints."""
    from app.database import get_db_connection
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM users WHERE username = 'CEO'")
            row = cur.fetchone()
            user_id = row[0] if row else '00000000-0000-0000-0000-000000000000'
    with client.session_transaction() as sess:
        sess['user_id'] = user_id
        sess['username'] = 'CEO'
        sess['role'] = 'admin'
        sess['is_admin'] = True
        sess['consent_value'] = 1
    return client


# ── Test fixtures ──
@pytest.fixture
def test_bid_file(tmp_path):
    """Create a sample Chinese bid .txt file for upload tests."""
    p = tmp_path / "bid.txt"
    p.write_text(
        "投标函\n\n"
        "致：中联国际招标有限公司\n\n"
        "我方已仔细阅读了贵方发布的招标文件（招标编号：ZLTB-2026-001），"
        "并完全理解所有条款和要求。我方承诺按照招标文件的规定，"
        "以人民币伍佰万元整（¥5,000,000）的总报价承担本项目的全部工作。\n\n"
        "投标人：测试建筑有限公司\n"
        "日期：2026年7月4日\n",
        encoding="utf-8",
    )
    return str(p)
