"""Database pool, connection management, and table initialization."""
from __future__ import annotations

import os
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)
from typing import TYPE_CHECKING

from psycopg2 import pool

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PgConn, cursor as PgCursor

from .config import logger

def get_db_connection_args():
    # 1. Explicit PG_* vars (PyCharm run config / system env)
    if os.getenv("PG_USER") and os.getenv("PG_PASSWORD"):
        return {
            'dbname': os.getenv('PG_DB', 'postgres'),
            'user': os.getenv('PG_USER'),
            'password': os.getenv('PG_PASSWORD'),
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': int(os.getenv('PG_PORT', 5432)),
            'client_encoding': 'utf8'
        }
    # 2. DATABASE_URL (Docker) or POSTGRES_URI (legacy)
    uri = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URI")
    if uri:
        from urllib.parse import urlparse, unquote
        result = urlparse(uri)
        dbname = result.path[1:] if result.path else ''
        user = result.username
        password = result.password
        if password:
            password = unquote(password)
        host = result.hostname
        port = result.port or 5432
        return {'dbname': dbname, 'user': user, 'password': password, 'host': host, 'port': port, 'client_encoding': 'utf8'}
    # 3. Local dev fallback (PyCharm IDE, no env vars set)
    return {
        'dbname': os.getenv('PG_DB', 'postgres'),
        'user': os.getenv('PG_USER', 'postgres'),
        'password': os.environ['PG_PASSWORD'],
        'host': os.getenv('PG_HOST', 'localhost'),
        'port': int(os.getenv('PG_PORT', 5432)),
        'client_encoding': 'utf8'
    }

_db_pool = None


def get_db_pool():
    global _db_pool
    if _db_pool is None:
        _db_pool = pool.SimpleConnectionPool(
            int(os.getenv("DB_POOL_MIN", "1")),
            int(os.getenv("DB_POOL_MAX", "20")),
            **get_db_connection_args()
        )
    return _db_pool


@contextmanager
def get_db_connection():
    p = get_db_pool()
    conn = p.getconn()
    try:
        conn.cursor().execute("SELECT 1")
        yield conn
    except Exception:
        logger.error("Database health check failed, reconnecting", exc_info=True)
        p.putconn(conn, close=True)
        conn = p.getconn()
        # Re-raise the ORIGINAL exception. Previously this block yielded a
        # second time, which contextlib turns into a misleading
        # RuntimeError("generator didn't stop after throw()") that masked
        # every real DB error surfaced inside `with` bodies.
        raise
    finally:
        p.putconn(conn)


@contextmanager
def db_transaction(conn: "PgConn"):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_postgres_tables():
    """Initialize all PostgreSQL tables. Called once at startup.

    Serialized with a PostgreSQL advisory lock so that concurrent workers
    (gunicorn multi-worker boot) do not deadlock on CREATE INDEX.
    """
    with get_db_connection() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_lock(732014)")
                import_init_tables(cur)
                conn.commit()
                cur.execute("SELECT pg_advisory_unlock(732014)")
        except Exception:
            conn.rollback()
            raise
    logger.info("PostgreSQL tables initialized.")


def import_init_tables(cur: "PgCursor"):
    """Import and execute the full table initialization. Kept in a separate function
    to avoid bloating the module top-level."""
    _run_table_creation(cur)


def _run_table_creation(cur: "PgCursor"):
    """All CREATE TABLE and ALTER TABLE statements."""
    cur.execute("""
                CREATE TABLE IF NOT EXISTS users
                (
                    user_id    TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """)
    cur.execute("""
        DO $$ 
        BEGIN
            BEGIN ALTER TABLE users ADD COLUMN username TEXT UNIQUE; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE users ADD COLUMN pin_hash TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE users ADD COLUMN pin_length INTEGER; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE users ADD COLUMN email TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE users ADD COLUMN deletion_requested BOOLEAN DEFAULT FALSE; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE users ADD COLUMN deletion_code TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE users ADD COLUMN is_auditor BOOLEAN DEFAULT FALSE; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    # Migrate legacy skill_auditor role → is_auditor flag
    try:
        cur.execute("UPDATE users SET is_auditor = TRUE, role = 'user' WHERE role = 'skill_auditor'")
    except Exception:
        pass  # column may not exist yet on first run
    cur.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")

    # Seed admin accounts: CEO and COO (share same PIN as sys-admin)
    import uuid as _uuid
    admin_pin = os.getenv('ADMIN_PIN', '888888')
    import hashlib as _hl
    _salt = os.urandom(16).hex()
    admin_pin_hash = _salt + ":" + _hl.pbkdf2_hmac('sha256', admin_pin.encode(), _salt.encode(), 100000).hex()
    for uname in ('CEO', 'COO'):
        cur.execute("""
            INSERT INTO users (user_id, username, pin_hash, pin_length, role, is_active)
            VALUES (%s,%s,%s,%s,'admin',TRUE)
            ON CONFLICT (username) DO NOTHING
        """, (str(_uuid.uuid4()), uname, admin_pin_hash, len(admin_pin)))
    print("[DB] Seeded admin accounts: CEO, COO")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id SERIAL PRIMARY KEY, user_id TEXT REFERENCES users(user_id),
            thread_id TEXT UNIQUE NOT NULL, title TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id SERIAL PRIMARY KEY, thread_id TEXT REFERENCES chat_sessions(thread_id),
            role TEXT, content TEXT, thinking TEXT, timestamp TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_files (
            id SERIAL PRIMARY KEY, user_id TEXT REFERENCES users(user_id),
            thread_id TEXT REFERENCES chat_sessions(thread_id), filename TEXT, content TEXT,
            size_bytes BIGINT, created_at TIMESTAMPTZ DEFAULT NOW(), expires_at TIMESTAMPTZ,
            original_stored_path TEXT, file_hash TEXT, original_expires_at TIMESTAMPTZ,
            meta_data JSONB DEFAULT '{}', original_name TEXT, UNIQUE (thread_id, filename)
        )
    """)
    # Migrate existing integer columns to bigint to support files > 2GB
    cur.execute("""
        DO $$ BEGIN
            ALTER TABLE user_files ALTER COLUMN size_bytes TYPE BIGINT;
        EXCEPTION WHEN others THEN NULL;
        END $$;
    """)
    cur.execute("""
        DO $$
        BEGIN
            BEGIN ALTER TABLE user_files ADD COLUMN IF NOT EXISTS file_hash TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE user_files ADD COLUMN IF NOT EXISTS meta_data JSONB DEFAULT '{}'; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE user_files ADD COLUMN IF NOT EXISTS original_expires_at TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE user_files ADD COLUMN IF NOT EXISTS original_name TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE user_files ALTER COLUMN expires_at DROP DEFAULT; EXCEPTION WHEN others THEN NULL; END;
        END $$;
    """)
    cur.execute("UPDATE user_files SET original_name = filename WHERE original_name IS NULL")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS archived_sessions (
            thread_id TEXT PRIMARY KEY, user_id TEXT, archive_path TEXT, archived_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS image_description_cache (
            file_hash TEXT PRIMARY KEY, description TEXT, created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS file_usage (
            id SERIAL PRIMARY KEY, user_id TEXT, thread_id TEXT, filename TEXT,
            usage_type TEXT, question TEXT, timestamp TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS consent (
            thread_id TEXT PRIMARY KEY, consent_given INTEGER NOT NULL, timestamp TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY, thread_id TEXT, user_message TEXT,
            assistant_response TEXT, rating TEXT, comment TEXT, file_name TEXT, timestamp TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS message_responses (
            message_id TEXT PRIMARY KEY, thread_id TEXT, user_message TEXT,
            assistant_response TEXT, thinking TEXT, created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY, name TEXT NOT NULL, description TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW(),
            created_by TEXT REFERENCES users(user_id), status TEXT DEFAULT 'active',
            archived_at TIMESTAMPTZ, deletion_scheduled_at TIMESTAMPTZ
        )
    """)
    cur.execute("""
        DO $$ 
        BEGIN
            BEGIN ALTER TABLE projects ADD COLUMN status TEXT DEFAULT 'active'; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE projects ADD COLUMN archived_at TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE projects ADD COLUMN deletion_scheduled_at TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE projects ADD COLUMN archive_filename TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE projects ADD COLUMN industry TEXT DEFAULT 'general'; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE projects ADD COLUMN bidding_category TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE projects ADD COLUMN bid_method TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    # Migration: add project_id and is_grilling to chat_sessions (after projects table exists)
    cur.execute("""
        DO $$ BEGIN
            BEGIN ALTER TABLE chat_sessions ADD COLUMN project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL;
            EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE chat_sessions ADD COLUMN is_grilling BOOLEAN DEFAULT FALSE;
            EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_project ON chat_sessions(project_id)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_members (
            id SERIAL PRIMARY KEY, project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(user_id), role TEXT NOT NULL,
            added_at TIMESTAMPTZ DEFAULT NOW(), added_by TEXT REFERENCES users(user_id),
            status TEXT DEFAULT 'active', last_read_at TIMESTAMPTZ,
            UNIQUE (project_id, user_id)
        )
    """)
    cur.execute("""
        DO $$ BEGIN
            ALTER TABLE project_members ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active';
        END $$;
    """)
    cur.execute("""
        DO $$ BEGIN
            ALTER TABLE project_members ADD COLUMN IF NOT EXISTS last_read_at TIMESTAMPTZ;
        END $$;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_folders (
            id SERIAL PRIMARY KEY, project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            parent_folder_id INTEGER REFERENCES project_folders(id) ON DELETE CASCADE,
            name TEXT NOT NULL, path TEXT, created_at TIMESTAMPTZ DEFAULT NOW(),
            created_by TEXT REFERENCES users(user_id), UNIQUE (project_id, parent_folder_id, name)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_files (
            id SERIAL PRIMARY KEY, project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            folder_id INTEGER REFERENCES project_folders(id) ON DELETE CASCADE,
            filename TEXT NOT NULL, original_name TEXT NOT NULL, file_size INTEGER,
            mime_type TEXT, stored_path TEXT NOT NULL, version INTEGER DEFAULT 1,
            uploaded_at TIMESTAMPTZ DEFAULT NOW(), uploaded_by TEXT REFERENCES users(user_id),
            comment TEXT, file_hash TEXT, UNIQUE (project_id, folder_id, filename)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_file_versions (
            id SERIAL PRIMARY KEY, file_id INTEGER REFERENCES project_files(id) ON DELETE CASCADE,
            version INTEGER NOT NULL, stored_path TEXT NOT NULL, file_size INTEGER,
            uploaded_at TIMESTAMPTZ DEFAULT NOW(), uploaded_by TEXT REFERENCES users(user_id), comment TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_file_comments (
            id SERIAL PRIMARY KEY, file_id INTEGER REFERENCES project_files(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(user_id), comment TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        DO $$ BEGIN
            BEGIN ALTER TABLE project_files ADD COLUMN content TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE project_files ADD COLUMN skill_summary TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE project_files ADD COLUMN skill_generated_at TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE project_files ADD COLUMN skill_summary_hash TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE project_files ADD COLUMN category TEXT DEFAULT '通用'; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE project_files ADD COLUMN status TEXT DEFAULT 'draft'; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_ai_memory (
            id SERIAL PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(user_id),
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            content_md TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_memory_project_user ON project_ai_memory(project_id, user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_memory_created ON project_ai_memory(project_id, user_id, created_at)")
    cur.execute("""
        DO $$ BEGIN
            BEGIN ALTER TABLE project_ai_memory ADD COLUMN question_hash TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_memory_hash ON project_ai_memory(project_id, question_hash)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS member_workflows (
            id SERIAL PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(user_id),
            workflow_name TEXT DEFAULT '默认工作流',
            steps JSONB DEFAULT '[]',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (project_id, user_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS workflow_kpi (
            id SERIAL PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(user_id),
            generations INTEGER DEFAULT 0,
            revision_rounds INTEGER DEFAULT 0,
            output_chars INTEGER DEFAULT 0,
            completed_count INTEGER DEFAULT 0,
            overlap_warnings INTEGER DEFAULT 0,
            last_active TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_wkpi_project_user ON workflow_kpi(project_id, user_id)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_file_usage (
            id SERIAL PRIMARY KEY, file_id INTEGER REFERENCES project_files(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(user_id), action TEXT NOT NULL, details JSONB, timestamp TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_folder_comments (
            id SERIAL PRIMARY KEY, folder_id INTEGER REFERENCES project_folders(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(user_id), comment TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS task_deposit_items (
            id SERIAL PRIMARY KEY, original_user_id TEXT REFERENCES users(user_id),
            original_username TEXT, project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            project_name TEXT, item_type TEXT NOT NULL, item_data JSONB NOT NULL,
            stored_path TEXT, transferred_to_user_id TEXT REFERENCES users(user_id),
            transferred_at TIMESTAMPTZ, created_at TIMESTAMPTZ DEFAULT NOW(), deleted_at TIMESTAMPTZ
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS task_deposit_permissions (
            id SERIAL PRIMARY KEY, project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            manager_id TEXT REFERENCES users(user_id), can_view_deposit BOOLEAN DEFAULT FALSE,
            granted_by TEXT REFERENCES users(user_id), granted_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            thread_id TEXT NOT NULL, checkpoint_id TEXT NOT NULL, parent_checkpoint_id TEXT,
            type TEXT, checkpoint JSONB, metadata JSONB, PRIMARY KEY (thread_id, checkpoint_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS checkpoint_writes (
            thread_id TEXT NOT NULL, checkpoint_id TEXT NOT NULL, task_id TEXT NOT NULL,
            idx INTEGER NOT NULL, value JSONB, PRIMARY KEY (thread_id, checkpoint_id, task_id, idx)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recycle_bin (
            id SERIAL PRIMARY KEY, original_table TEXT NOT NULL, original_id INTEGER NOT NULL,
            user_id TEXT REFERENCES users(user_id), file_name TEXT, file_content TEXT,
            file_size INTEGER, original_stored_path TEXT, file_hash TEXT, thread_id TEXT,
            deleted_at TIMESTAMPTZ DEFAULT NOW(), expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '3 days'
        )
    """)
    cur.execute("""DO $$ 
        BEGIN
            BEGIN ALTER TABLE recycle_bin ADD COLUMN original_thread_id TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE recycle_bin ADD COLUMN deletion_reason TEXT DEFAULT 'manual'; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("""DO $$ 
        BEGIN
            BEGIN ALTER TABLE recycle_bin ADD COLUMN uploaded_by TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE recycle_bin ADD COLUMN deleted_by TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("""DO $$ 
        BEGIN
            BEGIN ALTER TABLE recycle_bin DROP COLUMN IF EXISTS original_data; EXCEPTION WHEN undefined_column THEN NULL; END;
        END $$;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_recycle_bin (
            id SERIAL PRIMARY KEY, original_table TEXT NOT NULL, original_id INTEGER NOT NULL,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE, folder_id INTEGER,
            file_name TEXT, original_name TEXT, file_size INTEGER, stored_path TEXT,
            file_hash TEXT, version INTEGER, uploaded_by TEXT REFERENCES users(user_id),
            deleted_at TIMESTAMPTZ DEFAULT NOW(), expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '3 days'
        )
    """)
    cur.execute("""
        DO $$ BEGIN
            BEGIN ALTER TABLE project_recycle_bin ADD COLUMN deleted_by TEXT REFERENCES users(user_id); EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_folders_recycle_bin (
            id SERIAL PRIMARY KEY, original_id INTEGER NOT NULL,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            name TEXT NOT NULL, parent_folder_id INTEGER, original_parent_id INTEGER,
            created_at TIMESTAMPTZ, created_by TEXT, deleted_at TIMESTAMPTZ DEFAULT NOW(),
            expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '3 days'
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kb_recycle_bin (
            id SERIAL PRIMARY KEY,
            original_table TEXT NOT NULL,
            original_id INTEGER NOT NULL,
            user_id TEXT REFERENCES users(user_id),
            filename TEXT, original_name TEXT,
            file_size INTEGER, content TEXT,
            file_hash TEXT, stored_path TEXT,
            category TEXT, uploaded_by TEXT REFERENCES users(user_id),
            deleted_by TEXT REFERENCES users(user_id),
            deleted_at TIMESTAMPTZ DEFAULT NOW(),
            expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '3 days'
        )
    """)
    cur.execute("""
        DO $$ BEGIN
            BEGIN ALTER TABLE kb_recycle_bin ADD COLUMN skill_summary TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_lab_files (
            id SERIAL PRIMARY KEY, user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
            filename TEXT NOT NULL, original_name TEXT NOT NULL, file_size INTEGER,
            content TEXT, file_hash TEXT UNIQUE, stored_path TEXT,
            uploaded_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lab_user ON knowledge_lab_files(user_id)")
    # Skill columns for knowledge_lab_files (each in its own exception block to avoid cascading rollback)
    cur.execute("""
        DO $$ BEGIN
            BEGIN ALTER TABLE knowledge_lab_files ADD COLUMN skill_summary TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("""
        DO $$ BEGIN
            BEGIN ALTER TABLE knowledge_lab_files ADD COLUMN skill_generated_at TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("""
        DO $$ BEGIN
            BEGIN ALTER TABLE knowledge_lab_files ADD COLUMN skill_summary_hash TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("""
        DO $$ BEGIN
            BEGIN ALTER TABLE knowledge_lab_files ADD COLUMN category TEXT DEFAULT '通用'; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS company_knowledge_base (
            id SERIAL PRIMARY KEY, filename TEXT NOT NULL, original_name TEXT NOT NULL,
            file_size INTEGER, content TEXT, file_hash TEXT UNIQUE, stored_path TEXT,
            category TEXT, uploaded_by TEXT REFERENCES users(user_id),
            uploaded_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_company_kb_category ON company_knowledge_base(category)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_company_kb_filename ON company_knowledge_base(filename)")
    cur.execute("""
        DO $$ BEGIN
            BEGIN ALTER TABLE company_knowledge_base ADD COLUMN skill_summary TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("""
        DO $$ BEGIN
            BEGIN ALTER TABLE company_knowledge_base ADD COLUMN skill_generated_at TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("""
        DO $$ BEGIN
            BEGIN ALTER TABLE company_knowledge_base ADD COLUMN skill_summary_hash TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_company_kb_content
            ON company_knowledge_base USING gin (to_tsvector('simple', content))
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS file_text_cache (
            id SERIAL PRIMARY KEY, file_path TEXT NOT NULL, last_modified TIMESTAMPTZ,
            file_hash TEXT NOT NULL, extracted_text TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_file_text_cache_path ON file_text_cache(file_path)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_file_text_cache_hash ON file_text_cache(file_hash)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS wiki_origin_links (
            id SERIAL PRIMARY KEY,
            wiki_page_path TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_file_id INTEGER NOT NULL,
            source_name TEXT NOT NULL,
            source_status TEXT DEFAULT 'active',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(wiki_page_path, source_type, source_file_id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_wiki_origin_source ON wiki_origin_links(source_type, source_file_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_wiki_origin_page ON wiki_origin_links(wiki_page_path)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_audit_log (
            id SERIAL PRIMARY KEY, admin_user_id TEXT REFERENCES users(user_id),
            admin_username TEXT, action TEXT NOT NULL, table_name TEXT NOT NULL,
            row_id TEXT, column_name TEXT, old_value TEXT, new_value TEXT,
            ip_address TEXT, success BOOLEAN DEFAULT TRUE, error_message TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_admin_audit_log_admin ON admin_audit_log(admin_user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_admin_audit_log_created ON admin_audit_log(created_at)")

    # ── Unified Bid Audit ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_config (
            id SERIAL PRIMARY KEY,
            function_name TEXT UNIQUE NOT NULL,
            enabled_by_default BOOLEAN DEFAULT true,
            fail_threshold REAL DEFAULT 50,
            weight REAL DEFAULT 14.28,
            severity_thresholds JSONB DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_runs (
            id SERIAL PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(user_id),
            status TEXT NOT NULL DEFAULT 'running',
            config_snapshot JSONB,
            overall_score REAL,
            overall_status TEXT,
            bidder_count INTEGER,
            file_count INTEGER,
            docx_path TEXT,
            xlsx_path TEXT,
            error_message TEXT,
            started_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_file_results (
            id SERIAL PRIMARY KEY,
            run_id INTEGER REFERENCES audit_runs(id) ON DELETE CASCADE,
            file_id INTEGER REFERENCES project_files(id) ON DELETE SET NULL,
            folder_id INTEGER REFERENCES project_folders(id) ON DELETE SET NULL,
            bidder_label TEXT,
            filename TEXT,
            function_name TEXT,
            score REAL,
            status TEXT DEFAULT 'pending',
            findings JSONB,
            error_message TEXT,
            retry_count INTEGER DEFAULT 0,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ
        )
    """)

    # ── Bidding Categories & Schedule Templates ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bidding_categories (
            id SERIAL PRIMARY KEY,
            code TEXT UNIQUE NOT NULL,
            name_zh TEXT NOT NULL,
            regime TEXT NOT NULL DEFAULT 'bidding_law',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bidding_schedule_templates (
            id SERIAL PRIMARY KEY,
            category_code TEXT NOT NULL,
            method_code TEXT NOT NULL,
            milestone_code TEXT NOT NULL,
            milestone_name TEXT NOT NULL,
            days_from_start INTEGER,
            days_from_prev_milestone INTEGER,
            prev_milestone_code TEXT,
            duration_days INTEGER,
            date_type TEXT NOT NULL DEFAULT 'calendar',
            mandatory BOOLEAN DEFAULT FALSE,
            law_ref TEXT,
            description TEXT,
            sort_order INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (category_code, method_code, milestone_code)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bidding_templates_cat_method ON bidding_schedule_templates(category_code, method_code, sort_order)")

    # ── Project Timeline tables ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_timelines (
            id SERIAL PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            name TEXT NOT NULL DEFAULT '主招标流程',
            category_code TEXT NOT NULL,
            method_code TEXT NOT NULL,
            planned_start_date DATE NOT NULL,
            planned_end_date DATE,
            actual_start_date DATE,
            actual_end_date DATE,
            status TEXT DEFAULT 'active',
            created_by TEXT REFERENCES users(user_id),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_timelines_project ON project_timelines(project_id, status)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_timeline_milestones (
            id SERIAL PRIMARY KEY,
            timeline_id INTEGER REFERENCES project_timelines(id) ON DELETE CASCADE,
            milestone_code TEXT NOT NULL,
            milestone_name TEXT NOT NULL,
            planned_date DATE,
            actual_date DATE,
            diff_days INTEGER,
            diff_reason TEXT,
            reason_category TEXT,
            status TEXT DEFAULT 'pending',
            sort_order INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(timeline_id, milestone_code)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_timeline_ms_timeline ON project_timeline_milestones(timeline_id, sort_order)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS timeline_diff_log (
            id SERIAL PRIMARY KEY,
            milestone_id INTEGER REFERENCES project_timeline_milestones(id) ON DELETE CASCADE,
            milestone_code TEXT NOT NULL,
            planned_date DATE,
            actual_date DATE,
            diff_days INTEGER,
            diff_type TEXT,
            reason_category TEXT,
            reason_detail TEXT,
            created_by TEXT REFERENCES users(user_id),
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_diff_log_milestone ON timeline_diff_log(milestone_id, created_at DESC)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_workflow_steps (
            id SERIAL PRIMARY KEY,
            milestone_id INTEGER REFERENCES project_timeline_milestones(id) ON DELETE CASCADE,
            step_name TEXT NOT NULL,
            step_order INTEGER DEFAULT 0,
            assigned_to TEXT,
            completed BOOLEAN DEFAULT FALSE,
            completed_at TIMESTAMPTZ,
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_workflow_milestone ON project_workflow_steps(milestone_id, step_order)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS timeline_suggestions (
            id SERIAL PRIMARY KEY,
            timeline_id INTEGER REFERENCES project_timelines(id) ON DELETE CASCADE,
            milestone_code TEXT NOT NULL,
            type TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'medium',
            content TEXT NOT NULL,
            suggestion TEXT,
            is_read BOOLEAN DEFAULT FALSE,
            is_actioned BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_timeline_sugs_timeline ON timeline_suggestions(timeline_id, created_at DESC)")

    # Seed default audit config if empty
    cur.execute("SELECT COUNT(*) FROM audit_config")
    if cur.fetchone()[0] == 0:
        defaults = [
            ('rule_extraction', True, 40, 15.0, '{"min_extracted_rules": 5}'),
            ('compliance_check', True, 50, 25.0, '{"critical": 1, "violation": 3}'),
            ('typo_detection', True, 60, 10.0, '{"penalty_per_10k": 5}'),
            ('quote_anomaly', True, 50, 20.0, '{"same_rate": 0.05, "drop": 0.15}'),
            ('relationship_extraction', True, 60, 10.0, '{"risk_signal_weight": 15}'),
            ('ai_doc_review', True, 50, 15.0, '{"min_chars": 500}'),
            ('style_analysis', False, 70, 5.0, '{}'),
        ]
        for func_name, enabled, threshold, weight, sev in defaults:
            cur.execute(
                """INSERT INTO audit_config (function_name, enabled_by_default, fail_threshold, weight, severity_thresholds)
                   VALUES (%s, %s, %s, %s, %s::jsonb)""",
                (func_name, enabled, threshold, weight, sev)
            )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS file_analysis (
            id SERIAL PRIMARY KEY, file_hash TEXT NOT NULL, file_type TEXT NOT NULL,
            original_filename TEXT, user_id TEXT REFERENCES users(user_id),
            thread_id TEXT, project_id INTEGER, file_size INTEGER,
            extracted_text TEXT, usage_count INTEGER DEFAULT 0,
            last_used_at TIMESTAMPTZ DEFAULT NOW(), created_at TIMESTAMPTZ DEFAULT NOW(),
            deleted_at TIMESTAMPTZ
        )
    """)

    # ── Project Todo system ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_todos (
            id SERIAL PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(user_id),
            message_id BIGINT,
            content_copy TEXT NOT NULL,
            original_role TEXT,
            original_author TEXT,
            status TEXT DEFAULT 'pending',
            done_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # ── Quote tree system ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS message_quotes (
            id SERIAL PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            quoted_message_id BIGINT NOT NULL,
            quoting_message_id BIGINT,
            parent_quote_id INTEGER REFERENCES message_quotes(id),
            thread_id TEXT REFERENCES chat_sessions(thread_id),
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # ── Regeneration vote system ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS regen_votes (
            id SERIAL PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            message_id BIGINT NOT NULL,
            original_content TEXT,
            new_content TEXT,
            status TEXT DEFAULT 'active',
            round INTEGER DEFAULT 1,
            expires_at TIMESTAMPTZ,
            resolved_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS regen_vote_ballots (
            id SERIAL PRIMARY KEY,
            vote_id INTEGER REFERENCES regen_votes(id) ON DELETE CASCADE,
            voter_id TEXT REFERENCES users(user_id),
            vote TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (vote_id, voter_id)
        )
    """)

    # All indexes
    idx_statements = [
        "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id_timestamp ON chat_messages(thread_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_user_files_expires_at ON user_files(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_user_files_user_id ON user_files(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_file_usage_user_filename ON file_usage(user_id, filename)",
        "CREATE INDEX IF NOT EXISTS idx_message_responses_created_at ON message_responses(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_project_members_user ON project_members(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_project_folders_parent ON project_folders(parent_folder_id)",
        "CREATE INDEX IF NOT EXISTS idx_project_files_folder ON project_files(folder_id)",
        "CREATE INDEX IF NOT EXISTS idx_project_files_hash ON project_files(file_hash)",
        "CREATE INDEX IF NOT EXISTS idx_task_deposit_items_original_user ON task_deposit_items(original_user_id)",
        "CREATE INDEX IF NOT EXISTS idx_task_deposit_items_project ON task_deposit_items(project_id)",
        "CREATE INDEX IF NOT EXISTS idx_recycle_bin_user_id ON recycle_bin(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_recycle_bin_expires_at ON recycle_bin(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_project_recycle_bin_project_id ON project_recycle_bin(project_id)",
        "CREATE INDEX IF NOT EXISTS idx_project_recycle_bin_expires_at ON project_recycle_bin(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_project_folders_recycle_bin_project ON project_folders_recycle_bin(project_id)",
        "CREATE INDEX IF NOT EXISTS idx_kb_recycle_bin_user ON kb_recycle_bin(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_kb_recycle_bin_expires ON kb_recycle_bin(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_recycle_bin_original_thread_id ON recycle_bin(original_thread_id)",
        "CREATE INDEX IF NOT EXISTS idx_file_analysis_hash_type ON file_analysis(file_hash, file_type)",
        "CREATE INDEX IF NOT EXISTS idx_file_analysis_user ON file_analysis(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_user_files_hash ON user_files(user_id, file_hash)",
        "CREATE INDEX IF NOT EXISTS idx_user_files_thread_id ON user_files(user_id, thread_id)",
        "CREATE INDEX IF NOT EXISTS idx_file_usage_thread_filename ON file_usage(thread_id, filename)",
        "CREATE INDEX IF NOT EXISTS idx_lab_skill_hash ON knowledge_lab_files(skill_summary_hash)",
        "CREATE INDEX IF NOT EXISTS idx_company_kb_skill_hash ON company_knowledge_base(skill_summary_hash)",
        "CREATE INDEX IF NOT EXISTS idx_project_files_skill_hash ON project_files(skill_summary_hash)",
        "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id_id ON chat_messages(thread_id, id)",
        "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated ON chat_sessions(user_id, updated_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_feedback_thread_id ON feedback(thread_id)",
        "CREATE INDEX IF NOT EXISTS idx_audit_runs_project ON audit_runs(project_id, started_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_audit_file_results_run ON audit_file_results(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_audit_file_results_run_func ON audit_file_results(run_id, function_name)",
        "CREATE INDEX IF NOT EXISTS idx_audit_config_func ON audit_config(function_name)",
    ]
    for stmt in idx_statements:
        cur.execute(stmt)

    # ── Batch comparison results table (permanent) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS batch_comparison_results (
            id              SERIAL PRIMARY KEY,
            user_id         TEXT REFERENCES users(user_id),
            task_id         TEXT UNIQUE NOT NULL,
            file_count      INTEGER DEFAULT 0,
            pair_count      INTEGER DEFAULT 0,
            max_risk        REAL DEFAULT 0,
            file_names      TEXT,
            zip_path        TEXT NOT NULL,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Compliance check results + user feedback (for LoRA training pipeline)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS compliance_feedback (
            id              SERIAL PRIMARY KEY,
            user_id         TEXT NOT NULL,
            task_id         TEXT NOT NULL,
            bid_doc_name    TEXT,
            check_file_name TEXT NOT NULL,
            rule_count      INTEGER DEFAULT 0,
            summary_json    JSONB,
            ai_verdict      TEXT,          -- original AI verdict: pass/warning/violation/critical
            user_verdict    TEXT NOT NULL, -- forced feedback: true_violation | false_positive | not_matter
            user_explain    TEXT,          -- brief explanation why
            results_json    JSONB,         -- full check results for training context
            report_html     TEXT,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_compliance_feedback_task
        ON compliance_feedback(user_id, task_id)
    """)

    # ── Quote anomaly results (per-document + cross-bidder) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS quote_anomaly_results (
            id              SERIAL PRIMARY KEY,
            user_id         TEXT REFERENCES users(user_id),
            task_id         TEXT NOT NULL,
            doc_name        TEXT NOT NULL,
            prices          JSONB DEFAULT '[]',
            percentages     JSONB DEFAULT '[]',
            cv              REAL DEFAULT 0,
            same_rate_flag  BOOLEAN DEFAULT FALSE,
            abnormal_drop_flag BOOLEAN DEFAULT FALSE,
            clustering_flag BOOLEAN DEFAULT FALSE,
            benford_deviation REAL DEFAULT 0,
            risk_score      REAL DEFAULT 0,
            details         JSONB DEFAULT '[]',
            matched_prices  JSONB DEFAULT '{}',
            cross_same_rate BOOLEAN DEFAULT FALSE,
            cross_clustering BOOLEAN DEFAULT FALSE,
            max_cross_risk  REAL DEFAULT 0,
            avg_cross_cv    REAL DEFAULT 0,
            checked_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_quote_anomaly_task ON quote_anomaly_results(task_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_quote_anomaly_user ON quote_anomaly_results(user_id)")

    # ── Entity relationships extracted from bid documents ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS entity_relationships (
            id              SERIAL PRIMARY KEY,
            user_id         TEXT REFERENCES users(user_id),
            task_id         TEXT NOT NULL,
            doc_name        TEXT,
            source_entity   TEXT NOT NULL,
            target_entity   TEXT NOT NULL,
            relation_type   TEXT NOT NULL,
            relation_subtype TEXT,
            confidence      REAL DEFAULT 0,
            evidence_text   TEXT,
            external_verified BOOLEAN DEFAULT FALSE,
            external_source TEXT,
            risk_flag       BOOLEAN DEFAULT FALSE,
            risk_reason     TEXT,
            module          TEXT NOT NULL,
            checked_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_entity_rel_task ON entity_relationships(task_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_entity_rel_source ON entity_relationships(source_entity)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_entity_rel_target ON entity_relationships(target_entity)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_entity_rel_type ON entity_relationships(relation_type)")

    # ── Typo / misspelling detection results ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS typo_detection_results (
            id              SERIAL PRIMARY KEY,
            user_id         TEXT REFERENCES users(user_id),
            task_id         TEXT NOT NULL,
            doc_name        TEXT NOT NULL,
            layer           TEXT NOT NULL,
            suspect_text    TEXT NOT NULL,
            context_snippet TEXT,
            suggestions     JSONB DEFAULT '[]',
            confidence      REAL DEFAULT 0,
            position_start  INTEGER,
            position_end    INTEGER,
            is_daxie_error  BOOLEAN DEFAULT FALSE,
            daxie_expected  TEXT,
            daxie_actual    TEXT,
            user_action     TEXT,
            checked_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_typo_results_task ON typo_detection_results(task_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_typo_results_layer ON typo_detection_results(layer)")

    # ── Relationship extraction risk summary (cross-document aggregated score) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS relationship_risk_summary (
            id              SERIAL PRIMARY KEY,
            user_id         TEXT REFERENCES users(user_id),
            task_id         TEXT UNIQUE NOT NULL,
            total_entities  INTEGER DEFAULT 0,
            total_relations INTEGER DEFAULT 0,
            red_flags       JSONB DEFAULT '[]',
            risk_score      REAL DEFAULT 0,
            modules_run     JSONB DEFAULT '[]',
            tianyancha_used BOOLEAN DEFAULT FALSE,
            details         JSONB DEFAULT '{}',
            checked_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_risk_task ON relationship_risk_summary(task_id)")

    # ── Law library (Wiki U1+U3+U4) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS law_masters (
            id                  SERIAL PRIMARY KEY,
            law_name            TEXT NOT NULL,
            short_name          TEXT,
            category            TEXT NOT NULL,
            issuing_authority   TEXT,
            effective_date      DATE,
            expiry_date         DATE,
            status              TEXT DEFAULT 'active',
            scope               TEXT DEFAULT 'national',
            created_at          TIMESTAMPTZ DEFAULT NOW(),
            updated_at          TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS law_versions (
            id              SERIAL PRIMARY KEY,
            law_id          INTEGER REFERENCES law_masters(id) ON DELETE CASCADE,
            version_label   TEXT NOT NULL,
            version_date    DATE,
            is_current      BOOLEAN DEFAULT FALSE,
            change_summary  TEXT,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS law_articles (
            id              SERIAL PRIMARY KEY,
            version_id      INTEGER REFERENCES law_versions(id) ON DELETE CASCADE,
            article_label   TEXT NOT NULL,
            article_text    TEXT NOT NULL,
            tags            TEXT[] DEFAULT '{}',
            sort_order      INTEGER DEFAULT 0,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_law_articles_tags ON law_articles USING GIN(tags)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_law_versions_current ON law_versions(law_id, is_current)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS law_regions (
            id          SERIAL PRIMARY KEY,
            region_code TEXT UNIQUE NOT NULL,
            region_name TEXT NOT NULL,
            parent_code TEXT,
            level       TEXT DEFAULT 'national',
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS law_region_bindings (
            id              SERIAL PRIMARY KEY,
            law_id          INTEGER REFERENCES law_masters(id) ON DELETE CASCADE,
            region_code     TEXT REFERENCES law_regions(region_code) ON DELETE CASCADE,
            binding_type    TEXT DEFAULT 'baseline',
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(law_id, region_code)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_law_region_bindings_region ON law_region_bindings(region_code)")

    # ── law version diffs (U3) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS law_version_diffs (
            id              SERIAL PRIMARY KEY,
            law_id          INTEGER REFERENCES law_masters(id) ON DELETE CASCADE,
            from_version_id INTEGER REFERENCES law_versions(id) ON DELETE CASCADE,
            to_version_id   INTEGER REFERENCES law_versions(id) ON DELETE CASCADE,
            diff_data       JSONB NOT NULL,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(law_id, from_version_id, to_version_id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_law_diff_law ON law_version_diffs(law_id)")

    # ── bid templates (U5) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bid_templates (
            id              SERIAL PRIMARY KEY,
            name            TEXT NOT NULL,
            category        TEXT NOT NULL,
            description     TEXT,
            sections        JSONB NOT NULL DEFAULT '[]',
            tags            TEXT[] DEFAULT '{}',
            is_active       BOOLEAN DEFAULT TRUE,
            version         INTEGER DEFAULT 1,
            created_by      TEXT REFERENCES users(user_id),
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            updated_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bid_templates_cat ON bid_templates(category, is_active)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bid_templates_tags ON bid_templates USING GIN(tags)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS bid_template_versions (
            id              SERIAL PRIMARY KEY,
            template_id     INTEGER REFERENCES bid_templates(id) ON DELETE CASCADE,
            version_label   TEXT NOT NULL,
            snapshot        JSONB NOT NULL,
            change_summary  TEXT,
            created_by      TEXT REFERENCES users(user_id),
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_btv_template ON bid_template_versions(template_id)")

    # ── template usage log (U8) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS template_usage_log (
            id              SERIAL PRIMARY KEY,
            template_id     INTEGER REFERENCES bid_templates(id),
            user_id         TEXT REFERENCES users(user_id),
            project_id      INTEGER,
            used_at         TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tul_template ON template_usage_log(template_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tul_used_at ON template_usage_log(used_at)")

    # ── law change events (U14) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS law_change_events (
            id              SERIAL PRIMARY KEY,
            law_id          INTEGER REFERENCES law_masters(id),
            from_version_id INTEGER,
            to_version_id   INTEGER,
            description     TEXT,
            submitted_by    TEXT REFERENCES users(user_id),
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lce_law ON law_change_events(law_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lce_created ON law_change_events(created_at)")

    # ── audit cases (U13) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_cases (
            id                  SERIAL PRIMARY KEY,
            title               TEXT NOT NULL,
            description         TEXT,
            category            TEXT NOT NULL,
            severity            TEXT NOT NULL,
            resolution          TEXT,
            source_finding_id   INTEGER REFERENCES audit_file_results(id) ON DELETE SET NULL,
            source_run_id       INTEGER REFERENCES audit_runs(id) ON DELETE SET NULL,
            project_id          INTEGER REFERENCES projects(id) ON DELETE SET NULL,
            file_id             INTEGER REFERENCES project_files(id) ON DELETE SET NULL,
            law_refs            JSONB DEFAULT '[]',
            template_refs       JSONB DEFAULT '[]',
            is_resolved         BOOLEAN DEFAULT FALSE,
            created_by          TEXT REFERENCES users(user_id),
            created_at          TIMESTAMPTZ DEFAULT NOW(),
            updated_at          TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_cases_severity ON audit_cases(severity)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_cases_project ON audit_cases(project_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_cases_resolved ON audit_cases(is_resolved)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS case_tags (
            id      SERIAL PRIMARY KEY,
            case_id INTEGER REFERENCES audit_cases(id) ON DELETE CASCADE,
            tag     TEXT NOT NULL,
            UNIQUE(case_id, tag)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_case_tags_case ON case_tags(case_id)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS case_law_links (
            id          SERIAL PRIMARY KEY,
            case_id     INTEGER REFERENCES audit_cases(id) ON DELETE CASCADE,
            article_id  INTEGER REFERENCES law_articles(id) ON DELETE CASCADE,
            relation    TEXT DEFAULT 'cited',
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(case_id, article_id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_case_law_case ON case_law_links(case_id)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS case_template_links (
            id          SERIAL PRIMARY KEY,
            case_id     INTEGER REFERENCES audit_cases(id) ON DELETE CASCADE,
            template_id INTEGER REFERENCES bid_templates(id) ON DELETE CASCADE,
            section_id  TEXT,
            relation    TEXT DEFAULT 'related',
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_case_tpl_case ON case_template_links(case_id)")

    # ── credit check reports ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS credit_check_reports (
            id              SERIAL PRIMARY KEY,
            user_id         TEXT NOT NULL,
            task_id         TEXT,
            file_path       TEXT,
            companies_count INTEGER DEFAULT 0,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ccr_user ON credit_check_reports(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ccr_created ON credit_check_reports(created_at)")

    # ── Anonymous chat history (JSONB per thread) — replaces JSON file storage ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS anon_chat_messages (
            thread_id   TEXT PRIMARY KEY,
            messages    JSONB NOT NULL DEFAULT '[]'::jsonb,
            updated_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
