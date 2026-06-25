"""Database pool, connection management, and table initialization."""
import os
import threading
from contextlib import contextmanager
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from .config import logger

def get_db_connection_args():
    if os.getenv("PG_USER") and os.getenv("PG_PASSWORD"):
        return {
            'dbname': os.getenv('PG_DB', 'postgres'),
            'user': os.getenv('PG_USER'),
            'password': os.getenv('PG_PASSWORD'),
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': int(os.getenv('PG_PORT', 5432)),
            'client_encoding': 'utf8'
        }
    else:
        uri = os.getenv("POSTGRES_URI")
        if not uri:
            raise ValueError("No database connection configuration found.")
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

conn_args = get_db_connection_args()
db_pool = pool.SimpleConnectionPool(1, 20, **conn_args)


@contextmanager
def get_db_connection():
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)


@contextmanager
def db_transaction(conn):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_postgres_tables():
    """Initialize all PostgreSQL tables. Called once at startup."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            import_init_tables(cur, conn)
            conn.commit()
    logger.info("PostgreSQL tables initialized.")


def import_init_tables(cur, conn):
    """Import and execute the full table initialization. Kept in a separate function
    to avoid bloating the module top-level."""
    _run_table_creation(cur, conn)


def _run_table_creation(cur, conn):
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
            BEGIN ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'; EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id SERIAL PRIMARY KEY, user_id TEXT REFERENCES users(user_id),
            thread_id TEXT UNIQUE NOT NULL, title TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
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
            size_bytes INTEGER, created_at TIMESTAMPTZ DEFAULT NOW(), expires_at TIMESTAMPTZ,
            original_stored_path TEXT, file_hash TEXT, original_expires_at TIMESTAMPTZ,
            meta_data JSONB DEFAULT '{}', original_name TEXT, UNIQUE (thread_id, filename)
        )
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
        END $$;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_members (
            id SERIAL PRIMARY KEY, project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            user_id TEXT REFERENCES users(user_id), role TEXT NOT NULL,
            added_at TIMESTAMPTZ DEFAULT NOW(), added_by TEXT REFERENCES users(user_id),
            UNIQUE (project_id, user_id)
        )
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
        END $$;
    """)
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
        CREATE TABLE IF NOT EXISTS project_folders_recycle_bin (
            id SERIAL PRIMARY KEY, original_id INTEGER NOT NULL,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            name TEXT NOT NULL, parent_folder_id INTEGER, original_parent_id INTEGER,
            created_at TIMESTAMPTZ, created_by TEXT, deleted_at TIMESTAMPTZ DEFAULT NOW(),
            expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '3 days'
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS credit_check_reports (
            id SERIAL PRIMARY KEY, user_id TEXT REFERENCES users(user_id),
            task_id TEXT UNIQUE NOT NULL, file_path TEXT NOT NULL,
            companies_count INTEGER DEFAULT 0, created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_credit_reports_user ON credit_check_reports(user_id)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_lab_files (
            id SERIAL PRIMARY KEY, user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
            filename TEXT NOT NULL, original_name TEXT NOT NULL, file_size INTEGER,
            content TEXT, file_hash TEXT UNIQUE, stored_path TEXT,
            uploaded_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lab_user ON knowledge_lab_files(user_id)")
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

    cur.execute("""
        CREATE TABLE IF NOT EXISTS file_analysis (
            id SERIAL PRIMARY KEY, file_hash TEXT NOT NULL, file_type TEXT NOT NULL,
            original_filename TEXT, user_id TEXT REFERENCES users(user_id),
            thread_id TEXT, project_id TEXT, file_size INTEGER,
            extracted_text TEXT, usage_count INTEGER DEFAULT 0,
            last_used_at TIMESTAMPTZ DEFAULT NOW(), created_at TIMESTAMPTZ DEFAULT NOW(),
            deleted_at TIMESTAMPTZ
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
        "CREATE INDEX IF NOT EXISTS idx_recycle_bin_original_thread_id ON recycle_bin(original_thread_id)",
        "CREATE INDEX IF NOT EXISTS idx_file_analysis_hash_type ON file_analysis(file_hash, file_type)",
        "CREATE INDEX IF NOT EXISTS idx_file_analysis_user ON file_analysis(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_user_files_hash ON user_files(user_id, file_hash)",
        "CREATE INDEX IF NOT EXISTS idx_user_files_thread_id ON user_files(user_id, thread_id)",
        "CREATE INDEX IF NOT EXISTS idx_file_usage_thread_filename ON file_usage(thread_id, filename)",
        "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)",
    ]
    for stmt in idx_statements:
        cur.execute(stmt)
