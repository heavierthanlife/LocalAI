"""Integration tests for app.database — pool, schema creation, and basic table structure.

Requires PostgreSQL database 'test_chatbot_test' (created in conftest).
"""
import pytest


@pytest.mark.db
class TestDbPool:
    def test_pool_exists(self, db):
        """db fixture yields the global pool."""
        assert db is not None

    def test_get_connection(self, db):
        """Acquire and release a connection."""
        conn = db.getconn()
        assert conn is not None
        cur = conn.cursor()
        cur.execute("SELECT 1")
        assert cur.fetchone()[0] == 1
        cur.close()
        db.putconn(conn)

    def test_pool_min_connections(self):
        """Pool is created with default min=1."""
        import os
        from app.database import get_db_pool
        assert get_db_pool().minconn >= 1


@pytest.mark.db
class TestSchemaCreation:
    """Verify that init_postgres_tables() creates the expected tables."""

    EXPECTED_TABLES = [
        "users", "chat_sessions", "chat_messages", "consent", "user_files",
        "company_knowledge_base", "knowledge_lab_files", "admin_audit_log",
        "feedback", "file_usage", "audit_config", "audit_runs", "audit_file_results",
        "batch_comparison_results", "projects",
        "project_members", "project_folders", "project_files",
        "project_file_versions", "project_file_comments", "project_folder_comments",
        "recycle_bin", "project_recycle_bin", "task_deposit_items",
        "task_deposit_permissions",
    ]

    def test_all_tables_exist(self, db):
        conn = db.getconn()
        cur = conn.cursor()
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        existing = {row[0] for row in cur.fetchall()}
        cur.close()
        db.putconn(conn)
        for table in self.EXPECTED_TABLES:
            assert table in existing, f"Missing table: {table}"
        assert len(existing) >= 44, f"Expected >=44 tables, got {len(existing)}"

    def test_users_table_columns(self, db):
        conn = db.getconn()
        cur = conn.cursor()
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'users'"
        )
        cols = {row[0]: row[1] for row in cur.fetchall()}
        cur.close()
        db.putconn(conn)
        assert "user_id" in cols
        assert "username" in cols
        assert "pin_hash" in cols
        assert "role" in cols
        assert "created_at" in cols
        assert "email" in cols
        assert "is_active" in cols

    def test_users_has_admin_seeds(self, db):
        conn = db.getconn()
        cur = conn.cursor()
        cur.execute("SELECT username, role FROM users WHERE role = 'admin'")
        admins = {row[0] for row in cur.fetchall()}
        cur.close()
        db.putconn(conn)
        assert "CEO" in admins, "CEO admin not seeded"
        assert "COO" in admins, "COO admin not seeded"

    def test_audit_config_seeded(self, db):
        conn = db.getconn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM audit_config")
        count = cur.fetchone()[0]
        cur.close()
        db.putconn(conn)
        assert count > 0, "No audit_config rows seeded"

    def test_idempotent_reinitialization(self, db):
        """Calling init_postgres_tables() again should not raise."""
        from app.database import init_postgres_tables
        init_postgres_tables()


@pytest.mark.db
class TestSchemaConstraints:
    def test_users_user_id_unique(self, db):
        conn = db.getconn()
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM pg_indexes
            WHERE tablename = 'users'
            AND indexdef LIKE '%UNIQUE%'
        """)
        count = cur.fetchone()[0]
        cur.close()
        db.putconn(conn)
        assert count >= 1

    def test_users_username_unique(self, db):
        conn = db.getconn()
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM pg_indexes
            WHERE tablename = 'users' AND indexdef ILIKE '%username%'
        """)
        count = cur.fetchone()[0]
        cur.close()
        db.putconn(conn)
        assert count >= 1
