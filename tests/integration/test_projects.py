"""Integration tests for projects blueprint (GET /user_project_files).

Requires PostgreSQL (clean_tables fixtures reset between tests).
"""
import pytest
import uuid
from werkzeug.security import generate_password_hash


pytestmark = pytest.mark.db


class TestUserProjectFiles:
    ROUTE = "/user_project_files"

    def _create_project_data(self, client, user_id):
        """Helper: insert a project + membership + file for the given user."""
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO projects (name, description, created_by) "
                    "VALUES (%s, %s, %s) RETURNING id",
                    ("测试项目", "Test project description", user_id)
                )
                project_id = cur.fetchone()[0]
                cur.execute(
                    "INSERT INTO project_members (project_id, user_id, role, added_by) "
                    "VALUES (%s, %s, %s, %s)",
                    (project_id, user_id, 'member', user_id)
                )
                cur.execute(
                    "INSERT INTO project_files (project_id, filename, original_name, "
                    "file_size, mime_type, stored_path, uploaded_by) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (project_id, "bid.pdf", "投标文件.pdf", 102400,
                     "application/pdf", "/tmp/test/bid.pdf", user_id)
                )
                cur.execute(
                    "INSERT INTO project_files (project_id, filename, original_name, "
                    "file_size, mime_type, stored_path, uploaded_by) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (project_id, "spec.docx", "技术规格书.docx", 204800,
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                     "/tmp/test/spec.docx", user_id)
                )
            conn.commit()
        return project_id

    def test_requires_consent(self, client):
        """Returns 403 when consent_value is not 1."""
        with client.session_transaction() as sess:
            sess['user_id'] = str(uuid.uuid4())
            sess['username'] = 'nobody'
        resp = client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_requires_login(self, client):
        """Returns 401 when user is not logged in (consent given but no user_id)."""
        with client.session_transaction() as sess:
            sess['consent_value'] = 1
        resp = client.get(self.ROUTE)
        assert resp.status_code == 401

    def test_returns_user_files(self, auth_client):
        """Returns files belonging to the logged-in user's projects."""
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM users WHERE username = 'testuser'")
                row = cur.fetchone()
                user_id = row[0] if row else str(uuid.uuid4())
        self._create_project_data(auth_client, user_id)
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["files"]) >= 2
        names = {f["original_name"] for f in data["files"]}
        assert "投标文件.pdf" in names
        assert "技术规格书.docx" in names

    def test_returns_empty_for_no_projects(self, auth_client):
        """Returns empty list when user has no project memberships."""
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["files"]) == 0

    def test_includes_project_name(self, auth_client):
        """Each file entry includes the project name."""
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM users WHERE username = 'testuser'")
                row = cur.fetchone()
                user_id = row[0] if row else str(uuid.uuid4())
        self._create_project_data(auth_client, user_id)
        resp = auth_client.get(self.ROUTE)
        data = resp.get_json()
        for f in data["files"]:
            assert "project_name" in f
            assert f["project_name"] == "测试项目"

    def test_does_not_see_other_users_files(self, auth_client):
        """A user should not see files from projects they don't belong to."""
        from app.database import get_db_connection
        # Create project + files belonging to another user
        other_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (user_id, username, pin_hash, pin_length, role, is_active) "
                    "VALUES (%s, %s, %s, %s, 'user', TRUE) ON CONFLICT (username) DO NOTHING",
                    (other_id, "otheruser", generate_password_hash("123456"), 6)
                )
                cur.execute(
                    "INSERT INTO projects (name, description, created_by) "
                    "VALUES (%s, %s, %s) RETURNING id",
                    ("其他项目", "Other user project", other_id)
                )
                pid = cur.fetchone()[0]
                cur.execute(
                    "INSERT INTO project_members (project_id, user_id, role, added_by) "
                    "VALUES (%s, %s, %s, %s)",
                    (pid, other_id, 'member', other_id)
                )
                cur.execute(
                    "INSERT INTO project_files (project_id, filename, original_name, "
                    "file_size, mime_type, stored_path, uploaded_by) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (pid, "secret.pdf", "秘密文件.pdf", 51200,
                     "application/pdf", "/tmp/secret.pdf", other_id)
                )
            conn.commit()
        resp = auth_client.get(self.ROUTE)
        data = resp.get_json()
        names = {f["original_name"] for f in data["files"]}
        assert "秘密文件.pdf" not in names
