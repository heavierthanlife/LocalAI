"""End-to-end workflow tests — full-stack scenarios with real DB + mock LLM."""

import json
import uuid
import pytest

pytestmark = [pytest.mark.db, pytest.mark.e2e, pytest.mark.usefixtures("mock_llm_http")]


class TestE2EAuthFlow:
    """Complete authentication workflow: register -> login -> session -> logout."""

    def test_create_account(self, client):
        """Create a new account and verify session is set."""
        resp = client.post("/create_account", json={
            "username": "e2euser", "pin": "123456"
        })
        data = resp.get_json()
        assert data["success"] is True
        assert data["username"] == "e2euser"

        with client.session_transaction() as sess:
            assert sess.get("user_id") is not None
            assert sess.get("username") == "e2euser"
            assert sess.get("role") == "user"
            assert sess.get("consent_value") == 1

        resp = client.get("/check_auth")
        data = resp.get_json()
        assert data["authenticated"] is True
        assert data["username"] == "e2euser"

    def test_create_account_validation(self, client):
        """Account creation validates inputs."""
        resp = client.post("/create_account", json={})
        assert resp.status_code == 400

        resp = client.post("/create_account", json={"username": "ab", "pin": "123456"})
        assert resp.status_code == 400

        resp = client.post("/create_account", json={"username": "validname", "pin": "abc"})
        assert resp.status_code == 400

    def test_login_after_logout(self, client):
        """Create account, logout, login again."""
        client.post("/create_account", json={
            "username": "loginuser", "pin": "123456"
        })

        with client.session_transaction() as sess:
            sess.clear()

        resp = client.get("/check_auth")
        data = resp.get_json()
        assert data["authenticated"] is False

        resp = client.post("/login", json={
            "username": "loginuser", "pin": "123456"
        })
        data = resp.get_json()
        assert data["success"] is True

        resp = client.get("/check_auth")
        data = resp.get_json()
        assert data["authenticated"] is True
        assert data["username"] == "loginuser"

    def test_admin_client(self, admin_client):
        """Admin session provides full admin access."""
        with admin_client.session_transaction() as sess:
            assert sess.get("role") == "admin"
        resp = admin_client.get("/admin/users")
        data = resp.get_json()
        assert data["success"] is True


class TestE2EChatFlow:
    """Complete chat workflow: new chat -> send -> receive -> share."""

    def test_new_chat(self, auth_client):
        """Start a new chat session."""
        resp = auth_client.post("/new_chat")
        data = resp.get_json()
        assert data["success"] is True
        assert "thread_id" in data

        with auth_client.session_transaction() as sess:
            assert sess.get("thread_id") == data["thread_id"]

    def test_send_message(self, auth_client):
        """Send a message and get an AI response."""
        auth_client.post("/new_chat")
        with auth_client.session_transaction() as sess:
            thread_id = sess.get("thread_id")

        import uuid as _uuid
        resp = auth_client.post("/send", data={
            "message": "请写一份招标公告",
            "message_id": str(_uuid.uuid4()),
            "thread_id": thread_id,
        })
        data = resp.get_json()
        assert data["success"] is True

        resp = auth_client.get(f"/load_session/{thread_id}")
        data = resp.get_json()
        assert data["success"] is True
        messages = data.get("messages") or data.get("data", [])
        all_text = " ".join(m.get("content", "") for m in messages)
        assert len(all_text) > 0

    def test_chat_history(self, auth_client):
        """Created chats appear in session list."""
        auth_client.post("/new_chat", json={"prompt": "历史测试"})

        resp = auth_client.get("/get_sessions")
        data = resp.get_json()
        assert data["success"] is True

    def test_share_conversation(self, auth_client):
        """Share a conversation and verify share token."""
        auth_client.post("/new_chat")
        with auth_client.session_transaction() as sess:
            thread_id = sess.get("thread_id")

        import uuid as _uuid
        resp = auth_client.post("/send", data={
            "message": "共享内容",
            "message_id": str(_uuid.uuid4()),
            "thread_id": thread_id,
        })
        assert resp.status_code == 200

        resp = auth_client.post("/share_conversation", json={
            "thread_id": thread_id
        })
        data = resp.get_json()
        assert data["success"] is True


class TestE2EFileFlow:
    """File upload -> process -> use in chat workflow."""

    def _ensure_chat_session(self, auth_client):
        """Ensure a valid chat session exists for file upload FK."""
        from app.database import get_db_connection
        user_id = None
        with auth_client.session_transaction() as sess:
            user_id = sess.get("user_id")
            thread_id = sess.get("thread_id", "e2e-thread")
            sess["thread_id"] = thread_id
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_sessions (user_id, thread_id, title) VALUES (%s, %s, %s) ON CONFLICT (thread_id) DO NOTHING",
                    (user_id, thread_id, "e2e-test")
                )
            conn.commit()
        return thread_id

    def test_upload_file(self, auth_client, test_bid_file):
        """Upload a bid document."""
        self._ensure_chat_session(auth_client)
        with open(test_bid_file, "rb") as f:
            resp = auth_client.post(
                "/upload_file",
                data={"file": (f, "投标书.txt")},
                content_type="multipart/form-data",
            )
        data = resp.get_json()
        assert data["success"] is True

    def test_file_station(self, auth_client, test_bid_file):
        """Uploaded files appear in file station."""
        self._ensure_chat_session(auth_client)
        with open(test_bid_file, "rb") as f:
            auth_client.post(
                "/upload_file",
                data={"file": (f, "投标书.txt")},
                content_type="multipart/form-data",
            )

        resp = auth_client.get("/get_file_station")
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data.get("files") or data.get("data"), list)

    def test_recent_files(self, auth_client, test_bid_file):
        """Recently uploaded files appear in recent list."""
        self._ensure_chat_session(auth_client)
        with open(test_bid_file, "rb") as f:
            auth_client.post(
                "/upload_file",
                data={"file": (f, "投标书.txt")},
                content_type="multipart/form-data",
            )
        resp = auth_client.get("/get_recent_files")
        data = resp.get_json()
        assert data["success"] is True


class TestE2EAdminFlow:
    """Admin dashboard workflows."""

    def test_admin_dashboard(self, admin_client):
        """Admin can access all major dashboard endpoints."""
        for ep in [
            "/admin/users", "/admin/analytics", "/admin/audit_log",
            "/admin/db_tables", "/admin/llm_providers", "/admin/vl_status",
            "/admin/system_prompt", "/admin/runtime_config",
            "/admin/runtime_config_schema", "/admin/db_tables_overview",
            "/admin/user_emails", "/admin/user_assets", "/admin/file_audit",
        ]:
            resp = admin_client.get(ep)
            data = resp.get_json()
            assert data["success"] is True, f"{ep} failed: {data.get('error')}"

    def test_admin_config_workflow(self, admin_client):
        """Admin can update runtime config values."""
        resp = admin_client.post("/admin/runtime_config", json={
            "llm_timeout_seconds": 120
        })
        assert resp.status_code == 200

        resp = admin_client.get("/admin/runtime_config")
        config = resp.get_json()["config"]
        assert config.get("llm_timeout_seconds") == 120

    def test_admin_notification_workflow(self, admin_client):
        """Admin notification list and mark-read."""
        resp = admin_client.get("/admin/notifications")
        data = resp.get_json()
        assert data["success"] is True

        resp = admin_client.post("/admin/notifications/mark_read", json={})
        assert resp.status_code == 200

    def test_admin_search_cache_ttl(self, admin_client):
        """Admin can update search cache TTL."""
        resp = admin_client.post("/admin/search_cache_config", json={
            "action": "set_ttl", "ttl_hours": 12
        })
        assert resp.status_code == 200

    def test_admin_change_provider(self, admin_client):
        """Admin can change active LLM provider."""
        resp = admin_client.post("/admin/runtime_config", json={
            "active_llm_provider": "deepseek"
        })
        assert resp.status_code == 200


class TestE2EErrorPages:
    """Error responses for edge cases."""

    def test_404_returns_error(self, client):
        """Unknown route returns 404 status."""
        resp = client.get("/admin/nonexistent_route_xyz")
        assert resp.status_code == 404

    def test_401_no_session(self, client):
        """No session returns 401 for protected routes."""
        resp = client.get("/check_auth")
        data = resp.get_json()
        assert data.get("authenticated") is False

    def test_403_non_admin(self, auth_client):
        """Non-admin cannot access admin routes."""
        resp = auth_client.get("/admin/users")
        assert resp.status_code == 403

    def test_405_wrong_method(self, client):
        """Wrong HTTP method returns client error."""
        resp = client.post("/check_auth")
        assert resp.status_code in (400, 404, 405)


class TestE2ELlmProviderFlow:
    """LLM provider listing and activation."""

    def test_list_providers(self, auth_client):
        """User can list available LLM providers."""
        resp = auth_client.get("/llm_providers")
        data = resp.get_json()
        assert data["success"] is True
        assert "available" in data
        assert "active" in data

    def test_set_provider(self, admin_client):
        """Admin can change active provider via runtime_config."""
        resp = admin_client.post("/admin/runtime_config", json={
            "active_llm_provider": "zhipu"
        })
        assert resp.status_code == 200


class TestE2ESearchFlow:
    """Search and session persistence."""

    def test_session_persistence(self, auth_client):
        """Session data persists across requests with thread_id."""
        auth_client.post("/new_chat")
        with auth_client.session_transaction() as sess:
            thread_id = sess.get("thread_id")
        assert thread_id is not None

        import uuid as _uuid
        resp = auth_client.post("/send", data={
            "message": "搜索测试",
            "message_id": str(_uuid.uuid4()),
            "thread_id": thread_id,
        })
        assert resp.status_code == 200

        resp = auth_client.get(f"/load_session/{thread_id}")
        data = resp.get_json()
        assert data["success"] is True

    def test_llm_providers_unauthenticated(self, client):
        """Unauthenticated user can still list providers (public endpoint)."""
        resp = client.get("/llm_providers")
        data = resp.get_json()
        assert data["success"] is True
        assert "available" in data
