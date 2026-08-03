"""Integration tests for chat blueprint endpoints.

Requires DB (pytest -m db). Mock LLM HTTP with mock_llm_http fixture.
"""
import pytest

pytestmark = pytest.mark.db


class TestNewChat:
    ROUTE = "/new_chat"

    def test_returns_new_thread_id(self, client):
        resp = client.post(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["thread_id"]) > 10

    def test_session_updates(self, client):
        first = client.post(self.ROUTE).get_json()["thread_id"]
        second = client.post(self.ROUTE).get_json()["thread_id"]
        assert first != second


class TestSetMaxTokens:
    ROUTE = "/set_max_tokens"

    def test_sets_tokens(self, client):
        resp = client.post(self.ROUTE, json={"max_tokens": 2000})
        data = resp.get_json()
        assert data["success"] is True
        assert data["max_tokens"] == 2000

    def test_clamps_min_100(self, client):
        resp = client.post(self.ROUTE, json={"max_tokens": 0})
        assert resp.get_json()["max_tokens"] == 100

    def test_clamps_max_4800(self, client):
        resp = client.post(self.ROUTE, json={"max_tokens": 99999})
        assert resp.get_json()["max_tokens"] == 4800


class TestLlmProviders:
    LIST_ROUTE = "/llm_providers"
    SET_ROUTE = "/llm_providers/set"

    def test_list_returns_providers(self, client):
        resp = client.get(self.LIST_ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert "available" in data
        assert "active" in data

    def test_set_requires_login(self, client):
        resp = client.post(self.SET_ROUTE, json={"provider": "deepseek", "model": "deepseek-chat"})
        assert resp.status_code == 403

    def test_set_accepts_logged_in(self, auth_client):
        resp = auth_client.post(self.SET_ROUTE, json={"provider": "deepseek", "model": "deepseek-chat"})
        data = resp.get_json()
        assert data["success"] is True


class TestUploadFile:
    ROUTE = "/upload_file"

    def test_requires_file(self, client):
        resp = client.post(self.ROUTE, data={}, content_type="multipart/form-data")
        assert resp.status_code == 400


class TestFileStation:
    GET_ROUTE = "/get_file_station"
    DELETE_ROUTE = "/delete_file_station"

    def test_anon_returns_empty_list(self, client):
        resp = client.get(self.GET_ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert "files" in data
        assert data["is_anon"] is True

    def test_delete_requires_file_id(self, client):
        resp = client.post(self.DELETE_ROUTE, json={})
        assert resp.status_code == 400

    def test_delete_nonexistent_returns_404(self, auth_client):
        resp = auth_client.post(self.DELETE_ROUTE, json={"file_id": "99999"})
        assert resp.status_code == 404


class TestRecentFiles:
    ROUTE = "/get_recent_files"

    def test_returns_ok(self, client):
        resp = client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert "recent_files" in data


class TestLoadCachedFile:
    ROUTE = "/load_cached_file"

    def test_no_thread_id_returns_401(self, client):
        resp = client.post(self.ROUTE, json={})
        assert resp.status_code == 401


class TestFeedback:
    ROUTE = "/feedback"

    def test_requires_login(self, client):
        resp = client.post(self.ROUTE, json={"rating": 5})
        assert resp.status_code == 403

    def test_submit_feedback(self, auth_client):
        resp = auth_client.post(self.ROUTE, json={"rating": 5, "comment": "Great!"})
        data = resp.get_json()
        assert data["success"] is True

    def test_accepts_any_rating(self, auth_client):
        resp = auth_client.post(self.ROUTE, json={"rating": 99})
        assert resp.get_json()["success"] is True


class TestShareConversation:
    ROUTE = "/share_conversation"

    def test_requires_login(self, client):
        resp = client.post(self.ROUTE)
        assert resp.status_code == 403

    def test_empty_chat_returns_error(self, auth_client):
        resp = auth_client.post(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is False


class TestSend:
    ROUTE = "/send"

    def test_empty_message_returns_400(self, auth_client):
        resp = auth_client.post(self.ROUTE, data={"message": ""}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_requires_login(self, client):
        resp = client.post(self.ROUTE, data={"message": "hello"}, content_type="multipart/form-data")
        assert resp.status_code == 403

    def test_message_too_long(self, auth_client):
        resp = auth_client.post(
            self.ROUTE,
            data={"message": "x" * 10001},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_requires_message_id(self, auth_client):
        resp = auth_client.post(self.ROUTE, data={"message": "hello"}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_send_with_mock_llm(self, auth_client, mock_llm_http):
        resp = auth_client.post(
            self.ROUTE,
            data={"message": "Hello", "message_id": "test-msg-001"},
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        assert data["success"] is True
        assert "assistant_message" in data


class TestSessionManagement:
    UPDATE_ROUTE = "/update_session_title"

    def test_update_title_requires_login(self, client):
        resp = client.post(self.UPDATE_ROUTE, json={"thread_id": "x", "title": "Test"})
        assert resp.status_code in (401, 403)

    def test_update_title_nonexistent(self, auth_client):
        resp = auth_client.post(self.UPDATE_ROUTE, json={"thread_id": "nonexistent", "title": "My Title"})
        assert resp.status_code == 404
