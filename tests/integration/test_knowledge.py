"""Integration tests for knowledge lab + company KB routes."""

import io
import json
import pytest

pytestmark = [pytest.mark.db, pytest.mark.usefixtures("mock_llm_http")]


def _upload_kb_file(client, route, filename="test.txt", content="Hello World", category=None):
    """Helper: upload a tiny text file to a knowledge route."""
    data = {"file": (io.BytesIO(content.encode()), filename)}
    if category:
        data["category"] = category
    return client.post(route, data=data, content_type="multipart/form-data")


class TestKnowledgeLabList:
    LIST_ROUTE = "/knowledge_lab/list"

    def test_requires_consent(self, client):
        resp = client.get(self.LIST_ROUTE)
        assert resp.status_code == 403

    def test_requires_login(self, auth_client):
        # Remove user_id from session
        with auth_client.session_transaction() as sess:
            sess.pop("user_id", None)
        resp = auth_client.get(self.LIST_ROUTE)
        assert resp.status_code == 401

    def test_returns_empty_list(self, auth_client):
        resp = auth_client.get(self.LIST_ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert data["files"] == []

    def test_returns_uploaded_files(self, auth_client):
        _upload_kb_file(auth_client, "/knowledge_lab/upload")
        resp = auth_client.get(self.LIST_ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["files"]) == 1
        assert data["files"][0]["original_name"] == "test.txt"

    def test_scoped_to_user(self, auth_client):
        _upload_kb_file(auth_client, "/knowledge_lab/upload", filename="user1.txt")
        resp = auth_client.get(self.LIST_ROUTE)
        data = resp.get_json()
        assert len(data["files"]) == 1


class TestKnowledgeLabUpload:
    UPLOAD_ROUTE = "/knowledge_lab/upload"

    def test_requires_login(self, client):
        resp = client.post(self.UPLOAD_ROUTE, data={}, content_type="multipart/form-data")
        assert resp.status_code in (401, 403)

    def test_requires_file(self, auth_client):
        resp = auth_client.post(self.UPLOAD_ROUTE, data={}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_upload_success(self, auth_client):
        resp = _upload_kb_file(auth_client, self.UPLOAD_ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert "file_id" in data
        assert data["filename"] == "test.txt"

    def test_rejects_unsupported_type(self, auth_client):
        resp = _upload_kb_file(auth_client, self.UPLOAD_ROUTE, filename="bad.exe")
        assert resp.status_code == 400

    def test_rejects_duplicate_hash(self, auth_client):
        _upload_kb_file(auth_client, self.UPLOAD_ROUTE)
        resp = _upload_kb_file(auth_client, self.UPLOAD_ROUTE)
        assert resp.status_code == 409


class TestKnowledgeLabDelete:
    DELETE_TPL = "/knowledge_lab/delete/%d"

    def _upload_and_delete(self, auth_client):
        resp = _upload_kb_file(auth_client, "/knowledge_lab/upload")
        fid = resp.get_json()["file_id"]
        return auth_client.post(self.DELETE_TPL % fid)

    def test_requires_login(self, client):
        resp = client.post(self.DELETE_TPL % 1)
        assert resp.status_code in (401, 403)

    def test_requires_consent(self, client):
        with client.session_transaction() as sess:
            sess["consent_value"] = 0
        resp = client.post(self.DELETE_TPL % 1)
        assert resp.status_code == 403

    def test_delete_existing(self, auth_client):
        resp = self._upload_and_delete(auth_client)
        data = resp.get_json()
        assert data["success"] is True

    def test_delete_nonexistent(self, auth_client):
        resp = auth_client.post(self.DELETE_TPL % 99999)
        assert resp.status_code == 404

    def test_cannot_delete_others_file(self, auth_client):
        resp = _upload_kb_file(auth_client, "/knowledge_lab/upload")
        fid = resp.get_json()["file_id"]
        import uuid
        other_id = str(uuid.uuid4())
        from app.database import get_db_connection
        from werkzeug.security import generate_password_hash
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (user_id, username, pin_hash, pin_length, role, is_active) "
                    "VALUES (%s, %s, %s, %s, 'user', TRUE) ON CONFLICT (username) DO NOTHING",
                    (other_id, "otheruser", generate_password_hash("123456"), 6)
                )
            conn.commit()
        with auth_client.session_transaction() as sess:
            sess['user_id'] = other_id
            sess['username'] = 'otheruser'
            sess['role'] = 'user'
        resp = auth_client.post(self.DELETE_TPL % fid)
        assert resp.status_code == 404


class TestKnowledgeLabContent:
    CONTENT_TPL = "/knowledge_lab/content/%d"

    def test_requires_login(self, client):
        resp = client.get(self.CONTENT_TPL % 1)
        assert resp.status_code in (401, 403)

    def test_returns_content(self, auth_client):
        resp = _upload_kb_file(auth_client, "/knowledge_lab/upload")
        fid = resp.get_json()["file_id"]
        resp = auth_client.get(self.CONTENT_TPL % fid)
        data = resp.get_json()
        assert data["success"] is True
        assert "Hello World" in data["content"]

    def test_nonexistent(self, auth_client):
        resp = auth_client.get(self.CONTENT_TPL % 99999)
        assert resp.status_code == 404


class TestKnowledgeLabRename:
    RENAME_TPL = "/knowledge_lab/rename/%d"

    def test_admin_required(self, auth_client):
        resp = auth_client.post(self.RENAME_TPL % 1, json={"name": "NewName"})
        assert resp.status_code == 403

    def test_rename_by_admin(self, admin_client):
        resp = _upload_kb_file(admin_client, "/knowledge_lab/upload")
        fid = resp.get_json()["file_id"]
        resp = admin_client.post(self.RENAME_TPL % fid, json={"name": "Renamed.txt"})
        assert resp.status_code == 200

    def test_empty_name_rejected(self, admin_client):
        resp = _upload_kb_file(admin_client, "/knowledge_lab/upload")
        fid = resp.get_json()["file_id"]
        resp = admin_client.post(self.RENAME_TPL % fid, json={"name": ""})
        assert resp.status_code == 400


class TestKnowledgeLabGenerateSkill:
    GENERATE_TPL = "/knowledge_lab/generate_skill/%d"

    def test_requires_login(self, client):
        resp = client.post(self.GENERATE_TPL % 1)
        assert resp.status_code == 401

    def test_missing_file(self, auth_client):
        resp = auth_client.post(self.GENERATE_TPL % 99999)
        assert resp.status_code == 404


class TestCompanyKbList:
    LIST_ROUTE = "/company_kb/list"

    def test_requires_consent(self, client):
        resp = client.get(self.LIST_ROUTE)
        assert resp.status_code == 403

    def test_returns_empty_list(self, auth_client):
        resp = auth_client.get(self.LIST_ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert data["files"] == []
        assert "total" in data

    def test_pagination(self, auth_client):
        resp = auth_client.get(self.LIST_ROUTE + "?page=1&per_page=10")
        assert resp.status_code == 200

    def test_filter_by_category(self, auth_client):
        resp = auth_client.get(self.LIST_ROUTE + "?category=测试")
        assert resp.status_code == 200


class TestCompanyKbCategories:
    CATEGORIES_ROUTE = "/company_kb/categories"

    def test_requires_consent(self, client):
        resp = client.get(self.CATEGORIES_ROUTE)
        assert resp.status_code == 403

    def test_returns_empty_initially(self, auth_client):
        resp = auth_client.get(self.CATEGORIES_ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["categories"], list)


class TestCompanyKbUpload:
    UPLOAD_ROUTE = "/company_kb/upload"

    def test_admin_required(self, auth_client):
        resp = _upload_kb_file(auth_client, self.UPLOAD_ROUTE, category="test")
        assert resp.status_code == 403

    def test_requires_category(self, admin_client):
        resp = _upload_kb_file(admin_client, self.UPLOAD_ROUTE)
        assert resp.status_code == 400

    def test_upload_success(self, admin_client):
        resp = _upload_kb_file(admin_client, self.UPLOAD_ROUTE, category="测试文档")
        data = resp.get_json()
        assert data["success"] is True
        assert "file_id" in data

    def test_rejects_unsupported_type(self, admin_client):
        resp = _upload_kb_file(admin_client, self.UPLOAD_ROUTE, filename="bad.exe", category="test")
        assert resp.status_code == 400


class TestCompanyKbContent:
    CONTENT_TPL = "/company_kb/content/%d"

    def test_requires_login(self, client):
        resp = client.get(self.CONTENT_TPL % 1)
        assert resp.status_code in (401, 403)

    def test_returns_content(self, admin_client):
        resp = _upload_kb_file(admin_client, "/company_kb/upload", content="Company secret", category="机密")
        fid = resp.get_json()["file_id"]
        resp = admin_client.get(self.CONTENT_TPL % fid)
        data = resp.get_json()
        assert data["success"] is True
        assert "Company secret" in data["content"]

    def test_nonexistent(self, auth_client):
        resp = auth_client.get(self.CONTENT_TPL % 99999)
        assert resp.status_code == 404

    def test_any_user_can_read(self, auth_client, admin_client):
        resp = _upload_kb_file(admin_client, "/company_kb/upload", content="Public doc", category="公开")
        fid = resp.get_json()["file_id"]
        resp = auth_client.get(self.CONTENT_TPL % fid)
        assert resp.status_code == 200


class TestCompanyKbDelete:
    DELETE_TPL = "/company_kb/delete/%d"

    def test_admin_required(self, auth_client):
        resp = auth_client.post(self.DELETE_TPL % 1)
        assert resp.status_code == 403

    def test_delete_existing(self, admin_client):
        resp = _upload_kb_file(admin_client, "/company_kb/upload", category="test")
        fid = resp.get_json()["file_id"]
        resp = admin_client.post(self.DELETE_TPL % fid)
        data = resp.get_json()
        assert data["success"] is True

    def test_delete_nonexistent(self, admin_client):
        resp = admin_client.post(self.DELETE_TPL % 99999)
        assert resp.status_code == 404


class TestCompanyKbSearch:
    SEARCH_ROUTE = "/company_kb/search"

    def test_requires_consent(self, client):
        resp = client.get(self.SEARCH_ROUTE)
        assert resp.status_code in (401, 403)

    def test_requires_query(self, auth_client):
        resp = auth_client.get(self.SEARCH_ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert data["results"] == []

    def test_short_query_returns_empty(self, auth_client):
        resp = auth_client.get(self.SEARCH_ROUTE + "?q=a")
        data = resp.get_json()
        assert data["results"] == []

    def test_search_finds_content(self, admin_client):
        _upload_kb_file(admin_client, "/company_kb/upload", content="特殊条款和条件适用于本次招标", category="招标")
        resp = admin_client.get(self.SEARCH_ROUTE + "?q=特殊条款")
        assert resp.status_code == 200
