"""Smoke tests for audit blueprint — preflight, start, progress, results, config."""

import json
import uuid
import pytest

pytestmark = [pytest.mark.db, pytest.mark.usefixtures("mock_llm_http")]


def _create_project_and_folder(admin_client):
    """Insert a minimal project + folder so audit routes pass validation."""
    from app.database import get_db_connection
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM users WHERE username = 'CEO'")
            uid = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO projects (name, status, created_by) VALUES (%s, 'active', %s) RETURNING id",
                ("测试项目", uid),
            )
            pid = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO project_folders (name, project_id, created_by) VALUES (%s, %s, %s) RETURNING id",
                ("测试文件夹", pid, uid),
            )
            fid = cur.fetchone()[0]
        conn.commit()
    return pid, fid


class TestAuditPreflight:
    URL = "/audit/preflight"

    def test_no_body(self, admin_client):
        resp = admin_client.post(self.URL, json={})
        assert resp.status_code == 400

    def test_valid_folders(self, admin_client):
        _, fid = _create_project_and_folder(admin_client)
        resp = admin_client.post(self.URL, json={"folder_ids": [fid]})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]

    def test_non_admin_forbidden(self, auth_client):
        resp = auth_client.post(self.URL, json={"folder_ids": [1]})
        assert resp.status_code == 403


class TestAuditStart:
    URL = "/audit/start"

    def test_missing_folder_ids(self, admin_client):
        resp = admin_client.post(self.URL, json={})
        assert resp.status_code == 400

    def test_missing_project_id(self, admin_client):
        resp = admin_client.post(self.URL, json={"folder_ids": [1]})
        assert resp.status_code == 400

    def test_valid_start(self, admin_client):
        pid, fid = _create_project_and_folder(admin_client)
        resp = admin_client.post(
            self.URL,
            json={
                "folder_ids": [fid],
                "project_id": pid,
                "enabled_functions": ["typo_detection"],
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]
        assert "run_id" in data

    def test_non_admin_forbidden(self, auth_client):
        resp = auth_client.post(self.URL, json={"folder_ids": [1]})
        assert resp.status_code == 403


class TestAuditProgress:
    URL = "/audit/progress/"

    def test_nonexistent_run(self, admin_client):
        """SSE endpoint always returns 200; consume stream to avoid context leak."""
        resp = admin_client.get(self.URL + "99999")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/event-stream")
        for _ in resp.response:
            break

    def test_non_admin_forbidden(self, auth_client):
        resp = auth_client.get(self.URL + "1")
        assert resp.status_code == 403


class TestAuditResult:
    URL = "/audit/result/"

    def test_nonexistent_run(self, admin_client):
        resp = admin_client.get(self.URL + "99999")
        assert resp.status_code == 404

    def test_non_admin_forbidden(self, auth_client):
        resp = auth_client.get(self.URL + "1")
        assert resp.status_code == 403


class TestAuditHistory:
    URL = "/audit/history/"

    def test_valid_project(self, admin_client):
        resp = admin_client.get(self.URL + "0")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]

    def test_non_admin_forbidden(self, auth_client):
        resp = auth_client.get(self.URL + "1")
        assert resp.status_code == 403


class TestAuditRunning:
    URL = "/audit/running/"

    def test_valid_project(self, admin_client):
        resp = admin_client.get(self.URL + "0")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]

    def test_non_admin_forbidden(self, auth_client):
        resp = auth_client.get(self.URL + "1")
        assert resp.status_code == 403


class TestAuditDownload:
    DOCX_URL = "/audit/download/99999/docx"
    XLSX_URL = "/audit/download/99999/xlsx"

    def test_docx_nonexistent(self, admin_client):
        resp = admin_client.get(self.DOCX_URL)
        assert resp.status_code == 404

    def test_xlsx_nonexistent(self, admin_client):
        resp = admin_client.get(self.XLSX_URL)
        assert resp.status_code == 404

    def test_docx_non_admin_forbidden(self, auth_client):
        resp = auth_client.get(self.DOCX_URL)
        assert resp.status_code == 403


class TestAuditConfig:
    GET_URL = "/audit/config"
    PUT_URL = "/audit/config"

    def test_get_config(self, admin_client):
        resp = admin_client.get(self.GET_URL)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]

    def test_update_config(self, admin_client):
        resp = admin_client.put(
            self.PUT_URL,
            json={
                "configs": [
                    {
                        "function_name": "test_fn",
                        "enabled_by_default": True,
                        "fail_threshold": 0.7,
                        "weight": 0.5,
                        "severity_thresholds": {
                            "critical": 0.9,
                            "warning": 0.7,
                            "info": 0.5,
                        },
                    }
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]

    def test_update_empty(self, admin_client):
        resp = admin_client.put(self.PUT_URL, json={})
        assert resp.status_code == 400

    def test_non_admin_forbidden_get(self, auth_client):
        resp = auth_client.get(self.GET_URL)
        assert resp.status_code == 403

    def test_non_admin_forbidden_put(self, auth_client):
        resp = auth_client.put(self.PUT_URL, json={})
        assert resp.status_code == 403
