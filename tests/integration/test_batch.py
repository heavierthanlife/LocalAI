"""Tests for batch blueprint — comparison, quote anomaly, relationships, typos."""

import json
import uuid
import io
import pytest

pytestmark = [pytest.mark.db, pytest.mark.usefixtures("mock_llm_http")]


# ==================== Helper ====================

def _file_data(path_or_str, filename="test.txt"):
    """Wrap a file path or string into a BytesIO for multipart upload."""
    if hasattr(path_or_str, "read"):
        return (path_or_str, filename)
    return (io.BytesIO(path_or_str.encode("utf-8")), filename)


# ==================== Compare Batch ====================

class TestCompareBatch:
    URL = "/compare_batch"

    def test_no_files(self, auth_client):
        resp = auth_client.post(self.URL, data={}, content_type="multipart/form-data")
        assert resp.status_code == 400
        data = resp.get_json()
        assert not data["success"]

    def test_single_file(self, auth_client, test_bid_file):
        with open(test_bid_file, "rb") as f:
            resp = auth_client.post(
                self.URL,
                data={"files": (f, "bid.txt")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 400
        data = resp.get_json()
        assert not data["success"]

    def test_two_files_success(self, auth_client):
        files = [
            _file_data("招标文件内容A " * 50, "a.txt"),
            _file_data("招标文件内容B " * 50, "b.txt"),
        ]
        resp = auth_client.post(
            self.URL,
            data={"files": files},
            content_type="multipart/form-data",
        )
        # May return 200 or 409 (resource_busy) depending on task lock state
        data = resp.get_json()
        if resp.status_code == 409:
            assert data.get("error") == "resource_busy"
        else:
            assert resp.status_code == 200
            assert data["success"]
            assert data["pair_count"] >= 1
            assert "download_url" in data

    def test_with_template(self, auth_client):
        files = [
            _file_data("招标文件内容X " * 50, "x.txt"),
            _file_data("招标文件内容Y " * 50, "y.txt"),
        ]
        template = _file_data("模板参考内容 " * 30, "template.txt")
        resp = auth_client.post(
            self.URL,
            data={"files": files, "template_file": template},
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        if resp.status_code == 409:
            assert data.get("error") == "resource_busy"
        else:
            assert resp.status_code == 200
            assert data["success"]

    def test_more_than_ten_files(self, auth_client):
        files = [_file_data(f"file{i} " * 10, f"f{i}.txt") for i in range(11)]
        resp = auth_client.post(
            self.URL,
            data={"files": files},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert not resp.get_json()["success"]

    def test_invalid_check_items_json(self, auth_client):
        files = [
            _file_data("招标文件内容 " * 50, "a.txt"),
            _file_data("招标文件内容 " * 50, "b.txt"),
        ]
        resp = auth_client.post(
            self.URL,
            data={"files": files, "check_items": "not-json"},
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        if resp.status_code == 409:
            assert data.get("error") == "resource_busy"
        else:
            assert resp.status_code == 200

    def test_no_consent(self, client):
        resp = client.post(self.URL, data={}, content_type="multipart/form-data")
        assert resp.status_code == 403


# ==================== Export Excel Download ====================

class TestExportBatchExcel:
    URL = "/export_batch_docx_download/"

    def test_invalid_token(self, auth_client):
        resp = auth_client.get(self.URL + "invalid-token")
        assert resp.status_code in (410, 404)

    def test_no_consent(self, client):
        resp = client.get(self.URL + "some-token")
        assert resp.status_code == 403


# ==================== Download Batch Result ====================

class TestDownloadBatchResult:
    URL = "/batch_result/"

    def test_nonexistent_task(self, auth_client):
        resp = auth_client.get(self.URL + str(uuid.uuid4()))
        assert resp.status_code == 404

    def test_no_consent(self, client):
        resp = client.get(self.URL + str(uuid.uuid4()))
        assert resp.status_code == 401


# ==================== List Batch Results ====================

class TestListBatchResults:
    URL = "/list_batch_results"

    def test_empty_list(self, auth_client):
        resp = auth_client.get(self.URL)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]
        assert data["results"] == []

    def test_with_db_row(self, auth_client):
        from app.database import get_db_connection
        task_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM users WHERE username = 'testuser'")
                uid = cur.fetchone()[0]
                cur.execute(
                    "INSERT INTO batch_comparison_results "
                    "(user_id, task_id, file_count, pair_count, max_risk, file_names, zip_path) "
                    "VALUES (%s, %s, 2, 1, 0.5, '[]', '/tmp/test.zip')",
                    (uid, task_id),
                )
            conn.commit()
        resp = auth_client.get(self.URL)
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) >= 1

    def test_no_consent(self, client):
        resp = client.get(self.URL)
        assert resp.status_code == 401


# ==================== Delete Batch Result ====================

class TestDeleteBatchResult:
    URL = "/delete_batch_result/"

    def test_admin_delete(self, admin_client):
        from app.database import get_db_connection
        task_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM users WHERE username = 'CEO'")
                uid = cur.fetchone()[0]
                cur.execute(
                    "INSERT INTO batch_comparison_results "
                    "(user_id, task_id, file_count, pair_count, max_risk, file_names, zip_path) "
                    "VALUES (%s, %s, 1, 0, 0, '[]', '/tmp/test.zip') RETURNING id",
                    (uid, task_id),
                )
                row_id = cur.fetchone()[0]
            conn.commit()
        resp = admin_client.post(self.URL + str(row_id))
        assert resp.status_code == 200
        assert resp.get_json()["success"]

    def test_non_admin_forbidden(self, auth_client):
        resp = auth_client.post(self.URL + "1")
        assert resp.status_code == 403

    def test_nonexistent(self, admin_client):
        resp = admin_client.post(self.URL + "999999")
        assert resp.status_code == 404

    def test_no_consent(self, client):
        resp = client.post(self.URL + "1")
        assert resp.status_code == 401


# ==================== Check Quote Anomaly ====================

class TestCheckQuoteAnomaly:
    URL = "/check_quote_anomaly"

    def test_no_file(self, auth_client):
        resp = auth_client.post(self.URL, data={}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_valid_file(self, auth_client, test_bid_file):
        with open(test_bid_file, "rb") as f:
            resp = auth_client.post(
                self.URL,
                data={"file": (f, "bid.txt")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]
        assert "risk_score" in data

    def test_no_consent(self, client):
        resp = client.post(self.URL, data={}, content_type="multipart/form-data")
        assert resp.status_code == 403


# ==================== Compare Bidders Quotes ====================

class TestCompareBiddersQuotes:
    URL = "/compare_bidders_quotes"

    def test_too_few_files(self, auth_client):
        resp = auth_client.post(self.URL, data={}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_two_files(self, auth_client):
        files = [
            _file_data("报价明细A " * 30, "a.txt"),
            _file_data("报价明细B " * 30, "b.txt"),
        ]
        resp = auth_client.post(
            self.URL, data={"files": files}, content_type="multipart/form-data"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]

    def test_no_consent(self, client):
        resp = client.post(self.URL, data={}, content_type="multipart/form-data")
        assert resp.status_code == 403


# ==================== Quote Anomaly Feedback ====================

class TestQuoteAnomalyFeedback:
    URL = "/check_quote_anomaly/feedback"

    def test_missing_fields(self, auth_client):
        resp = auth_client.post(self.URL, json={})
        assert resp.status_code == 400

    def test_invalid_rating(self, auth_client):
        resp = auth_client.post(self.URL, json={"doc_name": "test", "rating": 3})
        assert resp.status_code == 400

    def test_valid_rating_1(self, auth_client):
        resp = auth_client.post(self.URL, json={"doc_name": "test", "rating": 1})
        assert resp.status_code == 200
        assert resp.get_json()["success"]

    def test_valid_rating_5(self, auth_client):
        resp = auth_client.post(self.URL, json={"doc_name": "test", "rating": 5})
        assert resp.status_code == 200

    def test_no_consent(self, client):
        resp = client.post(self.URL, json={"doc_name": "x", "rating": 1})
        assert resp.status_code == 403


# ==================== Extract Relationships ====================

class TestExtractRelationships:
    URL = "/extract_relationships"

    def test_no_files(self, auth_client):
        resp = auth_client.post(self.URL, data={}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_one_file(self, auth_client, test_bid_file):
        with open(test_bid_file, "rb") as f:
            resp = auth_client.post(
                self.URL,
                data={"files": (f, "bid.txt")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]
        assert "entities" in data
        assert "relationships" in data

    def test_too_many_files(self, auth_client):
        files = [_file_data(f"content{i} " * 10, f"f{i}.txt") for i in range(11)]
        resp = auth_client.post(
            self.URL,
            data={"files": files},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_no_consent(self, client):
        resp = client.post(self.URL, data={}, content_type="multipart/form-data")
        assert resp.status_code == 403


# ==================== Check Typos ====================

class TestCheckTypos:
    URL = "/check_typos"

    def test_no_file(self, auth_client):
        resp = auth_client.post(self.URL, data={}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_valid_file(self, auth_client, test_bid_file):
        with open(test_bid_file, "rb") as f:
            resp = auth_client.post(
                self.URL,
                data={"file": (f, "bid.txt")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]
        assert "findings" in data

    def test_diff_mode(self, auth_client, test_bid_file):
        with open(test_bid_file, "rb") as f:
            resp = auth_client.post(
                self.URL,
                data={"file": (f, "bid.txt"), "diff_mode": "true"},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]

    def test_no_consent(self, client):
        resp = client.post(self.URL, data={}, content_type="multipart/form-data")
        assert resp.status_code == 403
