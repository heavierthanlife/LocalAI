"""Tests for compliance blueprint — laws, rules, compliance checks, feedback."""

import json
import os
import uuid
import io
import pytest

pytestmark = [pytest.mark.db, pytest.mark.usefixtures("mock_llm_http")]

LAW_TEXT = (
    "根据《中华人民共和国招标投标法》第三条规定，在中华人民共和国境内进行下列工程建设项目"
    "包括项目的勘察、设计、施工、监理以及与工程建设有关的重要设备、材料等的采购，必须进"
    "行招标：（一）大型基础设施、公用事业等关系社会公共利益、公众安全的项目；（二）全部"
    "或者部分使用国有资金投资或者国家融资的项目；（三）使用国际组织或者外国政府贷款、援"
    "助资金的项目。前款所列项目的具体范围和规模标准，由国务院发展计划部门会同国务院有关"
    "部门制订，报国务院批准。法律或者国务院对必须进行招标的其他项目的范围有规定的，依照"
    "其规定。第五条规定招标投标活动应当遵循公开、公平、公正和诚实信用的原则。"
    "\n\n"
    "投标人须知：投标人须具有独立法人资格。注册资金不低于500万元。投标人须持有"
    "建筑工程施工总承包一级资质证书。投标有效期不少于90天。质保期不少于24个月。"
    "投标保证金不低于投标总价的2%。有下列情形之一的应当废标：未按要求签字盖章。"
    "投标人不得转包、不得串通围标。"
)


def _file_data(content, filename="test.txt"):
    return (io.BytesIO(content.encode("utf-8")), filename)


def _save_rules_result(task_id, rules=None):
    """Create a saved rules result file for testing."""
    from app.config import DATA_DIR
    rules_dir = os.path.join(str(DATA_DIR), "compliance_results")
    os.makedirs(rules_dir, exist_ok=True)
    if rules is None:
        rules = [
            {
                "category": "资质要求",
                "rule": "投标人须具有独立法人资格",
                "severity_if_violated": "critical",
                "keywords": ["独立法人"],
            }
        ]
    path = os.path.join(rules_dir, f"rules_{task_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"rules": rules, "doc_name": "test_bid.pdf"}, f, ensure_ascii=False)


def _save_check_result(task_id):
    """Create a saved compliance check result file for testing."""
    from app.config import DATA_DIR
    results_dir = os.path.join(str(DATA_DIR), "compliance_results")
    os.makedirs(results_dir, exist_ok=True)
    result = {
        "bid_name": "test_bid.pdf",
        "rule_count": 1,
        "summary": {"critical": 1, "violation": 0, "warning": 0, "pass": 0},
        "results": [
            {
                "rule": "投标人须具有独立法人资格",
                "status": "critical",
                "detail": "未找到独立法人声明",
            }
        ],
        "report_html": "<html/>",
        "ai_used": False,
        "checked_at": "2026-07-10T00:00:00",
    }
    path = os.path.join(results_dir, f"{task_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


# ==================== Laws CRUD ====================

class TestLaws:
    LIST_URL = "/compliance/laws"
    UPLOAD_URL = "/compliance/laws/upload"

    def test_list_empty(self, auth_client):
        resp = auth_client.get(self.LIST_URL)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]

    def test_upload_and_list(self, auth_client):
        resp = auth_client.post(
            self.UPLOAD_URL,
            data={"file": _file_data(LAW_TEXT, "law.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200, resp.get_json()
        upload_data = resp.get_json()
        assert upload_data["success"]

    def test_upload_no_file(self, auth_client):
        resp = auth_client.post(
            self.UPLOAD_URL, data={}, content_type="multipart/form-data"
        )
        assert resp.status_code == 400

    def test_delete_valid(self, auth_client):
        resp = auth_client.post(
            self.UPLOAD_URL,
            data={"file": _file_data(LAW_TEXT, "del.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        law_id = resp.get_json().get("law_id")
        assert law_id
        resp = auth_client.delete(self.LIST_URL + f"/{law_id}")
        assert resp.status_code == 200

    def test_delete_nonexistent(self, auth_client):
        resp = auth_client.delete(self.LIST_URL + "/99999")
        assert resp.status_code == 404


# ==================== Extract Rules ====================

class TestExtractRules:
    URL = "/compliance/extract_rules"

    def test_no_file(self, auth_client):
        resp = auth_client.post(self.URL, data={}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_extract_without_ai(self, auth_client, test_bid_file):
        with open(test_bid_file, "rb") as f:
            resp = auth_client.post(
                self.URL,
                data={"file": (f, "bid.txt"), "use_ai": "false"},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["success"]

    def test_extract_with_ai(self, auth_client, test_bid_file):
        with open(test_bid_file, "rb") as f:
            resp = auth_client.post(
                self.URL,
                data={"file": (f, "bid.txt"), "use_ai": "true"},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["success"]


# ==================== Compliance Check ====================

class TestComplianceCheck:
    URL = "/compliance/check"

    def test_missing_rules_task_id(self, auth_client):
        resp = auth_client.post(self.URL, json={})
        assert resp.status_code == 400

    def test_valid_request(self, auth_client):
        rules_task_id = str(uuid.uuid4())
        _save_rules_result(rules_task_id)
        resp = auth_client.post(
            self.URL,
            json={
                "rules_task_id": rules_task_id,
                "bid_file_text": "本次招标文件内容示例" * 50,
                "bid_file_name": "test_bid.pdf",
                "use_ai": False,
                "include_laws": False,
            },
        )
        assert resp.status_code in (200, 202), resp.get_json()
        data = resp.get_json()
        assert data["success"]


# ==================== Rules Get/Update ====================

class TestRulesGetUpdate:
    BASE = "/compliance/rules"

    def test_get_nonexistent(self, auth_client):
        resp = auth_client.get(self.BASE + "/" + str(uuid.uuid4()))
        assert resp.status_code == 404

    def test_get_valid(self, auth_client):
        task_id = str(uuid.uuid4())
        _save_rules_result(task_id)
        resp = auth_client.get(self.BASE + "/" + task_id)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]

    def test_update_valid(self, auth_client):
        task_id = str(uuid.uuid4())
        _save_rules_result(task_id)
        resp = auth_client.put(
            self.BASE + "/" + task_id,
            json={
                "rules": [{"category": "修改后", "rule": "新规则", "severity_if_violated": "warning"}],
                "deleted_ids": [],
                "added_rules": [],
            },
        )
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["success"]


# ==================== Feedback ====================

class TestFeedback:
    URL = "/compliance/feedback"
    HISTORY_URL = "/compliance/feedback/history"

    def test_submit_missing_fields(self, auth_client):
        resp = auth_client.post(self.URL, json={})
        assert resp.status_code == 400

    def test_submit_valid(self, auth_client):
        task_id = str(uuid.uuid4())
        _save_check_result(task_id)
        resp = auth_client.post(
            self.URL,
            json={
                "task_id": task_id,
                "check_file_name": "test.pdf",
                "user_verdict": "true_violation",
                "user_explain": "确实存在违规",
            },
        )
        assert resp.status_code == 200, resp.get_json()
        assert resp.get_json()["success"]

    def test_history_empty(self, auth_client):
        resp = auth_client.get(self.HISTORY_URL)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]

    def test_history_with_data(self, auth_client):
        task_id = str(uuid.uuid4())
        _save_check_result(task_id)
        auth_client.post(
            self.URL,
            json={
                "task_id": task_id,
                "check_file_name": "test.pdf",
                "user_verdict": "false_positive",
            },
        )
        resp = auth_client.get(self.HISTORY_URL)
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data.get("records", [])) >= 1


# ==================== Training Data ====================

class TestTrainingData:
    URL = "/compliance/training_data"

    def test_empty(self, auth_client):
        resp = auth_client.get(self.URL)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"]

    def test_with_data(self, auth_client):
        task_id = str(uuid.uuid4())
        _save_check_result(task_id)
        auth_client.post(
            "/compliance/feedback",
            json={
                "task_id": task_id,
                "check_file_name": "train.pdf",
                "user_verdict": "not_matter",
            },
        )
        resp = auth_client.get(self.URL + "?limit=10")
        assert resp.status_code == 200
