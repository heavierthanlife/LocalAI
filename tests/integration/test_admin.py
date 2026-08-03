"""Integration tests for admin routes."""

import json
import pytest

pytestmark = [pytest.mark.db, pytest.mark.usefixtures("mock_llm_http")]


class TestAdminUsers:
    LIST_ROUTE = "/admin/users"

    def test_requires_consent(self, admin_client):
        with admin_client.session_transaction() as sess:
            sess["consent_value"] = 0
        resp = admin_client.get(self.LIST_ROUTE)
        assert resp.status_code == 403

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.LIST_ROUTE)
        assert resp.status_code == 403

    def test_returns_user_list(self, admin_client):
        resp = admin_client.get(self.LIST_ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["users"], list)
        usernames = [u["username"] for u in data["users"]]
        assert "CEO" in usernames
        assert "COO" in usernames

    def test_includes_normal_user(self, admin_client):
        from app.database import get_db_connection
        import uuid
        uid = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (user_id, username, role) VALUES (%s, %s, 'user')",
                    (uid, "normaluser")
                )
            conn.commit()
        resp = admin_client.get(self.LIST_ROUTE)
        usernames = [u["username"] for u in resp.get_json()["users"]]
        assert "normaluser" in usernames


class TestAdminAuditLog:
    LOG_ROUTE = "/admin/audit_log"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.LOG_ROUTE)
        assert resp.status_code == 403

    def test_returns_empty_log(self, admin_client):
        resp = admin_client.get(self.LOG_ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["logs"], list)

    def test_paginates(self, admin_client):
        resp = admin_client.get(self.LOG_ROUTE + "?page=1")
        data = resp.get_json()
        assert "total" in data
        assert "page" in data

    def test_filters_by_action(self, admin_client):
        resp = admin_client.get(self.LOG_ROUTE + "?action=UPDATE")
        assert resp.status_code == 200


class TestAdminAnalytics:
    ROUTE = "/admin/analytics"

    def test_requires_login(self, auth_client):
        with auth_client.session_transaction() as sess:
            sess.pop("user_id", None)
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 401

    def test_admin_sees_full_stats(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["is_admin_view"] is True

    def test_user_sees_own_stats(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["is_admin_view"] is False


class TestAdminRuntimeConfig:
    ROUTE = "/admin/runtime_config"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_get_config(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["config"], dict)

    def test_update_config(self, admin_client):
        resp = admin_client.post(self.ROUTE, json={"test_key": "test_value"})
        assert resp.status_code == 200

    def test_empty_update_rejected(self, admin_client):
        resp = admin_client.post(self.ROUTE, json={})
        assert resp.status_code == 400


class TestAdminSystemPrompt:
    ROUTE = "/admin/system_prompt"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_get_prompt(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert "prompt" in data

    def test_update_prompt(self, admin_client):
        resp = admin_client.post(self.ROUTE, json={"prompt": "Test prompt"})
        assert resp.status_code == 200

    def test_empty_prompt_rejected(self, admin_client):
        resp = admin_client.post(self.ROUTE, json={"prompt": ""})
        assert resp.status_code == 400


class TestAdminUserAssets:
    ROUTE = "/admin/user_assets"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_users_with_assets(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["users"], list)


class TestAdminLlmProviders:
    ROUTE = "/admin/llm_providers"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_providers(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert "providers" in data
        assert "active_provider" in data


class TestAdminDbTables:
    ROUTE = "/admin/db_tables"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_table_list(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data.get("tables") or data.get("data"), list)


class TestAdminArchivedSessions:
    ROUTE = "/admin/archived_sessions"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_empty_initially(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True


class TestAdminDbMigrations:
    ROUTE = "/admin/db_migrations"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_migration_list(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True


class TestAdminSendMail:
    ROUTE = "/admin/send_mail"

    def test_admin_required(self, auth_client):
        resp = auth_client.post(self.ROUTE, json={})
        assert resp.status_code == 403

    def test_requires_recipient(self, admin_client):
        resp = admin_client.post(self.ROUTE, json={"subject": "Test", "body": "Body"})
        assert resp.status_code == 400


class TestAdminEmbeddingCache:
    ROUTE = "/admin/embedding_cache"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_get_cache(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True


class TestAdminDbTablesOverview:
    ROUTE = "/admin/db_tables_overview"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_overview(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["tables"], list)
        assert len(data["tables"]) > 0
        table_names = [t["table_name"] for t in data["tables"]]
        assert "users" in table_names


class TestAdminDbSchema:
    ROUTE = "/admin/db_schema"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE + "?table=users")
        assert resp.status_code == 403

    def test_requires_table_param(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        assert resp.status_code == 400

    def test_invalid_table(self, admin_client):
        resp = admin_client.get(self.ROUTE + "?table=nonexistent_table_xyz")
        assert resp.status_code == 400

    def test_returns_columns(self, admin_client):
        resp = admin_client.get(self.ROUTE + "?table=users")
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["columns"], list)
        assert len(data["columns"]) > 0
        col_names = [c["column_name"] for c in data["columns"]]
        assert "user_id" in col_names
        assert "username" in col_names


class TestAdminVlStatus:
    ROUTE = "/admin/vl_status"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_status(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert "model" in data
        assert "provider" in data
        assert "available" in data


class TestAdminNotifications:
    ROUTE = "/admin/notifications"
    MARK_ROUTE = "/admin/notifications/mark_read"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_mark_read_admin_required(self, auth_client):
        resp = auth_client.post(self.MARK_ROUTE)
        assert resp.status_code == 403

    def test_returns_empty_notifications(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["notifications"], list)
        assert data["total"] == 0

    def test_mark_read_returns_ok(self, admin_client):
        resp = admin_client.post(self.MARK_ROUTE, json={})
        data = resp.get_json()
        assert data["success"] is True


class TestAdminUserEmails:
    ROUTE = "/admin/user_emails"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_active_users(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["users"], list)
        usernames = [u["username"] for u in data["users"]]
        assert "CEO" in usernames
        assert "COO" in usernames


class TestAdminSearchCacheConfig:
    ROUTE = "/admin/search_cache_config"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_config(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert "config" in data


class TestAdminQuoteAnomalyResults:
    ROUTE = "/admin/quote_anomaly_results"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_empty_list(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["results"], list)
        assert data["total"] == 0


class TestAdminTypoResults:
    ROUTE = "/admin/typo_results"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_empty_list(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["results"], list)
        assert data["total"] == 0


class TestAdminRelationshipResults:
    ROUTE = "/admin/relationship_results"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_empty_list(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["results"], list)
        assert data["total"] == 0


class TestAdminPendingDeletions:
    ROUTE = "/admin/pending_deletions"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_empty_initially(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["users"], list)


class TestAdminClearFileCache:
    ROUTE = "/admin/clear_file_cache"

    def test_admin_required(self, auth_client):
        resp = auth_client.post(self.ROUTE, json={})
        assert resp.status_code == 403

    def test_clears_cache(self, admin_client):
        resp = admin_client.post(self.ROUTE, json={})
        data = resp.get_json()
        assert data["success"] is True


class TestAdminClearEmbeddingCache:
    ROUTE = "/admin/embedding_cache/clear"

    def test_admin_required(self, auth_client):
        resp = auth_client.post(self.ROUTE, json={})
        assert resp.status_code == 403

    def test_clears_embedding_cache(self, admin_client):
        resp = admin_client.post(self.ROUTE, json={})
        data = resp.get_json()
        assert data["success"] is True


class TestAdminSystemCleanup:
    ROUTE = "/admin/system_cleanup"

    def test_admin_required(self, auth_client):
        resp = auth_client.post(self.ROUTE, json={})
        assert resp.status_code == 403

    def test_runs_cleanup(self, admin_client):
        resp = admin_client.post(self.ROUTE, json={})
        data = resp.get_json()
        assert data["success"] is True
        assert "results" in data


class TestAdminProjects:
    LIST_ROUTE = "/admin/projects"
    MEMBER_TPL = "/admin/projects/%d/members"

    def test_admin_required(self, auth_client):
        resp = auth_client.post(self.LIST_ROUTE, json={})
        assert resp.status_code == 403

    def test_create_project_requires_name(self, admin_client):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": ""})
        assert resp.status_code == 400

    def test_create_and_list_project(self, admin_client):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": "测试项目", "description": "集成测试用"})
        data = resp.get_json()
        assert data["success"] is True
        assert "id" in data
        project_id = data["id"]

        resp = admin_client.get(self.LIST_ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert data["has_projects"] is True
        names = [p["name"] for p in data["projects"]]
        assert "测试项目" in names

    def test_list_members_admin_required(self, auth_client):
        resp = auth_client.get(self.MEMBER_TPL % 1)
        assert resp.status_code == 403

    def test_list_members(self, admin_client):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": "成员测试项目"})
        pid = resp.get_json()["id"]
        resp = admin_client.get(self.MEMBER_TPL % pid)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["members"], list)
        assert len(data["members"]) == 1
        assert data["members"][0]["role"] == "admin"


class TestAdminTaskDeposit:
    ROUTE = "/admin/task_deposit"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_returns_empty_list(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["items"], list)


class TestAdminDbData:
    ROUTE = "/admin/db_data"

    def test_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE)
        assert resp.status_code == 403

    def test_requires_table(self, admin_client):
        resp = admin_client.get(self.ROUTE)
        assert resp.status_code == 400

    def test_invalid_table(self, admin_client):
        resp = admin_client.get(self.ROUTE + "?table=nonexistent")
        assert resp.status_code == 400

    def test_queries_users_table(self, admin_client):
        resp = admin_client.get(self.ROUTE + "?table=users")
        data = resp.get_json()
        assert data["success"] is True


class TestAdminProjectUpdateDelete:
    ROUTE_TPL = "/admin/projects/%d"
    LIST_ROUTE = "/admin/projects"

    def _create_project(self, admin_client, name="项目更新测试"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name, "description": "test"})
        return resp.get_json()["id"]

    def test_update_requires_name(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.put(self.ROUTE_TPL % pid, json={"name": ""})
        assert resp.status_code == 400

    def test_update_project(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.put(self.ROUTE_TPL % pid, json={"name": "新名称", "description": "新描述"})
        data = resp.get_json()
        assert data["success"] is True
        resp = admin_client.get(self.LIST_ROUTE)
        names = [p["name"] for p in resp.get_json()["projects"]]
        assert "新名称" in names

    def test_update_nonexistent(self, admin_client):
        resp = admin_client.put(self.ROUTE_TPL % 99999, json={"name": "不存在"})
        assert resp.status_code == 404

    def test_delete_active_project_fails(self, admin_client):
        pid = self._create_project(admin_client, "活跃项目")
        resp = admin_client.delete(self.ROUTE_TPL % pid)
        assert resp.status_code == 400

    def test_delete_nonexistent_project(self, admin_client):
        resp = admin_client.delete(self.ROUTE_TPL % 99999)
        assert resp.status_code == 404


class TestAdminProjectMembers:
    LIST_ROUTE = "/admin/projects"
    MEMBER_TPL = "/admin/projects/%d/members"
    SEARCH_TPL = "/admin/projects/%d/members/search"
    ALL_USERS_TPL = "/admin/projects/%d/all_users"

    def _create_project(self, admin_client, name="成员测试项目"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        return resp.get_json()["id"]

    def test_search_users_requires_query(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.SEARCH_TPL % pid)
        data = resp.get_json()
        assert data["success"] is True
        assert data["users"] == []

    def test_search_users_short_query(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.SEARCH_TPL % pid + "?q=a")
        data = resp.get_json()
        assert data["users"] == []

    def test_all_users_returns_non_members(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.ALL_USERS_TPL % pid)
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["users"], list)

    def test_add_member_invalid_user_id(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.MEMBER_TPL % pid, json={"user_id": "nonexistent-uuid"})
        assert resp.status_code == 404

    def test_add_member_invalid_role(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.MEMBER_TPL % pid, json={"user_id": "x", "role": "invalid"})
        assert resp.status_code == 400


class TestAdminProjectFolders:
    LIST_ROUTE = "/admin/projects"
    FOLDER_TPL = "/admin/projects/%d/folders"

    def _create_project(self, admin_client, name="文件夹测试项目"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        return resp.get_json()["id"]

    def test_list_folders_returns_root(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.FOLDER_TPL % pid)
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["folders"]) == 1
        assert data["folders"][0]["name"] == "文件夹测试项目"
        root_id = data["folders"][0]["id"]

    def test_create_subfolder(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        resp = admin_client.post(self.FOLDER_TPL % pid, json={"name": "子文件夹", "parent_folder_id": root_id})
        data = resp.get_json()
        assert data["success"] is True
        assert "id" in data
        resp = admin_client.get(self.FOLDER_TPL % pid)
        assert len(resp.get_json()["folders"]) == 1
        assert len(resp.get_json()["folders"][0]["children"]) == 1

    def test_create_folder_requires_name(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        resp = admin_client.post(self.FOLDER_TPL % pid, json={"name": "", "parent_folder_id": root_id})
        assert resp.status_code == 400

    def test_create_folder_invalid_parent(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.FOLDER_TPL % pid, json={"name": "子文件夹", "parent_folder_id": 99999})
        assert resp.status_code == 404


class TestAdminProjectFileOperations:
    LIST_ROUTE = "/admin/projects"
    FOLDER_TPL = "/admin/projects/%d/folders"
    UPLOAD_TPL = "/admin/projects/%d/folders/%d/upload"
    FILES_TPL = "/admin/projects/%d/folders/%d/files"
    CONTENT_TPL = "/admin/projects/%d/files/%d/content"
    RENAME_TPL = "/admin/projects/%d/files/%d/rename"

    def _create_project_and_folder(self, admin_client, name="文件测试项目"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        pid = resp.get_json()["id"]
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        return pid, root_id

    def test_upload_requires_file(self, admin_client):
        pid, root_id = self._create_project_and_folder(admin_client)
        resp = admin_client.post(self.UPLOAD_TPL % (pid, root_id), data={}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_upload_and_list_file(self, admin_client, test_bid_file):
        pid, root_id = self._create_project_and_folder(admin_client)
        with open(test_bid_file, "rb") as f:
            resp = admin_client.post(
                self.UPLOAD_TPL % (pid, root_id),
                data={"file": (f, "投标书.txt")},
                content_type="multipart/form-data",
            )
        data = resp.get_json()
        assert data["success"] is True
        assert "file_id" in data
        file_id = data["file_id"]

        resp = admin_client.get(self.FILES_TPL % (pid, root_id))
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["files"]) == 1
        assert data["files"][0]["original_name"] == "投标书.txt"

    def test_file_content(self, admin_client, test_bid_file):
        pid, root_id = self._create_project_and_folder(admin_client)
        with open(test_bid_file, "rb") as f:
            resp = admin_client.post(
                self.UPLOAD_TPL % (pid, root_id),
                data={"file": (f, "投标书.txt")},
                content_type="multipart/form-data",
            )
        file_id = resp.get_json()["file_id"]
        resp = admin_client.get(self.CONTENT_TPL % (pid, file_id))
        data = resp.get_json()
        assert data["success"] is True
        assert "text" in data
        assert len(data["text"]) > 0

    def test_file_content_nonexistent(self, admin_client):
        resp = admin_client.get(self.CONTENT_TPL % (1, 99999))
        assert resp.status_code == 404

    def test_rename_file(self, admin_client, test_bid_file):
        pid, root_id = self._create_project_and_folder(admin_client)
        with open(test_bid_file, "rb") as f:
            resp = admin_client.post(
                self.UPLOAD_TPL % (pid, root_id),
                data={"file": (f, "投标书.txt")},
                content_type="multipart/form-data",
            )
        file_id = resp.get_json()["file_id"]
        resp = admin_client.put(self.RENAME_TPL % (pid, file_id), json={"original_name": "新名称.txt"})
        data = resp.get_json()
        assert data["success"] is True

    def test_rename_file_requires_name(self, admin_client, test_bid_file):
        pid, root_id = self._create_project_and_folder(admin_client)
        with open(test_bid_file, "rb") as f:
            resp = admin_client.post(
                self.UPLOAD_TPL % (pid, root_id),
                data={"file": (f, "投标书.txt")},
                content_type="multipart/form-data",
            )
        file_id = resp.get_json()["file_id"]
        resp = admin_client.put(self.RENAME_TPL % (pid, file_id), json={"original_name": ""})
        assert resp.status_code == 400

    def test_rename_nonexistent_file(self, admin_client):
        resp = admin_client.put(self.RENAME_TPL % (1, 99999), json={"original_name": "新名称.txt"})
        assert resp.status_code == 404


class TestAdminProjectTodos:
    LIST_ROUTE = "/admin/projects"
    TODOS_TPL = "/admin/projects/%d/todos"
    DONE_TPL = "/admin/projects/%d/todos/%d/done"
    REMOVE_TPL = "/admin/projects/%d/todos/%d/remove"

    def _create_project(self, admin_client, name="待办测试项目"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        return resp.get_json()["id"]

    def test_todos_list_empty(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.TODOS_TPL % pid)
        data = resp.get_json()
        assert data["success"] is True
        assert data["todos"] == []

    def test_add_todo_requires_content(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.TODOS_TPL % pid, json={"content_copy": ""})
        assert resp.status_code == 400

    def test_add_and_list_todo(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.TODOS_TPL % pid, json={"content_copy": "需要完成的事项"})
        data = resp.get_json()
        assert data["success"] is True
        assert "todo_id" in data
        resp = admin_client.get(self.TODOS_TPL % pid)
        assert len(resp.get_json()["todos"]) == 1

    def test_mark_todo_done(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.TODOS_TPL % pid, json={"content_copy": "已完成的事项"})
        todo_id = resp.get_json()["todo_id"]
        resp = admin_client.post(self.DONE_TPL % (pid, todo_id))
        data = resp.get_json()
        assert data["success"] is True
        resp = admin_client.get(self.TODOS_TPL % pid)
        assert len(resp.get_json()["todos"]) == 0

    def test_remove_todo(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.TODOS_TPL % pid, json={"content_copy": "待删除的事项"})
        todo_id = resp.get_json()["todo_id"]
        resp = admin_client.post(self.REMOVE_TPL % (pid, todo_id))
        data = resp.get_json()
        assert data["success"] is True
        resp = admin_client.get(self.TODOS_TPL % pid)
        assert len(resp.get_json()["todos"]) == 0


class TestAdminProjectPresence:
    PING_TPL = "/admin/projects/%d/ping"
    PRESENCE_TPL = "/admin/projects/%d/presence"

    def _create_project(self, admin_client, name="在线测试项目"):
        resp = admin_client.post("/admin/projects", json={"name": name})
        return resp.get_json()["id"]

    def test_ping_updates_presence(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.PING_TPL % pid)
        assert resp.status_code == 200

    def test_presence_returns_active(self, auth_client):
        resp = auth_client.get(self.PRESENCE_TPL % 1)
        data = resp.get_json()
        assert data["success"] is True
        assert "active_users" in data


class TestAdminProjectLifecycle:
    LIST_ROUTE = "/admin/projects"
    ROUTE_TPL = "/admin/projects/%d"

    def _create_project(self, admin_client, name="生命周期测试项目"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        return resp.get_json()["id"]

    def test_abort_project(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.ROUTE_TPL % pid + "/abort")
        data = resp.get_json()
        assert data["success"] is True

    def test_abort_nonexistent(self, admin_client):
        resp = admin_client.post(self.ROUTE_TPL % 99999 + "/abort")
        assert resp.status_code == 404

    def test_finish_project_requires_files(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.ROUTE_TPL % pid + "/finish")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "No files" in data.get("error", "")

    def test_finish_nonexistent(self, admin_client):
        resp = admin_client.post(self.ROUTE_TPL % 99999 + "/finish")
        assert resp.status_code == 404

    def test_finish_and_download_archive(self, admin_client, test_bid_file):
        pid = self._create_project(admin_client, "打包测试")
        resp = admin_client.get(self.LIST_ROUTE + "/%d/folders" % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        with open(test_bid_file, "rb") as f:
            resp = admin_client.post(
                self.LIST_ROUTE + "/%d/folders/%d/upload" % (pid, root_id),
                data={"file": (f, "投标书.txt")},
                content_type="multipart/form-data",
            )
        resp = admin_client.post(self.ROUTE_TPL % pid + "/finish")
        data = resp.get_json()
        assert data["success"] is True
        assert "download_url" in data
        assert "zip_filename" in data
        resp = admin_client.get(data["download_url"])
        assert resp.status_code == 200

    def test_download_archive_not_found(self, admin_client):
        resp = admin_client.get(self.ROUTE_TPL % 1 + "/download_archive/nonexistent.zip")
        assert resp.status_code == 404

    def test_admin_required_abort(self, auth_client):
        resp = auth_client.post(self.ROUTE_TPL % 1 + "/abort")
        assert resp.status_code == 403

    def test_admin_required_finish(self, auth_client):
        resp = auth_client.post(self.ROUTE_TPL % 1 + "/finish")
        assert resp.status_code == 403


class TestAdminProjectMembersManage:
    LIST_ROUTE = "/admin/projects"
    MEMBER_TPL = "/admin/projects/%d/members"

    def _create_project(self, admin_client, name="成员管理测试"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        return resp.get_json()["id"]

    def test_update_member_role(self, admin_client):
        pid = self._create_project(admin_client)
        from app.database import get_db_connection
        import uuid
        uid = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO users (user_id, username, role) VALUES (%s, %s, 'user')",
                            (uid, "memberuser"))
                cur.execute("INSERT INTO project_members (project_id, user_id, role) VALUES (%s, %s, 'member')",
                            (pid, uid))
            conn.commit()
        resp = admin_client.put(self.MEMBER_TPL % pid + "/" + uid, json={"role": "manager"})
        data = resp.get_json()
        assert data["success"] is True
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT role FROM project_members WHERE project_id = %s AND user_id = %s", (pid, uid))
                assert cur.fetchone()[0] == "manager"

    def test_update_member_invalid_role(self, admin_client):
        pid = self._create_project(admin_client)
        from app.database import get_db_connection
        import uuid
        uid = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO users (user_id, username, role) VALUES (%s, %s, 'user')",
                            (uid, "invalidrole"))
                cur.execute("INSERT INTO project_members (project_id, user_id, role) VALUES (%s, %s, 'member')",
                            (pid, uid))
            conn.commit()
        resp = admin_client.put(self.MEMBER_TPL % pid + "/" + uid, json={"role": "superadmin"})
        assert resp.status_code == 400

    def test_update_member_nonexistent(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.put(self.MEMBER_TPL % pid + "/nonexistent-uuid", json={"role": "manager"})
        assert resp.status_code == 404

    def test_update_member_admin_role_denied(self, admin_client):
        pid = self._create_project(admin_client)
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id FROM project_members
                    WHERE project_id = %s AND role = 'admin' LIMIT 1
                """, (pid,))
                admin_member_id = cur.fetchone()[0]
        resp = admin_client.put(self.MEMBER_TPL % pid + "/" + admin_member_id, json={"role": "manager"})
        assert resp.status_code == 403

    def test_remove_member(self, admin_client):
        pid = self._create_project(admin_client)
        from app.database import get_db_connection
        import uuid
        uid = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO users (user_id, username, role) VALUES (%s, %s, 'user')",
                            (uid, "removeme"))
                cur.execute("INSERT INTO project_members (project_id, user_id, role) VALUES (%s, %s, 'member')",
                            (pid, uid))
            conn.commit()
        resp = admin_client.delete(self.MEMBER_TPL % pid + "/" + uid)
        data = resp.get_json()
        assert data["success"] is True
        assert data.get("quitted") is True

    def test_remove_member_nonexistent(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.delete(self.MEMBER_TPL % pid + "/nonexistent-uuid")
        assert resp.status_code == 404

    def test_remove_member_requires_manage(self, auth_client):
        pid = None
        from app.database import get_db_connection
        import uuid
        uid = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM users WHERE role = 'admin' LIMIT 1")
                admin_id = cur.fetchone()[0]
                cur.execute("INSERT INTO users (user_id, username, role) VALUES (%s, %s, 'user')",
                            (uid, "notallowed"))
                cur.execute("INSERT INTO projects (name) VALUES (%s) RETURNING id", ("权限测试",))
                pid = cur.fetchone()[0]
                cur.execute("INSERT INTO project_members (project_id, user_id, role) VALUES (%s, %s, 'admin')",
                            (pid, admin_id))
                cur.execute("INSERT INTO project_members (project_id, user_id, role) VALUES (%s, %s, 'member')",
                            (pid, uid))
            conn.commit()
        resp = auth_client.delete(self.MEMBER_TPL % pid + "/" + uid)
        assert resp.status_code == 403


class TestAdminProjectFoldersManage:
    LIST_ROUTE = "/admin/projects"
    FOLDER_TPL = "/admin/projects/%d/folders"

    def _create_project(self, admin_client, name="文件夹管理测试"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        return resp.get_json()["id"]

    def test_rename_folder(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        resp = admin_client.put(self.FOLDER_TPL % pid + "/%d/rename" % root_id, json={"name": "新根目录"})
        assert resp.status_code == 200
        resp = admin_client.get(self.FOLDER_TPL % pid)
        assert resp.get_json()["folders"][0]["name"] == "新根目录"

    def test_rename_folder_requires_name(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        resp = admin_client.put(self.FOLDER_TPL % pid + "/%d/rename" % root_id, json={"name": ""})
        assert resp.status_code == 400

    def test_rename_folder_nonexistent(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.put(self.FOLDER_TPL % pid + "/99999/rename", json={"name": "不存在"})
        assert resp.status_code == 404

    def test_delete_folder(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        resp = admin_client.post(self.FOLDER_TPL % pid, json={"name": "待删除文件夹", "parent_folder_id": root_id})
        sub_id = resp.get_json()["id"]
        resp = admin_client.delete(self.FOLDER_TPL % pid + "/%d" % sub_id)
        data = resp.get_json()
        assert data["success"] is True

    def test_delete_root_folder(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        resp = admin_client.delete(self.FOLDER_TPL % pid + "/%d" % root_id)
        data = resp.get_json()
        assert data["success"] is True
        assert data["folders_moved"] == 1

    def test_delete_folder_deletes_files(self, admin_client, test_bid_file):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        resp = admin_client.post(self.FOLDER_TPL % pid, json={"name": "有文件子目录", "parent_folder_id": root_id})
        sub_id = resp.get_json()["id"]
        with open(test_bid_file, "rb") as f:
            admin_client.post(
                "/admin/projects/%d/folders/%d/upload" % (pid, sub_id),
                data={"file": (f, "投标书.txt")},
                content_type="multipart/form-data",
            )
        resp = admin_client.delete(self.FOLDER_TPL % pid + "/%d" % sub_id)
        data = resp.get_json()
        assert data["success"] is True
        assert data["files_moved"] >= 1

    def test_folder_comments_empty(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        resp = admin_client.get(self.FOLDER_TPL % pid + "/%d/comments" % root_id)
        data = resp.get_json()
        assert data["success"] is True
        assert data["comments"] == []

    def test_add_folder_comment(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        resp = admin_client.post(self.FOLDER_TPL % pid + "/%d/comments" % root_id, json={"comment": "讨论文件夹内容"})
        data = resp.get_json()
        assert data["success"] is True
        resp = admin_client.get(self.FOLDER_TPL % pid + "/%d/comments" % root_id)
        assert len(resp.get_json()["comments"]) == 1
        assert resp.get_json()["comments"][0]["comment"] == "讨论文件夹内容"

    def test_add_folder_comment_requires_text(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        resp = admin_client.post(self.FOLDER_TPL % pid + "/%d/comments" % root_id, json={"comment": ""})
        assert resp.status_code == 400

    def test_folder_admin_required(self, auth_client):
        pid = 1
        resp = auth_client.delete("/admin/projects/%d/folders/1" % pid)
        assert resp.status_code == 403
        resp = auth_client.put("/admin/projects/%d/folders/1/rename" % pid, json={"name": "x"})
        assert resp.status_code == 403


class TestAdminProjectFileVersionsDownload:
    LIST_ROUTE = "/admin/projects"
    FOLDER_TPL = "/admin/projects/%d/folders"
    UPLOAD_TPL = "/admin/projects/%d/folders/%d/upload"
    VERSIONS_TPL = "/admin/projects/%d/files/%d/versions"
    DOWNLOAD_TPL = "/admin/projects/%d/files/%d/download"
    NEW_VERSION_TPL = "/admin/projects/%d/files/%d/new_version"

    def _create_project_and_upload(self, admin_client, test_bid_file, name="文件版本测试"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        pid = resp.get_json()["id"]
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        with open(test_bid_file, "rb") as f:
            resp = admin_client.post(
                self.UPLOAD_TPL % (pid, root_id),
                data={"file": (f, "投标书.txt")},
                content_type="multipart/form-data",
            )
        return pid, root_id, resp.get_json()["file_id"]

    def test_versions_returns_empty(self, admin_client, test_bid_file):
        pid, _, file_id = self._create_project_and_upload(admin_client, test_bid_file)
        resp = admin_client.get(self.VERSIONS_TPL % (pid, file_id))
        data = resp.get_json()
        assert data["success"] is True
        assert data["versions"] == []

    def test_new_version(self, admin_client, test_bid_file):
        pid, root_id, file_id = self._create_project_and_upload(admin_client, test_bid_file)
        with open(test_bid_file, "rb") as f:
            resp = admin_client.post(
                self.NEW_VERSION_TPL % (pid, file_id),
                data={"file": (f, "投标书_v2.txt")},
                content_type="multipart/form-data",
            )
        data = resp.get_json()
        assert data["success"] is True
        assert data["version"] == 2
        resp = admin_client.get(self.VERSIONS_TPL % (pid, file_id))
        assert len(resp.get_json()["versions"]) == 1

    def test_new_version_requires_file(self, admin_client):
        resp = admin_client.post(self.NEW_VERSION_TPL % (1, 1), data={}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_new_version_nonexistent_file(self, admin_client):
        from io import BytesIO
        data = {"file": (BytesIO(b"test content"), "test.txt")}
        resp = admin_client.post(self.NEW_VERSION_TPL % (1, 99999), data=data, content_type="multipart/form-data")
        assert resp.status_code == 404

    def test_download_file(self, admin_client, test_bid_file):
        pid, _, file_id = self._create_project_and_upload(admin_client, test_bid_file)
        resp = admin_client.get(self.DOWNLOAD_TPL % (pid, file_id))
        assert resp.status_code == 200
        assert resp.mimetype == 'application/octet-stream' or 'text/plain' in resp.mimetype

    def test_download_nonexistent(self, admin_client):
        resp = admin_client.get(self.DOWNLOAD_TPL % (1, 99999))
        assert resp.status_code == 404

    def test_download_file_requires_consent(self, admin_client):
        with admin_client.session_transaction() as sess:
            sess["consent_value"] = 0
        resp = admin_client.get(self.DOWNLOAD_TPL % (1, 1))
        assert resp.status_code == 403

    def test_list_root_files_empty(self, admin_client, test_bid_file):
        pid, root_id, file_id = self._create_project_and_upload(admin_client, test_bid_file)
        resp = admin_client.get("/admin/projects/%d/files" % pid)
        data = resp.get_json()
        assert data["success"] is True
        assert data["files"] == []


class TestAdminProjectFileComments:
    LIST_ROUTE = "/admin/projects"
    FOLDER_TPL = "/admin/projects/%d/folders"
    UPLOAD_TPL = "/admin/projects/%d/folders/%d/upload"
    COMMENTS_TPL = "/admin/projects/%d/files/%d/comments"

    def _create_project_and_upload(self, admin_client, test_bid_file, name="文件评论测试"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        pid = resp.get_json()["id"]
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        with open(test_bid_file, "rb") as f:
            resp = admin_client.post(
                self.UPLOAD_TPL % (pid, root_id),
                data={"file": (f, "投标书.txt")},
                content_type="multipart/form-data",
            )
        return pid, resp.get_json()["file_id"]

    def test_file_comments_empty(self, admin_client, test_bid_file):
        pid, file_id = self._create_project_and_upload(admin_client, test_bid_file)
        resp = admin_client.get(self.COMMENTS_TPL % (pid, file_id))
        data = resp.get_json()
        assert data["success"] is True
        assert data["comments"] == []

    def test_add_file_comment(self, admin_client, test_bid_file):
        pid, file_id = self._create_project_and_upload(admin_client, test_bid_file)
        resp = admin_client.post(self.COMMENTS_TPL % (pid, file_id), json={"comment": "这份文件需要修改"})
        assert resp.status_code == 200
        resp = admin_client.get(self.COMMENTS_TPL % (pid, file_id))
        assert len(resp.get_json()["comments"]) == 1
        assert resp.get_json()["comments"][0]["comment"] == "这份文件需要修改"

    def test_add_file_comment_requires_text(self, admin_client, test_bid_file):
        pid, file_id = self._create_project_and_upload(admin_client, test_bid_file)
        resp = admin_client.post(self.COMMENTS_TPL % (pid, file_id), json={"comment": ""})
        assert resp.status_code == 400


class TestAdminProjectFileMoveSearchBatch:
    LIST_ROUTE = "/admin/projects"
    FOLDER_TPL = "/admin/projects/%d/folders"
    UPLOAD_TPL = "/admin/projects/%d/folders/%d/upload"
    MOVE_TPL = "/admin/projects/%d/files/%d/move"
    SEARCH_TPL = "/admin/projects/%d/files/search"
    BATCH_MOVE_TPL = "/admin/projects/%d/files/batch_move"
    BATCH_DOWNLOAD_TPL = "/admin/projects/%d/batch_download"
    DELETE_TPL = "/admin/projects/%d/files/%d"

    def _create_project_with_files(self, admin_client, test_bid_file, name="文件移动搜索测试"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        pid = resp.get_json()["id"]
        resp = admin_client.get(self.FOLDER_TPL % pid)
        root_id = resp.get_json()["folders"][0]["id"]
        resp = admin_client.post(self.FOLDER_TPL % pid, json={"name": "子目录", "parent_folder_id": root_id})
        sub_id = resp.get_json()["id"]
        with open(test_bid_file, "rb") as f:
            resp = admin_client.post(
                self.UPLOAD_TPL % (pid, root_id),
                data={"file": (f, "投标书.txt")},
                content_type="multipart/form-data",
            )
        file_id = resp.get_json()["file_id"]
        return pid, root_id, sub_id, file_id

    def test_move_file_to_folder(self, admin_client, test_bid_file):
        pid, root_id, sub_id, file_id = self._create_project_with_files(admin_client, test_bid_file)
        resp = admin_client.post(self.MOVE_TPL % (pid, file_id), json={"folder_id": sub_id})
        assert resp.status_code == 200
        resp = admin_client.get(self.FOLDER_TPL % pid + "/%d/files" % sub_id)
        assert len(resp.get_json()["files"]) == 1

    def test_move_file_to_root(self, admin_client, test_bid_file):
        pid, root_id, sub_id, file_id = self._create_project_with_files(admin_client, test_bid_file)
        resp = admin_client.post(self.MOVE_TPL % (pid, file_id), json={"folder_id": root_id})
        assert resp.status_code == 200

    def test_move_file_invalid_folder(self, admin_client, test_bid_file):
        pid, _, _, file_id = self._create_project_with_files(admin_client, test_bid_file)
        resp = admin_client.post(self.MOVE_TPL % (pid, file_id), json={"folder_id": 99999})
        assert resp.status_code == 404

    def test_move_file_nonexistent(self, admin_client):
        resp = admin_client.post(self.MOVE_TPL % (1, 99999), json={"folder_id": 1})
        assert resp.status_code == 404

    def test_search_files_short_query(self, admin_client, test_bid_file):
        pid, _, _, _ = self._create_project_with_files(admin_client, test_bid_file)
        resp = admin_client.get(self.SEARCH_TPL % pid + "?q=a")
        data = resp.get_json()
        assert data["files"] == []

    def test_search_files_finds_result(self, admin_client, test_bid_file):
        pid, _, _, _ = self._create_project_with_files(admin_client, test_bid_file)
        resp = admin_client.get(self.SEARCH_TPL % pid + "?q=投标")
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["files"]) == 1
        assert data["files"][0]["original_name"] == "投标书.txt"

    def test_batch_move_files(self, admin_client, test_bid_file):
        pid, root_id, sub_id, file_id = self._create_project_with_files(admin_client, test_bid_file)
        resp = admin_client.post(self.BATCH_MOVE_TPL % pid, json={"file_ids": [file_id], "folder_id": sub_id})
        data = resp.get_json()
        assert data["success"] is True
        assert data["moved_count"] == 1

    def test_batch_move_no_files(self, admin_client):
        pid = 1
        resp = admin_client.post(self.BATCH_MOVE_TPL % pid, json={"file_ids": [], "folder_id": 1})
        assert resp.status_code == 400

    def test_batch_move_no_folder(self, admin_client, test_bid_file):
        pid, _, _, file_id = self._create_project_with_files(admin_client, test_bid_file)
        resp = admin_client.post(self.BATCH_MOVE_TPL % pid, json={"file_ids": [file_id]})
        assert resp.status_code == 400

    def test_batch_download_empty_selection(self, admin_client):
        pid = 1
        resp = admin_client.post(self.BATCH_DOWNLOAD_TPL % pid, json={"file_ids": []})
        assert resp.status_code == 400

    def test_batch_download(self, admin_client, test_bid_file):
        pid, _, _, file_id = self._create_project_with_files(admin_client, test_bid_file)
        resp = admin_client.post(self.BATCH_DOWNLOAD_TPL % pid, json={"file_ids": [file_id]})
        assert resp.status_code == 200
        assert resp.mimetype == 'application/zip'

    def test_delete_file(self, admin_client, test_bid_file):
        pid, _, _, file_id = self._create_project_with_files(admin_client, test_bid_file)
        resp = admin_client.delete(self.DELETE_TPL % (pid, file_id))
        data = resp.get_json()
        assert data["success"] is True

    def test_delete_file_nonexistent(self, admin_client):
        resp = admin_client.delete(self.DELETE_TPL % (1, 99999))
        assert resp.status_code == 404

    def test_list_root_files_requires_access(self, auth_client):
        resp = auth_client.get("/admin/projects/1/files")
        assert resp.status_code == 403


class TestAdminProjectAiActivity:
    LIST_ROUTE = "/admin/projects"

    def _create_project(self, admin_client, name="AI活动测试"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        return resp.get_json()["id"]

    def test_ai_activity_empty(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.LIST_ROUTE + "/%d/ai_activity" % pid)
        data = resp.get_json()
        assert data["success"] is True
        assert data["items"] == []

    def test_ai_activity_no_access(self, auth_client):
        resp = auth_client.get(self.LIST_ROUTE + "/1/ai_activity")
        data = resp.get_json()
        assert data["success"] is True
        assert data["items"] == []

    def test_unread_count_zero(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.LIST_ROUTE + "/%d/unread_count" % pid)
        data = resp.get_json()
        assert data["success"] is True
        assert data["count"] == 0

    def test_unread_count_no_access(self, auth_client):
        resp = auth_client.get(self.LIST_ROUTE + "/1/unread_count")
        data = resp.get_json()
        assert data["count"] == 0

    def test_mark_read(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.LIST_ROUTE + "/%d/mark_read" % pid)
        data = resp.get_json()
        assert data["success"] is True

    def test_mark_read_no_access(self, auth_client):
        resp = auth_client.post(self.LIST_ROUTE + "/1/mark_read")
        assert resp.status_code == 403


class TestAdminProjectRegenVotes:
    LIST_ROUTE = "/admin/projects"

    def _create_project(self, admin_client, name="投票测试"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        return resp.get_json()["id"]

    def test_regen_votes_empty(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.LIST_ROUTE + "/%d/regen_votes" % pid)
        data = resp.get_json()
        assert data["success"] is True
        assert data["votes"] == []

    def test_regen_votes_requires_project_access(self, auth_client):
        pid = 1
        resp = auth_client.get(self.LIST_ROUTE + "/%d/regen_votes" % pid)
        assert resp.status_code == 403


class TestAdminProjectTodosDoneLog:
    LIST_ROUTE = "/admin/projects"
    TODOS_TPL = "/admin/projects/%d/todos"

    def _create_project(self, admin_client, name="已完成待办日志"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        return resp.get_json()["id"]

    def test_done_log_empty(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.TODOS_TPL % pid + "/done_log")
        data = resp.get_json()
        assert data["success"] is True
        assert data["logs"] == []

    def test_done_log_shows_completed(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.TODOS_TPL % pid, json={"content_copy": "已完成事项"})
        todo_id = resp.get_json()["todo_id"]
        admin_client.post(self.TODOS_TPL % pid + "/%d/done" % todo_id)
        resp = admin_client.get(self.TODOS_TPL % pid + "/done_log")
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["logs"]) == 1
        assert data["logs"][0]["content_copy"] == "已完成事项"

    def test_done_log_admin_only(self, auth_client):
        resp = auth_client.get("/admin/projects/1/todos/done_log")
        assert resp.status_code == 403


class TestAdminProjectWorkflow:
    LIST_ROUTE = "/admin/projects"
    WORKFLOW_TPL = "/admin/projects/%d/my_workflow"

    def _create_project(self, admin_client, name="工作流测试"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        return resp.get_json()["id"]

    def test_get_my_workflow(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.get(self.WORKFLOW_TPL % pid)
        data = resp.get_json()
        assert data["success"] is True
        assert "workflow" in data

    def test_save_my_workflow(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.WORKFLOW_TPL % pid, json={"steps": ["撰写招标公告", "审核文档"]})
        data = resp.get_json()
        assert data["success"] is True

    def test_my_workflow_no_access(self, auth_client):
        resp = auth_client.get("/admin/projects/1/my_workflow")
        assert resp.status_code == 403


class TestAdminProjectQuoteTree:
    LIST_ROUTE = "/admin/projects"
    QUOTE_TPL = "/admin/projects/%d/quote"

    def _create_project(self, admin_client, name="引用测试"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        return resp.get_json()["id"]

    def test_quote_create(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.QUOTE_TPL % pid, json={"quoted_message_id": 1})
        data = resp.get_json()
        assert data["success"] is True
        assert "quote_id" in data

    def test_quote_create_requires_id(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.QUOTE_TPL % pid, json={})
        assert resp.status_code == 400

    def test_quote_create_no_access(self, auth_client):
        resp = auth_client.post(self.QUOTE_TPL % 1, json={"quoted_message_id": 1})
        assert resp.status_code == 403

    def test_quote_tree_requires_access(self, auth_client):
        resp = auth_client.get(self.LIST_ROUTE + "/1/quote_tree/1")
        assert resp.status_code == 403


class TestAdminProjectBackfillChat:
    LIST_ROUTE = "/admin/projects"

    def _create_project(self, admin_client, name="回填测试"):
        resp = admin_client.post(self.LIST_ROUTE, json={"name": name})
        return resp.get_json()["id"]

    def test_backfill_chat(self, admin_client):
        pid = self._create_project(admin_client)
        resp = admin_client.post(self.LIST_ROUTE + "/%d/backfill_chat" % pid)
        data = resp.get_json()
        assert data["success"] is True

    def test_backfill_chat_no_access(self, auth_client):
        resp = auth_client.post(self.LIST_ROUTE + "/1/backfill_chat")
        assert resp.status_code == 404


class TestAdminMiscRoutes:
    ROUTE = "/admin"

    def test_runtime_config_schema(self, admin_client):
        resp = admin_client.get(self.ROUTE + "/runtime_config_schema")
        data = resp.get_json()
        assert data["success"] is True
        assert "schema" in data

    def test_runtime_config_schema_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE + "/runtime_config_schema")
        assert resp.status_code == 403

    def test_file_audit(self, admin_client):
        resp = admin_client.get(self.ROUTE + "/file_audit")
        data = resp.get_json()
        assert data["success"] is True
        assert "orphans" in data
        assert "disk_leaks" in data

    def test_file_audit_admin_required(self, auth_client):
        resp = auth_client.get(self.ROUTE + "/file_audit")
        assert resp.status_code == 403

    def test_audit_note(self, admin_client):
        resp = admin_client.post(self.ROUTE + "/audit_note", json={"note": "硬件维护记录"})
        data = resp.get_json()
        assert data["success"] is True

    def test_audit_note_requires_text(self, admin_client):
        resp = admin_client.post(self.ROUTE + "/audit_note", json={"note": ""})
        assert resp.status_code == 400

    def test_audit_note_admin_required(self, auth_client):
        resp = auth_client.post(self.ROUTE + "/audit_note", json={"note": "test"})
        assert resp.status_code == 403

    def test_search_cache_config_clear(self, admin_client):
        resp = admin_client.post(self.ROUTE + "/search_cache_config", json={"action": "clear"})
        data = resp.get_json()
        assert data["success"] is True

    def test_search_cache_config_set_ttl(self, admin_client):
        resp = admin_client.post(self.ROUTE + "/search_cache_config", json={"action": "set_ttl", "ttl_hours": 24})
        data = resp.get_json()
        assert data["success"] is True

    def test_search_cache_config_set_requires_ttl(self, admin_client):
        resp = admin_client.post(self.ROUTE + "/search_cache_config", json={"action": "set_ttl"})
        assert resp.status_code == 400

    def test_search_cache_config_admin_required(self, auth_client):
        resp = auth_client.post(self.ROUTE + "/search_cache_config", json={"action": "clear"})
        assert resp.status_code == 403


class TestAdminArchivedSessionsDelete:
    ROUTE = "/admin/archived_sessions"

    def test_delete_single_not_found(self, admin_client):
        resp = admin_client.delete(self.ROUTE + "/nonexistent-thread")
        assert resp.status_code == 404

    def test_delete_selected_requires_ids(self, admin_client):
        resp = admin_client.delete(self.ROUTE, json={})
        assert resp.status_code == 400

    def test_delete_selected_empty(self, admin_client):
        resp = admin_client.delete(self.ROUTE, json={"thread_ids": []})
        assert resp.status_code == 400

    def test_delete_all(self, admin_client):
        resp = admin_client.delete(self.ROUTE + "/all")
        data = resp.get_json()
        assert data["success"] is True

    def test_delete_single_admin_required(self, auth_client):
        resp = auth_client.delete(self.ROUTE + "/some-thread")
        assert resp.status_code == 403

    def test_delete_selected_admin_required(self, auth_client):
        resp = auth_client.delete(self.ROUTE, json={"thread_ids": ["t1"]})
        assert resp.status_code == 403

    def test_delete_all_admin_required(self, auth_client):
        resp = auth_client.delete(self.ROUTE + "/all")
        assert resp.status_code == 403


class TestAdminTaskDepositTransfer:
    ROUTE = "/admin/task_deposit"

    def test_transfer_nonexistent(self, admin_client):
        resp = admin_client.post(self.ROUTE + "/transfer/99999", json={"target_user_id": "x"})
        assert resp.status_code == 404

    def test_transfer_admin_required(self, auth_client):
        resp = auth_client.post(self.ROUTE + "/transfer/1")
        assert resp.status_code == 403
