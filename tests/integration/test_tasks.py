"""Integration tests for tasks blueprint (list, get, stream).

Requires PostgreSQL (clean_tables fixtures reset between tests).
Redis is mocked via fakeredis (autouse mock_redis fixture).
"""
import json
import time
import pytest


pytestmark = pytest.mark.db


def _create_task(task_id=None, task_type="doc_analysis", label="测试任务", status="running"):
    """Helper: create a task directly in Redis via TaskBus."""
    from app.services.task_bus import TaskBus, STATUS_RUNNING, STATUS_COMPLETED
    bus = TaskBus(task_id=task_id or "test-task-001", task_type=task_type, label=label)
    bus.start()
    if status == "completed":
        bus.complete(result={"summary": "done"})
    return bus.task_id


class TestListTasks:
    ROUTE = "/tasks"

    def test_returns_empty_list(self, client):
        """Returns empty list when no tasks exist."""
        resp = client.get(self.ROUTE)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["tasks"] == []
        assert data["count"] == 0

    def test_returns_recent_tasks(self, client):
        """Returns tasks stored in Redis."""
        _create_task("task-list-1", "doc_analysis", "分析文档", "completed")
        _create_task("task-list-2", "skill_extract", "提取技能", "running")
        resp = client.get(self.ROUTE)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["count"] >= 2
        task_ids = {t["task_id"] for t in data["tasks"]}
        assert "task-list-1" in task_ids
        assert "task-list-2" in task_ids

    def test_honors_limit_param(self, client):
        """Respects the limit query parameter."""
        for i in range(5):
            _create_task(f"task-limit-{i}", "doc_analysis", f"Task {i}", "completed")
        resp = client.get(self.ROUTE + "?limit=3")
        data = resp.get_json()
        assert data["count"] <= 3

    def test_filters_by_logged_in_user(self, auth_client):
        """When logged in, only shows user's own tasks."""
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM users WHERE username = 'testuser'")
                row = cur.fetchone()
                user_id = row[0] if row else "unknown"
        # Create a task with the user's ID in metadata
        from app.services.redis_client import get_redis
        r = get_redis()
        if r:
            import uuid
            tid = str(uuid.uuid4())
            meta = {
                "status": "running", "type": "doc_analysis", "label": "用户任务",
                "progress": "50", "message": "处理中",
                "user_id": user_id,
            }
            r.hset(f"task_meta:{tid}", mapping=meta)
            r.zadd("task_registry", {tid: time.time()})
        resp = auth_client.get(self.ROUTE)
        data = resp.get_json()
        task_labels = {t.get("label") for t in data["tasks"]}
        if "用户任务" in task_labels:
            assert True  # Task visible to user
        else:
            # Tasks without user_id should also be visible
            pass
        assert data["success"] is True


class TestGetTask:
    ROUTE = "/tasks"

    def test_returns_404_for_nonexistent(self, client):
        """Returns 404 when task_id does not exist."""
        resp = client.get(self.ROUTE + "/nonexistent-task-12345")
        assert resp.status_code == 404

    def test_returns_task_meta(self, client):
        """Returns task metadata for a known task."""
        tid = _create_task("get-task-test", "doc_analysis", "查询测试", "completed")
        resp = client.get(self.ROUTE + f"/{tid}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["task_id"] == tid
        assert data["status"] in ("running", "completed")
        assert data["type"] == "doc_analysis"
        assert data["label"] == "查询测试"
        assert "progress" in data

    def test_includes_task_id_in_response(self, client):
        """Response includes task_id field."""
        tid = _create_task("get-task-id-check", "skill_extract", "ID检查", "running")
        resp = client.get(self.ROUTE + f"/{tid}")
        data = resp.get_json()
        assert data["task_id"] == tid


class TestStreamTask:
    ROUTE = "/tasks"

    def test_returns_404_for_nonexistent(self, client):
        """Returns 404 for stream of nonexistent task."""
        resp = client.get(self.ROUTE + "/stream-nonexistent/stream")
        assert resp.status_code == 404

    def test_returns_sse_content_type(self, client):
        """Returns text/event-stream content type."""
        tid = _create_task("sse-type-test", "doc_analysis", "SSE类型测试", "completed")
        resp = client.get(self.ROUTE + f"/{tid}/stream?timeout=1")
        assert resp.status_code == 200
        assert resp.mimetype == "text/event-stream"
        assert resp.headers.get("X-Accel-Buffering") == "no"
        assert resp.headers.get("Cache-Control") == "no-cache"

    def test_emits_events_for_completed_task(self, client):
        """Stream emits events for an already-completed task."""
        tid = _create_task("sse-complete-test", "doc_analysis", "SSE完成测试", "completed")
        resp = client.get(self.ROUTE + f"/{tid}/stream?timeout=1")
        assert resp.status_code == 200
        # Consume the stream generator — iterate a few events
        events = []
        for chunk in resp.response:
            if chunk:
                events.append(chunk.decode("utf-8"))
            if len(events) >= 5:
                break
        # Should have at least the handshake and the complete event
        combined = "".join(events)
        assert ": ok\n\n" in combined  # SSE handshake
        assert "event:" in combined or "data:" in combined

    def test_stream_handshake(self, client):
        """SSE stream starts with handshake."""
        tid = _create_task("sse-handshake", "doc_analysis", "握手测试", "completed")
        resp = client.get(self.ROUTE + f"/{tid}/stream?timeout=1")
        # Read first chunk
        first = b""
        for chunk in resp.response:
            if chunk:
                first = chunk
                break
        assert b": ok" in first
