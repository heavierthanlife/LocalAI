"""Core smoke tests — run without database or external services.

Run: pytest tests/test_smoke.py -v
"""

import io
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Test 1: API response helpers (no app imports needed) ─────────

def test_ok_helper():
    """ok() flat-merges dict data: {success:true, message, ...data_fields}."""
    from flask import Flask
    from app.utils.helpers import ok
    with Flask(__name__).test_request_context():
        resp, status = ok({"id": 1}, "created", 201)
        data = resp.get_json()
        assert data["success"] is True
        assert data["id"] == 1        # flat-merged from dict
        assert data["message"] == "created"
        assert status == 201

        # Non-dict data still uses envelope
        resp2, _ = ok([1, 2, 3], "list data", 200)
        d2 = resp2.get_json()
        assert d2["success"] is True
        assert d2["data"] == [1, 2, 3]

        # No data
        resp3, _ = ok(message="no data")
        d3 = resp3.get_json()
        assert d3["success"] is True
        assert "data" not in d3
        assert d3["message"] == "no data"


def test_err_helper():
    """err() returns {success:false, error, code}."""
    from flask import Flask
    from app.utils.helpers import err
    with Flask(__name__).test_request_context():
        resp, status = err("文件未找到", "NOT_FOUND", 404)
        data = resp.get_json()
        assert data["success"] is False
        assert data["error"] == "文件未找到"
        assert data["code"] == "NOT_FOUND"
        assert status == 404


# ── Test 2: Red Team agent prompt ───────────────────────────────

def test_redteam_prompt_spec():
    """REDTEAM_SYSTEM_PROMPT contains key requirements (read from file to avoid DB import)."""
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "app", "services", "redteam_agent.py"
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "REDTEAM_SYSTEM_PROMPT" in content
    assert "质问" in content
    assert "供应商" in content


# ── Test 3: split_thinking_answer ───────────────────────────────

def test_split_thinking_answer():
    from app.utils.helpers import split_thinking_answer

    text = "【思考】这是思考内容【回答】这是回答内容"
    thinking, answer = split_thinking_answer(text)
    assert thinking == "这是思考内容"
    assert answer == "这是回答内容"

    # No delimiter: everything is answer
    t2, a2 = split_thinking_answer("纯回答，无思考标记")
    assert t2 is None
    assert a2 == "纯回答，无思考标记"


# ── Test 4: safe_error_response ─────────────────────────────────

def test_safe_error_response():
    from app.utils.helpers import safe_error_response

    result = safe_error_response("测试错误")
    assert result.startswith("[错误]")
    assert "测试错误" in result

    result2 = safe_error_response()
    assert "处理文件时出错" in result2


# ── Test 5: beijing_now and utc_now ──────────────────────────────

def test_time_helpers():
    from app.utils.helpers import beijing_now, utc_now
    from datetime import datetime, timezone

    bj = beijing_now()
    assert isinstance(bj, str)
    assert len(bj) == 19  # YYYY-MM-DD HH:MM:SS

    utc = utc_now()
    assert isinstance(utc, datetime)
    assert utc.tzinfo == timezone.utc


# ── Test 6: Graph protocol ─────────

def test_graph_protocol():
    """to_graph_response outputs correct unified format."""
    from app.services.graph_protocol import GraphNode, GraphEdge, to_graph_response
    g = to_graph_response(
        [GraphNode(id='a', label='Test Co', type='company')],
        [GraphEdge(source='a', target='b', label='collusion_signal', weight=0.8)],
        'collusion',
    )
    assert g['web_type'] == 'collusion'
    assert g['nodes'][0]['type'] == 'company'
    assert g['nodes'][0]['id'] == 'a'
    assert g['edges'][0]['label'] == 'collusion_signal'
    assert g['edges'][0]['weight'] == 0.8
    assert g['stats']['node_count'] == 1
    assert g['stats']['edge_count'] == 1
