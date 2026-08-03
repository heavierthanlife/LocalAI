"""Unit tests for app.utils.helpers — API response, time, error, split.

All pure function tests — no DB, no app context needed for most.
"""
import json
import re
from datetime import timezone

import pytest


# ── ok() ─────────────────────────────────────────────────────────────────

@pytest.mark.usefixtures("app_context")
class TestOk:
    def test_dict_data_flat_merged(self):
        from app.utils.helpers import ok
        resp, status = ok({"id": 1}, "created", 201)
        data = resp.get_json()
        assert data["success"] is True
        assert data["id"] == 1
        assert data["message"] == "created"
        assert status == 201

    def test_list_data_envelope(self):
        from app.utils.helpers import ok
        resp, _ = ok([1, 2, 3], "list", 200)
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"] == [1, 2, 3]

    def test_no_data(self):
        from app.utils.helpers import ok
        resp, _ = ok(message="no data")
        data = resp.get_json()
        assert data["success"] is True
        assert "data" not in data
        assert data["message"] == "no data"

    def test_string_data_envelope(self):
        from app.utils.helpers import ok
        resp, _ = ok("hello", "ok")
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"] == "hello"

    def test_default_message(self):
        from app.utils.helpers import ok
        resp, _ = ok({"x": 1})
        data = resp.get_json()
        assert data["message"] == "ok"

    def test_default_status(self):
        from app.utils.helpers import ok
        resp, status = ok({"x": 1})
        assert status == 200

    def test_none_data_excluded(self):
        from app.utils.helpers import ok
        resp, _ = ok(None, "none")
        data = resp.get_json()
        assert "data" not in data


# ── err() ─────────────────────────────────────────────────────────────────

@pytest.mark.usefixtures("app_context")
class TestErr:
    def test_basic_error(self):
        from app.utils.helpers import err
        resp, status = err("文件未找到", "NOT_FOUND", 404)
        data = resp.get_json()
        assert data["success"] is False
        assert data["error"] == "文件未找到"
        assert data["code"] == "NOT_FOUND"
        assert status == 404

    def test_default_code(self):
        from app.utils.helpers import err
        resp, _ = err("something broke")
        data = resp.get_json()
        assert data["code"] == "UNKNOWN"

    def test_default_status(self):
        from app.utils.helpers import err
        resp, status = err("bad")
        assert status == 400

    def test_error_converted_to_string(self):
        from app.utils.helpers import err
        resp, _ = err(ValueError("oops"), "ERR")
        data = resp.get_json()
        assert "oops" in data["error"]


# ── beijing_now() ─────────────────────────────────────────────────────────

class TestBeijingNow:
    def test_format(self):
        from app.utils.helpers import beijing_now
        result = beijing_now()
        assert isinstance(result, str)
        assert len(result) == 19
        assert re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', result)

    def test_returns_string(self):
        from app.utils.helpers import beijing_now
        assert isinstance(beijing_now(), str)


# ── utc_now() ─────────────────────────────────────────────────────────────

class TestUtcNow:
    def test_returns_datetime(self):
        from app.utils.helpers import utc_now
        from datetime import datetime
        assert isinstance(utc_now(), datetime)

    def test_utc_timezone(self):
        from app.utils.helpers import utc_now
        assert utc_now().tzinfo == timezone.utc


# ── safe_error_response() ─────────────────────────────────────────────────

class TestSafeErrorResponse:
    def test_default_message(self):
        from app.utils.helpers import safe_error_response
        result = safe_error_response()
        assert result.startswith("[错误]")
        assert "处理文件时出错" in result

    def test_custom_message(self):
        from app.utils.helpers import safe_error_response
        result = safe_error_response("自定义错误")
        assert result == "[错误] 自定义错误"

    def test_with_log_error_no_crash(self):
        from app.utils.helpers import safe_error_response
        result = safe_error_response("err", log_error="something")
        assert result == "[错误] err"


# ── split_thinking_answer() ───────────────────────────────────────────────

class TestSplitThinkingAnswer:
    def test_json_format_thinking_answer(self):
        from app.utils.helpers import split_thinking_answer
        text = '{"思考":"step 1","回答":"final"}'
        think, ans = split_thinking_answer(text)
        assert think == "step 1"
        assert ans == "final"

    def test_json_format_english_keys(self):
        from app.utils.helpers import split_thinking_answer
        text = '{"thinking":"reasoning","answer":"result"}'
        think, ans = split_thinking_answer(text)
        assert think == "reasoning"
        assert ans == "result"

    def test_json_reasoning_key(self):
        from app.utils.helpers import split_thinking_answer
        text = '{"reasoning":"think","response":"answer"}'
        think, ans = split_thinking_answer(text)
        assert think == "think"
        assert ans == "answer"

    def test_double_braced_json(self):
        from app.utils.helpers import split_thinking_answer
        text = '{{"思考":"深思","回答":"终答"}}'
        think, ans = split_thinking_answer(text)
        assert think == "深思"
        assert ans == "终答"

    def test_line_starting_chinese_markers(self):
        from app.utils.helpers import split_thinking_answer
        text = "【思考】\n这是思考\n【回答】\n这是回答"
        think, ans = split_thinking_answer(text)
        assert think == "这是思考"
        assert ans == "这是回答"

    def test_inline_chinese_markers(self):
        from app.utils.helpers import split_thinking_answer
        text = "【思考】这是思考【回答】这是回答"
        think, ans = split_thinking_answer(text)
        assert think == "这是思考"
        assert ans == "这是回答"

    def test_colon_format(self):
        from app.utils.helpers import split_thinking_answer
        text = "思考：深入分析\n回答：最终结论"
        think, ans = split_thinking_answer(text)
        assert think == "深入分析"
        assert ans == "最终结论"

    def test_xml_format(self):
        from app.utils.helpers import split_thinking_answer
        text = "<思考>推理过程</思考>这是回答内容"
        think, ans = split_thinking_answer(text)
        assert think == "推理过程"
        assert ans == "这是回答内容"

    def test_no_match_returns_none_and_full(self):
        from app.utils.helpers import split_thinking_answer
        text = "纯回答，无思考标记"
        think, ans = split_thinking_answer(text)
        assert think is None
        assert ans == "纯回答，无思考标记"

    def test_empty_text(self):
        from app.utils.helpers import split_thinking_answer
        think, ans = split_thinking_answer("")
        assert think is None
        assert ans == ""

    def test_only_thinking_no_answer(self):
        from app.utils.helpers import split_thinking_answer
        text = "【思考】只有思考"
        think, ans = split_thinking_answer(text)
        assert think is None
        assert ans == "【思考】只有思考"

    def test_content_after_json_block(self):
        from app.utils.helpers import split_thinking_answer
        text = '一些前置内容\n\n{{"思考":"think","回答":"answer"}}'
        think, ans = split_thinking_answer(text)
        assert think == "think"
        assert ans == "answer"

    def test_json_no_answer_field_falls_through(self):
        from app.utils.helpers import split_thinking_answer
        text = '{"思考":"only think"}'
        think, ans = split_thinking_answer(text)
        assert think is None

    def test_multiline_answer(self):
        from app.utils.helpers import split_thinking_answer
        text = "【思考】步骤一\n步骤二\n【回答】结果一\n结果二"
        think, ans = split_thinking_answer(text)
        assert think == "步骤一\n步骤二"
        assert ans == "结果一\n结果二"


# ── _extract_json_block() — internal helper ───────────────────────────────

class TestExtractJsonBlock:
    def test_valid_json(self):
        from app.utils.helpers import _extract_json_block
        assert _extract_json_block('{"a":1}') == {"a": 1}

    def test_double_braced(self):
        from app.utils.helpers import _extract_json_block
        assert _extract_json_block('{{"a":1}}') == {"a": 1}

    def test_not_json(self):
        from app.utils.helpers import _extract_json_block
        assert _extract_json_block("hello") is None

    def test_json_array_returns_none(self):
        from app.utils.helpers import _extract_json_block
        assert _extract_json_block("[1,2]") is None

    def test_malformed_json(self):
        from app.utils.helpers import _extract_json_block
        assert _extract_json_block("{bad}") is None


class TestExtractJsonBlockAtEnd:
    def test_finds_last_json_block(self):
        from app.utils.helpers import _extract_json_block_at_end
        text = '原始内容\n{{"思考":"t","回答":"a"}}'
        result = _extract_json_block_at_end(text)
        assert result == {"思考": "t", "回答": "a"}

    def test_no_json_returns_none(self):
        from app.utils.helpers import _extract_json_block_at_end
        assert _extract_json_block_at_end("no json") is None
