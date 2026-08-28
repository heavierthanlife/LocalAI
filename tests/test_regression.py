"""Regression tests — verify known bug fixes don't silently regress.

These tests encode invariants from permanently-fixed bugs.
If any test fails, the corresponding fix has been accidentally undone.
"""
import pytest


# ── FIX-2026-07-15-001: NVIDIA provider removed (superseded) ──
def test_nvidia_provider_removed():
    from app.services.llm_provider import get_available_providers
    providers = get_available_providers()
    assert 'nvidia' not in providers, \
        "NVIDIA provider was removed from codebase; should not appear in available providers"


# ── FIX-2026-07-15-002: Wiki frontend no .data wrapper (superseded, check no stale patterns) ──
def test_wiki_frontend_no_data_envelope():
    with open('static/js/app.js', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'statsData.data' not in content, \
        "Wiki frontend must not reference .data envelope (ok() dict-spread means no wrapper)"


# ── FIX-2026-07-15-003: _score_timeline_compliance before SCORING_FUNCTIONS ──
def test_timeline_compliance_in_scoring_functions():
    from app.services.audit_engine import SCORING_FUNCTIONS
    assert 'timeline_compliance' in SCORING_FUNCTIONS, \
        "timeline_compliance must be in SCORING_FUNCTIONS dict"
    fn = SCORING_FUNCTIONS['timeline_compliance']
    assert fn.__name__ == '_score_timeline_compliance', \
        "SCORING_FUNCTIONS['timeline_compliance'] must reference _score_timeline_compliance function"


# ── FIX-2026-07-15-004: No bare nemotron in runtime_config ──
def test_runtime_config_no_bare_nemotron():
    with open('data/runtime_config.json', 'r', encoding='utf-8') as f:
        content = f.read()
    assert '"nemotron-3-ultra-550b-a55b"' not in content, \
        "runtime_config.json must not contain bare nemotron model name without nvidia/ prefix"


# ── FIX-2026-07-15-005: recycle_bin_service uses uploaded_by + user_col ──
def test_recycle_bin_has_user_col_detection():
    with open('app/services/recycle_bin_service.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert "'uploaded_by'" in content, \
        "recycle_bin_service.py must contain uploaded_by column reference"
    assert 'user_col' in content, \
        "recycle_bin_service.py must use dynamic user_id/uploaded_by column selection"


# ── FIX-2026-07-15-006: Recycle bin section data-attribute ──
def test_recycle_bin_section_data_attribute():
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'data-section="project_files"' in content, \
        "index.html must have data-section='project_files' on recycle bin buttons"


# ── FIX-2026-07-15-007: Chat response carries message IDs ──
def test_chat_response_carries_message_ids():
    with open('app/routes/chat.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert "user_message_id" in content, \
        "chat.py /send response must include user_message_id for poll cursor update"
    assert "assistant_message_id" in content, \
        "chat.py /send response must include assistant_message_id for poll cursor update"


# ── FIX-2026-07-15-007: Frontend poll cursor tracking ──
def test_chat_poll_cursor_tracking():
    with open('static/js/chat.js', 'r', encoding='utf-8') as f:
        content = f.read()
    assert '_pollLastId' in content, \
        "chat.js must track _pollLastId for message poll deduplication"
    assert '_lastKnownMessageId' in content, \
        "chat.js must track _lastKnownMessageId for poll cursor position"


# ── FIX-2026-07-15-008: Admin sidebar display set in verifyAuth ──
def test_admin_sidebar_extras_visibility():
    with open('static/js/app.js', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'adminExtras.style.display' in content, \
        "app.js must set sidebar admin extras visibility directly in verifyAuth"


# ── FIX-2026-07-19-001: Verdict case normalization ──
def test_verdict_uppercase_standard():
    from app.services.compliance_prompts import VERDICT_PASS
    assert VERDICT_PASS == 'PASS', \
        "VERDICT_PASS must be uppercase 'PASS' as canonical verdict format"


def test_verdict_normalization_in_compliance_checker():
    with open('app/services/compliance_checker.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert '.get("verdict"' in content, \
        "compliance_checker.py must access verdict via .get()"
    assert '.lower()' in content, \
        "compliance_checker.py must normalize verdicts via .lower() to avoid case mismatch"


# ── FIX-2026-07-19-002: XSS sanitization via _safeHTML ──
def test_compliance_xss_sanitization():
    with open('static/js/compliance.js', 'r', encoding='utf-8') as f:
        content = f.read()
    assert '_safeHTML' in content, \
        "compliance.js must define _safeHTML() for HTML sanitization"


def test_dompurify_cdn_in_index():
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'purify.min.js' in content, \
        "index.html must load DOMPurify CDN for XSS sanitization"


# ── FIX-2026-07-19-003: Tiptap ESM dynamic import (superseded) ──
def test_tiptap_esm_dynamic_import():
    with open('static/js/tiptap-editor.js', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'import(' in content, \
        "tiptap-editor.js must use ESM dynamic import() from CDN"


# ── FIX-2026-07-19-004: _taskIds.extracted preserved (superseded, check current state) ──
def test_taskids_extracted_preserved():
    with open('static/js/compliance.js', 'r', encoding='utf-8') as f:
        content = f.read()
    assert '_taskIds.extracted' in content, \
        "compliance.js must set _taskIds.extracted for rules task ID tracking"


# ── FIX-2026-07-19-005: law_monitor cursor fix ──
def test_law_monitor_no_stale_cursor():
    with open('app/services/law_monitor.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert "_compute_impact(cur," in content, \
        "law_monitor.py must pass a live cursor to _compute_impact (not closed cursor)"
    assert "cur if 'cur' in dir()" not in content, \
        "law_monitor.py must not use stale cursor fallback pattern"


# ── FIX-2026-08-28-001: /api/graph endpoints require login ──
def test_graph_endpoints_require_login(app):
    """Anonymous callers must be rejected; logged-in user must be allowed."""
    client = app.test_client()

    # Anonymous (no session) → rejected
    r = client.get('/api/graph/types')
    assert r.status_code == 403, "Anonymous /api/graph/types must be rejected (403)"

    # Simulate a logged-in admin session (bypasses DB-backed /login)
    with client.session_transaction() as sess:
        sess['user_id'] = 'test-admin'
        sess['consent_value'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['is_auditor'] = True

    r = client.get('/api/graph/types')
    assert r.status_code == 200, f"Authed /api/graph/types must succeed: {r.status_code}"


def test_graph_source_has_login_required():
    """graph.py must keep login_required on every route."""
    with open('app/routes/graph.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert content.count('@login_required') == 5, \
        "graph.py must decorate all 5 endpoints with @login_required"
    assert '_require_project_access' in content, \
        "graph.py must enforce project-membership (IDOR) check"


# ── FIX-2026-08-28-002: admin PIN fail-closed in production ──
def test_admin_pin_production_fail_closed():
    """APP_ENV=production without ADMIN_PIN/ADMIN_PASSWORD_HASH must refuse to start."""
    with open('app/__init__.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'app_env == "production" and not admin_pin' in content, \
        "create_app must fail closed in production when ADMIN_PIN is missing"
    assert 'requires ADMIN_PIN or ADMIN_PASSWORD_HASH' in content, \
        "fail-closed error must mention ADMIN_PIN / ADMIN_PASSWORD_HASH"
    assert 'weak default' in content, \
        "development fallback must log a weak-default warning"


def test_compose_sets_app_env_production():
    """docker-compose must set APP_ENV=production (so prod is fail-closed)."""
    with open('docker-compose.yml', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'APP_ENV=production' in content, \
        "docker-compose app service must set APP_ENV=production"
    assert 'ADMIN_PIN=${ADMIN_PIN:-}' in content, \
        "compose must not default ADMIN_PIN to 123456"


# ── FIX-2026-08-28-003: credit_tasks must be Redis-backed (cross-worker) ──
def test_credit_task_registry_roundtrip(mock_redis):
    """Credit task state must survive set→get→patch across a Redis-backed registry."""
    from app.services import credit_task_registry as reg
    reg.set_task('t-reg-1', {
        'status': 'running', 'progress': 0, 'total': 2,
        'captcha_image': b'\x89PNG\r\n\x1a\n', 'captcha_solution': None,
        'waiting': False, 'reload_captcha': False, 'download_url': 'http://x/r',
    })
    t = reg.get_task('t-reg-1')
    assert t is not None, "get_task must return the stored task"
    assert t['status'] == 'running'
    assert t['captcha_image'] == b'\x89PNG\r\n\x1a\n', \
        "captcha_image bytes must round-trip through Redis"
    assert t.get('captcha_solution') in (None, ''), \
        "None captcha_solution must be preserved as falsy"

    reg.patch_task('t-reg-1', progress=5, waiting=True, captcha_solution='abcd')
    t2 = reg.get_task('t-reg-1')
    assert t2['progress'] == 5
    assert t2['waiting'] is True
    assert t2['captcha_solution'] == 'abcd'

    assert reg.task_exists('t-reg-1') is True
    assert 't-reg-1' in reg.list_task_ids()
    reg.delete_task('t-reg-1')
    assert reg.task_exists('t-reg-1') is False


def test_credit_routes_no_inmemory_registry():
    """credit.py must not reference the old process-local credit_tasks dict."""
    with open('app/routes/credit.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'credit_tasks[' not in content, \
        "credit.py must not write to the in-memory credit_tasks dict"
    assert '_credit_tasks_lock' not in content, \
        "credit.py must not use the process-local credit_tasks lock"


# ── FIX-2026-08-28-004: anonymous chat history must be PostgreSQL-backed ──
def test_anon_chat_messages_table_defined():
    """database.py must define the anon_chat_messages JSONB table."""
    with open('app/database.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'anon_chat_messages' in content, \
        "database.py must define anon_chat_messages table"
    assert 'messages    JSONB' in content, \
        "anon_chat_messages.messages must be JSONB"


def test_anonymous_uses_db_not_json_file():
    """anonymous.py must persist history to PG, not per-thread JSON files."""
    with open('app/services/anonymous.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'ON CONFLICT (thread_id) DO UPDATE' in content, \
        "anonymous.py must use atomic UPSERT append to anon_chat_messages"
    assert 'get_db_connection' in content, \
        "anonymous.py must use the DB connection pool"
    assert 'FileLock' not in content, \
        "anonymous.py must not use file locks for history persistence"


# ── FIX-2026-08-28-005: prompt system — language consistency + dedup + dead code ──
def test_judge_prompt_is_chinese():
    """JUDGE_PROMPT must be Chinese (all-else-Chinese system consistency)."""
    with open('app/services/judge_review.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert '你是一名严格的中文质量审查员' in content, \
        "JUDGE_PROMPT must be Chinese"
    assert 'You are a quality reviewer' not in content, \
        "JUDGE_PROMPT must not remain English"
    assert 'You are a strict quality reviewer' not in content, \
        "duplicate-role English SystemMessage must be removed"


def test_structured_prompt_is_chinese():
    """ingest_pipeline STRUCTURED_PROMPT must be Chinese."""
    with open('app/services/ingest_pipeline.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert '你是一名专业的采购/招投标文档分析员' in content, \
        "STRUCTURED_PROMPT must be Chinese"
    assert 'You are a procurement document analyst' not in content, \
        "STRUCTURED_PROMPT must not remain English"


def test_main_agent_prompt_domain_expert():
    """The default agent prompt must position as a bidding-domain expert, not generic 答疑助手."""
    with open('app/globals.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert '中联招标智能助手' in content, \
        "main prompt must identify as 中联招标智能助手"
    assert '你是一个答疑助手' not in content, \
        "main prompt must not be the generic 答疑助手"
    # tool constraint must be generic, not hardcoding get_date/bocha_search
    assert '使用系统提供的工具' in content, \
        "tool constraint must be generic (auto-adapts to added tools)"


def test_call_llm_guard_dedup():
    """call_llm must not append the safety guard twice."""
    with open('app/services/llm_provider.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'guard not in system_prompt' in content, \
        "call_llm must dedup the safety guard (guard not in system_prompt)"


def test_dead_prompts_removed():
    """Dead prompt constants must not exist."""
    with open('app/services/analysis_prompts.py', 'r', encoding='utf-8') as f:
        a = f.read()
    assert 'BID_COMPARISON_SYSTEM' not in a, \
        "BID_COMPARISON_SYSTEM is dead code and must be removed"
    assert 'build_bid_analysis_prompt' not in a, \
        "build_bid_analysis_prompt is dead code and must be removed"
    with open('app/services/wiki_prompts.py', 'r', encoding='utf-8') as f:
        w = f.read()
    assert 'WIKI_LINT_SYSTEM_PROMPT' not in w, \
        "WIKI_LINT_SYSTEM_PROMPT is dead code and must be removed"
    assert 'WIKI_UPDATE_INDEX_PROMPT' not in w, \
        "WIKI_UPDATE_INDEX_PROMPT is dead code and must be removed"
    with open('app/services/prompt_safety.py', 'r', encoding='utf-8') as f:
        p = f.read()
    assert '_VL_CONSISTENCY_PROMPT' not in p, \
        "_VL_CONSISTENCY_PROMPT is dead code and must be removed"


def test_agent_prompt_file_not_test_override():
    """data/agent_prompt.json must not contain the broken 'Test prompt' override."""
    import json
    with open('data/agent_prompt.json', 'r', encoding='utf-8') as f:
        saved = json.load(f).get('prompt', '')
    assert saved.strip() != 'Test prompt', \
        "agent_prompt.json must not hold the leftover 'Test prompt' override"
    assert len(saved.strip()) > 100, \
        "agent_prompt.json should hold the real (long) default prompt"


# ── FIX-2026-08-28-006: clearance results move into chat (toolbar tab area removed) ──
def test_clearance_toolbar_results_removed():
    """index.html must no longer contain the clearance results tab area."""
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'clearanceResults' not in content, \
        "toolbar #clearanceResults area must be removed (results move to chat)"
    assert 'clearance-tab-btn' not in content, \
        "clearance tab buttons must be removed from the toolbar"
    assert 'clearanceDownloadLink' not in content, \
        "toolbar clearance download link must be removed"
    # Tool stays: file select, run button, progress
    assert 'runClearanceBtn' in content, "clearance run button must remain"
    assert 'clearanceProgress' in content, "clearance progress must remain"


def test_clearance_chat_persistence_backend():
    """clearance_engine must persist the result as a CLEARANCE_REPORT chat message."""
    with open('app/services/clearance_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'CLEARANCE_REPORT' in content, \
        "clearance_engine must mark the persisted message with CLEARANCE_REPORT"
    assert 'INSERT INTO chat_messages' in content, \
        "clearance_engine must INSERT into chat_messages (role=assistant)"
    assert "role, content, thinking, timestamp" in content, \
        "INSERT must target the chat_messages columns"


def test_clearance_chat_marker_handled():
    """chat.js renderAssistantMessageLegacy must handle the CLEARANCE_REPORT marker."""
    with open('static/js/chat.js', 'r', encoding='utf-8') as f:
        content = f.read()
    assert "includes('CLEARANCE_REPORT')" in content, \
        "chat.js must detect the CLEARANCE_REPORT marker"


def test_clearance_chat_render_functions():
    """app.js must expose chat-rendering helpers for clearance."""
    with open('static/js/app.js', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'buildClearanceReportHtml' in content, \
        "app.js must build clearance report HTML"
    assert 'appendClearanceToChat' in content, \
        "app.js must append clearance results into the chat"
    assert '_attachClearanceHandlers' in content, \
        "app.js must attach scoped clearance handlers (no inline onclick/CSP break)"
    assert 'renderClearanceResults' not in content, \
        "old renderClearanceResults (toolbar) must be removed"


def test_taskbus_extra_metadata():
    """TaskBus.start() must accept extra metadata (e.g. thread_id) via Redis hash."""
    with open('app/services/task_bus.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'def start(self, extra' in content, \
        "TaskBus.start must accept an extra metadata dict"
    assert 'extra.items()' in content, \
        "TaskBus.start must merge extra metadata into the hash"


# ── FIX-2026-08-28-007: task status 404 race — queued pre-registration ──
def test_taskbus_register_queued():
    """TaskBus must expose register_queued to pre-register a task synchronously."""
    with open('app/services/task_bus.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'def register_queued' in content, \
        "TaskBus must have register_queued to avoid status 404 before worker start"
    assert 'STATUS_QUEUED' in content, \
        "TaskBus must use the queued status constant"


def test_clearance_and_docanalysis_register_queued():
    """/clearance/run and /document_analysis/analyze must pre-register the task."""
    with open('app/routes/clearance.py', 'r', encoding='utf-8') as f:
        c = f.read()
    assert 'register_queued' in c, \
        "clearance route must pre-register the task as queued before send_task"
    with open('app/routes/document_analysis.py', 'r', encoding='utf-8') as f:
        d = f.read()
    assert 'register_queued' in d, \
        "document_analysis route must pre-register the task as queued before send_task"


def test_clearance_status_precheck_tolerates_404():
    """app.js status pre-check must tolerate a transient 404/race and fall through to SSE."""
    with open('static/js/app.js', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'statusCheck.ok' in content, \
        "app.js must only parse the status body when statusCheck.ok"
    assert 'statusData = {}' in content, \
        "app.js must tolerate a failed/404 status fetch"
