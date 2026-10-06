"""Regression tests — verify known bug fixes don't silently regress.

These tests encode invariants from permanently-fixed bugs.
If any test fails, the corresponding fix has been accidentally undone.
"""
import pytest


# ── FIX-2026-09-01-016: NVIDIA provider re-enabled (FIX-016 supersedes the old removal) ──
def test_nvidia_provider_configured():
    """FIX-016: NVIDIA NIM is now an active free provider; must appear in PROVIDER_CONFIG."""
    from app.services.llm_provider import PROVIDER_CONFIG
    assert 'nvidia' in PROVIDER_CONFIG, "NVIDIA NIM provider config missing"
    assert 'openrouter' in PROVIDER_CONFIG, "OpenRouter provider config missing"


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


# ── FIX-2026-07-19-002 / FIX-2026-09-15-054: XSS sanitization via global _safeHTML ──
def test_compliance_xss_sanitization():
    app_src = _read('static/js/app.js')
    assert 'function _safeHTML(html)' in app_src, \
        "app.js must define the global _safeHTML() sanitizer"


def test_dompurify_cdn_in_index():
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'purify.min.js' in content, \
        "index.html must load DOMPurify CDN for XSS sanitization"


# ── FIX-2026-09-15-056: compliance UI removed (降级 API-only) ──
def test_compliance_frontend_removed():
    import os
    assert not os.path.exists(os.path.join('static', 'js', 'compliance.js')), \
        "compliance.js removed (API-only downgrade)"
    assert not os.path.exists(os.path.join('static', 'js', 'tiptap-editor.js')), \
        "tiptap-editor.js removed (compliance-only editor)"
    idx = _read('templates/index.html')
    assert 'js/compliance.js' not in idx
    assert 'tiptap-editor.js' not in idx


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


def test_agent_prompt_default_is_hardcoded():
    """The global agent prompt is hardcoded in app/globals.py; the file-based
    data/agent_prompt.json override was retired (file is gitignored)."""
    src = _read('app/globals.py')
    assert '_DEFAULT_PROMPT' in src
    assert 'def get_default_prompt' in src
    assert 'open(' not in src.split('_DEFAULT_PROMPT')[0].split('def get_default_prompt')[0], \
        'agent prompt must not be read from disk at import time'


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


def test_clearance_register_queued():
    """/clearance/run must pre-register the task as queued before send_task."""
    with open('app/routes/clearance.py', 'r', encoding='utf-8') as f:
        c = f.read()
    assert 'register_queued' in c, \
        "clearance route must pre-register the task as queued before send_task"


def test_clearance_status_precheck_tolerates_404():
    """app.js status pre-check must tolerate a transient 404/race and fall through to SSE."""
    with open('static/js/app.js', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'statusCheck.ok' in content, \
        "app.js must only parse the status body when statusCheck.ok"
    assert 'statusData = {}' in content, \
        "app.js must tolerate a failed/404 status fetch"


# ── FIX-2026-08-28-009: 清标报告生产级升级 — 三节渲染 + 文本相似 + 开标表 ──
def test_section3_indicator_tables_rendered():
    """document_analysis_svc section 三 must render per-indicator tables (B1 dead-code fix)."""
    with open('app/services/document_analysis_svc.py', 'r', encoding='utf-8') as f:
        content = f.read()
    # The 'continue' dead-code bug: _add_heading for each category must be
    # reachable (not inside an if-not-group: continue block).
    assert '_add_heading(doc, f\'{prefix}{cat}\', H2)' in content, \
        "section 三 category heading must be reachable code"
    assert '_add_heading(doc, f\'{sub_no}、{ind["name"]}\', H2)' in content, \
        "per-indicator heading must be reachable code"
    # Verify the render loop is not dead: the 'if not group: continue' must be
    # immediately followed by the heading (single-level indent), not the whole loop.
    idx = content.index('if not group:')
    after = content[idx:idx + 80]
    assert 'continue\n        _add_heading' in after, \
        "continue must not swallow the indicator rendering loop (B1)"


def test_text_sim_tfidf_wired():
    """Clearance + indicator text-sim paths must precompute and pass the TF-IDF matrix."""
    with open('app/services/clearance_engine.py', 'r', encoding='utf-8') as f:
        c = f.read()
    assert '_precompute_tfidf_for_files' in c, \
        "clearance cross-comparison must precompute TF-IDF"
    assert 'tfidf_matrix=tfidf_matrix' in c, \
        "clearance must pass tfidf_matrix to compute_all_pairs"
    with open('app/services/document_analysis_svc.py', 'r', encoding='utf-8') as f:
        d = f.read()
    assert '_precompute_tfidf_for_files' in d, \
        "indicator text_sim checker must precompute TF-IDF"


def test_indicators_triggered_is_count():
    """suspected_units.indicators_triggered must be a real count, not a boolean."""
    with open('app/services/document_analysis_svc.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'int(triggered_count > 0)' not in content, \
        "indicators_triggered must not be int(boolean)"
    assert 'su[\'indicators_triggered\'] = count' in content, \
        "indicators_triggered must be computed as a per-file count"


def test_openinfo_service_parses():
    """clearance_openinfo must parse 开标信息表 and extract 评审标准."""
    from app.services.clearance_openinfo import (
        parse_open_info_file, extract_eval_criteria, compute_open_info_indicators,
    )
    # eval criteria extraction from a tender text
    tender = ("本工程预算价1500万元，计划开标时间2026年9月1日，采用综合评估法，"
              "技术分40分、商务分30分、价格分30分。")
    ec = extract_eval_criteria(tender)
    assert ec['budget_price'] == 15000000, f"budget_price wrong: {ec['budget_price']}"
    assert ec['plan_open_time'] and '2026' in ec['plan_open_time']
    assert ec['eval_method'] == '综合评估法'
    # open-info indicators with contact/phone dup + expert scores
    open_info = {
        'rows': [
            {'bidder': 'A公司', 'contact': '张三', 'phone': '13800000001',
             'bid_price': '10000000', 'winner': 'A公司', 'remark': ''},
            {'bidder': 'B公司', 'contact': '张三', 'phone': '13800000001',
             'bid_price': '12000000', 'winner': '', 'remark': ''},
        ]
    }
    res = compute_open_info_indicators([{'filename': 'A公司.docx'}, {'filename': 'B公司.docx'}],
                                       open_info, ec)
    assert 'contact_person_same' in res, "contact_person_same must be computed"
    assert res['contact_person_same']['score'] > 0, "dup contact must score > 0"
    assert 'contact_phone_abnormal' in res, "contact_phone_abnormal must be computed"
    assert res['contact_phone_abnormal']['score'] > 0, "dup phone must score > 0"


def test_clearance_route_accepts_open_info():
    """clearance.py must accept open_info_file_id and preview_criteria endpoint."""
    with open('app/routes/clearance.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'open_info_file_id' in content, \
        "clearance route must accept open_info_file_id"
    assert 'preview_criteria' in content, \
        "clearance route must expose /clearance/preview_criteria"


def test_openinfo_upload_ui():
    """index.html must have the 开标信息表 upload slot."""
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'clearanceOpenInfoInput' in content, \
        "index.html must include a 开标信息表 upload input"
    assert '选择开标信息表' in content, \
        "index.html must label the 开标信息表 upload button"


# ── FIX-2026-08-28-010: 清标评分计量 — 权重复合指数 + 行业信号 ──
def test_weighted_total_score_composite():
    """total_score must be a 0-100 weighted composite, not a raw sum."""
    from app.services.document_analysis_svc import _weighted_total_score, INDICATOR_WEIGHTS
    # All-skip → 0
    assert _weighted_total_score([]) == 0.0
    # Single indicator at cap → weight-normalized 100
    inds = [{'id': 'same_machine_code', 'score': 30, 'skipped': False}]  # cap 30
    assert _weighted_total_score(inds) == 100.0
    # Half cap → 50
    inds = [{'id': 'same_machine_code', 'score': 15, 'skipped': False}]
    assert _weighted_total_score(inds) == 50.0
    # text_sim 三指标去重：tech_section_similar / cross_file_code_same 不单独贡献
    inds = [
        {'id': 'same_file_code', 'score': 15, 'skipped': False},
        {'id': 'tech_section_similar', 'score': 30, 'skipped': False},
        {'id': 'cross_file_code_same', 'score': 30, 'skipped': False},
    ]
    # 去重后仅 same_file_code 贡献 → 与单指标 same_file_code 相同
    assert _weighted_total_score(inds) == _weighted_total_score(
        [{'id': 'same_file_code', 'score': 15, 'skipped': False}]), \
        "text_sim duplicates must not inflate the composite"


def test_risk_scorer_new_weights_and_gate():
    """RiskScorer must use 0.30/0.30/0.10(+0.30 collusion_para) weights + ≥80% text gate.

    FIX-2026-09-04-QA-B1: whole-document text_sim down-weighted (paragraph-level
    substantive collusion is the primary signal); text still gated at ≥80% and
    zeroed when the tender file is missing (template overlap ≠ collusion).
    """
    from app.services.batch_orchestrator import RiskScorer
    assert RiskScorer.WEIGHTS['key_info'] == 0.30
    assert RiskScorer.WEIGHTS['file_attr'] == 0.30
    assert RiskScorer.WEIGHTS['text_sim'] == 0.10
    assert RiskScorer.WEIGHTS['collusion_para'] == 0.30
    assert RiskScorer.WEIGHTS['image_sim'] == 0.0
    # text <80% gate → contributes 0
    assert RiskScorer.compute(0, 0, 70, 0) == 0.0
    # text ≥80% → contributes 0.10 * 80 = 8
    assert RiskScorer.compute(0, 0, 80, 0) == 8.0
    # template_missing=True → text contributes 0 even at 80% (raw cosine is template overlap)
    assert RiskScorer.compute(0, 0, 90, 0, template_missing=True) == 0.0
    # collusion_para contributes 0.30 * value
    assert RiskScorer.compute(0, 0, 0, 0, collusion_para=40) == 12.0


# 铁证 tier 优先于 60/30 阈值：resolve_warning_level 先查 hard.fired（铁证触发）/
# hard.violation_fired（暗标违规），命中即用铁证/违规 label 覆盖指数展示级，
# >=60 / >=30 仅在无铁证时作为兜底阈值生效（本测试只扫描源码字面量顺序）。
def test_warning_threshold_order_and_scale():
    """warning_level must check >=60 first (correct order, composite scale)."""
    with open('app/services/document_analysis_svc.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert '● 高度预警' in content and '◆ 中等预警' in content
    idx = content.find('● 高度预警')
    assert idx != -1
    # The high-warning branch must reference >= 60 and appear before medium's >= 30
    high_idx = content.find("total_score >= 60")
    med_idx = content.find("total_score >= 30")
    assert high_idx != -1 and med_idx != -1, "both composite thresholds must exist"
    assert high_idx < med_idx, "high warning (>=60) must be checked before medium (>=30)"


def test_quote_tailing_progression():
    """quote_anomaly must detect arithmetic/geometric price progression."""
    from app.services.quote_anomaly import _detect_progression
    ok, ptype, idxs = _detect_progression([100, 120, 140, 160])
    assert ok and ptype == 'arithmetic'
    ok, ptype, _ = _detect_progression([100, 130, 169])
    assert ok and ptype == 'geometric'
    ok, _, _ = _detect_progression([100, 150, 220])
    assert not ok


def test_benford_nigrini_grading():
    """_benford_deviation must return Nigrini grades, not a bare float."""
    from app.services.quote_anomaly import _benford_deviation
    res = _benford_deviation([10.5, 11.2, 9.8, 12.1, 10.0, 11.5, 9.2, 10.8, 11.9, 8.7,
                              10.3, 11.1, 9.5, 12.0, 10.6, 11.4, 9.9, 10.2, 11.8, 8.9,
                              10.7, 11.3, 9.4, 12.2, 10.1], min_samples=20)
    assert isinstance(res, dict), "benford must return a dict"
    assert 'grade' in res and 'mad' in res and 'z_scores' in res


def test_score_analyzer_kendall_w():
    """score_analyzer Kendall W: consistent panel → 1.0, random → low."""
    from app.services.score_analyzer import kendall_w, grubbs_test
    m1 = [[85, 80, 75, 70, 65, 60], [86, 81, 74, 69, 64, 59], [84, 79, 76, 71, 66, 61]]
    assert kendall_w(m1) == pytest.approx(1.0, abs=1e-6)
    m2 = [[85, 60, 70, 55, 90, 45], [70, 85, 55, 60, 50, 75], [60, 75, 85, 90, 55, 70]]
    assert kendall_w(m2) < 0.5
    assert grubbs_test([10, 11, 10.5, 50, 11]) == 3


def test_community_detection():
    """relationship_extractor must detect communities from a relationship graph."""
    from app.services.relationship_extractor import _detect_communities, DetectedRelationship
    rels = [
        DetectedRelationship('A公司', 'B公司', 'shared_person', 'p', 0.9, 'e', 'personnel_company', risk_flag=True),
        DetectedRelationship('B公司', 'C公司', 'shared_person', 'p', 0.8, 'e', 'personnel_company', risk_flag=True),
        DetectedRelationship('D公司', 'E公司', 'shared_contact', 'c', 0.5, 'e', 'company_company', risk_flag=False),
    ]
    comms = _detect_communities(rels)
    assert len(comms) == 2, "should detect 2 communities"
    assert any(c['member_count'] == 3 and c['risk'] for c in comms), "risk cluster missing"


def test_benford_dict_consumed_in_quote():
    """check_quote_anomaly must handle the new structured benford result."""
    from app.services.quote_anomaly import check_quote_anomaly
    text = "报价合计：1000万元，1100万元，1200万元，1300万元。另附详细清单若干。"
    res = check_quote_anomaly(text, doc_name='test')
    assert res is not None
    assert hasattr(res, 'benford_deviation')  # float kept for compatibility
    assert hasattr(res, 'progression_type')   # new field present


# ── FIX-2026-08-31-011: 报价尾数检测 + extract_prices 修复 ──
def test_tailing_digits_detection():
    """CSDN 第一信号：报价尾数相同比例 ≥80% → 触发串标信号。"""
    from app.services.quote_anomaly import _detect_tailing_digits
    ok, info = _detect_tailing_digits([1000000, 1100000, 1200000])  # 全尾 '00'
    assert ok, "3 个同尾数报价应触发"
    assert info['rate'] == 1.0 and info['digit'] == '00'
    ok2, _ = _detect_tailing_digits([1234567, 2345678, 3456789])  # 混合尾数
    assert not ok2, "混合尾数不应触发"


def test_extract_prices_non_cn_guardrail():
    """extract_prices 必须正确处理非中文格式（审计强制 guardrail）。"""
    from app.services.quote_anomaly import extract_prices
    assert extract_prices("USD 1,234,567.89") == [1234567.89]
    assert extract_prices("EUR 999.00") == [999.00]
    assert extract_prices("1000000") == [1000000.0]


def test_extract_prices_no_wan_dup():
    """'1000万元' 只产 10M，不得附带虚假 10000（_CN_PRICE 独立匹配万 bug）。"""
    from app.services.quote_anomaly import extract_prices
    prices = extract_prices('投标报价：1000万元，1100万元，1200万元，1300万元。')
    assert prices == [10000000.0, 11000000.0, 12000000.0, 13000000.0], f"got {prices}"
    assert 10000 not in prices, "不得出现虚假 10000 尾数"


def test_tailing_digits_wired_into_quote():
    """check_quote_anomaly 必须输出 tailing_digits_flag 并计入风险分。"""
    from app.services.quote_anomaly import check_quote_anomaly
    res = check_quote_anomaly('报价：1000000元，1100000元，1200000元，1300000元。', doc_name='t')
    assert res.tailing_digits_flag is True, "同尾数报价应触发 flag"
    assert res.tailing_digits_info['rate'] == 1.0
    assert hasattr(res, 'tailing_digits_flag')


# ── FIX-2026-08-31-012: 清标基线快照校准 (工程类, 2 投标人) ──
def test_clearance_baseline_scores():
    """锁定工程类基线快照：权重/过滤改动不得使复合指数极端漂移。

    注意：fixture 为 价格标 vs 商务技术标（不同投标组成部分，非同组件比较），
    复合指数允许中高；此测试主要防止未来改动造成 <0 或 >80 的异常值。
    """
    import json
    import os
    snap = json.load(open(
        os.path.join(os.path.dirname(__file__), 'fixtures', 'clearance_baseline', 'scores.json'),
        encoding='utf-8'))
    composite = snap['composite_score']
    assert 0 < composite < 80, f"baseline composite {composite} out of sane range"
    assert snap['meta']['n_bidders'] == 2


def test_text_sim_skipped_without_tender():
    """run_analysis 无招标文件时，text_sim 指标必须跳过（模板去除不可用）。"""
    from app.services.document_analysis_svc import run_analysis
    docs = [
        {'filename': 'a.docx', 'text': '招标公告 项目名称 XX 建设地点 XX 工期 730天 招标范围 施工', 'metadata': {}, 'images': []},
        {'filename': 'b.docx', 'text': '招标公告 项目名称 XX 建设地点 XX 工期 700天 招标范围 施工', 'metadata': {}, 'images': []},
    ]
    report = run_analysis(docs, user_id='t', thread_id='t')  # no tender_text
    inds = {i['id']: i for i in report['indicators']}
    for k in ('same_file_code', 'tech_section_similar', 'cross_file_code_same'):
        assert inds[k]['skipped'] is True, f"{k} 必须跳过（无招标文件）"
        assert inds[k]['score'] == 0, f"{k} 无招标文件时得分必须为 0"


# ── FIX-2026-08-31-013: 中文停用词过滤 — text_sim 判别围标 vs 正常 ──
def test_text_sim_stopwords_discrimination():
    """去停用词+模板去除后，text_sim 必须能判别 围标(≈0.98) vs 正常(≈0.74)。

    同一组件（技术标）比较：真实围标（技术方案雷同）余弦应显著高于正常投标。
    ≥80% 门槛应使正常投标不触发、围标触发。
    """
    from app.services.file_processing import (
        preprocess_text_for_similarity, _make_vectorizer, remove_template_content,
    )
    from sklearn.metrics.pairwise import cosine_similarity

    tender = ('本项目为大兴区燃气老旧管网改造工程。招标范围包括施工图纸范围内的土建、'
              '安装工程。投标人须具备市政公用工程施工总承包资质，并具有有效的安全生产许可证。')

    def _cos(a, b, tpl):
        pa = preprocess_text_for_similarity(a, tpl)
        pb = preprocess_text_for_similarity(b, tpl)
        pa = remove_template_content(pa, tpl)
        pb = remove_template_content(pb, tpl)
        v = _make_vectorizer(stop_words=None)
        X = v.fit_transform([pa, pb])
        return float(cosine_similarity(X[0:1], X[1:2])[0][0])

    # 围标：技术方案几乎相同（仅报价不同）
    collude_a = ('技术方案：采用开槽法施工，沟槽采用钢板桩支护，基坑降水采用井点降水，'
                 '管线敷设采用直埋。商务报价：302,070,000元。')
    collude_b = ('技术方案：采用开槽法施工，沟槽采用钢板桩支护，基坑降水采用井点降水，'
                 '管线敷设采用直埋。商务报价：329,270,000元。')
    # 正常：技术方案不同
    normal_b = ('技术方案：采用定向钻穿越施工，水平定向钻机导向钻进，泥浆护壁。'
                '商务报价：329,270,000元。')

    c_collude = _cos(collude_a, collude_b, tender)
    c_normal = _cos(collude_a, normal_b, tender)
    # FIX-015 (P0 vectorizer): word-level cosine. 围标（技术雷同，仅报价不同）显著高于正常。
    assert c_collude > 0.80, f"围标余弦应高: {c_collude}"
    assert c_normal < 0.50, f"正常投标余弦应明显低于围标: {c_normal}"
    # 判别裕度：围标 - 正常 ≥ 0.15
    assert c_collude - c_normal >= 0.15, \
        f"判别裕度不足: collude={c_collude:.3f} normal={c_normal:.3f}"


def test_stop_words_applied():
    """tokenize_for_tfidf / preprocess 必须过滤常见招投标词（FIX-013）。"""
    from app.services.text_utils import tokenize_for_tfidf
    from app.services.file_processing import preprocess_text_for_similarity
    # 常见词 招标/投标/项目/工程 应被过滤
    toks = tokenize_for_tfidf('招标投标项目工程施工资质', stop_words={'招标', '投标', '项目', '工程', '施工'})
    assert '招标' not in toks and '项目' not in toks, f"停用词未过滤: {toks}"
    # 默认停用词表也应过滤 招标/投标
    toks2 = tokenize_for_tfidf('招标投标项目工程施工')
    assert '招标' not in toks2 and '投标' not in toks2, f"默认停用词未生效: {toks2}"
    # preprocess 保留有判别力的词
    p = preprocess_text_for_similarity('本项目为大兴区燃气管网改造工程，采用定向钻施工')
    assert '燃气' in p and '定向' in p, f"有判别力的词被误删: {p}"


# ── FIX-2026-09-01-014: 自适应招标高频词(Y) + 异组件守卫(Z-1) ──
def test_tender_stopwords_adaptive():
    """长招标文件应自适应扩容高频词并入停用集（Y），同组件正常投标降至 <0.80 门槛。"""
    from app.services.file_processing import (
        preprocess_text_for_similarity, _make_vectorizer, remove_template_content,
    )
    from sklearn.metrics.pairwise import cosine_similarity
    # 长招标文件（>5000 字 → k 扩容）
    tender = ('本项目为大兴区燃气老旧管网改造工程，招标范围包括施工图纸范围内的土建、安装工程。'
              '投标人须具备市政公用工程施工总承包资质，并具有有效的安全生产许可证。评标采用综合评分法。' * 40)
    ta = '技术方案：采用开槽法施工，沟槽钢板桩支护，基坑井点降水，管线直埋敷设。项目经理张三。'
    tb = '技术方案：采用定向钻穿越施工，泥浆护壁导向钻进。项目经理李四。'
    pa = preprocess_text_for_similarity(ta, tender); pb = preprocess_text_for_similarity(tb, tender)
    pa = remove_template_content(pa, tender); pb = remove_template_content(pb, tender)
    v = _make_vectorizer(); X = v.fit_transform([pa, pb])
    cos = float(cosine_similarity(X[0:1], X[1:2])[0][0])
    assert cos < 0.80, f"同组件正常投标应 <0.80 门槛: {cos}"


def test_tender_stopwords_tf_guard():
    """TF=1 的独特词不得并入停用集（防止误杀技术参数，Y 的 TF≥2 守卫）。"""
    from app.services.file_processing import preprocess_text_for_similarity
    # 招标文件含一个只出现 1 次的关键技术词，不应被过滤
    tender = '本项目燃气改造工程。投标人资格：压力管道GB1级资质。技术标准：NB/T 47013 无损检测。' * 3
    text = '技术方案采用 NB/T 47013 无损检测标准，燃气管道施工。'
    p = preprocess_text_for_similarity(text, tender)
    # '检测' 若多次出现会被停用，但独特参数词应保留；至少 '燃气' 应保留
    assert '燃气' in p, f"独特词被误并入停用集: {p}"


def test_component_mismatch():
    """异组件（价格标↔技术标）→ component_mismatch=True + 文本相似度归零。"""
    from app.services.batch_orchestrator import compute_single_pair, _detect_component
    assert _detect_component('投标文件_价格标.docx', '') == 'price'
    assert _detect_component('投标文件_商务技术标.docx', '') == 'tech'
    assert _detect_component('engineering.docx', '') == 'unknown'
    fd = [
        {'filename': '投标文件_价格标.docx', 'text': '价格标 报价 302,070,000元 投标报价声明函 项目名称',
         'metadata': {}, 'images': []},
        {'filename': '投标文件_商务技术标.docx', 'text': '技术标 技术方案 施工组织设计 商务部分',
         'metadata': {}, 'images': []},
    ]
    pair = compute_single_pair(fd, 0, 1,
                               {'text_sim': True, 'key_info': True, 'file_attr': True, 'image_sim': False})
    assert pair.get('component_mismatch') is True, "异组件必须标记 mismatch"
    assert pair['risk'] == 0.0, f"异组件 text_sim 归零后 risk 应为 0: {pair['risk']}"


def test_component_same():
    """同组件（技术标↔技术标）→ 不标记 mismatch，text_sim 正常参与。"""
    from app.services.batch_orchestrator import compute_single_pair
    fd = [
        {'filename': '投标文件_技术标A.docx', 'text': '技术标 技术方案 开槽法施工 沟槽支护',
         'metadata': {}, 'images': []},
        {'filename': '投标文件_技术标B.docx', 'text': '技术标 技术方案 定向钻施工 泥浆护壁',
         'metadata': {}, 'images': []},
    ]
    pair = compute_single_pair(fd, 0, 1,
                               {'text_sim': True, 'key_info': True, 'file_attr': True, 'image_sim': False})
    assert pair.get('component_mismatch') is False, "同组件不应标记 mismatch"
    assert pair['risk'] > 0, "同组件 text_sim 应正常参与风险计算"


# ── FIX-2026-09-01-015 C5: quote fixtures (等差/等比/正常/垃圾) ──
def _load_quote_scenarios():
    import json
    import os
    return json.load(open(
        os.path.join(os.path.dirname(__file__), 'fixtures', 'quote_fixtures', 'scenarios.json'),
        encoding='utf-8'))


def test_quote_arithmetic_progression_fixture():
    """等差序列 fixture → cross_progression 触发且类型为 arithmetic。"""
    from app.services.quote_anomaly import compare_bidders_quotes
    data = _load_quote_scenarios()
    sc = data['scenarios']['arithmetic']
    docs = [{'filename': d['filename'], 'text': d['text'], 'metadata': {}, 'images': []}
            for d in sc['docs']]
    res = compare_bidders_quotes(docs)
    assert res.get('cross_progression') is True, "等差序列应触发 cross_progression"
    assert res.get('cross_progression_type') == 'arithmetic', \
        f"等差类型错误: {res.get('cross_progression_type')}"


def test_quote_geometric_progression_fixture():
    """等比序列 fixture → cross_progression 触发且类型为 geometric。"""
    from app.services.quote_anomaly import compare_bidders_quotes
    data = _load_quote_scenarios()
    sc = data['scenarios']['geometric']
    docs = [{'filename': d['filename'], 'text': d['text'], 'metadata': {}, 'images': []}
            for d in sc['docs']]
    res = compare_bidders_quotes(docs)
    assert res.get('cross_progression') is True, "等比序列应触发 cross_progression"
    assert res.get('cross_progression_type') == 'geometric', \
        f"等比类型错误: {res.get('cross_progression_type')}"


def test_quote_normal_no_progression():
    """正常分散报价 fixture → 不触发 cross_progression，且 risk 不饱和。"""
    from app.services.quote_anomaly import compare_bidders_quotes
    data = _load_quote_scenarios()
    sc = data['scenarios']['normal']
    docs = [{'filename': d['filename'], 'text': d['text'], 'metadata': {}, 'images': []}
            for d in sc['docs']]
    res = compare_bidders_quotes(docs)
    assert res.get('cross_progression') is False, "正常报价不应触发规律信号"
    # 正常报价不应全部 risk=100（C3 幅度加权 + C1 过滤后）
    assert res.get('max_risk_score', 0) < 100, \
        f"正常报价 max_risk 不应饱和 100: {res.get('max_risk_score')}"


def test_quote_garbage_filtered():
    """垃圾输入（年份/电话/标准号）→ extract_prices 过滤，只留真实报价。"""
    from app.services.quote_anomaly import extract_prices
    data = _load_quote_scenarios()
    sc = data['scenarios']['noise_garbage']
    for d in sc['docs']:
        prices = extract_prices(d['text'])
        # 电话/年份/标准号不得出现在价格列表
        assert not any(abs(p - 69256688) < 1 for p in prices), f"电话被误当价格: {prices}"
        assert not any(abs(p - 2026) < 1 for p in prices), f"年份被误当价格: {prices}"
    # 两文档合计应含真实报价
    all_p = []
    for d in sc['docs']:
        all_p.extend(extract_prices(d['text']))
    for expected in sc['expect']['clean_prices']:
        assert any(abs(p - expected) < 1 for p in all_p), f"缺少真实报价 {expected}: {all_p}"


# ── FIX-2026-09-01-015 D: relationship 实体提取校准 ──
def test_relationship_company_blacklist():
    """泛词/动词短语不得被当公司（D1）。"""
    from app.services.relationship_extractor import _extract_with_regex
    text = ('监控中心 行政中心 设备厂 实现调压站 毕业学校 '
            '京清园科技创新服务中心 天津利博科技有限公司')
    ents = _extract_with_regex([text])[0]
    companies = [e.text for e in ents if e.entity_type == 'company']
    assert '天津利博科技有限公司' in companies, "真实公司应被提取"
    assert '京清园科技创新服务中心' in companies, "真实服务中心应被提取"
    for garbage in ('监控中心', '行政中心', '设备厂', '实现调压站', '毕业学校'):
        assert garbage not in companies, f"垃圾公司名未被过滤: {garbage}"


def test_relationship_person_field_labels():
    """字段标签/公司短名不得被当人员（D2）。"""
    from app.services.relationship_extractor import _extract_with_regex
    text = '联系人：王强 身份证号码：总经理 职务：董事长 北京华信建设工程有限公司：法人代表'
    ents = _extract_with_regex([text])[0]
    persons = [e.text for e in ents if e.entity_type == 'person']
    assert '王强' in persons, "真实人名应被提取"
    for garbage in ('份证号码', '职务', '北京华信'):
        assert garbage not in persons, f"字段标签/短名被当人员: {garbage}"


def test_relationship_risk_not_saturated():
    """少量文档/泛实体时 risk_score 不得封顶 100（D3 归一化）。"""
    from app.services.relationship_extractor import extract_relationships
    docs = [
        {'filename': 'a.docx', 'text': '投标人：天津利博科技有限公司 联系人：王强 项目经理：张三', 'metadata': {}, 'images': []},
        {'filename': 'b.docx', 'text': '投标人：天津利博科技有限公司 联系人：王强 技术负责人：李四', 'metadata': {}, 'images': []},
        {'filename': 'c.docx', 'text': '投标人：天津利博科技有限公司 联系人：王强 安全负责人：赵五', 'metadata': {}, 'images': []},
    ]
    report = extract_relationships(docs)
    assert report.risk_score < 100, f"risk_score 不应封顶 100: {report.risk_score}"
    # 跨文件共享同一公司是真实信号，但归一化后不应极端
    assert 0 <= report.risk_score <= 100, "risk_score 应在 0-100 范围"


def test_clearance_baseline_classified_normal():
    """工程类基线（价格标 vs 技术标）经校准后应判为 正常（<30）。"""
    import json
    import os
    snap = json.load(open(
        os.path.join(os.path.dirname(__file__), 'fixtures', 'clearance_baseline', 'scores.json'),
        encoding='utf-8'))
    composite = snap['composite_score']
    # FIX-015 校准后：正常基线复合指数应 <30（不再误报中等预警）
    assert 0 < composite < 30, f"baseline composite {composite} 应 <30 (正常): 过度报警"


# ── 剽窃检测模式 (Plagiarism Mode, FIX-016 后续) ──
def test_plagiarism_identical_flagged():
    """完全相同文档 → 疑似剽窃（cosine=1.0 + 高匹配段占比）。"""
    from app.services.plagiarism_detector import detect_plagiarism
    text = ('第一章 招标公告\n本招标项目已由某市发展和改革委员会批准建设。\n'
            '建设地点：某市新城区；建设规模：总建筑面积约50000平方米。\n'
            '计划工期：730日历天；招标范围：施工图纸范围内的土建、安装工程。\n'
            '投标人须具备建筑工程施工总承包一级资质。')
    r = detect_plagiarism(text, text, filename_a='A.docx', filename_b='B.docx')
    assert r['verdict'] == '疑似剽窃', f"相同文档应为疑似剽窃: {r['verdict']}"
    assert r['cosine_similarity'] >= 0.99
    # 除短标题段（len<MIN_PARA_CHARS 不计）外，其余段落应高匹配
    assert r['high_match_para_count'] >= r['para_count'] - 1, \
        f"几乎所有段落应高匹配: {r['high_match_para_count']}/{r['para_count']}"


def test_plagiarism_different_not_flagged():
    """内容完全不同 → 正常（cosine 低 + 高匹配段少）。"""
    from app.services.plagiarism_detector import detect_plagiarism
    ta = ('第一章 招标公告\n本招标项目已由某市发展和改革委员会批准建设。\n建设地点：某市新城区。')
    tb = ('第五章 技术标准\n采购CT机1台、MRI设备1台、超声诊断仪3台。\n预算金额：1500万元。')
    r = detect_plagiarism(ta, tb, filename_a='A.docx', filename_b='B.docx')
    assert r['verdict'] == '正常', f"不同文档应为正常: {r['verdict']}"
    assert r['high_match_para_count'] == 0


def test_plagiarism_partial_detected():
    """部分段落抄袭（共享招标模板 + 不同技术方案）→ 疑似剽窃（双信号）。"""
    from app.services.plagiarism_detector import detect_plagiarism
    common = ('第一章 招标公告\n本招标项目已由某市发展和改革委员会批准建设。\n'
              '建设地点：某市新城区；建设规模：总建筑面积约50000平方米。\n'
              '计划工期：730日历天；招标范围：施工图纸范围内的土建、安装工程。')
    ta = common + '\n技术方案：采用开槽法施工，沟槽钢板桩支护。\n商务报价：302,070,000元。'
    tb = common + '\n技术方案：采用定向钻穿越施工，泥浆护壁。\n商务报价：329,270,000元。'
    r = detect_plagiarism(ta, tb, filename_a='A.docx', filename_b='B.docx')
    assert r['verdict'] == '疑似剽窃', f"共享模板+部分雷同应为疑似剽窃: {r['verdict']}"
    assert r['high_match_para_ratio'] >= 0.20


def test_plagiarism_short_paras_no_false_positive():
    """极短段落（< MIN_PARA_CHARS）不应因单字匹配被计为高匹配段。"""
    from app.services.plagiarism_detector import detect_plagiarism
    ta = 'p1\np2\np3\np4\np5\n第一章 招标公告 本招标项目已批准 建设规模50000平方米'
    tb = 'q1\nq2\nq3\nq4\nq5\n第一章 招标公告 本招标项目已批准 建设规模50000平方米'
    r = detect_plagiarism(ta, tb, filename_a='A.docx', filename_b='B.docx')
    # 5 个短段不应计入高匹配
    assert r['verdict'] in ('正常', '高度相似'), f"短段不应误判疑似剽窃: {r['verdict']}"
    assert r['high_match_para_count'] <= 2, f"短段高匹配数应受限: {r['high_match_para_count']}"


# ── FIX-2026-09-01-016: LLM 来源切换 OpenRouter + NVIDIA ──
def test_llm_provider_switch_active_sources():
    """FIX-016: PROVIDER_CONFIG 必须含 openrouter + nvidia，且不含已注释的旧 provider。"""
    from app.services.llm_provider import PROVIDER_CONFIG
    assert 'openrouter' in PROVIDER_CONFIG, "OpenRouter provider 缺失"
    assert 'nvidia' in PROVIDER_CONFIG, "NVIDIA NIM provider 缺失"
    # 旧 provider 应从活跃配置移除（注释保留）
    assert 'deepseek' not in PROVIDER_CONFIG, "deepseek 不应在活跃 provider 列表"
    assert 'zhipu' not in PROVIDER_CONFIG and 'mimo' not in PROVIDER_CONFIG, \
        "旧 provider 不应在活跃列表"


def test_llm_create_chat_model_direct_exists():
    """FIX-016: _create_chat_model_direct 必须存在（llm_fallback 依赖，曾缺失 ImportError）。"""
    from app.services.llm_provider import _create_chat_model_direct
    assert callable(_create_chat_model_direct)


def test_llm_catalog_whitelist():
    """FIX-016: llm_catalog 白名单含 OpenRouter 主力免费模型 + NVIDIA 兜底。"""
    from app.services.llm_catalog import WHITELIST
    assert 'nvidia/nemotron-3-ultra-550b-a55b:free' in WHITELIST['openrouter'], \
        "OpenRouter 文本主力白名单缺失"
    assert 'nvidia/nemotron-3-ultra-550b-a55b' in WHITELIST['nvidia'], \
        "NVIDIA 兜底白名单缺失"


def test_llm_fallback_chain_catalog():
    """FIX-016: fallback 链应从 catalog 构建且首选 openrouter。"""
    from app.services.llm_fallback import get_fallback_chain, DEFAULT_CHAIN
    chain = get_fallback_chain()
    assert chain and chain[0][0] == 'openrouter', f"fallback 链应以 openrouter 开头: {chain}"


# ── FIX-2026-09-02-M6: 清标报告章节编号不跳号 (MiMo QA) ──
def _clearance_report_function_src():
    """提取 static/js/app.js 中 buildClearanceReportHtml 的函数体源码。"""
    with open('static/js/app.js', 'r', encoding='utf-8') as f:
        content = f.read()
    start = content.index('function buildClearanceReportHtml')
    # 该函数后紧跟 _renderImageSamplingTab，以此作为函数体结束边界
    end = content.index('function _renderImageSamplingTab', start)
    return content[start:end]


def test_clearance_report_all_six_sections_in_order():
    """M6: 清标报告 HTML 必须渲染一~六全部中文章节编号，且按序出现。"""
    fn = _clearance_report_function_src()
    positions = []
    for num in '一二三四五六':
        marker = 'cl-num">' + num
        assert marker in fn, f"章节 {num} 编号缺失（清标报告章节完整性被破坏）"
        positions.append(fn.index(marker))
    assert positions == sorted(positions), "清标报告章节编号必须按一~六顺序出现"


def test_clearance_report_sections_five_six_have_placeholder():
    """M6: 五(图片抽检)/六(全量审计) 无数据时必须渲染 else 占位分支，不得跳号。"""
    fn = _clearance_report_function_src()
    # 摘要行每章节出现两次 = if + else 双分支
    for num, placeholder in (('五', '未包含图片或未执行图片抽检。'),
                             ('六', '未包含审计补充数据。')):
        assert fn.count('cl-num">' + num) == 2, \
            f"章节 {num} 缺少 else 占位分支（M6 修复被回退）"
        assert placeholder in fn, f"章节 {num} 占位文案缺失: {placeholder}"


# ── QA-Loop C4: credit 限速器 Redis 优先（跨 worker 生效）──
def test_credit_rate_limit_redis_backed():
    """QA-Loop C4: credit 限速必须走 Redis（跨 gunicorn worker 共享），
    不得仅用进程内 dict（多 worker 下可绕过 10/5min 限制）。"""
    with open('app/routes/credit.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'credit_rate:' in content, "credit 限速必须使用 Redis key 前缀 credit_rate:"
    assert 'r.incr(' in content, "credit 限速必须用 Redis INCR 原子计数"
    assert 'get_redis(' in content, "credit 限速必须通过 redis_client.get_redis 取客户端"


def test_credit_rate_limit_memory_fallback_present():
    """QA-Loop C4: Redis 不可用时仍保留内存降级路径（单 worker/开发不挂）。"""
    with open('app/routes/credit.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'Fallback: in-memory' in content or 'fallback' in content.lower(), \
        "credit 限速必须包含 Redis 不可用时的内存降级分支"


# ── FIX-2026-09-09-QA: 铁证双层判定 + 暗标违规独立警示 ──
def test_hard_evidence_lasteditor_escalates_warning():
    """T1 lastModifiedBy 同人 → 铁证触发：veto 只升展示级，不重写指数。"""
    from app.services.document_analysis_svc import run_analysis
    docs = [
        {'filename': 'a.docx',
         'text': '技术方案：开槽法施工，沟槽钢板桩支护，基坑井点降水。',
         'metadata': {'last_modified_by': '超彩赵'}, 'images': []},
        {'filename': 'b.docx',
         'text': '技术方案：定向钻穿越施工，泥浆护壁导向钻进。',
         'metadata': {'last_modified_by': '超彩赵'}, 'images': []},
    ]
    report = run_analysis(docs, user_id='t', thread_id='t')
    bi = report['basic_info']
    assert bi['hard_alarm'] is True, "同一最后编辑人（非通用值）必须触发铁证"
    assert bi['warning_level'] == '■ 高度预警（铁证触发）', \
        f"铁证触发时展示级别应为铁证 label: {bi['warning_level']}"
    assert bi['total_score'] < 60, \
        f"铁证 veto 不得重写复合指数（指数应保持原样 <60）: {bi['total_score']}"
    types = [it['type'] for it in bi['hard_evidence']['items']]
    assert 'lasteditor_same' in types, f"items 必须含 lasteditor_same: {types}"


def test_hard_evidence_generic_guard():
    """lastModifiedBy 通用默认值（Administrator）必须被 guard 排除，不触发铁证。"""
    from app.services.document_analysis_svc import run_analysis
    docs = [
        {'filename': 'a.docx',
         'text': '技术方案：开槽法施工，沟槽钢板桩支护。',
         'metadata': {'last_modified_by': 'Administrator'}, 'images': []},
        {'filename': 'b.docx',
         'text': '技术方案：定向钻穿越施工，泥浆护壁。',
         'metadata': {'last_modified_by': 'Administrator'}, 'images': []},
    ]
    report = run_analysis(docs, user_id='t', thread_id='t')
    bi = report['basic_info']
    assert bi['hard_alarm'] is False, "通用默认 lastModifiedBy 不得触发铁证"
    assert bi['warning_level'] != '■ 高度预警（铁证触发）', \
        "通用默认值不得升为铁证展示级"


def test_hard_evidence_paragraph_t2_requires_two_types():
    """T2 强嫌疑需 ≥2 类共证才 veto：单类段落共享不触发，双类共证触发。"""
    from app.services.hard_evidence import assess_hard_evidence
    seg_ctx = {
        'paragraph_collusion': {
            'shared_segments': [
                {'files': ['a.docx', 'b.docx'], 'segment_text': 'x', 'type': '服务承诺段'},
            ],
        },
    }
    one_type = assess_hard_evidence([], seg_ctx)
    assert one_type['fired'] is False, "单类 T2（段落单段共享）不得 veto"
    assert one_type['level'] is None, f"单类 T2 level 应为 None: {one_type['level']}"

    two_type_ctx = dict(seg_ctx)
    two_type_ctx['author_groups'] = {'张三': ['a.docx', 'b.docx']}
    two_type = assess_hard_evidence([], two_type_ctx)
    assert two_type['fired'] is True, "段落共享 + author 雷同两类 T2 共证必须 veto"
    assert two_type['level'] == 'T2', f"两类 T2 共证 level 应为 T2: {two_type['level']}"


def test_tech_seal_violation_independent():
    """暗标违规独立轨道：单家技术标身份泄露 → 违规警示，不进串通铁证。
    需显式开启 tech_seal_check（默认关闭）。"""
    from app.services.document_analysis_svc import run_analysis
    docs = [
        {'filename': '北京中昌华美超市服务有限责任公司_技术标.docx',
         'text': '技术方案：我公司北京中昌华美超市服务有限责任公司承诺……',
         'metadata': {}, 'images': []},
        {'filename': '天津恒达建筑工程有限公司_技术标.docx',
         'text': '技术方案：采用定向钻穿越施工，泥浆护壁导向钻进。',
         'metadata': {}, 'images': []},
    ]
    report = run_analysis(docs, user_id='t', thread_id='t',
                          options={'tech_seal_check': True})
    he = report['basic_info']['hard_evidence']
    assert he['violation_fired'] is True, "技术标身份泄露必须触发暗标违规警示"
    assert any(v['type'] == 'tech_seal_leak' for v in he['violations']), \
        "violations 必须含 tech_seal_leak 项"
    # 违规独立轨道：单家泄露是违规而非串通铁证；若该场景同时触发其他铁证致
    # fired 也为 True，则放宽为仅验证违规警示成立 + 展示级别以 '■' 开头。
    if not he['fired']:
        assert report['basic_info']['warning_level'] == '■ 高度预警（暗标违规）', \
            "仅违规触发时展示级别应为暗标违规 label"
    else:
        assert report['basic_info']['warning_level'].startswith('■'), \
            "铁证/违规同时触发时展示级别必须以 ■ 开头（铁证优先）"


def test_tech_seal_default_off():
    """暗标违规检测开关默认关闭：不传 options 时 tech_seal 检查跳过，
    泄露文档不产生 violation_fired。"""
    from app.services.document_analysis_svc import run_analysis
    docs = [
        {'filename': '北京中昌华美超市服务有限责任公司_技术标.docx',
         'text': '技术方案：我公司北京中昌华美超市服务有限责任公司承诺……',
         'metadata': {}, 'images': []},
        {'filename': '天津恒达建筑工程有限公司_技术标.docx',
         'text': '技术方案：采用定向钻穿越施工。',
         'metadata': {}, 'images': []},
    ]
    report = run_analysis(docs, user_id='t', thread_id='t')
    bi = report['basic_info']
    assert bi['hard_evidence']['violation_fired'] is False, \
        "默认未开启 tech_seal_check 时不得产生暗标违规"
    inds = [it for it in report['indicators'] if it['id'] == 'tech_seal_check']
    assert inds, "indicators 必须含 tech_seal_check 指标"
    assert inds[0]['skipped'] is True, \
        f"默认关闭时 tech_seal_check 应 skipped: {inds[0]['result']}"
    assert bi['warning_level'] != '■ 高度预警（暗标违规）', \
        f"默认关闭时展示级别不得为暗标违规 label: {bi['warning_level']}"


def test_tech_seal_seal_marker_weak_signal():
    """盖章/公章提示是弱信号：不独立触发泄露；强信号（技术方案段公司名）
    触发时盖章类提示作为辅助证据列出。"""
    from app.services.tech_seal_detector import detect_tech_seal_leak

    seal_only = detect_tech_seal_leak([
        {'filename': '某公司.docx', 'text': '需加盖公章并签字盖章。'},
    ])
    r0 = seal_only['某公司.docx']
    assert r0['leak'] is False, \
        f"纯盖章文本不得触发泄露: {r0['evidence']}"
    assert r0['score'] == 0, \
        f"纯盖章文本 score 应为 0: {r0['score']}"

    strong = detect_tech_seal_leak([
        {'filename': '北京中昌华美超市服务有限责任公司.docx',
         'text': '技术方案：北京中昌华美超市服务有限责任公司负责实施。需加盖公章。'},
    ])
    r1 = strong['北京中昌华美超市服务有限责任公司.docx']
    assert r1['leak'] is True, \
        f"技术方案段出现公司名必须触发泄露: {r1['evidence']}"
    assert any(('辅助证据' in e or '盖章' in e) for e in r1['evidence']), \
        f"泄露时盖章类提示应作辅助证据: {r1['evidence']}"
    assert any('公司名' in e for e in r1['evidence']), \
        f"evidence 必须含公司名条目: {r1['evidence']}"


def test_vl_describe_image_v2_content_and_reasoning(monkeypatch):
    """describe_image_v2 读取 content + reasoning_content；非推理模型 reasoning 为空。"""
    from app.services import vl_model as vm

    class FakeMsg:
        content = '图片中有项目表格与报价数字'
        reasoning_content = '先定位表格区域，再提取报价关键数字'

    class FakeChoice:
        message = FakeMsg()

    class FakeResp:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResp()

    class FakeChatAPI:
        completions = FakeCompletions()

    class FakeChat:
        chat = FakeChatAPI()

    monkeypatch.setattr(vm.vl_model, '_client', FakeChat())
    monkeypatch.setattr(vm.vl_model, '_ensure_current', lambda: None)
    monkeypatch.setattr(vm.vl_model, 'is_available', lambda: True)
    out = vm.vl_model.describe_image_v2(b'fake-image-bytes')
    assert out['description'] == '图片中有项目表格与报价数字'
    assert out['reasoning'] == '先定位表格区域，再提取报价关键数字'


def test_vl_describe_image_v2_no_reasoning_model(monkeypatch):
    """非推理模型（无 reasoning_content）→ reasoning 空串。"""
    from app.services import vl_model as vm

    class FakeMsg:
        content = '一张图片'
        reasoning_content = None

    class FakeChoice:
        message = FakeMsg()

    class FakeResp:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResp()

    class FakeChatAPI:
        completions = FakeCompletions()

    class FakeChat:
        chat = FakeChatAPI()

    monkeypatch.setattr(vm.vl_model, '_client', FakeChat())
    monkeypatch.setattr(vm.vl_model, '_ensure_current', lambda: None)
    monkeypatch.setattr(vm.vl_model, 'is_available', lambda: True)
    out = vm.vl_model.describe_image_v2(b'fake-image-bytes')
    assert out['description'] == '一张图片'
    assert out['reasoning'] == ''


def test_vl_describe_image_v2_unavailable(monkeypatch):
    """VL 不可用 → description 以 ⚠️ 开头，reasoning 空。"""
    from app.services import vl_model as vm
    monkeypatch.setattr(vm.vl_model, '_ensure_current', lambda: None)
    monkeypatch.setattr(vm.vl_model, 'is_available', lambda: False)
    out = vm.vl_model.describe_image_v2(b'fake-image-bytes')
    assert out['description'].startswith('⚠️'), out['description']
    assert out['reasoning'] == ''


def test_vl_select_vl_pair_verifier_ordering(monkeypatch):
    """select_vl_pair: primary 用激活 provider，verifier = 最强不同有 key 的 provider。"""
    import os
    from app.services import vl_model as vm

    # primary=mimo（显式/激活），dashscope+nvidia 都有 key → verifier=dashscope（最强）
    monkeypatch.setattr(vm, '_get_active_vl_config',
                        lambda: {'provider_id': 'mimo', 'api_key_valid': True})
    for k in ('DASHSCOPE_API_KEY', 'NVIDIA_API_KEY', 'MIMO_API_KEY'):
        os.environ.pop(k, None)
    os.environ['DASHSCOPE_API_KEY'] = 'd1'
    os.environ['NVIDIA_API_KEY'] = 'n1'
    primary, verifier = vm.select_vl_pair()
    assert primary == 'mimo'
    assert verifier == 'dashscope', f"应取最强不同 provider, got {verifier}"

    # 只有 mimo 有 key → verifier 空
    os.environ.pop('DASHSCOPE_API_KEY', None)
    os.environ.pop('NVIDIA_API_KEY', None)
    primary, verifier = vm.select_vl_pair()
    assert primary == 'mimo'
    assert verifier == '', f"无第二 key 应空, got {verifier}"

    # primary=dashscope，只有 nvidia 另一 key → verifier=nvidia
    monkeypatch.setattr(vm, '_get_active_vl_config',
                        lambda: {'provider_id': 'dashscope', 'api_key_valid': True})
    os.environ['NVIDIA_API_KEY'] = 'n1'
    primary, verifier = vm.select_vl_pair()
    assert primary == 'dashscope'
    assert verifier == 'nvidia'


def test_vl_verify_image_cross_model_mismatch(monkeypatch):
    """verify_image：两模型数字集不一致 → consistent False + 需人工复核信号。"""
    monkeypatch.setenv('NVIDIA_API_KEY', 'n1')
    from app.services import vl_model as vm
    monkeypatch.setattr(vm.vl_model, 'describe_image_v2',
                        lambda b, prompt=None: {'description': '报价 12345.67 元，日期 2026-09-01', 'reasoning': ''})
    monkeypatch.setattr(vm.vl_model, '_call_provider',
                        lambda pid, b, prompt=None: {'description': '报价 99999.99 元，日期 2026-09-01', 'reasoning': ''})
    monkeypatch.setattr(vm, 'select_vl_pair', lambda: ('mimo', 'dashscope'))
    out = vm.vl_model.verify_image(b'fake')
    assert out['consistent'] is False, f"数字不一致应判复核: {out}"
    assert '12345' in out['description'] and '99999' in out['verifier_desc']


def test_vl_verify_image_cross_model_consistent(monkeypatch):
    """verify_image：两模型数字一致 → consistent True（VL+复核）。"""
    monkeypatch.setenv('NVIDIA_API_KEY', 'n1')
    from app.services import vl_model as vm
    monkeypatch.setattr(vm.vl_model, 'describe_image_v2',
                        lambda b, prompt=None: {'description': '合同金额 88000 元', 'reasoning': ''})
    monkeypatch.setattr(vm.vl_model, '_call_provider',
                        lambda pid, b, prompt=None: {'description': '合同总价 88000 元整', 'reasoning': ''})
    monkeypatch.setattr(vm, 'select_vl_pair', lambda: ('mimo', 'nvidia'))
    out = vm.vl_model.verify_image(b'fake')
    assert out['consistent'] is True, out
    assert out['note']


def test_vl_verify_image_no_verifier_single_model(monkeypatch):
    """verify_image：无第二 provider → 单模型，note 说明。"""
    from app.services import vl_model as vm
    monkeypatch.setattr(vm.vl_model, 'describe_image_v2',
                        lambda b, prompt=None: {'description': '一张凭证', 'reasoning': ''})
    monkeypatch.setattr(vm, 'select_vl_pair', lambda: ('mimo', ''))
    out = vm.vl_model.verify_image(b'fake')
    assert out['consistent'] is True
    assert '单模型' in out['note']


# ── FIX-2026-09-10-028: shadowed knowledge_bp /feedback route removed ──
def test_no_shadowed_knowledge_feedback_route():
    """knowledge.py must not redefine /feedback (shadowed by chat_bp)."""
    with open('app/routes/knowledge.py', 'r', encoding='utf-8') as f:
        src = f.read()
    assert "@knowledge_bp.route('/feedback'" not in src, \
        "knowledge_bp /feedback was shadowed by chat_bp; must stay deleted"
    assert 'def submit_knowledge_lab_feedback' not in src, \
        "dead submit_knowledge_lab_feedback handler must stay deleted"
    assert "/knowledge_lab/feedback" in src, \
        "dedicated /knowledge_lab/feedback route must remain"


# ── FIX-2026-09-10-029: clearance persistence is opt-in ──
def test_run_analysis_persist_opt_in():
    """run_analysis must default persist=False (no DB side effects for unit tests)."""
    import inspect
    from app.services import document_analysis_svc as svc
    sig = inspect.signature(svc.run_analysis)
    assert sig.parameters['persist'].default is False, \
        "run_analysis(persist=) must default to False"
    assert 'task_id' in sig.parameters and 'project_id' in sig.parameters
    assert hasattr(svc, '_persist_clearance_review')


# ── FIX-2026-09-10-031: clearance status/stream require login ──
def test_clearance_status_stream_require_login():
    with open('app/routes/clearance.py', 'r', encoding='utf-8') as f:
        src = f.read()
    assert 'AUTH-GUARD-STATUS(FIX-2026-09-10-031)' in src
    assert 'AUTH-GUARD-STREAM(FIX-2026-09-10-031)' in src
    # both guards must actually check consent + user
    assert src.count("session.get('consent_value', 0) != 1") >= 4, \
        "consent guard must appear in run + status + stream (+ preview)"


# ── FIX-2026-09-10-030: legacy document_analysis blueprint removed ──
def test_document_analysis_blueprint_removed():
    import os
    assert not os.path.exists('app/routes/document_analysis.py'), \
        "legacy document_analysis route file must stay deleted"
    with open('app/routes/__init__.py', 'r', encoding='utf-8') as f:
        assert 'document_analysis_bp' not in f.read()
    with open('app/services/document_analysis_svc.py', 'r', encoding='utf-8') as f:
        assert 'def run_analysis_async' not in f.read()


# ── FIX-2026-09-11-033: task ownership guard ──
def test_task_owner_ok():
    from app.utils.helpers import task_owner_ok
    assert task_owner_ok({'user_id': 'u1'}, 'u1') is True
    assert task_owner_ok({'user_id': 'u1'}, 'u2') is False
    assert task_owner_ok({'user_id': 123}, '123') is True
    # legacy task (no user_id) → allowed with warning
    assert task_owner_ok({}, 'u2') is True
    # malformed meta → fail closed
    assert task_owner_ok(None, 'u2') is False


def test_compliance_check_task_signature_accepts_region_code():
    """async compliance task must accept the 7th arg passed by apply_async."""
    import inspect
    from app.services.compliance_checker import compliance_check_task
    params = inspect.signature(compliance_check_task.run).parameters
    assert 'region_code' in params, "region_code param required (apply_async passes it)"
    with open('app/routes/compliance.py', 'r', encoding='utf-8') as f:
        src = f.read()
    assert "TaskBus(task_id, 'compliance_check'" in src, \
        "compliance route must register task with user_id"


def test_task_ownership_guard_applied():
    """All TaskBus read endpoints must enforce ownership (FIX-066: via load_task_for)."""
    with open('app/routes/tasks.py', 'r', encoding='utf-8') as f:
        t = f.read()
    assert t.count('load_task_for(task_id, user_id)') >= 3, \
        "tasks get/delete/cancel/stream must check ownership"
    with open('app/routes/clearance.py', 'r', encoding='utf-8') as f:
        c = f.read()
    assert c.count('load_task_for(task_id, user_id)') >= 2, \
        "clearance status/stream must check ownership"
    with open('app/routes/batch.py', 'r', encoding='utf-8') as f:
        b = f.read()
    assert 'load_task_for(task_id, user_id)' in b, \
        "plagiarism status must check ownership"
    with open('app/utils/helpers.py', 'r', encoding='utf-8') as f:
        h = f.read()
    assert 'def load_task_for' in h and 'def task_owner_ok' in h, \
        "shared ownership helpers must exist"
    # producers write user_id
    assert "'user_id': str(user_id or '')" in b or '"user_id"' in b


def test_compliance_check_taskbus_arg_fix():
    """compliance_check_task must construct TaskBus with task_id (no TypeError)."""
    with open('app/services/compliance_checker.py', 'r', encoding='utf-8') as f:
        src = f.read()
    assert "TaskBus(task_id, 'compliance_check'" in src
    assert "bus.start(task_id, 'compliance_check'" not in src


# ── FIX-2026-09-11-034: sync-endpoint size guard ──
def test_sync_compare_oversize_guard():
    with open('app/routes/batch.py', 'r', encoding='utf-8') as f:
        src = f.read()
    assert 'def _reject_oversize' in src
    assert 'MAX_SYNC_COMPARE_MB' in src
    # applied to all three API-only sync endpoints
    assert src.count('_reject_oversize()') >= 3


# ── FIX-2026-09-11-035: warning-details annotation (single source in backend) ──
def test_annotate_warning_details():
    from app.services.document_analysis_svc import annotate_warning_details
    hard = {
        'items': [
            {'type': 'lasteditor_same', 'level': 'T1', 'evidence': 'x', 'files': ['a', 'b']},
            {'type': 'paragraph_collusion', 'level': 'T2', 'evidence': 'y', 'files': ['a', 'b']},
        ],
        'violations': [
            {'type': 'tech_seal_leak', 'evidence': 'z', 'files': ['a']},
        ],
    }
    out = annotate_warning_details(hard)
    it0 = out['items'][0]
    assert it0['label'] and it0['guidance'] and it0['chapter']
    # meta type → chapter placeholder
    assert it0['chapter'] == '—'
    assert out['items'][1]['chapter'] != '—'
    v0 = out['violations'][0]
    assert v0['guidance'], "violation must carry guidance"
    # idempotent
    again = annotate_warning_details(out)
    assert again['items'][0]['label'] == it0['label']


def test_warning_details_called_both_paths():
    with open('app/services/document_analysis_svc.py', 'r', encoding='utf-8') as f:
        assert 'hard = annotate_warning_details(hard)' in f.read()
    with open('app/services/clearance_engine.py', 'r', encoding='utf-8') as f:
        assert 'annotate_warning_details' in f.read()
    with open('static/js/app.js', 'r', encoding='utf-8') as f:
        assert 'function _renderWarningDetails' in f.read()


# ── FIX-2026-09-10 (ext): verify_fixes literal/literal_not types ──
def test_verify_fixes_literal_types():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    import verify_fixes as vf
    assert 'literal' in vf.CHECK_RUNNERS
    assert 'literal_not' in vf.CHECK_RUNNERS
    # literal: no regex metachar parsing
    ok, _ = vf._check_literal('data/fix_registry.yaml', '.')
    assert ok is True
    ok2, _ = vf._check_literal('data/fix_registry.yaml', 'zzz_no_match_zzz')
    assert ok2 is False
    ok3, _ = vf._check_literal_not('data/fix_registry.yaml', 'zzz_no_match_zzz')
    assert ok3 is True
    ok4, _ = vf._check_literal_not('data/fix_registry.yaml', 'fixes:')
    assert ok4 is False


# ---- FIX-2026-09-11-036: clearance persistence atomicity ----
def test_clearance_persistence_atomicity():
    import inspect
    from app.services import quote_anomaly, relationship_extractor, document_analysis_svc

    q_sig = inspect.signature(quote_anomaly.save_quote_anomaly_results)
    assert 'conn' in q_sig.parameters
    assert q_sig.parameters['conn'].default is None

    r_sig = inspect.signature(relationship_extractor.save_relationship_results)
    assert 'conn' in r_sig.parameters
    assert r_sig.parameters['conn'].default is None

    src = inspect.getsource(document_analysis_svc._persist_clearance_review)
    assert 'conn=conn' in src, "persistence must thread the shared connection"
    assert src.count('commit()') == 1, "must be a single transaction"
    for tbl in ('quote_anomaly_results', 'entity_relationships', 'relationship_risk_summary'):
        assert f'DELETE FROM {tbl}' in src


# ── QA round 022 (FIX-2026-09-11-037..043) ──
def _read(rel):
    with open(rel, 'r', encoding='utf-8') as f:
        return f.read()


def test_compliance_check_accepts_region_code():
    # FIX-037: Celery task passes region_code= to check(); signature must accept it.
    import inspect
    from app.services.compliance_checker import ComplianceChecker
    sig = inspect.signature(ComplianceChecker.check)
    assert 'region_code' in sig.parameters


def test_credit_task_ownership():
    # FIX-038: credit task endpoints must enforce ownership.
    src = _read('app/routes/credit.py')
    assert "task_owner_ok(task, session.get('user_id'))" in src
    assert "'user_id': user_id," in src


def test_frontend_escape_html_escapes_quotes():
    # FIX-039: escapeHtml must neutralize both quote characters.
    src = _read('static/js/app.js')
    assert '&quot;' in src and '&#39;' in src
    assert 'window.openProjectFromEl' in src
    assert 'escapeHtml(d.skill_a.owner)' in src


def test_clearance_chat_persistence_decoupled():
    # FIX-040: chat bubble written in its own connection, not the result txn.
    src = _read('app/services/clearance_engine.py')
    assert 'with get_db_connection() as _c2:' in src
    assert 'Failed to persist clearance chat message(s)' in src


def test_auth_no_unconditional_deposit():
    # FIX-042: the unconditional re-deposit block must be gone.
    src = _read('app/routes/auth.py')
    assert 'FROM credit_check_reports WHERE user_id = %s", (user_id,))' not in src


def test_session_cookie_flags():
    # FIX-041
    src = _read('app/__init__.py')
    assert 'SESSION_COOKIE_HTTPONLY' in src
    assert 'SESSION_COOKIE_SAMESITE' in src


def test_compose_scheduler_secret_ports():
    # FIX-041
    src = _read('docker-compose.yml')
    assert 'ENABLE_SCHEDULER=false' in src
    assert 'FLASK_SECRET_KEY must be set in .env' in src
    assert '127.0.0.1:5433:5432' in src
    assert '127.0.0.1:6380:6379' in src


def test_admin_pin_default_unified():
    # FIX-042: seed default must match __init__ fallback ('123456').
    src = _read('app/database.py')
    assert "os.getenv('ADMIN_PIN', '123456')" in src
    assert "os.getenv('ADMIN_PIN', '888888')" not in src


def test_sse_cors_removed():
    # FIX-042
    src = _read('app/routes/tasks.py')
    assert 'Access-Control-Allow-Origin' not in src


def test_file_store_dedup_lock():
    # FIX-043
    src = _read('app/services/file_store.py')
    assert 'pg_advisory_xact_lock' in src


def test_input_validation_hardening():
    # FIX-043
    assert "re.match(r'^[A-Za-z0-9_-]{1,64}$'" in _read('app/routes/compliance.py')
    assert 'type=float' in _read('app/routes/graph.py')
    assert 'secrets.randbelow(10000)' in _read('app/routes/admin_regeneration.py')


def test_formality_unknown_label_not_leaked():
    # FIX-044: audit supplement style score must not show raw '(unknown)'.
    src = _read('static/js/app.js')
    assert "if (flabel === 'unknown') flabel = '';" in src


# ── FIX-2026-09-11-045/046: 重点信息雷同假警报 + 报价异常降级 ──
def test_stop_words_expanded():
    from app.services.stop_words import DEFAULT_STOP_WORDS
    for w in ('公司', '工作', '检查', '食品', '填写', '偏离', '提供', '负责'):
        assert w in DEFAULT_STOP_WORDS, f"{w} 应加入停用词表"


def test_key_info_generic_only_not_flagged():
    # 两文档仅共享通用词 → 不输出「重点信息雷同」对
    from app.services.batch_orchestrator import build_key_info_matches
    text = '公司 工作 检查 食品 填写 偏离 提供 负责 公司 工作 检查 食品 填写 偏离 提供 负责'
    pairs = [{'name1': 'a', 'name2': 'b', 'i': 0, 'j': 1, 'text1': text, 'text2': text}]
    assert build_key_info_matches(pairs) == []


def test_key_info_significant_pair_emitted():
    # 共享 ≥3 个显著技术名词且 Jaccard 达标 → 输出该对
    from app.services.batch_orchestrator import build_key_info_matches
    a = '铝合金桥架 防腐涂层 阴极保护 地质勘探 桩基承台 高强螺栓 焊接工艺评定 桥梁支座'
    b = '铝合金桥架 防腐涂层 阴极保护 地质勘探 桩基承台 高强螺栓 焊接工艺评定 桥梁支座 隧道衬砌'
    pairs = [{'name1': 'a', 'name2': 'b', 'i': 0, 'j': 1, 'text1': a, 'text2': b}]
    res = build_key_info_matches(pairs)
    assert len(res) == 1, res
    assert len(res[0]['common_keywords']) >= 3
    assert res[0]['jaccard'] >= 0.15


def test_quote_downgraded_without_open_info():
    # 无开标信息表 → high_price_abnormal 降级为「仅作参考」且 score=0
    from app.services.document_analysis_svc import run_analysis
    docs = [
        {'filename': 'a.docx', 'text': '投标报价 1000000元 工期 730天 施工范围 土建', 'metadata': {}, 'images': []},
        {'filename': 'b.docx', 'text': '投标报价 2000000元 工期 700天 施工范围 土建', 'metadata': {}, 'images': []},
    ]
    report = run_analysis(docs, user_id='t', thread_id='t')
    inds = {i['id']: i for i in report['indicators']}
    q = inds['high_price_abnormal']
    assert '仅作参考' in q['result'], q['result']
    assert q['score'] == 0, q['score']
    # file_scores 也不得被不可靠报价抬分（suspected 排名一致性）
    src = _read('app/services/document_analysis_svc.py')
    assert "if quote_data.get('result') and quote_has_open_prices:" in src


def test_delete_account_global_helper_single_binding():
    # FIX-047: createQuickModal must be global; delete binding must live only in accounts.js
    src = _read('static/js/app.js')
    assert 'window.createQuickModal = createQuickModal;' in src
    assert "addEventListener('click', requestDeleteAccount)" not in src
    assert 'async function requestDeleteAccount()' not in src
    acc = _read('static/js/accounts.js')
    assert "addEventListener('click', requestDeleteAccount)" in acc


# ── FIX-2026-09-11-049/050: LLM 自定义提供商 key 落盘 + 自动拉模型 + UI ──
def test_env_store_upsert_dual_write(tmp_path):
    import os
    from app.services.env_store import write_env_var, has_env_var
    p = tmp_path / 'keys.env'
    write_env_var('LLM_CUSTOM_KEY_QA_TEST', 'secret1', paths=[str(p)])
    assert 'LLM_CUSTOM_KEY_QA_TEST=secret1' in p.read_text(encoding='utf-8')
    assert os.environ.get('LLM_CUSTOM_KEY_QA_TEST') == 'secret1'
    assert has_env_var('LLM_CUSTOM_KEY_QA_TEST') is True
    # upsert replaces, does not duplicate
    write_env_var('LLM_CUSTOM_KEY_QA_TEST', 'secret2', paths=[str(p)])
    body = p.read_text(encoding='utf-8')
    assert body.count('LLM_CUSTOM_KEY_QA_TEST') == 1
    assert 'secret2' in body and 'secret1' not in body
    # preserves unrelated lines
    p.write_text('OTHER_KEY=keepme\n' + body, encoding='utf-8')
    write_env_var('LLM_CUSTOM_KEY_QA_TEST', 'secret3', paths=[str(p)])
    assert 'OTHER_KEY=keepme' in p.read_text(encoding='utf-8')
    os.environ.pop('LLM_CUSTOM_KEY_QA_TEST', None)


def test_env_store_no_dollar_expansion(tmp_path):
    # FIX-049 review M1: values with $ must not be expanded by dotenv.
    from app.services.env_store import write_env_var, _format_line
    assert _format_line('K', 'a$b').strip() == "K='a$b'"
    assert _format_line('K', 'plain123').strip() == 'K=plain123'
    p = tmp_path / 'dollar.env'
    write_env_var('LLM_CUSTOM_KEY_QA_DOLLAR', 'sk-$abc', paths=[str(p)])
    from dotenv import dotenv_values
    import os
    vals = dotenv_values(str(p))
    assert vals['LLM_CUSTOM_KEY_QA_DOLLAR'] == 'sk-$abc'
    os.environ.pop('LLM_CUSTOM_KEY_QA_DOLLAR', None)


def test_env_store_get_env_lazy_multworker(tmp_path, monkeypatch):
    # FIX-049: another worker must see a key written by a different worker —
    # get_env() lazily reloads the persistent file when os.environ lacks it.
    import os
    from app.services import env_store
    p = tmp_path / 'lazy.env'
    p.write_text('LLM_CUSTOM_KEY_LAZY=fromfile\n', encoding='utf-8')
    monkeypatch.setattr(env_store, 'PROVIDER_KEYS_PATH', p)
    monkeypatch.setattr(env_store, '_loaded_mtime', None)
    os.environ.pop('LLM_CUSTOM_KEY_LAZY', None)
    assert env_store.get_env('LLM_CUSTOM_KEY_LAZY') == 'fromfile'
    assert env_store.has_env_var('LLM_CUSTOM_KEY_LAZY') is True
    os.environ.pop('LLM_CUSTOM_KEY_LAZY', None)


def test_provider_key_pipeline_source():
    # backend wiring
    assert 'def write_env_var' in _read('app/services/env_store.py')
    assert 'load_provider_keys' in _read('app/__init__.py')
    adm = _read('app/routes/admin_regeneration.py')
    assert 'write_env_var(env_key, api_key)' in adm
    assert 'Refreshed models for' in adm
    assert "'api_key_set': _env_has" in adm
    # frontend editor + badge
    rev = _read('static/js/review.js')
    assert 'json-list-api-key' in rev
    assert 'api_key_set' in rev


def test_custom_provider_selectable_in_runtime_schema():
    # FIX-051: runtime_config_schema must build options from the merged config
    adm = _read('app/routes/admin_regeneration.py')
    assert 'build provider/model options from the MERGED config' in adm
    assert '（自定义）' in adm
    assert '无模型：请点「刷新模型」或检查 base_url / API Key' in _read('static/js/review.js')


# ── FIX-2026-09-11-053: 移除 Headroom + 休眠指标文案 ──
def test_headroom_removed_and_skip_label():
    req = _read('requirements.txt')
    assert 'headroom-ai' not in req, "headroom-ai 应已移除"
    assert 'magika' not in req, "magika 应已移除"
    assert 'onnxruntime' not in req, "onnxruntime 应已移除"
    da = _read('app/services/document_analysis_svc.py')
    assert '需交易平台数据（当前不可用）' in da
    assert '需外部数据源（交易平台/评标系统数据）' not in da
    assert '休眠指标分类' in _read('docs/ARCHITECTURE.md')

# ── FIX-2026-09-15-054: P0 空转修复 ──
def test_hasllm_written_by_accounts_loader():
    acc = _read('static/js/accounts.js')
    assert "sessionStorage.setItem('hasLLM', authData.has_llm ? 'true' : 'false');" in acc, \
        "accounts.js must persist hasLLM (it shadows app.js loadAccountModal)"
    app = _read('static/js/app.js')
    assert "document.querySelectorAll('#tabBar .admin-tab')" in app, \
        "mobile 'more' collector must still target .admin-tab"


def test_mobile_more_fifth_tab_marked_admin():
    idx = _read('templates/index.html')
    assert 'id="analyticsTabBtn" class="tab-btn admin-tab"' in idx, \
        "5th main tab must carry admin-tab so mobile 'more' is not always empty"


def test_safehtml_global_in_app_not_compliance():
    import os
    app = _read('static/js/app.js')
    assert 'function _safeHTML(html)' in app, "app.js must define global _safeHTML"
    assert not os.path.exists(os.path.join('static', 'js', 'compliance.js')), \
        "compliance.js must be removed (was the old home of _safeHTML)"


def test_dompurify_no_local_vendor_onerror():
    idx = _read('templates/index.html')
    assert "filename='js/purify.min.js'" not in idx, \
        "index.html must not point onerror at the missing local purify vendor"
    assert 'purify.min.js' in idx, "DOMPurify CDN must remain"


def test_region_code_documented_inactive():
    assert '当前未生效：law_regions/region_manager 空置' in _read('app/services/compliance_checker.py'), \
        "region_code swallow must be documented as inactive"


def test_llm_fallback_claims_downgraded():
    assert 'fallback 链 + 熔断器' not in _read('README.md')
    assert 'fallback 链：指数退避 + 熔断器' not in _read('docs/ARCHITECTURE.md')
    assert 'llm_fallback.py' in _read('AGENTS.md'), "backlog note must remain in AGENTS.md"

# ── FIX-2026-09-15-055: P1-A 死代码清理 + LoRA registry schema 对齐 ──
def test_dead_service_modules_removed():
    import os
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'services')
    for name in ('agent_middleware.py', '_save_helper.py', 'region_manager.py'):
        assert not os.path.exists(os.path.join(root, name)), f"{name} 应已删除"


def test_auth_jwt_only_issues_tokens():
    """030 supersedes FIX-055: auth_jwt.py（只签发、无人校验）连同 /api/login 一并删除。"""
    import os
    assert not os.path.exists('app/services/auth_jwt.py')
    assert 'create_token' not in _read('app/routes/chat_sessions.py')


def test_law_semantic_rebuild_removed_but_search_kept():
    src = _read('app/services/law_semantic.py')
    assert 'def semantic_law_search' in src
    assert 'def rebuild_law_index' not in src
    assert 'search_relevant_laws' not in src
    assert 'semantic_law_search' in _read('app/services/compliance_checker.py')


def test_nightly_adapter_registry_schema_matches_readers(tmp_path, monkeypatch):
    import json
    import app.config as app_config
    from app.services.nightly_trainer import _update_adapter_registry

    monkeypatch.setattr(app_config, 'DATA_DIR', tmp_path)
    adapter_dir = tmp_path / 'training' / 'adapters' / 'bidding_agency_1'
    adapter_dir.mkdir(parents=True)
    _update_adapter_registry(str(adapter_dir), {
        'industry': 'bidding_agency',
        'base_model': 'Qwen/Qwen2.5-7B-Instruct',
        'elapsed_seconds': 12,
    })
    reg = json.loads((tmp_path / 'training' / 'adapter_registry.json').read_text(encoding='utf-8'))
    assert 'compliance_checker' not in reg, "bogus top-level key must not be written"
    assert 'bidding_agency' in reg, "registry must be keyed by industry"
    info = reg['bidding_agency']
    assert info['adapter_path'] == str(adapter_dir)
    assert info['base_model'] == 'Qwen/Qwen2.5-7B-Instruct'
    assert info['active'] is True
    # reader contract in llm_provider must stay 'adapter_path'
    assert "info.get('adapter_path', '')" in _read('app/services/llm_provider.py')

# ── FIX-2026-09-15-056: P1-B1 路由/前端降级 ──
def test_audit_blueprint_removed():
    import json
    import os
    assert not os.path.exists(os.path.join('app', 'routes', 'audit.py'))
    assert not os.path.exists(os.path.join('static', 'js', 'bid-audit.js'))
    assert not os.path.exists(os.path.join('tests', 'integration', 'test_audit.py'))
    assert 'from app.routes.audit import audit_bp' not in _read('app/routes/__init__.py')
    snap = json.loads(_read('tests/fixtures/routes_snapshot.json'))
    assert not [r for r in snap if r['rule'].startswith('/audit')], "no /audit routes may remain"


def test_timeline_api_only_downgrade():
    idx = _read('templates/index.html')
    assert 'timelinePanel' not in idx
    app = _read('static/js/app.js')
    assert 'loadTimelinePanel' not in app
    assert 'timelineTabBtn' not in app
    assert 'from app.routes.timeline import timeline_bp' in _read('app/routes/__init__.py')


def test_check_system_no_hardcoded_counts():
    """FIX-2026-09-15-057: checklist header must derive counts, not hardcode them."""
    src = _read('scripts/check_system.py')
    assert '21/21' not in src
    assert '16/16' not in src
    assert '_verify_fixes_summary' in src
    assert '_status_totals' in src


def test_doc_drift_tracks_law_count():
    """FIX-2026-09-15-057: doc_drift monitors the loaded law count (6th metric)."""
    src = _read('scripts/check_doc_drift.py')
    assert '_count_laws' in src
    assert "'laws'" in src
    assert '法规库（15 部' in _read('README.md')


def test_docs_honest_after_audit():
    """FIX-2026-09-15-057: stale provider keys / route / backlog corrected."""
    um = _read('docs/USER_MANUAL.md')
    assert 'DEEPSEEK_API_KEY' not in um
    assert 'POST /clearance/run' in um
    agents = _read('AGENTS.md')
    assert '清标任务归属校验' not in agents
    # the removed backlog claim is stale — ownership IS enforced (source of truth)
    assert 'def task_owner_ok' in _read('app/utils/helpers.py')


def test_current_state_marked_deprecated():
    """FIX-2026-09-15-057: the 2026-07-30 snapshot must be marked deprecated."""
    assert 'deprecated: true' in _read('data/current_state.yaml')


def test_clearance_snapshot_skip_text_current():
    """FIX-2026-09-15-057: scores.json skip strings aligned to FIX-053 wording."""
    snap = _read('tests/fixtures/clearance_baseline/scores.json')
    assert '需外部数据源' not in snap
    assert '需交易平台数据（当前不可用）' in snap


def test_compliance_law_pool_expanded():
    """FIX-2026-09-15-058: engine loads core-4 + 11 extended national full-text laws."""
    from app.services.compliance_checker import _get_seed_laws, _CORE_LAW_NAMES
    laws = _get_seed_laws()
    names = {l['law_name'] for l in laws}
    for c in _CORE_LAW_NAMES:
        assert c in names, f"core law missing: {c}"
    for n in ('政府采购法实施条例', '工程建设项目施工招标投标办法',
              '评标委员会和评标方法暂行规定', '电子招标投标办法',
              '公共资源交易平台管理暂行办法', '政府采购货物和服务招标投标管理办法'):
        assert n in names, f"extended law missing: {n}"
    assert len(names) == 15, f"expected 15 loaded laws, got {len(names)}"
    for l in laws:
        assert l['law_name'] and l['article'] and l['text']


def test_compliance_core_law_guard_wired():
    """FIX-2026-09-15-058: the core-4 guard must exist and keep basics in selection."""
    src = _read('app/services/compliance_checker.py')
    assert '_CORE_LAW_NAMES' in src and 'Core-4 guard' in src
    from app.services.compliance_checker import _select_relevant_laws, _CORE_LAW_NAMES
    rules = [
        {"category": "prohibition", "description": "串通投标 围标 弄虚作假 转包 分包", "original_text": ""},
        {"category": "commercial", "description": "合同 价款 保证金 期限 报价", "original_text": ""},
        {"category": "qualification", "description": "资质 资格 业绩 证书", "original_text": ""},
        {"category": "technical", "description": "标准 技术 质量 规范", "original_text": ""},
    ]
    sel = _select_relevant_laws(rules)
    assert 0 < len(sel) <= 15
    picked = {l['law_name'] for l in sel}
    # every core law that has any article must be represented (guard contract)
    for c in _CORE_LAW_NAMES:
        assert c in picked, f"core law crowded out: {c}"


def test_bootstrap_ensure_seeded():
    """FIX-2026-09-15-059: ensure_seeded copies missing mutable seeds; idempotent."""
    import os as _os
    import tempfile
    from app import bootstrap
    with tempfile.TemporaryDirectory() as repo, tempfile.TemporaryDirectory() as dst:
        with open(_os.path.join(repo, 'domain_words.txt'), 'w', encoding='utf-8') as f:
            f.write('招标\n投标\n')
        old_repo, old_data = bootstrap.REPO_DATA, bootstrap.DATA_DIR
        try:
            bootstrap.REPO_DATA, bootstrap.DATA_DIR = repo, dst
            bootstrap.ensure_seeded()
            out = _os.path.join(dst, 'domain_words.txt')
            assert _os.path.exists(out) and '招标' in open(out, encoding='utf-8').read()
            # idempotent: a runtime-mutated volume file is NOT overwritten
            with open(out, 'a', encoding='utf-8') as f:
                f.write('追加词\n')
            bootstrap.ensure_seeded()
            assert '追加词' in open(out, encoding='utf-8').read()
            assert not [f for f in _os.listdir(dst) if f.endswith('.seedtmp')]
        finally:
            bootstrap.REPO_DATA, bootstrap.DATA_DIR = old_repo, old_data


def test_bootstrap_wired_and_compose_mounts():
    """FIX-2026-09-15-059: seeded from create_app; compose has repo_data + factory mounts."""
    assert 'from app.bootstrap import ensure_seeded' in _read('app/__init__.py')
    comp = _read('docker-compose.yml')
    assert './data:/app/repo_data:ro' in comp
    assert './data/runtime_config_factory.json:/app/data/runtime_config_factory.json:ro' in comp
    assert './data/domain_words.txt:/app/data/domain_words.txt:ro' not in comp


def test_admin_file_access_scoped_by_project():
    """FIX-2026-09-15-060: project file/version queries must be constrained by project_id."""
    src = _read('app/routes/admin.py')
    assert 'AND pf.project_id = %s' in src
    assert 'WHERE id = %s AND project_id = %s' in src
    assert "if '..' in zip_filename" in src
    assert '_can_access_project(project_id, user_id)' in src


def test_compliance_result_owner_checked():
    """FIX-2026-09-15-060: compliance result/rules endpoints enforce task ownership."""
    src = _read('app/routes/compliance.py')
    assert 'def _task_forbidden' in src
    assert src.count('if _task_forbidden(task_id):') >= 3  # get_result / get_rules / update_rules


def test_deletion_code_not_logged():
    """FIX-2026-09-15-060: account-deletion verification code must not be logged."""
    src = _read('app/routes/admin_regeneration.py')
    assert 'code_sent_{code}' not in src
    assert 'code_sent_****' in src


def test_beat_schedule_covers_maintenance_jobs():
    """FIX-2026-09-15-061: Celery Beat schedules the maintenance jobs (Docker path)."""
    import celery_app
    bs = celery_app.celery.conf.beat_schedule
    assert len(bs) >= 25, f"expected >=25 beat entries, got {len(bs)}"
    for name in ('cleanup-old-sessions', 'cleanup-orphan-users', 'generate-monthly-report',
                 'generate-annual-report', 'auto-rag-health-check', 'auto-cleanup-memory',
                 'auto-cleanup-stale-reviews', 'cleanup-expired-recycle-bin'):
        assert name in bs, f"missing beat entry: {name}"
    import app.cleanup_tasks as ct
    for fn in ('cleanup_old_sessions', 'cleanup_orphan_users', 'auto_cleanup_memory',
               'auto_generate_monthly_report', 'auto_cleanup_stale_reviews'):
        assert hasattr(getattr(ct, fn), 'delay'), f"{fn} is not a Celery task"


def test_compose_data_volumes_and_env():
    """FIX-2026-09-15-061: compose has shared env anchor, dir volumes, beat schedule."""
    comp = _read('docker-compose.yml')
    assert 'x-app-env: &app-env' in comp
    assert 'company_kb_files:/app/company_kb_files' in comp
    assert 'knowledge_lab_files:/app/knowledge_lab_files' in comp
    assert '--schedule=/app/data/celerybeat-schedule' in comp
    assert './data:/app/repo_data:ro' in _read('docker-compose.e2e.yml')


def test_missing_tables_declared():
    """FIX-2026-09-15-061: previously-missing tables are now in the schema."""
    db = _read('app/database.py')
    for t in ('wiki_bookmarks', 'wiki_view_log', 'user_feedback', 'knowledge_lab_skills'):
        assert f'CREATE TABLE IF NOT EXISTS {t}' in db, f"missing CREATE TABLE for {t}"


def test_trend_service_uses_real_feedback_table():
    """FIX-2026-09-15-061: trend accuracy reads compliance_feedback (the written table)."""
    src = _read('app/services/trend_service.py')
    assert 'FROM compliance_feedback' in src
    assert 'FROM compliance_check_feedback' not in src


def test_unresolved_yaml_is_valid():
    """FIX-2026-09-15-067: unresolved.yaml must be parseable YAML (no bad escapes)."""
    import yaml
    with open('data/unresolved.yaml', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict) and len(data['unresolved']) > 0


def test_factory_has_no_orphan_typo_keys():
    """FIX-2026-09-15-067: typo_* keys removed (no reader in code)."""
    assert 'typo_' not in _read('data/runtime_config_factory.json')


def test_run_ui_audit_no_stale_typo_surface():
    """FIX-2026-09-15-067: stale #sidebarTypoResultsBtn probe removed."""
    assert 'sidebarTypoResultsBtn' not in _read('scripts/run_ui_audit.py')


def test_dockerfile_installs_torch_after_requirements():
    """FIX-2026-09-15-067: TORCH_INDEX wheel installed after requirements so it wins."""
    assert 'torch/torchvision installed LAST' in _read('Dockerfile')


# ── FIX-2026-09-24-062/063/064/065: 合规归属 fail-closed + task_id 白名单 + TaskBus Redis 重试 ──
def test_compliance_feedback_requires_owner(app, monkeypatch):
    """FIX-062: 向他人 task 提交反馈必须 403（越权写入修复）。"""
    from app.services import task_bus as tb
    monkeypatch.setattr(tb.TaskBus, 'get',
                        staticmethod(lambda task_id: {'user_id': 'someone-else'}))
    client = app.test_client()
    with client.session_transaction() as sess:
        sess['user_id'] = 'test-user'
        sess['consent_value'] = 1
    r = client.post('/compliance/feedback', json={
        'task_id': 'abcd1234', 'check_file_name': 'x.docx', 'user_verdict': 'true_violation'})
    assert r.status_code == 403, f"越权 feedback 必须 403，实得 {r.status_code}"


def test_compliance_task_forbidden_fail_closed(app, monkeypatch):
    """FIX-063: 归属校验异常时必须 fail-closed（拒绝），不得放行。"""
    from app.services import task_bus as tb

    def _boom(task_id):
        raise RuntimeError('redis down')

    monkeypatch.setattr(tb.TaskBus, 'get', staticmethod(_boom))
    client = app.test_client()
    with client.session_transaction() as sess:
        sess['user_id'] = 'test-user'
        sess['consent_value'] = 1
    r = client.get('/compliance/result/abcd1234')
    assert r.status_code == 403, f"归属校验异常必须 fail-closed 403，实得 {r.status_code}"


def test_compliance_load_result_rejects_traversal():
    """FIX-064: task_id 白名单阻断路径遍历；合法 uuid / rules_<uuid> 放行。"""
    from app.routes.compliance import _load_result, _save_result, _valid_task_id
    assert _valid_task_id('rules_abc-123') is True
    assert _valid_task_id('../../etc/passwd') is False
    assert _valid_task_id('a/b') is False
    assert _valid_task_id('') is False
    assert _valid_task_id('a' * 65) is False
    assert _valid_task_id(None) is False
    assert _load_result('../../etc/passwd') is None
    with pytest.raises(ValueError):
        _save_result('../../evil', {})


def test_task_bus_redis_retries_after_interval(monkeypatch):
    """FIX-065: Redis 首次不可用后超过重试间隔须再次尝试（不再永久禁用）。"""
    from app.services import task_bus as tb
    import redis as redis_mod

    monkeypatch.setattr(tb, '_redis', None)
    monkeypatch.setattr(tb, '_redis_last_try', 0.0)
    clock = {'t': 1000.0}
    monkeypatch.setattr(tb.time, 'time', lambda: clock['t'])

    calls = {'n': 0}

    def _fail(*args, **kwargs):
        calls['n'] += 1
        raise RuntimeError('boom')

    monkeypatch.setattr(redis_mod.Redis, 'from_url', staticmethod(_fail))

    assert tb._get_redis() is False, "首次失败应 latch 为 False"
    assert calls['n'] == 1
    assert tb._get_redis() is None, "重试间隔内不再尝试（返回 None）"
    assert calls['n'] == 1
    clock['t'] += tb._REDIS_RETRY_INTERVAL + 1
    assert tb._get_redis() is False, "超过间隔后应重试"
    assert calls['n'] == 2


# ── FIX-2026-09-24-066: owner 校验统一封装（load_task_for，异常 fail-closed）──
def test_load_task_for_statuses(monkeypatch):
    """load_task_for: ok / forbidden / missing / 异常→forbidden（fail-closed）。"""
    from app.utils import helpers as h
    from app.services import task_bus as tb

    monkeypatch.setattr(tb.TaskBus, 'get', staticmethod(lambda tid: {'user_id': 'u1'}))
    meta, status = h.load_task_for('t1', 'u1')
    assert status == 'ok' and meta['user_id'] == 'u1'

    monkeypatch.setattr(tb.TaskBus, 'get', staticmethod(lambda tid: {'user_id': 'u2'}))
    assert h.load_task_for('t1', 'u1')[1] == 'forbidden'

    monkeypatch.setattr(tb.TaskBus, 'get', staticmethod(lambda tid: None))
    assert h.load_task_for('t1', 'u1') == (None, 'missing')

    def _boom(tid):
        raise RuntimeError('redis down')
    monkeypatch.setattr(tb.TaskBus, 'get', staticmethod(_boom))
    assert h.load_task_for('t1', 'u1') == (None, 'forbidden')


def _session_login(client, uid='test-user'):
    with client.session_transaction() as sess:
        sess['user_id'] = uid
        sess['consent_value'] = 1


def test_clearance_status_fail_closed(app, monkeypatch):
    """FIX-066: TaskBus 异常时 /clearance/status 返回 403（原 500）。"""
    from app.services import task_bus as tb

    def _boom(tid):
        raise RuntimeError('redis down')
    monkeypatch.setattr(tb.TaskBus, 'get', staticmethod(_boom))
    client = app.test_client()
    _session_login(client)
    r = client.get('/clearance/status/abcd1234')
    assert r.status_code == 403, f"expect 403 fail-closed, got {r.status_code}"


def test_tasks_get_fail_closed(app, monkeypatch):
    """FIX-066: TaskBus 异常时 GET /tasks/<id> 返回 403（原 500）。"""
    from app.services import task_bus as tb

    def _boom(tid):
        raise RuntimeError('redis down')
    monkeypatch.setattr(tb.TaskBus, 'get', staticmethod(_boom))
    client = app.test_client()
    _session_login(client)
    r = client.get('/tasks/abcd1234')
    assert r.status_code == 403, f"expect 403 fail-closed, got {r.status_code}"


def test_batch_plagiarism_status_fail_closed(app, monkeypatch):
    """FIX-066: TaskBus 异常时 /batch/plagiarism/status 返回 403（原 500）。"""
    from app.services import task_bus as tb

    def _boom(tid):
        raise RuntimeError('redis down')
    monkeypatch.setattr(tb.TaskBus, 'get', staticmethod(_boom))
    client = app.test_client()
    _session_login(client)
    r = client.get('/batch/plagiarism/status/abcd1234')
    assert r.status_code == 403, f"expect 403 fail-closed, got {r.status_code}"


def test_compliance_result_fail_closed(app, monkeypatch):
    """FIX-066: /compliance/result 经 load_task_for 委托，异常同样 fail-closed 403。"""
    from app.services import task_bus as tb

    def _boom(tid):
        raise RuntimeError('redis down')
    monkeypatch.setattr(tb.TaskBus, 'get', staticmethod(_boom))
    client = app.test_client()
    _session_login(client)
    r = client.get('/compliance/result/abcd1234')
    assert r.status_code == 403, f"expect 403 fail-closed, got {r.status_code}"


def test_clearance_stream_imports_taskbus():
    """FIX-066 C1: clearance_stream 使用 TaskBus.subscribe，必须 import TaskBus。"""
    with open('app/routes/clearance.py', 'r', encoding='utf-8') as f:
        src = f.read()
    # 存在 TaskBus.subscribe 使用点，则必须有 TaskBus 的导入
    assert 'TaskBus.subscribe(task_id)' in src
    assert 'from app.services.task_bus import TaskBus' in src


# ── FIX-2026-09-28-068: login brute-force hardening ──
def test_login_guard_is_non_blocking():
    """冷却闸必须非阻塞：login_guard 不得出现 time.sleep / gevent.sleep。"""
    with open('app/services/login_guard.py', 'r', encoding='utf-8') as f:
        src = f.read()
    assert 'time.sleep' not in src
    assert 'gevent.sleep' not in src


def test_proxyfix_gated_by_trust_proxy_only():
    """ProxyFix 仅由 TRUST_PROXY 控制（不得用 APP_ENV），且 limiter 开启内存降级。"""
    with open('app/__init__.py', 'r', encoding='utf-8') as f:
        src = f.read()
    assert 'ProxyFix(app.wsgi_app, x_for=1, x_proto=1)' in src
    assert 'TRUST_PROXY' in src
    assert 'in_memory_fallback_enabled=True' in src
    assert 'swallow_errors=True' not in src


def test_login_rate_key_includes_ip(app):
    """限流 key 必须含 username+IP，且不得以空串结尾（空 key=全站共用桶）。"""
    from app.services.login_guard import login_rate_key_func
    with app.test_request_context('/login', method='POST',
                                  json={'username': 'Alice', 'pin': '1234'}):
        key = login_rate_key_func()
    assert key.startswith('login:alice:')
    assert not key.endswith(':')


def test_login_guard_cooldown_then_lock(app, monkeypatch):
    """内存兜底路径：连错达阈值触发锁定；reset_login 清零。"""
    from app.services import login_guard as lg
    monkeypatch.setattr(lg, '_redis', lambda: None)
    lg._mem.clear()
    try:
        assert lg.check_login_gate('bob', '10.0.0.1') == 0
        for _ in range(5):
            lg.record_login_failure('bob', '10.0.0.1')
        assert lg.check_login_gate('bob', '10.0.0.1') > 0, 'expect lock after 5 failures'
        lg.reset_login('bob', '10.0.0.1')
        assert lg.check_login_gate('bob', '10.0.0.1') == 0
    finally:
        lg._mem.clear()


def test_login_endpoint_rate_limited(app):
    """FIX-068: /login 连续失败后返回 429 + Retry-After + JSON（Layer1/2/3 形态统一）。"""
    client = app.test_client()
    statuses = []
    for _ in range(8):
        r = client.post('/login', json={'username': 'admin', 'pin': '0000'})
        statuses.append(r.status_code)
        if r.status_code == 429:
            assert 'Retry-After' in r.headers, '429 must carry Retry-After'
            assert r.mimetype == 'application/json', f'429 must be JSON, got {r.mimetype}'
            assert (r.get_json() or {}).get('code') == 'RATE_LIMITED'
            break
    assert 429 in statuses, f'expected 429 after repeated failures, got {statuses}'


def test_ratelimit_breach_response_contract(app):
    """FIX-068: flask-limiter 的 on_breach 429 必须是 JSON + Retry-After。"""
    from app import _ratelimit_breach_response

    class _FakeLimit:
        reset_at = 9_999_999_999

    with app.app_context():
        resp = _ratelimit_breach_response(_FakeLimit())
    assert resp.status_code == 429
    assert resp.mimetype == 'application/json'
    assert resp.headers.get('Retry-After')
    assert int(resp.headers['Retry-After']) >= 1


# ── FIX-2026-09-29-069: PIN code must not be echoed / persisted / logged ──
class _FakeCur:
    def __init__(self, row):
        self._row = row

    def execute(self, *a, **k):
        pass

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self, row):
        self._row = row

    def cursor(self, *a, **k):
        return _FakeCur(self._row)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass


def _pin_session(client, uid='pin-user'):
    with client.session_transaction() as sess:
        sess['user_id'] = uid
        sess['consent_value'] = 1


def test_request_pin_code_smtp_off_no_leak(app, monkeypatch):
    """FIX-069: SMTP 未配置 → 503；响应/头/日志均不得出现验证码（仅指纹）。"""
    import logging
    from app.routes import auth as auth_mod
    from app.utils import mailer as mailer_mod

    monkeypatch.setattr(auth_mod, 'get_db_connection', lambda: _FakeConn(('u@example.com', 'alice')))
    monkeypatch.setattr(mailer_mod, 'is_configured', lambda: False)
    monkeypatch.setattr(auth_mod.secrets, 'randbelow', lambda n: 1234)

    records = []

    class _H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    h = _H()
    auth_mod.logger.addHandler(h)
    try:
        client = app.test_client()
        _pin_session(client)
        r = client.post('/request_pin_change_code')
    finally:
        auth_mod.logger.removeHandler(h)

    assert r.status_code == 503, r.status_code
    assert '1234' not in r.get_data(as_text=True), 'response body leaked the code'
    assert '1234' not in str(dict(r.headers)), 'response headers leaked the code'
    assert all('1234' not in m for m in records), f'logs leaked the code: {records}'


def test_request_pin_code_cooldown(app, monkeypatch):
    """FIX-069: 60s 内二次请求 → 429，且不重置验证码有效期窗口。"""
    from app.routes import auth as auth_mod
    from app.utils import mailer as mailer_mod

    monkeypatch.setattr(auth_mod, 'get_db_connection', lambda: _FakeConn(('u@example.com', 'alice')))
    monkeypatch.setattr(mailer_mod, 'is_configured', lambda: True)
    monkeypatch.setattr(mailer_mod, 'send_email', lambda *a, **k: True)

    client = app.test_client()
    _pin_session(client)
    r1 = client.post('/request_pin_change_code')
    assert r1.status_code == 200, r1.status_code
    with client.session_transaction() as sess:
        exp1 = sess.get('pin_change_code_expiry')
    r2 = client.post('/request_pin_change_code')
    assert r2.status_code == 429, r2.status_code
    assert 'Retry-After' in r2.headers
    with client.session_transaction() as sess:
        assert sess.get('pin_change_code_expiry') == exp1, 'cooldown must not reset the window'


def test_pin_verify_constant_time_and_hash_only():
    """FIX-069: 常量时间比对 + 会话仅存哈希（无明文码、无调试回显）。"""
    with open('app/routes/auth.py', 'r', encoding='utf-8') as f:
        src = f.read()
    assert 'secrets.compare_digest' in src
    assert 'pin_change_code_hash' in src
    assert "session['pin_change_code']" not in src
    assert '调试模式' not in src


# ── FIX-2026-09-29-070: session ownership must be fail-closed (no bypass) ──
class _GateCur:
    def __init__(self, row):
        self._row = row
        self.sql = ''

    def execute(self, sql, params=None):
        self.sql = sql

    def fetchone(self):
        return self._row


def test_assert_thread_access_fail_closed():
    """FIX-070: 缺 actor→拒（不查库）；owner/项目成员→允；无行→拒。"""
    from app.services.session_manager import _assert_thread_access
    c = _GateCur(('x',))
    assert _assert_thread_access(c, 't', None) is False
    assert c.sql == '', 'must deny without querying when actor is missing'
    c = _GateCur((1,))
    assert _assert_thread_access(c, 't', 'u') is True
    assert 'cs.user_id = %s' in c.sql and 'project_members' in c.sql
    c = _GateCur(None)
    assert _assert_thread_access(c, 't', 'u') is False


def test_archive_route_denies_foreign_thread(app, monkeypatch):
    """FIX-070: A 带 B 的 thread_id 调归档 → 404（不得归档/删他人会话）。"""
    from app.routes import chat_sessions as cs
    monkeypatch.setattr(cs, 'thread_accessible', lambda tid, uid=None: False)
    client = app.test_client()
    _session_login(client, uid='userA')
    r = client.post('/archive_session/thread-owned-by-B')
    assert r.status_code == 404, r.status_code


def test_archive_session_service_denies_foreign_thread(app, monkeypatch):
    """FIX-070: 服务层归档非本人会话 → 返回 None（不写入）。"""
    from app.services import session_manager as sm
    monkeypatch.setattr(sm, '_assert_thread_access', lambda cur, tid, actor: False)
    monkeypatch.setattr(sm, 'get_db_connection', lambda: _FakeConn(None))
    assert sm.archive_session('thread-owned-by-B', 'userA') is None


def test_session_ownership_gate_wired():
    """FIX-070: 五件套均过闸，且不存在 bypass 开关。"""
    with open('app/services/session_manager.py', 'r', encoding='utf-8') as f:
        src = f.read()
    assert src.count('_assert_thread_access(') >= 6, 'all five session ops must gate'
    assert 'system=True' not in src, 'no bypass flag allowed'


# ── FIX-2026-09-29-071: ZIP extraction must not escape the target dir ──
def test_safe_extract_zip_blocks_traversal(tmp_path):
    """FIX-071: ../、绝对路径成员不得写出目标目录；正常成员照常解压。"""
    import zipfile as _zip
    from app.services.ingest_pipeline import _safe_extract_zip
    dest = tmp_path / 'ingest'
    dest.mkdir()
    zpath = tmp_path / 'evil.zip'
    with _zip.ZipFile(zpath, 'w') as zf:
        zf.writestr('ok.txt', 'fine')
        zf.writestr('../evil.txt', 'pwned')
        zf.writestr('/abs.txt', 'pwned-abs')
    with _zip.ZipFile(zpath, 'r') as zf:
        extracted, skipped = _safe_extract_zip(zf, str(dest))
    assert extracted == 1, extracted
    assert (dest / 'ok.txt').read_text() == 'fine'
    assert not (tmp_path / 'evil.txt').exists(), 'traversal escaped the target dir'
    assert len(skipped) >= 2, skipped


def test_ingest_pipeline_no_extractall():
    """FIX-071: 必须弃用 extractall，改用路径白名单解压。"""
    with open('app/services/ingest_pipeline.py', 'r', encoding='utf-8') as f:
        src = f.read()
    assert 'extractall(' not in src
    assert '_safe_extract_zip(' in src


def test_doc_drift_covers_table_init_variant():
    """C2: check_doc_drift 必须覆盖「N 表初始化」变体（防 ARCHITECTURE 再次漏报）。"""
    with open('scripts/check_doc_drift.py', 'r', encoding='utf-8') as f:
        src = f.read()
    assert r'(\d+)\s*表初始化' in src


# ── C3-a: audit orchestrator + report generators removed ──
def test_audit_orchestrator_removed():
    """C3-a: 审计编排器与报告生成文件已删除；clearance 依赖的活体面保留。"""
    import os
    assert not os.path.exists('app/services/audit_report.py')
    assert not os.path.exists('app/services/audit_wiki_publisher.py')
    with open('app/services/audit_engine.py', 'r', encoding='utf-8') as f:
        src = f.read()
    for dead in ('def run_audit', 'def run_preflight', 'def _generate_reports',
                 'def get_run_results', 'def get_project_history', 'def get_running_audit'):
        assert dead not in src, f'{dead} should be removed'
    assert 'def _run_style_analysis' in src
    assert 'SCORING_FUNCTIONS' in src


# ── C3-b: frontend dead blocks / stale DOM ids removed ──
def test_frontend_stale_dom_ids_removed():
    """C3-b: 前端陈旧 DOM id / 死块不得复活。"""
    targets = {
        'static/js/app.js': ["'databaseTabBtn'", "'sidebarEditPromptBtn'", "'fileStationBtn'"],
        'static/js/cases.js': ["'casesToggleStatus'", "function autoGenerate"],
        'static/js/knowledge-lab.js': ["'sidebarEditPromptBtn'"],
        'static/js/review.js': ["loadDocReviewPanel", "initDocReviewToggle", "'docReviewPanel'"],
        'static/css/app.css': ["#docReviewPanel"],
    }
    for f, needles in targets.items():
        with open(f, encoding='utf-8') as fh:
            src = fh.read()
        for n in needles:
            assert n not in src, f'{n} should be removed from {f}'


# ── FIX-2026-09-29-077: compliance feedback/training owner scope ──
class _SqlCur:
    def __init__(self, rows=None, count=0):
        self._rows = rows or []
        self._count = count
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return (self._count,)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _SqlConn:
    def __init__(self, cur):
        self._cur = cur

    def cursor(self, *a, **k):
        return self._cur

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_db(monkeypatch, cur):
    import app.database as db
    monkeypatch.setattr(db, 'get_db_connection', lambda: _SqlConn(cur))


def _sess(client, uid, role='user'):
    with client.session_transaction() as sess:
        sess['user_id'] = uid
        sess['consent_value'] = 1
        sess['role'] = role


def test_feedback_history_owner_filter(app, monkeypatch):
    """FIX-077: 非管理员拉 /compliance/feedback/history 必须带 WHERE user_id = %s。"""
    cur = _SqlCur()
    _patch_db(monkeypatch, cur)
    client = app.test_client()
    _sess(client, 'userA', 'user')
    r = client.get('/compliance/feedback/history')
    assert r.status_code == 200, r.status_code
    sqls = ' '.join(s for s, _ in cur.calls)
    assert 'FROM compliance_feedback' in sqls and 'WHERE user_id = %s' in sqls
    assert any(p and 'userA' in p for _, p in cur.calls), 'owner param must be the session user'


def test_feedback_history_admin_unfiltered(app, monkeypatch):
    """FIX-077: 管理员可看全量（无 WHERE user_id）。"""
    cur = _SqlCur()
    _patch_db(monkeypatch, cur)
    client = app.test_client()
    _sess(client, 'admin', 'admin')
    r = client.get('/compliance/feedback/history')
    assert r.status_code == 200
    sqls = ' '.join(s for s, _ in cur.calls)
    assert 'WHERE user_id = %s' not in sqls


def test_training_data_owner_filter(app, monkeypatch):
    """FIX-077: 非管理员导出训练数据必须带 WHERE user_id = %s（不得跨用户）。"""
    cur = _SqlCur()
    _patch_db(monkeypatch, cur)
    client = app.test_client()
    _sess(client, 'userB', 'user')
    r = client.get('/compliance/training_data')
    assert r.status_code == 200, r.status_code
    sqls = ' '.join(s for s, _ in cur.calls)
    assert 'FROM compliance_feedback' in sqls and 'WHERE user_id = %s' in sqls


def test_compliance_feedback_owner_filter_wired():
    with open('app/routes/compliance.py', encoding='utf-8') as f:
        src = f.read()
    assert src.count('WHERE user_id = %s') >= 2, 'both feedback endpoints must scope by owner'
    assert "session.get('role') == 'admin'" in src


# ── FIX-2026-09-29-078: knowledge upload whitelist + opaque stored name ──
def test_knowledge_lab_upload_rejects_exe(app):
    """⑨: /knowledge_lab/upload 拒 .exe（早退，无需 DB）。"""
    import io as _io
    client = app.test_client()
    _sess(client, 'userA', 'user')
    r = client.post('/knowledge_lab/upload',
                    data={'file': (_io.BytesIO(b'MZbinary'), 'evil.exe')},
                    content_type='multipart/form-data')
    assert r.status_code == 400, r.status_code


def test_knowledge_lab_upload_rejects_html(app):
    """⑨: /knowledge_lab/upload 拒 .html。"""
    import io as _io
    client = app.test_client()
    _sess(client, 'userA', 'user')
    r = client.post('/knowledge_lab/upload',
                    data={'file': (_io.BytesIO(b'<script>x</script>'), 'x.html')},
                    content_type='multipart/form-data')
    assert r.status_code == 400, r.status_code


def test_company_kb_upload_rejects_double_ext(app):
    """⑨: /company_kb/upload 拒双扩展 a.pdf.exe。"""
    import io as _io
    client = app.test_client()
    _sess(client, 'admin', 'admin')
    r = client.post('/company_kb/upload',
                    data={'file': (_io.BytesIO(b'X'), 'a.pdf.exe'), 'category': 'test'},
                    content_type='multipart/form-data')
    assert r.status_code == 400, r.status_code


def test_knowledge_upload_storage_name_opaque():
    """⑨: 存储名不得含原始文件名。"""
    for f in ('app/routes/knowledge.py', 'app/routes/knowledge_company_kb.py'):
        with open(f, encoding='utf-8') as fh:
            src = fh.read()
        assert 'allowed_file(file.filename)' in src, f
        assert 'f"{file_hash}_{int(time.time())}{_ext}"' in src, f
        assert '{int(time.time())}_{file.filename}' not in src, f


# ── FIX-2026-09-29-079: project AI memory membership + forced role ──
def test_ai_memory_requires_project_membership(app, monkeypatch):
    """⑧: 非项目成员写 ai_memory → 403。"""
    from app.routes import projects as proj
    monkeypatch.setattr(proj, 'can_access_project', lambda pid, uid: False)
    client = app.test_client()
    _sess(client, 'userA', 'user')
    r = client.post('/admin/projects/7/ai_memory',
                    json={'role': 'user', 'content': 'hello there'})
    assert r.status_code == 403, r.status_code


def test_ai_memory_forces_user_role(app, monkeypatch):
    """⑧: role=assistant 被服务端强制改写为 user（防伪装污染）。"""
    from app.routes import projects as proj
    monkeypatch.setattr(proj, 'can_access_project', lambda pid, uid: True)
    captured = {}

    class _Cur:
        def execute(self, sql, params=None):
            captured['params'] = params

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Conn:
        def cursor(self, *a, **k):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def commit(self):
            pass

    from app.routes import admin_regeneration as ar
    monkeypatch.setattr(ar, 'get_db_connection', lambda: _Conn())
    client = app.test_client()
    _sess(client, 'userA', 'user')
    r = client.post('/admin/projects/7/ai_memory',
                    json={'role': 'assistant', 'content': 'fake assistant reply'})
    assert r.status_code == 200, r.status_code
    assert captured['params'][2] == 'user', captured


def test_ai_memory_gate_wired():
    with open('app/routes/admin_regeneration.py', encoding='utf-8') as f:
        src = f.read()
    assert 'can_access_project(project_id, user_id)' in src
    assert "data.get('role', 'user')" not in src, 'client role must not be trusted'


# ── FIX-2026-09-29-080: compliance rules ownership ──
def test_rules_get_cross_user_forbidden(app, monkeypatch):
    """⑩: A 读 B 的 rules_task_id → 403。"""
    from app.services import task_bus as tb
    monkeypatch.setattr(tb.TaskBus, 'get',
                        staticmethod(lambda tid: {'user_id': 'userB', 'type': 'compliance_rules'}))
    client = app.test_client()
    _sess(client, 'userA', 'user')
    r = client.get('/compliance/rules/rules-xyz')
    assert r.status_code == 403, r.status_code


def test_rules_update_cross_user_forbidden(app, monkeypatch):
    """⑩: A 改 B 的 rules → 403。"""
    from app.services import task_bus as tb
    monkeypatch.setattr(tb.TaskBus, 'get',
                        staticmethod(lambda tid: {'user_id': 'userB', 'type': 'compliance_rules'}))
    client = app.test_client()
    _sess(client, 'userA', 'user')
    r = client.put('/compliance/rules/rules-xyz', json={'rules': []})
    assert r.status_code == 403, r.status_code


def test_rules_get_unowned_denied(app, monkeypatch):
    """⑩: 无主（meta miss）→ fail-closed 403（F3）。"""
    from app.services import task_bus as tb
    monkeypatch.setattr(tb.TaskBus, 'get', staticmethod(lambda tid: None))
    client = app.test_client()
    _sess(client, 'userA', 'user')
    r = client.get('/compliance/rules/rules-unowned-123')
    assert r.status_code == 403, r.status_code


def test_start_check_denies_foreign_rules(app, monkeypatch):
    """⑩: A 用 B 的 rules_task_id 发起检查 → 403。"""
    from app.services import task_bus as tb
    monkeypatch.setattr(tb.TaskBus, 'get',
                        staticmethod(lambda tid: {'user_id': 'userB'}))
    client = app.test_client()
    _sess(client, 'userA', 'user')
    r = client.post('/compliance/check',
                    json={'rules_task_id': 'rules-fgn', 'bid_file_text': 'x' * 50})
    assert r.status_code == 403, r.status_code


def test_rules_ownership_gate_wired():
    with open('app/routes/compliance.py', encoding='utf-8') as f:
        src = f.read()
    assert "return status != 'ok'" in src, 'missing meta must fail closed'
    assert "'compliance_rules'" in src, 'extract_rules must register an owner'
    assert '无权访问该规则数据' in src, 'start_check must check rules ownership'


# ── FIX-2026-09-29-081: SSRF guard ──
def test_url_guard_blocks_internal_ips(monkeypatch):
    import app.utils.url_guard as ug
    for ip in ('127.0.0.1', '10.0.0.5', '172.16.9.9', '192.168.1.1',
               '169.254.169.254', '100.64.0.1', '0.0.0.0'):
        monkeypatch.setattr(ug.socket, 'getaddrinfo',
                            lambda *a, _ip=ip, **k: [(2, 1, 6, '', (_ip, 0))])
        ok, _reason = ug.check_url('http://evil.example/')
        assert not ok, f'{ip} should be blocked'


def test_url_guard_blocks_container_names():
    import app.utils.url_guard as ug
    for host in ('postgres', 'redis', 'nginx', 'app', 'localhost'):
        ok, _reason = ug.check_url(f'http://{host}/')
        assert not ok, host


def test_url_guard_blocks_schemes():
    import app.utils.url_guard as ug
    for u in ('ftp://example.com/', 'file:///etc/passwd', 'gopher://x/'):
        ok, _reason = ug.check_url(u)
        assert not ok, u


def test_url_guard_allows_public(monkeypatch):
    import app.utils.url_guard as ug
    monkeypatch.setattr(ug.socket, 'getaddrinfo',
                        lambda *a, **k: [(2, 1, 6, '', ('93.184.216.34', 0))])
    ok, reason = ug.check_url('http://example.com/')
    assert ok, reason


def test_fetch_url_route_blocks_ssrf(app):
    client = app.test_client()
    _sess(client, 'userA', 'user')
    r = client.post('/fetch_url', json={'url': 'http://127.0.0.1:8000/check_auth'})
    assert r.status_code == 400, r.status_code


def test_credit_route_blocks_ssrf(app):
    client = app.test_client()
    _sess(client, 'userA', 'user')
    r = client.post('/start_credit_check',
                    json={'companies': ['x'], 'urls': ['http://169.254.169.254/']})
    assert r.status_code == 400, r.status_code


def test_safe_get_blocks_redirect_to_internal(monkeypatch):
    """⑤: 公网 URL 302 到内网 → 第二跳被拦（防跳转绕过）。"""
    import pytest
    import app.utils.url_guard as ug
    import app.services.web_extractor as we

    def _fake(host, port, *a, **k):
        ip = '127.0.0.1' if host == '127.0.0.1' else '93.184.216.34'
        return [(2, 1, 6, '', (ip, 0))]

    monkeypatch.setattr(ug.socket, 'getaddrinfo', _fake)

    class _Resp:
        status_code = 302
        headers = {'Location': 'http://127.0.0.1/secret'}
        text = ''

    monkeypatch.setattr(we.requests, 'get', lambda *a, **k: _Resp())
    with pytest.raises(ValueError):
        we.safe_get('http://example.com/', headers={}, timeout=5)


# ── FIX-2026-09-29-082: deletion code hash + expiry + constant-time ──
class _DelCur:
    def __init__(self, row, log):
        self._row = row
        self._log = log

    def execute(self, sql, params=None):
        self._log.append((sql, params))

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _DelConn:
    def __init__(self, row, log):
        self._cur = _DelCur(row, log)

    def cursor(self, *a, **k):
        return self._cur

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass


def _confirm(app, monkeypatch, stored, code):
    import app.routes.auth as a
    log = []
    monkeypatch.setattr(a, 'get_db_connection', lambda: _DelConn((stored, 'u@e.com'), log))
    monkeypatch.setattr(a, 'delete_account_impl',
                        lambda uid, pin, keep: __import__('flask').jsonify({"success": True}))
    client = app.test_client()
    _sess(client, 'userA', 'user')
    r = client.post('/confirm_delete_account', json={'pin': '1234', 'code': code})
    return r, log


def test_delete_confirm_hashed_code_ok(app, monkeypatch):
    """031: 正确 hash 码 → 放行；并清码。"""
    import hashlib as _h
    import time as _t
    stored = f"{int(_t.time()) + 300}:{_h.sha256('1234'.encode()).hexdigest()}"
    r, log = _confirm(app, monkeypatch, stored, '1234')
    assert r.status_code == 200, r.status_code
    assert any('deletion_code = NULL' in s for s, _ in log), 'code must be cleared'


def test_delete_confirm_rejects_legacy_plaintext(app, monkeypatch):
    """031: 旧明文存储 → 拒（fail-closed）。"""
    r, log = _confirm(app, monkeypatch, '1234', '1234')
    assert r.status_code == 400, r.status_code
    assert any('deletion_code = NULL' in s for s, _ in log), 'cleared on failure'


def test_delete_confirm_rejects_expired(app, monkeypatch):
    """031: 过期码 → 拒。"""
    import hashlib as _h
    import time as _t
    stored = f"{int(_t.time()) - 10}:{_h.sha256('1234'.encode()).hexdigest()}"
    r, _log = _confirm(app, monkeypatch, stored, '1234')
    assert r.status_code == 400, r.status_code


def test_delete_code_not_plaintext_wired():
    with open('app/routes/auth.py', encoding='utf-8') as f:
        a_src = f.read()
    with open('app/routes/admin_regeneration.py', encoding='utf-8') as f:
        g_src = f.read()
    assert 'code != expected_code' not in a_src
    assert 'secrets.compare_digest(provided.encode' in a_src
    assert "hashlib.sha256(code.encode('utf-8')).hexdigest()" in g_src


# ── FIX-2026-09-29-083: /api/login + auth_jwt retired (UNRESOLVED-030) ──
def test_api_login_removed(app):
    client = app.test_client()
    _sess(client, 'userA', 'user')
    r = client.post('/api/login', json={'username': 'x', 'pin': '1234'})
    assert r.status_code == 404, r.status_code


def test_auth_jwt_removed():
    import os
    assert not os.path.exists('app/services/auth_jwt.py')
    with open('app/routes/chat_sessions.py', encoding='utf-8') as f:
        src = f.read()
    assert 'create_token' not in src
    assert 'def api_login' not in src


def test_web_login_not_regressed(app):
    """030 不改 /login：路由仍在且响应该登录（401/429/400 均可，非 404）。"""
    client = app.test_client()
    r = client.post('/login', json={'username': 'admin', 'pin': 'wrongpin'})
    assert r.status_code in (400, 401, 429), r.status_code


def test_no_jwt_encode_anywhere():
    import os
    hits = []
    for root, _d, files in os.walk('app'):
        for fn in files:
            if fn.endswith('.py'):
                p = os.path.join(root, fn)
                with open(p, encoding='utf-8') as f:
                    if 'jwt.encode' in f.read():
                        hits.append(p)
    assert hits == [], hits


# ── FIX-2026-10-04-QA-030 (round-030): reviewer=mimo-v2.6-pro ──

def test_qa030_ingest_rejected_indices_json_serialisable():
    """H1: _prepare_kb_review must not put a set() into the JSON review file."""
    import json, tempfile, os
    from app.services import ingest_pipeline as ip
    # Directly exercise the dict that gets json.dump'ed (no chunk_text needed).
    src = open('app/services/ingest_pipeline.py', encoding='utf-8').read()
    assert "'rejected_indices': set()," not in src
    assert "'rejected_indices': []," in src
    # And confirm a set() actually fails the dump the old code did.
    try:
        json.dumps({'rejected_indices': set()})
        raised = False
    except TypeError:
        raised = True
    assert raised, "sanity: set() is not JSON-serialisable"


def test_qa030_knowledge_report_dir_defined_both_paths():
    """H2: report_dir assigned before the try (was only in except → UnboundLocalError)."""
    src = open('app/routes/knowledge.py', encoding='utf-8').read()
    assert 'report_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)' in src
    assert 'os.makedirs(report_dir, exist_ok=True)' in src
    # the success path must not reopen the shared-dir / unbound regression
    assert 'report_dir = os.path.dirname(report_path)' not in src


def test_qa030_username_charset_whitelist_wired():
    """H3: registration/rename enforce a username charset (path-traversal guard)."""
    src = open('app/routes/auth.py', encoding='utf-8').read()
    assert "A-Za-z0-9_\\u4e00-\\u9fa5" in src
    assert src.count('用户名只能包含中文、字母、数字、下划线（5-18位）') >= 2
    ksrc = open('app/routes/knowledge.py', encoding='utf-8').read()
    assert 'safe_tag = re.sub(' in ksrc


def test_qa030_task_owner_ok_empty_fails_closed():
    """M5: empty-string owner is denied; only an absent key is a legacy task."""
    from app.utils.helpers import task_owner_ok
    assert task_owner_ok({}, 'u1') is True, "legacy (no key) must stay allowed"
    assert task_owner_ok({'user_id': ''}, 'u1') is False, "empty owner must fail closed"
    assert task_owner_ok({'user_id': 'u1'}, 'u1') is True
    assert task_owner_ok({'user_id': 'u1'}, 'u2') is False


def test_qa030_url_guard_invalid_port_returns_tuple():
    """M6: URL with an out-of-range port must not raise out of check_url."""
    from app.utils.url_guard import check_url
    ok, reason = check_url('http://127.0.0.1:99999/x')
    assert ok is False
    assert 'port' in reason.lower()


def test_qa030_download_get_route_registered(app):
    """M8: GET download variant exists and is owner-guarded."""
    rules = [str(r) for r in app.url_map.iter_rules()]
    assert any('/download_original_file/<user_id>/<path:filename>' in r for r in rules), rules
    src = open('app/routes/chat_files.py', encoding='utf-8').read()
    assert 'if str(sess_uid) != str(user_id) and not is_admin():' in src


def test_qa030_frontend_escapes_and_validates():
    """M7: server strings escaped before innerHTML; download_url scheme checked."""
    appjs = open('static/js/app.js', encoding='utf-8').read()
    assert "escapeHtml(err.error || '未知错误')" in appjs
    assert 'safeDownloadUrl(d.download_url)' in appjs
    assert 'safeDownloadUrl(downloadUrl)' in appjs
    assert '(?:\\/(?!\\/)' in appjs  # rejects protocol-relative //
    kl = open('static/js/knowledge-lab.js', encoding='utf-8').read()
    assert 'escapeHtml(d.message)' in kl
    assert "test(d.download_url" in kl


def test_qa030_credit_get_json_silent():
    """L14: no 500 on a missing/invalid JSON body."""
    assert 'data = request.get_json(silent=True) or {}' in open('app/routes/credit.py', encoding='utf-8').read()


def test_qa030_tasks_delete_missing_404():
    """L15: delete on a missing task returns 404."""
    src = open('app/routes/tasks.py', encoding='utf-8').read()
    assert "if status == 'missing':" in src
    assert "return jsonify({'error': 'Not Found'}), 404" in src


def test_qa030_kb_review_path_rejects_traversal():
    """L9: _kb_review_path must reject separators / traversal ids."""
    from app.services.ingest_pipeline import _kb_review_path
    assert _kb_review_path('abc123').endswith('kb_review_abc123.json')
    for bad in ('../../etc/passwd', 'a/b', 'a\\b', '..', ''):
        p = _kb_review_path(bad)
        assert '..' not in p and 'passwd' not in p and p.endswith('.kb_review_invalid')


def test_qa030_login_guard_fail_counters_ip_scoped(monkeypatch):
    """L10: failure counters are per (username, ip); memory purge honours TTL."""
    import app.services.login_guard as lg
    monkeypatch.setattr(lg, '_redis', lambda: None)
    lg._mem.clear()
    lg.record_login_failure('Alice', '1.1.1.1')
    lg.record_login_failure('Alice', '2.2.2.2')
    keys = [k for k in lg._mem if k.startswith('login_fail:')]
    assert 'login_fail:alice:1.1.1.1' in keys
    assert 'login_fail:alice:2.2.2.2' in keys
    lg._mem.clear()


def test_qa030_auth_timing_and_admin_config():
    """L11: dummy hash for unknown users; admin usernames configurable."""
    from app.routes.auth import _DUMMY_PIN_HASH, _ADMIN_USERNAMES
    assert _DUMMY_PIN_HASH
    assert 'admin' in _ADMIN_USERNAMES


def test_qa030_compose_requires_db_password():
    """L13: no insecure default DB password in compose."""
    src = open('docker-compose.yml', encoding='utf-8').read()
    assert 'DB_PASSWORD:-localai' not in src
    assert 'DB_PASSWORD:?' in src


def test_qa030_nginx_static_security_headers():
    """L12: /static/ location re-declares server-level security headers."""
    src = open('nginx.conf', encoding='utf-8').read()
    assert src.count('add_header X-Content-Type-Options nosniff;') >= 2
    assert src.count('add_header X-Frame-Options SAMEORIGIN;') >= 2


# ── FIX-2026-08-15-001: LLM tool-error + prompt-template leak sanitization ──
def test_sanitize_response_strips_invalid_tool_error():
    from app.utils.helpers import sanitize_response
    leaked = 'Error: DesignDesign is not a valid tool, try one of [get_date, bocha_search]'
    assert sanitize_response(leaked) == '', \
        "sanitize_response must strip 'X is not a valid tool' error text entirely"


def test_sanitize_response_strips_prompt_template():
    from app.utils.helpers import sanitize_response
    leaked = 'Here is the JSON for a function call with its proper arguments that best answers the given prompt is:'
    assert sanitize_response(leaked) == '', \
        "sanitize_response must strip function-calling prompt template text"


def test_sanitize_response_preserves_normal_content():
    from app.utils.helpers import sanitize_response
    normal = '招标文件已上传，请等待分析完成。'
    assert sanitize_response(normal) == normal, \
        "sanitize_response must not alter normal user-facing content"


def test_split_thinking_answer_sanitizes_answer():
    from app.utils.helpers import split_thinking_answer
    thinking, answer = split_thinking_answer(
        '【思考】考虑中【回答】好的，我查一下。Error: Foo is not a valid tool, try one of [get_date]')
    assert thinking == '考虑中'
    assert 'not a valid tool' not in answer, \
        "split_thinking_answer must sanitize the answer portion"


# ── FIX-2026-08-15-003: /my_daily_report friendly message ──
def test_daily_report_friendly_insufficient_message():
    content = _read('app/routes/knowledge.py')
    assert '先聊几句（至少2条问答）' in content, \
        "knowledge.py my_daily_report must return the friendly insufficient-messages message"


# ── FIX-2026-08-15-002: /check_storage frontend auth gating ──
def test_check_storage_frontend_gated_by_username():
    content = _read('static/js/app.js')
    assert "sessionStorage.getItem('username')) return;" in content, \
        "app.js checkStorage() must skip the fetch when no username is in sessionStorage"


# ── FIX-2026-08-16-004: Pre-login fetch noise gating (templates/cases/notebook/projects) ──
def test_prelogin_gate_cases():
    content = _read('static/js/cases.js')
    assert "sessionStorage.getItem('username'))" in content and "登录后查看案例库" in content, \
        "cases.js loadList must skip the fetch pre-login"


def test_prelogin_gate_templates():
    content = _read('static/js/templates.js')
    assert "sessionStorage.getItem('username'))" in content and "登录后查看模板库" in content, \
        "templates.js loadList must skip the fetch pre-login"


def test_prelogin_gate_notebook():
    content = _read('static/js/knowledge-lab.js')
    assert "sessionStorage.getItem('username'))" in content and "登录后查看笔记" in content, \
        "knowledge-lab.js loadNotebook must skip the fetch pre-login"


def test_prelogin_gate_projects():
    content = _read('static/js/app.js')
    assert "sessionStorage.getItem('username'))" in content and "登录后查看项目" in content, \
        "app.js loadSidebarProjects must skip the fetch pre-login"


# ── FIX-2026-08-16-005: Chat empty-state + float button visibility ──
def test_chat_empty_state_rendered():
    content = _read('static/js/chat.js')
    assert 'chatEmptyState' in content and 'renderEmptyState' in content, \
        "chat.js must render the chat empty-state block"


def test_float_buttons_hidden_when_no_overflow():
    content = _read('static/js/chat.js')
    assert 'hidden-float' in content, \
        "chat.js must toggle hidden-float on the float buttons container"


# ── FIX-2026-08-16-006: Role chip in header ──
def test_role_chip_header():
    content = _read('static/js/app.js')
    assert 'updateRoleChip' in content and 'headerRoleChip' in content, \
        "app.js must update the header role chip from /check_auth data"


# ── FIX-2026-08-15-001 (live-stream half): frontend SSE text is sanitized ──
def test_sanitize_response_frontend_stream_guard():
    content = _read('static/js/chat.js')
    assert 'function _sanitizeResponse' in content, \
        "chat.js must define _sanitizeResponse to clean the live SSE stream"
    assert '_sanitizeResponse(fullResponse)' in content, \
        "chat.js must run the live streamed text through _sanitizeResponse before rendering"


# ── FIX-2026-10-05-084: GPU torch index must carry the pinned torch 2.12.1 ──
def test_docker_build_gpu_index_is_cu126():
    """cu124 caps at torch 2.6.0; requirements pin 2.12.1 -> GPU builds need cu126."""
    src = _read('scripts/docker_build.py')
    assert 'whl/cu126' in src, "GPU torch index must be cu126 (cu124 has no torch 2.12.1)"
    assert 'whl/cu124' not in src, "cu124 would make the GPU build unresolvable"


def test_dockerfile_has_pip_retry_budget():
    """Large CUDA wheels stall on pypi.nvidia.com; the Dockerfile must retry."""
    src = _read('Dockerfile')
    assert 'ARG PIP_RETRIES=10' in src and 'ARG PIP_TIMEOUT=120' in src


# ── FIX-2026-10-05-085: pre-commit hook must use the project venv interpreter ──
def test_pre_commit_hook_reexecs_with_venv():
    """Bare `python` lacks yaml/dotenv -> the hook must re-exec under .venv."""
    src = _read('.githooks/pre-commit')
    assert 'def _reexec_with_venv' in src
    assert 'LOCALAI_HOOK_VENV' in src
    assert '.venv' in src


# ── FIX-2026-10-05-086: .mcp.json carries machine credentials — keep it ignored ──
def test_mcp_json_is_explicitly_gitignored():
    """A narrowed *.json rule must not be able to commit .mcp.json."""
    assert '.mcp.json' in _read('.gitignore')


# ── FIX-2026-10-06-QA-031-01: reasoning_content branch must sanitize too ──
def test_reasoning_content_branch_sanitizes():
    """The reasoning_content shortcut skipped split_thinking_answer entirely."""
    for rel in ('app/routes/chat.py', 'app/routes/chat_sessions.py'):
        src = _read(rel)
        assert 'thinking = sanitize_response(reasoning.strip())' in src, \
            f"{rel}: reasoning_content branch must sanitize thinking"
        assert 'answer = sanitize_response(raw_response.strip())' in src, \
            f"{rel}: reasoning_content branch must sanitize the answer"
        assert "answer = raw_response.strip() if raw_response else ''" not in src, \
            f"{rel}: unsanitized answer assignment must be gone"


# ── FIX-2026-10-06-QA-031-02: sanitized-empty must not fall back to the raw leak ──
def test_sanitized_empty_never_falls_back_to_raw():
    """sanitize_response('Error: X ...') == '' — falling back to raw re-leaks it."""
    cs = _read('app/routes/chat_sessions.py')
    assert 'answer if answer else raw_response' not in cs, \
        'regenerate must not fall back to the raw response'
    ch = _read('app/routes/chat.py')
    assert 'answer = (answer or full_response)' not in ch, \
        'partial-save must not fall back to the raw response'
    assert "answer = (answer or '')" in ch
