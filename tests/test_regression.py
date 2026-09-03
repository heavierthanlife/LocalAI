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
    """RiskScorer must use 0.375/0.375/0.25 weights + ≥80% text gate."""
    from app.services.batch_orchestrator import RiskScorer
    assert RiskScorer.WEIGHTS['key_info'] == 0.375
    assert RiskScorer.WEIGHTS['file_attr'] == 0.375
    assert RiskScorer.WEIGHTS['text_sim'] == 0.25
    assert RiskScorer.WEIGHTS['image_sim'] == 0.0
    # text <80% gate → contributes 0
    assert RiskScorer.compute(0, 0, 70, 0) == 0.0
    # text ≥80% → contributes 0.25 * 80 = 20
    assert RiskScorer.compute(0, 0, 80, 0) == 20.0


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
    # typo 指标不再误报
    assert snap['indicators']['economic_error_similar']['score'] == 0.0, \
        "正常文档 typo 指标不应误报"


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
