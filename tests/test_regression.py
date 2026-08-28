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
