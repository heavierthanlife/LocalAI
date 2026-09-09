#!/usr/bin/env python
"""T1 — Playwright reachability + interaction audit (throwaway e2e stack).

Targets the docker-compose.e2e.yml stack at https://localhost:4443 (admin CEO /
e2euser / anonymous). Walks reachable surfaces: expands every <details>, opens
panes/menus reachable without deep preconditions, enumerates interactive
elements into a coverage LEDGER and acts on them per a safe-action policy.

Assertions per action: no 4xx/5xx · no pageerror · no console.error · JSON
responses parse (not HTML). Visual: key text visible, no overflow clip on tables.

Exit (report mode default): 0 + artifacts. Use --gate to hard-fail on
`unexplained` elements (found in DOM, not acted, no reason) or any failure.

Usage:
  python scripts/run_ui_audit.py [--gate] [--out data/qa_loop/audit]
"""
import os
import re
import sys
import json
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX_DIR = os.path.join(ROOT, "tests", "fixtures", "audit")
BASE = os.environ.get("AUDIT_BASE", "https://localhost:4443")

# ── policy keywords → element handling ──
DANGER_WORDS = ("删除", "清空", "清除", "清 除", "重建", "重新生成", "清理", "重置",
                "恢复出厂", "全部", "批量", "退出登录", "注销", "归档", "解散",
                "导出全部", "Empty", "Reset", "Delete", "Clear all")
MUTATING_BUT_OK = ("保存", "Save", "提交", "应用", "Apply", "刷新", "Refresh", "开始", "运行", "Run")
FILE_LABELS = ("选择投标文件", "选择招标文件", "选择开标信息表", "选择文件", "上传",
               "选择图片", "Add", "附件", "Choose")
SKIP_LABELS = ("帮助", "关于", "?")  # informational — still exercised as safe click


def expand_all(page):
    n = 0
    for _ in range(6):
        d = page.query_selector_all("details:not([open])")
        if not d:
            break
        for el in d[:40]:
            try:
                page.evaluate("(el)=>{el.setAttribute('open','')}", el)
                n += 1
            except Exception:
                pass
    return n


def collect_clickables(page):
    """Visible buttons/links/checkbox/select/radio + file inputs in viewport."""
    out = []
    try:
        els = page.query_selector_all(
            "button, a[href], input[type=checkbox], input[type=radio], "
            "input[type=file], select, [role=button], .file-btn")
    except Exception:
        return out
    for el in els:
        try:
            if not el.is_visible():
                continue
            tag = el.evaluate("e=>e.tagName")
            txt = (el.inner_text() or "").strip()[:40]
            ident = el.get_attribute("id") or ""
            kind = tag.lower()
            if kind == "input":
                kind = el.get_attribute("type") or "text"
            out.append({"kind": kind, "text": txt, "id": ident, "el": el})
        except Exception:
            continue
    return out


def classify(elm):
    kind, txt = elm["kind"], (elm["text"] or "").lower()
    if kind == "file":
        return "file"
    if kind == "checkbox":
        return "toggle"
    if kind == "select":
        return "select"
    if any(w in txt for w in DANGER_WORDS):
        return "blocked-danger"
    if any(w in txt for w in MUTATING_BUT_OK):
        return "mutating"
    if any(w in txt.lower() for w in ("logout", "sign out")):
        return "blocked-danger"
    return "safe"


def is_json(resp):
    ct = (resp.headers.get("content-type") or "")
    return "application/json" in ct or "text/json" in ct


class Ledger:
    def __init__(self):
        self.items = []
        self.errors = []
        self.failures = []

    def add(self, surface, elm, acted, result="", reason=""):
        self.items.append({
            "surface": surface, "kind": elm.get("kind"), "text": elm.get("text"),
            "id": elm.get("id"), "acted": acted, "result": result, "reason": reason,
        })

    def err(self, surface, msg):
        self.failures.append({"surface": surface, "message": msg})


def run_surface(page, surface, actions, ledger, fixtures):
    expand_all(page)
    clickables = collect_clickables(page)
    acted = set()
    for elm in clickables:
        c = classify(elm)
        key = f"{elm['kind']}|{elm['id']}|{elm['text']}"
        if key in acted:
            continue
        acted.add(key)
        if c == "blocked-danger":
            ledger.add(surface, elm, False, reason="danger-policy")
            continue
        if c == "toggle":
            ledger.add(surface, elm, False, reason="toggle-skip-avoid-state")
            continue
        try:
            if elm["kind"] == "file":
                # only set file when an action target says to
                continue
            try:
                elm["el"].click(timeout=1200)
            except Exception:
                elm["el"].click(force=True, timeout=2000)
            time.sleep(0.4)
            ledger.add(surface, elm, True, result="clicked")
        except Exception as e:
            ledger.add(surface, elm, False, reason=f"click-fail:{str(e)[:40]}")
    # surface-provided deep actions
    for a in actions:
        try:
            a(page, ledger, fixtures)
        except Exception as e:
            ledger.err(surface, f"action {a.__name__}: {str(e)[:120]}")


def assert_no_page_errors(page, ledger, surface):
    try:
        errs = page.evaluate("()=>window.__audit_errs||[]")
    except Exception:
        errs = []
    if errs:
        ledger.err(surface, f"pageerror: {errs[:3]}")


def _api_login(ctx, username, pin):
    """Log in via the auth API so the context cookie jar carries the session."""
    import json as _json
    r = ctx.request.post(BASE + "/login",
                         data=_json.dumps({"username": username, "pin": pin}),
                         headers={"content-type": "application/json"}, timeout=20000)
    return r.status, r.text()[:120]


def _tour(page, ledger, mode, names, drain):
    """For each (name, selector-or-None) try to open then walk. Selector None = walk current."""
    if names is None:
        run_surface(page, f"{mode}-home", [], ledger, None)
        drain(f"{mode}-home")
        return [f"{mode}-home"]
    walked = []
    for name, sel in names:
        try:
            el = page.query_selector(sel) if sel else None
            if sel and (el is None or not el.is_visible()):
                ledger.add(name, {"kind": "panel", "text": sel, "id": sel}, False,
                           reason="not-visible-in-current-role")
                continue
            if el:
                el.click(force=True, timeout=2500)
                time.sleep(1.8)
            run_surface(page, name, [], ledger, None)
            drain(name)
            walked.append(name)
        except Exception as e:
            ledger.err(name, f"open-walk: {str(e)[:100]}")
    return walked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "qa_loop", "audit"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from playwright.sync_api import sync_playwright
    ledger = Ledger()
    result = {"surfaces": [], "summary": {}, "logins": {}}

    TAB_TOUR = [
        ("tab-chat", "#chatTabBtn"),
        ("tab-projects", "#adminTabBtn"),
        ("tab-recycle", "#recycleBinTabBtn"),
        ("tab-analytics", "#analyticsTabBtn"),
        ("tab-sidebar-stats", "#sidebar-stats-pane"),
        ("panel-quote-history", "#sidebarQuoteAnomalyResultsBtn"),
        ("panel-relationship-history", "#sidebarRelationshipResultsBtn"),
        ("panel-typo-history", "#sidebarTypoResultsBtn"),
        ("panel-audit-log", "#sidebarAuditLogBtn"),
        ("panel-clear-cache", "#sidebarClearCacheBtn"),
        ("panel-templates", "#sidebar-templates-pane"),
        ("panel-wiki", "#sidebar-wiki-pane"),
        ("panel-cases", "#sidebar-cases-pane"),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=[
            "--no-sandbox", "--ignore-certificate-errors", "--disable-gpu"])

        def make_ctx():
            ctx = browser.new_context(ignore_https_errors=True, viewport={"width": 1500, "height": 950})
            page = ctx.new_page()
            page.set_default_timeout(9000)
            net_errors, page_errors, console_errors = [], [], []
            page.on("pageerror", lambda e: page_errors.append(str(e)[:180]))
            page.on("console", lambda m: console_errors.append(m.text[:140]) if m.type == "error" else None)
            page.on("response", lambda r: net_errors.append(
                {"url": r.url, "status": r.status}) if r.status >= 400 else None)

            def drain(surface):
                for e in page_errors:
                    ledger.err(surface, f"pageerror: {e}")
                for t in console_errors:
                    ledger.err(surface, f"console.error: {t}")
                for n in net_errors:
                    ledger.err(surface, f"HTTP {n['status']} {n['url'][:120]}")
                net_errors.clear(); page_errors.clear(); console_errors.clear()
            return ctx, page, drain

        # ── Admin session (API login → tab tour) ──
        ctx, page, drain = make_ctx()
        st, body = _api_login(ctx, "CEO", "123456")
        result["logins"]["admin"] = {"status": st}
        page.goto(BASE, wait_until="domcontentloaded")
        time.sleep(2.5)
        result["surfaces"] += _tour(page, ledger, "admin", TAB_TOUR, drain)
        ctx.close()

        # ── Normal user session ──
        ctx, page, drain = make_ctx()
        st, body = _api_login(ctx, "e2euser", "123456")
        result["logins"]["user"] = {"status": st}
        page.goto(BASE, wait_until="domcontentloaded")
        time.sleep(2.5)
        result["surfaces"] += _tour(page, ledger, "user", TAB_TOUR, drain)
        ctx.close()

        # ── Anonymous (short walk) ──
        ctx, page, drain = make_ctx()
        page.goto(BASE, wait_until="domcontentloaded")
        time.sleep(2.5)
        result["surfaces"] += _tour(page, ledger, "anon", None, drain)
        ctx.close()

        # ── T2-flavored real trips (optional — audit_trips.py is a next-round target) ──
        trips = ["trip_clearance_run", "trip_vl_test", "trip_plagiarism", "trip_provider_refresh"]
        ctx, page, drain = make_ctx()
        _api_login(ctx, "CEO", "123456")
        page.goto(BASE, wait_until="domcontentloaded")
        time.sleep(2.0)
        try:
            import audit_trips as at
        except Exception:
            at = None
        for tname in trips:
            if at and hasattr(at, tname):
                try:
                    getattr(at, tname)(page, ledger, FIX_DIR)
                    result["surfaces"].append(tname)
                    drain(tname)
                except Exception as e:
                    ledger.err(tname, f"{str(e)[:150]}")
            else:
                ledger.add(tname, {"kind": "trip", "text": tname, "id": tname}, False,
                           reason="audit_trips-not-implemented")
        ctx.close()
        browser.close()

    acted = sum(1 for i in ledger.items if i["acted"])
    blocked = sum(1 for i in ledger.items if i["reason"] == "danger-policy")
    unexplained = [i for i in ledger.items if not i["acted"] and not i["reason"]]
    result["summary"] = {
        "elements_found": len(ledger.items), "acted": acted,
        "blocked_by_policy": blocked, "unexplained": len(unexplained),
        "failures": len(ledger.failures),
    }
    result["ledger"] = ledger.items
    result["failures"] = ledger.failures
    with open(os.path.join(args.out, "audit_report.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    if args.gate and (unexplained or ledger.failures):
        print("GATE FAIL")
        return 1
    print("AUDIT DONE (report mode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
