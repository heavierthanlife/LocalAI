#!/usr/bin/env python
"""T2-flavored real deep trips for the e2e UI audit.

Each trip runs against the live e2e stack with the admin session (page.request
shares the context cookies). Raises AssertionError on failure → driver records
into the ledger. These exercise real backend pipelines the buttons call:
  trip_clearance_run    : /clearance/run on fixture bid+tender → poll status → report
  trip_vl_test          : /admin/vl_test real image → OCR + description + reasoning
  trip_plagiarism       : /batch/plagiarism/compare two bid fixtures
  trip_provider_refresh : /llm_providers/<pid>/models?refresh=1
"""
import os
import sys
import json
import time
import uuid
import ssl
import urllib.request

BASE = os.environ.get("AUDIT_BASE", "https://localhost:4443")
_SSLCTX = ssl._create_unverified_context()


def _cookie_header(page):
    try:
        return "; ".join(f"{c['name']}={c['value']}" for c in page.context.cookies())
    except Exception:
        return ""


def _multipart_post(url, cookie, fields, files, timeout=60):
    """Raw multipart POST (urllib) — Playwright request multipart with file lists flaky."""
    b = uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():
        body += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for name, filename, data in files:
        body += (f"--{b}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                 f"filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
        body += data + b"\r\n"
    body += f"--{b}--\r\n".encode()
    req = urllib.request.Request(url, data=bytes(body), method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={b}",
        "Cookie": cookie,
    })
    with urllib.request.urlopen(req, timeout=timeout, context=_SSLCTX) as r:
        return r.status, r.read().decode("utf-8", "replace")


def _j(resp):
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text()[:200]}


def _fp(path):
    with open(path, "rb") as fh:
        return {"name": os.path.basename(path),
                "mimeType": "application/octet-stream",
                "buffer": fh.read()}


def trip_clearance_run(page, ledger, fixtures):
    _a = open(os.path.join(fixtures, "bid_alpha.docx"), "rb").read()
    _b = open(os.path.join(fixtures, "bid_beta.docx"), "rb").read()
    _t = open(os.path.join(fixtures, "tender.docx"), "rb").read()
    opts = json.dumps({"indicator_analysis": True, "cross_comparison": True,
                       "compliance_check": True, "ai_review": False, "tech_seal_check": False})
    status, txt = _multipart_post(
        BASE + "/clearance/run", _cookie_header(page),
        {"options": opts, "project_id": ""},
        [("files", "bid_alpha.docx", _a), ("files", "bid_beta.docx", _b),
         ("tender_file", "tender.docx", _t)], timeout=60)
    d = json.loads(txt) if txt.strip().startswith("{") else {"raw": txt[:200]}
    assert 200 <= status < 300 and d.get("success"), f"clearance start failed: {status} {d}"
    task_id = d.get("task_id")
    assert task_id, f"no task_id: {d}"
    deadline = time.time() + 240
    last = {}
    while time.time() < deadline:
        r = page.request.get(BASE + "/clearance/status/" + task_id, timeout=30000)
        last = _j(r)
        st = (last.get("status") or "").lower()
        if st in ("done", "completed", "success"):
            break
        if st in ("failed", "error"):
            raise AssertionError(f"clearance failed: {last.get('message') or last}")
        time.sleep(5)
    st = (last.get("status") or "").lower()
    assert st in ("done", "completed", "success"), f"clearance timeout/status={st}"
    res = last.get("result")
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except Exception:
            res = {}
    report = {}
    if isinstance(res, dict):
        report = res.get("report") or res
    assert report.get("indicators"), f"clearance report missing indicators; result_keys={list(res.keys()) if isinstance(res, dict) else type(res)}"
    ledger.add("clearance-run", {"kind": "trip", "text": "clearance e2e"}, True,
               result=f"task={task_id} status={st} indicators={len(report.get('indicators', []))}")


def trip_vl_test(page, ledger, fixtures):
    img = os.path.join(fixtures, "vl_test.png")
    with open(img, "rb") as fh:
        data = fh.read()
    resp = page.request.post(BASE + "/admin/vl_test",
                             multipart={"image": {"name": "vl_test.png",
                                                  "mimeType": "image/png",
                                                  "buffer": data}},
                             timeout=180000)
    d = _j(resp)
    if resp.ok and d.get("status") != "ok":
        err = json.dumps(d, ensure_ascii=False)
        if any(k in err for k in ("503", "429", "ResourceExhausted", "rate limit")):
            ledger.add("vl-test", {"kind": "trip", "text": "vl real image"}, False,
                       reason="provider-rate-limited")
            return
    assert resp.ok and d.get("status") == "ok", f"vl_test failed: {resp.status} {d}"
    dd = d.get("data", {})
    desc = dd.get("description") or ""
    assert not desc.startswith("⚠️"), f"VL desc error: {desc[:120]}"
    ledger.add("vl-test", {"kind": "trip", "text": "vl real image"}, True,
               result=f"provider={dd.get('provider')} consistent={dd.get('consistent')} ocr={'有' if dd.get('ocr') else '无'}")


def trip_plagiarism(page, ledger, fixtures):
    _a = open(os.path.join(fixtures, "bid_alpha.docx"), "rb").read()
    _b = open(os.path.join(fixtures, "bid_beta.docx"), "rb").read()
    status, txt = _multipart_post(
        BASE + "/batch/plagiarism/compare", _cookie_header(page), {},
        [("files", "bid_alpha.docx", _a), ("files", "bid_beta.docx", _b)], timeout=120)
    d = json.loads(txt) if txt.strip().startswith("{") else {"raw": txt[:200]}
    assert 200 <= status < 300 and d.get("success") is True, f"plagiarism failed: {status} {d}"
    ledger.add("plagiarism-compare", {"kind": "trip", "text": "plagiarism e2e"}, True,
               result="ok")


def trip_provider_refresh(page, ledger, fixtures):
    resp = page.request.get(BASE + "/llm_providers/openrouter/models?refresh=1", timeout=60000)
    d = _j(resp)
    models = d.get("models") if isinstance(d.get("models"), list) else None
    assert models is not None, f"provider refresh shape: {d}"
    ledger.add("provider-refresh", {"kind": "trip", "text": "models refresh"}, True,
               result=f"models={len(models)} stale={d.get('stale')}")
