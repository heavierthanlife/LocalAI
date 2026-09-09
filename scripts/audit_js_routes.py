#!/usr/bin/env python
"""T0 — Static JS-URL ↔ Flask-route cross-check (FIX-2026-09-09-023).

Would have caught /admin/vl_test (frontend fetch → route never implemented).

Call-anchored extractor for fetch/axios/XHR/$.ajax/window.open/href/action.
Classification:
  app            : absolute path (leading '/'), no trailing-slash, path-like
  dynamic        : template-literal OR trailing-slash concat prefix (needs runtime/T1)
  external       : http(s)/data/blob/…/anchor
  nonurl         : bare word captured from non-call context (dropped)

app paths resolved against create_app().url_map — path-template + method aware.
Report buckets:
  no-route          : path matched zero rules → REAL dead-link candidate
  method-mismatch   : path exists but inferred method not allowed (low-confidence)
Usage: python scripts/audit_js_routes.py [--json out.json]   (report mode, exit 0)
"""
import os
import re
import sys
import json
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SECRET_KEY", "audit-secret-key")
os.environ.setdefault("WTF_CSRF_ENABLED", "false")
sys.path.insert(0, ROOT)

_MATCHERS = [
    (re.compile(r"""fetch\(\s*['"`]([^'"`]+)['"`]"""), None),
    (re.compile(r"""axios\.(get|post|put|patch|delete)\(\s*['"]([^'"]+)['"]"""), "verb"),
    (re.compile(r"""\.open\(\s*['"]([A-Z]+)['"]\s*,\s*['"]([^'"]+)['"]"""), "open"),
    (re.compile(r"""\.ajax\(\s*\{[^}]*?url\s*:\s*['"]([^'"]+)['"]"""), None),
    (re.compile(r"""window\.open\(\s*['"]([^'"]+)['"]"""), None),
    (re.compile(r"""(?:href|action)=["']([^"']+)["']"""), None),
]

_TRAILING_SLASH = re.compile(r"/$")


def _gather_sources():
    return (sorted(glob.glob(os.path.join(ROOT, "static", "js", "*.js")))
            + sorted(glob.glob(os.path.join(ROOT, "templates", "*.html"))))


def _extract(text, is_html):
    hits = []
    for rx, mode in _MATCHERS:
        for m in rx.finditer(text):
            if mode == "verb":
                url, method = m.group(2), m.group(1).upper()
            elif mode == "open":
                url, method = m.group(2), m.group(1).upper()
            else:
                url, method = m.group(1), None
            url = url.strip()
            if not url or url.startswith("#"):
                continue
            if is_html and url.startswith(("/static", "/favicon", "/icons", "javascript:")):
                continue
            hits.append((url, method))
    return hits


def _classify(url):
    low = url.lower()
    if low.startswith(("http://", "https://", "//", "data:", "blob:", "mailto:", "tel:", "javascript:", "ws:", "wss:", "file:")):
        return "external"
    if not url.startswith("/"):
        return "nonurl"  # bare word / relative non-call — not an app route
    if "${" in url or "${" in url or _TRAILING_SLASH.search(url):
        return "dynamic"
    if re.search(r"\s", url):
        return "nonurl"
    return "app"


def _iter_rules(app):
    return list(app.url_map.iter_rules())


def _path_match(rule_path, path):
    rp = [p for p in rule_path.split("/") if p]
    pp = [p for p in path.split("/") if p]
    if len(rp) != len(pp):
        return False
    return all(r == p or (r.startswith("<") and r.endswith(">")) for r, p in zip(rp, pp))


def main():
    from app import create_app
    app = create_app()
    rules = _iter_rules(app)
    reports = {"no_route": [], "method_mismatch": [], "dynamic": [], "external": 0, "matched": 0, "nonurl": 0}
    seen = set()

    for fpath in _gather_sources():
        rel = os.path.relpath(fpath, ROOT).replace("\\", "/")
        is_html = rel.endswith(".html")
        try:
            text = open(fpath, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for url, method in _extract(text, is_html):
            cls = _classify(url)
            key = (rel, url)
            if key in seen:
                continue
            seen.add(key)
            if cls == "external":
                reports["external"] += 1
                continue
            if cls == "nonurl":
                reports["nonurl"] += 1
                continue
            if cls == "dynamic":
                reports["dynamic"].append({"file": rel, "url": url, "method": method})
                continue
            path = url.split("?")[0].split("#")[0].rstrip("/") or "/"
            meth = (method or "GET").upper()
            matched_rule = None
            for rule in rules:
                if _path_match(rule.rule, path):
                    matched_rule = rule
                    break
            if matched_rule is None:
                reports["no_route"].append({"file": rel, "url": url, "method": meth})
            else:
                if meth in matched_rule.methods or meth == "GET" and "HEAD" in matched_rule.methods:
                    reports["matched"] += 1
                else:
                    reports["method_mismatch"].append(
                        {"file": rel, "url": url, "method": meth, "route": matched_rule.rule,
                         "allowed": sorted(matched_rule.methods)})

    print("=" * 80)
    print(f"T0: matched={reports['matched']} no_route={len(reports['no_route'])} "
          f"method_mismatch={len(reports['method_mismatch'])} dynamic={len(reports['dynamic'])} "
          f"external={reports['external']} nonurl={reports['nonurl']}")
    print("=" * 80)
    if reports["no_route"]:
        print("\n## NO-ROUTE (path not in route map — dead-link candidates)")
        for it in sorted(reports["no_route"], key=lambda x: (x["file"], x["url"])):
            print(f"  {it['file']}: {it['method']} {it['url']}")
    if reports["method_mismatch"]:
        print("\n## METHOD-MISMATCH (path exists; low confidence — check manually)")
        for it in reports["method_mismatch"][:40]:
            print(f"  {it['file']}: {it['method']} {it['url']}  → route {it['route']} allowed {it['allowed']}")
    if reports["dynamic"]:
        print(f"\n## DYNAMIC (template/trailing-slash concat, need runtime): {len(reports['dynamic'])}")
        for it in reports["dynamic"][:40]:
            print(f"  {it['file']}: {it['method'] or '?'} {it['url'][:110]}")

    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(reports, fh, ensure_ascii=False, indent=2)
        print(f"\n[report] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
