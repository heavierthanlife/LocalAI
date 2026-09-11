"""Regenerate tests/fixtures/routes_snapshot.json.

Run after an INTENTIONAL route change (add/remove), then update the
`expected_len` in tests/test_route_preservation.py to match the printed count.

Usage:
    python scripts/dump_routes.py
"""
import io
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Match tests/conftest.py env so create_app() boots without DB/network.
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest")
os.environ.setdefault("WTF_CSRF_ENABLED", "false")
os.environ.setdefault("ENABLE_SCHEDULER", "false")
os.environ.setdefault("RATELIMIT_STORAGE_URL", "memory://")
os.environ.setdefault("LOG_LEVEL", "CRITICAL")

SNAPSHOT = os.path.join(PROJECT_ROOT, 'tests', 'fixtures', 'routes_snapshot.json')


def _current_routes(app):
    routes = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint == 'static' or rule.endpoint.startswith('static'):
            continue
        methods = sorted(m for m in rule.methods if m not in ('HEAD', 'OPTIONS'))
        routes.append({'methods': methods, 'rule': rule.rule})
    routes.sort(key=lambda r: r['rule'])
    return routes


def main():
    from app import create_app
    _old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    app = create_app()
    sys.stdout = _old_stdout

    routes = _current_routes(app)
    with open(SNAPSHOT, 'w', encoding='utf-8') as f:
        json.dump(routes, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f"wrote {len(routes)} routes -> {SNAPSHOT}")
    print(f"update test_route_preservation.py expected_len to {len(routes)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
