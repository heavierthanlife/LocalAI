"""Route preservation guard.

Ensures the set of registered HTTP routes never silently shrinks. The snapshot
was captured before the C1-C3 refactors (admin.py / chat.py / knowledge.py
splits). Any intentional route change must update the snapshot.

Usage:
    pytest tests/test_route_preservation.py -v

Regenerate snapshot (when intentionally changing routes):
    python C:\\Users\\nana-\\AppData\\Local\\Temp\\opencode\\dump_routes.py
"""
import json
import os

import pytest

SNAPSHOT = os.path.join(os.path.dirname(__file__), 'fixtures', 'routes_snapshot.json')


def _current_routes(app):
    routes = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint == 'static' or rule.endpoint.startswith('static'):
            continue
        methods = sorted(m for m in rule.methods if m not in ('HEAD', 'OPTIONS'))
        routes.append({'methods': methods, 'rule': rule.rule})
    routes.sort(key=lambda r: r['rule'])
    return routes


@pytest.mark.parametrize('expected_len', [382])
def test_route_count_stable(app, expected_len):
    """Total non-static route count must match the pre-refactor snapshot."""
    assert len(_current_routes(app)) == expected_len


def test_route_set_unchanged(app):
    """Every route rule+method pair from the snapshot must still be registered."""
    with open(SNAPSHOT, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)
    current = _current_routes(app)
    current_set = {(r['rule'], tuple(r['methods'])) for r in current}
    missing = []
    for r in snapshot:
        key = (r['rule'], tuple(r['methods']))
        if key not in current_set:
            missing.append(r)
    assert not missing, f"{len(missing)} routes missing from snapshot:\n" + \
        "\n".join(f"  {r['methods']} {r['rule']}" for r in missing)


def test_no_duplicate_rules(app):
    """The set of duplicate (rule, methods) registrations must not grow.

    The pre-refactor codebase already registers 5 harmless duplicate pairs
    (e.g. '/' GET, '/feedback' POST). A split must not introduce NEW ones.
    """
    from collections import Counter
    with open(SNAPSHOT, 'r', encoding='utf-8') as f:
        snap_pairs = [(r['rule'], tuple(r['methods'])) for r in json.load(f)]
    known_dupes = {p for p, c in Counter(snap_pairs).items() if c > 1}

    cur_pairs = [(r['rule'], tuple(r['methods'])) for r in _current_routes(app)]
    cur_dupes = {p for p, c in Counter(cur_pairs).items() if c > 1}
    new_dupes = cur_dupes - known_dupes
    assert not new_dupes, f"New duplicate route registrations: {new_dupes}"
