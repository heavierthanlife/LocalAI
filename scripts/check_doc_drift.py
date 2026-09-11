#!/usr/bin/env python
"""Documentation drift checker — verify count-based facts in docs match code.

Recomputes a small set of stable, code-derived counts and asserts every place a
tracked doc states that count agrees. Prevents the "docs say 5 providers / 70
tables" class of rot (see AGENTS.md → 文档维护).

Run:      python scripts/check_doc_drift.py
Bypass:   SKIP_DOC_DRIFT=1 git commit ...
"""
import ast
import glob
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files scanned for count claims. CHANGELOG is intentionally excluded (history).
SCAN_FILES = [
    'README.md',
    'AGENTS.md',
    'docs/ARCHITECTURE.md',
    'docs/MANIFEST.md',
    'docs/USER_MANUAL.md',
]


def _read(rel):
    path = os.path.join(PROJECT_ROOT, rel)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def _count_blueprints():
    src = _read('app/routes/__init__.py') or ''
    return src.count('register_blueprint(')


def _count_tables():
    src = _read('app/database.py') or ''
    return len(re.findall(r'CREATE TABLE IF NOT EXISTS', src))


def _count_services():
    files = glob.glob(os.path.join(PROJECT_ROOT, 'app', 'services', '*.py'))
    return len([f for f in files if os.path.basename(f) != '__init__.py'])


def _count_js():
    files = glob.glob(os.path.join(PROJECT_ROOT, 'static', 'js', '*.js'))
    return len(files)


def _count_providers():
    src = _read('app/services/llm_provider.py') or ''
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == 'PROVIDER_CONFIG' and isinstance(node.value, ast.Dict):
                    return len(node.value.keys)
    return -1


def _count_indicators():
    src = _read('app/services/indicator_defs.py') or ''
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == 'INDICATOR_DEFS' and isinstance(node.value, ast.List):
                    return len(node.value.elts)
    return -1


# metric -> (compute, [(file, regex-with-1-group, human_label)])
CHECKS = {
    'blueprints': (_count_blueprints, [
        ('README.md', r'(\d+)\s*Blueprint'),
        ('AGENTS.md', r'(\d+)\s*Blueprints?'),
        ('docs/ARCHITECTURE.md', r'(\d+)\s*Blueprints?'),
        ('docs/ARCHITECTURE.md', r'(\d+)\s*蓝图'),
    ]),
    'tables': (_count_tables, [
        ('README.md', r'(\d+)\s*张表'),
        ('docs/ARCHITECTURE.md', r'(\d+)\s*张表'),
        ('docs/ARCHITECTURE.md', r'(\d+)\s*张初始化'),
    ]),
    'services': (_count_services, [
        ('docs/ARCHITECTURE.md', r'(\d+)\s*Services'),
    ]),
    'providers': (_count_providers, [
        ('README.md', r'(\d+)\s*供应商'),
        ('docs/ARCHITECTURE.md', r'(\d+)\s*供应商'),
    ]),
    'indicators': (_count_indicators, [
        ('docs/ARCHITECTURE.md', r'(\d+)\s*指标'),
        ('docs/USER_MANUAL.md', r'(\d+)\s*指标'),
    ]),
}


def main():
    if os.environ.get('SKIP_DOC_DRIFT'):
        print('[doc_drift] SKIP (SKIP_DOC_DRIFT set)')
        return 0

    failures = []
    checks = 0
    for metric, (compute, claims) in CHECKS.items():
        expected = compute()
        if expected < 0:
            print(f'[doc_drift] SKIP {metric}: could not compute')
            continue
        for rel, pattern in claims:
            content = _read(rel)
            if content is None:
                continue
            for m in re.finditer(pattern, content):
                checks += 1
                got = int(m.group(1))
                if got != expected:
                    line = content[:m.start()].count('\n') + 1
                    failures.append(
                        f'{rel}:{line}  {metric}: doc says {got}, code has {expected}')

    if failures:
        print('[doc_drift] ERROR: documentation count drift:')
        for f in failures:
            print(f'  - {f}')
        print('  Fix the doc number(s) or update this checker.')
        print(f'\n{checks} claim(s), {len(failures)} drift(s).')
        return 1

    print(f'[doc_drift] OK: {checks} claim(s) match code ({len(CHECKS)} metrics).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
