"""Fix Registry Verifier — check all recorded fixes still hold.

Standalone script. Run:
    python scripts/verify_fixes.py
Bypass in pre-commit:
    SKIP_FIX_CHECK=1 git commit ...
"""
import os
import re
import sys
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(PROJECT_ROOT, 'data', 'fix_registry.yaml')


def _read_file(rel_path):
    path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _check_grep(file_rel, pattern):
    content = _read_file(file_rel)
    if content is None:
        return False, f"file not found: {file_rel}"
    if re.search(pattern, content):
        return True, None
    return False, f"pattern '{pattern}' NOT found in {file_rel}"


def _check_grep_not(file_rel, pattern):
    content = _read_file(file_rel)
    if content is None:
        return True, None  # file missing is non-blocking for grep_not
    if re.search(pattern, content):
        return False, f"forbidden pattern '{pattern}' FOUND in {file_rel}"
    return True, None


def _check_function_order(file_rel, before_name, after_name):
    content = _read_file(file_rel)
    if content is None:
        return False, f"file not found: {file_rel}"
    before_idx = content.find(before_name)
    after_idx = content.find(after_name)
    if before_idx == -1:
        return False, f"function '{before_name}' not found in {file_rel}"
    if after_idx == -1:
        return False, f"function '{after_name}' not found in {file_rel}"
    if before_idx < after_idx:
        return True, None
    return False, f"'{before_name}' (line ~{content[:before_idx].count(chr(10))+1}) appears AFTER '{after_name}' (line ~{content[:after_idx].count(chr(10))+1}) in {file_rel}"


CHECK_RUNNERS = {
    'grep': lambda c: _check_grep(c['file'], c['pattern']),
    'grep_not': lambda c: _check_grep_not(c['file'], c['pattern']),
    'function_order': lambda c: _check_function_order(c['file'], c['before'], c['after']),
}


def main():
    if not os.path.exists(REGISTRY_PATH):
        print(f"ERROR: Fix registry not found at {REGISTRY_PATH}")
        return 1

    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry = yaml.safe_load(f)

    fixes = registry.get('fixes', [])
    if not fixes:
        print("No fixes registered.")
        return 0

    failures = 0
    total = 0

    for fix in fixes:
        fix_id = fix.get('id', 'unknown')
        title = fix.get('title', '')
        checks = fix.get('checks', [])

        for check in checks:
            total += 1
            check_type = check.get('type', '')
            if check_type not in CHECK_RUNNERS:
                print(f"  [SKIP] {fix_id}: unknown check type '{check_type}'")
                continue

            try:
                ok, error = CHECK_RUNNERS[check_type](check)
                if ok:
                    print(f"  [PASS] {fix_id}: {title}")
                else:
                    print(f"  [FAIL] {fix_id}: {title}")
                    print(f"         -> {error}")
                    failures += 1
            except Exception as e:
                print(f"  [ERR]  {fix_id}: {title}")
                print(f"         -> check error: {e}")
                failures += 1

    print(f"\n{total} check(s), {failures} failure(s).")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
