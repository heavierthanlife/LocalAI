"""repair_kit integrity checker — verify checklist covers all blueprints.

Standalone script (no pytest deps). Run:
    python repair_kit/check_integrity.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPAIR_KIT = os.path.dirname(os.path.abspath(__file__))
CHECKLIST_PATH = os.path.join(REPAIR_KIT, 'SYSTEM_CHECKLIST.md')
INIT_PATH = os.path.join(PROJECT_ROOT, 'app', 'routes', '__init__.py')
README_PATH = os.path.join(REPAIR_KIT, 'README.md')


def _get_all_blueprint_names():
    names = []
    with open(INIT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('from app.routes.') and 'import' in line:
                parts = line.split()
                if len(parts) >= 4:
                    names.append(parts[3])
    return names


def _read_file(path):
    if not os.path.exists(path):
        return ''
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def check_blueprint_coverage():
    """Every registered blueprint should have at least one checklist item."""
    blueprints = _get_all_blueprint_names()
    assert blueprints, f"ERROR: No blueprints found in {INIT_PATH}"

    content = _read_file(CHECKLIST_PATH)
    assert content, f"ERROR: SYSTEM_CHECKLIST.md not found at {CHECKLIST_PATH}"
    content_lower = content.lower()

    uncovered = []
    for bp in blueprints:
        bp_name = bp.replace('_bp', '').replace('_', ' ')
        if bp_name not in content_lower:
            uncovered.append(bp)

    if uncovered:
        print(f"WARNING: Blueprints without checklist coverage: {', '.join(uncovered)}")
        print("  Add items covering these blueprints to SYSTEM_CHECKLIST.md")
        return False
    return True


def check_readme_exists():
    content = _read_file(README_PATH)
    assert content, "ERROR: repair_kit/README.md is missing"
    assert 'SYSTEM_CHECKLIST.md' in content, "README.md must reference the checklist"
    assert 'AGENTS.md' in content, "README.md must reference AGENTS.md"
    return True


def check_all_sections():
    required = [
        '进程与服务', '核心路由', '数据库', 'AI', '文件系统',
        'Redis', '速率限制', '安全', '管理功能', '集成功能',
    ]
    content = _read_file(CHECKLIST_PATH)
    missing = [s for s in required if s not in content]
    if missing:
        print(f"WARNING: Checklist missing sections: {', '.join(missing)}")
        return False
    return True


def main():
    errors = 0
    for name, fn in [
        ('Blueprint coverage', check_blueprint_coverage),
        ('README.md integrity', check_readme_exists),
        ('All checklist sections', check_all_sections),
    ]:
        try:
            ok = fn()
            status = 'PASS' if ok else 'FAIL'
            if not ok:
                errors += 1
        except AssertionError as e:
            status = 'FAIL'
            print(f'  {e}')
            errors += 1
        print(f'  [{status}] {name}')

    print(f'\n{errors} failure(s)' if errors else '\nAll checks passed.')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
