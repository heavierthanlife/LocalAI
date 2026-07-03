"""Validate SKILL.md files against anthropics/skills spec.

Checks for required YAML frontmatter, naming conventions, and structural completeness.
"""

import os, re, yaml
from pathlib import Path

SKILLS_DIR = Path(__file__).parent.parent.parent / ".codebuddy" / "skills"
REQUIRED_FIELDS = {'name', 'description', 'version', 'tags'}
RECOMMENDED_FIELDS = {'author', 'trigger_keywords', 'license'}


def validate_all() -> dict:
    """Validate all SKILL.md files. Returns report dict."""
    results = {'total': 0, 'valid': 0, 'warnings': 0, 'errors': 0, 'details': []}
    if not SKILLS_DIR.exists():
        return results

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / 'SKILL.md'
        if not skill_md.exists():
            continue

        results['total'] += 1
        detail = {'skill': skill_dir.name, 'issues': []}

        try:
            content = skill_md.read_text(encoding='utf-8')
        except Exception:
            detail['issues'].append('Cannot read file')
            results['errors'] += 1
            results['details'].append(detail)
            continue

        # Check for YAML frontmatter
        fm_match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
        if not fm_match:
            detail['issues'].append('Missing YAML frontmatter (--- ... ---)')
            results['warnings'] += 1
        else:
            try:
                fm = yaml.safe_load(fm_match.group(1))
                if fm:
                    missing_required = REQUIRED_FIELDS - set(fm.keys())
                    if missing_required:
                        detail['issues'].append(f'Missing required fields: {missing_required}')
                        results['warnings'] += 1
                    missing_recommended = RECOMMENDED_FIELDS - set(fm.keys())
                    if missing_recommended:
                        detail['issues'].append(f'Missing recommended fields: {missing_recommended}')
                else:
                    detail['issues'].append('Empty YAML frontmatter')
                    results['warnings'] += 1
            except yaml.YAMLError:
                detail['issues'].append('Invalid YAML frontmatter')
                results['errors'] += 1

        # Check content after frontmatter
        body = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL).strip()
        if not body:
            detail['issues'].append('No content after frontmatter')
            results['warnings'] += 1
        elif len(body) < 100:
            detail['issues'].append(f'Content too short ({len(body)} chars)')
            results['warnings'] += 1

        if not detail['issues']:
            results['valid'] += 1
        results['details'].append(detail)

    return results
