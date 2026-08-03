"""Publish audit results as browsable wiki pages."""
import logging
import os
import re
from datetime import datetime

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

WIKI_DIR = os.path.join(DATA_DIR, 'wiki')
AUDIT_WIKI_DIR = "audit"


def publish_audit_to_wiki(run_id: int) -> str | None:
    try:
        from app.services.audit_engine import get_run_results
        results = get_run_results(run_id)
        if not results:
            logger.warning(f"Audit run {run_id} not found, skipping wiki publish")
            return None

        return _do_publish(run_id, results)

    except Exception as e:
        logger.warning(f"Wiki publish failed for audit run {run_id}: {e}", exc_info=True)
        return None


def _do_publish(run_id: int, results: dict) -> str | None:
    from app.database import get_db_connection
    from app.services import wiki_engine

    project_id = results.get('project_id')
    project_name = None
    if project_id:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM projects WHERE id = %s", (project_id,))
                row = cur.fetchone()
                if row:
                    project_name = row[0]

    started_at = results.get('started_at', '')
    if started_at:
        try:
            dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            date_str = dt.strftime('%Y-%m-%d')
            month_str = dt.strftime('%m')
            year_str = dt.strftime('%Y')
            time_str = dt.strftime('%H%M')
        except Exception:
            date_str = datetime.now().strftime('%Y-%m-%d')
            month_str = datetime.now().strftime('%m')
            year_str = datetime.now().strftime('%Y')
            time_str = datetime.now().strftime('%H%M')
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
        year_str = datetime.now().strftime('%Y')
        month_str = datetime.now().strftime('%m')
        time_str = datetime.now().strftime('%H%M')

    project_slug = _slugify(project_name or f"project_{project_id}")
    func_names = [fr.get('function_name', '') for fr in results.get('file_results', [])]
    unique_funcs = list(dict.fromkeys(func_names))
    audit_label = unique_funcs[0] if len(unique_funcs) == 1 else '综合审核'
    filename = f"{date_str}_{time_str}_{project_slug}_{audit_label}.md"
    wiki_path = f"{AUDIT_WIKI_DIR}/{year_str}/{month_str}/{filename}"

    score = results.get('overall_score', 0)
    status = results.get('overall_status', 'unknown')
    status_emoji = '\u2705' if status == 'PASS' else '\u26a0\ufe0f'

    file_results = results.get('file_results', [])
    bidder_scores = {}

    for fr in file_results:
        bidder = fr.get('bidder_label', 'unknown')
        if bidder not in bidder_scores:
            bidder_scores[bidder] = {'total': 0.0, 'count': 0, 'issues': []}
        bidder_scores[bidder]['total'] += fr.get('score', 0)
        bidder_scores[bidder]['count'] += 1
        findings = fr.get('findings')
        if isinstance(findings, dict) and findings.get('error_message'):
            bidder_scores[bidder]['issues'].append({
                'file': fr.get('filename', ''),
                'func': fr.get('function_name', ''),
                'score': fr.get('score', 0),
                'error': findings.get('error_message', ''),
            })
        elif isinstance(findings, dict) and findings.get('results'):
            for r in findings.get('results', []):
                if isinstance(r, dict) and r.get('verdict') == 'fail':
                    bidder_scores[bidder]['issues'].append({
                        'file': fr.get('filename', ''),
                        'func': fr.get('function_name', ''),
                        'issue': r.get('rule', r.get('name', 'unknown')),
                        'detail': r.get('detail', r.get('explanation', '')),
                    })
        elif isinstance(findings, dict) and findings.get('findings'):
            findings_list = findings.get('findings', [])
            if findings_list:
                bidder_scores[bidder]['issues'].append({
                    'file': fr.get('filename', ''),
                    'func': fr.get('function_name', ''),
                    'count': len(findings_list),
                })

    frontmatter = {
        'title': f"审计报告 — {project_name or f'项目-{project_id}'}",
        'date': date_str,
        'type': 'audit',
        'audit_run_id': run_id,
        'project_id': project_id,
        'project_name': project_name or '',
        'overall_score': score,
        'overall_status': status,
        'functions': unique_funcs,
    }

    lines = [
        f"# 审计报告 — {project_name or f'项目-{project_id}'}",
        f"",
        f"**审核时间:** {date_str} | **综合评分:** {score:.0f}/100 {status_emoji} {status}",
        f"**执行功能:** {', '.join(unique_funcs)}",
        f"**投标人数:** {results.get('bidder_count', 0)} | **文件数:** {results.get('file_count', 0)}",
        f"",
        f"## 概览",
        f"",
    ]

    lines.append("| 投标人 | 文件数 | 平均分 | 问题数 |")
    lines.append("|--------|--------|--------|--------|")
    for bidder, data in sorted(bidder_scores.items()):
        avg = data['total'] / max(data['count'], 1)
        issues = len(data['issues'])
        lines.append(f"| {bidder} | {data['count']} | {avg:.0f} | {issues} |")

    lines.append("")
    lines.append("## 主要问题")
    lines.append("")

    all_issues = []
    for bidder, data in bidder_scores.items():
        for issue in data['issues']:
            all_issues.append({'bidder': bidder, **issue})

    for i, issue in enumerate(all_issues[:5]):
        lines.append(f"### {i + 1}. {issue.get('bidder', '')} — {issue.get('file', '')}")
        if issue.get('error'):
            lines.append(f"- **问题:** {issue['error']}")
        elif issue.get('issue'):
            lines.append(f"- **违规项:** {issue['issue']}")
            if issue.get('detail'):
                lines.append(f"- **详情:** {issue['detail']}")
        elif issue.get('count'):
            lines.append(f"- **发现:** {issue['count']} 处问题 ({issue.get('func', '')})")
        lines.append(f"- **功能:** {issue.get('func', '')} | **得分:** {issue.get('score', '')}")
        lines.append("")

    lines.append("## 相关链接")
    lines.append("")
    lines.append(f"- [\U0001f4e5 下载DOCX报告](/audit/download/{run_id}/docx)")
    lines.append(f"- [\U0001f4ca 下载XLSX报告](/audit/download/{run_id}/xlsx)")
    lines.append(f"- [\U0001f50d 审计详情](/audit/result/{run_id})")
    lines.append("")
    lines.append("---")
    lines.append("*由AI审核引擎自动生成*")

    content = '\n'.join(lines)

    wiki_engine.write_wiki_page(wiki_path, frontmatter, content)

    _update_monthly_index(wiki_engine, year_str, month_str, filename, project_name or f'项目-{project_id}', audit_label, score, status, date_str)
    _update_master_index(wiki_engine, filename, project_name or f'项目-{project_id}', audit_label, score, status, date_str, year_str, month_str)

    wiki_engine.record_origin_link(
        wiki_path=wiki_path,
        source_type='audit_run',
        source_file_id=run_id,
        source_name=f"audit_{run_id}",
    )

    logger.info(f"Audit run {run_id} published to wiki: {wiki_path}")
    return wiki_path


def _update_monthly_index(wiki_engine, year_str, month_str, filename, project_name, audit_label, score, status, date_str):
    index_path = f"{AUDIT_WIKI_DIR}/{year_str}/{month_str}/index.md"
    today_str = date_str.split('-')[2]

    try:
        from filelock import FileLock
        lock_path = os.path.join(WIKI_DIR, index_path + ".lock")
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    except ImportError:
        logger.warning("filelock not installed, monthly index update may have race conditions")
        lock_path = None

    if lock_path:
        with FileLock(lock_path, timeout=5):
            _do_update_monthly_index(wiki_engine, index_path, filename, project_name, audit_label, score, status, today_str)
    else:
        _do_update_monthly_index(wiki_engine, index_path, filename, project_name, audit_label, score, status, today_str)


def _do_update_monthly_index(wiki_engine, index_path, filename, project_name, audit_label, score, status, today_str):
    status_icon = '\u2705' if status == 'PASS' else '\u274c'
    new_line = f"| {today_str} | [{project_name}]({filename}) | {audit_label} | {score:.0f} | {status_icon} |"

    fm, existing_content = read_wiki_or_empty(wiki_engine, index_path)

    content_lines = existing_content.split('\n')
    insert_idx = len(content_lines)
    for i, line in enumerate(content_lines):
        if line.strip().startswith('|') and i > 0:
            insert_idx = i + 1
        if line.strip().startswith('---'):
            break

    header = f"# 审计存档 — {filename[:7]}\n\n| 日期 | 项目 | 类型 | 评分 | 状态 |\n|------|------|------|------|------|"
    if not existing_content.strip():
        content = header + "\n" + new_line
    else:
        content_lines.insert(insert_idx, new_line)
        content = '\n'.join(content_lines)

    wiki_engine.write_wiki_page(index_path, fm, content)


def _update_master_index(wiki_engine, filename, project_name, audit_label, score, status, date_str, year_str, month_str):
    index_path = f"{AUDIT_WIKI_DIR}/index.md"

    try:
        from filelock import FileLock
        lock_path = os.path.join(WIKI_DIR, index_path + ".lock")
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    except ImportError:
        lock_path = None

    if lock_path:
        with FileLock(lock_path, timeout=5):
            _do_update_master_index(wiki_engine, index_path, filename, project_name, audit_label, score, status, date_str, year_str, month_str)
    else:
        _do_update_master_index(wiki_engine, index_path, filename, project_name, audit_label, score, status, date_str, year_str, month_str)


def _do_update_master_index(wiki_engine, index_path, filename, project_name, audit_label, score, status, date_str, year_str, month_str):
    status_icon = '\u2705' if status == 'PASS' else '\u274c'
    sub_path = f"{year_str}/{month_str}/{filename}"

    fm, existing_content = read_wiki_or_empty(wiki_engine, index_path)

    entries = []
    link_re = re.compile(r'\| \[([^\]]+)\]\(([^)]+)\) \|')
    for line in existing_content.split('\n'):
        m = link_re.search(line)
        if m:
            entries.append(line)

    new_entry = f"| {date_str} | [{project_name}]({sub_path}) | {audit_label} | {score:.0f} | {status_icon} |"
    entries.insert(0, new_entry)

    header = "# 审计存档\n\n| 日期 | 项目 | 审核类型 | 评分 | 状态 |\n|------|------|----------|------|------|"
    content = header + "\n" + '\n'.join(entries[:50])

    wiki_engine.write_wiki_page(index_path, fm, content)


def read_wiki_or_empty(wiki_engine, path):
    try:
        return wiki_engine.read_wiki_page(path)
    except Exception:
        return ({}, '')


def _slugify(name: str) -> str:
    s = re.sub(r'[^\w\u4e00-\u9fff-]', '_', name)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')[:40]
