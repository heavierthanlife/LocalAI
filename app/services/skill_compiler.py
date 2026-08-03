"""Skill Compiler — periodically clusters skills by topic within each category
    and generates composite skills that merge related documents.
    Composites are stored as knowledge_lab_files entries for the admin user."""

import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_COMPOSITE_USER_ID = None


def _get_admin_user_id(conn):
    """Find the admin user to own composite skills."""
    global _COMPOSITE_USER_ID
    if _COMPOSITE_USER_ID:
        return _COMPOSITE_USER_ID
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM users WHERE role = 'admin' ORDER BY created_at ASC LIMIT 1")
            row = cur.fetchone()
            if row:
                _COMPOSITE_USER_ID = row[0]
                return _COMPOSITE_USER_ID
    except Exception:
        pass
    return None


def _fetch_skills_by_category(conn):
    """Fetch all skill summaries grouped by category from all 3 tables."""
    by_cat = {}
    for table in ('knowledge_lab_files', 'company_knowledge_base', 'project_files'):
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT id, original_name, skill_summary, category, '{table}' as src_table
                    FROM {table}
                    WHERE skill_summary IS NOT NULL AND skill_summary != ''
                      AND category IS NOT NULL AND category != ''
                """)
                for row in cur.fetchall():
                    cat = row[3] or '通用'
                    by_cat.setdefault(cat, []).append({
                        'id': row[0], 'name': row[1], 'summary': row[2],
                        'table': row[4],
                    })
        except Exception:
            continue
    return by_cat


def _cluster_skills(skills, eps=0.3, min_samples=2):
    """Cluster skills by embedding similarity. Returns list of clusters."""
    if len(skills) < min_samples:
        return []
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception:
        return []
    summaries = [s['summary'][:2000] for s in skills]
    embs = model.encode(summaries, show_progress_bar=False)

    try:
        from sklearn.cluster import DBSCAN
        clusters = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embs)
    except Exception:
        return []

    result = []
    for cid in set(clusters.labels_):
        if cid == -1:
            continue
        members = [skills[i] for i, c in enumerate(clusters.labels_) if c == cid]
        if len(members) >= 2:
            result.append(members)
    return result


def _merge_summaries(members, category):
    """Merge N skill summaries into one composite markdown."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = [f"# [合成] {category} — {' + '.join(m['name'][:30] for m in members[:5])}"]
    lines.append(f"> Auto-generated composite skill | Category: {category} | {len(members)} sources | {now}")
    source_list = ', '.join(f"{m['name']} ({m['table']})" for m in members)
    lines.append(f"> Source files: {source_list}")
    lines.append("")

    sections = {}
    section_order = ['📌 核心概念', '📖 定义与术语', '🏗️ 框架与模型',
                     '⚖️ 原则与规则', '🔧 方法与技巧', '✅ 可执行步骤', '⚠️ 常见陷阱']
    current_section = None

    for m in members:
        for line in m['summary'].split('\n'):
            if line.startswith('## '):
                current_section = line[3:].strip()
                if current_section not in sections:
                    sections[current_section] = set()
            elif line.startswith('- ') and current_section in sections:
                sections[current_section].add(line)

    for header in section_order:
        key = next((k for k in sections if header in k), None)
        if key and sections[key]:
            lines.append(f"## {header}")
            for item in sorted(sections[key])[:15]:
                lines.append(item)
            lines.append("")

    lines.append("---")
    lines.append(f"*Auto-generated composite skill | Category: {category} | {now}*")
    return '\n'.join(lines)


def compile_skills(conn):
    """Main entry point: compile composite skills for all categories."""
    admin_id = _get_admin_user_id(conn)
    if not admin_id:
        logger.warning("Skill compiler: no admin user found, skipping")
        return 0

    by_cat = _fetch_skills_by_category(conn)
    total_composites = 0

    for category, skills in by_cat.items():
        if len(skills) < 2:
            continue
        clusters = _cluster_skills(skills)
        for cluster in clusters:
            composite_md = _merge_summaries(cluster, category)
            if not composite_md or len(composite_md) < 200:
                continue

            source_names = [m['name'] for m in cluster]
            composite_name = f"[合成]{category}_{source_names[0][:20]}等{len(cluster)}份来源.md"
            composite_hash = hashlib.sha256(composite_md.encode()).hexdigest()

            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM knowledge_lab_files WHERE file_hash = %s AND user_id = %s",
                        (composite_hash, admin_id)
                    )
                    existing = cur.fetchone()
                    if existing:
                        cur.execute(
                            "UPDATE knowledge_lab_files SET skill_summary = %s, updated_at = NOW() WHERE id = %s",
                            (composite_md, existing[0])
                        )
                    else:
                        cur.execute("""
                            INSERT INTO knowledge_lab_files
                                (user_id, filename, original_name, file_size, content, file_hash, stored_path,
                                 category, skill_summary, skill_generated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        """, (
                            admin_id, composite_name, composite_name, len(composite_md),
                            composite_md, composite_hash, '', category, composite_md,
                        ))
                    conn.commit()
                    total_composites += 1
                    logger.info(f"Composite skill created: {composite_name} ({category}, {len(cluster)} sources)")
            except Exception as e:
                logger.warning(f"Failed to store composite for {category}: {e}")

    logger.info(f"Skill compiler: {total_composites} composites created/updated")
    return total_composites
