"""Shared project context gathering — used by both AI assistant and chat system."""
import os
import hashlib
import logging
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)


def deduplicate_names(
    items: List[Dict],
    name_key: str = 'original_name',
    id_key: str = 'id',
    hash_key: str = None,
) -> List[str]:
    """Return display names with #short-hash suffix for duplicate entries.

    When the same original_name appears multiple times (e.g. two files both
    named "技术方案.docx"), append a 4-char hex hash to disambiguate in LLM
    prompts so the model never confuses entities or hallucinates UUIDs.

    Args:
        items: list of dicts with at least name_key and id_key fields
        name_key: dict key for the human-readable name (default 'original_name')
        id_key: dict key for a unique identifier used when hash_key is absent
        hash_key: optional dict key for a pre-computed hash (e.g. 'file_hash')

    Returns:
        List of display-name strings, same length as items.
        Non-duplicate names are returned as-is.
    """
    # Count occurrences of each name
    name_counts: Dict[str, int] = {}
    for item in items:
        name = (item.get(name_key) or '').strip()
        name_counts[name] = name_counts.get(name, 0) + 1

    # Build disambiguated names
    result: List[str] = []
    for item in items:
        name = (item.get(name_key) or '').strip()
        if name_counts.get(name, 1) > 1:
            # Use pre-computed hash if available, else hash the id
            if hash_key and item.get(hash_key):
                short_hash = hashlib.md5(str(item[hash_key]).encode()).hexdigest()[:4]
            elif item.get(id_key) is not None:
                short_hash = hashlib.md5(str(item[id_key]).encode()).hexdigest()[:4]
            else:
                short_hash = '0000'
            result.append(f"{name} #{short_hash}")
        else:
            result.append(name)

    return result


def gather_project_context(
    db_conn, project_id: int, query: str, user_id: str,
    username: str = '',
) -> dict:
    """Gather all relevant context for a project query.
    
    Returns dict with keys: project_context, rag_context, skill_hints,
    memory_context, workflow_section, proj_industry, warnings.
    Used by project_ai_assist and can be used by chat.py for project-scoped chat.
    """
    from psycopg2.extras import RealDictCursor

    result = {
        'project_context': '',
        'rag_context': '',
        'skill_hints': [],
        'memory_context': '',
        'workflow_section': '',
        'proj_industry': 'general',
        'warnings': [],
    }

    with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
        # 1. Project file texts
        cur.execute("""
            SELECT id, original_name, content, skill_summary, file_hash
            FROM project_files
            WHERE project_id = %s AND content IS NOT NULL AND content != ''
            ORDER BY uploaded_at DESC LIMIT 15
        """, (project_id,))
        project_files = cur.fetchall()
        # Disambiguate duplicate file names for LLM (e.g. "技术方案.docx #a3f2")
        display_names = deduplicate_names(project_files, name_key='original_name', id_key='id', hash_key='file_hash')
        project_texts = []
        for i, f in enumerate(project_files):
            text = (f.get('skill_summary') or '') + '\n' + (f.get('content') or '')
            if text.strip():
                project_texts.append(f"--- 项目文件: {display_names[i]} ---\n{text[:2000]}")
        result['project_context'] = '\n'.join(project_texts[:10]) if project_texts else '(本项目暂无文件内容)'

        # 2. RAG context
        try:
            from app.services.rag_engine import build_rag_context
            result['rag_context'] = build_rag_context(
                query, ['company_kb', 'knowledge_lab'], top_k=10, max_chars=6000
            )
        except Exception:
            pass

        # 3. Skills
        skill_hints = []
        for table in ('knowledge_lab_files', 'company_knowledge_base', 'project_files'):
            try:
                cur.execute(f"""
                    SELECT skill_summary FROM {table}
                    WHERE skill_summary IS NOT NULL AND skill_summary != ''
                    ORDER BY skill_generated_at DESC NULLS LAST LIMIT 10
                """)
                for row in cur.fetchall():
                    s = row.get('skill_summary', '')
                    if s and len(s) > 10:
                        skill_hints.append(s[:300])
            except Exception:
                pass
        result['skill_hints'] = skill_hints

        # 4. AI memory (identity-tagged)
        cur.execute("""
            SELECT pam.user_id, u.username, pam.role, pam.content, pam.created_at
            FROM project_ai_memory pam
            LEFT JOIN users u ON pam.user_id = u.user_id
            WHERE pam.project_id = %s
            ORDER BY pam.created_at DESC LIMIT 60
        """, (project_id,))
        memory_rows = cur.fetchall()
        if memory_rows and query:
            query_keywords = set(query.lower().split())
            scored = []
            for r in memory_rows:
                content_words = set((r['content'] or '').lower().split())
                overlap = len(query_keywords & content_words) if query_keywords and content_words else 0
                score = overlap / max(len(query_keywords), 1)
                scored.append((score, r))
            scored.sort(key=lambda x: -x[0])
            selected = []
            per_user = {}
            for score, r in scored:
                if score <= 0 and len(selected) >= 5:
                    break
                uid = r['user_id']
                if per_user.get(uid, 0) >= 3:
                    continue
                selected.append(r)
                per_user[uid] = per_user.get(uid, 0) + 1
                if len(selected) >= 15:
                    break
            selected.sort(key=lambda r: r.get('created_at') or '', reverse=False)
            memory_lines = []
            for r in selected:
                who = r.get('username') or r['user_id'] or '?'
                label = f"@{who}" if r['role'] == 'user' else f"AI→{who}"
                memory_lines.append(f"{label}: {r['content'][:400]}")
            result['memory_context'] = '\n'.join(memory_lines)

        # 5. Industry workflow
        cur.execute("SELECT industry FROM projects WHERE id = %s", (project_id,))
        row = cur.fetchone()
        if row:
            result['proj_industry'] = row.get('industry') or 'general'
        wf_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'workflows',
                              f"{result['proj_industry']}.md")
        try:
            if os.path.exists(wf_path):
                with open(wf_path, 'r', encoding='utf-8') as wf:
                    result['workflow_section'] = wf.read()
        except Exception:
            pass

        # 6. Concurrent activity warning
        try:
            cur.execute("""
                SELECT COUNT(*) as cnt FROM project_ai_memory
                WHERE project_id = %s AND role = 'user' AND created_at > NOW() - INTERVAL '5 minutes'
            """, (project_id,))
            recent = cur.fetchone()
            if recent and recent['cnt'] > 2:
                result['warnings'].append(f"💡 近5分钟内有 {recent['cnt']} 次AI助手使用，注意协作避免重复工作")
        except Exception:
            pass

    return result
