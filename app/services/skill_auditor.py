"""Skill Auditor — AI-powered skill curation engine with incremental cache.
   Runs periodic analysis: find duplicates, score usage, suggest actions.
   Remembers previous results via fingerprint cache; only re-analyzes changed skills.
   Humans (skill_auditor role) approve/reject AI suggestions."""

import hashlib, json, logging, os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.75  # cosine similarity above this = "duplicate"
UNUSED_DAYS_THRESHOLD = 30   # flag skills unused for 30+ days
MIN_USAGE_FOR_GOOD = 3       # used 3+ times = "useful"

# ── Cache file ──
from app.config import DATA_DIR
_CACHE_PATH = os.path.join(DATA_DIR, 'skill_audit_cache.json')


def _compute_skill_hash(skill_summary: str) -> str:
    """Compute a fingerprint hash for a skill_summary string.
    Uses first 2000 chars (same as embedding input) for consistency."""
    if not skill_summary:
        return ''
    return hashlib.sha256(skill_summary[:2000].encode('utf-8', errors='replace')).hexdigest()[:16]


def _load_cache() -> dict:
    """Load the cached audit state from disk."""
    if not os.path.exists(_CACHE_PATH):
        return {'fingerprints': {}, 'results': None, 'last_full_analysis': None}
    try:
        with open(_CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load skill audit cache: {e}")
        return {'fingerprints': {}, 'results': None, 'last_full_analysis': None}


def _save_cache(cache: dict):
    """Persist the audit cache to disk."""
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    try:
        with open(_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2, default=str)
    except OSError as e:
        logger.warning(f"Failed to save skill audit cache: {e}")


def _get_similarity_model():
    """Lazy-load sentence-transformers for similarity checks.
    
    Set HF_ENDPOINT=https://hf-mirror.com in .env to use a China-accessible mirror.
    """
    import os as _os
    if not _os.getenv('HF_ENDPOINT'):
        _os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
        _os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception as e:
        logger.warning(f"Similarity model not available (skill duplicate detection disabled): {e}")
        return None


def _fetch_all_skill_fingerprints(db_conn) -> dict:
    """Fetch all skills with their id, source, and current hash from DB.
    
    Returns dict: { 'source:skill_id': {'id': int, 'name': str, 'hash': str, ...} }
    Also updates the skill_summary_hash column if it's empty for any row.
    """
    from psycopg2.extras import RealDictCursor

    skills = {}

    with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
        # ── Personal KB ──
        cur.execute("""
            SELECT k.id, k.original_name, k.skill_summary, k.skill_summary_hash,
                   k.user_id, k.uploaded_at, u.username,
                   COALESCE(fu.usage_count, 0) as usage_count,
                   fu.last_used_at
            FROM knowledge_lab_files k
            JOIN users u ON k.user_id = u.user_id
            LEFT JOIN (
                SELECT file_hash, COUNT(*) as usage_count, MAX(created_at) as last_used_at
                FROM file_analysis WHERE deleted_at IS NULL
                GROUP BY file_hash
            ) fu ON k.file_hash = fu.file_hash
            WHERE k.skill_summary IS NOT NULL
        """)
        for row in cur.fetchall():
            h = row.get('skill_summary_hash') or ''
            if not h and row.get('skill_summary'):
                h = _compute_skill_hash(row['skill_summary'])
                # Backfill the hash column
                cur.execute("UPDATE knowledge_lab_files SET skill_summary_hash = %s WHERE id = %s", (h, row['id']))
            key = f"personal:{row['id']}"
            skills[key] = {
                'id': row['id'], 'name': row['original_name'], 'source': 'personal',
                'summary': row['skill_summary'], 'hash': h,
                'username': row['username'], 'user_id': row['user_id'],
                'usage_count': row['usage_count'], 'last_used_at': row.get('last_used_at'),
                'uploaded_at': row['uploaded_at'],
            }

        # ── Company KB ──
        cur.execute("""
            SELECT id, original_name, skill_summary, skill_summary_hash,
                   uploaded_by as user_id, uploaded_at,
                   (SELECT username FROM users WHERE user_id = uploaded_by) as username,
                   0 as usage_count, NULL as last_used_at
            FROM company_knowledge_base
            WHERE skill_summary IS NOT NULL
        """)
        for row in cur.fetchall():
            h = row.get('skill_summary_hash') or ''
            if not h and row.get('skill_summary'):
                h = _compute_skill_hash(row['skill_summary'])
                cur.execute("UPDATE company_knowledge_base SET skill_summary_hash = %s WHERE id = %s", (h, row['id']))
            key = f"company:{row['id']}"
            skills[key] = {
                'id': row['id'], 'name': row['original_name'], 'source': 'company',
                'summary': row['skill_summary'], 'hash': h,
                'username': row['username'], 'user_id': row['user_id'],
                'usage_count': 0, 'last_used_at': None,
                'uploaded_at': row['uploaded_at'],
            }

        # ── Project files ──
        cur.execute("""
            SELECT id, original_name, skill_summary, skill_summary_hash,
                   uploaded_by as user_id, uploaded_at,
                   (SELECT username FROM users WHERE user_id = uploaded_by) as username,
                   COALESCE(pfu.usage_count, 0) as usage_count, NULL as last_used_at
            FROM project_files pf
            LEFT JOIN (
                SELECT file_id, COUNT(*) as usage_count FROM project_file_usage GROUP BY file_id
            ) pfu ON pf.id = pfu.file_id
            WHERE pf.skill_summary IS NOT NULL
        """)
        for row in cur.fetchall():
            h = row.get('skill_summary_hash') or ''
            if not h and row.get('skill_summary'):
                h = _compute_skill_hash(row['skill_summary'])
                cur.execute("UPDATE project_files SET skill_summary_hash = %s WHERE id = %s", (h, row['id']))
            key = f"project:{row['id']}"
            skills[key] = {
                'id': row['id'], 'name': row['original_name'], 'source': 'project',
                'summary': row['skill_summary'], 'hash': h,
                'username': row['username'], 'user_id': row['user_id'],
                'usage_count': row['usage_count'], 'last_used_at': None,
                'uploaded_at': row['uploaded_at'],
            }

        db_conn.commit()  # commit backfill updates

    # ── Deduplicate names across sources ──
    # Same original_name from different sources (e.g. personal + company)
    # gets a #short-hash suffix so audit reports are unambiguous.
    name_counts: dict = {}
    for s in skills.values():
        n = (s.get('name') or '').strip()
        name_counts[n] = name_counts.get(n, 0) + 1
    for s in skills.values():
        n = (s.get('name') or '').strip()
        if name_counts.get(n, 1) > 1:
            h = hashlib.md5(str(s['id']).encode()).hexdigest()[:4]
            s['name'] = f"{n} #{h}"

    return skills


def _compute_usage_stats(skills: dict) -> dict:
    """Compute unused skills and promotion candidates from skills dict."""
    now = datetime.now(timezone.utc)
    unused = []
    promote_usage = defaultdict(set)

    for key, s in skills.items():
        if s['usage_count'] == 0:
            days = (now - s['uploaded_at']).days if s['uploaded_at'] else 999
            if days > UNUSED_DAYS_THRESHOLD:
                unused.append({
                    'skill_id': s['id'],
                    'name': s['name'],
                    'owner': s['username'],
                    'days_since_upload': days,
                    'source': s['source']
                })
        if s['usage_count'] >= MIN_USAGE_FOR_GOOD and s['source'] == 'personal':
            promote_usage[s['name']].add(s['username'])

    promote = [
        {'name': name, 'user_count': len(users), 'suggested_by': list(users)[0]}
        for name, users in promote_usage.items()
        if len(users) >= 2
    ]
    promote.sort(key=lambda p: p['user_count'], reverse=True)

    unused.sort(key=lambda u: u['days_since_upload'], reverse=True)

    return {
        'unused': unused[:20],
        'promote_candidates': promote[:10],
    }


def _compute_similarity_duplicates(skills_list: list) -> list:
    """Compute cosine similarity pairs among skills. Returns duplicate pairs."""
    model = _get_similarity_model()
    if not model or len(skills_list) < 2:
        return []

    summaries = [s['summary'][:2000] for s in skills_list]
    embeddings = model.encode(summaries, show_progress_bar=False)

    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)

    duplicates = []
    for i in range(len(skills_list)):
        for j in range(i + 1, len(skills_list)):
            if sim_matrix[i][j] > SIMILARITY_THRESHOLD:
                duplicates.append({
                    'skill_a': {
                        'name': skills_list[i]['name'],
                        'owner': skills_list[i]['username'],
                        'id': skills_list[i].get('id'),
                        'source': skills_list[i].get('source', 'knowledge_lab'),
                        'summary': skills_list[i].get('skill_summary', ''),
                    },
                    'skill_b': {
                        'name': skills_list[j]['name'],
                        'owner': skills_list[j]['username'],
                        'id': skills_list[j].get('id'),
                        'source': skills_list[j].get('source', 'knowledge_lab'),
                        'summary': skills_list[j].get('skill_summary', ''),
                    },
                    'similarity': round(float(sim_matrix[i][j]) * 100, 1),
                })
    duplicates.sort(key=lambda d: d['similarity'], reverse=True)
    return duplicates[:20]


def analyze_all_skills(db_conn, force: bool = False):
    """Run full or incremental skill audit.
    
    With incremental mode (default): only re-analyzes skills whose fingerprint
    has changed since the last run.  Unchanged results are loaded from cache.
    
    Set force=True to bypass the cache and re-analyze everything.
    
    Returns a dict with:
        duplicates: list of {skill_a, skill_b, similarity, merge_candidate}
        unused: list of {skill_id, name, days_since_last_use}
        promote_candidates: list of {skill_id, name, user_count, owner}
        summary: {total, with_skill, unused, duplicate_pairs}
        cached: bool - True if results came entirely from cache
        incremental: bool - True if this was an incremental (partial) run
    """
    # ── Step 1: Fetch all current skill fingerprints from DB ──
    current_skills = _fetch_all_skill_fingerprints(db_conn)

    # ── Step 2: Load cache and diff ──
    cache = _load_cache()
    cached_fingerprints = cache.get('fingerprints', {})

    # Build maps: current hash → key, cached hash → key
    current_map = {k: v['hash'] for k, v in current_skills.items()}
    cached_map = {k: v for k, v in cached_fingerprints.items() if isinstance(v, str)}

    # Determine which keys are new, changed, removed, unchanged
    all_keys = set(current_map.keys()) | set(cached_map.keys())
    new_keys = set(current_map.keys()) - set(cached_map.keys())
    removed_keys = set(cached_map.keys()) - set(current_map.keys())
    changed_keys = {k for k in all_keys if k in current_map and k in cached_map and current_map[k] != cached_map[k]}
    unchanged_keys = {k for k in all_keys if k in current_map and k in cached_map and current_map[k] == cached_map[k]}

    # ── Step 3: If nothing changed and we have cached results, return immediately ──
    cached_results = cache.get('results')
    # Stale guard: if cache has zero skills but DB has some, force re-analysis
    if cached_results and cached_results.get('total_skills', 0) == 0 and len(current_skills) > 0:
        logger.info(f"Skill audit: cache is stale (0 skills cached, {len(current_skills)} in DB), re-analyzing")
        cached_results = None
        # Delete stale cache file
        try:
            if os.path.exists(_CACHE_PATH):
                os.remove(_CACHE_PATH)
        except Exception:
            pass
    if not force and not new_keys and not changed_keys and not removed_keys and cached_results:
        logger.info(f"Skill audit: no changes detected ({len(unchanged_keys)} skills), returning cached results")
        cached_results['cached'] = True
        cached_results['incremental'] = True
        return cached_results

    # ── Step 4: Build the active skills list for analysis ──
    # Always include all skills for similarity computation (need full matrix)
    if force or not cache.get('results'):
        # Full analysis: embed all skills
        logger.info(f"Skill audit: full analysis of {len(current_skills)} skills")
        skills_for_embedding = [s for s in current_skills.values()]
        duplicates = _compute_similarity_duplicates(skills_for_embedding)

        # Compute usage stats
        usage = _compute_usage_stats(current_skills)

        results = {
            'total_skills': len(current_skills),
            'unused_count': len(usage['unused']),
            'duplicate_pairs': len(duplicates),
            'duplicates': duplicates,
            'unused': usage['unused'],
            'promote_candidates': usage['promote_candidates'],
            'cached': False,
            'incremental': False,
        }
    else:
        # Incremental: only re-embed changed/new skills, merge with cached results
        changed_or_new = new_keys | changed_keys
        logger.info(f"Skill audit: incremental — {len(changed_or_new)} changed/new, "
                    f"{len(unchanged_keys)} unchanged, {len(removed_keys)} removed")

        # Build full list for similarity computation (all current skills)
        all_skill_list = [s for s in current_skills.values()]
        duplicates = _compute_similarity_duplicates(all_skill_list)

        # Compute fresh usage stats over all current skills
        usage = _compute_usage_stats(current_skills)

        results = {
            'total_skills': len(current_skills),
            'unused_count': len(usage['unused']),
            'duplicate_pairs': len(duplicates),
            'duplicates': duplicates,
            'unused': usage['unused'],
            'promote_candidates': usage['promote_candidates'],
            'cached': False,
            'incremental': True,
        }

    # ── Step 5: Update cache ──
    cache['fingerprints'] = {k: v['hash'] for k, v in current_skills.items()}
    cache['results'] = {k: v for k, v in results.items() if k not in ('cached', 'incremental')}
    cache['last_full_analysis'] = datetime.now(timezone.utc).isoformat()
    _save_cache(cache)

    return results


def invalidate_audit_cache():
    """Clear the audit cache so next run does a full re-analysis."""
    if os.path.exists(_CACHE_PATH):
        try:
            os.remove(_CACHE_PATH)
            logger.info("Skill audit cache invalidated (full re-analysis on next run)")
            return True
        except OSError as e:
            logger.warning(f"Failed to invalidate audit cache: {e}")
            return False
    return True


def merge_skills(db_conn, keep_id, merge_id, source='knowledge_lab'):
    """Merge merge_id's skill into keep_id, then delete merge_id's skill record.
    Updates skill_summary_hash on the keep row after merge."""
    with db_conn.cursor() as cur:
        table_map = {
            'knowledge_lab': 'knowledge_lab_files',
            'company': 'company_knowledge_base',
            'project': 'project_files',
        }
        table = table_map.get(source, 'knowledge_lab_files')

        cur.execute(f"SELECT skill_summary FROM {table} WHERE id = %s", (merge_id,))
        row = cur.fetchone()
        if not row:
            return False
        merged_content = row[0]
        cur.execute(f"SELECT skill_summary FROM {table} WHERE id = %s", (keep_id,))
        keep_row = cur.fetchone()
        if keep_row and keep_row[0]:
            new_content = keep_row[0] + "\n\n--- 合并内容 ---\n" + (merged_content or '')
        else:
            new_content = merged_content

        new_hash = _compute_skill_hash(new_content) if new_content else ''
        cur.execute(f"UPDATE {table} SET skill_summary = %s, skill_generated_at = NOW(), "
                    f"skill_summary_hash = %s WHERE id = %s",
                    (new_content, new_hash, keep_id))
        cur.execute(f"UPDATE {table} SET skill_summary = NULL, skill_generated_at = NULL, "
                    f"skill_summary_hash = NULL WHERE id = %s",
                    (merge_id,))
        db_conn.commit()

    # Invalidate cache since skills changed
    invalidate_audit_cache()
    return True
