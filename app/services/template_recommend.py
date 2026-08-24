"""AI template recommendation service (U8).

Three-layer matching:
  - Metadata filter (0.3): category + project type match
  - Content similarity via ChromaDB (0.5): bid doc → template sections
  - Hotness boost (0.2): most-used templates get ranked higher

Dependencies: U5 (templates), ChromaDB for embedding similarity.
"""

import logging
from app.database import get_db_connection

logger = logging.getLogger(__name__)

_METADATA_WEIGHT = 0.3
_SIMILARITY_WEIGHT = 0.5
_HOTNESS_WEIGHT = 0.2


def recommend_templates(project_type: str = None, bid_text: str = None,
                         category: str = None, top_k: int = 5) -> list[dict]:
    """Recommend templates for a bid project.

    Args:
        project_type: e.g. '工程', '货物', '服务'
        bid_text: bid document text for content similarity
        category: optional category filter
        top_k: number of recommendations

    Returns:
        list of template dicts with {id, name, category, score, reasons}
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 1. Metadata score — category match
            conditions = ['bt.is_active = TRUE']
            params = []

            if category:
                conditions.append('bt.category = %s')
                params.append(category)
            if project_type:
                conditions.append('(bt.category = %s OR bt.category ILIKE %s)')
                params.extend([project_type, f'%{project_type}%'])

            where = ' AND '.join(conditions)

            cur.execute(f"""
                SELECT bt.id, bt.name, bt.category,
                       (SELECT COUNT(*) FROM bid_template_versions bv WHERE bv.template_id = bt.id) AS version_count,
                       COALESCE(btv.section_count, 0) AS section_count
                FROM bid_templates bt
                LEFT JOIN (SELECT template_id, COUNT(*) AS section_count FROM bid_template_versions GROUP BY template_id) btv
                    ON btv.template_id = bt.id
                WHERE {where}
                ORDER BY version_count DESC, bt.id
                LIMIT 50
            """, params)
            candidates = [
                {
                    'id': r[0], 'name': r[1], 'category': r[2],
                    'version_count': r[3] or 0, 'section_count': r[4] or 0,
                }
                for r in cur.fetchall()
            ]

            if not candidates:
                return []

            # 2. Hotness score — usage count from template_usage_log
            for c in candidates:
                cur.execute(
                    "SELECT COUNT(*) FROM template_usage_log WHERE template_id = %s",
                    (c['id'],)
                )
                c['usage_count'] = cur.fetchone()[0] or 0
                c['metadata_score'] = 1.0 if project_type and c['category'] and project_type in c['category'] else 0.5

    # 3. Content similarity via ChromaDB
    similarity_scores = {}
    if bid_text and candidates:
        try:
            similarity_scores = _chromadb_similarity(candidates, bid_text)
        except Exception as e:
            logger.warning(f"ChromaDB similarity failed, using metadata only: {e}")

    max_usage = max((c['usage_count'] for c in candidates), default=1) or 1
    for c in candidates:
        sim_score = similarity_scores.get(c['id'], 0.0)
        hotness = c['usage_count'] / max_usage

        c['final_score'] = round(
            _METADATA_WEIGHT * c['metadata_score'] +
            _SIMILARITY_WEIGHT * sim_score +
            _HOTNESS_WEIGHT * hotness,
            3
        )
        reasons = []
        reason_contribs = []
        if c['metadata_score'] > 0.5:
            reasons.append('类别匹配')
            reason_contribs.append(_METADATA_WEIGHT * c['metadata_score'])
        if sim_score > 0.3:
            reasons.append('内容相似')
            reason_contribs.append(_SIMILARITY_WEIGHT * sim_score)
        if hotness > 0.3:
            reasons.append('使用频繁')
            reason_contribs.append(_HOTNESS_WEIGHT * hotness)
        c['reasons'] = reasons
        c['reason_scores'] = [round(x, 3) for x in reason_contribs]

    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return candidates[:top_k]


def log_template_usage(template_id: int, user_id: str, project_id: int = None):
    """Record template usage for hotness tracking."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO template_usage_log (template_id, user_id, project_id)
                    VALUES (%s, %s, %s)
                """, (template_id, user_id, project_id))
                conn.commit()
    except Exception as e:
        logger.warning(f"Failed to log template usage: {e}")


def _chromadb_similarity(candidates: list[dict], bid_text: str) -> dict[int, float]:
    """Compute ChromaDB similarity between bid_text and template section content.

    Falls back to keyword-based Jaccard if ChromaDB is unavailable.
    """
    try:
        from app.services.vector_search import get_chroma_client
        client = get_chroma_client()
        if not client:
            raise RuntimeError("ChromaDB not available")

        collection = client.get_or_create_collection("bid_templates")

        # Embed bid_text
        from chromadb.utils import embedding_functions
        ef = embedding_functions.DefaultEmbeddingFunction()
        query_embedding = ef([bid_text[:2000]])

        results = collection.query(query_embeddings=query_embedding, n_results=min(len(candidates), 20))
        distances = results.get('distances', [[]])[0] if results.get('distances') else []
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []

        scores = {}
        for meta, dist in zip(metadatas, distances):
            tid = meta.get('template_id') if meta else None
            if tid and tid in {c['id'] for c in candidates}:
                similarity = max(0.0, 1.0 - (dist if dist else 0))
                if tid not in scores or similarity > scores[tid]:
                    scores[tid] = similarity
        return scores
    except Exception:
        return _keyword_fallback(candidates, bid_text)


def _keyword_fallback(candidates: list[dict], bid_text: str) -> dict[int, float]:
    """Simple keyword overlap fallback when ChromaDB is unavailable."""
    try:
        from app.database import get_db_connection
        conn = get_db_connection()
        cur = conn.cursor()

        bid_words = set(bid_text[:2000].lower().split())
        if not bid_words:
            return {}

        scores = {}
        for c in candidates:
            cur.execute(
                "SELECT btv.content FROM bid_template_versions btv WHERE btv.template_id = %s ORDER BY btv.id LIMIT 1",
                (c['id'],)
            )
            row = cur.fetchone()
            if not row or not row[0]:
                continue
            template_words = set(row[0][:2000].lower().split())
            if not template_words:
                continue
            intersection = bid_words & template_words
            union = bid_words | template_words
            scores[c['id']] = round(len(intersection) / len(union), 3) if union else 0.0

        cur.close()
        conn.close()
        return scores
    except Exception:
        return {}
