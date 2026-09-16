"""Semantic law search — embedding channel for compliance checking.

Uses the same embedding model as rag_engine.py. Results come from the
``rag_laws`` ChromaDB collection; when that collection is empty or ChromaDB is
unavailable the search returns [] and the keyword channel in
``_select_relevant_laws()`` still applies.

Usage:
    from app.services.law_semantic import semantic_law_search
    hits = semantic_law_search(query, top_k=15)
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

LAW_COLLECTION_NAME = "rag_laws"
_available = None  # tri-state: None=unchecked, True=ok, False=unavailable


def _is_chromadb_available() -> bool:
    """Check if ChromaDB is available (lightweight check)."""
    global _available
    if _available is not None:
        return _available
    try:
        import chromadb
        from app.services.rag_engine import _get_chroma_client
        client = _get_chroma_client()
        client.heartbeat()
        _available = True
    except Exception:
        _available = False
    return _available


def _get_law_collection():
    """Get or create the law collection in ChromaDB."""
    if not _is_chromadb_available():
        return None
    from app.services.rag_engine import _get_chroma_client
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=LAW_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def _get_law_index_count() -> int:
    """Return number of indexed articles, or 0 if unavailable."""
    try:
        collection = _get_law_collection()
        if not collection:
            return 0
        result = collection.get()
        return len(result.get("ids", [])) if result else 0
    except Exception:
        return 0


def semantic_law_search(query: str, top_k: int = 10) -> List[dict]:
    """Semantic search for relevant law articles matching the query.

    Returns list of {law_name, short_name, category, article, text, score}.
    Returns empty list if ChromaDB unavailable.
    """
    if not _is_chromadb_available():
        return []

    try:
        collection = _get_law_collection()
        if not collection or _get_law_index_count() == 0:
            return []

        from app.services.rag_engine import _get_model
        model = _get_model()
        query_emb = model.encode(query).tolist()

        results = collection.query(query_embeddings=[query_emb], n_results=top_k)

        hits = []
        if results and results.get("metadatas") and results["metadatas"][0]:
            for i, meta in enumerate(results["metadatas"][0]):
                doc = results["documents"][0][i] if results.get("documents") else ""
                dist = results["distances"][0][i] if results.get("distances") else 1.0
                # Cosine distance: 0=identical, 2=opposite. Convert to 0-1 score.
                score = max(0, 1.0 - dist / 2.0)
                hits.append({
                    "law_name": meta.get("law_name", ""),
                    "short_name": meta.get("short_name", ""),
                    "category": meta.get("category", ""),
                    "article": meta.get("article", ""),
                    "text": doc.split(": ", 1)[-1] if ": " in doc else doc,
                    "score": round(score, 3),
                })
        return hits
    except Exception as e:
        logger.warning(f"Semantic law search failed: {e}")
        return []
