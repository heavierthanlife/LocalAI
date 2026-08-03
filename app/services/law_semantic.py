"""Semantic law search — dual-channel (embedding + keyword) for compliance checking.

U2 upgrade: replaces pure keyword matching in _select_relevant_laws() with
semantic+keyword dual-channel, using the same embedding model as rag_engine.py.

Usage:
    from app.services.law_semantic import search_relevant_laws, rebuild_law_index
    results = search_relevant_laws(rules, top_k=15)
"""

import logging
from typing import List, Optional

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


def rebuild_law_index(law_articles: List[dict]) -> int:
    """Rebuild the ChromaDB law index from a list of law articles.

    Each article = {law_name, short_name, category, article, text, tags}.
    Returns number of articles indexed, or 0 on failure.
    """
    if not _is_chromadb_available():
        logger.warning("ChromaDB unavailable, skipping law index rebuild")
        return 0

    try:
        collection = _get_law_collection()
        if not collection:
            return 0

        # Delete existing data
        try:
            existing = collection.get()
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

        from app.services.rag_engine import _get_model
        model = _get_model()

        ids, embeddings, documents, metadatas = [], [], [], []
        for i, art in enumerate(law_articles):
            # Use article text as both the embedding source and searchable document
            text = art["text"]
            if not text or len(text) < 10:
                continue

            emb = model.encode(text).tolist()
            ids.append(f"law_{i}")
            embeddings.append(emb)
            # Document: law name + article label for display context
            documents.append(f"{art['law_name']} {art['article']}: {text}")
            metadatas.append({
                "law_name": art["law_name"],
                "short_name": art.get("short_name", ""),
                "category": art.get("category", ""),
                "article": art["article"],
            })

        if ids:
            collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            logger.info(f"Law index rebuilt: {len(ids)} articles indexed")
        return len(ids)
    except Exception as e:
        logger.warning(f"Failed to rebuild law index: {e}")
        return 0


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
