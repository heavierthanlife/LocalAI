"""Production-grade RAG engine: chunking, embedding, semantic retrieval via ChromaDB.

Architecture:
  Upload → extract_text → chunk → embed → store in ChromaDB
  Query  → embed query → semantic search → top-K chunks → context

Uses the same sentence-transformers model as skill_auditor for consistency.
ChromaDB is embedded — no external server needed.
"""

import os, logging, hashlib, re, json
from datetime import datetime, timezone
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────
CHROMA_DIR = os.path.join(DATA_DIR, 'chromadb')
CHUNK_SIZE = 500          # characters per chunk (adjustable)
CHUNK_OVERLAP = 100       # overlap between adjacent chunks
TOP_K_DEFAULT = 8         # how many chunks to retrieve per query
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# ── Lazy model loader ──────────────────────────────────────────
_embed_model = None
_collections = {}  # cache: collection_name → Collection


def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        # Use mirror if accessible
        if not os.getenv('HF_ENDPOINT'):
            os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
        _embed_model = SentenceTransformer(MODEL_NAME)
    return _embed_model


def _get_chroma_client():
    try:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        return chromadb.PersistentClient(path=CHROMA_DIR, settings=ChromaSettings(anonymized_telemetry=False))
    except Exception as e:
        logger.warning(f"ChromaDB client init failed: {e}")
        raise


def _get_collection(source: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection for a given file source."""
    name = f"rag_{source}"
    if name not in _collections:
        client = _get_chroma_client()
        _collections[name] = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    return _collections[name]


# ── Chunking ───────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks, respecting sentence boundaries.

    Strategy: split by sentences first, then greedily fill chunks with entire
    sentences until chunk_size is reached. This avoids mid-sentence cuts.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into sentences (Chinese-aware: split on 。！？\n and English punctuation)
    sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text[:chunk_size]]

    chunks = []
    current = ""
    for sentence in sentences:
        # If a single sentence exceeds chunk_size, split by word boundary
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            # Try jieba word-boundary splitting for more coherent sub-chunks
            try:
                from app.services.text_utils import lcut, has_chinese
                if has_chinese(sentence):
                    words = lcut(sentence)
                    sub = ""
                    for w in words:
                        if len(sub) + len(w) <= chunk_size:
                            sub += w
                        else:
                            if sub.strip():
                                chunks.append(sub.strip())
                            sub = w
                    if sub.strip():
                        chunks.append(sub.strip())
                    continue
            except ImportError:
                pass
            # Fallback: char-based splitting
            for i in range(0, len(sentence), chunk_size - overlap):
                sub = sentence[i:i + chunk_size]
                if sub.strip():
                    chunks.append(sub.strip())
            continue

        if len(current) + len(sentence) <= chunk_size:
            current = (current + " " + sentence).strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) >= 20]  # discard tiny chunks


# ── Embedding ──────────────────────────────────────────────────

def embed_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts."""
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


from functools import lru_cache as _lru_cache

@_lru_cache(maxsize=5000)
def _embed_query_cached(normalized_query: str) -> tuple:
    """Cacheable inner — returns tuple for hashability (lru_cache requires hashable)."""
    raw = embed_batch([normalized_query])[0]
    return tuple(raw)


def embed_query(query: str) -> List[float]:
    """Generate embedding for a single query, with LRU cache on normalized text.

    Cache hit saves ~100-200ms per query by skipping sentence-transformers inference.
    Normalization (lower + strip) increases hit rate for semantically identical queries.
    Cache stores up to 5000 unique queries (~15MB memory).
    """
    normalized = query.lower().strip()[:200]  # cap to 200 chars
    return list(_embed_query_cached(normalized))


def clear_embedding_cache():
    """Clear the embedding cache (useful after model reload)."""
    _embed_query_cached.cache_clear()


def embedding_cache_stats() -> dict:
    """Return cache hit/miss stats."""
    info = _embed_query_cached.cache_info()
    return {"hits": info.hits, "misses": info.misses, "maxsize": info.maxsize, "currsize": info.currsize}


# ── Indexing ───────────────────────────────────────────────────

def _compute_fingerprint(file_content: str) -> str:
    """Stable fingerprint for file content (not including chunks)."""
    return hashlib.sha256(file_content[:10000].encode()).hexdigest()[:16]


def index_file(
    file_id: int, content: str, source: str,
    metadata: Optional[Dict] = None, force: bool = False,
    skill_summary: Optional[str] = None
) -> Tuple[int, str]:
    """Index a single file: chunk, embed, store in ChromaDB.

    Args:
        file_id: database ID of the source file
        content: full text content of the file
        source: 'knowledge_lab' | 'company_kb' | 'project_file' | 'user_file'
        metadata: extra metadata dict (filename, owner, etc.)
        force: if True, delete existing chunks and re-index
        skill_summary: if provided, also index as skill-chunks with priority

    Returns:
        (num_chunks, fingerprint)
    """
    fingerprint = _compute_fingerprint(content)
    collection = _get_collection(source)

    # Check if unchanged
    existing = collection.get(where={"file_id": file_id}, include=["metadatas"])
    if existing['ids'] and not force:
        old_fp = existing['metadatas'][0].get('fingerprint', '') if existing['metadatas'] else ''
        if old_fp == fingerprint:
            return len(existing['ids']), fingerprint

    # Delete old chunks
    if existing['ids']:
        collection.delete(ids=existing['ids'])

    # Chunk
    chunks = chunk_text(content)
    if not chunks:
        logger.warning(f"No valid chunks for file_id={file_id} source={source}")
        return 0, fingerprint

    # Generate chunk IDs
    chunk_ids = [f"{source}_{file_id}_chunk_{i}" for i in range(len(chunks))]

    # Build metadata
    meta_base = {
        'file_id': file_id,
        'source': source,
        'fingerprint': fingerprint,
        'indexed_at': datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        meta_base.update({k: str(v) for k, v in metadata.items() if v is not None})

    metadatas = []
    for i in range(len(chunks)):
        m = dict(meta_base)
        m['chunk_index'] = i
        m['total_chunks'] = len(chunks)
        metadatas.append(m)

    # Embed and store main content chunks
    embeddings = embed_batch(chunks)
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    total_chunks = len(chunks)

    # Also index skill summary as priority chunks if available
    if skill_summary and skill_summary.strip():
        skill_chunks = chunk_text(skill_summary, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP // 2)
        if skill_chunks:
            skill_ids = [f"{source}_{file_id}_skill_{i}" for i in range(len(skill_chunks))]
            skill_metas = []
            for i in range(len(skill_chunks)):
                m = dict(meta_base)
                m['chunk_index'] = i
                m['total_chunks'] = len(skill_chunks)
                m['chunk_type'] = 'skill'       # ← marker for priority retrieval
                m['chunk_priority'] = 2          # 2 = higher than default (1)
                skill_metas.append(m)
            skill_embs = embed_batch(skill_chunks)
            collection.add(
                ids=skill_ids,
                embeddings=skill_embs,
                documents=skill_chunks,
                metadatas=skill_metas,
            )
            total_chunks += len(skill_chunks)
            logger.debug(f"Indexed {len(skill_chunks)} skill chunks for {source}.{file_id}")

    logger.info(f"Indexed file_id={file_id} source={source}: {total_chunks} chunks (incl. skills), fp={fingerprint}")
    return total_chunks, fingerprint


def delete_file_index(file_id: int, source: str):
    """Remove all chunks for a file from the index."""
    collection = _get_collection(source)
    existing = collection.get(where={"file_id": file_id})
    if existing['ids']:
        collection.delete(ids=existing['ids'])
        logger.info(f"Deleted index for file_id={file_id} source={source}")


# ── Retrieval ──────────────────────────────────────────────────

def retrieve(
    query: str, sources: List[str], top_k: int = TOP_K_DEFAULT,
    file_ids: Optional[List[int]] = None
) -> List[Dict]:
    """Semantic search: retrieve top-K chunks across specified sources.

    Args:
        query: user's question
        sources: list of source names to search (['knowledge_lab', 'company_kb', ...])
        top_k: number of chunks to return
        file_ids: optional filter — only search within specific file IDs

    Returns:
        list of dicts: [{text, source, file_id, chunk_index, metadata, score}, ...]
    """
    if not sources:
        return []

    query_embedding = embed_query(query)

    all_results = []
    for source in sources:
        try:
            collection = _get_collection(source)
            where_filter = None
            if file_ids and source in ('knowledge_lab', 'company_kb', 'project_file', 'user_file'):
                # ChromaDB doesn't support IN-filter efficiently;
                # we fetch more and filter post-hoc
                where_filter = None

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, 50),
                where=where_filter,
                include=['documents', 'metadatas', 'distances'],
            )
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    meta = results['metadatas'][0][i]
                    # Post-hoc file_id filter
                    if file_ids and meta.get('file_id') not in file_ids:
                        continue
                    all_results.append({
                        'text': results['documents'][0][i],
                        'source': source,
                        'file_id': meta.get('file_id'),
                        'chunk_index': meta.get('chunk_index', 0),
                        'metadata': meta,
                        'is_skill': meta.get('chunk_type') == 'skill',
                        'priority': int(meta.get('chunk_priority', 0)),
                        'score': 1.0 - results['distances'][0][i],  # cosine distance → similarity
                    })
        except Exception as e:
            logger.warning(f"Retrieval failed for source={source}: {e}")

    # Sort: skill chunks first (priority), then by similarity score
    all_results.sort(key=lambda r: (r['priority'], r['score']), reverse=True)
    return all_results[:top_k]


# ── Context Builder ────────────────────────────────────────────

def build_rag_context(
    query: str, sources: List[str], top_k: int = TOP_K_DEFAULT,
    max_chars: int = 8000, file_ids: Optional[List[int]] = None,
    deduplicate: bool = True
) -> str:
    """Retrieve chunks and assemble a context string for the LLM.

    Args:
        query: user question
        sources: which knowledge sources to search
        top_k: max chunks to retrieve
        max_chars: hard cap on total context length
        file_ids: optional filter
        deduplicate: remove near-duplicate chunks (same file, adjacent indices)

    Returns:
        A formatted context string ready for prompt injection
    """
    results = retrieve(query, sources, top_k=top_k, file_ids=file_ids)
    if not results:
        return ""

    # Deduplicate adjacent chunks from same file (keep highest-score)
    if deduplicate:
        seen = set()
        deduped = []
        for r in results:
            key = (r['source'], r['file_id'], r['chunk_index'])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
                # Also mark adjacent chunks as "seen" to avoid redundancy
                seen.add((r['source'], r['file_id'], r['chunk_index'] + 1))
                seen.add((r['source'], r['file_id'], r['chunk_index'] - 1))
        results = deduped

    # Assemble context with source attribution
    lines = []
    total_chars = 0
    for i, r in enumerate(results):
        source_label = {
            'knowledge_lab': '个人知识库',
            'company_kb': '公司知识库',
            'project_file': '项目文件',
            'user_file': '用户文件',
        }.get(r['source'], r['source'])

        header = f"[来源: {source_label} | 相似度: {r['score']:.0%}]"
        chunk_text = f"{header}\n{r['text']}"

        if total_chars + len(chunk_text) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 60:
                chunk_text = chunk_text[:remaining] + "..."
            else:
                break

        lines.append(chunk_text)
        total_chars += len(chunk_text)

    return "\n\n---\n\n".join(lines)


# ── Collection Management ──────────────────────────────────────

def get_index_stats() -> Dict:
    """Return stats about all RAG collections. Returns zeros if ChromaDB unavailable."""
    stats = {}
    try:
        from chromadb import PersistentClient
        _get_chroma_client()
    except Exception:
        return {'knowledge_lab': 0, 'company_kb': 0, 'project_file': 0, 'user_file': 0, 'total': 0}
    for source in ['knowledge_lab', 'company_kb', 'project_file', 'user_file']:
        try:
            collection = _get_collection(source)
            count = collection.count()
            stats[source] = count
        except Exception:
            stats[source] = 0
    stats['total'] = sum(stats.values())
    return stats


def rebuild_all_indexes(db_conn):
    """Rebuild indexes for ALL files across all sources. For admin use."""
    from psycopg2.extras import RealDictCursor

    sources = [
        ('knowledge_lab', 'knowledge_lab_files', 'user_id', 'id, user_id, content, original_name'),
        ('company_kb', 'company_knowledge_base', 'uploaded_by', 'id, uploaded_by as user_id, content, original_name'),
        ('project_file', 'project_files', 'uploaded_by', 'id, uploaded_by as user_id, content, original_name'),
        ('user_file', 'user_files', 'user_id', 'id, user_id, content, original_name'),
    ]

    total_indexed = 0
    with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
        for source, table, owner_col, cols in sources:
            try:
                cur.execute(
                    f"SELECT {cols} FROM {table} WHERE content IS NOT NULL AND content != ''"
                )
                rows = cur.fetchall()
                for row in rows:
                    try:
                        metadata = {
                            'original_name': row.get('original_name', ''),
                            'owner': row.get('user_id', ''),
                        }
                        index_file(row['id'], row['content'], source, metadata, force=True)
                        total_indexed += 1
                    except Exception as e:
                        logger.warning(f"Failed to index {source}.{row['id']}: {e}")
            except Exception as e:
                logger.warning(f"Failed to query {table}: {e}")

    return total_indexed
