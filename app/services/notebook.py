"""Personal notebook system — private markdown notes per user.

Stores notes as .md files under data/notebooks/{user_id}/.
Integrated with RAG for AI-assisted search and summarization.
"""

import os, json, logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
NOTEBOOK_DIR = DATA_DIR / "notebooks"
_lock = Lock()


def _ensure_dir(user_id: str):
    d = NOTEBOOK_DIR / user_id
    os.makedirs(d, exist_ok=True)
    return d


def list_notes(user_id: str) -> list[dict]:
    """List all notes for a user, sorted by modified time desc."""
    d = _ensure_dir(user_id)
    notes = []
    for fname in os.listdir(d):
        if fname.endswith('.md'):
            fp = d / fname
            st = fp.stat()
            notes.append({
                'id': fname[:-3],  # strip .md
                'filename': fname,
                'size_bytes': st.st_size,
                'modified': datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                'preview': _read_preview(fp),
            })
    notes.sort(key=lambda n: n['modified'], reverse=True)
    return notes


def _read_preview(fp: Path) -> str:
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            text = f.read(300)
        return text[:150].replace('\n', ' ')
    except Exception:
        return ""


def get_note(user_id: str, note_id: str) -> dict | None:
    """Read a single note."""
    fp = _ensure_dir(user_id) / f"{note_id}.md"
    if not fp.exists():
        return None
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            content = f.read()
        st = fp.stat()
        return {
            'id': note_id,
            'content': content,
            'modified': datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        }
    except Exception:
        return None


def save_note(user_id: str, note_id: str, content: str) -> dict:
    """Create or update a note. Returns the note object."""
    d = _ensure_dir(user_id)
    fp = d / f"{note_id}.md"
    with _lock:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(content)
    # Auto-index to RAG
    try:
        from app.services.rag_engine import chunk_text, embed_batch, _get_collection
        col = _get_collection('knowledge_lab')
        # Remove old chunks for this note
        try:
            col.delete(where={'note_id': note_id, 'user_id': user_id})
        except Exception:
            pass
        chunks = chunk_text(content)
        if chunks:
            embeddings = embed_batch(chunks)
            col.add(
                documents=chunks,
                embeddings=embeddings,
                ids=[f"note_{user_id}_{note_id}_{i}" for i in range(len(chunks))],
                metadatas=[{'source': 'notebook', 'user_id': user_id,
                           'note_id': note_id, 'chunk_index': i} for i in range(len(chunks))]
            )
    except Exception as e:
        logger.debug(f"Notebook RAG indexing skipped: {e}")

    return {
        'id': note_id,
        'content': content,
        'modified': datetime.now(timezone.utc).isoformat(),
    }


def delete_note(user_id: str, note_id: str) -> bool:
    """Delete a note and its RAG index."""
    fp = _ensure_dir(user_id) / f"{note_id}.md"
    if not fp.exists():
        return False
    with _lock:
        fp.unlink()
    try:
        from app.services.rag_engine import _get_collection
        col = _get_collection('knowledge_lab')
        col.delete(where={'note_id': note_id, 'user_id': user_id})
    except Exception:
        pass
    return True


def ai_summarize_note(user_id: str, note_id: str) -> str | None:
    """Generate AI summary of a note."""
    note = get_note(user_id, note_id)
    if not note or not note['content'].strip():
        return None
    try:
        from app.services.llm_provider import create_chat_model
        from app.services.prompt_safety import build_safe_system_guard
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = create_chat_model(streaming=False, temperature=0.3, max_tokens=300, timeout=30)
        system_text = "Summarize the following note in 2-3 concise sentences in Chinese. Be factual. If unclear, say '内容不明确'." + build_safe_system_guard()
        resp = llm.invoke([
            SystemMessage(content=system_text),
            HumanMessage(content=note['content'][:3000])
        ])
        return resp.content if hasattr(resp, 'content') else str(resp)
    except Exception as e:
        logger.warning(f"Note summarization failed: {e}")
        return None


def search_notes(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Semantic search across user's notes via RAG."""
    try:
        from app.services.rag_engine import embed_query, _get_collection
        col = _get_collection('knowledge_lab')
        emb = embed_query(query)
        results = col.query(
            query_embeddings=[emb],
            n_results=top_k,
            where={'user_id': user_id, 'source': 'notebook'},
        )
        matches = []
        for i, doc in enumerate(results.get('documents', [[]])[0]):
            meta = results.get('metadatas', [[]])[0][i] if i < len(results.get('metadatas', [[]])[0]) else {}
            matches.append({
                'note_id': meta.get('note_id', ''),
                'snippet': doc[:200],
                'score': round(1 - results.get('distances', [[]])[0][i], 3) if results.get('distances') else 0,
            })
        return matches
    except Exception:
        return []
