"""Wiki engine — Obsidian-flavored markdown wiki with RAG integration."""

import os, re, logging, json
from typing import List, Dict, Optional, Tuple

import yaml

from app.config import DATA_DIR
from app.database import get_db_connection
from app.services.rag_engine import _get_chroma_client, chunk_text, embed_batch

logger = logging.getLogger(__name__)

WIKI_DIR = os.path.join(DATA_DIR, 'wiki')

_FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)


def _ensure_wiki_dir():
    os.makedirs(WIKI_DIR, exist_ok=True)


def _ensure_wiki_tables():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS wiki_origin_links (
                    id SERIAL PRIMARY KEY,
                    wiki_page_path TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_file_id INTEGER NOT NULL,
                    source_name TEXT,
                    source_status TEXT DEFAULT 'active',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (wiki_page_path, source_type, source_file_id)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_wiki_origin_page ON wiki_origin_links(wiki_page_path)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_wiki_origin_source ON wiki_origin_links(source_type, source_file_id)")
            conn.commit()


def _parse_frontmatter(text: str) -> Tuple[Dict, str]:
    m = _FRONTMATTER_RE.match(text)
    if m:
        try:
            frontmatter = yaml.safe_load(m.group(1)) or {}
        except Exception:
            frontmatter = {}
        content = text[m.end():].lstrip('\n')
        return frontmatter, content
    return {}, text.lstrip('\n')


def _build_markdown(frontmatter: Dict, content: str) -> str:
    fm_yaml = yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False).strip()
    return f"---\n{fm_yaml}\n---\n\n{content.strip()}"


def _resolve_path(path: str) -> str:
    if path.startswith('/'):
        path = path[1:]
    full = os.path.normpath(os.path.join(WIKI_DIR, path))
    if not full.startswith(os.path.normpath(WIKI_DIR)):
        raise ValueError("Path traversal detected")
    return full


def _slugify(path: str) -> str:
    s = path.replace('\\', '/').replace('.md', '').replace('/', '_').replace(' ', '_')
    return re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff]', '', s)


def read_wiki_page(path: str) -> Tuple[Dict, str, str]:
    _ensure_wiki_dir()
    full_path = _resolve_path(path)
    if not os.path.isfile(full_path):
        return {}, '', full_path
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        logger.warning(f"Failed to read wiki page {full_path}: {e}")
        return {}, '', full_path
    frontmatter, content = _parse_frontmatter(text)
    return frontmatter, content, full_path


def write_wiki_page(path: str, frontmatter: Dict, content: str, lock: bool = False) -> str:
    _ensure_wiki_dir()
    full_path = _resolve_path(path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    text = _build_markdown(frontmatter, content)

    def _write():
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(text)

    if lock:
        try:
            from filelock import FileLock
            lock_path = full_path + '.lock'
            with FileLock(lock_path, timeout=10):
                _write()
        except ImportError:
            logger.warning("filelock not installed, write_wiki_page() lock ignored")
            _write()
    else:
        _write()

    try:
        from app.services.wiki_entity_service import _invalidate_comparisons_for
        _invalidate_comparisons_for(path)
    except Exception:
        pass

    return full_path


def delete_wiki_page(path: str) -> bool:
    full_path = _resolve_path(path)
    if os.path.isfile(full_path):
        try:
            os.remove(full_path)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete wiki page {full_path}: {e}")
            return False
    return False


def list_wiki_tree(prefix: str = "") -> Dict:
    _ensure_wiki_dir()

    def _build(dir_path: str, rel_path: str) -> Dict:
        name = os.path.basename(dir_path) or os.path.basename(WIKI_DIR)
        entry = {'name': name, 'type': 'dir', 'path': rel_path, 'children': []}
        try:
            items = sorted(os.listdir(dir_path))
        except Exception:
            return entry
        for item in items:
            item_full = os.path.join(dir_path, item)
            item_rel = (os.path.join(rel_path, item) if rel_path else item).replace('\\', '/')
            if os.path.isdir(item_full):
                entry['children'].append(_build(item_full, item_rel))
            elif item.endswith('.md'):
                entry['children'].append({'name': item, 'type': 'file', 'path': item_rel})
        return entry

    if prefix:
        prefix_path = _resolve_path(prefix)
        if os.path.isdir(prefix_path):
            return _build(prefix_path, prefix)
        return {'name': prefix, 'type': 'dir', 'path': prefix, 'children': []}
    return _build(WIKI_DIR, '')


def read_wiki_index() -> List[Dict]:
    path = os.path.join(WIKI_DIR, 'index.md')
    if not os.path.isfile(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        logger.warning(f"Failed to read wiki index: {e}")
        return []
    _, content = _parse_frontmatter(text)
    entries = []
    link_re = re.compile(r'\[([^\]]+)\]\(([^)]+)\)\s*(.*)')
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        m = link_re.match(line)
        if m:
            entries.append({
                'title': m.group(1),
                'path': m.group(2),
                'description': m.group(3).strip(),
            })
    return entries


def search_wiki(query: str) -> List[Dict]:
    _ensure_wiki_dir()
    results = []
    if not query:
        return results
    try:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
    except Exception:
        return results
    for root, dirs, files in os.walk(WIKI_DIR):
        for f in files:
            if not f.endswith('.md'):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, WIKI_DIR).replace('\\', '/')
            try:
                with open(full, 'r', encoding='utf-8') as fh:
                    lines = fh.readlines()
            except Exception:
                continue
            for i, line in enumerate(lines):
                if pattern.search(line):
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    snippet = ''.join(lines[start:end]).strip()
                    results.append({'path': rel, 'filename': f, 'snippet': snippet})
                    break
    return results


def index_wiki_to_rag(page_path: str, content: str, source_file_ids: List[int]) -> int:
    chunks = chunk_text(content)
    if not chunks:
        return 0
    try:
        client = _get_chroma_client()
        collection = client.get_or_create_collection(
            name='rag_wiki',
            metadata={"hnsw:space": "cosine"}
        )
        slug = _slugify(page_path)
        chunk_ids = [f"wiki_{slug}_{i}" for i in range(len(chunks))]
        meta_base = {
            'wiki_page_path': page_path,
            'chunk_type': 'wiki_page',
            'chunk_priority': 3,
            'sourced_from': str(source_file_ids),
            'source_status': 'active',
        }
        metadatas = []
        for i in range(len(chunks)):
            m = dict(meta_base)
            m['chunk_index'] = i
            m['total_chunks'] = len(chunks)
            metadatas.append(m)
        embeddings = embed_batch(chunks)
        collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        logger.info(f"Indexed wiki page '{page_path}' -> {len(chunks)} chunks")
        return len(chunks)
    except Exception as e:
        logger.warning(f"Failed to index wiki page to RAG: {e}")
        return 0


def delete_wiki_from_rag(wiki_page_path: str) -> bool:
    try:
        client = _get_chroma_client()
        collection = client.get_or_create_collection(
            name='rag_wiki',
            metadata={"hnsw:space": "cosine"}
        )
        existing = collection.get(where={"wiki_page_path": wiki_page_path})
        if existing['ids']:
            collection.delete(ids=existing['ids'])
            logger.info(f"Deleted {len(existing['ids'])} RAG chunks for wiki page '{wiki_page_path}'")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete wiki page from RAG: {e}")
        return False


def record_origin_link(wiki_page_path: str, source_type: str, source_file_id: int, source_name: str) -> bool:
    _ensure_wiki_tables()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO wiki_origin_links (wiki_page_path, source_type, source_file_id, source_name)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (wiki_page_path, source_type, source_file_id)
                    DO UPDATE SET source_name = EXCLUDED.source_name, updated_at = NOW()
                """, (wiki_page_path, source_type, source_file_id, source_name))
                conn.commit()
                return True
    except Exception as e:
        logger.warning(f"Failed to record origin link: {e}")
        return False


def update_source_status(source_type: str, source_file_id: int, new_status: str) -> int:
    _ensure_wiki_tables()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE wiki_origin_links
                    SET source_status = %s, updated_at = NOW()
                    WHERE source_type = %s AND source_file_id = %s
                """, (new_status, source_type, source_file_id))
                conn.commit()
                return cur.rowcount
    except Exception as e:
        logger.warning(f"Failed to update source status: {e}")
        return 0


def get_origin_links_for_page(wiki_page_path: str) -> List[Dict]:
    _ensure_wiki_tables()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT source_type, source_file_id, source_name, source_status, created_at
                    FROM wiki_origin_links
                    WHERE wiki_page_path = %s
                    ORDER BY source_type, source_file_id
                """, (wiki_page_path,))
                rows = cur.fetchall()
                return [
                    {
                        'source_type': r[0],
                        'source_file_id': r[1],
                        'source_name': r[2],
                        'source_status': r[3],
                        'created_at': r[4].isoformat() if hasattr(r[4], 'isoformat') else str(r[4]),
                    }
                    for r in rows
                ]
    except Exception as e:
        logger.warning(f"Failed to get origin links for page: {e}")
        return []


def get_pages_for_source(source_type: str, source_file_id: int) -> List[Dict]:
    _ensure_wiki_tables()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT wiki_page_path, source_name, source_status, created_at
                    FROM wiki_origin_links
                    WHERE source_type = %s AND source_file_id = %s
                    ORDER BY wiki_page_path
                """, (source_type, source_file_id))
                rows = cur.fetchall()
                return [
                    {
                        'wiki_page_path': r[0],
                        'source_name': r[1],
                        'source_status': r[2],
                        'created_at': r[3].isoformat() if hasattr(r[3], 'isoformat') else str(r[3]),
                    }
                    for r in rows
                ]
    except Exception as e:
        logger.warning(f"Failed to get pages for source: {e}")
        return []


def mark_source_deleted(source_type: str, source_file_id: int) -> bool:
    pages = get_pages_for_source(source_type, source_file_id)
    try:
        client = _get_chroma_client()
        collection = client.get_or_create_collection(
            name='rag_wiki',
            metadata={"hnsw:space": "cosine"}
        )
        for page in pages:
            page_path = page['wiki_page_path']
            existing = collection.get(where={"wiki_page_path": page_path})
            if existing['ids']:
                metadatas = existing['metadatas']
                for m in metadatas:
                    m['source_status'] = 'deleted'
                collection.update(ids=existing['ids'], metadatas=metadatas)
        update_source_status(source_type, source_file_id, 'deleted')
        return True
    except Exception as e:
        logger.warning(f"Failed to mark source deleted: {e}")
        return False


def mark_source_restored(source_type: str, source_file_id: int) -> bool:
    pages = get_pages_for_source(source_type, source_file_id)
    try:
        client = _get_chroma_client()
        collection = client.get_or_create_collection(
            name='rag_wiki',
            metadata={"hnsw:space": "cosine"}
        )
        for page in pages:
            page_path = page['wiki_page_path']
            existing = collection.get(where={"wiki_page_path": page_path})
            if existing['ids']:
                metadatas = existing['metadatas']
                for m in metadatas:
                    m['source_status'] = 'active'
                collection.update(ids=existing['ids'], metadatas=metadatas)
        update_source_status(source_type, source_file_id, 'active')
        return True
    except Exception as e:
        logger.warning(f"Failed to mark source restored: {e}")
        return False


def get_wiki_stats() -> Dict:
    _ensure_wiki_dir()
    _ensure_wiki_tables()
    total_pages = 0
    for root, dirs, files in os.walk(WIKI_DIR):
        total_pages += sum(1 for f in files if f.endswith('.md'))
    total_sources = 0
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(DISTINCT source_file_id) FROM wiki_origin_links")
                total_sources = cur.fetchone()[0] or 0
    except Exception as e:
        logger.warning(f"Failed to count sources: {e}")
    orphan_count = 0
    try:
        linked_pages = set()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT wiki_page_path FROM wiki_origin_links")
                for row in cur.fetchall():
                    linked_pages.add(row[0])
        all_pages = set()
        for root, dirs, files in os.walk(WIKI_DIR):
            for f in files:
                if f.endswith('.md'):
                    rel = os.path.relpath(os.path.join(root, f), WIKI_DIR).replace('\\', '/')
                    all_pages.add(rel)
        orphan_count = len(all_pages - linked_pages)
    except Exception as e:
        logger.warning(f"Failed to count orphans: {e}")
    return {'total_pages': total_pages, 'total_sources': total_sources, 'orphan_count': orphan_count}


def get_recent_wiki_pages(limit: int = 5) -> List[Dict]:
    _ensure_wiki_dir()
    files_with_mtime = []
    for root, dirs, names in os.walk(WIKI_DIR):
        for fname in names:
            if not fname.endswith('.md'):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, WIKI_DIR).replace('\\', '/')
            files_with_mtime.append((os.path.getmtime(full), rel))
    files_with_mtime.sort(reverse=True)
    result = []
    for mtime, relpath in files_with_mtime[:limit]:
        path_no_ext = relpath[:-3] if relpath.endswith('.md') else relpath
        fm, _, _ = read_wiki_page(relpath)
        title = (fm or {}).get('title', os.path.basename(path_no_ext))
        result.append({'path': path_no_ext, 'title': title, 'mtime': mtime})
    return result
