"""Batch document ingestion pipeline: OCR → merge → 3 target pipelines.

Accepts a ZIP of scanned textbook pages, auto-OCRs each, merges text,
then routes to:
  A) Domain dictionary  — candidate keywords review queue
  B) Company knowledge  — RAG ChromaDB index
  C) Company skills     — kb_skill_engine extraction → skills DB

Progress tracked per-task, errors logged and skipped (resilient).
"""

import os, re, json, shutil, zipfile, logging, tempfile, threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import fitz  # PyMuPDF for PDF

from app.config import DATA_DIR
from app.services.prompt_safety import build_safe_system_guard

logger = logging.getLogger(__name__)

INGEST_DIR = os.path.join(DATA_DIR, 'ingest')
DOMAIN_REVIEW_PATH = os.path.join(DATA_DIR, 'domain_words_review.json')

os.makedirs(INGEST_DIR, exist_ok=True)

# ── Progress tracking ──
_ingest_tasks: dict = {}  # task_id → {status, progress, results, error}
_task_lock = threading.Lock()

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.pdf'}


def _update_progress(task_id: str, **kwargs):
    with _task_lock:
        if task_id in _ingest_tasks:
            _ingest_tasks[task_id].update(kwargs)


def get_task_status(task_id: str) -> dict | None:
    with _task_lock:
        return _ingest_tasks.get(task_id)


def cleanup_task(task_id: str):
    with _task_lock:
        _ingest_tasks.pop(task_id, None)
        # Remove temp dir
        tmp = os.path.join(INGEST_DIR, task_id)
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


# ── OCR helpers ──

def _ocr_image(image_path: str) -> str:
    """OCR a single image file using EasyOCR."""
    try:
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img)
        from app.services.ocr import OCRManager
        ocr = OCRManager()
        if not ocr.is_available():
            raise RuntimeError("OCR engine not available")
        return ocr.run_ocr(arr)
    except Exception as e:
        logger.warning(f"OCR failed for {image_path}: {e}")
        return ""


def _ocr_pdf(pdf_path: str, task_id: str, total_pages: int) -> str:
    """OCR a PDF: render each page to image, then OCR."""
    texts = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        _update_progress(task_id, progress_pct=round((i + 1) / max(total_pages, 1) * 100),
                        detail=f"OCR PDF page {i+1}/{len(doc)}")
        pix = page.get_pixmap(dpi=200)
        img_path = os.path.join(INGEST_DIR, task_id, f'_page_{i+1:04d}.png')
        pix.save(img_path)
        txt = _ocr_image(img_path)
        if txt:
            texts.append(txt)
    doc.close()
    return '\n\n'.join(texts)


def _merge_deduplicate(texts: list[str]) -> str:
    """Merge all page texts, deduplicate repeated headers/footers."""
    lines_seen = set()
    merged = []
    for t in texts:
        for line in t.split('\n'):
            stripped = line.strip()
            if not stripped or len(stripped) < 3:
                continue
            # Skip page numbers and common headers
            if re.match(r'^\d{1,4}$', stripped):
                continue
            if re.match(r'^第[一二三四五六七八九十\d]+章|^第[一二三四五六七八九十\d]+节', stripped):
                # Chapter/section headers — keep
                merged.append(stripped)
                continue
            norm = re.sub(r'\s+', '', stripped)[:60]
            if norm not in lines_seen:
                lines_seen.add(norm)
                merged.append(stripped)
    return '\n'.join(merged)


# ── Pipeline A: Domain Dictionary ──

def _extract_domain_candidates(text: str, top_k: int = 200) -> list[dict]:
    """Extract candidate domain words from merged text."""
    from app.services.text_utils import top_keywords as tk
    kw_list = tk(text, top_k=top_k, min_len=2)
    # Load existing words to mark "already in dict"
    existing = set()
    from app.services.text_utils import DOMAIN_DICT
    if DOMAIN_DICT.exists():
        with open(DOMAIN_DICT, 'r', encoding='utf-8') as f:
            existing = {l.strip() for l in f if l.strip() and not l.startswith('#')}
    return [
        {'word': w, 'count': c, 'already_in_dict': w in existing}
        for w, c in kw_list if not w.isdigit()
    ]


def _save_domain_review(candidates: list[dict]):
    """Save candidate words to review queue."""
    existing = []
    if os.path.exists(DOMAIN_REVIEW_PATH):
        try:
            with open(DOMAIN_REVIEW_PATH, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            pass
    # Merge: add new, deduplicate by word
    seen = {e['word'] for e in existing}
    for c in candidates:
        if c['word'] not in seen:
            existing.append(c)
            seen.add(c['word'])
    existing.sort(key=lambda x: x['count'], reverse=True)
    os.makedirs(INGEST_DIR, exist_ok=True)
    with open(DOMAIN_REVIEW_PATH, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


# ── Pipeline B: Company KB (with review queue) ──

KB_REVIEW_PATH_TEMPLATE = os.path.join(INGEST_DIR, 'kb_review_{task_id}.json')


def _prepare_kb_review(text: str, task_id: str, title: str = "Batch Ingestion") -> dict:
    """Chunk text and save to review queue. Returns {chunk_count, review_path}."""
    from app.services.rag_engine import chunk_text
    chunks = chunk_text(text)
    if not chunks:
        return {'chunk_count': 0, 'review_path': ''}

    # Save full chunks + sample for preview
    review_data = {
        'task_id': task_id,
        'title': title,
        'total_chunks': len(chunks),
        'chunks': chunks,
        'sample_size': min(10, len(chunks)),
        'sample_indices': _pick_sample_indices(len(chunks), min(10, len(chunks))),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'pending_review',
        'edited_chunks': {},  # {index: corrected_text}
        'rejected_indices': set(),
    }

    review_path = KB_REVIEW_PATH_TEMPLATE.format(task_id=task_id)
    with open(review_path, 'w', encoding='utf-8') as f:
        json.dump(review_data, f, ensure_ascii=False, indent=2)

    return {'chunk_count': len(chunks), 'review_path': review_path}


def _pick_sample_indices(total: int, count: int) -> list[int]:
    """Pick evenly spaced sample indices for review preview."""
    if total <= count:
        return list(range(total))
    step = max(1, total // count)
    return list(range(0, total, step))[:count]


def get_kb_review_sample(task_id: str) -> dict | None:
    """Get sample chunks for admin review."""
    review_path = KB_REVIEW_PATH_TEMPLATE.format(task_id=task_id)
    if not os.path.exists(review_path):
        return None
    try:
        with open(review_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        samples = []
        for idx in data.get('sample_indices', []):
            text = data.get('edited_chunks', {}).get(str(idx), data['chunks'][idx])
            samples.append({
                'index': idx,
                'text': text[:300],
                'is_edited': str(idx) in data.get('edited_chunks', {}),
            })
        return {
            'task_id': task_id,
            'title': data.get('title', ''),
            'status': data.get('status', 'unknown'),
            'total_chunks': data.get('total_chunks', 0),
            'samples': samples,
        }
    except Exception:
        return None


def get_kb_review_chunk(task_id: str, chunk_index: int) -> dict | None:
    """Get a specific chunk for editing, with surrounding context."""
    review_path = KB_REVIEW_PATH_TEMPLATE.format(task_id=task_id)
    if not os.path.exists(review_path):
        return None
    try:
        with open(review_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if chunk_index >= len(data['chunks']):
            return None
        text = data.get('edited_chunks', {}).get(str(chunk_index), data['chunks'][chunk_index])
        prev_text = data['chunks'][chunk_index - 1][:100] if chunk_index > 0 else ''
        next_text = data['chunks'][chunk_index + 1][:100] if chunk_index + 1 < len(data['chunks']) else ''
        return {
            'index': chunk_index,
            'text': text,
            'prev_context': prev_text,
            'next_context': next_text,
            'is_edited': str(chunk_index) in data.get('edited_chunks', {}),
        }
    except Exception:
        return None


def update_kb_review_chunk(task_id: str, chunk_index: int, new_text: str) -> bool:
    """Admin corrects a chunk's OCR errors."""
    review_path = KB_REVIEW_PATH_TEMPLATE.format(task_id=task_id)
    if not os.path.exists(review_path):
        return False
    try:
        with open(review_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if chunk_index >= len(data['chunks']):
            return False
        data.setdefault('edited_chunks', {})[str(chunk_index)] = new_text
        with open(review_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def approve_kb_review(task_id: str) -> int:
    """Approve KB review: index all non-rejected chunks to ChromaDB. Returns chunk count."""
    review_path = KB_REVIEW_PATH_TEMPLATE.format(task_id=task_id)
    if not os.path.exists(review_path):
        return 0
    try:
        with open(review_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return 0

    title = data.get('title', 'Batch Ingestion')
    rejected = set(data.get('rejected_indices', []))
    edited = data.get('edited_chunks', {})

    # Build final chunk list
    final_chunks = []
    for i, chunk in enumerate(data['chunks']):
        if i in rejected:
            continue
        final_chunks.append(edited.get(str(i), chunk))

    if not final_chunks:
        data['status'] = 'rejected_all'
        with open(review_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return 0

    # Index to ChromaDB
    from app.services.rag_engine import embed_batch, _get_collection
    col = _get_collection('company_kb')
    embeddings = embed_batch(final_chunks)
    fid = str(hash(task_id + title))[:12]
    col.add(
        documents=final_chunks,
        embeddings=embeddings,
        ids=[f"{fid}_{i}" for i in range(len(final_chunks))],
        metadatas=[{'source': 'ingest_pipeline', 'task_id': task_id,
                     'title': title, 'chunk_index': i} for i in range(len(final_chunks))]
    )

    data['status'] = 'approved'
    data['indexed_chunks'] = len(final_chunks)
    data['approved_at'] = datetime.now(timezone.utc).isoformat()
    with open(review_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return len(final_chunks)


def reject_kb_chunk(task_id: str, chunk_index: int) -> bool:
    """Admin rejects a specific chunk (won't be indexed)."""
    review_path = KB_REVIEW_PATH_TEMPLATE.format(task_id=task_id)
    if not os.path.exists(review_path):
        return False
    try:
        with open(review_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rejected = set(data.get('rejected_indices', []))
        rejected.add(chunk_index)
        data['rejected_indices'] = list(rejected)
        with open(review_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# ── Pipeline D: AI Structured Document Extraction ──

STRUCTURED_PROMPT = """你是一名专业的采购/招投标文档分析员。请从以下中文招投标文档文本中提取结构化数据。

请识别并提取以下字段（如存在；只输出 JSON，不要解释）：
{
  "document_type": "bidding_announcement | evaluation_report | contract | qualification | other",
  "project_name": "...",
  "bid_number": "...",
  "procurement_agency": "...",
  "bid_opening_date": "...",
  "evaluation_method": "comprehensive_scoring | lowest_price | other",
  "budget_amount_cny": number 或 null,
  "bid_sections": ["section1", "section2"],
  "key_requirements": ["req1", "req2"],
  "winning_bidder": "..." 或 null,
  "winning_amount_cny": number 或 null,
  "contract_parties": ["partyA", "partyB"],
  "deadline_date": "...",
  "summary": "1-2 句摘要"
}

如果找不到某字段，使用 null。只输出合法 JSON，不要用 ```json 代码块包裹，不要附加任何说明。仅输出 JSON 对象。"""
STRUCTURED_PROMPT += build_safe_system_guard()


def _ai_extract_structured(text: str, task_id: str) -> dict | None:
    """Use LLM to extract structured procurement data from merged text."""
    try:
        from app.services.llm_provider import create_chat_model
        from app.services.prompt_safety import safe_json_parse
        from langchain_core.messages import SystemMessage, HumanMessage
        # Truncate to ~8000 chars to avoid token limits
        truncated = text[:8000]
        llm = create_chat_model(streaming=False, temperature=0.2, max_tokens=1000,
                                timeout=60)
        resp = llm.invoke([SystemMessage(content=STRUCTURED_PROMPT),
                          HumanMessage(content=truncated)])
        raw = resp.content if hasattr(resp, 'content') else str(resp)

        # Use safe_json_parse with auto-retry for ```json fences etc.
        result = safe_json_parse(raw, max_retries=1)
        if result:
            return result

        # If still failing, retry once with explicit error correction prompt
        logger.warning(f"JSON parse failed for {task_id}, retrying with correction prompt...")
        retry_prompt = (
            "The previous response was not valid JSON. Please output ONLY the JSON object, "
            "no markdown fences, no commentary. If a field is missing, use null.\n\n"
            f"Previous response (invalid): {raw[:500]}"
        )
        resp2 = llm.invoke([SystemMessage(content=STRUCTURED_PROMPT),
                           HumanMessage(content=truncated + '\n\n' + retry_prompt)])
        raw2 = resp2.content if hasattr(resp2, 'content') else str(resp2)
        return safe_json_parse(raw2)

    except Exception as e:
        logger.warning(f"AI structured extraction failed for {task_id}: {e}")
        return None


STRUCTURED_DIR = os.path.join(INGEST_DIR, 'structured')


def _save_structured_data(task_id: str, data: dict) -> str:
    """Save extracted structured data to JSON file."""
    os.makedirs(STRUCTURED_DIR, exist_ok=True)
    path = os.path.join(STRUCTURED_DIR, f'{task_id}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def get_structured_documents() -> list[dict]:
    """List all AI-extracted structured documents."""
    if not os.path.exists(STRUCTURED_DIR):
        return []
    docs = []
    for fname in os.listdir(STRUCTURED_DIR):
        if fname.endswith('.json'):
            try:
                with open(os.path.join(STRUCTURED_DIR, fname), 'r', encoding='utf-8') as f:
                    docs.append(json.load(f))
            except Exception:
                pass
    docs.sort(key=lambda d: d.get('project_name', ''))
    return docs


# ── Pipeline C: Skills ──

def _ingest_to_skills(text: str, task_id: str, title: str = "Batch Ingestion") -> dict:
    """Extract skills from merged text and store to DB. Returns counts per type."""
    from app.services.kb_skill_engine import (
        extract_frameworks, extract_principles, extract_techniques,
        extract_antipatterns, extract_key_concepts, extract_definitions,
        extract_checkable_steps
    )
    results = {
        'frameworks': extract_frameworks(text),
        'principles': extract_principles(text),
        'techniques': extract_techniques(text),
        'antipatterns': extract_antipatterns(text),
        'concepts': extract_key_concepts(text),
        'definitions': extract_definitions(text),
        'steps': extract_checkable_steps(text),
    }

    # Store to DB
    try:
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for category, items in results.items():
                    for item_text in items:
                        text_val = item_text if isinstance(item_text, str) else str(item_text)
                        cur.execute(
                            """INSERT INTO knowledge_lab_skills (category, content, source, is_company)
                               VALUES (%s, %s, %s, TRUE)
                               ON CONFLICT DO NOTHING""",
                            (category, text_val[:500], f'ingest:{task_id}')
                        )
                conn.commit()
    except Exception as e:
        logger.warning(f"Failed to store skills to DB: {e}")

    return {k: len(v) for k, v in results.items()}


# ── Main pipeline ──

def run_ingestion_pipeline(task_id: str, zip_path: str, targets: list[str] = None):
    """Run the full ingestion pipeline.

    When Celery is available, runs as an async task (non-blocking).
    When Celery is not available, falls back to direct execution (blocking).

    Args:
        task_id: unique task identifier
        zip_path: path to uploaded ZIP file
        targets: list of pipeline targets: ['domain', 'knowledge', 'skills']
                 Defaults to all three.
    """
    # Try Celery first, fall back to direct execution
    try:
        from celery_app import celery
        _run_ingestion_pipeline_async.delay(task_id, zip_path, targets or ['domain', 'knowledge', 'skills'])
        _init_task_status(task_id, targets or ['domain', 'knowledge', 'skills'])
        return
    except Exception:
        pass  # Celery not available, run directly

    if targets is None:
        targets = ['domain', 'knowledge', 'skills']
    _run_ingestion_pipeline_sync(task_id, zip_path, targets)


def _run_ingestion_pipeline_sync(task_id, zip_path, targets):
    """Direct execution (fallback when Celery is not available)."""

    with _task_lock:
        _ingest_tasks[task_id] = {
            'status': 'starting', 'progress_pct': 0, 'detail': 'Extracting ZIP...',
            'started_at': datetime.now(timezone.utc).isoformat(),
            'targets': targets, 'results': {}, 'errors': [],
        }

    tmp_dir = os.path.join(INGEST_DIR, task_id)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # Step 1: Extract ZIP
        _update_progress(task_id, progress_pct=5, detail='Extracting ZIP...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmp_dir)

        # Collect all supported files
        files = []
        for root, dirs, filenames in os.walk(tmp_dir):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in SUPPORTED_EXTS:
                    files.append(os.path.join(root, fn))

        total = len(files)
        if total == 0:
            _update_progress(task_id, status='error', detail='No supported files found in ZIP')
            return

        _update_progress(task_id, progress_pct=10, detail=f'Found {total} files. Starting OCR...',
                        total_files=total)

        # Step 2: OCR all files
        texts = []
        pdf_pages = []
        for i, fp in enumerate(sorted(files)):
            ext = os.path.splitext(fp)[1].lower()
            pct = 10 + int((i / max(total, 1)) * 40)  # 10%→50%
            _update_progress(task_id, progress_pct=pct,
                            detail=f'OCR {i+1}/{total}: {os.path.basename(fp)}')

            try:
                if ext == '.pdf':
                    pdf_page_count = len(fitz.open(fp))
                    txt = _ocr_pdf(fp, task_id, pdf_page_count)
                    pdf_pages.extend([fp] * pdf_page_count)
                else:
                    txt = _ocr_image(fp)
                if txt:
                    texts.append(txt)
            except Exception as e:
                _update_progress(task_id, errors=_ingest_tasks[task_id].get('errors', []) +
                                [f'{os.path.basename(fp)}: {str(e)[:80]}'])

        if not texts:
            _update_progress(task_id, status='error', detail='OCR produced no text from any file')
            return

        # Step 3: Merge & deduplicate
        _update_progress(task_id, progress_pct=55, detail='Merging and deduplicating...')
        merged = _merge_deduplicate(texts)

        title = f"Batch Ingestion {task_id[:8]}"

        # Step 4: Run selected pipelines
        pct_per = 40 / max(len(targets), 1)
        cur_pct = 60

        if 'domain' in targets:
            _update_progress(task_id, progress_pct=cur_pct, detail='Pipeline A: extracting domain words...')
            candidates = _extract_domain_candidates(merged)
            _save_domain_review(candidates)
            _ingest_tasks[task_id]['results']['domain_candidates'] = len(candidates)
            cur_pct += pct_per

        if 'knowledge' in targets:
            _update_progress(task_id, progress_pct=round(cur_pct),
                            detail='Pipeline B: preparing KB chunks for review...')
            kb_info = _prepare_kb_review(merged, task_id, title)
            _ingest_tasks[task_id]['results']['knowledge_chunks'] = kb_info['chunk_count']
            _ingest_tasks[task_id]['results']['kb_awaiting_review'] = True
            _ingest_tasks[task_id]['results']['kb_review_task_id'] = task_id
            cur_pct += pct_per

        if 'skills' in targets:
            _update_progress(task_id, progress_pct=round(cur_pct),
                            detail='Pipeline C: extracting skills...')
            skill_counts = _ingest_to_skills(merged, task_id, title)
            _ingest_tasks[task_id]['results']['skills'] = skill_counts
            cur_pct += pct_per

        if 'structured' in targets:
            _update_progress(task_id, progress_pct=round(cur_pct),
                            detail='Pipeline D: AI extracting structured fields...')
            structured = _ai_extract_structured(merged, task_id)
            if structured:
                path = _save_structured_data(task_id, structured)
                _ingest_tasks[task_id]['results']['structured'] = {
                    'document_type': structured.get('document_type', 'unknown'),
                    'project_name': structured.get('project_name', ''),
                    'saved_to': path,
                }
            else:
                _ingest_tasks[task_id]['results']['structured'] = None
            cur_pct += pct_per

        _update_progress(task_id, status='completed', progress_pct=100,
                        detail='Ingestion complete', completed_at=datetime.now(timezone.utc).isoformat())

    except Exception as e:
        logger.error(f"Ingestion pipeline {task_id} failed: {e}", exc_info=True)
        _update_progress(task_id, status='error', detail=str(e)[:200],
                        error=str(e))


def start_ingestion_async(task_id: str, zip_path: str, targets: list[str] = None):
    """Start ingestion — automatically uses Celery if available, else thread."""
    try:
        # Celery path (non-blocking)
        from celery_app import celery
        _run_ingestion_pipeline_async.delay(task_id, zip_path, targets or ['domain', 'knowledge', 'skills'])
        _init_task_status(task_id, targets or ['domain', 'knowledge', 'skills'])
        return task_id
    except Exception:
        pass
    # Thread fallback (blocking in-thread)
    t = threading.Thread(target=run_ingestion_pipeline, args=(task_id, zip_path, targets),
                        daemon=True, name=f'ingest_{task_id}')
    t.start()
    return task_id


# ── Celery + Redis task support ──

def _init_task_status(task_id, targets):
    """Initialize Redis task status for cross-worker progress tracking."""
    try:
        import redis, json, os
        r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
        r.setex(f'ingest:{task_id}', 86400, json.dumps({
            'status': 'queued', 'progress_pct': 0, 'detail': 'Queued for processing...',
            'targets': targets, 'started_at': None, 'results': {}, 'errors': [],
        }))
    except Exception:
        pass  # Redis not available, fall back to in-memory tracking


def _update_redis_progress(task_id, **kwargs):
    """Update task progress in Redis (safe — no-op if Redis unavailable)."""
    try:
        import redis, json, os
        r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
        key = f'ingest:{task_id}'
        data = json.loads(r.get(key) or '{}')
        data.update(kwargs)
        r.setex(key, 86400, json.dumps(data, default=str))
    except Exception:
        pass


def _run_ingestion_pipeline_async(task_id, zip_path, targets):
    """Celery task wrapper — runs in worker process."""
    _update_redis_progress(task_id, status='running', detail='Worker started...')
    try:
        _run_ingestion_pipeline_sync(task_id, zip_path, targets)
    except Exception as e:
        _update_redis_progress(task_id, status='error', error=str(e)[:200])


# ── Domain words review API helpers ──

def get_domain_review_candidates() -> list[dict]:
    """Get pending domain word candidates for admin review."""
    if not os.path.exists(DOMAIN_REVIEW_PATH):
        return []
    try:
        with open(DOMAIN_REVIEW_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def approve_domain_words(word_list: list[str]) -> int:
    """Approve selected domain words: append to domain_words.txt, remove from review."""
    if not word_list:
        return 0
    from app.services.text_utils import DOMAIN_DICT

    # Append to domain dictionary
    os.makedirs(os.path.dirname(DOMAIN_DICT), exist_ok=True)
    with open(DOMAIN_DICT, 'a', encoding='utf-8') as f:
        for w in word_list:
            f.write(f'{w}\n')

    # Remove from review queue
    approved = set(word_list)
    all_candidates = get_domain_review_candidates()
    remaining = [c for c in all_candidates if c['word'] not in approved]
    with open(DOMAIN_REVIEW_PATH, 'w', encoding='utf-8') as f:
        json.dump(remaining, f, ensure_ascii=False, indent=2)

    logger.info(f"Approved {len(word_list)} domain words, {len(remaining)} remaining in review")
    return len(word_list)


def reject_domain_words(word_list: list[str]) -> int:
    """Reject domain words: remove from review queue without adding to dict."""
    if not word_list:
        return 0
    rejected = set(word_list)
    all_candidates = get_domain_review_candidates()
    remaining = [c for c in all_candidates if c['word'] not in rejected]
    with open(DOMAIN_REVIEW_PATH, 'w', encoding='utf-8') as f:
        json.dump(remaining, f, ensure_ascii=False, indent=2)
    return len(word_list)


# ── Stale review detection & auto-cleanup ──

def _get_warn_days() -> int:
    """Get review warning threshold from runtime_config (default 3 days)."""
    try:
        from app.services.runtime_config import get as rc_get
        return rc_get('ingest_review_warn_days', 3)
    except Exception:
        return 3


def _get_cleanup_grace_days() -> int:
    """Reviews are auto-cleaned after warn_days + 3 more days of inactivity."""
    return _get_warn_days() + 3


def check_stale_reviews() -> dict:
    """Scan for stale review data (domain words + KB reviews).

    Returns dict with counts of stale items and their ages.
    """
    warn_days = _get_warn_days()
    cleanup_days = _get_cleanup_grace_days()
    now = datetime.now(timezone.utc)
    result = {'domain_candidates': 0, 'kb_reviews': 0, 'kb_review_tasks': [],
               'overdue_cleanup': 0, 'warn_days': warn_days, 'cleanup_days': cleanup_days}

    # Check domain review queue
    if os.path.exists(DOMAIN_REVIEW_PATH):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(DOMAIN_REVIEW_PATH), tz=timezone.utc)
            age_days = (now - mtime).total_seconds() / 86400
            if age_days > warn_days:
                with open(DOMAIN_REVIEW_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                result['domain_candidates'] = len(data) if isinstance(data, list) else 0
                if age_days > cleanup_days:
                    result['overdue_cleanup'] += 1
        except Exception:
            pass

    # Check KB review files
    for fname in os.listdir(INGEST_DIR):
        if fname.startswith('kb_review_') and fname.endswith('.json'):
            fpath = os.path.join(INGEST_DIR, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                status = data.get('status', '')
                if status in ('pending_review',):
                    mtime = datetime.fromtimestamp(os.path.getmtime(fpath), tz=timezone.utc)
                    age_days = (now - mtime).total_seconds() / 86400
                    if age_days > warn_days:
                        task_id = data.get('task_id', fname.replace('kb_review_', '').replace('.json', ''))
                        result['kb_review_tasks'].append({
                            'task_id': task_id, 'title': data.get('title', ''),
                            'chunks': data.get('total_chunks', 0), 'age_days': round(age_days, 1)
                        })
                        result['kb_reviews'] += 1
                        if age_days > cleanup_days:
                            result['overdue_cleanup'] += 1
            except Exception:
                pass

    result['has_stale'] = result['domain_candidates'] > 0 or result['kb_reviews'] > 0
    return result


def cleanup_stale_reviews() -> dict:
    """Auto-cleanup stale reviews that exceed cleanup_days. Logs all deletions.

    Called by scheduled job. Returns count of cleaned items.
    """
    result = check_stale_reviews()
    cleanup_days = _get_cleanup_grace_days()
    cleaned = 0

    if result['domain_candidates'] > 0:
        # Check if domain review is overdue for cleanup
        if os.path.exists(DOMAIN_REVIEW_PATH):
            mtime = datetime.fromtimestamp(os.path.getmtime(DOMAIN_REVIEW_PATH), tz=timezone.utc)
            age_days = (datetime.now(timezone.utc) - mtime).total_seconds() / 86400
            if age_days > cleanup_days:
                try:
                    with open(DOMAIN_REVIEW_PATH, 'r', encoding='utf-8') as f:
                        count = len(json.load(f))
                    os.remove(DOMAIN_REVIEW_PATH)
                    cleaned += count
                    logger.info(f"Auto-cleaned {count} stale domain word candidates ({round(age_days,1)}d old)")
                except Exception as e:
                    logger.warning(f"Failed to cleanup stale domain review: {e}")

    for task in result['kb_review_tasks']:
        if task['age_days'] > cleanup_days:
            fpath = KB_REVIEW_PATH_TEMPLATE.format(task_id=task['task_id'])
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    cleaned += 1
                    logger.info(f"Auto-cleaned stale KB review {task['task_id']} ({task['age_days']}d old)")
                except Exception as e:
                    logger.warning(f"Failed to cleanup stale KB review {task['task_id']}: {e}")

    if cleaned > 0:
        logger.info(f"Stale review cleanup: removed {cleaned} items (cleanup threshold: {cleanup_days}d)")

    return {'cleaned': cleaned, 'domain_cleaned': 0, 'kb_cleaned': 0, 'total_stale': result['kb_reviews'] + (1 if result['domain_candidates'] > 0 else 0)}
