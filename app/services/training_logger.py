"""Training data logger: captures all AI interactions + feedback for future LoRA fine-tuning.

Directory structure:
    data/training/
        raw/                         # individual session logs
            {thread_id}/
                messages.json        # conversation
                feedback.json        # ratings
                context.json         # knowledge/RAG/file context
                metadata.json        # model, tokens, timestamp
        exports/                     # auto-generated JSONL for LoRA
            manual_{YYYY-MM-DD}_q{N}.jsonl
            incremental_{YYYY-MM-DD}_q{N}.jsonl
        export_watermark.json        # last-export cursor for incremental exports
"""

import os, json, logging
from datetime import datetime, timezone
from threading import Lock

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

TRAINING_DIR = os.path.join(DATA_DIR, 'training')
RAW_DIR = os.path.join(TRAINING_DIR, 'raw')
EXPORT_DIR = os.path.join(TRAINING_DIR, 'exports')
WATERMARK_PATH = os.path.join(TRAINING_DIR, 'export_watermark.json')
_lock = Lock()


def _ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)


def log_interaction(thread_id: str, user_msg: str, assistant_response: str,
                    thinking: str = None, rating: int = None, rating_comment: str = None,
                    knowledge_files: list = None, rag_context: str = None,
                    uploaded_files: list = None, headroom_used: bool = False,
                    model: str = None, tokens: int = None, latency_ms: float = None,
                    source: str = None, search_cache_hit: bool = None):
    """Log a complete AI interaction for training data.

    Call this AFTER the assistant response is stored and feedback is received.
    """
    try:
        _ensure_dirs()
        session_dir = os.path.join(RAW_DIR, thread_id)
        os.makedirs(session_dir, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()

        # Messages
        msgs_path = os.path.join(session_dir, 'messages.json')
        messages = []
        if os.path.exists(msgs_path):
            try:
                with open(msgs_path, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
            except Exception:
                pass
        messages.append({
            'role': 'user',
            'content': user_msg,
            'timestamp': now,
        })
        if assistant_response:
            messages.append({
                'role': 'assistant',
                'content': assistant_response,
                'thinking': thinking or '',
                'timestamp': now,
            })
        with open(msgs_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

        # Feedback
        if rating is not None:
            fb_path = os.path.join(session_dir, 'feedback.json')
            feedbacks = []
            if os.path.exists(fb_path):
                try:
                    with open(fb_path, 'r', encoding='utf-8') as f:
                        feedbacks = json.load(f)
                except Exception:
                    pass
            feedbacks.append({
                'message_index': len(messages) - 1,  # the assistant response
                'rating': rating,
                'comment': rating_comment or '',
                'timestamp': now,
            })
            with open(fb_path, 'w', encoding='utf-8') as f:
                json.dump(feedbacks, f, ensure_ascii=False, indent=2)

        # Context (knowledge + RAG + files)
        ctx_path = os.path.join(session_dir, 'context.json')
        contexts = []
        if os.path.exists(ctx_path):
            try:
                with open(ctx_path, 'r', encoding='utf-8') as f:
                    contexts = json.load(f)
            except Exception:
                pass
        contexts.append({
            'message_index': len(messages) - 1,
            'knowledge_files': knowledge_files or [],
            'rag_context': rag_context or '',
            'uploaded_files': uploaded_files or [],
            'headroom_used': headroom_used,
            'search_cache_hit': search_cache_hit,
            'timestamp': now,
        })
        with open(ctx_path, 'w', encoding='utf-8') as f:
            json.dump(contexts, f, ensure_ascii=False, indent=2)

        # Metadata
        meta_path = os.path.join(session_dir, 'metadata.json')
        metadata = {
            'thread_id': thread_id,
            'model': model or 'unknown',
            'max_tokens': tokens or 0,
            'latency_ms': round(latency_ms or 0, 1),
            'source': source or 'chat',
            'last_updated': now,
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.warning(f"Training log failed for {thread_id}: {e}")


# ── Export helpers ──

def _get_min_rating() -> int:
    """Resolve min_rating from runtime_config, env, or default 3."""
    try:
        from app.services.runtime_config import get as rc_get
        return rc_get('training_min_rating', 3)
    except Exception:
        return int(os.environ.get('TRAINING_MIN_RATING', '3'))


def _build_entry(user: dict, assistant: dict, fb_map: dict, ctx_map: dict,
                 session_name: str, metadata: dict, msg_index: int) -> dict:
    """Build a single JSONL entry from raw data."""
    fb = fb_map.get(msg_index, {})
    ctx = ctx_map.get(msg_index, {})

    context_parts = []
    if ctx.get('rag_context'):
        context_parts.append(f"[RAG context]\n{ctx['rag_context']}")
    if ctx.get('knowledge_files'):
        kf_names = [k.get('filename', k.get('source', '')) for k in ctx['knowledge_files']]
        context_parts.append(f"[Knowledge files] {', '.join(kf_names)}")

    return {
        'instruction': 'You are a professional AI assistant. Answer questions accurately based on context and knowledge base content.',
        'input': user['content'],
        'output': assistant['content'],
        'thinking': assistant.get('thinking', ''),
        'context': '\n'.join(context_parts) if context_parts else '',
        'rating': fb.get('rating'),
        'rating_comment': fb.get('comment', ''),
        'source': metadata.get('source', 'chat'),
        'thread_id': session_name,
        'timestamp': user.get('timestamp', ''),
        'model': metadata.get('model', ''),
        'search_cache_hit': ctx.get('search_cache_hit'),
    }


def _scan_raw_dir(for_incremental: bool = False, since_ts: str = None, skip_corrupt: bool = False):
    """Yield (session_name, messages, feedbacks, contexts, metadata) tuples.

    If for_incremental, only yield sessions/entries timestamped after since_ts.
    If skip_corrupt, skip sessions with a .health_status.json marked 'corrupt'.
    """
    if not os.path.exists(RAW_DIR):
        return
    for session_name in sorted(os.listdir(RAW_DIR)):
        session_dir = os.path.join(RAW_DIR, session_name)
        if not os.path.isdir(session_dir):
            continue

        # Skip corrupt-marked sessions
        if skip_corrupt:
            hs = _read_health_status(session_dir)
            if hs and hs.get('status') == 'corrupt':
                continue

        msgs_path = os.path.join(session_dir, 'messages.json')
        if not os.path.exists(msgs_path):
            continue
        try:
            with open(msgs_path, 'r', encoding='utf-8') as f:
                messages = json.load(f)
        except Exception:
            continue

        # Quick skip for incremental: check if this session has any new data
        if for_incremental and since_ts:
            has_new = any(
                m.get('timestamp', '') > since_ts
                for m in messages if m.get('role') == 'user'
            )
            if not has_new:
                continue

        feedbacks = []
        fb_path = os.path.join(session_dir, 'feedback.json')
        if os.path.exists(fb_path):
            try:
                with open(fb_path, 'r', encoding='utf-8') as f:
                    feedbacks = json.load(f)
            except Exception:
                pass

        contexts = []
        ctx_path = os.path.join(session_dir, 'context.json')
        if os.path.exists(ctx_path):
            try:
                with open(ctx_path, 'r', encoding='utf-8') as f:
                    contexts = json.load(f)
            except Exception:
                pass

        metadata = {}
        meta_path = os.path.join(session_dir, 'metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception:
                pass

        yield session_name, messages, feedbacks, contexts, metadata


# ── Watermark management ──

def _load_watermark() -> dict:
    """Load export watermark from disk, or return empty dict."""
    if not os.path.exists(WATERMARK_PATH):
        return {}
    try:
        with open(WATERMARK_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_watermark(wm: dict):
    """Persist export watermark to disk."""
    os.makedirs(TRAINING_DIR, exist_ok=True)
    with open(WATERMARK_PATH, 'w', encoding='utf-8') as f:
        json.dump(wm, f, ensure_ascii=False, indent=2)


def _record_export(wm: dict, filename: str, mode: str, count: int, since: str = None):
    """Append an export record to the watermark history."""
    exports = wm.get('exports', [])
    record = {
        'file': filename,
        'time': datetime.now(timezone.utc).isoformat(),
        'mode': mode,
        'count': count,
    }
    if since:
        record['since'] = since
    exports.append(record)
    # Keep last 50 records max
    if len(exports) > 50:
        exports = exports[-50:]
    wm['exports'] = exports


def _get_latest_exported_ts(wm: dict) -> str | None:
    """Return the last_exported_timestamp from watermark (ISO format), or None."""
    return wm.get('last_exported_timestamp')


def _update_watermark_after_export(wm: dict, latest_ts: str, mode: str, filename: str, count: int):
    """Update watermark with latest exported timestamp and record this export."""
    wm['last_exported_timestamp'] = latest_ts
    wm['last_export_time'] = datetime.now(timezone.utc).isoformat()
    if mode == 'full':
        wm['last_full_export'] = datetime.now(timezone.utc).isoformat()
    else:
        wm['last_incremental_export'] = datetime.now(timezone.utc).isoformat()
    _record_export(wm, os.path.basename(filename), mode, count,
                   since=wm.get('last_full_export') if mode == 'incremental' else None)
    _save_watermark(wm)


# ── Public export API ──

def export_training_jsonl(label: str = 'manual', min_rating: int = None,
                          min_length: int = 100) -> str:
    """Full export: all training data as JSONL.

    Args:
        label: 'manual' | 'weekly' — used in filename
        min_rating: minimum star rating. None = auto-resolve from config (default 3).
                     Pass 0 to export everything.
        min_length: skip assistant responses shorter than this

    Returns the path of the exported file, or empty string if nothing exported.
    """
    if min_rating is None:
        min_rating = _get_min_rating()

    _ensure_dirs()
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M')
    suffix = f"_q{min_rating}" if min_rating > 0 else '_all'
    export_path = os.path.join(EXPORT_DIR, f"{label}_{date_str}{suffix}.jsonl")
    exported = skipped = 0
    latest_ts = '1970-01-01T00:00:00Z'

    with _lock:
        with open(export_path, 'w', encoding='utf-8') as out:
            for session_name, messages, feedbacks, contexts, metadata in _scan_raw_dir(skip_corrupt=True):
                fb_map = {f['message_index']: f for f in feedbacks}
                ctx_map = {c['message_index']: c for c in contexts}

                for i in range(0, len(messages) - 1, 2):
                    user = messages[i]
                    assistant = messages[i + 1] if i + 1 < len(messages) else None
                    if not assistant or user['role'] != 'user' or assistant['role'] != 'assistant':
                        continue

                    # Quality filter
                    fb = fb_map.get(i + 1, {})
                    if min_rating > 0:
                        rating = fb.get('rating')
                        if rating is None or rating < min_rating:
                            skipped += 1
                            continue
                    if min_length > 0 and len(assistant.get('content', '')) < min_length:
                        skipped += 1
                        continue

                    entry = _build_entry(user, assistant, fb_map, ctx_map,
                                         session_name, metadata, i + 1)
                    out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    exported += 1

                    ts = user.get('timestamp', '')
                    if ts > latest_ts:
                        latest_ts = ts

        if exported == 0:
            os.remove(export_path)
            return ''

        wm = _load_watermark()
        _update_watermark_after_export(wm, latest_ts, 'full', export_path, exported)

    logger.info(f"Full export: {exported} samples (skipped {skipped}) → {export_path} [q={min_rating}]")
    return export_path


def export_training_jsonl_incremental(min_rating: int = None, min_length: int = 100) -> str:
    """Incremental export: only interactions since the last export watermark.

    Reads export_watermark.json to find the last exported timestamp,
    then only exports entries newer than that cursor.

    Returns the path of the exported file, or empty string if nothing new to export.
    """
    if min_rating is None:
        min_rating = _get_min_rating()

    _ensure_dirs()

    wm = _load_watermark()
    since_ts = _get_latest_exported_ts(wm)

    if not since_ts:
        # No prior export — do a full export instead
        logger.info("No prior export watermark found; running full export.")
        return export_training_jsonl('incremental', min_rating=min_rating, min_length=min_length)

    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M')
    suffix = f"_q{min_rating}" if min_rating > 0 else '_all'
    export_path = os.path.join(EXPORT_DIR, f"incremental_{date_str}{suffix}.jsonl")
    exported = skipped = 0
    latest_ts = since_ts  # start from existing watermark

    with _lock:
        with open(export_path, 'w', encoding='utf-8') as out:
            for session_name, messages, feedbacks, contexts, metadata in _scan_raw_dir(
                for_incremental=True, since_ts=since_ts, skip_corrupt=True
            ):
                fb_map = {f['message_index']: f for f in feedbacks}
                ctx_map = {c['message_index']: c for c in contexts}

                for i in range(0, len(messages) - 1, 2):
                    user = messages[i]
                    assistant = messages[i + 1] if i + 1 < len(messages) else None
                    if not assistant or user['role'] != 'user' or assistant['role'] != 'assistant':
                        continue

                    # Only include entries newer than watermark
                    user_ts = user.get('timestamp', '')
                    if user_ts <= since_ts:
                        skipped += 1
                        continue

                    # Quality filter
                    fb = fb_map.get(i + 1, {})
                    if min_rating > 0:
                        rating = fb.get('rating')
                        if rating is None or rating < min_rating:
                            skipped += 1
                            continue
                    if min_length > 0 and len(assistant.get('content', '')) < min_length:
                        skipped += 1
                        continue

                    entry = _build_entry(user, assistant, fb_map, ctx_map,
                                         session_name, metadata, i + 1)
                    out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    exported += 1

                    if user_ts > latest_ts:
                        latest_ts = user_ts

        if exported == 0:
            os.remove(export_path)
            logger.info(f"Incremental export: 0 new samples (all up to date since {since_ts})")
            return ''

        _update_watermark_after_export(wm, latest_ts, 'incremental', export_path, exported)

    logger.info(f"Incremental export: {exported} new samples (skipped {skipped}) → {export_path}")
    return export_path


def get_export_history() -> dict:
    """Return export history and watermark status for admin panel."""
    wm = _load_watermark()
    exports = wm.get('exports', [])
    # Count total exported samples by mode
    total_full = sum(e['count'] for e in exports if e.get('mode') == 'full')
    total_incr = sum(e['count'] for e in exports if e.get('mode') == 'incremental')

    # Count pending (new interactions since last export)
    pending = 0
    since_ts = _get_latest_exported_ts(wm)
    if since_ts:
        try:
            from app.services.runtime_config import get as rc_get
            min_r = rc_get('training_min_rating', 3)
        except Exception:
            min_r = int(os.environ.get('TRAINING_MIN_RATING', '3'))
        for session_name, messages, feedbacks, contexts, metadata in _scan_raw_dir(
            for_incremental=True, since_ts=since_ts, skip_corrupt=True
        ):
            for i in range(0, len(messages) - 1, 2):
                user = messages[i]
                assistant = messages[i + 1] if i + 1 < len(messages) else None
                if not assistant:
                    continue
                if user.get('timestamp', '') <= since_ts:
                    continue
                fb = {f['message_index']: f for f in feedbacks}.get(i + 1, {})
                if min_r > 0 and fb.get('rating', 0) < min_r:
                    continue
                pending += 1

    return {
        'has_watermark': bool(wm),
        'last_exported_timestamp': wm.get('last_exported_timestamp'),
        'last_export_time': wm.get('last_export_time'),
        'last_full_export': wm.get('last_full_export'),
        'last_incremental_export': wm.get('last_incremental_export'),
        'total_exported_full': total_full,
        'total_exported_incremental': total_incr,
        'pending_new': pending,
        'export_files': [f for f in os.listdir(EXPORT_DIR) if f.endswith('.jsonl')]
            if os.path.exists(EXPORT_DIR) else [],
        'recent_exports': exports[-10:],  # last 10 export records
    }


def reset_export_watermark():
    """Admin: delete watermark to force a fresh full export next time."""
    if os.path.exists(WATERMARK_PATH):
        os.remove(WATERMARK_PATH)
        logger.info("Export watermark reset.")


def get_training_stats() -> dict:
    """Return stats about collected training data."""
    _ensure_dirs()
    sessions = 0
    total_interactions = 0
    rated_interactions = 0
    for session_name in os.listdir(RAW_DIR):
        session_dir = os.path.join(RAW_DIR, session_name)
        if not os.path.isdir(session_dir):
            continue
        sessions += 1
        msgs_path = os.path.join(session_dir, 'messages.json')
        if os.path.exists(msgs_path):
            try:
                with open(msgs_path, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
                pairs = len(messages) // 2
                total_interactions += pairs
            except Exception:
                pass
        fb_path = os.path.join(session_dir, 'feedback.json')
        if os.path.exists(fb_path):
            try:
                with open(fb_path, 'r', encoding='utf-8') as f:
                    rated_interactions += len(json.load(f))
            except Exception:
                pass
    # Quality distribution
    quality = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, 'unrated': total_interactions - rated_interactions}
    for session_name in os.listdir(RAW_DIR):
        fb_path = os.path.join(RAW_DIR, session_name, 'feedback.json')
        if os.path.exists(fb_path):
            try:
                with open(fb_path, 'r', encoding='utf-8') as f:
                    for fb in json.load(f):
                        r = str(fb.get('rating', ''))
                        if r in quality:
                            quality[r] += 1
            except Exception:
                pass
    return {
        'sessions': sessions,
        'interactions': total_interactions,
        'rated': rated_interactions,
        'quality': quality,
        'qualifying': sum(v for k, v in quality.items() if k in ('3', '4', '5')),
        'export_files': [f for f in os.listdir(EXPORT_DIR) if f.endswith('.jsonl')]
        if os.path.exists(EXPORT_DIR) else [],
    }


# ── Data lifecycle: cleanup old training sessions ──

def cleanup_training_sessions(retention_days: int = 90) -> int:
    """Remove training data sessions older than retention_days.

    Reads each session's metadata.json for last_updated timestamp.
    Sessions without metadata are skipped (preserved).
    Deletes the entire {thread_id}/ directory.

    Args:
        retention_days: sessions older than this are purged (default 90 = ~quarterly).

    Returns number of sessions removed.
    """
    if not os.path.exists(RAW_DIR):
        return 0

    import shutil
    from datetime import timezone as tz

    cutoff = datetime.now(tz.utc).isoformat()
    removed = 0
    skipped_no_meta = 0

    with _lock:
        for session_name in os.listdir(RAW_DIR):
            session_dir = os.path.join(RAW_DIR, session_name)
            if not os.path.isdir(session_dir):
                continue

            meta_path = os.path.join(session_dir, 'metadata.json')
            if not os.path.exists(meta_path):
                skipped_no_meta += 1
                continue

            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                last_updated = metadata.get('last_updated', '')
                if not last_updated:
                    skipped_no_meta += 1
                    continue

                # Parse ISO timestamp, compare with retention
                try:
                    ts = datetime.fromisoformat(last_updated)
                    age_days = (datetime.now(tz.utc) - ts).total_seconds() / 86400
                except (ValueError, TypeError):
                    skipped_no_meta += 1
                    continue

                if age_days > retention_days:
                    shutil.rmtree(session_dir, ignore_errors=True)
                    removed += 1

            except Exception as e:
                logger.debug(f"Skipping training session {session_name} during cleanup: {e}")
                continue

    if removed > 0:
        logger.info(f"Training data cleanup: removed {removed} sessions > {retention_days}d "
                    f"(skipped {skipped_no_meta} without valid metadata)")
    return removed


def get_training_cleanup_stats() -> dict:
    """Return stats about training data age for admin preview."""
    if not os.path.exists(RAW_DIR):
        return {'total_sessions': 0, 'oldest_days': 0, 'newest_days': 0, 'older_than_90d': 0}

    from datetime import timezone as tz
    now = datetime.now(tz.utc)
    ages = []
    retention = 90
    try:
        from app.services.runtime_config import get as rc_get
        retention = rc_get('training_retention_days', 90)
    except Exception:
        pass

    older_count = 0
    for session_name in os.listdir(RAW_DIR):
        session_dir = os.path.join(RAW_DIR, session_name)
        if not os.path.isdir(session_dir):
            continue
        meta_path = os.path.join(session_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            last_updated = metadata.get('last_updated', '')
            if last_updated:
                ts = datetime.fromisoformat(last_updated)
                age = (now - ts).total_seconds() / 86400
                ages.append(age)
                if age > retention:
                    older_count += 1
        except Exception:
            continue

    return {
        'total_sessions': len(ages),
        'oldest_days': round(max(ages), 1) if ages else 0,
        'newest_days': round(min(ages), 1) if ages else 0,
        'older_than_threshold': older_count,
        'retention_days': retention,
    }


# ── Health check & auto-repair ──

HEALTH_LOG_PATH = os.path.join(TRAINING_DIR, 'health_log.json')


def _parse_iso_safe(ts_str: str):
    """Parse ISO timestamp safely, return None on failure."""
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


def _validate_json_file(path: str) -> list[str]:
    """Validate a single JSON file. Returns list of error strings (empty = valid)."""
    errors = []
    if not os.path.exists(path):
        return ['missing']
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f'corrupt JSON: {e}']
    except Exception as e:
        return [f'unreadable: {e}']
    if data is None:
        return ['null content']
    return errors


HEALTH_STATUS_FILE = '.health_status.json'


def _read_health_status(session_dir: str) -> dict | None:
    """Read the cached health status file for a session, if it exists."""
    hp = os.path.join(session_dir, HEALTH_STATUS_FILE)
    if not os.path.exists(hp):
        return None
    try:
        with open(hp, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _write_health_status(session_dir: str, status: dict):
    """Write health status marker to session directory."""
    hp = os.path.join(session_dir, HEALTH_STATUS_FILE)
    try:
        with open(hp, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"Failed to write health status for {os.path.basename(session_dir)}: {e}")


def _check_session_integrity(session_dir: str, session_name: str) -> dict:
    """Deep-inspect a single training session. Returns detailed health report.
    Also writes/updates .health_status.json in the session directory.
    """
    # Check existing health status first
    prev = _read_health_status(session_dir)
    result = {
        'session': session_name,
        'status': 'healthy',  # healthy | warning | corrupt | missing_data
        'issues': [],
        'repairable': [],
        'files': {},
        'last_check': datetime.now(timezone.utc).isoformat(),
        'previous_status': prev.get('status') if prev else None,
    }

    # Check all 4 files
    for fname in ['messages.json', 'feedback.json', 'context.json', 'metadata.json']:
        fpath = os.path.join(session_dir, fname)
        file_errors = _validate_json_file(fpath)
        if file_errors:
            if 'missing' in file_errors:
                if fname == 'metadata.json':
                    result['status'] = 'warning'
                    result['issues'].append(f'{fname}: missing (age tracking disabled)')
                elif fname == 'messages.json':
                    result['status'] = 'corrupt'
                    result['issues'].append(f'{fname}: missing (core data)')
                else:
                    result['status'] = 'warning'
                    result['issues'].append(f'{fname}: missing')
            else:
                result['status'] = 'corrupt'
                result['issues'].append(f'{fname}: {file_errors[0]}')
            result['files'][fname] = file_errors
        else:
            result['files'][fname] = 'ok'

    # If messages.json is missing/corrupt, skip deeper checks
    if result['files'].get('messages.json') != 'ok':
        _write_health_status(session_dir, result)
        return result

    # Load data for deep checks
    try:
        with open(os.path.join(session_dir, 'messages.json'), 'r', encoding='utf-8') as f:
            messages = json.load(f)
    except Exception:
        _write_health_status(session_dir, result)
        return result

    # Check message pairing
    user_msgs = 0
    assistant_msgs = 0
    for m in messages:
        if m.get('role') == 'user':
            user_msgs += 1
        elif m.get('role') == 'assistant':
            assistant_msgs += 1

    if user_msgs != assistant_msgs:
        result['issues'].append(f'pairing: {user_msgs} user vs {assistant_msgs} assistant')
        if result['status'] == 'healthy':
            result['status'] = 'warning'
        if user_msgs > assistant_msgs:
            result['repairable'].append('truncate_last_user (unpaired)')

    # Check timestamps are valid ISO
    ts_issues = 0
    for i, m in enumerate(messages):
        ts = m.get('timestamp', '')
        if not ts or not _parse_iso_safe(ts):
            ts_issues += 1
    if ts_issues > 0:
        result['issues'].append(f'{ts_issues} messages with invalid/missing timestamp')
        if result['status'] == 'healthy':
            result['status'] = 'warning'

    # Check assistant thinking/content is not empty for all pairs
    empty_assistant = 0
    for i in range(1, len(messages), 2):
        if messages[i].get('role') == 'assistant':
            content = messages[i].get('content', '')
            if not content or not content.strip():
                empty_assistant += 1
    if empty_assistant > 0:
        result['issues'].append(f'{empty_assistant} empty assistant responses')
        if result['status'] == 'healthy':
            result['status'] = 'warning'

    # Cross-check feedback indices
    if result['files'].get('feedback.json') == 'ok':
        try:
            with open(os.path.join(session_dir, 'feedback.json'), 'r', encoding='utf-8') as f:
                feedbacks = json.load(f)
            max_msg_idx = len(messages)
            for fb in feedbacks:
                idx = fb.get('message_index', -1)
                if idx < 0 or idx >= max_msg_idx:
                    result['issues'].append(f'orphan feedback index {idx} (max={max_msg_idx})')
                    result['repairable'].append(f'remove_orphan_feedback:{idx}')
                    if result['status'] == 'healthy':
                        result['status'] = 'warning'
        except Exception:
            pass

    # Cross-check context indices
    if result['files'].get('context.json') == 'ok':
        try:
            with open(os.path.join(session_dir, 'context.json'), 'r', encoding='utf-8') as f:
                contexts = json.load(f)
            max_msg_idx = len(messages)
            for ctx in contexts:
                idx = ctx.get('message_index', -1)
                if idx < 0 or idx >= max_msg_idx:
                    result['issues'].append(f'orphan context index {idx} (max={max_msg_idx})')
                    result['repairable'].append(f'remove_orphan_context:{idx}')
                    if result['status'] == 'healthy':
                        result['status'] = 'warning'
        except Exception:
            pass

    _write_health_status(session_dir, result)
    return result


def run_training_health_check(repair: bool = False) -> dict:
    """Full health scan of all training data sessions.

    Checks:
    - All 4 JSON files exist and are valid JSON
    - Message pairing (equal user/assistant count)
    - Valid ISO timestamps on all messages
    - Non-empty assistant responses
    - Feedback indices within message range
    - Context indices within message range

    If repair=True, attempts to auto-fix repairable issues (orphan indices only).

    Returns health report dict with summary + per-session details.
    """
    if not os.path.exists(RAW_DIR):
        return {
            'status': 'ok',
            'total': 0,
            'healthy': 0, 'warning': 0, 'corrupt': 0,
            'issues_found': 0, 'repaired': 0,
            'sessions': [],
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

    sessions = []
    stats = {'healthy': 0, 'warning': 0, 'corrupt': 0, 'issues_found': 0, 'repaired': 0}
    corrupt_marked_skipped = 0  # sessions already marked corrupt before this scan

    with _lock:
        for session_name in sorted(os.listdir(RAW_DIR)):
            session_dir = os.path.join(RAW_DIR, session_name)
            if not os.path.isdir(session_dir):
                continue

            # Count previously-marked sessions that exports will skip
            prev_status = _read_health_status(session_dir)
            if prev_status and prev_status.get('status') == 'corrupt':
                corrupt_marked_skipped += 1

            report = _check_session_integrity(session_dir, session_name)
            stats[report['status']] += 1
            stats['issues_found'] += len(report['issues'])
            sessions.append(report)

            # Auto-repair if requested
            if repair and report['repairable']:
                repaired = _repair_session(session_dir, report)
                stats['repaired'] += repaired

    # Save health log
    health_record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'total': len(sessions),
        'healthy': stats['healthy'],
        'warning': stats['warning'],
        'corrupt': stats['corrupt'],
        'issues_found': stats['issues_found'],
        'repaired': stats['repaired'],
        'repair_enabled': repair,
        'corrupt_marked_skipped': corrupt_marked_skipped,
    }
    _append_health_log(health_record)

    return {
        'status': 'ok' if stats['corrupt'] == 0 else ('warning' if stats['corrupt'] < 3 else 'degraded'),
        **health_record,
        'sessions': sessions if len(sessions) <= 50 else sessions[:50],  # limit detail to 50
    }


def _repair_session(session_dir: str, report: dict) -> int:
    """Attempt auto-repair on a single session. Returns number of fixes applied."""
    fixed = 0

    for r in report['repairable']:
        if r.startswith('remove_orphan_feedback:'):
            idx_str = r.split(':')[1]
            idx = int(idx_str)
            fb_path = os.path.join(session_dir, 'feedback.json')
            if os.path.exists(fb_path):
                try:
                    with open(fb_path, 'r', encoding='utf-8') as f:
                        fbs = json.load(f)
                    new_fbs = [f for f in fbs if f.get('message_index') != idx]
                    if len(new_fbs) != len(fbs):
                        with open(fb_path, 'w', encoding='utf-8') as f:
                            json.dump(new_fbs, f, ensure_ascii=False, indent=2)
                        fixed += 1
                except Exception:
                    pass

        elif r.startswith('remove_orphan_context:'):
            idx_str = r.split(':')[1]
            idx = int(idx_str)
            ctx_path = os.path.join(session_dir, 'context.json')
            if os.path.exists(ctx_path):
                try:
                    with open(ctx_path, 'r', encoding='utf-8') as f:
                        ctxs = json.load(f)
                    new_ctxs = [c for c in ctxs if c.get('message_index') != idx]
                    if len(new_ctxs) != len(ctxs):
                        with open(ctx_path, 'w', encoding='utf-8') as f:
                            json.dump(new_ctxs, f, ensure_ascii=False, indent=2)
                        fixed += 1
                except Exception:
                    pass

        elif r == 'truncate_last_user':
            # Remove unpaired last user message
            msgs_path = os.path.join(session_dir, 'messages.json')
            if os.path.exists(msgs_path):
                try:
                    with open(msgs_path, 'r', encoding='utf-8') as f:
                        msgs = json.load(f)
                    if msgs and msgs[-1].get('role') == 'user':
                        msgs.pop()
                        with open(msgs_path, 'w', encoding='utf-8') as f:
                            json.dump(msgs, f, ensure_ascii=False, indent=2)
                        fixed += 1
                except Exception:
                    pass

    if fixed:
        logger.info(f"Auto-repaired {fixed} issue(s) in {os.path.basename(session_dir)}")

    return fixed


def _append_health_log(record: dict):
    """Append a health check record to the health log (keep last 20)."""
    logs = []
    if os.path.exists(HEALTH_LOG_PATH):
        try:
            with open(HEALTH_LOG_PATH, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except Exception:
            pass
    logs.append(record)
    if len(logs) > 20:
        logs = logs[-20:]
    os.makedirs(TRAINING_DIR, exist_ok=True)
    with open(HEALTH_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def get_health_history() -> dict:
    """Return health check history and last run summary."""
    logs = []
    if os.path.exists(HEALTH_LOG_PATH):
        try:
            with open(HEALTH_LOG_PATH, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except Exception:
            pass

    last = logs[-1] if logs else None
    # Trend: healthy vs warning vs corrupt over time
    trend = []
    for entry in logs:
        trend.append({
            'time': entry.get('timestamp', '')[:16],
            'healthy': entry.get('healthy', 0),
            'warning': entry.get('warning', 0),
            'corrupt': entry.get('corrupt', 0),
        })

    return {
        'last_check': last,
        'total_checks': len(logs),
        'trend': trend[-10:],
        'history': logs[-5:],  # last 5 detailed records
    }


# ── Export file lifecycle ──

def cleanup_old_exports(keep_count: int = None) -> dict:
    """Delete old export files, keeping the last N (sorted by mtime).

    Args:
        keep_count: number of most recent export files to keep.
                    Defaults to runtime_config ``export_retention_count`` (default 20).

    Returns dict with {deleted: [filenames...], kept: count}.
    """
    if keep_count is None:
        try:
            from app.services.runtime_config import get as rc_get
            keep_count = rc_get('export_retention_count', 20)
        except Exception:
            keep_count = 20

    if not os.path.exists(EXPORT_DIR):
        return {'deleted': [], 'kept': 0}

    files = []
    for f in os.listdir(EXPORT_DIR):
        fp = os.path.join(EXPORT_DIR, f)
        if os.path.isfile(fp) and f.endswith('.jsonl'):
            files.append((fp, f, os.path.getmtime(fp)))

    if len(files) <= keep_count:
        return {'deleted': [], 'kept': len(files)}

    # Sort by mtime ascending (oldest first), delete oldest extras
    files.sort(key=lambda x: x[2])
    to_delete = files[:len(files) - keep_count]
    deleted = []

    for fp, fname, _ in to_delete:
        try:
            os.remove(fp)
            deleted.append(fname)
        except Exception as e:
            logger.warning(f"Failed to delete old export {fname}: {e}")

    kept = len(files) - len(deleted)
    if deleted:
        logger.info(f"Export cleanup: deleted {len(deleted)} old files, kept {kept} (retention={keep_count})")

    return {'deleted': deleted, 'kept': kept}


def delete_export_file(filename: str) -> bool:
    """Delete a specific export file by name. Returns True on success."""
    fp = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(fp) or not filename.endswith('.jsonl'):
        return False
    # Path traversal guard
    if '..' in filename or '/' in filename or '\\' in filename:
        return False
    try:
        os.remove(fp)
        logger.info(f"Export file deleted by admin: {filename}")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete export {filename}: {e}")
        return False


def get_export_files_detail() -> list:
    """Return detailed list of export files with size and mtime."""
    if not os.path.exists(EXPORT_DIR):
        return []
    files = []
    for f in os.listdir(EXPORT_DIR):
        fp = os.path.join(EXPORT_DIR, f)
        if os.path.isfile(fp) and f.endswith('.jsonl'):
            st = os.stat(fp)
            files.append({
                'filename': f,
                'size_bytes': st.st_size,
                'size_mb': round(st.st_size / 1048576, 2),
                'mtime': datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                'mtime_display': datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M'),
            })
    files.sort(key=lambda x: x['mtime'], reverse=True)
    return files
