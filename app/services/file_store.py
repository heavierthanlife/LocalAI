"""File store — stream uploads to disk under user_files (long-term retention).

Design:
  - POST /upload_file streams the request body in 8MB chunks to
    USER_FILES_ORIGINAL_ROOT/{user_id}/{sha256}_{ts}{ext}; memory usage stays
    constant regardless of file size.
  - Metadata is registered in the existing user_files table
    (original_stored_path, file_hash, size_bytes BIGINT).
  - Quota: MAX_TOTAL_UPLOAD_GB per user enforced via SUM(size_bytes).
"""
import hashlib
import logging
import os
import time

from app.config import USER_FILES_ORIGINAL_ROOT, to_rel_path, resolve_path
from app.database import get_db_connection

logger = logging.getLogger(__name__)

CHUNK_SIZE = 8 * 1024 * 1024          # 8 MB streaming chunks
MAX_TOTAL_UPLOAD_GB = float(os.getenv('MAX_TOTAL_UPLOAD_GB', '10'))


class QuotaExceeded(Exception):
    """Raised when the upload would push a user over MAX_TOTAL_UPLOAD_GB."""
    def __init__(self, used_bytes: int, limit_bytes: int):
        self.used_bytes = used_bytes
        self.limit_bytes = limit_bytes
        super().__init__(f'quota {used_bytes}/{limit_bytes}')


def _user_dir(user_id: str) -> str:
    d = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)
    os.makedirs(d, exist_ok=True)
    return d


def used_bytes_for_user(user_id: str) -> int:
    """SUM(size_bytes) of this user's stored originals."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM user_files WHERE user_id = %s",
                (user_id,),
            )
            return int(cur.fetchone()[0] or 0)


def check_quota(user_id: str, incoming_bytes: int):
    """Raise QuotaExceeded if adding incoming_bytes exceeds the per-user cap."""
    limit = int(MAX_TOTAL_UPLOAD_GB * 1024 * 1024 * 1024)
    used = used_bytes_for_user(user_id)
    if used + incoming_bytes > limit:
        raise QuotaExceeded(used + incoming_bytes, limit)


def save_stream(file_storage, user_id: str, thread_id=None) -> dict:
    """Stream a Flask FileStorage to disk and register it in user_files.

    Dedupes by sha256 (re-uses the existing row if the same user already has
    identical content). Returns {file_id, filename, size, sha256}.
    """
    filename = os.path.basename(file_storage.filename or 'unnamed')
    ext = os.path.splitext(filename)[1].lower()
    user_dir = _user_dir(user_id)

    # Pass 1: hash while streaming to a temp part-file (constant memory)
    hasher = hashlib.sha256()
    tmp_path = os.path.join(user_dir, f'.part_{int(time.time() * 1000)}')
    size = 0
    try:
        with open(tmp_path, 'wb') as out:
            while True:
                chunk = file_storage.stream.read(CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                size += len(chunk)
                out.write(chunk)
        sha = hasher.hexdigest()
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Quota check after we know the real size; on failure drop the temp file
    try:
        check_quota(user_id, size)
    except QuotaExceeded:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Dedupe: same user + same hash → reuse existing row
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM user_files WHERE user_id = %s AND file_hash = %s LIMIT 1",
                (user_id, sha),
            )
            row = cur.fetchone()
            if row:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                return {'file_id': row[0], 'filename': filename, 'size': size, 'sha256': sha}

    final_name = f'{sha}_{int(time.time())}{ext}'
    final_path = os.path.join(user_dir, final_name)
    os.replace(tmp_path, final_path)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_files (user_id, thread_id, filename, size_bytes,
                                        expires_at, original_stored_path, file_hash,
                                        original_expires_at, original_name)
                VALUES (%s, %s, %s, %s, NULL, %s, %s, NULL, %s)
                RETURNING id
            """, (user_id, thread_id, filename, size,
                  to_rel_path(final_path), sha, filename))
            file_id = cur.fetchone()[0]
            conn.commit()

    logger.info(f'[file_store] saved file_id={file_id} name={filename} size={size} sha={sha[:12]}')
    return {'file_id': file_id, 'filename': filename, 'size': size, 'sha256': sha}


def resolve(file_id: int, user_id: str = None):
    """Resolve a file_id to (abs_path, filename, size).

    When user_id is given, only files owned by that user resolve (ownership check).
    Returns None if not found / not owned / missing on disk.
    """
    try:
        fid = int(file_id)
    except (TypeError, ValueError):
        return None
    sql = ("SELECT original_stored_path, original_name, COALESCE(size_bytes, 0), user_id "
           "FROM user_files WHERE id = %s")
    args = [fid]
    if user_id is not None:
        sql += " AND user_id = %s"
        args.append(str(user_id))
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(args))
            row = cur.fetchone()
    if not row:
        return None
    rel, name, size, owner = row[0], row[1], int(row[2]), row[3]
    abs_path = resolve_path(rel) if rel else None
    if not abs_path or not os.path.exists(abs_path):
        return None
    return {'abs_path': abs_path, 'filename': name or os.path.basename(abs_path),
            'size': size, 'user_id': owner}
