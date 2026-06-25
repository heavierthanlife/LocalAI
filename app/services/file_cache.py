"""File cache management for chat file station."""
import hashlib
from threading import RLock

from flask import session

from ..database import get_db_connection
from ..config import is_valid_extracted_text, logger


class FileTextCache:
    """Static helpers for file text caching in database."""

    def __init__(self):
        pass

    @staticmethod
    def get_key(file_storage):
        from ..utils import utc_now
        file_bytes = file_storage.read()
        file_storage.seek(0)
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        size = len(file_bytes)
        return f"{file_hash}_{size}"

    @staticmethod
    def get_cached_text(file_storage, max_age_seconds=86400):
        from ..utils import utc_now
        key = FileTextCache.get_key(file_storage)
        from ..database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT extracted_text, updated_at FROM file_text_cache WHERE file_hash = %s ORDER BY updated_at DESC LIMIT 1",
                    (key,))
                row = cur.fetchone()
                if row:
                    extracted_text, updated_at = row
                    if (utc_now() - updated_at).total_seconds() < max_age_seconds:
                        if is_valid_extracted_text(extracted_text):
                            return extracted_text
                        else:
                            logger.warning(f"Cached text for key {key} is invalid. Ignoring.")
                            cur.execute("DELETE FROM file_text_cache WHERE file_hash = %s", (key,))
                            conn.commit()
        return None

    @staticmethod
    def store_cached_text(file_storage, extracted_text):
        if not extracted_text or not is_valid_extracted_text(extracted_text):
            logger.warning(f"Not storing invalid extracted text for {file_storage.filename}")
            return
        key = FileTextCache.get_key(file_storage)
        from ..database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO file_text_cache (file_hash, extracted_text, updated_at) VALUES (%s, %s, NOW()) ON CONFLICT (file_hash) DO UPDATE SET extracted_text = EXCLUDED.extracted_text, updated_at = NOW()",
                    (key, extracted_text))
                conn.commit()


class FileCacheManager:
    """In-memory file content cache with per-thread LRU eviction."""

    def __init__(self, max_cached_files=10, max_content_size=50 * 1024):
        self._lock = RLock()
        self.caches = {}
        self.recent = {}
        self.max_cached = max_cached_files
        self.max_size = max_content_size

    def add(self, thread_id, filename, content, user_id):
        if content is None:
            content = ''
        with self._lock:
            if len(content) > self.max_size:
                content = content[:self.max_size] + "\n[内容已截断，仅保留前50KB]"
            cache = self.caches.setdefault(thread_id, {})
            recent_list = self.recent.setdefault(thread_id, [])
            cache[filename] = content
            if filename in recent_list:
                recent_list.remove(filename)
            recent_list.insert(0, filename)
            while len(recent_list) > self.max_cached:
                old = recent_list.pop()
                del cache[old]

    def load_from_db(self, thread_id, user_id):
        with self._lock:
            if session.get('consent_value', 0) != 1:
                self.caches[thread_id] = {}
                self.recent[thread_id] = []
                return
            from ..database import get_db_connection
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT filename, content FROM user_files WHERE thread_id = %s AND user_id = %s AND (expires_at IS NULL OR expires_at > NOW())",
                        (thread_id, user_id))
                    rows = cur.fetchall()
                    if rows:
                        cache = {}
                        recent_list = []
                        for filename, content in rows:
                            if content is None:
                                content = ''
                            cache[filename] = content
                            recent_list.append(filename)
                        self.caches[thread_id] = cache
                        self.recent[thread_id] = recent_list
                    else:
                        self.caches[thread_id] = {}
                        self.recent[thread_id] = []

    def get_recent_with_lock(self, thread_id):
        with self._lock:
            return self.recent.get(thread_id, []).copy()

    def get_content(self, thread_id, filename):
        with self._lock:
            return self.caches.get(thread_id, {}).get(filename)

    def clear_thread(self, thread_id):
        with self._lock:
            self.caches.pop(thread_id, None)
            self.recent.pop(thread_id, None)

    def evict_oldest(self, max_threads=20):
        with self._lock:
            while len(self.caches) > max_threads:
                oldest = list(self.caches.keys())[0]
                self.caches.pop(oldest, None)
                self.recent.pop(oldest, None)

    def add_thread(self, thread_id):
        with self._lock:
            self.evict_oldest()


# Global singleton
file_cache_manager = FileCacheManager()


def add_to_cache(thread_id, filename, content, user_id):
    if content is None:
        content = ''
    file_cache_manager.add(thread_id, filename, content, user_id)


def load_cache_from_db(thread_id, user_id):
    file_cache_manager.load_from_db(thread_id, user_id)
    file_cache_manager.evict_oldest()
