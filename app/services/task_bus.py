"""Generic async task progress bus via Redis pub/sub.

Architecture:
    Celery worker ──publish──▶ Redis pub/sub channel ──subscribe──▶ Flask SSE endpoint ──stream──▶ Browser

    Task metadata lives in Redis hash:  task_meta:{task_id}
    Live progress flows via pub/sub:     task_progress:{task_id}

Usage from any Celery task:
    from app.services.task_bus import TaskBus
    bus = TaskBus()
    bus.start('doc_analysis', '分析文档')
    bus.progress(50, '正在解析第5页/共10页')
    bus.complete(result={'summary': '...'})
    bus.fail('处理超时')
"""
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

_redis = None


def _get_redis():
    """Lazy Redis connection (shared across Flask + Celery)."""
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis
        _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        _redis.ping()
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}), task bus disabled")
        _redis = False
    return _redis


# ── Status constants ──
STATUS_QUEUED = 'queued'
STATUS_RUNNING = 'running'
STATUS_COMPLETED = 'completed'
STATUS_FAILED = 'failed'

# Redis key prefixes
META_PREFIX = 'task_meta:'
PROGRESS_CHANNEL = 'task_progress:'
TASK_REGISTRY = 'task_registry'  # Sorted Set: {score=created_ts, member=task_id}

# Keep metadata for 7 days
META_TTL = 7 * 86400


class TaskBus:
    """Publish progress updates from Celery worker to Redis pub/sub."""

    def __init__(self, task_id: str = None, task_type: str = None, label: str = None):
        self.task_id = task_id
        self.task_type = task_type or 'unknown'
        self.label = label or task_id or 'unknown'

    # ── Publish side (called from Celery worker) ──

    def register_queued(self, extra: dict = None):
        """Synchronously write task_meta as ``queued`` so status/SSE endpoints
        never 404 in the window between the task being enqueued and the worker
        actually starting it (``start()`` then upgrades it to ``running``).

        Called from the Flask route (e.g. /clearance/run) before
        ``celery.send_task`` so ``TaskBus.get()`` always finds the task.
        """
        r = _get_redis()
        if not r:
            return
        now = datetime.now(timezone.utc).isoformat()
        meta = {
            'status': STATUS_QUEUED,
            'type': self.task_type,
            'label': self.label,
            'progress': 0,
            'message': '排队中...',
            'created_at': now,
            'started_at': now,
            'result': '',
        }
        if extra:
            meta.update({k: v for k, v in extra.items() if v is not None})
        pipe = r.pipeline()
        pipe.hset(META_PREFIX + self.task_id, mapping=meta)
        pipe.expire(META_PREFIX + self.task_id, META_TTL)
        pipe.zadd(TASK_REGISTRY, {self.task_id: time.time()})
        pipe.execute()

    def start(self, extra: dict = None):
        """Mark task as running, publish initial metadata.

        ``extra`` (optional) adds arbitrary fields to the Redis hash metadata
        (e.g. ``thread_id`` for the origin chat thread). Redis hashes accept
        new fields freely — no schema migration needed.
        """
        r = _get_redis()
        if not r:
            return
        now = datetime.now(timezone.utc).isoformat()
        meta = {
            'status': STATUS_RUNNING,
            'type': self.task_type,
            'label': self.label,
            'progress': 0,
            'message': '开始处理...',
            'created_at': now,
            'started_at': now,
            'result': '',
        }
        if extra:
            meta.update({k: v for k, v in extra.items() if v is not None})
        pipe = r.pipeline()
        pipe.hset(META_PREFIX + self.task_id, mapping=meta)
        pipe.expire(META_PREFIX + self.task_id, META_TTL)
        pipe.zadd(TASK_REGISTRY, {self.task_id: time.time()})
        pipe.execute()
        self._emit('start', {'progress': 0, 'message': '开始处理...'})

    def progress(self, percent: int, message: str = ''):
        """Update progress 0–100, with optional message."""
        r = _get_redis()
        if not r:
            return
        percent = max(0, min(100, int(percent)))
        pipe = r.pipeline()
        pipe.hset(META_PREFIX + self.task_id, mapping={
            'progress': percent,
            'message': message or f'{percent}%',
        })
        pipe.expire(META_PREFIX + self.task_id, META_TTL)
        pipe.execute()
        self._emit('progress', {'progress': percent, 'message': message or f'{percent}%'})

    def complete(self, result=None):
        """Mark task as completed."""
        r = _get_redis()
        if not r:
            return
        now = datetime.now(timezone.utc).isoformat()
        pipe = r.pipeline()
        pipe.hset(META_PREFIX + self.task_id, mapping={
            'status': STATUS_COMPLETED,
            'progress': 100,
            'message': '完成',
            'completed_at': now,
            'result': json.dumps(result, ensure_ascii=False) if result else '',
        })
        pipe.expire(META_PREFIX + self.task_id, META_TTL)
        pipe.execute()
        self._emit('complete', {'progress': 100, 'message': '完成', 'result': result})

    def fail(self, error: str):
        """Mark task as failed with error message."""
        r = _get_redis()
        if not r:
            return
        now = datetime.now(timezone.utc).isoformat()
        pipe = r.pipeline()
        pipe.hset(META_PREFIX + self.task_id, mapping={
            'status': STATUS_FAILED,
            'message': error[:500],
            'failed_at': now,
        })
        pipe.expire(META_PREFIX + self.task_id, META_TTL)
        pipe.execute()
        self._emit('error', {'progress': -1, 'message': error[:500]})

    def _emit(self, event_type: str, data: dict):
        """Publish an event to the task's pub/sub channel."""
        r = _get_redis()
        if not r:
            return
        payload = json.dumps({'event': event_type, **data}, ensure_ascii=False)
        r.publish(PROGRESS_CHANNEL + self.task_id, payload)

    # ── Query side (called from Flask routes) ──

    @staticmethod
    def get(task_id: str) -> Optional[dict]:
        """Get task metadata as dict, or None if not found."""
        r = _get_redis()
        if not r:
            return None
        data = r.hgetall(META_PREFIX + task_id)
        if not data:
            return None
        data['progress'] = int(data.get('progress', 0))
        return data

    @staticmethod
    def list_tasks(limit: int = 50) -> list[dict]:
        """List recent tasks (newest first)."""
        r = _get_redis()
        if not r:
            return []
        task_ids = r.zrevrange(TASK_REGISTRY, 0, limit - 1)
        tasks = []
        for tid in task_ids:
            meta = r.hgetall(META_PREFIX + tid)
            if meta:
                meta['task_id'] = tid
                meta['progress'] = int(meta.get('progress', 0))
                tasks.append(meta)
        return tasks

    @staticmethod
    def delete(task_id: str) -> bool:
        """Remove a task from the registry and drop its metadata hash.

        Returns True if the task existed (meta removed), False otherwise.
        """
        r = _get_redis()
        if not r:
            return False
        pipe = r.pipeline()
        pipe.zrem(TASK_REGISTRY, task_id)
        pipe.delete(META_PREFIX + task_id)
        pipe.execute()
        existed = r.exists(META_PREFIX + task_id) == 0
        return True

    @staticmethod
    def subscribe(task_id: str, timeout: float = 300):
        """Generator: yield SSE-formatted progress events from Redis pub/sub.

        Blocks until task completes/fails or timeout. Yields strings ready for
        Flask Response with mimetype='text/event-stream'.

        Usage:
            return Response(
                TaskBus.subscribe(task_id),
                mimetype='text/event-stream',
                headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'}
            )
        """
        r = _get_redis()
        if not r:
            yield f"event: error\ndata: {json.dumps({'message': 'Redis unavailable'})}\n\n"
            return

        pubsub = r.pubsub()
        channel = PROGRESS_CHANNEL + task_id
        pubsub.subscribe(channel)

        # First, check if task already completed
        meta = TaskBus.get(task_id)
        if meta and meta.get('status') in (STATUS_COMPLETED, STATUS_FAILED):
            event_type = 'complete' if meta['status'] == STATUS_COMPLETED else 'error'
            payload = json.dumps({k: v for k, v in meta.items() if k != 'result'}, ensure_ascii=False)
            yield f"event: {event_type}\ndata: {payload}\n\n"
            pubsub.unsubscribe(channel)
            return

        deadline = time.time() + timeout
        try:
            while time.time() < deadline:
                msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg['type'] == 'message':
                    yield f"data: {msg['data']}\n\n"

                    # Check if terminal event
                    try:
                        data = json.loads(msg['data'])
                        if data.get('event') in ('complete', 'error'):
                            break
                    except json.JSONDecodeError:
                        pass

                # Heartbeat to keep connection alive
                yield ": heartbeat\n\n"

                # Check Redis metadata as fallback (in case pub/sub message lost)
                meta = TaskBus.get(task_id)
                if meta and meta.get('status') in (STATUS_COMPLETED, STATUS_FAILED):
                    event_type = 'complete' if meta['status'] == STATUS_COMPLETED else 'error'
                    payload = json.dumps({k: v for k, v in meta.items() if k != 'result'}, ensure_ascii=False)
                    yield f"event: {event_type}\ndata: {payload}\n\n"
                    break

        finally:
            pubsub.unsubscribe(channel)
