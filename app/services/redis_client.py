"""Shared Redis client singleton — lazily initialized from REDIS_URL.

Usage:
    from app.services.redis_client import get_redis
    r = get_redis()
    r.ping()
"""
import os
import logging

_REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
_clients = {}

logger = logging.getLogger(__name__)


def get_redis(decode_responses: bool = True):
    key = "str" if decode_responses else "bytes"
    if key in _clients:
        return _clients[key]
    try:
        import redis
        client = redis.Redis.from_url(_REDIS_URL, decode_responses=decode_responses)
        client.ping()
        _clients[key] = client
        logger.info(f"Redis connected: {_REDIS_URL} (decode_responses={decode_responses})")
    except Exception as e:
        logger.warning(f"Redis unavailable at {_REDIS_URL}: {e}")
        return None
    return _clients[key]
