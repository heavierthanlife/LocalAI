"""Shared DB save boilerplate — wraps save functions with connection, commit, error handling.

Usage:
    from app.services._save_helper import with_db_save

    @with_db_save(__name__)
    def save_my_results(cur, user_id, task_id, my_data):
        saved = 0
        for item in my_data:
            cur.execute("INSERT INTO ... VALUES (...)", (...))
            saved += 1
        return saved
"""
import logging
from functools import wraps


def with_db_save(logger_name: str):
    """Decorator: wraps a save function with DB connection, commit, and error handling.

    The wrapped function receives (cur, user_id, task_id, *args, **kwargs)
    and must return an int (number of rows saved).
    """
    log = logging.getLogger(logger_name)

    def decorator(fn):
        @wraps(fn)
        def wrapper(user_id: str, task_id: str, *args, **kwargs):
            from app.database import get_db_connection
            saved = 0
            try:
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        saved = fn(cur, user_id, task_id, *args, **kwargs)
                    conn.commit()
                log.info(f"Saved {saved} results for task {task_id}")
            except Exception as e:
                log.error(f"Failed to save results: {e}", exc_info=True)
            return saved
        return wrapper
    return decorator
