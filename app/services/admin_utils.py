"""Admin utility functions (auto-extracted)."""
import time, logging
from functools import wraps
from flask import request, jsonify
import app.globals as g
from app.database import get_db_connection

logger = logging.getLogger(__name__)

def admin_rate_limiter(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get client IP
        ip = request.remote_addr
        if ip is None:
            ip = 'unknown'
        now = time.time()
        # Clean old entries
        for key in list(g.admin_rate_limit.keys()):
            if g.admin_rate_limit[key]['timestamp'] < now - g.ADMIN_RATE_WINDOW:
                del g.admin_rate_limit[key]
        # Check limit
        key = f"{ip}:admin_action"
        if key in g.admin_rate_limit:
            if g.admin_rate_limit[key]['count'] >= g.ADMIN_RATE_LIMIT:
                logger.warning(f"Rate limit exceeded for admin action from IP {ip}")
                return jsonify({"error": "Too many attempts. Please try again later."}), 429
        else:
            g.admin_rate_limit[key] = {'count': 0, 'timestamp': now}
        # Increment count and call the function
        g.admin_rate_limit[key]['count'] += 1
        return f(*args, **kwargs)
    return decorated_function

def log_admin_action(admin_user_id, admin_username, action, table_name, row_id=None,
                     column_name=None, old_value=None, new_value=None,
                     success=True, error_message=None):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO admin_audit_log
                (admin_user_id, admin_username, action, table_name, row_id, column_name,
                 old_value, new_value, ip_address, success, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                admin_user_id, admin_username, action, table_name, row_id,
                column_name, old_value, new_value, request.remote_addr,
                success, error_message
            ))
            conn.commit()

