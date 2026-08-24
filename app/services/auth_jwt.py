"""JWT authentication for API access — WeChat Enterprise / external integrations.

Supports BOTH session cookies (web UI) and JWT tokens (API).
Sessions take priority; if no session exists, falls back to JWT.
"""
import os
import time
import logging
from functools import wraps
from flask import request, jsonify, session

import jwt

logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'change-me-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRY_HOURS = int(os.getenv('JWT_EXPIRY_HOURS', '24'))  # 24h default
JWT_ISSUER = os.getenv('JWT_ISSUER', 'local-ai')


def create_token(user_id: str, username: str, role: str) -> str:
    """Create a JWT access token."""
    now = int(time.time())
    payload = {
        'sub': user_id,
        'username': username,
        'role': role,
        'iat': now,
        'exp': now + JWT_EXPIRY_HOURS * 3600,
        'iss': JWT_ISSUER,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    """Decode and validate a JWT token. Returns payload dict or None."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM],
                         issuer=JWT_ISSUER, options={'require': ['exp', 'iss', 'sub']})
    except jwt.ExpiredSignatureError:
        logger.debug("JWT expired")
    except jwt.InvalidTokenError as e:
        logger.debug(f"JWT invalid: {e}")
    return None


def jwt_required(f):
    """Decorator: require valid session OR JWT token.

    Usage: @jwt_required
           def my_api_endpoint():
               user_id = request.user_id  # injected by this decorator
               ...
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # 1. Check session cookie (web UI)
        user_id = session.get('user_id')
        if user_id and session.get('consent_value', 0) == 1:
            request.user_id = user_id
            request.username = session.get('username', '')
            request.user_role = session.get('role', 'user')
            return f(*args, **kwargs)

        # 2. Check JWT token (API)
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            payload = decode_token(token)
            if payload:
                request.user_id = payload['sub']
                request.username = payload.get('username', '')
                request.user_role = payload.get('role', 'user')
                return f(*args, **kwargs)

        return jsonify({"error": "Authentication required. Use Bearer token or login."}), 401
    return decorated


def jwt_optional(f):
    """Decorator: extract user identity if available, but don't block if missing."""
    @wraps(f)
    def decorated(*args, **kwargs):
        request.user_id = None
        request.username = ''
        request.user_role = 'anonymous'

        # Session
        uid = session.get('user_id')
        if uid and session.get('consent_value', 0) == 1:
            request.user_id = uid
            request.username = session.get('username', '')
            request.user_role = session.get('role', 'user')
            return f(*args, **kwargs)

        # JWT
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            payload = decode_token(auth_header[7:])
            if payload:
                request.user_id = payload['sub']
                request.username = payload.get('username', '')
                request.user_role = payload.get('role', 'user')
                return f(*args, **kwargs)

        return f(*args, **kwargs)
    return decorated
