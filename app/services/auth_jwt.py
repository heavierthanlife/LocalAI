"""JWT token issuance for API access — WeChat Enterprise / external integrations.

Only token creation is in use (app/routes/chat_sessions.py). Token-verification
decorators were removed in FIX-055 (JWT was issued but never validated; API auth
is session-based).
"""
import os
import time

import jwt

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
