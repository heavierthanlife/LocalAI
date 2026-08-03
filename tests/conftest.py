"""Root conftest: unit-test fixtures, no DB/network needed.
"""
import os
import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest")
os.environ.setdefault("WTF_CSRF_ENABLED", "false")
os.environ.setdefault("ENABLE_SCHEDULER", "false")
os.environ.setdefault("RATELIMIT_STORAGE_URL", "memory://")
os.environ.setdefault("LOG_LEVEL", "CRITICAL")
os.environ.setdefault("PG_USER", "postgres")
os.environ.setdefault("PG_PASSWORD", "")
os.environ.setdefault("PG_DB", "test_chatbot_test")
os.environ.setdefault("PG_HOST", "localhost")
os.environ.setdefault("PG_PORT", "5432")


def pytest_configure(config):
    if hasattr(config.option, "capture") and config.option.capture and config.option.capture != "no":
        config.option.capture = "no"


@pytest.fixture(scope="session")
def app():
    from app import create_app
    import sys, io
    _old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    _app = create_app()
    sys.stdout = _old_stdout
    _app.config.update(
        TESTING=True,
        PRESERVE_CONTEXT_ON_EXCEPTION=False,
        SERVER_NAME="localhost.localdomain",
        WTF_CSRF_ENABLED=False,
    )
    # Override .env ADMIN_PIN with known test value
    from werkzeug.security import generate_password_hash
    _app.config["ADMIN_PASSWORD_HASH"] = generate_password_hash("123456")
    yield _app


@pytest.fixture
def app_context(app):
    with app.app_context():
        yield


@pytest.fixture
def request_context(app):
    with app.test_request_context():
        yield


@pytest.fixture(autouse=True)
def mock_redis(monkeypatch):
    import fakeredis
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("app.services.redis_client.get_redis", lambda **kw: fake)
    monkeypatch.setattr("app.services.redis_client._clients", {"str": fake})
    return fake
