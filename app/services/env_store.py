"""Environment-style secret store for admin-added LLM provider API keys.

Keys are persisted as ``KEY=value`` lines to a file under ``DATA_DIR`` (which
survives Docker container recreation via the ``app_data`` volume) and, when a
root ``.env`` exists and is writable (local dev), dual-written there too.
Every write also updates ``os.environ`` so the running process picks the value
up without a restart.

Secrets are write-only: never echo raw values back to clients.
"""
import os
import re
import logging
from pathlib import Path

from app.config import BASE_DIR, DATA_DIR

logger = logging.getLogger(__name__)

# Primary, persistent store (Docker: /app/data is a named volume).
PROVIDER_KEYS_PATH = DATA_DIR / 'llm_provider_keys.env'
# Local-dev convenience target (only written when the file already exists).
ROOT_ENV_PATH = BASE_DIR / '.env'

_KEY_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=')


def _target_paths(paths=None):
    if paths is not None:
        return [Path(p) for p in paths]
    targets = [PROVIDER_KEYS_PATH]
    if ROOT_ENV_PATH.exists():
        targets.append(ROOT_ENV_PATH)
    return targets


def _format_line(key, value):
    v = '' if value is None else str(value)
    # Defensive: keys never contain newlines; strip to avoid multi-line values.
    v = v.replace('\r', '').replace('\n', '')
    if v == '' or re.search(r'''[\s#\'"$`\\]''', v):
        if "'" not in v:
            # Single-quoted dotenv values are literal — no $-expansion, no escapes.
            v = "'" + v + "'"
        else:
            # Fall back to double quotes, escaping shell-expansion chars.
            v = '"' + v.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$') + '"'
    return f'{key}={v}\n'


def _upsert(path, key, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if path.exists():
        try:
            lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
        except Exception as e:
            # Do NOT clobber a file we cannot read (e.g. non-UTF-8 authored .env).
            raise RuntimeError(f'cannot read {path}: {e}')
    out = []
    replaced = False
    for ln in lines:
        m = _KEY_RE.match(ln.lstrip('\ufeff'))
        if m and m.group(1) == key:
            out.append(_format_line(key, value))
            replaced = True
        else:
            out.append(ln)
    if not replaced:
        if out and not out[-1].endswith('\n'):
            out[-1] += '\n'
        out.append(_format_line(key, value))
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(''.join(out), encoding='utf-8')
    os.replace(tmp, path)


def write_env_var(key, value, paths=None):
    """Upsert ``key=value`` into the provider env store (+ root .env if present),
    then set ``os.environ[key]``. Returns the list of files written."""
    key = (key or '').strip()
    if not _KEY_RE.match(key + '='):
        raise ValueError(f'invalid env key: {key!r}')
    written = []
    for p in _target_paths(paths):
        try:
            _upsert(p, key, value)
            written.append(str(p))
        except Exception as e:
            logger.warning(f'env_store: write {p} failed: {e}')
    os.environ[key] = '' if value is None else str(value)
    return written


def has_env_var(key) -> bool:
    return bool((get_env(key) or '').strip())


_loaded_mtime = None


def _ensure_loaded():
    """Load the persistent provider env file into os.environ if it changed
    (cheap mtime check). Covers multi-worker setups (gunicorn) where only the
    worker that handled the save has the var in its own ``os.environ``."""
    global _loaded_mtime
    try:
        mtime = PROVIDER_KEYS_PATH.stat().st_mtime if PROVIDER_KEYS_PATH.exists() else None
    except OSError:
        mtime = None
    if mtime is None or mtime == _loaded_mtime:
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(str(PROVIDER_KEYS_PATH), override=True)
        _loaded_mtime = mtime
    except Exception as e:
        logger.warning(f'env_store: reload provider keys failed: {e}')


def get_env(key, default=None):
    """Read an env var, lazily reloading the persistent provider file so
    admin-added keys are visible to every worker process."""
    key = key or ''
    val = os.getenv(key)
    if val:
        return val
    _ensure_loaded()
    return os.getenv(key, default)


def load_provider_keys():
    """Load the persistent provider env file into os.environ (no override)."""
    try:
        from dotenv import load_dotenv
        if PROVIDER_KEYS_PATH.exists():
            load_dotenv(str(PROVIDER_KEYS_PATH), override=False)
    except Exception as e:
        logger.warning(f'env_store: load provider keys failed: {e}')
