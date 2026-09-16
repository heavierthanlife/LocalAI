"""Boot-time seeding of repo data assets into the writable data volume.

In Docker, ``/app/data`` is a named volume (``app_data``) that shadows the image,
and ``.dockerignore`` excludes the repo's ``data/`` dir entirely. Immutable code
assets are bind-mounted read-only (``laws/``, ``industry_words/``,
``runtime_config_factory.json``). This module handles the remaining class:

  *mutable seed* files that must live in the volume (e.g. ``domain_words.txt``,
which ``approve_domain_words()`` appends to) — they are copied once from the
read-only repo mount when absent, then owned by the volume thereafter.

``ensure_seeded()`` is idempotent and cheap (a few ``os.path.exists`` calls), safe
to call from every entrypoint: gunicorn (``run:app``) and celery worker/beat all
go through ``create_app()`` (see ``celery_app.init_flask_context``).
"""
import logging
import os
import shutil

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

# Read-only bind mount of the repo's data/ dir (see docker-compose.yml).
REPO_DATA = os.environ.get('REPO_DATA_DIR', '/app/repo_data')

# Mutable seed assets: src (relative to REPO_DATA) -> dst (relative to DATA_DIR).
# Copied only when the destination is absent; thereafter the volume copy wins.
_SEED_FILES = {
    'domain_words.txt': 'domain_words.txt',
}


def ensure_seeded() -> None:
    """Copy missing mutable seed assets from REPO_DATA into DATA_DIR. Never raises."""
    try:
        if not os.path.isdir(REPO_DATA):
            return
        for src_rel, dst_rel in _SEED_FILES.items():
            src = os.path.join(REPO_DATA, src_rel)
            dst = os.path.join(str(DATA_DIR), dst_rel)
            if not os.path.exists(src) or os.path.exists(dst):
                continue
            parent = os.path.dirname(dst)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = dst + '.seedtmp'
            # Atomic publish: avoids partial files and tolerates multi-container races
            # (app/worker/beat all seed concurrently).
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            logger.info(f"bootstrap: seeded {dst_rel} from repo_data")
    except FileNotFoundError:
        # Another container (worker/beat) seeded first and removed the temp file.
        pass
    except Exception as e:  # never block application startup
        logger.warning(f"bootstrap: seeding skipped ({e})")
