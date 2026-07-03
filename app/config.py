"""Application configuration, paths, and constants."""
import os
import re
import sys
import io
import logging
from pathlib import Path
from logging.config import dictConfig

# ---------------- Base Directories ----------------
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
USER_FILES_DIR = DATA_DIR / "user_files"
PROJECT_FILES_DIR = DATA_DIR / "project_files"
CREDIT_REPORTS_DIR = DATA_DIR / "credit_reports"
DUMP_DIR = DATA_DIR / "dump"
SESSION_DIR = DATA_DIR / "flask_session"
TEMP_DIR = DATA_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, USER_FILES_DIR, PROJECT_FILES_DIR, CREDIT_REPORTS_DIR,
          DUMP_DIR, SESSION_DIR, TEMP_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TEMP_ROOT = str(TEMP_DIR)
USER_FILES_ORIGINAL_ROOT = str(USER_FILES_DIR)
PROJECT_FILES_ROOT = str(PROJECT_FILES_DIR)
CREDIT_REPORTS_DIR_STR = str(CREDIT_REPORTS_DIR)

# ---------------- Logging ----------------
LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'default': {'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'},
        'detailed': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'},
    },
    'handlers': {
        'console': {'class': 'logging.StreamHandler', 'level': 'INFO', 'formatter': 'default',
                    'stream': 'ext://sys.stdout'},
        'file': {'class': 'logging.handlers.RotatingFileHandler', 'level': 'DEBUG', 'formatter': 'detailed',
                 'filename': str(LOGS_DIR / 'app.log'), 'maxBytes': 10485760, 'backupCount': 5},
    },
    'root': {'level': 'DEBUG', 'handlers': ['console', 'file']},
}
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# ---------------- Edge Driver Path ----------------
EDGE_DRIVER_PATH: str | None = None  # intentionally reassigned below


def preinstall_edgedriver():
    global EDGE_DRIVER_PATH
    # Check env var first, then common locations, then auto-download
    env_path = os.environ.get('EDGEDRIVER_PATH')
    if env_path and os.path.exists(env_path):
        EDGE_DRIVER_PATH = env_path  # pyright: ignore[reportConstantRedefinition]
        logger.info(f"Edge WebDriver from env: {EDGE_DRIVER_PATH}")
        return
    # Try local msedgedriver.exe in project root
    local = os.path.join(BASE_DIR, "msedgedriver.exe")
    if os.path.exists(local):
        EDGE_DRIVER_PATH = local  # pyright: ignore[reportConstantRedefinition]
        logger.info(f"Edge WebDriver found at: {EDGE_DRIVER_PATH}")
        return
    # Fall back to auto-download via webdriver_manager
    try:
        from webdriver_manager.microsoft import EdgeChromiumDriverManager
        EDGE_DRIVER_PATH = EdgeChromiumDriverManager().install()  # pyright: ignore[reportConstantRedefinition]
        logger.info(f"Edge WebDriver auto-downloaded to: {EDGE_DRIVER_PATH}")
    except Exception as e:
        logger.warning(f"Could not pre-install Edge WebDriver: {e}. It will be downloaded on first use.")


# ---------------- File Type Validation ----------------
def is_valid_extracted_text(text: str, min_length: int = 20, min_ratio: float = 0.6) -> bool:
    if not text or len(text) < min_length:
        return False
    allowed = re.compile(r'[\u4e00-\u9fff\w\s.,;:!?()\-<>/{}[\]"\'=&#@+*|]')
    allowed_count = len(allowed.findall(text))
    ratio = allowed_count / len(text)
    return ratio >= min_ratio


ALLOWED_EXTENSIONS = {'.txt', '.md', '.text', '.csv', '.pdf', '.docx', '.docm', '.dotx', '.dotm', '.doc',
                      '.xlsx', '.xlsm', '.xltx', '.xltm', '.xlsb', '.xls', '.pptx', '.pptm', '.potx', '.ppsx',
                      '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.wps', '.et', '.dps', '.webp',
                      '.html', '.htm', '.json'}


def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS
