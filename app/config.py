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
                 'filename': str(BASE_DIR / 'app.log'), 'maxBytes': 10485760, 'backupCount': 5},
    },
    'root': {'level': 'DEBUG', 'handlers': ['console', 'file']},
}
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# ---------------- Edge Driver Path ----------------
EDGE_DRIVER_PATH = None


def preinstall_edgedriver():
    global EDGE_DRIVER_PATH
    try:
        EDGE_DRIVER_PATH = r"D:\PyCharm\Local_AI\msedgedriver.exe"
        logger.info(f"Edge WebDriver pre-installed at {EDGE_DRIVER_PATH}")
    except Exception as e:
        logger.error(f"Failed to pre-install Edge WebDriver: {e}")


# ---------------- File Type Validation ----------------
def is_valid_extracted_text(text, min_length=20, min_ratio=0.6):
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


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS
