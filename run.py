"""Entry point for the AI_Services application.

Usage:
    python run.py          # HTTPS on :5443 + HTTP→HTTPS redirect on :5000
"""
import io
import logging
import os
import sys
import warnings
import ssl
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── Fix Windows GBK encoding for emoji in logs (MUST be before any logging) ──
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
# Patch root logger's existing handlers too
for _h in logging.root.handlers:
    if hasattr(_h, 'stream') and hasattr(_h.stream, 'buffer'):
        try:
            _h.stream = io.TextIOWrapper(_h.stream.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass  # stream already closed or invalid

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress Pydantic V1 deprecation spam on Python 3.14+
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
os.environ["PYDANTIC_V1_COMPAT"] = "true"



from app import create_app, init_services

# Initialize services (DB tables, drivers, models)
init_services()

# Create the Flask application
app = create_app()

# ── One-time cleanup: remove ghost empty chats (harmless, no messages) ──
try:
    from app.database import get_db_connection
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_sessions WHERE project_id IS NULL "
                "AND thread_id NOT IN (SELECT DISTINCT thread_id FROM chat_messages)"
            )
            if cur.rowcount:
                import logging
                logging.getLogger(__name__).info(f"Cleaned up {cur.rowcount} ghost empty chats")
            conn.commit()
except Exception:
    pass  # Non-critical

# ── Prometheus Metrics (1-line activation) ──
# Exposes /metrics endpoint for Prometheus scraping.
# Also adds automatic request count, latency, and error rate metrics.
try:
    from prometheus_flask_exporter import PrometheusMetrics
    metrics = PrometheusMetrics(app)
    # Custom business metric: LLM call counter
    metrics.info('localai_info', 'Local AI application', version='2026.06')
except ImportError:
    pass  # Optional dependency

# ── SSL context ──
CERT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cert')
CERT_FILE = os.path.join(CERT_DIR, 'cert.pem')
KEY_FILE = os.path.join(CERT_DIR, 'key.pem')

if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(CERT_FILE, KEY_FILE)
else:
    print("[WARN] cert/cert.pem or cert/key.pem missing — falling back to adhoc SSL")
    ssl_context = 'adhoc'


# ── Tiny HTTP→HTTPS redirector ──
class RedirectHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(301)
        host = self.headers.get('Host', 'localhost').split(':')[0]
        self.send_header('Location', f'https://{host}:5443{self.path}')
        self.end_headers()
    def do_POST(self):
        self.send_response(307)
        host = self.headers.get('Host', 'localhost').split(':')[0]
        self.send_header('Location', f'https://{host}:5443{self.path}')
        self.end_headers()
    def log_message(self, fmt, *args):
        pass  # silent


def start_redirect():
    httpd = HTTPServer(('0.0.0.0', 5000), RedirectHandler)
    httpd.serve_forever()


if __name__ == '__main__':
    # Start HTTP→HTTPS redirect in a background thread
    t = threading.Thread(target=start_redirect, daemon=True)
    t.start()
    # Run Flask with HTTPS
    app.run(host='0.0.0.0', port=5443, ssl_context=ssl_context, threaded=True)
