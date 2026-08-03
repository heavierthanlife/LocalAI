"""Batch ingest all existing KB files into the wiki.

Usage:
    python scripts/batch_wiki_ingest.py
    python scripts/batch_wiki_ingest.py --force  (re-ingest even if already indexed)
"""
import os, sys, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    from app import create_app
    app = create_app()
    with app.app_context():
        from app.services.wiki_ingest import batch_ingest_all
        from app.database import get_db_connection

        logger.info("Starting batch wiki ingest of all KB files...")
        with get_db_connection() as conn:
            result = batch_ingest_all(conn)
        logger.info(f"Batch ingest complete: {result}")

if __name__ == '__main__':
    main()
