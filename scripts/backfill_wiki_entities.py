#!/usr/bin/env python
"""Backfill entity extraction for all existing uploaded files.

Usage:
    python scripts/backfill_wiki_entities.py [--batch-size N] [--delay S] [--max-count N] [--dry-run]

Processes all existing uploaded files (knowledge_lab, company_kb, project_files)
and runs entity extraction on each. Files already extracted are skipped by hash.

Rate limiting:
    --batch-size N   Process N files per batch (default 10)
    --delay S         Seconds between batches (default 6, ≈ 10 RPM)
    --max-count N     Maximum files to process (default 0 = all)
    --dry-run         Show what would be processed without doing it

The script respects the LLM provider rate limits. At 10 RPM, 1000 files
takes approximately 100 minutes with default settings.
"""

import argparse
import hashlib
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill_wiki_entities")


def init_app():
    from app import create_app
    from app.database import get_db_connection
    app = create_app()
    app.app_context().push()
    return get_db_connection


def query_files(db_conn, max_count=0):
    from psycopg2.extras import RealDictCursor

    tables = [
        ("knowledge_lab", "knowledge_lab_files", "id, content, original_name"),
        ("company_kb", "company_knowledge_base", "id, content, original_name"),
        ("project_files", "project_files", "id, content, original_name"),
    ]

    all_rows = []
    with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
        for source_type, table, cols in tables:
            try:
                cur.execute(
                    f"SELECT {cols} FROM {table} WHERE content IS NOT NULL AND content != ''"
                )
                rows = cur.fetchall()
                for r in rows:
                    r["source_type"] = source_type
                    r["table"] = table
                all_rows.extend(rows)
                logger.info(f"Found {len(rows)} files in {table}")
            except Exception as e:
                logger.warning(f"Failed to query {table}: {e}")

    if max_count and max_count > 0:
        all_rows = all_rows[:max_count]

    return all_rows


def process_file(db_conn, row, processed_hashes):
    from app.services.document_classifier import classify_and_categorize
    from app.services.wiki_entity_service import process_upload_entity_extraction

    file_id = row["id"]
    content = row["content"]
    original_name = row.get("original_name", "")
    source_type = row["source_type"]

    if not content or len(content) < 50:
        return {"status": "skipped", "reason": "content too short"}

    content_hash = hashlib.sha256(
        (content[:5000] or "").encode("utf-8", errors="replace")
    ).hexdigest()
    if content_hash in processed_hashes:
        return {"status": "skipped", "reason": "content hash already processed"}

    processed_hashes.add(content_hash)

    try:
        doc_type, wiki_category = classify_and_categorize(
            content, original_name or "", ""
        )
        result = process_upload_entity_extraction(
            file_id=file_id,
            text_content=content,
            filename=original_name or f"file_{file_id}",
            source_type=source_type,
            doc_type=doc_type,
            wiki_category=wiki_category,
            metadata={"original_name": original_name or ""},
        )
        return {"status": "ok", **result}
    except Exception as e:
        logger.warning(f"Failed to process {source_type}.{file_id}: {e}")
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Backfill wiki entity extraction")
    parser.add_argument("--batch-size", type=int, default=10, help="Files per batch")
    parser.add_argument("--delay", type=float, default=6.0, help="Seconds between batches")
    parser.add_argument("--max-count", type=int, default=0, help="Max files (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    db_conn = init_app()
    rows = query_files(db_conn, max_count=args.max_count)

    if args.dry_run:
        logger.info(f"DRY RUN: would process {len(rows)} files")
        for r in rows[:20]:
            logger.info(f"  - {r['source_type']}.{r['id']}: {r.get('original_name', '')}")
        if len(rows) > 20:
            logger.info(f"  ... and {len(rows) - 20} more")
        return

    logger.info(f"Processing {len(rows)} files (batch_size={args.batch_size}, delay={args.delay}s)")

    processed_hashes = set()
    stats = {"total": len(rows), "ok": 0, "skipped": 0, "errors": 0}
    start_time = time.time()

    for i, row in enumerate(rows):
        if i > 0 and i % args.batch_size == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Progress: {i}/{len(rows)} ({stats['ok']} ok, {stats['skipped']} skipped, "
                f"{stats['errors']} errors, {elapsed:.0f}s elapsed)"
            )
            time.sleep(args.delay)

        result = process_file(db_conn, row, processed_hashes)
        if result["status"] == "ok":
            stats["ok"] += 1
        elif result["status"] == "skipped":
            stats["skipped"] += 1
        else:
            stats["errors"] += 1

    elapsed = time.time() - start_time
    logger.info(
        f"Backfill complete: {stats['ok']} ok, {stats['skipped']} skipped, "
        f"{stats['errors']} errors in {elapsed:.0f}s"
    )

    db_conn.close()


if __name__ == "__main__":
    main()
