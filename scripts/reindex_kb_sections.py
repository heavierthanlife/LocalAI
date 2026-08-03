#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild RAG indexes for KB files with structured section-aware chunking.

Parses markdown content for ## headings to build section structures.
Incremental by file content hash — skips unchanged files.

Usage:
    python scripts/reindex_kb_sections.py                            # Full run
    python scripts/reindex_kb_sections.py --dry-run                  # Preview only
    python scripts/reindex_kb_sections.py --resume                   # Skip already-indexed
    python scripts/reindex_kb_sections.py --sources knowledge_lab    # Single source
"""

import argparse
import json
import os
import re
import sys
import hashlib
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_db_connection
from app.services.rag_engine import index_file


def compute_content_hash(content):
    return hashlib.sha256((content or "")[:10000].encode("utf-8", errors="replace")).hexdigest()[:16]


_MD_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')


def _parse_markdown_sections(content: str) -> list:
    """Parse markdown content for heading-based section structure.

    Returns list of {id, title, content, level} dicts compatible with
    the index_file(sections=) parameter.

    Section ID format: "3" (level 1 chapter), "3-15" (level 2 article), "0-N" (other).
    """
    if not content:
        return []

    lines = content.split('\n')
    sections = []
    ch = 0
    art = 0
    other = 0

    for line in lines:
        m = _MD_HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if not title:
                continue
            if level == 1:
                ch += 1
                art = 0
                sid = str(ch)
            elif level == 2:
                art += 1
                sid = f"{ch}-{art}"
            else:
                other += 1
                sid = f"0-{other}"
            sections.append({
                "id": sid,
                "title": title,
                "content": "",
                "level": level,
            })
        elif sections:
            sections[-1]["content"] += line + "\n"

    # Filter empty-content sections — no value in embedding empty headers
    sections = [s for s in sections if s["content"].strip()]

    return sections


def reindex_table(table_name, source_name, id_col, content_col,
                  user_id_col=None, dry_run=False, resume=False):
    """Rebuild indexes for one KB source table.

    Returns stats dict: {total, skipped, rebuilt, failed}.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            where = f"WHERE {user_id_col} IS NOT NULL" if user_id_col else ""
            cur.execute(
                f"SELECT {id_col}, {content_col}, original_name"
                f" FROM {table_name} {where}"
                f" ORDER BY {id_col}"
            )
            rows = cur.fetchall()

    stats = {"total": len(rows), "skipped": 0, "rebuilt": 0, "failed": 0}
    label = f"[{source_name}]"
    if dry_run:
        label += " (dry-run)"

    sys.stderr.write(f"\n{label} Scanning {len(rows)} files in {table_name}...\n")

    for row in rows:
        fid, content, original_name = row
        current_hash = compute_content_hash(content)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT last_indexed_hash FROM {table_name} WHERE {id_col} = %s",
                    (fid,),
                )
                stored = cur.fetchone()
                last_hash = stored[0] if stored else None

        if current_hash and last_hash and current_hash == last_hash:
            stats["skipped"] += 1
            continue

        if resume and last_hash:
            stats["skipped"] += 1
            continue

        try:
            sections = _parse_markdown_sections(content or "")
            if not sections:
                sections = None

            if dry_run:
                status = f"sections={len(sections)}" if sections else "flat"
                stats["rebuilt"] += 1
                sys.stderr.write(f"  {label} [DRY] id={fid} ({original_name}) → {status}\n")
                continue

            index_file(
                fid, content, source_name,
                metadata={"original_name": original_name or "", "category": ""},
                sections=sections,
                force=True,
            )

            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE {table_name} SET last_indexed_hash = %s WHERE {id_col} = %s",
                        (current_hash, fid),
                    )
                    conn.commit()

            stats["rebuilt"] += 1
            if stats["rebuilt"] % 5 == 0:
                sys.stderr.write(
                    f"  {label} {stats['rebuilt']} rebuilt, {stats['skipped']} skipped...\n"
                )

        except Exception as e:
            stats["failed"] += 1
            sys.stderr.write(f"  {label} FAILED file_id={fid} ({original_name}): {e}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Rebuild KB RAG indexes with section-aware chunking")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be reindexed without touching ChromaDB")
    parser.add_argument("--resume", action="store_true", help="Skip files that already have a last_indexed_hash")
    parser.add_argument("--sources", nargs="+", default=["knowledge_lab", "company_kb"],
                        choices=["knowledge_lab", "company_kb"],
                        help="Which KB sources to reindex (default: both)")
    args = parser.parse_args()

    start_time = time.time()

    sources = {
        "knowledge_lab": {
            "table": "knowledge_lab_files",
            "id_col": "id",
            "content_col": "content",
            "user_id_col": "user_id",
        },
        "company_kb": {
            "table": "company_knowledge_base",
            "id_col": "id",
            "content_col": "content",
            "user_id_col": None,
        },
    }

    if args.dry_run:
        sys.stderr.write("=" * 60 + "\n")
        sys.stderr.write("KB Section-Aware Reindex (DRY RUN)\n")
        sys.stderr.write("=" * 60 + "\n")
    else:
        sys.stderr.write("=" * 60 + "\n")
        sys.stderr.write("KB Section-Aware Reindex\n")
        sys.stderr.write("=" * 60 + "\n")

    results = {}
    for source_name in args.sources:
        cfg = sources[source_name]
        stats = reindex_table(
            table_name=cfg["table"],
            source_name=source_name,
            id_col=cfg["id_col"],
            content_col=cfg["content_col"],
            user_id_col=cfg.get("user_id_col"),
            dry_run=args.dry_run,
            resume=args.resume,
        )
        results[source_name] = stats

    elapsed = round(time.time() - start_time, 1)

    sys.stderr.write("\n" + "=" * 60 + "\n")
    sys.stderr.write("Summary\n")
    sys.stderr.write("=" * 60 + "\n")
    for source_name, s in results.items():
        status = " (dry-run)" if args.dry_run else ""
        sys.stderr.write(
            f"  [{source_name}]{status} total={s['total']}"
            f" skipped={s['skipped']} rebuilt={s['rebuilt']} failed={s['failed']}\n"
        )
    sys.stderr.write(f"  Elapsed: {elapsed}s\n")

    summary = {
        "dry_run": args.dry_run,
        "resume": args.resume,
        "elapsed_seconds": elapsed,
        "sources": {
            name: stats for name, stats in results.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
