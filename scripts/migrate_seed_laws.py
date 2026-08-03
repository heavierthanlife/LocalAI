#!/usr/bin/env python3
"""U1: Migrate seed laws from JSON → PostgreSQL law library tables.

Usage:
    python scripts/migrate_seed_laws.py            # full migration
    python scripts/migrate_seed_laws.py --dry-run  # preview only
    python scripts/migrate_seed_laws.py --reset    # delete all laws and re-import
"""

import json
import os
import sys
import shutil
import argparse
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.database import get_db_connection, init_postgres_tables
from psycopg2.extras import RealDictCursor

LAWS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'laws', 'extended_laws.json')
BACKUP_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'laws', 'backup')


def ensure_tables():
    """Ensure DB tables exist."""
    init_postgres_tables()
    print("[OK] Database tables initialized.")


def backup_seed_json():
    """Backup seed_laws.json before migration."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    src = os.path.join(os.path.dirname(LAWS_FILE), 'seed_laws.json')
    if os.path.exists(src):
        dst = os.path.join(BACKUP_DIR, f'seed_laws_{date.today().isoformat()}.json')
        shutil.copy2(src, dst)
        print(f"[OK] Backed up seed_laws.json → {dst}")


def load_extended_laws():
    """Load extended laws JSON."""
    if not os.path.exists(LAWS_FILE):
        print(f"[ERROR] Extended laws file not found: {LAWS_FILE}")
        sys.exit(1)
    with open(LAWS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def migrate(dry_run=False):
    """Import laws into PostgreSQL."""
    laws = load_extended_laws()
    print(f"Found {len(laws)} laws in extended_laws.json")

    if dry_run:
        print("\n── DRY RUN ──")
        for law in laws:
            versions = law.get("versions", [])
            total_articles = sum(len(v.get("articles", [])) for v in versions)
            print(f"  {law['law_name']}: {len(versions)} version(s), {total_articles} article(s)")
        print(f"\nTotal: {len(laws)} laws, {sum(len(v.get('articles',[])) for l in laws for v in l.get('versions',[]))} articles")
        return

    backup_seed_json()

    inserted_laws = 0
    inserted_versions = 0
    inserted_articles = 0

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for law in laws:
                law_name = law["law_name"]

                # Check if law already exists
                cur.execute("SELECT id FROM law_masters WHERE law_name = %s", (law_name,))
                existing = cur.fetchone()
                if existing:
                    law_id = existing["id"]
                    print(f"  [SKIP] {law_name} already exists (id={law_id})")
                else:
                    cur.execute("""
                        INSERT INTO law_masters (law_name, short_name, category, issuing_authority,
                                                 effective_date, expiry_date, status, scope)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                        RETURNING id
                    """, (
                        law_name,
                        law.get("short_name", law_name),
                        law.get("category", ""),
                        law.get("issuing_authority"),
                        law.get("effective_date"),
                        law.get("expiry_date"),
                        law.get("status", "active"),
                        law.get("scope", "national"),
                    ))
                    law_id = cur.fetchone()["id"]
                    print(f"  [NEW] {law_name} (id={law_id})")
                    inserted_laws += 1

                # Insert versions
                for version in law.get("versions", []):
                    cur.execute("""
                        SELECT id FROM law_versions
                        WHERE law_id = %s AND version_label = %s
                    """, (law_id, version["version_label"]))
                    existing_ver = cur.fetchone()
                    if existing_ver:
                        version_id = existing_ver["id"]
                        print(f"    [SKIP] version '{version['version_label']}' exists")
                    else:
                        # Unset any previous current version
                        if version.get("is_current"):
                            cur.execute("UPDATE law_versions SET is_current = FALSE WHERE law_id = %s", (law_id,))

                        cur.execute("""
                            INSERT INTO law_versions (law_id, version_label, version_date, is_current, change_summary)
                            VALUES (%s,%s,%s,%s,%s)
                            RETURNING id
                        """, (
                            law_id,
                            version["version_label"],
                            law.get("effective_date"),
                            version.get("is_current", False),
                            version.get("change_summary"),
                        ))
                        version_id = cur.fetchone()["id"]
                        print(f"    [NEW] version '{version['version_label']}' (id={version_id})")
                        inserted_versions += 1

                    # Insert articles
                    for i, article in enumerate(version.get("articles", [])):
                        cur.execute("""
                            SELECT id FROM law_articles
                            WHERE version_id = %s AND article_label = %s
                        """, (version_id, article["article"]))
                        if cur.fetchone():
                            continue  # skip existing

                        cur.execute("""
                            INSERT INTO law_articles (version_id, article_label, article_text, tags, sort_order)
                            VALUES (%s,%s,%s,%s,%s)
                        """, (
                            version_id,
                            article["article"],
                            article["text"],
                            article.get("tags", []),
                            i + 1,
                        ))
                        inserted_articles += 1

                conn.commit()

    print(f"\n── COMPLETE ──")
    print(f"  Laws:     {inserted_laws} new")
    print(f"  Versions: {inserted_versions} new")
    print(f"  Articles: {inserted_articles} new")


def reset_all():
    """Delete all laws and re-import from scratch."""
    print("[WARN] Resetting all laws...")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM law_region_bindings")
            cur.execute("DELETE FROM law_articles")
            cur.execute("DELETE FROM law_versions")
            cur.execute("DELETE FROM law_masters")
            conn.commit()
    print("[OK] All laws deleted. Re-importing...")
    migrate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrate seed laws to PostgreSQL')
    parser.add_argument('--dry-run', action='store_true', help='Preview only')
    parser.add_argument('--reset', action='store_true', help='Delete all laws and re-import')
    args = parser.parse_args()

    ensure_tables()

    if args.reset:
        reset_all()
    else:
        migrate(dry_run=args.dry_run)
