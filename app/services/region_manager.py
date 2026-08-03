"""Region variant management for law library (U4).

Provides region hierarchy queries and law-region binding lookups.
Integrates with compliance checker to load region-appropriate laws.

Usage:
    from app.services.region_manager import get_applicable_laws, seed_regions
    laws = get_applicable_laws(region_code="440300")  # 深圳市
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Region DB operations ──

def seed_regions() -> int:
    """Seed region data into law_regions table. Idempotent."""
    import json, os
    from app.database import get_db_connection

    seed_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'laws', 'regions.json')
    if not os.path.exists(seed_path):
        logger.warning(f"Regions seed file not found: {seed_path}")
        return 0

    with open(seed_path, 'r', encoding='utf-8') as f:
        regions = json.load(f)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            inserted = 0
            for r in regions:
                cur.execute("""
                    INSERT INTO law_regions (region_code, region_name, parent_code, level)
                    VALUES (%s,%s,%s,%s)
                    ON CONFLICT (region_code) DO UPDATE
                    SET region_name = EXCLUDED.region_name,
                        parent_code = EXCLUDED.parent_code,
                        level = EXCLUDED.level
                """, (r["region_code"], r["region_name"], r.get("parent_code"), r.get("level", "national")))
                inserted += 1
            conn.commit()
    logger.info(f"Seeded {inserted} regions")
    return inserted


def get_region_hierarchy(region_code: str) -> List[str]:
    """Get the ancestor chain for a region. E.g., 深圳市 → [000000, 440000, 440300]."""
    from app.database import get_db_connection

    codes = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                current = region_code
                while current:
                    cur.execute("SELECT region_code, parent_code FROM law_regions WHERE region_code = %s", (current,))
                    row = cur.fetchone()
                    if not row:
                        break
                    codes.append(row[0])
                    current = row[1]
        # Return from root to leaf
        return list(reversed(codes))
    except Exception as e:
        logger.warning(f"Region hierarchy lookup failed: {e}")
        return ["000000"]


def bind_law_to_region(law_id: int, region_code: str, binding_type: str = "baseline"):
    """Bind a law to a region. Idempotent if same binding exists."""
    from app.database import get_db_connection

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO law_region_bindings (law_id, region_code, binding_type)
                    VALUES (%s,%s,%s)
                    ON CONFLICT (law_id, region_code) DO UPDATE
                    SET binding_type = EXCLUDED.binding_type
                """, (law_id, region_code, binding_type))
                conn.commit()
    except Exception as e:
        logger.warning(f"Failed to bind law {law_id} to region {region_code}: {e}")


def get_applicable_law_ids(region_code: Optional[str] = None) -> List[int]:
    """Get law IDs applicable to a given region (including inherited from ancestors).

    If region_code is None, returns all national laws.
    """
    from app.database import get_db_connection
    from psycopg2.extras import RealDictCursor

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if region_code:
                    hierarchy = get_region_hierarchy(region_code)
                    cur.execute("""
                        SELECT DISTINCT lrb.law_id
                        FROM law_region_bindings lrb
                        WHERE lrb.region_code = ANY(%s)
                    """, (hierarchy,))
                else:
                    cur.execute("""
                        SELECT DISTINCT lrb.law_id
                        FROM law_region_bindings lrb
                        JOIN law_regions lr ON lr.region_code = lrb.region_code
                        WHERE lr.level = 'national'
                    """)
                return [row["law_id"] for row in cur.fetchall()]
    except Exception as e:
        logger.warning(f"Failed to get applicable laws for region {region_code}: {e}")
        return []
