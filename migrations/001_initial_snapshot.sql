-- Migration 001: Initial snapshot marker
-- This migration does nothing — it serves as a baseline marker.
-- Future migrations will build on top of this.
-- The actual schema is managed by database.py:init_postgres_tables().
--
-- After this baseline, ALL schema changes should go into migration files,
-- not into database.py ALTER blocks. This ensures:
--   1. Every change is versioned
--   2. Every change has a corresponding .rollback.sql
--   3. Changes can be reviewed before applying
--   4. The admin panel shows pending/risky migrations before execution

SELECT 1;
