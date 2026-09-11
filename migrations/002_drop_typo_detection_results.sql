-- Migration 002: Drop the dead typo_detection_results table
-- The typo-detection subsystem was removed in QA round 021; this table is no
-- longer written by any code path. Previously dropped unconditionally on every
-- app boot (app/database.py); moved here so it runs once, is versioned, and
-- cannot silently destroy data on a future restart.

DROP TABLE IF EXISTS typo_detection_results;
