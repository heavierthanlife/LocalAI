-- Rollback for migration 002.
-- The typo_detection_results table belonged to the removed typo-detection
-- subsystem, so there is nothing meaningful to restore. No-op on purpose.

SELECT 1;
