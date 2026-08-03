-- Database Schema Snapshot — Local_AI
-- Generated: 2026-07-14 (placeholder: run `python scripts/manage_db.py snapshot` to regenerate)
-- Actual schema definition: app/database.py:init_postgres_tables()
--
-- HOW TO REGENERATE:
--   1. Ensure database is running and .env is configured
--   2. python scripts/manage_db.py snapshot
--   3. Copy data/current_schema.sql to repair_kit/SCHEMA_SNAPSHOT.sql
--
-- Below is the authoritative table list extracted from init_postgres_tables().
-- For full column definitions, see the CREATE TABLE statements in app/database.py
-- or regenerate this file with a live database connection.

-- Core
--   users                        — User accounts, roles, PIN auth
--   chat_sessions                — Chat conversation sessions
--   chat_messages                — Individual messages within sessions
--   user_files                   — User-uploaded personal files
--   archived_sessions            — Archived/closed chat sessions
--   image_description_cache      — Cached VL model descriptions
--   file_usage                   — File access tracking
--   consent                      — User consent records
--   feedback                     — User feedback on LLM responses
--   message_responses            — Cached LLM responses
--   message_quotes               — Quoted message references
--   regen_votes                  — Regeneration voting
--   regen_vote_ballots           — Individual votes in regen elections

-- Projects
--   projects                     — Bid projects
--   project_members              — Project team membership
--   project_folders              — Project file folder structure
--   project_files                — Project-attached files
--   project_file_versions        — Version history for project files
--   project_file_comments        — Comments on project files
--   project_ai_memory            — AI persistent memory per project
--   member_workflows             — Workflow assignments per member
--   workflow_kpi                 — KPI tracking for workflows
--   project_file_usage           — File access within projects
--   project_folder_comments      — Folder-level comments
--   project_todos                — Project-level todo items

-- Knowledge & Company KB
--   knowledge_lab_files          — Personal knowledge base files
--   company_knowledge_base       — Shared company knowledge base
--   file_text_cache              -- Cached extracted text

-- Wiki
--   wiki_origin_links            -- Wiki page origin/source tracking

-- Recycle Bins
--   recycle_bin                  -- Chat/file recycle bin
--   project_recycle_bin          -- Project recycle bin
--   project_folders_recycle_bin  -- Project folder recycle bin
--   kb_recycle_bin               -- Knowledge base recycle bin

-- Admin & Audit
--   admin_audit_log              — Admin action audit trail
--   audit_config                 — Audit engine configuration
--   audit_runs                   — Audit execution records
--   audit_file_results           — Per-file audit results
--   file_analysis                — File analysis results

-- Batch & Compliance
--   batch_comparison_results     — Batch file comparison
--   compliance_feedback          — Compliance review feedback
--   compliance_reports           -- Compliance check reports
--   quote_anomaly_results        — Quote anomaly detection
--   typo_detection_results       — Typo detection results

-- Relationship & Entity
--   entity_relationships         — Named entity relationship graph
--   relationship_risk_summary    — Risk assessment per relationship

-- Task Deposit
--   task_deposit_items           — Deposited items in task system
--   task_deposit_permissions     — Access control for deposited items

-- Checkpoint (AI Agent)
--   checkpoints                  — AI agent state checkpoints
--   checkpoint_writes            — Checkpoint incremental writes

-- Other
--   skill_usage_cache            — Skill usage tracking cache
--   search_cache                 -- Search result cache
--   semantic_cache               -- Semantic query cache
