# Unified Bid Audit Engine — Design Spec

**Date:** 2026-07-06
**Status:** approved
**Branch:** master

## Overview

One-click "Full Audit" that runs all 7 bid-auditing functions across multiple bidders' files, producing a scored, structured report (DOCX + XLSX) with composite scoring, fatal-flaw detection, and persistent history.

## Architecture

```
POST /admin/audit/preflight          — check text extraction status per file
POST /admin/audit/start              — launch audit (background thread)
GET  /admin/audit/progress/<run_id>  — SSE live progress
GET  /admin/audit/result/<run_id>    — structured JSON result
GET  /admin/audit/history/<proj_id>  — past audit runs for project
GET  /admin/audit/download/<run_id>/docx  — download DOCX report
GET  /admin/audit/download/<run_id>/xlsx  — download XLSX report
GET  /admin/audit/config             — get global audit config
PUT  /admin/audit/config             — update global audit config
```

New blueprint: `app/routes/audit.py` (`/audit` prefix, admin-required).

## New Files

| File | Purpose |
|------|---------|
| `app/services/audit_engine.py` | Orchestrator: preflight, run dispatch, scoring, retry, progress emission |
| `app/services/audit_report.py` | DOCX + XLSX generation with TOC, scoring tables, narrative sections |
| `app/routes/audit.py` | REST endpoints for audit lifecycle |

## Modified Files

| File | Change |
|------|--------|
| `app/database.py` | 3 new tables: `audit_runs`, `audit_file_results`, `audit_config` |
| `app/routes/__init__.py` | Register audit blueprint |
| `templates/index.html` | Full Audit button (project + per-file), progress modal (SSE), results/storage panel in Review tab, admin Audit Settings section |
| `static/js/app.js` | Audit flow (preflight→start→SSE progress→results→download), toast on completion, history browser, admin config CRUD |
| `static/css/app.css` | Progress modal, audit results panel, admin config form styles |

## DB Schema

```sql
CREATE TABLE audit_runs (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    user_id TEXT REFERENCES users(user_id),
    status TEXT NOT NULL DEFAULT 'running',  -- running, completed, failed
    config_snapshot JSONB,       -- frozen copy of weights/thresholds at run time
    overall_score REAL,
    overall_status TEXT,          -- PASS / FAIL / ERROR
    bidder_count INTEGER,
    file_count INTEGER,
    docx_path TEXT,
    xlsx_path TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE audit_file_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES audit_runs(id) ON DELETE CASCADE,
    file_id INTEGER REFERENCES project_files(id) ON DELETE SET NULL,
    folder_id INTEGER REFERENCES project_folders(id) ON DELETE SET NULL,
    bidder_label TEXT,            -- folder name
    filename TEXT,
    function_name TEXT,           -- rule_extraction, compliance_check, etc.
    score REAL,
    status TEXT,                  -- success, failed, skipped, error
    findings JSONB,               -- raw function output
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE TABLE audit_config (
    id SERIAL PRIMARY KEY,
    function_name TEXT UNIQUE NOT NULL,
    enabled_by_default BOOLEAN DEFAULT true,
    fail_threshold REAL DEFAULT 50,
    weight REAL DEFAULT 14.28,    -- 100/7, normalized at runtime
    severity_thresholds JSONB,    -- e.g. {"critical": 3, "high": 10}
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

Indexes: `(run_id, file_id)`, `(run_id, function_name)`, `(project_id, started_at DESC)`.

## Data Flow

### 1. Preflight
- Input: array of folder IDs (sibling folders = bidders)
- For each file in each folder, check if `project_files.content` is non-null and non-empty
- Return per-file status: `ready`, `missing` (no extracted text), `error` (extraction previously failed)
- Frontend shows missing/error counts, user chooses to proceed (extract-on-demand) or skip unready files

### 2. Start
- Input: folder IDs, enabled functions list, extraction flag
- Creates `audit_run` row (status=running, config_snapshot=frozen config)
- Spawns background thread to run orchestrator
- Returns `run_id` immediately

### 3. Orchestrator (background thread)
For each bidder folder:
  For each file:
    Extract text if flagged (on-demand extraction)
    For each enabled function:
      Call existing service function with file text
      On failure: retry once after 3s delay
      Apply scoring formula → produce 0–100 score
      Check fail_threshold → if below, mark fatal-flaw
      Write `audit_file_results` row
      Emit progress event

After all files:
  Compute bidder-level aggregates (avg per function)
  Compute cross-bidder comparison matrix
  Compute global composite score:
    composite = SUM(weight_i * score_i) for all functions, all bidders
  If any function fell below fail_threshold → overall_status = FAIL
  Generate DOCX via python-docx
  Generate XLSX via openpyxl
  Persist file paths to audit_runs
  Update audit_runs.status = completed

### 4. Progress (SSE)
Events:
- `phase` — {phase: "extracting"|"auditing"|"scoring"|"reporting"}
- `file_start` — {bidder, filename, file_index, total_files}
- `function_done` — {bidder, filename, function, score, status}
- `function_error` — {bidder, filename, function, error, retrying}
- `complete` — {run_id, overall_score, overall_status}
- `error` — {message} (fatal audit error)

Frontend: modal with per-file progress bars, green checkmarks / red X per function, closeable (toast notifies on complete).

### 5. Results
- Load `audit_runs` + all `audit_file_results` for the run
- Return structured JSON with: run metadata, per-bidder scores, per-file scores, cross-bidder comparison, overall composite, fatal-flaw flags, download URLs

### 6. History
- `GET /admin/audit/history/<project_id>` → list of past audit runs (id, started_at, overall_score, overall_status, user_id)
- Click any past run → loads results panel with that run's data
- Download buttons available for completed runs with persisted files

## Scoring Formulas (v1)

| Function | Formula | Range |
|----------|---------|-------|
| Rule Extraction | `min(100, extracted_count / expected_min * 100)` | 0–100 |
| Compliance Check | `(rules_passed / total_rules) * 100` | 0–100 |
| Typo Detection | `max(0, 100 - findings_per_10k_chars * penalty)` | 0–100 |
| Quote Anomaly | `100 - severity_index` (composite of outlier count, clustering, Benford) | 0–100 |
| Relationship Extraction | `100 - risk_signals * risk_weight` | 0–100 |
| AI Document Review | Average of 5 axis scores parsed from LLM response | 0–100 |
| Style Analysis | Formality + consistency composite from style profile | 0–100 |

Configurable per function: `fail_threshold` (default 50), `weight` (default 100/7 ≈ 14.28).

## Fatal-Flaw Logic

When any function score < `fail_threshold` for any file → overall_status = FAIL.
All functions still run to completion (no abort).
Report marks failed functions with red banner and explanation.

## Report Structure

### DOCX
1. Executive Summary (overall composite score, fatal-flaw status, top risks)
2. Bidder Comparison (score matrix table, ranked)
3. Per-Bidder Detail (for each bidder):
   3.N.1 Overall Score
   3.N.2 Rule Extraction (score + findings)
   3.N.3 Compliance Check (score + violations)
   3.N.4 Typo Detection (score + error list)
   3.N.5 Quote Analysis (score + anomalies)
   3.N.6 Relationship Analysis (score + signals)
   3.N.7 AI Document Review (score + narrative)
   3.N.8 Style Analysis (score + profile)
4. Appendices (raw data tables, audit metadata, config snapshot)

TOC generated via python-docx heading styles. Skipped/toggled-off functions get a placeholder: "未启用 (Not enabled for this audit)".

### XLSX
- Sheet "Summary": bidder × function score matrix + composite
- Sheet per function: raw findings, one row per finding
- Sheet "Comparison": cross-bidder ranking table
- Sheet "Config": frozen config snapshot

## Cross-Bidder Comparison

Enabled when ≥2 bidders audited. Same function toggles locked across all bidders for fairness.
Output: score matrix table (bidders × functions), delta columns, rank per function.

## Error Handling

- Per-function failure: retry once after 3s → if still fails, mark as "error" with N/A score, continue
- Text extraction failure: skip file, mark all functions for that file as "skipped"
- Full audit crash (unhandled exception in orchestrator): mark run as "failed", persist partial results

## Admin Config UI

Located in Admin tab → "Audit Settings" section.
Per-function row: enabled_by_default toggle, fail_threshold slider (0–100), weight slider (0–100%), severity thresholds (key-value pairs, e.g., critical→3).
Global save button. Weights auto-normalized to 100% on save if they don't sum to 100%.

## Frontend States

### Full Audit Button
- Project page: "Full Audit" button visible when project has ≥1 folder with files
- File row: "Run All Audits" in action dropdown
- Both disabled during a running audit (check via polling audit_runs status)

### Progress Modal
- Phases: Preflight → Extracting → Auditing → Reporting
- Per-file progress: filename + 7 function dots (gray→spinner→green/red)
- Close button: hides modal, leaves background task running
- Toast on complete: "Full audit complete — 项目X — PASS/FAIL"

### Results Panel (Review Tab)
- Existing Review tab gets new "Audit History" section
- Per-project: list of past runs with date, score, status badge
- Click to expand: full per-function, per-file, per-bidder breakdown
- Download buttons (DOCX, XLSX) per completed run

### Empty/Error States
- No files in project: button disabled, tooltip "Add files to start audit"
- No extracted text: preflight shows warning, user chooses extract-or-skip
- Audit in progress: button shows spinner, tooltip "Audit running..."
- All functions failed: error banner, retry suggestion
- No past audits: "No audits yet. Run your first full audit."

## Implementation Order

1. DB tables + audit_config seeding
2. `audit_engine.py` — orchestrator, scoring, retry, progress queue
3. `audit_report.py` — DOCX + XLSX generation
4. `audit.py` routes — all endpoints + SSE
5. Admin config UI (HTML + JS + CSS)
6. Full Audit button + progress modal (HTML + JS + CSS)
7. Results panel + history browser (HTML + JS + CSS)
8. Per-file "Run All Audits" action
9. End-to-end testing

## Dependencies

- `python-docx` — DOCX generation (already in requirements)
- `openpyxl` — XLSX generation (check if present)
- Existing 7 services — called as libraries, no changes required
- SSE pattern — reuse existing pattern from chat/batch progress
- `app.services.file_processing.extract_text_from_file` — on-demand extraction
