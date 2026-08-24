# Unified Bid Audit Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One-click "Full Audit" button that runs 7 bid-auditing functions across multiple bidders' files, producing scored DOCX + XLSX reports with composite scoring, fatal-flaw detection, and persistent history.

**Architecture:** New `audit` blueprint + `audit_engine.py` orchestrator + `audit_report.py` generator. Existing 7 services called as libraries — no changes to them. Background thread with SSE progress. Results stored in 3 new DB tables, reports persisted to disk.

**Tech Stack:** Flask SSE (existing pattern), python-docx 1.2.0, openpyxl 3.1.5, threading for background work, existing `ok()`/`err()` helpers.

## Global Constraints

- All endpoints use `ok()`/`err()` from `app.utils.helpers`
- All admin endpoints use `@admin_required` decorator from `app.routes.admin`
- SSE uses `Response(stream_with_context(gen), mimetype='text/event-stream')` pattern from `chat.py:355`
- No changes to existing 7 auditing services
- Bidder grouping via sibling folders (same parent_folder_id)

---
### Task 1: DB Schema — audit_runs, audit_file_results, audit_config

**Files:**
- Modify: `app/database.py`

- [ ] **Step 1: Add CREATE TABLE statements**

In `app/database.py`, after the existing `admin_audit_log` table creation, add:



- [ ] **Step 2: Add indexes**

In the same `init_db()` function, after existing index statements:



- [ ] **Step 3: Seed default audit config**

After table creation, seed default config if table is empty:



- [ ] **Step 4: Commit**

[master 5410aa2] feat: add audit_runs, audit_file_results, audit_config tables with indexes and seed data
 1 file changed, 3 insertions(+)
