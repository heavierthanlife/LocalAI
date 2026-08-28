# System Health Checklist — Automated Audit

> Generated: auto  |  verify_fixes.py: 21/21 pass  |  
> regression tests: 16/16 pass
> Fix applied: law_monitor cursor (FIX-019-005), XSS/DOMPurify (FIX-019-002), 
> NVIDIA cleanup, pre-commit blocking hook

Legend: `[x]` = verified  |  `[?]` = needs runtime/DB  |  `[ ]` = check failed

## PROCESS

_(6/6 verified automatically)_

- [x] Flask app factory importable
- [x] run.py exists
- [x] HTTPS :5443 configured
- [x] HTTP→HTTPS redirect :5000
- [x] Celery app file exists
- [x] APScheduler import

## ROUTES

_(19/19 verified automatically)_

- [x] Route module files exist: 18
- [x] Route: admin.py
- [x] Route: audit.py
- [x] Route: auth.py
- [x] Route: batch.py
- [x] Route: cases.py
- [x] Route: chat.py
- [x] Route: clearance.py
- [x] Route: compliance.py
- [x] Route: credit.py
- [x] Route: document_analysis.py
- [x] Route: graph.py
- [x] Route: knowledge.py
- [x] Route: projects.py
- [x] Route: tasks.py
- [x] Route: templates.py
- [x] Route: timeline.py
- [x] Route: upload.py
- [x] Route: wiki.py

## DATABASE

_(28/28 verified automatically)_

- [x] Tables defined in database.py: 70
- [x] PostgreSQL driver (psycopg2)
- [x] Core table: users
- [x] Core table: chat_messages
- [x] Core table: chat_sessions
- [x] Core table: knowledge_lab_files
- [x] Core table: company_knowledge_base
- [x] Core table: project_files
- [x] Core table: user_files
- [x] Core table: recycle_bin
- [x] Core table: kb_recycle_bin
- [x] Core table: project_recycle_bin
- [x] Core table: admin_audit_log
- [x] Core table: project_ai_memory
- [x] Core table: file_text_cache
- [x] Core table: image_description_cache
- [x] Core table: batch_comparison_results
- [x] Core table: task_deposit_items
- [x] Core table: compliance_feedback
- [x] Core table: audit_runs
- [x] Core table: audit_file_results
- [x] Core table: audit_config
- [x] Core table: wiki_origin_links
- [x] Core table: bid_templates
- [x] Core table: entity_relationships
- [x] Connection pool configured
- [x] Connection validation (SELECT 1)
- [x] Migration tool (manage_db.py)

## AI

_(5/5 verified automatically)_

- [x] LLM provider module
- [x] API keys documented in .env.example: 4/4
- [x] Model list function exists
- [x] SSE streaming implemented
- [x] Thinking chain split

## FILESYSTEM

_(16/17 verified automatically)_

- [x] Directory exists: data/
- [x] Directory exists: data/user_files/
- [x] Directory exists: data/project_files/
- [x] Directory exists: data/dump/
- [x] Directory exists: data/chromadb/
- [x] Directory exists: data/flask_session/
- [x] Directory exists: data/workflows/
- [x] Directory exists: data/temp/
- [x] MAX_CONTENT_LENGTH: 50 MB
- [?] MAX_CONCURRENT_UPLOADS set  —  may use env var fallback
- [x] File upload type whitelist
- [x] Recycle bin service module
- [x] Temp file + ghost chat cleanup
- [x] OCR module exists
- [x] Skill validator module
- [x] SKILL.md files valid (0/0)
- [x] ChromaDB integration

## REDIS

_(7/7 verified automatically)_

- [x] Celery app file
- [x] Celery task modules registered
- [x] Celery beat schedule
- [x] Nightly trainer module
- [x] REDIS_URL/HOST env set
- [x] Flask session configured
- [x] Credit task registry Redis-backed

## RATELIMIT

_(2/5 verified automatically)_

- [x] flask-limiter initialized
- [x] Global rate limit: 120/min
- [?] Chat rate limits
- [?] Upload rate limits
- [?] Auth rate limits

## SECURITY

_(10/10 verified automatically)_

- [x] SECRET_KEY set in env
- [x] .env file exists
- [x] .env.example exists
- [x] HTTPS redirect code
- [x] CSRF opt-in (JSON safe)
- [x] Session signing enabled
- [x] Admin permission checks
- [x] File upload type whitelist
- [x] Graph API requires login
- [x] Admin PIN fail-closed in production

## ADMIN

_(5/5 verified automatically)_

- [x] Admin blueprint registered
- [x] Admin DB tables overview
- [x] Runtime config service
- [x] Admin user management
- [x] Admin audit log view

## INTEGRATION

_(10/10 verified automatically)_

- [x] Chat send endpoint
- [x] RAG retrieval
- [x] Knowledge lab search
- [x] Wiki CRUD + search
- [x] Batch compare
- [x] Compliance check + report
- [x] Audit engine run + report
- [x] Law corpus clean files: 14
- [x] Law corpus national coverage
- [x] Law corpus beijing coverage

## META

_(6/6 verified automatically)_

- [x] Fix registry exists
- [x] Fix verifier script
- [x] Regression tests
- [x] Pre-commit hook
- [x] Unresolved issues tracker
- [x] Current state compilation

## SUMMARY

| Status | Count |
|--------|-------|
| [x] Pass | 114 |
| [?] Manual | 4 |
| **Total** | **118** |

**Verification rate**: 114/118 auto-verified (96%)
