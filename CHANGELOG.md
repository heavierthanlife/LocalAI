# Changelog

All notable changes to 中联招标智能助手. Maintained by Claude Code across sessions.

---

## [2026-07-06] — Cross-device sync + unread tracking

### Added
- `GET /chat/poll/<thread_id>?since_id=N` — lightweight delta-fetch for new messages
- Unified real-time polling: common chats (5s) and project chats (3s)
- Per-browser unread badges on all sidebar threads via `localStorage` (`zlai_read_<thread_id>`)
- Unread count clears on scroll-to-bottom, debounced at 800ms
- `last_msg_id` field in `get_user_sessions()` response for unread calculation

### Changed
- Project chat polling now uses `/chat/poll` instead of `/admin/.../ai_activity` (delta vs full reload)

---

## [2026-07-06] — Mobile responsiveness (3-tier)

### Added
- Phone breakpoint (<640px): sidebar overlay, tab "更多" dropdown, fixed input, safe-area support
- Tablet breakpoint (640–1024px): sidebar narrowed to 180px, adjusted font sizes
- `font-size: 16px` on consent modal inputs to prevent iOS auto-zoom
- Swipe-to-close gesture on sidebar overlay
- Touch targets min 44px across all interactive elements

### Changed
- Sidebar breakpoint refactored: 768px → 640px (phone) + 1024px (tablet)
- Admin panels: tables get `overflow-x: auto`, secondary columns hidden on phone
- Knowledge lab: 2-column grid stacks to 1-column on phone

---

## [2026-07-06] — is_grilling query blind spots (7 fixes)

### Fixed
- `backfill_project_chat`: SQL now excludes grilling threads (`is_grilling = FALSE`)
- `update_project`: title sync skips grilling threads
- `add_project_member`: auto-backfill check excludes grilling threads
- `generate` endpoint: backfill queries exclude grilling threads (2 locations)
- `project_ai_activity`: excludes grilling thread messages
- `project_unread_count`: excludes grilling thread messages
- Frontend: 3 `find(s => s.project_id == ...)` calls now filter `!s.is_grilling`

---

## [2026-07-04] — Skills audit + AI document review + API format + tests

### Added
- `POST /admin/review/document` — AI five-axis document review (code-review-and-quality skill)
- "🤖 AI 文档审查" panel in Review tab with axis checkboxes and result table
- `ok()` and `err()` unified API response helpers in `app/utils/helpers.py`
- 6 pytest smoke tests in `tests/test_smoke.py`
- `pytest.ini` configuration
- `IMPROVEMENTS_SKIPPED.md` — 9 deferred improvements with rationale

### Changed
- Red Team endpoints now use `ok()`/`err()` unified format
- `IMPROVEMENTS_SKIPPED.md` records all skipped upgrades with timestamps

---

## [2026-07-04] — Document pipeline upgrade: EasyOCR → RapidOCR + MinerU

### Added
- MinerU (`_try_mineru`, `_strip_markdown`) as primary PDF/DOCX/PPTX/XLSX parser in `file_processing.py`
- `_ocr_pdf_legacy` fallback in `ingest_pipeline.py`

### Changed
- `app/services/ocr.py`: EasyOCR → RapidOCR (ONNX, CPU-optimized, 30MB vs 300MB)
- `extract_text_from_file`: MinerU tried first for structured formats, legacy code retained as fallback
- `ingest_pipeline.py`: `_ocr_pdf` → MinerU, `_ocr_image` → RapidOCR (via updated OCRManager)

### Removed
- `easyocr` from `requirements.txt`; `ocr_manager`/`run_ocr` placeholders from `globals.py`

---

## [2026-07-03] — Red Team (质问模式) frontend completion

### Added
- "🔥 质问模式" button in chat sidebar + "🔥 质问" button in project tabs
- `_isCurrentSessionGrill` flag and red banner in chat area
- `is_grilling` field in `get_user_sessions()` response
- 🔥 prefix on grill threads in sidebar

### Fixed
- `/send_stream`, `/send`, `/regenerate` now actually use `get_redteam_agent()` instead of just swapping prompt
- `summary` CSS: replaced `display: inline-block` with custom ▶ collapse indicator
- `.token-control` and `.action-group` missing `display: flex` restored
- Chat toolbar restructured: 4 detection features + prompt templates moved into collapsible section

---

## [2026-07-03] — Initial audit (prior assistant handoff)

### Verified
- `is_grilling BOOLEAN DEFAULT FALSE` in `chat_sessions` table
- `redteam_agent.py` with `REDTEAM_SYSTEM_PROMPT` and `get_redteam_agent()`
- `/api/chat/create_grill_thread` and `/api/projects/<id>/get_or_create_grill_thread` endpoints

### Found broken
- Red Team agent never invoked (only prompt swap)
- Frontend HTML/JS completely missing (0% done)
- CSS flex containers and collapse indicators missing
