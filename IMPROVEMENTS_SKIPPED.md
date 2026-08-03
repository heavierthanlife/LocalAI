# Skipped Improvements Record

> Session: 2026-07-03 ~ 2026-07-04
> Status: **Historical record** — superseded by `data/unresolved.yaml` as active tracker.
> Context: Full-stack audit, Red Team (质问模式) completion, document pipeline upgrade (EasyOCR→RapidOCR + MinerU), skill-to-app integration planning

---

## 1. PaddleOCR instead of RapidOCR

| Field | Detail |
|-------|--------|
| **Status** | Skipped — chose RapidOCR |
| **Dependencies** | `paddlepaddle` (~300MB), `paddleocr` |
| **Hardware** | CPU usable, GPU recommended for speed |
| **Pros** | 98% Chinese accuracy (highest), PP-Structure for table recognition, handwritten text support |
| **Cons** | 300MB+ install, PaddlePaddle framework install often fails on Windows, 2-7x slower than RapidOCR on CPU |
| **Dropped** | 2026-07-04 |
| **Why** | RapidOCR uses same PP-OCRv4 model via ONNX, 93% accuracy is sufficient for printed bid documents, installs in 30MB, 2-7x faster on CPU. Table recognition deferred to MinerU. |

---

## 2. Docling as Office Document Parser

| Field | Detail |
|-------|--------|
| **Status** | Skipped — kept python-docx/pptx/openpyxl, added MinerU |
| **Dependencies** | `docling` (~1GB), `docling-serve` (optional FastAPI) |
| **Hardware** | CPU usable, GPU optional |
| **Pros** | Native DOCX/PPTX/XLSX support (IBM enterprise heritage), JSON layout coordinates, TableFormer table extraction, MIT license, LangChain/LlamaIndex native integration |
| **Cons** | Chinese document support weak (trained on English), OmniDocBench ~82 vs MinerU ~95, API still maturing (2024 open-source) |
| **Dropped** | 2026-07-04 |
| **Why** | Chinese bidding documents need native Chinese layout understanding. MinerU dominates (95.69 OmniDocBench, Shanghai AI Lab). python-docx/pptx/openpyxl kept for Office read/write. |

---

## 3. Marker as PDF Parser

| Field | Detail |
|-------|--------|
| **Status** | Skipped — chose MinerU |
| **Dependencies** | `marker-pdf`, `surya-ocr` |
| **Hardware** | CPU usable, GPU recommended (H100: 120 pages/sec) |
| **Pros** | Fastest PDF→Markdown (0.18s/page serial, 120 pages/sec batched on H100), 95.67 heuristic accuracy on marker_benchmark, strong table extraction with `--use_llm` flag |
| **Cons** | Chinese text support weak (Surya OCR optimized for English), GPL license (commercial needs separate license), no Office format support, struggles with scanned/degraded PDFs (32% accuracy) |
| **Dropped** | 2026-07-04 |
| **Why** | Chinese bidding documents are the primary use case. Chinese support is non-negotiable. GPL license risk for commercial deployment. |

---

## 4. Unstructured.io as Unified ETL Pipeline

| Field | Detail |
|-------|--------|
| **Status** | Skipped — kept custom pipeline |
| **Dependencies** | `unstructured`, `unstructured-client`, Poppler, Tesseract (system-level) |
| **Hardware** | Docker image ~6.6GB, CPU usable, GPU optional |
| **Pros** | 64+ file format support, best-in-class table extraction (0.820 accuracy, self-reported), lowest hallucination rate (0.051), 30+ metadata fields per chunk, LangChain/LlamaIndex native, FedRAMP authorized |
| **Cons** | Massive install footprint (6.6GB Docker), complex local setup (Poppler + Tesseract system deps), OSS version accuracy reportedly dipped in recent versions, enterprise pricing at scale |
| **Dropped** | 2026-07-04 |
| **Why** | Overkill for current scale. MinerU + RapidOCR cover the needed formats. 6.6GB image too heavy for current deployment. Revisit if processing volume exceeds 100+ documents/day with mixed formats. |

---

## 5. GPU Hardware / MinerU VLM Backend

| Field | Detail |
|-------|--------|
| **Status** | Deferred — using MinerU pipeline (CPU) backend |
| **Dependencies** | NVIDIA GPU ≥8GB VRAM, CUDA 12.x |
| **Hardware** | RTX 3060/4060 minimum, RTX 3090 recommended |
| **Pros** | MinerU hybrid/VLM backend: 0.4s/page (vs 3-8s CPU), OmniDocBench 95.69 (SOTA), VLM-powered layout understanding superior to pipeline |
| **Cons** | $300-800 hardware cost, power consumption, Docker GPU passthrough setup |
| **Dropped** | 2026-07-04 |
| **Why** | Current deployment is CPU-only. MinerU pipeline backend on CPU is still faster than EasyOCR (3-8s vs 5-10s/page) with better accuracy. GPU purchase requires budget approval. |

---

## 6. Structured Logging Infrastructure

| Field | Detail |
|-------|--------|
| **Status** | Skipped — single user, plain text logs sufficient |
| **Dependencies** | `structlog` (Python), Loki + Promtail (Docker containers, ~300MB RAM) |
| **Hardware** | +300MB RAM for Loki + Promtail containers |
| **Pros** | JSON structured logs queryable via LogQL in Grafana, unified metrics+logs dashboard, alert rules on log patterns, multi-instance log aggregation (needed if scaling beyond 1 instance) |
| **Cons** | Additional Docker services to maintain, 300MB RAM overhead, learning curve for LogQL, zero benefit for single-instance single-user deployment |
| **Dropped** | 2026-07-04 |
| **Why** | User is the only person reading logs. Plain text `logging` module with file rotation is sufficient. Revisit if deployment scales to multiple instances or multiple operators. |

---

## 7. Full E2E Test Suite (30+ Playwright tests)

| Field | Detail |
|-------|--------|
| **Status** | Scoped down — implementing 5 core smoke tests instead |
| **Dependencies** | `playwright`, `pytest-playwright`, `pytest` |
| **Hardware** | None additional (headless Chromium) |
| **Pros** | Full regression coverage, CI-ready, catches frontend breakage before deployment |
| **Cons** | 30+ tests require weekly maintenance as HTML/CSS changes, test selector brittleness, ~15min CI runtime, dedicated QA time needed |
| **Dropped** | 2026-07-04 |
| **Why** | Frontend recently underwent major restructuring. HTML selectors still stabilizing. 5 core smoke tests (login, chat, upload, project create, knowledge upload) give 80% coverage at 20% cost. Expand when frontend stabilizes. |

---

## 8. admin.py Full Modular Split

| Field | Detail |
|-------|--------|
| **Status** | Deferred — restructuring plan approved but too large for current session |
| **Dependencies** | None (pure refactor) |
| **Hardware** | N/A |
| **Pros** | 4571-line monolith → ~8 domain files, cleaner imports, easier maintenance, parallel development possible |
| **Cons** | Risk of breaking route registrations, circular import hazards, ~4-6 hours of careful surgery |
| **Dropped** | 2026-07-04 |
| **Why** | Current structure works. Splitting is pure refactoring with no user-visible benefit. Defer until next major feature addition that would bloat the file further. |

---

## 9. Theme Factory — 10 Pre-set Skins

| Field | Detail |
|-------|--------|
| **Status** | Skipped — themes not visually appealing enough |
| **Dependencies** | None (CSS variables + JS selector) |
| **Hardware** | N/A |
| **Pros** | Ocean Depths, Sunset Boulevard, Forest Canopy, etc. — 10 professionally designed color/font combinations, cosmetic variety for users |
| **Cons** | Theme designs evaluated as not meeting aesthetic bar for professional bidding system, themes designed for presentations/landing pages not data-heavy work tools |
| **Dropped** | 2026-07-04 |
| **Why** | User reviewed theme-showcase.pdf and found the themes not beautiful enough for the system. Light/dark toggle already sufficient. |

