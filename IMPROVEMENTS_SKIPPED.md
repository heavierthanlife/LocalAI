# Skipped Improvements Record

> **Session**: 2026-07-03 ~ 2026-07-04（原始记录）
> **Status**: **Historical record** — superseded by `data/unresolved.yaml` as active tracker, and by `DECISIONS.md` for decision rationale.
> **Context**: Full-stack audit, Red Team (质问模式) completion, document pipeline upgrade (EasyOCR→RapidOCR + MinerU), skill-to-app integration planning

---

> ## ⚠️ 2026-09-01 现实对齐更正
>
> 原始文档声称「已迁移到 MinerU + RapidOCR」，但经代码核验，**实际使用的文档管线是**：
>
> | 层 | 实际技术 | 证据 |
> |---|---|---|
> | 结构化提取 | **MarkItDown** 0.1.6 | `requirements.txt`；`file_processing.py` 11 处引用 |
> | .doc 转换 | **LibreOffice/soffice** | `_convert_doc_via_soffice` 6 处 |
> | PDF 流式提取 | **PyMuPDF (fitz)** | `file_processing.py` 17 处 |
> | 扫描件 OCR | **EasyOCR** 1.7.2 | `ocr.py`（仍为 EasyOCR 单例，无 RapidOCR） |
> | 归档/电子书依赖 | **未安装**（rarfile/py7zr/ebooklib/extract-msg 不在 requirements） | `requirements.txt` |
>
> **含义**：本清单中第 1-5 项的「参照系」（RapidOCR/MinerU 已在用）不成立。真实基线是
> **EasyOCR + MarkItDown + LibreOffice + PyMuPDF**。下方各条已按真实基线重估。

---

## 1. PaddleOCR instead of RapidOCR

| Field | Detail |
|-------|--------|
| **Status** | Skipped — 现仍维持 EasyOCR（未采用 RapidOCR 亦未采用 PaddleOCR） |
| **Dependencies** | `paddlepaddle` (~300MB), `paddleocr` |
| **Hardware** | CPU usable, GPU recommended for speed |
| **Pros** | 98% Chinese accuracy (highest), PP-Structure for table recognition, handwritten text support |
| **Cons** | 300MB+ install, PaddlePaddle framework install often fails on Windows, 2-7x slower on CPU |
| **Dropped** | 2026-07-04 |
| **2026-09-01 重估** | ⚠️ **仍未必要**。若想升级 OCR 精度，**RapidOCR（ONNX 30MB）仍是比 PaddleOCR 更轻的优先项**；PaddleOCR 安装难 + Windows 失败率高的结论不变。 |

---

## 2. Docling as Office Document Parser

| Field | Detail |
|-------|--------|
| **Status** | Skipped — 保持 python-docx/pptx/openpyxl + MarkItDown |
| **Dependencies** | `docling` (~1GB), `docling-serve` (optional FastAPI) |
| **Hardware** | CPU usable, GPU optional |
| **Pros** | Native DOCX/PPTX/XLSX support (IBM enterprise heritage), JSON layout coordinates, TableFormer table extraction, MIT license, LangChain/LlamaIndex native integration |
| **Cons** | Chinese document support weak (trained on English), OmniDocBench ~82 vs MinerU ~95, API still maturing |
| **Dropped** | 2026-07-04 |
| **2026-09-01 重估** | ⚠️ **重估**。原「对比 MinerU」的参照已不成立（无 MinerU）。中文弱 + API 未成熟仍成立。MarkItDown 满足当前需求。 |

---

## 3. Marker as PDF Parser

| Field | Detail |
|-------|--------|
| **Status** | Skipped — 保持 PyMuPDF + EasyOCR |
| **Dependencies** | `marker-pdf`, `surya-ocr` |
| **Hardware** | CPU usable, GPU recommended (H100: 120 pages/sec) |
| **Pros** | Fastest PDF→Markdown (0.18s/page serial, 120 pages/sec batched), 95.67 heuristic accuracy, strong table extraction |
| **Cons** | Chinese text support weak (Surya OCR optimized for English), **GPL license (commercial needs separate license)**, no Office format support, struggles with scanned/degraded PDFs (32% accuracy) |
| **Dropped** | 2026-07-04 |
| **2026-09-01 重估** | ⚠️ **仍否决**。中文弱 + **GPL 商用风险**是硬否决，与参照系无关。 |

---

## 4. Unstructured.io as Unified ETL Pipeline

| Field | Detail |
|-------|--------|
| **Status** | Skipped — 保持自研管线（MarkItDown + LibreOffice + PyMuPDF + EasyOCR） |
| **Dependencies** | `unstructured`, `unstructured-client`, Poppler, Tesseract (system-level) |
| **Hardware** | Docker image ~6.6GB, CPU usable |
| **Pros** | 64+ file format support, best-in-class table extraction, lowest hallucination rate, 30+ metadata fields, LangChain/LlamaIndex native |
| **Cons** | Massive install footprint (6.6GB Docker), complex local setup, OSS accuracy dipped in recent versions, enterprise pricing at scale |
| **Dropped** | 2026-07-04 |
| **2026-09-01 重估** | ❌ **过时**。6.6GB 太重、当前规模（<100 文档/天）用不上。评估无变化。 |

---

## 5. GPU Hardware / MinerU VLM Backend

| Field | Detail |
|-------|--------|
| **Status** | **Deferred → 已有一台 RTX 2080 Super 8GB 笔记本** |
| **Dependencies** | NVIDIA GPU ≥8GB VRAM, CUDA |
| **Hardware** | RTX 2080 Super (8GB) — 已有 ✅ |
| **Pros** | GPU 加速 OCR/嵌入/微调；VLM 布局理解 |
| **Cons** | 需要重装 CUDA 版 torch（当前项目装的是 `torch==2.12.1+cpu`，CUDA 不可用） |
| **Dropped** | 2026-07-04（当时 CPU-only）；**2026-09-01 重估：值得启用** |
| **Why** | 见下方「RTX 2080 Super 建议」。 |

---

## 6. Structured Logging Infrastructure

| Field | Detail |
|-------|--------|
| **Status** | Skipped — 单用户，纯文本日志足够 |
| **Dependencies** | `structlog` (Python), Loki + Promtail (Docker, ~300MB RAM) |
| **Hardware** | +300MB RAM |
| **Pros** | JSON structured logs queryable via LogQL in Grafana, unified metrics+logs dashboard, alert rules, multi-instance log aggregation |
| **Cons** | Additional Docker services, 300MB RAM overhead, LogQL learning curve, zero benefit for single-instance single-user |
| **Dropped** | 2026-07-04 |
| **2026-09-01 重估** | ❌ **过时**。单用户单实例，除非未来多实例/多操作员才重估。 |

---

## 7. Full E2E Test Suite (30+ Playwright tests)

| Field | Detail |
|-------|--------|
| **Status** | Scoped down → 5 smoke tests（现已有 7 smoke + 64 regression + 批量快照） |
| **Dependencies** | `playwright`, `pytest-playwright`, `pytest` |
| **Pros** | Full regression coverage, CI-ready, catches frontend breakage before deployment |
| **Cons** | 30+ tests require maintenance as HTML/CSS changes, selector brittleness, ~15min CI runtime |
| **Dropped** | 2026-07-04 |
| **2026-09-01 重估** | ⚠️ **部分价值 — 建议重估**。前端 app.js ~10k 行已稳定，全量测试已到 92/92。**Playwright 端到端现比 7 月更有价值**，但建议用 **5-8 条核心流程**（登录/上传/聊天/清标/下载）而非 30 条。 |

---

## 8. admin.py Full Modular Split

| Field | Detail |
|-------|--------|
| **Status** | ✅ **已解决（2026-08-28 FIX-003）** |
| **Dependencies** | None (pure refactor) |
| **Why** | `admin.py` 已从 4,820 → **1,653** 行，拆分为 `admin_regeneration` / `admin_knowledge_lab` / `admin_ops` 子模块。**此条目作废**。 |

---

## 9. Theme Factory — 10 Pre-set Skins

| Field | Detail |
|-------|--------|
| **Status** | Skipped — 主题美感未达专业招标系统标准 |
| **Dependencies** | None |
| **Why** | 用户评估后认为主题不够美观。L/D 切换已足够。**2026-09-01 重估**：无变化，保留不考虑。 |

---

## RTX 2080 Super (8GB) — 建议（2026-09-01 新增）

用户有 **RTX 2080 Super 8GB** 笔记本。当前项目 torch 是 **CPU 版**（`torch==2.12.1+cpu`，`cuda.is_available()=False`），所以 GPU 完全未启用。以下是启用与价值排序：

### 当前就能无代码改动受益的（推荐先做）

| 收益点 | 现状 | 改法 | 价值 |
|---|---|---|---|
| **EasyOCR GPU 加速** | `ocr.py` 已支持 `OCR_GPU=auto` 探测 cuda，但 cuda 不可用 | 重装 CUDA torch | 中文扫描件 OCR 5-10x 快 |
| **MiniLM 嵌入**（语义/RAG） | sentence-transformers CPU 推理 | 同上 | 语义搜索/RAG 提速 |
| **LoRA 微调**（`run_lora_training.py`, Unsloth Qwen2.5-7B） | CPU 几乎不可行 | CUDA torch + unsloth cu121 | 8GB 可跑 Qwen 7B QLoRA（4bit） |

### 启用方法

```bash
# 1. 卸载 CPU torch
pip uninstall torch torchvision
# 2. 装 CUDA 版（2080S 是 Ampere 前代 → cu121 兼容）
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu121
# 3. 验证
python -c "import torch; print(torch.cuda.is_available())"   # → True
# 4. OCR 自动用 GPU（OCR_GPU 默认 auto）；LoRA 用 unsloth[cu121]
```

### 不建议的（8GB 显存不够或收益低）

| 项 | 原因 |
|---|---|
| MinerU VLM 后端 | 8GB 勉强，VLM 布局理解收益对中文招标文档有限 |
| 大模型本地推理（≥13B） | 8GB 只能 QLoRA 7B 或量化推理，日常 LLM 仍走云 API |
| 本地 embedding 大模型 | MiniLM 已足够，不必上大模型 |

**核心建议**：2080S 的最大价值 = **OCR GPU 加速 + LoRA 微调 Qwen2.5-7B**（Unsloth 官方支持 8GB QLoRA）。两者都只需重装 CUDA torch，无代码改动。
