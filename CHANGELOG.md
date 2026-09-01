# Changelog

All notable changes to 中联招标智能助手.

**格式**：Keep a Changelog 风格（Added / Changed / Fixed / Removed），日期降序。
**维护约定**：每次功能升级/修复在顶部新增条目；合规相关改动必须带 `regression: 3/3 baseline passed` 验证（见 `CONTRIBUTING.md` §回归测试）。
**详细迭代记录**：2026-08-28 起的完整工作记录见 `20260827log.md` §10。

---

## [2026-09-01] — 清标评分系列（FIX-010 ~ FIX-014）

### FIX-014 自适应招标高频词 + 异组件文本相似度守卫（方案 Y + Z-1）
- **方案 Y** `file_processing.py`：招标文件高频词并入停用集从固定 top-50 改为自适应 `k = min(200, max(50, len//500))`，并加 **TF≥2 守卫**（不误杀 TF=1 的独特技术参数）
- **方案 Z-1** `batch_orchestrator.py`：新增 `_detect_component`（价格标/技术标/商务标/unknown，双 unknown fail-safe）；异组件 pair（如价格标↔技术标）`text_sim`/`key_sim` 归零 + 标记 `component_mismatch` + risk 重算
- **报告批注** `document_analysis_svc.py`：6.6 明细异组件对显示「异组件(不计)」+ 图例说明
- **实证**（真实 EPCM 招标 49216 字, k=98）：同组件正常技术标 0.89→**0.7695** <0.80 门槛；围标（技术雷同）**0.9784** 仍触发
- 回归 +4 测试（adaptive / tf_guard / component_mismatch / component_same）；全量 92/92；fix_registry 70/70

### FIX-013 中文停用词过滤 — 消除模板重叠误报
- 新增 `stop_words.py` `DEFAULT_STOP_WORDS`（~150 招投标/功能词）
- `tokenize_for_tfidf` 默认过滤；三条向量化路径统一接入
- **实证**：围标（技术雷同）cosine 0.98 vs 正常（技术不同）0.74，≥80% 门槛可判别
- 回归 88/88；fix_registry 66/66

### FIX-012 清标基线校准
- 工程类 2 投标人脱敏基线（`tests/fixtures/clearance_baseline/`）
- 无招标文件时 `text_sim` 指标**跳过**（模板去除不可用）
- 下调易误报指标权重；`test_clearance_baseline_scores` 锁快照防漂移；报告样本量免责（N<5）
- 回归 86/86；fix_registry 62/62

### FIX-011 报价尾数检测
- 尾数相同≥80% 检测（CSDN 第一信号）+ `extract_prices` 修复（_CN_PRICE 虚假 10000；双路去重）

### FIX-010 清标评分计量升级
- `total_score` 从裸加总（max~209）改为 **0-100 权重复合指数**（45 项 `INDICATOR_WEIGHTS` + score cap + text_sim 三指标去重）
- RiskScorer → 0.375 key + 0.375 attr + 0.25 text（去死图片权重）；text_sim ≥80% 门槛
- 预警阈值统一：≥60 高度 / ≥30 中等 / <30 正常
- 行业信号：报价等差/等比（`_detect_progression`）、Benford Nigrini 分级+卡方+Z、score_analyzer（Grubbs/Kendall W/Spearman）、关系社区检测、law_semantic 接入
- 回归 80/80；fix_registry 57/57

---

## [2026-08-28] — 清标结果移入聊天 + 报告生产级升级

### FIX-009 清标报告生产级升级
- **三节 continue 死代码修复**：`document_analysis_svc.py:494` 使整段渲染成死代码 → 45 项指标 6 行表全渲染
- **文本相似度恒 0 修复**：`_precompute_tfidf_for_files` 从未在清标路径调用 → 现预计算 + `tender_text` 模板去除
- **涉及指标数量恒 1**：`int(triggered_count>0)` 布尔 bug → 按指标 details 引用计数
- **开标信息表 + 评审标准**：新增 `clearance_openinfo.py`（Excel/CSV/JSON + 评审标准提取）→ 激活 14 指标
- 前端开标表上传 + `/clearance/preview_criteria` 预览可编辑
- 评分合理化：`keyword_overlap` 对 <4 关键词的短/模板文本返回低值
- 路由 372（新增 preview_criteria）；回归 72/72

### FIX-006 清标结果移入聊天
- 移除工具栏结果区 → 结果渲染进聊天（10 节可折叠富 HTML，含热力矩阵）
- `<!-- CLEARANCE_REPORT -->` + JSON 落库 `chat_messages`；重载时从 JSON 重建富 HTML
- 线程定向、下载入口统一到聊天气泡

### 全量审计合并入清标
- `audit_bp` 从 `register_blueprint()` 摘除，功能合并入清标 5 维度；`audit_runs`/`audit_config`/`audit_file_results` 表保留供 graph.py/cases.py 依赖

### FIX-005 Prompt 体系优化
- JUDGE_PROMPT / STRUCTURED_PROMPT 英文→中文；主 agent prompt「中联招标智能助手」
- 修复 `data/agent_prompt.json` 残留 `{"prompt":"Test prompt"}` 覆盖 bug
- 双 guard 去重、死代码清理

---

## [2026-08-28] — 安全加固 + 路由拆分 + 存储迁移

### 安全修复
- **C5** graph API 加 `@login_required` + 项目成员检查（FIX-2026-08-28-001）
- **C6** 管理员默认 PIN 生产 fail-closed（FIX-2026-08-28-002）；开发保留默认+告警
- **M3** 匿名存储迁移 PG JSONB `anon_chat_messages`，原子 UPSERT（FIX-2026-08-28-004）
- **M4** `credit_tasks` 内存共享 → Redis 注册表，跨 worker 可用（FIX-2026-08-28-003）

### 路由拆分（C1/C2/C3）
- `admin.py` 4,820→1,653 行（admin_regeneration / admin_knowledge_lab / admin_ops 子模块）
- `chat.py` 2,025→1,122 行（chat_files / chat_sessions / chat_config）
- `knowledge.py` 2,018→941 行（knowledge_notebook/company_kb/style/ingest/training + shared）
- 路由守护测试基线 expected_len=382

### 其他
- 二进制/私钥出库：`cert/key.pem`、`msedgedriver.exe` 移除跟踪
- `.gitignore` `*.json` 全局排除 → 定向规则 + 18 个跟踪 JSON 保留

---

## [2026-07-16] — Wiki 修复 + 建议引擎 f-string 修复

### Fixed
- `suggestion_engine.py`：4 处 f-string 语法错误（双引号闭合导致 `{name}` 变 set literal）→ `/timeline/:id/suggestions` 500 根因
- `app.js`：Wiki 编辑/删除按钮静默失败 → `data-edit-path`/`data-delete-path` 属性 + 单委托监听器

### Test
- 10/10 smoke + regression 通过；25/28 integration 因 hermes venv 缺 flask_limiter 失败（预存环境问题）

---

## [2026-07-15] — Timeline + Wiki 数据契约修复（app.js ~230 行）

### Added
- **Phase 8 Timeline Tab**：`allPanels`/`tabMap` 注册、时间线加载、里程碑表（planned/actual/diff）、状态徽章、HTML ~40 行
- 项目招标字段（bidding_category/bid_method）数据流贯通（modal/项目表/项目头）
- 文件状态列、版本历史状态切换（"设为正式"/"设为草稿"）

### Fixed
- Wiki Tab 数据契约：`statsData.data` → `statsData.stats`，`indexData.data.pages` → `indexData.pages`（5 处）
- 回收站侧边栏按钮缺 body/headers → 补上
- 流式消息重复：SSE done 更新 `_pollLastId`/`_lastKnownMessageId`；补工具栏/反馈/操作按钮
- 配置清空管理员可见性即时生效

---

## [2026-07-13] — 聊天渲染竞态 + Admin DB 429 缓解

### Fixed
- **Chat Render Race**（app.js 4 处）：sidebar onclick 异步 + `await loadSession()`；面板可见性守卫（隐藏→强制重载，可见→跳过）；`innerHTML=''` 前移；`isLoadingSession` 守卫 toast
- **Admin DB 429**：移除前端逐表 fallback 循环（消除 47 请求突发）→ 服务端 try/except + `COUNT(*)` fallback 单请求

### Test
- Smoke 6/6；Unit 103/103

---

## [2026-07-11] — LLM 自动 Fallback 链 + 全蓝图测试覆盖

### Added
- **`llm_fallback.py`**：7 步 fallback 引擎 + 熔断器（DEFAULT_CHAIN、degraded 检查、指数退避 cap 300s、thread-local 活跃 provider）
- `create_chat_model()` 接入 fallback；流式重启：服务器 `fallback_retry` SSE + 客户端重发（max 3）
- Runtime config：`llm_fallback_enabled`/`llm_fallback_chain`/`llm_fallback_cooldown_seconds`
- Admin UI 拖拽排序 fallback 链（provider+model）
- Nemotron 模型加入 NVIDIA provider
- **11 个蓝图全部有集成测试**（`tests/integration/`）

### Test
- Smoke 6/6；Unit 52/52（+14）；Integration(db) 45/45（+17）

---

## [2026-07-09] — 分类感知提取系统

### Added
- `CATEGORY_CONFIG`：每分类信号集/章节标题/文件名前缀 `[分类]name_skill.md`/`Category:` 头
- `category` 列入 `knowledge_lab_files` 和 `project_files`；贯穿上传/generate_skill 端点
- RAG 分类过滤（`retrieve()` 接受 `categories` → ChromaDB `$in`）
- skill_auditor 分类感知去重；上传超驰检测（`_check_skill_overlap()` + 合并建议对话框）
- Skill 编译器（DBSCAN 每分类主题聚类 + 复合 skill）；模板→文档生成器（`template_renderer.py`）

### Fixed
- JS TDZ bug：`syncActiveTabWithView()` try/catch、`pinnedSessions` 提升到文件级
- 移除损坏的 update hook；`loadSidebarDb()` 50ms 节流降 429

---

## [2026-07-08] — 统一文件管线 + 首次全审计

### Added
- **统一文件处理管线**：单一 `FILE_TYPE_REGISTRY`（44 类型）、分层提取（MinerU→MarkItDown→格式特定→OCR→LibreOffice）、`allowed_file` 校验
- 新增 4 个 pip 依赖：rarfile / py7zr / ebooklib / extract-msg

### Fixed（首次全审计 139 文件 / 5 发现）
- **HIGH** bare `except: pass` → `except OSError` + logging（admin cleanup）
- **MEDIUM** f-string SQL → 表/列白名单校验（admin.py, rag_engine.py, recycle_bin_service.py, skill_auditor.py）
- **LOW** 45+ 宽 `except Exception:` → 补 logger；knowledge.py f-string WHERE 消除
- **Critical 运维 bug**：过期系统 Python 进程占用 :5443 携带旧 ALLOWED_EXTENSIONS → taskkill 终止

---

## [2026-07-07] — NVIDIA LLM/VL 提供商 + 上传限制修复

### Added
- `ChatNVIDIA` 提供商（`langchain_nvidia_ai_endpoints`），模型 `z-ai/glm-5.2` → `moonshotai/kimi-k2.6`
- 多提供商 VL 模型（NVIDIA + SiliconFlow），VL 管理 UI（状态横幅/配置/测试）
- `POST /set_video_analysis` 端点 + 视频分析复选框

### Fixed
- SSE 重复守卫：NVIDIA 重发完整文本 → `full_text.startswith(chunk)` 去重
- 413 → `MAX_CONTENT_LENGTH` 500MB
- `g._streaming_agent` 缓存失效；`split_thinking_answer` 支持 6 种格式（双花括号 JSON 等）
- `audit_report.py` NameError → 导入提升到模块顶层

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
