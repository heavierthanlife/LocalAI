# Changelog

All notable changes to 中联招标智能助手.

**格式**：Keep a Changelog 风格（Added / Changed / Fixed / Removed），日期降序。
**维护约定**：每次功能升级/修复在顶部新增条目；合规相关改动必须带 `regression: 3/3 baseline passed` 验证（见 `AGENTS.md` §Session Operating Protocol）。
**详细迭代记录**：2026-08-28 起的完整工作记录见本文件（按日期降序）。

---

## [2026-09-24] — QA-Loop Round 029：owner 校验统一封装（FIX-066）

### Changed
- **owner 校验统一封装**（FIX-2026-09-24-066 / `UNRESOLVED-029`）：新增 `app/utils/helpers.py::load_task_for(task_id, user_id) -> (meta, status)`，`status ∈ {ok, missing, forbidden}`，内部封装 `TaskBus.get` + `task_owner_ok`，**查询异常归一为 `forbidden`（fail-closed）**。`clearance.py`（`/status`、`/stream`）、`tasks.py`（get/delete/cancel/stream）、`batch.py`（`plagiarism_status`）改用该 helper；`compliance.py::_task_forbidden` 改为薄封装委托。
- **行为变化**：上述端点在 TaskBus/Redis 异常时由 **500 → 403**（与 compliance 一致、可审计）。语义等价：`missing` 在 clearance/tasks/batch → 404，在 compliance → 放行（结果持久化超 TTL，FIX-060 模型）。`credit.py` 用自有 registry（非 TaskBus），不纳入。

### Fixed
- **回归（评审发现，合入前修复）**：`clearance_stream` 重构时误删局部 `from app.services.task_bus import TaskBus`，而函数体仍用 `TaskBus.subscribe` → `NameError`（500）。已补回 import，并加源码守护测试 `test_clearance_stream_imports_taskbus`。同时清理 `tasks.py` 死 import（`META_TTL`）、消除 `batch.py` 中 `status` 变量遮蔽（→ `task_status`）。

### Added
- 回归测试 6 个：`test_load_task_for_statuses`（ok/forbidden/missing/异常）、`test_clearance_status_fail_closed`、`test_tasks_get_fail_closed`、`test_batch_plagiarism_status_fail_closed`、`test_compliance_result_fail_closed`、`test_clearance_stream_imports_taskbus`。

### Notes
- FIX-033 / FIX-063 注册表检查随重构同步（`task_owner_ok(meta,user_id)` → `load_task_for(task_id,user_id)`；FIX-063 fail-closed 检查迁至 `helpers.py`）。`@code-reviewer` 复核 1 Critical（已修）→ 复确认 0 Critical/High。
- `UNRESOLVED-029` 置 `resolved`。

regression: 1/1 clearance baseline passed
168/168 regression · verify_fixes 293/0 · doc_drift 15/15 · check_system 131/0/134

---

## [2026-09-24] — QA-Loop Round 028：合规归属 fail-closed + task_id 白名单 + TaskBus Redis 重试

### Fixed
- **`/compliance/feedback` 越权写入（IDOR）**（FIX-2026-09-24-062，`app/routes/compliance.py`）：`submit_feedback` 从 JSON body 取 `task_id` 后直接 `_load_result` 并写反馈，无 owner 校验 → 任意登录用户可对他人 task 提交反馈（污染 LoRA 训练数据）。现加 `_task_forbidden(task_id)` 守卫（与 `get_result`/`get_rules` 同模型，返回 403）。
- **`_task_forbidden` 异常 fail-open**（FIX-2026-09-24-063）：归属校验在 `TaskBus`/`helpers` 抛异常时 `return False`（放行）。现改为 `logger.warning` + `return True`（fail-closed）；`meta=None`（含 Redis 不可用）仍按「元数据缺失→放行」处理，不影响合法用户。
- **合规结果 `task_id` 路径遍历**（FIX-2026-09-24-064）：`task_id` 可由 `/compliance/feedback` body 提供，拼入 `{task_id}.json` 可跨目录读写任意 `.json`。现加白名单 `^[A-Za-z0-9_-]{1,64}$`，`_load_result` 非法返回 `None`、`_save_result` 非法抛 `ValueError`；`rules_<uuid>` 合成串不受影响。
- **`TaskBus` Redis 首次失败永久禁用**（FIX-2026-09-24-065，`app/services/task_bus.py`）：`_get_redis` 失败后 latch `_redis=False`，而守卫 `if _redis is not None` 使其被永久缓存 → Redis 短暂不可用会让整个进程生命周期内 TaskBus（含 owner meta）恒 `None`。现记录 `_redis_last_try`，每 `_REDIS_RETRY_INTERVAL`(60s) 重试一次。

### Added
- 回归测试 4 个：`test_compliance_feedback_requires_owner` · `test_compliance_task_forbidden_fail_closed` · `test_compliance_load_result_rejects_traversal` · `test_task_bus_redis_retries_after_interval`。

### Notes
- QA-Loop Round 028（增量 `bf82dbb..1cdb28b`）定级：C2 High（IDOR）；C1/C3/Bonus Medium；C4/C5 降级 Low（防御纵深 / 入库前已 `[:500]` 截断）。`@code-reviewer` 复核 0 Critical/High（1 Medium 已修 docstring）。
- 视觉项因截图过期（2026-09-11）延期至下轮 COLLECT，记 **`UNRESOLVED-028`**（扩展 `tests/visual_regression.py` + 12 缺页 + 清标数据夹具）。
- 其他路由（`clearance.py`/`tasks.py`/`batch.py`）直接调 `task_owner_ok` 未统一封装，记 **`UNRESOLVED-029`**（复核 M2，非本轮引入）。

regression: 1/1 clearance baseline passed
164/164 regression · verify_fixes 287/0 · doc_drift 15/15 · check_system 131/0/134

---

## [2026-09-16] — S3/S4/S5 清理：UI 死引用 + 配置诚实性 + 文档漂移

### Fixed
- 前端「绑定时间线」按钮引用**未定义**的 `_switchSubTab('timeline')`（FIX-056 残留，点击即 `ReferenceError`）→ 移除该按钮及处理器（`static/js/knowledge-lab.js`）。
- `has_llm` 收敛为活跃供应商（`OPENROUTER_API_KEY`/`NVIDIA_API_KEY`）——原统计已退役的 DEEPSEEK/ZHIPU/QWEN/SILICONFLOW，可能误报。
- `LOG_LEVEL` 现真正生效（`app/config.py` 读取，原硬编码 INFO/DEBUG）。

### Changed
- **`MAX_CONCURRENT_UPLOADS` 去文档化**（从未实现）：移除 AGENTS/USER_MANUAL 宣称 + `check_system.py` 检查（清单 135→134）。
- 补依赖 `xlrd`（.xls 解析，原静默失败）与 `python-louvain`（社区检测，原静默降级）；`.env.example` 补 `LOG_LEVEL`。
- 文档诚实性：README 去「语义检索」（`rag_laws` 空转、实为关键词检索）/去「Tiptap」（已删）；ARCHITECTURE 修 `agent_prompt.json` 说法、beat 6→25、ChromaDB 4→5、导入点 21→27；SECURITY「70/70」→按实际；AGENTS/USER_MANUAL 清单 135→134；AGENTS API-only backlog 补 timeline(20)/compliance(23)。
- 回归 `test_agent_prompt_file_not_test_override`（读 gitignored 文件 → 全新克隆会报错）→ 改为断言硬编码默认提示词。
- `UNRESOLVED-026` 置 `resolved`（FIX-058 已实现其全部规格）。

### Notes
- 残余清理记 **`UNRESOLVED-027`**：audit 死模块/编排器外科删除、`review.js` docReviewPanel 死块、陈旧 DOM id、`run_ui_audit` TAB_TOUR 行、factory `typo_*` 键、Dockerfile torch 去重。

regression: 1/1 clearance baseline passed
160/160 regression · 170/170 含路由守护+冒烟 · verify_fixes 280/0 · doc_drift 15/15

---

## [2026-09-16] — S2 部署完整性：Beat 定时任务 + 数据卷 + 补表 + compose（FIX-061）

### Fixed
- **Docker 下 18 个维护任务从不执行**：`ENABLE_SCHEDULER=false` 关掉 APScheduler，而 Beat 仅 7 条 → 回收站/孤儿用户/下载令牌/陈旧任务/训练数据/月年报/RAG 健康等从不运行。现将 18 个函数注册为 Celery 任务（Task 对象仍可直接调用，APScheduler 路径不受影响），`beat_schedule` 7→25 条；并修复 `generate-weekly-report` 误传 `kwargs` 的 latent bug。
- **`company_kb_files/` + `knowledge_lab_files/` 数据丢失**：写在 `BASE_DIR` 且非卷挂载 → `--force-recreate` 即丢；compose 新增命名卷挂原路径。
- **schema 缺表**：`wiki_bookmarks`/`wiki_view_log`/`user_feedback`（fresh-deploy 会坏）+ `knowledge_lab_skills`（功能已坏，`ingest_pipeline` 写入）→ `database.py` 补 `CREATE TABLE IF NOT EXISTS`（含 ON CONFLICT 所需 UNIQUE）。
- **趋势准确率永久不可用**：`trend_service` 读无写入方的 `compliance_check_feedback` → 改读实际写入的 `compliance_feedback`。
- **`docker-compose.e2e.yml` 缺 laws/repo_data/factory 挂载**（E2E 会 0 法规）→ 对齐主 compose。

### Changed
- compose 三服务 env 用 `x-app-env` 锚点统一（消除 worker/beat 缺 key）；`celery-beat` 加 `--schedule=/app/data/celerybeat-schedule`（跨 recreate 持久化）。

### Added
- fix_registry FIX-2026-09-15-061（10 不变量）+ 回归 4 测试。

regression: 1/1 clearance baseline passed
160/160 regression · 170/170 含路由守护+冒烟 · verify_fixes 280/0 · doc_drift 15/15

---

## [2026-09-16] — S1 安全修复：项目文件越权 + 合规结果归属 + 验证码脱敏（FIX-060）

### Fixed
- **项目文件/归档越权（IDOR）**（`app/routes/admin.py`）：`get_file_versions`、`download_project_file`（含 version 路径）原校验 `project_id` 却按 `file_id` 查询 → 跨项目文件访问；现查询 `JOIN project_files` 并加 `AND project_id=%s`。`download_archive` 原仅校验登录 → 加 `is_admin()/_can_access_project` + 文件名白名单（防 `..`/`\` 穿越）。
- **合规结果/规则越权**（`app/routes/compliance.py`）：`get_result`/`get_rules`/`update_rules` 仅校验登录；新增 `_task_forbidden()`（TaskBus meta + `task_owner_ok`，meta 缺失按 legacy 放行）→ 仅 owner 可读写。
- **注销验证码明文入审计日志**（`app/routes/admin_regeneration.py`）：`code_sent_{code}` → `code_sent_****`。

### Docs
- AGENTS「资源归属模型（FIX-060）」：live task=owner；batch/credit 归档 list+download 全用户（单机构有意）。

regression: 1/1 clearance baseline passed
156/156 regression · 166/166 含路由守护+冒烟 · verify_fixes 270/0 · doc_drift 15/15

---

## [2026-09-16] — 仓库卫生 + Docker 数据资产规则文档化

### Fixed
- **停止跟踪运行时产物**：`data/user_files/**`（2 docx + 4 html 用户上传件）与 `data/checkpoints.db-shm`/`-wal`（SQLite WAL）—— 二者早已在 `.gitignore`，但「忽略不溯既往」已 tracked 需 `git rm --cached`（工作区文件保留）。`.gitignore` 补 `data/checkpoints.db-*`。保留 `data/qa_loop/audit/final_state.png`（审计证据）。**不重写历史**（未 force-push）。

### Docs
- `AGENTS.md` Gotchas 增「Docker 数据资产三规则」：① 随代码演进 → `:ro` 直挂；② 运行时可变种子 → 启动 seeding；③ 纯运行态 → `app_data` 卷。附 `.dockerignore` 丢整个 `data/` 的告警与 `repo_data:ro` 暴露说明。

regression: 1/1 clearance baseline passed（无清空/合规代码路径变更）

---

## [2026-09-16] — Docker 数据资产统一挂载 + 启动 seeding（FIX-059）

### Fixed
- **Docker 数据资产「0 加载」根因治理**：`.dockerignore` 丢弃整个 `data/` 且无启动 seeding，导致每个 repo 数据资产都需手工挂载、易静默为空（FIX-058 的 laws 即此坑）。本次：
  - compose 为 app/celery-worker/celery-beat 增加 `./data:/app/repo_data:ro`（通用 seed 源）；
  - 补 `./data/runtime_config_factory.json:/app/data/runtime_config_factory.json:ro` —— 此前 Docker 下该出厂基线缺失，「恢复出厂设置」失效（退化为硬编码 `DEFAULTS`）；
  - 新增 `app/bootstrap.py::ensure_seeded()`：`domain_words.txt`（运行时追加写，不可 :ro）在卷缺失时**原子复制**进 `/app/data`，由 `create_app()` 调用（覆盖 app + celery worker/beat）；容器内 jieba 领域词典恢复加载。

### Added
- `app/bootstrap.py`；`fix_registry FIX-2026-09-15-059`（6 不变量）；回归 `test_bootstrap_ensure_seeded` / `test_bootstrap_wired_and_compose_mounts`。

### Notes
- 卷集合（3 服务一致）：`app_data` + `repo_data:ro` + `laws:ro` + `industry_words:ro` + `runtime_config_factory.json:ro` + EasyOCR/cert。`laws`/`industry_words` 保持直挂（随代码最新，不走 seeding 以免静默过期）。
- `domain_words.txt` 卷内副本会随「审批领域词」追加而漂移于 repo 基线（预期行为）。

regression: 1/1 clearance baseline passed
153/153 regression · 163/163 含路由守护+冒烟 · verify_fixes 263/0 · doc_drift 15/15

---

## [2026-09-16] — 合规法规库扩展：接入 11 部国家级法规全文 + 核心 4 部守卫（FIX-058）

### Added
- **合规引擎法规池 4 → 15 部 / 619 条**（`app/services/compliance_checker.py`）：`_load_seed_laws` 合并 seed 核心 4 部 + `extended_laws.json` 中 `scope=national` 且 `source_url` 完整的 11 部（去重、核心优先）。新增：政府采购法实施条例(79) · 工程建设项目施工招标投标办法(92) · 评标委员会和评标方法暂行规定(62) · 必须招标的工程项目规定(6) · 政府采购货物和服务招标投标管理办法(88) · 政府采购非招标采购方式管理办法(62) · 政府采购质疑和投诉办法(45) · 电子招标投标办法(66) · 政府采购促进中小企业发展管理办法(25) · 公共资源交易平台管理暂行办法(48) · 招标投标违法行为记录公告暂行办法(21)。全部溯源自 www.gov.cn 官方公报/政策库，条文数与法定一致（详见 UNRESOLVED-026）。
- **核心 4 部守卫**（`_select_relevant_laws`）：`max_laws=15` 截断后，若某核心法（招标投标法/实施条例/政府采购法/民法典合同编）有相关条文却被挤出，则用其最高分条文替换末位非核心法，确保基础法条不被新增规章挤出。

### Changed
- **README 法规库口径 4 → 15 部**；`scripts/check_doc_drift.py::_count_laws` 口径同步为「已加载法规数」=15。
- **`docker-compose.yml` 修复潜在生产缺陷**：为 app/celery-worker/celery-beat 增加 `./data/laws:/app/data/laws:ro` 只读绑定——`/app/data` 是 `app_data` 卷，会遮蔽镜像内 `data/laws`，**此前 Docker 下合规引擎加载 0 部法规**。修复后容器内 `_get_seed_laws()` 返回 15 部/619 条。
- `data/fix_registry.yaml`：`FIX-2026-09-15-058`（6 不变量）；回归新增 `test_compliance_law_pool_expanded` / `test_compliance_core_law_guard_wired`。
- `scripts/fetch_extended_laws.py`：解析器加行首约束（剔除正文内交叉引用）+ law-name 校验守卫（错页拒绝）。

### Notes
- 余 5 部未接入：38号/119号（未收录 gov.cn 政策库）· 规范招标投标主体行为若干意见（非条文结构）· GB50500（要点摘录）· 广东省办法（local 排除接线）。

regression: 1/1 clearance baseline passed
151/151 regression · 161/161 含路由守护+冒烟 · verify_fixes 256/0 · doc_drift 15/15

---

## [2026-09-16] — 文档诚实性整改：生成器硬编码 + 法规库口径 + 陈旧文档/快照（FIX-057）

### Fixed
- **`scripts/check_system.py` 头部硬编码计数**：原写死 `verify_fixes.py: 21/21` 与 `regression tests: 16/16`，每次重新生成 `SYSTEM_CHECKLIST.md` 都写回数月前的旧值。改为动态：`_verify_fixes_summary()` 实跑 `verify_fixes.py` 解析真实通过数，`_status_totals()` 汇总本次检查项 → 头部现值 `verify_fixes.py: 250/250 pass (0 failure(s))` / `135 items — 131 pass / 4 manual / 0 fail`。
- **`README.md` 法规库口径**：「法规库（24 部）」失真 —— 引擎实际仅加载 `data/laws/seed_laws.json` 的 **4 部**；`extended_laws.json` 的 20 部（前 4 部与 seed 逐条重复，真实并集 = 20）从未接入。改为「核心法规库（4 部国家级基础法规：招标投标法 / 实施条例 / 政府采购法 / 民法典合同编）+ 扩展库 16 部待补齐条文后接入」。
- **`docs/USER_MANUAL.md`**：供应商 key 由陈旧的 `DEEPSEEK/ZHIPU/QWEN/SILICONFLOW_API_KEY` 改为当前活跃的 `OPENROUTER_API_KEY / NVIDIA_API_KEY`；清标入口 `POST /clearance` 更正为 `POST /clearance/run`。
- **`AGENTS.md` Long-term Backlog**：移除条目 1（清标任务归属校验）与条目 2（警示详情前端渲染）—— 二者已分别由 FIX-033（`task_owner_ok`，`app/utils/helpers.py:48` + clearance/batch/credit/tasks 全部调用）与 FIX-035（`_renderWarningDetails`，`static/js/app.js:9512`）完成，属陈旧陈述；保留条目 3（API-only 端点清单，9 端点已验证存在）。
- **`data/current_state.yaml`**：标注 `deprecated: true`（2026-07-30 快照，fix_registry/回归/unresolved 计数与 `known_bugs` 均已过期；三项 known_bugs 均已由 FIX-013/014 及 credit_bp 注册修复）。
- **`tests/fixtures/clearance_baseline/scores.json`**：34 条 skip 文案由旧「○ 需外部数据源（交易平台/评标系统数据）」对齐至 FIX-053 的「○ 需交易平台数据（当前不可用）」（`document_analysis_svc.py:608`）。

### Changed
- **`scripts/check_doc_drift.py`**：新增第 6 指标 `laws`（`_count_laws()` 读 `seed_laws.json` 去重 `law_name`，口径=已接入引擎的法规数），`README.md` 由正则 `法规库[（(](\d+)\s*部` 监控 → 15 claims / 6 metrics。
- `data/fix_registry.yaml`：`FIX-2026-09-15-057`（9 个不变量）；回归 `tests/test_regression.py` 新增 5 个测试（check_system 无硬编码 / doc_drift laws 指标 / 文档诚实性 / current_state deprecated / scores.json 文案）。

### Notes
- **批次 B（待执行，data-blocked）**：`extended_laws` 16 部条文补齐后接入（仅 `scope=national` 19 部 + 核心 4 部始终包含守卫 + `versions[].articles` 归一化），已记 `UNRESOLVED-026`。

regression: 1/1 clearance baseline passed（本批无清标算法改动）
149/149 regression · 159/159 含路由守护+冒烟 · verify_fixes 250/0 · doc_drift 15/15 · check_system 131/0 · run_tests 28/28

---

## [2026-09-15] — P1-B1 路由/前端降级：删 /audit + compliance/tiptap/timeline API-only（FIX-056）

### Removed
- **`/audit` 蓝图（10 条路由，全死）**：唯一前端消费者 `bid-audit.js` 从未被 `index.html` 加载。删除 `app/routes/audit.py` + `__init__.py` 注册块 + `tests/integration/test_audit.py` + `static/js/bid-audit.js`。审计引擎 `app/services/audit_engine.py` 保留（清标在用）。
- **compliance UI（API-only 降级）**：`compliance.js` 的 23 个 DOM id 在 `index.html` 全部不存在（加载即空转）；`tiptap-editor.js` 仅服务 `#complianceTiptapEditor`。两文件删除 + `index.html` 摘除 script 标签。后端 compliance 路由保留。
- **timeline UI（API-only 降级）**：`index.html` 时间线子标签 + `#timelinePanel` 删除；`app.js` 的 Timeline Tab 全部前端逻辑/接线删除（`loadTimelinePanel`/`_renderTimelineList`/`_loadTimelineDetail`/`_setupTimelineCreationForm`/`_wireTimelineDetailActions` 等）。后端 timeline 蓝图保留。

### Changed
- `scripts/check_system.py`：`audit.py` grep 检查改指向 `app/services/audit_engine.py`。
- **路由集 401 → 391**（仅移除 `/audit` 10 条）：`tests/fixtures/routes_snapshot.json` 重生成；`test_route_preservation.py` `expected_len` 401 → 391。
- **蓝图数 17 → 16**：`README.md` / `AGENTS.md` / `docs/ARCHITECTURE.md` 同步；`repair_kit/SYSTEM_CHECKLIST.md` 重生成（135 项）。
- `data/fix_registry.yaml`：`FIX-2026-07-19-002` 检查改指向 `app.js::_safeHTML`；`-003`/`-004` 标 superseded（checks 清空）；`FIX-2026-09-09-023` audit 检查改为断言未注册。
- 确认 `credit`（9 端点）/ `LoRA`（3 端点）本就是 API-only（前端零引用），无需改动。

### Added
- fix_registry `FIX-2026-09-15-056`（10 项 invariant）；回归 `test_audit_blueprint_removed` / `test_timeline_api_only_downgrade` / `test_compliance_frontend_removed`（原 tiptap/taskids 两个 superseded 测试替换）。

### regression: 1/1 clearance baseline passed（无清标算法改动）
143/143 regression · 152/152 含路由守护+冒烟 · run_tests 28/28 · verify_fixes 241/0 · doc_drift 14/14 · audit_js_routes no_route=0

---

## [2026-09-15] — P1-A 死代码清理 + LoRA registry schema 对齐（FIX-055）

### Removed
- **零引用模块**：`app/services/agent_middleware.py`（导入不存在的 `langchain.agents.middleware`，全项目零引用）、`app/services/_save_helper.py`、`app/services/region_manager.py`（`law_regions`/`law_region_bindings` 空置，地方法规接入=P2）。
- **`auth_jwt.py` 未使用函数**：`jwt_required` / `jwt_optional` / `decode_token`（JWT 只签发不校验，零调用）。保留 `create_token`（`chat_sessions.py` 在用）。
- **`law_semantic.py::rebuild_law_index`**：零引用，且是唯一能填充 `rag_laws` 索引的函数（故 `semantic_law_search` 实际恒返回 `[]`）。保留 `semantic_law_search`（`compliance_checker` 调用，空索引时优雅返回）。

### Fixed
- **LoRA registry schema 三处对齐**：`nightly_trainer._update_adapter_registry` 原先写顶层键 `"compliance_checker"` + 字段 `adapter_dir`，而读取方 `llm_provider._load_industry_models` / `lora_trainer.get_adapter_info` 期望 **顶层键 = industry** + 字段 **`adapter_path`**（规范写入方 `scripts/run_lora_training.py::_register_adapter`）→ 夜间训练的适配器永不被加载。改为写 `registry[industry] = {adapter_path, base_model, registered_at, active, source}`；`_run_lora_training` 成功结果补 `industry` 字段供传递。

### Changed
- `docs/ARCHITECTURE.md`：删 `agent_middleware.py` 宣称；Services 计数 101 → 98。
- `docs/SECURITY.md`：认证条目改为「session 会话 + JWT 仅签发」；删 InvalidToolGuard 宣称。
- `data/unresolved.yaml`：`UNRESOLVED-016`（agent_middleware）标记 resolved。

### Added
- fix_registry `FIX-2026-09-15-055`（12 项 invariant）；回归 `test_dead_service_modules_removed` / `test_auth_jwt_only_issues_tokens` / `test_law_semantic_rebuild_removed_but_search_kept` / `test_nightly_adapter_registry_schema_matches_readers`（含功能性 schema 校验，+4）。

### regression: 1/1 clearance baseline passed（无清标算法改动）
143/143 regression · 153/153 含路由守护+冒烟 · verify_fixes 233/0 · doc_drift 14/14 · routes 集合未变（401 不变）

---

## [2026-09-15] — P0 空转修复：hasLLM / _safeHTML / 更多下拉 / fallback 文档降级（FIX-054）

### Fixed
- **清标「AI 评审」恒关（①）**：`hasLLM` 唯一写点位于 `app.js` 的 `loadAccountModal`，但被后加载的 `accounts.js` 同名函数覆盖且未带该行 → `hasLLM` 恒为旧值/空。在 `accounts.js:38` 的 `loadAccountModal` 补回 `sessionStorage.setItem('hasLLM', authData.has_llm ? 'true' : 'false')`。
- **移动端「更多」下拉恒空（⑩）**：CSS 设计为「first 4 visible, tabs 5+ into more」，但 `index.html` 第 5 个主标签 `analyticsTabBtn`（系统管理）缺 `admin-tab` 类 → `app.js` 收集器 `#tabBar .admin-tab` 与 `updateMobileLayout` 均匹配不到元素。补类对齐。**偏离 handoff 字面**（原计划改 `app.js` 选 `.tab-btn`，会把 5 个标签含「智能对话」全部藏进「更多」，属 UX 回退；已与用户确认采用补给第 5 标签加类方案）。
- **XSS 兜底缺失（⑬）**：`_safeHTML` 定义于 `compliance.js`，但 `chat.js:8/1005` 依赖它，缺省时回退原始 `innerHTML`。将 `_safeHTML` 迁到 `app.js` 作安全全局；`compliance.js` 内部 2 处调用改走全局；`chat.js` 注释同步更新。
- **DOMPurify 本地 vendor 404**：`index.html` 的 DOMPurify `onerror` 指向不存在的 `static/js/purify.min.js`，删除该 `onerror`（CDN 失败时由 `_safeHTML` 纯文本转义兜底）。

### Changed
- **region_code 现状注释（⑤）**：`compliance_checker.py` 的 `check()` 签名与 docstring 标注「当前未生效：law_regions/region_manager 空置，地方法规接入=P2」。不改签名、不改 6 处调用。
- **LLM fallback 文档降级（②）**：删 `README.md` / `docs/ARCHITECTURE.md` 中「fallback 链 + 熔断器」宣称（`llm_fallback.py` 未接线，仅 tests 引用），改为如实描述单供应商直连；`AGENTS.md` 的 fallback 条目同步修正。

### Added
- fix_registry `FIX-2026-09-15-054`（8 项 invariant）；回归 `test_hasllm_written_by_accounts_loader` / `test_mobile_more_fifth_tab_marked_admin` / `test_safehtml_global_in_app_not_compliance` / `test_dompurify_no_local_vendor_onerror` / `test_region_code_documented_inactive` / `test_llm_fallback_claims_downgraded`（+6）。`test_compliance_xss_sanitization` 改为断言全局定义在 `app.js`。
- `data/unresolved.yaml` `UNRESOLVED-018`（llm_fallback 未接线 backlog）。

### regression: 1/1 clearance baseline passed（无清标算法改动）
139/139 regression · verify_fixes 221/0 · doc_drift 14/14

---

## [2026-09-15] — 移除 Headroom/Kompress + 休眠指标文案明确化（FIX-053）

### Removed
- `requirements.txt` 移除 `headroom-ai==0.27.0`、`magika==0.6.3`、`onnxruntime==1.20.1`。实测 headroom 对**中文招标文本零压缩**（Kompress 为英文模型 + `ContentRouter` 上下文压力低时 `min_ratio≈0.85` 直接 `router:noop` + 单条消息受默认 `protect_recent=4` 保护），且其 Kompress/ModernBERT 模型占用 HF 缓存 ~0.85GB。`app/utils/headroom_utils.py` **保留为 soft no-op**（`import headroom` 失败 → `_available=False` → `compress_*` 原样返回），`chat.py`/`agent.py` 调用点不改 → **零回归**。

### Changed
- **休眠指标文案**：34/45 `checker:'skip'` 指标的占位串由「○ 需外部数据源（交易平台/评标系统数据）」改为「**○ 需交易平台数据（当前不可用）**」，并追加各自 `skip_reason`（如"无标段分组输入"）→ 用户可明确这是**平台对接缺失**而非系统故障。
- `docs/ARCHITECTURE.md` 清标章节新增「**休眠指标分类**」：13 项可由"开标信息表/评审标准"激活，21 项恒不可用（依赖交易平台/评标系统）。

### Added
- fix_registry `FIX-2026-09-11-053`；回归 `test_headroom_removed_and_skip_label`。

### regression: 1/1 clearance baseline passed（无清标算法改动）
133/133 regression · verify_fixes 213/0 · doc_drift 14/14

---

## [2026-09-15] — Docker HF 模型缓存持久化（FIX-052）+ Headroom 实测

### Changed
- **Docker HF 缓存持久化**：`docker-compose.yml` 给 `app` / `celery-worker` / `celery-beat` 增设 `HF_HOME=/app/data/hf_cache`（+ `HF_HUB_DISABLE_SYMLINKS_WARNING=1`），模型缓存落到既有持久 `app_data` 卷；`Dockerfile` 预建 `hf_cache` 目录。此前容器无 HF 缓存，sentence-transformers / Headroom-Kompress 模型会下到临时层、`--force-recreate` 后丢失并重复下载。现跨 recreate 保留（app 与 worker 共享）。

### Findings（实测，无代码改动）
- **Headroom 基本不压缩**：`headroom-ai 0.27.0` 已接入（`chat.py`/`agent.py`，`runtime_config.headroom_enabled` 默认 True），但实测 `ContentRouter` 在上下文压力低时阈值 `min_ratio≈0.85` → 短/中文本直接 `router:noop`（host Windows 与容器 Linux 均 `saved=0`）；`compress_file_content`/`compress_search_results` 传单条 message 还受默认 `protect_recent=4` 保护；Kompress 为英文模型。结论：其拉取的 Kompress ONNX + ModernBERT tokenizer 模型大多不产生 token 节省。已在 `AGENTS.md` Gotchas 记录。

### Added
- fix_registry `FIX-2026-09-11-052`（HF_HOME 持久化）；`AGENTS.md` 环境变量表新增 `HF_HOME`。

### regression: 1/1 clearance baseline passed（无代码路径改动）
132/132 regression · verify_fixes 208/0 · doc_drift 14/14

---

## [2026-09-11] — 运行配置 LLM 选择器纳入自定义提供商（FIX-051）

### Fixed
- **自定义 LLM 提供商"能加不能用"（FIX-051）**：`/admin/runtime_config_schema` 的 `LLM 服务商` / `LLM 模型` 下拉选项只从 `PROVIDER_CONFIG`（内置 openrouter/nvidia）构建 → 管理员新增的自定义 provider **无法在运行配置面板选中**。改为从 `get_merged_provider_config()` 构建（含自定义 provider + 其模型），自定义项名称标「（自定义）」。选中后现有 change 处理器自动切换模型下拉；保存后 `active_llm_provider/active_llm_model` 生效（LLM 调用经 `get_provider_config` + `env_store.get_env` 读 key）。
- **空模型提示（Low）**：`review.js` 中所选 provider 无模型（如保存时 `/models` 拉取失败）时，在「刷新模型」旁提示「无模型：请点「刷新模型」或检查 base_url / API Key」。

### Added
- fix_registry `FIX-2026-09-11-051`；回归 `test_custom_provider_selectable_in_runtime_schema`。

### regression: 1/1 clearance baseline passed
132/132 regression · verify_fixes 206/0 · doc_drift 14/14 · node --check OK

---

## [2026-09-11] — LLM 提供商管理补齐：key 落 .env + 自动拉模型（FIX-049/050）

### Added
- **`app/services/env_store.py`**：`write_env_var()` 以 `KEY=value` **原子 upsert** 并同步 `os.environ`；**双写** `data/llm_provider_keys.env`（持久，Docker `app_data` 卷）与根 `.env`（存在时，本地开发）；`load_provider_keys()` 启动加载（`app/__init__.py`；celery worker 经 `create_app` 覆盖）。**多 worker 修正**：`get_env()`/`has_env_var()` 按 mtime **懒加载**该文件（override），未处理保存请求的 gunicorn worker 也能读到新 key；LLM key 读取点（`llm_provider` / `llm_fallback` / `llm_catalog` / `chat_config` / `admin_regeneration`）统一改用 `env_store.get_env`。
- **保存即生效**：`update_runtime_config` 保存 `llm_custom_providers` 时，将字面 `api_key` 写入 env 存储（**明文永不落 runtime_config.json**，条目只留 `api_key_env`），并用刚写入的 key **对每个自定义 provider 调 `/models` 回写模型列表** + 清 agent 缓存 → 模型下拉无需手动刷新。
- **UI（review.js）**：自定义 Provider 行编辑器新增 **API Key 密码框**（只写不回显）+ ✓/✗ **key 状态角标**（来自 `/admin/llm_providers.api_key_set`）+ 使用说明；dirty 跟踪纳入 `api_key`。

### Changed
- `/admin/llm_providers` 每 provider 增 `api_key_set` / `custom`。
- `docs/ARCHITECTURE.md` 服务计数 100→101（新增 env_store）。

### regression: 1/1 clearance baseline passed
129/129 regression · verify_fixes 203/0 · doc_drift 14/14 · node --check OK

---

## [2026-09-11] — 删除账户功能断裂修复（FIX-047）

### Fixed
- **删除账户按钮点击无响应（High，FIX-047）**：UI 走查发现 `accounts.js` 调用 `createQuickModal` 抛 `ReferenceError`。根因：`app.js` 的 `createQuickModal` 为**闭包内声明**（不在 `window`），而 `accounts.js` 作为独立 `<script defer>` 无法访问 → 删除账户流程整体断裂。同时 `app.js:915` 与 `accounts.js:88` **双重绑定** `deleteAccountBtn`，一次点击触发两次 `requestDeleteAccount`（双请求/双弹窗）。
  - 修复：`app.js` 暴露 `window.createQuickModal / escapeHtml / showToast`（后两者原本已是全局，一并显式化以防回退）。
  - 去重：删除 `app.js` 的 legacy `requestDeleteAccount` / `deleteAccount` / `showConfirmDeleteModal`（共 121 行，与 `accounts.js` 逐字重复）及 `deleteAccountBtn` 绑定 → 收敛为 `accounts.js` 单一事实源。

### Added
- fix_registry `FIX-2026-09-11-047`；回归 `test_delete_account_global_helper_single_binding`。

### regression: 1/1 clearance baseline passed
127/127 regression · verify_fixes 195/0 · doc_drift 14/14 · node --check OK

---

## [2026-09-11] — 清标假警报根治：重点信息雷同 + 报价异常降级（FIX-045/046）

### Fixed
- **「重点信息雷同」假警报（High，FIX-045）**：两不相关标书必然共享「公司/工作/检查/食品/填写/偏离/提供/负责」等通用词，而 N=2 时共有词 df=2、TF-IDF 无区分力，原 `build_key_info_matches` 交集非空即输出 → 6.7 表假警报 + `file_scores` 虚高。修复三处：
  1. `DEFAULT_STOP_WORDS` 扩容 ~40 个通用业务/名词（`stop_words.py`）。
  2. `extract_keywords` 新增可选 `pos_filter`，经 `_keep_nounish`（jieba.posseg）剔除动词/形容词；**默认关闭**（短文本可能跌破 key_sim 的 4 词下限），仅 key-info 展示路径启用。
  3. `build_key_info_matches(pairs, extra_stop_words, template_text)` 改用与 key_sim 一致的预处理/模板去除文本，且仅当 **共有词 ≥3 且 Jaccard ≥0.15** 才输出该对（否则 6.7 表自动隐藏）。
- **报价异常不可靠假警报（High，FIX-046）**：无开标信息表时 `quote` 指标从正文正则抽价（日期/行项目被当报价），N=2 统计意义弱，却仍计分并显示「发现 N 个投标单位报价疑义」。改为：未注入权威开标报价（无 `【开标报价】` 标记）时 `score=0`、result 标注「○ 无结构化开标报价，报价异常仅作参考（建议上传开标信息表）」，与 text_sim 无招标文件时的诚实降级一致；`file_scores` 同步不计入，避免「重点嫌疑单位」排名被不可靠报价抬分。

### Changed
- 关键词/停用词扩容使 TF-IDF 余弦轻微漂移，`tests/test_batch_orchestrator.py` 快照重标定（0.1462→0.1489、0.7774→0.7717；slight_diff 0.4→0.3571；near_identical/no_overlap 不变，判别力保留）。

### Added
- fix_registry `FIX-2026-09-11-045`/`046`；回归测试 4 项（通用词不误报、显著对才输出、报价降级、停用词扩容）。

### regression: 1/1 clearance baseline passed（仅工程类校准）
126/126 regression · 23/23 collusion+batch · verify_fixes 191/0 · doc_drift 14/14

---

## [2026-09-11] — QA 全面轮次 022：安全/稳定性加固（FIX-037~043）

### Fixed
- **合规异步任务必崩（High，FIX-037）**：`compliance_check_task` 调用 `checker.check(..., region_code=region_code)`，但 `ComplianceChecker.check()` 无该形参 → 每次 `TypeError`、任务必失败。为 `check()` 增加 `region_code: str = None`。
- **征信任务端点越权（High，FIX-038）**：`credit_check_status/resume/get_captcha_image/reload_captcha/solve_captcha/download_credit_report` 仅校验 `consent`，任意登录用户可读/改/下载他人征信任务。建任务时写入 `user_id`，读改端用 `task_owner_ok` 比对；`download` 回退查 `credit_check_reports.user_id`（owner-only）。
- **前端 XSS（High，FIX-039）**：`escapeHtml` 未转义引号 → ~80 处属性上下文可属性注入。补 `"`→`&quot;`、`'`→`&#39;`；`owner`/`username`/`uploaded_at` 等一律 `escapeHtml`；管理员项目列表内联 `onclick`（拼接 name/status 可绕过）改为 `data-*` + `openProjectFromEl` 委托。
- **清标结果可能整体丢失（High，FIX-040）**：`clearance_engine` 原在结果事务内 INSERT `chat_messages`，单条失败使事务 aborted → `commit` 失败被外层 `except` 吞掉、`batch_comparison_results` 一并丢失。改为结果先 commit，聊天气泡在独立连接/事务写（失败仅告警）。
- **账户删除重复 deposit（High，FIX-042）**：`auth.py` 在选择性 deposit(keep_map) 后又无条件全量再插一遍 → 保留项重复、非保留项也被 deposit。删除无条件块。
- **会话 Cookie 缺安全标志（High，FIX-041）**：补 `SESSION_COOKIE_HTTPONLY=True` / `SESSION_COOKIE_SAMESITE='Lax'`（`SECURE` 于 `APP_ENV=production` 开启）。
- **Docker 调度重复（High，FIX-041）**：compose `app` 服务补 `ENABLE_SCHEDULER=false`，避免 4 个 gunicorn worker 各跑一份 APScheduler。
- **SSE 越权 CORS（Medium，FIX-042）**：`/tasks/<id>/stream` 移除 `Access-Control-Allow-Origin: *`。
- **上传去重 TOCTOU（Medium，FIX-043）**：`file_store` 去重 SELECT 与 INSERT 分属两连接，并发可重复插入。改用 `pg_advisory_xact_lock` 串行化（无 schema 变更）。
- **输入校验/加固（Medium，FIX-043）**：`delete_law` 的 `law_id` 加白名单校验；`graph` threshold 改 `type=float`（防 500）；账户删除验证码改 `secrets.randbelow`。
- **风格分原始标签泄露（Medium，FIX-044，视觉复核新增）**：全量审计补充检查的风格分显示「75.0 分 (unknown)」，前端在标签为空/`unknown` 时不再输出括号内容。

### Changed
- `database.py` 的 `ADMIN_PIN` 默认值与 `__init__.py` 对齐（`'888888'`→`'123456'`，FIX-042）。
- compose 移除已知默认 secret（`FLASK_SECRET_KEY`/`WTF_CSRF_SECRET_KEY` 改 `:?` 必填）；`postgres`/`redis` 端口绑定 `127.0.0.1`（FIX-041）。
- 错别字遗留表 `typo_detection_results` 的 `DROP` 从每次启动移入一次性迁移 `migrations/002`（FIX-043）。
- `chat.js` 轮询清理改 `clearTimeout` + generation/stopped 守卫（防 stop 后 in-flight 续命）。

### Added
- fix_registry `FIX-2026-09-11-037` ~ `043`；回归测试 11 项（`tests/test_regression.py`）。

### regression: 121/121 regression · route_preservation 3/3 · smoke 7/7 · verify_fixes 185/0 · doc_drift 14/14

---

## [2026-09-11] — 清标持久化事务化（FIX-036）

### Fixed
- **清标结果持久化原子性（High，FIX-036）**：`_persist_clearance_review` 原先先 `DELETE` 三表（`quote_anomaly_results` / `entity_relationships` / `relationship_risk_summary`）并 `commit`，再分别调用两个 `save_*`（各自连接、各自 commit）。DELETE 提交后若任一 `save_*` 失败，任务将被清空且无新数据落库（数据丢失窗口）。改为**单连接单事务**：DELETE + `save_quote_anomaly_results(..., conn=conn)` + `save_relationship_results(..., conn=conn)` + 一次 `commit`。两 `save_*` 新增可选 `conn` 参数（默认 `None` → 自持事务并 commit），外部 4 处调用点（`batch.py`×2 / `document_analysis_svc.py`×2）不变。

### Changed
- `data/qa_loop/last_head` 修正为 `13f5788`（原为无效字面量 `round-021`）。

### Removed
- 清理 26 张过期 Playwright 截图（`.playwright-mcp/`，2026-09-02/04）与过期 `tests/visual_screenshots/manifest.json`（待重拍再生）。

### regression: 110/110 regression · route_preservation 3/3 · smoke 7/7 · verify_fixes 164/0 · doc_drift 14/14
> 清标基线：`regression: 1/1 baseline passed`（仅工程类校准，`test_clearance_baseline_scores` 通过；货物/服务缺真实文档 → UNRESOLVED-017）。

---

## [2026-09-11] — 清标开放案例清零：任务归属/大小守卫/警示指引（FIX-033~035）

### Fixed
- **异步任务归属校验（High，FIX-033）**：新增 `helpers.task_owner_ok`（legacy 缺 `user_id` → allow+log）。`user_id` 现于 `register_queued` 写入（clearance/plagiarism）。全部 TaskBus 读端点加归属守卫：`/clearance/status|stream`、`/tasks/<id>`、`/tasks/<id>/stream`、`/tasks/<id>/delete`、`/tasks/<id>/cancel`、`/batch/plagiarism/status`。`/tasks` 列表：**匿名返回空、登录仅返回本人**（此前匿名可见全部任务、登录可见所有人任务）。
- **`compliance_check_task` 修复**：`TaskBus()` 无参构造 + `start()` 三位置参数（TypeError，且会写 `task_meta:None`）→ 改为 `TaskBus(task_id, 'compliance_check', ...); bus.start()`。
- **警示详情与处理指引（Medium，FIX-035）**：DOCX 有 5 表、前端仅有计数。后端 `annotate_warning_details()` 在 `hard_evidence.items/violations` 附加 `label/chapter/guidance`（复用 `_type_label`/`_CHAPTER_BY_TYPE`/`_guidance_for`，单一数据源）；`run_analysis` 与 `clearance_engine` 两处 reassessment 均调用；前端 `_renderWarningDetails` 渲染两张表。

### Added
- **同步对比端点大小守卫（Medium，FIX-034）**：`_reject_oversize()` — `/check_quote_anomaly`、`/compare_bidders_quotes`、`/extract_relationships` 超 `MAX_SYNC_COMPARE_MB`(40MB) 返回 413，引导用异步路径。
- `verify_fixes.py` 新增 `literal_not` 检查类型（regex-free 缺失断言，根治 `fix-registry-regex-paren`）。
- 基线校准 `UNRESOLVED-017`（货物/服务类，**blocked**：缺真实文档）。

### regression: 108/108 regression · route_preservation 3/3 · verify_fixes 159/0 · doc_drift 14/14

---

## [2026-09-11] — 文档全重组 + 计数防漂移（FIX-2026-09-11-032）

### Changed
- **根目录精简至 3 份文档**：`README.md` / `AGENTS.md` / `CHANGELOG.md`；`ARCHITECTURE` / `MANIFEST` / `DECISIONS` / `SECURITY` / `USER_MANUAL` / `IMPROVEMENTS_SKIPPED` 移入 `docs/`（平铺，`git mv` 保历史）。
- `CONTRIBUTING.md` 并入 `AGENTS.md`（提交与分支 / 测试运行器 / 文档维护约定 / 代码规范 / 评审门禁），原文件删除。
- `docs/MANIFEST.md` 砍为**纯目录地图**（去统计摘要 + 逐文件行数，杜绝最快腐烂项）。
- `docs/ARCHITECTURE.md` 移除「模块规模 Top 5」逐文件行数表。

### Added
- `scripts/check_doc_drift.py`：从代码重算 {蓝图/表/服务/供应商/指标} 并与文档声明比对；接入 `pre-commit`（失败即拒提交，旁路 `SKIP_DOC_DRIFT=1`）。
- fix_registry `FIX-2026-09-11-032` 锁住该不变式。

### Fixed
- 修正文档计数漂移：供应商 5→2、表 70→72、指标 46→45、蓝图 15→17；修正全部跨文件相对链接（`docs/`、`repair_kit/`）。

### regression: 文档重组，无代码路径改动（verify_fixes 142→147 通过；check_doc_drift 14/14 claims OK）

---

## [2026-09-11] — 会话自举 + 清理（chore）

### Added
- **会话自举**：`AGENTS.md` 新增 `Session Operating Protocol`（完成前 gate 自证 + 只读复核门禁 + 交付推送）；全局插件 `session-bootstrap.ts`（镜像 shared-agent-infra）新会话一次性注入 handoff + unresolved + findings。
- `.githooks/post-commit`：仅写 `data/qa_loop/pending.flag`（待运行标记，不自动跑 loop）；`pending.flag` 加入 `.gitignore`。

### Removed
- 删除废弃的 `20260827log.md`（内容已沉淀入 CHANGELOG/ARCHITECTURE/DECISIONS）；删除未被引用、含语法错误的 `screenshots/`。
- 视觉回归统一到 `tests/visual_regression.py` + `tests/visual_screenshots/`（截图不入仓，`manifest.json` 入仓）。

### Fixed
- 修正 `20260827log.md` / `CONTRIBUTING.md` 的悬空引用。

---

## [2026-09-11] — 清标 P0/P1：死路由/落库/前端补表/删历史蓝图/鉴权（FIX-2026-09-10-028~031）

### Fixed
- 删除被 `chat_bp` 覆盖的 `knowledge_bp` 死路由 `/feedback`（FIX-028）。
- 清标结果落库：`run_analysis(persist=False)` opt-in + `_persist_clearance_review` 幂等(DELETE→save)，管理端 quote/relationship 面板不再恒空（FIX-029）。
- `/clearance/status|stream` 补登录校验（consent + user_id）（FIX-031）。

### Changed
- 删除历史遗留 `document_analysis` 蓝图/任务（功能已并入清标），保留 `document_analysis_svc`（FIX-030）。
- 前端补「基本信息表 + 开标信息表」（对齐 DOCX）；删除 `batch_orchestrator` 6 个零调用 builder。
- `verify_fixes.py` 新增 `literal` 检查类型（纯字符串，规避正则元字符）；routes_snapshot 405→401 + `scripts/dump_routes.py`。

### regression: 121/121 regression+collusion · route_preservation 3/3 · smoke 7/7 · run_tests 28/28

---


### Removed
- **错别字检测子系统整体删除**：误报根因是手写 `_BIDDING_CONFUSION_PAIRS` 把 **正确常用词**（必须/截止/权利/签订/缴纳/期间/形式/权力/制定/定金/截至/其间/订金/交纳/必需）当可疑词逐次标记（每份标书数百假警）；`pycorrector`/`symspellpy` 未安装（英文/中文层实为空）。**与 jieba 无关**（typo_detector 不用 jieba）
- 删除文件：`app/services/typo_detector.py`、`app/services/typo_whitelist.py`
- 清标指标 `economic_error_similar`（**46→45 项**）+ 权重/cap + `_run_checker` typo 分支 + 文件分加成 + 指标构建分支
- 投标审计 `typo_detection`：`audit.py all_funcs`、`audit_engine`（评分/dispatch/执行/回读）、`audit_report` 标签与扣分分支、`bid-audit.js` 标签/阈值
- 路由：`/check_typos`(batch)、`/admin/typo_results`(admin_ops)；图谱 `_merge_typo_cross`；`batch_orchestrator` typo 子检查器与报告段
- DB：`typo_detection_results` 表/索引/`audit_config` 种子 + 幂等 `DROP TABLE`（丢历史）
- `runtime_config` 7 个 `typo_*` 键；前端 `#sidebarTypoResultsBtn` + `renderTypoHistory`；`requirements.txt` pycorrector/pyspellchecker/symspellpy（**jieba 保留**）

### Changed
- 基线 `scores.json` 刷新（45 项，composite 19.0→16.1）；`routes_snapshot.json` 重新生成（383→405，纳入此前未入快照的路由）+ `test_route_preservation` expected_len 重定；fix_registry FIX-015 移除指向已删文件的检查

### regression: 130/130 tests passed · verify_fixes 128/128 · T0 no_route=0
- 保留 `relationship_extractor` 独立"相同格式/元数据"启发式（非 typo_detector，不产生 300+ 假警）

---

## [2026-09-09] — 主 agent 提示词：不可变默认 + 每用户自定义（≤2）+ 消息模板统一（FIX-2026-09-09-026）

### Changed
- **提示词模型重构**：服务器默认 = 硬编码 `_DEFAULT_PROMPT`（**不可变**，任何路由不可写）；退役管理员全局覆盖 `/admin/system_prompt` 与 `data/agent_prompt.json`。**每个登录用户**可查看原始版、保存**自己的 ≤2 个系统提示词版本**（选一生效，无则回退默认），用户提示词自动追加安全 guard
- **消息模板统一入库**：原聊天输入框的 localStorage 片段模板改为 DB（每用户 ≤5），与系统提示词并入同一编辑器

### Added
- 表 `user_prompts(id, user_id, kind, name, content, is_active, created_at, updated_at)`（`kind='agent'` ≤2 + 单 active；`kind='template'` ≤5）
- `app/services/user_prompt.py`：`resolve_user_prompt` / `list_user_prompts` / `save_user_prompt` / `activate_prompt` / `delete_prompt` / `migrate_templates`
- 路由（任意登录用户）：`GET /prompts/default` · `GET /prompts/mine` · `POST /prompts/save|activate|delete|migrate_templates`
- 前端统一编辑器 `openPromptEditor`（系统提示词 / 消息模板 两 tab）；入口 `#promptEditorBtn`（全员可见）+ `#promptTemplatesBtn`（模板页）

### Fixed
- **agent 按用户解析提示词**：`chat.py`（流式/隔离）+ `agent.py get_agent` 改 `resolve_user_prompt(user_id)`；缓存键 `(user_id, prompt_hash, max_tokens)` + LRU 8；移除模块级提示词快照（原 `agent.py:26`）

### regression: 128/128 tests passed · verify_fixes 125/125 · T0 no_route=0
- 容器 API 实测：default len=782 不可变 · agent v1/v2=200 / v3=400（≤2）· activate 唯一 active · template ≤5 · migrate 满则 imported=0 · 跨用户隔离 · `/admin/system_prompt`→404

---

## [2026-09-09] — 大文件对比：413 修复 + 异步剽窃任务（FIX-2026-09-09-025）

### Fixed
- **`/batch/plagiarism/compare` 413（Content Too Large）**：直传 multipart 撞全局 `MAX_CONTENT_LENGTH=50MB`。改为：`batch_bp.before_request` 抬 per-request 上限至 11GB；5 个对比端点（plagiarism/compare、compare_bidders_quotes、check_quote_anomaly、extract_relationships、check_typos）接入共享 `_collect_docs()`——优先 `file_ids`（`/stream_upload` 预上传 → `file_store.resolve` → `extract_text_from_path` 分页、内存恒定），回退直传
- **大文件同步计算 OOM**：~200MB 纯文本在 web worker 内对比触发 OOM/502。新增异步路径 `POST /batch/plagiarism/run` + `GET /batch/plagiarism/status/<task_id>`（Celery `plagiarism_task` + TaskBus）；`plagiarism/compare` 对 >40MB file_ids 自动转异步；前端剽窃按钮优先预上传 file_ids → 异步 + 进度轮询

### Added
- `app/services/plagiarism_task.py`（Celery 异步剽窃对比）；`celery_app.py` include 注册

### regression: 128/128 tests passed · verify_fixes 120/120 · T0 no_route=0
- 大文件上传 206MB×2 成功；异步全链路 PASS（16MB/114k 段 → completed，verdict 高度相似，cosine 0.80）

---

## [2026-09-09] — UI 审计 T2 深度之旅揪出 3 个 schema/类型真 bug + anon 守门 + hard gate（FIX-2026-09-09-024）

### Fixed（T2 真实链路发现）
- **`batch_comparison_results` 缺 `project_id` 列**：clearance 写入 INSERT 引用但 DDL 无 → 全新库 `column "project_id" does not exist`。补 CREATE 列 + 幂等 `ALTER TABLE`
- **`batch_pair_results` 表全新库不存在**：clearance/document_analysis INSERT 与 graph SELECT 都用到 → 补建表（含 `UNIQUE(task_id,file_a,file_b)` 供 ON CONFLICT + task 索引）
- **numpy 标量泄漏入 SQL**：`np.float32/float64` 直接作 psycopg2 参数 → `schema "np" does not exist`。`clearance_engine`/`document_analysis_svc` 写入处 `float()/int()` 归一 + `json.dumps(default=float)`
- **anon 引导特权请求**：`/check_storage`×3、`/notebook`、`/admin/projects`、`/cases`、`/templates` 在未登录时仍发起（服务器 401/403 + 控制台噪声）。fetch 拦截器移至 `index.html <head>` 内联脚本（早于 templates.js/cases.js 自跑 loader），匿名短路为合成 401

### Added
- `scripts/audit_trips.py`：4 趟真实深度之旅（clearance fixture 端到端 46 指标 / VL 实图 / 剽窃对比 / provider 实时刷新），provider 503/429 容错
- `run_ui_audit.py --gate`：coverage ledger 三态（acted/blocked:reason/unexplained）+ benign 白名单（anon 引导 401/403）→ hard gate **PASS**（319 元素 / 236 acted / unexplained 0 / real failures 0）

### regression: 128/128 tests passed · verify_fixes 115/115 · e2e gate exit 0 · T2 4/4 绿

---

## [2026-09-09] — UI/interaction "find-all" 审计 T0：静态死链交叉检查（FIX-2026-09-09-023）

### Added
- `scripts/audit_js_routes.py`：扫描 16 JS + templates 的 fetch/axios/XHR/ajax/href/action/window.open → 路径模板 + 方法感知解析 Flask url_map → 三档分类（no-route/dynamic/external）。首轮**报告模式**跑出并修复 5 处前端调用→不存在路由的死链

### Fixed（T0 揪出的真实死链）
- **audit_bp 从未注册**：register_all() 缺 audit → `/audit/*` 全部 404（审计后端整块闲置）；补注册复活
- `/knowledge_lab/feedback` 无路由（技能点赞点踩）→ 参照 ingest 反馈实现（落 user_feedback + 训练日志）
- `/set_video_analysis` 无路由 → 镜像 `/set_image_analysis`（session['analyze_videos']）
- `/admin/skill_supersession/respond` 无路由 → 前端改指现有 `/admin/skill_merge`（能力早已存在）
- `/batch/plagiarism/compare` 404 → batch_bp 根级路由是 `/plagiarism/compare`，补 alias 路由（AGENTS.md 文档 URL 保持可用）

### Added（T1 审计基建，首切）
- `docker-compose.e2e.yml`：throwaway 栈（复用镜像、`localai-e2e-*`、:4443、e2e_* 新卷）
- `scripts/e2e_seed.py`（CEO admin / e2euser，幂等）+ `tests/fixtures/audit/`（vl_test.png + 合成标书 docx）
- `scripts/run_ui_audit.py`：expand-all + 交互元素枚举 → safe-action 策略 + coverage ledger（exercised/blocked:reason/unexplained）+ pageerror/console.error/HTTP≥400 采集
- 首切实跑 anon-home：34 元素/acted 23/15 failures → 均为匿名页引导 401/403（/cases /templates /notebook /check_storage×3 /admin/projects），低危真实，列入 backlog

### regression: 128/128 tests passed · verify_fixes 110/110 · e2e site 200 · T0 no_route=0
- backlog：admin 登录驱动+递归 surface walk、/check_storage 引导 401/403 判定、audit_trips 深度之旅（清标 fixture/VL 实图/剽窃/provider refresh）

---

## [2026-09-09] — VL 识别可靠性：OCR ground-truth + 最强 provider + 交叉验证（FIX-2026-09-09-022）

### Changed
- **VL provider 质量排序**：auto 解析 dashscope → nvidia → mimo（`VL_STRENGTH`）；显式 pin 尊重但受 OCR+verifier 守卫

### Added
- **OCR ground-truth 层**：`ocr.py ocr_text_from_bytes`；图片抽检 OCR-first（确定性读文本/数字，命中即不再调弱 VL）；OCR-empty 子集才走 VL
- **`verify_image()` 交叉验证**：primary（当前激活）+ 按强度序候选 verifier（数字集/长度一致性比对，容忍坏 key 顺延）；无第二 key 单模型标注
- **不静默丢**：抽检行带 `[来源]` 标签（OCR / VL识别 / VL+复核 / 需人工复核 / 无法识别），无法识别显式标注
- **`/admin/vl_test` 升级**：响应含 `ocr/provider/verifier_desc/consistent/note`；前端并列展示 OCR 识别文字 + VL 描述 + 一致/复核徽标 + 推理 + 复核文本——弱 VL 是否读对一眼可辨
- 抽检入口守卫放宽：OCR 或 VL 任一可用即可跑（VL 熔断不再阻断 OCR 抽检）

### regression: 128/128 tests passed · verify_fixes 105/105 · check_system 133/137
- +4 VL 单测（select_vl_pair 排序 / 交叉数字不一致→复核 / 一致 / 无 verifier 单模型）
- 容器实机：OCR 精确读出生成图 `12345.67`/`88000`；真实 verify_image：mimo 主读正确 → dashscope 401 → 顺延 nvidia 交叉一致（坏 key 容忍 PASS）

---

## [2026-09-09] — admin VL 测试组件 404 修复 + 推理展示（FIX-2026-09-09-021）

### Fixed
- **`/admin/vl_test` 404**：admin VL 测试组件（review.js `handleVLTest`）上传图片 POST 到从未实现的路由 → HTML 404 → 前端报 "Unexpected token '<'... is not valid JSON"。新增 `POST /admin/vl_test`（multipart `image`，`@admin_required`），响应裸 jsonify 匹配前端契约（`{status:'ok', data:{description, reasoning}}` / 无文件 400 / ⚠️ 失败串→`{status:'error'}`）

### Added
- **`vl_model.describe_image_v2`**：单图描述同时返回 `reasoning_content`（mimo 等推理 VL 模型）；失败沿用 ⚠️ 错误串模式；现有 `describe_image` 字符串调用方零影响

### regression: 124/124 tests passed · verify_fixes 102/102 · check_system 133/137
- +3 单测（describe_image_v2 content+reasoning / 非推理空 reasoning / 不可用 ⚠️）+ test_admin.py `TestAdminVLTest` 4 用例（db 标记）
- 容器实机：真实 mimo-v2.5 调用 PASS（图→描述 + reasoning 正确）；路由三态 PASS
- 测试文件上传改用 `io.BytesIO`（werkzeug 对 raw bytes tuple 不当文件解析）

---

## [2026-09-09] — 紧急 500 修复 + LLM provider/model 选择器改造

### Fixed
- **`/clearance/run` 500**（FIX-2026-09-09-020）：round-012 把 `target_threads`（list）塞进 `TaskBus.register_queued(extra)` 直写 Redis → `DataError`。改为 `json.dumps(target_threads)` 序列化
- **`/admin/runtime_config` 500**：镜像构建烘焙的 `data/runtime_config.json`/`llm_catalog.json` 为 root 属主，app(uid=1000) 写入 `PermissionError`。容器 `chown -R localai:localai /app/data` 修复（Dockerfile 已有 `COPY --chown` 保障新卷）

### Added
- **自定义 LLM Provider**：admin LLM 组新增 `llm_custom_providers`（json-list 行式编辑器：id/name/base_url/api_key_env，增删行）；API key 存 `.env`（`LLM_CUSTOM_KEY_<ID>`，provider 记录只存环境变量名）；`llm_provider.get_merged_provider_config()` 防御性拷贝叠加自定义（内置永不覆盖）
- **provider 校验**：`validate_custom_provider`（id 非空/`^[A-Za-z0-9_-]+$`/非内置冲突；base_url `https://` 开头，localhost 豁免；api_key_env 合法 env 名）；`runtime_config.update` 整批拒绝非法（含重复 id、reasoning_effort 越界）
- **实时模型拉取**：`GET /llm_providers/<pid>/models?refresh=1`（chat + admin 双端点）；带 key→匿名→catalog 缓存→静态 models 四级兜底，失败返回 `stale:true`；`llm_catalog._fetch_provider_models(base_url, api_key, free_only)` 参数化，refresh 遍历静态+自定义
- **统一 high thinking**：`_create_chat_model_direct`（17 个调用点统一工厂）加 `extra_body={"reasoning_effort": ...}`，默认 high，admin 可调 `llm_reasoning_effort`（low/medium/high）；顺带接入死配置 `llm_temperature`/`llm_max_tokens`
- **前端**：模型下拉"刷新模型"按钮（loading + 在途防抖 + stale/错误提示）；`auth.py has_llm` 补 OPENROUTER/NVIDIA key（修恒 false，影响 AI 评审开关）；`.env.example` 补 `LLM_CUSTOM_KEY_<ID>` 说明

### regression: 121/121 tests passed · verify_fixes 101/101 · check_system 133/137 · node ×2 OK
- P1-A 22 断言（合并防覆盖/校验 7 用例/签名保留）· P1-B 18 断言（合法入库/非法与重复拒绝/离线不崩）· P1-C 15 断言（实时端点四级兜底/未知 404/has_llm）· 容器 R1/R2 实机验证

---

## [2026-09-09] — 清标报告 UI 改造 + 归属双写 + DOCX 警示表格式化

### Added
- **警示总结表格**：清标报告下载链接与大标题之间新增 `_renderAlertSummary`（9 行汇总：冒烟指数/预警级别/铁证/暗标违规/高险指标/高嫌疑单位/集团/段落雷同/合规严重项，按最高 severity 着色）
- **大标题跟随着色**：各章节 `<details class="cl-l1">` 按下属最高警示动态加 `cl-danger`/`cl-warn`（颜色+左边框）
- **DOCX 警示详情章节**：无编号"警示详情与处理指引"——铁证/违规/6.9 段落/高风险组合/合规 critical 五类警示源各一表（原文摘录/所在章节/处理指引），`_guidance_for` 内置类型→指引映射；元数据/平台类警示显示"证据来源：文件属性/交易平台记录"

### Changed
- **折叠箭头**：quoteBubble 与报告全部章节折叠改为 Material Symbols `expand_more` + `.collapsed` CSS 旋转（`_toggleArrow`/`_clArrow`）
- **章节参数统一表格**：指标 div 卡片→表格（含 details 折叠行）、围标集团→表格、合规徽章→表格、AI 评分→表格、图片抽检/审计文本→表格
- **封面铁证/违规红字→表格**（证据类型/级别/证据文本/涉及文件 + 红色判定语）

### Fixed
- **清标结果误进项目对话**：新增 `resolve_clearance_threads`（跟随当前对话归属 + 同步该用户最新个人对话；个人对话跑→进最新个人对话；无会话则新建"分析结果"会话；DB 失败回退当前线程）；`run_clearance_async` 持久化遍历目标线程各插一条 CLEARANCE_REPORT。另修 `window.currentProjectId` 恒空 bug（→ 裸 `currentProjectId`）

### regression: 121/121 tests passed · verify_fixes 101/101 · check_system 133/137 · node --check app.js OK
- F1 渲染冒烟 21/21（XSS/老报告守卫/severity 映射）· F2 真库 5 用例（项目双写/个人单写/个人非最新双写/空会话建/DB失败回退）· B1 合成 15/15（5 类警示子节/元数据占位/章节映射/空报告守卫）
- 遗留：报告不携带正文 text → DOCX 章节定位用段类型映射（`_locate_chapter` 回扫逻辑已备，待 report 携带 text 时启用）；页码明确不做

---

## [2026-09-09] — 暗标违规检测开关（默认关闭）+ 盖章弱信号降权（FIX-2026-09-09-019）

### Changed
- **暗标违规检测开关**：分析维度行（横向对比/指标分析/合规审查/AI 评审）新增 `暗标违规` 复选框（`optTechSeal`），**默认不勾选**。`options['tech_seal_check']=False`（默认）时 `run_analysis` 跳过 tech_seal checker → 指标显示"○ 跳过（未开启暗标违规检查）"、不产生"⚠ 暗标违规"警示；`run_clearance`/`clearance.py` 透传 options。默认关同时覆盖非清标文档分析路径（run_analysis 调用方不传 options）

### Fixed
- **暗标全标误报**：`tech_seal_detector` 把普通标书正文必然出现的 `盖章/公章/签字盖章/投标专用章` 当暗标身份泄露 → 任意含"盖章"的标书触发 `■ 高度预警（暗标违规）`。修复：`_SEAL_MARKERS` 降级为**辅助证据**（不再独立触发 leak，仅当强信号——技术方案段公司名 / ≥4 处人员姓名——已判泄露时追加）；检测顺序重排为 公司名→人员姓名→盖章辅助

### regression: 121/121 tests passed · verify_fixes 101/101 · check_system 133/137 · node --check app.js OK
- `test_tech_seal_violation_independent` 改显式开启 `options={'tech_seal_check': True}`；新增 `test_tech_seal_default_off`（默认关无违规+指标 skipped）、`test_tech_seal_seal_marker_weak_signal`（纯盖章不触发 / 强信号+盖章辅助）
- 合成两态：默认关→violation_fired False + skipped；开启→`■ 高度预警（暗标违规）`

---

## [2026-09-09] — 全方位审核修复：死代码清理 / XSS 加固 / 报价信号完整呈现 / 段落证据前端 / 社区持久化（FIX-2026-09-09-018）

### Fixed
- **报价信号丢失**：`cross_tailing_digits`/`cross_progression`（跨投标人）与 per-bidder `tailing_digits_flag`/`progression_type` 此前计算后未持久化。`quote_anomaly_results` 表 +5 列（含幂等 ALTER），`save_quote_anomaly_results` INSERT 18→23 列；HTML 报表加"尾数一致/等比规律"列 + 2 条 cross 提示；管理端历史列表 SELECT 补齐新列
- **铁证证据前端缺失（P0）**：段落级雷同证据此前仅 DOCX 有 6.9 表。新增 `_renderParagraphCollusionEvidence`（`_renderIndicatorsTab` 调用，照抄 DOCX 筛选/排序/截断逻辑，`_clearanceEscape` 全转义，模板段折叠参考）；数据已三路下发前端，后端零改动，live + 聊天重载双路径生效
- **死代码清理**：删除零调用的 `renderDocAnalysisResults`（app.js，80 行）与 `loadAuditHistory`（bid-audit.js，83 行，含 return 后 70 行不可达）
- **XSS 加固**：`renderQuoteAnomalyHistory`/`renderRelationshipHistory`/`renderTypoHistory` 的 `id`/`checked_at`/`task_id` 裸插值补 `escapeHtml`（`suggestions` 原已转义未重复包裹）
- **社区检测持久化**：`relationship_risk_summary.details` 由单存 `company_personnel_map` 改为并入 `communities`（Louvain 团伙分组）；`/admin/relationship_results/<task_id>` 返回 communities；前端新增"团伙"详情按钮 + 社区区块渲染
- **阈值对齐**：bid-audit.js `drop` 默认 0.15→0.30（对齐后端 `quote_anomaly_drop_threshold`）
- **chat.js emoji 统一**：置顶态 `textContent='📌'` → `_icon('📌')`（与其他态一致）

### regression: 119/119 tests passed · verify_fixes 96/96 · check_system 133/137 · node --check ×3 OK
- 契约验证：`_run_cross_comparison` 产出 `paragraph_collusion`（服务承诺段 surprise=0.35/98% 一致）与前端渲染契约匹配
- 管理端历史表新增"本福特/尾数一致/等比规律"三列（Benford >0.15 标黄）

---



## [2026-09-09] — 铁证双层判定：铁证信号独立成硬警报，不再被复合指数稀释（FIX-2026-09-09-017）

### Added
- **铁证双层判定层** `app/services/hard_evidence.py`：铁证信号不参与加权复合指数打分（软嫌疑度指数保留），独立判定层 **veto 只提升展示级别**（`warning_level`），不重写指数
  - **T1 确认级（单命中即 veto → `■ 高度预警（铁证触发）`）**：`lastModifiedBy` 同人（guard 排除 Administrator/User/微软用户/lenovo 等通用值）、平台加密锁/文件码雷同（仅交易平台来源）、联系人+电话同组双命中、段落同对 ≥2 段或 1 段 ≥3 家共享
  - **T2 强嫌疑（需 ≥2 类共证才 veto）**：author 雷同（guard）、上传/解密 IP 同、段落单段共享
- **暗标违规独立轨道**：`tech_seal` 泄露（单家违规非串通证据）触发 `■ 高度预警（暗标违规）`，不进串通铁证；铁证与违规可各自独立出现，同时触发时串标优先、违规附加
- **报告接入**：`run_analysis` basic_info 新增 `hard_alarm`/`hard_label`/`hard_evidence`；`run_clearance` 最终出口补入横向层段落雷同后终判；DOCX 封面铁证/违规红色警示段 + 预警单位 `★`（hard_flag）；`suspected_units.hard_flag`
- **前端**：`app.js` 清标结果按 `hard_alarm`/`warning_level` 联合着色（修"绿分+红字"矛盾），历史列表 `★铁证` 红标

### Fixed
- 铁证被加权稀释：`_weighted_total_score` 分母含所有非 skip 指标，铁证权重最高 0.10 → 单铁证实际贡献 ~5 分；段落级逐字雷同（最强证据）此前不计入复合指数，封面与证据脱节

### regression: 119/119 tests passed · verify_fixes 96/96 · check_system 133/137 · app.js node --check OK
- 新增 4 回归测试：lastModifiedBy 铁证升级（指数不被改写）/ guard 反例（Administrator 不触发）/ T2 双类共证 / 暗标违规独立触发
- **真实 3 文件复测 PASS**：元丰+中昌华美 `lastModifiedBy='超彩赵'`（物美='唯一的麦麦儿'）→ T1 veto `fired=True label='■ 高度预警（铁证触发）'`
- 复合指数/基线 scores.json 19.0/DB max_risk 不变（veto 只升展示级，历史可比性保留）

---

## [2026-09-08] — 46 项指标语义错配系统修复：联系人/关系/暗标/投标数（FIX-2026-09-07-QA-C4）

### Fixed
- **联系人雷同误报**（用户报告 3.1.6）：`contact_person_same`/`cross_contact_same`/`contact_phone_abnormal` 不再用 key_info 关键词重合冒充，新增 `app/services/contact_extractor.py` 从投标文件正文提取真实联系人/手机号/邮箱并跨文件比对；无联系人数据落"○ 需开标信息表/联系人数据"占位（score=0）
- **关系指标误报**：`bidder_agent_contact`/`expert_bidder_closeness` 无代理/评委名单时改真 skip（此前误跑通用关系报告给 28.5 分，文案自相矛盾）
- **暗标检测空壳**：`tech_seal_check` 不再用错别字检测冒充，新增 `app/services/tech_seal_detector.py` 真实检测（技术方案段公司名/印章提示/大量人员姓名→泄露；校准避免自指称谓误判）
- **投标数漏报**：`bidder_count_abnormal` 本地 `n<3` 触发（此前被误标 skip 吞掉）
- **行业词污染关键词**：`extract_keywords`/`keyword_overlap_similarity`/`build_key_info_matches` 接入行业词表
- **指标去重/语义**：`same_machine_code`→"文件作者/编制人雷同"改名、`cross_machine_code`→skip；`cross_contact_same` 加权去重；专家指标占位保持 skipped 不压低指数；`_weighted_total_score` docstring 与实现一致
- **平台列激活**：`clearance_openinfo` 放开 IP/文件码/加密锁列映射 + `_platform_signals`（开标表含平台列且跨单位重复时激活对应指标）；`extract_metadata` docx 补 creator/producer；`suspected_units` 补 lasteditor 组加成

### regression: 115/115 tests passed · verify_fixes 89/89 · check_system 133/137
- 基线 scores.json 刷新（contact 0.5→0、relationship 28.5→skip、tech_seal 空壳→真检测、bidder_count skip→触发），composite 19.0 正常
- 新增 4 回归测试：联系人真实比对/无数据占位/暗标泄露与非泄露/投标数本地触发

---



## [2026-09-08] — 清标证据链提纯：模板段排除 + 元数据硬信号 + 6.9 铁证优先（FIX-2026-09-07-QA-C3）

### Fixed
- **6.9「共享实质段落」信号纯度**：投标函/声明模板段（"一、按照招标文件要求提交投标文件正本1份…"等套招标模板的合规内容）不再当串标证据——`_LEGAL_TERMS` 扩充（正本/副本/有效期/声明/真实有效/待命/供应/腐烂/变质…）+ 新增「投标函声明段」段落类型 + 门槛修正（legal>=2/tech>=2）；**真实 3 文件复测 shared_segments 195→36**
- **集团判定过度**：`detect_gangs` 门槛从"≥1 共享段"提至"**≥2 非模板实质段**"（`evidence_counts` + `min_evidence=2`）——只共享投标函模板的正常同行不再误判集团
- **元数据硬信号纳入**：`extract_metadata` 补 `cp:lastModifiedBy` 提取 + 新指标 `file_attr_lasteditor_same`（触发指标, 权重 0.10, cap 30）——"同一最后编辑人做两家标书"是比文本更硬的串标信号
- **6.9 呈现**：铁证优先（只列服务承诺/技术方案段，按惊讶度降序，预览加长 100 字）+ **6.9.2 模板/声明段折叠**（灰字参考，标注"非串标证据"）

### Changed
- `collusion_score`/`per_pair_count` 只统计非模板实质段（模板段不再污染评分与集团证据）

### regression: 111/111 tests passed · verify_fixes 89/89 · check_system 133/137
- 真实 3 文件复测：服务承诺段铁证清晰（应急响应"2分钟/10分钟"数字微调、食堂氛围、食材新鲜、应急预案块）；元数据「超彩赵」（元丰+中昌华美）分组触发；6.9 铁证优先 + 模板折叠渲染
- 新增 3 回归测试：投标函声明段排除 / last_modified_by 提取 / lasteditor 指标触发

---



## [2026-09-07] — 真实文件清标测试：段落检测器性能加固 + 行业词表接线（FIX-2026-09-07-QA-C2）

### Fixed
- **段落级实质雷同检测 O(n²) 性能爆炸**（实测 300 段×3 文件 >300s 超时）：锚子串倒排索引替代段长桶遍历 + `MAX_CAND_PER_PARA=40` 候选上限 + `MAX_PARAS_PER_FILE=2500` 分层抽样 → 300 段 2.67s / 1000 段 8.96s / 2500 段 28.9s；**真实 3 文件（~3500 段）1.6s**
- **内存：弃全量 bigram frozenset 指纹**（数万段可达 2-4GB）
- **`merged` 去重 key 用文件集合 → 吞不同实质段**：改「段落类型 + 内容前缀」指纹，同一实质段跨 3 家正确合并
- **行业词表未接指标层**：`preprocess_text_for_similarity`/`_precompute_tfidf_for_files`/`compute_all_pairs` 加 `extra_stop_words`；`run_analysis`/`run_clearance` 统一 ptype 下传 → 军营超市行业词（超市/收银/理货）不再抬高 key_info/余弦（sim 87.4→83.0 等）
- ThreadPool 4→5、删 `batch_orchestrator.py` 死代码、删 `run_analysis_async` 重复 `init_flask_context`

### Changed
- `collusion_para_map` 评分 `min(100, c*25)` → `min(100, sqrt(c)*25)`（避免 4 段即满分）

### regression: 108/108 tests passed · verify_fixes 89/89 · check_system 133/137
- **真实 3 文件容器内测试**（元丰/物美/中昌华美 军营超市项目，已证实串标）：
  - 算法层：`total_score=36 中等预警`、`collusion_score=70`、195 共享段、25 服务承诺段（match 0.85-1.00, surprise 0.53-0.78）、三对 collusion_para=100、集团 1（有实质证据）
  - 端到端 Celery：`batch_comparison_results` 落库成功，产物 DOCX+PDF ZIP 取回桌面
- 新增 4 回归测试：性能上限 / 内容指纹去重 / 行业词接线（preprocess + run_analysis）

---



## [2026-09-07] — Docker 构建加速：GPU 自动检测 + torch CPU/CUDA 分流（FIX-2026-09-07-QA-C1）

### Fixed
- **Docker build 从 1.5-2 小时降到 ~90 秒**：根因是 `torch==2.12.1` 在 Linux PyPI/清华源解析为 **CUDA 版**（拖入 nvidia-cublas 423MB/cuda-toolkit/nvidia-\* 全家桶，pip 层 6.69GB），而运行时**纯 CPU**（代码零 `.cuda()`、easyocr 日志实证 "Using CPU"、compose 不分配 GPU）
- **两机同仓 GPU 智能分流**：新增 `scripts/docker_build.py`——宿主 `nvidia-smi` 检测 → 无 GPU 装 `download.pytorch.org/whl/cpu`（`torch 2.12.1+cpu`），有 GPU 装 `.../whl/cu124`（`TORCH_CUDA_INDEX` 可覆盖）；Dockerfile `ARG TORCH_INDEX` 默认 CPU
- **构建提速基建**：apt/pip 全改 BuildKit `--mount=type=cache`（跨构建复用下载）；torch/torchvision 独立一层利于缓存；`.dockerignore` 收紧（上下文 185MB→~10MB，排除 `local_cache/`95MB/`.opencode/`55MB/`tools/`22MB 等）
- **堵密钥泄入镜像**：`.dockerignore` 补 `.env`（文件，此前只排了 `.env/` 目录）
- **新增 `docker-compose.gpu.yml`**：app/celery-worker GPU 设备保留（GPU 机 `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`）
- **修复行业词表被 `app_data` 卷遮蔽**：compose 三服务加 `./data/industry_words:/app/data/industry_words:ro` 只读 bind 挂载

### Changed
- 镜像体积 **12.8GB → 5.94GB**（砍掉 CUDA 载荷）；容器内 `torch 2.12.1+cpu`、`cuda_available: False`

### regression: 104/104 tests passed · verify_fixes 89/89 · check_system 133/137
- 本机 CPU 分支实测：build ~90s、站点 200、industry_words 挂载生效、容器内 4 项修复标记命中
- 注：GPU 机（RTX 2080 Super, Turing sm_75）需真机验证 cu124 兼容性；不兼容则改 `TORCH_CUDA_INDEX`

---



## [2026-09-07] — QA-Loop round-004：清标算法可靠性重构（段落级实质雷同）+ 字体排版

### Added
- **段落级实质雷同检测**（`app/services/paragraph_collusion_detector.py`，FIX-2026-09-04-QA-B1）：在原始文本上分段，两两 `SequenceMatcher ≥0.85` 找近逐字雷同段，按**惊讶度**（段内非行业/非模板词占比）过滤样板段，仅"服务承诺段/技术方案段"等实质内容的跨文件雷同算围标信号
- **行业词三表**（`data/industry_words/{engineering,goods,services}.txt` + `app/services/industry_words.py`）：双层（国家标准词层：财政部《政府采购品目分类目录 2022》+ 住建部《建筑业企业资质标准》；常见运营词层含超市经营词），采购类型探测（`get_procurement_type`），守卫测试锁定评价性措辞（热情/周到/细致/尽职 等）不入表
- **跨文件共享错别字**（`typo_detector.find_shared_typos`，FIX-2026-09-04-QA-B2）：仅跨 ≥2 家逐字相同的 suspect_text 计分，白名单+行业词排除，天然消除 pycorrector 随机误报
- 报告新增 **6.9、共享实质段落** 章节（段落类型/共享单位/一致率/惊讶度/内容预览）+ 表头底色 + 标题黑体排版

### Changed
- `RiskScorer` 权重重构（`batch_orchestrator.py`）：`text_sim 0.25→0.10`（整篇余弦是信号平均器）、新增 `collusion_para 0.30`（段落级实质雷同为主信号）
- `clearance_engine._run_cross_comparison`：无招标文件时 `template_missing=True` → text_sim 对 risk 零贡献（与指标层 skip 对齐），原始余弦仍保留在 6.2 矩阵标注"仅参考"
- `detect_gangs`：集团必须含 ≥1 对共享实质段（纯模板/行业重叠不再判集团）
- `economic_error_similar`：由"错别字总数"改为"跨文件共享错别字数"
- 封面增加"缺招标文件：文本/关键词类指标未计入，冒烟指数为下限估计"红字标注

### Fixed
- 矩阵表头显示纯 ".docx"（`truncate_filename(fname, 8)` 对中文长名退化为扩展名）→ 20 字符 + `file_processing.py` `available<1` 返回名称开头而非纯扩展名
- `build_attr_details`/`compute_single_pair` 对缺失 `metadata`/`images` 键防御

### regression: 104/104 tests passed · verify_fixes 89/89 · check_system 133/137
- 3 家已证实串标案例（元丰/物美/中昌华美 军营超市项目）：元丰↔物美 服务承诺段近逐字雷同（match 0.94, surprise 0.67）被段落检测器命中；无招标文件时 text_sim 不进风险；合法同行（行业词重叠但独特段不同）不触发
- 基线快照（工程类 价格标 vs 商务技术标）复合指数保持 <30 正常区间

---



## [2026-09-04] — QA-Loop 基础设施：九阶段流程固化（文档同步 + 镜像重建）

### Added
- `qa-loop` 三方自动循环升级为**九阶段**：`COLLECT → VERIFY → CROSS-EXAM → CONFIRM → IMPLEMENT → DOCS → PUSH → IMAGE → RE-CHECK`
  - **⑥ DOCS**：每轮代码变更后，`CHANGELOG.md` 按 FIX 编号细分条目 + `AGENTS.md` 约定同步 + `round-NNN.md`，随代码同 commit（PUSH 前）
  - **⑧ IMAGE**：项目 `has_docker:true` 时 `docker compose build` + `up -d` + 健康检查 + 容器内代码抽查，保证镜像 = HEAD
- 全局 skill（`shared-agent-infra/skills/qa-loop`）+ 全局 command `/qa-loop` + 项目参数（`.opencode/qa-loop.project.md`）同步九阶段
- 维护约定：每轮更新完毕后必须同步文档并重建镜像，保证「仓库 HEAD、文档、运行镜像」三者一致

### round-003（增量无代码变更）
- `9205efe..HEAD` 增量 diff 仅含 docs → 质量闸门通过，直接收尾（commit `50fcce6`）

## [2026-09-04] — QA-Loop round-002：安全纵深补强（D1–D3）

### Fixed
- `FIX-2026-09-04-QA-D1` `chat.js` — `COMPARE_REPORT` 分支直接 `innerHTML=htmlContent` 绕过 DOMPurify（与 C8 纵深不一致）→ 补 `_safeHTML()` 消毒（commit `3f7483f`）
- `FIX-2026-09-04-QA-D2` `app.js` — 三处错误消息（`data.error`/`err.message`/`e.message`）未转义拼 innerHTML → 统一 `escapeHtml()`（commit `498f8b1`）
- `FIX-2026-09-04-QA-D3` `knowledge.py` — `skill_hash` 变量重赋值遮蔽 → `kb_file_hash` + check_system 守卫 + C5 守卫正则修正（commit `9205efe`）

### Changed
- check_system 守卫增至 133/137（新增「skill_hash not shadowed by kb hash」）；tests/test_regression.py 全量 exit=0

## [2026-09-03] — QA-Loop round-001：代码层 11 项修复 + 基础设施固化

### Added
- **QA-Loop 三方自动循环基础设施**（commit `4b178c3`）：全局 skill + 全局 command `/qa-loop` + 项目参数 `.opencode/qa-loop.project.md`；`data/qa_loop/` 基线 + round-001 审计记录

### Fixed
- `FIX-2026-09-03-QA-C1` `knowledge.py` — 空路由装饰器致 `/admin/all_user_kb`(GET) 误绑 `generate_work_report` → 恢复真实端点 + check_system「No orphaned route decorators」守卫（commit `c85fc34`）
- `FIX-2026-09-03-QA-C2` `knowledge.py` — `generate_project_file_skill` IDOR（无项目成员校验）→ 补 `get_user_role_in_project` 成员校验 + 守卫（commit `1cb94dc`）
- `FIX-2026-09-03-QA-C3` `admin.py` — `admin_required`/`auditor_required` 缺失 consent+user_id 会话校验 → 抽公共 `_check_session_valid()` + 守卫（commit `85775f2`）
- `FIX-2026-09-03-QA-C4` `credit.py` — 限速器进程内存 dict 跨 gunicorn worker 失效 → 迁 Redis `credit_rate:{ip}` INCR+TTL + 内存降级 + 2 回归测试 + 守卫（commit `9eecab3`）
- `FIX-2026-09-03-QA-C5` `knowledge.py` — 工作报告 zip 文件句柄泄漏（匿名 `open().read()`）→ `with open` + 守卫（commit `27be656`）
- `FIX-2026-09-03-QA-C6` `app.js` — `checkStorage` warning message 未转义拼 innerHTML → `escapeHtml()`（commit `df6327e`）
- `FIX-2026-09-03-QA-C7` `knowledge.py` — work_report user filter 用 `str.replace('cs.user_id')` 后处理脆弱 → 结构化构建两套 filter + 守卫（commit `7393161`）
- `FIX-2026-09-03-QA-C8` `chat.js` — `md.render()` 输出未过 DOMPurify（15 处）→ 统一 `_renderMarkdown()` 消毒 + 助手（commit `7489653`）
- `FIX-2026-09-03-QA-C9` `file_processing.py` — 文本提取乱码/控制字符（U+FFFD/C0/C1/代理）→ `clean_extracted_text()` + `_CONTROL_FILTER`（commit `c153ee3`）
- `FIX-2026-09-03-QA-C10` `app.js` — 时间线空态无操作引导 → 加「选择项目」按钮（commit `93c9585`）
- `FIX-2026-09-03-QA-C11` `app.js`/`review.js`/`icons.js` — 统计区图标 emoji 混排/方块 → 统一 Material Symbols + `🟢→monitoring` 映射（commit `fb91b59`）

### Changed
- `scripts/check_system.py` 新增多项回归守卫（孤立装饰器/成员校验/会话校验/Redis 限速/zip 句柄/结构化 filter/变量遮蔽）
- `tests/test_regression.py` 新增 credit 限速 Redis 回归测试 ×2
- 基线 `data/qa_loop/last_head` 推进；round-001/002/003 记录在 `data/qa_loop/`

---

## [2026-09-01] — 文档现实对齐更正 + RTX 2080 Super 支持评估

### 文档管线事实更正
- 经代码核验，文档声称的「RapidOCR + MinerU」与代码不符。**实际管线 = MarkItDown 0.1.6 + LibreOffice/soffice + EasyOCR 1.7.2 + PyMuPDF (fitz)**，无 MinerU/RapidOCR，无 rarfile/py7zr/ebooklib/extract-msg
- 更正：`CHANGELOG.md` 07-04/07-08、`DECISIONS.md`、`README.md`、`MANIFEST.md`、`ARCHITECTURE.md`、`IMPROVEMENTS_SKIPPED.md`
- `IMPROVEMENTS_SKIPPED.md` 逐项按真实基线重估；#8 admin.py 拆分标记「已解决」

### RTX 2080 Super 支持评估
- 发现 torch 为 **CPU 版**（`2.12.1+cpu`，CUDA 不可用）→ GPU 完全未启用
- 2080S (8GB) 最大价值：**EasyOCR GPU 加速 + LoRA 微调 Qwen2.5-7B**（Unsloth QLoRA 4bit），均只需重装 CUDA torch 无代码改动
- `ocr.py` 的 `OCR_GPU=auto` 已支持自动探测

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
- **统一文件处理管线**：单一 `FILE_TYPE_REGISTRY`（44 类型）、分层提取（MarkItDown→格式特定→OCR→LibreOffice）、`allowed_file` 校验
  - **2026-09-01 更正**：实际无 MinerU 层；rarfile/py7zr/ebooklib/extract-msg 4 个依赖**未加入 requirements**（归档/电子书仍不受支持）

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

> **⚠️ 2026-09-01 更正**：经代码核验，该升级**未实际落地为 RapidOCR/MinerU**。当前真实管线为 **MarkItDown + LibreOffice + PyMuPDF + EasyOCR**。以下为历史计划记录，供追溯；实际状态见 `IMPROVEMENTS_SKIPPED.md` §现实对齐。

### Added (原计划)
- MinerU (`_try_mineru`, `_strip_markdown`) as primary PDF/DOCX/PPTX/XLSX parser in `file_processing.py`
- `_ocr_pdf_legacy` fallback in `ingest_pipeline.py`

### Changed (实际落地)
- `app/services/ocr.py`: 仍为 **EasyOCR** 1.7.2（含 `OCR_GPU=auto` 探测，GPU 可用时自动启用）
- `file_processing.py`: 结构化提取用 **MarkItDown** 0.1.6；`.doc` 转换用 **LibreOffice/soffice**；PDF 流式提取用 **PyMuPDF (fitz)**
- `requirements.txt`: 无 rapidocr/mineru；rarfile/py7zr/ebooklib/extract-msg 未安装

### Removed
- `easyocr` from `requirements.txt` → **未移除**（EasyOCR 仍在用，`ocr.py` 依赖）

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
