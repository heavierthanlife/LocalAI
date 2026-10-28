# 论文 v1.0 第5章技术事实核验报告

- 核验对象：`/home/guava5tar_/dsh/playground/review/thesis_v1.txt`（python-docx 提取，行号=`thesis_v1.txt:N`，方括号内为段落号）；表格 5-1/5-2 在提取文本中被截断为前 2 行，全文以构建脚本 `build/build_v1.py:56-64`（表5-1）、`build/build_v1.py:65-73`（表5-2）为准；图5-1/图5-2 的图内文字以 `build/make_figures_project.py:62-127` 为准。
- 代码仓：`/mnt/d/AI_Tools/Projects/Local_AI`，`master` 分支，HEAD=`aefee49`（只读）。
- 运行系统：Docker（`localai-app` / `localai-postgres` / `localai-redis` / `localai-celery-worker` / `localai-celery-beat` / `localai-nginx`），库 `localai`（`localai` 用户），仅执行 SELECT / 元数据查询。
- 方法：`git ls-files`/`grep`/`glob`/`wc` 只读检索 + `docker exec ... psql -tAc` 只读查询 + 容器内只读 import 实测（不写库、不改仓库、不重启容器）。
- 结论口径：**正确**＝论文值与实测一致；**需修正**＝与实测不符/无依据；**部分吻合**＝方向对但表述过强；**无法核实**＝现有环境无法取得证据。

---

## 一、结论摘要（按修订优先级）

**必须改（3 项，均可能被答辩追问）：**

1. **"私有化部署，数据不出内网"与代码事实冲突（最严重）**。表5-1"模型服务"行写"私有化部署，数据不出内网"，正文 [222] 写"模型与数据均部署于公司内网"；实际大模型供应商是**公网 API**（`llm_provider.py:27` OpenRouter、`:34` NVIDIA NIM），容器 env 内有 9 个云厂商 key，docker-compose 中无任何本地模型服务。平台/数据库/文件/向量索引确实是内网容器私有化，但"模型在内网"不成立。
2. **5.4 "以接口方式与既有业务系统对接…回传档案系统…财务数据关联"无任何代码依据**（[249]）。且仓库自带架构文档明确写"本系统是投标文件分析工具，非交易平台对接工具……未纳入开发范围"（`docs/ARCHITECTURE.md:98`）。属"设计设想"被写成了"已实现"。
3. **"2025年2月上线智能作业平台"与代码时间线矛盾**（[21][29][98][142][195]）。仓库首次提交 2026-04-17 01:03(+08)，共 250 次提交持续到 2026-10-18，全部作者同一人；论文封面为二○二六年十月。第5章描述的系统即该仓库，则"2025年2月已上线"缺乏依据，需作者说明（如改为"立项/试点"）或调整时间线。

**建议改（表述精度类）：**

4. 76 张表：**建表语句口径正确（76）**，但运行库实测 **77** 张（多出 1 张待迁移删除的 `typo_detection_results`），建议加注口径。
5. "法规结构化入库"（[245]）：15 部/619 条是**运行时从 JSON 文件加载**（容器内实测日志 `Total built-in laws: 15 laws, 619 articles`），`law_masters/law_versions/law_articles` 三张表**均为 0 行**；且核心 4 部为节选（10/5/5/5 条），"全文库"只对 11 部成立。
6. 定时任务"法规同步"（[250]）：25 项 beat 任务中**没有法规同步**，法规变更由管理端触发。
7. 图5-2 字段名与真实列名不符：`planned_start` 等应为 `planned_start_date` 等；`version_no`/`updated_at` 不存在（实为 `version`/`uploaded_at`）。图注写"字段名据数据库表结构整理"，需改。
8. 企业信用查询（[247]）：代码为**半自动**（Selenium 驱动 Edge，验证码需人工处理：`credit_checker.py:85-191`），且容器内无 Chrome/Edge，Docker 部署下不可直接运行。
9. 项目协同"版本对比"（[247]）：有版本历史端点（`admin.py:1282`），未找到文件级 diff 对比端点，建议改"版本历史"。
10. "全量操作审计"若用于描述 `admin_audit_log`：该表仅覆盖后台管理动作（实测 15 行），非"全量"。正文"关键操作留痕"表述可保留。
11. "首次提交 2026-04-16"（交接/画像文档）：按 git 记录应为 **2026-04-17 01:03 +0800 = 2026-04-16 17:03 UTC**，建议标注时区。
12. 60,461 行：口径须写明 `git ls-files '*.py' | xargs wc -l`（197 个被跟踪文件 = 60,461）；裸 `find . -name "*.py"` 若排除 .venv 为 62,878，不可复现 60,461。

**全部核实为"正确"的硬指标（可直接答辩引用）：**
76 张表（建表语句口径）、16 Blueprint、28 路由模块、97 service、60,461 行、250 次提交、单作者、Flask 3.1.3、PostgreSQL 16.15、Redis 7.4.10、Celery 5.4.0、LangGraph 1.2.6、ChromaDB 1.5.9、EasyOCR 1.7.2、Selenium 4.45、PyMuPDF/python-docx、Docker 六服务、20+ 定时任务（25 项）、8 大能力域、五级对象模型表与关键字段、清标五维、围串标四类信号（TF-IDF 相似度/报价尾数/Benford/关系网络）、15 部法规库（619 条）、SSE 流式输出、提示注入防护、文件版本化、结果快照化、权限与审计。

---

## 二、核验总表

> 结论列：✅正确 / ⚠️需修正 / ◐部分吻合 / ❓无法核实

### A. 代码规模与结构

| # | 断言（出处） | 论文/说明值 | 实测值 | 证据 | 结论 |
|---|---|---|---|---|---|
| A1 | 数据库表数 [236] L209；图5-1 | 76 张 | 建表语句 **76** 条；运行库 public 基表 **77** | `grep -c "CREATE TABLE IF NOT EXISTS" app/database.py` → 76（首 L124 `users`，末 L1297 `user_prompts`）；`psql -tAc "select count(*) from information_schema.tables where table_schema='public'"` → 77（多出 `typo_detection_results`，`migrations/002_drop_typo_detection_results.sql` 未执行） | ✅（附注） |
| A2 | 幂等建表 + 版本化迁移 [236] | 有 | 有 | `app/database.py:124` 起全部 `CREATE TABLE IF NOT EXISTS`；`migrations/001_initial_snapshot.sql`（基线空迁移）、`002_...`（drop 历史表，未执行） | ✅ |
| A3 | Blueprint 数 图5-1 | 16 | 16 | `app/routes/__init__.py:66,70,76,82,86,92,98,104,110,116,122,128,136,142,151,157`（16 处 `register_blueprint`）；`app/routes/*.py` 内 `Blueprint(` 定义 16 个 | ✅ |
| A4 | 路由模块数 图5-1 | 28 | 28 | `ls app/routes/*.py | wc -l` → 29（含 `__init__.py`），即 28 个路由模块 | ✅ |
| A5 | service 数 图5-1 | 97 | 97 | `ls app/services/*.py | wc -l` → 97 | ✅ |
| A6 | Python 代码行数（交接/画像文档） | 60,461 行 | 60,461 | `git ls-files '*.py' | xargs wc -l | tail -1` → 60461（197 文件）；排除 .venv 的裸 find → 62,878（口径不符） | ✅（口径需写明） |
| A7 | 提交数（交接:99/画像:3） | 250 次 | 250 | `git log --oneline | wc -l` → 250 | ✅ |
| A8 | 单作者（同上） | YGY | YGY `<jasoncoop@126.com>` | `git log --format='%an <%ae>' | sort -u` → 仅 1 行；另有 3 条提交带 `Co-Authored-By: Claude 4.7` trailer（AI 协作痕迹，作者身份仍单一） | ✅（附注） |
| A9 | 首次提交 2026-04-16（交接:99/148、画像:3） | 2026-04-16 | 2026-04-17 01:03:28 **+0800** = 2026-04-16 17:03:28 UTC | `git log --reverse --format='%H|%ai|%ci|%an|%s' | head -1` → `007d20b|2026-04-17 01:03:28 +0800|…|1st` | ⚠️需标时区 |
| A10 | 最新提交 2026-10-18（交接/画像） | 2026-10-18 | 2026-10-18 20:57:31 +0800 | `git log -1 --format='%h %ai'` → `aefee49 2026-10-18 20:57:31 +0800` | ✅ |
| A11 | HEAD=aefee49、master（任务说明） | 一致 | 一致（但工作区有 160 项未提交改动） | `git rev-parse --abbrev-ref HEAD` → master；`git status --porcelain | wc -l` → 160 | ✅（复现需固定提交） |

### B. 技术选型（表5-1 / 图5-1）

| # | 断言 | 论文值 | 实测值 | 证据 | 结论 |
|---|---|---|---|---|---|
| B1 | 后端框架 | Flask 3.1 + 应用工厂 | Flask 3.1.3；`create_app()` | `requirements.txt:26 Flask==3.1.3`；容器 `importlib.metadata` → 3.1.3；`app/__init__.py:103 def create_app()` | ✅ |
| B2 | 数据存储 | PostgreSQL 16，76 张表 | PostgreSQL 16.15 | `docker-compose.yml:100 image: postgres:16-alpine`；`psql -tAc "show server_version"` → 16.15 | ✅ |
| B3 | 缓存与消息 | Redis 7 + Celery 5 | Redis 7.4.10；Celery 5.4.0 | `docker-compose.yml:121 redis:7-alpine`；`redis-cli INFO server` → 7.4.10；`requirements.txt:165 celery==5.4.0`、`:164 redis==5.2.1`；容器 metadata → celery 5.4.0 / redis 5.2.1 | ✅ |
| B4 | 智能编排 | LangGraph（状态图代理） | langgraph 1.2.6，用于 agent 编排 + checkpoint | `requirements.txt:62 langgraph==1.2.6`；`app/routes/chat.py:9-10,348,352,358`（`create_agent` + `MemorySaver()` + `agent.stream`）；`app/services/agent.py:16 AsyncSqliteSaver`；`app/services/redteam_agent.py:5 create_react_agent`。无手写 `StateGraph` 图定义（"状态图代理"= LangChain `create_agent` 内部 StateGraph + checkpointer） | ✅（表述可接受） |
| B5 | 模型服务 | 双供应商大模型接入 | OpenRouter + NVIDIA NIM（恰 2 个活跃 provider） | `app/services/llm_provider.py:27` `https://openrouter.ai/api/v1`、`:34` `https://integrate.api.nvidia.com/v1`；`app/services/llm_catalog.py:3` | ✅ |
| B6 | 模型服务说明 | **私有化部署，数据不出内网** | **不成立**：provider 全为公网 API | 同上 base_url 均为公网；`docker exec localai-app printenv` 含 OPENROUTER/NVIDIA/DEEPSEEK/QWEN/ZHIPU/SILICONFLOW/MIMO/DASHSCOPE/BOCHA 共 9 个云 key（仅名称，未取明文）；`docker-compose.yml` 六服务中无本地模型/推理服务 | ⚠️需修正（重大） |
| B7 | 检索增强 | ChromaDB + 向量模型 | ChromaDB 1.5.9（嵌入式 PersistentClient）+ sentence-transformers | `requirements.txt:173 chromadb==1.5.9`；`app/services/rag_engine.py:16-17` import chromadb、`:49 PersistentClient(path=data/chromadb)`、`:28 MODEL_NAME='paraphrase-multilingual-MiniLM-L12-v2'` | ✅ |
| B8 | 文档处理 | PyMuPDF / python-docx / OCR | PyMuPDF 1.27.2.3、python-docx 1.2.0、EasyOCR 1.7.2（中英） | `requirements.txt:102,106,22`；`app/services/ocr.py:49-53 easyocr.Reader(['ch_sim','en'])`；`app/services/document_parser.py:1-17` | ✅ |
| B9 | 部署运维 | Docker 编排 + Nginx，六服务容器化 | app/postgres/redis/celery-worker/celery-beat/nginx 恰 6 个 | `docker-compose.yml:48,99,120,138,180,220`；`docker ps` 6 个 `localai-*` 容器 | ✅ |
| B10 | 接入层 | 反向代理 + WSGI | Nginx（12G 请求体上限）+ gunicorn 4 workers gevent 120s | `nginx.conf:29 client_max_body_size 12G`；`Dockerfile:95 gunicorn -w ${WORKERS:-4} -k gevent --bind 0.0.0.0:8000 --timeout 120`；`docker-compose.yml:36 WORKERS=4` | ✅ |
| B11 | 通用/GPU 按需切换 [250] | 支持 | 支持 | `docker-compose.gpu.yml:16,24`（`capabilities: [gpu]`）；`app/services/ocr.py:10-14 OCR_GPU` 开关 | ✅ |

### C. 五级流程对象模型（5.2 / 图5-2）

| # | 断言 | 论文值 | 实测值（真实列名） | 证据 | 结论 |
|---|---|---|---|---|---|
| C1 | 项目 project [231] | 项目单元 | `projects`：id/name/status/created_at/updated_at/created_by | `psql -c "\d projects"` | ✅ |
| C2 | 时间轴：计划与实际起止 [231] | planned/actual start/end | `project_timelines`：**planned_start_date / planned_end_date / actual_start_date / actual_end_date**/status | `psql -c "\d project_timelines"` | ◐概念正确，图5-2 写成 `planned_start…` 缺 `_date` |
| C3 | 里程碑：计划日期、实际日期、偏差天数、偏差原因分类 [231] | 四要素 | `project_timeline_milestones`：planned_date/actual_date/diff_days/diff_reason/reason_category/status | `psql -c "\d project_timeline_milestones"` | ✅ |
| C4 | 环节：责任人与完成状态 [231] | 责任人/完成 | `project_workflow_steps`：step_name/step_order/assigned_to/completed/completed_at | `psql -c "\d project_workflow_steps"` | ✅ |
| C5 | 待办：来自对话、创建/完成时间 [231] | 提取待办 | `project_todos`：content_copy/original_role/original_author/message_id/status/created_at/done_at | `psql -c "\d project_todos"` | ✅ |
| C6 | 可计算指标表（图5-2/表6-1） | workflow_kpi、audit_runs | `workflow_kpi`：generations/revision_rounds/output_chars/completed_count；`audit_runs`：started_at/completed_at/overall_score/overall_status/config_snapshot(jsonb) | `psql -c "\d workflow_kpi"`、`\d audit_runs` | ✅ |
| C7 | 文件版本化 [237] | 每次上传新版本+操作者+时间戳 | `project_file_versions`：version/stored_path/uploaded_by/uploaded_at/comment；上传即插版本 | `psql -c "\d project_file_versions"`；`app/routes/admin.py:1148 INSERT INTO project_file_versions` | ✅ |
| C8 | 评审/清标结果快照化 [237] | 可复现 | `audit_runs.config_snapshot`、`audit_file_results.findings` 均为 jsonb | `psql -c "\d audit_runs"`、`\d audit_file_results`；`app/database.py:595,613` | ✅ |
| C9 | 法规版本化、保留历史、标记受影响条款 [237][245] | 有 | 代码有：`law_version.py:87,95` 写 law_versions/law_articles；`law_change_events`、`law_version_diffs`；`law_monitor.compute_impact` 被 `routes/compliance.py:1019`、`routes/graph.py:94` 调用 | 运行库 `law_masters/law_versions/law_articles` 均 **0 行**（未导入） | ◐能力有、数据无 |

### D. 六大功能模块（5.3 / 表5-2 / 图5-1）

| # | 断言 | 论文值 | 实测值 | 证据 | 结论 |
|---|---|---|---|---|---|
| D1 | 文档智能解析：多格式+OCR+章节条款表格切分+结果缓存 [242] | 有 | 有 | `app/services/document_parser.py:1-3,17,65-92`（章节/条款切分，复用 `law_parser._is_chapter_line/_is_article_line`）；`app/services/file_processing.py:960,1193 for table in doc.tables`；`app/services/ocr.py:49-53`；`file_text_cache` 缓存（`file_cache.py:34-58`） | ✅ |
| D2 | 清标**五维**评分 [243]；图5-1"5 维度复合评分" | 指标分析/交叉比较/合规检查/智能审查/全量审计 | 代码 5 个维度函数：`:99` 指标分析、`:118` 横向对比、`:209` 合规审查、`:272` AI 评审（五轴）、`:316` 全量审计补充检查 | `app/services/clearance_engine.py`；README:15 与论文五维命名一致（指标分析/交叉比较/合规检查/AI 审查/全量审计补充）。注：该文件 L1 docstring 仍写"4 个分析维度"（陈旧注释） | ✅（附注） |
| D3 | "对各维度计算得分，**按权重合成综合评分**" [243] | 加权合成 | 复合指数：基础冒烟指数 + 合规严重度加权（`criticals*5 + violations*2`，封顶 +20）；批量串标另有 RiskScorer 加权（0.375 key + 0.375 attr + 0.25 text） | `app/services/clearance_engine.py:445-451`；`docs/ARCHITECTURE.md:99` | ◐基本吻合，措辞按"0-100 复合指数"更准 |
| D4 | 文本相似度识别异常雷同 [244] | 有 | TF-IDF + jieba 分词 + cosine | `app/services/plagiarism_detector.py:50-71`；`app/services/file_processing.py:11-12 TfidfVectorizer/cosine_similarity` | ✅ |
| D5 | 报价尾数分布 [244] | 有 | `_detect_tailing_digits`：后 2 位相同比例 ≥80% 判串标嫌疑 | `app/services/quote_anomaly.py:482-494,592-598` | ✅ |
| D6 | Benford 检验 [244] | 有 | `_benford_deviation`：MAD/Nigrini 等级/卡方/首位 Z 值，min_samples=20 | `app/services/quote_anomaly.py:346-391` | ✅ |
| D7 | 关联关系构建风险网络 [244] | 有 | `entity_relationships`、`relationship_risk_summary` 表 + networkx 构图 | `app/database.py:1033,1059`；`app/services/relationship_extractor.py:767 G.add_edge`；`app/routes/graph.py` | ✅ |
| D8 | 多类信号汇总为风险提示 [244] | 有 | 尾数/相似度等信号汇入风险报告 | `app/services/clearance_engine.py:592-598`；RiskScorer（`docs/ARCHITECTURE.md:99`） | ✅ |
| D9 | 法规库 15 部（图5-1"合规检查（15 部法规库）"） | 15 部 | **15 部 / 619 条**（运行时加载） | 容器内实测 `_get_seed_laws()` 日志：`Loaded 4 core laws from seed data`、`Loaded 11 extended national laws (full text)`、`Total built-in laws: 15 laws, 619 articles`；`README.md:17`；`CHANGELOG.md:551` | ✅（注：核心 4 部为节选 10/5/5/5 条，11 部为全文） |
| D10 | "国家级法规与部门规章**结构化入库**" [245] | 入库 | 法规从 `data/laws/seed_laws.json` + `data/laws/extended_laws.json` 加载（`compliance_checker.py:42-88`），**未写入 DB**；law 三表 0 行 | `psql -tAc "select count(*) from law_masters"` → 0；同 law_versions/law_articles → 0 | ⚠️需修正 |
| D11 | 条款级规则/关键词映射、增量检查 [245] | 有 | 有 | `app/services/rule_extractor.py:62,206,215`；`app/services/incremental_check.py`（Redis 缓存增量）；`app/services/compliance_checker.py:110 _select_relevant_laws` | ✅ |
| D12 | 智能对话：状态图代理 + RAG + 流式 + 注入防护 [246] | 有 | 有 | `app/routes/chat.py:142,320,403-404`（SSE `text/event-stream`、`stream_with_context`）；`app/services/prompt_safety.py:16` 注入模式表；`app/services/rag_engine.py` | ✅ |
| D13 | 企业信用查询"自动化采集…形成信用报告" [247] | 自动化 | **半自动**：Selenium 驱动 Microsoft Edge 导航/填表，验证码需人工提交（`submit_captcha`/`refresh_captcha`/`capture_viewport`）；容器内无任何浏览器二进制 | `app/services/credit_checker.py:9-15（Edge）、:85-191`；`docker exec localai-app sh -c "which google-chrome chromium msedge"` → 无（rc=1）；`credit_check_reports` 0 行 | ⚠️需修正 |
| D14 | 项目协同：文件共享、**版本对比**、评论、待办 [247] | 有 | 文件/版本历史/评论/待办均有端点；**未见文件级 diff 对比端点** | `app/routes/admin.py:1282 /versions`（版本历史）、`:1333/:1352`（文件评论）、`:1487`（文件夹评论）；`app/routes/admin_ops.py:24,50`（待办 CRUD） | ◐部分吻合 |
| D15 | 纳入统一权限体系与操作审计 [247] | 有 | 有：`users.role`/`is_auditor`；`project_members` 角色 manager/member；`admin_audit_log` 记录后台管理动作（实测 15 行）；`audit_logger` 写操作日志 | `psql -c "\d users"`；`app/routes/projects.py:49-87` 角色判定；`app/database.py:571`、`app/services/admin_utils.py:41-52`；`app/services/audit_logger.py:1` | ✅（若表述为"全量审计"则需修正） |

### E. 部署、定时任务与集成（5.4）

| # | 断言 | 论文值 | 实测值 | 证据 | 结论 |
|---|---|---|---|---|---|
| E1 | 容器化六服务单元 [250] | 应用/数据库/缓存/异步/调度/反向代理 | 完全对应 6 个 compose 服务 | `docker-compose.yml:48,99,120,138,180,220` | ✅ |
| E2 | 20+ 定时任务（图5-1"20+ 定时任务"） | 20+ | Celery Beat **25** 项；APScheduler 24 个 `add_job` | `celery_app.py:53 beat_schedule={…}`（25 个任务键）；`app/__init__.py:254-345`（24 处 `scheduler.add_job`，受 `ENABLE_SCHEDULER` 门控） | ✅ |
| E3 | 定时任务承担数据清理、指标汇总与**法规同步** [250] | 含法规同步 | 数据清理 ✅、指标汇总 ✅（周/月/年报）、LLM 目录刷新/健康检查等；**无法规同步任务** | `celery_app.py:53-…` 25 键清单（`cleanup-*`、`generate-*-report`、`refresh-llm-catalog`、`auto-*-health-check` 等）；全仓 grep 未发现法规同步节拍任务 | ⚠️需修正 |
| E4 | 以接口对接既有业务系统/回传档案系统/财务数据关联 [249] | 已对接 | **无任何实现**；架构文档声明不做 | `docs/ARCHITECTURE.md:98`："本系统是投标文件分析工具，非交易平台对接工具。上述 21 项需各地交易中心/评标系统接口，未纳入开发范围。"；全仓无 ERP/OA/档案/财务对接代码；`app/routes/` 28 模块中无外部集成模块 | ⚠️需修正（重大） |
| E5 | 安全：会话管理、请求限流、文件类型校验、操作审计、上传隔离存储 [250] | 有 | 有 | `flask_limiter`（`app/__init__.py:59-74`）+ `login_guard.py`；`Flask-Session`；`app/routes/upload.py:13,47 allowed_file`；`admin_audit_log`；`file_store.py:5,34`（`user_files/{user_id}/{sha256}_{ts}{ext}` 按用户隔离） | ✅ |
| E6 | 试点—评估—推广—固化 [251] | 实施路径 | 计划性表述，无执行记录 | — | ❓无法核实 |

### F. 时间线与运行实例数据

| # | 断言 | 论文值 | 实测值 | 证据 | 结论 |
|---|---|---|---|---|---|
| F1 | "2025年2月上线智能作业平台" [21][29][98][142][195] | 2025-02 已上线 | 仓库首提交 2026-04-17(+08)，持续开发至 2026-10-18，共 250 次提交；论文封面 2026-10 | A9/A10 的 git 输出；`thesis_v1.txt:16,24,90,125,173` | ❓需作者说明（与代码时间线矛盾） |
| F2 | "试运行观测/一线可用"隐含证据 [257 等第6章] | 有试运行 | 运行实例近乎空库：projects=1、project_timelines=1、project_todos=1、audit_runs=0、credit_check_reports=0、law_*=0、users=6 | `psql -tAc "select count(*)…"`（各表） | ❓无法核实（无业务数据） |

---

## 三、必须修正项的精确替换建议（旧文 → 新文）

### 修正 1（表5-1 "模型服务"行；构建脚本 `build/build_v1.py:61`）
- 旧：`层次=模型服务｜选型=双供应商大模型接入｜说明=私有化部署，数据不出内网`
- 新：`层次=模型服务｜选型=双供应商大模型接入（OpenRouter / NVIDIA NIM）｜说明=平台与数据私有化部署；大模型推理经外部 API，可配置自建模型接口`

### 修正 2（正文 [222] L198，原则二）
- 旧：`其二是私有化部署，模型与数据均部署于公司内网，客户资料与评审信息不出内网，满足保密与合规要求。`
- 新：`其二是私有化部署，平台应用、关系数据库、文件与向量索引均部署于公司内网；大模型推理当前经外部 API 完成，出网内容须以脱敏与合规评估为前置条件，后续可切换至自建本地模型。`

### 修正 3（正文 [249] L221，5.4 系统集成）
- 旧：`在系统集成方面，平台以接口方式与既有业务系统对接：项目主数据由业务系统同步，成果文件由平台生成后回传档案系统，财务数据按项目编号关联，避免重复录入。对暂不具备接口条件的系统，提供标准模板导入导出作为过渡方案…`
- 新：`在系统集成方面，当前版本尚未与既有业务系统打通，仅预留接口与标准模板导入导出能力；项目主数据同步、成果回传档案系统、财务数据按项目编号关联属于后续集成规划——交易平台与评标系统接口因需各地系统配合，未纳入本期开发范围。`

### 修正 4（正文 [245] L217，法规库）
- 旧：`平台将国家级法规与部门规章结构化入库，按条款建立规则与关键词映射…`
- 新：`平台以结构化法规库文件加载 15 部国家级法规（合计 619 条，其中 11 部为全文、4 部核心法律为节选），按条款建立规则与关键词映射…`

### 修正 5（正文 [250] L222，定时任务）
- 旧：`定时任务承担数据清理、指标汇总与法规同步。`
- 新：`定时任务承担数据清理、指标汇总、模型目录刷新与健康检查等 25 项作业；法规变更由管理端触发影响分析。`

### 修正 6（图5-2 字段名；`build/make_figures_project.py:111,123`）
- 旧：`project_timelines: planned_start / planned_end / actual_start / actual_end`
- 新：`planned_start_date / planned_end_date / actual_start_date / actual_end_date`
- 旧：`project_files / versions: version_no / updated_at`
- 新：`project_files: version / uploaded_at；project_file_versions: version / uploaded_at / uploaded_by`

### 修正 7（正文 [247] L219，信用查询）
- 旧：`平台以自动化方式采集公开渠道的企业信用信息，形成信用报告；`
- 新：`平台以浏览器自动化辅助采集公开渠道企业信用信息（验证码等环节需人工介入），形成信用报告；该功能依赖本机浏览器环境，服务器容器部署时需外挂浏览器或由客户端执行；`

### 修正 8（正文 [247] L219，版本对比）
- 旧：`以项目空间承载文件共享、版本对比、评论与待办`
- 新：`以项目空间承载文件共享、版本历史、评论与待办`

### 修正 9（如使用"全量操作审计"表述）
- 旧：`全量操作审计（admin_audit_log）`
- 新：`管理端操作审计（admin_audit_log，记录后台数据变更的新旧值与操作人）+ 业务操作日志（audit_logger 等）`

### 修正 10（交接/画像文档中的首提交日期）
- 旧：`2026-04-16 首次提交`
- 新：`2026-04-16（UTC，即 2026-04-17 01:03 北京时间）首次提交`

### 修正 11（76 张表，建议加注而非改数）
- 旧：`数据库共设计76张表`
- 新：`数据库共设计76张表（以建表语句口径计；当前运行实例另行残留 1 张待迁移清理的历史表）`

### 修正 12（60,461 行口径，建议加注）
- 旧：`60,461 行（find . -name "*.py" | xargs wc -l）`
- 新：`60,461 行（git 跟踪的 197 个 .py 文件，口径：git ls-files '*.py' | xargs wc -l）`

---

## 四、当前环境无法核实的事实

1. **第3章问卷与访谈原始数据**（185 份问卷、10 人访谈、68.6%/45.9%/52.4% 等比例）：仓库与环境无原始数据文件，无法核；本报告不涉及统计口径核验。
2. **"2025年2月上线"这一企业侧事实**：仓库最早记录为 2026-04-17，无法证明 2025 年即已上线；需企业侧材料（上线通知/验收单）。
3. **第6章情景测算**：论文自述为测算非实测（[281] 注），无需代码证据；但运行实例 `audit_runs=0`、`projects=1`，无法提供"试运行观测"支撑。
4. **"数据不出内网"的实际网络行为**：需网络出口审计；代码层面已证明 LLM 调用走公网 API，该表述至少不完整。
5. **11 部法规"条文数与法定一致"**：来自 `CHANGELOG.md:551` 的自述，未与 gov.cn 公报逐条比对（超出只读代码核验范围）。
6. **私有化部署的合规/保密结论**：属企业合规评估范畴，代码无法证明。
7. **"试点—评估—推广—固化"执行情况**：[251] 为计划性表述，无执行记录。
8. **法规库 DB 版本表数据**：表已建但为空（0 行），无法验证“保留历史版本并标记受影响条款”在真实数据上的行为。
9. **Selenium 信用查询的实际运行效果**：容器内无浏览器二进制，无法运行时验证（代码在 Windows/宿主浏览器环境下才具备条件）。

---

## 五、附注（答辩/内部一致性风险，非论文更正项）

1. `app/services/clearance_engine.py:1` docstring 写"统一调度 4 个分析维度"，而实际实现 5 个维度（L99/118/209/272/316）；`indicator_defs.py:1` 写"46 项"，`clearance_engine.py:4` 写"45 项"。若答辩被追问"到底几维/几项指标"，需统一口径（以代码实现为准：5 维度）。
2. `app/services/graph_service.py` 是"知识图谱（Cytoscape）"，与 LangGraph 无关；论文"图"与"状态图代理"勿混。
3. LangGraph 无手写 `StateGraph` 图定义，实际是 `langchain.agents.create_agent` + LangGraph checkpointer（`chat.py:9-10,348-358`）；表述为"状态图代理编排"可接受，但不要说"自研状态图"。
4. 3 次提交带 `Co-Authored-By: Claude Opus 4.7`，其余 247 次无 trailer；作者身份单一（YGY），如被问"是否使用 AI 辅助"，建议如实说明。
5. 运行实例确有 15 条 `admin_audit_log`、6 个用户、1 个项目，可作为"系统可运行"的最小证据，但不构成"生产运行"证据。
6. `migrations/002_drop_typo_detection_results.sql` 未执行：说明"版本化迁移"机制存在但当前实例未全部应用，且这正是 76 vs 77 差异来源。
7. 法规核心 4 部为节选（招标投标法仅 10 条，实际 68 条），README/CHANGELOG 的"全文"表述只对 11 部成立——若论文出现"15 部法规全文"，风险较高（当前 v1.0 正文未出现"全文"，图5-1 仅"15 部法规库"）。

---

## 六、证据清单（命令与输出摘要）

> 全部为只读操作；未修改仓库、未写数据库、未重启容器。命令均可用已验证的可复现形式重跑。

```bash
# 1) 仓库与提交
cd /mnt/d/AI_Tools/Projects/Local_AI
git log --oneline -1                       # aefee49 fix(chat): ...
git log --oneline | wc -l                  # 250
git log --format='%an <%ae>' | sort -u     # YGY <jasoncoop@126.com>
git log --reverse --format='%H|%ai|%ci|%s' | head -1
# 007d20b...|2026-04-17 01:03:28 +0800|2026-04-17 01:03:28 +0800|1st
git log -1 --format='%h %ai'               # aefee49 2026-10-18 20:57:31 +0800
git rev-parse --abbrev-ref HEAD            # master
git status --porcelain | wc -l             # 160（工作区脏）
git log --format='%b' | grep -ci "co-authored"   # 3

# 2) 规模统计
grep -c "CREATE TABLE IF NOT EXISTS" app/database.py        # 76（首个 L124，末个 L1297）
ls app/routes/*.py | wc -l                                  # 29（含 __init__.py → 28 模块）
ls app/services/*.py | wc -l                                # 97
git ls-files '*.py' | xargs wc -l | tail -1                 # 60461 total（197 文件）
grep -rn "register_blueprint" app/routes/__init__.py | wc -l # 16
grep -rn "Blueprint(" app/routes/*.py | wc -l                # 16
wc -l static/js/app.js                                       # 10134（画像文档"约 1 万行"成立）

# 3) 版本（依赖声明 + 容器实测）
grep -inE "^(flask|celery|redis|langgraph|chromadb|selenium|easyocr|pymupdf|python-docx)==" requirements.txt
# Flask==3.1.3 / celery==5.4.0 / redis==5.2.1 / langgraph==1.2.6 /
# chromadb==1.5.9 / selenium==4.45.0 / easyocr==1.7.2 / PyMuPDF==1.27.2.3 / python-docx==1.2.0
docker exec localai-app python -c "import importlib.metadata as md; [print(p, md.version(p)) for p in ['Flask','celery','redis','langgraph','chromadb','selenium']]"
# Flask 3.1.3 / celery 5.4.0 / redis 5.2.1 / langgraph 1.2.6 / chromadb 1.5.9 / selenium 4.45.0
docker exec localai-postgres psql -U localai -d localai -tAc "show server_version;"   # 16.15
docker exec localai-redis redis-cli INFO server | grep redis_version                  # 7.4.10
grep -nE "image:" docker-compose.yml   # postgres:16-alpine / redis:7-alpine / nginx:alpine / local-ai:latest ×3
grep -nE "^  [a-z0-9_-]+:" docker-compose.yml | head -6  # app/postgres/redis/celery-worker/celery-beat/nginx

# 4) 运行库元数据（只读）
docker exec localai-postgres psql -U localai -d localai -tAc \
 "select count(*) from information_schema.tables where table_schema='public';"   # 77
docker exec localai-postgres psql -U localai -d localai -tAc \
 "select count(*) from information_schema.tables where table_schema='public' and table_name='typo_detection_results';"  # 1（迁移002未执行）
docker exec localai-postgres psql -U localai -d localai -c "\d project_timelines"   # planned_start_date … actual_end_date
docker exec localai-postgres psql -U localai -d localai -c "\d project_timeline_milestones"  # planned_date/actual_date/diff_days/reason_category
docker exec localai-postgres psql -U localai -d localai -c "\d project_workflow_steps"       # assigned_to/completed/completed_at
docker exec localai-postgres psql -U localai -d localai -c "\d project_todos"                # message_id/status/created_at/done_at
docker exec localai-postgres psql -U localai -d localai -c "\d workflow_kpi"                 # generations/revision_rounds/output_chars/completed_count
docker exec localai-postgres psql -U localai -d localai -c "\d audit_runs"                   # config_snapshot(jsonb)/overall_score/started_at/completed_at
docker exec localai-postgres psql -U localai -d localai -tAc "select count(*) from law_masters;"       # 0
docker exec localai-postgres psql -U localai -d localai -tAc "select count(*) from law_versions;"      # 0
docker exec localai-postgres psql -U localai -d localai -tAc "select count(*) from law_articles;"      # 0
docker exec localai-postgres psql -U localai -d localai -tAc \
 "select 'admin_audit_log='||count(*) from admin_audit_log union all select 'projects='||count(*) from projects union all select 'project_timelines='||count(*) from project_timelines union all select 'project_todos='||count(*) from project_todos union all select 'audit_runs='||count(*) from audit_runs union all select 'users='||count(*) from users;"
# admin_audit_log=15 / projects=1 / project_timelines=1 / project_todos=1 / audit_runs=0 / users=6

# 5) 15 部法规运行时实测（只读 import，不写库）
docker exec localai-app python -c "
from app.services.compliance_checker import _get_seed_laws
laws=_get_seed_laws(); print('entries',len(laws))
names={}
for L in laws: names[L.get('law_name') or L.get('name')]=names.get(L.get('law_name') or L.get('name'),0)+1
print('distinct_laws',len(names))"
# 日志：Loaded 4 core laws from seed data / Loaded 11 extended national laws (full text)
#       Total built-in laws: 15 laws, 619 articles
#       entries 619 / distinct_laws 15

# 6) 代码证据（grep -n）
#   LLM 供应商：app/services/llm_provider.py:27 openrouter.ai/api/v1；:34 integrate.api.nvidia.com/v1
#   应用工厂：app/__init__.py:103 def create_app()
#   SSE 流式：app/routes/chat.py:142,320,403,404
#   LangGraph：app/routes/chat.py:9-10,348,352,358；app/services/agent.py:16；app/services/redteam_agent.py:5
#   RAG：app/services/rag_engine.py:16-17,28,49
#   OCR：app/services/ocr.py:1,49-53
#   文档解析：app/services/document_parser.py:1-3,17,65-92；file_processing.py:960
#   清标五维：app/services/clearance_engine.py:99,118,209,272,316（+445-451 复合指数）
#   相似度：app/services/plagiarism_detector.py:50-71；file_processing.py:11-12
#   尾数/Benford：app/services/quote_anomaly.py:482-494,346-391,592-598
#   关系网络：app/database.py:1033,1059；app/services/relationship_extractor.py:767
#   法规加载：app/services/compliance_checker.py:42-88；法规版本：law_version.py:87,95；变更影响：law_monitor→compliance.py:1019
#   信用查询：app/services/credit_checker.py:9-15,85-191
#   版本/评论/待办：app/routes/admin.py:1148,1282,1333,1352,1487；app/routes/admin_ops.py:24,50
#   审计：app/database.py:571；app/services/admin_utils.py:41-52；app/services/audit_logger.py:1
#   定时任务：celery_app.py:53 beat_schedule（25 项）；app/__init__.py:254-345（24 个 add_job）
#   集成声明：docs/ARCHITECTURE.md:98（"非交易平台对接工具…未纳入开发范围"）
#   法规口径：README.md:17；CHANGELOG.md:551,555,556
#   图内数字：build/make_figures_project.py:66-92（16 Blueprint/28 路由/97 Service/15 部/76 表/20+ 定时任务）
#   表5-1/5-2：build/build_v1.py:56-73
```

---

## 七、给作者的一句话总结

第5章的**硬指标几乎全部可核**（76 表、16 蓝图、28 路由、97 service、60,461 行、六服务、六版本号、15 部/619 条法规、20+ 定时任务、五级对象与字段全部对得上）；**唯三需要动笔改**的是：①"模型私有化、数据不出内网"（实际走 OpenRouter/NVIDIA 公网 API）；②"已与既有业务系统接口对接/回传档案系统"（无实现，仓库文档明确不做）；③"2025 年 2 月已上线"（代码 2026-04 起才有提交）。另有 6 处表述精度问题（法规"入库"、法规同步、图5-2 字段名、信用查询自动化程度、版本对比、审计"全量"）按第三节替换即可。
