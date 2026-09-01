# MANIFEST — Local_AI 项目清单

项目完整模块库存量与统计摘要。用于快速定位代码位置与理解系统组成。

> **来源**：2026-08-27 全项目技术审计。审计报告细节见 [`SECURITY.md`](SECURITY.md)。
> **审计时间**：2026-08-27（审计员：LMcode AI Agent）| **审计基线 commit**：5410aa21

---

## 统计摘要

| 指标 | 值 |
|---|---|
| 项目类型 | 招标代理业务平台（中联招标智能助手） |
| 根目录文件 | ~30 个 |
| Python 路由模块 | 17 蓝图，~12,000 行 |
| Python 服务模块 | 86 个，~26,000 行 |
| 核心应用模块 | 6 个（__init__ / config / database / globals / cleanup_tasks / celery_app） |
| 前端 JS | 15 个，~20,000 行 |
| 前端 CSS | 4 个，~2,500 行 |
| SQL 表 | 70 张 |
| API 端点 | 300+ 个 |
| 定时任务 | 20+ 个 |
| LLM 提供商 | 5 个（DeepSeek/智谱/通义/SiliconFlow/Mimo） |
| Docker 服务 | 6 个 |
| 跟踪文件 | ~300+（排除 node_modules, .venv, __pycache__） |

---

## 根目录文件

| 文件 | 说明 |
|---|---|
| `AGENTS.md` | 项目主开发者指南（完整技术栈、命令、约定、Gotchas） |
| `README.md` | 项目入口文档 |
| `MANIFEST.md` | 本文件：项目清单 |
| `ARCHITECTURE.md` | 深度架构文档 |
| `CHANGELOG.md` | 版本化变更历史 |
| `DECISIONS.md` | 技术决策记录 |
| `SECURITY.md` | 安全审计与修复基线 |
| `USER_MANUAL.md` | 运维/管理员操作手册 |
| `CONTRIBUTING.md` | 开发流程指南 |
| `Dockerfile` (60 行) | python:3.12-slim 基础，libpq/gcc/OpenCV/Edge/LibreOffice+CJK，非 root `localai` 用户，gunicorn 4 workers gevent 120s |
| `Dockerfile.celery-fix` (2 行) | 最小覆盖 Dockerfile，仅更新 celery_app.py |
| `docker-compose.yml` (195 行) | 6 服务编排，4 命名卷 + 1 桥接网络 |
| `run.py` (112 行) | 本地开发入口（Windows GBK 修复、SSL、HTTP→HTTPS 重定向） |
| `requirements.txt` (206 行) | ~156 个包（完整锁定） |
| `requirements_min.txt` (51 行) | 精简依赖（无版本锁定） |
| `nginx.conf` (108 行) | 反向代理，client_max_body_size 12G，HTTPS TLS 1.2/1.3 |
| `celery_app.py` (90 行) | Redis broker+backend，8 task 模块，6 beat 调度 |
| `.env.example` (52 行) | 环境变量模板 |
| `.mcp.json` (45 行) | 5 个 MCP 服务器（filesystem/postgres/git/agentmemory/skill） |
| `opencode.json` (63 行) | OpenCode 配置（4 MCP + 5 斜杠命令） |
| `pytest.ini` (11 行) | 4 markers: slow/e2e/db/redis |
| `.gitignore` / `.dockerignore` | 排除规则 |
| `skills-lock.json` (35 行) | 5 个外部 skill 锁定 |

---

## 核心应用层 `app/`

| 文件 | 行数 | 功能 |
|---|---|---|
| `__init__.py` | 313 | App factory：加载 .env、SECRET_KEY 校验、filesystem session (30天)、CSRF opt-in、Rate Limiter (Redis 120/min)、Admin 密码 hash、413 处理器、Swagger、17 Blueprint、APScheduler 20+ 任务 |
| `_core.py` | 8 | 兼容存根（路由已全部迁移 Blueprint） |
| `config.py` | 125 | 路径常量（BASE_DIR 等 8 个）、`to_rel_path()`/`resolve_path()`、日志（console INFO + rotating file DEBUG 10MB×5）、WebDriver 定位、扩展名白名单 |
| `database.py` | 600+ | PG 连接池（SimpleConnectionPool 1-20）、三级连接参数、健康检查 + 自动重连、事务 manager、70 表初始化 |
| `globals.py` | 100 | 全局单例（_agent RLock、checkpointer、_semantic_model、user_active_tasks、download_tokens、credit_tasks、admin_rate_limit） |
| `cleanup_tasks.py` | 554 | 20+ 定时清理任务（session/文件/锁/回收站/训练数据/报告） |
| `celery_app.py` | 90 | 见根目录 |

---

## 工具模块 `app/utils/`

| 文件 | 行数 | 功能 |
|---|---|---|
| `helpers.py` | 90 | `ok()`/`err()` 标准化 JSON 响应、`beijing_now()`/`utc_now()`、`split_thinking_answer()` |
| `chinese_numbers.py` | 141 | 中文数字解析、`parse_daxie_amount()` 大写金额 |
| `headroom_utils.py` | 69 | Headroom 上下文压缩 |
| `mailer.py` | 79 | SMTP 邮件（TLS/SSL，异步线程） |

---

## 路由层 `app/routes/`（17 蓝图）

| 文件 | 行数 | 端点数 | 功能 |
|---|---|---|---|
| `__init__.py` | 163 | — | 注册中枢，`register_all()` 逐一注册计时 |
| `admin.py` | 1,653 | 100+ | 项目管理/成员/文件/版本/评论/数据库浏览/AI SSE/工作流/KPI/系统配置/DB 迁移/LLM 状态/审核 |
| `chat.py` | 1,122 | 40+ | SSE 流式聊天、会话管理、文件管理（SHA-256 去重）、回收站、分享、全文搜索、LLM 切换 |
| `knowledge.py` | 941 | 70+ | 知识实验室、skill 生成/审计/合并、训练管线、写作风格、RAG 管理、公司知识库 |
| `compliance.py` | 937 | 23 | 法规 CRUD、规则提取、合规检查、增量检查、趋势/仪表板/图谱 |
| `auth.py` | 605 | 10 | check_auth、create_account、login、PIN 变更、账户删除流程 |
| `timeline.py` | 495 | 17 | 时间线、里程碑、法律分类、工作流步骤 |
| `wiki.py` | 442 | 18 | Wiki 索引/页面 CRUD/树/搜索/stats/实体图谱 |
| `graph.py` | 435 | 5 | 串通图、合规图、法规影响图、引用网图（已加认证） |
| `credit.py` | 355 | 9 | Selenium 企业信用查询 |
| `batch.py` | 351 | 9 | 批量结果下载、报价异常检测、投标人比较、反馈、关系提取、错别字 |
| `templates.py` | 316 | 14 | 模板 CRUD、.docx 导入导出、版本管理、AI 推荐 |
| `audit.py` | 306 | 8 | 全量审计（**已摘除注册**，功能合并入清标；文件保留） |
| `cases.py` | 271 | 14 | 案例 CRUD、法律关联、模板关联、auto-generate |
| `document_analysis.py` | 123 | 3 | 深度分析（1-10 文件 Celery + SSE） |
| `tasks.py` | 115 | 5 | 任务列表/状态/删除/取消/SSE |
| `projects.py` | 104 | 1 | 项目路由 + 7 权限检查工具函数 |
| `clearance.py` | 173 | 3 | 统一清标入口（5 维度 Celery 异步 + status + stream SSE） |
| `upload.py` | 61 | 1 | 流式上传（8MB 分块，10GB 配额） |

---

## 服务层 `app/services/`（86 模块，~26,000 行）

### 核心基础设施
| 文件 | 行数 | 功能 |
|---|---|---|
| `task_bus.py` | 256 | 异步任务总线（Redis pub/sub → SSE） |
| `task_locking.py` | 32 | 每用户单任务锁 |
| `session_manager.py` | 326 | 会话/聊天/消息管理 |
| `runtime_config.py` | 330 | 100+ 运行时配置 |
| `redis_client.py` | 30 | Redis 单例 |
| `admin_utils.py` | 51 | 频率限制 + 审计日志 |
| `_save_helper.py` / `_shared_helpers.py` | 41/85 | DB 保存装饰器 / AI 审查 prompt + 回收站恢复 |

### AI/LLM 核心
| 文件 | 行数 | 功能 |
|---|---|---|
| `agent.py` | 310 | LangGraph 代理（Bocha 搜索 + 日期工具，72h 缓存） |
| `agent_middleware.py` | 65 | InvalidToolGuard 幻觉工具调用防护 |
| `llm_provider.py` | 276 | 5 LLM 路由 |
| `llm_fallback.py` | 206 | Fallback 链（指数退避、熔断器） |
| `redteam_agent.py` | 69 | 质问/红队对抗审查 |
| `judge_review.py` | 95 | Judge 模型二次审查 |
| `prompt_safety.py` | 306 | Prompt 安全层（注入防护 / anti-hallucination） |
| `analysis_prompts.py` / `compliance_prompts.py` | 180/194 | 分析 / 合规 prompt |

### 文档处理
| 文件 | 行数 | 功能 |
|---|---|---|
| `file_processing.py` | 1,551 | 全格式文本提取 + 相似度 + VL 描述 + 熔断器 |
| `document_parser.py` | 305 | 统一文档解析器 |
| `document_classifier.py` | 168 | 两层分类（正则 + LLM） |
| `document_schema.py` | 57 | 中间文档格式 |
| `file_cache.py` | 153 | 两级缓存（PG + 内存 LRU） |
| `file_store.py` | 162 | 流式上传（8MB 分块，SHA256 去重） |
| `file_generator.py` | 250 | Markdown→DOCX/XLSX/PPTX |
| `ocr.py` | 87 | EasyOCR 线程安全单例（`OCR_GPU=auto` GPU 探测） |
| `document_analysis_svc.py` | 1,288 | 文档深度分析（6 检查器，DOCX 报告） |
| `text_utils.py` | — | 分词 / TF-IDF / top_keywords |

### 合规与审计
| 文件 | 行数 | 功能 |
|---|---|---|
| `compliance_checker.py` | 489 | 合规引擎（规则→法规→语义→LLM） |
| `rule_extractor.py` | 262 | 规则提取（AI + 正则双通道，5 类） |
| `audit_engine.py` | 775 | 审计编排（8 评分函数，SSE，DOCX/XLSX，Wiki） |
| `audit_report.py` | 766 | 专业中文报告生成 |
| `audit_logger.py` | 88 | 结构化审计日志 |
| `audit_wiki_publisher.py` | 280 | 审计→Wiki 发布 |
| `incremental_check.py` | 164 | 增量合规检查 |
| `compare_service.py` / `dashboard_service.py` | 152/168 | 跨项目比较 / 合规仪表板 |
| `clearance_engine.py` | 483 | 清标编排（5 维度并行，Celery） |

### 投标检测
| 文件 | 行数 | 功能 |
|---|---|---|
| `batch_orchestrator.py` | 1,004 | 批量比较（RiskScorer，TF-IDF，gang detection，组件守卫） |
| `batch_compare_svc.py` | 32 | 批量比较辅助（TF-IDF 矩阵预计算） |
| `quote_anomaly.py` | 557 | 报价异常（Benford/变异系数/聚类/大写验证/尾数检测） |
| `relationship_extractor.py` | 680 | 实体关系提取（HanLP + 正则，天眼查 API） |
| `indicator_defs.py` | 533 | 46 个检测指标定义 |
| `stop_words.py` | — | 中文停用词表（~150 招投标/功能词） |
| `clearance_openinfo.py` | — | 开标信息表解析（Excel/CSV/JSON）+ 评审标准提取 |

### 法律系统
| 文件 | 行数 | 功能 |
|---|---|---|
| `law_parser.py` | 98 | 法律文本解析 |
| `law_semantic.py` | 152 | 语义法律搜索（ChromaDB） |
| `law_version.py` / `law_monitor.py` | 232/174 | 版本管理 / 变更监控 |
| `legal_schedule_service.py` | 324 | 法律进度模板 |
| `region_manager.py` | 118 | 区域变体管理 |

### 知识库与 Wiki
| 文件 | 行数 | 功能 |
|---|---|---|
| `rag_engine.py` | 466 | RAG 引擎（ChromaDB 4 collection，LRU 5000） |
| `kb_skill_engine.py` / `skill_auditor.py` / `skill_compiler.py` / `skill_validator.py` | 268/401/170/74 | Skill 生成/审计/编译/验证 |
| `ingest_pipeline.py` | 815 | 批量摄取（ZIP→OCR→4 管线） |
| `notebook.py` | 165 | 个人笔记本（ChromaDB RAG） |
| `project_wiki_publisher.py` | 210 | 项目→Wiki 发布 |
| `wiki_*.py` | — | Wiki 引擎/实体/摄取/树状结构 |

### 项目管理
| 文件 | 行数 | 功能 |
|---|---|---|
| `project_timeline_service.py` | 427 | 时间线 CRUD |
| `case_service.py` | 429 | 案例库 |
| `recycle_bin_service.py` | 403 | 统一回收站（4 表，递归恢复） |
| `workflow_*.py` / `template_*.py` | — | 工作流 / 模板引擎 |

### 用户与安全
| 文件 | 行数 | 功能 |
|---|---|---|
| `auth_jwt.py` | 108 | JWT 认证 |
| `anonymous.py` | 99 | 匿名用户管理（PG JSONB 原子 UPSERT） |
| `credit_checker.py` | 184 | Selenium 企业信用查询 |
| `style_engine.py` / `suggestion_engine.py` | 444/368 | 写作风格 / 时间线建议 |

### 其他
`context_utils.py`(187), `graph_protocol.py`(78), `graph_service.py`(264), `langutils.py`(58), `semantic.py`(157), `lora_trainer.py`(217), `nightly_trainer.py`(393), `review_logger.py`(99), `vl_model.py`, `web_extractor.py`, `trend_service.py`, `typo_detector.py`, `score_analyzer.py`（Grubbs/Kendall W/Spearman）等。

---

## 前端 `templates/` + `static/`

| 文件 | 行数 | 功能 |
|---|---|---|
| `templates/index.html` | 812 | SPA 壳（highlight.js, markdown-it, DOMPurify） |
| `static/js/app.js` | 9,798 | 主 SPA 逻辑（聊天/文件/项目/管理） |
| `static/js/knowledge-lab.js` | 2,783 | 知识实验室 |
| `static/js/review.js` | 1,408 | AI 审查 |
| `static/js/chat.js` | 1,104 | 聊天（含清标报告渲染） |
| `static/js/compliance.js` | 1,013 | 合规 |
| `static/js/bid-audit.js` | 847 | ⚠️ 未加载（审计合并入清标后遗留） |
| `static/js/wiki.js` / `templates.js` / `file-station.js` / `accounts.js` / `cases.js` / `tiptap-editor.js` / `graph-view.js` | 693/605/561/548/513/394/335 | 各功能面板 |
| `static/css/` | 4 文件 | app.css(1,834) / darkly.css / flatly.css / tokens.css(53) |
| `static/` PWA | — | manifest.json, sw.js, favicon.ico, icons/ |

---

## 目录速查

```
app/                 核心应用（__init__/config/database/globals/cleanup_tasks）
app/routes/          17 蓝图
app/services/        86 服务模块
app/utils/           工具函数
templates/           index.html（SPA 壳）
static/              JS/CSS/PWA 资源
migrations/          迁移 SQL（001 为无操作基线）
scripts/             运维脚本（16 个）
tests/               测试（smoke/regression/batch/unit/e2e/integration/factories/mock_data）
tools/skill-mcp/     自定义 MCP 服务器（236 skill 暴露为 MCP tools）
repair_kit/          崩溃恢复参考（SYSTEM_CHECKLIST + SCHEMA_SNAPSHOT + check_integrity）
data/                agent_prompt.json, 法规库, 上传文件, 知识库等运行时数据
docs/                功能规格 + 升级方案
.audit/              审计增量日志 + state
.remember/           会话记忆（AI 代理）
.githooks/           提交钩子（pre-commit 校验 fix registry）
```

---

*清单由 2026-08-27 技术审计归纳整理，2026-09-01 正规化。*
