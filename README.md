# Local_AI — 中联招标智能助手

AI 驱动的招标代理业务平台，覆盖文档解析、智能对话、合规检查、串通投标检测、清标评分、企业信用查询、知识库与 Wiki 等核心业务。

**技术栈**：Flask 3.1 / Python 3.12 / PostgreSQL 16 / Redis 7 / Celery 5 / vanilla JS SPA

> **项目文档导航**：本文档是入口。开发者指南见 [`AGENTS.md`](AGENTS.md)，架构深度见 [`ARCHITECTURE.md`](ARCHITECTURE.md)，模块清单见 [`MANIFEST.md`](MANIFEST.md)，变更历史见 [`CHANGELOG.md`](CHANGELOG.md)，技术决策见 [`DECISIONS.md`](DECISIONS.md)，安全基线见 [`SECURITY.md`](SECURITY.md)，运维操作见 [`USER_MANUAL.md`](USER_MANUAL.md)，开发流程见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

---

## 核心能力

- **AI 智能对话**：LangGraph 代理 + 流式 SSE，多轮上下文，RAG 知识库检索，质问/红队模式
- **文档全格式解析**：PDF/DOCX/XLSX/PPTX/扫描件 OCR（MarkItDown + LibreOffice + PyMuPDF + EasyOCR）
- **清标分析**：5 维度（指标分析 / 交叉比较 / 合规检查 / AI 审查 / 全量审计补充），0-100 权重复合评分
- **串通投标检测**：TF-IDF 文本相似度 + 组件守卫 + 报价尾数/Benford 异常检测 + 关系网络
- **合规检查**：法规库（24 部）+ 规则提取 + 语义检索 + 增量检查
- **企业信用查询**：Selenium 自动化政府官网查询
- **知识库 / Wiki**：RAG 摄取、技能库、法规版本管理、图谱

## 技术架构一览

| 层 | 技术 | 说明 |
|---|---|---|
| Web | Flask 3.1 + gunicorn/gevent | App factory，17 Blueprint |
| 数据库 | PostgreSQL 16 + psycopg2 连接池 | 70 张表，`CREATE TABLE IF NOT EXISTS` 幂等管理 |
| 缓存/消息 | Redis 7 + Celery 5 | Broker + Beat 调度 |
| AI/LLM | LangChain + LangGraph + DeepSeek/智谱/通义/SiliconFlow/Mimo | 5 供应商 + fallback 链 + 熔断器 |
| 向量 | ChromaDB + sentence-transformers | 语义检索、RAG、知识库 |
| 文档处理 | python-docx / PyMuPDF / openpyxl / MarkItDown / EasyOCR | 全格式提取 + OCR |
| 自动化 | Selenium + Edge/Chrome WebDriver | 企业信用查询 |
| 前端 | 原生 JS SPA + Tiptap + markdown-it + DOMPurify | 流式聊天 + 文件管理 + 知识库 |
| 调度 | APScheduler（进程内）/ Celery Beat（Docker） | 20+ 定时任务 |
| 部署 | Docker（6 服务）+ nginx HTTPS | app/postgres/redis/worker/beat/nginx |

## 快速开始

```bash
# 1. 复制并编辑环境变量
cp .env.example .env

# 2. 启用提交钩子（pre-commit 校验文档与代码一致性）
git config core.hooksPath .githooks

# 3. 开发服务器（HTTPS :5443，HTTP→HTTPS 重定向 :5000）
python run.py

# 4. Celery worker（Docker 或本地 Redis）
celery -A celery_app worker -l info -c 2
```

> 首次启动无 `SECRET_KEY`/`FLASK_SECRET_KEY` 会失败——必须先在 `.env` 设置。

## Docker 部署

```bash
docker-compose up -d
# 6 服务: app, postgres, redis, celery-worker, celery-beat, nginx
# nginx 在 :80/:443，HTTPS 终结
# 健康检查: curl -f http://localhost:8000/check_auth
```

完整服务说明见 [`USER_MANUAL.md`](USER_MANUAL.md) §Docker 部署。

## 测试

```bash
# 冒烟测试 — 快、无 DB、无外部服务
pytest tests/test_smoke.py -v

# 集成测试 — 仅导入，无需 pytest
python tests/run_tests.py

# 全量回归
pytest tests/test_regression.py tests/test_batch_orchestrator.py tests/test_smoke.py

# 修复不变异校验
python scripts/verify_fixes.py
```

**Windows Python 3.12 注意**：`ValueError('I/O operation on closed file.')` 出现在 session teardown。`pytest.ini` 已设置 `PYTEST_ADDOPTS=-p no:capture`。

## 文档地图

| 文件 | 内容 | 适合谁 |
|---|---|---|
| [`README.md`](README.md) | 项目入口：能力、架构、快速开始 | 所有人 |
| [`AGENTS.md`](AGENTS.md) | 开发者指南：命令、约定、Gotchas、Fix Registry | AI Agent / 开发者 |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | 逐模块深度架构 + 模块清单 | 架构师 / 新人上手 |
| [`MANIFEST.md`](MANIFEST.md) | 项目清单：模块库存量、统计摘要 | 快速定位代码 |
| [`CHANGELOG.md`](CHANGELOG.md) | 版本化变更历史 | 想知道最近改了什么 |
| [`DECISIONS.md`](DECISIONS.md) | 技术决策记录（ADR）+ 已评估未采纳方案 | 技术选型追溯 |
| [`SECURITY.md`](SECURITY.md) | 安全基线：审计发现 + 修复 + Fix Registry | 安全审查 |
| [`USER_MANUAL.md`](USER_MANUAL.md) | 运维/管理员操作手册 | 运维 / 操作员 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 开发流程：测试、回归、提交、评审 | 开发者 |
| [`repair_kit/README.md`](repair_kit/README.md) | 崩溃恢复参考 | 排障恢复 |
| [`docs/superpowers/`](docs/superpowers/) | 功能规格说明书（SDD） | 功能实现细节 |
| [`docs/wiki_upgrade_plan.md`](docs/wiki_upgrade_plan.md) | Wiki 系统升级方案 | 路线规划 |
