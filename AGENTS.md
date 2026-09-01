# Local_AI (中联招标智能助手)

AI-powered bidding agency platform. Flask 3.1 / Python 3.12 / PostgreSQL 16 / Redis 7 / Celery 5.

> **文档导航**：`README.md`（入口）· `MANIFEST.md`（模块清单）· `ARCHITECTURE.md`（深度架构）· `CHANGELOG.md`（变更历史）· `DECISIONS.md`（技术决策）· `SECURITY.md`（安全基线）· `USER_MANUAL.md`（运维手册）· `CONTRIBUTING.md`（开发流程）· 本文件为开发者/Agent 命令与约定指南。

## Quick start

```bash
# Copy env and edit
cp .env.example .env
# Enable auto-sync hooks (pre-commit checks reference docs against code)
git config core.hooksPath .githooks
# Dev server (HTTPS :5443, HTTP→HTTPS redirect :5000)
python run.py
# Celery worker (Docker or local Redis)
celery -A celery_app worker -l info -c 2
```

## Testing

Two runners with different scope:

```bash
# Smoke tests — fast, no DB, no external services
pytest tests/test_smoke.py -v

# Integration tests — import-only, no pytest needed
python tests/run_tests.py
```

**Windows Python 3.12 bug**: `ValueError('I/O operation on closed file.')` during session teardown. Set `PYTEST_ADDOPTS=-p no:capture` before running (pytest.ini already has this). If using `create_app` in fixtures, redirect `sys.stdout` to avoid interfering with Flask's `click.echo`.

Tests that need a database are marked `@pytest.mark.db` and deselected by default (`-m "not db"`). To run them, set env vars (`PG_USER`, `PG_PASSWORD`, etc.) and pass `-m db`.

### 容器内测试注意事项（Docker）

容器镜像**不包含 pytest**（`.dockerignore` 排除 `tests/`）。容器内 E2E 验证方式：

```bash
# 1. 把改动的服务文件 + fixture 文档复制进运行中容器
docker cp app/services/xxx.py localai-app:/app/app/services/xxx.py
docker cp tests/fixtures/clearance_baseline/xxx.docx localai-app:/app/tests/fixtures/clearance_baseline/

# 2. 写 Python 验证脚本，docker exec 运行（不依赖 pytest）
docker exec localai-app python /tmp/verify.py
```

- 容器内跑 pytest：需临时 `docker exec localai-app pip install pytest`（不持久化）
- 本地 `pytest` 仍是权威回归 gate；容器 E2E 用于验证运行时行为（清标 8 指标、复合指数、DOCX 报告）

**可选依赖**：`pycorrector`（typo 中文层）、`hanlp`（relationship NER）、`pyspellchecker`/`symspellpy`（英文拼写）均标注在 `requirements.txt` 可选段，缺失时各层自动走后备（domain dict / regex），不影响核心功能。

## Regression Testing

Every feature upgrade must include regression verification:

1. Prepare 3 known bidding documents as baseline (different project types: 工程/货物/服务)
2. Run compliance check before and after the upgrade
3. Compare results: no existing findings may be lost (new findings allowed, lost findings = regression)
4. Record verification results in the upgrade commit message: `regression: 3/3 baseline passed`

适用于合规检查器、规则提取、AI 审查等所有合规相关路径的改动。

## LLM 来源（FIX-016）

- **活跃 provider**：OpenRouter（`:free` 免费池）+ NVIDIA NIM。旧 5 个 provider（deepseek/zhipu/qwen/siliconflow/mimo）已注释保留，`PROVIDER_CONFIG` 只含活跃 2 个。
- **动态免费目录**：`app/services/llm_catalog.py` 每天（Celery beat 2:30 + APScheduler 24h + 启动补拉）拉取两源 `/models`，过滤免费模型存 `data/llm_catalog.json`。**固定白名单 + 动态兜底**：白名单主力 `nvidia/nemotron-3-ultra-550b-a55b:free`（1M ctx），池内缺失则回退第一个免费模型。
- **fallback 链**：`llm_fallback.py` 从 catalog 构建（openrouter → nvidia），`DEFAULT_CHAIN` 兜底。
- **VL**：`vl_model.py` 独立解耦，默认 `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`。
- **环境变量**：`OPENROUTER_API_KEY` / `NVIDIA_API_KEY`（.env / docker-compose 透传）。

## Architecture

| Layer | Location | Notes |
|---|---|---|
| **App factory** | `app/__init__.py` — `create_app()` | Loads `.env`, validates SECRET_KEY, registers all blueprints |
| **Routes** | `app/routes/` — 15 Blueprints | All eager-loaded in `register_all()` at `app/routes/__init__.py` |
| **Services** | `app/services/` | LLM providers, RAG, OCR, audit, training, etc. |
| **Database** | `app/database.py` | psycopg2 pool, schema defined as `CREATE TABLE IF NOT EXISTS` + `ALTER TABLE` in `init_postgres_tables()` |
| **Frontend** | `templates/index.html` + `static/js/app.js` (~10k lines vanilla JS) | SPA with streaming chat, file management, knowledge bases |
| **Async tasks** | `celery_app.py` — Celery with Redis broker | Heavy work: OCR, skill extraction, RAG indexing, nightly training |
| **Scheduler** | APScheduler (in-process) + Celery Beat (Docker) | APScheduler runs in the main Flask process; Celery Beat replaces it in Docker |

## Schema management

- **Baseline** (`migrations/001_initial_snapshot.sql`): just `SELECT 1;`
- **Actual schema**: `app/database.py:init_postgres_tables()` — idempotent `CREATE TABLE IF NOT EXISTS`
- **Migrations tool**: `python scripts/manage_db.py [check|migrate|rollback|history|snapshot]`
- New migration files go in `migrations/` with `.sql` + `.rollback.sql` pairs
- **Anonymous chat history**: `anon_chat_messages(thread_id PK, messages JSONB, updated_at)` — PostgreSQL-backed, atomic UPSERT append via `app/services/anonymous.py` (replaced per-thread JSON files)

## 清标评分 (clearance scoring)

- **权重复合指数**: `total_score` = 0-100 加权复合指数 (`document_analysis_svc._weighted_total_score`)，45 项 `INDICATOR_WEIGHTS` + 每指标 score cap + text_sim 三指标去重。预警阈值：≥60 高度 / ≥30 中等 / <30 正常。
- **RiskScorer**: `0.375 key + 0.375 attr + 0.25 text`（图片权重 0.0，图片相似度未接入清标，仅图片描述抽检生效）。text_sim 需 ≥80% 相似度才计入（模板重叠门槛）。
- **中文停用词**（FIX-013/014/015）：`app/services/stop_words.py` `DEFAULT_STOP_WORDS`（~150 招投标/功能词）在 `tokenize_for_tfidf` 默认过滤；`preprocess_text_for_similarity(text, template_text)` 自适应并入招标文件 top-k 高频词（k=min(200,max(50,len//500)), TF≥2 守卫）。实证判别（FIX-015 词级）：围标(技术雷同) cosine≈0.85 vs 正常(技术不同)≈0.04（裕度 0.8）。
- **异组件守卫**（FIX-014）：`_detect_component` 识别 价格标/技术标/商务标；异组件 pair（如 价格标↔技术标）text_sim/key_sim 归零并标记 `component_mismatch`（结构性条款重叠不作串标依据）。
- **P0 向量化**（FIX-015）：`_make_vectorizer._tokenizer` 必须返回 list（sklearn 逐字符迭代 bug 曾使全部相似度失真）；修复后 TF-IDF 基于真实 jieba 词。
- **子系统校准**（FIX-015）：typo（大写需阿拉伯配对/混淆词上下文守卫/白名单/熔断200）· quote（`_is_price_like` 上下文过滤/`quote_with_open_price` 开标价注入/risk 幅度加权）· relationship（公司/人员正则黑名单/risk 归一化）。
- **数据门槛（避免误报）**：
  - `text_sim` 指标：无招标文件时**跳过**（无法做 `remove_template_content` 模板去除，高余弦是模板重叠非围标）。
  - `quote` 指标：无结构化开标报价时，原始文本价格提取噪声大（日期/行项目被误当报价），结果仅作参考。
  - `relationship`：正常投标天然共享项目名称/主体/人员 → 易误报，权重已调低。
- **基线校准**：`tests/fixtures/clearance_baseline/` 含工程类脱敏文档（价格标金额已替换保留尾数）+ `scores.json` 快照 + `meta.json`（校准范围：仅工程类；`missing_types: [goods, services]` 待真实文档）。`test_clearance_baseline_scores` 锁快照防漂移。
- **样本量免责**: 投标人 <5 时复合指数统计意义有限，报告应提示"样本量小，仅供参考"。

## Key conventions

- **CSRF is opt-in** — disabled by default for JSON API routes via `WTF_CSRF_CHECK_DEFAULT=False`. Not needed for AJAX/JSON endpoints.
- **Sessions**: Filesystem (`data/flask_session/`). `SESSION_TYPE='filesystem'` in `app/__init__.py`. 30-day lifetime. Redis available for Celery but not for Flask sessions.
- **Rate limiting**: `flask-limiter` with Redis backend. Chat 30/min, upload 10/min, login 5/min, global 120/min. Credit check uses custom in-memory limiter (10/5min).
- **Upload semaphore**: max 3 concurrent file uploads (`MAX_CONCURRENT_UPLOADS`). Returns 429 if busy.
- **Time zone**: Asia/Shanghai everywhere (Celery, APScheduler, `beijing_now()` helper)
- **API responses**: `ok(data, message, status)` → `{success:true, message, ...data}`, `err(error, code, status)` → `{success:false, error, code}`
- **Bilingual**: Chinese (primary) + English (code comments, some tooling)
- **File upload limit**: 50 MB (`MAX_CONTENT_LENGTH`)
- **Temp cleanup**: ghost empty chats cleaned on startup, stale sessions every 24h, recycle bin every 3 days
- **repair_kit/**: crash-recovery reference. `repair_kit/SYSTEM_CHECKLIST.md` — auto-generated by `scripts/check_system.py`. `repair_kit/check_integrity.py` verifies checklist covers all blueprints.

## Environment essentials

| Variable | Required | Notes |
|---|---|---|
| `SECRET_KEY` or `FLASK_SECRET_KEY` | Yes | Session signing |
| `OPENROUTER_API_KEY` / `NVIDIA_API_KEY` | At least one | LLM providers (FIX-016: OpenRouter :free pool + NVIDIA NIM) |
| `DATABASE_URL` (Docker) or `PG_USER`+`PG_PASSWORD` (local) | Yes | PostgreSQL |
| `REDIS_URL` | For Celery | Default: `redis://localhost:6379/0` |
| `ADMIN_PIN` | No | Default: `123456`; used for admin accounts |
| `BOCHA_API_KEY` | No | Web search tool |
| `LOG_LEVEL` | No | Root logger level (`INFO`/`DEBUG`). Default: `INFO` |
| `MAX_CONCURRENT_UPLOADS` | No | Concurrent file processing limit. Default: `3` |

Full list in `.env.example`.

## Docker stack

```bash
docker-compose up -d
# 6 services: app, postgres, redis, celery-worker, celery-beat, nginx
# nginx on :80/:443 with HTTPS termination
```

- `app` runs gunicorn with 4 gevent workers on :8000
- `celery-worker`: 2 workers, 50 max tasks per child
- `celery-beat`: hourly/weekly/daily/nightly schedules
- Health check: `curl -f http://localhost:8000/check_auth`

## Edge/Chrome drivers

Auto-downloaded via `webdriver-manager` on first use. Can override with `EDGEDRIVER_PATH` env var. Both are lazy-loaded (not at boot) to avoid slowing startup.

## Scripts

| Command | Purpose |
|---|---|
| `python scripts/manage_db.py check` | Dry-run pending migrations |
| `python scripts/manage_db.py migrate` | Apply pending migrations |
| `python scripts/check_system.py` | Auto-generate system health checklist (108 items) |
| `python scripts/verify_fixes.py` | Validate fix_registry invariants (runs in pre-commit) |
| `python scripts/run_lora_training.py` | LoRA fine-tuning with Unsloth (Qwen2.5-7B) |
| `python scripts/recover_all.py` | Emergency recovery from session dumps |
| `python scripts/remember_sqlite.py` | SQLite schema migration for session handoff |
| `python scripts/migrate_seed_laws.py` | Seed law data migration |
| `python scripts/backfill_wiki_entities.py` | Backfill wiki entity definitions |
| `python scripts/batch_wiki_ingest.py` | Batch ingest markdown into wiki |

## Gotchas

- First startup will fail if no `SECRET_KEY`/`FLASK_SECRET_KEY` is set
- `ENABLE_SCHEDULER=false` when running multiple gunicorn workers to avoid duplicate APScheduler jobs
- Edge/Chrome WebDriver is **not** installed at boot — first Selenium call will auto-download it
- The `migrations/001_initial_snapshot.sql` is a no-op marker — actual schema lives in `app/database.py`
- `pytest` smoke tests need a Flask app context but no database; the `app` fixture in `conftest.py` sets `SECRET_KEY=test-secret-key-for-pytest`
- Windows users: UTF-8 encoding fixes are applied in `run.py` for emoji/log compatibility
- Celery tasks call `init_flask_context()` to get DB access (one-time per worker)
- DB connections are validated with `SELECT 1` on checkout from the pool. Stale connections are closed and retried once.

## Shared Agent Infrastructure

Skills and plugins are centralized at `D:\AI_Tools\shared-agent-infra\` and shared across opencode, hermes, pi, and lmcode.

| Tool | Config | Shared skills mechanism |
|------|--------|------------------------|
| **opencode** | `~/.config/opencode/skills\` → junction to `D:\AI_Tools\shared-agent-infra\skills\` | Junction at global skills dir |
| **hermes** | `D:\AI_Tools\hermes\config.yaml` → `skills.external_dirs` | List entry |
| **pi** | `C:\Users\nana-\.pi\agent\settings.json` → `"skills"` array | Config array |
| **lmcode** | `D:\AI_Tools\npm-global\lm.cmd` / `lm.ps1` → `--skills-dir` flag | CLI arg |

**236 unique skills** in the shared pool. 4 collisions resolved (grill-me identical, spike hermes-wins, tdd merged, docx merged). Losers archived in `skills/_old/`. Per-tool extensions in `tool-extensions/`.

## Long-term Backlog

- *(No items — all blueprints have integration tests.)*

## Fix Registry

When a bug is fixed, add an entry to `data/fix_registry.yaml` with the invariant that must hold. The pre-commit hook runs `scripts/verify_fixes.py` to catch regressions. Bypass with `SKIP_FIX_CHECK=1`. For Python backend fixes, also add a test to `tests/test_regression.py`.
