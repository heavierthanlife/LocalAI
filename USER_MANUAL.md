# USER_MANUAL — 运维与管理手册

面向系统管理员与操作员的日常运维指南。开发者命令与约定见 `AGENTS.md`。

---

## 1. 环境变量

完整模板见 `.env.example`。关键变量：

| 变量 | 必填 | 说明 |
|---|---|---|
| `SECRET_KEY` / `FLASK_SECRET_KEY` | ✅ | 会话签名，缺失启动失败 |
| `DEEPSEEK_API_KEY` / `ZHIPU_API_KEY` / `QWEN_API_KEY` / `SILICONFLOW_API_KEY` | 至少一个 | LLM 提供商 |
| `DATABASE_URL`（Docker）或 `PG_USER`+`PG_PASSWORD`（本地） | ✅ | PostgreSQL |
| `REDIS_URL` | Celery 用 | 默认 `redis://localhost:6379/0` |
| `ADMIN_PIN` | ❌ | 默认 `123456`，管理员账户 |
| `BOCHA_API_KEY` | ❌ | 联网搜索工具 |
| `LOG_LEVEL` | ❌ | 根日志级别，默认 INFO |
| `MAX_CONCURRENT_UPLOADS` | ❌ | 并发上传处理上限，默认 3 |
| `ENABLE_SCHEDULER` | ❌ | 多 worker 必须 `false` |

---

## 2. 启动与停止

### 本地开发

```bash
python run.py                 # HTTPS :5443 + HTTP→HTTPS 重定向 :5000
celery -A celery_app worker -l info -c 2   # Celery worker（另开终端）
```

### Docker 生产

```bash
docker-compose up -d          # 6 服务启动
docker-compose ps             # 查看状态
docker-compose logs -f app    # 跟踪应用日志
docker-compose down           # 停止（保留数据卷）
docker-compose down -v        # 停止并删除数据卷（⚠️ 数据丢失）
```

**服务**：app（gunicorn 4 gevent workers :8000）、postgres（:5433）、redis（:6380）、celery-worker（2 workers, 50 任务/child）、celery-beat、nginx（:80/:443 HTTPS）。

### 更新部署

```bash
docker build -t local-ai:latest .
docker compose up -d --force-recreate app celery-worker
```

---

## 3. 健康检查与排障

| 检查 | 命令 |
|---|---|
| 应用健康 | `curl -f http://localhost:8000/check_auth` |
| 迁移状态 | `python scripts/manage_db.py check` |
| 系统健康清单 | `python scripts/check_system.py` → `repair_kit/SYSTEM_CHECKLIST.md` |
| 修复不变式 | `python scripts/verify_fixes.py` |
| 蓝图覆盖 | `python repair_kit/check_integrity.py` |
| 冒烟测试 | `pytest tests/test_smoke.py -v` |

### 常见问题

| 症状 | 排查 |
|---|---|
| 启动失败「no SECRET_KEY」 | `.env` 未设置 `SECRET_KEY`/`FLASK_SECRET_KEY` |
| 重复定时任务 | 多 worker 未设 `ENABLE_SCHEDULER=false` |
| WebDriver 报错 | 首次 Selenium 自动下载；可设 `EDGEDRIVER_PATH` 指定 |
| DB 连接失败 | 检查 `.env` 连接参数；Docker 内用 `DATABASE_URL`；本地用 `PG_USER`+`PG_PASSWORD` |
| 中文乱码 | Windows 需 UTF-8（`run.py` 已处理） |

---

## 4. 数据库管理

```bash
# 迁移（幂等，actual schema 在 database.py）
python scripts/manage_db.py check      # 干跑待处理迁移
python scripts/manage_db.py migrate    # 应用迁移
python scripts/manage_db.py rollback   # 回滚
python scripts/manage_db.py history    # 迁移历史
python scripts/manage_db.py snapshot   # 导出 schema → repair_kit/SCHEMA_SNAPSHOT.sql
```

**架构约定**：
- `migrations/001_initial_snapshot.sql` 是无操作基线（`SELECT 1;`）
- 实际 schema 在 `app/database.py:init_postgres_tables()`（幂等 CREATE + ALTER）
- 新迁移放 `migrations/`，`.sql` + `.rollback.sql` 成对

---

## 5. 清标操作（核心业务）

清标统一入口 `POST /clearance`，5 维度并行（Celery 异步）：

1. **指标分析**：46 指标，0-100 权重复合指数
2. **交叉比较**：投标文件两两对比（TF-IDF + RiskScorer + 组件守卫）
3. **合规检查**：法规库比对
4. **AI 审查**：LLM 深度审查
5. **全量审计补充**：审计引擎补充

**结果**：推送到聊天会话（`CLEARANCE_REPORT`），可下载 DOCX 报告。

**预警阈值**：≥60 高度 / ≥30 中等 / <30 正常。投标人 <5 时报告含样本量免责声明。

**可选**：上传开标信息表（Excel/CSV/JSON）激活 14 个附加指标（预算价/开标时间/评标办法等）。

---

## 6. 账户与权限

| 角色 | 能力 |
|---|---|
| admin | 全部（项目 CRUD、成员管理、系统配置、DB 浏览、审核） |
| manager | 项目管理、文件、成员 |
| editor | 内容编辑 |
| viewer | 只读 |
| user | 基础使用 |

- 管理员密码从 `ADMIN_PIN` 自动 hash
- 账户创建需 PIN（4/6 位）
- 删除账户流程：请求→确认→选择→执行

---

## 7. 定时任务与清理

**20+ 定时任务**（`app/cleanup_tasks.py`）自动维护：

| 任务 | 周期 |
|---|---|
| 过期 session 清理 | 15 天 |
| 过期文件删除 | 6 小时 |
| 空消息清理 | 1 小时 |
| 回收站清理 | 3 天 |
| 下载令牌清理 | 24 小时 |
| 训练数据清理 | 季度 |
| 周/月/年报生成 | 自动 |

**手动清理**：`docker exec` 进容器执行或重启应用触发启动时清理（ghost 空聊天、过期 session）。

---

## 8. 备份与恢复

**崩溃恢复**：见 `repair_kit/README.md`（环境检查 → 健康清单 → 迁移检查 → 冒烟测试）。

**紧急恢复脚本**：
- `python scripts/recover_all.py` — 从 session 转储紧急恢复
- `python scripts/remember_sqlite.py` — SQLite schema 迁移（会话交接）
- `repair_kit/SCHEMA_SNAPSHOT.sql` — 当前 schema 参考

**数据目录**：`data/`（用户文件、知识库、法规库、上传），`cert/`（SSL），`company_kb_files/`、`knowledge_lab_files/`、`local_cache/`（本地嵌入模型）。

---

## 9. 关键脚本速查

| 命令 | 用途 |
|---|---|
| `python scripts/check_system.py` | 生成系统健康清单（130 项） |
| `python scripts/verify_fixes.py` | 校验 fix_registry 不变式 |
| `python scripts/run_lora_training.py` | LoRA 微调（Unsloth, Qwen2.5-7B） |
| `python scripts/migrate_seed_laws.py` | 法规种子数据迁移 |
| `python scripts/backfill_wiki_entities.py` | Wiki 实体定义回填 |
| `python scripts/batch_wiki_ingest.py` | 批量摄取 markdown 进 wiki |
