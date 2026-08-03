# Repair Kit — Local_AI 系统恢复参考

**目的：** 崩溃后或长时间未操作后，快速定位到"系统正常应该是什么样"的参考信息。

## 崩溃后第一步

1. **环境检查** — `.env` 是否存在，`SECRET_KEY` 是否设置（`AGENTS.md` §环境 essentials）
2. **健康清单** — 逐项确认系统正常：`repair_kit/SYSTEM_CHECKLIST.md`
3. **数据库** — `python scripts/manage_db.py check` 确认迁移状态
4. **快速测试** — `pytest tests/test_smoke.py -v`（无需数据库）

---

## 参考文件地图

| 文件 | 内容 | 适用场景 |
|---|---|---|
| `AGENTS.md` | 架构、命令、环境变量、Gotchas | 首次部署、迁移、排障 |
| `USER_MANUAL.md` | 管理员操作手册（中文） | 日常运维、功能操作 |
| `.env.example` | 完整环境变量配置模板 | 重建 `.env` |
| `CHANGELOG.md` | 变更历史（日期排序） | 想知道最近改了什么 |
| `IMPROVEMENTS_SKIPPED.md` | 已评估但未采用的技术方案 | 技术选型决策追溯 |
| `.audit/last-full-audit.md` | 安全/可靠性审计基线 | 安全审查、合规检查 |
| `docs/superpowers/` | 功能规格说明书 | 功能实现细节、API 设计 |
| `tests/test_smoke.py` | 快速冒烟测试（6项） | 不依赖数据库的快速验证 |
| `scripts/manage_db.py` | 数据库迁移管理 | 迁移、回滚、历史查询 |
| `docker-compose.yml` | Docker 全栈部署 | Docker 生产环境 |

---

## 关键命令速查

```bash
# 开发环境
python run.py                          # 启动（HTTPS :5443 + HTTP :5000 重定向）
celery -A celery_app worker -l info -c 2  # Celery 工作进程

# 数据库
python scripts/manage_db.py check      # 检查待处理迁移
python scripts/manage_db.py migrate    # 应用迁移
python scripts/manage_db.py snapshot   # 导出当前 schema

# 测试
pytest tests/test_smoke.py -v          # 冒烟测试（无需 DB）
python repair_kit/check_integrity.py   # 检查清单覆盖率（无依赖）

# Docker
docker-compose up -d                   # 全栈启动（6 服务）
```

---

## 维护约定

- **修改了系统行为**（新增路由、修改数据库、变更配置）→ 同步更新 `SYSTEM_CHECKLIST.md`
- **`SCHEMA_SNAPSHOT.sql`** 在 `python scripts/manage_db.py migrate` / `rollback` 时自动重新生成并复制到本目录
- **`repair_kit/check_integrity.py`** 会自动检查清单是否覆盖所有 blueprints。运行 `python repair_kit/check_integrity.py`
- **预提交钩子**（`.githooks/pre-commit`）在 `git commit` 时自动检测引用文件与代码的偏差并给出警告。启用方式：`git config core.hooksPath .githooks`
