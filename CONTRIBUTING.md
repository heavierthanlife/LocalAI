# CONTRIBUTING — 开发流程指南

为本地_AI 项目贡献代码的流程与约定。

---

## 开发环境

```bash
# 前置：Python 3.12, PostgreSQL 16, Redis 7
cp .env.example .env          # 编辑 SECRET_KEY 等
git config core.hooksPath .githooks   # 启用提交钩子

# 依赖
pip install -r requirements.txt       # 完整（~156 包）
# 或精简
pip install -r requirements_min.txt   # 无版本锁定
```

---

## 分支与提交

- 分支：`master`（单主线开发）
- 提交信息：`type: 中文摘要`（feat/fix/docs/refactor/style/test/chore）
- **不要**推送包含密钥/私钥/二进制（`cert/key.pem`、`msedgedriver.exe`）

### 提交钩子

`.githooks/pre-commit` 自动执行：
1. `scripts/verify_fixes.py` — 校验 `data/fix_registry.yaml` 不变式
2. `repair_kit/check_integrity.py` — 校验健康清单覆盖所有 blueprints

旁路：`SKIP_FIX_CHECK=1`（仅调试用，不要常规使用）。

---

## 测试规范

### 测试运行器

| 运行器 | 命令 | 范围 |
|---|---|---|
| 冒烟 | `pytest tests/test_smoke.py -v` | 快、无 DB、无外部服务 |
| 回归 | `pytest tests/test_regression.py` | FIX 回归锁 |
| 批量 | `pytest tests/test_batch_orchestrator.py` | 串通检测/TF-IDF |
| 集成 | `python tests/run_tests.py` | 仅导入，无 pytest |
| 全量 | `pytest tests/test_smoke.py tests/test_document_pipeline.py tests/test_regression.py tests/test_route_preservation.py tests/test_batch_orchestrator.py` | 92/92 |

**Windows Python 3.12 bug**：session teardown 报 `ValueError('I/O operation on closed file.')`。`pytest.ini` 已设 `PYTEST_ADDOPTS=-p no:capture`。fixture 中用 `create_app` 时重定向 `sys.stdout`。

**DB 测试**：标记 `@pytest.mark.db`，默认排除（`-m "not db"`）。运行需设 `PG_USER`/`PG_PASSWORD` 等 env + `-m db`。

---

## 回归测试（必须）

> 适用于合规检查器、规则提取、AI 审查等所有合规相关路径的改动。

每次功能升级必须：

1. 准备 3 个已知招标文件作为基线（不同项目类型：工程/货物/服务）
2. 升级前后分别运行合规检查
3. 对比结果：**不允许丢失已有 finding**（新发现允许，丢失=回归）
4. 验证结果记录到 commit message：`regression: 3/3 baseline passed`

基线夹具：`tests/fixtures/clearance_baseline/`（工程类脱敏文档 + scores.json + meta.json）。

---

## Fix Registry 规范

修复 bug 时：

1. 向 `data/fix_registry.yaml` 添加条目（含不变式 + grep 校验）
2. Python 后端修复必须向 `tests/test_regression.py` 添加回归测试
3. 运行 `python scripts/verify_fixes.py` 确认 100% 通过
4. commit message 记录 FIX 编号

**当前状态**：70/70 校验通过。

---

## 文档维护约定

| 文件 | 何时更新 |
|---|---|
| `README.md` | 功能入口、快速开始变化 |
| `ARCHITECTURE.md` | 架构层变化（新蓝图/新服务/数据流变化） |
| `MANIFEST.md` | 模块数量/行数/统计变化 |
| `CHANGELOG.md` | 每次功能/修复（顶部新增，含 regression 标注） |
| `DECISIONS.md` | 技术选型决策 |
| `SECURITY.md` | 安全发现/修复 |
| `USER_MANUAL.md` | 运维命令/配置/操作变化 |
| `AGENTS.md` | 开发命令、约定、Gotchas、环境变量 |
| `repair_kit/SYSTEM_CHECKLIST.md` | 修改系统行为（新路由/表/配置）后由 `check_system.py` 自动更新 |
| `20260827log.md` | 迭代详细记录（§10.x 追加） |

---

## 代码规范

- **双语言**：中文为主，英文代码注释
- **API 响应**：`ok()`/`err()`（`app/utils/helpers.py`）
- **时区**：Asia/Shanghai（`beijing_now()`）
- **路径**：`to_rel_path()`/`resolve_path()`（不写死绝对路径）
- **CSRF**：JSON API 无需（opt-in）
- **不添加注释**除非必要（遵循仓库风格）

---

## 评审清单

提交前自检：
- [ ] 全量测试通过（92/92）
- [ ] `verify_fixes.py` 通过（70/70）
- [ ] 合规路径改动完成 3/3 基线回归
- [ ] 无密钥/二进制入仓
- [ ] CHANGELOG 已更新
- [ ] 修复已登记 fix_registry + 回归测试
