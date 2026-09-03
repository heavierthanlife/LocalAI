# LocalAI — qa-loop 项目参数

本文件被全局 `qa-loop` skill 读取，作为本项目的收集/验证配置（优先级高于 skill 默认兜底）。

## 项目标识
- 名称：LocalAI（中联招标智能助手）
- 技术栈：Flask 3.1 / Python 3.12 / PostgreSQL 16 / Redis 7 / Celery 5
- 前端：templates/index.html + static/js/*.js（无框架，原生 JS SPA 风格）

## 收集范围
- 后端：`app/services/`、`app/routes/`（15 个 blueprint）、`app/database.py`、`app/__init__.py`、`celery_app.py`
- 前端：`static/js/*.js`（重点 app.js / chat.js / knowledge-lab.js / bid-audit.js）、`static/css/`、`static/icons.js`
- 模板：`templates/index.html`
- 配置/规范：`AGENTS.md`、`.env.example`、`docker-compose.yml`、`requirements.txt`、`pytest.ini`
- 修复不变量：`data/fix_registry.yaml`、`tests/test_regression.py`、`scripts/verify_fixes.py`

## 视觉输入
- 截图目录：`.playwright-mcp/`（Playwright MCP 自动落盘；注意截图是时间戳命名，判定"过期"时对照 commit 时间）
- 补拍建议页：清标报告展开详情、知识图谱（法规影响/全局引用）、审计日志详情、编辑提示词、模板编辑器、文件预览弹窗、窄屏(≤768px)

## 测试/验证命令
- 后端回归：`.venv\Scripts\python.exe -m pytest tests/test_regression.py`（Windows 需 `PYTEST_ADDOPTS=-p no:capture`，见 AGENTS.md）
- 全部后端：`tests/run_tests.py`（import-only）
- 前端语法：`node --check <js>`
- 修复不变量：`python scripts/verify_fixes.py`（89 检查）
- 系统健康：`python scripts/check_system.py`

## 基线与记录
- 路径：`data/qa_loop/`（提交跟踪，不 gitignore）
  - `last_head` — 上次已审 commit SHA（增量基线）
  - `round-NNN.md` — 每轮 findings + 判定记录
  - `pending.flag` — post-commit 钩子的"待运行"标记（消费后清空）
- 轮数上限：默认 3（可改：`data/qa_loop/config.json` 或本文件 `max_rounds`）

## 防自激
- git post-commit 钩子（`.githooks/post-commit`）**只写 `data/qa_loop/pending.flag`**，绝不自动运行 loop。

## 测试约定（明确不做/边界）
- pytest smoke 需要 Flask app context 但无 DB；`@pytest.mark.db` 默认 deselect。
- 容器内无 pytest，E2E 用 `docker exec + python 脚本`（见 AGENTS.md）。
- 数据门槛：text_sim 无招标文件时跳过、quote 无结构化开标价仅参考、relationship 权重已调低——QA 关注的是与这些门槛相关的误报。