# QA-Loop Round 029 (2026-09-24)

基线: last_head=c999e37 | 模式: 定向修复（IMPLEMENT）＋ 视觉专项 COLLECT（下步）| 触发: manual

> 方向（用户定）：① UNRESOLVED-029  owner 校验统一封装；② UNRESOLVED-028 视觉专项 COLLECT。

---

## ① 背景与来源

- **工作流 A（FIX-066）**：QA-Loop Round 028 @code-reviewer 的 M2 / `UNRESOLVED-029` —— `clearance.py`/`tasks.py`/`batch.py` 的 owner 校验在 `TaskBus.get` 抛异常时 500，与 `compliance._task_forbidden` 的 fail-closed 403 不一致。
- **工作流 B（FIX-028/视觉）**：Round 028 视觉侧因截图过期（2026-09-11）+ `visual_regression.py` 覆盖面不足（10 泛化 CP）无法判定；需扩展脚本 + 数据夹具并在新镜像上重拍。

## ④ CONFIRM（用户 2026-09-24 批准）

- [x] 决策 1：TaskBus 异常时 clearance/tasks/batch 统一 **403 fail-closed**（原 500）
- [x] 决策 2：`credit.py`（自有 `_credit_get` registry，非 TaskBus）**不纳入**，仅记备注
- [x] 决策 3：视觉采集用 **基线 docx 自动播种清标任务**（非仅拍无数据页）

## ⑤ IMPLEMENT — 工作流 A（FIX-2026-09-24-066）

- 新增 `app/utils/helpers.py::load_task_for(task_id, user_id) -> (meta, status)`，`status ∈ {ok, missing, forbidden}`；内部 `TaskBus.get` + `task_owner_ok`，**异常 → forbidden + logger.warning**（fail-closed）。
- 调用点改造：`clearance.py`（`/status`、`/stream`）· `tasks.py`（get/delete/cancel/stream）· `batch.py`（`plagiarism_status`）；`compliance.py::_task_forbidden` → 薄封装委托。
- 语义：`missing` 在 clearance/tasks/batch → 404；compliance → 放行（结果超 TTL 持久化）。

### 评审与合入前修复（@code-reviewer）
- **Critical（已修）**：`clearance_stream` 重构误删局部 `from app.services.task_bus import TaskBus`，函数体仍用 `TaskBus.subscribe` → NameError/500。补回 import + 加源码守护测试。
- **Medium（已修）**：`tasks.py` 死 import `META_TTL`；`batch.py` `status` 变量遮蔽 → `task_status`。
- 复确认：**0 Critical / 0 High**。

### 门禁实证
- pytest `tests/test_regression.py` — **168/168**（+6 新测：`load_task_for` 四态 · clearance/tasks/batch/compliance fail-closed · clearance_stream import 守护）
- `verify_fixes.py` — **293/0**（FIX-033/063 检查随重构同步）· `check_doc_drift` — **15/15** · `check_system` — **131/0/134** · `check_integrity` — 全过

## ⑥ DOCS
- CHANGELOG `[2026-09-24]` Round 029 · AGENTS 资源归属模型补 `load_task_for` · `UNRESOLVED-029 → resolved` · 本 round-029.md

## ⑦ PUSH
- 待执行：`git push LocalAI master`

## ⑧ IMAGE
- 待执行：`docker_build` + `up -d --force-recreate app celery-worker celery-beat` + 健康/抽查（含 `load_task_for` 标记）

## ⑨ RE-CHECK — 工作流 B（视觉专项 COLLECT，未开始）
- 扩展 `tests/visual_regression.py`：保留 10 CP，新增 12 缺页 CP；用 `tests/fixtures/clearance_baseline/*.docx` 经 `/clearance/run` 播种清标任务，使报告类页面有真实内容。
- 在 ⑧ 的新镜像上重拍 → ② VERIFY 判定 H1/H2/H3（巨大空白/表格溢出/压输入框）。
- 顺带修正 `qa-loop.project.md` 健康检查命令（本机 nginx 仅发布 `:80`）。

## 收尾
- `data/qa_loop/last_head` → 本批最后代码/文档 commit；消费 `pending.flag`
