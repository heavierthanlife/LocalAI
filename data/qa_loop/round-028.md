# QA-Loop Round 028 (2026-09-24)

基线: last_head=bf82dbb（陈旧，落后 HEAD） | 模式: incremental (`bf82dbb..1cdb28b`, 24 commits, 62 files) | 触发: hook-pending (`pending.flag` sha=1cdb28b, 2026-09-17T14:44:28)

> 本文件为 **one-round 计划**（④ CONFIRM 已由用户批准）。⑤–⑧ 在 IMPLEMENT/DOCS/PUSH/IMAGE 阶段逐项勾选完成后回填实证。

---

## ① COLLECT 摘要

- **@code-reviewer**：13 条（C:0 H:5 M:5 L:3）
- **@mimo-vision**：11 条（H:3 M:5 L:3）+ 12 个缺截图页面
- 仅读收集；两 subagent 均未修改文件。

## ② VERIFY 初判表

| # | 原判 | 严重级 | 初判 | 理由 |
|---|---|---|---|---|
| C1 | `_task_forbidden` 异常 fail-open | High | **Medium（降级）** | `TaskBus.get` 吞连接失败返回 None → 放行，与 docstring 一致；`task_owner_ok` 对 malformed meta fail-closed；无用户可控输入稳定触发 except |
| C2 | `/compliance/feedback` 缺 `_task_forbidden`（IDOR 写） | High | **有效 High** | `compliance.py:511-548` 从 body 取 `task_id`，越权写反馈训练数据 |
| C3 | `_load_result` 路径遍历 | Medium | **有效 Medium** | `compliance.py:44-55` + `:533,546`，body `task_id` 无白名单 → 任意 `.json` 读（需登录） |
| C4 | `download_project_file` `stored_path` 未约束 | High | **Low（降级）** | `stored_path` 由服务端 `uuid.hex+ext` 生成（`admin.py:1014-1039,941-944`），需 DB 篡改；IDOR 已封 |
| C5 | `knowledge_lab_skills UNIQUE(source,content)` TEXT | High | **Low（降级）** | `ingest_pipeline.py:452` 入参 `text_val[:500]`（≤1500B UTF-8）远低于 btree 2704B 上限；且插入有 try/except 兜底 |
| M6 | `logger.debug` 掩盖 ownership-skip | Medium | 有效 Medium | `compliance.py:436` |
| M7 | 法规 module-level 缓存不重载 | Low | 有效 Low（设计如此） | `compliance_checker.py` |
| M8 | `generate-weekly-report` 去 kwargs | Medium | **误报/非问题** | `cleanup_tasks.py:377` 无参，改动正确 |
| M9 | `cleanup_tasks.py` 双 `try: from celery_app` | Low | 有效 Low | `:531,577` |
| M10 | `bootstrap.ensure_seeded` 窄 except | Low | 有效 Low | `bootstrap.py:53-57` 已有通用兜底 |
| L11 | `x-app-env` 过宽（beat 继承） | Low | 有效 Low | `docker-compose.yml` |
| L12 | `_safeHTML` 降级压平富文本 | Low | 有效 Low（安全优先，设计取舍） | `app.js:367-374` |
| L13 | `LOG_LEVEL` 经 root 同时约束 file handler | Low | 有效 Low（注释与行为不符） | `config.py:60-74` |
| B | **Bonus**：`task_bus._get_redis()` 首次失败永久缓存 False | — | **有效 Medium** | `task_bus.py:31-43`，Redis 短暂不可用 → 进程生命周期内 TaskBus 恒 None |

## ② VERIFY — 视觉侧

25 张截图均为 **2026-09-11**；之后 `static/js/{app,chat,knowledge-lab,review,accounts,bid-audit,compliance,tiptap-editor}.js` + `templates/index.html` 已变更（`1cdb28b`/`3a56992`/`9903225` 等）。
→ **全部视觉项判定 = 待运行验证（截图过期）**，不得基于旧图入 CONFIRM。

- H1 巨大空白（`01`,`10_section_0`）· H2 指标表全宽溢出（`12_section_2`）· H3 表格压输入框（`10_section_2`）
- M1 单位列截断 · M2 顶栏裁切 · M3 AI 评审空态文案 · M4 debug 字段暴露 · M5 原文列截断
- L1–L3
- 缺截图 12 页：清标详情展开/知识图谱/审计日志详情/编辑提示词/模板编辑器/文件预览弹窗/窄屏≤768/知识中心/项目管理/系统管理/回收站/质问模式

## ③ CROSS-EXAM

仅对 VERIFY≠原判且原判 ≥High 的两条抛回 @code-reviewer，均**接受裁决**：

| # | 争议 | 裁决 | 依据 |
|---|---|---|---|
| C1 | High→Medium | **降级 Medium** | Redis 宕机走 `get()`→None 不放行路径；except 仅 Redis 中途断连 / `int(progress)` 非数字；无用户可控稳定触发 |
| C4 | High→Low | **降级 Low** | 写入链 `secure_filename`+`uuid4().hex`+`to_rel_path` 归一化，无 `..` 注入面 |

---

## ④ CONFIRM 清单（用户批准：2026-09-24）

### 批准项（四 fix 合一批，C2 → C1/M6 → C3 → Bonus）
- [x] **C2** | High | `compliance.py` `submit_feedback` | 加 `_task_forbidden(task_id)` 守卫
- [x] **C1+M6** | Medium | `compliance.py` `_task_forbidden` | `except` → `logger.warning` + `return True`（fail-closed）
- [x] **C3** | Medium | `compliance.py` `_load_result/_save_result` | `task_id` 白名单正则 `^[A-Za-z0-9_-]{1,64}$`
- [x] **Bonus** | Medium | `task_bus.py` `_get_redis` | 连接失败改「间隔重试」而非永久缓存 False（C1 fail-closed 的安全垫，需与 C1 同批）

### 驳回/降级/延期项（记入本记录）
- C4 → **降级 Low**（纵深防御，本轮不做，可留 backlog）
- C5 → **降级 Low**（`text_val[:500]` 截断已规避；残余 nit：UNIQUE 未含 category）
- M7/M9/M10/L11/L12/L13 → **Low，本轮不做**
- M8 → **误报/非问题**
- 视觉全部 → **延期至下轮 COLLECT**（截图过期 + 采集脚本需扩展）

---

## ⑤ IMPLEMENT 计划（每 fix 一 commit）

> 顺序约束：**Bonus 先于或同批于 C1 落地**（否则 C1 fail-closed 在 Redis 抖动时会误拒合法访问）。实际执行：C2 → C1/M6 → C3 → Bonus（同批推送，同一次 review/rebuild）。

### commit 1 — C2（FIX-2026-09-24-062）
- 文件：`app/routes/compliance.py`（`submit_feedback`，`orig = _load_result(task_id)` 之前）
- 改动：`if _task_forbidden(task_id): return err("无权访问该任务", "FORBIDDEN", 403)`
- 回归：`tests/test_regression.py::test_compliance_feedback_requires_owner`（他人有效 task 的 meta.user_id ≠ 当前会话 → 403）

### commit 2 — C1+M6（FIX-2026-09-24-063）
- 文件：`app/routes/compliance.py`（`_task_forbidden`，`except` 分支）
- 改动：`except Exception as e:` → `logger.warning(f"_task_forbidden check failed, denying: {e}")` + `return True`
- 回归：`test_compliance_task_forbidden_fail_closed`（monkeypatch `TaskBus.get` 抛异常 → `/compliance/result/<id>` 返回 403）

### commit 3 — C3（FIX-2026-09-24-064）
- 文件：`app/routes/compliance.py`
- 改动：
  - 模块级 `_TASK_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,64}$')` + `_valid_task_id()`
  - `_save_result`：非法 → `raise ValueError("invalid task_id")`
  - `_load_result`：非法 → `return None`
  - （`rules_<uuid>` 合成串仅含字母数字下划线，不被误杀）
- 回归：`test_compliance_load_result_rejects_traversal`（`_load_result('../../etc/passwd')` → None；`_save_result` 抛 ValueError）

### commit 4 — Bonus（FIX-2026-09-24-065）
- 文件：`app/services/task_bus.py`（`_get_redis`）
- 改动：新增 `_redis_last_try` / `_REDIS_RETRY_INTERVAL = 60.0`；失败设 `_redis=False` + 记录时间；`_redis is False` 且未到间隔 → 返回 None（falsy）；到期重试连接。保持 `if not r` 调用约定不变。
- 回归：`test_task_bus_redis_retries_after_interval`（monkeypatch `time.time` + `redis.Redis.from_url` 先抛后成 → 间隔后重新尝试）

## ⑥ DOCS 计划
- `CHANGELOG.md` 顶部新增 `[2026-09-24]` 条目，按 FIX-062..065 细分，附 `regression: 1/1 clearance baseline passed`
- `AGENTS.md`：资源归属模型补「合规 `_task_forbidden` fail-closed + `task_id` 白名单」
- `data/unresolved.yaml`：新增 **UNRESOLVED-028**（视觉回归脚本扩展 + 12 缺页 + 清标数据夹具，下轮 COLLECT 独立立项）；C4 纵深防御可并入 UNRESOLVED-027 或新立
- 本 `round-028.md` 回填 ⑤–⑧ 实证

## ⑦ PUSH 计划
- `git push LocalAI master`；失败则 `git pull --rebase` 或报告，禁止强推。推送后工作树干净方可进 ⑧。

## ⑧ IMAGE 计划（has_docker: true）
- 前置：启动 Docker Desktop（当前引擎未运行）→ `python scripts/docker_build.py` → `docker compose up -d --force-recreate app celery-worker celery-beat`
- 健康：`curl -k -s -o /dev/null -w "%{http_code}" https://127.0.0.1/check_auth` == 200
- 容器抽查（qa-loop.project.md 基准 4 条）＋ 本轮新增标记：`docker exec localai-app sh -c "grep -c '_TASK_ID_RE' /app/app/routes/compliance.py"` ≥1
- 抽查失败 → 回滚上一镜像并报告，不进入下一轮

## ⑨ RE-CHECK / 下轮
- 本批无新增 Critical；视觉 H1/H2/H3 未决 → 下一轮为**视觉专项 COLLECT**：
  - 扩展 `tests/visual_regression.py`（补齐 12 缺页 + 清标数据夹具，保留现有 10 个泛化 CP）
  - 在 ⑧ 新镜像上重新采集，再走 ②VERIFY 判定 H1/H2/H3

## 收尾
- `data/qa_loop/last_head` → `1cdb28b`
- 清空 `data/qa_loop/pending.flag`（已记触发来源=hook-pending）
- 工作树保持干净（除既有 2 个 `.bak`）

## 验收命令（本批）
```powershell
.venv\Scripts\python.exe -m pytest tests/test_regression.py -s   # 全绿 + 4 新测试
python scripts/verify_fixes.py                                    # 全过（新增 4 FIX）
python scripts/check_doc_drift.py                                 # 15/15
python scripts/check_system.py                                    # 清单重生成
python scripts/check_integrity.py                                 # 蓝图↔清单覆盖
# 合规/清标路径 → 1/1 基线（3/3 blocked：UNRESOLVED-017）
@code-reviewer 只读复核 diff（app/routes + app/services 命中门禁）
```

---

## 执行记录（2026-09-24）

### ⑤ IMPLEMENT
- `ab8f1c7` — `fix(security): QA-028 合规归属 fail-closed + task_id 白名单 + TaskBus Redis 重试（FIX-062~065）`
  - `app/routes/compliance.py`：C2 守卫 / C1 fail-closed（+ docstring 澄清）/ C3 白名单
  - `app/services/task_bus.py`：FIX-065 间隔重试
  - `tests/test_regression.py`：4 个新回归测试（含 L1 边界用例）
  - `data/fix_registry.yaml`：FIX-2026-09-24-062..065 · `repair_kit/SYSTEM_CHECKLIST.md`（自动重生成）

### 门禁实证
- pytest `tests/test_regression.py` — **164/164**，EXIT=0（前置：`docker compose up -d redis`，rate-limiter 依赖 :6380）
- `verify_fixes.py` — **287/0** · `check_doc_drift.py` — **15/15** · `check_system.py` — **131/0/134** · `check_integrity.py` — 全过
- `@code-reviewer` — **0 Critical / 0 High**；M1 docstring 已修，L1 测试边界已补，**M2 → UNRESOLVED-029**

### ⑥ DOCS
- `c999e37` — `docs: QA-028 CHANGELOG/AGENTS/unresolved + round-028 记录`
  - CHANGELOG `[2026-09-24]`（FIX-062..065 + regression 标记）· AGENTS 资源归属模型 · `UNRESOLVED-028`（视觉）/`UNRESOLVED-029`（M2）

### ⑦ PUSH
- `git push LocalAI master` → `1cdb28b..c999e37`，OK（工作树干净，仅 2 个既有 `.bak`）

### ⑧ IMAGE
- `python scripts/docker_build.py`：`local-ai:latest` 构建 OK（GPU=False / torch cpu）
- `docker compose up -d --force-recreate app celery-worker celery-beat` → 3 容器 Recreated
- 健康：app 容器内 `/check_auth` = **200** + Docker `healthy`；worker 连接 `redis://redis:6379/0`；app 全蓝图 eager ready
- 抽查（全部命中）：`kb_file_hash`=2 · `credit_rate:`=2 · `_renderMarkdown`=17 · `escapeHtml(data.message)`=1 · **新增** `_TASK_ID_RE`=2 · `Fail closed (FIX-063)`=1 · `_REDIS_RETRY_INTERVAL`=2
- ⚠️ 口径修正：本机 compose 的 nginx **只在宿主机发布 :80**（443 为容器内，宿主机 :443 被其它进程占用）→ `https://127.0.0.1/check_auth` 不可达（000）；健康以 app 容器内 200 + nginx :80 的 301 重定向为准。qa-loop.project.md 的健康检查命令应更新。

### ⑨ RE-CHECK / 下轮
- 本批无新增 Critical/High。视觉 H1/H2/H3 仍待验证 → 下轮为视觉专项 COLLECT（扩展 `tests/visual_regression.py` + 12 缺页 + 数据夹具，见 `UNRESOLVED-028`），在新镜像上重拍后走 ② VERIFY。

### 收尾
- `data/qa_loop/last_head` → `c999e37`（本批最后一个代码/文档 commit）
- 消费 `pending.flag`（触发来源=hook-pending, sha=1cdb28b）

