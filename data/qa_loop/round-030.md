# QA-Loop Round 030 (2026-10-04)

基线: last_head=c999e37 | 模式: incremental (c999e37..ea1ea54, 25 commits) | 触发: hook-pending (pending.flag sha=ea1ea54)

引擎适配（本轮自定义）:
- reviewer = `xiaomi-token-plan-cn / mimo-v2.6-pro`（只读工具 `read,grep,find,ls`）
- 工作/视觉 subagent 模型 = 主 agent 同款 `agentrouter / deepseek-v4-flash`
- 视觉 = 项目内 EasyOCR (`app/services/ocr.OCRManager`) + MiMo v2.6-pro 视觉 A/B
- 子 agent 启动：`HOME=/mnt/c/Users/nana- pi --print --no-session -ne -t <tools> --provider <p> --model <m> "<prompt>"`
  （`-ne` 关闭崩溃的 mcp-bridge 扩展；WSL 下必须把 HOME 指向 Windows 配置目录）

---

## ① COLLECT 摘要

- **code-reviewer（mimo-v2.6-pro）**: 15 条 (C:0 H:3 M:5 L:7) —— 见 §② 表
- **视觉侧（OCR + mimo-v2.6-pro A/B）**: 见 §V 视觉专项

## ② VERIFY 初判表（主 agent 对照 HEAD 复核）

| # | 原判 | 严重级 | 初判 | 证据 |
|---|---|---|---|---|
| H1 | ingest_pipeline `rejected_indices: set()` → json.dump TypeError | High | **有效** | `app/services/ingest_pipeline.py:227` set() + `:231` json.dump 无 default；调用点 `:628`（Pipeline B 主路径）；外层 except `:659` 吞掉 → status=error，批量入库中断 |
| H2 | knowledge.py `report_dir` 仅 except 分支赋值 | High | **有效** | `app/routes/knowledge.py:696` 定义于 except；成功路径（`style_engine.generate_report_file` 存在，`style_engine.py:348`）→ `:706` `os.path.join(report_dir,…)` UnboundLocalError → 500 |
| H3 | username 路径穿越（注册仅校验长度） | High | **有效** | `app/routes/auth.py:74-75` 仅 5-18 长度、无字符集；`knowledge.py:688` `filename_prefix` 内嵌 `username_tag`（`:608` 来自 DB username）→ `:698/:706` join 成路径，`..\..` 可越出 `USER_FILES_ORIGINAL_ROOT` |
| M4 | compliance `missing` 一律 403 + docstring 矛盾 | Medium | **有效** | `app/routes/compliance.py:459-470` `status != 'ok'` 即拒；`app/utils/helpers.py:75-77` docstring 称 compliance 结果可超 TTL 存活，行为相反 |
| M5 | compliance 注册 owner 可为空串 → legacy-allow | Medium | **有效** | `compliance.py:258/339/357` `str(session.get('user_id') or '')`；`helpers.py:57-62` `if not owner: return True`（legacy 放行）→ FIX-080 fail-closed 被绕过 |
| M6 | url_guard `.port` ValueError 不在 try 内 | Medium | **有效** | `app/utils/url_guard.py:57` `parts.port` 在 46-48 try 之外；越界端口(>65535) 抛 ValueError；调用方 `chat_files.py:214`/`credit.py:101`/`web_extractor.safe_get` 未捕获 → 500 |
| M7 | 前端 innerHTML 未转义 | Medium | **有效** | `static/js/app.js:4778` `err.error`；`static/js/knowledge-lab.js:244/256` `d.message`/`d.error`；`:119` `d.download_url` 进 href（未校验 scheme） |
| M8 | 报告 download_url 走 GET 但路由仅 POST | Medium | **有效** | `app/routes/knowledge.py:738/832` 返回 `/download_original_file/{uid}/{file}`；`app/routes/chat_files.py:162` 仅 `methods=['POST']` → 前端 `<a href>`(GET, `knowledge-lab.js:119`) 405，且路径参数不匹配 |
| L9 | kb_review task_id 未做 regex 校验 | Low | **有效** | `ingest_pipeline.py:206` `KB_REVIEW_PATH_TEMPLATE.format(task_id=…)` 来自 URL 段，无 `_TASK_ID_RE`（仅 admin 可达） |
| L10 | login_guard 用户名冷却键与 IP 无关 + 内存/Redis horizon 不一致 | Low | **有效** | `app/services/login_guard.py:78-79/88-101/68-70` |
| L11 | auth 登录时序侧信道 + 硬编码特权角色名 | Low | **有效** | `app/routes/auth.py:195-196` 用户不存在即短路；`:138` `("admin","CEO","COO")` |
| L12 | nginx /static `add_header` 取消 server 级安全头 | Low | **有效** | `nginx.conf:71-75` location 内 `add_header Cache-Control` → 按 nginx 继承规则取消 58-60 的三枚安全头 |
| L13 | compose 弱默认口令 + Redis 无 requirepass | Low | **有效** | `docker-compose.yml` `DB_PASSWORD:-localai` |
| L14 | credit `get_json()` 无 silent | Low | **有效** | `app/routes/credit.py:93/332` |
| L15 | tasks delete `missing` 也删且返 success | Low | **有效** | `app/routes/tasks.py:78-82` 仅拦 forbidden |

**死代码核查（要求 4）**：`audit_report`/`audit_wiki_publisher`/`auth_jwt`/`run_audit` 在 `app/`、`celery_app.py`、`scripts/` 零 import（仅 docs/CHANGELOG/测试断言引用）；`audit_engine` 仅被 `clearance_engine.py` 调 `_run_style_analysis`/`_score_*` → 无 NameError 风险。**确认删除干净。**

## ③ CROSS-EXAM

无 ≥High 争议（主 agent 复核结论与 reviewer 原判一致，全部有效），未消耗质问轮。

## ④ CONFIRM 清单（待用户批准后 IMPLEMENT）

### 批准项（建议分 4 批）
- [ ] **A1 | High | `app/services/ingest_pipeline.py:227`** — `set()` → `[]`（下游按 list 存取）
- [ ] **A2 | High | `app/routes/knowledge.py:696/706`** — `report_dir` 提到 try 前统一赋值
- [ ] **A3 | Medium | `app/utils/url_guard.py:57`** — `.port` 包进 try，异常返回 `(False, ...)`
- [ ] **B1 | High | `app/routes/auth.py` + `knowledge.py`** — 注册/改名 username 加 `^[A-Za-z0-9_\u4e00-\u9fa5]{5,18}$`；拼接 filename 做 basename+realpath 前缀校验
- [ ] **B2 | Medium | `app/routes/compliance.py:258/339/357`** — user_id 为空则拒绝注册（或 `task_owner_ok` 空 owner 也 fail-closed）；修 `helpers.py` docstring
- [ ] **C1 | Medium | 前端** — app.js:4778 / knowledge-lab.js:119/244/256 补 `escapeHtml` + download_url scheme 校验
- [ ] **C2 | Medium | `app/routes/knowledge.py:738/832` + `chat_files.py`** — 补 GET 下载路由（含 owner 校验）或前端改 fetch+Blob
- [ ] **C3 | Low | `nginx.conf:71-75`** — /static 内补三枚安全头
- [ ] **C4 | Low | `app/routes/credit.py:93/332`** — `get_json(silent=True) or {}`
- [ ] **C5 | Low | `app/routes/tasks.py:78-82`** — `missing` 同样 404
- [ ] **D1 | Low | `app/services/ingest_pipeline.py` + `knowledge_ingest.py`** — kb_review task_id 加 regex
- [ ] **D2 | Low | `app/services/login_guard.py`** — 冷却键并入 IP；内存 purge 按 ts 分 TTL
- [ ] **D3 | Low | `app/routes/auth.py:195`** — 用户不存在跑 dummy hash；`:138` 角色名配置化
- [ ] **D4 | Low | `docker-compose.yml`** — `DB_PASSWORD` 强制必填

### 驳回/降级项
- 无（15 条全部复核有效；严重级微调：M5 保持 Medium，未升 High——登录态端点空 user_id 实际可达性低）

## ⑤ IMPLEMENT（用户批准：分四批全处理）
- 批 A（High 后端）：`de30505` — H1 `rejected_indices` set()→list · H2 `report_dir` 前置 · M6 url_guard port
- 批 B（安全）：`de30505` — H3 username 白名单 + `safe_tag` · M5 `task_owner_ok` 空 owner fail-closed
- 批 C（前端/功能）：`de30505` — M7 前端 escapeHtml · M8 GET 下载路由 · nginx 静态头 · credit get_json · tasks 404
- 批 D（加固）：`de30505` — D1 kb_review task_id · D2 login_guard {u}:{ip} · D3 auth 等时/`ADMIN_USERNAMES` · D4 compose 强口令
- 复核后补（同 commit）：`de30505` — GET 根放宽到 `DATA_DIR`（覆盖 docx）· `report_dir` 不再取共享 user_styles · app.js 三处 `safeDownloadUrl`
- 终检 M：`1ad5980` — `safeDownloadUrl` 拒绝协议相对 `//`
- 门禁：237/237 regression · verify_fixes 391/0 · node --check OK

## ⑥ DOCS
- CHANGELOG `[2026-10-04]`（14 条 FIX 细分 + LLM 拒答实测 + 视觉 A/B）· `data/fix_registry.yaml` 14 条 FIX · `AGENTS.md`（`ADMIN_USERNAMES` + 空 owner fail-closed）· `.env.example` · 本文件 · `repair_kit/SYSTEM_CHECKLIST.md`（自动重生成）— `de30505`

## ⑦ PUSH
- `git push LocalAI master` → `ea1ea54..de30505` OK（`1ad5980` 随收尾一并推送）；远端 == 本地 HEAD（de30505 后为 1ad5980）

## ⑧ IMAGE
- `docker compose build --build-arg TORCH_INDEX=…/cpu app`（临时把 Dockerfile 的 tuna 镜像换为 **ustc https** + `pypi.org`，构建后 `git checkout` 还原）→ `local-ai:latest` 重建成功 → `up -d --force-recreate app celery-worker celery-beat`
- 健康：`/check_auth` = **200**，app healthy；容器内 grep 抽查 10/10 命中（safe_tag / download_original_file_get / safeDownloadUrl / _kb_review_path / 空 owner / ADMIN_USERNAMES / _DUMMY_PIN_HASH / invalid port / get_json silent）→ **镜像 = HEAD**
- 环境坑：本机 apt 需 ustc **https**（tuna 403；aliyun http trixie 404），已固化进 skill

## ⑨ RE-CHECK
- 终检（mimo-v2.6-pro 只读）：5/5 rework 修复正确；新增 **C:0 H:0 M:1 L:3**（M=协议相对 URL，已修 `1ad5980`；L=DOM `a.href` 赋值 / docx 共享目录 / `_save_structured_data` task_id，均在本次范围外，记 backlog）
- **新增 Critical/High = 0 且 pending 空 → 停跑闸门达成（loop 结束）**

---

## §V 视觉专项：MiMo v2.6-pro 视觉 vs 项目 EasyOCR A/B

样本：
1. `login-18080.png`（当轮实拍，整页，正常尺寸；scratch 已清理）
2. `tests/visual_screenshots/01_clearance_report_full.png`（1740×15315 巨幅整页）

| 维度 | EasyOCR (CPU) | MiMo v2.6-pro 视觉 |
|---|---|---|
| UI 中文文本 | **大量错字**：质问棹式/无迸行十豹迂务/庳己有钓用广名和P[/欢迎使用中联A1 | **基本正确**：质问模式/无进行中的任务/使用已有的用户名和PIN登录/欢迎使用中联AI |
| 巨幅整页(7.67×降采样) | **返回空**（缩放后文字成块） | 明确拒读并请求 1:1 裁切，**不臆造数字**（行为良好） |
| 布局判断 | 无（仅文本） | 能给出结构/空白/对比度判断 |
| 准确性 | 对文档扫描（设计用途）可靠；对 UI 小字不可靠 | 偶有**假阳性**：把 `list_alt`（DOM 实证字体已加载、非 tofu）判成"空心方框缺字形" |
| emoji/符号混用 | 漏读 🌐 | **漏报**（声称"未见 emoji 混用"，实际有 🌐/▶） |

**结论**：
- MiMo v2.6-pro 视觉 **可用于实战的布局/结构审查**，读 UI 文字远胜 EasyOCR；但**有假阳性/漏报**，不能作为唯一权威。
- 精确文字/数字仍以 **OCR + DOM 实证** 为 ground truth；OCR 只对"文档扫描/1:1 裁切"有效，对整页 UI 截图不可用。
- **重要可操作项**：清标报告整页截图 **~98% 为空白** → 高度疑似报告页内容懒加载/未渲染就被截图（或真实巨幅空白）。需在**新镜像上重拍**并在滚动/等待网络空闲后截取，再判定 H1/H2/H3（round-029 遗留的视觉专项）。

**视觉侧建议**：视觉收集器改用 **viewport/元素级裁切截图**（非全页巨图），并与 DOM/OCR 交叉验证后再定级。
