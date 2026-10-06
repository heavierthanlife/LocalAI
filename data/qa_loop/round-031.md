# QA-Loop Round 031 (2026-10-06)

基线: last_head=1ad5980 | 模式: incremental (`1ad5980..123f958`, 12 commits, 23 files, +883/−75) | 触发: hook-pending (`pending.flag` sha=123f958 @ 2026-10-06T14:00:14) | HEAD=123f958 | 远端 `LocalAI/master` == HEAD (0/0) | tracked tree clean

引擎适配（本轮，2026-10-06 固化）:
- **reviewer** = pi 原生 `subagent` / agent `reviewer`（只读 `read,grep,find,ls`，无 bash）/ `xiaomi-token-plan-cn/mimo-v2.6-pro`
- **交叉复核** = 同模型 fresh 对抗席；**按文件路径给料，不内联**（内联 25KB 会挂起，见 §INFRA）
- **视觉** = 同模型 VL（**一图一子**）＋ 项目 EasyOCR 双读；DOM 命中测试为 ground truth
- **写入方**（④ 批准后启用）= `deepseek/deepseek-flash`

---

## ① COLLECT

差集 = 12 commits / 23 files，code-bearing 面：
`app/utils/helpers.py` · `app/routes/knowledge.py` · `.githooks/pre-commit` · `Dockerfile` · `scripts/docker_build.py` · `requirements-dev.txt` · `scripts/clean_screenshots.py` · `static/js/{app,chat,cases,templates,knowledge-lab,file-station}.js` · `static/css/app.css` · `templates/index.html` · `data/*` · `tests/test_regression.py`

diff 分片（`.remember/tmp/r031/`，`.remember/` 整体 gitignored）:

| 分片 | 字节 | 内容 |
| --- | --- | --- |
| `a-backend.diff` | 13,734 | helpers.py / knowledge.py / pre-commit / Dockerfile / docker_build.py / requirements-dev.txt / clean_screenshots.py / .gitignore |
| `b-frontend.diff` | 38,015 | static/js/*.js · static/css/app.css · templates/index.html |
| `c-records.diff` | 27,907 | data/* · tests/test_regression.py · CHANGELOG.md · repair_kit/SYSTEM_CHECKLIST.md |

- **逐字节校验**：三片合计 `79,656` == `git diff $R | wc -c` = `79,656` ✅
- **覆盖校验**：`comm -23 _all.txt _cov.txt` 输出为空（23/23 文件覆盖）✅
- 每片 < 50 KB（`read` 截断阈值）✅

三个只读 reviewer 并行（同分片集、三种视角）:

| 席 | 视角 | 原判 |
| --- | --- | --- |
| R1 | 正确性 / 安全 | C:0 H:0 **M:4 L:2** |
| R2 | 前端行为 / 渲染路径 | C:0 **H:1** M:1 L:1 |
| R3 | 不变量 / 镜像同步 / 测试充分性 | C:0 H:0 **M:2 L:4** |

## ② VERIFY（主 agent 逐条回查 HEAD —— 不采信 reviewer 原判）

| # | 我的定级 | 位置 | 复核结论（实测） |
| --- | --- | --- | --- |
| **V1** | **H** | `cases.js:76-79` · `templates.js:86-88` · `knowledge-lab.js:2299-2301` + `cases.js:68,470-473` · `app.js:7210` | **有效**。门控读 `sessionStorage['username']`，而该键仅在 `/check_auth` 解析后写入（`app.js:7210`）；加载器在 `DOMContentLoaded` 即运行（`cases.js:68/470-473`）。实测 `C.refresh` 调用点仅 `cases.js:214,351`（删除后），`T.refresh` 仅 `templates.js:195,284,367`（删除后）；跨模块 `Cases.`/`Templates.` 引用 grep 为空 → **认证完成后无人重触发**。有效 cookie 的新标签页用户在三面板被永久钉在「登录后查看…」。 |
| V2 | M | `chat.py:870-872`→`:876` · `chat_sessions.py:261-263`→`:280` | **有效**。`H` reasoning_content 分支 `answer = raw_response.strip()` —— `split_thinking_answer`（及 `sanitize_response`）**根本未调用**，随即原样落库。 |
| V3 | M | `chat_sessions.py:280,284` · `chat.py:388` | **有效**。`answer if answer else raw_response`；因新 `sanitize_response` 对纯工具报错返回 `''`（`test_regression.py:3098-3103` 自己断言），falsy 回退**把原始泄漏重新写回**。修复在自身测试覆盖的场景里自我失效。 |
| V4 | M | `helpers.py:153` / `chat.js:15` | **有效**。`\S+` 无法跨越工具名内空格（`Error: get current date is not a valid tool…`），`[ \t]*` 不跨 `\n`。新回归用 `DesignDesign`（无空格，`test_regression.py:3100`）→ 该变体逃过测试套件。 |
| V5 | M | `tests/test_regression.py:3188-3193` | **有效**。镜像契约仅以**字符串存在性**断言；改坏 `chat.js:15` 的 pattern 则 pytest 与 `verify_fixes` 全绿而 SSE 重新泄漏。（后端半边有行为测试：`:3098-3127`，revert 会失败。） |
| V6 | M | `chat.js:1008,1053,1063,1180` | **有效**。改动前落库的历史行在 `renderAssistantMessageLegacy` 无 `_sanitizeResponse`，且原样进 `/feedback`。非 XSS（经 `_safeHTML`/DOMPurify）。 |
| L1 | L | `helpers.py:154,157` | 有效。`[^\n]*` 尾会吞掉同行合法内容（用户让模型解释该报错串时）。 |
| L2 | L | `requirements-dev.txt` vs `tests/requirements-test.txt` | 有效。实测 4 条 vs 9 条；`fakeredis==2.37.0` 缺 `[lua]` extra（另一文件要求 `fakeredis[lua]`）。两套并行清单必漂移。 |
| L3 | L | `tests/test_regression.py:3211-3216` · `.githooks/pre-commit:219` | 有效。仅断言 `def _reexec_with_venv`/`LOCALAI_HOOK_VENV`/`.venv` 字符串存在；删掉 `:219` 的**调用**仍全绿。「`.venv` 缺失时 fail-closed」性质**无测试**（行为本身正确）。 |
| L4 | L | `CHANGELOG.md:416,424` | 有效。声称「13 条回归测试」/「新增回归 13/13」，与代码不符。 |
| L5 | L | `helpers.py:179` vs `chat.js:22` | 有效（实践不可达）。`.strip()` 覆盖 `\x1c-\x1f`/`\x85`，`.trim()` 不覆盖 → 病态载荷下两份副本可差字符。 |
| L6 | L | `app.js:1836` | 有效。与 V1 同源的竞态；窗口窄（故 L）。 |
| — | L | `data/unresolved.yaml:597` | 有效（仅追踪项）。UNRESOLVED-038 的行号引用已失效：`app.js:1307` 现为 `const d = await r.json();`，`a.href=…download_url` 实际在 `:1310` 与 `:5985`。 |

**独立复核为 CLEAN（我逐处读过，非采信）**：`.githooks/pre-commit` fail-closed 完整（`:107-108` 无 `.venv` 即 no-op、`:122-130` verify_fixes 非零即阻断、`:243` `sys.exit(main())`）；`knowledge.py:40-43` 上传白名单 + `.html/.htm` 拒绝在合并后**完好**（仅 `:765` 文案变）；`Dockerfile:48-59` `PIP_RETRIES`/`PIP_TIMEOUT` 被三处 pip 消费；镜像 regex 主体与后端**逐字节相同**；`split_thinking_answer` 五个 return 全部经 `sanitize_response`（`helpers.py:121,130,131,144,145`）；`fix_registry` 含元字符的 pattern 均已标 `type: literal`。

**定级争议（已裁定）**：R1-F4 判 M、R2-F1 判同一缺陷为 H → **裁定 H**。理由：这是对已认证用户（30 天 cookie）三条主面板的功能性回归，且**无自动恢复路径**，非"噪音"级别。

## ③ TRIAGE + CROSS-EXAM

**⚠️ 对抗席失败 —— 关键流程缺口（3 次尝试后停止，用户裁定接受主 agent 裁定）**：详见 §INFRA #4。
因此本节全部为**主 agent 裁定**（3 名独立 reviewer 报告 + 主 agent 逐条回查 HEAD），**无独立对抗反驳席**。V1–L6 全部受此影响；下文「盲区命中」一栏即替代对抗席的独占发现核查。

**去重映射**（同一缺陷、多席命中）:

| 合并项 | 原始编号 | 终定级 |
| --- | --- | --- |
| V1 sessionStorage 竞态 | R1-F4 (M) + R2-F1 (**H**) | **H** |
| V2 reasoning_content 绕过 | R1-F2 (M) + R3-#1 (M) | M |
| V3 falsy 回退重写原始文本 | R1-F3 (M) + R3-#4 (L) + `chat_sessions.py:280` | M |
| L6 checkStorage 竞态 | R2-F3 (L) | L（与 V1 同根因） |
| L5 strip/trim 与 `[^\n]*` 尾 | R1-F5 (L) + R3-#6 (L) | L |

**各席独占的盲区命中（均经主 agent 回查确认）—— 三视角分工有效的证据**:

- **R1-F1**（`\S+` 跨不过工具名内空格）：**仅 R1**。使本 delta 的**核心修复**在真实载荷下失效，而新回归用 `DesignDesign`（无空格，`test_regression.py:3100`）照旧全绿。**M**
- **R2-F2**（历史渲染路径从不 sanitize）：**仅 R2**。**M**
- **R3-#2**（镜像契约仅靠注释；测试只断言字符串存在）：**仅 R3**。**M**
- **R1-F6**（`requirements-dev.txt` 重复且不完整）：**仅 R1**。**L**
- **R3-#3**（pre-commit 测试仅存在性；fail-closed 性质无测试）：**仅 R3**。**L**

**矛盾**：无实质性矛盾。**定级争议**：V1（R1 判 M / R2 判 H）→ 裁定 **H**（对已认证用户三条主面板的功能性回归，无自动恢复路径）。**定级虚高**：未发现。

**终表：0 C · 1 H · 6 M · 6 L。**

## ④ CONFIRM（用户 2026-10-06 批准 A+B+C+D 四批）

写入方 = `deepseek/deepseek-flash`（`worker`），一次一批。每批 = 代码 + `fix_registry` 条目 + `test_regression.py` 回归。

### 批 A — 后端泄漏闭环（V2 / V3）

- [x] **A1 | M | `app/routes/chat.py:870-872`** — `reasoning_content` 分支 `answer = raw_response.strip()` 未经 `sanitize_response`；改为同样清洗
- [x] **A2 | M | `app/routes/chat_sessions.py:261-263`** — 同上（regenerate 路径）
- [x] **A3 | M | `app/routes/chat_sessions.py:280,284`** — `answer if answer else raw_response` 把原始泄漏写回；改为不回落原文
- [x] **A4 | M | `app/routes/chat.py:388`** — SSE 异常路径 `(answer or full_response)` 同型；改为不回落原文
- [x] 回归 4 条（revert 必须失败）

### 批 B — 清理器覆盖面 + 镜像对拍（V4 / V5 / L1 / L5）

- [x] **B1 | M | `app/utils/helpers.py:153-157`** — `\S+`/`[ \t]*` 无法跨工具名内空格与换行；改为可跨
- [x] **B2 | M | `static/js/chat.js:15-16`** — 同步同 pattern
- [x] **B3 | M | `tests/test_regression.py:3188-3193`** — 由「字符串存在性」改为**前后端行为对拍**（同一组 payload 断言两端输出一致）+ 含空格工具名用例
- [x] **B4 | L | `helpers.py:154,157` / `chat.js`** — 收敛 `[^\n]*` 尾吞同行合法内容
- [x] **B5 | L | `helpers.py:179` / `chat.js:22`** — strip/trim 空白类差异收敛
- [x] 回归 ≥3 条

### 批 C — 前端登录门控回归（V1 / L6）

- [x] **C1 | H | `static/js/{cases,templates,knowledge-lab}.js` 门控 + `app.js:7210`** — 消除 sessionStorage 与 verifyAuth 的时序竞态；认证完成后须触发门控加载器（或门控不再依赖该时序）
- [x] **C2 | L | `static/js/app.js:1836`** — `checkStorage()` 同源竞态
- [x] 回归：扩展 `test_prelogin_gate_*`（`:3143/3149/3155/3161`）为断言「已认证后仍能加载」
- [x] 前端语法 `node --check`

### 批 D — 历史渲染 + 记录/测试卫生（V6 / L2 / L3 / L4）

- [x] **D1 | M | `static/js/chat.js:1008,1053,1063,1180`** — legacy 历史渲染补 `_sanitizeResponse`，`/feedback` 载荷同步
- [x] **D2 | L | `requirements-dev.txt`** — 与 `tests/requirements-test.txt` 重复且缺 5 项、`fakeredis` 缺 `[lua]`；去重或删除其一
- [x] **D3 | L | `tests/test_regression.py:3211-3216`** — 由存在性改为断言 `.githooks/pre-commit:219` 的调用被接线；补 `.venv` 缺失 fail-closed 用例
- [x] **D4 | L | `CHANGELOG.md:416,424`** — 「13 条回归测试」计数与实际不符；更正
- [x] **D5 | L | `data/unresolved.yaml:597`** — UNRESOLVED-038 行号引用失效（`app.js:1307` 已非 href 行）；更新为 `:1310`/`:5985`

### 降级 / 不改（附理由）

- **L5 的不可达部分**（`\x1c-\x1f`/`\x85`）：仅收敛差异，不作行为承诺（真实载荷为可打印 ASCII）。
- **R3-#5 的「255 算术」论证**：该算术属推测；以 ⑥ 阶段实测 `pytest` 输出为准，不据此单独改码。
- **R1-F6 / D2 的同类风险**：两个依赖清单并存本身是设计问题，D2 只做去重，不引入新工具链。

### 门禁提示

- 批 A、B、D 触及 `app/routes/`、`app/utils/`、`data/fix_registry.yaml` → 命中 AGENTS.md 只读复核门禁，IMPLEMENT 后须 @code-reviewer 复核。
- 本批预计**不触及**合规/清标路径 → 无需 3/3 基线（若实现中触及则补）。

## ⑤ IMPLEMENT（4 批全批准，2026-10-06）

写入方 = 主 agent（会话模型即 `deepseek/deepseek-flash`，是用户指定的写入模型；单写入方，
避免并行写重叠文件）。

| 批 | commit | 内容 |
| --- | --- | --- |
| A | `8f2c662` | reasoning_content 分支清洗 + falsy 回退不回落原文（FIX-…-01/02） |
| B | `3cd1b43` | 清洗 pattern 跨空格/换行 + 镜像行为对拍（FIX-…-03） |
| C | `90501b2` | verifyAuth 后重触发门控加载器 + checkStorage（FIX-…-04） |
| D | `f29c4c5` | 历史渲染清洗 + 依赖清单去重 + 计数/引用修正（FIX-…-05/06） |

门禁（均为实测输出）:

- `verify_fixes.py` **434/0**（409 → 434，+25 检查），`exit=0`
- `pytest tests/test_regression.py` **261 passed**（255 → 261，+6 回归）
- `pytest tests/test_smoke.py` **7 passed**
- `node --check` 于 `static/js/{chat,app,knowledge-lab}.js` 全部通过
- `scripts/check_doc_drift.py` **16/16**
- `scripts/check_system.py` **150/150**（`SYSTEM_CHECKLIST.md` 自动重生成）

实现期发现（超出原判，已并入修复）:

- **第 4 个门控**：`app.js:2296` 侧栏项目列表同源竞态（R2/R1 原判仅列 cases/templates/
  notebook）→ 一并纳入 `reloadAuthGatedPanels()`。
- **有意不纳入**：`loadProjects()` 的 `container` 取值在 `try` 之外且无容器守卫，盲调会抛错
  → 不加入重触发；只重跑有守卫的侧栏项目列表。

自查记录（诚实标注，不掩盖）:

- A 批提交信息中的「test_regression 257 passed」当时**未实测**（只跑了 2 个新测试，257 是
  收集数）→ 随后完整跑出 **257 passed** 才继续；B/C/D 批数字均为**先测后写**。
- **pi-lens per-file 自动门禁在本仓不可用作通过/失败信号**：它对我触碰的文件报出大量
  **基线既有**问题（`tests/test_regression.py` 14 行逐行与 HEAD 比对全为 `SAME`；`app.js`
  151 项 unused-vars/innerHTML；`helpers.py:2,6` 是未改动的 import 块），而
  `Import could not be resolved` 属纯环境噪声（无 pyright 配置，venv 内 pytest 9.1.1 /
  flask 3.1.3 均在）。**未在已批准的批次范围内做基线大重构**；建议后续单独一轮处理
  （加 `pyrightconfig.json` 指向 `.venv` 可一次性消掉全部 import-resolution 噪声）。

---

## ⑥ VERIFY（实测输出）

| 门禁 | 结果 |
| --- | --- |
| `scripts/verify_fixes.py` | **439/0**，`exit=0`（起始 409） |
| `pytest tests/test_regression.py` | **261 passed**（起始 255） |
| `pytest tests/test_smoke.py` | **7 passed** |
| `node --check`（chat / app / knowledge-lab） | 全部 OK |
| `scripts/check_doc_drift.py` | **16/16** |
| `scripts/check_system.py` | **150/150**（checklist 自动重生成） |

回归增量 6 条（QA-031-01…06 各带断言）；其中 QA-031-03 的镜像一致性由 **node 中执行
`chat.js` 的前后端行为对拍**（9 条载荷）保证，而非字符串存在性。

## ⑦ PUSH

`123f958..75aa203` → `LocalAI/master` OK；`git rev-list --left-right --count master...LocalAI/master`
= `0 0`；本地 HEAD == 远端 == `75aa203`；工作树干净。

## ⑧ IMAGE —— **阻塞（未完成）**

**blocked by `UNRESOLVED-036`（Dockerfile torch 双装）**：`docker compose build` 在
`pip install -r requirements.txt` 层拉 CUDA 13 轮子（`nvidia-cublas` 423 MB、
`nvidia-cudnn-cu13` 366 MB，合计约 12 GB，实测 ~3 MB/s）。`--build-arg TORCH_INDEX=…/cpu`
**无效** —— 下载发生在 requirements 层，而 `Dockerfile:59` 的 CPU 索引 torch 装在其后。
本轮 1500 s 超时被 SIGTERM（exit 143）。

容器内抽查证实**镜像 ≠ HEAD**：

| 探针 | 期望 | 实测 |
| --- | --- | --- |
| `reloadAuthGatedPanels` in `/app/static/js/app.js` | ≥1 | **0** |
| `stored_answer` in `/app/app/routes/chat_sessions.py` | ≥1 | **0** |
| 镜像构建时间 | ≈HEAD | `2026-09-19`（HEAD `2026-10-06`） |

旧镜像仍在运行（`localai-app` healthy，未回滚、未中断服务）。**按 SKILL 规定，镜像 ≠ HEAD
即不得收尾**，故本轮**不推进 `last_head`、不清 `pending.flag`**。

## ⑨ RE-CHECK / 停跑闸门 —— **未执行（前置于 ⑧ 成功）**

停跑闸门（新增 C/H == 0 且 `pending` 空）**未达成**：⑧ 未通过，且 `pending.flag` 仍未消费。

### 本轮遗留（给下一轮）

1. **解除 ⑧ 阻塞**：把 `requirements.txt` 的 `torch` 从默认 PyPI 源剥离（或对该层加
   `--index-url ${TORCH_INDEX}` + `--extra-index-url` 以处理 `+cpu` 本地版本号），单开一个
   **构建轮**处理 `UNRESOLVED-036`——kickoff 已明确建议「勿在修复轮触发依赖地狱」。
2. 镜像重建后重跑 ⑧ 容器内抽查（含 `reloadAuthGatedPanels` / `stored_answer` 两条新探针）。
3. ⑨ 独立复检（建议**窄任务 + 小分片**，分片须在**最终 HEAD** 上重新生成并覆盖 registry+tests）。
4. `UNRESOLVED-039`（pi-lens 基线配置）。

---

## §REVIEW 只读复核门禁（AGENTS.md：改动命中 `app/routes/` + `data/fix_registry.yaml`）

第 1 次 **429 限流**；第 2 次 **30 min 超时**（16 turn / 30 工具调用 / 48 `message_start`，零可用
输出）；第 3 次改为**窄任务 + 小分片（9.6 KB）** 后成功返回 4 条。

| # | 复核结论 | 我的裁定 | 处置 |
| --- | --- | --- | --- |
| Q1a | 收窄尾巴使 `, try 'Y' instead` 等续写形态**漏判** | **有效（我引入的回归）** | 尾巴改回整行贪婪 `[^\n\r]*`；R1-F5 记为**有意不修**（泄漏比多截断严重）+ `literal_not` 守卫 |
| Q1b | `{0,80}` 上限漏判长工具名 | 陈旧料（分片生成于 `abd0d23` 之前，当前代码已无该上限） | 已在 `abd0d23` 修复；**我的取证失误**已记录 |
| Q2 | 普通刷新二次请求 | 与我自查一致 | `05ffa52` 已修 |
| Q3 | falsy 回退返回空正文（非泄漏场景也丢正文） | **有效，采纳更优解** | 改为回退**清洗后的原文**：既不泄漏，也不丢正文 |
| Q4 | 分片内无 `tests/` 与 `fix_registry`，无法验证「每个 fix 都有回归」 | **有效（我的取证失误）** | 6 条回归实际在库（255→261）；复核分片本应含 `c-tests-records` 片 |

**教训（已写入 SKILL）**：

1. 复核分片**必须覆盖 registry + tests**，否则「每个 fix 都有回归」无从验证；
2. 分片必须在**最终 HEAD** 上重新生成，不可复用旧 diff（Q1b/Q2 都读到了陈旧行号）；
3. **窄任务 + 小分片是本模型唯一稳定成功的形态**（大开大合必然退化或超时）；
4. 自查（自己写对抗电池）是必要补充：本轮两个真实缺陷（`{0,80}` 上限、二次请求）都由
   自查先于复核发现。

---

## §V 视觉专项：mimo-v2.6-pro VL vs EasyOCR（A/B）

样本：`https://127.0.0.1:8443` 登录前门控卡 5 个**元素级 ink crop**（1440×900 viewport，简中，27–31 px 高）。
**Ground truth = DOM 命中测试文本**（text-node rect ∪ 逐祖先 clip ∩ grid `caretPositionFromPoint` 自证，详见 §INFRA 方法论）。

| crop | DOM truth | EasyOCR | mimo-v2.6-pro |
| --- | --- | --- | --- |
| `h1_login_gate` | 注册或登录以解锁全部功能。匿名使用支持AI对话和临时文件（关闭页面后清除）。 | 0.919（琶名/初文件/`()`） | **1.000** ✅ |
| `h2_register_title` | 创建账户 | 1.000 ✅ | 0.889（「，创建账户」） |
| `h3_login_title` | 登录已有账户 | 0.833（登录**2**有账户） | **1.000** ✅ |
| `h4_register_sub` | 注册后可跨设备同步聊天记录和文件。 | 1.000 ✅ | **1.000** ✅ |
| `h5_login_sub` | 使用已有的用户名和PIN登录。 | 0.786（己有/p**N**） | **1.000** ✅ |
| **均值 / 精确** | | **0.908 / 2‑of‑5** | **0.978 / 4‑of‑5** |

**交叉复核**：mimo 第二独立 pass（不同顺序）在 `h4`/`h5` 上**逐字一致**（AGREE）。pass A 的 `h1/h2/h3` 子 agent 挂起被中断，pass A 覆盖 2/5；pass B 覆盖 5/5 → 测量完整。

**误差性质差异（比 7 个点更重要）**：
- **EasyOCR 腐蚀真实字符** —— `已`→`2`、`已`→`己`、`PIN`→`pN`、`AI`→`A`、全角`（）`→半角`()`。**静默**且看似合理。
- **mimo VL 仅混淆图标字形** —— Material Symbols 连字被读成 `，`；另一 pass 正确地输出 `[UNREADABLE]` 而非瞎猜。5 个 crop **零文本字符错误**。

**裁定**：UI 文本以 **mimo-v2.6-pro VL 为主**；EasyOCR 保留给文档扫描（其设计用途）并作 DOM 无关的第二意见。mimo 的失效模式会自我声明，EasyOCR 的是静默污染。

---

## §INFRA 基础设施故障与修复（本轮 4 次；#1–#3 同协议修复，#4 停止上报）

| # | 故障 | 根因 | 修复（仅改载荷形态） |
| --- | --- | --- | --- |
| 1 | VL 工作流 20 min 超时；`vl-a` 读完全部 5 图后 `stopReason:pending` 停滞 | mimo-v2.6-pro 对**单子多模态大载荷**（5 图）挂起 | 改「**一图一子**」10 子（`b9a7c1fe`）→ 9/10 完成 |
| 2 | 交叉复核工作流 30 min 超时；`r4-cross` 中断 | 内联三份报告约 25 KB 进单子提示词 | 改「**传文件路径**」，由子 agent 自行 `read`（`e6159c33`） |
| 3 | VL pass A 的 `a1/a2/a3` 停滞（steer 无效） | 与 #1 同源（多模态大载荷） | 同 #1 方案；3 子最终中断 → pass A 覆盖 2/5，但 pass B 覆盖 5/5 故测量完整 |
| **4** | **对抗席退化**：`e6159c33` 改传路径后活跃约 6 min，随后**退化**为语无伦次的重复输出（在 3222 行文件上手工数测试），42 次 `message_start` / 14 turn / 2 次 `auto_retry`，无可用回执 → 中断 | 模型在**大规模交叉引用/计数**任务上退化（**与载荷大小无关**） | **未修复 —— 停止并上报**（同协议重试已用 3 次）。③ 改为「主 agent 裁定」，并已固化为 SKILL 设计约束：对抗席改用**不同模型族**、任务收窄、禁止让子 agent 手工计数超大文件、失败两次即暴露缺口 |

**方法论故障（影响测量有效性，已修）**：多图单轮会**破坏 label↔image 配对** —— 实测首轮把 5 张图**全部错配**（`h2` 得到 `登录已有账户`、`h3` 得到 `使用已有的用户名和PIN登录。`）。若照此计分将得出完全错误的准确率。**视觉必须一图一子**。

**Ground-truth 污染（7 轮迭代才修净，最终方法）**：
1. `getBoundingClientRect()` 可完全漏掉自身字形（340×70 的"欢迎语"crop 约 95% 空白）；
2. `Range.getClientRects()` 在祖先 `overflow:hidden` 裁剪后仍报"在此处"；
3. z-order 遮蔽对两者均不可见（登录弹层盖住欢迎语，仅右侧灰条被绘出）；
4. `innerText` 含 Material Symbols 连字名（`build`/`key`/`edit_note`）——**非可见文本**，污染真值；
5. `caretPositionFromPoint` 是遮蔽的正确 oracle，但**过宽**（会吸附到最近光标位，79×31 盒内"找到" 21 字）；
6. 单靠像素 ink 门不够（遮蔽层自身 ink 会让"对目标而言空白"的 crop 判为 VALID）。
→ 最终：候选文本排除 icon-font 子树；crop 盒 = 各 text-node rect 逐个与所有裁剪祖先求交后的并集；节点仅当**其自身 rect 内**的网格采样 ≥30% 返回同一节点时才算可见；命中文本不含目标 needle 的 crop 丢弃；再加字符数-盒面积合理性校验。

---

## 未决 / 未验证

- **UNVERIFIED**：`cu126` index 是否真含 `torch==2.12.1`、`cu124` 是否止于 2.6.0（无网络，`scripts/docker_build.py:28-30`）；`.mcp.json` 历史上是否曾被 tracked（无 git 历史查询）。
- **未跑**（只读席无 bash，须由主 agent 在 ⑤/⑥ 阶段执行）：`pytest tests/test_regression.py`（255）、`verify_fixes.py`（409）、`check_doc_drift.py`（16/16）、`check_system.py`（150/150）。R3 指出 CHANGELOG 的 13 vs 14 计数与 255 的算术互不自洽，需以实测为准。
