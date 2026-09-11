# QA-Loop Round 022 (2026-09-11)
基线: last_head=13f5788 | 模式: full | 触发: manual（用户 "go"；pending.flag 来自自查 commit 7bb570d）

## ① COLLECT 摘要
code-reviewer（4 路并行，只读）：
- routes（17 蓝图）：C:1 H:3 M:6 L:2
- services（清标/LLM/核心）：C:2 H:3 M:5 L:2
- frontend（app.js/chat.js/knowledge-lab.js/bid-audit.js/index.html）：C:0 H:4 M:6 L:4
- infra（database/__init__/celery/scripts/tests/compose/nginx/hooks）：C:2 H:6 M:6 L:1
mimo-vision：待截图（B 完成后）。

## ② VERIFY 初判表（主 agent 对照 HEAD 代码逐条裁决）
| # | 原判 | 严重级 | 初判 | 理由（file:line 实证） |
|---|---|---|---|---|
| R-C1 | auth.py 账户删除重复 deposit | High | **有效** | `auth.py` 选择性 deposit(keep_map) 后，又无条件 `SELECT ... credit_check_reports/batch_comparison_results WHERE user_id` 全量再插一遍 → 保留项重复、非保留项也被 deposit（keep_map 失效）。 |
| R-H1 | credit.py 任务端点无归属校验(IDOR) | High | **有效** | `credit.py:258-322` 仅校验 `consent_value==1`；registry 仅按 task_id 键（`get_task/patch_task`），无 user 比对；`download_credit_report` docstring 自述"任意注册用户可下载"。 |
| R-H2 | compliance.delete_law 路径穿越 | High | **降级 Medium** | `os.path.join(LAWS_DIR, f"{law_id}.json")` 未校验；但 Flask 默认 `<law_id>` 段不含 `/`，穿越需编码分隔符，可行性低 → 仍建议白名单校验。 |
| R-H3 | admin_regeneration 验证码用 random.randint | High | **有效 Medium** | `random` 非 CSPRNG；`auth.py:290` 已用 secrets，属不一致。 |
| R-M1 | tasks.py SSE `Access-Control-Allow-Origin:*` | Medium | 有效 | 泄露任务进度/文件名/bidder 名。 |
| R-M2 | graph.py float() 未防护 500 | Medium | 有效 | `float(request.args.get('threshold',0.5))` 无 try。 |
| S-C1 | compliance_checker.check() 不接受 region_code → TypeError | High | **有效** | `compliance_checker.py:279-286` check() 无 `region_code` 参数；`:493` 却传 `region_code=region_code` → `compliance_check_task` 每次必崩。FIX-033 只补了 task 签名/调用，漏了方法签名，且 verify_fixes 的 literal 检查太弱未拦住。 |
| S-C2 | clearance_engine 聊天气泡 INSERT 与主结果同事务 | High | **有效** | `clearance_engine.py:622-649` 同 `conn` 内 INSERT chat_messages，单条失败使事务 aborted → 649 commit 失败 → 664 外层 except 丢全部结果（含 batch_comparison_results）。 |
| S-H1 | file_store 去重 TOCTOU | High | **有效 Medium-High** | `save_stream` SELECT 与 INSERT 分属不同连接，并发可重复插入/未处理 IntegrityError。 |
| S-H3 | save_* 独立路径吞异常 | High | **降级 Low** | 属既有设计（conn=None 仅 log）；调用方拿 saved=0 无感。可加返回值/异常，低优先。 |
| F-H1 | escapeHtml 未转义引号 → 属性注入 XSS | High | **有效** | `app.js:354-362` 仅转 `&<>`；大量 `'<x a="'+escapeHtml(v)+'"'` 属性上下文。 |
| F-H2 | owner/username 未转义进 innerHTML | High | **有效** | `app.js:2432/3127/3145/3150/3172`（d.skill_a.owner、u.owner、s.username 等）原始拼接。 |
| F-H3 | 内联 onclick 拼接项目名可绕过引号转义 | High | **有效 Medium-High** | `app.js:2404` escapeHtml 后再 `.replace(/'/g,"\\'")`，反斜杠可绕过；`p.status` 完全未转义。 |
| F-H4 | chat.js clearInterval 清 setTimeout | High | **有效 Medium** | `chat.js:854-858` `_pollTimer=setTimeout` 却 `clearInterval`；stop 后 in-flight 会再 schedule。 |
| I-C1 | ADMIN_PIN 默认值不一致(888888 vs 123456) | Critical | **有效 Medium** | `database.py:153` 默认 '888888'，`__init__.py:125` 默认 '123456' → 未设 ADMIN_PIN 时 seed 与 auth 哈希不一致、admin 无法登录。（本机 .env 已设 200486，无即时影响） |
| I-C2 | compose FLASK_SECRET_KEY 默认已知值 | Critical | **有效 High** | `docker-compose.yml:34` `${FLASK_SECRET_KEY:-change-me-in-production}`；`_validate_env` 未拒绝该常量（需确认）。.env 已覆盖，故非即时泄露。 |
| I-H3 | database.py 每次启动无条件 DROP typo 表 | High | **降级 Medium** | `database.py:1263` `DROP TABLE IF EXISTS typo_detection_results` 在建表流程内每次 boot 执行；表已废弃，但幂等 DROP 属残留清理，宜移入一次性 migration。 |
| I-H4 | compose 暴露 5433/6380 到宿主 | High | 有效 Medium-High | 无绑定 127.0.0.1、Redis 无 requirepass。 |
| I-H5 | 无 SESSION_COOKIE_SECURE/HTTPONLY/SAMESITE | High | **有效** | `app/__init__.py` 仅设 SESSION_USE_SIGNER。 |
| I-H6 | Docker 未设 ENABLE_SCHEDULER=false | High | **有效** | `app/__init__.py:184` 默认 true；WORKERS=4 → 4×APScheduler；AGENTS.md 已警示。compose/.env 均无该变量。 |
| I-H7 | nginx 12G 缓冲 vs Flask 50MB | High | **误报** | `nginx.conf:88/103` `proxy_buffering off`；12G 为 /stream_upload 大文件设计，Flask 按路由抬限。 |
| I-H8 | database.py logger 被 config.logger 覆盖 | High | **有效 Low** | `database.py:8` 与 `:16` 双 logger。 |

## ③ CROSS-EXAM
未触发（VERIFY 初判均为确认或降级，无 ≥High 的"初判≠原判且主 agent 不确定"争议）。

## ④ CONFIRM 清单（待用户批准后 IMPLEMENT）
### 拟批 1（High，低风险高收益）
- [ ] F1 | High | `app/services/compliance_checker.py:279` | `check()` 增加 `region_code: str=None` 形参（或去掉调用处传参）；补 verify_fixes 反射检查 + 回归。
- [ ] F2 | High | `app/routes/auth.py:562-572` | 删除两个无条件全量 deposit 块（保留 keep_map 选择性 deposit）。
- [ ] F3 | High | `app/routes/credit.py:258-322` | credit 任务 registry 写入 user_id，读/改端点加归属校验（复用 task_owner_ok 模式）。
- [ ] F4 | High | `static/js/app.js:354-362` | `escapeHtml` 补 `"`→`&quot;`、`'`→`&#39;`（一次修复覆盖属性上下文 80+ 处）。
- [ ] F5 | High | `app/__init__.py` | 增 `SESSION_COOKIE_SECURE/HTTPONLY/SAMESITE`。
- [ ] F6 | High | `docker-compose.yml` | app 服务增 `ENABLE_SCHEDULER=false`（避免 4×APScheduler）。
- [ ] F7 | High | `app/services/clearance_engine.py:635-649` | 聊天气泡持久化移出主事务（commit 后单独写，或 SAVEPOINT），保证结果不丢。
- [ ] F8 | High | `app/routes/tasks.py:142` | 去掉 SSE 的 `Access-Control-Allow-Origin: *`。
### 拟批 2（Medium）
- [ ] M1 | Medium | `static/js/app.js`（2432/3127/3145/3150/3172） | owner/username 等一律 escapeHtml。
- [ ] M2 | Medium | `app.js:2404` | 内联 onclick 改 data 属性 + 事件委托（项目名/status 转义）。
- [ ] M3 | Medium | `chat.js:854-858` | clearTimeout + stopped 标志。
- [ ] M4 | Medium | `app/database.py:153` | ADMIN_PIN 默认统一为 '123456'（与 __init__ 一致）或强制单点。
- [ ] M5 | Medium | `docker-compose.yml:34/35/130/159` | 去掉/校验已知默认 secret（或 `_validate_env` 拒绝 change-me*）。
- [ ] M6 | Medium | `docker-compose.yml:84/102` | 端口绑 `127.0.0.1:`（或去映射）。
- [ ] M7 | Medium | `app/database.py:1263` | 把 DROP typo 表移入一次性 migration。
- [ ] M8 | Medium | `app/routes/compliance.py:delete_law` | law_id 白名单校验。
- [ ] M9 | Medium | `app/routes/admin_regeneration.py:875` | 用 secrets 生成删除验证码。
- [ ] M10 | Medium | `app/routes/graph.py:33` | `request.args.get(..., type=float)`。
- [ ] M11 | Medium | `app/services/file_store.py:100-129` | 去重改单连接 ON CONFLICT。
### 驳回/误报
- [ ] REJ1 | I-H7 | 误报：proxy_buffering off。
- [ ] REJ2 | R-H2/S-H3 降级 | 见上。

## ⑤ IMPLEMENT
用户批准 Batch 1（8 High）+ Batch 2（11 Medium）。单次批量提交（pre-commit 的 verify_fixes 要求 registry 与代码同 commit）：
- 安全/正确性：FIX-037 compliance check() region_code；FIX-038 credit 任务归属；FIX-042 auth 重复 deposit / ADMIN_PIN 默认 / SSE CORS；FIX-039 前端 XSS（escapeHtml 引号 + owner 转义 + 项目项委托）。
- 稳定性/基础设施：FIX-040 清标聊天气泡事务解耦；FIX-041 Cookie 标志 + ENABLE_SCHEDULER=false + secret 必填 + 端口收敛；FIX-043 上传去重 advisory lock + delete_law 校验 + graph float + secrets 验证码 + DROP typo 迁移化。
- 验证：回归 121/121 · route_preservation 3/3 · smoke 7/7 · verify_fixes 185/0 · doc_drift 14/14 · check_integrity PASS · py_compile/node --check OK。

## ⑥ DOCS
- CHANGELOG 顶部新增「QA 全面轮次 022」（FIX-037~043）。
- fix_registry FIX-2026-09-11-037 ~ 043；`tests/test_regression.py` 增 11 项回归。
- `migrations/002_drop_typo_detection_results.sql`(+.rollback)，database.py 移除每次启动 DROP。
- round-022.md 本文件。

## ⑦ PUSH
（待）

## ⑧ IMAGE
（待）

## ⑨ RE-CHECK
（待）
