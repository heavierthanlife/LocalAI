# QA-Loop Round 017 (2026-09-09)

基线: last_head=82ddc53 | 模式: full（T1 Playwright 审计 admin/user 深面 + anon 引导 401/403 判定） | 触发: user（go → 续跑 backlog）

## ① 完成
- e2e 栈重启 + seed（幂等）
- **登录驱动改 API 直登**（accounts modal 难逆向；admin /login 走 ADMIN_PASSWORD_HASH←ADMIN_PIN）：ctx.request POST /login JSON → cookie 共享 → 页面即会话。admin(CEO)/user(e2euser) 均 200
- `run_ui_audit.py` 重构 main：3 会话（admin/user tour 4 主 tab + anon 短走）+ pageerror/console.error/HTTP≥400 采集

## ② 结果
- **admin + user tour：0 failures** —— tab-chat/projects/recycle/analytics 各 x2，214 次动作，无 pageerror/console.error/4xx/5xx。这覆盖了"账户面板/系统管理/回收站/项目"深面，未再发现死链类 bug（T0 已清）
- ledger：295 元素 / acted 214 / blocked-danger 20 / unexplained 0
- **anon-home：14 failures（全部）** —— 匿名页引导调用特权端点：
  - `/cases?page=1` `/templates?page=1` `/notebook` → 401（未登录仍调用，cases.js/knowledge-lab 视图函数在 SPA 启动即跑）
  - `/check_storage` → 403 ×3（三重调用）
  - `/admin/projects` → 403（knowledge-lab 项目归档视图匿名触发）

## ③ 判定（anon 引导噪声）
- **非 bug 级**：401/403 正确返回、无崩溃、admin/user 正常。属"客户端未按权限守门 → 控制台 401/403 噪声 + 无用请求"
- **verdict**：client-side gate（匿名不调 /cases /templates /notebook /admin/projects；check_storage 去重为一次）→ backlog 低危清理项

## ④ 提交
- 本文件 + 无代码变更（驱动已在上轮提交）。audit 产物 data/qa_loop/audit/ 已含本轮 admin/user 全绿 ledger
- e2e 栈 teardown

## ⑤ backlog（下轮）
- anon bootstrap client 守门（cases/templates/notebook/admin-projects/check_storage×3）
- audit_trips 深度之旅（清标 fixture 实跑 / VL 实图 / 剽窃对比 / provider refresh）
- ledger 递归多 surface（侧栏项目/公司/wiki 面板按权限展开）+ visual overflow 断言
- --gate 白名单：anon 引导 401/403 类已知项允许放行后再开 hard gate
