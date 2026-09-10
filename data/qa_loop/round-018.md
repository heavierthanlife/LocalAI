# QA-Loop Round 018 (2026-09-09)

基线: last_head=7b0d26a | 模式: full（T1 全量点按 + T2 深度之旅 + anon 守门 + hard gate） | 触发: user（"全做"）

## ① anon 引导守门（FIX-2026-09-09-024）
- fetch 拦截器移至 `templates/index.html` **<head> 内联**（必须早于 templates.js/cases.js——它们在 app.js 前解析且自跑 loader）。匿名时对 `/cases /templates /notebook /admin/projects /check_storage` 短路为合成 401 Response（调用方早已容忍 401，零网络零噪声）
- 效果：`/check_storage`×3、`/notebook`、`/admin/projects` 噪声消除；残留 `/cases` `/templates` 各一（以正确 401 返回）→ hard gate 白名单放行

## ② T2 深度之旅（scripts/audit_trips.py，admin 会话 API-real）
首轮即由真实链路揪出 **3 个真 bug 并修复**：
1. `batch_comparison_results.project_id`：clearance INSERT 引用但 DDL 无列 → 全新库报 `column "project_id" does not exist`。补列（CREATE + 幂等 ALTER）
2. **`batch_pair_results` 表全新库根本不存在**（clearance/document_analysis INSERT + graph SELECT）→ 补建（含 UNIQUE 供 ON CONFLICT）
3. **numpy 标量泄漏入 SQL**：`np.float32` 直接作 psycopg2 参数 → `schema "np" does not exist`。clearance_engine + document_analysis_svc 的 INSERT 全部 `float()/int()` 归一 + `json default=float`

四趟结果（gate 下全绿）：
- clearance e2e：completed，**46 指标**，report 正常
- vl real image：provider=nvidia，consistent=True，OCR 命中
- plagiarism e2e：ok
- models refresh：models=21，stale=False

## ③ hard gate
`run_ui_audit.py --gate`：**PASS** —— 319 元素 / acted 236 / unexplained 0 / failures_real 0 / benign 14（anon 引导 401/403）。ledger 三态 + benign 白名单显式化

## ④ 验证
- 回归 128/128 · verify 115/115 · e2e gate exit 0 · T2 四趟全绿
- 镜像重建（prod 获得 schema + numpy 修复）

## ⑤ backlog
- 递归多 surface（侧栏按权限子面板）+ visual overflow 断言像素级
- `/cases` `/templates` 匿名引导最后一处守门（当前白名单放行）
- 将 T1/T2 纳入按需 CI 或 nightly
