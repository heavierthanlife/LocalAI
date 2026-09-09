# QA-Loop Round 016 (2026-09-09)

基线: last_head=f573588 | 模式: full（UI/interaction "find-all" 审计 T0→T1，VL 可靠性 round-015 前置完成） | 触发: user（"上次你说查未完成功能没找全，否则 vl_test 早该发现"+ 要求规划 Playwright 全量点按审计）

## ① 结论（T0 静态死链交叉检查 — 已交付并修复）
新增 `scripts/audit_js_routes.py`：call-anchored 提取 16 JS+templates 的 fetch/axios/XHR/ajax/href/action/window.open → 路径模板+方法感知解析 app url_map → 三档（no-route/dynamic/external）。**首轮跑出 5 处真死链并全修**：
1. **audit_bp 从未注册**（register_all 缺 audit，audit.py 全路由闲置）→ /audit/* 全部 404 → 补注册复活整个审计后端
2. /knowledge_lab/feedback 无路由（技能点赞点踩 404）→ 参照 ingest 反馈实现
3. /set_video_analysis 无路由 → 镜像 /set_image_analysis
4. /admin/skill_supersession/respond 无路由 → 前端指向现有 /admin/skill_merge（路径+响应字段双错）
5. /batch/plagiarism/compare 404 → batch_bp 根级路由，前端 /batch 前缀 → 补 alias

**T0 复跑：matched=84 no_route=0**。method_mismatch(104)/dynamic(122) 低置信/运行时项留 T1。这正是用户"否则 vl_test 早该发现"的同类 bug——vl_test 属 dynamic?（实际是静态/路由缺失，T0 即抓）——本轮先以静态方式把"前端调用指向不存在路由"类全部揪出。

## ② T1（Playwright 可达性审计 — 首切已通）
- `docker-compose.e2e.yml`（throwaway：复用镜像 local-ai:latest、container_name localai-e2e-*、:4443 nginx TLS、e2e_* 新卷、`docker-compose -p e2e`）
- `scripts/e2e_seed.py`（CEO admin + e2euser + PIN 123456，幂等）
- fixtures `tests/fixtures/audit/`：vl_test.png（已知数字 12345.67/88000）+ bid_alpha/bid_beta/tender.docx
- `scripts/run_ui_audit.py`：expand-all + 枚举交互元素 → safe-action 策略 + ledger（exercised/blocked:reason/unexplained 三态）+ pageerror/console.error/HTTP≥400 采集；报告模式 exit 0
- **实跑（anon-home）**：34 元素 / acted 23 / failures 15 → 均为**匿名页引导调用 401/403**（/cases /templates /notebook 401；/check_storage×3 403；/admin/projects 403）——低危但真实（客户端在未登录/非管理员时仍发权限端点请求，/check_storage 重复 3 次疑似重试循环），列入待办
- **登录未达 admin 面**：应用首页无登录表单（需点 账户设置→登录 tab）→ admin 深面（runtime-config LLM/VL、审计历史、分析面板）walk 为下一迭代

## ③ 记录
- e2e 栈 `docker-compose -p e2e -f docker-compose.e2e.yml down`（保留卷供续跑）；admin 登录驱动 + 递归 tab walk + audit_trips 深度之旅为下轮
- CHANGELOG round-016 + fix_registry FIX-2026-09-09-023（T0）+ AGENTS.md 测试节补 T0/T1 用法

## ④ 验证
- 回归 128/128 · verify 110/110 · 容器 e2e site 200 healthy · T0 no_route=0

## ⑤ backlog（下轮）
- admin 会话注入/登录驱动（账户设置→登录 tab）→ walk admin 深面
- /check_storage 等引导 401/403 判定（静默降级 or 前端守门）
- audit_trips（清标 fixture 实跑 / vl_test 实图 / 剽窃对比 / provider refresh）
- ledger 递归多 surface + visual overflow 断言完善
