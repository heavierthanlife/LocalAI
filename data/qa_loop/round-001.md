# QA-Loop Round 001 (2026-09-03)

基线: last_head=3c04174b | 模式: full（首轮） | 触发: manual（会话内首次，含已完成的 COLLECT）

## ① COLLECT 摘要
- code-reviewer: 15 条 (C:3 H:5 M:5 L:2)
- mimo-vision: 12 条 (H:6 M:4 L:2)
- 素材：代码库实时审查 + `.playwright-mcp/` 2026-09-02 截图 21 张

## ② VERIFY 初判表

### code-reviewer（代码侧）

| # | 原判 | 位置 | 初判 | 理由 |
|---|------|------|------|------|
| R1 | Critical | knowledge.py:512 空路由装饰器 | **有效** | `/admin/all_user_kb`(GET) 装饰器孤立，Python 绑定到下一个 def `admin_generate_work_report`；真实 `admin_all_user_kb`(867) 无路由不可达。双重 bug |
| R2 | Critical | knowledge.py:262 IDOR | **有效** | `generate_project_file_skill` 只查 session.user_id 存在，不校验项目成员；任意登录用户可遍历 file_id 读取/外带任意项目文件内容并写入自己 knowledge_lab |
| R3 | Critical | database.py:66-83 double-put | **误报→Low** | line 75 `conn,close=True` 关旧坏连接；line 76 `conn=` 重新赋值取新连接；finally 归还是新连接，两物理连接各 get/put 一次，无 double-put。残留：无用借还开销（CROSS-EXAM 确认） |
| R4 | High | admin.py:142-148 admin_required | **降级→Medium** | 与 login_required 不一致属实，但需已有 admin session 才可利用 → 纵深防御缺口非独立漏洞（CROSS-EXAM 确认） |
| R5 | High | credit.py:27 进程内存限速 | **有效** | `_credit_rate_limit={}` 进程内 dict，gunicorn 4 worker 各自独立，可绕 10次/5min |
| R6 | High | knowledge.py:708 文件句柄 | **有效** | `open(zip_path,'rb').read()` 匿名 open 未 close，异常/高并发路径 FD 泄漏 |
| R7 | High | app.js:1944 warningSpan.innerHTML | **有效(降Medium)** | `/check_storage` 自建 API，message 若非纯服务端则存储 XSS；应 escapeHtml 或 textContent |
| R8 | High | chat.js:317 md.render XSS | **降级→Low** | markdownit `html:false`(app.js:46) 转义原始 HTML + 默认 validateLink 拦危险协议，无可复现绕过（CROSS-EXAM 确认）。DOMPurify 属纵深建议 |
| R9 | Medium | knowledge.py:569 字符串替换列名 | **有效** | `user_filter.replace('cs.user_id','user_id')` 对空格/换行格式敏感，静默失败风险 |
| R10 | Medium | admin_knowledge_lab.py:61 count(*) 全表 | **有效(降Low)** | 表名来自 PG 系统表非用户输入，仅超大表 DoS 风险，建议白名单/超时 |
| R11 | Medium | knowledge.py:439,479 无 consent | **有效(降Low)** | admin 权限已限制；与 admin_required 同类不一致，随 R4 一并修 |
| R12 | Medium | docker-compose.yml:34 弱默认密钥 | **有效(降Low)** | 占位默认 + 启动校验缺口，建议 _validate_env 加非占位校验 |
| R13 | Medium | database.py:153 admin seed 哈希不更新 | **有效(doc-only)** | ADMIN_PIN 修改后已建账号不更新，需文档说明 |
| R14 | Low | knowledge.py:714 绝对路径返回 | **有效** | 返回完整 zip_path 泄露目录结构 |
| R15 | Low | upload.py:28 11GB vs 50MB 文档不一致 | **有效** | 实际行为有 quota 兜底，文档需对齐 |

### mimo-vision（视觉侧）

| # | 原判 | 截图/位置 | 初判 | 理由 |
|---|------|-----|------|------|
| V1 | High | 侧边栏搜索框 HTML 泄露 | **过期** | 截图(09-02 06:xx)早于 `1b34740`(16:42+0800)；index.html:33/629 现为纯文本 "快速筛选..."/"搜索笔记..." |
| V2 | High | 笔记本搜索框 HTML 泄露 | **过期** | 同 V1，已修 |
| V3 | High | 批量对比乱码文件名 | **有效** | DOCX 解析出的二进制内容被当文本显示，编码清洗缺失 |
| V4 | High | 聊天消息重复显示 | **部分有效** | 需运行时复验（消息列表 append 或后端重复返回），标记待运行时确认 |
| V5 | High | 全部技能"加载失败"无重试 | **部分有效** | 底层加载失败仍在；M4(3c04174b) 已加错误码区分，显示层改善，根因待运行时确认 |
| V6 | High | 统计图标 emoji/方块乱码 | **有效(降Medium)** | 截图见 👥5·🟢0·💾5 混排；app.js:7891 用 emoji 内联 `<span>`，与全站 MSI 不一致 |
| V7 | Medium | 清标报告跳"五" | **过期** | 同 V1 时段截图；M6 已在 `1b34740` 加 else 占位（现 五/六 均有占位） |
| V8 | Medium | 项目名 KekKekKek233 | **资料性** | 测试/恢复项目名，非代码 bug，运维处理 |
| V9 | Medium | 回收站统计冗余 | **有效(降Low)** | 空状态 UI 冗余，非功能缺陷 |
| V10 | Low | 时间线空态无引导 | **有效** | 空状态加"去选择项目"链接 |
| V11 | Low | PIN 输入框非密码类型 | **误报** | index.html:691/693/702 均已 `type="password"` |
| V12 | Low | 按钮配色不统一 | **有效(降Low)** | 风格建议，非缺陷 |

## ③ CROSS-EXAM

| # | 争议 | 裁决 | 依据 |
|---|------|------|------|
| 3 项 | R3 double-put / R8 XSS / R4 admin_required | **全部接受降级** | code-reviewer 复核：R3 conn 重赋值无 double-put；R8 html:false+validateLink 无可复现注入；R4 纵深防御缺口非独立漏洞 |

## ④ CONFIRM 清单

### 批准项（待用户审批后进入 IMPLEMENT）
- [ ] C1 | High | knowledge.py:512 | 补齐 `/admin/all_user_kb` 函数实现（或删除孤立装饰器），解除对 generate_work_report 的错误绑定
- [ ] C2 | High | knowledge.py:262 | generate_project_file_skill 增加项目成员校验（复用 _require_project_access 类逻辑）
- [ ] C3 | Medium | admin.py:142 | admin_required 复用 consent+user_id 检查（抽公共 `_check_session_valid()`）
- [ ] C4 | Medium | credit.py:27 | _credit_rate_limit 迁移 Redis 或 flask-limiter Redis backend
- [ ] C5 | Medium | knowledge.py:708 | 文件打开改 `with open(...)` 关闭，消除 FD 泄漏
- [ ] C6 | Medium | app.js:1944 | data.message 用 escapeHtml 或 textContent 注入
- [ ] C7 | Low | knowledge.py:569 | 列名适配改为结构化构建，去掉 str.replace 后处理
- [ ] C8 | Low | chat.js | （纵深）统一补 DOMPurify sanitize 包装 md.render 输出
- [ ] C9 | Low | V3 | DOCX 文本提取后过滤非 UTF-8 可打印字符，防乱码显示
- [ ] C10 | Low | V10 | 时间线空态加"去选择项目"引导链接
- [ ] C11 | Low | V6 | app.js:7891 状态统计换 MSI 图标对齐全站

### 驳回 / 降级记录（核心资产）
- R1-R15 / V1-V12 全部条目判定如上表；V1/V2/V7 过期（已修）、V11 误报（密码类型）、R3/R8/R4 降级（CROSS-EXAM 三方确认）、R10/R11/R12/R6/V9/V12 降级。
- **教训**：mimo-vision 截图只用昨天的旧图 → 主 agent 必须以截图时间戳 vs commit 时间比对，才能避免把已修问题当新问题（本轮回圈已验证该机制有效）。

## ⑤ IMPLEMENT

用户批准 A（全部 11 条）。批内每 fix 一 commit：

| # | commit | 内容 |
|---|--------|------|
| C1 | c85fc34 | knowledge.py 空路由装饰器 + check_system 孤立装饰器守卫 |
| C2 | 1cb94dc | generate_project_file_skill IDOR 成员校验 + 守卫 |
| C3 | 85775f2 | admin_required/auditor_required 会话校验 + 守卫 |
| C4 | 9eecab3 | credit 限速器迁 Redis + 2 回归测试 + 守卫 |
| C5 | 27be656 | 工作报告 zip 文件句柄 with open + 守卫 |
| C6 | df6327e | app.js checkStorage message 转义 |
| C7 | 7393161 | work_report filter 结构化构建 + 守卫 |
| C8 | 7489653 | chat.js md.render 统一 DOMPurify 消毒 |
| C9 | c153ee3 | 文本提取乱码/控制字符清洗 |
| C10 | 93c9585 | 时间线空态"选择项目"引导按钮 |
| C11 | fb91b59 | 统计区图标统一 Material Symbols |

验证：16 个 JS node --check 全过 · tests/test_regression.py exit=0 · verify_fixes 89/89 · smoke 7/7

## ⑥ PUSH

- git push LocalAI master — 结果: 待执行（qa-loop skill v2 新增 PUSH 阶段，推送后补录）
- 规则：先推送远端 + 工作树干净，才允许进入下一轮 RE-CHECK。

## ⑦ RE-CHECK

状态: 未跑（建议下一轮增量基线 last_head=fb91b59，按 qa-loop skill 流程 RE-CHECK）。
预期：C1/C2 路由面无新增端点（守卫项 4 项已入 check_system 135→136 项）。
新增回归测试：C4 限速 Redis ×2（tests/test_regression.py 84 项）。