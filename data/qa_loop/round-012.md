# QA-Loop Round 012 (2026-09-09)

基线: last_head=4933d63 | 模式: full（清标报告 UI 改造 + 归属双写 + DOCX 警示表格式化） | 触发: user（6 项需求）

## ① COLLECT 摘要（用户 6 项）
1. 折叠箭头 ▼/▶ → Material Symbols expand_more + CSS 旋转
2. 对比报告误进项目对话，应在个人对话
3. 大标题颜色跟随下属最高警示
4. 下载链接与大标题之间加警示总结（表格）
5. 各大标题下参数展示统一表格
6. DOCX 报告所有警示表格化；铁证不再纯字；警示项带页码/原文/指引

## ② VERIFY（2 explore 子代理深挖）
- F1 前端：入口 `buildClearanceReportHtml`（下载链接→六章 details/cl-l1）；指标=div 卡片混合；大标题颜色固定无跟随逻辑；警示总结插入点 :9292；quoteBubble 用 ▼/▶；`_toggleArrow`+`msi-arrow` CSS 已就绪可复用
- F2 归属根因：提交后无条件 append 进全局 #chatMessages（当前激活 thread）；后端用 `session['thread_id']`（可能是项目对话）；`_handleTaskClick` 有 thread-aware 保护但主路径缺；附带 bug `window.currentProjectId` 恒空
- B1 DOCX：封面铁证/违规确认为无排版纯文字（:854-872）；指标/六~十章多已表格；`extract_text_from_path` 返回 `(text, page_texts)` 但调用方全丢弃；文本类警示可反查原文、元数据/平台类无原文

## ③④ CROSS-EXAM / CONFIRM（用户拍板）
1. **归属双重**：跟随当前项目对话归属 + 同步该用户最新个人对话；个人对话跑→进最新个人对话
2. **不做页码**：只指目录章节位置（已有上下文）

## ⑤ IMPLEMENT（3 并行 + 1 行修复）
- **F1（frontend-dev）** 4 项：
  - 箭头：quoteBubble `_toggleArrow`（+index.html 初始态）；报告全部 cl-l1/cl-l2/cl-l3/alert-parent summary 加 `_clArrow` + 委托点击同步 `.collapsed`
  - 警示总结 `_renderAlertSummary`（:9341，9 行汇总表，插入 :9398 下载链接后）
  - 大标题着色 `_clearanceSectionSev` + `cl-danger/cl-warn` class + app.css 样式
  - 统一表格：一 指标 div→表格（含 details 折叠行）、二 集团→表格、三 徽章→表格、四 评分→表格、五/六 文本→表格
- **F2（general）** 归属双写：`resolve_clearance_threads(user_id, current)`（clearance_engine.py:32，判项目/查最新个人无则建/组目标列表去重；修 project_id INTEGER 坑用 `COALESCE(project_id::text,'')=''`）；`run_clearance_async` 持久化遍历目标线程各插 CLEARANCE_REPORT；clearance.py:158 预解析入 extra。真库 5 用例验证 + 保底回退
- **B1（general）** DOCX：封面铁证/违规→表格（证据类型/级别/证据文本/涉及文件）；新增无编号"警示详情与处理指引"节（5 类警示源各一表：铁证/违规/6.9 段落/高风险 pairs/合规 critical）；helpers `_locate_chapter`/`_excerpt`/`_guidance_for`/`_warning_table`；report 不携带正文 text → 原文取 evidence 内嵌 snippet、章节用 type 映射、元数据/平台类写"证据来源：文件属性/交易平台记录"；不重排 6.1-6.10 编号
- **self** F2 前端一行：`window.currentProjectId` → 裸 `currentProjectId`

## ⑥ DOCS
- CHANGELOG [2026-09-09] round-012 条目 + SYSTEM_CHECKLIST 再生。

## ⑦ PUSH
- `4933d63..fead0e6` 已推送（LocalAI master），工作树干净。last_head=fead0e6。

## ⑧ 验证
- 回归 **121/121** · verify_fixes **101/101** · check_system 133/137 · node --check app.js · 后端 syntax OK
- F1 冒烟 21/21（含 XSS 泄漏检测、老报告守卫、章节 severity 映射）· F2 真库 5 用例（项目双写/个人单写/个人非最新双写/空会话建/DB失败回退）· B1 合成 15/15（5 类警示子节、元数据占位、章节映射、空报告守卫）

## ⑨ RE-CHECK / 停跑判定
- 无新增 Critical/High → 闸门通过。
- 遗留：B1 章节定位目前用 type 映射（report 不携带正文 text）；若未来 report 携带 `_pages`/text，`_locate_chapter` 回扫逻辑可启用（本轮按用户决定不做页码）。DOCX 页码明确不做。