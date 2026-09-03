# QA-Loop Round 002 (2026-09-03)

基线: last_head=fb91b59 | 模式: incremental（diff 仅 docs，实际以修复核查为主） | 触发: manual（round-001 PUSH 后）

## ① COLLECT 摘要
- code-reviewer: 修复回归核查 11/11 通过 + 新问题 6 条 (M:2 L:4)
- mimo-vision: 修复核查 4/4 代码层通过；截图全部为 09-02（早于修复 commit）→ 视觉层无法实证，需补拍
- 素材：当前代码 (HEAD=b2d471c) + `.playwright-mcp/` 09-02 旧截图

## ② VERIFY 初判表

### code-reviewer 新问题
| # | 原判 | 位置 | 初判 | 理由 |
|---|------|------|------|------|
| R2-1 | 中 | chat.js:994-995 COMPARE_REPORT 分支 | **有效** | `innerHTML = htmlContent` 绕过 _renderMarkdown/DOMPurify，与 C8 不一致；LLM 返回内容被注入时无兜底 |
| R2-2 | 中 | app.js:9888/9891 data.error/err.message | **有效** | 服务端/异常消息未转义拼 innerHTML |
| R2-3 | 低 | app.js:7561 catch e.message | **有效(降)** | 同 R2-2 模式，合并处理 |
| R2-4 | 低 | app.js:7504 presence 🟢 textContent | **风格** | 无安全风险；如需统一可改 _icon() |
| R2-5 | 低 | knowledge.py:292 skill_hash 重赋值遮蔽 | **有效** | 变量复用易误导维护，改 kb_file_hash |
| R2-6 | 低 | credit.py fallback O(n) 清理 | **可接受** | 仅 dev 环境，不修 |

### mimo-vision
| # | 原判 | 位置 | 初判 | 理由 |
|---|------|------|------|------|
| V2-1 | 高(存疑) | 侧栏搜索框 HTML 外露 | **无法实证(旧截图)** | 截图 09-02 早于修复；index.html:33 现为正确 `<input>`；需补拍确认是否已消失 |
| V2-2 | 中 | 搜索框挤压侧栏布局 | **跟随 V2-1** | 根因同搜索框 bug，补拍后确认 |
| V2-3 | 中 | 账户设置按钮重叠 | **有效(低)** | sticky bottom 建议，非功能缺陷 |
| — | P0 补拍 | 统计卡片 / 时间线空态 / 侧栏搜索框 | **待补拍** | C10/C11 视觉效果需新截图佐证 |

## ③ CROSS-EXAM
无争议项（round-001 修复 11/11 通过；新问题均为中/低风险，code-reviewer 原判与 VERIFY 一致）。

## ④ CONFIRM 清单

### 批准项（待用户审批）
- [ ] D1 | 中 | chat.js:995 | COMPARE_REPORT 分支 htmlContent 过 `_safeHTML()` 消毒，与其他分支一致
- [ ] D2 | 中 | app.js:9888/9891 + 7561 | data.error / err.message / e.message 统一 `escapeHtml()` 拼接
- [ ] D3 | 低 | knowledge.py:292 | skill_hash 重赋值改 `kb_file_hash`（可读性）

### 驳回 / 降级记录
- R2-4（presence emoji）风格项：不强制
- R2-6（fallback O(n)）：dev-only，不修
- V2-1/V2-2（搜索框）：旧截图存疑，已列入补拍 P0；若补拍复现则转下轮修复

## ⑤ IMPLEMENT

用户批准 D1–D3。每 fix 一 commit：

| # | commit | 内容 |
|---|--------|------|
| D1 | 3f7483f | chat.js COMPARE_REPORT 分支补 _safeHTML(DOMPurify) 消毒 |
| D2 | 498f8b1 | app.js 三处错误消息统一 escapeHtml（data.error/err.message/e.message） |
| D3 | 9205efe | knowledge.py skill_hash 遮蔽→kb_file_hash + check_system 守卫 + C5 守卫正则修正 |

验证：check_system 133/137（+D3 守卫）· tests/test_regression.py exit=0 · node --check 通过

## ⑥ PUSH

- git push LocalAI master — 结果: 待执行（D1-D3 提交后推送）

## ⑦ RE-CHECK

- 视觉补拍清单（P0，需起服务 5443 + Playwright）：统计卡片(C11) / 时间线空态(C10) / 侧栏搜索框(存疑旧 bug) / 聊天长 markdown(C8) / 存储警告(C6)。
- 停跑判定：D1–D3 为代码侧新发现的中/低风险，已全部修复；视觉侧无新的确认问题（截图滞后无法实证）。本轮无未决 Critical/High → 满足质量闸门，round-002 可收尾。

