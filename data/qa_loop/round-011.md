# QA-Loop Round 011 (2026-09-09)

基线: last_head=4b3931c | 模式: focused（暗标违规检测开关 + 盖章弱信号降权） | 触发: user（普通标书盖章误报暗标违规）

## ① COLLECT 摘要（用户反馈）
- 用户贴出封面警示：`⚠ 暗标违规：正文出现「盖章」提示；「公章」提示；「签字盖章」提示；技术方案段出现公司名「北京物美…」「北京中昌华美…」`。
- 用户判断：绝大多数标**不是暗标**（普通标书正文含盖章/公章属招标正常要求），应加开关：
  - 放在分析维度行（横向对比/指标分析/合规审查/AI 评审同一行）
  - **默认不勾选，默认不检查暗标违规**

## ② VERIFY 根因
- `tech_seal_detector.py` `_SEAL_MARKERS`（盖章/公章/签字盖章/投标专用章）任一命中即 `leak=True` → 任何含"盖章"的普通标书都触发 `■ 高度预警（暗标违规）` 硬警示 → 全标误报。
- 技术方案段出现**公司名**才是真实暗标身份泄露；盖章词是弱信号（普通标书必然出现）。

## ③④ CROSS-EXAM / CONFIRM（用户拍板 2 项）
1. **默认关闭**同时影响非清标文档分析路径（`run_analysis` 调用方不传 options → 默认 None → tech_seal 跳过）
2. **盖章弱信号降权**：`_SEAL_MARKERS` 不再独立触发 leak，仅当强信号（技术方案段公司名 / ≥4 人员姓名）已判泄露时作为辅助证据列出

## ⑤ IMPLEMENT（3 并行 + 1 串行）
- **Wave1-A（general）** 后端门控：`run_analysis` 加 `options=None`；checker 循环 gate（`tech_seal_enabled`，未开启 → `{'skipped':True,'error':'未开启暗标违规检查'}`）；`_run_indicator_analysis` + `run_clearance` futures 透传 options；`clearance.py` `setdefault('tech_seal_check', False)`。两态验证：默认关→violation_fired False + skipped；开启→violation_fired True + `■ 高度预警（暗标违规）`
- **Wave1-B（general）** `tech_seal_detector.py` 降权：顺序重排（公司名→人员→盖章辅助）；`seal_hits` 仅当 leak 已 True 时追加辅助证据；docstring 更新。三用例验证（纯盖章→False；公司名技术段→True；+盖章辅助）
- **Wave1-C（frontend-dev）** `templates/index.html` 分析维度行加 `optTechSeal` 复选框（无 checked 默认关）+ title 提示；`app.js` options 加 `tech_seal_check`。node OK
- **Wave2-D（general）** `tests/test_regression.py`：`test_tech_seal_violation_independent` 改传 `options={'tech_seal_check': True}`；新增 `test_tech_seal_default_off` + `test_tech_seal_seal_marker_weak_signal`。111 passed（两文件）

## ⑥ DOCS
- CHANGELOG [2026-09-09] round-011 条目 + fix_registry FIX-2026-09-09-019（6 条 invariant）。

## ⑦ PUSH
- `4b3931c..8c40637` 已推送（LocalAI master），工作树干净。last_head=8c40637。

## ⑧ 验证
- 回归 **121/121 passed**（原 119 + 2 新）· verify_fixes **101/101**（+5，去 1 重复）· check_system 133/137 · node --check app.js · 后端 syntax 全过
- fix_registry 修坑：`()` 在正则里是分组符非字面括号 → 带括号的 pattern 语义错误，改用无元字符 pattern（`tech_seal_check: document` / `tech_seal_check', False`）
- 合成三态验证（Wave1-A/B）：默认关→无违规；开启→`■ 高度预警（暗标违规）`；纯盖章→leak False

## ⑨ RE-CHECK / 停跑判定
- 无新增 Critical/High → 闸门通过。
- 遗留：非清标文档分析路径（无勾选框）tech_seal 永久关闭（用户确认的默认行为）；若未来该路径也要可配，需另加开关。