# QA-Loop Round 009 (2026-09-09)

基线: last_head=9b8761c | 模式: full（铁证双层判定 + 暗标违规独立轨道） | 触发: manual（用户提出铁证信号是否还应参与打分）

## ① COLLECT 摘要（用户问题）
- 用户：铁证级参数（lastModifiedBy 同人"超彩赵"、服务承诺逐字雷同、联系人+电话同组）还需要参与加权打分吗？可以直接拉满跳警报？还是有更好的处理方式？
- 要求：跟子 agent 商量。

## ② VERIFY / 设计推敲（explore 子代理 + 量化）
- **根因**：`_weighted_total_score` 分母含所有非 skip 指标，铁证权重最高仅 0.10 →
  单个铁证归一化打满最多贡献 ~15 分，实际 norm（group_count×10/30）≈5 分；
  段落级逐字雷同（最强证据）根本不计入复合指数（只在横向层矩阵）→ 封面与证据脱节。
- **方案对比**：a 铁证独立 veto / b 权重拉满（数学上会污染正常案例）/ c 两层评分（过度设计）/ d 双层判定（复合指数保留 + 铁证层只升展示级）→ 推荐 d。

## ③④ CROSS-EXAM / CONFIRM（用户拍板）
1. 封面最高档文案：**`■ 高度预警（铁证触发）`**
2. T1 **单命中即 veto**
3. tech_seal 泄露 → **独立"暗标违规"警示，警示高度等同于串标**
4. 铁证+违规并存：可各自独立出现，串标优先、违规附加
5. 设计决策已存 agentmemory（mem_mtsi4efd）

## ⑤ IMPLEMENT（核心 self + 子 agent 并行）
- **核心（self）**：新 `app/services/hard_evidence.py`（HARD_ALARM_RULES 语义 + guard 词表 +
  `assess_hard_evidence()`）；`document_analysis_svc.py` 接入（`resolve_warning_level` +
  `_build_hard_context` + run_analysis basic_info.hard_* + suspected hard_flag + DOCX 封面铁证/违规红字警示段 +
  预警单位 ★）；`clearance_engine.py` run_clearance 终判（补入横向层 paragraph_collusion）
- **frontend-dev 子 agent**：`static/js/app.js` 铁证联合着色（`isClearanceHighRisk/MidRisk` helper +
  三处渲染路径 + 历史列表 ★铁证 + 预警单位 hard_flag ★），node --check 通过
- **general 子 agent**：`tests/test_regression.py` 新增 4 项铁证测试，regression+collusion 109 passed

## ⑥ DOCS
- CHANGELOG [2026-09-09] 条目 + fix_registry FIX-2026-09-09-017（7 条 invariant 检查）。

## ⑦ PUSH
- `9b8761c..32313f8` 已推送（LocalAI master），工作树干净。last_head=32313f8。

## ⑧ 验证
- 回归 **119/119 passed**（原 115 + 4 新增）· verify_fixes **96/96**（+7）· check_system 133/137 · app.js node --check OK
- 逻辑单测 11/11（T1 各信号/guard 反例/T2 共证/段 2 段升级/段 3 家升级/违规独立）
- **真实 3 文件复测 PASS**：元丰+中昌华美 `lastModifiedBy='超彩赵'`（物美='唯一的麦麦儿'）→
  T1 veto，`fired=True level=T1 label='■ 高度预警（铁证触发）'`，证据"最后编辑人「超彩赵」同时出现在 2 家投标文件"
- 复合指数未被改写（铁证只升展示级），基线 scores.json 19.0 不变、DB max_risk 可比性保留

## ⑨ RE-CHECK / 停跑判定
- 无新增 Critical/High → 质量闸门通过。
- 遗留：真实 3 文件容器内全量 run_analysis 复测仍受大文件提取耗时限制未跑；
  已用真实元数据（core.xml 直读）+ 合成段落/联系人用例覆盖核心判定路径。