# QA-Loop Round 007 (2026-09-08)

基线: last_head=9c147db | 模式: full（用户要求通读三源文件+新报告，评估算法贴切度） | 触发: manual

## ① COLLECT 摘要（用户要求 + 子代理分工）
- **用户要求**：通读桌面三源文件 + 本次生成的《串通投标线索分析报告 (2).docx》，评估算法与结论是否贴切，子代理分工研究后给详细报告。
- **explore 子代理**（通读三源文件，2,346/3,506/2,132 段）：
  - **真铁证**：①元丰↔物美 服务理念 8 条逐字复制（含"职工食堂氛围"外源残留）；②元丰↔中昌华美 应急响应块逐字复制（仅"2分钟/10分钟"数字微调）；③元丰↔物美 应急预案/仓储/冷链/货源多块逐字；④**元数据**：元丰与中昌华美 `lastModifiedBy` 都是「超彩赵」+ 同日(2024-12-17) + 同 WPS 版本——同一人/同机做两家标书
  - **模板噪音**：投标函 6 条、评审索引表、「完全响应第六章」承诺等（套招标模板的合规内容）
- **code-reviewer 子代理**（评估报告贴切度）：6.9 信号纯度不贴切（195 行中 190 行是"其他"，投标函模板段污染）、集团判定过度（门槛=1 共享段）、风险度部分虚高、冒烟指数 36 合理偏保守、漏报（元数据硬信号未进报告）

## ② VERIFY 初判表
| # | 位置 | 初判 |
|---|------|------|
| C3-1 | `_LEGAL_TERMS` 缺投标函模板词 | **有效**：扩充 正本/副本/有效期/声明/真实有效/待命/供应/腐烂/变质 等 |
| C3-2 | `_classify_paragraph` 门槛失衡 | **有效**：新增「投标函声明段」类型，legal>=2、tech>=2 |
| C3-3 | collusion_score 含模板段 | **有效**：评分/per_pair 只统计非模板实质段 |
| C3-4 | 集团判定门槛过低 | **有效**：detect_gangs 要求 ≥2 非模板实质段（evidence_counts） |
| C3-5 | 元数据硬信号缺失 | **有效**：extract_metadata 补 last_modified_by + 新指标 file_attr_lasteditor_same |
| C3-6 | 6.9 呈现噪音 | **有效**：铁证优先（按 surprise 降序）+ 6.9.2 模板折叠 |

## ③ CROSS-EXAM
- 无争议。

## ④ CONFIRM 清单
- 全部批准（用户确认：元数据信号纳入；6.9 铁证优先+模板折叠；集团门槛=2）。

## ⑤ IMPLEMENT
- `paragraph_collusion_detector.py`：扩充 _LEGAL_TERMS；新增「投标函声明段」分类 + 门槛修正（legal>=2/tech>=2/svc>=2）；匹配分流实质段 vs 模板段（template_segments）；collusion_score/per_pair 只统计实质段
- `file_processing.py`：extract_metadata 补 `last_modified_by`（cp:lastModifiedBy）
- `batch_orchestrator.py`：build_attr_details 补 last_modified_by/modified；detect_gangs 加 evidence_counts + min_evidence=2
- `indicator_defs.py`：新增 `file_attr_lasteditor_same`（触发指标，权重 0.10，cap 30）
- `document_analysis_svc.py`：file_attr 分支按指标分流（lasteditor_groups）；INDICATOR_WEIGHTS/SCORE_CAPS 补新指标；6.9 铁证优先 + 6.9.2 模板折叠
- `clearance_engine.py`：_run_cross_comparison 生成 evidence_counts 并传 detect_gangs

## ⑥ DOCS
- CHANGELOG 新增 [2026-09-08] round-007 条目。

## ⑦ PUSH
- `9c147db..ff7ca33` 已推送（LocalAI master），工作树干净。last_head=ff7ca33。

## ⑧ IMAGE / 真实文件复测
- 容器内重跑（同步代码 + 真实 3 文件）：
  - **shared_segments 195 → 36**（模板噪音排除），collusion_score 69.8
  - 服务承诺段铁证清晰：应急响应"2分钟/10分钟"、食堂氛围、食材新鲜、应急预案块
  - 6.9 铁证优先（surprise 降序）+ 6.9.2 模板折叠 + 最后编辑人指标 均渲染
  - **元数据验证**：`last_modified_by` 提取到「超彩赵」（元丰+中昌华美）→ 分组成功触发
  - 报告 ZIP 取回桌面 `clearance_round007_report.zip`
- 回归 **111/111 passed** · verify_fixes 89/89 · check_system 133/137

## ⑨ RE-CHECK / 停跑判定
- 无新增 Critical/High → 质量闸门通过。
- 遗留观察：6.9 铁证表内"消防/应急预案"公共模板段 surprise 也较高（0.83-0.90）排前，属实质雷同证据但可考虑进一步降权公共安全模板；worker 端到端 file_count=2 边界（round-006 已记录）。
