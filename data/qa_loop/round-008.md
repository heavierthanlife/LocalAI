# QA-Loop Round 008 (2026-09-08)

基线: last_head=ff7ca33 | 模式: full（46 项指标语义错配系统性排查 + A/B/C 三批修复） | 触发: manual（用户发现 3.1.6 联系人雷同误报，要求排查同类问题）

## ① COLLECT 摘要（用户反馈 + 全量排查）
- **用户反馈**：3.1.6「同标段单位联系人雷同」得分 15，但"共同关键词"全是行业词（包装/供应商/采购/商品/超市），不是联系人——误报。
- **explore 子代理全量核对 46 项指标**（20 项发现），基快照验证：contact 0.5 分误报、relationship 28.5 分误报、tech_seal 空壳、bidder_count 漏报、text_sim 三指标重复、same/cross_machine_code 同逻辑等。

## ② VERIFY 初判表（子代理 + 基线实证）
| # | 位置 | 初判 |
|---|------|------|
| C4-1 | contact_person_same/cross_contact_same/contact_phone_abnormal checker=key_info/skip | **有效误报/漏报**：关键词冒充联系人雷同 |
| C4-2 | bidder_agent_contact/expert_bidder_closeness checker=relationship | **有效误报**：无名单仍跑关系报告给 28.5 分 |
| C4-3 | tech_seal_check checker=typo | **有效错配**：错别字冒充暗标检测 |
| C4-4 | bidder_count_abnormal checker=skip | **有效漏报**：n<3 真信号被吞 |
| C4-5 | extract_keywords 不消费行业词表 | **有效**：共同关键词行业词污染 |
| C4-6 | text_sim 三指标重复 / same·cross_machine_code 同逻辑 / 专家占位压低指数 | **有效**：去重与语义修正 |

## ③④ CROSS-EXAM / CONFIRM
- 全部批准（用户确认：A+B+C 全做；relationship 真 skip；真暗标检测；刷新基线）。

## ⑤ IMPLEMENT（批次 A+B+C）
- **A1** 新 `app/services/contact_extractor.py`：正文提取联系人姓名/手机号/邮箱，跨文件比对；无数据→"○ 需开标信息表/联系人数据"占位（score=0）
- **A2** `bidder_agent_contact`/`expert_bidder_closeness` → checker=skip（真 skip，不再跑关系报告）
- **A3** 新 `app/services/tech_seal_detector.py` + `tech_seal_check` checker=tech_seal：技术方案段公司名/印章/大量人名→泄露（校准：自指称谓不独立触发）
- **A4** `bidder_count_abnormal` → checker=bidder_count：本地 n<3 触发
- **A5** `extract_keywords`/`keyword_overlap_similarity`/`build_key_info_matches` 加 extra_stop_words（行业词过滤）
- **B1** `_weighted_total_score` docstring 修正 + cross_contact_same 去重
- **B3** `same_machine_code`→"文件作者/编制人雷同"改名；`cross_machine_code`→skip
- **B5** 专家占位保持 skipped（不压指数）；`specific_expert_score` 长期 skip
- **C1** `clearance_openinfo.MAP_COLUMNS` 放开平台列 + `_platform_signals`（开标表含 IP/文件码/加密锁时激活）
- **C2** `extract_metadata` docx 补 creator/producer
- **C3** `suspected_units` file_score 补 lasteditor 组加成

## ⑥ DOCS
- CHANGELOG 新增 [2026-09-08] round-008 条目。

## ⑦ PUSH
- 见 commit：<待填>

## ⑧ 验证
- 回归 **115/115 passed**（新增 4：contact 真实比对/无数据占位/tech_seal 泄露与非泄露/bidder_count 本地）· verify_fixes 89/89 · check_system 133/137
- 基线 scores.json 刷新（contact 0.5→0、relationship 28.5→skip、tech_seal 空壳→检测、bidder_count skip→触发），composite 19.0（正常）
- 注：真实 3 文件容器内全量复测因大文件提取超时未完成；contact/tech_seal 逻辑已用合成用例验证（联系人李晓锋/王月/邱光彩各不同→0 分；技术方案段公司名→泄露）

## ⑨ RE-CHECK / 停跑判定
- 无新增 Critical/High → 质量闸门通过。
- 遗留：真实 3 文件容器内全量 run_analysis 复测待容器资源充足时补跑（本机合成已验证核心逻辑）。
