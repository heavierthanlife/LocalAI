# QA-Loop Round 023 (2026-09-11)
基线: last_head=91f0a04 | 模式: 定向修复（用户报告：重点信息雷同假警报 + 报价异常可靠性）| 触发: manual

## 背景
用户清标报告发现两类假警报：
1. 「重点信息雷同」6.7 表把通用词（公司/工作/检查/食品/填写/偏离/提供/负责）当证据。
2. `high_price_abnormal`（报价异常）在无开标信息表 + N=2 时仍报「发现 2 个投标单位报价疑义」。

## ① COLLECT / ② VERIFY（主 agent 读码定位）
- 假警报根因：`build_key_info_matches` 交集非空即输出；在原始文本抽词（未模板去除）；停用词表缺通用词；N=2 时 IDF 无区分力。
- 报价异常根因：无结构化开标价时 `extract_prices` 噪声大；指标仍计分。

## ④ CONFIRM（用户确认）
- 重点信息雷同：保留但仅显著时显示（≥3 词 + Jaccard≥0.15）。
- 报价异常：无开标表时降级「仅参考」。
- 方案：A（停用词扩容）+ C（显著性门槛）+ D（词性过滤）+ file_scores 校准。

## ⑤ IMPLEMENT（commit 见下）
- FIX-045：`stop_words.py` 扩容；`file_processing.extract_keywords(pos_filter=False)` + `_keep_nounish`；`build_key_info_matches(template_text=...)` 预处理+模板去除 + ≥3/Jaccard≥0.15 门槛；两调用点传 `tender_text`；`file_scores` 自动只计显著词。
- FIX-046：`document_analysis_svc.run_analysis` 计算 `quote_has_open_prices`，无开标报价 → quote 指标 `score=0` + 「仅作参考」。
- 快照重标定：`tests/test_batch_orchestrator.py`（tfidf 0.14892089/0.77167538；slight_diff 0.35714286）。

## ⑥ DOCS
- CHANGELOG 顶部新增条目；fix_registry FIX-045/046；`tests/test_regression.py` +4 项。

## ⑦ PUSH
（见 commit 结果）

## ⑧ IMAGE
纯后端改动（`app/services/*`），但涉及运行服务逻辑 → 重建镜像并 `--force-recreate` app/celery-worker/celery-beat。

## ⑨ RE-CHECK
- 判别力验证：`test_text_sim_stopwords_discrimination`、`test_industry_words_*`、`test_key_info_*` 全通过；near(0.556) > slight(0.357) > none(0.0) 保持。
- 新增 Critical/High：0。
