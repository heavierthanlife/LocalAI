# QA-Loop Round 004 (2026-09-07)

基线: last_head=9205efe | 模式: full（用户定向：真实串标 3 家案例暴露清标算法不可靠） | 触发: manual

## ① COLLECT 摘要（用户报告 + 报告实证）
- **用户反馈**：用已证实串标的 3 家（元丰/物美/中昌华美 军营超市项目）测试，清标系统没抓到真问题——整篇余弦 85-87% 是模板/行业重叠，稀释了服务承诺段「热情」逐字雷同。
- **desktop 报告实证**：无招标文件时指标层正确 skip text_sim（`document_analysis_svc.py:106-110`），但横向对比层 `_run_cross_comparison` 无条件算原始余弦 → RiskScorer ≥80% 门槛放行 → 3 家全判"疑似围标集团"，证据是模板重叠非真信号；`economic_error_similar` 按错别字总数（314 处）计分；矩阵表头显示纯 ".docx"。
- **源文档实证**：元丰 vs 物美「1、/（一）热情、主动、耐心、周到、细致、尽职尽责…」近逐字复制（SequenceMatcher ≈1.0）。

## ② VERIFY 初判表（子代理调查后确认）
| # | 位置 | 初判 |
|---|------|------|
| B1 | `clearance_engine.py:63-67` + `batch_orchestrator.py:62,68` | **有效**：无招标文件时 text_sim 不进风险、矩阵标参考、集团需共享实质段证据 |
| B2 | `document_analysis_svc.py:357` | **有效**：economic_error_similar 改跨文件共享错别字（含 pycorrector 误报治理） |
| B3 | `document_analysis_svc.py:891` + `file_processing.py:795-806` | **有效**：矩阵表头 ".docx" 显示 bug |
| B4 | `stop_words.py` 缺货物/服务类行业词 + `preprocess` 单行 token 化使段落级模板去除失效 | **有效**：建行业词三表 + 段落实质雷同检测器 |

## ③ CROSS-EXAM
- 无争议条目（全部实证确凿）。

## ④ CONFIRM 清单
- 全部批准（用户已逐条确认设计 + 三张行业词表官方来源 + 双层结构 + 混合类型回退工程）。

## ⑤ IMPLEMENT（每 fix 一并落地，统一验证）
- **B1** `batch_orchestrator.py`（RiskScorer 0.30/0.30/0.10/0.30 + template_missing + compute_single_pair/compute_all_pairs 线程化 + detect_gangs 证据验证）、`clearance_engine.py`（_run_cross_comparison template_missing + paragraph_collusion + 注记）
- **B2** `typo_detector.py`（find_shared_typos + 默认置信度 0.70）、`document_analysis_svc.py`（typo 分支改共享错别字）
- **B3** `file_processing.py`（truncate_filename available<1 返回名称开头）、`document_analysis_svc.py`（矩阵 8→20 字符）
- **B4** `data/industry_words/{engineering,goods,services}.txt`（官方源双层）+ `industry_words.py`（懒加载/类型探测/守卫）+ `paragraph_collusion_detector.py`（惊讶度 + 段类型）
- **排版** `document_analysis_svc.py`：标题黑体/正文宋体/表头底色 D9E2F3/封面黑体/标题 16/14/封面 42pt；报告新增 6.9 共享实质段落；封面缺招标文件红字注记
- **测试** `tests/test_collusion_detection.py`（8 项新增）+ 更新 `test_regression.py::test_risk_scorer_new_weights_and_gate`、`test_batch_orchestrator.py::test_snapshot_risk_formula`

## ⑥ DOCS
- CHANGELOG 顶部新增 [2026-09-07] round-004 条目（FIX-2026-09-04-QA-B1/B2，regression 标记）。

## ⑦ PUSH
- 见 commit：<待填>

## ⑧ IMAGE
- 镜像重建 + 容器内抽查：见部署记录。

## ⑨ RE-CHECK / 停跑判定
- 回归 **104/104 passed**（regression + batch_orchestrator + collusion_detection + smoke）、verify_fixes 89/89、check_system 133/137。
- 真实案例复验：段落检测器命中元丰↔物美服务承诺段（match 0.94, surprise 0.67）；无招标文件 text_sim 不进风险；合法同行不触发。
- **无新增 Critical/High → 质量闸门通过。**
