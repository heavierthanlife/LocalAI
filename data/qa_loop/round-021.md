# QA-Loop Round 021 (2026-09-09)

基线: last_head=abbb139 | 模式: full（彻底删除错别字检测系统） | 触发: user（jieba/词典把正确词当错字，每份 300+ 假警）

## ① 根因
- 误报**不是 jieba**（`typo_detector.py` 不用 jieba）；来源是手写 `_BIDDING_CONFUSION_PAIRS`——18 个键里 ~15 个是**正确常用词**（必须/截止/权利/签订/缴纳/期间/形式/权力/制定/定金/截至/其间/订金/交纳/必需），每次出现记一个"警告"；`pycorrector`/`symspellpy` 均未安装（那两层本可增值，实为空）。→ 每份标书数百假警，并污染"跨文件共享错别字"信号与审计"每万字错别字率"。

## ② 决策（用户确认）
1 彻底删除 · 2 审计 typo_detection 一并删 · 3 DB 丢历史 · 4 前端移除入口 · 5 指标 46→45 + 基线刷新（可接受）。**保留** `relationship_extractor` 的独立"相同格式/元数据"启发式与 **jieba**。

## ③ 实现
**后端（子 agent A）**
- 删 `typo_detector.py`、`typo_whitelist.py`
- 指标：`indicator_defs` 删 `economic_error_similar`；`document_analysis_svc` 删权重/cap、`_run_checker` typo 分支、file_scores 加成、指标构建分支
- `runtime_config` 删 7 个 `typo_*` 键
- 审计：`audit.py all_funcs`、`audit_engine`（评分/dispatch/执行/回读）、`audit_report` 标签与扣分分支
- 路由：删 `/check_typos`(batch)、`/admin/typo_results`(admin_ops)
- 图谱：删 `_merge_typo_cross` + 调用 + SELECT
- `batch_orchestrator`：删 `_run_typo_check` + 注册表 + 报告错别字段 + `typo_results` 形参
- DB：删建表/索引/`audit_config` 种子 + 幂等 `DROP TABLE typo_detection_results`
- `requirements.txt` 删 pycorrector/pyspellchecker/symspellpy（保留 jieba）；AGENTS.md 校正

**前端（子 agent B）**：删 `#sidebarTypoResultsBtn` + app.js 绑定与 `renderTypoHistory` + bid-audit.js `typo_detection` 标签/阈值

**收尾（self）**：fix_registry FIX-015 去掉指向已删文件的检查；app.css 去 `#typoDetectionToggle`；测试清理（collusion/admin/batch/audit）；`routes_snapshot.json` 重新生成（383→405，含此前未入快照的 /prompts/* 等）+ `expected_len` 重定；基线 `scores.json` 刷新（46→45，composite 19.0→16.1）

## ④ 验证
- 回归 **130/130**（含新增 route_preservation）· verify_fixes **128/128** · T0 **no_route=0** · node ×2 OK
- `rg typo` 复查：仅剩 `database.py` 的 DROP、`relationship_extractor` 独立启发式、`app.js typographer`(第三方)
- 基线 composite 16.1 <30（正常）

## ⑤ 备注
- route_preservation 此前**不在回归 gate** 内，长期因路由增长而 stale；本轮纳入并重新基线。
- backlog：无（错别字系统已彻底移除）。
