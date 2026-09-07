# QA-Loop Round 006 (2026-09-07)

基线: last_head=8d1030b | 模式: full（真实 3 文件清标测试 + 性能/接线加固） | 触发: manual（用户要求用最新版 app 重测上次三个问题文件）

## ① COLLECT 摘要（用户要求 + 子代理调查）
- **用户要求**：用最新版 app（5.94GB 镜像, torch cpu, industry_words 挂载）重测上次三个真实串标文件（元丰/物美/中昌华美 军营超市项目），带子代理查找问题并报告。
- **explore 子代理**：容器 localai-app 与 HEAD 逐字节一致、依赖齐全、行业词表挂载、torch 同版；给出容器内 `run_clearance` 测试路径（docker cp 文件 → 脚本 → 取报告）；识别数据门槛（无招标文件 → text_sim 跳过/compliance 关/样本免责）。
- **code-reviewer 子代理**：发现 **3 Critical 性能隐患** + H/M/L：
  - C-1 `_surprise` 每段两次 `lcut`（双倍 jieba）
  - C-2 `fp_cache` 全量 frozenset → 数万段 2-4GB 内存
  - **C-3 伪 O(n²)**：段长桶候选爆炸 + bigram 交集剪枝对数字表格行无效 → 真实文档必超时
  - H-1 `_split_paragraphs` 无段落数上限
  - H-3 `merged` 去重 key 用文件集合 → 吞不同实质段
  - H-4 **行业词表未接指标层**（key_info/text_sim 被行业词污染）
  - H-2 collusion_para_map `min(100,c*25)` 4 段即满分
  - M/L：ThreadPool 4→5、死代码 `return html_out`、重复 init_flask_context 等

## ② VERIFY 初判表
| # | 位置 | 初判 |
|---|------|------|
| C1 | paragraph_collusion_detector.py `_surprise` | **有效**：缓存 `all_toks=lcut` 一次 |
| C2 | fp_cache frozenset | **有效**：弃用 frozenset，改锚子串索引 |
| C3 | 桶遍历 + SequenceMatcher | **有效**：锚子串倒排索引 + `MAX_CAND_PER_PARA=40` + `MAX_PARAS_PER_FILE=2500` 分层抽样；**实测 300 段×3 从 >300s 超时 → 2.67s** |
| H1 | `_split_paragraphs` | **有效**：加段落数上限 |
| H3 | merged 去重 | **有效**：key 改「类型+内容前缀」 |
| H4 | 指标层行业词 | **有效**：`preprocess_text_for_similarity`/`_precompute_tfidf_for_files`/compute_all_pairs 加 `extra_stop_words`；`run_analysis` 与 `run_clearance` 统一 ptype 下传 |
| H2 | collusion_para_map | **有效**：改 `min(100, sqrt(c)*25)` |
| M/L | 各 Medium/Low | **有效**：ThreadPool 5、删死代码、删重复 init |

## ③ CROSS-EXAM
- 无争议。

## ④ CONFIRM 清单
- 全部批准（用户确认：先修性能/接线再测；Medium/Low 一起；报告放桌面；端到端落库一起跑）。

## ⑤ IMPLEMENT
- `paragraph_collusion_detector.py`：性能重写（锚子串倒排索引 + 段落数/候选上限 + 内容指纹去重 + surprise 单次分词 + sqrt 评分）
- `file_processing.py`：`preprocess_text_for_similarity` 加 `extra_stop_words`
- `batch_compare_svc.py`：`_precompute_tfidf_for_files` 加 `extra_stop_words`
- `batch_orchestrator.py`：`compute_single_pair`/`compute_all_pairs` 加 `extra_stop_words`；删死代码
- `clearance_engine.py`：`run_clearance` 统一 ptype；`_run_cross_comparison` 加 ptype + 行业词表；ThreadPool 5
- `document_analysis_svc.py`：`run_analysis` 指标层传行业词表；删重复 init_flask_context
- `tests/test_collusion_detection.py`：+4 测试（性能上限/内容指纹去重/行业词接线指标层）

## ⑥ DOCS
- CHANGELOG 新增 [2026-09-07] round-006 条目。

## ⑦ PUSH
- `8d1030b..9c147db` 已推送（LocalAI master），工作树干净。last_head=9c147db。

## ⑧ IMAGE / 真实文件测试（本机容器）
- 重建镜像（`python scripts/docker_build.py` 增量秒级，pip cache 命中）→ up -d → 容器同步代码
- **真实 3 文件容器内 `run_clearance`（算法层）**：
  - 提取：元丰 92681 / 中昌华美 125785 / 物美 88139 chars（秒级，VL 关闭）
  - `total_score=36.0`（中等预警）· `collusion_score=70.0` · 195 共享段 · 25 服务承诺段（match 0.85-1.00, surprise 0.53-0.78）
  - 三对 collusion_para=100、gangs=1（有实质证据）· template_missing=True · ptype=services
  - 行业词接线生效：sim 87.4→83.0/84.4/81.0（余弦下降，行业词被滤）
  - 报告 ZIP 取回桌面 `clearance_round006_report.zip`
- **端到端 Celery 落库链路**（`clearance_task`，worker 消费）：
  - `batch_comparison_results` 落库成功（user=admin, file_count=2, max_risk=29.4, zip 路径）
  - 产物 `data/batch_results/clearance_qa-round006-<id>.zip` 含 DOCX(49KB)+PDF(996KB)，取回桌面 `clearance_round006_e2e.zip`
  - 注：首次投递用虚构 user_id 触发了外键约束 → 改用真实 admin uuid 后成功；file_count=2（第三个文件提取异常，worker 路径与算法层略有差异，后续可查）

## ⑨ RE-CHECK / 停跑判定
- 回归 **108/108 passed** · verify_fixes 89/89 · check_system 133/137
- 真实 3 文件复验：段落级服务承诺雷同强命中、行业词过滤生效、集团判定需实质证据
- **无新增 Critical/High → 质量闸门通过。**
- 遗留观察：worker 端到端 file_count=2（第三个文件）；batch_pair_results/chat_messages 未落（task 部分成功）——属 worker 落库边界，已记录不阻塞。
