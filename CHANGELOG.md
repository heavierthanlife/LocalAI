# Changelog

All notable changes to 中联招标智能助手.

**格式**：Keep a Changelog 风格（Added / Changed / Fixed / Removed），日期降序。
**维护约定**：每次功能升级/修复在顶部新增条目；合规相关改动必须带 `regression: 3/3 baseline passed` 验证（见 `CONTRIBUTING.md` §回归测试）。
**详细迭代记录**：2026-08-28 起的完整工作记录见 `20260827log.md` §10。

---

## [2026-09-09] — 全方位审核修复：死代码清理 / XSS 加固 / 报价信号完整呈现 / 段落证据前端 / 社区持久化（FIX-2026-09-09-018）

### Fixed
- **报价信号丢失**：`cross_tailing_digits`/`cross_progression`（跨投标人）与 per-bidder `tailing_digits_flag`/`progression_type` 此前计算后未持久化。`quote_anomaly_results` 表 +5 列（含幂等 ALTER），`save_quote_anomaly_results` INSERT 18→23 列；HTML 报表加"尾数一致/等比规律"列 + 2 条 cross 提示；管理端历史列表 SELECT 补齐新列
- **铁证证据前端缺失（P0）**：段落级雷同证据此前仅 DOCX 有 6.9 表。新增 `_renderParagraphCollusionEvidence`（`_renderIndicatorsTab` 调用，照抄 DOCX 筛选/排序/截断逻辑，`_clearanceEscape` 全转义，模板段折叠参考）；数据已三路下发前端，后端零改动，live + 聊天重载双路径生效
- **死代码清理**：删除零调用的 `renderDocAnalysisResults`（app.js，80 行）与 `loadAuditHistory`（bid-audit.js，83 行，含 return 后 70 行不可达）
- **XSS 加固**：`renderQuoteAnomalyHistory`/`renderRelationshipHistory`/`renderTypoHistory` 的 `id`/`checked_at`/`task_id` 裸插值补 `escapeHtml`（`suggestions` 原已转义未重复包裹）
- **社区检测持久化**：`relationship_risk_summary.details` 由单存 `company_personnel_map` 改为并入 `communities`（Louvain 团伙分组）；`/admin/relationship_results/<task_id>` 返回 communities；前端新增"团伙"详情按钮 + 社区区块渲染
- **阈值对齐**：bid-audit.js `drop` 默认 0.15→0.30（对齐后端 `quote_anomaly_drop_threshold`）
- **chat.js emoji 统一**：置顶态 `textContent='📌'` → `_icon('📌')`（与其他态一致）

### regression: 119/119 tests passed · verify_fixes 96/96 · check_system 133/137 · node --check ×3 OK
- 契约验证：`_run_cross_comparison` 产出 `paragraph_collusion`（服务承诺段 surprise=0.35/98% 一致）与前端渲染契约匹配
- 管理端历史表新增"本福特/尾数一致/等比规律"三列（Benford >0.15 标黄）

---



## [2026-09-09] — 铁证双层判定：铁证信号独立成硬警报，不再被复合指数稀释（FIX-2026-09-09-017）

### Added
- **铁证双层判定层** `app/services/hard_evidence.py`：铁证信号不参与加权复合指数打分（软嫌疑度指数保留），独立判定层 **veto 只提升展示级别**（`warning_level`），不重写指数
  - **T1 确认级（单命中即 veto → `■ 高度预警（铁证触发）`）**：`lastModifiedBy` 同人（guard 排除 Administrator/User/微软用户/lenovo 等通用值）、平台加密锁/文件码雷同（仅交易平台来源）、联系人+电话同组双命中、段落同对 ≥2 段或 1 段 ≥3 家共享
  - **T2 强嫌疑（需 ≥2 类共证才 veto）**：author 雷同（guard）、上传/解密 IP 同、段落单段共享
- **暗标违规独立轨道**：`tech_seal` 泄露（单家违规非串通证据）触发 `■ 高度预警（暗标违规）`，不进串通铁证；铁证与违规可各自独立出现，同时触发时串标优先、违规附加
- **报告接入**：`run_analysis` basic_info 新增 `hard_alarm`/`hard_label`/`hard_evidence`；`run_clearance` 最终出口补入横向层段落雷同后终判；DOCX 封面铁证/违规红色警示段 + 预警单位 `★`（hard_flag）；`suspected_units.hard_flag`
- **前端**：`app.js` 清标结果按 `hard_alarm`/`warning_level` 联合着色（修"绿分+红字"矛盾），历史列表 `★铁证` 红标

### Fixed
- 铁证被加权稀释：`_weighted_total_score` 分母含所有非 skip 指标，铁证权重最高 0.10 → 单铁证实际贡献 ~5 分；段落级逐字雷同（最强证据）此前不计入复合指数，封面与证据脱节

### regression: 119/119 tests passed · verify_fixes 96/96 · check_system 133/137 · app.js node --check OK
- 新增 4 回归测试：lastModifiedBy 铁证升级（指数不被改写）/ guard 反例（Administrator 不触发）/ T2 双类共证 / 暗标违规独立触发
- **真实 3 文件复测 PASS**：元丰+中昌华美 `lastModifiedBy='超彩赵'`（物美='唯一的麦麦儿'）→ T1 veto `fired=True label='■ 高度预警（铁证触发）'`
- 复合指数/基线 scores.json 19.0/DB max_risk 不变（veto 只升展示级，历史可比性保留）

---

## [2026-09-08] — 46 项指标语义错配系统修复：联系人/关系/暗标/投标数（FIX-2026-09-07-QA-C4）

### Fixed
- **联系人雷同误报**（用户报告 3.1.6）：`contact_person_same`/`cross_contact_same`/`contact_phone_abnormal` 不再用 key_info 关键词重合冒充，新增 `app/services/contact_extractor.py` 从投标文件正文提取真实联系人/手机号/邮箱并跨文件比对；无联系人数据落"○ 需开标信息表/联系人数据"占位（score=0）
- **关系指标误报**：`bidder_agent_contact`/`expert_bidder_closeness` 无代理/评委名单时改真 skip（此前误跑通用关系报告给 28.5 分，文案自相矛盾）
- **暗标检测空壳**：`tech_seal_check` 不再用错别字检测冒充，新增 `app/services/tech_seal_detector.py` 真实检测（技术方案段公司名/印章提示/大量人员姓名→泄露；校准避免自指称谓误判）
- **投标数漏报**：`bidder_count_abnormal` 本地 `n<3` 触发（此前被误标 skip 吞掉）
- **行业词污染关键词**：`extract_keywords`/`keyword_overlap_similarity`/`build_key_info_matches` 接入行业词表
- **指标去重/语义**：`same_machine_code`→"文件作者/编制人雷同"改名、`cross_machine_code`→skip；`cross_contact_same` 加权去重；专家指标占位保持 skipped 不压低指数；`_weighted_total_score` docstring 与实现一致
- **平台列激活**：`clearance_openinfo` 放开 IP/文件码/加密锁列映射 + `_platform_signals`（开标表含平台列且跨单位重复时激活对应指标）；`extract_metadata` docx 补 creator/producer；`suspected_units` 补 lasteditor 组加成

### regression: 115/115 tests passed · verify_fixes 89/89 · check_system 133/137
- 基线 scores.json 刷新（contact 0.5→0、relationship 28.5→skip、tech_seal 空壳→真检测、bidder_count skip→触发），composite 19.0 正常
- 新增 4 回归测试：联系人真实比对/无数据占位/暗标泄露与非泄露/投标数本地触发

---



## [2026-09-08] — 清标证据链提纯：模板段排除 + 元数据硬信号 + 6.9 铁证优先（FIX-2026-09-07-QA-C3）

### Fixed
- **6.9「共享实质段落」信号纯度**：投标函/声明模板段（"一、按照招标文件要求提交投标文件正本1份…"等套招标模板的合规内容）不再当串标证据——`_LEGAL_TERMS` 扩充（正本/副本/有效期/声明/真实有效/待命/供应/腐烂/变质…）+ 新增「投标函声明段」段落类型 + 门槛修正（legal>=2/tech>=2）；**真实 3 文件复测 shared_segments 195→36**
- **集团判定过度**：`detect_gangs` 门槛从"≥1 共享段"提至"**≥2 非模板实质段**"（`evidence_counts` + `min_evidence=2`）——只共享投标函模板的正常同行不再误判集团
- **元数据硬信号纳入**：`extract_metadata` 补 `cp:lastModifiedBy` 提取 + 新指标 `file_attr_lasteditor_same`（触发指标, 权重 0.10, cap 30）——"同一最后编辑人做两家标书"是比文本更硬的串标信号
- **6.9 呈现**：铁证优先（只列服务承诺/技术方案段，按惊讶度降序，预览加长 100 字）+ **6.9.2 模板/声明段折叠**（灰字参考，标注"非串标证据"）

### Changed
- `collusion_score`/`per_pair_count` 只统计非模板实质段（模板段不再污染评分与集团证据）

### regression: 111/111 tests passed · verify_fixes 89/89 · check_system 133/137
- 真实 3 文件复测：服务承诺段铁证清晰（应急响应"2分钟/10分钟"数字微调、食堂氛围、食材新鲜、应急预案块）；元数据「超彩赵」（元丰+中昌华美）分组触发；6.9 铁证优先 + 模板折叠渲染
- 新增 3 回归测试：投标函声明段排除 / last_modified_by 提取 / lasteditor 指标触发

---



## [2026-09-07] — 真实文件清标测试：段落检测器性能加固 + 行业词表接线（FIX-2026-09-07-QA-C2）

### Fixed
- **段落级实质雷同检测 O(n²) 性能爆炸**（实测 300 段×3 文件 >300s 超时）：锚子串倒排索引替代段长桶遍历 + `MAX_CAND_PER_PARA=40` 候选上限 + `MAX_PARAS_PER_FILE=2500` 分层抽样 → 300 段 2.67s / 1000 段 8.96s / 2500 段 28.9s；**真实 3 文件（~3500 段）1.6s**
- **内存：弃全量 bigram frozenset 指纹**（数万段可达 2-4GB）
- **`merged` 去重 key 用文件集合 → 吞不同实质段**：改「段落类型 + 内容前缀」指纹，同一实质段跨 3 家正确合并
- **行业词表未接指标层**：`preprocess_text_for_similarity`/`_precompute_tfidf_for_files`/`compute_all_pairs` 加 `extra_stop_words`；`run_analysis`/`run_clearance` 统一 ptype 下传 → 军营超市行业词（超市/收银/理货）不再抬高 key_info/余弦（sim 87.4→83.0 等）
- ThreadPool 4→5、删 `batch_orchestrator.py` 死代码、删 `run_analysis_async` 重复 `init_flask_context`

### Changed
- `collusion_para_map` 评分 `min(100, c*25)` → `min(100, sqrt(c)*25)`（避免 4 段即满分）

### regression: 108/108 tests passed · verify_fixes 89/89 · check_system 133/137
- **真实 3 文件容器内测试**（元丰/物美/中昌华美 军营超市项目，已证实串标）：
  - 算法层：`total_score=36 中等预警`、`collusion_score=70`、195 共享段、25 服务承诺段（match 0.85-1.00, surprise 0.53-0.78）、三对 collusion_para=100、集团 1（有实质证据）
  - 端到端 Celery：`batch_comparison_results` 落库成功，产物 DOCX+PDF ZIP 取回桌面
- 新增 4 回归测试：性能上限 / 内容指纹去重 / 行业词接线（preprocess + run_analysis）

---



## [2026-09-07] — Docker 构建加速：GPU 自动检测 + torch CPU/CUDA 分流（FIX-2026-09-07-QA-C1）

### Fixed
- **Docker build 从 1.5-2 小时降到 ~90 秒**：根因是 `torch==2.12.1` 在 Linux PyPI/清华源解析为 **CUDA 版**（拖入 nvidia-cublas 423MB/cuda-toolkit/nvidia-\* 全家桶，pip 层 6.69GB），而运行时**纯 CPU**（代码零 `.cuda()`、easyocr 日志实证 "Using CPU"、compose 不分配 GPU）
- **两机同仓 GPU 智能分流**：新增 `scripts/docker_build.py`——宿主 `nvidia-smi` 检测 → 无 GPU 装 `download.pytorch.org/whl/cpu`（`torch 2.12.1+cpu`），有 GPU 装 `.../whl/cu124`（`TORCH_CUDA_INDEX` 可覆盖）；Dockerfile `ARG TORCH_INDEX` 默认 CPU
- **构建提速基建**：apt/pip 全改 BuildKit `--mount=type=cache`（跨构建复用下载）；torch/torchvision 独立一层利于缓存；`.dockerignore` 收紧（上下文 185MB→~10MB，排除 `local_cache/`95MB/`.opencode/`55MB/`tools/`22MB 等）
- **堵密钥泄入镜像**：`.dockerignore` 补 `.env`（文件，此前只排了 `.env/` 目录）
- **新增 `docker-compose.gpu.yml`**：app/celery-worker GPU 设备保留（GPU 机 `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`）
- **修复行业词表被 `app_data` 卷遮蔽**：compose 三服务加 `./data/industry_words:/app/data/industry_words:ro` 只读 bind 挂载

### Changed
- 镜像体积 **12.8GB → 5.94GB**（砍掉 CUDA 载荷）；容器内 `torch 2.12.1+cpu`、`cuda_available: False`

### regression: 104/104 tests passed · verify_fixes 89/89 · check_system 133/137
- 本机 CPU 分支实测：build ~90s、站点 200、industry_words 挂载生效、容器内 4 项修复标记命中
- 注：GPU 机（RTX 2080 Super, Turing sm_75）需真机验证 cu124 兼容性；不兼容则改 `TORCH_CUDA_INDEX`

---



## [2026-09-07] — QA-Loop round-004：清标算法可靠性重构（段落级实质雷同）+ 字体排版

### Added
- **段落级实质雷同检测**（`app/services/paragraph_collusion_detector.py`，FIX-2026-09-04-QA-B1）：在原始文本上分段，两两 `SequenceMatcher ≥0.85` 找近逐字雷同段，按**惊讶度**（段内非行业/非模板词占比）过滤样板段，仅"服务承诺段/技术方案段"等实质内容的跨文件雷同算围标信号
- **行业词三表**（`data/industry_words/{engineering,goods,services}.txt` + `app/services/industry_words.py`）：双层（国家标准词层：财政部《政府采购品目分类目录 2022》+ 住建部《建筑业企业资质标准》；常见运营词层含超市经营词），采购类型探测（`get_procurement_type`），守卫测试锁定评价性措辞（热情/周到/细致/尽职 等）不入表
- **跨文件共享错别字**（`typo_detector.find_shared_typos`，FIX-2026-09-04-QA-B2）：仅跨 ≥2 家逐字相同的 suspect_text 计分，白名单+行业词排除，天然消除 pycorrector 随机误报
- 报告新增 **6.9、共享实质段落** 章节（段落类型/共享单位/一致率/惊讶度/内容预览）+ 表头底色 + 标题黑体排版

### Changed
- `RiskScorer` 权重重构（`batch_orchestrator.py`）：`text_sim 0.25→0.10`（整篇余弦是信号平均器）、新增 `collusion_para 0.30`（段落级实质雷同为主信号）
- `clearance_engine._run_cross_comparison`：无招标文件时 `template_missing=True` → text_sim 对 risk 零贡献（与指标层 skip 对齐），原始余弦仍保留在 6.2 矩阵标注"仅参考"
- `detect_gangs`：集团必须含 ≥1 对共享实质段（纯模板/行业重叠不再判集团）
- `economic_error_similar`：由"错别字总数"改为"跨文件共享错别字数"
- 封面增加"缺招标文件：文本/关键词类指标未计入，冒烟指数为下限估计"红字标注

### Fixed
- 矩阵表头显示纯 ".docx"（`truncate_filename(fname, 8)` 对中文长名退化为扩展名）→ 20 字符 + `file_processing.py` `available<1` 返回名称开头而非纯扩展名
- `build_attr_details`/`compute_single_pair` 对缺失 `metadata`/`images` 键防御

### regression: 104/104 tests passed · verify_fixes 89/89 · check_system 133/137
- 3 家已证实串标案例（元丰/物美/中昌华美 军营超市项目）：元丰↔物美 服务承诺段近逐字雷同（match 0.94, surprise 0.67）被段落检测器命中；无招标文件时 text_sim 不进风险；合法同行（行业词重叠但独特段不同）不触发
- 基线快照（工程类 价格标 vs 商务技术标）复合指数保持 <30 正常区间

---



## [2026-09-04] — QA-Loop 基础设施：九阶段流程固化（文档同步 + 镜像重建）

### Added
- `qa-loop` 三方自动循环升级为**九阶段**：`COLLECT → VERIFY → CROSS-EXAM → CONFIRM → IMPLEMENT → DOCS → PUSH → IMAGE → RE-CHECK`
  - **⑥ DOCS**：每轮代码变更后，`CHANGELOG.md` 按 FIX 编号细分条目 + `AGENTS.md` 约定同步 + `round-NNN.md`，随代码同 commit（PUSH 前）
  - **⑧ IMAGE**：项目 `has_docker:true` 时 `docker compose build` + `up -d` + 健康检查 + 容器内代码抽查，保证镜像 = HEAD
- 全局 skill（`shared-agent-infra/skills/qa-loop`）+ 全局 command `/qa-loop` + 项目参数（`.opencode/qa-loop.project.md`）同步九阶段
- 维护约定：每轮更新完毕后必须同步文档并重建镜像，保证「仓库 HEAD、文档、运行镜像」三者一致

### round-003（增量无代码变更）
- `9205efe..HEAD` 增量 diff 仅含 docs → 质量闸门通过，直接收尾（commit `50fcce6`）

## [2026-09-04] — QA-Loop round-002：安全纵深补强（D1–D3）

### Fixed
- `FIX-2026-09-04-QA-D1` `chat.js` — `COMPARE_REPORT` 分支直接 `innerHTML=htmlContent` 绕过 DOMPurify（与 C8 纵深不一致）→ 补 `_safeHTML()` 消毒（commit `3f7483f`）
- `FIX-2026-09-04-QA-D2` `app.js` — 三处错误消息（`data.error`/`err.message`/`e.message`）未转义拼 innerHTML → 统一 `escapeHtml()`（commit `498f8b1`）
- `FIX-2026-09-04-QA-D3` `knowledge.py` — `skill_hash` 变量重赋值遮蔽 → `kb_file_hash` + check_system 守卫 + C5 守卫正则修正（commit `9205efe`）

### Changed
- check_system 守卫增至 133/137（新增「skill_hash not shadowed by kb hash」）；tests/test_regression.py 全量 exit=0

## [2026-09-03] — QA-Loop round-001：代码层 11 项修复 + 基础设施固化

### Added
- **QA-Loop 三方自动循环基础设施**（commit `4b178c3`）：全局 skill + 全局 command `/qa-loop` + 项目参数 `.opencode/qa-loop.project.md`；`data/qa_loop/` 基线 + round-001 审计记录

### Fixed
- `FIX-2026-09-03-QA-C1` `knowledge.py` — 空路由装饰器致 `/admin/all_user_kb`(GET) 误绑 `generate_work_report` → 恢复真实端点 + check_system「No orphaned route decorators」守卫（commit `c85fc34`）
- `FIX-2026-09-03-QA-C2` `knowledge.py` — `generate_project_file_skill` IDOR（无项目成员校验）→ 补 `get_user_role_in_project` 成员校验 + 守卫（commit `1cb94dc`）
- `FIX-2026-09-03-QA-C3` `admin.py` — `admin_required`/`auditor_required` 缺失 consent+user_id 会话校验 → 抽公共 `_check_session_valid()` + 守卫（commit `85775f2`）
- `FIX-2026-09-03-QA-C4` `credit.py` — 限速器进程内存 dict 跨 gunicorn worker 失效 → 迁 Redis `credit_rate:{ip}` INCR+TTL + 内存降级 + 2 回归测试 + 守卫（commit `9eecab3`）
- `FIX-2026-09-03-QA-C5` `knowledge.py` — 工作报告 zip 文件句柄泄漏（匿名 `open().read()`）→ `with open` + 守卫（commit `27be656`）
- `FIX-2026-09-03-QA-C6` `app.js` — `checkStorage` warning message 未转义拼 innerHTML → `escapeHtml()`（commit `df6327e`）
- `FIX-2026-09-03-QA-C7` `knowledge.py` — work_report user filter 用 `str.replace('cs.user_id')` 后处理脆弱 → 结构化构建两套 filter + 守卫（commit `7393161`）
- `FIX-2026-09-03-QA-C8` `chat.js` — `md.render()` 输出未过 DOMPurify（15 处）→ 统一 `_renderMarkdown()` 消毒 + 助手（commit `7489653`）
- `FIX-2026-09-03-QA-C9` `file_processing.py` — 文本提取乱码/控制字符（U+FFFD/C0/C1/代理）→ `clean_extracted_text()` + `_CONTROL_FILTER`（commit `c153ee3`）
- `FIX-2026-09-03-QA-C10` `app.js` — 时间线空态无操作引导 → 加「选择项目」按钮（commit `93c9585`）
- `FIX-2026-09-03-QA-C11` `app.js`/`review.js`/`icons.js` — 统计区图标 emoji 混排/方块 → 统一 Material Symbols + `🟢→monitoring` 映射（commit `fb91b59`）

### Changed
- `scripts/check_system.py` 新增多项回归守卫（孤立装饰器/成员校验/会话校验/Redis 限速/zip 句柄/结构化 filter/变量遮蔽）
- `tests/test_regression.py` 新增 credit 限速 Redis 回归测试 ×2
- 基线 `data/qa_loop/last_head` 推进；round-001/002/003 记录在 `data/qa_loop/`

---

## [2026-09-01] — 文档现实对齐更正 + RTX 2080 Super 支持评估

### 文档管线事实更正
- 经代码核验，文档声称的「RapidOCR + MinerU」与代码不符。**实际管线 = MarkItDown 0.1.6 + LibreOffice/soffice + EasyOCR 1.7.2 + PyMuPDF (fitz)**，无 MinerU/RapidOCR，无 rarfile/py7zr/ebooklib/extract-msg
- 更正：`CHANGELOG.md` 07-04/07-08、`DECISIONS.md`、`README.md`、`MANIFEST.md`、`ARCHITECTURE.md`、`IMPROVEMENTS_SKIPPED.md`
- `IMPROVEMENTS_SKIPPED.md` 逐项按真实基线重估；#8 admin.py 拆分标记「已解决」

### RTX 2080 Super 支持评估
- 发现 torch 为 **CPU 版**（`2.12.1+cpu`，CUDA 不可用）→ GPU 完全未启用
- 2080S (8GB) 最大价值：**EasyOCR GPU 加速 + LoRA 微调 Qwen2.5-7B**（Unsloth QLoRA 4bit），均只需重装 CUDA torch 无代码改动
- `ocr.py` 的 `OCR_GPU=auto` 已支持自动探测

---

## [2026-09-01] — 清标评分系列（FIX-010 ~ FIX-014）

### FIX-014 自适应招标高频词 + 异组件文本相似度守卫（方案 Y + Z-1）
- **方案 Y** `file_processing.py`：招标文件高频词并入停用集从固定 top-50 改为自适应 `k = min(200, max(50, len//500))`，并加 **TF≥2 守卫**（不误杀 TF=1 的独特技术参数）
- **方案 Z-1** `batch_orchestrator.py`：新增 `_detect_component`（价格标/技术标/商务标/unknown，双 unknown fail-safe）；异组件 pair（如价格标↔技术标）`text_sim`/`key_sim` 归零 + 标记 `component_mismatch` + risk 重算
- **报告批注** `document_analysis_svc.py`：6.6 明细异组件对显示「异组件(不计)」+ 图例说明
- **实证**（真实 EPCM 招标 49216 字, k=98）：同组件正常技术标 0.89→**0.7695** <0.80 门槛；围标（技术雷同）**0.9784** 仍触发
- 回归 +4 测试（adaptive / tf_guard / component_mismatch / component_same）；全量 92/92；fix_registry 70/70

### FIX-013 中文停用词过滤 — 消除模板重叠误报
- 新增 `stop_words.py` `DEFAULT_STOP_WORDS`（~150 招投标/功能词）
- `tokenize_for_tfidf` 默认过滤；三条向量化路径统一接入
- **实证**：围标（技术雷同）cosine 0.98 vs 正常（技术不同）0.74，≥80% 门槛可判别
- 回归 88/88；fix_registry 66/66

### FIX-012 清标基线校准
- 工程类 2 投标人脱敏基线（`tests/fixtures/clearance_baseline/`）
- 无招标文件时 `text_sim` 指标**跳过**（模板去除不可用）
- 下调易误报指标权重；`test_clearance_baseline_scores` 锁快照防漂移；报告样本量免责（N<5）
- 回归 86/86；fix_registry 62/62

### FIX-011 报价尾数检测
- 尾数相同≥80% 检测（CSDN 第一信号）+ `extract_prices` 修复（_CN_PRICE 虚假 10000；双路去重）

### FIX-010 清标评分计量升级
- `total_score` 从裸加总（max~209）改为 **0-100 权重复合指数**（45 项 `INDICATOR_WEIGHTS` + score cap + text_sim 三指标去重）
- RiskScorer → 0.375 key + 0.375 attr + 0.25 text（去死图片权重）；text_sim ≥80% 门槛
- 预警阈值统一：≥60 高度 / ≥30 中等 / <30 正常
- 行业信号：报价等差/等比（`_detect_progression`）、Benford Nigrini 分级+卡方+Z、score_analyzer（Grubbs/Kendall W/Spearman）、关系社区检测、law_semantic 接入
- 回归 80/80；fix_registry 57/57

---

## [2026-08-28] — 清标结果移入聊天 + 报告生产级升级

### FIX-009 清标报告生产级升级
- **三节 continue 死代码修复**：`document_analysis_svc.py:494` 使整段渲染成死代码 → 45 项指标 6 行表全渲染
- **文本相似度恒 0 修复**：`_precompute_tfidf_for_files` 从未在清标路径调用 → 现预计算 + `tender_text` 模板去除
- **涉及指标数量恒 1**：`int(triggered_count>0)` 布尔 bug → 按指标 details 引用计数
- **开标信息表 + 评审标准**：新增 `clearance_openinfo.py`（Excel/CSV/JSON + 评审标准提取）→ 激活 14 指标
- 前端开标表上传 + `/clearance/preview_criteria` 预览可编辑
- 评分合理化：`keyword_overlap` 对 <4 关键词的短/模板文本返回低值
- 路由 372（新增 preview_criteria）；回归 72/72

### FIX-006 清标结果移入聊天
- 移除工具栏结果区 → 结果渲染进聊天（10 节可折叠富 HTML，含热力矩阵）
- `<!-- CLEARANCE_REPORT -->` + JSON 落库 `chat_messages`；重载时从 JSON 重建富 HTML
- 线程定向、下载入口统一到聊天气泡

### 全量审计合并入清标
- `audit_bp` 从 `register_blueprint()` 摘除，功能合并入清标 5 维度；`audit_runs`/`audit_config`/`audit_file_results` 表保留供 graph.py/cases.py 依赖

### FIX-005 Prompt 体系优化
- JUDGE_PROMPT / STRUCTURED_PROMPT 英文→中文；主 agent prompt「中联招标智能助手」
- 修复 `data/agent_prompt.json` 残留 `{"prompt":"Test prompt"}` 覆盖 bug
- 双 guard 去重、死代码清理

---

## [2026-08-28] — 安全加固 + 路由拆分 + 存储迁移

### 安全修复
- **C5** graph API 加 `@login_required` + 项目成员检查（FIX-2026-08-28-001）
- **C6** 管理员默认 PIN 生产 fail-closed（FIX-2026-08-28-002）；开发保留默认+告警
- **M3** 匿名存储迁移 PG JSONB `anon_chat_messages`，原子 UPSERT（FIX-2026-08-28-004）
- **M4** `credit_tasks` 内存共享 → Redis 注册表，跨 worker 可用（FIX-2026-08-28-003）

### 路由拆分（C1/C2/C3）
- `admin.py` 4,820→1,653 行（admin_regeneration / admin_knowledge_lab / admin_ops 子模块）
- `chat.py` 2,025→1,122 行（chat_files / chat_sessions / chat_config）
- `knowledge.py` 2,018→941 行（knowledge_notebook/company_kb/style/ingest/training + shared）
- 路由守护测试基线 expected_len=382

### 其他
- 二进制/私钥出库：`cert/key.pem`、`msedgedriver.exe` 移除跟踪
- `.gitignore` `*.json` 全局排除 → 定向规则 + 18 个跟踪 JSON 保留

---

## [2026-07-16] — Wiki 修复 + 建议引擎 f-string 修复

### Fixed
- `suggestion_engine.py`：4 处 f-string 语法错误（双引号闭合导致 `{name}` 变 set literal）→ `/timeline/:id/suggestions` 500 根因
- `app.js`：Wiki 编辑/删除按钮静默失败 → `data-edit-path`/`data-delete-path` 属性 + 单委托监听器

### Test
- 10/10 smoke + regression 通过；25/28 integration 因 hermes venv 缺 flask_limiter 失败（预存环境问题）

---

## [2026-07-15] — Timeline + Wiki 数据契约修复（app.js ~230 行）

### Added
- **Phase 8 Timeline Tab**：`allPanels`/`tabMap` 注册、时间线加载、里程碑表（planned/actual/diff）、状态徽章、HTML ~40 行
- 项目招标字段（bidding_category/bid_method）数据流贯通（modal/项目表/项目头）
- 文件状态列、版本历史状态切换（"设为正式"/"设为草稿"）

### Fixed
- Wiki Tab 数据契约：`statsData.data` → `statsData.stats`，`indexData.data.pages` → `indexData.pages`（5 处）
- 回收站侧边栏按钮缺 body/headers → 补上
- 流式消息重复：SSE done 更新 `_pollLastId`/`_lastKnownMessageId`；补工具栏/反馈/操作按钮
- 配置清空管理员可见性即时生效

---

## [2026-07-13] — 聊天渲染竞态 + Admin DB 429 缓解

### Fixed
- **Chat Render Race**（app.js 4 处）：sidebar onclick 异步 + `await loadSession()`；面板可见性守卫（隐藏→强制重载，可见→跳过）；`innerHTML=''` 前移；`isLoadingSession` 守卫 toast
- **Admin DB 429**：移除前端逐表 fallback 循环（消除 47 请求突发）→ 服务端 try/except + `COUNT(*)` fallback 单请求

### Test
- Smoke 6/6；Unit 103/103

---

## [2026-07-11] — LLM 自动 Fallback 链 + 全蓝图测试覆盖

### Added
- **`llm_fallback.py`**：7 步 fallback 引擎 + 熔断器（DEFAULT_CHAIN、degraded 检查、指数退避 cap 300s、thread-local 活跃 provider）
- `create_chat_model()` 接入 fallback；流式重启：服务器 `fallback_retry` SSE + 客户端重发（max 3）
- Runtime config：`llm_fallback_enabled`/`llm_fallback_chain`/`llm_fallback_cooldown_seconds`
- Admin UI 拖拽排序 fallback 链（provider+model）
- Nemotron 模型加入 NVIDIA provider
- **11 个蓝图全部有集成测试**（`tests/integration/`）

### Test
- Smoke 6/6；Unit 52/52（+14）；Integration(db) 45/45（+17）

---

## [2026-07-09] — 分类感知提取系统

### Added
- `CATEGORY_CONFIG`：每分类信号集/章节标题/文件名前缀 `[分类]name_skill.md`/`Category:` 头
- `category` 列入 `knowledge_lab_files` 和 `project_files`；贯穿上传/generate_skill 端点
- RAG 分类过滤（`retrieve()` 接受 `categories` → ChromaDB `$in`）
- skill_auditor 分类感知去重；上传超驰检测（`_check_skill_overlap()` + 合并建议对话框）
- Skill 编译器（DBSCAN 每分类主题聚类 + 复合 skill）；模板→文档生成器（`template_renderer.py`）

### Fixed
- JS TDZ bug：`syncActiveTabWithView()` try/catch、`pinnedSessions` 提升到文件级
- 移除损坏的 update hook；`loadSidebarDb()` 50ms 节流降 429

---

## [2026-07-08] — 统一文件管线 + 首次全审计

### Added
- **统一文件处理管线**：单一 `FILE_TYPE_REGISTRY`（44 类型）、分层提取（MarkItDown→格式特定→OCR→LibreOffice）、`allowed_file` 校验
  - **2026-09-01 更正**：实际无 MinerU 层；rarfile/py7zr/ebooklib/extract-msg 4 个依赖**未加入 requirements**（归档/电子书仍不受支持）

### Fixed（首次全审计 139 文件 / 5 发现）
- **HIGH** bare `except: pass` → `except OSError` + logging（admin cleanup）
- **MEDIUM** f-string SQL → 表/列白名单校验（admin.py, rag_engine.py, recycle_bin_service.py, skill_auditor.py）
- **LOW** 45+ 宽 `except Exception:` → 补 logger；knowledge.py f-string WHERE 消除
- **Critical 运维 bug**：过期系统 Python 进程占用 :5443 携带旧 ALLOWED_EXTENSIONS → taskkill 终止

---

## [2026-07-07] — NVIDIA LLM/VL 提供商 + 上传限制修复

### Added
- `ChatNVIDIA` 提供商（`langchain_nvidia_ai_endpoints`），模型 `z-ai/glm-5.2` → `moonshotai/kimi-k2.6`
- 多提供商 VL 模型（NVIDIA + SiliconFlow），VL 管理 UI（状态横幅/配置/测试）
- `POST /set_video_analysis` 端点 + 视频分析复选框

### Fixed
- SSE 重复守卫：NVIDIA 重发完整文本 → `full_text.startswith(chunk)` 去重
- 413 → `MAX_CONTENT_LENGTH` 500MB
- `g._streaming_agent` 缓存失效；`split_thinking_answer` 支持 6 种格式（双花括号 JSON 等）
- `audit_report.py` NameError → 导入提升到模块顶层

---

## [2026-07-06] — Cross-device sync + unread tracking

### Added
- `GET /chat/poll/<thread_id>?since_id=N` — lightweight delta-fetch for new messages
- Unified real-time polling: common chats (5s) and project chats (3s)
- Per-browser unread badges on all sidebar threads via `localStorage` (`zlai_read_<thread_id>`)
- Unread count clears on scroll-to-bottom, debounced at 800ms
- `last_msg_id` field in `get_user_sessions()` response for unread calculation

### Changed
- Project chat polling now uses `/chat/poll` instead of `/admin/.../ai_activity` (delta vs full reload)

---

## [2026-07-06] — Mobile responsiveness (3-tier)

### Added
- Phone breakpoint (<640px): sidebar overlay, tab "更多" dropdown, fixed input, safe-area support
- Tablet breakpoint (640–1024px): sidebar narrowed to 180px, adjusted font sizes
- `font-size: 16px` on consent modal inputs to prevent iOS auto-zoom
- Swipe-to-close gesture on sidebar overlay
- Touch targets min 44px across all interactive elements

### Changed
- Sidebar breakpoint refactored: 768px → 640px (phone) + 1024px (tablet)
- Admin panels: tables get `overflow-x: auto`, secondary columns hidden on phone
- Knowledge lab: 2-column grid stacks to 1-column on phone

---

## [2026-07-06] — is_grilling query blind spots (7 fixes)

### Fixed
- `backfill_project_chat`: SQL now excludes grilling threads (`is_grilling = FALSE`)
- `update_project`: title sync skips grilling threads
- `add_project_member`: auto-backfill check excludes grilling threads
- `generate` endpoint: backfill queries exclude grilling threads (2 locations)
- `project_ai_activity`: excludes grilling thread messages
- `project_unread_count`: excludes grilling thread messages
- Frontend: 3 `find(s => s.project_id == ...)` calls now filter `!s.is_grilling`

---

## [2026-07-04] — Skills audit + AI document review + API format + tests

### Added
- `POST /admin/review/document` — AI five-axis document review (code-review-and-quality skill)
- "🤖 AI 文档审查" panel in Review tab with axis checkboxes and result table
- `ok()` and `err()` unified API response helpers in `app/utils/helpers.py`
- 6 pytest smoke tests in `tests/test_smoke.py`
- `pytest.ini` configuration
- `IMPROVEMENTS_SKIPPED.md` — 9 deferred improvements with rationale

### Changed
- Red Team endpoints now use `ok()`/`err()` unified format
- `IMPROVEMENTS_SKIPPED.md` records all skipped upgrades with timestamps

---

## [2026-07-04] — Document pipeline upgrade: EasyOCR → RapidOCR + MinerU

> **⚠️ 2026-09-01 更正**：经代码核验，该升级**未实际落地为 RapidOCR/MinerU**。当前真实管线为 **MarkItDown + LibreOffice + PyMuPDF + EasyOCR**。以下为历史计划记录，供追溯；实际状态见 `IMPROVEMENTS_SKIPPED.md` §现实对齐。

### Added (原计划)
- MinerU (`_try_mineru`, `_strip_markdown`) as primary PDF/DOCX/PPTX/XLSX parser in `file_processing.py`
- `_ocr_pdf_legacy` fallback in `ingest_pipeline.py`

### Changed (实际落地)
- `app/services/ocr.py`: 仍为 **EasyOCR** 1.7.2（含 `OCR_GPU=auto` 探测，GPU 可用时自动启用）
- `file_processing.py`: 结构化提取用 **MarkItDown** 0.1.6；`.doc` 转换用 **LibreOffice/soffice**；PDF 流式提取用 **PyMuPDF (fitz)**
- `requirements.txt`: 无 rapidocr/mineru；rarfile/py7zr/ebooklib/extract-msg 未安装

### Removed
- `easyocr` from `requirements.txt` → **未移除**（EasyOCR 仍在用，`ocr.py` 依赖）

---

## [2026-07-03] — Red Team (质问模式) frontend completion

### Added
- "🔥 质问模式" button in chat sidebar + "🔥 质问" button in project tabs
- `_isCurrentSessionGrill` flag and red banner in chat area
- `is_grilling` field in `get_user_sessions()` response
- 🔥 prefix on grill threads in sidebar

### Fixed
- `/send_stream`, `/send`, `/regenerate` now actually use `get_redteam_agent()` instead of just swapping prompt
- `summary` CSS: replaced `display: inline-block` with custom ▶ collapse indicator
- `.token-control` and `.action-group` missing `display: flex` restored
- Chat toolbar restructured: 4 detection features + prompt templates moved into collapsible section

---

## [2026-07-03] — Initial audit (prior assistant handoff)

### Verified
- `is_grilling BOOLEAN DEFAULT FALSE` in `chat_sessions` table
- `redteam_agent.py` with `REDTEAM_SYSTEM_PROMPT` and `get_redteam_agent()`
- `/api/chat/create_grill_thread` and `/api/projects/<id>/get_or_create_grill_thread` endpoints

### Found broken
- Red Team agent never invoked (only prompt swap)
- Frontend HTML/JS completely missing (0% done)
- CSS flex containers and collapse indicators missing
