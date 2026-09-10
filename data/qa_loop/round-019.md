# QA-Loop Round 019 (2026-09-09)

基线: last_head=4097b1c | 模式: full（大文件对比：413 修复 + 异步任务） | 触发: user（/batch/plagiarism/compare 413，要所有对比支持数百 MB）

## ① 根因
- nginx `client_max_body_size 12G`（非瓶颈）；**Flask 全局 `MAX_CONTENT_LENGTH=50MB`** 是 413 来源。
- 清标走 `/stream_upload` 预上传（per-request 抬到 11GB + 流式落盘）+ `file_ids` + `file_store.resolve` + 分页提取；而 `/batch/plagiarism/compare` 等 5 个对比端点直接 `request.files` → 撞 50MB。

## ② 实现（范围 A：5 端点全做）
- **`batch_bp.before_request`** 抬 per-request 上限至 11GB（不再受全局 50MB 限制，直传兜底）。
- **共享 `_collect_docs()`**：优先 `file_ids`（`file_store.resolve` → `extract_text_from_path` 分页、内存恒定），回退直传 `files`；`_read_template_text()` 同构。
- 5 端点接入：plagiarism/compare、compare_bidders_quotes、check_quote_anomaly、extract_relationships、check_typos。
- **异步大文件路径**：新增 `POST /batch/plagiarism/run`（file_ids → Celery `plagiarism_task` → TaskBus）+ `GET /batch/plagiarism/status/<task_id>`；`plagiarism/compare` 对 >40MB file_ids 自动转异步（防 web worker OOM）。
- 前端剽窃按钮：有预上传 `file_ids` → 走异步 + 轮询进度；否则直传同步。

## ③ 验证
- 大文件上传：206MB × 2 `/stream_upload` 成功（413 消失）。
- **异步全链路 PASS**：16MB/114k 段 → task completed，verdict=高度相似，cosine=0.802。
- 回归 128/128 · verify 120/120 · T0 no_route=0。

## ④ 发现与限制
- 同步路径在 ~200MB **纯文本**（非真实分布）时 gunicorn worker OOM（已由异步路径 + 自动转异步规避 → web 不再 502）。
- 提示：真实标书"文件几百 MB"多因**图片/扫描件**，提取后的**文本量远小**；异步 + file_ids 已覆盖该场景。纯文本数百 MB 的算法耗时随段数增长，属算法范畴（后续可段落级流式/分块优化）。

## ⑤ backlog
- 其余 4 个对比端点也补异步任务（当前仅 plagiarism 异步；其余支持 file_ids 大上传但同步计算）
- detect_plagiarism 大文本分块/流式优化
