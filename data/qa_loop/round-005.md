# QA-Loop Round 005 (2026-09-07)

基线: last_head=a292a36 | 模式: full（用户定向：Docker build 过慢 + 两机 GPU 智能分流） | 触发: manual

## ① COLLECT 摘要（用户反馈 + 实证）
- **用户反馈**：Docker 镜像 build 太慢；本机无 GPU、另一台 RTX 2080 Super 用同一 git，需智能决定是否下载 CUDA。
- **实证（build 日志 + docker history）**：
  - `torch==2.12.1` 从清华/PyPI 在 Linux 解析为 **CUDA 版**，拖入 `nvidia-cublas`(423MB)/`cuda-toolkit`/`nvidia-*` 全家桶；镜像 pip 层达 **6.69GB**，build 卡在下载 1-2MB/s 龟速，20+ 分钟仍在 Collecting。
  - **运行时 100% CPU**：代码零 `.cuda()`；easyocr 日志实证 "Using CPU"；compose 未分配 GPU；本机 `.venv` 即 `torch 2.12.1+cpu`。
  - `--no-cache-dir` → 每次 build 全量重下；构建上下文 ~185MB 垃圾（`local_cache/`95MB、`.opencode/`55MB、`tools/`22MB）且 **`.env`（文件，含密钥）未被排除**。
  - `download.pytorch.org/whl/cpu` 确认存在 `torch-2.12.1+cpu-cp312` wheel。
- **子代理结论**：CUDA 完全不需要（CPU 推理实证）；torch/torchvision 必须保留（依赖），但应换 CPU 版；pip/apt 可加 BuildKit cache mount。

## ② VERIFY 初判表
| # | 位置 | 初判 |
|---|------|------|
| C1 | `Dockerfile:37-38` `pip install --no-cache-dir` + 清华源 | **有效**：torch 独立分层 + TORCH_INDEX ARG（GPU 自动分流）+ BuildKit cache mount |
| C2 | `.dockerignore` 漏排 `.env`(文件)/local_cache/.opencode/tools 等 | **有效**：收紧，上下文 185MB→~10MB，堵密钥泄入镜像 |
| C3 | 无两机 GPU 智能检测 | **有效**：新增 `scripts/docker_build.py`（nvidia-smi 检测）+ `docker-compose.gpu.yml` |

## ③ CROSS-EXAM
- 无争议。

## ④ CONFIRM 清单
- 全部批准（用户确认：GPU 机 = Windows + Docker Desktop；CUDA 索引默认 cu124 可覆盖）。

## ⑤ IMPLEMENT
- `Dockerfile`：`# syntax=docker/dockerfile:1`；apt/pip 全部改 `--mount=type=cache`；torch==2.12.1+torchvision==0.27.1 独立一层，`ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu` 默认 CPU、GPU 机由脚本传 cu124。
- `.dockerignore`：加 `.env`、`local_cache/`、`company_kb_files/`、`knowledge_lab_files/`、`.opencode/`、`.remember/`、`.playwright-mcp/`、`tools/`、`*.tar.gz`、`cert/`。
- `scripts/docker_build.py`（新增）：`nvidia-smi -L` 检测 → CPU/cu124 分流 → `docker compose build --build-arg TORCH_INDEX=...` → 打印 up 指引；`--up` 可选。
- `docker-compose.gpu.yml`（新增）：app/celery-worker 加 GPU 设备保留（GPU 机专用 override）。
- `docker-compose.yml`：app/celery-worker/celery-beat 加 `./data/industry_words:/app/data/industry_words:ro`（修复 `app_data` 命名卷遮蔽导致行业词表在容器内不可见）。

## ⑥ DOCS
- CHANGELOG 顶部新增 [2026-09-07] 条目；README/USER_MANUAL 补两机构建用法。

## ⑦ PUSH
- 见 commit：<commit-fill>

## ⑧ IMAGE（本机 CPU 分支实测）
- build：`python scripts/docker_build.py` → `GPU detected: False` → CPU 索引 → **~90 秒完成**（原 CUDA 全量 1.5-2h）。
- 镜像：12.8GB → **5.94GB**；容器内 `torch 2.12.1+cpu`、`cuda_available: False`。
- 部署：`docker compose up -d` 滚动更新，app healthy，`https://127.0.0.1/` 200。
- 抽查：industry_words 3 表挂载成功（services 161 词、守卫空）；`kb_file_hash`/`credit_rate:`/`_renderMarkdown`/`escapeHtml` 全部命中。
- 回归 104/104、verify_fixes 89/89、check_system 133/137。

## ⑨ RE-CHECK / 停跑判定
- 无新增 Critical/High → 质量闸门通过。
- GPU 机（RTX 2080 Super）需在真机验证 cu124 + sm_75 兼容性；若不兼容，改 `TORCH_CUDA_INDEX`（如 cu121/cu128）或回退 CPU。
