# QA-Loop Round 026 (2026-09-15)
基线: last_head=095cddd | 模式: 基础设施（Docker HF 缓存）+ Headroom 实测 | 触发: manual

## 背景
用户追问 `~/.cache/huggingface/hub` 里的模型是否在用；确认 5 个模型全部在用（3× sentence-transformers + 2× Headroom/Kompress）。随后任务：① 实测 Headroom 是否真正生效；③ 评估并实施 Docker 复用 HF 缓存。

## 用户决定
- ③ 方案 B（`HF_HOME` 持久卷）。
- ① 与 ③ 一起做；先按"容器自行下载"，失败再预置。

## ③ IMPLEMENT（FIX-052）
- `docker-compose.yml`：`app`/`celery-worker`/`celery-beat` 增 `HF_HOME=/app/data/hf_cache` + `HF_HUB_DISABLE_SYMLINKS_WARNING=1`。
- `Dockerfile`：数据目录 mkdir 增 `hf_cache`。
- 重建镜像 + `--force-recreate`。

## ③ 实测（持久化）
- 三服务 `printenv HF_HOME` = `/app/data/hf_cache`。
- 容器内同步下载 `chopratejas/kompress-v2-base` int8 ONNX（262MB）→ 落在 `/app/data/hf_cache/hub/...`。
- `docker compose up -d --force-recreate` 后仍在（app 与 celery-worker 皆可见）→ **持久化成功**。
- 备注：短时 `docker exec` 进程退出会中断 headroom 的**后台**下载线程；需同步下载（`hf_hub_download`）或长驻进程才能完成。

## ① 实测（Headroom，host + container）
- `headroom-ai 0.27.0` 已装/已加载；`headroom.compress` 经 `ContentRouter` 路由到 Kompress。
- **结果：基本不压缩（saved=0）**：
  - `content_router: 1 msgs → 1 unchanged (ratio>=0.85)` → `router:noop`（host 与 container 一致）。
  - 阈值来自 `content_router.py:655` `min_ratio_relaxed=0.85`（上下文压力低时）；本项目输入远小于 `model_limit(200000)` → 压力≈0 → 判"不值得压缩"。
  - `compress_file_content`/`compress_search_results` 传**单条** message，另受默认 `protect_recent=4` 保护。
  - Kompress 为**英文**模型（项目主要中文）。
- 结论：Headroom/ Kompress 作为依赖与缓存存在，但对本项目的 token 节省**基本为 0**。已记入 `AGENTS.md` Gotchas。

## 文档 / 交付
- `AGENTS.md`：环境变量表 + Gotchas。
- `CHANGELOG.md` 顶部；`fix_registry` FIX-2026-09-11-052。
- 镜像重建=HEAD；`/check_auth`=200。

## 备注 / 后续可选（未做）
- 若要真正启用 Headroom：需下调 `min_ratio` / 传 `protect_recent=0` / `target_ratio`，或直接 `headroom_enabled=false` 关闭以免无谓加载。
- 若不打算用 Headroom，可考虑卸载 `headroom-ai` 并清理 kompress 缓存（~0.8GB）——需另立决策。
