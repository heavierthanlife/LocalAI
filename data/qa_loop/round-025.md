# QA-Loop Round 025 (2026-09-14)
基线: last_head=b7fdab0 | 模式: 定向功能补齐（用户报告：LLM 提供商增删/key 界面缺失）| 触发: manual

## 背景
用户反馈"让管理员添加/删除 LLM 提供商并自动读取刷新其模型"只做了一半：只见刷新按钮，
未见增删 provider（url/key）界面。定位：功能其实已存在（运行配置面板 `llm_custom_providers`
json-list 行编辑器 + `/admin/llm_providers/<pid>/models?refresh=1`），缺口在于
① 只能填 `api_key_env` 不能直接粘 key；② 保存后不自动拉模型；③ 不失效缓存；④ 无 key 状态提示；⑤ 发现性差。

## 用户确认
1. 目标：**改进现有 json-list 编辑器**（非新建页面）。
2. key 落盘：**双写** `data/llm_provider_keys.env`（持久） + 根 `.env`（存在时）。
3. 保存后自动对每个自定义 provider 即时调 `/models`（最佳努力）。
4. PWA 警告（beforeinstallprompt preventDefault 提示）：保留现状（自定义按钮）。

## ⑤ IMPLEMENT
- **FIX-049（后端）**：新增 `app/services/env_store.py`（`write_env_var` 原子 upsert + os.environ 同步 + 双写 + `load_provider_keys`）；`app/__init__.py` 启动加载；`update_runtime_config` 保存钩子（写 key→env、校验、自动拉模型回写、清 agent 缓存）；`/admin/llm_providers` 增 `api_key_set`/`custom`。
- **FIX-050（前端）**：`review.js` 行编辑器增 API Key 密码框 + ✓/✗ 角标 + 说明；dirty 纳入 api_key。
- `docs/ARCHITECTURE.md` 服务数 100→101。

## ⑧/⑨ 验证
- 回归 131/131 · verify_fixes 203/0 · doc_drift 14/14 · node --check OK · code-reviewer 复核（0 Critical/High，1 Medium `$` 展开已修）。
- 镜像重建=HEAD（env_store / api_key_set / review.js key 字段 / load_provider_keys 全中），`/check_auth`=200。
- 浏览器实测：面板新增测试 provider（localhost base_url + key）→ 保存 → 生成 `qa_test`（runtime_config 仅存 api_key_env，
  明文 key 落 `data/llm_provider_keys.env`）→ `/admin/llm_providers` `api_key_set=true` → 已清理测试 provider 与 env 行。

## ⑩ 追加发现（多 worker 可见性，FIX-049 修正 commit 79a5f19）
- 现象：保存后 `/admin/llm_providers` 一度 `api_key_set=false`。
- 根因：gunicorn 4 worker，`write_env_var` 只改了处理保存请求那个进程的 `os.environ`，其余 worker 读不到。
- 修复：`env_store.get_env()/has_env_var()` 按 mtime **懒加载** `data/llm_provider_keys.env`（override）；
  `llm_provider/llm_fallback/llm_catalog/chat_config/admin_regeneration` 的 key 读取点统一改用 `env_store.get_env`。
- 复验：重建后 `api_key_set=true`；独立进程 `has_env_var` 亦 True。

## ⑪ 追加（FIX-051，用户反馈"找不到在哪选用自定义提供商+模型"）
- 根因：`/admin/runtime_config_schema` 的 `active_llm_provider`/`active_llm_model` 选项仅从
  `PROVIDER_CONFIG` 构建 → 自定义 provider 不在下拉；用户侧 `loadProviderSelector` 为死代码（无调用点）。
- 决定：不复活用户侧选择器；只修运行配置面板（管理员路径）。
- 修复：schema 选项改从 `get_merged_provider_config()` 构建（含自定义 + 模型，标「（自定义）」）；
  review.js 空模型提示。
- 验证：回归 132/132 · verify_fixes 206/0 · doc_drift 14/14；浏览器实测运行配置 LLM 段出现自定义 provider 并可存 active_llm_provider/model。

## 备注
- Docker 内根 `.env` 非挂载（compose 插值注入），故主持久化为 `data/llm_provider_keys.env`（`app_data` 卷）；
  本地开发若根 `.env` 存在则双写。
- 明文 key 只写不读：接口仅返回布尔 `api_key_set`，runtime_config.json 只存 `api_key_env`。
