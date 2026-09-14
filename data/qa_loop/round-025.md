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
- 回归 129/129 · verify_fixes 203/0 · doc_drift 14/14 · node --check OK · code-reviewer 复核。
- 容器抽查 + 浏览器实测：运行配置加测试 provider → 保存 → key 角标 ✓ → 模型下拉自动出现。

## 备注
- Docker 内根 `.env` 非挂载（compose 插值注入），故主持久化为 `data/llm_provider_keys.env`（`app_data` 卷）；
  本地开发若根 `.env` 存在则双写。
- 明文 key 只写不读：接口仅返回布尔 `api_key_set`，runtime_config.json 只存 `api_key_env`。
