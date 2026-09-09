# QA-Loop Round 013 (2026-09-09)

基线: last_head=49fea74 | 模式: full（紧急 500 修复 + LLM provider/model 选择器改造） | 触发: user（/clearance/run 与 /admin/runtime_config 双 500 + 选择器改造需求）

## ① COLLECT 摘要（用户反馈）
1. `/clearance/run` POST 500（每次清标提交失败）
2. `/admin/runtime_config` POST 500（"could not save my provider and model preferences"）
3. 想要可编辑 provider/model 选择器：type-in provider URL、每 provider 动态实时拉模型、统一 high thinking；现有下拉不准/过时

## ② VERIFY 根因（2 处 500）
- **R1 /clearance/run**：round-012 F2 在 `clearance.py:167` 把 `target_threads`（Python **list**）塞进 `TaskBus.register_queued(extra=...)` → `task_bus.py:96 hset` 写 Redis → `DataError: Invalid input of type: 'list'`。
- **R2 /admin/runtime_config**：`runtime_config.py:219 update()` 写 `/app/data/runtime_config.json` → `PermissionError`。镜像 `COPY --chown` 早期版本把 `data/runtime_config.json` 烘焙成 root:root；命名卷 `app_data` 保留旧 root 属主；app 以 uid=1000(localai) 运行。**llm_catalog.json 每日刷新同患**。

## ③④ CROSS-EXAM / CONFIRM（用户 + auditor 拍板）
- 用户：R1 用方案 A（json.dumps）；自定义 key 存项目 .env；实时拉取仅"刷新模型"按钮；high thinking 默认 high + admin 开关。
- auditor 加固 3 点：① `_merged_provider_config()` 防御性拷贝 + 内置不覆盖；② 自定义 provider 三重校验（id/base_url/api_key_env）；③ 刷新按钮 loading + 在途防抖。

## ⑤ IMPLEMENT（R 自做 + P1 四路并行）
- **R1** `clearance.py:167` → `json.dumps(target_threads, ensure_ascii=False)`（容器内验证 register_queued 无 DataError）
- **R2** ① `docker exec --user root localai-app chown -R localai:localai /app/data`（活容器立即生效，已验证 runtime_config.json/llm_catalog.json 可写）；② Dockerfile 已有 `COPY --chown=localai:localai` + `chown -R`（新卷持久，无需改）
- **P1-A（general）** `llm_provider.py`：`get_merged_provider_config()`（深拷贝 + 自定义叠加 + `pid in merged` 防覆盖）、`validate_custom_provider()`（三重校验）、`get_available_providers`/`get_provider_config` 改读合并视图、`_create_chat_model_direct` 加 `extra_body={"reasoning_effort": rc_get('llm_reasoning_effort','high')}` + 接入 `llm_temperature`/`llm_max_tokens`。22 断言 PASS
- **P1-B（general)** `runtime_config.py`（DEFAULTS 加 `llm_custom_providers:[]`/`llm_reasoning_effort:'high'` + update 整批校验拒绝非法）+ `llm_catalog.py`（`_fetch_provider_models(base_url, api_key, free_only)` 参数化 + refresh 遍历静态+自定义）。18/18 PASS
- **P1-C（general）** `chat_config.py`（`GET /llm_providers/<pid>/models?refresh=1` 四级兜底 + `/llm_providers` 合并自定义）+ `admin_regeneration.py`（admin 实时端点 + schema `llm_custom_providers` json-list / `llm_reasoning_effort` select）+ `auth.py`（has_llm 补 OPENROUTER/NVIDIA）。15/15 PASS
- **P1-D（frontend-dev）** `review.js`（json-list 行式 Provider 编辑器 + 模型下拉"刷新模型"按钮 loading/在途防抖/stale 提示）+ `.env.example`（`LLM_CUSTOM_KEY_<ID>` 说明）。node OK

## ⑥ DOCS
- CHANGELOG [2026-09-09] round-013 条目 + fix_registry FIX-2026-09-09-020（R1 invariant）+ SYSTEM_CHECKLIST 注。

## ⑦ PUSH
- `49fea74..788ec16` 已推送（LocalAI master），工作树干净。last_head=788ec16。

## ⑧ 验证
- 回归 **121/121** · verify_fixes **101/101** · check_system 133/137 · node --check review/app.js · 后端 7 文件 syntax OK
- 容器：runtime_config.json 可写（R2）、register_queued 无 DataError（R1）
- 子 agent：P1-A 22 断言（防覆盖/校验/签名保留）· P1-B 18 断言（合法入库/非法拒绝/重复拒绝/离线不崩）· P1-C 15 断言（实时端点兜底/未知 404/has_llm）· P1-D node OK

## ⑨ RE-CHECK / 停跑判定
- 无新增 Critical/High → 闸门通过。
- backlog：`llm_fallback_chain` ordered-list 死代码（DEFAULTS 无 key，保存被丢弃）；用户侧 accountModal 选择器接入（现由 admin 面板取代）。