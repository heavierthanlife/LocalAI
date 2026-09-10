# QA-Loop Round 020 (2026-09-09)

基线: last_head=b44c4a4 | 模式: full（主 agent 提示词：不可变默认 + 每用户自定义 + 消息模板统一） | 触发: user（提示词模板路由合并到默认硬编码；每用户可查看/改/存，≤2；默认不可变）

## ① 设计（用户确认）
- **服务器默认 = 硬编码 `_DEFAULT_PROMPT`，不可变**（任何路由不可写）。
- **每用户**可查看原始版 + 保存**自己的 ≤2 个 agent 版本**（选一生效，无则回退默认）。
- **消息模板**（原 localStorage 片段）**统一进 DB**，每用户 **≤5**。
- 用户提示词自动追加安全 guard；入口对**所有登录用户**开放。
- 丢弃 `data/agent_prompt.json`（旧管理员全局覆盖）。

## ② 数据模型
`user_prompts(id, user_id, kind, name, content, is_active, created_at, updated_at)`，索引 `(user_id, kind)`。
`kind='agent'`：≤2/用户、至多一 active；`kind='template'`：≤5/用户。存原始 content，解析时追加 guard。

## ③ 实现
**后端（子 agent A）**
- `app/database.py`：新增 `user_prompts` 表 + 索引。
- `app/globals.py`：`_DEFAULT_PROMPT` 不可变 + `get_default_prompt()`；删除 `_load_prompt`/`save_prompt`/`_PROMPT_FILE`；新增 `_agent_cache`/`_agent_cache_lock`。
- 新 `app/services/user_prompt.py`：`resolve_user_prompt`（active+guard / 默认+guard）· `list_user_prompts` · `save_user_prompt`（≤2/≤5 校验，首版自动 active）· `activate_prompt`（唯一 active）· `delete_prompt` · `migrate_templates`（≤5 截断）。
- `app/routes/chat_config.py`：新增 `GET /prompts/default`、`GET /prompts/mine`、`POST /prompts/save|activate|delete|migrate_templates`（登录校验，user 隔离）；`g._agent=None` → `g._agent_cache.clear()`。
- `app/routes/admin_regeneration.py`：**退役** `/admin/system_prompt` GET/POST。
- `app/routes/chat.py`（:343 流式 / :834 隔离）+ `app/services/agent.py get_agent`：改 `resolve_user_prompt(session['user_id'])`；`get_agent` 缓存键 = `(user_id, sha1(prompt)[:12], max_tokens)` + LRU 上限 8；删除 `agent.py:26` 模块快照。

**前端（子 agent B）**
- 统一 `openPromptEditor(initialTab)` 弹窗：两 tab（系统提示词 / 消息模板）。
  - 系统提示词：原始版只读 + 我的 ≤2 版（编辑/重命名/删除/设生效 radio）+ 新建（满 2 禁用）
  - 消息模板：我的 ≤5（使用→插入输入框 / 重命名 / 删除 / 新建）；首次打开把旧 localStorage 模板迁移入库并清空本地
- 入口：`#promptEditorBtn`（设置区，全员可见）打开 system 页；`#promptTemplatesBtn` 打开 template 页；删除 admin 区旧 `#sidebarEditPromptBtn`；合并删除 app.js:3325 / knowledge-lab.js:51 旧副本。

## ④ 验证
- 回归 **128/128** · verify_fixes **125/125** · T0 **no_route=0** · node ×2 OK · 后端 7 文件 syntax OK
- 容器 API 实测：`/prompts/default` len=782（不可变）· agent save v1/v2=200、v3=400（≤2）· activate→唯一 active · template 5 行、第 6 拒 · migrate 已满 imported=0 · 跨用户隔离（user=0）· `/admin/system_prompt` → **404（已退役）**
- 后端子 agent 本地单测全过（≤2/≤5/active/隔离/迁移 6→5/401）

## ⑤ 备注
- e2e 库偶发 DB pool 重连导致脚本轮询停顿（非代码问题）；核心断言均已确认。
- backlog：agent 缓存 LRU=8 可按需调；旧 localStorage 模板迁移为一次性。
