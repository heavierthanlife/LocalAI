# MANIFEST — Local_AI 目录地图

项目目录结构速查，用于快速定位代码位置。**本文件只描述目录职责，不记录文件数/行数等易变计数**（计数类事实由 `scripts/check_doc_drift.py` 对照代码校验）。

> **来源**：2026-08-27 全项目技术审计。安全审计细节见 [`SECURITY.md`](SECURITY.md)，架构细节见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

---

## 目录速查

```
app/                 核心应用（__init__/config/database/globals/cleanup_tasks/celery_app）
app/routes/          HTTP 蓝图（清标/合规/聊天/知识库/管理/Wiki/项目…）
app/services/        业务逻辑层（LLM/RAG/OCR/清标/审计/合规/文档处理…）
app/utils/           工具函数（helpers/chinese_numbers/headroom/mailer）
templates/           index.html（SPA 壳）
static/              JS/CSS/PWA 资源
migrations/          迁移 SQL（001 为无操作基线，实际 schema 在 app/database.py）
scripts/             运维脚本（manage_db/check_system/verify_fixes/dump_routes…）
tests/               测试（smoke/regression/batch/route-preservation/integration/fixtures）
repair_kit/          崩溃恢复参考（SYSTEM_CHECKLIST + SCHEMA_SNAPSHOT + check_integrity）
data/                运行时数据（法规库/上传文件/知识库/unresolved.yaml/fix_registry.yaml）
docs/                项目文档（ARCHITECTURE/MANIFEST/DECISIONS/SECURITY/USER_MANUAL/IMPROVEMENTS_SKIPPED + 规格）
.audit/              审计增量日志 + state
.remember/           会话记忆（AI 代理）；新会话由 session-bootstrap 插件自注入 handoff + unresolved + findings
.githooks/           提交钩子（pre-commit: fix registry + 蓝图覆盖 + 文档计数 drift；post-commit: 写 qa-loop 待运行标记）
D:/AI_Tools/shared-agent-infra/tool-extensions/opencode/  共享 agent 扩展（skill-mcp 服务器 + 插件）
```

---

## 文档地图

| 位置 | 内容 |
|---|---|
| `README.md`（根） | 项目入口：能力、架构、快速开始 |
| `AGENTS.md`（根） | 开发者/Agent 指南：命令、约定、Gotchas、SOP、Fix Registry |
| `CHANGELOG.md`（根） | 版本化变更历史 |
| `docs/ARCHITECTURE.md` | 逐模块深度架构 + 数据流 |
| `docs/MANIFEST.md` | 本文件：目录地图 |
| `docs/DECISIONS.md` | 技术决策（ADR）+ 已评估未采纳方案 |
| `docs/SECURITY.md` | 安全基线 + 审计发现 + Fix Registry 机制 |
| `docs/USER_MANUAL.md` | 运维 / 管理员操作手册 |
| `docs/IMPROVEMENTS_SKIPPED.md` | 历史记录（活跃未决项见 `data/unresolved.yaml`） |
| `repair_kit/README.md` | 崩溃恢复参考 |

---

*目录地图由 2026-08-27 技术审计归纳；2026-09-11 精简为纯目录地图（去计数）。*
