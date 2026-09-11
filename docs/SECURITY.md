# SECURITY — 安全基线与修复记录

## 报告漏洞 / 安全问题

项目使用 **GitHub Security Advisories** 与 **Security.md** 模板，请联系项目维护者报告问题。

- **报告渠道**：项目仓库 Issues（私密）或直接联系维护者
- **修复时效**：High/Critical 24h 内响应，Medium 7 天内

---

## 审计基线

**全项目审计**（2026-08-27，LMcode AI Agent）发现并修复的问题：

### 严重（Critical，已修复）

| # | 问题 | 位置 | 修复 |
|---|---|---|---|
| C1 | admin.py 体量过大 | `routes/admin.py` (4,820→1,653) | ✅ 拆分 3 子模块 |
| C2 | chat.py 体量过大 | `routes/chat.py` (2,025→1,122) | ✅ 拆分 3 子模块 |
| C3 | knowledge.py 体量过大 | `routes/knowledge.py` (2,018→941) | ✅ 拆分 5 子模块 |
| C4 | file_processing.py 体量过大 | `services/file_processing.py` (1,551) | ⏸️ 保留（21 导入点，拆分收益低） |
| C5 | graph API 无认证 | `routes/graph.py` | ✅ 5 端点加 `@login_required` + 项目成员检查 |
| C6 | 管理员默认 PIN 123456 | `__init__.py:111` | ✅ 生产 fail-closed，开发保留默认+告警 |

### 中等（Medium，已修复）

| # | 问题 | 状态 |
|---|---|---|
| M1 | DB 迁移散布 | ✅ `WHEN others` → `WHEN undefined_column`（真实错误浮出） |
| M2 | 全局可变状态 | ✅ credit_tasks 迁移 Redis；其余良性缓存 |
| M3 | 匿名存储 JSON 文件 | ✅ 迁移 `anon_chat_messages` JSONB 表，原子 UPSERT |
| M4 | credit_tasks 内存共享 | ✅ 迁移 Redis 注册表 |
| M5 | `.gitignore` `*.json` 全局排除 | ✅ 定向规则 + 18 个跟踪 JSON 保留 |
| M6 | SSL 私钥在仓库 | ✅ `git rm --cached cert/key.pem` |
| M7 | msedgedriver.exe 在仓库 (20.8MB) | ✅ 移除跟踪 + 运行时自动下载 |
| M8 | pycorrector tar.gz (4.4MB) | ✅ 无需处理（未跟踪） |
| M9 | 部分 service 无 docstring | ✅ 5 文件补 docstring + BOM 移除 |

### 低（Low，已处理）

| # | 问题 | 状态 |
|---|---|---|
| L1 | Windows 特定代码 | ✅ config.py stdout 包装 hasattr 防护 |
| L2 | 代码风格不统一 | ⏸️ 持续改进 |
| L3 | 测试覆盖率未知 | ✅ 路由守护 + 回归测试 |
| L4 | 重复验证脚本 | ✅ 无需处理（过期条目） |

### 早期审计发现（2026-07-08，`.audit/last-full-audit.md`）

| 级别 | 发现 | 状态 |
|---|---|---|
| HIGH | admin cleanup 5 处裸 `except: pass` 吞异常 | ✅ 改为 `except OSError` |
| MEDIUM | f-string SQL 拼接（admin/rag/recycle_bin/skill_auditor） | ✅ 现全部 `%s` 占位符（硬编码 dict 安全） |
| LOW | 45+ 处宽 `except Exception:` 无日志 | ✅ 补 logger |
| LOW | knowledge 路由 f-string WHERE | ✅ 现 `%s` 占位符 |
| LOW | auth 装饰器模式不一致 | ⏸️ 维持（均安全，可维护性关注） |

---

## 安全架构要点

- **认证**：session + JWT（`auth_jwt.py`），`check_auth` 健康检查
- **权限**：`@admin_required` / `@login_required` + 项目成员检查（`routes/projects.py` 权限工具）
- **CSRF**：opt-in，JSON API 默认关闭（AJAX 无需）
- **Rate Limiting**：flask-limiter + Redis（聊天 30/min、上传 10/min、登录 5/min、全局 120/min）
- **上传**：50MB 限制、扩展名白名单、SHA-256 去重、流式分块（8MB）、上传信号量（并发 3）
- **Prompt 安全**：`prompt_safety.py` 注入防护 + anti-hallucination；`agent_middleware.py` InvalidToolGuard
- **路径安全**：`to_rel_path()`/`resolve_path()` 防止路径遍历；分享 token 7 天 + 遍历防护
- **密钥管理**：`SECRET_KEY`/`FLASK_SECRET_KEY` 必填，缺失启动失败；`.env` gitignored

---

## Fix Registry 机制

`data/fix_registry.yaml` 记录每个修复的**不变式**，`.githooks/pre-commit` 运行 `scripts/verify_fixes.py` 防止修复变异。

- 修复 bug 时必须添加 registry 条目（含 grep 校验）
- 旁路：`SKIP_FIX_CHECK=1`
- Python 后端修复必须补 `tests/test_regression.py` 测试
- 当前：**70/70 校验通过**

**不变量示例**：
- FIX-014：`file_processing.py` 含 `min(200,max(50,len(template_text)//500))`；`batch_orchestrator.py` 含 `_detect_component` + `component_mismatch`
- FIX-013：三处向量化路径统一过滤停用词

---

## 部署安全

- Docker 非 root 用户（`localai`）
- nginx HTTPS TLS 1.2/1.3 + 安全头
- 12G body limit（支持大文件上传，生产按需评估）
- 生产多 worker 必须 `ENABLE_SCHEDULER=false`
- Edge/Chrome WebDriver 延迟加载，`EDGEDRIVER_PATH` 可覆盖
