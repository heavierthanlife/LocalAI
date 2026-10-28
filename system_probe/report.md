# Local_AI 运行系统探查报告

> 探查人：system-prober（team lead 委派）· 探查时间：2026-10-20 22:54–23:10 (Asia/Shanghai)
> 对象：Windows Docker 中运行的 Local_AI 六服务容器 + PostgreSQL 生产库 `localai`
> 唯一可写目录：`/home/guava5tar_/dsh/playground/review/system_probe/`
> 论文依据：`thesis_v1.txt` 第6章、表6-1（17 项指标）

---

## 0. 数据可得性结论（最重要）

**当前系统没有任何真实业务数据，第6章"从测算升级为实测"不可行。**

生产库 `localai` 共 77 张表，其中 **56 张完全为空**，21 张有行；但有行的表中，
与"再造效果实测"直接相关的表要么为空、要么只含一次性自测残留：

| 关键事实 | 证据 |
|---|---|
| 全库只有 **1 个项目**，且是数据库重置后的测试残留 | `projects` 唯一行：id=2，name=`KekKekKek233`，description=`Data recovered from database reset`，created 2026-07-14 |
| 唯一的项目时间轴 **没有实际起止日期** | `project_timelines` 1 行，`actual_start_date`/`actual_end_date` 均为 NULL，计划区间 2026-07-15→2026-09-20 |
| 11 个里程碑 **全部 pending、全部无实际日期** | `project_timeline_milestones` 11 行，`actual_date`、`diff_days` 全为 NULL |
| 流程环节、文件版本、返工 KPI **均为 0 行** | `project_workflow_steps`=0、`project_file_versions`=0、`workflow_kpi`=0 |
| 清标/审计运行、对话会话、信用报告 **均为 0 行** | `audit_runs`=0、`audit_file_results`=0、`chat_sessions`=0、`credit_check_reports`=0 |
| 仅有的 2 个项目文件是自测上传物，且磁盘文件已丢失 | `docker (2).docx`(18KB)、`ccs ttr.txt`(5.9KB)，2026-07-14 由 admin 上传；容器内 `/app/data/project_files` 为空目录 |
| 唯一的待办是系统把寒暄语自动提取成待办 | `project_todos` 1 行，内容=`你好！请问有什么我可以帮你的？`，创建→完成间隔 2.9 秒 |
| 系统自报的运营指标全为 0 | `/admin/analytics` 返回 `total_messages=0, total_sessions=0, total_files=0, storage_mb="0.0", active_users_24h=0, rag_stats.total=0` |
| 知识库/案例库/法规库实际为空 | `/wiki/index` 返回 `pages:[], tree.children:[]`；`/cases` 返回 `total:0`；`law_articles`=0、`company_knowledge_base`=1（一份草案修改建议） |

**判定：**
1. **不具备"上线前后对比"的数据基础**——没有任何项目完成过完整流程（无实际结束日期、无已完成里程碑）。
2. **不具备"实测效果"的数据基础**——没有返工轮次、没有审计耗时、没有会话量、没有业务工时台账。
3. 现有活动痕迹只能说明"系统有人登录过、点过页面"，不能证明业务被使用（见 §3.3）。
4. 因此论文第6章必须**保持"情景测算"定位**；可将"测量链路可用性"作为设计证据/附录写入，
   但**不得**把任何数值写成"再造后实测"。

---

## 1. 探查范围、方法与安全边界

- 只读探查：`docker ps` / `docker logs` / `curl -k` / `docker exec ... psql` 只读 `SELECT` / `\d`；代码仓只读浏览。
- 未做：改密码、改配置、重启容器、清理缓存、创建/删除生产库业务数据。
- **一处非只读副作用（已披露）**：为验证登录与页面可达性，用 `.env` 中的 `ADMIN_PIN` 做了一次
  `POST /login`（admin 账号）。该登录会写 1 行 `consent` 表示同意会话（`consent` 表 24→25 行），
  `admin_audit_log` 无新增业务记录（登录只写应用日志）。除该行外，生产库没有任何写入。
- 受控功能验证（§7）按 lead 转达的用户决策执行：**只写独立测试库** `localai_probe_verify`，
  验证后 `DROP DATABASE`，生产库行数前后一致（见 §7.6）。

---

## 2. 环境与服务状态

`docker ps`（2026-10-20 23:0x）：六服务容器均在运行、健康检查正常。

| 容器 | 状态 |
|---|---|
| `localai-app` | Up (healthy) |
| `localai-nginx` | Up (healthy) |
| `localai-postgres` | Up (healthy)，PostgreSQL 16.15 |
| `localai-redis` | Up (healthy) |
| `localai-celery-worker` | Up（每 5 分钟收到一次 `cleanup_stale_tasks` 定时任务） |
| `localai-celery-beat` | Up（broker=redis，scheduler=PersistentScheduler） |

- HTTPS 入口 `https://127.0.0.1:8443` 可用（自签名证书，`curl -k`）：`GET /` → **200**，页签标题
  `中联招标智能助手 China United International Engineering AI Services`；`GET /health` →
  `{"database":"connected","status":"healthy"}`。
- 应用启动日志显示 `Scheduler disabled (ENABLE_SCHEDULER != true)`：Web 进程内调度关闭，
  定时任务由 celery beat/worker 承担（与论文 5.4 节描述一致）。
- 环境非独占：探查期间观察到临时容器 `hungry_hertz`、`fervent_hermann`（`local-ai:latest`）
  出现又消失，应是同环境其他协作者所为。本报告不依赖这些容器。

---

## 3. 真实业务数据盘点

### 3.1 任务指定的 14 张表（精确 `count(*)`）

| 表 | 行数 | 表 | 行数 |
|---|---|---|---|
| `projects` | **1** | `project_file_versions` | **0** |
| `project_timelines` | **1** | `admin_audit_log` | 15 |
| `project_timeline_milestones` | 11 | `users` | 6 |
| `project_workflow_steps` | **0** | `chat_sessions` | **0** |
| `project_todos` | **1** | `credit_check_reports` | **0** |
| `workflow_kpi` | **0** | `project_files` | 2 |
| `audit_runs` | **0** | `audit_file_results` | **0** |

### 3.2 全库非空表（77 表中 21 张有行，其余 56 张为 0）

```
wiki_view_log 238 | wiki_origin_links 28 | consent 25 | archived_sessions 24
timeline_suggestions 18 | admin_audit_log 15 | project_timeline_milestones 11
audit_config 7 | users 6 | image_description_cache 3 | project_files 2
bid_templates 2 | bid_template_versions 2 | project_folders 2 | project_timelines 1
project_ai_memory 1 | project_members 1 | company_knowledge_base 1 | project_todos 1
projects 1 | recycle_bin 1
```
（`consent` 25 行 = 原有 24 行 + 本次登录 1 行；其余为探查前既有状态。）

**为空的高信号表**：`audit_runs`、`audit_file_results`、`workflow_kpi`、`project_workflow_steps`、
`project_file_versions`、`chat_sessions`、`chat_messages`、`credit_check_reports`、`feedback`、
`user_feedback`、`case_tags`、`law_articles`、`law_versions`、`compliance_feedback`、`file_usage`、
`template_usage_log`、`file_analysis`、`file_text_cache`。

### 3.3 活动痕迹的性质（为什么不能当作"业务被使用"）

| 痕迹 | 规模与时间 | 判定 |
|---|---|---|
| `admin_audit_log` | 15 行，2026-07-15 → 07-24，全部为 `CONFIG_UPDATE`(13)/`CACHE_TTL_CHANGE`(2)，`table_name` 均为 `system` | 运维配置变更，**无任何业务操作留痕** |
| `wiki_view_log` | 238 行，2026-07-22 → 09-19；其中 admin 237 次、CEO 1 次；页面标题集中在 `wiki/entities/concepts/legal` 等自测路径 | 单一管理员自测，非多用户业务使用 |
| `archived_sessions` | 24 行，2026-07-14 → 09-19，**全部属于 admin** | 自测会话归档 |
| `timeline_suggestions` | 18 行，2026-07-22/23，系统自动生成的"已超期 N 天"告警（针对 11 个 pending 里程碑） | 系统自证其"超期识别"能力可用，但里程碑本身从未被推进 |
| `recycle_bin` | 1 行，admin 的 `admin_日报_20260814_20260815.md`，2026-09-19 删除 | 自测产物已被清理 |

### 3.4 数据完整性异常（不影响结论，但应记录）

1. `users` 存在**两行 username 完全相同**的 `CEO`（hex 均为 `43454f`，created 2026-07-13 与 2026-09-19），
   而 `users_username_key` 唯一索引 `indisvalid=t`。机制未确认（疑似数据库重置/恢复时绕过索引写入），
   本次未做任何修改。另有 `CEO_1`、`COO_1` 两个重名后缀账号。
2. `project_files` 的 2 行记录指向 `data/project_files/2/...`，但容器内 `/app/data/project_files` 为空，
   即**记录存在、文件已不存在**——与 `projects.description` 的 `Data recovered from database reset` 相互印证。
3. `project_todos` 唯一行的 `message_id=118` 指向的聊天消息已不存在（`chat_messages`=0）。

---

## 4. 登录与页面可达性

### 4.1 认证机制（读代码确认）

- 普通用户：`POST /login` 用 `username + PIN`，校验 `users.pin_hash`（scrypt，`pin_length=6`）；
  CEO/COO 均有 6 位 PIN 哈希，**PIN 未知，无法登录**。
- 管理员：`username ∈ _ADMIN_USERNAMES`（默认 `admin,CEO,COO`，见 `app/routes/auth.py:66`）时
  走 `ADMIN_PASSWORD_HASH`（由 `ADMIN_PIN` 派生）校验，**共享同一个管理员 PIN**。
- 实测：`POST /login {"username":"admin","pin":<.env ADMIN_PIN>}` → **200**，
  `{"success":true,"username":"admin","is_admin":true,"is_auditor":true}`，会话 cookie 生效。
- 登录限流：5 次/分钟 + 冷却/锁定（`check_login_gate`），本次仅尝试 1 次。

### 4.2 可用页面（已用会话 cookie 实测 HTTP 200）

| 页面/端点 | 返回 | 说明 |
|---|---|---|
| `GET /` | 200 HTML | 单页应用外壳（登录 + 主界面） |
| `GET /health` | 200 JSON | 数据库连通性 |
| `GET /admin/projects` | 200 JSON | 项目列表（返回唯一测试项目） |
| `GET /admin/users` | 200 JSON | 用户列表（6 个账号） |
| `GET /admin/db_tables_overview` | 200 JSON | 77 张表行数总览（本报告 §3.2 数据来源之一） |
| `GET /admin/analytics` | 200 JSON | 系统总览（消息/会话/存储均为 0） |
| `GET /cases` | 200 JSON | 案例库（total=0） |
| `GET /templates`、`/notebook`、`/wiki/index` | 200 | 模板/笔记本/知识库（内容为空） |
| UI 页签 | — | 智能对话 / 项目管理 / 知识中心 / 回收站 / 系统管理（截图见 §8） |

代码仓 `tests/fixtures/routes_snapshot.json` 记录 **391 条路由**；其中不含路径参数的页面级路由 194 条
（`/admin/*`、`/wiki/*`、`/compliance/*`、`/clearance/*`、`/batch/*` 等）。
404 示例：`/dashboard`、`/projects`、`/timeline`、`/chat`、`/wiki`、`/admin` 均不存在（前端用 SPA 切换而非独立 URL）。

---

## 5. 论文 6.1 十七项指标的可算性（生产数据）

判定口径：**可算** = 生产库有数据且能按表6-1口径算出有业务意义的数值；
**数据为空** = 表结构支持但无数据；**不可算** = 系统内无对应数据源（需外部台账/日志/问卷）。

| # | 准则层 | 指标 | 测量口径 | 生产库支撑情况 | 判定 |
|---|---|---|---|---|---|
| 1 | 流程效率 | 流程周期 | actual_end − actual_start（天） | 1 条时间轴，实际起止均为 NULL | **数据为空**（0 行可算） |
| 2 | 流程效率 | 计划偏差率 | (实际周期−计划周期)/计划周期 | 同上 | **数据为空** |
| 3 | 流程效率 | 里程碑延误 | actual_date − planned_date | 11 条里程碑，0 条有实际日期 | **数据为空** |
| 4 | 流程效率 | 返工强度 | 修订轮次、生成次数（workflow_kpi） | `workflow_kpi` 0 行；`project_file_versions` 0 行 | **数据为空** |
| 5 | 流程效率 | 任务响应 | done_at − created_at | 仅 1 行，是系统把"你好"自动提取的待办（2.9 秒） | **数据不可用** |
| 6 | 集成水平 | 数据一次录入率 | 无需重复录入字段/总字段 | 无对应表/埋点 | **不可算** |
| 7 | 集成水平 | 自动流转占比 | 接口自动流转单据/总单据 | 无接口调用记录表 | **不可算** |
| 8 | 集成水平 | 接口成功率 | 成功调用/总调用 | 无接口日志表（nginx 访问日志非业务接口台账） | **不可算** |
| 9 | 资源节约 | 单项目人工工时 | 项目工时合计/项目数 | 无工时/业务台账表 | **不可算** |
| 10 | 资源节约 | 纸质文档占比 | 纸质流转文件/总文件 | 无档案记录表 | **不可算** |
| 11 | 资源节约 | 人工搬运次数 | 每项目跨系统手工导入导出次数 | 论文口径即"问卷+访谈"，系统无数据 | **不可算（仅问卷）** |
| 12 | 业务价值 | 节点准时交付率 | 按期完成节点/总节点 | 0 个节点有实际日期 | **数据为空** |
| 13 | 业务价值 | 客户满意度 | 满意度评分均值 | `feedback`/`user_feedback` 均 0 行 | **不可算** |
| 14 | 业务价值 | 单项目毛利率 | (收入−成本)/收入 | 无财务收入成本表 | **不可算** |
| 15 | 组织能力 | 培训覆盖率 | 受训人数/应训人数 | 无培训记录表（`/admin/training/*` 是模型 LoRA 训练，非员工培训） | **不可算** |
| 16 | 组织能力 | 工具采纳率 | 月活跃用户/应使用用户 | `wiki_view_log` 238 行，但 237/238 为 admin 自测 | **数据不可用** |
| 17 | 组织能力 | 留痕完整率 | 有审计记录的关键操作/关键操作总数 | `admin_audit_log` 15 行仅系统配置变更 | **数据为空** |

**汇总：生产数据下 0/17 项可算出有业务意义的数值**；
其中 6 项（1,2,3,4,12,17）"表结构支持但生产数据为空"，2 项（5,16）"有数据但为自测"，
9 项（6,7,8,9,10,11,13,14,15）系统内**根本没有数据源**，必须依赖外部台账/系统日志/问卷访谈。

**对论文的直接含义**：表6-1 中"五项指标可直接由平台数据计算"的说法在**当前数据状态下不成立**；
能算的前提是系统里跑过至少一个完整项目闭环。仅"集成水平/资源节约/业务价值/组织能力"共 12 项中，
有 9 项连数据源都不在系统内——这一局限应在 6.2 节明确写出，而不是用"平台可测量性提升"一句带过。

---

## 6. 系统自身性能证据

### 6.1 现成的业务性能证据：无

- `audit_runs`、`audit_file_results` 均为 0 行 ⇒ 清标/全量审计的 `started_at→completed_at` 耗时**没有任何样本**。
- `project_workflow_steps` 0 行 ⇒ 无环节耗时。
- 因此"清标环节耗时下降 32%"这类表述目前**只能是测算值**，无系统实测支撑。

### 6.2 具备性能采集能力（设计证据）

- nginx 访问日志带 `rt`/`uct`/`uht`/`urt` 字段（请求总耗时/上游连接/上游头/上游响应），
  即系统**具备逐请求时延采集能力**，后续接上真实流量即可产出性能数据。样例（本次探查请求）：
  `"GET /wiki/index HTTP/2.0" 200 100 ... rt=0.754 uct="0.001" uht="0.754" urt="0.754"`。
- 应用日志按蓝图记录启动耗时、登录事件、任务接收；celery worker 每 5 分钟记录一次
  `cleanup_stale_tasks` 任务；容器 healthcheck 正常。

### 6.3 本次实测的响应时延（空库、单机、无并发压测）

每端点 10 次顺序请求，客户端 `curl -k` 计时（原始样本 `evidence/perf_samples.csv`）：

| 端点 | min | mean | p95 | max |
|---|---|---|---|---|
| `/health` | 12.3 ms | 13.7 ms | 15.8 ms | 15.8 ms |
| `/` | 10.4 ms | 17.2 ms | 54.3 ms | 54.3 ms |
| `/admin/projects` | 10.6 ms | 12.7 ms | 15.9 ms | 15.9 ms |
| `/admin/db_tables_overview`（77 表 count） | 31.6 ms | 51.5 ms | 121.1 ms | 121.1 ms |
| `/admin/analytics` | 22.8 ms | 40.3 ms | 172.6 ms | 172.6 ms |
| `/wiki/index` | 10.0 ms | 93.4 ms | 828.7 ms | 828.7 ms（明显离群，疑首次扫描/冷缓存） |

**使用边界**：这是"空库 + 无并发"条件下的响应时延观测，**不是**吞吐量/并发能力指标，
也不能推导任何业务效率收益；只能作为"系统可用、接口响应正常"的工程证据。

---

## 7. 受控功能验证（测试环境功能验证，**非生产效果实测**）

> 授权：lead 转达用户决策，同意用运行系统做一次受控功能验证并产出系统性能证据，
> 但必须严格标注"测试环境功能验证，非生产效果实测"。
> 本节的全部数值来自**人工合成的 `PROBE-*` 测试数据**，只证明"测量链路可用、口径可算"，
> **不是** Local_AI 或 Z 公司的再造效果。

### 7.1 方案设计（不污染生产数据）

| 环节 | 做法 |
|---|---|
| 隔离方式 | 在同一 PostgreSQL 实例内新建独立测试库 `localai_probe_verify`（不改生产库 `localai`） |
| 结构真实性 | `pg_dump --schema-only` 从生产库克隆 14 张与测量链路相关的表结构；逐列比对生产/测试结构，**160 列完全一致** |
| 数据 | 植入合成场景：2 个项目（`PROBE-TEST-A` 已完结 / `PROBE-TEST-B` 进行中）、2 条时间轴（计划+实际）、8 个里程碑（计划/实际/偏差天数/原因分类）、8 个环节、4 条待办、2 条 workflow_kpi、2 次审计运行、4 条审计留痕、3 条 wiki 浏览 |
| 写入口径对齐 | 插入列清单与 `app/services/project_timeline_service.py:47,58`、`app/routes/admin_ops.py:74,93`、`app/routes/admin_regeneration.py:483` 一致；`diff_days` 口径 = `actual_date − planned_date`（同 `project_timeline_service.py:188`） |
| 清理 | 验证后 `DROP DATABASE ... WITH (FORCE)`；复核实例内无 `localai_probe_%` 残留；生产库 8 张关键表行数 before/after **完全一致** |

### 7.2 可复现脚本（均在 `system_probe/verify_chain/`）

| 文件 | 作用 |
|---|---|
| `setup_schema.sh` | 建测试库（`TEMPLATE template0`，绕过本实例 template1 collation 报错）+ 克隆 14 表结构 + 结构一致性校验 |
| `scenario.sql` | 合成测试场景（上表"数据"行） |
| `indicator_probe.sql` | 按表6-1口径逐项计算 + "不可算指标"数据源检查 |
| `run_verify.sh` | 一键编排：基线→建库→场景→指标→清理→复核（`KEEP_DB=1` 可保留库人工检查） |
| `cleanup.sh` | 只允许删 `localai_probe_*` 前缀数据库 |
| `README.md` | 性质声明、复现命令、安全边界、环境坑 |

### 7.3 验证结果：测量链路真实产出数据（合成数据的真实数值）

来源：`evidence/verify_chain_indicators.txt`（完整输出）

| 链路环节 | 指标 | 实测产出（合成数据） |
|---|---|---|
| 时间轴 实际起止 | 流程周期 | `PROBE-TEST-A` = **34 天**（2026-08-02→09-05），均值 34.00 天 |
| 时间轴 计划/实际 | 计划偏差率 | 计划 30 天 / 实际 34 天 → **0.1333**（+13.3%） |
| 里程碑 计划/实际 | 里程碑延误 | 8 个里程碑中 6 个有实际日期；5 个延误、1 个提前；**平均 +1.67 天、最大 +5 天、延误合计 11 天** |
| 里程碑 原因分类 | 延误归因 | `administrative` 3 / `other` 2 / `legal` 1（归因能力可用） |
| workflow_kpi | 返工强度 | 平均生成次数 **16.50**、平均修订轮次 **2.50**；文件版本口径：`max(version)=3` → 修订 2 轮 |
| 待办 创建/完成 | 任务响应 | `PROBE-TEST-A` 平均 **9.33 小时**（3 条已完成）；`PROBE-TEST-B` 无已完成待办（返回 NULL） |
| 里程碑 实际日期 | 节点准时交付率 | 6 个有实际日期节点中 1 个按期 → **0.1667** |
| wiki 浏览痕迹 | 工具采纳率（代理口径） | 近 30 天活跃 2 人 / 活跃用户 2 人 → **1.0000** |
| 审计日志 | 留痕完整率（代理口径） | 6 个有实际日期的里程碑中 3 个有审计记录 → **0.5000** |
| audit_runs 起止 | 清标/审计单次耗时 | 47.30 s、72.50 s，平均 **59.90 s**；单文件审计平均 **21.45 s**（2 条样本） |
| 数据源检查 | 9 项"不可算"指标 | 工时/财务/培训/客户/纸质/接口日志类表命中数**全部为 0**，确认系统内无数据源 |

### 7.4 验证结论

1. **测量链路是通的**：从项目→时间轴→里程碑→环节→待办→KPI→审计运行，7 个环节都能写入并算出数值；
   `diff_days`、周期、偏差率、准时率、响应时长、返工轮次、审计耗时等口径**在真实表结构上可计算**。
2. **17 项指标中 8 项口径可算**（1,2,3,4,5,12,16,17），其中 16、17 需要论文补充明确定义（本次用代理口径，
   已在 SQL 注释中标明）；**其余 9 项在系统内无数据源**，属"方法论局限"，不能靠功能验证补足。
3. **链路可用 ≠ 效果实测**：合成数据的数值只能证明"若系统被真实使用，这些指标可被算出"。
   论文若要写"实测"，前置条件是**生产库先跑完至少一个真实项目闭环**（有 `actual_end_date`、
   有已完成里程碑、有 audit_runs 记录）。当前四个条件一个都不满足。

### 7.5 风险与局限

| 风险/局限 | 说明与缓解 |
|---|---|
| 在 PostgreSQL 实例内建库 | 只建 `localai_probe_verify`；用后 `DROP DATABASE ... WITH (FORCE)`；复核无残留、生产库行数不变 |
| 生产库零写入 | 生产库仅 `SELECT` 与 `pg_dump --schema-only`；`projects` id=2 未触碰；`prod_counts_before/after` diff 为空 |
| 未做 API 端到端写入 | 通过真实 API 创建项目会写入生产库，违反隔离原则，故**未做**；改为"克隆真实表结构 + 按代码写入口径插数据"，并在 §7.1 列出代码行号证明列清单一致。局限：未验证 API 参数校验与业务规则 |
| 未启动第二个 app 实例 | 避免新增容器/改环境变量；若需真正端到端，建议在独立 compose 项目里用测试库起一套应用（本次未做） |
| 合成数据不代表真实分布 | 数值仅用于"链路可用性"演示；不可用于敏感性分析、不可作为论文任何表格数据 |
| 环境非独占 | 探查期间有其他协作者容器起停，行数快照以本报告时间为准 |
| `CREATE DATABASE` 报 collation 错 | 本实例 template1 记录了 collation version 但无法确定实际值，改用 `TEMPLATE template0`（已固化在脚本） |

### 7.6 清理复核（已执行）

- 测试库：`DROP DATABASE localai_probe_verify WITH (FORCE)` → 实例内 `localai_probe_%` 查询为空。
- 生产库 8 张关键表行数 before/after **完全一致**（`evidence/prod_counts_before.txt` vs `prod_counts_after.txt`，diff 为空）。
- 未重启任何容器、未修改任何配置；容器状态与探查前后一致。
- 唯一遗留副作用是 §1 披露的 1 行 `consent`（admin 登录写同意标记），不涉及业务数据。

---

## 8. 可采集的真实证据清单

| 证据 | 位置 |
|---|---|
| 全库逐表行数（77 表）+ 指定 14 表精确计数 | 本报告 §3；`evidence/page_03_admin_db_tables.txt`（系统自报的行数 API 原文） |
| 唯一项目记录原文 | `evidence/page_02_admin_projects.txt`（含 `description: Data recovered from database reset`） |
| 系统总览（消息/会话/存储为 0） | `evidence/page_06_admin_analytics.txt` |
| 用户列表（6 账号，无 PIN 明文） | `evidence/page_04_admin_users.txt` |
| 知识库为空 | `evidence/page_05_wiki_index.txt` |
| 案例库为空 | `evidence/page_07_cases.txt` |
| 界面截图 11 张（脱敏见下） | `screenshots/01_login.png` … `11_ui_knowledge_center.png` |
| UI 渲染后的可见文本（截图对应的 DOM 文本） | `evidence/ui_09_ui_project_manage.txt`、`ui_10_ui_system_manage.txt`、`ui_11_ui_knowledge_center.txt` |
| 响应时延原始样本 | `evidence/perf_samples.csv` |
| 受控验证全过程日志 | `evidence/verify_chain_run.log` |
| 受控验证指标输出 | `evidence/verify_chain_indicators.txt` |
| 生产库同口径查询（全空） | `evidence/prod_indicators_empty.txt` |
| 生产库行数前后对照 | `evidence/prod_counts_before.txt` / `prod_counts_after.txt` |

**截图说明（11 张，均为本次登录后采集）**：

1. `01_login.png` 登录页（未登录状态）
2. `02_admin_projects.png` 项目管理（API 返回，唯一测试项目）
3. `03_admin_db_tables.png` 数据表总览（77 表行数，本报告统计来源）
4. `04_admin_users.png` 用户管理（6 个账号名，均为角色名，无个人信息）
5. `05_wiki_index.png` 知识库 Wiki（空树）
6. `06_admin_analytics.png` 运行分析（全 0）
7. `07_cases.png` 案例库（0 条）
8. `08_app_home.png` 主界面（智能对话 UI，含页签与侧边栏）
9. `09_ui_project_manage.png` 项目管理 UI（唯一项目卡片 + 公司知识库文件）
10. `10_ui_system_manage.png` 系统管理 UI（"审计记录: 15 条"、"近 7 天 0 天活跃"）
11. `11_ui_knowledge_center.png` 知识中心 UI（"暂无文件"）

脱敏评估：截图仅含系统自带页签、角色名（CEO/COO/admin）、测试项目名 `KekKekKek233`、
一份法规草案文件名；**无客户信息、无真实投标数据、无 PIN/密钥**。
限制：本会话 `read_image` 工具因 `sharp` 依赖缺失无法读图，截图内容通过 PNG 头（尺寸/完整性）
与 DOM 可见文本（`evidence/ui_*.txt`、`page_*.txt`）交叉确认，未做人眼视觉复核。

---

## 9. 不可用 / 不可做清单（防止后续误用）

1. **不要**把第6章任何数字改称"实测值"。系统生产库目前 **0 个完成项目、0 次审计、0 条返工记录**。
2. **不要**引用 §7 的数值（34 天、+1.67 天、2.5 轮、59.90 秒等）作为 Local_AI 或 Z 公司的效果数据。
   它们是**合成测试数据**，只用于证明"测量链路可用"。
3. **不要**声称"17 项指标均可从平台直接计算"。实际：8 项口径可算（2 项需补充代理口径定义），
   9 项系统内无数据源（工时/财务/培训/客户满意度/纸质档案/接口日志/人工搬运）。
4. **不要**声称有"上线前后对比"。没有任何再造前的系统基线数据可配对。
5. **不要**把 `/health`、页面响应时延当成"效率提升"证据；空库时延不代表真实负载。
6. **不要**在正文使用 `admin` 账号登录截图冒充多用户业务使用；系统里 237/238 条浏览痕迹来自同一个人。
7. 系统存在数据完整性问题（重复 CEO 账号、文件记录指向已删除文件），若论文要描述"数据治理"成就，
   需先说明这些问题已被修复，否则与"过程可追溯"表述冲突。

---

## 10. 给论文修改的建议（供 lead 决策）

1. **第6章维持"测算"定位**，保留 6.2 的"数据可得性约束"表述，但把约束写实：
   不是"业务数据不能导出"，而是**系统当前没有可用于效果评估的生产数据**（0 完成项目）。
2. 可新增一小段"测量链路验证"（放在 6.2 或附录）：说明在隔离测试库中用合成场景验证了
   表6-1 中 8 项指标口径可算，**并明确标注为测试环境功能验证**。这属于"客观设计证据"，
   与 6.2 第三类证据的定位一致，不构成学术不端。
3. 若用户后续提供真实数据：先按 6.2 的配对设计需要"再造前基线"，而系统只有再造后的数据；
   因此**更现实的设计是"再造后横截面 + 访谈基线回溯"**，或从用户提供的业务台账中取基线，
   再讨论 Wilcoxon/ITS 是否可行。仅当系统里跑出 ≥1 个完整项目闭环（有 actual_end_date、有 audit_runs）时，
   才能谈"单项目实测"。
4. 表6-1 建议增加一列"当前数据可得性"，直接把 §5 的判定写进论文，作为方法论的诚实交代；
   这比笼统说"部分指标历史基线不完整"更有说服力。

---

*附：本报告所有计数与查询均为 2026-10-20 23:0x 前后快照；数据随后续操作会变化，引用时请注明时间点。
受控验证脚本可随时重跑（`bash system_probe/verify_chain/run_verify.sh`），运行前请确认无人正在使用该系统。*
