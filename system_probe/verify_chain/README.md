# 受控功能验证（测量链路可用性）— 复现说明

**性质声明：本目录的一切内容属于「测试环境功能验证」，不是生产效果实测。**
它回答的问题只有一个：论文表6-1的指标口径，在 Local_AI 的**真实表结构**上能否算出数值。
产出的数值全部来自人工合成的 `PROBE-*` 测试数据，**禁止**在任何论文表格、图注或正文中
作为"再造后实测值"引用。

## 为什么用独立数据库

生产库 `localai` 当前只有 1 个测试项目（id=2）、0 条已完成里程碑、0 条 audit_runs/chat_sessions，
直接在生产库写入测试项目会污染业务数据（且原始授权要求只读）。因此：

- 在同一个 PostgreSQL 实例里新建**独立测试库** `localai_probe_verify`；
- 用 `pg_dump --schema-only` 从生产库克隆 14 张与测量链路相关的表结构（不导出任何数据行）；
- 在测试库里植入合成场景并跑指标 SQL；
- 跑完 `DROP DATABASE ... WITH (FORCE)`，再复核生产库行数与执行前完全一致。

## 一键复现

```bash
# 完整执行：建库 → 克隆结构 → 植入合成场景 → 算指标 → 清理 → 复核生产库未变
bash run_verify.sh

# 只想人工检查测试库内容（执行后不自动删库）
KEEP_DB=1 bash run_verify.sh
bash cleanup.sh      # 人工检查后手动清理（只允许删 localai_probe_* 前缀）
```

证据输出：`../evidence/verify_chain_run.log`、`../evidence/verify_chain_indicators.txt`、
`../evidence/prod_indicators_empty.txt`、`../evidence/prod_counts_before.txt|prod_counts_after.txt`。

## 文件说明

| 文件 | 作用 |
|---|---|
| `setup_schema.sh` | 建测试库（`CREATE DATABASE ... TEMPLATE template0`）并克隆 14 表结构，逐列比对生产/测试结构一致性 |
| `scenario.sql` | 植入合成场景：2 个测试项目、时间轴（计划/实际起止）、8 个里程碑（计划/实际/偏差/原因分类）、环节、待办（创建/完成）、workflow_kpi（返工轮次/生成次数）、audit_runs（审计耗时）、审计日志、wiki 浏览痕迹 |
| `indicator_probe.sql` | 按表6-1口径逐项计算，并检查"不可算指标"是否缺少数据源 |
| `run_verify.sh` | 编排 + 证据落盘 + 清理 + 生产库行数前后复核 |
| `cleanup.sh` | 仅删除 `localai_probe_` 前缀数据库，防止误删生产库 |

## 安全边界（脚本内已做硬编码校验）

1. 测试库名必须以 `localai_probe_` 开头，`localai` 会被拒绝；
2. 生产库只做 `pg_dump --schema-only`（只读）与 `SELECT`；
3. 不重启容器、不改任何配置、不动 `projects` 表中 id=2 的记录；
4. 结束后 `DROP DATABASE`，并比对 8 张关键表行数 before/after。

## 已知环境坑

- 该 PostgreSQL 实例的 `template1` 记录了 collation version 但实际无法确定，
  直接 `CREATE DATABASE` 会报
  `template database "template1" has a collation version, but no actual collation version could be determined`；
  脚本改用 `TEMPLATE template0` 绕过（已在 `setup_schema.sh` 中固化）。
- 生产库 `users` 表存在两行 `username` 完全相同的 `CEO`（hex 均为 `43454f`），
  与唯一索引 `users_username_key` 的定义矛盾，机制未确认；属既有状态，本次未做任何修改。
