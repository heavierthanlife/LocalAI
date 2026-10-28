#!/usr/bin/env bash
# =============================================================================
# 受控功能验证 · 编排脚本（只写测试库，不写生产库）
#
# 步骤：
#   0. 记录生产库关键表行数基线（只读）
#   1. 建独立测试库 localai_probe_verify，克隆 14 张表结构（setup_schema.sh）
#   2. 植入合成测试场景（scenario.sql）
#   3. 按论文表6-1口径跑指标 SQL，落盘证据（indicator_probe.sql）
#   4. 同样的指标 SQL 在生产库 localai 上跑一遍（只读，结果应为空/0）
#   5. DROP 测试库，并复核生产库行数与基线一致
#
# 用法： bash run_verify.sh            # 完整执行并自动清理
#        KEEP_DB=1 bash run_verify.sh # 保留测试库供人工复核（需再手动跑 cleanup.sh）
# =============================================================================
set -euo pipefail

DB_NAME="${DB_NAME:-localai_probe_verify}"
PG_CONTAINER="${PG_CONTAINER:-localai-postgres}"
PROD_DB="${PROD_DB:-localai}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EVID="$(cd "$HERE/.." && pwd)/evidence"
mkdir -p "$EVID"

BASELINE_SQL="select 'projects' t, count(*) n from projects
 union all select 'project_timelines', count(*) from project_timelines
 union all select 'project_timeline_milestones', count(*) from project_timeline_milestones
 union all select 'project_todos', count(*) from project_todos
 union all select 'project_files', count(*) from project_files
 union all select 'workflow_kpi', count(*) from workflow_kpi
 union all select 'audit_runs', count(*) from audit_runs
 union all select 'admin_audit_log', count(*) from admin_audit_log
 order by 1;"

echo "===== 0. 生产库行数基线 =====" | tee "$EVID/verify_chain_run.log"
docker exec "$PG_CONTAINER" psql -U localai -d "$PROD_DB" -c "$BASELINE_SQL" 2>/dev/null \
  | tee -a "$EVID/verify_chain_run.log" | tee "$EVID/prod_counts_before.txt" >/dev/null

echo "===== 1. 建测试库并克隆结构 =====" | tee -a "$EVID/verify_chain_run.log"
DB_NAME="$DB_NAME" PG_CONTAINER="$PG_CONTAINER" PROD_DB="$PROD_DB" \
  bash "$HERE/setup_schema.sh" 2>&1 | tee -a "$EVID/verify_chain_run.log"

echo "===== 2. 植入合成测试场景 =====" | tee -a "$EVID/verify_chain_run.log"
docker exec -i "$PG_CONTAINER" psql -U localai -d "$DB_NAME" -q -v ON_ERROR_STOP=1 \
  -f - < "$HERE/scenario.sql" 2>&1 | tee -a "$EVID/verify_chain_run.log"
echo "   已写入合成场景（PROBE-* 项目 / 时间轴 / 里程碑 / 待办 / KPI / 审计运行）" | tee -a "$EVID/verify_chain_run.log"

echo "===== 3. 在测试库上计算 17 项指标（表6-1口径）=====" | tee -a "$EVID/verify_chain_run.log"
docker exec -i "$PG_CONTAINER" psql -U localai -d "$DB_NAME" -f - \
  < "$HERE/indicator_probe.sql" > "$EVID/verify_chain_indicators.txt" 2>&1
sed -i '/no actual collation version/d' "$EVID/verify_chain_indicators.txt"
echo "   指标计算输出 -> evidence/verify_chain_indicators.txt" | tee -a "$EVID/verify_chain_run.log"

echo "===== 4. 同一套指标 SQL 在**生产库**上的结果（只读对照，证明生产数据为空）=====" \
  | tee -a "$EVID/verify_chain_run.log"
{
  echo "【生产库 localai 上的同样口径查询】——用以对照：合成测试数据可算出数值，生产库为空/不可算"
  echo
  docker exec "$PG_CONTAINER" psql -U localai -d "$PROD_DB" -c "
    select '流程周期' ind, count(*) filter (where actual_start_date is not null and actual_end_date is not null) computable_rows,
           count(*) total_rows from project_timelines
    union all select '里程碑延误', count(*) filter (where actual_date is not null), count(*) from project_timeline_milestones
    union all select '返工强度', count(*), count(*) from workflow_kpi
    union all select '任务响应', count(*) filter (where done_at is not null), count(*) from project_todos
    union all select '审计耗时', count(*) filter (where completed_at is not null), count(*) from audit_runs
    union all select '留存证据(审计日志)', count(*), count(*) from admin_audit_log order by 1;" 2>/dev/null
  echo
  echo "【生产库 projects / project_timelines 实际内容】"
  docker exec "$PG_CONTAINER" psql -U localai -d "$PROD_DB" -c \
    "select id,name,description,created_at from projects order by id;" 2>/dev/null
} > "$EVID/prod_indicators_empty.txt"
echo "   对照结果 -> evidence/prod_indicators_empty.txt" | tee -a "$EVID/verify_chain_run.log"

if [ "${KEEP_DB:-0}" = "1" ]; then
  echo "KEEP_DB=1：保留测试库 $DB_NAME，请人工复核后执行 bash $HERE/cleanup.sh" | tee -a "$EVID/verify_chain_run.log"
  exit 0
fi

echo "===== 5. 清理测试库并复核生产库未被改动 =====" | tee -a "$EVID/verify_chain_run.log"
DB_NAME="$DB_NAME" PG_CONTAINER="$PG_CONTAINER" bash "$HERE/cleanup.sh" 2>&1 \
  | tee -a "$EVID/verify_chain_run.log"

echo "===== 6. 生产库行数复核 =====" | tee -a "$EVID/verify_chain_run.log"
docker exec "$PG_CONTAINER" psql -U localai -d "$PROD_DB" -c "$BASELINE_SQL" 2>/dev/null \
  > "$EVID/prod_counts_after.txt"
if diff -q "$EVID/prod_counts_before.txt" "$EVID/prod_counts_after.txt" >/dev/null; then
  echo "   OK：生产库行数与执行前完全一致（未污染）" | tee -a "$EVID/verify_chain_run.log"
else
  echo "   !! 警告：生产库行数发生变化" | tee -a "$EVID/verify_chain_run.log"
  diff "$EVID/prod_counts_before.txt" "$EVID/prod_counts_after.txt" | head | tee -a "$EVID/verify_chain_run.log"
fi
echo "完成。证据：$EVID/verify_chain_indicators.txt" | tee -a "$EVID/verify_chain_run.log"
