#!/usr/bin/env bash
# =============================================================================
# 受控功能验证 · 第 1 步：在同一个 PostgreSQL 实例内创建一个独立测试库，
# 并把生产库 localai 中与"测量链路"相关的 14 张表的**结构**克隆进去。
#
# 安全边界（必须遵守）：
#   * 只创建/操作名字以 localai_probe_ 开头的数据库；生产库 localai 只做只读 pg_dump。
#   * 不写入生产库任何一行记录。
#   * 不重启容器、不改配置、不动 projects 表中已有数据。
# =============================================================================
set -euo pipefail

: "${DB_NAME:?必须显式指定测试库名，例如 DB_NAME=localai_probe_verify}"
case "$DB_NAME" in
  localai_probe_*) : ;;
  *) echo "拒绝执行：测试库名必须以 localai_probe_ 开头（当前：$DB_NAME）" >&2; exit 2 ;;
esac
[ "$DB_NAME" = "localai" ] && { echo "拒绝执行：不能使用生产库名" >&2; exit 2; }

PG_CONTAINER="${PG_CONTAINER:-localai-postgres}"
PROD_DB="${PROD_DB:-localai}"
TABLES="users projects project_folders project_files project_file_versions \
project_timelines project_timeline_milestones project_workflow_steps project_todos \
workflow_kpi audit_runs audit_file_results wiki_view_log admin_audit_log"

echo "== 1. 创建测试库 $DB_NAME =="
docker exec "$PG_CONTAINER" psql -U localai -d postgres -v ON_ERROR_STOP=1 \
  -c "DROP DATABASE IF EXISTS \"$DB_NAME\";" \
  -c "CREATE DATABASE \"$DB_NAME\" TEMPLATE template0;"

echo "== 2. 从生产库导出 14 张表的结构（只读，不导出数据）=="
T_ARGS=""
for t in $TABLES; do T_ARGS="$T_ARGS -t $t"; done
docker exec "$PG_CONTAINER" sh -c \
  "pg_dump -U localai -d $PROD_DB --schema-only --no-owner --no-privileges $T_ARGS" \
  > /tmp/localai_probe_schema.sql
echo "   schema dump: $(wc -l < /tmp/localai_probe_schema.sql) 行"

echo "== 3. 载入测试库 =="
docker exec -i "$PG_CONTAINER" psql -U localai -d "$DB_NAME" -q -v ON_ERROR_STOP=1 \
  < /tmp/localai_probe_schema.sql

echo "== 4. 结构一致性校验（生产库 vs 测试库，逐列比对）=="
IN_LIST=$(echo $TABLES | tr ' ' '\n' | sed "s/.*/'&'/" | paste -sd, -)
SIG_SQL="select table_name||'.'||column_name||':'||data_type||':'||is_nullable||':'||coalesce(column_default,'') as sig
         from information_schema.columns where table_schema='public'
           and table_name in ($IN_LIST) order by 1;"
docker exec "$PG_CONTAINER" psql -U localai -d "$PROD_DB" -tAc "$SIG_SQL" > /tmp/probe_sig_prod.txt
docker exec "$PG_CONTAINER" psql -U localai -d "$DB_NAME" -tAc "$SIG_SQL" > /tmp/probe_sig_test.txt
if diff -q /tmp/probe_sig_prod.txt /tmp/probe_sig_test.txt >/dev/null; then
  echo "   OK：测试库 14 张表结构与生产库完全一致（$(wc -l < /tmp/probe_sig_test.txt) 列）"
else
  echo "   !! 结构差异：" >&2; diff /tmp/probe_sig_prod.txt /tmp/probe_sig_test.txt | head -20 >&2; exit 3
fi
echo "== 完成：$DB_NAME 就绪（空表）=="
