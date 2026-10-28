#!/usr/bin/env bash
# =============================================================================
# 受控功能验证 · 清理：删除测试库。只允许删除 localai_probe_ 前缀的数据库。
# 生产库 localai 不在此脚本的权限范围内（会被前缀校验拒绝）。
# =============================================================================
set -euo pipefail
: "${DB_NAME:?必须显式指定测试库名，例如 DB_NAME=localai_probe_verify}"
case "$DB_NAME" in
  localai_probe_*) : ;;
  *) echo "拒绝执行：只能删除 localai_probe_ 前缀的数据库（当前：$DB_NAME）" >&2; exit 2 ;;
esac
PG_CONTAINER="${PG_CONTAINER:-localai-postgres}"

echo "== 清理测试库 $DB_NAME =="
docker exec "$PG_CONTAINER" psql -U localai -d postgres -v ON_ERROR_STOP=1 \
  -c "DROP DATABASE IF EXISTS \"$DB_NAME\" WITH (FORCE);"
echo "== 复核：实例内剩余数据库清单 =="
docker exec "$PG_CONTAINER" psql -U localai -d postgres -tAc \
  "select datname from pg_database where datname like 'localai_probe_%';"
echo "   （上面为空表示测试库已删除干净）"
