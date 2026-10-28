#!/usr/bin/env bash
# 只读性能采样：对代表性 GET 端点各请求 N 次，记录 curl 客户端耗时（秒）
# 注意：这是"空库/测试环境"下的响应时延观测，不能当作生产并发吞吐量指标。
set -u
BASE="https://127.0.0.1:8443"
N="${N:-10}"
OUT="${OUT:-evidence/perf_samples.csv}"
COOKIE_FILE="${COOKIE_FILE:-cookies.txt}"
echo "endpoint,iter,http_code,time_total_s,time_connect_s,size_bytes" > "$OUT"
for ep in /health / /admin/projects /admin/db_tables_overview /wiki/index /admin/analytics; do
  for i in $(seq 1 "$N"); do
    read -r code tt tc size < <(curl -k -s -b "$COOKIE_FILE" -o /dev/null \
      -w "%{http_code} %{time_total} %{time_connect} %{size_download}" "$BASE$ep")
    echo "$ep,$i,$code,$tt,$tc,$size" >> "$OUT"
  done
done
echo "wrote $OUT"
