-- =============================================================================
-- 受控功能验证 · 第 3 步：按论文表6-1的指标口径，在**合成测试数据**上实际计算。
-- 运行对象：测试库 localai_probe_*（不是生产库 localai）。
-- 说明：下列数值来自合成测试场景，仅证明"测量链路可用、口径可计算"，
--       不得作为任何生产效果实测值引用。
-- =============================================================================
\echo '################ 1) 流程效率 / 流程周期（天）= actual_end - actual_start ################'
SELECT t.id AS timeline_id, p.name AS project, t.actual_start_date, t.actual_end_date,
       (t.actual_end_date - t.actual_start_date) AS cycle_days
  FROM project_timelines t JOIN projects p ON p.id = t.project_id
 WHERE t.actual_start_date IS NOT NULL AND t.actual_end_date IS NOT NULL
 ORDER BY t.id;
SELECT count(*) AS n_projects,
       round(avg(t.actual_end_date - t.actual_start_date)::numeric, 2) AS avg_cycle_days
  FROM project_timelines t
 WHERE t.actual_start_date IS NOT NULL AND t.actual_end_date IS NOT NULL;

\echo '################ 2) 流程效率 / 计划偏差率 = (实际周期 - 计划周期) / 计划周期 ################'
SELECT p.name AS project,
       (t.planned_end_date - t.planned_start_date) AS planned_days,
       (t.actual_end_date  - t.actual_start_date)  AS actual_days,
       round((((t.actual_end_date - t.actual_start_date)
             - (t.planned_end_date - t.planned_start_date))::numeric
             / NULLIF((t.planned_end_date - t.planned_start_date), 0)), 4) AS deviation_ratio
  FROM project_timelines t JOIN projects p ON p.id = t.project_id
 WHERE t.actual_start_date IS NOT NULL AND t.actual_end_date IS NOT NULL
   AND t.planned_start_date IS NOT NULL AND t.planned_end_date IS NOT NULL
 ORDER BY t.id;

\echo '################ 3) 流程效率 / 里程碑延误（天）= actual_date - planned_date ################'
SELECT count(*) AS n_milestones,
       count(actual_date) AS n_with_actual,
       count(*) FILTER (WHERE diff_days > 0) AS n_delayed,
       count(*) FILTER (WHERE diff_days <= 0 AND actual_date IS NOT NULL) AS n_on_time_or_early,
       round(avg(diff_days)::numeric, 2) AS avg_diff_days,
       max(diff_days) AS max_delay_days,
       sum(diff_days) FILTER (WHERE diff_days > 0) AS total_delay_days
  FROM project_timeline_milestones;
\echo '-- 延误归因（reason_category 分布，展示"延误可归因"能力）'
SELECT coalesce(reason_category,'(无实际记录)') AS reason_category, count(*) AS n
  FROM project_timeline_milestones GROUP BY 1 ORDER BY 2 DESC;

\echo '################ 4) 流程效率 / 返工强度（修订轮次、生成次数）################'
SELECT p.name AS project, k.generations, k.revision_rounds, k.output_chars
  FROM workflow_kpi k JOIN projects p ON p.id = k.project_id ORDER BY k.id;
SELECT round(avg(generations)::numeric,2) AS avg_generations,
       round(avg(revision_rounds)::numeric,2) AS avg_revision_rounds
  FROM workflow_kpi;
\echo '-- 返工强度的第二个口径：文件版本数（max(version)）'
SELECT p.name AS project, f.original_name, max(v.version) AS max_version,
       max(v.version) - 1 AS revision_rounds_from_versions
  FROM project_file_versions v JOIN project_files f ON f.id = v.file_id
  JOIN projects p ON p.id = f.project_id GROUP BY 1,2 ORDER BY 1,2;

\echo '################ 5) 流程效率 / 任务响应（小时）= done_at - created_at ################'
SELECT p.name AS project,
       round(avg(extract(epoch FROM (td.done_at - td.created_at))/3600)::numeric, 2) AS avg_response_hours,
       count(td.done_at) AS n_done, count(*) AS n_total
  FROM project_todos td JOIN projects p ON p.id = td.project_id
 GROUP BY 1 ORDER BY 1;
SELECT round(avg(extract(epoch FROM (done_at - created_at))/3600)::numeric, 2) AS overall_avg_response_hours
  FROM project_todos WHERE done_at IS NOT NULL;

\echo '################ 6) 业务价值 / 节点准时交付率 = 按期完成节点 / 已登记实际的节点 ################'
SELECT count(*) FILTER (WHERE actual_date IS NOT NULL) AS n_with_actual,
       count(*) FILTER (WHERE actual_date IS NOT NULL AND actual_date <= planned_date) AS n_on_time,
       round((count(*) FILTER (WHERE actual_date IS NOT NULL AND actual_date <= planned_date))::numeric
             / NULLIF(count(*) FILTER (WHERE actual_date IS NOT NULL), 0), 4) AS on_time_rate
  FROM project_timeline_milestones;

\echo '################ 7) 组织能力 / 工具采纳率（代理口径：近30天有查看痕迹用户 / 活跃用户）################'
SELECT (SELECT count(DISTINCT user_id) FROM wiki_view_log
         WHERE viewed_at >= (SELECT max(viewed_at) FROM wiki_view_log) - interval '30 days') AS active_30d,
       (SELECT count(*) FROM users WHERE is_active) AS active_users,
       round((SELECT count(DISTINCT user_id) FROM wiki_view_log
               WHERE viewed_at >= (SELECT max(viewed_at) FROM wiki_view_log) - interval '30 days')::numeric
             / NULLIF((SELECT count(*) FROM users WHERE is_active), 0), 4) AS adoption_rate_proxy;

\echo '################ 8) 组织能力 / 留痕完整率（代理口径：已登记实际日期的里程碑中，有审计记录的比例）################'
SELECT count(*) AS milestones_with_actual,
       count(*) FILTER (WHERE has_audit) AS with_audit,
       round(count(*) FILTER (WHERE has_audit)::numeric / NULLIF(count(*), 0), 4) AS trace_completeness_proxy
  FROM (
    SELECT m.id,
           EXISTS (SELECT 1 FROM admin_audit_log a
                    WHERE a.table_name = 'project_timeline_milestones'
                      AND a.row_id = m.id::text) AS has_audit
      FROM project_timeline_milestones m
     WHERE m.actual_date IS NOT NULL
  ) s;

\echo '################ 9) 附加：清标/全量审计单次耗时（秒）= completed_at - started_at ################'
SELECT id AS run_id, project_id, status,
       round(extract(epoch FROM (completed_at - started_at))::numeric, 2) AS duration_seconds,
       file_count, bidder_count
  FROM audit_runs WHERE completed_at IS NOT NULL ORDER BY id;
SELECT round(avg(extract(epoch FROM (completed_at - started_at)))::numeric, 2) AS avg_run_seconds,
       count(*) AS n_runs
  FROM audit_runs WHERE completed_at IS NOT NULL;
\echo '-- 单文件审计耗时'
SELECT round(avg(extract(epoch FROM (completed_at - started_at)))::numeric, 2) AS avg_file_seconds,
       count(*) AS n_file_results
  FROM audit_file_results WHERE completed_at IS NOT NULL;

\echo '################ 10) 不可算指标的数据源检查（应为 0，即系统内无对应数据表/台账）################'
SELECT '数据一次录入率/自动流转占比/接口成功率(需接口日志表)' AS indicator,
       count(*) AS matching_tables
  FROM pg_tables WHERE schemaname='public'
   AND tablename ~* '(api_log|interface|integration|sync_log|request_log)'
UNION ALL
SELECT '单项目人工工时(需工时/业务台账)', count(*) FROM pg_tables WHERE schemaname='public'
   AND tablename ~* '(timesheet|worklog|labor|manhour|cost|budget|finance)'
UNION ALL
SELECT '纸质文档占比(需档案/纸质流转记录)', count(*) FROM pg_tables WHERE schemaname='public'
   AND tablename ~* '(paper|archive|physical|archive_record)'
UNION ALL
SELECT '客户满意度(需客户回访记录)', count(*) FROM pg_tables WHERE schemaname='public'
   AND tablename ~* '(satisfaction|survey|interview|customer)'
UNION ALL
SELECT '单项目毛利率(需财务收入成本台账)', count(*) FROM pg_tables WHERE schemaname='public'
   AND tablename ~* '(revenue|income|gross_margin|ledger|invoice)'
UNION ALL
SELECT '培训覆盖率(需培训记录)', count(*) FROM pg_tables WHERE schemaname='public'
   AND tablename ~* '(training_record|course|curriculum|train_attendance)';
