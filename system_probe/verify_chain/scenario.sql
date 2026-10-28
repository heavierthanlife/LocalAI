-- =============================================================================
-- 受控功能验证 · 第 2 步：在独立测试库中植入**合成场景**（不是生产数据）
-- 目的：验证"项目 → 时间轴(计划/实际) → 里程碑(计划/实际/偏差/原因) → 环节
--       → 待办(创建/完成) → workflow_kpi(返工轮次/生成次数) → 审计运行耗时"
--       这条测量链路能否真实产出可计算的数据。
--
-- 数据性质声明：本文件全部数据为人工合成的测试数据（前缀 PROBE-），
-- 仅存在于测试库 localai_probe_* 中，禁止用于任何"生产效果实测"论述。
-- 写入的列清单与 app/services/project_timeline_service.py:47,58、
-- app/routes/admin_ops.py:74,93、app/routes/admin_regeneration.py:483 保持一致。
-- =============================================================================
BEGIN;

-- 1) 两条测试用户（role=user，无 PIN，不参与登录）
INSERT INTO users (user_id, username, role, is_active) VALUES
 ('probe-user-1', 'PROBE_OPERATOR', 'user', true),
 ('probe-user-2', 'PROBE_REVIEWER', 'user', true);

-- 2) 两个测试项目：A=已完结（有实际起止），B=进行中（无实际结束）
INSERT INTO projects (id, name, description, created_by, status, industry) VALUES
 (91001, 'PROBE-TEST-A', '[受控验证] 合成测试项目A：全链路已完结', 'probe-user-1', 'active', 'general'),
 (91002, 'PROBE-TEST-B', '[受控验证] 合成测试项目B：进行中', 'probe-user-2', 'active', 'general');

INSERT INTO project_folders (id, project_id, name, path, created_by) VALUES
 (98001, 91001, '投标文件', '投标文件', 'probe-user-1'),
 (98002, 91002, '投标文件', '投标文件', 'probe-user-2');

INSERT INTO project_files (id, project_id, folder_id, filename, original_name, file_size, stored_path, uploaded_by, category) VALUES
 (99001, 91001, 98001, 'probe_bid_a1.docx', 'PROBE-投标文件甲.docx', 102400, 'probe/probe_bid_a1.docx', 'probe-user-1', '通用'),
 (99002, 91001, 98001, 'probe_bid_a2.docx', 'PROBE-投标文件乙.docx', 98304,  'probe/probe_bid_a2.docx', 'probe-user-1', '通用'),
 (99003, 91001, 98001, 'probe_tender_a.docx', 'PROBE-招标文件.docx', 204800, 'probe/probe_tender_a.docx', 'probe-user-1', '通用'),
 (99004, 91002, 98002, 'probe_bid_b1.docx', 'PROBE-投标文件丙.docx', 88064,  'probe/probe_bid_b1.docx', 'probe-user-2', '通用');

-- 文件版本化：99001 有 3 个版本（对应"修订轮次"的另一种口径）
INSERT INTO project_file_versions (id, file_id, version, stored_path, file_size, uploaded_by) VALUES
 (99501, 99001, 1, 'probe/probe_bid_a1.docx',   102400, 'probe-user-1'),
 (99502, 99001, 2, 'probe/probe_bid_a1_v2.docx', 104000, 'probe-user-1'),
 (99503, 99001, 3, 'probe/probe_bid_a1_v3.docx', 105500, 'probe-user-1');

-- 3) 时间轴：计划起止；A 已填实际起止（由 update_timeline 写入），B 只有实际开始
INSERT INTO project_timelines (id, project_id, name, category_code, method_code,
                               planned_start_date, planned_end_date, status, created_by) VALUES
 (92001, 91001, 'PROBE-测试时间轴A', 'construction', 'open_bid',
  DATE '2026-08-01', DATE '2026-08-31', 'archived', 'probe-user-1'),
 (92002, 91002, 'PROBE-测试时间轴B', 'construction', 'open_bid',
  DATE '2026-09-01', DATE '2026-09-30', 'active', 'probe-user-2');

UPDATE project_timelines SET actual_start_date = DATE '2026-08-02',
                             actual_end_date   = DATE '2026-09-05'
 WHERE id = 92001;
UPDATE project_timelines SET actual_start_date = DATE '2026-09-03'
 WHERE id = 92002;

-- 4) 里程碑：计划日期 + 实际日期 + 偏差天数 + 偏差原因分类
--    （diff_days 口径 = actual_date - planned_date，见 project_timeline_service.py:188）
INSERT INTO project_timeline_milestones
 (id, timeline_id, milestone_code, milestone_name, planned_date, actual_date,
  diff_days, diff_reason, reason_category, status, sort_order) VALUES
 (93001, 92001, 'TENDER_NOTICE',      '发布招标公告',     DATE '2026-08-01', DATE '2026-08-02',  1, '招标人资料补齐延迟', 'administrative', 'delayed', 1),
 (93002, 92001, 'DOC_SALE_END',       '招标文件发售截止', DATE '2026-08-08', DATE '2026-08-07', -1, '提前完成',           'other',          'done',    2),
 (93003, 92001, 'BID_OPENING',        '开标',             DATE '2026-08-20', DATE '2026-08-22',  2, '投标人澄清要求延期', 'other',          'delayed', 3),
 (93004, 92001, 'EVALUATION',         '评标',             DATE '2026-08-25', DATE '2026-08-27',  2, '评审专家档期冲突',   'administrative', 'delayed', 4),
 (93005, 92001, 'CONTRACT_SIGN',      '签订书面合同',     DATE '2026-08-31', DATE '2026-09-05',  5, '合同条款谈判反复',   'legal',          'delayed', 5),
 (93006, 92002, 'TENDER_NOTICE',      '发布招标公告',     DATE '2026-09-05', DATE '2026-09-06',  1, '公告挂网审核延迟',   'administrative', 'done',    1),
 (93007, 92002, 'BID_SUBMIT_DEADLINE','投标文件递交截止', DATE '2026-09-15', NULL,             NULL, NULL,               NULL,             'pending', 2),
 (93008, 92002, 'AWARD_NOTICE',       '发出中标通知书',   DATE '2026-09-30', NULL,             NULL, NULL,               NULL,             'pending', 3);

-- 5) 环节（步骤）：责任人与完成状态
INSERT INTO project_workflow_steps (id, milestone_id, step_name, step_order, assigned_to, completed, completed_at) VALUES
 (94001, 93001, '编制公告',           1, 'probe-user-1', true,  TIMESTAMPTZ '2026-08-02 09:30+00'),
 (94002, 93001, '内部复核公告',       2, 'probe-user-2', true,  TIMESTAMPTZ '2026-08-02 15:00+00'),
 (94003, 93003, '组织开标会',         1, 'probe-user-1', true,  TIMESTAMPTZ '2026-08-22 10:00+00'),
 (94004, 93004, '组建评标委员会',     1, 'probe-user-1', true,  TIMESTAMPTZ '2026-08-25 09:00+00'),
 (94005, 93004, '汇总评标报告',       2, 'probe-user-2', true,  TIMESTAMPTZ '2026-08-27 18:00+00'),
 (94006, 93005, '合同条款谈判',       1, 'probe-user-1', true,  TIMESTAMPTZ '2026-09-04 17:00+00'),
 (94007, 93006, '编制公告',           1, 'probe-user-2', true,  TIMESTAMPTZ '2026-09-06 11:00+00'),
 (94008, 93007, '接收投标文件',       1, 'probe-user-2', false, NULL);

-- 6) 待办：创建时间 / 完成时间（用于"任务响应"= done_at - created_at）
INSERT INTO project_todos (id, project_id, user_id, content_copy, original_role, status, created_at, done_at) VALUES
 (95001, 91001, 'probe-user-1', 'PROBE-补齐招标人授权委托书', 'assistant', 'done',    TIMESTAMPTZ '2026-08-01 09:00+00', TIMESTAMPTZ '2026-08-01 10:30+00'),
 (95002, 91001, 'probe-user-1', 'PROBE-通知投标人澄清',       'assistant', 'done',    TIMESTAMPTZ '2026-08-19 14:00+00', TIMESTAMPTZ '2026-08-19 14:30+00'),
 (95003, 91001, 'probe-user-2', 'PROBE-催办评审专家确认',     'assistant', 'done',    TIMESTAMPTZ '2026-08-24 08:00+00', TIMESTAMPTZ '2026-08-25 10:00+00'),
 (95004, 91002, 'probe-user-2', 'PROBE-准备开标准备',         'assistant', 'pending', TIMESTAMPTZ '2026-09-07 09:00+00', NULL);

-- 7) workflow_kpi：生成次数 / 返工轮次（返工强度口径）
INSERT INTO workflow_kpi (id, project_id, user_id, generations, revision_rounds, output_chars, completed_count, last_active) VALUES
 (96001, 91001, 'probe-user-1', 24, 3, 52000, 6, TIMESTAMPTZ '2026-09-05 12:00+00'),
 (96002, 91002, 'probe-user-2',  9, 2, 18000, 2, TIMESTAMPTZ '2026-09-07 10:00+00');

-- 8) 审计运行（清标/全量审计）：started_at / completed_at → 单次耗时
INSERT INTO audit_runs (id, project_id, user_id, status, overall_score, bidder_count, file_count,
                        started_at, completed_at, task_id) VALUES
 (97001, 91001, 'probe-user-1', 'done', 86.5, 2, 2,
  TIMESTAMPTZ '2026-08-21 10:00:00+00', TIMESTAMPTZ '2026-08-21 10:00:47.300+00', 'probe-task-1'),
 (97002, 91001, 'probe-user-2', 'done', 88.0, 2, 2,
  TIMESTAMPTZ '2026-08-21 10:05:00+00', TIMESTAMPTZ '2026-08-21 10:06:12.500+00', 'probe-task-2');

INSERT INTO audit_file_results (id, run_id, file_id, folder_id, bidder_label, filename, function_name,
                                score, status, started_at, completed_at) VALUES
 (97101, 97001, 99001, 98001, '投标人甲', 'PROBE-投标文件甲.docx', 'compliance', 85.0, 'done',
  TIMESTAMPTZ '2026-08-21 10:00:02+00', TIMESTAMPTZ '2026-08-21 10:00:21.100+00'),
 (97102, 97001, 99002, 98001, '投标人乙', 'PROBE-投标文件乙.docx', 'compliance', 88.0, 'done',
  TIMESTAMPTZ '2026-08-21 10:00:22+00', TIMESTAMPTZ '2026-08-21 10:00:45.800+00');

-- 9) 操作留痕（用于"留痕完整率"代理口径）：3 条里程碑变更审计 + 1 条系统配置
INSERT INTO admin_audit_log (admin_user_id, admin_username, action, table_name, row_id, created_at) VALUES
 ('probe-user-1', 'PROBE_OPERATOR', 'UPDATE', 'project_timeline_milestones', '93001', TIMESTAMPTZ '2026-08-02 09:00+00'),
 ('probe-user-1', 'PROBE_OPERATOR', 'UPDATE', 'project_timeline_milestones', '93003', TIMESTAMPTZ '2026-08-22 09:00+00'),
 ('probe-user-1', 'PROBE_OPERATOR', 'UPDATE', 'project_timeline_milestones', '93005', TIMESTAMPTZ '2026-09-05 09:00+00'),
 ('probe-user-1', 'PROBE_OPERATOR', 'CONFIG_UPDATE', 'system', NULL, TIMESTAMPTZ '2026-08-01 08:00+00');

-- 10) 使用痕迹（用于"工具采纳率"代理口径）
INSERT INTO wiki_view_log (user_id, wiki_path, article_title, viewed_at) VALUES
 ('probe-user-1', 'legal/laws/index', 'legal',  TIMESTAMPTZ '2026-08-02 08:00+00'),
 ('probe-user-1', 'templates/index',  'templates', TIMESTAMPTZ '2026-08-03 08:00+00'),
 ('probe-user-2', 'legal/laws/index', 'legal',  TIMESTAMPTZ '2026-08-04 08:00+00');

COMMIT;
