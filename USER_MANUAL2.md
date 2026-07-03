# 中联招标智能助手 — 完整审计报告

> 版本: 2026-07-01 | 完整前端+后端+基础设施审计

---

## 一、前端 8 个标签页及其完整功能

### 标签页 1: 💬 对话 (Chat) — 全员可见

**侧边栏 (sidebar-chat-pane)**:
| 元素 ID | 功能 |
|---|---|
| `newChatBtn` | 新建对话 |
| `sidebarSearch` | 快速筛选对话列表 |
| `openSearchModalBtn` | 打开搜索聊天记录弹窗 |
| `historyList` | 普通对话历史列表 |
| `projChatHeader` | 项目对话标题（条件显示） |
| `projectHistoryList` | 项目对话历史列表 |
| `todoHeader` + `todoCount` + `todoList` | 我的待办（最多5条，项目内右键添加） |
| `bgTasksHeader` + `bgTaskCount` + `bgTasksList` | 后台任务列表（SSE进度） |

**聊天区域**:
| 元素 ID | 功能 | 后端路由 |
|---|---|---|
| `chatMessages` | 消息显示区域 | — |
| `messageInput` | 消息输入框 | POST `/send_stream` (SSE流式) |
| `fileBtn` + `fileInput` | 文件上传（多文件） | POST `/upload_file` |
| `knowledgeBaseBtn` | 知识库选择弹窗 | GET `/knowledge_lab/list` |
| `promptTemplatesBtn` | 提示词模板下拉 | 前端硬编码 |
| `sendBtn` | 发送消息 | POST `/send_stream` |
| `scrollTopBtn` / `scrollBottomBtn` | 浮动滚动按钮 | — |

**高级工具 (折叠的 `<details>`)**:
| 元素 ID | 功能 |
|---|---|
| `maxTokensInput` + `applyTokensBtn` | Token输出控制 (100-4800) |
| `analyzeImagesCheckbox` | VL图片分析开关 |
| `checkTextSim` | 文本对比因子 |
| `checkKeyInfo` | 关键信息对比因子 |
| `checkFileAttr` | 文件属性对比因子 |
| `checkImageSim` | 图片对比因子 |
| `checkSemantic` | 智能语义分析开关 |
| `batchCompareBtn` | 批量对比按钮 |
| `clearBatchFilesBtn` | 清空已选文件 |
| `batchFileInputContainer` | 批量文件输入容器 |
| `templateFileInput` + `selectTemplateBtn` + `clearTemplateBtn` | 模板文件选择 |
| `fileStationBtn` | 文件站按钮 |
| `storageWarning` | 存储空间警告 |

**头部按钮**:
| 元素 ID | 功能 | 后端路由 |
|---|---|---|
| `dailyReportChatBtn` | 生成今日工作日报 | POST `/admin/generate_work_report` |
| `exportChatBtn` | 导出对话为Markdown | 前端生成 |
| `pwaInstallBtn` | PWA安装 | 浏览器API |
| `themeToggleBtn` | 深色/浅色主题切换 | localStorage + CSS |
| `processingIndicator` | 处理中状态指示器 | — |
| `progressBar` + `progressBarFill` | 顶部进度条 (SSE驱动) | — |
| `progressToast` | 底部进度提示浮层 | — |

### 标签页 2: 📂 项目 (Projects) — 全员可见

**侧边栏 (sidebar-projects-pane)**:
| 元素 ID | 功能 |
|---|---|
| `sidebarCreateProjectBtn` | 新建项目 |
| `sidebarMyFilesBtn` | 我的文件 |
| `sidebarProjectsList` | 项目列表 |

**主面板 (adminPanel)**:
| 元素 ID | 功能 | 后端路由 |
|---|---|---|
| `createProjectBtn` | 新建项目 | POST `/admin/projects` |
| `myFilesBtn` | 我的文件 | GET `/user_project_files` |
| `projectsList` | 项目列表 | GET `/admin/projects` |
| `fileExplorerView` | 文件浏览器视图（打开项目后显示） | — |
| `backToProjectsBtn` | 返回项目列表 | — |
| `presenceIndicator` | 在线状态指示器 | GET `/admin/projects/<id>/presence` |
| `fileExplorerContent` | 文件内容 | GET `/admin/projects/<id>/files` |

**项目内功能（通过JS触发）**:
| 功能 | 后端路由 |
|---|---|
| 项目聊天 (SSE流式) | POST `/admin/projects/<id>/ai_assist` |
| 文件上传到项目 | POST `/admin/projects/<id>/folders/<folder_id>/upload` |
| 文件下载 | GET `/admin/projects/<id>/files/<file_id>/download` |
| 文件版本 | GET `/admin/projects/<id>/files/<file_id>/versions` |
| 文件评论 | GET/POST `/admin/projects/<id>/files/<file_id>/comments` |
| 文件移动 | POST `/admin/projects/<id>/files/<file_id>/move` |
| 批量移动 | POST `/admin/projects/<id>/files/batch_move` |
| 批量下载 | POST `/admin/projects/<id>/batch_download` |
| 文件搜索 | GET `/admin/projects/<id>/files/search` |
| 文件重命名 | PUT `/admin/projects/<id>/files/<file_id>/rename` |
| 新版本上传 | POST `/admin/projects/<id>/files/<file_id>/new_version` |
| 文件夹管理 | GET/POST `/admin/projects/<id>/folders` |
| 文件夹重命名 | PUT `/admin/projects/<id>/folders/<folder_id>/rename` |
| 文件夹删除 | DELETE `/admin/projects/<id>/folders/<folder_id>` |
| 文件夹评论 | GET/POST `/admin/projects/<id>/folders/<folder_id>/comments` |
| 成员管理 | GET/POST `/admin/projects/<id>/members` |
| 成员搜索 | GET `/admin/projects/<id>/members/search` |
| 成员角色更新 | PUT `/admin/projects/<id>/members/<user_id>` |
| 成员移除 | DELETE `/admin/projects/<id>/members/<user_id>` |
| 转移项目经理 | POST `/admin/projects/<id>/transfer_manager/<user_id>` |
| 项目中止 | POST `/admin/projects/<id>/abort` |
| 项目完成 | POST `/admin/projects/<id>/finish` |
| 项目删除 | DELETE `/admin/projects/<id>` |
| 项目更新 | PUT `/admin/projects/<id>` |
| 下载归档 | GET `/admin/projects/<id>/download_archive/<zip>` |
| 回填聊天 | POST `/admin/projects/<id>/backfill_chat` |
| 在线心跳 | POST `/admin/projects/<id>/ping` |
| 在线状态 | GET `/admin/projects/<id>/presence` |
| AI活动轮询 | GET `/admin/projects/<id>/ai_activity` |
| 未读数 | GET `/admin/projects/<id>/unread_count` |
| 标记已读 | POST `/admin/projects/<id>/mark_read` |

**项目内协作工具（右键菜单）**:
| 功能 | 后端路由 |
|---|---|
| 添加待办 | POST `/admin/projects/<id>/todos` |
| 引用追问 | POST `/admin/projects/<id>/quote` |
| 复制内容 | 前端clipboard |
| 待办列表 | GET `/admin/projects/<id>/todos` |
| 完成待办 | POST `/admin/projects/<id>/todos/<todo_id>/done` |
| 删除待办 | POST `/admin/projects/<id>/todos/<todo_id>/remove` |
| 待办完成日志 (admin) | GET `/admin/projects/<id>/todos/done_log` |
| 引用树 | GET `/admin/projects/<id>/quote_tree/<message_id>` |
| 投票列表 | GET `/admin/projects/<id>/regen_votes` |
| 投票 | POST `/admin/projects/<id>/regen_votes/<vote_id>/cast` |
| 裁决 (manager) | POST `/admin/projects/<id>/regen_votes/<vote_id>/resolve` |
| AI输出下载 | GET `/admin/projects/<id>/ai/download/<memory_id>` |
| 工作流定制 | GET/POST `/admin/projects/<id>/my_workflow` |

### 标签页 3: 📚 知识库 (Knowledge Lab) — 已登录可见

**侧边栏 (sidebar-knowledge-pane)**:
| 元素 ID | 功能 | 后端路由 |
|---|---|---|
| `sidebarUploadKnowledgeBtn` | 上传个人知识文件 | POST `/knowledge_lab/upload` |
| `sidebarRefreshKnowledgeBtn` | 刷新列表 | GET `/knowledge_lab/list` |
| `sidebarKnowledgeFiles` | 个人文件列表 | GET `/knowledge_lab/list` |
| `sidebarUploadCompanyBtn` | 上传公司文件 | POST `/knowledge_lab/upload` (company=True) |
| `sidebarCompanyFiles` | 公司文件列表 | — |

**主面板 (knowledgeLabPanel)**:
| 元素 ID | 功能 | 后端路由 |
|---|---|---|
| `uploadLabFileBtn` + `labFileInput` | 上传文件 | POST `/knowledge_lab/upload` |
| `refreshLabListBtn` | 刷新 | GET `/knowledge_lab/list` |
| `labFileList` | 个人知识库文件列表 | — |
| `companyFileInput` + `uploadCompanyFileBtn` | 上传公司文件 (admin) | POST `/knowledge_lab/upload` |
| `companyCategorySelect` + `companyCustomCategory` | 公司文件分类选择 | — |
| `companyKbSearch` + `companyKbCategoryFilter` + `refreshCompanyKbBtn` | 公司知识库搜索筛选 | — |
| `companyKbList` | 公司知识库文件列表 | — |
| `notebookNewBtn` + `notebookRefreshBtn` + `notebookSearch` | 笔记本操作 | GET `/notebook` |
| `notebookList` | 笔记列表 | GET `/notebook` |
| `notebookEditor` + `notebookEditTitle` + `notebookEditContent` | 笔记编辑器 | POST `/notebook/<id>` |
| `notebookSaveBtn` | 保存笔记 | POST `/notebook/<id>` |
| `notebookSummarizeBtn` | AI摘要 | POST `/notebook/<id>/summarize` |
| `notebookCancelBtn` | 取消编辑 | — |
| `refreshSkillOverviewBtn` | 刷新技能总览 | — |
| `skillOverviewList` + `skillOverviewCount` | 技能总览列表 | GET `/admin/skill_audit` |

**知识库后端端点**:
| 路由 | 方法 | 功能 |
|---|---|---|
| `/knowledge_lab/upload` | POST | 上传到个人/公司知识库 |
| `/knowledge_lab/list` | GET | 个人知识库列表 |
| `/knowledge_lab/skill/<file_id>` | GET | 查看文件技能摘要 |
| `/knowledge_lab/content/<file_id>` | GET | 查看文件内容 |
| `/knowledge_lab/rename/<file_id>` | POST | 重命名知识库文件 |
| `/knowledge_lab/rename_skill/<file_id>` | POST | 重命名技能 |
| `/company_kb/rename/<file_id>` | POST | 重命名公司文件 |
| `/knowledge_lab/delete/<file_id>` | POST | 删除知识库文件 |
| `/knowledge_lab/generate_skill/<file_id>` | POST | 生成技能摘要 |
| `/project_files/<file_id>/generate_skill` | POST | 为项目文件生成技能 |
| `/notebook` | GET | 笔记列表 |
| `/notebook/<note_id>` | GET/POST/DELETE | 查看/保存/删除笔记 |
| `/notebook/<note_id>/summarize` | POST | AI摘要 |
| `/notebook/search` | POST | 语义搜索笔记 |
| `/admin/all_user_kb` | GET | 查看所有用户知识库 (admin) |

### 标签页 4: 🗑️ 回收站 (Recycle Bin) — 全员可见

**侧边栏 (sidebar-recycle-pane)**:
| 元素 ID | 功能 |
|---|---|
| `sidebarRecycleStats` | 回收站统计信息 |
| `sidebarRecycleFilters` | 过滤按钮组 (all/chat/user_file/knowledge_lab/company_kb) |
| `sidebarRestoreAllBtn` | 恢复筛选结果 |
| `sidebarEmptyAllBtn` | 清空筛选结果 |

**主面板 (recycleBinPanel)**:
| 元素 ID | 功能 |
|---|---|
| `emptyAllRecycleBtn` | 清空全部 |
| `chatRecycleList` + `chatRecycleCount` | 聊天文件回收站 |
| `labRecycleList` + `kbRecycleCount` | 知识库+技能回收站 |
| `companyRecycleList` | 公司知识库回收站 |
| `skillRecycleList` | 技能回收站 |
| `projectRecycleList` + `projectRecycleCount` | 项目文件回收站 |
| `folderRecycleList` | 文件夹回收站 |

**后端路由**:
| 路由 | 方法 | 功能 |
|---|---|---|
| `/get_recycle_bin` | GET | 获取回收站列表 (支持source筛选) |
| `/restore_from_recycle_bin` | POST | 恢复单项 |
| `/delete_recycle_item` | POST | 删除单项 |
| `/empty_recycle_bin` | POST | 清空回收站 |

### 标签页 5: 🧠 技能 (Skill Audit) — Admin/Auditor 可见

**侧边栏 (sidebar-audit-pane)**:
| 元素 ID | 功能 |
|---|---|
| `sidebarRunAuditBtn` | 运行分析 |
| `sidebarAuditQuickMergeBtn` | 快速合并 |
| `sidebarAuditQuickArchiveBtn` | 批量清理 |
| `sidebarAuditStats` | 审计统计 |

**主面板 (skillAuditPanel)**:
| 元素 ID | 功能 |
|---|---|
| `skillAuditContent` | 技能审计内容 |

**后端路由**:
| 路由 | 方法 | 功能 |
|---|---|---|
| `/admin/skill_audit` | GET | 获取审计概览 |
| `/admin/skill_merge` | POST | 合并技能 |
| `/admin/skill_archive/<skill_id>` | POST | 归档技能 |

### 标签页 6: 🔍 审核 (Review) — Admin/Auditor 可见

**侧边栏 (sidebar-review-pane)**:
| 元素 ID | 功能 |
|---|---|
| `sidebarIngestUploadBtn` + `sidebarIngestFileInput` | 上传文档包(ZIP) |
| `sidebarReviewStatus` | 待处理状态 |
| `sidebarViewStructuredBtn` | 查看结构化文档 |
| `sidebarViewWorkloadBtn` | 工作量统计 |

**主面板 (reviewPanel)** — 5个折叠区域:
| 元素 ID | 功能 | 后端路由 |
|---|---|---|
| `ingestDetails` + `ingestPanel` | 批量文档摄入 | POST `/admin/ingest/upload` |
| `trainingDetails` + `trainingExportPanel` | 训练数据 | GET `/admin/training_stats` |
| `ingestHistoryDetails` + `ingestHistoryPanel` | 摄入历史 | GET `/admin/ingest/stale_status` |
| `structuredDocsDetails` + `structuredDocsPanel` | 结构化文档 | GET `/admin/ingest/structured` |
| `workloadDetails` + `workloadPanel` | 审核工作量 | GET `/admin/ingest/review_workload` |

**摄入后端路由**:
| 路由 | 方法 | 功能 |
|---|---|---|
| `/admin/ingest/upload` | POST | 上传ZIP开始摄入 |
| `/admin/ingest/status/<task_id>` | GET | 查看摄入进度 |
| `/admin/ingest/domain_review` | GET | 待审核专业词汇 |
| `/admin/ingest/domain_approve` | POST | 批准词汇 |
| `/admin/ingest/domain_reject` | POST | 拒绝词汇 |
| `/admin/ingest/kb_review/<task_id>` | GET | 待审核知识库内容 |
| `/admin/ingest/kb_chunk/<task_id>/<idx>` | GET/POST | 查看/修正某一段 |
| `/admin/ingest/kb_approve/<task_id>` | POST | 批准知识库内容 |
| `/admin/ingest/kb_reject/<task_id>/<idx>` | POST | 拒绝某一段 |
| `/admin/ingest/stale_status` | GET | 过期审核状态 |
| `/admin/ingest/review_workload` | GET | 审核员工作量 |
| `/admin/ingest/structured` | GET | 结构化文档列表 |

### 标签页 7: 🗄️ 数据 (Database) — 可见但内容admin-gated

**侧边栏 (sidebar-db-pane)**:
| 元素 ID | 功能 |
|---|---|
| `sidebarDbTableSelect` | 表选择器 |
| `sidebarDbTableInfo` | 表信息 |
| `sidebarDbSchemaBtn` | 查看表结构 |
| `sidebarDbExportCsvBtn` | 导出CSV |
| `sidebarDbExportJsonBtn` | 导出JSON |
| `sidebarDbOverview` | 表概览 |

**主面板 (databasePanel)**:
| 元素 ID | 功能 |
|---|---|
| `dbTableSelect` | 表选择 |
| `dbSearchInput` + `dbSearchColumnSelect` | 搜索过滤 |
| `dbRefreshBtn` | 刷新 |
| `dbPerPageSelect` | 每页条数 (20/50/100) |
| `dbAutoRefreshToggle` + `autoRefreshTimer` | 自动刷新开关 |
| `dbDataTable` + `dbTableHeader` + `dbTableBody` | 数据表格 |
| `dbPagination` | 分页 |

**后端路由**:
| 路由 | 方法 | 功能 |
|---|---|---|
| `/admin/db_tables` | GET | 获取表列表 |
| `/admin/db_data` | GET | 分页查询数据 |
| `/admin/db_schema` | GET | 查看表结构 |
| `/admin/db_table_data` | POST | 查询表数据 |
| `/admin/db_update_row` | POST | 编辑行 (需PIN验证) |
| `/admin/db_delete_row` | POST | 删除行 (需PIN验证) |

### 标签页 8: 📊 总览 (Analytics) — 全员可见

**侧边栏 (sidebar-stats-pane)**:
| 元素 ID | 功能 | 后端路由 |
|---|---|---|
| `sidebarStatsActivity` | 活跃度统计 | GET `/admin/analytics` |
| `sidebarStatsSystem` | 系统资源 | — |
| `sidebarExportStatsBtn` | 导出统计报告 | — |
| `sidebarUserRoles` | 用户角色管理 (admin) | POST `/admin/role` |

**Admin Extras (admin-only)**:
| 元素 ID | 功能 | 后端路由 |
|---|---|---|
| `sidebarAuditLogBtn` | 审计日志 | — |
| `sidebarEditPromptBtn` | 编辑系统提示词 | GET/POST `/admin/system_prompt` |
| `sidebarWorkReportBtn` | 工作报告 | POST `/admin/generate_work_report` |
| `sidebarClearCacheBtn` | 清除缓存 | POST `/admin/clear_file_cache` |
| `sidebarCleanupNowBtn` | 清理数据 | POST `/admin/cleanup` |
| `sidebarSearchCacheBtn` | 搜索缓存配置 | — |
| `sidebarRagStatsBtn` | RAG索引统计 | — |
| `sidebarRagRebuildBtn` | 重建RAG索引 | — |
| `sidebarTrainingStatsBtn` | 训练统计 | GET `/admin/training_stats` |
| `sidebarTrainingExportBtn` | 导出JSONL | POST `/admin/training_export` |
| `sidebarSystemCleanupBtn` | 一键系统清理 | — |
| `sidebarClearAllDataBtn` | 清空全部数据 | — |

**主面板 (analyticsPanel)** — Admin折叠区域:
| 元素 ID | 功能 |
|---|---|
| `rcDetails` + `runtimeConfigContent` | 运行时配置 (50+参数) |
| `assetDetails` + `assetManager` | 资产管理器 |
| `archiveDetails` + `archivedSessionsAdmin` | 已归档会话管理 |
| `stylesDetails` + `styleManagerPanel` | 写作风格画像 |

### 弹窗 (Modals)
| 弹窗 ID | 功能 |
|---|---|
| `consentModal` | 注册/登录/匿名试用选择 (未登录时显示) |
| `fileStationModal` | 文件站 (文件保留3天) |
| `knowledgeBaseModal` | 知识库选择弹窗 |
| `accountModal` | 账户设置弹窗 |
| `searchModal` | 搜索聊天记录弹窗 |
| `msgContextMenu` | 右键菜单 (待办/引用/复制) |

---

## 二、后端隐藏功能（无前端入口或仅API）

### 账号系统 (auth_bp — 11个端点)
| 路由 | 方法 | 功能 |
|---|---|---|
| GET `/check_auth` | GET | 检查登录状态 (含DB回退恢复) |
| POST `/create_account` | POST | 注册新用户 (5-18位用户名 + 4/6位PIN) |
| POST `/login` | POST | 登录 (admin特殊处理) |
| POST `/update_account` | POST | 更新账号 (用户名/邮箱/PIN) |
| POST `/request_pin_change_code` | POST | 请求PIN修改验证码 (发邮箱) |
| POST `/set_email` | POST | 设置邮箱 |
| POST `/request_delete_account` | POST | 申请删除 (返回数据清单) |
| POST `/confirm_delete_account` | POST | 确认删除 (验证码+PIN) |
| POST `/submit_delete_choices` | POST | 提交保留/删除选择 |
| POST `/delete_account` | POST | 直接删除 (admin用) |
| POST `/logout` | POST | 登出 |

### 聊天系统 (chat_bp — 39个端点)
| 路由 | 方法 | 功能 |
|---|---|---|
| `/` | GET | 首页 |
| `/get_csrf_token` | GET | CSRF token |
| `/logout` | POST | 登出 |
| `/favicon.ico` | GET | 网站图标 |
| `/share_conversation` | POST | 分享对话 (生成token+URL) |
| `/shared/<token>` | GET | 查看分享的对话 |
| `/send_stream` | POST | **核心**: SSE流式对话 |
| `/send` | POST | 标准对话 (含judge review) |
| `/set_max_tokens` | POST | 设置token上限 |
| `/llm_providers` | GET | 获取可用AI模型列表 |
| `/llm_providers/set` | POST | 切换AI模型 |
| `/feedback` | POST | 消息评分 (1-5星) |
| `/get_recent_files` | GET | 最近文件列表 |
| `/load_cached_file` | POST | 加载缓存文件 |
| `/new_chat` | POST | 新建对话 |
| `/api/login` | POST | API登录 (JWT) |
| `/get_sessions` | GET | 获取会话列表 |
| `/load_session/<thread_id>` | GET | 加载会话 |
| `/delete_session/<thread_id>` | POST | 删除会话 |
| `/update_session_title` | POST | 更新会话标题 |
| `/archive_session/<thread_id>` | POST | 归档会话 |
| `/restore_session/<thread_id>` | POST | 恢复归档会话 |
| `/list_archived_sessions` | GET | 归档会话列表 |
| `/regenerate` | POST | 重新生成回答 |
| `/check_storage` | GET | 检查存储空间 |
| `/cleanup_now` | POST | 立即清理 |
| `/cleanup_anon_temp` | POST | 清理匿名临时文件 |
| `/set_image_analysis` | POST | 设置图片分析开关 |
| `/search_chat` | GET | 搜索聊天记录 |
| `/upload_file` | POST | 上传文件 |
| `/download_original_file` | POST | 下载原始文件 |
| `/fetch_url` | POST | 抓取URL内容 |
| `/delete_file_station` | POST | 删除文件站文件 |
| `/get_file_station` | GET | 获取文件站列表 |
| `/load_project_file` | POST | 加载项目文件 |
| `/get_recycle_bin` | GET | 获取回收站 |
| `/restore_from_recycle_bin` | POST | 恢复回收站项 |
| `/delete_recycle_item` | POST | 删除回收站项 |
| `/empty_recycle_bin` | POST | 清空回收站 |

### 项目文件高级操作 (admin_bp — 无直接前端按钮)
| 路由 | 方法 | 功能 |
|---|---|---|
| `/admin/projects/<id>/files/<file_id>` | DELETE | 删除项目文件 |
| `/admin/users` | GET | 列出所有用户 |
| `/admin/projects/<id>/all_users` | GET | 项目可用用户列表 |
| `/admin/archived_sessions` | GET/DELETE | 归档会话管理 |
| `/admin/archived_sessions/all` | DELETE | 删除全部归档 |
| `/admin/clear_file_cache` | POST | 清除文件缓存 |
| `/admin/task_deposit` | GET | 查看任务押金 |
| `/admin/task_deposit/transfer/<item_id>` | POST | 转移押金 |

### 训练数据系统 (knowledge_bp — 完整管线)
| 路由 | 方法 | 功能 |
|---|---|---|
| `/admin/training_stats` | GET | 训练数据统计 |
| `/admin/training_export` | POST | 导出 (full/incremental/quality/all/reset_watermark) |
| `/admin/training_export_history` | GET | 导出历史+水印+pending |
| `/admin/training_cleanup_stats` | GET | 清理预览 |
| `/admin/training_cleanup` | POST | 执行清理 |
| `/admin/training_health` | GET/POST | 健康检查/修复 |
| `/admin/training_health_history` | GET | 健康历史 |
| `/admin/training_exports_list` | GET | 导出文件列表 |
| `/admin/training_exports_cleanup` | POST | 清理旧文件 |
| `/admin/training_exports_delete/<name>` | POST | 删除单个文件 |
| `/admin/training_exports_download/<name>` | GET | 下载文件 |

### LoRA 微调系统 (knowledge_bp — 新)
| 路由 | 方法 | 功能 |
|---|---|---|
| `/admin/training/lora/datasets` | GET | 可用数据集列表 |
| `/admin/training/lora/adapters` | GET | 已训练适配器列表 |
| `/admin/training/run_lora` | POST | 启动LoRA微调 (返回task_id) |
| `/admin/training/lora/<industry>/activate` | POST | 激活适配器 |
| `/admin/training/lora/<industry>/deactivate` | POST | 停用适配器 |

### 企业信用查询 (credit_bp — 9个端点)
| 路由 | 方法 | 功能 |
|---|---|---|
| `/start_credit_check` | POST | 启动信用查询 (Edge+Selenium) |
| `/credit_check_status/<task_id>` | GET | 查询状态 |
| `/credit_check_resume/<task_id>` | POST | 恢复查询 |
| `/get_captcha_image/<task_id>` | GET | 获取验证码图片 |
| `/reload_captcha/<task_id>` | POST | 刷新验证码 |
| `/solve_captcha/<task_id>` | POST | 提交验证码 |
| `/download_credit_report/<task_id>` | GET | 下载报告 |
| `/list_credit_reports` | GET | 报告列表 |
| `/delete_credit_report/<id>` | POST | 删除报告 |

### 批量文件对比 (batch_bp — 5个端点)
| 路由 | 方法 | 功能 |
|---|---|---|
| `/compare_batch` | POST | 批量对比 (TF-IDF+语义+图片+属性) |
| `/export_batch_excel_download/<token>` | GET | 导出Excel |
| `/batch_result/<task_id>` | GET | 查看对比结果 |
| `/list_batch_results` | GET | 对比结果列表 |
| `/delete_batch_result/<id>` | POST | 删除结果 |

### 后台任务系统 (tasks_bp — 3个端点)
| 路由 | 方法 | 功能 |
|---|---|---|
| `/tasks` | GET | 最近的异步任务列表 |
| `/tasks/<task_id>` | GET | 任务状态查询 (轮询) |
| `/tasks/<task_id>/stream` | GET | SSE实时进度流 |

### 其他后端端点
| 路由 | 方法 | 功能 | 蓝图 |
|---|---|---|---|
| `/admin/role` | POST | 修改用户角色 (admin/auditor/user) | knowledge |
| `/admin/generate_work_report` | POST | 生成日报/周报/月报/年报 | knowledge |
| `/my_writing_style` | GET/POST | 查看/修改自己的风格 | knowledge |
| `/my_writing_style/analyze` | POST | 分析自己的风格 | knowledge |
| `/admin/user_styles` | GET | 所有用户风格列表 | knowledge |
| `/admin/user_styles/<user_id>` | GET/POST | 查看/编辑某用户风格 | knowledge |
| `/admin/user_styles/<user_id>/analyze` | POST | 分析某用户风格 | knowledge |
| `/admin/user_styles/<user_id>/delete` | POST | 删除风格画像 | knowledge |
| `/admin/user_styles/analyze_all` | POST | 批量分析所有用户 | knowledge |
| `/user_project_files` | GET | 用户项目文件列表 | projects |

---

## 三、后端服务层（36个文件）

| 文件 | 功能 |
|---|---|
| `admin_utils.py` | admin速率限制装饰器 + 审计日志记录 |
| `agent.py` | LangGraph Agent (DeepSeek V4 Pro) + get_date/bocha_search工具 + 搜索缓存(72h) |
| `analysis_prompts.py` | 招标分析提示词 + 工作报告提示词 |
| `anonymous.py` | 匿名用户管理 (UUID + 临时文件 + IP限制) |
| `auth_jwt.py` | JWT Token签发/验证 |
| `batch_compare_svc.py` | 批量对比服务 (TF-IDF + 文件预处理) |
| `context_utils.py` | 项目上下文聚合 (文件+RAG+技能+记忆+工作流+去重) |
| `credit_checker.py` | 企业信用查询 (Edge + Selenium爬虫) |
| `file_cache.py` | 文件处理结果缓存 |
| `file_generator.py` | 生成.docx/.xlsx文件 |
| `file_processing.py` | 文件解析/预处理/TF-IDF/关键词提取/语义相似度 |
| `ingest_pipeline.py` | 批量文档摄入 (ZIP→OCR→三路管道: 领域词/知识库/技能) |
| `judge_review.py` | 双模型审查 (第二个LLM打分0-10) |
| `kb_skill_engine.py` | 知识库技能提取引擎 |
| `langutils.py` | 语言检测 (zh/en, 基于CJK字符比例) |
| `llm_provider.py` | 4个LLM provider + 行业模型路由 + LoRA适配器自动发现 |
| `lora_trainer.py` | LoRA训练启动器 (子进程 + task_bus进度) |
| `notebook.py` | 个人笔记本 (Markdown + AI摘要 + 语义搜索) |
| `ocr.py` | EasyOCR封装 (中文+英文) |
| `prompt_safety.py` | 12层安全防护 (注入检测/内容隔离/JSON容错/Judge解析/Token预算/Markdown验证/VL交叉检查) |
| `rag_engine.py` | ChromaDB RAG检索引擎 |
| `review_logger.py` | 审核操作日志 |
| `runtime_config.py` | 50+项运行时配置 (factory→runtime→env三层加载) |
| `semantic.py` | 多模型语义相似度 (bge-zh/para-multi/distiluse，语言自动切换) |
| `session_manager.py` | 会话管理 |
| `skill_auditor.py` | 技能审计 (自动提取+去重+归档) |
| `skill_validator.py` | 技能文件规范验证 (21个技能模块) |
| `style_engine.py` | 写作风格分析 (70%新+30%旧迭代融合) |
| `task_bus.py` | Redis pub/sub进度总线 (通用，Celery→Flask→SSE→浏览器) |
| `task_locking.py` | 任务锁 (防重复提交) |
| `text_utils.py` | 文本工具 |
| `training_logger.py` | 训练数据采集+导出+健康检查+清理+修复 (1096行) |
| `vl_model.py` | 视觉语言模型 (qwen3-vl-plus, DashScope API) |
| `web_extractor.py` | URL内容抓取 |
| `workflow_engine.py` | 行业工作流引擎 (招标代理/工程造价/工程审计) |

---

## 四、基础设施

| 组件 | 详情 |
|---|---|
| Web框架 | Flask 3.1.3 + Flask-Session + Flask-WTF |
| 数据库 | PostgreSQL 16 (psycopg2连接池 min=1 max=20) |
| 缓存/队列 | Redis 7 (会话 + Celery broker + pub/sub进度总线) |
| 异步队列 | Celery 5.4.0 (Docker) / APScheduler 3.11.2 (standalone) |
| WSGI | Gunicorn 23.0.0 + gevent (Docker) / Flask dev (standalone) |
| 反向代理 | Nginx (Docker) — HTTP→HTTPS，静态文件 |
| AI模型 | DeepSeek V4 Pro (主agent) + Zhipu GLM-4 + Qwen 3.7 + SiliconFlow Qwen2.5-7B/72B |
| Embedding | bge-large-zh-v1.5 (1024-dim) + paraphrase-multilingual (384-dim) + distiluse (512-dim, 旧版回退) |
| VL模型 | qwen3-vl-plus-2025-12-19 (DashScope API) |
| RAG | ChromaDB + sentence-transformers (chunk=500, overlap=100, top_k=8) |
| OCR | EasyOCR (中文+英文, CPU) |
| NLP | jieba 0.42.1 + scikit-learn 1.9.0 + numpy 2.5.0 |
| 搜索 | Bocha API (72h缓存) |
| 监控 | prometheus-flask-exporter |
| 安全 | CSRF + JWT + PIN认证 + 速率限制 + 12层prompt安全 |
| 容器化 | Docker (flask + celery-worker + celery-beat + redis + postgres + nginx) |
| API文档 | flasgger (Swagger UI at /apidocs) |

---

## 五、自动调度任务（共22个）

### Celery Beat (Docker模式 — 4个)
| 任务 | 频率 |
|---|---|
| cleanup_stale_sessions | 每小时 |
| cleanup_temp_files | 每小时 |
| run_skill_audit | 每周 |
| generate_weekly_report | 每周 |

### APScheduler (Standalone模式 — 18个)
| 任务 | 频率 |
|---|---|
| cleanup_old_sessions | 每小时 |
| delete_expired_original_files | 每小时 |
| cleanup_stale_tasks | 每小时 |
| cleanup_stale_message_responses | 每小时 |
| cleanup_old_anon_temp_files | 每小时 |
| schedule_project_deletion_cleanup | 每日 |
| cleanup_expired_recycle_bin | 每小时 |
| cleanup_expired_share_files | 每小时 |
| cleanup_stale_download_tokens | 每小时 |
| cleanup_orphan_users | 每周 |
| cleanup_old_training_data | 每季度 (Jan/Apr/Jul/Oct 1 04:00) |
| cleanup_old_training_exports | 每季度 (同日期 04:30) |
| auto_generate_weekly_report | 每周 |
| auto_generate_monthly_report | 每月 |
| auto_generate_annual_report | 每年 |
| auto_rag_health_check | 每日 |
| auto_training_health_check | 每周日 03:30 |
| auto_cleanup_stale_reviews | 每日 |

---

## 六、数据库表一览（30+张）

### 用户与认证
| 表名 | 关键列 |
|---|---|
| `users` | user_id, username, pin_hash, email, role, is_auditor, is_active, deletion_requested |
| `user_consents` | user_id, consent_value, consent_date |

### 聊天
| 表名 | 关键列 |
|---|---|
| `chat_sessions` | id, user_id, thread_id, title, project_id, created_at, updated_at |
| `chat_messages` | id, thread_id, role, content, thinking, timestamp |
| `archived_sessions` | thread_id, user_id, archive_path, archived_at |
| `anonymous_sessions` | anon_id, thread_id, created_at |

### 文件
| 表名 | 关键列 |
|---|---|
| `user_files` | id, user_id, thread_id, filename, content, file_hash, expires_at, meta_data |
| `project_files` | id, project_id, folder_id, original_name, content, skill_summary, file_hash, version |
| `knowledge_lab_files` | id, user_id, filename, content, skill_summary, is_company |
| `company_knowledge_base` | id, filename, content, category |
| `image_description_cache` | file_hash, description, created_at |
| `file_usage` | id, user_id, thread_id, filename, usage_type |
| `file_text_cache` | (文本缓存) |

### 项目
| 表名 | 关键列 |
|---|---|
| `projects` | id, name, created_by, industry, created_at, status |
| `project_members` | project_id, user_id, role, last_read_at, permissions |
| `project_ai_memory` | id, project_id, user_id, role, content, content_md, created_at |
| `project_todos` | id, project_id, user_id, message_id, content_copy, status, done_at |
| `message_quotes` | id, project_id, quoted_message_id, quoting_message_id, parent_quote_id, thread_id |
| `regen_votes` | id, project_id, message_id, original_content, new_content, status, round, expires_at |
| `regen_vote_ballots` | id, vote_id, voter_id, vote, cast_at |
| `member_workflows` | project_id, user_id, workflow_data |
| `project_folders` | id, project_id, name, parent_id |

### 技能
| 表名 | 关键列 |
|---|---|
| `skills` | id, skill_name, skill_content, source_file_id, source_type, version |
| `skill_audit_results` | id, audit_batch_id, skill_id, score, issue_type, detail |
| `kb_skills` | id, file_id, skill_content, extracted_at |

### 审核与摄入
| 表名 | 关键列 |
|---|---|
| `ingest_tasks` | id, task_id, status, uploaded_by, file_count, progress |
| `domain_words` | word, source_file, approved, approved_by |
| `domain_words_review` | word, source_file, status |

### 回收与审计
| 表名 | 关键列 |
|---|---|
| `recycle_bin` | id, original_table, original_id, data_snapshot, deleted_by, deleted_at |
| `admin_audit_log` | id, admin_user_id, action, table_name, row_id, old_values, new_values |

### 其他
| 表名 | 用途 |
|---|---|
| `credit_reports` | 企业信用报告 |
| `download_tokens` | 文件下载token |
| `share_files` | 文件分享 |
| `batch_results` | 批量对比结果 |
| `celery_taskmeta` | Celery任务元数据 (自动) |
| `celery_tasksetmeta` | Celery任务集元数据 (自动) |

---

> **生成日期**: 2026-07-01 | **来源**: 完整代码审计 (36个服务 × 9个路由文件 × HTML + JS)
