# 中联招标智能助手 — 用户操作手册

> 中联国际 AI 智能服务平台  
> 版本: 2026-07-01 | 适用角色: 管理员 · 审核员 · 普通用户

---

## 目录

- [1. 产品介绍](#1-产品介绍)
- [2. AI 安全与可靠性体系](#2-ai-安全与可靠性体系)
- [3. 系统核心能力一览](#3-系统核心能力一览)
- [4. 管理员操作手册](#4-管理员操作手册)
  - [4.1 使用统计面板](#41-使用统计面板)
  - [4.2 运行时配置](#42-运行时配置)
  - [4.3 资产管理器](#43-资产管理器)
  - [4.4 归档会话管理](#44-归档会话管理)
  - [4.5 写作风格画像](#45-写作风格画像)
  - [4.6 数据库视图](#46-数据库视图)
  - [4.7 搜索缓存配置](#47-搜索缓存配置)
  - [4.8 系统提示词](#48-系统提示词)
  - [4.9 AI 智能压缩与双模型审查](#49-ai-智能压缩与双模型审查)
  - [4.10 文件审计](#410-文件审计)
  - [4.11 系统清理](#411-系统清理)
  - [4.12 CEO / COO 管理员账户](#412-ceo--coo-管理员账户)
  - [4.13 后台任务面板](#413-后台任务面板)
  - [4.14 训练数据管理](#414-训练数据管理)
- [5. 审核员操作手册](#5-审核员操作手册)
  - [5.1 批量文档摄入](#51-批量文档摄入)
  - [5.2 领域词库审核](#52-领域词库审核)
  - [5.3 知识库内容审核](#53-知识库内容审核)
  - [5.4 结构化文档查看](#54-结构化文档查看)
  - [5.5 训练数据管理与 LoRA 微调](#55-训练数据管理与-lora-微调)
  - [5.6 审核员工作量统计](#56-审核员工作量统计)
  - [5.7 技能审核](#57-技能审核)
  - [5.8 过期审核提醒与自动清理](#58-过期审核提醒与自动清理)
  - [5.9 技能文件规范验证](#59-技能文件规范验证)
- [6. 普通用户操作手册](#6-普通用户操作手册)
  - [6.1 AI 智能对话](#61-ai-智能对话)
  - [6.2 账户设置](#62-账户设置)
  - [6.3 写作风格分析](#63-写作风格分析)
  - [6.4 项目管理](#64-项目管理)
  - [6.5 项目 AI 协作助手](#65-项目-ai-协作助手)
  - [6.6 行业标准工作流](#66-行业标准工作流)
  - [6.7 知识库](#67-知识库)
  - [6.8 回收站](#68-回收站)
  - [6.9 生成日报](#69-生成日报)
  - [6.10 个人笔记本](#610-个人笔记本)
  - [6.11 任务规划命令](#611-任务规划命令)
  - [6.12 AI 自我审查命令](#612-ai-自我审查命令)
- [7. 接口地址参考](#7-接口地址参考)
- [8. 附录](#8-附录)
  - [8.1 数据库表一览](#81-数据库表一览)
  - [8.2 自动调度任务一览](#82-自动调度任务一览)
  - [8.3 数据文件目录](#83-数据文件目录)
  - [8.4 基础设施详情](#84-基础设施详情)
  - [8.5 技术架构一览](#85-技术架构一览)

---

## 1. 产品介绍

### 我们解决什么问题

在招标代理行业，每天面对的是成百上千页的技术标书、复杂的合规要求、繁重的文档起草工作，以及团队协作中信息不对称带来的重复劳动。传统的"人工逐页审读 + Word 模板套用"模式效率低下、容易出错、知识难以沉淀。

**中联招标智能助手**正是为此而生——它不是又一个聊天机器人，而是一个**深植于招标代理工作流的 AI 协作平台**，将大语言模型（LLM）的能力与行业专业知识、企业知识库、团队协作机制深度整合。

### 我们是谁

中联招标智能助手是面向招标代理机构、工程造价咨询公司、工程审计机构的 **AI 赋能型业务操作系统**。它将 AI 嵌入到投标分析、文档起草、知识管理和团队协作的每一个环节，让每个业务人员都拥有一个永不疲倦的"AI 专家同事"。

### 我们如何与众不同

#### 🧠 不是通用 AI，是专业 AI

普通 AI 聊天工具只能做泛泛的问答。我们的系统内置了 **行业标准工作流**（招标代理、工程造价、工程审计三大领域）、**公司知识库检索增强生成（RAG）** 和 **AI 技能引擎**——从文件中自动提取的框架、原则和技术方法。同时支持 **行业级 LoRA 微调模型**，用公司积累的高质量对话持续训练开源模型，让 AI 的回答风格和对行业术语的理解越来越精准。

#### 👥 不是单机工具，是协作平台

传统 AI 工具是一对一的：你问、AI 答。我们的系统支持**项目级多人共享 AI 上下文**——同一个投标项目里，所有人的 AI 会话互相可见，AI 记住每个人说过什么、做过什么分析。项目内还支持**待办事项**、**引用树**（可引用别人/自己的消息建立对话链）、**语义差异投票**（AI 回答变化较大时自动触发投票，决定保留哪个版本）。

#### 🛡️ 不是"黑箱"，是可信 AI

普通大模型会"一本正经地胡说八道"——编造不存在的引用、虚构统计数据、混淆相似文件。我们构建了 **12 层 AI 安全防护体系**（详见第 2 章），从输入净化到输出校验，从上下文去重到多模态交叉验证，系统性降低幻觉率。

#### 📊 不是孤立输出，是全流程闭环

从文档摄入 → 知识提取 → 技能生成 → AI 分析 → 文档生成 → 报告输出 → 训练数据沉淀 → LoRA 微调 → 模型部署，形成完整的知识循环。每一次 AI 对话都变成可复用的资产。

### 谁在使用

| 角色 | 典型场景 |
|---|---|
| **投标专员** | 用 AI 起草投标函、技术方案；批量对比多家投标文件，发现雷同风险 |
| **项目经理** | 在项目 AI 助手中协调团队，追踪每个成员的工作流进度和 KPI |
| **审核员** | 批量摄入扫描文档，审核 AI 提取的专业词汇和知识片段，管理训练数据 |
| **部门负责人** | 一键生成日报/周报/月报，查看团队 AI 使用数据，掌握工作效率 |
| **管理员** | 配置 AI 模型、管理用户资产、审计文件完整性、清理系统垃圾、启动 LoRA 微调 |

### 前后端功能一览

系统共有 **8 个前端标签页** + **后台 18 个自动调度任务** + **50+ 项可调参数**（运行时配置）。

| 标签页 | 可见性 | 核心功能 |
|---|---|---|
| 对话 | 全员 | 流式聊天、文件分析、批量对比、VL 图片分析、分享对话、导出 MD |
| 项目 | 全员 | 项目文件管理、AI 协作助手、行业工作流、待办/引用/投票、企业信用查询 |
| 知识库 | 已登录 | 个人/公司知识库、RAG 检索、技能总览、个人笔记本 |
| 回收站 | 全员 | 分来源筛选、批量恢复/清空、自动过期清理 |
| 技能审计 | Admin/Auditor | 审计分析、技能合并、批量清理、统计追踪 |
| 审核 | Admin/Auditor | 批量文档摄入(OCR)、领域词审核、知识库片段审核、结构化文档 |
| 数据工具 | Admin | 表浏览、CSV/JSON 导出、数据编辑 |
| 总览 | 全员 | 使用统计、系统资源、Admin Extras（审计日志/提示词/RAG/训练/清理） |

---

## 2. AI 安全与可靠性体系

本系统实现了业界领先的 **12 层 LLM 幻觉防护架构**，覆盖从用户输入到 AI 输出的全链路。每一层独立运作、互为补充，确保 AI 输出可靠、可追溯、可审计。

### 防护架构图

```
用户输入 / 数据库内容
        │
        ▼
  ① 注入模式检测与剥离   ← 识别并过滤 [SYSTEM OVERRIDE] 等攻击模式
        │
        ▼
  ② XML 标签安全隔离     ← <USER_QUERY>...</USER_QUERY> 边界标记
        │
        ▼
  ③ 同名实体去重         ← 同名文件/技能追加 #短hash 后缀区分
        │
        ▼
  ④ Token 预算管理       ← 5 段上下文按比例动态分配，防止中间丢失
        │
        ▼
  ⑤ 知识库优先级规则     ← RAG > 用户文件 > 内部知识，冲突时明确告知
        │
        ▼
  ⑥ 安全约束自动注入     ← 每个 system prompt 追加 4 条防幻觉指令
        │
        ▼
  ⑦ 多模态交叉验证       ← VL 图片描述用两个不同 prompt 交叉比对数值
        │
        ▼
  ⑧ 链式调用事实锚定     ← 工作流审查必须标注段落位置，禁止编造问题
        │
        ▼
  ⑨ 引用真实性约束       ← 只能引用对话历史中确实存在的 @某人 发言
        │
        ▼
  ⑩ 结构化输出容错       ← JSON/评审格式自动纠错 + 失败重试
        │
        ▼
  ⑪ 文档结构预检         ← Markdown 语法检查 + 自动修复未闭合代码块
        │
        ▼
  ⑫ 双模型交叉审查       ← 第二个 AI 模型独立审查回答质量（可选开启）
```

### 各层详解

| 层 | 防护名称 | 防止什么幻觉 | 工作方式 |
|---|---|---|---|
| ① | **Prompt 注入防御** | 用户上传文件中夹带的"忽略之前指令"等攻击文本 | 正则匹配 + 日志告警 + 超长内容截断 |
| ② | **内容安全隔离** | LLM 将用户数据误当作系统命令执行 | 所有用户/数据库内容包裹在 `<TAG>...</TAG>` XML 标签中 |
| ③ | **同名实体去重** | 两个都叫"技术方案.docx"的文件被 LLM 混淆 | 同名文件追加 `#a3f2` 短 hash 后缀，LLM 可精确指代 |
| ④ | **上下文预算管理** | 超长 prompt 导致关键信息被 LLM 忽略（中间丢失效应） | 5 段上下文按比例动态分配 7000 token，保底每段 200 token |
| ⑤ | **知识库优先级** | RAG 检索为空时 LLM 自行编造知识 | 空结果显式告知"知识库暂无相关内容"，5 级优先级规则 |
| ⑥ | **安全指令注入** | LLM 编造统计数据、日期、金额等不存在的事实 | `call_llm()` 自动在每条 system prompt 末尾追加 4 条约束 |
| ⑦ | **VL 交叉验证** | 视觉模型看错表格数字，导致下游分析全盘错误 | 同一图片两次不同角度描述 → 数值差异检测 → 不一致时标记 ⚠️ |
| ⑧ | **工作流事实锚定** | 审查步骤无中生有指出"缺少第X节"，修订步骤强行添加 | 审查必须标注段落位置；修订只改审查指出的问题 |
| ⑨ | **引用真实性约束** | AI 引用"@张三 之前分析过…"但张三从未说过 | 强制约束：只有对话历史中存在的发言才能被引用 |
| ⑩ | **结构化输出容错** | JSON 包裹在 ` ```json ` 中、评审字段名漂移 | 自动去 fence → 修复尾逗号 → 失败重试一次；评审用模糊正则容错 |
| ⑪ | **文档结构校验** | AI 生成的 Markdown 表格缺分隔行、代码块未闭合 | `generate_file()` 入口预检 + 自动修复未闭合代码块 |
| ⑫ | **双模型审查** | 单模型自身错误难以自查 | 第二个不同品牌模型独立评分（1-10），不合格的回答自动替换 |

**实现文件**：`app/services/prompt_safety.py`（7 个安全函数）、`app/services/judge_review.py`（第⑫层）、`app/services/context_utils.py`（第③层去重）、`app/services/vl_model.py`（第⑦层交叉验证）。

### 安全能力覆盖

| 能力 | 传统 LLM 应用 | 中联招标智能助手 |
|---|---|---|
| 幻觉率控制 | 5%-22% 无防护 | 多层拦截 + 自动纠错 |
| 注入攻击防御 | 无 | ① 模式检测 + ② 标签隔离 |
| 视觉模型验证 | 单次调用 | ⑦ 双重描述交叉比对 |
| 长文档可靠性 | 中间丢失 | ④ 动态 token 预算分配 |
| 同名文件区分 | LLM 自行猜测 | ③ #短hash 精确指代 |
| 结构化输出 | 手动检查 | ⑩ 自动解析 + 失败重试 |
| 引用可追溯 | 无法验证 | ⑨ 必须来自实际对话 |

---

## 3. 系统核心能力一览

**中联招标智能助手**是一套专门为招标代理行业打造的 AI 辅助工作平台，具备以下核心能力：

| 功能 | 说明 | 版本 |
|---|---|---|
| 🤖 AI 智能对话 | 接入 DeepSeek V4 Pro、智谱 GLM-4、通义千问、硅基流动等多家大模型，支持 Agent 工具调用 | v1 |
| 🌐 联网搜索 | 需要实时信息时自动搜索（Bocha API），结果缓存 72 小时复用 | v1 |
| 📄 文件分析 | 上传 PDF/Word/Excel/PPT/图片，AI 自动识别内容并回答 | v1 |
| 📊 批量文件对比 | 多文件相似度矩阵分析（TF-IDF + 语义 + 图片 + 属性）+ AI 专业围串标风险研判 | v1 |
| 🏢 企业征信查询 | 自动查询公开信用信息（Edge + Selenium 爬虫） | v1 |
| 📚 知识库检索增强生成 | ChromaDB RAG 检索引擎，支持个人/公司/项目三级知识库 | v1 |
| 📝 工作报告生成 | 一键日报/周报/月报/年报，自动总结工作内容 | v1 |
| 📥 文档批量摄入 | ZIP 包上传 → OCR → 三路管道（领域词/知识库/技能） | v1 |
| ✍️ 写作风格画像 | 学习个人文风，AI 输出更贴合用户习惯（迭代进化，70% 新 + 30% 旧融合） | v1 |
| 🎓 训练数据管理 | 自动采集 → 质量筛选 → 增量/全量 JSONL 导出 → 健康检查 → 自动修复 → 过期清理 | v1 |
| 🧪 LoRA 模型微调 | Unsloth 执行器：导出 JSONL → 训练开源模型 → 注册行业路由 → 自动部署 | 🆕 |
| 👥 **项目 AI 协作助手** | 同一项目多人共享 AI 上下文，AI 记住所有成员对话，身份标签引用；支持待办、引用树、语义差异投票 | 🆕 |
| 🏭 **行业标准工作流** | 内置招标代理/工程造价/工程审计三大行业工作流，指导 AI 输出 | 🆕 |
| 📑 **AI 文件生成** | AI 输出一键导出为 Word(.docx) / Excel(.xlsx) / PPT(.pptx) | 🆕 |
| 🔄 **成员工作流与 KPI** | 项目成员自定义步骤 + 生成次数/修改轮次/输出字数 KPI 追踪 | 🆕 |
| 🛡️ **12 层幻觉防护** | 注入防御、内容隔离、同名去重、token 预算、VL 交叉验证等 | 🆕 |
| 🔍 **文件审计** | 管理员一键扫描数据库记录 vs 磁盘文件，发现孤儿文件和存储泄漏 | 🆕 |
| 🧹 **系统清理** | 一键清理过期会话、临时文件、内存泄漏、过期技能 | 🆕 |
| 👔 **CEO/COO 管理员** | 多管理员账户，CEO 和 COO 共享 PIN 码统一管理 | 🆕 |
| 🗑️ **增强回收站** | 分来源折叠展示，技能摘要归档留存，到期自动清理 | 🆕 |
| 📓 **个人笔记本** | Markdown 笔记 + AI 摘要 + 语义搜索 + 自动接入知识库 | v1 |
| 🔄 **语义差异投票** | AI 回答大幅变化时自动触发投票（24h 有效期，语义三路融合检测 + 数值/否定词变化感知） | 🆕 |
| 💬 **引用对话链** | 项目内右键引用任意消息，构建对话引用树，AI 自动感知引用上下文 | 🆕 |
| ✅ **待办追踪** | 项目内右键添加待办，最多 5 条，完成/删除操作，完成记录写入 AI 记忆 | 🆕 |
| ⚡ **后台任务面板** | Celery 异步任务进度 SSE 实时推送，Redis pub/sub 通用总线，可复用于所有长任务 | 🆕 |
| 🤖 **多语言智能语义** | bge-large-zh-v1.5 (中文) / paraphrase-multilingual (多语言) 自动语言检测切换，数值变化感知 | 🆕 |

---

## 4. 管理员操作手册

管理员拥有系统的全部权限，包括修改配置、管理用户数据、启动 LoRA 训练、查看训练数据、管理写作风格等。系统支持多个管理员账户（详见 4.12 节）。

### 4.1 使用统计面板

**怎么打开**: 点击页面顶部的 `📊 总览` 标签

页面顶部会显示一串关键数字：
```
👥总用户数 · 🟢最近24小时活跃人数 · 💬总会话数 · ✉️总消息数 · 💾占用存储空间 · 🔍征信查询次数 · 📂活跃项目数 · 最近7天消息量趋势图
```

往下滚动可以展开更多管理功能面板。

### 4.2 运行时配置

**怎么打开**: 总览面板 → Runtime Config

这里可以随时调整系统的各种运行参数。修改任何参数后，记得点 `💾 Save All Changes` 保存。页面上方会显示 `● modified` 提示你有未保存的修改。

**共 50+ 项可调参数**，以下为常用重要参数：

| 分类 | 参数名 | 默认值 | 这是干什么的 |
|---|---|---|---|
| AI 模型设置 | 请求超时时间 | 120秒 | AI 回答问题允许的最长等待时间 |
| | 默认输出长度 | 1600 Token | AI 每次回答的大致字数上限（100-4800 可调） |
| | 当前使用的模型 | 自动选择 | 可选 DeepSeek/智谱/通义千问/硅基流动 |
| | 当前使用的 Provider | 自动选择 | 可选 deepseek/zhipu/qwen/siliconflow |
| 搜索与缓存 | 搜索结果缓存有效期 | 72小时 | 网上搜到的结果存多久，到期自动刷新 |
| RAG 知识库 | 文本分块大小 | 500字 | 把大文件切成小块便于检索，每块多大 |
| | 分块重叠 | 100字 | 相邻文本块之间的重叠字数 |
| | 默认返回 Top-K | 8 | 每次检索返回最相关的几条结果 |
| | 最大上下文长度 | 8000字 | RAG 上下文最大字符数 |
| VL 视觉模型 | 最大图片尺寸 | 1024px | 发送给 VL 模型的图片最大宽度 |
| | JPEG 质量 | 85% | 图片压缩质量 |
| Headroom | 智能压缩开关 | 开启 | 把文件内容/搜索结果/聊天历史精简后再发给 AI |
| Judge Review | 双模型审查开关 | 关闭 | 用另一个 AI 模型审查回答质量（需 ≥2 个 Provider） |
| 文件缓存 | 缓存有效期 | 24小时 | 上传文件的处理结果缓存多久 |
| | 最大缓存文件数 | 10 | 同时缓存多少个文件的处理结果 |
| 文件处理 | 模板相似度阈值 | 0.85 | 文件对比时多高算"模板相似" |
| | 关键词 Top-K | 20 | 提取多少个关键词 |
| 自动清理 | 旧会话保留天数 | 15天 | 超过这个天数的旧聊天记录自动删除 |
| | 项目删除等待期 | 30天 | 标记删除的项目等多少天后真正删除 |
| | 回收站保留天数 | 3天 | 文件在回收站保留多久 |
| | 匿名临时文件保留 | 1天 | 未登录用户的临时文件保留多久 |
| | 分享文件有效期 | 7天 | 分享链接的有效期 |
| 速率限制 | admin 操作频率 | 5次/30分钟 | 管理员操作速率限制 |
| | 征信查询频率 | 10次/5分钟 | 企业征信查询速率限制 |
| 匿名用户 | 最大文件数 | 5个 | 未登录用户最多上传几个文件 |
| | 最大文件大小 | 5MB | 未登录用户单个文件限制 |
| | 消息长度限制 | 10000字符 | 未登录用户每句话最长多长 |
| 训练数据 | 最低评分要求 | 3星 | 导出训练数据时，只保留评分 ≥3 星的对话 |
| | 最低回答长度 | 100字 | 导出时跳过回答过短的对话 |
| | 训练数据保留天数 | 90天 | 超过这个天数的训练数据自动清理 |
| | 导出文件保留数量 | 20个 | 只保留最近 20 个导出的训练数据文件 |
| 报告 | 最低消息数 | 5条 | 生成日报至少需要多少条消息 |
| | 报告保留天数 | 90天 | 生成的报告保留多久 |

**出厂预设**：
- 如果你对当前所有的配置都很满意，可以点击 `🏭 Save as Factory Presets` 保存为出厂基准。保存后这些配置就变成了"官方标准"，不会被随意修改。
- 以后想恢复出厂设置，点 `↩ Restore Factory Presets` 一键还原。
- AI 模型选择（`active_llm_provider` / `active_llm_model`）不受出厂预设影响，可以单独调整。

### 4.3 资产管理器

**怎么打开**: 在"总览"面板中 → 展开 `📦 Asset Manager`

- 查看所有用户的数字资产（聊天记录、上传的文件、知识库内容、征信报告等）
- 搜索某个用户 → 勾选要转移的数据 → 选择接收人 → 点击"转移"
- 查看"孤数据托管区"（已删除账号留下的数据，可以转给别的用户）
- 批量将押金转移到指定用户名下

### 4.4 归档会话管理

**怎么打开**: 在"总览"面板中 → 展开 `📁 Archived Sessions`

- 查看所有已归档的聊天会话（普通用户看不到这里的列表）
- 把归档会话恢复到活跃状态，用户可以继续对话
- 永久删除不需要的归档会话

### 4.5 写作风格画像

**怎么打开**: 在"总览"面板中 → 展开 `✍️ Writing Style Profiles`

系统会分析每个用户的聊天记录，提取他们的写作风格（比如习惯说长句还是短句、正式还是随意、喜欢用哪些词）。

- 这里显示所有用户的风格画像列表
- 点 `🔄 Analyze All` 可以一次性分析所有用户
- 每个用户可以单独点 `🔄` 重新分析，或点 `🗑` 删除画像
- 风格是"迭代进化"的——每次分析会保留 70% 新结果 + 30% 旧结果，慢慢变得越来越准

### 4.6 数据库视图

**怎么打开**: 点击页面顶部的 `🗄️ 数据` 标签

- 选择一个数据库表 → 查看里面的所有数据
- 支持搜索过滤、翻页浏览
- 可以把数据导出为 CSV 表格文件或 JSON 格式
- 查看表结构
- 编辑或删除某一行数据（需要输入 PIN 码验证身份）

系统共有 **30+ 张表**，完整列表见 [附录 8.1](#81-数据库表一览)。

### 4.7 搜索缓存配置

**怎么打开**: 左边侧边栏 → 总览标签 → `⚙️ 搜索缓存配置`

- 查看当前缓存了多少条搜索结果，有效期是几小时
- 可以修改缓存有效期（设成 0 就是关闭缓存，每次搜索都去网上查）
- 点 `🗑️ 清除全部缓存` 可以立刻清空所有缓存的搜索结果

### 4.8 系统提示词

**怎么打开**: 左边侧边栏 → 总览标签 → `✏️ 编辑系统提示词`

系统提示词就是告诉 AI "你是谁、该怎么做"的一段指令。修改后下一次对话就会生效。系统会自动将最新的安全约束追加到你自定义的提示词末尾，无需手动添加。提示词保存到 `data/agent_prompt.json` 文件中。

### 4.9 AI 智能压缩与双模型审查

**怎么打开**: 总览 → Runtime Config → 搜索与缓存 / AI 模型设置

**智能压缩**（参数名 `headroom_enabled`，默认开启）：
- 在 AI 处理你的问题之前，系统会先把文件内容、搜索结果、聊天历史"精简"一下，去掉不重要的部分
- 可以节省 60% 到 95% 的 Token 消耗
- 答案的质量不会降低，但处理速度会变快，费用会降低
- 触发条件：消息 >800 字符或 prompt >3000 字符时自动
- 依赖：`headroom-ai` 包（可选安装）
- 建议一直开着，不需要关

**双模型审查**（参数名 `judge_review_enabled`，默认关闭）：
- 开启后，AI 回答完你的问题，系统会用第二个不同的 AI 模型再审查一遍回答是否靠谱
- 第二个模型会打分（1-10分），判定是"通过"、"需要改进"还是"不合格"
- 如果判定"需要改进"，系统会自动用修正后的答案替换原来的回答
- 这个功能需要至少配置了两个不同品牌的 AI 服务（比如 DeepSeek + 智谱），只有一个 AI 的话用不了
- 默认关闭是为了给大家省钱，有需要再开
- **实现文件**：`app/services/judge_review.py` — `review_response()` 函数

### 4.10 文件审计

**怎么打开**: 在"总览"面板中 → 展开 `🔍 File Audit`

文件审计功能扫描整个系统的文件完整性，帮你发现三类问题：

- **孤儿文件**（Orphan Files）：磁盘上有文件，但数据库里没有对应记录——说明文件"丢了户口"，可能是上传中断或删除不全留下的。
- **存储泄漏**（Storage Leak）：数据库记录指向的文件在磁盘上不存在——说明文件被意外删除或磁盘损坏，用户访问时会报错。
- **重复文件**：相同内容 MD5 的文件存了多份，白白浪费存储空间。

点 `🔍 运行文件审计` 开始扫描。结果会列出每种问题的数量和具体文件名，方便你逐个处理。

### 4.11 系统清理

**怎么打开**: 在"总览"面板中 → 展开 `🧹 System Cleanup`

一键清理系统中的各种过期数据和资源泄漏：

- **过期会话**：超过配置天数的闲置聊天记录（默认 15 天）
- **临时文件**：上传处理后残留的临时文件（超过 24 小时）
- **内存泄漏**：孤立的 AI 记忆记录（超过 1 小时未被引用的 memory）
- **过期技能**：超过保留期仍未被使用的技能缓存

点 `🧹 运行系统清理` 一键执行。系统有定时任务自动运行这些清理（详见 [附录 8.2](#82-自动调度任务一览)），这里主要是手动触发用的。

### 4.12 CEO / COO 管理员账户

系统在首次启动时会自动创建两个高级管理员账户：

| 用户名 | 角色 | 权限 |
|---|---|---|
| `CEO` | 超级管理员 | 所有权限，包括清空全部数据 |
| `COO` | 运营管理员 | 配置管理 + 数据审计 + 用户管理 |

两个账户共享同一个 PIN 码（在 `.env` 文件中配置 `ADMIN_PIN`）。登录时使用 `CEO` 或 `COO` 作为用户名即可。这与原有的 `admin` 账户并存，三者权限基本一致，只是 `CEO`/`COO` 的命名更符合企业组织架构习惯。

### 4.13 后台任务面板

**怎么打开**: 左侧侧边栏 → 对话标签 → 底部 `⚡ 后台任务`

系统自动显示当前进行中的异步任务（Celery 执行），包括：

- 任务名称和类型
- 进度百分比和进度条
- 当前状态（排队中 / 运行中 / 已完成 / 失败）

**技术原理**：
- 后端：Celery worker 通过 `app/services/task_bus.py` 将进度发布到 Redis pub/sub
- 前端：通过 SSE（Server-Sent Events）实时接收进度更新，回退到每 5 秒轮询
- 任何长任务（批量对比、技能审计、训练数据导出、LoRA 微调）都可以复用这套总线

**API 端点**：
- `GET /tasks` — 最近的异步任务列表
- `GET /tasks/<id>` — 任务状态查询（轮询）
- `GET /tasks/<id>/stream` — SSE 实时进度流

### 4.14 训练数据管理

**怎么打开**: 总览面板 → 展开 `📊 Training Data`

详见 [§5.5 训练数据管理与 LoRA 微调](#55-训练数据管理与-lora-微调)。

---

## 5. 审核员操作手册

审核员可以访问"审核面板"和"技能审计面板"，负责审核系统自动提取的内容。审核员不能修改系统配置，也不能管理其他用户的数据。

### 5.1 批量文档摄入

**怎么打开**: 审核面板 → 展开 `📥 Batch Document Ingestion`

这个功能可以让你把一大摞扫描的教科书或者文件一次性喂给系统，系统会自动提取有用的信息。

**操作步骤**：

1. **准备文件**：把要处理的文件打成一个 ZIP 压缩包（支持 PNG、JPG、BMP、TIFF、WebP、PDF 格式的扫描件）
2. **勾选你要的处理方式**：
   - ☑ 领域词库 → 系统自动提取专业词汇（比如"招标代理""评标委员会"这种），提取出来后你审核通过就会加入专业词库
   - ☑ 知识库 → 提取文字内容，你审核通过后存入 ChromaDB RAG 索引，全员都可以在聊天中搜到
   - ☑ 技能提取 → 自动从文字中提取框架、原则、技术方法、常见陷阱等
   - ☐ 结构化提取 → AI 自动识别文档中的项目名、招标编号、金额等关键信息，存成结构化数据
3. 点 `🚀 Start Ingestion` 开始处理（启动 Celery 任务或直接执行）
4. 看着进度条走完
5. 完成后按提示进行审核

**三路管道技术细节**：
- A 路：领域词典候选词 → `data/domain_words_review.json`（审核通过 → 追加到 `data/domain_words.txt`）
- B 路：公司知识库 → OCR → 文本合并去重 → ChromaDB 向量化
- C 路：公司技能 → 文本提取 → kb_skill_engine → 技能数据库

**实现文件**：`app/services/ingest_pipeline.py`（含 EasyOCR 中文+英文引擎，支持 png/jpg/bmp/tiff/webp/pdf）。

### 5.2 领域词库审核

**什么时候做**: 批量摄入完成后系统会提示，或者审核面板有黄色过期提醒时

**审核界面是这样的**：
```
☑ 招标代理  ✅  ×156次     ← 这个词已经在词库里了，自动勾上
☐ 评标办法       ×42次     ← 出现了42次，还没加进词库
☐ 工程量计算     ×38次
[✅ 批准选中的词]  [🗑 拒绝选中的词]  [☑ 全选]  [☐ 取消全选]
```

- 勾上要加入词库的词，点 `✅ 批准选中的词`
- 不想加的，勾上后点 `🗑 拒绝选中的词` 删掉
- 已经在 `data/domain_words.txt` 文件里的词会自己勾上，并且显示 ✅

### 5.3 知识库内容审核

**什么时候做**: 处理完知识库管道之后

**审核界面是这样的**：
```
📄 "批量摄入 abc123" · 一共156个文本块 · 展示10个样本

┌──────────────────────────────────┐
│ 第12块                           │
│ 招标文件应包括投标人须知、评标办法  │
│ [✏️ 编辑修正] [🗑 拒绝不要]        │
├──────────────────────────────────┤
│ 第28块  ✏️ 已修改                 │
│ （OCR识别错的地方已经手工改过了）   │
│ [✏️ 编辑修正] [🗑 拒绝不要]        │
└──────────────────────────────────┘

[✅ 批准全部并导入知识库]  [🔄 换一批样本看看]
```

- **✏️ 编辑修正**：点开可以看到这一块文字前后是什么，方便对照修正扫描识别（OCR）产生的错误
- **🗑 拒绝不要**：这一块文字是噪声或无关内容，不导入知识库
- **✅ 批准全部并导入知识库**：所有没被拒绝的块一起写入知识库
- **🔄 换一批样本看看**：重新随机抽10个块来检查

### 5.4 结构化文档查看

**怎么打开**: 审核面板 → 展开 `📑 Structured Documents`

AI 自动从扫描文件中提取的结构化数据，以表格展示：

| 项目名称 | 类型 | 招标编号 | 金额 |
|---|---|---|---|
| XX大桥施工监理 | 招标公告 | ZLTB-001 | ¥5,000,000 |

包含的字段有：项目名、招标编号、代理机构、开标日期、评标方法、预算金额、中标人、中标金额、合同方等。

### 5.5 训练数据管理与 LoRA 微调

**怎么打开**: 总览面板 → 展开 `📊 Training Data`

系统会自动记录每次 AI 对话（包括用户消息、助手回答、思考链、评分、知识库上下文、文件信息、模型名称、token 数、延迟等），数据存储在 `data/training/raw/{thread_id}/` 目录下，用于以后训练改进。

#### 数据导出

- **📥 增量导出**：只导出上次导出之后新增的对话数据（不会重复导出旧的）
- **📦 全量导出（高质量）**：导出所有数据，但只保留评分 3 星及以上的
- **📦 全量导出（全部）**：不管质量高低，全部导出
- **↺ 重置导出标记**：清掉"上次导出到哪了"的记录，下次导出就会变成全量
- 导出格式：JSONL（每行一个 JSON 对象），兼容 LoRA 微调格式

#### 健康检查

- `🔍 扫描检查`：检查所有数据的完整性（JSON 有效性、消息配对、时间戳、反馈索引越界等）
- `🔧 扫描并自动修复`：发现问题后自动修复孤儿反馈/上下文索引
- 健康检查历史记录：`data/training/health_log.json`（最近 20 条）

#### 数据清理

- `🔍 预演一下`：看看有多少旧数据会被删掉，不真的删
- `🗑️ 立即清理`：真的删掉超过保留天数（默认 90 天）的旧数据

#### 文件管理

展开 `📄 硬盘上的导出文件` → 可以下载或删除单个文件，也可以点 `🧹 清理旧文件` 自动删掉最早的（保留最近 20 个）。

#### LoRA 微调（🆕）

**这是训练流水线的最后一块拼图**。系统已经在后台自动收集和导出训练数据，现在可以直接在管理面板中启动 LoRA 模型微调。

**完整流水线**：

```
训练数据自动采集 (training_logger.py)
    ↓
质量筛选导出 JSONL (export_training_jsonl)
    ↓
Unsloth LoRA 微调 (scripts/run_lora_training.py)  ← 新功能
    ↓
适配器权重保存 (data/training/adapters/)
    ↓
自动注册到行业模型映射 (adapter_registry.json)
    ↓
llm_provider.py 自动发现 → 行业路由生效
    ↓
Ollama / vLLM 部署适配器
```

**启动微调**：
- `POST /admin/training/run_lora` — 提交训练配置（基础模型、数据集、行业、LoRA rank、epochs）
- 系统返回 `task_id`，可通过后台任务面板（§4.13）实时追踪训练进度
- 训练参数默认值：`Qwen/Qwen2.5-7B-Instruct`、`rank=16`、`epochs=3`、`lr=2e-4`

**管理适配器**：
- `GET /admin/training/lora/adapters` — 查看已训练的适配器列表
- `GET /admin/training/lora/datasets` — 查看可用数据集
- `POST /admin/training/lora/<industry>/activate` — 激活适配器（行业路由生效）
- `POST /admin/training/lora/<industry>/deactivate` — 停用适配器

**硬件要求**：Linux/WSL2 + NVIDIA GPU（≥16GB VRAM for 7B），Flask 应用本身不需要 Unsloth，训练在独立子进程中运行。

### 5.6 审核员工作量统计

**怎么打开**: 审核面板 → 展开 `📈 Reviewer Workload`

| 审核员 | 角色 | 操作次数 | 处理条目数 | 分类明细 | 最近操作 |
|---|---|---|---|---|---|
| 管理员 | admin | 15 | 256 | 批准领域词:8, 批准知识库:3 | 2026-06-27 |
| 审核员小李 | auditor | 5 | 42 | 拒绝领域词:2, 修正知识库:3 | 2026-06-25 |

点开下面的"最近操作记录"可以看到每条操作的详细信息。

### 5.7 技能审核

**怎么打开**: 点击页面顶部的 `🧠 技能` 标签

- 查看系统从知识库中自动提取的所有技能
- 标记重复的技能
- 标记长期未使用的技能
- 推荐适合推广给全员使用的技能
- 同名技能会自动追加 `#短hash` 后缀区分，避免混淆（如"投标函模板 #a3f2"）
- 支持快速合并（批量去重）和批量清理

**技能提取**：由 `app/services/skill_auditor.py` 自动从知识库和项目文件中提取框架、原则、技术方法和反模式。

系统会在每周日自动运行一次技能审计。

### 5.8 过期审核提醒与自动清理

**系统会自动做这些事**：
- 如果上传了文件、提取了词汇，但超过 3 天没人审核 → 审核面板顶部会弹出 **黄色警告条**提醒你
- 如果超过 6 天还没人审核 → 系统自动清理这些待审核数据，并记录日志
- 所有审核操作都会写入工作日志，方便统计

在运行时配置中可以调整"几天算过期"（参数名 `ingest_review_warn_days`，默认 3 天）。

### 5.9 技能文件规范验证

系统内置了 **21 个 AI 技能模块**（代码审查、安全检查、界面设计等），每个技能都有一套标准的说明文件格式。

**验证工具的使用**：在命令行中运行以下命令可以检查所有技能文件是否规范：
```
python -c "from app.services.skill_validator import validate_all; print(validate_all())"
```
这个命令会扫描所有技能文件，检查格式是否完整、内容是否够详细，然后输出一个报告。报告里会告诉你：一共多少个技能、几个合格、几个有警告、几个有错误。

如果想新建一个技能，可以参考模板文件：`.codebuddy/skills/SKILL_TEMPLATE.md`

---

## 6. 普通用户操作手册

### 6.1 AI 智能对话

**怎么打开**: 打开系统后默认就是这个页面

**基础用法**：
- 在底部输入框打字 → 点发送（或按回车），AI 就会回答你
- AI 会自动判断需不需要上网搜索（Bocha API）、需不需要查询当前日期
- 需要上传文件的话，点输入框旁边的 📎 按钮，支持 PDF、Word、Excel、图片等格式
- 想找以前的聊天记录，左边侧边栏 `🔍 搜索聊天记录` 可以关键词搜索
- 想开一个新话题，左边侧边栏点 `💬 新聊天`

**高级工具**（聊天界面中折叠的 "🛠️ 高级工具" 区域）：
- **Token 输出控制**：拖动滑条设定 AI 每次回答的最大字数（100-4800）
- **VL 图片分析开关**：开启后 AI 会用视觉模型分析上传的图片内容
- **批量文件对比**：同时选多个文件，勾选对比因子（文本/关键信息/文件属性/图片/语义），点 `📊 批量对比` 即可
- **模板文件**：选择一个模板文件作为对比基准
- **文件站**：查看当前上传的所有文件列表

**左边侧边栏**：
- 聊天历史列表（普通对话 + 项目对话分开显示）
- 待办事项（最多 5 条，项目内右键消息添加）
- 后台任务面板（显示正在运行的异步任务进度）
- 批量文件对比功能
- 企业信用查询功能

### 6.2 账户设置

**怎么打开**: 左边侧边栏 → `⚙️ 账户设置`

- **创建账户**：填一个用户名（5-18个字符）+ 设置一个 PIN 码（4位或6位纯数字）
- **登录**：用户名 + PIN 码
- **修改信息**：可以改用户名、PIN 码、绑定的邮箱
- 邮箱的作用是：改 PIN 码时系统会发验证码到邮箱，防止别人乱改
- **删除账户**：可以选择保留哪些数据（项目文件会永久保留）
- 退出登录

**后端隐藏端点**（无前端入口）：
| 路由 | 功能 |
|---|---|
| POST /create_account | 注册新用户 |
| POST /login | 登录 |
| POST /update_account | 更新账号（用户名/邮箱/PIN） |
| POST /request_pin_change_code | 请求 PIN 修改验证码（发邮箱） |
| POST /set_email | 设置邮箱 |
| POST /request_delete_account | 申请删除账号（返回数据清单） |
| POST /confirm_delete_account | 确认删除（需验证码+PIN） |
| POST /submit_delete_choices | 提交保留/删除数据选择 |
| POST /delete_account | 直接删除（admin 用） |
| POST /logout | 登出 |
| GET /check_auth | 检查登录状态 |

### 6.3 写作风格分析

**怎么打开**: 账户设置 → `✍️ My Writing Style`

- 第一次用，点 `✍️ 分析我的写作风格`，系统会根据你的聊天记录分析你的说话习惯
- 分析完会显示一个风格标签（比如"正式严谨 · 详细缜密"）和一段描述
- 点"刷新"可以重新分析（新的分析结果会和旧的结果融合，70% 新 + 30% 旧，越分析越准）
- 这个风格画像会影响 AI 帮你生成的报告的文风

### 6.4 项目管理

**怎么打开**: 点击页面顶部的 `🛠️ 项目` 标签

- 新建项目（可选行业：招标代理 / 工程造价 / 工程审计）
- 给项目上传文件，管理文件夹
- 管理项目成员（管理员可以添加负责人和普通成员）
- 企业信用查询：在项目文件中选择企业名称 → 自动查询信用信息
- 批量文件对比：同时选多个文件，系统会对比它们的相似度（TF-IDF + 语义 + 图片 + 属性）
- 项目聊天：每个项目自动生成独立的共享聊天会话（详见 6.5 节）

### 6.5 项目 AI 协作助手

**怎么打开**: 在项目内 → 点击 `🤖 AI 协作助手`

这是本系统最强大的功能之一。不同于普通的"一对一" AI 对话，项目 AI 助手实现了**团队级共享 AI 上下文**：

**核心特性**：
- **多人共享记忆**：同一个项目的所有成员共享 AI 上下文。@张三 问过的内容、@李四 做过的分析，AI 都记得。
- **身份标签**：每个成员在 AI 视角中有独立的 `@用户名` 身份标签。AI 可以明确说"@张三 之前分析过这个文件中的风险点"——而且系统会验证这个引用是否真实存在（不会编造）。
- **自动合并**：新成员加入项目时，AI 自动将历史协作记忆同步给新成员，无需重复沟通。
- **实时协作感知**：如果近 5 分钟内有其他成员也在使用 AI 助手，系统会提示你，避免重复工作。

**可用功能**：
| 功能 | 说明 |
|---|---|
| 💬 自由问答 | 像普通聊天一样问 AI，但 AI 会结合项目文件、知识库、成员历史来回答 |
| 📄 生成文档 | 选择输出格式（Word / Excel / PPT），AI 生成后直接下载，无需手工排版 |
| 🔄 工作流执行 | 按行业标准工作流逐步生成专业内容（详见 6.6 节），支持每步修改 |
| 💾 上下文持久化 | 所有对话自动保存到 `chat_messages` 表，下次打开项目时无缝继续 |

**协作工具**：
- ✅ **待办事项**：在项目聊天中右键任意消息 → 添加待办（最多 5 条），完成/删除操作，完成记录自动写入 AI 记忆
- 💬 **引用对话**：右键消息 → 引用，构建对话引用树（支持三级引用链），AI 自动感知引用上下文
- 🔄 **语义差异投票**：当 AI 对同一消息重新生成后，如果语义差异超过阈值（自动检测数值/否定词变化），自动触发 24h 投票，成员投票决定保留原版还是新版
- 📋 **实时轮询**：项目聊天页面每 3 秒自动拉取新消息和投票更新
- 🔴 **未读计数**：显示自上次阅读后的新消息数，滚动到底部自动标记已读

**输出格式支持**：
- `.docx`（Word 文档）— 自动排版，适合投标函、技术方案等正式文档
- `.xlsx`（Excel 表格）— 自动表格化，适合报价对比、评分汇总
- `.pptx`（PPT 演示）— 自动生成幻灯片大纲

### 6.6 行业标准工作流

**怎么打开**: 在项目 AI 协作助手中 → 选择工作流模式

系统内置了三大行业的标准化工作流模板，AI 会严格按照行业规范指导内容生成：

| 工作流 | 适用场景 | 核心步骤 |
|---|---|---|
| 🏗️ **招标代理** | 投标函起草、招标文件编制 | 资格审查 → 技术方案 → 商务报价 → 综合评审 |
| 📐 **工程造价** | 工程量清单、造价分析 | 工程量计算 → 单价分析 → 造价汇总 → 审核 |
| 🔍 **工程审计** | 审计报告、合规审查 | 资料收集 → 合规检查 → 问题梳理 → 报告撰写 |

**成员工作流与 KPI 追踪**：
- 每个项目成员可以自定义自己的执行步骤（存于 `member_workflows` 表）
- 系统自动追踪：本步骤生成了几次、修改了几轮、最终输出了多少字
- 当不同成员的输出出现大量重复（>60% 相似度）时，系统会发出重叠警告
- KPI 数据仅供管理员查看，帮助团队优化协作效率

### 6.7 知识库

**怎么打开**: 点击页面顶部的 `📚 知识库` 标签

- **我的知识库**：上传自己的文件，只有自己能看到（存于 `knowledge_lab_files` 表）
- **公司知识库**：管理员上传的公司文件，所有人都能看到。AI 的回答会优先参考公司知识库中的权威信息（存于 `company_knowledge_base` 表 + ChromaDB 向量索引）
- **技能总览**：查看从知识库中提取的各种框架、原则、技术方法等。同名技能会带有 `#短hash` 后缀确保不混淆
- **个人笔记本**：往下翻），自己的 Markdown 笔记，详见 6.10 节
- 知识库文件可以重命名，不影响 AI 检索

**检索机制**：ChromaDB 向量检索（`app/services/rag_engine.py`），默认返回 Top-8 最相关内容，最大上下文 8000 字。RAG 参数可通过运行时配置调整（§4.2）。

### 6.8 回收站

**怎么打开**: 点击页面顶部的 `🗑️ 回收站` 标签

- 查看被删除的内容，按来源分类折叠展示：
  - 💬 聊天文件
  - 📚 知识库 + 技能
  - 📁 项目文件
- 每个类别显示 `恢复 / 清空` 按钮
- 支持从回收站恢复文件到原位置（保留原始记录）
- 支持批量恢复筛选结果和清空筛选结果
- 删除技能时会保留技能摘要到回收站，方便日后查证
- 超过保留期的文件（默认 3 天）系统会自动清理，不需要手动管

**数据表**：`recycle_bin`（`data_snapshot` JSONB 列保存完整快照，用于恢复）。

### 6.9 生成日报

**怎么打开**: 聊天界面顶部标题栏 → `📊`

- 点一下，AI 会自动总结你今天的聊天内容，生成一份日报
- 内容包括：今天讨论了什么话题、有哪些工作产出
- 可以下载为 Word 文件（.docx）或 Markdown 文件（.md）
- 生成的报告会使用你的写作风格画像，读起来像你自己写的

### 6.10 个人笔记本

**怎么打开**: 知识库页面 → 往下翻到 `📓 My Notebook`

这是一个属于你自己的私人笔记本，所有内容只有你自己能看到。系统会自动把你的笔记接入知识库，以后在聊天中也能搜到你记过的东西。

**怎么用**：
- **➕ 新建笔记**：创建一个新笔记，支持 Markdown 格式（一种很简单的排版方式，用 # 表示标题、用 - 表示列表）
- **✏️ 编辑笔记**：点一下已有的笔记就可以打开修改
- **🤖 AI 摘要**：点一下让 AI 自动帮你把笔记总结成 2-3 句话
- **🔍 搜索笔记**：输入关键词按回车，系统会根据意思搜索你的笔记（不是简单匹配文字，而是真的理解你在找什么）
- **💾 保存**：写完点保存，系统会自动把你的笔记加入知识库
- **🗑 删除**：不需要的笔记可以删掉

数据保存在服务器上的 `data/notebooks/你的用户ID/` 目录里。

### 6.11 任务规划命令

**怎么用**: 在聊天输入框里直接输入

输入格式：`/plan 你想规划的事情`

例如：
```
/plan 2026年度招标代理工作计划
```

输完之后 AI 会自动生成一份结构化的项目计划，包含：目标、关键里程碑、可能的风险、时间安排、成功标准。生成完的计划会自动保存到你的笔记本里，随时可以查看和修改。

### 6.12 AI 自我审查命令

**怎么用**: 在聊天输入框里直接输入 `/review`

当你觉得 AI 刚才的回答可能不太对，或者想看看有没有更好的回答，输入这个命令。系统会用一个不同的 AI 模型来审查刚才的回答是否靠谱。

审查结果会告诉你：
- 打分（1-10 分）
- 判定结果（通过 / 需要改进 / 不合格）
- 具体有哪些问题
- 如果"需要改进"，还会给一个修正后的版本

> 注意：这个功能需要管理员在"运行时配置"中把 `judge_review_enabled` 打开，而且系统至少配置了两个不同公司的 AI 服务才可以用。

---

## 7. 接口地址参考

以下列出系统所有的后台接口地址，供技术人员对接使用。

### 管理员接口

| 接口地址 | 请求方式 | 作用 |
|---|---|---|
| `/admin/analytics/usage` | GET | 获取系统统计数据 |
| `/admin/analytics/system` | GET | 获取系统资源统计 |
| `/admin/analytics/export` | GET | 导出统计报告 |
| `/admin/analytics/roles` | GET | 查看用户角色管理 |
| `/admin/audit_log` | GET | 查看审计日志 |
| `/admin/prompt` | GET/POST | 查看和修改系统提示词 |
| `/admin/report/generate` | POST | 生成工作报告 |
| `/admin/clear_cache` | POST | 清除缓存 |
| `/admin/cleanup` | POST | 手动触发数据清理 |
| `/admin/cleanup/all` | POST | 一键清理所有 |
| `/admin/search_cache` | GET/POST | 查看和修改搜索缓存 |
| `/admin/rag/stats` | GET | RAG 索引统计 |
| `/admin/rag/rebuild` | POST | 重建 RAG 索引 |
| `/admin/runtime_config` | GET/POST | 查看和修改运行时配置 |
| `/admin/runtime_config_schema` | GET | 获取配置项的定义信息 |
| `/admin/llm_providers` | GET | 获取当前可用的 AI 模型列表 |
| `/admin/user_assets` | GET | 查看所有用户的数字资产 |
| `/admin/transfer_assets` | POST | 批量转移用户资产 |
| `/admin/db_tables` | GET | 查看数据库有哪些表 |
| `/admin/db_data` | GET | 分页查询数据 |
| `/admin/db_schema/<table>` | GET | 查看表结构 |
| `/admin/db_export_csv/<table>` | GET | 导出 CSV |
| `/admin/db_export_json/<table>` | GET | 导出 JSON |
| `/admin/db_overview` | GET | 表概览 |
| `/admin/search_cache_config` | GET/POST | 查看和修改搜索缓存设置 |
| `/admin/system_prompt` | GET/POST | 查看和修改系统提示词 |
| `/admin/training_stats` | GET | 获取训练数据统计 |
| `/admin/training_export` | POST | 导出训练数据（mode: incremental/full/quality/all/reset_watermark） |
| `/admin/training_export_history` | GET | 查看导出历史记录 + 水印 + 待导出数量 |
| `/admin/training_exports_list` | GET | 列出所有导出文件 |
| `/admin/training_exports_cleanup` | POST | 清理旧的导出文件 |
| `/admin/training_exports_delete/<文件名>` | POST | 删除指定导出文件 |
| `/admin/training_exports_download/<文件名>` | GET | 下载指定导出文件 |
| `/admin/training_cleanup_stats` | GET | 查看数据清理预览 |
| `/admin/training_cleanup` | POST | 执行数据清理 |
| `/admin/training_health` | GET/POST | 健康检查 / 自动修复 |
| `/admin/training_health_history` | GET | 查看健康检查历史 |
| `/admin/training/lora/datasets` | GET | 查看可用训练数据集列表 |
| `/admin/training/lora/adapters` | GET | 查看已训练的 LoRA 适配器 |
| `/admin/training/run_lora` | POST | 启动 LoRA 微调任务 |
| `/admin/training/lora/<industry>/activate` | POST | 激活行业适配器 |
| `/admin/training/lora/<industry>/deactivate` | POST | 停用行业适配器 |
| `/admin/user_styles` | GET | 查看所有用户的风格画像 |
| `/admin/user_styles/<用户ID>` | GET/POST | 查看/编辑某个用户的风格 |
| `/admin/user_styles/<用户ID>/analyze` | POST | 分析某个用户的写作风格 |
| `/admin/user_styles/analyze_all` | POST | 批量分析所有用户 |
| `/admin/ingest/upload` | POST | 上传文档压缩包开始摄入 |
| `/admin/ingest/status/<任务ID>` | GET | 查看摄入任务进度 |
| `/admin/ingest/domain_review` | GET | 查看待审核的专业词汇 |
| `/admin/ingest/domain_approve` | POST | 批准选中的专业词汇 |
| `/admin/ingest/domain_reject` | POST | 拒绝选中的专业词汇 |
| `/admin/ingest/kb_review/<任务ID>` | GET | 查看待审核的知识库内容 |
| `/admin/ingest/kb_chunk/<任务ID>/<序号>` | GET/POST | 查看/修正知识库中的某一段 |
| `/admin/ingest/kb_approve/<任务ID>` | POST | 批准知识库内容并导入 |
| `/admin/ingest/kb_reject/<任务ID>/<序号>` | POST | 拒绝知识库中的某一段 |
| `/admin/ingest/structured` | GET | 查看结构化提取的文档列表 |
| `/admin/ingest/stale_status` | GET | 查看过期审核状态 |
| `/admin/ingest/review_workload` | GET | 查看审核员工作量统计 |

### 普通用户接口

| 接口地址 | 请求方式 | 作用 |
|---|---|---|
| `/send_stream` | POST | 实时流式对话（SSE 打字机效果） |
| `/send` | POST | 标准对话 |
| `/new_chat` | POST | 创建新对话 |
| `/load_session` | GET | 加载历史会话 |
| `/search_chat` | GET | 搜索聊天记录 |
| `/share_conversation` | POST | 分享对话 |
| `/upload` | POST | 上传文件（多文件） |
| `/check_auth` | GET | 查看当前登录状态 |
| `/create_account` | POST | 注册新账户 |
| `/login` | POST | 登录 |
| `/logout` | POST | 退出登录 |
| `/update_account` | POST | 更新账号信息 |
| `/my_writing_style` | GET/POST | 查看/修改自己的写作风格 |
| `/my_writing_style/analyze` | POST | 分析自己的写作风格 |
| `/my_daily_report` | GET | 生成个人日报 |
| `/notebook` | GET | 获取自己的笔记列表 |
| `/notebook/<笔记名>` | GET/POST/DELETE | 查看/保存/删除笔记 |
| `/notebook/<笔记名>/summarize` | POST | AI 自动摘要 |
| `/notebook/search` | POST | 语义搜索笔记 |

### 任务进度接口

| 接口地址 | 请求方式 | 作用 |
|---|---|---|
| `/tasks` | GET | 列出最近的异步任务 |
| `/tasks/<task_id>` | GET | 查询单个任务状态 |
| `/tasks/<task_id>/stream` | GET | SSE 实时进度流 |

### 管理员与审核员共享接口

| 接口地址 | 请求方式 | 作用 |
|---|---|---|
| `/admin/ingest/*` | GET/POST | 文档摄入和审核相关（全部） |
| `/admin/training_health*` | GET/POST | 训练数据健康检查 |
| `/admin/training_exports_list` | GET | 列出导出文件 |
| `/admin/ingest/stale_status` | GET | 查看过期审核状态 |
| `/admin/ingest/structured` | GET | 查看结构化文档 |

### 项目接口

| 接口地址 | 请求方式 | 作用 |
|---|---|---|
| `/admin/projects` | GET/POST | 项目列表 / 新建 |
| `/admin/projects/<id>/files` | GET | 项目文件列表 |
| `/admin/projects/<id>/upload` | POST | 上传文件到项目 |
| `/admin/projects/<id>/chat` | POST | 项目聊天（SSE 流式） |
| `/admin/projects/<id>/ai_assist` | POST | AI 辅助回复 |
| `/admin/projects/<id>/ai_batch_analysis` | POST | AI 批量分析 |
| `/admin/projects/<id>/ai/download/<memory_id>` | GET | 下载 AI 生成文档 |
| `/admin/projects/<id>/members` | GET/POST | 成员管理 |
| `/admin/projects/<id>/my_workflow` | GET/POST | 工作流定制 |
| `/admin/projects/<id>/todos` | GET/POST | 待办列表 / 添加 |
| `/admin/projects/<id>/todos/<id>/done` | POST | 标记已完成 |
| `/admin/projects/<id>/todos/<id>/remove` | POST | 删除待办 |
| `/admin/projects/<id>/todos/done_log` | GET | 已完成记录（仅管理员） |
| `/admin/projects/<id>/quote` | POST | 添加引用 |
| `/admin/projects/<id>/quote_tree/<msg_id>` | GET | 查看引用树 |
| `/admin/projects/<id>/regen_votes` | GET | 查看活跃投票 |
| `/admin/projects/<id>/regen_votes/<id>/cast` | POST | 投票 |
| `/admin/projects/<id>/regen_votes/<id>/resolve` | POST | 裁决（项目经理） |
| `/admin/projects/<id>/unread_count` | GET | 未读消息数 |
| `/admin/projects/<id>/mark_read` | POST | 标记已读 |

### 知识库接口

| 接口地址 | 请求方式 | 作用 |
|---|---|---|
| `/knowledge/upload` | POST | 上传到个人知识库 |
| `/knowledge/upload/company` | POST | 上传到公司知识库 |
| `/knowledge/files` | GET | 个人知识库文件列表 |
| `/knowledge/company/files` | GET | 公司知识库文件列表 |
| `/knowledge/file/<id>` | DELETE | 删除知识库文件 |
| `/knowledge/search` | GET | 搜索知识库 |

### 回收站接口

| 接口地址 | 请求方式 | 作用 |
|---|---|---|
| `/recycle_bin` | GET | 回收站列表（支持 source 筛选） |
| `/recycle_bin/<type>/<id>/restore` | POST | 恢复单项 |
| `/recycle_bin/restore_all` | POST | 批量恢复 |
| `/recycle_bin/empty` | DELETE | 清空回收站 |
| `/recycle_bin/stats` | GET | 统计信息 |

---

## 8. 附录

### 8.1 数据库表一览

系统使用 PostgreSQL 16，通过 `psycopg2.pool.SimpleConnectionPool` 连接池（min=1, max=20）。共 **30+ 张表**。

#### 用户与认证
| 表名 | 关键列 | 用途 |
|------|--------|------|
| `users` | user_id, username, pin_hash, email, role, is_auditor, is_active, deletion_requested | 用户账号 |
| `user_consents` | user_id, consent_value, consent_date | 隐私同意记录 |

#### 聊天
| 表名 | 关键列 | 用途 |
|------|--------|------|
| `chat_sessions` | id, user_id, thread_id, title, project_id, created_at, updated_at | 聊天会话 |
| `chat_messages` | id, thread_id, role, content, thinking, timestamp | 聊天消息 |
| `archived_sessions` | thread_id, user_id, archive_path, archived_at | 归档会话 |
| `anonymous_sessions` | anon_id, thread_id, created_at | 匿名会话 |

#### 文件
| 表名 | 关键列 | 用途 |
|------|--------|------|
| `user_files` | id, user_id, thread_id, filename, content, file_hash, expires_at, meta_data | 用户聊天文件 |
| `project_files` | id, project_id, original_name, content, skill_summary, file_hash | 项目文件 |
| `knowledge_lab_files` | id, user_id, filename, content, skill_summary, is_company | 知识库文件 |
| `company_knowledge_base` | id, filename, content | 公司知识库 |
| `image_description_cache` | file_hash, description, created_at | VL 图片描述缓存 |
| `file_usage` | id, user_id, thread_id, filename, usage_type | 文件使用记录 |

#### 项目
| 表名 | 关键列 | 用途 |
|------|--------|------|
| `projects` | id, name, created_by, industry, created_at | 项目 |
| `project_members` | project_id, user_id, role, last_read_at, permissions | 项目成员 |
| `project_ai_memory` | id, project_id, user_id, role, content, content_md, created_at | 项目 AI 记忆 |
| `project_todos` | id, project_id, user_id, message_id, content_copy, status, done_at | 待办事项 |
| `message_quotes` | id, project_id, quoted_message_id, quoting_message_id, parent_quote_id, thread_id | 引用树 |
| `regen_votes` | id, project_id, message_id, original_content, new_content, status, round, expires_at | 重新生成投票 |
| `regen_vote_ballots` | id, vote_id, voter_id, vote, cast_at | 投票记录 |
| `member_workflows` | project_id, user_id, workflow_data | 成员工作流定制 |

#### 技能
| 表名 | 关键列 | 用途 |
|------|--------|------|
| `skills` | id, skill_name, skill_content, source_file_id, source_type, version | 技能库 |
| `skill_audit_results` | id, audit_batch_id, skill_id, score, issue_type, detail | 审计结果 |
| `kb_skills` | id, file_id, skill_content, extracted_at | 知识库提取技能 |

#### 审核与摄入
| 表名 | 关键列 | 用途 |
|------|--------|------|
| `ingest_tasks` | id, task_id, status, uploaded_by, file_count, progress | 摄入任务 |
| `domain_words` | word, source_file, approved, approved_by | 领域词典 |
| `domain_words_review` | word, source_file, status | 领域词审核队列 |

#### 回收与审计
| 表名 | 关键列 | 用途 |
|------|--------|------|
| `recycle_bin` | id, original_table, original_id, data_snapshot, deleted_by, deleted_at | 回收站 |
| `admin_audit_log` | id, admin_user_id, action, table_name, row_id, old_values, new_values | 审计日志 |

#### 其他
| 表名 | 关键列 | 用途 |
|------|--------|------|
| `credit_reports` | id, user_id, company_name, report_path, created_at | 企业信用报告 |
| `download_tokens` | token, file_path, created_at, expires_at | 文件下载 token |
| `share_files` | id, file_id, share_token, expires_at | 文件分享 |
| `batch_results` | id, thread_id, result_json, created_at | 批量对比结果 |
| `celery_taskmeta` | (Celery 自动) | Celery 任务元数据 |
| `celery_tasksetmeta` | (Celery 自动) | Celery 任务集元数据 |

### 8.2 自动调度任务一览

系统有两套调度器：

#### Celery Beat（Docker 模式）
| 任务 | 频率 | 文件 |
|------|------|------|
| cleanup_stale_sessions | 每小时 | `cleanup_tasks.py` |
| cleanup_temp_files | 每小时 | `cleanup_tasks.py` |
| run_skill_audit | 每周 | `app/services/skill_auditor.py` |
| generate_weekly_report | 每周 | `cleanup_tasks.py` |

#### APScheduler（Standalone 模式）
| 任务 | 频率 |
|------|------|
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
| cleanup_old_training_data | 每季度（Jan/Apr/Jul/Oct 1st 04:00） |
| cleanup_old_training_exports | 每季度（同日期 04:30） |
| auto_generate_weekly_report | 每周 |
| auto_generate_monthly_report | 每月 |
| auto_generate_annual_report | 每年 |
| auto_rag_health_check | 每日 |
| auto_training_health_check | 每周日 03:30 |
| auto_cleanup_stale_reviews | 每日 |
| run_skill_audit | 每周 |

### 8.3 数据文件目录

| 路径 | 内容 |
|------|------|
| `data/runtime_config.json` | 运行时配置覆盖 |
| `data/runtime_config_factory.json` | 工厂预设（只读，chmod 444） |
| `data/agent_prompt.json` | 自定义系统提示词 |
| `data/training/raw/{thread_id}/` | 训练原始数据（messages/feedback/context/metadata.json） |
| `data/training/exports/*.jsonl` | 训练数据导出（LoRA 格式） |
| `data/training/adapters/{industry}_{timestamp}/` | LoRA 适配器权重 |
| `data/training/adapter_registry.json` | 适配器注册表 |
| `data/training/export_watermark.json` | 导出游标 |
| `data/training/health_log.json` | 健康检查历史 |
| `data/domain_words.txt` | 领域词典 |
| `data/domain_words_review.json` | 领域词审核队列 |
| `data/workflows/{industry}.md` | 行业工作流模板 |
| `data/user_files/` | 用户上传文件 |
| `data/project_files/` | 项目文件 |
| `data/credit_reports/` | 信用报告 |
| `data/flask_session/` | Flask 文件系统会话 |
| `data/temp/` | 临时文件 |
| `data/notebooks/` | 个人笔记本 |
| `data/ingest/` | 摄入任务暂存 |

### 8.4 基础设施详情

| 组件 | 详情 |
|------|------|
| **Web 框架** | Flask 3.1.3 + Flask-Session + Flask-WTF |
| **数据库** | PostgreSQL 16（psycopg2-binary 连接池，min=1 max=20），Redis 7（Flask 会话 + Celery broker + pub/sub 进度总线） |
| **队列** | Celery 5.4.0（Docker 模式，soft_timeout=600s, hard_timeout=900s, result_expires=24h）/ APScheduler 3.11.2（standalone 回退） |
| **WSGI** | Gunicorn 23.0.0 + gevent（Docker），或 Flask dev server（standalone） |
| **代理** | Nginx（Docker）— HTTP→HTTPS 重定向，/api/ 代理，静态文件服务 |
| **AI 模型** | DeepSeek V4 Pro（主 agent），Zhipu GLM-4，Qwen Qwen3.7-Plus，SiliconFlow Qwen2.5-7B/72B（均通过 langchain 统一接口） |
| **Embedding** | bge-large-zh-v1.5（1024-dim 中文）/ paraphrase-multilingual-MiniLM-L12-v2（384-dim 多语言）/ distiluse-base-multilingual-cased（512-dim 旧版回退），语言自动检测切换 |
| **VL 模型** | qwen3-vl-plus-2025-12-19（DashScope API，图片分析/交叉验证） |
| **RAG** | ChromaDB + sentence-transformers（chunk_size=500, overlap=100, top_k=8, max_chars=8000） |
| **OCR** | EasyOCR（中文+英文，CPU 模式） |
| **NLP** | jieba 0.42.1（分词/TF-IDF/关键词），scikit-learn 1.9.0（cosine），numpy 2.5.0 |
| **搜索** | Bocha API（web search，72h 缓存） |
| **监控** | prometheus-flask-exporter（/metrics 端点） |
| **安全** | CSRF（flask-wtf），JWT（PyJWT 2.10.1），PIN 认证，admin 速率限制（5次/30分钟），prompt 注入检测 |
| **容器化** | Docker + docker-compose.yml（4 服务：flask + celery-worker + celery-beat + redis + postgres + nginx） |
| **SSL** | cert/ 目录下 cert.pem + key.pem（自签名开发证书） |
| **API 文档** | flasgger 0.9.7.1（Swagger UI at /apidocs） |
| **模型下载缓存** | ~/.cache/huggingface/hub/（Windows） |

### 8.5 技术架构一览

```
┌─────────────────────────────────────────────────────────┐
│                    前端 (HTML5 + JS)                      │
│    对话 · 项目 · 知识库 · 回收站 │ 数据 · 审核 · 技能 · 总览│
│    待办 · 引用树 · 语义投票 · 后台任务 │ 深色主题 · PWA   │
├─────────────────────────────────────────────────────────┤
│                  AI 安全层 (12 层防护)                     │
│  注入检测 → 内容隔离 → 安全指令 → 输出校验 → 交叉验证      │
│  实现: prompt_safety.py + judge_review.py + context_utils │
├─────────────────────────────────────────────────────────┤
│                    核心 AI 引擎                           │
│  主 Agent (LangGraph + DeepSeek V4 Pro)                  │
│  备选: 智谱 GLM-4 · 通义千问 · 硅基流动 (4 providers)    │
│  RAG 引擎 (ChromaDB) · VL 视觉模型 (qwen3-vl-plus)        │
│  Embedding: bge-zh / para-multilingual (自动语言切换)     │
│  语义差异: TF-IDF + 关键词 + 语义 (三路融合 + 数值/否定检测)│
├─────────────────────────────────────────────────────────┤
│                    业务服务层                             │
│  投标分析 (analysis_prompts) · 工作流引擎 (context_utils) │
│  文件生成 (docx/xlsx/pptx) · 技能审计 (skill_auditor)     │
│  征信查询 (credit_checker + Selenium) · 批量对比 (batch)  │
│  训练数据采集 (training_logger) · LoRA 微调 (Unsloth)      │
│  摄入管道 (ingest_pipeline + EasyOCR) · 裁判审查 (judge)  │
│  后台任务总线 (task_bus + Redis pub/sub + SSE)            │
├─────────────────────────────────────────────────────────┤
│                    中间件层                               │
│  Redis pub/sub 进度总线 · Celery 异步队列 · APScheduler   │
│  psycopg2 连接池 · JWT 认证 · CSRF 防护 · 速率限制        │
│  Nginx 反向代理 · Gunicorn + gevent · Docker 编排         │
├─────────────────────────────────────────────────────────┤
│                    数据层 (PostgreSQL 16)                 │
│  用户 · 项目 · 知识库 · AI 记忆 · 待办 · 引用 · 投票      │
│  训练数据 · 技能 · 审计 · 回收站 · 摄入 · 领域词典        │
│  Celery 状态 · 信用报告 · 分享 · 下载 · 文件使用记录       │
│  (30+ 张表, 全 JSONB 快照回收站, Multi-column 索引)       │
└─────────────────────────────────────────────────────────┘
```

---

> **文档版本**: 2026-07-01（合并自 USER_MANUAL.md + CODEBASE_AUDIT.md）  
> **系统位置**: `d:/PyCharm/Local_AI`（Windows 版）/ `d:/PyCharm/local_AI_Linux`（Linux 版）  
> **生成工具**: CodeBuddy AI  
> **变更历史**:  
> - v2026-06-29: 初版（USER_MANUAL.md）  
> - v2026-07-01: 合并审计报告 — 新增 §4.13 后台任务面板、§4.14 训练数据管理、§5.5 LoRA 微调、§6.5 协作工具详情、§7 完整 API 列表、§8 附录（数据库/调度/数据目录/基础设施/架构图）