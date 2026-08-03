# Wiki 系统升级方案（最终版）

15 项升级，4 个 Phase，86-103 天（并行优化）

---

## 技术决策（已确认）

| 决策 | 结论 | 理由 |
|------|------|------|
| 编辑器选型 | **Tiptap/ProseMirror**（Phase 1 第 1 周 PoC） | 原生注解支持，文档标注场景最优 |
| U7 架构 | **Composition 模式**：`TemplateDeviationChecker` 组合进 `ComplianceChecker` | 避免 God class，独立测试，可扩展 |
| U1+U3 | **合并启动**：DB schema 设计 → 直接导入 | 省掉 JSON→DB 二次迁移 |
| JS 拆分 | 按 Phase 拆分：`compliance.js`（P1 末）+ `templates.js`（P2 初） | 不影响 app.js 已有功能 |
| 迁移回滚 | 备份 + DB 故障回退 JSON + `--dry-run` | 三保险 |
| U14 监控 | **MVP = 手动提交** + AI 影响评估 | RSS/爬虫不稳定且运维成本高 |

---

## Phase 1: 法规库升级（17-22 天）

### U1+U3: 法规数据库 + 版本管理

**合并**：DB schema 设计直接覆盖版本管理需求，JSON 文件做只读备份。

- 24 部法规（20 全国 + 4 省）→ `law_masters` / `law_versions` / `law_articles` 三表
- `_select_relevant_laws()` 改为 DB 查询，保留 JSON 回退路径
- 版本管理 UI（法规树 + 时间线 + 条款 diff）
- API: 10 个端点（CRUD + 版本激活 + diff + 历史）
- 迁移脚本：`scripts/migrate_seed_laws.py`，支持 `--dry-run`

**工期**: 12-16 天（合并后）

### U2: 法规条款语义检索

- 复用现有 `paraphrase-multilingual-MiniLM-L12-v2` + `ChromaDB`
- 新增 `rag_laws` collection，条款分块 ~350 条
- `_select_relevant_laws()` 重构为语义+关键词双通道
- ChromaDB 索引重建触发器（法规增删时自动更新）
- 降级：ChromaDB 不可用时回退纯关键词匹配
- 核心嵌入逻辑 ~40 行，含集成 4-5 天

**工期**: 4-5 天（可并行 U1+U3）

### U4: 地区变体管理

- 三表：`law_regions` / `law_region_bindings`（多对多）
- 绑定类型：baseline / supplement / exclusive
- 层级继承：全国 → 省 → 市 → 县（自动包含上级法规）
- 合规检查集成：`region_code` 参数

**工期**: 5-7 天（依赖 U1+U3 的 DB schema）

---

## Phase 2: 模板库整合（30-32 天）

### U5: 统一模板管理 CRUD

- `bid_templates` 表（JSONB sections）
- AI 辅助导入：`.docx` → 解析章节结构 → 用户确认 → 入库
- 前端：模板列表/详情/创建/编辑/导入（~6 面板）
- **同步启动**：`templates.js` 拆分

**工期**: 10.5 天

### U6: 模板版本管理 + diff

- 不可变快照设计：`bid_template_versions` / `bid_template_diffs`
- 一键回滚
- Diff 视图：章节级 + 内容级 inline diff（绿/红/黄）

**工期**: 9.5 天（依赖 U5）

### U7: 模板→合规检查集成

- **Composition 模式**：`TemplateDeviationChecker` 组合进 `ComplianceChecker`
- 功能类型 `template_deviation`：
  - 结构检查（快速，无 AI）：比对模板章节 vs 投标章节
  - 内容检查（AI 驱动）：逐章对比内容偏差
- 偏差级别：MATCH / MINOR_DEVIATION / MAJOR_DEVIATION / MISSING
- `check()` 方法作为编排层，不承担具体检查逻辑

**工期**: 8.5 天（依赖 U5）

### U8: AI 自动推荐模板

- 三层匹配：元数据过滤（0.3） + 内容相似度 ChromaDB（0.5） + 热度提升（0.2）
- `template_usage_log` 追踪

**工期**: 9.5 天（依赖 U5）

---

## Phase 3: 实时合规 + 趋势（25-32 天）

### U13: 案例库独立管理

- 四表：`audit_cases` / `case_tags` / `case_law_links` / `case_template_links`
- 自动生成：VIOLATION/CRITICAL 级别 finding → 案例

**工期**: 6-7 天

### U10: 合规历史趋势分析

- 指标：合规分数趋势、违规类型分布、规则有效性
- 可视化：折线图 + 柱状图 + 热力图

**工期**: 5-6 天

### U11: 合规仪表盘

- 组件：总审核数/通过率/常见违规 TOP5
- 项目级 vs 组织级视图
- 关键违规实时告警

**工期**: 6-7 天（依赖 U10）

### U15: 多项目合规对比

- 物化视图 `mv_project_compliance_summary`
- 能力：项目×功能矩阵、热力图、模式检测、XLSX 导出

**工期**: 5-6 天（依赖 U10）

### U9: 实时合规提示（分两期）

- **U9a**（5 天）：后端增量检查 API + 缓存层
  - 仅对修改过的章节执行合规检查
  - 300ms 防抖 + 缓存
  - 复用现有 `compliance_checker.py` 逻辑
- **U9b**（10-15 天，取决于编辑器选型）：编辑器集成 + 内联标注
  - Tiptap/ProseMirror 集成
  - 内联标注 + 侧边栏建议 + 状态栏合规度
  - **工期变量**：编辑器 PoC 后确定具体天数

**工期**: 15-20 天（合计）

---

## Phase 4: 增强功能（14-17 天）

### U12: 知识图谱可视化

- 节点类型：法规、条款、模板、案例、技能、项目
- 边类型：引用、依据、冲突、包含、支持
- 前端：Cytoscape.js 力导向图
- **后置依赖**：U5（模板）、U13（案例）

**工期**: 7-9 天

### U14: 法规变更监控推送

- **MVP**：手动提交变更 + AI 影响评估
- 影响评估：自动识别受影响的模板和规则
- 推送：应用内通知 + 邮件
- RSS/爬虫作为可选项（标记为不稳定，+3-5 天）

**工期**: 7-8 天

---

## 依赖图

```
Phase 1 (Week 1-3)
├── 编辑器选型 PoC（U9b 前置，第 1 周）
├── U1+U3 (合并: DB schema + 24 法规 + 版本管理 + 迁移)
├── U4 (地区变体, 依赖 DB schema)
├── U2 (语义检索, 独立, 可并行)
│
Phase 2 (Week 4-7)
├── JS 拆分: templates.js 启动
├── U5 (模板 CRUD)
│   ├── U6 (版本 + diff)
│   ├── U7 (模板偏差, Composition 接入 ComplianceChecker)
│   └── U8 (AI 推荐)
│
Phase 3 (Week 8-12)
├── U13 (案例库)
├── U10 (趋势) → U11 (仪表盘) → U15 (多项目对比)
├── U9a (增量检查 API + 缓存)
│   └── U9b (编辑器集成 + 标注)
│
Phase 4 (Week 13-16)
├── U12 (图谱, 依赖 U13+U5)
└── U14 (监控 MVP)
```

---

## 工期汇总

| Phase | 内容 | 工期 |
|-------|------|------|
| Phase 1 | 法规库升级（U1+U3, U2, U4） + 编辑器 PoC | 17-22 天 |
| Phase 2 | 模板库整合（U5, U6, U7, U8） + JS 拆分 | 30-32 天 |
| Phase 3 | 实时合规 + 趋势（U9a, U9b, U10, U11, U13, U15） | 25-32 天 |
| Phase 4 | 增强功能（U12, U14） | 14-17 天 |
| **总计** | | **86-103 天** |

最大风险：**U9b（编辑器集成）**。建议 Phase 1 第 1 周完成 PoC，确认工期。

---

## 每项升级的回归测试要求

见 `AGENTS.md` 的 [Regression Testing](#regression-testing) 段：

1. 准备 3 个已知招标文件作为基线（工程/货物/服务）
2. 升级前后分别运行合规检查
3. 结果对比：不允许丢失已有 finding
4. 验证结果记录到 commit message：`regression: 3/3 baseline passed`

适用于合规检查器、规则提取、AI 审查等所有合规相关路径的改动。

---

## 未覆盖的风险

| 风险 | 措施 |
|------|------|
| U14 RSS/爬虫不稳定 | MVP 只做手动提交，爬虫为可选项 |
| 编辑器 PoC 失败 | 回退到 U9 拆分（先做增量 API，编辑器延后） |
| DB 迁移故障 | 备份 seed_laws.json，`--dry-run`，JSON 回退路径 |
| `_select_relevant_laws()` 重构引入回归 | U2 回归测试覆盖此函数 |
| app.js 膨胀 | Phase 2 拆 `templates.js`，Phase 1 末拆 `compliance.js` |
