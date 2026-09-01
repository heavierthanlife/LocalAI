# DECISIONS — 技术决策记录

项目的架构决策（ADR）与已评估未采纳的技术方案，用于选型追溯。

> **2026-09-01 更正**：此前文档声称采用 RapidOCR + MinerU，经代码核验实际管线为
> **EasyOCR + MarkItDown + LibreOffice + PyMuPDF**。本表相关行已按实际更正。

---

## 1. 当前决策记录

| 日期 | 决策 | 结论 | 理由 | 关联 |
|---|---|---|---|---|
| 2026-07-04 | OCR 引擎选型 | **保持 EasyOCR**（RapidOCR/PaddleOCR 均未采纳） | EasyOCR 1.7.2 满足中文印刷文档；`OCR_GPU=auto` 支持 GPU | `IMPROVEMENTS_SKIPPED` #1 |
| 2026-07-04 | 文档解析器 | **MarkItDown + LibreOffice + PyMuPDF**（弃 Docling/Marker/Unstructured） | 中文文档适用；其余英文优化/GPL 风险/6.6GB 过重 | `IMPROVEMENTS_SKIPPED` #2/#3/#4 |
| 2026-07-06 | 统一投标审计引擎 | 新增 `audit` blueprint + `audit_engine.py`，7 服务作库调用 | 一键全审 + 复合评分 + 持久历史 | `docs/superpowers/` |
| 2026-08-27 | 全量审计合并入清标 | **摘除 `audit_bp` 注册**，功能并入清标 5 维度 | 避免重复入口；表保留供 graph/cases 依赖 | `20260827log.md` §10 |
| 2026-08-28 | Prompt 语言 | JUDGE/STRUCTURED 中文化 | 领域术语一致性 | FIX-005 |
| 2026-08-31 | 无招标文件时 text_sim 处理 | **跳过**该指标 | 模板去除不可用，高余弦是模板重叠非围标 | FIX-012 |
| 2026-09-01 | 招标高频词并入停用集 | 自适应 k + **TF≥2 守卫** | 避免误杀 TF=1 独特技术参数；长招标覆盖不足 | FIX-014 (方案 Y) |
| 2026-09-01 | 异组件文本相似度处理 | **直接归零**（弃封顶 30） | 30×0.25=7.5 分仍污染 30/60 阈值底线 | FIX-014 (方案 Z-1) |
| 2026-09-01 | 段落相似度阈值 | **不调整**（维持 0.85） | 方案 X 被审计否决 | FIX-014 |
| 2026-07-06 | 编辑器选型（Wiki U9b） | **Tiptap/ProseMirror**（弃 Quill） | Mark 一等公民，内联合规标注最优 | `docs/editor_poc/report.md` |
| 2026-07-06 | Wiki U7 架构 | **Composition 模式** | 避免 God class，独立测试 | `docs/wiki_upgrade_plan.md` |

---

## 2. 已评估未采纳（Skipped）

> 完整 9 项见 `IMPROVEMENTS_SKIPPED.md`。以下是摘要与理由：

| # | 方案 | 结论 | 关键理由 |
|---|---|---|---|
| 1 | PaddleOCR | 弃 → 保持 EasyOCR | 300MB 安装难、Windows 常失败、慢 |
| 2 | Docling | 弃 → MarkItDown | 中文弱（英文训练）、API 未成熟 |
| 3 | Marker | 弃 → PyMuPDF | 中文弱、GPL 商用风险 |
| 4 | Unstructured.io | 弃 → 自研管线 | 6.6GB 过重、当前规模够用 |
| 5 | GPU / MinerU VLM | 延后 → **已有 2080S，建议启用 GPU** | 重装 CUDA torch 即可，见 `IMPROVEMENTS_SKIPPED.md` §RTX 2080 Super |
| 6 | structlog + Loki | 弃 → 纯文本日志 | 单用户单实例，300MB 开销不值 |
| 7 | 30+ Playwright E2E | 收缩 → 5 冒烟 | 前端选择器未稳定，维护成本高 |
| 8 | admin.py 全拆分 | 延后 | 纯重构无用户收益，风险高 |
| 9 | Theme Factory 主题 | 弃 | 主题美感不达专业招标系统标准 |

---

## 3. 清标评分体系决策链

清标评分经多轮审计迭代，决策如下（详细见 `AGENTS.md` §清标评分）：

1. **FIX-010**：裸加总 → 0-100 权重复合指数；RiskScorer 0.375/0.375/0.25；text_sim ≥80% 门槛；阈值 ≥60/≥30/<30
2. **FIX-011**：报价尾数检测（尾数相同≥80%）+ extract_prices 修复
3. **FIX-012**：无招标文件 text_sim 跳过；基线校准（仅工程类）
4. **FIX-013**：中文停用词表 + 统一三条向量化路径
5. **FIX-014**：自适应高频词（Y）+ 异组件归零（Z-1）；否决 Z-2 封顶、X 段落阈值

**数据门槛原则**（避免误报）：
- `text_sim`：无招标文件时跳过
- `quote`：无结构化开标报价时仅作参考
- `relationship`：正常投标天然共享项目名 → 权重已调低

---

## 4. 安全决策

| 日期 | 决策 | 说明 |
|---|---|---|
| 2026-08-28 | graph API 认证 | 5 端点 `@login_required` + 项目成员检查（C5） |
| 2026-08-28 | 管理员 PIN fail-closed | 生产无 PIN 即拒绝启动（C6） |
| 2026-08-28 | 匿名存储 PG JSONB | 弃 per-thread JSON 文件（M3） |
| 2026-08-28 | credit_tasks 移 Redis | 弃内存共享（M4） |
| 2026-07-08 | 审计基线 | 见 `SECURITY.md` |

---

*决策记录随功能迭代持续更新。*
