# 论文工具与源文件（Z公司工作流程数字化再造研究）

本目录是论文 v1.8 与答辩 PPT 的**可复现源文件与脚本**，用于在另一台电脑上继续修改与重生成。

## 1. 依赖

```bash
pip install python-docx matplotlib python-pptx pypdf pillow
```

- 中文字体：脚本默认使用 Windows 自带的 `SimHei`（`/mnt/c/Windows/Fonts/simhei.ttf`，WSL）或系统 `SimHei`；若在纯 Windows 下运行，把脚本里的字体路径改为 `C:/Windows/Fonts/simhei.ttf`。
- 论文版式依赖学校规范（`格式.doc`），正文宋体/Times New Roman 小四、1.25 倍行距、首行缩进 2 字符等。

## 2. 目录结构

```
scripts/    全部生成脚本（按用途分）
content/    content_v16.pkl —— v1.8 的结构化内容（段落/图/表/题注）
figures/    画风 A 的 16 张论文图（PNG，300dpi，15cm 宽）
reports/    体检报告、各版本修订说明、模板与答辩 PPT 说明
```

## 3. 复现路径

### 3.1 论文正文（v1.8）

```bash
# 1) 从 docx 提取结构化内容（已有 content_v16 时跳过）
python scripts/extract_content.py          # 输入 v12/v1.3_收尾修订版.docx

# 2) 应用 v1.5 内容修订（摘要/因果/口径/ESRI 等）
python scripts/patch_v15.py                # 生成 v12/content_v15.pkl

# 3) 重绘图片（画风 A）
python scripts/regen_style.py thesis_a 01 02 03 04 05 06 07 08 09 10 11 12 13 16 17 18
# 输出 figs_style_thesis_a/，再把图替换进 content_v16.pkl（见 make 脚本）

# 4) 生成 v1.6 图风版 docx
python scripts/build_school_v16.py         # 依赖 content_v16.pkl + figs_style_thesis_a/

# 5) 摘要单文件
python scripts/make_abstract700.py         # 中文 712 字摘要
python scripts/make_abstract_en.py         # 英文摘要
```

> 脚本中的相对路径（`v12/...`、`figs_style_thesis_a/...`）按原工作目录书写；在新机器上请把本目录的 `content/`、`figures/` 对应放好，或改脚本顶部常量。

### 3.2 答辩 PPT（32 页，4 号模板原样）

```bash
python scripts/make_ppt_official.py
```

- 依赖：`论文/PPT/4-红色中国石油大学（北京）模板.pptx`（脚本内 `TPL` 常量）
- 输出：32 页 PPT（模板设计不动，仅填内容 + 5 张论文图）

其他版本脚本：
- `make_defense_ppt_swiss.py`：学术极简版（30 页，Tufte/Swiss 风格）
- `make_defense_ppt_tpl4.py`：4 号模板重排版（30 页）
- `make_defense_ppt_tpl2.py`：2 号通用模板版（30 页）

### 3.3 格式修复（Word 手工改动后）

```bash
python scripts/fix_format_v17.py           # 按学校规范重排 styles/页眉页脚/三线表
python scripts/stage_v18.py && python scripts/fix_v18.py   # v1.8 因果校正 + 版式
```

## 4. 当前成果状态（2026-10-29）

| 内容 | 文件 |
|---|---|
| 论文终版 | `论文/11414_2024245834_YY_Z公司工作流程数字化再造研究.docx`（v1.8） |
| 答辩 PPT（推荐） | `论文/PPT/答辩PPT_4号模板_内容重排/…pptx`（32 页） |
| 答辩 PPT（备选） | 学术极简版 / 2 号模板版 / 4 号重排版 |
| 图表源 | 本目录 `figures/`（画风 A） |

## 5. 关键约定

- **不改数据、不编内容**：所有数值来自论文终版 PDF 与公司许可的聚合数据；测算值一律标注"测算值"。
- **问卷时点**：2024 年 10—11 月（平台 2025-02 上线前），因此第 3 章是"再造前诊断"。
- **结论口径**：17 项指标中 7 项有实测证据（3 项前后对比 + 4 项上线后观测），结论限定"流程周期与清标环节的同期改善"。
- **模板**：答辩 PPT 必须沿用学校下发的模板；只能选模板，不能改模板（封面/目录/分隔页/页眉页脚/配色/字号均不动）。
