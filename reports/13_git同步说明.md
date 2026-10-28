# Git 同步说明（跨电脑继续工作）

## 一、已推送到哪里

- 远端仓库：`github.com/heavierthanlife/LocalAI`
- 分支：**`mba-thesis`**（提交 `2666f35`）
- 原 `main` / `master` 分支**未改动**（本分支是独立历史，不影响原仓库代码）

## 二、另一台电脑怎么拉

在线（推荐）：
```bash
git clone --single-branch -b mba-thesis git@github.com:heavierthanlife/LocalAI.git mba-thesis
# 没有 SSH key 时用 HTTPS：
# git clone --single-branch -b mba-thesis https://github.com/heavierthanlife/LocalAI.git mba-thesis
```

离线（仓库根目录有 `thesis-mba.bundle`，23 MB）：
```bash
git clone -b mba-thesis thesis-mba.bundle mba-thesis
cd mba-thesis
git remote set-url origin git@github.com:heavierthanlife/LocalAI.git
```

## 三、仓库内容

| 目录 | 内容 |
|---|---|
| `build/` | 论文构建源（build_v11.py、text_part1–3、refs_v11.py、make_figures_v11–14.py）、16 张新图与旧图、v1.0/v1.1/v1.2/作者修订版/图件重制版 docx |
| `reports/` | 修订说明 09–12、数据与文献核验、源文件打包 |
| `verification/` `audit/` `system_probe/` `refs/` | 系统事实核验、文本审计、运行系统探查、文献候选项 |
| `docs/handoff/` | 原始交接说明 + 历次修订文档 |
| `tools/` | docx 提取、图件替换、编辑脚本 |
| `README.md` | 当前状态、环境、待办、学术诚信边界 |

## 四、后续怎么同步

```bash
git add -A
git commit -m "说明本次改动"
git push origin mba-thesis
```

## 五、注意

- `main`/`master` 保持原样；论文项目只走 `mba-thesis`。
- Local_AI 源码仓库里 160 个 modified 文件是 **CRLF/LF 换行噪音**（增删行数完全相同），未提交；如需提交请单独处理换行规则。
- 本仓库不含：`.venv/`、`pylibs/`、`fonts/`（大文件/字体授权）、`pkg/`（原始交付包解压产物，Desktop 有 zip）。
