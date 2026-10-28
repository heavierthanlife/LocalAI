# -*- coding: utf-8 -*-
"""v1.1 图件修复脚本（仅重绘有硬伤的图，不触碰论文正文与其它构建脚本）。

修复清单（对应审计报告 M-20 ~ M-24 与 M-23 的图1-1）：
  图6-3  按表6-2 四项指标重绘（AS-IS=100，基准情景 85/68/60/55），标注“测算值，非实测值”
  图6-2  子图(b) 综合指数改为等权合成结果 [100, 82.3, 67.0, 54.5]，标题改为“等权标准化合成”
  图6-1  按表6-1 的 17 项指标（5 准则）重绘，底注改为等权合成/情景测算口径
  图2-1  图内标题“资产评估…”→“招标代理业务…”，清除资产评估残留
  图5-2  字段名按数据库真实列名修正；删除表6-1 之外的“审计耗时/质量水平”
  图1-1  “鱼骨图·SWOT”→“鱼骨图·访谈归纳”；“配对检验·ITS”→“情景测算·敏感性分析”

输入：无（全部为脚本内硬编码数据，来源 build_v1.py 的 TABLES["t61"]/["t62"] 与数据库表结构）
输出：figures_v11/；原图备份到 figures_orig_backup/
运行：docker run ... -w /w/build --entrypoint python local-ai:latest make_figures_v11.py
"""
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib import font_manager
import numpy as np

# ---------------------------------------------------------------- 字体
HERE = os.path.dirname(os.path.abspath(__file__))
for _f in ("/w/fonts/simhei.ttf", os.path.join(HERE, "..", "fonts", "simhei.ttf")):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        _name = font_manager.FontProperties(fname=_f).get_name()
        plt.rcParams["font.sans-serif"] = [_name, "SimHei"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.12

OUT = "figures_v11"          # 输出目录
ORIG = "figures_orig_backup"  # 原图备份目录
os.makedirs(OUT, exist_ok=True)
os.makedirs(ORIG, exist_ok=True)

C = dict(primary="#1F3864", blue="#2E75B6", light="#DEEBF7", mid="#9DC3E6",
         orange="#C55A11", orange_l="#FBE5D6", green="#548235", green_l="#E2EFDA",
         gray="#595959", lgray="#EDEDED", red="#C00000", gold="#BF9000")

RESULTS = []  # (图号, 文件名, 字节, 像素, 非白像素比)


# ---------------------------------------------------------------- 通用绘图件
def canvas(w, h, xlim=(0, 100), ylim=(0, 100)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#1F3864", fs=9, tc="#1A1A1A",
        lw=1.2, bold=False, ls="-", zorder=3, pad=0.35, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad={pad},rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                linestyle=ls, zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            zorder=zorder + 1, linespacing=1.55)


def arrow(ax, p1, p2, color="#1F3864", lw=1.3, ms=11, ls="-", zorder=2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, linewidth=lw, linestyle=ls,
                                 zorder=zorder))


# ---------------------------------------------------------------- 自检
def selfcheck(tag, path):
    from PIL import Image
    size = os.path.getsize(path)
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    ink = float((arr < 245).any(axis=2).mean())
    RESULTS.append((tag, os.path.basename(path), size, im.size, ink))
    if size < 100 * 1024:
        print(f"  !! 文件小于 100KB: {path} ({size} B)")
    if ink < 0.01:
        print(f"  !! 疑似空白图: {path} (ink={ink:.4f})")
    print(f"  [ok] {os.path.basename(path)}  {size/1024:.0f} KB  "
          f"{im.size[0]}x{im.size[1]}  ink={ink*100:.1f}%")


def backup(orig_name):
    src = os.path.join("figures", orig_name)
    if os.path.exists(src):
        dst = os.path.join(ORIG, orig_name)
        if not os.path.exists(dst):  # 只留最原始的一份，重复运行不覆盖
            shutil.copy2(src, dst)
            print(f"  [backup] {orig_name} -> {ORIG}/")
        else:
            print(f"  [backup] {orig_name} 已存在，跳过")


# ================================================================ 图6-3
def fig63():
    """图6-3 再造前后四项关键指标的对比（AS-IS=100）。
    数据：表6-2 基准情景（流程周期 -15%、清标环节耗时 -32%、文件返工轮次 -40%、
    人工数据搬运次数 -45%），AS-IS 归一化为 100。
    """
    backup("图10_再造前后目标指标对比.png")
    cats = ["流程周期", "清标环节耗时", "文件返工轮次", "人工数据搬运次数"]
    base = [100, 100, 100, 100]
    target = [85, 68, 60, 55]
    drops = ["-15%", "-32%", "-40%", "-45%"]
    x = np.arange(len(cats))
    w = 0.34

    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    b1 = ax.bar(x - w / 2, base, w, color="#BFBFBF", edgecolor="white",
                label="AS-IS（现状 = 100）", zorder=2)
    b2 = ax.bar(x + w / 2, target, w, color=C["blue"], edgecolor="white",
                label="TO-BE（基准情景测算值）", zorder=2)
    for xx in x - w / 2:
        ax.text(xx, 102, "100", ha="center", va="bottom", fontsize=8.4, color=C["gray"])
    for xx, v, d in zip(x + w / 2, target, drops):
        ax.text(xx, v + 2.5, f"{v}\n{d}", ha="center", va="bottom",
                fontsize=8.4, color=C["primary"], fontweight="bold", linespacing=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9.2)
    ax.set_ylabel("指数（AS-IS = 100）", fontsize=9.2)
    ax.set_ylim(0, 124)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(fontsize=8.6, frameon=False, loc="upper right", ncol=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", color="#D9D9D9", zorder=0)
    ax.set_axisbelow(True)
    # 图内水印：测算值声明（替代原“待实测替换”字样）
    ax.text(0.5, 0.46, "测算值，非实测值", transform=ax.transAxes, fontsize=19,
            color="red", alpha=0.13, ha="center", va="center", rotation=12,
            fontweight="bold", zorder=1)
    fig.tight_layout()
    p = f"{OUT}/图6-3_再造前后指标对比.png"
    fig.savefig(p)
    plt.close(fig)
    selfcheck("图6-3", p)


# ================================================================ 图6-2
def fig62():
    """图6-2 三情景下的指标变化与流程综合指数。
    (a) 表6-2 三情景降幅（保守/基准/乐观），数值不变；
    (b) 等权标准化合成的综合指数：100 -> 保守 82.3 / 基准 67.0 / 乐观 54.5
        （对应等权平均降幅 17.75% / 33.0% / 45.5%）。
    """
    backup("图16_情景测算结果.png")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1),
                             gridspec_kw=dict(width_ratios=[1.18, 1]))

    # ---- (a) 三情景指标变化区间
    ax = axes[0]
    metrics = ["流程周期", "清标环节耗时", "文件返工轮次", "人工数据搬运次数"]
    cons = [-8, -18, -20, -25]
    bench = [-15, -32, -40, -45]
    opt = [-22, -45, -55, -60]
    x = np.arange(len(metrics))
    for i, (lo, mid, hi) in enumerate(zip(cons, bench, opt)):
        ax.plot([i, i], [lo, hi], color=C["blue"], lw=8, alpha=0.30,
                solid_capstyle="round", zorder=2)
        ax.plot([i], [mid], marker="o", ms=8.5, color=C["primary"], zorder=3)
        ax.text(i, hi - 1.6, f"{hi}%", ha="center", va="top", fontsize=7.6,
                color=C["primary"])
        ax.text(i, lo + 1.6, f"{lo}%", ha="center", va="bottom", fontsize=7.6,
                color=C["primary"])
        ax.text(i + 0.17, mid, f"{mid}%", ha="left", va="center", fontsize=8.2,
                color=C["primary"], fontweight="bold")
    ax.axhline(0, color="#595959", lw=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8.4)
    ax.set_ylabel("相对基线的变化率（%）", fontsize=8.8)
    ax.set_xlim(-0.55, 3.62)
    ax.set_ylim(-68, 6)
    ax.set_title("(a) 三情景下的指标变化（保守 / 基准 / 乐观）", fontsize=9.6,
                 color=C["primary"], fontweight="bold", pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", color="#D9D9D9")
    ax.set_axisbelow(True)
    handles = [Line2D([0], [0], color=C["blue"], lw=8, alpha=0.30,
                      solid_capstyle="round", label="保守—乐观区间"),
               Line2D([0], [0], marker="o", ls="", ms=8.5, color=C["primary"],
                      label="基准情景")]
    # 图例放在 0 基线以下、右侧空档，避免压住 0% 基线
    ax.legend(handles=handles, fontsize=7.6, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.90))

    # ---- (b) 等权合成综合指数
    ax = axes[1]
    labels = ["AS-IS", "保守", "基准", "乐观"]
    vals = [100.0, 82.3, 67.0, 54.5]
    txts = ["100", "82.3", "67.0", "54.5"]
    cols = ["#BFBFBF", "#9DC3E6", C["blue"], C["primary"]]
    bars = ax.bar(labels, vals, color=cols, width=0.56, edgecolor="white", zorder=2)
    for b, t in zip(bars, txts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.4, t,
                ha="center", va="bottom", fontsize=9.0,
                color=C["primary"], fontweight="bold")
    ax.set_ylim(44, 108)
    ax.set_ylabel("流程综合指数（AS-IS = 100）", fontsize=8.8)
    ax.set_title("(b) 综合指数（等权标准化合成）", fontsize=9.6,
                 color=C["primary"], fontweight="bold", pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", color="#D9D9D9")
    ax.set_axisbelow(True)
    ax.text(0.5, 0.06, "测算值，非实测值", transform=ax.transAxes, ha="center",
            va="center", fontsize=15, color="red", alpha=0.13, rotation=8,
            fontweight="bold", zorder=1)

    fig.tight_layout(w_pad=2.0)
    p = f"{OUT}/图6-2_三情景测算结果.png"
    fig.savefig(p)
    plt.close(fig)
    selfcheck("图6-2", p)


# ================================================================ 图6-1
def fig61():
    """图6-1 再造效果评价指标体系（5 准则 × 17 指标，名称与表6-1 完全一致）。"""
    backup("图11_再造效果评价指标体系.png")
    fig, ax = canvas(10.6, 5.9)

    box(ax, 28, 88.2, 44, 9.6, "工作流程数字化再造效果（目标层）", fc=C["primary"],
        ec=C["primary"], fs=10.6, tc="white", bold=True)
    # 体系规模写在目标层框内，避免与“目标层→准则层”的引导箭头交叉
    ax.text(50, 90.0, "准则层 5 个 ｜ 指标层 17 项", ha="center", va="center",
            fontsize=8.2, color="white", alpha=0.92)

    crit = [
        ("流程效率", ["流程周期", "计划偏差率", "里程碑延误", "返工强度", "任务响应"],
         C["light"], C["primary"]),
        ("集成水平", ["数据一次录入率", "自动流转占比", "接口成功率"], "#FFFFFF", C["primary"]),
        ("资源节约", ["单项目人工工时", "纸质文档占比", "人工搬运次数"], C["green_l"], C["green"]),
        ("业务价值", ["节点准时交付率", "客户满意度", "单项目毛利率"], C["green_l"], C["green"]),
        ("组织能力", ["培训覆盖率", "工具采纳率", "留痕完整率"], C["orange_l"], C["orange"]),
    ]
    w, gap, x = 18.0, 1.6, 2.0
    for name, inds, fc, ec in crit:
        box(ax, x, 71.5, w, 8.6, name, fc=fc, ec=ec, fs=9.8, bold=True)
        arrow(ax, (50, 87.8), (x + w / 2, 80.5), color="#A6A6A6", lw=1.0, ls="--", ms=9)
        body = "\n".join(f"· {i}" for i in inds)
        box(ax, x + 0.7, 39, w - 1.4, 28, body, fc="#FAFAFA", ec=ec, fs=7.7, lw=0.9, ls=":")
        arrow(ax, (x + w / 2, 71.1), (x + w / 2, 67.4), color=ec, lw=1.2, ms=9)
        x += w + gap

    ax.plot([3, 97], [35.6, 35.6], color="#BFBFBF", lw=0.9, ls=":")
    arrow(ax, (50, 35.2), (50, 27.6), color="#BFBFBF", lw=1.0, ms=9)
    box(ax, 3, 9.5, 94, 17.4,
        "赋权方法：本文情景测算采用等权合成（后续实测阶段拟以熵权法与AHP组合赋权）\n"
        "评价方法：情景测算、敏感性分析（后续实测阶段拟采用配对检验与中断时间序列）",
        fc="#F2F2F2", ec=C["gray"], fs=8.6, ls="--")
    p = f"{OUT}/图6-1_再造效果评价指标体系.png"
    fig.savefig(p)
    plt.close(fig)
    selfcheck("图6-1", p)


# ================================================================ 图2-1
def fig02():
    """图2-1 理论整合分析框架：图内标题改为招标代理口径。"""
    backup("图02_理论整合分析框架.png")
    fig, ax = canvas(8.8, 5.9)
    ax.text(50, 97.0, "理论基础", ha="center", fontsize=9.6, color=C["gray"],
            fontweight="bold")
    theories = [
        (3, "业务流程再造理论\nBPR ｜ ESIA 原则", C["light"], C["primary"]),
        (34.5, "TOE 框架\n技术 · 组织 · 环境", C["orange_l"], C["orange"]),
        (66, "动态能力理论\n感知 · 抓取 · 重构", C["green_l"], C["green"]),
    ]
    for x, t, fc, ec in theories:
        box(ax, x, 83, 28, 11, t, fc=fc, ec=ec, fs=9.6, bold=True)
        arrow(ax, (x + 14, 82.6), (x + 14, 76.9), color="#8497B0", lw=1.2)
    # M-20：原“资产评估工作流程数字化再造分析框架” -> 招标代理口径
    box(ax, 8, 66.5, 84, 10, "招标代理业务工作流程数字化再造分析框架",
        fc=C["primary"], ec=C["primary"], fs=12.5, tc="white", bold=True)
    layers = [
        (3.5, 25, "【诊断层】\n工具集成度\n流程冗余度\n员工数字能力", C["light"], C["primary"]),
        (37.5, 25, "【解释层】\n技术维度\n组织维度\n环境维度", C["orange_l"], C["orange"]),
        (71.5, 25, "【路径层】\nESIA 精简\n平台化集成\n政策对接·培训", C["green_l"], C["green"]),
    ]
    for x, w, body, fc, ec in layers:
        box(ax, x, 42, w, 16, body, fc=fc, ec=ec, fs=8.8)
        arrow(ax, (50, 66.2), (x + w / 2, 58.6), color="#A6A6A6", lw=1.1, ls="--", ms=10)
    box(ax, 10, 23, 80, 9, "再造效果：周期缩短 ｜ 资源节约 ｜ 成本下降 ｜ 动态能力提升",
        fc="#FFFFFF", ec=C["primary"], fs=10, bold=True)
    for x, w, *_ in layers:
        arrow(ax, (x + w / 2, 41.6), (x + w / 2, 32.6), color="#A6A6A6", lw=1.2)
    box(ax, 16, 7, 68, 7, "反馈迭代：绩效审计 → 流程再优化（持续改善闭环）",
        fc=C["lgray"], ec=C["gray"], fs=9, ls="--")
    arrow(ax, (50, 22.6), (50, 14.6), color=C["gray"], lw=1.2, ls="--")
    p = f"{OUT}/图2-1_理论整合分析框架.png"
    fig.savefig(p)
    plt.close(fig)
    selfcheck("图2-1", p)


# ================================================================ 图5-2
def fig52():
    """图5-2 流程数据模型与可计算的过程指标。
    字段名按数据库真实列名修正；指标仅保留表6-1 体系内的 6 项。
    """
    backup("图13_流程数据模型与可算指标.png")
    fig, ax = canvas(9.8, 6.4)
    # 主链条（列名已按 \d project_timelines 等真实表结构核对）
    box(ax, 2, 76, 26, 15, "projects\n项目主表\nid · name · status\ncreated_at / updated_at",
        fc=C["primary"], ec=C["primary"], fs=7.8, tc="white", bold=True)
    box(ax, 32, 76, 30, 15,
        "project_timelines\n流程主表\nplanned_start_date / planned_end_date\n"
        "actual_start_date / actual_end_date",
        fc=C["light"], ec=C["primary"], fs=7.2)
    box(ax, 66, 76, 32, 15,
        "project_timeline_milestones\n里程碑表\nplanned_date / actual_date\n"
        "diff_days / reason_category",
        fc=C["orange_l"], ec=C["orange"], fs=7.4)
    arrow(ax, (28.2, 83.5), (31.6, 83.5))
    arrow(ax, (62.2, 83.5), (65.6, 83.5))
    # 下挂表
    box(ax, 66, 55, 32, 13,
        "project_workflow_steps\n环节步骤表\nstep_name / assigned_to\ncompleted / completed_at",
        fc="#FFFFFF", ec=C["orange"], fs=7.4)
    arrow(ax, (82, 75.6), (82, 68.6), color=C["orange"])
    box(ax, 32, 55, 30, 13, "project_todos\n任务待办表\ncreated_at / done_at / status",
        fc="#FFFFFF", ec=C["primary"], fs=7.6)
    arrow(ax, (47, 75.6), (47, 68.6))
    # M-24：version_no -> version；updated_at -> uploaded_at（project_files 表真实列名）
    box(ax, 2, 55, 26, 13, "project_files\n项目文件与版本\nversion / uploaded_at",
        fc="#FFFFFF", ec=C["primary"], fs=7.6)
    arrow(ax, (15, 75.6), (15, 68.6))
    # KPI 与审计
    box(ax, 2, 35, 26, 13,
        "workflow_kpi\ngenerations / revision_rounds\noutput_chars / completed_count",
        fc=C["green_l"], ec=C["green"], fs=7.4)
    box(ax, 32, 35, 30, 13,
        "audit_runs\nstarted_at / completed_at\noverall_score / overall_status",
        fc=C["green_l"], ec=C["green"], fs=7.6)
    box(ax, 66, 35, 32, 13,
        "admin_audit_log / timeline_diff_log\n操作审计与时间轴变更留痕\n（可追溯性证据）",
        fc=C["green_l"], ec=C["green"], fs=7.6)
    for x in (15, 47, 82):
        arrow(ax, (x, 54.6), (x, 48.6), color=C["green"], lw=1.0, ls="--")
    # 派生指标（仅表6-1 体系内的 6 项）
    ax.add_patch(Rectangle((1.2, 2), 97.6, 27, facecolor="#FFF8E7",
                           edgecolor="#BF9000", lw=1.0, ls="--"))
    ax.text(2.8, 26.4, "可计算的过程指标（第6章表6-1 指标体系的子集）", fontsize=9.2,
            color="#7F6000", fontweight="bold")
    metrics = [
        ("流程周期", "actual_end_date - actual_start_date（天）"),
        ("计划偏差率", "(实际周期 - 计划周期) / 计划周期（%）"),
        ("里程碑延误", "diff_days 与 reason_category 分布"),
        ("返工强度", "workflow_kpi.revision_rounds（轮）"),
        ("任务响应", "todos.done_at - created_at（小时）"),
        ("留痕完整率", "有审计记录的关键操作 / 关键操作总数（%）"),
    ]
    for i, (name, formula) in enumerate(metrics):
        row, col = divmod(i, 2)
        x = 3.4 + col * 48.0
        y = 21.0 - row * 5.2
        ax.text(x, y, "·", fontsize=11, color="#BF9000", va="center")
        ax.text(x + 1.4, y, name, fontsize=8.2, color="#1A1A1A", va="center",
                fontweight="bold")
        ax.text(x + 12.6, y, formula, fontsize=7.2, color=C["gray"], va="center")
    p = f"{OUT}/图5-2_流程数据模型与可算指标.png"
    fig.savefig(p)
    plt.close(fig)
    selfcheck("图5-2", p)


# ================================================================ 图1-1（可选）
def fig11_route():
    """图1-1 研究技术路线图：方法标签改为正文实际使用的方法（M-23）。"""
    backup("图01_研究技术路线图.png")
    fig, ax = canvas(8.8, 8.6)
    rows = [
        ("研究背景：数字经济 + 数据要素市场化 + 行业转型压力", "", C["light"], C["primary"]),
        ("文献综述 → 研究缺口：招标代理行业微观流程再造缺乏实证", "文献研究法", "#FFFFFF", C["primary"]),
        ("理论基础：BPR ｜ TOE 框架 ｜ 动态能力理论", "理论演绎", C["light"], C["primary"]),
        ("现状调查：问卷 185 份 + 访谈 10 人 + 系统数据", "案例研究·调查法", "#FFFFFF", C["primary"]),
        ("问题诊断：工具集成度低｜流程冗余浪费｜员工技能不足", "统计分析法", C["orange_l"], C["orange"]),
        ("成因剖析：管理机制｜技术基础｜环境脱节｜培训体系", "鱼骨图·访谈归纳", "#FFFFFF", C["primary"]),
        ("方案设计与系统实现：架构—模块—接口—试点推广", "设计科学研究", C["green_l"], C["green"]),
        ("效果评估：指标体系 + 三情景测算 + 敏感性分析", "情景测算·敏感性分析", C["green_l"], C["green"]),
        ("结论与展望：为招标代理行业提供可复制的再造范式", "归纳总结", C["light"], C["primary"]),
    ]
    n = len(rows)
    top, bot = 93.0, 6.0
    step = (top - bot) / (n - 1)
    x0, w, h = 17.0, 56.0, 6.6
    for i, (txt, method, fc, ec) in enumerate(rows):
        y = top - i * step
        box(ax, x0, y - h / 2, w, h, txt, fc=fc, ec=ec, fs=9.0, lw=1.3,
            bold=(i in (0, n - 1)))
        if method:
            ax.text(x0 + w + 3.0, y, method, ha="left", va="center", fontsize=7.9,
                    color=C["gray"], style="italic")
        if i < n - 1:
            arrow(ax, (x0 + w / 2, y - h / 2 - 0.3),
                  (x0 + w / 2, y - step + h / 2 + 0.3), color="#8497B0", lw=1.3)
    stages = [("提出问题", 0, 0), ("理论构建", 1, 2), ("现状与问题", 3, 4),
              ("成因剖析", 5, 5), ("方案与验证", 6, 7), ("得出结论", 8, 8)]
    for name, a, b in stages:
        ytop, ybot = top - a * step + h / 2 + 0.8, top - b * step - h / 2 - 0.8
        ax.plot([11.5, 11.5], [ybot, ytop], color=C["mid"], lw=6,
                solid_capstyle="butt", alpha=0.9)
        ax.text(8.6, (ytop + ybot) / 2, name, ha="center", va="center", fontsize=8.4,
                color=C["primary"], rotation=90, fontweight="bold")
    p = f"{OUT}/图1-1_研究技术路线图.png"
    fig.savefig(p)
    plt.close(fig)
    selfcheck("图1-1", p)


if __name__ == "__main__":
    print("font:", plt.rcParams["font.sans-serif"][0])
    for fn in (fig63, fig62, fig61, fig02, fig52, fig11_route):
        print(f"-- {fn.__name__}: {fn.__doc__.strip().splitlines()[0]}")
        fn()
    print("\n===== 汇总 =====")
    bad = 0
    for tag, name, size, px, ink in RESULTS:
        flag = "OK" if (size >= 100 * 1024 and ink >= 0.01) else "FAIL"
        if flag == "FAIL":
            bad += 1
        print(f"{flag:4s} {tag:6s} {name:52s} {size/1024:7.1f}KB  {px[0]}x{px[1]}  ink={ink*100:4.1f}%")
    print(f"共 {len(RESULTS)} 张，失败 {bad} 张")
    sys.exit(1 if bad else 0)
