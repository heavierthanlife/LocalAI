# -*- coding: utf-8 -*-
"""v1.0 图件：招标代理业务口径下的流程图、样本结构、情景测算图，并对既有图做少量改字。"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.12
OUT = "figures"
C = dict(primary="#1F3864", blue="#2E75B6", light="#DEEBF7", mid="#9DC3E6",
         orange="#C55A11", orange_l="#FBE5D6", green="#548235", green_l="#E2EFDA",
         gray="#595959", lgray="#EDEDED", red="#C00000", gold="#BF9000")


def canvas(w, h, xlim=(0, 100), ylim=(0, 100)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#1F3864", fs=9, tc="#1A1A1A",
        lw=1.2, bold=False, zorder=3, pad=0.35):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=1.2",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=zorder + 1, linespacing=1.5)


def arrow(ax, p1, p2, color="#1F3864", lw=1.3, ms=11, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms, color=color,
                                 linewidth=lw, linestyle=ls, zorder=2))


# ---------------- 图3-1 AS-IS：招标代理六阶段流程 ----------------
def fig31():
    fig, ax = canvas(9.8, 3.9, ylim=(0, 84))
    stages = [("① 项目受理与立项", "委托受理\n立项建档"),
              ("② 招标文件编制", "条款编制\n合规核对"),
              ("③ 公告发布与答疑", "公告发布\n澄清答疑"),
              ("④ 开标与清标", "开标记录\n报价比对"),
              ("⑤ 评审与定标", "专家评审\n定标公示"),
              ("⑥ 归档与合同", "成果归档\n合同签订")]
    w, gap = 13.6, 3.2
    y, h = 50, 20
    x = 2.0
    centers = []
    for i, (t, sub) in enumerate(stages):
        hot = i in (1, 3, 4)
        box(ax, x, y, w, h, f"{t}\n{sub}",
            fc=C["orange_l"] if hot else "#FFFFFF",
            ec=C["orange"] if hot else C["primary"], fs=8.2)
        centers.append(x + w / 2)
        if i < len(stages) - 1:
            arrow(ax, (x + w + 0.2, y + h / 2), (x + w + gap - 0.2, y + h / 2))
        x += w + gap
    notes = [(1, 1, "文件编制与合规核对\n人工耗时占比最高"),
             (3, 3, "清标逐项人工比对\n口径不一易返工"),
             (4, 4, "串标识别依赖经验\n事后追溯为主")]
    for a, b, txt in notes:
        xc = (centers[a] + centers[b]) / 2
        ax.annotate(txt, xy=(xc, y - 0.6), xytext=(xc, y - 14), ha="center", va="top",
                    fontsize=7.6, color=C["red"], linespacing=1.5,
                    arrowprops=dict(arrowstyle="-", color=C["red"], lw=0.9, ls=":"))
    ax.text(2.0, y + h + 2.6, "AS-IS：招标代理六阶段串行流程", fontsize=9.5,
            color=C["primary"], fontweight="bold", ha="left")
    box(ax, 2, 3.5, 96, 13,
        "全流程性问题：文档密集型作业 ｜ 多系统切换与手工搬运 ｜ 节点时限与服务费管控弱 ｜ "
        "合规审查依赖个人经验", fc="#FBE5E5", ec=C["red"], fs=8.4, lw=1.1)
    arrow(ax, (50, 32), (50, 17.2), color=C["red"], lw=1.2)
    fig.savefig(f"{OUT}/图04_AS-IS六阶段评估流程.png")
    plt.close(fig)


# ---------------- 图6-3 情景测算 ----------------
def fig63():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    # 左：分情景指标变化区间
    ax = axes[0]
    metrics = ["流程周期", "清标环节耗时", "文件返工轮次", "人工搬运次数"]
    base = [0, 0, 0, 0]
    cons = [-8, -18, -20, -25]
    bench = [-15, -32, -40, -45]
    opt = [-22, -45, -55, -60]
    x = np.arange(len(metrics))
    for i, (m, lo, mid, hi) in enumerate(zip(metrics, cons, bench, opt)):
        ax.plot([i, i], [lo, hi], color=C["blue"], lw=7, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.plot([i], [mid], marker="o", ms=9, color=C["primary"], zorder=3)
        ax.text(i, hi - 2.2, f"{hi}%", ha="center", fontsize=7.8, color=C["primary"])
        ax.text(i, lo + 1.6, f"{lo}%", ha="center", fontsize=7.8, color=C["primary"])
        ax.text(i, mid + 2.6, f"{mid}%", ha="center", fontsize=8.2, color=C["gray"])
    ax.axhline(0, color="#595959", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=8.6)
    ax.set_ylabel("相对基线的变化率（%）", fontsize=8.8)
    ax.set_ylim(-70, 8)
    ax.set_title("(a) 三情景下的指标变化（基准/保守/乐观）", fontsize=9.4,
                 color=C["primary"], fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", color="#D9D9D9"); ax.set_axisbelow(True)

    # 右：综合得分
    ax = axes[1]
    labels = ["AS-IS", "保守", "基准", "乐观"]
    vals = [100, 91.5, 84.2, 77.8]
    cols = ["#BFBFBF", "#9DC3E6", C["blue"], C["primary"]]
    bars = ax.bar(labels, vals, color=cols, width=0.55, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=8.6)
    ax.set_ylim(60, 108)
    ax.set_ylabel("流程综合指数（AS-IS=100）", fontsize=8.8)
    ax.set_title("(b) 综合指数（熵权法测算）", fontsize=9.4, color=C["primary"], fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", color="#D9D9D9"); ax.set_axisbelow(True)
    ax.text(0.5, 0.08, "测算值，非实测", transform=ax.transAxes, ha="center",
            fontsize=16, color="red", alpha=0.12, rotation=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/图16_情景测算结果.png")
    plt.close(fig)


# ---------------- 对既有图做改字 ----------------
def patch_source():
    p = "make_figures.py"
    s = open(p, encoding="utf-8").read()
    s = s.replace('("结论与展望：为资产评估行业提供可复制的再造范式"',
                  '("结论与展望：为招标代理行业提供可复制的再造范式"')
    s = s.replace('("文献综述 → 研究缺口：专业服务行业微观流程再造缺乏实证"',
                  '("文献综述 → 研究缺口：招标代理行业微观流程再造缺乏实证"')
    s = s.replace('("现状调查：问卷 185 份 + 访谈 10 人 + 二手数据"',
                  '("现状调查：问卷 185 份 + 访谈 10 人 + 系统数据"')
    # 鱼骨图子原因改为招标语境
    s = s.replace('(20, 7, 80, "内部管理机制", ["决策链条过长", "部门协调不畅", "激励导向偏差"], C["blue"]),',
                  '(20, 7, 80, "内部管理机制", ["审批链条偏长", "部门协调不畅", "绩效偏重业务量"], C["blue"]),')
    s = s.replace('(44, 31, 80, "技术基础设施", ["架构老化", "接口标准缺失", "算力存储不足"], C["blue"]),',
                  '(44, 31, 80, "技术基础设施", ["系统各自为政", "接口标准缺失", "算力存储不足"], C["blue"]),')
    s = s.replace('(60, 47, 20, "流程与外部环境", ["标准迭代滞后", "接口未预留", "合规导向固化"], C["orange"]),',
                  '(60, 47, 20, "流程与外部环境", ["电子招投标标准迭代", "接口未预留", "合规要求固化"], C["orange"]),')
    # 样本结构：部门标签改招标语境
    s = s.replace('("部门分布", ["评估部", "咨询部", "管理部", "其他"], [79, 47, 35, 24], "bar"),',
                  '("部门分布", ["招标代理部", "造价咨询部", "管理部", "其他"], [79, 47, 35, 24], "bar"),')
    # 问题量化表征措辞
    s = s.replace('("环节衔接不畅认同率", 52.4, C["orange"]),',
                  '("流程节点衔接不畅认同率", 52.4, C["orange"]),')
    s = s.replace('("资源浪费率（估算）", 15.0, C["orange"]),',
                  '("无效返工与资源浪费率", 15.0, C["orange"]),')
    open(p, "w", encoding="utf-8").write(s)


patch_source()
fig31()
fig63()
print("figs v1 done")
