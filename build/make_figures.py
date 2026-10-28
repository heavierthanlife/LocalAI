# -*- coding: utf-8 -*-
"""生成 MBA 论文图表包（图1–图11）。
数据来源：论文 docx 中已有的 10 张表格与正文描述，未新增任何数据。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.12

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

C = dict(
    primary="#1F3864", blue="#2E75B6", light="#DEEBF7", mid="#9DC3E6",
    orange="#C55A11", orange_l="#FBE5D6", green="#548235", green_l="#E2EFDA",
    gray="#595959", lgray="#EDEDED", red="#C00000", gold="#BF9000",
)


def canvas(w, h, xlim=(0, 100), ylim=(0, 100)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#1F3864", fs=10, tc="#1A1A1A",
        lw=1.2, bold=False, ls="-", va="center", ha="center", zorder=3, pad=0.35):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=1.2",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha=ha, va=va, fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=zorder + 1, linespacing=1.5)
    return p


def arrow(ax, p1, p2, color="#1F3864", lw=1.4, style="-|>", rad=0.0, ls="-", zorder=2, ms=12):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, color=color,
                        linewidth=lw, connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=zorder)
    ax.add_patch(a)
    return a


# ============================ 图1 技术路线图 ============================
def fig01():
    fig, ax = canvas(8.8, 8.6)
    rows = [
        ("研究背景：数字经济 + 数据要素市场化 + 行业转型压力", "", C["light"], C["primary"]),
        ("文献综述 → 研究缺口：招标代理行业微观流程再造缺乏实证", "文献研究法", "#FFFFFF", C["primary"]),
        ("理论基础：BPR ｜ TOE 框架 ｜ 动态能力理论", "理论演绎", C["light"], C["primary"]),
        ("现状调查：问卷 185 份 + 访谈 10 人 + 系统数据", "案例研究·调查法", "#FFFFFF", C["primary"]),
        ("问题诊断：工具集成度低｜流程冗余浪费｜员工技能不足", "统计分析法", C["orange_l"], C["orange"]),
        ("成因剖析：管理机制｜技术基础｜环境脱节｜培训体系", "鱼骨图·SWOT", "#FFFFFF", C["primary"]),
        ("方案设计与系统实现：架构—模块—接口—试点推广", "设计科学研究", C["green_l"], C["green"]),
        ("效果评估：指标体系 + 前后对比检验 + 综合效果评价", "配对检验·ITS", C["green_l"], C["green"]),
        ("结论与展望：为招标代理行业提供可复制的再造范式", "归纳总结", C["light"], C["primary"]),
    ]
    n = len(rows)
    top, bot = 93.0, 6.0
    step = (top - bot) / (n - 1)
    x0, w, h = 17.0, 56.0, 6.6
    for i, (txt, method, fc, ec) in enumerate(rows):
        y = top - i * step
        box(ax, x0, y - h / 2, w, h, txt, fc=fc, ec=ec, fs=9.0, lw=1.3, bold=(i in (0, n - 1)))
        if method:
            ax.text(x0 + w + 3.0, y, method, ha="left", va="center", fontsize=7.9,
                    color=C["gray"], style="italic")
        if i < n - 1:
            arrow(ax, (x0 + w / 2, y - h / 2 - 0.3), (x0 + w / 2, y - step + h / 2 + 0.3),
                  color="#8497B0", lw=1.3)
    stages = [("提出问题", 0, 0), ("理论构建", 1, 2), ("现状与问题", 3, 4),
              ("成因剖析", 5, 5), ("方案与验证", 6, 7), ("得出结论", 8, 8)]
    for name, a, b in stages:
        ytop, ybot = top - a * step + h / 2 + 0.8, top - b * step - h / 2 - 0.8
        ax.plot([11.5, 11.5], [ybot, ytop], color=C["mid"], lw=6, solid_capstyle="butt", alpha=0.9)
        ax.text(8.6, (ytop + ybot) / 2, name, ha="center", va="center", fontsize=8.4,
                color=C["primary"], rotation=90, fontweight="bold")
    fig.savefig(f"{OUT}/图01_研究技术路线图.png")
    plt.close(fig)


# ============================ 图2 理论整合框架 ============================
def fig02():
    fig, ax = canvas(8.8, 5.9)
    ax.text(50, 97.0, "理论基础", ha="center", fontsize=9.6, color=C["gray"], fontweight="bold")
    theories = [
        (3, "业务流程再造理论\nBPR ｜ ESIA 原则", C["light"], C["primary"]),
        (34.5, "TOE 框架\n技术 · 组织 · 环境", C["orange_l"], C["orange"]),
        (66, "动态能力理论\n感知 · 抓取 · 重构", C["green_l"], C["green"]),
    ]
    for x, t, fc, ec in theories:
        box(ax, x, 83, 28, 11, t, fc=fc, ec=ec, fs=9.6, bold=True)
        arrow(ax, (x + 14, 82.6), (x + 14, 76.9), color="#8497B0", lw=1.2)
    box(ax, 14, 66.5, 72, 10, "资产评估工作流程数字化再造分析框架", fc=C["primary"],
        ec=C["primary"], fs=12.5, tc="white", bold=True)
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
    fig.savefig(f"{OUT}/图02_理论整合分析框架.png")
    plt.close(fig)


# ============================ 图3 TOE 映射矩阵 ============================
def fig03():
    fig, ax = canvas(10.2, 4.9)
    cols = ["维度", "现状表征", "核心问题", "成因诊断", "对策设计", "观测指标"]
    rows = [
        ("技术（T）", "多工具并存\n集成度低", "数据孤岛\n手工搬运", "架构老化\n接口缺失",
         "平台化\n微服务+API", "自动流转率\n接口成功率", C["light"], C["primary"]),
        ("组织（O）", "层级式管理\n绩效偏传统", "多轮审批\n流程冗余", "决策链长\n缺统一标准",
         "流程精简\nBPR 治理", "审批轮次\n一次通过率", C["green_l"], C["green"]),
        ("环境（E）", "数字中国+数据要素\n政策要求", "与外部标准\n脱节", "流程迭代\n滞后",
         "政策对接\n标准前置", "合规覆盖率\n电子签章率", C["orange_l"], C["orange"]),
        ("动态能力", "感知强\n抓取弱", "能力未转化", "机制不闭环", "培训+认证\n持续迭代",
         "培训覆盖率\n功能采纳率", "#F2F2F2", C["gray"]),
    ]
    x0 = 1.0
    colw = [12.0, 15.0, 14.0, 14.0, 16.0, 16.0]
    ytop, rowh = 88.0, 17.0
    x = x0
    for j, c in enumerate(cols):
        ax.add_patch(Rectangle((x, ytop), colw[j], 9, facecolor=C["primary"], edgecolor="white", lw=1.5))
        ax.text(x + colw[j] / 2, ytop + 4.5, c, ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")
        x += colw[j]
    for i, (dim, c1, c2, c3, c4, c5, fc, ec) in enumerate(rows):
        y = ytop - (i + 1) * rowh
        x = x0
        for j, txt in enumerate([dim, c1, c2, c3, c4, c5]):
            bg = fc if j == 0 else "#FFFFFF"
            ax.add_patch(Rectangle((x, y), colw[j], rowh, facecolor=bg, edgecolor="#BFBFBF", lw=0.9))
            ax.text(x + colw[j] / 2, y + rowh / 2, txt, ha="center", va="center",
                    fontsize=8.4, color=ec if j == 0 else "#1A1A1A",
                    fontweight="bold" if j == 0 else "normal", linespacing=1.5)
            x += colw[j]
    ax.text(x0 + 2, 2.0, "注：横读为“技术—组织—环境”维度在诊断、归因、对策与评价中的对应关系，",
            fontsize=7.8, color=C["gray"], ha="left")
    ax.text(x0 + 2, -3.0, "　　　纵读为同一维度从现状到指标的逻辑链条。", fontsize=7.8, color=C["gray"], ha="left")
    ax.set_ylim(-6, 100)
    fig.savefig(f"{OUT}/图03_TOE维度映射矩阵.png")
    plt.close(fig)


# ============================ 图4 AS-IS 流程图 ============================
def fig04():
    fig, ax = canvas(9.8, 3.4, ylim=(0, 80))
    stages = [
        ("① 项目受理", "审查资料\n签订合同"),
        ("② 现场调查", "实地勘察\n数据采集"),
        ("③ 数据分析", "模型估算\n价值判断"),
        ("④ 报告编制", "撰写报告\n内容校核"),
        ("⑤ 审核复核", "多级审核\n修改确认"),
        ("⑥ 交付跟进", "报告交付\n后续咨询"),
    ]
    w, gap = 13.6, 3.2
    y, h = 50, 20
    x = 2.0
    centers = []
    for i, (t, sub) in enumerate(stages):
        fc = C["orange_l"] if i in (1, 2, 4) else "#FFFFFF"
        ec = C["orange"] if i in (1, 2, 4) else C["primary"]
        box(ax, x, y, w, h, f"{t}\n{sub}", fc=fc, ec=ec, fs=8.4)
        centers.append(x + w / 2)
        if i < len(stages) - 1:
            arrow(ax, (x + w + 0.2, y + h / 2), (x + w + gap - 0.2, y + h / 2),
                  color=C["primary"], lw=1.3)
        x += w + gap
    notes = [(1, 2, "人工传输 3–5 天\n数据需 Excel 中转"), (4, 4, "3 轮审批\n平均延时 2 天")]
    for a, b, txt in notes:
        xc = (centers[a] + centers[b]) / 2
        ax.annotate(txt, xy=(xc, y - 0.6), xytext=(xc, y - 15), ha="center", va="top",
                    fontsize=7.8, color=C["red"], linespacing=1.5,
                    arrowprops=dict(arrowstyle="-", color=C["red"], lw=0.9, ls=":"))
    ax.text(2.0, y + h + 2.6, "AS-IS：六阶段串行流程", fontsize=9.5, color=C["primary"],
            fontweight="bold", ha="left")
    box(ax, 2, 3.5, 96, 13,
        "全流程性问题：纸质文档占比 40% ｜ 环节衔接不畅认同率 52.4% ｜ 资源浪费率约 15% ｜ "
        "工具集成满意率仅 45.9%", fc="#FBE5E5", ec=C["red"], fs=8.6, lw=1.1)
    arrow(ax, (50, 32), (50, 17.2), color=C["red"], lw=1.2, ms=11)
    fig.savefig(f"{OUT}/图04_AS-IS六阶段评估流程.png")
    plt.close(fig)


# ============================ 图5 问题量化表征 ============================
def fig05():
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    items = [
        ("数字化工具使用率", 68.6, C["blue"]),
        ("工具集成满意率", 45.9, C["blue"]),
        ("流程节点衔接不畅认同率", 52.4, C["orange"]),
        ("纸质文档使用占比", 40.0, C["orange"]),
        ("无效返工与资源浪费率", 15.0, C["orange"]),
        ("数字化培训覆盖率", 55.0, C["green"]),
        ("激励机制认同率", 40.5, C["green"]),
    ]
    labels = [i[0] for i in items][::-1]
    vals = [i[1] for i in items][::-1]
    cols = [i[2] for i in items][::-1]
    y = np.arange(len(items))
    ax.barh(y, vals, color=cols, height=0.62, edgecolor="white", zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + 1.2, yy, f"{v:.1f}%", va="center", fontsize=9, color="#1A1A1A")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.2)
    ax.set_xlim(0, 80)
    ax.set_xlabel("比例（%）", fontsize=9)
    ax.grid(axis="x", ls=":", color="#BFBFBF", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = [plt.Rectangle((0, 0), 1, 1, color=C["blue"]),
           plt.Rectangle((0, 0), 1, 1, color=C["orange"]),
           plt.Rectangle((0, 0), 1, 1, color=C["green"])]
    ax.legend(leg, ["数字化应用（问题：集成不足）", "流程与资源（问题：冗余浪费）", "人力与机制（问题：能力不足）"],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=7.8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/图05_核心问题量化表征.png")
    plt.close(fig)


# ============================ 图6 鱼骨图 ============================
def fig06():
    fig, ax = canvas(10.4, 5.2)
    y0 = 50
    ax.annotate("", xy=(81, y0), xytext=(3, y0),
                arrowprops=dict(arrowstyle="-|>", color=C["primary"], lw=2.2, mutation_scale=16))
    box(ax, 81.5, 42, 17.5, 16, "三大问题\n① 集成度低\n② 流程冗余\n③ 技能不足",
        fc=C["primary"], ec=C["primary"], fs=8.2, tc="white", bold=True)
    bones = [
        (20, 7, 80, "内部管理机制", ["审批链条偏长", "部门协调不畅", "绩效偏重业务量"], C["blue"]),
        (44, 31, 80, "技术基础设施", ["系统各自为政", "接口标准缺失", "算力存储不足"], C["blue"]),
        (60, 47, 20, "流程与外部环境", ["电子招投标标准迭代", "接口未预留", "合规要求固化"], C["orange"]),
        (76, 63, 20, "人力资源培训", ["课程结构失衡", "覆盖不均衡", "无闭环考核"], C["orange"]),
    ]
    for ax_x, tx, ty, title, subs, col in bones:
        ax.plot([ax_x, tx], [y0, ty], color=col, lw=1.8, zorder=2)
        ax.text(tx, ty + (3.0 if ty > y0 else -3.0), title, ha="center",
                va="bottom" if ty > y0 else "top", fontsize=9.2, color=col, fontweight="bold")
        for k, s in enumerate(subs):
            f = 0.35 + 0.20 * k
            bx = ax_x + (tx - ax_x) * f
            by = y0 + (ty - y0) * f
            ax.plot([bx, bx + 6], [by, by], color=col, lw=0.9, alpha=0.85, zorder=2)
            ax.text(bx + 6.8, by, s, va="center", ha="left", fontsize=7.6, color="#404040",
                    zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    fig.savefig(f"{OUT}/图06_问题成因鱼骨图.png")
    plt.close(fig)


# ============================ 图7 样本结构 ============================
def fig07():
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.5))
    data = [
        ("性别分布", ["男性", "女性"], [105, 80], "pie"),
        ("年龄分布", ["30岁以下", "31–40岁", "41–50岁", "51岁以上"], [53, 84, 38, 10], "bar"),
        ("工作年限分布", ["5年以下", "6–10年", "11年以上"], [60, 76, 49], "bar"),
        ("职位层级", ["普通员工", "中层管理", "高层管理"], [109, 51, 25], "bar"),
        ("部门分布", ["招标代理部", "造价咨询部", "管理部", "其他"], [79, 47, 35, 24], "bar"),
        ("地域分布", ["总部（北京）", "分支机构"], [121, 64], "bar"),
    ]
    for ax, (title, labs, vals, kind) in zip(axes.ravel(), data):
        tot = sum(vals)
        if kind == "pie":
            ax.pie(vals, labels=[f"{l}\n{v}人\n{v/tot*100:.1f}%" for l, v in zip(labs, vals)],
                   colors=["#2E75B6", "#9DC3E6"], startangle=90,
                   wedgeprops=dict(width=0.42, edgecolor="white"),
                   textprops=dict(fontsize=8.2), counterclock=False)
        else:
            ax.bar(range(len(vals)), vals, color="#2E75B6", width=0.6, edgecolor="white")
            for i, v in enumerate(vals):
                ax.text(i, v + 2.0, f"{v}\n{v/tot*100:.1f}%", ha="center", va="bottom", fontsize=7.8)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labs, fontsize=8.0)
            ax.set_ylim(0, max(vals) * 1.34)
            ax.set_ylabel("人数", fontsize=8)
            ax.tick_params(axis="y", labelsize=7.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(axis="y", ls=":", color="#D9D9D9")
            ax.set_axisbelow(True)
        ax.set_title(title, fontsize=9.6, color=C["primary"], fontweight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/图07_调查样本结构.png")
    plt.close(fig)


# ============================ 图8 工具使用率与集成度 ============================
def fig08():
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), gridspec_kw=dict(width_ratios=[1.05, 1]))
    labs = ["云计算/大数据", "人工智能/自动化", "区块链/安全工具", "整体集成度"]
    use = [72.4, 58.9, 45.4, 45.9]
    mean = [4.1, 3.6, 3.1, 3.2]
    sd = [0.9, 1.2, 1.3, 1.1]
    y = np.arange(len(labs))[::-1]
    ax = axes[0]
    ax.barh(y, use, color=[C["blue"]] * 3 + [C["orange"]], height=0.58, edgecolor="white")
    for yy, v in zip(y, use):
        ax.text(v + 1.2, yy, f"{v:.1f}", va="center", fontsize=8.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8.4)
    ax.set_xlim(0, 88)
    ax.set_xlabel("使用率（%）", fontsize=8.6)
    ax.set_title("(a) 工具使用率", fontsize=9.6, color=C["primary"], fontweight="bold")
    ax = axes[1]
    ax.barh(y, mean, xerr=sd, color=["#9DC3E6"] * 3 + [C["orange_l"]], height=0.58,
            edgecolor="white", error_kw=dict(ecolor="#595959", lw=1.0, capsize=3))
    for yy, v, s in zip(y, mean, sd):
        ax.text(5.15, yy, f"{v:.1f}±{s:.1f}", va="center", ha="left", fontsize=8.2)
    ax.set_yticks([])
    ax.set_xlim(1, 5.95)
    ax.set_xlabel("均值（Likert 1–5）", fontsize=8.6)
    ax.set_title("(b) 满意度均值与标准差", fontsize=9.6, color=C["primary"], fontweight="bold")
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", ls=":", color="#D9D9D9")
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/图08_数字化工具使用率与集成度.png")
    plt.close(fig)


# ============================ 图9 问题-成因-对策 传导链 ============================
def fig09():
    fig, ax = canvas(10.0, 5.2)
    probs = [(74, "① 数字化工具\n集成度低"), (48, "② 流程冗余\n资源浪费"), (22, "③ 员工意识\n与技能不足")]
    causes = [(82, "内部管理机制滞后", "org"), (62, "技术基础设施薄弱", "tech"),
              (41, "流程设计与\n外部环境脱节", "env"), (20, "人力资源培训\n体系不健全", "org")]
    fixes = [(82, "构建数字化\n工作活动平台", "tech"), (62, "流程再造与\n集成设计", "tech"),
             (41, "政策对接与\n流程精简", "env"), (20, "分层分类\n技能培训体系", "org")]
    for y, t in probs:
        box(ax, 2, y - 7, 20, 14, t, fc="#FBE5E5", ec=C["red"], fs=8.8, lw=1.3)
    cmap = dict(org=(C["green"], C["green_l"]), tech=(C["blue"], C["light"]), env=(C["orange"], C["orange_l"]))
    for y, t, k in causes:
        ec, fc = cmap[k]
        box(ax, 33, y - 6.5, 24, 13, t, fc=fc, ec=ec, fs=8.6)
    for y, t, k in fixes:
        ec, fc = cmap[k]
        box(ax, 70, y - 6.5, 26, 13, t, fc=fc, ec=ec, fs=8.6, bold=True)
    ax.text(12, 96, "存在问题", ha="center", fontsize=10, color=C["red"], fontweight="bold")
    ax.text(45, 96, "成因诊断", ha="center", fontsize=10, color=C["primary"], fontweight="bold")
    ax.text(83, 96, "对策设计", ha="center", fontsize=10, color=C["green"], fontweight="bold")
    links = [(82, 74, -0.20), (82, 48, -0.25), (62, 74, -0.15), (62, 48, -0.12),
             (41, 48, 0.15), (41, 22, 0.20), (20, 22, 0.25), (20, 74, -0.30)]
    for cy, py, rad in links:
        arrow(ax, (33, cy), (22.4, py), color="#A6A6A6", lw=1.0, rad=rad, ms=9)
    for cy, fy in [(82, 82), (62, 62), (41, 41), (20, 20)]:
        arrow(ax, (57, cy), (69.6, fy), color=C["primary"], lw=1.4, ms=11)
    ax.text(50, 4, "灰线：问题 ← 成因（归因关系）；深蓝线：成因 → 对策（对应关系）",
            fontsize=7.8, color=C["gray"], ha="center")
    fig.savefig(f"{OUT}/图09_问题-成因-对策传导链.png")
    plt.close(fig)


# ============================ 图10 AS-IS vs TO-BE 指数 ============================
def fig10():
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    cats = ["流程周期", "复核审批轮次", "纸质文档占比"]
    base = [100, 100, 100]
    target = [82.5, 33.3, 25.0]
    x = np.arange(len(cats))
    w = 0.34
    ax.bar(x - w / 2, base, w, color="#BFBFBF", edgecolor="white", label="AS-IS（现状=100）")
    ax.bar(x + w / 2, target, w, color=C["blue"], edgecolor="white", label="TO-BE（目标指数）")
    for xx, v in zip(x - w / 2, base):
        ax.text(xx, v + 3, "100", ha="center", fontsize=8.4, color=C["gray"])
    red = ["-17.5%", "-66.7%", "-75.0%"]
    for xx, v, r in zip(x + w / 2, target, red):
        ax.text(xx, v + 3, f"{v:.0f}\n{r}", ha="center", fontsize=8.0, color=C["blue"])
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("指数（AS-IS = 100）", fontsize=9)
    ax.set_ylim(0, 122)
    ax.legend(fontsize=8.4, frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", color="#D9D9D9")
    ax.set_axisbelow(True)
    ax.text(0.5, 0.45, "目标值（示意，待实测替换）", transform=ax.transAxes, fontsize=17,
            color="red", alpha=0.12, ha="center", rotation=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/图10_再造前后目标指标对比.png")
    plt.close(fig)


# ============================ 图11 指标体系框架 ============================
def fig11():
    fig, ax = canvas(10.2, 5.2)
    box(ax, 30, 88, 40, 9, "工作流程数字化再造效果（目标层）", fc=C["primary"], ec=C["primary"],
        fs=10.5, tc="white", bold=True)
    crit = [
        ("流程效率", ["周期 P50/P90", "审批轮次", "一次通过率"], C["light"], C["primary"]),
        ("集成水平", ["自动流转率", "接口成功率", "重复录入字段"], "#FFFFFF", C["primary"]),
        ("资源节约", ["单项目工时", "差旅次数", "无纸化率"], C["green_l"], C["green"]),
        ("业务价值", ["准时交付率", "客户满意度", "单项目毛利"], C["green_l"], C["green"]),
        ("组织能力", ["培训覆盖率", "认证通过率", "功能采纳率"], C["orange_l"], C["orange"]),
    ]
    w, gap = 18.0, 1.6
    x = 2.0
    for name, inds, fc, ec in crit:
        box(ax, x, 63, w, 10, name, fc=fc, ec=ec, fs=9.8, bold=True)
        arrow(ax, (50, 87.6), (x + w / 2, 73.6), color="#A6A6A6", lw=1.0, ls="--", ms=9)
        body = "\n".join(f"· {i}" for i in inds)
        box(ax, x + 0.6, 33, w - 1.2, 24, body, fc="#FAFAFA", ec=ec, fs=7.8, lw=0.9, ls=":")
        arrow(ax, (x + w / 2, 62.6), (x + w / 2, 57.4), color=ec, lw=1.2, ms=9)
        x += w + gap
    ax.plot([3, 97], [29, 29], color="#BFBFBF", lw=0.9, ls=":")
    arrow(ax, (50, 28.6), (50, 21.6), color="#BFBFBF", lw=1.0, ms=9)
    box(ax, 6, 8, 88, 13,
        "赋权方法：熵权法（客观）+ AHP（主观）组合赋权\n评价方法：配对检验 ｜ 中断时间序列 ｜ 稳健性检验",
        fc="#F2F2F2", ec=C["gray"], fs=8.6, ls="--")
    fig.savefig(f"{OUT}/图11_再造效果评价指标体系.png")
    plt.close(fig)


for fn in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11]:
    fn()
    print("ok", fn.__name__)
print("done")
