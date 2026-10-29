# -*- coding: utf-8 -*-
"""v1.4 图件：按“Word 中最终插入 15cm 宽”重制全部 16 张 matplotlib 图。

设计口径（关键）：
- 画布宽度 W_IN = 5.906 in = 15 cm，即按最终尺寸 1:1 出图，插入 Word 不再缩放，
  因此图中 fontsize 的 pt 数就是读者看到的 pt 数。
- 正文/标签 12pt，坐标轴刻度 11pt，子图/图内标题 13–15pt，主标题 15–17pt。
- 线宽/箭头/边框/marker 按 1.4–1.6 倍加粗。
- 全部删除图内脚注文字（原以“注”字开头，改由论文图题下的图注承担）；图9 的箭头语义改为小图例。
- 文字宽度用 TextPath 实测后自动换行（wrap()），框高按行数计算（bh()），避免越界/压框。
- 内容与数据严格沿用 v1.0–v1.2 各脚本，不新增、不改动任何数值。

输出：build/figures_v14/（文件名与 build/figures_final/ 完全一致，16 张）
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.textpath import TextToPath
from matplotlib import font_manager
import numpy as np

# ---------------------------------------------------------------- 全局版式
W_IN = 5.906                     # 15 cm
BODY, TICK, SUBT, TITLE, BIG = 12.0, 11.0, 13.0, 15.0, 17.0
LW = 1.7                         # 基础线宽（原 1.1–1.2 → ×1.5）
UPU = 100.0 / (72.0 * W_IN)      # 每 pt 对应的图内单位（1 单位 = 0.059 in）

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = next((p for p in ("/w/fonts/simhei.ttf",
                         os.path.join(HERE, "..", "fonts", "simhei.ttf"))
             if os.path.exists(p)), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
FP = font_manager.FontProperties(fname=FONT) if FONT else font_manager.FontProperties(family="SimHei")
plt.rcParams.update({
    "font.sans-serif": [FP.get_name(), "SimHei"],
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.0,
    "font.size": BODY,
    "axes.titlesize": SUBT,
    "axes.labelsize": BODY,
    "xtick.labelsize": TICK,
    "ytick.labelsize": TICK,
    "legend.fontsize": BODY - 0.5,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,
})

OUT = "figures_v14"
os.makedirs(OUT, exist_ok=True)

C = dict(primary="#1F3864", blue="#2E75B6", light="#DEEBF7", mid="#9DC3E6",
         orange="#C55A11", orange_l="#FBE5D6", green="#548235", green_l="#E2EFDA",
         gray="#595959", lgray="#EDEDED", red="#C00000", gold="#BF9000",
         purple="#5B2C6F", purple_l="#E8DAEF")

RESULTS = []


# ---------------------------------------------------------------- 文字工具
_TTP = TextToPath()


def tw(s, fs):
    """字符串在图坐标单位下的宽度（按字体 advance 实测，与渲染宽度一致）。"""
    if not s:
        return 0.0
    prop = FP.copy()
    prop.set_size(fs)
    w, _h, _d = _TTP.get_text_width_height_descent(s, prop, False)
    return w * UPU


def lh(fs=BODY):
    """单行行高（图内单位）。"""
    return fs * 1.34 * UPU


CLOSERS = "，。、；：）」』”%）,.;:!?！？｜"


def _tokens(line):
    toks, i, n = [], 0, len(line)
    while i < n:
        ch = line[i]
        if ch == " ":
            j = i
            while j < n and line[j] == " ":
                j += 1
            toks.append(line[i:j])
            i = j
        elif re.match(r"[A-Za-z0-9_]", ch):
            j = i
            while j < n and re.match(r"[A-Za-z0-9_.\-/+%±]", line[j]):
                j += 1
            # 长标识符在 _ / - 处切成可断行的小段（贪心拼接，不影响不需断行的情形）
            k = i
            for m in re.finditer(r"[A-Za-z0-9.]+?[_/\-](?=[A-Za-z0-9])", line[i:j]):
                toks.append(line[i + m.start():i + m.end()])
                k = i + m.end()
            if k < j:
                toks.append(line[k:j])
            i = j
        else:
            toks.append(ch)
            i += 1
    return toks


def wrap(s, width, fs=BODY):
    """按实测宽度换行；保留原有换行符，收尾标点不回行首。"""
    lines = []
    for hard in str(s).split("\n"):
        cur = ""
        for t in _tokens(hard):
            if not cur and t.strip() == "":
                continue
            cand = cur + t
            if tw(cand, fs) <= width or not cur:
                cur = cand
            elif t in CLOSERS and tw(cand, fs) <= width + tw(t, fs) * 1.05:
                lines.append((cur + t).rstrip())
                cur = ""
            else:
                lines.append(cur.rstrip())
                cur = "" if t.strip() == "" else t
        lines.append(cur.rstrip())
    return "\n".join(lines)


def nlines(s, width, fs=BODY):
    return wrap(s, width, fs).count("\n") + 1


def bh(s, width, fs=BODY, pad=2.3):
    """容纳文字所需的框高（图内单位）。"""
    return nlines(s, width, fs) * lh(fs) + pad


# ---------------------------------------------------------------- 绘图件
def canvas(h_in):
    H = 100.0 * h_in / W_IN
    fig, ax = plt.subplots(figsize=(W_IN, h_in))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.patch.set_facecolor("white")
    return fig, ax, H


def box(ax, x, y, w, h, text, fs=BODY, fc="#FFFFFF", ec=C["primary"],
        tc="#1A1A1A", lw=LW, bold=False, ls="-", zorder=3, rad=0.7, padx=1.2):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={rad}",
                           facecolor=fc, edgecolor=ec, linewidth=lw,
                           linestyle=ls, zorder=zorder)
    patch._container = True
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, wrap(text, w - 2 * padx, fs), ha="center",
            va="center", fontsize=fs, color=tc, zorder=zorder + 1,
            fontweight="bold" if bold else "normal", linespacing=1.34)


def arrow(ax, p1, p2, color=C["primary"], lw=LW * 1.05, ms=13, ls="-", rad=0.0, zorder=2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, linewidth=lw, linestyle=ls, zorder=zorder,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    selfcheck(name, p)


def selfcheck(name, path):
    from PIL import Image
    size = os.path.getsize(path)
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    ink = float((arr < 245).any(axis=2).mean())
    RESULTS.append(dict(name=name, bytes=size, px=im.size,
                        inch=(im.size[0] / 300.0, im.size[1] / 300.0), ink=ink))
    flag = "OK " if (size >= 100 * 1024 and ink >= 0.01) else "FAIL"
    print(f"  [{flag}] {name:42s} {size/1024:6.1f}KB {im.size[0]}x{im.size[1]} ink={ink*100:4.1f}%")


# ================================================================ 图01 技术路线图
def fig01():
    name = "图01_研究技术路线图.png"
    rows = [
        ("研究背景：数字经济 + 数据要素市场化\n+ 行业转型压力", ""),
        ("文献综述 → 研究缺口：招标代理行业\n微观流程再造缺乏实证", "文献研究法"),
        ("理论基础：BPR ｜ TOE 框架 ｜ 动态能力理论", "理论演绎"),
        ("现状调查：问卷 185 份 + 访谈 10 人\n+ 系统数据", "案例研究·调查法"),
        ("问题诊断：工具集成度低｜流程冗余浪费\n｜员工技能不足", "统计分析法"),
        ("成因剖析：管理机制｜技术基础\n｜环境脱节｜培训体系", "鱼骨图·访谈归纳"),
        ("方案设计与系统实现：架构—模块\n—接口—试点推广", "设计科学研究"),
        ("效果评估：指标体系 + 三情景测算\n+ 敏感性分析", "情景测算·敏感性分析"),
        ("结论与展望：为招标代理行业提供\n可复制的再造范式", "归纳总结"),
    ]
    X0, BW = 9.5, 64.0
    MX, MW = 77.5, 21.0
    GAP, TOP, BOT = 1.8, 3.0, 3.0
    hs = [max(bh(t, BW), bh(m, MW) if m else 0.0, 8.0) for t, m in rows]
    H_units = TOP + BOT + sum(hs) + GAP * (len(rows) - 1)
    fig, ax, H = canvas(H_units * W_IN / 100.0)
    y = H - TOP
    tops, bots = [], []
    for i, (t, m) in enumerate(rows):
        h = hs[i]
        fc, ec = (C["light"], C["primary"]) if i in (0, len(rows) - 1) else ("#FFFFFF", C["primary"])
        if i == 4:
            fc, ec = C["orange_l"], C["orange"]
        if i == 7:
            fc, ec = C["green_l"], C["green"]
        if i == 6:
            fc, ec = C["green_l"], C["green"]
        box(ax, X0, y - h, BW, h, t, fs=BODY, fc=fc, ec=ec, bold=(i in (0, len(rows) - 1)),
            lw=LW * 1.05)
        if m:
            ax.text(MX + MW / 2, y - h / 2, wrap(m, MW - 1.2), ha="center", va="center",
                    fontsize=BODY, color=C["gray"], style="italic", linespacing=1.34)
        tops.append(y)
        bots.append(y - h)
        if i < len(rows) - 1:
            arrow(ax, (X0 + BW / 2, y - h - 0.2), (X0 + BW / 2, y - h - GAP + 0.2),
                  color="#8497B0", lw=LW * 0.85, ms=12)
        y = y - h - GAP
    stages = [("提出问题", 0, 0), ("理论构建", 1, 2), ("现状与问题", 3, 4),
              ("成因剖析", 5, 5), ("方案与验证", 6, 7), ("得出结论", 8, 8)]
    for nm, a, b in stages:
        ytop, ybot = tops[a] + 0.5, bots[b] - 0.5
        ax.plot([5.4, 5.4], [ybot, ytop], color=C["mid"], lw=7.5,
                solid_capstyle="butt", alpha=0.95, zorder=1)
        ax.text(2.5, (ytop + ybot) / 2, nm, rotation=90, ha="center", va="center",
                fontsize=BODY, color=C["primary"], fontweight="bold")
    save(fig, name)


# ================================================================ 图02 理论整合框架
def fig02():
    name = "图02_理论整合分析框架.png"
    theories = [
        ("业务流程再造理论\nBPR ｜ ESIA 原则", C["light"], C["primary"]),
        ("TOE 框架\n技术 · 组织 · 环境", C["orange_l"], C["orange"]),
        ("动态能力理论\n感知 · 抓取 · 重构", C["green_l"], C["green"]),
    ]
    tw_ = 30.0
    tx = [1.5, 35.0, 68.5]
    th = max(bh(t, tw_) for t, *_ in theories)
    title_txt = "招标代理业务工作流程数字化再造分析框架"
    title_h = bh(title_txt, 92.0, 16.0) + 0.5
    layers = [
        ("【诊断层】\n工具集成度\n流程冗余度\n员工数字能力", C["light"], C["primary"]),
        ("【解释层】\n技术维度\n组织维度\n环境维度", C["orange_l"], C["orange"]),
        ("【路径层】\nESIA 精简\n平台化集成\n政策对接·培训", C["green_l"], C["green"]),
    ]
    lh_ = max(bh(t, tw_) for t, *_ in layers)
    eff = "再造效果：周期缩短 ｜ 资源节约 ｜ 成本下降 ｜ 动态能力提升"
    eff_h = bh(eff, 92.0, BODY)
    fb = "反馈迭代：绩效审计 → 流程再优化（持续改善闭环）"
    fb_h = bh(fb, 84.0, BODY)
    g = 3.4
    H_units = 4.5 + th + g + title_h + g + lh_ + g + eff_h + g + fb_h + 4.0
    fig, ax, H = canvas(H_units * W_IN / 100.0)
    y = H - 4.5
    ax.text(50, y + 0.2, "理论基础", ha="center", va="bottom", fontsize=BODY,
            color=C["gray"], fontweight="bold")
    y -= 0.6
    for x, (t, fc, ec) in zip(tx, theories):
        box(ax, x, y - th, tw_, th, t, fs=BODY, fc=fc, ec=ec, bold=True, lw=LW * 1.05)
        arrow(ax, (x + tw_ / 2, y - th - 0.3), (x + tw_ / 2, y - th - g + 0.4),
              color="#8497B0", lw=LW * 0.85, ms=11)
    y -= th + g
    box(ax, 4.0, y - title_h, 92.0, title_h, title_txt, fs=16.0, fc=C["primary"],
        ec=C["primary"], tc="white", bold=True)
    y -= title_h + g
    for x, (t, fc, ec) in zip(tx, layers):
        box(ax, x, y - lh_, tw_, lh_, t, fs=BODY, fc=fc, ec=ec, lw=LW * 1.05)
        arrow(ax, (50, y + lh_ + g - 0.4), (x + tw_ / 2, y + 0.6),
              color="#A6A6A6", lw=LW * 0.8, ls="--", ms=11)
    y -= lh_ + g
    box(ax, 4.0, y - eff_h, 92.0, eff_h, eff, fs=BODY, fc="#FFFFFF", ec=C["primary"],
        bold=True, lw=LW * 1.05)
    for x in tx:
        arrow(ax, (x + tw_ / 2, y + eff_h + g - 0.4), (x + tw_ / 2, y + 0.5),
              color="#A6A6A6", lw=LW * 0.9, ms=12)
    y -= eff_h + g
    box(ax, 8.0, y - fb_h, 84.0, fb_h, fb, fs=BODY, fc=C["lgray"], ec=C["gray"],
        ls="--", lw=LW * 0.95)
    arrow(ax, (50, y + fb_h + g - 0.5), (50, y + 0.4), color=C["gray"], lw=LW * 0.9, ls="--", ms=12)
    save(fig, name)


# ================================================================ 图03 TOE 矩阵
def fig03():
    name = "图03_TOE维度映射矩阵.png"
    cols = ["维度", "现状表征", "核心问题", "成因诊断", "对策设计", "观测指标"]
    cw = [13.0, 17.0, 15.0, 15.0, 19.0, 19.0]
    rows = [
        ("技术（T）", "多工具并存\n集成度低", "数据孤岛\n手工搬运", "架构老化\n接口缺失",
         "平台化\n微服务 + API", "自动流转率\n接口成功率", C["light"], C["primary"]),
        ("组织（O）", "层级式管理\n绩效偏传统", "多轮审批\n流程冗余", "决策链长\n缺统一标准",
         "流程精简\nBPR 治理", "审批轮次\n一次通过率", C["green_l"], C["green"]),
        ("环境（E）", "数字中国\n+ 数据要素\n政策要求", "与外部标准\n脱节", "流程迭代\n滞后",
         "政策对接\n标准前置", "合规覆盖率\n电子签章率", C["orange_l"], C["orange"]),
        ("动态能力", "感知强\n抓取弱", "能力未转化", "机制不闭环", "培训 + 认证\n持续迭代",
         "培训覆盖率\n功能采纳率", "#F2F2F2", C["gray"]),
    ]
    x0 = 1.0
    hh = bh("维度", cw[0], SUBT) + 0.6
    rhs = [max(bh(t, cw[j]) for j, t in enumerate(r[:6])) for r in rows]
    H_units = 3.0 + hh + sum(rhs) + 3.0
    fig, ax, H = canvas(H_units * W_IN / 100.0)
    y = H - 3.0
    x = x0
    for j, cn in enumerate(cols):
        ax.add_patch(Rectangle((x, y - hh), cw[j], hh, facecolor=C["primary"],
                               edgecolor="white", lw=1.6))
        ax.text(x + cw[j] / 2, y - hh / 2, cn, ha="center", va="center", fontsize=SUBT,
                color="white", fontweight="bold")
        x += cw[j]
    y -= hh
    for dim, c1, c2, c3, c4, c5, fc, ec in rows:
        h = max(bh(dim, cw[0]), bh(c1, cw[1]), bh(c2, cw[2]), bh(c3, cw[3]),
                bh(c4, cw[4]), bh(c5, cw[5]))
        x = x0
        for j, txt in enumerate([dim, c1, c2, c3, c4, c5]):
            ax.add_patch(Rectangle((x, y - h), cw[j], h,
                                   facecolor=fc if j == 0 else "#FFFFFF",
                                   edgecolor="#BFBFBF", lw=1.0))
            ax.text(x + cw[j] / 2, y - h / 2, wrap(txt, cw[j] - 2.0), ha="center",
                    va="center", fontsize=BODY, color=ec if j == 0 else "#1A1A1A",
                    fontweight="bold" if j == 0 else "normal", linespacing=1.34)
            x += cw[j]
        y -= h
    save(fig, name)


# ================================================================ 图04 AS-IS 六阶段
def fig04():
    name = "图04_AS-IS六阶段评估流程.png"
    stages = [
        ("① 项目受理与立项\n委托受理\n立项建档", False),
        ("② 招标文件编制\n条款编制\n合规核对", True),
        ("③ 公告发布与答疑\n公告发布\n澄清答疑", False),
        ("④ 开标与清标\n开标记录\n报价比对", True),
        ("⑤ 评审与定标\n专家评审\n定标公示", True),
        ("⑥ 归档与合同\n成果归档\n合同签订", False),
    ]
    notes = [(1, 1, "文件编制与合规核对\n人工耗时占比最高"),
             (3, 3, "清标逐项人工比对\n口径不一易返工"),
             (4, 4, "串标识别依赖经验\n事后追溯为主")]
    banner = ("全流程性问题：文档密集型作业 ｜ 多系统切换与手工搬运 ｜ "
              "节点时限与服务费管控弱 ｜ 合规审查依赖个人经验")
    BW, GAP = 14.2, 2.0
    x0 = 1.0
    centers = [x0 + i * (BW + GAP) + BW / 2 for i in range(6)]
    sh = max(bh(t, BW) for t, _ in stages)
    bh_h = bh(banner, 94.0)
    nh_ = max(bh(t, 26.0) for _, _, t in notes)
    H_units = (2.0 + lh(SUBT) + 1.6 + sh + 2.5 + 2 * (nh_ + 2.5) + 2.0 + 0.6
               + bh_h + 3.0)
    fig, ax, H = canvas(H_units * W_IN / 100.0)
    y = H - 2.0
    ax.text(50, y, "AS-IS：招标代理六阶段串行流程", ha="center", va="top",
            fontsize=SUBT, color=C["primary"], fontweight="bold")
    y -= lh(SUBT) + 1.6
    for i, (t, hot) in enumerate(stages):
        x = x0 + i * (BW + GAP)
        box(ax, x, y - sh, BW, sh, t, fs=BODY,
            fc=C["orange_l"] if hot else "#FFFFFF",
            ec=C["orange"] if hot else C["primary"], lw=LW * 1.05)
        if i < 5:
            arrow(ax, (x + BW + 0.15, y - sh / 2), (x + BW + GAP - 0.15, y - sh / 2),
                  lw=LW, ms=12)
    y -= sh + 2.5
    # 注解：两排错开，虚线引导
    rows_n = [(0, [n for n in notes if n[0] in (1, 3)]), (1, [n for n in notes if n[0] == 4])]
    for r, group in rows_n:
        yrow = y - r * (nh_ + 2.5)
        for a, b, txt in group:
            xc = (centers[a] + centers[b]) / 2
            ax.annotate(wrap(txt, 26.0), xy=(xc, y + 0.6), xytext=(xc, yrow),
                        ha="center", va="top", fontsize=BODY, color=C["red"],
                        linespacing=1.34,
                        arrowprops=dict(arrowstyle="-", color=C["red"], lw=1.4, ls=":"))
    y -= 2 * (nh_ + 2.5) + 2.0
    arrow(ax, (50, y + 1.8), (50, y - 0.6), color=C["red"], lw=LW, ms=13)
    box(ax, 1.0, y - bh_h - 0.6, 96.0, bh_h, banner, fs=BODY, fc="#FBE5E5",
        ec=C["red"], lw=LW * 1.05)
    save(fig, name)


# ================================================================ 图05 问题量化
def fig05():
    name = "图05_核心问题量化表征.png"
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
    fig, ax = plt.subplots(figsize=(W_IN, 5.0))
    fig.subplots_adjust(left=0.345, right=0.975, top=0.985, bottom=0.235)
    ax.barh(y, vals, color=cols, height=0.64, edgecolor="white", zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + 1.4, yy, f"{v:.1f}%", va="center", fontsize=BODY, color="#1A1A1A",
                fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=BODY)
    ax.set_xlim(0, 82)
    ax.set_xlabel("比例（%）", fontsize=BODY, labelpad=4)
    ax.tick_params(axis="x", labelsize=TICK)
    ax.grid(axis="x", ls=":", color="#BFBFBF", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = [plt.Rectangle((0, 0), 1, 1, color=C["blue"]),
           plt.Rectangle((0, 0), 1, 1, color=C["orange"]),
           plt.Rectangle((0, 0), 1, 1, color=C["green"])]
    ax.legend(leg, ["数字化应用（问题：集成不足）", "流程与资源（问题：冗余浪费）",
                    "人力与机制（问题：能力不足）"],
              loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1,
              fontsize=BODY, frameon=False, handlelength=1.4,
              handletextpad=0.6, labelspacing=0.35)
    save(fig, name)


# ================================================================ 图06 鱼骨图
def fig06():
    name = "图06_问题成因鱼骨图.png"
    H_units = 108.0
    up = [(6.0, 98.0, 30.0, "内部管理机制",
           ["审批链条偏长", "部门协调不畅", "绩效偏重业务量"], C["blue"]),
          (42.0, 98.0, 58.0, "技术基础设施",
           ["系统各自为政", "接口标准缺失", "算力存储不足"], C["blue"])]
    dn = [(6.0, 8.0, 30.0, "流程与外部环境",
           ["电子招投标标准迭代", "接口未预留", "合规要求固化"], C["orange"]),
          (42.0, 8.0, 58.0, "人力资源培训",
           ["课程结构失衡", "覆盖不均衡", "无闭环考核"], C["orange"])]
    fig, ax, H = canvas(H_units * W_IN / 100.0)
    y0 = 50.0
    ax.annotate("", xy=(75.0, y0), xytext=(2.5, y0),
                arrowprops=dict(arrowstyle="-|>", color=C["primary"], lw=LW * 2.2,
                                mutation_scale=22, shrinkA=0, shrinkB=0))
    box(ax, 76.0, 36.0, 23.0, 22.0, "三大问题\n① 集成度低\n② 流程冗余\n③ 技能不足",
        fs=BODY, fc=C["primary"], ec=C["primary"], tc="white", bold=True, lw=LW * 1.05)
    for group, updown in ((up, 1), (dn, -1)):
        for bx, by, xcross, title, subs, col in group:
            ax.plot([bx, xcross], [by, y0], color=col, lw=LW * 1.25, zorder=2)
            for k, s_ in enumerate(subs):
                f = 0.28 + 0.18 * k
                px = bx + (xcross - bx) * f
                py = by + (y0 - by) * f
                ax.plot([px, px + 4.6], [py, py], color=col, lw=LW * 0.7, zorder=2)
                ax.text(px + 5.4, py, s_, va="center", ha="left", fontsize=BODY,
                        color="#1A1A1A", zorder=4,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
            ax.text(bx - 0.8, by + 1.6 * updown, title, ha="left",
                    va="bottom" if updown > 0 else "top", fontsize=SUBT,
                    color=col, fontweight="bold")
    save(fig, name)


# ================================================================ 图07 样本结构
# ================================================================ 图07 样本结构
def fig07():
    name = "图07_调查样本结构.png"
    data = [
        ("性别分布", ["男性", "女性"], [105, 80], "pie"),
        ("年龄分布", ["30岁以下", "31–40岁", "41–50岁", "51岁以上"], [53, 84, 38, 10], "bar"),
        ("工作年限分布", ["5年以下", "6–10年", "11年以上"], [60, 76, 49], "bar"),
        ("职位层级", ["普通员工", "中层管理", "高层管理"], [109, 51, 25], "bar"),
        ("部门分布", ["招标代理部", "造价咨询部", "管理部", "其他"], [79, 47, 35, 24], "bar"),
        ("地域分布", ["总部（北京）", "分支机构"], [121, 64], "bar"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(W_IN, 6.9))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.955, bottom=0.135,
                        wspace=0.34, hspace=0.70)
    for ax, (title, labs, vals, kind) in zip(axes.ravel(), data):
        tot = sum(vals)
        if kind == "pie":
            ax.pie(vals, labels=[f"{l}\n{v}人\n{v/tot*100:.1f}%" for l, v in zip(labs, vals)],
                   colors=["#2E75B6", "#9DC3E6"], startangle=90,
                   wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.4),
                   textprops=dict(fontsize=BODY), labeldistance=1.02,
                   counterclock=False, radius=0.80)
        else:
            ax.bar(range(len(vals)), vals, color="#2E75B6", width=0.62,
                   edgecolor="white", zorder=2)
            for i, v in enumerate(vals):
                ax.text(i, v + max(vals) * 0.03, f"{v}\n{v/tot*100:.1f}%", ha="center",
                        va="bottom", fontsize=BODY, linespacing=1.3)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labs, fontsize=TICK, rotation=30, ha="right",
                               rotation_mode="anchor")
            ax.set_ylim(0, max(vals) * 1.42)
            ax.set_ylabel("人数", fontsize=BODY, labelpad=2)
            ax.tick_params(axis="y", labelsize=TICK)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(axis="y", ls=":", color="#D9D9D9", zorder=0)
            ax.set_axisbelow(True)
        ax.set_title(title, fontsize=SUBT, color=C["primary"], fontweight="bold", pad=7)
    save(fig, name)


# ================================================================ 图08 工具使用率
def fig08():
    name = "图08_数字化工具使用率与集成度.png"
    labs = ["云计算/大数据", "人工智能/自动化", "区块链/安全工具", "整体集成度"]
    use = [72.4, 58.9, 45.4, 45.9]
    mean = [4.1, 3.6, 3.1, 3.2]
    sd = [0.9, 1.2, 1.3, 1.1]
    y = np.arange(len(labs))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 4.0), gridspec_kw=dict(width_ratios=[1.12, 1]))
    fig.subplots_adjust(left=0.305, right=0.975, top=0.885, bottom=0.185, wspace=0.10)
    ax = axes[0]
    ax.barh(y, use, color=[C["blue"]] * 3 + [C["orange"]], height=0.6,
            edgecolor="white", zorder=2)
    for yy, v in zip(y, use):
        ax.text(v + 1.6, yy, f"{v:.1f}", va="center", fontsize=BODY, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=TICK)
    ax.set_xlim(0, 90)
    ax.set_xlabel("使用率（%）", fontsize=BODY, labelpad=3)
    ax.set_title("(a) 工具使用率", fontsize=SUBT, color=C["primary"], fontweight="bold")
    ax.tick_params(axis="x", labelsize=TICK)
    ax = axes[1]
    ax.barh(y, mean, xerr=sd, color=["#9DC3E6"] * 3 + [C["orange_l"]], height=0.6,
            edgecolor="white", zorder=2,
            error_kw=dict(ecolor="#404040", lw=1.6, capsize=4, capthick=1.6))
    for yy, v, s in zip(y, mean, sd):
        ax.text(5.55, yy, f"{v:.1f}±{s:.1f}", va="center", ha="right", fontsize=BODY)
    ax.set_yticks([])
    ax.set_xlim(1.0, 6.2)
    ax.set_xlabel("均值（Likert 1–5）", fontsize=BODY, labelpad=3)
    ax.set_title("(b) 满意度均值与标准差", fontsize=SUBT, color=C["primary"], fontweight="bold")
    ax.tick_params(axis="x", labelsize=TICK)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(axis="x", ls=":", color="#D9D9D9", zorder=0)
        a.set_axisbelow(True)
    save(fig, name)


# ================================================================ 图09 传导链
def fig09():
    name = "图09_问题-成因-对策传导链.png"
    probs = [(80.0, "① 数字化工具\n集成度低"), (52.0, "② 流程冗余\n资源浪费"),
             (24.0, "③ 员工意识\n与技能不足")]
    causes = [(86.0, "内部管理机制滞后", "org"), (66.0, "技术基础设施薄弱", "tech"),
              (46.0, "流程设计与\n外部环境脱节", "env"), (26.0, "人力资源培训\n体系不健全", "org")]
    fixes = [(86.0, "构建数字化\n工作活动平台", "tech"), (66.0, "流程再造与\n集成设计", "tech"),
             (46.0, "政策对接与\n流程精简", "env"), (26.0, "分层分类\n技能培训体系", "org")]
    PW, PX = 22.0, 1.0
    CW, CX = 26.0, 30.5
    FW, FX = 32.0, 65.0
    ph = max(bh(t, PW) for _, t in probs)
    ch = max(bh(t, CW) for _, t, _ in causes)
    fh = max(bh(t, FW) for _, t, _ in fixes)
    H_units = 108.0
    fig, ax, H = canvas(H_units * W_IN / 100.0)
    colmap = dict(org=(C["green"], C["green_l"]), tech=(C["blue"], C["light"]),
                  env=(C["orange"], C["orange_l"]))
    for xc, txt, col in ((PX + PW / 2, "存在问题", C["red"]),
                         (CX + CW / 2, "成因诊断", C["primary"]),
                         (FX + FW / 2, "对策设计", C["green"])):
        ax.text(xc, H - 2.0, txt, ha="center", va="top", fontsize=SUBT,
                color=col, fontweight="bold")
    ymap_p, ymap_c, ymap_f = {}, {}, {}
    for ytop, t in probs:
        box(ax, PX, ytop - ph, PW, ph, t, fs=BODY, fc="#FBE5E5", ec=C["red"], lw=LW * 1.05)
        ymap_p[ytop] = ytop - ph / 2
    for ytop, t, k in causes:
        ec, fc = colmap[k]
        box(ax, CX, ytop - ch, CW, ch, t, fs=BODY, fc=fc, ec=ec, lw=LW * 1.05)
        ymap_c[ytop] = ytop - ch / 2
    for ytop, t, k in fixes:
        ec, fc = colmap[k]
        box(ax, FX, ytop - fh, FW, fh, t, fs=BODY, fc=fc, ec=ec, lw=LW * 1.1, bold=True)
        ymap_f[ytop] = ytop - fh / 2
    links = [(86, 80, -0.12), (86, 52, -0.18), (66, 80, -0.10), (66, 52, -0.08),
             (46, 52, 0.10), (46, 24, 0.14), (26, 24, 0.18), (26, 80, -0.22)]
    for cy, py, rad in links:
        arrow(ax, (CX, ymap_c[cy]), (PX + PW + 0.3, ymap_p[py]), color="#A6A6A6",
              lw=LW * 0.75, rad=rad, ms=11)
    for cy, fy in [(86, 86), (66, 66), (46, 46), (26, 26)]:
        arrow(ax, (CX + CW, ymap_c[cy]), (FX - 0.3, ymap_f[fy]), color=C["primary"],
              lw=LW * 1.1, ms=13)
    ly = 5.5
    ax.plot([4.0, 12.0], [ly + 3.4, ly + 3.4], color="#A6A6A6", lw=LW * 1.6)
    ax.text(13.4, ly + 3.4, "灰线：问题 ← 成因（归因关系）", va="center", fontsize=BODY,
            color=C["gray"])
    ax.plot([4.0, 12.0], [ly, ly], color=C["primary"], lw=LW * 1.8)
    ax.text(13.4, ly, "深蓝线：成因 → 对策（对应关系）", va="center", fontsize=BODY,
            color=C["gray"])
    save(fig, name)


# ================================================================ 图10 再造前后指标
# ================================================================ 图10 再造前后指标
def fig10():
    name = "图10_再造前后目标指标对比.png"
    cats = ["流程周期", "清标环节\n耗时", "文件返工\n轮次", "人工数据\n搬运次数"]
    base = [100, 100, 100, 100]
    target = [85, 68, 60, 55]
    drops = ["-15%", "-32%", "-40%", "-45%"]
    x = np.arange(len(cats))
    w = 0.34
    fig, ax = plt.subplots(figsize=(W_IN, 4.5))
    fig.subplots_adjust(left=0.135, right=0.975, top=0.855, bottom=0.20)
    ax.bar(x - w / 2, base, w, color="#BFBFBF", edgecolor="white",
           label="AS-IS（现状 = 100）", zorder=2)
    ax.bar(x + w / 2, target, w, color=C["blue"], edgecolor="white",
           label="TO-BE（基准情景测算值）", zorder=2)
    for xx in x - w / 2:
        ax.text(xx, 102, "100", ha="center", va="bottom", fontsize=BODY, color=C["gray"])
    for xx, v, d in zip(x + w / 2, target, drops):
        ax.text(xx, v + 2.5, f"{v}\n{d}", ha="center", va="bottom", fontsize=BODY,
                color=C["primary"], fontweight="bold", linespacing=1.35)
    ax.set_xticks(x)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=BODY)
    ax.set_ylabel("指数（AS-IS = 100）", fontsize=BODY)
    ax.set_ylim(0, 146)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=TICK)
    ax.legend(fontsize=BODY, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), ncol=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", color="#D9D9D9", zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.5, 0.16, "测算值，非实测值", transform=ax.transAxes, fontsize=15,
            color="red", alpha=0.13, ha="center", va="center", rotation=12,
            fontweight="bold", zorder=1)
    save(fig, name)


# ================================================================ 图11 指标体系
def fig11():
    name = "图11_再造效果评价指标体系.png"
    crit = [
        ("流程效率", ["流程周期", "计划偏差率", "里程碑延误", "返工强度", "任务响应"],
         C["light"], C["primary"]),
        ("集成水平", ["数据一次录入率", "自动流转占比", "接口成功率"], "#FFFFFF", C["primary"]),
        ("资源节约", ["单项目人工工时", "纸质文档占比", "人工搬运次数"], C["green_l"], C["green"]),
        ("业务价值", ["节点准时交付率", "客户满意度", "单项目毛利率"], C["green_l"], C["green"]),
        ("组织能力", ["培训覆盖率", "工具采纳率", "留痕完整率"], C["orange_l"], C["orange"]),
    ]
    HD = 6.6                       # 准则层表头高度
    tgt_txt = "工作流程数字化再造效果（目标层）"
    tgt_h = lh(TITLE) + lh(BODY) + 4.6
    r1 = crit[:3]
    r2 = crit[3:]
    cw1, gap1, x0 = 30.5, 2.75, 1.5
    body1 = max(len(c[1]) for c in r1) * lh(BODY) + 2.6
    hh1 = HD + body1
    cw2, gap2 = 47.5, 2.0
    body2 = max(len(c[1]) for c in r2) * lh(BODY) + 2.6
    hh2 = HD + body2
    g = 4.2
    H_units = 3.0 + tgt_h + 3.2 + g + hh1 + g + g + hh2 + 3.0
    fig, ax, H = canvas(H_units * W_IN / 100.0)
    y = H - 3.0
    box(ax, 18.0, y - tgt_h, 64.0, tgt_h, tgt_txt, fs=TITLE, fc=C["primary"],
        ec=C["primary"], tc="white", bold=True)
    ax.text(50, y - tgt_h + tgt_h * 0.235, "准则层 5 个 ｜ 指标层 17 项", ha="center",
            va="center", fontsize=BODY, color="white")
    y -= tgt_h
    bus1 = y - 2.6
    ax.plot([1.0, 99.0], [bus1, bus1], color=C["primary"], lw=LW)
    ax.plot([50, 50], [y, bus1], color=C["primary"], lw=LW)
    y = bus1 - 0.2
    x = x0
    xs1 = []
    for name_c, inds, fc, ec in r1:
        xs1.append(x + cw1 / 2)
        arrow(ax, (x + cw1 / 2, bus1), (x + cw1 / 2, y - 0.2), color=C["primary"],
              lw=LW * 0.9, ms=12)
        ax.add_patch(Rectangle((x, y - 0.2 - hh1), cw1, hh1, facecolor="#FFFFFF",
                               edgecolor=ec, lw=LW * 1.05))
        ax.add_patch(Rectangle((x, y - 0.2 - HD), cw1, HD, facecolor=fc, edgecolor=ec,
                               lw=LW * 1.05))
        ax.text(x + cw1 / 2, y - 0.2 - HD / 2, name_c, ha="center", va="center",
                fontsize=SUBT, color=ec, fontweight="bold")
        ax.text(x + 1.8, y - 0.2 - HD - 1.2,
                "\n".join("· " + i for i in inds), ha="left", va="top", fontsize=BODY,
                color="#1A1A1A", linespacing=1.5)
        x += cw1 + gap1
    row1_bot = y - 0.2 - hh1
    bus2 = row1_bot - 4.4
    ax.plot([1.0, 99.0], [bus2, bus2], color=C["primary"], lw=LW)
    ax.plot([1.0, 1.0], [bus1, bus2], color=C["primary"], lw=LW)
    y2 = bus2 - 0.2
    x = 1.5
    for name_c, inds, fc, ec in r2:
        arrow(ax, (x + cw2 / 2, bus2), (x + cw2 / 2, y2 - 0.2), color=C["primary"],
              lw=LW * 0.9, ms=12)
        ax.add_patch(Rectangle((x, y2 - 0.2 - hh2), cw2, hh2, facecolor="#FFFFFF",
                               edgecolor=ec, lw=LW * 1.05))
        ax.add_patch(Rectangle((x, y2 - 0.2 - HD), cw2, HD, facecolor=fc, edgecolor=ec,
                               lw=LW * 1.05))
        ax.text(x + cw2 / 2, y2 - 0.2 - HD / 2, name_c, ha="center", va="center",
                fontsize=SUBT, color=ec, fontweight="bold")
        ax.text(x + 1.8, y2 - 0.2 - HD - 1.2, "\n".join("· " + i for i in inds),
                ha="left", va="top", fontsize=BODY, color="#1A1A1A", linespacing=1.5)
        x += cw2 + gap2
    save(fig, name)


# ================================================================ 图12 系统架构
def fig12():
    name = "图12_系统总体技术架构.png"
    acc = ["Nginx（HTTPS 终结 · 12G 请求体上限 · 反向代理）",
           "gunicorn + gevent（4 workers · 120s 超时）"]
    app_ = ["Flask 3.1\nApp Factory（create_app）", "16 Blueprint\n28 个路由模块",
            "97 个 Service\n业务逻辑层"]
    mods = ["项目与时间轴\n（里程碑/步骤/待办）", "文档解析\n（PDF/Word/Excel/OCR）",
            "清标分析\n（5 维度复合评分）", "串通投标检测\n（相似度/尾数/Benford）",
            "合规检查\n（15 部法规库）", "智能对话\n（LangGraph 代理）",
            "知识库 / Wiki\n（RAG + 法规版本）", "企业信用查询\n（Selenium 自动化）"]
    ai = ["LangGraph\n状态图代理", "LLM 双供应商\nOpenRouter / NVIDIA NIM",
          "ChromaDB 向量库\nsentence-transformers", "提示安全层\n注入防护 · 反幻觉"]
    db = ["PostgreSQL 16\n76 张表（幂等建表）", "Redis 7 + Celery 5\nBroker / 缓存 / Beat",
          "文件与向量存储\n项目文件版本 · 法规库", "横切：权限 · 审计日志\n20+ 定时任务"]
    x0 = 1.5
    W4, G4 = 23.2, 1.4
    W3, G3 = 31.0, 2.0
    W2, G2 = 47.0, 2.0
    lab_h = 4.8
    h_acc = max(bh(t, W2) for t in acc)
    h_app = max(bh(t, W3) for t in app_)
    h_mod = max(bh(t, W4) for t in mods)
    h_ai = max(bh(t, W4) for t in ai)
    h_db = max(bh(t, W4) for t in db)
    specs = [("接入层", C["gray"], acc, W2, G2, 2, h_acc, 1),
             ("应用层", C["primary"], app_, W3, G3, 3, h_app, 1),
             ("业务能力层", C["blue"], mods, W4, G4, 4, h_mod * 2 + 1.2, 2),
             ("AI 能力层", C["purple"], ai, W4, G4, 4, h_ai, 1),
             ("数据与任务层", C["green"], db, W4, G4, 4, h_db, 1)]
    GAPB = 2.6
    H_units = 3.0 + sum(lab_h + h + GAPB for _, _, _, _, _, _, h, _ in specs) + 7.0
    fig, ax, H = canvas(H_units * W_IN / 100.0)
    y = H - 3.0
    tops = []
    for lname, col, content, bw, gap, ncol, hh, nrows in specs:
        ax.add_patch(Rectangle((1.0, y - lab_h - hh - 1.0), 98.0, lab_h + hh + 1.0,
                               facecolor="#FAFAFA", edgecolor="#D9D9D9", lw=1.0, zorder=0))
        ax.text(1.8, y - 0.8, lname, ha="left", va="top", fontsize=SUBT, color=col,
                fontweight="bold")
        yy = y - lab_h
        rowh = hh / nrows
        for i, m in enumerate(content):
            r, c = divmod(i, ncol)
            x = x0 + c * (bw + gap)
            box(ax, x, yy - rowh - r * (rowh + 1.2), bw, rowh, m, fs=BODY,
                fc="#FFFFFF", ec=col, lw=LW * 0.95)
        tops.append((y, y - lab_h - hh - 1.0, col))
        y -= lab_h + hh + GAPB
    # 层间箭头（上一层层底 → 下一层层顶）
    for k in range(len(tops) - 1):
        y_from = tops[k][1] - 0.2
        y_to = tops[k + 1][0] + 0.4
        for x in (24.0, 50.0, 76.0):
            arrow(ax, (x, y_from), (x, y_to), color="#8497B0", lw=LW * 0.8, ms=11)
    ax.text(99.0, 1.0, wrap("Docker 编排：app / postgres / redis / worker / beat / nginx", 96.0),
            ha="right", va="bottom", fontsize=BODY, color=C["gray"])
    save(fig, name)


# ================================================================ 图13 数据模型
# ================================================================ 图13 数据模型
def fig13():
    name = "图13_流程数据模型与可算指标.png"
    row1 = [("projects\n项目主表\nid · name · status\ncreated_at / updated_at", C["primary"], True),
            ("project_timelines\n流程主表\nplanned_start_date /\nplanned_end_date\n"
             "actual_start_date /\nactual_end_date", C["primary"], False),
            ("project_timeline_milestones\n里程碑表\nplanned_date / actual_date\n"
             "diff_days / reason_category", C["orange"], False)]
    row2 = [("project_workflow_steps\n环节步骤表\nstep_name / assigned_to\ncompleted / completed_at",
             C["orange"], False),
            ("project_todos\n任务待办表\ncreated_at / done_at / status", C["primary"], False),
            ("project_files\n项目文件与版本\nversion / uploaded_at", C["primary"], False)]
    row3 = [("workflow_kpi\ngenerations / revision_rounds\noutput_chars / completed_count",
             C["green"], False),
            ("audit_runs\nstarted_at / completed_at\noverall_score / overall_status",
             C["green"], False),
            ("admin_audit_log / timeline_diff_log\n操作审计与时间轴变更留痕", C["green"], False)]
    cw, gap, x0 = 31.0, 2.0, 1.5
    def rh(row):
        return max(bh(t, cw) for t, _, _ in row)
    r1h, r2h, r3h = rh(row1), rh(row2), rh(row3)
    metrics = [("流程周期", "actual_end_date - actual_start_date（天）"),
               ("计划偏差率", "(实际周期 - 计划周期) / 计划周期（%）"),
               ("里程碑延误", "diff_days 与 reason_category 分布"),
               ("返工强度", "workflow_kpi.revision_rounds（轮）"),
               ("任务响应", "todos.done_at - created_at（小时）"),
               ("留痕完整率", "有审计记录的关键操作 / 关键操作总数（%）")]
    mw = 47.0
    FW_M = mw - 2.5
    item_h = lh(BODY) + max(nlines(f, FW_M, TICK) for _, f in metrics) * lh(TICK) + 2.6
    panel_h = lh(SUBT) + 3 * item_h + 2.0
    H_units = 3.0 + r1h + 2.1 + r2h + 2.1 + r3h + 3.4 + panel_h + 3.0
    fig, ax, H = canvas(H_units * W_IN / 100.0)
    y = H - 3.0
    for k, (row, hh) in enumerate([(row1, r1h), (row2, r2h), (row3, r3h)]):
        for j, (t, ec, head) in enumerate(row):
            fc = C["primary"] if head else "#FFFFFF"
            tc = "white" if head else "#1A1A1A"
            box(ax, x0 + j * (cw + gap), y - hh, cw, hh, t, fs=BODY, fc=fc, ec=ec,
                tc=tc, lw=LW * 0.95)
        if k == 0:
            arrow(ax, (x0 + cw + 0.1, y - r1h / 2), (x0 + cw + gap - 0.1, y - r1h / 2),
                  lw=LW, ms=12)
            arrow(ax, (x0 + 2 * cw + gap + 0.1, y - r1h / 2),
                  (x0 + 2 * cw + 2 * gap - 0.1, y - r1h / 2), lw=LW, ms=12)
        y -= hh + 2.1
    ax.add_patch(Rectangle((1.0, y - panel_h + 2.0), 98.0, panel_h - 2.0,
                           facecolor="#FFF8E7", edgecolor=C["gold"], lw=LW, ls="--", zorder=0))
    ax.text(3.0, y - 1.4, "可计算的过程指标（第6章表6-1 指标体系的子集）", ha="left",
            va="top", fontsize=SUBT, color="#7F6000", fontweight="bold")
    my = y - 5.6
    for i, (nm, fm) in enumerate(metrics):
        r, c = divmod(i, 2)
        mx = 4.0 + c * mw
        yy = my - r * item_h
        ax.text(mx, yy - lh(BODY) / 2, "· " + nm, ha="left", va="center", fontsize=BODY,
                color="#1A1A1A", fontweight="bold")
        ax.text(mx, yy - lh(BODY) - 0.4, wrap(fm, FW_M, TICK), ha="left",
                va="top", fontsize=TICK, color=C["gray"])
    save(fig, name)


# ================================================================ 图16 情景测算
def fig16():
    name = "图16_情景测算结果.png"
    metrics = ["流程周期", "清标环节\n耗时", "文件返工\n轮次", "人工数据\n搬运次数"]
    cons = [-8, -18, -20, -25]
    bench = [-15, -32, -40, -45]
    opt = [-22, -45, -55, -60]
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 4.9), gridspec_kw=dict(width_ratios=[1.18, 1]))
    fig.subplots_adjust(left=0.135, right=0.975, top=0.875, bottom=0.205, wspace=0.34)
    ax = axes[0]
    for i, (lo, mid, hi) in enumerate(zip(cons, bench, opt)):
        ax.plot([i, i], [lo, hi], color=C["blue"], lw=11, alpha=0.30,
                solid_capstyle="round", zorder=2)
        ax.plot([i], [mid], marker="o", ms=10, color=C["primary"], zorder=3)
        ax.text(i, hi - 1.4, f"{hi}%", ha="center", va="top", fontsize=BODY,
                color=C["primary"], fontweight="bold")
        ax.text(i, lo + 1.4, f"{lo}%", ha="center", va="bottom", fontsize=BODY,
                color=C["primary"], fontweight="bold")
        ax.text(i + 0.18, mid, f"{mid}%", ha="left", va="center", fontsize=BODY,
                color=C["primary"], fontweight="bold")
    ax.axhline(0, color="#404040", lw=1.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=TICK)
    ax.set_ylabel("相对基线的变化率（%）", fontsize=BODY)
    ax.set_xlim(-0.55, 3.62)
    ax.set_ylim(-70, 8)
    ax.set_title("(a) 三情景下的指标变化\n（保守 / 基准 / 乐观）", fontsize=SUBT,
                 color=C["primary"], fontweight="bold", pad=6)
    ax.tick_params(axis="y", labelsize=TICK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", color="#D9D9D9")
    ax.set_axisbelow(True)
    handles = [plt.Line2D([0], [0], color=C["blue"], lw=9, alpha=0.30,
                          solid_capstyle="round", label="保守—乐观区间"),
               plt.Line2D([0], [0], marker="o", ls="", ms=9, color=C["primary"],
                          label="基准情景")]
    ax.legend(handles=handles, fontsize=BODY, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.92), handlelength=1.6)
    ax = axes[1]
    labels = ["AS-IS", "保守", "基准", "乐观"]
    vals = [100.0, 82.3, 67.0, 54.5]
    txts = ["100", "82.3", "67.0", "54.5"]
    cols = ["#BFBFBF", "#9DC3E6", C["blue"], C["primary"]]
    bars = ax.bar(labels, vals, color=cols, width=0.58, edgecolor="white", zorder=2)
    for b, t in zip(bars, txts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.6, t, ha="center",
                va="bottom", fontsize=BODY, color=C["primary"], fontweight="bold")
    ax.set_ylim(44, 110)
    ax.set_ylabel("流程综合指数（AS-IS = 100）", fontsize=BODY)
    ax.set_title("(b) 综合指数\n（等权标准化合成）", fontsize=SUBT, color=C["primary"],
                 fontweight="bold", pad=6)
    ax.tick_params(labelsize=TICK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", color="#D9D9D9")
    ax.set_axisbelow(True)
    ax.text(0.5, 0.08, "测算值，非实测值", transform=ax.transAxes, ha="center",
            va="center", fontsize=13, color="red", alpha=0.13, rotation=8,
            fontweight="bold", zorder=1)
    save(fig, name)


# ================================================================ 图17 返工基线
def fig17():
    name = "图17_返工轮次基线.png"
    cats = ["一般行业", "政府采购", "部队及武警"]
    vals = [1.2, 2.2, 3.0]
    colors = ["#8FB3D9", "#4F81BD", "#1F4E79"]
    fig, ax = plt.subplots(figsize=(W_IN, 4.3))
    fig.subplots_adjust(left=0.155, right=0.975, top=0.855, bottom=0.135)
    bars = ax.bar(cats, vals, color=colors, width=0.52, edgecolor="white", zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.07, f"{v:.1f}", ha="center",
                va="bottom", fontsize=BODY, fontweight="bold", color="#333333")
    ax.set_ylabel("平均返工轮次（次）", fontsize=BODY)
    ax.set_ylim(0, 3.7)
    ax.set_title("分业务类型平均返工轮次（实测基线）", fontsize=TITLE,
                 fontweight="bold", color=C["primary"], pad=12)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=BODY)
    ax.tick_params(axis="y", labelsize=TICK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    save(fig, name)


# ================================================================ 图18 实测对比
def fig18():
    name = "图18_再造前后实测对比.png"
    fig, axes = plt.subplots(1, 3, figsize=(W_IN, 4.6))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.80, bottom=0.155, wspace=0.42)
    before, after = "#9DB7D5", "#1F4E79"

    ax = axes[0]
    ax.bar(["上线前", "上线后"], [450, 570], color=[before, after], width=0.5, zorder=2)
    for i, v in enumerate([450, 570]):
        ax.text(i, v + 14, f"约{v}", ha="center", fontsize=BODY, fontweight="bold",
                color="#333333")
    ax.set_title("(a) 年项目数\n（个）", fontsize=SUBT, fontweight="bold")
    ax.set_ylim(0, 700)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["上线前", "上线后"], fontsize=TICK)

    ax = axes[1]
    ax.bar(["上线前", "上线后"], [27, 20], color=[before, after], width=0.5, zorder=2)
    for i, v in enumerate([27, 20]):
        ax.text(i, v + 0.8, f"约{v}天", ha="center", fontsize=BODY, fontweight="bold",
                color="#333333")
    ax.set_title("(b) 平均流程\n周期（天）", fontsize=SUBT, fontweight="bold")
    ax.set_ylim(0, 34)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["上线前", "上线后"], fontsize=TICK)

    ax = axes[2]
    vals = [1440, 20]
    ax.bar(["上线前", "上线后"], vals, color=[before, after], width=0.5, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(10, 5200)
    ax.text(0, 1440 * 1.35, "约3个工作日\n(≈1440分钟)", ha="center", va="bottom",
            fontsize=TICK, color="#333333", linespacing=1.3)
    ax.text(1, 20 * 1.5, "约20分钟", ha="center", va="bottom", fontsize=TICK,
            fontweight="bold", color="#333333")
    ax.set_title("(c) 清标环节\n平均耗时\n（分钟，对数）", fontsize=SUBT, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["上线前", "上线后"], fontsize=TICK)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=TICK)
    fig.suptitle("再造前后关键指标对比（实测）", fontsize=TITLE, fontweight="bold",
                 color=C["primary"], y=0.97)
    save(fig, name)


FIGURES = [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08,
           fig09, fig10, fig11, fig12, fig13, fig16, fig17, fig18]

if __name__ == "__main__":
    for fn in FIGURES:
        fn()
    print("\n===== 汇总 =====")
    bad = 0
    for r in RESULTS:
        ok = r["bytes"] >= 100 * 1024 and r["ink"] >= 0.01
        bad += 0 if ok else 1
        print(f"{'OK ' if ok else 'FAIL'} {r['name']:42s} {r['bytes']/1024:7.1f}KB "
              f"{r['px'][0]}x{r['px'][1]} ink={r['ink']*100:4.1f}%")
    print(f"共 {len(RESULTS)} 张，失败 {bad} 张")
