# -*- coding: utf-8 -*-
"""图件 v1.4 自检：尺寸/非空白 + 文字越界/重叠/压框 + 字号 + 无“注：”+ 文件名对齐。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_figures_v14 as m
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

EXPECTED = [
    "图01_研究技术路线图.png", "图02_理论整合分析框架.png", "图03_TOE维度映射矩阵.png",
    "图04_AS-IS六阶段评估流程.png", "图05_核心问题量化表征.png", "图06_问题成因鱼骨图.png",
    "图07_调查样本结构.png", "图08_数字化工具使用率与集成度.png", "图09_问题-成因-对策传导链.png",
    "图10_再造前后目标指标对比.png", "图11_再造效果评价指标体系.png", "图12_系统总体技术架构.png",
    "图13_流程数据模型与可算指标.png", "图16_情景测算结果.png", "图17_返工轮次基线.png",
    "图18_再造前后实测对比.png",
]

captured = []
_orig_close = plt.close


def _close(fig=None):
    captured.append(fig)
    _orig_close(fig)


m.plt.close = _close
for fn in m.FIGURES:
    fn()


def texts_of(fig):
    out, seen = [], set()

    def add(t):
        if t is None or id(t) in seen:
            return
        seen.add(id(t))
        if not t.get_text().strip() or not t.get_visible():
            return
        out.append(t)

    for ax in fig.get_axes():
        for t in ax.texts:
            add(t)
        if ax.axison:   # 只有可见坐标轴才取刻度（axis("off") 的图不取，且过滤视窗外的 tick）
            for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                for tick in axis.get_major_ticks():
                    try:
                        loc = tick.get_loc()
                    except Exception:
                        continue
                    if lim[0] - 1e-9 <= loc <= lim[1] + 1e-9:
                        add(tick.label1)
        add(ax.title)
        add(ax.xaxis.label)
        add(ax.yaxis.label)
        if ax.get_legend() is not None:
            for t in ax.get_legend().get_texts():
                add(t)
    for t in fig.texts:
        add(t)
    add(getattr(fig, "_suptitle", None))
    return out


def check(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    items = []
    for t in texts_of(fig):
        if t is None:
            continue
        s = t.get_text()
        if not s.strip():
            continue
        items.append((s, t.get_window_extent(renderer=r), float(t.get_fontsize())))
    # 1) 越界（画布）
    oob = [(s, bb) for s, bb, _ in items
           if bb.x0 < -2 or bb.y0 < -2 or bb.x1 > W + 2 or bb.y1 > H + 2]
    # 2) 文字重叠（面积占比 > 40%）
    overlaps = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s1, b1, _ = items[i]
            s2, b2, _ = items[j]
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > 2 and iy > 2:
                a = ix * iy
                amin = min(b1.width * b1.height, b2.width * b2.height)
                if amin > 0 and a / amin > 0.40:
                    overlaps.append((s1[:16], s2[:16], round(a / amin, 2)))
    # 3) 文字溢出所属框（文字中心落在某 patch 内，但 bbox 越出该 patch）
    spill = []
    for ax in fig.get_axes():
        if ax.axison:      # 图表类：只认显式标记的容器框（柱形等不算容器）
            cand_patches = [p for p in ax.patches if getattr(p, "_container", False)]
        else:              # 示意/矩形拼图类：矩形框即容器（箭头不算）
            cand_patches = [p for p in ax.patches
                            if isinstance(p, (FancyBboxPatch, Rectangle))
                            and not isinstance(p, FancyArrowPatch)]
        pats = [(p, p.get_window_extent(renderer=r)) for p in cand_patches]
        for s_, bb, _ in items:
            cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
            cand = [pb for p, pb in pats
                    if pb.x0 <= cx <= pb.x1 and pb.y0 <= cy <= pb.y1
                    and pb.width > 8 and pb.height > 6]
            if not cand:
                continue
            pb = min(cand, key=lambda b: b.width * b.height)
            if (bb.x0 < pb.x0 - 2.5 or bb.x1 > pb.x1 + 2.5
                    or bb.y0 < pb.y0 - 2.5 or bb.y1 > pb.y1 + 2.5):
                spill.append((s_[:18], f"txt {bb.width:.0f}x{bb.height:.0f} "
                                        f"box {pb.width:.0f}x{pb.height:.0f}"))
    # 4) 图中不得出现“注：”
    notes = [s_ for s_, _, _ in items if "注：" in s_ or "注:" in s_]
    # 5) 字号统计
    sizes = sorted({round(fs, 1) for _, _, fs in items})
    minfs = min(fs for _, _, fs in items) if items else 0
    return dict(name=name, W=W, H=H, n=len(items), oob=oob, overlaps=overlaps,
                spill=spill, notes=notes, sizes=sizes, minfs=minfs)


bad = 0
print(f"{'图名':44s} {'文字':>4s} {'越界':>4s} {'重叠':>4s} {'溢出':>4s} {'注':>3s} 字号")
for fig, name in zip(captured, EXPECTED):
    res = check(fig, name)
    ok = not (res["overlaps"] or res["spill"] or res["notes"]) and res["minfs"] >= 10.9
    bad += 0 if ok else 1
    print(f"{name:44s} {res['n']:4d} {len(res['oob']):4d} {len(res['overlaps']):4d} "
          f"{len(res['spill']):4d} {len(res['notes']):3d} {res['sizes']}"
          + ("   [提示]越界" if res["oob"] else ""))
    if res["oob"]:
        print("   越界:", [(s[:18], (round(b.x0), round(b.y0), round(b.x1), round(b.y1)))
                          for s, b in res["oob"][:4]], "canvas", res["W"], res["H"])
    if res["overlaps"]:
        print("   重叠:", res["overlaps"][:6])
    if res["spill"]:
        print("   溢出:", res["spill"][:6])

# ---- 文件级检查 ----
OUT = m.OUT
files = sorted(f for f in os.listdir(OUT) if f.endswith(".png"))
missing = [f for f in EXPECTED if f not in files]
extra = [f for f in files if f not in EXPECTED]
print("\n文件数:", len(files), "缺失:", missing or "无", "多余:", extra or "无")
small = []
for f in EXPECTED:
    p = os.path.join(OUT, f)
    if os.path.exists(p) and os.path.getsize(p) < 100 * 1024:
        small.append((f, os.path.getsize(p)))
print("小于 100KB:", small or "无")
print("\n结论:", "全部通过" if (bad == 0 and not missing and not small and not extra) else f"{bad} 张存在问题")
sys.exit(0 if (bad == 0 and not missing and not small and not extra) else 1)
