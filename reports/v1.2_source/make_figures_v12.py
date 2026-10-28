# -*- coding: utf-8 -*-
"""v1.2 补充图：图6-4 分业务类型平均返工轮次（实测基线）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont("/w/fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

C_PRIMARY = "#1F4E79"

cats = ["一般行业", "政府采购", "部队及武警"]
vals = [1.2, 2.2, 3.0]
colors = ["#8FB3D9", "#4F81BD", C_PRIMARY]

fig, ax = plt.subplots(figsize=(7.2, 4.4))
bars = ax.bar(cats, vals, color=colors, width=0.52, edgecolor="white")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.06, f"{v:.1f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="#333333")
ax.set_ylabel("平均返工轮次（次）", fontsize=11)
ax.set_ylim(0, 3.6)
ax.set_title("分业务类型平均返工轮次（实测基线）", fontsize=13, fontweight="bold", color=C_PRIMARY, pad=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=10)
fig.text(0.5, -0.02, "注：数据为近一年聚合结果，经四舍五入处理，非逐项目明细。",
         ha="center", fontsize=8.5, color="#666666")
fig.savefig("figures_final/图17_返工轮次基线.png")
print("saved figures_final/图17_返工轮次基线.png")
