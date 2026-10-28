# -*- coding: utf-8 -*-
"""v1.3 补充图：图6-5 再造前后关键指标对比（实测）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont("/w/fonts/simhei.ttf")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

BEFORE, AFTER = "#9DB7D5", "#1F4E79"
fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2))

# (a) 项目数
ax = axes[0]
ax.bar(["上线前", "上线后"], [450, 570], color=[BEFORE, AFTER], width=0.5)
for i, v in enumerate([450, 570]):
    ax.text(i, v + 12, f"约{v}", ha="center", fontsize=11, fontweight="bold", color="#333")
ax.set_title("(a) 年项目数（个）", fontsize=11, fontweight="bold")
ax.set_ylim(0, 680)

# (b) 平均流程周期
ax = axes[1]
ax.bar(["上线前", "上线后"], [27, 20], color=[BEFORE, AFTER], width=0.5)
for i, v in enumerate([27, 20]):
    ax.text(i, v + 0.7, f"约{v}天", ha="center", fontsize=11, fontweight="bold", color="#333")
ax.set_title("(b) 平均流程周期（天）", fontsize=11, fontweight="bold")
ax.set_ylim(0, 33)

# (c) 清标环节耗时（分钟，对数）
ax = axes[2]
vals = [1440, 20]
ax.bar(["上线前", "上线后"], vals, color=[BEFORE, AFTER], width=0.5)
ax.set_yscale("log")
ax.set_ylim(10, 4000)
ax.text(0, 1440*1.25, "约3个工作日\n(≈1440分钟)", ha="center", fontsize=9.5, color="#333")
ax.text(1, 20*1.25, "约20分钟", ha="center", fontsize=10, fontweight="bold", color="#333")
ax.set_title("(c) 清标环节平均耗时（分钟，对数）", fontsize=11, fontweight="bold")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

fig.suptitle("再造前后关键指标对比（实测）", fontsize=13, fontweight="bold", color="#1F4E79", y=1.02)
fig.text(0.5, -0.03, "注：数据经公司许可、聚合与四舍五入处理；清标前后计时口径不完全一致，仅作方向性比较。",
         ha="center", fontsize=8.5, color="#666")
fig.savefig("figures_final/图18_再造前后实测对比.png")
print("saved figures_final/图18_再造前后实测对比.png")
