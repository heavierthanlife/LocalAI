# -*- coding: utf-8 -*-
"""按画风预设批量重绘论文图（数据与脚本内容不变）。
用法: python regen_style.py thesis_a 05 12 16
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style_presets import PRESETS

preset = sys.argv[1]
want = sys.argv[2:]
P = PRESETS[preset]
OUT = f"figs_style_{preset}"
os.makedirs(OUT, exist_ok=True)

src = open("v12/make_figures_v14.py", encoding="utf-8").read()
# 字体：用 Windows 自带 SimHei（与现有 16 张图同字库）
src = src.replace('("/w/fonts/simhei.ttf",', '("/mnt/c/Windows/Fonts/simhei.ttf",')
src = src.replace('OUT = "figures_v14"', f'OUT = "{OUT}"')
# 版式常量
src = re.sub(r"BODY, TICK, SUBT, TITLE, BIG = [\d.]+, [\d.]+, [\d.]+, [\d.]+, [\d.]+",
             f"BODY, TICK, SUBT, TITLE, BIG = {P['BODY']}, {P['TICK']}, {P['SUBT']}, {P['TITLE']}, {P['BIG']}", src)
src = re.sub(r"LW = [\d.]+", f"LW = {P['LW']}", src, count=1)
# 配色
c = P["C"]
src = re.sub(r'C = dict\(.*?\)\n', "C = " + repr(c) + "\n", src, count=1, flags=re.S)
# 线宽
src = src.replace('"axes.linewidth": 1.2', f'"axes.linewidth": {P["axes_lw"]}')
src = src.replace('"lines.linewidth": 2.2', f'"lines.linewidth": {P["line_lw"]}')
src = src.replace('linewidth=lw, linestyle=ls, zorder=zorder)',
                  'linewidth=lw * ' + str(P["box_lw"] / 1.2) + ', linestyle=ls, zorder=zorder)')
src = src.replace('rounding_size=1.4', f'rounding_size={P["round"]}')
src = src.replace('rounding_size=1.2', f'rounding_size={P["round"]}')


# 去掉与 Word 图题重复的图内主标题（图17/图18），并压缩顶部留白
src = re.sub(r'^[ \t]*ax\.set_title\("分业务类型平均返工轮次.*?\)\n', '', src, flags=re.S | re.M)
src = re.sub(r'^[ \t]*fig\.suptitle\("再造前后关键指标对比（实测）".*?\)\n', '', src, flags=re.S | re.M)
src = src.replace('fig.subplots_adjust(left=0.155, right=0.975, top=0.855, bottom=0.135)',
                  'fig.subplots_adjust(left=0.155, right=0.975, top=0.95, bottom=0.135)')
src = src.replace('fig.subplots_adjust(left=0.13, right=0.985, top=0.80, bottom=0.155, wspace=0.42)',
                  'fig.subplots_adjust(left=0.13, right=0.985, top=0.94, bottom=0.155, wspace=0.42)')
src = src.split("FIGURES = [fig01")[0]
g = {"__name__": "figs", "__file__": os.path.abspath("v12/make_figures_v14.py")}
exec(compile(src, "make_figures_v14_styled", "exec"), g)
name_map = {f"fig{n:02d}": f"fig{n:02d}" for n in range(1, 19)}
for code in want:
    fn = g[f"fig{int(code):02d}"]
    fn()
print("生成:", sorted(os.listdir(OUT)))
