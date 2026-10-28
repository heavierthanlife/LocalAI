# -*- coding: utf-8 -*-
"""图件几何与内容自检（因本机图片预览不可用，改为解析图元 bbox 与文字内容）。

检查项：
  1) 每张图所有文字 bbox 是否落在画布内（防越界/裁切）；
  2) 文字 bbox 两两是否重叠（排除刻意叠放的水印与容器内长文本）；
  3) 关键文案/数值是否存在于图中（防改漏）；
  4) 旧口径字样是否已绝迹（资产评估、熵权法测算、SWOT、配对检验·ITS、审计耗时、质量水平等）。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_figures_v11 as m

captured = []
_orig_close = plt.close


def _close(fig=None):
    captured.append(fig)
    _orig_close(fig)


m.plt.close = _close

for fn in (m.fig63, m.fig62, m.fig61, m.fig02, m.fig52, m.fig11_route):
    fn()

# 期望/禁止文案
EXPECT = {
    "图6-3": dict(must=["人工数据搬运次数", "清标环节耗时", "文件返工轮次", "流程周期",
                       "测算值", "非实测值", "85", "-15%", "-32%", "-40%", "-45%", "100"],
                  deny=["复核审批轮次", "纸质文档占比", "17.5", "82.5", "33.3", "66.7", "待实测替换"]),
    "图6-2": dict(must=["等权标准化合成", "82.3", "67.0", "54.5", "100", "保守", "基准", "乐观",
                       "人工数据搬运次数", "-8%", "-18%", "-20%", "-25%",
                       "-15%", "-32%", "-40%", "-45%", "-22%", "-45%", "-55%", "-60%"],
                  deny=["熵权法", "91.5", "84.2", "77.8"]),
    "图6-1": dict(must=["目标层", "流程效率", "集成水平", "资源节约", "业务价值", "组织能力",
                       "流程周期", "计划偏差率", "里程碑延误", "返工强度", "任务响应",
                       "数据一次录入率", "自动流转占比", "接口成功率",
                       "单项目人工工时", "纸质文档占比", "人工搬运次数",
                       "节点准时交付率", "客户满意度", "单项目毛利率",
                       "培训覆盖率", "工具采纳率", "留痕完整率",
                       "17", "等权合成", "熵权法与AHP组合赋权", "情景测算", "敏感性分析",
                       "配对检验与中断时间序列"],
                  deny=["审批轮次", "一次通过率", "差旅次数", "认证通过率", "功能采纳率",
                        "稳健性检验", "无纸化率"]),
    "图2-1": dict(must=["招标代理业务工作流程数字化再造分析框架", "诊断层", "解释层", "路径层",
                       "BPR", "TOE", "动态能力理论"],
                  deny=["资产评估"]),
    "图5-2": dict(must=["planned_start_date", "planned_end_date", "actual_start_date",
                       "actual_end_date", "version / uploaded_at", "project_files",
                       "completed_at", "diff_days", "reason_category",
                       "流程周期", "计划偏差率", "里程碑延误", "返工强度", "任务响应",
                       "留痕完整率", "表6-1"],
                  deny=["审计耗时", "质量水平", "version_no", "project_files / versions",
                        "planned_start /", "actual_start /"]),
    "图1-1": dict(must=["鱼骨图·访谈归纳", "情景测算·敏感性分析",
                       "指标体系 + 三情景测算 + 敏感性分析"],
                  deny=["SWOT", "配对检验·ITS", "前后对比检验"]),
}


def texts_of(fig):
    out = []
    for ax in fig.get_axes():
        for t in ax.texts:
            out.append(t)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            out.append(t)
        out += [ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.get_legend() is not None:
            for t in ax.get_legend().get_texts():
                out.append(t)
    return out


def check(tag, fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    items = []
    for t in texts_of(fig):
        s = t.get_text()
        if not s.strip():
            continue
        bb = t.get_window_extent(renderer=r)
        items.append((s, bb))
    # 1) 越界
    oob = [(s, bb) for s, bb in items
           if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1]
    # 2) 重叠（面积占比 > 35% 视为真重叠；忽略同一字符串的超大背景水印）
    overlaps = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s1, b1 = items[i]
            s2, b2 = items[j]
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > 1.5 and iy > 1.5:
                a = ix * iy
                amin = min(b1.width * b1.height, b2.width * b2.height)
                if amin > 0 and a / amin > 0.35:
                    overlaps.append((s1[:22], s2[:22], round(a / amin, 2)))
    # 2b) 文字压到框线（文字 bbox 与 patch 边框相交 = 压线）
    border_hits = []
    for ax in fig.get_axes():
        for p in ax.patches:
            pb = p.get_window_extent(renderer=r)
            if pb.width < 5 or pb.height < 5:
                continue
            for s_, bb in items:
                if "测算值" in s_:   # 水印按设计叠放在柱形/框上
                    continue
                ix = min(bb.x1, pb.x1) - max(bb.x0, pb.x0)
                iy = min(bb.y1, pb.y1) - max(bb.y0, pb.y0)
                if ix > 1.5 and iy > 1.5:
                    inside = (bb.x0 >= pb.x0 - 1 and bb.x1 <= pb.x1 + 1
                              and bb.y0 >= pb.y0 - 1 and bb.y1 <= pb.y1 + 1)
                    if not inside:
                        border_hits.append((s_[:18], round(ix), round(iy)))
    n_bullet = sum(s_.count("· ") for s_, _ in items)
    allt = " ".join(s for s, _ in items)
    exp = EXPECT[tag]
    missing = [k for k in exp["must"] if k not in allt]
    leftover = [k for k in exp["deny"] if k in allt]
    print(f"\n=== {tag}  文字 {len(items)} 条  canvas {W}x{H}")
    print(f"  bbox 越界: {len(oob)}" + (f" -> {[s[:20] for s,_ in oob][:5]}" if oob else ""))
    print(f"  bbox 重叠: {len(overlaps)}" + (f" -> {overlaps[:6]}" if overlaps else ""))
    print(f"  文字压框线: {len(border_hits)}" + (f" -> {border_hits[:6]}" if border_hits else ""))
    if tag == "图6-1":
        print(f"  指标条目计数: {n_bullet}（期望 17）"
              f"{'  <-- 不符' if n_bullet != 17 else ''}")
        if n_bullet != 17:
            missing.append("指标计数=17")
    print(f"  缺失文案: {missing if missing else '无'}")
    print(f"  残留旧文案: {leftover if leftover else '无'}")
    return len(oob) == 0 and len(overlaps) == 0 and not missing and not leftover


ok = True
for fn, tag in ((m.fig63, "图6-3"), (m.fig62, "图6-2"), (m.fig61, "图6-1"),
                (m.fig02, "图2-1"), (m.fig52, "图5-2"), (m.fig11_route, "图1-1")):
    ok = check(tag, captured[["图6-3", "图6-2", "图6-1", "图2-1", "图5-2", "图1-1"].index(tag)]) and ok

print("\n几何+内容自检:", "全部通过" if ok else "存在问题，见上")
sys.exit(0 if ok else 1)
