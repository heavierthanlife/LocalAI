# -*- coding: utf-8 -*-
"""基于真实项目代码（heavierthanlife/LocalAI, master 分支）绘制系统架构图与流程数据模型图。
事实来源：README.md / docs/ARCHITECTURE.md / app/database.py / requirements.txt / docker-compose.yml
"""
import os
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
os.makedirs(OUT, exist_ok=True)

C = dict(primary="#1F3864", blue="#2E75B6", light="#DEEBF7", mid="#9DC3E6",
         orange="#C55A11", orange_l="#FBE5D6", green="#548235", green_l="#E2EFDA",
         gray="#595959", lgray="#EDEDED", red="#C00000", purple="#5B2C6F", purple_l="#E8DAEF")


def canvas(w, h, xlim=(0, 100), ylim=(0, 100)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#1F3864", fs=9, tc="#1A1A1A",
        lw=1.1, bold=False, ls="-", zorder=3, pad=0.3, r=1.0, va="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va=va, fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=zorder + 1, linespacing=1.5)


def arrow(ax, p1, p2, color="#1F3864", lw=1.3, ls="-", ms=11, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms, color=color,
                                 linewidth=lw, linestyle=ls, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))


# ============================ 图6-1 系统总体架构 ============================
def fig_arch():
    fig, ax = canvas(9.6, 7.0)
    # 层标签
    layers = [
        (88, 10, "接入层", C["gray"]),
        (72, 13, "应用层", C["primary"]),
        (46, 21, "业务能力层", C["blue"]),
        (26, 15, "AI 能力层", C["purple"]),
        (6, 15, "数据与任务层", C["green"]),
    ]
    for y, h, name, col in layers:
        ax.add_patch(Rectangle((1.2, y), 97.6, h, facecolor="#FAFAFA", edgecolor="#D9D9D9", lw=0.8, zorder=0))
        ax.text(1.2, y + h + 1.1, name, fontsize=9.6, color=col, fontweight="bold", ha="left")

    # 接入层
    box(ax, 6, 88, 40, 8, "Nginx（HTTPS 终结 · 12G 请求体上限 · 反向代理）", fc=C["lgray"], ec=C["gray"], fs=8.6)
    box(ax, 51, 88, 42, 8, "gunicorn + gevent（4 workers · 120s 超时）", fc=C["lgray"], ec=C["gray"], fs=8.6)
    # 应用层
    box(ax, 4, 73, 28, 10, "Flask 3.1\nApp Factory（create_app）", fc=C["light"], ec=C["primary"], fs=9, bold=True)
    box(ax, 36, 73, 28, 10, "16 Blueprint\n28 个路由模块", fc=C["light"], ec=C["primary"], fs=9)
    box(ax, 68, 73, 28, 10, "97 个 Service\n业务逻辑层", fc=C["light"], ec=C["primary"], fs=9)
    # 业务能力层 8 模块
    mods = ["项目与时间轴\n（里程碑/步骤/待办）", "文档解析\n（PDF/Word/Excel/OCR）", "清标分析\n（5 维度复合评分）",
            "串通投标检测\n（相似度/尾数/Benford）", "合规检查\n（15 部法规库）", "智能对话\n（LangGraph 代理）",
            "知识库 / Wiki\n（RAG + 法规版本）", "企业信用查询\n（Selenium 自动化）"]
    w, gap = 22.6, 1.6
    for i, m in enumerate(mods):
        row, col = divmod(i, 4)
        x = 2.4 + col * (w + gap)
        y = 46 + (1 - row) * 10.6
        fc = C["orange_l"] if i < 4 else "#FFFFFF"
        ec = C["orange"] if i < 4 else C["blue"]
        box(ax, x, y, w, 9.4, m, fc=fc, ec=ec, fs=7.8)
    # AI 层
    ai = ["LangGraph\n状态图代理", "LLM 双供应商\nOpenRouter / NVIDIA NIM",
          "ChromaDB 向量库\nsentence-transformers", "提示安全层\n注入防护 · 反幻觉"]
    for i, m in enumerate(ai):
        box(ax, 2.4 + i * (w + gap), 27, w, 9.4, m, fc=C["purple_l"], ec=C["purple"], fs=7.8)
    # 数据层
    db = ["PostgreSQL 16\n76 张表（幂等建表）", "Redis 7 + Celery 5\nBroker / 缓存 / Beat",
          "文件与向量存储\n项目文件版本 · 法规库", "横切：权限 · 审计日志\n20+ 定时任务"]
    for i, m in enumerate(db):
        box(ax, 2.4 + i * (w + gap), 7, w, 9.4, m, fc=C["green_l"], ec=C["green"], fs=7.8)
    # 层间箭头
    for x in (26, 50, 74):
        arrow(ax, (x, 87.6), (x, 83.4), color="#8497B0", lw=1.2)
        arrow(ax, (x, 72.6), (x, 66.6), color="#8497B0", lw=1.2)
    for x in (13, 37, 61, 85):
        arrow(ax, (x, 45.6), (x, 36.8), color="#8497B0", lw=1.0, ls="--")
        arrow(ax, (x, 26.6), (x, 16.8), color="#8497B0", lw=1.0, ls="--")
    # 右侧部署注记
    ax.text(99.2, 1.2, "Docker 编排：app / postgres / redis / worker / beat / nginx",
            fontsize=7.4, color=C["gray"], ha="right", va="bottom")
    fig.savefig(f"{OUT}/图12_系统总体技术架构.png")
    plt.close(fig)


# ============================ 图6-2 流程数据模型与指标 ============================
def fig_data():
    fig, ax = canvas(9.8, 6.4)
    # 主链条
    box(ax, 2, 76, 26, 15, "projects\n项目主表\nid · name · status\ncreated_at / updated_at",
        fc=C["primary"], ec=C["primary"], fs=7.8, tc="white", bold=True)
    box(ax, 32, 76, 30, 15, "project_timelines\n流程主表\nplanned_start / planned_end\nactual_start / actual_end",
        fc=C["light"], ec=C["primary"], fs=7.6)
    box(ax, 66, 76, 32, 15, "project_timeline_milestones\n里程碑表\nplanned_date / actual_date\ndiff_days / reason_category",
        fc=C["orange_l"], ec=C["orange"], fs=7.6)
    arrow(ax, (28.2, 83.5), (31.6, 83.5))
    arrow(ax, (62.2, 83.5), (65.6, 83.5))
    # 下挂
    box(ax, 66, 55, 32, 13, "project_workflow_steps\n环节步骤表\nstep_name / assigned_to\ncompleted / completed_at",
        fc="#FFFFFF", ec=C["orange"], fs=7.6)
    arrow(ax, (82, 75.6), (82, 68.6), color=C["orange"])
    box(ax, 32, 55, 30, 13, "project_todos\n任务待办表\ncreated_at / done_at / status",
        fc="#FFFFFF", ec=C["primary"], fs=7.6)
    arrow(ax, (47, 75.6), (47, 68.6))
    box(ax, 2, 55, 26, 13, "project_files / versions\n项目文件与版本\nversion_no / updated_at",
        fc="#FFFFFF", ec=C["primary"], fs=7.6)
    arrow(ax, (15, 75.6), (15, 68.6))
    # KPI 与审计
    box(ax, 2, 35, 26, 13, "workflow_kpi\ngenerations / revision_rounds\noutput_chars / completed_count",
        fc=C["green_l"], ec=C["green"], fs=7.6)
    box(ax, 32, 35, 30, 13, "audit_runs\naudit_runs.started_at / completed_at\noverall_score / overall_status",
        fc=C["green_l"], ec=C["green"], fs=7.6)
    box(ax, 66, 35, 32, 13, "admin_audit_log / timeline_diff_log\n操作审计与时间轴变更留痕\n（可追溯性证据）",
        fc=C["green_l"], ec=C["green"], fs=7.6)
    for x in (15, 47, 82):
        arrow(ax, (x, 54.6), (x, 48.6), color=C["green"], lw=1.0, ls="--")
    # 派生指标
    ax.add_patch(Rectangle((1.2, 2), 97.6, 27, facecolor="#FFF8E7", edgecolor="#BF9000", lw=1.0, ls="--"))
    ax.text(2.8, 26.6, "可计算的过程指标（对应论文第6章效果评估）", fontsize=9.2, color="#7F6000", fontweight="bold")
    metrics = [
        ("流程周期", "actual_end - actual_start（天）"),
        ("计划偏差率", "(实际 - 计划) / 计划（%）"),
        ("里程碑延误", "diff_days 与 reason_category 分布"),
        ("返工强度", "workflow_kpi.revision_rounds（轮）"),
        ("任务响应", "todos.done_at - created_at（小时）"),
        ("审计耗时", "audit_runs 完成 - 开始（秒）"),
        ("质量水平", "audit_runs.overall_score（分）"),
        ("过程留痕", "admin_audit_log 记录数 / 项目"),
    ]
    for i, (name, formula) in enumerate(metrics):
        row, col = divmod(i, 2)
        x = 3.4 + col * 48.0
        y = 21.6 - row * 5.0
        ax.text(x, y, "·", fontsize=11, color="#BF9000", va="center")
        ax.text(x + 1.4, y, name, fontsize=8.2, color="#1A1A1A", va="center", fontweight="bold")
        ax.text(x + 12.0, y, formula, fontsize=7.3, color=C["gray"], va="center")
    fig.savefig(f"{OUT}/图13_流程数据模型与可算指标.png")
    plt.close(fig)


fig_arch()
fig_data()
print("done")
