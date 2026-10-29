# -*- coding: utf-8 -*-
"""按学校 4 号模板原样填充内容：不动模板设计，只改文字。"""
import copy, os
from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.opc.constants import RELATIONSHIP_TYPE as RT

TPL = "/mnt/c/Users/nana-/Desktop/论文/PPT/4-红色中国石油大学（北京）模板.pptx"
prs = Presentation(TPL)
BLANK = prs.slide_layouts[6]

def walk(shapes):
    for sh in shapes:
        yield sh
        if sh.shape_type == 6:
            yield from walk(sh.shapes)

def set_text(slide, name, text):
    for sh in walk(slide.shapes):
        if sh.name == name and sh.has_text_frame:
            tf = sh.text_frame
            p = tf.paragraphs[0]
            if p.runs:
                p.runs[0].text = text
                for r in p.runs[1:]:
                    r.text = ""
            else:
                p.add_run().text = text
            for extra in tf.paragraphs[1:]:
                for r in extra.runs:
                    r.text = ""
            return True
    return False

def dup(src):
    dst = prs.slides.add_slide(src.slide_layout)
    for sp in list(dst.shapes):
        sp._element.getparent().remove(sp._element)
    rid = {}
    for rId, rel in src.part.rels.items():
        if rel.reltype in (RT.IMAGE, RT.MEDIA):
            rid[rId] = dst.part.relate_to(rel.target_part, rel.reltype)
    for sh in src.shapes:
        el = copy.deepcopy(sh._element)
        for blip in el.iter(qn("a:blip")):
            r = blip.get(qn("r:embed"))
            if r in rid:
                blip.set(qn("r:embed"), rid[r])
        dst.shapes._spTree.append(el)
    return dst

SRC_COVER, SRC_TOC, SRC_END = prs.slides[0], prs.slides[1], prs.slides[38]
SRC_DIV = {1: prs.slides[2], 2: prs.slides[7], 3: prs.slides[12], 4: prs.slides[17], 5: prs.slides[22]}
SRC_CONTENT = prs.slides[3]
SRC_FIG = prs.slides[29]
ORIG_IDS = [id(s) for s in prs.slides]
order = []

# ---------------- 封面
cover = SRC_COVER
set_text(cover, "文本框 19", "Z公司工作流程数字化再造研究")
set_text(cover, "文本框 12", "Research on the Digital Re-engineering of Workflows in Company Z")
for sh in walk(cover.shapes):
    if sh.has_text_frame:
        t = sh.text_frame.text.strip()
        if t.startswith("答辩人"):
            tf = sh.text_frame; tf.paragraphs[0].runs[0].text = "答辩人：姚　烨"
            for r in tf.paragraphs[0].runs[1:]: r.text = ""
        elif t.startswith("指导教师"):
            tf = sh.text_frame; tf.paragraphs[0].runs[0].text = "指导教师：郭庆方"
            for r in tf.paragraphs[0].runs[1:]: r.text = ""
        elif t.startswith("日期"):
            tf = sh.text_frame; tf.paragraphs[0].runs[0].text = "日期：2026年10月"
            for r in tf.paragraphs[0].runs[1:]: r.text = ""
order.append(cover)

# ---------------- 目录（两页，各 6 项）
TOC1 = ["引　言", "第1章　绪论", "第2章　概念与理论", "第3章　现状与问题", "第4章　原因分析", "第5章　平台设计"]
TOC2 = ["第6章　效果评估", "第7章　结论与展望", "参考文献", "致　谢", "附录A　问卷", "附录B　访谈"]
TOC_NAMES = ["文本框 9", "文本框 11", "文本框 13", "文本框 15", "文本框 17", "文本框 19"]
for idx, items in enumerate((TOC1, TOC2)):
    s = SRC_TOC if idx == 0 else dup(SRC_TOC)
    set_text(s, "文本框 5", "CONTENTS")
    for nm, txt in zip(TOC_NAMES, items):
        set_text(s, nm, txt)
    if idx == 1:
        for nm, num in zip(["椭圆 8", "椭圆 10", "椭圆 12", "椭圆 14", "椭圆 16", "椭圆 18"],
                           ["07", "08", "09", "10", "11", "12"]):
            set_text(s, nm, num)
    order.append(s)

def fill_divider(no, title, en):
    s = SRC_DIV[no]
    set_text(s, "矩形 7", title)
    set_text(s, "文本框 8", en)
    set_text(s, "文本框 5", str(no))
    order.append(s)

# ---------------- 内容页
def fill_content(title, en, bullets, tags, page):
    s = dup(SRC_CONTENT)
    set_text(s, "文本框 4", title)
    set_text(s, "文本框 6", en)
    set_text(s, "文本框 7", page)
    # 正文
    for sh in walk(s.shapes):
        if sh.name == "矩形 19" and sh.has_text_frame:
            tf = sh.text_frame
            tf.clear()
            for i, b in enumerate(bullets):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                lead, _, rest = b.partition("：")
                r1 = p.add_run(); r1.text = "· " + lead + ("：" if rest else "")
                r1.font.size = Pt(19); r1.font.bold = True
                r1.font.name = "微软雅黑"; r1.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
                if rest:
                    r2 = p.add_run(); r2.text = rest
                    r2.font.size = Pt(19); r2.font.name = "微软雅黑"
                    r2.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
                p.space_after = Pt(8)
                p.line_spacing = 1.15
            break
    for nm, tag in zip(["文本框 21", "文本框 22", "文本框 23", "文本框 24", "文本框 25"], tags):
        set_text(s, nm, tag)
    order.append(s)

CONTENT = [
 ("研究背景", "RESEARCH BACKGROUND", [
   "政策：数字经济、数据要素市场化",
   "行业：电子招投标与公平竞争审查",
   "竞争：市场调节价、同质化加剧",
   "企业之困：平台上线≠流程再造"],
  ["数字经济", "电子招投标", "公平竞争", "同质化", "流程再造"]),
 ("研究问题", "RESEARCH QUESTIONS", [
   "现状：流程数字化卡在哪三个环节",
   "成因：技术、组织、环境、能力如何作用",
   "方案：如何落到可运行、可测量的平台",
   "效果：数据受限下如何评估"],
  ["问题诊断", "成因", "方案设计", "效果评估", "闭环"]),
 ("研究方法", "METHODOLOGY", [
   "问卷：2024年10—11月，有效185份",
   "访谈：10名管理者与项目经理",
   "台账：再造前后各12个月",
   "设计科学：需求—设计—实现—验证"],
  ["问卷", "访谈", "台账", "设计科学", "三角验证"]),
 ("理论基础", "THEORETICAL BASIS", [
   "BPR：根本性反思与ESIA原则",
   "TOE：技术、组织、环境三重约束",
   "动态能力：感知—抓取—重构",
   "整合：诊断—设计—评估框架"],
  ["BPR", "TOE", "动态能力", "ESIA", "整合框架"]),
 ("文献综述", "LITERATURE REVIEW", [
   "流程再造与流程数字化研究",
   "招标采购数字化实践",
   "智能审核、风险识别与合规",
   "缺口：环节重组与可测量评估结合不足"],
  ["流程再造", "招标数字化", "智能审核", "合规", "研究缺口"]),
 ("概念界定", "KEY CONCEPTS", [
   "数字化再造：环节级重构，而非线上化",
   "数字化转型：技术—业务—组织协同",
   "流程集成：工具互联、数据贯通",
   "成熟度：从事后统计到实时可见"],
  ["流程再造", "数字化转型", "流程集成", "成熟度", "可计算"]),
 ("现状分析", "CURRENT STATUS", [
   "六阶段：受理—编制—答疑—清标—评审—归档",
   "标准化程度不低：85.4%认可",
   "三个瓶颈：人工传递3—5天、复核约2天",
   "纸质流转：40.0%认为频繁"],
  ["六阶段", "标准化", "人工传递", "复核", "纸质流转"]),
 ("问题诊断", "PROBLEM DIAGNOSIS", [
   "集成度低：仅45.9%认同无缝对接",
   "节点冗余：52.4%认为衔接不畅",
   "技能不足：55.0%认为培训覆盖不足",
   "共同点：流程不可计算"],
  ["集成度", "节点冗余", "技能不足", "数据孤岛", "不可计算"]),
 ("成因分析", "CAUSES", [
   "管理机制：决策链长、激励偏业务量",
   "技术基础：系统分散、接口缺失",
   "外部标准：法规与接口快速迭代",
   "培训体系：结构失衡、覆盖不足"],
  ["管理机制", "技术基础", "外部标准", "培训体系", "结构性约束"]),
 ("方案设计", "SOLUTION DESIGN", [
   "目标：可见、可助、可控、可积、环节精简",
   "架构：接入—应用—业务能力—AI—数据",
   "私有化部署，数据不出内网",
   "关键操作留痕，过程可追溯"],
  ["五项目标", "五层架构", "私有化", "留痕", "平台化"]),
 ("对象模型", "PROCESS OBJECT MODEL", [
   "五级对象：项目—时间轴—里程碑—环节—待办",
   "可计算：周期、偏差、延误、返工、响应",
   "可追溯：文件版本化、结果快照化",
   "把流程变成可查询、可统计的数据结构"],
  ["五级对象", "时间轴", "里程碑", "可计算", "可追溯"]),
 ("环节再造", "ESIA RE-ENGINEERING", [
   "复核：三级会签→一级复核+异常抽检",
   "签章：纸质→电子签章，归档线上化",
   "会签：串行→线上并行、超时提醒",
   "清标：逐项人工→系统初筛+人工复核"],
  ["ESIA", "复核", "电子签章", "并行会签", "清标"]),
 ("功能模块", "FUNCTIONAL MODULES", [
   "时间轴、文档解析、清标辅助",
   "围串标识别、合规检查",
   "智能对话与知识库",
   "项目协同与信用查询（法规库15部619条）"],
  ["七模块", "文档解析", "清标", "围串标", "合规"]),
 ("实施推广", "IMPLEMENTATION", [
   "四步：试点—评估—推广—固化",
   "治理：流程治理小组、节点责任人",
   "绩效：节点准时率、一次通过率",
   "能力：分层培训，制度固化"],
  ["试点", "推广", "固化", "流程治理", "绩效"]),
 ("效果评估", "EVALUATION DESIGN", [
   "5个准则、17项指标",
   "3项前后实测、4项上线后观测",
   "10项已建口径、待数据验证",
   "约束：无逐项目明细，不做统计检验"],
  ["指标体系", "17项", "证据分层", "数据约束", "评估口径"]),
 ("实测结果", "MEASURED RESULTS", [
   "项目数：约450→约570个（约+27%）",
   "流程周期：约27→约20天（约-26%）",
   "清标：约3个工作日→约20分钟",
   "口径：台账/时间轴，均含等待期"],
  ["项目数", "周期", "清标", "台账", "口径"]),
 ("情景测算", "SCENARIO ANALYSIS", [
   "返工：总体1.8次（1.2/2.2/3.0）",
   "搬运：约5次/项目",
   "三情景：保守/基准/乐观",
   "边界：参数未校准，仅方向性参考"],
  ["返工", "搬运", "情景", "综合指数", "方向性"]),
 ("结论与展望", "CONCLUSION", [
   "问题是系统性的，单点优化无效",
   "环节+平台+组织，三位一体",
   "周期与清标环节同期显著改善",
   "局限：聚合数据、无检验、V2未纳入"],
  ["系统性", "三位一体", "同期改善", "局限", "展望"]),
]


FIG = "figs_style_thesis_a"
FIGFILES = {
 "framework": "图02_理论整合分析框架.png",
 "problem": "图05_核心问题量化表征.png",
 "fishbone": "图06_问题成因鱼骨图.png",
 "arch": "图12_系统总体技术架构.png",
 "data": "图13_流程数据模型与可算指标.png",
 "measured": "图18_再造前后实测对比.png",
 "scenario": "图16_情景测算结果.png",
}

def fill_figure(title, en, key, side_title, side_body, bottom, page):
    s = dup(SRC_FIG)
    set_text(s, "文本框 4", title)
    set_text(s, "文本框 6", en)
    set_text(s, "文本框 7", page)
    # 删掉模板左侧占位图，换成论文图（等比缩放到 6.12 x 4.12 区域内）
    for sh in list(s.shapes):
        if sh.shape_type == 13 and sh.name == "图片 16":
            sh._element.getparent().remove(sh._element)
    from PIL import Image
    path = os.path.join(FIG, FIGFILES[key])
    iw, ih = Image.open(path).size
    maxw, maxh = 6.12, 4.12
    ratio = min(maxw / iw, maxh / ih)
    w, h = iw * ratio, ih * ratio
    s.shapes.add_picture(path, Emu(int((0.57 + (maxw - w) / 2) * 914400)),
                         Emu(int((1.41 + (maxh - h) / 2) * 914400)),
                         Emu(int(w * 914400)), Emu(int(h * 914400)))
    set_text(s, "文本框 17", side_title)
    set_text(s, "矩形 15", side_body)
    set_text(s, "矩形 19", bottom)
    order.append(s)

# 组装顺序（分隔页 + 内容页 + 图页）
seq = []
seq.append(("div", 1))
for i in range(0, 4):
    seq.append(("c", i))
seq.append(("fig", ("理论框架", "THEORETICAL FRAMEWORK", "framework",
                    "三段递进", "BPR 管怎么改，TOE 管为何能/不能改，动态能力管能否持续；三者共同支撑诊断—设计—评估。",
                    "图2.1　业务流程再造—TOE—动态能力整合分析框架")))
seq.append(("div", 2))
for i in range(4, 9):
    seq.append(("c", i))
seq.append(("fig", ("三大问题", "THREE PROBLEMS", "problem",
                    "证据一致", "45.9% 认同无缝对接；52.4% 认为衔接不畅；55.0% 认为培训不足；三个问题共同指向“流程不可计算”。",
                    "图3.4　核心问题量化表征（问卷 N=185）")))
seq.append(("div", 3))
for i in range(9, 13):
    seq.append(("c", i))
seq.append(("fig", ("平台架构", "PLATFORM ARCHITECTURE", "arch",
                    "五层解耦", "接入—应用—业务能力—AI—数据与任务；私有化部署，数据不出内网。",
                    "图5.1　数字化工作活动平台总体技术架构")))
seq.append(("fig", ("对象模型", "PROCESS DATA MODEL", "data",
                    "可计算", "五级对象把流程拆成可计算的结构：周期、偏差、延误、返工、响应均可自动统计。",
                    "图5.2　流程数据模型与可计算的过程指标")))
seq.append(("div", 4))
for i in range(13, 16):
    seq.append(("c", i))
seq.append(("fig", ("实测结果", "MEASURED COMPARISON", "measured",
                    "三个数", "项目数约+27%，流程周期约−26%，清标由约3个工作日降至约20分钟；口径与归因已在正文说明。",
                    "图6.3　再造前后关键指标对比（实测，各12个月）")))
seq.append(("div", 5))
for i in range(16, 18):
    seq.append(("c", i))

for kind, payload in seq:
    if kind == "div":
        fill_divider(payload, ["", "研究设计", "现状与成因", "方案设计", "效果评估", "结论与展望"][payload],
                     ["", "RESEARCH DESIGN", "STATUS AND CAUSES", "SOLUTION DESIGN", "EFFECT EVALUATION", "CONCLUSION"][payload])
    elif kind == "c":
        fill_content(*CONTENT[payload], page="%02d" % (len(order) + 1))
    else:
        fill_figure(*payload, page="%02d" % (len(order) + 1))

# ---------------- 结束页
end = SRC_END
set_text(end, "副标题 16", "答辩人：姚　烨　｜　指导教师：郭庆方　｜　工商管理（MBA）")
for sh in walk(end.shapes):
    if sh.has_text_frame:
        t = sh.text_frame.text.strip()
        if t.startswith("答辩人"):
            tf = sh.text_frame; tf.paragraphs[0].runs[0].text = "答辩人：姚　烨"
            for r in tf.paragraphs[0].runs[1:]: r.text = ""
        elif t.startswith("指导教师"):
            tf = sh.text_frame; tf.paragraphs[0].runs[0].text = "指导教师：郭庆方"
            for r in tf.paragraphs[0].runs[1:]: r.text = ""
        elif t.startswith("日期"):
            tf = sh.text_frame; tf.paragraphs[0].runs[0].text = "日期：2026年10月"
            for r in tf.paragraphs[0].runs[1:]: r.text = ""
order.append(end)

# ---------------- 删除模板多余页、按顺序重排
sldIds = list(prs.slides._sldIdLst)
all_slides = list(prs.slides)
id_of = {id(s): sldIds[i] for i, s in enumerate(all_slides)}
keep_ids = {id(s) for s in order}
for i, s in enumerate(all_slides):
    if id(s) not in keep_ids:
        prs.slides._sldIdLst.remove(sldIds[i])
for el in list(prs.slides._sldIdLst):
    prs.slides._sldIdLst.remove(el)
for s in order:
    prs.slides._sldIdLst.append(id_of[id(s)])

out = "v12/答辩PPT_4号模板_内容重排.pptx"
prs.save(out)
print("saved", out, "| slides:", len(prs.slides))
