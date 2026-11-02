# -*- coding: utf-8 -*-
"""v3.0 格式收尾：4.1.4 图表归位、失效交叉引用修订、按新 6 章重建分节与页眉页码。"""
import copy, re, docx
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

d = docx.Document("v12/v3.0_结构重构版.docx")
body = d.element.body

def txt(el):
    if el.tag == qn("w:tbl"): return "[TABLE]"
    return "".join(t.text or "" for t in el.iter(qn("w:t")))

# ---------- 1) 4.1.4 内部重排
children = [el for el in body.iterchildren() if el.tag in (qn("w:p"), qn("w:tbl"))]
ts = [txt(e) for e in children]
i414 = next(i for i,t in enumerate(ts) if t.strip().startswith("4.1.4"))
i42 = next(i for i,t in enumerate(ts) if t.strip().startswith("4.2  核心环节"))
seg = children[i414:i42]
seg_t = ts[i414:i42]
def idx(pref):
    for k,t in enumerate(seg_t):
        if t.strip().startswith(pref): return k
    raise KeyError(pref)
a_eval = idx("指标体系的构建遵循三条原则")
a_meas = idx("经公司许可，本研究获得再造前后各12个月")
a_scen = idx("在实测对比的基础上")
a_sens = idx("本节从三方面讨论结论的稳健性")
a_stat = idx("调查现状情况聚焦于Z公司工作流程的实际运行")
a_sum  = idx("本章构建了包含5个准则")
order = list(range(0,a_eval)) + list(range(a_stat,a_sum)) + list(range(a_eval,a_meas)) + \
        list(range(a_meas,a_scen)) + list(range(a_scen,a_sens)) + list(range(a_sens,a_stat)) + list(range(a_sum,len(seg)))
anchor = children[i42]
for el in seg: body.remove(el)
for k in order: anchor.addprevious(seg[k])
print("4.1.4 reordered:", len(order))

# ---------- 2) 失效交叉引用
FIX = [
 ("并在6.5节讨论情景参数与指标选择的影响", "并在本节后段讨论情景参数与指标选择的影响"),
 ("情景参数敏感性见6.5节", "情景参数敏感性见本节"),
 ("数据来源：3.5.2节与3.6节调查结果", "数据来源：4.3.3节与4.4节调查结果"),
 ("见6.3节", "见4.1.4节"),
 ("第6章6.2节", "第4章4.5节"),
]
for p in d.paragraphs:
    for r in p.runs:
        for a,b in FIX:
            if a in r.text: r.text = r.text.replace(a,b)

# ---------- 3) 重建分节
H1S = [p for p in d.paragraphs if p.style.name == "标题1" and (p.text.strip().startswith("第") or p.text.strip().startswith("参 考") or p.text.strip().startswith("致") or p.text.strip().startswith("附录"))]
body_start = next(p for p in H1S if p.text.strip().startswith("第1章"))._p
# 删除正文区内的分节符（保留前置与封面）
for p in d.paragraphs:
    el = p._p
    if el is body_start: pass
    # 位置在 body_start 之后
    if body_start.getprevious() is not None:
        pass
removed = 0
after = False
for p in d.paragraphs:
    if p._p is body_start: after = True; continue
    if not after: continue
    pPr = p._p.find(qn("w:pPr"))
    if pPr is not None and pPr.find(qn("w:sectPr")) is not None:
        pPr.remove(pPr.find(qn("w:sectPr"))); removed += 1
print("removed body sectPr:", removed)

final_sect = d.sections[-1]._sectPr
# 在每个 H1（第1章之后）前插入分节符
for p in H1S:
    newp = OxmlElement("w:p"); pPr = OxmlElement("w:pPr")
    pPr.append(copy.deepcopy(final_sect)); newp.append(pPr)
    p._p.addprevious(newp)

# ---------- 4) 页眉页脚
def clear_refs(sec):
    for tag in ("w:headerReference", "w:footerReference"):
        for el in sec._sectPr.findall(qn(tag)): sec._sectPr.remove(el)

def hf_text(hf, text, cjk="黑体", latin="Arial", size=10.5):
    p = hf.paragraphs[0]
    for r in list(p.runs): r._r.getparent().remove(r._r)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    if text:
        r = p.add_run(text); r.font.size = Pt(size); r.font.name = latin
        rPr = r._element.get_or_add_rPr(); rf = rPr.find(qn("w:rFonts"))
        if rf is None: rf = OxmlElement("w:rFonts"); rPr.append(rf)
        rf.set(qn("w:ascii"), latin); rf.set(qn("w:hAnsi"), latin); rf.set(qn("w:eastAsia"), cjk)

def hf_page(hf, size=10.5, cjk="宋体", latin="Arial"):
    p = hf.paragraphs[0]
    for r in list(p.runs): r._r.getparent().remove(r._r)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r"); rPr = OxmlElement("w:rPr")
    rf = OxmlElement("w:rFonts"); rf.set(qn("w:ascii"), latin); rf.set(qn("w:hAnsi"), latin); rf.set(qn("w:eastAsia"), cjk)
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "21")
    rPr.append(rf); rPr.append(sz); r.append(rPr)
    t = OxmlElement("w:t"); t.text = "1"; r.append(t); fld.append(r); p._p.append(fld)

def set_pg(sec, fmt=None, start=None):
    sectPr = sec._sectPr
    for old in sectPr.findall(qn("w:pgNumType")): sectPr.remove(old)
    if fmt is None and start is None: return
    pg = OxmlElement("w:pgNumType")
    if fmt: pg.set(qn("w:fmt"), fmt)
    if start is not None: pg.set(qn("w:start"), str(start))
    cols = sectPr.find(qn("w:cols"))
    (cols.addprevious(pg) if cols is not None else sectPr.append(pg))

titles = [p.text.strip().replace("  ", " ") for p in H1S]
print("H1 titles:", titles)
for i, sec in enumerate(d.sections):
    clear_refs(sec)
    sec.header.is_linked_to_previous = False
    sec.even_page_header.is_linked_to_previous = False
    sec.footer.is_linked_to_previous = False
    if i == 0:
        hf_text(sec.header, ""); hf_text(sec.even_page_header, "")
        p = sec.footer.paragraphs[0]
        for r in list(p.runs): r._r.getparent().remove(r._r)
        set_pg(sec)
    elif i == 1:
        hf_text(sec.header, ""); hf_text(sec.even_page_header, "")
        hf_page(sec.footer, cjk="宋体", latin="Times New Roman")
        set_pg(sec, "upperRoman", 1)
    else:
        hf_text(sec.header, "硕士专业学位论文")
        title = titles[i-2] if i-2 < len(titles) else ""
        hf_text(sec.even_page_header, title)
        hf_page(sec.footer)
        if i == 2: set_pg(sec, "decimal", 1)
        else: set_pg(sec)

d.save("v12/v3.0_结构重构版.docx")
d2 = docx.Document("v12/v3.0_结构重构版.docx")
print("sections", len(d2.sections), "paras", len(d2.paragraphs))
for i,s in enumerate(d2.sections):
    print(i, repr(s.even_page_header.paragraphs[0].text.strip()[:30]))
