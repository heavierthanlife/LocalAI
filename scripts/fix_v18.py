# -*- coding: utf-8 -*-
"""v1.7 格式修复：在用户手工改过文字的 v1.6 文件上，按《格式.doc》重排版式。
原则：不改任何文字内容（唯一例外：把被误留的"目录占位提示"恢复为自动目录域）。
"""
import re
import docx
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = "v12/_v18_stage.docx"
DST = "v12/v1.8_因果校正版.docx"
SONG, HEI, TNR, ARIAL = "宋体", "黑体", "Times New Roman", "Arial"

d = Document(SRC)
body = d.element.body

# ---------------------------------------------------------------- 样式配置
def set_fonts(el_rPr, cjk, latin):
    rF = el_rPr.find(qn("w:rFonts"))
    if rF is None:
        rF = OxmlElement("w:rFonts"); el_rPr.append(rF)
    rF.set(qn("w:ascii"), latin); rF.set(qn("w:hAnsi"), latin); rF.set(qn("w:eastAsia"), cjk)


def cfg(name, cjk, latin, size, bold=False, align=None, line=None, before=0, after=0, first=None):
    try:
        st = d.styles[name]
    except KeyError:
        st = d.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    st.base_style = d.styles["Normal"]
    st.font.name = latin
    st.font.size = Pt(size)
    st.font.bold = bold
    st.font.color.rgb = RGBColor(0, 0, 0)
    rPr = st.element.get_or_add_rPr()
    set_fonts(rPr, cjk, latin)
    pf = st.paragraph_format
    if align is not None: pf.alignment = align
    if line is not None: pf.line_spacing = line
    pf.space_before = Pt(before); pf.space_after = Pt(after)
    pf.first_line_indent = Pt(first) if first is not None else None
    return st


BODY = "正文1"
cfg(BODY, SONG, TNR, 12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.25, first=24)
cfg("标题1", HEI, ARIAL, 15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.5, after=11)
cfg("标题2", HEI, ARIAL, 14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.5, before=9)
cfg("标题3", HEI, ARIAL, 12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.5, before=8)
cfg("图题", HEI, ARIAL, 10.5, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0, before=3)
cfg("图题EN", TNR, TNR, 10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0, after=3)
cfg("表题", HEI, ARIAL, 10.5, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0, before=6)
cfg("图注", SONG, TNR, 10.5, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.0, after=6)
cfg("参考文献", SONG, TNR, 10.5, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.25)
d.styles["参考文献"].paragraph_format.left_indent = Pt(21)
d.styles["参考文献"].paragraph_format.first_line_indent = Pt(-21)
cfg("关键词", HEI, ARIAL, 12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.25, first=24, before=12)
cfg("正文EN", TNR, TNR, 12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.25, first=24)
cfg("标题1EN", TNR, TNR, 15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.5, after=11)
cfg("英文题目", TNR, TNR, 15, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.25, before=36)
cfg("声明标题", HEI, ARIAL, 15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.5, after=11)

# 大纲级别（目录域按 1-3 级收录）
for n, lvl in (("标题1", 0), ("标题2", 1), ("标题3", 2), ("声明标题", 0), ("标题1EN", 0)):
    pPr = d.styles[n].element.get_or_add_pPr()
    for old in pPr.findall(qn("w:outlineLvl")):
        pPr.remove(old)
    ol = OxmlElement("w:outlineLvl"); ol.set(qn("w:val"), str(lvl))
    pPr.insert(0, ol)

# ---------------------------------------------------------------- 工具
CJK_PAT = re.compile(r"[\u4e00-\u9fa5]")

def clear_direct(p):
    pf = p.paragraph_format
    pf.alignment = None
    pf.first_line_indent = None
    pf.left_indent = None
    pf.right_indent = None
    pf.line_spacing = None
    pf.space_before = None
    pf.space_after = None
    pf.keep_with_next = None


def set_run(r, cjk, latin, size, bold=None, keep_sup=True):
    sup = r.font.superscript
    r.font.name = latin
    r._element.get_or_add_rPr()
    set_fonts(r._element.rPr, cjk, latin)
    r.font.size = Pt(size)
    r.bold = bold
    r.font.color.rgb = RGBColor(0, 0, 0)
    r.font.underline = False
    if keep_sup:
        r.font.superscript = sup


def norm_para(p, style, cjk=SONG, latin=TNR, size=12, bold=None, sup_ok=True):
    clear_direct(p)
    p.style = d.styles[style]
    for r in p.runs:
        set_run(r, cjk, latin, size, bold=bold, keep_sup=sup_ok)


def is_fig_para(p):
    return bool(p._p.findall(".//" + qn("a:blip")))


# ---------------------------------------------------------------- 段落分类与重排
TOC_PLACEHOLDER = "（目录为自动域"
H1_TEXTS = ("摘", "ABSTRACT", "目", "引", "第", "参 考 文 献", "参 考", "致", "附录")
counts = {}
toc_placeholder_p = None

for p in d.paragraphs:
    t = p.text.strip()
    if TOC_PLACEHOLDER in t:
        toc_placeholder_p = p
        continue
    style = p.style.name
    # 图片段落
    if is_fig_para(p):
        clear_direct(p)
        p.style = d.styles[BODY]
        pf = p.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(12); pf.space_after = Pt(0)
        pf.line_spacing = 1.0; pf.keep_with_next = True
        continue
    if not t:
        clear_direct(p)
        p.style = d.styles[BODY]
        p.paragraph_format.first_line_indent = Pt(0)
        continue
    # 目录条目（Word 生成的 TOC 1/2/3 样式段落）
    if style.lower().startswith("toc") or style.startswith("目录"):
        lvl = 1
        m = re.search(r"([1-3])$", style)
        if m: lvl = int(m.group(1))
        clear_direct(p)
        p.style = d.styles[BODY]
        pf = p.paragraph_format
        pf.first_line_indent = Pt(0); pf.left_indent = Pt(24 * (lvl - 1))
        pf.line_spacing = 1.25
        for r in p.runs:
            set_run(r, SONG, TNR, 10.5)
        counts["目录"] = counts.get("目录", 0) + 1
        continue
    # 标题类（保留原样式，重设格式）
    if style in ("标题1", "标题2", "标题3", "标题1EN", "声明标题", "英文题目"):
        size = {"标题1": 15, "标题2": 14, "标题3": 12, "标题1EN": 15, "声明标题": 15, "英文题目": 15}[style]
        cjk = TNR if style in ("标题1EN", "英文题目") else HEI
        latin = TNR if style in ("标题1EN", "英文题目") else ARIAL
        norm_para(p, style, cjk, latin, size, bold=None)
        counts[style] = counts.get(style, 0) + 1
        continue
    # 正文/普通段落：按文字特征重新归类
    if re.match(r"^图\s*\d+\.\d+", t):
        norm_para(p, "图题", HEI, ARIAL, 10.5); p.paragraph_format.keep_with_next = True
        counts["图题"] = counts.get("图题", 0) + 1; continue
    if re.match(r"^表\s*\d+\.\d+", t):
        norm_para(p, "表题", HEI, ARIAL, 10.5); p.paragraph_format.keep_with_next = True
        counts["表题"] = counts.get("表题", 0) + 1; continue
    if t.startswith(("Fig.", "Table ")):
        norm_para(p, "图题EN", TNR, TNR, 10.5, bold=True)
        counts["图题EN"] = counts.get("图题EN", 0) + 1; continue
    if t.startswith(("注：", "数据来源")):
        norm_para(p, "图注", SONG, TNR, 10.5)
        counts["图注"] = counts.get("图注", 0) + 1; continue
    if t.startswith("关键词"):
        norm_para(p, "关键词", HEI, ARIAL, 12); counts["关键词"] = counts.get("关键词", 0) + 1; continue
    if t.lower().startswith(("key words", "keywords")):
        norm_para(p, "正文EN", TNR, TNR, 12, bold=None)
        if p.runs:
            p.runs[0].bold = True
        counts["KeyWords"] = counts.get("KeyWords", 0) + 1; continue
    if re.match(r"^\[\d+\]", t):
        clear_direct(p); p.style = d.styles["参考文献"]
        p.paragraph_format.first_line_indent = Pt(-21); p.paragraph_format.left_indent = Pt(21)
        for r in p.runs:
            set_run(r, SONG, TNR, 10.5)
        counts["参考文献条目"] = counts.get("参考文献条目", 0) + 1; continue
    _t0 = t.replace(" ", "")
    _is_h1 = (_t0 in ("摘要", "目录", "引言") or re.match(r"^第\d+章", t)
              or _t0 in ("参考文献", "致谢") or _t0.startswith("附录"))
    if _is_h1:
        norm_para(p, "标题1", HEI, ARIAL, 15); counts["标题1"] = counts.get("标题1", 0) + 1; continue
    if re.match(r"^\d+(\.\d+)+", t):
        dots = t.split(" ")[0].count(".")
        st = "标题2" if dots <= 1 else "标题3"
        size = 14 if st == "标题2" else 12
        norm_para(p, st, HEI, ARIAL, size); counts[st] = counts.get(st, 0) + 1; continue
    if re.match(r"^[一二三四五六七八九十]、|^第[一二三四五六]部分", t):
        norm_para(p, "标题2", HEI, ARIAL, 14); counts["标题2"] = counts.get("标题2", 0) + 1; continue
    # 封面文字：保持原样式但统一字体
    if t in ("中图分类号：", "硕士专业学位论文", "Z公司工作流程数字化再造研究",
             "Research on the Digital Re-engineering of Workflows in Company Z", "二○二六年十月"):
        counts["封面"] = counts.get("封面", 0) + 1
        continue
    norm_para(p, BODY)
    counts["正文"] = counts.get("正文", 0) + 1

print("段落归类:", counts)

# ---------------------------------------------------------------- 目录域重建
if toc_placeholder_p is not None:
    newp = d.add_paragraph(style=d.styles[BODY])
    newp.paragraph_format.first_line_indent = Pt(0)
    newp.paragraph_format.line_spacing = 1.25
    r = newp.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), 'TOC \\o "1-3" \\h \\z \\u')
    ph = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    col = OxmlElement("w:color"); col.set(qn("w:val"), "808080")
    ital = OxmlElement("w:i")
    rPr.append(ital); rPr.append(col)
    ph.append(rPr)
    t = OxmlElement("w:t"); t.text = "（目录为自动域：右键 → 更新域 → 更新整个目录）"
    ph.append(t); fld.append(ph)
    r._r.addnext(fld)
    toc_placeholder_p._p.addnext(newp._p)
    toc_placeholder_p._p.getparent().remove(toc_placeholder_p._p)
    print("目录域已重建")

# ---------------------------------------------------------------- 表格重排（三线表）
def cell_borders(cell, top=None, bottom=None, sz="4"):
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:tcBorders")):
        tcPr.remove(old)
    cb = OxmlElement("w:tcBorders")
    for edge, use in (("top", top), ("bottom", bottom)):
        if use:
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), sz); e.set(qn("w:color"), "000000")
            cb.append(e)
    if len(cb):
        tcPr.append(cb)


def table_borders_none(tb):
    tblPr = tb._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "none")
        b.append(e)
    tblPr.append(b)


for ti, tb in enumerate(d.tables):
    first = tb.cell(0, 0).text.strip()
    if first == "论文类型":      # 封面表：只统一字体
        for row in tb.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        set_run(r, SONG, TNR, 12)
        continue
    table_borders_none(tb)
    # 表头重复 + 行高
    tr0 = tb.rows[0]._tr
    trPr = tr0.get_or_add_trPr()
    for old in trPr.findall(qn("w:tblHeader")):
        trPr.remove(old)
    th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
    for ri, row in enumerate(tb.rows):
        trPr = row._tr.get_or_add_trPr()
        for old in trPr.findall(qn("w:trHeight")):
            trPr.remove(old)
        trh = OxmlElement("w:trHeight"); trh.set(qn("w:val"), "448"); trh.set(qn("w:hRule"), "atLeast")
        trPr.insert(0, trh)
        for ci, cell in enumerate(row.cells):
            for p in cell.paragraphs:
                clear_direct(p)
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.first_line_indent = Pt(0)
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1.25
                for r in p.runs:
                    set_run(r, SONG, TNR, 10.5, bold=(ri == 0), keep_sup=False)
            top = (ri == 0)
            bottom = (ri == 0) or (ri == len(tb.rows) - 1)
            cell_borders(cell, top=top, bottom=bottom)

# ---------------------------------------------------------------- 分节、页眉页脚、页码
def page_setup(sec):
    sec.page_width = Cm(21); sec.page_height = Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(3)
    sec.top_margin = sec.bottom_margin = Cm(3)
    sec.header_distance = Cm(2); sec.footer_distance = Cm(2)


def hf_set(hf, text, size=10.5, cjk=HEI, latin=ARIAL):
    p = hf.paragraphs[0]
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    if text:
        r = p.add_run(text)
        r.font.name = latin; r.font.size = Pt(size)
        r._element.get_or_add_rPr(); set_fonts(r._element.rPr, cjk, latin)


def hf_page(hf, size=10.5, cjk=SONG, latin=ARIAL):
    p = hf.paragraphs[0]
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r"); rPr = OxmlElement("w:rPr")
    rf = OxmlElement("w:rFonts")
    rf.set(qn("w:ascii"), latin); rf.set(qn("w:hAnsi"), latin); rf.set(qn("w:eastAsia"), cjk)
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(int(size * 2)))
    rPr.append(rf); rPr.append(sz); r.append(rPr)
    t = OxmlElement("w:t"); t.text = "1"; r.append(t); fld.append(r)
    p._p.append(fld)


def set_pg_num(sec, fmt=None, start=None):
    sectPr = sec._sectPr
    for old in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(old)
    if fmt is None and start is None:
        return
    pg = OxmlElement("w:pgNumType")
    if fmt: pg.set(qn("w:fmt"), fmt)
    if start is not None: pg.set(qn("w:start"), str(start))
    # 插到 cols 之前
    cols = sectPr.find(qn("w:cols"))
    if cols is not None:
        cols.addprevious(pg)
    else:
        sectPr.append(pg)


h1_titles = [p.text.strip() for p in d.paragraphs if p.style.name == "标题1"]
body_titles = [t for t in h1_titles if re.match(r"^(引|第\d+章|参 考|致|附录)", t)]
print("正文节标题:", body_titles, len(body_titles))
secs = d.sections
# 清除各节共享的页眉页脚引用，避免"相邻节共用同一页眉而互相覆盖"
for sec in secs:
    for tag in ("w:headerReference", "w:footerReference"):
        for el in sec._sectPr.findall(qn(tag)):
            sec._sectPr.remove(el)
for i, sec in enumerate(secs):
    page_setup(sec)
    sec.header.is_linked_to_previous = False
    sec.even_page_header.is_linked_to_previous = False
    sec.footer.is_linked_to_previous = False
    if i == 0:                       # 封面
        hf_set(sec.header, ""); hf_set(sec.even_page_header, ""); 
        p = sec.footer.paragraphs[0]
        for r in list(p.runs): r._r.getparent().remove(r._r)
        set_pg_num(sec)
    elif i == 1:                     # 声明/摘要/ABSTRACT/目录（罗马页码）
        hf_set(sec.header, ""); hf_set(sec.even_page_header, "")
        hf_page(sec.footer, cjk=SONG, latin=TNR)
        set_pg_num(sec, "upperRoman", 1)
    else:                            # 正文各节
        hf_set(sec.header, "硕士专业学位论文")
        title = body_titles[i - 2] if (i - 2) < len(body_titles) else ""
        hf_set(sec.even_page_header, title.replace("  ", " "))
        hf_page(sec.footer, cjk=SONG, latin=ARIAL)
        if i == 2:
            set_pg_num(sec, "decimal", 1)
        else:
            set_pg_num(sec)

# ---------------------------------------------------------------- 保存
d.save(DST)
print("saved", DST)

# ---------------------------------------------------------------- 封面文字格式
from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
COVER_FMT = {
    "中图分类号：": (SONG, 12, False, WD_ALIGN_PARAGRAPH.LEFT),
    "二○二五年十二月": (SONG, 14, True, WD_ALIGN_PARAGRAPH.CENTER),
    "硕士专业学位论文": (HEI, 26, True, WD_ALIGN_PARAGRAPH.CENTER),
    "Z公司工作流程数字化再造研究": (HEI, 22, True, WD_ALIGN_PARAGRAPH.CENTER),
    "Research on the Digital Re-engineering of Workflows in Company Z": (TNR, 14, True, WD_ALIGN_PARAGRAPH.CENTER),
    "二○二六年十月": (SONG, 14, True, WD_ALIGN_PARAGRAPH.CENTER),
}
cover_done = False
for p in d.paragraphs:
    if p.style.name == "声明标题":
        cover_done = True
    if cover_done:
        continue
    t = p.text.strip()
    if t.startswith("中图分类号"):
        t = "中图分类号："
    if t in COVER_FMT:
        cjk, size, bold, align = COVER_FMT[t]
        p.style = d.styles[BODY]
        clear_direct(p)
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.line_spacing = 1.25
        p.alignment = align
        for r in p.runs:
            r.font.name = TNR if cjk in (TNR, SONG) else ARIAL
            r._element.get_or_add_rPr(); set_fonts(r._element.rPr, cjk, r.font.name)
            r.font.size = Pt(size); r.bold = bold; r.font.color.rgb = RGBColor(0, 0, 0)

# ---------------------------------------------------------------- 目录样式（供更新域后使用）
for lvl in (1, 2, 3):
    name = f"TOC {lvl}"
    try:
        st = d.styles[name]
    except KeyError:
        st = d.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    st.base_style = d.styles[BODY]
    st.font.name = TNR; st.font.size = Pt(12); st.font.bold = False
    rPr = st.element.get_or_add_rPr(); set_fonts(rPr, SONG, TNR)
    pf = st.paragraph_format
    pf.first_line_indent = Pt(0)
    pf.left_indent = Pt(24 * (lvl - 1))
    pf.line_spacing = 1.25
    pf.space_before = Pt(0); pf.space_after = Pt(0)
    pf.tab_stops.add_tab_stop(Cm(15), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)

d.save(DST)
print("saved(2)", DST)
