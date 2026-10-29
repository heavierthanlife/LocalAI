# -*- coding: utf-8 -*-
"""按学校格式规范（格式.doc）重建论文：v1.4 学校格式版。"""
import os
import re
import pickle
from io import BytesIO

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SONG, HEI, TNR, ARIAL = "宋体", "黑体", "Times New Roman", "Arial"
DATA = pickle.load(open("v12/content_v15.pkl", "rb"))
ITEMS = DATA["items"]
TITLE_EN = DATA["title_en"]
FIG_EN = DATA["fig_en"]
TAB_EN = DATA["tab_en"]

TEXT_W_CM = 15.0
FIG_W_CM = 14.9

# ------------------------------------------------------------------ 基础工具
ORDER = {
    "pPr": ["pStyle","keepNext","keepLines","pageBreakBefore","framePr","widowControl","numPr",
            "suppressLineNumbers","pBdr","shd","tabs","suppressAutoHyphens","kinsoku","wordWrap",
            "overflowPunct","topLinePunct","autoSpaceDE","autoSpaceDN","bidi","adjustRightInd",
            "snapToGrid","spacing","ind","contextualSpacing","mirrorIndents","suppressOverlap","jc",
            "textDirection","textAlignment","textboxTightWrap","outlineLvl","divId","cnfStyle","rPr",
            "sectPr","pPrChange"],
    "sectPr": ["headerReference","footerReference","footnotePr","endnotePr","type","pgSz","pgMar",
               "paperSrc","pgBorders","lnNumType",
               "pgNumType","cols","formProt","vAlign","noEndnote","titlePg","textDirection","bidi",
               "rtlGutter","docGrid","printerSettings","sectPrChange"],
    "tblPr": ["tblStyle","tblpPr","tblOverlap","bidiVisual","tblStyleRowBandSize","tblStyleColBandSize",
              "tblW","jc","tblCellSpacing","tblInd","tblBorders","shd","tblLayout","tblCellMar",
              "tblLook","tblCaption","tblDescription"],
    "trPr": ["cnfStyle","divId","gridBefore","gridAfter","wBefore","wAfter","cantSplit","trHeight",
             "tblHeader","tblCellSpacing","jc","hidden","ins","del","trPrChange"],
    "tcPr": ["cnfStyle","tcW","gridSpan","hMerge","vMerge","tcBorders","shd","noWrap","tcMar",
             "textDirection","tcFitText","vAlign","hideMark"],
    "settings": ["writeProtection","view","zoom","removePersonalInformation","removeDateAndTime",
                 "doNotDisplayPageBoundaries","displayBackgroundShape","printPostScriptOverText",
                 "printFractionalCharacterWidth","printFormsData","embedTrueTypeFonts",
                 "embedSystemFonts","saveSubsetFonts","saveFormsData","mirrorMargins",
                 "alignBordersAndEdges","bordersDoNotSurroundHeader","bordersDoNotSurroundFooter",
                 "gutterAtTop","hideSpellingErrors","hideGrammaticalErrors","activeWritingStyle",
                 "proofState","formsDesign","attachedTemplate","linkStyles","stylePaneFormatFilter",
                 "stylePaneSortMethod","documentType","mailMerge","revisionView","trackChanges",
                 "doNotTrackMoves","doNotTrackFormatting","documentProtection","autoFormatOverride",
                 "styleLockTheme","styleLockQFSet","defaultTabStop","autoHyphenation",
                 "consecutiveHyphenLimit","hyphenationZone","doNotHyphenateCaps","showEnvelope",
                 "summaryLength","clickAndTypeStyle","defaultTableStyle","evenAndOddHeaders",
                 "bookFoldRevPrinting","bookFoldPrinting","bookFoldPrintingSheets","drawingGridHorizontalSpacing",
                 "drawingGridVerticalSpacing","displayHorizontalDrawingGridEvery",
                 "displayVerticalDrawingGridEvery","doNotUseMarginsForDrawingGridOrigin",
                 "drawingGridHorizontalOrigin","drawingGridVerticalOrigin","doNotShadeFormData",
                 "noPunctuationKerning","characterSpacingControl","printTwoOnOne","strictFirstAndLastChars",
                 "noLineBreaksAfter","noLineBreaksBefore","savePreviewPicture","doNotValidateAgainstSchema",
                 "saveInvalidXml","ignoreMixedContent","alwaysShowPlaceholderText","doNotDemarcateInvalidXml",
                 "saveXmlDataOnly","useXSLTWhenSaving","saveThroughXslt","showXMLTags",
                 "alwaysMergeEmptyNamespace","updateFields","hdrShapeDefaults","footnotePr","endnotePr",
                 "compat","docVars","rsids","mathPr","attachedSchema","themeFontLang","clrSchemeMapping",
                 "doNotIncludeSubdocsInStats","doNotAutoCompressPictures","forceUpgrade","captions",
                 "readModeInkLockDown","smartTagType","schemaLibrary","shapeDefaults","doNotEmbedSmartTags",
                 "decimalSymbol","listSeparator"]
}

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def ins(parent, child, kind):
    """按 OOXML schema 顺序把 child 插入 parent（同标签已存在则先删除）。"""
    tag = child.tag.replace(W, "")
    for old in parent.findall(child.tag):
        parent.remove(old)
    order = ORDER[kind]
    idx = order.index(tag) if tag in order else len(order)
    for ch in list(parent):
        ctag = ch.tag.replace(W, "")
        pos = order.index(ctag) if ctag in order else len(order)
        if pos > idx:
            ch.addprevious(child)
            return child
    parent.append(child)
    return child


def set_run_fonts(run, cjk=SONG, latin=TNR, size=12, bold=False, color=None):
    run.font.name = latin
    run.font.size = Pt(size)
    run.bold = bold
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), latin)
    rFonts.set(qn("w:hAnsi"), latin)
    rFonts.set(qn("w:eastAsia"), cjk)
    if color:
        run.font.color.rgb = color


def add_field(paragraph, instr):
    r = paragraph.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), instr)
    r._r.addnext(fld)


def set_pg_num(section, fmt=None, start=None):
    sectPr = section._sectPr
    pg = OxmlElement("w:pgNumType")
    ins(sectPr, pg, "sectPr")
    if fmt:
        pg.set(qn("w:fmt"), fmt)
    if start is not None:
        pg.set(qn("w:start"), str(start))


def page_setup(section):
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = section.right_margin = Cm(3)
    section.top_margin = section.bottom_margin = Cm(3)
    section.header_distance = Cm(2)
    section.footer_distance = Cm(2)


def clear_para(p):
    for r in list(p.runs):
        r._r.getparent().remove(r._r)


def hf_text(hf, text, size=10.5):
    p = hf.paragraphs[0]
    clear_para(p)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    if text:
        r = p.add_run(text)
        set_run_fonts(r, cjk=HEI, latin=ARIAL, size=size)


def hf_page(hf, size=10.5, cjk=SONG, latin=ARIAL):
    p = hf.paragraphs[0]
    clear_para(p)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rf = OxmlElement("w:rFonts")
    rf.set(qn("w:ascii"), latin); rf.set(qn("w:hAnsi"), latin); rf.set(qn("w:eastAsia"), cjk)
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(int(size * 2)))
    rPr.append(rf); rPr.append(sz)
    r.append(rPr)
    t = OxmlElement("w:t"); t.text = "1"
    r.append(t)
    fld.append(r)
    p._p.append(fld)


# ------------------------------------------------------------------ 样式
doc = Document()
# 奇偶页不同页眉
settings = doc.settings.element
if settings.find(qn("w:evenAndOddHeaders")) is None:
    ins(settings, OxmlElement("w:evenAndOddHeaders"), "settings")


def style_para(name, base="Normal"):
    try:
        return doc.styles[name]
    except KeyError:
        return doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)


def config_style(name, cjk, latin, size, bold=False, align=None, line=None,
                 before=0, after=0, first_indent=None, base="Normal"):
    st = style_para(name, base)
    st.base_style = doc.styles[base]
    st.font.name = latin
    st.font.size = Pt(size)
    st.font.bold = bold
    rPr = st.element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts"); rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), latin)
    rFonts.set(qn("w:hAnsi"), latin)
    rFonts.set(qn("w:eastAsia"), cjk)
    pf = st.paragraph_format
    if align is not None:
        pf.alignment = align
    if line is not None:
        pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if first_indent is not None:
        pf.first_line_indent = Pt(first_indent)
    return st


def set_outline(style_name, lvl):
    st = doc.styles[style_name]
    pPr = st.element.get_or_add_pPr()
    ol = OxmlElement("w:outlineLvl")
    ol.set(qn("w:val"), str(lvl))
    ins(pPr, ol, "pPr")


config_style("正文", SONG, TNR, 12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.25,
             first_indent=24)
config_style("标题1", HEI, ARIAL, 15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
             line=1.5, before=0, after=11)
config_style("标题2", HEI, ARIAL, 14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
             line=1.5, before=9, after=0)
config_style("标题3", HEI, ARIAL, 12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
             line=1.5, before=8, after=0)
config_style("图题", HEI, ARIAL, 10.5, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0,
             before=3, after=0)
config_style("图题EN", TNR, TNR, 10.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
             line=1.0, before=0, after=3)
config_style("图注", SONG, TNR, 10.5, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.0, before=0, after=6)
config_style("表题", HEI, ARIAL, 10.5, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0,
             before=6, after=0)
config_style("参考文献", SONG, TNR, 10.5, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.25)
doc.styles["参考文献"].paragraph_format.left_indent = Pt(21)
doc.styles["参考文献"].paragraph_format.first_line_indent = Pt(-21)
config_style("关键词", HEI, ARIAL, 12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
             line=1.25, first_indent=24, before=12, after=0)
config_style("正文EN", TNR, TNR, 12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.25, first_indent=24)
config_style("标题1EN", TNR, TNR, 15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
             line=1.5, before=0, after=11)
config_style("英文题目", TNR, TNR, 15, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER,
             line=1.25, before=36, after=0)
config_style("声明标题", HEI, ARIAL, 15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
             line=1.5, before=0, after=11)
for _n, _l in (("标题1", 0), ("标题2", 1), ("标题3", 2), ("声明标题", 0), ("标题1EN", 0)):
    set_outline(_n, _l)


def P(text="", style="正文", align=None, space_before=None, space_after=None,
      first_indent=None, cjk=None, latin=None, size=None, bold=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    if space_before is not None:
        pf.space_before = Pt(space_before)
    if space_after is not None:
        pf.space_after = Pt(space_after)
    if first_indent is not None:
        pf.first_line_indent = Pt(first_indent)
    if text:
        add_text_runs(p, text, cjk, latin, size, bold)
    return p


CITE = re.compile(r"(\[[\d,\-–，、\s]+\])")


def add_text_runs(p, text, cjk=None, latin=None, size=None, bold=False):
    """把 [n] 引用转成上标 run。"""
    cjk = cjk or SONG
    latin = latin or TNR
    size = size or 12
    for seg in CITE.split(text):
        if not seg:
            continue
        r = p.add_run(seg)
        set_run_fonts(r, cjk=cjk, latin=latin, size=size, bold=bold)
        if CITE.fullmatch(seg):
            r.font.superscript = True


# ---------------------------------------------------------------- 编号规范化
def fix_num(text):
    text = re.sub(r"图\s*(\d+)-(\d+)", r"图\1.\2", text)
    text = re.sub(r"表\s*(\d+)-(\d+)", r"表\1.\2", text)
    return text


def norm_h1(text):
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    m = re.match(r"^第(\d+)章\s*(.+)$", t)
    if m:
        return f"第{m.group(1)}章  {m.group(2)}"
    m = re.match(r"^附录\s*([A-Z])\s*(.+)$", t)
    if m:
        return f"附录{m.group(1)}  {m.group(2)}"
    if t.replace(" ", "") == "目录":
        return "目    录"
    if t.replace(" ", "") == "致谢":
        return "致    谢"
    if t.replace(" ", "") == "引言":
        return "引    言"
    if t.replace(" ", "") == "参考文献":
        return "参 考 文 献"
    return t


def norm_h23(text):
    t = re.sub(r"\s+", " ", text.strip())
    m = re.match(r"^(\d+(?:\.\d+)+)\s*(.+)$", t)
    if m:
        return f"{m.group(1)}  {m.group(2)}"
    return t


# ------------------------------------------------------------------- 封面
sec = doc.sections[0]
page_setup(sec)
sec.header.is_linked_to_previous = False
sec.even_page_header.is_linked_to_previous = False
sec.footer.is_linked_to_previous = False
hf_text(sec.header, "")
hf_text(sec.even_page_header, "")
hf_text(sec.footer, "")
set_pg_num(sec, fmt="decimal", start=1)

p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Pt(0)
r = p.add_run("中图分类号：")
set_run_fonts(r, size=12)

for _ in range(2):
    doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.first_line_indent = Pt(0)
r = p.add_run("硕士专业学位论文")
set_run_fonts(r, cjk=HEI, latin=ARIAL, size=26, bold=True)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.first_line_indent = Pt(0)
r = p.add_run("Z公司工作流程数字化再造研究")
set_run_fonts(r, cjk=HEI, latin=ARIAL, size=22, bold=True)
doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.first_line_indent = Pt(0)
r = p.add_run(TITLE_EN)
set_run_fonts(r, cjk=TNR, latin=TNR, size=14, bold=True)

for _ in range(3):
    doc.add_paragraph()

COVER = [("论文类型", "专题研究类论文"), ("题目", "Z公司工作流程数字化再造研究"),
         ("专业或领域", "工商管理"), ("研究方向", ""),
         ("专业硕士生", "姚　烨（2024245834）"), ("指导教师", "郭庆方"),
         ("现场导师", "")]
t = doc.add_table(rows=len(COVER), cols=2)
t.alignment = WD_TABLE_ALIGNMENT.CENTER
t.autofit = False
for ri, (k, v) in enumerate(COVER):
    c0, c1 = t.cell(ri, 0), t.cell(ri, 1)
    c0.width = Cm(4.2); c1.width = Cm(8.0)
    for c, txt, bold in ((c0, k, True), (c1, v, False)):
        c.text = ""
        cp = c.paragraphs[0]
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.first_line_indent = Pt(0)
        cp.paragraph_format.space_before = Pt(6)
        cp.paragraph_format.space_after = Pt(6)
        rr = cp.add_run(txt)
        set_run_fonts(rr, size=12, bold=bold)
    # 下划线
    tcPr = c1._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    bt = OxmlElement("w:bottom")
    bt.set(qn("w:val"), "single"); bt.set(qn("w:sz"), "6"); bt.set(qn("w:color"), "000000")
    borders.append(bt)
    tcPr.append(borders)

for _ in range(4):
    doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.first_line_indent = Pt(0)
r = p.add_run("二○二六年十月")
set_run_fonts(r, size=14, bold=True)

# --------------------------------------------------------------- 声明与授权书
sec2 = doc.add_section(WD_SECTION.NEW_PAGE)
page_setup(sec2)
sec2.header.is_linked_to_previous = False
sec2.even_page_header.is_linked_to_previous = False
sec2.footer.is_linked_to_previous = False
hf_text(sec2.header, "")
hf_text(sec2.even_page_header, "")
hf_page(sec2.footer, size=10.5, cjk=SONG, latin=TNR)
set_pg_num(sec2, fmt="upperRoman", start=1)

doc.add_paragraph(style="声明标题").add_run("独 创 性 声 明")
P("本人声明所呈交的学位论文是本人在导师指导下进行的研究工作及取得的研究成果。"
  "除了文中特别加以标注和致谢的地方外，论文中不包含其他人已经发表或撰写过的研究成果，"
  "也不包含为获得中国石油大学（北京）或其他教育机构的学位或证书而使用过的材料。"
  "与我一同工作的同志对本研究所做的任何贡献均已在论文中作了明确的说明并表示谢意。", space_before=12)
for _ in range(3):
    doc.add_paragraph()
P("学位论文作者签名：　　　　　　　　　　签字日期：　　　年　　月　　日")
for _ in range(6):
    doc.add_paragraph()
doc.add_paragraph(style="声明标题").add_run("硕士专业学位论文版权使用授权书")
P("本学位论文作者完全了解中国石油大学（北京）有关保留、使用学位论文的规定，"
  "有权保留并向国家有关部门或机构送交论文的复印件和磁盘，允许论文被查阅和借阅。"
  "本人授权中国石油大学（北京）可以将学位论文的全部或部分内容编入有关数据库进行检索，"
  "可以采用影印、缩印或扫描等复制手段保存、汇编学位论文。"
  "（保密的学位论文在解密后适用本授权书）", space_before=12)
for _ in range(3):
    doc.add_paragraph()
P("学位论文作者签名：　　　　　　　　　　导师签名：")
P("签字日期：　　　年　　月　　日　　　　签字日期：　　　年　　月　　日")

# ------------------------------------------------------------------ 前置内容
# 拆分：摘要 / ABSTRACT / 正文
idx_abs = next(i for i, it in enumerate(ITEMS) if it["t"] == "h1" and it["text"].replace(" ", "") == "摘要")
idx_en = next(i for i, it in enumerate(ITEMS) if it["t"] == "h1" and it["text"] == "ABSTRACT")
idx_body = next(i for i, it in enumerate(ITEMS) if it["t"] == "h1" and it["text"].replace(" ", "") == "引言")
ABS = ITEMS[idx_abs + 1:idx_en]

EN = ITEMS[idx_en + 1:idx_body]
BODY = ITEMS[idx_body:]

# 摘要
doc.add_page_break()
h = doc.add_paragraph(style="标题1")
h.add_run("摘    要")
for it in ABS:
    if it["t"] == "p":
        txt = it["text"]
        if txt.startswith("关键词"):
            P(txt, style="关键词")
        else:
            P(fix_num(txt))

# ABSTRACT
doc.add_page_break()
P(TITLE_EN, style="英文题目")
h = doc.add_paragraph(style="标题1EN")
h.add_run("ABSTRACT")
for it in EN:
    if it["t"] == "p":
        txt = it["text"]
        if txt.lower().startswith("key words"):
            p = doc.add_paragraph(style="正文EN")
            for seg in re.split(r"(Key Words[:：]?)", txt):
                if not seg:
                    continue
                r = p.add_run(seg)
                set_run_fonts(r, cjk=TNR, latin=TNR, size=12,
                              bold=(seg.startswith("Key Words")))
        else:
            P(fix_num(txt), style="正文EN")

# 目录
doc.add_page_break()
h = doc.add_paragraph(style="标题1")
h.add_run("目    录")
p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Pt(0)
_instr = 'TOC \\o "1-3" \\h \\z \\u'
_r = p.add_run()
_fld = OxmlElement("w:fldSimple")
_fld.set(qn("w:instr"), _instr)
_ph = OxmlElement("w:r")
_ph_rPr = OxmlElement("w:rPr")
_ph_color = OxmlElement("w:color"); _ph_color.set(qn("w:val"), "808080")
_ph_ital = OxmlElement("w:i")
_ph_rPr.append(_ph_color); _ph_rPr.append(_ph_ital)
_ph.append(_ph_rPr)
_ph_t = OxmlElement("w:t")
_ph_t.text = "（目录为自动域：请在 Word 中右键此处 → 更新域 → 更新整个目录）"
_ph.append(_ph_t)
_fld.append(_ph)
_r._r.addnext(_fld)

# ------------------------------------------------------------------ 正文
first_sec = True
for it in BODY:
    if it["t"] == "h1":
        sec = doc.add_section(WD_SECTION.NEW_PAGE)
        page_setup(sec)
        sec.header.is_linked_to_previous = False
        sec.even_page_header.is_linked_to_previous = False
        sec.footer.is_linked_to_previous = False
        hf_text(sec.header, "硕士专业学位论文")
        title = norm_h1(it["text"])
        hf_text(sec.even_page_header, title.replace("  ", " "))
        hf_page(sec.footer, size=10.5, cjk=SONG, latin=ARIAL)
        if first_sec:
            set_pg_num(sec, fmt="decimal", start=1)
            first_sec = False
        else:
            pg = sec._sectPr.find(qn("w:pgNumType"))
            if pg is not None:
                sec._sectPr.remove(pg)
        h = doc.add_paragraph(style="标题1")
        h.add_run(title)
    elif it["t"] == "h2":
        doc.add_paragraph(norm_h23(it["text"]), style="标题2")
    elif it["t"] == "h3":
        doc.add_paragraph(norm_h23(it["text"]), style="标题3")
    elif it["t"] == "h4":
        p = docs_h4 = doc.add_paragraph(style="正文")
        r = p.add_run(it["text"])
        set_run_fonts(r, cjk=HEI, latin=ARIAL, size=12, bold=True)
    elif it["t"] == "p":
        P(fix_num(it["text"]))
    elif it["t"] == "fig":
        cap = fix_num(it["caption"] or "")
        m = re.match(r"图\s*([\d.]+)", cap)
        key = m.group(1).replace(".", "-") if m else None
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(BytesIO(it["blob"]), width=Cm(FIG_W_CM))
        pc = doc.add_paragraph(style="图题")
        pc.paragraph_format.keep_with_next = True
        pc.add_run(cap)
        if key and key in FIG_EN:
            pe = doc.add_paragraph(style="图题EN")
            pe.paragraph_format.keep_with_next = True
            pe.add_run(f"Fig. {key.replace('-', '.')} {FIG_EN[key]}")
        if key and key in FIG_EN:
            pass
        if it["note"]:
            P(it["note"], style="图注")
    elif it["t"] == "table":
        cap = fix_num(it["caption"] or "")
        m = re.match(r"表\s*([\d.]+)", cap)
        key = m.group(1).replace(".", "-") if m else None
        pc = doc.add_paragraph(style="表题")
        pc.paragraph_format.keep_with_next = True
        pc.add_run(cap)
        if key and key in TAB_EN:
            pe = doc.add_paragraph(style="图题EN")
            pe.paragraph_format.keep_with_next = True
            pe.add_run(f"Table {key.replace('-', '.')} {TAB_EN[key]}")
        rows = it["rows"]
        nrow, ncol = len(rows), len(rows[0])
        tb = doc.add_table(rows=nrow, cols=ncol)
        tb.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = tb._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "none")
            borders.append(e)
        ins(tblPr, borders, "tblPr")
        # 表头重复
        trPr0 = tb.rows[0]._tr.get_or_add_trPr()
        th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true")
        ins(trPr0, th, "trPr")
        for ri, row in enumerate(rows):
            tr = tb.rows[ri]
            trPr = tr._tr.get_or_add_trPr()
            trh = OxmlElement("w:trHeight")
            trh.set(qn("w:val"), "448"); trh.set(qn("w:hRule"), "atLeast")
            cs = OxmlElement("w:cantSplit")
            ins(trPr, cs, "trPr")
            ins(trPr, trh, "trPr")
            for ci in range(ncol):
                cell = tb.cell(ri, ci)
                cell.text = ""
                cp = cell.paragraphs[0]
                cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cp.paragraph_format.first_line_indent = Pt(0)
                cp.paragraph_format.space_before = Pt(1)
                cp.paragraph_format.space_after = Pt(1)
                cp.paragraph_format.line_spacing = 1.25
                rr = cp.add_run(row[ci] if ci < len(row) else "")
                set_run_fonts(rr, size=10.5, bold=(ri == 0))
                tcPr = cell._tc.get_or_add_tcPr()
                cb = OxmlElement("w:tcBorders")
                if ri == 0:
                    for edge in ("top", "bottom"):
                        e = OxmlElement(f"w:{edge}")
                        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000")
                        cb.append(e)
                if ri == nrow - 1:
                    e = OxmlElement("w:bottom")
                    e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000")
                    cb.append(e)
                ins(tcPr, cb, "tcPr")
        if it["note"]:
            P(it["note"], style="图注")

doc.save("v12/v1.5_学校格式版.docx")
print("saved v12/v1.5_学校格式版.docx")
