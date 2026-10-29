# -*- coding: utf-8 -*-
"""按《中国石油大学（北京）硕士专业学位论文写作规范》（格式.doc）重排 v1.3 论文。

关键规则（摘自 格式.doc）：
- A4，上下左右页边距 3.0 cm，页眉/页脚距边 2.0 cm
- 正文：中文宋体 + 英文数字 Times New Roman，小四(12pt)，多倍行距 1.25，段前后 0，首行缩进 2 字
- 标题1（章）：黑体/Arial，小三(15pt)，居中，1.5 倍行距，段前 0，段后 11 磅；每章另起一节
- 标题2（节）：黑体/Arial，四号(14pt)，居左，1.5 倍行距，段前 0.5 行，段后 0；标题号与标题空两格
- 标题3：黑体/Arial，小四(12pt)，居左，1.5 倍行距，段前 0.5 行，段后 0；空两格
- 图题：图下方，黑体五号居中；英文图题 Times New Roman 五号粗体居中
- 表题：表上方，黑体五号居中；英文表题 Times New Roman 五号粗体居中；三线表，0.5 磅线，表内五号
- 参考文献：五号，居左，1.25 倍行距，续行缩进
- 前置部分罗马数字页码，正文从引言起阿拉伯数字第 1 页；页脚 Arial 五号居中
- 页眉：奇数页“硕士专业学位论文”，偶数页“章节标题”，黑体五号居中
- 图/表编号：分章编号，用点号（图2.1、表3.2）
"""
import os
import re
import struct
import docx
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_BREAK
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph

SRC = "v12/v1.3_收尾修订版.docx"
DST = "v12/v1.4_学校格式版.docx"

SONG, HEI, TNR, ARIAL = "宋体", "黑体", "Times New Roman", "Arial"

TITLE_EN = "Research on the Digital Re-engineering of Workflows in Company Z"

FIG_EN = {
    "1-1": "Research technical roadmap",
    "2-1": "Integrated analytical framework of BPR, TOE and dynamic capability",
    "2-2": "Mapping matrix of TOE dimensions and dynamic capability across problems, causes, countermeasures and indicators",
    "3-1": "Six-stage workflow of tendering agency and its bottleneck stages",
    "3-2": "Structure of the questionnaire sample (N=185)",
    "3-3": "Utilization rate, integration and satisfaction of digital tools",
    "3-4": "Quantitative characterization of the three core problems",
    "4-1": "Fishbone diagram of the causes of workflow problems",
    "4-2": "Transmission chain of problems, causes and countermeasures",
    "5-1": "Overall technical architecture of the digital work-activity platform",
    "5-2": "Process data model and computable process indicators",
    "6-1": "Framework of the re-engineering effect evaluation indicator system",
    "6-2": "Average rework rounds by business type (measured after go-live)",
    "6-3": "Comparison of key indicators before and after re-engineering (measured)",
    "6-4": "Indicator changes and composite process index under three scenarios",
    "6-5": "Comparison of four key indicators before and after re-engineering (AS-IS=100)",
}
TAB_EN = {
    "3-1": "Gender distribution of the sample",
    "3-2": "Age distribution of the sample",
    "3-3": "Educational background of the sample",
    "3-4": "Distribution of working years",
    "3-5": "Distribution of position levels",
    "3-6": "Distribution of departments",
    "3-7": "Regional distribution of the sample",
    "3-8": "Application of digital tools",
    "3-9": "Key indicators of process and resource efficiency",
    "3-10": "Key indicators of human resources and incentives",
    "5-1": "Main technology stack of the platform",
    "5-2": "Key functional modules and their implementation",
    "6-1": "Evaluation indicators and measurement criteria",
    "6-2": "Comparison of key indicators before and after re-engineering",
    "6-3": "Scenario parameter settings (change rate relative to baseline)",
}

# ---------------------------------------------------------------- 读取源文档
src = Document(SRC)
body = src.element.body
items = []          # 结构化内容
pending_caption = None
i = 0
children = list(body.iterchildren())
from docx.table import Table
from docx.text.paragraph import Paragraph as P


def p_of(el):
    return P(el, src)


def table_of(el):
    return Table(el, src)


imgs = 0
tb = 0
paras = 0
while i < len(children):
    el = children[i]
    if el.tag == qn("w:p"):
        p = p_of(el)
        style = p.style.name
        text = p.text.strip()
        has_img = bool(el.findall(".//" + qn("a:blip")))
        if has_img:
            blip = el.find(".//" + qn("a:blip"))
            rid = blip.get(qn("r:embed"))
            blob = src.part.related_parts[rid].blob
            # 找图题
            cap, note = None, None
            j = i + 1
            while j < len(children) and children[j].tag == qn("w:p"):
                if children[j].findall(".//" + qn("a:blip")):
                    break
                t = p_of(children[j]).text.strip()
                if not t:
                    j += 1
                    continue
                if re.match(r"^图\s*\d+", t) and cap is None:
                    cap = t
                elif (t.startswith("注：") or t.startswith("数据来源")) and note is None:
                    note = t
                else:
                    break
                j += 1
            items.append({"t": "fig", "blob": blob, "caption": cap, "note": note})
            imgs += 1
            i = j
            continue
        if style.startswith("toc"):
            i += 1
            continue
        if not text:
            i += 1
            continue
        if re.match(r"^表\s*\d+", text):
            # 表题 → 下一元素为表格
            cap = text
            j = i + 1
            while j < len(children) and children[j].tag == qn("w:p") and not p_of(children[j]).text.strip():
                j += 1
            if j < len(children) and children[j].tag == qn("w:tbl"):
                t = table_of(children[j])
                rows = [[c.text.strip() for c in r.cells] for r in t.rows]
                note = None
                k = j + 1
                while k < len(children) and children[k].tag == qn("w:p"):
                    if children[k].findall(".//" + qn("a:blip")):
                        break
                    tt = p_of(children[k]).text.strip()
                    if tt.startswith("注：") or tt.startswith("数据来源"):
                        note = tt
                        k += 1
                    elif not tt:
                        k += 1
                    else:
                        break
                items.append({"t": "table", "caption": cap, "rows": rows, "note": note})
                tb += 1
                i = k
                continue
        if style in ("Heading 1", "Heading 2", "Heading 3", "Heading 4"):
            lvl = int(style[-1])
            items.append({"t": f"h{lvl}", "text": text})
        else:
            items.append({"t": "p", "text": text})
            paras += 1
    i += 1

print(f"解析完成：图片 {imgs}，表格 {tb}，正文段 {paras}，总条目 {len(items)}")

# 丢掉封面（第一条 h1 之前的条目；源文档首条 h1 为“目    录”）
k0 = next(k for k, it in enumerate(items) if it["t"] == "h1" and it["text"].replace(" ", "") == "目录")
items = items[k0 + 1:]
print("去封面/目录后条目:", len(items))
os.makedirs("v12", exist_ok=True)
import pickle
pickle.dump({"items": items, "title_en": TITLE_EN, "fig_en": FIG_EN, "tab_en": TAB_EN},
            open("v12/content_v13.pkl", "wb"))
print("已保存结构化内容 v12/content_v13.pkl")
