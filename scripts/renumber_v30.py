# -*- coding: utf-8 -*-
"""v3.0 第二步：按新章节首现顺序重排表/图编号与参考文献编号。"""
import re, docx
d = docx.Document("v12/v3.0_draft.docx")
ps = d.paragraphs
ch = 0
tmap, fmap = {}, {}
tseq = fseq = 0
prev_tbl_new = None
for p in ps:
    t = p.text.strip()
    if p.style.name == "标题1":
        m = re.match(r"第(\d)章", t)
        if m: ch = int(m.group(1)); tseq = fseq = 0
    if p.style.name == "表题":
        m = re.match(r"表(\d+)\.(\d+)", t)
        if m:
            tseq += 1; old = f"{m.group(1)}.{m.group(2)}"; new = f"{ch}.{tseq}"
            tmap[old] = new; prev_tbl_new = new
    elif p.style.name == "图题":
        m = re.match(r"图(\d+)\.(\d+)", t)
        if m:
            fseq += 1; old = f"{m.group(1)}.{m.group(2)}"; new = f"{ch}.{fseq}"
            fmap[old] = new
# 题注与引用替换
for p in ps:
    for r in p.runs:
        s = r.text
        s = re.sub(r"表(\d+)\.(\d+)", lambda m: "表" + tmap.get(f"{m.group(1)}.{m.group(2)}", f"{m.group(1)}.{m.group(2)}"), s)
        s = re.sub(r"图(\d+)\.(\d+)", lambda m: "图" + fmap.get(f"{m.group(1)}.{m.group(2)}", f"{m.group(1)}.{m.group(2)}"), s)
        if s != r.text: r.text = s
# 英文题注：按顺序跟随中文题注
tbl_pairs = []; fig_pairs = []
cur = None
for p in ps:
    if p.style.name == "表题": cur = ("t", re.match(r"表([\d.]+)", p.text.strip()).group(1))
    elif p.style.name == "图题": cur = ("f", re.match(r"图([\d.]+)", p.text.strip()).group(1))
    elif p.style.name == "图题EN":
        m = re.search(r"(Table|Fig\.)\s*([\d.]+)", p.text)
        if m and cur:
            new = (tmap if m.group(1) == "Table" else fmap).get(m.group(2))
            if new:
                for r in p.runs:
                    if not r.text: continue
                    r.text = re.sub(r"(Table|Fig\.)\s*[\d.]+", lambda mm: mm.group(1) + " " + new, r.text)
# 参考文献重排
body_txt = []
in_refs = False
cites = []
for p in ps:
    if p.style.name == "标题1" and "参 考 文 献" in p.text: in_refs = True; continue
    if in_refs:
        if p.style.name == "标题1": break
        m = re.match(r"\[(\d+)\]", p.text.strip())
        if m: continue
    for m in re.finditer(r"\[([\d,\-–]+)\]", p.text):
        for part in re.split(r"[,，]", m.group(1)):
            part = part.strip().replace("–", "-")
            if re.fullmatch(r"\d+", part): cites.append(int(part))
            elif re.fullmatch(r"\d+-\d+", part):
                a, b = part.split("-"); cites += list(range(int(a), int(b)+1))
cmap = {}
for n in cites:
    if n not in cmap: cmap[n] = len(cmap) + 1
for p in ps:
    for r in p.runs:
        if not r.text: continue
        def rep(m):
            parts = re.split(r"([,，])", m.group(1)); out = []
            for x in parts:
                if re.fullmatch(r"[\d\-–]+", x.strip()):
                    x = x.strip().replace("–", "-")
                    if re.fullmatch(r"\d+-\d+", x):
                        a, b = x.split("-")
                        if len(a) > 2 or len(b) > 2: out.append(x)
                        else: out.append(f"{cmap.get(int(a), int(a))}-{cmap.get(int(b), int(b))}")
                    else:
                        out.append(str(cmap.get(int(x), int(x))) if len(x) <= 2 else x)
                else: out.append(x)
            return "[" + "".join(out) + "]"
        r.text = re.sub(r"\[([\d,\-–]+)\]", rep, r.text)
# 参考文献条目按新号排序
ref_ps = []
in_refs = False
for p in ps:
    if p.style.name == "标题1" and "参 考 文 献" in p.text: in_refs = True; continue
    if in_refs:
        if p.style.name == "标题1": break
        if re.match(r"\[\d+\]", p.text.strip()): ref_ps.append(p)
newrefs = sorted(ref_ps, key=lambda p: int(re.match(r"\[(\d+)\]", p.text.strip()).group(1)))
if ref_ps:
    anchor = ref_ps[0]._p
    for p in newrefs:
        p._p.getparent().remove(p._p)
    for p in newrefs:
        anchor.addprevious(p._p)
d.save("v12/v3.0_结构重构版.docx")
print("tables map", len(tmap), "figs map", len(fmap), "cites", len(cmap))
print("v3.0 saved")
