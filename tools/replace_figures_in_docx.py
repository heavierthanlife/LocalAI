# -*- coding: utf-8 -*-
"""在作者已手工修改的 v1.2 docx 中，仅替换 16 张图件（保留正文与用户排版改动）。"""
import os, sys, hashlib, glob
import docx
from docx.shared import Emu
from docx.image.image import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else None
DST = sys.argv[2] if len(sys.argv) > 2 else None
OLD_DIR = "build/figures_final"
NEW_DIR = "build/figures_v14"

def md5(b): return hashlib.md5(b).hexdigest()

oldmap = {md5(open(f,'rb').read()): os.path.basename(f) for f in glob.glob(OLD_DIR+'/*.png')}
newmap = {os.path.basename(f): f for f in glob.glob(NEW_DIR+'/*.png')}

doc = docx.Document(SRC)
replaced, skipped = [], []
for i, sh in enumerate(doc.inline_shapes):
    rId = sh._inline.graphic.graphicData.pic.blipFill.blip.embed
    part = doc.part.related_parts[rId]
    h = md5(part.blob)
    name = oldmap.get(h)
    if not name or name not in newmap:
        skipped.append((i, name or 'UNKNOWN')); continue
    data = open(newmap[name], 'rb').read()
    img = Image.from_blob(data)
    part._blob = data
    # 保持作者设定的宽度（上限 15.0cm），高度按新图纵横比
    w = sh.width
    if w.cm > 15.0:
        w = Emu(int(15.0 * 360000))  # 15cm in EMU
    hgt = int(w * img.px_height / img.px_width)
    # 高度上限 21.5cm，超出则按高度反推宽度
    maxh = int(21.5 * 360000)
    if hgt > maxh:
        hgt = maxh
        w = Emu(int(hgt * img.px_width / img.px_height))
    sh.width = w
    sh.height = Emu(hgt)
    replaced.append((i, name, round(w.cm,1), round(hgt/360000,1)))

doc.save(DST)
print(f"replaced {len(replaced)} images, skipped {len(skipped)}")
for r in replaced: print(" ", r)
if skipped: print("SKIPPED:", skipped)
