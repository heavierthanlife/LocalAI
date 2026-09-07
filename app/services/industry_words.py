"""Industry common-word tables (工程/货物/服务) for collusion detection.

Purpose: in the same-industry tender, generic industry vocabulary (设备/施工/服务/
顾客/商品…) overlaps between any two bidders — it must NOT count as collusion
evidence. These tables discount such generic words in:
  - text/keyword similarity stop-words merging
  - paragraph "surprise" scoring (near-verbatim substantive segments)
  - cross-file shared-typo guard (domain word = legitimate usage)

Data files: data/industry_words/{engineering,goods,services}.txt
  - two layers per file: `# === 国家标准词 ===` (official classification from
    财政部《政府采购品目分类目录(2022年版)》/ 住建部《建筑业企业资质标准》)
    and `# === 常见运营词 ===` (curated operational words, extensible).
  - GUARDRAIL: evaluative / distinctive phrasing (热情/周到/细致/尽职/宾至如归/
    微笑) must never be added — locked by tests.test_regression.

Types: "engineering" | "goods" | "services". Detection via dominant keyword hits,
fallback "engineering" (the calibrated baseline).
"""
import os
from functools import lru_cache

from app.services.stop_words import DEFAULT_STOP_WORDS

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "industry_words")

LAYER_NATIONAL = "national"
LAYER_OPERATIONAL = "operational"

# ── 类型探测：每类一小撮"强区分词"（独立于词表，避免表间重叠干扰判定）──
TYPE_KEYWORDS = {
    "engineering": {
        "施工", "工程", "建设", "安装", "装修", "修缮", "构筑物", "地基", "主体结构",
        "钢结构", "桩基", "开挖", "浇筑", "塔吊", "脚手架", "工法", "竣工", "验收",
        "施工总承包", "专业承包", "安全生产许可证", "项目经理", "施工组织设计", "工地",
    },
    "goods": {
        "货物", "设备", "物资", "采购", "供货", "交付", "质保", "耗材", "配件",
        "库存", "条形码", "装箱单", "产品合格证", "技术参数", "型号", "规格", "易耗品",
        "备品备件", "办公用品", "家具", "图书", "仪表", "器材",
    },
    "services": {
        "服务", "运维", "培训", "咨询", "监理", "保洁", "物业", "保安", "餐饮",
        "会议", "展览", "住宿", "运输", "仓储", "物流", "客服", "超市", "门店",
        "收银", "陈列", "理货", "会员", "响应时间", "售后服务",
    },
}


def _load_raw(ptype: str) -> str:
    path = os.path.join(_DIR, f"{ptype}.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _parse(text: str):
    """Split the two layers, dropping comment lines and blanks."""
    layers = {LAYER_NATIONAL: set(), LAYER_OPERATIONAL: set()}
    current = LAYER_NATIONAL
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            if line == "# === 国家标准词 ===":
                current = LAYER_NATIONAL
            elif line == "# === 常见运营词 ===":
                current = LAYER_OPERATIONAL
            continue
        # 词条含空格视为多词组合，按整体保留（分词后会命中）——先按词条整体，同时拆成短词
        layers[current].add(line)
        if len(line) <= 4:
            layers[current].add(line)
    return layers


@lru_cache(maxsize=8)
def load_industry_words(ptype: str) -> dict:
    """Return {national: set, operational: set} for a procurement type."""
    return _parse(_load_raw(ptype))


def load_industry_stopwords(ptype: str) -> frozenset:
    """Merged set of generic industry words (both layers) for similarity discount."""
    layers = load_industry_words(ptype)
    return frozenset(layers[LAYER_NATIONAL] | layers[LAYER_OPERATIONAL])


def merged_stop_words(ptype: str, extra: frozenset | None = None) -> frozenset:
    """DEFAULT_STOP_WORDS + industry words (+ optional caller extras)."""
    s = set(DEFAULT_STOP_WORDS)
    s |= load_industry_stopwords(ptype)
    if extra:
        s |= set(extra)
    return frozenset(s)


def get_procurement_type(*texts: str) -> str:
    """Detect dominant procurement type by keyword hits.

    Returns "engineering" | "goods" | "services". Ambiguous/unknown → "engineering"
    (the calibrated baseline; safest fallback per design decision).
    """
    combined = " ".join(t or "" for t in texts)[:200000]
    if not combined.strip():
        return "engineering"
    scores = {t: 0 for t in TYPE_KEYWORDS}
    for t, kws in TYPE_KEYWORDS.items():
        scores[t] = sum(1 for kw in kws if kw in combined)
    # dominant with clear margin; ties → engineering fallback
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, second = ranked[0], ranked[1]
    if top[1] > 0 and top[1] > second[1]:
        return top[0]
    if top[1] > 0 and top[1] == second[1]:
        return "engineering"
    return "engineering"


def guardrail_check() -> dict:
    """Confirm no evaluative/distinctive phrasing leaked into any table."""
    forbidden = {"热情", "周到", "细致", "尽职", "尽责", "宾至如归", "微笑", "亲切", "真诚", "殷切"}
    found = {}
    for ptype in ("engineering", "goods", "services"):
        layers = load_industry_words(ptype)
        hit = sorted(w for w in (layers[LAYER_NATIONAL] | layers[LAYER_OPERATIONAL]) if any(f in w for f in forbidden))
        found[ptype] = hit
    return found
