"""Paragraph-level collusion detection — near-verbatim SUBSTANTIVE segments.

Why this exists (FIX-2026-09-04-QA-B1):
  Whole-document TF-IDF cosine is a "signal averager": in a 50k-char bid doc,
  a few hundred chars of verbatim-copied service commitment (e.g. the "热情、
  主动、耐心…" service-attitude paragraph shared by colluding bidders) adds ~1%
  cosine while template/industry overlap adds ~85%.  The real collusion evidence
  is *cross-file near-verbatim paragraphs whose wording is distinctive*, i.e.
  paragraphs that are NOT generic industry/template/legal boilerplate.

This detector:
  1. splits each raw document (newlines preserved) into paragraphs,
  2. finds near-verbatim paragraph pairs (SequenceMatcher ratio >= threshold),
  3. scores each shared segment's "surprise" = share of tokens that survive
     removal of industry words + generic stop words + legal boilerplate,
  4. only shared segments with surprise >= SURPRISE_THRESHOLD count as a
     collusion signal (service commitments / technical specifics), while
     standard clauses (应当/必须/不得…) are discounted.

Reuses the vectorizer/tokenizer invariants (FIX-015: tokenizer returns list,
stop-words filtered inside) and the industry word tables (data/industry_words/).
"""
from __future__ import annotations

import difflib
import logging

logger = logging.getLogger(__name__)

from app.services.industry_words import get_procurement_type, load_industry_stopwords
from app.services.stop_words import DEFAULT_STOP_WORDS

# ── 阈值（由回归测试锁定）──────────────────────────────────────
MIN_PARA_CHARS = 30          # 段落最短有效长度（太短无统计意义）
NEAR_DUPLICATE_RATIO = 0.85  # 两段 SequenceMatcher 匹配率 ≥ 85% 视为 near-verbatim
SURPRISE_THRESHOLD = 0.35    # 段内"非行业/非模板词"占比 ≥ 35% 才算实质内容
LEN_BAND = 2.2               # 长度比过滤：lenA/lenB 必须落在 [1/LEN_BAND, LEN_BAND]
MAX_PARA_CHARS = 4000        # 忽略超长段（多为表格/清单）
MAX_SEGMENTS = 200           # 结果段上限（防报告爆炸）

# ── 法律/模板句式词（标准条款段用，计入"惊讶度"折扣）──────────
_LEGAL_TERMS = frozenset({
    "应当", "必须", "不得", "可以", "能够", "应该", "需要", "确保", "保证",
    "按照", "根据", "依照", "依据", "符合", "满足", "具备", "遵守", "违反",
    "法律", "法规", "规章", "规定", "条款", "办法", "细则", "办法", "标准",
    "招标文件", "投标文件", "招标人", "投标人", "采购人", "供应商", "甲方",
    "乙方", "双方", "当事人", "责任", "义务", "权利", "要求", "提交", "提供",
    "签订", "履行", "执行", "承担", "招标", "投标", "项目", "单位", "文件",
    "合同", "报价", "金额", "工期", "质量", "安全", "资质", "证书",
    # ── 模板/格式高频（服务承诺/通用承诺的样板措辞）──
    "服务", "承诺", "态度", "水平", "认真", "优质", "国家", "本", "我", "我们",
    "贵", "部", "部队", "官兵", "家属", "需求", "顾客", "客人", "方便", "提供",
    "工作", "进行", "有关", "相关", "要求", "内容", "情况", "问题", "管理",
    "健全", "完善", "制度", "机制", "体系", "全面", "有效", "积极", "主动",
    "及时", "充分", "切实", "不断", "持续", "开展", "组织", "实施", "落实",
    "确保", "达到", "符合", "满足", "职责", "各项", "所有", "任何", "有关",
    "重要", "主要", "基本", "相应", "必要", "能够", "可以", "应当", "不得",
    "本文件", "本项目", "本服务", "合同要求", "招标要求", "服务标准", "服务内容",
    "服务质量", "服务对象", "服务承诺", "服务期限", "服务地点", "服务费用",
})

# ── 段类型标记（报告标注用）───────────────────────────────────
_SVC_MARKERS = ("服务承诺", "服务态度", "热情", "周到", "耐心", "细致", "尽职",
                "尽责", "宾至如归", "微笑", "真诚", "亲切", "贴心", "主动服务",
                "顾客", "客户", "响应", "售后", "承诺", "态度", "满意度")
_TECH_MARKERS = ("技术方案", "施工方案", "施工组织", "工艺", "措施", "方法", "方案",
                 "设计", "结构", "材料", "设备选型", "性能", "参数", "调试", "安装",
                 "施工", "工序", "流程", "系统组成", "功能")
_LEGAL_MARKERS = ("应当", "必须", "不得", "可以", "按照", "根据", "依照", "依据",
                  "本招标文件", "本投标文件", "第", "条", "法律法规", "本办法",
                  "本次招标", "资格审查")


def _split_paragraphs(text: str):
    """Return non-empty paragraphs (len in [MIN_PARA_CHARS, MAX_PARA_CHARS])."""
    return [p.strip() for p in (text or "").split("\n")
            if MIN_PARA_CHARS <= len(p.strip()) <= MAX_PARA_CHARS]


def _classify_paragraph(para: str) -> str:
    """服务承诺段 / 技术方案段 / 标准条款段 / 其他."""
    s = para[:600]
    svc = sum(1 for m in _SVC_MARKERS if m in s)
    tech = sum(1 for m in _TECH_MARKERS if m in s)
    legal = sum(1 for m in _LEGAL_MARKERS if m in s)
    if svc >= 1 and svc >= tech:
        return "服务承诺段"
    if tech >= 2 and tech > legal:
        return "技术方案段"
    if legal >= 3:
        return "标准条款段"
    if tech >= 1:
        return "技术方案段"
    if svc >= 1:
        return "服务承诺段"
    return "其他"


def _surprise(para: str, stop: frozenset) -> float:
    """Fraction of jieba tokens that survive stop-word removal.

    High surprise = distinctive wording (collusion-relevant).  Low surprise =
    generic industry/template/legal boilerplate (not collusion evidence).
    Returns (fraction, distinct_token_count).
    """
    from app.services.text_utils import lcut
    merged = set(stop) | set(DEFAULT_STOP_WORDS) | set(_LEGAL_TERMS)
    toks = [w for w in lcut(para) if len(w) >= 2 and w not in merged]
    total = len([w for w in lcut(para) if len(w) >= 2])
    if total == 0:
        return 0.0, 0
    return round(len(toks) / total, 3), len(toks)


def _candidates(plen: int, buckets: dict[int, list[tuple[int, int]]],
                lo: float, hi: float) -> list[tuple[int, int]]:
    """Paragraphs in other files whose length falls in the same band (pre-filter)."""
    out = []
    for b in range(int(lo) // 15, int(hi) // 15 + 1):
        out.extend(buckets.get(b, []))
    return out


def detect_shared_substantive_segments(file_data: list[dict], ptype: str | None = None) -> dict:
    """Find cross-file near-verbatim paragraphs that carry substantive (non-boilerplate)
    content.  Returns the collusion evidence list + per-pair stats + a 0-100 score.

    file_data: [{filename, text, ...}] — `text` is RAW (newlines preserved).
    """
    n = len(file_data)
    if n < 2:
        return {"shared_segments": [], "per_pair_count": {}, "per_pair_ratio": {},
                "collusion_score": 0.0, "ptype": ptype or "engineering", "n_files": n}

    if ptype is None:
        ptype = get_procurement_type(*[fd.get("text", "") for fd in file_data])
    stop = frozenset(load_industry_stopwords(ptype))

    # 1) 每文件分段 + 预计算（保留 >2000 字长段给表格，其余按段长入桶）
    docs: list[tuple[str, list[tuple[str, str, float, int]]]] = []
    for fi, fd in enumerate(file_data):
        paras = _split_paragraphs(fd.get("text", ""))
        docs.append((fd.get("filename", f"file{fi}"), [
            (p, _classify_paragraph(p), surp, dcnt) for p in paras
            for surp, dcnt in [_surprise(p, stop)]
        ]))

    # 2) 段长桶索引 + 每段字符 bigram 指纹（哈希剪枝，避免全量 SequenceMatcher）
    buckets: dict[int, list[tuple[int, int]]] = {}
    for di, (_f, plist) in enumerate(docs):
        for pi, (p, _c, _s, _d) in enumerate(plist):
            buckets.setdefault(len(p) // 15, []).append((di, pi))

    def _fingerprint(s: str) -> frozenset:
        return frozenset(s[i:i + 2] for i in range(len(s) - 1))

    fp_cache: dict[tuple[int, int], frozenset] = {}
    for di, (_f, plist) in enumerate(docs):
        for pi, (p, _c, _s, _d) in enumerate(plist):
            fp_cache[(di, pi)] = _fingerprint(p)

    # 3) 两两 near-verbatim 段匹配（长度比带内 + bigram 指纹交集剪枝）
    segments: dict[str, dict] = {}
    seg_order: list[str] = []
    for di in range(n):
        plist = docs[di][1]
        for pi, (para, cls, surp, dcnt) in enumerate(plist):
            plen = len(para)
            lo, hi = plen / LEN_BAND, plen * LEN_BAND
            cand = _candidates(plen, buckets, lo, hi)
            for dj, pj in cand:
                if dj <= di:
                    continue
                other, ocls, osurp, odcnt = docs[dj][1][pj]
                if abs(len(other) - plen) / max(plen, len(other)) > 0.5:
                    continue
                if min(surp, osurp) < SURPRISE_THRESHOLD or min(dcnt, odcnt) < 2:
                    continue
                # bigram 指纹交集：无共同 2-gram 则不可能 near-verbatim
                if not (fp_cache.get((di, pi), frozenset()) & fp_cache.get((dj, pj), frozenset())):
                    continue
                ratio = difflib.SequenceMatcher(None, para, other).ratio()
                if ratio < NEAR_DUPLICATE_RATIO:
                    continue
                key = f"{di}:{pi}|{dj}:{pj}"
                seg = {
                    "segment_text": para[:200],
                    "char_len": plen,
                    "type": cls,
                    "surprise": round(min(surp, osurp), 3),
                    "match_ratio": round(ratio, 3),
                    "matching_files": [(docs[di][0], docs[dj][0], round(ratio, 3))],
                    "files": {docs[di][0], docs[dj][0]},
                }
                segments[key] = seg
                seg_order.append(key)

    # 4) 合并"同一实质段"记录（去重同文件对 + 汇集 ≥3 家共享）
    merged: list[dict] = []
    merged_key: dict[frozenset, int] = {}
    for key in seg_order:
        seg = segments[key]
        fk = frozenset(seg["files"])
        if fk in merged_key:
            idx = merged_key[fk]
            if seg["match_ratio"] > merged[idx]["match_ratio"]:
                merged[idx]["match_ratio"] = seg["match_ratio"]
            continue
        merged_key[fk] = len(merged)
        seg["n_files"] = len(fk)
        seg["files"] = sorted(fk)
        merged.append(seg)

    # 5) 每对统计
    per_pair_count: dict[tuple[int, int], int] = {}
    for seg in merged:
        fs = [i for i, fd in enumerate(file_data) if fd["filename"] in seg["files"]]
        for a in range(len(fs)):
            for b in range(a + 1, len(fs)):
                k = (min(fs[a], fs[b]), max(fs[a], fs[b]))
                per_pair_count[k] = per_pair_count.get(k, 0) + 1

    per_pair_ratio = {}
    for (i, j), cnt in per_pair_count.items():
        tot = min(len(docs[i][1]), len(docs[j][1]))
        per_pair_ratio[(i, j)] = round(cnt / max(tot, 1), 4)

    # 6) 围标风险分（0-100）：数量 + 多文件一致性 + 平均惊讶度
    score = 0.0
    if merged:
        cnt = min(len(merged), 5)
        n_files3 = sum(1 for s in merged if s["n_files"] >= 3)
        avg_surp = sum(s["surprise"] for s in merged) / len(merged)
        score = min(100, cnt * 15 + n_files3 * 20 + avg_surp * 20)

    return {
        "shared_segments": merged[:MAX_SEGMENTS],
        "per_pair_count": {f"({i},{j})": c for (i, j), c in per_pair_count.items()},
        "per_pair_ratio": {f"({i},{j})": r for (i, j), r in per_pair_ratio.items()},
        "collusion_score": round(score, 1),
        "ptype": ptype,
        "n_files": n,
    }
