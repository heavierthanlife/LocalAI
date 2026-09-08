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
MAX_PARAS_PER_FILE = 2500    # 单文件段落数上限（超限按步长抽样，防 O(n²) 爆炸）
MAX_CAND_PER_PARA = 40       # 每段候选对上限（按共享锚数降序截断，锁定复杂度 O(段数×K)）
ANCHOR_WIN = 12              # 锚子串窗口长（近逐字段必共享长连续子串）
ANCHOR_MIN_SHARE = 1         # 候选对需共享 ≥N 个锚子串（1 即可，配合长度/surprise 预筛控量）

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
    # ── 投标函/声明模板措辞（FIX-2026-09-07-QA-C3：防止声明段被当实质雷同）──
    "正本", "副本", "电子版", "份", "有效期", "截止日", "之日起", "日内",
    "自愿", "接受", "执行", "全部条款", "仔细研究", "相关材料", "声明",
    "真实有效", "不实", "后果", "承担", "待命", "供应", "腐烂", "变质",
    "过期", "伪劣", "指定地点", "响应", "交货", "确认函", "授权书", "承诺书",
    "资格证明", "我方", "本单位", "本项目", "充分理解", "同意", "条款", "投标",
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
    """Return non-empty paragraphs (len in [MIN_PARA_CHARS, MAX_PARA_CHARS]).

    FIX-2026-09-07-QA-C2: cap paragraph count per file (MAX_PARAS_PER_FILE);
    when exceeded, keep a length-stratified sample so giant docs (tens of
    thousands of table rows) don't blow up the pairwise matcher.
    """
    paras = [p.strip() for p in (text or "").split("\n")
             if MIN_PARA_CHARS <= len(p.strip()) <= MAX_PARA_CHARS]
    if len(paras) <= MAX_PARAS_PER_FILE:
        return paras
    # 按段长分层抽样：保持长短段比例，避免只留某一段长
    import math
    buckets: dict[int, list[str]] = {}
    for p in paras:
        buckets.setdefault(len(p) // 50, []).append(p)
    total = len(paras)
    keep = []
    quota = MAX_PARAS_PER_FILE
    # 逐桶按占比分配名额
    for b in sorted(buckets):
        frac = len(buckets[b]) / total
        n_keep = max(1, int(quota * frac))
        step = max(1, len(buckets[b]) // max(n_keep, 1))
        keep.extend(buckets[b][::step][:n_keep])
    if len(keep) < MIN_PARA_CHARS:
        keep = paras[:MAX_PARAS_PER_FILE]
    return keep[:MAX_PARAS_PER_FILE]


def _classify_paragraph(para: str) -> str:
    """投标函声明段 / 服务承诺段 / 技术方案段 / 标准条款段 / 其他.

    FIX-2026-09-07-QA-C3: 投标函/响应声明段（"一、按照招标文件要求提交投标文件
    正本1份…""我方已完全理解招标文件…"）是套用招标模板的正常合规内容，不算串标
    证据——单独归类并在评分中排除。优先级：声明段 > 标准条款 > 技术方案 > 服务承诺。
    """
    s = para
    # 投标函/响应声明指纹：我方/本单位/本投标 + 招标文件/条款/正本/副本/有效期/声明/承诺
    declare = sum(1 for m in ("我方", "本单位", "本投标", "本文件", "招标文件", "正本",
                              "副本", "有效期", "截止", "之日起", "日内", "声明",
                              "自愿", "完全理解", "全部条款", "真实有效", "执行") if m in s)
    svc = sum(1 for m in _SVC_MARKERS if m in s)
    tech = sum(1 for m in _TECH_MARKERS if m in s)
    legal = sum(1 for m in _LEGAL_MARKERS if m in s)
    if declare >= 2:
        return "投标函声明段"
    if legal >= 2:
        return "标准条款段"
    if tech >= 2 and tech > svc:
        return "技术方案段"
    if svc >= 2 and svc >= tech:
        return "服务承诺段"
    if tech >= 1:
        return "技术方案段"
    if svc >= 1:
        return "服务承诺段"
    return "其他"


def _surprise(para: str, merged_stop: frozenset) -> tuple[float, int]:
    """Fraction of jieba tokens that survive stop-word removal.

    High surprise = distinctive wording (collusion-relevant).  Low surprise =
    generic industry/template/legal boilerplate (not collusion evidence).
    Returns (fraction, distinct_token_count).
    """
    from app.services.text_utils import lcut
    all_toks = lcut(para)
    toks = [w for w in all_toks if len(w) >= 2 and w not in merged_stop]
    total = len([w for w in all_toks if len(w) >= 2])
    if total == 0:
        return 0.0, 0
    return round(len(toks) / total, 3), len(toks)


def _anchors(s: str) -> tuple:
    """5 overlapping 12-char windows of a paragraph (head / quarters / tail).

    Near-verbatim paragraphs share long contiguous substrings → shared anchors.
    Using multiple overlapping positions tolerates small prefix drift (e.g.
    "服务承诺：" vs "服务承诺：（一）").  Numeric table rows / list items with
    different 编号 do NOT share anchors → pruned before SequenceMatcher.
    """
    n = len(s)
    if n <= ANCHOR_WIN:
        return (s,)
    if n <= ANCHOR_WIN * 2:
        return (s[:ANCHOR_WIN], s[-ANCHOR_WIN:])
    pos = [0, n // 4, n // 2, (3 * n) // 4, n - ANCHOR_WIN]
    return tuple(s[p:p + ANCHOR_WIN] for p in pos)


def detect_shared_substantive_segments(file_data: list[dict], ptype: str | None = None) -> dict:
    """Find cross-file near-verbatim paragraphs that carry substantive (non-boilerplate)
    content.  Returns the collusion evidence list + per-pair stats + a 0-100 score.

    file_data: [{filename, text, ...}] — `text` is RAW (newlines preserved).

    FIX-2026-09-07-QA-C2 (perf): anchor-substring inverted index replaces the
    O(n²) length-bucket scan + bigram-set intersection.  Only paragraphs sharing
    ≥ ANCHOR_MIN_SHARE anchors run SequenceMatcher, so large docs are feasible.
    """
    n = len(file_data)
    if n < 2:
        return {"shared_segments": [], "per_pair_count": {}, "per_pair_ratio": {},
                "collusion_score": 0.0, "ptype": ptype or "engineering", "n_files": n}

    if ptype is None:
        ptype = get_procurement_type(*[fd.get("text", "") for fd in file_data])
    stop = frozenset(load_industry_stopwords(ptype))
    merged_stop = frozenset(set(stop) | set(DEFAULT_STOP_WORDS) | set(_LEGAL_TERMS))

    # 1) 每文件分段 + 预计算
    docs: list[tuple[str, list[tuple[str, str, float, int]]]] = []
    for fi, fd in enumerate(file_data):
        paras = _split_paragraphs(fd.get("text", ""))
        docs.append((fd.get("filename", f"file{fi}"), [
            (p, _classify_paragraph(p), surp, dcnt) for p in paras
            for surp, dcnt in [_surprise(p, merged_stop)]
        ]))

    # 2) 锚子串倒排索引：anchor -> [(doc_idx, para_idx)]
    anchor_index: dict[str, list[tuple[int, int]]] = {}
    for di, (_f, plist) in enumerate(docs):
        for pi, (p, _c, _s, _d) in enumerate(plist):
            for a in set(_anchors(p)):
                anchor_index.setdefault(a, []).append((di, pi))

    # 3) 匹配：锚倒排查候选 → 候选段必须真的包含我方某个窗口子串 → SequenceMatcher
    segments: dict[str, dict] = {}
    seg_order: list[str] = []
    seen_pairs: set[tuple[int, int, int, int]] = set()
    for di in range(n):
        plist = docs[di][1]
        for pi, (para, cls, surp, dcnt) in enumerate(plist):
            plen = len(para)
            my_windows = set(_anchors(para))
            # 统计每个候选段与我共享的锚窗口数
            shared_count: dict[tuple[int, int], int] = {}
            for a in my_windows:
                for dj, pj in anchor_index.get(a, ()):
                    if dj == di:
                        continue
                    k = (dj, pj)
                    shared_count[k] = shared_count.get(k, 0) + 1
            # 候选按共享锚数降序，截断到 MAX_CAND_PER_PARA（真实 near-verbatim 段
            # 共享锚最多必排前；限制比较次数以锁定性能）
            ordered = sorted(shared_count.items(), key=lambda kv: -kv[1])[:MAX_CAND_PER_PARA]
            for (dj, pj), shared in ordered:
                if shared < ANCHOR_MIN_SHARE:
                    continue
                if (di, pi, dj, pj) in seen_pairs or (dj, pj, di, pi) in seen_pairs:
                    continue
                other, ocls, osurp, odcnt = docs[dj][1][pj]
                if abs(len(other) - plen) / max(plen, len(other)) > 0.5:
                    continue
                if min(surp, osurp) < SURPRISE_THRESHOLD or min(dcnt, odcnt) < 2:
                    continue
                # 硬剪枝：候选段必须真的包含我方某个 12 字窗口（锚交集只说明"各有一个同串"，
                # 对 编号行/表格行 这类同前缀段无效——用 `in` 验证连续子串真出现在对方）。
                if not any(w in other for w in my_windows):
                    continue
                ratio = difflib.SequenceMatcher(None, para, other).ratio()
                if ratio < NEAR_DUPLICATE_RATIO:
                    continue
                seen_pairs.add((di, pi, dj, pj))
                key = f"{di}:{pi}|{dj}:{pj}"
                seg = {
                    "segment_text": para[:200],
                    "char_len": plen,
                    "type": cls,
                    "surprise": round(min(surp, osurp), 3),
                    "match_ratio": round(ratio, 3),
                    "matching_files": [(docs[di][0], docs[dj][0], round(ratio, 3))],
                    "files": {docs[di][0], docs[dj][0]},
                    "_content_key": f"{cls}:{para[:80]}",
                }
                segments[key] = seg
                seg_order.append(key)

    # 4) 分流：模板类段（投标函声明/标准条款/其他/项目信息）与 实质段（服务承诺/技术方案）
    #    FIX-2026-09-07-QA-C3：声明/条款段是套招标模板的正常合规内容，不作串标证据。
    TEMPLATE_TYPES = {"投标函声明段", "标准条款段", "其他"}
    template_merged: list[dict] = []
    template_key: dict[str, int] = {}
    merged: list[dict] = []
    merged_key: dict[str, int] = {}
    for key in seg_order:
        seg = segments[key]
        ck = seg.pop("_content_key", None) or f"{seg['type']}:{seg['segment_text'][:80]}"
        is_tpl = seg["type"] in TEMPLATE_TYPES
        bucket, bkey = (template_merged, template_key) if is_tpl else (merged, merged_key)
        if ck in bkey:
            idx = bkey[ck]
            bucket[idx]["files"].update(seg["files"])
            bucket[idx]["matching_files"].extend(seg["matching_files"])
            if seg["match_ratio"] > bucket[idx]["match_ratio"]:
                bucket[idx]["match_ratio"] = seg["match_ratio"]
            continue
        bkey[ck] = len(bucket)
        seg["files"] = set(seg["files"])
        seg["n_files"] = len(seg["files"])
        bucket.append(seg)
    for seg in merged + template_merged:
        seg["files"] = sorted(seg["files"])

    # 5) 每对统计（仅实质段）
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

    # 6) 围标风险分（0-100）：仅基于非模板实质段；数量 √ 曲线 + 多文件一致性 + 平均惊讶度
    score = 0.0
    if merged:
        import math
        cnt = min(len(merged), 5)
        n_files3 = sum(1 for s in merged if s["n_files"] >= 3)
        avg_surp = sum(s["surprise"] for s in merged) / len(merged)
        score = min(100, math.sqrt(cnt) * 25 + n_files3 * 20 + avg_surp * 20)

    return {
        "shared_segments": merged[:MAX_SEGMENTS],
        "template_segments": template_merged[:MAX_SEGMENTS],
        "per_pair_count": {f"({i},{j})": c for (i, j), c in per_pair_count.items()},
        "per_pair_ratio": {f"({i},{j})": r for (i, j), r in per_pair_ratio.items()},
        "collusion_score": round(score, 1),
        "ptype": ptype,
        "n_files": n,
    }
