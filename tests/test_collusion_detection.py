"""Regression tests for paragraph-level collusion detection (FIX-2026-09-04-QA-B1/B2).

Covers:
  - industry word tables (guardrail: no evaluative wording)
  - procurement type detection
  - near-verbatim substantive segment detection (the 元丰↔物美 服务承诺段 case)
  - legitimate same-industry bidders NOT flagged
  - RiskScorer template-missing gate (no tender → text_sim contributes 0)
  - truncate_filename tiny-width fix (matrix headers were collapsing to ".docx")
  - cross-file shared typo detection
"""

import difflib
import os

# ── 1. 行业词表守卫 ──────────────────────────────────────────────
def test_industry_words_guardrail_no_evaluative_wording():
    """评价性/独特措辞绝不能进行业词表（否则"热情"段雷同会被当行业词抹掉）。"""
    from app.services.industry_words import guardrail_check
    found = guardrail_check()
    for ptype, hits in found.items():
        assert hits == [], f"{ptype} 表混入评价性词: {hits}"


def test_industry_words_tables_nonempty_and_layered():
    from app.services.industry_words import load_industry_words
    for ptype in ("engineering", "goods", "services"):
        layers = load_industry_words(ptype)
        assert len(layers["national"]) > 30, f"{ptype} 国家标准词层过薄"
        assert len(layers["operational"]) > 10, f"{ptype} 常见运营词层过薄"
        # 服务表应含超市经营词（用户案例刚需）
        if ptype == "services":
            allw = layers["national"] | layers["operational"]
            assert {"商品", "陈列", "理货", "收银", "顾客"}.issubset(allw)


def test_get_procurement_type_detection():
    from app.services.industry_words import get_procurement_type
    assert get_procurement_type("本项目为建筑工程施工总承包，含桩基、主体结构、装修") == "engineering"
    assert get_procurement_type("采购办公设备一批，含计算机、打印机、耗材，供货安装调试") == "goods"
    assert get_procurement_type("超市门店运营服务，含商品陈列、理货、收银、顾客服务") == "services"
    # 无法判断 → 回退工程（设计决策）
    assert get_procurement_type("") == "engineering"
    assert get_procurement_type("一般性内容，无明显行业特征") == "engineering"


# ── 2. 段落级实质雷同检测 ────────────────────────────────────────
_BOILER = ("应当遵守国家法律法规和招标文件要求，按照投标文件承诺的服务标准，"
           "认真履行合同义务，确保服务质量，满足部队官兵及其家属的需求。")

_SVC_YUANFENG = "服务承诺：热情、主动、耐心、周到、细致、尽职尽责，对顾客必须树立尊重和友好的态度。"
_SVC_WUMEI = "服务承诺：（一）热情、主动、耐心、周到、细致、尽职尽责，对顾客必须树立尊重和友好的态度。"
_SVC_ZHONGCHANG = "服务态度：礼貌、热情、微笑、真诚，保证顾客在本超市受到亲切专业的服务。"


def _three_colluding_docs():
    return [
        {"filename": "元丰商务技术文件.docx",
         "text": _BOILER + "\n" + _SVC_YUANFENG + "\n超市经营方案：商品陈列规范，理货及时，收银准确。"},
        {"filename": "物美商务技术文件.docx",
         "text": _BOILER + "\n" + _SVC_WUMEI + "\n超市经营方案：注重商品陈列与理货。"},
        {"filename": "中昌华美商务技术文件.docx",
         "text": _BOILER + "\n" + _SVC_ZHONGCHANG + "\n经营思路：丰富商品种类。"},
    ]


def test_paragraph_collusion_detects_service_commitment():
    """元丰↔物美 服务承诺段近逐字雷同必须被检出（surprise 高 + 一致率 ≥0.85）。"""
    from app.services.paragraph_collusion_detector import detect_shared_substantive_segments
    r = detect_shared_substantive_segments(_three_colluding_docs(), ptype="services")
    segs = r["shared_segments"]
    assert len(segs) >= 1, "必须检出共享实质段"
    best = max(segs, key=lambda s: s["match_ratio"])
    assert best["match_ratio"] >= 0.85, f"一致率应高: {best['match_ratio']}"
    assert best["surprise"] >= 0.5, f"服务承诺段惊讶度应高: {best['surprise']}"
    assert best["type"] == "服务承诺段"
    # 样板段不应作为实质雷同（surprise 低被排除）
    for s in segs:
        assert "应当遵守" not in s["segment_text"], "标准条款/样板段不应作为实质雷同信号"


def test_paragraph_collusion_legit_bidders_not_flagged():
    """合法同行：行业词重叠但独特段不同 → 无实质雷同信号。"""
    from app.services.paragraph_collusion_detector import detect_shared_substantive_segments
    docs = [
        {"filename": "a.docx",
         "text": _BOILER + "\n技术方案：采用开槽法施工，沟槽钢板桩支护，井点降水。\n超市经营方案：商品陈列规范。"},
        {"filename": "b.docx",
         "text": _BOILER + "\n技术方案：采用定向钻穿越施工，泥浆护壁。\n超市经营方案：注重商品质量与保质期管理。"},
        {"filename": "c.docx",
         "text": _BOILER + "\n技术方案：采用盾构法施工，管片拼装。\n超市经营方案：加强库存周转控制。"},
    ]
    r = detect_shared_substantive_segments(docs, ptype="services")
    assert r["shared_segments"] == [], f"合法同行不应有实质雷同: {r['shared_segments']}"
    assert r["collusion_score"] == 0.0


def test_paragraph_collusion_score_bounded():
    from app.services.paragraph_collusion_detector import detect_shared_substantive_segments
    r = detect_shared_substantive_segments(_three_colluding_docs(), ptype="services")
    assert 0.0 <= r["collusion_score"] <= 100.0


# ── 3. RiskScorer 门槛对齐 ───────────────────────────────────────
def test_risk_scorer_template_missing_zeroes_text():
    """无招标文件时整篇 text_sim 对 risk 零贡献（模板重叠 ≠ 围标）。"""
    from app.services.batch_orchestrator import RiskScorer
    assert RiskScorer.compute(0, 0, 87, 0) > 0, "有模板时 87% 余弦应计入（≥80 门槛）"
    assert RiskScorer.compute(0, 0, 87, 0, template_missing=True) == 0.0, "无招标文件时必须归零"
    assert RiskScorer.compute(0, 0, 95, 0, template_missing=True) == 0.0


def test_cross_comparison_template_missing_flag():
    """_run_cross_comparison 无招标文件时必须标记 template_missing 并保留文本矩阵参考。"""
    from app.services.clearance_engine import _run_cross_comparison
    cross = _run_cross_comparison(_three_colluding_docs(), tender_text=None)
    assert cross.get("template_missing") is True
    # 文本矩阵保留原始值（供参考），但 pair risk 不含 text_sim 贡献
    m = cross.get("text_matrix") or []
    if m:
        assert m[0][1] > 0, "文本矩阵应保留参考值"
    for p in cross.get("pairs", []):
        assert p.get("collusion_para", 0) >= 0


# ── 4. 矩阵表头修复 ─────────────────────────────────────────────
def test_truncate_filename_tiny_width_not_extension_only():
    """max_len 过小时不能退化成纯扩展名 '.docx'。"""
    from app.services.file_processing import truncate_filename
    name = "12.18元丰 军营超市项目（三次）--商务技术文件.docx"
    short = truncate_filename(name, 8)
    assert short != ".docx", f"max_len=8 不应只剩扩展名: {short!r}"
    assert len(short) <= 8, f"截断后长度应 ≤8: {short!r}"


# ── 5. 跨文件共享错别字 ─────────────────────────────────────────
def test_find_shared_typos_cross_file():
    """三家写同一错别字 → 检出；各家不同错别字 → 不共享。"""
    from app.services.typo_detector import find_shared_typos, detect_typos_batch
    docs = [
        {"filename": "a.docx", "text": "双方签定合同后生效，签定补充协议需书面确认。"},
        {"filename": "b.docx", "text": "中标后签定施工合同，逾期视为放弃。"},
        {"filename": "c.docx", "text": "先签定框架协议，再按需签定订单。"},
    ]
    batch = detect_typos_batch(docs)
    r = find_shared_typos(docs, ptype="engineering", precomputed=batch, min_confidence=0.70)
    assert r["shared_typo_count"] >= 1, "共享错别字必须检出"
    texts = {s["suspect_text"] for s in r["shared_typos"]}
    assert "签定" in texts or any("签定" in s["suspect_text"] for s in r["shared_typos"])

    docs2 = [
        {"filename": "a.docx", "text": "质量检查合格证齐全，必须严格按照标准执行。"},
        {"filename": "b.docx", "text": "工期紧任务重，应当合理安排施工计划。"},
        {"filename": "c.docx", "text": "安全生产责任制落实到岗到人。"},
    ]
    batch2 = detect_typos_batch(docs2)
    r2 = find_shared_typos(docs2, ptype="engineering", precomputed=batch2, min_confidence=0.70)
    assert r2["shared_typo_count"] == 0, "无跨文件共享错别字时不应误报"
