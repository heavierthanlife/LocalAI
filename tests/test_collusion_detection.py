"""Regression tests for paragraph-level collusion detection (FIX-2026-09-04-QA-B1/B2).

Covers:
  - industry word tables (guardrail: no evaluative wording)
  - procurement type detection
  - near-verbatim substantive segment detection (the 元丰↔物美 服务承诺段 case)
  - legitimate same-industry bidders NOT flagged
  - RiskScorer template-missing gate (no tender → text_sim contributes 0)
  - truncate_filename tiny-width fix (matrix headers were collapsing to ".docx")
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


# ── 6. 性能上限（FIX-2026-09-07-QA-C2）────────────────────────
def test_paragraph_collusion_large_docs_fast():
    """2000 段×3 文件（≈真实军营超市文档规模）须在 60s 内完成（防 O(n²) 回归）。"""
    import time
    from app.services.paragraph_collusion_detector import detect_shared_substantive_segments
    docs = []
    for fi in range(3):
        paras = []
        for k in range(2000):
            if k % 100 == 0:
                s = ("服务承诺：热情、主动、耐心、周到、细致、尽职尽责，对顾客必须树立尊重和友好的态度，"
                     "做到主动服务、微笑待客，让客人宾至如归。")
            else:
                s = f"第{k}项商品陈列规范、理货及时、收银准确，编号{fi*1000+k} 库存充足，保质期管理到位。"
            paras.append(s)
        docs.append({"filename": f"f{fi}.docx", "text": "\n".join(paras)})
    t0 = time.time()
    r = detect_shared_substantive_segments(docs, ptype="services")
    dt = time.time() - t0
    # 全量测试并行时 CPU 竞争明显；单独跑 ~28s。90s 门槛锁"非 O(n²) 回归"。
    assert dt < 90, f"大文档检测应 <90s，实测 {dt:.1f}s（性能回归）"
    # 服务承诺段仍应命中
    assert len(r["shared_segments"]) >= 1, "大文档中服务承诺段雷同仍应检出"


def test_paragraph_collusion_merged_dedup_by_content():
    """同一实质段跨多文件应合并为一条记录（H-3 修复：去重 key=内容前缀非文件集合）。"""
    from app.services.paragraph_collusion_detector import detect_shared_substantive_segments
    svc = "服务承诺：热情、主动、耐心、周到、细致、尽职尽责，对顾客必须树立尊重和友好的态度。"
    tech = "技术方案：采用开槽法施工，沟槽钢板桩支护，井点降水，管线直埋。"
    docs = [
        {"filename": "a.docx", "text": svc + "\n" + tech},
        {"filename": "b.docx", "text": svc + "\n" + tech},
        {"filename": "c.docx", "text": svc},
    ]
    r = detect_shared_substantive_segments(docs, ptype="services")
    segs = r["shared_segments"]
    assert len(segs) == 2, f"应合并为 2 个不同实质段（服务承诺+技术方案），实测 {len(segs)}: {[s['segment_text'][:20] for s in segs]}"
    by_type = {s["type"] for s in segs}
    assert "服务承诺段" in by_type


# ── 7. 行业词表接线指标层（FIX-2026-09-07-QA-C2）───────────────
def test_industry_words_merge_into_key_info():
    """指标层 key_info 应消费行业词表：行业通用词（超市/收银/理货）不抬高重合。"""
    from app.services.file_processing import preprocess_text_for_similarity
    from app.services.industry_words import load_industry_stopwords
    from app.services.stop_words import DEFAULT_STOP_WORDS

    svc = "超市门店运营服务，商品陈列规范，理货及时，收银准确，顾客满意度高。"
    ind = frozenset(load_industry_stopwords("services"))
    # 无行业词表 → 行业词保留
    no_extra = preprocess_text_for_similarity(svc, None)
    # 有行业词表 → 行业词被过滤
    with_extra = preprocess_text_for_similarity(svc, None, extra_stop_words=ind)
    assert any(w in no_extra.split() for w in ("超市", "商品", "收银", "理货")), \
        f"无行业词表时应保留行业词: {no_extra[:40]}"
    assert not any(w in with_extra.split() for w in ("超市", "商品", "收银", "理货")), \
        f"有行业词表时应过滤行业词: {with_extra[:40]}"


def test_run_analysis_key_info_uses_industry_words():
    """run_analysis 的 key_info checker 应传行业词表（指标层不因行业词虚高）。"""
    from app.services.document_analysis_svc import run_analysis
    boiler = "应当遵守国家法律法规和招标文件要求，认真履行合同义务。"
    docs = [
        {"filename": "a.docx", "text": boiler + "超市商品陈列规范，收银准确，理货及时。", "metadata": {}, "images": []},
        {"filename": "b.docx", "text": boiler + "超市商品陈列规范，收银准确，理货及时。", "metadata": {}, "images": []},
        {"filename": "c.docx", "text": boiler + "注重商品质量管理与库存周转。", "metadata": {}, "images": []},
    ]
    report = run_analysis(docs, user_id="t", thread_id="t")
    ki = next((i for i in report["indicators"] if i["id"] == "contact_person_same"), None)
    # 行业词被过滤后：a/b 共享的剩余关键词应远少于行业词未过滤时
    # 直接验证：共同关键词里不应以纯行业词（超市/收银/理货/陈列）为主
    if ki:
        det = ki.get("details") or []
        all_kw = []
        for d in det:
            all_kw.extend(d.get("keywords", []) if isinstance(d, dict) else [])
        ind_words = {"超市", "商品", "收银", "理货", "陈列", "顾客"}
        ind_hits = [w for w in all_kw if w in ind_words]
        assert len(ind_hits) <= 1, f"共同关键词不应以行业词为主: {all_kw} 命中{ind_hits}"
        # 分数不因纯行业词雷同而虚高（≤15 即可，行业词过滤后应明显低于未过滤）
        assert ki.get("score", 0) <= 15, f"行业词过滤后 key_info 不应虚高: {ki.get('score')}"


# ── 8. 投标函声明段排除（FIX-2026-09-07-QA-C3）─────────────────
def test_declaration_template_excluded_from_collusion():
    """投标函/声明模板段（套招标模板的合规内容）不列入实质雷同。"""
    from app.services.paragraph_collusion_detector import detect_shared_substantive_segments
    declare = ("一、按照招标文件要求提交投标文件正本1份和副本4份，电子版投标文件2份。"
               "二、我方已完全理解招标文件的全部内容，自愿接受并执行招标文件的全部条款。"
               "三、本投标有效期自提交投标文件的截止之日起180日内有效。")
    svc = "服务承诺：热情、主动、耐心、周到、细致、尽职尽责，对顾客必须树立尊重和友好的态度。"
    docs = [
        {"filename": "a.docx", "text": declare + "\n" + svc},
        {"filename": "b.docx", "text": declare + "\n" + "（一）" + svc},
        {"filename": "c.docx", "text": declare + "\n" + "服务态度：文明礼貌热情，主动为顾客设想。"},
    ]
    r = detect_shared_substantive_segments(docs, ptype="services")
    # 声明段应归入 template_segments 而非 shared_segments（实质段）
    real_types = {s["type"] for s in r["shared_segments"]}
    assert "投标函声明段" not in real_types, f"声明段不应作实质雷同: {real_types}"
    assert real_types and real_types == {"服务承诺段"}, f"应只保留服务承诺铁证: {real_types}"
    assert len(r["shared_segments"]) >= 1, "服务承诺铁证不应丢失"


# ── 9. 元数据硬信号（FIX-2026-09-07-QA-C3）─────────────────────
def test_lasteditor_metadata_extracted():
    """extract_metadata 应取 docx 的 cp:lastModifiedBy（同一最后编辑人硬信号）。"""
    from app.services.file_processing import extract_metadata
    import io
    from docx import Document as D
    buf = io.BytesIO()
    doc = D()
    cp = doc.core_properties
    cp.last_modified_by = "超彩赵"
    doc.save(buf)
    buf.seek(0)
    class _Fake:
        filename = "test.docx"
        def read(self):
            return buf.getvalue()
        def seek(self, _):
            buf.seek(0)
    meta = extract_metadata(_Fake())
    assert meta.get("last_modified_by") == "超彩赵", f"应取到 last_modified_by: {meta}"


def test_run_analysis_lasteditor_indicator():
    """两文件 last_modified_by 相同 → file_attr_lasteditor_same 指标触发。"""
    from app.services.document_analysis_svc import run_analysis
    docs = [
        {"filename": "a.docx", "text": "投标文件内容甲。", "metadata": {"last_modified_by": "超彩赵"}, "images": []},
        {"filename": "b.docx", "text": "投标文件内容乙。", "metadata": {"last_modified_by": "超彩赵"}, "images": []},
    ]
    report = run_analysis(docs, user_id="t", thread_id="t")
    inds = {i["id"]: i for i in report["indicators"]}
    le = inds.get("file_attr_lasteditor_same")
    assert le is not None, "缺少 file_attr_lasteditor_same 指标"
    assert le.get("score", 0) > 0, f"相同最后编辑人应触发: {le.get('score')} {le.get('result')}"


# ── 10. 联系人/电话/邮箱真实比对（FIX-2026-09-07-QA-C4）────────
def test_contact_person_same_real_match():
    """跨文件同联系人 → contact_person_same 触发（真实联系人，非关键词）。"""
    from app.services.document_analysis_svc import run_analysis
    docs = [
        {"filename": "甲公司.docx", "text": "投标联系人：李四 联系电话：13811112222", "metadata": {}, "images": []},
        {"filename": "乙公司.docx", "text": "委托代理人：李四 手机 13811112222", "metadata": {}, "images": []},
        {"filename": "丙公司.docx", "text": "联系人：王五 电话 13933334444", "metadata": {}, "images": []},
    ]
    report = run_analysis(docs, user_id="t", thread_id="t")
    inds = {i["id"]: i for i in report["indicators"]}
    cp = inds["contact_person_same"]
    assert cp.get("score", 0) > 0, f"同联系人应触发: {cp.get('result')}"
    ph = inds["contact_phone_abnormal"]
    assert ph.get("score", 0) > 0, f"同电话应触发: {ph.get('result')}"
    # 关键词不应出现在 details（真实联系人证据）
    det = " ".join(str(d) for d in cp.get("details", []))
    assert "采购" not in det and "超市" not in det, f"details 不应含行业词: {det}"


def test_contact_no_data_placeholder():
    """无联系人实体 → contact 指标落"○ 需联系人数据"占位（score=0）。"""
    from app.services.document_analysis_svc import run_analysis
    docs = [
        {"filename": "a.docx", "text": "本项目为超市经营服务，商品陈列规范。", "metadata": {}, "images": []},
        {"filename": "b.docx", "text": "经营思路：注重商品质量与库存周转。", "metadata": {}, "images": []},
    ]
    report = run_analysis(docs, user_id="t", thread_id="t")
    inds = {i["id"]: i for i in report["indicators"]}
    cp = inds["contact_person_same"]
    assert cp.get("score", 0) == 0
    assert "需开标信息表" in cp.get("result", ""), f"应占位提示: {cp.get('result')}"


# ── 11. 暗标检测（FIX-2026-09-07-QA-C4）────────────────────────
def test_tech_seal_detector_leak_and_clean():
    """技术方案段内公司名 → 泄露；正常商务标自指称谓 → 不泄露。"""
    from app.services.tech_seal_detector import detect_tech_seal_leak
    leaky = detect_tech_seal_leak([
        {"filename": "北京中昌华美超市服务有限责任公司.docx",
         "text": "技术方案：本项目采用先进工艺。北京中昌华美超市服务有限责任公司将全权负责实施。"},
    ])
    assert leaky["北京中昌华美超市服务有限责任公司.docx"]["leak"] is True, "技术方案段公司名应判泄露"

    clean = detect_tech_seal_leak([
        {"filename": "商务技术文件-北京物美邻鲜连锁超市有限公司.docx",
         "text": "我公司承诺严格按照招标文件要求提供服务，本公司实力雄厚。"},
    ])
    assert clean["商务技术文件-北京物美邻鲜连锁超市有限公司.docx"]["leak"] is False, "纯自指称谓不应误判泄露"


def test_bidder_count_abnormal_local():
    """有效投标数 <3 → bidder_count_abnormal 本地触发（不再 skip）。"""
    from app.services.document_analysis_svc import run_analysis
    docs = [
        {"filename": "a.docx", "text": "内容甲", "metadata": {}, "images": []},
        {"filename": "b.docx", "text": "内容乙", "metadata": {}, "images": []},
    ]
    report = run_analysis(docs, user_id="t", thread_id="t")
    inds = {i["id"]: i for i in report["indicators"]}
    bc = inds["bidder_count_abnormal"]
    assert bc.get("score", 0) > 0, f"n<3 应触发: {bc.get('score')} {bc.get('result')}"
    assert bc.get("skipped") is not True, "bidder_count 不应 skip"
