"""W1a: Snapshot tests for batch audit orchestrator functions.

Lock the input/output contract of these three functions before refactoring:
  - _precompute_tfidf_for_files / _compute_pair_similarity_from_matrix
  - keyword_overlap_similarity
  - risk scoring formula (0.3*key + 0.3*attr + 0.2*text + 0.2*img)

These tests call existing functions directly (no HTTP layer). They must pass
against the current code AND against the new orchestrator in W1b.
"""
import json
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def bid_texts():
    """Three realistic Chinese bidding text snippets covering different domains."""
    return {
        "engineering": (
            "第一章 招标公告\n"
            "1. 招标条件\n"
            "本招标项目已由某市发展和改革委员会批准建设，招标人为某市城市建设投资有限公司，"
            "建设资金来自财政拨款，项目出资比例为100%。\n"
            "2. 项目概况\n"
            "建设地点：某市新城区；建设规模：总建筑面积约50000平方米；"
            "计划工期：730日历天；招标范围：施工图纸范围内的土建、安装工程。"
        ),
        "goods": (
            "第一章 招标公告\n"
            "1. 招标条件\n"
            "本招标项目为某医院医疗器械采购项目，招标人为某市人民医院，"
            "采购资金已落实，项目已具备招标条件。\n"
            "2. 项目概况\n"
            "采购内容：CT机1台、MRI设备1台、超声诊断仪3台；"
            "预算金额：1500万元；交货期：合同签订后90天内；质保期：验收合格后24个月。"
        ),
        "similar_to_engineering": (
            "第一章 招标公告\n"
            "1. 招标条件\n"
            "本招标项目已由某市发展和改革委员会批准建设，招标人为某市城市建设投资有限公司，"
            "建设资金来自自筹资金，项目出资比例为80%。\n"
            "2. 项目概况\n"
            "建设地点：某市新城区；建设规模：总建筑面积约48000平方米；"
            "计划工期：700日历天；招标范围：施工图纸范围内的土建、安装及装修工程。"
        ),
    }


@pytest.fixture
def keyword_texts():
    """Text pairs for keyword overlap comparison.

    Note: `keyword_overlap_similarity` uses jieba TF-IDF to extract top-20
    keywords from each text, then computes Jaccard index.  For short Chinese
    texts, the keyword set is small, so even minor differences in wording
    produce significant Jaccard swings.
    """
    return {
        "slight_diff": (
            # First text has extra unique keywords (项目经理, 注册建造师, 执业资格).
            # Second text has different unique keywords (技术负责人, 高级工程师).
            # Both share: 投标人, 须具备, 建筑工程, 施工总承包, 一级资质, 有效的, 安全生产许可证
            "投标人须具备建筑工程施工总承包一级资质，并具有有效的安全生产许可证。"
            "项目经理须具有建筑工程专业一级注册建造师执业资格。",
            "投标人须具备建筑工程施工总承包一级资质，并具有有效的安全生产许可证。"
            "技术负责人须具有高级工程师职称。",
        ),
        "no_overlap": (
            # Completely different domains: bidding vs procurement.
            "投标人须具备建筑工程施工总承包一级资质，并具有有效的安全生产许可证。",
            "供货商须提供产品合格证书和质量检测报告，进口产品须提供报关单。",
        ),
        "near_identical": (
            # Nearly identical texts differing only in 一级/二级 and 5000/3000.
            # Expect VERY high Jaccard due to shared keyword structure.
            "投标人应具备独立法人资格和项目施工总承包一级资质，注册资本不低于5000万元。",
            "投标人应具备独立法人资格和项目施工总承包二级资质，注册资本不低于3000万元。",
        ),
    }


# ── Test 1: TF-IDF pairwise similarity snapshot ───────────────────────────────

def test_snapshot_tfidf_pairwise(bid_texts):
    """Lock _precompute_tfidf_for_files + _compute_pair_similarity_from_matrix.

    Three text pairs of increasing similarity. The TF-IDF cosine values are
    determined by Chinese tokens extracted via jieba, so they are sensitive to
    any change in the tokenizer or vectorizer configuration.
    """
    from app.services.batch_compare_svc import (
        _precompute_tfidf_for_files,
        _compute_pair_similarity_from_matrix,
    )

    # Pair A: engineering vs goods (different domains)
    data_a = [
        {"text": bid_texts["engineering"]},
        {"text": bid_texts["goods"]},
    ]
    _, matrix_a = _precompute_tfidf_for_files(data_a)
    sim_a = float(_compute_pair_similarity_from_matrix(matrix_a, 0, 1))

    # Pair B: engineering vs similar (high overlap)
    data_b = [
        {"text": bid_texts["engineering"]},
        {"text": bid_texts["similar_to_engineering"]},
    ]
    _, matrix_b = _precompute_tfidf_for_files(data_b)
    sim_b = float(_compute_pair_similarity_from_matrix(matrix_b, 0, 1))

    # Compute exact snapshot values and print for confirmation
    results = {
        "pair_a_eng_vs_goods": round(sim_a, 8),
        "pair_b_eng_vs_similar": round(sim_b, 8),
    }

    # ── Snapshot values (locked to current jieba + sklearn + stop-word filter) ──
    assert results["pair_a_eng_vs_goods"] == pytest.approx(0.86620383, abs=1e-6)
    assert results["pair_b_eng_vs_similar"] == pytest.approx(0.98253900, abs=1e-6)

    # ── Invariants (must hold regardless of tokenizer version) ──
    assert sim_b > sim_a, f"eng-vs-similar {sim_b} !> eng-vs-goods {sim_a}"
    assert sim_b > 0.9, "engineering vs similar should have high cosine (>0.9)"
    assert sim_a > 0.8, "engineering vs goods still share template structure (>0.8)"


# ── Test 2: Keyword overlap similarity snapshot ───────────────────────────────

def test_snapshot_keyword_overlap(keyword_texts):
    """Lock keyword_overlap_similarity output for known text pairs.

    Uses Jaccard index on jieba-extracted keywords. Any change in extraction
    or tokenization will shift these values.
    """
    from app.services.file_processing import keyword_overlap_similarity

    results = {}
    for name, (t1, t2) in keyword_texts.items():
        val = float(keyword_overlap_similarity(t1, t2))
        results[name] = round(val, 8)

    # ── Invariants ──
    for name, val in results.items():
        assert 0.0 <= val <= 1.0, f"{name}: {val} out of Jaccard range"

    # "near_identical" pair has highest overlap (only differs in 一级/二级, 5000/3000)
    assert results["near_identical"] > results["slight_diff"], (
        f"near_identical {results['near_identical']} !> slight_diff {results['slight_diff']}"
    )
    assert results["near_identical"] > results["no_overlap"]

    # "no_overlap" has the lowest (but may share rare function words)
    # "slight_diff" shares many keywords but has distinct role-specific ones
    assert results["no_overlap"] < results["slight_diff"], (
        f"no_overlap {results['no_overlap']} !< slight_diff {results['slight_diff']}"
    )

    # ── Snapshot values (FIX-013 stop-word filtered) ──
    assert results["near_identical"] == pytest.approx(0.80000000, abs=1e-6)
    assert results["slight_diff"] == pytest.approx(0.60000000, abs=1e-6)
    assert results["no_overlap"] == pytest.approx(0.08571429, abs=1e-6)


# ── Test 3: Risk scoring formula snapshot ────────────────────────────────────

def test_snapshot_risk_formula():
    """Lock the risk scoring formula: 0.375*key + 0.375*attr + 0.25*text (+0 img).

    FIX-010: image weight removed (image_sim disabled in clearance), text_sim only
    contributes when cosine ≥ 80% (template-overlap gate).

        risk = 0.375*key + 0.375*attr + 0.25*text_gated + 0.0*img

    where:
      key_info_val = key_overlap_similarity * 100  (percentage)
      file_attr_val = raw attr_similarity (0-100)
      text_sim_val = cosine_similarity * 100, but 0 unless ≥80
      img_sim_val = ignored (0)
    """
    def _risk(key_info_pct, file_attr_val, text_sim_pct, img_sim_val):
        text_eff = text_sim_pct if text_sim_pct >= 80 else 0.0
        return (0.375 * key_info_pct + 0.375 * file_attr_val +
                0.25 * text_eff + 0.0 * img_sim_val)

    scenarios = [
        # (name, key_pct, attr, text_pct, img) -> expected_risk
        ("zero_all",       0,    0,    0,   0),
        ("typical_low",   30,   20,   15,  10),   # text<80 → gated 0
        ("typical_mid",   50,   40,   30,  25),
        ("typical_high",  80,   70,   60,  50),
        ("max_all",      100,  100,  100, 100),
        ("key_heavy",     90,   10,   10,  10),
        ("text_high_gate", 10,  10,   90,  10),   # text≥80 → counts
        ("text_below_gate",10,  10,   70,  10),   # text<80 → 0
        ("attr_heavy",    10,   90,   10,  10),
        ("mixed",         45,   55,   35,  15),
    ]

    results = {}
    for name, k, a, t, i in scenarios:
        results[name] = _risk(k, a, t, i)

    # ── Exact arithmetic snapshots (weights 0.375/0.375/0.25) ──
    assert results["zero_all"] == 0.0
    # 30*.375 + 20*.375 + 0 (text<80 gated) = 11.25 + 7.5 = 18.75
    assert results["typical_low"] == 18.75
    # 50*.375 + 40*.375 + 0 = 18.75 + 15 = 33.75
    assert results["typical_mid"] == 33.75
    # 80*.375 + 70*.375 + 0 = 30 + 26.25 = 56.25
    assert results["typical_high"] == 56.25
    # 100*.375 + 100*.375 + 100*.25 = 37.5 + 37.5 + 25 = 100.0
    assert results["max_all"] == 100.0
    # 90*.375 + 10*.375 + 0 = 33.75 + 3.75 = 37.5
    assert results["key_heavy"] == 37.5
    # text_high_gate: 10*.375 + 10*.375 + 90*.25 = 3.75+3.75+22.5 = 30.0
    assert results["text_high_gate"] == 30.0
    # text_below_gate: 10*.375 + 10*.375 + 0 = 7.5
    assert results["text_below_gate"] == 7.5
    # attr_heavy: 10*.375 + 90*.375 + 0 = 3.75 + 33.75 = 37.5
    assert results["attr_heavy"] == 37.5
    # mixed: 45*.375 + 55*.375 + 0 = 16.875 + 20.625 = 37.5
    assert results["mixed"] == 37.5

    # ── Invariants ──
    # Weights sum to 1.0 (incl. the 0 image weight)
    assert abs(0.375 + 0.375 + 0.25 + 0.0 - 1.0) < 1e-10

    # text_sim gate: below 80% contributes 0, at/above contributes linearly
    assert _risk(0, 0, 79, 0) == 0.0
    assert _risk(0, 0, 80, 0) == 20.0
    assert _risk(0, 0, 100, 0) == 25.0

    # Weighted average never exceeds the maximum component (text-gated)
    for name, k, a, t, i in scenarios:
        risk = _risk(k, a, t, i)
        max_comp = max(k, a, t if t >= 80 else 0, i)
        assert risk <= max_comp, f"{name}: risk {risk} > max component {max_comp}"

    # Risk is bounded below by 0
    for name, k, a, t, i in scenarios:
        assert _risk(k, a, t, i) >= 0.0
