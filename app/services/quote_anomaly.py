"""Quote anomaly detection for bid documents.

Detects suspicious bidding patterns in extracted prices/percentages:
  - Coefficient of variation (CV) outliers
  - Same-rate / near-identical bidding (collusion signal)
  - Abnormal drop rates (below historical/reference threshold)
  - Price clustering (bidders cluster unnaturally close)
  - Benford's Law deviation on leading digits
  - Chinese daxie (大写) numeral cross-validation

Thresholds are configurable via runtime_config.py (admin panel).
All functions are pure-text/regex based; the stats engine requires numpy/scipy only.
"""

from __future__ import annotations

import json as _json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import chain
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# --- Chinese numerals (小写 + 大写) -------------------------------------------
_CN_NUM = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5,
    '陆': 6, '柒': 7, '捌': 8, '玖': 9, '拾': 10,
    '佰': 100, '仟': 1000, '万': 10000, '亿': 100000000,
}
_CN_UNIT = {'十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}

# Daxie-only mapping for cross-referencing amounts (大写金额)
_DAXIE_DIGIT = {'壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5,
                '陆': 6, '柒': 7, '捌': 8, '玖': 9, '零': 0}
_DAXIE_UNIT = {'拾': 10, '佰': 100, '仟': 1000, '万': 10000, '亿': 100000000,
               '元': 1, '角': 0.1, '分': 0.01}


def _cn_to_arabic(cn: str) -> Optional[float]:
    """Convert common Chinese numerals to Arabic number."""
    if not cn:
        return None
    # Simple case: pure digits
    if re.fullmatch(r'[\d.,]+', cn):
        try:
            return float(cn.replace(',', ''))
        except ValueError:
            return None

    unit = 1
    result = 0.0
    partial = 0.0
    digits = []
    for ch in reversed(cn):
        if ch in _CN_NUM:
            digit = _CN_NUM[ch]
            if digit >= 10:
                if partial == 0:
                    partial = 1
                result += partial * digit
                partial = 0
            else:
                partial += digit
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if partial == 0:
                partial = 1
            result += partial * unit
            partial = 0
            if unit >= 10000:
                unit = 1
        elif ch.isdigit():
            digits.append(ch)
        else:
            continue
    if digits:
        num_str = ''.join(reversed(digits))
        result = float(num_str) * (unit if unit and unit < 10000 else 1)
    result += partial
    return result


def parse_daxie_amount(text: str) -> list[dict]:
    """Extract daxie (大写) amounts and cross-reference with Arabic equivalents.

    Returns list of {'daxie': str, 'arabic_from_daxie': float,
                     'context_arabic': float|None, 'match': bool}
    """
    daxie_pattern = re.compile(
        r'(?P<daxie>'
        r'(?:壹|贰|叁|肆|伍|陆|柒|捌|玖|拾|佰|仟|万|亿|零|元|角|分|整)+'
        r')'
    )
    results = []
    for m in daxie_pattern.finditer(text):
        daxie_str = m.group('daxie')
        val = _cn_to_arabic(daxie_str)
        if val is None:
            continue
        # Look for nearby Arabic amount to cross-validate
        nearby = text[max(0, m.start()-50):m.end()+50]
        arabic_nearby = re.findall(r'[\d,]+\.?\d*\s*(?:万元|元|万)', nearby)
        context_val = None
        for a in arabic_nearby:
            try:
                context_val = float(re.sub(r'[^\d.]', '', a))
                if '万' in a:
                    context_val *= 10000
                break
            except ValueError:
                continue
        results.append({
            'daxie': daxie_str,
            'arabic_from_daxie': val,
            'context_arabic': context_val,
            'match': abs(val - context_val) < 0.01 if context_val is not None else None,
        })
    return results


# --- Price extraction regexes ------------------------------------------------

_PRICE_UNIT = r'(?:元|万元|亿人民币|万元整|元整|￥|¥|RMB|CNY|万元)?'
_DECIMAL_ARABIC = re.compile(
    r'(?<!\d)'
    r'(?P<num>[\d]{1,3}(?:,[\d]{3})+\.[\d]+|[\d]{1,3}(?:,[\d]{3})+|[\d]+\.[\d]+|[\d]{4,})'
    r'\s*'
    r'(?P<unit>万元|万人民币|元人民币|元|亿元|万人民币元)?'
    r'(?=\s|，|。|,|\.|[一-鿿]|$)',
    re.IGNORECASE,
)
_PERCENT = re.compile(
    r'(?P<num>[\d]+(?:\.[\d]+)?)\s*%(?:\s*[一-鿿])?'
    r'|'
    r'(?:百分之|折扣率|下浮率|费率)(?P<num2>[\d\.]+)',
    re.IGNORECASE,
)
_CN_PRICE = re.compile(
    # cn 组要求至少含一个数字字符（排除独立匹配 "万"/"亿" → 假 10000/1e8），
    # 保留 拾佰仟 位值字符以支持 壹佰贰拾万元 类写法
    r'(?P<cn>(?:[零一二两三四五六七八九壹贰叁肆伍陆柒捌玖拾佰仟]+[万亿]?|[万亿](?=[零一二两三四五六七八九壹贰叁肆伍陆柒捌玖]))+)'
    r'\s*(?P<unit>元|万元|元人民币|万人民币元|亿元)',
    re.IGNORECASE,
)


def extract_prices(text: str) -> list[float]:
    """Extract numeric prices from bid text (Arabic + Chinese numerals).

    Dedupes by (value, span) across both _DECIMAL_ARABIC and _CN_PRICE so the
    same price is never double-counted (FIX-011).
    """
    prices = []
    seen = set()

    for m in chain(_DECIMAL_ARABIC.finditer(text), _CN_PRICE.finditer(text)):
        try:
            if m.groupdict().get('num'):
                val = float(m.group('num').replace(',', ''))
                unit = m.group('unit') or ''
                if '万' in unit:
                    val *= 10000
                elif '亿' in unit:
                    val *= 100000000
            elif m.groupdict().get('cn'):
                val = _cn_to_arabic(m.group('cn'))
                if val is None:
                    continue
                unit = m.group('unit') or ''
                if '万' in unit:
                    val *= 10000
                elif '亿' in unit:
                    val *= 100000000
            else:
                continue
            if val > 0 and val < 1e12:
                key = (round(val, 2), m.span())
                if key not in seen:
                    seen.add(key)
                    prices.append(val)
        except (ValueError, IndexError, TypeError):
            continue
    return prices


def extract_percentages(text: str) -> list[float]:
    """Extract percentage values (e.g. discount rates, fees)."""
    pcts = []
    for m in _PERCENT.finditer(text):
        try:
            raw = m.group('num') or m.group('num2')
            val = float(raw)
            if 0 <= val <= 100:
                pcts.append(val)
        except (ValueError, TypeError):
            continue
    return pcts


# --- Config helpers ----------------------------------------------------------

def _get_thresholds() -> dict:
    """Load thresholds from runtime_config, falling back to hardcoded defaults."""
    try:
        from app.services.runtime_config import get
        return {
            'same_rate': get('quote_anomaly_same_rate_threshold', 0.05),
            'drop': get('quote_anomaly_drop_threshold', 0.30),
            'cv_low': get('quote_anomaly_cv_low_alert', 0.05),
            'cv_high': get('quote_anomaly_cv_high_alert', 1.5),
            'benford': get('quote_anomaly_benford_deviation_alert', 0.15),
            'bandwidth': get('quote_anomaly_clustering_bandwidth', 0.02),
            'min_cluster': int(get('quote_anomaly_min_cluster_size', 3)),
            'min_benford': int(get('quote_anomaly_min_prices_for_benford', 20)),
        }
    except Exception:
        return {
            'same_rate': 0.05, 'drop': 0.30, 'cv_low': 0.05, 'cv_high': 1.5,
            'benford': 0.15, 'bandwidth': 0.02, 'min_cluster': 3, 'min_benford': 20,
        }


# --- Anomaly detection -------------------------------------------------------

@dataclass
class QuoteAnomalyResult:
    doc_name: str
    prices: list[float] = field(default_factory=list)
    percentages: list[float] = field(default_factory=list)
    cv: float = 0.0
    same_rate_flag: bool = False
    abnormal_drop_flag: bool = False
    clustering_flag: bool = False
    progression_type: str = ''
    tailing_digits_flag: bool = False
    tailing_digits_info: dict = field(default_factory=dict)
    benford_deviation: float = 0.0
    risk_score: float = 0.0
    details: list[str] = field(default_factory=list)
    matched_prices: dict[str, list[float]] = field(default_factory=dict)
    daxie_mismatches: list[dict] = field(default_factory=list)


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    if mean == 0:
        return 0.0
    return float(arr.std(ddof=1) / abs(mean))


def _benford_deviation(values: list[float], min_samples: int = 20) -> dict:
    """Benford's Law analysis: MAD + Nigrini grade + chi-square + leading-digit Z-scores.

    Returns a dict (or a MAD-only float fallback if insufficient data):
        {'mad': float, 'grade': str, 'chisq': float|None, 'z_scores': dict, 'samples': int}
    Nigrini MAD grades: <0.006 合规 / 0.006-0.012 可接受 / 0.012-0.015 勉强可接受 / >0.015 不符合.
    """
    if len(values) < min_samples:
        return {'mad': 0.0, 'grade': '数据不足', 'chisq': None, 'z_scores': {}, 'samples': len(values)}
    leading = []
    for v in values:
        if v <= 0:
            continue
        first = int(str(abs(v)).lstrip('0').replace('.', '')[0])
        if 1 <= first <= 9:
            leading.append(first)
    if not leading:
        return {'mad': 0.0, 'grade': '无有效数据', 'chisq': None, 'z_scores': {}, 'samples': len(leading)}
    counts = Counter(leading)
    total = len(leading)
    benford = [np.log10(1 + 1 / d) for d in range(1, 10)]
    observed = [counts.get(d, 0) / total for d in range(1, 10)]
    mad = float(np.mean(np.abs(np.array(observed) - np.array(benford))))

    if mad < 0.006:
        grade = '合规'
    elif mad < 0.012:
        grade = '可接受'
    elif mad < 0.015:
        grade = '勉强可接受'
    else:
        grade = '不符合'

    # Chi-square test + per-digit Z-scores (Nigrini)
    chisq = None
    z_scores = {}
    if total >= min_samples:
        try:
            from scipy import stats as _s
            expected = [benford[i] * total for i in range(9)]
            observed_counts = [counts.get(d, 0) for d in range(1, 10)]
            chisq = float(_s.chisquare(observed_counts, f_exp=expected).statistic)
            for i, d in enumerate(range(1, 10)):
                exp = benford[i] * total
                if exp > 0:
                    z_scores[d] = float((observed_counts[i] - exp) / np.sqrt(exp * (1 - benford[i])))
        except Exception:
            pass

    return {'mad': mad, 'grade': grade, 'chisq': chisq, 'z_scores': z_scores, 'samples': total}


def _detect_same_rate(values: list[float], threshold: float = 0.05) -> tuple[bool, list[tuple[int, int, float]]]:
    """Detect pairs of values that are suspiciously close (same-rate bidding)."""
    flags = []
    if len(values) < 2:
        return False, flags
    arr = np.array(values, dtype=float)
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            a, b = arr[i], arr[j]
            if a == 0 and b == 0:
                continue
            denom = max(abs(a), abs(b), 1e-9)
            rel_diff = abs(a - b) / denom
            if rel_diff <= threshold:
                flags.append((i, j, rel_diff))
    return len(flags) > 0, flags


def _detect_clustering(values: list[float], min_cluster_size: int = 3, bandwidth_factor: float = 0.02) -> tuple[bool, list[list[int]]]:
    """Detect unnatural clustering using a simple density threshold."""
    if len(values) < min_cluster_size:
        return False, []
    arr = np.array(sorted(values), dtype=float)
    range_v = max(arr.max() - arr.min(), 1e-6)
    bandwidth = max(range_v * bandwidth_factor, arr.mean() * 0.005 if arr.mean() > 0 else 1e-3)
    clusters = []
    current = [0]
    for idx in range(1, len(arr)):
        if arr[idx] - arr[current[-1]] <= bandwidth:
            current.append(idx)
        else:
            if len(current) >= min_cluster_size:
                clusters.append(current[:])
            current = [idx]
    if len(current) >= min_cluster_size:
        clusters.append(current)
    return len(clusters) > 0, clusters


def _detect_abnormal_drop(prices: list[float], reference_price: float | None = None,
                          drop_threshold: float = 0.30) -> tuple[bool, list[tuple[int, float]]]:
    """Detect bids that are abnormally lower than a reference or the mean."""
    if len(prices) < 2:
        return False, []
    arr = np.array(prices, dtype=float)
    ref = reference_price if reference_price and reference_price > 0 else arr.mean()
    drops = []
    for idx, p in enumerate(arr):
        if p <= 0 or ref <= 0:
            continue
        drop = (ref - p) / ref
        if drop >= drop_threshold:
            drops.append((idx, float(drop)))
    return len(drops) > 0, drops


def _detect_progression(prices: list[float], tolerance: float = 0.02) -> tuple[bool, str, list[tuple[int, float]]]:
    """Detect 等差/等比 报价规律（定向陪标信号）。

    若 ≥3 个报价近似构成等差或等比序列（公差/公比一致），视为可疑。
    Returns: (found, progression_type('arithmetic'|'geometric'|''), flagged_indices).
    """
    arr = [p for p in prices if p and p > 0]
    n = len(arr)
    if n < 3:
        return False, '', []

    # 等差：相邻差近似一致
    diffs = [arr[i + 1] - arr[i] for i in range(n - 1)]
    if diffs and all(d > 0 for d in diffs):
        d0 = diffs[0]
        if all(abs(d - d0) / max(abs(d0), 1e-9) < tolerance for d in diffs[1:]):
            return True, 'arithmetic', list(range(n))

    # 等比：相邻比近似一致
    ratios = [arr[i + 1] / arr[i] for i in range(n - 1)]
    if ratios and all(r > 1.0001 for r in ratios):
        r0 = ratios[0]
        if all(abs(r - r0) / abs(r0) < tolerance for r in ratios[1:]):
            return True, 'geometric', list(range(n))

    return False, '', []


def _detect_tailing_digits(prices: list[float], threshold: float = 0.8, digits: int = 2) -> tuple[bool, dict]:
    """CSDN 模型第一信号：报价尾数（后 digits 位）相同比例 ≥ threshold → 串标嫌疑。

    Returns: (flag, {'rate': float, 'digit': str, 'count': int, 'total': int}).
    """
    if len(prices) < 3:
        return False, {}
    tails = []
    for p in prices:
        if p and p > 0:
            s = str(int(p))
            # 取整数部分后 digits 位（如 1234567 → '67'；1000000 → '00'）
            tails.append(s[-digits:] if len(s) >= digits else s)
    if len(tails) < 3:
        return False, {}
    digit, count = Counter(tails).most_common(1)[0]
    rate = count / len(tails)
    if rate >= threshold:
        return True, {'rate': round(rate, 2), 'digit': digit, 'count': count, 'total': len(tails)}
    return False, {}


def check_quote_anomaly(
    text: str,
    doc_name: str = "",
    reference_price: float | None = None,
    drop_threshold: float | None = None,
    same_rate_threshold: float | None = None,
    audit = None,
) -> QuoteAnomalyResult:
    """Run the full quote anomaly detector on a single bid document.

    Args:
        text: bid document text
        doc_name: document identifier for logging
        reference_price: optional known reference price
        drop_threshold: override config (None = use runtime_config)
        same_rate_threshold: override config (None = use runtime_config)
        audit: optional AuditLogger instance
    """
    cfg = _get_thresholds()
    st = same_rate_threshold if same_rate_threshold is not None else cfg['same_rate']
    dt = drop_threshold if drop_threshold is not None else cfg['drop']

    if audit:
        audit.component("quote_anomaly_extract", input_chars=len(text))

    prices = extract_prices(text)
    percentages = extract_percentages(text)
    daxie_results = parse_daxie_amount(text)

    daxie_mismatches = [d for d in daxie_results if d['match'] is False]
    if audit:
        audit.component("quote_anomaly_extract", status="OK",
                        prices_found=len(prices), pct_found=len(percentages),
                        daxie_checked=len(daxie_results), daxie_mismatches=len(daxie_mismatches))

    result = QuoteAnomalyResult(
        doc_name=doc_name, prices=prices, percentages=percentages,
        daxie_mismatches=daxie_mismatches,
    )

    if len(prices) < 2 and len(percentages) < 2:
        result.details.append("未提取到足够价格/费率数据进行异常分析")
        if audit:
            audit.component("quote_anomaly_check", status="SKIPPED",
                            reason="insufficient_data")
        return result

    values = prices if len(prices) >= 2 else percentages

    # Coefficient of variation
    result.cv = _coefficient_of_variation(values)
    if result.cv < cfg['cv_low']:
        result.details.append(f"价格离散系数极低 (CV={result.cv:.4f})，存在串通报价可能")
    elif result.cv > cfg['cv_high']:
        result.details.append(f"价格离散系数极高 (CV={result.cv:.4f})，需关注报价合理性")
    else:
        result.details.append(f"价格离散系数 CV={result.cv:.4f}")

    # Same-rate bidding
    same_rate, same_rate_pairs = _detect_same_rate(values, threshold=st)
    result.same_rate_flag = same_rate
    if same_rate:
        result.details.append(f"发现 {len(same_rate_pairs)} 组异常接近的报价/费率（相对差≤{st*100:.0f}%）")
        result.matched_prices['same_rate'] = [values[i] for i, _, _ in same_rate_pairs]

    # Abnormal drop
    drop, drops = _detect_abnormal_drop(prices, reference_price, dt)
    result.abnormal_drop_flag = drop
    if drop:
        result.details.append(f"发现 {len(drops)} 个报价异常低于基准（降幅≥{dt*100:.0f}%）")
        result.matched_prices['abnormal_drop'] = [prices[i] for i, _ in drops]

    # Clustering
    cluster, clusters = _detect_clustering(values, min_cluster_size=cfg['min_cluster'],
                                           bandwidth_factor=cfg['bandwidth'])
    result.clustering_flag = cluster
    if cluster:
        result.details.append(f"发现 {len(clusters)} 个报价/费率聚类（≥{cfg['min_cluster']}个报价密集聚集）")
        result.matched_prices['clustering'] = [values[i] for c in clusters for i in c]

    # 等差/等比 报价规律（定向陪标）
    prog, ptype, pidx = _detect_progression(prices, tolerance=cfg.get('progression_tolerance', 0.02))
    result.progression_type = ptype
    if prog:
        label = '等差' if ptype == 'arithmetic' else '等比'
        result.details.append(f"报价呈{label}规律分布（{len(pidx)} 个报价），存在定向陪标嫌疑")
        result.matched_prices['progression'] = [prices[i] for i in pidx]

    # 报价尾数相同（CSDN 第一信号：尾数相同比例 ≥80%）
    tail_flag, tail_info = _detect_tailing_digits(prices)
    result.tailing_digits_flag = tail_flag
    result.tailing_digits_info = tail_info
    if tail_flag:
        result.details.append(
            f"报价尾数高度一致（尾数 {tail_info['digit']} 占 {tail_info['rate']:.0%}，"
            f"{tail_info['count']}/{tail_info['total']}），存在围标嫌疑")

    # Benford's Law (Nigrini 分级 + 卡方 + Z 检验)
    benford_res = _benford_deviation(prices, min_samples=cfg['min_benford'])
    result.benford_deviation = benford_res['mad']
    if benford_res['grade'] in ('不符合', '勉强可接受'):
        extra = ''
        if benford_res.get('z_scores'):
            top_z = sorted(benford_res['z_scores'].items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            extra = '；显著首位: ' + ', '.join(f'{d}(Z={z:.1f})' for d, z in top_z)
        result.details.append(
            f"价格首位分布不符本福特（MAD={benford_res['mad']:.3f}，{benford_res['grade']}{extra}），建议人工复核")
    else:
        result.details.append(f"本福特偏差={benford_res['mad']:.3f}（{benford_res['grade']}）")

    # Daxie cross-validation
    if daxie_mismatches:
        result.details.append(f"发现 {len(daxie_mismatches)} 处大写金额与阿拉伯数字不一致")
        result.matched_prices['daxie_mismatch'] = [d['arabic_from_daxie'] for d in daxie_mismatches]

    # Aggregate risk score (0-100)
    score = 0.0
    if result.same_rate_flag:
        score += 35.0
    if result.abnormal_drop_flag:
        score += 30.0
    if result.clustering_flag:
        score += 20.0
    if result.progression_type:
        score += 25.0
    if result.tailing_digits_flag:
        score += 15.0
    if result.cv < cfg['cv_low']:
        score += 15.0
    if result.benford_deviation > cfg['benford']:
        score += 10.0
    if daxie_mismatches:
        score += 15.0
    result.risk_score = min(score, 100.0)

    if audit:
        audit.component("quote_anomaly_check", status="OK",
                        risk_score=round(result.risk_score, 1),
                        cv=round(result.cv, 4),
                        same_rate=result.same_rate_flag,
                        abnormal_drop=result.abnormal_drop_flag,
                        clustering=result.clustering_flag,
                        benford=round(result.benford_deviation, 3),
                        daxie_mismatches=len(daxie_mismatches))
    return result


def compare_bidders_quotes(
    file_data: list[dict],
    reference_price: float | None = None,
    drop_threshold: float | None = None,
    same_rate_threshold: float | None = None,
    audit = None,
) -> dict:
    """Compare quotes across multiple bidders and return cross-bidder anomaly signals.

    file_data: list of {'filename': str, 'text': str, ...}
    Returns aggregated dict with per-bidder results and cross-bidder signals.
    """
    if audit:
        audit.component("quote_anomaly_cross_bidder", file_count=len(file_data))

    per_bidder = []
    for fd in file_data:
        res = check_quote_anomaly(
            fd['text'], doc_name=fd['filename'],
            reference_price=reference_price,
            drop_threshold=drop_threshold,
            same_rate_threshold=same_rate_threshold,
            audit=audit,
        )
        per_bidder.append({
            'filename': fd['filename'],
            'cv': res.cv,
            'risk_score': res.risk_score,
            'same_rate_flag': res.same_rate_flag,
            'abnormal_drop_flag': res.abnormal_drop_flag,
            'clustering_flag': res.clustering_flag,
            'progression_type': res.progression_type,
            'tailing_digits_flag': res.tailing_digits_flag,
            'tailing_digits_info': res.tailing_digits_info,
            'benford_deviation': res.benford_deviation,
            'prices': res.prices[:20],
            'percentages': res.percentages[:20],
            'details': res.details,
            'daxie_mismatches': res.daxie_mismatches,
        })

    # Cross-bidder same-rate on the first/main price of each bidder
    main_prices = [pb['prices'][0] if pb['prices'] else 0 for pb in per_bidder]
    cfg = _get_thresholds()
    st = same_rate_threshold if same_rate_threshold is not None else cfg['same_rate']
    cross_same_rate, cross_pairs = _detect_same_rate(main_prices, threshold=st)
    cross_clustering, cross_clusters = _detect_clustering(main_prices, min_cluster_size=cfg['min_cluster'],
                                                          bandwidth_factor=cfg['bandwidth'])
    cross_progression, cross_prog_type, _ = _detect_progression(main_prices)
    cross_tailing, cross_tailing_info = _detect_tailing_digits(main_prices)

    result = {
        'per_bidder': per_bidder,
        'cross_same_rate': cross_same_rate,
        'cross_same_rate_pairs': cross_pairs,
        'cross_clustering': cross_clustering,
        'cross_clustering_indices': cross_clusters,
        'cross_progression': cross_progression,
        'cross_progression_type': cross_prog_type,
        'cross_tailing_digits': cross_tailing,
        'cross_tailing_info': cross_tailing_info,
        'max_risk_score': max(pb['risk_score'] for pb in per_bidder) if per_bidder else 0,
        'avg_cv': float(np.mean([pb['cv'] for pb in per_bidder])) if per_bidder else 0,
    }

    if audit:
        audit.component("quote_anomaly_cross_bidder", status="OK",
                        bidders=len(per_bidder),
                        max_risk=round(result['max_risk_score'], 1),
                        avg_cv=round(result['avg_cv'], 4),
                        cross_same_rate=cross_same_rate,
                        cross_clustering=cross_clustering)
    return result


# --- DB persistence ----------------------------------------------------------

def save_quote_anomaly_results(
    user_id: str,
    task_id: str,
    per_bidder: list[dict],
    cross_result: dict,
    project_id: int = None,
) -> int:
    """Persist quote anomaly results to the database. Returns count of rows saved."""
    saved = 0
    try:
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for pb in per_bidder:
                    cur.execute("""
                        INSERT INTO quote_anomaly_results
                            (user_id, task_id, project_id, doc_name, prices, percentages,
                             cv, same_rate_flag, abnormal_drop_flag, clustering_flag,
                             benford_deviation, risk_score, details, matched_prices,
                             cross_same_rate, cross_clustering,
                             max_cross_risk, avg_cross_cv)
                        VALUES (%s,%s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s, %s,%s, %s,%s)
                    """, (
                        user_id, task_id, project_id, pb['filename'],
                        _json.dumps(pb.get('prices', [])),
                        _json.dumps(pb.get('percentages', [])),
                        pb.get('cv', 0), pb.get('same_rate_flag', False),
                        pb.get('abnormal_drop_flag', False), pb.get('clustering_flag', False),
                        pb.get('benford_deviation', 0), pb.get('risk_score', 0),
                        _json.dumps(pb.get('details', []), ensure_ascii=False),
                        _json.dumps(pb.get('matched_prices', {})),
                        cross_result.get('cross_same_rate', False),
                        cross_result.get('cross_clustering', False),
                        cross_result.get('max_risk_score', 0),
                        cross_result.get('avg_cv', 0),
                    ))
                    saved += 1
                conn.commit()
        logger.info(f"Saved {saved} quote anomaly results for task {task_id}")
    except Exception as e:
        logger.error(f"Failed to save quote anomaly results: {e}", exc_info=True)
    return saved
