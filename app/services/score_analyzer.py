"""专家评分统计分析器 — 用于激活专家类围串标指标。

提供:
  - grubbs_test: Grubbs 离群检验（单变量离群值识别）
  - kendall_w: Kendall 一致性系数 W（评委间一致性，0=随机, 1=完全一致）
  - spearman_rank: Spearman 秩相关（两评委打分关联）
  - panel_outlier_scores: 对专家×单位打分矩阵做逐行/逐列偏离分析

纯 numpy/scipy 实现，无外部数据依赖。
"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def grubbs_test(values: list[float], alpha: float = 0.05) -> Optional[int]:
    """Grubbs 离群检验（双侧）。返回离群值索引；无离群返回 None。

    G = max|x - mean| / std；临界值按 t 分布计算。
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 3:
        return None
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std < 1e-9:
        return None
    devs = np.abs(arr - mean)
    idx = int(devs.argmax())
    g = devs[idx] / std
    try:
        from scipy import stats as _s
        # 双侧临界值
        t_crit = _s.t.ppf(1 - alpha / (2 * n), n - 2)
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit ** 2 / (n - 2 + t_crit ** 2))
    except Exception:
        # 保守近似
        g_crit = (n - 1) / np.sqrt(n) * np.sqrt(3.0 / (n - 2 + 3.0))
    return idx if g > g_crit else None


def kendall_w(matrix: list[list[float]]) -> Optional[float]:
    """Kendall 一致性系数 W（评委×对象 打分矩阵）。

    W = 12·Σ(rank_sum - mean_rank)² / (m²·(n³ - n))
    W 接近 1 → 评委打分高度一致（可能抱团/事先串通）。
    matrix: list of rows, each row = one 评委's scores for the objects.
    """
    m = len(matrix)  # 评委数
    if m < 2:
        return None
    n = len(matrix[0]) if matrix and matrix[0] else 0
    if n < 2:
        return None
    # 每评委内部秩
    ranks = []
    for row in matrix:
        arr = np.asarray(row, dtype=float)
        # 处理并列（用平均秩）
        order = arr.argsort()
        r = np.empty(n)
        r[order] = np.arange(1, n + 1)
        # 并列处理
        sorted_vals = np.sort(arr)
        for v in set(arr.tolist()):
            idxs = np.where(arr == v)[0]
            if len(idxs) > 1:
                avg = float(r[idxs[0]])  # 并列位置的平均秩
                # 实际平均
                positions = np.where(sorted_vals == v)[0]
                avg = float(np.mean(positions + 1))
                r[idxs] = avg
        ranks.append(r)
    R = np.sum(np.vstack(ranks), axis=0)  # 每对象秩和
    denom = m ** 2 * (n ** 3 - n)
    if denom == 0:
        return None
    W = 12 * np.sum((R - m * (n + 1) / 2) ** 2) / denom
    return float(np.clip(W, 0.0, 1.0))


def spearman_rank(a: list[float], b: list[float]) -> Optional[float]:
    """Spearman 秩相关系数（两评委打分是否高度相关）。"""
    if len(a) < 3 or len(b) < 3 or len(a) != len(b):
        return None
    try:
        from scipy import stats as _s
        rho, _ = _s.spearmanr(a, b)
        return float(rho)
    except Exception:
        return None


def panel_outlier_scores(panel: dict) -> dict:
    """分析 专家×单位 打分面板，返回离群与一致性信号。

    panel: {'experts': {expert_name: {bidder: score}}, ...} 或
           {'matrix': [[...]], 'experts': [names], 'bidders': [names]}
    Returns: {'outliers': [...], 'kendall_w': float|None, 'per_expert_dev': [...],
              'signals': [str]}
    """
    out = {'outliers': [], 'kendall_w': None, 'per_expert_dev': [], 'signals': []}
    matrix = panel.get('matrix')
    experts = panel.get('experts') or []
    bidders = panel.get('bidders') or []
    if not matrix or len(matrix) < 2:
        return out

    # 每位评委对各单位打分 → 离群（评委内部偏离）
    for ri, row in enumerate(matrix):
        vals = [float(x) for x in row if x is not None]
        if len(vals) < 2:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals)) if len(vals) > 1 else 0
        for ci, v in enumerate(row):
            if v is None:
                continue
            dev = abs(float(v) - mean)
            if std > 1e-9 and dev > max(8.0, 1.5 * std):
                out['per_expert_dev'].append({
                    'expert': experts[ri] if ri < len(experts) else f'评委{ri + 1}',
                    'bidder': bidders[ci] if ci < len(bidders) else f'单位{ci + 1}',
                    'score': float(v), 'avg': round(mean, 1), 'dev': round(dev, 1),
                })

    # Kendall W（评委一致性）
    try:
        w = kendall_w(matrix)
        if w is not None:
            out['kendall_w'] = w
            if w >= 0.8:
                out['signals'].append(f"评委打分一致性极高 (W={w:.2f})，可能存在抱团/串通")
            elif w <= 0.3:
                out['signals'].append(f"评委打分一致性极低 (W={w:.2f})，评分标准可能不统一")
    except Exception as e:
        logger.debug(f"kendall_w failed: {e}")

    # Grubbs 离群（评委×单位扁平列表）
    flat = [float(x) for row in matrix for x in row if x is not None]
    try:
        gidx = grubbs_test(flat)
        if gidx is not None:
            out['outliers'].append({'index': gidx, 'value': flat[gidx]})
    except Exception as e:
        logger.debug(f"grubbs failed: {e}")

    return out
