"""开标信息表 + 评审标准解析与指标计算。

为清标补充两类结构化输入：
  1. 开标信息表 (bid-opening info) — Excel/CSV/JSON，中文表头自动映射到标准列。
     IP/文件码/加密锁 等交易平台列解析后忽略（技术不可行，见 FIX-2026-08-28-009）。
  2. 评审标准 (evaluation criteria) — 从招标文件自动提取（预算价/计划开标时间/评标办法/
     评分点/客观分规则），供预览与人工确认。

同时实现基于这两类数据激活的指标计算（OPEN_INFO + TENDER checker）。
"""
import json
import logging
import os
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 标准开标信息表 15 列（对齐行业模板）────────────────────────────
# IP/文件码/加密锁 列为交易平台字段，解析后忽略（不可行）
STD_COLUMNS = [
    '序号', '开标时间', '投标单位', '联系人', '联系电话',
    '文件下载IP', '标书上传时间', '标书上传IP', '解密状态', '解密IP',
    '文件码', '加密锁', '报价方式', '投标报价', '备注',
]
# 有意义（非平台）列 → 内部字段
MAP_COLUMNS = {
    '序号': 'seq', '开标时间': 'open_time', '投标单位': 'bidder',
    '联系人': 'contact', '联系电话': 'phone',
    '报价方式': 'price_mode', '投标报价': 'bid_price',
    '备注': 'remark',
}
# 可选的扩展列（评审/中标信息，若开标表提供）
EXTRA_COLUMNS = {
    '中标单位': 'winner', '中标金额': 'win_amount', '评标得分': 'score',
    '废标标记': 'waste_flag', '专家': 'expert', '专家打分': 'expert_score',
}

_PHONE_RE = re.compile(r'1[3-9]\d{9}')
_PRICE_RE = re.compile(r'(\d[\d,]*\.?\d*)\s*(万元|元|万)?')


def _to_number(v) -> Optional[float]:
    """宽松数字转换（支持 1,234.5 / 1500万元 / 壹佰贰拾万元）。"""
    if v is None:
        return None
    s = str(v).strip().replace(',', '')
    if not s:
        return None
    # 中文大写金额
    try:
        from app.utils.chinese_numbers import cn_to_arabic
        arab = cn_to_arabic(s)
        if arab is not None and not re.fullmatch(r'[\d.,]+', s):
            # cn_to_arabic 对纯阿拉伯串会直接返回 float
            return arab
    except Exception:
        pass
    m = _PRICE_RE.search(s)
    if m:
        try:
            val = float(m.group(1))
            if '万' in s:
                val *= 10000
            return val
        except ValueError:
            return None
    return None


def parse_open_info_file(abs_path: str, filename: str = '') -> dict:
    """解析开标信息表（Excel/CSV/JSON）为结构化 dict。

    Returns: {'rows': [{...}], 'parsed': bool, 'error': str, 'notes': [str]}
    """
    ext = (filename or abs_path).lower().rsplit('.', 1)[-1]
    rows = []
    try:
        if ext in ('json',):
            with open(abs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            rows = data.get('rows', data if isinstance(data, list) else [])
        elif ext in ('csv',):
            import csv
            with open(abs_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        elif ext in ('xlsx', 'xls'):
            import openpyxl
            wb = openpyxl.load_workbook(abs_path, read_only=True, data_only=True)
            ws = wb.active
            header = None
            for r_i, row in enumerate(ws.iter_rows(values_only=True)):
                if r_i == 0:
                    header = [str(c).strip() if c else '' for c in row]
                    continue
                rows.append(dict(zip(header, [c for c in row])))
        else:
            return {'rows': [], 'parsed': False, 'error': f'不支持的文件类型: .{ext}', 'notes': []}
    except Exception as e:
        logger.warning(f"开标信息表解析失败: {e}")
        return {'rows': [], 'parsed': False, 'error': str(e), 'notes': []}

    # 表头映射：中文列 → 内部字段
    normalized = []
    notes = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        rec = {}
        for k, v in raw.items():
            kk = str(k).strip()
            if kk in MAP_COLUMNS:
                rec[MAP_COLUMNS[kk]] = v
            elif kk in EXTRA_COLUMNS:
                rec[EXTRA_COLUMNS[kk]] = v
        if rec.get('bidder'):
            normalized.append(rec)
    if not normalized:
        notes.append('未解析到有效投标单位行，请检查表头（投标单位/联系人/投标报价等中文列名）')
    return {'rows': normalized, 'parsed': True, 'error': '', 'notes': notes}


def build_bidder_map(open_info: Optional[dict]) -> dict:
    """按投标单位名（模糊匹配）索引开标信息行。"""
    out = {}
    if not open_info or not open_info.get('rows'):
        return out
    for r in open_info['rows']:
        name = str(r.get('bidder', '')).strip()
        if name:
            out.setdefault(name, r)
    return out


# ── 评审标准提取（从招标文件）──────────────────────────────────────

def extract_eval_criteria(tender_text: Optional[str]) -> dict:
    """从招标文件自动提取评审标准，供预览 + 人工确认。

    Returns: {'budget_price': float|None, 'plan_open_time': str|None,
              'eval_method': str|None, 'score_points': [str],
              'objective_rules': [str], 'confidence': str, 'raw': {}}
    """
    res = {
        'budget_price': None, 'plan_open_time': None, 'eval_method': None,
        'score_points': [], 'objective_rules': [], 'confidence': 'low', 'raw': {},
    }
    if not tender_text:
        return res
    text = tender_text

    # 预算价/控制价/最高限价
    for pat in [r'(?:预算价|最高限价|控制价|招标控制价|最高投标限价)\s*[:：]?\s*([\d,，]+\.?\d*)\s*(万元|元|万)?',
                r'(?:最高投标限价|控制价)\s*[为是]\s*([\d,，]+\.?\d*)\s*(万元)?']:
        m = re.search(pat, text)
        if m:
            try:
                val = float(m.group(1).replace(',', '').replace('，', ''))
                if m.group(2) and '万' in m.group(2):
                    val *= 10000
                res['budget_price'] = val
                break
            except ValueError:
                pass

    # 计划开标时间
    m = re.search(r'(?:开标时间|投标截止时间|递交截止时间)\s*[:：]?\s*(\d{4}[-年]\d{1,2}[-月]\d{1,2}日?)', text)
    if m:
        res['plan_open_time'] = m.group(1)

    # 评标办法
    for meth in ['综合评估法', '综合评分法', '最低评标价法', '经评审的最低投标价法',
                 '性价比法', '合理低价法', '两阶段评标法']:
        if meth in text:
            res['eval_method'] = meth
            break

    # 评分点/客观分规则
    score_pats = [
        r'([\u4e00-\u9fa5]{2,12}（?\d{1,3}分）?)',
        r'(?:技术分|商务分|价格分|客观分|主观分)[：:]?\s*(\d{1,3})\s*分',
    ]
    for pat in score_pats:
        res['score_points'] = list(dict.fromkeys(re.findall(pat, text)))[:10]
        if res['score_points']:
            break
    res['objective_rules'] = re.findall(r'[^\n。；;]{0,40}(?:客观分|评分标准|得分规则|评分办法)[^\n。；;]{0,60}', text)[:10]

    hits = sum(1 for v in [res['budget_price'], res['plan_open_time'], res['eval_method']] if v)
    res['confidence'] = 'high' if hits >= 2 else ('medium' if hits == 1 else 'low')
    res['raw'] = {'hits': hits}
    return res


# ── OPEN_INFO + TENDER 指标计算 ─────────────────────────────────────

def compute_open_info_indicators(file_data, open_info, eval_criteria):
    """为 OPEN_INFO / TENDER 类指标计算得分与详情。

    Returns: dict { indicator_id: {'score': float, 'result': str, 'details': [dict]} }
    """
    out = {}
    bidders = build_bidder_map(open_info)
    filenames = [fd.get('filename', '') for fd in file_data]

    def _lookup(fname):
        # 精确或后缀模糊匹配开标信息行的投标单位
        for name, row in bidders.items():
            if fname == name or fname.rstrip('.docx') in name or name in fname:
                return row
        return None

    rows = (open_info or {}).get('rows', [])

    # ── 联系电话跨单位雷同 (contact_phone_abnormal) ──
    phones = {}
    for r in rows:
        ph = re.sub(r'\D', '', str(r.get('phone', '') or ''))
        if _PHONE_RE.fullmatch(ph):
            phones.setdefault(ph, []).append(str(r.get('bidder', '')))
    dup_phones = {p: names for p, names in phones.items() if len(names) >= 2}
    if dup_phones:
        details = [{'phone': p, 'bidders': ', '.join(n[:20])} for p, n in dup_phones.items()]
        out['contact_phone_abnormal'] = {
            'score': min(10 + 5 * len(details), 30),
            'result': f"▲ 发现 {len(details)} 组不同投标单位共用同一联系电话，存在围串标嫌疑。",
            'details': details,
        }
    else:
        out['contact_phone_abnormal'] = {
            'score': 0, 'result': '√ 未发现不同投标单位联系电话雷同。', 'details': []}

    # ── 联系人跨单位雷同 (contact_person_same / cross_contact_same) ──
    contacts = {}
    for r in rows:
        c = str(r.get('contact', '') or '').strip()
        if c:
            contacts.setdefault(c, []).append(str(r.get('bidder', '')))
    dup_contacts = {c: names for c, names in contacts.items() if len(names) >= 2}
    if dup_contacts:
        details = [{'contact': c, 'bidders': ', '.join(n[:20])} for c, n in dup_contacts.items()]
        out['contact_person_same'] = {
            'score': min(15 + 5 * len(details), 40),
            'result': f"▲ 发现 {len(details)} 组不同投标单位使用同一联系人。",
            'details': details,
        }
        out['cross_contact_same'] = out['contact_person_same']
    else:
        out['contact_person_same'] = {'score': 0, 'result': '√ 未发现联系人雷同。', 'details': []}
        out['cross_contact_same'] = {'score': 0, 'result': '√ 未发现跨标段联系人雷同。', 'details': []}

    # ── 中标/评标相关 (candidate_give_up) ──
    winners = [r for r in rows if str(r.get('winner', '') or '').strip()]
    give_ups = []
    for r in winners:
        # 有中标单位且备注/列标记放弃 → 嫌疑
        if re.search(r'放弃|弃标', str(r.get('remark', '') or '')):
            give_ups.append(r)
    if give_ups:
        details = [{'bidder': str(r.get('bidder', '')), 'remark': str(r.get('remark', ''))} for r in give_ups]
        out['candidate_give_up'] = {
            'score': min(len(details) * 15, 45),
            'result': f"▲ 发现 {len(details)} 个中标候选人放弃中标。",
            'details': details,
        }
    else:
        out['candidate_give_up'] = {'score': 0, 'result': '√ 未发现中标候选人放弃中标。', 'details': []}

    # ── 专家打分明细相关（若开标表含专家/打分列）──
    expert_rows = [r for r in rows if str(r.get('expert', '') or '').strip() or str(r.get('expert_score', '') or '').strip()]
    if expert_rows:
        scores = []
        for r in expert_rows:
            try:
                s = float(str(r.get('expert_score', '') or '0').replace(',', ''))
                scores.append((str(r.get('bidder', '')), str(r.get('expert', '')), s))
            except ValueError:
                pass
        if len(scores) >= 2:
            vals = [s for _, _, s in scores]
            mean = sum(vals) / len(vals)
            devs = [(b, e, s, abs(s - mean)) for b, e, s in scores]
            outliers = [d for d in devs if d[3] > max(10, mean * 0.15)]
            # subjective_expert_units: 同一专家对各单位打分偏离
            by_expert = {}
            for b, e, s in scores:
                by_expert.setdefault(e, []).append((b, s))
            for e, lst in by_expert.items():
                if len(lst) >= 2:
                    vs = [s for _, s in lst]
                    em = sum(vs) / len(vs)
                    for b, s in lst:
                        if abs(s - em) > max(10, em * 0.15):
                            out.setdefault('subjective_expert_units', {
                                'score': 0, 'result': '', 'details': []})
                            out['subjective_expert_units']['details'].append(
                                {'expert': e, 'bidder': b, 'score': s, 'avg': round(em, 1),
                                 'deviation': round(s - em, 1)})
            if outliers:
                out['expert_deviation_abnormal'] = {
                    'score': min(len(outliers) * 8, 35),
                    'result': f"▲ 发现 {len(outliers)} 个专家打分显著偏离均值。",
                    'details': [{'bidder': b, 'expert': e, 'score': s, 'dev': round(d, 1)}
                                for b, e, s, d in outliers[:10]],
                }
            else:
                out['expert_deviation_abnormal'] = {'score': 0, 'result': '√ 专家打分未见显著偏离。', 'details': []}
            if 'subjective_expert_units' in out:
                out['subjective_expert_units']['score'] = min(len(out['subjective_expert_units']['details']) * 8, 35)
                out['subjective_expert_units']['result'] = (
                    f"▲ 同一专家对不同单位打分偏离异常（{len(out['subjective_expert_units']['details'])} 处）。")
        else:
            for k in ('expert_deviation_abnormal', 'subjective_expert_units'):
                out[k] = {'score': 0, 'result': '√ 专家打分数据不足，未发现异常。', 'details': []}

    # ── 技术标/商务标得分异常 (tech/commercial_score_abnormal) ──
    for key, col in (('tech_score_abnormal', 'tech_score'), ('commercial_score_abnormal', 'com_score')):
        scored = []
        for r in rows:
            v = r.get(col)
            if v is not None:
                try:
                    scored.append((str(r.get('bidder', '')), float(str(v).replace(',', ''))))
                except ValueError:
                    pass
        if len(scored) >= 2:
            vals = [s for _, s in scored]
            mean = sum(vals) / len(vals)
            outliers = [(b, s) for b, s in scored if abs(s - mean) > max(10, mean * 0.2)]
            if outliers:
                out[key] = {
                    'score': min(len(outliers) * 8, 35),
                    'result': f"▲ 发现 {len(outliers)} 个单位得分异常偏离均值。",
                    'details': [{'bidder': b, 'score': s, 'avg': round(mean, 1)} for b, s in outliers[:10]],
                }
            else:
                out[key] = {'score': 0, 'result': '√ 未发现得分异常。', 'details': []}

    # ── 废标率 (waste_rate_abnormal) ──
    if rows:
        total = len(rows)
        wasted = [r for r in rows if re.search(r'废标|无效', str(r.get('waste_flag', '') or '') or str(r.get('remark', '') or ''))]
        if wasted:
            rate = len(wasted) / total
            out['waste_rate_abnormal'] = {
                'score': min(rate * 100, 40),
                'result': f"▲ 废标率 {rate:.0%}（{len(wasted)}/{total}），异常偏高。",
                'details': [{'bidder': str(r.get('bidder', '')), 'flag': str(r.get('waste_flag', ''))} for r in wasted[:10]],
            }
        else:
            out['waste_rate_abnormal'] = {'score': 0, 'result': '√ 未发现废标异常。', 'details': []}

    # ── 投标单位数异常 (bidder_count_abnormal, TENDER: 标底价) ──
    if eval_criteria and eval_criteria.get('budget_price'):
        budget = eval_criteria['budget_price']
        prices = []
        for r in rows:
            p = _to_number(r.get('bid_price'))
            if p:
                prices.append((str(r.get('bidder', '')), p))
        if len(prices) >= 2:
            mean_price = sum(p for _, p in prices) / len(prices)
            dev = (mean_price - budget) / budget if budget else 0
            if abs(dev) > 0.15:
                out['bidder_count_abnormal'] = {
                    'score': min(abs(dev) * 100, 40),
                    'result': f"▲ 投标均价 {mean_price / 10000:.1f} 万元与标底 {budget / 10000:.1f} 万元偏离 {dev:.1%}。",
                    'details': [{'bidder': b, 'price': round(p, 0)} for b, p in prices[:10]],
                }
            else:
                out['bidder_count_abnormal'] = {'score': 0, 'result': '√ 投标价格与标底价基本一致。', 'details': []}

    # ── 招标延期异常 (extension_abnormal, TENDER: 计划 vs 实际开标) ──
    if eval_criteria and eval_criteria.get('plan_open_time'):
        plan = eval_criteria['plan_open_time']
        actuals = [r.get('open_time') for r in rows if r.get('open_time')]
        if actuals and all(str(a).strip() == str(actuals[0]).strip() for a in actuals):
            # 简化：若所有实际开标时间一致且晚于计划，标记延期（按日比较）
            a = str(actuals[0])
            if plan[:10] in ('',) or plan[:10] not in a:
                out['extension_abnormal'] = {
                    'score': 10,
                    'result': f"▲ 实际开标时间 {a} 与计划 {plan} 不一致，存在延期可能。",
                    'details': [{'plan': plan, 'actual': a}],
                }
            else:
                out['extension_abnormal'] = {'score': 0, 'result': '√ 开标时间与计划一致。', 'details': []}

    # ── 客观分异常 (objective_score_abnormal, TENDER: 评标办法) ──
    if eval_criteria and eval_criteria.get('objective_rules') and rows:
        # 有客观分规则且无法从开标表验证 → 中等提示（数据不足）
        out['objective_score_abnormal'] = {
            'score': 0,
            'result': '○ 已提取评标办法/评分规则，但开标信息表未提供客观分数据，无法判定的部分保留人工复核。',
            'details': [{'rule': r} for r in eval_criteria['objective_rules'][:5]],
        }

    # ── 其他 OPEN_INFO 指标：数据不足时给出明确占位 ──
    placeholder_open = {
        'subjective_expert_spread': '需专家打分明细（开标信息表未提供）',
        'specific_expert_score': '需专家打分明细（开标信息表未提供）',
        'clique_expert_scoring': '需多评委打分矩阵（开标信息表未提供）',
        'clique_expert_consistency': '需多评委打分矩阵（开标信息表未提供）',
    }
    for k, note in placeholder_open.items():
        if k not in out:
            out[k] = {'score': 0, 'result': f'○ {note}。', 'details': []}

    return out


def quote_with_open_price(file_data, open_info, reference_price=None):
    """优先用开标信息表投标报价列作为各投标人总价，回填到 file_data 文本。

    供 compare_bidders_quotes 使用（避免 extract_prices 取 prices[0] 的脆弱性）。
    """
    if not open_info or not open_info.get('rows'):
        return file_data, reference_price
    bidder_map = build_bidder_map(open_info)
    out = list(file_data)
    prices = []
    for i, fd in enumerate(out):
        row = None
        for name, r in bidder_map.items():
            fname = fd.get('filename', '')
            if fname == name or name in fname:
                row = r
                break
        if row:
            p = _to_number(row.get('bid_price'))
            if p:
                prices.append(p)
                # 在文本末尾追加权威报价标记（供 quote_anomaly 提取）
                fd = dict(fd)
                fd['text'] = (fd.get('text', '') + f"\n【开标报价】{p:.2f}元")
                out[i] = fd
    # 若开标表提供了多单位报价，用中位数作为参考价（更稳）
    if reference_price is None and len(prices) >= 2:
        sp = sorted(prices)
        reference_price = sp[len(sp) // 2]
    return out, reference_price
