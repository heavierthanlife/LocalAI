"""铁证双层判定 + 暗标违规独立警示 (FIX-2026-09-09-QA).

设计决策（与用户确认，agentmemory: clearance-hard-evidence）：
- 铁证信号**不参与**加权复合指数打分（软嫌疑度指数原样保留），独立成
  "硬警报"判定层：veto 只提升展示级别（warning_level），不重写指数。
- T1 确认级：单命中即 veto → ``■ 高度预警（铁证触发）``
  · `lastModifiedBy` 同人（guard 排除 Administrator/User/微软用户等通用值）
  · 平台加密锁 / 平台文件码雷同（仅交易平台来源；正文 text_sim 路径不算）
  · 联系人 + 电话同组双命中（两家文件同时共享同一联系人和同一手机号）
  · 段落雷同升级：同对共享 ≥2 段非模板实质段，或同一段被 ≥3 家共享
- T2 强嫌疑：需 ≥2 类共证才 veto
  · author 雷同（guard 通用值）· 上传/解密 IP 同 · 段落单段共享
- 暗标违规（tech_seal 泄露）**独立轨道**：单家违规非串通证据，
  触发升 ``■ 高度预警（暗标违规）``，不进串通铁证。
  展示：铁证与违规可各自独立出现；同时触发时串标铁证优先、违规附加。
"""
from __future__ import annotations

HARD_LABEL_COLLUSION = '■ 高度预警（铁证触发）'
HARD_LABEL_VIOLATION = '■ 高度预警（暗标违规）'

# 交易平台信号指标 id（仅平台来源的才算铁证；正文 text_sim 路径不算）
PLATFORM_INDICATORS = {
    'bid_ip_same', 'decrypt_ip_same', 'download_ip_same',
    'same_file_code', 'same_dongle',
}

# lastModifiedBy / author 通用值 guard（厂商默认、本机名、工具名——两家的
# Office 默认 author 相同是常见误报源，绝不可当铁证）
GENERIC_METADATA_VALUES = frozenset(
    s.strip().lower() for s in [
        '', 'administrator', 'admin', 'user', 'users', 'default', 'default user',
        'owner', 'pc', 'desktop', 'laptop', 'computer', 'company',
        'microsoft', 'microsoft office user', 'office user', 'windows user',
        'local user', '本地用户', '微软用户',
        'lenovo', '联想', 'dell', 'dell inc', 'hp', 'hp customer',
        'acer', 'asus', 'toshiba', 'samsung', 'sony', 'fujitsu',
        'wps office', 'wps', 'kingsoft', '金山', 'wps文字',
        'kaifa', '开发', '研发', '管理员',
    ]
)


def _clean_meta(value) -> str:
    return str(value or '').strip().lower()


def _is_generic_meta(value) -> bool:
    return _clean_meta(value) in GENERIC_METADATA_VALUES


def _file_pairs(files) -> list[tuple[str, str]]:
    fs = sorted(set(files))
    return [(fs[a], fs[b]) for a in range(len(fs)) for b in range(a + 1, len(fs))]


def _dedup_items(items: list[dict]) -> list[dict]:
    """按 (type, frozenset(files)) 去重，保留更高级别。"""
    best: dict[tuple, dict] = {}
    order = []
    for it in items:
        key = (it['type'], frozenset(it['files']))
        if key not in best:
            best[key] = dict(it)
            order.append(key)
        else:
            if it['level'] == 'T1' and best[key]['level'] != 'T1':
                best[key] = dict(it)
    return [best[k] for k in order]


def assess_hard_evidence(indicators: list[dict], ctx: dict | None = None) -> dict:
    """铁证判定（报告级）。

    indicators: run_analysis 产出的指标列表（本函数实际不依赖，保留签名对齐）。
    ctx: 判定上下文 dict，可选键：
      lasteditor_groups: {lastModifiedBy 值: [filename, ...]}   (file_attr checker)
      author_groups:     {author 值: [filename, ...]}
      contact_data:      compare_contacts() 输出（contact_person_same /
                         contact_phone_abnormal 的 details）
      platform:          {indicator_id: {'score', 'details'}}   (open_info 平台信号)
      seal_data:         {'results': {filename: {'leak','evidence'}}}  (tech_seal)
      paragraph_collusion: detect_shared_substantive_segments() 输出

    Returns:
      {'fired': bool, 'level': 'T1'|'T2'|None, 'items': [...],
       'label': str|None, 'violation_fired': bool, 'violations': [...],
       'violation_label': str|None}
    """
    ctx = ctx or {}
    items: list[dict] = []
    violations: list[dict] = []

    # ── T1: lastModifiedBy 同人（"同一人/同机做两家标书"硬信号，guard 通用值）──
    for le, files in (ctx.get('lasteditor_groups') or {}).items():
        files = list(dict.fromkeys(files))
        if len(files) < 2 or _is_generic_meta(le):
            continue
        items.append({
            'type': 'lasteditor_same',
            'level': 'T1',
            'evidence': f'最后编辑人「{le}」同时出现在 {len(files)} 家投标文件'
                        f'（同一人/同机编制多份标书）',
            'files': sorted(set(files)),
        })

    # ── T1: 联系人 + 电话同组双命中 ──
    contact_data = ctx.get('contact_data') or {}
    person_pairs: dict[tuple, str] = {}
    phone_pairs: set[tuple] = set()
    for d in ((contact_data.get('contact_person_same') or {}).get('details') or []):
        files = [f.strip() for f in str(d.get('files', '')).split(',') if f.strip()]
        for a, b in _file_pairs(files):
            person_pairs[(a, b)] = d.get('contact', '?')
    for d in ((contact_data.get('contact_phone_abnormal') or {}).get('details') or []):
        files = [f.strip() for f in str(d.get('files', '')).split(',') if f.strip()]
        phone_pairs.update(_file_pairs(files))
    for a, b in sorted(person_pairs.keys() & phone_pairs):
        items.append({
            'type': 'contact_phone_double',
            'level': 'T1',
            'evidence': f'联系人「{person_pairs[(a, b)]}」与同一手机号同时出现在 {a} 与 {b}'
                        f'（同一自然人代理两家投标人）',
            'files': sorted({a, b}),
        })

    # ── T1: 平台加密锁 / 文件码雷同（仅平台来源）──
    platform = ctx.get('platform') or {}
    for ind_id, label in (('same_dongle', '加密锁'), ('same_file_code', '文件码')):
        res = platform.get(ind_id)
        if not res or not res.get('score'):
            continue
        for d in (res.get('details') or []):
            bidders = [b.strip() for b in str(d.get('bidders', '')).split(',') if b.strip()]
            if len(bidders) < 2:
                continue
            val = d.get('加密锁') or d.get('文件码') or d.get(label) or ''
            items.append({
                'type': f'platform_{ind_id}',
                'level': 'T1',
                'evidence': f'交易平台记录：不同投标单位共用同一{label}「{val}」',
                'files': sorted(set(bidders)),
            })

    # ── T2: author 雷同（guard 通用值）──
    for a, files in (ctx.get('author_groups') or {}).items():
        files = list(dict.fromkeys(files))
        if len(files) < 2 or _is_generic_meta(a):
            continue
        items.append({
            'type': 'author_same',
            'level': 'T2',
            'evidence': f'文件作者/编制人「{a}」同时出现在 {len(files)} 家投标文件',
            'files': sorted(set(files)),
        })

    # ── T2/T1: 段落级实质雷同（数据来自横向层 detect_shared_substantive_segments）──
    pc = ctx.get('paragraph_collusion') or {}
    segs = pc.get('shared_segments') or []
    seg_by_pair: dict[tuple, list[dict]] = {}
    for seg in segs:
        seg_files = sorted(set(seg.get('files') or []))
        if len(seg_files) < 2:
            continue
        for a, b in _file_pairs(seg_files):
            seg_by_pair.setdefault((a, b), []).append(seg)
    for (a, b), pair_segs in seg_by_pair.items():
        snippet = (pair_segs[0].get('segment_text') or '').strip()[:40]
        n_seg = len(pair_segs)
        if n_seg >= 2:
            items.append({
                'type': 'paragraph_collusion',
                'level': 'T1',
                'evidence': f'{a} 与 {b} 共享 {n_seg} 段非模板实质段'
                            f'（如「{snippet}…」），逐字雷同',
                'files': sorted({a, b}),
            })
        else:
            items.append({
                'type': 'paragraph_collusion',
                'level': 'T2',
                'evidence': f'{a} 与 {b} 共享实质段「{snippet}…」',
                'files': sorted({a, b}),
            })
    # 同一实质段被 ≥3 家共享 → 升级 T1
    for seg in segs:
        seg_files = sorted(set(seg.get('files') or []))
        if len(seg_files) < 3:
            continue
        snippet = (seg.get('segment_text') or '').strip()[:40]
        items.append({
            'type': 'paragraph_multi_bidder',
            'level': 'T1',
            'evidence': f'实质段「{snippet}…」被 {len(seg_files)} 家投标文件共享',
            'files': seg_files,
        })

    # ── T2: 上传/解密 IP 同（平台来源）──
    for ind_id, label in (('bid_ip_same', '上传IP'), ('decrypt_ip_same', '解密IP')):
        res = platform.get(ind_id)
        if not res or not res.get('score'):
            continue
        for d in (res.get('details') or []):
            bidders = [b.strip() for b in str(d.get('bidders', '')).split(',') if b.strip()]
            if len(bidders) < 2:
                continue
            val = d.get('上传IP') or d.get('解密IP') or d.get(label) or ''
            items.append({
                'type': f'platform_{ind_id}',
                'level': 'T2',
                'evidence': f'交易平台记录：不同投标单位共用同一{label}「{val}」',
                'files': sorted(set(bidders)),
            })

    items = _dedup_items(items)
    t1 = [it for it in items if it['level'] == 'T1']
    t2_types = {it['type'] for it in items if it['level'] == 'T2'}

    # T1 单命中即 veto；T2 需 ≥2 类共证
    fired = bool(t1) or len(t2_types) >= 2
    level = 'T1' if t1 else ('T2' if len(t2_types) >= 2 else None)

    # ── 暗标违规（独立轨道，不进串通铁证）──
    seal_data = ctx.get('seal_data') or {}
    for fname, r in (seal_data.get('results') or {}).items():
        if not r.get('leak'):
            continue
        ev = '；'.join(r.get('evidence', []) or [])[:200]
        violations.append({
            'type': 'tech_seal_leak',
            'evidence': ev or f'「{fname}」技术标含可识别身份信息',
            'files': [fname],
        })

    return {
        'fired': fired,
        'level': level,
        'items': items,
        'label': HARD_LABEL_COLLUSION if fired else None,
        'violation_fired': bool(violations),
        'violations': violations,
        'violation_label': HARD_LABEL_VIOLATION if violations else None,
    }