"""清标指标定义（46 项，四分组：触发/核心/基础/扩展）。

结构对齐参考报告《串通投标线索分析报告.docx》3.1-3.4 四节。
每个指标带 rule_ref（法规索引）与 rule_text（法规原文，来自 data/laws/clean/）。
rule_text 通过 load_law_text() 按 法规key+条文号 从清洗后的法规库实时读取，
保证报告中引用的规则文本可溯源到官方法规文件。
"""
import re
from pathlib import Path

from app.config import to_rel_path, resolve_path

_LAW_TEXT_CACHE = {}


def _law_clean_path(key: str) -> Path:
    rel = f'data/laws/clean/{key}.md'
    return Path(resolve_path(rel)) if not __import__('os').path.isabs(rel) else Path(rel)


def load_law_text(key: str) -> str:
    """Return full plain text of a cleaned law corpus file (stripped of frontmatter)."""
    if key in _LAW_TEXT_CACHE:
        return _LAW_TEXT_CACHE[key]
    path = _law_clean_path(key)
    if not path.exists():
        _LAW_TEXT_CACHE[key] = ''
        return ''
    text = path.read_text(encoding='utf-8')
    # strip yaml frontmatter
    if text.startswith('---'):
        end = text.find('\n---', 3)
        if end != -1:
            text = text[end + 4:]
    _LAW_TEXT_CACHE[key] = text
    return text


def rule_text(key: str, article: str) -> str:
    """Extract the text of a specific 第X条 from a law corpus file.

    Falls back to the raw article marker itself if the law text is unavailable
    (e.g. network-only source not yet archived).
    """
    body = load_law_text(key)
    if not body:
        return f'（{article} 原文待补充，引用法规：{key}）'
    # normalise spacing for matching
    compact = re.sub(r'\s+', '', body)
    m = re.search(re.escape(article), compact)
    if not m:
        return f'（{article} 未在法规库中检索到，引用法规：{key}）'
    start = m.start()
    # next article marker after this one
    nxt = re.search(r'第[一二三四五六七八九十百零〇0-9]+条', compact[start + len(article):])
    if nxt:
        end = start + len(article) + nxt.start()
    else:
        end = start + 800
    snippet = compact[start:end]
    return snippet


def _rule_ref(key: str, article: str, law_label: str) -> dict:
    return {'law': key, 'article': article, 'label': f'{law_label}{article}'}


# ── 46 指标四分组定义 ───────────────────────────────────────────────
INDICATOR_DEFS = [
    # ── 3.1 触发指标（11 项）──────────────────────────────────
    {
        'id': 'same_machine_code',
        'name': '同标段机器码雷同',
        'category': '触发指标',
        'problem': '不同投标文件由同一台机器制作（机器硬件指纹/文件作者一致）。',
        'rule': '同一标段内 ≥2 份投标文件机器码（制作机器标识）一致，记为疑似，每组得基准分 30 分。',
        'rule_ref': _rule_ref('002_tender_regs', '第四十条', '招标投标法实施条例'),
        'checker': 'file_attr',
        'local': True,
    },
    {
        'id': 'same_file_code',
        'name': '同标段文件码雷同',
        'category': '触发指标',
        'problem': '不同投标文件内容哈希/文件指纹雷同，属于同一模板直接套用。',
        'rule': '同一标段内 ≥2 份投标文件文件码（指纹）一致的，记为疑似，每组得 30 分。',
        'rule_ref': _rule_ref('002_tender_regs', '第四十条', '招标投标法实施条例'),
        'checker': 'text_sim',
        'local': True,
    },
    {
        'id': 'same_dongle',
        'name': '同标段加密锁雷同',
        'category': '触发指标',
        'problem': '不同投标人使用同一把投标加密锁/CA 证书制作或提交文件。',
        'rule': '同一标段内 ≥2 份投标文件使用同一加密锁序列号的，记为疑似，每组得 30 分。',
        'rule_ref': _rule_ref('007_ebidding_measures', '第十六条', '电子招标投标办法'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'tech_section_similar',
        'name': '同标段技术标雷同',
        'category': '触发指标',
        'problem': '不同投标文件技术标内容高度雷同（同一团队编制）。',
        'rule': '技术标文本相似度 ≥80% 的组合，中标单位每对 +5 分、非中标单位每对 +2 分。',
        'rule_ref': _rule_ref('001_tender_law', '第三十二条', '招标投标法'),
        'checker': 'text_sim',
        'local': True,
    },
    {
        'id': 'contact_person_same',
        'name': '同标段单位联系人雷同',
        'category': '触发指标',
        'problem': '不同投标单位的联系人/联系方式雷同。',
        'rule': '同一标段 ≥2 份投标文件载明的联系人为同一人或同一电话/邮箱，记为疑似，每组得 30 分。',
        'rule_ref': _rule_ref('014_procurement_goods_bidding', '第三十七条', '政府采购货物和服务招标投标管理办法'),
        'checker': 'key_info',
        'local': True,
    },
    {
        'id': 'economic_error_similar',
        'name': '同标段经济标错误雷同',
        'category': '触发指标',
        'problem': '不同投标文件经济标出现相同/雷同的错误（含错别字、计算错误）。',
        'rule': '同一标段 ≥2 份投标文件经济标出现相同疑似错误 ≥3 处，记为疑似，每组得 30 分。',
        'rule_ref': _rule_ref('002_tender_regs', '第四十条', '招标投标法实施条例'),
        'checker': 'typo',
        'local': True,
    },
    {
        'id': 'bid_ip_same',
        'name': '同标段投标IP雷同',
        'category': '触发指标',
        'problem': '不同投标人从同一 IP 提交投标文件。',
        'rule': '同一标段 ≥2 家投标人提交投标的 IP 地址一致，记为疑似，每组得 30 分。',
        'rule_ref': _rule_ref('014_procurement_goods_bidding', '第三十七条', '政府采购货物和服务招标投标管理办法'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'decrypt_ip_same',
        'name': '同标段解密IP雷同',
        'category': '触发指标',
        'problem': '不同投标文件的解密操作出自同一 IP。',
        'rule': '同一标段 ≥2 家投标文件解密 IP 一致，记为疑似，每组得 30 分。',
        'rule_ref': _rule_ref('007_ebidding_measures', '第十六条', '电子招标投标办法'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'download_ip_same',
        'name': '同标段下载IP雷同',
        'category': '触发指标',
        'problem': '不同投标人从同一 IP 下载招标文件。',
        'rule': '同一标段 ≥2 家投标人下载招标文件的 IP 一致，记为疑似，每组得 30 分。',
        'rule_ref': _rule_ref('014_procurement_goods_bidding', '第三十七条', '政府采购货物和服务招标投标管理办法'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'cross_file_code_same',
        'name': '异标段文件码雷同',
        'category': '触发指标',
        'problem': '同一批文件在不同标段间共享同一模板/指纹（关联投标人跨标段抱团）。',
        'rule': '不同标段投标文件文件码一致且投标人组合高度重合的，记为疑似，每组得 30 分。',
        'rule_ref': _rule_ref('009_bidder_conduct_opinions', '第十条', '发改法规规〔2022〕1117号'),
        'checker': 'text_sim',
        'local': True,
    },
    {
        'id': 'cross_contact_same',
        'name': '异标段单位联系人雷同',
        'category': '触发指标',
        'problem': '不同标段投标文件的联系人雷同，可能由同一人代理多家投标。',
        'rule': '不同标段投标文件联系人雷同且主体不同的，记为疑似，每组得 30 分。',
        'rule_ref': _rule_ref('009_bidder_conduct_opinions', '第十条', '发改法规规〔2022〕1117号'),
        'checker': 'key_info',
        'local': True,
    },

    # ── 3.2 核心指标（21 项）──────────────────────────────────
    {
        'id': 'tender_query',
        'name': '招标质疑',
        'category': '核心指标',
        'problem': '投标人对招标文件/评标结果大量或异常质疑。',
        'rule': '存在针对招标文件或结果的异常质疑且指向特定投标人的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('002_tender_regs', '第二十二条', '招标投标法实施条例'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'candidate_give_up',
        'name': '中标候选人放弃中标',
        'category': '核心指标',
        'problem': '中标候选人异常放弃中标，可能让位于预先约定的单位。',
        'rule': '中标候选人放弃中标且无正当理由的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('006_bid_eval_regs', '第四十八条', '评标委员会和评标方法暂行规定'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'subjective_expert_spread',
        'name': '主观分同一单位不同专家评分偏离度异常',
        'category': '核心指标',
        'problem': '同一投标单位在不同评委处的主观评分偏离度异常。',
        'rule': '同一投标单位主观评分极差/标准差超过阈值的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('009_bidder_conduct_opinions', '第十条', '发改法规规〔2022〕1117号'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'low_win_rate',
        'name': '低中标率异常',
        'category': '核心指标',
        'problem': '投标人参与投标频繁但中标率极低（疑似陪标）。',
        'rule': '历史中标率明显低于同批投标人平均水平的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('009_bidder_conduct_opinions', '第十条', '发改法规规〔2022〕1117号'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'high_win_rate',
        'name': '高中标率异常',
        'category': '核心指标',
        'problem': '投标人中标率异常高（疑似内定/围标受益方）。',
        'rule': '历史中标率明显高于同批投标人平均水平的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('009_bidder_conduct_opinions', '第十条', '发改法规规〔2022〕1117号'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'bidder_count_abnormal',
        'name': '投标单位数异常',
        'category': '核心指标',
        'problem': '投标单位数量异常（过少＝围标易达成，过多＝排斥竞争疑点）。',
        'rule': '有效投标数 <3 或异常集中于少数关联主体的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('002_tender_regs', '第四十四条', '招标投标法实施条例'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'upload_interval_abnormal',
        'name': '标书上传间隔异常',
        'category': '核心指标',
        'problem': '多家投标人在极短时间内连续上传投标文件（同批制作/同一场所）。',
        'rule': '同一标段投标文件上传时间高度聚集（间隔 ≤ 阈值）的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('007_ebidding_measures', '第二十七条', '电子招标投标办法'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'subjective_expert_units',
        'name': '主观分同一专家对不同单位评分偏离度异常',
        'category': '核心指标',
        'problem': '同一评委对不同投标单位的主观评分偏离度异常（人为拉高/压低）。',
        'rule': '同一评委对各单位主观评分离散度异常且偏向特定单位的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('009_bidder_conduct_opinions', '第十条', '发改法规规〔2022〕1117号'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'specific_expert_score',
        'name': '特定评委打分异常',
        'category': '核心指标',
        'problem': '特定评委对特定投标人持续给高分/低分。',
        'rule': '特定评委对特定单位评分持续异常（显著高于/低于其他评委）的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('006_bid_eval_regs', '第十三条', '评标委员会和评标方法暂行规定'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'clique_expert_scoring',
        'name': '抱团评委打分倾向性',
        'category': '核心指标',
        'problem': '多名评委对同一投标人抱团给出高分（串通评标）。',
        'rule': '≥3 名评委对同一单位评分同时偏高且与其他评委显著不一致的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('009_bidder_conduct_opinions', '第十条', '发改法规规〔2022〕1117号'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'clique_expert_consistency',
        'name': '抱团评委打分一致性',
        'category': '核心指标',
        'problem': '多名评委打分排序高度一致（失去独立评审）。',
        'rule': '≥3 名评委对各单位评分排序高度一致且偏离统计预期的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('006_bid_eval_regs', '第十七条', '评标委员会和评标方法暂行规定'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'objective_score_abnormal',
        'name': '客观分得分异常',
        'category': '核心指标',
        'problem': '客观分（商务/资质/业绩）得分异常集中于特定投标人。',
        'rule': '客观分评审结果与应得分明显不符或集中偏高的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('006_bid_eval_regs', '第二十九条', '评标委员会和评标方法暂行规定'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'high_price_abnormal',
        'name': '投标单位高价投标异常',
        'category': '核心指标',
        'problem': '投标人报价异常偏高（陪标方抬价，为特定单位让路）。',
        'rule': '报价显著高于其他投标且无合理成本说明的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('002_tender_regs', '第四十条', '招标投标法实施条例'),
        'checker': 'quote',
        'local': True,
    },
    {
        'id': 'extension_abnormal',
        'name': '招标延期异常',
        'category': '核心指标',
        'problem': '招标文件/投标截止时间频繁异常延期（等待特定投标人）。',
        'rule': '同一项目异常延期 ≥2 次或延期后仅特定投标人受益的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('002_tender_regs', '第二十一条', '招标投标法实施条例'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'tender_fail_abnormal',
        'name': '招标失败异常',
        'category': '核心指标',
        'problem': '项目反复流标/招标失败后重新招标（洗牌排斥对手）。',
        'rule': '同一项目连续 ≥2 次招标失败的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('002_tender_regs', '第四十四条', '招标投标法实施条例'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'waste_rate_abnormal',
        'name': '单位废标率异常',
        'category': '核心指标',
        'problem': '投标人废标率异常高（配合围标主动制造废标）。',
        'rule': '历史废标率明显高于平均水平的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('006_bid_eval_regs', '第二十条', '评标委员会和评标方法暂行规定'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'download_no_bid',
        'name': '标书获取单位投标异常',
        'category': '核心指标',
        'problem': '大量获取标书的单位未实际投标（陪标凑数）。',
        'rule': '标书获取单位数显著多于投标单位数且集中放弃的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('009_bidder_conduct_opinions', '第十条', '发改法规规〔2022〕1117号'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'no_show_abnormal',
        'name': '投标单位未开标异常',
        'category': '核心指标',
        'problem': '已投标单位在开标前集中撤回/缺席。',
        'rule': '开标前集中撤回投标或未出席开标的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('002_tender_regs', '第三十五条', '招标投标法实施条例'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'expert_deviation_abnormal',
        'name': '专家打分偏离度异常',
        'category': '核心指标',
        'problem': '专家打分与最终结果/其他评委普遍偏离（异常干预）。',
        'rule': '某评委评分与全体评委均值持续显著偏离的，记为可疑，得 10 分。',
        'rule_ref': _rule_ref('006_bid_eval_regs', '第十三条', '评标委员会和评标方法暂行规定'),
        'checker': 'skip',
        'local': False,
    },

    # ── 3.3 基础指标（11 项）──────────────────────────────────
    {
        'id': 'cross_bid_ip',
        'name': '异标段投标IP雷同',
        'category': '基础指标',
        'problem': '不同标段投标人使用同一 IP 提交文件（跨标段关联）。',
        'rule': '不同标段投标 IP 一致且主体不同的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('014_procurement_goods_bidding', '第三十七条', '政府采购货物和服务招标投标管理办法'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'cross_decrypt_ip',
        'name': '异标段解密IP雷同',
        'category': '基础指标',
        'problem': '不同标段解密操作 IP 一致。',
        'rule': '不同标段解密 IP 一致且主体不同的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('007_ebidding_measures', '第十六条', '电子招标投标办法'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'cross_machine_code',
        'name': '异标段机器码雷同',
        'category': '基础指标',
        'problem': '不同标段投标文件由同一机器制作。',
        'rule': '不同标段投标文件机器码一致且主体不同的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('009_bidder_conduct_opinions', '第十条', '发改法规规〔2022〕1117号'),
        'checker': 'file_attr',
        'local': True,
    },
    {
        'id': 'bidder_agent_contact',
        'name': '投标人与招标代理联系人雷同',
        'category': '基础指标',
        'problem': '投标人联系人与招标代理机构联系人雷同（内外勾连）。',
        'rule': '投标人投标文件联系人与招标代理联系人雷同的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('002_tender_regs', '第四十一条', '招标投标法实施条例'),
        'checker': 'relationship',
        'local': True,
    },
    {
        'id': 'expert_tenderer_closeness',
        'name': '评委与招标人亲密度',
        'category': '基础指标',
        'problem': '评委与招标人存在社交/任职关联（影响独立评审）。',
        'rule': '评委与招标人存在任职、股权或社交关联的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('006_bid_eval_regs', '第十二条', '评标委员会和评标方法暂行规定'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'expert_agent_closeness',
        'name': '评委与招标代理亲密度',
        'category': '基础指标',
        'problem': '评委与招标代理机构存在关联。',
        'rule': '评委与招标代理存在任职、股权或社交关联的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('006_bid_eval_regs', '第十二条', '评标委员会和评标方法暂行规定'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'cross_download_ip',
        'name': '异标段下载IP雷同',
        'category': '基础指标',
        'problem': '不同标段下载招标文件 IP 一致。',
        'rule': '不同标段下载 IP 一致且主体不同的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('014_procurement_goods_bidding', '第三十七条', '政府采购货物和服务招标投标管理办法'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'cross_dongle',
        'name': '异标段加密锁雷同',
        'category': '基础指标',
        'problem': '不同标段使用同一加密锁。',
        'rule': '不同标段加密锁序列号一致且主体不同的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('007_ebidding_measures', '第十六条', '电子招标投标办法'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'tech_seal_check',
        'name': '技术标暗标检测异常',
        'category': '基础指标',
        'problem': '技术标暗标中发现可识别身份信息（泄露单位身份）。',
        'rule': '技术标中出现单位名称、人员姓名等可识别身份信息的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('006_bid_eval_regs', '第十九条', '评标委员会和评标方法暂行规定'),
        'checker': 'typo',
        'local': True,
    },
    {
        'id': 'expert_bidder_closeness',
        'name': '评委与投标人亲密度',
        'category': '基础指标',
        'problem': '评委与投标人存在关联（影响独立评审）。',
        'rule': '评委与投标人存在任职、股权或社交关联的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('006_bid_eval_regs', '第十二条', '评标委员会和评标方法暂行规定'),
        'checker': 'relationship',
        'local': True,
    },
    {
        'id': 'bad_expert_score',
        'name': '不良评委打分异常',
        'category': '基础指标',
        'problem': '有不良记录评委参与打分且评分异常。',
        'rule': '存在不良记录评委且其评分显著异常的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('006_bid_eval_regs', '第五十四条', '评标委员会和评标方法暂行规定'),
        'checker': 'skip',
        'local': False,
    },

    # ── 3.4 扩展指标（4 项）──────────────────────────────────
    {
        'id': 'tech_score_abnormal',
        'name': '技术标得分异常',
        'category': '扩展指标',
        'problem': '技术标得分异常集中于特定投标人。',
        'rule': '技术标得分极差/集中度异常的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('006_bid_eval_regs', '第三十五条', '评标委员会和评标方法暂行规定'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'commercial_score_abnormal',
        'name': '商务标得分异常',
        'category': '扩展指标',
        'problem': '商务标得分异常集中于特定投标人。',
        'rule': '商务标得分极差/集中度异常的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('006_bid_eval_regs', '第三十五条', '评标委员会和评标方法暂行规定'),
        'checker': 'skip',
        'local': False,
    },
    {
        'id': 'quote_proportional_float',
        'name': '投标总价等比浮动异常',
        'category': '扩展指标',
        'problem': '多家投标总价呈等比/等差数列浮动（串通报价规律）。',
        'rule': '投标总价呈等比或规律性浮动（报价规律性差异）的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('014_procurement_goods_bidding', '第三十七条', '政府采购货物和服务招标投标管理办法'),
        'checker': 'quote',
        'local': True,
    },
    {
        'id': 'contact_phone_abnormal',
        'name': '投标单位联系人手机号异常',
        'category': '扩展指标',
        'problem': '投标联系人电话号码异常（多单位共用、空号、重复号码）。',
        'rule': '不同投标单位的联系人电话雷同或同一号码出现在多个单位的，记为可疑，得 5 分。',
        'rule_ref': _rule_ref('002_tender_regs', '第四十条', '招标投标法实施条例'),
        'checker': 'skip',
        'local': False,
    },
]


def get_indicator_by_id(ind_id: str):
    for ind in INDICATOR_DEFS:
        if ind['id'] == ind_id:
            return ind
    return None