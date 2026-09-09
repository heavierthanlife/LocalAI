"""暗标（技术标匿名评审）身份泄露检测.

FIX-2026-09-07-QA-C4: `tech_seal_check` 此前被误用 typo(错别字) checker 实现，
输出"√ 未发现文本质量问题"——从没真正检测过暗标身份泄露。

暗标要求：技术标文件不得出现可识别投标人身份的信息（单位名称、人员姓名、
印章、电话等），否则评审时即可辨识投标人，破坏匿名性。

实现策略（对"商务技术标"合并文件，技术标与商务标无法可靠切分）：
  - 从文件名提取投标人公司名（如"北京中昌华美超市服务有限责任公司"）
  - 检测该公司名是否以"异常高频/明显暴露"的方式出现在正文中
    （单纯出现在封面/签名页是正常的，但贯穿正文属泄露风险）
  - 另检测正文中大量出现的人员姓名、电话；公章/盖章提示仅作辅助证据
  合并文件场景保守起见：若公司名在正文出现次数超过阈值或出现在
  "技术方案/技术参数"等章节片段中，判为泄露。
  注意：盖章/公章提示是弱信号（普通标书正文含"盖章/公章"属正常要求），
  不独立触发泄露，仅当已由强信号（公司名/多处人员姓名）判定泄露时作为辅助证据列出。
"""
from __future__ import annotations

import re

_COMPANY_SUFFIX_RE = re.compile(
    r'([一-鿿]{2,20}?(?:有限公司|有限责任公司|股份有限公司|集团有限公司|集团|'
    r'事务所|合伙企业|分公司|子公司|总公司|母公司))'
)
_SELF_REF_MARKERS = ("我公司", "我方", "本公司", "我单位", "本单位")
_SEAL_MARKERS = ("盖章", "公章", "签字盖章", "投标专用章")
# 正文中"技术方案/技术参数/施工组织"等章节起点（若命中这些区间，身份泄露更可疑）
_TECH_SECTION_RE = re.compile(r'技术方案|技术参数|技术指标|施工组织|服务方案|实施方案')


def _extract_company_from_filename(filename: str) -> str:
    """从文件名提取公司名（文件名通常含投标人全称）。"""
    m = _COMPANY_SUFFIX_RE.search(filename or "")
    return m.group(1) if m else ""


def detect_tech_seal_leak(file_data: list[dict]) -> dict:
    """对每份文件检测暗标身份泄露。

    Returns {filename: {'company': 公司名, 'leak': bool,
                        'evidence': [str], 'score': float}}
    score 0-30：公司名贯穿技术方案段 / 多处人员姓名 → 泄露，盖章提示仅辅助。

    注意（FIX-2026-09-07-QA-C4 校准）：对"商务技术标"合并文件，正文中的
    "我公司/本公司"自指称谓是商务标的正常表述，**不作为独立泄露信号**。
    只有：① 公司名在技术方案章节片段内出现；② ≥4 处人员姓名，才判泄露——
    避免把正常商务标误判为暗标违规。③ 盖章/公章提示为弱信号（普通标书
    正文含"盖章/公章"属正常要求），不独立触发泄露，仅作辅助证据。
    """
    results = {}
    for fd in file_data:
        name = fd.get("filename", "?")
        text = fd.get("text", "")
        company = _extract_company_from_filename(name)
        evidence: list[str] = []
        leak = False

        if not text:
            results[name] = {"company": company, "leak": False, "evidence": [], "score": 0}
            continue

        # 1) 技术方案章节片段内出现公司名（真实泄露证据）
        tech_matches = list(_TECH_SECTION_RE.finditer(text))
        if company:
            for tm in tech_matches:
                seg = text[tm.start():tm.start() + 400]
                if company in seg:
                    leak = True
                    evidence.append(f"技术方案段出现公司名「{company}」")
                    break

        # 2) 大量人员姓名（>4 处"姓名"样式）
        try:
            from app.services.relationship_extractor import _PERSON_NAME_RE
            person_count = len(list(_PERSON_NAME_RE.finditer(text)))
            if person_count >= 4:
                leak = True
                evidence.append(f"正文出现 {person_count} 处人员姓名（匿名评审应避免）")
        except Exception:
            pass

        # 3) 盖章/公章提示 —— 弱信号（普通标书正文含"盖章/公章"属正常要求），
        #    不独立触发泄露；仅当已由强信号判定泄露时作为辅助证据列出。
        seal_hits = [mk for mk in _SEAL_MARKERS if mk in text]
        if seal_hits and leak:
            evidence.append(f"正文出现盖章类提示「{'、'.join(seal_hits)}」（辅助证据）")

        score = 30 if leak else 0
        results[name] = {
            "company": company,
            "leak": leak,
            "evidence": evidence[:5],
            "score": score,
        }
    return results
