"""Contact / phone / email extraction for clearance collusion indicators.

FIX-2026-09-07-QA-C4: `contact_person_same` / `cross_contact_same` /
`contact_phone_abnormal` previously fell back to `key_info` keyword overlap
(fake "contact match" from shared industry words) or were hard `skip`. This
module extracts real contact entities from bid-document text and compares them
across files. When no contact entity can be extracted, indicators report a
honest "needs 开标信息表/联系人数据" placeholder instead of a bogus score.

Reuses:
  - relationship_extractor._PERSON_NAME_RE / _PERSON_TITLE / _is_valid_person
    (person-name extraction + filtering)
  - clearance_openinfo._PHONE_RE (11-digit mobile)

Score formulas are aligned with the open-info path:
  - same contact person across >=2 files: min(15 + 5*n_groups, 40)
  - same phone across >=2 files:          min(10 + 5*n_groups, 30)
"""
from __future__ import annotations

import re
from collections import defaultdict

_EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
_PHONE_RE = re.compile(r'1[3-9]\d{9}')


def extract_contacts(text: str) -> dict:
    """Extract contact persons / phones / emails from one document's raw text.

    Returns {'contacts': set[str], 'phones': set[str], 'emails': set[str]}.
    Person names are only kept when they follow a contact-ish title
    (联系人/电话/委托代理人/负责人 等) to avoid generic-name noise.
    """
    from app.services.relationship_extractor import _PERSON_NAME_RE, _is_valid_person

    contacts: set[str] = set()
    phones: set[str] = set()
    emails: set[str] = set()

    if not text:
        return {"contacts": contacts, "phones": phones, "emails": emails}

    # 手机号：正文扫描（11 位 1[3-9] 开头）
    for m in _PHONE_RE.finditer(text):
        # 上下文守卫：排除日期/编号等误报——手机号前后不应紧跟"年/月/日/编号/号"
        ctx = text[max(0, m.start() - 4):m.end() + 4]
        if any(w in ctx for w in ("年", "月", "日", "编号", "序号")):
            continue
        phones.add(m.group())

    # 邮箱
    emails.update(_EMAIL_RE.findall(text))

    # 联系人：只认 标题[：:]名字 或 名字[：:]标题 且标题为联系人语境
    for m in _PERSON_NAME_RE.finditer(text):
        name = (m.group("name") or m.group("name2") or "").strip()
        title = (m.group("title") or m.group(1) or "").strip()
        if not name:
            continue
        title_ctx = title + " " + text[max(0, m.start() - 6):m.start() + 6]
        if not any(k in title_ctx for k in ("联系人", "委托代理人", "授权代表", "被授权人", "项目负责人", "负责人", "法人")):
            continue
        if not _is_valid_person(name):
            continue
        contacts.add(name)

    return {"contacts": contacts, "phones": phones, "emails": emails}


def compare_contacts(file_data: list[dict]) -> dict:
    """Cross-file contact comparison → per-indicator results.

    file_data: [{filename, text, ...}].

    Returns {indicator_id: {'score', 'result', 'details'}} for:
      - contact_person_same   (same contact person in >=2 files)
      - cross_contact_same    (same contact person across files; without a
                               bid-section grouping this equals contact_person_same)
      - contact_phone_abnormal(same phone in >=2 files)
    When no contact entity is found at all, results carry
    `placeholder: True` so the caller can render "○ 需开标信息表/联系人数据".
    """
    n = len(file_data)
    persons: dict[str, list[str]] = defaultdict(list)
    phones: dict[str, list[str]] = defaultdict(list)
    emails: dict[str, list[str]] = defaultdict(list)
    any_contact_found = False

    for fd in file_data:
        name = fd.get("filename", "?")
        got = extract_contacts(fd.get("text", ""))
        if got["contacts"] or got["phones"] or got["emails"]:
            any_contact_found = True
        for c in got["contacts"]:
            persons[c].append(name)
        for p in got["phones"]:
            phones[p].append(name)
        for e in got["emails"]:
            emails[e].append(name)

    out = {}

    # 联系人雷同（≥2 家同人）
    dup_persons = {c: fs for c, fs in persons.items() if len(set(fs)) >= 2}
    if dup_persons:
        details = [{"contact": c, "files": ", ".join(sorted(set(fs))[:5])}
                   for c, fs in dup_persons.items()]
        out["contact_person_same"] = {
            "score": min(15 + 5 * len(details), 40),
            "result": f"▲ 发现 {len(details)} 组不同投标文件使用同一联系人（共{len(dup_persons)}个）。",
            "details": details,
        }
        # 无标段分组输入时，跨标段=同标段（同源），标注同值
        out["cross_contact_same"] = dict(out["contact_person_same"])
        out["cross_contact_same"]["result"] += "（无跨标段分组数据，按同标段判定）"
    else:
        out["contact_person_same"] = {"score": 0, "result": "√ 未发现联系人雷同。", "details": []}
        out["cross_contact_same"] = {"score": 0, "result": "√ 未发现跨标段联系人雷同。", "details": []}

    # 手机号雷同（≥2 家同号）
    dup_phones = {p: fs for p, fs in phones.items() if len(set(fs)) >= 2}
    if dup_phones:
        details = [{"phone": p, "files": ", ".join(sorted(set(fs))[:5])}
                   for p, fs in dup_phones.items()]
        out["contact_phone_abnormal"] = {
            "score": min(10 + 5 * len(details), 30),
            "result": f"▲ 发现 {len(details)} 组不同投标文件共用同一联系电话，存在围串标嫌疑。",
            "details": details,
        }
    else:
        out["contact_phone_abnormal"] = {"score": 0, "result": "√ 未发现不同投标文件联系电话雷同。", "details": []}

    # 无任何联系人实体 → 占位提示（诚实：非"确认无雷同"）
    if not any_contact_found:
        for k in out:
            out[k]["score"] = 0
            out[k]["placeholder"] = True
            out[k]["result"] = "○ 需开标信息表/联系人数据（投标文件正文未提取到联系人/电话/邮箱）"

    return out
