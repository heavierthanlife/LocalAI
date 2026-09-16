#!/usr/bin/env python
"""Fetch + parse national bidding/procurement laws into data/laws/extended_laws.json.

Batch B (UNRESOLVED-026): completes the 16 extended-only laws so the compliance
engine can load them (national-only). Source of truth = official government pages
(gov.cn 国务院公报 / ndrc.gov.cn / mof.gov.cn / mohurd.gov.cn).

Design:
  * SOURCES below maps each law to a search query and a preferred domain list.
  * URL discovery: if no direct `url`, query DuckDuckGo/Bing (best-effort — search
    engines often block scripted queries, so **prefer explicit `url`**).
  * Parsing: strip HTML, split the law body on 第X条 markers (sequential-number
    acceptance drops in-text cross-references).
  * Validation: the fetched page must contain the law name, else it is rejected —
    a wrong URL can never silently pollute the data.
  * Merge: existing entries in extended_laws.json are updated by law_name;
    `versions[0].articles` is replaced with the parsed full article list.
  * Every entry records source_url + fetched_at (traceable 法源台账).

Usage:
    python scripts/fetch_extended_laws.py --only 必须招标的工程项目规定
    python scripts/fetch_extended_laws.py            # all sources
    python scripts/fetch_extended_laws.py --dry-run  # parse + report, no write
"""
import argparse
import datetime
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAWS_PATH = os.path.join(PROJECT_ROOT, 'data', 'laws', 'extended_laws.json')

UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# law_name -> {short_name, category, query, domains, url?, version_label}
# national-only (广东省 办法 excluded — see UNRESOLVED-026).
SOURCES = [
    {"law_name": "政府采购法实施条例", "short_name": "政府采购法实施条例", "category": "行政法规",
     "query": "政府采购法实施条例 国务院令第658号 全文", "domains": ["gov.cn", "mof.gov.cn"]},
    {"law_name": "工程建设项目施工招标投标办法", "short_name": "工程施工招标投标办法", "category": "部门规章",
     "query": "工程建设项目施工招标投标办法 七部委30号令 全文", "domains": ["ndrc.gov.cn", "gov.cn"]},
    {"law_name": "评标委员会和评标方法暂行规定", "short_name": "评标委员会规定", "category": "部门规章",
     "query": "评标委员会和评标方法暂行规定 七部委12号令 全文", "domains": ["ndrc.gov.cn", "gov.cn"]},
    {"law_name": "必须招标的工程项目规定", "short_name": "必须招标规定", "category": "部门规章",
     "query": "必须招标的工程项目规定 发改委16号令 全文", "domains": ["gov.cn"],
     "url": "https://www.gov.cn/gongbao/content/2018/content_5296544.htm"},
    {"law_name": "政府采购货物和服务招标投标管理办法", "short_name": "政府采购87号令", "category": "部门规章",
     "query": "政府采购货物和服务招标投标管理办法 财政部87号令 全文", "domains": ["mof.gov.cn", "gov.cn"]},
    {"law_name": "政府采购非招标采购方式管理办法", "short_name": "政府采购74号令", "category": "部门规章",
     "query": "政府采购非招标采购方式管理办法 财政部74号令 全文", "domains": ["mof.gov.cn", "gov.cn"]},
    {"law_name": "政府采购质疑和投诉办法", "short_name": "政府采购94号令", "category": "部门规章",
     "query": "政府采购质疑和投诉办法 财政部94号令 全文", "domains": ["mof.gov.cn", "gov.cn"]},
    {"law_name": "电子招标投标办法", "short_name": "电子招标投标办法", "category": "部门规章",
     "query": "电子招标投标办法 八部委20号令 全文", "domains": ["ndrc.gov.cn", "gov.cn"]},
    {"law_name": "政府采购促进中小企业发展管理办法", "short_name": "中小企业46号", "category": "规范性文件",
     "query": "政府采购促进中小企业发展管理办法 财库2020 46号 全文", "domains": ["mof.gov.cn", "gov.cn"]},
    {"law_name": "关于促进政府采购公平竞争优化营商环境的通知", "short_name": "营商环境38号", "category": "规范性文件",
     "query": "关于促进政府采购公平竞争优化营商环境的通知 财库2019 38号 全文", "domains": ["mof.gov.cn", "gov.cn"]},
    {"law_name": "公共资源交易平台管理暂行办法", "short_name": "公共资源交易平台办法", "category": "部门规章",
     "query": "公共资源交易平台管理暂行办法 发改委39号令 全文", "domains": ["ndrc.gov.cn", "gov.cn"]},
    {"law_name": "建设工程工程量清单计价规范", "short_name": "清单计价规范", "category": "国家标准",
     "query": "建设工程工程量清单计价规范 GB50500-2013 总则 计价方式 条文", "domains": ["mohurd.gov.cn", "gov.cn"]},
    {"law_name": "关于进一步规范招标投标主体行为的通知", "short_name": "规范主体行为通知", "category": "规范性文件",
     "query": "关于进一步规范招标投标主体行为的通知 全文", "domains": ["gov.cn", "ndrc.gov.cn"]},
    {"law_name": "政府采购进口产品管理办法", "short_name": "进口产品办法", "category": "规范性文件",
     "query": "政府采购进口产品管理办法 财库2007 119号 全文", "domains": ["mof.gov.cn", "gov.cn"]},
    {"law_name": "招标投标违法行为记录公告暂行办法", "short_name": "违法行为记录办法", "category": "规范性文件",
     "query": "招标投标违法行为记录公告暂行办法 全文", "domains": ["gov.cn", "ndrc.gov.cn"]},
]

ART_MARK = re.compile(r'第[一二三四五六七八九十百千零〇]+条')

_CN = {'零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
       '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
_UNIT = {'十': 10, '百': 100, '千': 1000}


def _cn_num(s: str):
    """Chinese numeral → int (一..千), or None."""
    total = 0
    section = 0
    number = 0
    for ch in s:
        if ch in _CN:
            number = _CN[ch]
        elif ch in _UNIT:
            unit = _UNIT[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
        else:
            return None
    val = total + section + number
    return val if val > 0 else None


def _strip_html(src: str) -> str:
    src = re.sub(r'(?is)<(script|style).*?</\1>', '', src)
    src = re.sub(r'(?is)<br\s*/?>', '\n', src)
    src = re.sub(r'(?is)</p>', '\n', src)
    src = re.sub(r'(?is)<[^>]+>', '', src)
    src = html.unescape(src)
    src = re.sub(r'[\u3000\xa0]', ' ', src)
    src = re.sub(r'[ \t]+', ' ', src)
    src = re.sub(r'\n\s*\n+', '\n', src)
    return src.strip()


def _discover_candidates(query: str):
    """Return candidate result URLs. Tries DuckDuckGo html, then Bing."""
    out = []

    def _abs(u):
        if u.startswith('//'):
            return 'https:' + u
        return u

    # 1) DuckDuckGo html (uddg=<urlencoded>)
    try:
        url = 'https://duckduckgo.com/html/?q=' + urllib.parse.quote(query)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
            page = r.read().decode('utf-8', 'replace')
        for raw in re.findall(r'uddg=([^&"]+)', page):
            link = _abs(urllib.parse.unquote(raw))
            if link not in out:
                out.append(link)
    except Exception:
        pass

    # 2) Bing (direct hrefs in result titles)
    if not out:
        try:
            url = 'https://www.bing.com/search?q=' + urllib.parse.quote(query) + '&count=20'
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
                page = r.read().decode('utf-8', 'replace')
            for raw in re.findall(r'href="(https?://[^"]+)"', page):
                link = raw
                if 'bing.com' in link or 'microsoft.com' in link or 'msn.com' in link:
                    continue
                if link not in out:
                    out.append(link)
        except Exception:
            pass
    return out


def _discover_url(query: str, domains) -> str:
    for link in _discover_candidates(query):
        if any(d in link for d in domains):
            return link
    return ''


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    for enc in ('utf-8', 'gb18030'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', 'replace')


def _parse_articles(text: str):
    """Split law body into 第X条 blocks → [{article:'第N条', text:..., tags:[]}].

    Article markers inside the body may be cross-references (e.g. "根据《…》第三
    条的规定"), so only markers whose number continues the 1,2,3,… sequence are
    accepted as real article boundaries.
    """
    body = text
    accepted = []
    expected = 1
    for m in ART_MARK.finditer(body):
        n = _cn_num(m.group(0)[1:-1])
        if n == expected:
            accepted.append((m.start(), m.group(0), n))
            expected += 1

    arts = []
    for i, (pos, marker, n) in enumerate(accepted):
        end = accepted[i + 1][0] if i + 1 < len(accepted) else len(body)
        content = body[pos + len(marker):end].strip()
        if len(content) < 8:
            continue
        arts.append({
            'article': f'第{n}条',
            'text': content,
            'tags': [],
        })
    return arts


def process(src: dict, dry: bool):
    law = src['law_name']
    url = src.get('url') or ''
    if not url:
        try:
            url = _discover_url(src['query'], src.get('domains', ['gov.cn']))
        except Exception as e:
            print(f'  [SEARCH-FAIL] {law}: {e}')
            return None
    if not url:
        print(f'  [NO-URL] {law}: no result on {src.get("domains")}')
        return None
    try:
        page = _fetch(url)
    except Exception as e:
        print(f'  [FETCH-FAIL] {law}: {e}')
        return None
    text = _strip_html(page)
    if src['law_name'] not in text and src['short_name'] not in text:
        print(f'  [VALIDATE-FAIL] {law}: page does not contain the law name (wrong URL?)')
        return None
    arts = _parse_articles(text)
    print(f'  {law}: url={url[:80]} articles={len(arts)}')
    if not arts:
        return None
    return {
        'law_name': law,
        'short_name': src['short_name'],
        'category': src['category'],
        'scope': 'national',
        'source_url': url,
        'fetched_at': datetime.date.today().isoformat(),
        'versions': [{
            'version_label': src.get('version_label', ''),
            'is_current': True,
            'articles': arts,
        }],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', action='append', default=[])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--discover', action='store_true', help='print candidate URLs only')
    args = ap.parse_args()

    if args.discover:
        for s in SOURCES:
            if s.get('url'):
                print(f"\n{s['law_name']}: (direct) {s['url']}")
                continue
            print(f"\n{s['law_name']}: {s['query']}")
            try:
                for c in _discover_candidates(s['query'])[:6]:
                    mark = '*' if any(d in c for d in s.get('domains', [])) else ' '
                    print(f'  {mark} {c}')
            except Exception as e:
                print('  [search-fail]', e)
        return 0

    with open(LAWS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    by_name = {x['law_name']: x for x in data}

    todo = [s for s in SOURCES if not args.only or s['law_name'] in args.only]
    updated = 0
    for src in todo:
        entry = process(src, args.dry_run)
        if not entry:
            continue
        old = by_name.get(entry['law_name'])
        if old:
            old.update(entry)
        else:
            data.append(entry)
            by_name[entry['law_name']] = entry
        updated += 1

    print(f'\nupdated {updated}/{len(todo)} laws')
    if not args.dry_run:
        with open(LAWS_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print('written', LAWS_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main())
