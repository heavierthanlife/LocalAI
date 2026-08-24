#!/usr/bin/env python3
"""Convert data/laws/raw/*.html -> data/laws/clean/*.md (frontmatter + plain text).

Phase A of 清标 feature. Best-effort text extraction; manually curated
texts (from official pages) can be placed in data/laws/curated/ and will
win over HTML extraction.
"""
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / 'data' / 'laws' / 'raw'
CLEAN = BASE / 'data' / 'laws' / 'clean'
CURATED = BASE / 'data' / 'laws' / 'curated'

META = {
    '001_tender_law': dict(
        title='中华人民共和国招标投标法', doc_no='主席令第21号(1999通过/2017修正)',
        issued_by='全国人大常委会', effective_date='2000-01-01',
        version='2017年12月27日修正'),
    '002_tender_regs': dict(
        title='中华人民共和国招标投标法实施条例', doc_no='国务院令第613号(2019第三次修订)',
        issued_by='国务院', effective_date='2012-02-01',
        version='2019年3月2日第三次修订'),
    '003_procurement_law': dict(
        title='中华人民共和国政府采购法', doc_no='主席令第68号(2002通过/2014修正)',
        issued_by='全国人大常委会', effective_date='2003-01-01',
        version='2014年8月31日修正'),
    '004_procurement_regs': dict(
        title='中华人民共和国政府采购法实施条例', doc_no='国务院令第658号',
        issued_by='国务院', effective_date='2015-03-01',
        version='2015年1月30日公布'),
    '005_must_tender_projects': dict(
        title='必须招标的工程项目规定', doc_no='国家发展和改革委员会令第16号',
        issued_by='国家发展和改革委员会', effective_date='2018-06-01',
        version='2018年3月27日发布'),
    '006_bid_eval_regs': dict(
        title='评标委员会和评标方法暂行规定', doc_no='七部委令第12号(2013修订)',
        issued_by='国家计委等七部委', effective_date='2001-07-05',
        version='2013年3月11日修订'),
    '007_ebidding_measures': dict(
        title='电子招标投标办法', doc_no='八部委令第20号',
        issued_by='国家发展改革委等八部委', effective_date='2013-05-01',
        version='2013年2月4日发布'),
    '008_notice_publish_measures': dict(
        title='招标公告和公示信息发布管理办法', doc_no='国家发展和改革委员会令第10号',
        issued_by='国家发展和改革委员会', effective_date='2018-01-01',
        version='2017年11月23日发布'),
    '009_bidder_conduct_opinions': dict(
        title='关于严格执行招标投标法规制度进一步规范招标投标主体行为的若干意见',
        doc_no='发改法规规〔2022〕1117号', issued_by='国家发展改革委等13部门',
        effective_date='2022-09-01', version='2022年7月18日印发(有效期至2027年8月31日)'),
    '010_bj_tender_regs': dict(
        title='北京市招标投标条例', doc_no='北京市人大常委会公告(2021修正)',
        issued_by='北京市人大常委会', effective_date='2002-11-01',
        version='2021年9月24日修正'),
    '011_bj_construction_supervision': dict(
        title='北京市建设工程招标投标监督管理规定', doc_no='北京市人民政府令第122号',
        issued_by='北京市人民政府', effective_date='2003-06-01',
        version='2003年4月2日公布'),
    '012_bj_public_resource_supervision': dict(
        title='北京市公共资源交易监督管理办法(试行)', doc_no='京发改〔2017〕1102号',
        issued_by='北京市发展和改革委员会', effective_date='2017-07-25',
        version='2017年7月24日印发'),
    '013_criminal_law_ref': dict(
        title='串通投标罪刑事追诉参考', doc_no='刑法第223条+立案追诉标准(二)第68条',
        issued_by='最高人民法院(法答网)/最高人民检察院、公安部', effective_date='2021-04-30',
        version='法答网精选答问第二十四批',
        source_url='https://www.court.gov.cn/zixun/xiangqing/469691.html'),
    '014_procurement_goods_bidding': dict(
        title='政府采购货物和服务招标投标管理办法', doc_no='财政部令第87号',
        issued_by='财政部', effective_date='2017-10-01',
        version='2017年7月11日公布'),
}

TAG_RE = re.compile(r'<[^>]+>')
SCRIPT_RE = re.compile(r'<script.*?</script>', re.S | re.I)
STYLE_RE = re.compile(r'<style.*?</style>', re.S | re.I)
JS_RE = re.compile(r'function\s*\(\)\s*\{.*?\}\s*\(\);', re.S)
WS_RE = re.compile(r'[ \t\u3000]+')
NL_RE = re.compile(r'\n{3,}')


def extract(html_text: str) -> str:
    t = SCRIPT_RE.sub('', html_text)
    t = STYLE_RE.sub('', t)
    t = TAG_RE.sub('\n', t)
    t = html.unescape(t)
    t = JS_RE.sub('', t)
    t = t.replace('\u3000', ' ')
    lines = [WS_RE.sub(' ', ln).strip() for ln in t.splitlines()]
    lines = [ln for ln in lines if ln]
    text = '\n'.join(lines)
    return NL_RE.sub('\n\n', text).strip()


def write_clean(key, meta, body, rel, checksum, source, index):
    frontmatter = (
        '---\n'
        f'title: "{meta.get("title", key)}"\n'
        f'doc_no: "{meta.get("doc_no", "")}"\n'
        f'issued_by: "{meta.get("issued_by", "")}"\n'
        f'effective_date: "{meta.get("effective_date", "")}"\n'
        f'version: "{meta.get("version", "")}"\n'
        f'source_url: "{index.get(meta.get("title", ""), {}).get("url", meta.get("source_url", ""))}"\n'
        f'raw_file: "{rel}"\n'
        f'raw_sha256_prefix: "{checksum}"\n'
        f'clean_source: "{source}"\n'
        f'fetched_at: "{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}"\n'
        '---\n\n'
    )
    out = CLEAN / f'{key}.md'
    out.write_text(frontmatter + body + '\n', encoding='utf-8')
    print(f'[ok] {rel} -> clean/{key}.md ({len(body)} chars, {source})')


def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    CURATED.mkdir(parents=True, exist_ok=True)
    index = json.loads((BASE / 'data' / 'laws' / 'index.json').read_text(encoding='utf-8'))

    # 1) raw html twins -> clean (curated content wins when present)
    for raw in sorted(RAW.rglob('*.html')):
        key = raw.stem
        rel = raw.relative_to(RAW).as_posix()
        meta = META.get(key, {})
        curated = CURATED / f'{key}.md'
        checksum = hashlib.sha256(raw.read_bytes()).hexdigest()[:16]
        if curated.exists():
            body = curated.read_text(encoding='utf-8')
            source = 'curated'
        else:
            body = extract(raw.read_text(encoding='utf-8', errors='replace'))
            source = 'html-extracted'
        write_clean(key, meta, body, rel, checksum, source, index)

    # 2) curated files without a raw html twin (e.g. criminal law reference)
    for curated in sorted(CURATED.glob('*.md')):
        key = curated.stem
        if (RAW / f'{key}.html').exists():
            continue
        meta = META.get(key, {})
        body = curated.read_text(encoding='utf-8')
        rel = f'curated/{curated.name}'
        checksum = hashlib.sha256(curated.read_bytes()).hexdigest()[:16]
        write_clean(key, meta, body, rel, checksum, 'curated', index)

    print('done')


if __name__ == '__main__':
    main()