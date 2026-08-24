#!/usr/bin/env python3
"""Collect official bid-law texts into data/laws/raw/ (national + beijing).

Phase A of 清标 feature: only official government sources (.gov.cn domains).
Usage:
    python scripts/collect_laws_raw.py
"""
import json
import os
import re
import ssl
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / 'data' / 'laws' / 'raw'

SOURCES = [
    {
        'key': 'national/001_tender_law',
        'name': '招标投标法',
        'version': '2017修正',
        'url': 'http://www.npc.gov.cn/npc/c2/c30834/201905/t20190521_279157.html',
        'domain': 'npc.gov.cn',
    },
    {
        'key': 'national/002_tender_regs',
        'name': '招标投标法实施条例',
        'version': '2019第三次修订',
        'url': 'https://www.ccgp.gov.cn/zcfg/gjfg/202309/t20230907_20661132.htm',
        'domain': 'ccgp.gov.cn',
    },
    {
        'key': 'national/003_procurement_law',
        'name': '政府采购法',
        'version': '2014修正',
        'url': 'https://www.audit.gov.cn/n7/n34/n58/c109659/content.html',
        'domain': 'audit.gov.cn',
    },
    {
        'key': 'national/004_procurement_regs',
        'name': '政府采购法实施条例',
        'version': '国务院令第658号',
        'url': 'https://www.mof.gov.cn/gp/xxgkml/gks/201504/t20150403_2511241.htm',
        'domain': 'mof.gov.cn',
    },
    {
        'key': 'national/005_must_tender_projects',
        'name': '必须招标的工程项目规定',
        'version': '国家发展改革委令第16号',
        'url': 'https://www.ndrc.gov.cn/xxgk/zcfb/fzggwl/201803/t20180330_960858.html',
        'domain': 'ndrc.gov.cn',
    },
    {
        'key': 'national/006_bid_eval_regs',
        'name': '评标委员会和评标方法暂行规定',
        'version': '七部委令第12号(2013修订)',
        'url': 'http://bjjs.zjw.beijing.gov.cn/bjjs/fwgl/fwzscqgl/ygdxx/flfg/328539/index.shtml',
        'domain': 'zjw.beijing.gov.cn',
    },
    {
        'key': 'national/007_ebidding_measures',
        'name': '电子招标投标办法',
        'version': '八部委令第20号',
        'url': 'https://www.gov.cn/gongbao/content/2013/content_2396614.htm',
        'domain': 'gov.cn',
    },
    {
        'key': 'national/008_notice_publish_measures',
        'name': '招标公告和公示信息发布管理办法',
        'version': '国家发展改革委令第10号',
        'url': 'https://www.moj.gov.cn/pub/sfbgwapp/zwgk/tzggApp/202105/t20210517_395532.html',
        'domain': 'moj.gov.cn',
    },
    {
        'key': 'national/009_bidder_conduct_opinions',
        'name': '关于严格执行招标投标法规制度进一步规范招标投标主体行为的若干意见',
        'version': '发改法规规〔2022〕1117号',
        'url': 'https://www.ndrc.gov.cn/xxgk/zcfb/ghxwj/202208/t20220801_1332495_ext.html',
        'domain': 'ndrc.gov.cn',
    },
    {
        'key': 'beijing/010_bj_tender_regs',
        'name': '北京市招标投标条例',
        'version': '2021修正',
        'url': 'http://www.beijing.gov.cn/zhengce/dfxfg/202111/t20211103_2528555.html',
        'domain': 'beijing.gov.cn',
    },
    {
        'key': 'beijing/011_bj_construction_supervision',
        'name': '北京市建设工程招标投标监督管理规定',
        'version': '市政府令第122号',
        'url': 'https://www.beijing.gov.cn/gongkai/zfxxgk/zc/gz/202112/t20211216_2562665.html',
        'domain': 'beijing.gov.cn',
    },
    {
        'key': 'beijing/012_bj_public_resource_supervision',
        'name': '北京市公共资源交易监督管理办法(试行)',
        'version': '京发改〔2017〕1102号',
        'url': 'https://www.beijing.gov.cn/zhengce/zhengcefagui/201905/t20190522_60372.html',
        'domain': 'beijing.gov.cn',
    },
    {
        'key': 'national/014_procurement_goods_bidding',
        'name': '政府采购货物和服务招标投标管理办法',
        'version': '财政部令第87号',
        'url': 'https://www.beijing.gov.cn/zhengce/zhengcefagui/qtwj/202302/t20230208_2913671.html',
        'domain': 'beijing.gov.cn',
    },
]

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/126.0 Safari/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30, context=CTX) as resp:
        raw = resp.read()
    for enc in ('utf-8', 'gb18030', 'gbk'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace')


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    index = {}
    for s in SOURCES:
        out = RAW / f'{s["key"]}.html'
        if out.exists() and out.stat().st_size > 1000:
            print(f'[skip] {s["name"]} (already exists)')
            index[s['name']] = {'file': str(out.relative_to(RAW)), 'url': s['url']}
            continue
        try:
            html = fetch(s['url'])
            out.write_text(html, encoding='utf-8')
            index[s['name']] = {'file': str(out.relative_to(RAW)), 'url': s['url']}
            print(f'[ok]   {s["name"]} -> {out.relative_to(RAW)} ({len(html)} bytes)')
        except Exception as e:
            print(f'[FAIL] {s["name"]}: {e}')
        time.sleep(1.0)

    index_path = BASE / 'data' / 'laws' / 'index.json'
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
    print('index written:', index_path)


if __name__ == '__main__':
    main()