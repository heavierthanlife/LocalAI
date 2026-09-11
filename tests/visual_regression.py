import os, sys, time, json
from pathlib import Path

SCREENSHOT_DIR = Path(r"D:/PyCharm/Local_AI/tests/visual_screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)

from playwright.sync_api import sync_playwright

BASE_URL = "https://localhost:443"
TIMEOUT = 15000
results = []

def snap(page, name, num):
    fn = f"{num:02d}_{name}.png"
    fp = SCREENSHOT_DIR / fn
    page.screenshot(path=str(fp), full_page=True)
    results.append({"checkpoint": num, "name": name, "file": str(fp), "status": "captured"})
    print(f"[CP-{num}] {name} -> {fn}")

def try_click(page, sels):
    for s in sels:
        try:
            el = page.query_selector(s)
            if el and el.is_visible():
                el.click()
                time.sleep(2)
                return True
        except Exception:
            pass
    return False

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox","--ignore-certificate-errors","--disable-gpu"])
        ctx = browser.new_context(viewport={"width":1440,"height":900}, ignore_https_errors=True, locale="zh-CN")
        page = ctx.new_page()
        page.set_default_timeout(TIMEOUT)

        # CP1
        try:
            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=20000)
            time.sleep(3)
            snap(page, "homepage_login", 1)
        except Exception as e:
            results.append({"checkpoint":1,"name":"homepage_login","error":str(e)})
            print(f"[CP-1] FAIL: {e}")

        # CP2
        try:
            pw = page.query_selector('input[type="password"]')
            if pw:
                ui = page.query_selector('input[type="text"],input[name="username"]')
                if ui: ui.fill("admin")
                pw.fill("123456")
                btn = page.query_selector('button[type="submit"],.login-btn,#loginBtn')
                if btn: btn.click(); time.sleep(3)
            snap(page, "chat_interface", 2)
        except Exception as e:
            results.append({"checkpoint":2,"name":"chat_interface","error":str(e)})
            print(f"[CP-2] FAIL: {e}")

        # CP3
        try:
            if not try_click(page, ["text=清标","text=批量清标","text=合规检查"]):
                for p2 in ["/batch","/clearance","/compliance"]:
                    try: page.goto(BASE_URL+p2, wait_until="domcontentloaded", timeout=10000); time.sleep(1); break
                    except: pass
            snap(page, "clearance_tool", 3)
        except Exception as e:
            results.append({"checkpoint":3,"name":"clearance_tool","error":str(e)})
            print(f"[CP-3] FAIL: {e}")

        # CP4
        try:
            try_click(page, ["text=查看报告","text=报告",".report-link"])
            snap(page, "clearance_report", 4)
        except Exception as e:
            results.append({"checkpoint":4,"name":"clearance_report","error":str(e)})
            print(f"[CP-4] FAIL: {e}")

        # CP5
        try:
            try_click(page, ["text=指标分析","text=一、","#tab-indicators"])
            snap(page, "indicators_tab", 5)
        except Exception as e:
            results.append({"checkpoint":5,"name":"indicators_tab","error":str(e)})
            print(f"[CP-5] FAIL: {e}")

        # CP6
        try:
            try_click(page, ["text=段落级","text=实质雷同","text=段落证据"])
            snap(page, "paragraph_evidence", 6)
        except Exception as e:
            results.append({"checkpoint":6,"name":"paragraph_evidence","error":str(e)})
            print(f"[CP-6] FAIL: {e}")

        # CP7
        try:
            try_click(page, ["text=横向对比","text=二、","#tab-cross"])
            snap(page, "cross_comparison", 7)
        except Exception as e:
            results.append({"checkpoint":7,"name":"cross_comparison","error":str(e)})
            print(f"[CP-7] FAIL: {e}")

        # CP8
        try:
            if not try_click(page, ["text=管理","text=Admin","#nav-admin"]):
                for p2 in ["/admin","/admin/"]:
                    try: page.goto(BASE_URL+p2, wait_until="domcontentloaded", timeout=10000); time.sleep(1); break
                    except: pass
            pi = page.query_selector('input[name="pin"],input[placeholder*="PIN"]')
            if pi:
                pi.fill("123456")
                btn = page.query_selector('button[type="submit"]')
                if btn: btn.click(); time.sleep(2)
            try_click(page, ["text=报价异常","text=异常检测记录"])
            snap(page, "admin_quote_anomaly", 8)
        except Exception as e:
            results.append({"checkpoint":8,"name":"admin_quote_anomaly","error":str(e)})
            print(f"[CP-8] FAIL: {e}")

        # CP9
        try:
            try_click(page, ["text=关联关系","text=关联分析记录"])
            snap(page, "relationship_history", 9)
        except Exception as e:
            results.append({"checkpoint":9,"name":"relationship_history","error":str(e)})
            print(f"[CP-9] FAIL: {e}")

        # CP10
        try:
            el = page.query_selector("text=高度预警")
            if not el: el = page.query_selector("text=铁证触发")
            if el: el.scroll_into_view_if_needed(); time.sleep(1)
            snap(page, "hard_evidence", 10)
        except Exception as e:
            results.append({"checkpoint":10,"name":"hard_evidence","error":str(e)})
            print(f"[CP-10] FAIL: {e}")

        browser.close()

    manifest = SCREENSHOT_DIR / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    c = sum(1 for r in results if r.get("status")=="captured")
    e = sum(1 for r in results if "error" in r)
    print(f"Done: {c} captured, {e} failed out of {len(results)}")

if __name__ == "__main__": main()