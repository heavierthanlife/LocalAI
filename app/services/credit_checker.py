"""Credit checker using Selenium Edge WebDriver for automated company searches."""
import os
import sys
import time
import logging
import subprocess
from io import BytesIO
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager

logger = logging.getLogger(__name__)


class CreditChecker:
    def __init__(self):
        self.driver = None
        self._init_driver()

    def _get_edge_binary(self):
        edge_path = os.environ.get('EDGE_BINARY_PATH')
        if edge_path and os.path.exists(edge_path):
            return edge_path
        default_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        if os.path.exists(default_path):
            return default_path
        return "msedge"

    def _get_user_data_dir(self):
        user_data = os.environ.get('EDGE_USER_DATA_DIR')
        if user_data and os.path.exists(user_data):
            return user_data
        return None

    def _get_webdriver_path(self):
        driver_path = os.environ.get('EDGEDRIVER_PATH')
        if driver_path and os.path.exists(driver_path):
            return driver_path
        local_driver = os.path.join(os.getcwd(), "msedgedriver.exe")
        if os.path.exists(local_driver):
            return local_driver
        try:
            return EdgeChromiumDriverManager().install()
        except Exception as e:
            logger.warning(f"Auto WebDriver download failed: {e}")
        raise FileNotFoundError(
            "Could not locate EdgeDriver. Please set EDGEDRIVER_PATH or place msedgedriver.exe in project root.")

    def _init_driver(self):
        # Kill any orphaned Edge instances spawned by WebDriver (Windows only)
        if sys.platform == 'win32':
            try:
                subprocess.run(
                    ["taskkill", "/f", "/im", "msedge.exe"],
                    capture_output=True, timeout=5
                )
                time.sleep(1)
                logger.info("Closed existing Edge instances")
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                logger.debug("taskkill failed — proceeding anyway")

        options = Options()
        options.binary_location = self._get_edge_binary()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("--inprivate")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        driver_path = self._get_webdriver_path()
        service = Service(driver_path)
        self.driver = webdriver.Edge(service=service, options=options)
        self.driver.set_window_size(1920, 1080)
        self.driver.implicitly_wait(5)
        logger.info(f"Edge WebDriver ready (driver: {driver_path})")

    # ========== Navigation and Search Methods ==========

    def navigate_and_fill(self, company_name, url):
        self.driver.get(url)
        time.sleep(2)
        self._fill_search(company_name, url)
        if 'zxgk.court.gov.cn' in url or 'creditchina.gov.cn' in url:
            self.driver.execute_script("document.body.style.zoom='75%'")
            time.sleep(0.5)

    def _fill_search(self, company_name, url):
        if 'zxgk.court.gov.cn' in url:
            selector = "#searchInput"
        elif 'creditchina.gov.cn' in url:
            selector = "#keyword"
        elif 'ccgp.gov.cn' in url:
            selector = "#searchWord"
        else:
            selector = None

        if selector:
            try:
                elem = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                elem.clear()
                elem.send_keys(company_name)
                return
            except Exception as e:
                logger.warning(f"Primary selector failed: {e}")

        try:
            inputs = self.driver.find_elements(By.CSS_SELECTOR, 'input[type="text"], input:not([type])')
            for inp in inputs:
                if inp.is_displayed() and inp.is_enabled():
                    inp.clear()
                    inp.send_keys(company_name)
                    return
        except Exception as e:
            logger.warning(f"Fallback fill failed: {e}")

        try:
            self.driver.execute_script(
                f"document.querySelector('input[type=text]').value = '{company_name}';")
        except Exception as e:
            logger.error(f"All fill methods failed: {e}")

    # ========== CAPTCHA Handling ==========

    def submit_captcha(self, solution):
        try:
            input_el = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.XPATH,
                    "//input[contains(@id,'captcha') or contains(@class,'captcha') or contains(@id,'yzm')]")))
            input_el.clear()
            input_el.send_keys(solution)
            submit_btns = self.driver.find_elements(By.XPATH,
                "//button[contains(text(),'确定') or contains(text(),'提交') or contains(text(),'查询')]")
            if submit_btns:
                submit_btns[0].click()
            else:
                input_el.send_keys('\n')
            time.sleep(2)
            return True
        except Exception as e:
            logger.error(f"Failed to submit CAPTCHA: {e}")
            return False

    def refresh_captcha(self):
        try:
            captcha_img = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.XPATH,
                    "//img[contains(@id,'captcha') or contains(@class,'captcha') or contains(@src,'captcha')]")))
            captcha_img.click()
            time.sleep(1)
            return True
        except Exception as e:
            logger.warning(f"Failed to refresh CAPTCHA: {e}")
            return False

    def get_captcha_element_screenshot(self):
        try:
            captcha_img = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.XPATH,
                    "//img[contains(@id,'captcha') or contains(@class,'captcha') or contains(@src,'captcha')]")))
            return BytesIO(captcha_img.screenshot_as_png)
        except Exception as e:
            logger.error(f"Failed to get CAPTCHA screenshot: {e}")
            return None

    def _is_captcha_present(self):
        try:
            self.driver.find_element(By.XPATH, "//img[contains(@id,'captcha') or contains(@class,'captcha')]")
            return True
        except Exception:
            return False

    def capture_viewport(self):
        return BytesIO(self.driver.get_screenshot_as_png())

    def close(self):
        if self.driver:
            self.driver.quit()
