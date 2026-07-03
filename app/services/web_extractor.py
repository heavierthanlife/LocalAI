"""Web page content extraction with retry + exponential backoff (ported from douban-ai-analyzer)."""

import re
import time
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# User-Agent rotation pool
UA_POOL = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
]

# Common noise selectors to remove
NOISE_SELECTORS = [
    'blockquote', '.quote', '.reply-quote', '.quote-content',
    'script', 'style', 'nav', 'footer', 'header',
    '.operation', '.report', '.action', '[class*="oper"]',
    '.likes', '[class*="vote"]', '.reply-count',
    '.pubdate', '.user-face', '.comment-time', '[class*="time"]',
    'h4', 'h5',
]


def _random_ua() -> str:
    import random
    return random.choice(UA_POOL)


def fetch_page(url: str, retries: int = 3, timeout: int = 15,
               cookie: Optional[str] = None) -> Tuple[str, int]:
    """Fetch a page with exponential backoff retry.

    Returns:
        (html_content, status_code)
    """
    last_err = None
    headers = {'User-Agent': _random_ua()}
    if cookie:
        headers['Cookie'] = cookie

    for attempt in range(retries):
        try:
            if attempt > 0:
                backoff = 1.0 * (2 ** (attempt - 1))
                time.sleep(backoff)
                logger.debug(f"Retry {attempt+1}/{retries} for {url}")

            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text, resp.status_code

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (429, 500, 502, 503, 504):
                last_err = e
                continue
            raise
        except requests.RequestException as e:
            last_err = e
            if attempt == retries - 1:
                raise

    raise last_err or RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def extract_text_from_html(html: str, main_selector: Optional[str] = None) -> str:
    """Extract clean text from HTML, removing noise elements."""
    soup = BeautifulSoup(html, 'html.parser')

    # Remove noise
    for selector in NOISE_SELECTORS:
        for el in soup.select(selector):
            el.decompose()

    # Focus on main content if selector provided
    if main_selector:
        main = soup.select_one(main_selector)
        if main:
            soup = main

    text = soup.get_text(separator='\n')
    # Clean up whitespace
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    text = '\n'.join(lines)
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def extract_comments_from_html(html: str,
                                comment_selector: str = '.comment-item, .reply-item',
                                username_selector: str = '.comment-info a',
                                content_selector: str = '.comment-content, .reply-content',
                                likes_selector: str = '.likes') -> List[Dict]:
    """Extract structured comments from HTML using CSS selectors.

    Returns list of dicts with keys: username, content, likes
    """
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.select(comment_selector)
    results = []

    for item in items:
        # Username
        user_el = item.select_one(username_selector)
        username = user_el.get_text(strip=True) if user_el else ''

        # Content
        content_el = item.select_one(content_selector)
        if not content_el:
            content_el = item
        # Remove noise from content clone
        clone = BeautifulSoup(str(content_el), 'html.parser')
        for ns in NOISE_SELECTORS:
            for el in clone.select(ns):
                el.decompose()
        content = clone.get_text(separator=' ')
        content = re.sub(r'\s+', ' ', content).strip()

        # Filter obvious noise
        if len(content) < 8:
            continue

        # Likes
        likes = 0
        likes_el = item.select_one(likes_selector)
        if likes_el:
            m = re.search(r'(\d+)', likes_el.get_text())
            if m:
                likes = int(m.group(1))

        results.append({'username': username, 'content': content, 'likes': likes})

    return results


def extract_total_pages(html: str) -> int:
    """Try to detect total page count from paginator HTML."""
    # Method 1: data-total-page attribute
    m = re.search(r'data-total-page=["\']?(\d+)', html)
    if m:
        return int(m.group(1))

    # Method 2: Last page link
    last_page = re.findall(r'start=(\d+)', html)
    if last_page:
        max_start = max(int(s) for s in last_page)
        return (max_start // 100) + 1

    # Method 3: Comment count element
    m = re.search(r'class="[^"]*count[^"]*"[^>]*>[\s\D]*(\d+)', html)
    if m:
        count = int(m.group(1))
        return max(1, (count + 99) // 100)

    return 1


def page_url_hash(url: str) -> str:
    """Hash a URL for cache key."""
    return hashlib.sha256(url.strip().encode()).hexdigest()[:16]
