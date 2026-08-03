"""Law text parser — splits legal documents into chapters and articles.

Designed for PRC legal texts (bid/tender law domain) with standard structure:
    第X章 标题
        第X条 内容...
        第X条 内容...

Supports 附则 as a special chapter marker.
"""

import re
import logging

logger = logging.getLogger(__name__)

CHAPTER_PATTERN = re.compile(
    r'^第[一二三四五六七八九十百千\d]+章\b',
    re.UNICODE
)
ARTICLE_PATTERN = re.compile(
    r'^第[一二三四五六七八九十百千\d]+条\b',
    re.UNICODE
)
APPENDIX_PATTERN = re.compile(
    r'^附\s*则\b',
    re.UNICODE
)


def _is_chapter_line(text: str) -> bool:
    stripped = text.strip()
    if APPENDIX_PATTERN.match(stripped):
        return True
    return bool(CHAPTER_PATTERN.match(stripped))


def _is_article_line(text: str) -> bool:
    return bool(ARTICLE_PATTERN.match(text.strip()))


def split_law_text(text: str) -> str:
    """Convert plain law text into structured markdown.

    Splits on 第X章 (chapter) and 第X条 (article) markers, treating everything
    between two article markers as belonging to the first article.

    Also handles 附则 as the final chapter marker.

    Returns markdown with # chapter headers and ## article headers.
    """
    if not text:
        return ""

    lines = text.split('\n')
    out_lines = []
    in_article = False
    in_preamble = True  # everything before the first chapter or article

    for line in lines:
        stripped = line.strip()

        if not stripped:
            out_lines.append('')
            continue

        if _is_chapter_line(stripped):
            in_preamble = False
            in_article = False
            out_lines.append('')
            out_lines.append(f'# {stripped}')
            out_lines.append('')
            continue

        if _is_article_line(stripped):
            in_preamble = False
            in_article = True
            out_lines.append('')
            out_lines.append(f'## {stripped}')
            out_lines.append('')
            continue

        if in_article:
            out_lines.append(stripped)
        elif in_preamble:
            out_lines.append(stripped)

    result = '\n'.join(out_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = result.strip() + '\n'

    article_count = sum(1 for _ in ARTICLE_PATTERN.finditer(result))
    chapter_count = sum(1 for _ in CHAPTER_PATTERN.finditer(result))
    if APPENDIX_PATTERN.search(result):
        chapter_count += 1
    logger.info(
        f"Law text split: {chapter_count} chapters, {article_count} articles"
    )
    return result
