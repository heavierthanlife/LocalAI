"""Chinese numeral parsing utilities — standard numerals and daxie (大写) amounts."""
import re
from typing import Optional

# Standard Chinese numerals (小写 + 大写 digit mapping)
_CN_NUM = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5,
    '陆': 6, '柒': 7, '捌': 8, '玖': 9, '拾': 10,
    '佰': 100, '仟': 1000, '万': 10000, '亿': 100000000,
}
_CN_UNIT = {'十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}

# Daxie-specific (大写) — used for cross-referencing amounts in bids
_DAXIE_DIGIT = {'壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5,
                '陆': 6, '柒': 7, '捌': 8, '玖': 9, '零': 0}
_DAXIE_UNIT = {'拾': 10, '佰': 100, '仟': 1000, '万': 10000, '亿': 100000000,
               '元': 1, '角': 0.1, '分': 0.01}

_DAXIE_AMOUNT_RE = re.compile(
    r'(?P<daxie>'
    r'(?:壹|贰|叁|肆|伍|陆|柒|捌|玖|拾|佰|仟|万|亿|零|元|角|分|整)+'
    r')'
)


def cn_to_arabic(cn: str) -> Optional[float]:
    """Convert common Chinese numerals to Arabic number."""
    if not cn:
        return None
    if re.fullmatch(r'[\d.,]+', cn):
        try:
            return float(cn.replace(',', ''))
        except ValueError:
            return None

    result = 0.0
    partial = 0.0
    digits = []
    for ch in reversed(cn):
        if ch in _CN_NUM:
            digit = _CN_NUM[ch]
            if digit >= 10:
                if partial == 0:
                    partial = 1
                result += partial * digit
                partial = 0
            else:
                partial += digit
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if partial == 0:
                partial = 1
            result += partial * unit
            partial = 0
            if unit >= 10000:
                unit = 1
        elif ch.isdigit():
            digits.append(ch)
        else:
            continue

    if digits:
        num_str = ''.join(reversed(digits))
        result = float(num_str) * 1
    result += partial
    return result


def parse_daxie_amount(text: str) -> list[dict]:
    """Extract daxie (大写) amounts and cross-reference with Arabic equivalents.

    Returns list of {'daxie': str, 'arabic_from_daxie': float,
                     'context_arabic': float|None, 'match': bool}
    """
    results = []
    for m in _DAXIE_AMOUNT_RE.finditer(text):
        daxie_str = m.group('daxie')
        val = _daxie_to_number(daxie_str)
        if val is None:
            continue
        nearby = text[max(0, m.start() - 50):m.end() + 50]
        arabic_nearby = re.findall(r'[\d,]+\.?\d*\s*(?:万元|元|万)', nearby)
        context_val = None
        for a in arabic_nearby:
            try:
                context_val = float(re.sub(r'[^\d.]', '', a))
                if '万' in a:
                    context_val *= 10000
                break
            except ValueError:
                continue
        results.append({
            'daxie': daxie_str,
            'arabic_from_daxie': val,
            'context_arabic': context_val,
            'match': abs(val - context_val) < 0.01 if context_val is not None else None,
        })
    return results


def _daxie_to_number(daxie: str) -> Optional[float]:
    """Parse a single daxie (大写) amount to its numeric value."""
    digit_map = {'壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5,
                 '陆': 6, '柒': 7, '捌': 8, '玖': 9, '零': 0}
    unit_map = {'拾': 10, '佰': 100, '仟': 1000, '万': 10000, '亿': 100000000}

    result = 0.0
    segment = 0.0
    has_unit = False
    for ch in daxie:
        if ch in digit_map:
            segment = digit_map[ch]
        elif ch in unit_map:
            if segment == 0:
                segment = 1
            unit = unit_map[ch]
            if unit >= 10000:
                result = (result + segment) * unit
                segment = 0
            else:
                segment *= unit
                result += segment
                segment = 0
            has_unit = True
        elif ch == '元':
            result += segment
            segment = 0
        elif ch == '角':
            result += segment * 0.1
            segment = 0
        elif ch == '分':
            result += segment * 0.01
            segment = 0
        elif ch == '整':
            pass
        else:
            continue
    result += segment
    return result if has_unit or result > 0 else None
