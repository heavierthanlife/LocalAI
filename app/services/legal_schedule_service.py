"""Legal schedule service — query bidding milestone templates from seed data."""
import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from app.config import DATA_DIR
from app.database import get_db_connection

logger = logging.getLogger(__name__)

SCHEDULES_PATH = os.path.join(DATA_DIR, 'legal_schedules', 'default_schedules.json')

_schedules_cache: Optional[Dict] = None


def _load_schedules() -> Dict:
    global _schedules_cache
    if _schedules_cache is not None:
        return _schedules_cache
    with open(SCHEDULES_PATH, 'r', encoding='utf-8') as f:
        _schedules_cache = json.load(f)
    return _schedules_cache


def _is_working_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    try:
        from chinese_calendar import is_holiday, is_workday
        if is_workday(d):
            return True
        if is_holiday(d):
            return False
    except ImportError:
        pass
    return d.weekday() < 5


def _add_working_days(start: date, days: int) -> date:
    current = start
    remaining = days
    while remaining > 0:
        current += timedelta(days=1)
        if _is_working_day(current):
            remaining -= 1
    return current


def _add_calendar_days(start: date, days: int) -> date:
    return start + timedelta(days=days)


def get_categories() -> List[Dict]:
    schedules = _load_schedules()
    return schedules.get('categories', [])


def get_category_info(category_code: str) -> Optional[Dict]:
    for cat in get_categories():
        if cat['code'] == category_code:
            regimes = _load_schedules().get('regimes', {})
            regime_key = cat.get('regime', '')
            regime = regimes.get(regime_key, {})
            applicable_to = regime.get('applicable_to', [])
            return {
                **cat,
                'regime_name': regime.get('name', ''),
                'law': regime.get('law', ''),
                'regulation': regime.get('regulation', ''),
                'methods': list(regime.get('methods', {}).keys()),
            }
    return None


def get_methods(category_code: str) -> List[Dict]:
    cat_info = get_category_info(category_code)
    if not cat_info:
        return []
    regimens = _load_schedules().get('regimes', {})
    regime = regimens.get(cat_info.get('regime', ''), {})
    methods = regime.get('methods', {})
    return [
        {'code': k, 'name': v['name'], 'description': v.get('description', '')}
        for k, v in methods.items()
    ]


def get_schedule(category_code: str, method_code: str) -> Optional[Dict]:
    cat_info = get_category_info(category_code)
    if not cat_info:
        return None
    regimes = _load_schedules().get('regimes', {})
    regime = regimes.get(cat_info.get('regime', ''), {})
    methods = regime.get('methods', {})
    method = methods.get(method_code)
    if not method:
        return None
    return {
        'category_code': category_code,
        'category_name': cat_info['name'],
        'regime': cat_info.get('regime', ''),
        'regime_name': regime.get('name', ''),
        'method_code': method_code,
        'method_name': method['name'],
        'method_description': method.get('description', ''),
        'law_ref': method.get('law_ref', ''),
        'milestones': method['milestones'],
    }


def compute_planned_dates(milestones: List[Dict], start_date: date) -> List[Dict]:
    result = []
    milestone_map = {}
    for m in milestones:
        m = dict(m)
        planned_date = None
        if m.get('days_from_start') is not None:
            days = m['days_from_start']
            if m.get('date_type') == 'working':
                planned_date = _add_working_days(start_date, days)
            else:
                planned_date = _add_calendar_days(start_date, days)
        else:
            prev_code = m.get('prev_milestone_code')
            if prev_code and prev_code in milestone_map:
                prev_milestone = milestone_map[prev_code]
                prev_date = prev_milestone.get('_planned_date')
                if prev_date:
                    days = m.get('days_from_prev_milestone')
                    if days is not None:
                        if m.get('date_type') == 'working':
                            planned_date = _add_working_days(prev_date, days)
                        else:
                            planned_date = _add_calendar_days(prev_date, days)
                    elif m.get('duration_days'):
                        planned_date = _add_calendar_days(prev_date, m['duration_days'])
            elif m.get('duration_days'):
                planned_date = _add_calendar_days(start_date, m['duration_days'])

        m['_planned_date'] = planned_date
        milestone_map[m['code']] = m
        result.append(m)
    return result


def get_deadline_warning(milestones: List[Dict]) -> List[Dict]:
    today = date.today()
    warnings = []
    for m in milestones:
        planned = m.get('_planned_date')
        if not planned:
            continue
        actual = m.get('actual_date')
        if actual:
            continue
        days_until = (planned - today).days
        if days_until < 0:
            level = 'overdue'
        elif days_until <= 3:
            level = 'critical'
        elif days_until <= 7:
            level = 'warning'
        else:
            continue
        warnings.append({
            'code': m['code'],
            'name': m['name'],
            'planned_date': planned.isoformat(),
            'days_until': days_until,
            'level': level,
            'mandatory': m.get('mandatory', False),
        })
    return warnings


def seed_database():
    schedules = _load_schedules()
    categories = schedules.get('categories', [])
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for cat in categories:
                cur.execute(
                    "INSERT INTO bidding_categories (code, name_zh, regime) VALUES (%s, %s, %s) ON CONFLICT (code) DO UPDATE SET name_zh = EXCLUDED.name_zh, regime = EXCLUDED.regime",
                    (cat['code'], cat['name'], cat.get('regime', ''))
                )
            conn.commit()

        with conn.cursor() as cur:
            sort_order = 0
            for cat in categories:
                cat_info = get_category_info(cat['code'])
                if not cat_info:
                    continue
                regime_key = cat.get('regime', '')
                regime = schedules.get('regimes', {}).get(regime_key, {})
                methods = regime.get('methods', {})
                for method_code, method_data in methods.items():
                    for milestone in method_data['milestones']:
                        sort_order += 1
                        cur.execute("""
                            INSERT INTO bidding_schedule_templates
                                (category_code, method_code, milestone_code, milestone_name,
                                 days_from_start, days_from_prev_milestone, prev_milestone_code,
                                 duration_days, date_type, mandatory, law_ref, description, sort_order)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (category_code, method_code, milestone_code)
                            DO UPDATE SET
                                milestone_name = EXCLUDED.milestone_name,
                                days_from_start = EXCLUDED.days_from_start,
                                days_from_prev_milestone = EXCLUDED.days_from_prev_milestone,
                                prev_milestone_code = EXCLUDED.prev_milestone_code,
                                duration_days = EXCLUDED.duration_days,
                                date_type = EXCLUDED.date_type,
                                mandatory = EXCLUDED.mandatory,
                                law_ref = EXCLUDED.law_ref,
                                description = EXCLUDED.description,
                                sort_order = EXCLUDED.sort_order
                        """, (
                            cat['code'], method_code, milestone['code'], milestone['name'],
                            milestone.get('days_from_start'),
                            milestone.get('days_from_prev_milestone'),
                            milestone.get('prev_milestone_code'),
                            milestone.get('duration_days'),
                            milestone.get('date_type', 'calendar'),
                            milestone.get('mandatory', False),
                            milestone.get('law_ref', ''),
                            milestone.get('description', ''),
                            sort_order,
                        ))
            conn.commit()

    logger.info("Bidding schedule seed data loaded from %s", SCHEDULES_PATH)


def seed_wiki_pages():
    from app.services import wiki_engine

    wiki_engine._ensure_wiki_dir()
    legal_dir = os.path.join(wiki_engine.WIKI_DIR, 'legal', 'schedules')
    os.makedirs(legal_dir, exist_ok=True)

    schedules = _load_schedules()
    laws = schedules.get('laws_referenced', [])
    regimes = schedules.get('regimes', {})

    for regime_key, regime in regimes.items():
        methods = regime.get('methods', {})
        for method_code, method_data in methods.items():
            filename = f"{regime_key}_{method_code}.md"
            filepath = os.path.join(legal_dir, filename)

            fm = {
                'title': f"{method_data['name']} — {regime['name']}",
                'regime': regime_key,
                'regime_name': regime['name'],
                'method': method_code,
                'method_name': method_data['name'],
                'law': regime.get('law', ''),
                'regulation': regime.get('regulation', ''),
                'type': 'legal_schedule',
                'generated': datetime.now().isoformat(),
            }

            lines = [
                f"# {method_data['name']}",
                f"",
                f"**法律体系:** {regime['name']}  "
                f"**法律:** {regime.get('law', '')}  "
                f"**行政法规:** {regime.get('regulation', '')}",
                f"",
                f"## 说明",
                f"",
                f"{method_data.get('description', '')}",
                f"",
                f"法律依据: {method_data.get('law_ref', '')}",
                f"",
                f"## 法定时间节点",
                f"",
            ]

            for m in method_data['milestones']:
                days_info = ''
                if m.get('days_from_start') is not None:
                    d_type = '工作日' if m.get('date_type') == 'working' else '日历日'
                    days_info = f"**D+{m['days_from_start']}** ({d_type})"
                elif m.get('days_from_prev_milestone') is not None:
                    d_type = '工作日' if m.get('date_type') == 'working' else '日历日'
                    prev_name = ''
                    for pm in method_data['milestones']:
                        if pm['code'] == m.get('prev_milestone_code'):
                            prev_name = pm['name']
                            break
                    days_info = f"**前节点后+{m['days_from_prev_milestone']}** ({d_type}, 相对'{prev_name}')"
                elif m.get('duration_days'):
                    days_info = f"**预计{m['duration_days']}日**"

                mandatory_mark = '⚠️ 强制' if m.get('mandatory') else ''

                lines.append(f"- **{m['code']}** — {m['name']}  {days_info}  {mandatory_mark}")
                if m.get('law_ref'):
                    lines.append(f"  - 法律依据: {m['law_ref']}")
                if m.get('description'):
                    lines.append(f"  - {m['description']}")

            lines.append("")
            lines.append("## 参考法律法规")
            lines.append("")
            for law in laws:
                lines.append(f"- [{law['name']}]({law.get('url', '#')}) ({law.get('year', '')})")

            content = '\n'.join(lines)
            wiki_engine.write_wiki_page(
                f"legal/schedules/{filename}", fm, content
            )

    logger.info("Wiki legal schedule pages seeded to %s", legal_dir)


def reload_schedules():
    global _schedules_cache
    _schedules_cache = None
    return _load_schedules()
