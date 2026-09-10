"""Per-user agent prompts and message templates.

Replaces the old single global agent prompt.  Each user may store up to
``AGENT_MAX`` agent prompt versions (at most one active) and ``TEMPLATE_MAX``
message templates.  Raw content is stored as-is; the safety guard is appended
only when a prompt is resolved for actual use (see ``resolve_user_prompt``).
"""
import logging

from app.database import get_db_connection
from app.globals import get_default_prompt
from app.services.prompt_safety import build_safe_system_guard

logger = logging.getLogger(__name__)

AGENT_MAX = 2
TEMPLATE_MAX = 5
VALID_KINDS = ('agent', 'template')


def _clean(value) -> str:
    return (value or '').strip()


def resolve_user_prompt(user_id: str) -> str:
    """Return the active agent system prompt for a user, safety guard included.

    取该用户 kind='agent' 且 is_active 的行；无 active 或 user_id 为空时回退
    不可变默认提示词。guard 已存在则不重复追加。
    """
    if not user_id:
        return get_default_prompt()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM user_prompts "
                    "WHERE user_id = %s AND kind = 'agent' AND is_active = TRUE "
                    "ORDER BY updated_at DESC, id DESC LIMIT 1",
                    (user_id,),
                )
                row = cur.fetchone()
    except Exception as e:
        logger.error(f"resolve_user_prompt failed for user={user_id}: {e}", exc_info=True)
        return get_default_prompt()
    if not row or not _clean(row[0]):
        return get_default_prompt()
    content = row[0]
    guard = build_safe_system_guard()
    if guard in content:
        return content
    return content + '\n' + guard


def list_user_prompts(user_id: str) -> dict:
    """Return the user's agent versions and templates (for GET /prompts/mine)."""
    result = {
        'agent': [],
        'templates': [],
        'agent_active_id': None,
        'agent_max': AGENT_MAX,
        'template_max': TEMPLATE_MAX,
    }
    if not user_id:
        return result
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, content, is_active, updated_at FROM user_prompts "
                    "WHERE user_id = %s AND kind = 'agent' ORDER BY updated_at DESC, id DESC",
                    (user_id,),
                )
                for row in cur.fetchall():
                    result['agent'].append({
                        'id': row[0],
                        'name': row[1] or '',
                        'content': row[2] or '',
                        'is_active': bool(row[3]),
                        'updated_at': row[4].isoformat() if row[4] else None,
                    })
                    if row[3]:
                        result['agent_active_id'] = row[0]
                cur.execute(
                    "SELECT id, name, content, updated_at FROM user_prompts "
                    "WHERE user_id = %s AND kind = 'template' ORDER BY updated_at DESC, id DESC",
                    (user_id,),
                )
                for row in cur.fetchall():
                    result['templates'].append({
                        'id': row[0],
                        'name': row[1] or '',
                        'content': row[2] or '',
                        'updated_at': row[3].isoformat() if row[3] else None,
                    })
    except Exception as e:
        logger.error(f"list_user_prompts failed for user={user_id}: {e}", exc_info=True)
    return result


def save_user_prompt(user_id: str, kind: str, content: str, name=None, pid=None) -> int:
    """Create or update a prompt/template; returns the row id.

    新建时校验限额（agent ≤2 / template ≤5），超限抛 ValueError。
    首个 agent 版本自动设为 active，保证开箱即用。
    """
    if not user_id:
        raise ValueError("未登录")
    kind = (kind or '').strip()
    if kind not in VALID_KINDS:
        raise ValueError("kind 必须为 agent 或 template")
    content = (content or '').strip()
    if not content:
        raise ValueError("content 不能为空")
    name = _clean(name)
    limit = AGENT_MAX if kind == 'agent' else TEMPLATE_MAX
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 更新已有记录（校验归属）
            if pid:
                cur.execute(
                    "SELECT 1 FROM user_prompts WHERE id = %s AND user_id = %s",
                    (pid, user_id),
                )
                if not cur.fetchone():
                    raise ValueError("记录不存在或无权访问")
                cur.execute(
                    "UPDATE user_prompts SET content = %s, "
                    "name = COALESCE(NULLIF(%s, ''), name), updated_at = NOW() "
                    "WHERE id = %s AND user_id = %s",
                    (content, name, pid, user_id),
                )
                conn.commit()
                return pid
            # 新建：限额校验
            cur.execute(
                "SELECT COUNT(*) FROM user_prompts WHERE user_id = %s AND kind = %s",
                (user_id, kind),
            )
            count = cur.fetchone()[0]
            if count >= limit:
                if kind == 'agent':
                    raise ValueError(f"最多保存 {limit} 个版本，请先删除一个")
                raise ValueError(f"最多保存 {limit} 个模板，请先删除一个")
            make_active = kind == 'agent' and count == 0
            if not name:
                name = f"版本 {count + 1}" if kind == 'agent' else f"模板 {count + 1}"
            cur.execute(
                "INSERT INTO user_prompts (user_id, kind, name, content, is_active) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (user_id, kind, name, content, make_active),
            )
            new_id = cur.fetchone()[0]
            conn.commit()
            return new_id


def activate_prompt(user_id: str, pid) -> bool:
    """Set the given agent prompt as the user's sole active version."""
    if not user_id or not pid:
        raise ValueError("参数不完整")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM user_prompts WHERE id = %s AND user_id = %s AND kind = 'agent'",
                (pid, user_id),
            )
            if not cur.fetchone():
                raise ValueError("记录不存在或无权访问")
            # 保留 updated_at，避免仅激活导致列表排序跳动
            cur.execute(
                "UPDATE user_prompts SET is_active = FALSE "
                "WHERE user_id = %s AND kind = 'agent' AND is_active = TRUE",
                (user_id,),
            )
            cur.execute(
                "UPDATE user_prompts SET is_active = TRUE, updated_at = NOW() "
                "WHERE id = %s AND user_id = %s",
                (pid, user_id),
            )
            conn.commit()
    return True


def delete_prompt(user_id: str, pid) -> bool:
    """Delete a user-owned prompt/template. 删除 active agent 后回退默认。"""
    if not user_id or not pid:
        raise ValueError("参数不完整")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM user_prompts WHERE id = %s AND user_id = %s",
                (pid, user_id),
            )
            deleted = cur.rowcount
            conn.commit()
    if not deleted:
        raise ValueError("记录不存在或无权访问")
    return True


def migrate_templates(user_id: str, templates) -> int:
    """One-shot import of message templates; truncated to the ≤5 limit.

    返回实际导入条数。
    """
    if not user_id or not templates:
        return 0
    imported = 0
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM user_prompts WHERE user_id = %s AND kind = 'template'",
                (user_id,),
            )
            count = cur.fetchone()[0]
            for item in templates:
                if count >= TEMPLATE_MAX:
                    break
                if not isinstance(item, dict):
                    continue
                content = (item.get('content') or '').strip()
                if not content:
                    continue
                name = (item.get('name') or '').strip() or f"模板 {count + 1}"
                cur.execute(
                    "INSERT INTO user_prompts (user_id, kind, name, content, is_active) "
                    "VALUES (%s, 'template', %s, %s, FALSE)",
                    (user_id, name, content),
                )
                count += 1
                imported += 1
            conn.commit()
    return imported
