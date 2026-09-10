"""LLM-configuration and misc chat routes for the chat blueprint family.

Registered on the shared ``chat_bp`` Blueprint object from
app/routes/chat.py. Covers /set_max_tokens, /llm_providers, /llm_providers/set,
/feedback, /get_recent_files, and /load_cached_file.
"""
import os
import re

from flask import request, jsonify, session

from app.database import get_db_connection
from app.utils.helpers import ok, err, utc_now
import app.globals as g
from app.services.session_manager import get_user_id
from app.services.file_cache import file_cache_manager, add_to_cache
from app.services.anonymous import get_anon_temp_dir
from app.routes.chat import chat_bp, BEIJING_TZ

from psycopg2.extras import RealDictCursor


@chat_bp.route('/set_max_tokens', methods=['POST'])
def set_max_tokens():
    data = request.get_json()
    tokens = data.get('max_tokens', 4800)
    tokens = max(100, min(4800, tokens))
    session['max_tokens'] = tokens
    with g._agent_cache_lock:
        g._agent_cache.clear()
    return jsonify({"success": True, "max_tokens": tokens})

# ── LLM Provider / Model selection ──

@chat_bp.route('/llm_providers', methods=['GET'])
def list_llm_providers():
    """Return available LLM providers and the currently active one.

    FIX-016: includes per-provider model list (static config + daily free-model
    catalog) so the frontend can build a fully dynamic selector.
    """
    try:
        from app.services.llm_provider import get_available_providers, get_active_provider, get_merged_provider_config
    except ImportError:
        return jsonify({"available": [], "active": None, "error": "llm_provider module not loaded"})
    active = get_active_provider()
    providers = {}
    for pid, cfg in get_merged_provider_config().items():
        models = list(cfg.get('models', []))
        # enrich with free-model catalog (whitelist-first)
        try:
            from app.services.llm_catalog import get_free_models
            cat_models = get_free_models(pid, max_results=15)
            if cat_models:
                cat_ids = [m['id'] for m in cat_models]
                # dedupe preserving static order, then append catalog extras
                models = [m for m in models if m not in cat_ids] + cat_ids
        except Exception:
            pass
        providers[pid] = {
            'name': cfg.get('name', pid),
            'models': models,
        }
    return jsonify({
        "available": get_available_providers(),
        "active": active,
        "providers": providers,
    })

@chat_bp.route('/llm_providers/<pid>/models', methods=['GET'])
def get_llm_provider_models(pid):
    """实时获取某 provider 的模型列表（refresh=1 直连 /models，四级兜底）。"""
    from app.services.llm_provider import get_provider_config, get_merged_provider_config
    cfg = None
    try:
        cfg = get_provider_config(pid)
    except Exception:
        cfg = None
    if cfg is None:
        # 自定义 provider 不在静态 PROVIDER_CONFIG，回退到合并视图
        cfg = get_merged_provider_config().get(pid)
    if cfg is None:
        return jsonify({'error': 'unknown provider'}), 404

    refresh = request.args.get('refresh')
    if refresh == '1':
        # 实时拉取：自定义 provider 用其 env_key 读 API key，免费过滤仅限内置 provider
        api_key = os.getenv(cfg.get('env_key', '')) or None
        free_only = not cfg.get('custom')
        models = []
        try:
            from app.services import llm_catalog
            models = llm_catalog._fetch_provider_models(
                cfg.get('base_url'), api_key=api_key, free_only=free_only)
        except Exception:
            models = []
        if models:
            return jsonify({'models': [m['id'] for m in models], 'stale': False})
        # 实时失败/空 → 回退 catalog 缓存 → 再回退静态 models
        stale_models = []
        try:
            from app.services.llm_catalog import get_free_models
            cat_models = get_free_models(pid)
            if cat_models:
                stale_models = [m['id'] for m in cat_models]
        except Exception:
            stale_models = []
        if not stale_models:
            stale_models = list(cfg.get('models', []))
        return jsonify({'models': stale_models, 'stale': True})

    # 非 refresh → 返回当前缓存 models（静态 + catalog 免费）
    cached = list(cfg.get('models', []))
    try:
        from app.services.llm_catalog import get_free_models
        cat_models = get_free_models(pid, max_results=15)
        if cat_models:
            cat_ids = [m['id'] for m in cat_models]
            cached = [m for m in cached if m not in cat_ids] + cat_ids
    except Exception:
        pass
    return jsonify({'models': cached})

@chat_bp.route('/llm_providers/set', methods=['POST'])
def set_llm_provider():
    """Set the active LLM provider and model via session."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Login required"}), 403
    data = request.get_json()
    provider = data.get('provider', '').strip()
    model = data.get('model', '').strip()
    if provider:
        session['llm_provider'] = provider
    if model:
        session['llm_model'] = model
    # Invalidate agent cache so next request picks up new model
    with g._agent_cache_lock:
        g._agent_cache.clear()
    return jsonify({"success": True, "provider": provider, "model": model})

@chat_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Feedback not allowed – no consent"}), 403
    data = request.get_json()
    rating = data.get('rating')
    comment = data.get('comment', '')
    user_message = data.get('user_message')
    assistant_response = data.get('assistant_response')
    if not user_message or not assistant_response:
        user_message = session.get('last_user_msg', '')
        assistant_response = session.get('last_assistant_msg', '')
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (thread_id, user_message, assistant_response, rating, comment, timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
                (session['thread_id'], user_message, assistant_response, rating, comment, utc_now())
            )
            conn.commit()
    # Log to training data pipeline
    try:
        from app.services.training_logger import log_interaction
        log_interaction(session['thread_id'], user_message, assistant_response,
                       rating=rating, rating_comment=comment)
    except Exception:
        pass
    return jsonify({"status": "ok"})

@chat_bp.route('/get_recent_files', methods=['GET'])
def get_recent_files():
    thread_id = session.get('thread_id')
    if not thread_id:
        return jsonify({"recent_files": []})
    recent = file_cache_manager.get_recent_with_lock(thread_id)
    files_with_usage = []
    if session.get('consent_value', 0) == 1:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for filename in recent:
                    cur.execute(
                        """SELECT usage_type, question, timestamp
                           FROM file_usage
                           WHERE thread_id = %s
                             AND filename = %s
                           ORDER BY timestamp DESC
                           LIMIT 5""",
                        (thread_id, filename)
                    )
                    usage_records = []
                    for row in cur.fetchall():
                        ts_utc = row['timestamp']
                        if ts_utc:
                            ts_beijing = ts_utc.astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            ts_beijing = None
                        usage_records.append({
                            "type": row['usage_type'],
                            "question": row['question'],
                            "time": ts_beijing
                        })
                    files_with_usage.append({
                        "filename": filename,
                        "usage": usage_records
                    })
    else:
        for filename in recent:
            files_with_usage.append({"filename": filename, "usage": []})
    return jsonify({"recent_files": files_with_usage})

@chat_bp.route('/load_cached_file', methods=['POST'])
def load_cached_file():
    data = request.get_json()
    filename = data.get('filename')
    thread_id = session.get('thread_id')
    if not thread_id:
        return jsonify({"error": "Session expired"}), 401
    content = file_cache_manager.get_content(thread_id, filename)
    if content:
        return jsonify({"content": content})
    if session.get('consent_value', 0) != 1:
        user_id = get_user_id()
        temp_dir = get_anon_temp_dir(user_id)
        safe_name = re.sub(r'[^\w\-_\. ]', '_', filename) + '.txt'
        fpath = os.path.join(temp_dir, safe_name)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            add_to_cache(thread_id, filename, content, user_id)
            return jsonify({"content": content})
        else:
            return jsonify({"error": "File not found"}), 404
    user_id = get_user_id()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content FROM user_files WHERE user_id = %s AND filename = %s AND (expires_at IS NULL OR expires_at > NOW())",
                (user_id, filename)
            )
            row = cur.fetchone()
            if row:
                content = row[0] or ''
                add_to_cache(thread_id, filename, content, user_id)
                return jsonify({"content": content})
    return jsonify({"error": "File not found"}), 404


# ── Per-user prompts (immutable default + custom versions + templates) ──

def _prompt_user_id():
    return session.get('user_id') or ''


@chat_bp.route('/prompts/default', methods=['GET'])
def prompts_default():
    """返回不可变默认主 agent 提示词（原始版，不含 guard）。"""
    from app.globals import get_default_prompt
    return ok({'prompt': get_default_prompt()})


@chat_bp.route('/prompts/mine', methods=['GET'])
def prompts_mine():
    """返回当前用户的 agent 版本与消息模板。"""
    user_id = _prompt_user_id()
    if not user_id:
        return err("未登录", "AUTH_REQUIRED", 401)
    from app.services.user_prompt import list_user_prompts
    return ok(list_user_prompts(user_id))


@chat_bp.route('/prompts/save', methods=['POST'])
def prompts_save():
    """新建/更新用户提示词或模板。"""
    user_id = _prompt_user_id()
    if not user_id:
        return err("未登录", "AUTH_REQUIRED", 401)
    data = request.get_json(silent=True) or {}
    from app.services.user_prompt import save_user_prompt
    try:
        pid = save_user_prompt(
            user_id,
            data.get('kind'),
            data.get('content'),
            name=data.get('name'),
            pid=data.get('id'),
        )
    except ValueError as e:
        return err(str(e), "VALIDATION_ERROR", 400)
    return ok({'id': pid})


@chat_bp.route('/prompts/activate', methods=['POST'])
def prompts_activate():
    """将某个 agent 版本设为当前用户唯一 active。"""
    user_id = _prompt_user_id()
    if not user_id:
        return err("未登录", "AUTH_REQUIRED", 401)
    data = request.get_json(silent=True) or {}
    from app.services.user_prompt import activate_prompt
    try:
        activate_prompt(user_id, data.get('id'))
    except ValueError as e:
        return err(str(e), "VALIDATION_ERROR", 400)
    return ok({})


@chat_bp.route('/prompts/delete', methods=['POST'])
def prompts_delete():
    """删除用户提示词/模板（校验归属）。"""
    user_id = _prompt_user_id()
    if not user_id:
        return err("未登录", "AUTH_REQUIRED", 401)
    data = request.get_json(silent=True) or {}
    from app.services.user_prompt import delete_prompt
    try:
        delete_prompt(user_id, data.get('id'))
    except ValueError as e:
        return err(str(e), "VALIDATION_ERROR", 400)
    return ok({})


@chat_bp.route('/prompts/migrate_templates', methods=['POST'])
def prompts_migrate_templates():
    """一次性导入消息模板（受 ≤5 约束，超出截断）。"""
    user_id = _prompt_user_id()
    if not user_id:
        return err("未登录", "AUTH_REQUIRED", 401)
    data = request.get_json(silent=True) or {}
    from app.services.user_prompt import migrate_templates
    imported = migrate_templates(user_id, data.get('templates') or [])
    return ok({'imported': imported})
