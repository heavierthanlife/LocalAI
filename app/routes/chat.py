"""Blueprint: chat routes (auto-extracted)."""
import os, re, json, uuid, time, logging, hashlib, io, shutil, secrets
from datetime import timezone, timedelta, datetime
from io import BytesIO
from flask import Blueprint, request, jsonify, session, send_file, render_template, url_for, Response, stream_with_context
from flask_wtf.csrf import generate_csrf
from werkzeug.datastructures import FileStorage
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from app.config import (
    BASE_DIR, DATA_DIR, TEMP_ROOT, TEMP_DIR, USER_FILES_ORIGINAL_ROOT,
    to_rel_path, resolve_path,
    is_valid_extracted_text, ALLOWED_EXTENSIONS, allowed_file, logger as config_logger,
)
from app.database import get_db_connection, db_transaction
from app.utils.helpers import utc_now, beijing_now, safe_error_response, split_thinking_answer
import app.globals as g
from app.services.file_cache import file_cache_manager, add_to_cache, load_cache_from_db

from psycopg2.extras import RealDictCursor
from app.utils.headroom_utils import compress_file_content as headroom_compress

from app.services.session_manager import (
    get_user_id, ensure_user_exists, get_or_create_session,
    get_session_messages, get_user_sessions, store_message,
    delete_session, archive_session, record_file_usage,
    get_chat_short_name, get_user_total_storage_size, db_execute_readonly,
    cleanup_old_sessions,
)
from app.services.anonymous import get_session_messages_anon, get_anon_temp_dir, cleanup_anon_temp
from app.services.file_processing import extract_text_from_file, get_or_extract_file_analysis
from app.routes.admin import is_admin
from app.routes.projects import can_access_project
from app.services.agent import get_date, bocha_search, get_agent
from app.services.redteam_agent import get_redteam_agent, REDTEAM_SYSTEM_PROMPT
from app.services.task_locking import acquire_task_lock, release_task_lock, cleanup_stale_tasks

logger = logging.getLogger(__name__)
BEIJING_TZ = timezone(timedelta(hours=8))

chat_bp = Blueprint('chat', __name__, template_folder=str(BASE_DIR / 'templates'), static_folder=str(BASE_DIR / 'static'))

# No-op poll endpoint — silences legacy frontend polling 404s
@chat_bp.route('/chat/poll/<thread_id>', methods=['GET'])
def chat_poll_noop(thread_id):
    return '', 204

@chat_bp.route('/')
def index():
    if 'consent_value' not in session:
        session['consent_value'] = 0
    if 'thread_id' not in session:
        session['thread_id'] = str(uuid.uuid4())
        get_or_create_session(session['thread_id'])
    if 'chat_history' not in session:
        session['chat_history'] = get_session_messages(session['thread_id'])
    user_id = get_user_id()
    load_cache_from_db(session['thread_id'], user_id)
    return render_template('index.html',
                           consent_given=(session.get('consent_value', 0) == 1),
                           chat_history=session['chat_history'],
                           recent_files=file_cache_manager.get_recent_with_lock(session['thread_id']))

@chat_bp.route('/get_csrf_token', methods=['GET'])
def get_csrf_token():
    return jsonify({'csrf_token': generate_csrf()})

@chat_bp.route('/logout', methods=['POST'])
def logout():
    session.clear()
    session['consent_value'] = 0
    session['thread_id'] = str(uuid.uuid4())
    get_or_create_session(session['thread_id'])
    return jsonify({"status": "ok"})


@chat_bp.route('/favicon.ico')
def favicon():
    favicon_path = os.path.join(os.getcwd(), 'static', 'favicon.ico')
    if os.path.isfile(favicon_path):
        return send_file(favicon_path, mimetype='image/vnd.microsoft.icon')
    # Return empty response to avoid 500 error
    return '', 204

@chat_bp.route('/share_conversation', methods=['POST'])
def share_conversation():
    """Generate a shareable token for the current conversation."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    thread_id = session.get('thread_id')
    if not thread_id:
        return jsonify({"error": "No active conversation"}), 400

    messages = get_session_messages(thread_id)
    if not messages:
        return jsonify({"error": "No messages to share"}), 400

    token = secrets.token_urlsafe(16)
    share_path = os.path.join(TEMP_DIR, f"share_{token}.json")
    with open(share_path, 'w', encoding='utf-8') as f:
        json.dump({
            'thread_id': thread_id,
            'messages': messages,
            'created_at': beijing_now(),
            'expires_at': (datetime.now(BEIJING_TZ) + timedelta(days=7)).isoformat()
        }, f, ensure_ascii=False)

    share_url = url_for('.view_shared_conversation', token=token, _external=True)
    return jsonify({"share_url": share_url, "token": token})

@chat_bp.route('/shared/<token>')
def view_shared_conversation(token):
    """View a shared conversation."""
    # Validate token format to prevent path traversal (token_urlsafe produces [A-Za-z0-9_-])
    if not re.fullmatch(r'[A-Za-z0-9_-]{16,64}', token):
        return render_template('index.html', consent_given=False, chat_history=[],
                              shared_error="无效的分享链接")
    share_path = os.path.realpath(os.path.join(TEMP_DIR, f"share_{token}.json"))
    # Ensure resolved path stays within TEMP_DIR
    real_temp = os.path.realpath(TEMP_DIR)
    if not share_path.startswith(real_temp + os.sep) or not os.path.exists(share_path):
        return render_template('index.html', consent_given=False, chat_history=[],
                              shared_error="此分享链接已过期或不存在")
    try:
        with open(share_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return render_template('index.html', consent_given=False,
                              chat_history=data['messages'],
                              shared_mode=True)
    except Exception:
        return render_template('index.html', consent_given=False, chat_history=[],
                              shared_error="分享数据已损坏")

@chat_bp.route('/send_stream', methods=['POST'])
def send_message_stream():
    """Streaming SSE endpoint — open to all users including anonymous."""
    user_msg = request.form.get('message', '').strip()
    if len(user_msg) > 10000:
        return jsonify({"error": "消息不能超过10000字"}), 400
    if not user_msg and 'files' not in request.files:
        return jsonify({"error": "Empty message and no files"}), 400

    thread_id = session.get('thread_id', str(uuid.uuid4()))
    session['thread_id'] = thread_id
    is_anon = session.get('consent_value', 0) != 1
    user_id = get_user_id()
    get_or_create_session(thread_id)

    is_grilling = False
    try:
        from app.database import get_db_connection
        from psycopg2.extras import RealDictCursor
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT is_grilling FROM chat_sessions WHERE thread_id = %s", (thread_id,))
                row = cur.fetchone()
                if row and row.get('is_grilling'):
                    is_grilling = True
    except Exception as e:
        logger.warning(f"Failed to check is_grilling: {e}")


    # Store user message — DB for registered, memory for anonymous
    if is_anon:
        session.setdefault('chat_history', []).append({'role': 'user', 'content': user_msg})
    else:
        store_message(thread_id, 'user', user_msg)

    # Build the query the same way as the non-streaming route
    has_files = any(f.filename for f in request.files.getlist('files'))
    knowledge_files_json = request.form.get('knowledge_files')
    knowledge_files = json.loads(knowledge_files_json) if knowledge_files_json else []

    file_text = ""
    if has_files:
        file_contents = []
        for f in request.files.getlist('files'):
            if not f.filename or not allowed_file(f.filename):
                continue
            content, _ = extract_text_from_file(f)
            if content and not content.startswith("["):
                file_contents.append(content)
        if file_contents:
            file_text = "File content(s):\n" + "\n\n".join(file_contents) + "\n\nUser query:\n" + user_msg
        else:
            file_text = user_msg
    else:
        file_text = user_msg

    # RAG retrieval for knowledge base files
    kb_context = ""
    if knowledge_files:
        rag_sources = [f['source'] for f in knowledge_files if f['source'] in ('knowledge_lab', 'company_kb')]
        if rag_sources:
            try:
                from app.services.rag_engine import build_rag_context
                kb_context = build_rag_context(
                    user_msg, list(set(rag_sources)), top_k=8, max_chars=6000
                )
            except Exception as e:
                logger.warning(f"Stream RAG failed: {e}")

    from app.services.prompt_safety import wrap_user_content, sanitize_for_prompt
    if kb_context:
        query = f"{wrap_user_content(kb_context, 'KNOWLEDGE_BASE')}\n\n{wrap_user_content(sanitize_for_prompt(file_text, 'stream_file'), 'UPLOADED_FILES')}"
    else:
        query = sanitize_for_prompt(file_text, 'stream_query')

    # ── Quote chain context for project chats (3-tier priority) ──
    quote_chain_context = ""
    quoted_message_id = request.form.get('quoted_message_id')
    if quoted_message_id:
        try:
            from app.database import get_db_connection
            from psycopg2.extras import RealDictCursor
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Find the project_id for this thread
                    cur.execute("SELECT project_id FROM chat_sessions WHERE thread_id = %s", (thread_id,))
                    sess_row = cur.fetchone()
                    if sess_row and sess_row['project_id']:
                        proj_id = sess_row['project_id']
                        # Fetch the quoted message content
                        cur.execute("SELECT id, role, content, timestamp FROM chat_messages WHERE id = %s", (quoted_message_id,))
                        quoted_msg = cur.fetchone()
                        if quoted_msg:
                            # Store full quoted message (no truncation — AI needs ALL of it)
                            quoted_full_content = quoted_msg['content'] or ''
                            quoted_role = quoted_msg['role']

                            # Traverse quote tree ancestry (for context depth)
                            chain_nodes = []
                            current_msg_id = int(quoted_message_id)
                            visited = set()
                            while current_msg_id and current_msg_id not in visited:
                                visited.add(current_msg_id)
                                cur.execute("""
                                    SELECT cm.id, cm.role, cm.content, cm.timestamp,
                                           mq.id as quote_id, mq.parent_quote_id
                                    FROM chat_messages cm
                                    LEFT JOIN message_quotes mq ON mq.quoted_message_id = cm.id
                                    WHERE cm.id = %s
                                """, (current_msg_id,))
                                row = cur.fetchone()
                                if not row:
                                    break
                                chain_nodes.append({
                                    'role': row['role'],
                                    'content': row['content'],
                                    'msg_id': row['id'],
                                })
                                if row['parent_quote_id']:
                                    cur.execute("SELECT quoted_message_id FROM message_quotes WHERE id = %s", (row['parent_quote_id'],))
                                    parent = cur.fetchone()
                                    if parent:
                                        current_msg_id = parent['quoted_message_id']
                                    else:
                                        break
                                else:
                                    break
                            # Build context with FULL immediate quote + truncated ancestors
                            chain_nodes.reverse()
                            parts = []
                            # The last node is the immediate quoted message — show FULL content
                            if chain_nodes:
                                immediate = chain_nodes[-1]
                                role_label = '用户' if immediate['role'] == 'user' else 'AI'
                                quoted_full = immediate['content'][:4000]  # 4000 chars max, enough for document context
                                parts.append(f"[QUOTED] ({role_label}):\n{quoted_full}")
                                # Ancestors (if any) — truncated for context
                                if len(chain_nodes) > 1:
                                    ancestors = chain_nodes[:-1]
                                    ancestor_text = '\n'.join([
                                        f"{'用户' if n['role']=='user' else 'AI'}: {n['content'][:400]}"
                                        for n in ancestors
                                    ])
                                    parts.insert(0, f"--- 引用链上下文({len(ancestors)}层) ---\n{ancestor_text}\n---")
                            else:
                                role_label = '用户' if quoted_role == 'user' else 'AI'
                                quoted_full = quoted_full_content[:4000]
                                parts.append(f"[QUOTED] ({role_label}):\n{quoted_full}")
                            quote_chain_context = '\n\n'.join(parts)
                            logger.info(f"Quote chain built with {len(chain_nodes)} nodes for project {proj_id}")
        except Exception as e:
            logger.warning(f"Quote chain context build failed: {e}")

    # Headroom compression on long streaming queries
    try:
        from app.services.runtime_config import get as rc_get
        if rc_get("headroom_enabled", True) and len(query) > 800:
            compressed = headroom_compress(query)
            if compressed and len(compressed) < len(query):
                logger.info(f"Headroom stream: {len(query)} -> {len(compressed)} chars")
                query = compressed
    except Exception:
        pass

    # Inject quote chain context as high-priority reference prefix
    if quote_chain_context:
        query = f"{quote_chain_context}\n=== 当前用户追问 ===\n{query}"

    # Get LLM via multi-provider router
    try:
        from app.services.llm_provider import create_chat_model, get_any_api_key
    except ImportError:
        return jsonify({"error": "LLM provider module not available"}), 500
    if not get_any_api_key():
        return jsonify({"error": "AI service not configured"}), 500

    def generate():
        """SSE generator for streaming response."""
        try:
            # Resolve provider/model: runtime_config (admin) > session (legacy) > auto-detect
            provider_id = session.get('llm_provider') or None
            model_id = session.get('llm_model') or None
            try:
                from app.services.runtime_config import get as rc_get
                rc_provider = rc_get('active_llm_provider', '')
                rc_model = rc_get('active_llm_model', '')
                if rc_provider and rc_provider != 'auto':
                    provider_id = rc_provider
                if rc_model and rc_model != 'auto':
                    model_id = rc_model
            except Exception:
                pass
            if is_grilling:
                agent = get_redteam_agent(max_tokens=session.get('max_tokens', 1600))
                config = {"configurable": {"thread_id": thread_id}}
            else:
                llm = create_chat_model(
                    provider_id=provider_id,
                    model=model_id,
                    streaming=True,
                    temperature=0.7,
                    max_tokens=session.get('max_tokens', 1600),
                    timeout=int(os.getenv("LLM_TIMEOUT", "120")),
                )
                agent = create_agent(
                    model=llm,
                    tools=[get_date, bocha_search],
                    system_prompt=g.AGENT_SYSTEM_PROMPT,
                    checkpointer=MemorySaver(),
                )
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            full_response = ""

            # Use stream_mode="custom" to get raw LLM token chunks inside agent events
            for event in agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                config,
                stream_mode="messages"
            ):
                # stream_mode="messages" yields tuples: (message, metadata)
                # Each message is an AIMessageChunk with .content holding incremental tokens
                if isinstance(event, tuple) and len(event) >= 1:
                    msg = event[0]
                else:
                    msg = event

                chunk = None
                if hasattr(msg, 'content') and msg.content:
                    chunk = msg.content
                elif isinstance(msg, dict) and 'content' in msg:
                    chunk = msg['content']

                if chunk:
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'content', 'text': chunk})}\n\n"

            # Store assistant response — DB for registered, memory for anonymous
            if full_response:
                thinking, answer = split_thinking_answer(full_response)
                if is_anon:
                    session.setdefault('chat_history', []).append({'role': 'assistant', 'content': answer, 'thinking': thinking})
                else:
                    store_message(thread_id, 'assistant', answer, thinking)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            # Save partial response before sending error to client
            if full_response:
                thinking, answer = split_thinking_answer(full_response)
                answer = (answer or full_response) + "\n\n[回复中断] 发送任意消息继续"
                if is_anon:
                    session.setdefault('chat_history', []).append({'role': 'assistant', 'content': answer, 'thinking': thinking})
                else:
                    store_message(thread_id, 'assistant', answer, thinking)
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)[:200], 'partial_saved': bool(full_response)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )

@chat_bp.route('/send', methods=['POST'])
def send_message():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403

    user_msg = request.form.get('message', '').strip()
    if len(user_msg) > 10000:
        return jsonify({"error": "消息不能超过10000字"}), 400

    # ── /review command: AI self-reviews last response (Pre-PR gate pattern) ──
    if user_msg.startswith('/review'):
        from app.services.judge_review import review_response
        history = session.get('chat_history', [])
        last_user = ''
        last_assistant = ''
        for msg in reversed(history):
            if msg['role'] == 'assistant' and not last_assistant:
                last_assistant = msg.get('content', '')
            if msg['role'] == 'user' and not last_user:
                last_user = msg.get('content', '')
        if not last_assistant:
            return jsonify({"error": "No previous response to review"}), 400
        review = review_response(last_user or 'previous query', last_assistant)
        store_message(thread_id, 'user', user_msg)
        if review:
            result_text = f"""# Self-Review
**Score**: {review['score']}/10
**Verdict**: {review['verdict']}
**Issues**: {review['issues'][:500]}
**Corrected** (if applicable): {review.get('corrected') or 'N/A'}"""
        else:
            result_text = "Self-review unavailable (judge_review_enabled=false or only one LLM provider configured)."
        store_message(thread_id, 'assistant', result_text)
        return jsonify({"status": "ok", "review": review, "message": result_text})

    # ── /plan command: create structured plan via AI, save to notebook ──
    if user_msg.startswith('/plan '):
        plan_name = user_msg[6:].strip()
        if not plan_name:
            return jsonify({"error": "Usage: /plan <project name>"}), 400
        try:
            from app.services.llm_provider import create_chat_model
            from langchain_core.messages import SystemMessage, HumanMessage
            from app.services.notebook import save_note

            plan_prompt = f"""Create a structured project plan for: {plan_name}
Format as markdown with these sections:
# {plan_name}
## Objective (1-2 sentences)
## Key Milestones (3-5 items with bullet points)
## Risks (2-4 items)
## Timeline (estimated phases)
## Success Criteria (2-3 measurable outcomes)
Keep it concise, professional, in Chinese."""
            llm = create_chat_model(streaming=False, temperature=0.4, max_tokens=800, timeout=30)
            resp = llm.invoke([SystemMessage(content="You are a professional project planner. Output clean markdown."),
                              HumanMessage(content=plan_prompt)])
            plan_text = resp.content if hasattr(resp, 'content') else str(resp)
            note_id = f"plan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}_{plan_name[:30].replace(' ','_')}"
            save_note(session.get('user_id'), note_id, plan_text)
            store_message(thread_id, 'user', user_msg)
            store_message(thread_id, 'assistant', plan_text)
            return jsonify({
                "status": "ok", "plan": plan_text, "note_id": note_id,
                "message": f"Plan saved to notebook: {note_id}"
            })
        except Exception as e:
            logger.error(f"/plan failed: {e}")
            return jsonify({"error": f"Plan generation failed: {str(e)[:100]}"}), 500

    message_id = request.form.get('message_id')
    if not user_msg and 'files' not in request.files:
        return jsonify({"error": "Empty message and no files"}), 400
    if not message_id:
        return jsonify({"error": "Missing message_id"}), 400

    thread_id = session.get('thread_id', str(uuid.uuid4()))
    session['thread_id'] = thread_id
    user_id = get_user_id()
    get_or_create_session(thread_id)

    is_grilling = False
    try:
        from app.database import get_db_connection
        from psycopg2.extras import RealDictCursor
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT is_grilling FROM chat_sessions WHERE thread_id = %s", (thread_id,))
                row = cur.fetchone()
                if row and row.get('is_grilling'):
                    is_grilling = True
    except Exception as e:
        logger.warning(f"Failed to check is_grilling: {e}")


    # Idempotency check
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute("""
                    INSERT INTO message_responses (message_id, thread_id, user_message, assistant_response, thinking)
                    VALUES (%s, %s, %s, '', '')
                    ON CONFLICT (message_id) DO NOTHING
                    RETURNING assistant_response, thinking
                """, (message_id, thread_id, user_msg))
                row = cur.fetchone()
                if row and row['assistant_response'] == '':
                    conn.commit()
                elif row:
                    return jsonify({
                        "user_message": user_msg,
                        "assistant_message": row['assistant_response'],
                        "thinking": row['thinking'],
                        "cached": True
                    })
            except Exception:
                conn.rollback()
                cur.execute("SELECT assistant_response, thinking FROM message_responses WHERE message_id = %s", (message_id,))
                row = cur.fetchone()
                if row:
                    return jsonify({
                        "user_message": user_msg,
                        "assistant_message": row['assistant_response'],
                        "thinking": row['thinking'],
                        "cached": True
                    })

    uploaded_files = request.files.getlist('files')
    has_files = len(uploaded_files) > 0 and uploaded_files[0].filename
    file_contents = []
    is_image = False
    image_analysis_used = True

    if has_files:
        for f in uploaded_files:
            if not allowed_file(f.filename):
                return jsonify({"error": f"不支持的文件类型: {f.filename}"}), 400
        success, busy_thread, busy_name = acquire_task_lock(user_id, thread_id, 'ocr_upload')
        if not success:
            return jsonify({
                "error": "resource_busy",
                "busy_chat": busy_name,
                "message": f"另一个资源密集型任务正在聊天“{busy_name}”中进行，请稍后再试。"
            }), 409
    else:
        success = True

    try:
        uploaded_filenames = []
        if has_files:
            for uploaded in uploaded_files:
                if not uploaded.filename:
                    continue
                uploaded_filenames.append(uploaded.filename)
                ext = os.path.splitext(uploaded.filename)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                    is_image = True

                if session.get('consent_value', 0) == 1:
                    ensure_user_exists(user_id)
                    file_bytes = uploaded.read()
                    file_hash = hashlib.sha256(file_bytes).hexdigest()
                    uploaded.seek(0)
                    with get_db_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT id FROM user_files WHERE user_id = %s AND file_hash = %s", (user_id, file_hash))
                            existing = cur.fetchone()
                            if not existing:
                                unique_name = f"{file_hash}_{int(time.time())}{ext}"
                                original_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)
                                os.makedirs(original_dir, exist_ok=True)
                                original_path = os.path.join(original_dir, unique_name)
                                uploaded.save(original_path)
                                uploaded.seek(0)  # save() consumes stream, reset for later reads
                                cur.execute("""
                                    INSERT INTO user_files (user_id, thread_id, filename, size_bytes, expires_at,
                                                            original_stored_path, file_hash, original_expires_at, original_name, content)
                                    VALUES (%s, %s, %s, %s, NULL, %s, %s, NOW() + INTERVAL '3 days', %s, %s)
                                    ON CONFLICT (thread_id, filename) DO UPDATE SET
                                        size_bytes = EXCLUDED.size_bytes,
                                        original_stored_path = EXCLUDED.original_stored_path,
                                        file_hash = EXCLUDED.file_hash,
                                        original_expires_at = EXCLUDED.original_expires_at,
                                        original_name = EXCLUDED.original_name,
                                        content = EXCLUDED.content
                                """, (user_id, thread_id, uploaded.filename, len(file_bytes), original_path, file_hash, uploaded.filename, ""))
                                conn.commit()
                    # Use robust cache
                    file_content = get_or_extract_file_analysis(uploaded, 'chat', user_id, thread_id=thread_id)

                if file_content and not file_content.startswith("["):
                    # Also validate again (the robust cache already did, but double‑check)
                    if is_valid_extracted_text(file_content):
                        add_to_cache(thread_id, uploaded.filename, file_content, user_id)
                        record_file_usage(thread_id, uploaded.filename, 'chat', user_msg)
                        file_contents.append(file_content)
                    else:
                        logger.warning(f"Extracted text from {uploaded.filename} is invalid, skipping.")
                        file_contents.append(f"[文件 {uploaded.filename} 的内容无法读取，请检查文件格式。]")
                else:
                    # No text extracted – inform user
                    file_contents.append(f"[文件 {uploaded.filename} 的内容无法读取，请检查文件格式。]")

        # Build uploaded file text
        file_text = ""
        if has_files:
            if file_contents:
                combined = "\n\n".join(file_contents)
                if is_image:
                    file_text = f"The user uploaded an image. Extracted description:\n{combined}\n\nUser query: {user_msg}"
                else:
                    file_text = f"File content(s):\n{combined}\n\nUser query:\n{user_msg}"
            else:
                file_text = f"The user uploaded a file but no readable text could be extracted. The user's question is: {user_msg}"
        else:
            # Include recent batch compare files in context if available
            batch_files = session.get('batch_compare_files')
            if batch_files:
                batch_context = "The user previously ran a batch comparison on the following files:\n"
                for bf in batch_files:
                    batch_context += f"\n--- File: {bf['filename']} ---\n{bf['text']}\n"
                batch_context += f"\n\nUser query:\n{user_msg}"
                file_text = batch_context
            else:
                file_text = user_msg

        # Process knowledge files
        knowledge_files_json = request.form.get('knowledge_files')
        knowledge_files = []
        if knowledge_files_json:
            try:
                knowledge_files = json.loads(knowledge_files_json)
            except Exception:
                pass
        # Track KB file usage
        for kf in knowledge_files:
            try:
                record_file_usage(thread_id, kf.get('filename', 'knowledge_file'), kf.get('source', 'knowledge'), user_msg)
            except Exception:
                pass

        knowledge_content = []
        # Track seen names across all sources to deduplicate for LLM
        _seen_names: dict = {}  # name -> count
        def _dedup_name(raw_name: str, entity_id) -> str:
            """Append #short_hash suffix if name already seen."""
            name = raw_name or 'unknown'
            _seen_names[name] = _seen_names.get(name, 0) + 1
            if _seen_names[name] > 1:
                h = hashlib.md5(str(entity_id).encode()).hexdigest()[:4]
                return f"{name} #{h}"
            return name

        for kf in knowledge_files:
            source = kf.get('source')
            fid = kf.get('id')
            filename = kf.get('filename')
            if source == 'user_file':
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT content FROM user_files WHERE id = %s AND user_id = %s", (fid, user_id))
                        row = cur.fetchone()
                        dname = _dedup_name(filename, fid)
                        if row and row[0]:
                            # Validate content
                            if is_valid_extracted_text(row[0]):
                                knowledge_content.append(f"--- 文件: {dname} ---\n{row[0]}")
                            else:
                                knowledge_content.append(f"--- 文件: {dname} (内容无效) ---\n无法读取文件内容")
            elif source == 'project_file':
                # similar validation
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT pf.content, pf.original_name
                            FROM project_files pf
                            JOIN project_members pm ON pf.project_id = pm.project_id
                            WHERE pf.id = %s AND pm.user_id = %s
                        """, (fid, user_id))
                        row = cur.fetchone()
                        if row and row[0]:
                            dname = _dedup_name(row[1], fid)
                            if is_valid_extracted_text(row[0]):
                                knowledge_content.append(f"--- 文件: {dname} ---\n{row[0]}")
                            else:
                                knowledge_content.append(f"--- 文件: {dname} (内容无效) ---\n无法读取文件内容")
            elif source == 'knowledge_lab':
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT content, original_name, skill_summary FROM knowledge_lab_files WHERE id = %s AND user_id = %s",
                            (fid, user_id))
                        row = cur.fetchone()
                        if row and row[0]:
                            text = row[0]
                            dname = _dedup_name(row[1], fid)
                            # Append skill summary if available (paired KB-skill)
                            skill_col = row[2] if len(row) > 2 else None
                            if skill_col and is_valid_extracted_text(skill_col):
                                text = f"[技能摘要]\n{skill_col}\n\n[原始文档]\n{text}"
                            if is_valid_extracted_text(text):
                                knowledge_content.append(f"--- 知识库实验室文件: {dname} ---\n{text}")
                            else:
                                knowledge_content.append(f"--- 知识库实验室文件: {dname} (内容无效) ---\n无法读取文件内容")
            elif source == 'company_kb':
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT content, original_name, skill_summary FROM company_knowledge_base WHERE id = %s", (fid,))
                        row = cur.fetchone()
                        if row and row[0]:
                            text = row[0]
                            dname = _dedup_name(row[1], fid)
                            skill_col = row[2] if len(row) > 2 else None
                            if skill_col and is_valid_extracted_text(skill_col):
                                text = f"[技能摘要]\n{skill_col}\n\n[原始文档]\n{text}"
                            if is_valid_extracted_text(text):
                                knowledge_content.append(f"--- 公司知识库文件: {dname} ---\n{text}")
                            else:
                                knowledge_content.append(f"--- 公司知识库文件: {dname} (内容无效) ---\n无法读取文件内容")

        # Build final query — hybrid: RAG for KB files, full content for direct uploads
        rag_used = False
        if knowledge_content:
            # Attempt RAG retrieval for knowledge_lab + company_kb files
            rag_sources = [f['source'] for f in knowledge_files if f['source'] in ('knowledge_lab', 'company_kb')]
            rag_ids = [f['id'] for f in knowledge_files if f['source'] in ('knowledge_lab', 'company_kb')]
            rag_context = ''
            if rag_sources:
                try:
                    from app.services.rag_engine import build_rag_context
                    rag_context = build_rag_context(
                        user_msg, list(set(rag_sources)),
                        top_k=12, max_chars=10000, file_ids=rag_ids
                    )
                except Exception as e:
                    logger.warning(f"RAG retrieval failed, falling back to dump-all: {e}")

            rag_used = bool(rag_context)
            if rag_context:
                # RAG mode: use semantic chunks
                knowledge_text = rag_context + "\n\n" + "\n\n".join(knowledge_content)
                max_knowledge = 20000
            else:
                # Fallback: traditional dump-all
                knowledge_text = "\n\n".join(knowledge_content)
                max_knowledge = 12000

            if len(knowledge_text) > max_knowledge:
                knowledge_text = knowledge_text[:max_knowledge] + "\n\n[知识库内容已截断，请提出更具体的问题以获得更精准的结果]"
            from app.services.prompt_safety import wrap_user_content, build_rag_priority_rules, sanitize_for_prompt
            safe_user_msg = sanitize_for_prompt(user_msg, 'chat_user_msg')
            final_query = f"""你是一个基于知识库的助手。以下知识库内容具有最高优先级。请严格依据这些内容回答问题。如果知识库中没有相关信息，请明确告知用户。
{build_rag_priority_rules()}

{wrap_user_content(knowledge_text if knowledge_text else '（知识库中暂无相关内容——请如实告知用户，不得自行编造）', 'KNOWLEDGE_BASE')}

{wrap_user_content(file_text, 'UPLOADED_FILES')}

{wrap_user_content(safe_user_msg, 'USER_QUESTION')}"""
        else:
            from app.services.prompt_safety import sanitize_for_prompt
            final_query = sanitize_for_prompt(file_text, 'file_text_query')

        # ── Auto-resume from interrupted stream ──
        interrupted_context = ""
        try:
            msgs = get_session_messages(thread_id)
            if msgs:
                last = msgs[-1]
                if last.get('role') == 'assistant' and '[回复中断]' in (last.get('content') or ''):
                    interrupted_context = last['content'].replace('[回复中断] 发送任意消息继续', '').strip()
        except Exception:
            pass
        if interrupted_context and len(user_msg) < 20:
            final_query = (
                f"[之前的回复因故中断，请从断点处直接继续，不要重复已输出的内容。]\n\n"
                f"=== 已输出的部分 ===\n{interrupted_context}\n\n"
                f"=== 用户问题(继续) ===\n{user_msg}"
            )

        # Log what the AI actually receives (first 500 chars) to debug stale cache issues
        logger.info(f"/send query (len={len(final_query)}): {final_query[:500]}...")

        # Headroom compression: controlled by runtime_config headroom_enabled
        compressed_headroom = False
        try:
            from app.services.runtime_config import get as rc_get
            if rc_get("headroom_enabled", True):
                headroom_threshold = 3000 if rag_used or has_files else 800
                if len(final_query) > headroom_threshold:
                    compressed = headroom_compress(final_query)
                    if compressed and len(compressed) < len(final_query):
                        pct = 100 - len(compressed) * 100 // max(len(final_query), 1)
                        logger.info(f"Headroom /send: {len(final_query)} -> {len(compressed)} chars ({pct}%)")
                        final_query = compressed
                        compressed_headroom = True
        except Exception:
            pass

        # Store user message and get its ID
        user_msg_id = store_message(thread_id, 'user', user_msg)

        # ========== AGENT INVOCATION WITH ISOLATION ==========
        use_isolated_thread = (knowledge_files and len(knowledge_files) > 0) or has_files
        if use_isolated_thread:
            temp_thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": temp_thread_id}}
            from langgraph.checkpoint.memory import MemorySaver
            from app.services.llm_provider import create_chat_model
            # Resolve provider/model: runtime_config > session > auto-detect
            iso_provider = session.get('llm_provider') or None
            iso_model = session.get('llm_model') or None
            try:
                from app.services.runtime_config import get as rc_get
                rc_provider = rc_get('active_llm_provider', '')
                rc_model = rc_get('active_llm_model', '')
                if rc_provider and rc_provider != 'auto':
                    iso_provider = rc_provider
                if rc_model and rc_model != 'auto':
                    iso_model = rc_model
            except Exception:
                pass
            llm = create_chat_model(
                provider_id=iso_provider,
                model=iso_model,
                streaming=False,
                temperature=0.7,
                max_tokens=session.get('max_tokens', 1600),
                timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            )
            system_prompt = REDTEAM_SYSTEM_PROMPT if is_grilling else g.AGENT_SYSTEM_PROMPT
            checkpointer = MemorySaver()
            isolated_agent = create_agent(
                model=llm,
                tools=[get_date, bocha_search],
                system_prompt=system_prompt,
                checkpointer=checkpointer
            )
            try:
                response = isolated_agent.invoke(
                    {"messages": [{"role": "user", "content": final_query}]},
                    config
                )
            except Exception as e:
                logger.error(f"Isolated agent invoke failed: {e}", exc_info=True)
                return jsonify({"error": "AI 服务暂时不可用"}), 500
            finally:
                del isolated_agent
                del checkpointer
                del llm
        else:
            agent = get_redteam_agent() if is_grilling else get_agent()
            config = {"configurable": {"thread_id": thread_id}}
            try:
                response = agent.invoke({"messages": [{"role": "user", "content": final_query}]}, config)
            except Exception as e:
                logger.error(f"Agent invoke failed: {e}", exc_info=True)
                return jsonify({"error": "AI 服务暂时不可用"}), 500

        # Process response
        assistant_message = response["messages"][-1]
        raw_response = assistant_message.content
        reasoning = assistant_message.additional_kwargs.get('reasoning_content', '')
        if reasoning and reasoning.strip():
            thinking = reasoning.strip()
            answer = raw_response.strip() if raw_response else ''
        else:
            thinking, answer = split_thinking_answer(raw_response)

        assistant_msg_id = store_message(thread_id, 'assistant', answer, thinking)

        # ── Judge Model Review (optional, controlled by runtime_config) ──
        try:
            from app.services.judge_review import review_response
            review = review_response(user_msg, answer)
            if review and review.get('verdict') == 'NEEDS_IMPROVEMENT' and review.get('corrected'):
                logger.info(f"Judge review: replacing response with corrected version (score={review['score']})")
                answer = review['corrected']
                # Update stored message with corrected response
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE message_responses SET assistant_response = %s WHERE message_id = %s",
                                    (answer, assistant_msg_id))
                        conn.commit()
            elif review:
                logger.info(f"Judge review passed: score={review['score']}, verdict={review['verdict']}")
        except Exception:
            pass

        # Log to training data pipeline
        try:
            from app.services.training_logger import log_interaction
            from app.services.agent import get_last_search_cache_hit
            cache_hit = get_last_search_cache_hit()
            log_interaction(thread_id, user_msg, answer, thinking=thinking,
                           knowledge_files=knowledge_files,
                           rag_context=rag_context if rag_used else None,
                           uploaded_files=uploaded_filenames if has_files else None,
                           headroom_used=bool(compressed_headroom),
                           model=session.get('llm_model', '') or session.get('llm_provider', ''),
                           search_cache_hit=cache_hit)
        except Exception:
            pass

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE message_responses SET assistant_response = %s, thinking = %s WHERE message_id = %s",
                            (answer, thinking, message_id))
                conn.commit()

        new_history = session.get('chat_history', [])
        new_history.append({"role": "user", "content": user_msg})
        new_history.append({"role": "assistant", "content": answer, "thinking": thinking})
        session['chat_history'] = new_history
        session['last_user_msg'] = user_msg
        session['last_assistant_msg'] = answer

        # Clear batch compare context after successful chat (one-time injection)
        session.pop('batch_compare_files', None)

        return jsonify({
            "assistant_message": answer,
            "thinking": thinking,
            "file_processed": len(uploaded_filenames) > 0,
            "ocr_attempted": is_image,
            "is_batch_report": False,
            "image_analysis_used": image_analysis_used,
            "assistant_message_id": assistant_msg_id,
            "user_message_id": user_msg_id
        })
    except Exception as e:
        logger.error(f"/send failed: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {e}"}), 500
    finally:
        if has_files:
            release_task_lock(user_id)

@chat_bp.route('/set_max_tokens', methods=['POST'])
def set_max_tokens():
    data = request.get_json()
    tokens = data.get('max_tokens', 4800)
    tokens = max(100, min(4800, tokens))
    session['max_tokens'] = tokens
    with g._agent_lock:
        g._agent = None
    return jsonify({"success": True, "max_tokens": tokens})

# ── LLM Provider / Model selection ──

@chat_bp.route('/llm_providers', methods=['GET'])
def list_llm_providers():
    """Return available LLM providers and the currently active one."""
    try:
        from app.services.llm_provider import get_available_providers, get_active_provider
    except ImportError:
        return jsonify({"available": [], "active": None, "error": "llm_provider module not loaded"})
    active = get_active_provider()
    return jsonify({
        "available": get_available_providers(),
        "active": active,
    })

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
    with g._agent_lock:
        g._agent = None
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

@chat_bp.route('/new_chat', methods=['POST'])
def new_chat():
    new_thread_id = str(uuid.uuid4())
    session['thread_id'] = new_thread_id
    session['chat_history'] = []
    get_or_create_session(new_thread_id)
    return jsonify({"thread_id": new_thread_id})

@chat_bp.route('/api/login', methods=['POST'])
def api_login():
    """JWT login for external API access (WeChat Enterprise, CLI tools, etc.).

    POST body: {"username": "...", "pin": "1234"}
    Returns: {"access_token": "eyJ...", "user_id": "...", "username": "...", "role": "..."}
    """
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get('username') or '').strip()
    pin = (data.get('pin') or '').strip()
    if not username or not pin:
        return jsonify({"error": "username and pin required"}), 400
    try:
        from app.services.auth_jwt import create_token
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id, username, role, pin_hash FROM users WHERE username = %s AND is_active = TRUE",
                    (username,))
                row = cur.fetchone()
                if not row:
                    return jsonify({"error": "Invalid credentials"}), 401
                user_id, uname, role, pin_hash = row
                import hashlib
                if hashlib.sha256(pin.encode()).hexdigest() != pin_hash:
                    return jsonify({"error": "Invalid credentials"}), 401
                token = create_token(user_id, uname, role)
                return jsonify({
                    "access_token": token,
                    "user_id": user_id,
                    "username": uname,
                    "role": role,
                    "expires_in_hours": 24,
                })
    except Exception as e:
        logger.error(f"API login failed: {e}")
        return jsonify({"error": "Login failed"}), 500

@chat_bp.route('/get_sessions', methods=['GET'])
def get_sessions():
    sessions = get_user_sessions()
    return jsonify({"sessions": sessions})

@chat_bp.route('/load_session/<thread_id>', methods=['GET'])
def load_session(thread_id):
    if session.get('consent_value', 0) != 1:
        messages = get_session_messages_anon(thread_id)
        session['thread_id'] = thread_id
        session['chat_history'] = messages
        return jsonify({"messages": messages, "thread_id": thread_id})

    user_sessions = get_user_sessions()
    if not any(s['thread_id'] == thread_id for s in user_sessions):
        return jsonify({"error": "Session not found"}), 404

    with get_db_connection() as conn:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                db_execute_readonly(cur)
                cur.execute(
                    "SELECT id, role, content, thinking, timestamp FROM chat_messages WHERE thread_id = %s ORDER BY id ASC",
                    (thread_id,)
                )
                rows = cur.fetchall()
                messages = []
                for row in rows:
                    ts_utc = row['timestamp']
                    ts_beijing = ts_utc.astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S') if ts_utc else None
                    messages.append({
                        "id": row['id'],
                        "role": row['role'],
                        "content": row['content'],
                        "thinking": row['thinking'],
                        "timestamp": ts_beijing
                    })
            session['thread_id'] = thread_id
            session['chat_history'] = messages
            user_id = get_user_id()
            load_cache_from_db(thread_id, user_id)
            # Include sessions so frontend can determine project chat context
            return jsonify({"messages": messages, "thread_id": thread_id, "sessions": user_sessions})
        except Exception as e:
            logger.error(f"load_session failed for {thread_id}: {e}", exc_info=True)
            return jsonify({"error": f"Failed to load session: {e}"}), 500

@chat_bp.route('/delete_session/<thread_id>', methods=['POST'])
def delete_session_route(thread_id):
    user_sessions = get_user_sessions()
    if not any(s['thread_id'] == thread_id for s in user_sessions):
        return jsonify({"error": "Session not found"}), 404
    # Block deletion of project chats — they're archived with the project
    target = next((s for s in user_sessions if s['thread_id'] == thread_id), None)
    if target and target.get('project_id'):
        return jsonify({"error": "项目对话不能单独删除，请通过归档/删除项目来管理"}), 403
    user_id = get_user_id()
    with g.user_task_lock:
        cleanup_stale_tasks()
        if user_id in g.user_active_tasks and g.user_active_tasks[user_id]['thread_id'] == thread_id:
            return jsonify({
                "error": "task_running",
                "message": "无法删除：该聊天正在进行资源密集型任务，请等待任务完成后再试。"
            }), 409
    try:
        archive_session(thread_id, user_id, reason="manual")
    except Exception as e:
        logger.error(f"Archive session failed for {thread_id}: {e}", exc_info=True)
    try:
        delete_session(thread_id)
        logger.info(f"Session {thread_id} deleted successfully for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to delete session {thread_id}: {e}", exc_info=True)
        return jsonify({"error": "删除失败，请稍后重试"}), 500
    new_thread_id = None
    if session.get('thread_id') == thread_id:
        new_thread_id = str(uuid.uuid4())
        session['thread_id'] = new_thread_id
        session['chat_history'] = []
        get_or_create_session(new_thread_id)
        load_cache_from_db(new_thread_id, get_user_id())
    return jsonify({
        "status": "ok",
        "new_thread_id": new_thread_id,
        "messages": []
    })

@chat_bp.route('/update_session_title', methods=['POST'])
def update_session_title():
    """Allow users to rename their chat sessions."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json(silent=True) or {}
    thread_id = data.get('thread_id', '').strip()
    title = data.get('title', '').strip()
    if not thread_id or not title:
        return jsonify({"error": "Missing thread_id or title"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET title = %s WHERE thread_id = %s AND user_id = %s",
                (title, thread_id, user_id)
            )
            if cur.rowcount == 0:
                return jsonify({"error": "Session not found or access denied"}), 404
    return jsonify({"status": "ok", "title": title})

@chat_bp.route('/archive_session/<thread_id>', methods=['POST'])
def archive_session_route(thread_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    try:
        archive_path = archive_session(thread_id, user_id, reason="manual")
        if archive_path:
            delete_session(thread_id)   # remove from active sessions
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Archive failed"}), 500
    except Exception as e:
        logger.error(f"Archive session error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@chat_bp.route('/restore_session/<thread_id>', methods=['POST'])
def restore_session_route(thread_id):
    """Restore an archived session back to active chat."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT archive_path FROM archived_sessions WHERE thread_id = %s AND user_id = %s", (thread_id, user_id))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Archived session not found"}), 404
            archive_file = os.path.join(row['archive_path'], f"{thread_id}_session.json")
            if not os.path.exists(archive_file):
                return jsonify({"error": "Archive file missing"}), 404
            try:
                with open(archive_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                return jsonify({"error": "Archive data corrupted"}), 500
            title = data.get('session', {}).get('title', 'Restored Chat')
            cur.execute("INSERT INTO chat_sessions (thread_id, user_id, title, created_at, updated_at) VALUES (%s, %s, %s, %s, NOW()) ON CONFLICT (thread_id) DO UPDATE SET updated_at = NOW()", (thread_id, user_id, title, data.get('session', {}).get('created_at')))
            for msg in data.get('messages', []):
                content = msg.get('content', '')
                thinking = msg.get('thinking', '')
                role = msg.get('role', 'user')
                store_message(thread_id, role, content, thinking if role == 'assistant' else None)
            cur.execute("DELETE FROM archived_sessions WHERE thread_id = %s", (thread_id,))
            conn.commit()
    return jsonify({"success": True, "title": title})


@chat_bp.route('/list_archived_sessions', methods=['GET'])
def list_archived_sessions():
    """Admin only: list all archived sessions across all users."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    if session.get('role') != 'admin':
        return jsonify({"error": "Admin only"}), 403
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT a.thread_id, a.archived_at, a.user_id, u.username
                FROM archived_sessions a LEFT JOIN users u ON a.user_id = u.user_id
                ORDER BY a.archived_at DESC LIMIT 30""")
            sessions = cur.fetchall()
    return jsonify({"sessions": [{
        'thread_id': s['thread_id'],
        'archived_at': s['archived_at'].isoformat() if s['archived_at'] else None,
        'username': s.get('username', '?'),
    } for s in sessions]})

@chat_bp.route('/regenerate', methods=['POST'])
def regenerate():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    data = request.get_json()
    user_message = data.get('user_message')
    if not user_message:
        return jsonify({"error": "Missing user_message"}), 400
    thread_id = session['thread_id']
    get_or_create_session(thread_id)

    is_grilling = False
    try:
        from app.database import get_db_connection
        from psycopg2.extras import RealDictCursor
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT is_grilling FROM chat_sessions WHERE thread_id = %s", (thread_id,))
                row = cur.fetchone()
                if row and row.get('is_grilling'):
                    is_grilling = True
    except Exception as e:
        logger.warning(f"Failed to check is_grilling: {e}")

    agent = get_redteam_agent() if is_grilling else get_agent()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        response = agent.invoke({"messages": [{"role": "user", "content": user_message}]}, config)
    except Exception as e:
        logger.error(f"Regenerate invoke failed: {e}", exc_info=True)
        return jsonify({"error": "AI 服务暂时不可用"}), 500
    assistant_message = response["messages"][-1]
    raw_response = assistant_message.content
    reasoning = assistant_message.additional_kwargs.get('reasoning_content', '')
    if reasoning and reasoning.strip():
        thinking = reasoning.strip()
        answer = raw_response.strip() if raw_response else ''
    else:
        thinking, answer = split_thinking_answer(raw_response)
    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                cur.execute("""
                            DELETE
                            FROM chat_messages
                            WHERE id IN (SELECT id
                                         FROM chat_messages
                                         WHERE thread_id = %s
                                         ORDER BY timestamp DESC
                                         LIMIT 2)
                            """, (thread_id,))
                conn.commit()
    store_message(thread_id, 'user', user_message)
    store_message(thread_id, 'assistant', answer if answer else raw_response, thinking if thinking else "")
    new_messages = get_session_messages(thread_id)
    session['chat_history'] = new_messages
    return jsonify({
        "assistant_message": answer if answer else raw_response,
        "thinking": thinking if thinking else ""
    })

@chat_bp.route('/check_storage', methods=['GET'])
def check_storage():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = get_user_id()
    total_bytes = get_user_total_storage_size(user_id)
    total_mb = total_bytes / (1024 * 1024)
    warning = total_mb > 300
    return jsonify({
        "total_mb": round(total_mb, 2),
        "warning": warning,
        "message": f"已使用 {total_mb:.2f} MB / 300 MB" if warning else None
    })

@chat_bp.route('/cleanup_now', methods=['POST'])
def cleanup_now():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    cleanup_old_sessions(days=15)
    return jsonify({"status": "ok", "message": "Cleanup completed"})

@chat_bp.route('/cleanup_anon_temp', methods=['POST'])
def cleanup_anon_temp():
    if session.get('consent_value', 0) != 1:
        user_id = get_user_id()
        temp_dir = get_anon_temp_dir(user_id)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up anonymous temp directory for user {user_id}")
    return jsonify({"status": "ok"})

# Account routes
@chat_bp.route('/set_image_analysis', methods=['POST'])
def set_image_analysis():
    data = request.get_json()
    enabled = data.get('enabled', True)
    session['analyze_images'] = enabled
    return jsonify({"success": True})

@chat_bp.route('/search_chat', methods=['GET'])
def search_chat():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify({"error": "Search query must be at least 2 characters"}), 400

    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    fuzzy = request.args.get('fuzzy', 'false').lower() == 'true'
    role = request.args.get('role', 'assistant')

    if fuzzy:
        search_pattern = f"%{q}%"
    else:
        search_pattern = q

    date_condition = ""
    params = [user_id, search_pattern]
    if start_date:
        date_condition += " AND cm.timestamp >= %s"
        params.append(start_date)
    if end_date:
        date_condition += " AND cm.timestamp <= %s"
        params.append(end_date)

    if role == 'assistant':
        role_condition = " AND cm.role = 'assistant'"
    elif role == 'user':
        role_condition = " AND cm.role = 'user'"
    else:
        role_condition = ""

    is_admin = session.get('role') == 'admin'
    query = f"""
        SELECT cs.thread_id, cs.title, cm.role, cm.content, cm.timestamp, cm.id as message_id,
               SUBSTRING(cm.content, 1, 200) as snippet
        FROM chat_messages cm
        JOIN chat_sessions cs ON cm.thread_id = cs.thread_id
        WHERE (cs.user_id = %s{f' OR TRUE' if is_admin else ''})
          AND cm.content ILIKE %s
          {role_condition}
          {date_condition}
        ORDER BY cm.timestamp DESC
        LIMIT 100
    """

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            results = cur.fetchall()
            formatted_results = []
            for row in results:
                ts_utc = row['timestamp']
                ts_beijing = ts_utc.astimezone(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S') if ts_utc else None
                snippet = row['snippet'] or ""
                if fuzzy:
                    escaped = re.escape(q)
                    highlighted = re.sub(f"({escaped})", r'<mark>\1</mark>', snippet, flags=re.IGNORECASE)
                else:
                    highlighted = snippet
                formatted_results.append({
                    'thread_id': row['thread_id'],
                    'title': row['title'],
                    'role': row['role'],
                    'snippet': snippet,
                    'timestamp_str': ts_beijing,
                    'highlighted_snippet': highlighted,
                    'message_id': row['message_id']
                })
            return jsonify({"results": formatted_results})

# ---------- Admin database browser ----------

@chat_bp.route('/upload_file', methods=['POST'])
def upload_file():
    """Upload a file — registered users get persistent storage, anonymous get temp storage."""
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"不支持的文件类型: {file.filename}"}), 400

    user_id = get_user_id()
    thread_id = session.get('thread_id')
    if not thread_id:
        thread_id = str(uuid.uuid4())
        session['thread_id'] = thread_id
        get_or_create_session(thread_id)

    is_grilling = False
    try:
        from app.database import get_db_connection
        from psycopg2.extras import RealDictCursor
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT is_grilling FROM chat_sessions WHERE thread_id = %s", (thread_id,))
                row = cur.fetchone()
                if row and row.get('is_grilling'):
                    is_grilling = True
    except Exception as e:
        logger.warning(f"Failed to check is_grilling: {e}")


    is_anon = session.get('consent_value', 0) != 1
    file_bytes = file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Extract text
    fake_file = FileStorage(BytesIO(file_bytes), filename=file.filename)
    extracted_text, _ = extract_text_from_file(fake_file)
    if not extracted_text or extracted_text.startswith("["):
        extracted_text = ""

    # ── Anonymous: temp-only storage ──
    if is_anon:
        anon_files = session.get('anon_files', [])
        if len(anon_files) >= 5:
            return jsonify({"error": "匿名用户最多上传5个临时文件，请注册以解锁更多功能"}), 400
        if len(file_bytes) > 5 * 1024 * 1024:
            return jsonify({"error": "匿名用户单文件限制5MB，请注册以解锁"}), 400
        anon_files.append({
            'filename': file.filename,
            'hash': file_hash,
            'size': len(file_bytes),
            'text': extracted_text,
        })
        session['anon_files'] = anon_files
        session.modified = True
        add_to_cache(thread_id, file.filename, extracted_text, user_id)
        return jsonify({
            "success": True, "filename": file.filename, "is_anon": True,
            "anon_count": len(anon_files), "anon_max": 5
        })

    # Registered user: check for existing file by hash
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, filename, original_stored_path FROM user_files WHERE user_id = %s AND file_hash = %s", (user_id, file_hash))
            existing = cur.fetchone()
            if existing and request.form.get('force') != 'true':
                return jsonify({
                    "exists": True,
                    "file_id": existing[0],
                    "filename": existing[1],
                    "original_path": existing[2] if existing[2] else None
                })

    ext = os.path.splitext(file.filename)[1]
    unique_name = f"{file_hash}_{int(time.time())}{ext}"
    original_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)
    os.makedirs(original_dir, exist_ok=True)
    original_path = os.path.join(original_dir, unique_name)
    original_rel = to_rel_path(original_path)
    # Save original binary file
    with open(original_path, 'wb') as f:
        f.write(file_bytes)

    # Add to in‑memory cache
    add_to_cache(thread_id, file.filename, extracted_text, user_id)
    record_file_usage(thread_id, file.filename, 'standalone_upload', "上传文件供日后使用")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if existing and request.form.get('force') == 'true':
                old_path = resolve_path(existing[2])
                if old_path and os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass
                cur.execute("""
                    UPDATE user_files
                    SET filename = %s,
                        size_bytes = %s,
                        original_stored_path = %s,
                        file_hash = %s,
                        expires_at = NULL,
                        original_expires_at = NOW() + INTERVAL '3 days',
                        original_name = %s,
                        content = %s
                    WHERE id = %s
                """, (file.filename, len(file_bytes), original_rel, file_hash, file.filename, extracted_text, existing[0]))
            else:
                ensure_user_exists(user_id)
                cur.execute("""
                    INSERT INTO user_files (user_id, thread_id, filename, size_bytes, expires_at,
                                            original_stored_path, file_hash, original_expires_at, original_name, content)
                    VALUES (%s, %s, %s, %s, NULL, %s, %s, NOW() + INTERVAL '3 days', %s, %s)
                    ON CONFLICT (thread_id, filename) DO UPDATE SET
                        size_bytes = EXCLUDED.size_bytes,
                        original_stored_path = EXCLUDED.original_stored_path,
                        file_hash = EXCLUDED.file_hash,
                        original_expires_at = EXCLUDED.original_expires_at,
                        original_name = EXCLUDED.original_name,
                        content = EXCLUDED.content
                """, (user_id, thread_id, file.filename, len(file_bytes), original_rel, file_hash, file.filename, extracted_text))
            conn.commit()

    return jsonify({"success": True, "filename": file.filename})

@chat_bp.route('/download_original_file', methods=['POST'])
def download_original_file():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403

    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({"error": "Missing filename"}), 400

    user_id = get_user_id()
    thread_id = session.get('thread_id')
    if not thread_id:
        return jsonify({"error": "No active session"}), 400

    if session.get('consent_value', 0) != 1:
        return jsonify({
            "error": "anonymous_not_allowed",
            "message": "匿名用户无法下载原文件。请注册或登录账户后使用此功能。"
        }), 403

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT original_stored_path
                FROM user_files
                WHERE user_id = %s AND thread_id = %s AND filename = %s
                  AND (original_expires_at IS NULL OR original_expires_at > NOW())
            """, (user_id, thread_id, filename))
            row = cur.fetchone()
            if not row or not row[0]:
                return jsonify({"error": "Original file not found or expired"}), 404
            original_path = resolve_path(row[0])
            if not os.path.exists(original_path):
                return jsonify({"error": "File missing on server"}), 404
            return send_file(original_path, as_attachment=True, download_name=filename)

# ── Web page fetch (URL analysis) ──

@chat_bp.route('/fetch_url', methods=['POST'])
def fetch_url():
    """Fetch and extract text content from a web page URL."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Login required"}), 403
    data = request.get_json()
    url = data.get('url', '').strip()
    if not url:
        return jsonify({"error": "URL required"}), 400
    if not url.startswith(('http://', 'https://')):
        return jsonify({"error": "Invalid URL"}), 400

    try:
        from app.services.web_extractor import fetch_page, extract_text_from_html
        html, status = fetch_page(url, retries=2, timeout=15)
        text = extract_text_from_html(html)
        # Truncate to reasonable size
        if len(text) > 50000:
            text = text[:50000] + "\n\n[内容已截断，仅保留前50000字符]"
        return jsonify({"success": True, "text": text, "length": len(text), "url": url})
    except Exception as e:
        logger.error(f"fetch_url failed for {url}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch URL: {e}"}), 500

@chat_bp.route('/delete_file_station', methods=['POST'])
def delete_file_station():
    data = request.get_json()
    file_id = data.get('file_id')
    if not file_id:
        return jsonify({"error": "Missing file_id"}), 400

    user_id = get_user_id()
    is_anon = session.get('consent_value', 0) != 1

    if is_anon:
        anon_files = session.get('anon_files', [])
        idx = int(file_id.replace('anon_', '')) if file_id.startswith('anon_') else -1
        if 0 <= idx < len(anon_files):
            anon_files.pop(idx)
            session['anon_files'] = anon_files
            session.modified = True
            return jsonify({"success": True})
        return jsonify({"error": "File not found"}), 404

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, filename, original_name, content, size_bytes, original_stored_path, file_hash, thread_id, user_id
                    FROM user_files
                    WHERE id = %s AND user_id = %s
                """, (file_id, user_id))
                file_record = cur.fetchone()
                if not file_record:
                    return jsonify({"error": "File not found or not owned"}), 404

                cur.execute("""
                            INSERT INTO recycle_bin
                            (original_table, original_id, user_id, file_name, file_content, file_size,
                             original_stored_path, file_hash, thread_id, deleted_at, expires_at,
                             uploaded_by, deleted_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW() + INTERVAL '3 days',
                                    %s, %s)
                            """, (
                                'user_files', file_record['id'], user_id, file_record['original_name'],
                                file_record['content'], file_record['size_bytes'], file_record['original_stored_path'],
                                file_record['file_hash'], file_record['thread_id'],
                                file_record['user_id'],
                                user_id
                            ))

                cur.execute("DELETE FROM user_files WHERE id = %s AND user_id = %s", (file_id, user_id))

                conn.commit()
                return jsonify({"success": True, "moved_to_recycle_bin": True})

@chat_bp.route('/get_file_station', methods=['GET'])
def get_file_station():
    user_id = get_user_id()
    is_admin_user = session.get('role') == 'admin'
    is_anon = session.get('consent_value', 0) != 1

    # ── Anonymous: return session-based temp files ──
    if is_anon:
        anon_files = session.get('anon_files', [])
        files = [{
            "id": f"anon_{i}",
            "filename": af.get('filename', ''),
            "size_bytes": af.get('size', 0),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": None,
            "is_anon": True,
            "uploaded_by_name": "匿名用户",
        } for i, af in enumerate(anon_files)]
        return jsonify({"files": files, "is_anon": True, "anon_note": "匿名用户文件仅本次会话有效，关闭页面后自动清除，不支持下载"})

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # User files
            cur.execute("""
                SELECT 
                    'user_file' as source,
                    uf.id::text as id,
                    uf.original_name as filename,
                    uf.size_bytes,
                    uf.created_at,
                    uf.expires_at,
                    uf.meta_data,
                    uf.user_id as owner_id,
                    (uf.user_id = %s) as can_delete,
                    NULL as project_name,
                    NULL as project_id,
                    NULL as folder_path,
                    (SELECT json_agg(
                        json_build_object(
                            'usage_type', fu.usage_type,
                            'question', fu.question,
                            'timestamp', fu.timestamp,
                            'thread_id', fu.thread_id
                        ) ORDER BY fu.timestamp DESC
                    ) FROM file_usage fu WHERE fu.user_id = uf.user_id AND fu.filename = uf.original_name LIMIT 10) as usage
                FROM user_files uf
                WHERE uf.user_id = %s AND (uf.expires_at IS NULL OR uf.expires_at > NOW())
                ORDER BY uf.created_at DESC
            """, (user_id, user_id))
            user_files = cur.fetchall()

            # Project files
            if is_admin_user:
                cur.execute("""
                    SELECT 
                        'project_file' as source,
                        pf.id::text as id,
                        pf.original_name as filename,
                        pf.file_size as size_bytes,
                        pf.uploaded_at as created_at,
                        NULL as expires_at,
                        p.name as project_name,
                        p.id as project_id,
                        (SELECT string_agg(f.name, '/') FROM project_folders f WHERE f.id = pf.folder_id) as folder_path,
                        NULL as usage
                    FROM project_files pf
                    JOIN projects p ON pf.project_id = p.id
                    ORDER BY pf.uploaded_at DESC
                """)
            else:
                cur.execute("""
                    SELECT 
                        'project_file' as source,
                        pf.id::text as id,
                        pf.original_name as filename,
                        pf.file_size as size_bytes,
                        pf.uploaded_at as created_at,
                        NULL as expires_at,
                        p.name as project_name,
                        p.id as project_id,
                        (SELECT string_agg(f.name, '/') FROM project_folders f WHERE f.id = pf.folder_id) as folder_path,
                        NULL as usage
                    FROM project_files pf
                    JOIN projects p ON pf.project_id = p.id
                    JOIN project_members pm ON p.id = pm.project_id
                    WHERE pm.user_id = %s
                    ORDER BY pf.uploaded_at DESC
                """, (user_id,))
            project_files = cur.fetchall()

    all_files = user_files + project_files
    return jsonify({"files": all_files, "is_admin": is_admin_user})

@chat_bp.route('/load_project_file', methods=['POST'])
def load_project_file():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = get_user_id()
    data = request.get_json()
    project_id = data.get('project_id')
    file_id = data.get('file_id')
    if not project_id or not file_id:
        return jsonify({"error": "Missing project_id or file_id"}), 400

    if not is_admin() and not can_access_project(project_id, user_id):
        return jsonify({"error": "Access denied"}), 403

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT stored_path, original_name
                FROM project_files
                WHERE id = %s AND project_id = %s
            """, (file_id, project_id))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "File not found"}), 404
            stored_path, original_name = row
            stored_path = resolve_path(stored_path)
            if not os.path.exists(stored_path):
                return jsonify({"error": "File missing on server"}), 404

            with open(stored_path, 'rb') as f:
                file_bytes = f.read()

            fake_file = FileStorage(BytesIO(file_bytes), filename=original_name)
            text, _ = extract_text_from_file(fake_file)
            if not text or text.startswith("["):
                return jsonify({"error": "Could not extract text from file"}), 400

            return jsonify({"content": text, "filename": original_name})

# ---------- Batch compare endpoints ----------

@chat_bp.route('/get_recycle_bin', methods=['GET'])
def get_recycle_bin():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DO $$ 
                BEGIN
                    BEGIN
                        ALTER TABLE recycle_bin ADD COLUMN deletion_reason TEXT DEFAULT 'manual';
                    EXCEPTION WHEN duplicate_column THEN NULL;
                    END;
                END $$;
            """)
            conn.commit()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            from app.services.recycle_bin_service import get_recycle_items
            return jsonify(get_recycle_items(user_id, cur))

@chat_bp.route('/restore_from_recycle_bin', methods=['POST'])
def restore_from_recycle_bin():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    item_id = data.get('item_id')
    source = data.get('source')
    section = data.get('section')
    restore_all = data.get('restore_all', False)

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if restore_all:
                    from app.services.recycle_bin_service import bulk_restore_all
                    restored_count = bulk_restore_all(section, user_id, conn, cur)
                    return jsonify({"success": True, "restored_count": restored_count})

                from app.services.recycle_bin_service import restore_recycle_item
                restore_recycle_item(item_id, source, conn, cur, user_id)
                conn.commit()
                return jsonify({"success": True})

@chat_bp.route('/delete_recycle_item', methods=['POST'])
def delete_recycle_item():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    item_id = data.get('item_id')
    source = data.get('source')

    if not item_id or not source:
        return jsonify({"error": "Missing item_id or source"}), 400

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                from app.services.recycle_bin_service import permanently_delete_item
                permanently_delete_item(item_id, source, cur, user_id)
                conn.commit()
                return jsonify({"success": True})

@chat_bp.route('/empty_recycle_bin', methods=['POST'])
def empty_recycle_bin():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    source = data.get('source')

    with get_db_connection() as conn:
        with db_transaction(conn):
            with conn.cursor() as cur:
                from app.services.recycle_bin_service import empty_recycle_bin
                empty_recycle_bin(source, user_id, cur)
                conn.commit()
                return jsonify({"success": True})



@chat_bp.route('/api/chat/create_grill_thread', methods=['POST'])
def create_grill_thread():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Unauthorized"}), 403
    user_id = get_user_id()
    thread_id = str(uuid.uuid4())
    title = "质问模式对话"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_sessions (user_id, thread_id, title, is_grilling, created_at, updated_at) VALUES (%s, %s, %s, TRUE, %s, %s)",
                    (user_id, thread_id, title, utc_now(), utc_now())
                )
                conn.commit()
        return jsonify({"thread_id": thread_id})
    except Exception as e:
        logger.error(f"Failed to create grill thread: {e}")
        return jsonify({"error": "Server error"}), 500

@chat_bp.route('/api/projects/<int:project_id>/get_or_create_grill_thread', methods=['POST'])
def get_or_create_project_grill_thread(project_id):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Unauthorized"}), 403
    user_id = get_user_id()
    if not can_access_project(user_id, project_id):
        return jsonify({"error": "Forbidden"}), 403
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT thread_id FROM chat_sessions WHERE project_id = %s AND is_grilling = TRUE LIMIT 1", (project_id,))
                row = cur.fetchone()
                if row:
                    return jsonify({"thread_id": row['thread_id']})
                
                # Create it
                thread_id = str(uuid.uuid4())
                cur.execute("SELECT name FROM projects WHERE id = %s", (project_id,))
                proj_name = cur.fetchone()['name']
                title = f"{proj_name} - 项目质问模式"
                
                cur.execute(
                    "INSERT INTO chat_sessions (user_id, thread_id, title, project_id, is_grilling, created_at, updated_at) VALUES (%s, %s, %s, %s, TRUE, %s, %s)",
                    (user_id, thread_id, title, project_id, utc_now(), utc_now())
                )
                conn.commit()
                return jsonify({"thread_id": thread_id})
    except Exception as e:
        logger.error(f"Failed to get or create project grill thread: {e}")
        return jsonify({"error": "Server error"}), 500
