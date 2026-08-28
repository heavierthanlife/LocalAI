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



@chat_bp.route('/check_storage', methods=['GET'])
def check_storage():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = get_user_id()
    total_bytes = get_user_total_storage_size(user_id)
    from app.services.file_store import MAX_TOTAL_UPLOAD_GB
    limit_gb = float(MAX_TOTAL_UPLOAD_GB)
    total_gb = total_bytes / (1024 * 1024 * 1024)
    # Warn at 80% of the unified upload quota (same cap as stream_upload)
    warning = total_bytes > limit_gb * 0.8 * (1024 ** 3)
    return jsonify({
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "limit_gb": limit_gb,
        "warning": warning,
        "message": f"文件存储已使用 {total_gb:.1f} GB / {limit_gb:.0f} GB" if warning else None
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


# ── Sub-modules (route groups registered on the shared chat_bp) ──
from app.routes import chat_files  # noqa: F401  (registers file-station + recycle-bin routes)
from app.routes import chat_sessions  # noqa: F401  (registers session-management routes)
from app.routes import chat_config  # noqa: F401  (registers llm-config + misc routes)
