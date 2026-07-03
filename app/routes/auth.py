"""Blueprint: auth routes (auto-extracted)."""
import os, json, uuid, time, logging, hashlib, io, secrets
from flask import Blueprint, request, jsonify, session, send_file, render_template, url_for

from app.config import BASE_DIR, DATA_DIR, TEMP_ROOT, TEMP_DIR, USER_FILES_ORIGINAL_ROOT, logger
from app.database import get_db_connection, db_transaction
from app.utils.helpers import utc_now, beijing_now, safe_error_response, split_thinking_answer
import app.globals as g
from app.services.file_cache import file_cache_manager, add_to_cache, load_cache_from_db
from app.services.session_manager import get_or_create_session

from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint('auth', __name__, template_folder=str(BASE_DIR / 'templates'), static_folder=str(BASE_DIR / 'static'))

@auth_bp.route('/check_auth', methods=['GET'])
def check_auth():
    if session.get('consent_value', 0) != 1:
        return jsonify({"authenticated": False, "reason": "consent_not_given"})
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"authenticated": False, "reason": "no_user_id"})

    username = session.get('username')
    role = session.get('role')

    # If username not in session, try to fetch from database
    if not username:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT username, role, is_auditor FROM users WHERE user_id = %s", (user_id,))
                row = cur.fetchone()
                if row:
                    username = row[0]
                    role = row[1] or 'user'
                    is_auditor = bool(row[2]) if len(row) > 2 else False
                    # Restore to session for future requests
                    session['username'] = username
                    session['role'] = role
                    session['is_auditor'] = is_auditor
                else:
                    return jsonify({"authenticated": False, "reason": "user_deleted"})

    is_auditor = session.get('is_auditor', False)

    return jsonify({
        "authenticated": True,
        "username": username,
        "is_admin": role == 'admin',
        "is_auditor": is_auditor,
        "role": role,
        "user_id": user_id
    })

@auth_bp.route('/create_account', methods=['POST'])
def create_account():
    # Registration implies consent — no separate /consent call needed
    data = request.get_json()
    username = data.get('username', '').strip()
    pin = data.get('pin', '').strip()
    pin_length = data.get('pin_length', 6)

    if not username or not pin:
        return jsonify({"error": "用户名和PIN不能为空"}), 400
    if len(username) < 5 or len(username) > 18:
        return jsonify({"error": "用户名长度应为5-18个字符"}), 400
    if pin_length not in [4, 6] or len(pin) != pin_length:
        return jsonify({"error": f"PIN必须是{pin_length}位数字"}), 400
    if not pin.isdigit():
        return jsonify({"error": "PIN只能包含数字"}), 400

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                return jsonify({"error": "用户名已存在"}), 409

            user_id = session.get('user_id')
            if not user_id:
                user_id = str(uuid.uuid4())
                session['user_id'] = user_id

            pin_hash = generate_password_hash(pin)

            cur.execute("""
                INSERT INTO users (user_id, username, pin_hash, pin_length, created_at, role)
                VALUES (%s, %s, %s, %s, NOW(), 'user')
                ON CONFLICT (user_id) DO UPDATE SET
                    username = EXCLUDED.username,
                    pin_hash = EXCLUDED.pin_hash,
                    pin_length = EXCLUDED.pin_length,
                    role = 'user'
                RETURNING user_id
            """, (user_id, username, pin_hash, pin_length))

            # Record consent (registration implies consent)
            cur.execute(
                "INSERT INTO consent (thread_id, consent_given, timestamp) VALUES (%s, %s, NOW()) ON CONFLICT (thread_id) DO UPDATE SET consent_given = EXCLUDED.consent_given, timestamp = EXCLUDED.timestamp",
                (session.get('thread_id', str(uuid.uuid4())), 1)
            )

            conn.commit()
            session['username'] = username
            session['role'] = 'user'
            session['is_auditor'] = False
            session['consent_value'] = 1
            session.modified = True

            return jsonify({"success": True, "username": username})

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    pin = data.get('pin', '').strip()
    if not username or not pin:
        return jsonify({"error": "用户名和PIN不能为空"}), 400

    if username in ("admin", "CEO", "COO"):
        from flask import current_app
        # All admin accounts share the same PIN
        admin_hash = current_app.config.get('ADMIN_PASSWORD_HASH')
        if not admin_hash:
            logger.error("ADMIN_PASSWORD_HASH not set in environment")
            return jsonify({"error": "管理员账户未配置"}), 500
        if not check_password_hash(admin_hash, pin):
            logger.warning(f"Admin login failed for {username}")
            return jsonify({"error": "用户名或PIN错误"}), 401

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM users WHERE username = %s", (username,))
                admin_row = cur.fetchone()
                if admin_row:
                    user_id = admin_row[0]
                else:
                    user_id = str(uuid.uuid4())
                    cur.execute(
                        "INSERT INTO users (user_id, username, role) VALUES (%s, %s, %s)",
                        (user_id, username, 'admin')
                    )
                conn.commit()

        session['user_id'] = user_id
        session['consent_value'] = 1
        session['username'] = username
        session['role'] = 'admin'
        session['is_auditor'] = True  # admin auto-includes auditor
        session.permanent = True

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO consent (thread_id, consent_given, timestamp) VALUES (%s, %s, NOW()) ON CONFLICT (thread_id) DO UPDATE SET consent_given = EXCLUDED.consent_given, timestamp = EXCLUDED.timestamp",
                    (session.get('thread_id', str(uuid.uuid4())), 1)
                )
                conn.commit()

        logger.info(f"Admin logged in: {user_id}")
        return jsonify({
            "success": True,
            "username": "admin",
            "is_admin": True,
            "is_auditor": True,
            "user_id": session['user_id']
        })

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT user_id, pin_hash, pin_length, role, is_auditor FROM users WHERE username = %s AND is_active = TRUE AND pin_hash IS NOT NULL",
                (username,)
            )
            user = cur.fetchone()
            if not user or not check_password_hash(user['pin_hash'], pin):
                return jsonify({"error": "用户名或PIN错误"}), 401

            session['user_id'] = user['user_id']
            session['consent_value'] = 1
            session['username'] = username
            session['role'] = user.get('role', 'user')
            session['is_auditor'] = bool(user.get('is_auditor', False))
            session.permanent = True

            with conn.cursor() as cur2:
                cur2.execute(
                    "INSERT INTO consent (thread_id, consent_given, timestamp) VALUES (%s, %s, NOW()) ON CONFLICT (thread_id) DO UPDATE SET consent_given = EXCLUDED.consent_given, timestamp = EXCLUDED.timestamp",
                    (session.get('thread_id', str(uuid.uuid4())), 1)
                )
            conn.commit()
            return jsonify({
                "success": True,
                "username": username,
                "is_admin": session['role'] == 'admin',
                "is_auditor": session.get('is_auditor', False),
                "user_id": session['user_id']
            })

@auth_bp.route('/update_account', methods=['POST'])
def update_account():
    """Update username (direct) or request PIN change (needs email verification)."""
    if session.get('consent_value', 0) != 1 or not session.get('user_id'):
        return jsonify({"error": "未登录"}), 401

    data = request.get_json()
    new_username = data.get('new_username', '').strip()
    new_pin = data.get('new_pin', '').strip()
    pin_length = int(data.get('pin_length', 6))
    current_pin = data.get('current_pin', '').strip()
    verify_code = data.get('verify_code', '').strip()

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT pin_hash, email FROM users WHERE user_id = %s", (session['user_id'],))
            user = cur.fetchone()
            if not user:
                return jsonify({"error": "用户不存在"}), 404

            existing_hash = user['pin_hash']
            user_email = user.get('email', '')

            if existing_hash is not None:
                if not current_pin or not check_password_hash(existing_hash, current_pin):
                    return jsonify({"error": "当前PIN错误"}), 401

            updates = []
            params = []

            if new_username:
                if len(new_username) < 5 or len(new_username) > 18:
                    return jsonify({"error": "用户名长度应为5-18个字符"}), 400
                cur.execute("SELECT 1 FROM users WHERE username = %s AND user_id != %s",
                            (new_username, session['user_id']))
                if cur.fetchone():
                    return jsonify({"error": "用户名已存在"}), 409
                updates.append("username = %s")
                params.append(new_username)
                session['username'] = new_username

            if new_pin:
                # PIN change requires email verification code
                if not user_email:
                    return jsonify({"error": "请先在账户面板设置邮箱"}), 400
                if not verify_code:
                    return jsonify({"error": "需要验证码"}), 400
                expected_code = session.get('pin_change_code')
                code_expiry = session.get('pin_change_code_expiry', 0)
                if not expected_code or time.time() > code_expiry:
                    session.pop('pin_change_code', None)
                    session.pop('pin_change_code_expiry', None)
                    return jsonify({"error": "验证码已过期，请重新获取"}), 400
                if verify_code != expected_code:
                    return jsonify({"error": "验证码错误"}), 400
                # Code verified — clean up
                session.pop('pin_change_code', None)
                session.pop('pin_change_code_expiry', None)
                if pin_length not in (4, 6) or len(new_pin) != pin_length:
                    return jsonify({"error": f"PIN必须是{pin_length}位数字"}), 400
                if not new_pin.isdigit():
                    return jsonify({"error": "PIN只能包含数字"}), 400
                updates.append("pin_hash = %s")
                params.append(generate_password_hash(new_pin))
                updates.append("pin_length = %s")
                params.append(pin_length)

            if updates:
                params.append(session['user_id'])
                cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE user_id = %s", params)
                conn.commit()
            return jsonify({"success": True})


@auth_bp.route('/request_pin_change_code', methods=['POST'])
def request_pin_change_code():
    """Send a 4-digit verification code to user's email for PIN change."""
    if session.get('consent_value', 0) != 1 or not session.get('user_id'):
        return jsonify({"error": "未登录"}), 401
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT email, username FROM users WHERE user_id = %s", (session['user_id'],))
            row = cur.fetchone()
            if not row or not row[0]:
                return jsonify({"error": "未设置邮箱，请先在账户面板填写"}), 400
            user_email, username = row
    code = f"{secrets.randbelow(10000):04d}"
    session['pin_change_code'] = code
    session['pin_change_code_expiry'] = time.time() + 300  # 5 minutes
    session.modified = True
    from app.utils.mailer import send_email, is_configured
    if not is_configured():
        logger.info(f"[PIN_CODE] User {username}: {code} (SMTP not configured)")
        return jsonify({"status": "ok", "hint": f"(调试模式) 验证码: {code}"})
    send_email(user_email, "[中联AI] PIN变更验证码",
               f"验证码: {code}\n有效期5分钟。如非本人操作请忽略。", async_mode=True)
    return jsonify({"status": "ok", "hint": f"验证码已发送至 {user_email}"})


@auth_bp.route('/set_email', methods=['POST'])
def set_user_email():
    """Set the user's email address for verification purposes."""
    if session.get('consent_value', 0) != 1 or not session.get('user_id'):
        return jsonify({"error": "未登录"}), 401
    data = request.get_json(silent=True) or {}
    email = (data.get('email', '') or '').strip()
    if not email or '@' not in email:
        return jsonify({"error": "请输入有效的邮箱地址"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET email = %s WHERE user_id = %s", (email, session['user_id']))
            conn.commit()
    return jsonify({"status": "ok", "email": email})

@auth_bp.route('/request_delete_account', methods=['POST'])
def request_delete_account():
    """User requests account deletion — returns inventory for keep/delete selection."""
    if session.get('consent_value', 0) != 1 or not session.get('user_id'):
        return jsonify({"error": "请先登录"}), 401
    user_id = session['user_id']
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT email FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            user_email = row['email'] if row else ''
            inventory = []
            cur.execute("SELECT id, filename, size_bytes, original_name FROM user_files WHERE user_id = %s ORDER BY filename", (user_id,))
            for f in cur.fetchall():
                inventory.append({'type': 'chat_file', 'id': f['id'], 'name': f['original_name'] or f['filename'], 'size_kb': round((f['size_bytes'] or 0) / 1024, 1), 'choosable': True})
            cur.execute("SELECT id, filename, file_size, original_name FROM knowledge_lab_files WHERE user_id = %s ORDER BY filename", (user_id,))
            for f in cur.fetchall():
                inventory.append({'type': 'kb_file', 'id': f['id'], 'name': f['original_name'] or f['filename'], 'size_kb': round((f['file_size'] or 0) / 1024, 1), 'choosable': True})
            cur.execute("SELECT id, task_id, companies_count FROM credit_check_reports WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
            for f in cur.fetchall():
                inventory.append({'type': 'credit_report', 'id': f['id'], 'name': f'征信报告({f["companies_count"]}家公司)', 'size_kb': 0, 'choosable': True})
            cur.execute("SELECT id, task_id, file_count FROM batch_comparison_results WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
            for f in cur.fetchall():
                inventory.append({'type': 'batch_result', 'id': f['id'], 'name': f'批量对比({f["file_count"]}个文件)', 'size_kb': 0, 'choosable': True})
            cur.execute("SELECT COUNT(*) as cnt FROM chat_sessions WHERE user_id = %s", (user_id,))
            session_cnt = cur.fetchone()['cnt']
            cur.execute("SELECT COUNT(*) as cnt FROM project_files WHERE uploaded_by = %s", (user_id,))
            proj_cnt = cur.fetchone()['cnt']
            if proj_cnt > 0:
                inventory.append({'type': 'project_file', 'id': 0, 'name': f'{proj_cnt}个项目文件', 'size_kb': 0, 'choosable': False, 'note': '默认保留归入公司托管'})
    return jsonify({"status": "ok", "inventory": inventory, "session_count": session_cnt, "email": user_email})

@auth_bp.route('/confirm_delete_account', methods=['POST'])
def confirm_delete_account():
    """User confirms deletion with approval code + PIN — then runs actual deletion."""
    if session.get('consent_value', 0) != 1 or not session.get('user_id'):
        return jsonify({"error": "未登录"}), 401
    data = request.get_json()
    pin = data.get('pin', '').strip()
    code = data.get('code', '').strip()
    user_id = session['user_id']
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT deletion_code, email FROM users WHERE user_id = %s AND deletion_requested = TRUE", (user_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "未找到待处理的删除请求"}), 400
            expected_code = row[0]
            if not expected_code or code != expected_code:
                return jsonify({"error": "验证码错误"}), 400
    # Code is correct — proceed with deletion using stored choices
    keep_ids = session.get('delete_keep_ids', [])
    session.pop('delete_keep_ids', None)
    return delete_account_impl(user_id, pin, keep_ids)

@auth_bp.route('/submit_delete_choices', methods=['POST'])
def submit_delete_choices():
    """User selects keep/delete items, then admin is notified for approval."""
    if session.get('consent_value', 0) != 1 or not session.get('user_id'):
        return jsonify({"error": "请先登录"}), 401
    user_id = session['user_id']
    username = session.get('username', '')
    data = request.get_json(silent=True) or {}
    keep_ids = data.get('keep_ids', [])
    # Store choices in session and set deletion_requested flag
    session['delete_keep_ids'] = keep_ids
    session.modified = True
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT email FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            user_email = row[0] if row else ''
            cur.execute("UPDATE users SET deletion_requested = TRUE WHERE user_id = %s", (user_id,))
            conn.commit()
    from app.services.admin_utils import log_admin_action
    log_admin_action(user_id, username, 'DELETE_REQUEST', 'users', user_id, column_name='deletion_requested',
                    old_value='false', new_value=f'keep_{len(keep_ids)}items')
    from app.utils.mailer import notify_admin, is_configured
    if is_configured():
        notify_admin("账户删除申请", f"用户 {username} ({user_email}) 申请删除账户（保留{len(keep_ids)}项数据），请登录审核。")
    return jsonify({"status": "ok", "message": "已提交删除申请，等待管理员审核"})


@auth_bp.route('/delete_account', methods=['POST'])
def delete_account():
    """Legacy direct deletion (kept for admin override)."""
    if session.get('consent_value', 0) != 1 or not session.get('user_id'):
        return jsonify({"error": "未登录"}), 401
    data = request.get_json()
    pin = data.get('pin', '').strip()
    user_id = session['user_id']
    return delete_account_impl(user_id, pin)


def delete_account_impl(user_id, pin, keep_ids=None):
    """Core account deletion logic — shared by direct and approved paths.
    keep_ids: list of {'type': 'chat_file'|'kb_file'|..., 'id': int} to deposit before deletion.
    Project files are ALWAYS deposited regardless of keep_ids.
    """
    if keep_ids is None:
        keep_ids = []

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT pin_hash FROM users WHERE user_id = %s", (user_id,))
            user = cur.fetchone()
            if not user:
                return jsonify({"error": "用户不存在"}), 404

            pin_hash = user['pin_hash']

            if pin_hash is None:
                return jsonify({"error": "您的账户没有设置PIN，无法删除。请先通过账户设置设置PIN后再试。"}), 400

            if not check_password_hash(pin_hash, pin):
                return jsonify({"error": "PIN错误"}), 401

            with db_transaction(conn):
                cur.execute("""
                    SELECT DISTINCT p.id, p.name, p.created_by
                    FROM projects p
                    LEFT JOIN project_members pm ON p.id = pm.project_id
                    WHERE pm.user_id = %s OR p.created_by = %s
                """, (user_id, user_id))
                projects = cur.fetchall()
                for proj in projects:
                    proj_id = proj['id']
                    proj_name = proj['name']
                    cur.execute("""
                        INSERT INTO task_deposit_items (original_user_id, original_username, project_id,
                                                        project_name, item_type, item_data)
                        VALUES (%s, %s, %s, %s, 'project', %s)
                    """, (user_id, user.get('username', 'unknown'), proj_id, proj_name,
                          json.dumps({'project_id': proj_id, 'name': proj_name})))
                    cur.execute("""
                        SELECT id, original_name, stored_path, uploaded_by, folder_id, filename, version, file_size
                        FROM project_files WHERE project_id = %s
                    """, (proj_id,))
                    files = cur.fetchall()
                    for f in files:
                        cur.execute("""
                            INSERT INTO task_deposit_items (original_user_id, original_username, project_id,
                                                            project_name, item_type, item_data, stored_path)
                            VALUES (%s, %s, %s, %s, 'file', %s, %s)
                        """, (user_id, user.get('username', 'unknown'), proj_id, proj_name,
                              json.dumps(dict(f)), f['stored_path']))
                    cur.execute("""
                        SELECT id, name, parent_folder_id, created_by
                        FROM project_folders WHERE project_id = %s
                    """, (proj_id,))
                    folders = cur.fetchall()
                    for fold in folders:
                        cur.execute("""
                            INSERT INTO task_deposit_items (original_user_id, original_username, project_id,
                                                            project_name, item_type, item_data)
                            VALUES (%s, %s, %s, %s, 'folder', %s)
                        """, (user_id, user.get('username', 'unknown'), proj_id, proj_name,
                              json.dumps(dict(fold))))
                    cur.execute("""
                        SELECT id, file_id, user_id, comment, created_at
                        FROM project_file_comments
                        WHERE file_id IN (SELECT id FROM project_files WHERE project_id = %s)
                    """, (proj_id,))
                    comments = cur.fetchall()
                    for comm in comments:
                        cur.execute("""
                            INSERT INTO task_deposit_items (original_user_id, original_username, project_id,
                                                            project_name, item_type, item_data)
                            VALUES (%s, %s, %s, %s, 'comment', %s)
                        """, (user_id, user.get('username', 'unknown'), proj_id, proj_name,
                              json.dumps(dict(comm))))

                cur.execute("SELECT user_id FROM users WHERE role = 'admin' AND user_id != %s LIMIT 1", (user_id,))
                admin_row = cur.fetchone()
                admin_id = admin_row['user_id'] if admin_row else None

                if admin_id:
                    cur.execute("UPDATE projects SET created_by = %s WHERE created_by = %s", (admin_id, user_id))
                else:
                    cur.execute("UPDATE projects SET created_by = NULL WHERE created_by = %s", (user_id,))

                if admin_id:
                    cur.execute("UPDATE project_folders SET created_by = %s WHERE created_by = %s", (admin_id, user_id))
                else:
                    cur.execute("UPDATE project_folders SET created_by = NULL WHERE created_by = %s", (user_id,))

                if admin_id:
                    cur.execute("UPDATE project_files SET uploaded_by = %s WHERE uploaded_by = %s", (admin_id, user_id))
                else:
                    cur.execute("UPDATE project_files SET uploaded_by = NULL WHERE uploaded_by = %s", (user_id,))

                if admin_id:
                    cur.execute("""
                        UPDATE project_file_versions SET uploaded_by = %s
                        WHERE uploaded_by = %s
                    """, (admin_id, user_id))
                else:
                    cur.execute("UPDATE project_file_versions SET uploaded_by = NULL WHERE uploaded_by = %s", (user_id,))

                if admin_id:
                    cur.execute("UPDATE project_members SET added_by = %s WHERE added_by = %s", (admin_id, user_id))
                else:
                    cur.execute("UPDATE project_members SET added_by = NULL WHERE added_by = %s", (user_id,))

                cur.execute("UPDATE project_file_comments SET user_id = NULL WHERE user_id = %s", (user_id,))
                cur.execute("UPDATE project_folder_comments SET user_id = NULL WHERE user_id = %s", (user_id,))
                cur.execute("UPDATE task_deposit_items SET transferred_to_user_id = NULL WHERE transferred_to_user_id = %s", (user_id,))

                # ── Deposit user-selected non-project items before deletion ──
                keep_map = {}
                for k in keep_ids:
                    t = k.get('type', '') if isinstance(k, dict) else None
                    kid = k.get('id', 0) if isinstance(k, dict) else None
                    if t and kid is not None:
                        keep_map.setdefault(t, []).append(kid)

                if 'chat_file' in keep_map:
                    kept_ids = tuple(keep_map['chat_file'])
                    cur.execute(f"SELECT id, filename, size_bytes, original_name, original_stored_path FROM user_files WHERE user_id = %s AND id IN %s", (user_id, kept_ids))
                    for uf in cur.fetchall():
                        cur.execute("INSERT INTO task_deposit_items (original_user_id, original_username, project_id, project_name, item_type, item_data, stored_path) VALUES (%s,%s,NULL,'个人文件','user_file',%s,%s)",
                            (user_id, user.get('username','unknown'), json.dumps(dict(uf)), uf['original_stored_path']))

                if 'kb_file' in keep_map:
                    kept_ids = tuple(keep_map['kb_file'])
                    cur.execute(f"SELECT id, filename, file_size, original_name FROM knowledge_lab_files WHERE user_id = %s AND id IN %s", (user_id, kept_ids))
                    for kf in cur.fetchall():
                        cur.execute("INSERT INTO task_deposit_items (original_user_id, original_username, project_id, project_name, item_type, item_data) VALUES (%s,%s,NULL,'个人知识库','knowledge_lab',%s)",
                            (user_id, user.get('username','unknown'), json.dumps(dict(kf))))

                if 'credit_report' in keep_map:
                    kept_ids = tuple(keep_map['credit_report'])
                    cur.execute(f"SELECT id, task_id, companies_count, file_path FROM credit_check_reports WHERE user_id = %s AND id IN %s", (user_id, kept_ids))
                    for cr in cur.fetchall():
                        cur.execute("INSERT INTO task_deposit_items (original_user_id, original_username, project_id, project_name, item_type, item_data, stored_path) VALUES (%s,%s,NULL,'征信报告','credit_report',%s,%s)",
                            (user_id, user.get('username','unknown'), json.dumps(dict(cr)), cr['file_path']))

                if 'batch_result' in keep_map:
                    kept_ids = tuple(keep_map['batch_result'])
                    cur.execute(f"SELECT id, task_id, file_count, pair_count, file_names, zip_path FROM batch_comparison_results WHERE user_id = %s AND id IN %s", (user_id, kept_ids))
                    for br in cur.fetchall():
                        cur.execute("INSERT INTO task_deposit_items (original_user_id, original_username, project_id, project_name, item_type, item_data, stored_path) VALUES (%s,%s,NULL,'批量对比','batch_result',%s,%s)",
                            (user_id, user.get('username','unknown'), json.dumps(dict(br)), br['zip_path']))

                cur.execute("SELECT id, task_id, companies_count, file_path FROM credit_check_reports WHERE user_id = %s", (user_id,))
                for cr in cur.fetchall():
                    cur.execute("""INSERT INTO task_deposit_items (original_user_id, original_username, project_id,
                        project_name, item_type, item_data, stored_path) VALUES (%s,%s,NULL,'征信报告','credit_report',%s,%s)""",
                        (user_id, user.get('username','unknown'), json.dumps(dict(cr)), cr['file_path']))

                cur.execute("SELECT id, task_id, file_count, pair_count, file_names, zip_path FROM batch_comparison_results WHERE user_id = %s", (user_id,))
                for br in cur.fetchall():
                    cur.execute("""INSERT INTO task_deposit_items (original_user_id, original_username, project_id,
                        project_name, item_type, item_data, stored_path) VALUES (%s,%s,NULL,'批量对比','batch_result',%s,%s)""",
                        (user_id, user.get('username','unknown'), json.dumps(dict(br)), br['zip_path']))

                cur.execute("DELETE FROM recycle_bin WHERE user_id = %s", (user_id,))
                cur.execute("UPDATE project_recycle_bin SET uploaded_by = NULL WHERE uploaded_by = %s", (user_id,))
                cur.execute("UPDATE task_deposit_items SET original_user_id = NULL WHERE original_user_id = %s", (user_id,))
                cur.execute("UPDATE task_deposit_permissions SET manager_id = NULL WHERE manager_id = %s", (user_id,))
                cur.execute("UPDATE task_deposit_permissions SET granted_by = NULL WHERE granted_by = %s", (user_id,))
                cur.execute("DELETE FROM project_members WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM chat_messages WHERE thread_id IN (SELECT thread_id FROM chat_sessions WHERE user_id = %s)", (user_id,))
                cur.execute("DELETE FROM user_files WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM file_usage WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM feedback WHERE thread_id IN (SELECT thread_id FROM chat_sessions WHERE user_id = %s)", (user_id,))
                cur.execute("DELETE FROM consent WHERE thread_id IN (SELECT thread_id FROM chat_sessions WHERE user_id = %s)", (user_id,))
                cur.execute("DELETE FROM chat_sessions WHERE user_id = %s", (user_id,))
                cur.execute("DELETE FROM users WHERE user_id = %s", (user_id,))

                conn.commit()

            session.clear()
            session['consent_value'] = 0
            session['thread_id'] = str(uuid.uuid4())
            get_or_create_session(session['thread_id'])
            return jsonify({"success": True})
# Check if column_name exists in table_name
def validate_table_column(table_name, column_name):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s AND column_name = %s
            """, (table_name, column_name))
            return cur.fetchone() is not None

# Task deposit endpoints
