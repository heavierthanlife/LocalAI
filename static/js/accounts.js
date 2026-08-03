/* Account Management module - extracted from app.js (W4) */
    // ======================== Account Management ========================
    async function loadUserStyle() {
        const section = document.getElementById('userStyleSection');
        if (!section) return;
        try {
            const r = await fetch('/my_writing_style', { credentials: 'include' });
            const d = await r.json();
            const s = d.style || {};
            if (s.style_label && s.style_label !== 'Not analyzed') {
                section.innerHTML = `<div>📊 <b>${escapeHtml(s.style_label)}</b></div>
                    <div style="font-size:0.65rem;margin-top:2px;">${escapeHtml(s.style_description||'')}</div>
                    <div style="font-size:0.6rem;color:var(--card-muted);margin-top:2px;">
                        基于 ${s.total_analyzed||0} 条消息 · v${s.version||1}
                        · <button id="reAnalyzeStyleBtn" style="font-size:0.6rem;background:none;border:none;color:#3b82f6;cursor:pointer;text-decoration:underline;">刷新</button>
                    </div>`;
                document.getElementById('reAnalyzeStyleBtn').onclick = async () => {
                    section.innerHTML = '分析中...';
                    await fetch('/my_writing_style/analyze', { method:'POST', credentials:'include' });
                    loadUserStyle();
                };
            } else {
                section.innerHTML = `<span style="font-size:0.7rem;">暂无风格画像。</span>
                    <br><button id="analyzeStyleBtn" class="file-btn" style="font-size:0.65rem;padding:2px 8px;margin-top:4px;">✍️ 分析我的风格</button>`;
                document.getElementById('analyzeStyleBtn').onclick = async () => {
                    section.innerHTML = '分析中...';
                    await fetch('/my_writing_style/analyze', { method:'POST', credentials:'include' });
                    loadUserStyle();
                };
            }
        } catch (_) { section.innerHTML = '<span style="font-size:0.7rem;">不可用</span>'; }
    }

    var accountModal = document.getElementById('accountModal');
    var closeAccountModal = document.getElementById('closeAccountModal');
    var accountContent = document.getElementById('accountContent');

    async function loadAccountModal() {
        // Fetch current auth status from server (consent_given now comes from API)
        const authRes = await fetch('/check_auth', { credentials: 'include' });
        const authData = await authRes.json();
        // Consent survives logout via localStorage so returning users skip re-consent
        const consentGiven = authData.consent_given || localStorage.getItem('consent_given') === '1';
        const isLoggedIn = authData.authenticated;
        // Save role for tab visibility (skill_auditor needs it)
        if (authData.role) sessionStorage.setItem('role', authData.role);
        else if (authData.is_admin) sessionStorage.setItem('role', 'admin');
        sessionStorage.setItem('is_auditor', (authData.is_auditor || authData.is_admin) ? '1' : '0');
        const username = authData.username || sessionStorage.getItem('username') || '';
        const isAdmin = authData.is_admin || false;

        if (consentGiven && isLoggedIn) {
            const roleLabel = authData.is_admin ? '管理员' : (authData.is_auditor ? '审核员' : '用户');
            accountContent.innerHTML = `
                <p>已登录: <strong>${escapeHtml(username)}</strong> · <span style="font-size:.8rem;color:var(--card-muted);">角色: ${roleLabel}</span></p>
                <hr>
                <div style="margin-bottom:12px;">
                    <strong>✍️ 我的写作风格</strong>
                    <div id="userStyleSection" style="font-size:0.75rem;color:var(--card-muted);margin-top:4px;">加载中...</div>
                </div>
                <hr>
                <label>邮箱 (用于PIN变更验证):</label>
                <div style="display:flex;gap:6px;margin-bottom:8px;">
                    <input type="email" id="userEmail" placeholder="your@company.com" style="flex:1;">
                    <button id="saveEmailBtn" class="file-btn" style="padding:4px 12px;font-size:.7rem;">保存</button>
                </div>
                <hr>
                <label>新用户名 (5-18字符):</label>
                <input type="text" id="newUsername" placeholder="留空则不修改" style="width:100%; margin-bottom:8px;">
                <label>新PIN (4或6位数字):</label>
                <select id="newPinLength" style="width:100%; margin-bottom:4px;">
                    <option value="4">4位</option>
                    <option value="6" selected>6位</option>
                </select>
                <input type="password" id="newPin" placeholder="新PIN" style="width:100%; margin-bottom:4px;">
                <input type="password" id="confirmNewPin" placeholder="确认新PIN" style="width:100%; margin-bottom:4px;">
                <div style="display:flex;gap:6px;margin-bottom:8px;">
                    <input type="text" id="pinVerifyCode" placeholder="邮箱验证码(4位)" maxlength="4" style="flex:1;">
                    <button id="requestPinCodeBtn" class="file-btn" style="padding:4px 10px;font-size:.7rem;background:#bccfde;">获取验证码</button>
                </div>
                <label>当前PIN (必填):</label>
                <input type="password" id="currentPin" placeholder="当前PIN" style="width:100%; margin-bottom:8px;">
                <button id="updateAccountBtn" class="file-btn" style="margin-top: 10px;">更新账户</button>
                <button id="deleteAccountBtn" class="file-btn" style="margin-top: 10px; font-weight: bold;">${sessionStorage.getItem('deletion_pending') === '1' ? '⚠️ 输入验证码确认删除' : '⚠️ 申请删除账户'}</button>
                <button id="logoutBtn" class="file-btn" style="margin-top: 10px;">退出登录</button>
            `;
            document.getElementById('updateAccountBtn')?.addEventListener('click', updateAccount);
            document.getElementById('deleteAccountBtn')?.addEventListener('click', requestDeleteAccount);
            document.getElementById('logoutBtn')?.addEventListener('click', logout);
            document.getElementById('saveEmailBtn')?.addEventListener('click', saveEmail);
            document.getElementById('requestPinCodeBtn')?.addEventListener('click', requestPinCode);
            loadUserStyle();  // async, doesn't block
        } else if (consentGiven && !isLoggedIn) {
            // Consent given but not logged in – show registration / login options
            accountContent.innerHTML = `
                <p>您已同意数据收集，但尚未创建账户。</p>
                <p>创建账户后可跨设备同步聊天记录和文件。</p>
                <form onsubmit="return false" style="margin:0;">
                    <input type="text" id="regUsernameModal" placeholder="用户名 (5-18字符)" autocomplete="username" style="width:100%; margin-bottom:8px;">
                    <select id="pinLengthModal" style="width:100%; margin-bottom:8px;">
                        <option value="4">4位</option>
                        <option value="6" selected>6位</option>
                    </select>
                    <input type="password" id="regPinModal" placeholder="PIN" autocomplete="new-password" style="width:100%; margin-bottom:8px;">
                    <input type="password" id="confirmPinModal" placeholder="确认PIN" autocomplete="new-password" style="width:100%; margin-bottom:8px;">
                    <button id="createAccountModalBtn" class="file-btn" style="margin-top: 10px;">创建账户</button>
                </form>
                <button id="loginBtn" class="file-btn" style="margin-top: 10px;">已有账户？登录</button>
            `;
            document.getElementById('createAccountModalBtn')?.addEventListener('click', createAccountFromModal);
            document.getElementById('loginBtn')?.addEventListener('click', showLoginModal);
        } else {
            // Anonymous (consent not given)
            accountContent.innerHTML = `
                <p>您当前以匿名身份使用，未同意数据收集。</p>
                <p>对话记录和文件仅保存在当前会话中，关闭浏览器后即丢失。</p>
                <p>同意数据收集并创建账户后，可跨设备同步数据。</p>
                <button id="agreeAndCreateBtn" class="file-btn" style="margin-top: 10px;">同意数据收集并创建账户</button>
                <button id="loginExistingBtn" class="file-btn" style="margin-top: 10px;">已有账户？登录</button>
            `;
            document.getElementById('agreeAndCreateBtn')?.addEventListener('click', showConsentAndAccountFlow);
            document.getElementById('loginExistingBtn')?.addEventListener('click', showLoginFlowForAnonymous);
        }
    }

    async function createAccountFromModal() {
        console.log("createAccountFromModal called");
        const username = document.getElementById('regUsernameModal').value.trim();
        const pin = document.getElementById('regPinModal').value;
        const confirmPin = document.getElementById('confirmPinModal').value;
        const pinLength = parseInt(document.getElementById('pinLengthModal').value);
        if (!username || !pin) { alert('请填写用户名和PIN'); return; }
        if (pin !== confirmPin) { alert('PIN不一致'); return; }
        if (pin.length !== pinLength || !/^\d+$/.test(pin)) { alert(`PIN必须是${pinLength}位数字`); return; }
        if (username.length < 5 || username.length > 18) { alert('用户名长度应为5-18个字符'); return; }
        try {
            const accountRes = await fetch('/create_account', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ username, pin, pin_length: pinLength })
            });
            const accountData = await accountRes.json();
            if (accountRes.ok) {
                showToast('账户创建成功', 'success', 3000);
                sessionStorage.setItem('username', username);
                location.reload();
            } else {
                alert(accountData.error || '创建失败');
            }
        } catch (err) {
            console.error(err);
            alert('网络错误，请重试');
        }
    }

    function showLoginModal() {
        console.log("showLoginModal called");
        // Close account modal first, then show clean login overlay
        accountModal.style.display = 'none';
        // Remove any existing consent overlays
        document.querySelectorAll('.consent-overlay').forEach(o => o.remove());
        const overlay = document.createElement('div');
        overlay.className = 'consent-overlay';
        overlay.innerHTML = `
            <div class="consent-card" style="max-width:400px;">
                <h3>登录</h3>
                <form onsubmit="return false" style="margin:0;">
                <input type="text" id="loginUsername" placeholder="用户名" autocomplete="username" style="width:100%;margin-bottom:8px;padding:8px;border-radius:8px;border:1px solid var(--card-border);">
                <input type="password" id="loginPin" placeholder="PIN" autocomplete="current-password" style="width:100%;margin-bottom:8px;padding:8px;border-radius:8px;border:1px solid var(--card-border);">
                <button id="doLoginBtn" class="consent-action-btn" style="margin-top:0;">登录</button>
                </form>
                <button id="cancelLoginBtn" style="margin-top:8px;background:none;border:none;color:var(--card-muted);cursor:pointer;">取消</button>
            </div>
        `;
        document.body.appendChild(overlay);
        document.getElementById('doLoginBtn').addEventListener('click', () => {
            const username = document.getElementById('loginUsername').value.trim();
            const pin = document.getElementById('loginPin').value;
            if (!username || !pin) { alert('请填写用户名和PIN'); return; }
            loginAndClose(overlay, username, pin);
        });
        document.getElementById('cancelLoginBtn').addEventListener('click', () => overlay.remove());
    }

    async function loginAndClose(overlay, username, pin) {
        try {
            const csrf = await getCsrfToken();
            const res = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf },
                credentials: 'include',
                body: JSON.stringify({ username, pin })
            });
            const data = await res.json();
            if (res.ok) {
                sessionStorage.setItem('username', username);
                sessionStorage.setItem('user_id', data.user_id);
                sessionStorage.setItem('just_logged_in', '1');
                sessionStorage.setItem('is_auditor', (data.is_auditor || data.is_admin) ? '1' : '0');
                if (data.is_admin) {
                    sessionStorage.setItem('isAdmin', 'true');
                }
                // Close overlay, then reload
                overlay.remove();
                document.querySelectorAll('.consent-overlay, .modal[style*=\"block\"]').forEach(o => { o.style.display = 'none'; try { o.remove(); } catch(e) {} });
                showToast('登录成功', 'success', 1500);
                setTimeout(() => location.reload(), 500);
            } else {
                alert(data.error || '登录失败');
            }
        } catch (err) {
            console.error(err);
            alert('网络错误，请重试');
        }
    }

    var _csrfToken = '';
    async function getCsrfToken() {
        if (_csrfToken) return _csrfToken;
        try {
            const r = await fetch('/get_csrf_token', { credentials: 'include' });
            const d = await r.json();
            _csrfToken = d.csrf_token || '';
        } catch(e) { _csrfToken = ''; }
        return _csrfToken;
    }

    async function login(username, pin) {
        console.log("login called", username);
        try {
            const csrf = await getCsrfToken();
            const res = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf },
                credentials: 'include',
                body: JSON.stringify({ username, pin })
            });
            const data = await res.json();
            if (res.ok) {
                sessionStorage.setItem('username', username);
                sessionStorage.setItem('user_id', data.user_id);
                sessionStorage.setItem('just_logged_in', '1');
                sessionStorage.setItem('is_auditor', (data.is_auditor || data.is_admin) ? '1' : '0');
                if (data.is_admin) {
                    sessionStorage.setItem('isAdmin', 'true');
                    sessionStorage.setItem('forceAdminTab', 'true');
                }
                location.reload();
            } else {
                alert(data.error || '登录失败');
            }
        } catch (err) {
            console.error(err);
            alert('网络错误，请重试');
        }
    }

    async function logout() {
        try {
            const res = await fetch('/logout', { method: 'POST', credentials: 'include' });
            if (res.ok) {
                sessionStorage.clear();
                location.href = '/';
            } else {
                console.error('Logout failed');
                sessionStorage.clear();
                location.href = '/';
            }
        } catch (err) {
            console.error('Logout error', err);
            sessionStorage.clear();
            location.href = '/';
        }
    }

    function showConsentAndAccountFlow() {
        console.log("showConsentAndAccountFlow called");
        const overlay = document.createElement('div');
        overlay.className = 'consent-overlay';
        overlay.innerHTML = `
            <div class="consent-card" style="max-width: 450px;">
                <h3>📋 数据收集与账户注册</h3>
                <p>请先同意数据收集，然后注册账户以继续使用。</p>
                <div class="consent-scrollbox" style="max-height:150px;">
                    <p>我们会匿名存储您的对话内容和反馈。请勿输入个人隐私信息。</p>
                </div>
                <label><input type="checkbox" id="tempConsentCheckbox"> 我已阅读并同意</label>
                <div id="tempAccountFields" style="display:none; margin-top:15px;">
                    <input type="text" id="tempUsername" placeholder="用户名 (5-18字符)" style="width:100%; margin-bottom:8px;">
                    <select id="tempPinLength" style="width:100%; margin-bottom:8px;">
                        <option value="4">4位</option>
                        <option value="6" selected>6位</option>
                    </select>
                    <input type="password" id="tempPin" placeholder="PIN" style="width:100%; margin-bottom:8px;">
                    <input type="password" id="tempConfirmPin" placeholder="确认PIN" style="width:100%; margin-bottom:8px;">
                    <button id="tempCreateBtn" style="background:#27ae60; color:white; border:none; border-radius:24px; padding:8px;">创建账户</button>
                </div>
                <button id="tempCloseBtn" style="margin-top:15px;">取消</button>
            </div>
        `;
        document.body.appendChild(overlay);
        const tempConsentCheckbox = overlay.querySelector('#tempConsentCheckbox');
        const tempAccountFields = overlay.querySelector('#tempAccountFields');
        const tempCreateBtn = overlay.querySelector('#tempCreateBtn');
        const tempCloseBtn = overlay.querySelector('#tempCloseBtn');
        tempConsentCheckbox.onchange = () => {
            tempAccountFields.style.display = tempConsentCheckbox.checked ? 'block' : 'none';
        };
        tempCreateBtn.onclick = async () => {
            const username = overlay.querySelector('#tempUsername').value.trim();
            const pin = overlay.querySelector('#tempPin').value;
            const confirmPin = overlay.querySelector('#tempConfirmPin').value;
            const pinLength = parseInt(overlay.querySelector('#tempPinLength').value);
            if (!username || !pin) { alert('请填写用户名和PIN'); return; }
            if (pin !== confirmPin) { alert('PIN不一致'); return; }
            if (pin.length !== pinLength || !/^\d+$/.test(pin)) { alert(`PIN必须是${pinLength}位数字`); return; }
            if (username.length < 5 || username.length > 18) { alert('用户名长度应为5-18个字符'); return; }
            // Registration auto-sets consent — no separate /consent call needed
            const accountRes = await fetch('/create_account', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ username, pin, pin_length: pinLength })
            });
            const accountData = await accountRes.json();
            if (accountRes.ok) {
                localStorage.setItem('consent_given', '1');
                showToast('账户创建成功', 'success', 3000);
                sessionStorage.setItem('username', username);
                location.reload();
            } else alert(accountData.error || '创建失败');
        };
        tempCloseBtn.onclick = () => overlay.remove();
    }

    function showLoginFlowForAnonymous() {
        console.log("showLoginFlowForAnonymous called");
        const overlay = document.createElement('div');
        overlay.className = 'consent-overlay';
        overlay.innerHTML = `
            <div class="consent-card" style="max-width: 450px;">
                <h3>📋 登录需要同意数据收集</h3>
                <p>登录账户前，请先同意数据收集政策。</p>
                <div class="consent-scrollbox" style="max-height:150px;">
                    <p>我们会根据隐私政策存储您的对话记录和文件，以便跨设备同步。</p>
                </div>
                <label><input type="checkbox" id="tempConsentCheckbox"> 我已阅读并同意隐私政策</label>
                <div id="tempLoginFields" style="display:none; margin-top:15px;">
                    <form onsubmit="return false" style="margin:0;">
                    <input type="text" id="tempLoginUsername" placeholder="用户名" autocomplete="username" style="width:100%; margin-bottom:8px;">
                    <input type="password" id="tempLoginPin" placeholder="PIN" autocomplete="current-password" style="width:100%; margin-bottom:8px;">
                    <button id="tempLoginBtn" style="background:#27ae60; color:white; border:none; border-radius:24px; padding:8px;">登录</button>
                    </form>
                </div>
                <button id="tempCloseBtn" style="margin-top:15px;">取消</button>
            </div>
        `;
        document.body.appendChild(overlay);
        const tempConsentCheckbox = overlay.querySelector('#tempConsentCheckbox');
        const tempLoginFields = overlay.querySelector('#tempLoginFields');
        const tempLoginBtn = overlay.querySelector('#tempLoginBtn');
        const tempCloseBtn = overlay.querySelector('#tempCloseBtn');
        tempConsentCheckbox.onchange = () => {
            tempLoginFields.style.display = tempConsentCheckbox.checked ? 'block' : 'none';
        };
        tempLoginBtn.onclick = async () => {
            const username = overlay.querySelector('#tempLoginUsername').value.trim();
            const pin = overlay.querySelector('#tempLoginPin').value;
            if (!username || !pin) { alert('请填写用户名和PIN'); return; }
            const csrf = await getCsrfToken();
            localStorage.setItem('consent_given', '1');
            const loginRes = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf },
                credentials: 'include',
                body: JSON.stringify({ username, pin })
            });
            const loginData = await loginRes.json();
            if (loginRes.ok) {
                showToast('登录成功', 'success', 3000);
                sessionStorage.setItem('username', username);
                location.reload();
            } else alert(loginData.error || '登录失败');
        };
        tempCloseBtn.onclick = () => overlay.remove();
    }

    async function updateAccount() {
        const newUsername = document.getElementById('newUsername')?.value.trim();
        const newPin = document.getElementById('newPin')?.value;
        const verifyCode = document.getElementById('pinVerifyCode')?.value.trim();
        const confirmPin = document.getElementById('confirmNewPin')?.value;
        const currentPin = document.getElementById('currentPin')?.value;
        if (!currentPin) { alert('请输入当前PIN'); return; }
        if (newPin && newPin !== confirmPin) { alert('新PIN与确认PIN不一致'); return; }
        try {
            const res = await fetch('/update_account', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                    new_username: newUsername,
                    new_pin: newPin,
                    pin_length: parseInt(document.getElementById('newPinLength')?.value||6, 10),
                    current_pin: currentPin,
                    verify_code: newPin ? verifyCode : null
                })
            });
            const data = await res.json();
            if (res.ok) {
                showToast('账户信息已更新', 'success', 3000);
                if (newUsername) sessionStorage.setItem('username', newUsername);
                location.reload();
            } else alert(data.error || '更新失败');
        } catch (err) {
            console.error(err);
            alert('网络错误，请重试');
        }
    }

    async function saveEmail() {
        const email = document.getElementById('userEmail')?.value.trim();
        if (!email || !email.includes('@')) { alert('请输入有效邮箱'); return; }
        const res = await fetch('/set_email', { method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include', body:JSON.stringify({email}) });
        if (res.ok) showToast('邮箱已保存', 'success');
        else { const d = await res.json(); alert(d.error || '保存失败'); }
    }

    async function requestPinCode() {
        const res = await fetch('/request_pin_change_code', { method:'POST', credentials:'include' });
        const d = await res.json();
        if (res.ok) showToast(d.hint || '验证码已发送', 'info', 5000);
        else alert(d.error || '请求失败');
    }

    async function requestDeleteAccount() {
        if (sessionStorage.getItem('deletion_pending') === '1') {
            showConfirmDeleteModal();
            return;
        }
        const res = await fetch('/request_delete_account', { method:'POST', credentials:'include' });
        const d = await res.json();
        if (!res.ok) { alert(d.error || '加载失败'); return; }
        const inventory = d.inventory || [];
        const sessionCnt = d.session_count || 0;
        // Build selection modal
        const modal = createQuickModal('选择保留数据');
        let html = `<p style="font-size:.78rem;color:var(--card-muted);margin-bottom:4px;">选择要保留给公司的数据（其余将永久删除）。项目文件默认保留。</p>
            <p style="font-size:.7rem;margin-bottom:8px;">💬 ${sessionCnt}个聊天会话（将永久删除）</p>
            <div style="max-height:300px;overflow-y:auto;margin-bottom:8px;">
            <table style="width:100%;font-size:.73rem;border-collapse:collapse;">`;
        for (const item of inventory) {
            const checked = item.choosable ? 'checked' : 'checked disabled';
            const disabled = item.choosable ? '' : 'disabled';
            html += `<tr style="border-bottom:1px solid var(--card-border);">
                <td style="padding:3px 4px;"><input type="checkbox" class="del-keep-cb" data-type="${item.type}" data-id="${item.id}" ${checked} ${disabled}></td>
                <td>${escapeHtml(item.name)}</td>
                <td style="color:var(--card-muted);font-size:.65rem;">${item.size_kb > 0 ? item.size_kb+'KB' : ''}</td>
                <td style="font-size:.6rem;color:var(--card-muted);">${item.note||''}</td>
            </tr>`;
        }
        html += `</table></div>
            <div style="display:flex;gap:8px;align-items:center;">
            <button id="delSelectAll" class="file-btn" style="font-size:.7rem;">全选</button>
            <button id="delDeselectAll" class="file-btn" style="font-size:.7rem;">全不选</button>
            <button id="delSubmitBtn" class="file-btn" style="background:#e74c3c;color:white;margin-left:auto;font-size:.78rem;">⚠️ 提交删除申请</button>
            <span id="delStatus" style="font-size:.7rem;"></span></div>`;
        modal.innerHTML(html);
        modal.querySelector('#delSelectAll').onclick = () => modal.querySelectorAll('.del-keep-cb:not([disabled])').forEach(c => c.checked = true);
        modal.querySelector('#delDeselectAll').onclick = () => modal.querySelectorAll('.del-keep-cb:not([disabled])').forEach(c => c.checked = false);
        modal.querySelector('#delSubmitBtn').onclick = async () => {
            const keep = [];
            modal.querySelectorAll('.del-keep-cb:checked').forEach(c => keep.push({type: c.dataset.type, id: parseInt(c.dataset.id)}));
            if (!confirm(`确定提交删除申请吗？\\n保留 ${keep.length} 项数据给公司，其余 ${inventory.length - keep.length} 项将永久删除。`)) return;
            const r = await fetch('/submit_delete_choices', { method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include', body:JSON.stringify({keep_ids:keep}) });
            const rd = await r.json();
            if (r.ok) { showToast(rd.message || '已提交', 'info', 6000); modal.remove(); sessionStorage.setItem('deletion_pending', '1'); showConfirmDeleteModal(); }
            else alert(rd.error || '提交失败');
        };
    }

    function showConfirmDeleteModal() {
        const modal = createQuickModal('确认删除账户');
        const lastRequest = sessionStorage.getItem('deletion_last_request') || '';
        if (lastRequest) {
            const elapsed = (Date.now() - parseInt(lastRequest)) / 1000;
            if (elapsed < 300) {
                modal.innerHTML(`<p>请等待 ${Math.ceil(300 - elapsed)} 秒后重试。</p>`);
                return;
            }
        }
        modal.innerHTML(`
            <p style="font-size:.78rem;color:var(--card-muted);margin-bottom:8px;">管理员已发送验证码到您的邮箱，请在此输入。</p>
            <label>验证码 (4位):</label>
            <input type="text" id="delConfirmCode" maxlength="4" placeholder="管理员发送的验证码" style="width:100%;margin-bottom:8px;">
            <label>PIN:</label>
            <input type="password" id="delConfirmPin" placeholder="您的PIN" style="width:100%;margin-bottom:8px;">
            <div style="display:flex;gap:8px;">
                <button id="delConfirmBtn" class="file-btn" style="background:#e74c3c;color:white;">确认删除</button>
                <button id="delCancelReqBtn" class="file-btn">取消删除请求</button>
            </div>
            <span id="delConfirmStatus" style="font-size:.7rem;color:var(--card-muted);"></span>
        `);
        modal.querySelector('#delConfirmBtn').onclick = async () => {
            const code = modal.querySelector('#delConfirmCode').value.trim();
            const pin = modal.querySelector('#delConfirmPin').value.trim();
            if (!code || !pin) { modal.querySelector('#delConfirmStatus').textContent = '请填写验证码和PIN'; return; }
            const btn = modal.querySelector('#delConfirmBtn');
            btn.disabled = true; btn.textContent = '...';
            try {
                const res = await fetch('/confirm_delete_account', { method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include', body:JSON.stringify({code, pin}) });
                const d = await res.json();
                if (res.ok) {
                    sessionStorage.removeItem('deletion_pending');
                    sessionStorage.setItem('deletion_last_request', Date.now().toString());
                    showToast('账户已删除', 'success', 3000);
                    modal.remove();
                    setTimeout(() => { location.href = '/'; }, 1000);
                } else {
                    modal.querySelector('#delConfirmStatus').textContent = d.error || '确认失败';
                }
            } catch (_) {
                modal.querySelector('#delConfirmStatus').textContent = '网络错误';
            }
            btn.disabled = false; btn.textContent = '确认删除';
        };
        modal.querySelector('#delCancelReqBtn').onclick = async () => {
            if (!confirm('取消删除请求？')) return;
            sessionStorage.removeItem('deletion_pending');
            sessionStorage.setItem('deletion_last_request', Date.now().toString());
            showToast('删除请求已取消', 'info', 3000);
            modal.remove();
        };
    }

    var accountSettingsBtn = document.getElementById('accountSettingsBtn');
    if (accountSettingsBtn) {
        accountSettingsBtn.onclick = () => {
            console.log("Account settings button clicked");
            accountModal.style.display = 'block';
            loadAccountModal();
        };
    }
    if (closeAccountModal) {
        closeAccountModal.onclick = () => { accountModal.style.display = 'none'; };
    }

