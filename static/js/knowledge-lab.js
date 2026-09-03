/* Knowledge Lab module - extracted from app.js (W3) */
/* Dependencies: showToast, escapeHtml, switchToPanel, switchSidebarPane,
                 saveActiveTab, fetch, currentProjectId (kernel globals) */

let currentProjectName = '';

    // ======================== Knowledge Lab Tab ========================
    if (knowledgeLabTab) {
        knowledgeLabTab.onclick = async () => {
            stopRealtimePoll();
            saveActiveTab('knowledge');
            showSubTabBar('knowledge');
            resetSubTabs('knowledgeSubTabs');
            switchToPanel('knowledgeLabPanel');
            switchSidebarPane('knowledge');
            toggleQuickLinksButton(false);
            syncActiveTabWithView();
            // Fetch both lists ONCE and share
            const [labRes, coRes] = await Promise.all([
                fetch('/knowledge_lab/list', { credentials: 'include' }),
                fetch('/company_kb/list', { credentials: 'include' })
            ]);
            const labData = await labRes.json();
            const coData = await coRes.json();
            loadSidebarKnowledge(labData, coData);
            loadKnowledgeLabFiles(labData);
            loadCompanyKnowledgeBase(coData);
            loadCompanyCategories();
            loadSkillOverview(labData, coData);
        };
    }

    // --- Wire up sidebar button handlers ---
    document.addEventListener('DOMContentLoaded', () => {
        // Projects sidebar
        const scpBtn = document.getElementById('sidebarCreateProjectBtn');
        const smfBtn = document.getElementById('sidebarMyFilesBtn');
        if (scpBtn) scpBtn.onclick = () => { document.getElementById('createProjectBtn')?.click(); };
        if (smfBtn) smfBtn.onclick = () => { document.getElementById('myFilesBtn')?.click(); };

        // Admin extras visibility and tool buttons (all buttons now in HTML)
        const adminExtras = document.getElementById('sidebarAdminExtras');
        if (adminExtras && sessionStorage.getItem('isAdmin') === 'true') {
            adminExtras.style.display = '';
        }
        // Wire admin tool buttons
        const auditLogBtn = document.getElementById('sidebarAuditLogBtn');
        setTimeout(() => {
            const cacheBtn = document.getElementById('sidebarClearCacheBtn');
            const cleanupBtn = document.getElementById('sidebarCleanupNowBtn');
            const promptBtn = document.getElementById('sidebarEditPromptBtn');
            const workReportBtn = document.getElementById('sidebarWorkReportBtn');
            const ragStatsBtn = document.getElementById('sidebarRagStatsBtn');
            const ragRebuildBtn = document.getElementById('sidebarRagRebuildBtn');
            const trainingStatsBtn = document.getElementById('sidebarTrainingStatsBtn');
            const trainingExportBtn = document.getElementById('sidebarTrainingExportBtn');
            const searchCacheBtn = document.getElementById('sidebarSearchCacheBtn');

            if (cacheBtn) cacheBtn.onclick = async () => {
                try {
                    const res = await fetch('/admin/clear_file_cache', { method: 'POST', credentials: 'include' });
                    if (res.ok) showToast('缓存已清除', 'success');
                    else showToast('清除失败', 'error');
                } catch(_) { showToast('网络错误', 'error'); }
            };
                if (cleanupBtn) cleanupBtn.onclick = async () => {
                    if (!confirm('确定要手动清理所有过期的会话、文件和匿名临时数据吗？')) return;
                    try {
                        const res = await fetch('/cleanup_now', { method: 'POST', credentials: 'include' });
                        if (res.ok) showToast('清理完成', 'success');
                        else showToast('清理失败', 'error');
                    } catch(_) { showToast('网络错误', 'error'); }
                };
                if (promptBtn) promptBtn.onclick = async () => {
                    const modal = createQuickModal('系统提示词');
                    let currentPrompt = '';
                    try {
                        const r = await fetch('/admin/system_prompt', { credentials: 'include' });
                        const d = await r.json();
                        currentPrompt = d.prompt || '';
                    } catch(_) { currentPrompt = '(加载失败)'; }
                    modal.innerHTML(`<p style="font-size:.7rem;color:var(--card-muted);margin-bottom:8px;">编辑AI助手的系统提示词。修改后立即生效，已持久化到磁盘。</p>
                        <textarea id="promptEditor" style="width:100%;height:300px;font-family:monospace;font-size:.78rem;padding:8px;border-radius:6px;border:1px solid var(--card-border);resize:vertical;margin-bottom:8px;">${escapeHtml(currentPrompt)}</textarea>
                        <div style="display:flex;gap:8px;">
                            <button id="savePromptBtn" class="file-btn" style="background:#16a34a;color:white;padding:6px 16px;">💾 保存</button>
                            <button id="resetPromptBtn" class="file-btn" style="background:#e2e8f0;color:#334155;padding:6px 16px;">🔄 恢复默认</button>
                            <span id="promptStatus" style="font-size:.75rem;align-self:center;"></span>
                        </div>`);
                    const saveBtn = modal.querySelector('#savePromptBtn');
                    const resetBtn = modal.querySelector('#resetPromptBtn');
                    const statusEl = modal.querySelector('#promptStatus');
                    if (saveBtn) saveBtn.onclick = async () => {
                        const txt = modal.querySelector('#promptEditor').value.trim();
                        if (!txt) { statusEl.textContent = '提示词不能为空'; return; }
                        saveBtn.disabled = true; statusEl.textContent = '保存中...';
                        try {
                            const r = await fetch('/admin/system_prompt', { method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include', body:JSON.stringify({prompt:txt}) });
                            const d = await r.json();
                            if (r.ok) { statusEl.textContent = '✅ '+(d.message||'已保存'); showToast('系统提示词已更新', 'success'); }
                            else { statusEl.textContent = '❌ '+(d.error||'保存失败'); }
                        } catch(_) { statusEl.textContent = '❌ 网络错误'; }
                        saveBtn.disabled = false;
                    };
                    if (resetBtn) resetBtn.onclick = () => { modal.querySelector('#promptEditor').value = currentPrompt; statusEl.textContent = '已恢复为最后保存的版本'; };
                };
                if (workReportBtn) workReportBtn.onclick = async () => {
                    const modal = createQuickModal('工作报告');
                    let users = [];
                    try {
                        const r = await fetch('/admin/user_emails', { credentials: 'include' });
                        const d = await r.json();
                        users = (d.users || []).filter(u => u.role !== 'admin');
                    } catch(_) {}
                    modal.innerHTML(`<h3 style="margin-bottom:4px;">📊 生成AI工作报告</h3>
                        <p style="font-size:.7rem;color:var(--card-muted);margin-bottom:4px;">基于AI对话记录（含已归档/已删除），生成正式工作报告。</p>
                        <div style="margin-bottom:8px;font-size:.75rem;">
                            <label style="cursor:pointer;margin-right:12px;"><input type="checkbox" id="wrAllUsers" checked onchange="document.querySelectorAll('.wr-user-cb').forEach(c=>c.checked=this.checked)"> 全选</label>
                            <span id="wrSelectedCount" style="color:var(--card-muted);"></span>
                        </div>
                        <div id="wrUserList" style="max-height:120px;overflow-y:auto;margin-bottom:12px;font-size:.75rem;display:flex;flex-wrap:wrap;gap:4px;">
                            ${users.map(u => `<label style="cursor:pointer;background:var(--card-bg);border:1px solid var(--card-border);border-radius:4px;padding:2px 8px;"><input type="checkbox" class="wr-user-cb" value="${escapeHtml(u.user_id)}" checked> ${escapeHtml(u.username)}</label>`).join('')}
                        </div>
                        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">
                            <button data-period="daily" class="report-btn" style="flex:1;min-width:80px;padding:10px;border:2px solid var(--card-border);border-radius:8px;background:var(--card-bg);cursor:pointer;font-size:.85rem;text-align:center;">📅 日报<br><small style="color:var(--card-muted);">~400字</small></button>
                            <button data-period="weekly" class="report-btn" style="flex:1;min-width:80px;padding:10px;border:2px solid #2563eb;border-radius:8px;background:#eff6ff;cursor:pointer;font-size:.85rem;text-align:center;">📆 周报<br><small style="color:var(--card-muted);">~800字</small></button>
                            <button data-period="monthly" class="report-btn" style="flex:1;min-width:80px;padding:10px;border:2px solid var(--card-border);border-radius:8px;background:var(--card-bg);cursor:pointer;font-size:.85rem;text-align:center;">📊 月报<br><small style="color:var(--card-muted);">~1200字</small></button>
                            <button data-period="annual" class="report-btn" style="flex:1;min-width:80px;padding:10px;border:2px solid var(--card-border);border-radius:8px;background:var(--card-bg);cursor:pointer;font-size:.85rem;text-align:center;">📈 年报<br><small style="color:var(--card-muted);">~2000字</small></button>
                        </div>
                        <div id="workReportResult" style="font-size:.8rem;"></div>`);
                    const updateCount = () => {
                        const cbs = modal.querySelectorAll('.wr-user-cb:checked');
                        const countEl = modal.querySelector('#wrSelectedCount');
                        if (countEl) countEl.textContent = `已选 ${cbs.length}/${users.length} 人`;
                    };
                    updateCount();
                    modal.querySelectorAll('.wr-user-cb').forEach(cb => cb.onchange = updateCount);
                    modal.querySelectorAll('.report-btn').forEach(btn => {
                        btn.onclick = async () => {
                            const period = btn.dataset.period;
                            const selectedIds = Array.from(modal.querySelectorAll('.wr-user-cb:checked')).map(c => c.value);
                            if (!selectedIds.length) { alert('请至少选择一位用户'); return; }
                            modal.querySelectorAll('.report-btn').forEach(b => { b.disabled = true; b.style.opacity = '0.5'; });
                            const resultDiv = modal.querySelector('#workReportResult');
                            resultDiv.innerHTML = '<p>' + _icon('hourglass_empty') + ' AI正在汇总数据并撰写报告，请稍候（约30-60秒）...</p>';
                            try {
                                const r = await fetch('/admin/generate_work_report', { method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include', body:JSON.stringify({period, user_ids:selectedIds}) });
                                const d = await r.json();
                                if (r.ok) {
                                    resultDiv.innerHTML = `<div style="background:#dcfce7;border:1px solid #bbf7d0;border-radius:8px;padding:12px;">
                                        <p style="color:#16a34a;font-weight:bold;">✅ 报告已生成</p>
                                        <p>📄 ${escapeHtml(d.filename)} · ${d.size_kb}KB</p>
                                        <a href="${d.download_url}" download class="file-btn" style="display:inline-block;margin-top:6px;background:#16a34a;color:white;padding:6px 16px;text-decoration:none;">📥 立即下载</a>
                                    </div>`;
                                } else {
                                    resultDiv.innerHTML = `<p style="color:#dc2626;">❌ ${escapeHtml(d.error||'生成失败')}</p>`;
                                }
                            } catch(_) { resultDiv.innerHTML = '<p style="color:#dc2626;">' + _icon('cancel') + ' 网络错误</p>'; }
                            modal.querySelectorAll('.report-btn').forEach(b => { b.disabled = false; b.style.opacity = '1'; });
                        };
                    });
                };
            if (ragStatsBtn) ragStatsBtn.onclick = async () => {
                try {
                    ragStatsBtn.disabled = true; ragStatsBtn.textContent = '⏳ ...';
                    const res = await fetch('/admin/rag_stats', { credentials: 'include' });
                    const d = await res.json();
                    const s = d.stats || {};
                    showToast(`RAG: 个人${s.knowledge_lab||0} · 公司${s.company_kb||0} · 合计${s.total||0} chunks`, 'info', 4000);
                } catch(_) { showToast('加载失败', 'error'); }
                ragStatsBtn.disabled = false; ragStatsBtn.textContent = '🔍 索引统计';
            };
            if (ragRebuildBtn) ragRebuildBtn.onclick = async () => {
                if (!confirm('重建RAG索引将为所有文件重新生成嵌入向量。\n\n此操作可能需要数分钟，期间不影响正常使用。确定继续？')) return;
                ragRebuildBtn.disabled = true; ragRebuildBtn.textContent = '⏳ 重建中...';
                try {
                    const res = await fetch('/admin/rag_rebuild', { method: 'POST', credentials: 'include' });
                    const d = await res.json();
                    if (res.ok) showToast(`RAG重建完成: ${d.indexed} 个文件已索引`, 'success', 4000);
                    else showToast('重建失败: ' + (d.error || ''), 'error', 4000);
                } catch(_) { showToast('网络错误', 'error'); }
                ragRebuildBtn.disabled = false; ragRebuildBtn.textContent = '🔨 重建全部索引';
            };
            if (trainingStatsBtn) trainingStatsBtn.onclick = async () => {
                try {
                    const [sr, hr] = await Promise.all([
                        fetch('/admin/training_stats', { credentials: 'include' }),
                        fetch('/admin/training_export_history', { credentials: 'include' })
                    ]);
                    const sd = await sr.json();
                    const hd = await hr.json();
                    const s = sd.stats || {};
                    const h = (hd.history) || {};
                    const pending = h.pending_new || 0;
                    showToast(`训练数据: ${s.sessions||0}个会话 · ${s.interactions||0}条交互 · ${s.rated||0}已评分 · ${pending}条待导出`, 'info', 5000);
                } catch(_) { showToast('加载失败', 'error'); }
            };
            if (trainingExportBtn) trainingExportBtn.onclick = async () => {
                trainingExportBtn.disabled = true; trainingExportBtn.textContent = '⏳ ...';
                try {
                    const res = await fetch('/admin/training_export', {
                        method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                        body:JSON.stringify({mode:'incremental'})
                    });
                    const d = await res.json();
                    if (res.ok) showToast(d.message || 'Exported', 'success', 4000);
                    else showToast(d.error || 'Export failed', 'error');
                } catch(_) { showToast('网络错误', 'error'); }
                trainingExportBtn.disabled = false; trainingExportBtn.textContent = '📥 增量导出JSONL';
            };
            const sysCleanupBtn = document.getElementById('sidebarSystemCleanupBtn');
            if (sysCleanupBtn) sysCleanupBtn.onclick = async () => {
                if (!confirm('执行系统清理？\n将清理过期会话、临时文件、内存残留、并审计文件泄漏。')) return;
                sysCleanupBtn.disabled = true; sysCleanupBtn.textContent = '⏳ 清理中...';
                try {
                    const r = await fetch('/admin/system_cleanup', {method:'POST', credentials:'include'});
                    const d = await r.json();
                    if (r.ok) {
                        const parts = Object.entries(d.results||{}).map(([k,v])=>`${k}: ${v}`);
                        showToast('✅ 清理完成: ' + parts.join(' | '), 'success', 6000);
                    } else {
                        showToast('❌ ' + (d.error||'清理失败'), 'error');
                    }
                } catch(_) { showToast('网络错误', 'error'); }
                sysCleanupBtn.disabled = false; sysCleanupBtn.textContent = '🧹 一键系统清理';
            };
            const clearAllBtn = document.getElementById('sidebarClearAllDataBtn');
            if (clearAllBtn) clearAllBtn.onclick = async () => {
                if (!confirm('⚠️ 确定清空全部数据？\n\n这将删除：\n- 所有上传的文件和内容\n- 所有生成的AI技能\n- 项目文件内容\n- AI记忆和缓存\n- 磁盘上的文件\n\n保留：用户账号、项目结构、聊天记录')) return;
                if (!confirm('再次确认：此操作不可撤销！')) return;
                clearAllBtn.disabled = true; clearAllBtn.textContent = '⏳ 清空中...';
                try {
                    const r = await fetch('/admin/clear_all_data', {method:'POST', credentials:'include'});
                    const d = await r.json();
                    if (r.ok) showToast('✅ 已清空: ' + JSON.stringify(d.results).substring(0,200), 'success', 8000);
                    else showToast('❌ '+(d.error||'失败'), 'error');
                } catch(_) { showToast('网络错误', 'error'); }
                clearAllBtn.disabled = false; clearAllBtn.textContent = '⚠️ 清空全部数据';
            };
            if (searchCacheBtn) searchCacheBtn.onclick = async () => {
                // Fetch current config
                let cfg = { ttl_hours: 72, entries: 0 };
                try {
                    const r = await fetch('/admin/search_cache_config', { credentials: 'include' });
                    const d = await r.json();
                    if (d.config) cfg = d.config;
                } catch (_) {}
                // Build modal
                const m = createQuickModal('搜索缓存配置');
                m.innerHTML(`
                    <h3 style="margin:0 0 16px;">🔍 Bocha 搜索缓存</h3>
                    <div style="margin-bottom:12px;font-size:0.85rem;color:var(--card-muted);">
                        当前缓存条目: <b>${cfg.entries || 0}</b> &nbsp;|&nbsp; 当前 TTL: <b id="cacheTtlDisplay">${cfg.ttl_hours || 0}</b> 小时
                    </div>
                    <label style="display:block;margin-bottom:6px;font-weight:600;">缓存 TTL（小时）</label>
                    <div style="display:flex;gap:8px;margin-bottom:12px;">
                        <input id="searchCacheTtlInput" type="number" value="${cfg.ttl_hours || 72}" min="0" step="0.5" style="flex:1;padding:8px;border-radius:6px;border:1px solid var(--card-border);">
                        <button id="searchCacheSaveBtn" class="file-btn" style="background:#3b82f6;color:white;border-color:#2563eb;white-space:nowrap;">💾 保存</button>
                    </div>
                    <p style="font-size:0.7rem;color:var(--card-muted);margin-bottom:12px;">设为 0 = 禁用缓存。推荐 24~168 小时。</p>
                    <hr style="margin:12px 0;">
                    <button id="searchCacheClearBtn" class="file-btn" style="width:100%;background:#ef4444;color:white;border-color:#dc2626;">🗑️ 清除全部缓存</button>
                    <div id="searchCacheMsg" style="margin-top:8px;font-size:0.8rem;"></div>
                `);
                const msgEl = m.getElementById('searchCacheMsg');
                const ttlInput = m.getElementById('searchCacheTtlInput');
                const ttlDisplay = m.getElementById('cacheTtlDisplay');
                m.getElementById('searchCacheSaveBtn').onclick = async () => {
                    const hrs = parseFloat(ttlInput.value);
                    if (isNaN(hrs) || hrs < 0) { msgEl.innerHTML = '<span style="color:#ef4444;">请输入有效数字</span>'; return; }
                    try {
                        const r = await fetch('/admin/search_cache_config', {
                            method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                            body:JSON.stringify({action:'set_ttl', ttl_hours:hrs})
                        });
                        const d = await r.json();
                        if (r.ok) { ttlDisplay.textContent = hrs; msgEl.innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' '+d.message+'</span>'; }
                        else msgEl.innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' '+(d.error||'失败')+'</span>';
                    } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
                };
                m.getElementById('searchCacheClearBtn').onclick = async () => {
                    if (!confirm('确定清除全部搜索缓存？')) return;
                    try {
                        const r = await fetch('/admin/search_cache_config', {
                            method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                            body:JSON.stringify({action:'clear'})
                        });
                        const d = await r.json();
                        if (r.ok) msgEl.innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' '+d.message+'</span>';
                        else msgEl.innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' '+(d.error||'失败')+'</span>';
                    } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
                };
            };
        }, 200);

        // Helper: create a full-featured modal
        function createQuickModal(title) {
            const overlay = document.createElement('div');
            overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.5);z-index:10000;display:flex;align-items:center;justify-content:center;';
            const card = document.createElement('div');
            card.className = 'quick-modal-card';
            card.style.cssText = 'background:var(--card-bg,white);border-radius:12px;padding:24px;max-width:750px;width:90%;max-height:85vh;overflow-y:auto;position:relative;box-shadow:0 8px 32px rgba(0,0,0,.2);';
            const closeBtn = document.createElement('span');
            closeBtn.innerHTML = '&times;';
            closeBtn.style.cssText = 'position:absolute;top:10px;right:16px;cursor:pointer;font-size:22px;color:var(--card-muted);z-index:1;';
            closeBtn.onclick = () => overlay.remove();
            card.appendChild(closeBtn);
            overlay.appendChild(card);
            document.body.appendChild(overlay);
            // ESC key to close
            const escHandler = (e) => { if (e.key === 'Escape') { overlay.remove(); document.removeEventListener('keydown', escHandler); } };
            document.addEventListener('keydown', escHandler);
            overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
            const api = {
                overlay, card, closeBtn,
                close: () => overlay.remove(),
                querySelector: (s) => card.querySelector(s) || card,
                getElementById: (id) => card.querySelector('#'+id),
                innerHTML: (h) => { card.innerHTML = ''; card.appendChild(closeBtn); card.insertAdjacentHTML('beforeend', h); },
            };
            api.querySelector('.quick-modal-card');
            return api;
        }

        // Audit log handler (defined here so createQuickModal is in scope)
        if (auditLogBtn) auditLogBtn.onclick = () => {
            const modal = createQuickModal('审计日志');
            modal.innerHTML('<h3 style="margin-bottom:4px;">📊 管理员操作审计</h3>' +
                '<p style="font-size:.7rem;color:var(--card-muted);margin-bottom:4px;">记录全部管理操作，不可修改不可删除，仅可查看和导出</p>' +
                '<div style="display:flex;gap:4px;margin-bottom:4px;">' +
                '<button id="auditExportCsv" style="background:var(--card-bg);border:1px solid var(--card-border);border-radius:4px;padding:3px 10px;font-size:.7rem;cursor:pointer;">📥 导出CSV</button>' +
                '<button id="auditExportJson" style="background:var(--card-bg);border:1px solid var(--card-border);border-radius:4px;padding:3px 10px;font-size:.7rem;cursor:pointer;">📥 导出JSON</button>' +
                '<button id="auditAddNote" style="background:#dbeafe;border:1px solid #93c5fd;border-radius:4px;padding:3px 10px;font-size:.7rem;cursor:pointer;color:#1e40af;">📝 添加运维备注</button>' +
                '</div>' +
                '<div style="display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap;">' +
                '<input id="auditSearch" placeholder="🔍 搜索操作/表名/操作人..." style="flex:1;min-width:180px;padding:6px;border-radius:6px;border:1px solid var(--card-border);font-size:.78rem;">' +
                '<select id="auditActionFilter" style="padding:6px;border-radius:6px;border:1px solid var(--card-border);font-size:.78rem;">' +
                '<option value="">全部操作</option><option value="UPDATE">UPDATE</option><option value="DELETE">DELETE</option></select>' +
                '<select id="auditResultFilter" style="padding:6px;border-radius:6px;border:1px solid var(--card-border);font-size:.78rem;">' +
                '<option value="">全部结果</option><option value="true">成功</option><option value="false">失败</option></select>' +
                '</div>' +
                '<div id="auditLogContent">加载中...</div>' +
                '<div id="auditPagination" style="display:flex;gap:8px;justify-content:center;margin-top:12px;align-items:center;"></div>');
            let currentPage = 1, totalPages = 1, searchQuery = '', actionFilter = '', resultFilter = '';

            function buildDetails(l) {
                const parts = [];
                if (l.admin_username) parts.push(`👤 ${escapeHtml(l.admin_username)}`);
                if (l.table_name) parts.push(`📋 ${escapeHtml(l.table_name)}`);
                if (l.row_id) parts.push(`#${escapeHtml(String(l.row_id))}`);
                if (l.column_name) parts.push(`✏️ ${escapeHtml(l.column_name)}`);
                if (l.old_value != null && l.old_value !== '') {
                    const old = String(l.old_value).substring(0, 60);
                    const nu = l.new_value != null ? String(l.new_value).substring(0, 60) : '—';
                    parts.push(`${escapeHtml(old)} → ${escapeHtml(nu)}`);
                }
                if (l.ip_address) parts.push(`🌐 ${escapeHtml(l.ip_address)}`);
                if (l.error_message) parts.push(`❌ ${escapeHtml(l.error_message).substring(0, 60)}`);
                return parts.join(' · ') || '—';
            }

            async function loadPage(page) {
                currentPage = page;
                const container = modal.getElementById('auditLogContent');
                container.innerHTML = '加载中...';
                try {
                    const params = new URLSearchParams({ page });
                    if (searchQuery) params.set('search', searchQuery);
                    if (actionFilter) params.set('action', actionFilter);
                    if (resultFilter) params.set('success', resultFilter);
                    const res = await fetch(`/admin/audit_log?${params}`, { credentials: 'include' });
                    const data = await res.json();
                    const logs = data.logs || [];
                    totalPages = Math.ceil(data.total / 50) || 1;
                    if (!logs.length) {
                        container.innerHTML = '<p style="color:var(--card-muted);text-align:center;padding:30px;">📭 暂无匹配记录</p>';
                    } else {
                        container.innerHTML =
                            '<table style="width:100%;font-size:.76rem;border-collapse:collapse;">' +
                            '<thead><tr style="background:var(--card-bg);position:sticky;top:0;z-index:1;">' +
                            '<th style="padding:6px 8px;border-bottom:2px solid var(--card-border);text-align:left;white-space:nowrap;">时间</th>' +
                            '<th style="padding:6px 8px;border-bottom:2px solid var(--card-border);text-align:left;white-space:nowrap;">操作</th>' +
                            '<th style="padding:6px 8px;border-bottom:2px solid var(--card-border);text-align:left;white-space:nowrap;">结果</th>' +
                            '<th style="padding:6px 8px;border-bottom:2px solid var(--card-border);text-align:left;">详情</th></tr></thead><tbody>' +
                            logs.map(l => {
                                const timeStr = l.created_at ? new Date(l.created_at).toLocaleString('zh-CN', {month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit'}) : '—';
                                const actionBadge = l.action === 'DELETE'
                                    ? '<span style="background:#fef2f2;color:#dc2626;border-radius:4px;padding:1px 5px;font-size:.65rem;">DELETE</span>'
                                    : '<span style="background:#eff6ff;color:#2563eb;border-radius:4px;padding:1px 5px;font-size:.65rem;">UPDATE</span>';
                                const resultIcon = l.success === false
                                    ? '<span style="color:#dc2626;" title="' + escapeHtml(l.error_message||'') + '">❌</span>'
                                    : '<span style="color:#16a34a;">✅</span>';
                                return `<tr style="border-bottom:1px solid var(--card-border);${l.success===false?'background:#fff5f5;':''}">
                                    <td style="padding:4px 8px;font-size:.68rem;white-space:nowrap;">${timeStr}</td>
                                    <td style="padding:4px 8px;white-space:nowrap;">${actionBadge}</td>
                                    <td style="padding:4px 8px;text-align:center;">${resultIcon}</td>
                                    <td style="padding:4px 8px;font-size:.7rem;color:var(--card-muted);">${buildDetails(l)}</td>
                                </tr>`;
                            }).join('') + '</tbody></table>';
                    }
                    // Pagination
                    const pg = modal.getElementById('auditPagination');
                    pg.innerHTML = `<span style="font-size:.75rem;color:var(--card-muted);">共 ${data.total||0} 条</span>` +
                        (totalPages > 1 ? `<button class="pg-btn" ${currentPage<=1?'disabled':''} style="padding:4px 14px;border-radius:4px;border:1px solid var(--card-border);cursor:pointer;background:var(--card-bg);">◀</button>
                        <span style="padding:4px;font-size:.78rem;">${currentPage}/${totalPages}</span>
                        <button class="pg-btn" ${currentPage>=totalPages?'disabled':''} style="padding:4px 14px;border-radius:4px;border:1px solid var(--card-border);cursor:pointer;background:var(--card-bg);">▶</button>` : '');
                    pg.querySelectorAll('.pg-btn').forEach((b,i) => b.onclick = () => loadPage(i===0?currentPage-1:currentPage+1));
                } catch(e) { container.innerHTML = '<p style="color:#ef4444;">加载失败</p>'; }
            }
            loadPage(1);
            // Filters
            setTimeout(() => {
                const si = modal.getElementById('auditSearch');
                const af = modal.getElementById('auditActionFilter');
                const rf = modal.getElementById('auditResultFilter');
                if (si) si.oninput = () => { searchQuery = si.value.trim(); loadPage(1); };
                if (af) af.onchange = () => { actionFilter = af.value; loadPage(1); };
                if (rf) rf.onchange = () => { resultFilter = rf.value; loadPage(1); };

                // Export buttons
                const csvBtn = modal.getElementById('auditExportCsv');
                const jsonBtn = modal.getElementById('auditExportJson');
                const noteBtn = modal.getElementById('auditAddNote');
                if (csvBtn) csvBtn.onclick = () => exportAuditLog('csv', searchQuery, actionFilter, resultFilter);
                if (jsonBtn) jsonBtn.onclick = () => exportAuditLog('json', searchQuery, actionFilter, resultFilter);
                if (noteBtn) noteBtn.onclick = async () => {
                    const note = prompt('运维备注内容:');
                    if (!note) return;
                    try {
                        const res = await fetch('/admin/audit_note', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            credentials: 'include',
                            body: JSON.stringify({ note })
                        });
                        if (res.ok) { showToast('运维备注已记录', 'success'); loadPage(1); }
                        else showToast('记录失败', 'error');
                    } catch(_) { showToast('网络错误', 'error'); }
                };
            }, 100);
        };

        // Helper: export audit log
        async function exportAuditLog(format, search, action, success) {
            let allLogs = [];
            let page = 1, hasMore = true;
            while (hasMore) {
                const params = new URLSearchParams({ page, per_page: 500 });
                if (search) params.set('search', search);
                if (action) params.set('action', action);
                if (success) params.set('success', success);
                const res = await fetch(`/admin/audit_log?${params}`, { credentials: 'include' });
                const d = await res.json();
                allLogs = allLogs.concat(d.logs || []);
                hasMore = d.logs && d.logs.length === 500;
                page++;
            }
            if (format === 'json') {
                const blob = new Blob([JSON.stringify(allLogs, null, 2)], { type: 'application/json' });
                const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
                a.download = `audit_log_${new Date().toISOString().slice(0,10)}.json`; a.click();
            } else {
                if (!allLogs.length) { showToast('无数据可导出', 'error'); return; }
                const cols = ['created_at','admin_username','action','table_name','row_id','column_name','old_value','new_value','ip_address','success','error_message'];
                let csv = '\uFEFF' + cols.join(',') + '\n' + allLogs.map(r => cols.map(c => '"' + String(r[c]||'').replace(/"/g,'""') + '"').join(',')).join('\n');
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
                const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
                a.download = `audit_log_${new Date().toISOString().slice(0,10)}.csv`; a.click();
            }
            showToast(`已导出 ${allLogs.length} 条记录`, 'success');
        }


        // Knowledge sidebar
        const sukBtn = document.getElementById('sidebarUploadKnowledgeBtn');
        const srkBtn = document.getElementById('sidebarRefreshKnowledgeBtn');
        const sucBtn = document.getElementById('sidebarUploadCompanyBtn');
        if (sukBtn) sukBtn.onclick = () => { document.getElementById('labFileInput')?.click(); };
        if (srkBtn) srkBtn.onclick = () => { loadKnowledgeLabFiles(); loadCompanyKnowledgeBase(); loadSidebarKnowledge(); };
        if (sucBtn) sucBtn.onclick = () => { document.getElementById('companyFileInput')?.click(); };

        // Recycle sidebar — respect active filter
        const sraBtn = document.getElementById('sidebarRestoreAllBtn');
        const seaBtn = document.getElementById('sidebarEmptyAllBtn');
        const getActiveFilter = () => {
            const active = document.querySelector('.recycle-filter[style*=\"bccfde\"]');
            return active ? active.dataset.source : 'all';
        };
        if (sraBtn) sraBtn.onclick = async () => {
            const filter = getActiveFilter();
            const label = filter === 'all' ? '所有项目' : '筛选的项目';
            if (!confirm('恢复' + label + '？')) return;
            try {
                const res = await fetch('/restore_from_recycle_bin', { method: 'POST', headers: {'Content-Type':'application/json'}, credentials: 'include', body: JSON.stringify({restore_all: true, source: filter !== 'all' ? filter : undefined}) });
                if (res.ok) { showToast('已恢复', 'success'); loadSidebarRecycle(); if (typeof loadRecycleBin === 'function') loadRecycleBin(); }
            } catch(e) {}
        };
        if (seaBtn) seaBtn.onclick = async () => {
            const filter = getActiveFilter();
            const label = filter === 'all' ? '所有项目' : '筛选的项目';
            if (!confirm('确定永久删除' + label + '？此操作不可恢复。')) return;
            try {
                const res = await fetch('/empty_recycle_bin', { method: 'POST', credentials: 'include', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({source: filter || 'all'}) });
                if (res.ok) { showToast('已清空', 'success'); loadSidebarRecycle(); if (typeof loadRecycleBin === 'function') loadRecycleBin(); }
            } catch(e) { console.error('sidebar empty failed:', e); }
        };

        // Age-based cleanup buttons (delegated, since they're created dynamically)
        document.addEventListener('click', async (e) => {
            if (e.target.classList.contains('recycle-age-btn')) {
                const btn = e.target;
                const isOld = btn.textContent.includes('30天前');
                if (!confirm('确定永久删除' + (isOld ? '30天前' : '7天前') + '的项目？此操作不可恢复。')) return;
                btn.disabled = true; btn.textContent = '清理中...';
                try {
                    const res = await fetch('/empty_recycle_bin', { method: 'POST', credentials: 'include', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({source: 'all'}) });
                    if (res.ok) { showToast('已清理', 'success'); loadSidebarRecycle(); if (typeof loadRecycleBin === 'function') loadRecycleBin(); }
                } catch(err) { console.error('age cleanup failed:', err); }
                btn.disabled = false;
            }
        });
    });

    // Initial tab bar visibility
    async function loadProjects(cachedData) {
        const container = document.getElementById('projectsList');
        container.innerHTML = '加载中...';
        try {
            const data = cachedData || await (await fetch('/admin/projects', { credentials: 'include' })).json();
            if (data.error) { container.innerHTML = '<p>您没有任何项目。请联系管理员添加。</p>'; return; }
            const projects = data.projects || [];
            if (projects.length === 0) {
                container.innerHTML = '<p>暂无项目。点击“新建项目”开始。</p>';
                return;
            }
            let html = '<ul style="list-style: none; padding: 0;">';
            for (const p of projects) {
                const isQuitted = p.member_status === 'quitted';
                const rowOpacity = isQuitted ? 'opacity:0.45;' : '';
                const rowPointer = isQuitted ? 'cursor:not-allowed;' : '';

                let statusBadge = '';
                if (p.status === 'active') statusBadge = '<span class="project-status-badge status-active">进行中</span>';
                else if (p.status === 'archived') statusBadge = '<span class="project-status-badge status-archived">已归档</span>';
                else if (p.status === 'aborted') statusBadge = '<span class="project-status-badge status-aborted">已中止</span>';
                if (isQuitted) statusBadge += ' <span style="font-size:.65rem;background:#fef2f2;color:#dc2626;border-radius:4px;padding:1px 5px;">已退出</span>';
                if (p.member_role) statusBadge += ` <span style="font-size:.65rem;color:var(--card-muted);">${p.member_role === 'manager' ? '👑 管理' : '👤 成员'}</span>`;
                const industryLabels = {bidding_agency:'🏗️ 招标代理', engineering_cost:'💰 工程造价', engineering_audit:'🔍 工程审计', general:'📋 通用'};
                if (p.industry && p.industry !== 'general') statusBadge += ` <span style="font-size:.65rem;background:#eff6ff;color:#1e40af;border-radius:4px;padding:1px 5px;">${industryLabels[p.industry]||p.industry}</span>`;
                const catLabels = {construction:'🏗️ 工程建设', goods:'📦 货物采购', services:'💼 服务采购'};
                const methLabels = {open_bid:'公开招标', invited_bid:'邀请招标', competitive_negotiation:'竞争性谈判', inquiry:'询价', single_source:'单一来源', competitive_consultation:'竞争性磋商'};
                if (p.bidding_category) statusBadge += ` <span style="font-size:.65rem;background:#f0fff4;color:#166534;border-radius:4px;padding:1px 5px;">${catLabels[p.bidding_category]||p.bidding_category}</span>`;
                if (p.bid_method) statusBadge += ` <span style="font-size:.65rem;background:#fefce8;color:#854d0e;border-radius:4px;padding:1px 5px;">${methLabels[p.bid_method]||p.bid_method}</span>`;

                let buttonsHtml = '';
                if (!isQuitted) {
                    buttonsHtml = `<button class="open-project" data-id="${p.id}" data-name="${escapeHtml(p.name)}" data-status="${p.status}" style="background: #27ae60; color: white; border: none; border-radius: 4px; padding: 4px 8px;">${_icon('📂')} 打开</button>`;
                    if (p.status === 'active') {
                        buttonsHtml += `
                            <button class="member-manager-btn" data-id="${p.id}" style="background: #3498db; color: white; border: none; border-radius: 4px; padding: 4px 8px;">${_icon('👥')} 成员管理</button>
                            <button class="finish-project-btn" data-id="${p.id}" style="background: #f39c12; color: white; border: none; border-radius: 4px; padding: 4px 8px;">${_icon('🏁')} 完成并归档</button>
                            <button class="abort-project" data-id="${p.id}" style="background: #e67e22; color: white; border: none; border-radius: 4px; padding: 4px 8px;">${_icon('⛔')} 中止</button>
                        `;
                    }
                    if (p.status === 'archived') {
                        if (p.archive_filename) {
                            buttonsHtml += `<a href="/admin/projects/${p.id}/download_archive/${encodeURIComponent(p.archive_filename)}" class="download-archive-link" style="background: #3498db; color: white; text-decoration:none; border-radius:4px; padding:4px 8px; display:inline-block;">📥 下载归档</a>`;
                        }
                        buttonsHtml += `<button class="delete-project" data-id="${p.id}" style="background: #e74c3c; color: white; border: none; border-radius: 4px; padding: 4px 8px;">🗑️ 删除</button>`;
                    }
                    if (p.status === 'aborted') {
                        buttonsHtml += `<button class="delete-project" data-id="${p.id}" style="background: #e74c3c; color: white; border: none; border-radius: 4px; padding: 4px 8px;">🗑️ 删除</button>`;
                    }
                }
                html += `
                    <li class="project-row" data-id="${p.id}" data-name="${escapeHtml(p.name)}" data-status="${p.status}" data-member-status="${p.member_status||'active'}" style="border:1px solid #ddd; border-radius:8px; margin-bottom:12px; padding:12px; ${rowOpacity}">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                            <div class="project-clickable" style="flex:1; ${rowPointer}">
                                <strong>${escapeHtml(p.name)} ${statusBadge}</strong>
                                <p style="margin:8px 0;">${escapeHtml(p.description || '')}</p>
                                <small>创建于: ${new Date(p.created_at).toLocaleString()}</small>
                                <small>最后修改: ${p.updated_at ? new Date(p.updated_at).toLocaleString() : '从未'}</small>
                            </div>
                            <div class="project-buttons" style="flex-shrink:0;">${buttonsHtml}</div>
                        </div>
                    </li>
                `;
            }
            html += '</ul>';
            container.innerHTML = html;

            document.querySelectorAll('.project-clickable').forEach(clickable => {
                clickable.onclick = (e) => {
                    e.stopPropagation();
                    const row = clickable.closest('.project-row');
                    if (row) openProject(row.dataset.id, row.dataset.name, row.dataset.status);
                };
            });
            document.querySelectorAll('.open-project').forEach(btn => { btn.onclick = (e) => { e.stopPropagation(); openProject(btn.dataset.id, btn.dataset.name, btn.dataset.status); }; });
            document.querySelectorAll('.member-manager-btn').forEach(btn => { btn.onclick = (e) => { e.stopPropagation(); showMemberManager(btn.dataset.id); }; });
            document.querySelectorAll('.finish-project-btn').forEach(btn => { btn.onclick = (e) => { e.stopPropagation(); finishProject(btn.dataset.id); }; });
            document.querySelectorAll('.abort-project').forEach(btn => { btn.onclick = (e) => { e.stopPropagation(); abortProject(btn.dataset.id); }; });
            document.querySelectorAll('.delete-project').forEach(btn => { btn.onclick = (e) => { e.stopPropagation(); deleteProject(btn.dataset.id); }; });
        } catch (err) {
            console.error(err);
            container.innerHTML = '<p>加载失败，请确保您已登录。</p>';
        }
    }

    async function openProject(projectId, projectName, status) {
        // Block quitted members
        const row = document.querySelector(`.project-row[data-id="${projectId}"]`);
        if (row && row.dataset.memberStatus === 'quitted') {
            showToast('你已退出此项目，无法访问', 'error', 3000);
            return;
        }
        currentProjectId = projectId;
        currentProjectName = projectName;
        let description = '';
        let archiveFilename = '';
        let industry = 'general';
        let biddingCategory = '';
        let bidMethod = '';
        try {
            const res = await fetch(`/admin/projects`, { credentials: 'include' });
            if (!res.ok) throw new Error('Failed to fetch projects');
            const data = await res.json();
            const project = data.projects.find(p => p.id == projectId);
            if (project) {
                description = project.description || '';
                archiveFilename = project.archive_filename || '';
                industry = project.industry || 'general';
                biddingCategory = project.bidding_category || '';
                bidMethod = project.bid_method || '';
                if (!status) status = project.status || 'active';
            }
        } catch(e) { console.warn('Could not fetch description', e); }
        if (!status) status = 'active';
        if (!projectName) projectName = '项目#' + projectId;

        const projectsListView = document.getElementById('projectsListView');
        const fileExplorerView = document.getElementById('fileExplorerView');
        projectsListView.style.display = 'none';
        fileExplorerView.style.display = 'block';
        toggleQuickLinksButton(true);
        await loadProjectInfo(projectId, projectName, status, description, archiveFilename, industry, biddingCategory, bidMethod);
        await loadFolderTree(projectId);

        // ── Auto-load project chat into sidebar ──
        // Find the project's shared chat session and pre-load it so it
        // appears immediately when the user switches to the chat tab.
        try {
            await loadHistoryList(true);  // force=true bypasses _loadingHistory lock
            // Verify the project chat is in the list; if missing, backfill
            const sessionsRes = await fetch('/get_sessions', { credentials: 'include' });
            const sessionsData = await sessionsRes.json();
            const projectChat = (sessionsData.sessions || []).find(s => s.project_id == projectId && !s.is_grilling);
            if (!projectChat) {
                // Backfill: call admin endpoint to create missing project chat
                console.log('Backfilling missing project chat for', projectId);
                try {
                    const bfRes = await fetch(`/admin/projects/${projectId}/backfill_chat`, { method: 'POST', credentials: 'include' });
                    if (bfRes.ok) {
                        await loadHistoryList(true);
                    }
                } catch(e) { console.warn('Backfill failed:', e); }
            }
        } catch(e) { console.warn('Project chat load failed:', e); }

        syncActiveTabWithView();
    }

    async function loadProjectInfo(projectId, projectName, status, description, archiveFilename, industry, biddingCategory, bidMethod) {
        const container = document.getElementById('fileExplorerContent');
        const statusLabel = status === 'active' ? '进行中' : (status === 'archived' ? '已归档' : '已中止');
        const statusClass = status === 'active' ? 'status-active' : (status === 'archived' ? 'status-archived' : 'status-aborted');
        const industryLabels = {bidding_agency:'🏗️ 招标代理', engineering_cost:'💰 工程造价', engineering_audit:'🔍 工程审计', general:'📋 通用'};
        const industryBadge = (industry && industry !== 'general') ? `<span style="font-size:.65rem;background:#eff6ff;color:#1e40af;border-radius:4px;padding:1px 5px;">${industryLabels[industry]||industry}</span>` : '';
        const catLabels = {construction:'🏗️ 工程建设', goods:'📦 货物采购', services:'💼 服务采购'};
        const methLabels = {open_bid:'公开招标', invited_bid:'邀请招标', competitive_negotiation:'竞争性谈判', inquiry:'询价', single_source:'单一来源', competitive_consultation:'竞争性磋商'};
        const catBadge = biddingCategory ? `<span style="font-size:.65rem;background:#f0fff4;color:#166534;border-radius:4px;padding:1px 5px;">${catLabels[biddingCategory]||biddingCategory}</span>` : '';
        const methBadge = bidMethod ? `<span style="font-size:.65rem;background:#fefce8;color:#854d0e;border-radius:4px;padding:1px 5px;">${methLabels[bidMethod]||bidMethod}</span>` : '';

        // Archive banner for non-active projects
        let archiveBanner = '';
        if (status === 'archived') {
            archiveBanner = `<div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:10px 14px; margin-bottom:12px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                <span style="font-size:0.85rem; color:#1e40af;">${_icon('📦')} 此项目已归档，仅可查看和下载</span>
                ${archiveFilename ? `<a href="/admin/projects/${projectId}/download_archive/${encodeURIComponent(archiveFilename)}" style="background:#2563eb; color:white; text-decoration:none; border-radius:6px; padding:6px 14px; font-size:0.82rem;">${_icon('📥')} 下载归档ZIP</a>` : ''}
            </div>`;
        } else if (status === 'aborted') {
            archiveBanner = `<div style="background:#fef2f2; border:1px solid #fecaca; border-radius:8px; padding:10px 14px; margin-bottom:12px;">
                <span style="font-size:0.85rem; color:#991b1b;">${_icon('⛔')} 此项目已中止，仅可查看</span>
            </div>`;
        }

        container.innerHTML = `
            ${archiveBanner}
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
                <div style="flex:1;">
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        <h4 id="projectNameDisplay" style="margin:0;">${escapeHtml(projectName)}</h4>
                        <span class="project-status-badge ${statusClass}">${statusLabel}</span>
                        ${industryBadge}
                        ${catBadge}
                        ${methBadge}
                        ${status === 'active' ? '<button id="editProjectBtn" class="file-btn" style="padding: 2px 8px; font-size:0.7rem;">✏️ 编辑项目</button>' : ''}
                    </div>
                    <div id="projectDescriptionDisplay" style="margin-top:8px;color:#666;font-size:0.85rem;cursor:text;" title="双击编辑项目描述">
                        ${escapeHtml(description) || '无描述'}
                    </div>
                </div>
                ${status === 'active' ? `<button id="openProjectChatBtn" class="file-btn" style="background:#16a34a;color:white;border-color:#15803d;padding:6px 14px;font-size:0.78rem;white-space:nowrap;">💬 项目对话</button>
                <button id="openProjectGrillBtn" class="file-btn" style="background:#fef2f2;color:#991b1b;border-color:#fca5a5;padding:6px 14px;font-size:0.78rem;white-space:nowrap;" title="为此项目创建质问模式，AI将模拟挑剔供应商拷问招标文件">🔥 质问</button>
                <button id="bindTimelineBtn" class="file-btn" style="background:#dbeafe;color:#1e40af;border-color:#93c5fd;padding:6px 14px;font-size:0.78rem;white-space:nowrap;">📅 绑定时间线</button>` : ''}
            </div>
            <!-- AI助手: natural language content generation -->
            ${status === 'active' ? `
            <details id="aiAssistSection" style="margin-bottom:16px;border:1px solid #bfdbfe;border-radius:8px;padding:10px 14px;background:linear-gradient(135deg,#eff6ff,#f0f9ff);">
                <summary style="font-weight:600;font-size:0.82rem;cursor:pointer;color:#1e40af;">🤖 AI助手 — 用自然语言描述需求，AI帮你生成内容</summary>
                <div style="margin-top:8px;">
                    <p style="font-size:0.7rem;color:#6b7280;margin-bottom:6px;">
                        💡 试试这样说："帮我写一份这个项目的投标函"、"分析这些文件的风险点"、"根据模板生成报价单"
                    </p>
                    <textarea id="aiAssistInput" rows="3" placeholder="用一句话描述您需要AI帮您做什么..." style="width:100%;padding:8px;border-radius:6px;border:1px solid #93c5fd;font-size:0.82rem;resize:vertical;min-height:60px;"></textarea>
                    <div style="display:flex;gap:8px;margin-top:6px;align-items:center;flex-wrap:wrap;">
                        <button id="aiAssistSendBtn" class="file-btn" style="background:#2563eb;color:white;border-color:#1d4ed8;padding:6px 16px;">✨ 生成</button>
                        <select id="aiAssistFormat" style="padding:4px 8px;border-radius:4px;border:1px solid #93c5fd;font-size:0.75rem;">
                            <option value="">📝 纯文本</option>
                            <option value="docx">📄 Word (.docx)</option>
                            <option value="xlsx">📊 Excel (.xlsx)</option>
                            <option value="pptx">📽️ PPT (.pptx)</option>
                        </select>
                        <span id="aiAssistStatus" style="font-size:0.7rem;color:var(--card-muted);"></span>
                        <button id="aiWorkflowBtn" class="file-btn" style="background:#7c3aed;color:white;border-color:#6d28d9;padding:4px 10px;font-size:0.7rem;">🔄 多步工作流</button>
                        <input type="file" id="aiAnalyzeFileInput" accept=".xlsx,.xls,.csv" style="display:none;">
                        <button id="aiAnalyzeBtn" class="file-btn" style="padding:4px 10px;font-size:0.7rem;">📊 分析数据</button>
                        <span id="aiAssistDownloadArea" style="display:none;">
                            <select id="aiAssistDlFormat" style="padding:2px 6px;border-radius:4px;border:1px solid #93c5fd;font-size:0.7rem;">
                                <option value="docx">📄 .docx</option>
                                <option value="xlsx">📊 .xlsx</option>
                                <option value="pptx">📽️ .pptx</option>
                            </select>
                            <a id="aiAssistDownload" style="font-size:0.7rem;color:#2563eb;text-decoration:underline;cursor:pointer;">📥 下载</a>
                        </span>
                    </div>
                    <div id="aiAssistResult" style="margin-top:8px;padding:10px;background:white;border-radius:6px;border:1px solid #e5e7eb;font-size:0.82rem;line-height:1.6;white-space:pre-wrap;display:none;"></div>
                </div>
            </details>` : ''}
            <!-- Sub-tab bar -->
            <div class="project-sub-tab-bar" style="display:flex;gap:0;margin-bottom:16px;border-bottom:2px solid var(--card-border);">
                <button class="project-sub-tab-btn active" data-tab="files" style="padding:8px 20px;font-size:0.82rem;border:2px solid transparent;border-bottom:none;background:transparent;cursor:pointer;color:var(--card-muted);border-radius:8px 8px 0 0;">📁 文件</button>
                <button class="project-sub-tab-btn" data-tab="graph" style="padding:8px 20px;font-size:0.82rem;border:2px solid transparent;border-bottom:none;background:transparent;cursor:pointer;color:var(--card-muted);border-radius:8px 8px 0 0;">🕸️ 图谱</button>
            </div>
            <!-- Files tab -->
            <div id="projectFilesTab">
                <div id="folderTreeContainer" style="margin-bottom: 20px;"></div>
                <div id="fileListContainer"></div>
            </div>
            <!-- Graph tab (hidden by default) -->
            <div id="projectGraphTab" style="display:none;">
                <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
                    <button id="projectCollusionGraphBtn" class="graph-type-btn active" style="padding:6px 16px;font-size:0.82rem;border:1px solid var(--card-border);border-radius:6px;background:var(--card-bg);cursor:pointer;transition:all 0.15s;">🕸️ 围串标</button>
                    <button id="projectComplianceGraphBtn" class="graph-type-btn" style="padding:6px 16px;font-size:0.82rem;border:1px solid var(--card-border);border-radius:6px;background:var(--card-bg);cursor:pointer;transition:all 0.15s;">🏛️ 合规违规</button>
                    <button id="projectCitationGraphBtn" class="graph-type-btn" style="padding:6px 16px;font-size:0.82rem;border:1px solid var(--card-border);border-radius:6px;background:var(--card-bg);cursor:pointer;transition:all 0.15s;">📄 文档引用</button>
                </div>
                <div id="projectGraphStats" style="margin-bottom:6px;min-height:18px;font-size:0.7rem;color:var(--card-muted);"></div>
                <div id="projectGraphContainer" style="position:relative;width:100%;height:450px;border:1px solid var(--card-border);border-radius:6px;overflow:hidden;">
                    <span id="projectGraphPlaceholder" style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:0.78rem;">选择图谱类型查看</span>
                </div>
            </div>
        `;

        let currentDescription = description || '';

        // Wire "💬 项目对话" button — jumps to shared project chat
        // innerHTML is synchronous so buttons exist immediately
        const chatBtn = document.getElementById('openProjectChatBtn');
        if (chatBtn) {
            chatBtn.onclick = async () => {
                try {
                    const res = await fetch('/get_sessions', { credentials: 'include' });
                    const data = await res.json();
                    const projChat = (data.sessions || []).find(s => s.project_id == projectId && !s.is_grilling);
                    if (projChat) {
                        document.getElementById('chatTabBtn')?.click();
                        await loadSession(projChat.thread_id);
                    } else {
                        const bfRes = await fetch(`/admin/projects/${projectId}/backfill_chat`, { method: 'POST', credentials: 'include' });
                        if (bfRes.ok) {
                            const bf = await bfRes.json();
                            document.getElementById('chatTabBtn')?.click();
                            await loadSession(bf.thread_id);
                        } else {
                            showToast('未找到项目对话，请刷新重试', 'error', 3000);
                        }
                    }
                } catch(e) {
                    console.error('Project chat button error:', e);
                    showToast('打开项目对话失败', 'error', 3000);
                }
            };
        }
        const grillBtn = document.getElementById('openProjectGrillBtn');
        if (grillBtn) {
            grillBtn.onclick = async () => {
                try {
                    const res = await fetch(`/api/projects/${projectId}/get_or_create_grill_thread`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        credentials: 'include'
                    });
                    if (res.ok) {
                        const data = await res.json();
                        document.getElementById('chatTabBtn')?.click();
                        await loadSession(data.thread_id);
                        await loadHistoryList();
                    } else {
                        showToast('创建项目质问模式失败', 'error', 3000);
                    }
                } catch(e) {
                    console.error('Project grill button error:', e);
                    showToast('创建项目质问模式失败', 'error', 3000);
                }
            };
        }

        const bindTimelineBtn = document.getElementById('bindTimelineBtn');
        if (bindTimelineBtn) {
            bindTimelineBtn.onclick = () => {
                _switchSubTab('timeline');
            };
        }

        // Wire project sub-tab switching
        setTimeout(() => {
            // Sub-tab switching
            document.querySelectorAll('.project-sub-tab-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const tab = btn.getAttribute('data-tab');
                    document.querySelectorAll('.project-sub-tab-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    document.getElementById('projectFilesTab').style.display = tab === 'files' ? '' : 'none';
                    document.getElementById('projectGraphTab').style.display = tab === 'graph' ? '' : 'none';
                });
            });

            // Graph type button wiring
            var collusionBtn = document.getElementById('projectCollusionGraphBtn');
            var complianceBtn = document.getElementById('projectComplianceGraphBtn');
            var citationBtn = document.getElementById('projectCitationGraphBtn');

            function switchGraphType(activeBtn) {
                var container = document.getElementById('projectGraphContainer');
                var statsEl = document.getElementById('projectGraphStats');
                [collusionBtn, complianceBtn, citationBtn].forEach(function(b) {
                    if (b) { b.classList.remove('active'); b.style.background = 'var(--card-bg)'; b.style.color = ''; b.style.fontWeight = ''; }
                });
                if (activeBtn) { activeBtn.classList.add('active'); activeBtn.style.background = '#1e293b'; activeBtn.style.color = 'white'; activeBtn.style.fontWeight = '600'; }
                if (statsEl) statsEl.textContent = '';
                if (container) { container.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">加载中...</span>'; }
            }

            if (collusionBtn) collusionBtn.addEventListener('click', function() {
                switchGraphType(collusionBtn);
                loadProjectCollusionGraph();
            });
            if (complianceBtn) complianceBtn.addEventListener('click', function() {
                switchGraphType(complianceBtn);
                loadProjectComplianceGraph();
            });
            if (citationBtn) citationBtn.addEventListener('click', function() {
                switchGraphType(citationBtn);
                loadProjectCitationGraph();
            });
        }, 100);

        // Wire AI Assist buttons
        setTimeout(() => {
            const aiInput = document.getElementById('aiAssistInput');
            const aiSendBtn = document.getElementById('aiAssistSendBtn');
            const aiFormat = document.getElementById('aiAssistFormat');
            const aiStatus = document.getElementById('aiAssistStatus');
            const aiResult = document.getElementById('aiAssistResult');
            const aiDownload = document.getElementById('aiAssistDownload');
            const aiDlArea = document.getElementById('aiAssistDownloadArea');
            const aiDlFormat = document.getElementById('aiAssistDlFormat');
            let _lastMemoryId = null;

            // Poll for real-time AI activity from other members
            let _lastPollTime = new Date().toISOString();
            const _pollInterval = setInterval(async () => {
                try {
                    const r = await fetch(`/admin/projects/${projectId}/ai_activity?since=${encodeURIComponent(_lastPollTime)}`, {credentials:'include'});
                    const d = await r.json();
                    if (d.items && d.items.length > 0) {
                        const newAssists = d.items.filter(i => i.role === 'assistant');
                        if (newAssists.length > 0) {
                            const latest = newAssists[newAssists.length - 1];
                            showToast(`📢 @${escapeHtml(latest.username)} 刚刚生成了新内容`, 'info', 4000);
                        }
                    }
                    if (d.now) _lastPollTime = d.now;
                } catch(_) {}
            }, 10000); // poll every 10s

            if (aiSendBtn) {
                aiSendBtn.onclick = async () => {
                    const query = (aiInput?.value || '').trim();
                    const fmt = aiFormat?.value || '';
                    if (!query) { if (aiStatus) aiStatus.textContent = '请先输入需求描述'; return; }
                    aiSendBtn.disabled = true; aiSendBtn.textContent = '⏳ 生成中...';
                    if (aiStatus) aiStatus.innerHTML = '';
                    if (aiResult) { aiResult.style.display = 'block'; aiResult.textContent = ''; }
                    if (aiDlArea) aiDlArea.style.display = 'none';

                    // Start timer
                    const startTime = Date.now();
                    const timerInterval = setInterval(() => {
                        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                        if (aiStatus) aiStatus.innerHTML = '<span class="typing-dots"><span>.</span><span>.</span><span>.</span></span> <small>' + elapsed + 's</small>';
                    }, 200);

                    try {
                        const body = { query, output_format: fmt };
                        if (_currentQuoteContext && _currentQuoteContext.quotedMessageId) {
                            body.quoted_message_id = _currentQuoteContext.quotedMessageId;
                        }
                        const r = await fetch(`/admin/projects/${projectId}/ai_assist/stream`, {
                            method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                            body: JSON.stringify(body)
                        });
                        clearInterval(timerInterval);

                        if (!r.ok) {
                            const d = await r.json();
                            if (aiStatus) aiStatus.textContent = '❌ ' + (d.error || '生成失败');
                            aiSendBtn.disabled = false; aiSendBtn.textContent = '✨ 生成';
                            return;
                        }

                        // Read SSE stream
                        const reader = r.body.getReader();
                        const decoder = new TextDecoder();
                        let fullText = '';
                        let buffer = '';
                        if (aiStatus) aiStatus.textContent = '生成中...';

                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            buffer += decoder.decode(value, { stream: true });
                            const lines = buffer.split('\n');
                            buffer = lines.pop() || '';
                            for (const line of lines) {
                                if (line.startsWith('data: ')) {
                                    try {
                                        const d = JSON.parse(line.slice(6));
                                        if (d.text) {
                                            fullText += d.text;
                                            if (aiResult) {
                                                aiResult.textContent = fullText;
                                                aiResult.style.display = 'block';
                                            }
                                        }
                                    } catch(e) {}
                                } else if (line.startsWith('event: done')) {
                                    // handled below
                                } else if (line.startsWith('event: error')) {
                                    try {
                                        const d = JSON.parse(line.slice(5));
                                        if (aiStatus) aiStatus.textContent = '❌ ' + (d.error || '生成失败');
                                    } catch(e) {}
                                }
                            }
                        }

                        if (aiStatus) {
                            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                            aiStatus.textContent = '✅ 生成完成 (' + elapsed + 's)';
                        }

                        // Toggle: save output format result
                        if (fullText) {
                            _lastMemoryId = 'stream-' + Date.now();
                            if (aiDlArea) {
                                aiDlArea.style.display = 'inline';
                                aiDownload.onclick = () => {
                                    const dlFmt = aiDlFormat?.value || 'docx';
                                    // Download via blob
                                    const mime = dlFmt === 'xlsx' ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' : 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
                                    const blob = new Blob([fullText], { type: 'text/markdown' });
                                    const url = URL.createObjectURL(blob);
                                    const a = document.createElement('a');
                                    a.href = url;
                                    a.download = 'ai_output.' + dlFmt;
                                    a.click();
                                    URL.revokeObjectURL(url);
                                };
                            }
                        }
                        loadHistoryList().catch(()=>{});
                    } catch(_) { if (aiStatus) aiStatus.textContent = '❌ 网络错误'; clearInterval(timerInterval); }
                    _currentQuoteContext = null;
                    setTimeout(() => { document.querySelectorAll('.quote-badge').forEach(b => b.remove()); }, 1000);
                    aiSendBtn.disabled = false; aiSendBtn.textContent = '✨ 生成';
                };
                if (aiInput) aiInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); aiSendBtn.click(); }
                });
            }
            // Workflow state
            let _wfSteps = [], _wfStepIdx = 0, _wfResults = [];

            // Check if workflow needs setup (first visit)
            async function _checkWorkflowSetup() {
                try {
                    const r = await fetch(`/admin/projects/${projectId}/my_workflow`, {credentials:'include'});
                    const d = await r.json();
                    if (d.needs_setup) {
                        _showWorkflowSetup();
                    } else if (d.workflow) {
                        _wfSteps = d.workflow.steps || [];
                    }
                } catch(_) {}
            }
            setTimeout(_checkWorkflowSetup, 200);

            function _showWorkflowSetup() {
                if (aiResult) {
                    aiResult.innerHTML = `<div style="background:#eff6ff;border:2px solid #3b82f6;border-radius:8px;padding:12px;margin-bottom:8px;">
                        <h4 style="margin:0 0 8px;color:#1e40af;">🔧 设置你的工作流（首次必做）</h4>
                        <p style="font-size:0.72rem;color:#6b7280;margin-bottom:8px;">定义你在这个项目中的工作步骤，AI将按你的步骤逐一执行。</p>
                        <div id="wfSetupSteps"></div>
                        <div style="display:flex;gap:6px;margin-top:8px;">
                            <button id="wfAddStepBtn" class="file-btn" style="font-size:0.7rem;">➕ 添加步骤</button>
                            <button id="wfResetStepsBtn" class="file-btn" style="font-size:0.7rem;">🔄 重置为默认</button>
                        </div>
                        <div style="margin-top:8px;">
                            <input id="wfNameInput" placeholder="工作流名称(可选)" style="padding:4px 8px;border-radius:4px;border:1px solid #93c5fd;font-size:0.75rem;width:200px;">
                            <button id="wfSaveBtn" class="file-btn" style="background:#2563eb;color:white;border-color:#1d4ed8;font-size:0.75rem;padding:4px 12px;margin-left:6px;">💾 保存并开始</button>
                        </div>
                    </div>`;
                    aiResult.style.display = 'block';

                    const defaultSteps = [
                        {step:'需求分析', desc:'分析项目需求和目标'},
                        {step:'起草初稿', desc:'根据需求起草文档初稿'},
                        {step:'自审修改', desc:'审查初稿并修改'},
                        {step:'最终定稿', desc:'润色并输出最终文档'},
                    ];
                    let tempSteps = [...defaultSteps];

                    function renderSteps() {
                        const el = document.getElementById('wfSetupSteps');
                        if (!el) return;
                        el.innerHTML = tempSteps.map((s,i) => `
                            <div style="display:flex;gap:4px;align-items:center;margin-bottom:4px;">
                                <span style="font-weight:600;min-width:20px;">${i+1}.</span>
                                <input class="wf-step-name" value="${escapeHtml(s.step)}" placeholder="步骤名" style="flex:1;padding:3px 6px;border-radius:4px;border:1px solid #d1d5db;font-size:0.72rem;">
                                <input class="wf-step-desc" value="${escapeHtml(s.desc)}" placeholder="步骤描述" style="flex:2;padding:3px 6px;border-radius:4px;border:1px solid #d1d5db;font-size:0.72rem;">
                                <button class="wf-del-step" data-i="${i}" style="color:#ef4444;background:none;border:none;cursor:pointer;font-size:0.7rem;">✕</button>
                            </div>`).join('');
                        document.querySelectorAll('.wf-del-step').forEach(b => {
                            b.onclick = () => { tempSteps.splice(parseInt(b.dataset.i),1); renderSteps(); };
                        });
                    }
                    renderSteps();
                    document.getElementById('wfAddStepBtn').onclick = () => {
                        tempSteps.push({step:'新步骤', desc:'描述此步骤'});
                        renderSteps();
                    };
                    document.getElementById('wfResetStepsBtn').onclick = () => {
                        tempSteps = [...defaultSteps];
                        renderSteps();
                    };
                    document.getElementById('wfSaveBtn').onclick = async () => {
                        const names = [...document.querySelectorAll('.wf-step-name')].map(i=>i.value.trim()).filter(Boolean);
                        const descs = [...document.querySelectorAll('.wf-step-desc')].map(i=>i.value.trim()).filter(Boolean);
                        const steps = names.map((n,i)=>({step:n, desc:descs[i]||''}));
                        if (!steps.length) { alert('请至少添加一个步骤'); return; }
                        const name = document.getElementById('wfNameInput')?.value.trim() || '默认工作流';
                        const r = await fetch(`/admin/projects/${projectId}/my_workflow`, {
                            method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                            body: JSON.stringify({steps, name})
                        });
                        if (r.ok) {
                            _wfSteps = steps; _wfStepIdx = 0; _wfResults = [];
                            if (aiResult) aiResult.innerHTML = '<div style="color:#22c55e;text-align:center;padding:20px;">' + _icon('check_circle') + ' 工作流已保存！<br>在输入框中描述你的需求，然后点击 <b>🔄 执行工作流</b></div>';
                        }
                    };
                }
            }

            // Workflow button — step-by-step interactive
            const aiWorkflowBtn = document.getElementById('aiWorkflowBtn');
            if (aiWorkflowBtn) aiWorkflowBtn.onclick = async () => {
                if (!_wfSteps.length) { _checkWorkflowSetup(); if (aiStatus) aiStatus.textContent = '请先设置工作流'; return; }
                const query = (aiInput?.value || '').trim();
                if (!query && !_wfResults.length) { if (aiStatus) aiStatus.textContent = '请先输入需求描述'; return; }

                const currentStep = _wfSteps[_wfStepIdx];
                if (!currentStep) { if (aiStatus) aiStatus.textContent = '✅ 所有步骤已完成'; return; }

                aiWorkflowBtn.disabled = true;
                aiWorkflowBtn.textContent = `⏳ ${currentStep.step}...`;
                if (aiStatus) aiStatus.textContent = `步骤${_wfStepIdx+1}/${_wfSteps.length}: ${currentStep.step}`;
                if (aiResult) aiResult.style.display = 'block';

                try {
                    const r = await fetch(`/admin/projects/${projectId}/ai_workflow_step`, {
                        method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                        body: JSON.stringify({query, step_index: _wfStepIdx, step_action: 'execute'})
                    });
                    const d = await r.json();
                    if (r.ok && d.result) {
                        _wfResults[_wfStepIdx] = d.result;
                        let html = `<div style="background:#f0fdf4;border-radius:6px;padding:8px;margin-bottom:8px;">
                            <strong>📋 ${escapeHtml(d.step_name)} (${_wfStepIdx+1}/${d.total_steps})</strong>
                            ${d.warning ? `<div style="background:#fef3c7;border-radius:4px;padding:4px;margin-top:4px;font-size:0.7rem;color:#92400e;">${escapeHtml(d.warning)}</div>` : ''}
                            <div style="white-space:pre-wrap;font-size:0.78rem;margin-top:4px;">${escapeHtml(d.result)}</div></div>`;

                        // Show previous steps
                        _wfResults.forEach((res, i) => {
                            if (i < _wfStepIdx && res) {
                                html += `<details style="margin-bottom:4px;"><summary>✅ ${escapeHtml(_wfSteps[i]?.step||'步骤'+(i+1))} (已完成)</summary><div style="font-size:0.72rem;white-space:pre-wrap;">${escapeHtml(res)}</div></details>`;
                            }
                        });

                        // Action buttons for current step
                        html += `<div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;">
                            <button id="wfApproveBtn" class="file-btn" style="background:#22c55e;color:white;font-size:0.7rem;padding:4px 10px;">✅ 确认，下一步</button>
                            <input id="wfReviseInput" placeholder="修改意见..." style="flex:1;padding:3px 6px;border-radius:4px;border:1px solid #d1d5db;font-size:0.7rem;">
                            <button id="wfReviseBtn" class="file-btn" style="background:#f59e0b;color:white;font-size:0.7rem;padding:4px 10px;">✏️ 修改</button>
                            <button id="wfSkipBtn" class="file-btn" style="font-size:0.7rem;padding:4px 10px;">⏭️ 跳过</button>
                        </div>`;

                        if (aiResult) aiResult.innerHTML = html;

                        // Wire action buttons
                        document.getElementById('wfApproveBtn').onclick = async () => {
                            _wfStepIdx++;
                            if (_wfStepIdx >= _wfSteps.length) {
                                if (aiResult) aiResult.innerHTML += '<div style="color:#22c55e;text-align:center;padding:10px;font-weight:600;">🎉 工作流全部完成！</div>';
                                _wfStepIdx = 0; _wfResults = [];
                                aiWorkflowBtn.textContent = '🔄 执行工作流';
                            } else {
                                aiWorkflowBtn.textContent = `🔄 第${_wfStepIdx+1}步: ${_wfSteps[_wfStepIdx]?.step||''}`;
                                if (aiStatus) aiStatus.textContent = `点击继续: ${_wfSteps[_wfStepIdx]?.step}`;
                            }
                            aiWorkflowBtn.disabled = false;
                        };
                        document.getElementById('wfReviseBtn').onclick = async () => {
                            const revQ = document.getElementById('wfReviseInput')?.value.trim();
                            if (!revQ) return;
                            aiWorkflowBtn.disabled = true;
                            aiWorkflowBtn.textContent = '⏳ 修改中...';
                            const rr = await fetch(`/admin/projects/${projectId}/ai_workflow_step`, {
                                method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                                body: JSON.stringify({query, step_index: _wfStepIdx, step_action:'revise', revised_query: revQ})
                            });
                            const rd = await rr.json();
                            if (rr.ok && rd.result) {
                                _wfResults[_wfStepIdx] = rd.result;
                                aiWorkflowBtn.onclick(); // re-render
                            }
                            aiWorkflowBtn.disabled = false;
                        };
                        document.getElementById('wfSkipBtn').onclick = () => {
                            document.getElementById('wfApproveBtn')?.click();
                        };
                    } else {
                        if (aiStatus) aiStatus.textContent = '❌ ' + (d.error || '执行失败');
                    }
                } catch(_) { if (aiStatus) aiStatus.textContent = '❌ 网络错误'; }
                aiWorkflowBtn.disabled = false;
            };
            // Data analysis button
            const aiAnalyzeBtn = document.getElementById('aiAnalyzeBtn');
            const aiAnalyzeFile = document.getElementById('aiAnalyzeFileInput');
            if (aiAnalyzeBtn && aiAnalyzeFile) {
                aiAnalyzeBtn.onclick = () => aiAnalyzeFile.click();
                aiAnalyzeFile.onchange = async () => {
                    const file = aiAnalyzeFile.files[0];
                    if (!file) return;
                    aiAnalyzeBtn.disabled = true; aiAnalyzeBtn.textContent = '⏳ 分析中...';
                    if (aiStatus) aiStatus.textContent = '';
                    if (aiResult) { aiResult.style.display = 'none'; aiResult.textContent = ''; }
                    const form = new FormData(); form.append('file', file);
                    try {
                        const r = await fetch(`/admin/projects/${projectId}/ai_analyze`, {method:'POST',credentials:'include',body:form});
                        const d = await r.json();
                        if (r.ok && d.analysis) {
                            const a = d.analysis;
                            let html = `<div><b>${a.rows}行 × ${a.columns}列</b> | 列: ${(a.columns_list||[]).join(', ')}</div>`;
                            if (a.stats) {
                                html += '<details><summary>📊 统计摘要</summary><table style="font-size:0.65rem;">';
                                for (const [col, stat] of Object.entries(a.stats)) {
                                    html += `<tr><td>${col}</td><td>均值:${stat.mean}</td><td>最大:${stat.max}</td><td>最小:${stat.min}</td><td>合计:${stat.sum}</td></tr>`;
                                }
                                html += '</table></details>';
                            }
                            if (a.comparison) {
                                html += `<details open><summary>📈 ${a.comparison_key} vs ${a.comparison_value}</summary><table style="font-size:0.65rem;">`;
                                a.comparison.forEach(c => { html += `<tr><td>${escapeHtml(c.name)}</td><td><b>${c.value}</b></td></tr>`; });
                                html += '</table></details>';
                            }
                            if (a.anomalies_count > 0) {
                                html += `<details open><summary>⚠️ 异常检测: ${a.anomalies_count}条</summary><div style="font-size:0.7rem;color:#dc2626;">`;
                                (a.anomalies||[]).forEach(an => { html += `<div>#${an.row}: ${escapeHtml(an.label)}</div>`; });
                                html += '</div></details>';
                            }
                            if (aiResult) { aiResult.innerHTML = html; aiResult.style.display = 'block'; }
                        } else {
                            if (aiStatus) aiStatus.textContent = '❌ ' + (d.error || '分析失败');
                        }
                    } catch(_) { if (aiStatus) aiStatus.textContent = '❌ 网络错误'; }
                    aiAnalyzeBtn.disabled = false; aiAnalyzeBtn.textContent = '📊 分析数据';
                    aiAnalyzeFile.value = '';
                };
            }
        }, 100);

        const editBtn = document.getElementById('editProjectBtn');
        if (editBtn) {
            editBtn.onclick = async () => {
                const newName = await prompt('请输入新的项目名称:', projectName);
                if (newName && newName.trim() !== projectName) {
                    const updateRes = await fetch(`/admin/projects/${projectId}`, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        credentials: 'include',
                        body: JSON.stringify({ name: newName.trim(), description: currentDescription })
                    });
                    if (updateRes.ok) {
                        projectName = newName.trim();
                        document.getElementById('projectNameDisplay').innerText = projectName;
                        if (typeof currentProjectName !== 'undefined') currentProjectName = projectName;
                    } else {
                        const err = await updateRes.json();
                        alert('更新项目名称失败: ' + (err.error || '未知错误'));
                    }
                }
            };
        }

        // ── Inline description editing (double-click) ──
        const descDisplay = document.getElementById('projectDescriptionDisplay');
        if (descDisplay) {
            descDisplay.style.cursor = 'text';
            descDisplay.title = '双击编辑项目描述';
            descDisplay.ondblclick = async () => {
                const currentText = descDisplay.innerText === '无描述' ? '' : descDisplay.innerText;
                const input = document.createElement('textarea');
                input.value = currentText;
                input.style.cssText = 'width:100%;padding:6px 8px;border:1px solid #93c5fd;border-radius:6px;font-size:0.85rem;resize:vertical;min-height:40px;box-sizing:border-box;';
                descDisplay.replaceWith(input);
                input.focus();
                let done = false;
                const save = async () => {
                    if (done) return; done = true;
                    const newText = input.value.trim();
                    const span = document.createElement('div');
                    span.id = 'projectDescriptionDisplay';
                    span.style.cssText = 'margin-top:8px;color:#666;font-size:0.85rem;cursor:text;';
                    span.title = '双击编辑项目描述';
                    span.innerText = newText || '无描述';
                    span.ondblclick = descDisplay.ondblclick;
                    input.replaceWith(span);
                    if (newText !== currentText) {
                        try {
                            const updateRes = await fetch(`/admin/projects/${projectId}`, {
                                method: 'PUT', headers: { 'Content-Type': 'application/json' },
                                credentials: 'include',
                                body: JSON.stringify({ name: projectName, description: newText })
                            });
                            if (updateRes.ok) {
                                currentDescription = newText;
                                showToast('描述已更新', 'success', 2000);
                            } else {
                                span.innerText = currentText || '无描述';
                                const err = await updateRes.json();
                                showToast('更新失败: ' + (err.error || '未知错误'), 'error', 3000);
                            }
                        } catch(e) {
                            span.innerText = currentText || '无描述';
                            showToast('网络错误', 'error', 2000);
                        }
                    }
                };
                input.onblur = save;
                input.onkeydown = (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); save(); }
                    if (e.key === 'Escape') { input.blur(); }
                };
            };
        }
    }

    async function loadFolderTree(projectId, selectedFolderId = null) {
        const container = document.getElementById('folderTreeContainer');
        container.innerHTML = '加载中...';
        try {
            const res = await fetch(`/admin/projects/${projectId}/folders`, { credentials: 'include' });
            if (!res.ok) {
                const err = await res.json();
                container.innerHTML = `<p>加载失败: ${err.error || '未知错误'}</p>`;
                return;
            }
            const data = await res.json();
            const folders = data.folders || [];

            function renderTree(folderList, level = 0) {
                let html = '<ul class="folder-tree" style="list-style: none; padding-left: ' + (level * 20) + 'px;">';
                for (const f of folderList) {
                    const isSelected = (selectedFolderId == f.id);
                    html += `
                        <li>
                            <div class="folder-item ${isSelected ? 'selected' : ''}" data-folder-id="${f.id}">
                                <span class="folder-icon">📁</span>
                                <span class="folder-name" data-folder-id="${f.id}" data-folder-name="${escapeHtml(f.name)}">${escapeHtml(f.name)}</span>
                                <span style="font-size:0.7rem; color:#888;">${escapeHtml(f.path)}</span>
                                <button class="rename-folder-btn" data-folder-id="${f.id}" data-folder-name="${escapeHtml(f.name)}" style="margin-left:8px; background:none; border:none; cursor:pointer; font-size:0.7rem;" title="重命名文件夹">✏️</button>
                                <button class="delete-folder-btn" data-folder-id="${f.id}" data-folder-name="${escapeHtml(f.name)}" style="margin-left:4px; background:none; border:none; cursor:pointer; font-size:0.8rem; color:#e74c3c;" title="删除文件夹">❌</button>
                            </div>
                            ${f.children && f.children.length ? renderTree(f.children, level + 1) : ''}
                        </li>
                    `;
                }
                html += '</ul>';
                return html;
            }

            let treeHtml = renderTree(folders);
            container.innerHTML = treeHtml;

            document.querySelectorAll('.rename-folder-btn').forEach(btn => {
                btn.onclick = (e) => {
                    e.stopPropagation();
                    const folderId = btn.dataset.folderId;
                    const oldName = btn.dataset.folderName;
                    renameFolder(projectId, folderId, oldName);
                };
            });
            document.querySelectorAll('.folder-name').forEach(nameSpan => {
                nameSpan.ondblclick = (e) => {
                    e.stopPropagation();
                    const folderId = nameSpan.dataset.folderId;
                    const oldName = nameSpan.dataset.folderName;
                    renameFolder(projectId, folderId, oldName);
                };
            });
            document.querySelectorAll('.delete-folder-btn').forEach(btn => {
                btn.onclick = (e) => {
                    e.stopPropagation();
                    const folderId = btn.dataset.folderId;
                    const folderName = btn.dataset.folderName;
                    deleteFolder(projectId, folderId, folderName);
                };
            });
            document.querySelectorAll('.folder-item').forEach(el => {
                el.onclick = (e) => {
                    if (e.target.classList.contains('rename-folder-btn') || e.target.classList.contains('delete-folder-btn')) return;
                    document.querySelectorAll('.folder-item').forEach(f => f.classList.remove('selected'));
                    el.classList.add('selected');
                    const folderId = el.dataset.folderId;
                    loadFilesInFolder(projectId, folderId);
                    currentFolderId = folderId;
                };
            });

            if (selectedFolderId) {
                const selectedEl = document.querySelector(`.folder-item[data-folder-id="${selectedFolderId}"]`);
                if (selectedEl) selectedEl.click();
            } else if (folders.length > 0) {
                const firstFolder = document.querySelector('.folder-item');
                if (firstFolder) firstFolder.click();
            } else {
                loadFilesInFolder(projectId, null);
            }
        } catch (err) {
            console.error(err);
            container.innerHTML = '<p>加载文件夹失败</p>';
        }
    }

    async function renameFolder(projectId, folderId, oldName) {
        const newName = await prompt('请输入新的文件夹名称:', oldName);
        if (!newName || newName === oldName) return;
        try {
            const res = await fetch(`/admin/projects/${projectId}/folders/${folderId}/rename`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ name: newName })
            });
            if (res.ok) {
                await loadFolderTree(projectId, currentFolderId);
            } else {
                const err = await res.json();
                alert('重命名失败: ' + (err.error || '未知错误'));
            }
        } catch (err) {
            alert('网络错误');
        }
    }

    async function deleteFolder(projectId, folderId, folderName) {
        if (!await confirm(`确定要删除文件夹“${folderName}”及其所有内容吗？回收站保存时间3天。`)) return;
        try {
            const res = await fetch(`/admin/projects/${projectId}/folders/${folderId}`, {
                method: 'DELETE',
                credentials: 'include'
            });
            if (res.ok) {
                showToast('文件夹已删除', 'success', 3000);
                await loadFolderTree(projectId, null);
                loadFilesInFolder(projectId, null);
                currentFolderId = null;
            } else {
                const err = await res.json();
                alert('删除失败: ' + (err.error || '未知错误'));
            }
        } catch (err) {
            alert('网络错误');
        }
    }

    async function createSubfolder(projectId, parentId) {
        const name = await prompt('请输入文件夹名称:');
        if (!name) return;
        try {
            const res = await fetch(`/admin/projects/${projectId}/folders`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ name, parent_folder_id: parentId })
            });
            if (res.ok) {
                await loadFolderTree(projectId, currentFolderId);
            } else {
                const err = await res.json();
                alert('创建失败: ' + (err.error || '未知错误'));
            }
        } catch (err) {
            alert('网络错误');
        }
    }

    async function loadFilesInFolder(projectId, folderId) {
        window._currentProjectId = projectId;
        const container = document.getElementById('fileListContainer');
        container.innerHTML = '加载中...';
        try {
            const url = folderId ? `/admin/projects/${projectId}/folders/${folderId}/files` : `/admin/projects/${projectId}/files`;
            const res = await fetch(url, { credentials: 'include' });
            if (!res.ok) throw new Error('Failed to load files');
            const data = await res.json();
            const files = data.files || [];

            let html = `
                <div class="batch-actions">
                    <input type="file" id="batchUploadInput" multiple style="display:none">
                    <button id="batchUploadBtn" class="file-btn" style="background:#27ae60; color:white;">📤 批量上传</button>
                    <button id="batchDownloadBtn" class="file-btn" style="background:#3498db; color:white;">⬇️ 批量下载选中</button>
                    <button id="batchMoveBtn" class="file-btn" style="background:#f39c12; color:white;">📂 批量移动</button>
                    <button id="batchDeleteBtn" class="file-btn" style="background:#e74c3c; color:white;">🗑️ 删除选中</button>
                    <button id="batchCompareProjectBtn" class="file-btn" style="background:#16a34a; color:white;">🧠 批量对比选中</button>
                    <button id="createSubfolderBtn" class="file-btn" style="background:#f39c12; color:white;">📁 为当前选中文件夹新建子文件夹</button>
                    <input type="text" id="searchFileInput" placeholder="搜索文件名..." style="padding:4px 8px; border-radius:4px;">
                </div>
            `;

            if (files.length === 0) {
                html += `<p style="margin-top: 16px;">此文件夹为空。使用上方按钮上传文件。</p>`;
            } else {
                html += `
                    <div class="file-list">
                        <table style="width:100%; border-collapse: collapse;">
                            <thead>
                                <tr class="file-item">
                                    <th class="file-checkbox"><input type="checkbox" id="selectAllCheckbox"></th>
                                    <th>文件名</th>
                                    <th>大小</th>
                                    <th>状态</th>
                                    <th>上传人</th>
                                    <th>上传时间</th>
                                    <th>操作</th>
                                </tr>
                            </thead>
                            <tbody id="fileTableBody"></tbody>
                        </table>
                    </div>
                `;
            }
            container.innerHTML = html;

            if (files.length > 0) {
                const tbody = document.getElementById('fileTableBody');
                for (const f of files) {
                    const row = tbody.insertRow();
                    row.className = 'file-item';
                    row.setAttribute('data-id', f.id);
                    row.insertCell(0).innerHTML = `<input type="checkbox" class="file-select" data-id="${f.id}">`;
                    const nameCell = row.insertCell(1);
                    nameCell.className = 'file-name-cell';

                    let versionIndicator = '';
                    if (f.has_versions) {
                        versionIndicator = `<span class="version-history-indicator" data-id="${f.id}" style="margin-left:8px; cursor:pointer; background:#ff9800; color:white; border-radius:12px; padding:2px 8px; font-size:0.65rem;" title="此文件有旧版本。点击查看历史。">📜 旧版本</span>`;
                    }
                    nameCell.innerHTML = `
                        <span class="file-name">${escapeHtml(f.original_name)}</span>
                        <span class="version-badge" title="版本 ${f.version}">v${f.version}</span>
                        ${versionIndicator}
                    `;

                    row.insertCell(2).innerHTML = `${f.file_size_kb} KB`;
                    const statusCell = row.insertCell(3);
                    const st = f.status || 'draft';
                    const stColors = {final:'#38a169', draft:'#a0aec0'};
                    const stLabels = {final:'定稿', draft:'草稿'};
                    statusCell.innerHTML = `<span style="font-size:.6rem;background:${stColors[st]||'#a0aec0'};color:#fff;padding:1px 5px;border-radius:3px;cursor:pointer;" title="点击切换状态" onclick="event.stopPropagation();fetch('/admin/projects/${projectId}/files/${f.id}/status',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:st==='final'?'draft':'final'}),credentials:'include'}).then(r=>r.json()).then(()=>loadFilesInFolder(projectId,folderId)).catch(()=>{})">${stLabels[st]||st}</span>`;
                    row.insertCell(4).innerHTML = escapeHtml(f.uploaded_by_name || f.uploaded_by);
                    row.insertCell(5).innerHTML = f.uploaded_at_str;
                    const actionsCell = row.insertCell(6);
                    actionsCell.className = 'action-buttons';
                    let buttons = '';
                    if (f.can_download !== false) buttons += `<button class="download-file" data-id="${f.id}" style="background:#3498db; color:white; border:none; border-radius:4px; padding:4px 8px;">⬇️ 下载</button>`;
                    buttons += `<button class="comment-file" data-id="${f.id}" style="background:#9b59b6; color:white; border:none; border-radius:4px; padding:4px 8px;">💬 备注</button>`;
                    buttons += `<button class="generate-project-skill" data-id="${f.id}" style="background:#16a34a; color:white; border:none; border-radius:4px; padding:4px 8px; font-size:0.78rem;" title="为此文件提取技能框架并保存至个人知识库">🧠 提取技能</button>`;
                    if (f.can_move !== false) buttons += `<button class="move-file" data-id="${f.id}" style="background:#2c3e50; color:white; border:none; border-radius:4px; padding:4px 8px;">📂 移动</button>`;
                    if (f.can_delete) buttons += `<button class="delete-file" data-id="${f.id}" style="background:#e74c3c; color:white; border:none; border-radius:4px; padding:4px 8px;">🗑️ 删除</button>`;
                    actionsCell.innerHTML = buttons;
                }
            }
            const fileTable = document.querySelector('#fileListContainer table');
            if (fileTable) {
                new FileListManager(fileTable, {
                    selectableClass: '.file-item',
                    onDoubleClick: (item) => {
                        const fileId = item.querySelector('.download-file')?.dataset.id;
                        if (fileId) downloadFile(fileId);
                    }
                });
            }

            document.getElementById('batchUploadBtn').onclick = () => document.getElementById('batchUploadInput').click();
            document.getElementById('batchUploadInput').onchange = async (e) => {
                const uploadFiles = Array.from(e.target.files);
                const conflicts = [];
                const nonConflictUploads = [];

                for (const file of uploadFiles) {
                    const result = await uploadFileToFolder(projectId, folderId, file);
                    if (result && result.conflict) {
                        conflicts.push({
                            index: conflicts.length,
                            conflict_type: result.conflict_type,
                            existing_file: result.existing_file,
                            new_filename: result.new_filename,
                            file: result.file
                        });
                    } else if (result && result.success) {
                        nonConflictUploads.push(file.name);
                    }
                }

                if (conflicts.length > 0) {
                    const panelResult = await showBatchConflictPanel(conflicts);
                    if (panelResult.applied) {
                        for (const r of panelResult.results) {
                            const c = conflicts[r.index];
                            if (r.action === 'keep') {
                                // Skip — keep existing file
                                continue;
                            } else if (r.action === 'replace') {
                                // Upload as new version of the existing file
                                const formData = new FormData();
                                formData.append('file', c.file);
                                try {
                                    await fetch(`/admin/projects/${projectId}/files/${c.existing_file.id}/new_version`, {
                                        method: 'POST',
                                        credentials: 'include',
                                        body: formData
                                    });
                                } catch (err) {
                                    console.error('Replace failed:', c.new_filename, err);
                                }
                            } else if (r.action === 'rename') {
                                // Rename the existing file then upload new one
                                try {
                                    await fetch(`/admin/projects/${projectId}/files/${c.existing_file.id}/rename`, {
                                        method: 'PUT',
                                        headers: { 'Content-Type': 'application/json' },
                                        credentials: 'include',
                                        body: JSON.stringify({ original_name: c.new_filename })
                                    });
                                    // Now re-upload the new file
                                    const formData = new FormData();
                                    formData.append('file', c.file);
                                    await fetch(`/admin/projects/${projectId}/folders/${folderId}/upload`, {
                                        method: 'POST',
                                        credentials: 'include',
                                        body: formData
                                    });
                                } catch (err) {
                                    console.error('Rename+upload failed:', c.new_filename, err);
                                }
                            }
                        }
                    }
                }

                await loadFilesInFolder(projectId, folderId);
            };
            document.getElementById('batchDownloadBtn').onclick = () => batchDownloadFiles(projectId);
            document.getElementById('batchCompareProjectBtn').onclick = () => batchCompareSelectedProjectFiles(projectId);
            document.getElementById('selectAllCheckbox')?.addEventListener('change', (e) => {
                document.querySelectorAll('.file-select').forEach(cb => cb.checked = e.target.checked);
            });
            document.getElementById('searchFileInput').oninput = debounce(async (e) => {
                const query = e.target.value.trim();
                if (query.length >= 2) await searchFiles(projectId, query);
                else await loadFilesInFolder(projectId, folderId);
            }, 500);
            document.getElementById('createSubfolderBtn')?.addEventListener('click', () => {
                if (!currentFolderId) { alert('请先选择一个文件夹'); return; }
                createSubfolder(currentProjectId, currentFolderId);
            });
            document.getElementById('batchDeleteBtn')?.addEventListener('click', async () => {
                const selectedIds = Array.from(document.querySelectorAll('.file-select:checked')).map(cb => cb.dataset.id);
                if (selectedIds.length === 0) { alert('请先选择要删除的文件'); return; }
                if (!await confirm(`确定要删除 ${selectedIds.length} 个文件吗？回收站保存时间3天。`)) return;
                let failed = false;
                for (const fileId of selectedIds) {
                    const res = await fetch(`/admin/projects/${currentProjectId}/files/${fileId}`, { method: 'DELETE', credentials: 'include' });
                    if (!res.ok) failed = true;
                }
                if (failed) alert('部分文件删除失败（可能无权限）');
                await loadFilesInFolder(currentProjectId, currentFolderId);
            });
            document.getElementById('batchMoveBtn').onclick = async () => {
                const selectedCheckboxes = document.querySelectorAll('.file-select:checked');
                const selectedIds = Array.from(selectedCheckboxes).map(cb => cb.dataset.id);
                if (selectedIds.length === 0) {
                    alert('请先选择要移动的文件');
                    return;
                }
                const res = await fetch(`/admin/projects/${projectId}/folders`, { credentials: 'include' });
                const data = await res.json();
                const folders = data.folders || [];

                function renderFolderOptions(folderList, level = 0) {
                    let html = '';
                    for (const f of folderList) {
                        html += `<option value="${f.id}">${'—'.repeat(level)} ${escapeHtml(f.name)}</option>`;
                        if (f.children) html += renderFolderOptions(f.children, level + 1);
                    }
                    return html;
                }
                let options = renderFolderOptions(folders);
                if (!options) options = '<option disabled>没有可用的文件夹</option>';

                const modalHtml = `
                    <div class="custom-modal-overlay" id="batchMoveModal">
                        <div class="custom-modal" style="max-width: 500px;">
                            <h3>批量移动文件</h3>
                            <p>将 ${selectedIds.length} 个文件移动到：</p>
                            <select id="targetFolderSelect" style="width:100%; padding:8px; margin:10px 0;">${options}</select>
                            <div class="custom-modal-buttons">
                                <button class="confirm" id="batchMoveConfirm">确认移动</button>
                                <button class="cancel" id="batchMoveCancel">取消</button>
                            </div>
                        </div>
                    </div>
                `;
                const modalContainer = document.createElement('div');
                modalContainer.innerHTML = modalHtml;
                document.body.appendChild(modalContainer);
                const modal = document.getElementById('batchMoveModal');
                const confirmBtn = document.getElementById('batchMoveConfirm');
                const cancelBtn = document.getElementById('batchMoveCancel');

                confirmBtn.onclick = async () => {
                    const targetFolderId = document.getElementById('targetFolderSelect').value;
                    if (!targetFolderId) {
                        alert('请选择目标文件夹');
                        return;
                    }
                    confirmBtn.disabled = true;
                    confirmBtn.textContent = '移动中...';
                    const moveRes = await fetch(`/admin/projects/${projectId}/files/batch_move`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        credentials: 'include',
                        body: JSON.stringify({ file_ids: selectedIds, folder_id: targetFolderId })
                    });
                    if (moveRes.ok) {
                        showToast(`成功移动 ${selectedIds.length} 个文件`, 'success', 3000);
                        modal.remove();
                        await loadFilesInFolder(projectId, currentFolderId);
                        await loadFolderTree(projectId, currentFolderId);
                    } else {
                        const err = await moveRes.json();
                        alert('移动失败: ' + (err.error || '未知错误'));
                        confirmBtn.disabled = false;
                        confirmBtn.textContent = '确认移动';
                    }
                };
                cancelBtn.onclick = () => modal.remove();
            };
            document.querySelectorAll('.version-history-indicator').forEach(indicator => {
                indicator.onclick = (e) => {
                    e.stopPropagation();
                    const fileId = indicator.dataset.id;
                    const fileName = indicator.closest('tr')?.cells[1]?.querySelector('.file-name')?.innerText || '文件';
                    showVersionHistory(fileId, fileName);
                };
            });
            attachFileListEvents();
        } catch (err) {
            console.error(err);
            container.innerHTML = '<p>加载文件失败</p>';
        }
    }

    async function uploadFileToFolder(projectId, folderId, file, category, status) {
        showProgress(`上传中: ${file.name}`, 'bar');
        updateProgress(10, `正在处理 ${file.name}...`);
        const formData = new FormData();
        formData.append('file', file);
        if (category) formData.append('category', category);
        if (status) formData.append('status', status);
        updateProgress(30, '正在传输...');
        let res, data;
        try {
            res = await fetch(`/admin/projects/${projectId}/folders/${folderId}/upload`, {
                method: 'POST',
                credentials: 'include',
                body: formData
            });
            data = await res.json();
        } catch (err) {
            finishProgress(false, '网络错误，请重试');
            return null;
        }
        updateProgress(70, '正在保存...');
        // ok() wraps payload in data.data — check nested first, then flat for non-ok() endpoints
        const payload = (data.data && typeof data.data === 'object') ? data.data : data;
        if (payload.duplicate) {
            finishProgress(false, '发现重复文件');
            return {
                conflict: true,
                conflict_type: payload.conflict_type || 'hash',
                existing_file: payload.existing_file,
                new_filename: payload.new_filename,
                file: file
            };
        }
        if (res.ok && data.success) {
            finishProgress(true);
            return { success: true };
        }
        finishProgress(false, data.error || '上传失败，请检查文件格式后重试');
        return null;
    }

    function getCurrentProjectId() {
        return window._currentProjectId;
    }

    async function showBatchConflictPanel(conflicts) {
        return new Promise((resolve) => {
            const modal = document.createElement('div');
            modal.className = 'custom-modal-overlay';
            modal.id = 'batchConflictPanel';

            function sizeFmt(s) {
                if (!s) return '?';
                return s > 1024*1024 ? (s/(1024*1024)).toFixed(1)+'MB' : s > 1024 ? (s/1024).toFixed(1)+'KB' : s+'B';
            }

            const rows = conflicts.map((c, i) => {
                const ef = c.existing_file;
                const newSize = c.file ? c.file.size : 0;
                return `
                    <div class="conflict-pair" data-index="${i}" data-action="pending">
                        <div class="conflict-pair-header">
                            <span class="conflict-badge ${c.conflict_type === 'hash' ? 'badge-hash' : 'badge-name'}">${c.conflict_type === 'hash' ? '内容相同' : '同名文件'}</span>
                            <span class="conflict-pair-name">${escapeHtml(c.new_filename)}</span>
                        </div>
                        <div class="conflict-cards">
                            <div class="conflict-card card-existing">
                                <div class="conflict-card-label">已存在</div>
                                <div class="conflict-card-name">${escapeHtml(ef.original_name)}</div>
                                <div class="conflict-card-meta">v${ef.version} · ${sizeFmt(ef.file_size)}</div>
                            </div>
                            <div class="conflict-card card-new">
                                <div class="conflict-card-label">新上传</div>
                                <div class="conflict-card-name">${escapeHtml(c.new_filename)}</div>
                                <div class="conflict-card-meta">${sizeFmt(newSize)}</div>
                            </div>
                        </div>
                        <div class="conflict-actions">
                            <button class="conflict-btn btn-keep" data-action="keep">保留已有</button>
                            <button class="conflict-btn btn-replace" data-action="replace">替换为新</button>
                            <button class="conflict-btn btn-rename" data-action="rename">重命名新文件</button>
                            ${c.conflict_type === 'name' ? '<button class="conflict-btn btn-compare" data-action="compare">比较内容</button>' : ''}
                        </div>
                        <div class="conflict-compare-view" style="display:none;"></div>
                    </div>`;
            }).join('');

            modal.innerHTML = `
                <div class="custom-modal conflict-panel" style="max-width:720px;max-height:85vh;overflow-y:auto;">
                    <h3>发现 ${conflicts.length} 个文件冲突</h3>
                    <p style="font-size:0.8rem;color:var(--card-muted);margin-bottom:12px;">请为每个冲突选择处理方式，或使用批量操作</p>
                    <div class="conflict-bulk-actions">
                        <button id="bulkKeepAll" class="confirm" style="margin-right:8px;">保留所有已有</button>
                        <button id="bulkReplaceAll" class="cancel">替换所有为新</button>
                    </div>
                    <div class="conflict-pairs-list">${rows}</div>
                    <div class="custom-modal-buttons" style="margin-top:16px;">
                        <button id="conflictApplyBtn" class="confirm">应用选择</button>
                        <button id="conflictCancelBtn" class="cancel">取消全部上传</button>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);

            // Per-pair: highlight active action
            modal.querySelectorAll('.conflict-pair').forEach(pair => {
                const idx = parseInt(pair.dataset.index);
                pair.querySelectorAll('.conflict-btn[data-action]').forEach(btn => {
                    btn.onclick = async () => {
                        const action = btn.dataset.action;
                        if (action === 'compare') {
                            await doContentCompare(pair, conflicts[idx]);
                            return;
                        }
                        pair.dataset.action = action;
                        pair.querySelectorAll('.conflict-btn').forEach(b => b.classList.remove('active'));
                        btn.classList.add('active');
                    };
                });
            });

            // Bulk: set all pairs
            modal.querySelector('#bulkKeepAll').onclick = () => {
                modal.querySelectorAll('.conflict-pair').forEach(p => {
                    p.dataset.action = 'keep';
                    p.querySelectorAll('.conflict-btn').forEach(b => b.classList.remove('active'));
                    p.querySelector('.btn-keep').classList.add('active');
                });
            };
            modal.querySelector('#bulkReplaceAll').onclick = () => {
                modal.querySelectorAll('.conflict-pair').forEach(p => {
                    p.dataset.action = 'replace';
                    p.querySelectorAll('.conflict-btn').forEach(b => b.classList.remove('active'));
                    p.querySelector('.btn-replace').classList.add('active');
                });
            };

            // Apply / Cancel
            modal.querySelector('#conflictApplyBtn').onclick = () => {
                const results = [];
                modal.querySelectorAll('.conflict-pair').forEach(pair => {
                    const action = pair.dataset.action === 'pending' ? 'keep' : pair.dataset.action;
                    results.push({ index: parseInt(pair.dataset.index), action });
                });
                modal.remove();
                resolve({ applied: true, results });
            };
            modal.querySelector('#conflictCancelBtn').onclick = () => {
                modal.remove();
                resolve({ applied: false, results: [] });
            };
        });

        async function doContentCompare(pairEl, conflict) {
            const viewEl = pairEl.querySelector('.conflict-compare-view');
            viewEl.style.display = 'block';
            viewEl.innerHTML = '<div class="compare-loading">正在提取文本内容...</div>';

            // Get existing file content from server
            let existingText = '';
            try {
                const ef = conflict.existing_file;
                const projectId = getCurrentProjectId();
                const res = await fetch(`/admin/projects/${projectId}/files/${ef.id}/content`, {
                    credentials: 'include'
                });
                if (res.ok) {
                    const d = await res.json();
                    existingText = d.text || '';
                }
            } catch (err) {
                existingText = '[加载失败]';
            }

            // Get new file content client-side via FileReader
            let newText = '';
            const file = conflict.file;
            if (file) {
                const ext = (file.name || '').split('.').pop().toLowerCase();
                const textExts = ['txt', 'md', 'csv', 'json', 'xml', 'html', 'css', 'js', 'py', 'log', 'yaml', 'yml', 'toml', 'ini', 'cfg'];
                if (textExts.includes(ext)) {
                    try {
                        newText = await new Promise((res, rej) => {
                            const reader = new FileReader();
                            reader.onload = () => res(reader.result);
                            reader.onerror = () => rej(reader.error);
                            reader.readAsText(file);
                        });
                    } catch (err) {
                        newText = '[文件读取失败]';
                    }
                } else {
                    newText = `[二进制文件 (${ext || '未知类型'}) — 无法在浏览器中预览，请根据文件大小和元数据判断]`;
                }
            } else {
                newText = '[文件不可用]';
            }

            viewEl.innerHTML = `
                <div class="compare-side-by-side">
                    <div class="compare-pane">
                        <div class="compare-pane-header">已存在: ${escapeHtml(conflict.existing_file.original_name)} (v${conflict.existing_file.version})</div>
                        <pre class="compare-content">${escapeHtml(existingText)}</pre>
                    </div>
                    <div class="compare-pane">
                        <div class="compare-pane-header">新文件: ${escapeHtml(conflict.new_filename)}</div>
                        <pre class="compare-content">${escapeHtml(newText)}</pre>
                    </div>
                </div>`;
        }
    }

    async function batchDownloadFiles(projectId) {
        const selectedIds = Array.from(document.querySelectorAll('.file-select:checked')).map(cb => cb.dataset.id);
        if (selectedIds.length === 0) { alert('请先选择要下载的文件'); return; }
        const res = await fetch(`/admin/projects/${projectId}/batch_download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ file_ids: selectedIds })
        });
        if (res.ok) {
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `project_${projectId}_files.zip`;
            a.click();
            window.URL.revokeObjectURL(url);
        } else {
            const err = await res.json();
            alert('下载失败: ' + (err.error || '未知错误'));
        }
    }

    async function batchCompareSelectedProjectFiles(projectId) {
        const selected = Array.from(document.querySelectorAll('.file-select:checked'));
        if (selected.length < 2) { alert('请至少勾选2个文件进行批量对比'); return; }
        if (selected.length > 10) { alert('最多选择10个文件'); return; }
        showToast('正在准备文件...', 'info', 2000);
        const formData = new FormData();
        for (const cb of selected) {
            const fileId = cb.dataset.id;
            const filename = cb.dataset.filename || `file_${fileId}`;
            // Download each file from project and re-upload to batch compare
            try {
                const dlRes = await fetch(`/admin/projects/${projectId}/files/${fileId}/download`, { credentials: 'include' });
                if (!dlRes.ok) continue;
                const blob = await dlRes.blob();
                formData.append('files', new File([blob], filename));
            } catch(_) { continue; }
        }
        if (formData.getAll('files').length < 2) { alert('无法获取足够的文件内容'); return; }
        formData.append('project_id', projectId || '');
        try {
            const res = await fetch('/clearance/run', { method: 'POST', credentials: 'include', body: formData });
            let data = {};
            try { data = await res.json(); } catch (_) { data = { error: '服务器错误 (' + res.status + ')' }; }
            if (res.ok) {
                showToast(`清标已启动！${data.file_count}个文件，task_id=${data.task_id}`, 'success', 3000);
                // Refresh batch history if the modal is open
                if (typeof loadBatchHistory === 'function') loadBatchHistory();
            } else {
                alert(data.error || '清标启动失败');
            }
        } catch(_) { alert('网络错误'); }
    }

    async function searchFiles(projectId, query) {
        const container = document.getElementById('fileListContainer');
        const res = await fetch(`/admin/projects/${projectId}/files/search?q=${encodeURIComponent(query)}`, { credentials: 'include' });
        const data = await res.json();
        const files = data.files || [];
        if (files.length === 0) {
            container.innerHTML = '<p>未找到匹配的文件。</p>';
            return;
        }
        let html = `
            <div class="batch-actions" style="display:flex; gap:6px; flex-wrap:wrap;">
                <button id="batchDownloadBtn" class="file-btn" style="background:#3498db; color:white;">⬇️ 批量下载选中</button>
                <button id="batchCompareProjectBtn" class="file-btn" style="background:#16a34a; color:white;">🧠 批量对比选中</button>
            </div>
            <div class="file-list">
                <table>
                    <thead>
                        <tr><th class="file-checkbox"><input type="checkbox" id="selectAllCheckbox"></th><th>文件名</th><th>文件夹</th><th>大小</th><th>上传时间</th><th>操作</th></tr>
                    </thead>
                    <tbody id="searchFileTableBody"></tbody>
                </table>
            </div>
        `;
        container.innerHTML = html;
        const tbody = document.getElementById('searchFileTableBody');
        for (const f of files) {
            const row = tbody.insertRow();
            row.insertCell(0).innerHTML = `<input type="checkbox" class="file-select" data-id="${f.id}">`;
            row.insertCell(1).innerHTML = escapeHtml(f.original_name);
            row.insertCell(2).innerHTML = escapeHtml(f.folder_name || '根目录');
            row.insertCell(3).innerHTML = `${f.file_size_kb} KB`;
            row.insertCell(4).innerHTML = new Date(f.uploaded_at).toLocaleString();
            const actions = row.insertCell(5);
            actions.innerHTML = `
                <button class="download-file" data-id="${f.id}">⬇️</button>
                <button class="version-history" data-id="${f.id}">📜</button>
                <button class="comment-file" data-id="${f.id}">💬</button>
            `;
        }
        document.getElementById('batchDownloadBtn').onclick = () => batchDownloadFiles(projectId);
        document.getElementById('batchCompareProjectBtn').onclick = () => batchCompareSelectedProjectFiles(projectId);
        document.getElementById('selectAllCheckbox').onchange = (e) => {
            document.querySelectorAll('.file-select').forEach(cb => cb.checked = e.target.checked);
        };
        attachFileListEvents();
    }

    // Back to projects list button
    var backToProjectsBtn = document.getElementById('backToProjectsBtn');
    if (backToProjectsBtn) {
        backToProjectsBtn.onclick = () => {
            const projectsListView = document.getElementById('projectsListView');
            const fileExplorerView = document.getElementById('fileExplorerView');
            if (projectsListView && fileExplorerView) {
                projectsListView.style.display = 'block';
                fileExplorerView.style.display = 'none';
                loadProjects();
                syncActiveTabWithView();
            }
        };
    }


    // ======================== Knowledge Lab ========================
    async function loadKnowledgeLabFiles(cachedData) {
        const container = document.getElementById('labFileList');
        container.innerHTML = '<p>加载中...</p>';
        try {
            const raw = cachedData || await (await fetch('/knowledge_lab/list', { credentials: 'include' })).json();
            const data = raw;
            const files = data.files || [];
            if (files.length === 0) {
                container.innerHTML = '<p>暂无文件。点击上方按钮上传。</p>';
                return;
            }
            let html = '<ul style="list-style:none; padding-left:0; margin:0;">';
            for (const f of files) {
                const hasSkill = !!f.has_skill;
                const escapedName = escapeHtml(f.original_name);
                const sizeKb = ((f.file_size||0)/1024).toFixed(1);
                const uploadedStr = new Date(f.uploaded_at).toLocaleString();
                const isAdmin = sessionStorage.getItem('isAdmin') === 'true';
                const renameBtn = isAdmin ? `<button class="rename-lab-file" data-id="${f.id}" data-type="lab" style="background:none;border:none;cursor:pointer;font-size:0.65rem;padding:0 2px;" title="重命名">✏️</button>` : '';
                html += `
                    <li class="file-item" data-id="${f.id}" data-filename="${escapedName}" style="margin-bottom:8px; padding:6px 8px; background: var(--card-bg, #f9f9f9); border-radius:6px; border:1px solid var(--border-color, #e0e0e0);">
                        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                            <span style="font-size:0.85rem; font-weight:500;" title="服务器文件: ${escapeHtml(f.filename)}">📄 ${escapedName}</span>${renameBtn}
                            ${hasSkill ? '<span style="font-size:0.7rem; background:#dcfce7; color:#16a34a; border-radius:8px; padding:1px 6px;">🧠 已提取技能</span>' : ''}
                            <span style="font-size:0.7rem; color:#888;">(${sizeKb} KB)</span>
                            <span style="font-size:0.7rem; color:#888;">上传于 ${uploadedStr}</span>
                            <button class="delete-lab-file" data-id="${f.id}" style="margin-left:auto; background:#e74c3c; color:white; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">删除</button>
                        </div>
                        <div style="margin-top:4px; display:flex; gap:4px; flex-wrap:wrap;">
                            <button class="view-lab-content" data-id="${f.id}" data-filename="${escapedName}" style="background:#2c3e50; color:white; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">📄 预览源文件</button>
                            ${hasSkill
                                ? `<a href="/knowledge_lab/skill/${f.id}" target="_blank" style="background:#16a34a; color:white; text-decoration:none; border-radius:4px; padding:2px 8px; font-size:0.7rem; display:inline-block;">📥 下载技能</a>`
                                : `<button class="generate-lab-skill" data-id="${f.id}" style="background:#bccfde; color:#1e293b; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">🧠 提取技能</button>`
                            }
                        </div>
                    </li>
                `;
            }
            html += '</ul>';
            container.innerHTML = html;

            // Attach rename button handlers
            document.querySelectorAll('.rename-lab-file').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const newName = await prompt('新名称:', btn.closest('.file-item')?.dataset.filename || '');
                    if (newName && newName.trim()) {
                        const url = btn.dataset.type === 'lab' ? `/knowledge_lab/rename/${btn.dataset.id}` : `/company_kb/rename/${btn.dataset.id}`;
                        const r = await fetch(url, {method:'POST',headers:{'Content-Type':'application/json'},credentials:'include',body:JSON.stringify({name:newName.trim()})});
                        if (r.ok) { loadKnowledgeLabFiles(); loadSidebarKnowledge(); }
                        else { const d=await r.json().catch(()=>({})); alert(d.error||'重命名失败'); }
                    }
                };
            });
            // Attach delete button handlers
            document.querySelectorAll('.delete-lab-file').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const fileId = btn.dataset.id;
                    if (await confirm('确定永久删除此文件吗？')) {
                        const res = await fetch(`/knowledge_lab/delete/${fileId}`, { method: 'POST', credentials: 'include' });
                        if (res.ok) {
                            showToast('删除成功', 'success', 2000);
                            loadKnowledgeLabFiles();
                            loadSidebarKnowledge();
                        } else alert('删除失败');
                    }
                };
            });

            // View source file content
            document.querySelectorAll('.view-lab-content').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const fileId = btn.dataset.id;
                    const filename = btn.dataset.filename;
                    try {
                        const res = await fetch(`/knowledge_lab/content/${fileId}`, { credentials: 'include' });
                        if (res.ok) {
                            const data = await res.json();
                            if (data.content) showContentModal(filename, data.content);
                            else alert('文件内容为空');
                        } else alert('加载失败');
                    } catch(_) { alert('加载失败'); }
                };
            });

            // Generate skill on-demand
            document.querySelectorAll('.generate-lab-skill').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const fileId = btn.dataset.id;
                    btn.disabled = true;
                    btn.textContent = '⏳ 分析中...';
                    try {
                        const res = await fetch(`/knowledge_lab/generate_skill/${fileId}`, { method: 'POST', credentials: 'include' });
                        const data = await res.json();
                        if (res.ok) {
                            showSkillFeedback(btn, data, 'knowledge_lab', parseInt(fileId));
                        } else {
                            const msg = data.error || '生成失败';
                            const hint = data.hint || '';
                            alert(msg + (hint ? '\n' + hint : ''));
                            btn.disabled = false;
                            btn.textContent = '🧠 提取技能';
                        }
                    } catch(_) {
                        alert('网络错误');
                        btn.disabled = false;
                        btn.textContent = '🧠 提取技能';
                    }
                };
            });

            // Initialize selection manager
            new FileListManager(container, {
                onDoubleClick: (item) => {
                    const fileId = item.dataset.id;
                    const filename = item.dataset.filename;
                    fetch(`/knowledge_lab/content/${fileId}`).then(r => r.json()).then(data => {
                        if (data.content) showContentModal(filename, data.content);
                        else alert('无法加载内容');
                    }).catch(() => alert('加载失败'));
                }
            });
        } catch (err) {
            console.error(err);
            container.innerHTML = '<p>加载失败</p>';
        }
    }
    function updateKnowledgeBaseButton() {
        const btn = document.getElementById('knowledgeBaseBtn');
        if (selectedKnowledgeFiles.length) {
            btn.innerHTML = `📚 知识库(${selectedKnowledgeFiles.length})`;
        } else {
            btn.innerHTML = '📚 知识库';
        }
    }
    // Personal KB category handling
    var labCatSelect = document.getElementById('labCategorySelect');
    var labCustomCat = document.getElementById('labCustomCategory');
    if (labCatSelect) {
        labCatSelect.addEventListener('change', () => {
            if (labCatSelect.value === '自定义') {
                labCustomCat.style.display = 'inline-block';
            } else {
                labCustomCat.style.display = 'none';
                labCustomCat.value = '';
            }
        });
    }
    document.getElementById('uploadLabFileBtn').onclick = () => document.getElementById('labFileInput').click();
    document.getElementById('labFileInput').addEventListener('change', async (e) => {
        const files = Array.from(e.target.files);
        if (!files.length) return;

        const container = document.getElementById('labFileList');
        if (!container) return;

        // Helper to create a temporary placeholder element
        function createPlaceholder(filename, fileSize) {
            const placeholder = document.createElement('div');
            placeholder.className = 'lab-file-placeholder';
            placeholder.style.opacity = '0.5';
            placeholder.style.padding = '8px 0';
            placeholder.style.borderBottom = '1px solid #ddd';
            placeholder.style.display = 'flex';
            placeholder.style.justifyContent = 'space-between';
            placeholder.style.alignItems = 'center';
            placeholder.innerHTML = `
                <span>⏳ 上传中: ${escapeHtml(filename)} (${(fileSize/1024).toFixed(1)} KB) <span class="spinner">⏳</span></span>
                <span style="color:#888;">处理中...</span>
            `;
            return placeholder;
        }

        // Helper to create the final file entry (replaces placeholder)
        function createFinalEntry(file) {
            const fileDiv = document.createElement('div');
            fileDiv.style.borderBottom = '1px solid #ddd';
            fileDiv.style.padding = '8px 0';
            fileDiv.style.display = 'flex';
            fileDiv.style.justifyContent = 'space-between';
            fileDiv.style.alignItems = 'center';
            const skillInfo = file.skill_generated ? 
                '<span style="font-size:.65rem;color:#22c55e;margin:0 4px;" title="技能已生成">🧠</span>' :
                '<span class="gen-skill-link" data-fileid="'+file.file_id+'" style="font-size:.65rem;color:var(--card-muted);cursor:pointer;margin:0 4px;text-decoration:underline;text-decoration-style:dotted;" title="从此文件中提取结构化知识框架">🧠提取技能</span>';
            fileDiv.innerHTML = `
                <span>📄 ${escapeHtml(file.filename)} (${(file.file_size/1024).toFixed(1)} KB) - 上传于 ${new Date(file.uploaded_at).toLocaleString()}</span>
                <span style="display:flex;align-items:center;gap:4px;">
                    ${skillInfo}
                    <button class="delete-lab-file" data-id="${file.file_id}" style="background:#e74c3c; color:white; border:none; border-radius:4px; padding:4px 8px;">删除</button>
                </span>
            `;
            // Skill extract link handler
            setTimeout(() => {
                const genLink = fileDiv.querySelector('.gen-skill-link');
                if (genLink) genLink.onclick = async () => {
                    genLink.textContent = '⏳ 提取中...';
                    genLink.style.cursor = 'default';
                    try {
                        const res = await fetch('/knowledge_lab/generate_skill/' + genLink.dataset.fileid, { method: 'POST', credentials: 'include' });
                        const data = await res.json();
                        if (res.ok) { showSkillFeedback(genLink, data, 'knowledge_lab', parseInt(genLink.dataset.fileid)); }
                        else { genLink.textContent = '重试'; genLink.style.cursor = 'pointer'; genLink.style.color = '#ef4444'; showToast(data.error || '提取失败', 'error'); }
                    } catch(e) { genLink.textContent = '重试'; genLink.style.cursor = 'pointer'; genLink.style.color = '#ef4444'; }
                };
            }, 50);
            // Attach delete handler
            const delBtn = fileDiv.querySelector('.delete-lab-file');
            delBtn.onclick = async () => {
                if (await confirm('确定永久删除此文件吗？')) {
                    const res = await fetch(`/knowledge_lab/delete/${file.file_id}`, { method: 'POST', credentials: 'include' });
                    if (res.ok) {
                        fileDiv.remove();
                        showToast('删除成功', 'success', 2000);
                    } else {
                        alert('删除失败');
                    }
                }
            };
            return fileDiv;
        }

        // Insert placeholders at the top of the list
        const placeholders = [];
        for (const file of files) {
            const placeholder = createPlaceholder(file.name, file.size);
            container.prepend(placeholder);
            placeholders.push({ file, placeholder });
        }

        // Upload each file sequentially (to avoid overwhelming the server)
        let uploaded = 0;
        let hadErrors = false;
        showProgress(`上传 ${files.length} 个文件...`, 'bar');
        for (const item of placeholders) {
            const { file, placeholder } = item;
            const formData = new FormData();
            formData.append('file', file);
            let labCat = labCatSelect ? labCatSelect.value : '';
            if (labCat === '自定义') labCat = labCustomCat ? labCustomCat.value.trim() : '';
            if (labCat) formData.append('category', labCat);
            try {
                updateProgress((uploaded / files.length) * 90, `上传中: ${file.name} (${uploaded+1}/${files.length})`);
                const res = await fetch('/knowledge_lab/upload', { method: 'POST', credentials: 'include', body: formData });
                const data = await res.json();
                if (res.ok && data.success) {
                    // Replace placeholder with final entry
                    const finalEntry = createFinalEntry({
                        file_id: data.file_id,
                        filename: data.filename,
                        file_size: data.file_size,
                        uploaded_at: data.uploaded_at
                    });
                    placeholder.replaceWith(finalEntry);
                    showToast(`✅ ${data.filename} 上传成功`, 'success', 2000);
                } else {
                    // Show error and remove the placeholder
                    placeholder.remove();
                    hadErrors = true;
                    showToast('上传失败，请检查文件格式', 'error', 3000);
                }
            } catch (err) {
                placeholder.remove();
                hadErrors = true;
                showToast('上传失败，请检查网络连接后重试', 'error', 3000);
            }
            uploaded++;
        }
        finishProgress(!hadErrors, hadErrors ? '部分文件上传失败' : undefined);

        // Clear the input so the same files can be uploaded again
        e.target.value = '';
    });
    document.getElementById('refreshLabListBtn').onclick = loadKnowledgeLabFiles;
    document.getElementById('refreshSkillOverviewBtn').onclick = loadSkillOverview;

    // ── Notebook ──
    async function loadNotebook() {
        const list = document.getElementById('notebookList'); if (!list) return;
        try {
            const r = await fetch('/notebook', {credentials:'include'});
            const d = await r.json();
            const notes = d.notes || [];
            if (!notes.length) { list.innerHTML = '<span style="font-size:0.75rem;color:var(--card-muted);">暂无笔记。点击➕新建笔记。</span>'; return; }
            list.innerHTML = notes.map(n => `<div style="background:var(--card-bg);border:1px solid var(--card-border);border-radius:6px;padding:8px;cursor:pointer;" onclick="_openNotebook('${escapeHtml(n.id)}')">
                <div style="font-weight:600;font-size:0.75rem;">📝 ${escapeHtml(n.id)}</div>
                <div style="font-size:0.62rem;color:var(--card-muted);margin-top:2px;">${escapeHtml(n.preview||'')}</div>
                <div style="font-size:0.55rem;color:var(--card-muted);margin-top:4px;display:flex;justify-content:space-between;">
                    <span>${new Date(n.modified).toLocaleString()}</span>
                    <button onclick="event.stopPropagation();_deleteNotebook('${escapeHtml(n.id)}')" style="color:#ef4444;background:none;border:none;font-size:0.55rem;cursor:pointer;">🗑</button>
                </div>
            </div>`).join('');
        } catch(_) { list.innerHTML = '<span style="color:#ef4444;font-size:0.75rem;">加载失败</span>'; }
    }

    window._openNotebook = async function(noteId) {
        const r = await fetch('/notebook/' + encodeURIComponent(noteId), {credentials:'include'});
        const d = await r.json();
        const note = d.note;
        document.getElementById('notebookEditTitle').value = note.id;
        document.getElementById('notebookEditContent').value = note.content;
        document.getElementById('notebookEditor').style.display = 'block';
        document.getElementById('notebookSummary').innerHTML = '';
    };

    window._deleteNotebook = async function(noteId) {
        if (!confirm('删除笔记 "'+noteId+'"？')) return;
        await fetch('/notebook/' + encodeURIComponent(noteId), {method:'DELETE', credentials:'include'});
        loadNotebook();
    };

    document.getElementById('notebookNewBtn').onclick = () => {
        document.getElementById('notebookEditTitle').value = '';
        document.getElementById('notebookEditContent').value = '';
        document.getElementById('notebookEditor').style.display = 'block';
        document.getElementById('notebookSummary').innerHTML = '';
    };
    document.getElementById('notebookRefreshBtn').onclick = loadNotebook;
    document.getElementById('notebookCancelBtn').onclick = () => {
        document.getElementById('notebookEditor').style.display = 'none';
    };
    document.getElementById('notebookSaveBtn').onclick = async () => {
        const title = document.getElementById('notebookEditTitle').value.trim();
        const content = document.getElementById('notebookEditContent').value;
        if (!title) { alert('标题不能为空'); return; }
        const r = await fetch('/notebook/' + encodeURIComponent(title), {
            method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
            body:JSON.stringify({content})
        });
        if (r.ok) { document.getElementById('notebookEditor').style.display = 'none'; loadNotebook(); }
        else alert('保存失败');
    };
    document.getElementById('notebookSummarizeBtn').onclick = async () => {
        const title = document.getElementById('notebookEditTitle').value.trim();
        if (!title) return;
        const el = document.getElementById('notebookSummary');
        el.innerHTML = '' + _icon('hourglass_empty') + ' AI摘要生成中...';
        const r = await fetch('/notebook/' + encodeURIComponent(title) + '/summarize', {method:'POST', credentials:'include'});
        const d = await r.json();
        el.innerHTML = d.summary ? '🤖 ' + escapeHtml(d.summary) : '失败';
    };
    document.getElementById('notebookSearch').addEventListener('keydown', async (e) => {
        if (e.key !== 'Enter') return;
        const q = e.target.value.trim();
        if (!q) { loadNotebook(); return; }
        const r = await fetch('/notebook/search', {
            method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
            body:JSON.stringify({query:q})
        });
        const d = await r.json();
        const results = d.results || [];
        const list = document.getElementById('notebookList');
        if (!results.length) { list.innerHTML = '<span style="font-size:0.75rem;color:var(--card-muted);">无匹配结果。</span>'; return; }
        list.innerHTML = results.map(r => `<div style="background:var(--card-bg);border:1px solid var(--card-border);border-radius:6px;padding:6px;cursor:pointer;" onclick="_openNotebook('${escapeHtml(r.note_id)}')">
            <div style="font-weight:600;font-size:0.7rem;">📝 ${escapeHtml(r.note_id)} (${(r.score*100).toFixed(0)}%)</div>
            <div style="font-size:0.6rem;color:var(--card-muted);">${escapeHtml(r.snippet||'')}</div>
        </div>`).join('');
    });
    loadNotebook();
    function showContentModal(title, content) {
        const existing = document.getElementById('contentPreviewModal');
        if (existing) existing.remove();

        const modal = document.createElement('div');
        modal.id = 'contentPreviewModal';
        modal.className = 'modal';
        modal.style.display = 'block';
        modal.innerHTML = `
            <div class="modal-content" style="width: 80%; max-width: 800px; max-height: 80vh; overflow-y: auto;">
                <span class="close" style="float: right; cursor: pointer;">&times;</span>
                <h4>${escapeHtml(title)}</h4>
                <div style="white-space: pre-wrap; font-family: monospace; font-size: 0.85rem; line-height: 1.5; background: #f5f5f5; padding: 12px; border-radius: 8px; margin-top: 12px;">
                    ${escapeHtml(content).replace(/\n/g, '<br>')}
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        modal.querySelector('.close').onclick = () => modal.remove();
        modal.onclick = (e) => { if (e.target === modal) modal.remove(); };
    }
    function renderCompanyKb(container, data) {
        const files = data.files || [];
        if (!files.length) { container.innerHTML = '<p>暂无公司知识库文件。</p>'; return; }
        const isAdmin = sessionStorage.getItem('isAdmin') === 'true';
        let html = '<ul style="list-style:none; padding-left:0; margin:0;">';
        for (const f of files) {
            const hasSkill = !!f.has_skill;
            const escapedName = escapeHtml(f.filename);
            const sizeKb = ((f.file_size||0)/1024).toFixed(1);
            const uploadedStr = new Date(f.uploaded_at).toLocaleString();
            html += `<li class="file-item" data-id="${f.id}" data-filename="${escapedName}" style="margin-bottom:8px; padding:6px 8px; background: var(--card-bg, #f9f9f9); border-radius:6px; border:1px solid var(--border-color, #e0e0e0);">
                <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <span style="font-size:0.85rem; font-weight:500;">📄 ${escapedName}</span>${isAdmin ? `<button class="rename-company-file" data-id="${f.id}" style="background:none;border:none;cursor:pointer;font-size:0.65rem;padding:0 2px;" title="重命名">✏️</button>` : ''}
                    ${hasSkill ? '<span style="font-size:0.7rem; background:#dcfce7; color:#16a34a; border-radius:8px; padding:1px 6px;">🧠 已提取技能</span>' : ''}
                    <span style="font-size:0.7rem; color:#888;">(${sizeKb} KB)</span>
                    <span style="font-size:0.7rem; color:#888;">分类: ${escapeHtml(f.category || '无')}</span>
                    <span style="font-size:0.7rem; color:#888;">${escapeHtml(f.uploaded_by_name || 'admin')}</span>
                    <span style="font-size:0.7rem; color:#888;">${uploadedStr}</span>
                    ${isAdmin ? `<button class="delete-company-file" data-id="${f.id}" style="margin-left:auto; background:#e74c3c; color:white; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">删除</button>` : ''}
                </div>
                <div style="margin-top:4px; display:flex; gap:4px; flex-wrap:wrap;">
                    <button class="view-company-content" data-id="${f.id}" data-filename="${escapedName}" style="background:#2c3e50; color:white; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">📄 预览源文件</button>
                    ${hasSkill
                        ? `<a href="/company_kb/skill/${f.id}" target="_blank" style="background:#16a34a; color:white; text-decoration:none; border-radius:4px; padding:2px 8px; font-size:0.7rem; display:inline-block;">📥 下载技能</a>`
                        : (isAdmin ? `<button class="generate-company-skill" data-id="${f.id}" style="background:#bccfde; color:#1e293b; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">🧠 提取技能</button>` : '')
                    }
                    ${hasSkill && f.category === '模板'
                        ? `<button class="generate-template-doc" data-id="${f.id}" data-source="company_knowledge_base" style="background:#f59e0b; color:white; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">📝 生成文档</button>`
                        : ''
                    }
                </div>
            </li>`;
        }
        container.innerHTML = html + '</ul>';
        wireCompanyKbButtons();
    }

    async function loadCompanyKnowledgeBase(cachedData) {
        const container = document.getElementById('companyKbList');
        if (!container) return;
        const search = document.getElementById('companyKbSearch')?.value.trim() || '';
        const category = document.getElementById('companyKbCategoryFilter')?.value || '';
        if (cachedData && !search && !category) { renderCompanyKb(container, cachedData); return; }
        container.innerHTML = '<p>加载中...</p>';
        let url = `/company_kb/list?`;
        if (search) url += `search=${encodeURIComponent(search)}&`;
        if (category) url += `category=${encodeURIComponent(category)}&`;
        try {
            const res = await fetch(url, { credentials: 'include' });
            if (!res.ok) throw new Error('Failed to load company KB');
            renderCompanyKb(container, await res.json());
        } catch(e) { container.innerHTML = '<p>加载失败</p>'; }
    }

    function wireCompanyKbButtons() {
            document.querySelectorAll('.rename-company-file').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const newName = await prompt('新名称:', btn.closest('.file-item')?.dataset.filename || '');
                    if (newName && newName.trim()) {
                        const r = await fetch(`/company_kb/rename/${btn.dataset.id}`, {method:'POST',headers:{'Content-Type':'application/json'},credentials:'include',body:JSON.stringify({name:newName.trim()})});
                        if (r.ok) { loadCompanyKnowledgeBase(); loadCompanyCategories(); }
                        else { const d=await r.json().catch(()=>({})); alert(d.error||'重命名失败'); }
                    }
                };
            });
            document.querySelectorAll('.delete-company-file').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation(); const fileId = btn.dataset.id;
                    if (await confirm('确定永久删除此公司知识库文件吗？')) {
                        const res = await fetch(`/company_kb/delete/${fileId}`, { method: 'POST', credentials: 'include' });
                        if (res.ok) { showToast('删除成功', 'success', 2000); loadCompanyKnowledgeBase(); loadCompanyCategories(); }
                        else alert('删除失败');
                    }
                };
            });
            document.querySelectorAll('.view-company-content').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation(); const fileId = btn.dataset.id; const filename = btn.dataset.filename;
                    const res = await fetch(`/company_kb/content/${fileId}`, { credentials: 'include' });
                    const data = await res.json();
                    if (res.ok && data.content) showContentModal(filename, data.content); else alert('无法加载内容');
                };
            });
            document.querySelectorAll('.generate-company-skill').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation(); const fileId = btn.dataset.id;
                    btn.disabled = true; btn.textContent = '⏳ 分析中...';
                    try {
                        const res = await fetch(`/company_kb/generate_skill/${fileId}`, { method: 'POST', credentials: 'include' });
                        const d = await res.json();
                        if (res.ok) { showSkillFeedback(btn, d, 'company_kb', parseInt(fileId)); }
                        else { alert((d.error||'生成失败')+((d.hint||'')?'\n'+d.hint:'')); btn.disabled = false; btn.textContent = '🧠 提取技能'; }
                    } catch(_) { alert('网络错误'); btn.disabled = false; btn.textContent = '🧠 提取技能'; }
                };
            });
            document.querySelectorAll('.generate-template-doc').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation(); const fileId = btn.dataset.id; const source = btn.dataset.source || 'company_knowledge_base';
                    btn.disabled = true; const origText = btn.textContent; btn.textContent = '⏳ 生成中...';
                    try {
                        const formData = new FormData(); formData.append('source', source);
                        const res = await fetch(`/templates/${fileId}/generate_doc`, { method: 'POST', credentials: 'include', body: formData });
                        if (res.ok) {
                            const blob = await res.blob(); const url = URL.createObjectURL(blob);
                            const a = document.createElement('a'); a.href = url; a.download = '文档.docx'; a.click();
                            URL.revokeObjectURL(url); showToast('✅ 文档已生成', 'success', 2000);
                        } else { const d = await res.json().catch(()=>({})); alert(d.error || '生成失败'); }
                    } catch(_) { alert('网络错误'); }
                    btn.disabled = false; btn.textContent = origText;
                };
            });
        }

    async function loadCompanyCategories() {
        try {
            const res = await fetch('/company_kb/categories', { credentials: 'include' });
            const data = await res.json();
            const categories = data.categories || [];
            const select = document.getElementById('companyKbCategoryFilter');
            select.innerHTML = '<option value="">所有分类</option>';
            for (const cat of categories) {
                select.innerHTML += `<option value="${escapeHtml(cat)}">${escapeHtml(cat)}</option>`;
            }
        } catch (err) {
            console.error('Failed to load categories', err);
        }
    }

    async function loadSkillOverview(labData, coData) {
        const container = document.getElementById('skillOverviewList');
        const counter = document.getElementById('skillOverviewCount');
        if (!container) return;
        container.innerHTML = '<p style="grid-column:1/-1; color:var(--card-muted);">加载中...</p>';

        if (counter) {
            try {
                const ragRes = await fetch('/admin/rag_stats', { credentials: 'include' });
                if (ragRes.ok) {
                    const rag = await ragRes.json(); const s = rag.stats || {}; const total = s.total || 0;
                    const status = total > 0 ? `<span style="font-size:0.7rem; background:#dcfce7; color:#16a34a; border-radius:8px; padding:1px 6px; margin-left:4px;">RAG 🟢</span>` : `<span style="font-size:0.7rem; background:#fef2f2; color:#dc2626; border-radius:8px; padding:1px 6px; margin-left:4px;" title="请管理员点侧边栏重建全部索引">RAG ⚠️</span>`;
                    counter.innerHTML = status;
                }
            } catch(_) {}
        }
        try {
            if (!labData) { const r = await fetch('/knowledge_lab/list', { credentials: 'include' }); if (!r.ok) throw new Error('HTTP ' + r.status); labData = await r.json(); }
            if (!coData) { const r = await fetch('/company_kb/list', { credentials: 'include' }); if (!r.ok) throw new Error('HTTP ' + r.status); coData = await r.json(); }
            const labSkills = (labData.files || []).filter(f => f.has_skill);
            const coSkills = (coData.files || []).filter(f => f.has_skill);
            const all = [...labSkills.map(f => ({...f, _src: 'personal', _url: '/knowledge_lab/skill/'})),
                        ...coSkills.map(f => ({...f, _src: 'company', _url: '/company_kb/skill/', _name: f.filename}))];

            if (counter) counter.innerHTML = `${counter.innerHTML} (${all.length}个技能)`;
            if (!all.length) {
                container.innerHTML = '<p style="grid-column:1/-1; color:var(--card-muted);">暂无技能。上传文件后系统会自动提取，或手动点击"提取技能"按钮生成。</p>';
                return;
            }
            container.innerHTML = all.map(f => {
                const name = f._src === 'company' ? (f._name || f.original_name) : (f.original_name || f.filename);
                return `<div style="background:var(--card-bg); border:1px solid var(--card-border); border-radius:8px; padding:10px 12px; display:flex; flex-direction:column; gap:6px;">
                    <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:4px;">
                        <div style="font-size:0.85rem; font-weight:500; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${escapeHtml(name)}">
                            ${f._src === 'company' ? '🏢' : '📁'} ${escapeHtml(name)}</div>
                        <span style="font-size:0.65rem; color:var(--card-muted); flex-shrink:0;">${f._src === 'company' ? '公司' : '个人'} · ${((f.file_size||0)/1024).toFixed(0)}KB</span>
                    </div>
                    <div style="display:flex; gap:6px;">
                        <a href="${f._url}${f.id}" target="_blank" style="background:#16a34a; color:white; text-decoration:none; border-radius:4px; padding:3px 10px; font-size:0.72rem;">📥 下载技能</a>
                        ${f._src === 'personal' ? `<button class="view-lab-content" data-id="${f.id}" data-filename="${escapeHtml(name)}" style="background:#2c3e50; color:white; border:none; border-radius:4px; padding:3px 10px; font-size:0.72rem;">📄 源文件</button>` : ''}
                    </div>
                </div>`;
            }).join('');
            // Wire source-file preview buttons
            document.querySelectorAll('#skillOverviewList .view-lab-content').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const fileId = btn.dataset.id;
                    const filename = btn.dataset.filename;
                    try {
                        const res = await fetch(`/knowledge_lab/content/${fileId}`, { credentials: 'include' });
                        if (res.ok) { const d = await res.json(); if (d.content) showContentModal(filename, d.content); else alert('文件内容为空'); }
                        else alert('加载失败');
                    } catch(_) { alert('加载失败'); }
                };
            });
        } catch(e) {
            // M4 (FIX-016 后续): 区分错误码给出有意义提示，而非泛化"加载失败"
            var statusMsg = '';
            var m = /HTTP (\d+)/.exec(String(e && e.message || e));
            if (m) {
                var code = parseInt(m[1], 10);
                statusMsg = code === 401 ? '请先登录' : (code === 403 ? '权限不足' : (code >= 500 ? '服务器错误 (' + code + ')' : '请求失败 (' + code + ')'));
            } else if (e && e.name === 'TypeError') {
                statusMsg = '网络错误';
            }
            container.innerHTML = '<p style="grid-column:1/-1; color:#ef4444;">加载失败' + (statusMsg ? '：' + statusMsg : '') + '</p>';
        }
    }

    // Company Knowledge Base category handling
    var categorySelect = document.getElementById('companyCategorySelect');
    var customCategoryInput = document.getElementById('companyCustomCategory');

    if (categorySelect) {
        categorySelect.addEventListener('change', () => {
            if (categorySelect.value === '自定义') {
                customCategoryInput.style.display = 'inline-block';
                customCategoryInput.required = true;
            } else {
                customCategoryInput.style.display = 'none';
                customCategoryInput.required = false;
                customCategoryInput.value = '';
            }
        });
    }

    // Company file upload – validate category before opening file picker
    var uploadCompanyBtn = document.getElementById('uploadCompanyFileBtn');
    var companyFileInput = document.getElementById('companyFileInput');

    if (uploadCompanyBtn && companyFileInput) {
        uploadCompanyBtn.onclick = () => {
            let category = categorySelect.value;
            if (category === '自定义') {
                category = customCategoryInput.value.trim();
                if (!category) {
                    alert('请先输入自定义分类名称');
                    return;
                }
            }
            if (!category) {
                alert('请先选择或输入分类');
                return;
            }
            companyFileInput.click();
        };

        companyFileInput.onchange = async (e) => {
            const files = e.target.files;
            if (!files.length) return;

            uploadCompanyBtn.disabled = true;
            const originalText = uploadCompanyBtn.textContent;
            uploadCompanyBtn.textContent = '⏳ 上传中...';

            let category = categorySelect.value;
            if (category === '自定义') {
                category = customCategoryInput.value.trim();
            }

            for (const file of files) {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('category', category);
                try {
                    const res = await fetch('/company_kb/upload', { method: 'POST', credentials: 'include', body: formData });
                    const data = await res.json();
                    if (res.ok) {
                        showToast(`✅ ${data.filename} 上传成功 (分类: ${category})`, 'success', 2000);
                        if (data.overlap_suggestions && data.overlap_suggestions.length) {
                            setTimeout(() => showOverlapSuggestions(data.overlap_suggestions, data.file_id, 'company_knowledge_base'), 500);
                        }
                    } else {
                        showToast('上传失败，请重试', 'error', 3000);
                    }
                } catch (err) {
                    showToast('❌ 网络错误', 'error', 3000);
                }
            }
            await loadCompanyKnowledgeBase();
            await loadCompanyCategories();
            uploadCompanyBtn.disabled = false;
            uploadCompanyBtn.textContent = originalText;
            e.target.value = '';
        };
    }

    document.getElementById('refreshCompanyKbBtn').onclick = () => {
        loadCompanyKnowledgeBase();
    };
    document.getElementById('companyKbSearch').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') loadCompanyKnowledgeBase();
    });
    if (sessionStorage.getItem('isAdmin') === 'true') {
        document.getElementById('adminCompanyTools').style.display = 'block';
    }

    // ── Category filter pills for chat RAG ──
    window._selectedRagCategory = '';
    var catFilterBar = document.getElementById('categoryFilterBar');
    var catPills = document.querySelectorAll('.cat-pill');
    var catClearBtn = document.getElementById('catFilterClear');

    function showCatFilterIfNeeded() {
        const el = document.getElementById('categoryFilterBar');
        if (!el) return;
        if (selectedKnowledgeFiles.length > 0) {
            el.style.display = 'flex';
        } else {
            el.style.display = 'none';
            window._selectedRagCategory = '';
        }
    }

    if (catPills.length) {
        catPills.forEach(pill => {
            pill.addEventListener('click', () => {
                catPills.forEach(p => {
                    p.style.background = '';
                    p.style.color = '';
                });
                pill.style.background = 'var(--accent)';
                pill.style.color = 'white';
                window._selectedRagCategory = pill.dataset.cat || '';
                if (catClearBtn) catClearBtn.style.display = window._selectedRagCategory ? 'inline' : 'none';
            });
        });
    }
    if (catClearBtn) {
        catClearBtn.addEventListener('click', () => {
            catPills.forEach(p => { p.style.background = ''; p.style.color = ''; });
            const allPill = document.querySelector('.cat-pill[data-cat=""]');
            if (allPill) { allPill.style.background = 'var(--accent)'; allPill.style.color = 'white'; }
            window._selectedRagCategory = '';
            catClearBtn.style.display = 'none';
        });
    }
    // ── Skill overlap suggestions after upload ──
    function showOverlapSuggestions(overlaps, newFileId, sourceTable) {
        if (!overlaps || !overlaps.length) return;
        const msg = overlaps.map(o =>
            `📄 ${escapeHtml(o.name)} (相似度 ${o.similarity}%)`
        ).join('\n');
        if (!confirm(`检测到新上传的文件与以下 ${overlaps.length} 个同分类技能高度相似，是否合并？\n\n${msg}\n\n选择「确定」合并，「取消」忽略`)) return;
        // Merge the top overlap into the new file
        const top = overlaps[0];
        fetch('/admin/skill_supersession/respond', {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: 'merge',
                keep_id: newFileId,
                merge_id: top.id,
                source: sourceTable || 'company_knowledge_base'
            })
        }).then(r => r.json()).then(data => {
            if (data.success) showToast(`✅ 已与「${top.name}」合并`, 'success', 3000);
            else showToast('❌ 合并失败: ' + (data.error || ''), 'error', 5000);
        }).catch(() => showToast('❌ 合并请求失败', 'error', 3000));
    }

