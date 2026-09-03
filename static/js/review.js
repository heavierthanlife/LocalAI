/* Review / Admin tab module — extracted from app.js (W2 Phase 1b) */
/* Dependencies: showToast, escapeHtml, switchToPanel, switchSidebarPane,
                 saveActiveTab, currentProjectId (kernel globals) */

    // ======================== Review Tab (Admin + Auditor) ========================
    var reviewPanel = document.getElementById('reviewPanel');
    if (reviewTabBtn && reviewPanel) {
        let _reviewLoaded = { ingest: false, training: false, history: false, structured: false, workload: false };

        async function _initReviewPanel() {
            _reviewLoaded = { ingest: false, training: false, history: false, structured: false, workload: false };
            const content = document.getElementById('reviewContent');
            const sections = document.getElementById('reviewSections');
            try {
                const res = await fetch('/admin/analytics', { credentials: 'include' });
                if (res.status === 403) { content.innerHTML = '<span>需要登录</span>'; return; }
                if (!res.ok) { content.innerHTML = '<span style="color:#e74c3c;">服务器错误 (' + res.status + ')</span>'; return; }
                const stats = await res.json();
                const items = ['<span title="用户总数">' + _icon('👥') + '<b>' + stats.total_users + '</b></span>', '<span title="会话总数">' + _icon('💬') + '<b>' + stats.total_sessions + '</b></span>', '<span title="消息总数">' + _icon('✉️') + '<b>' + stats.total_messages + '</b></span>'];
                content.innerHTML = '<div style="font-size:0.7rem;color:var(--card-muted);">' + items.join(' · ') + '</div>';
                if (sections) { sections.style.display = 'block'; _setupReviewLazySections(); }
                _checkStaleReviews();
                _updateReviewSidebarStatus();
            } catch(e) { console.error('Review stats load error:', e); content.innerHTML = '<span style="color:#e74c3c;">加载失败</span>'; }
        }
        window._initReviewPanel = _initReviewPanel;

        // Check for stale reviews (pending >3 days) and warn
        async function _checkStaleReviews() {
            try {
                const r = await fetch('/admin/ingest/stale_status', {credentials:'include'});
                const d = await r.json();
                const stale = d.stale || {};
                const tasks = stale.kb_review_tasks || [];
                const oldTasks = tasks.filter(t => t.days_pending >= 3);
                if (oldTasks.length) {
                    showToast(`⚠️ ${oldTasks.length}个KB审核超过3天未处理`, 'warning', 6000);
                }
            } catch(_) {}
        }

        // Review sidebar: quick status + wiring
        async function _updateReviewSidebarStatus() {
            const statusEl = document.getElementById('sidebarReviewStatus');
            if (!statusEl) return;
            try {
                const r = await fetch('/admin/ingest/stale_status', {credentials:'include'});
                const d = await r.json();
                const s = d.stale || {};
                let parts = [];
                if (s.kb_review_tasks?.length) parts.push(`📋 ${s.kb_review_tasks.length}个KB待审`);
                if (s.domain_candidates) parts.push(`📝 ${s.domain_candidates}个词待审`);
                if (!parts.length) parts.push('✅ 暂无待审核项');
                statusEl.innerHTML = parts.join('<br>');
            } catch(_) { statusEl.textContent = '加载失败'; }
        }

        // Wire review sidebar buttons
        setTimeout(() => {
            const ingestUploadBtn = document.getElementById('sidebarIngestUploadBtn');
            const ingestFileInput = document.getElementById('sidebarIngestFileInput');
            if (ingestUploadBtn && ingestFileInput) {
                ingestUploadBtn.onclick = () => ingestFileInput.click();
                ingestFileInput.onchange = async () => {
                    const file = ingestFileInput.files[0];
                    if (!file) return;
                    const form = new FormData();
                    form.append('file', file);
                    ingestUploadBtn.disabled = true; ingestUploadBtn.textContent = '⏳ 上传中...';
                    try {
                        const r = await fetch('/admin/ingest/upload', {method:'POST',credentials:'include',body:form});
                        const d = await r.json();
                        if (r.ok) showToast('✅ 文档包上传成功，正在处理...', 'success');
                        else showToast('❌ '+(d.error||'上传失败'), 'error');
                    } catch(_) { showToast('网络错误', 'error'); }
                    ingestUploadBtn.disabled = false; ingestUploadBtn.textContent = '📥 上传文档包';
                    ingestFileInput.value = '';
                };
            }
            const viewStructuredBtn = document.getElementById('sidebarViewStructuredBtn');
            if (viewStructuredBtn) {
                viewStructuredBtn.onclick = () => {
                    const details = document.getElementById('structuredDocsDetails');
                    if (details) { details.open = true; loadStructuredDocsPanel(); details.scrollIntoView({behavior:'smooth'}); }
                };
            }
            const viewWorkloadBtn = document.getElementById('sidebarViewWorkloadBtn');
            if (viewWorkloadBtn) {
                viewWorkloadBtn.onclick = () => {
                    const details = document.getElementById('workloadDetails');
                    if (details) { details.open = true; loadWorkloadPanel(); details.scrollIntoView({behavior:'smooth'}); }
                };
            }
        }, 200);


        function _setupReviewLazySections() {
            const map = [
                ['ingestDetails', 'ingest', 'loadIngestPanel'],
                ['trainingDetails', 'training', 'loadTrainingExportPanel'],
                ['ingestHistoryDetails', 'history', 'loadIngestHistory'],
                ['structuredDocsDetails', 'structured', 'loadStructuredDocsPanel'],
                ['workloadDetails', 'workload', 'loadWorkloadPanel'],
            ];
            for (const [id, key, fnName] of map) {
                const el = document.getElementById(id);
                if (el && !el._listenerSet) {
                    el._listenerSet = true;
                    el.addEventListener('toggle', () => { if (el.open && !_reviewLoaded[key]) { _reviewLoaded[key] = true; const fn = window[fnName]; if (typeof fn === 'function') fn(); } });
                }
            }
        }
    }

    // ── Review panel helper functions ──

    async function loadIngestHistory() {
        const panel = document.getElementById('ingestHistoryPanel'); if (!panel) return;
        panel.innerHTML = '<span style="color:var(--card-muted);">加载中...</span>';
        try {
            const [staleR, structR] = await Promise.all([
                fetch('/admin/ingest/stale_status', {credentials:'include'}),
                fetch('/admin/ingest/structured', {credentials:'include'})
            ]);
            const staleD = await staleR.json();
            const structD = await structR.json();
            const stale = staleD.stale || {};
            const docs = structD.documents || [];
            let html = '<div style="font-size:0.68rem;margin-bottom:8px;">';
            if (docs.length) html += `📑 ${docs.length} 份结构化文档已提取。`;
            if (stale.kb_review_tasks?.length) html += `📋 ${stale.kb_review_tasks.length} 个KB审核待处理。`;
            if (stale.domain_candidates) html += `📝 ${stale.domain_candidates} 个领域词待审核。`;
            html += '</div>';
            if (docs.length) {
                html += '<details style="font-size:0.65rem;"><summary>结构化文档 ('+docs.length+')</summary><table style="width:100%;font-size:0.62rem;border-collapse:collapse;">';
                for (const d of docs) {
                    html += `<tr style="border-bottom:1px solid var(--card-border);"><td><b>${escapeHtml(d.project_name||d.document_type||'?')}</b></td>
                        <td>${escapeHtml(d.bid_number||'')}</td><td>${d.budget_amount_cny?d.budget_amount_cny.toLocaleString()+'\u00a5':''}</td></tr>`;
                }
                html += '</table></details>';
            }
            panel.innerHTML = html || '<span style="color:var(--card-muted);">暂无摄入历史。</span>';
        } catch(_) { panel.innerHTML = '<span style="color:#ef4444;">加载失败</span>'; }
    }

    async function loadDocReviewPanel() {
        const panel = document.getElementById('docReviewPanel'); if (!panel) return;
        const fileInput = document.getElementById('docReviewFileInput');
        const selectBtn = document.getElementById('selectDocReviewFileBtn');
        const fileName = document.getElementById('docReviewFileName');
        const runBtn = document.getElementById('runDocReviewBtn');
        const status = document.getElementById('docReviewStatus');
        const results = document.getElementById('docReviewResults');

        if (selectBtn && fileInput) {
            selectBtn.onclick = () => fileInput.click();
            fileInput.onchange = () => {
                if (fileInput.files.length) {
                    fileName.textContent = fileInput.files[0].name;
                    if (runBtn) runBtn.disabled = false;
                } else {
                    fileName.textContent = '';
                    if (runBtn) runBtn.disabled = true;
                }
            };
        }

        if (runBtn) {
            runBtn.onclick = async () => {
                if (!fileInput || !fileInput.files.length) {
                    if (status) status.textContent = '请先选择文件';
                    return;
                }
                runBtn.disabled = true;
                if (status) status.textContent = '⏳ AI正在审查...';
                if (results) { results.style.display = 'none'; results.innerHTML = ''; }

                const form = new FormData();
                form.append('file', fileInput.files[0]);

                const selectedAxes = [];
                document.querySelectorAll('.doc-review-axis:checked').forEach(cb => selectedAxes.push(cb.value));
                if (selectedAxes.length < 5) form.append('axes', selectedAxes.join(','));

                try {
                    const res = await fetch('/admin/review/document', {
                        method: 'POST',
                        credentials: 'include',
                        body: form
                    });
                    const data = await res.json();
                    if (!data.success) {
                        if (status) status.textContent = '❌ ' + (data.error || '审查失败');
                        runBtn.disabled = false;
                        return;
                    }
                    const r = data;
                    if (status) status.textContent = '✅ 审查完成';

                    let html = '';
                    if (r.scores) {
                        html += '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;">';
                        for (const [k, v] of Object.entries(r.scores)) {
                            const color = v >= 7 ? '#16a34a' : (v >= 5 ? '#d97706' : '#dc2626');
                            html += `<span style="background:${color};color:white;border-radius:6px;padding:4px 10px;font-size:0.75rem;"><b>${k}: ${v}</b></span>`;
                        }
                        html += '</div>';
                    }
                    if (r.overall) {
                        html += `<div style="font-size:0.9rem;margin-bottom:6px;">综合评分: <b style="font-size:1.1rem;">${r.overall}/10</b> — ${r.verdict||''}</div>`;
                    }
                    if (r.issues && r.issues.length) {
                        html += '<table style="width:100%;font-size:0.7rem;border-collapse:collapse;">';
                        html += '<tr style="background:var(--card-bg);"><th style="padding:4px;">维度</th><th>严重度</th><th>位置</th><th>问题</th><th>建议</th></tr>';
                        for (const iss of r.issues) {
                            const sevColor = iss.severity === '高' ? '#dc2626' : (iss.severity === '中' ? '#d97706' : '#6b7280');
                            html += `<tr style="border-top:1px solid var(--card-border);">
                                <td style="padding:4px;">${escapeHtml(iss.axis||'')}</td>
                                <td style="color:${sevColor};font-weight:600;">${escapeHtml(iss.severity||'')}</td>
                                <td>${escapeHtml(iss.location||'')}</td>
                                <td>${escapeHtml(iss.finding||'')}</td>
                                <td>${escapeHtml(iss.suggestion||'')}</td></tr>`;
                        }
                        html += '</table>';
                    }
                    if (r.summary) {
                        html += `<div style="margin-top:10px;padding:8px;background:#f8fafc;border-radius:6px;font-size:0.75rem;">📝 ${escapeHtml(r.summary)}</div>`;
                    }
                    if (r.parse_error) {
                        html += `<div style="margin-top:10px;padding:8px;background:#fef3c7;border-radius:6px;font-size:0.72rem;white-space:pre-wrap;">⚠️ AI返回格式异常，原文如下：\n${escapeHtml(r.raw_analysis||'')}</div>`;
                    }
                    results.innerHTML = html;
                    results.style.display = '';
                } catch(e) {
                    console.error('Doc review error:', e);
                    if (status) status.textContent = '❌ 网络错误';
                }
                runBtn.disabled = false;
            };
        }
    }

    // ── Doc Review toggle listener (moved to chat's docAnalysisTools) ──
    (function initDocReviewToggle() {
        const el = document.getElementById('docReviewDetails');
        if (el) {
            let loaded = false;
            el.addEventListener('toggle', () => { if (el.open && !loaded) { loaded = true; loadDocReviewPanel(); } });
            // Also load eagerly if already open
            if (el.open) { loaded = true; loadDocReviewPanel(); }
        }
    })();

    async function loadStructuredDocsPanel() {
        const panel = document.getElementById('structuredDocsPanel'); if (!panel) return;
        panel.innerHTML = '<span style="color:var(--card-muted);">加载中...</span>';
        try {
            const r = await fetch('/admin/ingest/structured', {credentials:'include'});
            const docs = (await r.json()).documents || [];
            if (!docs.length) { panel.innerHTML = '<span style="color:var(--card-muted);">暂无结构化文档。</span>'; return; }
            let html = `<div style="font-size:0.65rem;margin-bottom:4px;">${docs.length} 份文档</div>
                <table style="width:100%;font-size:0.65rem;border-collapse:collapse;">`;
            for (const d of docs) {
                html += `<tr style="border-bottom:1px solid var(--card-border);">
                    <td><b>${escapeHtml(d.project_name||'?')}</b></td>
                    <td>${escapeHtml(d.document_type||'')}</td>
                    <td>${escapeHtml(d.bid_number||'')}</td>
                    <td>${d.budget_amount_cny?d.budget_amount_cny.toLocaleString()+'\u00a5':''}</td></tr>`;
            }
            html += '</table>';
            panel.innerHTML = html;
        } catch(_) { panel.innerHTML = '<span style="color:#ef4444;">加载失败</span>'; }
    }

    async function loadWorkloadPanel() {
        const panel = document.getElementById('workloadPanel'); if (!panel) return;
        panel.innerHTML = '<span style="color:var(--card-muted);">加载中...</span>';
        try {
            const r = await fetch('/admin/ingest/review_workload', {credentials:'include'});
            const d = await r.json();
            const wl = d.workload || {};
            const recent = d.recent_log || [];
            if (!Object.keys(wl).length) { panel.innerHTML = '<span style="color:var(--card-muted);">暂无审核记录。</span>'; return; }
            let html = '<table style="width:100%;font-size:0.65rem;border-collapse:collapse;">';
            html += '<tr style="border-bottom:2px solid var(--card-border);text-align:left;"><th>审核人</th><th>角色</th><th>操作数</th><th>项目数</th><th>分类</th><th>最近</th></tr>';
            for (const [uid, w] of Object.entries(wl)) {
                const bd = Object.entries(w.by_type||{}).map(([k,v])=>`${k}:${v}`).join(', ');
                html += `<tr style="border-bottom:1px solid var(--card-border);">
                    <td>${escapeHtml(w.username)}</td><td>${escapeHtml(w.role)}</td>
                    <td><b>${w.total_actions}</b></td><td><b>${w.total_items}</b></td>
                    <td style="font-size:0.55rem;">${bd}</td><td>${new Date(w.last_action).toLocaleDateString()}</td></tr>`;
            }
            html += '</table>';
            if (recent.length) {
                html += '<details style="font-size:0.6rem;margin-top:4px;"><summary>最近操作 ('+recent.length+')</summary>';
                for (const r of recent) {
                    html += `<div>${new Date(r.timestamp).toLocaleString()} ${escapeHtml(r.username)}(${r.role}): ${r.action_type} \u00d7${r.count}</div>`;
                }
                html += '</details>';
            }
            panel.innerHTML = html;
        } catch(_) { panel.innerHTML = '<span style="color:#ef4444;">加载失败</span>'; }
    }

    // ── Archived Sessions Admin ──
    async function loadArchivedSessionsAdmin() {
        const panel = document.getElementById('archivedSessionsAdmin');
        if (!panel) return;
        panel.innerHTML = '<span style="color:var(--card-muted);">加载中...</span>';
        try {
            const r = await fetch('/admin/archived_sessions', {credentials:'include'});
            const d = await r.json();
            const sessions = d.sessions || [];
            if (!sessions.length) { panel.innerHTML = '<span style="color:var(--card-muted);">暂无归档会话</span>'; return; }
            let html = '<table style="width:100%;font-size:0.65rem;border-collapse:collapse;">';
            html += '<tr style="border-bottom:2px solid var(--card-border);"><th>会话</th><th>用户</th><th>归档时间</th></tr>';
            for (const s of sessions) {
                html += `<tr style="border-bottom:1px solid var(--card-border);">
                    <td>${escapeHtml(s.title||s.thread_id?.substring(0,8))}</td>
                    <td>${escapeHtml(s.user_id||'?')}</td>
                    <td>${s.archived_at ? new Date(s.archived_at).toLocaleString() : ''}</td></tr>`;
            }
            html += '</table>';
            panel.innerHTML = html;
        } catch(_) { panel.innerHTML = '<span style="color:#ef4444;">加载失败</span>'; }
    }

    // ── Ingest Panel (batch document ingestion UI) ──
    async function loadIngestPanel() {
        const panel = document.getElementById('ingestPanel');
        if (!panel) return;
        panel.innerHTML = `
            <div style="font-size:0.72rem;color:var(--card-muted);margin-bottom:8px;">
                上传ZIP压缩包（含多个文档），AI自动完成：解压→OCR→分类→提取→生成技能
            </div>
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                <input type="file" id="ingestFileInput" accept=".zip" style="display:none;">
                <button id="ingestUploadBtn" class="file-btn" style="background:#2563eb;color:white;border-color:#1d4ed8;">📥 选择ZIP文件</button>
                <span id="ingestStatus" style="font-size:0.7rem;color:var(--card-muted);"></span>
            </div>
            <div id="ingestProgress" style="margin-top:8px;font-size:0.68rem;"></div>
            <div id="ingestResults" style="margin-top:8px;"></div>
        `;
        setTimeout(() => {
            const fileInput = document.getElementById('ingestFileInput');
            const uploadBtn = document.getElementById('ingestUploadBtn');
            const statusEl = document.getElementById('ingestStatus');
            const progressEl = document.getElementById('ingestProgress');
            if (uploadBtn && fileInput) {
                uploadBtn.onclick = () => fileInput.click();
                fileInput.onchange = async () => {
                    const file = fileInput.files[0];
                    if (!file) return;
                    const form = new FormData(); form.append('file', file);
                    uploadBtn.disabled = true; uploadBtn.textContent = '⏳ 上传中...';
                    statusEl.textContent = '';
                    try {
                        const r = await fetch('/admin/ingest/upload', {method:'POST',credentials:'include',body:form});
                        const d = await r.json();
                        if (r.ok) {
                            statusEl.textContent = '✅ 上传成功，后台处理中...';
                            if (d.task_id) {
                                progressEl.innerHTML = `<span>任务ID: ${d.task_id}</span>`;
                                // Poll status
                                const poll = setInterval(async () => {
                                    const sr = await fetch(`/admin/ingest/status/${d.task_id}`, {credentials:'include'});
                                    const sd = await sr.json();
                                    if (sd.status === 'done') {
                                        progressEl.innerHTML += '<br>' + _icon('check_circle') + ' 处理完成';
                                        clearInterval(poll);
                                        progressEl.innerHTML += '<div style="margin-top:6px;display:flex;gap:6px;"><button class="fb-btn" onclick="window.submitIngestFeedback(\''+d.task_id+'\',1,this)">👍 满意</button><button class="fb-btn" onclick="window.submitIngestFeedback(\''+d.task_id+'\',-1,this)">👎 不满意</button></div>';
                                    }
                                    else if (sd.status === 'failed') { progressEl.innerHTML += '<br>' + _icon('cancel') + ' 处理失败'; clearInterval(poll); }
                                    else { progressEl.innerHTML = `<span>任务ID: ${d.task_id} — ${sd.status||'processing'} ${sd.progress||''}</span>`; }
                                }, 3000);
                            }
                        } else {
                            statusEl.textContent = '❌ ' + (d.error || '上传失败');
                        }
                    } catch(_) { statusEl.textContent = '❌ 网络错误'; }
                    uploadBtn.disabled = false; uploadBtn.textContent = '📥 选择ZIP文件';
                    fileInput.value = '';
                };
            }
        }, 100);
    }

        // ── Writing Style Manager (Admin) ──
    async function loadStyleManager() {
        const panel = document.getElementById('styleManagerPanel');
        if (!panel) return;
        panel.innerHTML = '<span style="color:var(--card-muted);">加载中...</span>';
        try {
            const r = await fetch('/admin/user_styles', { credentials: 'include' });
            const d = await r.json();
            const styles = d.styles || [];
            if (!styles.length) { panel.innerHTML = '<span style="color:var(--card-muted);">暂无风格画像。</span>'; return; }
            let html = `<div style="margin-bottom:6px;display:flex;gap:6px;align-items:center;">
                <span style="font-size:0.68rem;">${styles.length} 个画像</span>
                <button id="styleAnalyzeAllBtn" class="file-btn" style="font-size:0.65rem;padding:2px 8px;">🔄 全量分析</button>
            </div>
            <table style="width:100%;font-size:0.65rem;border-collapse:collapse;">`;
            for (const s of styles) {
                const kwPreview = (s.keywords||[]).slice(0,5).map(k=>k.word).join(', ');
                html += `<tr style="border-bottom:1px solid var(--card-border);">
                    <td style="padding:3px 4px;"><b>${escapeHtml(s.user_id).substring(0,12)}</b></td>
                    <td>${escapeHtml(s.style_label||'无')}</td>
                    <td>${s.total_analyzed||0} 条消息</td>
                    <td style="font-size:0.55rem;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(kwPreview)}">${escapeHtml(kwPreview)}</td>
                    <td>
                        <button class="styleAnalyzeBtn" data-uid="${escapeHtml(s.user_id)}" style="font-size:0.55rem;">🔄</button>
                        <button class="styleDelBtn" data-uid="${escapeHtml(s.user_id)}" style="font-size:0.55rem;color:#ef4444;">🗑</button>
                    </td></tr>`;
            }
            html += '</table><div id="styleMsg" style="margin-top:6px;font-size:0.65rem;"></div>';
            panel.innerHTML = html;

            document.getElementById('styleAnalyzeAllBtn').onclick = async () => {
                const btn = document.getElementById('styleAnalyzeAllBtn');
                btn.disabled = true; btn.textContent = '⏳...';
                try {
                    await fetch('/admin/user_styles/analyze_all', { method:'POST', credentials:'include' });
                    document.getElementById('styleMsg').innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' 批量分析已触发</span>';
                    loadStyleManager();
                } catch(_) { document.getElementById('styleMsg').innerHTML = '<span style="color:#ef4444;">失败</span>'; }
                btn.disabled = false; btn.textContent = '🔄 全量分析';
            };
            panel.querySelectorAll('.styleAnalyzeBtn').forEach(btn => {
                btn.onclick = async () => {
                    btn.disabled = true;
                    await fetch('/admin/user_styles/'+btn.dataset.uid+'/analyze', { method:'POST', credentials:'include' });
                    loadStyleManager();
                };
            });
            panel.querySelectorAll('.styleDelBtn').forEach(btn => {
                btn.onclick = async () => {
                    if (!confirm('删除用户 '+btn.dataset.uid+' 的风格画像？')) return;
                    await fetch('/admin/user_styles/'+btn.dataset.uid+'/delete', { method:'POST', credentials:'include' });
                    loadStyleManager();
                };
            });
        } catch (_) { panel.innerHTML = '<span style="color:#ef4444;">加载失败</span>'; }
    }

    // ── Training Data Export Panel ──
    async function loadTrainingExportPanel() {
        const panel = document.getElementById('trainingExportPanel');
        if (!panel) return;
        panel.innerHTML = '<span style="color:var(--card-muted);">加载中...</span>';
        try {
            const [hr, sr] = await Promise.all([
                fetch('/admin/training_export_history', { credentials: 'include' }),
                fetch('/admin/training_stats', { credentials: 'include' })
            ]);
            const hd = await hr.json();
            const sd = await sr.json();
            const h = hd.history || {};
            const s = sd.stats || {};

            const pending = h.pending_new || 0;
            const hasWm = h.has_watermark;
            const lastTs = h.last_exported_timestamp
                ? new Date(h.last_exported_timestamp).toLocaleString()
                : '从未';
            const totalFull = h.total_exported_full || 0;
            const totalIncr = h.total_exported_incremental || 0;
            const recent = h.recent_exports || [];
            const files = h.export_files || [];

            let html = `<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px;font-size:0.72rem;">
                <span>💾 <b>${s.sessions||0}</b> 个会话</span> ·
                <span>✉️ <b>${s.interactions||0}</b> 对</span> ·
                <span>⭐ <b>${s.qualifying||0}</b> 已评分 ≥3★</span> ·
                <span style="color:${pending>0?'#f59e0b':'#22c55e'};">🆕 <b>${pending}</b> 待导出</span>
            </div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;">
                <button id="trExportIncrBtn" class="file-btn" style="background:#3b82f6;color:white;border-color:#2563eb;font-size:0.72rem;padding:4px 10px;">📥 增量导出${pending>0?` (${pending} 条新)`:' (已是最新)'}</button>
                <button id="trExportFullBtn" class="file-btn" style="font-size:0.72rem;padding:4px 10px;">📦 全量导出</button>
                <button id="trExportAllBtn" class="file-btn" style="font-size:0.72rem;padding:4px 10px;">📦 全量(含低质量)</button>
                <button id="trResetWmBtn" class="file-btn" style="font-size:0.72rem;padding:4px 10px;background:#ef4444;color:white;border-color:#dc2626;">↺ 重置水印</button>
            </div>`;

            // Watermark status
            html += `<div style="font-size:0.68rem;color:var(--card-muted);margin-bottom:8px;">
                水印: ${hasWm ? '✅ 已激活' : '❌ 未设置'} · 上次导出: ${lastTs} ·
                已导出: ${totalFull} 次全量 + ${totalIncr} 次增量 = <b>${totalFull+totalIncr}</b> 总计
                ${files.length ? ` · ${files.length} 个文件在磁盘` : ''}
            </div>`;

            // Recent export history
            if (recent.length) {
                html += `<details style="font-size:0.68rem;margin-bottom:4px;"><summary>导出历史 (最近 ${recent.length} 条)</summary>
                    <table style="width:100%;font-size:0.65rem;border-collapse:collapse;margin-top:4px;">
                    <tr style="border-bottom:1px solid var(--card-border);text-align:left;"><th>文件</th><th>模式</th><th>数量</th><th>时间</th></tr>`;
                for (let i = recent.length - 1; i >= 0; i--) {
                    const r = recent[i];
                    html += `<tr style="border-bottom:1px solid var(--card-border);">
                        <td style="font-family:monospace;font-size:0.6rem;">${escapeHtml(r.file||'')}</td>
                        <td>${r.mode==='incremental'?'🔄 增量':'📦 全量'}</td>
                        <td><b>${r.count||0}</b></td>
                        <td>${new Date(r.time).toLocaleString()}</td>
                    </tr>`;
                }
                html += '</table></details>';
            }

            // Export file manager
            html += `<details id="exportFilesDetails" style="font-size:0.68rem;">
                <summary>📄 导出文件列表 <span id="exportFileCount">(${files.length})</span></summary>
                <div id="exportFilesContent" style="margin-top:4px;font-size:0.62rem;color:var(--card-muted);">点击加载详情...</div>
            </details>`;

            html += '<div id="trMsg" style="margin-top:6px;font-size:0.7rem;"></div>';

            // ── Cleanup section (build HTML, don't render yet) ──
            let cleanupStats = null;
            try {
                const cr = await fetch('/admin/training_cleanup_stats', { credentials: 'include' });
                cleanupStats = (await cr.json()).stats || null;
            } catch (_) {}

            if (cleanupStats && cleanupStats.total_sessions > 0) {
                const cs = cleanupStats;
                html += `<hr style="margin:12px 0;">
                    <div style="font-size:0.7rem;margin-bottom:6px;">
                        🗑️ <b>数据生命周期</b> · ${cs.total_sessions} 个会话 ·
                        最旧 ${cs.oldest_days}天 · 最新 ${cs.newest_days}天 ·
                        <span style="color:${cs.older_than_threshold>0?'#ef4444':'#22c55e'};">${cs.older_than_threshold} 个超过 ${cs.retention_days}天保留期</span>
                        <br><span style="color:var(--card-muted);font-size:0.62rem;">自动清理每季度运行 (1月/4月/7月/10月1日)</span>
                    </div>
                    <div style="display:flex;gap:6px;">
                        <button id="trCleanupPreviewBtn" class="file-btn" style="font-size:0.68rem;padding:3px 8px;">🔍 预览</button>
                        <button id="trCleanupNowBtn" class="file-btn" style="font-size:0.68rem;padding:3px 8px;background:#ef4444;color:white;border-color:#dc2626;">🗑️ 立即清理</button>
                    </div>`;
            }

            // ── Health check section (build HTML, don't render yet) ──
            let healthSummary = null;
            try {
                const hr = await fetch('/admin/training_health_history', { credentials: 'include' });
                healthSummary = (await hr.json()).history || null;
            } catch (_) {}

            if (healthSummary && healthSummary.last_check) {
                const hc = healthSummary.last_check;
                const statusColor = hc.status === 'ok' ? '#22c55e' : (hc.status === 'warning' ? '#f59e0b' : '#ef4444');
                const statusIcon = hc.status === 'ok' ? '✅' : (hc.status === 'warning' ? '⚠️' : '❌');
                const statusLabel = hc.status === 'ok' ? '正常' : (hc.status === 'warning' ? '警告' : '异常');
                html += `<hr style="margin:12px 0;">
                    <div style="font-size:0.7rem;margin-bottom:6px;">
                        🩺 <b>健康检查</b> ${statusIcon} <span style="color:${statusColor};">${statusLabel}</span>
                        · ${hc.total||0} 个会话 · 🟢${hc.healthy||0} 🟡${hc.warning||0} 🔴${hc.corrupt||0}
                        · ${hc.issues_found||0} 个问题 · 上次: ${new Date(hc.timestamp).toLocaleString()}
                        <br><span style="color:var(--card-muted);font-size:0.62rem;">自动检查每周运行 (周日 03:30 UTC)</span>
                    </div>
                    <div style="display:flex;gap:6px;flex-wrap:wrap;">
                        <button id="trHealthScanBtn" class="file-btn" style="font-size:0.68rem;padding:3px 8px;background:#3b82f6;color:white;border-color:#2563eb;">🔍 健康扫描</button>
                        <button id="trHealthRepairBtn" class="file-btn" style="font-size:0.68rem;padding:3px 8px;background:#f59e0b;color:white;border-color:#d97706;">🔧 扫描并修复</button>
                        <button id="trHealthHistoryBtn" class="file-btn" style="font-size:0.68rem;padding:3px 8px;">📋 历史</button>
                    </div>`;
            }

            // ── ONE final render ──
            panel.innerHTML = html;
            const msgEl = document.getElementById('trMsg');

            // ── All button handlers (after DOM exists) ──
            document.getElementById('trExportIncrBtn').onclick = async () => {
                const btn = document.getElementById('trExportIncrBtn');
                btn.disabled = true; btn.textContent = '⏳ 导出中...';
                try {
                    const r = await fetch('/admin/training_export', {
                        method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                        body:JSON.stringify({mode:'incremental'})
                    });
                    const d = await r.json();
                    if (r.ok) { msgEl.innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' '+d.message+'</span>'; loadTrainingExportPanel(); }
                    else msgEl.innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' '+(d.error||'失败')+'</span>';
                } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
                btn.disabled = false; btn.textContent = '📥 增量导出';
            };

            document.getElementById('trExportFullBtn').onclick = async () => {
                if (!confirm('全量导出（仅≥3★高质量）？')) return;
                const btn = document.getElementById('trExportFullBtn');
                btn.disabled = true; btn.textContent = '⏳ ...';
                try {
                    const r = await fetch('/admin/training_export', {
                        method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                        body:JSON.stringify({mode:'quality'})
                    });
                    const d = await r.json();
                    if (r.ok) { msgEl.innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' '+d.message+'</span>'; loadTrainingExportPanel(); }
                    else msgEl.innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' '+(d.error||'失败')+'</span>';
                } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
                btn.disabled = false; btn.textContent = '📦 全量导出';
            };

            document.getElementById('trExportAllBtn').onclick = async () => {
                if (!confirm('全量导出所有数据（含低质量）？')) return;
                const btn = document.getElementById('trExportAllBtn');
                btn.disabled = true; btn.textContent = '⏳ ...';
                try {
                    const r = await fetch('/admin/training_export', {
                        method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                        body:JSON.stringify({mode:'all'})
                    });
                    const d = await r.json();
                    if (r.ok) { msgEl.innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' '+d.message+'</span>'; loadTrainingExportPanel(); }
                    else msgEl.innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' '+(d.error||'失败')+'</span>';
                } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
                btn.disabled = false; btn.textContent = '📦 全量(含低质量)';
            };

            document.getElementById('trResetWmBtn').onclick = async () => {
                if (!confirm('重置导出水印？\n\n下次导出将为全量导出。')) return;
                try {
                    const r = await fetch('/admin/training_export', {
                        method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                        body:JSON.stringify({mode:'reset_watermark'})
                    });
                    const d = await r.json();
                    if (r.ok) { msgEl.innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' '+d.message+'</span>'; loadTrainingExportPanel(); }
                    else msgEl.innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' '+(d.error||'失败')+'</span>';
                } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
            };

            // Cleanup buttons
            const trCleanupPreviewBtn = document.getElementById('trCleanupPreviewBtn');
            const trCleanupNowBtn = document.getElementById('trCleanupNowBtn');
            if (trCleanupPreviewBtn) trCleanupPreviewBtn.onclick = async () => {
                try {
                    const r = await fetch('/admin/training_cleanup', {
                        method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                        body:JSON.stringify({dry_run:true})
                    });
                    const d = await r.json();
                    if (r.ok && d.stats) {
                        const cs = d.stats;
                        msgEl.innerHTML = `<span style="color:var(--card-muted);">将清理 <b>${cs.older_than_threshold}</b> 个 (共${cs.total_sessions}个) 超过${cs.retention_days}天的会话</span>`;
                    } else msgEl.innerHTML = '<span style="color:#ef4444;">预览失败</span>';
                } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
            };
            if (trCleanupNowBtn) trCleanupNowBtn.onclick = async () => {
                if (!confirm(`清理超过 ${cleanupStats?.retention_days||90} 天的训练数据？\n\n这将永久删除旧的训练会话。`)) return;
                trCleanupNowBtn.disabled = true; trCleanupNowBtn.textContent = '⏳ ...';
                try {
                    const r = await fetch('/admin/training_cleanup', {
                        method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                        body:JSON.stringify({})
                    });
                    const d = await r.json();
                    if (r.ok) { msgEl.innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' '+d.message+'</span>'; loadTrainingExportPanel(); }
                    else msgEl.innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' '+(d.error||'失败')+'</span>';
                } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
                trCleanupNowBtn.disabled = false; trCleanupNowBtn.textContent = '🗑️ 立即清理';
            };

            // Health check buttons
            const trHealthScanBtn = document.getElementById('trHealthScanBtn');
            const trHealthRepairBtn = document.getElementById('trHealthRepairBtn');
            const trHealthHistoryBtn = document.getElementById('trHealthHistoryBtn');

            if (trHealthScanBtn) trHealthScanBtn.onclick = async () => {
                trHealthScanBtn.disabled = true; trHealthScanBtn.textContent = '⏳ 扫描中...';
                try {
                    const r = await fetch('/admin/training_health', { credentials: 'include' });
                    const d = await r.json();
                    if (r.ok && d.report) {
                        const rp = d.report;
                        let issuesHtml = '';
                        for (const s of (rp.sessions||[]).filter(s => s.issues.length)) {
                            const prevBadge = s.previous_status
                                ? ` <span style="font-size:0.55rem;opacity:0.6;">(原状态: ${s.previous_status})</span>`
                                : '';
                            const badge = s.status === 'corrupt' ? '🔴' : (s.status === 'warning' ? '🟡' : '🟢');
                            issuesHtml += `<div style="font-size:0.62rem;margin-bottom:2px;">${badge} 📁 ${escapeHtml(s.session)} [${s.status}]${prevBadge} — ${s.issues.join('; ')}</div>`;
                        }
                        const skipped = rp.corrupt_marked_skipped || 0;
                        msgEl.innerHTML = `<span style="color:${rp.corrupt>0?'#ef4444':'#22c55e'};">
                            ${rp.corrupt>0?'❌':'✅'} ${rp.healthy} 正常, ${rp.warning} 警告, ${rp.corrupt} 异常, ${rp.issues_found} 问题
                            ${skipped>0?` · ${skipped} 跳过(已标记异常)` : ''}
                        </span>`;
                        if (issuesHtml) {
                            const details = document.createElement('details');
                            details.style.cssText = 'font-size:0.65rem;margin-top:4px;';
                            details.innerHTML = `<summary>${rp.sessions.filter(s=>s.issues.length).length} 个会话有问题</summary>${issuesHtml}</details>`;
                            msgEl.appendChild(details);
                        }
                    } else msgEl.innerHTML = '<span style="color:#ef4444;">扫描失败</span>';
                } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
                trHealthScanBtn.disabled = false; trHealthScanBtn.textContent = '🔍 健康扫描';
            };

            if (trHealthRepairBtn) trHealthRepairBtn.onclick = async () => {
                if (!confirm('运行健康检查并自动修复？\n\n将修复孤立反馈/上下文索引以及截断的消息。')) return;
                trHealthRepairBtn.disabled = true; trHealthRepairBtn.textContent = '⏳ 修复中...';
                try {
                    const r = await fetch('/admin/training_health', {
                        method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include'
                    });
                    const d = await r.json();
                    if (r.ok && d.report) {
                        msgEl.innerHTML = `<span style="color:#22c55e;">✅ 扫描并修复完成 · ${d.report.repaired||0} 项修复 · ${d.report.healthy} 正常</span>`;
                        loadTrainingExportPanel();
                    } else msgEl.innerHTML = '<span style="color:#ef4444;">修复失败</span>';
                } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
                trHealthRepairBtn.disabled = false; trHealthRepairBtn.textContent = '🔧 扫描并修复';
            };

            if (trHealthHistoryBtn) trHealthHistoryBtn.onclick = async () => {
                try {
                    const r = await fetch('/admin/training_health_history', { credentials: 'include' });
                    const d = await r.json();
                    const h = d.history || {};
                    const trend = h.trend || [];
                    const recent = h.history || [];
                    let histHtml = '<div style="font-size:0.68rem;">';
                    if (trend.length) {
                        histHtml += '<b>Health Trend (last checks):</b><br>';
                        for (const t of trend.slice(-8)) {
                            const icon = (t.corrupt||0) > 0 ? '❌' : ((t.warning||0) > 0 ? '⚠️' : '✅');
                            histHtml += `${icon} ${t.time}: 🟢${t.healthy||0} 🟡${t.warning||0} 🔴${t.corrupt||0}<br>`;
                        }
                    }
                    if (recent.length) {
                        histHtml += `<br><b>Last ${recent.length} detailed records:</b><br>`;
                        for (const r of recent) {
                            histHtml += `📅 ${new Date(r.timestamp).toLocaleString()}: healthy=${r.healthy} warn=${r.warning} corrupt=${r.corrupt} issues=${r.issues_found} fixed=${r.repaired||0}<br>`;
                        }
                    }
                    histHtml += '</div>';
                    msgEl.innerHTML = histHtml;
                } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">Load failed</span>'; }
            };

            // ── Export file manager (load on details toggle) ──
            const exportFilesDetails = document.getElementById('exportFilesDetails');
            if (exportFilesDetails) {
                exportFilesDetails.addEventListener('toggle', async () => {
                    if (!exportFilesDetails.open || exportFilesDetails._loaded) return;
                    exportFilesDetails._loaded = true;
                    const fileContent = document.getElementById('exportFilesContent');
                    fileContent.innerHTML = '<span style="color:var(--card-muted);">Loading...</span>';
                    try {
                        const r = await fetch('/admin/training_exports_list', { credentials: 'include' });
                        const d = await r.json();
                        const flist = d.files || [];
                        const retention = d.retention_count || 20;
                        if (!flist.length) {
                            fileContent.innerHTML = '<span style="color:var(--card-muted);">No export files</span>';
                            return;
                        }
                        let fhtml = `<div style="margin-bottom:4px;color:var(--card-muted);">${flist.length} 个文件, 保留最近 ${retention} 个 · <button id="trExportCleanupBtn" class="file-btn" style="font-size:0.62rem;padding:1px 6px;">🧹 清理旧文件</button></div>`;
                        fhtml += '<table style="width:100%;font-size:0.6rem;border-collapse:collapse;">';
                        for (const f of flist) {
                            fhtml += `<tr style="border-bottom:1px solid var(--card-border);">
                                <td style="font-family:monospace;">📄 ${escapeHtml(f.filename)}</td>
                                <td>${f.size_mb}MB</td>
                                <td>${f.mtime_display}</td>
                                <td>
                                    <a href="/admin/training_exports_download/${encodeURIComponent(f.filename)}" download style="font-size:0.6rem;color:#3b82f6;">⬇</a>
                                    <button class="trExportDelBtn" data-fname="${escapeHtml(f.filename)}" style="font-size:0.6rem;color:#ef4444;background:none;border:none;cursor:pointer;">🗑</button>
                                </td></tr>`;
                        }
                        fhtml += '</table>';
                        fileContent.innerHTML = fhtml;

                        // Cleanup button
                        document.getElementById('trExportCleanupBtn').onclick = async () => {
                            if (!confirm(`删除旧导出文件，保留最近 ${retention} 个？`)) return;
                            const btn = document.getElementById('trExportCleanupBtn');
                            btn.disabled = true; btn.textContent = '...';
                            try {
                                const rr = await fetch('/admin/training_exports_cleanup', {
                                    method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include'
                                });
                                const dd = await rr.json();
                                if (rr.ok) {
                                    msgEl.innerHTML = `<span style="color:#22c55e;">✅ ${dd.message}</span>`;
                                    exportFilesDetails._loaded = false;  // force reload
                                    exportFilesDetails.open = false;
                                    document.getElementById('exportFileCount').textContent = `(${dd.kept})`;
                                } else msgEl.innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' '+(dd.error||'失败')+'</span>';
                            } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
                            btn.disabled = false; btn.textContent = '🧹 清理旧文件';
                        };

                        // Delete buttons
                        fileContent.querySelectorAll('.trExportDelBtn').forEach(btn => {
                            btn.onclick = async () => {
                                const fname = btn.dataset.fname;
                                if (!confirm(`Delete ${fname} permanently?`)) return;
                                btn.disabled = true;
                                try {
                                    const rr = await fetch('/admin/training_exports_delete/' + encodeURIComponent(fname), {
                                        method:'POST', credentials:'include'
                                    });
                                    const dd = await rr.json();
                                    if (rr.ok) {
                                        msgEl.innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' '+dd.message+'</span>';
                                        // Reload file list
                                        exportFilesDetails._loaded = false;
                                        exportFilesDetails.open = false;
                                        setTimeout(() => { exportFilesDetails.open = true; }, 100);
                                    } else msgEl.innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' '+(dd.error||'Failed')+'</span>';
                                } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">Network error</span>'; btn.disabled = false; }
                            };
                        });
                    } catch(_) { fileContent.innerHTML = '<span style="color:#ef4444;">Load failed</span>'; }
                });
            }
        } catch (_) { panel.innerHTML = '<span style="color:#ef4444;">Load failed</span>'; }
    }

    // ── Runtime Config Panel ──
    var _rcData = {}, _rcSchema = {}, _rcDirty = {};

    async function loadRuntimeConfig() {
        const panel = document.getElementById('runtimeConfigContent');
        const msgEl = document.getElementById('rcMsg');
        if (!panel) return;
        panel.innerHTML = '<p style="font-size:.75rem;color:var(--card-muted);">加载配置中...</p>';
        _rcDirty = {};
        const dot = document.getElementById('rcModifiedDot');
        if (dot) dot.style.display = 'none';
        let hasFactory = false, factoryData = null, nonFactoryKeys = [], llmInfo = null, vlInfo = null;
        try {
            const [cr, sr, lr, vr] = await Promise.all([
                fetch('/admin/runtime_config', { credentials: 'include' }),
                fetch('/admin/runtime_config_schema', { credentials: 'include' }),
                fetch('/admin/llm_providers', { credentials: 'include' }),
                fetch('/admin/vl_status', { credentials: 'include' })
            ]);
            const cd = await cr.json();
            const sd = await sr.json();
            _rcData = cd.config || {};
            _rcSchema = sd.schema || {};
            hasFactory = sd.has_factory || false;
            factoryData = sd.factory_presets || null;
            nonFactoryKeys = sd.non_factory_keys || [];
            try { const ld = await lr.json(); if (ld.status === 'ok') llmInfo = ld; } catch (_) {}
            try { const vd = await vr.json(); if (vd.status === 'ok') vlInfo = vd; } catch (_) {}
        } catch (_) { panel.innerHTML = '<p style="color:#ef4444;">Load failed</p>'; return; }

        // Update factory status in summary
        const rcSummary = document.querySelector('#rcDetails summary');
        if (rcSummary) {
            const dotHtml = '<span id="rcModifiedDot" style="display:none;color:#f59e0b;font-size:0.65rem;">● 已修改</span>';
            rcSummary.innerHTML = `⚙️ 运行配置 ${hasFactory
                ? '<span style="color:#22c55e;font-size:0.65rem;">[出厂预设: 已保存]</span>'
                : '<span style="color:#f59e0b;font-size:0.65rem;">[出厂预设: 未保存]</span>'} ${dotHtml}`;
        }

        // Group by schema group
        const groups = {};
        for (const [key, sch] of Object.entries(_rcSchema)) {
            const g = sch.group || 'Other';
            if (!groups[g]) groups[g] = [];
            groups[g].push({ key, ...sch, value: _rcData[key], is_not_factory: nonFactoryKeys.includes(key) });
        }

        // LLM active status banner
        let llmBanner = '';
        if (llmInfo) {
            const pName = llmInfo.providers[llmInfo.active_provider]?.name || llmInfo.active_provider;
            llmBanner = `<div style="background:linear-gradient(135deg,#1e293b,#334155);color:#e2e8f0;border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:.75rem;display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                <span>🤖 <b>当前LLM:</b> ${escapeHtml(pName||'自动')} / ${escapeHtml(llmInfo.active_model||'默认')}</span>
                ${llmInfo.session_provider ? `<span style="color:#94a3b8;font-size:.65rem;">(会话: ${escapeHtml(llmInfo.session_provider)}/${escapeHtml(llmInfo.session_model||'')})</span>` : ''}
            </div>`;
        }

        // VL status banner
        let vlBanner = '';
        if (vlInfo) {
            const availDot = vlInfo.available
                ? '<span style="color:#22c55e;">●</span>'
                : vlInfo.has_api_key
                    ? '<span style="color:#f59e0b;">●</span>'
                    : '<span style="color:#ef4444;">●</span>';
            const availText = vlInfo.available ? '可用' : (vlInfo.has_api_key ? '初始化失败' : '未配置API Key');
            vlBanner = `<div style="background:linear-gradient(135deg,#1a2a3a,#2d4a5a);color:#e2e8f0;border-radius:8px;padding:8px 14px;margin-bottom:12px;font-size:.7rem;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                <span>👁️ <b>当前VL模型:</b> ${escapeHtml(vlInfo.model||'未设置')} (<b>${escapeHtml(vlInfo.provider||'')}</b>)</span>
                <span style="display:inline-flex;align-items:center;gap:3px;">${availDot} ${availText}</span>
                ${!vlInfo.has_api_key ? '<span style="color:#f87171;">请设置 NVIDIA_API_KEY</span>' : ''}
                <span style="color:#94a3b8;font-size:.65rem;">${vlInfo.config.max_image_size}px / ${vlInfo.config.max_tokens}tok / t=${vlInfo.config.temperature}</span>
            </div>`;
        }

        let html = llmBanner + vlBanner;

        const groupOrder = ['LLM/AI Model', 'VL Model', 'Search & Cache', 'RAG Engine', 'File Processing', 'Session & Messages', 'Auto Cleanup', 'Rate Limits', 'Anonymous Limits', 'Training Data', 'Auto Reports', 'Other'];
        const groupLabels = {
            'LLM/AI Model': '🤖 LLM/AI 模型',
            'VL Model': '👁️ VL 视觉模型',
            'Search & Cache': '🔍 搜索与缓存',
            'RAG Engine': '🧠 RAG 引擎',
            'File Processing': '📄 文件处理',
            'Session & Messages': '💬 会话与消息',
            'Auto Cleanup': '🧹 自动清理',
            'Rate Limits': '⏱️ 频率限制',
            'Anonymous Limits': '👤 匿名用户限制',
            'Training Data': '📊 训练数据',
            'Auto Reports': '📋 自动报告',
            'Other': '📦 其他',
        };

        for (const gn of groupOrder) {
            if (!groups[gn] || !groups[gn].length) continue;
            html += `<details open style="margin-bottom:8px;border:1px solid var(--card-border);border-radius:8px;padding:8px 12px;background:var(--card-bg);">
                <summary style="font-weight:600;font-size:.8rem;cursor:pointer;color:var(--card-muted);">${groupLabels[gn]||gn} (${groups[gn].length})</summary>
                <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:6px;margin-top:8px;">`;
            for (const item of groups[gn]) {
                const labelStyle = 'flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;';
                if (item.type === 'ordered-list') {
                    // FIX-016: default chain = OpenRouter :free → NVIDIA NIM
                    const chain = Array.isArray(item.value) ? item.value : [['openrouter','nvidia/nemotron-3-ultra-550b-a55b:free'],['openrouter','z-ai/glm-5.2:free'],['nvidia','nvidia/nemotron-3-ultra-550b-a55b']];
                    const nfMark = item.is_not_factory ? ' <span style="color:#f59e0b;font-size:.6rem;" title="不在出厂预设范围内">[非出厂项]</span>' : '';
                    const provLabels = {auto:'自动检测',openrouter:'OpenRouter',nvidia:'NVIDIA NIM'};
                    let chainHtml = `<div style="display:flex;align-items:center;gap:4px;font-size:.7rem;margin-bottom:4px;" title="${escapeHtml(item.label)}">
                        <label style="${labelStyle}">${item.label}${nfMark}</label>
                    </div>
                    <div data-key="${item.key}" data-type="ordered-list" style="margin-left:4px;">`;
                    for (let i = 0; i < chain.length; i++) {
                        const [cp, cm] = chain[i];
                        const provOpts = Object.entries(provLabels).map(([v,l]) => `<option value="${v}"${cp===v?' selected':''}>${l}</option>`).join('');
                        chainHtml += `<div class="chain-row" data-index="${i}" draggable="true" style="display:flex;align-items:center;gap:3px;margin-bottom:3px;padding:3px 4px;border:1px solid var(--card-border);border-radius:4px;background:var(--card-bg);font-size:.68rem;">
                            <span class="chain-drag" style="cursor:grab;color:var(--card-muted);user-select:none;">☰</span>
                            <select class="chain-provider" style="width:90px;padding:2px 3px;border-radius:3px;border:1px solid var(--card-border);font-size:.65rem;">${provOpts}</select>
                            <select class="chain-model" style="flex:1;min-width:80px;padding:2px 3px;border-radius:3px;border:1px solid var(--card-border);font-size:.65rem;"></select>
                            <button class="chain-remove" style="background:none;border:none;color:#ef4444;cursor:pointer;font-size:.8rem;padding:0 2px;" title="移除此服务商">×</button>
                        </div>`;
                    }
                    chainHtml += `<button class="chain-add" style="width:100%;padding:4px;border:1px dashed var(--card-border);border-radius:4px;background:transparent;color:var(--card-muted);cursor:pointer;font-size:.65rem;margin-top:2px;">+ 添加服务商</button>`;
                    chainHtml += '</div>';
                    html += chainHtml;
                } else if (item.type === 'select') {
                    const options = item.options || [];
                    const labels = item.option_labels || {};
                    const selOpts = options.map(o => `<option value="${o}"${String(item.value)===o?' selected':''}>${labels[o]||o}</option>`).join('');
                    const nfMark = item.is_not_factory ? ' <span style="color:#f59e0b;font-size:.6rem;" title="不在出厂预设范围内">[非出厂项]</span>' : '';
                    html += `<div style="display:flex;align-items:center;gap:4px;font-size:.7rem;" title="${escapeHtml(item.label)}">
                        <label style="${labelStyle}">${item.label}${nfMark}</label>
                        <select data-key="${item.key}" data-type="select" style="width:130px;flex-shrink:0;padding:3px 4px;border-radius:4px;border:1px solid var(--card-border);font-size:.68rem;">${selOpts}</select>
                    </div>`;
                } else if (item.type === 'bool') {
                    const nfMark = item.is_not_factory ? ' <span style="color:#f59e0b;font-size:.6rem;" title="不在出厂预设范围内">[非出厂项]</span>' : '';
                    html += `<div style="display:flex;align-items:center;gap:4px;font-size:.7rem;" title="${escapeHtml(item.label)}">
                        <label style="${labelStyle}">${item.label}${nfMark}</label>
                        <input type="checkbox" data-key="${item.key}" data-type="bool" ${item.value ? 'checked' : ''} style="width:16px;height:16px;cursor:pointer;flex-shrink:0;">
                    </div>`;
                } else {
                    const step = item.step || (item.type === 'float' ? '0.1' : '1');
                    const inputAttrs = item.type === 'float'
                        ? `type="number" step="${step}" min="${item.min||0}" max="${item.max||999999}"`
                        : `type="number" step="1" min="${item.min||0}" max="${item.max||999999}"`;
                    const nfMark = item.is_not_factory ? ' <span style="color:#f59e0b;font-size:.6rem;" title="不在出厂预设范围内">[非出厂项]</span>' : '';
                    html += `<div style="display:flex;align-items:center;gap:4px;font-size:.7rem;" title="${escapeHtml(item.label)}">
                        <label style="${labelStyle}">${item.label}${nfMark}</label>
                        <input data-key="${item.key}" ${inputAttrs} value="${item.value}" style="width:72px;flex-shrink:0;padding:3px 4px;border-radius:4px;border:1px solid var(--card-border);font-size:.68rem;text-align:right;">
                        <span style="width:28px;color:var(--card-muted);text-align:left;font-size:.65rem;flex-shrink:0;">${item.unit||''}</span>
                    </div>`;
                }
            }
            html += '</div></details>';
        }
        // Any groups not in groupOrder
        const done = new Set(groupOrder);
        for (const gn of Object.keys(groups)) {
            if (done.has(gn)) continue;
            html += `<details style="margin-bottom:8px;border:1px solid var(--card-border);border-radius:8px;padding:8px 12px;background:var(--card-bg);">
                <summary style="font-weight:600;font-size:.8rem;cursor:pointer;color:var(--card-muted);">${groupLabels[gn]||gn} (${groups[gn].length})</summary>
                <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:6px;margin-top:8px;">`;
            for (const item of groups[gn]) {
                html += `<div style="display:flex;align-items:center;gap:4px;font-size:.7rem;">
                    <label style="flex:1;">${item.label}</label>
                    <input data-key="${item.key}" type="number" value="${item.value}" style="width:72px;padding:3px 4px;border-radius:4px;border:1px solid var(--card-border);font-size:.68rem;text-align:right;">
                    <span style="width:28px;color:var(--card-muted);font-size:.65rem;">${item.unit||''}</span>
                </div>`;
            }
            html += '</div></details>';
        }

        // Factory action buttons
        html += `<div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;">
            ${!hasFactory
                ? '<button id="rcSaveFactoryBtn" class="file-btn" style="background:#f59e0b;color:white;border-color:#d97706;font-size:0.75rem;">🏭 保存为出厂预设</button>'
                : '<button id="rcRestoreFactoryBtn" class="file-btn" style="background:#8b5cf6;color:white;border-color:#7c3aed;font-size:0.75rem;">↩ 恢复出厂预设</button>'}
            <span style="font-size:.65rem;color:var(--card-muted);align-self:center;">${hasFactory ? '出厂预设已锁定(只读)' : '保存当前值作为不可变的出厂基准'}</span>
        </div>`;

        panel.innerHTML = html;

        // LLM provider change -> reload model options
        const provSelect = panel.querySelector('select[data-key="active_llm_provider"]');
        const modelSelect = panel.querySelector('select[data-key="active_llm_model"]');
        if (provSelect && modelSelect && llmInfo) {
            provSelect.addEventListener('change', () => {
                const pid = provSelect.value;
                const models = (pid !== 'auto' && llmInfo.providers[pid])
                    ? ['auto', ...llmInfo.providers[pid].models]
                    : ['auto'];
                const labels = llmInfo.providers[pid]?.name
                    ? { auto: 'Auto (use ' + llmInfo.providers[pid].name + ' default)' }
                    : { auto: 'Auto (provider default)' };
                modelSelect.innerHTML = models.map(m => `<option value="${m}">${labels[m]||m}</option>`).join('');
                // Mark dirty
                _rcDirty['active_llm_provider'] = provSelect.value !== (_rcData['active_llm_provider']||'') ? provSelect.value : undefined;
                if (_rcDirty['active_llm_provider'] === undefined) delete _rcDirty['active_llm_provider'];
                _rcDirty['active_llm_model'] = modelSelect.value !== (_rcData['active_llm_model']||'') ? modelSelect.value : undefined;
                if (_rcDirty['active_llm_model'] === undefined) delete _rcDirty['active_llm_model'];
                const modDot = document.getElementById('rcModifiedDot');
                if (modDot) modDot.style.display = Object.keys(_rcDirty).length ? 'inline' : 'none';
            });
            modelSelect.addEventListener('change', () => {
                _rcDirty['active_llm_model'] = modelSelect.value !== (_rcData['active_llm_model']||'') ? modelSelect.value : undefined;
                if (_rcDirty['active_llm_model'] === undefined) delete _rcDirty['active_llm_model'];
                const modDot = document.getElementById('rcModifiedDot');
                if (modDot) modDot.style.display = Object.keys(_rcDirty).length ? 'inline' : 'none';
            });
        }

        // VL provider change → reload VL model options from schema
        const vlProvSelect = panel.querySelector('select[data-key="active_vl_provider"]');
        const vlModelSelect = panel.querySelector('select[data-key="active_vl_model"]');
        if (vlProvSelect && vlModelSelect) {
            vlProvSelect.addEventListener('change', () => {
                const pid = vlProvSelect.value;
                // Fetch updated schema to get provider-specific VL models
                fetch('/admin/runtime_config_schema')
                    .then(r => r.json())
                    .then(schema => {
                        if (schema.success && schema.schema && schema.schema.active_vl_model) {
                            const vlSchema = schema.schema.active_vl_model;
                            const models = vlSchema.options || ['auto'];
                            const labels = vlSchema.option_labels || {};
                            const currentModel = _rcData['active_vl_model'] || '';
                            vlModelSelect.innerHTML = models.map(m => {
                                const label = labels[m] || m;
                                const sel = m === currentModel ? ' selected' : '';
                                return `<option value="${m}"${sel}>${label}</option>`;
                            }).join('');
                        }
                    }).catch(() => {}); // Silent fail — keep current options
                _rcDirty['active_vl_provider'] = vlProvSelect.value !== (_rcData['active_vl_provider']||'') ? vlProvSelect.value : undefined;
                if (_rcDirty['active_vl_provider'] === undefined) delete _rcDirty['active_vl_provider'];
                const modDot = document.getElementById('rcModifiedDot');
                if (modDot) modDot.style.display = Object.keys(_rcDirty).length ? 'inline' : 'none';
            });
            vlModelSelect.addEventListener('change', () => {
                _rcDirty['active_vl_model'] = vlModelSelect.value !== (_rcData['active_vl_model']||'') ? vlModelSelect.value : undefined;
                if (_rcDirty['active_vl_model'] === undefined) delete _rcDirty['active_vl_model'];
                const modDot = document.getElementById('rcModifiedDot');
                if (modDot) modDot.style.display = Object.keys(_rcDirty).length ? 'inline' : 'none';
            });
        }

        // Chain list widget — drag-reorder, add/remove entries
        function populateChainModels(row) {
            const provSel = row.querySelector('.chain-provider');
            const modSel = row.querySelector('.chain-model');
            if (!provSel || !modSel) return;
            const pid = provSel.value;
            const currentModel = modSel.dataset.current || modSel.value || '';
            const models = llmInfo?.providers?.[pid]?.models ? ['auto', ...llmInfo.providers[pid].models] : ['auto'];
            if (currentModel && !models.includes(currentModel)) models.push(currentModel);
            const labels = llmInfo?.providers?.[pid]?.labels || {};
            modSel.innerHTML = models.map(m => `<option value="${m}"${m===currentModel?' selected':''}>${labels[m]||m}</option>`).join('');
            modSel.dataset.current = currentModel;
        }
        function chainMarkDirty() {
            const container = document.querySelector('[data-key="llm_fallback_chain"][data-type="ordered-list"]');
            if (!container) return;
            const rows = container.querySelectorAll('.chain-row');
            // FIX-016: default chain = OpenRouter :free → NVIDIA NIM
            const now = Array.from(rows).map(r => [r.querySelector('.chain-provider')?.value || 'openrouter', r.querySelector('.chain-model')?.value || 'nvidia/nemotron-3-ultra-550b-a55b:free']);
            const orig = _rcData['llm_fallback_chain'];
            const nowStr = JSON.stringify(now);
            _rcDirty['llm_fallback_chain'] = (nowStr !== JSON.stringify(orig ?? [['openrouter','nvidia/nemotron-3-ultra-550b-a55b:free']])) ? now : undefined;
            if (_rcDirty['llm_fallback_chain'] === undefined) delete _rcDirty['llm_fallback_chain'];
            const modDot = document.getElementById('rcModifiedDot');
            if (modDot) modDot.style.display = Object.keys(_rcDirty).length ? 'inline' : 'none';
        }
        const chainContainer = document.querySelector('[data-key="llm_fallback_chain"][data-type="ordered-list"]');
        if (chainContainer) {
            // Populate model dropdowns for each row
            chainContainer.querySelectorAll('.chain-row').forEach(r => populateChainModels(r));
            // Provider change → update models
            chainContainer.addEventListener('change', e => {
                const row = e.target.closest('.chain-row');
                if (!row) return;
                if (e.target.classList.contains('chain-provider')) {
                    const modSel = row.querySelector('.chain-model');
                    if (modSel) modSel.dataset.current = ''; // reset
                    populateChainModels(row);
                }
                chainMarkDirty();
            });
            // Remove
            chainContainer.addEventListener('click', e => {
                if (e.target.classList.contains('chain-remove')) {
                    const row = e.target.closest('.chain-row');
                    if (row && chainContainer.querySelectorAll('.chain-row').length > 1) {
                        row.remove();
                        chainMarkDirty();
                    }
                }
            });
            // Add
            const addBtn = chainContainer.querySelector('.chain-add');
            if (addBtn) {
                addBtn.addEventListener('click', () => {
                    const firstRow = chainContainer.querySelector('.chain-row');
                    const clone = firstRow.cloneNode(true);
                    clone.dataset.index = chainContainer.querySelectorAll('.chain-row').length;
                    const provSel = clone.querySelector('.chain-provider');
                    const modSel = clone.querySelector('.chain-model');
                    if (provSel) provSel.value = 'openrouter';
                    if (modSel) { modSel.dataset.current = ''; modSel.value = ''; }
                    populateChainModels(clone);
                    chainContainer.insertBefore(clone, addBtn);
                    chainMarkDirty();
                });
            }
            // Drag & drop
            let dragSrcRow = null;
            chainContainer.addEventListener('dragstart', e => {
                const row = e.target.closest('.chain-row');
                if (row) { dragSrcRow = row; row.style.opacity = '0.4'; e.dataTransfer.effectAllowed = 'move'; e.dataTransfer.setData('text/plain', ''); }
            });
            chainContainer.addEventListener('dragend', e => {
                const row = e.target.closest('.chain-row');
                if (row) row.style.opacity = '';
                chainMarkDirty();
            });
            chainContainer.addEventListener('dragover', e => {
                e.preventDefault();
                const row = e.target.closest('.chain-row');
                if (row && dragSrcRow && row !== dragSrcRow) {
                    const rect = row.getBoundingClientRect();
                    const midY = rect.top + rect.height / 2;
                    if (e.clientY < midY) row.parentNode.insertBefore(dragSrcRow, row);
                    else row.parentNode.insertBefore(dragSrcRow, row.nextSibling);
                }
            });
            chainContainer.addEventListener('drop', e => { e.preventDefault(); });
        }

        // VL test widget — drag-drop image analysis
        const vlGroup = document.querySelector('details summary');
        const vlTestHtml = `<div style="margin-top:10px;border:1px dashed var(--card-border);border-radius:8px;padding:12px;text-align:center;">
            <div id="vlDropZone" style="border:2px dashed #4a5a6a;border-radius:8px;padding:20px;cursor:pointer;transition:border-color .2s;">
                <p style="margin:0;font-size:.75rem;color:var(--card-muted);">📸 拖拽图片到此处或点击上传，测试VL模型</p>
                <input type="file" id="vlTestInput" accept="image/*" style="display:none;">
            </div>
            <div id="vlTestResult" style="margin-top:8px;font-size:.7rem;text-align:left;display:none;background:#1e293b;border-radius:6px;padding:10px;max-height:300px;overflow-y:auto;white-space:pre-wrap;color:#e2e8f0;"></div>
        </div>`;
        panel.insertAdjacentHTML('beforeend', vlTestHtml);

        // Wire VL test widget
        const vlDropZone = document.getElementById('vlDropZone');
        const vlTestInput = document.getElementById('vlTestInput');
        const vlTestResult = document.getElementById('vlTestResult');
        if (vlDropZone && vlTestInput) {
            vlDropZone.onclick = () => vlTestInput.click();
            vlDropZone.addEventListener('dragover', e => { e.preventDefault(); vlDropZone.style.borderColor = '#22c55e'; });
            vlDropZone.addEventListener('dragleave', () => { vlDropZone.style.borderColor = '#4a5a6a'; });
            vlDropZone.addEventListener('drop', e => { e.preventDefault(); vlDropZone.style.borderColor = '#4a5a6a'; if (e.dataTransfer.files.length) handleVLTest(e.dataTransfer.files[0]); });
            vlTestInput.onchange = () => { if (vlTestInput.files.length) handleVLTest(vlTestInput.files[0]); };
        }
        async function handleVLTest(file) {
            if (!file) return;
            vlTestResult.style.display = 'block';
            vlTestResult.innerHTML = '' + _icon('hourglass_empty') + ' 分析中...';
            const fd = new FormData();
            fd.append('image', file);
            try {
                const r = await fetch('/admin/vl_test', { method:'POST', credentials:'include', body:fd });
                const d = await r.json();
                if (d.status === 'ok') {
                    const txt = escapeHtml(d.data?.description || '');
                    const reasoning = d.data?.reasoning ? escapeHtml(d.data.reasoning) : '';
                    vlTestResult.innerHTML = (reasoning ? `<div style="color:#94a3b8;font-size:.65rem;margin-bottom:6px;border-left:2px solid #4a5a6a;padding-left:8px;"><b>推理:</b> ${reasoning}</div>` : '') + `<div>${txt}</div>`;
                } else {
                    vlTestResult.innerHTML = `<span style="color:#ef4444;">${escapeHtml(d.error||'分析失败')}</span>`;
                }
            } catch(e) {
                vlTestResult.innerHTML = `<span style="color:#ef4444;">网络错误: ${escapeHtml(e.message)}</span>`;
            }
        }

        // Track dirty changes for number inputs
        panel.querySelectorAll('input[data-key]').forEach(inp => {
            const eventType = inp.type === 'checkbox' ? 'change' : 'input';
            inp.addEventListener(eventType, () => {
                const key = inp.dataset.key;
                const orig = _rcData[key];
                const now = inp.type === 'checkbox' ? inp.checked : inp.value;
                const nowStr = String(now);
                _rcDirty[key] = (nowStr !== String(orig ?? '')) ? now : undefined;
                if (_rcDirty[key] === undefined) delete _rcDirty[key];
                const modDot = document.getElementById('rcModifiedDot');
                if (modDot) modDot.style.display = Object.keys(_rcDirty).length ? 'inline' : 'none';
            });
        });

        // Factory save button
        const saveFactoryBtn = document.getElementById('rcSaveFactoryBtn');
        if (saveFactoryBtn) saveFactoryBtn.onclick = async () => {
            if (!confirm('Save current config as factory presets?\n\nFactory presets are IMMUTABLE — they cannot be edited or deleted. This is a one-time operation.')) return;
            saveFactoryBtn.disabled = true; saveFactoryBtn.textContent = '⏳ Saving...';
            try {
                const r = await fetch('/admin/runtime_config', {
                    method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                    body:JSON.stringify({_action:'save_factory'})
                });
                const d = await r.json();
                if (r.ok) { document.getElementById('rcMsg').innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' '+d.message+'</span>'; loadRuntimeConfig(); }
                else document.getElementById('rcMsg').innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' '+(d.error||'失败')+'</span>';
            } catch(_) { document.getElementById('rcMsg').innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
            saveFactoryBtn.disabled = false; saveFactoryBtn.textContent = '🏭 保存为出厂预设';
        };

        // Factory restore button
        const restoreFactoryBtn = document.getElementById('rcRestoreFactoryBtn');
        if (restoreFactoryBtn) restoreFactoryBtn.onclick = async () => {
            if (!confirm('恢复所有配置到出厂预设？\n\n这将丢弃全部自定义修改。LLM服务商/模型设置将保留。')) return;
            restoreFactoryBtn.disabled = true; restoreFactoryBtn.textContent = '⏳ 恢复中...';
            try {
                const r = await fetch('/admin/runtime_config', {
                    method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                    body:JSON.stringify({_action:'restore_factory'})
                });
                const d = await r.json();
                if (r.ok) { document.getElementById('rcMsg').innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' '+d.message+'</span>'; loadRuntimeConfig(); }
                else document.getElementById('rcMsg').innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' '+(d.error||'失败')+'</span>';
            } catch(_) { document.getElementById('rcMsg').innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
            restoreFactoryBtn.disabled = false; restoreFactoryBtn.textContent = '↩ 恢复出厂预设';
        };
    }

    // Save all button
    document.getElementById('rcSaveAllBtn').addEventListener('click', async () => {
        const dirty = Object.entries(_rcDirty).filter(([,v]) => v !== undefined);
        if (!dirty.length) { document.getElementById('rcMsg').innerHTML = '<span style="color:var(--card-muted);">无修改</span>'; return; }
        const payload = {};
        for (const [k, v] of dirty) {
            const sch = _rcSchema[k];
            if (sch?.type === 'select') { payload[k] = v; }
            else if (sch?.type === 'bool') { payload[k] = v === true || v === 'true'; }
            else if (sch?.type === 'ordered-list') {
                const parsed = typeof v === 'string' ? (() => { try { return JSON.parse(v); } catch(_) { return v; } })() : v;
                payload[k] = parsed;
            }
            else if (sch?.type === 'float') { payload[k] = parseFloat(v); }
            else { payload[k] = parseInt(v); }
        }
        const btn = document.getElementById('rcSaveAllBtn');
        btn.disabled = true; btn.textContent = '⏳ 保存中...';
        try {
            const r = await fetch('/admin/runtime_config', {
                method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                body:JSON.stringify(payload)
            });
            const d = await r.json();
            if (r.ok) {
                document.getElementById('rcMsg').innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' '+d.message+'</span>';
                _rcDirty = {};
                const modDot = document.getElementById('rcModifiedDot');
                if (modDot) modDot.style.display = 'none';
                loadRuntimeConfig();
            } else {
                document.getElementById('rcMsg').innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' '+(d.error||'保存失败')+'</span>';
            }
        } catch(_) { document.getElementById('rcMsg').innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
        btn.disabled = false; btn.textContent = '💾 Save All Changes';
    });

    // Refresh button — moved to lazy-load handler inside rcDetails toggle

    // Reset button
    document.getElementById('rcResetBtn').addEventListener('click', async () => {
        if (!confirm('重置所有运行配置到默认值？\n这将丢弃全部自定义修改。')) return;
        try {
            const r = await fetch('/admin/runtime_config', {
                method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                body:JSON.stringify({_action:'reset'})
            });
            const d = await r.json();
            if (r.ok) {
                document.getElementById('rcMsg').innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' '+d.message+'</span>';
                _rcDirty = {};
                const modDot = document.getElementById('rcModifiedDot');
                if (modDot) modDot.style.display = 'none';
                loadRuntimeConfig();
            } else {
                document.getElementById('rcMsg').innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' '+(d.error||'Failed')+'</span>';
            }
        } catch(_) { document.getElementById('rcMsg').innerHTML = '<span style="color:#ef4444;">Network error</span>'; }
    });

    async function loadAssetManager() {
        const container = document.getElementById('assetManager');
        if (!container) return;
        let allUsers = [], deposits = [], selectedSrc = new Set(), selectedDep = new Set();
        try {
            const r = await fetch('/admin/user_assets', { credentials: 'include' });
            const d = await r.json();
            allUsers = d.users || [];
            deposits = d.deposits || [];
        } catch(_) { container.innerHTML = '<p style="color:#e74c3c">加载失败</p>'; return; }

        function render() {
            const filter = (container.querySelector('#assetSearch')?.value||'').toLowerCase();
            const filtered = filter ? allUsers.filter(u => u.username.toLowerCase().includes(filter)) : allUsers;
            const depositCount = deposits.length;
            let html = `<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:6px;">
                <input id="assetSearch" placeholder="🔍 搜索用户..." style="flex:1;min-width:150px;padding:5px 8px;border-radius:6px;border:1px solid var(--card-border);font-size:.75rem;">
                <button id="assetSelectAll" class="file-btn" style="padding:3px 8px;font-size:.7rem;">全选</button>
                <span style="font-size:.7rem;color:var(--card-muted);">已选 <b id="assetSelCount">0</b> 用户</span>
                <select id="assetTarget" style="padding:5px;border-radius:6px;border:1px solid var(--card-border);font-size:.75rem;">
                    <option value="">-- 选择接收者 --</option>${allUsers.map(u=>`<option value="${u.user_id}">${escapeHtml(u.username)}</option>`).join('')}</select>
                <button id="assetTransferBtn" class="file-btn" style="background:#2563eb;color:white;padding:4px 12px;font-size:.72rem;">转移选中</button>
                <button id="assetRefreshBtn" class="file-btn" style="padding:3px 8px;font-size:.7rem;">🔄</button>
            </div>`;

            // Deposit section
            if (depositCount > 0) {
                html += `<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:6px;padding:8px;margin-bottom:8px;">
                    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:4px;">
                        <strong style="font-size:.75rem;">📦 孤数据托管 (${depositCount}项)</strong>
                        <span style="font-size:.65rem;color:var(--card-muted);">已删除账户的遗留资产</span>
                    </div>
                    <table style="width:100%;font-size:.68rem;margin-top:4px;border-collapse:collapse;">
                    <tr style="text-align:left;border-bottom:1px solid var(--card-border);"><th style="padding:2px 4px;"><input type="checkbox" id="depSelectAll"></th><th>来源</th><th>类型</th><th>日期</th></tr>`;
                for (const item of deposits.slice(0,20)) {
                    const checked = selectedDep.has(item.id);
                    html += `<tr style="border-bottom:1px solid var(--card-border);${checked?'background:#eff6ff;':''}">
                        <td style="padding:2px 4px;"><input type="checkbox" class="dep-cb" data-id="${item.id}" ${checked?'checked':''}></td>
                        <td>${escapeHtml(item.original_username||'?')}</td>
                        <td>${escapeHtml(item.item_type||'?')}</td>
                        <td>${new Date(item.created_at).toLocaleDateString()}</td></tr>`;
                }
                html += '</table></div>';
            }

            // User asset table
            html += `<div style="overflow-x:auto;">
            <table style="width:100%;font-size:.7rem;border-collapse:collapse;">
            <tr style="text-align:left;border-bottom:2px solid var(--card-border);background:var(--card-bg);position:sticky;top:0;">
                <th style="padding:4px 6px;"><input type="checkbox" id="userSelectAll"></th>
                <th>用户</th><th>会话</th><th>聊天文件</th><th>知识库</th><th>批量对比</th><th>项目</th><th>合计</th>
            </tr>`;
            for (const u of filtered) {
                const checked = selectedSrc.has(u.user_id);
                html += `<tr style="border-bottom:1px solid var(--card-border);${checked?'background:#eff6ff;':''}${u.total===0?' color:var(--card-muted);':''}">
                    <td style="padding:2px 4px;"><input type="checkbox" class="user-cb" data-uid="${u.user_id}" ${checked?'checked':''}></td>
                    <td style="white-space:nowrap;"><b>${escapeHtml(u.username)}</b></td>
                    <td>${u.sessions}</td><td>${u.chat_files}${u.chat_mb>0?`<small> ${u.chat_mb}MB</small>`:''}</td>
                    <td>${u.kb_files}</td><td>${u.batch_results}</td><td>${u.projects}</td>
                    <td><b>${u.total}</b></td></tr>`;
            }
            html += '</table></div>';
            container.innerHTML = html;

            // Wire checkboxes
            const updateCount = () => {
                const cnt = container.querySelector('#assetSelCount'); if(cnt) cnt.textContent = selectedSrc.size;
            };
            container.querySelector('#assetSelectAll').onclick = () => {
                filtered.forEach(u => selectedSrc.add(u.user_id)); updateCount(); render();
            };
            container.querySelector('#userSelectAll').onchange = (e) => {
                filtered.forEach(u => e.target.checked ? selectedSrc.add(u.user_id) : selectedSrc.delete(u.user_id));
                updateCount(); render();
            };
            container.querySelectorAll('.user-cb').forEach(cb => {
                cb.onchange = () => { cb.checked ? selectedSrc.add(cb.dataset.uid) : selectedSrc.delete(cb.dataset.uid); updateCount(); };
            });
            container.querySelectorAll('.dep-cb').forEach(cb => {
                cb.onchange = () => { cb.checked ? selectedDep.add(parseInt(cb.dataset.id)) : selectedDep.delete(parseInt(cb.dataset.id)); };
            });
            const depAll = container.querySelector('#depSelectAll');
            if (depAll) depAll.onchange = (e) => {
                deposits.forEach(item => e.target.checked ? selectedDep.add(item.id) : selectedDep.delete(item.id));
                render();
            };
            updateCount();
            // Transfer button
            container.querySelector('#assetTransferBtn').onclick = async () => {
                const target = container.querySelector('#assetTarget').value;
                if (!target) { alert('请选择接收用户'); return; }
                if (!selectedSrc.size && !selectedDep.size) { alert('请至少选择一个来源用户或托管项'); return; }
                const count = selectedSrc.size + selectedDep.size;
                if (!confirm(`将 ${count} 个来源的资产转移给目标用户？`)) return;
                const res = await fetch('/admin/transfer_assets', {
                    method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                    body:JSON.stringify({target_user_id:target, source_user_ids:[...selectedSrc], deposit_ids:[...selectedDep], types:['all']})
                });
                const d = await res.json();
                if (res.ok) { showToast(`已转移 ${d.transferred} 项`, 'success'); selectedSrc.clear(); selectedDep.clear(); loadAssetManager(); }
                else alert(d.error||'转移失败');
            };
            container.querySelector('#assetRefreshBtn').onclick = () => loadAssetManager();
            // Search
            container.querySelector('#assetSearch').oninput = () => render();
        }
        render();
    }

