/* Unified Bid Audit module - extracted from app.js (W5) */
    // ======================== Unified Bid Audit ========================
    var _auditFunctionLabels = {
        rule_extraction: '规则提取',
        compliance_check: '合规审查',
        typo_detection: '错别字检测',
        quote_anomaly: '报价异常',
        relationship_extraction: '关系分析',
        ai_doc_review: 'AI文档审查',
        style_analysis: '文风分析',
    };

    // Severity threshold definitions per function: [{key, label, type, default}]
    var _auditThresholdDefs = {
        rule_extraction: [
            {key: 'min_extracted_rules', label: '最少提取规则数', type: 'int', default: 5}
        ],
        compliance_check: [
            {key: 'critical', label: '严重违规≥', type: 'int', default: 1},
            {key: 'violation', label: '一般违规≥', type: 'int', default: 3}
        ],
        typo_detection: [
            {key: 'penalty_per_10k', label: '每万字扣分', type: 'int', default: 5}
        ],
        quote_anomaly: [
            {key: 'same_rate', label: '雷同报价阈值', type: 'float', default: 0.05},
            {key: 'drop', label: '异常降价阈值', type: 'float', default: 0.15}
        ],
        relationship_extraction: [
            {key: 'risk_signal_weight', label: '风险信号权重', type: 'int', default: 15}
        ],
        ai_doc_review: [
            {key: 'min_chars', label: '最少字符数', type: 'int', default: 500}
        ],
        style_analysis: []
    };

    // ── Audit Config Panel (analytics admin) ──
    async function loadAuditConfig() {
        const tbody = document.getElementById('auditConfigTbody');
        const msgEl = document.getElementById('auditConfigMsg');
        if (!tbody) return;
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--card-muted);padding:8px;">加载中...</td></tr>';
        let cfg = [];
        try {
            const r = await fetch('/audit/config', { credentials: 'include' });
            if (!r.ok) { tbody.innerHTML = '<tr><td colspan="5" style="color:#ef4444;text-align:center;padding:8px;">加载失败</td></tr>'; return; }
            const raw = await r.json();
            cfg = Array.isArray(raw) ? raw : (raw.configs || raw.data || []);
        } catch (_) {
            tbody.innerHTML = '<tr><td colspan="5" style="color:#ef4444;text-align:center;padding:8px;">网络错误</td></tr>';
            return;
        }
        if (!Array.isArray(cfg) || !cfg.length) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--card-muted);padding:8px;">暂无审计配置</td></tr>';
            return;
        }
        let html = '';
        const _origCfg = {};
        for (const c of cfg) {
            _origCfg[c.function_name] = { ...c };
            const fn = c.function_name;
            const label = _auditFunctionLabels[fn] || fn;
            const defs = _auditThresholdDefs[fn] || [];
            const sev = (typeof c.severity_thresholds === 'object' && c.severity_thresholds)
                ? c.severity_thresholds
                : (typeof c.severity_thresholds === 'string' ? JSON.parse(c.severity_thresholds || '{}') : {});

            // Build severity threshold inputs
            let sevHtml = '';
            if (defs.length === 0) {
                sevHtml = '<span style="font-size:0.6rem;color:var(--card-muted);">-</span>';
            } else {
                for (const d of defs) {
                    const val = sev[d.key] !== undefined ? sev[d.key] : d.default;
                    sevHtml += '<span style="white-space:nowrap;margin-right:6px;">' +
                        '<span style="font-size:0.58rem;color:var(--card-muted);">' + escapeHtml(d.label) + '</span> ' +
                        '<input type="number" class="audit-cfg-thr" data-fn="' + fn + '" data-key="' + d.key +
                        '" value="' + val + '" step="' + (d.type === 'float' ? '0.01' : '1') +
                        '" style="width:' + (d.type === 'float' ? '55px' : '45px') +
                        ';font-size:0.62rem;padding:1px 3px;border-radius:3px;border:1px solid var(--card-border);">' +
                        '</span>';
                }
            }

            html += '<tr>' +
                '<td style="padding:4px;font-weight:600;">' + escapeHtml(label) + '</td>' +
                '<td style="text-align:center;"><input type="checkbox" class="audit-cfg-cb" data-fn="' + fn + '" ' + (c.enabled_by_default ? 'checked' : '') + '></td>' +
                '<td style="text-align:center;">' +
                    '<input type="range" class="audit-cfg-range" data-fn="' + fn + '" data-field="fail_threshold" min="0" max="100" value="' + c.fail_threshold + '" style="width:60px;vertical-align:middle;">' +
                    '<span class="audit-cfg-val" style="margin-left:2px;font-size:0.65rem;">' + c.fail_threshold + '</span>' +
                '</td>' +
                '<td style="text-align:center;">' +
                    '<input type="range" class="audit-cfg-range" data-fn="' + fn + '" data-field="weight" min="0" max="100" value="' + c.weight + '" style="width:60px;vertical-align:middle;">' +
                    '<span class="audit-cfg-val" style="margin-left:2px;font-size:0.65rem;">' + c.weight + '</span>' +
                '</td>' +
                '<td style="font-size:0.62rem;">' + sevHtml + '</td>' +
            '</tr>';
        }
        tbody.innerHTML = html;

        const dot = document.getElementById('auditConfigModifiedDot');
        const _dirty = {};
        function _updateDot() {
            if (dot) dot.style.display = Object.keys(_dirty).length ? 'inline' : 'none';
        }

        tbody.querySelectorAll('.audit-cfg-range').forEach(inp => {
            inp.addEventListener('input', () => {
                const fn = inp.dataset.fn;
                const field = inp.dataset.field;
                const valEl = inp.parentElement.querySelector('.audit-cfg-val');
                if (valEl) valEl.textContent = inp.value;
                if (!_dirty[fn]) _dirty[fn] = {};
                _dirty[fn][field] = parseInt(inp.value);
                _updateDot();
            });
        });
        tbody.querySelectorAll('.audit-cfg-cb').forEach(cb => {
            cb.addEventListener('change', () => {
                const fn = cb.dataset.fn;
                if (!_dirty[fn]) _dirty[fn] = {};
                _dirty[fn].enabled_by_default = cb.checked;
                _updateDot();
            });
        });
        tbody.querySelectorAll('.audit-cfg-thr').forEach(inp => {
            inp.addEventListener('input', () => {
                const fn = inp.dataset.fn;
                if (!_dirty[fn]) _dirty[fn] = {};
                if (!_dirty[fn].severity_thresholds) {
                    _dirty[fn].severity_thresholds = {};
                    const origSev = _origCfg[fn] && _origCfg[fn].severity_thresholds;
                    if (typeof origSev === 'object' && origSev) {
                        Object.assign(_dirty[fn].severity_thresholds, origSev);
                    }
                }
                const defs = _auditThresholdDefs[fn] || [];
                const d = defs.find(dd => dd.key === inp.dataset.key);
                const val = d && d.type === 'float' ? parseFloat(inp.value) : parseInt(inp.value);
                _dirty[fn].severity_thresholds[inp.dataset.key] = isNaN(val) ? 0 : val;
                _updateDot();
            });
        });

        document.getElementById('auditConfigSaveBtn').onclick = async () => {
            const dirtyKeys = Object.keys(_dirty);
            if (!dirtyKeys.length) { msgEl.innerHTML = '<span style="color:var(--card-muted);">无修改</span>'; return; }
            const configs = dirtyKeys.map(fn => {
                const d = _dirty[fn];
                const orig = _origCfg[fn] || {};
                return {
                    function_name: fn,
                    enabled_by_default: d.enabled_by_default !== undefined ? d.enabled_by_default : orig.enabled_by_default,
                    fail_threshold: d.fail_threshold !== undefined ? d.fail_threshold : orig.fail_threshold,
                    weight: d.weight !== undefined ? d.weight : orig.weight,
                    severity_thresholds: d.severity_thresholds !== undefined ? d.severity_thresholds : orig.severity_thresholds,
                };
            });
            const btn = document.getElementById('auditConfigSaveBtn');
            btn.disabled = true; btn.textContent = '⏳ 保存中...';
            try {
                const r = await fetch('/audit/config', {
                    method: 'PUT', headers: { 'Content-Type': 'application/json' }, credentials: 'include',
                    body: JSON.stringify({ configs })
                });
                const d = await r.json();
                if (r.ok) {
                    msgEl.innerHTML = '<span style="color:#22c55e;">' + _icon('check_circle') + ' ' + (d.message || '保存成功') + '</span>';
                    Object.keys(_dirty).forEach(k => delete _dirty[k]);
                    _updateDot();
                    loadAuditConfig();
                } else {
                    msgEl.innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' ' + (d.error || '保存失败') + '</span>';
                }
            } catch (_) { msgEl.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; }
            btn.disabled = false; btn.textContent = '💾 保存审计配置';
        };
    }

    // ── Audit History (review tab) ──
    async function loadAuditHistory() {
        const panel = document.getElementById('auditHistoryPanel');
        if (!panel) return;
        const pid = window._currentProjectId || currentProjectId;
        if (!pid) {
            panel.innerHTML = '<span style="color:var(--card-muted);">请先打开一个项目查看审计历史。</span>';
            return;
        }
        // Migration notice: audit history is now per-project in the project view
        panel.innerHTML = '<div style="font-size:0.72rem;color:var(--card-muted);padding:8px;background:#f0f9ff;border-radius:6px;margin-bottom:8px;">📋 审计历史现已移至<strong>项目视图</strong>。请在项目管理中打开具体项目，即可在项目页面的底部查看该项目的审计记录和下载报告。</div>' +
            '<span style="font-size:0.65rem;color:var(--card-muted);">加载中...</span>';
        return;
        panel.innerHTML = '<span style="color:var(--card-muted);">加载中...</span>';
        let runs = [];
        try {
            const r = await fetch('/audit/history/' + pid, { credentials: 'include' });
            if (!r.ok) { panel.innerHTML = '<span style="color:#ef4444;">加载失败</span>'; return; }
            const rawRuns = await r.json();
            runs = Array.isArray(rawRuns) ? rawRuns : (rawRuns.data || []);
        } catch (_) { panel.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; return; }
        if (!Array.isArray(runs) || !runs.length) {
            panel.innerHTML = '<div style="font-size:0.72rem;color:var(--card-muted);margin-bottom:6px;">暂无审计记录。</div>';
            return;
        }
        runs.sort((a, b) => new Date(b.started_at || b.created_at || 0) - new Date(a.started_at || a.created_at || 0));
        let html = '<div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">' +
            '<span style="font-size:0.7rem;color:var(--card-muted);">共 ' + runs.length + ' 次审计</span>' +
            '<button id="auditHistoryRefreshBtn" class="file-btn" style="font-size:0.65rem;padding:2px 8px;">🔄 刷新</button>' +
            '</div>';
        for (const run of runs) {
            const isPass = run.overall_status === 'PASS';
            const score = run.overall_score != null ? parseFloat(run.overall_score).toFixed(1) : '-';
            const date = (run.started_at || run.created_at) ? new Date(run.started_at || run.created_at).toLocaleString() : '?';
            const fileCount = run.file_count ?? run.total_files ?? 0;
            const bidderCount = run.bidder_count ?? run.total_bidders ?? '?';
            const runId = run.id;

            html += '<div class="audit-history-card" data-run-id="' + runId + '" style="border:1px solid var(--card-border);border-radius:6px;padding:8px 10px;margin-bottom:6px;background:var(--card-bg);cursor:pointer;">' +
                '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;">' +
                '<span style="font-size:0.72rem;">' + date + '</span>' +
                '<span style="font-size:0.7rem;">得分: <b>' + score + '</b></span>' +
                '<span class="audit-status-badge ' + (isPass ? 'audit-pass' : 'audit-fail') + '" style="font-size:0.65rem;padding:1px 8px;border-radius:10px;font-weight:600;">' + (isPass ? 'PASS' : 'FAIL') + '</span>' +
                '<span style="font-size:0.65rem;color:var(--card-muted);">' + fileCount + ' 文件 · ' + bidderCount + ' 投标人</span>' +
                '<span style="font-size:0.6rem;color:var(--card-muted);">▶ 展开</span>' +
                '</div>' +
                '<div class="audit-history-detail" style="display:none;margin-top:8px;border-top:1px solid var(--card-border);padding-top:8px;"></div>' +
                '</div>';
        }
        panel.innerHTML = html;

        const refreshBtn = document.getElementById('auditHistoryRefreshBtn');
        if (refreshBtn) refreshBtn.onclick = loadAuditHistory;

        panel.querySelectorAll('.audit-history-card').forEach(card => {
            card.onclick = async () => {
                const detailEl = card.querySelector('.audit-history-detail');
                if (!detailEl) return;
                if (detailEl.style.display !== 'none') {
                    detailEl.style.display = 'none';
                    card.querySelector('span:last-child').textContent = '▶ 展开';
                    return;
                }
                const runId = card.dataset.runId;
                detailEl.innerHTML = '<span style="font-size:0.65rem;color:var(--card-muted);">加载中...</span>';
                detailEl.style.display = 'block';
                card.querySelector('span:last-child').textContent = '▼ 收起';
                try {
                    const r = await fetch('/audit/result/' + runId, { credentials: 'include' });
                    if (!r.ok) { detailEl.innerHTML = '<span style="color:#ef4444;font-size:0.65rem;">加载失败</span>'; return; }
                    const result = await r.json();
                    detailEl.innerHTML = _renderAuditResultDetail(result);
                } catch (_) {
                    detailEl.innerHTML = '<span style="color:#ef4444;font-size:0.65rem;">网络错误</span>';
                }
            };
        });
    }

    async function loadProjectAuditHistory() {
        const panel = document.getElementById('projectAuditHistoryPanel');
        if (!panel) return;
        const pid = window._currentProjectId || currentProjectId;
        if (!pid) {
            panel.innerHTML = '<span style="color:var(--card-muted);">请先打开一个项目。</span>';
            return;
        }
        panel.innerHTML = '<span style="color:var(--card-muted);">加载中...</span>';
        let runs = [];
        try {
            const r = await fetch('/audit/history/' + pid, { credentials: 'include' });
            if (!r.ok) { panel.innerHTML = '<span style="color:#ef4444;">加载失败</span>'; return; }
            const rawRuns = await r.json();
            runs = Array.isArray(rawRuns) ? rawRuns : (rawRuns.data || []);
        } catch (_) { panel.innerHTML = '<span style="color:#ef4444;">网络错误</span>'; return; }
        if (!Array.isArray(runs) || !runs.length) {
            panel.innerHTML = '<div style="font-size:0.72rem;color:var(--card-muted);margin-bottom:6px;">暂无审计记录。</div>';
            const countEl = document.getElementById('projectAuditCount');
            if (countEl) countEl.textContent = '';
            return;
        }
        runs.sort((a, b) => new Date(b.started_at || b.created_at || 0) - new Date(a.started_at || a.created_at || 0));
        const countEl = document.getElementById('projectAuditCount');
        if (countEl) countEl.textContent = `(${runs.length} 条记录)`;
        let html = '<div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">' +
            '<span style="font-size:0.7rem;color:var(--card-muted);font-weight:600;">审计记录</span>' +
            '<button id="projectAuditHistoryRefreshBtn" class="file-btn" style="font-size:0.65rem;padding:2px 8px;">🔄 刷新</button>' +
            '</div>';
        for (const run of runs) {
            const isPass = run.overall_status === 'PASS';
            const score = run.overall_score != null ? parseFloat(run.overall_score).toFixed(1) : '-';
            const date = (run.started_at || run.created_at) ? new Date(run.started_at || run.created_at).toLocaleString() : '?';
            const fileCount = run.file_count ?? run.total_files ?? 0;
            const bidderCount = run.bidder_count ?? run.total_bidders ?? '?';
            const runId = run.id;
            html += '<div class="audit-history-card" data-run-id="' + runId + '" style="border:1px solid var(--card-border);border-radius:6px;padding:8px 10px;margin-bottom:6px;background:var(--card-bg);cursor:pointer;">' +
                '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;">' +
                '<span style="font-size:0.72rem;">' + date + '</span>' +
                '<span style="font-size:0.7rem;">得分: <b>' + score + '</b></span>' +
                '<span class="audit-status-badge ' + (isPass ? 'audit-pass' : 'audit-fail') + '" style="font-size:0.65rem;padding:1px 8px;border-radius:10px;font-weight:600;">' + (isPass ? 'PASS' : 'FAIL') + '</span>' +
                '<span style="font-size:0.65rem;color:var(--card-muted);">' + fileCount + ' 文件 · ' + bidderCount + ' 投标人</span>' +
                '<span style="font-size:0.6rem;color:var(--card-muted);">▶ 展开</span>' +
                '</div>' +
                '<div class="audit-history-detail" style="display:none;margin-top:8px;border-top:1px solid var(--card-border);padding-top:8px;"></div>' +
                '</div>';
        }
        panel.innerHTML = html;
        const refreshBtn = document.getElementById('projectAuditHistoryRefreshBtn');
        if (refreshBtn) refreshBtn.onclick = loadProjectAuditHistory;
        panel.querySelectorAll('.audit-history-card').forEach(card => {
            card.onclick = async () => {
                const detailEl = card.querySelector('.audit-history-detail');
                if (!detailEl) return;
                if (detailEl.style.display !== 'none') {
                    detailEl.style.display = 'none';
                    card.querySelector('span:last-child').textContent = '▶ 展开';
                    return;
                }
                const runId = card.dataset.runId;
                detailEl.innerHTML = '<span style="font-size:0.65rem;color:var(--card-muted);">加载中...</span>';
                detailEl.style.display = 'block';
                card.querySelector('span:last-child').textContent = '▼ 收起';
                try {
                    const r = await fetch('/audit/result/' + runId, { credentials: 'include' });
                    if (!r.ok) { detailEl.innerHTML = '<span style="color:#ef4444;font-size:0.65rem;">加载失败</span>'; return; }
                    const result = await r.json();
                    detailEl.innerHTML = _renderAuditResultDetail(result);
                } catch (_) {
                    detailEl.innerHTML = '<span style="color:#ef4444;font-size:0.65rem;">网络错误</span>';
                }
            };
        });
    }

    function _renderAuditResultDetail(result) {
        let html = '';

        // Download buttons
        if (result.docx_path || result.xlsx_path) {
            html += '<div style="display:flex;gap:6px;margin-bottom:8px;">';
            if (result.docx_path) html += '<button onclick="window.open(\'/audit/download/' + result.id + '/docx\', \'_blank\')" class="file-btn" style="font-size:0.65rem;padding:2px 8px;">📄 下载DOCX</button>';
            if (result.xlsx_path) html += '<button onclick="window.open(\'/audit/download/' + result.id + '/xlsx\', \'_blank\')" class="file-btn" style="font-size:0.65rem;padding:2px 8px;">📊 下载XLSX</button>';
            html += '</div>';
        }

        // Score summary
        const score = result.overall_score != null ? parseFloat(result.overall_score).toFixed(1) : '-';
        const isPass = result.overall_status === 'PASS';
        html += '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px;padding:4px 8px;border-radius:6px;background:' + (isPass ? '#dcfce7' : '#fef2f2') + ';">' +
            '<span style="font-size:0.7rem;font-weight:600;">综合得分: ' + score + '</span>' +
            '<span class="audit-status-badge ' + (isPass ? 'audit-pass' : 'audit-fail') + '">' + (isPass ? 'PASS' : 'FAIL') + '</span>' +
            '<span style="font-size:0.65rem;color:var(--card-muted);">' + (result.file_count || 0) + ' 文件 · ' + (result.bidder_count || 0) + ' 投标人</span>' +
            '</div>';

        const fileResults = result.file_results || [];
        if (!fileResults.length) {
            html += '<span style="font-size:0.65rem;color:var(--card-muted);">无详细结果</span>';
            return html;
        }

        // Group flat file_results by bidder -> file -> function
        const bidders = {};
        for (const fr of fileResults) {
            const bName = fr.bidder_label || '未知';
            const fName = fr.filename || '?';
            if (!bidders[bName]) bidders[bName] = {};
            if (!bidders[bName][fName]) bidders[bName][fName] = [];
            bidders[bName][fName].push(fr);
        }

        // Compute bidder-level aggregate score
        for (const [bName, files] of Object.entries(bidders)) {
            let allScores = [];
            let bFail = false;
            for (const [fName, funcs] of Object.entries(files)) {
                for (const fn of funcs) {
                    if (fn.score != null) allScores.push(fn.score);
                    if (fn.status === 'error' || fn.status === 'skipped') bFail = true;
                }
            }
            const bScore = allScores.length ? (allScores.reduce((a, b) => a + b, 0) / allScores.length) : 0;
            const bPass = !bFail && allScores.length > 0;

            html += '<div style="margin-bottom:8px;border:1px solid var(--card-border);border-radius:6px;padding:6px 8px;background:#f8fafc;">' +
                '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;margin-bottom:4px;">' +
                '<strong style="font-size:0.7rem;">' + escapeHtml(bName) + '</strong>' +
                '<span style="font-size:0.65rem;">得分: <b>' + bScore.toFixed(1) + '</b></span>' +
                '<span class="audit-status-badge ' + (bPass ? 'audit-pass' : 'audit-fail') + '" style="font-size:0.6rem;padding:0 6px;border-radius:8px;">' + (bPass ? 'PASS' : 'FAIL') + '</span>' +
                '</div>';

            for (const [fName, funcs] of Object.entries(files)) {
                html += '<div style="font-size:0.62rem;padding:4px 0;border-top:1px solid var(--card-border);">' +
                    '<div style="font-weight:600;">📄 ' + escapeHtml(fName) + '</div>' +
                    '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:2px;">';
                for (const fn of funcs) {
                    const fnLabel = _auditFunctionLabels[fn.function_name] || fn.function_name;
                    const fnScore = fn.score != null ? parseFloat(fn.score).toFixed(1) : '-';
                    const fnOk = fn.status === 'success';
                    html += '<span style="background:' + (fnOk ? '#dcfce7' : (fn.status === 'error' ? '#fef2f2' : '#fef3c7')) + ';border-radius:4px;padding:1px 6px;font-size:0.6rem;">' +
                        escapeHtml(fnLabel) + ': <b>' + fnScore + '</b> ' + (fnOk ? '✅' : (fn.status === 'skipped' ? '⏭' : '❌')) +
                        '</span>';
                }
                html += '</div></div>';
            }
            html += '</div>';
        }
        return html;
    }

    // ── Audit Modal (preflight + progress) ──
    var _auditSSE = null;
    var _auditModalActive = false;

    async function showAuditModal() {
        const pid = window._currentProjectId || currentProjectId;
        if (!pid) {
            alert('请先在"项目管理"中打开一个项目，然后再使用全量审计功能。');
            return;
        }
        if (_auditModalActive) return;
        _auditModalActive = true;

        let folders = [], config = [];
        try {
            const [fRes, cRes] = await Promise.all([
                fetch('/admin/projects/' + pid + '/folders', { credentials: 'include' }),
                fetch('/audit/config', { credentials: 'include' })
            ]);
            if (fRes.ok) folders = (await fRes.json()).folders || [];
            if (cRes.ok) { const cRaw = await cRes.json(); config = Array.isArray(cRaw) ? cRaw : (cRaw.data || cRaw.configs || []); }
        } catch (_) { showToast('加载项目数据失败', 'error'); _auditModalActive = false; return; }

        if (!folders.length) {
            showToast('该项目没有文件夹', 'error');
            _auditModalActive = false;
            return;
        }

        const defaultFuncs = {};
        if (Array.isArray(config)) {
            for (const c of config) {
                defaultFuncs[c.function_name] = c.enabled_by_default !== false;
            }
        }
        if (!Object.keys(defaultFuncs).length) {
            for (const k of Object.keys(_auditFunctionLabels)) {
                defaultFuncs[k] = true;
            }
        }

        const overlay = document.createElement('div');
        overlay.className = 'custom-modal-overlay';
        overlay.id = 'auditModalOverlay';
        overlay.style.zIndex = '10000';

        const folderCheckboxes = folders.map((f, i) => {
            const fid = f.id || f.folder_id || i;
            const fname = f.name || f.folder_name || '文件夹 ' + fid;
            return '<label style="display:block;font-size:0.72rem;margin:2px 0;"><input type="checkbox" class="audit-folder-cb" value="' + fid + '" checked> ' + escapeHtml(fname) + '</label>';
        }).join('');

        const funcToggles = Object.entries(defaultFuncs).map(([fn, enabled]) => {
            const label = _auditFunctionLabels[fn] || fn;
            return '<label style="font-size:0.72rem;margin:2px 6px;"><input type="checkbox" class="audit-func-cb" data-fn="' + fn + '" ' + (enabled ? 'checked' : '') + '> ' + escapeHtml(label) + '</label>';
        }).join('');

        overlay.innerHTML =
            '<div class="custom-modal" style="max-width:650px;max-height:90vh;overflow-y:auto;">' +
            '<h3 style="margin-bottom:8px;">📋 全量审计</h3>' +
            '<div id="auditModalBody">' +
            // Preflight phase
            '<div id="auditPreflightPhase">' +
            '<div style="margin-bottom:10px;">' +
            '<strong style="font-size:0.75rem;">选择文件夹:</strong>' +
            '<div style="margin-top:4px;max-height:150px;overflow-y:auto;border:1px solid var(--card-border);border-radius:4px;padding:6px;">' +
            folderCheckboxes +
            '</div>' +
            '<label style="font-size:0.68rem;margin-top:2px;display:block;"><input type="checkbox" id="auditFolderToggleAll" checked> 全选/取消</label>' +
            '</div>' +
            '<div style="margin-bottom:10px;">' +
            '<strong style="font-size:0.75rem;">审计功能:</strong>' +
            '<div style="margin-top:4px;display:flex;flex-wrap:wrap;gap:2px;">' + funcToggles + '</div>' +
            '</div>' +
            '<div style="margin-bottom:10px;">' +
            '<label style="font-size:0.72rem;"><input type="checkbox" id="auditExtractOnDemand" checked> 按需提取结构化数据</label>' +
            '</div>' +
            '<div id="auditPreflightStatus" style="font-size:0.7rem;color:var(--card-muted);margin-bottom:8px;">点击"预检文件"查看可审计的文档...</div>' +
            '<div style="display:flex;gap:8px;">' +
            '<button id="auditStartBtn" class="file-btn" style="background:#7c3aed;color:#fff;border-color:#6d28d9;flex:1;padding:8px;">🔍 预检文件</button>' +
            '<button id="auditCancelBtn" class="file-btn" style="padding:8px;">取消</button>' +
            '</div>' +
            '</div>' +
            // File selection phase (shown after preflight)
            '<div id="auditFileSelectPhase" style="display:none;">' +
            '<div style="margin-bottom:8px;font-size:0.72rem;">' +
            '<span id="auditFileSelectSummary"></span>' +
            '<label style="margin-left:12px;font-size:0.68rem;"><input type="checkbox" id="auditFileToggleAll" checked> 全选文档</label>' +
            '</div>' +
            '<div id="auditFileSelectList" style="max-height:200px;overflow-y:auto;border:1px solid var(--card-border);border-radius:4px;padding:6px;margin-bottom:8px;font-size:0.68rem;"></div>' +
            '<div id="auditFileSkipped" style="font-size:0.62rem;color:var(--card-muted);margin-bottom:8px;"></div>' +
            '<div style="display:flex;gap:8px;">' +
            '<button id="auditConfirmStartBtn" class="file-btn" style="background:#22c55e;color:#fff;border-color:#16a34a;flex:1;padding:8px;">✅ 确认开始审计</button>' +
            '<button id="auditBackBtn" class="file-btn" style="padding:8px;">← 返回</button>' +
            '</div>' +
            '</div>' +
            '</div>' +
            // Progress phase
            '<div id="auditProgressPhase" style="display:none;">' +
            '<div style="margin-bottom:8px;">' +
            '<div style="display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:4px;">' +
            '<span id="auditPhaseLabel">准备中...</span>' +
            '<span id="auditProgressPct">0%</span>' +
            '</div>' +
            '<div style="height:6px;background:#e2e8f0;border-radius:3px;overflow:hidden;">' +
            '<div id="auditProgressBar" style="height:100%;width:0%;background:linear-gradient(90deg,#7c3aed,#3b82f6);transition:width .3s;"></div>' +
            '</div>' +
            '</div>' +
            '<div id="auditProgressFiles" style="max-height:350px;overflow-y:auto;font-size:0.68rem;border:1px solid var(--card-border);border-radius:4px;padding:6px;margin-bottom:8px;">' +
            '<span style="color:var(--card-muted);">等待开始...</span>' +
            '</div>' +
            '<button id="auditProgressCloseBtn" class="file-btn" style="width:100%;padding:6px;">隐藏 (完成后会通知您)</button>' +
            '</div>' +
            '</div>' +
            '</div>';
        document.body.appendChild(overlay);
        overlay.onclick = (e) => { if (e.target === overlay) _closeAuditModal(); };

        const toggleAll = overlay.querySelector('#auditFolderToggleAll');
        if (toggleAll) {
            toggleAll.onclick = function() {
                overlay.querySelectorAll('.audit-folder-cb').forEach(function(cb) { cb.checked = toggleAll.checked; });
            };
        }

        overlay.querySelector('#auditCancelBtn').onclick = _closeAuditModal;

        overlay.querySelector('#auditProgressCloseBtn').onclick = function() {
            overlay.style.display = 'none';
            showToast('审计正在后台运行，完成后将通知您', 'info', 4000);
        };

        // Store state for the two-phase flow
        let _auditSelectedFolders = [];
        let _auditEnabledFuncs = [];
        let _auditPreflightData = null;

        overlay.querySelector('#auditStartBtn').onclick = async function() {
            const btn = overlay.querySelector('#auditStartBtn');
            const statusEl = overlay.querySelector('#auditPreflightStatus');
            btn.disabled = true; btn.textContent = '⏳ 预检中...';

            const selectedFolders = [];
            overlay.querySelectorAll('.audit-folder-cb:checked').forEach(function(cb) { selectedFolders.push(parseInt(cb.value)); });

            const enabledFunctions = [];
            overlay.querySelectorAll('.audit-func-cb:checked').forEach(function(cb) { enabledFunctions.push(cb.dataset.fn); });

            if (!selectedFolders.length) {
                statusEl.innerHTML = '<span style="color:#ef4444;">请至少选择一个文件夹</span>';
                btn.disabled = false; btn.textContent = '🔍 预检文件';
                return;
            }
            if (!enabledFunctions.length) {
                statusEl.innerHTML = '<span style="color:#ef4444;">请至少选择一个审计功能</span>';
                btn.disabled = false; btn.textContent = '🔍 预检文件';
                return;
            }

            _auditSelectedFolders = selectedFolders;
            _auditEnabledFuncs = enabledFunctions;

            try {
                const pfRes = await fetch('/audit/preflight', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include',
                    body: JSON.stringify({ folder_ids: selectedFolders })
                });
                const pfData = await pfRes.json();
                if (!pfRes.ok) {
                    statusEl.innerHTML = '<span style="color:#ef4444;">' + _icon('cancel') + ' ' + (pfData.error || '预检失败') + '</span>';
                    btn.disabled = false; btn.textContent = '🔍 预检文件';
                    return;
                }
                _auditPreflightData = pfData;

                // Group files by folder
                const files = pfData.files || [];
                const byFolder = {};
                for (const f of files) {
                    if (!byFolder[f.folder_id]) byFolder[f.folder_id] = [];
                    byFolder[f.folder_id].push(f);
                }

                // Render file selection
                const listEl = overlay.querySelector('#auditFileSelectList');
                let listHtml = '';
                for (const fid of selectedFolders) {
                    const folderFiles = byFolder[fid] || [];
                    if (!folderFiles.length) continue;
                    listHtml += '<div style="font-weight:600;margin-top:4px;font-size:0.7rem;color:var(--card-muted);">📁 ' + escapeHtml(folderFiles[0].folder_name || ('文件夹 ' + fid)) + '</div>';
                    for (const f of folderFiles) {
                        if (f.status === 'skipped') continue; // non-doc files shown separately
                        const disabled = f.status === 'missing' ? ' disabled' : '';
                        const checked = f.status === 'ready' ? ' checked' : '';
                        const label = f.status === 'missing' ? ' ⚠️未提取' : '';
                        listHtml += '<label style="display:block;font-size:0.65rem;margin:1px 0;' + (f.status === 'missing' ? 'color:var(--card-muted);' : '') + '">' +
                            '<input type="checkbox" class="audit-file-cb" data-file-id="' + f.file_id + '" data-folder="' + fid + '"' + checked + disabled + '> ' +
                            escapeHtml(f.filename) + label + '</label>';
                    }
                }
                listEl.innerHTML = listHtml || '<span style="color:var(--card-muted);">没有可审计的文档文件</span>';

                // Show skipped (non-doc) files
                const skipped = files.filter(f => f.status === 'skipped');
                const skippedEl = overlay.querySelector('#auditFileSkipped');
                if (skipped.length) {
                    skippedEl.innerHTML = '已排除非文档文件: ' + skipped.map(f => '<span title="' + escapeHtml(f.reason || '') + '">' + escapeHtml(f.filename) + '</span>').join(', ');
                } else {
                    skippedEl.innerHTML = '';
                }

                // Summary
                const summaryEl = overlay.querySelector('#auditFileSelectSummary');
                summaryEl.innerHTML = '已找到 <b>' + (pfData.ready_count || 0) + '</b> 个就绪文档' +
                    (pfData.missing_count ? ', <span style="color:#f59e0b;">' + pfData.missing_count + ' 个未提取</span>' : '') +
                    (pfData.skipped_count ? ', <span style="color:var(--card-muted);">' + pfData.skipped_count + ' 个非文档已排除</span>' : '');

                // Toggle all
                const toggleAll = overlay.querySelector('#auditFileToggleAll');
                if (toggleAll) {
                    toggleAll.checked = true;
                    toggleAll.onclick = function() {
                        overlay.querySelectorAll('.audit-file-cb:not([disabled])').forEach(function(cb) { cb.checked = toggleAll.checked; });
                    };
                }

                // Hide preflight, show file selection
                overlay.querySelector('#auditPreflightPhase').style.display = 'none';
                overlay.querySelector('#auditFileSelectPhase').style.display = 'block';

                // Back button
                overlay.querySelector('#auditBackBtn').onclick = function() {
                    overlay.querySelector('#auditFileSelectPhase').style.display = 'none';
                    overlay.querySelector('#auditPreflightPhase').style.display = 'block';
                    btn.disabled = false; btn.textContent = '🔍 预检文件';
                    statusEl.innerHTML = '点击"预检文件"查看可审计的文档...';
                };

                // Confirm start button
                overlay.querySelector('#auditConfirmStartBtn').onclick = async function() {
                    const confirmBtn = overlay.querySelector('#auditConfirmStartBtn');
                    confirmBtn.disabled = true; confirmBtn.textContent = '⏳ 启动中...';

                    const selectedFileIds = [];
                    overlay.querySelectorAll('.audit-file-cb:checked').forEach(function(cb) {
                        selectedFileIds.push(parseInt(cb.dataset.fileId));
                    });

                    if (!selectedFileIds.length) {
                        alert('请至少选择一个文件');
                        confirmBtn.disabled = false; confirmBtn.textContent = '✅ 确认开始审计';
                        return;
                    }

                    const extractOnDemand = overlay.querySelector('#auditExtractOnDemand').checked;
                    const startRes = await fetch('/audit/start', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include',
                        body: JSON.stringify({
                            folder_ids: _auditSelectedFolders,
                            file_ids: selectedFileIds,
                            enabled_functions: _auditEnabledFuncs,
                            extract_on_demand: extractOnDemand,
                            project_id: pid,
                        })
                    });
                    const startData = await startRes.json();
                    if (!startRes.ok) {
                        alert((startData.error || '启动失败'));
                        confirmBtn.disabled = false; confirmBtn.textContent = '✅ 确认开始审计';
                        return;
                    }

                    const fullBtn = document.getElementById('chatFullAuditBtn');
                    if (fullBtn) fullBtn.disabled = true;
                    const badge = document.getElementById('auditRunningBadge');
                    if (badge) badge.style.display = 'inline';

                    overlay.querySelector('#auditFileSelectPhase').style.display = 'none';
                    overlay.querySelector('#auditProgressPhase').style.display = 'block';
                    _startAuditSSE(startData.run_id, overlay);
                };

            } catch (_) {
                statusEl.innerHTML = '<span style="color:#ef4444;">网络错误</span>';
                btn.disabled = false; btn.textContent = '🔍 预检文件';
            }
        };
    }

    // Expose to global scope for inline onclick
    window._showAuditModal = showAuditModal;

    function _closeAuditModal() {
        const overlay = document.getElementById('auditModalOverlay');
        if (overlay) overlay.remove();
        _auditModalActive = false;
        if (_auditSSE) { _auditSSE.close(); _auditSSE = null; }
    }

    function _startAuditSSE(runId, overlay) {
        if (_auditSSE) _auditSSE.close();
        const es = new EventSource('/audit/progress/' + runId);
        _auditSSE = es;

        const filesContainer = overlay.querySelector('#auditProgressFiles');
        const phaseLabel = overlay.querySelector('#auditPhaseLabel');
        const progressBar = overlay.querySelector('#auditProgressBar');
        const progressPct = overlay.querySelector('#auditProgressPct');

        let fileRows = {};
        let completedFiles = 0;
        let totalFiles = 0;

        function updateProgress() {
            const pct = totalFiles > 0 ? Math.round((completedFiles / totalFiles) * 100) : 0;
            if (progressBar) progressBar.style.width = pct + '%';
            if (progressPct) progressPct.textContent = pct + '%';
        }

        es.onmessage = function(event) {
            let data;
            try { data = JSON.parse(event.data); } catch (_) { return; }

            switch (data.type) {
                case 'connected':
                    if (phaseLabel) phaseLabel.textContent = '已连接，准备审计...';
                    break;

                case 'phase':
                    var phaseMap = { auditing: '审计中...', extracting: '提取数据中...', reporting: '生成报告中...' };
                    if (phaseLabel) phaseLabel.textContent = phaseMap[data.phase] || data.phase || '';
                    break;

                case 'file_start':
                    totalFiles = data.total_files || 0;
                    var key = (data.bidder || '') + '||' + (data.filename || '');
                    if (filesContainer) {
                        var rowDiv = document.createElement('div');
                        rowDiv.className = 'audit-file-row';
                        rowDiv.style.cssText = 'border:1px solid var(--card-border);border-radius:4px;padding:4px 6px;margin-bottom:4px;';
                        rowDiv.innerHTML = '<div style="font-weight:600;font-size:0.65rem;">📁 ' + escapeHtml(data.bidder || '?') + ' / ' + escapeHtml(data.filename || '?') + '</div>' +
                            '<div class="audit-file-dots" style="display:flex;gap:4px;flex-wrap:wrap;margin-top:2px;"></div>';
                        filesContainer.prepend(rowDiv);
                        fileRows[key] = { div: rowDiv, dots: {} };
                    }
                    updateProgress();
                    break;

                case 'function_start':
                    if (phaseLabel) phaseLabel.textContent = '审计: ' + (_auditFunctionLabels[data.function] || data.function) + '...';
                    var fk = (data.bidder || '') + '||' + (data.filename || '');
                    var fr = fileRows[fk];
                    if (fr && fr.dots) {
                        var dotEl = document.createElement('span');
                        dotEl.className = 'audit-func-dot spinning';
                        dotEl.title = (_auditFunctionLabels[data.function] || data.function) + ' ...';
                        dotEl.textContent = '⏳';
                        dotEl.style.cssText = 'font-size:0.6rem;padding:1px 4px;border-radius:3px;background:#fef3c7;';
                        fr.dots[data.function] = dotEl;
                        fr.div.querySelector('.audit-file-dots').appendChild(dotEl);
                    }
                    break;

                case 'function_done':
                    var fdk = (data.bidder || '') + '||' + (data.filename || '');
                    var fdr = fileRows[fdk];
                    if (fdr && fdr.dots && fdr.dots[data.function]) {
                        var el = fdr.dots[data.function];
                        var isSuccess = data.status === 'success';
                        el.textContent = isSuccess ? '✅' : '❌';
                        el.className = 'audit-func-dot';
                        el.title = (_auditFunctionLabels[data.function] || data.function) + ': ' + (data.score != null ? parseFloat(data.score).toFixed(1) : '?');
                        el.style.background = isSuccess ? '#dcfce7' : '#fef2f2';
                    }
                    break;

                case 'file_error':
                    var fek = (data.bidder || '') + '||' + (data.filename || '');
                    var fer = fileRows[fek];
                    if (fer && fer.div) {
                        fer.div.style.borderLeft = '3px solid #ef4444';
                        var errSpan = document.createElement('div');
                        errSpan.style.cssText = 'font-size:0.6rem;color:#ef4444;margin-top:2px;';
                        errSpan.textContent = '❌ ' + (data.error || '错误');
                        fer.div.appendChild(errSpan);
                    }
                    completedFiles++;
                    updateProgress();
                    break;

                case 'complete':
                    if (_auditSSE) { _auditSSE.close(); _auditSSE = null; }
                    if (phaseLabel) phaseLabel.textContent = '✅ 审计完成';
                    if (progressBar) progressBar.style.width = '100%';
                    if (progressPct) progressPct.textContent = '100%';

                    var badge = document.getElementById('auditRunningBadge');
                    if (badge) badge.style.display = 'none';
                    var fullBtn = document.getElementById('chatFullAuditBtn');
                    if (fullBtn) fullBtn.disabled = false;

                    // Show download links if reports were generated
                    var downloadHtml = '';
                    if (data.docx_path) downloadHtml += '<a href="/audit/download/' + runId + '/docx" class="file-btn" style="display:inline-block;margin:4px;">📄 下载DOCX报告</a>';
                    if (data.xlsx_path) downloadHtml += '<a href="/audit/download/' + runId + '/xlsx" class="file-btn" style="display:inline-block;margin:4px;">📊 下载XLSX数据</a>';
                    if (downloadHtml) {
                        var dlDiv = document.createElement('div');
                        dlDiv.style.cssText = 'margin-top:8px;text-align:center;';
                        dlDiv.innerHTML = downloadHtml;
                        if (filesContainer) filesContainer.appendChild(dlDiv);
                    }

                    var overallScore = data.overall_score != null ? parseFloat(data.overall_score).toFixed(1) : '?';
                    var overallStatus = data.overall_status || 'PASS';
                    showToast('审计完成 — 得分 ' + overallScore + ' — ' + (overallStatus === 'PASS' ? '✅ PASS' : '❌ FAIL'), overallStatus === 'PASS' ? 'success' : 'error', 8000);
                    break;

                case 'error':
                    if (phaseLabel) phaseLabel.textContent = '❌ ' + (data.message || '错误');
                    if (_auditSSE) { _auditSSE.close(); _auditSSE = null; }
                    showToast('审计出错: ' + (data.message || ''), 'error', 6000);
                    var badge2 = document.getElementById('auditRunningBadge');
                    if (badge2) badge2.style.display = 'none';
                    var fullBtn2 = document.getElementById('chatFullAuditBtn');
                    if (fullBtn2) fullBtn2.disabled = false;
                    break;

                case 'heartbeat':
                    break;
            }
        };

        es.onerror = function(e) {
            console.error('Audit SSE error:', e);
            if (phaseLabel) phaseLabel.textContent = '❌ 连接中断，请检查审计状态';
            if (_auditSSE) { _auditSSE.close(); _auditSSE = null; }
        };
    }
