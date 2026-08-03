// ── Compliance Module (split from app.js) ──
// Exposed via window.Compliance namespace + legacy window aliases for inline handlers

function _safeHTML(html) {
    if (typeof DOMPurify !== 'undefined') {
        return DOMPurify.sanitize(html);
    }
    var div = document.createElement('div');
    div.textContent = html;
    return div.innerHTML;
}

window.Compliance = {
    _taskIds: {},
    markFeedback: null,
    updateRuleCount: null,
};

(function () {
    const Compliance = window.Compliance;

    function initComplianceChecker() {
        const bidDocInput = document.getElementById('complianceBidDocInput');
        const checkFileInput = document.getElementById('complianceCheckFileInput');
        const selectBidDocBtn = document.getElementById('selectComplianceBidDocBtn');
        const selectCheckFileBtn = document.getElementById('selectComplianceCheckFileBtn');
        const bidDocName = document.getElementById('complianceBidDocName');
        const checkFileNames = document.getElementById('complianceCheckFileNames');
        const extractBtn = document.getElementById('complianceExtractBtn');
        const checkBtn = document.getElementById('complianceCheckBtn');
        const useAIChk = document.getElementById('complianceUseAI');
        const statusSpan = document.getElementById('complianceStatus');
        const rulesPreview = document.getElementById('complianceRulesPreview');
        const rulesList = document.getElementById('complianceRulesList');
        const ruleCount = document.getElementById('complianceRuleCount');
        const resultsPanel = document.getElementById('complianceResultsPanel');

        let selectedBidDoc = null;
        let selectedCheckFiles = [];
        let extractedRulesTaskId = null;

        if (!bidDocInput || !checkFileInput) return;

        selectBidDocBtn.onclick = () => bidDocInput.click();
        selectCheckFileBtn.onclick = () => checkFileInput.click();

        bidDocInput.onchange = () => {
            if (bidDocInput.files.length) {
                selectedBidDoc = bidDocInput.files[0];
                bidDocName.textContent = selectedBidDoc.name;
                extractBtn.disabled = false;
            }
        };

        checkFileInput.onchange = () => {
            selectedCheckFiles = Array.from(checkFileInput.files);
            checkFileNames.textContent = selectedCheckFiles.length
                ? `${selectedCheckFiles.length} \u4e2a\u6587\u4ef6: ${selectedCheckFiles.map(f => f.name).join(', ')}`
                : '';
        };

        extractBtn.onclick = async () => {
            if (!selectedBidDoc) return;
            extractBtn.disabled = true;
            statusSpan.textContent = '\u6b63\u5728\u63d0\u53d6\u89c4\u5219...';
            statusSpan.style.color = '#f39c12';

            const formData = new FormData();
            formData.append('file', selectedBidDoc);
            formData.append('use_ai', useAIChk.checked ? 'true' : 'false');

            try {
                const resp = await fetch('/compliance/extract_rules', { method: 'POST', body: formData });
                const data = await resp.json();
                if (data.success) {
                    extractedRulesTaskId = data.task_id;
                    Compliance._taskIds.extracted = data.task_id;
                    statusSpan.textContent = `\u2705 \u63d0\u53d6\u5b8c\u6210: ${data.total} \u6761\u89c4\u5219 (AI:${data.ai_count} + \u6b63\u5219:${data.regex_count})`;
                    statusSpan.style.color = '#27ae60';
                    renderExtractedRules(data.rules);
                    checkBtn.disabled = false;
                } else {
                    statusSpan.textContent = '\u274c ' + (data.error || '\u63d0\u53d6\u5931\u8d25');
                    statusSpan.style.color = '#e74c3c';
                }
            } catch (e) {
                statusSpan.textContent = '\u274c \u8bf7\u6c42\u5931\u8d25: ' + e.message;
                statusSpan.style.color = '#e74c3c';
            } finally {
                extractBtn.disabled = false;
            }
        };

        checkBtn.onclick = async () => {
            if (!extractedRulesTaskId || !selectedCheckFiles.length) return;
            checkBtn.disabled = true;
            statusSpan.textContent = '\u23f3 \u6b63\u5728\u5ba1\u67e5...';
            statusSpan.style.color = '#f39c12';

            resultsPanel.style.display = 'block';
            resultsPanel.innerHTML = '<p style="font-size:0.75rem;color:var(--card-muted);">\u23f3 \u6b63\u5728\u8fdb\u884c\u5408\u89c4\u5ba1\u67e5\uff0c\u8bf7\u7a0d\u5019...</p>';

            let allResults = [];
            Compliance._taskIds = {};
            for (const file of selectedCheckFiles) {
                const formData = new FormData();
                formData.append('bid_file', file);
                formData.append('rules_task_id', extractedRulesTaskId);
                formData.append('use_ai', useAIChk.checked ? 'true' : 'false');
                formData.append('include_laws', 'true');

                try {
                    const resp = await fetch('/compliance/check', { method: 'POST', body: formData });
                    const data = await resp.json();
                    if (data.status === 'completed' || data.success) {
                        allResults.push({ filename: file.name, ...data });
                        if (data.task_id) Compliance._taskIds[file.name] = data.task_id;
                    } else if (data.task_id) {
                        Compliance._taskIds[file.name] = data.task_id;
                        const r = await _pollComplianceResult(data.task_id, file.name);
                        if (r) allResults.push(r);
                    }
                } catch (e) {
                    allResults.push({ filename: file.name, error: e.message });
                }
            }

            renderComplianceResults(allResults);
            checkBtn.disabled = false;
            statusSpan.textContent = `\u2705 \u5ba1\u67e5\u5b8c\u6210: ${allResults.length} \u4e2a\u6587\u4ef6`;
            statusSpan.style.color = '#27ae60';
        };

        async function _pollComplianceResult(taskId, filename) {
            for (let i = 0; i < 60; i++) {
                await new Promise(r => setTimeout(r, 2000));
                try {
                    const resp = await fetch(`/compliance/result/${taskId}`);
                    const data = await resp.json();
                    if (data.status === 'completed') {
                        return { filename, ...data };
                    }
                    if (data.status === 'failed') {
                        return { filename, error: data.error || '\u5ba1\u67e5\u5931\u8d25' };
                    }
                    statusSpan.textContent = `\u23f3 \u5ba1\u67e5\u4e2d... (${(i+1)*2}s)`;
                } catch (e) { /* retry */ }
            }
            return { filename, error: '\u5ba1\u67e5\u8d85\u65f6' };
        }

        let _rulesConfirmed = false;

        function renderExtractedRules(rules) {
            rulesPreview.style.display = 'block';
            ruleCount.textContent = rules.length;
            _rulesConfirmed = false;
            checkBtn.disabled = true;

            const catColors = {
                qualification: '#3498db', technical: '#2ecc71',
                commercial: '#9b59b6', rejection: '#e74c3c', prohibition: '#c0392b'
            };
            const catLabels = {
                qualification: '\u8d44\u8d28', technical: '\u6280\u672f', commercial: '\u5546\u52a1',
                rejection: '\u5e9f\u6807', prohibition: '\u7981\u6b62'
            };

            rulesList.innerHTML = `
                <div style="display:flex;gap:6px;margin-bottom:6px;font-size:0.7rem;">
                    <button id="selectAllRulesBtn" style="padding:2px 8px;border:1px solid var(--card-border);border-radius:4px;background:var(--card-bg);cursor:pointer;">\u5168\u9009</button>
                    <button id="deselectAllRulesBtn" style="padding:2px 8px;border:1px solid var(--card-border);border-radius:4px;background:var(--card-bg);cursor:pointer;">\u53d6\u6d88\u5168\u9009</button>
                </div>
                <div id="rulesCheckList"></div>
                <div style="margin-top:6px;display:flex;gap:6px;">
                    <button id="confirmRulesBtn" style="padding:4px 12px;background:#27ae60;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:0.72rem;">\u2705 \u786e\u8ba4\u9009\u4e2d\u7684\u89c4\u5219\u5e76\u5f00\u59cb\u5ba1\u67e5</button>
                    <span id="rulesConfirmStatus" style="font-size:0.68rem;color:var(--card-muted);align-self:center;"></span>
                </div>
            `;

            const checkList = document.getElementById('rulesCheckList');
            checkList.innerHTML = rules.map((r, i) => {
                const color = catColors[r.category] || '#95a5a6';
                const label = catLabels[r.category] || r.category;
                const sev = r.severity_if_violated || 'violation';
                const sevColors = { critical: '#e74c3c', violation: '#e67e22', warning: '#f39c12', pass: '#27ae60' };
                const desc = r.description ? r.description.replace(/</g,'&lt;') : r.original_text ? r.original_text.substr(0,120).replace(/</g,'&lt;') : '';
                return `<div data-idx="${i}" style="display:flex;align-items:flex-start;gap:6px;padding:3px 0;border-bottom:1px solid #eee;font-size:0.7rem;">
                    <input type="checkbox" class="rule-cb" checked style="margin-top:2px;flex-shrink:0;">
                    <span style="background:${color};color:#fff;padding:0 4px;border-radius:3px;font-size:0.65rem;flex-shrink:0;">${label}</span>
                    <span style="flex:1;">${desc}</span>
                    <span style="color:${sevColors[sev]||'#888'};font-size:0.65rem;flex-shrink:0;">${sev}</span>
                </div>`;
            }).join('');

            document.getElementById('selectAllRulesBtn').onclick = () => {
                checkList.querySelectorAll('.rule-cb').forEach(cb => cb.checked = true);
                _rulesConfirmed = false;
                checkBtn.disabled = true;
                document.getElementById('rulesConfirmStatus').textContent = '';
            };
            document.getElementById('deselectAllRulesBtn').onclick = () => {
                checkList.querySelectorAll('.rule-cb').forEach(cb => cb.checked = false);
                _rulesConfirmed = false;
                checkBtn.disabled = true;
                document.getElementById('rulesConfirmStatus').textContent = '';
            };

            document.getElementById('confirmRulesBtn').onclick = async () => {
                const selected = [];
                const items = checkList.querySelectorAll('[data-idx]');
                for (const el of items) {
                    const cb = el.querySelector('.rule-cb');
                    if (cb && cb.checked) {
                        const idx = parseInt(el.dataset.idx);
                        selected.push(rules[idx]);
                    }
                }
                if (!selected.length) {
                    document.getElementById('rulesConfirmStatus').textContent = '\u26a0\ufe0f \u8bf7\u81f3\u5c11\u9009\u62e9\u4e00\u6761\u89c4\u5219';
                    return;
                }
                const statusEl = document.getElementById('rulesConfirmStatus');
                statusEl.textContent = '\u23f3 \u4fdd\u5b58\u9009\u62e9...';
                try {
                    const resp = await fetch('/compliance/rules/' + extractedRulesTaskId, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ rules: selected }),
                    });
                    const data = await resp.json();
                    if (data.success) {
                        _rulesConfirmed = true;
                        checkBtn.disabled = false;
                        ruleCount.textContent = selected.length;
                        statusEl.textContent = `\u2705 \u5df2\u786e\u8ba4 ${selected.length} \u6761\u89c4\u5219\uff0c\u53ef\u5f00\u59cb\u5ba1\u67e5`;
                        statusEl.style.color = '#27ae60';
                    } else {
                        statusEl.textContent = '\u274c ' + (data.error || '\u4fdd\u5b58\u5931\u8d25');
                        statusEl.style.color = '#e74c3c';
                    }
                } catch (e) {
                    statusEl.textContent = '\u274c \u7f51\u7edc\u9519\u8bef: ' + e.message;
                    statusEl.style.color = '#e74c3c';
                }
            };
        }

        function renderComplianceResults(allResults) {
            resultsPanel.style.display = 'block';
            let html = '';
            for (const r of allResults) {
                if (r.error) {
                    html += `<div style="margin:4px 0;padding:8px;background:#fdf2f2;border-radius:6px;font-size:0.72rem;">
                        <strong style="color:#e74c3c;">\u274c ${r.filename}</strong>: ${r.error}</div>`;
                    continue;
                }
                const s = r.summary || {};
                const vStyle = s.critical > 0 ? '#e74c3c' : s.violation > 0 ? '#e67e22' : '#27ae60';
                const vText = s.critical > 0 ? '\u26a0\ufe0f \u4e25\u91cd\u8fdd\u89c4' : s.violation > 0 ? '\u26a0\ufe0f \u5b58\u5728\u8fdd\u89c4' : '\u2705 \u901a\u8fc7';
                const fileIdx = allResults.indexOf(r);
                html += `<div class="compliance-result-card" style="margin:4px 0;padding:8px;background:var(--card-bg);border:1px solid var(--card-border);border-radius:6px;font-size:0.72rem;" data-filename="${r.bid_name || r.filename}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <strong>${r.bid_name || r.filename}</strong>
                        <span style="color:${vStyle};">${vText}</span>
                    </div>
                    <div style="margin:4px 0;display:flex;gap:10px;font-size:0.68rem;color:var(--card-muted);">
                        <span>\u89c4\u5219: ${r.rule_count || 0}\u6761</span>
                        <span style="color:#27ae60;">\u2705 ${s.pass||0}</span>
                        <span style="color:#f39c12;">\u26a0\ufe0f ${s.warning||0}</span>
                        <span style="color:#e67e22;">\u274c ${s.violation||0}</span>
                        <span style="color:#e74c3c;">\ud83d\uded1 ${s.critical||0}</span>
                    </div>`;
                if (r.report_html) {
                    html += `<details style="margin-top:4px;"><summary style="cursor:pointer;font-size:0.68rem;color:#2980b9;">\ud83d\udccb \u67e5\u770b\u8be6\u7ec6\u62a5\u544a</summary>
                        <div style="margin-top:4px;padding:6px;background:var(--bg-color);border-radius:4px;">${_safeHTML(r.report_html)}</div></details>`;
                }
                html += `<div class="compliance-feedback-row" style="margin-top:6px;padding:6px;background:var(--bg-color);border-radius:4px;border:1px dashed var(--card-border);">
                    <div style="font-size:0.65rem;color:var(--card-muted);margin-bottom:4px;">\ud83d\udcdd <strong>\u5f3a\u5236\u53cd\u9988</strong> \u2014 \u6b64\u68c0\u67e5\u7ed3\u679c\u662f\u5426\u6b63\u786e\uff1f</div>
                    <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
                        <label style="font-size:0.65rem;display:flex;align-items:center;gap:3px;cursor:pointer;">
                            <input type="radio" name="feedback-${fileIdx}" value="true_violation" onchange="window.markFeedback(this, '${r.bid_name || r.filename}')">
                            <span style="color:#e74c3c;">\u2705 \u786e\u5b9e\u8fdd\u89c4</span>
                        </label>
                        <label style="font-size:0.65rem;display:flex;align-items:center;gap:3px;cursor:pointer;">
                            <input type="radio" name="feedback-${fileIdx}" value="false_positive" onchange="window.markFeedback(this, '${r.bid_name || r.filename}')">
                            <span style="color:#f39c12;">\u274c AI\u8bef\u5224</span>
                        </label>
                        <label style="font-size:0.65rem;display:flex;align-items:center;gap:3px;cursor:pointer;">
                            <input type="radio" name="feedback-${fileIdx}" value="not_matter" onchange="window.markFeedback(this, '${r.bid_name || r.filename}')">
                            <span style="color:#7f8c8d;">\u2796 \u65e0\u5173\u7d27\u8981</span>
                        </label>
                    </div>
                    <input type="text" class="feedback-explain" data-filename="${r.bid_name || r.filename}"
                        placeholder="\u7b80\u8981\u8bf4\u660e\u539f\u56e0\uff08\u5fc5\u586b\uff09" style="display:none;width:100%;margin-top:4px;font-size:0.65rem;padding:3px 6px;border:1px solid var(--card-border);border-radius:4px;background:var(--card-bg);">
                    <button class="feedback-submit-btn" data-filename="${r.bid_name || r.filename}"
                        style="display:none;margin-top:4px;font-size:0.62rem;padding:2px 8px;background:#2980b9;color:#fff;border:none;border-radius:4px;cursor:pointer;">\u63d0\u4ea4\u53cd\u9988</button>
                    <span class="feedback-saved" style="display:none;margin-left:8px;font-size:0.62rem;color:#27ae60;">\u2705 \u5df2\u4fdd\u5b58</span>
                </div>`;
                html += '</div>';
            }
            resultsPanel.innerHTML = html;

            document.querySelectorAll('.feedback-submit-btn').forEach(btn => {
                btn.onclick = async function() {
                    const fn = this.dataset.filename;
                    const card = this.closest('.compliance-result-card');
                    const radio = card.querySelector('input[type="radio"]:checked');
                    const explainInput = card.querySelector('.feedback-explain');
                    const savedSpan = card.querySelector('.feedback-saved');

                    if (!radio) { alert('\u8bf7\u9009\u62e9\u4e00\u4e2a\u5224\u5b9a'); return; }
                    const explain = explainInput.value.trim();
                    if (!explain) { alert('\u8bf7\u586b\u5199\u5224\u65ad\u539f\u56e0'); return; }

                    this.disabled = true;
                    this.textContent = '\u4fdd\u5b58\u4e2d...';
                    try {
                        const taskId = (Compliance._taskIds && Compliance._taskIds[fn]) || '';
                        const resp = await fetch('/compliance/feedback', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                task_id: taskId,
                                check_file_name: fn,
                                user_verdict: radio.value,
                                user_explain: explain,
                            }),
                        });
                        const data = await resp.json();
                        if (data.success) {
                            savedSpan.style.display = 'inline';
                            this.textContent = '\u5df2\u63d0\u4ea4';
                            this.style.background = '#27ae60';
                            radio.disabled = true;
                            card.querySelectorAll('input[type="radio"]').forEach(r => r.disabled = true);
                            explainInput.disabled = true;
                        } else {
                            alert('\u4fdd\u5b58\u5931\u8d25: ' + (data.error || '\u672a\u77e5\u9519\u8bef'));
                            this.disabled = false;
                            this.textContent = '\u63d0\u4ea4\u53cd\u9988';
                        }
                    } catch (e) {
                        alert('\u8bf7\u6c42\u5931\u8d25: ' + e.message);
                        this.disabled = false;
                        this.textContent = '\u63d0\u4ea4\u53cd\u9988';
                    }
                };
            });
        }

        // Global function for radio onchange to show/hide explain + submit
        window.markFeedback = Compliance.markFeedback = function(radio, filename) {
            const card = radio.closest('.compliance-result-card');
            const explainInput = card.querySelector('.feedback-explain');
            const submitBtn = card.querySelector('.feedback-submit-btn');
            if (radio.checked) {
                explainInput.style.display = 'block';
                submitBtn.style.display = 'inline-block';
                explainInput.focus();
            }
        };

        // Expose updateRuleCount
        window.updateRuleCount = Compliance.updateRuleCount = function() {
            const count = rulesList.children.length;
            ruleCount.textContent = count;
            const rules = [];
            for (const el of rulesList.children) {
                const spans = el.querySelectorAll('span');
                if (spans.length >= 2) {
                    rules.push({
                        description: spans[1].textContent,
                        category: spans[0].textContent,
                    });
                }
            }
        };
    }

    // ── Public API ──
    Compliance.init = initComplianceChecker;

    // Wire on DOMContentLoaded (in case app.js hasn't reached the init calls yet)
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () {
            initComplianceChecker();
        });
    } else {
        initComplianceChecker();
    }

})();

// ── Law Version UI (U3-FE) ──
(function () {
    const Law = window.Compliance.Law = {
        loadVersions: loadVersions,
        showTimeline: showTimeline,
        showDiff: showDiff,
        activateVersion: activateVersion,
        showVersionArticles: showVersionArticles,
    };

    function loadVersions(lawId, container) {
        container.innerHTML = '<span style="font-size:0.7rem;color:var(--card-muted);">\u23f3 \u52a0\u8f7d\u7248\u672c...</span>';
        fetch('/compliance/laws/' + lawId + '/versions', { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                renderVersionList(data.versions, container, lawId);
            })
            .catch(e => {
                container.innerHTML = '<span style="font-size:0.68rem;color:#e74c3c;">\u7248\u672c\u52a0\u8f7d\u5931\u8d25: ' + e.message + '</span>';
            });
    }

    function renderVersionList(versions, container, lawId) {
        if (!versions || !versions.length) {
            container.innerHTML = '<span style="font-size:0.7rem;color:var(--card-muted);">\u6682\u65e0\u7248\u672c\u6570\u636e</span>';
            return;
        }
        let html = '<div style="max-height:300px;overflow-y:auto;">';
        versions.forEach((v, i) => {
            const isCur = v.is_current;
            html += '<div style="display:flex;align-items:center;gap:6px;padding:4px 0;border-bottom:1px solid var(--card-border);font-size:0.7rem;">';
            html += '<span style="' + (isCur ? 'color:#27ae60;font-weight:600;' : 'color:var(--card-muted);') + '">' + hlVerse(v.version_label) + '</span>';
            if (v.version_date) html += '<span style="font-size:0.62rem;color:var(--card-muted);">' + v.version_date + '</span>';
            if (isCur) html += '<span style="background:#27ae60;color:#fff;padding:0 4px;border-radius:3px;font-size:0.58rem;">\u5f53\u524d</span>';
            if (v.change_summary) html += '<span style="font-size:0.62rem;color:var(--card-muted);">' + v.change_summary + '</span>';
            html += '<span style="font-size:0.6rem;color:var(--card-muted);">(' + (v.article_count || 0) + '\u6761)</span>';
            html += '<span style="flex:1;"></span>';
            if (!isCur) {
                html += '<button class="law-activate-btn" data-vid="' + v.id + '" style="font-size:0.58rem;padding:1px 6px;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">\u6fc0\u6d3b</button>';
            }
            html += '<button class="law-detail-btn" data-vid="' + v.id + '" style="font-size:0.58rem;padding:1px 6px;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">\u6761\u6b3e</button>';
            if (i > 0 || versions.length > 1) {
                html += '<button class="law-diff-btn" data-from="' + v.id + '" data-to="' + versions[0].id + '" style="font-size:0.58rem;padding:1px 6px;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">\u2194\ufe0e Diff</button>';
            }
            html += '</div>';
        });
        html += '</div>';
        if (versions.length >= 2) {
            html += '<div style="margin-top:6px;font-size:0.65rem;display:flex;gap:6px;align-items:center;">';
            html += '<button class="law-timeline-btn" style="padding:2px 8px;border:1px solid var(--card-border);border-radius:4px;background:var(--card-bg);cursor:pointer;">📜 版本时间线</button>';
            html += '<button class="law-impact-btn" style="padding:2px 8px;border:1px solid #e74c3c;color:#e74c3c;border-radius:4px;background:var(--card-bg);cursor:pointer;">⚠️ 变更影响评估</button>';
            html += '</div>';
        }
        container.innerHTML = html;

        container.querySelectorAll('.law-activate-btn').forEach(btn => {
            btn.onclick = () => activateVersion(lawId, parseInt(btn.dataset.vid), container);
        });
        container.querySelectorAll('.law-detail-btn').forEach(btn => {
            btn.onclick = () => showVersionArticles(parseInt(btn.dataset.vid));
        });
        container.querySelectorAll('.law-diff-btn').forEach(btn => {
            btn.onclick = () => showDiff(lawId, parseInt(btn.dataset.from), parseInt(btn.dataset.to));
        });
        if (versions.length >= 2) {
            const tlBtn = container.querySelector('.law-timeline-btn');
            if (tlBtn) tlBtn.onclick = () => showTimeline(versions);
            const impactBtn = container.querySelector('.law-impact-btn');
            if (impactBtn) impactBtn.onclick = () => showLawImpact(lawId);
        }
    }

    function showTimeline(versions) {
        const sorted = [...versions].sort((a, b) => (a.version_date || '').localeCompare(b.version_date || ''));
        let html = '<div class="modal" id="lawTimelineModal" style="display:flex;">';
        html += '<div class="modal-content" style="max-width:700px;max-height:80vh;overflow-y:auto;">';
        html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">';
        html += '<strong style="font-size:0.85rem;">\ud83d\udcdc \u7248\u672c\u65f6\u95f4\u7ebf</strong>';
        html += '<button onclick="this.closest(\'.modal\').remove()" style="border:none;background:none;cursor:pointer;font-size:1.2rem;">\u2716</button>';
        html += '</div>';
        html += '<div style="position:relative;padding-left:24px;border-left:2px solid var(--card-border);margin-left:8px;">';
        sorted.forEach((v, i) => {
            html += '<div style="position:relative;margin-bottom:12px;">';
            html += '<div style="position:absolute;left:-30px;top:4px;width:10px;height:10px;border-radius:50%;background:' + (v.is_current ? '#27ae60' : '#95a5a6') + ';border:2px solid var(--card-bg);"></div>';
            html += '<div style="font-size:0.75rem;font-weight:600;">' + hlVerse(v.version_label) + (v.is_current ? ' <span style="color:#27ae60;font-size:0.6rem;">(\u5f53\u524d)</span>' : '') + '</div>';
            if (v.version_date) html += '<div style="font-size:0.62rem;color:var(--card-muted);">' + v.version_date + '</div>';
            if (v.change_summary) html += '<div style="font-size:0.65rem;color:var(--card-muted);">' + v.change_summary + '</div>';
            html += '</div>';
        });
        html += '</div></div></div>';
        document.body.insertAdjacentHTML('beforeend', html);
        const modal = document.getElementById('lawTimelineModal');
        modal.addEventListener('click', function (e) { if (e.target === modal) modal.remove(); });
    }

    function showLawImpact(lawId) {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.style.display = 'flex';
        modal.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.5);z-index:10000;display:flex;align-items:center;justify-content:center;';
        modal.innerHTML = '<div class="modal-content" style="background:var(--card-bg);border-radius:12px;padding:20px;max-width:650px;width:90%;max-height:80vh;overflow-y:auto;">' +
            '<span style="float:right;cursor:pointer;font-size:1.2rem;" onclick="this.closest(\'.modal\').remove()">&times;</span>' +
            '<h4 style="margin:0 0 8px;">⚠️ 法规变更影响评估</h4>' +
            '<span style="font-size:0.7rem;color:var(--card-muted);">正在分析受影响案例和模板...</span></div>';
        document.body.appendChild(modal);
        modal.addEventListener('click', function (e) { if (e.target === modal) modal.remove(); });

        fetch('/compliance/laws/monitor/impact', {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ law_id: lawId }),
        })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                const impact = data.data || data;
                let html = '<span style="float:right;cursor:pointer;font-size:1.2rem;" onclick="this.closest(\'.modal\').remove()">&times;</span>';
                html += '<h4 style="margin:0 0 8px;">⚠️ 法规变更影响评估</h4>';

                html += '<div style="display:flex;gap:10px;margin-bottom:12px;">';
                html += _impactStat('涉及条款', impact.affected_articles || 0, '#8e44ad');
                html += _impactStat('影响案例', impact.total_affected_cases || 0, '#e67e22');
                html += _impactStat('影响模板', impact.total_affected_templates || 0, '#27ae60');
                html += '</div>';

                if (impact.severity_breakdown) {
                    const sb = impact.severity_breakdown;
                    html += '<div style="margin-bottom:10px;font-size:0.65rem;">';
                    html += '<span style="font-weight:600;">严重程度：</span>';
                    html += '<span style="color:#e74c3c;">🚫 ' + (sb.critical || 0) + '</span> ';
                    html += '<span style="color:#e67e22;">❌ ' + (sb.violation || 0) + '</span> ';
                    html += '<span style="color:#f39c12;">⚠️ ' + (sb.warning || 0) + '</span> ';
                    html += '| <span style="color:#e74c3c;">打开 ' + (sb.open || 0) + '</span> ';
                    html += '<span style="color:#27ae60;">已解决 ' + (sb.resolved || 0) + '</span>';
                    html += '</div>';
                }

                html += '<div style="margin-bottom:10px;">';
                html += '<button id="lawImpactGraphBtn" style="font-size:0.62rem;padding:3px 10px;border:1px solid #f39c12;border-radius:4px;background:#fefce8;color:#92400e;cursor:pointer;">🕸️ 查看影响图谱</button>';
                html += '</div>';
                html += '<div id="lawImpactGraphContainer" style="display:none;position:relative;width:100%;height:350px;margin-bottom:10px;border:1px solid var(--card-border);border-radius:6px;overflow:hidden;"></div>';

                if (impact.affected_cases && impact.affected_cases.length) {
                    html += '<div style="margin-bottom:10px;"><strong style="font-size:0.68rem;">📋 受影响案例</strong>';
                    html += '<div style="max-height:180px;overflow-y:auto;font-size:0.62rem;margin-top:4px;">';
                    impact.affected_cases.forEach(c => {
                        const sevCol = c.severity === 'critical' ? '#e74c3c' : c.severity === 'violation' ? '#e67e22' : '#f39c12';
                        html += '<div style="padding:2px 4px;border-bottom:1px solid var(--card-border);display:flex;justify-content:space-between;">';
                        html += '<span>' + hlVerse(c.title) + '</span>';
                        html += '<span style="color:' + sevCol + ';">' + c.severity + (c.is_resolved ? ' ✅' : '') + '</span>';
                        html += '</div>';
                    });
                    html += '</div></div>';
                }

                if (impact.affected_templates && impact.affected_templates.length) {
                    html += '<div><strong style="font-size:0.68rem;">📑 受影响模板</strong>';
                    html += '<div style="font-size:0.62rem;margin-top:4px;">';
                    impact.affected_templates.forEach(t => {
                        html += '<span style="background:var(--bg-color);padding:2px 6px;border-radius:3px;margin:2px;display:inline-block;">' + hlVerse(t.name) + '</span>';
                    });
                    html += '</div></div>';
                }

                modal.querySelector('.modal-content').innerHTML = html;

                setTimeout(function () {
                    var btn = document.getElementById('lawImpactGraphBtn');
                    if (btn) {
                        btn.onclick = function () {
                            var container = document.getElementById('lawImpactGraphContainer');
                            if (!container) return;
                            if (container.style.display === 'block') {
                                container.style.display = 'none';
                                btn.textContent = '🕸️ 查看影响图谱';
                                return;
                            }
                            container.style.display = 'block';
                            container.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:0.7rem;">加载中...</span>';
                            btn.textContent = '🕸️ 隐藏图谱';
                            fetch('/api/graph/law-impact?law_id=' + lawId + '&depth=3', { credentials: 'include' })
                                .then(function (r) { return r.json(); })
                                .then(function (gData) {
                                    if (gData.success && gData.nodes && gData.nodes.length) {
                                        renderGraph(container, gData);
                                    } else {
                                        container.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:0.7rem;">暂无级联影响数据</span>';
                                    }
                                })
                                .catch(function () {
                                    container.innerHTML = '<span style="color:#ef4444;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:0.7rem;">加载失败</span>';
                                });
                        };
                    }
                }, 100);
            })
            .catch(e => {
                modal.querySelector('.modal-content').innerHTML =
                    '<span style="float:right;cursor:pointer;font-size:1.2rem;" onclick="this.closest(\'.modal\').remove()">&times;</span>' +
                    '<span style="color:#e74c3c;">影响评估失败: ' + e.message + '</span>';
            });
    }

    function _impactStat(label, value, color) {
        return '<div style="flex:1;padding:8px;background:var(--bg-color);border-radius:6px;text-align:center;border-left:3px solid ' + color + ';">' +
            '<div style="font-size:1.2rem;font-weight:700;color:' + color + ';">' + (value || 0) + '</div>' +
            '<div style="font-size:0.55rem;color:var(--card-muted);">' + label + '</div></div>';
    }

    function showDiff(lawId, fromVid, toVid) {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.style.display = 'flex';
        modal.innerHTML = '<div class="modal-content" style="max-width:90vw;max-height:85vh;width:900px;overflow:auto;background:var(--card-bg);border-radius:8px;padding:16px;"><span style="font-size:0.85rem;">\u23f3 \u52a0\u8f7d\u5dee\u5f02...</span></div>';
        document.body.appendChild(modal);
        modal.addEventListener('click', function (e) { if (e.target === modal) modal.remove(); });

        fetch('/compliance/laws/' + lawId + '/diff?from=' + fromVid + '&to=' + toVid, { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                const d = data.data || data;
                renderDiffModal(modal, d, lawId);
            })
            .catch(e => {
                modal.querySelector('.modal-content').innerHTML = '<span style="color:#e74c3c;font-size:0.85rem;">\u5dee\u5f02\u52a0\u8f7d\u5931\u8d25: ' + e.message + '</span><br><button onclick="this.closest(\'.modal\').remove()" style="margin-top:8px;">\u5173\u95ed</button>';
            });
    }

    function renderDiffModal(modal, diffData, lawId) {
        const changes = diffData.changes || [];
        const summary = diffData.summary || {};
        let html = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">';
        html += '<strong style="font-size:0.85rem;">\u2194\ufe0e \u7248\u672c\u5dee\u5f02</strong>';
        html += '<button onclick="this.closest(\'.modal\').remove()" style="border:none;background:none;cursor:pointer;font-size:1.2rem;">\u2716</button>';
        html += '</div>';
        html += '<div style="display:flex;gap:12px;margin-bottom:10px;font-size:0.68rem;">';
        html += '<span style="color:#27ae60;">\u2795 \u65b0\u589e: ' + (summary.added || 0) + '</span>';
        html += '<span style="color:#e74c3c;">\u2796 \u5220\u9664: ' + (summary.removed || 0) + '</span>';
        html += '<span style="color:#e67e22;">\u270f\ufe0f \u4fee\u6539: ' + (summary.changed || 0) + '</span>';
        html += '</div>';
        if (!changes.length) {
            html += '<p style="font-size:0.75rem;color:var(--card-muted);">\u4e24\u4e2a\u7248\u672c\u5b8c\u5168\u76f8\u540c\u3002</p>';
        }
        html += '<div style="max-height:60vh;overflow-y:auto;">';
        changes.forEach(c => {
            const bg = c.status === 'added' ? '#e6ffe6' : c.status === 'removed' ? '#ffe6e6' : '#fff8e6';
            const tag = c.status === 'added' ? '\u2795 \u65b0\u589e' : c.status === 'removed' ? '\u2796 \u5220\u9664' : '\u270f\ufe0f \u4fee\u6539';
            const tagColor = c.status === 'added' ? '#27ae60' : c.status === 'removed' ? '#e74c3c' : '#e67e22';
            html += '<div style="margin-bottom:8px;background:' + bg + ';border-radius:6px;padding:8px;font-size:0.7rem;">';
            html += '<div style="font-weight:600;margin-bottom:4px;"><span style="background:' + tagColor + ';color:#fff;padding:0 4px;border-radius:3px;font-size:0.6rem;">' + tag + '</span> ' + hlVerse(c.article_label) + '</div>';
            if (c.diff_html) {
                html += '<div style="font-size:0.68rem;overflow-x:auto;">' + _safeHTML(c.diff_html) + '</div>';
            }
            html += '</div>';
        });
        html += '</div>';
        modal.querySelector('.modal-content').innerHTML = html;
    }

    function showVersionArticles(versionId) {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.style.display = 'flex';
        modal.innerHTML = '<div class="modal-content" style="max-width:700px;max-height:85vh;overflow-y:auto;background:var(--card-bg);border-radius:8px;padding:16px;"><span style="font-size:0.85rem;">\u23f3 \u52a0\u8f7d\u6761\u6b3e...</span></div>';
        document.body.appendChild(modal);
        modal.addEventListener('click', function (e) { if (e.target === modal) modal.remove(); });

        fetch('/compliance/laws/0/versions/' + versionId, { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                const v = data.data || data;
                let html = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">';
                html += '<strong style="font-size:0.85rem;">\ud83d\udcdc ' + hlVerse(v.version_label || '') + '</strong>';
                html += '<button onclick="this.closest(\'.modal\').remove()" style="border:none;background:none;cursor:pointer;font-size:1.2rem;">\u2716</button>';
                html += '</div>';
                html += '<div style="font-size:0.68rem;color:var(--card-muted);margin-bottom:8px;">' + (v.version_date || '') + ' | ' + (v.article_count || (v.articles || []).length) + ' \u6761\u6b3e</div>';
                const articles = v.articles || [];
                articles.forEach(a => {
                    html += '<div style="margin-bottom:8px;padding:8px;background:var(--bg-color);border-radius:4px;font-size:0.7rem;">';
                    html += '<div style="font-weight:600;margin-bottom:4px;">' + hlVerse(a.article_label) + '</div>';
                    html += '<div style="color:var(--card-muted);line-height:1.5;">' + (a.article_text || '').replace(/</g, '&lt;').replace(/\n/g, '<br>') + '</div>';
                    if (a.tags && a.tags.length) {
                        html += '<div style="margin-top:4px;">' + a.tags.map(t => '<span style="background:var(--card-border);padding:0 4px;border-radius:3px;font-size:0.58rem;margin-right:3px;">' + t + '</span>').join('') + '</div>';
                    }
                    html += '</div>';
                });
                modal.querySelector('.modal-content').innerHTML = html;
            })
            .catch(e => {
                modal.querySelector('.modal-content').innerHTML = '<span style="color:#e74c3c;font-size:0.85rem;">\u6761\u6b3e\u52a0\u8f7d\u5931\u8d25: ' + e.message + '</span><br><button onclick="this.closest(\'.modal\').remove()" style="margin-top:8px;">\u5173\u95ed</button>';
            });
    }

    function activateVersion(lawId, versionId, container) {
        if (!confirm('\u786e\u8ba4\u6fc0\u6d3b\u6b64\u7248\u672c\uff1f\u5f53\u524d\u7248\u672c\u5c06\u88ab\u66ff\u6362\u3002')) return;
        fetch('/compliance/laws/' + lawId + '/versions/activate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ version_id: versionId }),
        })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                loadVersions(lawId, container);
            })
            .catch(e => alert('\u6fc0\u6d3b\u5931\u8d25: ' + e.message));
    }

    function hlVerse(s) { return (s || '').replace(/</g, '&lt;'); }

    // ── Incremental Check (U9a) ──
    // 300ms debounce for compliance editor text changes.
    // TODO: U9b Tiptap integration — changed_sections should be per-chapter, not full text

    const _incrCheckState = {
        timer: null,
        rulesTaskId: null,
        bidDocName: '',
        DEBOUNCE_MS: 300,
    };

    Compliance.startIncrementalCheck = function (text, bidDocName, rulesTaskId) {
        _incrCheckState.rulesTaskId = rulesTaskId;
        _incrCheckState.bidDocName = bidDocName;

        clearTimeout(_incrCheckState.timer);
        _incrCheckState.timer = setTimeout(() => {
            fetch('/compliance/incremental_check', {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rules_task_id: _incrCheckState.rulesTaskId,
                    bid_doc_name: _incrCheckState.bidDocName,
                    changed_sections: [
                        { id: 'main', title: '文本编辑', content: text },
                    ],
                    use_ai: true,
                }),
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.summary) {
                        const events = data.results || [];
                        const findings = [];
                        events.forEach(ev => {
                            (ev.results || []).forEach(f => {
                                findings.push({ ...f, section_id: ev.section_id, section_title: ev.section_title });
                            });
                        });
                        if (typeof window.Compliance.renderIncrementalResults === 'function') {
                            window.Compliance.renderIncrementalResults(findings, data.summary);
                        }
                    }
                })
                .catch(e => console.error('Incremental check error:', e));
        }, _incrCheckState.DEBOUNCE_MS);
    };

    // Default incremental results renderer (can be overridden)
    Compliance.renderIncrementalResults = function (findings, summary) {
        const container = document.getElementById('complianceResultsPanel');
        if (!container || !findings.length) return;
        const existing = container.querySelector('#incrementalResults');
        if (existing) existing.remove();
        let html = '<div id="incrementalResults" style="margin-top:8px;padding:8px;background:var(--bg-color);border-radius:6px;font-size:0.65rem;">';
        html += '<div style="display:flex;gap:16px;margin-bottom:6px;">';
        html += '<span style="color:#27ae60;">✅ 通过: ' + (summary.pass || 0) + '</span>';
        html += '<span style="color:#f39c12;">⚠️ 警告: ' + (summary.warning || 0) + '</span>';
        html += '<span style="color:#e67e22;">❌ 违规: ' + (summary.violation || 0) + '</span>';
        html += '<span style="color:#e74c3c;">🚫 严重: ' + (summary.critical || 0) + '</span>';
        html += '</div>';
        findings.slice(0, 10).forEach(f => {
            const sc = f.status === 'critical' ? '#e74c3c' : f.status === 'violation' ? '#e67e22' : '#f39c12';
            html += '<div style="display:flex;justify-content:space-between;padding:2px 4px;border-bottom:1px solid var(--card-border);">';
            html += '<span>' + hlVerse(f.section_title || f.function_name || '') + ': ' + hlVerse(f.description || f.message || '') + '</span>';
            html += '<span style="color:' + sc + ';">' + f.status + '</span>';
            html += '</div>';
        });
        if (findings.length > 10) html += '<div style="text-align:center;color:var(--card-muted);">... 还有 ' + (findings.length - 10) + ' 条结果</div>';
        html += '</div>';
        container.insertAdjacentHTML('beforeend', html);
    };

    // ── Compliance Dashboard (U11) ──

    const _dashInit = function () {
        const btn = document.getElementById('complianceDashboardBtn');
        if (btn) btn.onclick = showDashboard;
        const cmpBtn = document.getElementById('complianceCompareBtn');
        if (cmpBtn) cmpBtn.onclick = showCompare;
    };
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _dashInit);
    } else {
        _dashInit();
    }

    function showDashboard() {
        const panel = document.getElementById('complianceDashboardPanel');
        if (!panel) return;
        panel.style.display = 'block';
        panel.innerHTML = '<span style="color:var(--card-muted);font-size:0.7rem;">⏳ 加载仪表盘...</span>';

        fetch('/compliance/dashboard?days=30', { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                const d = data.data || data;
                const ov = d.overall || {};

                let html = '<div style="padding:8px 0;">';

                // Stats row
                html += '<div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap;">';
                html += _dashCard('总审查数', ov.total_runs || 0, '#2980b9');
                const prColor = (ov.pass_rate || 0) >= 70 ? '#27ae60' : (ov.pass_rate || 0) >= 40 ? '#f39c12' : '#e74c3c';
                html += _dashCard('通过率', (ov.pass_rate || 0) + '%', prColor);
                html += _dashCard('均分', ov.avg_score || 0, '#8e44ad');
                html += _dashCard('文件检查', ov.total_file_checks || 0, '#16a085');
                html += '</div>';

                // Score distribution bar
                if (d.score_distribution && d.score_distribution.length) {
                    const maxCnt = Math.max(...d.score_distribution.map(s => s.count), 1);
                    html += '<div style="margin-bottom:10px;"><strong style="font-size:0.68rem;">📊 分数分布</strong>';
                    html += '<div style="display:flex;gap:4px;margin-top:4px;height:30px;align-items:flex-end;">';
                    d.score_distribution.forEach(s => {
                        const pct = Math.max((s.count / maxCnt * 100), 4);
                        const barColor = parseInt(s.bucket) >= 70 ? '#27ae60' : parseInt(s.bucket) >= 50 ? '#f39c12' : '#e74c3c';
                        html += '<div style="flex:1;display:flex;flex-direction:column;align-items:center;">';
                        html += '<span style="font-size:0.52rem;">' + s.count + '</span>';
                        html += '<div style="width:100%;height:' + pct + '%;background:' + barColor + ';border-radius:2px 2px 0 0;min-height:4px;"></div>';
                        html += '<span style="font-size:0.48rem;color:var(--card-muted);margin-top:1px;">' + s.bucket + '</span>';
                        html += '</div>';
                    });
                    html += '</div></div>';
                }

                // Top violations
                if (d.top_violations && d.top_violations.length) {
                    html += '<div style="margin-bottom:10px;"><strong style="font-size:0.68rem;">⚠️ 常见违规 TOP5</strong>';
                    html += '<div style="font-size:0.62rem;margin-top:4px;">';
                    d.top_violations.forEach((v, i) => {
                        const sc = v.status === 'critical' ? '#e74c3c' : '#e67e22';
                        html += '<div style="display:flex;justify-content:space-between;padding:3px 4px;border-bottom:1px solid var(--card-border);">';
                        html += '<span>' + (i + 1) + '. ' + hlVerse(v.function) + '</span>';
                        html += '<span style="color:' + sc + ';">' + v.count + ' 次</span>';
                        html += '</div>';
                    });
                    html += '</div></div>';
                }

                // Recent runs
                if (d.recent_runs && d.recent_runs.length) {
                    html += '<div style="margin-bottom:10px;"><strong style="font-size:0.68rem;">📋 最近审查</strong>';
                    html += '<div style="max-height:180px;overflow-y:auto;font-size:0.62rem;margin-top:4px;">';
                    html += '<div style="display:flex;font-weight:600;padding:3px 4px;border-bottom:2px solid var(--card-border);">';
                    html += '<span style="flex:1;">项目/时间</span><span style="width:50px;text-align:center;">分数</span><span style="width:40px;text-align:center;">严重</span><span style="width:40px;text-align:center;">违规</span><span style="width:50px;text-align:center;">状态</span>';
                    html += '</div>';
                    d.recent_runs.forEach(r => {
                        const scColor = r.status === 'PASS' ? '#27ae60' : '#e74c3c';
                        const scoreColor = (r.score || 0) >= 80 ? '#27ae60' : (r.score || 0) >= 50 ? '#f39c12' : '#e74c3c';
                        html += '<div style="display:flex;padding:3px 4px;border-bottom:1px solid var(--card-border);">';
                        html += '<span style="flex:1;">' + hlVerse('#' + r.id + ' ' + (r.started_at || '')) + '</span>';
                        html += '<span style="width:50px;text-align:center;font-weight:600;color:' + scoreColor + ';">' + (r.score || '-') + '</span>';
                        html += '<span style="width:40px;text-align:center;color:#e74c3c;">' + (r.critical_count || 0) + '</span>';
                        html += '<span style="width:40px;text-align:center;color:#e67e22;">' + (r.violation_count || 0) + '</span>';
                        html += '<span style="width:50px;text-align:center;color:' + scColor + ';">' + (r.status === 'PASS' ? '✅' : '❌') + '</span>';
                        html += '</div>';
                    });
                    html += '</div></div>';
                }

                // Critical alerts
                if (d.critical_alerts && d.critical_alerts.length) {
                    html += '<div style="margin-bottom:8px;"><strong style="font-size:0.68rem;color:#e74c3c;">🚨 严重告警</strong>';
                    html += '<div style="font-size:0.62rem;margin-top:4px;">';
                    d.critical_alerts.forEach(a => {
                        html += '<div style="padding:4px 6px;margin-bottom:3px;background:#fdf0f0;border-left:3px solid #e74c3c;border-radius:2px;">';
                        html += '<span style="font-weight:600;">' + hlVerse(a.function) + '</span>';
                        html += ' <span style="color:var(--card-muted);">' + hlVerse(a.filename || '') + '</span>';
                        html += ' <span style="float:right;">score: ' + a.score + '</span>';
                        html += '</div>';
                    });
                    html += '</div></div>';
                }

                html += '</div>';
                panel.innerHTML = html;
            })
            .catch(e => {
                panel.innerHTML = '<span style="color:#e74c3c;font-size:0.7rem;">仪表盘加载失败: ' + e.message + '</span>';
            });
    }

    function _dashCard(label, value, color) {
        return '<div style="flex:1;min-width:80px;padding:8px;background:var(--bg-color);border-radius:6px;text-align:center;border-top:3px solid ' + color + ';">' +
            '<div style="font-size:1.1rem;font-weight:700;color:' + color + ';">' + value + '</div>' +
            '<div style="font-size:0.55rem;color:var(--card-muted);">' + label + '</div></div>';
    }

    // ── Multi-Project Comparison (U15) ──

    function showCompare() {
        const panel = document.getElementById('complianceComparePanel');
        if (!panel) return;

        if (panel.style.display === 'block') {
            panel.style.display = 'none';
            return;
        }
        panel.style.display = 'block';
        panel.innerHTML = '<span style="color:var(--card-muted);font-size:0.7rem;">⏳ 加载对比数据...</span>';

        fetch('/compliance/compare?days=90', { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                const d = data.data || data;
                const projects = d.projects || [];
                const functions = d.functions || [];
                const matrix = d.matrix || [];
                const funcSummaries = d.function_summaries || [];
                const projSummaries = d.project_summaries || [];

                if (!projects.length) {
                    panel.innerHTML = '<span style="color:var(--card-muted);font-size:0.7rem;">暂无项目数据用于对比</span>';
                    return;
                }

                let html = '<div style="padding:8px 0;">';

                // Project summary row
                html += '<div style="margin-bottom:10px;"><strong style="font-size:0.68rem;">📊 项目概览</strong>';
                html += '<div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap;">';
                projSummaries.forEach(p => {
                    const sc = p.avg_score >= 80 ? '#27ae60' : p.avg_score >= 50 ? '#f39c12' : '#e74c3c';
                    html += '<div style="padding:6px 10px;background:var(--bg-color);border-radius:4px;font-size:0.62rem;text-align:center;min-width:100px;">';
                    html += '<div style="font-weight:600;">项目 #' + p.project_id + '</div>';
                    html += '<div style="font-size:1rem;color:' + sc + ';font-weight:700;">' + p.avg_score + '</div>';
                    html += '<div style="color:var(--card-muted);">' + p.runs + ' 次审查 | 通过率 ' + p.pass_rate + '%</div>';
                    html += '</div>';
                });
                html += '</div></div>';

                // Heatmap
                if (functions.length <= 20) {
                    const maxWidth = Math.min(projects.length * 100, 500);
                    html += '<div style="margin-bottom:10px;overflow-x:auto;"><strong style="font-size:0.68rem;">🗺️ 项目×功能 合规热力图</strong>';
                    html += '<div style="font-size:0.55rem;margin-top:4px;">';
                    html += '<table style="border-collapse:collapse;width:auto;">';
                    html += '<thead><tr><th style="padding:3px 6px;position:sticky;left:0;background:var(--bg-color);"></th>';
                    functions.forEach(fn => {
                        html += '<th style="padding:3px 6px;writing-mode:vertical-lr;text-orientation:mixed;max-height:100px;font-weight:400;">' + hlVerse(fn) + '</th>';
                    });
                    html += '</tr></thead><tbody>';
                    matrix.forEach(row => {
                        html += '<tr>';
                        html += '<td style="padding:3px 6px;font-weight:600;position:sticky;left:0;background:var(--bg-color);"># ' + row.project_id + '</td>';
                        row.function_scores.forEach(fs => {
                            const score = fs.avg_score;
                            let bg, color;
                            if (score === null || score === undefined) {
                                bg = 'var(--card-border)';
                                color = 'var(--card-muted)';
                            } else if (score >= 80) {
                                bg = '#d4edda';
                                color = '#155724';
                            } else if (score >= 50) {
                                bg = '#fff3cd';
                                color = '#856404';
                            } else if (score >= 30) {
                                bg = '#f8d7da';
                                color = '#721c24';
                            } else {
                                bg = '#f5c6cb';
                                color = '#721c24';
                            }
                            html += '<td style="padding:3px 6px;text-align:center;background:' + bg + ';color:' + color + ';font-weight:600;min-width:36px;">' + (score !== null ? score : '-') + '</td>';
                        });
                        html += '</tr>';
                    });
                    html += '</tbody></table></div></div>';
                } else {
                    html += '<div style="color:var(--card-muted);font-size:0.62rem;">⚠️ 功能维度过多 (' + functions.length + ')，仅显示汇总。</div>';
                }

                // Most problematic functions
                if (funcSummaries.length) {
                    html += '<div style="margin-bottom:8px;"><strong style="font-size:0.68rem;">⚠️ 常见问题功能</strong>';
                    html += '<div style="font-size:0.62rem;margin-top:4px;">';
                    funcSummaries.slice(0, 10).forEach(f => {
                        const sc = f.avg_score >= 80 ? '#27ae60' : f.avg_score >= 50 ? '#f39c12' : '#e74c3c';
                        html += '<div style="display:flex;justify-content:space-between;padding:3px 4px;border-bottom:1px solid var(--card-border);">';
                        html += '<span>' + hlVerse(f.function) + '</span>';
                        html += '<span>均分 <b style="color:' + sc + ';">' + f.avg_score + '</b> | 问题率 ' + f.problem_rate + '% (' + f.problem_count + '/' + f.total_checks + ')</span>';
                        html += '</div>';
                    });
                    html += '</div></div>';
                }

                html += '</div>';
                panel.innerHTML = html;
            })
            .catch(e => {
                panel.innerHTML = '<span style="color:#e74c3c;font-size:0.7rem;">对比数据加载失败: ' + e.message + '</span>';
            });
    }

    // ── Knowledge Graph (U12) ──
    // (Moved to project detail sub-tab "🕸️ 图谱" — see app.js loadProjectInfo)

})();
