// ── Audit Case Library (U13) ──

window.Cases = {};

(function () {
    const C = window.Cases;
    let currentFilter = {};

    C.loadList = loadList;
    C.showDetail = showDetail;
    C.showStats = showStats;
    C.refresh = function () { loadList(currentFilter); };

    function initPanel() {
        const panel = document.getElementById('casesPanel');
        if (!panel) return;
        panel.innerHTML = `
        <div style="display:flex;height:100%;">
            <div id="casesSidebar" style="width:280px;border-right:1px solid var(--card-border);padding:12px;overflow-y:auto;flex-shrink:0;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <strong style="font-size:0.85rem;">📊 案例库</strong>
                    <button id="casesStatsBtn" style="font-size:0.62rem;padding:2px 8px;background:#2980b9;color:#fff;border:none;border-radius:3px;cursor:pointer;">📈 统计</button>
                </div>
                <div style="margin-bottom:8px;">
                    <input id="casesSearchInput" type="text" placeholder="搜索案例..." style="width:100%;font-size:0.7rem;padding:4px 8px;border:1px solid var(--card-border);border-radius:4px;background:var(--card-bg);">
                </div>
                <div id="casesFilterSeverity" style="display:flex;gap:4px;margin-bottom:8px;flex-wrap:wrap;">
                    <button class="cases-sev-btn" data-sev="" style="font-size:0.62rem;padding:2px 8px;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">全部</button>
                    <button class="cases-sev-btn" data-sev="violation" style="font-size:0.62rem;padding:2px 8px;border:1px solid #e67e22;color:#e67e22;border-radius:3px;background:var(--card-bg);cursor:pointer;">⚠️ 违规</button>
                    <button class="cases-sev-btn" data-sev="critical" style="font-size:0.62rem;padding:2px 8px;border:1px solid #e74c3c;color:#e74c3c;border-radius:3px;background:var(--card-bg);cursor:pointer;">🚫 严重</button>
                </div>
                <div style="display:flex;gap:4px;margin-bottom:8px;">
                    <button id="casesFilterOpen" style="font-size:0.62rem;padding:2px 8px;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">未解决</button>
                    <button id="casesFilterResolved" style="font-size:0.62rem;padding:2px 8px;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">已解决</button>
                </div>
                <div id="casesListContainer" style="font-size:0.7rem;"></div>
                <div id="casesPagination" style="margin-top:8px;font-size:0.65rem;text-align:center;"></div>
            </div>
            <div id="casesDetail" style="flex:1;padding:16px;overflow-y:auto;"></div>
        </div>`;
        panel.style.display = 'block';

        document.getElementById('casesStatsBtn').onclick = showStats;
        document.getElementById('casesSearchInput').addEventListener('keydown', function (e) {
            if (e.key === 'Enter') { currentFilter.search = this.value; loadList(currentFilter); }
        });

        document.querySelectorAll('.cases-sev-btn').forEach(btn => {
            btn.onclick = function () {
                document.querySelectorAll('.cases-sev-btn').forEach(b => b.style.fontWeight = '');
                this.style.fontWeight = '600';
                currentFilter.severity = this.dataset.sev || undefined;
                loadList(currentFilter);
            };
        });

        document.getElementById('casesFilterOpen').onclick = () => {
            currentFilter.resolved = false; loadList(currentFilter);
        };
        document.getElementById('casesFilterResolved').onclick = () => {
            currentFilter.resolved = true; loadList(currentFilter);
        };

        loadList({});
    }

    function loadList(filter) {
        filter = filter || {};
        currentFilter = filter;
        const container = document.getElementById('casesListContainer');
        if (!container) return;
        container.innerHTML = '<span style="color:var(--card-muted);">⏳ 加载中...</span>';

        let url = '/cases?page=1&per_page=50';
        if (filter.severity) url += '&severity=' + filter.severity;
        if (filter.resolved !== undefined) url += '&resolved=' + filter.resolved;
        if (filter.search) url += '&search=' + encodeURIComponent(filter.search);

        fetch(url, { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                const items = (data.data || data).items || [];
                const total = (data.data || data).total || 0;
                const pages = (data.data || data).pages || 1;

                if (!items.length) {
                    container.innerHTML = '<div style="color:var(--card-muted);padding:12px;text-align:center;">暂无案例</div>';
                } else {
                    let html = '';
                    items.forEach(c => {
                        const sevColor = c.severity === 'critical' ? '#e74c3c' : c.severity === 'violation' ? '#e67e22' : '#f39c12';
                        const sevLabel = c.severity === 'critical' ? '🚫 严重' : c.severity === 'violation' ? '⚠️ 违规' : 'ℹ️ ' + c.severity;
                        html += '<div class="case-list-item" data-id="' + c.id + '" style="padding:6px 4px;border-bottom:1px solid var(--card-border);cursor:pointer;font-size:0.7rem;">';
                        html += '<div style="font-weight:500;">' + _h(c.title) + '</div>';
                        html += '<div style="display:flex;gap:6px;font-size:0.62rem;margin-top:2px;">';
                        html += '<span style="color:' + sevColor + ';">' + sevLabel + '</span>';
                        html += '<span style="color:var(--card-muted);">' + _h(c.category) + '</span>';
                        if (c.is_resolved) html += '<span style="color:#27ae60;">✅ 已解决</span>';
                        if (c.tags && c.tags.length) {
                            html += c.tags.slice(0, 3).map(t => '<span style="background:var(--card-border);padding:0 3px;border-radius:2px;margin:0 1px;">' + t + '</span>').join('');
                        }
                        html += '</div></div>';
                    });
                    container.innerHTML = html;
                    container.querySelectorAll('.case-list-item').forEach(el => {
                        el.onclick = function () { showDetail(parseInt(this.dataset.id)); };
                    });
                }

                const pag = document.getElementById('casesPagination');
                if (pages > 1) {
                    let ph = '';
                    for (let p = 1; p <= Math.min(pages, 10); p++) {
                        ph += '<span style="cursor:pointer;padding:2px 6px;margin:0 2px;border-radius:3px;border:1px solid var(--card-border);">' + p + '</span>';
                    }
                    pag.innerHTML = ph + '<span style="margin-left:6px;">共 ' + total + ' 个</span>';
                } else {
                    pag.innerHTML = total > 0 ? '<span>共 ' + total + ' 个</span>' : '';
                }
            })
            .catch(e => { container.innerHTML = '<span style="color:#e74c3c;">加载失败: ' + e.message + '</span>'; });
    }

    function showDetail(cid) {
        const detail = document.getElementById('casesDetail');
        detail.innerHTML = '<span style="color:var(--card-muted);">⏳ 加载中...</span>';

        fetch('/cases/' + cid, { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                const c = data.data || data;
                const sevColor = c.severity === 'critical' ? '#e74c3c' : c.severity === 'violation' ? '#e67e22' : '#f39c12';

                let html = '<div style="display:flex;justify-content:space-between;margin-bottom:12px;">';
                html += '<div><h3 style="margin:0;">' + _h(c.title) + '</h3>';
                html += '<div style="font-size:0.7rem;margin-top:4px;">';
                html += '<span style="color:' + sevColor + ';font-weight:600;">' + _h(c.severity) + '</span>';
                html += ' | ' + _h(c.category);
                if (c.project_id) html += ' | 项目 #' + c.project_id;
                if (c.created_at) html += ' | ' + c.created_at;
                html += '</div></div>';
                html += '<div style="display:flex;gap:6px;">';
                html += '<button class="case-resolve-btn" style="font-size:0.62rem;padding:3px 10px;border:1px solid #27ae60;color:#27ae60;border-radius:3px;cursor:pointer;">' + (c.is_resolved ? '🔄 重新打开' : '✅ 标记解决') + '</button>';
                html += '</div></div>';

                if (c.description) {
                    html += '<div style="margin-bottom:12px;padding:10px;background:var(--bg-color);border-radius:6px;font-size:0.75rem;white-space:pre-wrap;">' + _h(c.description) + '</div>';
                }
                if (c.resolution) {
                    html += '<div style="margin-bottom:12px;border-left:3px solid #2980b9;padding-left:10px;">';
                    html += '<div style="font-size:0.65rem;color:var(--card-muted);">建议解决方案</div>';
                    html += '<div style="font-size:0.72rem;">' + _h(c.resolution) + '</div></div>';
                }

                if (c.tags && c.tags.length) {
                    html += '<div style="margin-bottom:12px;">' + c.tags.map(t => '<span style="background:var(--card-border);padding:2px 8px;border-radius:3px;font-size:0.62rem;margin-right:4px;">' + _h(t) + '</span>').join('') + '</div>';
                }

                if (c.law_links && c.law_links.length) {
                    html += '<div style="margin-bottom:12px;"><strong style="font-size:0.7rem;">📜 关联法规</strong>';
                    c.law_links.forEach(l => {
                        html += '<div style="margin:4px 0;padding:6px;background:var(--bg-color);border-radius:4px;font-size:0.68rem;">';
                        html += '<span style="font-weight:500;">' + _h(l.article_label) + '</span>';
                        html += ' <span style="color:var(--card-muted);">' + _h(l.article_text) + '</span>';
                        html += '</div>';
                    });
                    html += '</div>';
                }

                if (c.template_links && c.template_links.length) {
                    html += '<div style="margin-bottom:12px;"><strong style="font-size:0.7rem;">📋 关联模板</strong>';
                    c.template_links.forEach(l => {
                        html += '<div style="margin:4px 0;padding:6px;background:var(--bg-color);border-radius:4px;font-size:0.68rem;">';
                        html += '<span style="font-weight:500;">' + _h(l.template_name) + '</span>';
                        if (l.section_id) html += ' <span style="color:var(--card-muted);">(章节: ' + _h(l.section_id) + ')</span>';
                        html += '</div>';
                    });
                    html += '</div>';
                }

                detail.innerHTML = html;
                detail.querySelector('.case-resolve-btn').onclick = () => {
                    fetch('/cases/' + cid, {
                        method: 'PUT', credentials: 'include',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ is_resolved: !c.is_resolved }),
                    })
                        .then(r => r.json())
                        .then(() => showDetail(cid));
                };
            })
            .catch(e => { detail.innerHTML = '<span style="color:#e74c3c;">加载失败: ' + e.message + '</span>'; });
    }

    function showStats() {
        const detail = document.getElementById('casesDetail');
        detail.innerHTML = '<span style="color:var(--card-muted);">⏳ 加载统计...</span>';

        fetch('/cases/stats', { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                const s = data.data || data;
                let html = '<h4>📈 案例库统计</h4>';
                html += '<div style="display:flex;gap:12px;margin:12px 0;flex-wrap:wrap;">';
                html += '<div style="flex:1;min-width:120px;padding:12px;background:var(--bg-color);border-radius:6px;text-align:center;"><div style="font-size:1.5rem;font-weight:700;">' + (s.open || 0) + '</div><div style="font-size:0.65rem;color:var(--card-muted);">未解决</div></div>';
                html += '<div style="flex:1;min-width:120px;padding:12px;background:var(--bg-color);border-radius:6px;text-align:center;"><div style="font-size:1.5rem;font-weight:700;">' + (s.resolved || 0) + '</div><div style="font-size:0.65rem;color:var(--card-muted);">已解决</div></div>';
                html += '<div style="flex:1;min-width:120px;padding:12px;background:var(--bg-color);border-radius:6px;text-align:center;"><div style="font-size:1.5rem;font-weight:700;">' + (s.total || 0) + '</div><div style="font-size:0.65rem;color:var(--card-muted);">总计</div></div>';
                html += '</div>';

                if (s.by_severity && Object.keys(s.by_severity).length) {
                    html += '<h5 style="margin:10px 0 4px;">按严重程度</h5><div style="display:flex;gap:8px;flex-wrap:wrap;">';
                    for (const [k, v] of Object.entries(s.by_severity)) {
                        const sc = k === 'critical' ? '#e74c3c' : k === 'violation' ? '#e67e22' : '#f39c12';
                        html += '<span style="padding:4px 10px;background:' + sc + '20;border:1px solid ' + sc + ';border-radius:4px;font-size:0.7rem;">' + k + ': <strong>' + v + '</strong></span>';
                    }
                    html += '</div>';
                }

                if (s.by_category && s.by_category.length) {
                    html += '<h5 style="margin:10px 0 4px;">按类别</h5><div style="font-size:0.68rem;">';
                    s.by_category.forEach(c => {
                        html += '<div style="padding:2px 0;">' + _h(c.category) + ': <strong>' + c.count + '</strong></div>';
                    });
                    html += '</div>';
                }

                if (s.top_template_issues && s.top_template_issues.length) {
                    html += '<h5 style="margin:10px 0 4px;">🔗 最常见的违规模板 TOP 10</h5>';
                    html += '<div style="font-size:0.68rem;">';
                    s.top_template_issues.forEach(t => {
                        html += '<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid var(--card-border);">';
                        html += '<span>' + _h(t.template_name) + '</span>';
                        html += '<span style="font-weight:600;">' + t.case_count + ' 案例</span>';
                        if (t.template_id) {
                            html += '<button class="case-goto-tpl" data-tid="' + t.template_id + '" style="font-size:0.55rem;border:1px solid var(--card-border);border-radius:3px;cursor:pointer;">编辑模板</button>';
                        }
                        html += '</div>';
                    });
                    html += '</div>';
                }

                detail.innerHTML = html;
                detail.querySelectorAll('.case-goto-tpl').forEach(btn => {
                    btn.onclick = () => {
                        const tid = parseInt(btn.dataset.tid);
                        const tplTab = document.getElementById('templatesTabBtn');
                        if (tplTab) tplTab.click();
                        setTimeout(() => { if (window.Templates && window.Templates.showDetail) window.Templates.showDetail(tid); }, 200);
                    };
                });
            })
            .catch(e => { detail.innerHTML = '<span style="color:#e74c3c;">统计加载失败: ' + e.message + '</span>'; });
    }

    function _h(s) { return (s || '').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initPanel);
    } else {
        initPanel();
    }

    const observer = new MutationObserver(function () {
        const panel = document.getElementById('casesPanel');
        if (panel && panel.style.display !== 'none' && !document.getElementById('casesSidebar')) {
            initPanel();
        }
    });
    const panel = document.getElementById('casesPanel');
    if (panel) observer.observe(panel, { attributes: true, attributeFilter: ['style'] });

})();
