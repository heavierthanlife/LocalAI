/* Wiki Tab module — extracted from app.js (W2 Phase 1a) */
/* Dependencies: showToast, escapeHtml, switchToPanel, switchSidebarPane,
                 saveActiveTab, wikiPanel, wikiTab, templatesTab, casesTab (kernel globals) */

    // ======================== Wiki Tab ========================
    if (wikiTab && wikiPanel) {
        (function() {
            var wc = document.getElementById('wikiContent');
            if (!wc) return;
            wc.addEventListener('click', function(e) {
                var editBtn = e.target.closest('.wiki-edit-btn');
                if (editBtn && editBtn.dataset.editPath) {
                    var container = editBtn.closest('#wikiContent > div') || editBtn.parentElement.parentElement;
                    if (!container) container = wc;
                    _startWikiEdit(container, null, editBtn.dataset.editPath);
                    return;
                }
                var wikiLink = e.target.closest('.wiki-page-link');
                if (wikiLink && wikiLink.dataset.path) {
                    _loadWiki(wikiLink.dataset.path);
                    return;
                }
                var delBtn = e.target.closest('.wiki-delete-btn');
                if (delBtn && delBtn.dataset.deletePath) {
                    if (!confirm('确定删除此页面？')) return;
                    fetch('/wiki/page/' + encodeURIComponent(delBtn.dataset.deletePath), { method: 'DELETE', credentials: 'include' })
                        .then(function(r) { return r.json(); }).then(function(r) { showToast(r.message, 'success'); var wt2 = document.getElementById('wikiTabBtn'); if (wt2) wt2.click(); }).catch(function() { showToast('删除失败', 'error'); });
                }
            });
        })();
        var _currentWikiPrefix = '';

        async function _loadWiki(prefix) {
            _currentWikiPrefix = prefix || '';
            var idxUrl = '/wiki/index';
            if (_currentWikiPrefix) idxUrl += '?prefix=' + encodeURIComponent(_currentWikiPrefix);
            var content = document.getElementById('wikiContent');
            try {
                var indexRes = await fetch(idxUrl, { credentials: 'include' });
                var statsRes = await fetch('/wiki/stats', { credentials: 'include' });
                var indexData = await indexRes.json();
                var statsData = await statsRes.json();
                // Log view for recent-views tracking
                try {
                    fetch('/wiki/view-log', { method: 'POST', credentials: 'include',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({action: 'view', wiki_path: _currentWikiPrefix || 'index',
                            article_title: (indexData && indexData.tree && indexData.tree.name) || 'Wiki'}) });
                } catch(_) {}
                var badge = document.getElementById('wikiStatsBadge');
                if (badge && statsData.success) {
                    var s = statsData.stats || {};
                    badge.textContent = (s.total_pages||0) + '页 · ' + (s.total_sources||0) + '来源';
                }
                loadSidebarWiki(indexData, statsData);
                var html = '';
                var tree = indexData && indexData.tree;
                var recent = indexData && indexData.recent;
                if (tree && tree.children && tree.children.length) {
                    var sections = [];
                    function _findLeafSections(nodes) {
                        if (!nodes) return;
                        nodes.forEach(function(n) {
                            if (n.type === 'dir') {
                                var hasDirectFiles = false;
                                if (n.children) {
                                    n.children.forEach(function(c) { if (c.type === 'file') hasDirectFiles = true; });
                                    if (hasDirectFiles) sections.push(n);
                                    else _findLeafSections(n.children);
                                }
                            }
                        });
                    }
                    _findLeafSections(tree.children);
                    sections.forEach(function(node) {
                        var count = 0;
                        function _cnt(n) { if (n.type === 'file') count++; if (n.children) n.children.forEach(_cnt); }
                        _cnt(node);
                        var sampleFiles = [];
                        function _collect(n) { if (n.type === 'file' && sampleFiles.length < 5) sampleFiles.push(n); if (n.children && sampleFiles.length < 5) n.children.forEach(_collect); }
                        _collect(node);
                        var icon = '📁';
                        var pp = (node.path||'').toLowerCase();
                        if (pp.startsWith('legal')) icon = '📜';
                        else if (pp.startsWith('projects')) icon = '📊';
                        else if (pp.startsWith('sources')) icon = '📄';
                        else if (pp.startsWith('entities')) icon = '🏢';
                        else if (pp.startsWith('concepts')) icon = '📖';
                        html += '<div style="margin-bottom:10px;border:1px solid var(--card-border);border-radius:8px;overflow:hidden;">';
                        html += '<div style="background:var(--card-bg);padding:8px 12px;font-size:0.8rem;font-weight:600;border-bottom:1px solid var(--card-border);display:flex;justify-content:space-between;align-items:center;">';
                        html += '<span>' + icon + ' ' + escapeHtml(node.name) + ' <span style="font-weight:400;font-size:0.72rem;color:var(--card-muted);">(' + count + '页)</span></span>';
                        html += '<span class="wiki-page-link" data-path="' + escapeHtml((node.path||'') + '/index') + '" style="font-size:0.7rem;color:var(--accent-color);cursor:pointer;">更多 →</span>';
                        html += '</div><div style="padding:4px 8px;">';
                        sampleFiles.forEach(function(f) {
                            var name = f.name.replace(/\.md$/i, '');
                            html += '<div class="wiki-page-link" data-path="' + escapeHtml(f.path) + '" style="padding:5px 8px;font-size:0.78rem;cursor:pointer;border-radius:3px;border:1px solid transparent;">📄 ' + escapeHtml(name) + '</div>';
                        });
                        if (count > 5) html += '<div class="wiki-page-link" data-path="' + escapeHtml((node.path||'') + '/index') + '" style="padding:3px 8px;font-size:0.72rem;cursor:pointer;border-radius:3px;border:1px solid transparent;color:var(--card-muted);">全部 ' + count + ' 个页面 →</div>';
                        html += '</div></div>';
                    });
                    tree.children.forEach(function(node) {
                        if (node.type === 'file') {
                            var name = node.name.replace(/\.md$/i, '');
                            html += '<div class="wiki-page-link" data-path="' + escapeHtml(node.path) + '" style="padding:5px 8px;font-size:0.78rem;cursor:pointer;border-radius:3px;border:1px solid transparent;margin-bottom:4px;">📝 ' + escapeHtml(name) + '</div>';
                        }
                    });
                    if (recent && recent.length) {
                        html += '<div style="margin-top:16px;border-top:1px solid var(--card-border);padding-top:10px;">';
                        html += '<div style="font-size:0.78rem;font-weight:600;margin-bottom:6px;">🕐 最近更新</div>';
                        recent.forEach(function(r) {
                            var name = (r.title||'').replace(/\.md$/i, '');
                            var mtimeStr = '';
                            try {
                                var d = new Date(r.mtime * 1000);
                                mtimeStr = d.getFullYear() + '-' + String(d.getMonth()+1).padStart(2,'0') + '-' + String(d.getDate()).padStart(2,'0') + ' ' + String(d.getHours()).padStart(2,'0') + ':' + String(d.getMinutes()).padStart(2,'0');
                            } catch(e) { mtimeStr = ''; }
                            html += '<div class="wiki-page-link" data-path="' + escapeHtml(r.path) + '" style="padding:4px 8px;font-size:0.78rem;cursor:pointer;border-radius:3px;border:1px solid transparent;display:flex;justify-content:space-between;">';
                            html += '<span>📄 ' + escapeHtml(name) + '</span>';
                            html += '<span style="color:var(--card-muted);font-size:0.68rem;">' + mtimeStr + '</span></div>';
                        });
                        html += '</div>';
                    }
                } else if (prefix) {
                    var _hasAnyFiles = false;
                    function _hasFileNodes(nodes) {
                        if (!nodes || _hasAnyFiles) return;
                        nodes.forEach(function(n) {
                            if (n.type === 'file') _hasAnyFiles = true;
                            if (n.children) _hasFileNodes(n.children);
                        });
                    }
                    if (tree) _hasFileNodes(tree.children || []);
                    if (_hasAnyFiles) {
                        html += '<h3 style="font-size:0.85rem;margin:0 0 8px;">📁 ' + escapeHtml(prefix) + '</h3>';
                        function _renderFiles(nodes, indent) {
                            if (!nodes) return;
                            nodes.forEach(function(n) {
                                if (n.type === 'file') {
                                    var name = n.name.replace(/\.md$/i, '');
                                    html += '<div class="wiki-page-link" data-path="' + escapeHtml(n.path) + '" style="padding:5px 8px;font-size:0.78rem;cursor:pointer;border-radius:3px;border:1px solid transparent;margin-bottom:4px;padding-left:' + (8 + (indent||0) * 16) + 'px;">📄 ' + escapeHtml(name) + '</div>';
                                } else if (n.type === 'dir') {
                                    html += '<div style="font-size:0.72rem;color:var(--card-muted);padding:3px 8px;margin-top:4px;">📁 ' + escapeHtml(n.name) + '</div>';
                                    _renderFiles(n.children, (indent||0) + 1);
                                }
                            });
                        }
                        _renderFiles(tree.children || [], 0);
                    } else {
                        try {
                            var pagePath = prefix.replace(/\.md$/i, '');
                            var pageRes = await fetch('/wiki/page/' + encodeURIComponent(pagePath), { credentials: 'include' });
                            var pageData = await pageRes.json();
                            if (pageData.success) {
                                var d = pageData;
                                html += '<div style="display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap;">';
                                html += '<button class="wiki-back-btn" style="background:#e2e8f0;border:none;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">← 返回Wiki首页</button>';
                                html += '<button class="wiki-compare-btn" data-path="' + escapeHtml(pagePath) + '" style="background:#dbeafe;border:none;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">🔗 对比</button>';
                                html += '<button class="wiki-graph-btn" data-path="' + escapeHtml(pagePath) + '" style="background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">🕸️ 引用图谱</button>';
                                if (sessionStorage.getItem('isAdmin') === 'true') {
                                    html += '<button class="wiki-edit-btn" data-edit-path="' + escapeHtml(pagePath) + '" style="background:#fef3c7;border:1px solid #f59e0b;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">✏️ 编辑</button>';
                                    html += '<button class="wiki-delete-btn" data-delete-path="' + escapeHtml(pagePath) + '" style="background:#fee2e2;border:1px solid #ef4444;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">🗑️ 删除</button>';
                                }
                                html += '</div>';
                                html += '<div style="background:var(--card-bg);border-radius:8px;padding:16px;">';
                                if (d.frontmatter?.tags) {
                                    html += '<div style="margin-bottom:8px;">' + d.frontmatter.tags.map(function(t) { return '<span style="background:#e2e8f0;padding:2px 8px;border-radius:4px;font-size:.65rem;margin-right:4px;">#' + escapeHtml(t) + '</span>'; }).join('') + '</div>';
                                }
                                if (d.frontmatter?.mentions && d.frontmatter.mentions.length) {
                                    html += '<div style="margin-bottom:8px;font-size:.68rem;color:var(--card-muted);">📌 被引用: ' + d.frontmatter.mentions.length + ' 次</div>';
                                }
                                html += d.html;
                                html += '</div>';
                                html += '<div id="wikiCitationGraphContainer" style="display:none;position:relative;width:100%;height:400px;margin-top:12px;border:1px solid #fecaca;border-radius:8px;overflow:hidden;background:#fef2f2;"></div>';
                            } else {
                                html += '<p style="color:var(--card-muted);">此分类「' + escapeHtml(prefix) + '」暂无内容。上传知识库文件后，系统将自动生成Wiki页面。</p>';
                            }
                        } catch(e) {
                            console.error('Wiki page load failed:', e);
                            html += '<p style="color:#ef4444;font-size:.78rem;">加载页面失败</p>';
                        }
                    }
                } else {
                    html += '<p style="color:var(--card-muted);">Wiki暂无内容。上传知识库文件后，系统将自动生成Wiki页面。</p>';
                }
                content.innerHTML = html;
                var backBtn = content.querySelector('.wiki-back-btn');
                if (backBtn) backBtn.onclick = function() { var wt = document.getElementById('wikiTabBtn'); if (wt) wt.click(); };
                var compareBtn = content.querySelector('.wiki-compare-btn');
                if (compareBtn) compareBtn.onclick = function() { _showCompareDialog(compareBtn.dataset.path); };
                var graphBtn = content.querySelector('.wiki-graph-btn');
                if (graphBtn) graphBtn.onclick = function() { _showCitationGraph(graphBtn.dataset.path); };
                content.querySelectorAll('.wiki-page-link').forEach(function(el) {
                    el.onmouseenter = function() { el.style.borderColor = 'var(--card-border)'; };
                    el.onmouseleave = function() { el.style.borderColor = 'transparent'; };
                    el.onclick = async function() {
                        try {
                            var path = el.dataset.path.replace(/\.md$/i, '');
                            var pageRes = await fetch('/wiki/page/' + encodeURIComponent(path), { credentials: 'include' });
                            var pageData = await pageRes.json();
                                if (pageData.success) {
                                    var d = pageData;
                                    var detailHtml = '<div style="display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap;">';
                                    detailHtml += '<button class="wiki-back-btn" style="background:#e2e8f0;border:none;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">← 返回Wiki首页</button>';
                                    detailHtml += '<button class="wiki-compare-btn" data-path="' + escapeHtml(path) + '" style="background:#dbeafe;border:none;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">🔗 对比</button>';
                                    detailHtml += '<button class="wiki-graph-btn" data-path="' + escapeHtml(path) + '" style="background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">🕸️ 引用图谱</button>';
                                    if (sessionStorage.getItem('isAdmin') === 'true') {
                                        detailHtml += '<button class="wiki-edit-btn" data-edit-path="' + escapeHtml(path) + '" style="background:#fef3c7;border:1px solid #f59e0b;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">✏️ 编辑</button>';
                                        detailHtml += '<button class="wiki-delete-btn" data-delete-path="' + escapeHtml(path) + '" style="background:#fee2e2;border:1px solid #ef4444;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">🗑️ 删除</button>';
                                    }
                                    detailHtml += '</div>';
                                    detailHtml += '<div id="wikiPageContent" style="background:var(--card-bg);border-radius:8px;padding:16px;">';
                                    if (d.frontmatter?.tags) {
                                        detailHtml += '<div style="margin-bottom:8px;">' + d.frontmatter.tags.map(function(t) { return '<span style="background:#e2e8f0;padding:2px 8px;border-radius:4px;font-size:.65rem;margin-right:4px;">#' + escapeHtml(t) + '</span>'; }).join('') + '</div>';
                                    }
                                    if (d.frontmatter?.mentions && d.frontmatter.mentions.length) {
                                        detailHtml += '<div style="margin-bottom:8px;font-size:.68rem;color:var(--card-muted);">📌 被引用: ' + d.frontmatter.mentions.length + ' 次</div>';
                                    }
                                    detailHtml += d.html;
                                    detailHtml += '</div>';
                                    detailHtml += '<div id="wikiCitationGraphContainer" style="display:none;position:relative;width:100%;height:400px;margin-top:12px;border:1px solid #fecaca;border-radius:8px;overflow:hidden;background:#fef2f2;"></div>';
                                    content.innerHTML = detailHtml;
                                    content.querySelector('.wiki-back-btn').onclick = function() { var wt = document.getElementById('wikiTabBtn'); if (wt) wt.click(); };
                                    var detailCompareBtn = content.querySelector('.wiki-compare-btn');
                                    if (detailCompareBtn) detailCompareBtn.onclick = function() { _showCompareDialog(detailCompareBtn.dataset.path); };
                                    var detailGraphBtn = content.querySelector('.wiki-graph-btn');
                                    if (detailGraphBtn) detailGraphBtn.onclick = function() { _showCitationGraph(detailGraphBtn.dataset.path); };
                            } else {
                                console.warn('Wiki page load fail:', path, pageData);
                            }
                        } catch(e) {
                            console.error('Wiki page click error:', e);
                        }
                    };
                });
            } catch(e) {
                console.error('Wiki load failed:', e);
                content.innerHTML = '<p style="color:#ef4444;font-size:.78rem;">加载Wiki失败</p>';
            }
        }

        templatesTab.onclick = async () => {
            stopRealtimePoll();
            saveActiveTab('templates');
            showSubTabBar('knowledge');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            templatesTab.classList.add('active');
            switchToPanel('templatesPanel');
            switchSidebarPane('templates');
        };

        const casesTabBtn = document.getElementById('casesTabBtn');
        if (casesTabBtn) casesTabBtn.onclick = async () => {
            saveActiveTab('cases');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            casesTabBtn.classList.add('active');
            switchToPanel('casesPanel');
            switchSidebarPane('cases');
        };

        wikiTab.onclick = async () => {
            stopRealtimePoll();
            saveActiveTab('wiki');
            showSubTabBar('knowledge');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            wikiTab.classList.add('active');
            switchToPanel('wikiPanel');
            switchSidebarPane('wiki');
            const content = document.getElementById('wikiContent');
            content.innerHTML = '<p style="color:var(--card-muted);">加载Wiki...</p>';
            const legalImportBtn = document.getElementById('wikiImportLegalBtn');
            const legalFileInput = document.getElementById('wikiLegalFileInput');
            if (sessionStorage.getItem('isAdmin') === 'true') {
                if (legalImportBtn) legalImportBtn.style.display = '';
                if (legalImportBtn) legalImportBtn.onclick = () => { legalFileInput.click(); };
                if (legalFileInput) legalFileInput.onchange = async () => {
                    if (!legalFileInput.files.length) return;
                    const fileCount = legalFileInput.files.length;
                    const wikiContentEl = document.getElementById('wikiContent');
                    wikiContentEl.innerHTML = '<p style="color:var(--card-muted);">⏳ 正在导入 ' + fileCount + ' 个文件，请稍候...</p>';
                    const formData = new FormData();
                    for (const f of legalFileInput.files) formData.append('files', f);
                    try {
                        const res = await fetch('/wiki/legal/import', { method: 'POST', credentials: 'include', body: formData });
                        const d = await res.json();
                        showToast(d.message, d.success ? 'success' : 'error', 4000);
                        legalFileInput.value = '';
                        wikiTab.click();
                    } catch(e) { showToast('导入失败', 'error'); wikiContentEl.innerHTML = '<p style="color:var(--card-muted);">导入失败，请重试。</p>'; }
                };
            } else {
                if (legalImportBtn) legalImportBtn.style.display = 'none';
            }
            _loadWiki('');
        };
        // Search handler
        const searchInput = document.getElementById('wikiSearchInput');
        const refreshBtn = document.getElementById('wikiRefreshBtn');
        if (searchInput) {
            let searchTimer;
            searchInput.addEventListener('input', () => {
                clearTimeout(searchTimer);
                searchTimer = setTimeout(async () => {
                    const q = searchInput.value.trim();
                    if (!q) { wikiTab.click(); return; }
                    const res = await fetch('/wiki/search?q=' + encodeURIComponent(q), { credentials: 'include' });
                    const data = await res.json();
                    const content = document.getElementById('wikiContent');
                    if (data.success && data.data?.results?.length) {
                        let html = '<div style="margin-bottom:8px;font-size:.72rem;color:var(--card-muted);">搜索 ' + escapeHtml(q) + ' 共 ' + data.data.results.length + ' 条结果</div>';
                        data.data.results.forEach(r => {
                            html += `<div style="padding:6px 10px;background:var(--card-bg);border-radius:4px;margin-bottom:3px;font-size:.78rem;"><strong>${escapeHtml(r.filename||r.path)}</strong><br><small style="color:var(--card-muted);">${escapeHtml(r.snippet||'')}</small></div>`;
                        });
                        content.innerHTML = html;
                    } else {
                        content.innerHTML = '<p style="color:var(--card-muted);font-size:.78rem;">无匹配结果</p>';
                    }
                }, 300);
            });
        }
        if (refreshBtn) refreshBtn.onclick = () => wikiTab.click();
        window._loadWiki = _loadWiki;
    }

    function _startWikiEdit(container, pageData, path) {
        fetch('/wiki/page/' + encodeURIComponent(path) + '/raw', { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) { showToast('无法加载编辑内容', 'error'); return; }
                const raw = data.data.content || '';
                const html = '<div style="display:flex;gap:6px;margin-bottom:12px;">' +
                    '<button class="wiki-back-btn" style="background:#e2e8f0;border:none;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">← 返回</button>' +
                    '<button id="wikiEditSaveBtn" style="background:#38a169;color:#fff;border:none;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">💾 保存</button>' +
                    '<button id="wikiEditCancelBtn" style="background:#e2e8f0;border:none;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">❌ 取消</button>' +
                    '</div>' +
                    '<textarea id="wikiEditArea" style="width:100%;min-height:60vh;font-size:.78rem;padding:12px;border:1px solid var(--card-border);border-radius:8px;background:var(--card-bg);color:var(--text-color);font-family:monospace;resize:vertical;">' +
                    escapeHtml(raw) + '</textarea>';
                container.innerHTML = html;
                container.querySelector('.wiki-back-btn').onclick = () => { wikiTab.click(); };
                container.querySelector('#wikiEditCancelBtn').onclick = () => { wikiTab.click(); };
                container.querySelector('#wikiEditSaveBtn').onclick = async () => {
                    const newContent = container.querySelector('#wikiEditArea').value;
                    try {
                        const res = await fetch('/wiki/page/' + encodeURIComponent(path), {
                            method: 'PUT', headers: {'Content-Type': 'application/json'},
                            credentials: 'include',
                            body: JSON.stringify({content: newContent})
                        });
                        const d = await res.json();
                        showToast(d.message, d.success ? 'success' : 'error');
                        if (d.success) wikiTab.click();
                    } catch(e) { showToast('保存失败', 'error'); }
                };
            }).catch(() => showToast('加载失败', 'error'));
    }

    // ── Wiki sidebar: page tree navigation ──
    function loadSidebarWiki(indexData, statsData) {
        const statsEl = document.getElementById('sidebarWikiStats');
        const treeEl = document.getElementById('sidebarWikiTree');
        if (!statsEl || !treeEl) return;
        if (statsData?.success) {
            const s = statsData.stats || {};
            statsEl.textContent = `${s.total_pages||0}页 · ${s.total_sources||0}来源` + (s.orphan_count ? ` · \ud83d\udd78\ufe0f${s.orphan_count}孤立` : '');
        }
        // ── Bookmarks ──
        var bmEl = document.getElementById('sidebarWikiBookmarks');
        if (bmEl) {
            fetch('/wiki/bookmarks', { credentials: 'include' })
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    if (!d.success) return;
                    var bms = d.bookmarks || [];
                    if (!bms.length) { bmEl.innerHTML = ''; return; }
                    var html = '<div style="font-size:0.6rem;color:var(--card-muted);margin-bottom:2px;">📌 收藏</div>';
                    bms.forEach(function(b) {
                        html += '<div class="wiki-bm-item" data-path="' + (b.wiki_path || b.article_id) + '" style="padding:2px 4px;font-size:0.62rem;cursor:pointer;display:flex;justify-content:space-between;border-radius:3px;margin-bottom:1px;">' +
                            '<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">⭐ ' + escapeHtml(b.article_title) + '</span>' +
                            '<span class="wiki-bm-remove" data-bid="' + b.id + '" style="font-size:0.5rem;color:#e74c3c;cursor:pointer;flex-shrink:0;margin-left:4px;" title="取消收藏">✕</span></div>';
                    });
                    bmEl.innerHTML = html;
                    bmEl.querySelectorAll('.wiki-bm-item').forEach(function(el) {
                        el.onclick = function(e) {
                            if (e.target.classList.contains('wiki-bm-remove')) return;
                            var fn = (typeof _loadWiki === 'function') ? _loadWiki : window._loadWiki;
                            if (fn) fn(el.dataset.path);
                        };
                    });
                    bmEl.querySelectorAll('.wiki-bm-remove').forEach(function(btn) {
                        btn.onclick = function(e) {
                            e.stopPropagation();
                            fetch('/wiki/bookmarks/' + btn.dataset.bid, { method: 'DELETE', credentials: 'include' })
                                .then(function() { if (typeof loadSidebarWiki==='function') loadSidebarWiki(); });
                        };
                    });
                })
                .catch(function() {});
        }
        // ── Recent views ──
        var rvEl = document.getElementById('sidebarWikiRecentViews');
        if (rvEl) {
            fetch('/wiki/view-log', { method: 'POST', credentials: 'include',
                headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: 'list', limit: 5}) })
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    var views = d.recent_views || [];
                    if (!views.length) { rvEl.innerHTML = ''; return; }
                    var html = '<div style="font-size:0.6rem;color:var(--card-muted);margin-bottom:2px;">🕐 最近查看</div>';
                    views.forEach(function(v) {
                        var t = v.article_title || v.wiki_path;
                        if (t.length > 22) t = t.slice(0, 20) + '...';
                        html += '<div class="wiki-rv-item" data-path="' + v.wiki_path + '" style="padding:2px 4px;font-size:0.62rem;cursor:pointer;margin-bottom:1px;color:var(--card-muted);">' +
                            escapeHtml(t) + '</div>';
                    });
                    rvEl.innerHTML = html;
                    rvEl.querySelectorAll('.wiki-rv-item').forEach(function(el) {
                        el.onclick = function() {
                            var fn = (typeof _loadWiki === 'function') ? _loadWiki : window._loadWiki;
                            if (fn) fn(el.dataset.path);
                        };
                    });
                })
                .catch(function() {});
        }
        var tabsEl = document.getElementById('sidebarWikiTabs');
        if (tabsEl) {
            var WIKI_TABS = [
                { prefix: '', label: '全部', icon: '📖' },
                { prefix: 'legal', label: '法律', icon: '📜' },
                { prefix: 'projects', label: '项目', icon: '📊' },
                { prefix: 'audit', label: '审计', icon: '📋' },
                { prefix: 'sources', label: '来源', icon: '📄' },
                { prefix: 'entities', label: '实体', icon: '🏢' },
                { prefix: 'concepts', label: '概念', icon: '📖' },
                { prefix: 'regulations', label: '法规', icon: '📋' },
                { prefix: 'templates', label: '模板', icon: '📄' },
                { prefix: 'experts', label: '专家', icon: '💡' },
                { prefix: 'comparisons', label: '对比', icon: '🔗' }
            ];
            tabsEl.innerHTML = '';
            WIKI_TABS.forEach(function(t) {
                var tabBtn = document.createElement('span');
                tabBtn.textContent = t.icon + ' ' + t.label;
                tabBtn.style.cssText = 'display:inline-block;padding:2px 5px;font-size:0.62rem;border-radius:3px;cursor:pointer;white-space:nowrap;border:1px solid var(--card-border);';
                if (t.prefix === _currentWikiPrefix) {
                    tabBtn.style.background = 'var(--accent-color)';
                    tabBtn.style.color = 'white';
                }
                tabBtn.onmouseenter = function() {
                    if (t.prefix !== _currentWikiPrefix) tabBtn.style.background = 'var(--card-border)';
                };
                tabBtn.onmouseleave = function() {
                    if (t.prefix !== _currentWikiPrefix) tabBtn.style.background = '';
                };
                tabBtn.onclick = function() { _loadWiki(t.prefix); };
                tabsEl.appendChild(tabBtn);
            });
        }
        const tree = indexData?.tree;
        if (!tree || !tree.children?.length) {
            treeEl.innerHTML = '<p style="color:var(--card-muted);font-size:0.72rem;">暂无Wiki页面</p>';
            return;
        }
        function renderTree(node, depth) {
            if (node.type === 'file') {
                const displayName = node.name.replace(/\.md$/i, '');
                const div = document.createElement('div');
                div.style.cssText = `padding:3px 6px 3px ${12 + depth * 16}px;cursor:pointer;border-radius:3px;font-size:0.76rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`;
                div.className = 'wiki-sidebar-page';
                div.dataset.path = node.path;
                div.textContent = '📄 ' + displayName;
                div.onmouseenter = () => div.style.background = 'var(--card-border)';
                div.onmouseleave = () => div.style.background = '';
                div.onclick = async () => {
                    const content = document.getElementById('wikiContent');
                    if (!content) return;
                    content.innerHTML = '<p style="color:var(--card-muted);">加载中...</p>';
                    try {
                        const res = await fetch('/wiki/page/' + encodeURIComponent(node.path.replace(/\.md$/i, '')), { credentials: 'include' });
                        const data = await res.json();
                        if (data.success) {
                            const d = data;
                            const cleanPath = node.path.replace(/\.md$/i, '');
                            let html = '<div style="display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap;">';
                            html += '<button class="wiki-back-btn" style="background:#e2e8f0;border:none;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">← 返回Wiki首页</button>';
                            html += '<button class="wiki-compare-btn" data-path="' + escapeHtml(cleanPath) + '" style="background:#dbeafe;border:none;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">🔗 对比</button>';
                            if (sessionStorage.getItem('isAdmin') === 'true') {
                                html += '<button class="wiki-edit-btn" data-edit-path="' + escapeHtml(cleanPath) + '" style="background:#fef3c7;border:1px solid #f59e0b;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">✏️ 编辑</button>';
                                html += '<button class="wiki-delete-btn" data-delete-path="' + escapeHtml(cleanPath) + '" style="background:#fee2e2;border:1px solid #ef4444;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">🗑️ 删除</button>';
                            }
                            html += '</div>';
                            html += '<div style="background:var(--card-bg);border-radius:8px;padding:16px;">';
                            if (d.frontmatter?.tags) {
                                html += '<div style="margin-bottom:8px;">' + d.frontmatter.tags.map(t => `<span style="background:#e2e8f0;padding:2px 8px;border-radius:4px;font-size:.65rem;margin-right:4px;">#${escapeHtml(t)}</span>`).join('') + '</div>';
                            }
                            if (d.frontmatter?.mentions && d.frontmatter.mentions.length) {
                                html += '<div style="margin-bottom:8px;font-size:.68rem;color:var(--card-muted);">📌 被引用: ' + d.frontmatter.mentions.length + ' 次</div>';
                            }
                            html += d.html;
                            html += '</div>';
                            content.innerHTML = html;
                            content.querySelector('.wiki-back-btn').onclick = () => { const wt = document.getElementById('wikiTabBtn'); if (wt) wt.click(); };
                            const compBtn = content.querySelector('.wiki-compare-btn');
                            if (compBtn) compBtn.onclick = function() { _showCompareDialog(compBtn.dataset.path); };
                            content.querySelectorAll('.wiki-page-link').forEach(function(el) {
                                el.onmouseenter = function() { el.style.borderColor = 'var(--card-border)'; };
                                el.onmouseleave = function() { el.style.borderColor = 'transparent'; };
                                el.onclick = async function() {
                                    try {
                                        var path = el.dataset.path.replace(/\.md$/i, '');
                                        var pageRes = await fetch('/wiki/page/' + encodeURIComponent(path), { credentials: 'include' });
                                        var pageData = await pageRes.json();
                                        if (pageData.success) {
                                            var d = pageData;
                                            var detailHtml = '<div style="display:flex;gap:6px;margin-bottom:12px;">';
                                            detailHtml += '<button class="wiki-back-btn" style="background:#e2e8f0;border:none;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">← 返回Wiki首页</button>';
                                            if (sessionStorage.getItem('isAdmin') === 'true') {
                                                detailHtml += '<button class="wiki-edit-btn" data-edit-path="' + escapeHtml(path) + '" style="background:#fef3c7;border:1px solid #f59e0b;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">✏️ 编辑</button>';
                                                detailHtml += '<button class="wiki-delete-btn" data-delete-path="' + escapeHtml(path) + '" style="background:#fee2e2;border:1px solid #ef4444;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">🗑️ 删除</button>';
                                            }
                                            detailHtml += '</div>';
                                            detailHtml += '<div id="wikiPageContent" style="background:var(--card-bg);border-radius:8px;padding:16px;">';
                                            if (d.frontmatter?.tags) {
                                                detailHtml += '<div style="margin-bottom:8px;">' + d.frontmatter.tags.map(function(t) { return '<span style="background:#e2e8f0;padding:2px 8px;border-radius:4px;font-size:.65rem;margin-right:4px;">#' + escapeHtml(t) + '</span>'; }).join('') + '</div>';
                                            }
                                            detailHtml += d.html;
                                            detailHtml += '</div>';
                                            content.innerHTML = detailHtml;
                                            content.querySelector('.wiki-back-btn').onclick = function() { var wt = document.getElementById('wikiTabBtn'); if (wt) wt.click(); };
                                        } else {
                                            console.warn('Wiki page load fail:', path, pageData);
                                        }
                                    } catch(e) {
                                        console.error('Wiki page click error:', e);
                                    }
                                };
                            });
                        } else {
                            content.innerHTML = '<p style="color:#ef4444;">加载失败</p>';
                        }
                    } catch(e) {
                        content.innerHTML = '<p style="color:#ef4444;">加载失败</p>';
                    }
                };
                return div;
            }
            // Directory node
            const wrapper = document.createElement('div');
            const header = document.createElement('div');
            header.style.cssText = `padding:3px 6px 3px ${12 + depth * 16}px;cursor:pointer;border-radius:3px;font-size:0.76rem;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`;
            header.className = 'wiki-sidebar-folder';
            let folderIcon = '📁';
            const p = node.path || '';
            if (p.startsWith('sources')) folderIcon = '📄';
            else if (p.startsWith('entities')) folderIcon = '🏢';
            else if (p.startsWith('concepts')) folderIcon = '📖';
            else if (p.startsWith('regulations')) folderIcon = '📋';
            else if (p.startsWith('templates')) folderIcon = '📄';
            else if (p.startsWith('projects')) folderIcon = '📊';
            else if (p.startsWith('experts')) folderIcon = '💡';
            else if (p.startsWith('comparisons')) folderIcon = '🔗';
            header.textContent = folderIcon + ' ' + (node.name || 'Wiki');
            header.onmouseenter = () => header.style.background = 'var(--card-border)';
            header.onmouseleave = () => header.style.background = '';
            header.onclick = () => {
                childrenDiv.style.display = childrenDiv.style.display === 'none' ? '' : 'none';
                header.textContent = (childrenDiv.style.display === 'none' ? '📁 ' : '📂 ') + (node.name || 'Wiki');
            };
            wrapper.appendChild(header);
            const childrenDiv = document.createElement('div');
            childrenDiv.style.display = '';
            (node.children || []).forEach(child => {
                const childEl = renderTree(child, depth + 1);
                if (childEl) childrenDiv.appendChild(childEl);
            });
            wrapper.appendChild(childrenDiv);
            return wrapper;
        }
        treeEl.innerHTML = '';
        (tree.children || []).forEach(child => {
            const el = renderTree(child, 0);
            if (el) treeEl.appendChild(el);
        });
    }

    // ── Compare dialog ──
    function _showCompareDialog(pagePath) {
        var content = document.getElementById('wikiContent');
        if (!content) return;
        var html = '<div style="padding:8px;">';
        html += '<div style="display:flex;gap:6px;margin-bottom:12px;">';
        html += '<button class="wiki-back-btn" style="background:#e2e8f0;border:none;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">← 返回</button>';
        html += '<span style="font-size:.8rem;color:var(--card-muted);">选择要对比的页面:</span>';
        html += '</div>';
        html += '<div id="comparePageList" style="margin-bottom:12px;"><p style="color:var(--card-muted);">加载可用页面...</p></div>';
        html += '<div id="compareResult" style="display:none;"></div>';
        html += '</div>';
        content.innerHTML = html;
        content.querySelector('.wiki-back-btn').onclick = function() { var wt = document.getElementById('wikiTabBtn'); if (wt) wt.click(); };

        // Load available pages for comparison (same category as current page)
        var catPart = pagePath.split('/')[0] || '';
        fetch('/wiki/index' + (catPart ? '?prefix=' + encodeURIComponent(catPart) : ''), { credentials: 'include' })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                var listEl = document.getElementById('comparePageList');
                if (!listEl) return;
                var tree = data.tree || {};
                var pages = [];
                function _walk(node) {
                    if (node.type === 'file' && node.path !== pagePath && node.path !== pagePath + '.md') {
                        pages.push({name: node.name.replace(/\.md$/,''), path: node.path.replace(/\.md$/,'')});
                    }
                    (node.children||[]).forEach(_walk);
                }
                _walk(tree);
                var listHtml = '<div style="max-height:300px;overflow-y:auto;">';
                pages.forEach(function(p) {
                    listHtml += '<div class="wiki-compare-target" data-path="' + escapeHtml(p.path) + '" style="padding:5px 10px;cursor:pointer;font-size:.78rem;border-radius:4px;margin-bottom:3px;border:1px solid transparent;">📄 ' + escapeHtml(p.name) + '</div>';
                });
                listHtml += '</div>';
                if (!pages.length) listHtml = '<p style="color:var(--card-muted);">同分类下无其他页面可对比</p>';
                listEl.innerHTML = listHtml;
                listEl.querySelectorAll('.wiki-compare-target').forEach(function(el) {
                    el.onmouseenter = function() { el.style.borderColor = 'var(--accent-color)'; el.style.background = '#eff6ff'; };
                    el.onmouseleave = function() { el.style.borderColor = 'transparent'; el.style.background = ''; };
                    el.onclick = async function() {
                        var targetPath = el.dataset.path;
                        var resultEl = document.getElementById('compareResult');
                        resultEl.style.display = 'block';
                        resultEl.innerHTML = '<p style="color:var(--card-muted);">正在对比「' + escapeHtml(pagePath) + '」和「' + escapeHtml(targetPath) + '」...</p>';
                        try {
                            var res = await fetch('/wiki/compare?page_a=' + encodeURIComponent(pagePath) + '&page_b=' + encodeURIComponent(targetPath), { credentials: 'include' });
                            var d = await res.json();
                            if (d.success) {
                                var r = d;
                                resultEl.innerHTML = '<div style="border:1px solid var(--card-border);border-radius:8px;padding:12px;">' +
                                    '<div style="font-weight:600;margin-bottom:8px;">对比结果</div>' +
                                    '<div style="margin-bottom:8px;"><strong>相似度:</strong> ' + (r.similarity*100).toFixed(1) + '%</div>' +
                                    '<div style="margin-bottom:4px;"><strong>共同实体:</strong> ' + (r.entities?.shared?.length||0) + ' 个</div>' +
                                    (r.entities?.shared?.length ? '<div style="font-size:.78rem;margin-bottom:8px;">' + r.entities.shared.map(function(e) { return '<span style="background:#e2e8f0;padding:1px 6px;border-radius:3px;margin-right:4px;font-size:.68rem;">' + escapeHtml(e) + '</span>'; }).join('') + '</div>' : '') +
                                    '<div style="margin-bottom:4px;"><strong>A独有:</strong> ' + (r.entities?.a_only?.length||0) + ' 个</div>' +
                                    '<div style="margin-bottom:4px;"><strong>B独有:</strong> ' + (r.entities?.b_only?.length||0) + ' 个</div>' +
                                    '<div style="margin-top:8px;font-size:.68rem;color:var(--card-muted);">对比结果已保存至 Wiki</div>' +
                                    '</div>';
                            } else {
                                resultEl.innerHTML = '<p style="color:#ef4444;">对比失败: ' + escapeHtml(d.error||'') + '</p>';
                            }
                        } catch(e) { resultEl.innerHTML = '<p style="color:#ef4444;">对比请求失败</p>'; }
                    };
                });
            });
    }

    window._showCompareDialog = _showCompareDialog;

// ── Citation graph toggle ──
    function _showCitationGraph(pagePath) {
        var container = document.getElementById('wikiCitationGraphContainer');
        if (!container) return;
        if (container.style.display === 'block') {
            container.style.display = 'none';
            return;
        }
        container.style.display = 'block';
        container.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:0.7rem;">加载中...</span>';
        fetch('/api/graph/citation?path=' + encodeURIComponent(pagePath) + '&depth=2', { credentials: 'include' })
            .then(function (r) { return r.json(); })
            .then(function (gData) {
                var graphData = gData.data || gData;
                if (graphData.success !== false && graphData.nodes && graphData.nodes.length) {
                    if (graphData.success) {
                        renderGraph(container, graphData);
                    } else {
                        renderGraph(container, graphData);
                    }
                } else {
                    container.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:0.7rem;">暂无引用关系数据</span>';
                }
            })
            .catch(function () {
                container.innerHTML = '<span style="color:#ef4444;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:0.7rem;">加载失败</span>';
            });
    }

// ── Bootstrap: if wiki panel is currently visible (initial tab restore),
// trigger load now that module code has registered the handlers.
(function() {
    var panel = document.getElementById('wikiPanel');
    if (panel && panel.style.display !== 'none' && typeof _loadWiki === 'function') {
        _loadWiki('');
    }
})();

