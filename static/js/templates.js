// ── Bid Templates Module (U5) ──
// Bid template CRUD: list, create, edit, import .docx, version history, diff

window.Templates = {};

(function () {
    const T = window.Templates;

    let currentCategory = '';
    let currentPage = 1;

    // ── Public API ──
    T.loadList = loadList;
    T.showDetail = showDetail;
    T.showCreate = showCreate;
    T.showEdit = showEdit;
    T.showImport = showImport;
    T.showVersions = showVersions;
    T.showDiff = showDiff;
    T.refresh = function () { loadList(currentCategory, currentPage); };

    // ── HTML Template ──
    function _panelHtml() {
        return `
        <div style="display:flex;height:100%;">
            <div id="templatesSidebar" style="width:260px;border-right:1px solid var(--card-border);padding:12px;overflow-y:auto;flex-shrink:0;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <strong style="font-size:0.85rem;">📋 模板库</strong>
                    <div style="display:flex;gap:4px;">
                        <button id="tplCreateBtn" style="font-size:0.62rem;padding:2px 8px;background:#27ae60;color:#fff;border:none;border-radius:3px;cursor:pointer;">+ 新建</button>
                        <button id="tplImportBtn" style="font-size:0.62rem;padding:2px 8px;background:#2980b9;color:#fff;border:none;border-radius:3px;cursor:pointer;">📥 导入</button>
                        <button id="tplRecommendBtn" style="font-size:0.62rem;padding:2px 8px;background:#8e44ad;color:#fff;border:none;border-radius:3px;cursor:pointer;">🤖 推荐</button>
                    </div>
                </div>
                <div style="margin-bottom:8px;">
                    <input id="tplSearchInput" type="text" placeholder="搜索模板..." style="width:100%;font-size:0.7rem;padding:4px 8px;border:1px solid var(--card-border);border-radius:4px;background:var(--card-bg);">
                </div>
                <div id="tplCategoryFilter" style="display:flex;gap:4px;margin-bottom:8px;flex-wrap:wrap;">
                    <button class="tpl-cat-btn" data-cat="" style="font-size:0.62rem;padding:2px 8px;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">全部</button>
                    <button class="tpl-cat-btn" data-cat="工程" style="font-size:0.62rem;padding:2px 8px;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">🏗️ 工程</button>
                    <button class="tpl-cat-btn" data-cat="货物" style="font-size:0.62rem;padding:2px 8px;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">📦 货物</button>
                    <button class="tpl-cat-btn" data-cat="服务" style="font-size:0.62rem;padding:2px 8px;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">🛠️ 服务</button>
                </div>
                <div id="tplListContainer" style="font-size:0.7rem;"></div>
                <div id="tplPagination" style="margin-top:8px;font-size:0.65rem;text-align:center;"></div>
            </div>
            <div id="templatesDetail" style="flex:1;padding:16px;overflow-y:auto;"></div>
        </div>`;
    }

    // ── Panel Setup ──
    function initPanel() {
        const panel = document.getElementById('templatesPanel');
        if (!panel) return;
        panel.innerHTML = _panelHtml();
        panel.style.display = 'block';

        document.getElementById('tplCreateBtn').onclick = showCreate;
        document.getElementById('tplImportBtn').onclick = showImport;
        document.getElementById('tplRecommendBtn').onclick = showRecommend;

        document.getElementById('tplSearchInput').addEventListener('keydown', function (e) {
            if (e.key === 'Enter') loadList(currentCategory, 1, this.value);
        });

        document.querySelectorAll('.tpl-cat-btn').forEach(btn => {
            btn.onclick = function () {
                currentCategory = this.dataset.cat;
                currentPage = 1;
                document.querySelectorAll('.tpl-cat-btn').forEach(b => b.style.fontWeight = '');
                this.style.fontWeight = '600';
                document.getElementById('tplSearchInput').value = '';
                loadList(currentCategory, currentPage);
            };
        });

        loadList('', 1);
    }

    // ── Load List ──
    function loadList(category, page, search) {
        category = category || '';
        page = page || 1;
        const container = document.getElementById('tplListContainer');
        if (!container) return;
        container.innerHTML = '<span style="color:var(--card-muted);">' + _icon('hourglass_empty') + ' 加载中...</span>';

        let url = '/templates?page=' + page + '&per_page=30';
        if (category) url += '&category=' + encodeURIComponent(category);
        if (search) url += '&search=' + encodeURIComponent(search);

        fetch(url, { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                const items = (data.data || data).items || [];
                const total = (data.data || data).total || 0;
                const pages = (data.data || data).pages || 1;

                if (!items.length) {
                    container.innerHTML = '<div style="color:var(--card-muted);padding:12px;text-align:center;">暂无模板<br><button id="tplCreateEmpty" style="margin-top:8px;font-size:0.7rem;padding:4px 12px;background:#27ae60;color:#fff;border:none;border-radius:3px;cursor:pointer;">创建第一个模板</button></div>';
                    const btn = document.getElementById('tplCreateEmpty');
                    if (btn) btn.onclick = showCreate;
                } else {
                    let html = '';
                    items.forEach(t => {
                        const verBadge = t.version ? ' <span style="font-size:0.58rem;color:var(--card-muted);">v' + t.version + '</span>' : '';
                        html += '<div class="tpl-list-item" data-id="' + t.id + '" style="padding:6px 4px;border-bottom:1px solid var(--card-border);cursor:pointer;font-size:0.7rem;">';
                        html += '<div style="font-weight:500;">' + _h(t.name) + verBadge + '</div>';
                        html += '<div style="color:var(--card-muted);font-size:0.62rem;">' + _h(t.category || '') + ' | ' + (t.version_count || 0) + ' 个版本</div>';
                        if (t.tags && t.tags.length) {
                            html += '<div style="margin-top:2px;">' + t.tags.map(tg => '<span style="background:var(--card-border);padding:0 3px;border-radius:2px;font-size:0.58rem;margin-right:2px;">' + _h(tg) + '</span>').join('') + '</div>';
                        }
                        html += '</div>';
                    });
                    container.innerHTML = html;

                    container.querySelectorAll('.tpl-list-item').forEach(el => {
                        el.onclick = function () { showDetail(parseInt(this.dataset.id)); };
                    });
                }

                const pag = document.getElementById('tplPagination');
                if (pages > 1) {
                    let phtml = '';
                    for (let p = 1; p <= pages; p++) {
                        phtml += '<span class="tpl-page-btn" data-page="' + p + '" style="cursor:pointer;padding:2px 6px;margin:0 2px;border-radius:3px;' + (p === page ? 'background:#2980b9;color:#fff;' : 'border:1px solid var(--card-border);') + '">' + p + '</span>';
                    }
                    pag.innerHTML = phtml + '<span style="margin-left:6px;">共 ' + total + ' 个</span>';
                    pag.querySelectorAll('.tpl-page-btn').forEach(b => {
                        b.onclick = function () {
                            currentPage = parseInt(this.dataset.page);
                            loadList(category, currentPage, search);
                        };
                    });
                } else {
                    pag.innerHTML = total > 0 ? '<span>共 ' + total + ' 个</span>' : '';
                }
            })
            .catch(e => {
                container.innerHTML = '<span style="color:#e74c3c;">加载失败: ' + e.message + '</span>';
            });
    }

    // ── Detail View ──
    function showDetail(tid) {
        const detail = document.getElementById('templatesDetail');
        detail.innerHTML = '<span style="color:var(--card-muted);">' + _icon('hourglass_empty') + ' 加载中...</span>';

        fetch('/templates/' + tid, { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                const t = data.data || data;
                let html = '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">';
                html += '<div><h3 style="margin:0;">' + _h(t.name) + '</h3>';
                html += '<span style="font-size:0.7rem;color:var(--card-muted);">' + _h(t.category) + ' | v' + t.version + ' | ' + (t.sections || []).length + ' 个章节</span></div>';
                html += '<div style="display:flex;gap:6px;">';
                html += '<button class="tpl-edit-btn" style="font-size:0.62rem;padding:3px 10px;border:1px solid var(--card-border);border-radius:3px;cursor:pointer;">✏️ 编辑</button>';
                html += '<button class="tpl-versions-btn" style="font-size:0.62rem;padding:3px 10px;border:1px solid var(--card-border);border-radius:3px;cursor:pointer;">📜 版本</button>';
                html += '<button class="tpl-delete-btn" style="font-size:0.62rem;padding:3px 10px;border:1px solid #e74c3c;color:#e74c3c;border-radius:3px;cursor:pointer;">🗑️</button>';
                html += '</div></div>';

                if (t.description) {
                    html += '<div style="margin-bottom:12px;font-size:0.75rem;color:var(--card-muted);">' + _h(t.description) + '</div>';
                }
                if (t.tags && t.tags.length) {
                    html += '<div style="margin-bottom:12px;">' + t.tags.map(tg => '<span style="background:var(--card-border);padding:2px 6px;border-radius:3px;font-size:0.62rem;margin-right:4px;">' + _h(tg) + '</span>').join('') + '</div>';
                }

                html += '<div style="border-top:1px solid var(--card-border);padding-top:12px;">';
                html += '<h4 style="margin:0 0 8px;">📑 章节结构</h4>';
                (t.sections || []).forEach(sec => {
                    html += '<div style="margin-bottom:8px;padding:8px;background:var(--bg-color);border-radius:4px;">';
                    html += '<div style="font-weight:600;font-size:0.75rem;">' + _h(sec.title) + '</div>';
                    if (sec.content) {
                        const preview = sec.content.length > 200 ? sec.content.slice(0, 200) + '...' : sec.content;
                        html += '<div style="font-size:0.65rem;color:var(--card-muted);margin-top:4px;white-space:pre-wrap;">' + _h(preview) + '</div>';
                    }
                    html += '</div>';
                });
                html += '</div>';

                detail.innerHTML = html;
                detail.querySelector('.tpl-edit-btn').onclick = () => showEdit(tid);
                detail.querySelector('.tpl-versions-btn').onclick = () => showVersions(tid);
                detail.querySelector('.tpl-delete-btn').onclick = () => {
                    if (!confirm('确认删除此模板？')) return;
                    fetch('/templates/' + tid, { method: 'DELETE', credentials: 'include' })
                        .then(r => r.json())
                        .then(() => { T.refresh(); detail.innerHTML = '<span style="color:var(--card-muted);">模板已删除</span>'; });
                };
            })
            .catch(e => { detail.innerHTML = '<span style="color:#e74c3c;">加载失败: ' + e.message + '</span>'; });
    }

    // ── Create / Edit ──
    function showCreate() {
        showEditor(null);
    }

    function showEdit(tid) {
        fetch('/templates/' + tid, { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                showEditor(data.data || data);
            })
            .catch(e => { if (typeof showToast === 'function') showToast(e.message, 'error'); });
    }

    function showEditor(existing) {
        const isNew = !existing;
        const t = existing || { name: '', category: '工程', description: '', sections: [], tags: [] };
        const detail = document.getElementById('templatesDetail');

        let html = '<h4>' + (isNew ? '新建模板' : '编辑: ' + _h(t.name)) + '</h4>';

        html += '<div style="margin-bottom:8px;"><label style="font-size:0.7rem;">名称</label>';
        html += '<input id="tplEditName" value="' + _h(t.name || '') + '" style="width:100%;font-size:0.75rem;padding:6px;border:1px solid var(--card-border);border-radius:4px;background:var(--card-bg);"></div>';

        html += '<div style="margin-bottom:8px;"><label style="font-size:0.7rem;">分类</label>';
        html += '<select id="tplEditCategory" style="width:100%;font-size:0.75rem;padding:6px;border:1px solid var(--card-border);border-radius:4px;background:var(--card-bg);">';
        ['工程','货物','服务'].forEach(c => {
            html += '<option value="' + c + '"' + (t.category === c ? ' selected' : '') + '>' + c + '</option>';
        });
        html += '</select></div>';

        html += '<div style="margin-bottom:8px;"><label style="font-size:0.7rem;">描述</label>';
        html += '<textarea id="tplEditDesc" style="width:100%;font-size:0.7rem;padding:6px;border:1px solid var(--card-border);border-radius:4px;background:var(--card-bg);" rows="2">' + _h(t.description || '') + '</textarea></div>';

        html += '<div style="margin-bottom:8px;"><label style="font-size:0.7rem;">标签 (逗号分隔)</label>';
        html += '<input id="tplEditTags" value="' + _h((t.tags || []).join(',')) + '" style="width:100%;font-size:0.7rem;padding:6px;border:1px solid var(--card-border);border-radius:4px;background:var(--card-bg);"></div>';

        html += '<div style="margin-bottom:8px;"><label style="font-size:0.7rem;">章节 (JSON 格式)</label>';
        html += '<textarea id="tplEditSections" style="width:100%;font-size:0.65rem;font-family:monospace;padding:6px;border:1px solid var(--card-border);border-radius:4px;background:var(--card-bg);" rows="12">' + JSON.stringify(t.sections || [], null, 2) + '</textarea></div>';

        html += '<div style="margin-bottom:8px;"><label id="tplChangeLabel" style="font-size:0.7rem;display:' + (isNew ? 'none' : 'block') + ';">变更说明</label>';
        html += '<input id="tplEditChangeSummary" placeholder="版本变更说明" style="width:100%;font-size:0.7rem;padding:6px;border:1px solid var(--card-border);border-radius:4px;background:var(--card-bg);display:' + (isNew ? 'none' : 'block') + ';"></div>';

        html += '<div style="display:flex;gap:8px;">';
        html += '<button id="tplSaveBtn" style="padding:6px 16px;background:#27ae60;color:#fff;border:none;border-radius:4px;cursor:pointer;">' + (isNew ? '创建' : '保存') + '</button>';
        html += '<button id="tplCancelBtn" style="padding:6px 16px;border:1px solid var(--card-border);border-radius:4px;cursor:pointer;">取消</button>';
        html += '</div>';
        html += '<span id="tplEditStatus" style="font-size:0.65rem;margin-left:8px;color:var(--card-muted);"></span>';

        detail.innerHTML = html;

        document.getElementById('tplCancelBtn').onclick = () => {
            if (isNew) detail.innerHTML = '<span style="color:var(--card-muted);">已取消</span>';
            else showDetail(t.id);
        };

        document.getElementById('tplSaveBtn').onclick = async () => {
            const name = document.getElementById('tplEditName').value.trim();
            const category = document.getElementById('tplEditCategory').value;
            const description = document.getElementById('tplEditDesc').value.trim();
            const tagsStr = document.getElementById('tplEditTags').value.trim();
            const tags = tagsStr ? tagsStr.split(',').map(s => s.trim()).filter(Boolean) : [];
            let sections;
            try { sections = JSON.parse(document.getElementById('tplEditSections').value); }
            catch (e) { document.getElementById('tplEditStatus').textContent = '章节 JSON 格式错误!'; return; }
            const changeSummary = isNew ? '' : document.getElementById('tplEditChangeSummary').value.trim() || undefined;

            const body = { name, category, description, sections, tags };
            if (changeSummary) body.change_summary = changeSummary;

            const url = isNew ? '/templates' : '/templates/' + t.id;
            const method = isNew ? 'POST' : 'PUT';

            try {
                const resp = await fetch(url, {
                    method, credentials: 'include',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                const data = await resp.json();
                if (!data.success) throw new Error(data.error || 'failed');
                if (isNew) {
                    T.refresh();
                    showDetail((data.data || data).id);
                } else {
                    showDetail(t.id);
                }
                if (typeof showToast === 'function') showToast(isNew ? '模板已创建' : '模板已更新', 'success');
            } catch (e) {
                document.getElementById('tplEditStatus').textContent = '保存失败: ' + e.message;
                document.getElementById('tplEditStatus').style.color = '#e74c3c';
            }
        };
    }

    // ── Import .docx ──
    function showImport() {
        const detail = document.getElementById('templatesDetail');
        let html = '<h4>📥 从 .docx 导入模板</h4>';
        html += '<div style="margin:12px 0;padding:16px;border:2px dashed var(--card-border);border-radius:8px;text-align:center;">';
        html += '<input type="file" id="tplImportFileInput" accept=".docx" style="display:none;">';
        html += '<button id="tplSelectImportFile" style="padding:8px 20px;font-size:0.8rem;border:1px solid var(--card-border);border-radius:4px;cursor:pointer;">选择 .docx 文件</button>';
        html += '<div id="tplImportFileName" style="font-size:0.65rem;color:var(--card-muted);margin-top:4px;"></div>';
        html += '</div>';
        html += '<div id="tplImportPreview" style="margin-top:12px;"></div>';
        detail.innerHTML = html;

        document.getElementById('tplSelectImportFile').onclick = () => {
            document.getElementById('tplImportFileInput').click();
        };

        document.getElementById('tplImportFileInput').onchange = async function () {
            if (!this.files.length) return;
            const file = this.files[0];
            document.getElementById('tplImportFileName').textContent = file.name;

            const formData = new FormData();
            formData.append('file', file);

            const previewDiv = document.getElementById('tplImportPreview');
            previewDiv.innerHTML = '<span style="color:var(--card-muted);">' + _icon('hourglass_empty') + ' 解析中...</span>';

            try {
                const resp = await fetch('/templates/import/preview', { method: 'POST', credentials: 'include', body: formData });
                const data = await resp.json();
                if (!data.success) throw new Error(data.error || 'failed');
                const preview = data.data || data;

                let ph = '<div style="margin-top:8px;padding:12px;background:var(--bg-color);border-radius:6px;">';
                ph += '<div style="font-size:0.75rem;"><strong>检测标题:</strong> ' + _h(preview.detected_title) + '</div>';
                ph += '<div style="font-size:0.7rem;margin-top:4px;"><strong>推测分类:</strong> ' + _h(preview.category_guess) + '</div>';
                ph += '<div style="font-size:0.7rem;margin-top:4px;"><strong>章节数:</strong> ' + (preview.section_count || 0) + '</div>';
                ph += '<div style="margin-top:8px;max-height:300px;overflow-y:auto;">';
                (preview.sections || []).forEach(s => {
                    ph += '<div style="padding:4px 0;border-bottom:1px solid var(--card-border);font-size:0.68rem;">';
                    ph += '<span style="font-weight:500;">' + (s.level === 1 ? 'H1: ' : s.level === 2 ? 'H2: ' : '') + _h(s.title) + '</span>';
                    if (s.content) {
                        ph += '<div style="color:var(--card-muted);margin-left:8px;">' + _h(s.content.slice(0, 80)) + (s.content.length > 80 ? '...' : '') + '</div>';
                    }
                    ph += '</div>';
                });
                ph += '</div>';
                ph += '<div style="margin-top:10px;display:flex;gap:8px;">';
                ph += '<button id="tplImportConfirm" style="padding:6px 14px;background:#27ae60;color:#fff;border:none;border-radius:4px;cursor:pointer;">确认导入</button>';
                ph += '<button id="tplImportCancel" style="padding:6px 14px;border:1px solid var(--card-border);border-radius:4px;cursor:pointer;">取消</button>';
                ph += '</div>';
                if (preview.section_count > 20) ph += '<div style="color:#e67e22;font-size:0.62rem;margin-top:4px;">⚠️ 章节较多，可能需要调整后再导入</div>';
                ph += '</div>';

                previewDiv.innerHTML = ph;

                document.getElementById('tplImportConfirm').onclick = async () => {
                    try {
                        const resp = await fetch('/templates/import/confirm', {
                            method: 'POST', credentials: 'include',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                name: preview.detected_title,
                                category: preview.category_guess,
                                sections: preview.sections,
                            }),
                        });
                        const data = await resp.json();
                        if (!data.success) throw new Error(data.error || 'failed');
                        if (typeof showToast === 'function') showToast('模板导入成功', 'success');
                        T.refresh();
                        showDetail((data.data || data).id);
                    } catch (e) {
                        previewDiv.innerHTML += '<div style="color:#e74c3c;font-size:0.65rem;">导入失败: ' + e.message + '</div>';
                    }
                };
                document.getElementById('tplImportCancel').onclick = () => { previewDiv.innerHTML = ''; };
            } catch (e) {
                previewDiv.innerHTML = '<span style="color:#e74c3c;">解析失败: ' + e.message + '</span>';
            }
        };
    }

    // ── Versions ──
    function showVersions(tid) {
        const detail = document.getElementById('templatesDetail');
        detail.innerHTML = '<span style="color:var(--card-muted);">' + _icon('hourglass_empty') + ' 加载版本...</span>';

        fetch('/templates/' + tid + '/versions', { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                const versions = (data.data || data).versions || [];
                let html = '<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">';
                html += '<button class="tpl-back-detail" style="font-size:0.7rem;border:1px solid var(--card-border);border-radius:3px;cursor:pointer;">← 返回</button>';
                html += '<h4 style="margin:0;">📜 版本历史 (' + versions.length + ')</h4>';
                html += '</div>';
                if (!versions.length) {
                    html += '<span style="color:var(--card-muted);">暂无版本</span>';
                } else {
                    versions.forEach((v, i) => {
                        html += '<div style="margin-bottom:6px;padding:8px;background:var(--bg-color);border-radius:4px;font-size:0.7rem;">';
                        html += '<span style="font-weight:600;">' + _h(v.version_label) + '</span>';
                        if (v.created_at) html += ' <span style="font-size:0.62rem;color:var(--card-muted);">' + v.created_at + '</span>';
                        if (v.change_summary) html += '<div style="font-size:0.62rem;color:var(--card-muted);margin-top:2px;">' + _h(v.change_summary) + '</div>';
                        html += '<div style="margin-top:4px;display:flex;gap:6px;">';
                        html += '<button class="tpl-ver-view" data-vid="' + v.id + '" style="font-size:0.58rem;padding:1px 6px;border:1px solid var(--card-border);border-radius:3px;cursor:pointer;">查看快照</button>';
                        if (i < versions.length - 1) {
                            html += '<button class="tpl-ver-diff" data-from="' + v.id + '" data-to="' + (versions[i+1] ? versions[i+1].id : '') + '" style="font-size:0.58rem;padding:1px 6px;border:1px solid var(--card-border);border-radius:3px;cursor:pointer;">↔️ Diff with v' + (i+2) + '</button>';
                        }
                        html += '</div></div>';
                    });
                }
                detail.innerHTML = html;

                detail.querySelector('.tpl-back-detail').onclick = () => showDetail(tid);
                detail.querySelectorAll('.tpl-ver-view').forEach(btn => {
                    btn.onclick = () => {
                        const vid = parseInt(btn.dataset.vid);
                        viewVersionSnapshot(tid, vid);
                    };
                });
                detail.querySelectorAll('.tpl-ver-diff').forEach(btn => {
                    btn.onclick = () => {
                        const from = parseInt(btn.dataset.from);
                        const to = parseInt(btn.dataset.to);
                        if (from && to) showDiff(tid, from, to);
                    };
                });
            })
            .catch(e => { detail.innerHTML = '<span style="color:#e74c3c;">加载失败: ' + e.message + '</span>'; });
    }

    function viewVersionSnapshot(tid, vid) {
        const detail = document.getElementById('templatesDetail');
        detail.innerHTML = '<span style="color:var(--card-muted);">' + _icon('hourglass_empty') + ' 加载快照...</span>';

        fetch('/templates/' + tid + '/versions/' + vid, { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                const v = data.data || data;
                const sections = (v.snapshot && v.snapshot.sections) || [];
                let html = '<button class="tpl-back-versions" style="font-size:0.7rem;border:1px solid var(--card-border);border-radius:3px;cursor:pointer;margin-bottom:8px;">← 返回版本列表</button>';
                html += '<h4>快照: ' + _h(v.version_label) + '</h4>';
                html += '<div style="font-size:0.68rem;color:var(--card-muted);margin-bottom:8px;">' + (v.created_at || '') + '</div>';
                if (v.change_summary) html += '<div style="font-size:0.7rem;color:var(--card-muted);margin-bottom:12px;">' + _h(v.change_summary) + '</div>';
                sections.forEach(s => {
                    html += '<div style="margin-bottom:8px;padding:8px;background:var(--bg-color);border-radius:4px;">';
                    html += '<div style="font-weight:600;font-size:0.75rem;">' + _h(s.title) + '</div>';
                    if (s.content) html += '<div style="font-size:0.65rem;color:var(--card-muted);margin-top:4px;white-space:pre-wrap;">' + _h(s.content) + '</div>';
                    html += '</div>';
                });
                detail.innerHTML = html;
                detail.querySelector('.tpl-back-versions').onclick = () => showVersions(tid);
            })
            .catch(e => { detail.innerHTML = '<span style="color:#e74c3c;">加载失败: ' + e.message + '</span>'; });
    }

    // ── Diff ──
    function showDiff(tid, fromVid, toVid) {
        const detail = document.getElementById('templatesDetail');
        detail.innerHTML = '<span style="color:var(--card-muted);">' + _icon('hourglass_empty') + ' 加载差异...</span>';

        fetch('/templates/' + tid + '/diff?from=' + fromVid + '&to=' + toVid, { credentials: 'include' })
            .then(r => r.json())
            .then(data => {
                if (!data.success) throw new Error(data.error || 'failed');
                const d = data.data || data;
                const changes = d.changes || [];
                const summary = d.summary || {};
                let html = '<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">';
                html += '<button class="tpl-back-versions" style="font-size:0.7rem;border:1px solid var(--card-border);border-radius:3px;cursor:pointer;">← 返回版本列表</button>';
                html += '<strong style="font-size:0.85rem;">↔️ 版本差异</strong>';
                html += '</div>';
                html += '<div style="display:flex;gap:12px;margin-bottom:10px;font-size:0.68rem;">';
                html += '<span style="color:#27ae60;">➕ 新增: ' + (summary.added || 0) + '</span>';
                html += '<span style="color:#e74c3c;">➖ 删除: ' + (summary.removed || 0) + '</span>';
                html += '<span style="color:#e67e22;">✏️ 修改: ' + (summary.changed || 0) + '</span>';
                html += '</div>';
                if (!changes.length) {
                    html += '<p style="font-size:0.75rem;color:var(--card-muted);">两个版本完全相同。</p>';
                }
                changes.forEach(c => {
                    const bg = c.status === 'added' ? '#e6ffe6' : c.status === 'removed' ? '#ffe6e6' : '#fff8e6';
                    const tag = c.status === 'added' ? '➕ 新增' : c.status === 'removed' ? '➖ 删除' : '✏️ 修改';
                    const tagColor = c.status === 'added' ? '#27ae60' : c.status === 'removed' ? '#e74c3c' : '#e67e22';
                    html += '<div style="margin-bottom:8px;background:' + bg + ';border-radius:6px;padding:8px;font-size:0.7rem;">';
                    html += '<div style="font-weight:600;margin-bottom:4px;"><span style="background:' + tagColor + ';color:#fff;padding:0 4px;border-radius:3px;font-size:0.6rem;">' + tag + '</span> ' + _h(c.title || '') + '</div>';
                    if (c.diff_html) html += '<div style="font-size:0.65rem;overflow-x:auto;">' + c.diff_html + '</div>';
                    html += '</div>';
                });
                detail.innerHTML = html;
                detail.querySelector('.tpl-back-versions').onclick = () => showVersions(tid);
            })
            .catch(e => { detail.innerHTML = '<span style="color:#e74c3c;">差异加载失败: ' + e.message + '</span>'; });
    }

    function _h(s) { return (s || '').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

    // ── Auto-init ──
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initPanel);
    } else {
        initPanel();
    }

    // Listen for tab switching to refresh
    const observer = new MutationObserver(function () {
        const panel = document.getElementById('templatesPanel');
        if (panel && panel.style.display !== 'none' && !document.querySelector('.tpl-cat-btn')) {
            initPanel();
        }
    });
    const panel = document.getElementById('templatesPanel');
    if (panel) observer.observe(panel, { attributes: true, attributeFilter: ['style'] });

    // ── AI Template Recommendation (U8) ──

    function showRecommend() {
        const detail = document.getElementById('templatesDetail');
        detail.innerHTML = `
            <h4 style="margin:0 0 10px;">🤖 AI 模板推荐</h4>
            <div style="margin-bottom:8px;">
                <label style="font-size:0.7rem;">项目类型</label>
                <select id="recProjectType" style="width:100%;padding:6px;border:1px solid var(--card-border);border-radius:4px;font-size:0.7rem;margin-bottom:4px;">
                    <option value="">自动检测</option>
                    <option value="工程">🏗️ 工程</option>
                    <option value="货物">📦 货物</option>
                    <option value="服务">💼 服务</option>
                </select>
            </div>
            <div style="margin-bottom:8px;">
                <label style="font-size:0.7rem;">投标文件描述（可选，提升匹配精度）</label>
                <textarea id="recBidText" rows="3" placeholder="输入投标项目描述或粘贴招标文件摘要..." style="width:100%;padding:6px;border:1px solid var(--card-border);border-radius:4px;font-size:0.68rem;resize:vertical;"></textarea>
            </div>
            <div style="margin-bottom:8px;">
                <label style="font-size:0.7rem;">推荐数量</label>
                <input id="recTopK" type="number" value="5" min="1" max="20" style="width:100%;padding:6px;border:1px solid var(--card-border);border-radius:4px;font-size:0.7rem;">
            </div>
            <div style="display:flex;gap:8px;margin-bottom:10px;">
                <button id="recSubmitBtn" style="padding:6px 16px;background:#8e44ad;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:0.7rem;">获取推荐</button>
                <span id="recStatus" style="font-size:0.65rem;color:var(--card-muted);align-self:center;"></span>
            </div>
            <div id="recResults"></div>
        `;

        document.getElementById('recSubmitBtn').onclick = () => {
            const statusEl = document.getElementById('recStatus');
            const resultsEl = document.getElementById('recResults');
            statusEl.textContent = '分析中...';
            resultsEl.innerHTML = '';

            const bidText = document.getElementById('recBidText').value.trim();
            const projectType = document.getElementById('recProjectType').value;
            const topK = parseInt(document.getElementById('recTopK').value) || 5;

            fetch('/templates/recommend', {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_type: projectType || null,
                    bid_text: bidText || null,
                    top_k: topK,
                }),
            })
                .then(r => r.json())
                .then(data => {
                    if (!data.success) throw new Error(data.error || 'failed');
                    const recs = (data.data || data).recommendations || [];
                    if (!recs.length) {
                        statusEl.textContent = '未找到匹配的模板';
                        return;
                    }
                    statusEl.textContent = '找到 ' + recs.length + ' 个推荐';
                    let html = '';
                    recs.forEach((r, i) => {
                        const barPct = Math.round(r.final_score * 100);
                        const barColor = barPct >= 70 ? '#27ae60' : barPct >= 40 ? '#f39c12' : '#e74c3c';
                        html += '<div style="margin-bottom:8px;padding:8px;background:var(--bg-color);border-radius:6px;cursor:pointer;" class="rec-item" data-id="' + r.id + '">';
                        html += '<div style="display:flex;justify-content:space-between;align-items:center;">';
                        html += '<span style="font-weight:600;font-size:0.72rem;">' + (i + 1) + '. ' + _h(r.name) + '</span>';
                        html += '<span style="font-size:0.62rem;color:var(--card-muted);">' + _h(r.category || '') + '</span>';
                        html += '</div>';
                        html += '<div style="margin-top:4px;height:4px;background:var(--card-border);border-radius:2px;">';
                        html += '<div style="height:100%;width:' + barPct + '%;background:' + barColor + ';border-radius:2px;"></div></div>';
                        html += '<div style="display:flex;justify-content:space-between;font-size:0.58rem;color:var(--card-muted);margin-top:2px;">';
                        html += '<span>匹配度 ' + barPct + '%</span>';
                        if (r.reasons && r.reasons.length) {
                            html += '<span>' + r.reasons.join(' · ') + '</span>';
                        }
                        html += '</div></div>';
                    });
                    resultsEl.innerHTML = html;
                    resultsEl.querySelectorAll('.rec-item').forEach(el => {
                        el.onclick = () => {
                            const tid = parseInt(el.dataset.id);
                            T.showDetail(tid);
                            // Log usage
                            fetch('/templates/recommend', { method: 'POST', credentials: 'include', headers: {'Content-Type':'application/json'}, body: JSON.stringify({template_id: tid, log_usage: true}) });
                            if (typeof showToast === 'function') showToast('已记录模板使用', 'info');
                        };
                    });
                })
                .catch(e => {
                    statusEl.textContent = '推荐失败: ' + e.message;
                    statusEl.style.color = '#e74c3c';
                });
        };
    }

})();
