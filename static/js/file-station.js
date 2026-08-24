/* File Station + Knowledge Base Modal module - extracted from app.js (W5) */
    // ======================== File Station Functions ========================
    var fileStationData = [];
    var selectedFileIds = new Set();
    var fileStationBtn = document.getElementById('inputFileStationBtn') || document.getElementById('clearanceFileStationBtn');
    var fileStationModal = document.getElementById('fileStationModal');
    var closeFileStationModal = document.getElementById('closeFileStationModal');

    function openFileStation() {
        if (fileStationModal) {
            loadFileStation();          // load the file list
            fileStationModal.style.display = 'block';
        } else {
            console.error('File station modal not found');
        }
    }
    window.__openFileStation = window.__openFileStation || openFileStation;

    if (fileStationBtn) {
        fileStationBtn.onclick = openFileStation;
    }

    // ── Chat toolbar: daily report button ──
    var dailyReportChatBtn = document.getElementById('dailyReportChatBtn');
    if (dailyReportChatBtn) {
        dailyReportChatBtn.onclick = async () => {
            showToast('⏳ AI正在汇总今日对话并生成日报...', 'info', 10000);
            try {
                const r = await fetch('/my_daily_report', { method:'POST', credentials:'include' });
                const d = await r.json();
                if (r.ok) {
                    showToast(`✅ ${d.filename} · ${d.size_kb}KB · 已保存至文件站`, 'success', 5000);
                    const a = document.createElement('a'); a.href = d.download_url; a.download = d.filename;
                    a.style.display = 'none'; document.body.appendChild(a); a.click(); document.body.removeChild(a);
                } else { showToast(d.error || '生成失败', 'error'); }
            } catch(_) { showToast('网络错误', 'error'); }
        };
    }

    if (closeFileStationModal) {
        closeFileStationModal.onclick = () => {
            if (fileStationModal) fileStationModal.style.display = 'none';
        };
    }
    // Also close modal when clicking outside content
    if (fileStationModal) {
        fileStationModal.onclick = (e) => {
            if (e.target === fileStationModal) fileStationModal.style.display = 'none';
        };
    }
    // File station upload handling
    var uploadToStationBtn = document.getElementById('uploadToStationBtn');
    var stationFileInput = document.getElementById('stationFileInput');
    if (uploadToStationBtn && stationFileInput) {
        uploadToStationBtn.onclick = () => {
            stationFileInput.click();
        };
        stationFileInput.onchange = async (e) => {
            if (e.target.files.length) {
                const statusSpan = document.getElementById('uploadStatus');
                for (const file of e.target.files) {
                    if (statusSpan) statusSpan.textContent = `上传 ${file.name}...`;
                    await uploadToStation(file);
                }
                stationFileInput.value = ''; // reset so you can upload same file again
                await loadFileStation(); // refresh the list
                if (statusSpan) statusSpan.textContent = '';
            }
        };
    }
    async function loadFileStation() {
        const container = document.getElementById('fileStationList');
        if (!container) return;
        container.innerHTML = '<p>加载中...</p>';
        try {
            const res = await fetch('/get_file_station', { credentials: 'include' });
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                container.innerHTML = '<p>加载失败，请检查权限后重试</p>';
                return;
            }
            const data = await res.json();
            const files = data.files || [];
            const isAdmin = data.is_admin || false;
            const isAnon = data.is_anon || false;
            fileStationData = files;

            if (files.length === 0) {
                container.innerHTML = '<p>暂无文件。拖拽或点击上传。</p>';
                return;
            }

            const personalFiles = files.filter(f => f.source === 'user_file');
            const projectFiles = files.filter(f => f.source === 'project_file');

            function renderFileRows(fileArray, sourceType) {
                let html = '';
                for (const f of fileArray) {
                    const sizeKB = (f.size_bytes / 1024).toFixed(1);
                    const created = new Date(f.created_at).toLocaleString();
                    const expires = f.expires_at ? new Date(f.expires_at).toLocaleString() : '三天';
                    const location = f.project_name ? `${escapeHtml(f.project_name)} → ${escapeHtml(f.folder_path || '根目录')}` : '—';
                    const restoreBadge = (f.meta_data && f.meta_data.restored_from === 'chat_deletion') ? '<span style="margin-left:8px; background:#9b59b6; color:white; border-radius:12px; padding:2px 6px; font-size:0.65rem;">♻️ 恢复自聊天</span>' : '';

                    let actions = '';
                    if (sourceType === 'user_file') {
                        actions += `<button class="file-download" data-source="user_file" data-filename="${escapeHtml(f.filename)}" style="background:#2c3e50; color:white; border:none; border-radius:4px; padding:4px 8px;">⬇️ 下载原文件</button>`;
                        if (!isAdmin) {
                            actions += `<button class="file-load" data-id="${f.id}" style="margin-left:6px; background:#3498db; color:white; border:none; border-radius:4px; padding:4px 8px;">📂 加载到聊天</button>`;
                        }
                        if (isAdmin || f.can_delete === true) {
                            actions += `<button class="file-delete" data-source="user_file" data-id="${f.id}" style="margin-left:6px; background:#e74c3c; color:white; border:none; border-radius:4px; padding:4px 8px;">🗑️ 删除</button>`;
                        }
                    } else {
                        actions += `<button class="file-download-project" data-project-id="${f.project_id}" data-file-id="${f.id}" data-filename="${escapeHtml(f.filename)}" style="background:#2c3e50; color:white; border:none; border-radius:4px; padding:4px 8px;">⬇️ 下载文件</button>`;
                        if (!isAdmin) {
                            actions += `<button class="file-load-project" data-project-id="${f.project_id}" data-file-id="${f.id}" data-filename="${escapeHtml(f.filename)}" style="margin-left:6px; background:#3498db; color:white; border:none; border-radius:4px; padding:4px 8px;">📂 加载到聊天</button>`;
                        }
                        if (isAdmin) {
                            actions += `<button class="file-delete-project" data-project-id="${f.project_id}" data-file-id="${f.id}" style="margin-left:6px; background:#e74c3c; color:white; border:none; border-radius:4px; padding:4px 8px;">🗑️ 删除</button>`;
                        }
                    }

                    html += `
                        <tr class="file-item" data-id="${f.id}" data-source="${f.source}" data-filename="${escapeHtml(f.filename)}" data-project-id="${f.project_id || ''}">
                            <td style="padding:8px; border-bottom:1px solid #eee;">${escapeHtml(f.filename)}${restoreBadge}</td>
                            <td style="padding:8px; border-bottom:1px solid #eee;">${sizeKB}</td>
                            <td style="padding:8px; border-bottom:1px solid #eee;">${location}</td>
                            <td style="padding:8px; border-bottom:1px solid #eee;">${created}</td>
                            <td style="padding:8px; border-bottom:1px solid #eee;">${expires}</td>
                            <td style="padding:8px; border-bottom:1px solid #eee; white-space:nowrap;">${actions}</td>
                        </tr>
                    `;
                }
                return html;
            }

            let html = `<div style="overflow-x: auto;">`;
            if (personalFiles.length > 0) {
                html += `<h4 style="margin: 16px 0 8px 0; color: #2c3e50;">📁 个人文件</h4>`;
                html += `<table style="width:100%; border-collapse: collapse; font-size:0.8rem;">`;
                html += `<thead><tr class="file-item"><th style="padding:8px; border-bottom:1px solid #ddd;">文件名</th><th>大小(KB)</th><th>位置</th><th>上传时间</th><th>过期时间</th><th>操作</th></tr></thead><tbody>`;
                html += renderFileRows(personalFiles, 'user_file');
                html += `</tbody></table>`;
            }
            if (projectFiles.length > 0) {
                html += `<h4 style="margin: 16px 0 8px 0; color: #2c3e50;">🏗️ 项目文件</h4>`;
                html += `<table style="width:100%; border-collapse: collapse; font-size:0.8rem;">`;
                html += `<thead><tr class="file-item"><th style="padding:8px; border-bottom:1px solid #ddd;">文件名</th><th>大小(KB)</th><th>位置</th><th>上传时间</th><th>过期时间</th><th>操作</th><tr></thead><tbody>`;
                html += renderFileRows(projectFiles, 'project_file');
                html += `</tbody></table>`;
            }
            html += `</div>`;
            container.innerHTML = html;

            // Attach event handlers
            document.querySelectorAll('.file-download').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const filename = btn.dataset.filename;
                    const res = await fetch('/download_original_file', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        credentials: 'include',
                        body: JSON.stringify({ filename })
                    });
                    if (res.ok) {
                        const blob = await res.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = filename;
                        a.click();
                        window.URL.revokeObjectURL(url);
                    } else {
                        alert('下载失败');
                    }
                };
            });
            document.querySelectorAll('.file-download-project').forEach(btn => {
                btn.onclick = (e) => {
                    e.stopPropagation();
                    const projectId = btn.dataset.projectId;
                    const fileId = btn.dataset.fileId;
                    const filename = btn.dataset.filename;
                    window.open(`/admin/projects/${projectId}/files/${fileId}/download`, '_blank');
                };
            });
            document.querySelectorAll('.file-load').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const fileId = btn.dataset.id;
                    const file = personalFiles.find(f => f.id === fileId);
                    if (file) {
                        await loadSelectedFiles([fileId]);
                    }
                };
            });
            document.querySelectorAll('.file-load-project').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const projectId = btn.dataset.projectId;
                    const fileId = btn.dataset.fileId;
                    const filename = btn.dataset.filename;
                    try {
                        const res = await fetch('/load_project_file', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            credentials: 'include',
                            body: JSON.stringify({ project_id: projectId, file_id: fileId })
                        });
                        const data = await res.json();
                        if (res.ok && data.content) {
                            const blob = new Blob([data.content], { type: 'text/plain' });
                            const fakeFile = new File([blob], filename, { type: 'text/plain' });
                            selectedFiles.push(fakeFile);
                            showPersistentReminder(selectedFiles);
                            if (selectedFiles.length === 1) fileBtn.innerText = `📄 ${selectedFiles[0].name}`;
                            else fileBtn.innerText = `📄 ${selectedFiles.length} 个文件`;
                            const modal = document.getElementById('fileStationModal');
                            if (modal) modal.style.display = 'none';
                        } else {
                            alert('加载文件失败，请检查文件是否已被删除');
                        }
                    } catch (err) {
                        console.error(err);
                        alert('网络错误，加载失败');
                    }
                };
            });
            document.querySelectorAll('.file-delete').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const id = btn.dataset.id;
                    if (!await confirm('删除此文件？回收站保存时间3天。')) return;
                    const res = await fetch('/delete_file_station', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        credentials: 'include',
                        body: JSON.stringify({ file_id: id })
                    });
                    if (res.ok) {
                        showToast('文件已移至回收站', 'success', 2000);
                        loadFileStation();
                    } else {
                        const err = await res.json();
                        alert('删除失败: ' + (err.error || '未知错误'));
                    }
                };
            });
            document.querySelectorAll('.file-delete-project').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const projectId = btn.dataset.projectId;
                    const fileId = btn.dataset.fileId;
                    if (!await confirm('永久删除此项目文件？不可恢复。')) return;
                    const res = await fetch(`/admin/projects/${projectId}/files/${fileId}`, {
                        method: 'DELETE',
                        credentials: 'include'
                    });
                    if (res.ok) {
                        showToast('文件已删除', 'success', 2000);
                        loadFileStation();
                    } else {
                        const err = await res.json();
                        alert('删除失败: ' + (err.error || '未知错误'));
                    }
                };
            });
            const stationContainer = document.getElementById('fileStationList');
            if (stationContainer) {
                new FileListManager(stationContainer, {
                    selectableClass: '.file-item',
                    onSelectionChange: (items) => {
                        selectedFileIds = new Set(Array.from(items).map(el => el.dataset.id));
                    },
                    onDoubleClick: (item) => {
                        const source = item.dataset.source;
                        const id = item.dataset.id;
                        const filename = item.dataset.filename;
                        if (source === 'user_file') {
                            loadSelectedFiles([id]);
                        } else if (source === 'project_file') {
                            const projectId = item.dataset.projectId;
                            window.open(`/admin/projects/${projectId}/files/${id}/download`, '_blank');
                        }
                    }
                });
            }
        } catch (err) {
            console.error(err);
            container.innerHTML = '<p>加载失败，网络错误</p>';
        }
    }

    document.getElementById('loadSelectedBtn')?.addEventListener('click', () => loadSelectedFiles());

    async function loadSelectedFiles(fileIds = null) {
        const ids = fileIds || Array.from(selectedFileIds);
        if (ids.length === 0) { alert('请先选择文件。'); return; }
        const newFiles = [];
        for (const id of ids) {
            const fileData = fileStationData.find(f => String(f.id) === String(id));
            if (fileData && fileData.source === 'user_file') {
                try {
                    const res = await fetch('/load_cached_file', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        credentials: 'include',
                        body: JSON.stringify({ filename: fileData.filename })
                    });
                    const data = await res.json();
                    if (data.content) {
                        const blob = new Blob([data.content], { type: 'text/plain' });
                        const fakeFile = new File([blob], fileData.filename, { type: 'text/plain' });
                        newFiles.push(fakeFile);
                    } else {
                        alert(`无法加载文件 "${fileData.filename}": 内容为空`);
                    }
                } catch (err) {
                    console.error(err);
                    alert(`加载文件 "${fileData.filename}" 失败: 网络错误`);
                }
            } else if (fileData && fileData.source === 'project_file') {
                alert('项目文件请使用“加载到聊天”按钮单独加载。');
            }
        }
        if (newFiles.length > 0) {
            selectedFiles = newFiles;
            showPersistentReminder(selectedFiles);
            const modal = document.getElementById('fileStationModal');
            if (modal) modal.style.display = 'none';
            if (newFiles.length === 1) fileBtn.innerText = `📄 ${newFiles[0].name}`;
            else fileBtn.innerText = `📄 ${newFiles.length} 个文件`;
        } else {
            alert('无法加载选中的文件内容。');
        }
    }

    async function uploadToStation(file) {
        const formData = new FormData();
        formData.append('file', file);
        const statusSpan = document.getElementById('uploadStatus');
        if (statusSpan) statusSpan.textContent = '检查中...';
        try {
            const checkRes = await fetch('/upload_file', {
                method: 'POST',
                credentials: 'include',
                body: formData
            });
            const data = await checkRes.json();

            if (data.exists) {
                const reuse = confirm(`文件 "${data.filename}" 已存在。\n是否使用系统已提取的内容？\n\n点击“确定”使用现有文件，点击“取消”重新上传并覆盖。`);
                if (reuse) {
                    await loadFileStation();
                    if (statusSpan) statusSpan.textContent = `✅ 已保留现有文件: ${data.filename}`;
                    setTimeout(() => { if (statusSpan) statusSpan.textContent = ''; }, 3000);
                    return;
                } else {
                    const forceFormData = new FormData();
                    forceFormData.append('file', file);
                    forceFormData.append('force', 'true');
                    const uploadRes = await fetch('/upload_file', {
                        method: 'POST',
                        credentials: 'include',
                        body: forceFormData
                    });
                    const uploadData = await uploadRes.json();
                    if (uploadRes.ok && uploadData.success) {
                        if (statusSpan) statusSpan.textContent = `✅ ${uploadData.filename} 上传成功（已覆盖）`;
                        await loadFileStation();
                        setTimeout(() => { if (statusSpan) statusSpan.textContent = ''; }, 3000);
                    } else {
                        if (statusSpan) statusSpan.textContent = '❌ 上传失败: ' + (uploadData.error || '未知错误');
                    }
                    return;
                }
            }

            if (checkRes.ok && data.success) {
                if (statusSpan) statusSpan.textContent = `✅ ${data.filename} 上传成功`;
                await loadFileStation();
                setTimeout(() => { if (statusSpan) statusSpan.textContent = ''; }, 3000);
            } else {
                if (statusSpan) statusSpan.textContent = '❌ 上传失败: ' + ('操作失败，请重试');
            }
        } catch (err) {
            if (statusSpan) statusSpan.textContent = '❌ 网络错误';
            console.error(err);
        }
    }

    function closeFileStationAndClearSelection() {
        const modal = document.getElementById('fileStationModal');
        if (modal) modal.style.display = 'none';
        selectedFileIds.clear();
    }

    // Global variable to store selected knowledge files and IDs
    var selectedKnowledgeFiles = [];
    try {
        const stored = localStorage.getItem('selectedKnowledgeFiles');
        if (stored) {
            selectedKnowledgeFiles = JSON.parse(stored);
        }
    } catch (e) {
        console.warn('Failed to parse selectedKnowledgeFiles from localStorage:', e);
        selectedKnowledgeFiles = [];
    }
    // Initialize category filter bar visibility
    showCatFilterIfNeeded();
    async function loadKnowledgeFiles() {
        const container = document.getElementById('knowledgeFileList');
        if (!container) return;
        container.innerHTML = '<p>加载中...</p>';

        // Personal Knowledge Lab files
        const labRes = await fetch('/knowledge_lab/list', { credentials: 'include' });
        const labData = await labRes.json();
        const labFiles = labData.files || [];

        // Company Knowledge Base files
        const companyRes = await fetch('/company_kb/list', { credentials: 'include' });
        const companyData = await companyRes.json();
        const companyFiles = companyData.files || [];

        // Skill-only filter toggle state
        const showSkillsOnly = sessionStorage.getItem('kbModalShowSkillsOnly') === '1';

        let html = '';

        // Filter toggle
        html += `<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <label style="font-size:0.78rem; cursor:pointer; display:flex; align-items:center; gap:4px;">
                <input type="checkbox" id="kbSkillFilter" ${showSkillsOnly ? 'checked' : ''}
                       onchange="sessionStorage.setItem('kbModalShowSkillsOnly', this.checked?'1':'0'); loadKnowledgeFiles();">
                🧠 仅显示已提取技能的文件
            </label>
            <small style="color:var(--card-muted);">勾选文件以附加到当前对话</small>
        </div>`;

        // Section 1: Personal Knowledge Lab (collapsible)
        const labFiltered = showSkillsOnly ? labFiles.filter(f => f.has_skill) : labFiles;
        html += `<details open>
                    <summary style="cursor:pointer; font-weight:bold; margin:8px 0; font-size:0.9rem;">📁 我的知识库 (${labFiltered.length}/${labFiles.length})</summary>`;
        if (labFiltered.length) {
            html += `<ul style="margin:0; padding-left:16px; list-style:none;">`;
            for (const f of labFiltered) {
                const checked = selectedKnowledgeFiles.some(sf => sf.source === 'knowledge_lab' && String(sf.id) === String(f.id));
                html += `<li style="margin:6px 0; padding:4px 0; border-bottom:1px solid #eee;">
                            <label style="display:flex; align-items:center; gap:8px; font-size:0.8rem; flex-wrap:wrap;">
                                <input type="checkbox" class="knowledge-checkbox"
                                       data-source="knowledge_lab" data-id="${f.id}"
                                       data-filename="${escapeHtml(f.original_name)}" ${checked ? 'checked' : ''}>
                                <span><strong>${escapeHtml(f.original_name)}</strong> (${(f.file_size/1024).toFixed(1)} KB)</span>
                                ${f.has_skill ? '<span style="font-size:0.65rem; background:#dcfce7; color:#16a34a; border-radius:8px; padding:0 5px;">🧠 技能</span>' : ''}
                                <span style="font-size:0.7rem; color:#888;">${new Date(f.uploaded_at).toLocaleString()}</span>
                            </label>
                         </li>`;
            }
            html += `</ul>`;
        } else {
            html += `<p style="margin:8px 0 8px 16px; font-size:0.8rem; color:#888;">${showSkillsOnly ? '暂无已提取技能的文件' : '暂无文件。请先在知识库实验室标签页上传。'}</p>`;
        }
        html += `</details>`;

        // Section 2: Company Knowledge Base (collapsible)
        const coFiltered = showSkillsOnly ? companyFiles.filter(f => f.has_skill) : companyFiles;
        html += `<details open>
                    <summary style="cursor:pointer; font-weight:bold; margin:8px 0; font-size:0.9rem;">🏢 公司知识库 (${coFiltered.length}/${companyFiles.length})</summary>`;
        if (coFiltered.length) {
            html += `<ul style="margin:0; padding-left:16px; list-style:none;">`;
            for (const f of coFiltered) {
                const checked = selectedKnowledgeFiles.some(sf => sf.source === 'company_kb' && String(sf.id) === String(f.id));
                html += `<li style="margin:6px 0; padding:4px 0; border-bottom:1px solid #eee;">
                            <label style="display:flex; align-items:center; gap:8px; font-size:0.8rem; flex-wrap:wrap;">
                                <input type="checkbox" class="knowledge-checkbox"
                                       data-source="company_kb" data-id="${f.id}"
                                       data-filename="${escapeHtml(f.filename)}" ${checked ? 'checked' : ''}>
                                <span><strong>${escapeHtml(f.filename)}</strong> (${(f.file_size/1024).toFixed(1)} KB)</span>
                                ${f.has_skill ? '<span style="font-size:0.65rem; background:#dcfce7; color:#16a34a; border-radius:8px; padding:0 5px;">🧠 技能</span>' : ''}
                                <span style="font-size:0.7rem; color:#888;">分类: ${escapeHtml(f.category || '无')}</span>
                                <span style="font-size:0.7rem; color:#888;">${escapeHtml(f.uploaded_by_name || 'admin')}</span>
                            </label>
                         </li>`;
            }
            html += `</ul>`;
        } else {
            html += `<p style="margin:8px 0 8px 16px; font-size:0.8rem; color:#888;">${showSkillsOnly ? '暂无已提取技能的文件' : '暂无公司知识库文件。'}</p>`;
        }
        html += `</details>`;

        if (!labFiles.length && !companyFiles.length) {
            html = '<p>暂无可用文件。请上传文件到知识库实验室或联系管理员上传公司知识库。</p>';
        }

        container.innerHTML = html;

        // Attach change event to checkboxes
        document.querySelectorAll('.knowledge-checkbox').forEach(cb => {
            cb.onchange = () => {
                const source = cb.dataset.source;
                const id = cb.dataset.id;
                const filename = cb.dataset.filename;
                if (cb.checked) {
                    selectedKnowledgeFiles.push({ source, id, filename });
                } else {
                    selectedKnowledgeFiles = selectedKnowledgeFiles.filter(sf => !(sf.source === source && String(sf.id) === String(id)));
                }
                const btn = document.getElementById('knowledgeBaseBtn');
                if (selectedKnowledgeFiles.length) {
                    btn.innerHTML = `📚 知识库(${selectedKnowledgeFiles.length})`;
                } else {
                    btn.innerHTML = '📚 知识库';
                }
                localStorage.setItem('selectedKnowledgeFiles', JSON.stringify(selectedKnowledgeFiles));
                showCatFilterIfNeeded();
            };
        });
    }

    // Open knowledge base modal
    const kbBtn = document.getElementById('knowledgeBaseBtn');
    if (kbBtn) kbBtn.onclick = () => {
        loadKnowledgeFiles();
        document.getElementById('knowledgeBaseModal').style.display = 'block';
    };

    // Close modal
    function closeKnowledgeModal() {
        document.getElementById('knowledgeBaseModal').style.display = 'none';
        // Update category filter bar visibility after closing
        showCatFilterIfNeeded();
    }
    const closeKBBtn = document.getElementById('closeKnowledgeBaseModal');
    if (closeKBBtn) closeKBBtn.onclick = closeKnowledgeModal;
    const cancelKBBtn = document.getElementById('cancelKnowledgeBtn');
    if (cancelKBBtn) cancelKBBtn.onclick = closeKnowledgeModal;

    // Confirm selection
    const confirmKBBtn = document.getElementById('confirmKnowledgeBtn');
    if (confirmKBBtn) confirmKBBtn.onclick = () => {
        const checkboxes = document.querySelectorAll('#knowledgeFileList .knowledge-checkbox:checked');
        selectedKnowledgeFiles = Array.from(checkboxes).map(cb => ({
            source: cb.dataset.source,
            id: cb.dataset.id,
            filename: cb.dataset.filename
        }));
        closeKnowledgeModal();
        // Update button text to show count
        const btn = document.getElementById('knowledgeBaseBtn');
        if (selectedKnowledgeFiles.length) {
            btn.innerHTML = `📚 知识库(${selectedKnowledgeFiles.length})`;
        } else {
            btn.innerHTML = '📚 知识库';
        }
        // Update category filter bar visibility
        showCatFilterIfNeeded();
    };

