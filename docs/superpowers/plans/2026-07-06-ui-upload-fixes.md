# UI Upload Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 3 UI bugs: stuck drag-drop watermark, never-ending progress bar, and missing batch duplicate-file handling with "yes to all" + lazy content compare.

**Architecture:** All changes confined to `static/js/app.js` (~120 lines changed), `static/css/app.css` (~60 lines added), and `app/routes/admin.py` (~55 lines added). Tasks 1 and 2 are independent. Task 4 depends on Task 3. Each task produces a testable increment.

**Tech Stack:** Vanilla JS (no framework), Flask + psycopg2, plain CSS

## Global Constraints

- No new dependencies
- Follow existing code patterns (vanilla DOM manipulation, `fetch` + `credentials: 'include'`)
- Chinese UI strings (matching existing convention)
- All progress paths must call `finishProgress()`

---

### Task 1: Fix Drag-Drop Watermark (Enter/Leave Counter)

**Files:**
- Modify: `static/js/app.js:8790-8817`

**Interfaces:**
- Produces: `_dragCounter` module-level variable (private, no consumers)

- [ ] **Step 1: Replace the drag listeners**

Replace lines 8790-8817 in `static/js/app.js`:

```js
// ======================== Drag-and-Drop File Upload ========================
let _dragCounter = 0;
['dragenter', 'dragover', 'dragleave', 'drop', 'dragend'].forEach(evt => {
    document.addEventListener(evt, function(e) { e.preventDefault(); e.stopPropagation(); }, false);
});
document.addEventListener('dragenter', function() {
    _dragCounter++;
    chatInterface.classList.add('drag-over');
});
document.addEventListener('dragleave', function() {
    _dragCounter--;
    if (_dragCounter <= 0) { _dragCounter = 0; chatInterface.classList.remove('drag-over'); }
});
document.addEventListener('drop', function() {
    _dragCounter = 0;
    chatInterface.classList.remove('drag-over');
});
document.addEventListener('dragend', function() {
    _dragCounter = 0;
    chatInterface.classList.remove('drag-over');
});
chatInterface.addEventListener('drop', function(e) {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
        for (let i = 0; i < dt.files.length; i++) {
            selectedFiles.push(dt.files[i]);
        }
        const count = selectedFiles.length;
        fileBtn.innerText = count > 1 ? `📄 已选${count}个文件` : `📄 ${selectedFiles[0].name}`;
        showToast(`已添加 ${dt.files.length} 个文件`, 'success');
    }
});
```

- [ ] **Step 2: Verify no regressions**

Manual test in browser:
1. Drag a file over chat area → watermark appears
2. Drag away → watermark disappears
3. Drag over child elements within chat → watermark stays
4. Drop a file → watermark disappears, files added to selection
5. Drag outside browser window and release → watermark disappears (dragend safety net)
6. Press Escape during drag → watermark disappears

- [ ] **Step 3: Commit**

```bash
git add static/js/app.js
git commit -m "fix: drag-drop watermark stuck after drop — use enter/leave counter"
```

---

### Task 2: Fix Never-Ending Progress Bar

**Files:**
- Modify: `static/js/app.js:173-200` (finishProgress)
- Modify: `static/js/app.js:5983-6038` (uploadFileToFolder)
- Modify: `static/js/app.js:7075-7107` (knowledge lab bulk upload)
- Modify: `static/js/app.js:10757-10835` (watchTaskProgress — 30s timeout)

**Interfaces:**
- Modifies: `finishProgress(success, message)` — new behavior: green fade on true, red sticky+dismiss on false
- No signature changes to other functions

- [ ] **Step 1: Rewrite `finishProgress` with success/error differentiation**

Replace `finishProgress` function body (lines 190-200):

```js
function finishProgress(success, message) {
    const bar = _getProgressBar();
    const fill = _getProgressFill();
    const toast = _getProgressToast();
    if (bar && fill) {
        fill.style.width = '100%';
        fill.style.background = success ? '#16a34a' : '#ef4444';
        setTimeout(() => { bar.style.display = 'none'; fill.style.width = '0%'; fill.style.background = ''; }, success ? 600 : 0);
    }
    if (message && toast) {
        toast.textContent = (success ? '' : '❌ ') + message;
        toast.style.background = success ? '#16a34a' : '#ef4444';
        if (success) {
            setTimeout(() => { toast.style.display = 'none'; toast.style.background = '#1e293b'; }, 2000);
        } else {
            toast.style.cursor = 'pointer';
            toast.title = '点击关闭';
            const dismiss = () => { toast.style.display = 'none'; toast.style.background = '#1e293b'; toast.style.cursor = ''; toast.title = ''; toast.removeEventListener('click', dismiss); };
            toast.addEventListener('click', dismiss);
        }
    }
}
```

- [ ] **Step 2: Add `try/catch` + `finishProgress` to `uploadFileToFolder`**

Replace `uploadFileToFolder` (lines 5983-6038):

```js
async function uploadFileToFolder(projectId, folderId, file) {
    showProgress(`上传中: ${file.name}`, 'bar');
    updateProgress(10, `正在处理 ${file.name}...`);
    const formData = new FormData();
    formData.append('file', file);
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
    if (res.ok && data.success) {
        finishProgress(true);
        return { success: true };
    }
    if (data.duplicate) {
        finishProgress(false, '发现重复文件');
        return {
            conflict: true,
            conflict_type: data.conflict_type || 'hash',
            existing_file: data.existing_file,
            new_filename: data.new_filename,
            file: file
        };
    }
    finishProgress(false, data.error || '上传失败，请检查文件格式后重试');
    return null;
}
```

- [ ] **Step 3: Update batch upload caller to collect conflicts**

Replace the batch upload loop (lines 5865-5871):

```js
            document.getElementById('batchUploadInput').onchange = async (e) => {
                const uploadFiles = Array.from(e.target.files);
                const conflicts = [];
                for (const file of uploadFiles) {
                    const result = await uploadFileToFolder(projectId, folderId, file);
                    if (result && result.conflict) {
                        conflicts.push(result);
                    }
                }
                if (conflicts.length > 0) {
                    const decision = await showBatchConflictPanel(conflicts);
                    if (decision.applied) {
                        for (const r of decision.results) {
                            const c = conflicts[r.index];
                            if (r.action === 'replace') {
                                const vf = new FormData();
                                vf.append('file', c.file);
                                await fetch(`/admin/projects/${projectId}/files/${c.existing_file.id}/new_version`, {
                                    method: 'POST', credentials: 'include', body: vf
                                });
                            } else if (r.action === 'rename') {
                                await fetch(`/admin/projects/${projectId}/files/${c.existing_file.id}/rename`, {
                                    method: 'PUT',
                                    headers: { 'Content-Type': 'application/json' },
                                    credentials: 'include',
                                    body: JSON.stringify({ original_name: c.new_filename })
                                });
                            }
                            // 'keep' = do nothing
                        }
                    }
                }
                await loadFilesInFolder(projectId, folderId);
                await loadFolderTree(projectId, currentFolderId);
            };
```

- [ ] **Step 4: Add `finishProgress` error handling to knowledge lab bulk upload**

In the bulk upload loop (lines 7075-7107), change the `finishProgress(true)` at the end and add error tracking:

Replace lines 7075-7107:
```js
        let uploaded = 0;
        let hadErrors = false;
        showProgress(`上传 ${files.length} 个文件...`, 'bar');
        for (const item of placeholders) {
            const { file, placeholder } = item;
            const formData = new FormData();
            formData.append('file', file);
            try {
                updateProgress((uploaded / files.length) * 90, `上传中: ${file.name} (${uploaded+1}/${files.length})`);
                const res = await fetch('/knowledge_lab/upload', { method: 'POST', credentials: 'include', body: formData });
                const data = await res.json();
                if (res.ok && data.success) {
                    const finalEntry = createFinalEntry({
                        file_id: data.file_id,
                        filename: data.filename,
                        file_size: data.file_size,
                        uploaded_at: data.uploaded_at
                    });
                    placeholder.replaceWith(finalEntry);
                    showToast(`✅ ${data.filename} 上传成功`, 'success', 2000);
                } else {
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
```

- [ ] **Step 5: Add 30s idle timeout to `watchTaskProgress`**

In `watchTaskProgress` (line 10757), add a timeout. After `_activeTaskIds.add(taskId)` (line 10763), insert:

```js
        let idleTimer = setTimeout(() => {
            es.close();
            _activeTaskIds.delete(taskId);
            updateFloatingIndicator();
            if (_activeTaskIds.size === 0) {
                if (progBar) progBar.style.display = 'none';
                if (procInd) procInd.style.display = 'none';
            }
            finishProgress(false, '连接超时，任务可能仍在后台运行');
            loadBgTasks();
        }, 30000);
```

Then at the start of `es.onmessage` (line 10774): add `clearTimeout(idleTimer); idleTimer = setTimeout(() => { /* same timeout body */ }, 30000);`

In the `complete` path (before `es.close()` at line 10786): add `clearTimeout(idleTimer);`

In the `error` path (before `es.close()` at line 10802): add `clearTimeout(idleTimer);`

In `es.onerror` (line 10817): add `clearTimeout(idleTimer);`

- [ ] **Step 6: Verify progress bar behavior**

Manual tests:
1. Upload a file successfully → green flash + auto-hide (2s)
2. Upload while server is down → red sticky toast, click to dismiss
3. Upload duplicate file → red toast "发现重复文件", batch panel appears
4. SSE task completes normally → bar fills green, auto-hides
5. Kill server during SSE task → after 30s, red "连接超时" toast appears
6. Bulk knowledge lab upload, some files fail → progress bar ends red "部分文件上传失败"

- [ ] **Step 7: Commit**

```bash
git add static/js/app.js
git commit -m "fix: progress bar never ends — add missing finishProgress calls, error stickiness, and 30s SSE timeout"
```

---

### Task 3: Backend — Name-Conflict Detection + Content Compare Endpoint

**Files:**
- Modify: `app/routes/admin.py:926-940` (add name-conflict query)
- Modify: `app/routes/admin.py` (insert new endpoint after line 967)

**Interfaces:**
- Modifies: `POST /admin/projects/<id>/folders/<id>/upload` response — adds `conflict_type: "hash" | "name"`
- Produces: `POST /admin/projects/<id>/files/compare-content` — accepts `{file_id}`, returns `{name, text, size, version}` for one existing file

- [ ] **Step 1: Add name-conflict detection to upload endpoint**

Replace lines 926-940 (`app/routes/admin.py`):

```python
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check hash duplicate (identical content)
            cur.execute("SELECT id, original_name, stored_path, version, folder_id, file_size FROM project_files WHERE project_id = %s AND file_hash = %s", (project_id, file_hash))
            hash_dup = cur.fetchone()
            if hash_dup:
                return ok({
                    "duplicate": True,
                    "conflict_type": "hash",
                    "existing_file": {
                        "id": hash_dup['id'],
                        "original_name": hash_dup['original_name'],
                        "folder_id": hash_dup['folder_id'],
                        "version": hash_dup['version'],
                        "file_size": hash_dup['file_size']
                    },
                    "new_filename": original_name
                })
            # Check name duplicate (same name, different content)
            cur.execute("SELECT id, original_name, stored_path, version, folder_id, file_size, file_hash FROM project_files WHERE project_id = %s AND original_name = %s", (project_id, original_name))
            name_dup = cur.fetchone()
            if name_dup:
                return ok({
                    "duplicate": True,
                    "conflict_type": "name",
                    "existing_file": {
                        "id": name_dup['id'],
                        "original_name": name_dup['original_name'],
                        "folder_id": name_dup['folder_id'],
                        "version": name_dup['version'],
                        "file_size": name_dup['file_size'],
                        "file_hash": name_dup['file_hash']
                    },
                    "new_filename": original_name
                })
```

- [ ] **Step 2: Add content-compare endpoint (single-file, for existing file only)**

Insert after line 967 (before `new_file_version`):

```python
@admin_bp.route('/admin/projects/<int:project_id>/files/<int:file_id>/content', methods=['GET'])
def get_file_content(project_id, file_id):
    """Return extracted text content for a single project file. Used by conflict compare panel."""
    if session.get('consent_value', 0) != 1:
        return err("Consent not given", "FORBIDDEN", 403)
    user_id = session.get('user_id')
    if not is_admin() and not _can_access_project(project_id, user_id):
        return err("Access denied", "FORBIDDEN", 403)

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, original_name, content, file_size, version, stored_path
                FROM project_files
                WHERE id = %s AND project_id = %s
            """, (file_id, project_id))
            f = cur.fetchone()
            if not f:
                return err("File not found", "NOT_FOUND", 404)

            text = (f.get('content') or '').strip()
            if not text:
                stored = f.get('stored_path')
                if stored and os.path.exists(stored):
                    try:
                        with open(stored, 'rb') as fh:
                            fake = FileStorage(fh, filename=f['original_name'])
                            text, _ = extract_text_from_file(fake)
                            text = text or ''
                    except Exception:
                        text = ''
            if not text:
                text = '[无法提取文本内容]'

            return ok({
                "id": f['id'],
                "name": f['original_name'],
                "text": text,
                "size": f['file_size'],
                "version": f['version']
            })
```

- [ ] **Step 3: Verify backend**

```bash
# Start the app and test with curl or browser:
# 1. Upload a file twice → second upload returns duplicate:true with conflict_type:"hash"
# 2. Upload a different file with same name → returns duplicate:true with conflict_type:"name"
# 3. GET /admin/projects/<id>/files/<id>/content → returns text content
```

- [ ] **Step 4: Commit**

```bash
git add app/routes/admin.py
git commit -m "feat: add name-conflict detection and file content endpoint for duplicate handling"
```

---

### Task 4: Frontend — Batch Conflict Panel + "Yes to All" + Lazy Compare

**Files:**
- Modify: `static/js/app.js` (replace `showDuplicateFileOptions`, add `showBatchConflictPanel`)
- Modify: `static/css/app.css` (add conflict panel styles)

**Interfaces:**
- Consumes: `GET /admin/projects/<id>/files/<id>/content` (from Task 3)
- Produces: `showBatchConflictPanel(conflicts)` — conflicts is `[{conflict_type, existing_file, new_filename, file}]`, returns `{applied: bool, results: [{index, action}]}`
- Consumer: batch upload loop in `loadFolderContents` (caller updated in Task 2 Step 3)

- [ ] **Step 1: Add `showBatchConflictPanel` function**

Insert after the existing `showDuplicateFileOptions` function (after line 6063):

```js
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
                            <div class="conflict-card-label">📌 已存在</div>
                            <div class="conflict-card-name">${escapeHtml(ef.original_name)}</div>
                            <div class="conflict-card-meta">v${ef.version} · ${sizeFmt(ef.file_size)}</div>
                        </div>
                        <div class="conflict-card card-new">
                            <div class="conflict-card-label">🆕 新上传</div>
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
            // Use a project ID available in scope — read from the DOM or URL
            const projectId = getCurrentProjectId();
            const res = await fetch(`/admin/projects/${projectId}/files/${ef.id}/content`, {
                credentials: 'include'
            });
            if (res.ok) {
                const d = await res.json();
                existingText = d.data ? d.data.text : (d.text || '');
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
};
```

Note: `getCurrentProjectId()` must be available. Check how `projectId` is obtained in the calling context — it's a parameter in `loadFolderContents(projectId, folderId)`. Expose it as a module-level variable or read from `document.querySelector('[data-project-id]')`. The simplest fix: set `window._currentProjectId = projectId` in the calling function, and `getCurrentProjectId()` reads `window._currentProjectId`.

- [ ] **Step 2: Add `getCurrentProjectId` helper**

In the `loadFolderContents` function (which calls `uploadFileToFolder`), add at the top:

```js
window._currentProjectId = projectId;
```

And add the helper near the top of the file or near the conflict functions:

```js
function getCurrentProjectId() {
    return window._currentProjectId;
}
```

- [ ] **Step 3: Add CSS styles for conflict panel**

Append to `static/css/app.css`:

```css
/* ============================================================
   BATCH CONFLICT PANEL
   ============================================================ */
.conflict-panel { overflow-y: auto; }
.conflict-pairs-list { display: flex; flex-direction: column; gap: 12px; margin: 12px 0; }
.conflict-pair { border: 1px solid var(--card-border); border-radius: 8px; padding: 12px; background: var(--card-bg); }
.conflict-pair[data-action="keep"] { border-left: 3px solid #16a34a; }
.conflict-pair[data-action="replace"] { border-left: 3px solid #ef4444; }
.conflict-pair[data-action="rename"] { border-left: 3px solid #3b82f6; }
.conflict-pair-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.conflict-badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; font-weight: 600; }
.badge-hash { background: #fef3c7; color: #92400e; }
.badge-name { background: #dbeafe; color: #1e40af; }
.conflict-pair-name { font-size: 0.85rem; font-weight: 600; }
.conflict-cards { display: flex; gap: 12px; margin-bottom: 8px; }
.conflict-card { flex: 1; padding: 8px 10px; border-radius: 6px; border: 1px solid var(--card-border); }
.card-existing { background: #f8fafc; }
.card-new { background: #f0fdf4; }
.conflict-card-label { font-size: 0.65rem; color: var(--card-muted); margin-bottom: 2px; }
.conflict-card-name { font-size: 0.8rem; font-weight: 600; word-break: break-all; }
.conflict-card-meta { font-size: 0.7rem; color: var(--card-muted); margin-top: 2px; }
.conflict-actions { display: flex; gap: 6px; flex-wrap: wrap; }
.conflict-btn { padding: 4px 10px; border: 1px solid var(--card-border); border-radius: 6px; background: var(--card-bg); cursor: pointer; font-size: 0.75rem; }
.conflict-btn:hover { opacity: 0.8; }
.conflict-btn.active { border-color: #3b82f6; background: #eff6ff; color: #1d4ed8; }
.conflict-btn.btn-keep.active { border-color: #16a34a; background: #f0fdf4; color: #15803d; }
.conflict-btn.btn-replace.active { border-color: #ef4444; background: #fef2f2; color: #b91c1c; }
.conflict-btn.btn-rename.active { border-color: #3b82f6; background: #eff6ff; color: #1d4ed8; }
.conflict-bulk-actions { display: flex; gap: 8px; margin-bottom: 12px; }
.conflict-compare-view { margin-top: 8px; }
.compare-loading { font-size: 0.75rem; color: var(--card-muted); padding: 8px; }
.compare-side-by-side { display: flex; gap: 8px; max-height: 300px; }
.compare-pane { flex: 1; border: 1px solid var(--card-border); border-radius: 6px; overflow: hidden; display: flex; flex-direction: column; }
.compare-pane-header { font-size: 0.7rem; font-weight: 600; padding: 4px 8px; background: var(--card-bg); border-bottom: 1px solid var(--card-border); }
.compare-content { flex: 1; overflow-y: auto; padding: 8px; font-size: 0.7rem; line-height: 1.4; white-space: pre-wrap; word-break: break-all; margin: 0; }
.compare-error { color: #ef4444; font-size: 0.75rem; padding: 8px; }
```

- [ ] **Step 4: Verify end-to-end**

Manual tests:
1. Upload 3 files where 2 are duplicates (same hash) → batch panel shows 2 conflicts, non-conflict file uploaded
2. Click "保留所有已有" → all pairs set to keep, green left border
3. Click "替换所有为新" → all pairs set to replace, red left border
4. Override one pair individually → that pair changes action, others stay
5. Click "应用选择" → keep skips, replace uploads new version, rename renames
6. Upload a file with same name but different content → "同名文件" badge + "比较内容" button
7. Click "比较内容" on a text file pair → side-by-side text appears
8. Click "比较内容" on a binary file pair → shows "无法在浏览器中预览" message
9. Click "取消全部上传" → no files uploaded, panel closes

- [ ] **Step 5: Commit**

```bash
git add static/js/app.js static/css/app.css
git commit -m "feat: batch conflict panel with yes-to-all, per-pair actions, and lazy content compare"
```
