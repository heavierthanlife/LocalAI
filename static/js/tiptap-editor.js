// ── Compliance Tiptap Editor (U9b) ──
// ESM module — loads Tiptap v2 from esm.sh CDN
// Provides inline compliance annotations + incremental checking

let _editorInstance = null;
let _complianceMarks = {};

// ── Dynamic import from CDN ──
async function _loadTiptap() {
    const [{ Editor }, { StarterKit }] = await Promise.all([
        import('https://esm.sh/@tiptap/core@2.11.5'),
        import('https://esm.sh/@tiptap/starter-kit@2.11.5'),
    ]);
    return { Editor, StarterKit };
}

// ── Custom compliance highlight extension ──
function _createComplianceHighlights() {
    const { Node, Mark } = window._tiptapCore || {};
    if (!Mark) return null;

    return Mark.create({
        name: 'complianceFindings',
        addAttributes() {
            return {
                level: { default: 'violation', rendered: false },
                message: { default: '', rendered: false },
            };
        },
        parseHTML() {
            return [{ tag: 'mark[data-compliance]' }];
        },
        renderHTML({ HTMLAttributes }) {
            const levelColors = {
                critical: '#f5c6cb',
                violation: '#ffe0b2',
                warning: '#fff3cd',
                pass: '#d4edda',
            };
            const level = HTMLAttributes.level || 'violation';
            const color = levelColors[level] || '#f0f0f0';
            return [
                'mark',
                {
                    'data-compliance': level,
                    'data-message': HTMLAttributes.message || '',
                    style: `background-color:${color};border-bottom:2px solid ${color};padding:0 1px;border-radius:1px;`,
                },
                0,
            ];
        },
    });
}

// ── Initialization ──
async function initEditor() {
    const editorEl = document.getElementById('complianceTiptapEditor');
    if (!editorEl || editorEl.__tiptapReady) return;

    try {
        const { Editor, StarterKit } = await _loadTiptap();
        window._tiptapCore = { Editor, StarterKit, Node: Editor.Node || {}, Mark: Editor.Mark || {} };

        const ComplianceFindings = _createComplianceHighlights();

        const extensions = [StarterKit.configure({ heading: { levels: [1, 2, 3] } })];
        if (ComplianceFindings) extensions.push(ComplianceFindings);

        _editorInstance = new Editor({
            element: editorEl,
            extensions: extensions,
            content: '',
            editorProps: {
                attributes: {
                    class: 'compliance-editor-content',
                    spellcheck: 'true',
                },
            },
            onUpdate({ editor }) {
                _onEditorChange(editor);
            },
        });

        editorEl.__tiptapReady = true;
        _initToolbar();
        _updateScoreIndicator();
    } catch (e) {
        console.error('Tiptap editor load failed:', e);
        editorEl.innerHTML = '<span style="color:#e74c3c;font-size:0.7rem;">⚠️ 编辑器加载失败，Tiptap CDN 可能不可用。刷新重试或使用文件上传模式。</span>';
        editorEl.style.padding = '12px';
    }
}

// ── Toolbar ──
function _initToolbar() {
    const panel = document.getElementById('complianceEditorPanel');
    const existingToolbar = document.getElementById('complianceEditorToolbar');
    if (existingToolbar) existingToolbar.remove();

    const toolbar = document.createElement('div');
    toolbar.id = 'complianceEditorToolbar';
    toolbar.style.cssText = 'display:flex;gap:3px;margin-bottom:4px;flex-wrap:wrap;padding:4px 0;border-bottom:1px solid var(--card-border);';
    toolbar.innerHTML = `
        <button data-cmd="bold" title="粗体" style="padding:2px 8px;font-size:0.65rem;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">B</button>
        <button data-cmd="italic" title="斜体" style="padding:2px 8px;font-size:0.65rem;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;font-style:italic;">I</button>
        <span style="width:1px;background:var(--card-border);margin:2px 2px;"></span>
        <button data-cmd="heading1" title="标题1" style="padding:2px 8px;font-size:0.65rem;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">H1</button>
        <button data-cmd="heading2" title="标题2" style="padding:2px 8px;font-size:0.65rem;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">H2</button>
        <span style="width:1px;background:var(--card-border);margin:2px 2px;"></span>
        <button data-cmd="bulletList" title="无序列表" style="padding:2px 8px;font-size:0.65rem;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">• List</button>
        <button data-cmd="orderedList" title="有序列表" style="padding:2px 8px;font-size:0.65rem;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">1. List</button>
        <span style="flex:1;"></span>
        <button data-cmd="clearMarks" title="清除标注" style="padding:2px 8px;font-size:0.65rem;border:1px solid var(--card-border);border-radius:3px;background:#fdf0f0;color:#e74c3c;cursor:pointer;">清除标注</button>
        <button data-cmd="clearContent" title="清空内容" style="padding:2px 8px;font-size:0.65rem;border:1px solid var(--card-border);border-radius:3px;background:var(--card-bg);cursor:pointer;">清空</button>
    `;
    panel.insertBefore(toolbar, panel.firstChild);

    toolbar.querySelectorAll('button[data-cmd]').forEach(btn => {
        btn.addEventListener('mousedown', e => {
            e.preventDefault();
            const cmd = btn.dataset.cmd;
            if (!_editorInstance) return;

            if (cmd === 'clearMarks') {
                _clearComplianceMarks();
            } else if (cmd === 'clearContent') {
                if (confirm('确认清空编辑器内容？')) {
                    _editorInstance.commands.clearContent();
                    _updateScoreIndicator();
                }
            } else if (cmd === 'bold') {
                _editorInstance.chain().focus().toggleBold().run();
            } else if (cmd === 'italic') {
                _editorInstance.chain().focus().toggleItalic().run();
            } else if (cmd === 'heading1') {
                _editorInstance.chain().focus().toggleHeading({ level: 1 }).run();
            } else if (cmd === 'heading2') {
                _editorInstance.chain().focus().toggleHeading({ level: 2 }).run();
            } else if (cmd === 'bulletList') {
                _editorInstance.chain().focus().toggleBulletList().run();
            } else if (cmd === 'orderedList') {
                _editorInstance.chain().focus().toggleOrderedList().run();
            }
        });
    });
}

// ── Editor onChange → Incremental Check ──
let _debounceTimer = null;
let _lastText = '';

function _onEditorChange(editor) {
    if (!editor) return;
    const text = editor.getText();
    if (text === _lastText) return;
    _lastText = text;

    if (!text.trim()) {
        _updateScoreIndicator();
        return;
    }

    // Composition-aware (IME pending)
    if (editor.view && editor.view.composing) return;

    clearTimeout(_debounceTimer);
    _debounceTimer = setTimeout(() => {
        _runIncrementalCheck(editor);
    }, 300);
}

function _runIncrementalCheck(editor) {
    const text = editor.getText();
    if (!text.trim()) return;

    const rulesTaskId = window.Compliance?._taskIds?.extracted;
    const bidDocName = document.getElementById('complianceBidDocName')?.textContent || 'editor-edit';

    if (!rulesTaskId) return;

    // Split text into chapters for per-section checking
    const sections = _splitSections(text);

    fetch('/compliance/incremental_check', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            rules_task_id: rulesTaskId,
            bid_doc_name: bidDocName,
            changed_sections: sections,
            use_ai: true,
        }),
    })
        .then(r => r.json())
        .then(data => {
            if (!data.success) return;
            const results = data.results || [];
            const summary = data.summary || {};
            _applyFindingsToEditor(editor, results, summary);
        })
        .catch(e => console.error('Incremental check error:', e));
}

// ── Split text into chapters (for per-section checking) ──
function _splitSections(text) {
    const parts = text.split(/\n{2,}/);
    const sections = [];
    let idx = 0;
    for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed) continue;
        const lines = trimmed.split('\n');
        const title = lines[0].length > 50 ? lines[0].substring(0, 50) + '...' : lines[0];
        sections.push({
            id: 'sec-' + (idx++),
            title: title,
            content: trimmed,
        });
    }
    return sections.length > 0 ? sections : [{ id: 'sec-0', title: '全部内容', content: text }];
}

// ── Apply findings to editor as inline marks ──
function _applyFindingsToEditor(editor, results, summary) {
    if (!editor) return;

    // Clear existing marks
    _clearComplianceMarks();

    const findings = [];
    for (const ev of results) {
        if (ev.results) {
            for (const f of ev.results) {
                findings.push({
                    ...f,
                    section_id: ev.section_id,
                    section_title: ev.section_title,
                });
            }
        }
    }

    if (!findings.length) {
        _updateScoreIndicator();
        return;
    }

    const fullText = editor.getText();
    _complianceMarks = {};

    for (const f of findings) {
        const keyword = f.keyword || f.field_name || f.function_name || f.clue || '';
        if (!keyword) continue;

        // Find keyword position in editor
        const idx = fullText.indexOf(keyword);
        if (idx < 0) continue;

        try {
            const from = idx;
            const to = idx + keyword.length;
            const level = f.status || f.level || 'violation';

            editor.chain()
                .setTextSelection({ from, to })
                .setMark('complianceFindings', {
                    level: level,
                    message: f.description || f.message || f.detail || '',
                })
                .run();

            _complianceMarks[keyword] = { level, from, to, ...f };
        } catch (e) {
            // Skip annotation if position is invalid
        }
    }

    _updateScoreIndicator(summary);
    _renderFindingsList(findings);
}

function _clearComplianceMarks() {
    if (!_editorInstance) return;
    _complianceMarks = {};
    try {
        const doc = _editorInstance.state.doc;
        _editorInstance.chain()
            .setTextSelection({ from: 0, to: doc.content.size })
            .unsetMark('complianceFindings')
            .run();
    } catch (e) {
        // Reset on error
    }
    _updateScoreIndicator();
    const findingsEl = document.getElementById('complianceEditorFindings');
    if (findingsEl) findingsEl.innerHTML = '';
}

// ── Score indicator ──
function _updateScoreIndicator(summary) {
    const scoreEl = document.getElementById('complianceEditorScore');
    if (!scoreEl) return;

    if (!summary) {
        scoreEl.innerHTML = '';
        return;
    }

    const total = (summary.pass || 0) + (summary.warning || 0) + (summary.violation || 0) + (summary.critical || 0);
    if (!total) {
        scoreEl.innerHTML = '<span style="color:var(--card-muted);">✅ 无违规</span>';
        return;
    }

    const score = Math.round(
        ((summary.pass || 0) * 100 + (summary.warning || 0) * 60) / total
    );
    const scoreColor = score >= 80 ? '#27ae60' : score >= 50 ? '#f39c12' : '#e74c3c';
    scoreEl.innerHTML =
        '<span style="color:' + scoreColor + ';font-weight:600;">合规度 ' + score + '%</span>' +
        ' <span style="color:var(--card-muted);font-size:0.6rem;">' +
        ' ✅' + (summary.pass || 0) +
        ' ⚠️' + (summary.warning || 0) +
        ' ❌' + (summary.violation || 0) +
        ' 🚫' + (summary.critical || 0) +
        '</span>';
}

// ── Findings list below editor ──
function _renderFindingsList(findings) {
    const el = document.getElementById('complianceEditorFindings');
    if (!el) return;

    const top = findings.slice(0, 8);
    let html = '<div style="border-top:1px solid var(--card-border);padding-top:4px;">';
    html += '<span style="font-weight:600;font-size:0.62rem;">' +
        (findings.length > 8 ? '最近 ' + top.length + '/' + findings.length + ' 条发现' : '发现 ' + findings.length + ' 条') +
        '</span>';
    html += '<div style="margin-top:2px;">';
    top.forEach(f => {
        const sc = f.status === 'critical' ? '#e74c3c' : f.status === 'violation' ? '#e67e22' : f.status === 'warning' ? '#f39c12' : '#27ae60';
        const kw = f.keyword || f.field_name || f.function_name || f.clue || '';
        const desc = f.description || f.message || f.detail || '';
        html += '<div style="display:flex;gap:4px;padding:1px 0;font-size:0.6rem;">';
        html += '<span style="font-weight:600;color:' + sc + ';min-width:28px;">' + f.status + '</span>';
        html += '<span style="flex:1;">' + _esc(kw) + (desc ? ': ' + _esc(desc) : '') + '</span>';
        html += '</div>';
    });
    html += '</div></div>';
    el.innerHTML = html;
}

function _esc(s) {
    return (s || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Public API ──
window.ComplianceTiptap = {
    init: initEditor,
    getEditor: () => _editorInstance,
    getText: () => _editorInstance ? _editorInstance.getText() : '',
    getHTML: () => _editorInstance ? _editorInstance.getHTML() : '',
    clearMarks: _clearComplianceMarks,
    destroy: () => {
        if (_editorInstance) {
            _editorInstance.destroy();
            _editorInstance = null;
        }
    },
};

// ── DOM ready → watch for text mode button ──
document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('complianceTextModeBtn');
    const panel = document.getElementById('complianceEditorPanel');
    if (!btn || !panel) return;

    btn.addEventListener('click', async () => {
        const isVisible = panel.style.display === 'block';
        if (isVisible) {
            panel.style.display = 'none';
            btn.textContent = '✏️ 文字编辑';
            btn.style.background = '#2980b9';
            return;
        }

        panel.style.display = 'block';
        btn.textContent = '📄 文件模式';
        btn.style.background = '#e67e22';

        await initEditor();
    });
});
