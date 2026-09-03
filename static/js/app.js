/* AI_Services Application Logic */
// ======================== Global Variables & Setup ========================
    let selectedFiles = [];
    let currentFeedbackState = new Map();
    let isProcessing = false;
    let persistentReminderDiv = null;
    let isLoadingSession = false;
    let currentBatchAbortController = null;
    let activeStreamController = null;
    let currentProjectId = null;
    let _isCurrentSessionProjectChat = false;
    let _isCurrentSessionGrill = false;
    let currentFolderId = null;
    let selectedFileIdsForBatch = new Set();
    let currentUserRoleInProject = null;
    let toastQueue = {};
    let toastTimeout = null;
    let recycleBinTabBtn = document.getElementById('recycleBinTabBtn');
    let currentTable = '';
    let currentPage = 1;
    let currentPerPage = 50;
    let currentSearch = '';
    let currentSearchColumn = '';
    let autoRefreshInterval = null;
    let autoRefreshSeconds = 300;
    let autoRefreshEnabled = true;

    const chatInterface = document.getElementById('chatInterface');
    const adminPanel = document.getElementById('adminPanel');
    const recycleBinPanel = document.getElementById('recycleBinPanel');
    const databasePanel = document.getElementById('databasePanel');
    const knowledgeLabPanel = document.getElementById('knowledgeLabPanel');
    const chatTab = document.getElementById('chatTabBtn');
    const adminTab = document.getElementById('adminTabBtn');
    const recycleBinTab = document.getElementById('recycleBinTabBtn');
    const databaseTab = document.getElementById('databaseTabBtn');
    const knowledgeLabTab = document.getElementById('knowledgeLabTabBtn');
    const wikiPanel = document.getElementById('wikiPanel');
    const wikiTab = document.getElementById('wikiTabBtn');
    const templatesTab = document.getElementById('templatesTabBtn');

    // Get CSRF token from meta tag
    const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');

    const md = window.markdownit({
        html: false,
        breaks: true,
        linkify: true,
        typographer: true,
        highlight: function (str, lang) {
            if (lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(str, { language: lang }).value;
                } catch (__) {}
            }
            return '';
        }
    });
    const defaultLinkRender = md.renderer.rules.link_open || function(tokens, idx, options, env, self) {
        return self.renderToken(tokens, idx, options, env, self);
    };
    md.renderer.rules.link_open = function(tokens, idx, options, env, self) {
        const aIndex = tokens[idx].attrIndex('target');
        if (aIndex < 0) tokens[idx].attrPush(['target', '_blank']);
        else tokens[idx].attrs[aIndex][1] = '_blank';
        const relIndex = tokens[idx].attrIndex('rel');
        if (relIndex < 0) tokens[idx].attrPush(['rel', 'noopener noreferrer']);
        else tokens[idx].attrs[relIndex][1] = 'noopener noreferrer';
        return defaultLinkRender(tokens, idx, options, env, self);
    };

    // Custom modal replacements for alert, confirm, prompt
    window.alert = function(message) {
        return new Promise((resolve) => {
            const modal = document.createElement('div');
            modal.className = 'custom-modal-overlay';
            modal.innerHTML = `
                <div class="custom-modal">
                    <p>${escapeHtml(message)}</p>
                    <div class="custom-modal-buttons">
                        <button class="confirm" id="alertOk">确定</button>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
            modal.querySelector('#alertOk').onclick = () => {
                modal.remove();
                resolve();
            };
        });
    };

    window.confirm = function(message) {
        return new Promise((resolve) => {
            const modal = document.createElement('div');
            modal.className = 'custom-modal-overlay';
            modal.innerHTML = `
                <div class="custom-modal">
                    <p>${escapeHtml(message)}</p>
                    <div class="custom-modal-buttons">
                        <button class="confirm" id="confirmYes">确定</button>
                        <button class="cancel" id="confirmNo">取消</button>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
            modal.querySelector('#confirmYes').onclick = () => {
                modal.remove();
                resolve(true);
            };
            modal.querySelector('#confirmNo').onclick = () => {
                modal.remove();
                resolve(false);
            };
        });
    };

    window.prompt = function(message, defaultValue = '') {
        return new Promise((resolve) => {
            const modal = document.createElement('div');
            modal.className = 'custom-modal-overlay';
            modal.innerHTML = `
                <div class="custom-modal">
                    <p>${escapeHtml(message)}</p>
                    <input type="text" id="promptInput" value="${escapeHtml(defaultValue)}">
                    <div class="custom-modal-buttons">
                        <button class="confirm" id="promptOk">确定</button>
                        <button class="cancel" id="promptCancel">取消</button>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
            const input = modal.querySelector('#promptInput');
            input.focus();
            modal.querySelector('#promptOk').onclick = () => {
                const val = input.value;
                modal.remove();
                resolve(val);
            };
            modal.querySelector('#promptCancel').onclick = () => {
                modal.remove();
                resolve(null);
            };
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') modal.querySelector('#promptOk').click();
            });
        });
    };

    function fixLinksInContainer(container) {
        if (!container) return;
        container.querySelectorAll('a').forEach(link => {
            if (!link.getAttribute('target')) {
                link.setAttribute('target', '_blank');
                link.setAttribute('rel', 'noopener noreferrer');
            }
        });
    }

    const messagesDiv = document.getElementById('chatMessages');
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === Node.ELEMENT_NODE) fixLinksInContainer(node);
            });
        });
    });
    observer.observe(messagesDiv, { childList: true, subtree: true });

    // ======================== Unified Progress System ========================
    // Look up elements lazily (DOM might not be ready at script parse time)
    function _getProgressBar() { return document.getElementById('progressBar'); }
    function _getProgressFill() { return document.getElementById('progressBarFill'); }
    function _getProgressToast() { return document.getElementById('progressToast'); }

    function showProgress(label, type = 'bar') {
        const bar = _getProgressBar();
        const fill = _getProgressFill();
        const toast = _getProgressToast();
        if (type === 'bar' && bar) { bar.style.display = ''; if (fill) fill.style.width = '0%'; }
        if (toast) { toast.style.display = ''; toast.textContent = label; toast.style.background = '#1e293b'; }
    }

    function updateProgress(pct, label) {
        const bar = _getProgressBar();
        const fill = _getProgressFill();
        const toast = _getProgressToast();
        if (fill && bar && bar.style.display !== 'none')
            fill.style.width = Math.min(100, Math.max(0, pct || 0)) + '%';
        if (label && toast) toast.textContent = label;
    }

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

    function showToast(message, type = 'info', duration = 3000) {
        const key = `${type}:${message}`;
        if (toastQueue[key]) {
            clearTimeout(toastQueue[key].timeout);
            toastQueue[key].count++;
            toastQueue[key].element.querySelector('.toast-count').textContent = ` (${toastQueue[key].count})`;
            toastQueue[key].timeout = setTimeout(() => {
                removeToast(toastQueue[key].element, key);
            }, duration);
            return;
        }
        const container = document.getElementById('toast-container') || (() => {
            const div = document.createElement('div');
            div.id = 'toast-container';
            div.className = 'toast-container';
            document.body.appendChild(div);
            return div;
        })();
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `${escapeHtml(message)}<span class="toast-count" style="margin-left:8px;"></span>`;
        container.appendChild(toast);
        toastQueue[key] = {
            element: toast,
            count: 1,
            timeout: setTimeout(() => {
                removeToast(toast, key);
            }, duration)
        };
        const countSpan = toast.querySelector('.toast-count');
        if (countSpan) countSpan.textContent = '';
    }

    function removeToast(toast, key) {
        toast.remove();
        delete toastQueue[key];
        if (Object.keys(toastQueue).length === 0) {
            const container = document.getElementById('toast-container');
            if (container && container.children.length === 0) container.remove();
        }
    }

    // ======================== Smart Scroll System ========================
    let _userHasScrolled = false;  // true when user manually scrolls away from bottom
    let _autoScrollLocked = false; // temp lock during programmatic scroll

    function _getChatMessages() { return document.getElementById('chatMessages'); }

    // Track user scroll: if they scroll up, we stop auto-following
    // When user scrolls to bottom, auto mark project chat as read
    let _lastMarkReadTs = 0;
    let pinnedSessions = null;  // hoisted — initialized at ~line 9380
    document.addEventListener('DOMContentLoaded', () => {
        const mc = _getChatMessages();
        if (mc) {
            mc.addEventListener('scroll', () => {
                if (_autoScrollLocked) return;
                const atBottom = mc.scrollHeight - mc.scrollTop - mc.clientHeight < 100;
                _userHasScrolled = !atBottom;
                // Auto mark-read when user scrolls to bottom of project chat
                if (atBottom && currentProjectId) {
                    const now = Date.now();
                    if (now - _lastMarkReadTs > 3000) {  // throttle: max once per 3s
                        _lastMarkReadTs = now;
                        fetch(`/admin/projects/${currentProjectId}/mark_read`, { method: 'POST', credentials: 'include' }).then(() => {
                            loadHistoryList();
                        }).catch(() => {});
                    }
                }
            }, { passive: true });
        }
    });

    function scrollToBottom(force = false) {
        const mc = _getChatMessages();
        if (!mc) return;
        const atBottom = mc.scrollHeight - mc.scrollTop - mc.clientHeight < 200;
        if (force || (!_userHasScrolled && atBottom)) {
            _autoScrollLocked = true;
            fastSmoothScroll(mc, mc.scrollHeight, 200);
            setTimeout(() => { _autoScrollLocked = false; }, 250);
        }
    }

    // Floating scroll buttons
    setTimeout(() => {
        const scrollTopBtn = document.getElementById('scrollTopBtn');
        const scrollBottomBtn = document.getElementById('scrollBottomBtn');
        const mc = _getChatMessages();
        if (scrollTopBtn && mc) scrollTopBtn.onclick = () => { _userHasScrolled = false; fastSmoothScroll(mc, 0, 200); };
        if (scrollBottomBtn && mc) scrollBottomBtn.onclick = () => { _userHasScrolled = false; fastSmoothScroll(mc, mc.scrollHeight, 200); };
    }, 100);

    function fastSmoothScroll(element, targetTop, duration = 300) {
        const startTop = element.scrollTop;
        const distance = targetTop - startTop;
        if (Math.abs(distance) < 1) return;
        const startTime = performance.now();
        function step(now) {
            const elapsed = now - startTime;
            const progress = Math.min(1, elapsed / duration);
            const ease = progress < 0.5
                ? 4 * progress * progress * progress
                : 1 - Math.pow(-2 * progress + 2, 3) / 2;
            element.scrollTop = startTop + distance * ease;
            if (progress < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }
    function toggleThinking(headerElement) {
        const container = headerElement.closest('.thinking-container');
        const contentDiv = container.querySelector('.thinking-content');
        const arrow = headerElement.querySelector('.arrow');
        if (contentDiv.classList.contains('show')) {
            contentDiv.classList.remove('show');
            _toggleArrow(arrow, true);
        } else {
            contentDiv.classList.add('show');
            _toggleArrow(arrow, false);
        }
    }
    function formatElapsedTime(seconds) {
        if (seconds < 60) return `${seconds.toFixed(1)}s`;
        let minutes = Math.floor(seconds / 60);
        let remainingSecs = (seconds % 60).toFixed(1);
        return `${minutes}m ${remainingSecs}s`;
    }
    async function safeFetchJson(url, options = {}) {
        const res = await fetch(url, options);
        if (!res.ok) {
            const errText = await res.text().catch(() => '');
            let errMsg = `请求失败 (${res.status})`;
            try { const j = JSON.parse(errText); if (j.error) errMsg = j.error; } catch (_) {}
            throw new Error(errMsg);
        }
        return await res.json();
    }

    function escapeHtml(str) {
        if (str == null) return '';
        return String(str).replace(/[&<>]/g, function(m) {
            if (m === '&') return '&amp;';
            if (m === '<') return '&lt;';
            if (m === '>') return '&gt;';
            return m;
        });
    }

    window.showSkillFeedback = function(containerEl, data, source, fileId) {
        const preview = escapeHtml(data.skill_content || '').slice(0, 300);
        const hasMore = (data.skill_content || '').length > 300;
        containerEl.innerHTML = `<div style="margin:4px 0;padding:6px;background:var(--bg-color);border-radius:6px;border:1px solid var(--card-border);">
            <div style="font-size:.7rem;color:var(--card-muted);margin-bottom:4px;">🧠 技能提取结果</div>
            <div class="skill-preview" style="font-size:.75rem;line-height:1.5;max-height:80px;overflow:hidden;cursor:pointer;"
                 onclick="if(this.style.maxHeight!=='none'){this.style.maxHeight='none';this.style.cursor='default';if(this.nextSibling)this.nextSibling.style.display='none';}">
                ${preview}${hasMore ? '<span style="color:var(--accent);font-size:.65rem;"> 点击展开</span>' : ''}
            </div>
            <div style="margin-top:6px;display:flex;gap:6px;">
                <button class="fb-btn" onclick="window.submitSkillFeedback(${fileId},'${source}',1,this)">👍 满意</button>
                <button class="fb-btn" onclick="window.submitSkillFeedback(${fileId},'${source}',-1,this)">👎 不满意</button>
            </div>
        </div>`;
    };

    window.submitSkillFeedback = function(fileId, source, rating, btn) {
        const container = btn.closest('[data-feedback-skill]') || btn.parentElement.parentElement;
        fetch('/knowledge_lab/feedback', {
            method: 'POST', credentials: 'include',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({file_id: fileId, source: source, rating: rating})
        }).then(r => r.json()).then(function(d) {
            if (d.success) {
                container.innerHTML = '<span style="color:#22c55e;font-size:.7rem;">✅ 已反馈</span>';
                showToast('感谢反馈!', 'success');
            }
        }).catch(function(){});
    };

    window.submitIngestFeedback = function(taskId, rating, btn) {
        const container = btn.parentElement;
        fetch('/admin/ingest/feedback', {
            method: 'POST', credentials: 'include',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({task_id: taskId, rating: rating})
        }).then(r => r.json()).then(function(d) {
            if (d.success) {
                container.innerHTML = '<span style="color:#22c55e;font-size:.7rem;">✅ 已反馈</span>';
                showToast('感谢反馈!', 'success');
            }
        }).catch(function(){});
    };

    window.submitAuditFeedback = function(rating, btn) {
        const container = btn.parentElement;
        const auditRunId = container.closest('[data-audit-run]')?.getAttribute('data-audit-run') || '';
        fetch('/admin/skill_audit/feedback', {
            method: 'POST', credentials: 'include',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({audit_run_id: auditRunId, rating: rating})
        }).then(r => r.json()).then(function(d) {
            if (d.success) {
                container.innerHTML = '<div style="margin-top:8px;font-size:0.7rem;color:#22c55e;">✅ 已反馈</div>';
                showToast('感谢反馈!', 'success');
            }
        }).catch(function(){});
    };

    function fallbackCopy(text) {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed'; ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch(e) {}
        document.body.removeChild(ta);
    }

    function addCopyButton(wrapperEl, rawText) {
        if (!wrapperEl || wrapperEl.querySelector('.copy-btn')) return;
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.textContent = '📋';
        btn.title = '复制回答';
        btn.onclick = async function() {
            const txt = rawText || wrapperEl.textContent.trim();
            if (!navigator.clipboard) { fallbackCopy(txt); }
            else try {
                await navigator.clipboard.writeText(txt);
            } catch (e) { fallbackCopy(txt); }
            btn.textContent = '✓ 已复制';
            btn.classList.add('copied');
            setTimeout(() => { btn.textContent = '📋'; btn.classList.remove('copied'); }, 2000);
        };
        wrapperEl.appendChild(btn);
    }

    function addBranchButton(wrapperEl, groupEl) {
        if (!wrapperEl || !groupEl || wrapperEl.querySelector('.branch-btn')) return;
        const userMsg = groupEl.dataset.userMsg;
        if (!userMsg) return;
        const btn = document.createElement('button');
        btn.className = 'branch-btn';
        btn.textContent = '🔄';
        btn.title = '重新生成回答';
        btn.style.cssText = 'position:absolute;top:6px;right:44px;z-index:10;background:rgba(0,0,0,0.06);border:none;border-radius:4px;padding:3px 8px;font-size:0.75rem;cursor:pointer;opacity:0;transition:opacity 0.2s;';
        btn.onmouseenter = () => btn.style.opacity = '1';
        btn.onmouseleave = () => { if (!btn.matches(':hover')) btn.style.opacity = '0'; };
        btn.onclick = async function() {
            if (isProcessing) return;
            messageInput.value = userMsg;
            showToast('消息已载入，按 Enter 重新发送', 'info');
            messageInput.focus();
        };
        // Show on wrapper hover
        wrapperEl.addEventListener('mouseenter', () => btn.style.opacity = '1');
        wrapperEl.addEventListener('mouseleave', () => btn.style.opacity = '0');
        wrapperEl.appendChild(btn);
    }

    function asciiTableToMarkdown(text) {
        const lines = text.split('\n');
        let inAsciiTable = false;
        let markdownRows = [];
        for (let line of lines) {
            if (line.match(/^[┌├└]/) || line.includes('─┬─') || line.includes('─┼─')) {
                inAsciiTable = true;
                continue;
            }
            if (inAsciiTable && line.includes('│')) {
                const cells = line.split('│').filter(cell => cell.trim().length > 0).map(cell => cell.trim());
                if (cells.length > 0) markdownRows.push(cells);
            } else if (inAsciiTable && line.trim() === '') {
                inAsciiTable = false;
            }
        }
        if (markdownRows.length === 0) return text;
        let mdTable = '| ' + markdownRows[0].join(' | ') + ' |\n';
        mdTable += '|' + ' --- |'.repeat(markdownRows[0].length) + '\n';
        for (let i = 1; i < markdownRows.length; i++) {
            mdTable += '| ' + markdownRows[i].join(' | ') + ' |\n';
        }
        const asciiPattern = /┌[─┬┐]*┐[\s\S]*?└[─┴┘]*┘/;
        return text.replace(asciiPattern, mdTable);
    }
    function addUserMessage(content, messageId = null) {
        const group = document.createElement('div');
        group.className = 'message-group';
        if (messageId) {
            group.id = `msg-${messageId}`;
            group.dataset.msgId = messageId;
        }

        // Detect quote prefix: "--- 引用 @name ---\n[content]\n--- 追问 ---\n[query]"
        const quoteMatch = content.match(/^--- 引用( @\S+)? ---[\r\n]+([\s\S]*?)[\r\n]+--- 追问 ---[\r\n]+([\s\S]*)$/);
        if (quoteMatch) {
            const author = (quoteMatch[1] || '').trim();
            const quoteContent = quoteMatch[2];
            const queryText = quoteMatch[3];
            const lineCount = (quoteContent.match(/\n/g) || []).length + 1;
            const charCount = quoteContent.length;

            // Quote bubble (collapsed, right-aligned)
            const bubble = document.createElement('div');
            bubble.className = 'inline-quote-bubble';
            bubble.style.cssText = 'margin-bottom:2px;padding:4px 10px;border-radius:6px;background:#eff6ff;border:1px solid #bfdbfe;font-size:0.7rem;cursor:pointer;';
            const summary = document.createElement('div');
            summary.style.cssText = 'display:flex;align-items:center;gap:4px;color:#3b82f6;';
            summary.innerHTML = '<span class="inline-quote-icon msi msi-arrow collapsed">expand_more</span><span>' + ('引用' + author + ' — ' + lineCount + '行, ' + charCount + '字').replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</span>';
            const quoteBody = document.createElement('div');
            quoteBody.className = 'inline-quote-content';
            quoteBody.style.cssText = 'display:none;margin-top:4px;padding:4px 8px;background:white;border-radius:4px;border:1px solid #e5e7eb;max-height:200px;overflow-y:auto;white-space:pre-wrap;font-size:0.68rem;color:#374151;font-family:monospace;';
            quoteBody.textContent = quoteContent;
            bubble.appendChild(summary);
            bubble.appendChild(quoteBody);
            bubble.addEventListener('click', function(e) {
                const icon = this.querySelector('.inline-quote-icon');
                const body = this.querySelector('.inline-quote-content');
                if (body.style.display === 'none') { body.style.display = 'block'; _toggleArrow(icon, false); }
                else { body.style.display = 'none'; _toggleArrow(icon, true); }
            });

            // Name tag (right-aligned) — extract from content or use session
            const nameTag = document.createElement('div');
            nameTag.className = 'user-name-tag';
            const nameMatch = queryText.match(/^@(\S+?):\s/);
            const username = nameMatch ? nameMatch[1] : (sessionStorage.getItem('username') || '');
            nameTag.textContent = username ? '@' + username : '';

            // User's actual question
            const userDiv = document.createElement('div');
            userDiv.className = 'user-message';
            userDiv.textContent = queryText;

            if (nameTag.textContent) group.appendChild(nameTag);
            group.appendChild(bubble);
            group.appendChild(userDiv);
        } else {
            // Name tag — extract from content or use session
            const nameTag = document.createElement('div');
            nameTag.className = 'user-name-tag';
            const nameMatch = content.match(/^@(\S+?):\s/);
            const username = nameMatch ? nameMatch[1] : (sessionStorage.getItem('username') || '');
            nameTag.textContent = username ? '@' + username : '';

            const userDiv = document.createElement('div');
            userDiv.className = 'user-message';
            userDiv.textContent = content;

            if (nameTag.textContent) group.appendChild(nameTag);
            group.appendChild(userDiv);
        }
        messagesDiv.appendChild(group);
        scrollToBottom();
        return group;
    }
    function addSystemMessage(content) {
        const sysDiv = document.createElement('div');
        sysDiv.className = 'system-message';
        sysDiv.innerText = content;
        messagesDiv.appendChild(sysDiv);
        scrollToBottom();
    }
    function showPersistentReminder(files) {
        if (persistentReminderDiv) persistentReminderDiv.remove();
        const names = files.map(f => f.name).join(', ');
        persistentReminderDiv = document.createElement('div');
        persistentReminderDiv.className = 'persistent-reminder';
        persistentReminderDiv.innerHTML = `
            <span>📄 已加载文件: ${names}</span>
            <button class="clear-reminder" title="清除所有已加载文件">✖</button>
        `;
        const clearBtn = persistentReminderDiv.querySelector('.clear-reminder');
        clearBtn.onclick = async () => {
            const confirmed = await confirm('确定要清除所有已加载的文件吗？');
            if (confirmed) {
                selectedFiles = [];
                persistentReminderDiv.remove();
                persistentReminderDiv = null;
                addSystemMessage('已清除所有待发送的文件。');
            }
        };
        messagesDiv.insertBefore(persistentReminderDiv, messagesDiv.firstChild);
        scrollToBottom();
    }

    // Drag & Multi‑select Manager (no delete/escape)
    class FileListManager {
        constructor(container, options = {}) {
            this.container = container;
            this.onSelectionChange = options.onSelectionChange || (() => {});
            this.onDoubleClick = options.onDoubleClick || (() => {});
            this.selectableClass = options.selectableClass || '.file-item';
            this.selectedClass = 'selected';
            this.selectedItems = new Set();
            this.lastClicked = null;
            this.isDragging = false;
            this.selectionRect = null;
            this.init();
        }

        init() {
            // Click selection with Ctrl/Shift
            this.container.addEventListener('click', (e) => {
                const item = e.target.closest(this.selectableClass);
                if (!item) return;
                e.stopPropagation();
                const isCtrl = e.ctrlKey || e.metaKey;
                const isShift = e.shiftKey;
                if (isCtrl) {
                    this.toggleSelect(item);
                } else if (isShift && this.lastClicked) {
                    this.selectRange(this.lastClicked, item);
                } else {
                    this.clearSelection();
                    this.selectSingle(item);
                }
                this.lastClicked = item;
                this.onSelectionChange(Array.from(this.selectedItems));
            });

            // Double-click
            this.container.addEventListener('dblclick', (e) => {
                const item = e.target.closest(this.selectableClass);
                if (item) this.onDoubleClick(item);
            });

            // Drag rectangle (marquee)
            this.startMarquee = this.startMarquee.bind(this);
            this.updateMarquee = this.updateMarquee.bind(this);
            this.endMarquee = this.endMarquee.bind(this);
            this.container.addEventListener('mousedown', this.startMarquee);
            document.addEventListener('mousemove', this.updateMarquee);
            document.addEventListener('mouseup', this.endMarquee);
        }

        startMarquee(e) {
            if (e.target.closest(this.selectableClass)) return;
            if (e.button !== 0) return;
            this.isDragging = true;
            this.startX = e.clientX;
            this.startY = e.clientY;
            if (!this.selectionRect) {
                this.selectionRect = document.createElement('div');
                this.selectionRect.className = 'selection-rect';
                document.body.appendChild(this.selectionRect);
            }
            this.selectionRect.style.display = 'block';
            this.selectionRect.style.left = `${this.startX}px`;
            this.selectionRect.style.top = `${this.startY}px`;
            this.selectionRect.style.width = '0px';
            this.selectionRect.style.height = '0px';
        }

        updateMarquee(e) {
            if (!this.isDragging) return;
            const left = Math.min(e.clientX, this.startX);
            const top = Math.min(e.clientY, this.startY);
            const width = Math.abs(e.clientX - this.startX);
            const height = Math.abs(e.clientY - this.startY);
            this.selectionRect.style.left = `${left}px`;
            this.selectionRect.style.top = `${top}px`;
            this.selectionRect.style.width = `${width}px`;
            this.selectionRect.style.height = `${height}px`;
        }

        endMarquee(e) {
            if (!this.isDragging) return;
            this.isDragging = false;
            if (this.selectionRect) this.selectionRect.style.display = 'none';
            const rect = this.selectionRect.getBoundingClientRect();
            const items = this.container.querySelectorAll(this.selectableClass);
            this.clearSelection();
            for (const item of items) {
                const itemRect = item.getBoundingClientRect();
                if (rect.left < itemRect.right && rect.right > itemRect.left &&
                    rect.top < itemRect.bottom && rect.bottom > itemRect.top) {
                    this.selectSingle(item);
                }
            }
            this.onSelectionChange(Array.from(this.selectedItems));
        }

        selectSingle(item) {
            item.classList.add(this.selectedClass);
            this.selectedItems.add(item);
        }

        toggleSelect(item) {
            if (item.classList.contains(this.selectedClass)) {
                item.classList.remove(this.selectedClass);
                this.selectedItems.delete(item);
            } else {
                item.classList.add(this.selectedClass);
                this.selectedItems.add(item);
            }
        }

        selectRange(from, to) {
            const items = Array.from(this.container.querySelectorAll(this.selectableClass));
            const startIndex = items.indexOf(from);
            const endIndex = items.indexOf(to);
            if (startIndex === -1 || endIndex === -1) return;
            const [low, high] = [Math.min(startIndex, endIndex), Math.max(startIndex, endIndex)];
            this.clearSelection();
            for (let i = low; i <= high; i++) {
                this.selectSingle(items[i]);
            }
        }

        clearSelection() {
            this.selectedItems.forEach(item => item.classList.remove(this.selectedClass));
            this.selectedItems.clear();
        }

        getSelectedIds() {
            return Array.from(this.selectedItems).map(item => item.dataset.id);
        }

        destroy() {
            this.container.removeEventListener('mousedown', this.startMarquee);
            document.removeEventListener('mousemove', this.updateMarquee);
            document.removeEventListener('mouseup', this.endMarquee);
            if (this.selectionRect) this.selectionRect.remove();
        }
    }

    // ======================== Account Management ========================
    // ── Provider / Model selector ──
    const PROVIDER_NAMES = { deepseek:'DeepSeek', zhipu:'智谱AI', qwen:'通义千问', siliconflow:'硅基流动', nvidia:'NVIDIA' };
    const PROVIDER_MODELS = {
        deepseek: ['deepseek-v4-pro','deepseek-v4-flash','deepseek-chat'],
        zhipu: ['glm-4-flash','glm-4-plus','glm-4-air'],
        qwen: ['qwen3.7-plus','qwen-max','qwen-plus','qwen-turbo'],
        siliconflow: ['Qwen/Qwen2.5-7B-Instruct','deepseek-ai/DeepSeek-V3','Qwen/Qwen2.5-72B-Instruct'],
        nvidia: ['z-ai/glm-5.2','moonshotai/kimi-k2.6','meta/llama-3.1-8b-instruct','meta/llama-3.1-70b-instruct','meta/llama-3.1-405b-instruct','mistralai/mixtral-8x22b-instruct-v0.1','nvidia/llama-3.1-nemotron-70b-instruct']
    };

    async function loadUserStyle() {
        const section = document.getElementById('userStyleSection');
        if (!section) return;
        try {
            const r = await fetch('/my_writing_style', { credentials: 'include' });
            const d = await r.json();
            const s = d.style || {};
            if (s.style_label && s.style_label !== 'Not analyzed') {
                section.innerHTML = `<div>📊 <b>${escapeHtml(s.style_label)}</b></div>
                    <div style="font-size:0.65rem;margin-top:2px;">${escapeHtml(s.style_description||'')}</div>
                    <div style="font-size:0.6rem;color:var(--card-muted);margin-top:2px;">
                        基于 ${s.total_analyzed||0} 条消息 · v${s.version||1}
                        · <button id="reAnalyzeStyleBtn" style="font-size:0.6rem;background:none;border:none;color:#3b82f6;cursor:pointer;text-decoration:underline;">刷新</button>
                    </div>`;
                document.getElementById('reAnalyzeStyleBtn').onclick = async () => {
                    section.innerHTML = '分析中...';
                    await fetch('/my_writing_style/analyze', { method:'POST', credentials:'include' });
                    loadUserStyle();
                };
            } else {
                section.innerHTML = `<span style="font-size:0.7rem;">暂无风格画像。</span>
                    <br><button id="analyzeStyleBtn" class="file-btn" style="font-size:0.65rem;padding:2px 8px;margin-top:4px;">✍️ 分析我的风格</button>`;
                document.getElementById('analyzeStyleBtn').onclick = async () => {
                    section.innerHTML = '分析中...';
                    await fetch('/my_writing_style/analyze', { method:'POST', credentials:'include' });
                    loadUserStyle();
                };
            }
        } catch (_) { section.innerHTML = '<span style="font-size:0.7rem;">不可用</span>'; }
    }

    async function loadProviderSelector() {
        try {
            const res = await fetch('/llm_providers', { credentials: 'include' });
            if (!res.ok) throw new Error('Provider list unavailable');
            const data = await res.json();
            // FIX-016: dynamic provider list + models from backend (openrouter/nvidia)
            const available = (data.available && data.available.length) ? data.available : Object.keys(data.providers || {});
            const providersMeta = data.providers || {};
            // static fallback for display names
            const staticNames = { openrouter:'OpenRouter', nvidia:'NVIDIA NIM', deepseek:'DeepSeek', zhipu:'智谱AI', qwen:'通义千问', siliconflow:'硅基流动', mimo:'Mimo' };
            const current = data.active || available[0] || 'openrouter';
            const currentModel = sessionStorage.getItem('llmModel') || '';
            let html = '<div style="margin-bottom:12px"><strong>🤖 AI 模型设置</strong></div>';
            html += '<label>服务商:</label>';
            html += '<select id="providerSelect" style="width:100%;margin-bottom:8px">';
            for (const pid of available) {
                const nm = (providersMeta[pid] && providersMeta[pid].name) || staticNames[pid] || pid;
                const sel = pid === current ? ' selected' : '';
                html += `<option value="${pid}"${sel}>${nm}</option>`;
            }
            html += '</select>';
            html += '<label>模型:</label>';
            html += '<select id="modelSelect" style="width:100%;margin-bottom:8px">';
            const models = (providersMeta[current] && providersMeta[current].models) || [];
            for (const m of models) {
                const sel = m === currentModel ? ' selected' : '';
                html += `<option value="${m}"${sel}>${m}</option>`;
            }
            html += '</select>';
            html += '<button id="applyProviderBtn" class="file-btn" style="width:100%">应用模型设置</button>';
            html += '<small id="providerStatus" style="color:#888">当前: ' + ((providersMeta[current]&&providersMeta[current].name)||staticNames[current]||current) + '</small>';
            // Return HTML; event binding happens after insertion
            setTimeout(() => {
                const provSel = document.getElementById('providerSelect');
                const modelSel = document.getElementById('modelSelect');
                if (provSel && modelSel) {
                    provSel.onchange = () => {
                        const models = (providersMeta[provSel.value] && providersMeta[provSel.value].models) || [];
                        modelSel.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
                    };
                }
                document.getElementById('applyProviderBtn')?.addEventListener('click', async () => {
                    const provider = document.getElementById('providerSelect').value;
                    const model = document.getElementById('modelSelect').value;
                    try {
                        const r = await fetch('/llm_providers/set', {
                            method: 'POST',
                            headers: {'Content-Type':'application/json'},
                            credentials: 'include',
                            body: JSON.stringify({ provider, model })
                        });
                        const d = await r.json();
                        document.getElementById('providerStatus').textContent = d.success
                            ? `✅ 已切换至 ${(providersMeta[provider]&&providersMeta[provider].name)||staticNames[provider]||provider} / ${model}`
                            : '❌ ' + (d.error || '设置失败');
                    } catch (e) {
                        document.getElementById('providerStatus').textContent = '❌ 设置失败';
                    }
                });
            }, 0);
            return html;
        } catch (e) {
            return '<p style="color:#888">AI 模型设置不可用</p>';
        }
    }

    var accountModal = document.getElementById('accountModal');
    var closeAccountModal = document.getElementById('closeAccountModal');
    var accountContent = document.getElementById('accountContent');

    async function loadAccountModal() {
        // Fetch current auth status from server (consent_given now comes from API)
        const authRes = await fetch('/check_auth', { credentials: 'include' });
        const authData = await authRes.json();
        // Consent survives logout via localStorage so returning users skip re-consent
        const consentGiven = authData.consent_given || localStorage.getItem('consent_given') === '1';
        const isLoggedIn = authData.authenticated;
        // Save role for tab visibility (skill_auditor needs it)
        if (authData.role) sessionStorage.setItem('role', authData.role);
        else if (authData.is_admin) sessionStorage.setItem('role', 'admin');
        sessionStorage.setItem('is_auditor', (authData.is_auditor || authData.is_admin) ? '1' : '0');
        sessionStorage.setItem('hasLLM', authData.has_llm ? 'true' : 'false');
        const username = authData.username || sessionStorage.getItem('username') || '';
        const isAdmin = authData.is_admin || false;

        if (consentGiven && isLoggedIn) {
            const roleLabel = authData.is_admin ? '管理员' : (authData.is_auditor ? '审核员' : '用户');
            accountContent.innerHTML = `
                <p>已登录: <strong>${escapeHtml(username)}</strong> · <span style="font-size:.8rem;color:var(--card-muted);">角色: ${roleLabel}</span></p>
                <hr>
                <div style="margin-bottom:12px;">
                    <strong>✍️ 我的写作风格</strong>
                    <div id="userStyleSection" style="font-size:0.75rem;color:var(--card-muted);margin-top:4px;">加载中...</div>
                </div>
                <hr>
                <label>邮箱 (用于PIN变更验证):</label>
                <div style="display:flex;gap:6px;margin-bottom:8px;">
                    <input type="email" id="userEmail" placeholder="your@company.com" style="flex:1;">
                    <button id="saveEmailBtn" class="file-btn" style="padding:4px 12px;font-size:.7rem;">保存</button>
                </div>
                <hr>
                <label>新用户名 (5-18字符):</label>
                <input type="text" id="newUsername" placeholder="留空则不修改" style="width:100%; margin-bottom:8px;">
                <label>新PIN (4或6位数字):</label>
                <select id="newPinLength" style="width:100%; margin-bottom:4px;">
                    <option value="4">4位</option>
                    <option value="6" selected>6位</option>
                </select>
                <input type="password" id="newPin" placeholder="新PIN" style="width:100%; margin-bottom:4px;">
                <input type="password" id="confirmNewPin" placeholder="确认新PIN" style="width:100%; margin-bottom:4px;">
                <div style="display:flex;gap:6px;margin-bottom:8px;">
                    <input type="text" id="pinVerifyCode" placeholder="邮箱验证码(4位)" maxlength="4" style="flex:1;">
                    <button id="requestPinCodeBtn" class="file-btn" style="padding:4px 10px;font-size:.7rem;background:#bccfde;">获取验证码</button>
                </div>
                <label>当前PIN (必填):</label>
                <input type="password" id="currentPin" placeholder="当前PIN" style="width:100%; margin-bottom:8px;">
                <button id="updateAccountBtn" class="file-btn" style="margin-top: 10px;">更新账户</button>
                <button id="deleteAccountBtn" class="file-btn" style="margin-top: 10px; font-weight: bold;">${sessionStorage.getItem('deletion_pending') === '1' ? '⚠️ 输入验证码确认删除' : '⚠️ 申请删除账户'}</button>
                <button id="logoutBtn" class="file-btn" style="margin-top: 10px;">退出登录</button>
            `;
            document.getElementById('updateAccountBtn')?.addEventListener('click', updateAccount);
            document.getElementById('deleteAccountBtn')?.addEventListener('click', requestDeleteAccount);
            document.getElementById('logoutBtn')?.addEventListener('click', logout);
            document.getElementById('saveEmailBtn')?.addEventListener('click', saveEmail);
            document.getElementById('requestPinCodeBtn')?.addEventListener('click', requestPinCode);
            loadUserStyle();  // async, doesn't block
        } else if (consentGiven && !isLoggedIn) {
            // Consent given but not logged in – show registration / login options
            accountContent.innerHTML = `
                <p>您已同意数据收集，但尚未创建账户。</p>
                <p>创建账户后可跨设备同步聊天记录和文件。</p>
                <form onsubmit="return false" style="margin:0;">
                    <input type="text" id="regUsernameModal" placeholder="用户名 (5-18字符)" autocomplete="username" style="width:100%; margin-bottom:8px;">
                    <select id="pinLengthModal" style="width:100%; margin-bottom:8px;">
                        <option value="4">4位</option>
                        <option value="6" selected>6位</option>
                    </select>
                    <input type="password" id="regPinModal" placeholder="PIN" autocomplete="new-password" style="width:100%; margin-bottom:8px;">
                    <input type="password" id="confirmPinModal" placeholder="确认PIN" autocomplete="new-password" style="width:100%; margin-bottom:8px;">
                    <button id="createAccountModalBtn" class="file-btn" style="margin-top: 10px;">创建账户</button>
                </form>
                <button id="loginBtn" class="file-btn" style="margin-top: 10px;">已有账户？登录</button>
            `;
            document.getElementById('createAccountModalBtn')?.addEventListener('click', createAccountFromModal);
            document.getElementById('loginBtn')?.addEventListener('click', showLoginModal);
        } else {
            // Anonymous (consent not given)
            accountContent.innerHTML = `
                <p>您当前以匿名身份使用，未同意数据收集。</p>
                <p>对话记录和文件仅保存在当前会话中，关闭浏览器后即丢失。</p>
                <p>同意数据收集并创建账户后，可跨设备同步数据。</p>
                <button id="agreeAndCreateBtn" class="file-btn" style="margin-top: 10px;">同意数据收集并创建账户</button>
                <button id="loginExistingBtn" class="file-btn" style="margin-top: 10px;">已有账户？登录</button>
            `;
            document.getElementById('agreeAndCreateBtn')?.addEventListener('click', showConsentAndAccountFlow);
            document.getElementById('loginExistingBtn')?.addEventListener('click', showLoginFlowForAnonymous);
        }
    }

    async function createAccountFromModal() {
        console.log("createAccountFromModal called");
        const username = document.getElementById('regUsernameModal').value.trim();
        const pin = document.getElementById('regPinModal').value;
        const confirmPin = document.getElementById('confirmPinModal').value;
        const pinLength = parseInt(document.getElementById('pinLengthModal').value);
        if (!username || !pin) { alert('请填写用户名和PIN'); return; }
        if (pin !== confirmPin) { alert('PIN不一致'); return; }
        if (pin.length !== pinLength || !/^\d+$/.test(pin)) { alert(`PIN必须是${pinLength}位数字`); return; }
        if (username.length < 5 || username.length > 18) { alert('用户名长度应为5-18个字符'); return; }
        try {
            const accountRes = await fetch('/create_account', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ username, pin, pin_length: pinLength })
            });
            const accountData = await accountRes.json();
            if (accountRes.ok) {
                showToast('账户创建成功', 'success', 3000);
                sessionStorage.setItem('username', username);
                location.reload();
            } else {
                alert(accountData.error || '创建失败');
            }
        } catch (err) {
            console.error(err);
            alert('网络错误，请重试');
        }
    }

    function showLoginModal() {
        console.log("showLoginModal called");
        // Close account modal first, then show clean login overlay
        accountModal.style.display = 'none';
        // Remove any existing consent overlays
        document.querySelectorAll('.consent-overlay').forEach(o => o.remove());
        const overlay = document.createElement('div');
        overlay.className = 'consent-overlay';
        overlay.innerHTML = `
            <div class="consent-card" style="max-width:400px;">
                <h3>登录</h3>
                <form onsubmit="return false" style="margin:0;">
                <input type="text" id="loginUsername" placeholder="用户名" autocomplete="username" style="width:100%;margin-bottom:8px;padding:8px;border-radius:8px;border:1px solid var(--card-border);">
                <input type="password" id="loginPin" placeholder="PIN" autocomplete="current-password" style="width:100%;margin-bottom:8px;padding:8px;border-radius:8px;border:1px solid var(--card-border);">
                <button id="doLoginBtn" class="consent-action-btn" style="margin-top:0;">登录</button>
                </form>
                <button id="cancelLoginBtn" style="margin-top:8px;background:none;border:none;color:var(--card-muted);cursor:pointer;">取消</button>
            </div>
        `;
        document.body.appendChild(overlay);
        document.getElementById('doLoginBtn').addEventListener('click', () => {
            const username = document.getElementById('loginUsername').value.trim();
            const pin = document.getElementById('loginPin').value;
            if (!username || !pin) { alert('请填写用户名和PIN'); return; }
            loginAndClose(overlay, username, pin);
        });
        document.getElementById('cancelLoginBtn').addEventListener('click', () => overlay.remove());
    }

    async function loginAndClose(overlay, username, pin) {
        try {
            const csrf = await getCsrfToken();
            const res = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf },
                credentials: 'include',
                body: JSON.stringify({ username, pin })
            });
            const data = await res.json();
            if (res.ok) {
                sessionStorage.setItem('username', username);
                sessionStorage.setItem('user_id', data.user_id);
                sessionStorage.setItem('just_logged_in', '1');
                sessionStorage.setItem('is_auditor', (data.is_auditor || data.is_admin) ? '1' : '0');
                if (data.is_admin) {
                    sessionStorage.setItem('isAdmin', 'true');
                }
                // Close overlay, then reload
                overlay.remove();
                document.querySelectorAll('.consent-overlay, .modal[style*=\"block\"]').forEach(o => { o.style.display = 'none'; try { o.remove(); } catch(e) {} });
                showToast('登录成功', 'success', 1500);
                setTimeout(() => location.reload(), 500);
            } else {
                alert(data.error || '登录失败');
            }
        } catch (err) {
            console.error(err);
            alert('网络错误，请重试');
        }
    }

    var _csrfToken = '';
    async function getCsrfToken() {
        if (_csrfToken) return _csrfToken;
        try {
            const r = await fetch('/get_csrf_token', { credentials: 'include' });
            const d = await r.json();
            _csrfToken = d.csrf_token || '';
        } catch(e) { _csrfToken = ''; }
        return _csrfToken;
    }

    async function login(username, pin) {
        console.log("login called", username);
        try {
            const csrf = await getCsrfToken();
            const res = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf },
                credentials: 'include',
                body: JSON.stringify({ username, pin })
            });
            const data = await res.json();
            if (res.ok) {
                sessionStorage.setItem('username', username);
                sessionStorage.setItem('user_id', data.user_id);
                sessionStorage.setItem('just_logged_in', '1');
                sessionStorage.setItem('is_auditor', (data.is_auditor || data.is_admin) ? '1' : '0');
                if (data.is_admin) {
                    sessionStorage.setItem('isAdmin', 'true');
                    sessionStorage.setItem('forceAdminTab', 'true');
                }
                location.reload();
            } else {
                alert(data.error || '登录失败');
            }
        } catch (err) {
            console.error(err);
            alert('网络错误，请重试');
        }
    }

    async function logout() {
        try {
            const res = await fetch('/logout', { method: 'POST', credentials: 'include' });
            if (res.ok) {
                sessionStorage.clear();
                location.href = '/';
            } else {
                console.error('Logout failed');
                sessionStorage.clear();
                location.href = '/';
            }
        } catch (err) {
            console.error('Logout error', err);
            sessionStorage.clear();
            location.href = '/';
        }
    }

    function showConsentAndAccountFlow() {
        console.log("showConsentAndAccountFlow called");
        const overlay = document.createElement('div');
        overlay.className = 'consent-overlay';
        overlay.innerHTML = `
            <div class="consent-card" style="max-width: 450px;">
                <h3>📋 数据收集与账户注册</h3>
                <p>请先同意数据收集，然后注册账户以继续使用。</p>
                <div class="consent-scrollbox" style="max-height:150px;">
                    <p>我们会匿名存储您的对话内容和反馈。请勿输入个人隐私信息。</p>
                </div>
                <label><input type="checkbox" id="tempConsentCheckbox"> 我已阅读并同意</label>
                <div id="tempAccountFields" style="display:none; margin-top:15px;">
                    <input type="text" id="tempUsername" placeholder="用户名 (5-18字符)" style="width:100%; margin-bottom:8px;">
                    <select id="tempPinLength" style="width:100%; margin-bottom:8px;">
                        <option value="4">4位</option>
                        <option value="6" selected>6位</option>
                    </select>
                    <input type="password" id="tempPin" placeholder="PIN" style="width:100%; margin-bottom:8px;">
                    <input type="password" id="tempConfirmPin" placeholder="确认PIN" style="width:100%; margin-bottom:8px;">
                    <button id="tempCreateBtn" style="background:#27ae60; color:white; border:none; border-radius:24px; padding:8px;">创建账户</button>
                </div>
                <button id="tempCloseBtn" style="margin-top:15px;">取消</button>
            </div>
        `;
        document.body.appendChild(overlay);
        const tempConsentCheckbox = overlay.querySelector('#tempConsentCheckbox');
        const tempAccountFields = overlay.querySelector('#tempAccountFields');
        const tempCreateBtn = overlay.querySelector('#tempCreateBtn');
        const tempCloseBtn = overlay.querySelector('#tempCloseBtn');
        tempConsentCheckbox.onchange = () => {
            tempAccountFields.style.display = tempConsentCheckbox.checked ? 'block' : 'none';
        };
        tempCreateBtn.onclick = async () => {
            const username = overlay.querySelector('#tempUsername').value.trim();
            const pin = overlay.querySelector('#tempPin').value;
            const confirmPin = overlay.querySelector('#tempConfirmPin').value;
            const pinLength = parseInt(overlay.querySelector('#tempPinLength').value);
            if (!username || !pin) { alert('请填写用户名和PIN'); return; }
            if (pin !== confirmPin) { alert('PIN不一致'); return; }
            if (pin.length !== pinLength || !/^\d+$/.test(pin)) { alert(`PIN必须是${pinLength}位数字`); return; }
            if (username.length < 5 || username.length > 18) { alert('用户名长度应为5-18个字符'); return; }
            // Registration auto-sets consent — no separate /consent call needed
            const accountRes = await fetch('/create_account', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ username, pin, pin_length: pinLength })
            });
            const accountData = await accountRes.json();
            if (accountRes.ok) {
                localStorage.setItem('consent_given', '1');
                showToast('账户创建成功', 'success', 3000);
                sessionStorage.setItem('username', username);
                location.reload();
            } else alert(accountData.error || '创建失败');
        };
        tempCloseBtn.onclick = () => overlay.remove();
    }

    function showLoginFlowForAnonymous() {
        console.log("showLoginFlowForAnonymous called");
        const overlay = document.createElement('div');
        overlay.className = 'consent-overlay';
        overlay.innerHTML = `
            <div class="consent-card" style="max-width: 450px;">
                <h3>📋 登录需要同意数据收集</h3>
                <p>登录账户前，请先同意数据收集政策。</p>
                <div class="consent-scrollbox" style="max-height:150px;">
                    <p>我们会根据隐私政策存储您的对话记录和文件，以便跨设备同步。</p>
                </div>
                <label><input type="checkbox" id="tempConsentCheckbox"> 我已阅读并同意隐私政策</label>
                <div id="tempLoginFields" style="display:none; margin-top:15px;">
                    <form onsubmit="return false" style="margin:0;">
                    <input type="text" id="tempLoginUsername" placeholder="用户名" autocomplete="username" style="width:100%; margin-bottom:8px;">
                    <input type="password" id="tempLoginPin" placeholder="PIN" autocomplete="current-password" style="width:100%; margin-bottom:8px;">
                    <button id="tempLoginBtn" style="background:#27ae60; color:white; border:none; border-radius:24px; padding:8px;">登录</button>
                    </form>
                </div>
                <button id="tempCloseBtn" style="margin-top:15px;">取消</button>
            </div>
        `;
        document.body.appendChild(overlay);
        const tempConsentCheckbox = overlay.querySelector('#tempConsentCheckbox');
        const tempLoginFields = overlay.querySelector('#tempLoginFields');
        const tempLoginBtn = overlay.querySelector('#tempLoginBtn');
        const tempCloseBtn = overlay.querySelector('#tempCloseBtn');
        tempConsentCheckbox.onchange = () => {
            tempLoginFields.style.display = tempConsentCheckbox.checked ? 'block' : 'none';
        };
        tempLoginBtn.onclick = async () => {
            const username = overlay.querySelector('#tempLoginUsername').value.trim();
            const pin = overlay.querySelector('#tempLoginPin').value;
            if (!username || !pin) { alert('请填写用户名和PIN'); return; }
            const csrf = await getCsrfToken();
            localStorage.setItem('consent_given', '1');
            const loginRes = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf },
                credentials: 'include',
                body: JSON.stringify({ username, pin })
            });
            const loginData = await loginRes.json();
            if (loginRes.ok) {
                showToast('登录成功', 'success', 3000);
                sessionStorage.setItem('username', username);
                location.reload();
            } else alert(loginData.error || '登录失败');
        };
        tempCloseBtn.onclick = () => overlay.remove();
    }

    async function updateAccount() {
        const newUsername = document.getElementById('newUsername')?.value.trim();
        const newPin = document.getElementById('newPin')?.value;
        const verifyCode = document.getElementById('pinVerifyCode')?.value.trim();
        const confirmPin = document.getElementById('confirmNewPin')?.value;
        const currentPin = document.getElementById('currentPin')?.value;
        if (!currentPin) { alert('请输入当前PIN'); return; }
        if (newPin && newPin !== confirmPin) { alert('新PIN与确认PIN不一致'); return; }
        try {
            const res = await fetch('/update_account', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                    new_username: newUsername,
                    new_pin: newPin,
                    pin_length: parseInt(document.getElementById('newPinLength')?.value||6, 10),
                    current_pin: currentPin,
                    verify_code: newPin ? verifyCode : null
                })
            });
            const data = await res.json();
            if (res.ok) {
                showToast('账户信息已更新', 'success', 3000);
                if (newUsername) sessionStorage.setItem('username', newUsername);
                location.reload();
            } else alert(data.error || '更新失败');
        } catch (err) {
            console.error(err);
            alert('网络错误，请重试');
        }
    }

    async function saveEmail() {
        const email = document.getElementById('userEmail')?.value.trim();
        if (!email || !email.includes('@')) { alert('请输入有效邮箱'); return; }
        const res = await fetch('/set_email', { method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include', body:JSON.stringify({email}) });
        if (res.ok) showToast('邮箱已保存', 'success');
        else { const d = await res.json(); alert(d.error || '保存失败'); }
    }

    async function requestPinCode() {
        const res = await fetch('/request_pin_change_code', { method:'POST', credentials:'include' });
        const d = await res.json();
        if (res.ok) showToast(d.hint || '验证码已发送', 'info', 5000);
        else alert(d.error || '请求失败');
    }

    async function requestDeleteAccount() {
        if (sessionStorage.getItem('deletion_pending') === '1') {
            showConfirmDeleteModal();
            return;
        }
        const res = await fetch('/request_delete_account', { method:'POST', credentials:'include' });
        const d = await res.json();
        if (!res.ok) { alert(d.error || '加载失败'); return; }
        const inventory = d.inventory || [];
        const sessionCnt = d.session_count || 0;
        // Build selection modal
        const modal = createQuickModal('选择保留数据');
        let html = `<p style="font-size:.78rem;color:var(--card-muted);margin-bottom:4px;">选择要保留给公司的数据（其余将永久删除）。项目文件默认保留。</p>
            <p style="font-size:.7rem;margin-bottom:8px;">💬 ${sessionCnt}个聊天会话（将永久删除）</p>
            <div style="max-height:300px;overflow-y:auto;margin-bottom:8px;">
            <table style="width:100%;font-size:.73rem;border-collapse:collapse;">`;
        for (const item of inventory) {
            const checked = item.choosable ? 'checked' : 'checked disabled';
            const disabled = item.choosable ? '' : 'disabled';
            html += `<tr style="border-bottom:1px solid var(--card-border);">
                <td style="padding:3px 4px;"><input type="checkbox" class="del-keep-cb" data-type="${item.type}" data-id="${item.id}" ${checked} ${disabled}></td>
                <td>${escapeHtml(item.name)}</td>
                <td style="color:var(--card-muted);font-size:.65rem;">${item.size_kb > 0 ? item.size_kb+'KB' : ''}</td>
                <td style="font-size:.6rem;color:var(--card-muted);">${item.note||''}</td>
            </tr>`;
        }
        html += `</table></div>
            <div style="display:flex;gap:8px;align-items:center;">
            <button id="delSelectAll" class="file-btn" style="font-size:.7rem;">全选</button>
            <button id="delDeselectAll" class="file-btn" style="font-size:.7rem;">全不选</button>
            <button id="delSubmitBtn" class="file-btn" style="background:#e74c3c;color:white;margin-left:auto;font-size:.78rem;">⚠️ 提交删除申请</button>
            <span id="delStatus" style="font-size:.7rem;"></span></div>`;
        modal.innerHTML(html);
        modal.querySelector('#delSelectAll').onclick = () => modal.querySelectorAll('.del-keep-cb:not([disabled])').forEach(c => c.checked = true);
        modal.querySelector('#delDeselectAll').onclick = () => modal.querySelectorAll('.del-keep-cb:not([disabled])').forEach(c => c.checked = false);
        modal.querySelector('#delSubmitBtn').onclick = async () => {
            const keep = [];
            modal.querySelectorAll('.del-keep-cb:checked').forEach(c => keep.push({type: c.dataset.type, id: parseInt(c.dataset.id)}));
            if (!confirm(`确定提交删除申请吗？\\n保留 ${keep.length} 项数据给公司，其余 ${inventory.length - keep.length} 项将永久删除。`)) return;
            const r = await fetch('/submit_delete_choices', { method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include', body:JSON.stringify({keep_ids:keep}) });
            const rd = await r.json();
            if (r.ok) { showToast(rd.message || '已提交', 'info', 6000); modal.remove(); sessionStorage.setItem('deletion_pending', '1'); showConfirmDeleteModal(); }
            else alert(rd.error || '提交失败');
        };
    }

    async function deleteAccount() {
        const pin = await prompt('请输入您的PIN以确认删除账户（所有数据将永久丢失）');
        if (!pin) return;
        try {
            const res = await fetch('/delete_account', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ pin })
            });
            const data = await res.json();
            if (res.ok) {
                alert('账户已删除');
                location.href = '/';
            } else alert(data.error || '删除失败');
        } catch (err) {
            console.error(err);
            alert('网络错误，请重试');
        }
    }

    function showConfirmDeleteModal() {
        const modal = createQuickModal('确认删除账户');
        const lastRequest = sessionStorage.getItem('deletion_last_request') || '';
        if (lastRequest) {
            const elapsed = (Date.now() - parseInt(lastRequest)) / 1000;
            if (elapsed < 300) {
                modal.innerHTML(`<p>请等待 ${Math.ceil(300 - elapsed)} 秒后重试。</p>`);
                return;
            }
        }
        modal.innerHTML(`
            <p style="font-size:.78rem;color:var(--card-muted);margin-bottom:8px;">管理员已发送验证码到您的邮箱，请在此输入。</p>
            <label>验证码 (4位):</label>
            <input type="text" id="delConfirmCode" maxlength="4" placeholder="管理员发送的验证码" style="width:100%;margin-bottom:8px;">
            <label>PIN:</label>
            <input type="password" id="delConfirmPin" placeholder="您的PIN" style="width:100%;margin-bottom:8px;">
            <div style="display:flex;gap:8px;">
                <button id="delConfirmBtn" class="file-btn" style="background:#e74c3c;color:white;">确认删除</button>
                <button id="delCancelReqBtn" class="file-btn">取消删除请求</button>
            </div>
            <span id="delConfirmStatus" style="font-size:.7rem;color:var(--card-muted);"></span>
        `);
        modal.querySelector('#delConfirmBtn').onclick = async () => {
            const code = modal.querySelector('#delConfirmCode').value.trim();
            const pin = modal.querySelector('#delConfirmPin').value.trim();
            if (!code || !pin) { modal.querySelector('#delConfirmStatus').textContent = '请填写验证码和PIN'; return; }
            const btn = modal.querySelector('#delConfirmBtn');
            btn.disabled = true; btn.textContent = '...';
            try {
                const res = await fetch('/confirm_delete_account', { method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include', body:JSON.stringify({code, pin}) });
                const d = await res.json();
                if (res.ok) {
                    sessionStorage.removeItem('deletion_pending');
                    sessionStorage.setItem('deletion_last_request', Date.now().toString());
                    showToast('账户已删除', 'success', 3000);
                    modal.remove();
                    setTimeout(() => { location.href = '/'; }, 1000);
                } else {
                    modal.querySelector('#delConfirmStatus').textContent = d.error || '确认失败';
                }
            } catch (_) {
                modal.querySelector('#delConfirmStatus').textContent = '网络错误';
            }
            btn.disabled = false; btn.textContent = '确认删除';
        };
        modal.querySelector('#delCancelReqBtn').onclick = async () => {
            if (!confirm('取消删除请求？')) return;
            sessionStorage.removeItem('deletion_pending');
            sessionStorage.setItem('deletion_last_request', Date.now().toString());
            showToast('删除请求已取消', 'info', 3000);
            modal.remove();
        };
    }

    var accountSettingsBtn = document.getElementById('accountSettingsBtn');
    if (accountSettingsBtn) {
        accountSettingsBtn.onclick = () => {
            console.log("Account settings button clicked");
            accountModal.style.display = 'block';
            loadAccountModal();
        };
    }
    if (closeAccountModal) {
        closeAccountModal.onclick = () => { accountModal.style.display = 'none'; };
    }

    // ======================== File Station Functions ========================
    var fileStationData = [];
    var selectedFileIds = new Set();
    var fileStationBtn = document.getElementById('fileStationBtn');
    var fileStationModal = document.getElementById('fileStationModal');
    var closeFileStationModal = document.getElementById('closeFileStationModal');

    if (fileStationBtn) {
        fileStationBtn.onclick = () => {
            if (fileStationModal) {
                loadFileStation();          // load the file list
                fileStationModal.style.display = 'block';
            } else {
                console.error('File station modal not found');
            }
        };
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
    (function() {
        var el = document.getElementById('categoryFilterBar');
        if (el) el.style.display = selectedKnowledgeFiles.length > 0 ? 'flex' : 'none';
    })();
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
    document.getElementById('knowledgeBaseBtn').onclick = () => {
        loadKnowledgeFiles();
        document.getElementById('knowledgeBaseModal').style.display = 'block';
    };

    // Close modal
    function closeKnowledgeModal() {
        document.getElementById('knowledgeBaseModal').style.display = 'none';
        // Update category filter bar visibility after closing
        showCatFilterIfNeeded();
    }
    document.getElementById('closeKnowledgeBaseModal').onclick = closeKnowledgeModal;
    document.getElementById('cancelKnowledgeBtn').onclick = closeKnowledgeModal;

    // Confirm selection
    document.getElementById('confirmKnowledgeBtn').onclick = () => {
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

    // ======================== Storage Check ========================
    async function checkStorage() {
        try {
            const res = await fetch('/check_storage', { credentials: 'include' });
            const data = await res.json();
            const warningSpan = document.getElementById('storageWarning');
            if (data.warning && warningSpan) warningSpan.innerHTML = '⚠️ ' + escapeHtml(data.message) + '，可在「我的文件」中清理旧文件释放空间。';
            else if (warningSpan) warningSpan.innerHTML = '';
        } catch (err) { console.error('Storage check failed', err); }
    }

    // ======================== Theme ========================
    function setTheme(theme) {
        if (theme === 'dark') {
            document.body.classList.remove('light');
            document.body.classList.add('dark');
            const darkCss = document.getElementById('highlight-dark');
            const lightCss = document.getElementById('highlight-light');
            document.getElementById('highlight-light').disabled = true;
            document.getElementById('highlight-dark').disabled = false;
            localStorage.setItem('theme', 'dark');
        } else {
            document.body.classList.remove('dark');
            document.body.classList.add('light');
            const darkCss = document.getElementById('highlight-dark');
            const lightCss = document.getElementById('highlight-light');
            document.getElementById('highlight-light').disabled = false;
            document.getElementById('highlight-dark').disabled = true;
            localStorage.setItem('theme', 'light');
        }
        document.querySelectorAll('pre code').forEach((block) => {
            if (window.hljs) hljs.highlightElement(block);
        });
    }

    // ======================== Sidebar Layout ========================
    let sidebarCollapsed = false;
    const sidebar = document.getElementById('sidebar');
    const sidebarOverlay = document.getElementById('sidebarOverlay');
    const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
    function openSidebar() { sidebar.classList.add('open'); sidebarOverlay.classList.add('open'); }
    function closeSidebar() { sidebar.classList.remove('open'); sidebarOverlay.classList.remove('open'); }
    if (toggleSidebarBtn) {
        toggleSidebarBtn.onclick = (e) => {
            e.stopPropagation();
            if (window.innerWidth <= 1024) { openSidebar(); }  // tablet+phone: overlay
            else {
                sidebarCollapsed = !sidebarCollapsed;
                if (sidebarCollapsed) { sidebar.classList.add('collapsed'); localStorage.setItem('sidebarCollapsed', 'true'); }
                else { sidebar.classList.remove('collapsed'); localStorage.setItem('sidebarCollapsed', 'false'); }
            }
        };
    }
    if (sidebarOverlay) {
        sidebarOverlay.onclick = closeSidebar;
        // Swipe-to-close on overlay
        let touchStartX = 0;
        sidebarOverlay.addEventListener('touchstart', (e) => { touchStartX = e.touches[0].clientX; }, {passive:true});
        sidebarOverlay.addEventListener('touchend', (e) => {
            if (e.changedTouches[0].clientX - touchStartX < -30) closeSidebar();
        }, {passive:true});
    }
    const savedCollapsed = localStorage.getItem('sidebarCollapsed');
    if (savedCollapsed === 'true' && window.innerWidth > 1024) { sidebar.classList.add('collapsed'); sidebarCollapsed = true; }
    else sidebarCollapsed = false;

    // ── Mobile tab "更多" dropdown ──
    const moreBtn = document.getElementById('mobileMoreTabsBtn');
    const moreDropdown = document.getElementById('mobileMoreDropdown');
    const tabSeparator = document.querySelector('#tabBar .tab-separator');

    function populateMobileDropdown() {
        if (!moreBtn || !moreDropdown) return;
        const adminTabs = document.querySelectorAll('#tabBar .admin-tab');
        const hiddenTabs = [];
        adminTabs.forEach(btn => {
            if (btn.style.display === 'none' || !btn.offsetParent) {
                hiddenTabs.push(btn);
            }
        });
        // Also add admin tabs that exist but might be forced hidden
        if (hiddenTabs.length === 0) {
            adminTabs.forEach(btn => hiddenTabs.push(btn));
        }
        moreDropdown.innerHTML = hiddenTabs.map(btn =>
            `<button data-tab-id="${btn.id}">${btn.textContent}</button>`
        ).join('');
        moreDropdown.querySelectorAll('button').forEach(b => {
            b.onclick = () => {
                const target = document.getElementById(b.dataset.tabId);
                if (target) { target.click(); moreDropdown.classList.remove('open'); }
            };
        });
    }

    function updateMobileLayout() {
        const isPhone = window.innerWidth <= 640;
        // Sidebar
        if (isPhone) {
            sidebar.classList.remove('collapsed');
            sidebar.style.width = '';
        }
        // Tab bar
        if (moreBtn) moreBtn.style.display = isPhone ? 'inline-block' : 'none';
        if (moreDropdown) { moreDropdown.classList.remove('open'); moreDropdown.style.display = isPhone ? '' : 'none'; }
        if (tabSeparator) tabSeparator.style.display = isPhone ? 'none' : '';
        // Admin tabs
        document.querySelectorAll('#tabBar .admin-tab').forEach(btn => {
            if (isPhone) {
                btn.classList.add('visible');
                if (btn.style.display === 'none') btn.style.display = '';
                // Hide all admin tabs; dropdown will show them
                btn.style.display = 'none';
            } else {
                btn.classList.remove('visible');
                btn.style.display = btn.id === 'reviewTabBtn'
                    ? (btn.style.display) : '';  // restore original visibility
                const role = sessionStorage.getItem('role') || '';
                const isAuditor = sessionStorage.getItem('is_auditor') === '1';
                if (btn.id === 'reviewTabBtn' && (role === 'admin' || isAuditor)) btn.style.display = 'inline-block';
                if (btn.id === 'databaseTabBtn') btn.style.display = 'inline-block';
                if (btn.id === 'analyticsTabBtn') btn.style.display = 'inline-block';
            }
        });
        if (isPhone) populateMobileDropdown();
    }

    if (moreBtn) {
        moreBtn.onclick = (e) => {
            e.stopPropagation();
            populateMobileDropdown();
            moreDropdown.classList.toggle('open');
        };
        document.addEventListener('click', () => { if (moreDropdown) moreDropdown.classList.remove('open'); });
    }

    window.addEventListener('resize', () => {
        if (window.innerWidth <= 1024) {
            sidebar.classList.remove('collapsed');
        } else if (localStorage.getItem('sidebarCollapsed') === 'true') {
            sidebar.classList.add('collapsed'); sidebarCollapsed = true;
        } else {
            sidebar.classList.remove('collapsed'); sidebarCollapsed = false;
        }
        updateMobileLayout();
    });
    updateMobileLayout();  // initial run

    // ======================== Consent Modal Logic ========================
    const consentModal = document.getElementById('consentModal');
    if (consentModal) {
        const createAccountForm = document.getElementById('createAccountForm');
        const loginForm = document.getElementById('loginForm');
        const consentCreateOption = document.getElementById('consentCreateOption');
        const consentLoginOption = document.getElementById('consentLoginOption');
        const confirmCreateAccountBtn = document.getElementById('confirmCreateAccountBtn');
        const confirmLoginBtn = document.getElementById('confirmLoginBtn');

        function collapseAllForms() {
            if (createAccountForm) createAccountForm.style.display = 'none';
            if (loginForm) loginForm.style.display = 'none';
            document.querySelectorAll('.consent-option-btn').forEach(btn => btn.classList.remove('selected'));
        }

        if (consentCreateOption) {
            consentCreateOption.onclick = (e) => {
                e.stopPropagation();
                collapseAllForms();
                if (createAccountForm) createAccountForm.style.display = 'block';
                consentCreateOption.classList.add('selected');
            };
        }
        if (consentLoginOption) {
            consentLoginOption.onclick = (e) => {
                e.stopPropagation();
                collapseAllForms();
                if (loginForm) loginForm.style.display = 'block';
                consentLoginOption.classList.add('selected');
            };
        }
        if (confirmCreateAccountBtn) {
            confirmCreateAccountBtn.onclick = async (e) => {
                e.stopPropagation();
                const username = document.getElementById('regUsername').value.trim();
                const pin = document.getElementById('regPin').value;
                const confirmPin = document.getElementById('regConfirmPin').value;
                const pinLength = parseInt(document.getElementById('regPinLength').value);
                if (!username || !pin) { alert('请填写用户名和PIN'); return; }
                if (pin !== confirmPin) { alert('PIN不一致'); return; }
                if (pin.length !== pinLength || !/^\d+$/.test(pin)) { alert(`PIN必须是${pinLength}位数字`); return; }
                if (username.length < 5 || username.length > 18) { alert('用户名长度应为5-18个字符'); return; }
                try {
                    // Registration auto-sets consent — no separate /consent call needed
                    const accountRes = await fetch('/create_account', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        credentials: 'include',
                        body: JSON.stringify({ username, pin, pin_length: pinLength })
                    });
                    const accountData = await accountRes.json();
                    if (accountRes.ok) {
                        localStorage.setItem('consent_given', '1');
                        showToast('账户创建成功！', 'success', 3000);
                        sessionStorage.setItem('username', username);
                        location.reload();
                    } else {
                        alert(accountData.error || '账户创建失败');
                    }
                } catch (err) {
                    console.error(err);
                    alert('网络错误，请重试');
                }
            };
        }

        if (confirmLoginBtn) {
            confirmLoginBtn.onclick = async (e) => {
                e.stopPropagation();
                const username = document.getElementById('loginUsername').value.trim();
                const pin = document.getElementById('loginPin').value;
                if (!username || !pin) { alert('请填写用户名和PIN'); return; }
                try {
                    // Login auto-sets consent — no separate /consent call needed
                    localStorage.setItem('consent_given', '1');
                    const loginRes = await fetch('/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        credentials: 'include',
                        body: JSON.stringify({ username, pin })
                    });
                    const loginData = await loginRes.json();
                    if (loginRes.ok) {
                        showToast('登录成功', 'success', 3000);
                        sessionStorage.setItem('username', username);
                        sessionStorage.setItem('user_id', loginData.user_id);
                        if (loginData.is_admin) sessionStorage.setItem('isAdmin', 'true');
                        location.reload();
                    } else {
                        alert(loginData.error || '登录失败');
                    }
                } catch (err) {
                    console.error(err);
                    alert('网络错误，请重试');
                }
            };
        }

        if (createAccountForm) createAccountForm.addEventListener('click', (e) => e.stopPropagation());
        if (loginForm) loginForm.addEventListener('click', (e) => e.stopPropagation());
    }

    // ======================== Original DOM Event Listeners ========================
    const fileBtn = document.getElementById('fileBtn');
    const fileInput = document.getElementById('fileInput');
    const inputFileStationBtn = document.getElementById('inputFileStationBtn');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const newChatBtn = document.getElementById('newChatBtn');
    const maxTokensInput = document.getElementById('maxTokensInput');
    const applyTokensBtn = document.getElementById('applyTokensBtn');
    const tokenStatus = document.getElementById('tokenStatus');

    if (applyTokensBtn) {
        applyTokensBtn.onclick = async () => {
            let tokens = parseInt(maxTokensInput.value, 10);
            if (isNaN(tokens)) tokens = 4800;
            tokens = Math.min(4800, Math.max(100, tokens));
            const res = await fetch('/set_max_tokens', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ max_tokens: tokens })
            });
            if (res.ok) {
                tokenStatus.textContent = `✅ 已设置为 ${tokens}`;
                setTimeout(() => { tokenStatus.textContent = ''; }, 2000);
            } else {
                tokenStatus.textContent = '❌ 设置失败';
            }
        };
    }

    fileBtn.onclick = () => fileInput.click();
    fileInput.onchange = (e) => {
        if (e.target.files.length) {
            selectedFiles = Array.from(e.target.files);
            showPersistentReminder(selectedFiles);
            if (selectedFiles.length === 1) fileBtn.innerText = `📄 ${selectedFiles[0].name}`;
            else fileBtn.innerText = `📄 ${selectedFiles.length} 个文件`;
        } else {
            selectedFiles = [];
            fileBtn.innerText = '📄随问题上传文件';
        }
        fileInput.value = '';
    };

    // ── URL Fetch button ──
    const urlBtn = document.createElement('div');
    urlBtn.className = 'file-btn';
    urlBtn.id = 'urlFetchBtn';
    urlBtn.innerHTML = '🌐 网页';
    urlBtn.title = '输入网页URL，自动抓取内容分析';
    if (fileBtn && fileBtn.parentNode) {
        fileBtn.parentNode.insertBefore(urlBtn, fileBtn.nextSibling);
    }
    urlBtn.onclick = async () => {
        const result = await prompt('请输入网页URL：');
        const url = (typeof result === 'string') ? result.trim() : '';
        if (!url) return;
        urlBtn.innerHTML = '⏳ 抓取中...';
        urlBtn.style.opacity = '0.6';
        fetch('/fetch_url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ url: url })
        }).then(r => r.json()).then(data => {
            urlBtn.innerHTML = '🌐 网页';
            urlBtn.style.opacity = '1';
            if (data.error) {
                alert('抓取失败，请检查链接是否有效');
                return;
            }
            const preview = data.text.substring(0, 200).replace(/\n/g, ' ');
            const useIt = confirm(
                `已提取 ${data.text.length} 字符。\n\n预览: ${preview}...\n\n是否将此内容填入输入框？`
            );
            if (useIt) {
                messageInput.value = data.text.substring(0, 10000);
                messageInput.focus();
            }
        }).catch(err => {
            urlBtn.innerHTML = '🌐 网页';
            urlBtn.style.opacity = '1';
            alert('抓取失败，请检查链接后重试');
        });
    };

    sendBtn.onclick = sendMessage;
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            if (e.ctrlKey || e.metaKey) return;
            if (e.shiftKey) return;
            e.preventDefault();
            sendMessage();
        }
    });

    newChatBtn.onclick = async () => {
        if (isProcessing) { addSystemMessage('请等待当前请求完成后再创建新聊天。'); return; }
        const confirmed = await confirm('确定要创建新聊天吗？');
        if (!confirmed) return;
        _isCurrentSessionGrill = false;
        const banner = document.getElementById('grillModeBanner');
        if (banner) banner.style.display = 'none';
        const res = await fetch('/new_chat', { method: 'POST', credentials: 'include' });
        if (res.ok) {
            const data = await res.json();
            await loadSession(data.data?.thread_id || data.thread_id);
            await loadHistoryList();
            scrollToBottom();
            closeSidebar();
            selectedKnowledgeFiles = [];
            localStorage.removeItem('selectedKnowledgeFiles');
            const btn = document.getElementById('knowledgeBaseBtn');
            if (btn) btn.innerHTML = '📚 知识库';
            showCatFilterIfNeeded();
        }
    };

    const grillChatBtn = document.getElementById('grillChatBtn');
    if (grillChatBtn) {
        grillChatBtn.onclick = async () => {
            if (isProcessing) { addSystemMessage('请等待当前请求完成后再创建新聊天。'); return; }
            const confirmed = await confirm('确定要创建质问模式对话吗？\n\nAI将扮演挑剔的供应商，对你的招标文件进行拷问，帮你发现漏洞和不合理条款。');
            if (!confirmed) return;
            try {
                const res = await fetch('/api/chat/create_grill_thread', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include'
                });
                if (res.ok) {
                    const data = await res.json();
                    await loadSession(data.thread_id);
                    await loadHistoryList();
                    scrollToBottom();
                    closeSidebar();
                    selectedKnowledgeFiles = [];
                    localStorage.removeItem('selectedKnowledgeFiles');
                    const btn = document.getElementById('knowledgeBaseBtn');
                    if (btn) btn.innerHTML = '📚 知识库';
                    showCatFilterIfNeeded();
                } else {
                    showToast('创建质问模式失败', 'error', 3000);
                }
            } catch (e) {
                console.error('创建质问模式失败:', e);
                showToast('创建质问模式失败', 'error', 3000);
            }
        };
    }

    const analyzeImagesCheckbox = document.getElementById('analyzeImagesCheckbox');
    if (analyzeImagesCheckbox) {
        analyzeImagesCheckbox.addEventListener('change', async () => {
            const enabled = analyzeImagesCheckbox.checked;
            await fetch('/set_image_analysis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ enabled })
            });
        });
    }

    const analyzeVideosCheckbox = document.getElementById('analyzeVideosCheckbox');
    if (analyzeVideosCheckbox) {
        analyzeVideosCheckbox.addEventListener('change', async () => {
            const enabled = analyzeVideosCheckbox.checked;
            await fetch('/set_video_analysis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ enabled })
            });
        });
    }

    // ── 清标 (unified) — replaces Smart Compare / Doc Analysis / Compliance / AI Review ──
    initClearanceTool();

    // ── File station quick access from input bar ──
    if (inputFileStationBtn && window.__openFileStation) {
        inputFileStationBtn.onclick = window.__openFileStation;
    }

    // ── Admin sidebar: result history viewers ──
    initAdminResultViewers();

    // ======================== Admin Panel & Tab Functions ========================
    
    // --- Sidebar content loaders per tab ---
    async function loadSidebarProjects(cachedData) {
        const list = document.getElementById('sidebarProjectsList');
        if (!list) return;
        const isAdmin = sessionStorage.getItem('isAdmin') === 'true';
        try {
            const data = cachedData || await (await fetch('/admin/projects', { credentials: 'include' })).json();
            const projects = data.projects || [];
            let html = '';
            if (!projects.length) {
                html = '<li style="color:var(--card-muted);">暂无项目</li>';
            } else {
                html = projects.map(p =>
                    `<li style="padding:4px 8px;margin-bottom:4px;border-radius:4px;cursor:pointer;transition:background .15s;display:flex;align-items:center;"
                         onmouseover="this.style.background='var(--card-bg)'" onmouseout="this.style.background=''">
                         <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:pointer;"
                               onclick="document.getElementById('adminTabBtn').click(); if(typeof openProject==='function') openProject(${p.id}, '${escapeHtml(p.name).replace(/'/g, "\\'")}', '${p.status || 'active'}')">
                         📁 ${escapeHtml(p.name)}
                         ${p.status && p.status !== 'active' ? `<span style="font-size:.7rem;color:var(--card-muted);">${p.status}</span>` : ''}
                         </span>
                     </li>`
                ).join('');
            }
            // Admin: show all skills from ALL sources
            if (isAdmin) {
                html += '<hr style="margin:12px 0;"><h4 style="font-size:0.8rem;color:var(--card-muted);margin-bottom:6px;">🧠 全部技能</h4>';
                try {
                    const [kbRes, coRes] = await Promise.all([
                        fetch('/admin/all_user_kb', { credentials: 'include' }),
                        fetch('/company_kb/list', { credentials: 'include' })
                    ]);
                    const kbData = await kbRes.json();
                    const coData = await coRes.json();
                    const personalSkills = (kbData.files || []).filter(f => f.is_skill && f.owner);
                    const companySkills = (coData.files || []).filter(f => f.has_skill);

                    const allSkills = [
                        ...personalSkills.map(f => ({ ...f, _source: 'personal', _label: '🧠' })),
                        ...companySkills.map(f => ({ ...f, _source: 'company', _label: '🏢', _filename: f.filename })),
                    ];

                    if (allSkills.length > 0) {
                        html += allSkills.slice(0, 15).map(f => {
                            const name = escapeHtml(f._source === 'company' ? f._filename : (f.original_name || f.filename));
                            const ownerLabel = f._source === 'company' ? '公司' : `<small style="color:var(--card-muted);">(${f.owner})</small>`;
                            return `<li style="padding:3px 6px;font-size:.72rem;border-bottom:1px solid var(--card-border);display:flex;justify-content:space-between;align-items:center;">
                                <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${f._label} ${name} ${ownerLabel}</span>
                                ${f._source === 'personal' ? `<button data-promote-skill="${f.id}" style="background:#bccfde;color:#1e293b;border:none;border-radius:3px;padding:1px 6px;font-size:.6rem;cursor:pointer;flex-shrink:0;margin-left:4px;" title="加入公司知识库">↑</button>` : ''}
                            </li>`;
                        }).join('');
                    } else {
                        html += '<li style="color:var(--card-muted);font-size:.75rem;">暂未生成任何技能</li>';
                    }
                } catch(e) { html += '<li style="color:var(--card-muted);">加载失败</li>'; }
            }
            list.innerHTML = html;

            // Wire promote buttons for admin
            document.querySelectorAll('#sidebarProjectsList [data-promote-skill]').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const id = btn.dataset.promoteSkill;
                    try {
                        const res = await fetch('/admin/promote_to_company/' + id, { method: 'POST', credentials: 'include' });
                        const data = await res.json();
                        if (res.ok) { showToast(data.message || '已加入公司知识库', 'success'); loadSidebarProjects(); }
                        else showToast('操作失败，请重试', 'error');
                    } catch(err) { showToast('网络错误', 'error'); }
                };
            });
        } catch(e) { list.innerHTML = '<li style="color:var(--card-muted);">加载失败</li>'; }
    }

    async function loadSidebarKnowledge(myData, coData) {
        const myFiles = document.getElementById('sidebarKnowledgeFiles');
        const coFiles = document.getElementById('sidebarCompanyFiles');
        const isAdmin = sessionStorage.getItem('isAdmin') === 'true';
        try {
            if (!myData) { const r = await fetch('/knowledge_lab/list', { credentials: 'include' }); myData = await r.json(); }
            if (!coData) { const r = await fetch('/company_kb/list', { credentials: 'include' }); coData = await r.json(); }
            const myList = myData.files || [];
            const coList = coData.files || [];

            // Separate skills (files with has_skill flag or .md names) from regular files
            const skills = myList.filter(f => f.has_skill || (f.original_name || f.filename || '').toLowerCase().endsWith('.md') || /SKILL/i.test(f.original_name || f.filename || ''));
            const regular = myList.filter(f => !skills.includes(f));

            if (myFiles) {
                let html = '';
                if (skills.length > 0) {
                    html += '<div style="font-size:.7rem;color:var(--card-muted);margin-bottom:4px;">🧠 技能文件</div>';
                    html += skills.slice(0,6).map(f =>
                        `<li style="padding:2px 4px;font-size:.72rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;justify-content:space-between;align-items:center;">
                            <span>🧠 ${escapeHtml((f.original_name || f.filename).replace(/\\.[^.]+$/,''))}</span>
                            <span>
                                ${f.has_skill ? `<a href="/knowledge_lab/skill/${f.id}" download style="text-decoration:none;font-size:.6rem;color:#5a7c9b;margin-right:2px;" title="下载技能文件">📥</a>` : ''}
                                ${isAdmin ? `<button data-promote="${f.id}" style="background:#bccfde;color:#1e293b;border:none;border-radius:3px;padding:0 4px;font-size:.6rem;cursor:pointer;" title="加入公司知识库">↑</button>` : ''}
                            </span>
                        </li>`
                    ).join('');
                }
                if (regular.length > 0) {
                    html += '<div style="font-size:.7rem;color:var(--card-muted);margin-bottom:4px;margin-top:'+(skills.length>0?'6px':'0')+';">📄 文档</div>';
                    html += regular.slice(0,6).map(f =>
                        `<li style="padding:2px 4px;font-size:.72rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">📄 ${escapeHtml(f.original_name || f.filename)}</li>`
                    ).join('');
                }
                myFiles.innerHTML = html || '<li style="color:var(--card-muted);font-size:.75rem;">暂无文件</li>';

                // Wire promote buttons for admin
                if (isAdmin) {
                    document.querySelectorAll('#sidebarKnowledgeFiles [data-promote]').forEach(btn => {
                        btn.onclick = async (e) => {
                            e.stopPropagation();
                            const id = btn.dataset.promote;
                            try {
                                const res = await fetch('/admin/promote_to_company/' + id, { method: 'POST', credentials: 'include' });
                const data = await res.json();
                content.setAttribute('data-audit-run', data.audit_run_id || '');
                                if (res.ok) { showToast(data.message || '已加入公司知识库', 'success'); loadSidebarKnowledge(); }
                                else showToast('操作失败，请重试', 'error');
                            } catch(err) { showToast('网络错误', 'error'); }
                        };
                    });
                }
            }
            if (coFiles) {
                if (coList.length) {
                    coFiles.innerHTML = coList.slice(0,8).map(f => `
                        <li style="padding:2px 4px;font-size:.72rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;justify-content:space-between;align-items:center;">
                            <span style="flex:1;overflow:hidden;text-overflow:ellipsis;" title="${escapeHtml(f.filename)}">🏢 ${escapeHtml(f.filename)}${f.has_skill ? ' 🧠' : ''}</span>
                            ${f.has_skill ? `<a href="/company_kb/skill/${f.id}" download style="text-decoration:none;font-size:.6rem;color:#16a34a;flex-shrink:0;margin-left:2px;" title="下载公司技能文件">📥</a>` : ''}
                        </li>
                    `).join('');
                } else {
                    coFiles.innerHTML = '<li style="color:var(--card-muted);font-size:.75rem;">暂无文件';
                }
            }
        } catch(e) {}
    }

    async function loadSidebarRecycle(cachedData) {
        const stats = document.getElementById('sidebarRecycleStats');
        if (!stats) return;
        try {
            const data = cachedData || await (await fetch('/get_recycle_bin', { credentials: 'include' })).json();
            // Combine all item arrays and normalize source keys for sidebar display
            const rawItems = [
                ...(data.chat_items || []).map(i => ({...i, _source: 'chat'})),
                ...(data.project_items || []).map(i => ({...i, _source: 'user_file'})),
                ...(data.folder_items || []).map(i => ({...i, _source: 'user_file', file_size: 0})),
                ...(data.kb_items || []).map(i => ({...i, _source: i.source || 'knowledge_lab'}))
            ];
            const items = rawItems.map(i => ({...i, source: i._source}));
            // Source breakdown
            const sources = {chat:0, user_file:0, knowledge_lab:0, company_kb:0};
            let totalSize = 0;
            items.forEach(i => { if (sources[i.source] !== undefined) sources[i.source]++; totalSize += (i.file_size||0); });
            const sizeMB = (totalSize/1024/1024).toFixed(1);
            stats.innerHTML = `<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:4px;">
                <div style="background:var(--card-bg);border-radius:4px;padding:4px;text-align:center;"><strong>${items.length}</strong><br><small>项</small></div>
                <div style="background:var(--card-bg);border-radius:4px;padding:4px;text-align:center;"><strong>${sizeMB}MB</strong><br><small>占用</small></div>
            </div>
            <div style="font-size:.65rem;margin-top:4px;display:grid;gap:2px;">
                <div>💬 对话: ${sources.chat}项</div>
                <div>📁 文件: ${sources.user_file}项</div>
                <div>🧪 知识库: ${sources.knowledge_lab}项</div>
                <div>🏢 公司: ${sources.company_kb}项</div>
            </div>`;

            // Storage breakdown by source
            const sourceDets = [
                {key:'chat',label:'对话',size:0},{key:'user_file',label:'文件',size:0},
                {key:'knowledge_lab',label:'知识库',size:0},{key:'company_kb',label:'公司',size:0}
            ];
            items.forEach(i => {
                const sd = sourceDets.find(s => s.key === i.source);
                if (sd) sd.size += (i.file_size || 0);
            });
            const maxSrcSize = Math.max(1, ...sourceDets.map(s=>s.size));
            stats.innerHTML += '<div style="margin-top:6px;font-size:.65rem;">' +
                sourceDets.filter(s => s.size > 0).map(s => {
                    const w = Math.max(2, Math.round(s.size/maxSrcSize*100));
                    return `<div style="display:flex;align-items:center;gap:4px;margin-bottom:2px;">
                        <span style="width:28px;">${s.label}</span>
                        <div style="flex:1;height:6px;background:var(--card-border);border-radius:3px;overflow:hidden;">
                            <div style="width:${w}%;height:100%;background:#5a7c9b;border-radius:3px;"></div>
                        </div>
                        <span style="color:var(--card-muted);">${(s.size/1024).toFixed(0)}KB</span>
                    </div>`;
                }).join('') + '</div>';

            // Age-based cleanup buttons
            const now = Date.now();
            const ages = {day7:0, day30:0};
            items.forEach(i => {
                const d = i.deleted_at || i.created_at || i.timestamp;
                if (d) { const age = (now - new Date(d).getTime()) / 86400000; if (age > 7) ages.day7++; if (age > 30) ages.day30++; }
            });
            if (ages.day7 > 0 || ages.day30 > 0) {
                stats.innerHTML += `<div style="margin-top:8px;display:flex;gap:4px;">
                    ${ages.day7 > 0 ? '<button class="recycle-age-btn" style="flex:1;padding:3px 6px;font-size:.65rem;background:var(--card-bg);border:1px solid var(--card-border);border-radius:4px;cursor:pointer;">清理7天前('+ages.day7+')</button>' : ''}
                    ${ages.day30 > 0 ? '<button class="recycle-age-btn" style="flex:1;padding:3px 6px;font-size:.65rem;background:#fef2f2;border:1px solid #fca5a5;border-radius:4px;cursor:pointer;color:#dc2626;">清理30天前('+ages.day30+')</button>' : ''}
                </div>`;
            }

            // Filter buttons
            let currentFilter = 'all';
            document.querySelectorAll('.recycle-filter').forEach(btn => {
                btn.onclick = () => {
                    document.querySelectorAll('.recycle-filter').forEach(b => b.style.background = 'var(--card-bg)');
                    btn.style.background = '#bccfde';
                    currentFilter = btn.dataset.source;
                    document.getElementById('sidebarRestoreAllBtn').textContent = currentFilter === 'all' ? '♻️ 恢复全部' : '♻️ 恢复筛选结果';
                    document.getElementById('sidebarEmptyAllBtn').textContent = currentFilter === 'all' ? '🗑️ 清空全部' : '🗑️ 清空筛选结果';
                };
            });
        } catch(e) { stats.textContent = '加载失败'; }
    }

    async function loadSidebarStats(cachedStats) {
        const activity = document.getElementById('sidebarStatsActivity');
        const system = document.getElementById('sidebarStatsSystem');
        if (!activity) return;
        try {
            const s = cachedStats || await (await fetch('/admin/analytics', { credentials: 'include' })).json();

            // Activity insight (distinct from main panel's card grid)
            const daysWithActivity = (s.messages_per_day || []).filter(d => d.count > 0);
            const activeLabel = daysWithActivity.length >= 5 ? '非常活跃' : daysWithActivity.length >= 2 ? '正常使用' : '需要更多互动';
            const trendLabel = daysWithActivity.length >= 2 && daysWithActivity[daysWithActivity.length-1].count > (daysWithActivity[0]?.count||0) ? '📈 上升' : '📉 平稳';

            activity.innerHTML = `<div style="background:var(--card-bg);border-radius:6px;padding:8px;margin-bottom:6px;">
                <div style="font-weight:500;">活跃度: ${activeLabel}</div>
                <div style="font-size:.7rem;color:var(--card-muted);">趋势: ${trendLabel} | 近7天有${daysWithActivity.length}天活跃</div>
            </div>`;

            // Messages per day mini-spark
            if (daysWithActivity.length) {
                const max = Math.max(...daysWithActivity.map(d=>d.count),1);
                const bars = daysWithActivity.map(d => {
                    const h = Math.max(2, Math.round(d.count/max*40));
                    return `<div style="width:10px;height:${h}px;background:#5a7c9b;border-radius:2px 2px 0 0;" title="${d.day}: ${d.count}条"></div>`;
                }).join('');
                activity.innerHTML += `<div style="display:flex;align-items:flex-end;gap:3px;height:45px;margin-top:6px;">${bars}</div>
                    <div style="font-size:.6rem;color:var(--card-muted);margin-top:2px;">近7天消息趋势</div>`;
            }

            // System resources (admin-only) + headroom savings
            if (s.is_admin_view && system) {
                getSystemResources(system);
            } else if (system) {
                system.innerHTML = `<div style="background:var(--card-bg);border-radius:6px;padding:6px;margin-top:4px;">
                    <div style="font-size:.7rem;">总存储: ${s.storage_mb}MB</div>
                    <div style="font-size:.7rem;">总文件: ${s.total_files}</div>
                </div>`;
            }

            // Top files & recent activity (from session data)
            try {
                const sessionsRes = await fetch('/get_sessions', { credentials: 'include' });
                const sessionsData = await sessionsRes.json();
                const sessions = sessionsData.sessions || [];
                if (sessions.length && activity) {
                    const recent = sessions.slice(0, 3);
                    activity.innerHTML += '<hr style="margin:10px 0;"><div style="font-size:.7rem;color:var(--card-muted);margin-bottom:4px;">最近活跃对话</div>' +
                        recent.map(s => `<div style="font-size:.72rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:2px 0;">${escapeHtml(s.title || '新对话')}</div>`).join('');
                }
            } catch(e) {}

            // Headroom savings indicator
            try {
                if (typeof headroom_compress === 'function') {
                    if (activity) activity.innerHTML += '<div style="margin-top:8px;font-size:.65rem;color:#22c55e;">🧠 Headroom 压缩就绪</div>';
                }
            } catch(e) {}
        } catch(e) {}
    }

    async function getSystemResources(container) {
        try {
            const [auditRes, sessionsRes] = await Promise.all([
                fetch('/admin/audit_log?page=1', { credentials: 'include' }),
                fetch('/get_sessions', { credentials: 'include' })
            ]);
            const audit = await auditRes.json();
            const sessions = await sessionsRes.json();
            container.innerHTML = `<div style="background:var(--card-bg);border-radius:6px;padding:6px;margin-top:4px;">
                <div style="font-size:.7rem;">审计记录: ${audit.total || 0}条</div>
                <div style="font-size:.7rem;">活跃会话: ${(sessions.sessions||[]).length}个</div>
            </div>`;

            // Admin: show user role assignment + pending deletions
            const userRolesDiv = document.getElementById('sidebarUserRoles');
            if (userRolesDiv) {
                userRolesDiv.style.display = '';
                userRolesDiv.innerHTML = '<hr style="margin:10px 0;"><h4 style="font-size:.78rem;color:var(--card-muted);margin-bottom:6px;">👥 角色管理</h4><div id="userRoleList" style="font-size:.7rem;">加载中...</div>';
                try {
                    const [emailsRes, delRes] = await Promise.all([
                        fetch('/admin/user_emails', { credentials: 'include' }),
                        fetch('/admin/pending_deletions', { credentials: 'include' })
                    ]);
                    const users = await emailsRes.json();
                    const delData = await delRes.json();
                    const pendingDeletions = delData.users || [];
                    const userList = (users.users || []).filter(u => u.role !== 'admin');
                    const roleList = document.getElementById('userRoleList');
                    if (!roleList) return;
                    let html = '';
                    // Pending deletions first
                    if (pendingDeletions.length) {
                        html += '<div style="margin-bottom:4px;font-size:.65rem;color:#dc2626;">⚠️ 待审核删除</div>';
                        pendingDeletions.forEach(u => {
                            html += `<div style="display:flex;align-items:center;justify-content:space-between;padding:2px 0;gap:4px;background:#fef2f2;border-radius:4px;padding:3px 6px;margin-bottom:2px;">
                                <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;">${escapeHtml(u.username)}</span>
                                <button class="approve-delete-btn" data-user="${escapeHtml(u.username)}" style="background:#16a34a;color:white;border:none;border-radius:3px;padding:1px 6px;font-size:.6rem;cursor:pointer;">批准</button>
                            </div>`;
                        });
                    }
                    html += userList.map(u => {
                        const isAuditor = u.is_auditor === true;
                        return `<div style="display:flex;align-items:center;justify-content:space-between;padding:2px 0;gap:4px;">
                            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;" title="${escapeHtml(u.username)} (${escapeHtml(u.role)})">${escapeHtml(u.username)}</span>
                            <select class="user-role-select" data-user="${escapeHtml(u.username)}" style="font-size:.65rem;padding:1px 3px;border-radius:3px;border:1px solid var(--card-border);width:72px;">
                                <option value="user" ${isAuditor ? '' : 'selected'}>user</option>
                                <option value="auditor" ${isAuditor ? 'selected' : ''}>auditor</option>
                            </select>
                        </div>`;
                    }).join('');
                    roleList.innerHTML = html;
                    // Wire role change
                    document.querySelectorAll('.user-role-select').forEach(sel => {
                        sel.onchange = async () => {
                            const username = sel.dataset.user;
                            const role = sel.value;
                            try {
                                const res = await fetch('/admin/role', { method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include', body:JSON.stringify({username,role}) });
                                if (res.ok) {
                                    // Update local sessionStorage if editing self
                                    if (username === sessionStorage.getItem('username')) {
                                        sessionStorage.setItem('is_auditor', role === 'auditor' ? '1' : '0');
                                    }
                                } else {
                                    const err = await res.json().catch(()=>({}));
                                    showToast(err.error || '更新失败', 'error');
                                    // Revert dropdown to previous state
                                    location.reload();
                                }
                            } catch(e) {}
                        };
                    });
                    // Wire approve-delete buttons
                    document.querySelectorAll('.approve-delete-btn').forEach(btn => {
                        btn.onclick = async () => {
                            const username = btn.dataset.user;
                            if (!confirm('批准 ' + username + ' 的账户删除申请？系统将发送验证码给该用户。')) return;
                            btn.disabled = true; btn.textContent = '...';
                            try {
                                const res = await fetch('/admin/approve_delete/' + encodeURIComponent(username), { method:'POST', credentials:'include' });
                                const d = await res.json();
                                if (res.ok) showToast(d.hint || '已批准', 'success');
                                else showToast(d.error || '操作失败', 'error');
                            } catch(_) { showToast('网络错误', 'error'); }
                            btn.disabled = false; btn.textContent = '批准';
                        };
                    });
                } catch(e) { console.error('Role management load error:', e); const rl = document.getElementById('userRoleList'); if (rl) rl.textContent = '加载失败'; }
            }
        } catch(e) { container.textContent = ''; }
    }

    // Audit sidebar buttons
    const runAuditBtn = document.getElementById('sidebarRunAuditBtn');
    const quickMergeBtn = document.getElementById('sidebarAuditQuickMergeBtn');
    const quickArchiveBtn = document.getElementById('sidebarAuditQuickArchiveBtn');
    const auditStatsEl = document.getElementById('sidebarAuditStats');
    if (runAuditBtn) runAuditBtn.onclick = () => {
        document.getElementById('analyticsTabBtn')?.click();
        const details = document.getElementById('skillAuditDetails');
        if (details) details.open = true;
    };
    if (quickMergeBtn) quickMergeBtn.onclick = () => {
        const firstPair = document.querySelector('.audit-pair');
        if (firstPair) { firstPair.click(); firstPair.scrollIntoView({behavior:'smooth'}); }
        else showToast('没有可合并的技能对', 'info');
    };
    if (quickArchiveBtn) quickArchiveBtn.onclick = async () => {
        const unusedBtns = document.querySelectorAll('.audit-archive-btn');
        if (unusedBtns.length === 0) { showToast('没有可清理的技能', 'info'); return; }
        if (!confirm(`确定批量移除全部 ${unusedBtns.length} 个未使用技能的标签？`)) return;
        for (const btn of unusedBtns) {
            btn.click();
            await new Promise(r => setTimeout(r, 300));
        }
        showToast('批量清理完成', 'success');
    };

    // Export stats button
    const exportStatsBtn = document.getElementById('sidebarExportStatsBtn');
    if (exportStatsBtn) exportStatsBtn.onclick = async () => {
        try {
            const res = await fetch('/admin/analytics', { credentials: 'include' });
            const data = await res.json();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'stats_report.json';
            a.click();
            showToast('统计报告已下载', 'success', 2000);
        } catch(e) { showToast('导出失败', 'error'); }
    };

    async function loadSidebarDb(cachedData) {
        const select = document.getElementById('sidebarDbTableSelect');
        const info = document.getElementById('sidebarDbTableInfo');
        const overview = document.getElementById('sidebarDbOverview');
        if (!select) return;
        try {
            const data = cachedData || await (await fetch('/admin/db_tables', { credentials: 'include' })).json();
            const tables = data.tables || [];
            select.innerHTML = tables.map(t => `<option value="${t}">${t}</option>`).join('');

            // Table overview with row counts (fetch ALL table sizes)
            if (overview) {
                overview.innerHTML = '<div style="padding:4px 0;">加载表信息...</div>';
                const tableInfos = [];
                try {
                    const ovRes = await fetch('/admin/db_tables_overview', { credentials: 'include' });
                    const ovData = await ovRes.json();
                    if (ovData.success) {
                        for (const t of (ovData.tables || [])) {
                            tableInfos.push({ name: t.table_name, rows: t.row_count });
                        }
                    } else {
                        throw new Error('Batch query failed');
                    }
                } catch(e) {
                    // Server-side already handles fallback; no client-side fallback
                }
                const totalRows = tableInfos.reduce((s,i) => s + Math.max(0,i.rows), 0);
                overview.innerHTML = '<div style="font-size:.65rem;color:var(--card-muted);margin-bottom:4px;">预估总行数: ' + totalRows.toLocaleString() + '</div>' +
                    tableInfos.map(ti =>
                        `<div style="padding:2px 0;display:flex;justify-content:space-between;cursor:pointer;font-size:.7rem;"
                              onclick="document.getElementById('sidebarDbTableSelect').value='${ti.name}'; document.getElementById('sidebarDbTableSelect').dispatchEvent(new Event('change'));">
                            <span>${ti.name}</span>
                            <span style="color:var(--card-muted);">${ti.rows >= 0 ? ti.rows.toLocaleString()+'行' : '—'}</span>
                        </div>`
                    ).join('');
            }
            // Wire to main panel
            select.onchange = () => {
                const mainSelect = document.getElementById('dbTableSelect');
                if (mainSelect) { mainSelect.value = select.value; mainSelect.dispatchEvent(new Event('change')); }
                if (info) info.textContent = '已选择: ' + select.value;
            };
        } catch(e) {}

        // Schema view button
        const schemaBtn = document.getElementById('sidebarDbSchemaBtn');
        if (schemaBtn) schemaBtn.onclick = async () => {
            const table = select.value;
            try {
                const res = await fetch(`/admin/db_schema?table=${table}`, { credentials: 'include' });
                const data = await res.json();
                const cols = data.columns || [];
                const tableInfo = document.getElementById('sidebarDbTableInfo');
                if (tableInfo) tableInfo.innerHTML = `<strong>${table}</strong> (${cols.length}列)<br>` +
                    cols.map(c => `<span style="font-size:.65rem;">${c.column_name} <span style="color:var(--card-muted);">${c.data_type}</span></span>`).join('<br>');
            } catch(e) {}
        };

        // Export buttons
        const csvBtn = document.getElementById('sidebarDbExportCsvBtn');
        const jsonBtn = document.getElementById('sidebarDbExportJsonBtn');
        const exportData = async (format) => {
            const table = select.value;
            try {
                const res = await fetch(`/admin/db_data?table=${table}&page=1&per_page=500`, { credentials: 'include' });
                const data = await res.json();
                const rows = data.rows || [];
                if (format === 'json') {
                    const blob = new Blob([JSON.stringify(rows, null, 2)], { type: 'application/json' });
                    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `${table}.json`; a.click();
                } else {
                    if (!rows.length) { showToast('无数据可导出', 'error'); return; }
                    const cols = Object.keys(rows[0]);
                    let csv = cols.join(',') + '\n' + rows.map(r => cols.map(c => '"' + String(r[c]||'').replace(/"/g,'""') + '"').join(',')).join('\n');
                    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' });
                    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `${table}.csv`; a.click();
                }
                showToast(`已导出 ${rows.length} 行`, 'success', 2000);
            } catch(e) { showToast('导出失败', 'error'); }
        };
        if (csvBtn) csvBtn.onclick = () => exportData('csv');
        if (jsonBtn) jsonBtn.onclick = () => exportData('json');
    }

    function syncActiveTabWithView() {
        try {
            const tabMap = [
                { panel: chatInterface, tab: chatTab },
                { panel: adminPanel, tab: adminTab },
                { panel: recycleBinPanel, tab: recycleBinTab },
                { panel: databasePanel, tab: databaseTab },
                { panel: knowledgeLabPanel, tab: knowledgeLabTab },
                { panel: wikiPanel, tab: wikiTab },
                { panel: timelinePanel, tab: timelineTabBtn },
                { panel: analyticsPanel, tab: analyticsTabBtn },
                { panel: reviewPanel, tab: reviewTabBtn },
            ];
            tabMap.forEach(({ tab }) => tab?.classList.remove('active'));
            for (const { panel, tab } of tabMap) {
                if (panel && panel.style.display !== 'none') {
                    tab?.classList.add('active');
                    break;
                }
            }
        } catch(_) { /* panels not yet initialized — skip */ }
    }

    async function updateProjectTabVisibility() {
        // Always show project/admin tabs — server handles access control
        const tabBar = document.getElementById('tabBar');
        const adminTab = document.getElementById('adminTabBtn');
        const recycleTab = document.getElementById('recycleBinTabBtn');
        const analyticsTab = document.getElementById('analyticsTabBtn');
        const timelineTab = document.getElementById('timelineTabBtn');

        if (tabBar) tabBar.style.display = 'flex';
        if (adminTab) adminTab.style.display = 'inline-block';
        if (recycleTab) recycleTab.style.display = 'inline-block';
        if (analyticsTab) analyticsTab.style.display = 'inline-block';
        if (timelineTab) timelineTab.style.display = currentProjectId ? 'inline-block' : 'none';
    }

    // ======================== Sidebar Tab Switching ========================
    function switchSidebarPane(tabName) {
        const panes = document.querySelectorAll('.sidebar-tab-pane');
        panes.forEach(p => p.style.display = 'none');
        const target = document.getElementById('sidebar-' + tabName + '-pane');
        if (target) target.style.display = '';
    }

    // Helper: hide all tab panels, then show only the active one
    function switchToPanel(activeId) {
        const allPanels = ['chatInterface','adminPanel','recycleBinPanel','databasePanel','knowledgeLabPanel','wikiPanel','timelinePanel','analyticsPanel','reviewPanel','templatesPanel','casesPanel'];
        allPanels.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = (id === activeId) ? (id === 'chatInterface' ? 'flex' : 'block') : 'none';
        });
    }

    // Tab event listeners
    // ======================== Tab Persistence ========================
    let _programmaticTabSwitch = false;
    function saveActiveTab(tabName) {
        if (_programmaticTabSwitch) return;
        localStorage.setItem('zlai_activeTab', tabName);
    }
    function restoreActiveTab() {
        const saved = localStorage.getItem('zlai_activeTab');
        const justLoggedIn = sessionStorage.getItem('just_logged_in') === '1';
        if (justLoggedIn) {
            sessionStorage.removeItem('just_logged_in');
            if (!saved) {
                const isAdmin = sessionStorage.getItem('isAdmin') === 'true';
                return isAdmin ? 'projects' : 'chat';
            }
        }
        return saved || 'chat';
    }

    // ======================== Sub-Tab Navigation ========================
    const SUB_TAB_GROUPS = { 'projects': 'adminSubTabs', 'knowledge': 'knowledgeSubTabs', 'stats': 'analyticsSubTabs' };

    function showSubTabBar(mainTabName) {
        document.querySelectorAll('.sub-tab-bar').forEach(b => b.style.display = 'none');
        const barId = SUB_TAB_GROUPS[mainTabName];
        if (barId) { const bar = document.getElementById(barId); if (bar) bar.style.display = ''; }
    }

    function resetSubTabs(barId) {
        const bar = document.getElementById(barId);
        if (!bar) return;
        bar.querySelectorAll('.sub-tab-btn').forEach(b => b.classList.remove('active'));
        const firstBtn = bar.querySelector('.sub-tab-btn');
        if (firstBtn) firstBtn.classList.add('active');
    }

    function loadSubTabContent(panelId) {
        switch (panelId) {
            case 'timelinePanel':
                if (typeof loadTimelinePanel === 'function') loadTimelinePanel();
                break;
            case 'wikiPanel':
                if (window._loadWiki && typeof window._loadWiki === 'function') window._loadWiki('');
                break;
            case 'reviewPanel':
                if (window._initReviewPanel && typeof window._initReviewPanel === 'function') window._initReviewPanel();
                break;
            case 'databasePanel':
                (async () => {
                    try {
                        const res = await fetch('/admin/db_tables', { credentials: 'include' });
                        const data = await res.json();
                        if (typeof loadSidebarDb === 'function') loadSidebarDb(data);
                        if (typeof loadDatabaseData === 'function') loadDatabaseData(data);
                    } catch(e) { console.warn('Database sub-tab load error:', e); }
                })();
                break;
        }
    }

    document.addEventListener('click', function(e) {
        const btn = e.target.closest('.sub-tab-btn');
        if (!btn) return;
        const panelId = btn.getAttribute('data-panel');
        const sidebar = btn.getAttribute('data-sidebar');
        if (!panelId) return;
        const bar = btn.closest('.sub-tab-bar');
        if (bar) bar.querySelectorAll('.sub-tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        switchToPanel(panelId);
        if (sidebar) switchSidebarPane(sidebar);
        loadSubTabContent(panelId);
    });

    if (chatTab && adminTab) {
        chatTab.onclick = () => {
            showSubTabBar('');
            saveActiveTab('chat');
            switchToPanel('chatInterface');
            switchSidebarPane('chat');
            toggleQuickLinksButton(false);
            syncActiveTabWithView();
            loadHistoryList();
            setTimeout(() => scrollToBottom(), 100);
            startRealtimePoll(currentProjectId || null);
        };
        const quickLinksBtn = document.createElement('button');
        quickLinksBtn.id = 'fixedQuickLinksBtn';
        quickLinksBtn.className = 'fixed-quick-links-btn';
        quickLinksBtn.innerHTML = '🔗 快捷查询';
        quickLinksBtn.style.display = 'none';
        document.body.appendChild(quickLinksBtn);
        function toggleQuickLinksButton(show) { quickLinksBtn.style.display = show ? 'block' : 'none'; }
        quickLinksBtn.onclick = () => {
            if (currentProjectId) {
                const projectName = document.querySelector('#fileExplorerContent h2')?.innerText.split(' ')[0] || '项目';
                showQuickLinksModal(currentProjectId, projectName);
            } else {
                showQuickLinksModal();
            }
        };
        adminTab.onclick = async () => {
            stopRealtimePoll();
            saveActiveTab('projects');
            showSubTabBar('projects');
            resetSubTabs('adminSubTabs');
            const res = await fetch('/admin/projects', { credentials: 'include' });
            if (!res.ok) { alert('加载项目失败，请刷新重试'); return; }
            const data = await res.json();
            const isAdmin = sessionStorage.getItem('isAdmin') === 'true';
            if (!data.has_projects && !isAdmin) {
                alert('您尚未加入任何项目。请联系管理员将您添加到项目中。');
                return;
            }
            switchToPanel('adminPanel');
            switchSidebarPane('projects');
            loadSidebarProjects(data);  // reuse fetched data
            toggleQuickLinksButton(true);
            if (typeof loadProjects === 'function') loadProjects(data);  // reuse
            syncActiveTabWithView();
        };
    }
    if (recycleBinTab) {
        recycleBinTab.onclick = async () => {
            stopRealtimePoll();
            saveActiveTab('recycle');
            showSubTabBar('');
            switchToPanel('recycleBinPanel');
            switchSidebarPane('recycle');
            toggleQuickLinksButton(false);
            const res = await fetch('/get_recycle_bin', { credentials: 'include' });
            if (res.ok) {
                const data = await res.json();
                loadSidebarRecycle(data);
                if (typeof loadRecycleBin === 'function') loadRecycleBin(data);
            }
            syncActiveTabWithView();
        };
    }
    if (databaseTab) {
        databaseTab.onclick = async () => {
            stopRealtimePoll();
            saveActiveTab('db');
            showSubTabBar('');
            switchToPanel('databasePanel');
            switchSidebarPane('db');
            const tablesRes = await fetch('/admin/db_tables', { credentials: 'include' });
            const tablesData = await tablesRes.json();
            loadSidebarDb(tablesData);
            toggleQuickLinksButton(false);
            if (typeof loadDatabaseData === 'function') loadDatabaseData(tablesData);
            syncActiveTabWithView();
        };
    }
    // Skill audit workspace (loaded inline in config panel)
    async function loadSkillAuditWorkspace() {
            const content = document.getElementById('skillAuditContent');
            content.innerHTML = '<p style="text-align:center;padding:40px;">🧠 AI正在分析技能库...</p>';
            try {
                const res = await fetch('/admin/skill_audit', { credentials: 'include' });
                if (!res.ok) throw new Error('HTTP ' + res.status);
                const data = await res.json();

                if (!data.total_skills && !data.duplicates?.length && !data.unused?.length) {
                    content.innerHTML = '<p style="text-align:center;padding:40px;color:var(--card-muted);">暂无技能数据。<br>请先在知识库中上传文件并提取技能。</p>';
                    return;
                }

                let html = '<div style="margin-bottom:16px;display:flex;gap:8px;align-items:center;">';
                html += '<div style="background:var(--card-bg);border-radius:8px;padding:10px 16px;flex:1;">总技能: <strong>'+data.total_skills+'</strong></div>';
                html += '<div style="background:#fef2f2;border-radius:8px;padding:10px 16px;flex:1;">未使用: <strong style="color:#dc2626;">'+data.unused_count+'</strong></div>';
                html += '<div style="background:#eff6ff;border-radius:8px;padding:10px 16px;flex:1;">相似对: <strong style="color:#2563eb;">'+data.duplicate_pairs+'</strong>' + (data.analysis_skipped ? ' <span style="font-size:0.6rem;color:#f59e0b;" title="'+escapeHtml(data.analysis_note||'')+'">(分析中)</span>' : '') + '</div>';
                html += '</div>';
                if (data.analysis_skipped) {
                    html += '<div style="background:#fef3c7;border:1px solid #fcd34d;border-radius:6px;padding:6px 10px;margin-bottom:12px;font-size:0.7rem;color:#92400e;">⏳ '+escapeHtml(data.analysis_note||'相似度分析模型首次加载中(需下载约120MB)，请稍后刷新。基础统计已可用。')+'</div>';
                }

                // All skills listing
                if (data.skills?.length) {
                    const catIcons = {'规章':'📋','模板':'📄','项目经验':'📊','专家意见':'💡'};
                    html += '<h4 style="margin:16px 0 8px;">📦 所有技能</h4>';
                    html += '<div style="margin-bottom:12px;">';
                    data.skills.forEach(s => {
                        const icon = catIcons[s.category] || '📄';
                        const srcLabel = s.source === 'company' ? '🏢' : s.source === 'personal' ? '👤' : '📁';
                        html += `<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 10px;background:var(--card-bg);border-radius:4px;margin-bottom:3px;font-size:.78rem;">
                            <span>${icon} ${escapeHtml(s.name)} <small style="color:var(--card-muted);">${srcLabel} ${s.category||'通用'} · ${s.username||'—'}</small></span>
                            <small style="color:var(--card-muted);">使用${s.usage_count}次 · ${s.uploaded_at||''}</small>
                        </div>`;
                    });
                    html += '</div>';
                }

                // Side-by-side merge workspace
                if (data.duplicates?.length) {
                    html += '<h4 style="margin:16px 0 8px;">🔗 相似技能 — 点击对查看详情并决定</h4>';
                    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;">';
                    data.duplicates.slice(0,6).forEach((d, idx) => {
                        const aContent = (d.skill_a.summary || d.skill_a.name || '').substring(0, 500);
                        const bContent = (d.skill_b.summary || d.skill_b.name || '').substring(0, 500);
                        html += `<div class="audit-pair" data-idx="${idx}" data-a="${escapeHtml(aContent)}" data-b="${escapeHtml(bContent)}" style="grid-column:1/-1;background:var(--card-bg);border-radius:8px;padding:12px;cursor:pointer;border:2px solid transparent;transition:border-color .2s;">
                            <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:start;">
                                <div style="padding:8px;background:#f8fafc;border-radius:6px;">
                                    <strong>${escapeHtml(d.skill_a.name)}</strong>
                                    <small style="color:var(--card-muted);display:block;">${d.skill_a.owner}</small>
                                </div>
                                <div style="text-align:center;font-weight:600;color:#2563eb;padding:8px;">${d.similarity}%</div>
                                <div style="padding:8px;background:#f8fafc;border-radius:6px;">
                                    <strong>${escapeHtml(d.skill_b.name)}</strong>
                                    <small style="color:var(--card-muted);display:block;">${d.skill_b.owner}</small>
                                </div>
                            </div>
                            <div class="audit-preview" style="display:none;margin-top:8px;display:none;grid-template-columns:1fr 1fr;gap:8px;">
                                <div style="padding:8px;background:#f0fdf4;border-radius:6px;font-size:0.72rem;white-space:pre-wrap;max-height:200px;overflow-y:auto;">${escapeHtml(aContent)}</div>
                                <div style="padding:8px;background:#fef2f2;border-radius:6px;font-size:0.72rem;white-space:pre-wrap;max-height:200px;overflow-y:auto;">${escapeHtml(bContent)}</div>
                            </div>
                        </div>`;
                    });
                    html += '</div>';
                    html += '<div id="auditActionBar" style="display:none;gap:8px;padding:12px;background:var(--card-bg);border-radius:8px;margin-bottom:16px;">';
                    html += '<button class="audit-merge-btn" style="background:#2563eb;color:white;border:none;border-radius:6px;padding:8px 16px;cursor:pointer;">🔗 合并这两个技能</button>';
                    html += '<button class="audit-keep-a-btn" style="background:#f8fafc;border:1px solid var(--card-border);border-radius:6px;padding:8px 16px;cursor:pointer;">保留A，删除B</button>';
                    html += '<button class="audit-keep-b-btn" style="background:#f8fafc;border:1px solid var(--card-border);border-radius:6px;padding:8px 16px;cursor:pointer;">保留B，删除A</button>';
                    html += '<span id="auditActionStatus" style="font-size:.8rem;color:var(--card-muted);margin-left:8px;"></span></div>';
                }

                // Unused skills
                if (data.unused?.length) {
                    html += '<h4 style="margin:16px 0 8px;">⚠️ 长期未使用</h4>';
                    data.unused.slice(0,8).forEach(u => {
                        html += `<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:var(--card-bg);border-radius:6px;margin-bottom:4px;font-size:.82rem;">
                            <span>🗑️ ${escapeHtml(u.name)} <small style="color:var(--card-muted);">(${u.owner}) — ${u.days_since_upload}天</small></span>
                            <button class="audit-archive-btn" data-id="${u.skill_id}" data-source="${u.source||'knowledge_lab'}" style="background:#ef4444;color:white;border:none;border-radius:4px;padding:4px 10px;font-size:.72rem;cursor:pointer;">移除技能</button>
                        </div>`;
                    });
                }

                // Promote candidates
                if (data.promote_candidates?.length) {
                    html += '<h4 style="margin:16px 0 8px;">⭐ 建议晋升到公司知识库</h4>';
                    data.promote_candidates.slice(0,5).forEach(p => {
                        html += `<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:var(--card-bg);border-radius:6px;margin-bottom:4px;font-size:.82rem;">
                            <span>📈 ${escapeHtml(p.name)} — ${p.user_count}位用户使用</span>
                            <span style="font-size:.7rem;color:var(--card-muted);">需手动操作</span>
                        </div>`;
                    });
                }
                html += '<div style="margin-top:16px;padding-top:12px;border-top:1px solid var(--card-border);display:flex;gap:6px;align-items:center;font-size:0.7rem;">';
                html += '<span style="color:var(--card-muted);">审计建议有帮助吗？</span>';
                html += '<button class="fb-btn" onclick="window.submitAuditFeedback(1,this)">👍 有帮助</button>';
                html += '<button class="fb-btn" onclick="window.submitAuditFeedback(-1,this)">👎 无帮助</button>';
                html += '</div>';
                content.innerHTML = html;

                // Wire archive buttons
                setTimeout(() => {
                    document.querySelectorAll('.audit-archive-btn').forEach(btn => {
                        btn.onclick = async () => {
                            if (!confirm('确定移除此技能标签？文件本身不会被删除。')) return;
                            btn.disabled = true; btn.textContent = '...';
                            try {
                                const res = await fetch('/admin/skill_archive/'+btn.dataset.id+'?source='+btn.dataset.source, {method:'POST',credentials:'include'});
                                if (res.ok) { btn.closest('div').remove(); showToast('已移除', 'success'); }
                            } catch(e) {}
                        };
                    });
                    // Wire audit pair clicks
                    content.querySelectorAll('.audit-pair').forEach(pair => {
                        pair.onclick = () => {
                            const idx = pair.dataset.idx;
                            const actionBar = content.querySelector('#auditActionBar');
                            const preview = pair.querySelector('.audit-preview');
                            const wasSelected = pair.style.borderColor === 'rgb(37,99,235)';
                            content.querySelectorAll('.audit-pair').forEach(p => { p.style.borderColor = 'transparent'; const pv = p.querySelector('.audit-preview'); if (pv) pv.style.display = 'none'; });
                            if (!wasSelected && actionBar) {
                                pair.style.borderColor = '#2563eb';
                                actionBar.style.display = 'flex';
                                actionBar.dataset.pairIdx = idx;
                                if (preview) preview.style.display = 'grid';
                            } else if (actionBar) {
                                actionBar.style.display = 'none';
                            }
                        };
                    });
                    // Wire merge button — actually call the backend
                    const mergeBtn = content.querySelector('.audit-merge-btn');
                    const keepABtn = content.querySelector('.audit-keep-a-btn');
                    const keepBBtn = content.querySelector('.audit-keep-b-btn');
                    const statusEl = document.getElementById('auditActionStatus');
                    const actionBar = content.querySelector('#auditActionBar');
                    if (data.duplicates) {
                        const getPair = () => {
                            const idx = parseInt(actionBar?.dataset?.pairIdx || '0');
                            return data.duplicates[idx];
                        };
                        const doMerge = async (action) => {
                            const pair = getPair();
                            if (!pair) return;
                            statusEl.textContent = '处理中...';
                            const a = pair.skill_a, b = pair.skill_b;
                            const source = pair.source || 'knowledge_lab';
                            const table = source === 'company_kb' ? 'company_knowledge_base' : 'knowledge_lab_files';
                            try {
                                if (action === 'merge') {
                                    const res = await fetch('/admin/skill_merge', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        credentials: 'include',
                                        body: JSON.stringify({ keep_id: a.id, merge_id: b.id, source: table })
                                    });
                                    const d = await res.json().catch(()=>({}));
                                    if (res.ok) { statusEl.textContent = '✅ ' + (d.status || '已合并'); showToast('技能合并成功', 'success'); }
                                    else { statusEl.textContent = '❌ ' + (d.error || '合并失败'); showToast(d.error || '合并失败', 'error'); }
                                } else {
                                    // keep_a = archive B, keep_b = archive A
                                    const archiveId = action === 'keep_a' ? b.id : a.id;
                                    const archiveSource = action === 'keep_a' ? (b.source || 'knowledge_lab') : (a.source || 'knowledge_lab');
                                    const res = await fetch('/admin/skill_archive/' + archiveId + '?source=' + archiveSource, { method: 'POST', credentials: 'include' });
                                    const which = action === 'keep_a' ? 'B' : 'A';
                                    if (res.ok) { statusEl.textContent = '✅ 已移除技能' + which; showToast('技能' + which + '已移除', 'success'); }
                                    else { statusEl.textContent = '❌ 移除失败'; }
                                }
                                // Refresh audit after 1.5s
                                setTimeout(() => loadSkillAuditWorkspace(), 1500);
                            } catch(e) { statusEl.textContent = '❌ 网络错误'; }
                        };
                        if (mergeBtn) mergeBtn.onclick = () => doMerge('merge');
                        if (keepABtn) keepABtn.onclick = () => doMerge('keep_a');
                        if (keepBBtn) keepBBtn.onclick = () => doMerge('keep_b');
                    }
                }, 100);
            } catch(e) {
                // M4 (FIX-016 后续): 区分错误码
                var sm = /HTTP (\d+)/.exec(String(e && e.message || e));
                var msg = '';
                if (sm) { var c = parseInt(sm[1],10); msg = c===401?'请先登录':(c===403?'权限不足':(c>=500?'服务器错误 ('+c+')':'请求失败 ('+c+')')); }
                else if (e && e.name==='TypeError') msg = '网络错误';
                content.innerHTML = '<p style="color:#ef4444;">加载失败' + (msg ? '：' + msg : '') + '</p>';
            }
        }

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
                            resultDiv.innerHTML = '<p>⏳ AI正在汇总数据并撰写报告，请稍候（约30-60秒）...</p>';
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
                            } catch(_) { resultDiv.innerHTML = '<p style="color:#dc2626;">❌ 网络错误</p>'; }
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
                        if (r.ok) { ttlDisplay.textContent = hrs; msgEl.innerHTML = '<span style="color:#22c55e;">✅ '+d.message+'</span>'; }
                        else msgEl.innerHTML = '<span style="color:#ef4444;">❌ '+(d.error||'失败')+'</span>';
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
                        if (r.ok) msgEl.innerHTML = '<span style="color:#22c55e;">✅ '+d.message+'</span>';
                        else msgEl.innerHTML = '<span style="color:#ef4444;">❌ '+(d.error||'失败')+'</span>';
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
        const auditLogBtn = document.getElementById('sidebarAuditLogBtn');
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

        // Knowledge graph sidebar buttons
        const lawImpactBtn = document.getElementById('sidebarLawImpactGraphBtn');
        if (lawImpactBtn) lawImpactBtn.onclick = () => _showKnowledgeGraphModal('law-impact');
        const globalCitBtn = document.getElementById('sidebarGlobalCitationGraphBtn');
        if (globalCitBtn) globalCitBtn.onclick = () => _showKnowledgeGraphModal('citation');

        function _showKnowledgeGraphModal(graphType) {
            var overlay = document.getElementById('kgOverlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = 'kgOverlay';
                overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.4);z-index:10000;display:flex;align-items:center;justify-content:center;';
                overlay.onclick = function(e) { if (e.target === overlay) { overlay.style.display = 'none'; overlay.innerHTML = ''; } };
                document.body.appendChild(overlay);
            }
            overlay.style.display = 'flex';
            overlay.innerHTML = '<div style="background:var(--card-bg);border-radius:12px;padding:20px;max-width:800px;width:90%;max-height:80vh;overflow-y:auto;box-shadow:0 8px 32px rgba(0,0,0,0.2);">' +
                '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">' +
                '<strong>' + (graphType === 'law-impact' ? '🏛️ 法规影响图谱' : '📄 全局引用图谱') + '</strong>' +
                '<button id="kgCloseBtn" style="background:none;border:none;font-size:1.2rem;cursor:pointer;">&times;</button>' +
                '</div>' +
                '<div id="kgGraphContainer" style="width:100%;height:450px;border:1px solid var(--card-border);border-radius:6px;overflow:hidden;">' +
                '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">加载中...</span>' +
                '</div>' +
                '</div>';
            document.getElementById('kgCloseBtn').onclick = function() { overlay.style.display = 'none'; overlay.innerHTML = ''; };

            _loadKnowledgeGraph(graphType, 'kgGraphContainer');
        }

        async function _loadKnowledgeGraph(graphType, containerId) {
            var container = document.getElementById(containerId);
            if (!container) return;
            try {
                var url, desc;
                if (graphType === 'law-impact') {
                    url = '/api/graph/law-impact?depth=3';
                    desc = '当前有效法规的影响传递链';
                } else {
                    url = '/api/graph/citation?path=招标投标法&depth=2';
                    desc = '文档实体引用网络';
                }
                var r = await fetch(url, { credentials: 'include' });
                var data = await r.json();
                if (!data.success || !data.nodes || !data.nodes.length) {
                    container.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">暂无数据</span>';
                    return;
                }
                container.style.position = 'relative';
                container.innerHTML = '';
                renderGraph(container, data);
            } catch (e) {
                console.error('Knowledge graph error:', e);
                container.innerHTML = '<span style="color:#ef4444;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">网络错误</span>';
            }
        }

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
                                ${p.status === 'archived' && p.archived_at ? `<br><small style="color:var(--card-muted);">归档于: ${new Date(p.archived_at).toLocaleString()}</small>` : ''}
                                ${p.deletion_scheduled_at ? `<br><small style="color:#ea580c;">⏳ 即将删除: ${new Date(p.deletion_scheduled_at).toLocaleString()}</small>` : ''}
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
                <button id="openProjectGrillBtn" class="file-btn" style="background:#fef2f2;color:#991b1b;border-color:#fca5a5;padding:6px 14px;font-size:0.78rem;white-space:nowrap;" title="为此项目创建质问模式，AI将模拟挑剔供应商拷问招标文件">🔥 质问</button>` : ''}
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
                </details>
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

        // Wire project sub-tab switching
        setTimeout(() => {
            // Sub-tab switching
            document.querySelectorAll('.project-sub-tab-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const tab = btn.getAttribute('data-tab');
                    document.querySelectorAll('.project-sub-tab-btn').forEach(b => { b.classList.remove('active'); b.style.cssText = b.style.cssText.replace(/color:[^;]+;/, ''); });
                    btn.classList.add('active');
                    document.getElementById('projectFilesTab').style.display = tab === 'files' ? '' : 'none';
                    document.getElementById('projectGraphTab').style.display = tab === 'graph' ? '' : 'none';
                });
            });

            // Graph type button wiring
            var collusionBtn = document.getElementById('projectCollusionGraphBtn');
            var complianceBtn = document.getElementById('projectComplianceGraphBtn');
            var citationBtn = document.getElementById('projectCitationGraphBtn');

            function switchGraphType(activeBtn, graphType) {
                var container = document.getElementById('projectGraphContainer');
                var statsEl = document.getElementById('projectGraphStats');
                var placeholder = document.getElementById('projectGraphPlaceholder');
                [collusionBtn, complianceBtn, citationBtn].forEach(function(b) {
                    if (b) { b.classList.remove('active'); b.style.background = 'var(--card-bg)'; b.style.color = ''; b.style.fontWeight = ''; }
                });
                if (activeBtn) { activeBtn.classList.add('active'); activeBtn.style.background = '#1e293b'; activeBtn.style.color = 'white'; activeBtn.style.fontWeight = '600'; }
                if (statsEl) statsEl.textContent = '';
                if (container) { container.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">加载中...</span>'; }
                if (placeholder) placeholder.style.display = 'none';
            }

            if (collusionBtn) collusionBtn.addEventListener('click', function() {
                switchGraphType(collusionBtn, 'collusion');
                loadProjectCollusionGraph();
            });
            if (complianceBtn) complianceBtn.addEventListener('click', function() {
                switchGraphType(complianceBtn, 'compliance');
                loadProjectComplianceGraph();
            });
            if (citationBtn) citationBtn.addEventListener('click', function() {
                switchGraphType(citationBtn, 'citation');
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
                            if (aiResult) aiResult.innerHTML = '<div style="color:#22c55e;text-align:center;padding:20px;">✅ 工作流已保存！<br>在输入框中描述你的需求，然后点击 <b>🔄 执行工作流</b></div>';
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

    // ======================== Member Management ========================
    async function showMemberManager(projectId) {
        try {
            const membersRes = await fetch(`/admin/projects/${projectId}/members`, { credentials: 'include' });
            if (!membersRes.ok) throw new Error('Failed to fetch members');
            const membersData = await membersRes.json();
            const isGlobalAdmin = sessionStorage.getItem('isAdmin') === 'true';
            const currentUserId = sessionStorage.getItem('user_id');
            let canManageMembers = isGlobalAdmin;
            if (!canManageMembers) {
                const currentUser = membersData.members.find(m => m.user_id === currentUserId);
                const role = currentUser ? currentUser.role : null;
                canManageMembers = role === 'manager';
            }

            let allUsers = [];
            if (isGlobalAdmin) {
                try {
                    const allUsersRes = await fetch(`/admin/projects/${projectId}/all_users`, { credentials: 'include' });
                    if (allUsersRes.ok) {
                        const allUsersData = await allUsersRes.json();
                        allUsers = allUsersData.users || [];
                    }
                } catch (err) { console.warn('Failed to fetch all users', err); }
            }

            function renderMembersList() {
                let html = '<ul style="list-style: none; padding: 0;">';
                const members = membersData.members || [];
                if (members.length === 0) {
                    html += '<li>暂无成员（项目创建者会自动成为管理员）。</li>';
                } else {
                    for (const m of members) {
                        if (m.role === 'admin') {
                            html += `<li><strong>${escapeHtml(m.username)}</strong> (全局管理员 - 不可编辑)</li>`;
                            continue;
                        }
                        html += `
                            <li style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span><strong>${escapeHtml(m.username)}</strong> (${m.role === 'manager' ? '经理' : '成员'})</span>
                                <div>
                        `;
                        if (isGlobalAdmin) {
                            html += `
                                <select class="change-role" data-user="${m.user_id}" data-role="${m.role}" style="margin-right: 8px;">
                                    <option value="member" ${m.role === 'member' ? 'selected' : ''}>成员</option>
                                    <option value="manager" ${m.role === 'manager' ? 'selected' : ''}>经理</option>
                                </select>
                            `;
                        }
                        if (isGlobalAdmin || (m.role !== 'admin' && canManageMembers)) {
                            html += `<button class="remove-member" data-user="${m.user_id}" style="background:#e74c3c; color:white; border:none; border-radius:4px; padding:2px 8px;">移除</button>`;
                        }
                        if (m.role === 'manager' && (isGlobalAdmin || canManageMembers)) {
                            html += `<button class="transfer-manager" data-user="${m.user_id}" data-name="${escapeHtml(m.username)}" style="background:#d97706; color:white; border:none; border-radius:4px; padding:2px 8px; margin-left:4px;" title="将经理权限转移给此成员（你将降为成员）">👑 转交</button>`;
                        }
                        html += `</div></li>`;
                    }
                }
                html += '</ul>';
                return html;
            }

            function renderAddableUsers(users, filter = '') {
                const filtered = filter ? users.filter(u => u.username.toLowerCase().includes(filter.toLowerCase())) : users;
                if (filtered.length === 0) return '<p>没有可添加的用户。</p>';
                let html = '<ul style="list-style: none; padding: 0;">';
                for (const u of filtered) {
                    if (isGlobalAdmin) {
                        html += `
                            <li style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span>${escapeHtml(u.username)}</span>
                                <div>
                                    <select class="role-select" data-user="${u.user_id}" style="margin-right: 8px;">
                                        <option value="member">成员</option>
                                        <option value="manager">经理</option>
                                    </select>
                                    <button class="add-member file-btn" data-user="${u.user_id}" data-name="${escapeHtml(u.username)}" style="padding: 4px 12px;">➕ 添加</button>
                                </div>
                            </li>
                        `;
                    } else {
                        html += `
                            <li style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span>${escapeHtml(u.username)}</span>
                                <button class="add-member file-btn" data-user="${u.user_id}" data-name="${escapeHtml(u.username)}" style="padding: 4px 12px;">➕ 添加</button>
                            </li>
                        `;
                    }
                }
                html += '</ul>';
                return html;
            }

            const modalHtml = `
                <div id="memberManagerModal" class="modal" style="display: block; z-index: 10000;">
                    <div class="modal-content" style="width: 700px; max-width: 90%;">
                        <span class="close" style="float: right; cursor: pointer;">&times;</span>
                        <h3>成员管理</h3>
                        <h4>现有成员</h4>
                        <div id="currentMembersList">${renderMembersList()}</div>
                        <hr>
                        <h4>添加新成员</h4>
                        <div style="margin: 10px 0;">
                            <input type="text" id="filterUsersInput" placeholder="按用户名筛选..." style="width: 100%; padding: 8px; border-radius: 4px;">
                        </div>
                        <div id="usersListContainer" style="max-height: 300px; overflow-y: auto;">
                            ${renderAddableUsers(allUsers)}
                        </div>
                        <div style="margin-top: 20px; text-align: right;">
                            <button id="closeMemberModalBtn" class="file-btn">关闭</button>
                        </div>
                    </div>
                </div>
            `;

            const existingModal = document.getElementById('memberManagerModal');
            if (existingModal) existingModal.remove();
            document.body.insertAdjacentHTML('beforeend', modalHtml);

            const modal = document.getElementById('memberManagerModal');
            const closeModal = () => modal.remove();

            modal.querySelector('.close').onclick = closeModal;
            modal.querySelector('#closeMemberModalBtn').onclick = closeModal;
            modal.onclick = (e) => { if (e.target === modal) closeModal(); };

            const escHandler = (e) => { if (e.key === 'Escape') { closeModal(); document.removeEventListener('keydown', escHandler); } };
            document.addEventListener('keydown', escHandler);

            function attachMemberActionHandlers() {
                modal.querySelectorAll('.change-role').forEach(select => {
                    select.onchange = async () => {
                        const userId = select.dataset.user;
                        const newRole = select.value;
                        if (!await confirm(`确定要将此成员的权限改为“${newRole === 'manager' ? '经理' : '成员'}”吗？`)) {
                            select.value = select.dataset.role;
                            return;
                        }
                        const res = await fetch(`/admin/projects/${projectId}/members/${userId}`, {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            credentials: 'include',
                            body: JSON.stringify({ role: newRole })
                        });
                        if (res.ok) {
                            closeModal();
                            showMemberManager(projectId);
                        } else {
                            const err = await res.json();
                            alert('更新失败: ' + (err.error || '未知错误'));
                            select.value = select.dataset.role;
                        }
                    };
                });

                modal.querySelectorAll('.remove-member').forEach(btn => {
                    btn.onclick = async () => {
                        const userId = btn.dataset.user;
                        if (!await confirm('确定要将此成员移出项目吗？')) return;
                        const res = await fetch(`/admin/projects/${projectId}/members/${userId}`, {
                            method: 'DELETE',
                            credentials: 'include'
                        });
                        if (res.ok) {
                            closeModal();
                            showMemberManager(projectId);
                        } else {
                            const err = await res.json();
                            alert('移除失败: ' + (err.error || '未知错误'));
                        }
                    };
                });
                modal.querySelectorAll('.transfer-manager').forEach(btn => {
                    btn.onclick = async () => {
                        const userId = btn.dataset.user;
                        const userName = btn.dataset.name;
                        if (!await confirm(`确定要将经理权限转移给 ${userName} 吗？\\n\\n你将成为普通成员，${userName} 将成为经理。`)) return;
                        btn.disabled = true; btn.textContent = '...';
                        const res = await fetch(`/admin/projects/${projectId}/transfer_manager/${userId}`, {
                            method: 'POST',
                            credentials: 'include'
                        });
                        if (res.ok) {
                            showToast(`经理权限已转移给 ${userName}`, 'success');
                            closeModal();
                            showMemberManager(projectId);
                        } else {
                            const err = await res.json();
                            alert('转移失败: ' + (err.error || '未知错误'));
                            btn.disabled = false; btn.textContent = '👑 转交';
                        }
                    };
                });
            }

            function attachAddHandlers() {
                modal.querySelectorAll('.add-member').forEach(btn => {
                    btn.onclick = async () => {
                        const userId = btn.dataset.user;
                        const userName = btn.dataset.name;
                        let role = 'member';
                        const roleSelect = btn.closest('li')?.querySelector('.role-select');
                        if (roleSelect) role = roleSelect.value;
                        if (!await confirm(`添加用户 "${userName}" 到项目，角色为 ${role === 'manager' ? '经理' : '成员'}？`)) return;
                        const res = await fetch(`/admin/projects/${projectId}/members`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            credentials: 'include',
                            body: JSON.stringify({ user_id: userId, role: role })
                        });
                        if (res.ok) {
                            closeModal();
                            showMemberManager(projectId);
                        } else {
                            const err = await res.json();
                            alert('添加失败: ' + (err.error || '未知错误'));
                        }
                    };
                });
            }

            const filterInput = modal.querySelector('#filterUsersInput');
            const usersContainer = modal.querySelector('#usersListContainer');
            filterInput.oninput = (e) => {
                usersContainer.innerHTML = renderAddableUsers(allUsers, e.target.value);
                attachAddHandlers();
            };

            attachMemberActionHandlers();
            attachAddHandlers();

            if (!isGlobalAdmin && allUsers.length === 0) {
                usersContainer.innerHTML = '<p>请输入用户名搜索以添加成员。</p>';
            }

        } catch (err) {
            console.error('Failed to load member manager:', err);
            alert('加载失败，请刷新页面后重试');
        }
    }

    // ======================== File Operation Functions ========================
    async function downloadFile(fileId) {
        window.open(`/admin/projects/${currentProjectId}/files/${fileId}/download`, '_blank');
    }

    async function showVersionHistory(fileId, fileName) {
        const res = await fetch(`/admin/projects/${currentProjectId}/files/${fileId}/versions`, { credentials: 'include' });
        const data = await res.json();
        const versions = data.versions || [];
        if (versions.length === 0) { alert('没有版本历史'); return; }
        let html = `<h4>${escapeHtml(fileName)} - 版本历史</h4>
        <div style="margin-bottom:8px;">
            <button onclick="event.stopPropagation();fetch('/admin/projects/${currentProjectId}/files/${fileId}/status',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:'final'}),credentials:'include'}).then(r=>r.json()).then(()=> { location.reload(); }).catch(()=>{})" style="background:#38a169;color:#fff;border:none;border-radius:4px;padding:4px 10px;font-size:.7rem;cursor:pointer;margin-right:4px;">设为定稿</button>
            <button onclick="event.stopPropagation();fetch('/admin/projects/${currentProjectId}/files/${fileId}/status',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:'draft'}),credentials:'include'}).then(r=>r.json()).then(()=> { location.reload(); }).catch(()=>{})" style="background:#a0aec0;color:#fff;border:none;border-radius:4px;padding:4px 10px;font-size:.7rem;cursor:pointer;">设为草稿</button>
        </div><ul>`;
        for (const v of versions) {
            html += `<li>版本 v${v.version} - ${(v.file_size / 1024).toFixed(1)} KB - 由 ${escapeHtml(v.uploaded_by_name || v.uploaded_by)}上传于 ${new Date(v.uploaded_at).toLocaleString()}
                      <button onclick="window.open('/admin/projects/${currentProjectId}/files/${fileId}/download?version=${v.version}', '_blank')">下载此版本</button></li>`;
        }
        html += '</ul>';
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.style.display = 'block';
        modal.innerHTML = `<div class="modal-content"><span class="close">&times;</span>${html}</div>`;
        document.body.appendChild(modal);
        modal.querySelector('.close').onclick = () => modal.remove();
        window.onclick = (e) => { if (e.target === modal) modal.remove(); };
    }

    async function showFileComments(fileId, fileName) {
        const res = await fetch(`/admin/projects/${currentProjectId}/files/${fileId}/comments`, { credentials: 'include' });
        const data = await res.json();
        const comments = data.comments || [];
        let html = `<h4>${escapeHtml(fileName)} - 标注</h4><div id="commentsList">`;
        for (const c of comments) {
            html += `<div><strong>${escapeHtml(c.username)}</strong> (${new Date(c.created_at).toLocaleString()}): ${escapeHtml(c.comment)}</div>`;
        }
        html += `</div><textarea id="newComment" rows="2" style="width:100%; margin-top:10px;" placeholder="添加标注..."></textarea>
                 <button id="addCommentBtn" style="margin-top:5px;">添加标注</button>`;
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.style.display = 'block';
        modal.innerHTML = `<div class="modal-content"><span class="close">&times;</span>${html}</div>`;
        document.body.appendChild(modal);
        modal.querySelector('.close').onclick = () => modal.remove();
        modal.querySelector('#addCommentBtn').onclick = async () => {
            const comment = modal.querySelector('#newComment').value.trim();
            if (!comment) return;
            const addRes = await fetch(`/admin/projects/${currentProjectId}/files/${fileId}/comments`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ comment })
            });
            if (addRes.ok) {
                modal.remove();
                showFileComments(fileId, fileName);
            } else alert('添加标注失败');
        };
    }

    async function showMoveFileDialog(fileId, currentFileName) {
        const res = await fetch(`/admin/projects/${currentProjectId}/folders`, { credentials: 'include' });
        const data = await res.json();
        const folders = data.folders;
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
            <div class="modal-content">
                <span class="close">&times;</span>
                <h4>移动文件: ${escapeHtml(currentFileName)}</h4>
                <select id="targetFolderSelect">${options}</select>
                <button id="confirmMoveBtn" style="margin-top:10px;">确认移动</button>
            </div>
        `;
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.style.display = 'block';
        modal.innerHTML = modalHtml;
        document.body.appendChild(modal);
        modal.querySelector('.close').onclick = () => modal.remove();
        modal.querySelector('#confirmMoveBtn').onclick = async () => {
            const btn = modal.querySelector('#confirmMoveBtn');
            btn.disabled = true;
            btn.textContent = '移动中...';
            let targetFolderId = modal.querySelector('#targetFolderSelect').value;
            const moveRes = await fetch(`/admin/projects/${currentProjectId}/files/${fileId}/move`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ folder_id: targetFolderId })
            });
            if (moveRes.ok) {
                showToast('文件已移动', 'success', 3000);
                modal.remove();
                if (targetFolderId) {
                    currentFolderId = targetFolderId;
                    await loadFolderTree(currentProjectId, targetFolderId);
                    await loadFilesInFolder(currentProjectId, targetFolderId);
                } else {
                    currentFolderId = null;
                    await loadFolderTree(currentProjectId, null);
                    await loadFilesInFolder(currentProjectId, null);
                }
            } else {
                const err = await moveRes.json();
                alert('移动失败: ' + (err.error || '未知错误'));
                btn.disabled = false;
                btn.textContent = '确认移动';
            }
        };
    }

    async function deleteFile(fileId) {
        if (!await confirm('确定要删除此文件吗？回收站保存时间3天。')) return;
        const res = await fetch(`/admin/projects/${currentProjectId}/files/${fileId}`, { method: 'DELETE', credentials: 'include' });
        if (res.ok) {
            await loadFilesInFolder(currentProjectId, currentFolderId);
        } else {
            const err = await res.json();
            alert('删除失败: ' + (err.error || '未知错误'));
        }
    }

    function attachFileListEvents() {
        document.querySelectorAll('.download-file').forEach(btn => btn.onclick = () => downloadFile(btn.dataset.id));
        document.querySelectorAll('.version-history').forEach(btn => {
            const fileId = btn.dataset.id;
            const fileName = btn.closest('tr')?.cells[1]?.innerText || '文件';
            btn.onclick = () => showVersionHistory(fileId, fileName);
        });
        document.querySelectorAll('.comment-file').forEach(btn => {
            const fileId = btn.dataset.id;
            const fileName = btn.closest('tr')?.cells[1]?.innerText || '文件';
            btn.onclick = () => showFileComments(fileId, fileName);
        });
        document.querySelectorAll('.move-file').forEach(btn => {
            const fileId = btn.dataset.id;
            const fileName = btn.closest('tr')?.cells[1]?.innerText || '文件';
            btn.onclick = () => showMoveFileDialog(fileId, fileName);
        });
        document.querySelectorAll('.delete-file').forEach(btn => btn.onclick = () => deleteFile(btn.dataset.id));
        document.querySelectorAll('.generate-project-skill').forEach(btn => {
            btn.onclick = async () => {
                const fileId = btn.dataset.id;
                btn.disabled = true;
                btn.textContent = '⏳ 分析中...';
                try {
                    const res = await fetch('/project_files/' + fileId + '/generate_skill', { method: 'POST', credentials: 'include' });
                    const data = await res.json();
                    if (res.ok) {
                        showSkillFeedback(btn, data, 'project_file', parseInt(fileId));
                    } else {
                        const msg = data.error || '生成失败';
                        const hint = data.hint || '';
                        alert((msg + (hint ? '\n' + hint : '')) || '生成失败。可能是文件无可提取的文字内容。');
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
    }

    const originalLoadFilesInFolder = loadFilesInFolder;
    window.loadFilesInFolder = async function(projectId, folderId) {
        await originalLoadFilesInFolder(projectId, folderId);
        attachFileListEvents();
    };

    // ======================== Finish, Abort, Delete Project ========================
    async function finishProject(projectId) {
        if (!await confirm('完成项目后，所有文件将打包为ZIP，项目变为只读。确定吗？')) return;
        const res = await fetch(`/admin/projects/${projectId}/finish`, { method: 'POST', credentials: 'include' });
        const data = await res.json();
        if (res.ok) {
            showToast('项目已归档，正在下载...', 'success', 3000);
            // Auto-trigger download
            if (data.download_url) {
                const a = document.createElement('a');
                a.href = data.download_url;
                a.download = data.zip_filename || '';
                a.style.display = 'none';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
            await loadProjects();
            projectsListView.style.display = 'block';
            fileExplorerView.style.display = 'none';
            toggleQuickLinksButton(false);
            syncActiveTabWithView();
        } else alert('归档失败: ' + (data.error || '操作失败，请重试'));
    }

    async function abortProject(projectId) {
        if (!await confirm('中止项目后，项目状态变为“已中止”，可以立即删除。确定吗？')) return;
        const res = await fetch(`/admin/projects/${projectId}/abort`, { method: 'POST', credentials: 'include' });
        if (res.ok) {
            alert('项目已中止');
            await loadProjects();
        } else {
            const err = await res.json();
            alert('中止失败: ' + (err.error || '未知错误'));
        }
    }

    async function deleteProject(projectId) {
        if (!await confirm('确定要删除此项目吗？')) return;
        const res = await fetch(`/admin/projects/${projectId}`, { method: 'DELETE', credentials: 'include' });
        const data = await res.json();
        if (res.ok) {
            alert('项目已删除');
            await loadProjects();
        } else alert('删除失败: ' + ('操作失败，请重试'));
    }

    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func(...args), wait);
        };
    }

    document.getElementById('createProjectBtn').onclick = () => showCreateProjectModal();

    async function showCreateProjectModal() {
        // ── Fetch user list once ──
        let allUsers = [];
        try {
            const ur = await fetch('/admin/users', { credentials: 'include' });
            if (ur.ok) { const d = await ur.json(); allUsers = d.users || []; }
        } catch(e) { console.warn('Failed to load user list', e); }

        const overlay = document.createElement('div');
        overlay.className = 'modal-overlay';
        overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.5);z-index:10000;display:flex;align-items:center;justify-content:center;';

        const buildContent = (selectedIndustry) => {
            return `
            <div class="custom-modal" style="max-width:480px;width:90%;max-height:85vh;overflow-y:auto;">
                <h3 style="margin:0 0 12px;">🆕 新建项目</h3>

                <label style="font-weight:600;font-size:0.82rem;display:block;margin-bottom:6px;">📋 行业类型（点击选择）</label>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:14px;" id="industryGrid">
                    ${['bidding_agency','engineering_cost','engineering_audit','general'].map(code => {
                        const labels = {bidding_agency:'🏗️ 招标代理', engineering_cost:'💰 工程造价', engineering_audit:'🔍 工程审计', general:'📋 通用'};
                        const selected = code === selectedIndustry;
                        return `<button class="industry-opt" data-code="${code}"
                            style="padding:10px 6px;border-radius:8px;border:2px solid ${selected?'#2563eb':'#d1d5db'};background:${selected?'#eff6ff':'white'};cursor:pointer;font-size:0.85rem;font-weight:${selected?'700':'400'};transition:all .15s;">
                            ${labels[code]}
                        </button>`;
                    }).join('')}
                </div>

                <label style="font-weight:600;font-size:0.82rem;display:block;margin-bottom:4px;">👤 项目负责人（必选）</label>
                <div style="position:relative;margin-bottom:4px;">
                    <input type="text" id="managerSearch" placeholder="🔍 输入用户名搜索..." autocomplete="off"
                        style="width:100%;padding:8px 10px;border-radius:6px;border:1px solid #d1d5db;font-size:0.85rem;box-sizing:border-box;">
                    <div id="managerDropdown" style="display:none;position:absolute;top:100%;left:0;right:0;max-height:180px;overflow-y:auto;background:white;border:1px solid #d1d5db;border-radius:0 0 6px 6px;z-index:10001;box-shadow:0 4px 12px rgba(0,0,0,.15);"></div>
                </div>
                <div id="selectedManager" style="font-size:0.75rem;color:#6b7280;min-height:18px;margin-bottom:12px;"></div>

                <label style="font-weight:600;font-size:0.82rem;display:block;margin-bottom:4px;">📋 招标类别（可选）</label>
                <select id="projBiddingCat" style="width:100%;padding:8px 10px;border-radius:6px;border:1px solid #d1d5db;font-size:0.85rem;box-sizing:border-box;margin-bottom:8px;">
                    <option value="">-- 不指定 --</option>
                    <option value="construction">🏗️ 工程建设</option>
                    <option value="goods">📦 货物采购</option>
                    <option value="services">💼 服务采购</option>
                </select>

                <label style="font-weight:600;font-size:0.82rem;display:block;margin-bottom:4px;">🔧 采购方式（可选）</label>
                <select id="projBidMethod" style="width:100%;padding:8px 10px;border-radius:6px;border:1px solid #d1d5db;font-size:0.85rem;box-sizing:border-box;margin-bottom:12px;">
                    <option value="">-- 不指定 --</option>
                    <option value="open_bid">公开招标</option>
                    <option value="invited_bid">邀请招标</option>
                    <option value="competitive_negotiation">竞争性谈判</option>
                    <option value="inquiry">询价</option>
                    <option value="single_source">单一来源</option>
                    <option value="competitive_consultation">竞争性磋商</option>
                </select>

                <label style="font-weight:600;font-size:0.82rem;display:block;margin-bottom:4px;">📝 项目名称</label>
                <input type="text" id="projNameInput" placeholder="输入项目名称..." style="width:100%;padding:8px 10px;border-radius:6px;border:1px solid #d1d5db;font-size:0.85rem;box-sizing:border-box;margin-bottom:12px;">

                <div style="display:flex;gap:8px;justify-content:flex-end;">
                    <button id="projCancelBtn" class="file-btn" style="background:#f3f4f6;color:#374151;border-color:#d1d5db;padding:8px 20px;">取消</button>
                    <button id="projCreateConfirm" class="file-btn" style="background:#2563eb;color:white;border-color:#1d4ed8;padding:8px 20px;">✅ 创建项目</button>
                </div>
            </div>`;
        };

        let state = { industry: 'general', managerId: '', managerName: '', bidding_category: '', bid_method: '' };
        overlay.innerHTML = buildContent('general');
        document.body.appendChild(overlay);

        const refreshUI = () => { overlay.innerHTML = buildContent(state.industry); wireEvents(); };
        const wireEvents = () => {
            // Industry buttons
            overlay.querySelectorAll('.industry-opt').forEach(btn => {
                btn.onclick = () => { state.industry = btn.dataset.code; refreshUI(); };
                btn.onmouseenter = (e) => { if (btn.dataset.code !== state.industry) e.target.style.background = '#f9fafb'; };
                btn.onmouseleave = (e) => { if (btn.dataset.code !== state.industry) e.target.style.background = 'white'; };
            });

            // Bidding category/method select preservation
            const catSel = overlay.querySelector('#projBiddingCat');
            const methSel = overlay.querySelector('#projBidMethod');
            if (catSel) { catSel.value = state.bidding_category; catSel.onchange = () => { state.bidding_category = catSel.value; }; }
            if (methSel) { methSel.value = state.bid_method; methSel.onchange = () => { state.bid_method = methSel.value; }; }

            // Searchable user dropdown
            const searchInput = overlay.querySelector('#managerSearch');
            const dropdown = overlay.querySelector('#managerDropdown');
            const selectedSpan = overlay.querySelector('#selectedManager');
            if (state.managerId && state.managerName) {
                searchInput.value = state.managerName;
                selectedSpan.textContent = '✅ 已选择: ' + state.managerName;
            }

            const filterUsers = (q) => {
                if (!q) return allUsers.slice(0, 15);
                const lower = q.toLowerCase();
                return allUsers.filter(u => (u.username||'').toLowerCase().includes(lower)).slice(0, 15);
            };

            const renderDropdown = (users) => {
                if (!users.length) { dropdown.style.display = 'none'; return; }
                dropdown.innerHTML = users.map(u =>
                    `<div class="user-opt" data-uid="${escapeHtml(u.user_id)}" data-uname="${escapeHtml(u.username)}"
                        style="padding:7px 10px;cursor:pointer;font-size:0.82rem;border-bottom:1px solid #f3f4f6;">
                        ${escapeHtml(u.username)} <span style="color:#9ca3af;font-size:0.7rem;">${escapeHtml(u.role||'')}</span>
                    </div>`
                ).join('');
                dropdown.style.display = '';

                dropdown.querySelectorAll('.user-opt').forEach(div => {
                    div.onclick = () => {
                        state.managerId = div.dataset.uid;
                        state.managerName = div.dataset.uname;
                        searchInput.value = state.managerName;
                        selectedSpan.textContent = '✅ 已选择: ' + state.managerName;
                        dropdown.style.display = 'none';
                    };
                    div.onmouseenter = (e) => e.target.style.background = '#f0f9ff';
                    div.onmouseleave = (e) => e.target.style.background = 'white';
                });
            };

            searchInput.onfocus = () => renderDropdown(filterUsers(searchInput.value));
            searchInput.oninput = () => renderDropdown(filterUsers(searchInput.value));
            searchInput.onblur = () => setTimeout(() => { dropdown.style.display = 'none'; }, 200);

            // Cancel
            overlay.querySelector('#projCancelBtn').onclick = () => overlay.remove();

            // Confirm
            overlay.querySelector('#projCreateConfirm').onclick = async () => {
                const name = (overlay.querySelector('#projNameInput')?.value || '').trim();
                if (!name) { alert('请输入项目名称'); return; }
                if (!state.managerId) { alert('请选择项目负责人'); return; }
                const biddingCat = overlay.querySelector('#projBiddingCat')?.value || '';
                const biddingMeth = overlay.querySelector('#projBidMethod')?.value || '';
                overlay.querySelector('#projCreateConfirm').disabled = true;
                overlay.querySelector('#projCreateConfirm').textContent = '⏳ 创建中...';

                try {
                    const res = await fetch('/admin/projects', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        credentials: 'include',
                        body: JSON.stringify({ name, industry: state.industry, manager_id: state.managerId, bidding_category: biddingCat, bid_method: biddingMeth })
                    });
                    if (res.ok) {
                        const result = await res.json();
                        overlay.remove();
                        showToast('项目已创建 ✅', 'success', 2500);
                        await loadProjects();
                        if (result.chat_thread_id) {
                            currentProjectId = result.id;
                            currentProjectName = name;
                            // Force-refresh history and ensure tab sticks
                            _loadingHistory = false;
                            await loadHistoryList(true);
                            // Verify project chat actually appeared; reload if not
                            const sRes = await fetch('/get_sessions', { credentials: 'include' });
                            const sData = await sRes.json();
                            const found = (sData.sessions || []).find(s => s.project_id == result.id && !s.is_grilling);
                            if (!found) {
                                await loadHistoryList(true);
                            }
                            switchToPanel('chatInterface');
                            switchSidebarPane('chat');
                            toggleQuickLinksButton(false);
                            localStorage.setItem('zlai_activeTab', 'chat');
                            syncActiveTabWithView();
                            if (typeof scrollToBottom === 'function') setTimeout(scrollToBottom, 100);
                        }
                    } else {
                        const err = await res.json();
                        alert('创建失败: ' + (err.error || '未知错误'));
                        overlay.querySelector('#projCreateConfirm').disabled = false;
                        overlay.querySelector('#projCreateConfirm').textContent = '✅ 创建项目';
                    }
                } catch(e) {
                    console.error('Create project error:', e);
                    alert('创建失败，请检查网络连接');
                    overlay.querySelector('#projCreateConfirm').disabled = false;
                    overlay.querySelector('#projCreateConfirm').textContent = '✅ 创建项目';
                }
            };

            // Focus name input
            setTimeout(() => overlay.querySelector('#projNameInput')?.focus(), 100);
        };

        wireEvents();

        // Click outside to close
        overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    }

    async function checkAdminStatus() {
        try {
            const res = await fetch('/admin/projects', { credentials: 'include' });
            if (res.ok) {
                const data = await res.json();
                if (data.has_projects || sessionStorage.getItem('isAdmin') === 'true') {
                    // Only make admin tabs visible — do NOT force-switch tab
                    const adminTabBtn = document.getElementById('adminTabBtn');
                    if (adminTabBtn) adminTabBtn.style.display = 'inline-block';
                }
            }
        } catch (e) {
            console.warn('checkAdminStatus error:', e);
        }
    }

    function showAdminTab() {
        const tabBar = document.getElementById('tabBar');
        if (tabBar) tabBar.style.display = 'flex';
        const adminTabBtn = document.getElementById('adminTabBtn');
        if (adminTabBtn) {
            adminTabBtn.style.display = 'inline-block';
            adminTabBtn.click();
        }
    }

    // ======================== Editable Quick Links ========================
    function getCustomLinksKey(projectId) {
        return projectId ? `project_quicklinks_${projectId}` : 'global_quicklinks';
    }
    function loadCustomLinks(projectId) {
        const stored = localStorage.getItem(getCustomLinksKey(projectId));
        return stored ? JSON.parse(stored) : [];
    }
    function saveCustomLinks(projectId, links) {
        localStorage.setItem(getCustomLinksKey(projectId), JSON.stringify(links));
    }
    function showQuickLinksModal(projectId = null, projectName = null) {
        const fixedLinks = [
            { url: "https://zxgk.court.gov.cn/shixin/", title: "⚖️ 失信被执行人查询" },
            { url: "https://www.creditchina.gov.cn/zhuanxiangchaxun/zhongdashuishouweifaanjian/", title: "💰 重大税收违法案件查询" },
            { url: "https://www.ccgp.gov.cn/search/cr/", title: "📋 政府采购严重违法失信名单" }
        ];

        let customLinks = loadCustomLinks(projectId);
        let modalTitle = projectId ? `快捷查询 - ${escapeHtml(projectName || '项目')}` : '快捷查询';

        let html = `
        <div class="modal-content" style="width: 800px; max-width: 95%; max-height: 90vh; overflow-y: auto;">
            <span class="close">&times;</span>
            <h3>${escapeHtml(modalTitle)}</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <!-- Left column: Links -->
                <div>
                    <h4>🔗 快捷链接</h4>
                    <div class="quick-links-list" id="quickLinksListContainer"></div>
                    <div class="add-link-form" style="margin-top: 16px;">
                        <input type="text" id="newLinkInput" placeholder="输入网址">
                        <button id="addLinkBtn">添加链接</button>
                    </div>
                </div>
                <!-- Right column: Batch compare history -->
                <div>
                    <h4>📦 批量对比历史</h4>
                    <div id="batchHistoryList" style="max-height:200px; overflow-y:auto;"></div>
                </div>
            </div>
        </div>`;

        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.style.display = 'block';
        modal.innerHTML = html;
        document.body.appendChild(modal);

        // Close handlers
        modal.querySelector('.close').onclick = () => modal.remove();
        modal.onclick = (e) => { if (e.target === modal) modal.remove(); };

        // ----- Link management -----
        const addBtn = modal.querySelector('#addLinkBtn');
        const inputField = modal.querySelector('#newLinkInput');
        const listContainer = modal.querySelector('#quickLinksListContainer');

        function refreshCustomLinksDisplay() {
            // Clear existing content except the container
            listContainer.innerHTML = '';

            // 1. Render fixed links
            for (const link of fixedLinks) {
                const item = document.createElement('div');
                item.className = 'quick-link-item fixed-link';
                item.innerHTML = `
                    <div class="quick-link-url">
                        <a href="${escapeHtml(link.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(link.title)}</a>
                    </div>
                    <span style="color:#888;">(固定)</span>
                `;
                listContainer.appendChild(item);
            }

            // 2. Render custom links
            for (let i = 0; i < customLinks.length; i++) {
                const link = customLinks[i];
                const item = document.createElement('div');
                item.className = 'quick-link-item';
                item.dataset.index = i;
                item.innerHTML = `
                    <div class="quick-link-url">
                        <a href="${escapeHtml(link.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(link.title || link.url)}</a>
                    </div>
                    <button class="quick-link-delete" data-index="${i}">删除</button>
                `;
                listContainer.appendChild(item);
                item.querySelector('.quick-link-delete').onclick = () => {
                    customLinks.splice(i, 1);
                    saveCustomLinks(projectId, customLinks);
                    refreshCustomLinksDisplay();
                };
            }
        }

        function addLink() {
            let url = inputField.value.trim();
            if (!url) return;
            if (!url.startsWith('http://') && !url.startsWith('https://')) url = 'https://' + url;
            customLinks.push({ url: url, title: url });
            saveCustomLinks(projectId, customLinks);
            inputField.value = '';
            refreshCustomLinksDisplay();
        }

        addBtn.onclick = addLink;
        inputField.addEventListener('keypress', (e) => { if (e.key === 'Enter') addLink(); });

        // Initial render of links
        refreshCustomLinksDisplay();

        loadBatchHistory();

        async function loadBatchHistory() {
            const listDiv = modal.querySelector('#batchHistoryList');
            try {
                const res = await fetch('/list_batch_results', { credentials: 'include' });
                const data = await res.json();
                const results = data.results || [];
                if (results.length === 0) {
                    listDiv.innerHTML = '<p style="color:#999;">暂无批量对比记录</p>';
                    return;
                }
                let html = '<ul style="list-style:none; padding:0;">';
                for (const r of results) {
                    const names = (typeof r.file_names === 'string' ? JSON.parse(r.file_names) : r.file_names) || [];
                    html += `
                        <li style="display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #eee; flex-wrap:wrap; gap:4px;">
                            <div>
                                <span>📦 ${r.file_count}个文件 · ${r.pair_count}对</span>
                                <small style="color:var(--card-muted);"> · ${escapeHtml(r.created_by_name || '?')} · ${new Date(r.created_at).toLocaleString()}</small>
                                ${r.max_risk > 10 ? `<span style="color:#dc2626;font-size:.65rem;"> ⚠${r.max_risk.toFixed(0)}</span>` : ''}
                            </div>
                            <div>
                                <a href="/batch_result/${r.task_id}" class="file-btn" style="padding:2px 8px; text-decoration:none;" download>📥 下载</a>
                                ${sessionStorage.getItem('isAdmin') === 'true' ? `<button class="delete-batch-btn" data-id="${r.id}" style="background:#e74c3c; color:white; border:none; border-radius:4px; padding:2px 8px; margin-left:4px;">🗑️</button>` : ''}
                            </div>
                        </li>`;
                }
                html += '</ul>';
                listDiv.innerHTML = html;
                listDiv.querySelectorAll('.delete-batch-btn').forEach(btn => {
                    btn.onclick = async () => {
                        if (!confirm('确定删除此批量对比结果？')) return;
                        const res = await fetch('/delete_batch_result/' + btn.dataset.id, { method:'POST', credentials:'include' });
                        if (res.ok) { showToast('已删除', 'success'); loadBatchHistory(); }
                        else alert('删除失败');
                    };
                });
            } catch (err) { listDiv.innerHTML = '<p style="color:red;">加载失败</p>'; }
        }
    }

    // Fallback: ensure tab bar is visible even if an error occurs later
    try {
        updateProjectTabVisibility();
    } catch (e) {
        console.error('Error initializing tabs:', e);
        const tabBar = document.getElementById('tabBar');
        if (tabBar) tabBar.style.display = 'flex';
        document.getElementById('chatTabBtn')?.classList.add('active');
    }
    // checkAdminStatus() called later inside init block with proper tab restore

    // ======================== Final Initialization ========================
    async function sendMessage() {
        if (isProcessing) {
            addSystemMessage('请等待当前请求完成后再发送新消息。');
            return;
        }
        const text = messageInput.value.trim();
        if (!text && selectedFiles.length === 0) {
            addSystemMessage('请输入问题或加载文件。');
            return;
        }
        // Prepend quote context so it shows in chat AND goes to AI
        let sendText = text;
        if (_currentQuoteContext && _currentQuoteContext.fullContent && text) {
            const author = _currentQuoteContext.author || '';
            sendText = '--- 引用' + (author ? ' @'+author : '') + ' ---\n' + _currentQuoteContext.fullContent + '\n--- 追问 ---\n' + text;
            console.log('sendMessage with quote, total chars:', sendText.length, 'author:', author);
        }
        if (text) addUserMessage(sendText);
        messageInput.value = '';
        const filesToSend = [...selectedFiles];
        selectedFiles = [];
        if (persistentReminderDiv) {
            persistentReminderDiv.remove();
            persistentReminderDiv = null;
        }

        isProcessing = true;
        const processingIndicator = document.getElementById('processingIndicator');
        if (processingIndicator) processingIndicator.style.display = 'inline-block';
        sendBtn.disabled = true;
        fileBtn.disabled = true;
        newChatBtn.disabled = true;
        document.querySelectorAll('.history-item').forEach(item => {
            item.style.pointerEvents = 'none';
            item.classList.add('processing-disabled');
        });

        const messageId = crypto.randomUUID ? crypto.randomUUID() : Date.now() + '-' + Math.random();
        try {
            // All messages use streaming
            await sendMessageStreaming(sendText, messageId, filesToSend);
        } catch (err) {
            addSystemMessage('网络错误，请稍后重试。');
            console.error(err);
        } finally {
            isProcessing = false;
            if (processingIndicator) processingIndicator.style.display = 'none';
            sendBtn.disabled = false;
            fileBtn.disabled = false;
            newChatBtn.disabled = false;
            document.querySelectorAll('.history-item').forEach(item => {
                item.style.pointerEvents = '';
                item.classList.remove('processing-disabled');
            });
            fileBtn.innerText = '📄随问题上传文件';
            selectedFiles = [];
            // Create quote association if quoting context exists
            if (_currentQuoteContext && _currentQuoteContext.quotedMessageId && _isCurrentSessionProjectChat && currentProjectId) {
                const threadId = sessionStorage.getItem('currentThreadId');
                fetch(`/admin/projects/${currentProjectId}/quote`, {
                    method: 'POST', headers: {'Content-Type':'application/json'}, credentials: 'include',
                    body: JSON.stringify({
                        quoted_message_id: _currentQuoteContext.quotedMessageId,
                        parent_quote_id: _currentQuoteContext.parentQuoteId || null,
                        thread_id: threadId,
                    })
                }).catch(() => {});
                // Clear quote bubble + badges after sending
                const bubble = document.getElementById('quoteBubble');
                if (bubble) bubble.style.display = 'none';
                const input = document.getElementById('messageInput');
                if (input) input.rows = 1;
                setTimeout(() => {
                    document.querySelectorAll('.quote-badge').forEach(b => b.remove());
                }, 1000);
                _currentQuoteContext = null;
            }
        }
    }

    if (sessionStorage.getItem('forceAdminTab') === 'true') {
        _programmaticTabSwitch = true;
        showAdminTab();
        _programmaticTabSwitch = false;
        sessionStorage.removeItem('forceAdminTab');
    }

    sessionStorage.setItem('currentThreadId', '');
    // Consent check deferred to loadAccountModal() — bootstrap route provides it
    // Always load UI; consent gate handled server-side and in account modal
    setTimeout(async () => {
            try {
                await loadHistoryList();
                const sessions = await (await fetch('/get_sessions', { credentials: 'include' })).json();
                const justLoggedIn = sessionStorage.getItem('just_logged_in') === '1';
                if (justLoggedIn) sessionStorage.removeItem('just_logged_in');
                if (sessions.sessions && sessions.sessions.length > 0) {
                    await loadSession(sessions.sessions[0].thread_id);
                } else if (!justLoggedIn) {
                    // Create a new session only if NOT just logged in (avoid creating empty chat on login)
                    const newChatRes = await fetch('/new_chat', { method: 'POST', credentials: 'include' });
                    const newData = await newChatRes.json();
                    await loadSession(newData.thread_id);
                }
                checkStorage();
                await checkAdminStatus();
                updateProjectTabVisibility();
                setTimeout(() => {
                    const activeTab = restoreActiveTab();
                    const tabMap = { chat:'chatTabBtn', projects:'adminTabBtn', recycle:'recycleBinTabBtn', db:'databaseTabBtn', knowledge:'knowledgeLabTabBtn', wiki:'wikiTabBtn', timeline:'timelineTabBtn', stats:'analyticsTabBtn', review:'reviewTabBtn', templates:'templatesTabBtn' };
                    const targetBtn = document.getElementById(tabMap[activeTab] || 'chatTabBtn');
                    _programmaticTabSwitch = true;
                    if (targetBtn) targetBtn.click();
                    _programmaticTabSwitch = false;
                    // Auto-to-bottom only on chat tab restore
                    if (activeTab === 'chat') {
                        setTimeout(() => { _userHasScrolled = false; scrollToBottom(true); }, 200);
                    }
                }, 500);

            } catch (err) {
                console.error('Failed to initialize chat:', err);
            }
        }, 100);

    // ======================== Global Keyboard Shortcuts ========================
    document.addEventListener('keydown', async (e) => {
        const activeElement = document.activeElement;
        const isInputFocused = activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA' || activeElement.isContentEditable);

        if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
            const openModal = document.querySelector('.modal[style*="display: block"], .custom-modal-overlay');
            if (openModal) {
                const confirmBtn = openModal.querySelector('.confirm, #confirmMoveBtn, #confirmCreateAccountBtn, #confirmLoginBtn, #confirmDeclineBtn, #confirmYes, #alertOk, #promptOk, .add-member, #addLinkBtn, .confirm-move, .confirm-delete, button[type="submit"]');
                if (confirmBtn) {
                    e.preventDefault();
                    confirmBtn.click();
                    return;
                }
            }
            if (activeElement && activeElement.id === 'messageInput') {
                return;
            }
        }

        if (e.key === 'Escape') {
            const openModal = document.querySelector('.modal[style*="display: block"], .custom-modal-overlay');
            if (openModal) {
                const closeBtn = openModal.querySelector('.close, .cancel, #closeMemberModal, .close-modal');
                if (closeBtn) {
                    e.preventDefault();
                    closeBtn.click();
                } else {
                    openModal.remove();
                }
                return;
            }
            if (fileExplorerView && fileExplorerView.style.display === 'block') {
                const backBtn = document.getElementById('backToProjectsBtn');
                if (backBtn) {
                    e.preventDefault();
                    backBtn.click();
                }
                return;
            }
            if (currentBatchAbortController) {
                e.preventDefault();
                currentBatchAbortController.abort();
                addSystemMessage('批量对比已取消');
                return;
            }
            if (fileStationModal && fileStationModal.style.display === 'block') {
                e.preventDefault();
                if (typeof closeFileStationAndClearSelection === 'function') {
                    closeFileStationAndClearSelection();
                } else {
                    fileStationModal.style.display = 'none';
                    selectedFileIds.clear();
                }
                return;
            }
            if (accountModal && accountModal.style.display === 'block') {
                e.preventDefault();
                accountModal.style.display = 'none';
                return;
            }
        }

        if (e.key === 'Delete' || (e.key === 'Backspace' && !isInputFocused)) {
            // Handle deletion from Knowledge Lab and Company KB using file-item selections
            const selectedFileItems = document.querySelectorAll('.file-item.selected');
            if (selectedFileItems.length > 0) {
                e.preventDefault();
                const labContainer = document.getElementById('labFileList');
                const companyContainer = document.getElementById('companyKbList');
                const isInLab = labContainer && labContainer.contains(selectedFileItems[0]);
                const isInCompany = companyContainer && companyContainer.contains(selectedFileItems[0]);
                if (isInLab || isInCompany) {
                    const ids = Array.from(selectedFileItems).map(item => item.dataset.id);
                    const type = isInLab ? '个人知识库' : '公司知识库';
                    if (await confirm(`确定要删除 ${ids.length} 个${type}文件吗？`)) {
                        for (const id of ids) {
                            const url = isInLab ? `/knowledge_lab/delete/${id}` : `/company_kb/delete/${id}`;
                            await fetch(url, { method: 'POST', credentials: 'include' });
                        }
                        if (isInLab) loadKnowledgeLabFiles();
                        else { loadCompanyKnowledgeBase(); loadCompanyCategories(); }
                        // Clear selections
                        selectedFileItems.forEach(item => item.classList.remove('selected'));
                    }
                    return;
                }
            }
            if (fileStationModal && fileStationModal.style.display === 'block' && selectedFileIds.size > 0) {
                e.preventDefault();
                if (await confirm(`确定要删除 ${selectedFileIds.size} 个文件吗？回收站保存时间3天。`)) {
                    for (const fileId of selectedFileIds) {
                        await fetch('/delete_file_station', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            credentials: 'include',
                            body: JSON.stringify({ file_id: fileId })
                        });
                    }
                    selectedFileIds.clear();
                    await loadFileStation();
                }
                return;
            }
            if (adminPanel && adminPanel.style.display === 'block' && fileExplorerView && fileExplorerView.style.display === 'block') {
                const selectedCheckboxes = document.querySelectorAll('.file-select:checked');
                if (selectedCheckboxes.length > 0) {
                    e.preventDefault();
                    if (await confirm(`确定要删除 ${selectedCheckboxes.length} 个文件吗？回收站保存时间3天。`)) {
                        for (const cb of selectedCheckboxes) {
                            const fileId = cb.dataset.id;
                            if (fileId) {
                                await fetch(`/admin/projects/${currentProjectId}/files/${fileId}`, { method: 'DELETE', credentials: 'include' });
                            }
                        }
                        await loadFilesInFolder(currentProjectId, currentFolderId);
                    }
                    return;
                }
            }
        }
    });

    // ======================== Recycle Bin functions ========================
    async function loadRecycleBin(cachedData) {
        try {
            const data = cachedData || await (await fetch('/get_recycle_bin', { credentials: 'include' })).json();

            const chatContainer = document.getElementById('chatRecycleList');
            if (data.chat_items && data.chat_items.length) {
                let html = '<ul style="list-style: none; padding: 0;">';
                for (const item of data.chat_items) {
                    const deletedDate = new Date(item.deleted_at).toLocaleString();
                    const expiresDate = new Date(item.expires_at).toLocaleString();
                    html += `
                        <li style="border:1px solid #ddd; padding:8px; margin-bottom:8px; border-radius:6px;">
                            <strong>${escapeHtml(item.file_name)}</strong> (${(item.file_size/1024).toFixed(1)} KB)
                            <br><small>创建人: ${escapeHtml(item.uploaded_by_name || '未知')} | 删除人: ${escapeHtml(item.deleted_by_name || '未知')}</small>
                            <br><small>删除于: ${deletedDate} | 过期: ${expiresDate}</small>
                            <button class="restore-item" data-id="${item.id}" data-source="chat" style="margin-left:10px; background:#27ae60; color:white; border:none; border-radius:4px; padding:2px 8px;">恢复</button>
                            <button class="delete-item" data-id="${item.id}" data-source="chat" style="margin-left:5px; background:#e74c3c; color:white; border:none; border-radius:4px; padding:2px 8px;">永久删除</button>
                        </li>
                    `;
                }
                html += '</ul>';
                chatContainer.innerHTML = html;
            } else {
                chatContainer.innerHTML = '<p>暂无删除的聊天文件</p>';
            }

            document.querySelectorAll('.restore-section-btn').forEach(btn => {
                btn.onclick = async () => {
                    const section = btn.dataset.section;
                    let confirmMsg = '';
                    if (section === 'chat') confirmMsg = '确定要恢复聊天区的所有文件吗？';
                    else if (section === 'project') confirmMsg = '确定要恢复整个项目区的所有文件和文件夹吗？';
                    else if (section === 'project_files') confirmMsg = '确定要恢复所有项目文件吗？';
                    else if (section === 'project_folders') confirmMsg = '确定要恢复所有项目文件夹吗？';
                    else if (section === 'knowledge_lab') confirmMsg = '确定要恢复知识库区的所有文件吗？';
                    if (await confirm(confirmMsg)) {
                        const restoreRes = await fetch('/restore_from_recycle_bin', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            credentials: 'include',
                            body: JSON.stringify({ restore_all: true, section: section })
                        });
                        if (restoreRes.ok) {
                            showToast('恢复成功', 'success', 3000);
                            await loadRecycleBin();
                            await loadFileStation();
                            if (currentProjectId) {
                                await loadFilesInFolder(currentProjectId, currentFolderId);
                                await loadFolderTree(currentProjectId, currentFolderId);
                            }
                        } else {
                            alert('恢复失败');
                        }
                    }
                };
            });

            document.querySelectorAll('.empty-section-btn').forEach(btn => {
                btn.onclick = async () => {
                    const section = btn.dataset.section;
                    let confirmMsg = '';
                    if (section === 'chat') confirmMsg = '确定要清空聊天区吗？此操作不可恢复。';
                    else if (section === 'project') confirmMsg = '确定要清空整个项目区（文件和文件夹）吗？此操作不可恢复。';
                    else if (section === 'project_files') confirmMsg = '确定要清空项目文件区吗？此操作不可恢复。';
                    else if (section === 'project_folders') confirmMsg = '确定要清空项目文件夹区吗？此操作不可恢复。';
                    else if (section === 'knowledge_lab') confirmMsg = '确定要清空知识库区吗？此操作不可恢复。';
                    if (await confirm(confirmMsg)) {
                        const emptyRes = await fetch('/empty_recycle_bin', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            credentials: 'include',
                            body: JSON.stringify({ source: section })
                        });
                        if (emptyRes.ok) {
                            showToast('清空成功', 'success', 3000);
                            await loadRecycleBin();
                        } else {
                            alert('清空失败');
                        }
                    }
                };
            });

            const projectContainer = document.getElementById('projectRecycleList');
            if (data.project_items && data.project_items.length) {
                let html = '<ul style="list-style: none; padding: 0;">';
                for (const item of data.project_items) {
                    const deletedDate = new Date(item.deleted_at).toLocaleString();
                    const expiresDate = new Date(item.expires_at).toLocaleString();
                    html += `
                        <li style="border:1px solid #ddd; padding:8px; margin-bottom:8px; border-radius:6px;">
                            <strong>${escapeHtml(item.file_name)}</strong> (${(item.file_size/1024).toFixed(1)} KB)
                            <br><small>项目: ${escapeHtml(item.project_name)}</small>
                            <br><small>创建人: ${escapeHtml(item.uploaded_by_name || '未知')} | 删除人: ${escapeHtml(item.deleted_by_name || '未知')}</small>
                            <br><small>删除于: ${deletedDate} | 过期: ${expiresDate}</small>
                            <button class="restore-item" data-id="${item.id}" data-source="project" style="margin-left:10px; background:#27ae60; color:white; border:none; border-radius:4px; padding:2px 8px;">恢复</button>
                            <button class="delete-item" data-id="${item.id}" data-source="project" style="margin-left:5px; background:#e74c3c; color:white; border:none; border-radius:4px; padding:2px 8px;">永久删除</button>
                        </li>
                    `;
                }
                html += '</ul>';
                projectContainer.innerHTML = html;
            } else {
                projectContainer.innerHTML = '<p>暂无删除的项目文件</p>';
            }

            const folderContainer = document.getElementById('folderRecycleList');
            if (data.folder_items && data.folder_items.length) {
                let html = '<ul style="list-style: none; padding: 0;">';
                for (const item of data.folder_items) {
                    const deletedDate = new Date(item.deleted_at).toLocaleString();
                    const expiresDate = new Date(item.expires_at).toLocaleString();
                    html += `
                        <li style="border:1px solid #ddd; padding:8px; margin-bottom:8px; border-radius:6px;">
                            <strong>📁 ${escapeHtml(item.name)}</strong>
                            <br><small>项目: ${escapeHtml(item.project_name)} | 删除于: ${deletedDate} | 过期: ${expiresDate}</small>
                            <button class="restore-item" data-id="${item.id}" data-source="folder" style="margin-left:10px; background:#27ae60; color:white; border:none; border-radius:4px; padding:2px 8px;">恢复文件夹</button>
                            <button class="delete-item" data-id="${item.id}" data-source="folder" style="margin-left:5px; background:#e74c3c; color:white; border:none; border-radius:4px; padding:2px 8px;">永久删除</button>
                        </li>
                    `;
                }
                html += '</ul>';
                folderContainer.innerHTML = html;
            } else {
                folderContainer.innerHTML = '<p>暂无删除的文件夹</p>';
            }

            // KB items — split by source
            const labContainer = document.getElementById('labRecycleList');
            const companyContainer = document.getElementById('companyRecycleList');
            if (data.kb_items && data.kb_items.length) {
                const labItems = data.kb_items.filter(i => i.source === 'knowledge_lab');
                const companyItems = data.kb_items.filter(i => i.source === 'company_kb');

                if (labItems.length) {
                    let html = '<ul style="list-style: none; padding: 0;">';
                    for (const item of labItems) {
                        const deletedDate = new Date(item.deleted_at).toLocaleString();
                        const expiresDate = new Date(item.expires_at).toLocaleString();
                        html += `
                            <li style="border:1px solid #ddd; padding:8px; margin-bottom:8px; border-radius:6px;">
                                <strong>${escapeHtml(item.original_name || item.filename)}</strong> (${((item.file_size||0)/1024).toFixed(1)} KB)
                                <br><small>分类: ${escapeHtml(item.category || '未分类')} | 创建人: ${escapeHtml(item.uploaded_by_name || '未知')}</small>
                                <br><small>删除于: ${deletedDate} | 过期: ${expiresDate}</small>
                                <button class="restore-item" data-id="${item.id}" data-source="knowledge_lab" style="margin-left:10px; background:#27ae60; color:white; border:none; border-radius:4px; padding:2px 8px;">恢复</button>
                                <button class="delete-item" data-id="${item.id}" data-source="knowledge_lab" style="margin-left:5px; background:#e74c3c; color:white; border:none; border-radius:4px; padding:2px 8px;">永久删除</button>
                            </li>
                        `;
                    }
                    html += '</ul>';
                    labContainer.innerHTML = html;
                } else {
                    labContainer.innerHTML = '<p>暂无删除的个人知识库文件</p>';
                }

                if (companyItems.length) {
                    let html = '<ul style="list-style: none; padding: 0;">';
                    for (const item of companyItems) {
                        const deletedDate = new Date(item.deleted_at).toLocaleString();
                        const expiresDate = new Date(item.expires_at).toLocaleString();
                        html += `
                            <li style="border:1px solid #ddd; padding:8px; margin-bottom:8px; border-radius:6px;">
                                <strong>${escapeHtml(item.original_name || item.filename)}</strong> (${((item.file_size||0)/1024).toFixed(1)} KB)
                                <br><small>分类: ${escapeHtml(item.category || '未分类')} | 删除人: ${escapeHtml(item.deleted_by_name || '未知')}</small>
                                <br><small>删除于: ${deletedDate} | 过期: ${expiresDate}</small>
                                <button class="restore-item" data-id="${item.id}" data-source="knowledge_lab" style="margin-left:10px; background:#27ae60; color:white; border:none; border-radius:4px; padding:2px 8px;">恢复</button>
                                <button class="delete-item" data-id="${item.id}" data-source="knowledge_lab" style="margin-left:5px; background:#e74c3c; color:white; border:none; border-radius:4px; padding:2px 8px;">永久删除</button>
                            </li>
                        `;
                    }
                    html += '</ul>';
                    companyContainer.innerHTML = html;
                } else {
                    companyContainer.innerHTML = '<p>暂无删除的公司知识库文件</p>';
                }
            } else {
                if (labContainer) labContainer.innerHTML = '<p>暂无删除的个人知识库文件</p>';
                if (companyContainer) companyContainer.innerHTML = '<p>暂无删除的公司知识库文件</p>';
            }

            document.querySelectorAll('.restore-item').forEach(btn => {
                btn.onclick = async () => {
                    const itemId = btn.dataset.id;
                    const source = btn.dataset.source;
                    const restoreRes = await fetch('/restore_from_recycle_bin', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        credentials: 'include',
                        body: JSON.stringify({ item_id: itemId, source: source })
                    });
                    if (restoreRes.ok) {
                        showToast('恢复成功', 'success', 3000);
                        await loadRecycleBin();
                        await loadFileStation();
                        if (source === 'project' || (source === 'folder' && currentProjectId)) {
                            await loadFilesInFolder(currentProjectId, currentFolderId);
                            await loadFolderTree(currentProjectId, currentFolderId);
                        }
                    } else {
                        alert('恢复失败');
                    }
                };
            });

            document.querySelectorAll('.delete-item').forEach(btn => {
                btn.onclick = async () => {
                    const itemId = btn.dataset.id;
                    const source = btn.dataset.source;
                    if (await confirm('永久删除此项目？此操作不可恢复。')) {
                        const delRes = await fetch('/delete_recycle_item', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            credentials: 'include',
                            body: JSON.stringify({ item_id: itemId, source: source })
                        });
                        if (delRes.ok) {
                            showToast('已永久删除', 'success', 3000);
                            await loadRecycleBin();
                            if (source === 'chat') await loadFileStation();
                            if (source === 'project' || source === 'folder') {
                                if (currentProjectId) {
                                    await loadFilesInFolder(currentProjectId, currentFolderId);
                                    await loadFolderTree(currentProjectId, currentFolderId);
                                }
                            }
                        } else {
                            const err = await delRes.json();
                            alert('删除失败: ' + (err.error || '未知错误'));
                        }
                    }
                };
            });

            document.getElementById('emptyAllRecycleBtn').onclick = async () => {
                if (await confirm('确定要清空回收站吗？所有文件将被永久删除，不可恢复。')) {
                    const emptyRes = await fetch('/empty_recycle_bin', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        credentials: 'include',
                        body: JSON.stringify({ source: 'all' })
                    });
                    if (emptyRes.ok) {
                        showToast('回收站已清空', 'success', 3000);
                        await loadRecycleBin();
                    } else {
                        alert('清空失败');
                    }
                }
            };
        } catch (err) {
            console.error(err);
            document.getElementById('chatRecycleList').innerHTML = '<p>加载失败</p>';
            document.getElementById('projectRecycleList').innerHTML = '<p>加载失败</p>';
            document.getElementById('folderRecycleList').innerHTML = '<p>加载失败</p>';
            const lb = document.getElementById('labRecycleList');
            if (lb) lb.innerHTML = '<p>加载失败</p>';
            const cb = document.getElementById('companyRecycleList');
            if (cb) cb.innerHTML = '<p>加载失败</p>';
        }
    }

    // ======================== Admin Database Browser ========================
    async function loadTableList(cachedData) {
        try {
            const data = cachedData || await (await fetch('/admin/db_tables_overview', { credentials: 'include' })).json();
            const select = document.getElementById('dbTableSelect');
            select.innerHTML = '<option value="">-- 选择表 --</option>';
            var tables = data.tables || [];
            for (const t of tables) {
                var name = t.table_name || t;
                var count = t.row_count != null ? ' (' + t.row_count + '行)' : '';
                select.innerHTML += '<option value="' + escapeHtml(name) + '">' + escapeHtml(name) + count + '</option>';
            }
        } catch (err) {
            console.error(err);
            alert('加载表列表失败');
        }
    }

    async function loadTableData() {
        if (!currentTable) return;
        const tbody = document.getElementById('dbTableBody');
        tbody.innerHTML = '<tr><td colspan="10">加载中...</td></tr>';
        try {
            const res = await fetch('/admin/db_table_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                    table: currentTable,
                    page: currentPage,
                    per_page: currentPerPage,
                    search: currentSearch,
                    search_column: currentSearchColumn
                })
            });
            if (!res.ok) throw new Error('Failed to fetch data');
            const data = await res.json();
            const columns = data.columns;        // only real database columns
            const rows = data.rows;
            const total = data.total;
            const page = data.page;
            const per_page = data.per_page;

            // Build header: all columns + one operation column at the end
            const thead = document.getElementById('dbTableHeader');
            let headerHtml = '<tr>';
            for (const col of columns) {
                headerHtml += `<th style="padding:6px; border:1px solid #ddd;">${escapeHtml(col)}</th>`;
            }
            headerHtml += `<th style="padding:6px; border:1px solid #ddd;">操作</th></tr>`;
            thead.innerHTML = headerHtml;

            // Build body: data cells + one operation cell per row
            let bodyHtml = '';
            for (const row of rows) {
                bodyHtml += '<tr>';
                for (const col of columns) {
                    let value = row[col];
                    if (value === null) value = 'NULL';
                    else if (typeof value === 'object') value = JSON.stringify(value);
                    else value = String(value);
                    bodyHtml += `<td style="padding:6px; border:1px solid #ddd; word-break:break-word;" data-col="${escapeHtml(col)}">${escapeHtml(value)}</td>`;
                }
                const rowId = row.id !== undefined ? row.id : row.thread_id;
                bodyHtml += `<td style="padding:6px; border:1px solid #ddd; text-align:center;">
                                <button class="edit-cell-btn" data-id="${rowId}" style="background:#3498db; color:white; border:none; border-radius:4px; padding:2px 6px;">✏️</button>
                                <button class="delete-row-btn" data-id="${rowId}" style="background:#e74c3c; color:white; border:none; border-radius:4px; padding:2px 6px;">🗑️</button>
                             </td>`;
                bodyHtml += '</tr>';
            }
            if (rows.length === 0) {
                bodyHtml = `<tr><td colspan="${columns.length + 1}">无数据</td></tr>`;
            }
            tbody.innerHTML = bodyHtml;

            // Pagination (unchanged)
            const totalPages = Math.ceil(total / per_page);
            let paginationHtml = '';
            if (totalPages > 1) {
                for (let i = 1; i <= totalPages; i++) {
                    paginationHtml += `<button class="page-btn" data-page="${i}" style="padding:4px 8px; ${i === page ? 'background:#2c3e50; color:white;' : ''}">${i}</button>`;
                }
            }
            document.getElementById('dbPagination').innerHTML = paginationHtml;
            document.querySelectorAll('.page-btn').forEach(btn => {
                btn.onclick = () => {
                    currentPage = parseInt(btn.dataset.page);
                    loadTableData();
                };
            });

            attachTableActionHandlers();
        } catch (err) {
            console.error(err);
            tbody.innerHTML = '<tr><td colspan="10">加载失败</td></tr>';
        }
    }

    function loadDatabaseData(cachedData) {
        loadTableList(cachedData).then(() => {
            if (currentTable) loadTableData();
        });
    }

    async function attachTableActionHandlers() {
        // Edit cell handler
        document.querySelectorAll('.edit-cell-btn').forEach(btn => {
            btn.onclick = async () => {
                const rowId = btn.dataset.id;
                if (!rowId || rowId === 'undefined') {
                    alert('Invalid row ID');
                    return;
                }
                const row = btn.closest('tr');
                const cells = row.querySelectorAll('td[data-col]');
                const colNames = Array.from(cells).map(td => td.dataset.col);
                const col = await prompt(`选择要编辑的列 (可用列: ${colNames.join(', ')})`);
                if (!col || !colNames.includes(col)) return;
                const cell = row.querySelector(`td[data-col="${col}"]`);
                const oldValue = cell ? cell.textContent : '';
                const newValue = await prompt(`编辑 ${col} (行ID ${rowId}):`, oldValue);
                if (newValue === null || newValue === oldValue) return;
                const pin = await prompt('请输入管理员PIN以确认修改:');
                if (!pin) return;
                const res = await fetch('/admin/db_update_row', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({
                        table: currentTable,
                        row_id: rowId,
                        column: col,
                        value: newValue,
                        pin: pin
                    })
                });
                const data = await res.json();
                if (res.ok) {
                    showToast('修改成功', 'success', 3000);
                    loadTableData();
                } else {
                    alert('修改失败: ' + ('操作失败，请重试'));
                }
            };
        });

        // Delete row handler
        document.querySelectorAll('.delete-row-btn').forEach(btn => {
            btn.onclick = async () => {
                const rowId = btn.dataset.id;
                if (!rowId || rowId === 'undefined') {
                    alert('Invalid row ID');
                    return;
                }
                if (!await confirm(`确定要删除行ID ${rowId} 吗？此操作不可恢复。`)) return;
                const pin = await prompt('请输入管理员PIN以确认删除:');
                if (!pin) return;
                const res = await fetch('/admin/db_delete_row', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({
                        table: currentTable,
                        row_id: rowId,
                        pin: pin
                    })
                });
                const data = await res.json();
                if (res.ok) {
                    showToast('删除成功', 'success', 3000);
                    loadTableData();
                } else {
                    alert('删除失败: ' + ('操作失败，请重试'));
                }
            };
        });
    }

    function startAutoRefresh() {
        if (autoRefreshInterval) clearInterval(autoRefreshInterval);
        autoRefreshInterval = setInterval(() => {
            if (autoRefreshEnabled && currentTable) loadTableData();
        }, autoRefreshSeconds * 1000);
        updateTimerDisplay();
    }

    function updateTimerDisplay() {
        const timerSpan = document.getElementById('autoRefreshTimer');
        if (timerSpan) {
            timerSpan.textContent = `${Math.floor(autoRefreshSeconds/60)}:${(autoRefreshSeconds%60).toString().padStart(2,'0')}`;
        }
    }

    document.getElementById('dbTableSelect')?.addEventListener('change', (e) => {
        currentTable = e.target.value;
        currentPage = 1;
        currentSearch = '';
        currentSearchColumn = '';
        document.getElementById('dbSearchInput').value = '';
        document.getElementById('dbSearchColumnSelect').innerHTML = '<option value="">所有文本列</option>';
        if (currentTable) {
            fetch('/admin/db_table_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ table: currentTable, page: 1, per_page: 1 })
            }).then(r => r.json()).then(data => {
                const select = document.getElementById('dbSearchColumnSelect');
                if (data.columns) {
                    data.columns.forEach(col => {
                        select.innerHTML += `<option value="${escapeHtml(col)}">${escapeHtml(col)}</option>`;
                    });
                }
            }).catch(console.error);
            loadTableData();
        } else {
            document.getElementById('dbTableBody').innerHTML = '<td><td colspan="10">请选择表</td></tr>';
            document.getElementById('dbTableHeader').innerHTML = '';
        }
    });

    document.getElementById('dbRefreshBtn')?.addEventListener('click', () => {
        if (currentTable) loadTableData();
    });

    document.getElementById('dbSearchInput')?.addEventListener('input', debounce((e) => {
        currentSearch = e.target.value;
        currentPage = 1;
        if (currentTable) loadTableData();
    }, 500));

    document.getElementById('dbSearchColumnSelect')?.addEventListener('change', (e) => {
        currentSearchColumn = e.target.value;
        currentPage = 1;
        if (currentTable) loadTableData();
    });

    document.getElementById('dbPerPageSelect')?.addEventListener('change', (e) => {
        currentPerPage = parseInt(e.target.value);
        currentPage = 1;
        if (currentTable) loadTableData();
    });

    document.getElementById('dbAutoRefreshToggle')?.addEventListener('click', () => {
        autoRefreshEnabled = !autoRefreshEnabled;
        const btn = document.getElementById('dbAutoRefreshToggle');
        if (autoRefreshEnabled) {
            btn.textContent = '暂停';
            startAutoRefresh();
        } else {
            btn.textContent = '启动';
            if (autoRefreshInterval) clearInterval(autoRefreshInterval);
            autoRefreshInterval = null;
        }
    });

    // ======================== Authentication & Fetch Interceptor ========================
    async function verifyAuth() {
        try {
            const res = await fetch('/check_auth', { credentials: 'include' });
            const data = await res.json();
            if (!data.authenticated) {
                sessionStorage.clear();
                if (data.consent_given) location.reload();
            } else {
                sessionStorage.setItem('username', data.username || '');
                if (data.is_admin) sessionStorage.setItem('isAdmin', 'true');
                else sessionStorage.removeItem('isAdmin');
                const adminExtras = document.getElementById('sidebarAdminExtras');
                if (adminExtras) adminExtras.style.display = data.is_admin ? '' : 'none';
                sessionStorage.setItem('is_auditor', (data.is_auditor || data.is_admin) ? '1' : '0');
                if (data.role) sessionStorage.setItem('role', data.role);
                sessionStorage.setItem('user_id', data.user_id || '');
                // Remove anonymous watermark
                const wm = document.getElementById('anonWatermark');
                if (wm) wm.remove();
            }
            const databaseTab = document.getElementById('databaseTabBtn');
            if (databaseTab) {
                if (data.is_admin) databaseTab.style.display = 'inline-block';
                else databaseTab.style.display = 'none';
            }
            // Knowledge Lab tab: registered only
            const knowledgeLabTabBtn = document.getElementById('knowledgeLabTabBtn');
            if (knowledgeLabTabBtn) {
                knowledgeLabTabBtn.style.display = data.authenticated ? 'inline-block' : 'none';
            }
            // Knowledge Base button in chat toolbar: registered only
            const kbBtn = document.getElementById('knowledgeBaseBtn');
            if (kbBtn) {
                kbBtn.style.display = data.authenticated ? '' : 'none';
            }
            // Daily report button in chat toolbar: registered only
            const dailyChatBtn = document.getElementById('dailyReportChatBtn');
            if (dailyChatBtn) {
                dailyChatBtn.style.display = data.authenticated ? '' : 'none';
            }

            // Review tab for admins and auditors
            const reviewTab = document.getElementById('reviewTabBtn');
            if (reviewTab) {
                if (data.is_admin || data.is_auditor) {
                    reviewTab.style.display = 'inline-block';
                }
            }
            // Anonymous watermark — consolidated restriction notice
            if (!data.authenticated) {
                const chatEl = document.getElementById('chatMessages');
                const existing = document.getElementById('anonWatermark');
                if (chatEl && !existing) {
                    const wm = document.createElement('div');
                    wm.id = 'anonWatermark';
                    wm.style.cssText = 'position:absolute;bottom:80px;left:50%;transform:translateX(-50%);font-size:.72rem;color:var(--card-muted);opacity:.4;pointer-events:none;z-index:0;text-align:center;line-height:1.6;max-width:85%;';
                    wm.textContent = '注册即可解锁：知识库 · 项目管理 · 文件站下载 · 聊天搜索 · 批量对比 · 工作报告';
                    chatEl.style.position = 'relative';
                    chatEl.appendChild(wm);
                }
                // Hide search button for anons
                const searchBtn = document.getElementById('openSearchModalBtn');
                if (searchBtn) searchBtn.style.display = 'none';
            } else {
                const searchBtn = document.getElementById('openSearchModalBtn');
                if (searchBtn) searchBtn.style.display = '';
            }
        } catch (err) {
            console.error('Auth check failed', err);
        }
    }

    const originalFetch = window.fetch;
    window.fetch = function(...args) {
        const method = args[1]?.method || 'GET';
        if (method && ['POST', 'PUT', 'DELETE', 'PATCH'].includes(method.toUpperCase())) {
            if (!args[1]) args[1] = {};
            if (!args[1].headers) args[1].headers = {};
            args[1].headers['X-CSRFToken'] = csrfToken;
            // Only set credentials if not already specified
            if (args[1].credentials === undefined) {
                args[1].credentials = 'include';
            }
        }
        return originalFetch.apply(this, args);
    };

    // ======================== Chat Search Modal ========================
    const openSearchModalBtn = document.getElementById('openSearchModalBtn');
    const searchModal = document.getElementById('searchModal');
    const closeSearchModal = document.getElementById('closeSearchModal');
    const dateFilterToggle = document.getElementById('dateFilterToggle');
    const dateRangeDiv = document.getElementById('dateRange');
    const doSearchBtn = document.getElementById('doSearchBtn');
    const searchQueryInput = document.getElementById('searchQuery');
    const searchResultsDiv = document.getElementById('searchResults');

    if (openSearchModalBtn) {
        openSearchModalBtn.onclick = () => {
            searchModal.style.display = 'block';
            searchResultsDiv.innerHTML = '<p>输入关键词后点击搜索</p>';
            searchQueryInput.value = '';
        };
    }
    if (closeSearchModal) {
        closeSearchModal.onclick = () => { searchModal.style.display = 'none'; };
    }
    if (dateFilterToggle) {
        dateFilterToggle.onchange = () => {
            dateRangeDiv.style.display = dateFilterToggle.checked ? 'flex' : 'none';
        };
    }
    if (doSearchBtn) {
        doSearchBtn.onclick = async () => {
            const query = searchQueryInput.value.trim();
            if (query.length < 2) {
                alert('搜索词至少2个字符');
                return;
            }
            const role = document.querySelector('input[name="searchRole"]:checked').value;
            const fuzzy = document.getElementById('fuzzySearch').checked;
            let startDate = '', endDate = '';
            if (dateFilterToggle.checked) {
                startDate = document.getElementById('startDate').value;
                endDate = document.getElementById('endDate').value;
            }
            const params = new URLSearchParams({
                q: query,
                role: role,
                fuzzy: fuzzy,
                start_date: startDate,
                end_date: endDate
            });
            searchResultsDiv.innerHTML = '<p>搜索中...</p>';
            try {
                const res = await fetch(`/search_chat?${params.toString()}`, { credentials: 'include' });
                const data = await res.json();
                if (data.error) {
                    searchResultsDiv.innerHTML = '<p>搜索失败，请重试</p>';
                    return;
                }
                const results = data.results || [];
                if (results.length === 0) {
                    searchResultsDiv.innerHTML = '<p>未找到匹配的聊天记录。</p>';
                    return;
                }
                let html = '<ul style="list-style: none; padding: 0;">';
                for (const r of results) {
                    const roleLabel = r.role === 'assistant' ? '🤖 助手' : '👤 用户';
                    html += `
                        <li style="border-bottom:1px solid #ddd; padding:12px 0;">
                            <div><strong>${escapeHtml(r.title)}</strong> <small>${roleLabel}</small> <small>${r.timestamp_str}</small></div>
                            <div style="margin-top:6px;">${r.highlighted_snippet}...</div>
                            <button class="load-session-btn"
                                    data-thread="${escapeHtml(r.thread_id)}"
                                    data-message-id="${escapeHtml(r.message_id)}"
                                    style="margin-top:6px; background:#2c3e50; color:white; border:none; border-radius:4px; padding:4px 8px;">跳转到会话</button>
                        </li>
                    `;
                }
                html += '</ul>';
                searchResultsDiv.innerHTML = html;
                document.querySelectorAll('.load-session-btn').forEach(btn => {
                    btn.onclick = () => {
                        const threadId = btn.getAttribute('data-thread');
                        const messageId = btn.getAttribute('data-message-id');
                        if (!messageId) {
                            console.warn('No message-id on button');
                        }
                        searchModal.style.display = 'none';
                        loadSession(threadId, true, messageId);
                    };
                });
            } catch (err) {
                console.error(err);
                searchResultsDiv.innerHTML = '<p>搜索失败，请重试。</p>';
            }
        };
    }
    searchQueryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') doSearchBtn.click();
    });

    // ======================== My Files Button ========================
    const myFilesBtn = document.getElementById('myFilesBtn');
    if (myFilesBtn) {
        myFilesBtn.onclick = () => {
            fileStationModal.style.display = 'block';
            loadFileStation();
        };
    }
    // Theme toggle logic
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.body.classList.contains('dark') ? 'dark' : 'light';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            setTheme(newTheme);
        });
    }
    // Load saved theme from localStorage
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        setTheme(savedTheme);
    } else {
        // Default to light
        setTheme('light');
    }

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

    // ======================== Message Editing ========================
    messagesDiv.addEventListener('dblclick', function(e) {
        const userMsg = e.target.closest('.user-message');
        if (!userMsg || isProcessing) return;
        const originalText = userMsg.textContent.trim();
        messageInput.value = originalText;
        messageInput.focus();
        messageInput.setSelectionRange(originalText.length, originalText.length);
        showToast('消息已载入输入框，修改后按 Enter 发送', 'info');
    });

    // ======================== Inline Feedback ========================
    function addFeedbackButtons(groupEl, msgId) {
        if (groupEl.querySelector('.feedback-btns')) return;
        const btns = document.createElement('div');
        btns.className = 'feedback-btns';
        btns.innerHTML = '<button class="fb-btn" data-rating="up">👍</button><button class="fb-btn" data-rating="down">👎</button>';
        btns.querySelectorAll('.fb-btn').forEach(btn => {
            btn.onclick = async function() {
                const rating = this.dataset.rating;
                const userMsg = groupEl.dataset.userMsg || '';
                const assistantDiv = groupEl.querySelector('.assistant-answer');
                const assistantMsg = assistantDiv ? assistantDiv.textContent.trim() : '';
                try {
                    await fetch('/feedback', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        credentials: 'include',
                        body: JSON.stringify({ rating, user_message: userMsg, assistant_response: assistantMsg })
                    });
                    btns.innerHTML = rating === 'up' ? '👍 已反馈' : '👎 已反馈';
                    btns.style.pointerEvents = 'none';
                    showToast('感谢反馈!', 'success');
                } catch (e) {
                    showToast('反馈提交失败', 'error');
                }
            };
        });
        groupEl.appendChild(btns);
    }

    // ======================== Prompt Templates ========================
    function loadTemplates() {
        try { return JSON.parse(localStorage.getItem('promptTemplates') || '[]'); }
        catch (e) { return []; }
    }
    function saveTemplates(list) { localStorage.setItem('promptTemplates', JSON.stringify(list)); }

    const promptTemplatesBtn = document.getElementById('promptTemplatesBtn');
    if (promptTemplatesBtn) {
        promptTemplatesBtn.onclick = function() {
            const templates = loadTemplates();
            const modal = document.createElement('div');
            modal.className = 'modal'; modal.style.display = 'flex';
            const listHtml = templates.length === 0
                ? '<p style="color:#999;">暂无模板，在输入框输入内容后点击保存</p>'
                : templates.map((t, i) => `<div class="template-item" data-idx="${i}">
                    <span class="template-text">${escapeHtml(t.name || t.text.slice(0,40))}</span>
                    <div class="template-actions">
                        <button class="tmpl-use" data-idx="${i}">📥 使用</button>
                        <button class="tmpl-del" data-idx="${i}">🗑</button>
                    </div></div>`).join('');
            modal.innerHTML = `<div class="modal-content" style="width:500px;max-width:90%">
                <span class="close">&times;</span>
                <h3>📋 提示词模板</h3>
                <div style="display:flex;gap:8px;margin-bottom:12px">
                    <input id="tmplName" placeholder="模板名称(可选)" style="flex:1;padding:6px">
                </div>
                <div class="template-list">${listHtml}</div>
                <button id="saveTemplateBtn" class="file-btn" style="background:#27ae60;color:white;margin-top:10px;">💾 保存当前输入为模板</button>
            </div>`;
            document.body.appendChild(modal);
            modal.querySelector('.close').onclick = () => modal.remove();
            modal.onclick = e => { if (e.target === modal) modal.remove(); };

            // Save
            modal.querySelector('#saveTemplateBtn').onclick = () => {
                const text = messageInput.value.trim();
                if (!text) return;
                const name = modal.querySelector('#tmplName').value.trim() || text.slice(0, 30);
                const tmpl = { name, text, time: Date.now() };
                const list = loadTemplates();
                list.push(tmpl);
                saveTemplates(list);
                showToast('模板已保存', 'success');
                modal.remove();
            };
            // Use
            modal.querySelectorAll('.tmpl-use').forEach(btn => {
                btn.onclick = () => {
                    const idx = parseInt(btn.dataset.idx);
                    messageInput.value = templates[idx].text;
                    modal.remove();
                    messageInput.focus();
                };
            });
            // Delete
            modal.querySelectorAll('.tmpl-del').forEach(btn => {
                btn.onclick = () => {
                    const idx = parseInt(btn.dataset.idx);
                    const list = loadTemplates();
                    list.splice(idx, 1);
                    saveTemplates(list);
                    modal.remove();
                    showToast('模板已删除', 'info');
                };
            });
        };
    }

    // ======================== Conversation Pinning ========================
    pinnedSessions = new Set();
    try { pinnedSessions = new Set(JSON.parse(localStorage.getItem('pinnedSessions') || '[]')); } catch(e) {}

    function persistPins() { localStorage.setItem('pinnedSessions', JSON.stringify([...pinnedSessions])); }

    function addPinButton(li, threadId) {
        // Pin now lives in the action area (right side, next to archive)
        // This is called from loadHistoryList inline
    }

    function sortPinnedFirst(items) {
        if (!pinnedSessions) return items;
        return [...items].sort((a, b) => {
            const aPin = pinnedSessions.has(a.thread_id) ? 0 : 1;
            const bPin = pinnedSessions.has(b.thread_id) ? 0 : 1;
            return aPin - bPin;
        });
    }

    // ======================== Sidebar Quick Search ========================
    const sidebarSearch = document.getElementById('sidebarSearch');
    if (sidebarSearch) {
        sidebarSearch.addEventListener('input', function() {
            const q = this.value.toLowerCase();
            document.querySelectorAll('.history-item').forEach(li => {
                const title = (li.querySelector('.session-title') || li).textContent.toLowerCase();
                li.style.display = q === '' || title.includes(q) ? '' : 'none';
            });
        });
    }

    // ======================== Keyboard Shortcut Cheatsheet ========================
    document.addEventListener('keydown', function(e) {
        if (e.key === '?' && !e.ctrlKey && !e.metaKey && !e.altKey && document.activeElement === document.body) {
            e.preventDefault();
            const sheet = document.createElement('div');
            sheet.className = 'shortcut-sheet';
            sheet.innerHTML = `<div class="shortcut-content">
                <span class="close" onclick="this.parentElement.parentElement.remove()">&times;</span>
                <h3>⌨️ 快捷键</h3>
                <table><tbody>
                <tr><td><kbd>Enter</kbd></td><td>发送消息</td></tr>
                <tr><td><kbd>Escape</kbd></td><td>关闭弹窗 / 返回</td></tr>
                <tr><td><kbd>?</kbd></td><td>显示此帮助</td></tr>
                <tr><td><kbd>Delete</kbd> / <kbd>Backspace</kbd></td><td>删除选中文件</td></tr>
                <tr><td>双击用户消息</td><td>编辑并重发</td></tr>
                <tr><td>拖放文件</td><td>上传到对话</td></tr>
                </tbody></table></div>`;
            sheet.onclick = function(e) { if (e.target === sheet) sheet.remove(); };
            document.body.appendChild(sheet);
        }
    });

    // ======================== PWA Service Worker ========================
    if ('serviceWorker' in navigator) {
        let swRegistration = null;
        window.addEventListener('load', () => {
            navigator.serviceWorker.register('/sw.js').then(reg => {
                swRegistration = reg;
                console.log('[PWA] ServiceWorker registered:', reg.scope);
                reg.update();
            }).catch(err => {
                if (err.name === 'SecurityError' || String(err).includes('SSL')) {
                    console.info('[PWA] Self-signed cert — ServiceWorker skipped (requires valid CA cert). PWA install available via browser menu.');
                } else {
                    console.warn('[PWA] ServiceWorker failed:', err);
                }
            });
        });

        // Listen for update messages from SW
        navigator.serviceWorker.addEventListener('message', event => {
            if (event.data && event.data.type === 'SW_UPDATED') {
                showToast('应用已更新，请刷新页面以使用最新版本', 'info', 6000);
            }
        });

        // PWA install button — show for ALL users (not just admin)
        let deferredPrompt = null;
        const pwaInstallBtn = document.getElementById('pwaInstallBtn');

        // Show PWA button when user is authenticated
        function showPwaButton() {
            if (pwaInstallBtn) {
                pwaInstallBtn.style.display = '';
                pwaInstallBtn.style.visibility = 'visible';
            }
        }

        // Try to show after auth check completes
        setTimeout(async () => {
            try {
                const authRes = await fetch('/check_auth', { credentials: 'include' });
                const auth = await authRes.json();
                if (auth.authenticated) showPwaButton();
            } catch(e) {}
        }, 2000);

        // Also listen for the native install event
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            showPwaButton();
        });

        if (pwaInstallBtn) {
            pwaInstallBtn.onclick = function() {
                if (!deferredPrompt) {
                    showToast('请通过浏览器菜单安装 (Chrome地址栏右侧)', 'info');
                    return;
                }
                deferredPrompt.prompt();
                deferredPrompt.userChoice.then(result => {
                    console.log('[PWA] Install:', result.outcome);
                    deferredPrompt = null;
                    if (pwaInstallBtn) pwaInstallBtn.style.display = 'none';
                });
            };
        }
        window.addEventListener('appinstalled', () => {
            if (pwaInstallBtn) pwaInstallBtn.style.display = 'none';
            deferredPrompt = null;
            showToast('应用已安装到桌面!', 'success');
        });
    }

    // ======================== Project Presence Polling ========================
    let presenceInterval = null;
    function startPresencePolling(projectId) {
        stopPresencePolling();
        if (!projectId) return;
        fetch(`/admin/projects/${projectId}/ping`, { method: 'POST', credentials: 'include' });
        presenceInterval = setInterval(async () => {
            try {
                await fetch(`/admin/projects/${projectId}/ping`, { method: 'POST', credentials: 'include' });
                const res = await fetch(`/admin/projects/${projectId}/presence`, { credentials: 'include' });
                const data = await res.json();
                const indicator = document.getElementById('presenceIndicator');
                if (indicator) {
                    const count = Object.keys(data.active_users || {}).length;
                    indicator.textContent = count > 0 ? `🟢 ${count}人在线` : '';
                }
            } catch (e) { /* ignore */ }
        }, 30000);
    }
    function stopPresencePolling() {
        if (presenceInterval) { clearInterval(presenceInterval); presenceInterval = null; }
        const indicator = document.getElementById('presenceIndicator');
        if (indicator) indicator.textContent = '';
    }

    // ======================== Timeline Tab ========================
    const timelineTabBtn = document.getElementById('timelineTabBtn');
    const timelinePanel = document.getElementById('timelinePanel');
    if (timelineTabBtn && timelinePanel) {
        timelineTabBtn.onclick = async () => {
            stopRealtimePoll();
            saveActiveTab('timeline');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            timelineTabBtn.classList.add('active');
            switchToPanel('timelinePanel');
            switchSidebarPane('projects');
            showSubTabBar('projects');
            loadTimelinePanel();
        };
    }

    async function loadTimelinePanel() {
        if (!currentProjectId) {
            const content = document.getElementById('timelineContent');
            if (content) content.innerHTML = '<p style="color:var(--card-muted);">请先在"项目"标签页中选择一个项目。</p>';
            return;
        }
        const setup = document.getElementById('timelineSetup');
        const content = document.getElementById('timelineContent');
        content.innerHTML = '<p style="color:var(--card-muted);">加载时间线...</p>';

        try {
            const listRes = await fetch('/timeline/' + currentProjectId + '/list', { credentials: 'include' });
            const listData = await listRes.json();

            if (listData.success && listData.timelines && listData.timelines.length > 0) {
                setup.style.display = 'none';
                content.innerHTML = _renderTimelineList(listData.timelines);
                _loadTimelineDetail(currentProjectId, listData.timelines[0].id);
            } else {
                setup.style.display = 'block';
                content.innerHTML = '';
                _setupTimelineCreationForm(currentProjectId);
            }
        } catch (e) {
            content.innerHTML = '<p style="color:var(--card-muted);">加载失败: ' + e.message + '</p>';
        }
    }

    function _renderTimelineList(timelines) {
        var h = '<div style="margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;">';
        h += '<span style="font-weight:600;">📋 项目时间线</span>';
        h += '<button id="tlNewBtn" style="background:var(--accent);color:white;border:none;border-radius:4px;padding:4px 12px;font-size:0.7rem;cursor:pointer;">➕ 新建时间线</button>';
        h += '</div>';
        h += '<div id="timelineDetailArea" style="margin-bottom:12px;"></div>';
        h += '<div id="timelineListContainer" style="display:flex;flex-direction:column;gap:6px;">';
        timelines.forEach(function(t) {
            h += '<div class="tl-list-entry" data-tid="' + t.id + '" style="display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:var(--card-bg);border:1px solid var(--card-border);border-radius:8px;cursor:pointer;transition:border-color .2s;">';
            h += '<div>';
            h += '<div style="font-weight:600;font-size:0.82rem;">📊 ' + (t.name || '主招标流程') + '</div>';
            h += '<div style="font-size:0.68rem;color:var(--card-muted);">' + (t.category_code || '') + ' / ' + (t.method_code || '') + ' · ' + _statusBadge(t.status || 'active') + '</div>';
            h += '</div>';
            h += '<div style="text-align:right;">';
            h += '<div style="font-size:0.68rem;color:var(--card-muted);">' + (t.planned_start_date || '') + ' → ' + (t.planned_end_date || '-') + '</div>';
            h += '</div>';
            h += '</div>';
        });
        h += '</div>';
        return h;
    }

    async function _loadTimelineDetail(projectId, timelineId) {
        var area = document.getElementById('timelineDetailArea');
        area.innerHTML = '<p style="color:var(--card-muted);font-size:0.75rem;">加载中...</p>';
        try {
            var res = await fetch('/timeline/' + projectId + '?timeline_id=' + timelineId, { credentials: 'include' });
            var tl = await res.json();
            if (tl.success && tl.id) {
                area.innerHTML = _renderTimelineView(tl);
                _wireTimelineDetailActions(projectId, tl);
            } else {
                area.innerHTML = '<p style="color:var(--card-muted);">未找到时间线</p>';
            }
        } catch(e) {
            area.innerHTML = '<p style="color:#ef4444;">加载失败: ' + e.message + '</p>';
        }
        // Highlight active entry
        document.querySelectorAll('.tl-list-entry').forEach(function(e) { e.style.borderColor = 'var(--card-border)'; });
        var active = document.querySelector('.tl-list-entry[data-tid="' + timelineId + '"]');
        if (active) active.style.borderColor = 'var(--accent)';
    }

    function _setupTimelineCreationForm(projectId) {
        var catSel = document.getElementById('timelineCategorySelect');
        var mSel = document.getElementById('timelineMethodSelect');
        fetch('/timeline/legal/categories', { credentials: 'include' }).then(function(r) { return r.json(); }).then(function(catData) {
            if (catData.success && catData.categories) {
                catSel.innerHTML = '<option value="">选择类别...</option>';
                catData.categories.forEach(function(c) {
                    catSel.innerHTML += '<option value="' + c.code + '">' + c.name + '</option>';
                });
                catSel.onchange = function() {
                    mSel.innerHTML = '<option value="">选择方式...</option>';
                    var sel = catData.categories.find(function(c) { return c.code === catSel.value; });
                    if (sel && sel.methods) {
                        sel.methods.forEach(function(m) {
                            mSel.innerHTML += '<option value="' + m.code + '">' + m.name + '</option>';
                        });
                    }
                };
                document.getElementById('timelineCreateBtn').onclick = async function() {
                    var cat = catSel.value;
                    var meth = mSel.value;
                    var start = document.getElementById('timelineStartDate').value;
                    var name = document.getElementById('timelineNameInput') ? document.getElementById('timelineNameInput').value : '';
                    if (!cat || !meth || !start) { alert('请填写所有必填项'); return; }
                    try {
                        var body = {category_code: cat, method_code: meth, planned_start_date: start};
                        if (name) body.name = name;
                        var cr = await fetch('/timeline/' + projectId, {
                            method: 'POST', headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(body), credentials: 'include'
                        });
                        var cd = await cr.json();
                        if (cd.success) { loadTimelinePanel(); } else { alert(cd.error || '创建失败'); }
                    } catch(e) { alert('创建失败: ' + e.message); }
                };
            }
        });
    }

    function _renderTimelineView(tl) {
        var ms = tl.milestones || [];
        var html = '<div style="margin-bottom:12px;">';
        html += '<div style="font-weight:600;font-size:0.9rem;margin-bottom:6px;">📊 ' + (tl.name || '主招标流程') + '</div>';
        html += '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px;">';
        html += '<span style="font-size:0.75rem;"><b>类别:</b> ' + (tl.category_code || '') + '</span>';
        html += '<span style="font-size:0.75rem;"><b>方式:</b> ' + (tl.method_code || '') + '</span>';
        html += '<span style="font-size:0.75rem;"><b>计划开始:</b> ' + (tl.planned_start_date || '') + '</span>';
        html += '<span style="font-size:0.75rem;"><b>计划结束:</b> ' + (tl.planned_end_date || '-') + '</span>';
        if (tl.actual_start_date || tl.actual_end_date) {
            html += '<span style="font-size:0.75rem;"><b>实际:</b> ' + (tl.actual_start_date || '?') + ' → ' + (tl.actual_end_date || '进行中') + '</span>';
        }
        html += '<span style="font-size:0.75rem;"><b>状态:</b> ' + _statusBadge(tl.status || '') + '</span>';
        html += '</div>';
        if (tl.created_at) {
            html += '<div style="font-size:0.68rem;color:var(--card-muted);margin-bottom:4px;">创建于 ' + tl.created_at + (tl.created_by ? ' · 由 ' + tl.created_by : '') + (tl.updated_at && tl.updated_at !== tl.created_at ? ' · 更新于 ' + tl.updated_at : '') + '</div>';
        }

        if (tl.diff_summary) {
            var ds = tl.diff_summary;
            html += '<div style="font-size:0.7rem;color:var(--card-muted);margin-bottom:8px;">';
            html += '总节点: ' + ds.total_milestones + ' | 已完成: ' + ds.completed;
            html += ' | 待处理: ' + ds.pending + ' | 延期: ' + ds.delayed;
            html += ' | 准点: ' + ds.on_time + ' | 提前: ' + ds.advanced;
            if (ds.total_delay_days > 0) html += ' | 累计延期: ' + ds.total_delay_days + '天';
            html += '</div>';
        }
        html += '</div>';

        html += '<div style="max-height:500px;overflow-y:auto;border:1px solid var(--card-border);border-radius:8px;">';
        ms.forEach(function(m, i) {
            var bg = m.status === 'completed' ? '#f0fff4' : (m.diff_days && m.diff_days > 0 ? '#fff5f5' : 'transparent');
            var reasonTip = m.diff_reason ? ' title="原因: ' + m.diff_reason.replace(/"/g, '&quot;') + '"' : '';
            html += '<div style="display:flex;align-items:center;padding:6px 12px;border-bottom:1px solid var(--card-border);background:' + bg + ';font-size:0.72rem;"' + reasonTip + '>';
            html += '<span style="width:24px;font-weight:700;color:var(--card-muted);">' + (i+1) + '</span>';
            html += '<span style="flex:1;font-weight:600;">' + (m.name || m.code) + '</span>';
            html += '<span style="width:90px;text-align:center;color:var(--card-muted);">' + (m.planned_date || '待定') + '</span>';
            html += '<span style="width:90px;text-align:center;' + (m.diff_days && m.diff_days > 0 ? 'color:#e53e3e;' : '') + '">' + (m.actual_date || '未完成') + '</span>';
            if (m.diff_days && m.diff_days > 0) {
                html += '<span style="width:60px;text-align:center;color:#e53e3e;">+' + m.diff_days + '天</span>';
            } else if (m.diff_days && m.diff_days < 0) {
                html += '<span style="width:60px;text-align:center;color:#38a169;">' + m.diff_days + '天</span>';
            } else {
                html += '<span style="width:60px;text-align:center;">-</span>';
            }
            if (m.reason_category) {
                html += '<span style="width:70px;text-align:center;font-size:0.62rem;background:#f1f5f9;padding:1px 6px;border-radius:10px;color:#475569;">' + m.reason_category + '</span>';
            }
            html += _statusBadge(m.status || 'pending');
            html += '</div>';
        });
        html += '</div>';

        if (tl.diff_summary && tl.diff_summary.by_reason_category && Object.keys(tl.diff_summary.by_reason_category).length > 0) {
            var cats = tl.diff_summary.by_reason_category;
            html += '<div style="margin-top:8px;font-size:0.7rem;border:1px solid var(--card-border);border-radius:8px;padding:8px;">';
            html += '<b style="font-size:0.72rem;">延期原因分类</b>';
            html += '<div style="display:flex;gap:10px;margin-top:4px;flex-wrap:wrap;">';
            Object.keys(cats).forEach(function(k) {
                html += '<span style="background:#f1f5f9;padding:2px 10px;border-radius:10px;">' + k + ': <b>' + cats[k] + '</b></span>';
            });
            html += '</div></div>';
        }

        html += '<div style="margin-top:12px;display:flex;gap:6px;">';
        html += '<button onclick="loadTimelinePanel()" class="file-btn" style="font-size:0.7rem;">🔄 刷新</button>';
        html += '<button onclick="showSuggestions(' + tl.project_id + ')" class="file-btn" style="font-size:0.7rem;">💡 查看建议</button>';
        html += '<button onclick="showDiffReport(' + tl.project_id + ')" class="file-btn" style="font-size:0.7rem;">📊 差异报告</button>';
        html += '</div>';

        return html;
    }

    function _statusBadge(status) {
        var colors = {completed:'#38a169',pending:'#a0aec0',active:'#3182ce',overdue:'#e53e3e',delayed:'#e53e3e'};
        var labels = {completed:'已完成',pending:'待处理',active:'进行中',overdue:'已超期',delayed:'延期',running:'运行中',failed:'失败',PASS:'通过',FAIL:'不通过'};
        var c = colors[status] || '#a0aec0';
        var l = labels[status] || status;
        return '<span style="display:inline-block;background:' + c + ';color:#fff;padding:1px 6px;border-radius:4px;font-size:0.6rem;margin-left:4px;">' + l + '</span>';
    }

    function _closeTimelineModal() {
        var ov = document.querySelector('.tl-modal-overlay');
        if (ov) ov.remove();
    }

    window.showDiffReport = async function(projectId) {
        _closeTimelineModal();
        var ov = document.createElement('div');
        ov.className = 'tl-modal-overlay';
        ov.onclick = function(e) { if (e.target === ov) _closeTimelineModal(); };
        ov.innerHTML = '<div class="tl-modal"><div class="tl-modal-header"><h3>差异报告</h3><button class="tl-modal-close" onclick="_closeTimelineModal()">×</button></div><div class="tl-modal-body" style="text-align:center;color:var(--card-muted);">加载中...</div></div>';
        document.body.appendChild(ov);
        try {
            var res = await fetch('/timeline/' + projectId + '/diff', { credentials: 'include' });
            var d = await res.json();
            if (!d.success) { ov.querySelector('.tl-modal-body').innerHTML = '<p style="color:#ef4444;">' + (d.error || '加载失败') + '</p>'; return; }
            var s = d.summary || {};
            var body = '<div class="tl-summary-grid">';
            body += '<div class="tl-summary-item"><div class="tl-summary-val">' + (s.total_milestones || 0) + '</div><div class="tl-summary-label">总里程碑</div></div>';
            body += '<div class="tl-summary-item"><div class="tl-summary-val" style="color:#16a34a;">' + (s.completed || 0) + '</div><div class="tl-summary-label">已完成</div></div>';
            body += '<div class="tl-summary-item"><div class="tl-summary-val" style="color:#ea580c;">' + (s.delayed || 0) + '</div><div class="tl-summary-label">延期</div></div>';
            body += '<div class="tl-summary-item"><div class="tl-summary-val">' + (s.on_time || 0) + '</div><div class="tl-summary-label">准点</div></div>';
            body += '<div class="tl-summary-item"><div class="tl-summary-val">' + (s.pending || 0) + '</div><div class="tl-summary-label">待处理</div></div>';
            body += '<div class="tl-summary-item"><div class="tl-summary-val" style="color:#2563eb;">' + (s.advanced || 0) + '</div><div class="tl-summary-label">提前</div></div>';
            if (s.total_delay_days > 0) body += '<div class="tl-summary-item"><div class="tl-summary-val" style="color:#dc2626;">+' + s.total_delay_days + '</div><div class="tl-summary-label">累计延期(天)</div></div>';
            body += '</div>';

            if (s.by_reason_category && Object.keys(s.by_reason_category).length > 0) {
                body += '<h4 style="font-size:0.85rem;margin-bottom:6px;">按原因分类</h4>';
                body += '<div class="tl-table-wrap"><table><thead><tr><th>原因类别</th><th>数量</th></tr></thead><tbody>';
                Object.keys(s.by_reason_category).forEach(function(k) {
                    body += '<tr><td>' + k + '</td><td>' + s.by_reason_category[k] + '</td></tr>';
                });
                body += '</tbody></table></div>';
            }

            if (d.milestones && d.milestones.length > 0) {
                body += '<h4 style="font-size:0.85rem;margin-bottom:6px;">里程碑详情</h4>';
                body += '<div class="tl-table-wrap"><table><thead><tr><th>节点名称</th><th>计划日期</th><th>实际日期</th><th>差异(天)</th><th>状态</th><th>原因</th></tr></thead><tbody>';
                d.milestones.forEach(function(m) {
                    var delayStyle = m.diff_days > 0 ? 'color:#dc2626;font-weight:600;' : (m.diff_days < 0 ? 'color:#16a34a;' : '');
                    body += '<tr><td>' + (m.name || m.code) + '</td>';
                    body += '<td>' + (m.planned_date || '-') + '</td>';
                    body += '<td>' + (m.actual_date || '-') + '</td>';
                    body += '<td style="' + delayStyle + '">' + (m.diff_days != null ? (m.diff_days > 0 ? '+' : '') + m.diff_days : '-') + '</td>';
                    body += '<td>' + (m.status || '-') + '</td>';
                    body += '<td style="max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + (m.diff_reason || m.reason_category || '-') + '</td></tr>';
                });
                body += '</tbody></table></div>';
            }

            if (d.diff_log && d.diff_log.length > 0) {
                body += '<h4 style="font-size:0.85rem;margin-bottom:6px;">差异日志 (' + d.diff_log.length + '条)</h4>';
                body += '<div class="tl-table-wrap"><table><thead><tr><th>节点</th><th>计划日期</th><th>实际日期</th><th>差异</th><th>原因</th><th>时间</th></tr></thead><tbody>';
                d.diff_log.forEach(function(lg) {
                    body += '<tr><td>' + (lg.milestone_code || '-') + '</td>';
                    body += '<td>' + (lg.planned_date || '-') + '</td>';
                    body += '<td>' + (lg.actual_date || '-') + '</td>';
                    body += '<td>' + (lg.diff_days != null ? (lg.diff_days > 0 ? '+' : '') + lg.diff_days + '天' : '-') + '</td>';
                    body += '<td style="max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + (lg.reason_detail || lg.reason_category || '-') + '</td>';
                    body += '<td>' + (lg.created_at || '-') + '</td></tr>';
                });
                body += '</tbody></table></div>';
            }

            ov.querySelector('.tl-modal-body').innerHTML = body;
        } catch(e) {
            ov.querySelector('.tl-modal-body').innerHTML = '<p style="color:#ef4444;">加载失败: ' + e.message + '</p>';
        }
    };

    window.showSuggestions = async function(projectId) {
        _closeTimelineModal();
        var ov = document.createElement('div');
        ov.className = 'tl-modal-overlay';
        ov.onclick = function(e) { if (e.target === ov) _closeTimelineModal(); };
        ov.innerHTML = '<div class="tl-modal"><div class="tl-modal-header"><h3>规则建议</h3><button class="tl-modal-close" onclick="_closeTimelineModal()">×</button></div><div class="tl-modal-body" style="text-align:center;color:var(--card-muted);">加载中...</div></div>';
        document.body.appendChild(ov);

        async function _loadSuggestions() {
            try {
                var res = await fetch('/timeline/' + projectId + '/suggestions', { credentials: 'include' });
                var d = await res.json();
                if (!d.success) { ov.querySelector('.tl-modal-body').innerHTML = '<p style="color:#ef4444;">' + (d.error || '加载失败') + '</p>'; return; }
                var suggestions = d.suggestions || [];
                var body = '';
                if (suggestions.length === 0) {
                    body = '<p style="text-align:center;color:var(--card-muted);padding:24px 0;">暂无建议</p>';
                } else {
                    var priorityLabels = { critical: '严重', high: '高', medium: '中', info: '信息' };
                    suggestions.forEach(function(s) {
                        var p = s.priority || 'medium';
                        body += '<div class="tl-suggestion-card">';
                        body += '<span class="tl-priority-badge tl-priority-' + p + '">' + (priorityLabels[p] || p) + '</span>';
                        body += '<div class="tl-suggestion-body">';
                        body += '<div class="tl-suggestion-content">' + (s.content || '') + '</div>';
                        if (s.suggestion) body += '<div class="tl-suggestion-advice">' + s.suggestion + '</div>';
                        body += '</div>';
                        body += '<button class="tl-suggestion-dismiss" onclick="event.stopPropagation();dismissSuggestion(' + projectId + ',' + s.id + ',this)">忽略</button>';
                        body += '</div>';
                    });
                }
                body += '<div class="tl-modal-actions"><button class="file-btn" style="font-size:0.72rem;" onclick="generateAiSuggestions(' + projectId + ')">🤖 AI生成建议</button></div>';
                ov.querySelector('.tl-modal-body').innerHTML = body;
            } catch(e) {
                ov.querySelector('.tl-modal-body').innerHTML = '<p style="color:#ef4444;">加载失败: ' + e.message + '</p>';
            }
        }
        _loadSuggestions();
    };

    window.dismissSuggestion = async function(projectId, suggestionId, btn) {
        btn.disabled = true;
        btn.textContent = '已忽略';
        try {
            await fetch('/timeline/' + projectId + '/suggestions/' + suggestionId, { method: 'POST', credentials: 'include' });
            var card = btn.closest('.tl-suggestion-card');
            if (card) card.style.opacity = '0.4';
        } catch(e) { /* silent */ }
    };

    window.generateAiSuggestions = async function(projectId) {
        var ov = document.querySelector('.tl-modal-overlay');
        if (!ov) return;
        var body = ov.querySelector('.tl-modal-body');
        body.innerHTML = '<p style="text-align:center;color:var(--card-muted);">AI分析中，请稍候...</p>';
        try {
            var res = await fetch('/timeline/' + projectId + '/suggestions/generate', { method: 'POST', credentials: 'include' });
            var d = await res.json();
            if (d.success) {
                showToast(d.message || 'AI建议已刷新', 'success');
                showSuggestions(projectId);
            } else {
                body.innerHTML = '<p style="color:#ef4444;">' + (d.error || 'AI分析失败') + '</p>';
            }
        } catch(e) {
            body.innerHTML = '<p style="color:#ef4444;">AI分析失败: ' + e.message + '</p>';
        }
    };

    // ======================== Usage Tab (Admin only) ========================
    const analyticsTabBtn = document.getElementById('analyticsTabBtn');
    const analyticsPanel = document.getElementById('analyticsPanel');
    if (analyticsTabBtn && analyticsPanel) {
        let _usageLoaded = { rc: false, assets: false, archives: false, styles: false, auditConfig: false, skillAudit: false };
        analyticsTabBtn.onclick = async function() {
            stopRealtimePoll();
            saveActiveTab('stats');
            showSubTabBar('stats');
            resetSubTabs('analyticsSubTabs');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            analyticsTabBtn.classList.add('active');
            switchToPanel('analyticsPanel');
            switchSidebarPane('stats');
            _usageLoaded = { rc: false, assets: false, archives: false, styles: false, auditConfig: false, skillAudit: false };
            const content = document.getElementById('analyticsContent');
            content.innerHTML = '<span style="font-size:0.72rem;color:var(--card-muted);">Loading...</span>';
            const adminSections = document.getElementById('analyticsAdminSections');
            if (adminSections) adminSections.style.display = 'none';
            try {
                const res = await fetch('/admin/analytics', { credentials: 'include' });
                if (res.status === 403) { content.innerHTML = '<span>需要登录</span>'; return; }
                if (res.status === 401) { content.innerHTML = '<span>请先登录</span>'; return; }
                if (!res.ok) { content.innerHTML = `<span style="color:#e74c3c;">服务器错误 (${res.status})</span>`; return; }
                const stats = await res.json();
                loadSidebarStats(stats);
                const isAdmin = stats.is_admin_view;
                const items = [];
                if (isAdmin) {
                    items.push(`<span title="用户总数">👥<b>${stats.total_users}</b></span>`, `<span title="24h活跃用户">🟢<b>${stats.active_users_24h}</b></span>`, `<span title="会话总数">💬<b>${stats.total_sessions}</b></span>`,
                        `<span title="消息总数">✉️<b>${stats.total_messages}</b></span>`, `<span title="今日消息">📨<b>${stats.messages_today||0}</b></span>`,
                        `<span title="存储用量">💾<b>${stats.storage_mb}MB</b></span>`,
                        `<span title="活跃项目数">📂<b>${stats.active_projects}</b></span>`);
                    if (stats.credit_checks != null) items.push(`<span title="信用查询">🔍<b>${stats.credit_checks}</b></span>`);
                    if (stats.rag_stats?.total > 0) items.push(`<span title="RAG索引数">🧠<b>${stats.rag_stats.total}</b></span>`);
                } else {
                    items.push(`<span title="会话总数">💬<b>${stats.total_sessions}</b></span>`, `<span title="消息总数">✉️<b>${stats.total_messages}</b></span>`,
                        `<span title="今日消息">📨<b>${stats.messages_today||0}</b></span>`,
                        `<span title="文件总数">📁<b>${stats.total_files||0}</b></span>`, `<span title="存储用量">💾<b>${stats.storage_mb}MB</b></span>`);
                }
                let sparkline = '';
                if (stats.messages_per_day?.length) {
                    const counts = stats.messages_per_day.map(d=>d.count);
                    const bars = ['\u2581','\u2582','\u2583','\u2584','\u2585','\u2586','\u2587','\u2588'];
                    sparkline = ' <span title="近30天消息量趋势" style="opacity:0.5;font-size:0.65rem;">'+counts.map(c=>bars[Math.min(Math.round(c/Math.max(...counts,1)*7),7)]).join('')+'</span>';
                }
                content.innerHTML = `<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:0.7rem;color:var(--card-muted);">${items.join(' \u00b7 ')}${sparkline}</div>`;
                if (adminSections && isAdmin) { adminSections.style.display = 'block'; _setupUsageLazySections(); }
            } catch (e) { console.error('Usage stats load error:', e); content.innerHTML = '<span style="color:#e74c3c;font-size:0.72rem;">加载失败</span>'; }
        };

        function _setupUsageLazySections() {
            const map = [
                ['rcDetails', 'rc', loadRuntimeConfig],
                ['assetDetails', 'assets', loadAssetManager],
                ['archiveDetails', 'archives', loadArchivedSessionsAdmin],
                ['stylesDetails', 'styles', loadStyleManager],
                ['skillAuditDetails', 'skillAudit', loadSkillAuditWorkspace],
            ];
            for (const [id, key, fn] of map) {
                const el = document.getElementById(id);
                if (el && !el._listenerSet) {
                    el._listenerSet = true;
                    el.addEventListener('toggle', () => { if (el.open && !_usageLoaded[key]) { _usageLoaded[key] = true; fn(); } });
                }
            }
        }
    }

    // ======================== Review Tab (Admin + Auditor) ========================
    const reviewTabBtn = document.getElementById('reviewTabBtn');
    var reviewPanel = document.getElementById('reviewPanel');
    if (reviewTabBtn && reviewPanel) {
        reviewTabBtn.onclick = async () => {
            stopRealtimePoll();
            saveActiveTab('review');
            showSubTabBar('stats');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            reviewTabBtn.classList.add('active');
            switchToPanel('reviewPanel');
            switchSidebarPane('review');
            if (window._initReviewPanel && typeof window._initReviewPanel === 'function') {
                window._initReviewPanel();
            }
        };
    }

    // ======================== Share Conversation ========================
    function addShareButton(wrapperEl) {
        if (!wrapperEl || wrapperEl.querySelector('.share-btn')) return;
        const btn = document.createElement('button');
        btn.className = 'share-btn';
        btn.textContent = '🔗';
        btn.title = '分享对话';
        btn.style.cssText = 'position:absolute;top:6px;right:82px;z-index:10;background:rgba(0,0,0,0.06);border:none;border-radius:4px;padding:3px 8px;font-size:0.75rem;cursor:pointer;opacity:0;transition:opacity 0.2s;';
        wrapperEl.addEventListener('mouseenter', () => btn.style.opacity = '1');
        wrapperEl.addEventListener('mouseleave', () => btn.style.opacity = '0');
        btn.onmouseenter = () => btn.style.opacity = '1';
        btn.onmouseleave = () => btn.style.opacity = '0';
        btn.onclick = async function() {
            try {
                const res = await fetch('/share_conversation', { method: 'POST', credentials: 'include' });
                if (!res.ok) { showToast('分享失败，请重试', 'error'); return; }
                const data = await res.json();
                if (data.share_url) {
                    await navigator.clipboard.writeText(data.share_url);
                    showToast('分享链接已复制到剪贴板 (7天有效)', 'success');
                } else {
                    showToast('分享失败，请重试', 'error');
                }
            } catch (e) { showToast('分享请求失败', 'error'); }
        };
        wrapperEl.appendChild(btn);
    }

    // ======================== Export Conversation ========================
    const exportChatBtn = document.getElementById('exportChatBtn');
    if (exportChatBtn) {
        exportChatBtn.onclick = function() {
            const messages = [];
            const groups = messagesDiv.querySelectorAll('.message-group');
            groups.forEach(g => {
                const user = g.querySelector('.user-message');
                const assistant = g.querySelector('.assistant-answer');
                const thinking = g.querySelector('.thinking-content');
                if (user) messages.push({ role: 'user', content: user.textContent.trim() });
                if (assistant) {
                    let content = assistant.textContent.trim();
                    if (thinking) {
                        content = '【思考】\n' + thinking.textContent.trim() + '\n\n【回答】\n' + content;
                    }
                    messages.push({ role: 'assistant', content: content });
                }
            });
            if (messages.length === 0) return;
            let md = '# AI 对话导出\n\n';
            md += `导出时间: ${new Date().toLocaleString()}\n\n---\n\n`;
            messages.forEach(m => {
                const label = m.role === 'user' ? '👤 用户' : '🤖 AI助手';
                md += `### ${label}\n\n${m.content}\n\n---\n\n`;
            });
            const blob = new Blob([md], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `chat_export_${new Date().toISOString().slice(0,10)}.md`;
            a.click();
            URL.revokeObjectURL(url);
            showToast('对话已导出为 Markdown', 'success');
        };
    }

    // Call verifyAuth on every page load
    verifyAuth();

    // ======================== Project Chat: Right-click Context Menu + Todo + Quote ========================

    let _ctxMenuTarget = null;  // { messageId, content, role, author }
    let _currentQuoteContext = null;  // { quotedMessageId, parentQuoteId }

    const ctxMenu = document.getElementById('msgContextMenu');

    // Show context menu on right-click in project chat messages
    document.addEventListener('contextmenu', (e) => {
        if (!_isCurrentSessionProjectChat) return;
        // Find closest message group
        const group = e.target.closest('.message-group');
        if (!group) return;
        const msgId = group.dataset.msgId;
        if (!msgId || msgId.startsWith('temp-')) return;
        const userDiv = group.querySelector('.user-message');
        const answerDiv = group.querySelector('.assistant-answer');
        const isUser = !!userDiv;
        const content = userDiv ? userDiv.innerText : (answerDiv ? answerDiv.innerText : group.innerText);
        // Extract author from content if possible
        let author = '';
        let role = isUser ? 'user' : 'assistant';
        if (isUser) {
            const m = content.match(/^@(.+?):\s/);
            if (m) author = m[1];
        } else {
            const m = content.match(/^@(.+?)对@(.+?)说/);
            if (m) author = m[1];
        }
        _ctxMenuTarget = { messageId: msgId, content, role, author, domElement: group };
        ctxMenu.style.display = 'block';
        ctxMenu.style.left = Math.min(e.clientX, window.innerWidth - 180) + 'px';
        ctxMenu.style.top = Math.min(e.clientY, window.innerHeight - 120) + 'px';
        e.preventDefault();
    });

    // Single click handler: menu actions + auto-dismiss
    document.addEventListener('click', function(e) {
        const item = e.target.closest('.ctx-item');
        if (!item) {
            ctxMenu.style.display = 'none';
            _ctxMenuTarget = null;
            return;
        }
        if (!_ctxMenuTarget) { return; }
        ctxMenu.style.display = 'none';

        const action = item.dataset.action;
        console.log('  -> action:', action);
        const projectId = currentProjectId;
        if (!projectId) { showToast('请在项目对话中使用此功能', 'error'); console.log('  -> no projectId'); return; }

        if (action === 'todo') {
            const payload = {
                message_id: _ctxMenuTarget.messageId,
                content_copy: _ctxMenuTarget.content || '',
                original_role: _ctxMenuTarget.role || 'user',
                original_author: _ctxMenuTarget.author || '',
            };
            console.log('  -> sending todo:', payload);
            fetch(`/admin/projects/${projectId}/todos`, {
                method: 'POST', headers: {'Content-Type':'application/json'}, credentials: 'include',
                body: JSON.stringify(payload)
            }).then(r => r.json()).then(d => {
                console.log('  -> todo response:', d);
                if (d.success) { showToast('已添加到待办', 'success'); loadProjectTodos(projectId); }
                else showToast(d.error || '添加失败', 'error');
            }).catch(() => showToast('网络错误', 'error'));
        } else if (action === 'quote') {
            document.querySelectorAll('.quote-badge').forEach(b => b.remove());
            const targetEl = _ctxMenuTarget.domElement;
            if (targetEl) {
                if (!targetEl.style.position || targetEl.style.position === 'static') targetEl.style.position = 'relative';
                const badge = document.createElement('span');
                badge.className = 'quote-badge';
                badge.textContent = '已引用';
                badge.style.cssText = 'position:absolute;top:2px;right:8px;font-size:0.65rem;background:#3b82f6;color:white;padding:1px 6px;border-radius:10px;z-index:5;';
                badge.title = '此消息已被引用';
                targetEl.appendChild(badge);
            }
            const author = _ctxMenuTarget.author || '';
            const fullContent = _ctxMenuTarget.content || '';
            const lineCount = (fullContent.match(/\n/g) || []).length + 1;
            const charCount = fullContent.length;

            // Show collapsible quote bubble above input
            const bubble = document.getElementById('quoteBubble');
            const label = document.getElementById('quoteBubbleLabel');
            const content = document.getElementById('quoteBubbleContent');
            const icon = document.getElementById('quoteBubbleIcon');
            if (bubble && label && content && icon) {
                label.textContent = '引用' + (author ? ' @' + author : '') + ' — ' + lineCount + '行, ' + charCount + '字';
                content.textContent = fullContent;
                bubble.style.display = 'block';
                _toggleArrow(icon, true);  // collapsed
                content.style.display = 'none';
            }

            _currentQuoteContext = {
                quotedMessageId: _ctxMenuTarget.messageId,
                fullContent: fullContent,
                author: author,
            };
            const input = document.getElementById('messageInput');
            if (input) { input.focus(); }
        } else if (action === 'copy') {
            navigator.clipboard.writeText(_ctxMenuTarget.content).then(() => showToast('已复制', 'success')).catch(() => showToast('复制失败', 'error'));
        } else {
            // Clicked outside ctxMenu or on non-action element — dismiss
            ctxMenu.style.display = 'none';
        }
    });

    // Hover effect for context menu items
    ctxMenu.querySelectorAll('.ctx-item').forEach(item => {
        item.addEventListener('mouseenter', () => item.style.background = '#f3f4f6');
        item.addEventListener('mouseleave', () => item.style.background = '');
    });

    // Quote bubble: toggle expand/collapse
    const quoteBubble = document.getElementById('quoteBubble');
    if (quoteBubble) {
        quoteBubble.addEventListener('click', function(e) {
            if (e.target.id === 'quoteBubbleRemove') return; // handled separately
            const content = document.getElementById('quoteBubbleContent');
            const icon = document.getElementById('quoteBubbleIcon');
            if (content && icon) {
                const isHidden = content.style.display === 'none';
                content.style.display = isHidden ? 'block' : 'none';
                icon.textContent = isHidden ? '▼' : '▶';
            }
        });
        const removeBtn = document.getElementById('quoteBubbleRemove');
        if (removeBtn) {
            removeBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                document.getElementById('quoteBubble').style.display = 'none';
                document.getElementById('messageInput').rows = 1;
                document.querySelectorAll('.quote-badge').forEach(b => b.remove());
                _currentQuoteContext = null;
        });
    }

    function _wireTimelineDetailActions(projectId, tl) {
        setTimeout(function() {
            document.querySelectorAll('.tl-list-entry').forEach(function(el) {
                el.onclick = function() {
                    var tid = parseInt(el.getAttribute('data-tid'));
                    if (tid) _loadTimelineDetail(projectId, tid);
                };
            });
            var newBtn = document.getElementById('tlNewBtn');
            if (newBtn) newBtn.onclick = function() {
                var setup = document.getElementById('timelineSetup');
                setup.style.display = 'block';
                _setupTimelineCreationForm(projectId);
            };
        }, 50);
    }
    }

    // ======================== Todo Panel ========================

    async function loadProjectTodos(projectId) {
        if (!projectId) return;
        const todoHeader = document.getElementById('todoHeader');
        const todoList = document.getElementById('todoList');
        if (!todoHeader || !todoList) return;
        try {
            const res = await fetch(`/admin/projects/${projectId}/todos`, { credentials: 'include' });
            const d = await res.json();
            const todos = d.todos || [];
            todoHeader.style.display = '';
            todoList.style.display = '';
            document.getElementById('todoCount').textContent = todos.length;
            todoList.innerHTML = '';
            for (const t of todos) {
                const li = document.createElement('li');
                li.className = 'history-item';
                li.style.cssText = 'padding:6px 8px;border:1px solid #e5e7eb;border-radius:6px;margin-bottom:4px;font-size:0.72rem;position:relative;';
                const roleIcon = t.original_role === 'assistant' ? '🤖' : '👤';
                const preview = t.content_copy.length > 60 ? t.content_copy.substring(0, 60) + '...' : t.content_copy;
                li.innerHTML = `<div style="margin-bottom:4px;">${roleIcon} ${t.original_author ? '@'+escapeHtml(t.original_author)+': ' : ''}${escapeHtml(preview)}</div>`;
                const btnRow = document.createElement('div');
                btnRow.style.cssText = 'display:flex;gap:4px;';
                const doneBtn = document.createElement('button');
                doneBtn.textContent = '✓';
                doneBtn.style.cssText = 'flex:1;padding:2px 6px;font-size:0.65rem;border:1px solid #22c55e;background:#f0fdf4;color:#16a34a;border-radius:4px;cursor:pointer;';
                doneBtn.onclick = async (e) => {
                    e.stopPropagation();
                    const r = await fetch(`/admin/projects/${projectId}/todos/${t.id}/done`, { method:'POST', credentials:'include' });
                    const dd = await r.json();
                    if (dd.success) { showToast('已完成', 'success'); loadProjectTodos(projectId); }
                    else showToast(dd.error || '操作失败', 'error');
                };
                const removeBtn = document.createElement('button');
                removeBtn.textContent = '✕';
                removeBtn.style.cssText = 'flex:1;padding:2px 6px;font-size:0.65rem;border:1px solid #ef4444;background:#fef2f2;color:#dc2626;border-radius:4px;cursor:pointer;';
                removeBtn.onclick = async (e) => {
                    e.stopPropagation();
                    const r = await fetch(`/admin/projects/${projectId}/todos/${t.id}/remove`, { method:'POST', credentials:'include' });
                    const dd = await r.json();
                    if (dd.success) { showToast('已删除', 'success'); loadProjectTodos(projectId); }
                    else showToast(dd.error || '操作失败', 'error');
                };
                btnRow.appendChild(doneBtn);
                btnRow.appendChild(removeBtn);
                li.appendChild(btnRow);
                todoList.appendChild(li);
            }
            if (todos.length === 0) {
                todoList.innerHTML = '<li style="color:#999;font-size:0.72rem;">暂无待办</li>';
            }
        } catch(e) { console.warn('loadProjectTodos error:', e); }
    }

    // ======================== Regeneration Vote Panel ========================

    async function loadProjectVotes(projectId) {
        if (!projectId) return;
        try {
            const res = await fetch(`/admin/projects/${projectId}/regen_votes`, { credentials: 'include' });
            const d = await res.json();
            const votes = d.votes || [];
            // Remove old vote banners
            document.querySelectorAll('.vote-banner').forEach(el => el.remove());
            if (votes.length === 0) return;
            const chatPanel = document.getElementById('chatInterface');
            if (!chatPanel) return;
            for (const v of votes) {
                const banner = document.createElement('div');
                banner.className = 'vote-banner';
                banner.style.cssText = 'background:#fef3c7;border:1px solid #fcd34d;border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:0.75rem;';
                const expires = v.expires_at ? new Date(v.expires_at) : null;
                const remaining = expires ? Math.max(0, expires - Date.now()) : 0;
                const remainingH = Math.floor(remaining / 3600000);
                const remainingM = Math.floor((remaining % 3600000) / 60000);
                const expiringSoon = remaining < 3600000;
                const myVote = v.my_vote;
                banner.innerHTML = `
                    <div style="font-weight:600;color:#92400e;margin-bottom:4px;">⚠️ 内容差异投票 (第${v.round}轮) ${expiringSoon ? '⏰ 即将到期!' : ''}</div>
                    <div style="margin-bottom:4px;"><b>原内容:</b> ${escapeHtml(v.original_content)}...</div>
                    <div style="margin-bottom:6px;"><b>新内容:</b> ${escapeHtml(v.new_content)}...</div>
                    <div style="display:flex;gap:8px;align-items:center;">
                        <button class="vote-btn" data-vote="keep_original" data-id="${v.id}" style="padding:4px 12px;border-radius:6px;border:1px solid #22c55e;background:${myVote==='keep_original'?'#22c55e':'#f0fdf4'};color:${myVote==='keep_original'?'white':'#16a34a'};cursor:pointer;font-size:0.7rem;">保留原版 (${v.keep_count})</button>
                        <button class="vote-btn" data-vote="replace" data-id="${v.id}" style="padding:4px 12px;border-radius:6px;border:1px solid #f59e0b;background:${myVote==='replace'?'#f59e0b':'#fffbeb'};color:${myVote==='replace'?'white':'#d97706'};cursor:pointer;font-size:0.7rem;">替换新版 (${v.replace_count})</button>
                        <span style="color:#92400e;font-size:0.65rem;">剩余 ${remainingH}h ${remainingM}m</span>
                    </div>`;
                chatPanel.insertBefore(banner, chatPanel.firstChild);
                banner.querySelectorAll('.vote-btn').forEach(btn => {
                    btn.onclick = async () => {
                        const choice = btn.dataset.vote;
                        const vid = btn.dataset.id;
                        const r = await fetch(`/admin/projects/${projectId}/regen_votes/${vid}/cast`, {
                            method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                            body: JSON.stringify({ vote: choice })
                        });
                        const dd = await r.json();
                        if (dd.success) { showToast('投票已提交', 'success'); loadProjectVotes(projectId); }
                        else showToast(dd.error || '投票失败', 'error');
                    };
                });
            }
        } catch(e) { console.warn('loadProjectVotes error:', e); }
    }

    // ======================== Background Tasks Panel ========================

    let _bgTasksActiveSSE = null;
    let _bgTasksPollTimer = null;
    let _bgTasksBackoff = 0;
    let _bgTasksStopped = false;
    const BG_TASKS_BASE = 5000;

    async function loadBgTasks() {
        try {
            const res = await fetch('/tasks?limit=20', { credentials: 'include' });
            const data = await res.json();
            const list = document.getElementById('bgTasksList');
            const countSpan = document.getElementById('bgTaskCount');
            if (!list) return;
            const tasks = data.tasks || [];
            countSpan.textContent = tasks.length;
            if (tasks.length === 0) {
                _bgTasksBackoff++;
                list.innerHTML = '<li style="color:#999;font-size:0.72rem;">无进行中的任务</li>';
                return;
            }
            _bgTasksBackoff = 0;
            list.innerHTML = tasks.map(t => {
                const statusIcon = t.status === 'running' ? '🔄' : t.status === 'completed' ? '✅' : t.status === 'failed' ? '❌' : '⏳';
                const barWidth = t.progress || 0;
                const barColor = t.status === 'failed' ? '#ef4444' : t.status === 'completed' ? '#22c55e' : '#5a7c9b';
                const taskId = t.task_id || '';
                const isRunning = (t.status === 'running' || t.status === 'queued' || t.status === 'pending');
                const isDone = (t.status === 'completed' || t.status === 'failed');
                // running → cancel button; done → two-step delete button
                const actionBtn = isRunning
                    ? `<button class="task-cancel-btn" data-task-id="${taskId}" title="取消任务" style="background:none;border:none;cursor:pointer;font-size:0.68rem;color:var(--card-muted);padding:0 2px;">⛔</button>`
                    : isDone
                        ? `<button class="task-delete-btn" data-task-id="${taskId}" title="删除任务" style="background:none;border:none;cursor:pointer;font-size:0.7rem;color:var(--card-muted);padding:0 2px;">✕</button>`
                        : '';
                const style = isDone ? 'padding:6px 8px;font-size:0.73rem;border-bottom:1px solid var(--border-color);cursor:pointer;' : 'padding:6px 8px;font-size:0.73rem;border-bottom:1px solid var(--border-color);';
                return `<li data-task-id="${taskId}" data-task-status="${t.status || ''}" style="${style}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span>${statusIcon} ${escapeHtml(t.label || t.type || '任务')}</span>
                        <span style="display:flex;align-items:center;gap:6px;"><span style="font-size:0.65rem;color:var(--card-muted);">${t.progress || 0}%</span>${actionBtn}</span>
                    </div>
                    ${isRunning ? `<div style="background:var(--border-color);height:3px;border-radius:2px;margin-top:3px;"><div style="background:${barColor};height:100%;width:${barWidth}%;border-radius:2px;transition:width .3s;"></div></div>` : ''}
                    ${t.message && t.status !== 'completed' ? `<div style="font-size:0.65rem;color:var(--card-muted);margin-top:2px;">${escapeHtml(t.message)}</div>` : ''}
                </li>`;
            }).join('');
            // Bind delete buttons (two-step inline confirm, no native dialog)
            list.querySelectorAll('.task-delete-btn').forEach(btn => {
                btn.onclick = (ev) => {
                    ev.stopPropagation();
                    _deleteTask(btn.dataset.taskId, btn);
                };
            });
            // Bind cancel buttons (running tasks): cancel then auto-remove
            list.querySelectorAll('.task-cancel-btn').forEach(btn => {
                btn.onclick = async (ev) => {
                    ev.stopPropagation();
                    btn.disabled = true;
                    btn.textContent = '...';
                    try {
                        const res = await fetch('/tasks/' + btn.dataset.taskId + '/cancel', { method: 'POST', credentials: 'include' });
                        if (res.ok) showToast('任务已取消', 'success', 2500);
                        else {
                            const d = await res.json().catch(() => ({}));
                            showToast(d.error || '取消失败（任务可能已结束）', 'error', 3000);
                        }
                    } catch (_) { showToast('取消失败：网络错误', 'error', 3000); }
                };
            });
        } catch(e) { _bgTasksBackoff++; /* silent */ }
    }

    function _scheduleBgTasksPoll() {
        if (_bgTasksStopped) return;
        const delay = Math.min(BG_TASKS_BASE * Math.pow(1.3, Math.min(_bgTasksBackoff, 12)), 30000);
        _bgTasksPollTimer = setTimeout(async () => {
            await loadBgTasks();
            _scheduleBgTasksPoll();
        }, delay);
    }

    function _handleTaskClick(e) {
        const li = e.target.closest('li[data-task-id]');
        if (!li) return;
        const tid = li.getAttribute('data-task-id');
        const status = li.getAttribute('data-task-status');
        if (!tid || (status !== 'completed' && status !== 'failed')) return;
        e.stopPropagation();
        console.log('[TASK] clicked task', tid, 'status', status);
        let tc = document.getElementById('toast-container');
        if (!tc) {
            tc = document.createElement('div');
            tc.id = 'toast-container';
            tc.setAttribute('class', 'toast-container');
            document.body.appendChild(tc);
        }
        const toast = document.createElement('div');
        toast.setAttribute('class', 'toast info');
        toast.textContent = 'Loading task ' + tid + '...';
        tc.appendChild(toast);
        fetch('/tasks/' + tid, { credentials: 'include' })
            .then(r => r.json())
            .then(d => {
                toast.remove();
                if (d.error || !d.status) { console.warn('[TASK] task not found:', tid); return; }

                // ── Completed tasks: always show a visible result modal first ──
                if (d.status === 'completed' && d.result) {
                    const result = (typeof d.result === 'object') ? d.result : {};
                    _showTaskResultModal(d, result);

                    // Then append into the chat (thread-aware):
                    // same thread → append directly; different/missing thread → warn only.
                    if (result.report && typeof appendClearanceToChat === 'function') {
                        try {
                            const activeThread = sessionStorage.getItem('currentThreadId') || '';
                            const taskThread = d.thread_id || '';
                            if (taskThread && activeThread && taskThread === activeThread) {
                                appendClearanceToChat(result.report, result.download_url || '');
                            } else if (!taskThread || !activeThread) {
                                // No reliable thread context — still show in current chat
                                // (best-effort; the modal always carries the download link).
                                appendClearanceToChat(result.report, result.download_url || '');
                            } else {
                                if (typeof showToast === 'function') {
                                    showToast('该清标结果属于另一会话，已在弹窗中展示摘要与下载', 'info', 4000);
                                }
                            }
                        } catch (err) {
                            console.error('[TASK] appendClearanceToChat failed:', err);
                        }
                    }
                    return;
                }

                // ── Failed tasks: error toast ──
                const t2 = document.createElement('div');
                t2.setAttribute('class', 'toast error');
                t2.textContent = (d.label || d.type || '任务') + ' — ' + d.status + (d.message ? ' | ' + d.message : '');
                t2.style.cursor = 'pointer';
                t2.onclick = () => t2.remove();
                tc.appendChild(t2);
                setTimeout(() => { if (t2.parentNode) t2.remove(); }, 8000);
            }).catch((err) => { toast.textContent = 'Error loading task'; console.error('[TASK] load failed:', err); });
    }

    // Centered modal shown on completed-task click — always visible regardless of panel rendering.
    function _showTaskResultModal(taskData, resultData) {
        const existingBackdrop = document.querySelector('.task-result-backdrop');
        if (existingBackdrop) existingBackdrop.remove();
        const existingModal = document.querySelector('.task-result-modal');
        if (existingModal) existingModal.remove();

        const label = taskData.label || taskData.type || '任务';
        const dlUrl = resultData.download_url || '';
        const fileCount = resultData.file_count || 0;
        const report = resultData.report || {};
        const info = report.basic_info || {};
        const totalScore = info.total_score;
        const warning = info.warning_level || '';
        const indicatorCount = (report.indicators || []).length;
        const crossPairCount = ((report.cross_comparison || {}).pairs || []).length;

        let html = '<div style="font-size:0.9rem;font-weight:700;margin-bottom:10px;color:#16a34a;">✅ ' + escapeHtml(label) + ' 完成</div>';
        html += '<div style="font-size:0.78rem;margin-bottom:14px;line-height:1.8;">';
        if (fileCount) html += '<div>📂 投标文件数: <b>' + fileCount + '</b></div>';
        if (totalScore !== undefined && totalScore !== null) html += '<div>📊 综合评分: <b style="color:' + (totalScore > 50 ? '#dc2626' : totalScore > 20 ? '#d97706' : '#16a34a') + '">' + totalScore + ' 分</b></div>';
        if (warning) html += '<div>⚠️ 预警级别: <b>' + escapeHtml(warning) + '</b></div>';
        if (indicatorCount) html += '<div>📋 指标分析: <b>' + indicatorCount + '</b> 项</div>';
        if (crossPairCount) html += '<div>🔀 横向对比: <b>' + crossPairCount + '</b> 对组合</div>';
        html += '</div>';
        if (dlUrl) {
            html += '<div style="text-align:center;margin-bottom:10px;"><button class="task-result-download" data-url="' + dlUrl + '" style="display:inline-block;background:#8e44ad;color:#fff;padding:10px 24px;border-radius:8px;border:none;cursor:pointer;font-weight:600;font-size:0.85rem;">📥 下载报告 (DOCX+PDF)</button></div>';
        }
        html += '<div style="text-align:center;"><button class="task-result-close" style="border:1px solid #d1d5db;border-radius:6px;padding:5px 18px;cursor:pointer;background:#f9fafb;font-size:0.75rem;">关闭</button></div>';

        const backdrop = document.createElement('div');
        backdrop.className = 'task-result-backdrop';
        backdrop.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.35);z-index:10003;';
        const modal = document.createElement('div');
        modal.className = 'task-result-modal';
        modal.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:10004;background:white;border:1px solid #e5e7eb;border-radius:16px;padding:22px;max-width:420px;width:90%;box-shadow:0 12px 32px rgba(0,0,0,.25);';
        modal.innerHTML = html;
        document.body.appendChild(backdrop);
        document.body.appendChild(modal);

        // Download via fetch + Blob URL — works on self-signed HTTPS where
        // native <a download> navigation is silently blocked by Chrome.
        const dlBtn = modal.querySelector('.task-result-download');
        if (dlBtn) {
            dlBtn.addEventListener('click', async function() {
                const url = this.getAttribute('data-url');
                if (!url) return;
                const btn = this;
                const origText = btn.textContent;
                btn.disabled = true;
                btn.textContent = '⏳ 下载中...';
                try {
                    const res = await fetch(url, { credentials: 'include' });
                    if (!res.ok) {
                        let msg = 'HTTP ' + res.status;
                        try { msg = (await res.json()).error || msg; } catch (_) {}
                        showToast('下载失败: ' + msg, 'error', 5000);
                        return;
                    }
                    const blob = await res.blob();
                    const blobUrl = URL.createObjectURL(blob);
                    const tmp = document.createElement('a');
                    tmp.href = blobUrl;
                    tmp.download = '串通投标线索分析报告.zip';
                    document.body.appendChild(tmp);
                    tmp.click();
                    setTimeout(() => { URL.revokeObjectURL(blobUrl); tmp.remove(); }, 1000);
                    showToast('✅ 报告已开始下载', 'success', 3000);
                } catch (err) {
                    console.error('[TASK] download failed:', err);
                    showToast('下载失败: ' + (err.message || '网络错误'), 'error', 5000);
                } finally {
                    btn.disabled = false;
                    btn.textContent = origText;
                }
            });
        }

        const close = () => { backdrop.remove(); modal.remove(); };
        backdrop.onclick = close;
        modal.querySelector('.task-result-close').onclick = close;
    }

    // Two-step inline delete — no native confirm() (Chrome dialog-blocker
    // silently cancels it after a few dialogs, making deletion impossible).
    function _deleteTask(taskId, btn) {
        if (btn.dataset.armed === '1') {
            btn.disabled = true;
            btn.textContent = '...';
            fetch('/tasks/' + taskId + '/delete', { method: 'POST', credentials: 'include' })
                .then(r => r.json())
                .then(d => {
                    if (d.success) {
                        const li = btn.closest('li');
                        if (li) li.remove();
                        showToast('任务已删除', 'success', 2000);
                    } else {
                        showToast('删除失败', 'error', 3000);
                        _disarmDelete(btn);
                    }
                })
                .catch(() => { showToast('删除失败：网络错误', 'error', 3000); _disarmDelete(btn); });
        } else {
            btn.dataset.armed = '1';
            btn.textContent = '确认删除';
            btn.style.color = '#dc2626';
            btn.style.fontWeight = '600';
            clearTimeout(btn._revertTimer);
            btn._revertTimer = setTimeout(() => _disarmDelete(btn), 3000);
        }
    }

    function _disarmDelete(btn) {
        if (!btn || !btn.isConnected) return;
        delete btn.dataset.armed;
        btn.textContent = '✕';
        btn.style.color = '';
        btn.style.fontWeight = '';
    }

    document.addEventListener('click', _handleTaskClick);
    console.log('[TASK] click handler registered on document');

    function startBgTasksPolling() {
        _bgTasksStopped = false;
        _bgTasksBackoff = 0;
        loadBgTasks();
        _scheduleBgTasksPoll();
    }

    function stopBgTasksPolling() {
        _bgTasksStopped = true;
        if (_bgTasksPollTimer) { clearTimeout(_bgTasksPollTimer); _bgTasksPollTimer = null; }
    }

    // Global: subscribe to SSE progress for a task, updates progressBar + toast
    // ======================== Floating Task Indicator + Progress ========================

    let _activeTaskIds = new Set();
    let _taskResults = {};  // taskId -> {type, label, result}
    let _taskMinimized = false;

    function updateFloatingIndicator() {
        const indicator = document.getElementById('floatingTaskIndicator');
        const badge = document.getElementById('floatingTaskBadge');
        if (!indicator || !badge) return;
        const count = _activeTaskIds.size;
        if (count > 0) {
            indicator.style.display = 'block';
            badge.textContent = count;
            if (count <= 3) {
                // Show label of running task
                indicator.title = Array.from(_activeTaskIds).slice(0, 3).map(id => {
                    const r = _taskResults[id];
                    return r ? r.label : id;
                }).join(', ') + (count > 3 ? ' +' + (count-3) + ' more' : '');
            }
        } else {
            indicator.style.display = 'none';
            _taskMinimized = false;
        }
    }

    // Click floating indicator → toggle progress bar visibility
    const floatIndicator = document.getElementById('floatingTaskIndicator');
    if (floatIndicator) {
        floatIndicator.addEventListener('click', () => {
            const progBar = document.getElementById('progressBar');
            const progToast = document.getElementById('progressToast');
            if (!progBar) return;
            if (progBar.style.display === 'block') {
                // Minimize
                progBar.style.display = 'none';
                if (progToast) progToast.style.display = 'none';
                _taskMinimized = true;
            } else {
                // Show
                progBar.style.display = 'block';
                _taskMinimized = false;
            }
        });
    }

    // Completion popup modal
    function showTaskCompletion(taskId, result) {
        const existing = document.querySelector('.task-completion-toast');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.className = 'task-completion-toast';
        toast.style.cssText = 'position:fixed;bottom:80px;right:60px;z-index:10003;background:white;border:1px solid #e5e7eb;border-radius:12px;padding:12px 16px;max-width:360px;box-shadow:0 8px 24px rgba(0,0,0,.15);animation:fadeInScale .3s ease;font-size:0.82rem;';
        toast.innerHTML = '<div style="font-weight:600;margin-bottom:6px;color:#16a34a;">✅ ' + (result.label || '任务完成') + '</div>'
            + (result.message ? '<div style="color:#374151;margin-bottom:4px;">' + result.message + '</div>' : '')
            + '<div style="display:flex;gap:6px;margin-top:8px;">'
            + '<button class="task-dismiss-btn" style="flex:1;padding:4px 8px;border:1px solid #d1d5db;border-radius:6px;background:#f9fafb;cursor:pointer;font-size:0.7rem;">关闭</button>'
            + (result.resultText ? '<button class="task-view-btn" style="flex:1;padding:4px 8px;border:none;border-radius:6px;background:#3b82f6;color:white;cursor:pointer;font-size:0.7rem;">查看详情</button>' : '')
            + '</div>';
        document.body.appendChild(toast);

        toast.querySelector('.task-dismiss-btn').onclick = () => toast.remove();
        const viewBtn = toast.querySelector('.task-view-btn');
        if (viewBtn && result.resultText) {
            viewBtn.onclick = () => {
                toast.remove();
                showContentModal(result.label || '结果', result.resultText);
            };
        }
        // Auto-dismiss after 15s
        setTimeout(() => { if (toast.parentNode) toast.remove(); }, 15000);
    }

    async function watchTaskProgress(taskId) {
        const progBar = document.getElementById('progressBar');
        const progFill = document.getElementById('progressBarFill');
        const progToast = document.getElementById('progressToast');
        const procInd = document.getElementById('processingIndicator');

        _activeTaskIds.add(taskId);
        _taskResults[taskId] = { label: '处理中...' };
        updateFloatingIndicator();

        if (!_taskMinimized) {
            if (progBar) progBar.style.display = 'block';
            if (procInd) procInd.style.display = 'inline-block';
        }

        let es;
        let idleTimer;
        try {
            es = new EventSource(`/tasks/${encodeURIComponent(taskId)}/stream`);
            idleTimer = setTimeout(() => {
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
            es.onmessage = (e) => {
                clearTimeout(idleTimer);
                idleTimer = setTimeout(() => {
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
                try {
                    const d = JSON.parse(e.data);
                    if (d.progress !== undefined && progFill && !_taskMinimized) {
                        progFill.style.width = d.progress + '%';
                    }
                    if (d.message && progToast && !_taskMinimized) {
                        progToast.style.display = 'block';
                        progToast.textContent = d.message;
                    }
                    _taskResults[taskId] = { label: d.message || d.label || '处理中...', progress: d.progress };
                    if (d.event === 'complete') {
                        clearTimeout(idleTimer);
                        es.close();
                        _activeTaskIds.delete(taskId);
                        updateFloatingIndicator();
                        finishProgress(true);
                        showTaskCompletion(taskId, { label: d.message || '任务完成', message: d.resultText || '', result: d.result });
                        if (_activeTaskIds.size === 0 && procInd) procInd.style.display = 'none';
                        loadBgTasks();
                    }
                    if (d.event === 'error') {
                        clearTimeout(idleTimer);
                        es.close();
                        _activeTaskIds.delete(taskId);
                        updateFloatingIndicator();
                        finishProgress(false, d.message || '任务失败');
                        if (_activeTaskIds.size === 0 && procInd) procInd.style.display = 'none';
                        loadBgTasks();
                    }
                } catch(parseErr) {}
            };
            es.onerror = () => {
                clearTimeout(idleTimer);
                es.close();
                _activeTaskIds.delete(taskId);
                updateFloatingIndicator();
                finishProgress(false, '连接失败');
                if (_activeTaskIds.size === 0 && procInd) procInd.style.display = 'none';
                loadBgTasks();
            };
        } catch(e) {
            _activeTaskIds.delete(taskId);
            updateFloatingIndicator();
            if (_activeTaskIds.size === 0) {
                if (progBar) progBar.style.display = 'none';
                if (procInd) procInd.style.display = 'none';
            }
        }
    }

    // Start background tasks polling on page load
    document.addEventListener('DOMContentLoaded', () => {
        startBgTasksPolling();
        initInputResize();
    });

    // ======================== Input area resize drag ========================
    function initInputResize() {
        const handle = document.getElementById('inputResizeHandle');
        const inputArea = document.getElementById('inputArea');
        const textarea = document.getElementById('messageInput');
        const chatApp = document.querySelector('.chat-app');
        if (!handle || !inputArea || !textarea || !chatApp) return;

        let dragging = false;
        let startY = 0;
        let startHeight = 0;
        const MIN_HEIGHT = 32;   // ~1 line
        const LINE_HEIGHT = 22;  // approximate line height

        handle.addEventListener('mousedown', (e) => {
            dragging = true;
            startY = e.clientY;
            startHeight = inputArea.offsetHeight;
            handle.style.background = '#3b82f6';
            document.body.style.userSelect = 'none';
            document.body.style.cursor = 'ns-resize';
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            const deltaY = startY - e.clientY;
            const newHeight = Math.max(MIN_HEIGHT, Math.min(
                chatApp.clientHeight / 3,
                startHeight + deltaY
            ));
            inputArea.style.height = newHeight + 'px';
            // Resize textarea to fill the input area
            const btnArea = inputArea.querySelector('.file-btn') ? 28 : 0;
            textarea.style.height = Math.max(MIN_HEIGHT - 8, newHeight - 12) + 'px';
            textarea.rows = Math.max(1, Math.floor((newHeight - 12) / LINE_HEIGHT));
        });

        const stopDrag = () => {
            if (!dragging) return;
            dragging = false;
            handle.style.background = 'transparent';
            document.body.style.userSelect = '';
            document.body.style.cursor = '';
        };
        document.addEventListener('mouseup', stopDrag);
        document.addEventListener('mouseleave', stopDrag);

        // Hover highlight
        handle.addEventListener('mouseenter', () => { if (!dragging) handle.style.background = '#e5e7eb'; });
        handle.addEventListener('mouseleave', () => { if (!dragging) handle.style.background = 'transparent'; });
    }

    // ══════════════════════════════════════════════════════════════════
    // 清标 (unified) — merges Smart Compare / Doc Analysis / Compliance / AI Review
    // ══════════════════════════════════════════════════════════════════
    function _clearanceEscape(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, function(c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    function initClearanceTool() {
        var fileInput = document.getElementById('clearanceFileInput');
        var selectBtn = document.getElementById('selectClearanceFilesBtn');
        var tenderInput = document.getElementById('clearanceTenderInput');
        var tenderBtn = document.getElementById('selectClearanceTenderBtn');
        var openInfoInput = document.getElementById('clearanceOpenInfoInput');
        var openInfoBtn = document.getElementById('selectClearanceOpenInfoBtn');
        var runBtn = document.getElementById('runClearanceBtn');
        var fileNames = document.getElementById('clearanceFileNames');
        var progDiv = document.getElementById('clearanceProgress');
        var progText = document.getElementById('clearanceProgressText');
        var progBar = document.getElementById('clearanceProgressBar');
        var optCompliance = document.getElementById('optCompliance');
        var optAI = document.getElementById('optAIReview');
        var hint = document.getElementById('clearanceHint');
        var stationBtn = document.getElementById('clearanceFileStationBtn');

        if (!selectBtn || !runBtn || !fileInput) return;

        var selectedFiles = [];
        var tenderFile = null;
        var hasLLM = (sessionStorage.getItem('hasLLM') === 'true');

        // File station (global) quick open — expose once for input bar reuse
        if (stationBtn && !window.__openFileStation) {
            window.__openFileStation = function() {
                var modal = document.getElementById('fileStationModal');
                if (modal) {
                    if (typeof loadFileStation === 'function') loadFileStation();
                    modal.style.display = 'block';
                }
            };
            stationBtn.onclick = window.__openFileStation;
        }

        // AI dimension hint: check_auth may still be in flight; backend skips
        // AI review when no LLM, and the AI tab hides if there is no result.
        if (!hasLLM) {
            if (hint) hint.textContent = hint.textContent || '未检测到 LLM 连接，AI 评审将自动跳过。';
        }

        // Compliance needs a tender doc
        function updateComplianceHint() {
            if (optCompliance) {
                var disabled = !tenderFile;
                optCompliance.disabled = false;
                optCompliance.checked = tenderFile ? true : false;
                var lbl = optCompliance.closest('label');
                if (lbl) lbl.style.opacity = tenderFile ? '1' : '0.5';
            }
            if (hint) {
                hint.textContent = tenderFile
                    ? ('招标文件: ' + tenderFile.name)
                    : '选择招标文件后可启用合规审查';
            }
        }

        // ── Pre-upload: stream each file to /upload_file, collect file_ids ──
        var uploadedFileIds = [];   // [{id, name, size}]
        var tenderFileId = null;
        var openInfoFile = null;
        var openInfoFileId = null;
        var _lastTenderName = '';
        var _lastOpenInfoName = '';
        var uploading = false;

        function _fmtSize(bytes) {
            if (bytes >= 1024 * 1024 * 1024) return (bytes / 1024 / 1024 / 1024).toFixed(1) + 'GB';
            if (bytes >= 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + 'MB';
            return (bytes / 1024).toFixed(0) + 'KB';
        }

        function _uploadOne(file, onProgress) {
            return new Promise(function(resolve, reject) {
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/stream_upload', true);
                xhr.withCredentials = true;
                xhr.upload.onprogress = function(e) {
                    if (e.lengthComputable && onProgress) onProgress(Math.round(e.loaded / e.total * 100));
                };
                xhr.onload = function() {
                    try {
                        var d = JSON.parse(xhr.responseText);
                        if (xhr.status === 200 && d.success) resolve({ id: d.file_id, name: d.filename, size: d.size });
                        else reject(new Error(d.error || ('HTTP ' + xhr.status)));
                    } catch (err) { reject(new Error('服务器响应解析失败')); }
                };
                xhr.onerror = function() { reject(new Error('网络错误')); };
                var fd = new FormData();
                fd.append('file', file);
                xhr.send(fd);
            });
        }

        async function _preUploadFiles() {
            uploading = true;
            runBtn.disabled = true;
            try {
                var imgExts = ['.pdf', '.docx', '.doc', '.docm', '.pptx', '.pptm'];
                // ── Incremental upload: only upload bid files not already uploaded
                //    (match by name+size to detect changes). Keeps existing file ids.
                var existingByKey = {};
                (uploadedFileIds || []).forEach(function(u) {
                    existingByKey[u.name + '|' + (u.size || '')] = u;
                });
                var newIds = [];
                for (var i = 0; i < selectedFiles.length; i++) {
                    var f = selectedFiles[i];
                    var key = f.name + '|' + f.size;
                    if (existingByKey[key]) {
                        newIds.push(existingByKey[key]);  // reuse
                        continue;
                    }
                    fileNames.textContent = '上传中 (' + (i + 1) + '/' + selectedFiles.length + '): '
                        + f.name + ' [' + _fmtSize(f.size) + '] — 0%';
                    var res = await _uploadOne(f, function(pct) {
                        fileNames.textContent = '上传中 (' + (i + 1) + '/' + selectedFiles.length + '): '
                            + f.name + ' [' + _fmtSize(f.size) + '] — ' + pct + '%';
                    });
                    newIds.push(res);
                }
                uploadedFileIds = newIds;
                // Tender / open-info: re-upload only if a (new) file is selected.
                // Track last uploaded names to avoid redundant re-uploads.
                if (tenderFile && tenderFile.name !== _lastTenderName) {
                    fileNames.textContent = '上传招标文件: ' + tenderFile.name + '...';
                    var tr = await _uploadOne(tenderFile);
                    tenderFileId = tr.id;
                    _lastTenderName = tenderFile.name;
                } else if (!tenderFile) {
                    tenderFileId = null;
                    _lastTenderName = '';
                }
                if (openInfoFile && openInfoFile.name !== _lastOpenInfoName) {
                    fileNames.textContent = '上传开标信息表: ' + openInfoFile.name + '...';
                    var or = await _uploadOne(openInfoFile);
                    openInfoFileId = or.id;
                    _lastOpenInfoName = openInfoFile.name;
                } else if (!openInfoFile) {
                    openInfoFileId = null;
                    _lastOpenInfoName = '';
                }
                var imgFiles = selectedFiles.filter(function(f) {
                    var ext = f.name.toLowerCase().slice(f.name.lastIndexOf('.'));
                    return imgExts.indexOf(ext) !== -1;
                });
                fileNames.textContent = '已上传 ' + uploadedFileIds.length + ' 份投标文件'
                    + (tenderFileId ? ' + 招标文件' : '') + '，可以开始清标';
                hint.textContent = imgFiles.length
                    ? ('📷 ' + imgFiles.length + ' 个文件含图片，将随机抽检 20 张进行识别')
                    : '';
                runBtn.disabled = false;
            } catch (err) {
                fileNames.textContent = '';
                hint.textContent = '';
                alert('上传失败: ' + err.message);
                runBtn.disabled = true;
            } finally {
                uploading = false;
            }
        }

        selectBtn.onclick = function() { fileInput.click(); };
        fileInput.onchange = function() {
            selectedFiles = Array.from(fileInput.files);
            if (selectedFiles.length > 0) {
                fileNames.textContent = selectedFiles.map(function(f) { return f.name; }).join(', ');
                runBtn.disabled = true;
                if (selectedFiles.length >= 2) {
                    _preUploadFiles();   // stream to disk, then enable 开始清标
                }
            } else {
                fileNames.textContent = '';
                runBtn.disabled = true;
            }
        };
        tenderBtn.onclick = function() { tenderInput.click(); };
        tenderInput.onchange = function() {
            tenderFile = tenderInput.files.length ? tenderInput.files[0] : null;
            updateComplianceHint();
            if (tenderFile && uploadedFileIds.length > 0) {
                // re-run pre-upload so the new tender doc gets an id too
                _preUploadFiles();
            }
        };
        if (openInfoBtn && openInfoInput) {
            openInfoBtn.onclick = function() { openInfoInput.click(); };
            openInfoInput.onchange = function() {
                openInfoFile = openInfoInput.files.length ? openInfoInput.files[0] : null;
                if (openInfoFile) {
                    var hintEl = document.getElementById('clearanceHint');
                    if (hintEl) hintEl.textContent = '📑 开标信息表: ' + openInfoFile.name + '（将填充开标表与相关指标）';
                    if (uploadedFileIds.length > 0) _preUploadFiles();
                }
            };
        }

        runBtn.onclick = function() {
            if (selectedFiles.length < 2) { alert('请至少选择 2 份投标文件'); return; }
            _showClearanceInfoModal();
        };

        // ── 剽窃检测模式 (Plagiarism Mode, FIX-016 后续) ──
        var plagBtn = document.getElementById('runPlagiarismBtn');
        var plagResult = document.getElementById('plagiarismResult');
        if (plagBtn) {
            plagBtn.onclick = async function() {
                if (!plagResult) return;
                if (selectedFiles.length < 2) {
                    alert('请先选择至少 2 份投标文件（取前两份进行剽窃对比）');
                    return;
                }
                plagResult.style.display = 'block';
                plagResult.innerHTML = '<span class="msi msi-sm">hourglass_empty</span> 正在对比...';
                try {
                    var fd = new FormData();
                    fd.append('files', selectedFiles[0]);
                    fd.append('files', selectedFiles[1]);
                    if (tenderFile) fd.append('template', tenderFile);
                    var res = await fetch('/batch/plagiarism/compare', {
                        method: 'POST', body: fd, credentials: 'include'
                    });
                    var d = await res.json();
                    if (!d.success) throw new Error(d.error || '对比失败');
                    var r = d.data || {};
                    var vColor = r.verdict === '疑似剽窃' ? '#dc2626' : (r.verdict === '高度相似' ? '#d97706' : '#16a34a');
                    var h = '<div style="font-weight:600;font-size:0.78rem;margin-bottom:6px;">' + _icon('balance') + ' 剽窃对比结果</div>';
                    h += '<div style="margin-bottom:8px;">' + _icon('swap_horiz') + ' <b>' + _clearanceEscape(r.doc_a || '') + '</b> ↔ <b>' + _clearanceEscape(r.doc_b || '') + '</b></div>';
                    h += '<div style="margin-bottom:6px;">文本相似度: <b>' + (r.cosine_similarity || 0).toFixed(2) + '</b>';
                    h += ' | 结论: <b style="color:' + vColor + '">' + _clearanceEscape(r.verdict || '') + '</b></div>';
                    h += '<div style="font-size:0.66rem;color:var(--card-muted);margin-bottom:6px;">高匹配段: ' + (r.high_match_para_count || 0) + '/' + (r.para_count || 0) + ' (占比 ' + ((r.high_match_para_ratio || 0) * 100).toFixed(0) + '%)</div>';
                    var hm = r.high_match_paragraphs || [];
                    if (hm.length) {
                        h += '<div style="font-size:0.68rem;margin-top:4px;"><b>疑似雷同段落 (' + hm.length + '):</b></div>';
                        hm.slice(0, 8).forEach(function(ps) {
                            var ratioPct = Math.round((ps.match_ratio || 0) * 100);
                            var bg = ratioPct >= 80 ? '#fef2f2' : ratioPct >= 50 ? '#fffbeb' : '#fff';
                            h += '<div style="background:' + bg + ';border-left:3px solid ' + vColor + ';border-radius:4px;padding:4px 8px;margin:3px 0;font-size:0.64rem;">';
                            h += '段落 ' + (ps.para_idx + 1) + ' · 匹配率 ' + ratioPct + '% · ' + _clearanceEscape((ps.snippet_a || '').substring(0, 60)) + '</div>';
                        });
                    }
                    plagResult.innerHTML = h;
                } catch (e) {
                    plagResult.innerHTML = '<span style="color:#dc2626;">' + _icon('cancel') + ' 对比失败: ' + _clearanceEscape(e.message || '网络错误') + '</span>';
                }
            };
        }

        function _showClearanceInfoModal() {
            // Remove any existing modal
            var oldBackdrop = document.querySelector('.clearance-info-backdrop');
            if (oldBackdrop) oldBackdrop.remove();
            var oldModal = document.querySelector('.clearance-info-modal');
            if (oldModal) oldModal.remove();

            var fields = [
                ['bid_number', '标段编号'], ['bid_open_time', '开标时间'],
                ['bidder_name', '招标单位'], ['agent_name', '招标代理'],
                ['eval_method', '评标办法'], ['award_announce_time', '中标公告发布时间'],
                ['winner', '中标单位'], ['award_amount', '中标金额'],
                ['region', '地区'], ['regulator', '监督部门'],
                ['platform', '真实交易平台']
            ];

            var fieldsHtml = fields.map(function(f) {
                return '<div style="display:flex;flex-direction:column;font-size:0.68rem;color:#6b7280;">'
                    + '<label style="margin-bottom:1px;">' + f[1] + '</label>'
                    + '<input id="ci_' + f[0] + '" type="text" placeholder="—" '
                    + 'style="padding:4px 6px;border:1px solid #d1d5db;border-radius:4px;font-size:0.75rem;" />'
                    + '</div>';
            }).join('');

            var backdrop = document.createElement('div');
            backdrop.className = 'clearance-info-backdrop';
            backdrop.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.35);z-index:10003;';
            var modal = document.createElement('div');
            modal.className = 'clearance-info-modal';
            modal.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:10004;background:white;border:1px solid #e5e7eb;border-radius:16px;padding:20px;max-width:520px;width:92%;box-shadow:0 12px 32px rgba(0,0,0,.25);';
            modal.innerHTML = ''
                + '<div style="font-size:0.85rem;font-weight:700;margin-bottom:6px;">基本信息填写（可选）</div>'
                + '<div style="font-size:0.68rem;color:#9ca3af;margin-bottom:10px;">不填写的字段报告中显示为 "—"</div>'
                + '<div id="ciCriteriaPreview" style="display:none;margin-bottom:10px;padding:8px 10px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;font-size:0.68rem;color:#166534;"></div>'
                + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 12px;margin-bottom:14px;">'
                + fieldsHtml
                + '</div>'
                + '<div style="display:flex;justify-content:flex-end;gap:8px;">'
                + '<button class="ci-skip" style="border:1px solid #d1d5db;border-radius:6px;padding:5px 16px;cursor:pointer;background:#f9fafb;font-size:0.75rem;">跳过</button>'
                + '<button class="ci-confirm" style="border:none;border-radius:6px;padding:5px 16px;cursor:pointer;background:#8e44ad;color:#fff;font-weight:600;font-size:0.75rem;">确认并开始</button>'
                + '</div>';
            document.body.appendChild(backdrop);
            document.body.appendChild(modal);

            // 评审标准预览（决定 2）：从招标文件自动提取，预填 budget/time，供人工确认
            (async function() {
                if (!tenderFileId) return;
                try {
                    var fd2 = new FormData();
                    fd2.append('tender_file_id', tenderFileId);
                    var r2 = await fetch('/clearance/preview_criteria', {
                        method: 'POST', body: fd2, credentials: 'include'
                    });
                    var d2 = await r2.json();
                    if (!r2.ok || !d2.success || d2.error) return;
                    var prev = document.getElementById('ciCriteriaPreview');
                    if (!prev) return;
                    var c = d2;
                    var lines = [];
                    if (c.budget_price) {
                        lines.push('💰 预算价/控制价: ' + (c.budget_price / 10000).toFixed(2) + ' 万元');
                        var bEl = modal.querySelector('#ci_award_amount');
                        if (bEl && !bEl.value.trim()) bEl.value = (c.budget_price / 10000).toFixed(2) + '万元';
                    }
                    if (c.plan_open_time) {
                        lines.push('🗓 计划开标时间: ' + c.plan_open_time);
                        var tEl = modal.querySelector('#ci_bid_open_time');
                        if (tEl && !tEl.value.trim()) tEl.value = c.plan_open_time;
                    }
                    if (c.eval_method) {
                        lines.push('⚖ 评标办法: ' + c.eval_method);
                        var mEl = modal.querySelector('#ci_eval_method');
                        if (mEl && !mEl.value.trim()) mEl.value = c.eval_method;
                    }
                    if (c.score_points && c.score_points.length) {
                        lines.push('📊 评分点: ' + c.score_points.slice(0, 4).join('、'));
                    }
                    if (lines.length) {
                        prev.innerHTML = '✅ 已从招标文件自动提取（可修改）：<br>' + lines.join('<br>');
                        prev.style.display = 'block';
                    }
                } catch (_) {}
            })();

            var submitWithInfo = async function(collectFields) {
                var infoFields = collectFields();
                backdrop.remove(); modal.remove();
                // append to options store for FormData collection
                window._clearanceInfoOverrides = infoFields;
                // trigger the actual submit
                await _runClearanceSubmit();
            };

            modal.querySelector('.ci-skip').onclick = function() { submitWithInfo(function() { return {}; }); };
            modal.querySelector('.ci-confirm').onclick = function() {
                var info = {};
                fields.forEach(function(f) {
                    var el = modal.querySelector('#ci_' + f[0]);
                    if (el && el.value.trim()) info[f[0]] = el.value.trim();
                });
                submitWithInfo(function() { return info; });
            };
            backdrop.onclick = function() { submitWithInfo(function() { return {}; }); };
        }

        async function _runClearanceSubmit() {
            runBtn.disabled = true;
            progDiv.style.display = 'block';
            progText.textContent = '正在提交清标...';
            progBar.style.width = '5%';

            var options = {
                indicator_analysis: document.getElementById('optIndicators').checked,
                cross_comparison: document.getElementById('optCrossComparison').checked,
                compliance_check: !!(tenderFile && document.getElementById('optCompliance').checked),
                ai_review: !!(hasLLM && document.getElementById('optAIReview').checked)
            };

            try {
                var formData = new FormData();
                if (uploadedFileIds.length >= 2) {
                    // Pre-uploaded path: submit ids only (no binary body)
                    uploadedFileIds.forEach(function(u) { formData.append('file_ids', u.id); });
                    if (tenderFileId) formData.append('tender_file_id', tenderFileId);
                    if (openInfoFileId) formData.append('open_info_file_id', openInfoFileId);
                } else {
                    selectedFiles.forEach(function(f) { formData.append('files', f); });
                    if (tenderFile) formData.append('tender_file', tenderFile);
                }
                formData.append('options', JSON.stringify(options));
                formData.append('project_id', window.currentProjectId || '');
                var infoOverrides = window._clearanceInfoOverrides || {};
                Object.keys(infoOverrides).forEach(function(k) {
                    if (infoOverrides[k]) formData.append(k, infoOverrides[k]);
                });
                window._clearanceInfoOverrides = null;
                var resp = await fetch('/clearance/run', {
                    method: 'POST', body: formData, credentials: 'include'
                });
                if (!resp.ok) {
                    var errData = {};
                    try { errData = await resp.json(); } catch (_) { errData = { error: '服务器错误 (' + resp.status + ')' }; }
                    progDiv.style.display = 'none';
                    alert(errData.error || '启动清标失败');
                    runBtn.disabled = false;
                    return;
                }
                var initData = await resp.json();
                if (!initData.success) {
                    progDiv.style.display = 'none';
                    alert(initData.error || '启动清标失败');
                    runBtn.disabled = false;
                    return;
                }
                var taskId = initData.task_id;
                progText.textContent = '清标已启动，正在建立连接...';

                // Check if already completed (SSE reconnection edge case).
                // Tolerate a transient 404/race before the worker registers the
                // task — just fall through to SSE in that case.
                var statusData = {};
                try {
                    var statusCheck = await fetch('/clearance/status/' + taskId, { credentials: 'include' });
                    if (statusCheck.ok) { try { statusData = await statusCheck.json(); } catch (_) { statusData = {}; } }
                } catch (_) { statusData = {}; }
                if (statusData.success && statusData.completed && statusData.result) {
                    progDiv.style.display = 'none';
                    if (typeof appendClearanceToChat === 'function') {
                        appendClearanceToChat(statusData.result.report || {}, statusData.result.download_url || '');
                    }
                    runBtn.disabled = false;
                    return;
                }

                // Subscribe to SSE
                var sse = new EventSource('/clearance/stream/' + taskId);
                sse.onmessage = function(e) {
                    try {
                        var evt = JSON.parse(e.data);
                        if (evt.event === 'progress') {
                            progBar.style.width = evt.progress + '%';
                            progText.textContent = evt.message || (evt.progress + '%');
                        } else if (evt.event === 'complete') {
                            sse.close();
                            progDiv.style.display = 'none';
                            if (typeof appendClearanceToChat === 'function') {
                                appendClearanceToChat((evt.result || {}).report || {}, (evt.result || {}).download_url || '');
                            }
                            runBtn.disabled = false;
                        } else if (evt.event === 'error') {
                            sse.close();
                            progDiv.style.display = 'none';
                            if (typeof showToast === 'function') showToast('清标失败: ' + (evt.message || '未知错误'), 'error', 5000);
                            runBtn.disabled = false;
                        }
                    } catch (_) {}
                };
                sse.onerror = function() {
                    sse.close();
                    progDiv.style.display = 'none';
                    if (typeof showToast === 'function') showToast('清标连接中断，请重试', 'error', 5000);
                    runBtn.disabled = false;
                };
            } catch (err) {
                progDiv.style.display = 'none';
                if (typeof showToast === 'function') showToast('清标网络错误: ' + (err.message || '未知错误'), 'error', 5000);
                runBtn.disabled = false;
            }
        };

        updateComplianceHint();
    }

    function buildClearanceReportHtml(result, downloadUrl) {
        var report = result.report || {};
        var cross = report.cross_comparison || {};
        var compliance = report.compliance;
        var ai = report.ai_review;
        var html = '';
        // FIX-016 后续: 默认全收起 + 中文编号 + Material Symbols 图标 + 层级类

        // ── 下载链接 ──
        if (downloadUrl) {
            html += '<div style="margin-bottom:10px;">';
            html += '<a href="' + downloadUrl + '" data-clearance-download="1" download style="color:#16a34a;text-decoration:none;font-weight:600;">' + _icon('📥') + ' 下载报告 (DOCX+PDF)</a>';
            html += '</div>';
        }

        // ── 一、指标分析 ──
        html += '<details class="cl-l1"><summary><span class="cl-num">一</span>' + _icon('📊') + ' 指标分析</summary>';
        html += '<div class="cl-l2">' + _renderIndicatorsTab(report) + '</div></details>';

        // ── 二、横向对比 ──
        if (cross && (cross.pairs || []).length) {
            html += '<details class="cl-l1"><summary><span class="cl-num">二</span>' + _icon('🔀') + ' 横向对比</summary>';
            html += '<div class="cl-l2">' + _renderCrossTab(cross, report._files || []) + '</div></details>';
        }

        // ── 三、合规审查 ──
        if (compliance && !compliance.skipped) {
            html += '<details class="cl-l1"><summary><span class="cl-num">三</span>' + _icon('⚖️') + ' 合规审查</summary>';
            html += '<div class="cl-l2">' + _renderComplianceTab(compliance) + '</div></details>';
        } else {
            html += '<details class="cl-l1"><summary><span class="cl-num">三</span>' + _icon('⚖️') + ' 合规审查</summary>';
            html += '<div class="cl-l2"><p style="color:var(--card-muted);font-size:0.72rem;">未提供招标文件，未执行合规审查。</p></div></details>';
        }

        // ── 四、AI 评审 ──
        if (ai && ai.per_file && ai.per_file.length) {
            html += '<details class="cl-l1"><summary><span class="cl-num">四</span>' + _icon('🤖') + ' AI 评审</summary>';
            html += '<div class="cl-l2">' + _renderAITab(ai) + '</div></details>';
        } else {
            html += '<details class="cl-l1"><summary><span class="cl-num">四</span>' + _icon('🤖') + ' AI 评审</summary>';
            html += '<div class="cl-l2"><p style="color:var(--card-muted);font-size:0.72rem;">未检测到 LLM 或审查失败，已跳过。</p></div></details>';
        }

        // ── 五、图片随机抽检说明 ──
        if (report.image_sampling && report.image_sampling.length) {
            html += '<details class="cl-l1"><summary><span class="cl-num">五</span>' + _icon('🖼️') + ' 图片随机抽检说明</summary>';
            html += '<div class="cl-l2">' + _renderImageSamplingTab(report.image_sampling) + '</div></details>';
        } else {
            html += '<details class="cl-l1"><summary><span class="cl-num">五</span>' + _icon('🖼️') + ' 图片随机抽检说明</summary>';
            html += '<div class="cl-l2"><p style="color:var(--card-muted);font-size:0.72rem;">未包含图片或未执行图片抽检。</p></div></details>';
        }

        // ── 六、全量审计补充检查 ──
        if (report.audit_supplement && report.audit_supplement.per_file && report.audit_supplement.per_file.length) {
            html += '<details class="cl-l1"><summary><span class="cl-num">六</span>' + _icon('🛡️') + ' 全量审计补充检查</summary>';
            html += '<div class="cl-l2">' + _renderAuditSupplementTab(report.audit_supplement) + '</div></details>';
        } else {
            html += '<details class="cl-l1"><summary><span class="cl-num">六</span>' + _icon('🛡️') + ' 全量审计补充检查</summary>';
            html += '<div class="cl-l2"><p style="color:var(--card-muted);font-size:0.72rem;">未包含审计补充数据。</p></div></details>';
        }

        return html;
    }

    function _renderImageSamplingTab(sampling) {
        var html = '<div style="font-size:0.72rem;color:var(--card-muted);margin-bottom:6px;">随机抽取部分图片进行视觉校验，检测可能被忽略的图纸/印章/数据差异。</div>';
        (sampling || []).forEach(function(sf) {
            html += '<details class="cl-l3"><summary>' + _icon('🖼️') + ' ' + _clearanceEscape(sf.filename || '') + ' (' + (sf.samples || []).length + '张)</summary>';
            (sf.samples || []).forEach(function(s) {
                html += '<div style="font-size:0.66rem;border:1px solid var(--card-border);border-radius:6px;padding:4px 8px;margin:4px 0;">';
                html += '<b>#' + (s.seq || '') + '</b> 位置: ' + _clearanceEscape(s.chapter || '') + '<br>';
                html += '前文: ' + _clearanceEscape((s.prev || '').substring(0, 10)) + ' | 后文: ' + _clearanceEscape((s.next || '').substring(0, 10)) + '<br>';
                html += '识别: ' + _clearanceEscape((s.desc || '').substring(0, 120));
                html += '</div>';
            });
            html += '</details>';
        });
        return html;
    }

    function _renderAuditSupplementTab(au) {
        var html = '';
        (au.per_file || []).forEach(function(pf) {
            var tl = pf.timeline || {};
            var st = pf.style || {};
            var ru = pf.rules || {};
            html += '<details class="cl-l3"><summary>' + _icon('🛡️') + ' ' + _clearanceEscape(pf.filename || '') + '</summary>';
            html += '<div style="font-size:0.66rem;padding:4px 8px;">';
            if (st.score != null) html += '风格分析: <b>' + (st.score || 0).toFixed(1) + '</b> 分 (' + _clearanceEscape(st.findings && st.findings.formality_label || '') + ')<br>';
            if (ru.count != null) html += '自规则提取: <b>' + (ru.count || 0) + '</b> 条 · 评分 ' + (ru.score || 0).toFixed(1) + '<br>';
            if (tl.skipped) html += '时间线合规: ' + _clearanceEscape(tl.note || '跳过') + '<br>';
            else if (tl.score != null) html += '时间线合规: <b>' + (tl.score || 0).toFixed(1) + '</b> 分<br>';
            html += '</div></details>';
        });
        return html;
    }

    function _attachClearanceHandlers(container, pairs, files) {
        // Scoped matrix-dimension switcher + click delegation (no inline onclick,
        // safe for innerHTML round-trip and CSP).
        if (!container) return;
        var btns = container.querySelectorAll('.cm-matrix-btn');
        btns.forEach(function(b) {
            b.onclick = function() {
                var name = b.getAttribute('data-m');
                container.querySelectorAll('.cm-matrix-btn').forEach(function(x) {
                    x.style.background = (x === b ? '#8e44ad' : 'transparent');
                    x.style.color = (x === b ? '#fff' : '');
                });
                container.querySelectorAll('.cm-matrix').forEach(function(m) {
                    m.style.display = (m.getAttribute('data-name') === name) ? 'block' : 'none';
                });
            };
        });
        // matrix cell click -> pair detail modal
        var allPairs = pairs || [];
        var allFiles = files || [];
        container.querySelectorAll('.cm-matrix td[data-i]').forEach(function(td) {
            td.onclick = function() {
                var i = this.getAttribute('data-i');
                var j = this.getAttribute('data-j');
                if (i == null || j == null) return;
                i = Number(i); j = Number(j);
                var p = allPairs.find(function(x) { return x.i === i && x.j === j; }) ||
                    allPairs.find(function(x) { return x.i === j && x.j === i; });
                if (!p) return;
                _showPairDetail(p, allFiles);
            };
        });
        // download link -> fetch+Blob (works on self-signed HTTPS)
        var dl = container.querySelector('a[data-clearance-download="1"]');
        if (dl) {
            dl.addEventListener('click', function(e) {
                e.preventDefault();
                _downloadClearanceReport(dl.getAttribute('href'));
            });
        }
    }

    function _downloadClearanceReport(url) {
        fetch(url, { credentials: 'include' })
            .then(function(res) { return res.ok ? res.blob() : Promise.reject(new Error('HTTP ' + res.status)); })
            .then(function(blob) {
                var blobUrl = URL.createObjectURL(blob);
                var tmp = document.createElement('a');
                tmp.href = blobUrl;
                tmp.download = '串通投标线索分析报告.zip';
                document.body.appendChild(tmp);
                tmp.click();
                setTimeout(function() { URL.revokeObjectURL(blobUrl); tmp.remove(); }, 1000);
            })
            .catch(function(err) { showToast('下载失败: ' + (err.message || '网络错误'), 'error', 5000); });
    }

    function appendClearanceToChat(report, downloadUrl, threadId) {
        var html = buildClearanceReportHtml({ report: report, download_url: downloadUrl }, downloadUrl);
        var cross = (report || {}).cross_comparison || {};
        var pairs = cross.pairs || [];
        var files = (report || {})._files || [];

        var group = document.createElement('div');
        group.className = 'message-group';
        group.id = 'msg-clearance-' + Date.now();
        var wrapper = document.createElement('div');
        wrapper.className = 'assistant-wrapper';
        var answerDiv = document.createElement('div');
        answerDiv.className = 'assistant-answer comparison-report';
        answerDiv.innerHTML = html;
        _attachClearanceHandlers(answerDiv, pairs, files);
        wrapper.appendChild(answerDiv);
        group.appendChild(wrapper);
        var messagesDiv = document.getElementById('chatMessages');
        if (messagesDiv) {
            messagesDiv.appendChild(group);
            scrollToBottom(true);
        }
    }

    function _renderIndicatorsTab(report) {
        var info = report.basic_info || {};
        var indicators = report.indicators || [];
        var suspected = report.suspected_units || [];
        var personnel = report.personnel_summary || {};
        var html = '';

        html += '<div style="font-size:0.82rem;margin-bottom:10px;padding:8px 10px;background:var(--card-highlight);border-radius:6px;">';
        html += '<strong>投标单位:</strong> ' + (info.bidder_count||0);
        html += ' | <strong>综合评分:</strong> <span style="color:' + ((info.total_score||0) >= 60 ? '#e74c3c' : (info.total_score||0) >= 30 ? '#e67e22' : '#27ae60') + '">' + (info.total_score||0).toFixed(1) + '分</span>';
        html += ' | <strong>预警:</strong> ' + (info.warning_level || '—');
        html += '</div>';

        if (suspected.length > 0) {
            // FIX-016 后续: 红色警报母条目 + 附注
            var warnCount = suspected.filter(function(su){ return (su.score||0) > 10; }).length;
            html += '<details class="alert-parent"><summary>' + _icon('🔴') + ' 预警嫌疑单位 (' + suspected.length + '家)' +
                '<span class="alert-note">含 ' + warnCount + ' 家高嫌疑</span></summary>';
            html += '<table style="width:100%;border-collapse:collapse;font-size:0.7rem;margin-top:4px;">';
            html += '<tr><th>单位</th><th>涉及指标</th><th>风险分</th></tr>';
            suspected.forEach(function(su) {
                var danger = (su.score||0) > 30;
                html += '<tr class="' + (danger ? 'alert-item' : '') + '">';
                html += '<td>' + ((su.score||0) > 10 ? _icon('★') : '') + _clearanceEscape((su.name||'').substring(0,30)) + '</td>';
                html += '<td>' + (su.indicators_triggered||0) + '</td>';
                html += '<td style="color:' + (danger ? '#e74c3c' : '#e67e22') + '">' + (su.score||0).toFixed(1) + '</td></tr>';
            });
            html += '</table></details>';
        } else {
            html += '<p class="severity-ok" style="font-size:0.72rem;margin-bottom:8px;">' + _icon('✅') + ' 未发现预警嫌疑单位</p>';
        }

        html += '<details class="cl-l2"><summary>' + _icon('📊') + ' 指标分析详情 (' + indicators.length + '项)</summary>';
        indicators.forEach(function(ind, idx) {
            var catTag = (ind.category || '') + (ind.skipped ? ' ' + _icon('⏭️') : '');
            var isDanger = ind.score >= 15 && !ind.skipped;
            html += '<div class="' + (isDanger ? 'alert-item' : '') + '" style="border:1px solid var(--card-border);border-radius:6px;margin-bottom:6px;margin-top:6px;padding:8px 10px;">';
            html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">';
            html += '<strong style="font-size:0.75rem;">' + (idx+1) + '. ' + _clearanceEscape(ind.name||'') + '</strong>';
            html += '<span style="font-size:0.65rem;color:var(--card-muted);">' + _clearanceEscape(catTag) + ' | 得分: <b>' + (ind.score||0).toFixed(1) + '</b></span>';
            html += '</div>';
            var resText = ind.result || '';
            // FIX-016 后续 + 阶段 D: 用 severity 字段/score 着色，替代 indexOf(emoji)
            var sev = ind.severity || (ind.score >= 15 ? 'danger' : (ind.score > 0 ? 'warn' : 'ok'));
            var cls = sev === 'danger' ? 'severity-danger' : (sev === 'warn' ? 'severity-warn' : 'severity-ok');
            html += '<div class="' + cls + '" style="font-size:0.68rem;">' + _clearanceEscape(resText) + '</div>';
            if (ind.details && ind.details.length > 0 && !ind.skipped) {
                html += '<div style="font-size:0.65rem;color:var(--card-muted);margin-top:4px;">';
                var keys = Object.keys(ind.details[0] || {});
                ind.details.slice(0, 5).forEach(function(d) {
                    html += keys.map(function(k) { return _clearanceEscape(k) + ': ' + _clearanceEscape(String(d[k]||'').substring(0, 60)); }).join(' | ');
                    html += '<br>';
                });
                if (ind.details.length > 5) html += '...共' + ind.details.length + '项';
                html += '</div>';
            }
            html += '</div>';
        });
        html += '</details>';

        if (personnel.list && personnel.list.length > 0) {
            html += '<details class="cl-l2"><summary>' + _icon('👥') + ' 关系人员汇总 (' + personnel.total + '人)</summary>';
            html += '<table style="width:100%;border-collapse:collapse;font-size:0.7rem;margin-top:4px;">';
            html += '<tr><th>单位</th><th>姓名</th><th>类型</th></tr>';
            personnel.list.slice(0, 20).forEach(function(p) {
                html += '<tr><td>' + _clearanceEscape((p.company||'').substring(0,20)) + '</td><td>' + _clearanceEscape(p.person||'') + '</td><td>' + _clearanceEscape(p.title||'') + '</td></tr>';
            });
            html += '</table></details>';
        }
        return html;
    }

    function _heatColor(v, maxV) {
        // green -> yellow -> red background by value relative to maxV
        var t = maxV > 0 ? Math.min(1, v / maxV) : 0;
        var r = Math.round(240 - 150 * t);
        var g = Math.round(255 - 150 * t);
        var b = Math.round(220 - 180 * t);
        return 'rgba(' + r + ',' + g + ',' + b + ',0.85)';
    }

    function _renderCrossTab(cross, files) {
        var pairs = cross.pairs || [];
        var riskMatrix = cross.risk_matrix || [];
        var textMatrix = cross.text_matrix || riskMatrix;
        var keyMatrix = cross.key_matrix || riskMatrix;
        var attrMatrix = cross.attr_matrix || riskMatrix;
        var keyInfo = cross.key_info_matches || [];
        var gangs = cross.gangs || [];
        var clusterOrder = cross.cluster_order || null;
        var html = '';

        var maxRisk = 0;
        riskMatrix.forEach(function(row) { row.forEach(function(v) { if (v > maxRisk) maxRisk = v; }); });

        html += '<div style="font-size:0.78rem;margin-bottom:8px;"><strong>横向对比:</strong> ' + files.length + ' 个投标单位 · ' + pairs.length + ' 对组合 · 点击矩阵单元格查看对详情</div>';

        // ── E2: 帮派 (红色警报) ──
        if (gangs && gangs.length) {
            html += '<details class="alert-parent"><summary>' + _icon('🕸️') + ' 疑似围标集团 (' + gangs.length + '个)<span class="alert-note">红色高嫌疑</span></summary>';
            gangs.forEach(function(g, gi) {
                html += '<div class="alert-item" style="font-size:0.7rem;">';
                html += '<b>集团' + (gi + 1) + '</b>：' + (g.files || []).map(_clearanceEscape).join(' ' + _icon('⚡') + ' ') +
                    ' | 成员 ' + (g.members || []).length + ' 家 | 最高风险 ' + (g.max_risk || 0).toFixed(1) +
                    ' | 平均 ' + (g.avg_risk || 0).toFixed(1);
                html += '</div>';
            });
            html += '</details>';
        }

        // ── 多维矩阵切换 ──
        html += '<details class="cl-l2"><summary>' + _icon('🔀') + ' 风险矩阵</summary>';
        html += '<div style="display:flex;gap:6px;margin:4px 0;">';
        html += '<button class="cm-matrix-btn" data-m="risk" style="border:1px solid var(--card-border);background:#8e44ad;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.65rem;cursor:pointer;">综合风险</button>';
        html += '<button class="cm-matrix-btn" data-m="text" style="border:1px solid var(--card-border);background:transparent;border-radius:4px;padding:2px 8px;font-size:0.65rem;cursor:pointer;">文本相似</button>';
        html += '<button class="cm-matrix-btn" data-m="key" style="border:1px solid var(--card-border);background:transparent;border-radius:4px;padding:2px 8px;font-size:0.65rem;cursor:pointer;">关键信息</button>';
        html += '<button class="cm-matrix-btn" data-m="attr" style="border:1px solid var(--card-border);background:transparent;border-radius:4px;padding:2px 8px;font-size:0.65rem;cursor:pointer;">文件属性</button>';
        html += '</div>';

        function matrixRows(m, fmt, isAttr) {
            var order = clusterOrder || files.map(function(_, idx) { return idx; });
            var r = '<tr><th style="position:sticky;left:0;background:var(--card-bg);">单位</th>';
            order.forEach(function(idx) { r += '<th>' + _clearanceEscape((files[idx] || '').substring(0, 6)) + '</th>'; });
            r += '</tr>';
            order.forEach(function(ii) {
                r += '<tr><td style="font-weight:bold;">' + _clearanceEscape((files[ii] || '').substring(0, 6)) + '</td>';
                order.forEach(function(jj) {
                    var v = ii === jj ? '--' : (m[ii] && m[ii][jj] != null ? m[ii][jj] : 0);
                    var disp = (v === '--') ? '--' : fmt(v);
                    var bg = (v === '--') ? '' : _heatColor(Number(v), maxRisk);
                    if (v !== '--') {
                        r += '<td data-i="' + ii + '" data-j="' + jj + '" style="background:' + bg + ';text-align:center;cursor:pointer;">' + disp + '</td>';
                    } else {
                        r += '<td style="text-align:center;">--</td>';
                    }
                });
                r += '</tr>';
            });
            return r;
        }

        var riskRows = matrixRows(riskMatrix, function(v) { return Number(v).toFixed(1); });
        var textRows = matrixRows(textMatrix, function(v) { return Number(v).toFixed(1); });
        var keyRows = matrixRows(keyMatrix, function(v) { return Number(v).toFixed(1); });
        var attrRows = matrixRows(attrMatrix, function(v) { return v === 1 ? '是' : '否'; });

        html += '<div class="cm-matrix" data-name="risk" style="overflow-x:auto;">' +
            '<table style="width:100%;border-collapse:collapse;font-size:0.62rem;margin-top:4px;">' + riskRows + '</table></div>';
        html += '<div class="cm-matrix" data-name="text" style="overflow-x:auto;display:none;">' +
            '<table style="width:100%;border-collapse:collapse;font-size:0.62rem;margin-top:4px;">' + textRows + '</table></div>';
        html += '<div class="cm-matrix" data-name="key" style="overflow-x:auto;display:none;">' +
            '<table style="width:100%;border-collapse:collapse;font-size:0.62rem;margin-top:4px;">' + keyRows + '</table></div>';
        html += '<div class="cm-matrix" data-name="attr" style="overflow-x:auto;display:none;">' +
            '<table style="width:100%;border-collapse:collapse;font-size:0.62rem;margin-top:4px;">' + attrRows + '</table></div>';
        html += '<div style="font-size:0.58rem;color:var(--card-muted);margin-top:2px;">色阶：绿(低) → 红(高，最高 ' + maxRisk.toFixed(1) + ')</div>';
        html += '</details>';

        // ── C: 全部组合明细 ──
        html += '<details class="cl-l2"><summary>' + _icon('📋') + ' 全部组合明细 (' + pairs.length + '对)</summary>';
        var maxP = Math.max.apply(null, pairs.map(function(p) { return p.risk || 0; }));
        var avgP = pairs.reduce(function(s, p) { return s + (p.risk || 0); }, 0) / (pairs.length || 1);
        var hiP = pairs.filter(function(p) { return (p.risk || 0) > 5; }).length;
        html += '<div style="font-size:0.65rem;color:var(--card-muted);margin:4px 0;">共 ' + pairs.length + ' 对 · 最高风险 ' + maxP.toFixed(1) + ' · 平均 ' + avgP.toFixed(1) + ' · 高风险(>5) ' + hiP + ' 对</div>';
        html += '<table style="width:100%;border-collapse:collapse;font-size:0.62rem;">';
        html += '<tr><th>单位1</th><th>单位2</th><th>风险</th><th>文本%</th><th>关键%</th><th>属性</th><th>坐标</th></tr>';
        pairs.slice(0, 100).forEach(function(p) {
            var c = (p.risk || 0) > 20 ? '#e74c3c' : (p.risk || 0) > 5 ? '#e67e22' : '';
            html += '<tr style="border-top:1px solid var(--card-border);"><td>' + _clearanceEscape((p.name1 || '').substring(0, 16)) + '</td>' +
                '<td>' + _clearanceEscape((p.name2 || '').substring(0, 16)) + '</td>' +
                '<td style="color:' + c + ';font-weight:600;">' + (p.risk || 0).toFixed(1) + '</td>' +
                '<td>' + (p.sim || 0).toFixed(1) + '</td>' +
                '<td>' + (p.key_sim || 0).toFixed(1) + '</td>' +
                '<td>' + (p.attr_same ? '是' : '否') + '</td>' +
                '<td>(' + (p.i != null ? p.i : '-') + ',' + (p.j != null ? p.j : '-') + ')</td></tr>';
        });
        html += '</table></details>';

        // ── 高风险组合 (红色警报) ──
        var high = pairs.filter(function(p) { return (p.risk || 0) > 5; });
        if (high.length > 0) {
            html += '<details class="alert-parent"><summary>' + _icon('⚠️') + ' 高风险组合 (' + high.length + '对)<span class="alert-note">风险 &gt; 5 需关注</span></summary>';
            html += '<table style="width:100%;border-collapse:collapse;font-size:0.7rem;margin-top:4px;">';
            html += '<tr><th>单位1</th><th>单位2</th><th>风险</th><th>文本相似</th><th>属性雷同</th></tr>';
            high.forEach(function(p) {
                var danger = (p.risk || 0) > 20;
                html += '<tr class="' + (danger ? 'alert-item' : '') + '"><td>' + _clearanceEscape((p.name1 || '').substring(0, 18)) + '</td><td>' + _clearanceEscape((p.name2 || '').substring(0, 18)) + '</td>';
                html += '<td style="color:' + (danger ? '#e74c3c' : '#e67e22') + ';">' + (p.risk || 0).toFixed(1) + '</td>';
                html += '<td>' + (p.sim || 0).toFixed(1) + '%</td><td>' + (p.attr_same ? '是' : '否') + '</td></tr>';
            });
            html += '</table></details>';
        } else {
            html += '<p class="severity-ok" style="font-size:0.72rem;margin-bottom:8px;">' + _icon('✅') + ' 未发现高风险组合</p>';
        }

        // ── 重点信息雷同 ──
        if (keyInfo.length > 0) {
            html += '<details class="cl-l2"><summary>' + _icon('🔑') + ' 重点信息雷同 (' + keyInfo.length + '组)</summary>';
            html += '<table style="width:100%;border-collapse:collapse;font-size:0.7rem;margin-top:4px;">';
            html += '<tr><th>单位1</th><th>单位2</th><th>共同关键词</th></tr>';
            keyInfo.slice(0, 20).forEach(function(ki) {
                html += '<tr><td>' + _clearanceEscape((ki.name1 || '').substring(0, 18)) + '</td><td>' + _clearanceEscape((ki.name2 || '').substring(0, 18)) + '</td>';
                html += '<td>' + _clearanceEscape((ki.common_keywords || []).slice(0, 8).join(', ')) + '</td></tr>';
            });
            html += '</table></details>';
        }

        // Matrix/interaction handlers are attached scoped by _attachClearanceHandlers
        // (called from appendClearanceToChat), not via global selectors, so multiple
        // clearance bubbles in chat don't conflict and handlers survive innerHTML.

        return html;
    }

    function _showPairDetail(p, files) {
        var html = '<div style="font-size:0.85rem;font-weight:700;margin-bottom:8px;">' + _icon('🔍') + ' ' + _clearanceEscape(p.name1 || '') + ' ↔ ' + _clearanceEscape(p.name2 || '') + '</div>';
        html += '<div style="font-size:0.75rem;margin-bottom:10px;line-height:1.7;">';
        html += '综合风险: <b style="color:' + ((p.risk || 0) > 20 ? '#dc2626' : (p.risk || 0) > 5 ? '#d97706' : '#16a34a') + '">' + (p.risk || 0).toFixed(1) + '</b>';
        html += ' · 文本相似: <b>' + (p.sim || 0).toFixed(1) + '%</b>';
        html += ' · 关键信息: <b>' + (p.key_sim || 0).toFixed(1) + '%</b>';
        html += ' · 属性雷同: <b>' + (p.attr_same ? '是' : '否') + '</b>';
        html += ' · 矩阵坐标: (' + (p.i != null ? p.i : '-') + ',' + (p.j != null ? p.j : '-') + ')';
        html += '</div>';

        var blocks = p.blocks || [];
        if (blocks.length) {
            html += '<div style="font-size:0.7rem;color:var(--card-muted);margin-bottom:4px;">相似文本片段 (' + blocks.length + ' 处，可展开查看)</div>';
            blocks.slice(0, 15).forEach(function(b, bi) {
                html += '<details style="font-size:0.68rem;border:1px solid var(--card-border);border-radius:6px;padding:4px 8px;margin-bottom:4px;">' +
                    '<summary style="cursor:pointer;">片段 ' + (bi + 1) + '（' + _clearanceEscape(b.id || '') + ' · ' + (b.size || '') + '字）</summary>';
                html += '<div style="margin-top:4px;color:#374151;">' + (_clearanceEscape(b.text1_snippet || '').substring(0, 200)) + '</div>';
                html += '<div style="color:#7c3aed;margin-top:2px;">— ' + (_clearanceEscape(b.text2_snippet || '').substring(0, 200)) + '</div>';
                html += '</details>';
            });
        } else {
            html += '<p style="font-size:0.7rem;color:var(--card-muted);">该对未见明显文本重合片段。</p>';
        }

        var backdrop = document.createElement('div');
        backdrop.className = 'clearance-pair-backdrop';
        backdrop.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.35);z-index:10003;';
        var modal = document.createElement('div');
        modal.className = 'clearance-pair-modal';
        modal.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:10004;background:white;border:1px solid #e5e7eb;border-radius:16px;padding:18px;max-width:520px;width:92%;max-height:80vh;overflow-y:auto;box-shadow:0 12px 32px rgba(0,0,0,.25);';
        modal.innerHTML = html + '<div style="text-align:center;margin-top:10px;"><button class="cp-close" style="border:1px solid #d1d5db;border-radius:6px;padding:4px 16px;cursor:pointer;background:#f9fafb;font-size:0.72rem;">关闭</button></div>';
        document.body.appendChild(backdrop);
        document.body.appendChild(modal);
        var close = function() { backdrop.remove(); modal.remove(); };
        backdrop.onclick = close;
        modal.querySelector('.cp-close').onclick = close;
    }

    function _renderComplianceTab(comp) {
        var html = '';
        var summary = comp.summary || {};
        html += '<div style="font-size:0.78rem;margin-bottom:8px;">' + _icon('⚖️') + ' 基于招标文件《' + _clearanceEscape(comp.tender_name||'') + '》' + (comp.rules||[]).length + ' 条规则</div>';
        html += '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;">';
        html += '<span style="background:#16a34a;color:#fff;border-radius:6px;padding:3px 10px;font-size:0.7rem;">通过 ' + (summary.pass||0) + '</span>';
        html += '<span style="background:#d97706;color:#fff;border-radius:6px;padding:3px 10px;font-size:0.7rem;">警告 ' + (summary.warning||0) + '</span>';
        html += '<span style="background:#dc2626;color:#fff;border-radius:6px;padding:3px 10px;font-size:0.7rem;">违规 ' + (summary.violation||0) + '</span>';
        html += '<span style="background:#7f1d1d;color:#fff;border-radius:6px;padding:3px 10px;font-size:0.7rem;">严重 ' + (summary.critical||0) + '</span>';
        html += '</div>';

        (comp.per_file || []).forEach(function(pf, pi) {
            var s = pf.summary || {};
            html += '<details ' + (pi === 0 ? 'open' : '') + ' class="cl-l3"><summary>' + _icon('📄') + ' ' + _clearanceEscape(pf.filename||'') + ' — 通过' + (s.pass||0) + ' 警告' + (s.warning||0) + ' 违规' + (s.violation||0) + ' 严重' + (s.critical||0) + '</summary>';
            var results = pf.results || [];
            if (results.length) {
                html += '<table style="width:100%;border-collapse:collapse;font-size:0.65rem;margin-top:4px;">';
                html += '<tr><th>规则</th><th>结论</th><th>证据</th></tr>';
                results.slice(0, 30).forEach(function(r) {
                    var vc = { 'CRITICAL': '#7f1d1d', 'VIOLATION': '#dc2626', 'WARNING': '#d97706', 'PASS': '#16a34a' }[r.verdict] || '';
                    html += '<tr><td>' + _clearanceEscape(r.rule_id||'') + '</td>';
                    html += '<td style="color:' + vc + ';font-weight:600;">' + _clearanceEscape(r.verdict||'') + '</td>';
                    html += '<td>' + _clearanceEscape((r.evidence||'').substring(0,60)) + '</td></tr>';
                });
                html += '</table>';
            }
            html += '</details>';
        });
        return html;
    }

    function _renderAITab(ai) {
        var html = '';
        (ai.per_file || []).forEach(function(pf, pi) {
            var r = pf.review || {};
            var scores = r.scores || {};
            html += '<details ' + (pi === 0 ? 'open' : '') + ' class="cl-l3"><summary>' + _icon('🤖') + ' ' + _clearanceEscape(pf.filename||'') + ' — ' + (r.verdict||'') + ' (' + (r.overall||0) + '/10)</summary>';
            if (Object.keys(scores).length) {
                html += '<div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0;">';
                Object.keys(scores).forEach(function(k) {
                    var v = scores[k];
                    var c = v >= 7 ? '#16a34a' : (v >= 5 ? '#d97706' : '#dc2626');
                    html += '<span style="background:' + c + ';color:#fff;border-radius:6px;padding:3px 8px;font-size:0.7rem;"><b>' + _clearanceEscape(k) + ': ' + v + '</b></span>';
                });
                html += '</div>';
            }
            if (r.issues && r.issues.length) {
                html += '<table style="width:100%;border-collapse:collapse;font-size:0.65rem;">';
                html += '<tr><th>维度</th><th>严重度</th><th>问题</th><th>建议</th></tr>';
                r.issues.slice(0, 15).forEach(function(iss) {
                    var sc = iss.severity === '高' ? '#dc2626' : (iss.severity === '中' ? '#d97706' : '#6b7280');
                    html += '<tr><td>' + _clearanceEscape(iss.axis||'') + '</td>';
                    html += '<td style="color:' + sc + ';">' + _clearanceEscape(iss.severity||'') + '</td>';
                    html += '<td>' + _clearanceEscape((iss.finding||'').substring(0,50)) + '</td>';
                    html += '<td>' + _clearanceEscape((iss.suggestion||'').substring(0,50)) + '</td></tr>';
                });
                html += '</table>';
            }
            if (r.summary) html += '<div style="margin-top:6px;font-size:0.7rem;">' + _icon('📝') + ' ' + _clearanceEscape(r.summary) + '</div>';
            html += '</details>';
        });
        return html;
    }

    function renderDocAnalysisResults(report, panel, downloadUrl) {
        if (!report) { panel.innerHTML = '<p>分析完成但无数据</p>'; panel.style.display = 'block'; return; }

        var info = report.basic_info || {};
        var indicators = report.indicators || [];
        var suspected = report.suspected_units || [];
        var personnel = report.personnel_summary || {};

        var html = '';

        // Basic info
        html += '<div style="font-size:0.82rem;margin-bottom:10px;padding:8px 10px;background:var(--card-highlight);border-radius:6px;">';
        html += '<strong>投标单位:</strong> ' + info.bidder_count;
        html += ' | <strong>综合评分:</strong> <span style="color:' + (info.total_score >= 60 ? '#e74c3c' : info.total_score >= 30 ? '#e67e22' : '#27ae60') + '">' + (info.total_score||0).toFixed(1) + '分</span>';
        html += ' | <strong>预警:</strong> ' + (info.warning_level || '—');
        if (downloadUrl) html += ' | <a href="' + downloadUrl + '" download style="color:#16a34a;text-decoration:none;">' + _icon('📥') + ' 下载DOCX报告</a>';
        html += '</div>';

        // Suspected units
        if (suspected.length > 0) {
            var warnCount2 = suspected.filter(function(su){ return (su.score||0) > 10; }).length;
            html += '<details class="alert-parent"><summary>' + _icon('🔴') + ' 预警嫌疑单位 (' + suspected.length + '家)<span class="alert-note">含 ' + warnCount2 + ' 家高嫌疑</span></summary>';
            html += '<table style="width:100%;border-collapse:collapse;font-size:0.7rem;margin-top:4px;">';
            html += '<tr><th>单位</th><th>涉及指标</th><th>风险分</th></tr>';
            suspected.forEach(function(su) {
                var danger = (su.score||0) > 30;
                html += '<tr class="' + (danger ? 'alert-item' : '') + '">';
                html += '<td>' + ((su.score||0) > 10 ? _icon('★') : '') + escapeHtml((su.name||'').substring(0,30)) + '</td>';
                html += '<td>' + (su.indicators_triggered||0) + '</td>';
                html += '<td style="color:' + (danger ? '#e74c3c' : '#e67e22') + '">' + (su.score||0).toFixed(1) + '</td></tr>';
            });
            html += '</table></details>';
        } else {
            html += '<p class="severity-ok" style="font-size:0.72rem;margin-bottom:8px;">' + _icon('✅') + ' 未发现预警嫌疑单位</p>';
        }

        // Indicator cards
        html += '<details class="cl-l2"><summary>' + _icon('📊') + ' 指标分析详情 (' + indicators.length + '项)</summary>';
        indicators.forEach(function(ind, idx) {
            var catTag = (ind.category || '') + (ind.skipped ? ' ' + _icon('⏭️') : '');
            html += '<div style="padding:8px 10px;border:1px solid var(--card-border);border-radius:6px;margin-bottom:6px;margin-top:6px;">';
            html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">';
            html += '<strong style="font-size:0.75rem;">' + (idx+1) + '. ' + escapeHtml(ind.name||'') + '</strong>';
            html += '<span style="font-size:0.65rem;color:var(--card-muted);">' + escapeHtml(catTag) + ' | 得分: <b>' + (ind.score||0).toFixed(1) + '</b></span>';
            html += '</div>';
            var resText = ind.result || '';
            // FIX-016 后续: severity/score 着色，替代 indexOf(emoji)
            var sev2 = ind.severity || (ind.score >= 15 ? 'danger' : (ind.score > 0 ? 'warn' : 'ok'));
            var cls2 = sev2 === 'danger' ? 'severity-danger' : (sev2 === 'warn' ? 'severity-warn' : 'severity-ok');
            html += '<div class="' + cls2 + '" style="font-size:0.68rem;">' + escapeHtml(resText) + '</div>';
            if (ind.details && ind.details.length > 0 && !ind.skipped) {
                html += '<div style="font-size:0.65rem;color:var(--card-muted);margin-top:4px;">';
                var keys = Object.keys(ind.details[0] || {});
                ind.details.slice(0, 5).forEach(function(d) {
                    html += keys.map(function(k) { return escapeHtml(k) + ': ' + escapeHtml(String(d[k]||'').substring(0, 60)); }).join(' | ');
                    html += '<br>';
                });
                if (ind.details.length > 5) html += '...共' + ind.details.length + '项';
                html += '</div>';
            }
            html += '</div>';
        });
        html += '</details>';

        // Personnel summary
        if (personnel.list && personnel.list.length > 0) {
            html += '<details style="margin-top:8px;"><summary style="cursor:pointer;font-weight:bold;font-size:0.78rem;">👥 关系人员汇总 (' + personnel.total + '人)</summary>';
            html += '<table style="width:100%;border-collapse:collapse;font-size:0.7rem;margin-top:4px;">';
            html += '<tr><th>单位</th><th>姓名</th><th>类型</th></tr>';
            personnel.list.slice(0, 20).forEach(function(p) {
                html += '<tr><td>' + escapeHtml((p.company||'').substring(0,20)) + '</td><td>' + escapeHtml(p.person||'') + '</td><td>' + escapeHtml(p.title||'') + '</td></tr>';
            });
            html += '</table></details>';
        }

        panel.innerHTML = html;
        panel.style.display = 'block';
    }

    // ══════════════════════════════════════════════════════════════════
    // ══════════════════════════════════════════════════════════════════
    function initAdminResultViewers() {
        const quoteBtn = document.getElementById('sidebarQuoteAnomalyResultsBtn');
        const relBtn = document.getElementById('sidebarRelationshipResultsBtn');
        const typoBtn = document.getElementById('sidebarTypoResultsBtn');

        if (quoteBtn) {
            quoteBtn.onclick = () => showResultHistoryModal('报价异常检测记录', '/admin/quote_anomaly_results', renderQuoteAnomalyHistory);
        }
        if (relBtn) {
            relBtn.onclick = () => showResultHistoryModal('关联关系分析记录', '/admin/relationship_results', renderRelationshipHistory);
        }
        if (typoBtn) {
            typoBtn.onclick = () => showResultHistoryModal('错别字检测记录', '/admin/typo_results', renderTypoHistory);
        }
    }

    async function showResultHistoryModal(title, endpoint, renderFn) {
        // Reuse the existing modal pattern
        let modal = document.getElementById('resultHistoryModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'resultHistoryModal';
            modal.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.5);z-index:10000;display:flex;align-items:center;justify-content:center;';
            modal.innerHTML = `
                <div style="background:var(--card-bg);border-radius:12px;padding:20px;max-width:900px;width:95%;max-height:80vh;display:flex;flex-direction:column;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                        <h3 style="margin:0;" id="resultHistoryTitle"></h3>
                        <button id="resultHistoryClose" style="background:none;border:none;font-size:1.2rem;cursor:pointer;">${_icon('✕')}</button>
                    </div>
                    <div id="resultHistoryContent" style="overflow-y:auto;flex:1;font-size:0.75rem;"></div>
                </div>`;
            document.body.appendChild(modal);
            document.getElementById('resultHistoryClose').onclick = () => modal.style.display = 'none';
            modal.onclick = (e) => { if (e.target === modal) modal.style.display = 'none'; };
        }

        document.getElementById('resultHistoryTitle').textContent = title;
        document.getElementById('resultHistoryContent').innerHTML = '<p>加载中...</p>';
        modal.style.display = 'flex';

        try {
            const resp = await fetch(endpoint);
            const data = await resp.json();
            if (resp.ok) {
                renderFn(data, document.getElementById('resultHistoryContent'));
            } else {
                document.getElementById('resultHistoryContent').innerHTML = '<p style="color:#e74c3c;">加载失败: ' + (data.error || '未知错误') + '</p>';
            }
        } catch (err) {
            document.getElementById('resultHistoryContent').innerHTML = '<p style="color:#e74c3c;">网络错误: ' + err.message + '</p>';
        }
    }

    function renderQuoteAnomalyHistory(data, container) {
        const results = data.results || [];
        if (results.length === 0) { container.innerHTML = '<p>暂无报价异常检测记录</p>'; return; }
        let html = `<p>共 ${data.total||results.length} 条记录</p>`;
        html += '<table style="width:100%;border-collapse:collapse;font-size:0.72rem;">';
        html += '<tr><th>ID</th><th>文档</th><th>风险评分</th><th>CV</th><th>同价</th><th>降幅</th><th>聚类</th><th>时间</th><th>用户</th></tr>';
        results.forEach(r => {
            html += `<tr>`;
            html += `<td>${r.id}</td><td>${escapeHtml((r.doc_name||'').substring(0,25))}</td>`;
            html += `<td style="color:${r.risk_score > 50 ? '#e74c3c' : r.risk_score > 20 ? '#e67e22' : '#27ae60'}">${(r.risk_score||0).toFixed(1)}</td>`;
            html += `<td>${(r.cv||0).toFixed(4)}</td>`;
            html += `<td>${r.same_rate_flag ? _icon('⚠️') : _icon('✅')}</td>`;
            html += `<td>${r.abnormal_drop_flag ? _icon('⬇️') : _icon('✅')}</td>`;
            html += `<td>${r.clustering_flag ? _icon('🔗') : _icon('✅')}</td>`;
            html += `<td>${(r.checked_at||'').substring(0,16)}</td>`;
            html += `<td>${escapeHtml(r.username||'')}</td></tr>`;
        });
        html += '</table>';
        container.innerHTML = html;
    }

    function renderRelationshipHistory(data, container) {
        const results = data.results || [];
        if (results.length === 0) { container.innerHTML = '<p>暂无关联关系分析记录</p>'; return; }
        let html = `<p>共 ${data.total||results.length} 条记录</p>`;
        html += '<table style="width:100%;border-collapse:collapse;font-size:0.72rem;">';
        html += '<tr><th>ID</th><th>任务ID</th><th>实体数</th><th>关系数</th><th>风险评分</th><th>模块</th><th>时间</th><th>用户</th></tr>';
        results.forEach(r => {
            html += `<tr>`;
            html += `<td>${r.id}</td><td>${(r.task_id||'').substring(0,12)}</td>`;
            html += `<td>${r.total_entities||0}</td><td>${r.total_relations||0}</td>`;
            html += `<td style="color:${r.risk_score > 50 ? '#e74c3c' : r.risk_score > 20 ? '#e67e22' : '#27ae60'}">${(r.risk_score||0).toFixed(1)}</td>`;
            html += `<td>${escapeHtml((r.modules_run||[]).join(', ').substring(0,25))}</td>`;
            html += `<td>${(r.checked_at||'').substring(0,16)}</td>`;
            html += `<td>${escapeHtml(r.username||'')}</td></tr>`;
        });
        html += '</table>';
        container.innerHTML = html;
    }

    function renderTypoHistory(data, container) {
        const results = data.results || [];
        if (results.length === 0) { container.innerHTML = '<p>暂无错别字检测记录</p>'; return; }
        let html = `<p>共 ${data.total||results.length} 条记录</p>`;
        html += '<table style="width:100%;border-collapse:collapse;font-size:0.72rem;">';
        html += '<tr><th>ID</th><th>文档</th><th>层次</th><th>疑似文本</th><th>建议</th><th>置信度</th><th>严重性</th><th>时间</th><th>用户</th></tr>';
        results.forEach(r => {
            html += `<tr>`;
            html += `<td>${r.id}</td><td>${escapeHtml((r.doc_name||'').substring(0,20))}</td>`;
            html += `<td>${escapeHtml(r.layer||'')}</td>`;
            html += `<td><code>${escapeHtml((r.suspect_text||'').substring(0,25))}</code></td>`;
            html += `<td>${escapeHtml((r.suggestions||'[]').substring(0,30))}</td>`;
            html += `<td>${((r.confidence||0)*100).toFixed(0)}%</td>`;
            html += `<td>${escapeHtml(r.severity||'info')}</td>`;
            html += `<td>${(r.checked_at||'').substring(0,16)}</td>`;
            html += `<td>${escapeHtml(r.username||'')}</td></tr>`;
        });
        html += '</table>';
        container.innerHTML = html;
    }

    // ======================== Unified Bid Audit ========================

    async function loadProjectCollusionGraph() {
        var panel = document.getElementById('projectGraphContainer');
        var statsEl = document.getElementById('projectGraphStats');
        if (!panel) return;
        var pid = window._currentProjectId || currentProjectId;
        if (!pid) {
            panel.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">请先打开一个项目。</span>';
            return;
        }
        panel.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">加载中...</span>';
        try {
            var r = await fetch('/api/graph/collusion?project_id=' + encodeURIComponent(pid), { credentials: 'include' });
            if (!r.ok) {
                var txt = await r.text();
                panel.innerHTML = '<span style="color:#ef4444;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">加载失败: ' + escapeHtml(txt.substring(0, 80)) + '</span>';
                return;
            }
            var data = await r.json();
            if (!data.success || !data.nodes || !data.nodes.length) {
                panel.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">暂无关联数据。请先对项目运行投标文档分析。</span>';
                return;
            }
            var nodeCount = data.nodes.length;
            var edgeCount = data.edges.length;
            if (statsEl) statsEl.textContent = '(' + nodeCount + ' 节点, ' + edgeCount + ' 条关联)';

            renderGraph(panel, data);

        } catch (e) {
            console.error('Collusion graph error:', e);
            panel.innerHTML = '<span style="color:#ef4444;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">网络错误</span>';
        }
    }

    async function loadProjectComplianceGraph() {
        var panel = document.getElementById('projectGraphContainer');
        var statsEl = document.getElementById('projectGraphStats');
        if (!panel) return;
        var pid = window._currentProjectId || currentProjectId;
        if (!pid) {
            panel.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">请先打开一个项目。</span>';
            return;
        }
        panel.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">加载中...</span>';
        try {
            var r = await fetch('/api/graph/compliance?project_id=' + encodeURIComponent(pid), { credentials: 'include' });
            if (!r.ok) {
                var txt = await r.text();
                panel.innerHTML = '<span style="color:#ef4444;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">加载失败: ' + escapeHtml(txt.substring(0, 80)) + '</span>';
                return;
            }
            var data = await r.json();
            if (!data.success || !data.nodes || !data.nodes.length) {
                panel.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">暂无合规违规数据。请先对该项目执行合规检查。</span>';
                return;
            }
            var nodeCount = data.nodes.length;
            var edgeCount = data.edges.length;
            if (statsEl) statsEl.textContent = '(' + nodeCount + ' 节点, ' + edgeCount + ' 条关联)';

            renderGraph(panel, data);
        } catch (e) {
            console.error('Compliance graph error:', e);
            panel.innerHTML = '<span style="color:#ef4444;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">网络错误</span>';
        }
    }

    async function loadProjectCitationGraph() {
        var panel = document.getElementById('projectGraphContainer');
        var statsEl = document.getElementById('projectGraphStats');
        if (!panel) return;
        var pid = window._currentProjectId || currentProjectId;
        if (!pid) {
            panel.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">请先打开一个项目。</span>';
            return;
        }
        panel.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">加载中...</span>';
        try {
            var r = await fetch('/api/graph/citation?project_id=' + encodeURIComponent(pid), { credentials: 'include' });
            if (!r.ok) {
                var txt = await r.text();
                panel.innerHTML = '<span style="color:#ef4444;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">加载失败: ' + escapeHtml(txt.substring(0, 80)) + '</span>';
                return;
            }
            var data = await r.json();
            if (!data.success || !data.nodes || !data.nodes.length) {
                panel.innerHTML = '<span style="color:var(--card-muted);position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">暂无文档引用数据。</span>';
                return;
            }
            var nodeCount = data.nodes.length;
            var edgeCount = data.edges.length;
            if (statsEl) statsEl.textContent = '(' + nodeCount + ' 节点, ' + edgeCount + ' 条关联)';

            renderGraph(panel, data);
        } catch (e) {
            console.error('Citation graph error:', e);
            panel.innerHTML = '<span style="color:#ef4444;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">网络错误</span>';
        }
    }
