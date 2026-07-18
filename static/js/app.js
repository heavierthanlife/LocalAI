/* AI_Services Application Logic */
// ======================== Global Variables & Setup ========================
    window._showAuditModal = null;  // placeholder — set after audit functions defined
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
    const casesTab = document.getElementById('casesTabBtn');

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
            arrow.textContent = '▶';
        } else {
            contentDiv.classList.add('show');
            arrow.textContent = '▼';
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
                <button class="fb-btn" onclick="window.submitSkillFeedback(${fileId},'${source}',5,this)">👍 满意</button>
                <button class="fb-btn" onclick="window.submitSkillFeedback(${fileId},'${source}',1,this)">👎 不满意</button>
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
        fetch('/admin/skill_audit/feedback', {
            method: 'POST', credentials: 'include',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({rating: rating})
        }).then(r => r.json()).then(function(d) {
            if (d.success) {
                container.innerHTML = '<div style="margin-top:8px;font-size:0.7rem;color:#22c55e;">✅ 已反馈</div>';
                showToast('感谢反馈!', 'success');
            }
        }).catch(function(){});
    };

    window.submitQuoteFeedback = function(docName, rating, btn) {
        const container = btn.parentElement.parentElement;
        fetch('/check_quote_anomaly/feedback', {
            method: 'POST', credentials: 'include',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({doc_name: docName, rating: rating})
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
            summary.innerHTML = '<span class="inline-quote-icon">▶</span><span>' + ('引用' + author + ' — ' + lineCount + '行, ' + charCount + '字').replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</span>';
            const quoteBody = document.createElement('div');
            quoteBody.className = 'inline-quote-content';
            quoteBody.style.cssText = 'display:none;margin-top:4px;padding:4px 8px;background:white;border-radius:4px;border:1px solid #e5e7eb;max-height:200px;overflow-y:auto;white-space:pre-wrap;font-size:0.68rem;color:#374151;font-family:monospace;';
            quoteBody.textContent = quoteContent;
            bubble.appendChild(summary);
            bubble.appendChild(quoteBody);
            bubble.addEventListener('click', function(e) {
                const icon = this.querySelector('.inline-quote-icon');
                const body = this.querySelector('.inline-quote-content');
                if (body.style.display === 'none') { body.style.display = 'block'; icon.textContent = '▼'; }
                else { body.style.display = 'none'; icon.textContent = '▶'; }
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
            const current = data.active || 'deepseek';
            const currentModel = sessionStorage.getItem('llmModel') || '';
            let html = '<div style="margin-bottom:12px"><strong>🤖 AI 模型设置</strong></div>';
            html += '<label>服务商:</label>';
            html += '<select id="providerSelect" style="width:100%;margin-bottom:8px">';
            for (const [id, name] of Object.entries(PROVIDER_NAMES)) {
                const sel = id === current ? ' selected' : '';
                html += `<option value="${id}"${sel}>${name}</option>`;
            }
            html += '</select>';
            html += '<label>模型:</label>';
            html += '<select id="modelSelect" style="width:100%;margin-bottom:8px">';
            const models = PROVIDER_MODELS[current] || PROVIDER_MODELS['deepseek'];
            for (const m of models) {
                const sel = m === currentModel ? ' selected' : '';
                html += `<option value="${m}"${sel}>${m}</option>`;
            }
            html += '</select>';
            html += '<button id="applyProviderBtn" class="file-btn" style="width:100%">应用模型设置</button>';
            html += '<small id="providerStatus" style="color:#888">当前: ' + (PROVIDER_NAMES[current]||current) + '</small>';
            // Return HTML; event binding happens after insertion
            setTimeout(() => {
                const provSel = document.getElementById('providerSelect');
                const modelSel = document.getElementById('modelSelect');
                if (provSel && modelSel) {
                    provSel.onchange = () => {
                        const models = PROVIDER_MODELS[provSel.value] || [];
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
                            ? `✅ 已切换至 ${PROVIDER_NAMES[provider]} / ${model}`
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

    const accountModal = document.getElementById('accountModal');
    const closeAccountModal = document.getElementById('closeAccountModal');
    const accountContent = document.getElementById('accountContent');

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

    let _csrfToken = '';
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

    const accountSettingsBtn = document.getElementById('accountSettingsBtn');
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

    // ======================== STREAMING SEND FUNCTION ========================
    async function sendMessageStreaming(userMsg, messageId, files, userGroup = null, retryCount = 0) {
        const formData = new FormData();
        formData.append('message_id', messageId);
        if (userMsg) formData.append('message', userMsg);
        for (let i = 0; i < files.length; i++) formData.append('files', files[i]);
        if (selectedKnowledgeFiles.length) {
            formData.append('knowledge_files', JSON.stringify(selectedKnowledgeFiles));
        }
        if (window._selectedRagCategory) {
            formData.append('rag_category', window._selectedRagCategory);
        }
        if (retryCount > 0) {
            formData.append('fallback_retry', String(retryCount));
        }
        // Quote context is already inlined in userMsg above — don't duplicate to backend

        const startTime = Date.now();
        const controller = new AbortController();
        activeStreamController = controller;

        // Create temp message group
        const tempGroup = document.createElement('div');
        tempGroup.className = 'message-group';
        tempGroup.dataset.msgId = 'temp-' + Date.now();

        // AI name tag for project chats
        if (_isCurrentSessionProjectChat) {
            const aiTag = document.createElement('div');
            aiTag.className = 'ai-name-tag';
            aiTag.textContent = '@中联招标AI';
            tempGroup.appendChild(aiTag);
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'assistant-wrapper';
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'temp-timer';
        loadingDiv.innerHTML = '<span class="typing-dots"><span>.</span><span>.</span><span>.</span></span> <small>0.0s</small>';
        wrapper.appendChild(loadingDiv);

        // Answer container (populated incrementally)
        const answerDiv = document.createElement('div');
        answerDiv.className = 'assistant-answer';
        answerDiv.style.display = 'none';
        wrapper.appendChild(answerDiv);

        // Stop button
        const stopBtn = document.createElement('button');
        stopBtn.className = 'stop-stream-btn';
        stopBtn.textContent = '⏹ 停止';
        stopBtn.onclick = () => { controller.abort(); stopBtn.remove(); };
        wrapper.appendChild(stopBtn);

        tempGroup.appendChild(wrapper);
        messagesDiv.appendChild(tempGroup);
        scrollToBottom();

        let timerInterval = setInterval(() => {
            const elapsed = (Date.now() - startTime) / 1000;
            const dots = loadingDiv.querySelector('.typing-dots');
            if (dots) {
                loadingDiv.querySelector('small').textContent = `${elapsed.toFixed(1)}s`;
            }
        }, 100);

        let fullResponse = '';
        let thinkingDone = false;  // true once thinking→answer split detected
        let thinkingHtml = '';     // cached thinking block HTML
        let noThinkingMarker = false; // true if AI is NOT using 【思考】/【回答】 markers
        const THINKING_END = /【回答】|回答：|<\/思考>/;
        const THINKING_START = /【思考】|思考：|<思考>/;

        try {
            const response = await fetch('/send_stream', {
                method: 'POST',
                credentials: 'include',
                body: formData,
                signal: controller.signal
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || 'Stream request failed');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            loadingDiv.style.display = 'none';
            answerDiv.style.display = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.type === 'content' && data.text) {
                                fullResponse += data.text;

                                // Detect: if after 200 chars there's no 【思考】 marker,
                                // the AI isn't using thinking format — treat everything as direct answer
                                if (!thinkingDone && !noThinkingMarker && fullResponse.length > 200) {
                                    if (!fullResponse.match(THINKING_START) && !fullResponse.match(THINKING_END)) {
                                        noThinkingMarker = true;
                                        thinkingDone = true; // Skip thinking phase
                                    }
                                }

                                // Real-time thinking/answer split detection
                                if (!thinkingDone) {
                                    const m = fullResponse.match(THINKING_END);
                                    if (m) {
                                        thinkingDone = true;
                                        const idx = fullResponse.indexOf(m[0]);
                                        const thinking = fullResponse.substring(0, idx).replace(/^【思考】|^思考：|^<思考>/i, '').trim();
                                        const answerStart = fullResponse.substring(idx + m[0].length);

                                        // Build collapsed thinking block
                                        const preview = thinking.length > 80 ? thinking.substring(0, 80) + '...' : thinking;
                                        thinkingHtml = `<div class="thinking-container">
                                            <div class="thinking-header" onclick="toggleThinking(this);">
                                                <span class="arrow">▶</span><span>思考过程</span>
                                                <span class="thinking-preview">${escapeHtml(preview)}</span>
                                            </div>
                                            <div class="thinking-content">${md.render(thinking)}</div>
                                        </div>`;

                                        // Render thinking block + answer
                                        answerDiv.innerHTML = thinkingHtml + md.render(answerStart);
                                        scrollToBottom(true);
                                        continue;
                                    }
                                }

                                if (thinkingDone) {
                                    if (noThinkingMarker) {
                                        // Direct answer — no thinking block
                                        answerDiv.innerHTML = md.render(fullResponse);
                                    } else {
                                        const splitIdx = fullResponse.search(THINKING_END);
                                        const answerText = splitIdx >= 0
                                            ? fullResponse.substring(splitIdx + fullResponse.match(THINKING_END)[0].length)
                                            : fullResponse;
                                        answerDiv.innerHTML = thinkingHtml + md.render(answerText);
                                    }
                                } else {
                                    // Still in thinking phase — show dimmed streaming text
                                    answerDiv.innerHTML = `<div class="stream-thinking">${escapeHtml(fullResponse)}</div>`;
                                }
                                scrollToBottom(true);
                            } else if (data.type === 'error') {
                                answerDiv.innerHTML += `<p class="stream-error">[Error: ${escapeHtml(data.text)}]</p>`;
                            } else if (data.type === 'fallback_retry') {
                                stopBtn.remove();
                                if (retryCount >= 3) {
                                    answerDiv.innerHTML += '<p class="stream-error">[自动重试已达上限，请手动重试]</p>';
                                    break;
                                }
                                clearInterval(timerInterval);
                                activeStreamController = null;
                                tempGroup.remove();
                                return sendMessageStreaming(userMsg, messageId, files, userGroup, retryCount + 1);
                            } else if (data.type === 'done') {
                                stopBtn.remove();
                                // If backend sent pre-split answer/thinking (e.g. JSON format),
                                // replace raw streaming content with properly formatted version
                                if (data.answer) {
                                    const finalThinking = data.thinking || '';
                                    const finalAnswer = data.answer || '';
                                    if (finalThinking) {
                                        const thinkingHtml2 = `<div class="message-thinking" style="font-size:0.72rem;color:var(--card-muted);background:#f5f7fa;border-radius:6px;padding:6px 10px;margin-bottom:6px;cursor:pointer;" onclick="this.classList.toggle('collapsed');const n=this.nextElementSibling;n.style.display=n.style.display==='none'?'block':'none';">
                                            💭 思考过程 <span style="font-size:0.6rem;">(点击展开)</span></div>
                                        <div class="thinking-content" style="display:none;font-size:0.72rem;color:var(--card-muted);background:#f5f7fa;border-radius:6px;padding:6px 10px;margin-bottom:6px;">${md.render(finalThinking)}</div>`;
                                        answerDiv.innerHTML = thinkingHtml2 + md.render(finalAnswer);
                                    } else {
                                        answerDiv.innerHTML = md.render(finalAnswer);
                                    }
                                    scrollToBottom(true);
                                }
                                if (data.user_msg_id || data.assistant_msg_id) {
                                    _pollLastId = Math.max(_pollLastId || 0, data.user_msg_id || 0, data.assistant_msg_id || 0);
                                    _lastKnownMessageId = Math.max(_lastKnownMessageId || 0, _pollLastId);
                                }
                                addBranchButton(wrapper, tempGroup);
                                if (data.assistant_msg_id) {
                                    addFeedbackButtons(tempGroup, data.assistant_msg_id);
                                    addActionButtons(tempGroup, userMsg, fullResponse, '', data.assistant_msg_id);
                                }
                            }
                        } catch (e) {
                            // Partial JSON — accumulate
                        }
                    }
                }
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                answerDiv.innerHTML += '<p class="stream-cancelled">[已停止生成]</p>';
            } else {
                answerDiv.innerHTML += '<p class="stream-error">[回答生成失败，请重试]</p>';
            }
        } finally {
            clearInterval(timerInterval);
            activeStreamController = null;
            stopBtn.remove();
            const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
            if (!answerDiv.querySelector('.response-time')) {
                const timeTag = document.createElement('small');
                timeTag.className = 'response-time';
                timeTag.textContent = `⏱ ${totalTime}s`;
                answerDiv.appendChild(timeTag);
            }
            // Token count estimate (~4 chars per token for Chinese)
            const estTokens = Math.round(fullResponse.length / 4);
            if (estTokens > 0) {
                const tokTag = document.createElement('small');
                tokTag.className = 'token-count';
                tokTag.textContent = `~${estTokens} tokens`;
                answerDiv.appendChild(tokTag);
            }
            // Fix links
            fixLinksInContainer(answerDiv);
            addCopyButton(wrapper, fullResponse);
            addShareButton(wrapper);
        }
    }

    // ======================== NON-STREAMING SEND FUNCTION ========================
    async function sendMessageNonStreaming(userMsg, messageId, files, userGroup = null) {
        const formData = new FormData();
        formData.append('message_id', messageId);
        if (userMsg) formData.append('message', userMsg);
        for (let i = 0; i < files.length; i++) formData.append('files', files[i]);

        const startTime = Date.now();
        const tempMsgId = 'temp-' + Date.now() + '-' + Math.random();

        const tempGroup = document.createElement('div');
        tempGroup.className = 'message-group';
        tempGroup.dataset.msgId = tempMsgId;
        tempGroup.dataset.userMsg = userMsg;

        const wrapper = document.createElement('div');
        wrapper.className = 'assistant-wrapper';

        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'temp-timer';
        loadingDiv.textContent = '⏳ 0.0s';
        wrapper.appendChild(loadingDiv);
        tempGroup.appendChild(wrapper);
        messagesDiv.appendChild(tempGroup);
        scrollToBottom();

        // Append knowledge files if any (you already have this)
        if (selectedKnowledgeFiles.length) {
            formData.append('knowledge_files', JSON.stringify(selectedKnowledgeFiles));
        }

        let timerInterval = setInterval(() => {
            const elapsed = (Date.now() - startTime) / 1000;
            loadingDiv.textContent = `⏳ ${elapsed.toFixed(1)}s`;
        }, 100);

        try {
            const response = await fetch('/send', {
                method: 'POST',
                credentials: 'include',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || '请求失败');
            }

            const data = await response.json();
            const totalTime = (Date.now() - startTime) / 1000;

            clearInterval(timerInterval);
            tempGroup.remove();

            // Assign ID to the user message group if provided and we have a user_message_id
            if (userGroup && data.user_message_id) {
                userGroup.id = `msg-${data.user_message_id}`;
            }

            const finalGroup = document.createElement('div');
            finalGroup.className = 'message-group';
            // Use the actual assistant message ID from the server
            finalGroup.id = `msg-${data.assistant_message_id}`;
            finalGroup.dataset.msgId = data.assistant_message_id;
            finalGroup.dataset.userMsg = userMsg;

            const finalWrapper = document.createElement('div');
            finalWrapper.className = 'assistant-wrapper';

            if (data.thinking && data.thinking.trim()) {
                const thinkingContainer = document.createElement('div');
                thinkingContainer.className = 'thinking-container';
                const header = document.createElement('div');
                header.className = 'thinking-header';
                header.onclick = function() { toggleThinking(this); };
                const arrow = document.createElement('span');
                arrow.className = 'arrow';
                arrow.textContent = '▶';
                const label = document.createElement('span');
                label.textContent = '思考过程';
                const preview = document.createElement('span');
                preview.className = 'thinking-preview';
                const previewText = data.thinking.length > 80 ? data.thinking.substring(0, 80) + '...' : data.thinking;
                preview.innerText = previewText;
                header.appendChild(arrow);
                header.appendChild(label);
                header.appendChild(preview);
                const contentDiv = document.createElement('div');
                contentDiv.className = 'thinking-content';
                contentDiv.innerHTML = md.render(data.thinking);
                thinkingContainer.appendChild(header);
                thinkingContainer.appendChild(contentDiv);
                finalWrapper.appendChild(thinkingContainer);
            }

            const answerDiv = document.createElement('div');
            answerDiv.className = 'assistant-answer';
            if (data.is_batch_report) {
                let htmlContent = data.assistant_message.replace(/^<!--.*?-->/, '').trim();
                answerDiv.innerHTML = htmlContent;
                answerDiv.classList.add('comparison-report');
            } else {
                let answerText = asciiTableToMarkdown(data.assistant_message);
                answerDiv.innerHTML = md.render(answerText);
            }
            answerDiv.querySelectorAll('pre code').forEach((block) => hljs.highlightElement(block));
            fixLinksInContainer(answerDiv);
            finalWrapper.appendChild(answerDiv);

            const timestampSpan = document.createElement('div');
            timestampSpan.className = 'response-timestamp';
            timestampSpan.textContent = `⏱️ ${formatElapsedTime(totalTime)}`;
            finalWrapper.appendChild(timestampSpan);

            finalGroup.appendChild(finalWrapper);
            messagesDiv.appendChild(finalGroup);

            addCopyButton(finalWrapper, data.assistant_message);
            addBranchButton(finalWrapper, finalGroup);
            addShareButton(finalWrapper);
            const estTok = Math.round((data.assistant_message || '').length / 4);
            if (estTok > 0) {
                const tokTag = document.createElement('small');
                tokTag.className = 'token-count';
                tokTag.textContent = `~${estTok} tokens`;
                finalWrapper.appendChild(tokTag);
            }
            addFeedbackButtons(finalGroup, data.assistant_message_id);
            addActionButtons(finalGroup, userMsg, data.assistant_message, data.thinking || '', data.assistant_message_id);
            scrollToBottom();

            const newHistory = sessionStorage.getItem('chat_history') ? JSON.parse(sessionStorage.getItem('chat_history')) : [];
            newHistory.push({ role: 'user', content: userMsg });
            newHistory.push({ role: 'assistant', content: data.assistant_message, thinking: data.thinking });
            sessionStorage.setItem('chat_history', JSON.stringify(newHistory));
            if (data.file_processed) await checkStorage();
            await loadHistoryList();
            // Cross-tab sync: project chat messages → AI memory
            _syncProjectChatToAiMemory(userMsg);

        } catch (err) {
            clearInterval(timerInterval);
            tempGroup.remove();
            addSystemMessage('发送失败: ' + err.message);
            console.error(err);
        }
    }

    function addActionButtons(group, userMsg, assistantMsg, thinking, msgId) {
        const wrapper = group.querySelector('.assistant-wrapper');
        const actionRow = document.createElement('div');
        actionRow.className = 'action-row';
        const copyBtn = document.createElement('button');
        copyBtn.className = 'action-btn';
        copyBtn.innerHTML = '📋 <span class="btn-text">复制全篇</span>';
        const regenerateBtn = document.createElement('button');
        regenerateBtn.className = 'action-btn';
        regenerateBtn.innerHTML = '🔄 <span class="btn-text">重新生成</span>';
        actionRow.appendChild(copyBtn);
        if (!_isCurrentSessionProjectChat) {
            actionRow.appendChild(regenerateBtn);
        }

        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'feedback-area';
        const ratingDiv = document.createElement('div');
        ratingDiv.className = 'rating-buttons';
        const upBtn = document.createElement('button');
        upBtn.className = 'rating-btn';
        upBtn.innerHTML = '👍 <span class="btn-text">有帮助</span>';
        const downBtn = document.createElement('button');
        downBtn.className = 'rating-btn';
        downBtn.innerHTML = '👎 <span class="btn-text">无帮助</span>';
        ratingDiv.appendChild(upBtn);
        ratingDiv.appendChild(downBtn);
        const commentInput = document.createElement('input');
        commentInput.type = 'text';
        commentInput.className = 'feedback-comment';
        commentInput.placeholder = '补充意见';
        const submitBtn = document.createElement('button');
        submitBtn.className = 'feedback-submit';
        submitBtn.innerHTML = '📨 <span class="btn-text">提交反馈</span>';
        const statusSpan = document.createElement('span');
        statusSpan.className = 'feedback-status';
        feedbackDiv.appendChild(ratingDiv);
        feedbackDiv.appendChild(commentInput);
        feedbackDiv.appendChild(submitBtn);
        feedbackDiv.appendChild(statusSpan);
        actionRow.appendChild(feedbackDiv);
        wrapper.appendChild(actionRow);

        copyBtn.onclick = async () => {
            const answerDiv = wrapper.querySelector('.assistant-answer');
            const htmlString = answerDiv.innerHTML;
            const plainText = answerDiv.innerText;
            try {
                const blobHtml = new Blob([htmlString], { type: 'text/html' });
                const blobText = new Blob([plainText], { type: 'text/plain' });
                await navigator.clipboard.write([
                    new ClipboardItem({ 'text/html': blobHtml, 'text/plain': blobText })
                ]);
                statusSpan.innerText = '已复制到剪贴板';
                setTimeout(() => { statusSpan.innerText = ''; }, 2000);
            } catch (err) {
                try {
                    await navigator.clipboard.writeText(plainText);
                    statusSpan.innerText = '已复制纯文本（表格可能丢失）';
                    setTimeout(() => { statusSpan.innerText = ''; }, 2000);
                } catch (fallbackErr) {
                    statusSpan.innerText = '复制失败，请手动选择';
                    setTimeout(() => { statusSpan.innerText = ''; }, 2000);
                }
            }
        };

        regenerateBtn.onclick = async () => {
            const confirmed = await confirm('重新生成将覆盖当前回答，确定吗？');
            if (!confirmed) return;
            const res = await fetch('/regenerate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ user_message: userMsg })
            });
            const data = await res.json();
            if (res.ok && data.assistant_message) {
                const answerDiv = wrapper.querySelector('.assistant-answer');
                answerDiv.innerHTML = md.render(data.assistant_message);
                fixLinksInContainer(answerDiv);
                group.dataset.assistantMsg = data.assistant_message;
                if (data.thinking) {
                    const thinkingContainer = wrapper.querySelector('.thinking-container');
                    if (thinkingContainer) {
                        const contentDiv = thinkingContainer.querySelector('.thinking-content');
                        contentDiv.innerHTML = md.render(data.thinking);
                        const previewSpan = thinkingContainer.querySelector('.thinking-preview');
                        previewSpan.innerText = data.thinking.substring(0, 80) + '...';
                    }
                }
            } else {
                alert('重新生成失败');
            }
        };

        let currentRating = null;
        upBtn.onclick = () => { currentRating = 'up'; upBtn.classList.add('selected'); downBtn.classList.remove('selected'); };
        downBtn.onclick = () => { currentRating = 'down'; downBtn.classList.add('selected'); upBtn.classList.remove('selected'); };
        submitBtn.onclick = async () => {
            if (!currentRating) { statusSpan.innerText = '请先选择评分'; return; }
            const comment = commentInput.value.trim();
            const payload = { rating: currentRating, comment, user_message: userMsg, assistant_response: assistantMsg };
            const res = await fetch('/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include', body: JSON.stringify(payload) });
            if (res.ok) {
                statusSpan.innerText = '感谢您的反馈！';
                submitBtn.disabled = true;
                upBtn.disabled = true;
                downBtn.disabled = true;
                commentInput.disabled = true;
            } else {
                statusSpan.innerText = '提交失败';
            }
        };
    }

    // ======================== Chat History & Sessions ========================

    // Cross-tab sync: when chatting in a project chat, sync message to AI memory
    async function _syncProjectChatToAiMemory(userMsg) {
        try {
            const sessions = await (await fetch('/get_sessions', {credentials:'include'})).json();
            const currentThread = sessionStorage.getItem('currentThreadId');
            const currentSession = (sessions.sessions||[]).find(s => s.thread_id === currentThread);
            if (currentSession && currentSession.project_id) {
                await fetch(`/admin/projects/${currentSession.project_id}/ai_memory`, {
                    method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
                    body: JSON.stringify({role:'user', content: userMsg})
                }).catch(()=>{});
            }
        } catch(_) {}
    }

    let _loadingHistory = false;
    async function loadHistoryList(force = false) {
        if (_loadingHistory && !force) {
            console.debug('loadHistoryList skipped: already loading');
            return;
        }
        if (_loadingHistory && force) {
            // Wait for the pending load to finish, then re-run
            let waited = 0;
            const maxWait = 5000; // 5s safety timeout
            while (_loadingHistory && waited < maxWait) {
                await new Promise(r => setTimeout(r, 50));
                waited += 50;
            }
            if (_loadingHistory) {
                console.error('loadHistoryList: force timeout — _loadingHistory stuck, resetting');
                _loadingHistory = false;
            }
        }
        _loadingHistory = true;
        try {
            const res = await fetch('/get_sessions', { credentials: 'include' });
            if (!res.ok) {
                console.error('loadHistoryList: /get_sessions returned', res.status);
                return;
            }
            const data = await res.json();
            let sessions = data.sessions || [];
            console.debug('loadHistoryList: got', sessions.length, 'sessions,', sessions.filter(s=>s.project_id).length, 'project');
            const historyList = document.getElementById('historyList');
            const projectHistoryList = document.getElementById('projectHistoryList');
            const projChatHeader = document.getElementById('projChatHeader');
            if (!historyList) return;
            // Split: common (no project_id) vs project chats
            const commonSessions = sessions.filter(s => !s.project_id);
            const projectSessions = sessions.filter(s => s.project_id);
            // Auto-create one common chat ONLY if user has zero chats total AND not viewing a project
            if (commonSessions.length === 0 && projectSessions.length === 0 && !currentProjectId) {
                const newRes = await fetch('/new_chat', { method: 'POST', credentials: 'include' });
                const newData = await newRes.json();
                commonSessions.push({ thread_id: newData.data?.thread_id || newData.thread_id, title: '新对话', updated_at: new Date().toISOString() });
                if (!sessionStorage.getItem('currentThreadId')) {
                    await loadSession(newData.data?.thread_id || newData.thread_id);
                }
            }
            
            // Render common chats
            historyList.innerHTML = '';
            const currentThreadId = sessionStorage.getItem('currentThreadId');
            const sorted = sortPinnedFirst(commonSessions);
            for (const sess of sorted) {
                const li = document.createElement('li');
                li.className = 'history-item';
                if (sess.thread_id === currentThreadId) li.classList.add('active-session');
                const infoDiv = document.createElement('div');
                infoDiv.className = 'history-info';
                const titleSpan = document.createElement('div');
                titleSpan.className = 'history-title';
                titleSpan.textContent = (sess.is_grilling ? '🔥 ' : '') + (sess.title || '新对话');
                titleSpan.title = '双击编辑标题';
                titleSpan.style.cursor = 'text';

                const makeEditable = (spanEl, sessionObj) => {
                    spanEl.ondblclick = (e) => {
                        e.stopPropagation();
                        const input = document.createElement('input');
                        input.type = 'text';
                        input.value = sessionObj.title || '';
                        input.style.cssText = 'width:100%;padding:2px 4px;border:1px solid #64748b;border-radius:4px;font-size:.82rem;font-weight:500;background:inherit;color:inherit;';
                        spanEl.replaceWith(input);
                        input.focus(); input.select();
                        let doneFired = false;
                        const done = async () => {
                            if (doneFired) return;
                            doneFired = true;
                            const newTitle = input.value.trim() || '新对话';
                            const newSpan = document.createElement('div');
                            newSpan.className = 'history-title';
                            newSpan.textContent = newTitle;
                            newSpan.title = '双击编辑标题';
                            newSpan.style.cursor = 'text';
                            makeEditable(newSpan, sessionObj);
                            input.replaceWith(newSpan);
                            if (newTitle !== (sessionObj.title || '新对话')) {
                                try {
                                    const res = await fetch('/update_session_title', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        credentials: 'include',
                                        body: JSON.stringify({ thread_id: sessionObj.thread_id, title: newTitle })
                                    });
                                    if (res.ok) {
                                        sessionObj.title = newTitle;
                                        if (typeof showToast === 'function') showToast('标题已更新', 'success', 2000);
                                    } else {
                                        const err = await res.json();
                                        console.error('Rename failed:', err);
                                        newSpan.textContent = sessionObj.title || '新对话';
                                        if (typeof showToast === 'function') showToast('保存失败: ' + (err.error || '未知错误'), 'error', 3000);
                                    }
                                } catch(e) {
                                    console.error('Rename network error:', e);
                                    newSpan.textContent = sessionObj.title || '新对话';
                                    if (typeof showToast === 'function') showToast('网络错误', 'error', 2000);
                                }
                            }
                        };
                        input.onblur = done;
                        input.onkeydown = (ke) => {
                            if (ke.key === 'Enter') { ke.preventDefault(); done(); }
                            if (ke.key === 'Escape') { input.blur(); }
                        };
                    };
                };
                makeEditable(titleSpan, sess);
                // Unread badge (localStorage per-browser)
                const unreadSpan = document.createElement('span');
                unreadSpan.className = 'unread-badge';
                unreadSpan.style.cssText = 'display:none;background:#ef4444;color:white;border-radius:10px;padding:1px 6px;font-size:0.65rem;margin-left:6px;';
                const commonUnread = _getUnreadCount(sess.thread_id, sess.last_msg_id || 0);
                if (commonUnread > 0) {
                    unreadSpan.textContent = commonUnread > 99 ? '99+' : commonUnread;
                    unreadSpan.style.display = '';
                }
                titleSpan.appendChild(unreadSpan);
                const timeSpan = document.createElement('div');
                timeSpan.className = 'history-time';
                const formatted = sess.updated_at ? new Date(sess.updated_at).toLocaleString() : '刚刚';
                timeSpan.textContent = formatted;
                infoDiv.appendChild(titleSpan);
                infoDiv.appendChild(timeSpan);
                // Pin + Archive buttons (right side)
                const pinIsActive = pinnedSessions && pinnedSessions.has(sess.thread_id);
                const pinBtn = document.createElement('button');
                pinBtn.className = 'pin-button';
                pinBtn.textContent = pinIsActive ? '📌' : '📍';
                pinBtn.title = pinIsActive ? '取消置顶' : '置顶';
                pinBtn.style.background = 'none';
                pinBtn.style.border = 'none';
                pinBtn.style.cursor = 'pointer';
                pinBtn.style.fontSize = '1rem';
                pinBtn.style.opacity = pinIsActive ? '1' : '0.4';
                pinBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (!pinnedSessions) return;
                    if (pinnedSessions.has(sess.thread_id)) {
                        pinnedSessions.delete(sess.thread_id);
                        pinBtn.textContent = '📍';
                        pinBtn.style.opacity = '0.4';
                    } else {
                        pinnedSessions.add(sess.thread_id);
                        pinBtn.textContent = '📌';
                        pinBtn.style.opacity = '1';
                    }
                    persistPins();
                    loadHistoryList();
                };
                // Archive button
                const archiveBtn = document.createElement('button');
                archiveBtn.className = 'archive-history';
                archiveBtn.textContent = '📦';
                archiveBtn.title = '归档聊天';
                archiveBtn.style.marginLeft = '8px';
                archiveBtn.style.background = 'none';
                archiveBtn.style.border = 'none';
                archiveBtn.style.cursor = 'pointer';
                archiveBtn.style.fontSize = '1rem';
                archiveBtn.onclick = async (e) => {
                    e.stopPropagation();
                    const confirmed = await confirm(`将聊天“${sess.title}”归档？归档后将从列表中移除。`);
                    if (!confirmed) return;
                    try {
                        const archiveRes = await fetch(`/archive_session/${sess.thread_id}`, { method: 'POST', credentials: 'include' });
                        if (archiveRes.ok) {
                            li.remove();
                            // If the archived session was the currently active one, load the most recent session
                            if (sess.thread_id === currentThreadId) {
                            // Clear knowledge base selection when archiving current session
                            selectedKnowledgeFiles = [];
                            localStorage.removeItem('selectedKnowledgeFiles');
                            const btn = document.getElementById('knowledgeBaseBtn');
                            if (btn) btn.innerHTML = '📚 知识库';
                            showCatFilterIfNeeded();
                            await loadMostRecentSession();
                            }
                            showToast('已归档', 'success', 2000);
                        } else {
                            const err = await archiveRes.json();
                            alert('归档失败: ' + (err.error || '未知错误'));
                        }
                    } catch (err) {
                        console.error(err);
                        alert('网络错误');
                    }
                };
                li.appendChild(pinBtn);
                li.appendChild(archiveBtn);
                // Delete button
                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'delete-history';
                deleteBtn.textContent = '🗑️';
                deleteBtn.onclick = async (e) => {
                    e.stopPropagation();
                    const confirmed = await confirm(`确定要删除聊天“${sess.title}”吗？此操作不可恢复。`);
                    if (!confirmed) return;
                    li.classList.add('history-item-deleting');
                    deleteBtn.disabled = true;
                    try {
                        const delRes = await fetch(`/delete_session/${sess.thread_id}`, { method: 'POST', credentials: 'include' });
                        if (delRes.status === 409) {
                            const errData = await delRes.json();
                            alert(errData.message);
                            li.classList.remove('history-item-deleting');
                            deleteBtn.disabled = false;
                            return;
                        }
                        if (!delRes.ok) throw new Error('删除失败');
                        const result = await delRes.json();
                        li.remove();
                        if (sess.thread_id === currentThreadId) {
                            if (result.new_thread_id) await loadSession(result.new_thread_id);
                            else {
                                await loadHistoryList();
                                const firstSession = document.querySelector('.history-item');
                                if (firstSession) {
                                    const firstThreadId = firstSession.querySelector('.history-info')?.parentElement?.dataset?.threadId;
                                    if (firstThreadId) await loadSession(firstThreadId);
                                }
                            }
                        }
                        await checkStorage();
                    } catch (err) {
                        console.error(err);
                        alert('删除失败，请稍后重试');
                        li.classList.remove('history-item-deleting');
                        deleteBtn.disabled = false;
                    }
                };
                li.appendChild(infoDiv);
                li.appendChild(deleteBtn);
                li.onclick = async () => { switchToPanel('chatInterface'); await loadSession(sess.thread_id); };
                historyList.appendChild(li);
            }
            if (commonSessions.length === 0) {
                const emptyLi = document.createElement('li');
                emptyLi.textContent = '暂无普通对话';
                emptyLi.style.color = '#999';
                historyList.appendChild(emptyLi);
            }
            // Render project chats
            if (projectHistoryList && projChatHeader) {
                projectHistoryList.innerHTML = '';
                if (projectSessions.length > 0) {
                    projChatHeader.style.display = '';
                    projectHistoryList.style.display = '';
                    const projSorted = sortPinnedFirst(projectSessions);
                    for (const sess of projSorted) {
                        const li = document.createElement('li');
                        li.className = 'history-item';
                        if (sess.thread_id === currentThreadId) li.classList.add('active-session');
                        const infoDiv = document.createElement('div');
                        infoDiv.className = 'history-info';
                        const titleSpan = document.createElement('div');
                        titleSpan.className = 'history-title';
                        titleSpan.textContent = (sess.is_grilling ? '🔥 ' : '📂 ') + (sess.title || '项目对话');
                        // Unread badge (localStorage per-browser)
                        const unreadSpan = document.createElement('span');
                        unreadSpan.className = 'unread-badge';
                        unreadSpan.style.cssText = 'display:none;background:#ef4444;color:white;border-radius:10px;padding:1px 6px;font-size:0.65rem;margin-left:6px;';
                        titleSpan.appendChild(unreadSpan);
                        infoDiv.appendChild(titleSpan);
                        const timeSpan = document.createElement('div');
                        timeSpan.className = 'history-time';
                        timeSpan.textContent = sess.updated_at ? new Date(sess.updated_at).toLocaleString() : '';
                        infoDiv.appendChild(timeSpan);
                        li.appendChild(infoDiv);
                        li.style.cursor = 'pointer';
                        li.onclick = async () => { switchToPanel('chatInterface'); await loadSession(sess.thread_id); };
                        projectHistoryList.appendChild(li);
                        // Unread: compare last_msg_id vs per-browser read position
                        const projectUnread = _getUnreadCount(sess.thread_id, sess.last_msg_id || 0);
                        if (projectUnread > 0) {
                            unreadSpan.textContent = projectUnread > 99 ? '99+' : projectUnread;
                            unreadSpan.style.display = '';
                        }
                    }
                } else {
                    projChatHeader.style.display = 'none';
                }
            }
        } catch (err) { console.error('加载历史列表失败:', err); }
        finally { _loadingHistory = false; }
    }

    async function loadSession(threadId, force = false, targetMessageId = null) {
        if (!force && isLoadingSession) {
            showToast('正在加载，请稍候…', 'info', 1500);
            return;
        }
        const currentActive = sessionStorage.getItem('currentThreadId');
        if (!force && currentActive === threadId && messagesDiv.children.length > 0) {
            const chatPanel = document.getElementById('chatInterface');
            if (chatPanel && chatPanel.style.display !== 'none') return;
        }
        if (isProcessing && !force) {
            addSystemMessage('请等待当前请求完成后再切换会话。');
            return;
        }

        isLoadingSession = true;
        const timeoutId = setTimeout(() => { isLoadingSession = false; }, 10000);
        try {
            messagesDiv.innerHTML = '';
            const res = await fetch(`/load_session/${threadId}`, { credentials: 'include' });
            const data = await res.json();
            if (res.ok && data.messages) {
                // Determine if this is a project chat (for quote mode + regen button)
                let isProjectChat = false;
                let projId = null;
                let isGrillSession = false;
                for (const s of data.sessions || []) {
                    if (s.thread_id === threadId && s.project_id) {
                        isProjectChat = true;
                        projId = s.project_id;
                    }
                    if (s.thread_id === threadId && s.is_grilling) {
                        isGrillSession = true;
                    }
                }
                _isCurrentSessionProjectChat = isProjectChat;
                _isCurrentSessionGrill = isGrillSession;
                if (projId) currentProjectId = projId;
                if (!isProjectChat) {
                    const todoHeader = document.getElementById('todoHeader');
                    const todoList = document.getElementById('todoList');
                    if (todoHeader) todoHeader.style.display = 'none';
                    if (todoList) todoList.style.display = 'none';
                }
                // Show grill mode banner
                const grillBanner = document.getElementById('grillModeBanner');
                if (grillBanner) {
                    grillBanner.style.display = isGrillSession ? '' : 'none';
                }
                for (let i = 0; i < data.messages.length; i++) {
                    const msg = data.messages[i];
                    if (msg.role === 'user') {
                        // Use real id if available, otherwise generate a temp one
                        const msgId = msg.id ? msg.id : `temp-${Date.now()}-${i}`;
                        addUserMessage(msg.content, msgId);
                    } else if (msg.role === 'assistant') {
                        let prevUserMsg = '';
                        if (isProjectChat) {
                            for (let j = i-1; j >= 0; j--) {
                                if (data.messages[j].role === 'user') {
                                    prevUserMsg = data.messages[j].content;
                                    break;
                                }
                            }
                        }
                        const msgId = msg.id ? msg.id : `temp-${Date.now()}-${i}`;
                        renderAssistantMessageLegacy(msgId, prevUserMsg, msg.content, msg.thinking || null);
                    }
                }
                fixLinksInContainer(messagesDiv);
                sessionStorage.setItem('currentThreadId', threadId);
                // Track last message ID for polling
                _lastKnownMessageId = 0;
                for (const m of data.messages || []) {
                    if (m.id && m.id > _lastKnownMessageId) _lastKnownMessageId = m.id;
                }
                if (_lastKnownMessageId > 0) { localStorage.setItem('zlai_read_' + threadId, String(_lastKnownMessageId)); }
                // Mark project chat as read + start polling (common and project)
                let foundProject = false;
                for (const s of data.sessions || []) {
                    if (s.thread_id === threadId && s.project_id) {
                        foundProject = true;
                        fetch(`/admin/projects/${s.project_id}/mark_read`, { method: 'POST', credentials: 'include' }).catch(() => {});
                        startRealtimePoll(s.project_id);  // 3s project poll
                        loadProjectTodos(s.project_id);
                        loadProjectVotes(s.project_id);
                        break;
                    }
                }
                if (!foundProject) {
                    startRealtimePoll(null);  // 5s common chat poll
                }
                await loadHistoryList();
                await checkStorage();
                // After all messages are added to DOM
                if (targetMessageId) {
                    // Wait for DOM to settle, then scroll to the exact message
                    setTimeout(() => {
                        const targetElement = document.getElementById(`msg-${targetMessageId}`);
                        if (targetElement) {
                            // Highlight effect
                            targetElement.style.transition = 'background-color 0.5s';
                            targetElement.style.backgroundColor = '#fff3cd';
                            setTimeout(() => {
                                targetElement.style.backgroundColor = '';
                            }, 2000);
                            // Scroll from current position (not from bottom)
                            targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        } else {
                            console.warn(`Message element msg-${targetMessageId} not found`);
                            scrollToBottom(); // fallback
                        }
                    }, 200);
                } else {
                    // Auto-to-bottom only when chat tab is active
                    const chatPanel = document.getElementById('chatInterface');
                    const isChatVisible = chatPanel && chatPanel.style.display !== 'none';
                    if (isChatVisible) {
                        setTimeout(() => { _userHasScrolled = false; scrollToBottom(true); }, 150);
                    }
                }
            } else {
                console.error('Failed to load session: no messages', data);
            }
        } catch (err) {
            console.error('Failed to load session:', err);
        } finally {
            clearTimeout(timeoutId);
            isLoadingSession = false;
        }
    }

    // ── Unified real-time polling (common + project chats) ──
    let _pollTimer = null;
    let _pollLastId = 0;

    function startRealtimePoll(projectId) {
        stopRealtimePoll();
        _pollLastId = _lastKnownMessageId || 0;
        const interval = projectId ? 3000 : 5000;  // 3s project, 5s common
        _pollTimer = setInterval(async () => {
            const currentThread = sessionStorage.getItem('currentThreadId');
            if (!currentThread || isProcessing) return;
            if (projectId && currentProjectId != projectId) return;
            try {
                const res = await fetch(`/chat/poll/${currentThread}?since_id=${_pollLastId}`, { credentials: 'include' });
                if (!res.ok) return;
                const data = await res.json();
                if (!data.success) return;
                const newMsgs = data.messages || [];
                if (projectId) loadProjectVotes(projectId);
                if (newMsgs.length > 0) {
                    _pollLastId = data.max_id;
                    const wasNearBottom = _isUserNearBottom();
                    for (const msg of newMsgs) {
                        if (msg.role === 'user') {
                            addUserMessage(msg.content, msg.id);
                        } else if (msg.role === 'assistant') {
                            const thinking = msg.thinking || null;
                            renderAssistantMessageLegacy(msg.id, '', msg.content, thinking);
                        }
                    }
                    if (wasNearBottom) scrollToBottom();
                    await loadHistoryList();
                }
            } catch(e) { /* silent */ }
        }, interval);
    }

    function stopRealtimePoll() {
        if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
    }

    let _lastKnownMessageId = 0;
    function _isUserNearBottom() {
        return messagesDiv.scrollHeight - messagesDiv.scrollTop - messagesDiv.clientHeight < 150;
    }

    // ── Per-browser read position (localStorage) ──
    function _getUnreadCount(threadId, lastMsgId) {
        if (!lastMsgId) return 0;
        const key = 'zlai_read_' + threadId;
        const readId = parseInt(localStorage.getItem(key) || '0', 10);
        return Math.max(0, lastMsgId - readId);
    }

    function _markThreadRead(threadId) {
        const key = 'zlai_read_' + threadId;
        localStorage.setItem(key, String(_lastKnownMessageId));
        loadHistoryList();  // refresh sidebar unread badges
    }

    // Mark read on scroll-to-bottom (debounced)
    let _readMarkTimer = null;
    messagesDiv.addEventListener('scroll', () => {
        if (_isUserNearBottom() && _lastKnownMessageId > 0) {
            const currentThread = sessionStorage.getItem('currentThreadId');
            if (currentThread) {
                const key = 'zlai_read_' + currentThread;
                const prevRead = parseInt(localStorage.getItem(key) || '0', 10);
                if (_lastKnownMessageId > prevRead) {
                    localStorage.setItem(key, String(_lastKnownMessageId));
                    if (_readMarkTimer) clearTimeout(_readMarkTimer);
                    _readMarkTimer = setTimeout(() => loadHistoryList(), 800);
                }
            }
        }
    }, { passive: true });

    async function loadMostRecentSession() {
        try {
            const res = await fetch('/get_sessions', { credentials: 'include' });
            const data = await res.json();
            const sessions = data.sessions || [];
            if (sessions.length > 0) await loadSession(sessions[0].thread_id);
            else {
                const newChatRes = await fetch('/new_chat', { method: 'POST', credentials: 'include' });
                if (newChatRes.ok) {
                    const newData = await newChatRes.json();
                    await loadSession(newData.data?.thread_id || newData.thread_id);
                }
            }
        } catch (err) { console.error('Failed to load most recent session', err); }
    }

    function renderAssistantMessageLegacy(msgId, userMsg, assistantMsg, thinking = null) {
        const group = document.createElement('div');
        group.className = 'message-group';
        group.id = `msg-${msgId}`;
        group.dataset.msgId = msgId;
        group.dataset.userMsg = userMsg;
        group.dataset.assistantMsg = assistantMsg;

        // AI name tag (left-aligned, different style from user)
        if (_isCurrentSessionProjectChat) {
            const aiTag = document.createElement('div');
            aiTag.className = 'ai-name-tag';
            aiTag.textContent = '@中联招标AI';
            group.appendChild(aiTag);
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'assistant-wrapper';

        if (thinking && thinking.trim()) {
            const thinkingContainer = document.createElement('div');
            thinkingContainer.className = 'thinking-container';
            const header = document.createElement('div');
            header.className = 'thinking-header';
            header.onclick = function() { toggleThinking(this); };
            const arrow = document.createElement('span');
            arrow.className = 'arrow';
            arrow.textContent = '▶';
            const label = document.createElement('span');
            label.textContent = '思考过程';
            const preview = document.createElement('span');
            preview.className = 'thinking-preview';
            const thinkingPreview = thinking.length > 80 ? thinking.substring(0, 80) + '...' : thinking;
            preview.innerText = thinkingPreview;
            header.appendChild(arrow);
            header.appendChild(label);
            header.appendChild(preview);
            const contentDiv = document.createElement('div');
            contentDiv.className = 'thinking-content';
            contentDiv.innerHTML = md.render(thinking);
            thinkingContainer.appendChild(header);
            thinkingContainer.appendChild(contentDiv);
            wrapper.appendChild(thinkingContainer);
        }

        // Quote: user's question in small italic above AI response
        if (userMsg && userMsg.trim()) {
            const quoteDiv = document.createElement('div');
            quoteDiv.className = 'assistant-quote';
            const preview = userMsg.length > 80 ? userMsg.substring(0, 80) + '...' : userMsg;
            quoteDiv.textContent = '↩ ' + preview;
            quoteDiv.style.cssText = 'font-size:0.7rem;color:#6b7280;font-style:italic;border-left:3px solid #e5e7eb;padding:2px 0 2px 8px;margin-bottom:6px;opacity:0.75;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;';
            wrapper.appendChild(quoteDiv);
        }

        const answerDiv = document.createElement('div');
        answerDiv.className = 'assistant-answer';
        if (assistantMsg && assistantMsg.includes('COMPARE_REPORT')) {
            let htmlContent = assistantMsg.replace(/^<!--.*?-->/, '').trim();
            answerDiv.innerHTML = htmlContent;
            answerDiv.classList.add('comparison-report');
        } else {
            answerDiv.innerHTML = md.render(asciiTableToMarkdown(assistantMsg));
        }
        fixLinksInContainer(answerDiv);
        wrapper.appendChild(answerDiv);
        group.appendChild(wrapper);

        const actionRow = document.createElement('div');
        actionRow.className = 'action-row';
        const copyBtn = document.createElement('button');
        copyBtn.className = 'action-btn';
        copyBtn.innerHTML = '📋 <span class="btn-text">复制全篇</span>';
        const regenerateBtn = document.createElement('button');
        regenerateBtn.className = 'action-btn';
        regenerateBtn.innerHTML = '🔄 <span class="btn-text">重新生成</span>';
        actionRow.appendChild(copyBtn);
        if (!_isCurrentSessionProjectChat) {
            actionRow.appendChild(regenerateBtn);
        }
        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'feedback-area';
        const ratingDiv = document.createElement('div');
        ratingDiv.className = 'rating-buttons';
        const upBtn = document.createElement('button');
        upBtn.className = 'rating-btn';
        upBtn.innerHTML = '👍 <span class="btn-text">有帮助</span>';
        const downBtn = document.createElement('button');
        downBtn.className = 'rating-btn';
        downBtn.innerHTML = '👎 <span class="btn-text">无帮助</span>';
        ratingDiv.appendChild(upBtn);
        ratingDiv.appendChild(downBtn);
        const commentInput = document.createElement('input');
        commentInput.type = 'text';
        commentInput.className = 'feedback-comment';
        commentInput.placeholder = '补充意见';
        const submitBtn = document.createElement('button');
        submitBtn.className = 'feedback-submit';
        submitBtn.innerHTML = '📨 <span class="btn-text">提交反馈</span>';
        const statusSpan = document.createElement('span');
        statusSpan.className = 'feedback-status';
        feedbackDiv.appendChild(ratingDiv);
        feedbackDiv.appendChild(commentInput);
        feedbackDiv.appendChild(submitBtn);
        feedbackDiv.appendChild(statusSpan);
        actionRow.appendChild(feedbackDiv);
        wrapper.appendChild(actionRow);

        copyBtn.onclick = async () => {
            const htmlString = answerDiv.innerHTML;
            const plainText = answerDiv.innerText;
            if (!navigator.clipboard) {
                fallbackCopy(plainText);
                statusSpan.innerText = '已复制纯文本';
                setTimeout(() => { statusSpan.innerText = ''; }, 2000);
                return;
            }
            try {
                await navigator.clipboard.write([
                    new ClipboardItem({ 'text/html': new Blob([htmlString], { type: 'text/html' }), 'text/plain': new Blob([plainText], { type: 'text/plain' }) })
                ]);
                statusSpan.innerText = '已复制到剪贴板';
                setTimeout(() => { statusSpan.innerText = ''; }, 2000);
            } catch (err) {
                try { await navigator.clipboard.writeText(plainText); } catch(e) { fallbackCopy(plainText); }
                statusSpan.innerText = '已复制纯文本';
                setTimeout(() => { statusSpan.innerText = ''; }, 2000);
            }
        };
        regenerateBtn.onclick = async () => {
            const confirmed = await confirm('重新生成将覆盖当前回答，确定吗？');
            if (!confirmed) return;
            const res = await fetch('/regenerate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ user_message: userMsg })
            });
            const data = await res.json();
            if (res.ok && data.assistant_message) {
                answerDiv.innerHTML = md.render(data.assistant_message);
                fixLinksInContainer(answerDiv);
                group.dataset.assistantMsg = data.assistant_message;
                if (data.thinking) {
                    let thinkingContainer = wrapper.querySelector('.thinking-container');
                    if (!thinkingContainer) {
                        thinkingContainer = document.createElement('div');
                        thinkingContainer.className = 'thinking-container';
                        const header = document.createElement('div');
                        header.className = 'thinking-header';
                        header.onclick = function() { toggleThinking(this); };
                        const arrowSpan = document.createElement('span');
                        arrowSpan.className = 'arrow';
                        arrowSpan.textContent = '▶';
                        const labelSpan = document.createElement('span');
                        labelSpan.textContent = '思考过程';
                        const previewSpan = document.createElement('span');
                        previewSpan.className = 'thinking-preview';
                        previewSpan.innerText = data.thinking.substring(0, 80) + '...';
                        header.appendChild(arrowSpan);
                        header.appendChild(labelSpan);
                        header.appendChild(previewSpan);
                        const contentDiv = document.createElement('div');
                        contentDiv.className = 'thinking-content';
                        contentDiv.innerHTML = md.render(data.thinking);
                        thinkingContainer.appendChild(header);
                        thinkingContainer.appendChild(contentDiv);
                        wrapper.insertBefore(thinkingContainer, answerDiv);
                    } else {
                        const contentDiv = thinkingContainer.querySelector('.thinking-content');
                        contentDiv.innerHTML = md.render(data.thinking);
                        const previewSpan = thinkingContainer.querySelector('.thinking-preview');
                        previewSpan.innerText = data.thinking.substring(0, 80) + '...';
                    }
                }
            } else {
                alert('重新生成失败');
            }
        };
        let currentRating = null;
        upBtn.onclick = () => { currentRating = 'up'; upBtn.classList.add('selected'); downBtn.classList.remove('selected'); };
        downBtn.onclick = () => { currentRating = 'down'; downBtn.classList.add('selected'); upBtn.classList.remove('selected'); };
        submitBtn.onclick = async () => {
            if (!currentRating) { statusSpan.innerText = '请先选择评分'; return; }
            const comment = commentInput.value.trim();
            const payload = { rating: currentRating, comment, user_message: userMsg, assistant_response: assistantMsg };
            const res = await fetch('/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include', body: JSON.stringify(payload) });
            if (res.ok) {
                statusSpan.innerText = '感谢您的反馈！';
                submitBtn.disabled = true;
                upBtn.disabled = true;
                downBtn.disabled = true;
                commentInput.disabled = true;
            } else {
                statusSpan.innerText = '提交失败';
            }
        };
        messagesDiv.appendChild(group);
        scrollToBottom();
    }

    // ======================== File Station Functions ========================
    let fileStationData = [];
    let selectedFileIds = new Set();
    const fileStationBtn = document.getElementById('fileStationBtn');
    const fileStationModal = document.getElementById('fileStationModal');
    const closeFileStationModal = document.getElementById('closeFileStationModal');

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
    const dailyReportChatBtn = document.getElementById('dailyReportChatBtn');
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
    const uploadToStationBtn = document.getElementById('uploadToStationBtn');
    const stationFileInput = document.getElementById('stationFileInput');
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
    let selectedKnowledgeFiles = [];
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
            if (data.warning && warningSpan) warningSpan.innerHTML = '⚠️ ' + data.message + '，请删除旧的聊天记录以释放空间。';
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
    const batchFileInputContainer = document.getElementById('batchFileInputContainer');
    const batchCompareBtn = document.getElementById('batchCompareBtn');
    const clearBatchFilesBtn = document.getElementById('clearBatchFilesBtn');
    const templateFileInput = document.getElementById('templateFileInput');
    const selectTemplateBtn = document.getElementById('selectTemplateBtn');
    const clearTemplateBtn = document.getElementById('clearTemplateBtn');
    const templateFileNameSpan = document.getElementById('templateFileName');
    const checkTextSim = document.getElementById('checkTextSim');
    const checkKeyInfo = document.getElementById('checkKeyInfo');
    const checkFileAttr = document.getElementById('checkFileAttr');
    const checkImageSim = document.getElementById('checkImageSim');
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

    let selectedTemplateFile = null;

    function updateBatchFileNames(files) {
        const span = document.getElementById('batchFileNames');
        const clearBtn = document.getElementById('clearBatchFilesBtn');
        if (files.length === 0) {
            span.textContent = '';
            if (clearBtn) clearBtn.style.display = 'none';
        } else {
            const names = files.map(f => f.name).join(', ');
            span.textContent = `📁 已选 ${files.length} 个文件: ${names}`;
            if (clearBtn) clearBtn.style.display = 'inline-block';
        }
    }

    batchCompareBtn.onclick = () => {
        batchFileInputContainer.innerHTML = '';
        const newInput = document.createElement('input');
        newInput.type = 'file';
        newInput.multiple = true;
        newInput.accept = '*/*';
        newInput.style.display = 'none';
        newInput.id = 'batchFileInput';
        batchFileInputContainer.appendChild(newInput);
        newInput.click();
        newInput.onchange = async (e) => {
            const files = Array.from(newInput.files);
            if (files.length === 0) return;
            if (files.length < 2) {
                alert('请至少选择2个文件进行对比');
                batchFileInputContainer.innerHTML = '';
                updateBatchFileNames([]);
                return;
            }
            if (files.length > 10) {
                alert('最多选择10个文件');
                batchFileInputContainer.innerHTML = '';
                updateBatchFileNames([]);
                return;
            }
            const semanticCheckbox = document.getElementById('checkSemantic');
            if (files.length > 10 && semanticCheckbox && semanticCheckbox.checked) {
                addSystemMessage('⚠️ 文件数超过10个，智能语义分析已自动关闭。');
                semanticCheckbox.checked = false;
            }
            const checkItems = {
                text_sim: checkTextSim.checked,
                key_info: checkKeyInfo.checked,
                file_attr: checkFileAttr.checked,
                image_sim: checkImageSim.checked,
                semantic: semanticCheckbox ? semanticCheckbox.checked : false
            };
            updateBatchFileNames(files);
            try {
                await performBatchCompare(files, selectedTemplateFile, checkItems);
            } catch (err) {
                console.error('Batch compare error:', err);
                addSystemMessage('批量对比失败: ' + (err.message || '未知错误'));
            } finally {
                batchFileInputContainer.innerHTML = '';
                updateBatchFileNames([]);
            }
        };
    };

    if (clearBatchFilesBtn) {
        clearBatchFilesBtn.onclick = () => {
            batchFileInputContainer.innerHTML = '';
            updateBatchFileNames([]);
        };
    }

    selectTemplateBtn.onclick = () => templateFileInput.click();
    templateFileInput.onchange = () => {
        if (templateFileInput.files.length) {
            selectedTemplateFile = templateFileInput.files[0];
            templateFileNameSpan.textContent = selectedTemplateFile.name;
        }
    };
    clearTemplateBtn.onclick = async () => {
        if (selectedTemplateFile) {
            const confirmed = await confirm('确定要清除已选中的模板文件吗？');
            if (!confirmed) return;
        }
        selectedTemplateFile = null;
        templateFileInput.value = '';
        templateFileNameSpan.textContent = '';
    };

    // ── Quote Anomaly standalone ──
    initQuoteAnomalyTool();

    // ── Relationship Extraction standalone ──
    initRelationshipTool();

    // ── Typo Detection standalone ──
    initTypoDetectionTool();

    // ── Admin sidebar: result history viewers ──
    initAdminResultViewers();

    async function performBatchCompare(files, templateFile, checkItems) {
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
        if (templateFile) formData.append('template_file', templateFile);
        formData.append('check_items', JSON.stringify(checkItems));
        const tempGroup = document.createElement('div');
        tempGroup.className = 'message-group';
        const tempTimerDiv = document.createElement('div');
        tempTimerDiv.className = 'temp-timer';
        tempTimerDiv.textContent = '⏱️ 0.0s (批量对比中)';
        tempGroup.appendChild(tempTimerDiv);
        messagesDiv.appendChild(tempGroup);
        scrollToBottom();
        const startTime = Date.now();
        // Register with floating task indicator
        const batchTaskId = 'batch-' + Date.now();
        _activeTaskIds.add(batchTaskId);
        _taskResults[batchTaskId] = { label: '批量对比中' };
        updateFloatingIndicator();
        const timerInterval = setInterval(() => {
            const elapsed = (Date.now() - startTime) / 1000;
            tempTimerDiv.textContent = `⏱️ ${elapsed.toFixed(1)}s (批量对比中)`;
        }, 100);
        if (currentBatchAbortController) currentBatchAbortController.abort();
        currentBatchAbortController = new AbortController();
        const signal = currentBatchAbortController.signal;
        const cancelBtn = document.createElement('button');
        cancelBtn.textContent = '取消对比';
        cancelBtn.style.marginLeft = '10px';
        cancelBtn.style.background = '#e74c3c';
        cancelBtn.style.color = 'white';
        cancelBtn.style.border = 'none';
        cancelBtn.style.borderRadius = '8px';
        cancelBtn.style.padding = '2px 8px';
        cancelBtn.onclick = () => {
            currentBatchAbortController.abort();
            cancelBtn.disabled = true;
            cancelBtn.textContent = '取消中...';
        };
        tempGroup.appendChild(cancelBtn);
        try {
            const response = await fetch('/compare_batch', {
                method: 'POST',
                credentials: 'include',
                body: formData,
                signal
            });
            clearInterval(timerInterval);
            _activeTaskIds.delete(batchTaskId);
            updateFloatingIndicator();
            tempGroup.remove();
            if (response.status === 409) {
                const data = await response.json();
                alert(data.message || '操作冲突');
                return;
            }
            if (response.status === 400) {
                const data = await response.json();
                alert('批量对比失败，请检查文件类型是否支持');
                return;
            }
            const data = await response.json();
            if (data.success) {
                const currentThread = sessionStorage.getItem('currentThreadId') ||
                    (await fetch('/get_sessions', { credentials: 'include' }).then(r=>r.json())).sessions[0]?.thread_id;
                if (currentThread) await loadSession(currentThread, true);
                else location.reload();
                await checkStorage();
                // Permanent download URL from new ZIP system
                if (data.download_url) {
                    showToast(`✅ 对比完成！${data.pair_count}对 · <a href="${data.download_url}" download style="color:#16a34a;">📦 下载完整报告</a> · <span id="batchAnalyzeLink" style="color:#7c3aed;cursor:pointer;text-decoration:underline;" onclick="event.stopPropagation();this.onclick=async()=>{const q=prompt('输入分析需求(可选):','生成这批对比的综合分析报告');if(!q)return;const r=await fetch('/admin/projects/'+(sessionStorage.getItem('lastProjectId')||1)+'/ai_assist',{method:'POST',headers:{'Content-Type':'application/json'},credentials:'include',body:JSON.stringify({query:'根据以下批量对比结果:\n'+JSON.stringify({pair_count:data.pair_count,groups:data.comparison_groups||[]}).substring(0,3000)+'\n\n'+q})});const d=await r.json();if(d.result)showContentModal('对比分析报告',d.result);}">📊 AI分析报告</span>`, 'success', 12000);
                } else if (data.token) {
                    // Fallback for old token-based system
                    showToast('对比完成！<a href="/export_batch_excel_download/'+data.token+'" download>📥 下载Excel报告</a>', 'success', 8000);
                }
            } else {
                addSystemMessage('批量对比失败，请重试');
            }
        } catch (err) {
            clearInterval(timerInterval);
            tempGroup.remove();
            if (err.name === 'AbortError') addSystemMessage('批量对比已取消。');
            else addSystemMessage('网络错误，批量对比失败');
        } finally {
            if (currentBatchAbortController === signal.controller) currentBatchAbortController = null;
        }
    }

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
                         ${(!p.status || p.status === 'active') ? `<button onclick="event.stopPropagation();if(!window._showAuditModal){showToast('审计模块未就绪，请刷新页面','error',3000);return;}window._currentProjectId=${p.id};window._showAuditModal()" style="background:#7c3aed;color:#fff;border:none;border-radius:4px;padding:2px 8px;font-size:.65rem;cursor:pointer;flex-shrink:0;margin-left:4px;" title="全量审计">📋</button>` : ''}
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

    if (chatTab && adminTab) {
        chatTab.onclick = () => {
            saveActiveTab('chat');
            switchToPanel('chatInterface');
            switchSidebarPane('chat');
            toggleQuickLinksButton(false);
            syncActiveTabWithView();
            loadHistoryList();
            setTimeout(() => scrollToBottom(), 100);
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
            saveActiveTab('projects');
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
            saveActiveTab('recycle');
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
            saveActiveTab('db');
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
                if (!res.ok) { content.innerHTML = '<p style="color:#ef4444;">权限不足</p>'; return; }
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
                html += '<button class="fb-btn" onclick="window.submitAuditFeedback(5,this)">👍 有帮助</button>';
                html += '<button class="fb-btn" onclick="window.submitAuditFeedback(1,this)">👎 无帮助</button>';
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
            } catch(e) { content.innerHTML = '<p style="color:#ef4444;">加载失败</p>'; }
        }

    // ======================== Knowledge Lab Tab ========================
    if (knowledgeLabTab) {
        knowledgeLabTab.onclick = async () => {
            saveActiveTab('knowledge');
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
        const auditLogBtn = document.getElementById('sidebarAuditLogBtn');
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
                    buttonsHtml = `<button class="open-project" data-id="${p.id}" data-name="${escapeHtml(p.name)}" data-status="${p.status}" style="background: #27ae60; color: white; border: none; border-radius: 4px; padding: 4px 8px;">📂 打开</button>`;
                    if (p.status === 'active') {
                        buttonsHtml += `
                            <button class="member-manager-btn" data-id="${p.id}" style="background: #3498db; color: white; border: none; border-radius: 4px; padding: 4px 8px;">👥 成员管理</button>
                            <button class="finish-project-btn" data-id="${p.id}" style="background: #f39c12; color: white; border: none; border-radius: 4px; padding: 4px 8px;">🏁 完成并归档</button>
                            <button class="abort-project" data-id="${p.id}" style="background: #e67e22; color: white; border: none; border-radius: 4px; padding: 4px 8px;">⛔ 中止</button>
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

        // Reset per-project audit history when switching projects
        const auditDetails = document.getElementById('projectAuditHistoryDetails');
        if (auditDetails) {
            auditDetails.open = false;
            auditDetails._auditLoaded = false;
            const auditPanel = document.getElementById('projectAuditHistoryPanel');
            if (auditPanel) auditPanel.innerHTML = '<span style="color:var(--card-muted);">点击展开查看该项目的审计记录...</span>';
            const auditCount = document.getElementById('projectAuditCount');
            if (auditCount) auditCount.textContent = '';
        }

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
                <span style="font-size:0.85rem; color:#1e40af;">📦 此项目已归档，仅可查看和下载</span>
                ${archiveFilename ? `<a href="/admin/projects/${projectId}/download_archive/${encodeURIComponent(archiveFilename)}" style="background:#2563eb; color:white; text-decoration:none; border-radius:6px; padding:6px 14px; font-size:0.82rem;">📥 下载归档ZIP</a>` : ''}
            </div>`;
        } else if (status === 'aborted') {
            archiveBanner = `<div style="background:#fef2f2; border:1px solid #fecaca; border-radius:8px; padding:10px 14px; margin-bottom:12px;">
                <span style="font-size:0.85rem; color:#991b1b;">⛔ 此项目已中止，仅可查看</span>
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
            <div id="folderTreeContainer" style="margin-bottom: 20px;"></div>
            <div id="fileListContainer"></div>
            <details id="projectAuditHistoryDetails" style="margin-top:16px;border:1px solid var(--card-border);border-radius:8px;padding:8px 12px;background:var(--card-bg);">
                <summary style="font-weight:600;font-size:0.82rem;cursor:pointer;color:var(--card-muted);">
                    📋 审计历史 <span id="projectAuditCount" style="font-size:0.7rem;color:var(--card-muted);font-weight:normal;"></span>
                </summary>
                <div id="projectAuditHistoryPanel" style="font-size:0.72rem;margin-top:8px;">
                    <span style="color:var(--card-muted);">点击展开查看该项目的审计记录...</span>
                </div>
            </details>
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

        // Wire audit history lazy-load toggle (inside project view)
        setTimeout(() => {
            const auditDetails = document.getElementById('projectAuditHistoryDetails');
            if (auditDetails && !auditDetails._auditListenerSet) {
                auditDetails._auditListenerSet = true;
                auditDetails.addEventListener('toggle', () => {
                    if (auditDetails.open && !auditDetails._auditLoaded) {
                        auditDetails._auditLoaded = true;
                        loadProjectAuditHistory();
                    }
                });
            }
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
            const res = await fetch('/compare_batch', { method: 'POST', credentials: 'include', body: formData });
            const data = await res.json();
            if (res.ok) {
                showToast(`对比完成！${data.pair_count}对结果已保存`, 'success', 3000);
                // Refresh batch history if the modal is open
                if (typeof loadBatchHistory === 'function') loadBatchHistory();
            } else {
                alert(data.error || '对比失败');
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
    const backToProjectsBtn = document.getElementById('backToProjectsBtn');
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

    // ======================== Knowledge Lab ========================
    async function loadKnowledgeLabFiles(cachedData) {
        const container = document.getElementById('labFileList');
        container.innerHTML = '<p>加载中...</p>';
        try {
            const raw = cachedData || await (await fetch('/knowledge_lab/list', { credentials: 'include' })).json();
            const data = raw;
            const files = data.files || [];
            if (files.length === 0) {
                container.innerHTML = '<p>暂无文件。点击上方按钮上传。</p>';
                return;
            }
            let html = '<ul style="list-style:none; padding-left:0; margin:0;">';
            for (const f of files) {
                const hasSkill = !!f.has_skill;
                const escapedName = escapeHtml(f.original_name);
                const sizeKb = ((f.file_size||0)/1024).toFixed(1);
                const uploadedStr = new Date(f.uploaded_at).toLocaleString();
                const isAdmin = sessionStorage.getItem('isAdmin') === 'true';
                const renameBtn = isAdmin ? `<button class="rename-lab-file" data-id="${f.id}" data-type="lab" style="background:none;border:none;cursor:pointer;font-size:0.65rem;padding:0 2px;" title="重命名">✏️</button>` : '';
                html += `
                    <li class="file-item" data-id="${f.id}" data-filename="${escapedName}" style="margin-bottom:8px; padding:6px 8px; background: var(--card-bg, #f9f9f9); border-radius:6px; border:1px solid var(--border-color, #e0e0e0);">
                        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                            <span style="font-size:0.85rem; font-weight:500;">📄 ${escapedName}</span>${renameBtn}
                            ${hasSkill ? '<span style="font-size:0.7rem; background:#dcfce7; color:#16a34a; border-radius:8px; padding:1px 6px;">🧠 已提取技能</span>' : ''}
                            <span style="font-size:0.7rem; color:#888;">(${sizeKb} KB)</span>
                            <span style="font-size:0.7rem; color:#888;">上传于 ${uploadedStr}</span>
                            <button class="delete-lab-file" data-id="${f.id}" style="margin-left:auto; background:#e74c3c; color:white; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">删除</button>
                        </div>
                        <div style="margin-top:4px; display:flex; gap:4px; flex-wrap:wrap;">
                            <button class="view-lab-content" data-id="${f.id}" data-filename="${escapedName}" style="background:#2c3e50; color:white; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">📄 预览源文件</button>
                            ${hasSkill
                                ? `<a href="/knowledge_lab/skill/${f.id}" target="_blank" style="background:#16a34a; color:white; text-decoration:none; border-radius:4px; padding:2px 8px; font-size:0.7rem; display:inline-block;">📥 下载技能</a>`
                                : `<button class="generate-lab-skill" data-id="${f.id}" style="background:#bccfde; color:#1e293b; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">🧠 提取技能</button>`
                            }
                        </div>
                    </li>
                `;
            }
            html += '</ul>';
            container.innerHTML = html;

            // Attach rename button handlers
            document.querySelectorAll('.rename-lab-file').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const newName = await prompt('新名称:', btn.closest('.file-item')?.dataset.filename || '');
                    if (newName && newName.trim()) {
                        const url = btn.dataset.type === 'lab' ? `/knowledge_lab/rename/${btn.dataset.id}` : `/company_kb/rename/${btn.dataset.id}`;
                        const r = await fetch(url, {method:'POST',headers:{'Content-Type':'application/json'},credentials:'include',body:JSON.stringify({name:newName.trim()})});
                        if (r.ok) { loadKnowledgeLabFiles(); loadSidebarKnowledge(); }
                        else { const d=await r.json().catch(()=>({})); alert(d.error||'重命名失败'); }
                    }
                };
            });
            // Attach delete button handlers
            document.querySelectorAll('.delete-lab-file').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const fileId = btn.dataset.id;
                    if (await confirm('确定永久删除此文件吗？')) {
                        const res = await fetch(`/knowledge_lab/delete/${fileId}`, { method: 'POST', credentials: 'include' });
                        if (res.ok) {
                            showToast('删除成功', 'success', 2000);
                            loadKnowledgeLabFiles();
                            loadSidebarKnowledge();
                        } else alert('删除失败');
                    }
                };
            });

            // View source file content
            document.querySelectorAll('.view-lab-content').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const fileId = btn.dataset.id;
                    const filename = btn.dataset.filename;
                    try {
                        const res = await fetch(`/knowledge_lab/content/${fileId}`, { credentials: 'include' });
                        if (res.ok) {
                            const data = await res.json();
                            if (data.content) showContentModal(filename, data.content);
                            else alert('文件内容为空');
                        } else alert('加载失败');
                    } catch(_) { alert('加载失败'); }
                };
            });

            // Generate skill on-demand
            document.querySelectorAll('.generate-lab-skill').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const fileId = btn.dataset.id;
                    btn.disabled = true;
                    btn.textContent = '⏳ 分析中...';
                    try {
                        const res = await fetch(`/knowledge_lab/generate_skill/${fileId}`, { method: 'POST', credentials: 'include' });
                        const data = await res.json();
                        if (res.ok) {
                            showSkillFeedback(btn, data, 'knowledge_lab', parseInt(fileId));
                        } else {
                            const msg = data.error || '生成失败';
                            const hint = data.hint || '';
                            alert(msg + (hint ? '\n' + hint : ''));
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

            // Initialize selection manager
            new FileListManager(container, {
                onDoubleClick: (item) => {
                    const fileId = item.dataset.id;
                    const filename = item.dataset.filename;
                    fetch(`/knowledge_lab/content/${fileId}`).then(r => r.json()).then(data => {
                        if (data.content) showContentModal(filename, data.content);
                        else alert('无法加载内容');
                    }).catch(() => alert('加载失败'));
                }
            });
        } catch (err) {
            console.error(err);
            container.innerHTML = '<p>加载失败</p>';
        }
    }
    function updateKnowledgeBaseButton() {
        const btn = document.getElementById('knowledgeBaseBtn');
        if (selectedKnowledgeFiles.length) {
            btn.innerHTML = `📚 知识库(${selectedKnowledgeFiles.length})`;
        } else {
            btn.innerHTML = '📚 知识库';
        }
    }
    // Personal KB category handling
    const labCatSelect = document.getElementById('labCategorySelect');
    const labCustomCat = document.getElementById('labCustomCategory');
    if (labCatSelect) {
        labCatSelect.addEventListener('change', () => {
            if (labCatSelect.value === '自定义') {
                labCustomCat.style.display = 'inline-block';
            } else {
                labCustomCat.style.display = 'none';
                labCustomCat.value = '';
            }
        });
    }
    document.getElementById('uploadLabFileBtn').onclick = () => document.getElementById('labFileInput').click();
    document.getElementById('labFileInput').addEventListener('change', async (e) => {
        const files = Array.from(e.target.files);
        if (!files.length) return;

        const container = document.getElementById('labFileList');
        if (!container) return;

        // Helper to create a temporary placeholder element
        function createPlaceholder(filename, fileSize) {
            const placeholder = document.createElement('div');
            placeholder.className = 'lab-file-placeholder';
            placeholder.style.opacity = '0.5';
            placeholder.style.padding = '8px 0';
            placeholder.style.borderBottom = '1px solid #ddd';
            placeholder.style.display = 'flex';
            placeholder.style.justifyContent = 'space-between';
            placeholder.style.alignItems = 'center';
            placeholder.innerHTML = `
                <span>⏳ 上传中: ${escapeHtml(filename)} (${(fileSize/1024).toFixed(1)} KB) <span class="spinner">⏳</span></span>
                <span style="color:#888;">处理中...</span>
            `;
            return placeholder;
        }

        // Helper to create the final file entry (replaces placeholder)
        function createFinalEntry(file) {
            const fileDiv = document.createElement('div');
            fileDiv.style.borderBottom = '1px solid #ddd';
            fileDiv.style.padding = '8px 0';
            fileDiv.style.display = 'flex';
            fileDiv.style.justifyContent = 'space-between';
            fileDiv.style.alignItems = 'center';
            const skillInfo = file.skill_generated ? 
                '<span style="font-size:.65rem;color:#22c55e;margin:0 4px;" title="技能已生成">🧠</span>' :
                '<span class="gen-skill-link" data-fileid="'+file.file_id+'" style="font-size:.65rem;color:var(--card-muted);cursor:pointer;margin:0 4px;text-decoration:underline;text-decoration-style:dotted;" title="从此文件中提取结构化知识框架">🧠提取技能</span>';
            fileDiv.innerHTML = `
                <span>📄 ${escapeHtml(file.filename)} (${(file.file_size/1024).toFixed(1)} KB) - 上传于 ${new Date(file.uploaded_at).toLocaleString()}</span>
                <span style="display:flex;align-items:center;gap:4px;">
                    ${skillInfo}
                    <button class="delete-lab-file" data-id="${file.file_id}" style="background:#e74c3c; color:white; border:none; border-radius:4px; padding:4px 8px;">删除</button>
                </span>
            `;
            // Skill extract link handler
            setTimeout(() => {
                const genLink = fileDiv.querySelector('.gen-skill-link');
                if (genLink) genLink.onclick = async () => {
                    genLink.textContent = '⏳ 提取中...';
                    genLink.style.cursor = 'default';
                    try {
                        const res = await fetch('/knowledge_lab/generate_skill/' + genLink.dataset.fileid, { method: 'POST', credentials: 'include' });
                        const data = await res.json();
                        if (res.ok) { showSkillFeedback(genLink, data, 'knowledge_lab', parseInt(genLink.dataset.fileid)); }
                        else { genLink.textContent = '重试'; genLink.style.cursor = 'pointer'; genLink.style.color = '#ef4444'; showToast(data.error || '提取失败', 'error'); }
                    } catch(e) { genLink.textContent = '重试'; genLink.style.cursor = 'pointer'; genLink.style.color = '#ef4444'; }
                };
            }, 50);
            // Attach delete handler
            const delBtn = fileDiv.querySelector('.delete-lab-file');
            delBtn.onclick = async () => {
                if (await confirm('确定永久删除此文件吗？')) {
                    const res = await fetch(`/knowledge_lab/delete/${file.file_id}`, { method: 'POST', credentials: 'include' });
                    if (res.ok) {
                        fileDiv.remove();
                        showToast('删除成功', 'success', 2000);
                    } else {
                        alert('删除失败');
                    }
                }
            };
            return fileDiv;
        }

        // Insert placeholders at the top of the list
        const placeholders = [];
        for (const file of files) {
            const placeholder = createPlaceholder(file.name, file.size);
            container.prepend(placeholder);
            placeholders.push({ file, placeholder });
        }

        // Upload each file sequentially (to avoid overwhelming the server)
        let uploaded = 0;
        let hadErrors = false;
        showProgress(`上传 ${files.length} 个文件...`, 'bar');
        for (const item of placeholders) {
            const { file, placeholder } = item;
            const formData = new FormData();
            formData.append('file', file);
            let labCat = labCatSelect ? labCatSelect.value : '';
            if (labCat === '自定义') labCat = labCustomCat ? labCustomCat.value.trim() : '';
            if (labCat) formData.append('category', labCat);
            try {
                updateProgress((uploaded / files.length) * 90, `上传中: ${file.name} (${uploaded+1}/${files.length})`);
                const res = await fetch('/knowledge_lab/upload', { method: 'POST', credentials: 'include', body: formData });
                const data = await res.json();
                if (res.ok && data.success) {
                    // Replace placeholder with final entry
                    const finalEntry = createFinalEntry({
                        file_id: data.file_id,
                        filename: data.filename,
                        file_size: data.file_size,
                        uploaded_at: data.uploaded_at
                    });
                    placeholder.replaceWith(finalEntry);
                    showToast(`✅ ${data.filename} 上传成功`, 'success', 2000);
                } else {
                    // Show error and remove the placeholder
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

        // Clear the input so the same files can be uploaded again
        e.target.value = '';
    });
    document.getElementById('refreshLabListBtn').onclick = loadKnowledgeLabFiles;
    document.getElementById('refreshSkillOverviewBtn').onclick = loadSkillOverview;

    // ── Notebook ──
    async function loadNotebook() {
        const list = document.getElementById('notebookList'); if (!list) return;
        try {
            const r = await fetch('/notebook', {credentials:'include'});
            const d = await r.json();
            const notes = d.notes || [];
            if (!notes.length) { list.innerHTML = '<span style="font-size:0.75rem;color:var(--card-muted);">暂无笔记。点击➕新建笔记。</span>'; return; }
            list.innerHTML = notes.map(n => `<div style="background:var(--card-bg);border:1px solid var(--card-border);border-radius:6px;padding:8px;cursor:pointer;" onclick="_openNotebook('${escapeHtml(n.id)}')">
                <div style="font-weight:600;font-size:0.75rem;">📝 ${escapeHtml(n.id)}</div>
                <div style="font-size:0.62rem;color:var(--card-muted);margin-top:2px;">${escapeHtml(n.preview||'')}</div>
                <div style="font-size:0.55rem;color:var(--card-muted);margin-top:4px;display:flex;justify-content:space-between;">
                    <span>${new Date(n.modified).toLocaleString()}</span>
                    <button onclick="event.stopPropagation();_deleteNotebook('${escapeHtml(n.id)}')" style="color:#ef4444;background:none;border:none;font-size:0.55rem;cursor:pointer;">🗑</button>
                </div>
            </div>`).join('');
        } catch(_) { list.innerHTML = '<span style="color:#ef4444;font-size:0.75rem;">加载失败</span>'; }
    }

    window._openNotebook = async function(noteId) {
        const r = await fetch('/notebook/' + encodeURIComponent(noteId), {credentials:'include'});
        const d = await r.json();
        const note = d.note;
        document.getElementById('notebookEditTitle').value = note.id;
        document.getElementById('notebookEditContent').value = note.content;
        document.getElementById('notebookEditor').style.display = 'block';
        document.getElementById('notebookSummary').innerHTML = '';
    };

    window._deleteNotebook = async function(noteId) {
        if (!confirm('删除笔记 "'+noteId+'"？')) return;
        await fetch('/notebook/' + encodeURIComponent(noteId), {method:'DELETE', credentials:'include'});
        loadNotebook();
    };

    document.getElementById('notebookNewBtn').onclick = () => {
        document.getElementById('notebookEditTitle').value = '';
        document.getElementById('notebookEditContent').value = '';
        document.getElementById('notebookEditor').style.display = 'block';
        document.getElementById('notebookSummary').innerHTML = '';
    };
    document.getElementById('notebookRefreshBtn').onclick = loadNotebook;
    document.getElementById('notebookCancelBtn').onclick = () => {
        document.getElementById('notebookEditor').style.display = 'none';
    };
    document.getElementById('notebookSaveBtn').onclick = async () => {
        const title = document.getElementById('notebookEditTitle').value.trim();
        const content = document.getElementById('notebookEditContent').value;
        if (!title) { alert('标题不能为空'); return; }
        const r = await fetch('/notebook/' + encodeURIComponent(title), {
            method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
            body:JSON.stringify({content})
        });
        if (r.ok) { document.getElementById('notebookEditor').style.display = 'none'; loadNotebook(); }
        else alert('保存失败');
    };
    document.getElementById('notebookSummarizeBtn').onclick = async () => {
        const title = document.getElementById('notebookEditTitle').value.trim();
        if (!title) return;
        const el = document.getElementById('notebookSummary');
        el.innerHTML = '⏳ AI摘要生成中...';
        const r = await fetch('/notebook/' + encodeURIComponent(title) + '/summarize', {method:'POST', credentials:'include'});
        const d = await r.json();
        el.innerHTML = d.summary ? '🤖 ' + escapeHtml(d.summary) : '失败';
    };
    document.getElementById('notebookSearch').addEventListener('keydown', async (e) => {
        if (e.key !== 'Enter') return;
        const q = e.target.value.trim();
        if (!q) { loadNotebook(); return; }
        const r = await fetch('/notebook/search', {
            method:'POST', headers:{'Content-Type':'application/json'}, credentials:'include',
            body:JSON.stringify({query:q})
        });
        const d = await r.json();
        const results = d.results || [];
        const list = document.getElementById('notebookList');
        if (!results.length) { list.innerHTML = '<span style="font-size:0.75rem;color:var(--card-muted);">无匹配结果。</span>'; return; }
        list.innerHTML = results.map(r => `<div style="background:var(--card-bg);border:1px solid var(--card-border);border-radius:6px;padding:6px;cursor:pointer;" onclick="_openNotebook('${escapeHtml(r.note_id)}')">
            <div style="font-weight:600;font-size:0.7rem;">📝 ${escapeHtml(r.note_id)} (${(r.score*100).toFixed(0)}%)</div>
            <div style="font-size:0.6rem;color:var(--card-muted);">${escapeHtml(r.snippet||'')}</div>
        </div>`).join('');
    });
    loadNotebook();
    function showContentModal(title, content) {
        const existing = document.getElementById('contentPreviewModal');
        if (existing) existing.remove();

        const modal = document.createElement('div');
        modal.id = 'contentPreviewModal';
        modal.className = 'modal';
        modal.style.display = 'block';
        modal.innerHTML = `
            <div class="modal-content" style="width: 80%; max-width: 800px; max-height: 80vh; overflow-y: auto;">
                <span class="close" style="float: right; cursor: pointer;">&times;</span>
                <h4>${escapeHtml(title)}</h4>
                <div style="white-space: pre-wrap; font-family: monospace; font-size: 0.85rem; line-height: 1.5; background: #f5f5f5; padding: 12px; border-radius: 8px; margin-top: 12px;">
                    ${escapeHtml(content).replace(/\n/g, '<br>')}
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        modal.querySelector('.close').onclick = () => modal.remove();
        modal.onclick = (e) => { if (e.target === modal) modal.remove(); };
    }
    function renderCompanyKb(container, data) {
        const files = data.files || [];
        if (!files.length) { container.innerHTML = '<p>暂无公司知识库文件。</p>'; return; }
        const isAdmin = sessionStorage.getItem('isAdmin') === 'true';
        let html = '<ul style="list-style:none; padding-left:0; margin:0;">';
        for (const f of files) {
            const hasSkill = !!f.has_skill;
            const escapedName = escapeHtml(f.filename);
            const sizeKb = ((f.file_size||0)/1024).toFixed(1);
            const uploadedStr = new Date(f.uploaded_at).toLocaleString();
            html += `<li class="file-item" data-id="${f.id}" data-filename="${escapedName}" style="margin-bottom:8px; padding:6px 8px; background: var(--card-bg, #f9f9f9); border-radius:6px; border:1px solid var(--border-color, #e0e0e0);">
                <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <span style="font-size:0.85rem; font-weight:500;">📄 ${escapedName}</span>${isAdmin ? `<button class="rename-company-file" data-id="${f.id}" style="background:none;border:none;cursor:pointer;font-size:0.65rem;padding:0 2px;" title="重命名">✏️</button>` : ''}
                    ${hasSkill ? '<span style="font-size:0.7rem; background:#dcfce7; color:#16a34a; border-radius:8px; padding:1px 6px;">🧠 已提取技能</span>' : ''}
                    <span style="font-size:0.7rem; color:#888;">(${sizeKb} KB)</span>
                    <span style="font-size:0.7rem; color:#888;">分类: ${escapeHtml(f.category || '无')}</span>
                    <span style="font-size:0.7rem; color:#888;">${escapeHtml(f.uploaded_by_name || 'admin')}</span>
                    <span style="font-size:0.7rem; color:#888;">${uploadedStr}</span>
                    ${isAdmin ? `<button class="delete-company-file" data-id="${f.id}" style="margin-left:auto; background:#e74c3c; color:white; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">删除</button>` : ''}
                </div>
                <div style="margin-top:4px; display:flex; gap:4px; flex-wrap:wrap;">
                    <button class="view-company-content" data-id="${f.id}" data-filename="${escapedName}" style="background:#2c3e50; color:white; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">📄 预览源文件</button>
                    ${hasSkill
                        ? `<a href="/company_kb/skill/${f.id}" target="_blank" style="background:#16a34a; color:white; text-decoration:none; border-radius:4px; padding:2px 8px; font-size:0.7rem; display:inline-block;">📥 下载技能</a>`
                        : (isAdmin ? `<button class="generate-company-skill" data-id="${f.id}" style="background:#bccfde; color:#1e293b; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">🧠 提取技能</button>` : '')
                    }
                    ${hasSkill && f.category === '模板'
                        ? `<button class="generate-template-doc" data-id="${f.id}" data-source="company_knowledge_base" style="background:#f59e0b; color:white; border:none; border-radius:4px; padding:2px 8px; font-size:0.7rem;">📝 生成文档</button>`
                        : ''
                    }
                </div>
            </li>`;
        }
        container.innerHTML = html + '</ul>';
        wireCompanyKbButtons();
    }

    async function loadCompanyKnowledgeBase(cachedData) {
        const container = document.getElementById('companyKbList');
        if (!container) return;
        const search = document.getElementById('companyKbSearch')?.value.trim() || '';
        const category = document.getElementById('companyKbCategoryFilter')?.value || '';
        if (cachedData && !search && !category) { renderCompanyKb(container, cachedData); return; }
        container.innerHTML = '<p>加载中...</p>';
        let url = `/company_kb/list?`;
        if (search) url += `search=${encodeURIComponent(search)}&`;
        if (category) url += `category=${encodeURIComponent(category)}&`;
        try {
            const res = await fetch(url, { credentials: 'include' });
            if (!res.ok) throw new Error('Failed to load company KB');
            renderCompanyKb(container, await res.json());
        } catch(e) { container.innerHTML = '<p>加载失败</p>'; }
    }

    function wireCompanyKbButtons() {
            document.querySelectorAll('.rename-company-file').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const newName = await prompt('新名称:', btn.closest('.file-item')?.dataset.filename || '');
                    if (newName && newName.trim()) {
                        const r = await fetch(`/company_kb/rename/${btn.dataset.id}`, {method:'POST',headers:{'Content-Type':'application/json'},credentials:'include',body:JSON.stringify({name:newName.trim()})});
                        if (r.ok) { loadCompanyKnowledgeBase(); loadCompanyCategories(); }
                        else { const d=await r.json().catch(()=>({})); alert(d.error||'重命名失败'); }
                    }
                };
            });
            document.querySelectorAll('.delete-company-file').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation(); const fileId = btn.dataset.id;
                    if (await confirm('确定永久删除此公司知识库文件吗？')) {
                        const res = await fetch(`/company_kb/delete/${fileId}`, { method: 'POST', credentials: 'include' });
                        if (res.ok) { showToast('删除成功', 'success', 2000); loadCompanyKnowledgeBase(); loadCompanyCategories(); }
                        else alert('删除失败');
                    }
                };
            });
            document.querySelectorAll('.view-company-content').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation(); const fileId = btn.dataset.id; const filename = btn.dataset.filename;
                    const res = await fetch(`/company_kb/content/${fileId}`, { credentials: 'include' });
                    const data = await res.json();
                    if (res.ok && data.content) showContentModal(filename, data.content); else alert('无法加载内容');
                };
            });
            document.querySelectorAll('.generate-company-skill').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation(); const fileId = btn.dataset.id;
                    btn.disabled = true; btn.textContent = '⏳ 分析中...';
                    try {
                        const res = await fetch(`/company_kb/generate_skill/${fileId}`, { method: 'POST', credentials: 'include' });
                        const d = await res.json();
                        if (res.ok) { showSkillFeedback(btn, d, 'company_kb', parseInt(fileId)); }
                        else { alert((d.error||'生成失败')+((d.hint||'')?'\n'+d.hint:'')); btn.disabled = false; btn.textContent = '🧠 提取技能'; }
                    } catch(_) { alert('网络错误'); btn.disabled = false; btn.textContent = '🧠 提取技能'; }
                };
            });
            document.querySelectorAll('.generate-template-doc').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation(); const fileId = btn.dataset.id; const source = btn.dataset.source || 'company_knowledge_base';
                    btn.disabled = true; const origText = btn.textContent; btn.textContent = '⏳ 生成中...';
                    try {
                        const formData = new FormData(); formData.append('source', source);
                        const res = await fetch(`/templates/${fileId}/generate_doc`, { method: 'POST', credentials: 'include', body: formData });
                        if (res.ok) {
                            const blob = await res.blob(); const url = URL.createObjectURL(blob);
                            const a = document.createElement('a'); a.href = url; a.download = '文档.docx'; a.click();
                            URL.revokeObjectURL(url); showToast('✅ 文档已生成', 'success', 2000);
                        } else { const d = await res.json().catch(()=>({})); alert(d.error || '生成失败'); }
                    } catch(_) { alert('网络错误'); }
                    btn.disabled = false; btn.textContent = origText;
                };
            });
        }

    async function loadCompanyCategories() {
        try {
            const res = await fetch('/company_kb/categories', { credentials: 'include' });
            const data = await res.json();
            const categories = data.categories || [];
            const select = document.getElementById('companyKbCategoryFilter');
            select.innerHTML = '<option value="">所有分类</option>';
            for (const cat of categories) {
                select.innerHTML += `<option value="${escapeHtml(cat)}">${escapeHtml(cat)}</option>`;
            }
        } catch (err) {
            console.error('Failed to load categories', err);
        }
    }

    async function loadSkillOverview(labData, coData) {
        const container = document.getElementById('skillOverviewList');
        const counter = document.getElementById('skillOverviewCount');
        if (!container) return;
        container.innerHTML = '<p style="grid-column:1/-1; color:var(--card-muted);">加载中...</p>';

        if (counter) {
            try {
                const ragRes = await fetch('/admin/rag_stats', { credentials: 'include' });
                if (ragRes.ok) {
                    const rag = await ragRes.json(); const s = rag.stats || {}; const total = s.total || 0;
                    const status = total > 0 ? `<span style="font-size:0.7rem; background:#dcfce7; color:#16a34a; border-radius:8px; padding:1px 6px; margin-left:4px;">RAG 🟢</span>` : `<span style="font-size:0.7rem; background:#fef2f2; color:#dc2626; border-radius:8px; padding:1px 6px; margin-left:4px;" title="请管理员点侧边栏重建全部索引">RAG ⚠️</span>`;
                    counter.innerHTML = status;
                }
            } catch(_) {}
        }
        try {
            if (!labData) { const r = await fetch('/knowledge_lab/list', { credentials: 'include' }); labData = await r.json(); }
            if (!coData) { const r = await fetch('/company_kb/list', { credentials: 'include' }); coData = await r.json(); }
            const labSkills = (labData.files || []).filter(f => f.has_skill);
            const coSkills = (coData.files || []).filter(f => f.has_skill);
            const all = [...labSkills.map(f => ({...f, _src: 'personal', _url: '/knowledge_lab/skill/'})),
                        ...coSkills.map(f => ({...f, _src: 'company', _url: '/company_kb/skill/', _name: f.filename}))];

            if (counter) counter.innerHTML = `${counter.innerHTML} (${all.length}个技能)`;
            if (!all.length) {
                container.innerHTML = '<p style="grid-column:1/-1; color:var(--card-muted);">暂无技能。上传文件后系统会自动提取，或手动点击"提取技能"按钮生成。</p>';
                return;
            }
            container.innerHTML = all.map(f => {
                const name = f._src === 'company' ? (f._name || f.original_name) : (f.original_name || f.filename);
                return `<div style="background:var(--card-bg); border:1px solid var(--card-border); border-radius:8px; padding:10px 12px; display:flex; flex-direction:column; gap:6px;">
                    <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:4px;">
                        <div style="font-size:0.85rem; font-weight:500; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${escapeHtml(name)}">
                            ${f._src === 'company' ? '🏢' : '📁'} ${escapeHtml(name)}</div>
                        <span style="font-size:0.65rem; color:var(--card-muted); flex-shrink:0;">${f._src === 'company' ? '公司' : '个人'} · ${((f.file_size||0)/1024).toFixed(0)}KB</span>
                    </div>
                    <div style="display:flex; gap:6px;">
                        <a href="${f._url}${f.id}" target="_blank" style="background:#16a34a; color:white; text-decoration:none; border-radius:4px; padding:3px 10px; font-size:0.72rem;">📥 下载技能</a>
                        ${f._src === 'personal' ? `<button class="view-lab-content" data-id="${f.id}" data-filename="${escapeHtml(name)}" style="background:#2c3e50; color:white; border:none; border-radius:4px; padding:3px 10px; font-size:0.72rem;">📄 源文件</button>` : ''}
                    </div>
                </div>`;
            }).join('');
            // Wire source-file preview buttons
            document.querySelectorAll('#skillOverviewList .view-lab-content').forEach(btn => {
                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const fileId = btn.dataset.id;
                    const filename = btn.dataset.filename;
                    try {
                        const res = await fetch(`/knowledge_lab/content/${fileId}`, { credentials: 'include' });
                        if (res.ok) { const d = await res.json(); if (d.content) showContentModal(filename, d.content); else alert('文件内容为空'); }
                        else alert('加载失败');
                    } catch(_) { alert('加载失败'); }
                };
            });
        } catch(e) {
            container.innerHTML = '<p style="grid-column:1/-1; color:#ef4444;">加载失败</p>';
        }
    }

    // Company Knowledge Base category handling
    const categorySelect = document.getElementById('companyCategorySelect');
    const customCategoryInput = document.getElementById('companyCustomCategory');

    if (categorySelect) {
        categorySelect.addEventListener('change', () => {
            if (categorySelect.value === '自定义') {
                customCategoryInput.style.display = 'inline-block';
                customCategoryInput.required = true;
            } else {
                customCategoryInput.style.display = 'none';
                customCategoryInput.required = false;
                customCategoryInput.value = '';
            }
        });
    }

    // Company file upload – validate category before opening file picker
    const uploadCompanyBtn = document.getElementById('uploadCompanyFileBtn');
    const companyFileInput = document.getElementById('companyFileInput');

    if (uploadCompanyBtn && companyFileInput) {
        uploadCompanyBtn.onclick = () => {
            let category = categorySelect.value;
            if (category === '自定义') {
                category = customCategoryInput.value.trim();
                if (!category) {
                    alert('请先输入自定义分类名称');
                    return;
                }
            }
            if (!category) {
                alert('请先选择或输入分类');
                return;
            }
            companyFileInput.click();
        };

        companyFileInput.onchange = async (e) => {
            const files = e.target.files;
            if (!files.length) return;

            uploadCompanyBtn.disabled = true;
            const originalText = uploadCompanyBtn.textContent;
            uploadCompanyBtn.textContent = '⏳ 上传中...';

            let category = categorySelect.value;
            if (category === '自定义') {
                category = customCategoryInput.value.trim();
            }

            for (const file of files) {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('category', category);
                try {
                    const res = await fetch('/company_kb/upload', { method: 'POST', credentials: 'include', body: formData });
                    const data = await res.json();
                    if (res.ok) {
                        showToast(`✅ ${data.filename} 上传成功 (分类: ${category})`, 'success', 2000);
                        if (data.overlap_suggestions && data.overlap_suggestions.length) {
                            setTimeout(() => showOverlapSuggestions(data.overlap_suggestions, data.file_id, 'company_knowledge_base'), 500);
                        }
                    } else {
                        showToast('上传失败，请重试', 'error', 3000);
                    }
                } catch (err) {
                    showToast('❌ 网络错误', 'error', 3000);
                }
            }
            await loadCompanyKnowledgeBase();
            await loadCompanyCategories();
            uploadCompanyBtn.disabled = false;
            uploadCompanyBtn.textContent = originalText;
            e.target.value = '';
        };
    }

    document.getElementById('refreshCompanyKbBtn').onclick = () => {
        loadCompanyKnowledgeBase();
    };
    document.getElementById('companyKbSearch').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') loadCompanyKnowledgeBase();
    });
    if (sessionStorage.getItem('isAdmin') === 'true') {
        document.getElementById('adminCompanyTools').style.display = 'block';
    }

    // ── Category filter pills for chat RAG ──
    window._selectedRagCategory = '';
    const catFilterBar = document.getElementById('categoryFilterBar');
    const catPills = document.querySelectorAll('.cat-pill');
    const catClearBtn = document.getElementById('catFilterClear');

    function showCatFilterIfNeeded() {
        const el = document.getElementById('categoryFilterBar');
        if (!el) return;
        if (selectedKnowledgeFiles.length > 0) {
            el.style.display = 'flex';
        } else {
            el.style.display = 'none';
            window._selectedRagCategory = '';
        }
    }

    if (catPills.length) {
        catPills.forEach(pill => {
            pill.addEventListener('click', () => {
                catPills.forEach(p => {
                    p.style.background = '';
                    p.style.color = '';
                });
                pill.style.background = 'var(--accent)';
                pill.style.color = 'white';
                window._selectedRagCategory = pill.dataset.cat || '';
                if (catClearBtn) catClearBtn.style.display = window._selectedRagCategory ? 'inline' : 'none';
            });
        });
    }
    if (catClearBtn) {
        catClearBtn.addEventListener('click', () => {
            catPills.forEach(p => { p.style.background = ''; p.style.color = ''; });
            const allPill = document.querySelector('.cat-pill[data-cat=""]');
            if (allPill) { allPill.style.background = 'var(--accent)'; allPill.style.color = 'white'; }
            window._selectedRagCategory = '';
            catClearBtn.style.display = 'none';
        });
    }
    // ── Skill overlap suggestions after upload ──
    function showOverlapSuggestions(overlaps, newFileId, sourceTable) {
        if (!overlaps || !overlaps.length) return;
        const msg = overlaps.map(o =>
            `📄 ${escapeHtml(o.name)} (相似度 ${o.similarity}%)`
        ).join('\n');
        if (!confirm(`检测到新上传的文件与以下 ${overlaps.length} 个同分类技能高度相似，是否合并？\n\n${msg}\n\n选择「确定」合并，「取消」忽略`)) return;
        // Merge the top overlap into the new file
        const top = overlaps[0];
        fetch('/admin/skill_supersession/respond', {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: 'merge',
                keep_id: newFileId,
                merge_id: top.id,
                source: sourceTable || 'company_knowledge_base'
            })
        }).then(r => r.json()).then(data => {
            if (data.success) showToast(`✅ 已与「${top.name}」合并`, 'success', 3000);
            else showToast('❌ 合并失败: ' + (data.error || ''), 'error', 5000);
        }).catch(() => showToast('❌ 合并请求失败', 'error', 3000));
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
                    const tabMap = { chat:'chatTabBtn', projects:'adminTabBtn', recycle:'recycleBinTabBtn', db:'databaseTabBtn', knowledge:'knowledgeLabTabBtn', wiki:'wikiTabBtn', timeline:'timelineTabBtn', stats:'analyticsTabBtn', review:'reviewTabBtn', templates:'templatesTabBtn', cases:'casesTabBtn' };
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
            const data = cachedData || await (await fetch('/admin/db_tables', { credentials: 'include' })).json();
            const select = document.getElementById('dbTableSelect');
            select.innerHTML = '<option value="">-- 选择表 --</option>';
            for (const table of data.tables) {
                select.innerHTML += `<option value="${escapeHtml(table)}">${escapeHtml(table)}</option>`;
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
                } else {
                    html += '<p style="color:var(--card-muted);">Wiki暂无内容。上传知识库文件后，系统将自动生成Wiki页面。</p>';
                }
                content.innerHTML = html;
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
            } catch(e) {
                console.error('Wiki load failed:', e);
                content.innerHTML = '<p style="color:#ef4444;font-size:.78rem;">加载Wiki失败</p>';
            }
        }

        templatesTab.onclick = async () => {
            saveActiveTab('templates');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            templatesTab.classList.add('active');
            switchToPanel('templatesPanel');
            switchSidebarPane('chat');
        };

        casesTab.onclick = async () => {
            saveActiveTab('cases');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            casesTab.classList.add('active');
            switchToPanel('casesPanel');
            switchSidebarPane('chat');
        };

        wikiTab.onclick = async () => {
            saveActiveTab('wiki');
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
                { prefix: 'comparisons', label: '对比', icon: '🔗' },
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
                            let html = '<div style="display:flex;gap:6px;margin-bottom:12px;">';
                            html += '<button class="wiki-back-btn" style="background:#e2e8f0;border:none;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">← 返回Wiki首页</button>';
                            if (sessionStorage.getItem('isAdmin') === 'true') {
                                html += '<button class="wiki-edit-btn" data-edit-path="' + escapeHtml(node.path.replace(/\.md$/i, '')) + '" style="background:#fef3c7;border:1px solid #f59e0b;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">✏️ 编辑</button>';
                                html += '<button class="wiki-delete-btn" data-delete-path="' + escapeHtml(node.path.replace(/\.md$/i, '')) + '" style="background:#fee2e2;border:1px solid #ef4444;border-radius:6px;padding:4px 12px;font-size:.75rem;cursor:pointer;">🗑️ 删除</button>';
                            }
                            html += '</div>';
                            html += '<div style="background:var(--card-bg);border-radius:8px;padding:16px;">';
                            if (d.frontmatter?.tags) {
                                html += '<div style="margin-bottom:8px;">' + d.frontmatter.tags.map(t => `<span style="background:#e2e8f0;padding:2px 8px;border-radius:4px;font-size:.65rem;margin-right:4px;">#${escapeHtml(t)}</span>`).join('') + '</div>';
                            }
                            html += d.html;
                            html += '</div>';
                            content.innerHTML = html;
                            content.querySelector('.wiki-back-btn').onclick = () => { const wt = document.getElementById('wikiTabBtn'); if (wt) wt.click(); };
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

    // ======================== Timeline Tab ========================
    const timelineTabBtn = document.getElementById('timelineTabBtn');
    const timelinePanel = document.getElementById('timelinePanel');
    if (timelineTabBtn && timelinePanel) {
        timelineTabBtn.onclick = async () => {
            saveActiveTab('timeline');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            timelineTabBtn.classList.add('active');
            switchToPanel('timelinePanel');
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
            const [catRes, tRes] = await Promise.all([
                fetch('/timeline/legal/categories', { credentials: 'include' }),
                fetch('/timeline/' + currentProjectId, { credentials: 'include' })
            ]);
            const catData = await catRes.json();
            const tData = await tRes.json();

            if (tData.success && tData.id) {
                setup.style.display = 'none';
                content.innerHTML = _renderTimelineView(tData);
            } else {
                setup.style.display = 'block';
                content.innerHTML = '';
                if (catData.success && catData.categories) {
                    const catSel = document.getElementById('timelineCategorySelect');
                    const mSel = document.getElementById('timelineMethodSelect');
                    catSel.innerHTML = '<option value="">选择类别...</option>';
                    catData.categories.forEach(c => {
                        catSel.innerHTML += '<option value="' + c.code + '">' + c.name + '</option>';
                    });
                    catSel.onchange = () => {
                        mSel.innerHTML = '<option value="">选择方式...</option>';
                        const sel = catData.categories.find(c => c.code === catSel.value);
                        if (sel && sel.methods) {
                            sel.methods.forEach(m => {
                                mSel.innerHTML += '<option value="' + m.code + '">' + m.name + '</option>';
                            });
                        }
                    };
                    document.getElementById('timelineCreateBtn').onclick = async () => {
                        const cat = catSel.value;
                        const meth = mSel.value;
                        const start = document.getElementById('timelineStartDate').value;
                        if (!cat || !meth || !start) { alert('请填写所有必填项'); return; }
                        try {
                            const cr = await fetch('/timeline/' + currentProjectId, {
                                method: 'POST', headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({category_code: cat, method_code: meth, planned_start_date: start}),
                                credentials: 'include'
                            });
                            const cd = await cr.json();
                            if (cd.success) { loadTimelinePanel(); } else { alert(cd.error || '创建失败'); }
                        } catch(e) { alert('创建失败: ' + e.message); }
                    };
                }
            }
        } catch (e) {
            content.innerHTML = '<p style="color:var(--card-muted);">加载失败: ' + e.message + '</p>';
        }
    }

    function _renderTimelineView(tl) {
        var ms = tl.milestones || [];
        var html = '<div style="margin-bottom:12px;">';
        html += '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px;">';
        html += '<span style="font-size:0.75rem;"><b>类别:</b> ' + (tl.category_code || '') + '</span>';
        html += '<span style="font-size:0.75rem;"><b>方式:</b> ' + (tl.method_code || '') + '</span>';
        html += '<span style="font-size:0.75rem;"><b>计划开始:</b> ' + (tl.planned_start_date || '') + '</span>';
        html += '<span style="font-size:0.75rem;"><b>状态:</b> ' + _statusBadge(tl.status || '') + '</span>';
        html += '</div>';

        if (tl.diff_summary) {
            var ds = tl.diff_summary;
            html += '<div style="font-size:0.7rem;color:var(--card-muted);margin-bottom:8px;">';
            html += '总节点: ' + ds.total_milestones + ' | 已完成: ' + ds.completed;
            html += ' | 延期: ' + ds.delayed + ' | 准点: ' + ds.on_time;
            if (ds.total_delay_days > 0) html += ' | 累计延期: ' + ds.total_delay_days + '天';
            html += '</div>';
        }
        html += '</div>';

        html += '<div style="max-height:500px;overflow-y:auto;border:1px solid var(--card-border);border-radius:8px;">';
        ms.forEach(function(m, i) {
            var bg = m.status === 'completed' ? '#f0fff4' : (m.diff_days && m.diff_days > 0 ? '#fff5f5' : 'transparent');
            html += '<div style="display:flex;align-items:center;padding:6px 12px;border-bottom:1px solid var(--card-border);background:' + bg + ';font-size:0.72rem;">';
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
            html += _statusBadge(m.status || 'pending');
            html += '</div>';
        });
        html += '</div>';

        html += '<div style="margin-top:12px;display:flex;gap:6px;">';
        html += '<button onclick="loadTimelinePanel()" class="file-btn" style="font-size:0.7rem;">🔄 刷新</button>';
        html += '<button onclick="fetch(\'/timeline/' + tl.project_id + '/suggestions\',{credentials:\'include\'}).then(r=>r.json()).then(d=>{if(d.success)alert(\'规则建议已刷新\')}).catch(e=>alert(e.message))" class="file-btn" style="font-size:0.7rem;">💡 查看建议</button>';
        html += '<button onclick="fetch(\'/timeline/' + tl.project_id + '/diff\',{credentials:\'include\'}).then(r=>r.json()).then(d=>{if(d.success){var s=JSON.stringify(d.summary,2);alert(s.substring(0,500))}}).catch(e=>alert(e.message))" class="file-btn" style="font-size:0.7rem;">📊 差异报告</button>';
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

    // ======================== Usage Tab (Admin only) ========================
    const analyticsTabBtn = document.getElementById('analyticsTabBtn');
    const analyticsPanel = document.getElementById('analyticsPanel');
    if (analyticsTabBtn && analyticsPanel) {
        let _usageLoaded = { rc: false, assets: false, archives: false, styles: false, auditConfig: false, skillAudit: false };
        analyticsTabBtn.onclick = async function() {
            saveActiveTab('stats');
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
                        `<span title="消息总数">✉️<b>${stats.total_messages}</b></span>`, `<span title="存储用量">💾<b>${stats.storage_mb}MB</b></span>`,
                        `<span title="活跃项目数">📂<b>${stats.active_projects}</b></span>`);
                    if (stats.rag_stats?.total > 0) items.push(`<span title="RAG索引数">🧠<b>${stats.rag_stats.total}</b></span>`);
                } else {
                    items.push(`<span title="会话总数">💬<b>${stats.total_sessions}</b></span>`, `<span title="消息总数">✉️<b>${stats.total_messages}</b></span>`,
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
                ['auditConfigDetails', 'auditConfig', loadAuditConfig],
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
    const reviewPanel = document.getElementById('reviewPanel');
    if (reviewTabBtn && reviewPanel) {
        console.log('Audit: reviewTab block entered');
        let _reviewLoaded = { ingest: false, training: false, history: false, structured: false, workload: false, auditHistory: false };
        reviewTabBtn.onclick = async function() {
            saveActiveTab('review');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            reviewTabBtn.classList.add('active');
            switchToPanel('reviewPanel');
            switchSidebarPane('review');
            _reviewLoaded = { ingest: false, training: false, history: false, structured: false, workload: false, auditHistory: false };
            const content = document.getElementById('reviewContent');
            const sections = document.getElementById('reviewSections');
            try {
                const res = await fetch('/admin/analytics', { credentials: 'include' });
                if (res.status === 403) { content.innerHTML = '<span>需要登录</span>'; return; }
                if (!res.ok) { content.innerHTML = `<span style="color:#e74c3c;">服务器错误 (${res.status})</span>`; return; }
                const stats = await res.json();
                const items = [`<span title="用户总数">👥<b>${stats.total_users}</b></span>`, `<span title="会话总数">💬<b>${stats.total_sessions}</b></span>`, `<span title="消息总数">✉️<b>${stats.total_messages}</b></span>`];
                content.innerHTML = `<div style="font-size:0.7rem;color:var(--card-muted);">${items.join(' \u00b7 ')}</div>
                    <div style="margin:10px 0;text-align:center;display:flex;align-items:center;justify-content:center;gap:12px;">
                        <span id="auditRunningBadge" style="display:none;font-size:0.72rem;color:#f59e0b;">\u23f3 \u5ba1\u8ba1\u8fdb\u884c\u4e2d...</span>
                    </div>`;
                if (sections) { sections.style.display = 'block'; _setupReviewLazySections(); _checkRunningAudit(); }
                _checkStaleReviews();
                _updateReviewSidebarStatus();
                _checkRunningAudit();
            } catch(e) { console.error('Review stats load error:', e); content.innerHTML = '<span style="color:#e74c3c;">加载失败</span>'; }
        };

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

        // ── Full Audit button uses inline onclick → window._showAuditModal ──

        async function _checkRunningAudit() {
            const badge = document.getElementById('auditRunningBadge');
            if (!badge) return;
            const pid = window._currentProjectId || currentProjectId;
            if (!pid) { badge.style.display = 'none'; return; }
            try {
                const r = await fetch(`/audit/running/${pid}`, {credentials:'include'});
                const d = await r.json();
                if (r.ok && d && d.run_id) {
                    badge.style.display = 'inline';
                    const chatBtn = document.getElementById('chatFullAuditBtn');
                    if (chatBtn) chatBtn.disabled = true;
                } else {
                    badge.style.display = 'none';
                    const chatBtn = document.getElementById('chatFullAuditBtn');
                    if (chatBtn) chatBtn.disabled = false;
                }
            } catch(_) { badge.style.display = 'none'; }
        }

        function _setupReviewLazySections() {
            const map = [
                ['ingestDetails', 'ingest', loadIngestPanel],
                ['trainingDetails', 'training', loadTrainingExportPanel],
                ['ingestHistoryDetails', 'history', loadIngestHistory],
                ['structuredDocsDetails', 'structured', loadStructuredDocsPanel],
                ['workloadDetails', 'workload', loadWorkloadPanel],
                ['auditHistoryDetails', 'auditHistory', loadAuditHistory],
            ];
            for (const [id, key, fn] of map) {
                const el = document.getElementById(id);
                if (el && !el._listenerSet) {
                    el._listenerSet = true;
                    el.addEventListener('toggle', () => { if (el.open && !_reviewLoaded[key]) { _reviewLoaded[key] = true; fn(); } });
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
                                        progressEl.innerHTML += '<br>✅ 处理完成';
                                        clearInterval(poll);
                                        progressEl.innerHTML += '<div style="margin-top:6px;display:flex;gap:6px;"><button class="fb-btn" onclick="window.submitIngestFeedback(\''+d.task_id+'\',5,this)">👍 满意</button><button class="fb-btn" onclick="window.submitIngestFeedback(\''+d.task_id+'\',1,this)">👎 不满意</button></div>';
                                    }
                                    else if (sd.status === 'failed') { progressEl.innerHTML += '<br>❌ 处理失败'; clearInterval(poll); }
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
                    document.getElementById('styleMsg').innerHTML = '<span style="color:#22c55e;">✅ 批量分析已触发</span>';
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
                    if (r.ok) { msgEl.innerHTML = '<span style="color:#22c55e;">✅ '+d.message+'</span>'; loadTrainingExportPanel(); }
                    else msgEl.innerHTML = '<span style="color:#ef4444;">❌ '+(d.error||'失败')+'</span>';
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
                    if (r.ok) { msgEl.innerHTML = '<span style="color:#22c55e;">✅ '+d.message+'</span>'; loadTrainingExportPanel(); }
                    else msgEl.innerHTML = '<span style="color:#ef4444;">❌ '+(d.error||'失败')+'</span>';
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
                    if (r.ok) { msgEl.innerHTML = '<span style="color:#22c55e;">✅ '+d.message+'</span>'; loadTrainingExportPanel(); }
                    else msgEl.innerHTML = '<span style="color:#ef4444;">❌ '+(d.error||'失败')+'</span>';
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
                    if (r.ok) { msgEl.innerHTML = '<span style="color:#22c55e;">✅ '+d.message+'</span>'; loadTrainingExportPanel(); }
                    else msgEl.innerHTML = '<span style="color:#ef4444;">❌ '+(d.error||'失败')+'</span>';
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
                    if (r.ok) { msgEl.innerHTML = '<span style="color:#22c55e;">✅ '+d.message+'</span>'; loadTrainingExportPanel(); }
                    else msgEl.innerHTML = '<span style="color:#ef4444;">❌ '+(d.error||'失败')+'</span>';
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
                                } else msgEl.innerHTML = '<span style="color:#ef4444;">❌ '+(dd.error||'失败')+'</span>';
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
                                        msgEl.innerHTML = '<span style="color:#22c55e;">✅ '+dd.message+'</span>';
                                        // Reload file list
                                        exportFilesDetails._loaded = false;
                                        exportFilesDetails.open = false;
                                        setTimeout(() => { exportFilesDetails.open = true; }, 100);
                                    } else msgEl.innerHTML = '<span style="color:#ef4444;">❌ '+(dd.error||'Failed')+'</span>';
                                } catch(_) { msgEl.innerHTML = '<span style="color:#ef4444;">Network error</span>'; btn.disabled = false; }
                            };
                        });
                    } catch(_) { fileContent.innerHTML = '<span style="color:#ef4444;">Load failed</span>'; }
                });
            }
        } catch (_) { panel.innerHTML = '<span style="color:#ef4444;">Load failed</span>'; }
    }

    // ── Runtime Config Panel ──
    let _rcData = {}, _rcSchema = {}, _rcDirty = {};

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
                    const chain = Array.isArray(item.value) ? item.value : [['zhipu','glm-4.5-air'],['nvidia','nemotron-3-ultra-550b-a55b'],['deepseek','deepseek-v4-flash'],['deepseek','deepseek-v4-pro']];
                    const nfMark = item.is_not_factory ? ' <span style="color:#f59e0b;font-size:.6rem;" title="不在出厂预设范围内">[非出厂项]</span>' : '';
                    const provLabels = {auto:'自动检测',deepseek:'DeepSeek',zhipu:'智谱AI',qwen:'Qwen',siliconflow:'硅基流动',nvidia:'NVIDIA'};
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
            const now = Array.from(rows).map(r => [r.querySelector('.chain-provider')?.value || 'zhipu', r.querySelector('.chain-model')?.value || 'glm-4.5-air']);
            const orig = _rcData['llm_fallback_chain'];
            const nowStr = JSON.stringify(now);
            _rcDirty['llm_fallback_chain'] = (nowStr !== JSON.stringify(orig ?? [['zhipu','glm-4.5-air']])) ? now : undefined;
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
                    if (provSel) provSel.value = 'zhipu';
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
            vlTestResult.innerHTML = '⏳ 分析中...';
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
                if (r.ok) { document.getElementById('rcMsg').innerHTML = '<span style="color:#22c55e;">✅ '+d.message+'</span>'; loadRuntimeConfig(); }
                else document.getElementById('rcMsg').innerHTML = '<span style="color:#ef4444;">❌ '+(d.error||'失败')+'</span>';
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
                if (r.ok) { document.getElementById('rcMsg').innerHTML = '<span style="color:#22c55e;">✅ '+d.message+'</span>'; loadRuntimeConfig(); }
                else document.getElementById('rcMsg').innerHTML = '<span style="color:#ef4444;">❌ '+(d.error||'失败')+'</span>';
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
                document.getElementById('rcMsg').innerHTML = '<span style="color:#22c55e;">✅ '+d.message+'</span>';
                _rcDirty = {};
                const modDot = document.getElementById('rcModifiedDot');
                if (modDot) modDot.style.display = 'none';
                loadRuntimeConfig();
            } else {
                document.getElementById('rcMsg').innerHTML = '<span style="color:#ef4444;">❌ '+(d.error||'保存失败')+'</span>';
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
                document.getElementById('rcMsg').innerHTML = '<span style="color:#22c55e;">✅ '+d.message+'</span>';
                _rcDirty = {};
                const modDot = document.getElementById('rcModifiedDot');
                if (modDot) modDot.style.display = 'none';
                loadRuntimeConfig();
            } else {
                document.getElementById('rcMsg').innerHTML = '<span style="color:#ef4444;">❌ '+(d.error||'Failed')+'</span>';
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
                icon.textContent = '▶';  // collapsed
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
                list.innerHTML = '<li style="color:#999;font-size:0.72rem;">无进行中的任务</li>';
                return;
            }
            list.innerHTML = tasks.map(t => {
                const statusIcon = t.status === 'running' ? '🔄' : t.status === 'completed' ? '✅' : t.status === 'failed' ? '❌' : '⏳';
                const barWidth = t.progress || 0;
                const barColor = t.status === 'failed' ? '#ef4444' : t.status === 'completed' ? '#22c55e' : '#5a7c9b';
                const taskId = t.task_id || '';
                const clickable = (t.status === 'completed' || t.status === 'failed');
                const style = clickable ? 'padding:6px 8px;font-size:0.73rem;border-bottom:1px solid var(--border-color);cursor:pointer;' : 'padding:6px 8px;font-size:0.73rem;border-bottom:1px solid var(--border-color);';
                return `<li data-task-id="${taskId}" data-task-status="${t.status || ''}" style="${style}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span>${statusIcon} ${escapeHtml(t.label || t.type || '任务')}</span>
                        <span style="font-size:0.65rem;color:var(--card-muted);">${t.progress || 0}%</span>
                    </div>
                    ${t.status === 'running' ? `<div style="background:var(--border-color);height:3px;border-radius:2px;margin-top:3px;"><div style="background:${barColor};height:100%;width:${barWidth}%;border-radius:2px;transition:width .3s;"></div></div>` : ''}
                    ${t.message && t.status !== 'completed' ? `<div style="font-size:0.65rem;color:var(--card-muted);margin-top:2px;">${escapeHtml(t.message)}</div>` : ''}
                </li>`;
            }).join('');
        } catch(e) { /* silent */ }
    }

    function _handleTaskClick(e) {
        const li = e.target.closest('li[data-task-id]');
        if (!li) return;
        const tid = li.getAttribute('data-task-id');
        const status = li.getAttribute('data-task-status');
        if (!tid || (status !== 'completed' && status !== 'failed')) return;
        e.stopPropagation();
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
                if (!d.success) return;
                const result = d.result || '';
                const shortResult = result.length > 80 ? result.substring(0, 80) + '...' : result;
                const info = (d.label || d.type || '') + ' — ' + d.status +
                    (d.message ? ' | ' + d.message : '') +
                    (shortResult ? ' | ' + shortResult : '');
                const t2 = document.createElement('div');
                t2.setAttribute('class', 'toast ' + (d.status === 'failed' ? 'error' : 'info'));
                t2.textContent = info;
                t2.style.cursor = 'pointer';
                t2.onclick = () => t2.remove();
                tc.appendChild(t2);
                setTimeout(() => { if (t2.parentNode) t2.remove(); }, 8000);
            }).catch(() => { toast.textContent = 'Error loading task'; });
    }
    document.addEventListener('click', _handleTaskClick);
    console.log('[TASK] click handler registered on document');

    function startBgTasksPolling() {
        loadBgTasks();
        if (_bgTasksPollTimer) clearInterval(_bgTasksPollTimer);
        _bgTasksPollTimer = setInterval(loadBgTasks, 5000);
    }

    function stopBgTasksPolling() {
        if (_bgTasksPollTimer) { clearInterval(_bgTasksPollTimer); _bgTasksPollTimer = null; }
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
    // Quote Anomaly standalone tool
    // ══════════════════════════════════════════════════════════════════
    function initQuoteAnomalyTool() {
        const fileInput = document.getElementById('quoteAnomalyFileInput');
        const selectBtn = document.getElementById('selectQuoteAnomalyFilesBtn');
        const runBtn = document.getElementById('runQuoteAnomalyBtn');
        const fileNames = document.getElementById('quoteAnomalyFileNames');
        const statusEl = document.getElementById('quoteAnomalyStatus');
        const resultsPanel = document.getElementById('quoteAnomalyResultsPanel');

        if (!selectBtn || !runBtn || !fileInput) return;

        let selectedFiles = [];

        selectBtn.onclick = () => fileInput.click();

        fileInput.onchange = () => {
            selectedFiles = Array.from(fileInput.files);
            if (selectedFiles.length > 0) {
                fileNames.textContent = selectedFiles.map(f => f.name).join(', ');
                runBtn.disabled = false;
            } else {
                fileNames.textContent = '';
                runBtn.disabled = true;
            }
        };

        runBtn.onclick = async () => {
            if (selectedFiles.length === 0) return;
            runBtn.disabled = true;
            statusEl.textContent = '正在分析报价异常...';
            statusEl.style.color = '#e67e22';
            resultsPanel.style.display = 'none';

            try {
                const formData = new FormData();
                selectedFiles.forEach(f => formData.append('files', f));

                // Use cross-bidder endpoint if multiple files, single-doc otherwise
                const endpoint = selectedFiles.length >= 2 ? '/compare_bidders_quotes' : '/check_quote_anomaly';
                if (selectedFiles.length === 1) formData.append('file', selectedFiles[0]);

                const resp = await fetch(endpoint, { method: 'POST', body: formData });
                const data = await resp.json();
                if (resp.ok) {
                    renderQuoteAnomalyResults(data, resultsPanel);
                    resultsPanel.style.display = 'block';
                    statusEl.textContent = '✓ 报价异常检测完成';
                    statusEl.style.color = '#27ae60';
                } else {
                    statusEl.textContent = '✗ ' + (data.error || '检测失败');
                    statusEl.style.color = '#e74c3c';
                }
            } catch (err) {
                statusEl.textContent = '✗ 网络错误: ' + err.message;
                statusEl.style.color = '#e74c3c';
            } finally {
                runBtn.disabled = false;
            }
        };
    }

    function renderQuoteAnomalyResults(data, panel) {
        let html = '';
        const perBidder = data.per_bidder || (data.doc_name ? [data] : []);

        if (perBidder.length === 0 && !data.risk_score && !data.doc_name) {
            panel.innerHTML = '<p>无数据</p>';
            return;
        }

        // Single doc mode
        if (!data.per_bidder && data.doc_name) {
            html += `<p><strong>文档:</strong> ${escapeHtml(data.doc_name)} | <strong>风险评分:</strong> <span style="color:${data.risk_score > 50 ? '#e74c3c' : data.risk_score > 20 ? '#e67e22' : '#27ae60'}">${(data.risk_score||0).toFixed(1)}</span></p>`;
            html += '<table style="width:100%;border-collapse:collapse;font-size:0.72rem;">';
            html += '<tr><th>CV</th><th>同价疑义</th><th>异常降幅</th><th>聚类</th><th>本福特偏差</th></tr>';
            html += `<tr>`;
            html += `<td>${(data.cv||0).toFixed(4)}</td>`;
            html += `<td>${data.same_rate_flag ? '⚠️ 是' : '否'}</td>`;
            html += `<td>${data.abnormal_drop_flag ? '⬇️ 是' : '否'}</td>`;
            html += `<td>${data.clustering_flag ? '🔗 是' : '否'}</td>`;
            html += `<td>${(data.benford_deviation||0).toFixed(3)}</td></tr></table>`;
            if (data.details && data.details.length) {
                html += '<ul style="font-size:0.7rem;margin-top:4px;">';
                data.details.forEach(d => html += `<li>${escapeHtml(d)}</li>`);
                html += '</ul>';
            }
            if (data.daxie_mismatches && data.daxie_mismatches.length) {
                html += '<p style="color:#e67e22;font-size:0.7rem;">⚠️ 大写金额不一致: ' + data.daxie_mismatches.length + ' 处</p>';
            }
        } else {
            // Cross-bidder mode
            html += `<p><strong>投标单位:</strong> ${perBidder.length} | <strong>最高风险:</strong> <span style="color:${(data.max_risk_score||0) > 50 ? '#e74c3c' : '#e67e22'}">${(data.max_risk_score||0).toFixed(1)}</span> | <strong>平均CV:</strong> ${(data.avg_cv||0).toFixed(4)}</p>`;
            if (data.cross_same_rate) html += '<p style="color:#e67e22;">⚠️ 跨投标单位同价疑义</p>';
            if (data.cross_clustering) html += '<p style="color:#e67e22;">🔗 跨投标单位聚类疑义</p>';
            html += '<table style="width:100%;border-collapse:collapse;font-size:0.7rem;">';
            html += '<tr><th>投标单位</th><th>风险</th><th>CV</th><th>同价</th><th>降幅</th><th>聚类</th></tr>';
            perBidder.forEach(pb => {
                html += `<tr>`;
                html += `<td>${escapeHtml((pb.filename||'').substring(0,25))}</td>`;
                html += `<td style="color:${pb.risk_score > 50 ? '#e74c3c' : pb.risk_score > 20 ? '#e67e22' : '#27ae60'}">${(pb.risk_score||0).toFixed(1)}</td>`;
                html += `<td>${(pb.cv||0).toFixed(4)}</td>`;
                html += `<td>${pb.same_rate_flag ? '⚠️' : '✓'}</td>`;
                html += `<td>${pb.abnormal_drop_flag ? '⬇️' : '✓'}</td>`;
                html += `<td>${pb.clustering_flag ? '🔗' : '✓'}</td></tr>`;
            });
            html += '</table>';
        }
        const docName = (data.per_bidder && data.per_bidder.length) ? data.per_bidder.map(function(p){return p.filename||''}).join(';').substring(0,80) : data.doc_name || '未知';
        html += '<div style="margin-top:8px;display:flex;gap:6px;font-size:0.7rem;align-items:center;">';
        html += '<span style="color:var(--card-muted);">分析结果有帮助吗？</span>';
        html += '<button class="fb-btn" onclick="window.submitQuoteFeedback(\''+escapeHtml(docName)+'\',5,this)">👍 有帮助</button>';
        html += '<button class="fb-btn" onclick="window.submitQuoteFeedback(\''+escapeHtml(docName)+'\',1,this)">👎 无帮助</button>';
        html += '</div>';
        panel.innerHTML = html;
    }

    // ══════════════════════════════════════════════════════════════════
    // Relationship Extraction standalone tool
    // ══════════════════════════════════════════════════════════════════
    function initRelationshipTool() {
        const fileInput = document.getElementById('relationshipFileInput');
        const selectBtn = document.getElementById('selectRelationshipFilesBtn');
        const runBtn = document.getElementById('runRelationshipBtn');
        const fileNames = document.getElementById('relationshipFileNames');
        const statusEl = document.getElementById('relationshipStatus');
        const resultsPanel = document.getElementById('relationshipResultsPanel');

        if (!selectBtn || !runBtn || !fileInput) return;

        let selectedFiles = [];

        selectBtn.onclick = () => fileInput.click();

        fileInput.onchange = () => {
            selectedFiles = Array.from(fileInput.files);
            if (selectedFiles.length > 0) {
                fileNames.textContent = selectedFiles.map(f => f.name).join(', ');
                runBtn.disabled = false;
            } else {
                fileNames.textContent = '';
                runBtn.disabled = true;
            }
        };

        runBtn.onclick = async () => {
            if (selectedFiles.length === 0) return;
            runBtn.disabled = true;
            statusEl.textContent = '正在提取关联关系...';
            statusEl.style.color = '#8e44ad';
            resultsPanel.style.display = 'none';

            try {
                const formData = new FormData();
                selectedFiles.forEach(f => formData.append('files', f));
                const resp = await fetch('/extract_relationships', { method: 'POST', body: formData });
                const data = await resp.json();
                if (resp.ok) {
                    renderRelationshipResults(data, resultsPanel);
                    resultsPanel.style.display = 'block';
                    statusEl.textContent = '✓ 关联关系分析完成';
                    statusEl.style.color = '#27ae60';
                } else {
                    statusEl.textContent = '✗ ' + (data.error || '分析失败');
                    statusEl.style.color = '#e74c3c';
                }
            } catch (err) {
                statusEl.textContent = '✗ 网络错误: ' + err.message;
                statusEl.style.color = '#e74c3c';
            } finally {
                runBtn.disabled = false;
            }
        };
    }

    function renderRelationshipResults(data, panel) {
        let html = '';
        html += `<p><strong>风险评分:</strong> <span style="color:${data.risk_score > 50 ? '#e74c3c' : data.risk_score > 20 ? '#e67e22' : '#27ae60'}">${(data.risk_score||0).toFixed(1)}</span> | <strong>实体:</strong> ${(data.entities||[]).length} | <strong>关系:</strong> ${(data.relationships||[]).length} | <strong>模块:</strong> ${(data.modules_run||[]).join(', ')}</p>`;

        if (data.red_flags && data.red_flags.length) {
            html += '<div style="margin-top:8px;"><strong style="color:#e74c3c;">🚨 风险警告:</strong><ul style="font-size:0.7rem;">';
            data.red_flags.slice(0, 10).forEach(f => html += `<li style="color:#e67e22;">${escapeHtml(f)}</li>`);
            html += '</ul></div>';
        }

        if (data.relationships && data.relationships.length) {
            html += '<table style="width:100%;border-collapse:collapse;font-size:0.7rem;margin-top:8px;">';
            html += '<tr><th>源实体</th><th>目标实体</th><th>类型</th><th>置信度</th><th>风险</th></tr>';
            data.relationships.slice(0, 20).forEach(r => {
                html += `<tr>`;
                html += `<td>${escapeHtml((r.source||'').substring(0,25))}</td>`;
                html += `<td>${escapeHtml((r.target||'').substring(0,25))}</td>`;
                html += `<td>${escapeHtml(r.type||'')}/${escapeHtml(r.subtype||'')}</td>`;
                html += `<td>${((r.confidence||0)*100).toFixed(0)}%</td>`;
                html += `<td>${r.risk_flag ? '⚠️' : '✓'}</td></tr>`;
            });
            html += '</table>';
            if (data.relationships.length > 20) html += `<p style="font-size:0.65rem;color:#888;">(仅显示前20项，共${data.relationships.length}项)</p>`;
        }

        // Company-personnel map (for manual review when 天眼查 is off)
        const cpm = data.company_personnel_map;
        if (cpm && cpm.companies && cpm.companies.length) {
            html += '<details style="margin-top:8px;"><summary style="cursor:pointer;font-weight:bold;font-size:0.72rem;">📋 公司与关键人员清单 (供人工审查)</summary>';
            html += '<table style="width:100%;border-collapse:collapse;font-size:0.68rem;margin-top:4px;">';
            html += '<tr><th>公司</th><th>关键人员</th><th>涉及文件</th></tr>';
            cpm.companies.slice(0, 15).forEach(c => {
                const personnel = (c.personnel||[]).map(p => `${escapeHtml(p.name)}(${escapeHtml(p.title)})`).join('; ');
                html += `<tr><td>${escapeHtml((c.name||'').substring(0,30))}</td><td>${personnel}</td><td>${c.file_count||0}个文件</td></tr>`;
            });
            html += '</table></details>';
        }

        if (!data.relationships || !data.relationships.length) {
            html += '<p style="font-size:0.72rem;color:#888;">未发现关联关系</p>';
        }

        panel.innerHTML = html;
    }

    // ══════════════════════════════════════════════════════════════════
    // Typo Detection standalone tool
    // ══════════════════════════════════════════════════════════════════
    function initTypoDetectionTool() {
        const fileInput = document.getElementById('typoFileInput');
        const selectBtn = document.getElementById('selectTypoFileBtn');
        const runBtn = document.getElementById('runTypoBtn');
        const fileName = document.getElementById('typoFileName');
        const statusEl = document.getElementById('typoStatus');
        const resultsPanel = document.getElementById('typoResultsPanel');
        const diffMode = document.getElementById('typoDiffMode');

        if (!selectBtn || !runBtn || !fileInput) return;

        let selectedFile = null;

        selectBtn.onclick = () => fileInput.click();

        fileInput.onchange = () => {
            selectedFile = fileInput.files[0] || null;
            if (selectedFile) {
                fileName.textContent = selectedFile.name;
                runBtn.disabled = false;
            } else {
                fileName.textContent = '';
                runBtn.disabled = true;
            }
        };

        runBtn.onclick = async () => {
            if (!selectedFile) return;
            runBtn.disabled = true;
            statusEl.textContent = '正在检测错别字...';
            statusEl.style.color = '#16a085';
            resultsPanel.style.display = 'none';

            try {
                const formData = new FormData();
                formData.append('file', selectedFile);
                if (diffMode && diffMode.checked) formData.append('diff_mode', 'true');

                const resp = await fetch('/check_typos', { method: 'POST', body: formData });
                const data = await resp.json();
                if (resp.ok) {
                    renderTypoResults(data, resultsPanel);
                    resultsPanel.style.display = 'block';
                    statusEl.textContent = '✓ 错别字检测完成 (' + data.total_suspects + '处疑似)';
                    statusEl.style.color = '#27ae60';
                } else {
                    statusEl.textContent = '✗ ' + (data.error || '检测失败');
                    statusEl.style.color = '#e74c3c';
                }
            } catch (err) {
                statusEl.textContent = '✗ 网络错误: ' + err.message;
                statusEl.style.color = '#e74c3c';
            } finally {
                runBtn.disabled = false;
            }
        };
    }

    function renderTypoResults(data, panel) {
        let html = '';
        html += `<p><strong>文档:</strong> ${escapeHtml(data.doc_name||'')} | <strong>疑似:</strong> ${data.total_suspects}处 | <strong>严重:</strong> ${data.critical_count}处 | <strong>检测层:</strong> ${(data.layers_run||[]).join(', ')}</p>`;

        if (data.findings && data.findings.length) {
            html += '<table style="width:100%;border-collapse:collapse;font-size:0.7rem;margin-top:8px;">';
            html += '<tr><th>层次</th><th>疑似文本</th><th>建议修正</th><th>置信度</th><th>严重性</th></tr>';
            data.findings.slice(0, 30).forEach(f => {
                const sevColor = f.severity === 'critical' ? '#e74c3c' : f.severity === 'warning' ? '#e67e22' : '#888';
                html += `<tr>`;
                html += `<td>${escapeHtml(f.layer||'')}</td>`;
                html += `<td><code style="font-size:0.68rem;">${escapeHtml((f.suspect_text||'').substring(0,30))}</code></td>`;
                html += `<td>${escapeHtml((f.suggestions||[]).join(', ').substring(0,40) || '—')}</td>`;
                html += `<td>${((f.confidence||0)*100).toFixed(0)}%</td>`;
                html += `<td style="color:${sevColor};font-weight:bold;">${escapeHtml(f.severity||'')}</td></tr>`;
                if (f.context_snippet) {
                    html += `<tr><td colspan="5" style="font-size:0.6rem;color:#888;padding-left:16px;">上下文: ${escapeHtml(f.context_snippet.substring(0,80))}</td></tr>`;
                }
                if (f.is_daxie_error) {
                    html += `<tr><td colspan="5" style="font-size:0.6rem;color:#e67e22;padding-left:16px;">⚠️ 大写金额不一致: 实际=${escapeHtml(f.daxie_actual||'')} 期望=${escapeHtml(f.daxie_expected||'无')}</td></tr>`;
                }
            });
            html += '</table>';
            if (data.findings.length > 30) html += `<p style="font-size:0.65rem;color:#888;">(仅显示前30项，共${data.findings.length}项)</p>`;
        } else {
            html += '<p style="color:#27ae60;font-size:0.72rem;">✓ 未发现疑似错别字</p>';
        }

        if (data.diff_text) {
            html += '<details style="margin-top:8px;"><summary style="cursor:pointer;font-weight:bold;font-size:0.72rem;">📋 差异审查 (修正预览)</summary>';
            html += `<div style="max-height:300px;overflow-y:auto;border:1px solid #ddd;padding:8px;margin-top:4px;font-size:0.7rem;line-height:1.6;white-space:pre-wrap;">${escapeHtml(data.diff_text.substring(0, 3000))}</div>`;
            html += '</details>';
        }

        panel.innerHTML = html;
    }

    // ══════════════════════════════════════════════════════════════════
    // Admin sidebar: result history viewers
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
                        <button id="resultHistoryClose" style="background:none;border:none;font-size:1.2rem;cursor:pointer;">✕</button>
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
            html += `<td>${r.same_rate_flag ? '⚠️' : '✓'}</td>`;
            html += `<td>${r.abnormal_drop_flag ? '⬇️' : '✓'}</td>`;
            html += `<td>${r.clustering_flag ? '🔗' : '✓'}</td>`;
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
    const _auditFunctionLabels = {
        rule_extraction: '规则提取',
        compliance_check: '合规审查',
        typo_detection: '错别字检测',
        quote_anomaly: '报价异常',
        relationship_extraction: '关系分析',
        ai_doc_review: 'AI文档审查',
        style_analysis: '文风分析',
    };

    // Severity threshold definitions per function: [{key, label, type, default}]
    const _auditThresholdDefs = {
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
                    msgEl.innerHTML = '<span style="color:#22c55e;">✅ ' + (d.message || '保存成功') + '</span>';
                    Object.keys(_dirty).forEach(k => delete _dirty[k]);
                    _updateDot();
                    loadAuditConfig();
                } else {
                    msgEl.innerHTML = '<span style="color:#ef4444;">❌ ' + (d.error || '保存失败') + '</span>';
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
    let _auditSSE = null;
    let _auditModalActive = false;

    async function showAuditModal() {
        const pid = window._currentProjectId || currentProjectId;
        if (!pid) {
            alert('请先在"项目管理"中打开一个项目，然后再使用全量审计功能。');
            return;
        }
        if (_auditModalActive) return;
        _auditModalActive = true;
        console.log('Audit: showAuditModal starting for project', pid);

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
                    statusEl.innerHTML = '<span style="color:#ef4444;">❌ ' + (pfData.error || '预检失败') + '</span>';
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
    console.log('Audit: window._showAuditModal set to', typeof showAuditModal);

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
