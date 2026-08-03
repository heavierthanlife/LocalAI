/* Chat System module - extracted from app.js (W4) */
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

    var _loadingHistory = false;
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
    var _pollTimer = null;
    var _pollLastId = 0;

    function startRealtimePoll(projectId) {
        stopRealtimePoll();
        _pollLastId = _lastKnownMessageId || 0;
        var _backoff = 0;
        const BASE = projectId ? 3000 : 5000;

        function _scheduleNext() {
            var delay = Math.min(BASE * Math.pow(1.3, Math.min(_backoff, 12)), 30000);
            _pollTimer = setTimeout(_poll, delay);
        }

        async function _poll() {
            const currentThread = sessionStorage.getItem('currentThreadId');
            if (!currentThread || isProcessing) { _scheduleNext(); return; }
            if (projectId && currentProjectId != projectId) { _scheduleNext(); return; }
            try {
                const res = await fetch(`/chat/poll/${currentThread}?since_id=${_pollLastId}`, { credentials: 'include' });
                if (res.status === 404) { stopRealtimePoll(); return; }
                if (!res.ok) { _backoff++; _scheduleNext(); return; }
                const data = await res.json();
                if (!data.success) { _backoff++; _scheduleNext(); return; }
                const newMsgs = data.messages || [];
                if (projectId) loadProjectVotes(projectId);
                if (newMsgs.length > 0) {
                    _backoff = 0;
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
                } else {
                    _backoff++;
                }
            } catch(e) { /* silent */ }
            _scheduleNext();
        }

        _pollTimer = setTimeout(_poll, BASE);
    }

    function stopRealtimePoll() {
        if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
    }

    var _lastKnownMessageId = 0;
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
    var _readMarkTimer = null;
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

