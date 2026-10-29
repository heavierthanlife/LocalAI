import os, json, re, html

workdir = "/home/guava5tar_/dsh/playground/casualFAQ"
cache_dir = os.path.join(workdir, "s1_thread_cache")

def clean_html(raw_html):
    # 1. Clean code blocks: convert Discuz ol/li to Markdown ``` ... ```
    def code_repl(m):
        ol_content = m.group(1)
        lines = re.findall(r'<li>(.*?)</li>', ol_content)
        clean_lines = [re.sub(r'<[^>]+>', '', l) for l in lines]
        return "\n```\n" + "\n".join(clean_lines) + "\n```\n"
    
    text = re.sub(r'<div class="blockcode">[\s\S]*?<ol>([\s\S]*?)</ol>[\s\S]*?复制代码</em>\s*</div>', code_repl, raw_html)
    
    # 2. Convert quote blocks: preserve quote context cleanly
    def quote_repl(m):
        quote_body = m.group(1).strip()
        quote_body = re.sub(r'<br\s*/?>', '\n> ', quote_body)
        quote_body = re.sub(r'<[^>]+>', '', quote_body)
        return "\n> [引用]\n> " + quote_body + "\n\n"
        
    text = re.sub(r'<div class="quote"><blockquote>([\s\S]*?)</blockquote></div>', quote_repl, text)
    
    # 3. Clean status headers (e.g. "本帖最后由 ... 于 ... 编辑")
    text = re.sub(r'<i class="pstatus">[\s\S]*?</i>', '', text)
    
    # 4. Filter smileys (static.stage1st.com/image/smiley/...)
    text = re.sub(r'<img [^>]*src="[^"]*smiley[^"]*"[^>]*>', '', text)
    
    # 5. Handle regular image attachments
    text = re.sub(r'<img [^>]*src="([^"]+)"[^>]*>', r'[图片: \1]', text)
    
    # 6. Line breaks
    text = re.sub(r'<br\s*/?>', '\n', text)
    
    # 7. Strip remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # 8. Decode entities and strip
    text = html.unescape(text).strip()
    return text

# Negative patterns (pure noise/flame/memes)
NOISE_EXACT = {
    'mark', 'Mark', 'MARK', '插眼', '火前留名', '前排', '支持', '好时代来临了', '好时代，来临力',
    '大的来了', '大的要来了', '赢麻了', '遥遥领先', '遥遥领先！', '乐', '草', '笑死', '确实',
    '蹲一个', '同问', '蹲', '蹲后续', '不明觉厉', '牛逼', '太牛了', 'nb', '666', '顶', 'mark一下',
    '来了来了', '留名', '占座', '先mark', '后排支持', '太真实了', '太强了'
}

def is_knowledge_post(post, clean_text):
    # Too short
    if len(clean_text) < 25 and not re.search(r'https?://[^\s]+', clean_text):
        return False, 0, "too_short"
    if clean_text in NOISE_EXACT:
        return False, 0, "noise_exact"

    # Filter out pure news headline copy-paste without user commentary
    lines = [l.strip() for l in clean_text.split('\n') if l.strip()]
    if len(lines) <= 2 and all(l.startswith('http') or l.startswith('[图片') or '快讯' in l or '发布' in l for l in lines) and len(clean_text) < 100:
        return False, 0, "pure_news"

    # Filter out pure one-sentence questions without answers/insights
    if len(clean_text) < 50 and ('?' in clean_text or '？' in clean_text) and not any(k in clean_text for k in ['方法', '解决', '踩坑', '发现', '教程', '配置', '其实']):
        return False, 0, "short_question"

    score = 0
    reasons = []

    # 1. AI 使用心得与工作流方法论
    workflow_kw = [
        '工作流', 'workflow', '提示词', 'system prompt', '全局规则', '全局纪律', '纪律', 
        '交接文档', 'handoff', 'grill-me', 'grill me', 'wayfinder', 'PRD', '需求拆解', 
        '分模块', '多轮迭代', '迭代', '会话管理', '新会话', '干净会话', '记忆', 'memory', 
        'RAG', '索引模式', '提示词缓存', 'prompt cache', 'prefill', '上下文', 'context',
        '知识库', 'Agent', 'agent', '智能体', '角色扮演', '审查', '评审', '闭环'
    ]
    for kw in workflow_kw:
        if kw.lower() in clean_text.lower():
            score += 3
            reasons.append(f"workflow:{kw}")

    # 2. 编程知识与工程落地
    coding_kw = [
        '代码', '编程', '重构', '静态检查', '静态工具', 'dap', 'linter', 'TDD', '测试用例', 
        '单元测试', '拼尸块', '文档法', '准代码块', '构建', 'compile', 'debug', 'trace',
        '接口', '架构', '原型', 'Spec', 'spec', '需求', '实现', '算法', '并发', '死锁'
    ]
    for kw in coding_kw:
        if kw.lower() in clean_text.lower():
            score += 2
            reasons.append(f"coding:{kw}")

    # 3. 插件、生态与工具推荐/配置
    tool_kw = [
        'harness', 'Harness', 'DSH', 'dsh', 'MCP', 'mcp', '插件', 'plugin', 'skill', 
        '技能', 'pwsh', 'powershell', 'bash', 'workbuddy', 'cursor', 'aider', 
        'chatbox', 'claude code', 'opencode', 'open-webui', 'libreoffice', 'whisper', 
        'funasr', 'asr', 'tts', 'stt', 'tailscale', 'ssh', 'tmux'
    ]
    for kw in tool_kw:
        if kw.lower() in clean_text.lower():
            score += 3
            reasons.append(f"tool:{kw}")

    # 4. API、路由、网络与风控避坑
    api_kw = [
        'API', 'api', 'OpenRouter', 'CommandCode', 'AgentRouter', 'Antigravity', 
        '风控', '封号', 'KYC', '家宽', '白名单', 'User-Agent', 'token', 'tokens', 
        '额度', '订阅', 'billing', 'proxy', '代理', '反代', 'DNS污染', '抓包'
    ]
    for kw in api_kw:
        if kw.lower() in clean_text.lower():
            score += 2
            reasons.append(f"api:{kw}")

    # Negative penalties for pure flame / petty arguments
    flame_kw = ['破防', '傻逼', '饭圈', '孝子', '急了', '小丑', '屁股', '吹jb', '水军', '营销']
    flame_count = sum(1 for kw in flame_kw if kw in clean_text)
    if flame_count >= 2 and score < 10:
        return False, score, "flame_war"

    # High quality indicators: code, structured text, in-depth explanation
    if '```' in clean_text:
        score += 5
        reasons.append("has_code")
    if len(clean_text) > 150:
        score += 3
    if len(clean_text) > 300:
        score += 4
    if any(k in clean_text for k in ['第一', '第二', '首先', '其次', '核心在于', '踩坑总结', '解决方案']):
        score += 3
        reasons.append("structured")

    # Threshold for knowledge
    if score >= 8:
        return True, score, ",".join(reasons)
    return False, score, "low_score"

def run_extraction():
    all_pages = sorted([int(re.search(r'page_(\d+)\.json', f).group(1)) 
                        for f in os.listdir(cache_dir) if f.startswith("page_") and f.endswith(".json")])
    print(f"Processing {len(all_pages)} cached pages...")
    
    selected_posts = []
    total_posts = 0
    for page in all_pages:
        filepath = os.path.join(cache_dir, f"page_{page}.json")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                posts = json.load(f)
        except Exception:
            continue
        
        for p in posts:
            total_posts += 1
            clean = clean_html(p.get('raw', ''))
            is_k, score, reason = is_knowledge_post(p, clean)
            if is_k:
                selected_posts.append({
                    'page': p['page'],
                    'floor': p['floor'],
                    'author': p['author'],
                    'date': p['date'],
                    'pid': p['pid'],
                    'clean_text': clean,
                    'score': score,
                    'reason': reason
                })

    print(f"Total posts scanned: {total_posts}")
    print(f"Selected knowledge posts: {len(selected_posts)}")
    
    out_json = os.path.join(workdir, "s1_knowledge_extracted.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(selected_posts, f, ensure_ascii=False, indent=2)
    print(f"Saved extracted posts to {out_json}")

if __name__ == "__main__":
    run_extraction()
