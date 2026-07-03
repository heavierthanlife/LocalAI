"""KB Skill Engine — auto-generate structured skill summaries from uploaded documents.
   Mimics book-to-skill logic: extracts frameworks, principles, techniques, anti-patterns.
   Stores skill as markdown alongside the original file in the knowledge base."""

import re, logging, os, json
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Pattern-based extraction (no external deps) ──

HEADING_PATTERNS = [
    (re.compile(r'^#+\s+(.+)', re.MULTILINE), 'concept'),
    (re.compile(r'^[一二三四五六七八九十]+[、．.]\s*(.+)', re.MULTILINE), 'principle'),
    (re.compile(r'^\d+[\.\)、]\s*(.+)', re.MULTILINE), 'technique'),
    (re.compile(r'^[-•▪▸►]\s*(.+)', re.MULTILINE), 'technique'),
]

FRAMEWORK_SIGNALS = [
    r'(?:框架|framework|model|methodology|方法论|体系|流程|pipeline|架构)\b',
    r'(?:步骤|step|stage|phase|环节|阶段)\s*[：:]\s*\d+',
    r'(?:原则|principle|rule|规则|guideline|准则)\s*[：:]\s*',
    r'\b(?:first|second|third|第一|第二|第三|首先|其次|然后|最后)\b',
    r'(?:核心|key|critical|关键|重要)\s*(?:概念|concept|要素|factor|指标|metric)',
]

PRINCIPLE_SIGNALS = [
    r'(?:原则|principle|rule|规则|guideline|准则|标准|standard)\s*\d*[：:]',
    r'(?:必须|must|shall|should|应当|需要|required)\s*[^，。]*[。；]',
    r'(?:禁止|never|don\'t|avoid|避免|不能|不可)\s*[^，。]*[。；]',
    r'(?:最佳实践|best\s*practice|推荐|recommend)\s*[：:]',
]

TECHNIQUE_SIGNALS = [
    r'(?:方法|method|technique|技术|approach|方案|做法)\s*[：:]',
    r'(?:如何|how\s*to|步骤|step)\s*[：:]',
    r'\b(?:使用|using|通过|via|via)\s+\w+\s*(?:方法|method|方式|technique)',
    r'(?:工具|tool|框架|framework|软件|software)\s*[：:]',
]

ANTIPATTERN_SIGNALS = [
    r'(?:误区|pitfall|陷阱|trap|常见错误|common\s*mistake)',
    r'(?:不要|don\'t|avoid|避免|warning|注意|caution)',
    r'(?:错误|error|失败|failure|问题|issue)\s*(?:原因|cause|根源|root)',
]


def extract_frameworks(text: str) -> list:
    """Extract conceptual frameworks from document text."""
    results = []
    lines = text.split('\n')
    in_framework = False
    framework_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if in_framework and len(framework_lines) >= 2:
                results.append(' '.join(framework_lines))
                framework_lines = []
                in_framework = False
            continue
        
        for pattern in FRAMEWORK_SIGNALS:
            if re.search(pattern, line_stripped, re.IGNORECASE):
                in_framework = True
                framework_lines.append(line_stripped)
                break
        else:
            if in_framework:
                framework_lines.append(line_stripped)
    
    # Flush remaining
    if in_framework and len(framework_lines) >= 2:
        results.append(' '.join(framework_lines))
    
    return results[:8]  # Max 8 frameworks


def extract_principles(text: str) -> list:
    """Extract principles and rules."""
    results = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) < 10:
            continue
        for pattern in PRINCIPLE_SIGNALS:
            if re.search(pattern, line, re.IGNORECASE):
                clean = re.sub(r'^[#\-\*•\d\.\)、\s]+', '', line)
                if len(clean) > 8:
                    results.append(clean)
                break
    return results[:12]


def extract_techniques(text: str) -> list:
    """Extract techniques and methods."""
    results = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) < 10:
            continue
        for pattern in TECHNIQUE_SIGNALS:
            if re.search(pattern, line, re.IGNORECASE):
                clean = re.sub(r'^[#\-\*•\d\.\)、\s]+', '', line)
                if len(clean) > 8:
                    results.append(clean)
                break
    return results[:12]


def extract_antipatterns(text: str) -> list:
    """Extract warnings and anti-patterns."""
    results = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) < 10:
            continue
        for pattern in ANTIPATTERN_SIGNALS:
            if re.search(pattern, line, re.IGNORECASE):
                clean = re.sub(r'^[#\-\*•\d\.\)、\s]+', '', line)
                if len(clean) > 8:
                    results.append(clean)
                break
    return results[:8]


def extract_key_concepts(text: str) -> list:
    """Extract key terms and concepts (capitalized, bold, or heading-style)."""
    concepts = set()
    # Bold/strong text patterns
    for m in re.finditer(r'\*\*(.+?)\*\*|__(.+?)__', text):
        term = (m.group(1) or m.group(2)).strip()
        if 3 <= len(term) <= 60:
            concepts.add(term)
    # Heading patterns
    for m in re.finditer(r'^#{1,3}\s+(.+)$', text, re.MULTILINE):
        term = m.group(1).strip()
        if 5 <= len(term) <= 80:
            concepts.add(term)
    # Chinese bookmarks: 【...】
    for m in re.finditer(r'【(.+?)】', text):
        concepts.add(m.group(1).strip())
    return list(concepts)[:15]


def extract_definitions(text: str) -> list:
    """Extract term definitions (term: definition or term is definition patterns)."""
    results = []
    for m in re.finditer(r'([^。；\n]{3,40})[：:]\s*([^。\n]{10,200})', text):
        term = m.group(1).strip()
        definition = m.group(2).strip()
        if any(kw in term.lower() for kw in ['定义','是','指','即','为','definition','refers','means']):
            results.append(f"**{term}**：{definition}")
    for m in re.finditer(r'([^。；\n]{3,30})\s*(?:是|是指|即|指的是|定义[为是])\s*([^。\n]{10,200})', text):
        term = m.group(1).strip()
        definition = m.group(2).strip()
        results.append(f"**{term}**：{definition}")
    return results[:10]


def extract_checkable_steps(text: str) -> list:
    """Extract numbered steps / checklists."""
    steps = []
    for m in re.finditer(r'(?:步骤|Step|Stage|Phase)\s*\d+[：:\s]*(.+?)(?=\s*(?:步骤|Step|Stage|Phase)\s*\d+|$)', text, re.IGNORECASE):
        steps.append(m.group(1).strip()[:200])
    if not steps:
        step_lines = re.findall(r'^\s*\d+[\.\)、]\s*(.{15,200})$', text, re.MULTILINE)
        if len(step_lines) >= 3:
            steps = step_lines[:10]
    return steps


def generate_skill_summary(text: str, filename: str, source_type: str = "knowledge_lab") -> str:
    """Generate a structured skill summary from document text.
    
    Returns a markdown string following book-to-skill format.
    """
    if not text or len(text) < 50:
        return ""
    
    frameworks = extract_frameworks(text)
    principles = extract_principles(text)
    techniques = extract_techniques(text)
    antipatterns = extract_antipatterns(text)
    concepts = extract_key_concepts(text)
    definitions = extract_definitions(text)
    steps = extract_checkable_steps(text)
    
    # Only generate if we found meaningful content
    total_items = len(frameworks) + len(principles) + len(techniques) + len(antipatterns)
    if total_items < 2:
        return ""
    
    name = os.path.splitext(filename)[0]
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    sections = []
    sections.append(f"# {name}")
    sections.append(f"> Auto-generated skill summary from: {filename}")
    sections.append(f"> Source: {source_type} | Generated: {date}")
    sections.append("")
    
    if concepts:
        sections.append("## 📌 核心概念")
        for c in concepts:
            sections.append(f"- {c}")
        sections.append("")
    
    if definitions:
        sections.append("## 📖 定义与术语")
        for d in definitions:
            sections.append(f"- {d}")
        sections.append("")
    
    if frameworks:
        sections.append("## 🏗️ 框架与模型")
        for f in frameworks:
            sections.append(f"- {f}")
        sections.append("")
    
    if principles:
        sections.append("## ⚖️ 原则与规则")
        for p in principles:
            sections.append(f"- {p}")
        sections.append("")
    
    if techniques:
        sections.append("## 🔧 方法与技巧")
        for t in techniques:
            sections.append(f"- {t}")
        sections.append("")
    
    if steps:
        sections.append("## ✅ 可执行步骤")
        for i, s in enumerate(steps, 1):
            sections.append(f"{i}. {s}")
        sections.append("")
    
    if antipatterns:
        sections.append("## ⚠️ 常见陷阱")
        for a in antipatterns:
            sections.append(f"- {a}")
        sections.append("")
    
    sections.append("---")
    sections.append(f"*Auto-generated by ZLAI services KB Skill Engine | {date}*")
    
    return '\n'.join(sections)


def generate_skill_for_file(file_content: str, filename: str, source: str = "knowledge_lab") -> str | None:
    """Main entry point: generate a skill summary for an uploaded file.
    
    Returns the skill markdown string, or None if not enough content to generate.
    """
    try:
        skill = generate_skill_summary(file_content, filename, source)
        if skill and len(skill) > 100:
            logger.info(f"Skill generated for {filename}: {len(skill)} chars, "
                        f"{skill.count('##')} sections")
            return skill
        else:
            logger.debug(f"Insufficient content for skill from {filename}")
            return None
    except Exception as e:
        logger.error(f"Skill generation failed for {filename}: {e}")
        return None
