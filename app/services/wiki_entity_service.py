"""Wiki entity extraction service.

Pipeline:
  Upload → classify → extract entities via LLM → resolve against entity index
         → create/update entity pages → create wikilinks → persist entity index

The entity index is a lightweight JSON file at data/wiki/.entity_index.json.
Entity pages are stored as wiki markdown in entities/{type}/{name}.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

WIKI_DIR = os.path.join(DATA_DIR, "wiki")
ENTITIES_DIR = os.path.join(WIKI_DIR, "entities")
ENTITY_INDEX_PATH = os.path.join(WIKI_DIR, ".entity_index.json")
ENTITY_INDEX_LOCK_PATH = os.path.join(WIKI_DIR, ".entity_index.json.lock")

_ENTITY_INDEX_CACHE: Optional[dict] = None
_ENTITY_INDEX_CACHE_TIME: float = 0
_ENTITY_INDEX_CACHE_TTL: float = 30.0

_ENTITY_TYPE_DIRS = {
    "org": "organizations",
    "person": "people",
    "law": "legal_refs",
    "standard": "standards",
    "concept": "concepts",
    "project": "projects",
}


def _normalize_name(name: str) -> str:
    n = name.strip()
    n = n.replace("\u3000", " ").replace("\uff08", "(").replace("\uff09", ")")
    n = n.replace("\u300a", "").replace("\u300b", "")
    n = re.sub(r"[《》『』「」]", "", n)
    n = re.sub(r"\s+", " ", n)
    return n


def _hash_name(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def _entity_page_slug(entity_type: str, name: str) -> str:
    dir_name = _ENTITY_TYPE_DIRS.get(entity_type, "concepts")
    safe = re.sub(r'[<>:"/\\|?*]', "_", name.strip())[:80]
    safe = safe.strip("._")
    return f"entities/{dir_name}/{safe}.md"


def _ensure_entity_dirs():
    os.makedirs(ENTITIES_DIR, exist_ok=True)
    for d in _ENTITY_TYPE_DIRS.values():
        os.makedirs(os.path.join(ENTITIES_DIR, d), exist_ok=True)


def _load_entity_index() -> dict:
    global _ENTITY_INDEX_CACHE, _ENTITY_INDEX_CACHE_TIME
    now = time.time()
    if _ENTITY_INDEX_CACHE is not None and (now - _ENTITY_INDEX_CACHE_TIME) < _ENTITY_INDEX_CACHE_TTL:
        return _ENTITY_INDEX_CACHE
    if os.path.isfile(ENTITY_INDEX_PATH):
        try:
            with open(ENTITY_INDEX_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Corrupt entity index, reinitializing: {e}")
            data = {"version": 1, "entities": {}, "name_index": {}, "alias_index": {}}
    else:
        data = {"version": 1, "entities": {}, "name_index": {}, "alias_index": {}}
    _ENTITY_INDEX_CACHE = data
    _ENTITY_INDEX_CACHE_TIME = now
    return data


def _save_entity_index(index: dict):
    global _ENTITY_INDEX_CACHE, _ENTITY_INDEX_CACHE_TIME
    _ensure_entity_dirs()
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    index["version"] = index.get("version", 1) + 1
    try:
        from filelock import FileLock
        with FileLock(ENTITY_INDEX_LOCK_PATH, timeout=10):
            _do_save(index)
    except ImportError:
        _do_save(index)


def _do_save(index: dict):
    with open(ENTITY_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    _ENTITY_INDEX_CACHE = index
    _ENTITY_INDEX_CACHE_TIME = time.time()


def _extract_entities_regex(text: str) -> List[dict]:
    entities = []

    law_matches = re.findall(r"《([^》]{2,40})》", text)
    for name in law_matches:
        if not re.search(r"^\d+[年元]$", name):
            entities.append({"name": _normalize_name(name), "type": "law", "aliases": [], "context": "", "properties": {}})

    standard_matches = re.findall(
        r"(?:GB/T|GB|ISO|IEC|IEEE|ASTM|DIN|BS|EN|ASME|API|JGJ|CJJ|CJ|DL|SH|SY|NB|HG)\s*[\d.]+[\-\d.]*",
        text, re.IGNORECASE,
    )
    for name in standard_matches:
        entities.append({"name": name.strip(), "type": "standard", "aliases": [], "context": "", "properties": {}})

    org_matches = re.findall(
        r"(?:中华人民共和国)?(?:[\u4e00-\u9fff]{2,6}(?:部|委|局|处|署|院|会|司|办|厅|中心|总院|所|站|组|分行|支行))",
        text,
    )
    for name in org_matches:
        entities.append({"name": _normalize_name(name), "type": "org", "aliases": [], "context": "", "properties": {}})

    return entities


def extract_entities(text: str, filename: str = "", doc_type: str = "") -> List[dict]:
    if not text or len(text) < 50:
        return []

    from app.services import wiki_prompts
    from app.services.llm_provider import call_llm

    system_prompt = wiki_prompts.WIKI_ENTITY_EXTRACT_SYSTEM
    user_prompt = wiki_prompts.WIKI_ENTITY_EXTRACT_USER.format(
        doc_type=doc_type or "general",
        content=text[:8000],
        filename=filename,
    )

    try:
        raw = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=2048,
        )
    except Exception as e:
        logger.warning(f"LLM entity extract failed for {filename}: {e}")
        return _extract_entities_regex(text)

    entities = _parse_entity_response(raw, text)
    return entities


def _parse_entity_response(raw: str, fallback_text: str = "") -> List[dict]:
    try:
        parsed = json.loads(raw)
        entities = parsed.get("entities", [])
        if entities and isinstance(entities, list):
            return entities
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            trimmed = raw[start:end + 1]
            parsed = json.loads(trimmed)
            entities = parsed.get("entities", [])
            if entities and isinstance(entities, list):
                return entities
    except (json.JSONDecodeError, TypeError):
        pass

    return _extract_entities_regex(fallback_text)


def resolve_entity(entity: dict, existing_index: dict) -> Tuple[str, bool]:
    raw_name = entity.get("name", "")
    entity_type = entity.get("type", "concept")
    name = _normalize_name(raw_name)
    if not name:
        return _hash_name(raw_name or "<empty>"), False

    entity_id = _hash_name(name)
    aliases = entity.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []

    name_index = existing_index.get("name_index", {})
    alias_index = existing_index.get("alias_index", {})
    entities = existing_index.get("entities", {})

    if name in name_index:
        return name_index[name], True

    for alias in aliases:
        alias_norm = _normalize_name(alias)
        if alias_norm in alias_index:
            return alias_index[alias_norm], True
        if alias_norm in name_index:
            return name_index[alias_norm], True

    for existing_name, existing_eid in name_index.items():
        dist = _edit_distance(name, existing_name)
        if dist == 0:
            return existing_eid, True
        if dist <= 2 and _same_entity_type(entity_type, entities.get(existing_eid, {}).get("type", "")):
            return existing_eid, True

    return entity_id, False


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 10:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr[0] = i
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[len(b)]


def _same_entity_type(type_a: str, type_b: str) -> bool:
    if type_a == type_b:
        return True
    interchangeable = [
        {"law", "standard"},
        {"org", "person"},
    ]
    for group in interchangeable:
        if type_a in group and type_b in group:
            return True
    return False


def save_entity_page(entity_id: str, entity_data: dict, source_page_path: str):
    from app.services import wiki_engine

    entity_type = entity_data.get("type", "concept")
    name = _normalize_name(entity_data.get("name", ""))
    aliases = entity_data.get("aliases", [])
    wiki_path = _entity_page_slug(entity_type, name)

    in_fm, in_content, _ = wiki_engine.read_wiki_page(wiki_path)
    if not in_content:
        entity_fm = {
            "type": f"entity_{entity_type}",
            "entity_id": entity_id,
            "name": name,
            "aliases": aliases,
            "mentions": [],
        }
        entity_content = f"# {name}\n\n"
        entity_content += f"**类型**: {entity_type}\n\n"
        if aliases:
            entity_content += f"**别名**: {', '.join(aliases)}\n\n"
        entity_content += "## 相关文档\n\n"
    else:
        entity_fm = dict(in_fm or {})
        entity_content = in_content

    mention_line = f"- [[../{source_page_path}]] — {entity_data.get('context', '') or '相关'}"
    if mention_line not in entity_content:
        if "## 相关文档" in entity_content:
            entity_content += f"\n{mention_line}"
        else:
            entity_content += f"## 相关文档\n\n{mention_line}"

    existing_mentions = entity_fm.get("mentions", [])
    if not isinstance(existing_mentions, list):
        existing_mentions = []
    source_display = source_page_path.replace("\\", "/")
    if source_display not in [m.get("page", "") for m in existing_mentions]:
        existing_mentions.append({
            "page": source_display,
            "context": entity_data.get("context", ""),
            "source_file_id": entity_data.get("source_file_id", 0),
        })
    entity_fm["mentions"] = existing_mentions

    wiki_engine.write_wiki_page(wiki_path, entity_fm, entity_content)


def update_entity_index(entity_id: str, entity_data: dict):
    index = _load_entity_index()
    name = _normalize_name(entity_data.get("name", ""))
    entity_type = entity_data.get("type", "concept")
    aliases = entity_data.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []

    index.setdefault("entities", {})[entity_id] = {
        "id": entity_id,
        "type": entity_type,
        "name": name,
        "wiki_path": _entity_page_slug(entity_type, name),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    index.setdefault("name_index", {})[name] = entity_id
    for alias in aliases:
        alias_norm = _normalize_name(alias)
        if alias_norm and alias_norm != name:
            index.setdefault("alias_index", {})[alias_norm] = entity_id

    _save_entity_index(index)


def get_entity_page_path(entity_id: str) -> Optional[str]:
    index = _load_entity_index()
    entity = index.get("entities", {}).get(entity_id, {})
    return entity.get("wiki_path")


def get_entity_backlinks(entity_id: str) -> List[Dict]:
    index = _load_entity_index()
    entity = index.get("entities", {}).get(entity_id, {})
    if not entity:
        return []

    from app.services import wiki_engine
    wiki_path = entity.get("wiki_path", "")
    if not wiki_path:
        return []

    fm, content, _ = wiki_engine.read_wiki_page(wiki_path)
    if not content:
        return []

    mentions = fm.get("mentions", []) if fm else []
    if not isinstance(mentions, list):
        mentions = []

    backlinks = []
    for m in mentions:
        backlinks.append({
            "page": m.get("page", ""),
            "context": m.get("context", ""),
            "entity_id": entity_id,
            "entity_name": entity.get("name", ""),
        })
    return backlinks


def process_upload_entity_extraction(
    file_id: int,
    text_content: str,
    filename: str,
    source_type: str,
    doc_type: str,
    wiki_category: str,
    metadata: dict = None,
) -> Dict:
    if not text_content or len(text_content) < 50:
        return {"entities_found": 0, "entities_new": 0, "entities_matched": 0}

    entities = extract_entities(text_content, filename, doc_type)
    if not entities:
        return {"entities_found": 0, "entities_new": 0, "entities_matched": 0}

    index = _load_entity_index()
    safe_fname = re.sub(r'[^\w\u4e00-\u9fff.-]', '_', filename)[:80]
    wiki_page_path = f"{wiki_category}/{safe_fname}.md"

    new_count = 0
    matched_count = 0

    for ent in entities:
        ent.setdefault("source_file_id", file_id)
        entity_id, is_existing = resolve_entity(ent, index)
        if is_existing:
            matched_count += 1
        else:
            new_count += 1
        save_entity_page(entity_id, ent, wiki_page_path)
        update_entity_index(entity_id, ent)

    logger.info(
        f"Entity extraction complete for {source_type}.{file_id}: "
        f"{len(entities)} total, {matched_count} matched, {new_count} new"
    )
    return {
        "entities_found": len(entities),
        "entities_new": new_count,
        "entities_matched": matched_count,
        "wiki_page_path": wiki_page_path,
    }


def remove_entity_backlink(source_page_path: str):
    index = _load_entity_index()
    name_index = index.get("name_index", {})
    entities = index.get("entities", {})

    from app.services import wiki_engine

    source_path = source_page_path.replace("\\", "/")
    for entity_id, ent_data in entities.items():
        wiki_path = ent_data.get("wiki_path", "")
        if not wiki_path:
            continue
        fm, content, _ = wiki_engine.read_wiki_page(wiki_path)
        if not fm or not content:
            continue
        mentions = fm.get("mentions", [])
        if not isinstance(mentions, list):
            continue
        updated = [m for m in mentions if m.get("page", "") != source_path]
        if len(updated) != len(mentions):
            fm["mentions"] = updated
            new_content = re.sub(
                rf"- \[\[.*?{re.escape(source_path)}.*?\]\].*?\n?", "", content
            )

            wiki_engine.write_wiki_page(wiki_path, fm, new_content)


def compare_wiki_pages(page_a_path: str, page_b_path: str) -> Dict:
    from app.services import wiki_engine

    fm_a, content_a, path_a = wiki_engine.read_wiki_page(page_a_path)
    fm_b, content_b, path_b = wiki_engine.read_wiki_page(page_b_path)

    if not content_a or not content_b:
        return {"error": "One or both pages not found"}

    title_a = (fm_a or {}).get("title", page_a_path) if fm_a else page_a_path
    title_b = (fm_b or {}).get("title", page_b_path) if fm_b else page_b_path
    type_a = (fm_a or {}).get("type", "") if fm_a else ""
    type_b = (fm_b or {}).get("type", "") if fm_b else ""

    entities_a = _extract_wikilinks(content_a)
    entities_b = _extract_wikilinks(content_b)

    set_a = set(_normalize_name(e) for e in entities_a)
    set_b = set(_normalize_name(e) for e in entities_b)
    shared = list(set_a & set_b)
    a_only = list(set_a - set_b)
    b_only = list(set_b - set_a)

    from app.services.file_processing import compute_similarity_with_numbers
    try:
        sim, _, _, _ = compute_similarity_with_numbers(content_a[:5000], content_b[:5000])
    except Exception:
        sim = 0.0

    comparison_fm = {
        "type": "comparison",
        "page_a": page_a_path,
        "page_b": page_b_path,
        "similarity": round(sim, 3),
        "shared_entities": len(shared),
        "a_only_entities": len(a_only),
        "b_only_entities": len(b_only),
        "compared_at": datetime.now(timezone.utc).isoformat(),
    }

    slug_a = re.sub(r'[^\w\u4e00-\u9fff]', '_', page_a_path.replace("/", "_"))[:30]
    slug_b = re.sub(r'[^\w\u4e00-\u9fff]', '_', page_b_path.replace("/", "_"))[:30]
    comp_path = f"comparisons/{slug_a}_vs_{slug_b}.md"

    comp_content = f"# 对比: {title_a} vs {title_b}\n\n"
    comp_content += f"**相似度**: {sim:.1%}\n\n"

    comp_content += f"### 共同实体 ({len(shared)})\n"
    for e in shared[:20]:
        comp_content += f"- {e}\n"
    if len(shared) > 20:
        comp_content += f"\n...及其他 {len(shared) - 20} 个\n"

    comp_content += f"\n### 「{title_a}」独有 ({len(a_only)})\n"
    for e in a_only[:10]:
        comp_content += f"- {e}\n"

    comp_content += f"\n### 「{title_b}」独有 ({len(b_only)})\n"
    for e in b_only[:10]:
        comp_content += f"- {e}\n"

    comp_content += "\n\n---\n*此比较由系统自动生成。如源页面更新，请重新生成以获取最新对比结果。*"

    wiki_engine.write_wiki_page(comp_path, comparison_fm, comp_content)

    return {
        "page_a": {"title": title_a, "path": page_a_path, "type": type_a},
        "page_b": {"title": title_b, "path": page_b_path, "type": type_b},
        "entities": {"shared": shared, "a_only": a_only, "b_only": b_only},
        "similarity": round(sim, 3),
        "comparison_path": comp_path,
    }


def _extract_wikilinks(content: str) -> List[str]:
    import re
    links = re.findall(r"\[\[([^\]]+)\]\]", content)
    result = []
    for link in links:
        parts = link.split("|")
        name = parts[-1].strip()
        full = parts[0].strip()
        if "/" in full:
            result.append(full)
        elif name:
            result.append(name)
    return result


def _invalidate_comparisons_for(page_path: str):
    from app.services import wiki_engine

    comp_dir = os.path.join(WIKI_DIR, "comparisons")
    if not os.path.isdir(comp_dir):
        return

    page_path_normalized = page_path.replace("\\", "/")
    for fname in os.listdir(comp_dir):
        if not fname.endswith(".md"):
            continue
        comp_page = f"comparisons/{fname}"
        fm, content, _ = wiki_engine.read_wiki_page(comp_page)
        if not fm:
            continue
        a = (fm.get("page_a") or "").replace("\\", "/")
        b = (fm.get("page_b") or "").replace("\\", "/")
        if a == page_path_normalized or b == page_path_normalized:
            fm["stale"] = True
            wiki_engine.write_wiki_page(comp_page, fm, content)


def get_entity_graph(root_entity_path: str, depth: int = 2, max_nodes: int = 50) -> Dict:
    from app.services import wiki_engine

    nodes = {}
    edges = {}

    initial_path = root_entity_path
    if not initial_path.endswith(".md"):
        initial_path = initial_path + ".md"

    fm, content, _ = wiki_engine.read_wiki_page(initial_path)
    if not content:
        return {"nodes": [], "edges": []}

    root_id = hashlib.sha256(root_entity_path.encode()).hexdigest()[:12]
    root_name = (fm or {}).get("name", root_entity_path) if fm else root_entity_path

    nodes[root_id] = {
        "id": root_id,
        "label": _normalize_name(str(root_name)),
        "type": "entity",
        "path": root_entity_path,
    }

    queue = [(root_entity_path, root_id, 0)]
    visited = {root_entity_path}
    node_counter = 1
    collapsed_count = 0

    while queue and len(nodes) < max_nodes:
        current_path, parent_id, current_depth = queue.pop(0)
        if current_depth >= depth:
            continue

        backlinks = []
        if current_path.startswith("entities/"):
            fm_curr, _, _ = wiki_engine.read_wiki_page(current_path)
            mentions = (fm_curr or {}).get("mentions", [])
            if isinstance(mentions, list):
                for m in mentions:
                    page = m.get("page", "")
                    if page and page not in visited:
                        backlinks.append((page, m.get("context", "")))

        _, content_curr, _ = wiki_engine.read_wiki_page(current_path)
        if content_curr:
            wikilinks = re.findall(r"\[\[([^\]]+)\]\]", content_curr)
            for link in wikilinks:
                clean = link.split("|")[0].strip()
                if clean.startswith("entities/") or clean.startswith("../"):
                    clean = re.sub(r"^(\.\./)+", "", clean)
                if clean and clean not in visited and clean not in [bl[0] for bl in backlinks]:
                    backlinks.append((clean, ""))

        for page, context in backlinks:
            if page in visited:
                continue
            if len(nodes) >= max_nodes:
                collapsed_count += 1
                continue

            node_id = hashlib.sha256(page.encode()).hexdigest()[:12]
            fm_p, _, _ = wiki_engine.read_wiki_page(page)
            node_label = (fm_p or {}).get("title", os.path.basename(page).replace(".md", "")) if fm_p else os.path.basename(page).replace(".md", "")

            is_entity = page.startswith("entities/")
            node_type = "entity" if is_entity else "document"

            nodes[node_id] = {
                "id": node_id,
                "label": _normalize_name(str(node_label)),
                "type": node_type,
                "path": page,
            }

            edge_id = f"{parent_id}_{node_id}"
            edges[edge_id] = {
                "source": parent_id,
                "target": node_id,
                "label": context or ("contains" if is_entity else "mentions"),
            }

            visited.add(page)
            node_counter += 1

            if is_entity and current_depth + 1 < depth:
                queue.append((page, node_id, current_depth + 1))

    if collapsed_count > 0:
        collapsed_id = "__collapsed"
        nodes[collapsed_id] = {
            "id": collapsed_id,
            "label": f"+{collapsed_count} more",
            "type": "collapsed",
        }
        parent_for_collapse = root_id
        if nodes:
            parent_for_collapse = list(nodes.keys())[0]
        edge_key = f"{parent_for_collapse}_{collapsed_id}"
        edges[edge_key] = {"source": parent_for_collapse, "target": collapsed_id, "label": "..."}

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def list_category_pages(wiki_category: str) -> List[Dict]:
    from app.services import wiki_engine

    cat_dir = os.path.join(WIKI_DIR, wiki_category)
    if not os.path.isdir(cat_dir):
        return []

    pages = []
    for fname in sorted(os.listdir(cat_dir)):
        if not fname.endswith(".md") or fname == "index.md":
            continue
        fp = f"{wiki_category}/{fname}"
        fm, content, _ = wiki_engine.read_wiki_page(fp)
        title = (fm or {}).get("title", fname[:-3]) if fm else fname[:-3]
        pages.append({
            "path": fp,
            "title": title,
            "type": (fm or {}).get("type", "") if fm else "",
        })
    return pages
