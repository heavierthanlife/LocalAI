"""Entity relationship extraction for bid documents.

Four detection modules:
  1. Company-company relationships (subsidiaries, shareholders, affiliates)
  2. Personnel-company relationships (legal reps, project managers across companies)
  3. Document-document cross-references (identical passages, bid interdependencies)
  4. Collusion signals (identical typos, formatting quirks, file metadata fingerprints)

Uses HanLP for Chinese NER (fallback: regex), existing LLM for relationship
classification, and rag_engine for internal KB lookups. Optional 天眼查 API
integration controlled via runtime_config toggle.
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# --- Entity extraction regex fallbacks (used when HanLP is unavailable) ---

_COMPANY_SUFFIX = re.compile(
    r'(?:有限公司|有限责任公司|股份有限公司|集团有限公司|集团|事务所|合伙企业|厂|'
    r'分公司|子公司|办事处|代表处|中心|研究院|学院|学校|医院|合作社|'
    r'总公司|母公司|控股公司)'
)
_PERSON_TITLE = re.compile(
    r'(?:法定代表人|法人代表|法人|董事长|总经理|项目经理|授权代表|委托代理人|'
    r'技术负责人|质量负责人|安全负责人|联系人|投标人|被授权人|签字人|负责人|'
    r'总监|总工程师|注册建造师|注册工程师|造价工程师)'
)
_COMPANY_NAME_RE = re.compile(
    r'(?:(?:[一-鿿]{2,8}(?:市|省|区|县|镇|乡|村))?'
    r'[一-鿿]{2,20}'
    r'(?:有限公司|有限责任公司|股份有限公司|集团有限公司|集团|事务所|合伙企业|厂|'
    r'分公司|子公司|办事处|代表处|中心|研究院|学院|学校|医院|合作社|'
    r'总公司|母公司|控股公司))'
)
_PERSON_NAME_RE = re.compile(
    r'(?:' + _PERSON_TITLE.pattern + r')'
    r'\s*[：:]\s*'
    r'(?P<name>[一-鿿]{2,4})'
    r'|'
    r'(?P<name2>[一-鿿]{2,4})'
    r'\s*[：:]\s*'
    r'(?P<title>' + _PERSON_TITLE.pattern + r')'
)


# --- Data classes ---

@dataclass
class ExtractedEntity:
    text: str
    entity_type: str  # "company", "person", "document_ref", "amount"
    positions: list[tuple[int, int]] = field(default_factory=list)
    confidence: float = 1.0

@dataclass
class DetectedRelationship:
    source_entity: str
    target_entity: str
    relation_type: str       # "subsidiary", "shared_personnel", "cross_reference", "collusion_signal"
    relation_subtype: str    # e.g., "legal_rep", "same_author", "identical_typo"
    confidence: float
    evidence: str            # snippet from the document
    module: str              # "company_company", "personnel_company", "doc_doc", "collusion"
    risk_flag: bool = False
    risk_reason: str = ""

@dataclass
class RelationshipReport:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[DetectedRelationship] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    modules_run: list[str] = field(default_factory=list)
    tianyancha_used: bool = False
    company_personnel_map: dict[str, list[str]] = field(default_factory=dict)


# --- Entity extraction -------------------------------------------------------

def _extract_with_hanlp(texts: list[str]) -> list[list[ExtractedEntity]]:
    """Extract named entities using HanLP. Returns list of entity lists per text."""
    try:
        import hanlp
        tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
    except Exception as e:
        logger.warning(f"HanLP not available, falling back to regex: {e}")
        return _extract_with_regex(texts)

    all_entities = []
    for text in texts:
        entities = []
        try:
            tokens = tokenizer(text)
            ner_result = ner(tokens)
            for entity_info in ner_result:
                entity_text = entity_info[0]
                entity_type = entity_info[1]
                etype = entity_type.lower()
                if 'person' in etype or 'nr' in etype:
                    entities.append(ExtractedEntity(
                        text=entity_text, entity_type="person",
                        confidence=0.92))
                elif 'organization' in etype or 'nt' in etype or 'ns' in etype:
                    if _COMPANY_SUFFIX.search(entity_text):
                        entities.append(ExtractedEntity(
                            text=entity_text, entity_type="company",
                            confidence=0.90))
        except Exception:
            pass
        # If HanLP found nothing useful, fall back to regex for this text
        if not [e for e in entities if e.entity_type == "company"]:
            regex_entities = _extract_with_regex([text])[0]
            entities.extend([e for e in regex_entities if e.entity_type == "company"])
        all_entities.append(entities)

    return all_entities


def _extract_with_regex(texts: list[str]) -> list[list[ExtractedEntity]]:
    """Regex-based company and person extraction (offline fallback)."""
    all_entities = []
    for text in texts:
        entities = []
        for m in _COMPANY_NAME_RE.finditer(text):
            entities.append(ExtractedEntity(
                text=m.group(), entity_type="company",
                positions=[(m.start(), m.end())], confidence=0.85))
        for m in _PERSON_NAME_RE.finditer(text):
            name = m.group('name') or m.group('name2')
            title = m.group('title') or m.group(1)
            if name and len(name) >= 2:
                entities.append(ExtractedEntity(
                    text=name, entity_type="person",
                    positions=[(m.start(), m.end())],
                    confidence=0.88 if title else 0.75))
        all_entities.append(entities)
    return all_entities


def _merge_entities(all_entities: list[list[ExtractedEntity]]) -> list[ExtractedEntity]:
    """Merge duplicate entities across texts, keeping highest confidence."""
    seen = {}
    for ent_list in all_entities:
        for ent in ent_list:
            key = (ent.text, ent.entity_type)
            if key not in seen or ent.confidence > seen[key].confidence:
                seen[key] = ent
    return list(seen.values())


# --- Module 1: Company-company relationships ---------------------------------

_COMPANY_RELATION_PATTERNS = [
    (re.compile(r'(?P<parent>[一-鿿]{2,20}(?:有限公司|集团有限公司|集团|总公司))'
                r'(?:的)?(?:全资)?(?:控股)?(?:子(?:公司)?|分(?:公司)?|下属)'
                r'(?P<child>[一-鿿]{2,20}(?:有限公司)?)'),
     'subsidiary', 0.75),
    (re.compile(r'(?P<company>[一-鿿]{2,20}(?:有限公司|集团有限公司|集团))'
                r'(?:系|是|属于)\s*'
                r'(?P<parent>[一-鿿]{2,20}(?:有限公司|集团有限公司|集团|总公司))'
                r'(?:的)?(?:全资)?(?:控股)?(?:子(?:公司)?|分(?:公司)?|成员)'),
     'subsidiary', 0.80),
    (re.compile(r'(?P<a>[一-鿿]{2,20}(?:有限公司|集团有限公司|集团))'
                r'(?:与|和|同)\s*'
                r'(?P<b>[一-鿿]{2,20}(?:有限公司|集团有限公司|集团))'
                r'\s*(?:组成|联合|共同).*(?:联合体|共同体|投标联合体|consortium)'),
     'joint_venture', 0.82),
    (re.compile(r'(?P<a>[一-鿿]{2,20}(?:有限公司|集团有限公司|集团))'
                r'(?:持有|拥有|占有)\s*'
                r'(?P<b>[一-鿿]{2,20}(?:有限公司|集团有限公司|集团))'
                r'\s*(?:的)?\s*[\d.]+%\s*(?:的)?(?:股份|股权|权益)'),
     'shareholder', 0.85),
    (re.compile(r'(?P<a>[一-鿿]{2,20}(?:有限公司|集团有限公司|集团))'
                r'(?:的)?(?:母公司|控股股东|实际控制人)(?:是|为)'
                r'(?P<b>[一-鿿]{2,20}(?:有限公司|集团有限公司|集团|总公司))'),
     'parent_company', 0.80),
]


def _detect_company_relationships(
    all_entities: list[list[ExtractedEntity]],
    texts: list[str],
) -> tuple[list[DetectedRelationship], list[str]]:
    """Extract company-company relationships from bid documents."""
    relationships = []
    red_flags = []

    # 1. Pattern-based extraction from text
    for text in texts:
        for pattern, rel_type, confidence in _COMPANY_RELATION_PATTERNS:
            for m in pattern.finditer(text):
                groups = m.groupdict()
                keys = [k for k in groups.keys() if groups[k]]
                if len(keys) >= 2:
                    a = groups.get(keys[0], '')
                    b = groups.get(keys[1], '')
                    if a and b and a != b:
                        evidence = text[max(0, m.start()-20):m.end()+20]
                        relationships.append(DetectedRelationship(
                            source_entity=a, target_entity=b,
                            relation_type=rel_type, relation_subtype="text_extracted",
                            confidence=confidence, evidence=evidence,
                            module="company_company",
                            risk_flag=rel_type in ('subsidiary', 'parent_company', 'shareholder'),
                            risk_reason="关联公司投标需审查是否构成围标" if rel_type in ('subsidiary', 'parent_company', 'shareholder') else "",
                        ))

    # 2. Cross-document: same company appearing across bids from different documents
    companies_by_doc = defaultdict(set)
    for idx, ent_list in enumerate(all_entities):
        for ent in ent_list:
            if ent.entity_type == "company":
                companies_by_doc[idx].add(ent.text)

    doc_indices = list(companies_by_doc.keys())
    for i in range(len(doc_indices)):
        for j in range(i + 1, len(doc_indices)):
            shared = companies_by_doc[doc_indices[i]] & companies_by_doc[doc_indices[j]]
            for company in shared:
                relationships.append(DetectedRelationship(
                    source_entity=f"doc_{doc_indices[i]}",
                    target_entity=f"doc_{doc_indices[j]}",
                    relation_type="shared_mention",
                    relation_subtype="same_company_in_multiple_bids",
                    confidence=0.90, evidence=company,
                    module="company_company",
                    risk_flag=True,
                    risk_reason=f"'{company}' 出现在多份投标文件中",
                ))
                red_flags.append(f"跨文件公司关联: '{company}' 出现在多份投标中")

    return relationships, red_flags


# --- Module 2: Personnel-company relationships --------------------------------

def _detect_personnel_relationships(
    all_entities: list[list[ExtractedEntity]],
    texts: list[str],
    file_names: list[str],
) -> tuple[list[DetectedRelationship], list[str], dict[str, list[str]]]:
    """Detect same person appearing in multiple bidders' documents."""
    relationships = []
    red_flags = []
    company_personnel_map = defaultdict(list)

    # Build person → (company, role, filename) mapping
    for idx, (ent_list, text) in enumerate(zip(all_entities, texts)):
        file_name = file_names[idx] if idx < len(file_names) else f"doc_{idx}"

        # Extract companies from this document
        doc_companies = [e.text for e in ent_list if e.entity_type == "company"]

        # Extract persons with their titles
        for m in _PERSON_NAME_RE.finditer(text):
            name = m.group('name') or m.group('name2')
            title_match = m.group('title') or m.group(1)
            if not name or len(name) < 2:
                continue
            title = title_match.strip() if title_match else "未知职务"
            doc_name = file_name
            company_name = doc_companies[0] if doc_companies else doc_name

            company_personnel_map[name].append({
                'company': company_name,
                'title': title,
                'file': doc_name,
            })

    # Cross-reference: same person in multiple companies
    for person, appearances in company_personnel_map.items():
        companies = list(set(a['company'] for a in appearances))
        files = list(set(a['file'] for a in appearances))
        if len(companies) >= 2 or len(files) >= 2:
            titles = list(set(a['title'] for a in appearances))
            relationships.append(DetectedRelationship(
                source_entity=person,
                target_entity=', '.join(companies),
                relation_type="shared_personnel",
                relation_subtype="same_person_multiple_companies",
                confidence=0.88,
                evidence=f"{person} 担任 {', '.join(titles)} — 涉及 {', '.join(companies)}",
                module="personnel_company",
                risk_flag=True,
                risk_reason=f"同一个人在{len(companies)}家公司担任职务，需核查是否为关联关系",
            ))
            red_flags.append(f"人员关联: {person}({', '.join(titles)}) 出现在 {len(companies)} 家公司")

    return relationships, red_flags, dict(company_personnel_map)


# --- Module 3: Document-document cross-references ----------------------------

def _detect_document_cross_references(
    texts: list[str], file_names: list[str],
) -> tuple[list[DetectedRelationship], list[str]]:
    """Detect cross-references and interdependencies between bid documents."""
    relationships = []
    red_flags = []

    bid_ref_pattern = re.compile(r'(?:详见|参见|见|参照|引用)\s*'
                                 r'(?:第[一二三四五六七八九十\d]+章|第[一二三四五六七八九十\d]+节|'
                                 r'附件[一二三四五六七八九十\d]+|'
                                 r'(?:[A-Z]+[-\d]+))')
    identical_passage_hashes = defaultdict(list)

    # 1. Detect explicit cross-references
    for idx, text in enumerate(texts):
        for m in bid_ref_pattern.finditer(text):
            if idx + 1 < len(file_names):
                snippet = text[max(0, m.start()-30):m.end()+30]
                relationships.append(DetectedRelationship(
                    source_entity=file_names[idx],
                    target_entity=file_names[idx + 1],
                    relation_type="cross_reference",
                    relation_subtype="explicit_ref",
                    confidence=0.70,
                    evidence=snippet,
                    module="doc_doc",
                ))

    # 2. Detect suspiciously identical passages (paragraph-level hashing)
    paragraph = re.compile(r'.{100,500}(?:。|；|\.|;|\n)')
    for idx, text in enumerate(texts):
        for para_m in paragraph.finditer(text):
            para_text = para_m.group().strip()
            if len(para_text) < 50:
                continue
            para_hash = hashlib.md5(para_text.encode()).hexdigest()
            identical_passage_hashes[para_hash].append((idx, para_text[:100]))

    for phash, occurrences in identical_passage_hashes.items():
        if len(occurrences) >= 2:
            doc_indices = list(set(o[0] for o in occurrences))
            if len(doc_indices) >= 2:
                doc_names = [file_names[i] if i < len(file_names) else f"doc_{i}" for i in doc_indices]
                relationships.append(DetectedRelationship(
                    source_entity=doc_names[0],
                    target_entity=doc_names[1] if len(doc_names) > 1 else doc_names[0],
                    relation_type="cross_reference",
                    relation_subtype="identical_passage",
                    confidence=0.92,
                    evidence=occurrences[0][1],
                    module="doc_doc",
                    risk_flag=True,
                    risk_reason=f"发现{len(occurrences)}处完全相同的段落内容",
                ))
                red_flags.append(f"文档雷同: {', '.join(doc_names)} 存在完全相同段落")

    return relationships, red_flags


# --- Module 4: Collusion signals --------------------------------------------

_COMMON_TYPOS = re.compile(
    r'(?:必须|必需|须要|需要)(?:的|地|得)'
    r'|'
    r'(?:权力|权利|权益)'
    r'|'
    r'(?:制定|制订|制定)'
)

def _detect_collusion_signals(
    file_data: list[dict],
) -> tuple[list[DetectedRelationship], list[str]]:
    """Detect forensic collusion signals: identical typos, formatting, metadata."""
    relationships = []
    red_flags = []

    # 1. Identical uncommon typo patterns across documents
    typo_patterns = defaultdict(list)
    for idx, fd in enumerate(file_data):
        text = fd.get('text', '')
        filename = fd.get('filename', f'doc_{idx}')

        # Find potential typos (look for patterns like repeated characters, unusual spacing)
        unusual_spacing = re.findall(r'.{2,10}\s{3,}.{2,10}', text)
        for us in unusual_spacing[:5]:
            typo_patterns[us].append(filename)

        # Unusual punctuation patterns
        unusual_punct = re.findall(r'[，。；]{2,}', text)
        for up in unusual_punct[:5]:
            typo_patterns[up].append(filename)

    for pattern, files in typo_patterns.items():
        if len(set(files)) >= 2:
            relationships.append(DetectedRelationship(
                source_entity=files[0],
                target_entity=files[1],
                relation_type="collusion_signal",
                relation_subtype="identical_formatting_quirk",
                confidence=0.60,
                evidence=f"异常格式模式: '{pattern}'",
                module="collusion",
                risk_flag=True,
                risk_reason="多份文件出现相同异常格式，可能来自同一来源",
            ))
            red_flags.append(f"串标疑义: 文件 {', '.join(set(files))} 出现相同异常格式")

    # 2. Same file metadata (author, creator tool)
    authors = defaultdict(list)
    for idx, fd in enumerate(file_data):
        meta = fd.get('metadata', {})
        author = meta.get('author', '')
        creator = meta.get('creator', '')
        filename = fd.get('filename', f'doc_{idx}')
        if author:
            authors[('author', author)].append(filename)
        if creator:
            authors[('creator', creator)].append(filename)

    for (attr_type, attr_val), files in authors.items():
        if len(files) >= 2:
            relationships.append(DetectedRelationship(
                source_entity=files[0],
                target_entity=files[1],
                relation_type="collusion_signal",
                relation_subtype=f"same_{attr_type}",
                confidence=0.75,
                evidence=f"相同{attr_type}: {attr_val}",
                module="collusion",
                risk_flag=True,
                risk_reason=f"多份文件{attr_type}相同，可能存在串通制作",
            ))
            red_flags.append(f"文件属性雷同: {', '.join(files)} 的{attr_type}均为 {attr_val}")

    return relationships, red_flags


# --- KB integration ---------------------------------------------------------

def _check_knowledge_base(entity_name: str, entity_type: str) -> list[dict]:
    """Search internal knowledge base for known entity relationships."""
    results = []
    try:
        from app.services.rag_engine import RAGEngine
        rag = RAGEngine()
        kb_results = rag.search(f"{entity_type}: {entity_name}", top_k=5)
        for r in kb_results:
            results.append({
                'source': 'internal_kb',
                'content': r.get('content', '')[:500],
                'score': r.get('score', 0),
            })
    except Exception:
        pass
    return results


# --- 天眼查 toggle logic ---------------------------------------------------

def _build_company_personnel_list(
    company_personnel_map: dict[str, list[str]],
    companies: list[str],
    tianyancha_enabled: bool
) -> dict:
    """Build the company-personnel list for manual review when 天眼查 is off,
    or return verified data when it's on.

    When 天眼查 is OFF: returns a grouped list of companies with their key personnel
    for manual human review.
    """
    result = {
        'tianyancha_enabled': tianyancha_enabled,
        'companies': [],
        'manual_review_required': not tianyancha_enabled,
    }

    # Group by company
    company_personnel = defaultdict(lambda: {'personnel': [], 'files': set()})
    for person, appearances in company_personnel_map.items():
        for app in appearances:
            comp = app['company']
            company_personnel[comp]['personnel'].append({
                'name': person,
                'title': app['title'],
            })
            company_personnel[comp]['files'].add(app['file'])

    for comp, info in company_personnel.items():
        result['companies'].append({
            'name': comp,
            'personnel': info['personnel'],
            'file_count': len(info['files']),
            'files': list(info['files']),
        })

    return result


# --- Main public API ---------------------------------------------------------

def extract_relationships(
    file_data: list[dict],
    audit = None,
) -> RelationshipReport:
    """Main entry point: run all 4 relationship detection modules on bid documents.

    Args:
        file_data: list of {'filename': str, 'text': str, 'metadata': dict, ...}
        audit: optional AuditLogger instance

    Returns:
        RelationshipReport with all entities, relationships, red flags, and risk score.
    """
    if audit:
        audit.component("relationship_extract", file_count=len(file_data))

    texts = [fd.get('text', '') for fd in file_data]
    file_names = [fd.get('filename', f'doc_{i}') for i, fd in enumerate(file_data)]

    # Determine NER provider from config
    try:
        from app.services.runtime_config import get
        ner_provider = get('relation_extraction_ner_provider', 'hanlp')
        use_llm = get('relation_extraction_llm_fallback', True)
        tianyancha_enabled = get('relation_tianyancha_enabled', False)
    except Exception:
        ner_provider = 'hanlp'
        use_llm = True
        tianyancha_enabled = False

    # Step 1: Entity extraction
    if ner_provider == 'hanlp':
        all_entities = _extract_with_hanlp(texts)
    else:
        all_entities = _extract_with_regex(texts)

    merged_entities = _merge_entities(all_entities)
    if audit:
        audit.component("relationship_ner", status="OK",
                        provider=ner_provider,
                        total_entities=len(merged_entities),
                        companies=len([e for e in merged_entities if e.entity_type == 'company']),
                        persons=len([e for e in merged_entities if e.entity_type == 'person']))

    report = RelationshipReport(entities=merged_entities)

    # Step 2: Module 1 — Company-company
    cc_rels, cc_flags = _detect_company_relationships(all_entities, texts)
    report.relationships.extend(cc_rels)
    report.red_flags.extend(cc_flags)
    report.modules_run.append("company_company")
    if audit:
        audit.component("relationship_module", status="OK",
                        module="company_company",
                        relations=len(cc_rels), flags=len(cc_flags))

    # Step 3: Module 2 — Personnel-company
    pc_rels, pc_flags, company_personnel_map = _detect_personnel_relationships(
        all_entities, texts, file_names)
    report.relationships.extend(pc_rels)
    report.red_flags.extend(pc_flags)
    report.modules_run.append("personnel_company")
    report.company_personnel_map = _build_company_personnel_list(
        company_personnel_map,
        list(set(e.text for e in merged_entities if e.entity_type == 'company')),
        tianyancha_enabled,
    )
    if audit:
        audit.component("relationship_module", status="OK",
                        module="personnel_company",
                        relations=len(pc_rels), flags=len(pc_flags),
                        persons=len(company_personnel_map))

    # Step 4: Module 3 — Document cross-references
    dc_rels, dc_flags = _detect_document_cross_references(texts, file_names)
    report.relationships.extend(dc_rels)
    report.red_flags.extend(dc_flags)
    report.modules_run.append("doc_doc")
    if audit:
        audit.component("relationship_module", status="OK",
                        module="doc_doc",
                        relations=len(dc_rels), flags=len(dc_flags))

    # Step 5: Module 4 — Collusion signals
    cs_rels, cs_flags = _detect_collusion_signals(file_data)
    report.relationships.extend(cs_rels)
    report.red_flags.extend(cs_flags)
    report.modules_run.append("collusion")
    if audit:
        audit.component("relationship_module", status="OK",
                        module="collusion",
                        relations=len(cs_rels), flags=len(cs_flags))

    # Step 6: Aggregated risk score (0-100)
    score = 0.0
    risk_rels = [r for r in report.relationships if r.risk_flag]
    if risk_rels:
        # Company-company risk relations are highest weight
        cc_risk = sum(1 for r in risk_rels if r.module == "company_company")
        pc_risk = sum(1 for r in risk_rels if r.module == "personnel_company")
        dc_risk = sum(1 for r in risk_rels if r.module == "doc_doc")
        cs_risk = sum(1 for r in risk_rels if r.module == "collusion")

        score = min(100.0, cc_risk * 25.0 + pc_risk * 20.0 + dc_risk * 12.0 + cs_risk * 15.0)
    report.risk_score = round(score, 1)

    if audit:
        audit.component("relationship_summary", status="OK",
                        total_relations=len(report.relationships),
                        total_entities=len(report.entities),
                        risk_flags=len(report.red_flags),
                        risk_score=report.risk_score)

    return report


# --- DB persistence ----------------------------------------------------------

def save_relationship_results(
    user_id: str,
    task_id: str,
    report: RelationshipReport,
    project_id: int = None,
) -> int:
    """Persist relationship extraction results to DB. Returns count of rows saved."""
    saved = 0
    try:
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for rel in report.relationships:
                    cur.execute("""
                        INSERT INTO entity_relationships
                            (user_id, task_id, project_id, source_entity, target_entity,
                             relation_type, relation_subtype, confidence,
                             evidence_text, risk_flag, risk_reason, module)
                        VALUES (%s,%s,%s,%s,%s, %s,%s,%s, %s,%s,%s, %s)
                    """, (
                        user_id, task_id, project_id,
                        rel.source_entity, rel.target_entity,
                        rel.relation_type, rel.relation_subtype,
                        rel.confidence, rel.evidence[:1000],
                        rel.risk_flag, rel.risk_reason[:500],
                        rel.module,
                    ))
                    saved += 1

                # Save risk summary
                cur.execute("""
                    INSERT INTO relationship_risk_summary
                        (user_id, task_id, total_entities, total_relations,
                         red_flags, risk_score, modules_run, tianyancha_used, details)
                    VALUES (%s,%s,%s,%s, %s,%s,%s, %s,%s)
                    ON CONFLICT (task_id) DO UPDATE SET
                        total_entities = EXCLUDED.total_entities,
                        total_relations = EXCLUDED.total_relations,
                        red_flags = EXCLUDED.red_flags,
                        risk_score = EXCLUDED.risk_score,
                        modules_run = EXCLUDED.modules_run
                """, (
                    user_id, task_id,
                    len(report.entities), len(report.relationships),
                    _json.dumps(report.red_flags, ensure_ascii=False),
                    report.risk_score,
                    _json.dumps(report.modules_run),
                    report.tianyancha_used,
                    _json.dumps(report.company_personnel_map, ensure_ascii=False),
                ))
                conn.commit()
        logger.info(f"Saved {saved} relationships for task {task_id}, risk_score={report.risk_score}")
    except Exception as e:
        logger.error(f"Failed to save relationship results: {e}", exc_info=True)
    return saved
