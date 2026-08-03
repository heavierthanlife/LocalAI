"""Unified Bid Audit Engine — orchestrator, scoring, retry, progress emission."""
import json
import logging
import queue
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any

from app.config import DATA_DIR, to_rel_path, resolve_path
from app.database import get_db_connection
from app.services.compliance_prompts import VERDICT_PASS

logger = logging.getLogger(__name__)

_progress_queues: dict[int, queue.Queue] = {}
_queues_lock = threading.Lock()

AUDIT_REPORTS_DIR = DATA_DIR / 'audit_reports'


def _score_rule_extraction(findings: dict) -> float:
    rules = findings.get('rules', [])
    if not rules:
        return 0.0
    expected_min = 5
    try:
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT severity_thresholds FROM audit_config WHERE function_name = 'rule_extraction'")
                row = cur.fetchone()
                if row and row[0]:
                    expected_min = row[0].get('min_extracted_rules', 5)
    except Exception:
        pass
    return min(100.0, len(rules) / max(expected_min, 1) * 100)


def _score_compliance_check(findings: dict) -> float:
    results = findings.get('results', [])
    if not results:
        return 100.0
    passed = sum(1 for r in results if r.get('verdict') == VERDICT_PASS)
    return (passed / len(results)) * 100


def _score_typo_detection(findings: dict, text_length: int) -> float:
    findings_list = findings.get('findings', [])
    if not findings_list:
        return 100.0
    penalty = 5
    try:
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT severity_thresholds FROM audit_config WHERE function_name = 'typo_detection'")
                row = cur.fetchone()
                if row and row[0]:
                    penalty = row[0].get('penalty_per_10k', 5)
    except Exception:
        pass
    per_10k = len(findings_list) / max(text_length / 10000, 0.01)
    return max(0.0, 100 - per_10k * penalty)


def _score_quote_anomaly(findings: dict) -> float:
    severity = findings.get('severity_index', 50)
    return max(0.0, 100 - severity)


def _score_relationship_extraction(findings: dict) -> float:
    signals = findings.get('collusion_signals', [])
    red_flags = findings.get('red_flags', [])
    total_risks = len(signals) + len(red_flags)
    weight = 15
    try:
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT severity_thresholds FROM audit_config WHERE function_name = 'relationship_extraction'")
                row = cur.fetchone()
                if row and row[0]:
                    weight = row[0].get('risk_signal_weight', 15)
    except Exception:
        pass
    return max(0.0, 100 - total_risks * weight)


def _score_ai_review(findings: dict) -> float:
    axes = findings.get('axes', {})
    if not axes:
        return 50.0
    scores = []
    for axis_name, axis_data in axes.items():
        if isinstance(axis_data, dict):
            s = axis_data.get('score', 0)
        elif isinstance(axis_data, (int, float)):
            s = axis_data
        else:
            s = 0
        if isinstance(s, (int, float)) and s >= 0:
            scores.append(min(s, 10))
    if not scores:
        return 50.0
    return (sum(scores) / len(scores)) * 10


def _score_style_analysis(findings: dict) -> float:
    formality = findings.get('formality_level', 50)
    consistency = findings.get('consistency', 50)
    return (formality + consistency) / 2


def _score_timeline_compliance(findings: dict) -> float:
    if not findings or findings.get('error'):
        return 100.0
    delayed = findings.get('delayed_count', 0)
    total_delay = findings.get('total_delay_days', 0)
    return max(0.0, 100.0 - delayed * 10 - total_delay * 2)


SCORING_FUNCTIONS = {
    'rule_extraction': _score_rule_extraction,
    'compliance_check': _score_compliance_check,
    'typo_detection': _score_typo_detection,
    'quote_anomaly': _score_quote_anomaly,
    'relationship_extraction': _score_relationship_extraction,
    'ai_doc_review': _score_ai_review,
    'style_analysis': _score_style_analysis,
    'timeline_compliance': _score_timeline_compliance,
}


def _emit(run_id: int, event_type: str, **data):
    with _queues_lock:
        q = _progress_queues.get(run_id)
    if q:
        payload = json.dumps({'type': event_type, **data}, ensure_ascii=False, default=str)
        try:
            q.put(payload, timeout=5)
        except queue.Full:
            pass


def register_progress_queue(run_id: int, q: queue.Queue):
    with _queues_lock:
        _progress_queues[run_id] = q


def unregister_progress_queue(run_id: int):
    with _queues_lock:
        _progress_queues.pop(run_id, None)


def _is_doc_file(filename: str) -> bool:
    """Check if file is eligible for text audit using unified registry."""
    from app.config import get_document_extensions
    ext = filename.rfind('.')
    return ext >= 0 and filename[ext:].lower() in get_document_extensions()


def run_preflight(folder_ids: list[int], file_ids: list[int] | None = None) -> dict:
    files_status = []
    ready_count = 0
    missing_count = 0
    skipped_count = 0
    # Cache folder names
    folder_names = {}
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for fid in folder_ids:
                cur.execute("SELECT name FROM project_folders WHERE id = %s", (fid,))
                row = cur.fetchone()
                folder_names[fid] = row[0] if row else f'Folder_{fid}'
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for fid in folder_ids:
                cur.execute("""
                    SELECT pf.id, pf.filename, pf.mime_type,
                           pf.content IS NOT NULL AND pf.content != '' AS has_text,
                           pf.stored_path
                    FROM project_files pf
                    WHERE pf.folder_id = %s
                    ORDER BY pf.filename
                """, (fid,))
                for row in cur.fetchall():
                    file_id, filename, mime_type, has_text, stored_path = row
                    is_doc = _is_doc_file(filename or '')
                    # If specific file_ids given, skip files not in the list
                    if file_ids is not None and file_id not in file_ids:
                        continue
                    if not is_doc:
                        files_status.append({
                            'file_id': file_id, 'filename': filename,
                            'folder_id': fid, 'folder_name': folder_names.get(fid, ''),
                            'status': 'skipped',
                            'reason': '非文档格式', 'mime_type': mime_type,
                        })
                        skipped_count += 1
                        continue
                    status = 'ready' if has_text else 'missing'
                    if has_text:
                        ready_count += 1
                    else:
                        missing_count += 1
                    files_status.append({
                        'file_id': file_id, 'filename': filename,
                        'folder_id': fid, 'folder_name': folder_names.get(fid, ''),
                        'status': status,
                        'mime_type': mime_type, 'stored_path': stored_path,
                    })
    return {
        'files': files_status,
        'ready_count': ready_count,
        'missing_count': missing_count,
        'skipped_count': skipped_count,
        'total_count': len(files_status),
    }


def _extract_text_on_demand(file_id: int, stored_path: str) -> str | None:
    try:
        from app.services.file_processing import extract_text_from_file
        import os
        stored_path = resolve_path(stored_path)
        if not os.path.exists(stored_path):
            logger.warning(f"File not found for extraction: {stored_path}")
            return None

        class _FakeFile:
            def __init__(self, path):
                self.filename = os.path.basename(path)
                self._path = path
            def read(self):
                with open(self._path, 'rb') as f:
                    return f.read()
            def seek(self, n):
                pass

        fake = _FakeFile(stored_path)
        text, _ = extract_text_from_file(fake)
        if text:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE project_files SET content = %s WHERE id = %s", (text, file_id))
                conn.commit()
        return text
    except Exception as e:
        logger.error(f"On-demand extraction failed for file {file_id}: {e}")
        return None


def _get_folder_name(folder_id: int) -> str:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM project_folders WHERE id = %s", (folder_id,))
                row = cur.fetchone()
                return row[0] if row else f"Folder_{folder_id}"
    except Exception:
        return f"Folder_{folder_id}"


def _get_file_text(file_id: int, stored_path: str = "") -> tuple[str | None, int]:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT content FROM project_files WHERE id = %s", (file_id,))
                row = cur.fetchone()
                if row and row[0]:
                    text = row[0]
                    return text, len(text)
        return None, 0
    except Exception as e:
        logger.error(f"Failed to get text for file {file_id}: {e}")
        return None, 0


def _call_function(func_name: str, text: str, doc_name: str, file_id: int,
                   rules_cache: dict | None = None) -> dict:
    if func_name == 'rule_extraction':
        from app.services.rule_extractor import RuleExtractor
        extractor = RuleExtractor()
        return extractor.extract(text, doc_name, use_ai=True)

    elif func_name == 'compliance_check':
        from app.services.compliance_checker import ComplianceChecker
        rules = rules_cache.get('rules', []) if rules_cache else []
        checker = ComplianceChecker()
        return checker.check(text, rules, doc_name, use_ai=True)

    elif func_name == 'typo_detection':
        from app.services.typo_detector import detect_typos
        report = detect_typos(text, doc_name)
        return {
            'findings': [
                {'type': f.layer, 'original': f.original, 'corrected': f.corrected,
                 'context': f.context, 'severity': f.severity}
                for f in report.findings
            ],
            'layers_run': report.layers_run,
            'total_findings': len(report.findings),
        }

    elif func_name == 'quote_anomaly':
        from app.services.quote_anomaly import check_quote_anomaly
        result = check_quote_anomaly(text, doc_name)
        findings = {}
        for key in ['prices', 'percentages', 'daxie_mismatches', 'same_rate',
                     'clustering', 'benford_deviation', 'cv', 'severity_index',
                     'flags', 'summary']:
            val = getattr(result, key, None)
            if val is not None:
                findings[key] = val
        return findings

    elif func_name == 'relationship_extraction':
        from app.services.relationship_extractor import extract_relationships
        file_data = [{'filename': doc_name, 'text': text, 'metadata': {'file_id': file_id}}]
        report = extract_relationships(file_data)
        return {
            'entities': [{'name': e.name, 'type': e.entity_type} for e in report.entities],
            'companies': report.companies,
            'personnel': report.personnel,
            'relationships': report.relationships,
            'collusion_signals': getattr(report, 'collusion_signals', []),
            'red_flags': getattr(report, 'red_flags', []),
            'risk_score': getattr(report, 'risk_score', 0),
        }

    elif func_name == 'ai_doc_review':
        return _run_ai_review(text, doc_name)

    elif func_name == 'style_analysis':
        return _run_style_analysis(text)

    return {'error': f'Unknown function: {func_name}'}


def _run_ai_review(text: str, doc_name: str) -> dict:
    import uuid as _uuid
    from app.services._shared_helpers import AI_DOC_REVIEW_PROMPT
    try:
        from app.services.agent import get_agent
        agent = get_agent()
        config = {"configurable": {"thread_id": f"audit_review_{_uuid.uuid4()}"}}
        response = agent.invoke(
            {"messages": [
                {"role": "user", "content": f"{AI_DOC_REVIEW_PROMPT}\n\n=== doc ===\n{text[:12000]}"}
            ]},
            config
        )
        raw = response["messages"][-1].content
    except Exception as e:
        logger.error(f"AI review failed: {e}")
        return {'raw_analysis': str(e), 'parse_error': True, 'error': str(e)}

    import re as _re
    try:
        json_match = _re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json.loads(json_match.group(0))
        return {'raw_analysis': raw, 'parse_error': True}
    except json.JSONDecodeError:
        return {'raw_analysis': raw, 'parse_error': True}


def _run_style_analysis(text: str) -> dict:
    if not text or len(text.strip()) < 100:
        return {'formality_level': 50, 'consistency': 50, 'error': 'doc too short'}
    try:
        from app.services.style_engine import _analyze_formality, _analyze_tone
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 20][:100]
        if not paragraphs:
            return {'formality_level': 50, 'consistency': 50}
        formality = _analyze_formality(paragraphs)
        tone = _analyze_tone(paragraphs)
        return {
            'formality_level': formality.get('formality_score', 50),
            'formality_label': formality.get('label', 'unknown'),
            'tone': tone,
            'avg_sentence_length': formality.get('avg_sentence_length', 0),
        }
    except Exception as e:
        logger.error(f"Style analysis failed: {e}")
        return {'formality_level': 50, 'consistency': 50, 'error': str(e)}


def _save_file_result(conn, run_id: int, file_id: int, folder_id: int,
                      bidder_label: str, filename: str, func_name: str,
                      score: float, status: str, findings: dict,
                      error_msg: str = "", retry_count: int = 0):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO audit_file_results
            (run_id, file_id, folder_id, bidder_label, filename, function_name,
             score, status, findings, error_message, retry_count,
             started_at, completed_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, NOW(), NOW())
    """, (run_id, file_id, folder_id, bidder_label, filename, func_name,
          score, status, json.dumps(findings, ensure_ascii=False, default=str),
          error_msg, retry_count))
    conn.commit()


def _run_project_level_function(func_name: str, project_id: int) -> tuple[dict, float]:
    if func_name == 'timeline_compliance':
        from app.services.project_timeline_service import get_timeline
        timeline = get_timeline(project_id)
        if not timeline:
            return ({'error': 'No timeline found for this project'}, 100.0)

        milestones = timeline.get('milestones', [])
        findings = []
        score = 100.0

        for m in milestones:
            if m.get('diff_days') is None:
                continue

            diff = m['diff_days']
            mandatory = m.get('mandatory') if 'mandatory' in m else False
            if diff > 0:
                penalty = 20 if mandatory else 10
                score -= penalty
                findings.append({
                    'milestone': m.get('name', m.get('code', '')),
                    'code': m.get('code', ''),
                    'diff_days': diff,
                    'diff_reason': m.get('diff_reason', ''),
                    'reason_category': m.get('reason_category', ''),
                    'verdict': 'delay',
                    'mandatory': mandatory,
                    'penalty': penalty,
                })
            elif diff == 0:
                score += 5
                findings.append({
                    'milestone': m.get('name', ''),
                    'code': m.get('code', ''),
                    'diff_days': 0,
                    'verdict': 'on_time',
                })
            else:
                findings.append({
                    'milestone': m.get('name', ''),
                    'code': m.get('code', ''),
                    'diff_days': diff,
                    'verdict': 'advanced',
                })

        score = max(0.0, min(100.0, score))
        total_delay = sum(f['diff_days'] for f in findings if f.get('diff_days', 0) > 0)

        return ({
            'findings': findings,
            'total_delay_days': total_delay,
            'delayed_count': len([f for f in findings if f.get('verdict') == 'delay']),
            'on_time_count': len([f for f in findings if f.get('verdict') == 'on_time']),
            'advanced_count': len([f for f in findings if f.get('verdict') == 'advanced']),
        }, score)

    return ({'error': f'Unknown project-level function: {func_name}'}, 0.0)


def run_audit(run_id: int, folder_ids: list[int], enabled_functions: list[str],
              extract_on_demand: bool, user_id: str, file_ids: list[int] | None = None,
              project_level_functions: list[str] | None = None):
    """Main orchestrator - runs in a background thread."""
    import os as _os
    _os.makedirs(str(AUDIT_REPORTS_DIR), exist_ok=True)

    try:
        config_snapshot = {}
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT function_name, enabled_by_default, fail_threshold, weight, severity_thresholds FROM audit_config")
                for row in cur.fetchall():
                    config_snapshot[row[0]] = {
                        'enabled': row[1], 'fail_threshold': row[2],
                        'weight': row[3], 'severity_thresholds': row[4] or {}
                    }

        total_weight = sum(config_snapshot[f]['weight'] for f in enabled_functions)
        for f in enabled_functions:
            config_snapshot[f]['weight'] /= max(total_weight, 0.01)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE audit_runs SET config_snapshot = %s::jsonb WHERE id = %s",
                    (json.dumps(config_snapshot, ensure_ascii=False, default=str), run_id))
                conn.commit()

        bidder_files: dict[int, dict] = {}
        total_files = 0
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for fid in folder_ids:
                    folder_name = _get_folder_name(fid)
                    cur.execute(
                        "SELECT id, filename, content, stored_path FROM project_files WHERE folder_id = %s ORDER BY filename",
                        (fid,))
                    files = []
                    for row in cur.fetchall():
                        file_id = row[0]
                        filename = row[1]
                        # Skip non-document files
                        if not _is_doc_file(filename or ''):
                            continue
                        # If specific file_ids given, skip files not in the list
                        if file_ids is not None and file_id not in file_ids:
                            continue
                        files.append({
                            'file_id': file_id, 'filename': filename,
                            'has_text': bool(row[2] and row[2].strip()),
                            'stored_path': row[3],
                        })
                    bidder_files[fid] = {'name': folder_name, 'files': files}
                    total_files += len(files)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE audit_runs SET bidder_count = %s, file_count = %s WHERE id = %s",
                    (len(folder_ids), total_files, run_id))
                conn.commit()

        _emit(run_id, 'phase', phase='auditing', bidder_count=len(folder_ids),
              file_count=total_files, functions=enabled_functions)

        all_scores = []
        fatal_flaw = False
        file_index = 0

        for folder_id, bidder in bidder_files.items():
            for file_info in bidder['files']:
                file_index += 1
                file_id = file_info['file_id']
                filename = file_info['filename']

                _emit(run_id, 'file_start', bidder=bidder['name'], filename=filename,
                      file_index=file_index, total_files=total_files)

                text, text_len = _get_file_text(file_id, file_info['stored_path'])
                if not text and extract_on_demand:
                    _emit(run_id, 'phase', phase='extracting', filename=filename)
                    text = _extract_text_on_demand(file_id, file_info['stored_path'])
                    if text:
                        text_len = len(text)

                if not text:
                    _emit(run_id, 'file_error', bidder=bidder['name'],
                          filename=filename, error='No text available')
                    for func_name in enabled_functions:
                        with get_db_connection() as conn:
                            _save_file_result(conn, run_id, file_id, folder_id,
                                              bidder['name'], filename, func_name, 0,
                                              'skipped', {}, 'No text content')
                    continue

                rules_cache = None

                for func_name in enabled_functions:
                    _emit(run_id, 'function_start', bidder=bidder['name'],
                          filename=filename, function=func_name)

                    findings = None
                    score = 0
                    status = 'success'
                    error_msg = ""
                    retry_count = 0

                    try:
                        findings = _call_function(func_name, text, filename, file_id, rules_cache)
                        if func_name == 'rule_extraction':
                            rules_cache = findings
                        if func_name == 'typo_detection':
                            score = SCORING_FUNCTIONS[func_name](findings, text_len)
                        else:
                            score = SCORING_FUNCTIONS[func_name](findings)
                    except Exception as e:
                        logger.warning("Function %s failed on %s, retrying: %s", func_name, filename, e)
                        time.sleep(3)
                        retry_count = 1
                        try:
                            findings = _call_function(func_name, text, filename, file_id, rules_cache)
                            if func_name == 'rule_extraction':
                                rules_cache = findings
                            if func_name == 'typo_detection':
                                score = SCORING_FUNCTIONS[func_name](findings, text_len)
                            else:
                                score = SCORING_FUNCTIONS[func_name](findings)
                        except Exception as e2:
                            logger.error("Function %s failed on %s after retry: %s", func_name, filename, e2)
                            status = 'error'
                            error_msg = str(e2)[:500]
                            findings = {'error': error_msg}
                            score = 0

                    all_scores.append(score)

                    threshold = config_snapshot.get(func_name, {}).get('fail_threshold', 50)
                    if score < threshold:
                        fatal_flaw = True

                    with get_db_connection() as conn:
                        _save_file_result(conn, run_id, file_id, folder_id,
                                          bidder['name'], filename, func_name,
                                          score, status, findings or {},
                                          error_msg, retry_count)

                    _emit(run_id, 'function_done', bidder=bidder['name'],
                          filename=filename, function=func_name,
                          score=score, status=status,
                          error=error_msg[:100] if error_msg else None)

        if project_level_functions:
            project_id = None
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT project_id FROM audit_runs WHERE id = %s", (run_id,))
                    row = cur.fetchone()
                    if row:
                        project_id = row[0]

            if project_id:
                for func_name in project_level_functions:
                    _emit(run_id, 'function_start', bidder='project', filename='timeline', function=func_name)
                    try:
                        findings, score = _run_project_level_function(func_name, project_id)
                        if score is not None:
                            all_scores.append(score)
                        threshold = config_snapshot.get(func_name, {}).get('fail_threshold', 50)
                        if score is not None and score < threshold:
                            fatal_flaw = True
                        with get_db_connection() as conn:
                            _save_file_result(conn, run_id, None, None, 'project', 'timeline',
                                              func_name, score or 0, 'success', findings or {})
                    except Exception as e:
                        logger.error(f"Project-level function {func_name} failed: {e}", exc_info=True)
                        with get_db_connection() as conn:
                            _save_file_result(conn, run_id, None, None, 'project', 'timeline',
                                              func_name, 0, 'error', {}, str(e)[:500])

        overall_score = sum(all_scores) / max(len(all_scores), 1)
        overall_status = 'FAIL' if fatal_flaw else 'PASS'

        _emit(run_id, 'phase', phase='reporting')
        docx_path, xlsx_path = _generate_reports(run_id, folder_ids, enabled_functions)

        now = datetime.now(timezone.utc)
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE audit_runs SET status = 'completed', overall_score = %s,
                        overall_status = %s, docx_path = %s, xlsx_path = %s,
                        completed_at = %s
                    WHERE id = %s
                """, (overall_score, overall_status, to_rel_path(docx_path), to_rel_path(xlsx_path), now, run_id))
                conn.commit()

        _emit(run_id, 'complete', overall_score=round(overall_score, 1),
              overall_status=overall_status, docx_path=docx_path, xlsx_path=xlsx_path)

        try:
            from app.services.audit_wiki_publisher import publish_audit_to_wiki
            publish_audit_to_wiki(run_id)
        except Exception as e:
            logger.warning(f"Wiki publish for run {run_id} failed (non-fatal): {e}")

    except Exception as e:
        logger.error("Audit run %s crashed: %s", run_id, e, exc_info=True)
        _emit(run_id, 'error', message=str(e)[:500])
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE audit_runs SET status = 'failed', error_message = %s, completed_at = NOW() WHERE id = %s",
                    (str(e)[:1000], run_id))
                conn.commit()


def _generate_reports(run_id: int, folder_ids: list[int],
                      enabled_functions: list[str]) -> tuple[str | None, str | None]:
    try:
        from app.services.audit_report import generate_docx, generate_xlsx
        docx_path = generate_docx(run_id, folder_ids, enabled_functions)
        xlsx_path = generate_xlsx(run_id, folder_ids, enabled_functions)
        return docx_path, xlsx_path
    except Exception as e:
        logger.error(f"Report generation failed for run {run_id}: {e}")
        return None, None


def get_run_results(run_id: int) -> dict | None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, project_id, user_id, status, config_snapshot, overall_score,
                       overall_status, bidder_count, file_count, docx_path, xlsx_path,
                       error_message, started_at, completed_at
                FROM audit_runs WHERE id = %s
            """, (run_id,))
            run_row = cur.fetchone()
            if not run_row:
                return None
            run_data = {
                'id': run_row[0], 'project_id': run_row[1], 'user_id': run_row[2],
                'status': run_row[3], 'config_snapshot': run_row[4],
                'overall_score': run_row[5], 'overall_status': run_row[6],
                'bidder_count': run_row[7], 'file_count': run_row[8],
                'docx_path': run_row[9], 'xlsx_path': run_row[10],
                'error_message': run_row[11],
                'started_at': run_row[12].isoformat() if run_row[12] else None,
                'completed_at': run_row[13].isoformat() if run_row[13] else None,
            }
            cur.execute("""
                SELECT id, file_id, folder_id, bidder_label, filename, function_name,
                       score, status, findings, error_message, retry_count,
                       started_at, completed_at
                FROM audit_file_results WHERE run_id = %s
                ORDER BY folder_id, file_id, function_name
            """, (run_id,))
            file_results = []
            for fr in cur.fetchall():
                file_results.append({
                    'id': fr[0], 'file_id': fr[1], 'folder_id': fr[2],
                    'bidder_label': fr[3], 'filename': fr[4], 'function_name': fr[5],
                    'score': fr[6], 'status': fr[7], 'findings': fr[8],
                    'error_message': fr[9], 'retry_count': fr[10],
                    'started_at': fr[11].isoformat() if fr[11] else None,
                    'completed_at': fr[12].isoformat() if fr[12] else None,
                })
            run_data['file_results'] = file_results
            return run_data


def get_project_history(project_id: int) -> list[dict]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, user_id, status, overall_score, overall_status,
                       bidder_count, file_count, docx_path, xlsx_path,
                       started_at, completed_at
                FROM audit_runs
                WHERE project_id = %s
                ORDER BY started_at DESC
                LIMIT 50
            """, (project_id,))
            results = []
            for row in cur.fetchall():
                results.append({
                    'id': row[0], 'user_id': row[1], 'status': row[2],
                    'overall_score': row[3], 'overall_status': row[4],
                    'bidder_count': row[5], 'file_count': row[6],
                    'docx_path': row[7], 'xlsx_path': row[8],
                    'started_at': row[9].isoformat() if row[9] else None,
                    'completed_at': row[10].isoformat() if row[10] else None,
                })
            return results


def get_running_audit(project_id: int) -> dict | None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM audit_runs WHERE project_id = %s AND status = 'running' ORDER BY started_at DESC LIMIT 1",
                (project_id,))
            row = cur.fetchone()
            return {'run_id': row[0]} if row else None