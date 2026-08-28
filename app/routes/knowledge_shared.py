"""Shared helpers for the knowledge blueprint family.

Extracted from app/routes/knowledge.py so sub-blueprint modules can reuse the
fire-and-forget indexing/wiki/entity helpers and the skill-auditor guard without
creating import cycles. The ``knowledge_bp`` Blueprint object itself is defined
in app/routes/knowledge.py; these helpers are consumed by it and its sibling
sub-modules (knowledge_notebook.py, knowledge_company_kb.py, ...).
"""
import logging
import threading

logger = logging.getLogger(__name__)


def _try_index_file(file_id, content, source, metadata=None, skill_summary=None):
    """Fire-and-forget RAG index update in background thread."""
    if not content:
        return
    def _do():
        try:
            from app.services.rag_engine import index_file
            index_file(file_id, content, source, metadata, skill_summary=skill_summary)
        except Exception as e:
            logger.warning(f"Background index failed for {source}.{file_id}: {e}")
    t = threading.Thread(target=_do, daemon=True)
    t.start()


def _try_wiki_ingest(file_id, content, filename, source_type, metadata=None):
    """Dispatch wiki ingest Celery task (fire-and-forget).

    LLM-based wiki page generation from uploaded documents. Uses the existing
    wiki_ingest_task Celery task which calls ingest_file() with max_retries=2.
    Failures are logged but do not affect the upload flow.
    """
    if not content:
        return
    try:
        from celery_app import celery as celery_app
        celery_app.send_task('wiki_ingest_task', args=[file_id, content, filename, source_type, metadata or {}])
    except Exception as e:
        logger.warning(f"Wiki ingest dispatch failed for {source_type}.{file_id}: {e}")


def _try_entity_extract(file_id, content, filename, source_type, doc_type="general", wiki_category="general", metadata=None):
    """Fire-and-forget entity extraction in background thread.

    Runs LLM-based entity extraction, resolves against existing entity index,
    and creates/updates entity wiki pages. Failures are logged but do not affect
    the upload flow.
    """
    if not content or len(content) < 50:
        return
    def _do():
        try:
            from app.services.wiki_entity_service import process_upload_entity_extraction
            process_upload_entity_extraction(
                file_id, content, filename, source_type,
                doc_type, wiki_category, metadata or {}
            )
        except Exception as e:
            logger.warning(f"Entity extraction failed for {source_type}.{file_id}: {e}")
    t = threading.Thread(target=_do, daemon=True)
    t.start()


def is_skill_auditor():
    """Check if current session is admin or auditor (skill review permissions)."""
    from flask import session
    role = session.get('role', 'user')
    is_auditor = session.get('is_auditor', False)
    return role == 'admin' or is_auditor
