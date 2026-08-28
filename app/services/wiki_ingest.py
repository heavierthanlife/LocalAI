"""Batch ingest of source documents into the project wiki."""
import os, logging, json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from psycopg2 import sql

from app.config import DATA_DIR

logger = logging.getLogger(__name__)


def init_flask_context():
    from app import create_app
    app = create_app()
    app.app_context().push()
    return app


def ingest_file(
    file_id: int,
    content: str,
    filename: str,
    source_type: str,
    metadata: dict = None,
) -> dict:
    from app.services.llm_provider import call_llm
    from app.services import wiki_engine
    from app.services import wiki_prompts

    if metadata is None:
        metadata = {}

    wiki_structure = {
        'index': wiki_engine.read_wiki_index(),
        'tree': wiki_engine.list_wiki_tree(),
    }

    truncated = content[:8000]

    provider_id = os.getenv('WIKI_DEDICATED_PROVIDER')
    model = os.getenv('WIKI_LLM_MODEL')

    user_prompt = wiki_prompts.WIKI_EXTRACT_USER_PROMPT.format(
        filename=filename,
        wiki_structure=json.dumps(wiki_structure, ensure_ascii=False, indent=2),
        content=truncated,
    )

    try:
        raw = call_llm(
            system_prompt=wiki_prompts.WIKI_INGEST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=4096,
            provider_id=provider_id,
            model=model,
        )
    except Exception as e:
        logger.warning(f"LLM call failed for file_id={file_id}: {e}")
        return {'pages_updated': 0, 'pages_created': 0, 'log_entry': ''}

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse LLM JSON for file_id={file_id}: {e}")
        return {'pages_updated': 0, 'pages_created': 0, 'log_entry': ''}

    updates = parsed.get('updates', [])
    new_pages = parsed.get('new_pages', [])
    index_updates = parsed.get('index_updates')
    log_entry = parsed.get('log_entry', '')

    pages_updated = 0
    pages_created = 0

    for page in updates:
        path = page['path']
        fm = page.get('frontmatter', {})
        page_content = page.get('content', '')
        wiki_engine.write_wiki_page(path, fm, page_content)
        wiki_engine.record_origin_link(
            path, source_type, file_id,
            metadata.get('original_name', filename),
        )
        wiki_engine.index_wiki_to_rag(path, page_content, [file_id])
        pages_updated += 1

    for page in new_pages:
        path = page['path']
        fm = page.get('frontmatter', {})
        page_content = page.get('content', '')
        wiki_engine.write_wiki_page(path, fm, page_content)
        wiki_engine.record_origin_link(
            path, source_type, file_id,
            metadata.get('original_name', filename),
        )
        wiki_engine.index_wiki_to_rag(path, page_content, [file_id])
        pages_created += 1

    if (index_updates and index_updates.get('additions')) or log_entry:
        try:
            from filelock import FileLock
            lock_path = os.path.join(wiki_engine.WIKI_DIR, 'index.md.lock')
            _have_filelock = True
        except ImportError:
            logger.warning("filelock not installed, wiki ingest may have race conditions")
            _have_filelock = False
            lock_path = None

        def _do_index_updates():
            if index_updates and index_updates.get('additions'):
                existing_fm, existing_content = wiki_engine.read_wiki_page('index.md')
                additions = index_updates['additions']
                new_lines = []
                for a in additions:
                    name = a.get('name', '')
                    desc = a.get('description', '')
                    path = a.get('path', '')
                    new_lines.append(f"- [{name}]({path}) {desc}")
                if new_lines:
                    new_index_content = existing_content.strip() + '\n\n' + '\n'.join(new_lines)
                    wiki_engine.write_wiki_page('index.md', existing_fm, new_index_content)

            if log_entry:
                existing_fm, existing_log = wiki_engine.read_wiki_page('log.md')
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                new_log = f"{existing_log.strip()}\n- {timestamp} | 源文件: {filename} (id={file_id}) | {log_entry}"
                wiki_engine.write_wiki_page('log.md', existing_fm, new_log)

        if _have_filelock:
            with FileLock(lock_path, timeout=10):
                _do_index_updates()
        else:
            _do_index_updates()

    return {
        'pages_updated': pages_updated,
        'pages_created': pages_created,
        'log_entry': log_entry,
    }


def batch_ingest_all(db_conn=None):
    from psycopg2.extras import RealDictCursor

    result = {
        'total': 0,
        'succeeded': 0,
        'failed': 0,
        'errors': [],
    }

    tables = [
        ('knowledge_lab', 'knowledge_lab_files', 'id, content, original_name'),
        ('company_kb', 'company_knowledge_base', 'id, content, original_name'),
    ]

    close_conn = False
    if db_conn is None:
        from app.database import get_db_connection
        db_conn = get_db_connection()
        close_conn = True

    try:
        with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            for source_type, table, cols in tables:
                try:
                    cur.execute(
                        sql.SQL("SELECT {} FROM {} WHERE content IS NOT NULL AND content != ''")
                        .format(sql.SQL(cols), sql.Identifier(table))
                    )
                    rows = cur.fetchall()
                    for row in rows:
                        result['total'] += 1
                        file_id = row['id']
                        content = row['content']
                        original_name = row.get('original_name', '')
                        try:
                            ingest_file(
                                file_id=file_id,
                                content=content,
                                filename=original_name,
                                source_type=source_type,
                                metadata={'original_name': original_name},
                            )
                            result['succeeded'] += 1
                        except Exception as e:
                            logger.warning(
                                f"Failed to ingest {table}.{file_id}: {e}"
                            )
                            result['failed'] += 1
                            result['errors'].append({
                                'file_id': file_id,
                                'source': source_type,
                                'error': str(e),
                            })
                except Exception as e:
                    logger.warning(f"Failed to query {table}: {e}")
                    result['errors'].append({
                        'file_id': None,
                        'source': source_type,
                        'error': str(e),
                    })
    finally:
        if close_conn:
            db_conn.close()

    return result


try:
    from celery_app import celery as celery_app

    @celery_app.task(bind=True, name='wiki_ingest_task', max_retries=2, acks_late=True)
    def wiki_ingest_task(self, file_id, content, filename, source_type, metadata=None):
        from app.services.wiki_ingest import init_flask_context
        init_flask_context()
        return ingest_file(file_id, content, filename, source_type, metadata)

    @celery_app.task(bind=True, name='wiki_batch_ingest_task', max_retries=1)
    def wiki_batch_ingest_task(self):
        from app.services.wiki_ingest import init_flask_context
        init_flask_context()
        return batch_ingest_all()

except Exception:
    pass
