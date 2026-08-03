"""Wiki tree merge service — combines DB folder structure with wiki file pages."""
import logging
import os
from typing import Dict, List, Optional

from app.database import get_db_connection
from app.config import DATA_DIR

logger = logging.getLogger(__name__)

WIKI_DIR = os.path.join(DATA_DIR, 'wiki')


def get_merged_tree(project_id: int) -> dict:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, industry, bidding_category, bid_method, status FROM projects WHERE id = %s",
                (project_id,))
            row = cur.fetchone()
            if not row:
                return {'project_id': project_id, 'project_name': '', 'folders': [], 'files': []}
            project_name = row[0]

            cur.execute("""
                SELECT id, parent_folder_id, name
                FROM project_folders WHERE project_id = %s
                ORDER BY name
            """, (project_id,))
            folders_rows = cur.fetchall()

            cur.execute("""
                SELECT pf.id, pf.folder_id, pf.filename, pf.original_name,
                       pf.status, pf.version, pf.file_size, pf.mime_type,
                       pf.content IS NOT NULL AND pf.content != '' AS has_text
                FROM project_files pf
                WHERE pf.project_id = %s
                ORDER BY pf.filename
            """, (project_id,))
            files_rows = cur.fetchall()

    folders = []
    for fr in folders_rows:
        folders.append({
            'id': fr[0],
            'parent_id': fr[1],
            'name': fr[2],
            'children': [],
            'files': [],
        })

    folder_map = {f['id']: f for f in folders}
    root_folders = []
    for f in folders:
        if f['parent_id'] is None:
            root_folders.append(f)
        elif f['parent_id'] in folder_map:
            folder_map[f['parent_id']]['children'].append(f)

    wiki_files = _get_wiki_files(project_id, project_name)

    for file_row in files_rows:
        file_id = file_row[0]
        folder_id = file_row[1]
        filename = file_row[2]
        original_name = file_row[3]
        status = file_row[4]
        version = file_row[5]
        file_size = file_row[6]
        mime_type = file_row[7]
        has_text = file_row[8]

        wiki_entry = wiki_files.get(file_id)

        file_node = {
            'id': file_id,
            'name': original_name or filename,
            'filename': filename,
            'status': status or 'draft',
            'version': version,
            'file_size': file_size,
            'mime_type': mime_type,
            'has_text': has_text,
            'is_final': status == 'final',
            'wiki_path': wiki_entry.get('wiki_path') if wiki_entry else None,
            'has_wiki_page': wiki_entry is not None,
        }

        target = None
        if folder_id and folder_id in folder_map:
            target = folder_map[folder_id]
        else:
            target = None

        if target:
            target['files'].append(file_node)
        else:
            root_folders.append(file_node)

    return {
        'project_id': project_id,
        'project_name': project_name,
        'industry': row[1],
        'bidding_category': row[2],
        'bid_method': row[3],
        'project_status': row[4],
        'tree': _build_tree(root_folders),
    }


def _build_tree(items) -> list:
    result = []
    for item in items:
        if isinstance(item, dict):
            if 'children' in item or 'files' in item:
                node = {
                    'type': 'folder',
                    'id': item.get('id'),
                    'name': item.get('name', ''),
                    'children': _build_tree(item.get('children', [])) + item.get('files', []),
                }
                result.append(node)
            else:
                result.append({
                    'type': 'file',
                    'id': item.get('id'),
                    'name': item.get('name', ''),
                    'filename': item.get('filename', ''),
                    'status': item.get('status', 'draft'),
                    'version': item.get('version', 1),
                    'file_size': item.get('file_size', 0),
                    'mime_type': item.get('mime_type', ''),
                    'has_text': item.get('has_text', False),
                    'is_final': item.get('is_final', False),
                    'wiki_path': item.get('wiki_path'),
                    'has_wiki_page': item.get('has_wiki_page', False),
                })
    return result


def _get_wiki_files(project_id: int, project_name: str) -> dict:
    from app.services import wiki_engine

    result = {}
    try:
        wiki_engine._ensure_wiki_dir()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT wiki_page_path, source_file_id
                    FROM wiki_origin_links
                    WHERE source_type = 'project_file'
                      AND source_file_id IN (
                          SELECT id FROM project_files WHERE project_id = %s
                      )
                """, (project_id,))
                for row in cur.fetchall():
                    result[row[1]] = {'wiki_path': row[0]}

        return result
    except Exception as e:
        logger.warning(f"Failed to get wiki files for project {project_id}: {e}")
        return result
