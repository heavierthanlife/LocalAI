"""Recycle bin CRUD — unified service for all 4 recycle bin tables.

Eliminates 4× parallel if/elif blocks in chat.py routes.
"""
import json
import logging
import os
from psycopg2 import sql
from app.config import resolve_path

logger = logging.getLogger(__name__)

TABLE_MAP = {
    'chat': 'recycle_bin',
    'knowledge_lab': 'kb_recycle_bin',
    'project': 'project_recycle_bin',
    'project_files': 'project_recycle_bin',
    'folder': 'project_folders_recycle_bin',
    'project_folders': 'project_folders_recycle_bin',
}

STORED_PATH_COLUMNS = {
    'recycle_bin': 'original_stored_path',
    'kb_recycle_bin': 'stored_path',
    'project_recycle_bin': 'stored_path',
    'project_folders_recycle_bin': None,
}


def _resolve_table(source: str) -> str:
    table = TABLE_MAP.get(source)
    if not table:
        raise ValueError(f"Unknown recycle bin source: {source}")
    return table


def _delete_physical_file(table: str, row):
    """Delete the physical file associated with a recycle bin row, if any."""
    col = STORED_PATH_COLUMNS.get(table)
    if col and row and row[0] and os.path.exists(resolve_path(row[0])):
        try:
            os.remove(resolve_path(row[0]))
        except OSError as e:
            logger.warning(f"Failed to delete {row[0]}: {e}")


def restore_recycle_item(item_id: str, source: str, conn, cur, user_id: str = None):
    """Route restore to the correct table and helper logic.

    Returns True on success, raises on error.
    """
    from psycopg2.extras import RealDictCursor

    if source == 'chat':
        cur.execute("SELECT * FROM recycle_bin WHERE id = %s AND user_id = %s AND expires_at > NOW()",
                    (item_id, user_id))
        item = cur.fetchone()
        if not item:
            raise ValueError("Item not found or expired")
        meta_data = {}
        if item.get('deletion_reason') == 'chat_deleted':
            meta_data['restored_from'] = 'chat_deletion'
            meta_data['original_thread_id'] = item.get('original_thread_id')
        meta_data_json = json.dumps(meta_data)
        cur.execute("""
            INSERT INTO user_files (user_id, thread_id, filename, content, size_bytes, expires_at,
                                    original_stored_path, file_hash, meta_data, original_name)
            VALUES (%s, %s, %s, %s, %s, NOW() + INTERVAL '3 days', %s, %s, %s::jsonb, %s)
        """, (user_id, None, item['file_name'], item['file_content'], item['file_size'],
              item['original_stored_path'], item['file_hash'], meta_data_json, item['file_name']))
        cur.execute("DELETE FROM recycle_bin WHERE id = %s", (item_id,))

    elif source == 'knowledge_lab':
        cur.execute("SELECT * FROM kb_recycle_bin WHERE id = %s AND expires_at > NOW()", (item_id,))
        item = cur.fetchone()
        if not item:
            raise ValueError("Item not found or expired")
        _restore_kb_item(item, conn, cur)

    elif source == 'project':
        cur.execute("SELECT * FROM project_recycle_bin WHERE id = %s", (item_id,))
        item = cur.fetchone()
        if not item:
            raise ValueError("Item not found")
        _restore_project_file(item, conn, cur)

    elif source == 'folder':
        cur.execute("SELECT * FROM project_folders_recycle_bin WHERE id = %s", (item_id,))
        folder = cur.fetchone()
        if not folder:
            raise ValueError("Folder not found")
        from app.services._shared_helpers import restore_folder_recursive
        restore_folder_recursive(folder, conn, cur)

    else:
        raise ValueError(f"Unknown source: {source}")


def permanently_delete_item(item_id: str, source: str, cur, user_id: str = None):
    """Permanently delete a recycle bin item (physical file + DB row)."""
    table = _resolve_table(source)

    if source == 'folder':
        # Folders need recursive cleanup
        _permanently_delete_folder(item_id, cur)
        return

    _known_tables = set(TABLE_MAP.values())
    _known_cols = set(STORED_PATH_COLUMNS.values()) - {None}
    _col = STORED_PATH_COLUMNS.get(table)
    if not _col or _col not in _known_cols or table not in _known_tables:
        raise ValueError(f"Unexpected table/column: {table}.{_col}")
    user_col = 'uploaded_by' if table == 'project_recycle_bin' else 'user_id'
    if user_id:
        cur.execute(sql.SQL("SELECT {} FROM {} WHERE id = %s AND {} = %s")
                    .format(sql.Identifier(_col), sql.Identifier(table), sql.Identifier(user_col)),
                    (item_id, user_id))
    else:
        cur.execute(sql.SQL("SELECT {} FROM {} WHERE id = %s")
                    .format(sql.Identifier(_col), sql.Identifier(table)),
                    (item_id,))
    row = cur.fetchone()
    _delete_physical_file(table, row)
    if user_id:
        cur.execute(sql.SQL("DELETE FROM {} WHERE id = %s AND {} = %s")
                    .format(sql.Identifier(table), sql.Identifier(user_col)),
                    (item_id, user_id))
    else:
        cur.execute(sql.SQL("DELETE FROM {} WHERE id = %s")
                    .format(sql.Identifier(table)),
                    (item_id,))


def empty_recycle_bin(source: str, user_id: str, cur):
    """Empty all items in a recycle bin section (physical files + DB rows)."""
    if source == 'user_file':
        source = 'chat'
    elif source == 'company_kb':
        source = 'knowledge_lab'

    if source == 'chat' or source == 'all':
        cur.execute("SELECT original_stored_path FROM recycle_bin WHERE user_id = %s", (user_id,))
        for (path,) in cur.fetchall():
            if path and os.path.exists(resolve_path(path)):
                try:
                    os.remove(resolve_path(path))
                except OSError:
                    pass
        cur.execute("DELETE FROM recycle_bin WHERE user_id = %s", (user_id,))

    if source == 'project_files' or source == 'all':
        cur.execute("SELECT stored_path FROM project_recycle_bin")
        for (path,) in cur.fetchall():
            if path and os.path.exists(resolve_path(path)):
                try:
                    os.remove(resolve_path(path))
                except OSError:
                    pass
        cur.execute("DELETE FROM project_recycle_bin")

    if source == 'project_folders' or source == 'all':
        cur.execute("DELETE FROM project_folders_recycle_bin")

    if source == 'knowledge_lab' or source == 'all':
        cur.execute("SELECT stored_path FROM kb_recycle_bin")
        for (path,) in cur.fetchall():
            if path and os.path.exists(resolve_path(path)):
                try:
                    os.remove(resolve_path(path))
                except OSError:
                    pass
        cur.execute("DELETE FROM kb_recycle_bin")


def bulk_restore_all(section: str, user_id: str, conn, cur):
    """Restore all items in a section."""
    from psycopg2.extras import RealDictCursor
    restored_count = 0

    if section == 'knowledge_lab':
        cur.execute("SELECT * FROM kb_recycle_bin WHERE expires_at > NOW()")
        for item in cur.fetchall():
            _restore_kb_item(item, conn, cur)
            restored_count += 1

    elif section == 'chat':
        cur.execute("SELECT * FROM recycle_bin WHERE user_id = %s AND expires_at > NOW()", (user_id,))
        for item in cur.fetchall():
            meta_data = {}
            if item.get('deletion_reason') == 'chat_deleted':
                meta_data['restored_from'] = 'chat_deletion'
                meta_data['original_thread_id'] = item.get('original_thread_id')
            meta_data_json = json.dumps(meta_data)
            cur.execute("""
                INSERT INTO user_files (user_id, thread_id, filename, content, size_bytes, expires_at,
                                        original_stored_path, file_hash, meta_data, original_name)
                VALUES (%s, %s, %s, %s, %s, NOW() + INTERVAL '3 days', %s, %s, %s::jsonb, %s)
            """, (user_id, None, item['file_name'], item['file_content'], item['file_size'],
                  item['original_stored_path'], item['file_hash'], meta_data_json, item['file_name']))
            cur.execute("DELETE FROM recycle_bin WHERE id = %s", (item['id'],))
            restored_count += 1

    elif section == 'project_files':
        cur.execute("SELECT * FROM project_recycle_bin WHERE expires_at > NOW()")
        for item in cur.fetchall():
            _restore_project_file(item, conn, cur)
            restored_count += 1

    elif section == 'project_folders':
        cur.execute("SELECT * FROM project_folders_recycle_bin WHERE expires_at > NOW()")
        for folder in cur.fetchall():
            from app.services._shared_helpers import restore_folder_recursive
            restore_folder_recursive(folder, conn, cur)
            restored_count += 1

    return restored_count


# ── Internal helpers ──


def _restore_kb_item(item, conn, cur):
    """Restore a single KB file from kb_recycle_bin back to its original table."""
    orig_table = item.get('original_table', 'knowledge_lab_files')
    file_content = item.get('file_content') or item.get('content') or ''
    if orig_table == 'knowledge_lab_files':
        cur.execute("""
            INSERT INTO knowledge_lab_files (id, user_id, filename, original_name, file_size, content,
                                             file_hash, stored_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                filename = EXCLUDED.filename, original_name = EXCLUDED.original_name,
                file_size = EXCLUDED.file_size, content = EXCLUDED.content,
                file_hash = EXCLUDED.file_hash, stored_path = EXCLUDED.stored_path,
                updated_at = NOW()
        """, (item['original_id'], item['user_id'], item['filename'], item['original_name'],
              item['file_size'], file_content, item['file_hash'], item.get('stored_path')))
    elif orig_table == 'company_knowledge_base':
        cur.execute("""
            INSERT INTO company_knowledge_base (id, filename, original_name, file_size, content,
                                                file_hash, stored_path, category, uploaded_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                filename = EXCLUDED.filename, original_name = EXCLUDED.original_name,
                file_size = EXCLUDED.file_size, content = EXCLUDED.content,
                file_hash = EXCLUDED.file_hash, stored_path = EXCLUDED.stored_path,
                category = EXCLUDED.category, updated_at = NOW()
        """, (item['original_id'], item['filename'], item['original_name'],
              item['file_size'], file_content, item['file_hash'], item.get('stored_path'),
              item.get('category', ''), item.get('uploaded_by')))
    cur.execute("DELETE FROM kb_recycle_bin WHERE id = %s", (item['id'],))


def _restore_project_file(item, conn, cur):
    """Restore a single project file from recycle bin."""
    folder_id = item['folder_id']
    if folder_id:
        cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s",
                    (folder_id, item['project_id']))
        if not cur.fetchone():
            from app.services._shared_helpers import restore_folder_path_for_file
            restore_folder_path_for_file(item, conn, cur)
    cur.execute("""
        INSERT INTO project_files (project_id, folder_id, filename, original_name, file_size,
                                   stored_path, version, uploaded_by, file_hash)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (item['project_id'], item['folder_id'], item['file_name'], item['original_name'],
          item['file_size'], item['stored_path'], item['version'],
          item['uploaded_by'], item['file_hash']))
    cur.execute("DELETE FROM project_recycle_bin WHERE id = %s", (item['id'],))


def _permanently_delete_folder(item_id: str, cur):
    """Permanently delete a folder and all its files from recycle bin."""
    cur.execute("SELECT project_id, original_id FROM project_folders_recycle_bin WHERE id = %s", (item_id,))
    folder = cur.fetchone()
    if folder:
        project_id = folder[0]
        original_folder_id = folder[1]
        cur.execute("SELECT stored_path FROM project_recycle_bin WHERE project_id = %s AND folder_id = %s",
                    (project_id, original_folder_id))
        for (stored_path,) in cur.fetchall():
            if stored_path and os.path.exists(resolve_path(stored_path)):
                try:
                    os.remove(resolve_path(stored_path))
                except OSError:
                    pass
        cur.execute("DELETE FROM project_recycle_bin WHERE project_id = %s AND folder_id = %s",
                    (project_id, original_folder_id))
        cur.execute("""
            WITH RECURSIVE folder_tree AS (
                SELECT id, original_id, project_id, original_parent_id
                FROM project_folders_recycle_bin
                WHERE id = %s
                UNION ALL
                SELECT pf.id, pf.original_id, pf.project_id, pf.original_parent_id
                FROM project_folders_recycle_bin pf
                INNER JOIN folder_tree ft ON pf.original_parent_id = ft.original_id AND pf.project_id = ft.project_id
            )
            SELECT id, original_id FROM folder_tree
        """, (item_id,))
        subfolders = cur.fetchall()
        for (sf_id, sf_orig_id) in subfolders:
            cur.execute("SELECT stored_path FROM project_recycle_bin WHERE project_id = %s AND folder_id = %s",
                        (project_id, sf_orig_id))
            for (sp,) in cur.fetchall():
                if sp and os.path.exists(resolve_path(sp)):
                    try:
                        os.remove(resolve_path(sp))
                    except OSError:
                        pass
            cur.execute("DELETE FROM project_recycle_bin WHERE project_id = %s AND folder_id = %s",
                        (project_id, sf_orig_id))
            cur.execute("DELETE FROM project_folders_recycle_bin WHERE id = %s", (sf_id,))
    else:
        cur.execute("DELETE FROM project_folders_recycle_bin WHERE id = %s", (item_id,))


def get_recycle_items(user_id: str, cur) -> dict:
    """Query all 4 recycle bin tables and return items grouped by source.

    Returns dict with 4 keys: chat_items, project_items, folder_items, kb_items.
    Each value is a list of RealDictRow objects matching the frontend schema.
    """
    from psycopg2.extras import RealDictCursor

    cur = cur.connection.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT rb.id,
               rb.original_table,
               rb.original_id,
               rb.file_name,
               rb.file_size,
               rb.deleted_at,
               rb.expires_at,
               rb.deletion_reason,
               'chat'              as source,
               u_uploader.username as uploaded_by_name,
               u_deleter.username  as deleted_by_name
        FROM recycle_bin rb
                 LEFT JOIN users u_uploader ON rb.uploaded_by = u_uploader.user_id
                 LEFT JOIN users u_deleter ON rb.deleted_by = u_deleter.user_id
        WHERE rb.user_id = %s
          AND rb.expires_at > NOW()
        ORDER BY rb.deleted_at DESC
    """, (user_id,))
    chat_items = cur.fetchall()

    cur.execute("""
        SELECT prb.id,
               prb.original_table,
               prb.original_id,
               prb.file_name,
               prb.file_size,
               prb.deleted_at,
               prb.expires_at,
               p.name              as project_name,
               'project'           as source,
               u_uploader.username as uploaded_by_name,
               u_deleter.username  as deleted_by_name
        FROM project_recycle_bin prb
                 JOIN projects p ON prb.project_id = p.id
                 LEFT JOIN users u_uploader ON prb.uploaded_by = u_uploader.user_id
                 LEFT JOIN users u_deleter ON prb.deleted_by = u_deleter.user_id
        WHERE prb.expires_at > NOW()
        ORDER BY prb.deleted_at DESC
    """)
    project_items = cur.fetchall()

    cur.execute("""
        SELECT pfrb.id, pfrb.original_id, pfrb.name, pfrb.original_parent_id,
               pfrb.deleted_at, pfrb.expires_at,
               p.name as project_name, 'folder' as source
        FROM project_folders_recycle_bin pfrb
        JOIN projects p ON pfrb.project_id = p.id
        WHERE pfrb.expires_at > NOW()
        ORDER BY pfrb.deleted_at DESC
    """)
    folder_items = cur.fetchall()

    cur.execute("""
        SELECT kbr.id, kbr.original_table, kbr.original_id, kbr.filename, kbr.original_name,
               kbr.file_size, kbr.deleted_at, kbr.expires_at, kbr.category,
               kbr.skill_summary,
               CASE WHEN kbr.original_table = 'knowledge_lab_files' THEN 'knowledge_lab'
                    ELSE 'company_kb' END as source,
               u_uploader.username as uploaded_by_name,
               u_deleter.username as deleted_by_name
        FROM kb_recycle_bin kbr
                 LEFT JOIN users u_uploader ON kbr.uploaded_by = u_uploader.user_id
                 LEFT JOIN users u_deleter ON kbr.deleted_by = u_deleter.user_id
        WHERE kbr.expires_at > NOW()
        ORDER BY kbr.deleted_at DESC
    """)
    kb_items = cur.fetchall()

    return {
        "chat_items": chat_items,
        "project_items": project_items,
        "folder_items": folder_items,
        "kb_items": kb_items,
    }
