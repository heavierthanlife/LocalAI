"""Publish project files as wiki pages — template-based, only when status=final."""
import logging
import os
import re
from datetime import datetime

from app.database import get_db_connection
from app.config import DATA_DIR

logger = logging.getLogger(__name__)

WIKI_DIR = os.path.join(DATA_DIR, 'wiki')
PROJECT_WIKI_DIR = "projects"

BINARY_MIME_PREFIXES = ('image/', 'video/', 'audio/', 'application/octet-stream')


def publish_project_file(file_id: int) -> str | None:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT pf.filename, pf.original_name, pf.content, pf.mime_type,
                           pf.stored_path, pf.status, pf.folder_id, pf.project_id,
                           pf.file_size, pf.created_at
                    FROM project_files pf
                    WHERE pf.id = %s
                """, (file_id,))
                row = cur.fetchone()
                if not row:
                    logger.warning(f"Project file {file_id} not found")
                    return None

                filename, original_name, content, mime_type, stored_path, status, folder_id, project_id, file_size, created_at = row

                if status != 'final':
                    logger.info(f"Project file {file_id} status={status}, skipping wiki publish")
                    return None

                if not content or not content.strip():
                    logger.info(f"Project file {file_id} has no text content, skipping wiki publish")
                    return None

                if mime_type and any(mime_type.startswith(p) for p in BINARY_MIME_PREFIXES):
                    logger.info(f"Project file {file_id} is binary ({mime_type}), skipping wiki publish")
                    return None

                project_name = None
                cur.execute("SELECT name FROM projects WHERE id = %s", (project_id,))
                prow = cur.fetchone()
                if prow:
                    project_name = prow[0]

                folder_path = ''
                if folder_id:
                    folder_path = _get_folder_path(cur, folder_id)

                return _do_publish(
                    file_id=file_id,
                    project_id=project_id,
                    project_name=project_name or f'project_{project_id}',
                    folder_path=folder_path,
                    filename=filename or original_name or f'file_{file_id}',
                    content=content,
                    mime_type=mime_type or '',
                    file_size=file_size or 0,
                    created_at=created_at,
                )

    except Exception as e:
        logger.warning(f"Wiki publish failed for project file {file_id}: {e}", exc_info=True)
        return None


def set_status_and_publish(file_id: int, status: str) -> bool:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE project_files SET status = %s WHERE id = %s",
                (status, file_id))
            conn.commit()
            if status == 'final':
                publish_project_file(file_id)
            elif status == 'draft':
                unpublish_project_file(file_id)
            return cur.rowcount > 0


def unpublish_project_file(file_id: int) -> bool:
    try:
        from app.services import wiki_engine

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT wiki_page_path FROM wiki_origin_links WHERE source_type = 'project_file' AND source_file_id = %s",
                    (file_id,))
                rows = cur.fetchall()
                for row in rows:
                    wiki_path = row[0]
                    wiki_engine.delete_wiki_page(wiki_path)

                cur.execute(
                    "DELETE FROM wiki_origin_links WHERE source_type = 'project_file' AND source_file_id = %s",
                    (file_id,))
                conn.commit()
        return True
    except Exception as e:
        logger.warning(f"Unpublish failed for file {file_id}: {e}")
        return False


def _do_publish(file_id: int, project_id: int, project_name: str, folder_path: str,
                filename: str, content: str, mime_type: str, file_size: int,
                created_at: datetime) -> str:
    from app.services import wiki_engine

    project_dir = _slugify(f"{project_name}_{project_id}")
    folder_dir = _slugify(folder_path) if folder_path else 'root'
    file_slug = _slugify(filename.replace('.md', '').replace('.MD', ''))

    wiki_path = f"{PROJECT_WIKI_DIR}/{project_dir}/{folder_dir}/{file_slug}.md"

    size_display = _format_size(file_size)
    created_str = created_at.isoformat() if created_at else datetime.now().isoformat()

    frontmatter = {
        'title': filename,
        'type': 'project_file',
        'file_id': file_id,
        'project_id': project_id,
        'project_name': project_name,
        'folder_path': folder_path,
        'filename': filename,
        'mime_type': mime_type,
        'file_size': file_size,
        'status': 'final',
        'created_at': created_str,
    }

    lines = [
        f"# {filename}",
        f"",
        f"**项目:** {project_name} | **文件夹:** {folder_path or '/'}",
        f"**文件类型:** {mime_type} | **大小:** {size_display}",
        f"**上传时间:** {created_str}",
        f"",
        f"## 文件操作",
        f"",
        f"- [\U0001f4e5 下载文件](/admin/projects/{project_id}/files/{file_id}/download)",
        f"- [\U0001f4cb 查看版本历史](/admin/projects/{project_id}/files/{file_id}/versions)",
        f"",
        f"## 文件内容",
        f"",
        f"<details>",
        f"<summary>\U0001f4c4 查看原文 (点击展开)</summary>",
        f"",
        f"```",
        content[:50000],
        f"```",
        f"",
        f"</details>",
        f"",
        f"---",
        f"*由系统自动生成*",
    ]

    page_content = '\n'.join(lines)
    wiki_engine.write_wiki_page(wiki_path, frontmatter, page_content)

    wiki_engine.record_origin_link(
        wiki_path=wiki_path,
        source_type='project_file',
        source_file_id=file_id,
        source_name=filename,
    )

    logger.info(f"Project file {file_id} published to wiki: {wiki_path}")
    return wiki_path


def _get_folder_path(cur, folder_id: int) -> str:
    parts = []
    current = folder_id
    visited = set()
    while current and current not in visited:
        visited.add(current)
        cur.execute("SELECT name, parent_folder_id FROM project_folders WHERE id = %s", (current,))
        row = cur.fetchone()
        if not row:
            break
        parts.insert(0, row[0])
        current = row[1]
    return '/'.join(parts)


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1048576:
        return f"{size / 1024:.1f} KB"
    if size < 1073741824:
        return f"{size / 1048576:.1f} MB"
    return f"{size / 1073741824:.1f} GB"


def _slugify(name: str) -> str:
    s = re.sub(r'[^\w\u4e00-\u9fff-]', '_', name)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')[:50]
