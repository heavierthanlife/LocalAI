"""Blueprint: wiki (Obsidian-flavored markdown wiki)."""
import logging
import os
import re
import hashlib
from flask import Blueprint, request, session
from app.utils.helpers import ok, err
from app.database import get_db_connection

logger = logging.getLogger(__name__)

wiki_bp = Blueprint('wiki', __name__)


def _render_markdown(text):
    import markdown as md
    return md.markdown(text, extensions=['extra', 'codehilite', 'toc'])


@wiki_bp.route('/wiki/index')
def wiki_index():
    try:
        from app.services.wiki_engine import read_wiki_index, list_wiki_tree, get_recent_wiki_pages
        prefix = request.args.get('prefix', '').strip()
        pages = read_wiki_index()
        tree = list_wiki_tree(prefix=prefix)
        recent = get_recent_wiki_pages(5)
        return ok({"pages": pages, "tree": tree, "recent": recent})
    except Exception as e:
        logger.error(f"wiki_index failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


def _resolve_page_path(page_path):
    page_path = page_path.replace('\\', '/')
    return page_path if page_path.endswith('.md') else page_path + '.md'


@wiki_bp.route('/wiki/page/<path:page_path>')
def wiki_page(page_path):
    try:
        from app.services.wiki_engine import read_wiki_page
        frontmatter, content, full_path = read_wiki_page(_resolve_page_path(page_path))
        if not full_path or not content:
            return err("Page not found", "NOT_FOUND", 404)
        if frontmatter.get('type') in ('legal_law', 'regulation', 'template', 'case', 'document'):
            from html import escape
            html = '<pre style="white-space:pre-wrap;word-wrap:break-word;font-family:inherit;font-size:0.78rem;line-height:1.6;">' + escape(content) + '</pre>'
        elif frontmatter.get('type') in ('legal_index', 'category_index'):
            html = content
        else:
            html = _render_markdown(content)
        return ok({"frontmatter": frontmatter, "content": content, "html": html})
    except Exception as e:
        logger.error(f"wiki_page failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@wiki_bp.route('/wiki/tree')
def wiki_tree():
    try:
        from app.services.wiki_engine import list_wiki_tree
        prefix = request.args.get('prefix', '').strip()
        tree = list_wiki_tree(prefix=prefix)
        return ok({"tree": tree})
    except Exception as e:
        logger.error(f"wiki_tree failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@wiki_bp.route('/wiki/search')
def wiki_search():
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return ok({"results": []})
        from app.services.wiki_engine import search_wiki
        results = search_wiki(query)
        return ok({"results": results})
    except Exception as e:
        logger.error(f"wiki_search failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@wiki_bp.route('/wiki/page/<path:page_path>/raw')
def wiki_page_raw(page_path):
    try:
        from app.services.wiki_engine import read_wiki_page
        frontmatter, content, full_path = read_wiki_page(_resolve_page_path(page_path))
        if not full_path or not content:
            return err("Page not found", "NOT_FOUND", 404)
        return ok({"frontmatter": frontmatter, "content": content})
    except Exception as e:
        logger.error(f"wiki_page_raw failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@wiki_bp.route('/wiki/stats')
def wiki_stats():
    try:
        from app.services.wiki_engine import get_wiki_stats
        stats = get_wiki_stats()
        return ok({"stats": stats})
    except Exception as e:
        logger.error(f"wiki_stats failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@wiki_bp.route('/wiki/legal/import', methods=['POST'])
def wiki_legal_import():
    if session.get('role', 'user') != 'admin':
        return err("需要管理员权限", "FORBIDDEN", 403)

    files = request.files.getlist('files')
    if not files or all(not f.filename for f in files):
        return err("请选择文件", "VALIDATION_ERROR", 400)

    from app.services.wiki_engine import write_wiki_page, record_origin_link, _ensure_wiki_dir, WIKI_DIR
    from app.services.file_processing import extract_text_from_file
    from app.services.document_classifier import classify_and_categorize
    _ensure_wiki_dir()

    imported = []
    errors = []

    for file in files:
        if not file.filename:
            continue
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ('.doc', '.docx', '.pdf'):
            errors.append(f"{file.filename}: 不支持的文件格式")
            continue

        try:
            from app.services.document_parser import process_file, to_wiki_markdown
            doc = process_file(file, file.filename)
            if not doc or not doc.get("sections"):
                errors.append(f"{file.filename}: 无法提取文本内容")
                continue

            text_content = "\n".join(s.get("content", "") for s in doc.get("sections", []))
            file_bytes = file.read()
            file_hash = hashlib.sha256(file_bytes or b"").hexdigest()
            file.seek(0)
            _, wiki_category = classify_and_categorize(text_content, file.filename, file_hash)
            doc_type = wiki_category

            frontmatter_type = "legal_law" if wiki_category == "laws" else wiki_category
            text = to_wiki_markdown(doc)

            safe_name = re.sub(r'[^\w\u4e00-\u9fff.-]', '_', file.filename.rsplit('.', 1)[0])
            safe_name = safe_name.strip('_')[:80]
            wiki_path = f"{wiki_category}/{safe_name}.md"

            write_wiki_page(wiki_path, {'title': file.filename, 'type': frontmatter_type}, text)
            record_origin_link(wiki_path, f'{wiki_category}_import', 0, file.filename)

            try:
                from app.services.wiki_entity_service import process_upload_entity_extraction
                import threading
                def _do_extract():
                    try:
                        process_upload_entity_extraction(
                            0, text_content, file.filename, 'wiki_legal_import',
                            doc_type, wiki_category,
                            {'original_name': file.filename},
                        )
                    except Exception as ex:
                        logger.warning(f"Entity extract failed during legal import: {ex}")
                t = threading.Thread(target=_do_extract, daemon=True)
                t.start()
            except Exception as ex:
                logger.warning(f"Failed to dispatch entity extraction: {ex}")

            imported.append({'path': wiki_path, 'filename': file.filename, 'category': wiki_category})
        except Exception as e:
            logger.warning(f"Legal import failed for {file.filename}: {e}")
            errors.append(f"{file.filename}: {e}")

    if imported:
        _rebuild_category_indices()

    return ok({'imported': imported, 'errors': errors}, f"导入 {len(imported)} 个文件，{len(errors)} 个错误")


@wiki_bp.route('/wiki/page/<path:page_path>', methods=['PUT'])
def wiki_page_update(page_path):
    if session.get('role', 'user') != 'admin':
        return err("需要管理员权限", "FORBIDDEN", 403)

    data = request.get_json(silent=True) or {}
    new_content = data.get('content')
    if new_content is None:
        return err("请提供 content", "VALIDATION_ERROR", 400)

    try:
        from app.services.wiki_engine import read_wiki_page, write_wiki_page
        frontmatter, _, full_path = read_wiki_page(_resolve_page_path(page_path))
        if not full_path:
            return err("Page not found", "NOT_FOUND", 404)
        write_wiki_page(_resolve_page_path(page_path), frontmatter or {}, new_content)
        return ok(None, "页面已更新")
    except Exception as e:
        logger.error(f"wiki_page_update failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@wiki_bp.route('/wiki/page/<path:page_path>', methods=['DELETE'])
def wiki_page_delete(page_path):
    if session.get('role', 'user') != 'admin':
        return err("需要管理员权限", "FORBIDDEN", 403)

    try:
        from app.services.wiki_engine import delete_wiki_page
        ok_result = delete_wiki_page(_resolve_page_path(page_path))
        if not ok_result:
            return err("Page not found", "NOT_FOUND", 404)
        return ok(None, "页面已删除")
    except Exception as e:
        logger.error(f"wiki_page_delete failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


def _rebuild_category_indices():
    from app.services.wiki_engine import read_wiki_page, write_wiki_page, WIKI_DIR

    category_config = {
        "laws": ("法律法规库", "📜"),
        "regulations": ("标准规范库", "📏"),
        "templates": ("模板库", "📄"),
        "cases": ("案例库", "📋"),
        "documents": ("文档库", "📊"),
    }

    for cat_dir, (label, icon) in category_config.items():
        cat_path = os.path.join(WIKI_DIR, cat_dir)
        if not os.path.isdir(cat_path):
            continue

        lines = [f'<h3>{label}</h3>', '<div style="margin-top:8px;">']
        for fname in sorted(os.listdir(cat_path)):
            if not fname.endswith('.md') or fname == 'index.md':
                continue
            try:
                fm, _ = read_wiki_page(f'{cat_dir}/{fname}')
                title = fm.get('title', fname) if fm else fname
                display = title.rsplit('.', 1)[0] if '.' in title else title
                path = f'{cat_dir}/{display}'
                lines.append(f'<div class="wiki-page-link" data-path="{path}" style="padding:6px 10px;background:var(--card-bg);border-radius:4px;margin-bottom:3px;cursor:pointer;font-size:.78rem;border:1px solid transparent;">{icon} {display}</div>')
            except Exception:
                path = f'{cat_dir}/{fname.replace(".md","")}'
                lines.append(f'<div class="wiki-page-link" data-path="{path}" style="padding:6px 10px;background:var(--card-bg);border-radius:4px;margin-bottom:3px;cursor:pointer;font-size:.78rem;border:1px solid transparent;">{icon} {fname.replace(".md","")}</div>')
        lines.append('</div>')
        write_wiki_page(f'{cat_dir}/index.md', {'title': label, 'type': 'category_index'}, ''.join(lines))


# ── Bookmarks ──
@wiki_bp.route('/wiki/bookmarks', methods=['GET'])
def wiki_bookmarks_list():
    user_id = session.get('user_id')
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, article_id, article_title, wiki_path, created_at
                    FROM wiki_bookmarks WHERE user_id = %s
                    ORDER BY created_at DESC
                """, (user_id,))
                rows = cur.fetchall()
                bookmarks = [{'id': r[0], 'article_id': r[1], 'article_title': r[2],
                              'wiki_path': r[3], 'created_at': str(r[4]) if r[4] else None}
                             for r in rows]
        return ok({"bookmarks": bookmarks})
    except Exception as e:
        return err(str(e), "SERVER_ERROR", 500)


@wiki_bp.route('/wiki/bookmarks', methods=['POST'])
def wiki_bookmark_add():
    user_id = session.get('user_id')
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)
    data = request.get_json(silent=True) or {}
    article_id = data.get('article_id', '').strip()
    article_title = data.get('article_title', '').strip()
    wiki_path = data.get('wiki_path', '').strip()
    if not article_id or not article_title:
        return err("article_id and article_title required", "VALIDATION_ERROR", 400)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO wiki_bookmarks (user_id, article_id, article_title, wiki_path)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, article_id) DO UPDATE SET article_title = EXCLUDED.article_title,
                        wiki_path = EXCLUDED.wiki_path, created_at = NOW()
                    RETURNING id
                """, (user_id, article_id, article_title, wiki_path or None))
                new_id = cur.fetchone()[0]
                conn.commit()
        return ok({"id": new_id, "message": "已收藏"})
    except Exception as e:
        return err(str(e), "SERVER_ERROR", 500)


@wiki_bp.route('/wiki/bookmarks/<int:bookmark_id>', methods=['DELETE'])
def wiki_bookmark_delete(bookmark_id):
    user_id = session.get('user_id')
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM wiki_bookmarks WHERE id = %s AND user_id = %s",
                            (bookmark_id, user_id))
                conn.commit()
        return ok({"message": "已取消收藏"})
    except Exception as e:
        return err(str(e), "SERVER_ERROR", 500)


# ── View log ──
@wiki_bp.route('/wiki/view-log', methods=['POST'])
def wiki_view_log():
    """Record a page view. Returns recent views. action=view to log, action=list for query."""
    user_id = session.get('user_id')
    if not user_id:
        return err("Not logged in", "AUTH_REQUIRED", 401)
    data = request.get_json(silent=True) or {}
    action = data.get('action', 'view')
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if action == 'view':
                    wiki_path = data.get('wiki_path', '').strip()
                    article_title = data.get('article_title', '').strip()
                    if wiki_path:
                        cur.execute("""
                            INSERT INTO wiki_view_log (user_id, wiki_path, article_title)
                            VALUES (%s, %s, %s)
                        """, (user_id, wiki_path, article_title or wiki_path))
                        conn.commit()
                limit = data.get('limit', 10)
                cur.execute("""
                    SELECT DISTINCT wiki_path, article_title, MAX(viewed_at) as last_viewed
                    FROM wiki_view_log WHERE user_id = %s
                    GROUP BY wiki_path, article_title
                    ORDER BY last_viewed DESC LIMIT %s
                """, (user_id, limit))
                rows = cur.fetchall()
                views = [{'wiki_path': r[0], 'article_title': r[1] or r[0],
                         'viewed_at': str(r[2]) if r[2] else None} for r in rows]
        return ok({"recent_views": views})
    except Exception as e:
        return err(str(e), "SERVER_ERROR", 500)


# ── Comparison ──
@wiki_bp.route('/wiki/compare')
def wiki_compare():
    page_a = request.args.get('page_a', '').strip()
    page_b = request.args.get('page_b', '').strip()
    if not page_a or not page_b:
        return err("page_a and page_b required", "VALIDATION_ERROR", 400)
    try:
        from app.services.wiki_entity_service import compare_wiki_pages
        result = compare_wiki_pages(page_a, page_b)
        if "error" in result:
            return err(result["error"], "NOT_FOUND", 404)
        return ok(result)
    except Exception as e:
        logger.error(f"wiki_compare failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@wiki_bp.route('/wiki/comparisons')
def wiki_comparisons_list():
    try:
        from app.services.wiki_engine import list_wiki_tree, read_wiki_page
        tree = list_wiki_tree(prefix="comparisons")
        files = []
        def _collect(node):
            if node.get("type") == "file":
                fm, _, _ = read_wiki_page(node["path"])
                stale = (fm or {}).get("stale", False) if fm else False
                files.append({
                    "path": node["path"].replace(".md", ""),
                    "filename": node["name"],
                    "stale": stale,
                })
            for child in node.get("children", []):
                _collect(child)
        _collect(tree)
        return ok({"comparisons": files})
    except Exception as e:
        logger.error(f"wiki_comparisons_list failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@wiki_bp.route('/wiki/comparisons/<path:comp_path>', methods=['DELETE'])
def wiki_comparison_delete(comp_path):
    if session.get('role', 'user') != 'admin':
        return err("需要管理员权限", "FORBIDDEN", 403)
    try:
        from app.services.wiki_engine import delete_wiki_page
        result = delete_wiki_page(f"comparisons/{comp_path}.md")
        if not result:
            return err("Comparison not found", "NOT_FOUND", 404)
        return ok(None, "对比已删除")
    except Exception as e:
        logger.error(f"wiki_comparison_delete failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


# ── Entity graph ──
@wiki_bp.route('/wiki/entity-graph')
def wiki_entity_graph():
    root = request.args.get('root', '').strip()
    if not root:
        return err("root entity path required", "VALIDATION_ERROR", 400)
    depth = int(request.args.get('depth', 2))
    try:
        from app.services.wiki_entity_service import get_entity_graph
        graph = get_entity_graph(root, depth=depth, max_nodes=50)
        return ok({"graph": graph})
    except Exception as e:
        logger.error(f"wiki_entity_graph failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


# ── Category pages ──
@wiki_bp.route('/wiki/category/<category_name>')
def wiki_category_pages(category_name):
    try:
        from app.services.wiki_entity_service import list_category_pages
        pages = list_category_pages(category_name)
        return ok({"pages": pages, "category": category_name})
    except Exception as e:
        logger.error(f"wiki_category_pages failed: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)
