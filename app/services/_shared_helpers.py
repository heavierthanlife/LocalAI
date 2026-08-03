"""Shared helpers extracted from app.routes.admin to break service→route dependency.

Admin re-exports these symbols for backward compatibility.
Services import directly from here.
"""

# ======================== AI Document Review ========================

AI_DOC_REVIEW_PROMPT = """你是一个资深的招标文件审查专家。请用以下五轴法审查这份文档，每个轴给1-10分，指出具体问题，最后给出综合评分和修改建议。

**审查五轴：**
1. **合规性** — 是否满足招标文件的所有硬性要求？有没有遗漏必需的资质、证书、签章？
2. **清晰度** — 语言是否清晰？结构和逻辑是否合理？数据和金额是否表述准确无歧义？
3. **完整性** — 所有必填部分是否齐全？技术方案、商务报价、资质证明是否完整？
4. **风险** — 是否存在不利条款、过高承诺、模糊免责声明、容易被质疑的计算或推理？
5. **专业性** — 格式、用词、排版是否专业？是否符合行业规范？

**输出格式（严格JSON，不要Markdown包裹）：
{
  "scores": {"合规性": N, "清晰度": N, "完整性": N, "风险": N, "专业性": N},
  "overall": N,
  "verdict": "通过 / 需修改 / 不合格",
  "issues": [
    {"axis": "轴名", "severity": "高/中/低", "location": "段落或位置描述", "finding": "具体问题", "suggestion": "修改建议"}
  ],
  "summary": "一段中文总结（100字内），概括主要问题和整体评价"
}"""


# ---------- Helper functions for recycle bin folder restoration ----------

def restore_folder_recursive(folder_item, conn, cur, target_parent_id=None):
    parent_id = target_parent_id if target_parent_id is not None else folder_item['original_parent_id']
    cur.execute("""
        INSERT INTO project_folders (id, project_id, parent_folder_id, name, created_at, created_by)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """, (folder_item['original_id'], folder_item['project_id'], parent_id,
          folder_item['name'], folder_item['created_at'], folder_item['created_by']))
    cur.execute("""
        SELECT * FROM project_recycle_bin
        WHERE project_id = %s AND folder_id = %s
    """, (folder_item['project_id'], folder_item['original_id']))
    files = cur.fetchall()
    for f in files:
        cur.execute("""
            INSERT INTO project_files (project_id, folder_id, filename, original_name, file_size, stored_path, version, uploaded_by, file_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (f['project_id'], folder_item['original_id'], f['file_name'], f['original_name'],
              f['file_size'], f['stored_path'], f['version'], f['uploaded_by'], f['file_hash']))
        cur.execute("DELETE FROM project_recycle_bin WHERE id = %s", (f['id'],))
    cur.execute("""
        SELECT * FROM project_folders_recycle_bin
        WHERE project_id = %s AND original_parent_id = %s
    """, (folder_item['project_id'], folder_item['original_id']))
    subfolders = cur.fetchall()
    for sf in subfolders:
        restore_folder_recursive(sf, conn, cur, target_parent_id=folder_item['original_id'])
    cur.execute("DELETE FROM project_folders_recycle_bin WHERE id = %s", (folder_item['id'],))


def restore_folder_path_for_file(file_item, conn, cur):
    folder_id = file_item['folder_id']
    if folder_id is None:
        return
    cur.execute("SELECT id FROM project_folders WHERE id = %s AND project_id = %s", (folder_id, file_item['project_id']))
    if cur.fetchone():
        return
    cur.execute("SELECT * FROM project_folders_recycle_bin WHERE original_id = %s AND project_id = %s",
                (folder_id, file_item['project_id']))
    folder = cur.fetchone()
    if not folder:
        return
    if folder['original_parent_id']:
        cur.execute("SELECT * FROM project_folders_recycle_bin WHERE original_id = %s AND project_id = %s",
                    (folder['original_parent_id'], file_item['project_id']))
        parent = cur.fetchone()
        if parent:
            restore_folder_path_for_file(parent, conn, cur)
    cur.execute("""
        INSERT INTO project_folders (id, project_id, parent_folder_id, name, created_at, created_by)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """, (folder['original_id'], folder['project_id'], folder['original_parent_id'],
          folder['name'], folder['created_at'], folder['created_by']))
