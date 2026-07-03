"""Blueprint: batch routes (auto-extracted)."""
import os, json, uuid, time, logging, hashlib, io, html
from datetime import datetime, timezone
from io import BytesIO
from flask import Blueprint, request, jsonify, session, send_file, render_template, url_for

from app.config import BASE_DIR, DATA_DIR, TEMP_ROOT, TEMP_DIR, USER_FILES_ORIGINAL_ROOT, is_valid_extracted_text, allowed_file
from app.database import get_db_connection, db_transaction
from app.utils.helpers import utc_now, beijing_now, safe_error_response, split_thinking_answer
import app.globals as g
from app.services.file_cache import file_cache_manager, add_to_cache, load_cache_from_db

logger = logging.getLogger(__name__)
from app.services.session_manager import get_user_id, get_or_create_session, ensure_user_exists, store_message, record_file_usage
from app.services.task_locking import acquire_task_lock, release_task_lock

import secrets, os as _os
from openpyxl import Workbook
from sklearn.metrics.pairwise import cosine_similarity
from app.services.file_processing import (
    extract_text_from_file, compute_similarity_with_numbers,
    compute_batch_semantic_similarity, extract_metadata, file_attr_similarity,
    extract_images_from_file, image_similarity, preprocess_text_for_similarity,
    remove_template_content, keyword_overlap_similarity, extract_keywords,
    truncate_filename, get_or_extract_file_analysis,
)
from app.services.batch_compare_svc import (
    _precompute_tfidf_for_files, _compute_pair_similarity_from_matrix,
    store_batch_comparison_temp, load_batch_comparison_temp
)

batch_bp = Blueprint('batch', __name__, template_folder=str(BASE_DIR / 'templates'), static_folder=str(BASE_DIR / 'static'))

@batch_bp.route('/compare_batch', methods=['POST'])
def compare_batch():
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({"error": "Need at least 2 files for comparison"}), 400
    if len(files) > 10:
        return jsonify({"error": "Maximum 10 files allowed"}), 400
    for f in files:
        if not allowed_file(f.filename):
            return jsonify({"error": f"不支持的文件类型: {f.filename}"}), 400

    user_id = get_user_id()
    thread_id = session['thread_id']
    success, busy_thread, busy_name = acquire_task_lock(user_id, thread_id, 'batch_compare')
    get_or_create_session(thread_id)
    if not success:
        return jsonify({
            "error": "resource_busy",
            "busy_chat": busy_name,
            "message": f"另一个资源密集型任务正在聊天“{busy_name}”中进行，请稍后再试。"
        }), 409

    try:
        template_file = request.files.get('template_file')
        template_text = None
        if template_file and template_file.filename:
            if not allowed_file(template_file.filename):
                return jsonify({"error": f"不支持的文件类型: {template_file.filename}"}), 400
            if session.get('consent_value', 0) == 1:
                template_text = get_or_extract_file_analysis(template_file, 'chat', user_id, thread_id=thread_id)
            else:
                template_text, _ = extract_text_from_file(template_file)
            if template_text and not template_text.startswith("["):
                if not is_valid_extracted_text(template_text):
                    template_text = None   # invalid template, ignore
                else:
                    record_file_usage(thread_id, template_file.filename, 'template_upload', "上传模板文件用于对比")

        check_items_json = request.form.get('check_items', '{}')
        try:
            check_items = json.loads(check_items_json)
        except (json.JSONDecodeError, TypeError):
            check_items = {}

        defaults = {
            'text_sim': True,
            'key_info': True,
            'file_attr': True,
            'image_sim': True,
            'semantic': False
        }
        for k, v in defaults.items():
            if k not in check_items:
                check_items[k] = v

        # ── Audit: detailed component participation log ──
        from app.services.audit_logger import AuditLogger
        _audit = AuditLogger("batch_compare", thread_id[:12] if thread_id else "")
        _audit.component("init", status="OK",
                         file_count=len(files), check_items=str(check_items),
                         has_template=bool(template_text))

        if len(files) > 10 and check_items.get('semantic'):
            check_items['semantic'] = False
            logger.info("Semantic analysis disabled because number of files exceeds 10.")
            _audit.component("semantic", status="SKIPPED", reason="files > 10")

        file_data = []
        for f in files:
            if not f.filename:
                continue
            if session.get('consent_value', 0) == 1:
                f.seek(0)
                file_bytes = f.read()
                file_hash = hashlib.sha256(file_bytes).hexdigest()
                file_size = len(file_bytes)
                f.seek(0)
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT id FROM user_files WHERE user_id = %s AND file_hash = %s", (user_id, file_hash))
                        existing = cur.fetchone()
                        if not existing:
                            ext = os.path.splitext(f.filename)[1]
                            unique_name = f"{file_hash}_{int(time.time())}{ext}"
                            original_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)
                            os.makedirs(original_dir, exist_ok=True)
                            original_path = os.path.join(original_dir, unique_name)
                            f.seek(0)
                            f.save(original_path)
                            f.seek(0)  # save() consumes stream, reset for later reads
                            ensure_user_exists(user_id)
                            cur.execute("""
                                INSERT INTO user_files (user_id, thread_id, filename, size_bytes, expires_at, original_stored_path, file_hash, original_expires_at, original_name)
                                VALUES (%s, %s, %s, %s, NULL, %s, %s, NOW() + INTERVAL '3 days', %s)
                                ON CONFLICT (thread_id, filename) DO UPDATE SET
                                    size_bytes = EXCLUDED.size_bytes,
                                    original_stored_path = EXCLUDED.original_stored_path,
                                    file_hash = EXCLUDED.file_hash,
                                    original_expires_at = EXCLUDED.original_expires_at,
                                    original_name = EXCLUDED.original_name
                            """, (user_id, thread_id, f.filename, file_size, original_path, file_hash, f.filename))
                            conn.commit()
                text = get_or_extract_file_analysis(f, 'chat', user_id, thread_id=thread_id)
            else:
                text, _ = extract_text_from_file(f)

            if text and not text.startswith("[") and is_valid_extracted_text(text):
                record_file_usage(thread_id, f.filename, 'compare_batch', "批量对比")
                f.seek(0)
                meta = extract_metadata(f)
                images = extract_images_from_file(f)
                file_data.append({
                    'filename': f.filename,
                    'text': text,
                    'metadata': meta,
                    'images': images
                })
            else:
                logger.warning(f"Skipping file {f.filename}: extraction failed or invalid (text='{text}')")
                continue

        if len(file_data) < 2:
            return jsonify({"error": "Could not extract valid text from at least two files"}), 400

        n = len(file_data)
        if check_items.get('text_sim', True) or check_items.get('key_info', True):
            vectorizer, tfidf_matrix = _precompute_tfidf_for_files(file_data, template_text)
        else:
            vectorizer = tfidf_matrix = None

        semantic_sim_matrix = None
        if check_items.get('semantic', False):
            all_texts = [fd['text'] for fd in file_data]
            semantic_sim_matrix = compute_batch_semantic_similarity(all_texts)
            logger.info("Semantic similarity matrix computed.")
            _audit.component("semantic", status="OK",
                             model=getattr(semantic_sim_matrix, '__class__', 'computed') if semantic_sim_matrix is not None else 'failed',
                             file_count=n)
        if semantic_sim_matrix is None and check_items.get('semantic', False):
            check_items['semantic'] = False
            logger.warning("Semantic analysis disabled due to model load failure.")
            _audit.component("semantic", status="FAILED", reason="model_load_failure")

        pairs = []
        risk_matrix = [[0] * n for _ in range(n)]
        _pair_audit_summary = {"text_sim_count": 0, "key_info_count": 0, "attr_count": 0, "img_count": 0, "total": 0}
        for i in range(n):
            for j in range(i + 1, n):
                text1 = file_data[i]['text']
                text2 = file_data[j]['text']
                meta1 = file_data[i]['metadata']
                meta2 = file_data[j]['metadata']
                images1 = file_data[i]['images']
                images2 = file_data[j]['images']
                img_sim = image_similarity(images1, images2) if check_items.get('image_sim', True) else 0.0
                if check_items.get('image_sim', True): _pair_audit_summary["img_count"] += 1

                if check_items.get('text_sim', True) and tfidf_matrix is not None:
                    sim = _compute_pair_similarity_from_matrix(tfidf_matrix, i, j)
                    _pair_audit_summary["text_sim_count"] += 1
                else:
                    sim = 0.0

                if check_items.get('key_info', True):
                    t1 = preprocess_text_for_similarity(text1)
                    t2 = preprocess_text_for_similarity(text2)
                    if template_text:
                        t1 = remove_template_content(t1, template_text)
                        t2 = remove_template_content(t2, template_text)
                    key_sim = keyword_overlap_similarity(t1, t2)
                    _pair_audit_summary["key_info_count"] += 1
                else:
                    key_sim = 0.0

                if check_items.get('file_attr', True) and meta1 and meta2:
                    attr_sim = file_attr_similarity(meta1, meta2)
                    _pair_audit_summary["attr_count"] += 1
                else:
                    attr_sim = 0.0

                text_sim_val = sim * 100
                key_info_val = key_sim * 100
                file_attr_val = attr_sim
                img_sim_val = img_sim

                risk = 0.3 * key_info_val + 0.3 * file_attr_val + 0.2 * text_sim_val + 0.2 * img_sim_val
                _pair_audit_summary["total"] += 1

                _, html1, html2, blocks = compute_similarity_with_numbers(text1, text2, template_text)

                pair_info = {
                    'i': i, 'j': j,
                    'name1': file_data[i]['filename'],
                    'name2': file_data[j]['filename'],
                    'text1': text1,
                    'text2': text2,
                    'sim': sim * 100,
                    'risk': risk,
                    'blocks': blocks,
                    'html1': html1,
                    'html2': html2,
                    'used_weights': {},
                    'attr_same': 1 if meta1.get('author') and meta1['author'] == meta2.get('author') else 0
                }
                pairs.append(pair_info)
                risk_matrix[i][j] = risk
                risk_matrix[j][i] = risk

        key_info_matches = []
        for p in pairs:
            kw1 = set(extract_keywords(p['text1'], 20))
            kw2 = set(extract_keywords(p['text2'], 20))
            common = kw1 & kw2
            key_info_matches.append({
                'name1': p['name1'],
                'name2': p['name2'],
                'common_keywords': list(common)[:10]
            })

        attr_details = []
        for fd in file_data:
            meta = fd['metadata']
            attr_details.append({
                'filename': fd['filename'],
                'author': meta.get('author', ''),
                'creation_date': meta.get('creationDate', ''),
                'creator': meta.get('creator', ''),
                'producer': meta.get('producer', '')
            })

        batch_data = {
            'file_data': [{'filename': fd['filename'], 'metadata': fd['metadata']} for fd in file_data],
            'pairs': pairs,
            'check_items': check_items,
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'key_info_matches': key_info_matches,
            'attr_details': attr_details,
            'semantic_sim_matrix': semantic_sim_matrix,
        }

        temp_path = store_batch_comparison_temp(batch_data)
        session['batch_comparison_path'] = temp_path

        high_risk_files = []
        strong_alert_files = []
        for i in range(n):
            for j in range(i + 1, n):
                if risk_matrix[i][j] > 20:
                    strong_alert_files.extend([file_data[i]['filename'], file_data[j]['filename']])
                elif risk_matrix[i][j] > 10:
                    high_risk_files.extend([file_data[i]['filename'], file_data[j]['filename']])
        strong_alert_files = list(set(strong_alert_files))
        high_risk_files = list(set(high_risk_files) - set(strong_alert_files))

        short_names = [truncate_filename(fd['filename'], 20) for fd in file_data]

        # ----- Build summary HTML (escaped) -----
        summary_html = '<details style="margin-bottom:4px; border-radius:6px; padding:6px;"><summary style="cursor:pointer; font-weight:bold; font-size:0.9rem;">📋 对比摘要 (点击展开)</summary><div style="margin-top:12px; border-left:8px solid #2c3e50; padding-left:8px;">'
        for fd in file_data:
            preview = html.escape(fd['text'][:200].replace('\n', ' ')) + '…'
            safe_filename = html.escape(fd['filename'])
            summary_html += f'<div style="margin-bottom:15px;"><strong>📄 {safe_filename}</strong><br><span style="color:#666; font-size:0.85rem;">{preview}</span></div>'
        if strong_alert_files:
            safe_strong = ', '.join(html.escape(f) for f in strong_alert_files)
            summary_html += f'<p style="color:#d9534f; font-weight:bold;">🚨 强烈警告：以下文件风险度超过20：{safe_strong}</p>'
        elif high_risk_files:
            safe_high = ', '.join(html.escape(f) for f in high_risk_files)
            summary_html += f'<p style="color:#f0ad4e; font-weight:bold;">⚠️ 可疑文件：以下文件风险度超过10：{safe_high}</p>'
        else:
            summary_html += '<p style="color:#5cb85c;">✅ 未发现高风险文件（风险度均≤10）</p>'
        summary_html += '</div></details>'

        if n == 2:
            p = pairs[0]
            if p['blocks']:
                detail_rows = ""
                for b in p['blocks']:
                    detail_rows += f'''
                    <tr>
                        <td style="border:1px solid #ccc; padding:8px; text-align:center;">{b["id"]}</td>
                        <td style="border:1px solid #ccc; padding:8px; text-align:center;">{b["size"]}</td>
                        <td style="border:1px solid #ccc; padding:8px; word-break:break-word; max-width:300px;">{html.escape(b["text1_snippet"])}</td>
                        <td style="border:1px solid #ccc; padding:8px; word-break:break-word; max-width:300px;">{html.escape(b["text2_snippet"])}</td>
                    </tr>
                    '''
                detailed_report = f'<details><summary style="cursor:pointer; font-weight:bold;">📋 详细相似度明细报告（共 {len(p["blocks"])} 个匹配块）</summary><div style="margin-top:12px;"><p><strong>总匹配字符数：</strong>{sum(b["size"] for b in p["blocks"])} 字符 &nbsp;|&nbsp;<strong>平均匹配块长度：</strong>{round(sum(b["size"] for b in p["blocks"]) / len(p["blocks"]), 1)} 字符</p><div style="overflow-x:auto;"><table style="width:100%; border-collapse:collapse; margin-top:10px;"><thead><tr style="background:#f0f0f0;"><th style="border:1px solid #ccc; padding:8px;">块序号</th><th style="border:1px solid #ccc; padding:8px;">匹配字符数</th><th style="border:1px solid #ccc; padding:8px;">文档A片段</th><th style="border:1px solid #ccc; padding:8px;">文档B片段</th></tr></thead><tbody>{detail_rows}</tbody></table></div></div></details>'
                main_report = detailed_report
            else:
                main_report = "<p>未检测到显著匹配块。</p>"
        else:
            matrix_html = '<details><summary style="cursor:pointer; font-weight:bold;">📊 风险度矩阵 (点击展开/折叠)</summary><div style="overflow-x:auto; margin-top:12px;"><table style="border-collapse:collapse; font-size:0.85rem; min-width:400px; width:100%;"><thead><tr><th style="padding:8px; border:1px solid #ddd;"></th>' + ''.join(f'<th style="padding:8px; border:1px solid #ddd; word-break:break-word;">{html.escape(short_names[i])}</th>' for i in range(n)) + '</tr></thead><tbody>'
            for i in range(n):
                matrix_html += f'<tr><td style="border:1px solid #ddd; padding:8px; font-weight:bold;">{html.escape(short_names[i])}</td>'
                for j in range(n):
                    if i == j:
                        val = '--'
                        bg = ''
                    else:
                        val = f'{risk_matrix[i][j]:.2f}'
                        if risk_matrix[i][j] > 20:
                            bg = ' style="background:#d9534f; color:white; font-weight:bold;"'
                        elif risk_matrix[i][j] > 10:
                            bg = ' style="background:#f0ad4e;"'
                        else:
                            bg = ''
                    matrix_html += f'<td style="border:1px solid #ddd; padding:8px; text-align:center;"{bg}>{html.escape(val)}</td>'
                matrix_html += '</tr>'
            matrix_html += '</tbody></table></div><p style="font-size:0.7rem; color:#666; margin-top:8px;">风险度矩阵（值越高风险越大）</p></details>'
            main_report = matrix_html

        # ── Generate Excel (same as before) ──
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment
        from openpyxl.utils import get_column_letter
        _ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        # Build Excel using same logic as export_batch_excel_download
        wb = Workbook()
        ws1 = wb.active; ws1.title = "规律性分析结果"
        ws1.merge_cells('A1:H1'); ws1['A1'] = "技术标规律性分析检查结果"; ws1['A1'].font = Font(bold=True, size=14)
        ws1['A2'] = "标段名称：用户自定义"; ws1['A3'] = f"投标单位个数：{len(file_data)}"; ws1['A4'] = f"创建时间：{_ts}"
        max_risk = max(p['risk'] for p in pairs) if pairs else 0; max_sim = max(p['sim'] for p in pairs) if pairs else 0
        ws1['A5'] = f"检查结果：文本相似度检查{'存在异常' if max_sim>0 else '无异常'}；重点信息无异常；文档属性检查无异常；图片相似度检查无异常"
        ws1['A6'] = "检查规则：检查相似度≥80%的段落，文本中重点信息，相似图片，相同作者；忽略与招标文件相同内容，忽略标点符号及小于6个字的内容，忽略目录，忽略文件中的技术标准"
        ws1.merge_cells('A6:H6'); row = 10
        ws1[f'A{row}'] = "一、标书围串风险分析结果"; ws1[f'A{row}'].font = Font(bold=True); row += 1
        headers = ["投标单位"] + [fd['filename'] for fd in file_data]
        for col, h in enumerate(headers, 1):
            c = ws1.cell(row=row, column=col, value=h); c.font = Font(bold=True); c.alignment = Alignment(horizontal='center')
        row += 1
        for i in range(n):
            ws1.cell(row=row, column=1, value=file_data[i]['filename'])
            for j in range(n):
                val = "--" if i==j else next((p['risk'] for p in pairs if (p['i']==i and p['j']==j)or(p['i']==j and p['j']==i)),0)
                ws1.cell(row=row, column=j+2, value=val)
            row += 1
        row += 2; ws1[f'A{row}'] = "二、分析结果详情"; ws1[f'A{row}'].font = Font(bold=True); row += 1
        detail_headers = ["序号","投标单位1","投标单位2","风险度","文本相似度（%）","语义相似度（%）","图片相似度（%）","文件属性雷同","重点信息雷同（项）"]
        for col, h in enumerate(detail_headers, 1):
            c = ws1.cell(row=row, column=col, value=h); c.font = Font(bold=True); c.alignment = Alignment(horizontal='center')
        row += 1
        for idx, p in enumerate(pairs, 1):
            ws1.cell(row=row, column=1, value=idx); ws1.cell(row=row, column=2, value=p['name1']); ws1.cell(row=row, column=3, value=p['name2'])
            ws1.cell(row=row, column=4, value=p['risk']); ws1.cell(row=row, column=5, value=p['sim'])
            ws1.cell(row=row, column=6, value=p.get('semantic_sim',0)); ws1.cell(row=row, column=7, value=0)
            ws1.cell(row=row, column=8, value="是" if p['attr_same'] else "否")
            ki_match = next((k for k in key_info_matches if k['name1']==p['name1'] and k['name2']==p['name2']), None)
            ws1.cell(row=row, column=9, value=len(ki_match['common_keywords']) if ki_match else 0)
            row += 1
        for col in range(1,10): ws1.column_dimensions[get_column_letter(col)].width = 20
        # Tab 2: text detail matches
        ws2 = wb.create_sheet("规律性分析详情（文本）"); ws2['A1']="规律性分析详情（文本）"; ws2.merge_cells('A1:I1')
        ws2['A5']="序号";ws2['B5']="目标单位";ws2['C5']="目标单位对应文档";ws2['D5']="页码";ws2['E5']="目标单位内容";ws2['F5']="对比单位";ws2['G5']="对比单位对应文档";ws2['H5']="页码";ws2['I5']="对比单位相似内容"
        for col in range(1,10): ws2.cell(row=5, column=col).font = Font(bold=True)
        r2=6;seq=1
        for p in pairs:
            if not p['blocks']: continue
            for blk in p['blocks']:
                ws2.cell(row=r2,column=1,value=seq);ws2.cell(row=r2,column=2,value=p['name1']);ws2.cell(row=r2,column=3,value=p['name1']+".pdf")
                ws2.cell(row=r2,column=4,value=blk.get('page1',''));ws2.cell(row=r2,column=5,value=blk['text1_snippet']);ws2.cell(row=r2,column=6,value=p['name2'])
                ws2.cell(row=r2,column=7,value=p['name2']+".pdf");ws2.cell(row=r2,column=8,value=blk.get('page2',''));ws2.cell(row=r2,column=9,value=blk['text2_snippet'])
                r2+=1;seq+=1
        for col in range(1,10): ws2.column_dimensions[get_column_letter(col)].width = 30
        # Tab 3: key info
        ws3 = wb.create_sheet("规律性分析详情（重点信息）"); ws3['A1']="规律性分析详情（重点信息）"; ws3.merge_cells('A1:I1')
        ws3['A5']="序号";ws3['B5']="AI识别类型";ws3['C5']="内容";ws3['D5']="目标单位";ws3['E5']="目标单位对应文档";ws3['F5']="页码";ws3['G5']="对比单位";ws3['H5']="对比单位对应文档";ws3['I5']="页码"
        for col in range(1,10): ws3.cell(row=5, column=col).font = Font(bold=True)
        r3=6;seq3=1
        for ki in key_info_matches:
            for kw in ki['common_keywords']:
                ws3.cell(row=r3,column=1,value=seq3);ws3.cell(row=r3,column=2,value="关键词");ws3.cell(row=r3,column=3,value=kw)
                ws3.cell(row=r3,column=4,value=ki['name1']);ws3.cell(row=r3,column=5,value=ki['name1']+".pdf")
                ws3.cell(row=r3,column=7,value=ki['name2']);ws3.cell(row=r3,column=8,value=ki['name2']+".pdf");r3+=1;seq3+=1
        for col in range(1,10): ws3.column_dimensions[get_column_letter(col)].width = 20

        # ── Typo detection (auto-included in all batch compares) ──
        typo_results = None
        try:
            from app.services.typo_detector import detect_typos_batch, save_typo_results as save_typo
            typo_results = detect_typos_batch(file_data, audit=_audit)
            save_typo(user_id, thread_id, typo_results)
            total_typos = sum(r.total_suspects for r in typo_results.values())
            total_crit = sum(r.critical_count for r in typo_results.values())
            _audit.component("typo_detection", status="OK",
                            files=len(typo_results), total_findings=total_typos,
                            critical=total_crit)
        except Exception as e:
            logger.warning(f"Typo detection failed (non-blocking): {e}")
            _audit.component("typo_detection", status="FAILED", error=str(e)[:100])

        # ── Relationship extraction (auto-included in all batch compares) ──
        rel_report = None
        try:
            from app.services.relationship_extractor import extract_relationships as run_rel, save_relationship_results as save_rel
            rel_report = run_rel(file_data, audit=_audit)
            save_rel(user_id, thread_id, rel_report)
            _audit.component("relationship_extraction", status="OK",
                            entities=len(rel_report.entities),
                            relations=len(rel_report.relationships),
                            risk_score=rel_report.risk_score,
                            flags=len(rel_report.red_flags))
        except Exception as e:
            logger.warning(f"Relationship extraction failed (non-blocking): {e}")
            _audit.component("relationship_extraction", status="FAILED", error=str(e)[:100])

        # ── Quote anomaly detection (auto-included in all batch compares) ──
        quote_result = None
        try:
            from app.services.quote_anomaly import compare_bidders_quotes, save_quote_anomaly_results
            quote_result = compare_bidders_quotes(file_data, audit=_audit)
            save_quote_anomaly_results(user_id, thread_id, quote_result['per_bidder'], quote_result)
            _audit.component("quote_anomaly", status="OK",
                            bidders=len(file_data),
                            max_risk=round(quote_result.get('max_risk_score', 0), 1),
                            cross_same_rate=quote_result.get('cross_same_rate', False),
                            cross_clustering=quote_result.get('cross_clustering', False))
        except Exception as e:
            logger.warning(f"Quote anomaly check failed (non-blocking): {e}")
            _audit.component("quote_anomaly", status="FAILED", error=str(e)[:100])

        # ── Generate AI summary HTML ──
        ai_html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><title>批量对比报告</title>
        <style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:900px;margin:0 auto;padding:20px;color:#1e293b;line-height:1.6}}
        h1{{color:#0f172a;border-bottom:3px solid #2563eb;padding-bottom:8px}}h2{{color:#334155;margin-top:24px}}
        .card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin:12px 0}}
        .risk-high{{color:#dc2626;font-weight:bold}}.risk-warn{{color:#d97706;font-weight:bold}}.risk-ok{{color:#16a34a}}
        table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:.85rem}}th,td{{border:1px solid #e2e8f0;padding:8px;text-align:left}}
        th{{background:#f1f5f9}}.highlight{{background:#fef9c3}}</style></head><body>
        <h1>📁 批量对比报告</h1>
        <div class="card"><strong>对比文件数：</strong>{len(file_data)} &nbsp;|&nbsp; <strong>对比对数：</strong>{len(pairs)} &nbsp;|&nbsp; <strong>最高风险度：</strong>{max_risk:.1f} &nbsp;|&nbsp; <strong>生成时间：</strong>{_ts}</div>
        <h2>📋 文件列表</h2><ul>{''.join(f'<li><strong>{html.escape(fd["filename"])}</strong></li>' for fd in file_data)}</ul>
        <h2>📊 风险度矩阵</h2><table><tr><th></th>{''.join(f'<th>{html.escape(truncate_filename(fd["filename"], 15))}</th>' for fd in file_data)}</tr>
        {''.join(f'<tr><td><strong>{html.escape(truncate_filename(file_data[i]["filename"], 15))}</strong></td>{"".join("<td style=\"text-align:center;\">--</td>" if i==j else f"<td class=\"{'risk-high' if risk_matrix[i][j]>20 else 'risk-warn' if risk_matrix[i][j]>10 else 'risk-ok'}\">{risk_matrix[i][j]:.1f}</td>" for j in range(n))}</tr>' for i in range(n))}
        </table>"""

        if strong_alert_files:
            ai_html += f'<div class="card" style="background:#fef2f2;border-color:#fecaca;"><h3 style="margin-top:0;color:#dc2626;">🚨 强烈警告</h3><p>以下文件风险度超过20：<strong>{", ".join(html.escape(f) for f in strong_alert_files)}</strong></p></div>'
        elif high_risk_files:
            ai_html += f'<div class="card" style="background:#fffbeb;border-color:#fde68a;"><h3 style="margin-top:0;color:#d97706;">⚠️ 需关注</h3><p>以下文件风险度超过10：<strong>{", ".join(html.escape(f) for f in high_risk_files)}</strong></p></div>'

        for p in pairs[:10]:
            risk_class = 'risk-high' if p['risk']>20 else ('risk-warn' if p['risk']>10 else 'risk-ok')
            ai_html += f'<div class="card"><h3>📄 {html.escape(p["name1"])} ↔ {html.escape(p["name2"])}</h3>'
            ai_html += f'<p>风险度: <span class="{risk_class}">{p["risk"]:.1f}</span> | 文本相似度: {p["sim"]:.1f}% | 相同作者: {"是" if p["attr_same"] else "否"}</p>'
            if p['blocks']:
                ai_html += f'<p>匹配段落数: {len(p["blocks"])} · 总匹配字符: {sum(b["size"] for b in p["blocks"])}</p>'
            ai_html += '</div>'
        # ── AI professional analysis of the highest-risk pair ──
        ai_analysis_html = ''
        top_pair = max(pairs, key=lambda p: p['risk']) if pairs else None
        if top_pair and top_pair['risk'] > 5:
            try:
                from app.services.analysis_prompts import build_bid_analysis_prompt, BID_COMPARISON_SYSTEM
                from app.services.llm_provider import create_chat_model
                from langchain_core.messages import SystemMessage, HumanMessage
                high_pairs = sorted(
                    [(p['name1'], p['name2'], p['risk'], p['sim']) for p in pairs if p['risk'] > 10],
                    key=lambda x: x[2], reverse=True
                )
                prompt = build_bid_analysis_prompt(
                    risk_matrix=risk_matrix,
                    file_names=short_names,
                    high_risk_pairs=high_pairs[:5],
                    top_pair_text1=top_pair['text1'],
                    top_pair_text2=top_pair['text2'],
                    top_pair_name1=top_pair['name1'],
                    top_pair_name2=top_pair['name2'],
                    top_pair_risk=top_pair['risk'],
                )
                llm = create_chat_model(streaming=False, temperature=0.3, max_tokens=1500,
                                        timeout=int(os.getenv("LLM_TIMEOUT", "90")))
                from app.services.prompt_safety import sanitize_for_prompt
                ai_resp = llm.invoke([SystemMessage(content=BID_COMPARISON_SYSTEM),
                                     HumanMessage(content=sanitize_for_prompt(prompt, 'bid_analysis'))])
                ai_text = ai_resp.content if hasattr(ai_resp, 'content') else str(ai_resp)
                if ai_text and len(ai_text) > 15:
                    ai_analysis_html = f'<div class="card" style="background:#eff6ff;border-color:#bfdbfe;margin-top:16px;"><h2 style="color:#1e40af;">🤖 AI专业分析</h2>{ai_text}</div>'
            except Exception as e:
                logger.warning(f"AI bid analysis failed: {e}")
                _audit.component("ai_analysis", status="FAILED", error=str(e)[:100])
            else:
                _audit.component("ai_analysis", status="OK",
                                 top_risk=top_pair['risk'], model="deepseek-v4-pro",
                                 response_chars=len(ai_text) if ai_text else 0)

        # ── Final audit: summarize all components ──
        max_risk_val = max(p['risk'] for p in pairs) if pairs else 0
        _audit.result(
            file_count=n, pair_count=len(pairs), max_risk=round(max_risk_val, 2),
            components=str(_pair_audit_summary),
            template_used=bool(template_text),
            ai_analysis=bool(ai_analysis_html),
        )

        # ── Typo detection section ──
        if typo_results:
            total_typos = sum(r.total_suspects for r in typo_results.values())
            total_crit = sum(r.critical_count for r in typo_results.values())
            if total_typos > 0:
                ai_html += '<h2>📝 错别字检测</h2>'
                ai_html += f'<div class="card"><p><strong>疑似错别字:</strong> {total_typos} 处 | <strong>严重:</strong> {total_crit} 处</p>'
                ai_html += '<table style="font-size:0.85rem;"><tr><th>文件</th><th>层次</th><th>疑似文本</th><th>建议</th><th>置信度</th><th>严重性</th></tr>'
                for doc_name, report in typo_results.items():
                    for f in report.findings[:30]:
                        sev_class = 'risk-high' if f.severity == 'critical' else ('risk-warn' if f.severity == 'warning' else '')
                        ai_html += f'<tr><td>{html.escape(doc_name[:20])}</td>'
                        ai_html += f'<td>{f.layer}</td>'
                        ai_html += f'<td><code>{html.escape(f.suspect_text[:40])}</code></td>'
                        ai_html += f'<td>{html.escape(", ".join(f.suggestions[:3]) if f.suggestions else "—")}</td>'
                        ai_html += f'<td>{f.confidence:.0%}</td>'
                        ai_html += f'<td class="{sev_class}">{f.severity}</td></tr>'
                ai_html += '</table>'
                if total_typos > 30:
                    ai_html += f'<p style="color:#64748b;font-size:.85rem;">（仅显示前30项，共{total_typos}项）</p>'
                ai_html += '</div>'

        # ── Relationship extraction section ──
        if rel_report and rel_report.red_flags:
            ai_html += '<h2>🔗 关联关系分析</h2>'
            ai_html += f'<div class="card"><p><strong>提取实体:</strong> {len(rel_report.entities)} | <strong>发现关系:</strong> {len(rel_report.relationships)} | <strong>风险评分:</strong> <span class="{"risk-high" if rel_report.risk_score > 50 else ("risk-warn" if rel_report.risk_score > 20 else "risk-ok")}">{rel_report.risk_score:.1f}</span></p>'
            ai_html += '<p><strong>检测模块:</strong> ' + ', '.join(rel_report.modules_run) + '</p>'
            if rel_report.red_flags:
                ai_html += '<ul>'
                for flag in rel_report.red_flags[:15]:
                    ai_html += f'<li class="risk-warn">{html.escape(flag)}</li>'
                ai_html += '</ul>'

            # Company-personnel map (for manual review when 天眼查 is off)
            cpm = rel_report.company_personnel_map
            if cpm.get('manual_review_required') and cpm.get('companies'):
                ai_html += '<details><summary style="cursor:pointer;font-weight:bold;margin-top:8px;">📋 公司与关键人员清单（供人工审查）</summary>'
                ai_html += '<table style="margin-top:8px;"><tr><th>公司名称</th><th>关键人员</th><th>涉及文件</th></tr>'
                for comp in cpm['companies'][:20]:
                    personnel_str = '; '.join(f"{p['name']}({p['title']})" for p in comp['personnel'][:5])
                    ai_html += f'<tr><td>{html.escape(comp["name"])}</td><td>{html.escape(personnel_str)}</td><td>{comp["file_count"]}个文件</td></tr>'
                ai_html += '</table></details>'
            ai_html += '</div>'

        # ── Quote anomaly section ──
        if quote_result and quote_result.get('per_bidder'):
            ai_html += '<h2>💰 报价异常检测</h2>'
            ai_html += '<div class="card"><table><tr><th>投标单位</th><th>风险评分</th><th>离散系数(CV)</th><th>同价疑义</th><th>异常降幅</th><th>聚类疑义</th><th>本福特偏差</th></tr>'
            for pb in quote_result['per_bidder']:
                flags = []
                if pb.get('same_rate_flag'): flags.append('⚠️同价')
                if pb.get('abnormal_drop_flag'): flags.append('⬇️异常降幅')
                if pb.get('clustering_flag'): flags.append('🔗聚类')
                flag_str = ', '.join(flags) if flags else '✅ 正常'
                risk_class = 'risk-high' if pb.get('risk_score', 0) > 50 else ('risk-warn' if pb.get('risk_score', 0) > 20 else 'risk-ok')
                ai_html += f'<tr><td>{html.escape(pb["filename"])}</td>'
                ai_html += f'<td class="{risk_class}">{pb.get("risk_score", 0):.1f}</td>'
                ai_html += f'<td>{pb.get("cv", 0):.4f}</td>'
                ai_html += f'<td>{"是" if pb.get("same_rate_flag") else "否"}</td>'
                ai_html += f'<td>{"是" if pb.get("abnormal_drop_flag") else "否"}</td>'
                ai_html += f'<td>{"是" if pb.get("clustering_flag") else "否"}</td>'
                ai_html += f'<td>{pb.get("benford_deviation", 0):.3f}</td></tr>'
            ai_html += '</table>'
            if quote_result.get('cross_same_rate'):
                ai_html += '<p class="risk-warn">⚠️ 跨投标单位同价疑义：多个投标单位首轮报价异常接近</p>'
            if quote_result.get('cross_clustering'):
                ai_html += '<p class="risk-warn">🔗 跨投标单位价格聚类：多个投标单位报价集中在异常窄区间</p>'
            ai_html += f'<p style="color:#64748b;font-size:.85rem;">最高报价风险评分: {quote_result.get("max_risk_score", 0):.1f} | 平均CV: {quote_result.get("avg_cv", 0):.4f}</p>'
            ai_html += '</div>'

        ai_html += '<p style="margin-top:24px;color:#64748b;font-size:.85rem;">完整风险矩阵、文本匹配详情、重点信息雷同、文件属性分析请参见配套Excel文件。</p>'
        if ai_analysis_html:
            ai_html += ai_analysis_html
        ai_html += '</body></html>'

        # ── ZIP: HTML + Excel ──
        import zipfile
        batch_task_id = str(uuid.uuid4())
        batch_dir = os.path.join(DATA_DIR, 'batch_results')
        os.makedirs(batch_dir, exist_ok=True)
        zip_name = f"batch_{batch_task_id}.zip"
        zip_path = os.path.join(batch_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("AI分析报告.html", ai_html.encode('utf-8'))
            excel_buf = BytesIO(); wb.save(excel_buf); excel_buf.seek(0)
            zf.writestr("清标分析结果.xlsx", excel_buf.getvalue())
        # Insert into DB for permanent access
        file_names_list = [fd['filename'] for fd in file_data]
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO batch_comparison_results (user_id, task_id, file_count, pair_count, max_risk, file_names, zip_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (user_id, batch_task_id, len(file_data), len(pairs), float(round(max_risk, 2)),
                      json.dumps(file_names_list, ensure_ascii=False), zip_path))
                conn.commit()

        batch_download_url = url_for('batch.download_batch_result', task_id=batch_task_id, _external=True)
        export_html = f'<p style="margin-top:12px;"><a href="{batch_download_url}" target="_blank" style="background:#27ae60; color:white; text-decoration:none; border-radius:8px; padding:8px 16px; display:inline-block;">📦 下载完整报告 (HTML+Excel，永久有效)</a></p>'
        full_message = f"<!--COMPARE_REPORT--><div style='font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, sans-serif; line-height:1.5; max-width:100%; overflow-x:auto;'><h4>📁 批量对比结果（{len(file_data)}个文件）</h4>{summary_html}{main_report}{export_html}</div>"

        if session.get('consent_value', 0) == 1:
            ensure_user_exists(user_id)
            report_filename = f"批量对比_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.html"
            report_bytes = full_message.encode('utf-8')
            report_hash = hashlib.sha256(report_bytes).hexdigest()
            ext = '.html'
            unique_name = f"{report_hash}_{int(time.time())}{ext}"
            original_dir = os.path.join(USER_FILES_ORIGINAL_ROOT, user_id)
            os.makedirs(original_dir, exist_ok=True)
            report_path = os.path.join(original_dir, unique_name)
            with open(report_path, 'wb') as f:
                f.write(report_bytes)
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO user_files (user_id, thread_id, filename, size_bytes, expires_at, original_stored_path, file_hash, original_expires_at, original_name)
                        VALUES (%s, %s, %s, %s, NULL, %s, %s, NOW() + INTERVAL '3 days', %s)
                        ON CONFLICT (thread_id, filename) DO NOTHING
                    """, (user_id, thread_id, report_filename, len(report_bytes), report_path, report_hash, report_filename))
                    conn.commit()
            record_file_usage(thread_id, report_filename, 'compare_batch_report', "批量对比生成的报告")

        store_message(thread_id, 'assistant', full_message, thinking="")
        session['chat_history'].append({
            "role": "assistant",
            "content": full_message,
            "thinking": ""
        })
        return jsonify({
            "success": True,
            "pair_count": len(pairs),
            "download_url": batch_download_url,
            "file_count": len(file_data),
        })
    finally:
        release_task_lock(user_id)

@batch_bp.route('/export_batch_excel_download/<token>', methods=['GET'])
def export_batch_excel_download(token):
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    remaining = g.download_tokens.get(token, 0)
    if remaining <= 0:
        return jsonify({"error": "Download link has expired or already used the maximum number of times."}), 410
    temp_path = session.get(f'download_path_{token}')
    if not temp_path or not os.path.exists(temp_path):
        return jsonify({"error": "Comparison data not found."}), 404
    try:
        batch_data = load_batch_comparison_temp(temp_path)
    except Exception as e:
        logger.error(f"Failed to load batch data: {e}")
        return jsonify({"error": "Comparison data corrupted."}), 400

    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment
    from openpyxl.utils import get_column_letter

    file_data = batch_data['file_data']
    pairs = batch_data['pairs']
    timestamp = batch_data['timestamp']
    check_items = batch_data.get('check_items', {})
    key_info_matches = batch_data.get('key_info_matches', [])
    attr_details = batch_data.get('attr_details', [])

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "规律性分析结果"
    ws1.merge_cells('A1:H1')
    ws1['A1'] = "技术标规律性分析检查结果"
    ws1['A1'].font = Font(bold=True, size=14)
    ws1['A2'] = "标段名称：用户自定义"
    ws1['A3'] = f"投标单位个数：{len(file_data)}"
    ws1['A4'] = f"创建时间：{timestamp}"
    max_risk = max(p['risk'] for p in pairs) if pairs else 0
    max_sim = max(p['sim'] for p in pairs) if pairs else 0
    result_parts = []
    if check_items.get('text_sim', True):
        result_parts.append("文本相似度检查存在异常" if max_sim > 0 else "文本相似度检查无异常")
    if check_items.get('key_info', True):
        result_parts.append("重点信息无异常")
    if check_items.get('file_attr', True):
        result_parts.append("文档属性检查无异常")
    if check_items.get('image_sim', True):
        result_parts.append("图片相似度检查无异常")
    result_str = "；".join(result_parts) if result_parts else "无异常"
    ws1['A5'] = f"检查结果：{result_str}"
    ws1['A6'] = "检查规则：检查相似度≥80%的段落，文本中重点信息，相似图片，相同作者；忽略与招标文件相同内容，忽略标点符号及小于6个字的内容，忽略目录，忽略文件中的技术标准，忽略【公司/组织、地名/地址、项目、人员、奖项、身份证号码、电话号码、统一社会信用代码、证书编号】"
    ws1['A7'] = "相似度计算说明：风险度=0.3×重点信息雷同风险+0.3×文件属性雷同风险+0.2×文本相似度×100+0.2×图片相似度×100\n*若某项不参与检查，则其余项按照比例进行折算"
    ws1.merge_cells('A6:H6')
    ws1.merge_cells('A7:H7')
    row = 10
    ws1[f'A{row}'] = "一、标书围串风险分析结果"
    ws1[f'A{row}'].font = Font(bold=True)
    row += 1
    headers = ["投标单位"] + [fd['filename'] for fd in file_data]
    for col, h in enumerate(headers, 1):
        cell = ws1.cell(row=row, column=col, value=h)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
    row += 1
    for i in range(len(file_data)):
        ws1.cell(row=row, column=1, value=file_data[i]['filename'])
        for j in range(len(file_data)):
            if i == j:
                val = "--"
            else:
                for p in pairs:
                    if (p['i'] == i and p['j'] == j) or (p['i'] == j and p['j'] == i):
                        val = p['risk']
                        break
                else:
                    val = 0
            ws1.cell(row=row, column=j+2, value=val)
        row += 1
    row += 2
    ws1[f'A{row}'] = "二、分析结果详情"
    ws1[f'A{row}'].font = Font(bold=True)
    row += 1
    detail_headers = ["序号", "投标单位1", "投标单位2", "风险度", "文本相似度（%）", "语义相似度（%）", "图片相似度（%）", "文件属性雷同", "重点信息雷同（项）"]
    for col, h in enumerate(detail_headers, 1):
        cell = ws1.cell(row=row, column=col, value=h)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
    row += 1
    for idx, p in enumerate(pairs, 1):
        ws1.cell(row=row, column=1, value=idx)
        ws1.cell(row=row, column=2, value=p['name1'])
        ws1.cell(row=row, column=3, value=p['name2'])
        ws1.cell(row=row, column=4, value=p['risk'])
        ws1.cell(row=row, column=5, value=p['sim'])
        ws1.cell(row=row, column=6, value=p.get('semantic_sim', 0))
        ws1.cell(row=row, column=7, value=0)
        ws1.cell(row=row, column=8, value="是" if p['attr_same'] else "否")
        ki_match = next((k for k in key_info_matches if k['name1'] == p['name1'] and k['name2'] == p['name2']), None)
        ki_count = len(ki_match['common_keywords']) if ki_match else 0
        ws1.cell(row=row, column=9, value=ki_count)
        row += 1
    for col in range(1, 9):
        ws1.column_dimensions[get_column_letter(col)].width = 20

    ws2 = wb.create_sheet("规律性分析详情（文本）")
    ws2['A1'] = "技术标规律性分析详情（文本）"
    ws2.merge_cells('A1:I1')
    ws2['A2'] = "标段名称：用户自定义"
    ws2['A3'] = "检查规则：检查相似度≥80%的段落，忽略与招标文件相同内容，忽略标点符号及小于6个字的内容，忽略目录，忽略文件中的技术标准"
    ws2['A5'] = "序号"
    ws2['B5'] = "目标单位"
    ws2['C5'] = "目标单位对应文档"
    ws2['D5'] = "页码"
    ws2['E5'] = "目标单位内容"
    ws2['F5'] = "对比单位"
    ws2['G5'] = "对比单位对应文档"
    ws2['H5'] = "页码"
    ws2['I5'] = "对比单位相似内容"
    for col in range(1, 10):
        cell = ws2.cell(row=5, column=col)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
    r = 6
    seq = 1
    for p in pairs:
        if not p['blocks']:
            continue
        for block in p['blocks']:
            ws2.cell(row=r, column=1, value=seq)
            ws2.cell(row=r, column=2, value=p['name1'])
            ws2.cell(row=r, column=3, value=p['name1'] + ".pdf")
            ws2.cell(row=r, column=4, value=block.get('page1', ''))
            ws2.cell(row=r, column=5, value=block['text1_snippet'])
            ws2.cell(row=r, column=6, value=p['name2'])
            ws2.cell(row=r, column=7, value=p['name2'] + ".pdf")
            ws2.cell(row=r, column=8, value=block.get('page2', ''))
            ws2.cell(row=r, column=9, value=block['text2_snippet'])
            r += 1
            seq += 1
    for col in range(1, 10):
        ws2.column_dimensions[get_column_letter(col)].width = 30

    ws3 = wb.create_sheet("规律性分析详情（重点信息）")
    ws3['A1'] = "技术标规律性分析详情（重点信息）"
    ws3.merge_cells('A1:I1')
    ws3['A2'] = "标段名称：用户自定义"
    ws3['A3'] = "检查规则：检查文本中重点信息；忽略【公司/组织、地名/地址、项目、人员、奖项、身份证号码、电话号码、统一社会信用代码、证书编号】"
    ws3['A5'] = "序号"
    ws3['B5'] = "AI识别类型"
    ws3['C5'] = "内容"
    ws3['D5'] = "目标单位"
    ws3['E5'] = "目标单位对应文档"
    ws3['F5'] = "页码"
    ws3['G5'] = "对比单位"
    ws3['H5'] = "对比单位对应文档"
    ws3['I5'] = "页码"
    for col in range(1, 10):
        cell = ws3.cell(row=5, column=col)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
    r = 6
    seq = 1
    for ki in key_info_matches:
        for kw in ki['common_keywords']:
            ws3.cell(row=r, column=1, value=seq)
            ws3.cell(row=r, column=2, value="关键词")
            ws3.cell(row=r, column=3, value=kw)
            ws3.cell(row=r, column=4, value=ki['name1'])
            ws3.cell(row=r, column=5, value=ki['name1'] + ".pdf")
            ws3.cell(row=r, column=7, value=ki['name2'])
            ws3.cell(row=r, column=8, value=ki['name2'] + ".pdf")
            r += 1
            seq += 1
    for col in range(1, 10):
        ws3.column_dimensions[get_column_letter(col)].width = 20

    ws4 = wb.create_sheet("技术标规律性分析详情（文件属性-汇总）")
    ws4['A1'] = "技术标规律性分析详情（文件属性）"
    ws4.merge_cells('A1:E1')
    ws4['A2'] = "标段名称：用户自定义"
    ws4['A3'] = "检查规则：相同作者"
    ws4['A5'] = "序号"
    ws4['B5'] = "单位名称"
    ws4['C5'] = "作者"
    ws4['D5'] = "属性相同单位数量"
    ws4['E5'] = "属性相同单位名称"
    for col in range(1, 6):
        cell = ws4.cell(row=5, column=col)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
    author_map = {}
    for fd in file_data:
        name = fd['filename']
        author = fd['metadata'].get('author', '')
        author_map.setdefault(author, []).append(name)
    r = 6
    seq = 1
    for fd in file_data:
        name = fd['filename']
        author = fd['metadata'].get('author', '')
        same_authors = author_map.get(author, [])
        same_count = len(same_authors) - 1
        same_names = ", ".join([n for n in same_authors if n != name]) if same_count > 0 else ""
        ws4.cell(row=r, column=1, value=seq)
        ws4.cell(row=r, column=2, value=name)
        ws4.cell(row=r, column=3, value=author)
        ws4.cell(row=r, column=4, value=same_count)
        ws4.cell(row=r, column=5, value=same_names)
        r += 1
        seq += 1
    for col in range(1, 6):
        ws4.column_dimensions[get_column_letter(col)].width = 30

    ws5 = wb.create_sheet("技术标规律性分析详情（文件属性-详情）")
    ws5['A1'] = "技术标规律性分析详情（文件属性日志信息）"
    ws5.merge_cells('A1:G1')
    ws5['A2'] = "标段名称：用户自定义"
    ws5['A3'] = "检查规则：相同作者"
    ws5['A5'] = "序号"
    ws5['B5'] = "单位名称"
    ws5['C5'] = "文档名称"
    ws5['D5'] = "作者"
    ws5['E5'] = "属性相同单位数量"
    ws5['F5'] = "属性相同单位名称"
    ws5['G5'] = "属性相同文档名称"
    for col in range(1, 8):
        cell = ws5.cell(row=5, column=col)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
    r = 6
    seq = 1
    for fd in file_data:
        name = fd['filename']
        author = fd['metadata'].get('author', '')
        same_authors = author_map.get(author, [])
        same_count = len(same_authors) - 1
        same_names = ", ".join([n for n in same_authors if n != name]) if same_count > 0 else "无重复单位"
        same_docs = ", ".join([n for n in same_authors if n != name]) if same_count > 0 else "无重复文档"
        ws5.cell(row=r, column=1, value=seq)
        ws5.cell(row=r, column=2, value=name)
        ws5.cell(row=r, column=3, value=name + ".pdf")
        ws5.cell(row=r, column=4, value=author)
        ws5.cell(row=r, column=5, value=same_count)
        ws5.cell(row=r, column=6, value=same_names)
        ws5.cell(row=r, column=7, value=same_docs)
        r += 1
        seq += 1
    for col in range(1, 8):
        ws5.column_dimensions[get_column_letter(col)].width = 30

    output = BytesIO()
    wb.save(output)
    output.seek(0)
    filename = f"清标分析结果_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.xlsx"
    g.download_tokens[token] -= 1
    if g.download_tokens[token] <= 0:
        del g.download_tokens[token]
        session.pop(f'download_path_{token}', None)
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    return send_file(output, as_attachment=True, download_name=filename, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')




@batch_bp.route('/batch_result/<task_id>', methods=['GET'])
def download_batch_result(task_id):
    """Download a permanently-stored batch result ZIP (HTML + Excel)."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "请先登录"}), 401
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT zip_path FROM batch_comparison_results WHERE task_id = %s", (task_id,))
            row = cur.fetchone()
            if not row or not os.path.exists(row[0]):
                return jsonify({"error": "Report not found"}), 404
    return send_file(row[0], as_attachment=True, download_name=f"batch_report_{task_id}.zip",
                    mimetype='application/zip')


@batch_bp.route('/list_batch_results', methods=['GET'])
def list_batch_results():
    """Return ALL batch comparison results visible to all registered users."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "请先登录"}), 401
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT r.id, r.task_id, r.file_count, r.pair_count, r.max_risk, r.file_names, r.created_at,
                       u.username as created_by_name
                FROM batch_comparison_results r
                JOIN users u ON r.user_id = u.user_id
                ORDER BY r.created_at DESC LIMIT 30
            """)
            results = cur.fetchall()
    return jsonify({"results": [dict(r) for r in results]})


@batch_bp.route('/delete_batch_result/<int:id>', methods=['POST'])
def delete_batch_result(id):
    """Admin only: delete a batch result."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "请先登录"}), 401
    if session.get('role') != 'admin':
        return jsonify({"error": "仅管理员可删除"}), 403
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT zip_path FROM batch_comparison_results WHERE id = %s", (id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Not found"}), 404
            if os.path.exists(row[0]):
                os.remove(row[0])
            cur.execute("DELETE FROM batch_comparison_results WHERE id = %s", (id,))
            conn.commit()
    return jsonify({"success": True})


# ── Standalone quote anomaly endpoints ──

@batch_bp.route('/check_quote_anomaly', methods=['POST'])
def check_quote_anomaly_standalone():
    """Standalone endpoint: detect quote anomalies in a single bid document."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported file type: {f.filename}"}), 400

    from app.services.quote_anomaly import check_quote_anomaly as run_qa
    from app.services.audit_logger import AuditLogger

    text, _ = extract_text_from_file(f)
    if not text or text.startswith("["):
        return jsonify({"error": "Could not extract text from file"}), 400

    _audit = AuditLogger("quote_anomaly_standalone", f.filename)
    result = run_qa(text, doc_name=f.filename, audit=_audit)
    _audit.result(risk_score=round(result.risk_score, 1), doc_name=f.filename)

    return jsonify({
        "doc_name": result.doc_name,
        "prices": result.prices[:20],
        "percentages": result.percentages[:20],
        "cv": result.cv,
        "same_rate_flag": result.same_rate_flag,
        "abnormal_drop_flag": result.abnormal_drop_flag,
        "clustering_flag": result.clustering_flag,
        "benford_deviation": result.benford_deviation,
        "risk_score": result.risk_score,
        "details": result.details,
        "daxie_mismatches": result.daxie_mismatches,
    })


@batch_bp.route('/compare_bidders_quotes', methods=['POST'])
def compare_bidders_quotes_endpoint():
    """Standalone endpoint: cross-bidder quote comparison without full batch compare."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({"error": "Need at least 2 files"}), 400
    if len(files) > 10:
        return jsonify({"error": "Maximum 10 files allowed"}), 400

    from app.services.quote_anomaly import compare_bidders_quotes as run_cbq, save_quote_anomaly_results
    from app.services.audit_logger import AuditLogger

    file_data = []
    for f in files:
        if not f.filename or not allowed_file(f.filename):
            continue
        text, _ = extract_text_from_file(f)
        if text and not text.startswith("["):
            file_data.append({'filename': f.filename, 'text': text})

    if len(file_data) < 2:
        return jsonify({"error": "Could not extract valid text from at least 2 files"}), 400

    thread_id = session.get('thread_id', '')
    _audit = AuditLogger("compare_bidders_quotes", thread_id[:12] if thread_id else "")
    result = run_cbq(file_data, audit=_audit)
    save_quote_anomaly_results(user_id, thread_id or str(uuid.uuid4()),
                               result['per_bidder'], result)
    _audit.result(max_risk=round(result.get('max_risk_score', 0), 1),
                  bidders=len(file_data))

    return jsonify(result)


# ── Standalone relationship extraction endpoints ──

@batch_bp.route('/extract_relationships', methods=['POST'])
def extract_relationships_endpoint():
    """Standalone endpoint: extract entity relationships from bid documents."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    files = request.files.getlist('files')
    if len(files) < 1:
        return jsonify({"error": "Need at least 1 file"}), 400
    if len(files) > 10:
        return jsonify({"error": "Maximum 10 files allowed"}), 400

    from app.services.relationship_extractor import extract_relationships, save_relationship_results
    from app.services.audit_logger import AuditLogger

    file_data = []
    for f in files:
        if not f.filename or not allowed_file(f.filename):
            continue
        text, _ = extract_text_from_file(f)
        if text and not text.startswith("["):
            f.seek(0)
            meta = extract_metadata(f)
            file_data.append({'filename': f.filename, 'text': text, 'metadata': meta})

    if not file_data:
        return jsonify({"error": "Could not extract valid text from any file"}), 400

    thread_id = session.get('thread_id', '')
    task_id = thread_id or str(uuid.uuid4())
    _audit = AuditLogger("relationship_extraction", task_id[:12] if task_id else "")

    report = extract_relationships(file_data, audit=_audit)
    save_relationship_results(user_id, task_id, report)
    _audit.result(risk_score=report.risk_score, entities=len(report.entities),
                  relations=len(report.relationships), flags=len(report.red_flags))

    return jsonify({
        "entities": [{'text': e.text, 'type': e.entity_type, 'confidence': e.confidence}
                     for e in report.entities[:50]],
        "relationships": [{
            'source': r.source_entity, 'target': r.target_entity,
            'type': r.relation_type, 'subtype': r.relation_subtype,
            'confidence': r.confidence, 'evidence': r.evidence[:200],
            'module': r.module, 'risk_flag': r.risk_flag,
            'risk_reason': r.risk_reason,
        } for r in report.relationships],
        "red_flags": report.red_flags,
        "risk_score": report.risk_score,
        "modules_run": report.modules_run,
        "company_personnel_map": report.company_personnel_map,
    })


# ── Standalone typo detection endpoint ──

@batch_bp.route('/check_typos', methods=['POST'])
def check_typos_endpoint():
    """Standalone endpoint: detect typos in a single document."""
    if session.get('consent_value', 0) != 1:
        return jsonify({"error": "Consent not given"}), 403
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    diff_mode = request.form.get('diff_mode', 'false').lower() == 'true'

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported file type: {f.filename}"}), 400

    from app.services.typo_detector import detect_typos, save_typo_results
    from app.services.audit_logger import AuditLogger

    text, _ = extract_text_from_file(f)
    if not text or text.startswith("["):
        return jsonify({"error": "Could not extract text from file"}), 400

    _audit = AuditLogger("typo_detection", f.filename)
    report = detect_typos(text, doc_name=f.filename, audit=_audit)

    thread_id = session.get('thread_id', '')
    save_typo_results(user_id, thread_id or str(uuid.uuid4()), {f.filename: report})
    _audit.result(total=report.total_suspects, critical=report.critical_count,
                  layers=','.join(report.layers_run))

    result = {
        "doc_name": f.filename,
        "total_suspects": report.total_suspects,
        "critical_count": report.critical_count,
        "layers_run": report.layers_run,
        "findings": [{
            'layer': f.layer,
            'suspect_text': f.suspect_text,
            'suggestions': f.suggestions,
            'confidence': f.confidence,
            'context_snippet': f.context_snippet,
            'severity': f.severity,
            'is_daxie_error': f.is_daxie_error,
            'daxie_expected': f.daxie_expected,
            'daxie_actual': f.daxie_actual,
        } for f in report.findings],
    }

    if diff_mode and report.diff_text:
        result['diff_text'] = report.diff_text[:5000]

    return jsonify(result)
