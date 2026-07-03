"""Compliance checking API routes.

Endpoints:
    POST /compliance/extract_rules     — Extract rules from uploaded bidding document
    POST /compliance/check             — Run compliance check (async, returns task_id)
    GET  /compliance/result/<task_id>  — Get check results
    GET  /compliance/laws              — List built-in + custom laws
    POST /compliance/laws/upload       — Upload custom law document
    DELETE /compliance/laws/<law_id>   — Remove a custom law
"""
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request, session
from app.config import DATA_DIR

logger = logging.getLogger(__name__)

compliance_bp = Blueprint('compliance', __name__, url_prefix='/compliance')

# ── In-memory result store (persisted to files) ──
COMPLIANCE_DIR = os.path.join(str(DATA_DIR), 'compliance_results')
os.makedirs(COMPLIANCE_DIR, exist_ok=True)


def _save_result(task_id: str, data: dict):
    path = os.path.join(COMPLIANCE_DIR, f"{task_id}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)


def _load_result(task_id: str) -> dict | None:
    path = os.path.join(COMPLIANCE_DIR, f"{task_id}.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ── Law storage ──
LAWS_DIR = os.path.join(DATA_DIR, 'user_laws')
os.makedirs(LAWS_DIR, exist_ok=True)


@compliance_bp.route('/laws', methods=['GET'])
def list_laws():
    """List all available laws (built-in seed + user-uploaded)."""
    try:
        from app.services.compliance_checker import _get_seed_laws
        seed = _get_seed_laws()

        # Group seed laws by law_name
        laws_map = {}
        for art in seed:
            name = art["law_name"]
            if name not in laws_map:
                laws_map[name] = {
                    "law_name": name,
                    "short_name": art.get("short_name", name),
                    "category": art.get("category", ""),
                    "source": "built-in",
                    "article_count": 0,
                    "articles": [],
                }
            laws_map[name]["articles"].append({
                "article": art["article"],
                "text": art["text"],
                "tags": art.get("tags", []),
            })
            laws_map[name]["article_count"] += 1

        # User-uploaded laws
        user_laws = []
        if os.path.exists(LAWS_DIR):
            for fn in os.listdir(LAWS_DIR):
                if fn.endswith('.json'):
                    try:
                        with open(os.path.join(LAWS_DIR, fn), 'r', encoding='utf-8') as f:
                            ul = json.load(f)
                        ul["source"] = "user"
                        ul["id"] = fn.replace('.json', '')
                        user_laws.append(ul)
                    except Exception:
                        pass

        return jsonify({
            "built_in": list(laws_map.values()),
            "user_laws": user_laws,
            "total_articles": len(seed),
        })
    except Exception as e:
        logger.error(f"List laws error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@compliance_bp.route('/laws/upload', methods=['POST'])
def upload_law():
    """Upload a user-defined law document for rule extraction."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "请上传法规文件"}), 400
        f = request.files['file']
        if not f.filename:
            return jsonify({"error": "文件名不能为空"}), 400

        # Extract text
        from app.services.file_processing import extract_text_from_file
        text, _ = extract_text_from_file(f)
        if not text or text.startswith('['):
            return jsonify({"error": "无法提取文件文本"}), 400

        # Extract law articles using AI
        from app.services.rule_extractor import RuleExtractor
        extractor = RuleExtractor()
        # For laws, we use a different extraction (just regex for now, no AI cost)
        rules_result = extractor.extract(text, f.filename, use_ai=False)

        if not rules_result.get("rules"):
            return jsonify({"error": "未能从文件中提取到法律条款"}), 400

        # Save as user law
        law_id = str(uuid.uuid4())[:8]
        law_data = {
            "law_name": os.path.splitext(f.filename)[0],
            "short_name": os.path.splitext(f.filename)[0],
            "source": "user",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "articles": rules_result["rules"],
        }
        with open(os.path.join(LAWS_DIR, f"{law_id}.json"), 'w', encoding='utf-8') as fw:
            json.dump(law_data, fw, ensure_ascii=False, default=str)

        return jsonify({
            "success": True,
            "law_id": law_id,
            "article_count": len(rules_result["rules"]),
            "message": f"已导入 {len(rules_result['rules'])} 条法律条款",
        })
    except Exception as e:
        logger.error(f"Upload law error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@compliance_bp.route('/laws/<law_id>', methods=['DELETE'])
def delete_law(law_id):
    """Delete a user-uploaded law."""
    path = os.path.join(LAWS_DIR, f"{law_id}.json")
    if os.path.exists(path):
        os.unlink(path)
        return jsonify({"success": True})
    return jsonify({"error": "法规不存在"}), 404


@compliance_bp.route('/extract_rules', methods=['POST'])
def extract_rules():
    """Extract rules from an uploaded bidding document (reference file).

    Request: multipart/form-data with 'file' field
    Response: {rules: [...], ai_count, regex_count, total, doc_name}
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "请上传招标文件"}), 400
        f = request.files['file']
        if not f.filename:
            return jsonify({"error": "文件名不能为空"}), 400

        use_ai = request.form.get('use_ai', 'true').lower() != 'false'

        # Extract text
        from app.services.file_processing import extract_text_from_file
        text, _ = extract_text_from_file(f)
        if not text or text.startswith('['):
            return jsonify({"error": "无法提取文件文本，请检查文件格式"}), 400

        # Extract rules
        from app.services.rule_extractor import RuleExtractor
        extractor = RuleExtractor()
        result = extractor.extract(text, f.filename, use_ai=use_ai)

        # Cache the rules for later use
        task_id = str(uuid.uuid4())
        _save_result(f"rules_{task_id}", result)

        return jsonify({
            "success": True,
            "task_id": task_id,
            **result,
        })
    except Exception as e:
        logger.error(f"Extract rules error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@compliance_bp.route('/check', methods=['POST'])
def start_check():
    """Start a compliance check (async via Celery or sync fallback).

    Request: JSON
    {
        "rules_task_id": "...",        // from extract_rules
        "bid_file_text": "...",        // OR upload bid file directly
        "bid_file_name": "...",
        "use_ai": true,
        "include_laws": true
    }
    OR: multipart/form-data with 'bid_file' + 'rules_task_id'

    Returns: {task_id, status: "queued"}
    """
    try:
        rules_task_id = None
        bid_text = None
        bid_name = ""
        use_ai = True
        include_laws = True

        if request.is_json:
            data = request.get_json() or {}
            rules_task_id = data.get('rules_task_id')
            bid_text = data.get('bid_file_text')
            bid_name = data.get('bid_file_name', '投标文件')
            use_ai = data.get('use_ai', True)
            include_laws = data.get('include_laws', True)
        else:
            rules_task_id = request.form.get('rules_task_id')
            use_ai = request.form.get('use_ai', 'true').lower() != 'false'
            include_laws = request.form.get('include_laws', 'true').lower() != 'false'
            if 'bid_file' in request.files:
                f = request.files['bid_file']
                if f.filename:
                    bid_name = f.filename
                    from app.services.file_processing import extract_text_from_file
                    bid_text, _ = extract_text_from_file(f)

        if not rules_task_id:
            return jsonify({"error": "缺少 rules_task_id，请先提取招标文件规则"}), 400
        if not bid_text:
            return jsonify({"error": "缺少投标文件内容"}), 400

        # Load rules
        rules_data = _load_result(f"rules_{rules_task_id}")
        if not rules_data:
            return jsonify({"error": "规则数据已过期，请重新提取"}), 404
        rules = rules_data.get("rules", [])
        if not rules:
            return jsonify({"error": "规则为空，请重新提取招标文件规则"}), 400

        # Try Celery async
        task_id = str(uuid.uuid4())
        try:
            from celery_app import compliance_check_task
            compliance_check_task.apply_async(
                args=[task_id, bid_text, rules, bid_name, use_ai, include_laws],
                task_id=task_id,
            )
            logger.info(f"Compliance check queued: {task_id}")
            return jsonify({
                "task_id": task_id,
                "status": "queued",
                "message": "合规检查已提交，请轮询结果",
            })
        except (ImportError, Exception) as e:
            logger.warning(f"Celery unavailable, running sync: {e}")
            # Sync fallback
            result = _run_check_sync(task_id, bid_text, rules, bid_name, use_ai, include_laws)
            return jsonify({
                "task_id": task_id,
                "status": "completed",
                **result,
            })

    except Exception as e:
        logger.error(f"Start check error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _run_check_sync(task_id: str, bid_text: str, rules: list, bid_name: str,
                    use_ai: bool, include_laws: bool) -> dict:
    """Run compliance check synchronously (Celery fallback)."""
    from app.services.compliance_checker import ComplianceChecker

    checker = ComplianceChecker()
    result = checker.check(bid_text, rules, bid_name, use_ai=use_ai)

    # Generate report
    report_html = checker.generate_report(
        result["results"], bid_name, len(rules), use_ai=use_ai
    )

    output = {
        "success": True,
        "bid_name": bid_name,
        "rule_count": len(rules),
        "summary": result["summary"],
        "results": result["results"],
        "laws_applied": result.get("laws_applied", []),
        "report_html": report_html,
        "ai_used": result.get("ai_used", False),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_result(task_id, output)
    return output


@compliance_bp.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    """Get compliance check result by task ID."""
    data = _load_result(task_id)
    if not data:
        # Check Celery
        try:
            from celery.result import AsyncResult
            from celery_app import celery_app
            task = AsyncResult(task_id, app=celery_app)
            if task.state == 'PENDING':
                return jsonify({"status": "pending", "message": "检查进行中..."})
            elif task.state == 'FAILED':
                return jsonify({"status": "failed", "error": str(task.info)})
            elif task.state == 'SUCCESS':
                result = task.result
                if isinstance(result, dict):
                    return jsonify({"status": "completed", **result})
        except ImportError:
            pass
        return jsonify({"status": "not_found", "error": "检查结果不存在或已过期"}), 404

    return jsonify({"status": "completed", **data})


@compliance_bp.route('/rules/<task_id>', methods=['GET'])
def get_rules(task_id):
    """Get extracted rules by task ID (for user review before check)."""
    data = _load_result(f"rules_{task_id}")
    if not data:
        return jsonify({"error": "规则数据不存在或已过期"}), 404
    return jsonify(data)


@compliance_bp.route('/rules/<task_id>', methods=['PUT'])
def update_rules(task_id):
    """Update/modify extracted rules (user review/edit)."""
    data = _load_result(f"rules_{task_id}")
    if not data:
        return jsonify({"error": "规则数据不存在或已过期"}), 404

    updates = request.get_json() or {}
    if "rules" in updates:
        data["rules"] = updates["rules"]
        data["total"] = len(updates["rules"])
        data["user_modified"] = True
    if "deleted_ids" in updates:
        ids = set(updates["deleted_ids"])
        data["rules"] = [r for r in data["rules"] if r.get("rule_id") not in ids]
        data["total"] = len(data["rules"])
        data["user_modified"] = True
    if "added_rules" in updates:
        data["rules"].extend(updates["added_rules"])
        data["total"] = len(data["rules"])
        data["user_modified"] = True

    _save_result(f"rules_{task_id}", data)
    return jsonify({"success": True, "total": data["total"]})


# ── Feedback Endpoints ──

@compliance_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit forced feedback on a compliance check result.

    Each file in a check result MUST get a verdict from the user:
      - true_violation: AI correctly flagged a real issue
      - false_positive: AI flagged something that doesn't matter
      - not_matter: the violation exists but is irrelevant to the decision

    Stored in DB for LoRA fine-tuning pipeline.

    Request JSON:
    {
        "task_id": "original check task_id",
        "check_file_name": "投标文件名.docx",
        "user_verdict": "true_violation | false_positive | not_matter",
        "user_explain": "brief explanation why"
    }
    """
    try:
        data = request.get_json() or {}
        task_id = data.get('task_id')
        check_file = data.get('check_file_name')
        user_verdict = data.get('user_verdict')
        user_explain = data.get('user_explain', '')

        if not all([task_id, check_file, user_verdict]):
            return jsonify({"error": "缺少必填字段: task_id, check_file_name, user_verdict"}), 400

        valid_verdicts = {'true_violation', 'false_positive', 'not_matter'}
        if user_verdict not in valid_verdicts:
            return jsonify({"error": f"user_verdict 必须为: {', '.join(valid_verdicts)}"}), 400

        # Load original check result
        orig = _load_result(task_id)
        if not orig:
            return jsonify({"error": "检查结果不存在"}), 404

        # Determine user context
        user_id = session.get('user_id', 'anonymous')
        bid_doc = orig.get('bid_name', '')
        rule_count = orig.get('rule_count', 0)
        summary = orig.get('summary', {})

        # AI verdict summary
        if summary.get('critical', 0) > 0:
            ai_verdict = 'critical'
        elif summary.get('violation', 0) > 0:
            ai_verdict = 'violation'
        elif summary.get('warning', 0) > 0:
            ai_verdict = 'warning'
        else:
            ai_verdict = 'pass'

        # Store in DB
        from app.database import get_db_connection
        import json as _json

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO compliance_feedback
                        (user_id, task_id, bid_doc_name, check_file_name,
                         rule_count, summary_json, ai_verdict,
                         user_verdict, user_explain, results_json, report_html)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id, task_id, bid_doc, check_file,
                    rule_count,
                    _json.dumps(summary, ensure_ascii=False),
                    ai_verdict,
                    user_verdict, user_explain,
                    _json.dumps(orig.get('results', []), ensure_ascii=False),
                    orig.get('report_html', ''),
                ))
                conn.commit()

        logger.info(
            f"Compliance feedback saved: task={task_id} file={check_file} "
            f"ai={ai_verdict} user={user_verdict} by={user_id}"
        )

        return jsonify({"success": True, "message": "反馈已保存"})
    except Exception as e:
        logger.error(f"Submit feedback error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@compliance_bp.route('/feedback/history', methods=['GET'])
def feedback_history():
    """List saved feedback records."""
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        from app.database import get_db_connection

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, user_id, task_id, bid_doc_name, check_file_name,
                           rule_count, ai_verdict, user_verdict, user_explain,
                           created_at
                    FROM compliance_feedback
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
                rows = cur.fetchall()

                cur.execute("SELECT COUNT(*) FROM compliance_feedback")
                total = cur.fetchone()[0]

        return jsonify({
            "total": total,
            "limit": limit,
            "offset": offset,
            "records": [{
                "id": r[0], "user_id": r[1], "task_id": r[2],
                "bid_doc_name": r[3], "check_file_name": r[4],
                "rule_count": r[5], "ai_verdict": r[6],
                "user_verdict": r[7], "user_explain": r[8],
                "created_at": r[9].isoformat() if r[9] else None,
            } for r in rows],
        })
    except Exception as e:
        logger.error(f"Feedback history error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@compliance_bp.route('/training_data', methods=['GET'])
def export_training_data():
    """Export feedback data in LoRA fine-tuning format.

    Returns JSONL-compatible array of training samples:
    {
        "instruction": "compliance check system prompt",
        "input": "bid document text + rules + check results",
        "output": "user verdict + explanation",
        "metadata": {user_verdict, ai_verdict, ...}
    }
    """
    try:
        limit = request.args.get('limit', 200, type=int)
        min_samples = request.args.get('min_samples', 10, type=int)
        from app.database import get_db_connection

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, user_id, task_id, bid_doc_name, check_file_name,
                           rule_count, summary_json, ai_verdict,
                           user_verdict, user_explain, results_json, created_at
                    FROM compliance_feedback
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
                rows = cur.fetchall()

        if len(rows) < min_samples:
            return jsonify({
                "ready": False,
                "message": f"需要至少 {min_samples} 条反馈才能导出（当前 {len(rows)} 条）",
                "current_count": len(rows),
                "min_required": min_samples,
            })

        from app.services.compliance_prompts import COMPLIANCE_CHECK_SYSTEM
        import json as _json

        samples = []
        for r in rows:
            summary = r[6] if isinstance(r[6], dict) else _json.loads(r[6] or '{}')
            results = r[9] if isinstance(r[9], list) else _json.loads(r[9] or '[]')

            # Build a compact training sample
            violations = []
            for res in results:
                if res.get('verdict') in ('VIOLATION', 'CRITICAL'):
                    violations.append({
                        'rule_id': res.get('rule_id'),
                        'verdict': res.get('verdict'),
                        'reasoning': res.get('reasoning', ''),
                        'evidence': res.get('evidence', ''),
                    })

            samples.append({
                "instruction": COMPLIANCE_CHECK_SYSTEM[:500],
                "input": {
                    "bid_doc": r[3],
                    "check_file": r[4],
                    "rule_count": r[5],
                    "ai_verdict": r[7],
                    "violations_found": violations[:5],
                },
                "output": {
                    "user_verdict": r[8],
                    "user_explain": r[9],
                },
                "metadata": {
                    "feedback_id": r[0],
                    "user_id": r[1],
                    "task_id": r[2],
                    "created_at": r[11].isoformat() if r[11] else None,
                },
            })

        # Stats
        verdict_counts = {}
        for s in samples:
            v = s['output']['user_verdict']
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

        logger.info(
            f"Training data export: {len(samples)} samples, "
            f"distribution: {verdict_counts}"
        )

        return jsonify({
            "ready": True,
            "total_samples": len(samples),
            "verdict_distribution": verdict_counts,
            "samples": samples,
            "export_ts": datetime.now(timezone.utc).isoformat(),
            "note": "用于 Unsloth LoRA 微调。导出为 JSONL 格式：每行一个训练样本。",
        })
    except Exception as e:
        logger.error(f"Training data export error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
