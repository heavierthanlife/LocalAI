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
from app.utils.helpers import ok, err
from app import limiter
from functools import wraps

logger = logging.getLogger(__name__)

compliance_bp = Blueprint('compliance', __name__, url_prefix='/compliance')


def _login_required(f):
    """Decorator: require registered user."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get('consent_value', 0) != 1:
            return err("请先登录", "AUTH_REQUIRED", 401)
        return f(*args, **kwargs)
    return wrapper

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
@_login_required
def list_laws():
    """List all available laws (DB built-in + user-uploaded)."""
    try:
        from app.services.compliance_checker import _get_seed_laws
        from app.database import get_db_connection
        from psycopg2.extras import RealDictCursor

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

        # Enrich with DB metadata (versions, effective_date, etc.)
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT lm.law_name, lm.category, lm.issuing_authority,
                               lm.effective_date, lm.expiry_date, lm.status, lm.scope,
                               lm.id AS law_id, COUNT(lv.id) AS version_count
                        FROM law_masters lm
                        LEFT JOIN law_versions lv ON lv.law_id = lm.id
                        WHERE lm.status = 'active'
                        GROUP BY lm.id
                    """)
                    for row in cur.fetchall():
                        name = row["law_name"]
                        if name in laws_map:
                            laws_map[name]["issuing_authority"] = row.get("issuing_authority")
                            laws_map[name]["effective_date"] = str(row["effective_date"]) if row.get("effective_date") else None
                            laws_map[name]["expiry_date"] = str(row["expiry_date"]) if row.get("expiry_date") else None
                            laws_map[name]["status"] = row.get("status")
                            laws_map[name]["scope"] = row.get("scope")
                            laws_map[name]["version_count"] = row.get("version_count", 1)
                            laws_map[name]["law_id"] = row.get("law_id")
        except Exception:
            logger.warning("Failed to enrich law list with DB metadata")

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
                        logger.warning(f"Failed to load user law: {fn}")

        return ok({
            "built_in": list(laws_map.values()),
            "user_laws": user_laws,
            "total_articles": len(seed),
        })
    except Exception as e:
        logger.error(f"List laws error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@compliance_bp.route('/laws/upload', methods=['POST'])
@limiter.limit("10/minute")
@_login_required
def upload_law():
    """Upload a user-defined law document for rule extraction."""
    try:
        if 'file' not in request.files:
            return err("请上传法规文件", "VALIDATION_ERROR", 400)
        f = request.files['file']
        if not f.filename:
            return err("文件名不能为空", "VALIDATION_ERROR", 400)

        # Extract text
        from app.services.file_processing import extract_text_from_file
        text, _ = extract_text_from_file(f)
        if not text or text.startswith('['):
            return err("无法提取文件文本", "VALIDATION_ERROR", 400)

        # Extract law articles using AI
        from app.services.rule_extractor import RuleExtractor
        extractor = RuleExtractor()
        # For laws, we use a different extraction (just regex for now, no AI cost)
        rules_result = extractor.extract(text, f.filename, use_ai=False)

        if not rules_result.get("rules"):
            return err("未能从文件中提取到法律条款", "VALIDATION_ERROR", 400)

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

        return ok({
            "success": True,
            "law_id": law_id,
            "article_count": len(rules_result["rules"]),
            "message": f"已导入 {len(rules_result['rules'])} 条法律条款",
        })
    except Exception as e:
        logger.error(f"Upload law error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@compliance_bp.route('/laws/<law_id>', methods=['DELETE'])
@_login_required
def delete_law(law_id):
    """Delete a user-uploaded law."""
    path = os.path.join(LAWS_DIR, f"{law_id}.json")
    if os.path.exists(path):
        os.unlink(path)
        return ok(message="ok")
    return err("法规不存在", "NOT_FOUND", 404)


@compliance_bp.route('/extract_rules', methods=['POST'])
@limiter.limit("10/minute")
@_login_required
def extract_rules():
    """Extract rules from an uploaded bidding document (reference file).

    Request: multipart/form-data with 'file' field
    Response: {rules: [...], ai_count, regex_count, total, doc_name}
    """
    try:
        if 'file' not in request.files:
            return err("请上传招标文件", "VALIDATION_ERROR", 400)
        f = request.files['file']
        if not f.filename:
            return err("文件名不能为空", "VALIDATION_ERROR", 400)

        use_ai = request.form.get('use_ai', 'true').lower() != 'false'

        # Extract text
        from app.services.file_processing import extract_text_from_file
        text, _ = extract_text_from_file(f)
        if not text or text.startswith('['):
            return err("无法提取文件文本，请检查文件格式", "VALIDATION_ERROR", 400)

        # Extract rules
        from app.services.rule_extractor import RuleExtractor
        extractor = RuleExtractor()
        result = extractor.extract(text, f.filename, use_ai=use_ai)

        # Cache the rules for later use
        task_id = str(uuid.uuid4())
        _save_result(f"rules_{task_id}", result)

        return ok({
            "success": True,
            "task_id": task_id,
            **result,
        })
    except Exception as e:
        logger.error(f"Extract rules error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@compliance_bp.route('/check', methods=['POST'])
@limiter.limit("10/minute")
@_login_required
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
            region_code = data.get('region_code')
        else:
            rules_task_id = request.form.get('rules_task_id')
            use_ai = request.form.get('use_ai', 'true').lower() != 'false'
            include_laws = request.form.get('include_laws', 'true').lower() != 'false'
            region_code = request.form.get('region_code')
            if 'bid_file' in request.files:
                f = request.files['bid_file']
                if f.filename:
                    bid_name = f.filename
                    from app.services.file_processing import extract_text_from_file
                    bid_text, _ = extract_text_from_file(f)

        if not rules_task_id:
            return err("缺少 rules_task_id，请先提取招标文件规则", "VALIDATION_ERROR", 400)
        if not bid_text:
            return err("缺少投标文件内容", "VALIDATION_ERROR", 400)

        # Load rules
        rules_data = _load_result(f"rules_{rules_task_id}")
        if not rules_data:
            return err("规则数据已过期，请重新提取", "NOT_FOUND", 404)
        rules = rules_data.get("rules", [])
        if not rules:
            return err("规则为空，请重新提取招标文件规则", "VALIDATION_ERROR", 400)

        # Try Celery async
        task_id = str(uuid.uuid4())
        try:
            from celery_app import compliance_check_task
            compliance_check_task.apply_async(
                args=[task_id, bid_text, rules, bid_name, use_ai, include_laws, region_code],
                task_id=task_id,
            )
            logger.info(f"Compliance check queued: {task_id}")
            return ok({
                "task_id": task_id,
                "status": "queued",
                "message": "合规检查已提交，请轮询结果",
            })
        except (ImportError, Exception) as e:
            logger.warning(f"Celery unavailable, running sync: {e}")
            # Sync fallback
            result = _run_check_sync(task_id, bid_text, rules, bid_name, use_ai, include_laws, region_code)
            return ok({
                "task_id": task_id,
                "status": "completed",
                **result,
            })

    except Exception as e:
        logger.error(f"Start check error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


def _run_check_sync(task_id: str, bid_text: str, rules: list, bid_name: str,
                    use_ai: bool, include_laws: bool, region_code: str = None) -> dict:
    """Run compliance check synchronously (Celery fallback)."""
    from app.services.compliance_checker import ComplianceChecker

    checker = ComplianceChecker()
    result = checker.check(bid_text, rules, bid_name, use_ai=use_ai, region_code=region_code)

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


@compliance_bp.route('/incremental_check', methods=['POST'])
@limiter.limit("20/minute")
@_login_required
def incremental_check():
    """Incremental compliance check — only re-checks modified sections (U9a).

    Caches section results via Redis + memory fallback.

    Request JSON:
    {
        "rules_task_id": "...",
        "bid_doc_name": "投标文件.pdf",
        "changed_sections": [{"id": "sec-1", "title": "...", "content": "..."}, ...],
        "use_ai": true,
        "region_code": "440000"  // optional
    }
    """
    try:
        data = request.get_json() or {}
        rules_task_id = data.get('rules_task_id')
        bid_doc_name = data.get('bid_doc_name', 'unknown')
        changed_sections = data.get('changed_sections', [])
        use_ai = data.get('use_ai', True)
        region_code = data.get('region_code')

        if not rules_task_id:
            return err("缺少 rules_task_id", "VALIDATION_ERROR", 400)
        if not changed_sections:
            return err("缺少 changed_sections", "VALIDATION_ERROR", 400)

        rules_data = _load_result(f"rules_{rules_task_id}")
        if not rules_data:
            return err("规则数据已过期，请重新提取", "NOT_FOUND", 404)
        rules = rules_data.get("rules", [])
        if not rules:
            return err("规则为空", "VALIDATION_ERROR", 400)

        from app.services.incremental_check import incremental_check as inc_check
        result = inc_check(
            bid_text='',  # individual sections provide content
            bid_doc_name=bid_doc_name,
            rules=rules,
            changed_sections=changed_sections,
            use_ai=use_ai,
            region_code=region_code,
        )
        return ok(result)
    except Exception as e:
        logger.error(f"incremental_check error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@compliance_bp.route('/result/<task_id>', methods=['GET'])
@_login_required
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
                return ok({"status": "pending", "message": "检查进行中..."})
            elif task.state == 'FAILED':
                return ok({"status": "failed", "error": str(task.info)})
            elif task.state == 'SUCCESS':
                result = task.result
                if isinstance(result, dict):
                    return ok({"status": "completed", **result})
        except ImportError:
            pass
        return ok({"status": "not_found", "error": "检查结果不存在或已过期"}), 404

    return ok({"status": "completed", **data})


@compliance_bp.route('/rules/<task_id>', methods=['GET'])
@_login_required
def get_rules(task_id):
    """Get extracted rules by task ID (for user review before check)."""
    data = _load_result(f"rules_{task_id}")
    if not data:
        return err("规则数据不存在或已过期", "NOT_FOUND", 404)
    return ok(data)


@compliance_bp.route('/rules/<task_id>', methods=['PUT'])
@_login_required
def update_rules(task_id):
    """Update/modify extracted rules (user review/edit)."""
    data = _load_result(f"rules_{task_id}")
    if not data:
        return err("规则数据不存在或已过期", "NOT_FOUND", 404)

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
    return ok({"total": data["total"]})


# ── Feedback Endpoints ──

@compliance_bp.route('/feedback', methods=['POST'])
@_login_required
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
            return err("缺少必填字段: task_id, check_file_name, user_verdict", "VALIDATION_ERROR", 400)

        valid_verdicts = {'true_violation', 'false_positive', 'not_matter'}
        if user_verdict not in valid_verdicts:
            return err(f"user_verdict 必须为: {', '.join(valid_verdicts)}", "VALIDATION_ERROR", 400)

        # Load original check result
        orig = _load_result(task_id)
        if not orig:
            return err("检查结果不存在", "NOT_FOUND", 404)

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

        # Dual-write to training_logger for unified training pipeline
        try:
            _rating_map = {'true_violation': 5, 'false_positive': 1, 'not_matter': 3}
            from app.services.training_logger import log_interaction
            log_interaction(
                thread_id=f"compliance_{task_id}_{check_file[:40]}",
                user_msg=f"标书: {bid_doc}\n检查文件: {check_file}\n规则数: {rule_count}\nAI判定: {ai_verdict}",
                assistant_response=f"用户判定: {user_verdict}\n说明: {user_explain}",
                rating=_rating_map.get(user_verdict, 3),
                source='compliance',
                model='compliance_checker',
            )
        except Exception:
            logger.warning("Failed to log compliance feedback to training", exc_info=True)

        return ok({"message": "反馈已保存"})
    except Exception as e:
        logger.error(f"Submit feedback error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@compliance_bp.route('/feedback/history', methods=['GET'])
@_login_required
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

        return ok({
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
        return err(str(e), "SERVER_ERROR", 500)


@compliance_bp.route('/training_data', methods=['GET'])
@_login_required
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
            return ok({
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

        return ok({
            "ready": True,
            "total_samples": len(samples),
            "verdict_distribution": verdict_counts,
            "samples": samples,
            "export_ts": datetime.now(timezone.utc).isoformat(),
            "note": "用于 Unsloth LoRA 微调。导出为 JSONL 格式：每行一个训练样本。",
        })
    except Exception as e:
        logger.error(f"Training data export error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


# ── Law Version Management (U3) ──

@compliance_bp.route('/laws/<int:law_id>/versions', methods=['GET'])
@_login_required
def list_law_versions(law_id: int):
    """List all versions of a DB law."""
    try:
        from app.services.law_version import list_versions
        versions = list_versions(law_id)
        return ok({"law_id": law_id, "versions": versions})
    except Exception as e:
        logger.error(f"list_law_versions error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@compliance_bp.route('/laws/<int:law_id>/versions', methods=['POST'])
@_login_required
def create_law_version(law_id: int):
    """Create a new version for a DB law."""
    try:
        if not request.is_json:
            return err("请求体需为 JSON", "VALIDATION_ERROR", 400)
        data = request.get_json() or {}
        label = data.get('version_label')
        articles = data.get('articles', [])
        if not label:
            return err("缺少 version_label", "VALIDATION_ERROR", 400)
        if not articles:
            return err("缺少 articles", "VALIDATION_ERROR", 400)
        from app.services.law_version import create_version
        result = create_version(
            law_id, label, articles,
            version_date=data.get('version_date'),
            change_summary=data.get('change_summary'),
        )
        return ok(result, message="版本已创建")
    except ValueError as e:
        return err(str(e), "NOT_FOUND", 404)
    except Exception as e:
        logger.error(f"create_law_version error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@compliance_bp.route('/laws/<int:law_id>/versions/activate', methods=['POST'])
@_login_required
def activate_law_version(law_id: int):
    """Activate a specific version (admin only)."""
    if session.get('role') != 'admin':
        return err("Admin access required", "FORBIDDEN", 403)
    try:
        if not request.is_json:
            return err("请求体需为 JSON", "VALIDATION_ERROR", 400)
        data = request.get_json() or {}
        version_id = data.get('version_id')
        if not version_id:
            return err("缺少 version_id", "VALIDATION_ERROR", 400)
        from app.services.law_version import activate_version
        result = activate_version(law_id, version_id)
        if result is None:
            return err("版本不存在", "NOT_FOUND", 404)
        return ok(result, message="版本已激活")
    except Exception as e:
        logger.error(f"activate_law_version error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@compliance_bp.route('/laws/<int:law_id>/diff', methods=['GET'])
@_login_required
def get_law_diff(law_id: int):
    """Get diff between two law versions."""
    try:
        from_vid = request.args.get('from', type=int)
        to_vid = request.args.get('to', type=int)
        if not from_vid or not to_vid:
            return err("缺少 from 和 to 参数 (version_id)", "VALIDATION_ERROR", 400)
        from app.services.law_version import get_diff
        result = get_diff(law_id, from_vid, to_vid)
        return ok(result.to_dict() if hasattr(result, 'to_dict') else result)
    except Exception as e:
        logger.error(f"get_law_diff error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@compliance_bp.route('/laws/<int:law_id>/versions/<int:version_id>', methods=['GET'])
@_login_required
def get_law_version(law_id: int, version_id: int):
    """Get a specific version with its articles."""
    try:
        from app.services.law_version import get_version
        result = get_version(version_id)
        if result is None:
            return err("版本不存在", "NOT_FOUND", 404)
        return ok(result)
    except Exception as e:
        logger.error(f"get_law_version error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


# ── Compliance Trends & Dashboard (U10/U11) ──

@compliance_bp.route('/trends', methods=['GET'])
@_login_required
def get_trends():
    """Get compliance history trends."""
    try:
        days = request.args.get('days', 90, type=int)
        type_ = request.args.get('type', 'all')
        from app.services.trend_service import get_score_trend, get_violation_distribution, get_feedback_accuracy

        result = {}
        if type_ in ('all', 'score'):
            result['score_trend'] = get_score_trend(days)
        if type_ in ('all', 'violation'):
            result['violation_distribution'] = get_violation_distribution(days)
        if type_ in ('all', 'accuracy'):
            result['feedback_accuracy'] = get_feedback_accuracy(days)

        return ok(result)
    except Exception as e:
        logger.error(f"get_trends error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)


@compliance_bp.route('/dashboard', methods=['GET'])
@_login_required
def get_dashboard():
    """Get compliance dashboard data (overall stats + violations + recent runs)."""
    try:
        days = request.args.get('days', 30, type=int)
        from app.services.dashboard_service import get_dashboard_data
        result = get_dashboard_data(days)
        return ok(result)
    except Exception as e:
        logger.error(f"get_dashboard error: {e}", exc_info=True)
        return err(str(e), "SERVER_ERROR", 500)
