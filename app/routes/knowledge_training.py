"""Admin training-pipeline routes for the knowledge blueprint family.

Registered on the shared ``knowledge_bp`` Blueprint object from
app/routes/knowledge.py. Covers /admin/training* stats/export/cleanup/health,
training exports, and /admin/training/lora/* (LoRA datasets/adapters/run/activate).
"""
import os

from flask import request, jsonify, session, send_file

from app.config import logger
from app.routes.knowledge import knowledge_bp


# ── Admin: Training Data Pipeline ──
@knowledge_bp.route('/admin/training_stats', methods=['GET'])
def admin_training_stats():
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import get_training_stats
        return jsonify({"stats": get_training_stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_export', methods=['POST'])
def admin_training_export():
    """Export training data. mode: 'full' | 'incremental' | 'quality' (legacy full+quality).

    - 'full': export all data, update watermark
    - 'incremental': export only new data since last export (fast, production use)
    - 'quality': legacy alias for full export with quality filter
    """
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'incremental')  # default to incremental for safety

    try:
        from app.services.training_logger import (
            export_training_jsonl, export_training_jsonl_incremental,
            reset_export_watermark
        )

        if mode == 'reset_watermark':
            reset_export_watermark()
            return jsonify({"status": "ok", "message": "Export watermark reset — next export will be full"})

        if mode == 'incremental':
            # Uses quality filter from runtime_config (default ≥3 stars)
            path = export_training_jsonl_incremental()
            if path:
                basename = os.path.basename(path)
                return jsonify({
                    "status": "ok", "path": path, "mode": "incremental",
                    "message": f"Incremental export: {basename}"
                })
            return jsonify({
                "status": "ok", "message": "No new data to export (up to date with last export)",
                "mode": "incremental", "path": ""
            })

        # Full export
        min_rating = 0 if mode == 'all' else None  # None = auto from config (default 3)
        path = export_training_jsonl('manual', min_rating=min_rating)
        if path:
            mode_label = 'all data' if mode == 'all' else 'quality ≥3★'
            return jsonify({
                "status": "ok", "path": path, "mode": "full",
                "message": f"Full export ({mode_label}): {os.path.basename(path)}"
            })
        return jsonify({"error": "No training data to export"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_export_history', methods=['GET'])
def admin_training_export_history():
    """Return export history + watermark status for admin panel."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import get_export_history
        return jsonify({"status": "ok", "history": get_export_history()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_cleanup_stats', methods=['GET'])
def admin_training_cleanup_stats():
    """Return age distribution of training data for cleanup preview."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import get_training_cleanup_stats
        return jsonify({"status": "ok", "stats": get_training_cleanup_stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_cleanup', methods=['POST'])
def admin_training_cleanup():
    """Manually trigger training data cleanup (admin only)."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json(silent=True) or {}
    dry_run = data.get('dry_run', False)
    try:
        if dry_run:
            from app.services.training_logger import get_training_cleanup_stats
            return jsonify({"status": "ok", "dry_run": True, "stats": get_training_cleanup_stats()})
        from app.services.training_logger import cleanup_training_sessions
        retention = data.get('retention_days')
        if retention is not None:
            removed = cleanup_training_sessions(retention_days=int(retention))
        else:
            from app.services.runtime_config import get as rc_get
            removed = cleanup_training_sessions(retention_days=rc_get('training_retention_days', 90))
        return jsonify({"status": "ok", "removed": removed, "message": f"Purged {removed} old training sessions"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Training Data Health Check ──

@knowledge_bp.route('/admin/training_health', methods=['GET'])
def admin_training_health():
    """Run training data health check (scan + report, no repair)."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import run_training_health_check
        report = run_training_health_check(repair=False)
        return jsonify({"status": "ok", "report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_health', methods=['POST'])
def admin_training_health_repair():
    """Run health check with auto-repair enabled."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import run_training_health_check
        report = run_training_health_check(repair=True)
        return jsonify({"status": "ok", "report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_health_history', methods=['GET'])
def admin_training_health_history():
    """Return health check history and trend data."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import get_health_history
        return jsonify({"status": "ok", "history": get_health_history()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Export file management ──

@knowledge_bp.route('/admin/training_exports_cleanup', methods=['POST'])
def admin_training_exports_cleanup():
    """Clean up old export files (keep last N, configurable)."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json(silent=True) or {}
    try:
        from app.services.training_logger import cleanup_old_exports
        from app.services.admin_utils import log_admin_action
        keep = data.get('keep_count')
        result = cleanup_old_exports(keep_count=keep if keep else None)
        log_admin_action(session.get('user_id', ''), session.get('username', ''),
                        'EXPORT_CLEANUP', 'training', None,
                        column_name='export_files',
                        new_value=f"deleted:{len(result['deleted'])} kept:{result['kept']}")
        return jsonify({"status": "ok", "deleted": len(result['deleted']), "kept": result['kept'],
                        "deleted_files": result['deleted'],
                        "message": f"Deleted {len(result['deleted'])} old export files, kept {result['kept']}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_exports_list', methods=['GET'])
def admin_training_exports_list():
    """Return detailed list of export files with size and mtime."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import get_export_files_detail
        from app.services.runtime_config import get as rc_get
        return jsonify({
            "status": "ok",
            "files": get_export_files_detail(),
            "retention_count": rc_get('export_retention_count', 20),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_exports_delete/<filename>', methods=['POST'])
def admin_training_exports_delete(filename):
    """Delete a specific export file (admin only, logged)."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    try:
        from app.services.training_logger import delete_export_file
        from app.services.admin_utils import log_admin_action
        ok = delete_export_file(filename)
        if ok:
            log_admin_action(session.get('user_id', ''), session.get('username', ''),
                            'EXPORT_DELETE', 'training', None,
                            column_name='export_file', old_value=filename, new_value='deleted')
            return jsonify({"status": "ok", "message": f"Deleted {filename}"})
        return jsonify({"error": "File not found or invalid"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knowledge_bp.route('/admin/training_exports_download/<filename>', methods=['GET'])
def admin_training_exports_download(filename):
    """Download a specific export file (admin only)."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    # Path traversal guard
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({"error": "Invalid filename"}), 400
    if not filename.endswith('.jsonl'):
        return jsonify({"error": "Only .jsonl files allowed"}), 400
    from app.services.training_logger import EXPORT_DIR
    filepath = os.path.join(EXPORT_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404
    from app.services.admin_utils import log_admin_action
    log_admin_action(session.get('user_id', ''), session.get('username', ''),
                    'EXPORT_DOWNLOAD', 'training', None,
                    column_name='export_file', new_value=filename)
    from flask import send_file
    return send_file(filepath, as_attachment=True, download_name=filename,
                     mimetype='application/x-ndjson')


# ── LoRA Fine-tuning Management ──

@knowledge_bp.route('/admin/training/lora/datasets', methods=['GET'])
def admin_lora_datasets():
    """List available datasets for LoRA training."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.lora_trainer import list_available_datasets
    return jsonify({"datasets": list_available_datasets()})


@knowledge_bp.route('/admin/training/lora/adapters', methods=['GET'])
def admin_lora_adapters():
    """List all registered LoRA adapters."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.lora_trainer import list_registered_adapters
    return jsonify({"adapters": list_registered_adapters()})


@knowledge_bp.route('/admin/training/run_lora', methods=['POST'])
def admin_run_lora():
    """Launch LoRA fine-tuning as a background subprocess.

    Body params:
        dataset: filename in exports/ or absolute path (required)
        base_model: HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)
        industry: industry label (default: bidding_agency)
        rank: LoRA rank (default: 16)
        epochs: training epochs (default: 3)
        learning_rate: peak LR (default: 2e-4)

    Returns:
        {task_id, pid, status, config}
    """
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json() or {}
    dataset = data.get('dataset', '').strip()
    if not dataset:
        return jsonify({"error": "dataset is required"}), 400
    from app.services.lora_trainer import launch_training
    try:
        result = launch_training(
            dataset=dataset,
            base_model=data.get('base_model'),
            industry=data.get('industry'),
            rank=data.get('rank'),
            epochs=data.get('epochs'),
            learning_rate=data.get('learning_rate'),
            max_seq_length=data.get('max_seq_length'),
            batch_size=data.get('batch_size'),
            gradient_accumulation=data.get('gradient_accumulation'),
        )
        from app.services.admin_utils import log_admin_action
        log_admin_action(session.get('user_id', ''), session.get('username', ''),
                        'LORA_TRAINING_LAUNCH', 'training', None,
                        column_name='task_id', new_value=result['task_id'])
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Launch failed: {str(e)[:200]}"}), 500


@knowledge_bp.route('/admin/training/lora/<industry>/activate', methods=['POST'])
def admin_lora_activate(industry):
    """Activate a LoRA adapter for an industry."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.lora_trainer import activate_adapter
    if activate_adapter(industry):
        return jsonify({"success": True, "industry": industry, "active": True})
    return jsonify({"error": "Adapter not found"}), 404


@knowledge_bp.route('/admin/training/lora/<industry>/deactivate', methods=['POST'])
def admin_lora_deactivate(industry):
    """Deactivate a LoRA adapter for an industry."""
    from app.routes.admin import is_admin as check_admin
    if not check_admin():
        return jsonify({"error": "Admin only"}), 403
    from app.services.lora_trainer import deactivate_adapter
    if deactivate_adapter(industry):
        return jsonify({"success": True, "industry": industry, "active": False})
    return jsonify({"error": "Adapter not found"}), 404
