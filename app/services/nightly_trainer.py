"""Nightly LoRA training automation.

Runs during off-work hours (default 2:00 AM) via Celery beat.
Checks if enough compliance feedback has accumulated, exports training data,
and launches the Unsloth LoRA pipeline.

Usage (manual):
    from app.services.nightly_trainer import run_nightly_training
    run_nightly_training()

Usage (Celery beat):
    Add to celery_app.py beat_schedule:
        'nightly-lora-training': {
            'task': 'app.services.nightly_trainer.run_nightly_training',
            'schedule': crontab(hour=2, minute=0),
        },
"""
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Minimum feedback samples needed before training is worthwhile
MIN_SAMPLES_DEFAULT = 10

# Training config
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_RANK = 16
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_INDUSTRY = "bidding_agency"


def _get_training_export_path(date_str: str = None) -> str:
    """Get the path for the nightly training export."""
    from app.config import DATA_DIR
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    export_dir = os.path.join(str(DATA_DIR), 'training', 'exports')
    os.makedirs(export_dir, exist_ok=True)
    return os.path.join(export_dir, f"nightly_{date_str}.jsonl")


def _get_adapter_output_dir(date_str: str = None) -> str:
    """Get the output directory for the trained adapter."""
    from app.config import DATA_DIR
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    adapter_dir = os.path.join(str(DATA_DIR), 'training', 'adapters', f"nightly_{date_str}")
    os.makedirs(adapter_dir, exist_ok=True)
    return adapter_dir


def _check_feedback_count() -> tuple[int, bool]:
    """Check how many compliance feedback samples exist.

    Returns (count, is_ready).
    """
    try:
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM compliance_feedback")
                count = cur.fetchone()[0]
        ready = count >= MIN_SAMPLES_DEFAULT
        logger.info(f"Nightly training check: {count} feedback samples (min={MIN_SAMPLES_DEFAULT}, ready={ready})")
        return count, ready
    except Exception as e:
        logger.error(f"Failed to check feedback count: {e}")
        return 0, False


def _export_training_data(output_path: str) -> tuple[int, str | None]:
    """Export compliance feedback as JSONL training data.

    Returns (sample_count, error_or_None).
    """
    try:
        from app.database import get_db_connection

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, user_id, task_id, bid_doc_name, check_file_name,
                           rule_count, summary_json, ai_verdict,
                           user_verdict, user_explain, results_json, report_html,
                           created_at
                    FROM compliance_feedback
                    ORDER BY created_at DESC
                    LIMIT 500
                """)
                rows = cur.fetchall()

        if not rows:
            logger.warning("Nightly training: no feedback data to export")
            return 0, "no data"

        from app.services.compliance_prompts import COMPLIANCE_CHECK_SYSTEM

        with open(output_path, 'w', encoding='utf-8') as f:
            for row in rows:
                summary = row[6] if isinstance(row[6], dict) else json.loads(row[6] or '{}')
                results = row[10] if isinstance(row[10], list) else json.loads(row[10] or '[]')

                violations = []
                for res in results:
                    if res.get('verdict') in ('VIOLATION', 'CRITICAL'):
                        violations.append({
                            'rule_id': res.get('rule_id'),
                            'verdict': res.get('verdict'),
                            'reasoning': res.get('reasoning', ''),
                        })

                sample = {
                    "instruction": COMPLIANCE_CHECK_SYSTEM[:800],
                    "input": f"Bid doc: {row[3]}\nCheck file: {row[4]}\n"
                             f"Rules: {row[5]} total\nAI verdict: {row[7]}",
                    "output": f"User verdict: {row[8]}\nExplanation: {row[9] or 'N/A'}",
                    "metadata": {
                        "feedback_id": row[0],
                        "ai_verdict": row[7],
                        "user_verdict": row[8],
                        "created_at": row[12].isoformat() if row[12] else None,
                    },
                }
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        logger.info(f"Nightly training: exported {len(rows)} samples to {output_path}")
        return len(rows), None
    except Exception as e:
        logger.error(f"Nightly training export failed: {e}", exc_info=True)
        return 0, str(e)


def _run_lora_training(jsonl_path: str, adapter_dir: str, industry: str = "bidding_agency") -> dict:
    """Run Unsloth LoRA training as subprocess.

    Returns dict with success/error info.
    """
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts", "run_lora_training.py"
    )

    if not os.path.exists(script_path):
        logger.warning(f"LoRA training script not found: {script_path}")
        return {"success": False, "error": "Training script not found"}

    cmd = [
        sys.executable, script_path,
        "--base-model", DEFAULT_BASE_MODEL,
        "--dataset", jsonl_path,
        "--output-dir", adapter_dir,
        "--industry", industry,
        "--rank", str(DEFAULT_RANK),
        "--epochs", str(DEFAULT_EPOCHS),
        "--learning-rate", str(DEFAULT_LEARNING_RATE),
    ]

    logger.info(f"Nightly training: launching {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour max
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            logger.info(f"Nightly training completed in {elapsed:.0f}s: {adapter_dir}")
            return {
                "success": True,
                "adapter_dir": adapter_dir,
                "elapsed_seconds": int(elapsed),
                "base_model": DEFAULT_BASE_MODEL,
            }
        else:
            logger.error(f"Nightly training failed (rc={result.returncode}): {result.stderr[:500]}")
            return {
                "success": False,
                "error": result.stderr[:500],
                "elapsed_seconds": int(elapsed),
            }
    except subprocess.TimeoutExpired:
        logger.error("Nightly training timed out (2h)")
        return {"success": False, "error": "timed out after 2 hours"}
    except Exception as e:
        logger.error(f"Nightly training error: {e}")
        return {"success": False, "error": str(e)[:300]}


def run_nightly_training(min_samples: int = None, force: bool = False):
    """Main entry point — call from Celery beat or manually.

    Checks feedback count, exports data, runs training if enough samples.

    Args:
        min_samples: override minimum sample threshold
        force: skip the sample count check

    Returns:
        dict with status, sample_count, training_result
    """
    threshold = min_samples or MIN_SAMPLES_DEFAULT
    now = datetime.now(timezone.utc)
    date_str = now.strftime('%Y-%m-%d')

    logger.info(f"=== Nightly LoRA training started: {now.isoformat()} ===")

    # Step 1: Check feedback count
    count, ready = _check_feedback_count()
    if not ready and not force:
        logger.info(f"Nightly training skipped: {count} < {threshold} samples required")
        return {
            "status": "skipped",
            "reason": f"insufficient samples ({count}/{threshold})",
            "sample_count": count,
            "checked_at": now.isoformat(),
        }

    # Step 2: Export training data
    export_path = _get_training_export_path(date_str)
    exported, export_error = _export_training_data(export_path)
    if export_error:
        return {
            "status": "failed",
            "reason": f"export failed: {export_error}",
            "sample_count": count,
        }

    # Step 3: Run LoRA training
    adapter_dir = _get_adapter_output_dir(date_str)
    training_result = _run_lora_training(export_path, adapter_dir)

    # Step 4: Update adapter registry if successful
    if training_result.get("success"):
        _update_adapter_registry(adapter_dir, training_result)

    result = {
        "status": "completed" if training_result.get("success") else "failed",
        "sample_count": count,
        "exported_count": exported,
        "export_path": export_path,
        "adapter_dir": adapter_dir,
        "training": training_result,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── Notify all admins ──
    _notify_admins_of_training(result)

    logger.info(f"=== Nightly training result: {result['status']} ({exported} samples) ===")
    return result


def _notify_admins_of_training(result: dict):
    """Log detailed training result and store notifications for all admins.

    Writes to:
      - logs/app.log (via logger, already configured)
      - data/ingest/training_notifications.json (for frontend polling)
    """
    import json as _json
    from app.config import DATA_DIR

    # ── Detailed log ──
    status = result.get("status", "unknown")
    samples = result.get("sample_count", 0)
    exported = result.get("exported_count", 0)
    ts = result.get("finished_at", "")

    logger.info("=" * 60)
    logger.info(f"NIGHTLY LORA TRAINING COMPLETE: {status}")
    logger.info(f"  Samples available: {samples}")
    logger.info(f"  Samples exported:  {exported}")
    logger.info(f"  Export path:       {result.get('export_path', 'N/A')}")
    logger.info(f"  Adapter dir:       {result.get('adapter_dir', 'N/A')}")
    training = result.get("training", {})
    if training.get("success"):
        logger.info(f"  Base model:        {training.get('base_model', 'N/A')}")
        logger.info(f"  Duration:          {training.get('elapsed_seconds', 0)}s")
        logger.info(f"  Adapter dir:       {training.get('adapter_dir', 'N/A')}")
    else:
        logger.error(f"  Training error:    {training.get('error', 'unknown')}")
    logger.info("=" * 60)

    # ── Query all active admins ──
    admin_list = []
    try:
        from app.database import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id, username FROM users WHERE role = 'admin' AND is_active = TRUE"
                )
                admin_list = [{"user_id": r[0], "username": r[1]} for r in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to query admins for notification: {e}")
        admin_list = [{"user_id": "system", "username": "system"}]

    logger.info(f"Training notification sent to {len(admin_list)} admin(s): "
                f"{', '.join(a['username'] for a in admin_list)}")

    # ── Persist notification for frontend polling ──
    notification = {
        "type": "nightly_training",
        "status": status,
        "sample_count": samples,
        "exported_count": exported,
        "export_path": result.get("export_path", ""),
        "adapter_dir": result.get("adapter_dir", ""),
        "base_model": training.get("base_model", ""),
        "elapsed_seconds": training.get("elapsed_seconds", 0),
        "error": training.get("error", ""),
        "notified_admins": [a["username"] for a in admin_list],
        "created_at": ts or datetime.now(timezone.utc).isoformat(),
        "seen_by": [],
    }

    notify_dir = os.path.join(str(DATA_DIR), 'ingest')
    os.makedirs(notify_dir, exist_ok=True)
    notify_path = os.path.join(notify_dir, 'training_notifications.json')

    existing = []
    if os.path.exists(notify_path):
        try:
            with open(notify_path, 'r', encoding='utf-8') as f:
                existing = _json.load(f)
        except Exception:
            existing = []

    existing.insert(0, notification)
    # Keep last 50
    existing = existing[:50]

    with open(notify_path, 'w', encoding='utf-8') as f:
        _json.dump(existing, f, ensure_ascii=False, default=str)

    logger.info(f"Training notification persisted: {notify_path}")


def _update_adapter_registry(adapter_dir: str, training_result: dict):
    """Update adapter_registry.json with the newly trained adapter."""
    from app.config import DATA_DIR
    import json as _json

    registry_path = os.path.join(str(DATA_DIR), 'training', 'adapter_registry.json')
    registry = {}

    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = _json.load(f)
        except Exception:
            registry = {}

    adapter_key = os.path.basename(adapter_dir)
    now_ts = datetime.now(timezone.utc).isoformat()

    registry["compliance_checker"] = {
        "adapter_dir": adapter_dir,
        "base_model": training_result.get("base_model", DEFAULT_BASE_MODEL),
        "trained_at": now_ts,
        "industry": "bidding_agency",
        "elapsed_seconds": training_result.get("elapsed_seconds"),
        "source": "nightly_training",
    }

    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, 'w', encoding='utf-8') as f:
        _json.dump(registry, f, ensure_ascii=False, indent=2)

    logger.info(f"Adapter registry updated: {adapter_key}")
