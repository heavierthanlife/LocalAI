"""LoRA training launcher service — bridges admin API and the Unsloth executor.

Flow:
    Admin POST /admin/training/run_lora
        → creates task_id in task_bus
        → launches scripts/run_lora_training.py as subprocess
        → subprocess publishes progress to Redis via task_bus
        → frontend watches /tasks/<task_id>/stream (SSE)
        → on completion, adapter auto-registers in adapter_registry.json

Why subprocess (not Celery)?
    Unsloth requires CUDA + specific torch versions that may conflict with
    the Flask app's dependencies. Running as a separate process isolates the
    GPU environment and lets training run on a different machine (via SSH).
"""
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.services.task_bus import TaskBus

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / 'scripts' / 'run_lora_training.py'
EXPORTS_DIR = PROJECT_ROOT / 'data' / 'training' / 'exports'
REGISTRY_PATH = PROJECT_ROOT / 'data' / 'training' / 'adapter_registry.json'

# Default training config
DEFAULTS = {
    'base_model': 'Qwen/Qwen2.5-7B-Instruct',
    'industry': 'bidding_agency',
    'rank': 16,
    'epochs': 3,
    'learning_rate': 2e-4,
    'max_seq_length': 2048,
    'batch_size': 2,
    'gradient_accumulation': 4,
}


def list_available_datasets() -> list[dict]:
    """List JSONL files available for training."""
    if not EXPORTS_DIR.exists():
        return []
    files = []
    for f in sorted(EXPORTS_DIR.glob('*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True):
        files.append({
            'filename': f.name,
            'path': str(f),
            'size_mb': round(f.stat().st_size / 1e6, 2),
            'modified': datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
        })
    return files


def list_registered_adapters() -> dict:
    """Return all registered LoRA adapters."""
    if not REGISTRY_PATH.exists():
        return {}
    try:
        with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def get_adapter_info(industry: str) -> Optional[dict]:
    """Get adapter info for a specific industry."""
    adapters = list_registered_adapters()
    return adapters.get(industry)


def launch_training(
    dataset: str,
    base_model: str = None,
    industry: str = None,
    rank: int = None,
    epochs: int = None,
    learning_rate: float = None,
    max_seq_length: int = None,
    batch_size: int = None,
    gradient_accumulation: int = None,
    python_executable: str = None,
) -> dict:
    """Launch LoRA training as a subprocess.

    Returns:
        {'task_id': str, 'pid': int, 'status': 'launched'}
    Raises:
        FileNotFoundError: if dataset or script not found
        RuntimeError: if launch fails
    """
    import uuid

    # Resolve dataset path
    dataset_path = dataset
    if not os.path.isabs(dataset_path):
        dataset_path = str(EXPORTS_DIR / dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Training script not found: {SCRIPT_PATH}")

    # Generate task ID
    task_id = f"lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Merge with defaults
    cfg = {**DEFAULTS}
    if base_model: cfg['base_model'] = base_model
    if industry: cfg['industry'] = industry
    if rank is not None: cfg['rank'] = rank
    if epochs is not None: cfg['epochs'] = epochs
    if learning_rate is not None: cfg['learning_rate'] = learning_rate
    if max_seq_length is not None: cfg['max_seq_length'] = max_seq_length
    if batch_size is not None: cfg['batch_size'] = batch_size
    if gradient_accumulation is not None: cfg['gradient_accumulation'] = gradient_accumulation

    # Initialize task in task_bus
    bus = TaskBus(task_id=task_id, task_type='lora_training', label=f'LoRA微调: {cfg["industry"]}')
    bus.start()
    bus.progress(0, '准备启动训练进程...')

    # Build command
    py = python_executable or sys.executable
    cmd = [
        py, str(SCRIPT_PATH),
        '--dataset', dataset_path,
        '--base-model', cfg['base_model'],
        '--industry', cfg['industry'],
        '--rank', str(cfg['rank']),
        '--epochs', str(cfg['epochs']),
        '--learning-rate', str(cfg['learning_rate']),
        '--max-seq-length', str(cfg['max_seq_length']),
        '--batch-size', str(cfg['batch_size']),
        '--gradient-accumulation', str(cfg['gradient_accumulation']),
        '--task-id', task_id,
    ]

    logger.info(f"Launching LoRA training: task_id={task_id}, dataset={dataset_path}")

    # Launch as detached subprocess
    try:
        if sys.platform == 'win32':
            # Windows: use CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
            )
        else:
            # Linux/WSL: use start_new_session
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    except Exception as e:
        bus.fail(f'启动训练进程失败: {e}')
        raise RuntimeError(f"Failed to launch training: {e}")

    logger.info(f"Training subprocess started: PID={proc.pid}")

    return {
        'task_id': task_id,
        'pid': proc.pid,
        'status': 'launched',
        'config': cfg,
        'dataset': dataset_path,
        'started_at': datetime.now(timezone.utc).isoformat(),
    }


def deactivate_adapter(industry: str) -> bool:
    """Mark an adapter as inactive (won't be used for routing)."""
    if not REGISTRY_PATH.exists():
        return False
    try:
        with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        if industry not in registry:
            return False
        registry[industry]['active'] = False
        registry[industry]['deactivated_at'] = datetime.now(timezone.utc).isoformat()
        with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to deactivate adapter: {e}")
        return False


def activate_adapter(industry: str) -> bool:
    """Re-activate a previously deactivated adapter."""
    if not REGISTRY_PATH.exists():
        return False
    try:
        with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        if industry not in registry:
            return False
        registry[industry]['active'] = True
        registry[industry]['activated_at'] = datetime.now(timezone.utc).isoformat()
        with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to activate adapter: {e}")
        return False
