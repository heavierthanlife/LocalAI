#!/usr/bin/env python3
"""LoRA fine-tuning executor using Unsloth — the missing piece of the training pipeline.

Takes JSONL exported by training_logger.py and produces a LoRA adapter for an
open-source base model (e.g., Qwen2.5-7B-Instruct). Progress is published to
the task_bus so the admin frontend can show real-time progress via SSE.

═══════════════════════════════════════════════════════════════════
PIPELINE FLOW:
    training_logger.py  →  export_training_jsonl()  →  THIS SCRIPT  →  adapter
                                                                    ↓
                                              llm_provider.py INDUSTRY_MODEL_MAP
                                                                    ↓
                                                      Ollama / vLLM serving
═══════════════════════════════════════════════════════════════════

PREREQUISITES:
    • Linux or WSL2 with NVIDIA GPU (min 16GB VRAM for 7B, 24GB for 14B)
    • pip install "unsloth[cu121]" torch peft trl datasets
    • Base model auto-downloads on first run (~15GB for Qwen2.5-7B)

USAGE (standalone):
    python scripts/run_lora_training.py \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --dataset data/training/exports/manual_2026-07-01_q3.jsonl \\
        --output-dir data/training/adapters/bidding_v1 \\
        --industry bidding_agency \\
        --rank 16 --epochs 3 --learning-rate 2e-4

USAGE (via admin API):
    POST /admin/training/run_lora  (launches this script as subprocess)

USAGE (via Celery):
    from app.services.lora_trainer import run_lora_training_task
    run_lora_training_task.delay(base_model, dataset, industry, ...)
"""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Insert project root for imports ──
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _publish_progress(task_id: str, percent: int, message: str, event: str = 'progress'):
    """Publish progress to Redis task_bus (best-effort — don't crash if Redis is down)."""
    try:
        from app.services.task_bus import TaskBus
        bus = TaskBus(task_id=task_id, task_type='lora_training', label='LoRA微调训练')
        if event == 'start':
            bus.start()
        elif event == 'complete':
            bus.complete(result={'message': message})
        elif event == 'error':
            bus.fail(message)
        else:
            bus.progress(percent, message)
    except Exception:
        pass  # Training must not fail because Redis is down


def _log(task_id: str, msg: str):
    """Print + publish a log message."""
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)
    if task_id:
        _publish_progress(task_id, -1, msg)


# ── Dataset loading & formatting ──

def load_dataset(jsonl_path: str, max_samples: int = 5000):
    """Load JSONL export and format to Unsloth-compatible chat messages.

    Expected JSONL format (from training_logger.py _build_entry):
        {
            "instruction": "...",
            "input": "user question",
            "output": "assistant response",
            "thinking": "reasoning chain (optional)",
            "context": "RAG/knowledge context (optional)",
            "rating": 4,
            ...
        }
    """
    from datasets import Dataset

    samples = []
    skipped = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            user_msg = entry.get('input', '').strip()
            assistant_msg = entry.get('output', '').strip()
            thinking = entry.get('thinking', '').strip()
            context = entry.get('context', '').strip()

            if not user_msg or not assistant_msg:
                skipped += 1
                continue

            # Build chat-formatted text (Qwen/ChatML format)
            # Include thinking if present (helps model learn reasoning)
            system_parts = ["你是一个专业的招投标AI助手。"]
            if context:
                system_parts.append(f"\n\n参考信息:\n{context[:2000]}")
            system_msg = ''.join(system_parts)

            if thinking:
                assistant_full = f"<think>{thinking}</think>\n\n{assistant_msg}"
            else:
                assistant_full = assistant_msg

            # ChatML format (compatible with Qwen, Llama, Mistral)
            text = (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_full}<|im_end|>"
            )
            samples.append({'text': text})

            if len(samples) >= max_samples:
                break

    print(f"Loaded {len(samples)} samples (skipped {skipped} invalid)")
    return Dataset.from_list(samples)


# ── Training callback for progress ──

class ProgressCallback:
    """Transformers callback that publishes training progress to task_bus."""

    def __init__(self, task_id: str, total_steps: int):
        self.task_id = task_id
        self.total_steps = max(total_steps, 1)
        self.last_report = 0

    def __call__(self, args, state, control, **kwargs):
        """Called after each training step."""
        try:
            current_step = state.global_step
            # Report every 5% or every 10 steps
            interval = max(self.total_steps // 20, 10)
            if current_step - self.last_report >= interval or current_step == self.total_steps:
                self.last_report = current_step
                pct = int(current_step / self.total_steps * 100)
                loss = state.log_history[-1].get('loss', 0) if state.log_history else 0
                msg = f"训练中: {current_step}/{self.total_steps} steps, loss={loss:.4f}"
                _publish_progress(self.task_id, pct, msg)
        except Exception:
            pass


# ── Main training function ──

def run_training(
    base_model: str = 'Qwen/Qwen2.5-7B-Instruct',
    dataset_path: str = None,
    output_dir: str = None,
    industry: str = 'bidding_agency',
    rank: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    batch_size: int = 2,
    gradient_accumulation: int = 4,
    task_id: str = None,
):
    """Run LoRA fine-tuning with Unsloth.

    Args:
        base_model: HuggingFace model ID (e.g., 'Qwen/Qwen2.5-7B-Instruct')
        dataset_path: Path to JSONL file from training_logger.py export
        output_dir: Where to save the LoRA adapter
        industry: Industry label for INDUSTRY_MODEL_MAP registration
        rank: LoRA rank (8/16/32 — higher = more capacity, more VRAM)
        epochs: Number of training epochs
        learning_rate: Peak learning rate
        max_seq_length: Max sequence length (longer = more VRAM)
        batch_size: Per-device batch size
        gradient_accumulation: Steps to accumulate before optimizer update
        task_id: If set, publish progress to Redis task_bus

    Returns:
        Dict with training results: adapter_path, steps, duration, samples
    """
    start_time = time.time()

    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if not output_dir:
        output_dir = str(PROJECT_ROOT / 'data' / 'training' / 'adapters' / f'{industry}_{int(time.time())}')
    os.makedirs(output_dir, exist_ok=True)

    _log(task_id, f"=" * 60)
    _log(task_id, f"LoRA Fine-tuning Started")
    _log(task_id, f"  Base model:    {base_model}")
    _log(task_id, f"  Dataset:       {dataset_path}")
    _log(task_id, f"  Output:        {output_dir}")
    _log(task_id, f"  Industry:      {industry}")
    _log(task_id, f"  LoRA rank:     {rank}")
    _log(task_id, f"  Epochs:        {epochs}")
    _log(task_id, f"  Learning rate: {learning_rate}")
    _log(task_id, f"  Seq length:    {max_seq_length}")
    _log(task_id, f"  Batch size:    {batch_size} × {gradient_accumulation} accumulation")
    _log(task_id, f"=" * 60)

    if task_id:
        _publish_progress(task_id, 0, '开始训练...', 'start')

    # ── Step 1: Load base model with Unsloth ──
    _log(task_id, "[1/5] Loading base model with Unsloth (4-bit quantization)...")
    _publish_progress(task_id, 5, '正在加载基础模型...')

    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError as e:
        msg = f"Unsloth not installed: {e}\nInstall with: pip install 'unsloth[cu121]'"
        _log(task_id, f"ERROR: {msg}")
        if task_id:
            _publish_progress(task_id, 0, msg, 'error')
        raise ImportError(msg)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # auto: fp16/bf16 based on GPU
        load_in_4bit=True,  # 4-bit quantization — saves ~70% VRAM
    )
    _log(task_id, f"  Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    _publish_progress(task_id, 15, '基础模型加载完成')

    # ── Step 2: Apply LoRA adapters ──
    _log(task_id, "[2/5] Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=rank,  # convention: alpha = rank
        lora_dropout=0,   # dropout=0 for full-capacity fine-tuning
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    _log(task_id, "  LoRA adapters applied.")
    _publish_progress(task_id, 20, 'LoRA适配器已应用')

    # ── Step 3: Load and format dataset ──
    _log(task_id, f"[3/5] Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    sample_count = len(dataset)
    _log(task_id, f"  {sample_count} training samples loaded.")
    _publish_progress(task_id, 25, f'已加载 {sample_count} 条训练数据')

    if sample_count < 10:
        msg = f"Not enough training data: {sample_count} samples. Need at least 10."
        _log(task_id, f"ERROR: {msg}")
        if task_id:
            _publish_progress(task_id, 0, msg, 'error')
        raise ValueError(msg)

    # ── Step 4: Train ──
    _log(task_id, "[4/5] Starting training loop...")
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Calculate total steps for progress tracking
    steps_per_epoch = (sample_count + batch_size * gradient_accumulation - 1) // (batch_size * gradient_accumulation)
    total_steps = steps_per_epoch * epochs

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=max(5, int(total_steps * 0.05)),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=max(1, total_steps // 50),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=output_dir,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",  # no wandb/tensorboard
        max_steps=total_steps,
    )

    progress_cb = ProgressCallback(task_id, total_steps) if task_id else None

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
        callbacks=[progress_cb] if progress_cb else None,
    )

    _publish_progress(task_id, 30, f'开始训练: {total_steps} steps, {epochs} epochs')

    trainer_stats = trainer.train()
    final_loss = trainer_stats.training_loss
    _log(task_id, f"  Training complete. Final loss: {final_loss:.4f}")
    _publish_progress(task_id, 85, f'训练完成, 最终loss={final_loss:.4f}')

    # ── Step 5: Save adapter ──
    _log(task_id, f"[5/5] Saving LoRA adapter to {output_dir}...")
    _publish_progress(task_id, 90, '正在保存适配器权重...')

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    metadata = {
        'base_model': base_model,
        'industry': industry,
        'rank': rank,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'max_seq_length': max_seq_length,
        'sample_count': sample_count,
        'total_steps': total_steps,
        'final_loss': final_loss,
        'training_duration_seconds': round(time.time() - start_time, 1),
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'adapter_path': output_dir,
        'dataset': dataset_path,
    }
    meta_path = os.path.join(output_dir, 'training_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    _log(task_id, f"  Adapter saved to: {output_dir}")
    _log(task_id, f"  Metadata saved to: {meta_path}")

    # ── Register in INDUSTRY_MODEL_MAP ──
    _publish_progress(task_id, 95, '正在注册到行业模型映射...')
    _register_adapter(industry, output_dir, base_model)

    duration = time.time() - start_time
    result_msg = (
        f"训练完成! 耗时 {duration/60:.1f} 分钟, "
        f"{sample_count} 样本, {total_steps} 步, "
        f"最终loss={final_loss:.4f}"
    )
    _log(task_id, f"\n{'=' * 60}")
    _log(task_id, result_msg)
    _log(task_id, f"{'=' * 60}")

    if task_id:
        _publish_progress(task_id, 100, result_msg, 'complete')

    return metadata


def _register_adapter(industry: str, adapter_path: str, base_model: str):
    """Register the trained adapter in the adapter registry (read by llm_provider.py)."""
    registry_path = PROJECT_ROOT / 'data' / 'training' / 'adapter_registry.json'

    registry = {}
    if registry_path.exists():
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
        except Exception:
            pass

    registry[industry] = {
        'adapter_path': adapter_path,
        'base_model': base_model,
        'registered_at': datetime.now(timezone.utc).isoformat(),
        'active': True,
    }

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    print(f"  Registered adapter for industry '{industry}' → {adapter_path}")


# ── CLI entry point ──

def main():
    parser = argparse.ArgumentParser(description='LoRA fine-tuning with Unsloth')
    parser.add_argument('--base-model', default='Qwen/Qwen2.5-7B-Instruct',
                        help='HuggingFace base model ID')
    parser.add_argument('--dataset', required=True,
                        help='Path to JSONL training data (from training_logger export)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for LoRA adapter (auto-generated if not set)')
    parser.add_argument('--industry', default='bidding_agency',
                        help='Industry label for model routing')
    parser.add_argument('--rank', type=int, default=16,
                        help='LoRA rank (8/16/32)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                        help='Peak learning rate')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                        help='Max sequence length')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Per-device batch size')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--task-id', default=None,
                        help='Task ID for Redis progress publishing')
    parser.add_argument('--list-datasets', action='store_true',
                        help='List available exported datasets and exit')

    args = parser.parse_args()

    if args.list_datasets:
        exports_dir = PROJECT_ROOT / 'data' / 'training' / 'exports'
        if exports_dir.exists():
            files = sorted(exports_dir.glob('*.jsonl'), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                print("Available datasets:")
                for f in files:
                    size_mb = f.stat().st_size / 1e6
                    print(f"  {f.name}  ({size_mb:.1f} MB)")
            else:
                print("No exported datasets found. Run export_training_jsonl() first.")
        else:
            print("Exports directory does not exist yet.")
        return

    try:
        metadata = run_training(
            base_model=args.base_model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            industry=args.industry,
            rank=args.rank,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            task_id=args.task_id,
        )
        print(f"\n✓ Training successful. Adapter: {metadata['adapter_path']}")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        traceback.print_exc()
        if args.task_id:
            _publish_progress(args.task_id, 0, str(e)[:500], 'error')
        sys.exit(1)


if __name__ == '__main__':
    main()
