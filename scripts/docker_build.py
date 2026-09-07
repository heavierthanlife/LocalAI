#!/usr/bin/env python3
"""GPU-aware Docker build helper for Local_AI (FIX-2026-09-07-QA-C1).

Detects whether the host has an NVIDIA GPU and picks the matching torch wheel
index, so the same git repo builds correctly on both a CPU-only machine and an
RTX GPU machine:

  - no GPU  -> https://download.pytorch.org/whl/cpu      (torch 2.12.1+cpu, ~200MB)
  - GPU     -> https://download.pytorch.org/whl/cu124    (torch CUDA build; 2080 Super / Turing sm_75)

Override the CUDA index with the TORCH_CUDA_INDEX env var if a different
cuXXXX build is needed (e.g. verified on the target GPU machine).

Usage (Windows PowerShell or any shell):
    python scripts/docker_build.py            # build app/celery-worker/celery-beat
    python scripts/docker_build.py --up       # also docker compose up -d afterwards

GPU machine after building:
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
"""
import os
import shutil
import subprocess
import sys

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
GPU_INDEX_DEFAULT = "https://download.pytorch.org/whl/cu124"
SERVICES = ["app", "celery-worker", "celery-beat"]


def _has_gpu() -> bool:
    """Detect an NVIDIA GPU by probing nvidia-smi (works on Windows + Linux)."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        r = subprocess.run([nvidia_smi, "-L"], capture_output=True, timeout=15)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def main() -> int:
    gpu = _has_gpu()
    torch_index = os.environ.get("TORCH_CUDA_INDEX", GPU_INDEX_DEFAULT) if gpu else CPU_INDEX
    print(f"[docker_build] GPU detected: {gpu}")
    print(f"[docker_build] torch index : {torch_index}")

    cmd = ["docker", "compose", "build", "--build-arg", f"TORCH_INDEX={torch_index}"] + SERVICES
    print(f"[docker_build] $ {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        print("[docker_build] build FAILED", file=sys.stderr)
        return rc

    print("\n[docker_build] build OK.")
    print("[docker_build] deploy: docker compose up -d app celery-worker celery-beat")
    if gpu:
        print("[docker_build]   (GPU machine) docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d app celery-worker celery-beat")

    if "--up" in sys.argv:
        up = ["docker", "compose", "up", "-d", "app", "celery-worker", "celery-beat"]
        print(f"[docker_build] $ {' '.join(up)}")
        return subprocess.call(up)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
