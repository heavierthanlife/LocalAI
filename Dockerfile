# syntax=docker/dockerfile:1
# ── Local_AI Production Docker Image ──
# Build: python scripts/docker_build.py   (auto GPU detection)
#        docker build -t local-ai:latest .
# Run:   docker-compose up -d

FROM python:3.12-slim

LABEL maintainer="Local_AI Team"
LABEL description="中联招标智能助手 — AI-powered bidding agency platform"

# ── System dependencies ──
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    curl gnupg unzip \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/edge stable main" > /etc/apt/sources.list.d/microsoft-edge.list \
    && apt-get update && apt-get install -y --no-install-recommends microsoft-edge-stable \
    # msedgedriver: webdriver-manager handles auto-download at runtime
    && apt-get purge -y curl gnupg unzip \
    && apt-get autoremove -y && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# ── LibreOffice (headless docx→pdf) ──
# Used by document analysis to produce the PDF variant of the report.
# fonts-noto-cjk required for Chinese rendering in the PDF.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer libreoffice-core fonts-noto-cjk \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*

# ── App user ──
RUN useradd -m -s /bin/bash localai && mkdir -p /app && chown localai:localai /app

# ── Python dependencies ──
# FIX-2026-09-07-QA-C1: torch/torchvision 独立一层 + 按宿主 GPU 自动分流。
#   无 GPU 机器装 CPU wheel（~200MB）；有 GPU 机器装 CUDA wheel（由
#   scripts/docker_build.py 检测后传 TORCH_INDEX）。BuildKit cache mount 使
#   pip/apt 下载跨构建复用，避免每次全量重下。
WORKDIR /app
COPY requirements.txt .

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --index-url ${TORCH_INDEX} torch==2.12.1 torchvision==0.27.1 \
 && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt \
 && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gunicorn gevent

# ── Application code ──
COPY --chown=localai:localai . .

# ── Data directories ──
RUN mkdir -p /app/data/user_files /app/data/project_files /app/data/credit_reports \
    /app/data/temp /app/data/dump /app/data/flask_session /app/data/search_cache \
    /app/data/notebooks /app/logs /app/cert \
    && chown -R localai:localai /app/data /app/logs

# ── Runtime ──
USER localai
EXPOSE 8000

# Health check (curl is purged from the image; use python stdlib instead)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/check_auth', timeout=5)" || exit 1

# ── Start ──
# Uses gunicorn with gevent workers for production performance.
# Set WORKERS env var to scale (default: 4).
CMD ["sh", "-c", "gunicorn -w ${WORKERS:-4} -k gevent --bind 0.0.0.0:8000 --timeout 120 --max-requests 1000 --max-requests-jitter 50 --access-logfile - --error-logfile - run:app"]
