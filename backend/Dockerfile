# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Builder: compile wheels for the heavy extras (MinerU / torch) so the runtime
# image stays small and reproducible.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

# Layered so dependency resolution is cached independently of source changes.
COPY pyproject.toml README.md ./
RUN uv pip install --system --no-cache "hatchling"

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./

# Change to install/uninstall heavy extras without touching this file:
#   docker build --build-arg EXTRAS='[postgres,openai,html]' .
ARG EXTRAS="[postgres,openai,html]"
RUN uv pip install --system --no-cache ".[${EXTRAS}]"

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    STORAGE_ROOT=/data/blobs \
    HF_HOME=/data/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libgl1 libglib2.0-0 curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 paper

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# MinerU downloads model weights on first use; keep them off the app layer.
RUN mkdir -p /data/blobs /data/models && chown -R paper:paper /data /app
USER paper

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]