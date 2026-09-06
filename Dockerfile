# hr-ai — staging image.
#
# HF_HOME is set explicitly here (§1.2 finding in the staging plan: with no
# override, sentence-transformers/huggingface_hub falls back to
# ~/.cache/huggingface inside the container's writable layer, which is NOT
# persistent across a container recreate). docker-compose.staging.yml mounts
# a NAMED VOLUME at $HF_HOME, so the BGE-M3 weights (~4.3 GB) are fetched
# once on the first `docker compose up` and never re-downloaded again as
# long as that volume persists — including across `resize-for-ingest.sh` /
# `resize-back.sh`, which only change the EC2 instance TYPE, never touch any
# volume. `/health/model` (app/main.py) gates on this same path.
#
# torch is pinned to the CPU-only wheel index explicitly (embedding runs on
# the background admin path only, per app/embeddings.py — CPU is fine, and
# the CPU wheel is a fraction of the size of the default CUDA-enabled one).

FROM python:3.11-slim

ENV HF_HOME=/model-cache \
    PYTHONUNBUFFERED=1

# awscli: used by the shared SSM entrypoint script (bind-mounted at deploy
# time) to resolve SecureString parameters via the instance profile.
RUN apt-get update && apt-get install -y --no-install-recommends \
      awscli curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p "$HF_HOME"

EXPOSE 8001

# Overridden by docker-compose.staging.yml so the SSM entrypoint always runs
# first — this CMD only matters for a bare `docker run` outside compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
