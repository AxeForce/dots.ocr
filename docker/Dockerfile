# docker/Dockerfile
# vLLM 0.9.1 + auto-download dots.ocr weights at container start

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps (pin vLLM 0.9.1)
RUN pip3 install --no-cache-dir "vllm==0.9.1" "huggingface_hub>=0.25.0"

# Runtime env
ENV MODEL_ID="rednote-hilab/dots.ocr" \
    DOWNLOAD_DIR="/models" \
    MODEL_DIR="/models/DotsOCR" \
    HF_HOME="/root/.cache/huggingface" \
    PYTHONUNBUFFERED=1 \
    VLLM_LOGGING_LEVEL=INFO

# Create dirs
RUN mkdir -p /app /models

# Lightweight entry script:
# - if weights missing, download to /models/DotsOCR
# - ensure PYTHONPATH includes the parent of MODEL_DIR (see dots.ocr docs)
# - start vLLM OpenAI-compatible server
ADD <<'BASH' /app/run.sh
#!/usr/bin/env bash
set -euo pipefail

# Optional: use HF_TOKEN if provided (for gated models)
: "${HF_TOKEN:=}"

# Download weights once (to a directory WITHOUT dots, per dots.ocr docs)
if [ ! -d "${MODEL_DIR}" ] || [ -z "$(ls -A "${MODEL_DIR}" 2>/dev/null || true)" ]; then
  echo "[run.sh] Downloading ${MODEL_ID} to ${MODEL_DIR} ..."
  python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id   = os.environ["MODEL_ID"]
local_dir  = os.environ["MODEL_DIR"]
token      = os.environ.get("HF_TOKEN") or None

# Download snapshot (avoid stray index files creating weird layouts)
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    token=token,
    ignore_patterns=["*.safetensors.index.json"]
)
print(f"Downloaded {model_id} to {local_dir}")
PY
else
  echo "[run.sh] Using existing weights at ${MODEL_DIR}"
fi

# Per dots.ocr docs: set PYTHONPATH to the PARENT of the weights dir
# so vLLM can import the custom model code when --trust-remote-code is used.
export PYTHONPATH="$(dirname "${MODEL_DIR}"):${PYTHONPATH:-}"

# Launch vLLM OpenAI server
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --trust-remote-code \
  --download-dir "${DOWNLOAD_DIR}" \
  --host 0.0.0.0 \
  --port 8000
BASH

RUN chmod +x /app/run.sh

EXPOSE 8000
CMD ["/app/run.sh"]
