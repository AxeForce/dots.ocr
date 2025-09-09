# docker/Dockerfile
# Base: vLLM OpenAI server v0.9.1
FROM vllm/vllm-openai:v0.9.1

# (Optional) tools used by healthcheck and logs
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Your required pins
RUN pip3 install --no-cache-dir flash_attn==2.8.0.post2
RUN pip3 install --no-cache-dir transformers==4.51.3

# Ensure we have a recent huggingface_hub for snapshot_download
RUN pip3 install --no-cache-dir "huggingface_hub>=0.25.0"

# Runtime env
ENV MODEL_ID="rednote-hilab/dots.ocr" \
    DOWNLOAD_DIR="/models" \
    HF_HOME="/root/.cache/huggingface" \
    PYTHONUNBUFFERED=1 \
    VLLM_LOGGING_LEVEL=INFO

# Entry script:
# - downloads weights to /models (persisted via volume)
# - launches vLLM OpenAI-compatible server
ADD <<'BASH' /app/run.sh
#!/usr/bin/env bash
set -euo pipefail
: "${HF_TOKEN:=}"   # set only if the HF repo is gated/private

mkdir -p "${DOWNLOAD_DIR}"

# Start vLLM; it will pull weights (or reuse cache/volume)
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --trust-remote-code \
  --download-dir "${DOWNLOAD_DIR}" \
  --host 0.0.0.0 \
  --port 8000
BASH

RUN chmod +x /app/run.sh

EXPOSE 8000
CMD ["/app/run.sh"]
