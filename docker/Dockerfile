# docker/Dockerfile
FROM vllm/vllm-openai:v0.9.1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Your required pins
RUN pip3 install --no-cache-dir flash_attn==2.8.0.post2
RUN pip3 install --no-cache-dir transformers==4.51.3

# Get dots.ocr code because the downloader script lives here
WORKDIR /opt
RUN git clone https://github.com/rednote-hilab/dots.ocr.git
WORKDIR /opt/dots.ocr
RUN pip3 install -e .

# Runtime env
ENV HF_HOME=/root/.cache/huggingface \
    HF_MODEL_PATH=/models/DotsOCR \
    PYTHONUNBUFFERED=1

# Boot script:
#  - downloads weights to /models/DotsOCR if missing (note: no dots in dir name)
#  - sets PYTHONPATH to parent of weights dir
#  - registers DotsOCR with vLLM via sed (as per official instructions)
#  - finally execs the vLLM API server with any extra flags passed in (from Koyeb Command)
ADD <<'BASH' /app/boot.sh
#!/usr/bin/env bash
set -euo pipefail

echo "[boot] HF_MODEL_PATH=${HF_MODEL_PATH}"

mkdir -p "$(dirname "${HF_MODEL_PATH}")"

# 1) Download weights at runtime (first boot only)
if [ ! -d "${HF_MODEL_PATH}" ] || [ -z "$(ls -A "${HF_MODEL_PATH}" 2>/dev/null || true)" ]; then
  echo "[boot] Downloading model weights to ${HF_MODEL_PATH} ..."
  python3 tools/download_model.py --save_dir "${HF_MODEL_PATH}"
else
  echo "[boot] Using existing weights at ${HF_MODEL_PATH}"
fi

# 2) Set PYTHONPATH to the *parent* of the weights dir (required by dots.ocr doc)
export PYTHONPATH="$(dirname "${HF_MODEL_PATH}"):${PYTHONPATH:-}"
echo "[boot] PYTHONPATH=${PYTHONPATH}"

# 3) Inject DotsOCR registration into vllm CLI (idempotent)
VLLM_BIN="$(which vllm)"
if ! grep -q "from DotsOCR import modeling_dots_ocr_vllm" "$VLLM_BIN"; then
  echo "[boot] Patching vllm CLI to import DotsOCR modeling..."
  sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from DotsOCR import modeling_dots_ocr_vllm' "$VLLM_BIN" || true
fi

# 4) Start the OpenAI-compatible API server; pass through any flags from CMD
#    (Koyeb "Command" will supply things like --host/--port/--download-dir)
exec python3 -m vllm.entrypoints.openai.api_server "$@"
BASH

RUN chmod +x /app/boot.sh

# Make the boot script the entrypoint so it always runs before the server
ENTRYPOINT ["/app/boot.sh"]

EXPOSE 8000


