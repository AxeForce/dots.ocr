# -------- Stage 1: fetch source + weights + build dots-ocr wheel (CPU OK)
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs && \
    git lfs install && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "huggingface_hub>=0.24.6" build setuptools wheel setuptools-scm hatchling

ARG DOTSOCR_REF=master
RUN git clone --depth 1 --branch "${DOTSOCR_REF}" https://github.com/rednote-hilab/dots.ocr.git /src/dots.ocr

# Download DotsOCR weights to a dot-free folder
RUN python - <<'PY'
import os, shutil
from huggingface_hub import snapshot_download
repo = snapshot_download(
    "rednote-hilab/dots.ocr",
    resume_download=True,
    allow_patterns=["*.safetensors","*.bin","*.json","*.py","tokenizer*","*.model","*.txt","merges.txt","vocab.json"]
)
dest = "/models/DotsOCR"
if os.path.isdir(dest): shutil.rmtree(dest)
shutil.copytree(repo, dest)
print("DotsOCR weights ->", dest)
PY

# >>> QWEN3: download Qwen3-8B-Instruct weights too
ARG QWEN3_REPO="Qwen/Qwen3-4B-FP8"
RUN python - <<'PY'
import os, shutil
from huggingface_hub import snapshot_download
repo = snapshot_download(
    os.environ.get("QWEN3_REPO"),
    resume_download=True
)
dest = "/models/Qwen3-4B-FP8"
if os.path.isdir(dest): shutil.rmtree(dest)
shutil.copytree(repo, dest)
print("Qwen3-4B-FP8 weights ->", dest)
PY

WORKDIR /src/dots.ocr
RUN python -m build --wheel -o /wheels


# -------- Stage 2: build FlashAttention wheel (needs nvcc)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS flashattn-builder
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_HOME=/usr/local/cuda MAX_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential ninja-build git cmake \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip && pip3 install --upgrade setuptools wheel packaging ninja
ARG TORCH_VERSION=2.7.0
RUN pip3 install --extra-index-url https://download.pytorch.org/whl/cu128 torch==${TORCH_VERSION}
RUN pip3 wheel --no-deps --no-build-isolation "flash-attn==2.8.0.post2" -w /flashwheels


# -------- Stage 3: runtime (Torch + vLLM + DotsOCR + weights)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates git tini \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

# Torch cu128
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# vLLM + deps
RUN pip install --no-cache-dir \
      vllm==0.9.1 \
      transformers==4.51.3 \
      "accelerate>=0.34.0" \
      sentencepiece \
      "protobuf<5"

# FlashAttention wheel (prebuilt)
COPY --from=flashattn-builder /flashwheels/*.whl /tmp/flashattn/
RUN pip install --no-cache-dir /tmp/flashattn/*.whl && rm -rf /tmp/flashattn

# dots-ocr wheel
COPY --from=builder /wheels/*.whl /tmp/dots-ocr/
RUN pip install --no-cache-dir /tmp/dots-ocr/*.whl && rm -rf /tmp/dots-ocr

# Bring models
COPY --from=builder ["/models/DotsOCR", "/models/DotsOCR"]
COPY --from=builder ["/models/Qwen3-4B-FP8", "/models/Qwen3-4B-FP8"]

# Quick Start env
ENV HF_DOTS_PATH=/models/DotsOCR
ENV HF_QWEN3_PATH=/models/Qwen3-4B-FP8
ENV PYTHONPATH=/models:${PYTHONPATH}
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Patch vLLM entrypoint to register DotsOCR
RUN VLLM_BIN="$(which vllm)" && \
    sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a from DotsOCR import modeling_dots_ocr_vllm' "$VLLM_BIN" && \
    grep -n "modeling_dots_ocr_vllm" "$VLLM_BIN"

# >>> multi-serve: tiny supervisor script to run both servers
RUN printf '%s\n' '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'PORT_OCR="${PORT_OCR:-8000}"' \
  'PORT_QWEN="${PORT_QWEN:-8001}"' \
  'GPU="${CUDA_VISIBLE_DEVICES:-0}"' \
  'export CUDA_VISIBLE_DEVICES="$GPU"' \
  '' \
  'echo "Starting DotsOCR on :$PORT_OCR (GPU $GPU)..."' \
  'vllm serve "$HF_DOTS_PATH" --tensor-parallel-size 1 --gpu-memory-utilization 0.95 \' \
  '  --chat-template-content-format string \' \
  '  --served-model-name rednote-hilab/dots.ocr \' \
  '  --trust-remote-code --enforce-eager --port "$PORT_OCR" &' \
  '' \
  'echo "Starting Qwen3-4B-FP8-Instruct on :$PORT_QWEN (GPU $GPU)..."' \
  'vllm serve "$HF_QWEN3_PATH" --tensor-parallel-size 1 --gpu-memory-utilization 0.95 \' \
  '  --served-model-name Qwen/Qwen3-4B-FP8 \' \
  '  --trust-remote-code --port "$PORT_QWEN" &' \
  '' \
  'wait -n' > /usr/local/bin/start-both.sh \
  && chmod +x /usr/local/bin/start-both.sh

EXPOSE 8000 8001
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/usr/local/bin/start-both.sh"]
