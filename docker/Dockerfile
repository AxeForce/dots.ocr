# ------------------------------------------------------
# Stage 1: fetch source + weights + build dots-ocr wheel
# ------------------------------------------------------
FROM python:3.10-slim AS builder

# System deps for git + LFS
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs && \
    git lfs install && rm -rf /var/lib/apt/lists/*

# Python build deps
RUN pip install --no-cache-dir "huggingface_hub>=0.24.6" \
    build setuptools wheel setuptools-scm hatchling

# Get dots.ocr repo
ARG DOTSOCR_REF=master
RUN git clone --depth 1 --branch "${DOTSOCR_REF}" https://github.com/rednote-hilab/dots.ocr.git /src/dots.ocr

# Download model weights into /models/DotsOCR
RUN python - <<'PY'
import os, shutil
from huggingface_hub import snapshot_download
repo = snapshot_download(
    "rednote-hilab/dots.ocr",
    resume_download=True,
    allow_patterns=["*.safetensors","*.bin","*.json","*.py",
                    "tokenizer*","*.model","*.txt","merges.txt","vocab.json"]
)
dest = "/models/DotsOCR"
if os.path.isdir(dest):
    shutil.rmtree(dest)
shutil.copytree(repo, dest)
PY

# CUDA 12.6 runtime to match cu126 wheels
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ca-certificates curl ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# ---------- Pin Torch 2.6 (cu126) and matching vision/audio ----------
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# VERY IMPORTANT: install libs that often try to upgrade torch, but block deps
RUN pip install --no-cache-dir --no-deps vllm==0.9.1
RUN pip install --no-cache-dir --no-deps transformers==4.51.3 accelerate==0.34.2
RUN pip install --no-cache-dir --no-deps sentencepiece==0.2.0 "protobuf<5"
# then allow deps in a small pass (won’t touch torch now that it’s pinned & satisfied)
RUN pip install --no-cache-dir "fastapi>=0.111,<1" "uvicorn>=0.29" "httpx>=0.26"

# ---------- FlashInfer: download the exact prebuilt wheel ----------
# We avoid directory listing/index issues by fetching the file directly.
# Python is 3.10 -> cp310; manylinux2014 is the common tag for x86_64 wheels.
ARG FI_VER=0.3.1
ARG PYTAG=cp310
ARG WHEEL_BASE=https://flashinfer.ai/whl/cu126/torch2.6

# Try both historical names (flashinfer / flashinfer_python), stop at first success.
RUN set -eux; \
  for pkg in flashinfer flashinfer_python; do \
    for tag in manylinux2014_x86_64 manylinux_2_17_x86_64.manylinux2014_x86_64; do \
      fname="${pkg}-${FI_VER}+cu126torch2.6-${PYTAG}-${PYTAG}-${tag}.whl"; \
      echo "Trying $WHEEL_BASE/$fname"; \
      if curl -fSL "$WHEEL_BASE/$fname" -o "/tmp/$fname"; then \
        pip install --no-cache-dir "/tmp/$fname"; \
        rm "/tmp/$fname"; \
        break 2; \
      fi; \
    done; \
  done

# Copy dots.ocr source + weights
COPY --from=builder /src/dots.ocr /app/dots.ocr
COPY --from=builder /models/DotsOCR /models/DotsOCR

WORKDIR /app

# Set FlashInfer as backend
ENV VLLM_ATTENTION_BACKEND=FLASHINFER

# Fix the CMD quoting and args
CMD ["bash","-lc","vllm serve /models/DotsOCR \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --max-model-len 8192 \
  --served-model-name rednote-hilab/dots.ocr \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 512"]
