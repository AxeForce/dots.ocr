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

# Torch cu128
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
      torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# vLLM + runtime deps (pin transformers to 4.51.3)
RUN pip install --no-cache-dir \
      vllm==0.9.1 \
      transformers==4.53.3 \
      "accelerate>=0.34.0" \
      sentencepiece \
      "protobuf<5"

# toolchain to compile JIT/AOT bits
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libc6-dev ninja-build && \
    rm -rf /var/lib/apt/lists/*

# FlashInfer 0.3.1 from GitHub — install from repo ROOT (no subdirectory)
RUN pip install --no-cache-dir --no-build-isolation -v \
    "git+https://github.com/flashinfer-ai/flashinfer.git@v0.3.1"

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
