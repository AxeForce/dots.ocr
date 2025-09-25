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

# ------------------------------------------------------
# Stage 2: final runtime image with CUDA + vLLM + FlashInfer
# ------------------------------------------------------
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS runtime

# System deps (add ninja-build for native extensions that expect it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ca-certificates ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Symlink python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Torch with CUDA 12.8 (cu128)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0

# vLLM + friends
RUN pip install --no-cache-dir \
    vllm==0.9.1 \
    transformers==4.51.3 \
    "accelerate>=0.34.0" \
    sentencepiece \
    "protobuf<5"

# FlashInfer prebuilt wheels for CUDA 12.x
# Use --find-links (not --extra-index-url) and the correct host/path.
RUN pip install --no-cache-dir \
    --find-links https://flashinfer.ai/whl/cu12x/ \
    flashinfer==0.3.1 \
 || pip install --no-cache-dir \
    --find-links https://flashinfer.ai/whl/cu12x/ \
    flashinfer==0.3.0

# Copy dots.ocr source + weights
COPY --from=builder /src/dots.ocr /app/dots.ocr
COPY --from=builder /models/DotsOCR /models/DotsOCR

WORKDIR /app

# Set FlashInfer as backend
ENV VLLM_ATTENTION_BACKEND=FLASHINFER

# Default vLLM server command (override in Koyeb dashboard if needed)
CMD ["bash,"lc", "vllm", "serve", "/models/DotsOCR", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--trust-remote-code", "--max-model-len", "8192", \
     "--served-model-name", "rednote-hilab/dots.ocr", \
     "--gpu-memory-utilization", "0.95", \
     "--kv-cache-dtype", "fp8", \
     "--max-num-batched-tokens", "32768", \
     "--max-num-seqs", "512"]
