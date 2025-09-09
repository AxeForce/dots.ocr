# -------- Stage 1: fetch source + weights + build dots-ocr wheel (CPU OK)
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs && \
    git lfs install && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir huggingface_hub>=0.24.6 build setuptools wheel setuptools-scm hatchling

ARG DOTSOCR_REF=master
RUN git clone --depth 1 --branch "${DOTSOCR_REF}" https://github.com/rednote-hilab/dots.ocr.git /src/dots.ocr

# Download model weights to a dot-free folder
RUN python - <<'PY'
import os, shutil
from huggingface_hub import snapshot_download
repo_path = snapshot_download(
    "rednote-hilab/dots.ocr",
    resume_download=True,
    allow_patterns=["*.safetensors","*.bin","*.json","*.py","tokenizer*","*.model","*.txt","merges.txt","vocab.json"]
)
dest = "/models/DotsOCR"
if os.path.isdir(dest): shutil.rmtree(dest)
shutil.copytree(repo_path, dest)
print("Weights copied to", dest)
PY

WORKDIR /src/dots.ocr
RUN python -m build --wheel -o /wheels


# -------- Stage 2: build FlashAttention wheel (needs nvcc)
# --- Builder: compile flash-attn against CUDA 12.1 + Torch 2.7.0
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS flashattn-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential ninja-build git cmake \
 && rm -rf /var/lib/apt/lists/*

# make sure build tooling exists (this fixes your 'packaging' error)
RUN python3 -m pip install --upgrade pip \
 && pip3 install --upgrade setuptools wheel packaging ninja

# Install the exact Torch you target (CUDA 12.1 wheels)
ARG TORCH_VERSION=2.7.0
RUN pip3 install --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==${TORCH_VERSION}


# Build wheel
RUN pip3 wheel --no-deps --no-build-isolation "flash-attn==2.8.0.post2" -w /flashwheels

# --- Runtime: install torch + the prebuilt wheel
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Torch must match the builder’s version/CUDA
ARG TORCH_VERSION=2.7.0
RUN python3 -m pip install --upgrade pip \
 && pip3 install --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==${TORCH_VERSION}

COPY --from=flashattn-builder /flashwheels /flashwheels
RUN pip3 install /flashwheels/flash_attn-2.8.0.post2-*.whl


# -------- Stage 3: runtime (Torch + vLLM + DotsOCR + weights)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Basics
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ca-certificates git && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

# Torch cu128
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# vLLM + runtime deps (pin transformers to 4.51.3)
RUN pip install --no-cache-dir \
      vllm==0.9.1 \
      transformers==4.51.3 \
      "accelerate>=0.34.0" \
      sentencepiece \
      "protobuf<5"

# Install prebuilt FlashAttention (no compile in runtime)
COPY --from=flashattn-builder /flashwheels/*.whl /tmp/flashattn/
RUN pip install --no-cache-dir /tmp/flashattn/*.whl && rm -rf /tmp/flashattn

# Install dots-ocr wheel you built in Stage 1
COPY --from=builder /wheels/*.whl /tmp/dots-ocr/
RUN pip install --no-cache-dir /tmp/dots-ocr/*.whl && rm -rf /tmp/dots-ocr

# Bring in the baked model weights (dot-free folder)
COPY --from=builder /models/DotsOCR /models/DotsOCR

# Quick Start env
ENV HF_MODEL_PATH=/models/DotsOCR
ENV PYTHONPATH=/models:${PYTHONPATH}
# Helps with CUDA memory fragmentation
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Patch vLLM entrypoint to register the model
RUN VLLM_BIN="$(which vllm)" && \
    sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a from DotsOCR import modeling_dots_ocr_vllm' "$VLLM_BIN" && \
    grep -n "modeling_dots_ocr_vllm" "$VLLM_BIN"

EXPOSE 8000
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

CMD ["bash","-lc", "\
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
vllm serve ${HF_MODEL_PATH} \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --chat-template-content-format string \
  --served-model-name rednote-hilab/dots.ocr \
  --trust-remote-code \
"]
