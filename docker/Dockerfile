# -------- Stage 1: fetch source + weights + build dots-ocr wheel (CPU OK)
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs && \
    git lfs install && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "huggingface_hub>=0.24.6" build setuptools wheel setuptools-scm hatchling

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
# Build against Torch 2.7.0 cu128 and CUDA 12.8
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS flashattn-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential ninja-build git cmake \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
 && pip3 install --upgrade setuptools wheel packaging ninja

# Install Torch matching the runtime (CUDA 12.8 wheels)
ARG TORCH_VERSION=2.7.0
RUN pip3 install --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==${TORCH_VERSION}

# Build a FlashAttention wheel in the supported range for vLLM 0.9.1
# (2.7.1–2.7.4). Use 2.7.3 here.
RUN pip3 wheel --no-deps --no-build-isolation "flash-attn==2.7.3" -w /flashwheels


# -------- Stage 3: runtime (Torch + vLLM + DotsOCR + weights)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

# Basics + compiler for Triton JIT (torch.compile)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ca-certificates git build-essential libc6-dev \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Torch cu128
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# vLLM + deps (transformers pinned)
RUN pip install --no-cache-dir \
      vllm==0.9.1 \
      transformers==4.51.3 \
      "accelerate>=0.34.0" \
      sentencepiece \
      "protobuf<5"

# Install FlashAttention wheel (built in stage 2) + FlashInfer runtime
COPY --from=flashattn-builder /flashwheels/*.whl /tmp/flashattn/
RUN pip install --no-cache-dir /tmp/flashattn/*.whl && rm -rf /tmp/flashattn
RUN pip install --no-cache-dir "flashinfer>=0.0.9"

# Install dots-ocr wheel you built in Stage 1
COPY --from=builder /wheels/*.whl /tmp/dots-ocr/
RUN pip install --no-cache-dir /tmp/dots-ocr/*.whl && rm -rf /tmp/dots-ocr

# Bring in the baked model weights (dot-free folder)
COPY --from=builder /models/DotsOCR /models/DotsOCR

# Quick runtime env
ENV HF_MODEL_PATH=/models/DotsOCR
ENV PYTHONPATH=/models:${PYTHONPATH}
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Use FlashInfer fast path for FP8 KV cache; reduce CUDA mem overhead
ENV VLLM_ATTENTION_BACKEND=FLASHINFER
ENV NCCL_CUMEM_ENABLE=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Patch vLLM entrypoint to register the model (kept from your file)
RUN VLLM_BIN="$(which vllm)" && \
    sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a from DotsOCR import modeling_dots_ocr_vllm' "$VLLM_BIN" && \
    grep -n "modeling_dots_ocr_vllm" "$VLLM_BIN"

EXPOSE 8000

# --- Default command (PORT-aware). Tune the batching as needed.
CMD ["bash","-lc", "\
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
vllm serve ${HF_MODEL_PATH} \
  --host 0.0.0.0 --port ${PORT:-8000} \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --max-model-len 8192 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 512 \
  --served-model-name rednote-hilab/dots.ocr \
  --trust-remote-code \
"]