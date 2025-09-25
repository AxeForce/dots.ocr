# ------------------------------------------------------
# Stage 1: fetch weights + code from Hugging Face
# ------------------------------------------------------
FROM python:3.10-slim AS builder

# System deps for git-lfs handled by HF hub; we only need python deps
RUN pip install --no-cache-dir "huggingface_hub>=0.24.6"

# Download full snapshot (all modeling files + weights) into /models/DotsOCR
# Using a folder name WITHOUT periods avoids transformers dynamic-module path issues.
RUN python - <<'PY'
from huggingface_hub import snapshot_download
import os, shutil

repo_id = "rednote-hilab/dots.ocr"
target = "/models/DotsOCR"

# Pull everything (model weights + *.py including modeling_dots_ocr_vllm.py)
cache_dir = snapshot_download(
    repo_id,
    repo_type="model",
    local_files_only=False,
    ignore_patterns=[],
    allow_patterns=None,
)

# Copy to a clean target dir
if os.path.isdir(target):
    shutil.rmtree(target)
shutil.copytree(cache_dir, target, dirs_exist_ok=False)
print("Copied HF snapshot to", target)
PY


# ------------------------------------------------------
# Stage 2: runtime image with CUDA + Torch 2.7 + vLLM 0.10 + FlashInfer
# ------------------------------------------------------
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS runtime

# Base system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ca-certificates \
    git build-essential ninja-build curl \
 && rm -rf /var/lib/apt/lists/*

# Make "python" available
RUN ln -sf /usr/bin/python3 /usr/bin/python && python --version && pip --version

# Torch 2.7.0 (CUDA 12.8)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# Core libs: vLLM 0.10 + Transformers + friends
# (Transformers 4.52.0 works with the latest dots.ocr dynamic modules)
RUN pip install --no-cache-dir \
    "vllm==0.10.0" \
    "transformers==4.52.0" \
    "accelerate>=1.0.0" \
    "sentencepiece>=0.2.0" \
    "protobuf<5"

# FlashInfer: build from PyPI sdist (provides the 'flashinfer' python module)
# This path has proven to work reliably on runtime images.
# Try 0.3.1 first, fall back to 0.3.0 if needed.
RUN pip install --no-cache-dir "flashinfer_python==0.3.1" \
 || pip install --no-cache-dir "flashinfer_python==0.3.0"

# (Optional) quick sanity check
RUN python - <<'PY'
import importlib, sys
m = importlib.import_module("flashinfer")
print("flashinfer import OK; module:", m.__name__, "version:", getattr(m, "__version__", "unknown"))
PY

# Copy the model snapshot (no dots in path)
COPY --from=builder /models/DotsOCR /models/DotsOCR

# Helpful caches/paths (optional)
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
# Use FlashInfer as the attention backend for vLLM
ENV VLLM_ATTENTION_BACKEND=FLASHINFER

WORKDIR /app

# Expose the default vLLM port
EXPOSE 8000

# Run vLLM against the LOCAL folder (/models/DotsOCR)
# Notes:
# - We set --trust-remote-code so vLLM can import the custom modeling files we downloaded.
# - We keep the directory name free of '.' to avoid dynamic module import issues.
CMD ["bash","-lc","vllm serve /models/DotsOCR \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --max-model-len 8192 \
  --served-model-name rednote-hilab/dots.ocr \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 512"]
