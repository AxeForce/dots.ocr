# -------- Stage 1: fetch source + weights (CPU OK)
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Install only what we need to download weights via HF Hub
RUN pip install --no-cache-dir "huggingface_hub>=0.24.6"

# 1) Get the DotsOCR repo (for version pinning/reference)
ARG DOTSOCR_REF=main
RUN git clone --depth 1 --branch ${DOTSOCR_REF} https://github.com/rednote-hilab/dots.ocr.git /src/dots.ocr

# 2) Download the model weights into a dot-free folder (/models/DotsOCR)
#    We use HF Hub directly to avoid extra build deps in this stage.
RUN python - <<'PY'
import os, shutil
from huggingface_hub import snapshot_download
repo_path = snapshot_download(
    "rednote-hilab/dots.ocr",
    resume_download=True,
    # keep it lean but safe; adjust if you need more files
    allow_patterns=["*.safetensors","*.bin","*.json","*.py","tokenizer*","*.model","*.txt","merges.txt","vocab.json"]
)
dest = "/models/DotsOCR"   # **NO DOTS** in dirname (per Quick Start)
if os.path.isdir(dest):
    shutil.rmtree(dest)
shutil.copytree(repo_path, dest)
print("Weights copied to", dest)
PY


# -------- Stage 2: CUDA runtime + Torch + vLLM + DotsOCR
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Basic runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Torch 2.7.0 with CUDA 12.8 wheels (as per Quick Start)
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# vLLM and runtime libs (pin vLLM 0.9.1 to match their evals)
RUN pip install --no-cache-dir \
      "vllm==0.9.1" \
      "transformers>=4.44.0" \
      "accelerate>=0.34.0" \
      "sentencepiece" \
      "protobuf<5"

# Bring in the DotsOCR source (we'll install it editable-style)
COPY --from=builder /src/dots.ocr /opt/DotsOCR-src

# Install the package so "from DotsOCR import ..." works
# The repo's package is named "DotsOCR" (capital D) in their examples.
WORKDIR /opt/DotsOCR-src
RUN pip install --no-cache-dir -e .

# Bring in the baked model weights (dot-free folder)
COPY --from=builder /models/DotsOCR /models/DotsOCR

# Required by Quick Start: set PYTHONPATH to the parent of hf_model_path
ENV HF_MODEL_PATH=/models/DotsOCR
ENV PYTHONPATH=/models:${PYTHONPATH}

# Patch vLLM entrypoint to register the model (Quick Start’s sed trick)
# This inserts: "from DotsOCR import modeling_dots_ocr_vllm" after the main import.
RUN VLLM_BIN="$(which vllm)" && \
    sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a from DotsOCR import modeling_dots_ocr_vllm' "$VLLM_BIN" && \
    echo "Patched: $VLLM_BIN" && \
    grep -n "modeling_dots_ocr_vllm" "$VLLM_BIN"

# vLLM server
EXPOSE 8000
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Launch exactly like the Quick Start suggests
# (served name can be anything; keeping HF id for clients)
CMD ["bash","-lc", "\
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
vllm serve ${HF_MODEL_PATH} \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --chat-template-content-format string \
  --served-model-name rednote-hilab/dots.ocr \
  --trust-remote-code \
"]
