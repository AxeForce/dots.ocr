# ------------------------------------------------------
# Stage 1: fetch weights + code from Hugging Face
# ------------------------------------------------------
FROM python:3.10-slim AS builder

# System deps for git-lfs handled by HF hub; we only need python deps
RUN pip install --no-cache-dir "huggingface_hub>=0.24.6"

# Pull full model snapshot into /models/DotsOCR (no periods in the folder name)
RUN python - <<'PY'
from huggingface_hub import snapshot_download
import os, shutil, json, pathlib

repo_id = "rednote-hilab/dots.ocr"
target  = "/models/DotsOCR"

snap = snapshot_download(repo_id, repo_type="model")

if os.path.isdir(target):
    shutil.rmtree(target)
shutil.copytree(snap, target, dirs_exist_ok=True)

# Ensure it's a Python package (helps some loaders)
pathlib.Path(target, "__init__.py").write_text("# package marker\n")

# Patch config.json to include auto_map so AutoModel* can resolve custom classes
cfg_path = os.path.join(target, "config.json")
with open(cfg_path, "r") as f:
    cfg = json.load(f)

cfg.setdefault("architectures", ["DotsOCRForCausalLM"])
cfg.setdefault("model_type", "dots")  # harmless if already present

auto_map = cfg.get("auto_map", {})
# Only fill if missing; do not overwrite if user already has entries
auto_map.setdefault("AutoConfig", "configuration_dots.DotsConfig")
# vLLM asks for a CausalLM-capable arch; provide both generic and CausalLM
auto_map.setdefault("AutoModel", "modeling_dots_ocr.DotsOCRForCausalLM")
auto_map.setdefault("AutoModelForCausalLM", "modeling_dots_ocr.DotsOCRForCausalLM")

cfg["auto_map"] = auto_map

with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
print("Patched auto_map in", cfg_path)
PY


# ------------------------------------------------------
# Stage 2: runtime image with CUDA + Torch 2.7 + vLLM 0.10 + FlashInfer
# ------------------------------------------------------
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS runtime

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
    "transformers==4.56.2" \
    "accelerate>=1.0.0" \
    "sentencepiece>=0.2.0" \
    "protobuf<5"

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging build

# FlashInfer: build from PyPI sdist (provides the 'flashinfer' python module)
# This path has proven to work reliably on runtime images.
RUN pip install --no-cache-dir -v \
    "git+https://github.com/flashinfer-ai/flashinfer.git@v0.3.1"

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
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 512"]
