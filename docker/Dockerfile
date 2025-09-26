# ------------------------------------------------------
# Stage 1: fetch weights + code from Hugging Face
# ------------------------------------------------------
FROM python:3.10-slim AS builder

# System deps for git-lfs handled by HF hub; we only need python deps
RUN pip install --no-cache-dir "huggingface_hub>=0.24.6"

# Pull full model snapshot into /models/DotsOCR (no periods in the folder name)
RUN python - <<'PY'
from huggingface_hub import snapshot_download
import json
import shutil
from pathlib import Path

repo_id = "rednote-hilab/dots.ocr"
target = Path("/models/DotsOCR")
package_root = Path("/opt/dots_ocr_hf")
target_package = target / "dots_ocr_hf"
aggregator_defs = {
    "dots_ocr_hf_configuration_dots.py": "from dots_ocr_hf.configuration_dots import DotsOCRConfig\n",
    "dots_ocr_hf_modeling_dots_ocr.py": "from dots_ocr_hf.modeling_dots_ocr import DotsOCRForCausalLM\n",
}

snap = Path(snapshot_download(repo_id, repo_type="model"))

if target.exists():
    shutil.rmtree(target)
shutil.copytree(snap, target, dirs_exist_ok=True)

# Ensure it's a Python package (helps some loaders)
(target / "__init__.py").write_text("# package marker\n", encoding="utf-8")

# Mirror core modeling code locally for vLLM (no remote imports required)
if package_root.exists():
    shutil.rmtree(package_root)
package_root.mkdir(parents=True, exist_ok=True)

if target_package.exists():
    shutil.rmtree(target_package)
target_package.mkdir(parents=True, exist_ok=True)

for filename in [
    "configuration_dots.py",
    "modeling_dots_ocr.py",
    "modeling_dots_vision.py",
]:
    matches = list(target.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} in {target}")
    shutil.copy(matches[0], package_root / filename)
    shutil.copy(matches[0], target_package / filename)

(package_root / "__init__.py").write_text(
    "from .configuration_dots import DotsOCRConfig\n"
    "from .modeling_dots_ocr import DotsOCRForCausalLM\n",
    encoding="utf-8",
)

(target_package / "__init__.py").write_text(
    "from .configuration_dots import DotsOCRConfig\n"
    "from .modeling_dots_ocr import DotsOCRForCausalLM\n",
    encoding="utf-8",
)

for fname, content in aggregator_defs.items():
    (target / fname).write_text(content, encoding="utf-8")

# Patch config.json to point auto_map at the local package
cfg_path = target / "config.json"
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

cfg.setdefault("architectures", ["DotsOCRForCausalLM"])
cfg.setdefault("model_type", "dots")  # harmless if already present

cfg["auto_map"] = {
    "AutoConfig": "dots_ocr_hf_configuration_dots.DotsOCRConfig",
    "AutoModel": "dots_ocr_hf_modeling_dots_ocr.DotsOCRForCausalLM",
    "AutoModelForCausalLM": "dots_ocr_hf_modeling_dots_ocr.DotsOCRForCausalLM",
}

cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
print("Prepared local dots_ocr_hf package and patched", cfg_path)
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

# Copy the model snapshot (no dots in path) and local modeling package
COPY --from=builder /opt/dots_ocr_hf /app/dots_ocr_hf
COPY --from=builder /models/DotsOCR /models/DotsOCR

# Helpful caches/paths (optional)
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
# Use FlashInfer as the attention backend for vLLM
ENV VLLM_ATTENTION_BACKEND=FLASHINFER
ENV PYTHONPATH=/app:/models/DotsOCR:${PYTHONPATH}

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
