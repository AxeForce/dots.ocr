#!/usr/bin/env bash
set -euo pipefail

# Ensure directories on the mounted volume
mkdir -p /data/hf /data/models

# 1) Prefetch to the cache on the volume (idempotent & resumable)
python3 - <<'PY'
from huggingface_hub import snapshot_download
# You can pin a revision to avoid surprise code changes:
# revision="<commit-sha>"
snapshot_download("rednote-hilab/dots.ocr", resume_download=True)
PY

# 2) Find the cached repo path and copy/rename to a stable local model dir
PYVER=$(python3 - <<'PY'
import os, glob
home=os.environ.get("HF_HOME","/data/hf")
mods=os.environ.get("HF_MODULES_CACHE", home+"/modules")
print(home, mods)
PY
)
# Copy from the model cache (weights/config live under TRANSFORMERS_CACHE too),
# but simpler is: use the hub download return path; we just did that above.
python3 - <<'PY'
import os, shutil
from huggingface_hub import snapshot_download
dest = "/data/models/dots_ocr"  # no dot!
repo = snapshot_download("rednote-hilab/dots.ocr", resume_download=True)
if os.path.isdir(dest):
    shutil.rmtree(dest)
shutil.copytree(repo, dest)
print("Local model ready at", dest)
PY

# 3) Start vLLM serving the LOCAL path
exec python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 \
  --model /data/models/dots_ocr \
  --trust-remote-code --dtype bfloat16
