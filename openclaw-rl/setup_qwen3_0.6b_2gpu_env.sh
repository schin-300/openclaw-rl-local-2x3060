#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH" >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    uv venv --python 3.12 --seed "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if ! python - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("torch") else 1)
PY
then
    python -m pip install \
        torch==2.9.1+cu129 \
        torchvision==0.24.1+cu129 \
        torchaudio==2.9.1+cu129 \
        --index-url https://download.pytorch.org/whl/cu129
fi

# Keep the runtime lean for 2x3060 colocated FSDP + LoRA.
# We deliberately avoid flash-attn/apex/transformer-engine here and use SDPA instead.
python -m pip install \
    accelerate==1.12.0 \
    aiohttp==3.13.3 \
    blobfile==3.0.0 \
    datasets==4.4.2 \
    fastapi==0.128.0 \
    grpcio==1.75.1 \
    hf_transfer==0.1.9 \
    httpx==0.28.1 \
    httpx-sse==0.4.3 \
    huggingface-hub==0.36.0 \
    omegaconf==2.3.0 \
    peft==0.12.0 \
    pillow==12.1.0 \
    prometheus_client==0.23.1 \
    protobuf==6.33.2 \
    pydantic==2.12.5 \
    pylatexenc==2.10 \
    python-multipart==0.0.21 \
    PyYAML==6.0.3 \
    ray[default]==2.53.0 \
    requests==2.32.5 \
    safetensors==0.7.0 \
    sentencepiece==0.2.1 \
    starlette==0.50.0 \
    tensorboard==2.20.0 \
    tqdm==4.67.1 \
    transformers==4.57.1 \
    uvicorn==0.40.0 \
    wandb==0.24.1

python -m pip install \
    "git+https://github.com/sgl-project/sglang.git@dce8b0606c06d3a191a24c7b8cbe8e238ab316c9#egg=sglang&subdirectory=python" \
    sglang-router==0.3.2

python -m pip install \
    "flashinfer-jit-cache==0.5.3+cu129" \
    --index-url https://flashinfer.ai/whl/cu129

python - <<'PY'
import fastapi
import httpx
import peft
import ray
import sglang
import torch
import transformers
import uvicorn

print("torch", torch.__version__)
print("ray", ray.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("sglang", getattr(sglang, "__version__", "unknown"))
print("fastapi", fastapi.__version__)
print("uvicorn", uvicorn.__version__)
print("httpx", httpx.__version__)
PY
