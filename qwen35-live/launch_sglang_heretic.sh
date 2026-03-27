#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SGLANG_VENV="${OPENCLAW_HERETIC_SGLANG_VENV:-${REPO_ROOT}/.venv}"
MODEL_REPO="${OPENCLAW_HERETIC_MODEL_REPO:-llmfan46/Qwen3.5-27B-heretic-v3-GGUF}"
MODEL_FILE="${OPENCLAW_HERETIC_MODEL_FILE:-Qwen3.5-27B-heretic-v3-Q4_K_M.gguf}"
TOKENIZER_REPO="${OPENCLAW_HERETIC_TOKENIZER_REPO:-llmfan46/Qwen3.5-27B-heretic-v3}"
CACHE_DIR="${OPENCLAW_HERETIC_MODEL_CACHE_DIR:-${HOME}/.cache/openclaw-models/heretic-qwen35-27b-v3}"
SERVED_MODEL_NAME="${OPENCLAW_HERETIC_SERVED_MODEL_NAME:-qwen3.5-27b-heretic-v3-q4_k_m}"
HOST="${OPENCLAW_HERETIC_HOST:-127.0.0.1}"
PORT="${OPENCLAW_HERETIC_PORT:-30121}"
CONTEXT_LENGTH="${OPENCLAW_HERETIC_CONTEXT_LENGTH:-4096}"
MEM_FRACTION_STATIC="${OPENCLAW_HERETIC_MEM_FRACTION_STATIC:-0.78}"
ATTENTION_BACKEND="${OPENCLAW_HERETIC_ATTENTION_BACKEND:-torch_native}"
CHUNKED_PREFILL_SIZE="${OPENCLAW_HERETIC_CHUNKED_PREFILL_SIZE:-1024}"
TENSOR_PARALLEL_SIZE="${OPENCLAW_HERETIC_TENSOR_PARALLEL_SIZE:-2}"
REASONING_PARSER="${OPENCLAW_HERETIC_REASONING_PARSER:-qwen3}"
API_KEY="${OPENCLAW_HERETIC_API_KEY:-openclaw-local}"

source "${SGLANG_VENV}/bin/activate"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export CACHE_DIR MODEL_REPO MODEL_FILE TOKENIZER_REPO
mkdir -p "${CACHE_DIR}"

MODEL_PATH="$(python - <<'PY'
import os
from huggingface_hub import hf_hub_download

cache_dir = os.environ["CACHE_DIR"]
repo_id = os.environ["MODEL_REPO"]
filename = os.environ["MODEL_FILE"]
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="model",
    local_dir=os.path.join(cache_dir, "gguf"),
)
print(path)
PY
)"

TOKENIZER_PATH="$(python - <<'PY'
import os
from huggingface_hub import snapshot_download

cache_dir = os.environ["CACHE_DIR"]
repo_id = os.environ["TOKENIZER_REPO"]
path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=os.path.join(cache_dir, "tokenizer"),
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "chat_template.jinja",
    ],
)
print(path)
PY
)"

EXTRA_ARGS=()
if [[ "${OPENCLAW_HERETIC_DISABLE_CUDA_GRAPH:-0}" =~ ^(1|true|yes|on)$ ]]; then
  EXTRA_ARGS+=(--disable-cuda-graph)
fi

exec python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --load-format gguf \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --reasoning-parser "${REASONING_PARSER}" \
  --api-key "${API_KEY}" \
  --skip-server-warmup \
  "${EXTRA_ARGS[@]}"
