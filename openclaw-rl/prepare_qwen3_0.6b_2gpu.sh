#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B}"
MODEL_REPO="${MODEL_REPO:-Qwen/${MODEL_NAME}}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models/${MODEL_NAME}}"

mkdir -p "$(dirname "${MODEL_DIR}")"

if [[ -f "${MODEL_DIR}/config.json" ]]; then
    echo "Model already present at ${MODEL_DIR}"
    exit 0
fi

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "${UV_BIN}" ]]; then
    echo "uv is required to download the model. Install uv first." >&2
    exit 1
fi

"${UV_BIN}" tool run --from huggingface_hub hf download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"

echo "Model downloaded to ${MODEL_DIR}"
