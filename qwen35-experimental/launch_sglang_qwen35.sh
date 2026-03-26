#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SGLANG_VENV="${QWEN35_SGLANG_VENV:-${REPO_ROOT}/.venv}"
QWEN35_VENV="${QWEN35_RUNTIME_VENV:-${REPO_ROOT}/.venv-qwen35}"
OVERLAY_DIR="${QWEN35_SGLANG_OVERLAY_DIR:-${REPO_ROOT}/.sglang-qwen35-overlay}"
ENABLE_OVERLAY="${QWEN35_SGLANG_ENABLE_OVERLAY:-0}"

if [[ "${ENABLE_OVERLAY,,}" =~ ^(1|true|yes|on)$ ]]; then
  mkdir -p "${OVERLAY_DIR}"
  rm -rf "${OVERLAY_DIR:?}/"*

  QWEN35_SITE_PACKAGES="$(find "${QWEN35_VENV}/lib" -maxdepth 3 -type d -path '*/site-packages' | head -n1)"
  if [[ -z "${QWEN35_SITE_PACKAGES}" ]]; then
    echo "Could not find site-packages under ${QWEN35_VENV}" >&2
    exit 1
  fi

  link_item() {
    local source_path="$1"
    local target_name="$2"
    if [[ -e "${source_path}" ]]; then
      ln -s "${source_path}" "${OVERLAY_DIR}/${target_name}"
    fi
  }

  link_item "${QWEN35_SITE_PACKAGES}/transformers" "transformers"
  link_item "${QWEN35_SITE_PACKAGES}/tokenizers" "tokenizers"
  link_item "${QWEN35_SITE_PACKAGES}/huggingface_hub" "huggingface_hub"
  link_item "${QWEN35_SITE_PACKAGES}/safetensors" "safetensors"

  for dist in tokenizers-*.dist-info huggingface_hub-*.dist-info safetensors-*.dist-info; do
    match="$(find "${QWEN35_SITE_PACKAGES}" -maxdepth 1 -name "${dist}" | head -n1 || true)"
    if [[ -n "${match}" ]]; then
      ln -s "${match}" "${OVERLAY_DIR}/$(basename "${match}")"
    fi
  done
fi

source "${SGLANG_VENV}/bin/activate"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

if [[ "${ENABLE_OVERLAY,,}" =~ ^(1|true|yes|on)$ ]]; then
  export PYTHONPATH="${OVERLAY_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

MODEL_ID="${QWEN35_SGLANG_MODEL_ID:-Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${QWEN35_SGLANG_SERVED_MODEL_NAME:-qwen3.5-4b-local}"
HOST="${QWEN35_SGLANG_HOST:-127.0.0.1}"
PORT="${QWEN35_SGLANG_PORT:-30101}"
CONTEXT_LENGTH="${QWEN35_SGLANG_CONTEXT_LENGTH:-2048}"
MEM_FRACTION_STATIC="${QWEN35_SGLANG_MEM_FRACTION_STATIC:-0.72}"
ATTENTION_BACKEND="${QWEN35_SGLANG_ATTENTION_BACKEND:-torch_native}"
CHUNKED_PREFILL_SIZE="${QWEN35_SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
MAX_LORA_RANK="${QWEN35_SGLANG_MAX_LORA_RANK:-16}"
REASONING_PARSER="${QWEN35_SGLANG_REASONING_PARSER:-qwen3}"
LORA_TARGET_MODULES="${QWEN35_SGLANG_LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}"
MODEL_IMPL="${QWEN35_SGLANG_MODEL_IMPL:-}"

EXTRA_ARGS=()
if [[ -n "${MODEL_IMPL}" ]]; then
  EXTRA_ARGS+=(--model-impl "${MODEL_IMPL}")
fi
if [[ "${QWEN35_SGLANG_DISABLE_CUDA_GRAPH:-0}" =~ ^(1|true|yes|on)$ ]]; then
  EXTRA_ARGS+=(--disable-cuda-graph)
fi

exec python -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --enable-lora \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --lora-target-modules ${LORA_TARGET_MODULES} \
  --reasoning-parser "${REASONING_PARSER}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --skip-server-warmup \
  "${EXTRA_ARGS[@]}"
