#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

LLAMA_SERVER_BIN="${OPENCLAW_27B_LLAMA_SERVER_BIN:-/home/main/.hyperspace/bin/llama-server}"
MODEL_PATH="${OPENCLAW_27B_MODEL_PATH:-}"
HF_REPO="${OPENCLAW_27B_HF_REPO:-llmfan46/Qwen3.5-27B-heretic-v3-GGUF}"
HF_FILE="${OPENCLAW_27B_HF_FILE:-Qwen3.5-27B-heretic-v3-Q4_K_M.gguf}"
CHAT_TEMPLATE_FILE="${OPENCLAW_27B_CHAT_TEMPLATE_FILE:-${SCRIPT_DIR}/heretic_plain_chat_template.jinja}"
CHAT_TEMPLATE_KWARGS="${OPENCLAW_27B_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false}}"
HOST="${OPENCLAW_27B_HOST:-127.0.0.1}"
PORT="${OPENCLAW_27B_PORT:-30121}"
CTX_SIZE="${OPENCLAW_27B_CTX_SIZE:-32768}"
GPU_LAYERS="${OPENCLAW_27B_GPU_LAYERS:-all}"
SPLIT_MODE="${OPENCLAW_27B_SPLIT_MODE:-layer}"
TENSOR_SPLIT="${OPENCLAW_27B_TENSOR_SPLIT:-1,1}"
DEVICE="${OPENCLAW_27B_DEVICE:-}"
MAIN_GPU="${OPENCLAW_27B_MAIN_GPU:-}"
FLASH_ATTN="${OPENCLAW_27B_FLASH_ATTN:-on}"
CACHE_TYPE_K="${OPENCLAW_27B_CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${OPENCLAW_27B_CACHE_TYPE_V:-q4_0}"
CACHE_RAM="${OPENCLAW_27B_CACHE_RAM:-}"
REASONING_FORMAT="${OPENCLAW_27B_REASONING_FORMAT:-none}"
REASONING_BUDGET="${OPENCLAW_27B_REASONING_BUDGET:-0}"
PROMPT_CACHE="${OPENCLAW_27B_PROMPT_CACHE:-off}"
TEMPERATURE="${OPENCLAW_27B_TEMPERATURE:-0}"
TOP_P="${OPENCLAW_27B_TOP_P:-1}"
PARALLEL="${OPENCLAW_27B_PARALLEL:-1}"
BATCH_SIZE="${OPENCLAW_27B_BATCH_SIZE:-1024}"
UBATCH_SIZE="${OPENCLAW_27B_UBATCH_SIZE:-512}"
THREADS="${OPENCLAW_27B_THREADS:-8}"
THREADS_BATCH="${OPENCLAW_27B_THREADS_BATCH:-8}"
NO_WARMUP="${OPENCLAW_27B_NO_WARMUP:-0}"

args=(
  --host "${HOST}"
  --port "${PORT}"
  --ctx-size "${CTX_SIZE}"
  --n-gpu-layers "${GPU_LAYERS}"
  --split-mode "${SPLIT_MODE}"
  --tensor-split "${TENSOR_SPLIT}"
  --flash-attn "${FLASH_ATTN}"
  --cache-type-k "${CACHE_TYPE_K}"
  --cache-type-v "${CACHE_TYPE_V}"
  --reasoning-format "${REASONING_FORMAT}"
  --reasoning-budget "${REASONING_BUDGET}"
  --temp "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --parallel "${PARALLEL}"
  --batch-size "${BATCH_SIZE}"
  --ubatch-size "${UBATCH_SIZE}"
  --threads "${THREADS}"
  --threads-batch "${THREADS_BATCH}"
  --no-webui
  --jinja
)

if [[ -n "${MODEL_PATH}" ]]; then
  args+=(-m "${MODEL_PATH}")
else
  args+=(--hf-repo "${HF_REPO}" --hf-file "${HF_FILE}")
fi

if [[ -n "${DEVICE}" ]]; then
  args+=(--device "${DEVICE}")
fi

if [[ -n "${MAIN_GPU}" ]]; then
  args+=(--main-gpu "${MAIN_GPU}")
fi

if [[ -n "${CHAT_TEMPLATE_FILE}" && -f "${CHAT_TEMPLATE_FILE}" ]]; then
  args+=(--chat-template-file "${CHAT_TEMPLATE_FILE}")
fi

if [[ -n "${CHAT_TEMPLATE_KWARGS}" ]]; then
  args+=(--chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}")
fi

if [[ -n "${CACHE_RAM}" ]]; then
  args+=(--cache-ram "${CACHE_RAM}")
fi

if [[ "${NO_WARMUP}" =~ ^(1|true|yes|on)$ ]]; then
  args+=(--no-warmup)
fi

case "${PROMPT_CACHE,,}" in
  0|false|no|off)
    args+=(--no-cache-prompt)
    ;;
  *)
    args+=(--cache-prompt)
    ;;
esac

exec "${LLAMA_SERVER_BIN}" "${args[@]}"
