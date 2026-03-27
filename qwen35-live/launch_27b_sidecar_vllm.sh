#!/usr/bin/env bash

set -euo pipefail

MODEL_REPO="${OPENCLAW_27B_MODEL_REPO:-cyankiwi/Qwen3.5-27B-AWQ-4bit}"
CONTAINER_NAME="${OPENCLAW_27B_CONTAINER_NAME:-openclaw-27b-sidecar}"
HOST="${OPENCLAW_27B_HOST:-127.0.0.1}"
PORT="${OPENCLAW_27B_PORT:-30121}"
GPU_MEMORY_UTILIZATION="${OPENCLAW_27B_GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${OPENCLAW_27B_MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${OPENCLAW_27B_MAX_NUM_SEQS:-1}"
TENSOR_PARALLEL_SIZE="${OPENCLAW_27B_TENSOR_PARALLEL_SIZE:-2}"
API_KEY="${OPENCLAW_27B_API_KEY:-openclaw-local}"
GPU_DEVICES="${OPENCLAW_27B_GPU_DEVICES:-0,1}"
HF_CACHE_DIR="${OPENCLAW_27B_HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
IMAGE="${OPENCLAW_27B_VLLM_IMAGE:-vllm/vllm-openai:cu130-nightly}"
REASONING_PARSER="${OPENCLAW_27B_REASONING_PARSER:-qwen3}"
KV_CACHE_DTYPE="${OPENCLAW_27B_KV_CACHE_DTYPE:-fp8}"
CPU_OFFLOAD_GB="${OPENCLAW_27B_CPU_OFFLOAD_GB:-8}"
ENFORCE_EAGER="${OPENCLAW_27B_ENFORCE_EAGER:-1}"
CALCULATE_KV_SCALES="${OPENCLAW_27B_CALCULATE_KV_SCALES:-1}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${OPENCLAW_27B_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ENABLE_PREFIX_CACHING="${OPENCLAW_27B_ENABLE_PREFIX_CACHING:-0}"
MAX_NUM_BATCHED_TOKENS="${OPENCLAW_27B_MAX_NUM_BATCHED_TOKENS:-32768}"
DISABLE_CUSTOM_ALL_REDUCE="${OPENCLAW_27B_DISABLE_CUSTOM_ALL_REDUCE:-0}"

mkdir -p "${HF_CACHE_DIR}"

docker_args=(
  --gpus all
  --ipc=host
  --name "${CONTAINER_NAME}"
  -p "${HOST}:${PORT}:8000"
  -e "CUDA_VISIBLE_DEVICES=${GPU_DEVICES}"
  -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF_VALUE}"
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
)

serve_args=(
  "${MODEL_REPO}"
  --host 0.0.0.0
  --port 8000
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --reasoning-parser "${REASONING_PARSER}"
  --language-model-only
  --api-key "${API_KEY}"
)

if [[ "${KV_CACHE_DTYPE}" != "auto" && -n "${KV_CACHE_DTYPE}" ]]; then
  serve_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

if [[ -n "${CPU_OFFLOAD_GB}" && "${CPU_OFFLOAD_GB}" != "0" ]]; then
  serve_args+=(--cpu-offload-gb "${CPU_OFFLOAD_GB}")
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  serve_args+=(--enforce-eager)
fi

if [[ "${CALCULATE_KV_SCALES}" == "1" ]]; then
  serve_args+=(--calculate-kv-scales)
fi

if [[ "${ENABLE_PREFIX_CACHING}" == "0" ]]; then
  serve_args+=(--no-enable-prefix-caching)
fi

if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  serve_args+=(--disable-custom-all-reduce)
fi

exec docker run --rm \
  "${docker_args[@]}" \
  "${IMAGE}" \
  "${serve_args[@]}"
