#!/usr/bin/env bash

set -euo pipefail

MODEL_REPO="${OPENCLAW_27B_MODEL_REPO:-groxaxo/Qwen3.5-27B-heretic-v3-autoround-w4a16}"
CONTAINER_NAME="${OPENCLAW_27B_CONTAINER_NAME:-openclaw-27b-sidecar}"
HOST="${OPENCLAW_27B_HOST:-127.0.0.1}"
PORT="${OPENCLAW_27B_PORT:-30121}"
GPU_MEMORY_UTILIZATION="${OPENCLAW_27B_GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${OPENCLAW_27B_MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${OPENCLAW_27B_MAX_NUM_SEQS:-1}"
TENSOR_PARALLEL_SIZE="${OPENCLAW_27B_TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${OPENCLAW_27B_PIPELINE_PARALLEL_SIZE:-2}"
API_KEY="${OPENCLAW_27B_API_KEY:-openclaw-local}"
GPU_DEVICES="${OPENCLAW_27B_GPU_DEVICES:-0,1}"
HF_CACHE_DIR="${OPENCLAW_27B_HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
IMAGE="${OPENCLAW_27B_VLLM_IMAGE:-vllm/vllm-openai:latest}"
REASONING_PARSER="${OPENCLAW_27B_REASONING_PARSER:-qwen3}"
KV_CACHE_DTYPE="${OPENCLAW_27B_KV_CACHE_DTYPE:-fp8}"
CPU_OFFLOAD_GB="${OPENCLAW_27B_CPU_OFFLOAD_GB:-8}"
ENFORCE_EAGER="${OPENCLAW_27B_ENFORCE_EAGER:-1}"
CALCULATE_KV_SCALES="${OPENCLAW_27B_CALCULATE_KV_SCALES:-1}"
DISABLE_GDN_WARMUP="${OPENCLAW_27B_DISABLE_GDN_WARMUP:-1}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${OPENCLAW_27B_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ENABLE_PREFIX_CACHING="${OPENCLAW_27B_ENABLE_PREFIX_CACHING:-0}"
MAX_NUM_BATCHED_TOKENS="${OPENCLAW_27B_MAX_NUM_BATCHED_TOKENS:-4096}"
DISABLE_CUSTOM_ALL_REDUCE="${OPENCLAW_27B_DISABLE_CUSTOM_ALL_REDUCE:-0}"
DTYPE="${OPENCLAW_27B_DTYPE:-bfloat16}"
QUANTIZATION="${OPENCLAW_27B_VLLM_QUANTIZATION:-auto-round}"
ALLOW_DEPRECATED_QUANTIZATION="${OPENCLAW_27B_ALLOW_DEPRECATED_QUANTIZATION:-1}"
SERVED_MODEL_NAME="${OPENCLAW_27B_SERVED_MODEL_NAME:-qwen3.5-27b-heretic-v3-w4a16}"
TRITON_CACHE_DIR="${OPENCLAW_27B_TRITON_CACHE_DIR:-${HOME}/.cache/triton}"
VLLM_CACHE_DIR="${OPENCLAW_27B_VLLM_CACHE_DIR:-${HOME}/.cache/vllm}"
PATCH_SCRIPT_PATH="${OPENCLAW_27B_PATCH_SCRIPT_PATH:-${HOME}/OpenClaw-RL-merge-main/qwen35-live/vllm_patches/patch_qwen3_next.py}"
CHAT_TEMPLATE_FILE="${OPENCLAW_27B_CHAT_TEMPLATE_FILE:-${HOME}/OpenClaw-RL-merge-main/qwen35-live/heretic_plain_chat_template.jinja}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${OPENCLAW_27B_DEFAULT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false}}"

mkdir -p "${HF_CACHE_DIR}"
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${VLLM_CACHE_DIR}"

docker_args=(
  --gpus all
  --ipc=host
  --name "${CONTAINER_NAME}"
  -p "${HOST}:${PORT}:8000"
  -e "CUDA_VISIBLE_DEVICES=${GPU_DEVICES}"
  -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF_VALUE}"
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
  -v "${TRITON_CACHE_DIR}:/root/.triton"
  -v "${VLLM_CACHE_DIR}:/root/.cache/vllm"
  -v "${CHAT_TEMPLATE_FILE}:/opt/openclaw/heretic_plain_chat_template.jinja:ro"
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
  --language-model-only
  --api-key "${API_KEY}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --chat-template /opt/openclaw/heretic_plain_chat_template.jinja
  --default-chat-template-kwargs "${DEFAULT_CHAT_TEMPLATE_KWARGS}"
)

if [[ "${PIPELINE_PARALLEL_SIZE}" != "1" ]]; then
  serve_args+=(--pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}")
fi

if [[ -n "${REASONING_PARSER}" && "${REASONING_PARSER}" != "none" ]]; then
  serve_args+=(--reasoning-parser "${REASONING_PARSER}")
fi

if [[ -n "${DTYPE}" ]]; then
  serve_args+=(--dtype "${DTYPE}")
fi

if [[ -n "${QUANTIZATION}" ]]; then
  serve_args+=(--quantization "${QUANTIZATION}")
fi

if [[ "${ALLOW_DEPRECATED_QUANTIZATION}" == "1" ]]; then
  serve_args+=(--allow-deprecated-quantization)
fi

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

if [[ "${DISABLE_GDN_WARMUP}" == "1" ]]; then
  docker_args+=(
    -v "${PATCH_SCRIPT_PATH}:/opt/openclaw/patch_qwen3_next.py:ro"
  )
  exec docker run --rm \
    "${docker_args[@]}" \
    --entrypoint /bin/sh \
    "${IMAGE}" \
    -lc 'python3 /opt/openclaw/patch_qwen3_next.py && exec /usr/local/bin/vllm serve "$@"' -- \
    "${serve_args[@]}"
fi

exec docker run --rm \
  "${docker_args[@]}" \
  "${IMAGE}" \
  "${serve_args[@]}"
