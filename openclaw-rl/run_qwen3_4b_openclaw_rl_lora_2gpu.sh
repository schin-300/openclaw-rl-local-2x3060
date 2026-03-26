#!/bin/bash

set -euo pipefail
set -x

# Keep stdout/stderr unbuffered in Ray jobs.
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

NUM_GPUS=${NUM_GPUS:-2}
ACTOR_GPUS=${ACTOR_GPUS:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
TP=${TP:-1}

if (( ACTOR_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS must be <= NUM_GPUS"
    exit 1
fi

if (( ROLLOUT_GPUS > NUM_GPUS )); then
    echo "ROLLOUT_GPUS must be <= NUM_GPUS"
    exit 1
fi

if (( TP > ROLLOUT_GPUS )); then
    echo "TP must be <= ROLLOUT_GPUS"
    exit 1
fi

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
    echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60
# The 4B FSDP actor causes large transient host-memory spikes during init on this
# 62 GiB box. Ray's memory monitor has been killing the actor before Linux is out
# of reclaimable headroom, so disable the guard for this Qwen3-4B profile and
# leave a near-max threshold as a fallback for code paths that still read it.
export RAY_DISABLE_MEMORY_MONITOR="${RAY_DISABLE_MEMORY_MONITOR:-1}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.995}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
VENV_ROOT="${VIRTUAL_ENV:-${REPO_ROOT}/.venv}"
NVIDIA_LIB_GLOB=("${VENV_ROOT}"/lib/python3.*/site-packages/nvidia/*/lib)
NVIDIA_LIB_DIRS=()
for libdir in "${NVIDIA_LIB_GLOB[@]}"; do
    if [[ -d "${libdir}" ]]; then
        NVIDIA_LIB_DIRS+=("${libdir}")
    fi
done
if (( ${#NVIDIA_LIB_DIRS[@]} > 0 )); then
    NVIDIA_LD_PATH="$(IFS=:; echo "${NVIDIA_LIB_DIRS[*]}")"
    export LD_LIBRARY_PATH="${NVIDIA_LD_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
HF_CKPT="${HF_CKPT:-${REPO_ROOT}/models/${MODEL_NAME}}"
REF_LOAD="${REF_LOAD:-${HF_CKPT}}"
SAVE_CKPT="${SAVE_CKPT:-${REPO_ROOT}/ckpt/qwen3-4b-openclaw-rl-lora-2gpu}"

mkdir -p "${REPO_ROOT}/models" "${REPO_ROOT}/ckpt" "${SCRIPT_DIR}/results"

if [[ ! -f "${HF_CKPT}/config.json" ]]; then
    echo "Model not found at ${HF_CKPT}" >&2
    echo "Run ${SCRIPT_DIR}/prepare_qwen3_4b_2gpu.sh first, or set HF_CKPT to an existing model." >&2
    exit 1
fi

ray stop --force || true
pkill -f '[s]glang' || true
sleep 2

export SGLANG_API_KEY="${SGLANG_API_KEY:-openclaw-local}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-local}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30002}"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
export OPENCLAW_RECORD_FILE="${OPENCLAW_RECORD_FILE:-${SCRIPT_DIR}/results/qwen3_4b_2gpu_record.jsonl}"
export OPENCLAW_PRM_SHARED_POLICY="${OPENCLAW_PRM_SHARED_POLICY:-1}"
export PRM_M="${PRM_M:-1}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen}"
export REASONING_PARSER="${REASONING_PARSER:-qwen3}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-torch_native}"
export TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-2147483648}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-256}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --load "${SAVE_CKPT}"
   --save "${SAVE_CKPT}"
   --save-interval 1
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path openclaw_rollout.generate_rollout_openclaw
   --num-rollout "${NUM_ROLLOUT:-100000000}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-1}"
   --n-samples-per-prompt 1
   --rollout-max-response-len "${MAX_RESPONSE_LEN:-128}"
   --rollout-max-context-len "${CONTEXT_LENGTH}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.6}"
   --reward-key score
   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-512}"
   --gradient-checkpointing
   --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE:-67108864}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-rewards-normalization
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-5e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

LORA_ARGS=(
   --use-lora
   --lora-rank "${LORA_RANK:-8}"
   --lora-alpha "${LORA_ALPHA:-16}"
   --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${TP}"
   --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
   --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
   --sglang-mem-fraction-static "${MEM_FRACTION_STATIC}"
   --sglang-context-length "${CONTEXT_LENGTH}"
   --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
   --sglang-enable-metrics
   --sglang-reasoning-parser "${REASONING_PARSER}"
   --sglang-disable-custom-all-reduce
)

FAULT_TOLERANCE_ARGS=(
   --use-fault-tolerance
   --rollout-health-check-interval "${ROLLOUT_HEALTH_CHECK_INTERVAL:-5}"
   --rollout-health-check-timeout "${ROLLOUT_HEALTH_CHECK_TIMEOUT:-10}"
   --rollout-health-check-first-wait "${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-20}"
)

CUSTOM_ARGS=(
   --custom-generate-function-path openclaw_api_server.generate
   --custom-rm-path openclaw_api_server.reward_func
)

MISC_ARGS=(
   --train-backend fsdp
   --attn-implementation "${ATTN_IMPLEMENTATION}"
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
   --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${ACTOR_GPUS}"
   --rollout-num-gpus "${ROLLOUT_GPUS}"
   --num-gpus-per-node "${NUM_GPUS}"
   --fsdp-cpu-offload
)

USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-openclaw_rl}
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ "${USE_WANDB}" = "1" ] && [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group qwen3-4b-openclaw-rl-lora-2gpu
    --wandb-key "${WANDB_KEY_VALUE}"
  )
else
  WANDB_ARGS=()
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8266

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}/Megatron-LM:${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"OPENCLAW_PRM_SHARED_POLICY\": \"${OPENCLAW_PRM_SHARED_POLICY}\",
    \"SGLANG_API_KEY\": \"${SGLANG_API_KEY}\",
    \"TOOL_CALL_PARSER\": \"${TOOL_CALL_PARSER}\",
    \"REASONING_PARSER\": \"${REASONING_PARSER}\"
  }
}"

ray job submit --address="http://127.0.0.1:8266" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_ROOT}/train_async.py" \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${FAULT_TOLERANCE_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${LORA_ARGS[@]}
