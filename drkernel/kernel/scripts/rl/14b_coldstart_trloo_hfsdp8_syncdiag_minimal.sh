#!/bin/bash
# Minimal actor-to-vLLM sync diagnostic for the 14B coldstart TRLOO setup.
# This is intentionally not a production training script. It keeps validation,
# checkpointing, and reward trials off/minimal, and disables sample filtering by
# default so a tiny diagnostic batch can survive long enough to inspect sync.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRKERNEL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_NAME="${PROJECT_NAME:-drkernel}"
RUN_LOG_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_RUN_PREFIX="${LOG_RUN_PREFIX:-trloo-14b-hfsdp8-syncdiag-minimal}"
TRAIN_LOG_HW="${TRAIN_LOG_HW:-8xA800}"
REWARD_LOG_HW="${REWARD_LOG_HW:-16x4090}"
RUN_LOG_PHASE="${RUN_LOG_PHASE:-run}"
RUN_LOG_BASENAME="${RUN_LOG_BASENAME:-${LOG_RUN_PREFIX}.train.${TRAIN_LOG_HW}.reward.${REWARD_LOG_HW}.${RUN_LOG_PHASE}.${RUN_LOG_TIMESTAMP}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${DRKERNEL_ROOT}/logs/${RUN_LOG_BASENAME}}"
MAIN_LOG="${MAIN_LOG:-${RUN_LOG_DIR}/main.log}"
TRAINER_LOG="${TRAINER_LOG:-${RUN_LOG_DIR}/trainer.log}"
ROLLOUT_LOG="${ROLLOUT_LOG:-${RUN_LOG_DIR}/rollout.log}"
REWARD_LOG="${REWARD_LOG:-${RUN_LOG_DIR}/reward.log}"
VLLM_LOG="${VLLM_LOG:-${RUN_LOG_DIR}/vllm.log}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUN_LOG_DIR}/torchinductor_cache}"

if [ "${DRKERNEL_LOGGING_INITIALIZED:-0}" != "1" ]; then
    mkdir -p "${RUN_LOG_DIR}/structured" "${TORCHINDUCTOR_CACHE_DIR}"
    export DRKERNEL_LOGGING_INITIALIZED=1
    export DRKERNEL_EVENT_LOG_DIR="${RUN_LOG_DIR}/structured"
    export MAIN_LOG
    export TRAINER_LOG
    export ROLLOUT_LOG
    export REWARD_LOG
    export VLLM_LOG
    export TORCHINDUCTOR_CACHE_DIR
    echo "Logging training run to: ${MAIN_LOG}"
    exec > >(python -u "${SCRIPT_DIR}/log_router.py") 2>&1
fi

# =============================================================================
# Data
# =============================================================================
TRAIN_DATASET=("/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM/drkernel/data/drkernel-rl-data/cuda_llm_rl_thinking_1025.parquet")
VALID_DATASET=("/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM/drkernel/data/drkernel-validation-data/validation_data_thinking.parquet")

# =============================================================================
# Model
# =============================================================================
MODEL_NAME="${MODEL_NAME:-hkust-nlp/drkernel-14b-coldstart}"
MODEL_PATH="${MODEL_PATH:-/nfs/FM/chenshuailin/checkpoints/hkust-nlp/drkernel-14b-coldstart}"

# =============================================================================
# Run identity
# =============================================================================
RUN_NAME="${RUN_NAME:-drkernel-14b-coldstart-trloo-hfsdp8-syncdiag-minimal}"
REWARD_MANAGER=kernel_async
REWARD_FUNC_NAME="calculate_reward_speedup"
REFERENCE_BACKEND="pytorch"

# =============================================================================
# Algorithm
# =============================================================================
ALGORITHM="trloo"
ROLLOUT_RS="${ROLLOUT_RS:-null}"
ROLLOUT_TOKEN_VETO_THRESHOLD="${ROLLOUT_TOKEN_VETO_THRESHOLD:-null}"
ROLLOUT_RS_KWARGS="${ROLLOUT_RS_KWARGS:-{lower:0.999,upper:1.001}}"

# =============================================================================
# Reward config: intentionally tiny for sync diagnostics.
# =============================================================================
KERNELGYM_SERVER_URL="${KERNELGYM_SERVER_URL:-http://192.168.16.39:8111}"
SPEEDUP_REWARD_UPPER_BOUND=3.0
SPEEDUP_REWARD_LOWER_BOUND=0.0

REWARD_TASK_TIMEOUT="${REWARD_TASK_TIMEOUT:-90}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-180}"
REWARD_ACQUIRE_TIMEOUT="${REWARD_ACQUIRE_TIMEOUT:-300}"
REWARD_MAX_CONCURRENT="${REWARD_MAX_CONCURRENT:-1}"
REWARD_MAX_RETRIES="${REWARD_MAX_RETRIES:-0}"
REWARD_PRINT_STATUS="${REWARD_PRINT_STATUS:-False}"
REWARD_TASK_TIMEOUT_CLIENT="${REWARD_TASK_TIMEOUT_CLIENT:-300}"

NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-1}"
NUM_WARMUP="${NUM_WARMUP:-0}"
PERF_TRIM_COUNT="${PERF_TRIM_COUNT:-0}"

COVERAGE_RS="${COVERAGE_RS:-null}"
COVERAGE_RS_THRESHOLD=0.3
COVERAGE_RS_FACTOR=0.1
COVERAGE_RS_KEY="time_coverage"
COVERAGE_REWARD_TYPE="time_coverage"
COVERAGE_REWARD_WEIGHT=0.5
COVERAGE_REWARD_ENABLE=True

# =============================================================================
# Training hyperparams: minimal diagnostic settings.
# =============================================================================
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
IS_GET_LAST_TURN=True
ENABLE_MULTI_TURN=True
MAX_TURN="${MAX_TURN:-1}"
N_VAL="${N_VAL:-1}"

ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
ACTOR_PARAMETER_OFFLOAD="${ACTOR_PARAMETER_OFFLOAD:-False}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_TOKEN="${PPO_MICRO_TOKEN:-8192}"

AUTOMATIC_OVERSAMPLING=False
REJECTION_SAMPLE=False

CLIP_RATIO=0.2_0.28
ENTROPY_CLIP_RATE=0.0
GRAD_CLIP=1.0
VLLM_IS_THRESHOLD=2.0
EXTREME_RISK_PROB_THRESHOLD=null
KL_LOSS_COEF=0.0
ENTROPY_COEFFIENT=0.0
KL_LOSS_TYPE="low_var_kl"

TEMPERATURE=1.0
MIN_P=0.0
TOP_P=1.0
TOP_K=-1
ROLLOUT_N="${ROLLOUT_N:-2}"
KL_COEF=0.0
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

# =============================================================================
# Infrastructure: default to one 8-GPU A800 node for speed.
# =============================================================================
NNODES="${NNODES:-1}"
GPUS_PER_NODE=8
SP_SIZE="${SP_SIZE:-4}"
FSDP_SIZE=8

ROLLOUT_GPU_MEMORY_UTIL="${ROLLOUT_GPU_MEMORY_UTIL:-0.75}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
ENFORCE_EAGER="${ENFORCE_EAGER:-False}"

ENABLE_GRADIENT_CHECKPOINTING=True
export GRADIENT_CHECKPOINT_INTERVAL=1

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
PROMPT_OVERSAMPLING_FACTOR="${PROMPT_OVERSAMPLING_FACTOR:-1.0}"
SAMPLE_OVERSAMPLING_FACTOR="${SAMPLE_OVERSAMPLING_FACTOR:-1.0}"
SAMPLE_SELECTION_STRATEGY=efficiency_stochastic
MAX_SKIP_STEPS="${MAX_SKIP_STEPS:-0}"

APPLY_CHAT_TEMPLATE=True
SAVE_FREQ=0
TEST_FREQ=0

# Checkpoint save to log dir, though saving is disabled above.
HDFS_CHECKPOINT_PATH="${RUN_LOG_DIR}/checkpoints"

# Keep the same pytorch eager reference backend behavior as the current run.
REFERENCE_CACHE_ENABLE=true
REFERENCE_CACHE_AUTO_UUID=true
ACTOR_USE_TORCH_COMPILE=""

# Enable gated actor/vLLM sync diagnostics in Ray workers.
export DRKERNEL_SYNC_DIAG="${DRKERNEL_SYNC_DIAG:-1}"

export WANDB_MODE=disabled
TRAINER_LOGGERS="['console']"

# Hydra search paths in kernel_trainer.yaml include paths relative to drkernel/.
cd "${DRKERNEL_ROOT}"

source "${SCRIPT_DIR}/train_rl_common.sh"

main "$@"
