#!/bin/bash
# drkernel-14b-coldstart TRLOO training with pytorch eager reference backend
# 16× A800 (2 nodes), 16× 4090 reward (2 nodes)
# FSDP-8 intra-node, DP across nodes, SP=1
# Fix: switched from torch_compile to pytorch eager to match paper's reference backend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRKERNEL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_NAME="${PROJECT_NAME:-drkernel}"
RUN_LOG_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_RUN_PREFIX="${LOG_RUN_PREFIX:-trloo-14b-hfsdp8-pytorch-eager}"
TRAIN_LOG_HW="${TRAIN_LOG_HW:-16xA800}"
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
RUN_NAME="${RUN_NAME:-drkernel-14b-coldstart-trloo-hfsdp8-pytorch-eager}"
REWARD_MANAGER=kernel_async
REWARD_FUNC_NAME="calculate_reward_speedup"
REFERENCE_BACKEND="pytorch"

# =============================================================================
# Algorithm
# =============================================================================
ALGORITHM="trloo"
ROLLOUT_RS="geometric"
ROLLOUT_TOKEN_VETO_THRESHOLD=1e-4
ROLLOUT_RS_KWARGS="{lower:0.999,upper:1.001}"

# =============================================================================
# Reward config
# =============================================================================
KERNELGYM_SERVER_URL="${KERNELGYM_SERVER_URL:-http://192.168.16.39:8111}"
SPEEDUP_REWARD_UPPER_BOUND=3.0
SPEEDUP_REWARD_LOWER_BOUND=0.0

REWARD_TASK_TIMEOUT="${REWARD_TASK_TIMEOUT:-30}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-1800}"
REWARD_ACQUIRE_TIMEOUT="${REWARD_ACQUIRE_TIMEOUT:-2400}"
REWARD_MAX_CONCURRENT="${REWARD_MAX_CONCURRENT:-48}"
REWARD_MAX_RETRIES="${REWARD_MAX_RETRIES:-3}"
REWARD_PRINT_STATUS="${REWARD_PRINT_STATUS:-True}"
REWARD_TASK_TIMEOUT_CLIENT="${REWARD_TASK_TIMEOUT_CLIENT:-2400}"

# Timing: 5 warmup + 50 trials + 20% trim (5 from each end)
NUM_PERF_TRIALS=50
NUM_WARMUP=5
PERF_TRIM_COUNT=5

# Coverage reward
COVERAGE_RS="turn"
COVERAGE_RS_THRESHOLD=0.3
COVERAGE_RS_FACTOR=0.1
COVERAGE_RS_KEY="time_coverage"
COVERAGE_REWARD_TYPE="time_coverage"
COVERAGE_REWARD_WEIGHT=0.5
COVERAGE_REWARD_ENABLE=True

# =============================================================================
# Training hyperparams
# =============================================================================
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
IS_GET_LAST_TURN=True
ENABLE_MULTI_TURN=True
MAX_TURN="${MAX_TURN:-3}"
N_VAL="${N_VAL:-8}"

ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
ACTOR_PARAMETER_OFFLOAD="${ACTOR_PARAMETER_OFFLOAD:-False}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_TOKEN="${PPO_MICRO_TOKEN:-8192}"

AUTOMATIC_OVERSAMPLING=False
REJECTION_SAMPLE=True

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
ROLLOUT_N="${ROLLOUT_N:-16}"
KL_COEF=0.0
TOTAL_EPOCHS=1000

# =============================================================================
# Infrastructure: 16× A800 (2 nodes × 8 GPUs)
# =============================================================================
NNODES=2
GPUS_PER_NODE=8
SP_SIZE=4
FSDP_SIZE=8    # 8-way FSDP intra-node, DP across nodes

ROLLOUT_GPU_MEMORY_UTIL="${ROLLOUT_GPU_MEMORY_UTIL:-0.75}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
ENFORCE_EAGER="${ENFORCE_EAGER:-False}"

# Gradient checkpointing: every layer (interval=2 OOMs on 80GB A800)
ENABLE_GRADIENT_CHECKPOINTING=True
export GRADIENT_CHECKPOINT_INTERVAL=1

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
PROMPT_OVERSAMPLING_FACTOR="${PROMPT_OVERSAMPLING_FACTOR:-1.7}"
SAMPLE_OVERSAMPLING_FACTOR="${SAMPLE_OVERSAMPLING_FACTOR:-1.0}"
SAMPLE_SELECTION_STRATEGY=efficiency_stochastic
MAX_SKIP_STEPS=5

APPLY_CHAT_TEMPLATE=True
SAVE_FREQ=10
TEST_FREQ=10

# Checkpoint save to log dir under a stable checkpoints/ root
HDFS_CHECKPOINT_PATH="${RUN_LOG_DIR}/checkpoints"

# Reference cache
REFERENCE_CACHE_ENABLE=true
REFERENCE_CACHE_AUTO_UUID=true

# torch.compile crashes with current PyTorch version (CompiledFxGraph incompatibility)
ACTOR_USE_TORCH_COMPILE=""

# wandb 0.16.6 crashes with protobuf 5.x; disable until upgraded
export WANDB_MODE=disabled
TRAINER_LOGGERS="['console']"

source "${SCRIPT_DIR}/train_rl_common.sh"

main "$@"
