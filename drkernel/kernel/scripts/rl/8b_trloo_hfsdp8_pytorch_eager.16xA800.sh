#!/bin/bash
# drkernel-8b TRLOO training with pytorch eager reference backend
# 16× A800 (2 nodes), 16× 4090 reward (2 nodes)
# FSDP-8 intra-node, DP across nodes, SP=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRKERNEL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/launcher_common.sh"

LOG_RUN_PREFIX="${LOG_RUN_PREFIX:-trloo-8b-hfsdp8-pytorch-eager}"
TRAIN_LOG_HW="${TRAIN_LOG_HW:-16xA800}"
REWARD_LOG_HW="${REWARD_LOG_HW:-16x4090}"
init_launcher_run_state
enable_launcher_log_router

# =============================================================================
# Data
# =============================================================================
TRAIN_DATASET=("/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM/drkernel/data/drkernel-rl-data/cuda_llm_rl_thinking_1025.parquet")
VALID_DATASET=("/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM/drkernel/data/drkernel-validation-data/validation_data_thinking.parquet")

# =============================================================================
# Model
# =============================================================================
MODEL_NAME="${MODEL_NAME:-hkust-nlp/drkernel-8b-coldstart}"
MODEL_PATH="${MODEL_PATH:-/nfs/FM/chenshuailin/checkpoints/hkust-nlp/drkernel-8b-coldstart}"

# =============================================================================
# Run identity
# =============================================================================
RUN_NAME="${RUN_NAME:-drkernel-8b-coldstart-trloo-hfsdp8-pytorch-eager}"
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
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32}"

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
PROMPT_OVERSAMPLING_FACTOR="${PROMPT_OVERSAMPLING_FACTOR:-2.0}"
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
