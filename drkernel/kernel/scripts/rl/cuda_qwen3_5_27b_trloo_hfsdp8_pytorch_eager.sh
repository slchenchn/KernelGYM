#!/bin/bash
# Qwen3.5-27B CUDA TRLOO training based on the hfsdp8 pytorch-eager launcher.
# Keep CUDA-specific data/prompt config here; shared mechanics live in train_rl_common.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRKERNEL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/launcher_common.sh"

LOG_RUN_PREFIX="${LOG_RUN_PREFIX:-trloo-qwen35-27b-cuda-hfsdp8-pytorch-eager}"
TRAIN_LOG_HW="${TRAIN_LOG_HW:-8XA800}"
REWARD_LOG_HW="${REWARD_LOG_HW:-16x4090}"
init_launcher_run_state
enable_launcher_log_router

# =============================================================================
# Data
# =============================================================================
HYDRA_CONFIG_NAME="${HYDRA_CONFIG_NAME:-cuda_kernel_trainer}"
TRAIN_DATASET=("${DRKERNEL_ROOT}/data/drkernel-rl-data-neutral/cuda_llm_rl_thinking_1025.parquet")
VALID_DATASET=("${DRKERNEL_ROOT}/data/drkernel-validation-data-neutral/validation_data_thinking.parquet")

# =============================================================================
# Model
# =============================================================================
MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B}"
MODEL_PATH="${MODEL_PATH:-/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.5-27B}"

# =============================================================================
# Run identity
# =============================================================================
RUN_NAME="${RUN_NAME:-qwen35-27b-cuda-trloo-hfsdp8-pytorch-eager}"
REWARD_MANAGER=kernel_async
REWARD_FUNC_NAME="calculate_reward_speedup"
REFERENCE_BACKEND="${REFERENCE_BACKEND:-pytorch}"

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

# Match the hfsdp8 pytorch-eager baseline unless the caller overrides it.
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-50}"
NUM_WARMUP="${NUM_WARMUP:-30}"
PERF_TRIM_COUNT="${PERF_TRIM_COUNT:-5}"

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
# .18 smoke colocates Qwen3.5-27B actor and vLLM rollout on one 8xA800 node.
# Offload actor shards before vLLM wake_up so vLLM can remap weights cleanly.
ACTOR_PARAMETER_OFFLOAD="${ACTOR_PARAMETER_OFFLOAD:-True}"
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
# Infrastructure: default to the .18 single-node smoke topology, override for full runs.
# =============================================================================
NNODES="${NNODES:-${ARNOLD_WORKER_NUM:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-8}}"
SP_SIZE="${SP_SIZE:-4}"
FSDP_SIZE="${FSDP_SIZE:-8}"

ROLLOUT_GPU_MEMORY_UTIL="${ROLLOUT_GPU_MEMORY_UTIL:-0.75}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-2}"
# Keep per-engine active requests below the observed full-length KV concurrency
# for 10k prompt + 12k response requests on 2-way tensor parallel A800 rollout.
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
ROLLOUT_DISABLE_LOG_STATS="${ROLLOUT_DISABLE_LOG_STATS:-False}"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
KERNELGYM_VLLM_STATS_LOG_INTERVAL="${KERNELGYM_VLLM_STATS_LOG_INTERVAL:-10}"

# Gradient checkpointing: every layer (interval=2 OOMs on 80GB A800)
ENABLE_GRADIENT_CHECKPOINTING=True
export GRADIENT_CHECKPOINT_INTERVAL=1

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12000}"
PROMPT_OVERSAMPLING_FACTOR="${PROMPT_OVERSAMPLING_FACTOR:-1.7}"
SAMPLE_OVERSAMPLING_FACTOR="${SAMPLE_OVERSAMPLING_FACTOR:-1.0}"
SAMPLE_SELECTION_STRATEGY=efficiency_stochastic
MAX_SKIP_STEPS=5

APPLY_CHAT_TEMPLATE=True
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"

# Checkpoint save to log dir under a stable checkpoints/ root
HDFS_CHECKPOINT_PATH="${RUN_LOG_DIR}/checkpoints"

# Reference cache
REFERENCE_CACHE_ENABLE="${REFERENCE_CACHE_ENABLE:-true}"
REFERENCE_CACHE_AUTO_UUID="${REFERENCE_CACHE_AUTO_UUID:-true}"

# torch.compile crashes with current PyTorch version (CompiledFxGraph incompatibility)
ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-}"

# wandb 0.16.6 crashes with protobuf/numpy upgrades; use console unless explicitly overridden.
export WANDB_MODE="${WANDB_MODE:-disabled}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-['console']}"

source "${SCRIPT_DIR}/train_rl_common.sh"

main "$@"
