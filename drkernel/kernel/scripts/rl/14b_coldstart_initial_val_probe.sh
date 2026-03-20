#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PWD=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM/drkernel
LOG_DIR="${PWD}/logs/init_val"
PROBE_LOG="${PROBE_LOG:-${LOG_DIR}/$(date +%Y%m%d-%H%M%S).log}"

cd "${PWD}"
mkdir -p "${LOG_DIR}"

echo "Logging initial val probe to: ${PROBE_LOG}"
exec > >(tee -a "${PROBE_LOG}") 2>&1

TRAIN_DATASET=${PWD}/data/drkernel-rl-data/cuda_llm_rl_thinking_1025.parquet
VALID_DATASET=${PWD}/data/drkernel-validation-data/validation_data_thinking.parquet

KERNELGYM_SERVER_URL="${KERNELGYM_SERVER_URL:-"http://192.168.16.23:8001"}"
MODEL_NAME="${MODEL_NAME:-hkust-nlp/drkernel-14b-coldstart}"
MODEL_PATH="${MODEL_PATH:-/nfs/FM/chenshuailin/checkpoints/hkust-nlp/drkernel-14b-coldstart}"

RUN_NAME="${RUN_NAME:-drkernel-14b-coldstart-initial-val-probe}"
REWARD_MANAGER=kernel_async
REWARD_FUNC_NAME="calculate_reward_speedup"
REFERENCE_BACKEND="torch_compile"

ALGORITHM="trloo"
VAL_ONLY=True
VAL_BEFORE_TRAIN=True
VAL_SAMPLE_SIZE="${VAL_SAMPLE_SIZE:-10}"
N_VAL="${N_VAL:-8}"
ENABLE_MULTI_TURN=True
MAX_TURN=3
VAL_MAX_TURN="${VAL_MAX_TURN:-$MAX_TURN}"

SPEEDUP_REWARD_UPPER_BOUND=3.0
SPEEDUP_REWARD_LOWER_BOUND=0.0
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-100}"
REWARD_TASK_TIMEOUT=300
REWARD_TIMEOUT=1800
REWARD_ACQUIRE_TIMEOUT=2400
REWARD_MAX_CONCURRENT=32
REWARD_MAX_RETRIES=3
REWARD_PRINT_STATUS=True
REWARD_TASK_TIMEOUT_CLIENT=2400

# Unused for this probe because VAL_ONLY=True.
# TRAIN_BATCH_SIZE=16
# PPO_MINI_BATCH_SIZE=16
# LEARNING_RATE=1e-6
# ROLLOUT_N=16
# TOTAL_EPOCHS=1
# SAVE_FREQ=1000000
# TEST_FREQ=1000000
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
SP_SIZE=4
APPLY_CHAT_TEMPLATE=True
FREE_CACHE_ENGINE=True
ENFORCE_EAGER=False
ROLLOUT_GPU_MEMORY_UTIL=0.75
MAX_PROMPT_LENGTH=10240
MAX_RESPONSE_LENGTH=8192
# Unused for this probe because VAL_ONLY=True.
# PROMPT_OVERSAMPLING_FACTOR=1.0
# SAMPLE_OVERSAMPLING_FACTOR=1.0
# SAMPLE_SELECTION_STRATEGY=efficiency_stochastic
# MAX_SKIP_STEPS=1
TRAINER_LOGGERS="${TRAINER_LOGGERS:-['console']}"

NNODES=$ARNOLD_WORKER_NUM
GPUS_PER_NODE=$ARNOLD_WORKER_GPU
if [ -z "$ARNOLD_WORKER_GPU" ]; then
    GPUS_PER_NODE=8
fi

source "${SCRIPT_DIR}/train_rl_common.sh"

main "$@"
