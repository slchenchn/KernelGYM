#!/usr/bin/env bash
# 1-node val-only for timing config comparison with locked GPU clocks
# Usage: KERNELGYM_SERVER_URL=http://... MODEL_PATH=... bash run_val_timing_test.sh
set -euo pipefail
source /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/.venv-vllm0180/bin/activate
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel

export WANDB_DISABLED=true
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export TRAINER_LOGGERS='[console]'

# Val only — no training
export VAL_ONLY=True
export VAL_BEFORE_TRAIN=True
export MAX_TURN=${MAX_TURN:-1}

# Model — use drkernel-14b (not coldstart)
export MODEL_PATH=${MODEL_PATH:-/nfs/FM/chenshuailin/checkpoints/hkust-nlp/drkernel-14b}
export RUN_NAME=${RUN_NAME:-val-timing-test}
export LOG_RUN_PREFIX=${LOG_RUN_PREFIX:-val-timing}
export RUN_LOG_PHASE=${RUN_LOG_PHASE:-locked-clk}

export MAX_PROMPT_LENGTH=10240
export MAX_RESPONSE_LENGTH=8192
export PPO_MICRO_TOKEN=${PPO_MICRO_TOKEN:-12288}
export ROLLOUT_GPU_MEMORY_UTIL=0.75
export ENFORCE_EAGER=${ENFORCE_EAGER:-False}
export KERNELGYM_SERVER_URL=${KERNELGYM_SERVER_URL:?"Must set KERNELGYM_SERVER_URL"}

# 2-node config
export ARNOLD_WORKER_GPU=8
export ARNOLD_WORKER_NUM=2
export N_GPUS_PER_NODE=8
export GPUS_PER_NODE=8
export NNODES=2
export FSDP_SIZE=8
export FSDP_REDUCE_DTYPE=bf16

# Attach to existing 2-node Ray cluster
export RAY_ADDRESS=${RAY_ADDRESS:-auto}

export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ens22f0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens22f0}
export NCCL_NET=${NCCL_NET:-Socket}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export VLLM_DISABLED_KERNELS=${VLLM_DISABLED_KERNELS:-AllSparkLinearKernel}

# Disable reference cache — we want fresh timing every time
export REFERENCE_CACHE_ENABLE=False

bash /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/14b_coldstart_trloo_mrs_pr_prs.sh --sp_size 4
