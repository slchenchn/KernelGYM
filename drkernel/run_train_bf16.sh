#!/usr/bin/env bash
# BF16 training with optimized reward (warm spare pool + disk-backed ref cache)
set -euo pipefail
source /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/.venv-vllm0180/bin/activate
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel

export WANDB_DISABLED=true
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export TRAINER_LOGGERS='[console]'

export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export RUN_NAME=trloo-14b-bf16-spare2-refcache
export LOG_RUN_PREFIX=trloo-14b
export RUN_LOG_PHASE=bf16.spare2.refcache

export MODEL_PATH=/nfs/FM/chenshuailin/checkpoints/hkust-nlp/drkernel-14b-coldstart
export MAX_PROMPT_LENGTH=10240
export MAX_RESPONSE_LENGTH=8192
export ROLLOUT_GPU_MEMORY_UTIL=0.75
export ENFORCE_EAGER=False

export KERNELGYM_SERVER_URL=http://192.168.16.39:8111
export ARNOLD_WORKER_GPU=8
export ARNOLD_WORKER_NUM=1
export N_GPUS_PER_NODE=8
export GPUS_PER_NODE=8
export NNODES=1

bash /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/14b_coldstart_trloo_mrs_pr_prs.sh --sp_size 4
