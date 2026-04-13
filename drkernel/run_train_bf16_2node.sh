#!/usr/bin/env bash
# BF16 training on 2 nodes (16 GPUs) with HYBRID_SHARD FSDP
# Rollout: 16 independent vLLM engines (no cross-node GPU comm)
# Training: FSDP shards within each 8-GPU node, gradient all-reduce across nodes
set -euo pipefail
source /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/.venv-vllm0180/bin/activate
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel

export WANDB_DISABLED=true
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export TRAINER_LOGGERS='[console]'

export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export RUN_NAME=trloo-14b-bf16-2node-hybrid
export LOG_RUN_PREFIX=trloo-14b
export RUN_LOG_PHASE=bf16.2node.hybrid

export MODEL_PATH=/nfs/FM/chenshuailin/checkpoints/hkust-nlp/drkernel-14b-coldstart
export MAX_PROMPT_LENGTH=10240
export MAX_RESPONSE_LENGTH=8192
export PPO_MICRO_TOKEN=${PPO_MICRO_TOKEN:-12288}
export ROLLOUT_GPU_MEMORY_UTIL=0.75
export ENFORCE_EAGER=${ENFORCE_EAGER:-False}

export KERNELGYM_SERVER_URL=http://192.168.16.39:8111

# 2-node config
export ARNOLD_WORKER_GPU=8
export ARNOLD_WORKER_NUM=2
export N_GPUS_PER_NODE=8
export GPUS_PER_NODE=8
export NNODES=2

# HYBRID_SHARD: shard within 8 GPUs (intra-node), replicate across nodes
export FSDP_SIZE=8
# Halve cross-node gradient traffic by all-reducing in bf16 instead of fp32
export FSDP_REDUCE_DTYPE=bf16

# Attach to an existing 2-node Ray cluster unless explicitly overridden.
export RAY_ADDRESS=${RAY_ADDRESS:-auto}

# Force Gloo and NCCL to use the ethernet path between nodes.
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ens22f0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens22f0}
export NCCL_NET=${NCCL_NET:-Socket}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Disable AllSpark kernel for W8A16 — its process_weights_after_loading
# calls a CUDA-only repack op that fails in EngineCore subprocess init.
export VLLM_DISABLED_KERNELS=${VLLM_DISABLED_KERNELS:-AllSparkLinearKernel}

bash /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/14b_coldstart_trloo_mrs_pr_prs.sh --sp_size 4
