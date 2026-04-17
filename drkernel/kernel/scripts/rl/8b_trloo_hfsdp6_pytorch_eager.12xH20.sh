#!/bin/bash
# drkernel-8b TRLOO training with pytorch eager reference backend
# 12× H20 (2 nodes), using GPUs 0-5 on each node
# FSDP-6 per node, DP=2 across nodes, SP=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOG_RUN_PREFIX="${LOG_RUN_PREFIX:-trloo-8b-hfsdp6-pytorch-eager}"
export RUN_NAME="${RUN_NAME:-drkernel-8b-coldstart-trloo-hfsdp6-pytorch-eager}"
export TRAIN_LOG_HW="${TRAIN_LOG_HW:-12xH20}"
export NNODES="${NNODES:-2}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-6}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-6}"
export FSDP_SIZE="${FSDP_SIZE:-6}"
export KERNELGYM_SERVER_URL="${KERNELGYM_SERVER_URL:-http://10.0.18.3:18112}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export PROMPT_OVERSAMPLING_FACTOR="${PROMPT_OVERSAMPLING_FACTOR:-2.0}"
export SAMPLE_OVERSAMPLING_FACTOR="${SAMPLE_OVERSAMPLING_FACTOR:-1.0}"

bash "${SCRIPT_DIR}/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh" "$@"
