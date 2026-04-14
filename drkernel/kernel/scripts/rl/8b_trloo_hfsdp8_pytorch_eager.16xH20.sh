#!/bin/bash
# drkernel-8b TRLOO training with pytorch eager reference backend
# 16× H20 (2 nodes), 16× 4090 reward workers reached through the local relay
# FSDP-8 per node, DP=2 across nodes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_LOG_HW="${TRAIN_LOG_HW:-16xH20}"
export NNODES="${NNODES:-2}"

bash "${SCRIPT_DIR}/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh" "$@"
