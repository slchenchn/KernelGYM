#!/bin/bash
# Compatibility loader for reward/training infrastructure scripts.
# Shared helpers live in infra_lib.sh; cluster-specific settings live in infra_profiles/.

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SUDO_PW="csl"
REWARD_REPO_PATH="${REWARD_REPO_PATH:-/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018}"

# Reward nodes
REWARD_API_HOST="192.168.16.39"
REWARD_API_PORT="8111"
REDIS_PORT="8110"
REWARD_NODES=("192.168.16.39" "192.168.16.40")

CONTAINER_IMAGE="192.168.14.129:80/fm/llmc:v1.1"
TRAIN_CLUSTER_PROFILE="${TRAIN_CLUSTER_PROFILE:-h20}"

source "${INFRA_DIR}/infra_lib.sh"

profile_path="${INFRA_DIR}/infra_profiles/${TRAIN_CLUSTER_PROFILE}.sh"
if [[ ! -f "${profile_path}" ]]; then
    error "Unknown TRAIN_CLUSTER_PROFILE '${TRAIN_CLUSTER_PROFILE}' (expected profile file at ${profile_path})"
    return 1 2>/dev/null || exit 1
fi
source "${profile_path}"
