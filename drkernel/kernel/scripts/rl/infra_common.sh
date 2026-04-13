#!/bin/bash
# Common variables and helpers for reward/training infrastructure scripts

VLLM018_PATH="/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018"
VENV="${VLLM018_PATH}/.venv-vllm0180"
SUDO_PW="csl"

# Reward nodes
REWARD_API_HOST="192.168.16.39"
REWARD_API_PORT="8111"
REDIS_PORT="8110"
REWARD_NODES=("192.168.16.39" "192.168.16.40")

# Training nodes
HEAD_NODE="192.168.16.18"
HEAD_PORT=20629
WORKER_NODE="192.168.16.24"
WORKER_PORT=14218

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# _filter_sudo_noise — strips sudo password echo and setlocale warnings
# ---------------------------------------------------------------------------
_filter_sudo_noise() {
    sed '/setlocale/d; s/\[sudo\] password for [^:]*: //'
}

# ---------------------------------------------------------------------------
# run_on_reward HOST CMD_STRING
#   Runs a command on a reward node via ssh -> sudo -> bash -c.
#   The command is base64-encoded to avoid all quoting issues through
#   the ssh -> sudo -> bash chain.
#   Returns the real exit code from the remote side.
# ---------------------------------------------------------------------------
run_on_reward() {
    local host=$1; shift
    local cmd="$*"
    local encoded
    encoded=$(printf '%s' "$cmd" | base64 -w0)
    local output rc
    # The SSH string is double-quoted. ${SUDO_PW} and ${encoded} are expanded
    # locally. \$() is escaped so base64 -d runs on the remote side.
    output=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        chenshuailin@"${host}" \
        "echo ${SUDO_PW} | sudo -S bash -c \"\$(echo ${encoded} | base64 -d)\"" 2>&1)
    rc=$?
    echo "$output" | _filter_sudo_noise
    return $rc
}

# ---------------------------------------------------------------------------
# docker_exec HOST CONTAINER CMD_STRING
#   Runs CMD_STRING inside a docker container on a remote reward node via
#   ssh -> sudo -> docker exec -> bash -c.
#   The command is base64-encoded to survive all quoting layers.
#   Returns the real docker exec exit code.
# ---------------------------------------------------------------------------
docker_exec() {
    local host=$1; local container=$2; shift 2
    local cmd="$*"
    local encoded
    encoded=$(printf '%s' "$cmd" | base64 -w0)
    local output rc
    # Flow: local shell expands ${SUDO_PW}, ${container}, ${encoded}
    #       remote shell (via SSH) runs: echo pw | sudo -S docker exec CONTAINER bash -c "$(echo B64 | base64 -d)"
    #       remote shell expands $() which decodes the base64 into the original command
    #       docker exec runs: bash -c "ORIGINAL_CMD"
    output=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        chenshuailin@"${host}" \
        "echo ${SUDO_PW} | sudo -S docker exec ${container} bash -c \"\$(echo ${encoded} | base64 -d)\"" 2>&1)
    rc=$?
    echo "$output" | _filter_sudo_noise
    return $rc
}

# ---------------------------------------------------------------------------
# docker_exec_env HOST CONTAINER [EXTRA_FLAGS...] -- CMD_STRING
#   Runs CMD_STRING inside a docker container with:
#     -w /tmp, standard PYTHONPATH/REDIS/API env, detached (-d)
#   EXTRA_FLAGS are additional docker exec flags (e.g., -e FOO=bar).
#   Returns the real exit code.
# ---------------------------------------------------------------------------
docker_exec_env() {
    local host=$1; local container=$2; shift 2

    # Collect extra docker flags until we hit "--"
    local extra_flags=""
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do
        extra_flags="${extra_flags} $1"
        shift
    done
    [ "$1" = "--" ] && shift

    local cmd="$*"
    local encoded
    encoded=$(printf '%s' "$cmd" | base64 -w0)
    local output rc
    output=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        chenshuailin@"${host}" \
        "echo ${SUDO_PW} | sudo -S docker exec \
            -w /tmp \
            -e PYTHONPATH=${VLLM018_PATH} \
            -e REDIS_HOST=${REWARD_API_HOST} \
            -e REDIS_PORT=${REDIS_PORT} \
            -e REDIS_KEY_PREFIX=kernelgym_locked \
            -e API_HOST=${REWARD_API_HOST} \
            -e API_PORT=${REWARD_API_PORT} \
            ${extra_flags} \
            -d ${container} bash -c \"\$(echo ${encoded} | base64 -d)\"" 2>&1)
    rc=$?
    echo "$output" | _filter_sudo_noise
    return $rc
}

container_name() {
    local node=$1
    local suffix="${node##*.}"
    echo "kernelgym-reward-${suffix}"
}

CONTAINER_IMAGE="192.168.14.129:80/fm/llmc:v1.1"

# ---------------------------------------------------------------------------
# recreate_container NODE
#   Force-removes the old container and creates a fresh one.
#   Returns 0 on success, 1 on failure.
# ---------------------------------------------------------------------------
recreate_container() {
    local node=$1
    local container
    container=$(container_name "$node")
    info "Recreating ${container} on ${node}..."

    info "  Removing old container..."
    run_on_reward "$node" "docker rm -f ${container} 2>/dev/null || true"
    sleep 2

    info "  Creating new container..."
    local output rc
    output=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        chenshuailin@"${node}" \
        "echo ${SUDO_PW} | sudo -S docker run -d \
            --name ${container} \
            --privileged \
            --network host \
            --ipc private \
            --shm-size 256g \
            --gpus all \
            -v /nfs:/nfs \
            -w /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM \
            ${CONTAINER_IMAGE} \
            sleep infinity" 2>&1)
    rc=$?
    output=$(echo "$output" | _filter_sudo_noise)
    info "  docker run output: ${output}"
    if [ $rc -ne 0 ]; then
        error "  docker run failed (exit code ${rc})!"
        return 1
    fi
    info "  Container created successfully"
    return 0
}
