#!/bin/bash

# Shared helper functions for reward/training infrastructure scripts.

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

python_env_prelude() {
    if [[ -n "${PYTHON_ENV_ACTIVATE:-}" ]]; then
        printf '%s' "${PYTHON_ENV_ACTIVATE}"
    elif [[ -f "${VENV}/bin/activate" ]]; then
        printf 'source %q' "${VENV}/bin/activate"
    else
        printf ':'
    fi
}

_run_on_train_target() {
    local mode=$1
    local ssh_target=$2
    local port=$3
    local container=$4
    shift 4
    local cmd="$*"

    if [[ "$mode" = "local" ]]; then
        bash -lc "$cmd"
        return $?
    fi

    local encoded
    encoded=$(printf '%s' "$cmd" | base64 -w0)
    local ssh_opts=(-o ConnectTimeout=10 -o StrictHostKeyChecking=no)
    if [[ -n "$port" && "$port" != "22" ]]; then
        ssh_opts+=(-p "$port")
    fi

    case "$mode" in
        ssh)
            ssh "${ssh_opts[@]}" "$ssh_target" \
                "bash -lc \"\$(echo ${encoded} | base64 -d)\""
            ;;
        ssh-docker)
            ssh "${ssh_opts[@]}" "$ssh_target" \
                "docker exec ${container} bash -lc \"\$(echo ${encoded} | base64 -d)\""
            ;;
        *)
            error "Unsupported train target mode: ${mode}"
            return 1
            ;;
    esac
}

run_on_train_head() {
    _run_on_train_target \
        "${TRAIN_HEAD_MODE}" \
        "${TRAIN_HEAD_SSH_TARGET}" \
        "${HEAD_PORT}" \
        "${TRAIN_HEAD_CONTAINER}" \
        "$*"
}

run_on_train_worker() {
    _run_on_train_target \
        "${TRAIN_WORKER_MODE}" \
        "${TRAIN_WORKER_SSH_TARGET}" \
        "${WORKER_PORT}" \
        "${TRAIN_WORKER_CONTAINER}" \
        "$*"
}

train_head_label() {
    if [[ "${TRAIN_HEAD_MODE}" = "local" ]]; then
        printf 'local-container@%s' "${HEAD_NODE}"
    elif [[ "${TRAIN_HEAD_MODE}" = "ssh-docker" ]]; then
        printf '%s/%s' "${TRAIN_HEAD_SSH_TARGET}" "${TRAIN_HEAD_CONTAINER}"
    else
        printf '%s' "${TRAIN_HEAD_SSH_TARGET}"
    fi
}

train_worker_label() {
    if [[ "${TRAIN_WORKER_MODE}" = "local" ]]; then
        printf 'local-container@%s' "${WORKER_NODE}"
    elif [[ "${TRAIN_WORKER_MODE}" = "ssh-docker" ]]; then
        printf '%s/%s' "${TRAIN_WORKER_SSH_TARGET}" "${TRAIN_WORKER_CONTAINER}"
    else
        printf '%s' "${TRAIN_WORKER_SSH_TARGET}"
    fi
}

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
            -e PYTHONPATH=${REWARD_REPO_PATH} \
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
