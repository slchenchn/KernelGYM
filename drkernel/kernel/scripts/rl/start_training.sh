#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--profile" ]]; then
        next=$((i + 1))
        if (( next > $# )); then
            echo "[ERROR] --profile requires a value" >&2
            exit 1
        fi
        TRAIN_CLUSTER_PROFILE="${!next}"
    fi
done

source "${SCRIPT_DIR}/infra_common.sh"

usage() {
    cat <<'USAGE'
Usage:
  bash drkernel/kernel/scripts/rl/start_training.sh [options]

Options:
  -f, --force-reward           Restart reward infrastructure with start_reward.sh -f
      --skip-reward            Reuse the existing reward stack instead of restarting it
      --train-script PATH      Training launcher script to run
      --profile NAME          Training cluster profile to load (e.g. h20, a800)
      --single-node            Launch only on the head/current training node
      --tmux-session NAME      Remote head-node tmux session name
      --local-log PATH         Remote head-node tee log path
      --workdir PATH           Remote working directory before launch
      --env KEY=VALUE          Extra environment assignment for the launcher (repeatable)
      --dry-run                Print the resolved launch plan without starting anything
  -h, --help                   Show this help

Defaults:
  --train-script  drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh
  --profile       TRAIN_CLUSTER_PROFILE / .infra_profile.local.sh / fallback h20
  --workdir       <VLLM018_PATH>/drkernel from the selected profile
  --tmux-session  derived from the launcher basename
  --local-log     /tmp/<tmux-session>.log

Examples:
  bash drkernel/kernel/scripts/rl/start_training.sh \
    --train-script drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.16xA800.sh

  bash drkernel/kernel/scripts/rl/start_training.sh \
    --train-script drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh \
    --skip-reward \
    --env RUN_LOG_DIR=/nfs/.../trloo-14b-hfsdp8-pytorch-eager.train...20260409-092519 \
    --env REWARD_TASK_TIMEOUT=30
USAGE
}

FORCE_REWARD=0
SKIP_REWARD=0
DRY_RUN=0
SINGLE_NODE=0
TRAIN_SCRIPT="${SCRIPT_DIR}/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh"
TMUX_SESSION=""
LOCAL_LOG=""
WORKDIR="${VLLM018_PATH}/drkernel"
declare -a EXTRA_ENV=()

resolve_path() {
    local candidate="$1"
    if [[ "$candidate" = /* ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    if [[ -f "$candidate" ]]; then
        python - <<'PY' "$candidate"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
        return 0
    fi
    if [[ -f "${VLLM018_PATH}/$candidate" ]]; then
        python - <<'PY' "${VLLM018_PATH}/$candidate"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
        return 0
    fi
    printf '%s\n' "$candidate"
}

sanitize_name() {
    local raw="$1"
    raw="${raw%.sh}"
    raw="${raw//[^A-Za-z0-9._-]/-}"
    printf '%s\n' "$raw"
}

shell_quote() {
    printf '%q' "$1"
}

get_env_override() {
    local key="$1"
    local kv
    for kv in "${EXTRA_ENV[@]}"; do
        if [[ "$kv" == "${key}="* ]]; then
            printf '%s\n' "${kv#*=}"
            return 0
        fi
    done
    return 1
}

wait_for_ray_ready() {
    local label="$1"
    local runner="$2"
    local address="$3"
    local attempts="${4:-30}"
    local sleep_s="${5:-2}"
    local cmd="${ENV_ACTIVATE_CMD} && python -c 'import socket,sys; s=socket.socket(); s.settimeout(2); s.connect((sys.argv[1], int(sys.argv[2]))); s.close()' $(shell_quote "${HEAD_NODE}") $(shell_quote "${RAY_HEAD_PORT}")"

    local i
    for ((i=1; i<=attempts; i++)); do
        if ${runner} "${cmd}" >/dev/null 2>&1; then
            if ${runner} "${ENV_ACTIVATE_CMD} && ray status --address=${address}" >/dev/null 2>&1; then
                info "${label} Ray readiness check passed (${i}/${attempts})"
                return 0
            fi
        fi
        sleep "${sleep_s}"
    done

    error "${label} Ray readiness check failed after ${attempts} attempts"
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--force-reward)
            FORCE_REWARD=1
            shift
            ;;
        --skip-reward)
            SKIP_REWARD=1
            shift
            ;;
        --train-script)
            [[ $# -ge 2 ]] || { error "--train-script requires a value"; exit 1; }
            TRAIN_SCRIPT="$2"
            shift 2
            ;;
        --profile)
            [[ $# -ge 2 ]] || { error "--profile requires a value"; exit 1; }
            TRAIN_CLUSTER_PROFILE="$2"
            shift 2
            ;;
        --single-node)
            SINGLE_NODE=1
            shift
            ;;
        --tmux-session)
            [[ $# -ge 2 ]] || { error "--tmux-session requires a value"; exit 1; }
            TMUX_SESSION="$2"
            shift 2
            ;;
        --local-log)
            [[ $# -ge 2 ]] || { error "--local-log requires a value"; exit 1; }
            LOCAL_LOG="$2"
            shift 2
            ;;
        --workdir)
            [[ $# -ge 2 ]] || { error "--workdir requires a value"; exit 1; }
            WORKDIR="$2"
            shift 2
            ;;
        --env)
            [[ $# -ge 2 ]] || { error "--env requires KEY=VALUE"; exit 1; }
            [[ "$2" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { error "Invalid --env assignment: $2"; exit 1; }
            EXTRA_ENV+=("$2")
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ "$SKIP_REWARD" = "1" && "$FORCE_REWARD" = "1" ]]; then
    error "--skip-reward and --force-reward cannot be used together"
    exit 1
fi

requested_nnodes="$(get_env_override NNODES || true)"
if [[ "$SINGLE_NODE" = "1" ]]; then
    if [[ -n "${requested_nnodes}" && "${requested_nnodes}" != "1" ]]; then
        error "--single-node conflicts with NNODES=${requested_nnodes}"
        exit 1
    fi
    if [[ -z "${requested_nnodes}" ]]; then
        EXTRA_ENV+=("NNODES=1")
        requested_nnodes="1"
    fi
fi
if [[ "${requested_nnodes}" == "1" ]]; then
    SINGLE_NODE=1
fi

TRAIN_SCRIPT="$(resolve_path "$TRAIN_SCRIPT")"
WORKDIR="$(resolve_path "$WORKDIR")"
ENV_ACTIVATE_CMD="$(python_env_prelude)"
resolved_script_base="$(basename "$TRAIN_SCRIPT")"

if [[ "$SINGLE_NODE" = "1" && "${TRAIN_CLUSTER_PROFILE}" = "a800" && "${resolved_script_base}" = "8b_trloo_hfsdp8_pytorch_eager.sh" ]]; then
    error "--single-node is not supported for ${resolved_script_base} with TRAIN_CLUSTER_PROFILE=a800; use the 16xA800 two-node launcher/profile"
    exit 1
fi

script_base="${resolved_script_base}"
script_name="$(sanitize_name "$script_base")"
TMUX_SESSION="${TMUX_SESSION:-train-${script_name}}"
LOCAL_LOG="${LOCAL_LOG:-/tmp/${TMUX_SESSION}.log}"
HEAD_RAY_LOG="/tmp/ray-head-${script_name}.log"
WORKER_RAY_LOG="/tmp/ray-worker-${script_name}.log"
HEAD_RAY_SESSION="ray-head-${script_name}"
WORKER_RAY_SESSION="ray-worker-${script_name}"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    error "Training script not found locally: $TRAIN_SCRIPT"
    exit 1
fi

launcher_env=(
    "TRAIN_CLUSTER_PROFILE=${TRAIN_CLUSTER_PROFILE}"
    "RAY_ADDRESS=${HEAD_NODE}:${RAY_HEAD_PORT}"
    "GLOO_SOCKET_IFNAME=${TRAIN_GLOO_SOCKET_IFNAME}"
    "NCCL_SOCKET_IFNAME=${TRAIN_NCCL_SOCKET_IFNAME}"
    "NCCL_IB_DISABLE=${TRAIN_NCCL_IB_DISABLE}"
    "NCCL_SOCKET_FAMILY=${TRAIN_NCCL_SOCKET_FAMILY}"
    "NCCL_DEBUG=${TRAIN_NCCL_DEBUG}"
)
if [[ -n "${TRAIN_NCCL_NET}" ]]; then
    launcher_env+=("NCCL_NET=${TRAIN_NCCL_NET}")
fi
if [[ -n "${TRAIN_NCCL_IB_HCA}" ]]; then
    launcher_env+=("NCCL_IB_HCA=${TRAIN_NCCL_IB_HCA}")
fi
launcher_env+=("${EXTRA_ENV[@]}")

launch_cmd="${ENV_ACTIVATE_CMD} && cd $(shell_quote "$WORKDIR") &&"
for kv in "${launcher_env[@]}"; do
    launch_cmd+=" $(shell_quote "$kv")"
done
launch_cmd+=" bash $(shell_quote "$TRAIN_SCRIPT") 2>&1 | tee $(shell_quote "$LOCAL_LOG")"
launch_cmd_quoted="$(shell_quote "$launch_cmd")"

if [[ "$DRY_RUN" = "1" ]]; then
    info "Dry run only; no processes will be started."
    if [[ "$SKIP_REWARD" = "1" ]]; then
        info "Reward step: skip existing reward stack"
    elif [[ "$FORCE_REWARD" = "1" ]]; then
        info "Reward step: restart reward stack with -f"
    else
        info "Reward step: normal reward startup"
    fi
    info "Training script: $TRAIN_SCRIPT"
    info "Cluster profile: ${TRAIN_CLUSTER_PROFILE}"
    if [[ "$SINGLE_NODE" = "1" ]]; then
        info "Topology: single node (${HEAD_NODE})"
    else
        info "Topology: head + worker (${HEAD_NODE} + ${WORKER_NODE})"
    fi
    info "Workdir: $WORKDIR"
    info "tmux session: $TMUX_SESSION"
    info "local log: $LOCAL_LOG"
    info "Extra launcher env: ${launcher_env[*]}"
    info "Resolved launch command:"
    printf '%s\n' "$launch_cmd"
    exit 0
fi

if [[ "$SKIP_REWARD" != "1" ]]; then
    force_flag=""
    if [[ "$FORCE_REWARD" = "1" ]]; then
        force_flag="-f"
    fi
    echo ""
    info "=== Step 1: Starting reward infrastructure ==="
    bash "${SCRIPT_DIR}/start_reward.sh" ${force_flag}
    reward_rc=$?
    if [[ $reward_rc -ne 0 ]]; then
        error "start_reward.sh failed (exit code ${reward_rc})!"
        exit 1
    fi
    info "Reward infrastructure is up."
else
    echo ""
    info "=== Step 1: Reusing existing reward infrastructure ==="
fi

echo ""
info "=== Step 2: Starting Ray cluster ==="

info "Starting Ray head on $(train_head_label)..."
head_start_cmd="${ENV_ACTIVATE_CMD} && cd $(shell_quote "$WORKDIR") && ray stop --force >/dev/null 2>&1 || true && ray start --head --node-ip-address=${HEAD_NODE} --port=${RAY_HEAD_PORT} --num-gpus=8 --dashboard-host=0.0.0.0 >$(shell_quote "${HEAD_RAY_LOG}") 2>&1 && tail -f /dev/null"
head_output=$(run_on_train_head \
    "tmux kill-session -t $(shell_quote "${HEAD_RAY_SESSION}") 2>/dev/null || true; tmux new-session -d -s $(shell_quote "${HEAD_RAY_SESSION}") bash -lc $(shell_quote "${head_start_cmd}")" 2>&1)
head_rc=$?
[[ -n "$head_output" ]] && echo "$head_output" | tail -5
if [[ $head_rc -ne 0 ]]; then
    error "Ray head start failed (rc=${head_rc})!"
    echo "$head_output"
    exit 1
fi

if [[ "$SINGLE_NODE" != "1" ]]; then
    info "Starting Ray worker on $(train_worker_label)..."
    worker_start_cmd="${ENV_ACTIVATE_CMD} && cd $(shell_quote "$WORKDIR") && ray stop --force >/dev/null 2>&1 || true && ray start --address=${HEAD_NODE}:${RAY_HEAD_PORT} --num-gpus=8 >$(shell_quote "${WORKER_RAY_LOG}") 2>&1 && tail -f /dev/null"
    worker_output=$(run_on_train_worker \
        "tmux kill-session -t $(shell_quote "${WORKER_RAY_SESSION}") 2>/dev/null || true; tmux new-session -d -s $(shell_quote "${WORKER_RAY_SESSION}") bash -lc $(shell_quote "${worker_start_cmd}")" 2>&1)
    worker_rc=$?
    [[ -n "$worker_output" ]] && echo "$worker_output" | tail -5
    if [[ $worker_rc -ne 0 ]]; then
        error "Ray worker start failed (rc=${worker_rc})!"
        echo "$worker_output"
        exit 1
    fi
else
    info "Single-node mode requested; skipping worker Ray startup."
fi

sleep 3
wait_for_ray_ready "Head" run_on_train_head "${HEAD_NODE}:${RAY_HEAD_PORT}" || {
    run_on_train_head "tail -n 80 $(shell_quote "${HEAD_RAY_LOG}")" || true
    exit 1
}
if [[ "$SINGLE_NODE" != "1" ]]; then
    wait_for_ray_ready "Worker" run_on_train_worker "${HEAD_NODE}:${RAY_HEAD_PORT}" || {
        run_on_train_worker "tail -n 80 $(shell_quote "${WORKER_RAY_LOG}")" || true
        exit 1
    }
fi
GPU_COUNT=$(run_on_train_head \
    "${ENV_ACTIVATE_CMD} && ray status --address=${HEAD_NODE}:${RAY_HEAD_PORT} 2>/dev/null | grep GPU || true" 2>&1)
info "Ray cluster: ${GPU_COUNT}"

echo ""
info "=== Step 3: Launching training ==="

if ! run_on_train_head "test -f $(shell_quote "$TRAIN_SCRIPT")" >/dev/null 2>&1; then
    error "Training script not found on ${HEAD_NODE}: ${TRAIN_SCRIPT}"
    exit 1
fi

run_on_train_head \
    "tmux kill-session -t $(shell_quote "$TMUX_SESSION") 2>/dev/null || true; \
     rm -f $(shell_quote "$LOCAL_LOG"); \
     tmux new-session -d -s $(shell_quote "$TMUX_SESSION") bash -lc ${launch_cmd_quoted}" 2>&1

tmux_rc=$?
if [[ $tmux_rc -ne 0 ]]; then
    error "Failed to launch training tmux session (rc=${tmux_rc})!"
    exit 1
fi

info "Training launched in tmux session '${TMUX_SESSION}' on $(train_head_label)"
info ""
info "Monitor:"
info "  ssh -p ${HEAD_PORT} root@${HEAD_NODE} 'tail -f ${LOCAL_LOG}'"
info ""
info "Reward health:"
info "  curl http://${REWARD_API_HOST}:${REWARD_API_PORT}/health | python3 -m json.tool"
