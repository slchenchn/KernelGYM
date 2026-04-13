#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/infra_common.sh"

usage() {
    cat <<'USAGE'
Usage:
  bash drkernel/kernel/scripts/rl/start_training.sh [options]

Options:
  -f, --force-reward           Restart reward infrastructure with start_reward.sh -f
      --skip-reward            Reuse the existing reward stack instead of restarting it
      --train-script PATH      Training launcher script to run
      --tmux-session NAME      Remote head-node tmux session name
      --local-log PATH         Remote head-node tee log path
      --workdir PATH           Remote working directory before launch
      --env KEY=VALUE          Extra environment assignment for the launcher (repeatable)
      --dry-run                Print the resolved launch plan without starting anything
  -h, --help                   Show this help

Defaults:
  --train-script  drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh
  --workdir       /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel
  --tmux-session  derived from the launcher basename
  --local-log     /tmp/<tmux-session>.log

Examples:
  bash drkernel/kernel/scripts/rl/start_training.sh \
    --train-script drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.sh

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

TRAIN_SCRIPT="$(resolve_path "$TRAIN_SCRIPT")"
WORKDIR="$(resolve_path "$WORKDIR")"

script_base="$(basename "$TRAIN_SCRIPT")"
script_name="$(sanitize_name "$script_base")"
TMUX_SESSION="${TMUX_SESSION:-train-${script_name}}"
LOCAL_LOG="${LOCAL_LOG:-/tmp/${TMUX_SESSION}.log}"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    error "Training script not found locally: $TRAIN_SCRIPT"
    exit 1
fi

launcher_env=(
    "RAY_ADDRESS=${HEAD_NODE}:6379"
    "GLOO_SOCKET_IFNAME=ens22f0"
    "NCCL_SOCKET_IFNAME=ens22f0"
    "NCCL_NET=Socket"
    "NCCL_IB_DISABLE=1"
    "NCCL_SOCKET_FAMILY=AF_INET"
    "NCCL_DEBUG=WARN"
)
launcher_env+=("${EXTRA_ENV[@]}")

launch_cmd="source $(shell_quote "${VENV}/bin/activate") && cd $(shell_quote "$WORKDIR") &&"
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

info "Starting Ray head on ${HEAD_NODE}..."
head_output=$(ssh -o ConnectTimeout=10 -p ${HEAD_PORT} root@${HEAD_NODE} \
    "source ${VENV}/bin/activate && ray stop --force >/dev/null 2>&1 || true && ray start --head --port=6379 --num-gpus=8 --dashboard-host=0.0.0.0" 2>&1)
head_rc=$?
echo "$head_output" | tail -5
if [[ $head_rc -ne 0 ]]; then
    error "Ray head start failed (rc=${head_rc})!"
    echo "$head_output"
    exit 1
fi

info "Starting Ray worker on ${WORKER_NODE}..."
worker_output=$(ssh -o ConnectTimeout=10 -p ${WORKER_PORT} root@${WORKER_NODE} \
    "source ${VENV}/bin/activate && ray stop --force >/dev/null 2>&1 || true && ray start --address=${HEAD_NODE}:6379 --num-gpus=8" 2>&1)
worker_rc=$?
echo "$worker_output" | tail -5
if [[ $worker_rc -ne 0 ]]; then
    error "Ray worker start failed (rc=${worker_rc})!"
    echo "$worker_output"
    exit 1
fi

sleep 3
GPU_COUNT=$(ssh -o ConnectTimeout=10 -p ${HEAD_PORT} root@${HEAD_NODE} \
    "source ${VENV}/bin/activate && ray status 2>/dev/null | grep GPU || true" 2>&1)
info "Ray cluster: ${GPU_COUNT}"

echo ""
info "=== Step 3: Launching training ==="

ssh -o ConnectTimeout=10 -p ${HEAD_PORT} root@${HEAD_NODE} "test -f $(shell_quote "$TRAIN_SCRIPT")" 2>/dev/null
if [[ $? -ne 0 ]]; then
    error "Training script not found on ${HEAD_NODE}: ${TRAIN_SCRIPT}"
    exit 1
fi

ssh -o ConnectTimeout=10 -p ${HEAD_PORT} root@${HEAD_NODE} \
    "tmux kill-session -t $(shell_quote "$TMUX_SESSION") 2>/dev/null || true; \
     rm -f $(shell_quote "$LOCAL_LOG"); \
     tmux new-session -d -s $(shell_quote "$TMUX_SESSION") bash -lc ${launch_cmd_quoted}" 2>&1

tmux_rc=$?
if [[ $tmux_rc -ne 0 ]]; then
    error "Failed to launch training tmux session (rc=${tmux_rc})!"
    exit 1
fi

info "Training launched in tmux session '${TMUX_SESSION}' on ${HEAD_NODE}"
info ""
info "Monitor:"
info "  ssh -p ${HEAD_PORT} root@${HEAD_NODE} 'tail -f ${LOCAL_LOG}'"
info ""
info "Reward health:"
info "  curl http://${REWARD_API_HOST}:${REWARD_API_PORT}/health | python3 -m json.tool"
