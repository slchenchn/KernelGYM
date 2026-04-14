#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

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

source "${REPO_ROOT}/drkernel/kernel/scripts/rl/infra_common.sh"

ENV_ACTIVATE_CMD="$(python_env_prelude)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            [[ $# -ge 2 ]] || { error "--profile requires a value"; exit 1; }
            TRAIN_CLUSTER_PROFILE="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'USAGE'
Usage:
  bash .agents/skills/stop_training/scripts/stop_ray_training.sh [--profile h20|a800]
USAGE
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

stop_node() {
    local runner="$1"
    local label="$2"

    echo "[INFO] Stopping ${label}"

    ${runner} \
        "${ENV_ACTIVATE_CMD} && ray stop --force || true; \
         sleep 2; \
         echo '--- GPU ---'; \
         nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; \
         echo '--- APPS ---'; \
         nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true"
}

stop_node run_on_train_head "head ($(train_head_label))"
stop_node run_on_train_worker "worker ($(train_worker_label))"

echo "[INFO] Remote training nodes have been stopped and checked."
