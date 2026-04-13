#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${REPO_ROOT}/drkernel/kernel/scripts/rl/infra_common.sh"

stop_node() {
    local host="$1"
    local port="$2"
    local label="$3"

    echo "[INFO] Stopping ${label} (${host}:${port})"

    ssh -n -p "${port}" "root@${host}" \
        "${VENV}/bin/ray stop --force || true; \
         for p in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do \
             kill -9 \$p 2>/dev/null || true; \
         done; \
         sleep 2; \
         echo '--- GPU ---'; \
         nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; \
         echo '--- APPS ---'; \
         nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true"
}

stop_node "${HEAD_NODE}" "${HEAD_PORT}" "head"
stop_node "${WORKER_NODE}" "${WORKER_PORT}" "worker"

echo "[INFO] Remote training nodes have been stopped and checked."
