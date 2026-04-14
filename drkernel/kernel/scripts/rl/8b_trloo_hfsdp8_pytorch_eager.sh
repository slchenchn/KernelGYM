#!/bin/bash
# Compatibility wrapper.
# Selects a hardware-specific launcher from TRAIN_CLUSTER_PROFILE and NNODES
# so users do not need to keep editing tracked defaults per environment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORWARDED_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            [[ $# -ge 2 ]] || { echo "[ERROR] --profile requires a value" >&2; exit 1; }
            TRAIN_CLUSTER_PROFILE="$2"
            shift 2
            ;;
        --single-node)
            NNODES=1
            shift
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            shift
            ;;
    esac
done

source "${SCRIPT_DIR}/infra_common.sh"

if [[ -n "${TRAIN_LAUNCH_VARIANT:-}" ]]; then
    variant="${TRAIN_LAUNCH_VARIANT}"
else
    case "${TRAIN_CLUSTER_PROFILE}" in
        a800)
            variant="16xA800"
            ;;
        h20)
            if [[ "${NNODES:-1}" == "2" ]]; then
                variant="16xH20"
            else
                variant="8xH20"
            fi
            ;;
        *)
            echo "[ERROR] Unsupported TRAIN_CLUSTER_PROFILE '${TRAIN_CLUSTER_PROFILE}'" >&2
            exit 1
            ;;
    esac
fi

target="${SCRIPT_DIR}/8b_trloo_hfsdp8_pytorch_eager.${variant}.sh"
if [[ ! -f "${target}" ]]; then
    echo "[ERROR] Resolved launcher not found: ${target}" >&2
    exit 1
fi

exec bash "${target}" "${FORWARDED_ARGS[@]}"
