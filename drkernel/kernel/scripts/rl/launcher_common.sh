#!/bin/bash

# Shared launcher bootstrap helpers.
# Keep experiment/model/topology config in the top-level launcher files.

init_launcher_run_state() {
    PROJECT_NAME="${PROJECT_NAME:-drkernel}"
    RUN_LOG_TIMESTAMP="${RUN_LOG_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
    RUN_LOG_PHASE="${RUN_LOG_PHASE:-run}"
    RUN_LOG_BASENAME="${RUN_LOG_BASENAME:-${LOG_RUN_PREFIX}.train.${TRAIN_LOG_HW}.reward.${REWARD_LOG_HW}.${RUN_LOG_PHASE}.${RUN_LOG_TIMESTAMP}}"
    RUN_LOG_DIR="${RUN_LOG_DIR:-${DRKERNEL_ROOT}/logs/${RUN_LOG_BASENAME}}"
    MAIN_LOG="${MAIN_LOG:-${RUN_LOG_DIR}/main.log}"
    TRAINER_LOG="${TRAINER_LOG:-${RUN_LOG_DIR}/trainer.log}"
    ROLLOUT_LOG="${ROLLOUT_LOG:-${RUN_LOG_DIR}/rollout.log}"
    REWARD_LOG="${REWARD_LOG:-${RUN_LOG_DIR}/reward.log}"
    VLLM_LOG="${VLLM_LOG:-${RUN_LOG_DIR}/vllm.log}"
    TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUN_LOG_DIR}/torchinductor_cache}"

    export PROJECT_NAME
    export RUN_LOG_TIMESTAMP
    export RUN_LOG_PHASE
    export RUN_LOG_BASENAME
    export RUN_LOG_DIR
    export MAIN_LOG
    export TRAINER_LOG
    export ROLLOUT_LOG
    export REWARD_LOG
    export VLLM_LOG
    export TORCHINDUCTOR_CACHE_DIR
}

enable_launcher_log_router() {
    if [ "${DRKERNEL_LOGGING_INITIALIZED:-0}" != "1" ]; then
        mkdir -p "${RUN_LOG_DIR}/structured" "${TORCHINDUCTOR_CACHE_DIR}"
        export DRKERNEL_LOGGING_INITIALIZED=1
        export DRKERNEL_EVENT_LOG_DIR="${RUN_LOG_DIR}/structured"
        echo "Logging training run to: ${MAIN_LOG}"
        exec > >(python -u "${SCRIPT_DIR}/log_router.py") 2>&1
    fi
}

enter_drkernel_root() {
    cd "${DRKERNEL_ROOT}"
}
