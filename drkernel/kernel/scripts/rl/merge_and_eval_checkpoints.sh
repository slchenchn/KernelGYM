#!/bin/bash
# Merge FSDP checkpoints to HF format and eval them.
#
# Usage:
#   bash merge_and_eval_checkpoints.sh
# If EVAL_STEPS is unset, the script auto-detects untested checkpoints under
# CKPT_BASE by scanning global_step_* directories and skipping any step that
# already has eval_results/step_*/metrics.json.

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
TRAIN_CLUSTER_PROFILE="${TRAIN_CLUSTER_PROFILE:-a800}"
source "${SCRIPT_DIR}/infra_common.sh"
ENV_ACTIVATE_CMD="$(python_env_prelude)"

CKPT_BASE="${CKPT_BASE:-/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-refcache.train.16xA800.reward.16x4090.run.20260404-032344/checkpoints}"
RUN_DIR="$(dirname "${CKPT_BASE}")"
RESULTS_DIR="${RUN_DIR}/eval_results"
EVAL_SCRIPT_NAME="${EVAL_SCRIPT_NAME:-drkernel-14b-coldstart-maxturns3-temp1.0-w5t50trim.sh}"
EVAL_USE_WORKER="${EVAL_USE_WORKER:-1}"
EVAL_KEEP_MERGED="${EVAL_KEEP_MERGED:-0}"
EVAL_CLEANUP_KILL_GPU_PIDS="${EVAL_CLEANUP_KILL_GPU_PIDS:-0}"
MERGE_CUDA_VISIBLE_DEVICES="${MERGE_CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${RESULTS_DIR}"

discover_untested_steps() {
    find "${CKPT_BASE}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' \
        | sed 's/^global_step_//' \
        | sort -n \
        | while read -r step; do
            [[ -z "${step}" ]] && continue
            [[ ! -d "${CKPT_BASE}/global_step_${step}/actor" ]] && continue
            [[ -f "${RESULTS_DIR}/step_${step}/metrics.json" ]] && continue
            printf '%s\n' "${step}"
        done
}

if [[ -n "${EVAL_STEPS:-}" ]]; then
    read -r -a STEPS <<<"${EVAL_STEPS}"
else
    mapfile -t STEPS < <(discover_untested_steps)
fi

if [[ ${#STEPS[@]} -eq 0 ]]; then
    info "No untested checkpoints found under ${CKPT_BASE}"
    exit 0
fi

# =============================================================================
# Step 1: Merge all FSDP checkpoints to HF format
# =============================================================================
# =============================================================================
# Step 1: Run merge + eval + cleanup in parallel on two nodes
# Each node merges its own checkpoints, evals, then deletes the merged files
# =============================================================================
if [[ "${EVAL_USE_WORKER}" == "0" || "${EVAL_USE_WORKER}" == "false" || "${EVAL_USE_WORKER}" == "False" ]]; then
    info "=== Running merge + eval on head node only ==="
else
    info "=== Running merge + eval (2 nodes in parallel) ==="
fi

STEPS_A=()
STEPS_B=()
if [[ "${EVAL_USE_WORKER}" == "0" || "${EVAL_USE_WORKER}" == "false" || "${EVAL_USE_WORKER}" == "False" ]]; then
    STEPS_A=("${STEPS[@]}")
else
    for i in "${!STEPS[@]}"; do
        if (( i % 2 == 0 )); then
            STEPS_A+=("${STEPS[$i]}")
        else
            STEPS_B+=("${STEPS[$i]}")
        fi
    done
fi

info "Node A ($(train_head_label)): steps ${STEPS_A[*]}"
if [[ ${#STEPS_B[@]} -gt 0 ]]; then
    info "Node B ($(train_worker_label)): steps ${STEPS_B[*]}"
fi

# Function to merge, eval, then delete merged checkpoint on a node
run_evals_on_node() {
    local runner=$1
    local node_name=$2
    shift 2
    local steps=("$@")

    for step in "${steps[@]}"; do
        CKPT_DIR="${CKPT_BASE}/global_step_${step}"
        HF_DIR="${CKPT_DIR}/actor/huggingface_merged"
        OUTPUT_DIR="${RESULTS_DIR}/step_${step}"
        METRICS_FILE="${OUTPUT_DIR}/metrics.json"

        if [[ ! -d "${CKPT_DIR}/actor" ]]; then
            info "[${node_name}] Step ${step}: checkpoint not found, skipping"
            continue
        fi

        if [ -f "${METRICS_FILE}" ]; then
            info "[${node_name}] Step ${step}: already evaluated"
            continue
        fi

        # Merge if needed
        if [ ! -f "${HF_DIR}/model.safetensors.index.json" ] && ! ls "${HF_DIR}"/model-*.safetensors >/dev/null 2>&1; then
            local merge_env
            local merge_args
            merge_env="PYTHONPATH=${VLLM018_PATH}/drkernel/verl:\$PYTHONPATH"
            merge_args="--backend fsdp --local_dir '${CKPT_DIR}/actor' --target_dir '${HF_DIR}'"

            if [[ -n "${MERGE_CUDA_VISIBLE_DEVICES}" ]]; then
                merge_env="CUDA_VISIBLE_DEVICES='${MERGE_CUDA_VISIBLE_DEVICES}' ${merge_env}"
            fi

            info "[${node_name}] Step ${step}: merging FSDP shards (GPU)..."
            "${runner}" \
                "${ENV_ACTIVATE_CMD} && cd ${VLLM018_PATH}/drkernel && ${merge_env} python3 -m verl.model_merger merge ${merge_args}" 2>&1 | tail -3
        fi

        if [ ! -f "${HF_DIR}/model.safetensors.index.json" ] && ! ls "${HF_DIR}"/model-*.safetensors >/dev/null 2>&1; then
            warn "[${node_name}] Step ${step}: merge failed, skipping"
            continue
        fi

        # Eval
        mkdir -p "${OUTPUT_DIR}"
        info "[${node_name}] Step ${step}: starting eval..."

        "${runner}" \
            "${ENV_ACTIVATE_CMD} && cd ${VLLM018_PATH}/drkernel && bash kernel/scripts/eval/${EVAL_SCRIPT_NAME} \
            --model_path ${HF_DIR} \
            --model_name step_${step} \
            --output_path ${OUTPUT_DIR}/graded_results.parquet \
            --metrics_output_path ${METRICS_FILE} \
            --raw_response_path ${OUTPUT_DIR}/raw_responses.jsonl \
            --experiment_name eval_step_${step} \
            --gradio_visualization False" 2>&1 | tail -3

        if [ -f "${METRICS_FILE}" ]; then
            info "[${node_name}] Step ${step}: eval complete"
        else
            warn "[${node_name}] Step ${step}: eval failed"
        fi

        info "[${node_name}] Step ${step}: cleaning up..."
        if [[ "${EVAL_KEEP_MERGED}" != "1" && "${EVAL_KEEP_MERGED}" != "true" && "${EVAL_KEEP_MERGED}" != "True" ]]; then
            rm -rf "${HF_DIR}"
        fi
        if [[ "${EVAL_CLEANUP_KILL_GPU_PIDS}" == "1" || "${EVAL_CLEANUP_KILL_GPU_PIDS}" == "true" || "${EVAL_CLEANUP_KILL_GPU_PIDS}" == "True" ]]; then
            "${runner}" \
                'for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done' 2>/dev/null
        fi
        sleep 5
    done
}

# Launch eval workers
run_evals_on_node run_on_train_head "nodeA" "${STEPS_A[@]}" &
PID_A=$!
PID_B=""
if [[ ${#STEPS_B[@]} -gt 0 ]]; then
    run_evals_on_node run_on_train_worker "nodeB" "${STEPS_B[@]}" &
    PID_B=$!
fi

info "Waiting for eval workers to finish..."
wait $PID_A
if [[ -n "${PID_B}" ]]; then
    wait $PID_B
fi
info "All evaluations complete"

# =============================================================================
# Step 3: Summary
# =============================================================================
info "=== Evaluation Summary ==="

printf "%-6s | %-8s | %-8s | %-8s | %-10s | %-10s\n" "Step" "pass@1" "correct" "compile" "fast@1" "fast@1.2" | tee "${RESULTS_DIR}/summary.txt"
printf "%-6s-+-%-8s-+-%-8s-+-%-8s-+-%-10s-+-%-10s\n" "------" "--------" "--------" "--------" "----------" "----------" | tee -a "${RESULTS_DIR}/summary.txt"

for step in "${STEPS[@]}"; do
    METRICS_FILE="${RESULTS_DIR}/step_${step}/metrics.json"
    if [ -f "${METRICS_FILE}" ]; then
        python3 - "${step}" "${METRICS_FILE}" <<'PY' 2>/dev/null | tee -a "${RESULTS_DIR}/summary.txt"
import json
import sys

step = sys.argv[1]
metrics_path = sys.argv[2]
d = json.load(open(metrics_path))
pass1 = d.get('val/test_score/kernelbench_level2_validation_pass@1', -1)
correct = d.get('val/kernel/best_by_turn_3/correctness_rate', -1)
compile_rate = d.get('val/test_score_extra/compilation_kernelbench_level2_validation', -1)
fast1 = d.get('val/kernel/best_by_turn_3/fast@1_in_all', d.get('val/kernel/best_by_turn_3/fast@1', -1))
fast12 = d.get('val/kernel/best_by_turn_3/fast@1.2_in_all', d.get('val/kernel/best_by_turn_3/fast@1.2', -1))
print(f"{step:<6} | {pass1:<8.4f} | {correct:<8.4f} | {compile_rate:<8.4f} | {fast1:<10.4f} | {fast12:<10.4f}")
PY
    else
        printf "%-6s | %-8s | %-8s | %-8s | %-10s | %-10s\n" "${step}" "N/A" "N/A" "N/A" "N/A" "N/A" | tee -a "${RESULTS_DIR}/summary.txt"
    fi
done

info "Summary saved to ${RESULTS_DIR}/summary.txt"
