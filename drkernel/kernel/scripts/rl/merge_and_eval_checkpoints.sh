#!/bin/bash
# Merge FSDP checkpoints to HF format and eval on both training nodes in parallel
#
# Usage:
#   bash merge_and_eval_checkpoints.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/infra_common.sh"

CKPT_BASE="/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-refcache.train.16xA800.reward.16x4090.run.20260404-032344/checkpoints"
RUN_DIR="$(dirname "${CKPT_BASE}")"
RESULTS_DIR="${RUN_DIR}/eval_results"
EVAL_SCRIPT_NAME="drkernel-14b-coldstart-maxturns3-temp1.0-w5t50trim.sh"
mkdir -p "${RESULTS_DIR}"

STEPS=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170)

# Two training nodes for parallel eval
NODE_A_HOST="192.168.16.18"
NODE_A_PORT=20629
NODE_B_HOST="192.168.16.24"
NODE_B_PORT=14218

# =============================================================================
# Step 1: Merge all FSDP checkpoints to HF format
# =============================================================================
# =============================================================================
# Step 1: Run merge + eval + cleanup in parallel on two nodes
# Each node merges its own checkpoints, evals, then deletes the merged files
# =============================================================================
info "=== Running merge + eval (2 nodes in parallel) ==="

# Split steps between two nodes
STEPS_A=()
STEPS_B=()
for i in "${!STEPS[@]}"; do
    if (( i % 2 == 0 )); then
        STEPS_A+=("${STEPS[$i]}")
    else
        STEPS_B+=("${STEPS[$i]}")
    fi
done

info "Node A (${NODE_A_HOST}): steps ${STEPS_A[*]}"
info "Node B (${NODE_B_HOST}): steps ${STEPS_B[*]}"

# Function to merge, eval, then delete merged checkpoint on a node
run_evals_on_node() {
    local host=$1
    local port=$2
    local node_name=$3
    shift 3
    local steps=("$@")

    for step in "${steps[@]}"; do
        CKPT_DIR="${CKPT_BASE}/global_step_${step}"
        HF_DIR="${CKPT_DIR}/actor/huggingface_merged"
        OUTPUT_DIR="${RESULTS_DIR}/step_${step}"
        METRICS_FILE="${OUTPUT_DIR}/metrics.json"

        if [ -f "${METRICS_FILE}" ]; then
            info "[${node_name}] Step ${step}: already evaluated"
            continue
        fi

        # Merge if needed
        if [ ! -f "${HF_DIR}/model.safetensors.index.json" ] && ! ls "${HF_DIR}"/model-*.safetensors >/dev/null 2>&1; then
            info "[${node_name}] Step ${step}: merging FSDP shards (CPU only)..."
            ssh -p ${port} root@${host} "source ${VENV}/bin/activate && cd ${VLLM018_PATH}/drkernel && CUDA_VISIBLE_DEVICES='' PYTHONPATH=${VLLM018_PATH}/drkernel/verl:\$PYTHONPATH python3 -m verl.model_merger merge \
                --backend fsdp \
                --use_cpu_initialization \
                --local_dir '${CKPT_DIR}/actor' \
                --target_dir '${HF_DIR}'" 2>&1 | tail -3
        fi

        if [ ! -f "${HF_DIR}/model.safetensors.index.json" ] && ! ls "${HF_DIR}"/model-*.safetensors >/dev/null 2>&1; then
            warn "[${node_name}] Step ${step}: merge failed, skipping"
            continue
        fi

        # Eval
        mkdir -p "${OUTPUT_DIR}"
        info "[${node_name}] Step ${step}: starting eval..."

        ssh -p ${port} root@${host} "source ${VENV}/bin/activate && cd ${VLLM018_PATH}/drkernel && bash kernel/scripts/eval/${EVAL_SCRIPT_NAME} \
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

        # Clean up: delete merged checkpoint + kill GPU processes
        info "[${node_name}] Step ${step}: cleaning up..."
        rm -rf "${HF_DIR}"
        ssh -p ${port} root@${host} 'for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done' 2>/dev/null
        sleep 5
    done
}

# Launch both in parallel
run_evals_on_node "${NODE_A_HOST}" "${NODE_A_PORT}" "nodeA" "${STEPS_A[@]}" &
PID_A=$!
run_evals_on_node "${NODE_B_HOST}" "${NODE_B_PORT}" "nodeB" "${STEPS_B[@]}" &
PID_B=$!

info "Waiting for both nodes to finish..."
wait $PID_A
wait $PID_B
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
        python3 -c "
import json
d = json.load(open('${METRICS_FILE}'))
pass1 = d.get('val/test_score/kernelbench_level2_validation_pass@1', -1)
correct = d.get('val/kernel/best_by_turn_3/correctness_rate', -1)
compile_rate = d.get('val/test_score_extra/compilation_kernelbench_level2_validation', -1)
fast1 = d.get('val/kernel/best_by_turn_3/fast@1_in_all', d.get('val/kernel/best_by_turn_3/fast@1', -1))
fast12 = d.get('val/kernel/best_by_turn_3/fast@1.2_in_all', d.get('val/kernel/best_by_turn_3/fast@1.2', -1))
print(f'${step:<6} | {pass1:<8.4f} | {correct:<8.4f} | {compile_rate:<8.4f} | {fast1:<10.4f} | {fast12:<10.4f}')
" 2>/dev/null | tee -a "${RESULTS_DIR}/summary.txt"
    else
        printf "%-6s | %-8s | %-8s | %-8s | %-10s | %-10s\n" "${step}" "N/A" "N/A" "N/A" "N/A" "N/A" | tee -a "${RESULTS_DIR}/summary.txt"
    fi
done

info "Summary saved to ${RESULTS_DIR}/summary.txt"
