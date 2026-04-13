#!/usr/bin/env bash
# Fair 3-way initial-validation benchmark:
#   1. BF16 baseline
#   2. Per-channel W8A8 (QuaRot, no QKV quant)
#   3. Blockwise W8A8 (QuaRot, no QKV quant)
#
# All runs use identical settings except quantization config.
# 8 GPUs for rollout, 16 GPUs for reward (remote workers).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/.venv-vllm0180/bin/activate
cd "$SCRIPT_DIR"

BENCHMARK_TS="$(date +%Y%m%d-%H%M%S)"
BENCHMARK_DIR="${SCRIPT_DIR}/logs/val_benchmark_${BENCHMARK_TS}"
mkdir -p "$BENCHMARK_DIR"

# ---- Shared settings ----
export WANDB_DISABLED=true
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export TRAINER_LOGGERS='[console]'
export VAL_ONLY=True
export VAL_BEFORE_TRAIN=True
export VAL_SAMPLE_SIZE=10
export N_VAL=8
export MAX_PROMPT_LENGTH=10240
export MAX_RESPONSE_LENGTH=8192
export ROLLOUT_GPU_MEMORY_UTIL=0.75
export ENFORCE_EAGER=False
export KERNELGYM_SERVER_URL=http://192.168.16.18:8111
export ARNOLD_WORKER_GPU=8
export ARNOLD_WORKER_NUM=1
export N_GPUS_PER_NODE=8
export GPUS_PER_NODE=8
export NNODES=1

BF16_MODEL=/nfs/FM/chenshuailin/checkpoints/hkust-nlp/drkernel-14b-coldstart
QUAROT_MODEL=/nfs/FM/chenshuailin/code/llmc/checkpoints/drkernel-14B-coldstart-fp16/quarot/w8a8/transformed_model
LAUNCH_SCRIPT=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/14b_coldstart_trloo_mrs_pr_prs.sh

run_experiment() {
    local name="$1"
    local model_path="$2"
    local quant_preset="${3:-}"
    local quant_ignore="${4:-}"

    echo "========================================"
    echo "  EXPERIMENT: $name"
    echo "  MODEL: $model_path"
    echo "  QUANT: ${quant_preset:-none}"
    echo "  START: $(date)"
    echo "========================================"

    export RUN_NAME="val-bench-${name}"
    export MODEL_PATH="$model_path"

    if [[ -n "$quant_preset" ]]; then
        export ROLLOUT_ONLINE_QUANTIZATION_PRESET="$quant_preset"
        export ROLLOUT_ONLINE_QUANTIZATION_IGNORE="$quant_ignore"
    else
        unset ROLLOUT_ONLINE_QUANTIZATION_PRESET 2>/dev/null || true
        unset ROLLOUT_ONLINE_QUANTIZATION_IGNORE 2>/dev/null || true
    fi

    local logfile="${BENCHMARK_DIR}/${name}.log"
    local start_ts=$(date +%s)

    bash "$LAUNCH_SCRIPT" --sp_size 4 2>&1 | tee "$logfile"
    local end_ts=$(date +%s)
    local elapsed=$((end_ts - start_ts))

    echo ""
    echo "---- $name completed in ${elapsed}s ----"
    echo "${name}_elapsed_s=${elapsed}" >> "${BENCHMARK_DIR}/timings.txt"

    # Clean up ray between runs
    ray stop --force 2>/dev/null || true
    sleep 5
}

echo "Benchmark started at $(date)" | tee "${BENCHMARK_DIR}/timings.txt"
echo "" >> "${BENCHMARK_DIR}/timings.txt"

# ---- Experiment 1: BF16 ----
run_experiment "bf16" "$BF16_MODEL"

# ---- Experiment 2: Per-channel W8A8 (QuaRot) ----
run_experiment "w8a8_perchannel" "$QUAROT_MODEL" \
    "W8A8" \
    "[lm_head,*.self_attn.q_proj,*.self_attn.k_proj,*.self_attn.v_proj]"

# ---- Experiment 3: Blockwise W8A8 (QuaRot) ----
run_experiment "w8a8_blockwise" "$QUAROT_MODEL" \
    "W8A8_BLOCK" \
    "[lm_head,*.self_attn.q_proj,*.self_attn.k_proj,*.self_attn.v_proj]"

echo "" >> "${BENCHMARK_DIR}/timings.txt"
echo "Benchmark finished at $(date)" >> "${BENCHMARK_DIR}/timings.txt"

echo ""
echo "========================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results in: ${BENCHMARK_DIR}"
echo "========================================"
cat "${BENCHMARK_DIR}/timings.txt"

# ---- Generate summary ----
echo ""
echo "========================================"
echo "  SUMMARY"
echo "========================================"
for name in bf16 w8a8_perchannel w8a8_blockwise; do
    logfile="${BENCHMARK_DIR}/${name}.log"
    if [[ ! -f "$logfile" ]]; then
        echo "$name: NO LOG"
        continue
    fi
    env_results=$(grep -c 'Env Result' "$logfile" 2>/dev/null || echo 0)
    successes=$(grep -oP 'success=True' "$logfile" 2>/dev/null | wc -l || echo 0)
    failures=$(grep -oP 'success=False' "$logfile" 2>/dev/null | wc -l || echo 0)
    compilations=$(grep -oP 'compilation=True' "$logfile" 2>/dev/null | wc -l || echo 0)
    decoys=$(grep -oP 'decoy=True' "$logfile" 2>/dev/null | wc -l || echo 0)
    mean_reward=$(grep -oP 'reward=\K[0-9.]+' "$logfile" 2>/dev/null | awk '{sum+=$1; n++} END {if(n>0) printf "%.4f", sum/n; else print "N/A"}')
    elapsed=$(grep "^${name}_elapsed_s=" "${BENCHMARK_DIR}/timings.txt" 2>/dev/null | cut -d= -f2 || echo "?")

    echo "$name:"
    echo "  env_results=$env_results  success=$successes  fail=$failures"
    echo "  compilation=$compilations  decoy=$decoys"
    echo "  mean_reward=$mean_reward"
    echo "  elapsed=${elapsed}s"
    echo "  success_rate=$(echo "scale=4; $successes / ($successes + $failures + 0.0001)" | bc 2>/dev/null || echo '?')"
    echo "  compilation_rate=$(echo "scale=4; $compilations / ($env_results + 0.0001)" | bc 2>/dev/null || echo '?')"
    echo ""
done
