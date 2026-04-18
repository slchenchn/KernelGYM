#!/bin/bash
# Merge FSDP checkpoints to HF format and eval them.
#
# Usage:
#   bash merge_and_eval_checkpoints.sh
#
# If EVAL_STEPS is unset, the script auto-detects untested checkpoints under
# CKPT_BASE by scanning global_step_* directories and skipping any step that
# already has eval_results/step_*/metrics.json on the head-node-visible results
# tree.
#
# The merge path works in both environments:
# - shared storage: the target node already sees all FSDP shards locally
# - split storage: each node sees only its own shard subset, so the script
#   stages missing model shards plus rank-0 metadata into a temporary local
#   merge dir before calling verl.model_merger

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

init_env() {
    local profile=""
    local i next

    for ((i=1; i<=$#; i++)); do
        if [[ "${!i}" == "--profile" ]]; then
            next=$((i + 1))
            if (( next > $# )); then
                echo "[ERROR] --profile requires a value" >&2
                exit 1
            fi
            profile="${!next}"
        fi
    done

    TRAIN_CLUSTER_PROFILE="${TRAIN_CLUSTER_PROFILE:-${profile:-a800}}"
    # shellcheck disable=SC1090
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
    MERGE_STAGE_DIR_BASENAME="${MERGE_STAGE_DIR_BASENAME:-.fsdp_merge_stage}"
}

is_false() {
    case "${1:-}" in
        0|false|False|FALSE) return 0 ;;
        *) return 1 ;;
    esac
}

is_true() {
    case "${1:-}" in
        1|true|True|TRUE) return 0 ;;
        *) return 1 ;;
    esac
}

run_on_target() {
    local target=$1
    shift
    case "${target}" in
        head) run_on_train_head "$*" ;;
        worker) run_on_train_worker "$*" ;;
        *)
            error "Unknown train target '${target}'"
            return 1
            ;;
    esac
}

train_target_mode() {
    case "$1" in
        head) printf '%s' "${TRAIN_HEAD_MODE}" ;;
        worker) printf '%s' "${TRAIN_WORKER_MODE}" ;;
        *) return 1 ;;
    esac
}

train_target_ssh_target() {
    case "$1" in
        head) printf '%s' "${TRAIN_HEAD_SSH_TARGET}" ;;
        worker) printf '%s' "${TRAIN_WORKER_SSH_TARGET}" ;;
        *) return 1 ;;
    esac
}

train_target_port() {
    case "$1" in
        head) printf '%s' "${HEAD_PORT}" ;;
        worker) printf '%s' "${WORKER_PORT}" ;;
        *) return 1 ;;
    esac
}

train_target_container() {
    case "$1" in
        head) printf '%s' "${TRAIN_HEAD_CONTAINER}" ;;
        worker) printf '%s' "${TRAIN_WORKER_CONTAINER}" ;;
        *) return 1 ;;
    esac
}

train_target_label() {
    case "$1" in
        head) train_head_label ;;
        worker) train_worker_label ;;
        *) return 1 ;;
    esac
}

build_train_ssh_opts() {
    local port=$1
    TRAIN_SSH_OPTS=(-o ConnectTimeout=10 -o StrictHostKeyChecking=no)
    if [[ -n "${port}" && "${port}" != "22" ]]; then
        TRAIN_SSH_OPTS+=(-p "${port}")
    fi
}

emit_tar_stream_from_target() {
    local target=$1
    local src_dir=$2
    shift 2
    local files=("$@")
    local mode ssh_target port container tar_cmd

    mode="$(train_target_mode "${target}")"
    ssh_target="$(train_target_ssh_target "${target}")"
    port="$(train_target_port "${target}")"
    container="$(train_target_container "${target}")"

    tar_cmd=$(printf 'cd %q && tar -cf - --' "${src_dir}")
    for rel in "${files[@]}"; do
        tar_cmd+=" $(printf '%q' "${rel}")"
    done

    case "${mode}" in
        local)
            bash -lc "${tar_cmd}"
            ;;
        ssh)
            build_train_ssh_opts "${port}"
            ssh "${TRAIN_SSH_OPTS[@]}" "${ssh_target}" "bash -lc $(printf %q "${tar_cmd}")"
            ;;
        ssh-docker)
            build_train_ssh_opts "${port}"
            ssh "${TRAIN_SSH_OPTS[@]}" "${ssh_target}" \
                "docker exec ${container} bash -lc $(printf %q "${tar_cmd}")"
            ;;
        *)
            error "Unsupported train target mode '${mode}' for tar source"
            return 1
            ;;
    esac
}

extract_tar_stream_to_target() {
    local target=$1
    local dst_dir=$2
    local mode ssh_target port container extract_cmd

    mode="$(train_target_mode "${target}")"
    ssh_target="$(train_target_ssh_target "${target}")"
    port="$(train_target_port "${target}")"
    container="$(train_target_container "${target}")"

    extract_cmd=$(printf 'mkdir -p %q && cd %q && tar -xf -' "${dst_dir}" "${dst_dir}")

    case "${mode}" in
        local)
            bash -lc "${extract_cmd}"
            ;;
        ssh)
            build_train_ssh_opts "${port}"
            ssh "${TRAIN_SSH_OPTS[@]}" "${ssh_target}" "bash -lc $(printf %q "${extract_cmd}")"
            ;;
        ssh-docker)
            build_train_ssh_opts "${port}"
            ssh "${TRAIN_SSH_OPTS[@]}" "${ssh_target}" \
                "docker exec -i ${container} bash -lc $(printf %q "${extract_cmd}")"
            ;;
        *)
            error "Unsupported train target mode '${mode}' for tar extract"
            return 1
            ;;
    esac
}

copy_relpaths_within_target() {
    local target=$1
    local src_dir=$2
    local dst_dir=$3
    shift 3
    local files=("$@")
    local copy_cmd

    [[ ${#files[@]} -eq 0 ]] && return 0

    copy_cmd=$(printf 'mkdir -p %q && cd %q && cp -a --' "${dst_dir}" "${src_dir}")
    for rel in "${files[@]}"; do
        copy_cmd+=" $(printf '%q' "${rel}")"
    done
    copy_cmd+=" $(printf '%q' "${dst_dir}/")"

    run_on_target "${target}" "${copy_cmd}"
}

copy_relpaths_between_targets() {
    local src_target=$1
    local src_dir=$2
    local dst_target=$3
    local dst_dir=$4
    shift 4
    local files=("$@")

    [[ ${#files[@]} -eq 0 ]] && return 0

    if [[ "${src_target}" == "${dst_target}" ]]; then
        copy_relpaths_within_target "${src_target}" "${src_dir}" "${dst_dir}" "${files[@]}"
        return $?
    fi

    (
        set -o pipefail
        emit_tar_stream_from_target "${src_target}" "${src_dir}" "${files[@]}" \
            | extract_tar_stream_to_target "${dst_target}" "${dst_dir}"
    )
}

target_dir_exists() {
    local target=$1
    local path=$2
    local cmd
    cmd=$(printf '[ -d %q ]' "${path}")
    run_on_target "${target}" "${cmd}" >/dev/null
}

target_file_exists() {
    local target=$1
    local path=$2
    local cmd
    cmd=$(printf '[ -f %q ]' "${path}")
    run_on_target "${target}" "${cmd}" >/dev/null
}

target_has_merged_hf() {
    local target=$1
    local hf_dir=$2
    local cmd
    cmd=$(printf '[ -f %q ] || ls %s/model-*.safetensors >/dev/null 2>&1' \
        "${hf_dir}/model.safetensors.index.json" "${hf_dir}")
    run_on_target "${target}" "${cmd}" >/dev/null
}

get_world_size_from_head() {
    local actor_dir=$1
    local script cmd
    script='import json, sys; print(json.load(open(sys.argv[1]))["world_size"])'
    cmd=$(printf 'python3 -c %q %q' "${script}" "${actor_dir}/fsdp_config.json")
    run_on_train_head "${cmd}"
}

list_target_model_ranks() {
    local target=$1
    local actor_dir=$2
    local world_size=$3
    local script cmd

    script='import pathlib, re, sys; actor_dir = pathlib.Path(sys.argv[1]); world_size = sys.argv[2]; pattern = re.compile(rf"model_world_size_{world_size}_rank_(\d+)\.pt$"); ranks = sorted(int(m.group(1)) for path in actor_dir.glob(f"model_world_size_{world_size}_rank_*.pt") for m in [pattern.fullmatch(path.name)] if m); print("\n".join(str(rank) for rank in ranks))'
    cmd=$(printf 'python3 -c %q %q %q' "${script}" "${actor_dir}" "${world_size}")
    run_on_target "${target}" "${cmd}"
}

target_has_merge_metadata() {
    local target=$1
    local actor_dir=$2
    local cmd
    cmd=$(printf '[ -f %q ] && [ -f %q ]' \
        "${actor_dir}/fsdp_config.json" \
        "${actor_dir}/huggingface/config.json")
    run_on_target "${target}" "${cmd}" >/dev/null
}

target_has_complete_merge_inputs() {
    local target=$1
    local actor_dir=$2
    local world_size=$3
    local ranks_output
    local -a ranks=()

    ranks_output="$(list_target_model_ranks "${target}" "${actor_dir}" "${world_size}")" || return 1
    if [[ -n "${ranks_output}" ]]; then
        mapfile -t ranks < <(printf '%s\n' "${ranks_output}" | sed '/^$/d')
    fi

    [[ ${#ranks[@]} -eq ${world_size} ]] || return 1
    target_has_merge_metadata "${target}" "${actor_dir}"
}

prepare_merge_local_dir() {
    local target=$1
    local peer=$2
    local node_name=$3
    local step=$4
    local actor_dir=$5
    local world_size local_ranks_output
    local stage_dir
    local -a local_ranks=()
    local -a local_model_files=()
    local -a metadata_files=("fsdp_config.json" "huggingface")
    local -a missing_model_files=()
    local rank
    local -A present=()

    PREPARED_MERGE_LOCAL_DIR=""

    world_size="$(get_world_size_from_head "${actor_dir}")" || return 1

    if target_has_complete_merge_inputs "${target}" "${actor_dir}" "${world_size}"; then
        PREPARED_MERGE_LOCAL_DIR="${actor_dir}"
        return 0
    fi

    stage_dir="${actor_dir}/${MERGE_STAGE_DIR_BASENAME}_${target}"
    info "[${node_name}] Step ${step}: local checkpoint storage is incomplete; staging missing merge inputs from $(train_target_label "${peer}")" >&2
    run_on_target "${target}" "$(printf 'rm -rf %q && mkdir -p %q' "${stage_dir}" "${stage_dir}")" || return 1

    local_ranks_output="$(list_target_model_ranks "${target}" "${actor_dir}" "${world_size}")" || return 1
    if [[ -n "${local_ranks_output}" ]]; then
        mapfile -t local_ranks < <(printf '%s\n' "${local_ranks_output}" | sed '/^$/d')
    fi

    for rank in "${local_ranks[@]}"; do
        [[ -z "${rank}" ]] && continue
        present["${rank}"]=1
        local_model_files+=("model_world_size_${world_size}_rank_${rank}.pt")
    done

    copy_relpaths_between_targets "${target}" "${actor_dir}" "${target}" "${stage_dir}" "${local_model_files[@]}" || {
        run_on_target "${target}" "$(printf 'rm -rf %q' "${stage_dir}")" >/dev/null 2>&1
        return 1
    }

    if target_has_merge_metadata "${target}" "${actor_dir}"; then
        copy_relpaths_between_targets "${target}" "${actor_dir}" "${target}" "${stage_dir}" "${metadata_files[@]}" || {
            run_on_target "${target}" "$(printf 'rm -rf %q' "${stage_dir}")" >/dev/null 2>&1
            return 1
        }
    else
        copy_relpaths_between_targets head "${actor_dir}" "${target}" "${stage_dir}" "${metadata_files[@]}" || {
            run_on_target "${target}" "$(printf 'rm -rf %q' "${stage_dir}")" >/dev/null 2>&1
            return 1
        }
    fi

    for ((rank=0; rank<world_size; rank++)); do
        if [[ -z "${present[${rank}]+x}" ]]; then
            missing_model_files+=("model_world_size_${world_size}_rank_${rank}.pt")
        fi
    done

    if [[ ${#missing_model_files[@]} -gt 0 ]]; then
        copy_relpaths_between_targets "${peer}" "${actor_dir}" "${target}" "${stage_dir}" "${missing_model_files[@]}" || {
            run_on_target "${target}" "$(printf 'rm -rf %q' "${stage_dir}")" >/dev/null 2>&1
            return 1
        }
    fi

    if ! target_has_complete_merge_inputs "${target}" "${stage_dir}" "${world_size}"; then
        warn "[${node_name}] Step ${step}: staged merge dir is still incomplete after shard collection" >&2
        run_on_target "${target}" "$(printf 'rm -rf %q' "${stage_dir}")" >/dev/null 2>&1
        return 1
    fi

    PREPARED_MERGE_LOCAL_DIR="${stage_dir}"
}

cleanup_merge_local_dir() {
    local target=$1
    local actor_dir=$2
    local merge_input_dir=$3

    [[ -z "${merge_input_dir}" || "${merge_input_dir}" == "${actor_dir}" ]] && return 0
    run_on_target "${target}" "$(printf 'rm -rf %q' "${merge_input_dir}")" >/dev/null 2>&1 || true
}

sync_step_results_to_head() {
    local target=$1
    local node_name=$2
    local step=$3

    if [[ "${target}" == "head" ]] || [[ -f "${RESULTS_DIR}/step_${step}/metrics.json" ]]; then
        return 0
    fi

    info "[${node_name}] Step ${step}: syncing eval results back to head-visible results dir..."
    copy_relpaths_between_targets "${target}" "${RESULTS_DIR}" head "${RESULTS_DIR}" "step_${step}"
}

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

run_evals_on_node() {
    local target=$1
    local peer=$2
    local node_name=$3
    shift 3
    local steps=("$@")

    for step in "${steps[@]}"; do
        local ckpt_dir="${CKPT_BASE}/global_step_${step}"
        local actor_dir="${ckpt_dir}/actor"
        local hf_dir="${actor_dir}/huggingface_merged"
        local output_dir="${RESULTS_DIR}/step_${step}"
        local metrics_file="${output_dir}/metrics.json"
        local merge_input_dir=""
        local merge_env
        local merge_args

        if ! target_dir_exists "${target}" "${actor_dir}"; then
            info "[${node_name}] Step ${step}: checkpoint not found on $(train_target_label "${target}"), skipping"
            continue
        fi

        if [[ -f "${metrics_file}" ]]; then
            info "[${node_name}] Step ${step}: already evaluated"
            continue
        fi

        if ! target_has_merged_hf "${target}" "${hf_dir}"; then
            prepare_merge_local_dir "${target}" "${peer}" "${node_name}" "${step}" "${actor_dir}" || {
                warn "[${node_name}] Step ${step}: failed to assemble a complete local merge dir"
                continue
            }
            merge_input_dir="${PREPARED_MERGE_LOCAL_DIR}"

            merge_env="PYTHONPATH=${VLLM018_PATH}/drkernel/verl:\$PYTHONPATH"
            merge_args="--backend fsdp --local_dir '${merge_input_dir}' --target_dir '${hf_dir}'"

            if [[ -n "${MERGE_CUDA_VISIBLE_DEVICES}" ]]; then
                merge_env="CUDA_VISIBLE_DEVICES='${MERGE_CUDA_VISIBLE_DEVICES}' ${merge_env}"
            fi

            info "[${node_name}] Step ${step}: merging FSDP shards (GPU)..."
            run_on_target "${target}" \
                "${ENV_ACTIVATE_CMD} && cd ${VLLM018_PATH}/drkernel && ${merge_env} python3 -m verl.model_merger merge ${merge_args}" \
                2>&1 | tail -3
        fi

        if ! target_has_merged_hf "${target}" "${hf_dir}"; then
            warn "[${node_name}] Step ${step}: merge failed, skipping"
            cleanup_merge_local_dir "${target}" "${actor_dir}" "${merge_input_dir}"
            continue
        fi

        run_on_target "${target}" "$(printf 'mkdir -p %q' "${output_dir}")" || {
            warn "[${node_name}] Step ${step}: failed to create output dir on $(train_target_label "${target}")"
            cleanup_merge_local_dir "${target}" "${actor_dir}" "${merge_input_dir}"
            continue
        }

        info "[${node_name}] Step ${step}: starting eval..."
        run_on_target "${target}" \
            "${ENV_ACTIVATE_CMD} && cd ${VLLM018_PATH}/drkernel && bash kernel/scripts/eval/${EVAL_SCRIPT_NAME} \
            --model_path ${hf_dir} \
            --model_name step_${step} \
            --output_path ${output_dir}/graded_results.parquet \
            --metrics_output_path ${metrics_file} \
            --raw_response_path ${output_dir}/raw_responses.jsonl \
            --experiment_name eval_step_${step} \
            --gradio_visualization False" 2>&1 | tail -3

        if target_file_exists "${target}" "${metrics_file}"; then
            sync_step_results_to_head "${target}" "${node_name}" "${step}" || {
                warn "[${node_name}] Step ${step}: eval finished but syncing results back to head failed"
            }
        fi

        if [[ -f "${metrics_file}" ]]; then
            info "[${node_name}] Step ${step}: eval complete"
        else
            warn "[${node_name}] Step ${step}: eval failed"
        fi

        info "[${node_name}] Step ${step}: cleaning up..."
        cleanup_merge_local_dir "${target}" "${actor_dir}" "${merge_input_dir}"
        if ! is_true "${EVAL_KEEP_MERGED}"; then
            run_on_target "${target}" "$(printf 'rm -rf %q' "${hf_dir}")" >/dev/null 2>&1 || true
        fi
        if is_true "${EVAL_CLEANUP_KILL_GPU_PIDS}"; then
            run_on_target "${target}" \
                'for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done' \
                >/dev/null 2>&1 || true
        fi
        sleep 5
    done
}

main() {
    local -a STEPS=()
    local -a STEPS_A=()
    local -a STEPS_B=()
    local i
    local pid_a pid_b=""

    init_env "$@"
    mkdir -p "${RESULTS_DIR}"

    if [[ -n "${EVAL_STEPS:-}" ]]; then
        read -r -a STEPS <<<"${EVAL_STEPS}"
    else
        mapfile -t STEPS < <(discover_untested_steps)
    fi

    if [[ ${#STEPS[@]} -eq 0 ]]; then
        info "No untested checkpoints found under ${CKPT_BASE}"
        exit 0
    fi

    if is_false "${EVAL_USE_WORKER}"; then
        info "=== Running merge + eval on head node only ==="
        STEPS_A=("${STEPS[@]}")
    else
        info "=== Running merge + eval (2 nodes in parallel) ==="
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

    run_evals_on_node head worker "nodeA" "${STEPS_A[@]}" &
    pid_a=$!
    if [[ ${#STEPS_B[@]} -gt 0 ]]; then
        run_evals_on_node worker head "nodeB" "${STEPS_B[@]}" &
        pid_b=$!
    fi

    info "Waiting for eval workers to finish..."
    wait "${pid_a}"
    if [[ -n "${pid_b}" ]]; then
        wait "${pid_b}"
    fi
    info "All evaluations complete"

    info "=== Evaluation Summary ==="
    printf "%-6s | %-8s | %-8s | %-8s | %-10s | %-10s\n" "Step" "pass@1" "correct" "compile" "fast@1" "fast@1.2" | tee "${RESULTS_DIR}/summary.txt"
    printf "%-6s-+-%-8s-+-%-8s-+-%-8s-+-%-10s-+-%-10s\n" "------" "--------" "--------" "--------" "----------" "----------" | tee -a "${RESULTS_DIR}/summary.txt"

    for step in "${STEPS[@]}"; do
        local metrics_file="${RESULTS_DIR}/step_${step}/metrics.json"
        if [[ -f "${metrics_file}" ]]; then
            python3 - "${step}" "${metrics_file}" <<'PY' 2>/dev/null | tee -a "${RESULTS_DIR}/summary.txt"
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
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
