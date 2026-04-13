# Online W8A8 Rollout Handoff

## Goal

Make `vllm==0.18.0` + `compressed-tensors` + online `W8A8` rollout work in RL training on the `KernelGYM-vllm018` worktree with:
- no hallmark output corruption
- at least one visible completed train step
- a reproducible debug ladder from minimal clean generation to the current full-restore failure

## Fast Start

- worktree: `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018`
- branch: `codex/vllm-0180-verify`
- venv: `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/.venv-vllm0180`
- training entrypoint: [14b_coldstart_trloo_mrs_pr_prs.sh](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/14b_coldstart_trloo_mrs_pr_prs.sh)

Always verify on the launch node before starting:

```bash
source /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/.venv-vllm0180/bin/activate
which python
python - <<'PY'
import os, sys
print(sys.executable)
print(os.environ.get("VIRTUAL_ENV"))
PY
```

Persistent jobs must run under `tmux`.

## Current Bottom Line

1. The old hallmark corruption mode has been removed on the `Quarot + no-qkv + W8A8` line.
2. The current blocker is now trainer-side: the fully restored run reaches generation, then dies after buffered retry with:
   - `Filtered batch: 768 -> 0 examples`
   - `RuntimeError: split_size must be a positive integer, but got 0`
3. A separate exported-model bug was fixed: static exported rollout checkpoints were previously assembled incorrectly and were unusable because duplicate tensor names caused vLLM to overwrite correct quantized weights with base float weights.
4. Important interpretation correction: the Quarot `transformed_model` used in this ladder is a bf16 checkpoint, not a pre-quantized vLLM checkpoint. `W8A8` init passes because `weight` key names still match and vLLM tolerates missing auxiliary quant tensors such as `weight_scale`; do not treat init success as proof that the pre-sync engine state is semantically correct.

## Reproduction Ladder

### A. Corruption-free reproduction baseline

Use:
- [run_formal_w8a8_noqkv_actorcompile_eager_initprofile_quarot.sh](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/run_formal_w8a8_noqkv_actorcompile_eager_initprofile_quarot.sh)

This setting is:
- model: `/nfs/FM/chenshuailin/code/llmc/checkpoints/drkernel-14B-coldstart-fp16/quarot/w8a8/transformed_model` (bf16 transformed checkpoint, not a packed quant checkpoint)
- online quantization: `W8A8`
- ignore list: `[lm_head,*.self_attn.q_proj,*.self_attn.k_proj,*.self_attn.v_proj]`
- `fake_dequant=False`
- actor `torch.compile=True`
- ref `torch.compile=True`
- rollout `enforce_eager=True`
- initial profile enabled

Validated run:
- [20260325-114346](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.run.20260325-114346)

Observed:
- generation works
- hallmark corruption scan stayed at `0` hits over the scanned rows
- still no completed visible trainer step, but this is the best current “clean generation” baseline

### B. Fully restored formal-path failure reproduction

Use:
- [run_formal_w8a8_noqkv_fullrestore_quarot.sh](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/run_formal_w8a8_noqkv_fullrestore_quarot.sh)

This restores:
- actor/ref compile to formal-script default behavior
- rollout `enforce_eager=False`
- CUDA graph path
- initial profile

Validated run:
- [20260325-120850](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.run.20260325-120850)

Current verified status of that run:
- successful visible train step: `0`
- current in-flight step: `1`
- retry / oversampling state: buffered retry happened, then fatal zero-batch
- WandB state: disabled / console-only
- `Initial validation metrics`: absent
- `Training Progress`: present at `0/4499000`
- latest trainer-visible step: `step:1`

## Important Init-Path Correction

- `/nfs/FM/chenshuailin/code/llmc/checkpoints/drkernel-14B-coldstart-fp16/quarot/w8a8/transformed_model` is bf16:
  - `config.json` reports `dtype: "bfloat16"`
  - there is no `compression_config`
  - the weight files do not include `weight_scale`, `input_scale`, `weight_packed`, or `weight_shape`
- `CompressedTensorsW8A8Int8` registers both `weight` and `weight_scale`, but vLLM's `AutoWeightsLoader` only iterates over tensors that actually exist in the checkpoint.
- That means init can proceed because `...weight` still matches, while missing `...weight_scale` is silently tolerated.
- vLLM's init-time `profile_run()` and, on non-eager runs, warmup / CUDA-graph capture still happen before the later FSDP→vLLM live sync.

Implication:
- the current W8A8 ladder should not be described as "starting from a correct quantized checkpoint"
- the observed clean generation on the eager line is evidence that later live sync can recover runtime behavior, not evidence that the pre-sync init state was already correct
- this is also why W8A8 cannot be used as proof that W8A16 only needs a simple `weight` -> `weight_packed` rename

Fatal sequence:
- `[Oversampling] Selected 160 of 256 required samples`
- `[Buffer] Save to buffer, current buffer size: 10`
- `[Buffer] Using buffered batch to form a new batch. The new batch size is: 78`
- `Filtered batch: 768 -> 0 examples`
- `RuntimeError: split_size must be a positive integer, but got 0`

## What Was Fixed

### 1. Exported rollout checkpoint assembly

Problem:
- the old export builder preserved base `model-*.safetensors`, appended `kernelgym-quant-*.safetensors`, and only rewrote the weight map
- vLLM then opened both file sets and loaded duplicate tensor names twice
- later base float weights overwrote correct quantized int8 weights

Fix:
- [build_quant_model_dir_from_export.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/debug/build_quant_model_dir_from_export.py) now builds:
  - `kernelgym-base-*.safetensors`
  - `kernelgym-quant-*.safetensors`
  - no legacy `model-*.safetensors`
  - no duplicate keys
  - `compression_config` preserved
  - `torch_dtype=bfloat16`

Fixed export dir:
- [rollout_export_noqkv_model](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/activation_compare/rollout_export_noqkv_model)

Sanity result:
- minimal prompt `你是谁？` now returns a normal answer instead of garbage

This is a real fix. Do not revert it.

### 2. Log noise cleanup

The following noisy debug logs are now env-gated and off by default:
- online quant sample-param logs
- online quant lifecycle debug logs
- reward-manager debug prints
- verbose vLLM init/config prints

Relevant env flags:
- `KERNELGYM_VLLM_ENABLE_ONLINE_QUANT_SAMPLE_LOGS=1`
- `KERNELGYM_VLLM_ENABLE_ONLINE_QUANT_DEBUG_LOGS=1`
- `KERNELGYM_REWARD_MANAGER_DEBUG_LOGS=1`
- `KERNELGYM_VLLM_ENABLE_VERBOSE_INIT_LOGS=1`

These changes can stay. They reduce noise without removing the ability to re-enable debugging.

## What Is Still Temporary

These are still workarounds, not final product settings:

- `WANDB_DISABLED=true`
- `TRAINER_LOGGERS='[console]'`
  - reason: environment-side WandB/protobuf issues on this line

- `RAY_memory_monitor_refresh_ms=0`
- `RAY_memory_usage_threshold=0.99`
  - reason: earlier quantized debug/formal runs hit Ray node-memory killer before reaching the real rollout failure boundary

- Quarot model path in the staged reproduction scripts
  - current model:
    `/nfs/FM/chenshuailin/code/llmc/checkpoints/drkernel-14B-coldstart-fp16/quarot/w8a8/transformed_model`
  - this directory is a bf16 `transformed_model`; online quantization is supplied at runtime rather than stored in the checkpoint
  - this is part of the current reproduction ladder, not necessarily the final intended production model choice

## How To Verify Status Correctly

Do not infer state from a log tail. Always check:

```bash
run=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/<RUN_DIR>
rg -n "Initial validation metrics|Training Progress|^step:" "$run/main.log" "$run/trainer.log"
rg -n "\\[Oversampling\\]|\\[Buffer\\]|Filtered batch|split_size must be a positive integer" "$run/main.log" "$run/trainer.log"
```

Keep these counters separate in any status report:
- successful visible train step
- current in-flight step
- retry / oversampling state
- WandB state

## How To Scan For The Old Corruption Pattern

The historical high-signal markers were:
- `<|im_start|>`
- `PyT期待`
- `こん`
- `爱美`
- `Cumhur`
- `\tTokenName`

Quick scan:

```bash
run=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/<RUN_DIR>/structured
rg -n "<\\|im_start\\|>|PyT期待|こん|爱美|Cumhur|\\tTokenName" "$run"/generated_code.pid*.jsonl
```

For the current Quarot `no-qkv` runs in the restoration ladder, these markers stayed absent in the scanned outputs.

## Why The Full-Restore Run Dies

The numbers in the logs are different units:
- `160 of 256`: selected `last-turn` rows after unified filtering
- `buffer size: 10`: `160 / 16`, i.e. prompt count in last-turn terms
- `new batch size: 78`: full multi-turn rows divided only by `rollout.n`; not real prompt count
- `768 -> 0`: full multi-turn rows after restoration, all dropped because final `response_mask` became zero

Current strongest interpretation:
- enough `last-turn` rows survive the unified filter to enter buffered retry
- but after restoring to full multi-turn form, those rows are already non-trainable because `loss_mask` has zeroed them
- the final training-stage `response_mask.sum(dim=1) == 0` filter removes all rows
- trainer then crashes when trying to split a zero-sized batch

Relevant code:
- [kernel_trainer.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/kernel_trainer.py)
- [vllm_async_engine.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/rollout/vllm_rollout/vllm_async_engine.py)
- [unified_filter.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/verl_patch/trainer/code/filters/unified_filter.py)

## Highest-Value Next Step

Instrument the final zero-batch boundary just before the last `response_mask == 0` drop and print:
- `loss_mask` zero/one counts
- `response_mask.sum(dim=1)` histogram
- per-turn `contain_error` or equivalent error-mask rate

Then rerun:
- [run_formal_w8a8_noqkv_fullrestore_quarot.sh](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/run_formal_w8a8_noqkv_fullrestore_quarot.sh)

That is the shortest path to answering why the formally restored Quarot line still cannot complete `step 1`.

## Useful Scripts

- export-model rebuild:
  - [build_quant_model_dir_from_export.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/debug/build_quant_model_dir_from_export.py)
- first-turn replay:
  - [repro_first_turn_request.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/debug/repro_first_turn_request.py)
- blob reader:
  - [read_generated_code_blob.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/debug/read_generated_code_blob.py)
- activation probe:
  - [run_offline_vllm_activation_probe.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/debug/run_offline_vllm_activation_probe.py)
- next-token replay:
  - [compare_two_models_nexttoken.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/debug/compare_two_models_nexttoken.py)
