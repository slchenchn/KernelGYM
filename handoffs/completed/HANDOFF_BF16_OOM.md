# BF16 Rollout OOM During Training Backward — Investigation Handoff

## Status: RESOLVED

The BF16 backward OOM is fixed and validated. Training has run 5 consecutive steps without failure.

## Root Cause

Eager optimizer-state loading in `update_actor()` caused a ~47 GiB GPU memory baseline jump at step=2.

### Mechanism

1. The actor uses `optim.AdamW`. Adam state tensors (`exp_avg`, `exp_avg_sq`) are lazily materialized — they do not exist until the first successful `optimizer.step()`.
2. The FSDP offload helper `load_fsdp_optimizer(...)` no-ops when `optimizer.state` is empty.
3. Therefore step=1 runs its entire micro-batch forward/backward loop without any optimizer state on GPU.
4. After step=1's `optimizer.step()` succeeds, the state materializes (~47 GiB for the 14.77B model).
5. The old code loaded optimizer state eagerly at the start of `update_actor()`, before the micro-batch loop. That meant every micro-batch in step=2 paid the full Adam-state baseline.
6. That higher baseline, plus normal gradient retention across micro-batches, pushed `micro_batch=1` over the 80 GiB A800 limit.

### Why earlier theories were wrong

- **"Step 0 left dirty memory behind"**: The original reproduced OOM had no completed visible train step before failure. There was no prior step whose residue could explain the crash.
- **"vLLM sleep left 71GB resident"**: The `before_backward` snapshots showed the actor at ~48-52 GiB before backward, not 71 GiB. The jump to 71 GiB happened during backward allocation pressure, not before it.

## Fix

Delay optimizer-state loading from the start of `update_actor()` to immediately before `optimizer.step()` inside `_optimizer_step()`.

### Implementation

In `fsdp_workers.py`, during actor init, when optimizer offload is enabled:

```python
self.actor._load_optimizer_state_before_step = (
    lambda: load_fsdp_optimizer(
        optimizer=self.actor_optimizer, device_id=torch.cuda.current_device()
    )
)
```

In `dp_actor.py`, `_optimizer_step()` calls the deferred load right before `optimizer.step()`:

```python
if self._load_optimizer_state_before_step is not None:
    self._load_optimizer_state_before_step()
self.actor_optimizer.step()
```

The old eager `load_fsdp_optimizer` call was removed from `update_actor()`. Actor params still load at `update_actor()` start when param offload is enabled. The offload at `update_actor()` end is unchanged.

### Correctness notes

- `torch.cuda.current_device()` is evaluated at call time, not capture time — correct.
- The non-finite grad_norm path calls `zero_grad()` without loading optimizer state — safe, because `zero_grad()` operates on param gradients, not optimizer state.
- `offload_fsdp_optimizer` after `update_policy` returns no-ops if state was never loaded (empty state check).
- If `ppo_epochs > 1`, optimizer state stays on GPU after epoch 0's step through subsequent epochs. The critical failure case (step=2, epoch 0, micro_batch=1) is fully covered.

## Validation Evidence

### Three runs tell the story

| Run | Config | Step=1 | Step=2 | Outcome |
|---|---|---|---|---|
| `...20260327-090301` | `PPO_MICRO_TOKEN=16384` (buggy auto) | OOM on retried step=1 | — | Token cap too high |
| `...20260327-122333` | `PPO_MICRO_TOKEN=12288`, old eager load | Passed | OOM on retried step=2 | Optimizer-state baseline |
| `...20260328-013235` | `PPO_MICRO_TOKEN=12288`, deferred load | Passed | **Passed** | Fix validated |

### Memory comparison (rank 3, retried step=2)

| Stage | Before fix (`...122333`) | After fix (`...013235`) |
|---|---|---|
| micro_batch=0 `after_batch_to_device` | ~57 GiB device used | ~9.76 GiB device used |
| micro_batch=1 `after_batch_to_device` | ~77 GiB device used | ~25 GiB device used |
| micro_batch=1 backward result | OOM (0.10 GiB free) | Success (~53 GiB free) |
| `ooms` counter | 1 | 0 |

The ~47 GiB difference matches the expected size of fully materialized AdamW state for a 14.77B parameter model (2 fp32 state tensors per parameter).

### Steady-state behavior confirmed

The validation run completed 5 consecutive steps (as of last check) with no OOM. The step=2 memory profile is indistinguishable from step=1: `after_batch_to_device` starts at ~10 GiB, `after_backward` drops back to ~25-26 GiB, and the pattern is stable across all micro-batches.

## Config / Script Fixes (also landed)

### 14B token-cap auto-detection was wrong

`train_rl_common.sh` only matched uppercase `14B`, but the model name is `hkust-nlp/drkernel-14b-coldstart`. That silently fell back to `PPO_MICRO_TOKEN=16384`. Fixed: the regex now uses `[bB]` to accept both cases.

### The true auto value (`4096`) is incompatible with the length budget

With `MAX_PROMPT_LENGTH=10240`, `MAX_RESPONSE_LENGTH=8192`, `sp_size=4`: required length is `18432`, but `4096 * 4 = 16384`. So `4096` is not valid for the current run shape. `PPO_MICRO_TOKEN=12288` is the working override (`12288 * 4 = 49152 >= 18432`).

### The 14B launcher ignored external `PPO_MICRO_TOKEN`

The launcher reset `PPO_MICRO_TOKEN=null` internally. Fixed: it now respects an existing environment override via `${PPO_MICRO_TOKEN:-null}`.

## Instrumentation (retained, gated behind `KERNELGYM_BF16_OOM_DEBUG=1`)

### Actor-side snapshots (`dp_actor.py`)

Logged at each micro-batch: `after_batch_to_device`, `after_forward`, `before_backward`, `after_backward`, `backward_oom`. Each includes epoch/batch/micro-batch indices, token counts, PyTorch allocator stats, and device memory.

### Rank-0 process snapshot

Rank 0 additionally logs `nvidia-smi --query-compute-apps` at each snapshot point, distinguishing actor self-usage from other GPU processes.

### Async engine snapshots

The async vLLM path logs snapshots around wake/sleep transitions.

## Remaining Considerations

### `PPO_MICRO_TOKEN` auto value needs updating

The 14B auto value of `4096` is incompatible with the current length budget. Either the auto table should be updated to a valid value (e.g., `12288`), or the prompt/response length budget should be adjusted. Currently the run depends on an explicit external override.

### `KERNELGYM_BF16_OOM_DEBUG` can be disabled

The debug logging adds some overhead (nvidia-smi subprocess calls on rank 0). Once confidence is established over more steps, the flag can be dropped from the launch script.

## File References

Key code:

- `drkernel/verl_patch/workers/code/actor/dp_actor.py` — `_optimizer_step()`, BF16 debug snapshots
- `drkernel/verl_patch/workers/code/fsdp_workers.py` — `_load_optimizer_state_before_step` wiring, `update_actor()`
- `drkernel/verl/verl/utils/fsdp_utils.py` — `load_fsdp_optimizer()`, `offload_fsdp_optimizer()`
- `drkernel/verl_patch/utils/cuda_memory_debug.py` — debug snapshot helpers
- `drkernel/kernel/scripts/rl/train_rl_common.sh` — token-cap auto-detection
- `drkernel/kernel/scripts/rl/14b_coldstart_trloo_mrs_pr_prs.sh` — 14B launcher

Key runs:

- Original reproduced OOM (token cap too high):
  - `drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.bf16.spare2.refcache.20260327-090301/`
- Lower-cap run, OOM at step=2 (optimizer-state baseline):
  - `drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.bf16.spare2.refcache.20260327-122333/`
- Fix validation run (resolved):
  - `drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.bf16.spare2.refcache.20260328-013235/`
