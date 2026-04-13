# Release `3a84417` vs Current Repo: High-Signal Differences

## Scope

This note compares:

- official release worktree: `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-official-3a84417`
- current workspace: `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018`

Current workspace HEAD during this review: `dc8afde`.

The focus is narrow:

- reward-side `reference_backend`
- PPO actor/ref `use_torch_compile`
- reward timing semantics (`num_perf_trials`, `num_warmup`, `perf_trim_count`)
- reference cache plumbing
- whether PRS/MRS core logic changed

This was a static code and log comparison only. No training job or evaluation job was launched.

## Executive Summary

1. The biggest release-to-current semantic change is on the reward-side reference timing path, not the PPO ref model path.
   - In official release `3a84417`, KernelGym reward reference timing is eager by default unless `reference_backend` is explicitly set.
   - In the current repo, DRKernel training defaults now push reward-side `reference_backend=torch_compile` end to end.

2. PPO actor/ref compile defaults in base YAML did not change, but the current launcher now overrides them.
   - Both release and current actor/ref config YAML still default `use_torch_compile: true`.
   - Current `train_rl_common.sh` injects `actor_rollout_ref.actor.use_torch_compile` and `actor_rollout_ref.ref.use_torch_compile` from shell variables whose defaults are empty strings.
   - The actor code treats falsey `use_torch_compile` as "do not compile".
   - The archived entropy-collapse run logged `use_torch_compile: ''` for actor/ref, while still logging `reference_backend: 'torch_compile'` for reward-side reference timing.

3. Reward timing semantics changed end to end.
   - Current repo adds `num_warmup` and `perf_trim_count` to task schema, server API, reward payloads, and timing statistics.
   - Current timing code can replace the primary reported mean with a trimmed mean.
   - Release training launcher only forwarded `num_perf_trials`; current launcher also forwards warmup and trim controls.

4. Reference cache is new in the current training path.
   - Current launcher, reward builder, and reward client all support `use_reference_cache`.

5. PRS/MRS core masking logic is not where the main release-to-current delta sits.
   - `drkernel/kernel/rewards/coverage_helper.py` is identical between release and current.
   - Focused diffing did not surface a high-signal change in the PRS/MRS sections of `kernel_trainer.py`.

## Detailed Differences

| Area | Official release `3a84417` | Current repo | Why it matters |
|---|---|---|---|
| Reward-side `reference_backend` default | `kernelgym/schema/task.py` keeps `reference_backend: Optional[str] = None`; README says eager is default; `pipeline.py` only compiles if a compile backend is explicitly requested | `drkernel/kernel/config/kernel_trainer.yaml` sets `reward_model.reference_backend: "torch_compile"`; `train_rl_common.sh` defaults `REFERENCE_BACKEND` to `torch_compile` and passes it into Hydra; `kernel_reward.py` also falls back to `torch_compile` | This is the clearest release-to-current semantic change on the reward reference-timing path |
| PPO actor/ref compile base config | `drkernel/verl_patch/trainer/code/config/actor/actor.yaml` defaults `use_torch_compile: true`; `ref/ref.yaml` inherits it | Same YAML defaults remain | Base YAML did not regress here by itself |
| PPO actor/ref compile launcher behavior | Release launcher does not inject actor/ref compile overrides | Current `train_rl_common.sh` explicitly injects `actor_rollout_ref.actor.use_torch_compile=$ACTOR_USE_TORCH_COMPILE` and `actor_rollout_ref.ref.use_torch_compile=$REF_USE_TORCH_COMPILE`; both shell defaults are empty strings | Actual run behavior can now diverge from YAML defaults |
| Archived entropy-collapse run config | Release run not inspected here | Archived current run logs `use_torch_compile: ''` twice for actor/ref and `reference_backend: 'torch_compile'` for reward-side reference timing | Important because the run used mixed semantics: reward reference compiled, actor/ref log-prob path uncompiled |
| Reward timing knobs | Release training path forwards `num_perf_trials` only | Current repo adds `num_warmup` and `perf_trim_count` through schema, API, reward builder, client, and timing functions | Timing distributions and the reported mean can shift even if model behavior is unchanged |
| Timing statistic definition | `get_timing_stats()` reports raw mean/std/min/max | Current `get_timing_stats(..., trim_count=...)` can compute `trimmed_mean`, store `raw_mean`, and overwrite primary `mean` with the trimmed value | This changes the semantics of the reward-side performance metric |
| Reference cache | Release reward path builds tasks with `use_reference_cache: False` and calls client with `use_reference_cache=False` | Current repo threads `reference_cache.enable` from launcher into reward tasks and client payloads | Can change runtime behavior and performance stability, though not obviously the masking logic itself |
| 14B run script defaults | Release `14b_trloo_mrs_pr_prs.sh` uses `NUM_PERF_TRIALS=100`, `PROMPT_OVERSAMPLING_FACTOR=1.0`, `ACTOR_PARAMETER_OFFLOAD=True`, `PPO_MICRO_TOKEN=null` | Current `14b_coldstart_trloo_hfsdp8_refcache.sh` sets `REFERENCE_BACKEND=\"torch_compile\"`, `NUM_PERF_TRIALS=50`, `NUM_WARMUP=5`, `PERF_TRIM_COUNT=5`, `PROMPT_OVERSAMPLING_FACTOR=1.7`, `ACTOR_PARAMETER_OFFLOAD=False`, `PPO_MICRO_TOKEN=8192`, `REFERENCE_CACHE_ENABLE=true`, `ACTOR_USE_TORCH_COMPILE=\"\"`, `WANDB_MODE=disabled`, `TRAINER_LOGGERS=['console']` | There are multiple run-surface differences; not all are causal, but they mean current experiments are not directly comparable to release runs |

## Evidence

### 1. Reward-side reference timing changed from eager-default to compile-default

Official release:

- `kernelgym/schema/task.py:23` and `:60` keep `reference_backend: Optional[str] = None`
- `README.md:462-475` documents eager mode as the default unless `reference_backend` is set
- `kernelgym/toolkit/kernelbench/pipeline.py:633-642` only applies `torch.compile(model)` when `reference_backend` is provided
- `drkernel/kernel/scripts/rl/train_rl_common.sh:770-780` forwards `reward_model.num_perf_trials` but not `reward_model.reference_backend`

Current repo:

- `drkernel/kernel/config/kernel_trainer.yaml:18` sets `reference_backend: "torch_compile"`
- `drkernel/kernel/scripts/rl/train_rl_common.sh:221` defaults `REFERENCE_BACKEND=${REFERENCE_BACKEND:-"torch_compile"}`
- `drkernel/kernel/scripts/rl/train_rl_common.sh:875` forwards `reward_model.reference_backend=$REFERENCE_BACKEND`
- `drkernel/kernel/rewards/kernel_reward.py:133` falls back to `getattr(reward_config, "reference_backend", "torch_compile")`

### 2. PPO actor/ref compile base defaults did not change, but launcher behavior did

Both release and current:

- `drkernel/verl_patch/trainer/code/config/actor/actor.yaml:100` sets `use_torch_compile: true`
- `drkernel/verl_patch/trainer/code/config/ref/ref.yaml:6` inherits from actor via `oc.select`

Current launcher behavior:

- `drkernel/kernel/scripts/rl/train_rl_common.sh:180-181` defaults `ACTOR_USE_TORCH_COMPILE` and `REF_USE_TORCH_COMPILE` to empty strings
- `drkernel/kernel/scripts/rl/train_rl_common.sh:809` and `:850` inject those values into Hydra
- `drkernel/verl_patch/workers/code/actor/dp_actor.py:131-142` uses `if self.config.get("use_torch_compile", True)`, so a falsey empty string disables compilation

Archived run evidence:

- `drkernel/logs/trloo-14b-hfsdp8-refcache.train.16xA800.reward.16x4090.run.20260404-032344/trainer.log:84`
- `drkernel/logs/trloo-14b-hfsdp8-refcache.train.16xA800.reward.16x4090.run.20260404-032344/trainer.log:138`
- `drkernel/logs/trloo-14b-hfsdp8-refcache.train.16xA800.reward.16x4090.run.20260404-032344/trainer.log:498`

Those lines show:

- actor/ref `use_torch_compile: ''`
- reward-side `reference_backend: 'torch_compile'`

So the archived run was not "compile everywhere". It was closer to:

- reward reference timing: compiled
- PPO actor/ref log-prob helper path: uncompiled

### 3. Timing semantics changed end to end

Schema/API additions in current repo:

- `kernelgym/schema/task.py:19-20`, `:58-59`, `:89-90` add `num_warmup` and `perf_trim_count`
- `kernelgym/schema/simple_task.py:18-19` adds the same fields
- `kernelgym/server/api/models.py:21-22` adds API validation for `num_warmup` and `perf_trim_count`

Reward/task plumbing in current repo:

- `drkernel/kernel/rewards/kernel_reward.py:126-127` reads `num_warmup` and `perf_trim_count`
- `drkernel/kernel/rewards/kernel_reward.py:159-160` places them into per-task payloads
- `drkernel/kernel/rewards/reward_client.py:694-695` forwards them to the reward server

Timing behavior changes in current repo:

- `kernelgym/toolkit/kernelbench/timing.py:141-159` adds trimmed statistics and can overwrite primary `mean`
- `kernelgym/toolkit/kernelbench/pipeline.py:691-728` records reference-backend compile timing and reference performance metadata
- `kernelgym/toolkit/kernelbench/pipeline.py:129-183` and `:365-595` add warmup/trim-aware kernel performance timing and more metadata

Training defaults:

- official `drkernel/kernel/config/kernel_trainer.yaml:44` uses `num_perf_trials: 100`
- current `drkernel/kernel/config/kernel_trainer.yaml:45-47` uses `num_perf_trials: 50`, `num_warmup: 30`, `perf_trim_count: 5`

Run-script defaults:

- official `drkernel/kernel/scripts/rl/14b_trloo_mrs_pr_prs.sh:38`, `:80`, `:92`
- current `drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_refcache.sh:55`, `:81-83`, `:150`

### 4. Reference cache is new on the training reward path

Release:

- `drkernel/kernel/rewards/kernel_reward.py:146` uses `use_reference_cache: False`
- `drkernel/kernel/rewards/kernel_reward.py:184` calls the client with `use_reference_cache=False`

Current:

- `drkernel/kernel/rewards/kernel_reward.py:152` uses `use_reference_cache: use_ref_cache`
- `drkernel/kernel/rewards/kernel_reward.py:192` calls the client with `use_reference_cache=use_ref_cache`
- `drkernel/kernel/rewards/reward_client.py:711-713` forwards `use_reference_cache`
- `drkernel/kernel/scripts/rl/train_rl_common.sh:887-888` wires `reward_model.reference_cache.*`
- `drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_refcache.sh:163` enables it

### 5. PRS/MRS core logic is not the main changed surface

Static comparison result:

- `drkernel/kernel/rewards/coverage_helper.py` is identical between release and current
- focused diffing did not surface a high-signal release-to-current change in the PRS/MRS-related sections of `drkernel/kernel/kernel_trainer.py`
- focused diffing also did not surface a high-signal change around `is_decoy_kernel`, `time_coverage`, or related reward-info keys in `drkernel/kernel/workers/reward_manager/kernel_async.py`

This matters because it pushes the likely release-to-current delta toward:

- changed reward/reference timing behavior
- changed launcher defaults
- changed timing statistics

rather than toward a newly introduced PRS formula bug.

## Interpretation For Entropy-Collapse Analysis

The highest-signal release-to-current differences are:

1. reward-side `reference_backend` moved from implicit eager-default to explicit compile-default
2. reward timing semantics changed from raw `100`-trial means toward warmup-aware and trim-capable timing
3. current runs can disable PPO actor/ref compile even though base YAML still says `true`
4. reference cache is now part of the reward path

The strongest safe interpretation is:

- current entropy-collapse analysis should not be framed as "the official release also used compile everywhere"
- the clearest release-to-current regression candidate is the reward-side reference timing stack plus the changed timing policy
- the archived run itself mixed settings: reward reference timing used `torch_compile`, while PPO actor/ref helper compile was effectively off
- PRS masking can still be the direct collapse mechanism, but the upstream input distribution that triggered it is more plausibly tied to reward/timing/config changes than to a rewritten PRS formula

What this comparison does **not** prove:

- it does not prove `torch_compile` is the sole cause of entropy collapse
- it does not prove warmup/trim/reference-cache changes are causal on their own

Those still need A/B validation with controlled runs.
