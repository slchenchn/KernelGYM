# Reward Timing and `num_perf_trials` Handoff

## Goal

Reduce reward-side performance measurement cost without changing the reward decision semantics in a way that destabilizes training.

The immediate tuning target is:

- current config: `num_perf_trials: 100`
- desired outcome: materially lower reward latency with no meaningful increase in reward-label flips near the current speedup threshold

## Current Bottom Line

1. Training reward does **not** use wall-clock end-to-end time for speedup.
2. Training reward does **not** use continuous speedup as a regression target.
3. The current performance reward is a **binary threshold** on whether `speedup >= 1.01`.
4. Because of that, `num_perf_trials` should be reduced based on **decision stability** against a `100`-trial baseline, not just on mean runtime drift.
5. Current remote 4090 reward nodes are **not visibly frequency-locked**.
6. Historical slowdown counters on the checked reward node point to **power-cap pressure**, not thermal throttling.

## What We Verified

### Reward decision semantics

Training uses `calculate_reward_weighted` with:

- `reward_func_name: "calculate_reward_weighted"`
- `speedup_eps: 0.01`
- correctness weight `0.5`
- performance weight `0.5`

That means the performance half of reward is based on:

- `is_speedup_positive = speedup >= (1 + speedup_eps)`
- current threshold: `speedup >= 1.01`

Relevant files:

- [kernel_trainer.yaml](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/config/kernel_trainer.yaml#L17)
- [reward_client.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/rewards/reward_client.py#L458)

Implication:

- the key failure mode is **label flip around the 1.01 threshold**
- using only average runtime error as a tuning metric is the wrong optimization target

### What timing is used for reward speedup

Reward speedup is derived from:

- `reference_runtime / kernel_runtime`

Those runtimes come from CUDA-event timing:

- warmup first
- then timed trials with `torch.cuda.Event`
- final `kernel_runtime` / `reference_runtime` use the mean of the timed trials

Relevant files:

- [result.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/schema/result.py#L181)
- [timing.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/timing.py#L17)
- [pipeline.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/pipeline.py#L154)
- [pipeline.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/pipeline.py#L698)

Implication:

- current reward speedup is based on **CUDA time**, not service wall-clock
- queue delay and general reward-manager wall-clock are still important for throughput, but they are **not** the speedup label source

### Whether compile time is included

For the timed-trial mean used in reward speedup:

- compile / load is tracked separately
- it is **not** part of the final mean used for `speedup`

Kernel side:

- compile/load is tracked as `kg_kernel_compile_and_load_s`
- timed execution comes later

Reference side:

- `torch.compile` wrapping time is tracked as `kg_reference_backend_compile_s`
- timed execution comes later

Relevant files:

- [pipeline.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/pipeline.py#L495)
- [pipeline.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/pipeline.py#L689)
- [pipeline.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/pipeline.py#L701)

Important nuance:

- first-execution JIT work can still land in **warmup**
- warmup and other wall-clock metrics therefore still reflect compile-side cost
- but the final timed-trial mean used for reward speedup is intended to exclude it

### Remote reward node clock state

Checked reward nodes from `SPEC.md`:

- `192.168.16.39`
- `192.168.16.40`

Observed on both:

- GPU type: `NVIDIA GeForce RTX 4090`
- `Applications Clocks`: `N/A`
- `Default Applications Clocks`: `N/A`
- `Persistence Mode`: `Disabled`
- sampled current clocks were idle `P8` values

Implication:

- current reward measurements should **not** be treated as already benefiting from explicit clock locking

### What likely limits fixed-clock stability

From `clocks_event_reasons` on the checked 4090 reward node:

- `hw_thermal_slowdown`: no visible accumulated time
- `sw_thermal_slowdown`: no visible accumulated time
- `sw_power_cap`: non-zero accumulated time on all GPUs

Current power limit on both reward nodes:

- `power.limit = 450 W`
- `power.max_limit = 450 W`

Implication:

- on these nodes, the first stability concern is more likely **power cap** than thermal cap
- locking too high can increase variance by repeatedly hitting `sw_power_cap`

## What This Means for `num_perf_trials`

### Wrong way to choose it

Do **not** choose a lower value by asking only:

- whether mean runtime changed only a little
- whether a few sampled speedup numbers still look similar

That misses the actual reward contract.

### Right way to choose it

Choose the smallest value whose **binary speedup decision** remains stable against the current `100`-trial baseline.

Primary metric:

- agreement of `is_speedup_positive` vs `100`-trial baseline

Secondary metrics:

- false positive rate
- false negative rate
- mean reward drift
- reward-side latency reduction

Suggested acceptance criteria:

- agreement `>= 99%`
- false positive `< 0.5%`
- false negative `< 0.5%`
- mean reward drift `< 1%`

## Recommended Experiment Plan

### Phase 1: fixed-clock validation

Before changing `num_perf_trials`, validate whether explicit clock control lowers runtime variance on the reward nodes.

Recommended first test:

- lock GPU core clock to `2700 MHz`

Why not start at the max:

- the checked 4090s already show historical `sw_power_cap`
- `450 W` is both the default and maximum power limit
- a too-high fixed core clock is more likely to produce power-cap oscillation than lower variance

Commands:

```bash
# set fixed core clock
nvidia-smi -lgc 2700,2700

# restore default later
nvidia-smi -rgc
```

Monitor during live reward load:

```bash
watch -n 1 'nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown --format=csv'
```

Interpretation:

- if `clocks.current.graphics` stays close to target and `sw_power_cap` stays inactive, the lock is sustainable
- if `sw_power_cap` frequently becomes active or effective clock repeatedly drops, lower the target by `60-120 MHz`

Recommended ladder:

- `2700`
- if stable, `2760`
- if still stable, optionally `2820`
- avoid starting at `2850+`

### Phase 2: trial-count calibration

Use a fixed sample set of recent valid reward cases:

- only samples with `compiled=True`
- only samples with `correctness=True`
- recommended sample count: `200-500`

For each sample, compare:

- `100` trials
- `30` trials
- `10` trials

Recommended order:

1. establish `100`-trial baseline
2. test `30`
3. if `30` is stable enough, test `10`

For each candidate value, compute:

- `is_speedup_positive` agreement vs `100`
- false positives
- false negatives
- average reward drift
- average reward latency reduction

Likely outcome:

- a stable reduction from `100` to `30` is plausible
- `10` may also work if most samples are not packed near `1.01`
- the final answer depends on how dense the sample distribution is around the threshold

### Phase 3: optional adaptive trials

If static reduction still leaves too much cost on easy cases, switch to an adaptive policy:

1. run `5` or `10` trials first
2. if speedup is clearly above threshold, accept early
3. if speedup is clearly below threshold, reject early
4. only samples near the threshold get promoted to `30` or `100`

This is the most natural fit for the current reward contract because the decision boundary is binary and tight.

Example policy:

- if early mean speedup `> 1.05`, mark positive
- if early mean speedup `< 0.97`, mark negative
- otherwise escalate

Exact cutoffs should be derived from the calibration dataset rather than hard-coded blindly.

## Secondary Optimization Ideas

These were discussed, but are not the first change to make:

- enable explicit reference cache in training reward path and measure hit rate
- use reward queue metrics to decide whether `REWARD_MAX_CONCURRENT` should rise above `32`
- consider split placement / async overlap only after the cheaper reward-side measurement wins are exhausted

These can help overall throughput, but they do not directly answer the `num_perf_trials` decision boundary problem.

## Important Caveats

### CPU load still matters, but less than before

If timing is based on CUDA events and each test owns an entire GPU, CPU noise matters less than in wall-clock timing. But it is not zero:

- host scheduling still affects launches and sync points
- warmup and wall-clock totals still see host-side noise
- service orchestration still affects throughput even if it does not change the speedup label directly

### CUDA Graph is not a full solution by itself

CUDA Graph can reduce host launch jitter if both sides are measured symmetrically, but it does not eliminate:

- GPU frequency / power variance
- wall-clock overhead outside replay
- boundary effects if only one side uses graph replay

It should be treated as a separate benchmarking refinement, not as a substitute for trial-count calibration.

## Current Config and Relevant Files

- reward config:
  - [kernel_trainer.yaml](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/config/kernel_trainer.yaml#L45)
- reward task assembly:
  - [kernel_reward.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/rewards/kernel_reward.py#L124)
- reward decision logic:
  - [reward_client.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/rewards/reward_client.py#L445)
- kernel timing:
  - [timing.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/timing.py#L17)
- kernel perf pipeline:
  - [pipeline.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/pipeline.py#L154)
- reference perf pipeline:
  - [pipeline.py](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/pipeline.py#L678)

## Recommended Next Action

If only one experiment can be run next, do this:

1. lock the reward-node 4090s to `2700 MHz`
2. collect a `200-500` sample calibration set
3. compare `100` vs `30`
4. if agreement is clean enough, compare `30` vs `10`

That is the shortest path to turning the current discussion into an actual reduction in reward latency.
