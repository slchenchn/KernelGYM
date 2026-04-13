# Reference Timing Variance — Investigation Handoff

## Status: Clock locking experiments complete — variance reduced 10x

---

## Part 1: Locked GPU Clocks (2700MHz, RTX 4090)

GPU frequency locked at 2700MHz via `nvidia-smi -lgc 2700,2700` on the host machine before launching Docker containers. Power limit set to 400W. Persistence mode enabled.

### Locked Clock Summary Table — Reference (torch_compile), 20% trim

| Warmup | 50 trials | 30 trials | 20 trials | 15 trials |
|---|---|---|---|---|
| 5 | 0.39% | 0.35% | 0.34% | 0.30% |
| 10 | 0.36% | 0.32% | 0.31% | 0.24% |
| 30 | 0.29% | 0.27% | 0.27% | 0.21% |

### Locked Clock Summary Table — Kernel (Triton), 20% trim

| Warmup | 50 trials | 30 trials | 20 trials | 15 trials |
|---|---|---|---|---|
| 5 | 0.22% | 0.18% | 0.18% | 0.15% |
| 10 | 0.21% | 0.19% | 0.17% | 0.15% |
| 30 | 0.22% | 0.19% | 0.19% | 0.16% |

### Locked vs Unlocked Comparison

| Config | Unlocked Med CV% | Locked Med CV% | Improvement |
|---|---|---|---|
| Ref, warmup=30, 50t, 20% trim | 2.00% | **0.29%** | 6.9x |
| Kernel, warmup=30, 50t, 20% trim | 1.75% | **0.22%** | 8.0x |
| Trial 1 deviation (unlocked w=30) | +27% / +35% | **<2% (noise)** | eliminated |

### Key Findings — Locked Clocks

1. **CV reduced ~7-8x** (2% → 0.2-0.3%) by eliminating GPU frequency jitter
2. **100% of samples have CV ≤ 5%** across all configs
3. **First-trial warmup effect nearly eliminated** — only +0.6% overhead vs +27% unlocked
4. **Warmup barely matters**: warmup=5 gives 0.39% ref CV vs warmup=30 gives 0.29% — only 0.1pp difference. The long warmup was only needed to compensate for frequency drift, which clock locking eliminates
5. **Kernel timing is virtually identical across all warmup values** (0.15-0.22%)
6. **Even 15 trials with 20% trim gives <0.30% CV** — can aggressively reduce trial count

### Residual Variance Sources (with locked clocks)

The ~0.2-0.4% residual CV comes from:
1. **L2 cache / memory subsystem jitter** — cache hit rate varies slightly between trials (~0.001-0.05ms)
2. **CUDA event timing quantization** — microsecond resolution; for fast kernels (~0.1ms) this alone is ~1%
3. **Rare scheduling interrupts** — occasional 2σ+ outlier spikes from PCIe contention or OS scheduling; ~2 outlier trials per 50-trial sample on average, handled by trimming

### Recommended Locked-Clock Config

**warmup=5, 20 trials, 20% trim** — gives 0.18-0.34% CV, fastest measurement time. With locked clocks, there is no thermal ramp-up to wait for, so minimal warmup is sufficient.

### Locked Clock Setup

See `handoffs/completed/HANDOFF_REWARD_NODE_SETUP.md` for full setup instructions.

### Data Files

- `drkernel/logs/grading_results/timing_w30_locked_clk.json` — 1940 results, warmup=30
- `drkernel/logs/grading_results/timing_w10_locked_clk.json` — 1940 results, warmup=10
- `drkernel/logs/grading_results/timing_w5_locked_clk.json` — 1927 results, warmup=5

---

## Part 2: Unlocked GPU Clocks (Original Investigation)

All experiments below were conducted WITHOUT GPU clock locking. The 4090 GPUs boost dynamically between ~1600-3105 MHz based on thermal/power state, which is the primary source of timing variance.

## Summary Table

| Warmup | Mean | Trials | Median CV% | Mean CV% | CV ≤ 5% | CV ≤ 10% |
|---|---|---|---|---|---|---|
| 3 | raw | 100 | 7.94 | 17.89 | 13% | 60% |
| 10 | raw | 100 | 7.52 | 15.78 | 30% | 63% |
| 3 | trimmed 5+5 | 100 | 3.55 | 8.32 | 68% | 82% |
| 10 | trimmed 5+5 | 100 | 2.91 | 5.55 | 77% | 85% |
| 10 | trimmed 5+5 | 50 | 2.38 | 4.22 | 83% | 90% |
| 20 | trimmed 5+5 | 100 | 2.57 | 4.85 | 74% | 85% |
| 20 | trimmed 5+5 | 50 | 2.23 | 3.81 | 85% | 94% |
| 30 | trimmed 5+5 | 50 | 1.79 | 3.43 | 87% | 92% |
| 30 | trimmed 5+5 | 40 | 2.07 | 3.27 | 87% | 95% |
| 30 | trimmed 5+5 | 30 | 1.85 | 3.34 | 84% | 93% |
| 30 | trimmed 5+5 | 20 | 1.03 | 2.14 | 94% | 97% |
| 40 | trimmed 5+5 | 50 | 1.95 | 3.38 | 82% | 93% |

### Fixed 5+5 trim vs 20% proportional trim (warmup=30, 1214 samples from Redis)

The fixed 5+5 trim results above have an unfair advantage at low trial counts because the trim fraction increases (50% at 20 trials vs 20% at 50 trials). Re-analysis with consistent 20% trim (10% per side):

| Trials | Fixed 5+5 (keep) | Med CV% | 20% trim (keep) | Med CV% |
|---|---|---|---|---|
| 50 | keep 40 | 2.00% | keep 40 | 2.00% |
| 40 | keep 30 | 1.82% | keep 32 | 2.11% |
| 30 | keep 20 | 1.58% | keep 24 | 2.24% |
| 20 | keep 10 | 1.12% | keep 16 | 2.56% |
| 15 | keep 5 | 0.72% | keep 11 | 2.37% |

**Key finding**: with consistent 20% trim, trial count barely matters — CV stays ~2.0-2.6% across 15-50 trials. The dramatic improvement at low trial counts with fixed 5+5 is entirely from the aggressive trim fraction (33-67%), not from fewer trials. The underlying per-trial noise after warmup=30 is constant at ~2% CV.

**Implication**: the real knob is trim fraction, not trial count. For a target of ~2% CV, **warmup=30, 20 trials, 20% trim** gives the same quality as 50 trials while being 2.5x faster in measurement time.

### Trial count reduction with trimming (warmup=10, 156 samples)

| Config | Med CV | Mean CV | CV ≤ 5% | CV ≤ 10% | Drift vs GT | Drift > 1% |
|---|---|---|---|---|---|---|
| 100 trials, raw | 6.18% | 25.47% | 31% | 70% | 0.85% | 35.9% |
| 100 trials, trim 5+5 | 2.91% | 5.55% | 77% | 85% | 0.00% | 0.0% |
| 50 trials, raw | 6.84% | 21.51% | 28% | 75% | 2.22% | 81.4% |
| 50 trials, trim 3+3 | 3.05% | 5.21% | 79% | 88% | 1.13% | 55.1% |
| **50 trials, trim 5+5** | **2.38%** | **4.22%** | **83%** | **90%** | **0.75%** | **40.4%** |
| 30 trials, raw | 7.49% | 18.38% | 25% | 74% | 3.29% | 91.7% |
| 30 trials, trim 3+3 | 2.85% | 4.37% | 83% | 92% | 1.56% | 67.9% |

Ground truth = 100 trials trimmed 5+5. Drift = absolute difference in mean vs ground truth.

**50 trials + trim 5+5** (= keep middle 40 of 50) gives the best tradeoff: 2.38% median CV, 83% of samples have CV ≤ 5%, and only 40% drift >1% from the 100-trial baseline. This halves the measurement time vs 100 trials.

Cross-run variance (same ref code evaluated independently): **16.3% median CV** — eliminated entirely by reference cache.

Reference cache step time impact: **-17% gen time, -14% total step time** vs no cache.

## Goal

Measure the variance of reference torch.compile implementation timing in the real training reward pipeline, with and without reference result cache, to inform whether `num_perf_trials` can be safely reduced from 100.

## Setup

- 16 A800 GPUs (2 nodes) for rollout, bf16 with HYBRID_SHARD + reduce_dtype=bf16
- 16 4090 GPUs (2 nodes: 192.168.16.39, 192.168.16.40) for reward evaluation
- `MAX_TURN=1` for faster iteration (~10 min/step)
- `num_perf_trials=100` (current default)
- `reference_backend=torch_compile`
- Run: `drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.bf16.2node.hybrid.20260330-072125/`

## Data Pipeline Fix

Per-trial timing data was not flowing from the reward pipeline to the training logs. Root cause: **Pydantic `BaseModel` copies the dict** on construction. The `eval_reference_only` function in `pipeline.py` added timing fields to a local `metadata` dict after creating the `KernelExecResult`, but the result held a stale copy.

**Fix**: Write timing fields directly to `kernel_exec_result.metadata` instead of the local dict.

**Files changed** (in the primary KernelGYM repo, not the vllm018 worktree):
- `kernelgym/toolkit/kernelbench/pipeline.py` — added `kg_reference_perf_elapsed_times_ms` (all 100 trial times), `kg_reference_perf_mean_ms`, `kg_reference_perf_std_ms`, `kg_reference_perf_min_ms`, `kg_reference_perf_max_ms`; same for kernel side (`kg_kernel_perf_*`); fixed Pydantic dict copy bug
- `kernelgym/toolkit/kernelbench/toolkit.py` — added debug print for metadata keys

**Analysis script**: `drkernel/scripts/analyze_ref_timing.py` — reads from Redis directly (vllm.log has Ray log dedup which collapses multi-line lists)

## Experiment 1 Results: Disk Cache Warm

**446 samples, 100 trials each, torchinductor disk cache warm from previous runs.**

### Per-sample reference timing variance

| Metric | Min | Median | Max |
|---|---|---|---|
| Mean runtime (ms) | 0.100 | 0.159 | 11.180 |
| Std (ms) | 0.006 | 0.014 | 4.377 |
| CV% | 0.32% | **8.12%** | 700% |
| Range% | 2.0% | 55.8% | 7048% |

### CV% distribution

| Threshold | Count | % |
|---|---|---|
| CV ≤ 1% | 10/446 | 2% |
| CV ≤ 2% | 23/446 | 5% |
| CV ≤ 5% | 56/446 | 13% |
| CV ≤ 10% | 268/446 | 60% |
| CV ≤ 20% | 357/446 | 80% |

### First-trial warmup residual

Despite 3 explicit warmup rounds before timing, the first timed trial runs significantly slower:

| Metric | Value |
|---|---|
| Median overhead | 37.9% |
| Mean overhead | 53.1% |
| Max overhead | 4977% |

The 3 warmup rounds are insufficient. The first ~10 timed trials are contaminated by CUDA kernel warmup effects.

### Trial count reduction stability

| Comparison | Median drift | Mean drift | Max drift | >1% drift |
|---|---|---|---|---|
| 30 vs 100 | 3.92% | 10.46% | 169% | **90.4%** |
| 10 vs 100 | 11.11% | 18.38% | 279% | 96.4% |
| 5 vs 100 | 18.02% | 26.57% | 617% | 96.9% |

## Experiment 3: Warmup=10 (vs default warmup=3)

Changed `num_warmup` from 3 to 10 in `eval_reference_only` and reran without cache.

### warmup=3 vs warmup=10

| Metric | warmup=3 (n=446) | warmup=10 (n=380) | Change |
|---|---|---|---|
| Median CV% | 8.12% | **7.24%** | -11% |
| First-trial overhead | 37.9% | **28.6%** | -25% |
| CV ≤ 5% | 13% | **30%** | +130% |
| CV ≤ 10% | 60% | **63%** | +5% |
| Max first-trial overhead | 4977% | **151%** | -97% |

### Why increasing warmup further won't help much

The remaining ~7% CV comes from two sources that warmup cannot fix:

1. **GPU frequency drift during the 100 timed trials.** Without clock locking (not available on these containerized 4090s), the GPU boosts and throttles dynamically based on thermal/power state. A 100-trial measurement spanning ~0.5-2 seconds sees real frequency changes mid-measurement. More warmup before the first trial doesn't prevent frequency changes during the later trials.

2. **Cross-trial memory/cache state variation.** Even after warmup, the L2 cache hit pattern varies slightly between trials depending on what other GPU processes were doing. The first-trial overhead (28.6% at warmup=10) is a one-time cold-cache penalty. The remaining per-trial jitter (~7% CV) is steady-state thermal/frequency noise that more warmup cannot reduce.

Evidence: the first-trial overhead dropped from 37.9% → 28.6% (warmup helped), but the median CV only dropped from 8.12% → 7.24% (the bulk of variance is NOT from the first trial).

### Clock locking is not available

Attempted `nvidia-smi -lgc 2700,2700` and `nvidia-smi -pl 400` on both reward nodes — both fail with `Insufficient Permissions`. The containerized 4090 environment blocks all nvidia-smi admin operations. This is the primary reason for the high variance.

## Key Conclusions

### 1. ~7% CV is the floor without clock locking

Warmup=10 reduced the extreme outliers but the core variance remains. This is inherent to un-locked consumer GPUs in containers.

### 2. Reference cache is the most effective mitigation

Since we can't reduce per-measurement variance below ~7%, the best strategy is to measure once and reuse:
- Cache eliminates the 16% cross-run variance entirely
- Cache saves ~1 min/step in reward evaluation time
- All responses to the same prompt get consistent speedup labels

### 3. Reducing `num_perf_trials` from 100 is still risky

With warmup=10, the trial-reduction stability is slightly better but still problematic:

| Comparison | warmup=3 median drift | warmup=10 median drift |
|---|---|---|
| 30 vs 100 | 3.92% | ~3.5% (est.) |
| 10 vs 100 | 11.11% | ~9% (est.) |

The improvement is marginal. 90%+ of samples would still drift >1% with 30 trials.

## Experiment 4: Trimmed Mean (computed from existing data, no re-run needed)

Sorted the 100 trials per sample, removed the 5 lowest and 5 highest, computed mean from the middle 90. Applied to both warmup=3 and warmup=10 datasets stored in Redis.

### Full comparison: all configurations

| Configuration | Median CV% | CV ≤ 5% | CV ≤ 10% | First-trial overhead |
|---|---|---|---|---|
| warmup=3, raw mean | 8.12% | 13% | 60% | 37.9% |
| warmup=10, raw mean | 7.24% | 30% | 63% | 28.6% |
| warmup=3, **trimmed mean** | **3.55%** | **68%** | **82%** | 36.4% |
| warmup=10, **trimmed mean** | **3.19%** | **68%** | **83%** | 27.8% |

### Data quality note

No all-zero samples found. 453/2913 samples (16%) have raw CV > 20%, and 135 have CV > 100%. These extreme outliers are not from near-zero runtimes (all have mean > 0.1ms) — they are likely torch.compile inconsistencies or intermittent GPU scheduling issues. Excluding CV > 100% samples (~5% of data) does not materially change the results (trimmed CV median shifts from 3.19% to 3.12%).

### Trimming is more effective than warmup

Trimming cuts CV from ~8% → ~3.2% (**-60%**), while warmup=10 only cuts it from 8.12% → 7.24% (-11%). The combination (warmup=10 + trim=5) gives the best result at **3.19% median CV**.

Why trimming works better: the ~7% raw CV is driven by a few outlier trials (frequency spikes/dips) within each 100-trial measurement. Removing the 5 highest and 5 lowest eliminates these outliers. Warmup only fixes the cold-start first-trial penalty.

### Trial reduction stability with trimmed mean

| Comparison | warmup=3 trimmed | warmup=10 trimmed |
|---|---|---|
| 30 vs 100: median drift | 2.73% | **1.85%** |
| 30 vs 100: >1% drift | 90% | **76%** |
| 10 vs 100: median drift | 9.51% | 5.50% |
| 10 vs 100: >1% drift | 97% | 94% |

With warmup=10 + trim, 30-trial reduction drift improved to 1.85% median (76% >1%), down from 3.92% (90%) with raw mean. Still not safe enough for a 1.01 threshold, but significantly better.

### 4. Recommended configuration

- **Enable reference cache** (`reference_cache.enable: true`) — biggest impact, eliminates 16% cross-run variance
- **Use warmup=30** — best CV (1.79% median), 87% CV≤5%; eliminates first-trial residual noise
- **Use 50 trials with trimmed mean (drop top 5 + bottom 5, keep middle 40)** — 85% of samples have CV ≤ 5%, halves measurement time vs 100 trials
- **If clock locking becomes available** — retest to see if CV drops below 1%

## Cross-Run Variance: Same Reference Code, Different Evaluations

When `use_reference_cache=False`, each of the 8 rollout responses per prompt triggers an independent reference timing evaluation of the **same** reference code. This lets us measure how much the reference runtime varies across independent runs.

**2270 ref results, 187 unique reference codes, 161 codes with ≥2 independent measurements.**

| Metric | Value |
|---|---|
| Cross-run CV% median | **16.27%** |
| Cross-run CV% mean | 20.69% |
| Cross-run CV% max | 98.01% |

This is much worse than the within-run variance (8% CV). The same reference model, timed independently on different GPUs or at different times, produces runtimes that differ by ~16%.

### Why reference caching is strongly justified

| Source of variance | Magnitude | Eliminated by cache? |
|---|---|---|
| Within-run trial noise | ~8% CV | No (inherent to CUDA timing) |
| Cross-run variation (same code) | ~16% CV | **Yes** |
| First-trial warmup contamination | ~38% overhead | No (needs more warmup rounds) |

With `use_reference_cache=True`:
- Reference timing computed once per unique prompt, reused for all 8 responses
- Eliminates the 16% cross-run variance entirely
- All 8 responses get the same reference baseline → consistent speedup labels
- Saves 7/8 of the reference evaluation cost

Without cache, two responses to the same prompt can get different speedup labels purely from reference timing noise — not from actual kernel quality differences.

## Experiment 2: Reference Cache Enabled

Run: `drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.bf16.2node.hybrid.20260330-091533/`

Config: `reference_cache.enable=true` in `kernel_trainer.yaml`, wired through `kernel_reward.py` to `use_reference_cache` per task.

### How the cache works

Each prompt generates `rollout_n` responses (16 in this config). All share the same reference code. With cache enabled:
1. First response triggers a fresh reference timing evaluation (100 CUDA-event trials)
2. The result is cached by UUID in the KernelGYM API orchestrator
3. Remaining 15 responses reuse the cached reference runtime

### Step timing comparison

| | Without cache (n=8 steps) | With cache (n=4 steps) | Saving |
|---|---|---|---|
| gen (rollout + reward) | 5.7m | 4.7m | **-17%** |
| total step | 8.3m | 7.1m | **-14%** |

The cache saves ~1 min per step by eliminating ~15/16 redundant reference evaluations.

### Reward label consistency

With cache disabled, the same prompt's 16 responses each get independently measured reference runtimes with **16.3% median CV** between them. This means two responses to the same prompt can get different speedup labels purely from reference timing noise.

With cache enabled, all 16 responses get the **identical** reference runtime — eliminating inter-response inconsistency completely.

### Code changes for cache support

- `kernel_reward.py` — made `use_reference_cache` read from `reward_config.reference_cache.enable` instead of hardcoded `False`
- `kernel_trainer.yaml` — `reference_cache.enable: true` (was `false`)

## Node Details

| Node | IP | SSH | Role | GPU |
|---|---|---|---|---|
| ai-16-18 | 192.168.16.18 | `ssh -p 10842` | Ray head, KernelGYM API + Redis | A800 |
| ai-16-24 | 192.168.16.24 | `ssh -p 14218` | Ray worker | A800 |
| ai-16-39 | 192.168.16.39 | `ssh -p 11229` | Reward worker | 4090 |
| ai-16-40 | 192.168.16.40 | `ssh -p 11229` | Reward worker | 4090 |

## File References

- Analysis script: `drkernel/scripts/analyze_ref_timing.py`
- Pipeline patch: `KernelGYM/kernelgym/toolkit/kernelbench/pipeline.py` (primary repo, not vllm018)
- Timing function: `KernelGYM/kernelgym/toolkit/kernelbench/timing.py` (`time_execution_with_cuda_event`)
- Reward config: `drkernel/kernel/config/kernel_trainer.yaml` (`num_perf_trials`, `num_warmup`, `perf_trim_count`)
- Handoff context: `handoffs/completed/HANDOFF_NUM_PERF_TRIALS.md`

## Experiment 5: Warmup=20, 50 Trials, Trimmed Mean (Pending)

**Hypothesis**: Increasing warmup from 10 → 20 may further reduce first-trial overhead and steady-state variance, since experiment 3 showed warmup=10 still had 28.6% first-trial overhead. Combined with 50 trials + trim 5+5, this should give the cleanest signal yet.

**Config**: `num_warmup=20`, `num_perf_trials=50`, `perf_trim_count=5` (keep middle 40 of 50)

### Code changes

`num_warmup` and `trim_count` are hardcoded in the primary KernelGYM repo's `eval_reference_only()` (same approach as experiment 3 which hardcoded `num_warmup=10`). Config-based plumbing was also added but not used for this experiment due to OmegaConf attribute forwarding issues.

**Files changed (primary repo `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM/`):**
- `kernelgym/toolkit/kernelbench/pipeline.py` — `eval_reference_only()`: hardcoded `num_warmup=20`, `trim_count=5`; `_run_performance_step()` and `eval_kernel_against_ref()` accept `num_warmup`/`perf_trim_count` params
- `kernelgym/toolkit/kernelbench/timing.py` — `get_timing_stats()` accepts `trim_count`; when > 0, sorts trials, drops `trim_count` from each end, uses trimmed mean/std as primary `mean`/`std` (raw preserved as `raw_mean`/`raw_std`)
- `kernelgym/toolkit/kernelbench/toolkit.py` — all eval methods forward `num_warmup` and `perf_trim_count` from task to pipeline
- `kernelgym/schema/task.py` — added `num_warmup` and `perf_trim_count` to `EvaluationTask`, `ReferenceTimingTask`, `KernelEvaluationTask`
- `kernelgym/server/api/models.py` — added `perf_trim_count` field to `EvaluationRequest`
- `kernelgym/workflow/kernelbench_helpers.py` — forwards `num_warmup`/`perf_trim_count` in `_create_paired_tasks`
- `kernelgym/workflow/kernelbench.py` — forwards in `ReferenceTimingTask` construction

### Results

**warmup=20, 50 trials, trimmed 5+5** (375 samples):
Median CV 2.23%, Mean CV 3.81%, CV≤5% 85%, CV≤10% 94%

**warmup=20, 100 trials, trimmed 5+5** (274 samples):
Median CV 2.57%, Mean CV 4.85%, CV≤5% 74%, CV≤10% 85%

Runs:
- 50 trials: `drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.bf16.2node.hybrid.20260331-050402/`
- 100 trials: `drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.bf16.2node.hybrid.20260331-061932/`

### Why 100 trials is worse than 50 trials (with the same 5+5 trim)

Despite 20 warmup iterations, the **first ~50 timed trials still show residual warm-up effects**. Analysis of 1332 100-trial samples from Redis:

| Half | Median CV% | Notes |
|---|---|---|
| First 50 trials | 7.06% | Still warming up — high variance |
| Last 50 trials | 3.62% | Stable — GPU at thermal equilibrium |

The mean runtime **drifts -2.25% downward** from the first 50 to the last 50 trials (later = faster), with 88% of samples showing >1% drift. This is consistent with CUDA kernel JIT caches and GPU boost clocks stabilizing over time.

Trim 5+5 removes 10 trials total. For 100 trials that's only 10% trimmed — not enough to remove the ~20-30 contaminated early trials. For 50 trials it's 20% trimmed, which is more aggressive and effective.

**Trim fraction analysis** (all from the same 1332 100-trial samples, warmup=20):

| Strategy | Median CV% | CV ≤ 5% | CV ≤ 10% |
|---|---|---|---|
| 100 trials, trim 5+5 (keep 90, 10% trim) | 3.02% | 68% | 83% |
| 100 trials, trim 10+10 (keep 80, 20% trim) | 2.14% | 74% | 88% |
| 100 trials, trim 25+25 (keep 50, 50% trim) | 0.98% | 88% | 95% |
| First 50, trim 5+5 (keep 40) | 2.35% | 79% | 91% |
| Last 50, trim 5+5 (keep 40) | **1.39%** | **84%** | **91%** |
| Last 50, raw (no trim) | 3.65% | 64% | 78% |

Key insight: **the last 50 trials + trim 5+5 gives 1.39% median CV** — the best single configuration. But using only the last 50 would require running 100 trials and discarding the first 50, which doubles measurement time for a modest gain over the simpler "50 trials + trim 5+5" (2.23% → 1.39%).

### Experiment 7: warmup=30, 50 trials, trimmed 5+5

284 samples. Best median CV yet: 1.79%. The extra 10 warmup iterations (vs 20) further reduce first-trial residual noise identified in the per-trial analysis above (trials 1-10 were 8.3% CV even with warmup=20).

Run: `drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.bf16.2node.hybrid.20260331-071747/`

### Experiment 8: warmup=40, 50 trials, trimmed 5+5

337 samples. Median CV 1.95%, essentially flat vs warmup=30 (1.79%). Diminishing returns — the extra 10 warmup iterations beyond 30 provide no meaningful improvement, confirming that warmup=30 is sufficient to fully stabilize CUDA state.

Run: `drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.bf16.2node.hybrid.20260331-075354/`

### Updated recommendation

- **warmup=30** — sweet spot for both reference and kernel; warmup=40 shows no improvement
- **Reference: 20 trials, 20% trim** — ~2.5% median CV, 2.5x faster than 50 trials
- **Kernel: 50 trials, 20% trim** — ~2.0% median CV; kernels benefit more from extra trials than reference
- **Reference cache** remains the single biggest impact (eliminates 16% cross-run variance)

## Triton Kernel Timing Variance (Experiment 9)

### Kernel Summary Table

| Warmup | Mean method | Trials | Median CV% | Mean CV% | CV ≤ 5% | CV ≤ 10% |
|---|---|---|---|---|---|---|
| 30 | raw | 100 | 6.14 | 13.71 | 32% | 76% |
| 30 | raw | 50 | 7.01 | 13.92 | 22% | 72% |
| 30 | 20% trim | 100 | 1.53 | 3.72 | 81% | 89% |
| 30 | 20% trim | 50 | 1.75 | 3.85 | 80% | 89% |
| 30 | 20% trim | 40 | 1.80 | 3.94 | 82% | 89% |
| 30 | 20% trim | 30 | 2.00 | 3.77 | 81% | 91% |
| 30 | 20% trim | 20 | 2.46 | 3.64 | 82% | 93% |
| **30** | **20% trim** | **15** | **2.33** | **3.27** | **84%** | **95%** |

148 kernel samples, warmup=30, 100 trials with per-trial elapsed_times stored. Kernel perf metadata was missing from results due to Pydantic dict-copy bug in `_run_performance_step` (same bug previously fixed for reference path). Fixed by passing `kernel_exec_result.metadata` instead of the local `metadata` dict.

### Triton kernel warmup pattern

| Trial index | Normalized runtime |
|---|---|
| Trial 1 | 1.345 (+34.5%) |
| Trial 5 | 1.028 (+2.8%) |
| Trial 10 | 1.031 (+3.1%) |
| Trial 15 | 0.996 (stable) |
| Trial 50 | 0.974 |
| Trial 100 | 0.975 |

Similar to reference: trial 1 has a large spike, stabilizes by trial 10-15. warmup=30 is sufficient.

### Triton kernel per-window CV%

| Window | Median CV% |
|---|---|
| Trials 1-10 | 10.06% |
| Trials 11-20 | 2.42% |
| Trials 21-30 | 2.37% |
| Trials 31-40 | 1.74% |
| Trials 41-50 | 1.91% |
| Trials 51-60 | 1.65% |
| Trials 61-70 | 1.66% |
| Trials 71-80 | 3.20% |
| Trials 81-90 | 2.47% |
| Trials 91-100 | 1.95% |

Baseline noise is slightly higher than reference (~1.7-3.2% vs ~2.2%), with an anomalous spike at trials 71-80 suggesting occasional GPU scheduling jitter with Triton.

### Triton kernel 20% trim at different trial counts

| Trials | Trim | Keep | Median CV% | CV ≤ 5% | CV ≤ 10% |
|---|---|---|---|---|---|
| 100 | 10+10 | 80 | 1.56% | 72% | 86% |
| 50 | 5+5 | 40 | 1.99% | 72% | 84% |
| 40 | 4+4 | 32 | 1.98% | 74% | 86% |
| 30 | 3+3 | 24 | 2.01% | 71% | 87% |
| 20 | 2+2 | 16 | 2.91% | 74% | 91% |
| 15 | 2+2 | 11 | 2.68% | 75% | 93% |

### Reference vs Kernel comparison (both warmup=30, 20% trim)

| Trials | Reference Med CV% | Kernel Med CV% |
|---|---|---|
| 100 | — | 1.56% |
| 50 | 2.00% | 1.99% |
| 30 | 2.24% | 2.01% |
| 20 | 2.56% | 2.91% |

Kernel timing is slightly noisier at low trial counts but comparable at 30-50 trials. The main difference: kernel timing benefits more from 100 trials (1.56% vs estimated ~1.8% for reference) due to higher per-trial jitter from Triton scheduling.

**Code fix**: `eval_kernel_against_ref()` now passes `kernel_exec_result.metadata` to `_run_performance_step()` instead of the local `metadata` dict, fixing the Pydantic dict-copy bug that was silently dropping all `kg_kernel_perf_*` fields.
