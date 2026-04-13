# Handoff: Entropy Collapse Root Cause & Fix

**Date**: 2026-04-09  
**Last updated**: 2026-04-10  
**Status**: Direct masking failure confirmed; raw-policy entropy not yet measured; likely coverage-denominator bug identified; mitigation run running/resumed  
**Reference backend mitigation under test**: reward-side `torch_compile` → `pytorch` eager  

## Executive Summary

The drkernel-14b-coldstart TRLOO training run (2026-04-04 to 2026-04-08) achieved pass@1=0.68 but suffered from **monotonic logged entropy collapse** (`actor/avg_entropy` 0.45 → 0.13 over 170 steps), preventing convergence to the paper's target (0.87).

**Confirmed direct cause of the observed masked-training signal failure**: PRS (coverage-based rejection sampling), combined with mismatch RS, left the archived run with almost no surviving positive training signal.

**Important measurement caveat**: current `actor/avg_entropy` and `actor/entropy_loss` are computed after MRS/PRS have already modified `response_mask`. The training logs do **not** currently contain raw/pre-MRS/pre-PRS policy entropy.

**Mechanism**: 
- the archived run shows extremely low training-batch coverage values (`coverage_rs_mean_coverage` in the 0.001-0.014 range)
- PRS threshold τ=0.3 rejects 100% of the currently identified correct-sample tokens
- geometric mismatch RS also rejects a large fraction of sequences; in logs this appears both as a sequence rejection fraction and as a token-weighted masked fraction
- surviving updates become dominated by incorrect samples, and the logged post-mask entropy collapses

**Current upstream hypothesis**: the leading cause is now a coverage-definition bug/mismatch: `time_coverage` divides matched custom-kernel CUDA time by a denominator that includes both CUDA time and CPU-side profiler time. Reward-side `reference_backend` may still affect the magnitude, but it is no longer the primary suspected mechanism.

**Important distinction from official release**: at official release commit `3a84417f8c0efaadb215ef638b37d12e71ed20f3`, KernelGym reward-side `reference_backend` defaults to eager when omitted, while PPO training-side `actor_rollout_ref.ref.use_torch_compile` separately defaults to `true`. These are different layers and should not be conflated.

**Mitigation under test**: switch the reward-side reference backend to `pytorch` eager and check whether coverage rises enough for PRS to stop fully masking correct samples.

**Next instrumentation gap**: add a raw entropy metric under the original rollout `response_mask` so future plots can separate true policy entropy collapse from post-filter/masked entropy collapse.

**New training run**: `14b_coldstart_trloo_hfsdp8_pytorch_eager.sh` launched 2026-04-09 06:28 UTC.

**Author response on GitHub issue #5**: the authors replied that their reported experiments used a pre-refactor internal environment and suggested tuning hyperparameters such as `COVERAGE_REWARD_WEIGHT` (example: reduce to `0.4`). They also shared a W&B report for their 14B run. This supports treating the public release as non-identical to the paper training environment.

---

## Part 1: Root Cause Analysis

### The Failure Mode

**Previous run (torch_compile backend):**
- Entropy: 0.45 → 0.13 (monotonic collapse)
- Pass@1: 0.53 → 0.68 (improvement, then plateau)
- Best checkpoint: Step 110 (pass@1=0.68, fast@1.2=0.20)
- Target: pass@1=0.87, fast@1.2=0.24

**Paper's failure mode (TRLOO-only, no MRS/PR/PRS):**
- Entropy: HIGH and unstable (Figure 7)
- MRS+PR+PRS brings entropy DOWN to stable level

**Our failure mode is inverted in the currently logged metric**: `actor/avg_entropy` collapses from the start, not elevated. Because this metric is post-MRS/PRS mask, this does not yet prove that raw policy entropy collapses at the same rate.

### Smoking Gun: Training Log Metrics

From `main.log` step-level metrics (extracted 2026-04-09):

| Metric | Step 1 | Step 170 | Expected (paper) |
|---|---|---|---|
| `mismatch/rollout_rs_masked_fraction` (MRS token-weighted masked fraction) | **0.52** | **0.89** | moderate (~0.3-0.5) |
| `mismatch/rollout_rs_seq_masked_fraction` (MRS sequence rejection fraction) | 0.48 | 0.85 | moderate |
| `coverage/coverage_rs_correct_only_masked_fraction` (PRS) | **1.00** | **1.00** | ~0.3-0.5 |
| `coverage/coverage_rs_mean_coverage` | 0.0016 | 0.0025 | should be > τ=0.3 |

**Key observation**: PRS rejects 100% of correct-sample tokens for the entire 170-step run.

### Author W&B Comparison: Coverage PRS

The author's shared W&B report shows `coverage/coverage_rs_correct_only_masked_fraction` mostly below about `0.8`, while our runs are usually above `0.95`:

| Run | Mean `coverage_rs_correct_only_masked_fraction` | Min | Max | Mean `coverage_rs_mean_coverage` |
|---|---:|---:|---:|---:|
| current eager run, 33 visible steps | 0.976 | 0.865 | 1.000 | 0.0042 |
| archived refcache run, 170 steps | 0.981 | 0.777 | 1.000 | 0.0047 |

This difference is large. With the current PRS settings:

```text
keep_prob = clip((coverage - 0.3) / 0.1, 0, 1)
masked_fraction ~= 1 - keep_prob
```

The mask fraction directly maps to the coverage band:

```text
mask = 0.80  -> keep = 0.20 -> coverage around 0.32 in the linear region
mask = 0.95  -> keep = 0.05 -> coverage around 0.305
mask = 0.98  -> keep = 0.02 -> coverage around 0.302
mask = 1.00  -> keep = 0.00 -> coverage <= 0.30
```

Interpretation:

- the author W&B curve implies many correct samples have `time_coverage` materially above the 0.3 threshold
- our curves imply most correct samples are at or below the threshold, often barely surviving only by chance when they are in the narrow interpolation band
- `coverage_rs_mean_coverage` in our logs is only around `0.004`, so the overall training batch has almost no custom-kernel CUDA-time coverage
- the current metric is computed after MRS has already modified `response_mask`, so it represents PRS rejection on MRS-survived correct tokens, not raw correct-sample PRS rejection

This is **not** directly explained by `COVERAGE_REWARD_WEIGHT`. That weight changes reward shaping:

```text
final_reward += COVERAGE_REWARD_WEIGHT * coverage
```

It does not enter the PRS keep probability. Lowering `COVERAGE_REWARD_WEIGHT` from `0.5` to `0.4` may reduce optimization pressure toward the coverage proxy, but on a fixed rollout batch it does not change `coverage_rs_correct_only_masked_fraction`.

Most likely explanations for the author-vs-ours gap:

1. The authors' pre-refactor environment measured coverage differently, or had a different profiling denominator/numerator.
2. Their correct samples actually placed substantially more runtime into generated custom kernels.
3. MRS in our run changes the population entering PRS and may leave mostly low-coverage correct tokens.
4. Their run may have used different PRS settings, such as a lower threshold, larger factor, `num_coverage`, or speedup-threshold OR keep logic.

Recommended next instrumentation:

- log correct-only raw `time_coverage` distribution before MRS
- log correct-only `time_coverage` distribution after MRS and before PRS
- log PRS keep probability distribution for correct samples
- log whether speedup-threshold OR logic is enabled and how often it keeps samples
- compare `time_coverage` and `num_coverage` side by side

### Likely Upstream Bug: CPU Time in `time_coverage` Denominator

Static inspection of the profiling path identified a likely denominator-inflation bug in `kernelgym/toolkit/kernelbench/profiling.py`.

Current implementation:

```python
total_time = 0.0
matched_cuda_time = 0.0

for prof_kernel in kernels_in_profiling:
    cuda_time = float(prof_kernel["cuda_time_us"])
    cpu_time = float(prof_kernel["cpu_time_us"])
    total_time += cuda_time + cpu_time

    if any(_matches_profiler_name(kernel_name, prof_name) for kernel_name in kernel_names):
        matched_cuda_time += cuda_time

return {
    "total_kernel_run_time_in_profiling_us": total_time,
    "custom_kernel_cuda_time_in_profiling_us": matched_cuda_time,
}
```

Downstream reward code computes:

```python
time_coverage = custom_kernel_cuda_time_in_profiling_us / total_kernel_run_time_in_profiling_us
```

So the current ratio is effectively:

```text
matched custom CUDA time / (all CUDA time + CPU-side profiler time)
```

This is not a pure CUDA-time coverage ratio. The same profiling module already computes a CUDA-only total:

```python
"total_cuda_time_us": sum(k["cuda_time_us"] for k in cuda_kernels)
```

but `compute_triton_kernel_coverage()` does not use it for `total_kernel_run_time_in_profiling_us`.

Why this matters:

- PRS expects `coverage` to represent how much kernel execution is covered by generated custom kernels.
- If CPU dispatch / Python / synchronization / profiler overhead is included in the denominator, `time_coverage` can collapse toward zero even when generated custom kernels account for a meaningful share of CUDA execution.
- Our observed `time_coverage` is around `0.001 - 0.015` in training logs, which is exactly the regime that makes PRS reject nearly all correct samples under threshold `0.3`.
- The author's W&B curve showing much lower `coverage_rs_correct_only_masked_fraction` is consistent with their pre-refactor environment using a less inflated coverage denominator or otherwise producing higher effective coverage.

Important caveats:

- This explains `time_coverage` collapse directly, but does not by itself explain low `num_coverage` around a few percent. Low `num_coverage` may indicate that many correct samples do not execute matched generated custom kernels, or that kernel-name matching/profiling capture is incomplete.
- PRS does not multiply masks by `coverage_rs_factor`; it samples a binary keep mask from `clip((coverage - threshold) / factor, 0, 1)`. Below threshold `0.3`, correct samples have keep probability `0`.
- The entropy impact is strongest through the training signal: correct/successful paths are masked out before actor loss. The logged entropy is also post-mask, so it reflects the surviving token population rather than raw policy entropy.

Recommended diagnostic:

```text
time_coverage_cpu_cuda = matched_cuda_time / sum(cuda_time + cpu_time)
time_coverage_cuda_only = matched_cuda_time / sum(cuda_time)
```

Log both values for reward outputs and PRS inputs. If `time_coverage_cuda_only` moves into the author-like range while current `time_coverage_cpu_cuda` stays near zero, the denominator bug is confirmed.

Recommended fix candidate:

```python
total_time += cuda_time
```

or explicitly store both fields:

```text
total_kernel_run_time_in_profiling_us_cuda_only
total_kernel_run_time_in_profiling_us_cpu_cuda
```

and use the CUDA-only field for PRS `time_coverage`.

### MRS Metric Semantics

Local paper extraction used for this interpretation: `MinerU_markdown_2602_dr.kernel.md`, Section 5.1, around lines 184-190.

The current hfsdp8 scripts set:

```bash
ROLLOUT_RS="geometric"
ROLLOUT_RS_KWARGS="{lower:0.999,upper:1.001}"
ROLLOUT_TOKEN_VETO_THRESHOLD=1e-4
```

This means the main MRS decision matches the paper's **sequence-level geometric MRS**:

- for multi-turn samples, the code flattens all turns/tokens in the sample, computes `exp(mean(log(pi_train / pi_rollout)))`, and accepts/rejects the whole sequence against `[0.999, 1.001]`
- the keep/reject decision is then broadcast back to the token-shaped `response_mask` because the actor loss consumes a token-level mask
- therefore `mismatch/rollout_rs_masked_fraction` is a **token-weighted masked fraction after sequence-level MRS**, not evidence that the current run is using token-level MRS
- `mismatch/rollout_rs_seq_masked_fraction` is the metric closest to the paper-level sample/sequence rejection rate

The paper also describes a separate strict token-level veto: if any single token ratio falls below `1e-4`, the entire sequence is rejected. The code implements this as `rollout_token_veto_threshold=1e-4` and logs it separately as:

- `mismatch/rollout_is_veto_fraction`: sequence fraction rejected because at least one token hit the veto
- `mismatch/rollout_is_catastrophic_token_fraction`: token fraction below the veto threshold

So the log contains token-shaped metrics, but the active `geometric` MRS mode is still sequence-level.

### Important Scope Distinction

There are two similarly named but different "ref" concepts in this stack:

- **Reward-side `reference_backend`**: controls how KernelGym / KernelBench times the reference implementation during reward evaluation
- **Training-side `actor_rollout_ref.ref.use_torch_compile`**: controls whether the PPO ref log-prob model uses `torch.compile`

Official release check at commit `3a84417f8c0efaadb215ef638b37d12e71ed20f3` shows:

- KernelGym reward-side `reference_backend` is eager by default when omitted
- DRKernel RL scripts at that commit did not explicitly pass `reward_model.reference_backend`
- PPO training-side `ref.use_torch_compile` still defaulted to `true`

So the official release code was **not** "compile everywhere". Reward-side reference timing defaulted to eager, while the PPO ref model still defaulted to compiled execution.

### Entropy Metric Scope

Current actor entropy logging is post-filter:

- `drkernel/verl_patch/workers/code/actor/dp_actor.py` computes `actor/avg_entropy` and `actor/entropy_loss` using `response_mask`
- `drkernel/kernel/kernel_trainer.py` saves `original_response_mask`, computes MRS and coverage PRS, then overwrites `batch.batch["response_mask"] = modified_response_mask` before actor update
- as a result, `actor/avg_entropy` is computed only over tokens that survived MRS and PRS

This is a key blind spot. The logs currently do not contain:

- raw entropy over the original rollout response mask
- pre-MRS entropy
- pre-PRS/post-MRS entropy

Recommended instrumentation:

- preserve `original_response_mask` in the actor batch as a separate key such as `raw_response_mask`
- log `actor/raw_avg_entropy = masked_mean(entropy, raw_response_mask)`
- optionally also log `actor/pre_prs_avg_entropy` if MRS and PRS effects need to be separated
- keep the existing `actor/avg_entropy` as `post_mask_avg_entropy` semantics, even if the metric name remains unchanged for compatibility

### Causal Chain

1. **Observed condition: training-batch coverage in the archived run is extremely low**
   - Coverage = `T_generated / T_total` (fraction of CUDA time in generated kernels)
   - Observed in the archived run: `coverage_rs_mean_coverage` stays around `0.0016 -> 0.0025`
   - This metric is aggregated over the training batch after reshaping turns; it is **not** a correct-only coverage metric
   - Static inspection now shows `T_total` likely includes CPU-side profiler time, making the denominator too large for a CUDA-time coverage metric

2. **PRS over-rejects correct samples**
   - PRS retention formula: `p = clip((coverage - τ) / s, 0, 1)` where τ=0.3, s=0.1
   - With coverage=0.001: `p = clip((0.001 - 0.3) / 0.1, 0, 1) = 0`
   - Result: 100% of correct-sample tokens get `response_mask[i,t] = 0` (masked out)
   - This happens at step 1, before any gradient update

3. **Training signal becomes one-sided**
   - Correct samples: effectively 100% masked in the archived run
   - Incorrect samples: not all are kept, but after mismatch RS and PRS the surviving batch contains no correct samples at step 1
   - Net effect: the actor update sees a sharply imbalanced signal dominated by incorrect samples

4. **Logged post-mask entropy collapses**
   - The update no longer receives the positive constraints that would reinforce successful behaviors
   - The policy keeps moving under an imbalanced signal instead of a balanced correct/incorrect mix
   - The currently logged entropy decreases monotonically over the surviving post-filter token set
   - Raw policy entropy may also be narrowing, but that has not yet been measured directly in the current logs

### Why the Paper Doesn't See This

**What is currently plausible, not yet proven:**
- If the paper and official/internal training path used a CUDA-only or otherwise less inflated coverage denominator, the PRS threshold `τ=0.3` would be much less risky
- The official release check at commit `3a84417f8c0efaadb215ef638b37d12e71ed20f3` supports eager-by-default behavior on the reward-side `reference_backend`
- However, repository evidence here does **not** yet prove the paper's exact training backend or exact coverage distribution during training
- The paper's entropy curve must be compared carefully against our metric because our current `actor/avg_entropy` is post-MRS/PRS mask; raw entropy is not logged yet

### Why This Isn't Just "Weak Reward Signal"

The reward coefficient difference (paper: 1.0, us: 0.5) is a red herring:
- Weaker rewards → slower policy updates, not faster entropy collapse
- The mechanism is PRS masking, not reward magnitude
- Even with identical reward coefficients, the archived run would still collapse if correct samples keep getting fully masked

---

## Part 2: Mitigation Under Test

### Change: Reference Backend

**File**: `drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh`

```bash
# Before (torch_compile)
REFERENCE_BACKEND="torch_compile"
LOG_RUN_PREFIX="${LOG_RUN_PREFIX:-trloo-14b-hfsdp8-refcache}"

# After (pytorch eager)
REFERENCE_BACKEND="pytorch"
LOG_RUN_PREFIX="${LOG_RUN_PREFIX:-trloo-14b-hfsdp8-pytorch-eager}"
```

**What this tests:**
- whether reward-side eager timing materially raises training-batch coverage from the archived `0.001` scale
- whether PRS rejection of correct samples drops below `1.0`
- whether logged post-mask entropy stops collapsing once positive samples survive masking

**Updated interpretation after denominator inspection**:
- reward-side eager may still change timing, but if the coverage denominator includes CPU time, backend switching alone may not fix PRS
- a direct coverage-denominator diagnostic/fix is now higher priority than further backend-only trials

### Why This Remains Plausible

**Paper (Section 6.1, line 236):**
> "We follow the official Torch backend in KernelBench and their implementations of correctness and speedup measurement."

**Official release check at `3a84417`**:
- KernelGym reward-side `reference_backend` defaults to eager when omitted
- DRKernel RL scripts at that commit did not explicitly pass `reward_model.reference_backend`
- PPO ref model compile defaults are separate and should not be read as reward-side backend defaults

**Inference**:
- it remains plausible that the paper trained with reward-side eager semantics and evaluated compile separately
- but that is still an inference, not a repository-level proof

---

## Part 3: Reproduction Tutorial

### Prerequisites

- 16× A800 GPUs (2 nodes, 8 per node)
- 16× 4090 GPUs (2 nodes, 8 per node) for reward evaluation
- vLLM 0.18.0 deployed on reward nodes
- Ray cluster configured
- Reward server running (see `start_reward.sh`)

### Step 1: Verify Reward Infrastructure

```bash
curl http://192.168.16.39:8111/health | python3 -m json.tool
# Expected: {"status": "healthy", "workers": 16, ...}
```

### Step 2: Launch Training

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018

# Option A: Use the new pytorch eager script
bash drkernel/kernel/scripts/rl/start_training.sh \
  drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh

# Option B: Set env var to override backend in existing script
REFERENCE_BACKEND=pytorch \
LOG_RUN_PREFIX=trloo-14b-hfsdp8-pytorch-eager \
bash drkernel/kernel/scripts/rl/start_training.sh \
  drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_refcache.sh
```

### Step 3: Monitor Training

```bash
# Monitor main log
ssh -p 20629 root@192.168.16.18 'tail -f /tmp/train-hfsdp8-pytorch-eager.log'

# Check logged post-mask entropy and coverage metrics
ssh -p 20629 root@192.168.16.18 'grep "actor/avg_entropy\|coverage/coverage_rs_mean_coverage" /path/to/main.log | tail -20'

# Hypothesis if eager mitigation works:
# - actor/avg_entropy should stop monotonic collapse, but this remains post-mask entropy
# - coverage/coverage_rs_mean_coverage should rise materially above the archived 0.001-scale
# - coverage/coverage_rs_correct_only_masked_fraction should drop below 1.0
```

### Step 4: Evaluate Checkpoints

After training completes (or at intermediate steps):

```bash
bash drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh

# This will:
# 1. Merge FSDP checkpoints to HF format (CPU-only, no GPU memory)
# 2. Evaluate on 2 nodes in parallel
# 3. Generate summary table and plots
```

### Step 5: Compare Results

```bash
# Plot fast@1 and fast@1.2 across steps
python3 drkernel/kernel/scripts/rl/plot_checkpoint_eval.py \
  /path/to/eval_results \
  --output-dir /path/to/output

# Expected: fast@1.2 should increase monotonically (not plateau at step 60-110)
```

---

## Part 4: Key Metrics to Watch

### Entropy

```
Step 1:   entropy ≈ 0.45
If mitigation works: entropy should stop monotonic collapse
Archived failure: 0.45 -> 0.13 over 170 steps
Current caveat: this is `actor/avg_entropy`, computed after MRS/PRS masking, not raw policy entropy
```

### Coverage

```
Archived failure: mean_coverage ≈ 0.0016 at step 1
Mitigation target: move materially above the archived 0.001-scale
Likely issue: current denominator may include CPU-side profiler time; compare CUDA-only coverage
```

### PRS Rejection

```
Archived failure: coverage_rs_correct_only_masked_fraction = 1.0
Mitigation target: drop below 1.0 and allow correct samples to survive
```

### Performance (should converge toward paper's target)

```
Baseline (coldstart): pass@1 = 0.53
Step 50:  pass@1 ≈ 0.65-0.70
Step 100: pass@1 ≈ 0.75-0.80
Step 170: pass@1 ≈ 0.80-0.87 (target: 0.87)
```

---

## Part 5: Troubleshooting

### If entropy still collapses:

1. **Check whether the collapse is raw or mask-induced**
   - Current logs only have post-mask `actor/avg_entropy`
   - Add or inspect `actor/raw_avg_entropy` before concluding true raw policy entropy collapse
   - Compare raw entropy, post-MRS/pre-PRS entropy, and final post-MRS/PRS entropy if instrumentation is available

2. **Verify REFERENCE_BACKEND is set correctly**
   ```bash
   grep "reference_backend" /path/to/trainer.log | head
   # Archived run logged this in trainer config output, not in main.log
   ```

3. **Check coverage values in logs**
   ```bash
   grep "coverage_rs_mean_coverage" /path/to/main.log | head -5
   # Mitigation target: materially above the archived 0.001-scale
   ```

4. **If coverage is still low**: do not assume `torch_compile` is the only cause. Check:
   - trainer config output (`reward_model.reference_backend`)
   - Hydra overrides in training command
   - whether train-batch coverage and validation coverage are being compared on the same population
   - whether `time_coverage` uses CUDA-only denominator or CPU+CUDA denominator
   - whether another profiling-distribution change is keeping coverage below threshold

### If training is slow:

- Check reward worker pool status: `curl http://192.168.16.39:8111/health`
- Verify MRS rejection rate with the right metric: `rollout_rs_seq_masked_fraction` is the sequence/sample rate; `rollout_rs_masked_fraction` is token-weighted
- If MRS rejection is high, consider widening bounds: `ROLLOUT_RS_KWARGS="{lower:0.995,upper:1.005}"`

### If training crashes:

- Check VRAM: `nvidia-smi` on training nodes
- Verify Ray cluster: `ray status`
- Check reward server logs: `docker logs kernelgym-reward-39`

---

## Part 6: Config Comparison

### Archived failure configuration

```bash
REFERENCE_BACKEND="torch_compile"
COVERAGE_RS_THRESHOLD=0.3
COVERAGE_RS_FACTOR=0.1
```

**Observed result**: archived run had coverage on the 0.001-scale, PRS rejected 100% of correct samples, and entropy collapsed

### Eager mitigation under test

```bash
REFERENCE_BACKEND="pytorch"
COVERAGE_RS_THRESHOLD=0.3
COVERAGE_RS_FACTOR=0.1
```

**Hypothesis**: coverage rises enough to stop full masking of correct samples

**Current caveat**: if `time_coverage` denominator remains CPU+CUDA, eager reward backend may not be sufficient.

---

## Part 7: Next Steps (If Needed)

### If pytorch eager still doesn't reach 0.87:

1. **Add raw entropy logging**: distinguish raw policy entropy from post-MRS/PRS masked entropy
2. **Add CUDA-only coverage logging/fix**: compare CPU+CUDA denominator against CUDA-only denominator
3. **Increase training steps**: 170 → 300 (paper's setting)
4. **Raise reward coefficients**: 0.5 → 1.0 (match paper)
5. **Widen MRS bounds**: [0.999, 1.001] → [0.995, 1.005]
6. **Increase max prompt length**: 10240 → 32768 (match paper)

### If you want to isolate PRS effect:

Disable PRS entirely:
```bash
COVERAGE_RS_THRESHOLD=-1  # Disables PRS
```

Then compare entropy trajectory with/without PRS.

---

## References

- **Local paper markdown**: `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/MinerU_markdown_2602_dr.kernel.md`
- **Paper MRS definition**: `MinerU_markdown_2602_dr.kernel.md`, Section 5.1, around lines 184-190 (`geometric MRS` plus `1e-4` token-level veto)
- **Paper PRS definition**: `MinerU_markdown_2602_dr.kernel.md`, Section 5.2, around lines 214-220 (`Profiling-based Rejection Sampling`)
- **Paper training dynamics figure**: `MinerU_markdown_2602_dr.kernel.md`, Figure 7 caption around line 469
- **Coverage denominator code**: `kernelgym/toolkit/kernelbench/profiling.py`, `compute_triton_kernel_coverage()`
- **Reward coverage code**: `drkernel/kernel/rewards/reward_client.py`, `compute_coverage_reward()`
- **PRS code**: `drkernel/kernel/rewards/coverage_helper.py`, `compute_rollout_rejection_mask()`
- **Previous analysis**: `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-refcache.train.16xA800.reward.16x4090.run.20260404-032344/TRAINING_ANALYSIS.md`
- **New training script**: `drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh`
- **Training launch**: `drkernel/kernel/scripts/rl/start_training.sh`
- **Official release inspection worktree**: `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-official-3a84417`
