# Progress

#### Early eval results for run `20260419-131619` are blocked by reward-service outage on `39/40`

##### Problem & Impact

- The post-fix `14B` run produced checkpoints `10/20/30`, but the early eval outputs were ambiguous and `step_20` collapsed to zero metrics.
- Without checking the reward path itself, the team could easily misread an infrastructure outage as a model-quality regression.

##### Resolution

- Recomputed the available checkpoint set from on-disk state and ran dedicated `.18` eval for `10/20/30`.
- Read the per-turn eval artifacts and confirmed repeated reward connectivity failures (`Connection refused` and `No route to host`).
- Checked `192.168.16.39` and `192.168.16.40` directly and confirmed both reward containers were down after host-level reboot; neither restarted automatically because they were configured with `restart=no`.
- Traced the reboot trigger and the underlying GPU fault signature:
  - both hosts entered a clean `systemd` reboot sequence immediately after `root` SSH login from `192.168.120.100`
  - previous-boot NVIDIA logs show real `Xid 109` (`CTX SWITCH TIMEOUT`) events on multiple GPUs across both hosts
  - those `Xid 109` events are surrounded by large volumes of `Xid 13` / `Xid 31` / occasional `Xid 43` from `python3`
  - the detailed fault strings are dominated by `MMU Fault`, `MMU NACK Errors`, `Out Of Range Address`, and `Invalid Address Space`

##### Result & Current State

- Early eval outputs for run `20260419-131619` should be treated as contaminated by reward outage rather than as clean model-quality signals.
- Reward recovery on `39/40` is the gating item for trustworthy fresh-run eval, not `.18` GPU availability.
- Current evidence points to reward-workload GPU faulting rather than one isolated bad card:
  - `.39` logged `Xid 109` on `GPU 7` (`PCI d6:00`) and `GPU 6` (`PCI d5:00`)
  - `.40` logged `Xid 109` on `GPU 1/3/4/6/7` (`PCI 52:00, 57:00, ce:00, d5:00, d6:00`)
  - the same signature has recurred across earlier `kern.log` history on both hosts
  - this is more consistent with generated-kernel evaluation provoking illegal-address / MMU-fault chains that escalate into context-switch timeout than with a single physically defective GPU

#### Historical `20260409-092519` checkpoint coverage now uses per-step `metrics.json` as the source of truth

##### Problem & Impact

- `eval_results/summary.txt` had drifted from the actual checkpoint/eval state for the historical `14B` run, so the remaining backlog could be misidentified.
- `step_260` also had a stale partial eval directory without a valid `metrics.json`.

##### Resolution

- Recomputed the remaining set from the checkpoint tree and the presence of `eval_results/step_*/metrics.json` instead of trusting `summary.txt`.
- Cleared the remaining backlog on `.18` for checkpoints `260/280/300`.
- The final summary rewrite hit `Disk quota exceeded`, so the textual summary file was left stale even though the per-step artifacts were complete.

##### Result & Current State

- The historical run no longer has missing checkpoint evals on disk.
- `metrics.json` under each step is the authoritative record; `eval_results/summary.txt` is stale for `260/280/300`.

#### The post-fix `14B` experiment was restarted as a fresh `50/51` A800 run with checkpoint eval isolated to `.18`

##### Problem & Impact

- The `time_coverage` semantic fix created a new experiment boundary, but earlier restarts still reused the historical run directory.
- Training and checkpoint eval had also been bouncing across different node layouts, which mixed runtime recovery with experiment changes.

##### Resolution

- Relaunched without `RUN_LOG_DIR` so `start_training.sh` created a fresh run directory instead of appending to historical logs.
- Standardized the operating split on `192.168.16.50/51` for training and `.18` for checkpoint eval.
- Kept validation disabled (`VAL_BEFORE_TRAIN=False`, `TEST_FREQ=0`) and retained `REWARD_TASK_TIMEOUT=30` for the new run.
- Revalidated the launch environment inside the training containers, including Python selection, virtualenv selection, and IB visibility.

##### Result & Current State

- The active post-fix experiment is `trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260419-131619`.
- The current `14B` A800 run is a fresh experiment, not a resume from the historical pre-fix tree.
- Training and checkpoint eval now use a stable node split instead of sharing the same machines.

#### `time_coverage` now uses CUDA-only runtime semantics

##### Problem & Impact

- `time_coverage` had been computed from matched custom CUDA time divided by CUDA time plus CPU profiler time.
- That denominator diluted coverage and could push PRS to reject otherwise valid samples.

##### Resolution

- Patched the producer in `kernelgym/toolkit/kernelbench/profiling.py` so `total_kernel_run_time_in_profiling_us` now means CUDA-only runtime.
- Updated downstream consumers in the pipeline and reward paths to use the CUDA-only denominator consistently.
- Added diagnostic fields for both CUDA-only total time and CPU+CUDA total time, then validated the change with `py_compile` and a synthetic ratio check.

##### Result & Current State

- The repository now computes `time_coverage` from CUDA-only runtime.
- The old CPU+CUDA denominator is retained only as a diagnostic field for comparison and debugging.

#### Empty post-adv batches no longer crash the `14B` trainer

##### Problem & Impact

- At `step 221`, post-advantage filtering removed all examples from the batch, but training still entered `update_actor`.
- That produced a `DataProto` chunking failure (`split_size must be a positive integer, but got 0`) and stopped training despite healthy rollout and reward infrastructure.

##### Resolution

- Patched `drkernel/kernel/kernel_trainer.py` so the trainer logs and skips actor/critic update when post-adv filtering leaves an empty batch.
- Rechecked the failure in the run logs and syntax-validated the patch before reuse.

##### Result & Current State

- The empty-batch guard is recorded in commit `9da31c2`.
- Later low-survivor steps progressed instead of crashing, so zero-row post-adv batches are no longer a known trainer stop condition.

#### Checkpoint eval was standardized around `eval_results/step_*`, auto-discovery, and GPU merge

##### Problem & Impact

- Checkpoint eval outputs and helper behavior had drifted across runs, which made resume and backfill eval dependent on hand-maintained step lists.
- The helper scripts also carried a broken summary formatter and multiple merge-path assumptions.

##### Resolution

- Standardized eval outputs under `eval_results/step_*`.
- Reduced merge handling to a single GPU-merge path.
- Made `EVAL_STEPS` optional so the helper can auto-discover missing checkpoints by scanning `CKPT_BASE/global_step_*` and checking for `metrics.json`.
- Fixed the broken summary-formatting path so the helper no longer depends on shell interpolation that produced invalid output.

##### Result & Current State

- Checkpoint backfill can now start from a checkpoint root and infer missing evals from artifact state instead of operator-maintained step lists.
- The helper behavior is aligned around one stable output layout and one merge strategy.

#### Checkpoint eval now supports split-disk H20 storage and isolated local-Ray execution

##### Problem & Impact

- H20 checkpoint eval had assumed one node could see a complete local checkpoint tree, a shared `eval_results` tree, current repo-root validation paths, and container-specific network settings.
- Those assumptions broke head-only eval on split-storage nodes and could also cause eval to attach to the live training Ray cluster by mistake.

##### Resolution

- Reworked `merge_and_eval_checkpoints.sh` so it can stage missing shards and metadata into a local merge dir, then sync worker-side `eval_results` back to the head-visible tree.
- Updated `main_grading.py` to honor `RAY_ADDRESS`, and made checkpoint eval default to `EVAL_RAY_ADDRESS=local` while allowing `EVAL_SOCKET_IFNAME`, `NNODES`, and `N_GPUS_PER_NODE` overrides.
- Added a validation-data remap helper in `grading_common.sh` so stale old-repo paths can be redirected into the current worktree when the local file exists.

##### Result & Current State

- The eval tooling now supports both shared-disk and split-disk storage layouts and can run under an isolated local-Ray session.
- One limitation remains explicit: `global_step_10` cannot be re-evaluated unless the missing worker-side shards are restored.

#### Reward long-tail diagnosis is split between incident recovery and structural synchronous-HTTP risk

##### Problem & Impact

- Extreme rollout long tails initially looked like generic networking noise, but one A800 incident also involved host-level NVIDIA/UVM failure on `16.40`.
- Treating every slowdown as one class of failure made recovery slower and hid the design risk in the reward path itself.

##### Resolution

- Correlated training logs (`main.log`, `trainer.log`, `reward.log`, `vllm.log`) with reward-host diagnostics.
- Separated the immediate A800 incident root cause from the structural effect of synchronous `/evaluate`, where reward-side stalls hold client-side requests open for the full HTTP duration.
- Wrote the incident analysis into the reward long-tail handoff instead of scattering the conclusions across restart notes.

##### Result & Current State

- The earlier A800 incident was recovered operationally at the time, but the durable lesson is architectural: reward-host health and synchronous `/evaluate` coupling must be analyzed separately.
- That split remains the right framework for later outages, including the current reboot-driven reward loss on `39/40`.

#### The `8B` H20 training thread was reframed as three distinct failure classes

##### Problem & Impact

- After fixing the early oversampling bottleneck, later H20 resumes still failed, but the symptoms did not share one root cause.
- Treating them as a single thread obscured which mitigation belonged to which failure.

##### Resolution

- Raised prompt oversampling so the run could clear the original low-batch retry loop.
- Separated the later failures into reward relay loss, async vLLM wake-up OOM, and actor-side OOM caused by foreign GPU jobs on the worker slice.
- Tied the later slow-step behavior back to reward-backed rollout long-tail rather than actor update or `old_log_prob`.

##### Result & Current State

- The H20 thread is no longer treated as one umbrella instability report.
- Later resumes and handoffs should reference the specific failure class they are addressing.

#### Training, eval, and working docs now follow canonical repo-local workflows

##### Problem & Impact

- Launch, stop, status checking, and plotting had drifted into ad hoc shell history.
- Working docs and handoffs were also accumulating as patch notes, which made active investigations harder to read.

##### Resolution

- Added repo-local skills for `start-training`, `stop-training`, and `check-training-status`, and kept orchestration anchored in repo scripts such as `drkernel/kernel/scripts/rl/start_training.sh`.
- Moved local cluster-default selection into untracked `.infra_profile.local.sh` rather than tracked infra defaults.
- Reorganized `handoffs/` into `completed/` and `in_progress/`, and kept stable repo guidance in harness files instead of transient run notes.

##### Result & Current State

- Training launch, stop, and status checks now follow canonical repo-local paths.
- Handoffs are organized around active investigations rather than incremental patch notes.
- `PROGRESS.md` is intended to remain a decision log, not a live-ops transcript.
