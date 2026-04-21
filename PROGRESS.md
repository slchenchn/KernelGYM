# Progress

#### CUDA RL prompts now split backend-neutral data from CUDA-Agent templates

##### Problem & Impact

- The materialized RL and validation parquet files still stored the original first-turn Triton prompt in the data itself, so even the CUDA overlay only changed follow-up feedback turns while the first rollout turn started with Triton instructions and examples.
- This made the data backend-specific and prevented clean switching between Triton/CUDA prompt templates.

##### Resolution

- Added `drkernel/kernel/scripts/materialize_backend_neutral_data.py` to extract only the fenced PyTorch model architecture from the source prompt and write backend-neutral parquet files under the current repo's `drkernel/data`.
- Generated:
  - `drkernel/data/drkernel-rl-data-neutral/cuda_llm_rl_thinking_1025.parquet`
  - `drkernel/data/drkernel-validation-data-neutral/validation_data_thinking.parquet`
- Generated matching human-readable text copies for manual prompt inspection:
  - `drkernel/data/drkernel-rl-data-neutral/cuda_llm_rl_thinking_1025.txt`
  - `drkernel/data/drkernel-validation-data-neutral/validation_data_thinking.txt`
- Generated a deterministic 100-row sample of train prompts after CUDA first-turn formatting:
  - `drkernel/data/drkernel-rl-data-neutral/cuda_llm_rl_thinking_1025.formatted_sample100.txt`
- Updated `drkernel/kernel/config/prompt_config/multi_turn_cuda_kernel.yaml` so the first turn puts CUDA-Agent instructions and the required `CUDA_KERNELS` / `APPLY_BINDINGS` / `MODEL_NEW` output format before the neutral PyTorch problem, then ends with `Let's think step by step.`.
- Updated the RL launcher path so `HYDRA_CONFIG_NAME` can select `cuda_kernel_trainer`, and made the current `14b_coldstart_trloo_mrs_pr_prs.sh` default to the CUDA config, neutral data, and `30s` CUDA reward timeout.
- Compared against `/nfs/FM/lihongbin/CODE/KernelGYM`: that external CUDA path keeps the long CUDA first-turn prompt and CUDA skeleton examples materialized inside parquet `user` messages, while its `multi_turn_cuda_kernel.yaml` leaves `first_turn.template: null` and only controls feedback turns.

##### Result & Current State

- The generated neutral train parquet has `71996` rows, with `0` rows containing `Triton` and `0` rows containing CUDA-Agent section markers.
- The generated neutral validation parquet has `100` rows, with `0` rows containing `Triton` and `0` rows containing CUDA-Agent section markers.
- Some train rows still contain the literal word `cuda` inside the PyTorch problem code itself, for example `device="cuda"` in `get_inputs`; these are problem semantics, not prompt/template instructions.
- Follow-up code inspection found that only `data.system_prompt_config` creates a true `role=system` message, and the CUDA launcher path does not set it; `prompt_config_path` per-turn templates are user-message content, and the shared rollout helper now supports a `{problem}` placeholder so the CUDA first-turn template controls where the neutral PyTorch problem appears.
- The text copies have the expected row separators: `71996` train rows and `100` validation rows; the formatted sample has `100` sampled train prompts with `100` CUDA output-format instructions and `100` neutral PyTorch model prompts.
- `tests/test_backend_neutral_data.py` and the expanded `tests/test_prompt_templates.py` pass, and the broader targeted CUDA support regression set passes `70/70`.

#### CUDA reward comparison against the external KernelGYM repo found one local parameter-propagation gap

##### Problem & Impact

- A follow-up comparison against `/nfs/FM/lihongbin/CODE/KernelGYM` showed that this branch already carries the core CUDA reward algorithm pieces needed for RL, but the local workflow split path did not propagate `num_warmup` and `perf_trim_count` from an `EvaluationTask` into the derived reference/kernel subtasks.
- This affected the intended training timing protocol: `drkernel/kernel/config/kernel_trainer.yaml` sets `num_perf_trials=50`, `num_warmup=30`, and `perf_trim_count=5`, and `drkernel/run_cuda_reward_dir.py` uses the same `30/5/50` defaults, but pre-fix paired workflow children preserved only `num_perf_trials` and fell back to schema defaults `num_warmup=3`, `perf_trim_count=0`.
- Correctness, compilation, precheck, and decoy classification were not changed by this parameter propagation bug, but speedup/reward timing was noisier than the intended protocol because it used fewer warmup iterations and no trimmed mean.

##### Resolution

- Updated `kernelgym/workflow/kernelbench_helpers.py` so both `ReferenceTimingTask` and `KernelEvaluationTask` preserve `num_warmup` and `perf_trim_count`.
- Updated the cache-eviction fallback path in `kernelgym/workflow/kernelbench.py` so a late-created reference timing task also preserves timing controls, toolkit, backend adapter, and resources.
- Added rollout-side tool-feedback truncation using the existing `rollout.multi_turn.max_tool_response_length` and `tool_response_truncate_side` config in the vLLM, multi-iter vLLM, and OpenAI multi-iter rollout engines.
- Added regression coverage in `tests/test_cuda_agent_support.py` for paired-task propagation, the reference-cache fallback path, and bounded feedback truncation.
- Installed the project-pinned pytest tooling into the current Python environment after user approval so the CUDA-Agent test file can be run directly.

##### Result & Current State

- `tests/test_cuda_agent_support.py` now passes `37/37`.
- `git diff --check` and targeted `py_compile` checks pass after the fix.
- Remaining differences from the external repo are implementation/feature-mode differences rather than missing core reward algorithm paths: `tvm_ffi` compile mode, CUPTI/flashinfer timing, and a looser profiler-based CUDA detect helper are not currently ported into this branch.

#### Reward nodes are locked and the CUDA full-set reward sweep completed against the live 16-worker service

##### Problem & Impact

- The CUDA reward path had been repaired and smoke-tested, but there was still no full-set validation proving that the branch could score the entire stored result directory against live `39/40` reward workers.
- The reward infra also was not immediately launch-ready for this branch:
  - `.40` did not have the expected `/nfs/FM` mount on the host
  - the canonical reward startup path still defaulted to the old non-CUDA repo root
- Without stabilizing reward-host clocks and the branch-specific reward service first, a full-set sweep would have mixed algorithm validation with infra noise.

##### Resolution

- Re-locked both reward hosts so all `16` RTX 4090s on `.39` and `.40` now report `Graphics/SM = 2700 MHz` with persistence enabled and `400 W` power limit.
- Restored `.40` host access to `/nfs/FM`, then relaunched the reward stack with `REWARD_REPO_PATH=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent` so the live service imports this branch instead of the older baseline repo.
- Re-verified the IPv4 reward API on `http://192.168.16.39:8111`, and confirmed `16` online workers split evenly as `8` on `reward-39` plus `8` on `reward-40`.
- Updated `drkernel/kernel/scripts/rl/infra_common.sh` so `start_reward.sh` now defaults `REWARD_REPO_PATH` to the current repo root instead of the older `KernelGYM-vllm018` path, removing the need for a manual branch-specific override on future restarts.
- Added a repo-local batch harness at `drkernel/run_cuda_reward_dir.py` for directory-wide CUDA reward evaluation, including:
  - local CUDA-Agent structural precheck classification so obvious malformed samples are scored as `precheck_fail` without consuming GPU workers
  - batched remote submission for the remaining samples through the repaired reward client
  - resumable `results.jsonl` and `summary.json` outputs
- Enabled `use_reference_cache=True` by default for future CUDA batch-harness remote tasks, with `--no-reference-cache` kept as an explicit opt-out for diagnostics.
- Ran an `.18` smoke batch over `8` representative samples from `/nfs/FM/gongoubo/cuda_kernel/parallel_drkernel_minimax_results`, which returned:
  - `2` local `precheck_fail`
  - `6` completed remote evaluations
  - `0` server failures
- Launched the full-set sweep from `.18` inside tmux session `cuda_reward_fullset_20260420` with output rooted at `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/tmp/cuda_reward_fullset_20260420_1251`.

##### Result & Current State

- The branch now has a completed full-directory CUDA reward validation sweep instead of only single-sample checks and unit tests.
- The full-set run used the shared `.18` venv, the IPv4 reward endpoint on `.39`, and the active `16`-worker `39/40` reward pool.
- The active full-set tmux process was launched before the reference-cache default change, so it will not retroactively pick up `use_reference_cache=True`; future or restarted harness runs will use the reference cache unless `--no-reference-cache` is passed.
- A single-sample `.18` diagnostic using `result_0.json` showed the same UUID path dropping from `17.58s` wall time on cache miss to `12.03s` on cache hit, while candidate CUDA compile/load still dominated at about `11.6-12.7s`.
- A follow-up compile-path review found that the CUDA-Agent backend was compiling four translation units per sample, including an invariant `binding_registry.cpp`; this has been collapsed into header-only registry code so future CUDA-Agent cold builds no longer spend a separate compile step on that fixed scaffold file.
- CUDA-Agent compile work directories now prefer `/dev/shm` when enough tmpfs space is available, with `KERNELGYM_CUDA_AGENT_TMPDIR` as an explicit override and automatic fallback to the default temp directory when tmpfs is unavailable or too small.
- CUDA-specific timeout defaults now use `30s` in both the CUDA batch harness and the `cuda_kernel_trainer.yaml` overlay instead of inheriting the general `600s` default.
- The current offline result directory has `8920 / 8920` unique extracted candidate-code hashes and `8920 / 8920` unique reference-code hashes, so content-hash extension caching would not materially accelerate this specific full-set sweep except for retries or repeated RL samples.
- Final observable state:
  - final progress snapshot on `2026-04-21 10:52 +08:00` is `8920 / 8920` samples recorded in `results.jsonl`
  - final result mix is `7385` completed and `1535` failed
  - final summary reports `reward_nonzero=5380`, `compiled_true=7377`, `correct_true=4195`, `decoy_true=8`, `local_precheck_fail=91`, and `server_fail=1444`
  - dominant failure classes are kernel compilation failure (`1094`), `Kernel evaluation failed: 'NoneType' object has no attribute 'metadata'` (`221`), local/code precheck failure (`91`), and decoy detection (`8`)
  - the earlier batch-1 long-tail on `result_24.json` did not prevent completion of the full sweep
  - representative local failures are being classified as `Task failed: code pre-check error` with `reward=-0.5`

#### Failed reward outcomes now route by failure type instead of collapsing to one generic penalty

##### Problem & Impact

- The CUDA reward-path port had preserved the main `weighted` / `speedup` success formulas, but it had dropped the original reward client's failure-type routing.
- As a result, `precheck` failures, CUDA compilation failures, and unrelated generic task failures could all collapse to the same `penalty_score`, which changed the reward algorithm even when the underlying evaluation metadata was correct.

##### Resolution

- Restored `apply_precheck_fail_penalty` / `apply_compilation_fail_penalty` handling in `drkernel/kernel/rewards/reward_client.py`, including:
  - failure-message normalization into `code pre-check error` versus `kernel compilation error`
  - dedicated `precheck_fail` and `compilation_fail` penalty lookups
  - routing in `calculate_reward_like_kernel`, `calculate_reward_weighted`, and `calculate_reward_speedup`
- Reintroduced the corresponding config knobs and `precheck_fail` penalty field in both `drkernel/kernel/config/kernel_trainer.yaml` and `drkernel/kernel/config/kernel_grading.yaml`, and set the defaults on this branch to enabled so the restored logic is active instead of dormant.
- Expanded `tests/test_cuda_agent_support.py` with explicit regression coverage for:
  - precheck-failure reward routing
  - compilation-failure reward routing
  - generic timeout / fallback penalty routing
  - disabled-toggle fallback to generic penalty
- Manually verified representative CUDA-result samples from `/nfs/FM/gongoubo/cuda_kernel/parallel_drkernel_minimax_results` on `.18` using the repaired reward logic:
  - `result_1429.json`: missing `.cu` file, normalized to `code pre-check error`
  - `result_1759.json`: missing `binding_registry.h`, normalized to `code pre-check error`
  - `result_0.json`: compiled and correct, measured `1.94x` speedup, reward `1.0`
  - `result_10.json`: compiled but incorrect, reward `0.0`

##### Result & Current State

- This branch now uses failure-type-aware reward routing again instead of treating all failed tasks as one generic outcome.
- Direct regression verification now passes `33/33` cases in `tests/test_cuda_agent_support.py`.
- Manual sample checks confirm that representative CUDA-result files map to the expected reward branch after the fix.

#### This branch now has a repo-native CUDA-Agent RL path instead of only Triton-specific rollout and reward wiring

##### Problem & Impact

- The branch started from a Triton-focused baseline where reward submission, multi-turn extraction, and prompts still assumed a single Python `ModelNew` answer.
- Porting the external CUDA work verbatim would have cut across the current repo's cleaner `kernelgym` backend layering and still left rollout responses collapsing the CUDA triple-section format before reward evaluation.

##### Resolution

- Added a repo-native `kernelbench.cuda_agent` backend in `kernelgym`, with dispatcher selection, API enum support, and a cheap CUDA-Agent precheck in `kernelgym/toolkit/validation.py`.
- Kept the port aligned with this repo's backend design by reusing `KernelBenchBackendBase` for model construction and runtime execution instead of carrying over a separate session lifecycle.
- Added shared CUDA response extraction in `drkernel/kernel/utils/kernel_code.py`, then used it in both `drkernel/kernel/rewards/kernel_reward.py` and `drkernel/kernel/workers/agent/kernel_agent.py` so CUDA submissions survive intact from rollout through reward scoring.
- Switched `drkernel/kernel/rewards/reward_client.py` to honor per-task `kernel_backend` instead of always submitting `backend="triton"`.
- Added a CUDA-specific multi-turn prompt template and a small Hydra overlay config at `drkernel/kernel/config/cuda_kernel_trainer.yaml` so the CUDA path can be enabled explicitly without overwriting the existing Triton default.
- Added a repo-local single-sample harness at `drkernel/test_cuda_reward.py` and kept it lightweight by loading `drkernel/kernel/utils/kernel_code.py` directly, avoiding the unrelated `watchdog` dependency pulled in by `drkernel.kernel.utils.__init__` on the shared eval venv.
- Ran a real `.18` A800 smoke against `/nfs/FM/gongoubo/cuda_kernel/datasets/0`, which exposed a Triton-only profiling assumption in `kernelgym/toolkit/kernelbench/pipeline.py`; fixed the performance path so Triton coverage and decoy detection are skipped when `enable_triton_detection=False`.
- Wired backend-native CUDA profiling hints from `kernelgym/backend/kernelbench/cuda_agent_backend.py` into the evaluation pipeline, then taught `kernelgym/toolkit/kernelbench/pipeline.py` and `kernelgym/toolkit/kernelbench/profiling.py` to compute coverage from CUDA `__global__` kernel names instead of only from Triton hook matches.
- Restored the `detect_decoy_kernel` control path end to end by adding it to the server/task schemas and toolkit evaluation flags, while keeping coverage computation independent from the decoy penalty so CUDA and Triton both still emit coverage metrics when profiling succeeds.
- Normalized reward-side decoy propagation by writing both `decoy_kernel` and `is_decoy_kernel` into `reward_extra_info`, and updated trainer/metric consumers to read either key.
- Moved the profiler CUDA self-test outside the active profiling window so backend coverage denominators are no longer polluted by the profiler bootstrap op itself.

##### Result & Current State

- The repository can now route `backend="cuda_agent"` requests through `kernelgym` and preserve the expected `CUDA_KERNELS` / `APPLY_BINDINGS` / `MODEL_NEW` answer format end to end.
- The default trainer and grading configs still default to `kernel_backend: "triton"`; the explicit CUDA entrypoint is `drkernel/kernel/config/cuda_kernel_trainer.yaml`.
- Verification now includes a full GPU compile-and-run smoke on `root@192.168.16.18:20629` using the shared `vllm0180` venv and sample dataset `0`.
- Current observed `.18` sample result for the CUDA path: compiled successfully, passed `5/5` correctness trials, mean kernel runtime `0.0201 ms` vs reference `0.0387 ms`, measured speedup `1.93x`, and no decoy marking.
- The CUDA reward path now also emits nonzero coverage metadata on `.18`, and the reward-side coverage helper resolves that sample to `num_custom_kernel=1`, `num_total_kernels=1`, and `time_coverage=1.0` instead of silently falling back to zero.
- A `.18` negative-case smoke where `ModelNew` ignored the compiled extension now returns `decoy_kernel=True` with `num_custom_kernels=0` and profiled only ATen kernels, confirming the backend-native CUDA decoy path is active instead of Triton-only.

#### CUDA reward-path regression coverage now includes paper-aligned decoy and lazy-optimization cases

##### Problem & Impact

- The first CUDA unit-test pass primarily covered extraction, schema wiring, coverage-field propagation, and one decoy branch, but it still left the paper’s two core failure modes under-tested at the training-signal layer.
- Without explicit regression coverage for profiling-based rewards and coverage-based rejection sampling, future changes could preserve basic execution while silently weakening the protections against reward hacking and lazy optimization.

##### Resolution

- Moved CUDA test snippets out of inline strings and into dedicated fixtures under `tests/fixtures/cuda_agent_support/`, then rewired `tests/test_cuda_agent_support.py` to load them from disk.
- Expanded the CUDA test suite to cover decoy-penalty precedence in `calculate_reward_weighted`, paper-style lazy-vs-better-fusion reward shaping using low (`0.00014`) versus high (`0.8615`) `time_coverage`, and coverage-based rejection sampling for both per-turn and geometric aggregation.
- Added deterministic PRS checks with hard thresholds so the suite now verifies:
  - low-coverage correct samples are masked while high-coverage correct samples survive
  - incorrect samples are not filtered by coverage RS
  - the speedup OR-condition can preserve a low-coverage sample

##### Result & Current State

- The CUDA reward-path regression suite now covers both reward-hacking rejection and lazy-optimization mitigation instead of only backend execution mechanics.
- Current direct test sweep for `tests/test_cuda_agent_support.py` passes `29/29`, and `git diff --check` passes.
- The suite now provides explicit regression protection for `PR` and `PRS` semantics in addition to the earlier `.18` runtime smokes.

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
