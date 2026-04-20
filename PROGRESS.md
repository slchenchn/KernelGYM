# Progress

#### The historical `20260409-092519` 14B run now has its remaining checkpoints `260 280 300` under dedicated eval on `.18` — ACTIVE

##### Problem & Impact

- The old `20260409-092519` run still had remaining checkpoints without completed eval results, but the previous on-disk signals had drifted:
  - `eval_results/summary.txt` no longer matched the actual checkpoint tree
  - `step_260` still had an incomplete eval directory without `metrics.json`
- Without rechecking from the real checkpoint and `metrics.json` state, the remaining eval set could easily be misidentified.

##### Resolution

- Recomputed the remaining eval set from the actual filesystem state instead of trusting `summary.txt`:
  - existing checkpoints under the old run: `20 60 100 140 180 220 260 280 300`
  - completed evals by `metrics.json`: through `230` plus `270`
  - actual remaining checkpoints: `260 280 300`
- Revalidated the `.18` eval environment before launch:
  - `which python`, `sys.executable`, and `VIRTUAL_ENV` all resolve to the shared repo venv
  - reward health is reachable from `.18`
  - the validation dataset, eval script, and all three target checkpoints are reachable
- Removed stale partial eval directories for `260`, `280`, and `300`, then launched a dedicated `.18` tmux session with:
  - `TRAIN_CLUSTER_PROFILE=a800`
  - `TRAIN_HEAD_MODE=local`
  - `EVAL_USE_WORKER=0`
  - `EVAL_STEPS='260 280 300'`
  - `EVAL_CLEANUP_KILL_GPU_PIDS=1`

##### Result & Current State

- The active `.18` eval session is `eval-14b-remaining-gpu-1618`.
- The live log is [`/tmp/eval-14b-remaining-gpu-1618.log`](/tmp/eval-14b-remaining-gpu-1618.log).
- The current batch is running serially on `.18` for the historical run:
  - `260`: active in `kernel.main_grading` with `8` live `ray::AsyncActorRolloutRefWorker.execute_method` processes
  - `280`: queued
  - `300`: queued
- `.18` currently has live eval processes and GPU occupancy for this batch.

#### The fresh `14B` A800 eager run now uses a new log directory so the fixed `time_coverage` semantics do not share the old run history — ACTIVE

##### Problem & Impact

- The previous post-fix attempt still reused the old run directory, so it appeared under the historical log tree.
- That was the wrong experiment boundary: once `time_coverage` semantics changed, the user wanted a fresh run with a new log directory instead of appending new steps to the old run history.

##### Resolution

- Re-scoped the experiment boundary so the same `14B` eager training configuration starts without `RUN_LOG_DIR`, allowing the canonical [`start_training.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_training.sh) path to create a fresh run directory.
- Kept the requested runtime behavior changes that still apply to the new experiment:
  - `VAL_BEFORE_TRAIN=False`
  - `TEST_FREQ=0`
  - `REWARD_TASK_TIMEOUT=30`
- Verified from the fresh head-container log that the launcher created a new run directory under the standard timestamped naming scheme.

##### Result & Current State

- The active run directory is now:
  - [`trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260419-131619`](</nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260419-131619>)
- The head log explicitly shows the new path in `Logging training run to: .../run.20260419-131619/main.log`.
- The current run starts from a fresh experiment boundary rather than restoring prior checkpoints:
  - there is no `Found checkpoint`
  - there is no `Resuming from`
  - the training command now points `trainer.default_local_dir` at the new run's `checkpoints/`
- The new run is live on `50/51`:
  - tmux session: `train-14b-hfsdp8-pytorch-eager-fresh-5051`
  - `ray status`: `2` active nodes, `16.0/16.0 GPU` in use
  - both nodes have live `ray::WorkerDict.actor_rollout_init_model` workers on GPU
  - the run has now made real training progress through `step 30`
  - the current in-flight step is `31`
  - the latest timing summary at `step 30` is:
    - `gen`: `11.3 min`
    - `old_log_prob`: `2.2 min`
    - `update_actor`: `2.1 min`
    - `total step`: `20.9 min`
  - recent PRS / coverage signals are materially healthier than the archived failure regime:
    - `coverage_rs_correct_only_masked_fraction`: `0.5202` at `step 30`
    - `coverage_rs_mean_coverage`: `0.0735` at `step 30`

#### `time_coverage` now uses a CUDA-only denominator instead of CUDA+CPU profiler time, and the diagnostic CPU+CUDA total is preserved separately — ACTIVE

##### Problem & Impact

- The entropy-collapse handoff identified a likely coverage-definition bug: `time_coverage` was being computed from `matched custom CUDA time / (all CUDA time + CPU profiler time)`.
- That inflates the denominator with host-side profiler time, which can collapse coverage toward zero and cause PRS to reject nearly all correct samples even when the generated kernels account for meaningful CUDA execution time.

##### Resolution

- Patched the producer in [`kernelgym/toolkit/kernelbench/profiling.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/profiling.py):
  - `total_kernel_run_time_in_profiling_us` now means total CUDA time only
  - added explicit diagnostic fields:
    - `total_kernel_cuda_time_in_profiling_us`
    - `total_kernel_run_time_in_profiling_us_cpu_cuda`
- Patched downstream consumers to use the CUDA-only denominator consistently:
  - [`kernelgym/toolkit/kernelbench/pipeline.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/pipeline.py)
  - [`drkernel/kernel/rewards/reward_client.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/rewards/reward_client.py)
  - [`drkernel/kernel/workers/reward_manager/kernel_async.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/reward_manager/kernel_async.py)
- Added the new diagnostic fields to the reward log routing and result-metadata sanitation paths:
  - [`drkernel/kernel/scripts/rl/log_router.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/log_router.py)
  - [`kernelgym/schema/result.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/schema/result.py)
- Validated the fix with:
  - `python -m py_compile` on all edited Python files
  - a synthetic coverage check showing:
    - CUDA-only total = `40`
    - CPU+CUDA total = `200`
    - fixed `time_coverage` = `10 / 40 = 0.25`
    - old diluted ratio would have been `10 / 200 = 0.05`
  - a `KernelRewardClient.compute_coverage_reward(...)` sanity check confirming the client now returns `coverage=0.25` from the CUDA-only denominator

##### Result & Current State

- The repository now computes `time_coverage` from CUDA-only runtime, which matches the intended semantics in the entropy-collapse diagnosis.
- The old CPU+CUDA denominator is still retained under a separate field for debugging and comparison.
- The reward environment has since been refreshed, so the new producer-side and reward-worker code is now live end-to-end rather than only validated synthetically.
- The current `50/51` fresh run is using the fixed `time_coverage` path.

#### Checkpoint eval on `.18` recovered `global_step_230`, and the next auto-discovered batch `240 250 260 270` is now running — SUPERSEDED

##### Problem & Impact

- `global_step_230` still had no valid eval result under [`eval_results/step_230`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/eval_results/step_230):
  - the first attempt failed at eval startup because stale `.18` GPU workers left too little free memory for vLLM
  - the later rerun failed again before producing `metrics.json`, and the same batch also hit `Disk quota exceeded` while trying to continue into later checkpoints
- That left `step_230` without a usable metric artifact and blocked the next untested checkpoint batch behind it.

##### Resolution

- Revalidated the actual `.18` eval environment before running the new batch:
  - `which python`, `sys.executable`, and `VIRTUAL_ENV` all resolve to the shared repo venv
  - reward health is reachable from `.18`
  - `/nfs/FM` now has free space again
  - `.18` GPUs were idle before the new batch
- Removed the stale partial [`eval_results/step_230`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/eval_results/step_230) directory and ran a dedicated single-step eval in tmux session `eval-14b-230-gpu-1618` using the canonical [`merge_and_eval_checkpoints.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh) path with:
  - `TRAIN_CLUSTER_PROFILE=a800`
  - `CKPT_BASE=<...>/checkpoints`
  - `EVAL_STEPS=230`
  - `EVAL_USE_WORKER=0`
  - `EVAL_CLEANUP_KILL_GPU_PIDS=1`
- Confirmed the rerun crossed the previous failure boundary and completed:
  - GPU merge completed and rewrote `huggingface_merged`
  - the job advanced into `python -m kernel.main_grading`
  - [`eval_results/step_230/metrics.json`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/eval_results/step_230/metrics.json) was written successfully
  - the summary row for `230` is now present in [`eval_results/summary.txt`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/eval_results/summary.txt)
- With `230` closed, continued into the next auto-discovered checkpoint batch in tmux session `eval-14b-auto-gpu-1618` with the same canonical script and `.18`-local dedicated-eval settings.
- The new batch auto-detected the remaining untested checkpoints:
  - `240 250 260 270`

##### Result & Current State

- `global_step_230` now has a completed eval result:
  - `pass@1=0.8033`
  - `correct=0.6787`
  - `compile=0.8938`
  - `fast@1=0.5250`
  - `fast@1.2=0.3275`
- `global_step_270` also completed in the same `.18` checkpoint-eval thread.
- The later `.18` `step_260` eval was interrupted when reward was restarted for the new `time_coverage` rollout, so `.18` is no longer the active workstream right now.

#### H20 checkpoint eval on `.3` now tolerates split-disk checkpoint storage and stale hardcoded validation-data paths while staying isolated to `CUDA_VISIBLE_DEVICES=6,7` — ACTIVE

##### Problem & Impact

- Head-only checkpoint eval for the active `8B` H20 run initially failed in two different ways:
  - split-storage merge needed to stage missing rank shards from `.5`, but the helper path still used stdout capture and could hang the shell around `prepare_merge_local_dir`
  - the eval launcher inherited a stale hardcoded validation-data path under the old `KernelGYM` repo root, so eval failed with `FileNotFoundError` even after merge succeeded
- That blocked use of the remaining `.3` GPUs `6,7` for checkpoint testing while training still occupied `0-5`.

##### Resolution

- Updated [`merge_and_eval_checkpoints.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh) so split-storage merge no longer depends on `merge_input_dir="$(prepare_merge_local_dir ...)"`.
  - the helper now sets `PREPARED_MERGE_LOCAL_DIR` directly instead of returning its path through stdout capture
  - staging logs remain on stderr so they do not pollute path handling
- Updated [`grading_common.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/eval/grading_common.sh) with a repo-data remap helper that:
  - detects missing old paths ending in `/drkernel/data/...`
  - remaps them to the current worktree's `${DRKERNEL_ROOT}/data/...` when the corresponding file exists locally
- Revalidated the current local validation dataset path:
  - `/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/data/drkernel-validation-data/validation_data_thinking.parquet`
- Ran checkpoint eval under tmux session `eval-8b-ckpt-gpu67` with:
  - `CUDA_VISIBLE_DEVICES=6,7`
  - `MERGE_CUDA_VISIBLE_DEVICES=6`
  - `EVAL_USE_WORKER=0`
  - `NNODES=1`
  - `N_GPUS_PER_NODE=2`
  - `REWARD_SERVER_URL=http://10.0.18.3:18112`

##### Result & Current State

- The new eval session is active on `.3` and is again processing the untested checkpoints `10 50 100 140`.
- Training remains isolated on GPUs `0-5`; the eval is constrained to the remaining two GPUs.
- The current live phase has progressed past both earlier failure boundaries:
  - `Step 10` split-storage shard staging completed
  - `Step 10` HF merge completed and wrote `huggingface_merged`
  - `Step 10` eval is now live under `python -m kernel.main_grading`
- No `metrics.json` has been written yet, so `step_10` is still in-flight rather than completed.

#### Checkpoint eval now handles both shared-disk and split-disk training nodes by auto-staging missing FSDP merge inputs and syncing worker eval outputs back to the head-visible results tree — COMPLETED

##### Problem & Impact

- [`merge_and_eval_checkpoints.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh) had been updated upstream for auto-discovery and GPU merge, but it still assumed one node could see a complete `actor/` checkpoint tree and a shared `eval_results/` tree locally.
- That assumption breaks on the current split-disk H20 layout:
  - `10.0.18.3` stores only rank `0-5` model shards plus rank-0 metadata
  - `10.0.18.5` stores only rank `6-11` model shards
  - worker-side eval outputs are not automatically visible from the head node
- Without a mixed-storage fix, worker-side merge and summary logic would fail even though the same script still needs to remain valid for shared-disk machines.

##### Resolution

- Fast-forwarded the worktree to the latest remote branch state before changing the script, so the fix builds on the current auto-discovery / GPU-merge flow instead of overwriting it.
- Reworked [`merge_and_eval_checkpoints.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh) so it now:
  - auto-detects whether the target node already has a complete local FSDP merge input set
  - stages missing `model_world_size_*_rank_*.pt` files from the peer node into a temporary local merge dir when storage is split
  - stages rank-0 merge metadata (`fsdp_config.json` and `huggingface/`) from the head node when the worker does not have it locally
  - creates and cleans eval artifacts on the node that actually runs the eval
  - copies worker-side `eval_results/step_*` outputs back to the head-visible results tree before local summary generation
  - remains sourceable for targeted helper testing instead of forcing immediate script execution on `source`
- Verified the live H20 split-storage facts through the new helpers:
  - `world_size=12`
  - head metadata present, worker metadata absent
  - head merge inputs incomplete, worker merge inputs incomplete
  - head ranks `0-5`, worker ranks `6-11`
- Smoke-tested the new cross-node tar copy path in both directions:
  - head metadata copy into a worker temp dir
  - worker result-dir copy back into a head temp dir

##### Result & Current State

- The checkpoint-eval merge path now supports both environments:
  - shared-disk nodes keep using local actor dirs directly
  - split-disk nodes auto-stage the missing merge inputs per step before merge
- Worker-side eval results no longer disappear from the head-side summary path on split storage.
- Validation completed with `bash -n` plus live helper smoke tests on the `h20` profile.

#### Post-adv empty-batch at `step 221` no longer crashes the `14B` A800 trainer after the empty-batch guard was added — ACTIVE

##### Problem & Impact

- The live `14B` A800 eager run on `50/51` crashed immediately after rollout completed for `step 221`.
- The failing case was not a reward outage or node loss: post-advantage filtering removed all examples from the batch, but training still continued into `update_actor`, which then crashed during `DataProto` chunking with `RuntimeError: split_size must be a positive integer, but got 0`.
- This blocked forward progress at `global_step_220` even though reward and rollout infrastructure were otherwise healthy.

##### Resolution

- Confirmed the failure from the run logs:
  - [`trainer.log`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/trainer.log) shows the post-adv filtering collapse to zero examples.
  - [`main.log`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/main.log) shows the resulting `split_size` runtime error.
- Patched [`kernel_trainer.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/kernel_trainer.py) so that if post-adv filtering leaves an empty batch, the trainer logs the condition and skips actor/critic update for that step instead of entering `update_actor` with zero rows.
- Syntax-validated the patch and confirmed the guarded code path could continue training instead of entering `update_actor` with zero rows.

##### Result & Current State

- The crash root cause is fixed in the current worktree and recorded in commit `9da31c2`.
- The empty-batch guard has now survived later low-survivor steps in live training:
  - `step 269` completed after `Filtered batch: 768 -> 15 examples`
  - `step 270` completed after `Filtered batch: 768 -> 14 examples`
- The latest visible completed training step is now `270`, and the run is currently in-flight on `step 271`.
- This confirms the trainer no longer crashes when post-adv filtering produces a tiny surviving batch; the patched run is making real forward progress.

#### The `14B` A800 training topology is standardized on `50/51` with IB, validation disabled, and checkpoint eval isolated to `.18` — ACTIVE

##### Problem & Impact

- The `14B` eager run had previously bounced across multiple topologies and restart points, which mixed together training, reward recovery, and checkpoint evaluation work.
- A stable operating split was needed:
  - `192.168.16.50/51` for live training
  - `192.168.16.18` for checkpoint evaluation
  - validation disabled so training throughput is not penalized

##### Resolution

- Standardized the current training topology on [`a800_docker_50_51.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/infra_profiles/a800_docker_50_51.sh).
- Revalidated the actual launch environment inside both training containers:
  - `which python`
  - `sys.executable`
  - `VIRTUAL_ENV`
  - `ray` import
- Kept validation disabled in the training configuration:
  - `VAL_BEFORE_TRAIN=False`
  - `TEST_FREQ=0`
- Verified that the live training processes are not only configured for IB but are actually using it:
  - fresh launch log includes `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, and `NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_6,mlx5_7`
  - live worker processes on both `50` and `51` have `/dev/infiniband/uverbs2/3/6/7` open
- Moved checkpoint eval off the training nodes and onto `.18`.

##### Result & Current State

- The `50/51` training slice uses live IB workers and no longer shares checkpoint-eval work with the training nodes.
- `.18` is now reserved for checkpoint eval rather than live training.

#### Reward long-tail diagnosis was consolidated into one root-cause thread, and the immediate A800 incident was recovered — COMPLETED

##### Problem & Impact

- Both H20 and A800 runs showed extreme rollout long tails that initially looked like generic networking or relay issues.
- On the A800 path, reward host `192.168.16.40` also entered a host-level NVIDIA/UVM failure state, which collapsed reward worker capacity and amplified the same symptom.

##### Resolution

- Correlated the full training-side logs (`main.log`, `trainer.log`, `reward.log`, `vllm.log`) with reward-host state and host-level diagnostics.
- Proved that the long-tail pattern was not specific to the H20 relay path by reproducing the same symptom on the same-LAN A800 reward path.
- Narrowed the A800 incident root cause to two layers:
  - incident root cause: `16.40` host-level NVIDIA/UVM/Xid failure caused reward worker collapse
  - structural amplification: synchronous `/evaluate` holds the client-side request/token until HTTP returns, so reward-side stalls become `20-50+` minute rollout tails
- Recovered the A800 reward environment operationally by rebooting `16.39/16.40`, cleaning broken Docker state, remounting `/nfs/FM` on `16.40`, restarting reward, and restoring the training path afterward.
- Wrote the final incident report to [`HANDOFF_REWARD_SYNC_HTTP_LONGTAIL.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/handoffs/in_progress/HANDOFF_REWARD_SYNC_HTTP_LONGTAIL.md).

##### Result & Current State

- The immediate A800 reward outage was recovered and the current live training run is again using `http://192.168.16.39:8111`.
- The structural synchronous `/evaluate` risk remains documented in the handoff and should still be treated as a design constraint during future slowdown investigations.

#### The `8B` H20 training thread exposed three distinct failure classes after the early oversampling bottleneck was fixed — SUPERSEDED

##### Problem & Impact

- The original `8B` `12xH20` run failed early because prompt oversampling was too low to reliably produce enough selected samples.
- After that bottleneck was fixed, later resumes revealed three different failure classes:
  - reward relay loss on the Windows-hosted reverse path
  - async vLLM wake-up OOM
  - actor-side OOM caused by unrelated foreign GPU jobs on the worker training slice

##### Resolution

- Raised prompt oversampling to `2.0` so the run could pass the original low-batch retry loop.
- Resumed through the canonical launcher against the same run directory rather than creating disconnected follow-up runs.
- Reduced `ROLLOUT_GPU_MEMORY_UTIL` to `0.6` to clear the wake-up OOM path.
- Rechecked live GPU ownership on the worker node and separated the actor-side OOM from the earlier rollout-side memory issue.
- Regenerated timing plots and tied the later slow-step behavior to reward-backed rollout long-tail rather than actor update or `old_log_prob`.

##### Result & Current State

- This H20 thread is no longer the primary active operating focus.
- Its durable findings are preserved here and in the reward long-tail handoff rather than across many narrow restart entries.

#### Training and eval orchestration were consolidated onto canonical repo-local scripts and skills — COMPLETED

##### Problem & Impact

- Launch, stop, status checking, and plotting had drifted into ad hoc shell history and topology-specific operator habits.
- That made relaunches harder to reproduce and increased the risk of making claims from partial evidence such as a tmux pane or a short tail.

##### Resolution

- Added repo-local skills for:
  - start-training
  - stop-training
  - check-training-status
- Kept canonical launch/stop behavior anchored in repo scripts instead of hand-built orchestration:
  - [`start_training.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_training.sh)
  - [`.agents/skills/stop_training/scripts/stop_ray_training.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/.agents/skills/stop_training/scripts/stop_ray_training.sh)
- Moved local cluster default selection into untracked `.infra_profile.local.sh` rather than tracked infra defaults.
- Consolidated plotting into the canonical run-directory entrypoint and moved live status procedure details out of `AGENTS.md`.
- Added explicit timestamp prefixing in `log_router.py` so file-routed logs are easier to correlate.

##### Result & Current State

- Training launch, stop, and status checks now follow canonical repo-local paths.
- Local per-worktree cluster selection is separated from tracked infra files.
- Plot generation and live-status workflow are now reusable rather than embedded in one-off operator notes.

#### Checkpoint-eval layout and helpers were normalized around `eval_results/step_*` and automatic untested-step discovery — COMPLETED

##### Problem & Impact

- Checkpoint eval outputs and helper scripts had been inconsistent across runs, which complicated resume, plotting, backfill eval, and step-to-step comparison.
- The checkpoint-eval helper also previously required a hand-maintained step list, and its merge path had drifted between CPU and GPU variants.

##### Resolution

- Standardized eval outputs under the canonical `eval_results/step_*` layout for the active run family.
- Flattened the active 14B eager run layout so auto-resume resolves from a short stable checkpoint root.
- Updated [`merge_and_eval_checkpoints.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh) so that:
  - GPU merge is the only supported merge path
  - `EVAL_STEPS` is optional
  - when `EVAL_STEPS` is unset, the script automatically discovers untested checkpoints by scanning `CKPT_BASE/global_step_*` and skipping any step that already has `eval_results/step_*/metrics.json`
  - the summary formatter no longer uses the broken `${step:<6}` shell-expanded interpolation
- Verified GPU-merge equivalence against regenerated CPU output for `global_step_150`: tensor contents matched exactly and only shard packing differed.

##### Result & Current State

- The active checkpoint-eval flow now assumes GPU merge only.
- The current helper can be pointed at a checkpoint root and will discover missing tests automatically when explicit steps are not provided.
- `.18` completed the newly discovered missing `210/220` eval batch for the active 14B run.
- The first `step_230` attempt did not finish eval successfully: GPU merge completed, but vLLM worker startup failed on `.18` because stale `ray::AsyncActorRolloutRefWorker` processes from earlier eval sessions were still holding roughly `18-40 GiB` per GPU, leaving less free memory than the eval launcher's `gpu_memory_utilization=0.5` requirement.
- `.18` was then cleaned with `ray stop --force`, the stale eval workers were removed, partial `step_230` outputs were deleted, and the eval batch was relaunched.
- The rerun exposed a new blocker: `step_230` eval did not complete, and `240/250/260` merge attempts also failed because the shared `/nfs/FM` mount hit `Disk quota exceeded` during writeout of merged HF artifacts and summary updates.
- As a result:
  - `210/220` remain the latest completed eval checkpoints
  - `230/240/250/260` currently have no valid `metrics.json`
  - the active `.18` blocker is now storage quota, not stale GPU/Ray state

#### Working docs and handoffs were reorganized around active investigations instead of accumulating patch notes — COMPLETED

##### Problem & Impact

- Repo documentation had accumulated too many narrow, append-only notes, which made active investigations harder to read.

##### Resolution

- Split `handoffs/` into `completed` and `in_progress`.
- Reworked the reward long-tail handoff into a full bug report instead of incremental append-only notes.
- Tightened the entropy-collapse handoff so it separates paper semantics, release behavior, and current-run evidence.
- Kept stable repo-level guidance in harness files and left run-specific or transient details in handoffs and `SPEC.md`.

##### Result & Current State

- Current investigations now live in focused handoffs with clearer scope boundaries.
- `PROGRESS.md` has been compressed back into a decision log instead of a transcript.
