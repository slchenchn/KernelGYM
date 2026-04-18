# Progress

#### H20 checkpoint eval on `.3` now tolerates split-disk checkpoint storage and stale hardcoded validation-data paths, and the head-only two-GPU eval was relaunched on `CUDA_VISIBLE_DEVICES=6,7` — ACTIVE

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
- Relaunched checkpoint eval under tmux session `eval-8b-ckpt-gpu67` with:
  - `CUDA_VISIBLE_DEVICES=6,7`
  - `MERGE_CUDA_VISIBLE_DEVICES=6`
  - `EVAL_USE_WORKER=0`
  - `NNODES=1`
  - `N_GPUS_PER_NODE=2`
  - `REWARD_SERVER_URL=http://10.0.18.3:18112`

##### Result & Current State

- The new eval session is active on `.3` and is again processing the untested checkpoints `10 50 100 140`.
- Training remains isolated on GPUs `0-5`; the eval relaunch is constrained to the remaining two GPUs.
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

#### The `14B` A800 eager run is active on `50/51` with IB, validation disabled, and checkpoint eval isolated to `.18` — ACTIVE

##### Problem & Impact

- The `14B` eager run had previously bounced across multiple topologies and restart points, which mixed together training, reward recovery, and checkpoint evaluation work.
- A stable operating split was needed:
  - `192.168.16.50/51` for live training
  - `192.168.16.18` for checkpoint evaluation
  - validation disabled during resume so training throughput is not penalized

##### Resolution

- Standardized the current training topology on [`a800_docker_50_51.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/infra_profiles/a800_docker_50_51.sh) and resumed through the canonical launcher [`start_training.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_training.sh).
- Revalidated the actual launch environment inside both training containers before relaunch:
  - `which python`
  - `sys.executable`
  - `VIRTUAL_ENV`
  - `ray` import
- Reused the existing run directory [`trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519) instead of creating a fresh run.
- Kept validation disabled at resume time:
  - `VAL_BEFORE_TRAIN=False`
  - `TEST_FREQ=0`
- Verified that the live training processes are not only configured for IB but are actually using it:
  - fresh launch log includes `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, and `NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_6,mlx5_7`
  - live worker processes on both `50` and `51` have `/dev/infiniband/uverbs2/3/6/7` open
- Moved checkpoint eval off the training nodes and onto `.18`.

##### Result & Current State

- The live `14B` run is currently training on `50/51` under tmux session `train-14b-hfsdp8-pytorch-eager-resume-5051`.
- `ray status` on the training cluster shows both nodes active with all `16` training GPUs in use.
- The checkpoint tracker for the active run currently points at `global_step_160`.
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
- Recovered the A800 reward environment operationally by rebooting `16.39/16.40`, cleaning broken Docker state, remounting `/nfs/FM` on `16.40`, restarting reward, and then resuming training.
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
- `.18` is currently running the missing `170/180/190/200` eval batch for the active 14B run, and that batch was launched without `EVAL_STEPS`, so the step list came from automatic untested-checkpoint discovery rather than a hand-maintained list.

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
