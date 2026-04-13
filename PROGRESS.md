# Progress

## Remote training with local reward topology guidance — COMPLETED

#### The current architecture already supports remote rollout/PPO with local reward via HTTP, so the immediate blocker is network reachability rather than a new standalone orchestrator

##### Problem & Impact

- There was a question about whether cloud-side rollout and PPO update can be combined with a local 4090 reward server, or whether this would require first splitting out a new standalone orchestrator.
- Without clarifying where reward requests originate and which side performs PPO updates, it was easy to over-scope the problem into a full training-control-plane redesign.

##### Resolution

- Reviewed the current trainer, rollout, and reward-client boundaries.
- Documented the conclusion in [`handoffs/in_progress/HANDOFF_REMOTE_TRAIN_LOCAL_REWARD.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/handoffs/in_progress/HANDOFF_REMOTE_TRAIN_LOCAL_REWARD.md).
- Added repo-independent test methods for:
  - reward API reachability
  - reverse-tunnel validation
  - split-service orchestration sanity checks

##### Result & Current State

- The repo now has a canonical handoff for the "remote cloud training + local reward" deployment shape.
- The current recommendation is to validate VPN or tunnel reachability first and keep the trainer as the PPO orchestrator.
- A standalone rollout-self-update service is explicitly not the first-step recommendation.

## Paired checkpoint-eval comparison script — COMPLETED

#### A dedicated comparison script was added so two runs' checkpoint-eval curves can be overlaid on one figure with detected backend labels

##### Problem & Impact

- Checkpoint-eval plotting existed for a single run, but there was no repo-local script to directly overlay two runs on one figure.
- Comparing the 14B refcache baseline against the 14B pytorch-eager run required a reusable side-by-side plotting entrypoint rather than ad hoc notebook work.

##### Resolution

- Added [`drkernel/kernel/scripts/rl/compare_checkpoint_eval_runs.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/compare_checkpoint_eval_runs.py).
- The script accepts two run directories or two `eval_results` directories, resolves the canonical eval root, detects the latest `reference_backend` from each run's `trainer.log`, and overlays the checkpoint-eval metrics for `test_score pass` plus the last-turn-only `turn_3/*` series on one figure.
- Added the official dashed reference lines for `turn_3/fast@1_in_all` (`0.4038`) and `turn_3/fast@1.2_in_all` (`0.2400`).
- Indexed the new script in [`INDEX.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/INDEX.md).

##### Result & Current State

- The repo now has a canonical script for two-run checkpoint-eval comparisons.
- The comparison figure intentionally excludes `best_by_turn_3/*`, focuses on last-turn behavior plus the test-score pass curve, and uses a fixed 2x2 layout.
- It is suitable for comparing the 14B refcache (`torch_compile`) and 14B pytorch-eager (`pytorch`) runs on one figure against the official dashed speedup references, with concise legends such as `ref: compile`, `ref: eager`, and `official(step300)`.

## Refcache run eval results flattened to run_dir/eval_results — COMPLETED

#### The migrated 14B refcache run now uses the same top-level eval-results layout as the eager run, and the stray conversations-only export tree was merged into that canonical root

##### Problem & Impact

- The 14B refcache run at [`trloo-14b-hfsdp8-refcache...20260404-032344`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-refcache.train.16xA800.reward.16x4090.run.20260404-032344) still stored standard checkpoint-eval outputs under `checkpoints/eval_results`.
- The same run also had a second long-path conversations-only export tree rooted at `...conversations_conversations.jsonl...`, which split one logical eval history across two places.
- After the codebase was updated to treat `run_dir/eval_results` as canonical, leaving this historical run unmigrated would keep path handling inconsistent across the main 14B baselines.

##### Resolution

- Moved the refcache run's canonical eval tree from [`checkpoints/eval_results`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-refcache.train.16xA800.reward.16x4090.run.20260404-032344/checkpoints) to [`eval_results`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-refcache.train.16xA800.reward.16x4090.run.20260404-032344/eval_results).
- Merged the conversations-only files from the old long-path tree into the matching `step_*` directories under that canonical [`eval_results`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-refcache.train.16xA800.reward.16x4090.run.20260404-032344/eval_results) root.
- Normalized the merged conversation-export filename to `graded_results_conversations.jsonl` and removed the emptied long-path directory tree.

##### Result & Current State

- The refcache run now has a single canonical eval root at [`eval_results`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-refcache.train.16xA800.reward.16x4090.run.20260404-032344/eval_results).
- Each `step_*` directory now co-locates the standard eval artifacts and the conversation export.
- The old long-path `...conversations_conversations.jsonl...` tree for this run has been removed.

## Eval results root flattened to run_dir/eval_results — COMPLETED

#### The canonical eval output for run directories was moved out of `checkpoints/`, and the stray conversations-only export tree was merged into the same `eval_results/step_*` directories

##### Problem & Impact

- The canonical checkpoint-eval outputs for the 14B eager run were still stored under `checkpoints/eval_results`, which mixed eval artifacts into the checkpoint root.
- A second long-path directory rooted at `...conversations_conversations.jsonl...` also existed for the same run and contained only `graded_results_conversations_conversations.jsonl` files, so one logical eval run was split across two trees.
- That split caused path confusion and made tooling such as [`plot_run_dynamics.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/plot_run_dynamics.py) discover multiple `eval_results` candidates under the same run directory.

##### Resolution

- Moved the canonical eval output tree for [`trloo-14b-hfsdp8-pytorch-eager...20260409-092519`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519) from [`checkpoints/eval_results`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/checkpoints) to [`eval_results`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/eval_results).
- Merged the conversations-only files from the old long-path tree into the matching `step_*` directories under the new canonical [`eval_results`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/eval_results) root.
- Normalized the conversation-export filename to `graded_results_conversations.jsonl`.
- Updated [`drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh) so future checkpoint evals write to `run_dir/eval_results` instead of `checkpoints/eval_results`.
- Updated [`drkernel/kernel/scripts/rl/plot_run_dynamics.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/plot_run_dynamics.py) to prefer the canonical top-level `run_dir/eval_results` when it exists.
- Updated [`drkernel/kernel/main_grading.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/main_grading.py) so JSONL conversation exports do not append `_conversations` twice.

##### Result & Current State

- The 14B eager run now has a single canonical eval root at [`eval_results`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/eval_results).
- Each `step_*` directory now co-locates the standard eval artifacts and the conversation export, for example [`step_100`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/eval_results/step_100).
- The stray long-path `...conversations_conversations.jsonl...` tree for this run has been removed.
- Future checkpoint evals launched through the repo script will write to `run_dir/eval_results`, and plot discovery will treat that top-level directory as the canonical source.

## Stable checkpoint root under run_dir/checkpoints — COMPLETED

#### Checkpoint saving was decoupled from the long `RUN_NAME`, and the existing 14B eager run directory was flattened so the new layout can auto-resume directly

##### Problem & Impact

- Checkpoints were being stored under `RUN_LOG_DIR/<very long RUN_NAME>/global_step_N`, because the launcher pointed `HDFS_CHECKPOINT_PATH` at `RUN_LOG_DIR` and [`train_rl_common.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/train_rl_common.sh) appended `RUN_NAME` again.
- That produced long, unstable checkpoint roots and made the run directory harder to operate on than a stable `checkpoints/` subdirectory.
- The existing 14B eager run at [`drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519) was already saved in the old long layout, so changing code alone would not let the new layout auto-resume that run.

##### Resolution

- Changed [`drkernel/kernel/scripts/rl/train_rl_common.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/train_rl_common.sh) so that when `HDFS_CHECKPOINT_PATH` is set, `CHECKPOINT_DIR` is exactly that path instead of `HDFS_CHECKPOINT_PATH/RUN_NAME`.
- Changed the launchers that pin checkpoints under the run log directory to use `HDFS_CHECKPOINT_PATH="${RUN_LOG_DIR}/checkpoints"`, including:
  - [`drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh)
  - [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.sh)
  - [`drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_refcache.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_refcache.sh)
  - [`drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_syncdiag_minimal.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_syncdiag_minimal.sh)
- Renamed the existing 14B eager run's old long checkpoint directory in place to [`checkpoints`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/checkpoints), which moved all `global_step_*`, `eval_results`, and `latest_checkpointed_iteration.txt` under the stable root without copying the 3.3T payload.
- Applied the same in-place migration to the remaining old-layout runs under `drkernel/logs/`, including the 14B refcache run `20260404-032344` and the 8B eager exploratory run `20260412-002122`.
- Removed the now-empty old long parent directories after the rename.
- Validated the new layout with the repo venv by calling [`find_latest_ckpt_path`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/verl/verl/utils/checkpoint/checkpoint_manager.py), which resolved the latest checkpoint as `checkpoints/global_step_100`.

##### Result & Current State

- New runs launched through the updated eager/refcache/syncdiag scripts will save checkpoints under `RUN_LOG_DIR/checkpoints/global_step_N`.
- The existing 14B eager run directory now also uses that layout, and its tracker file is at [`checkpoints/latest_checkpointed_iteration.txt`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/checkpoints/latest_checkpointed_iteration.txt) with current value `100`.
- The remaining historical old-layout runs under `drkernel/logs/` were also migrated, and there are no longer any `latest_checkpointed_iteration.txt` files outside `checkpoints/` roots under `drkernel/logs/`.
- Auto-resume now works against the flattened root for this run, because the new `default_local_dir` and the migrated tracker/`global_step_*` layout are aligned.

## 14B checkpoint eval for steps 80-100 on node 16.18 — COMPLETED

#### Checkpoint evaluation for `global_step_80`, `90`, and `100` was relaunched as a single-node run on `192.168.16.18` because the second A800 node is currently unavailable through the stored SSH endpoint

##### Problem & Impact

- The next requested checkpoint eval range is `80-100` for the 14B eager run at [`drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519).
- `global_step_80`, `90`, and `100` are present, but the previous two-node eval path could not be reused because `root@192.168.16.24:14218` is currently refusing connections.
- The user confirmed that this eval should use `REWARD_TASK_TIMEOUT=30`.

##### Resolution

- Verified on `192.168.16.18` that `which python`, `sys.executable`, reward health, the eval dataset path, and the checkpoint root are all reachable from the node that will execute the eval.
- Wrote a single-node checkpoint-eval runner for steps `80`, `90`, and `100` to `/tmp/eval_14b_ckpts_80_100_node18.sh` on `192.168.16.18`.
- Launched that runner in head-node tmux session `eval-14b-80-100-node18`.
- Passed `--reward_task_timeout 30` into the eval command for each checkpoint.

##### Result & Current State

- The active eval is now running on `192.168.16.18` only, with results written under the migrated `eval_results/` tree for the 14B eager run.
- The current orchestrator log is [`orchestrator.20260412-235932.step80_100.node18.log`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/eval_results/orchestrator.20260412-235932.step80_100.node18.log).
- The single-node eval run has finished and there is no remaining `eval_14b_ckpts_80_100_node18`, `kernel.main_grading`, or `verl.model_merger` process on `192.168.16.18`.
- New metrics are now present for `step_80`, `step_90`, and `step_100` under [`eval_results`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/eval_results).
- Key results from this batch are:
  - `step_80`: `pass@1=0.74`, `correctness=0.59875`, `fast@1.2_in_all=0.28375`
  - `step_90`: `pass@1=0.7233`, `correctness=0.60375`, `fast@1.2_in_all=0.27875`
  - `step_100`: `pass@1=0.79`, `correctness=0.59875`, `fast@1.2_in_all=0.29375`

## Canonical training startup skill and script parameterization — COMPLETED

#### Training startup was converted into a repo-local skill, and the top-level startup script was de-hardcoded so launches and resumes can use the existing launchers without hand-built orchestration

##### Problem & Impact

- Training launches were still easy to do incorrectly because the orchestration often fell back to manually assembled SSH/tmux commands.
- The existing [`drkernel/kernel/scripts/rl/start_training.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_training.sh) was hardcoded to one specific launcher and tmux/log naming scheme, which made resume flows and non-default launchers awkward.

##### Resolution

- Added the repo-local skill [`skills/start_training/SKILL.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/skills/start_training/SKILL.md).
- Reworked [`drkernel/kernel/scripts/rl/start_training.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_training.sh) into a generic orchestrator that still uses the repo's existing launcher scripts but now supports:
  - `--train-script PATH`
  - `--skip-reward`
  - `-f/--force-reward`
  - repeated `--env KEY=VALUE` overrides
  - `--tmux-session NAME`
  - `--local-log PATH`
  - `--dry-run`
- Updated [`INDEX.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/INDEX.md) so the canonical startup entrypoint is indexed with the launchers.

##### Result & Current State

- Training startup for this repo now has a standard skill and a standard script entrypoint.
- Launches can stay on the existing repo scripts while still handling resume-time overrides such as `RUN_LOG_DIR` and `REWARD_TASK_TIMEOUT` without hand-built orchestration.
- The startup script passed `bash -n`, `--help`, and `--dry-run` validation after the parameterization change.

## 14B eager resume with shorter reward timeout — ACTIVE

#### The active training target was switched from the 8B exploratory run back to the original 14B eager run, with `REWARD_TASK_TIMEOUT` reduced to `30` during resume

##### Problem & Impact

- The active training target needed to be switched away from the current 8B exploratory run and back to the original 14B eager run at [`drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519).
- The resumed 14B run also needed a shorter reward-side task timeout so long-tail reward jobs would be cut off at `30s` instead of the original `300s`.

##### Resolution

- Stopped the active 8B training through the repo stop script and cleared the local 8B backoff-monitor tmux session.
- Rebuilt the 2-node Ray cluster on the A800 training nodes.
- Relaunched the 14B eager launcher in a dedicated head-node tmux session with:
  - `RUN_LOG_DIR=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519`
  - `REWARD_TASK_TIMEOUT=30`
- Verified on the head node that `which python`, `sys.executable`, `VIRTUAL_ENV`, reward health, model path, dataset paths, and the old run directory were all valid before relaunch.
- Verified the live launcher environment via `/proc/<pid>/environ` and the new trainer config dump, both of which show `REWARD_TASK_TIMEOUT=30`.

##### Result & Current State

- The 8B training process is stopped and the 14B eager run remains the preserved target run under [`drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519).
- The relaunch confirmed `reward_model.task_timeout: 30` in [`trainer.log`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/trainer.log), and the run's checkpoint state is now tracked under [`checkpoints`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/checkpoints).
- The latest saved checkpoint currently present is `global_step_100`.
- There is currently no active head-node training process attached to this run.

## Routed log timestamp prefix — COMPLETED

#### File logs routed through `log_router.py` now get an explicit timestamp prefix at write time

##### Problem & Impact

- Training file logs such as `main.log`, `trainer.log`, `rollout.log`, `reward.log`, and `vllm.log` were plain routed stdout without explicit timestamps.
- That made it harder to reason about interleaving, timeout gaps, and long-tail stalls from the file logs alone.

##### Resolution

- Added timestamp prefixing in [`drkernel/kernel/scripts/rl/log_router.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/log_router.py).
- The router now prefixes each routed file-log line with a local ISO timestamp by default.
- Added `LOG_ROUTER_TIMESTAMP_MODE=none` as an escape hatch to disable prefixing if needed.

##### Result & Current State

- New lines written through `log_router.py` now carry timestamps in routed file logs.
- Existing historical logs are unchanged, and already-running router processes must be restarted to pick up the change.

## Handoff directory split and pending changeset review — COMPLETED

#### `handoffs/` was split into `completed` and `in_progress`, and the pending top-level changeset review was written as a dedicated handoff

##### Problem & Impact

- The repository's handoff notes were all stored flat under `handoffs/`, so active investigations and finished notes were mixed together.
- The recent review of accumulated uncommitted top-level changes also needed to be preserved as a handoff document rather than remaining only in chat.

##### Resolution

- Split `handoffs/` into [`handoffs/completed`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/handoffs/completed) and [`handoffs/in_progress`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/handoffs/in_progress).
- Moved existing handoff files into the appropriate subdirectory based on their current status.
- Added [`handoffs/in_progress/HANDOFF_UNCOMMITTED_CHANGESET_REVIEW.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/handoffs/in_progress/HANDOFF_UNCOMMITTED_CHANGESET_REVIEW.md) to capture the current top-level changeset review and recommended commit split.
- Updated references in [`AGENTS.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/AGENTS.md), [`INDEX.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/INDEX.md), [`SPEC.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/SPEC.md), and relevant handoff docs to the new paths.

##### Result & Current State

- Handoff docs are now separated into finished and active investigations.
- The online-W8A8 primary runbook path now points to the new `handoffs/in_progress/` location.
- The pending top-level changeset review is now preserved as a reusable handoff document instead of being only a transient conversation result.

## Training-status checking skill extraction — COMPLETED

#### Live training-status procedure was moved out of `AGENTS.md` into a repo-local skill

##### Problem & Impact

- `AGENTS.md` still contained a detailed operational workflow for checking live training status.
- That workflow was more procedural than repository-stable and made `AGENTS.md` too specific.

##### Resolution

- Removed the detailed training-status checking rules from [`AGENTS.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/AGENTS.md).
- Added the repo-local skill [`skills/check_training_status/SKILL.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/skills/check_training_status/SKILL.md).
- Added the skill entry to [`INDEX.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/INDEX.md).

##### Result & Current State

- `AGENTS.md` now keeps only the stable repository-level rule that detailed training-status procedures belong in a skill.
- The concrete training-status workflow now lives in a reusable repo-local skill.

## Log router summary matching fix — COMPLETED

#### `main.log` no longer mirrors vLLM kernel tracebacks just because their source code contains `Selected ...` comments

##### Problem & Impact

- `main.log` still contained some kernel-level Triton traceback lines even though those lines were already routed to `vllm.log`.
- The root cause was an overly broad summary keyword in `log_router.py`: the plain substring `Selected ` also matched traceback source-code snippets such as `# Selected lane mask`.

##### Resolution

- Removed the broad `Selected ` summary keyword from [`log_router.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/log_router.py).
- Replaced it with a targeted regex that only keeps real oversampling summary lines of the form `[Oversampling] Selected X of Y required samples ...` in `main.log`.

##### Result & Current State

- Real vLLM kernel traceback lines now route only to `vllm.log`.
- Real oversampling summary lines still route to `main.log` and `trainer.log`.

## 8B eager launcher and reference cache — ACTIVE

#### The 8B eager baseline was corrected and reward-side reference cache was turned into a real runtime feature

##### Problem & Impact

- The first 8B launcher diverged from the requested 14B eager baseline in the wrong place:
  `TRAIN_BATCH_SIZE` / `PPO_MINI_BATCH_SIZE` changed, instead of only enlarging old/ref log-prob
  micro-batch size.
- The 8B path also needed to be pinned to the local coldstart checkpoint rather than an HF path.
- Reference cache was enabled in trainer config, but the reward service initially had no effective
  provider registration, so reference timing still ran as if every lookup missed.

##### Resolution

- Standardized the 8B launcher on the intended baseline:
  - `MODEL_PATH=/nfs/FM/chenshuailin/checkpoints/hkust-nlp/drkernel-8b-coldstart`
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=32`
  - `REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=32`
- Added service-side reference-cache support through:
  - [`kernelgym/workflow/reference_cache.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/workflow/reference_cache.py)
  - [`kernelgym/server/api/server.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/server/api/server.py)
  - [`kernelgym/workflow/kernelbench.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/workflow/kernelbench.py)
  - [`drkernel/kernel/scripts/rl/start_reward.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_reward.sh)

##### Result & Current State

- The 8B eager launcher now matches the intended baseline shape.
- `prompt_rows` differences are explained by prompt oversampling, not by model size or ref backend.
- Reference cache is now a real reward-service path rather than only a trainer-side flag.

## 8B eager training stabilization — SUPERSEDED

#### The current 8B run uses smaller actor token budget, shorter reward timeout, disabled validation, and higher prompt oversampling

##### Problem & Impact

- `PPO_MICRO_TOKEN=16384` caused actor-backward OOM in the 8B eager run.
- `REWARD_TASK_TIMEOUT=300` kept reward-side long-tail samples too expensive.
- Validation and repeated relaunches obscured whether launcher changes had actually landed.

##### Resolution

- Reduced actor token budget to `PPO_MICRO_TOKEN=8192`.
- Reduced reward timeout to `REWARD_TASK_TIMEOUT=30`.
- Disabled validation via CLI:
  - `--val_before_train False`
  - `--test_freq -1`
- Increased prompt-level oversampling to `PROMPT_OVERSAMPLING_FACTOR=2.0`.
- Strengthened
  [`monitor_training_backoff.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/monitor_training_backoff.py)
  so it writes the full log, `*.latest`, and `*.alerts`.

##### Result & Current State

- The exploratory 8B run at [`trloo-8b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260412-002122`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260412-002122) has been stopped after the launcher/stability experiments were completed.
- Active 8B launcher settings now include:
  - `TRAIN_BATCH_SIZE=16`
  - `PPO_MINI_BATCH_SIZE=16`
  - `ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=32`
  - `REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=32`
  - `PPO_MICRO_TOKEN=8192`
  - `REWARD_TASK_TIMEOUT=30`
  - `PROMPT_OVERSAMPLING_FACTOR=2.0`
- Current state:
  - the launcher and reward-timeout adjustments remain documented here as the final 8B experiment settings
  - the 8B training and its monitor sessions are no longer the active run

## Repo-local stop-training skill — COMPLETED

#### Stopping distributed training was turned into a repo-local reusable skill instead of remaining ad hoc shell history

##### Problem & Impact

- The A800 stop procedure had converged on a specific workflow, but it only existed in terminal
  history and was easy to repeat incorrectly.

##### Resolution

- Added:
  - [`skills/stop_training/SKILL.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/skills/stop_training/SKILL.md)
  - [`skills/stop_training/scripts/stop_ray_training.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/skills/stop_training/scripts/stop_ray_training.sh)

##### Result & Current State

- The repo now has a canonical stop path:
  `bash skills/stop_training/scripts/stop_ray_training.sh`

## Eager checkpoint eval summary — COMPLETED

#### The stopped 14B eager run was evaluated across saved checkpoints and the best checkpoints are now known

##### Problem & Impact

- After stopping the 14B eager run, checkpoint quality had to be measured before choosing any
  later comparison baseline.

##### Resolution

- Merged and evaluated checkpoints `10, 20, 30, 40, 50, 60, 70` across the two A800 nodes.
- Generated eval plots under the run's `eval_results/plots` directory.

##### Result & Current State

- Best `pass@1` and `correctness`: `step_60`
- Best `fast@1_in_all`: `step_70`
- Best `fast@1.2_in_all`: `step_40`

## Training and eval plotting utilities — COMPLETED

#### Training/eval plotting was consolidated into a single run-directory entrypoint with grouped output layout

##### Problem & Impact

- Earlier plotting relied on two separate scripts and spread the operator workflow across a
  training-log path and a separate `eval_results` path.
- The repo also needed `train/*` and `val/*` metrics split into separate subdirectories while
  keeping raw log names as plot titles.

##### Resolution

- Consolidated training and checkpoint-eval plotting into
  `the canonical plotting entrypoint documented in INDEX.md`.
- The merged entrypoint now accepts a run directory, `main.log`, or `eval_results` directory and:
  - writes training plots under `training_dynamics_plots_by_group`
  - auto-discovers `eval_results` and writes eval plots under each `plots/` directory
  - keeps subplot titles aligned with raw log suffixes
  - keeps `train/*` and `val/*` metrics split into separate subdirectories
- Removed the obsolete standalone training-only and eval-only plotting scripts.

##### Result & Current State

- There is now one canonical plotting entrypoint for run directories.
- The canonical script path is recorded in `INDEX.md`, not `AGENTS.md`.

## Entropy-collapse analysis updates — COMPLETED

#### The handoff was tightened so it distinguishes paper semantics, release behavior, and current-run evidence

##### Problem & Impact

- Earlier entropy-collapse notes mixed paper-level MRS semantics, release-code behavior, and
  current training observations too loosely.

##### Resolution

- Updated
  [`HANDOFF_ENTROPY_COLLAPSE.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/handoffs/HANDOFF_ENTROPY_COLLAPSE.md)
  with:
  - paper-path references
  - sequence-level MRS clarification
  - masked entropy vs raw entropy distinction
  - official release/backend findings
  - author issue response
  - coverage denominator hypothesis

##### Result & Current State

- The handoff now separates direct evidence from inference and points to the relevant code and paper
  locations.

## Repository code quality report translated to Chinese — COMPLETED

#### The repo-level quality report was rewritten into Chinese for local handoff use

##### Problem & Impact

- The initial report content was useful but in the wrong language for the requested workflow.

##### Resolution

- Rewrote
  [`CODE_QUALITY_REPORT.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/CODE_QUALITY_REPORT.md)
  into Chinese without changing the underlying conclusions.

##### Result & Current State

- The report is now directly usable in the current repo workflow.

## Repository code quality assessment — COMPLETED

#### Static review identified `drkernel/kernel` as the main maintainability risk area

##### Problem & Impact

- The repository needed a first-party code-quality assessment that separated local code from
  vendored upstream code.

##### Resolution

- Performed static review of `kernelgym` and `drkernel/kernel`.
- Verified first-party Python syntax with `python -m compileall -q kernelgym drkernel/kernel`.
- Wrote the assessment to
  [`CODE_QUALITY_REPORT.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/CODE_QUALITY_REPORT.md).

##### Result & Current State

- `kernelgym` remains structurally stronger.
- `drkernel/kernel` remains the main maintainability hotspot.
- Repo-level automated quality enforcement for first-party code is still weak.
