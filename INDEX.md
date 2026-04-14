# Index

## Runbooks And Handoffs

- [`handoffs/in_progress/HANDOFF_ONLINE_W8A8.md`](handoffs/in_progress/HANDOFF_ONLINE_W8A8.md)
  Primary runbook for the online-quantization investigation.
- [`handoffs/in_progress/HANDOFF_ENTROPY_COLLAPSE.md`](handoffs/in_progress/HANDOFF_ENTROPY_COLLAPSE.md)
  Entropy-collapse investigation, MRS/PRS interpretation, and likely root-cause notes.
- [`handoffs/in_progress/HANDOFF_UNCOMMITTED_CHANGESET_REVIEW.md`](handoffs/in_progress/HANDOFF_UNCOMMITTED_CHANGESET_REVIEW.md)
  Review summary for the current accumulated uncommitted changes and recommended commit split.
- [`handoffs/completed/HANDOFF_REMOTE_TRAIN_LOCAL_REWARD.md`](handoffs/completed/HANDOFF_REMOTE_TRAIN_LOCAL_REWARD.md)
  Completed report for the relay-based remote-cloud training and local reward topology.
- [`handoffs/completed/HANDOFF_RELEASE_3A84417_DIFF.md`](handoffs/completed/HANDOFF_RELEASE_3A84417_DIFF.md)
  Diff summary between the official release commit and local training codepaths.

## Training Launchers

- [`drkernel/kernel/scripts/rl/start_training.sh`](drkernel/kernel/scripts/rl/start_training.sh)
  Canonical multi-node training startup entrypoint; supports `--profile h20|a800` and should be used instead of hand-built SSH/tmux launch commands.
- [`drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh`](drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh)
  Canonical 14B eager training launcher used as the current baseline.
- [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh`](drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh)
  Current single-node H20 8B eager training launcher.
- [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.16xH20.sh`](drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.16xH20.sh)
  Current two-node H20 8B eager training launcher.
- [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.16xA800.sh`](drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.16xA800.sh)
  Two-node A800 8B eager training launcher.
- [`drkernel/kernel/scripts/rl/train_rl_common.sh`](drkernel/kernel/scripts/rl/train_rl_common.sh)
  Shared Hydra/config assembly for RL launches.

## Training Infra And Orchestration

- [`drkernel/kernel/scripts/rl/infra_common.sh`](drkernel/kernel/scripts/rl/infra_common.sh)
  Infra compatibility loader; selects the active cluster profile and exposes shared orchestration defaults to the startup and stop scripts.
- [`drkernel/kernel/scripts/rl/infra_lib.sh`](drkernel/kernel/scripts/rl/infra_lib.sh)
  Shared shell helpers for local execution, `ssh` plus `docker exec` targets, Python environment prelude, and cluster-side command routing.
- [`drkernel/kernel/scripts/rl/infra_profiles/h20.sh`](drkernel/kernel/scripts/rl/infra_profiles/h20.sh)
  H20 training-cluster defaults used by the current active training environment.
- [`drkernel/kernel/scripts/rl/infra_profiles/a800.sh`](drkernel/kernel/scripts/rl/infra_profiles/a800.sh)
  A800 training-cluster defaults used by the legacy two-node training and eval environment.
- [`.agents/skills/stop_training/scripts/stop_ray_training.sh`](.agents/skills/stop_training/scripts/stop_ray_training.sh)
  Canonical stop entrypoint for repo-managed training clusters; supports `--profile h20|a800`.

## Monitoring, Logging, And Eval

- [`drkernel/kernel/scripts/rl/plot_run_dynamics.py`](drkernel/kernel/scripts/rl/plot_run_dynamics.py)
  Canonical plotting entrypoint for training and eval dynamics.
- [`drkernel/kernel/scripts/rl/monitor_training_backoff.py`](drkernel/kernel/scripts/rl/monitor_training_backoff.py)
  Backoff-style training monitor for repeated progress checks and alert summaries.
- [`drkernel/kernel/scripts/rl/log_router.py`](drkernel/kernel/scripts/rl/log_router.py)
  Log fan-out rules for `main.log`, `trainer.log`, `reward.log`, and `vllm.log`.
- [`drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh`](drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh)
  Merge FSDP checkpoints and run checkpoint evaluation.
- [`drkernel/kernel/scripts/rl/compare_checkpoint_eval_runs.py`](drkernel/kernel/scripts/rl/compare_checkpoint_eval_runs.py)
  Overlay checkpoint-eval results from two runs on a single figure.

## Reward Stack

- [`drkernel/kernel/scripts/rl/start_reward.sh`](drkernel/kernel/scripts/rl/start_reward.sh)
  Reward service bootstrap and environment wiring.
- [`drkernel/kernel/rewards/reward_client.py`](drkernel/kernel/rewards/reward_client.py)
  Reward-side aggregation, coverage computation, and RPC client path.
- [`kernelgym/server/api/server.py`](kernelgym/server/api/server.py)
  Reward API server and reference-cache provider registration.
- [`kernelgym/workflow/reference_cache.py`](kernelgym/workflow/reference_cache.py)
  Shared reference-runtime cache provider.

## Core Training Codepaths

- [`drkernel/kernel/main_kernel.py`](drkernel/kernel/main_kernel.py)
  Main training entrypoint and trainer assembly.
- [`drkernel/kernel/kernel_trainer.py`](drkernel/kernel/kernel_trainer.py)
  TRLOO training loop, rollout, masking, and metrics.
- [`drkernel/kernel/workers/rollout/async_server.py`](drkernel/kernel/workers/rollout/async_server.py)
  Async rollout orchestration and prompt-row accounting.
- [`kernelgym/toolkit/kernelbench/profiling.py`](kernelgym/toolkit/kernelbench/profiling.py)
  Profiling-time aggregation, including current coverage denominator logic.
