# Index

## Runbooks And Handoffs

- [`handoffs/in_progress/HANDOFF_ONLINE_W8A8.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/handoffs/in_progress/HANDOFF_ONLINE_W8A8.md)
  Primary runbook for the online-quantization investigation.
- [`handoffs/in_progress/HANDOFF_ENTROPY_COLLAPSE.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/handoffs/in_progress/HANDOFF_ENTROPY_COLLAPSE.md)
  Entropy-collapse investigation, MRS/PRS interpretation, and likely root-cause notes.
- [`handoffs/in_progress/HANDOFF_UNCOMMITTED_CHANGESET_REVIEW.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/handoffs/in_progress/HANDOFF_UNCOMMITTED_CHANGESET_REVIEW.md)
  Review summary for the current accumulated uncommitted changes and recommended commit split.
- [`handoffs/in_progress/HANDOFF_REMOTE_TRAIN_LOCAL_REWARD.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/handoffs/in_progress/HANDOFF_REMOTE_TRAIN_LOCAL_REWARD.md)
  Remote-cloud training with local reward guidance, including repo-independent topology tests.
- [`handoffs/completed/HANDOFF_RELEASE_3A84417_DIFF.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/handoffs/completed/HANDOFF_RELEASE_3A84417_DIFF.md)
  Diff summary between the official release commit and local training codepaths.

## Training Launchers

- [`drkernel/kernel/scripts/rl/start_training.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_training.sh)
  Canonical multi-node training startup entrypoint; use this instead of hand-built SSH/tmux launch commands.
- [`drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh)
  Canonical 14B eager training launcher used as the current baseline.
- [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.sh)
  Current 8B eager training launcher derived from the 14B eager baseline.
- [`drkernel/kernel/scripts/rl/train_rl_common.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/train_rl_common.sh)
  Shared Hydra/config assembly for RL launches.

## Monitoring, Logging, And Eval

- [`drkernel/kernel/scripts/rl/plot_run_dynamics.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/plot_run_dynamics.py)
  Canonical plotting entrypoint for training and eval dynamics.
- [`drkernel/kernel/scripts/rl/monitor_training_backoff.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/monitor_training_backoff.py)
  Backoff-style training monitor for repeated progress checks and alert summaries.
- [`drkernel/kernel/scripts/rl/log_router.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/log_router.py)
  Log fan-out rules for `main.log`, `trainer.log`, `reward.log`, and `vllm.log`.
- [`drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh)
  Merge FSDP checkpoints and run checkpoint evaluation.
- [`drkernel/kernel/scripts/rl/compare_checkpoint_eval_runs.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/compare_checkpoint_eval_runs.py)
  Overlay checkpoint-eval results from two runs on a single figure.

## Reward Stack

- [`drkernel/kernel/scripts/rl/start_reward.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_reward.sh)
  Reward service bootstrap and environment wiring.
- [`drkernel/kernel/rewards/reward_client.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/rewards/reward_client.py)
  Reward-side aggregation, coverage computation, and RPC client path.
- [`kernelgym/server/api/server.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/server/api/server.py)
  Reward API server and reference-cache provider registration.
- [`kernelgym/workflow/reference_cache.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/workflow/reference_cache.py)
  Shared reference-runtime cache provider.

## Core Training Codepaths

- [`drkernel/kernel/main_kernel.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/main_kernel.py)
  Main training entrypoint and trainer assembly.
- [`drkernel/kernel/kernel_trainer.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/kernel_trainer.py)
  TRLOO training loop, rollout, masking, and metrics.
- [`drkernel/kernel/workers/rollout/async_server.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/rollout/async_server.py)
  Async rollout orchestration and prompt-row accounting.
- [`kernelgym/toolkit/kernelbench/profiling.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/profiling.py)
  Profiling-time aggregation, including current coverage denominator logic.
