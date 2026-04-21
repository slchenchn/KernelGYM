# Index

## Runbooks And Handoffs

- [`handoffs/in_progress/HANDOFF_TRITON_TO_CUDA_RL.md`](handoffs/in_progress/HANDOFF_TRITON_TO_CUDA_RL.md)
  Current CUDA-Agent migration handoff covering Triton-to-CUDA reward/rollout changes, full-set sweep results, compile-path caveats, timeout/reference-cache defaults, max-model-len interpretation, and Qwen3.5 chat-template notes.
- [`handoffs/in_progress/HANDOFF_REWARD_SYNC_HTTP_LONGTAIL.md`](handoffs/in_progress/HANDOFF_REWARD_SYNC_HTTP_LONGTAIL.md)
  Root-cause handoff for the reward `/evaluate` long-tail stall where client-side synchronous waits hold tokens and stretch rollout steps into `50-60+ min`.
- [`handoffs/in_progress/HANDOFF_REWARD_ENV_ROBUSTNESS.md`](handoffs/in_progress/HANDOFF_REWARD_ENV_ROBUSTNESS.md)
  Root-cause and hardening handoff for reward-host GPU faulting, `Xid 109` escalation, reboot-triggered outages, and concrete robustness actions for `39/40`.
- [`handoffs/in_progress/HANDOFF_QUANTIZED_ROLLOUT.md`](handoffs/in_progress/HANDOFF_QUANTIZED_ROLLOUT.md)
  Primary runbook for the online-quantization investigation, including the active `W8A8` ladder and the preserved `W8A16` packing-format branch.
- [`handoffs/in_progress/HANDOFF_ENTROPY_COLLAPSE.md`](handoffs/in_progress/HANDOFF_ENTROPY_COLLAPSE.md)
  Entropy-collapse investigation, MRS/PRS interpretation, and likely root-cause notes.
- [`handoffs/completed/HANDOFF_REMOTE_TRAIN_LOCAL_REWARD.md`](handoffs/completed/HANDOFF_REMOTE_TRAIN_LOCAL_REWARD.md)
  Completed report for the relay-based remote-cloud training and local reward topology.
- [`handoffs/completed/HANDOFF_2NODE_ROLLOUT.md`](handoffs/completed/HANDOFF_2NODE_ROLLOUT.md)
  Historical benchmark note for the completed 2026-03 14B two-node A800 rollout investigation; useful for archived performance conclusions, not as a current launch runbook.
- [`handoffs/completed/HANDOFF_RELEASE_3A84417_DIFF.md`](handoffs/completed/HANDOFF_RELEASE_3A84417_DIFF.md)
  Diff summary between the official release commit and local training codepaths.

## Training Launchers

- [`drkernel/kernel/scripts/rl/start_training.sh`](drkernel/kernel/scripts/rl/start_training.sh)
  Canonical training startup entrypoint; supports `--profile h20|a800`, `--single-node`, and should be used instead of hand-built SSH/tmux launch commands.
- [`drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh`](drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh)
  Canonical 14B eager training launcher used as the current baseline.
- [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh`](drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh)
  Current single-node H20 8B eager training launcher.
- [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.16xH20.sh`](drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.16xH20.sh)
  Current two-node H20 8B eager training launcher.
- [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.16xA800.sh`](drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.16xA800.sh)
  Two-node A800 8B eager training launcher.
- [`drkernel/kernel/scripts/rl/train_rl_common.sh`](drkernel/kernel/scripts/rl/train_rl_common.sh)
  Shared Hydra/config assembly for RL launches, including Ray runtime env propagation for NCCL transport settings.

## Training Infra And Orchestration

- [`drkernel/kernel/scripts/rl/infra_common.sh`](drkernel/kernel/scripts/rl/infra_common.sh)
  Infra compatibility loader; selects the active cluster profile and exposes shared orchestration defaults to the startup and stop scripts.
- [`drkernel/kernel/scripts/rl/infra_lib.sh`](drkernel/kernel/scripts/rl/infra_lib.sh)
  Shared shell helpers for local execution, `ssh` plus `docker exec` targets, Python environment prelude, and cluster-side command routing.
- [`drkernel/kernel/scripts/rl/infra_profiles/h20.sh`](drkernel/kernel/scripts/rl/infra_profiles/h20.sh)
  H20 training-cluster defaults used by the current active training environment.
- [`drkernel/kernel/scripts/rl/infra_profiles/a800.sh`](drkernel/kernel/scripts/rl/infra_profiles/a800.sh)
  A800 training-cluster defaults used by the legacy two-node training and eval environment.
- [`drkernel/kernel/scripts/rl/infra_profiles/a800_docker_50_51.sh`](drkernel/kernel/scripts/rl/infra_profiles/a800_docker_50_51.sh)
  Docker-based A800 training-cluster defaults for `192.168.16.50/51`, preserving the legacy `a800` profile for fallback.
- [`drkernel/kernel/scripts/rl/infra_profiles/a800_18_51_socket.sh`](drkernel/kernel/scripts/rl/infra_profiles/a800_18_51_socket.sh)
  Mixed A800 profile for `192.168.16.18` head plus `192.168.16.51` docker worker, explicitly using socket NCCL and no IB.
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
  Reward-side aggregation, coverage computation, RPC client path, and the restored failure-type penalty routing for `precheck` versus compilation versus generic failures.
- [`drkernel/kernel/config/kernel_trainer.yaml`](drkernel/kernel/config/kernel_trainer.yaml)
  Main training reward configuration, including the active `apply_precheck_fail_penalty` / `apply_compilation_fail_penalty` toggles and penalty values.
- [`drkernel/kernel/config/kernel_grading.yaml`](drkernel/kernel/config/kernel_grading.yaml)
  Standalone grading reward configuration with the same failure-type penalty routing controls used for offline evaluation.
- [`drkernel/kernel/rewards/kernel_reward.py`](drkernel/kernel/rewards/kernel_reward.py)
  Reward extraction logic that now supports both Triton-style Python answers and CUDA-Agent multi-section answers.
- [`drkernel/test_cuda_reward.py`](drkernel/test_cuda_reward.py)
  Repo-local single-sample CUDA reward smoke harness for validating the `cuda_agent` path against a real sample on a GPU node.
- [`drkernel/run_cuda_reward_dir.py`](drkernel/run_cuda_reward_dir.py)
  Repo-local batch harness for scoring a directory of stored `result_*.json` CUDA samples through the repaired reward pipeline, with local `precheck_fail` routing and resumable JSONL output.
- [`tests/test_cuda_agent_support.py`](tests/test_cuda_agent_support.py)
  CUDA reward-path regression suite covering extraction, decoy detection, profiling-based reward normalization, and lazy-optimization filtering via coverage-based rejection sampling.
- [`tests/fixtures/cuda_agent_support/`](tests/fixtures/cuda_agent_support/)
  Dedicated CUDA / binding / model fixtures used by the CUDA reward-path regression suite.
- [`drkernel/kernel/utils/kernel_code.py`](drkernel/kernel/utils/kernel_code.py)
  Shared kernel-submission extraction helpers used by both reward scoring and the rollout agent.
- [`kernelgym/server/api/server.py`](kernelgym/server/api/server.py)
  Reward API server and reference-cache provider registration.
- [`kernelgym/workflow/reference_cache.py`](kernelgym/workflow/reference_cache.py)
  Shared reference-runtime cache provider.
- [`kernelgym/backend/kernelbench/cuda_agent_backend.py`](kernelgym/backend/kernelbench/cuda_agent_backend.py)
  CUDA-Agent backend implementation for compiling and loading multi-file CUDA submissions inside the current `kernelgym` backend abstraction.
- [`kernelgym/toolkit/kernelbench/pipeline.py`](kernelgym/toolkit/kernelbench/pipeline.py)
  KernelBench evaluation pipeline, including the performance/profiling path that now skips Triton-only coverage logic for `cuda_agent` runs.

## Core Training Codepaths

- [`drkernel/kernel/main_kernel.py`](drkernel/kernel/main_kernel.py)
  Main training entrypoint and trainer assembly.
- [`drkernel/kernel/kernel_trainer.py`](drkernel/kernel/kernel_trainer.py)
  TRLOO training loop, rollout, masking, and metrics.
- [`drkernel/kernel/workers/rollout/async_server.py`](drkernel/kernel/workers/rollout/async_server.py)
  Async rollout orchestration and prompt-row accounting.
- [`drkernel/kernel/workers/agent/kernel_agent.py`](drkernel/kernel/workers/agent/kernel_agent.py)
  Final-answer extraction for kernel rollout turns, including the CUDA-Agent triple-section format.
- [`kernelgym/toolkit/kernelbench/profiling.py`](kernelgym/toolkit/kernelbench/profiling.py)
  Profiling-time aggregation, including current coverage denominator logic.

## CUDA RL Entry Points

- [`drkernel/kernel/config/cuda_kernel_trainer.yaml`](drkernel/kernel/config/cuda_kernel_trainer.yaml)
  Minimal Hydra overlay that enables `kernel_backend: "cuda_agent"` and switches the rollout prompt to the CUDA-specific format.
- [`drkernel/kernel/config/prompt_config/multi_turn_cuda_kernel.yaml`](drkernel/kernel/config/prompt_config/multi_turn_cuda_kernel.yaml)
  CUDA-specific multi-turn prompt that asks for `CUDA_KERNELS`, `APPLY_BINDINGS`, and `MODEL_NEW` sections.
