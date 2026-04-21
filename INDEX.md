# Index

This file only lists high-signal entry points. Detailed history, machine-specific
paths, and one-off investigation notes belong in `PROGRESS.md`, `SPEC.md`, or the
active handoff.

## Current Handoff

- [`handoffs/in_progress/HANDOFF_TRITON_TO_CUDA_RL.md`](handoffs/in_progress/HANDOFF_TRITON_TO_CUDA_RL.md)
  Primary CUDA-Agent migration handoff: reward parity, CUDA prompt/data split,
  timeout/reference-cache choices, max-model-len notes, and verification status.

## Launch And Config

- [`drkernel/kernel/scripts/rl/14b_coldstart_trloo_mrs_pr_prs.sh`](drkernel/kernel/scripts/rl/14b_coldstart_trloo_mrs_pr_prs.sh)
  Current CUDA RL launcher.
- [`drkernel/kernel/scripts/rl/train_rl_common.sh`](drkernel/kernel/scripts/rl/train_rl_common.sh)
  Shared Hydra/config assembly used by RL launchers.
- [`drkernel/kernel/scripts/rl/start_reward.sh`](drkernel/kernel/scripts/rl/start_reward.sh)
  Reward service startup entrypoint.
- [`drkernel/kernel/config/cuda_kernel_trainer.yaml`](drkernel/kernel/config/cuda_kernel_trainer.yaml)
  CUDA RL config overlay.

## CUDA Reward Path

- [`kernelgym/backend/kernelbench/cuda_agent_backend.py`](kernelgym/backend/kernelbench/cuda_agent_backend.py)
  CUDA-Agent backend that compiles and loads multi-file CUDA submissions.
- [`kernelgym/toolkit/kernelbench/pipeline.py`](kernelgym/toolkit/kernelbench/pipeline.py)
  KernelBench evaluation pipeline and CUDA reward execution path.
- [`drkernel/kernel/rewards/reward_client.py`](drkernel/kernel/rewards/reward_client.py)
  Reward aggregation, coverage, RPC handling, and failure-type penalty routing.
- [`drkernel/kernel/rewards/kernel_reward.py`](drkernel/kernel/rewards/kernel_reward.py)
  Reward extraction for Triton-style and CUDA-Agent triple-section answers.

## Prompt And Data

- [`drkernel/kernel/config/prompt_config/multi_turn_cuda_kernel.yaml`](drkernel/kernel/config/prompt_config/multi_turn_cuda_kernel.yaml)
  CUDA first-turn and feedback prompt templates.
- [`drkernel/kernel/workers/rollout/prompt_templates.py`](drkernel/kernel/workers/rollout/prompt_templates.py)
  Shared helper that applies per-turn templates and `{problem}` placement.
- [`drkernel/kernel/scripts/materialize_backend_neutral_data.py`](drkernel/kernel/scripts/materialize_backend_neutral_data.py)
  Builds backend-neutral parquet/text data and sampled formatted prompt dumps.

## Tests

- [`tests/test_cuda_agent_support.py`](tests/test_cuda_agent_support.py)
  CUDA reward-path regression suite.
- [`tests/test_backend_neutral_data.py`](tests/test_backend_neutral_data.py)
  Backend-neutral data and formatted prompt dump tests.
- [`tests/test_prompt_templates.py`](tests/test_prompt_templates.py)
  Actual CUDA prompt-template rendering tests.
