# Uncommitted Changeset Review — Handoff

## Scope

This note reviews the current accumulated uncommitted changes in the top-level `KernelGYM-vllm018` worktree.

Assumption for this handoff: **ignore `drkernel/verl` for now**. The conclusions below only cover the top-level tracked and untracked changes outside that nested repo.

## Executive Summary

The current uncommitted changes should not be committed as a single batch.

The changes naturally split into several independent areas:

1. reward timing configuration propagation
2. reference runtime cache wiring
3. rollout / actor OOM mitigation and async sleep control
4. worker recycling / subprocess pool lifecycle
5. log routing and observability
6. docs and operator tooling
7. online quantization feature work

Before committing, there are still a few high-risk items that should be corrected.

## Findings

### 1. High risk: `subprocess_pool.py` can kill the wrong process after PID reuse

In [`kernelgym/worker/subprocess_pool.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/worker/subprocess_pool.py), the recycle path unconditionally falls through to `os.kill(old_pid, SIGKILL)` after earlier shutdown attempts.

Relevant lines during review:
- [`subprocess_pool.py#L729`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/worker/subprocess_pool.py#L729)
- [`subprocess_pool.py#L733`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/worker/subprocess_pool.py#L733)

If the original child has already exited and the PID has been reused, this can kill an unrelated process. This should be fixed before committing the worker-recycling changes.

### 2. Medium risk: generated-code event logging now persists token/logprob blobs for every turn

[`drkernel/kernel/event_logging.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/event_logging.py) now writes `.npz` blobs containing prompt token ids, response token ids, and logprobs for each generated-code event.

Blob persistence logic:
- [`event_logging.py#L104`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/event_logging.py#L104)
- [`event_logging.py#L129`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/event_logging.py#L129)

Current call sites are unconditional:
- [`vllm_async_engine.py#L2122`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/rollout/vllm_rollout/vllm_async_engine.py#L2122)
- [`vllm_async_engine_multi_iter.py#L1601`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/rollout/vllm_rollout/vllm_async_engine_multi_iter.py#L1601)
- [`openai_async_engine_multi_iter.py#L2714`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/rollout/vllm_rollout/openai_async_engine_multi_iter.py#L2714)

This is likely too heavy for a default path in long-running training. It should be behind a debug flag and likely sampled.

### 3. Low risk but noisy: unconditional reward debug print

[`drkernel/kernel/rewards/kernel_reward.py#L128`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/rewards/kernel_reward.py#L128) to [`kernel_reward.py#L129`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/rewards/kernel_reward.py#L129) prints `[REWARD_DEBUG] ...` on every batch. This should be gated behind an explicit debug switch before committing.

## Recommended Commit Split

### Commit 1: reward timing config propagation

Suggested scope:
- [`drkernel/kernel/config/kernel_grading.yaml`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/config/kernel_grading.yaml)
- [`drkernel/kernel/config/kernel_trainer.yaml`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/config/kernel_trainer.yaml)
- [`drkernel/kernel/scripts/eval/grading_common.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/eval/grading_common.sh)
- [`drkernel/kernel/rewards/reward_client.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/rewards/reward_client.py)
- [`drkernel/kernel/rewards/kernel_reward.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/rewards/kernel_reward.py)
- [`kernelgym/schema/task.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/schema/task.py)
- [`kernelgym/schema/simple_task.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/schema/simple_task.py)
- [`kernelgym/toolkit/kernelbench/pipeline.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/pipeline.py)
- [`kernelgym/toolkit/kernelbench/timing.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/timing.py)
- [`kernelgym/toolkit/kernelbench/toolkit.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernelbench/toolkit.py)
- [`kernelgym/toolkit/kernel_simple/toolkit.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/toolkit/kernel_simple/toolkit.py)

### Commit 2: reference runtime cache wiring

Suggested scope:
- [`kernelgym/workflow/reference_cache.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/workflow/reference_cache.py)
- [`kernelgym/workflow/kernelbench_helpers.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/workflow/kernelbench_helpers.py)
- [`kernelgym/workflow/kernelbench.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/workflow/kernelbench.py)
- [`kernelgym/server/api/server.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/server/api/server.py)
- [`kernelgym/server/api/models.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/server/api/models.py)
- [`kernelgym/config/settings.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/config/settings.py)
- [`drkernel/kernel/scripts/rl/start_reward.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_reward.sh)
- [`drkernel/kernel/scripts/rl/train_rl_common.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/train_rl_common.sh)

### Commit 3: rollout / actor OOM mitigation and async sleep control

Suggested scope:
- [`drkernel/kernel/kernel_trainer.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/kernel_trainer.py)
- [`drkernel/kernel/main_kernel.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/main_kernel.py)
- [`drkernel/kernel/workers/rollout/async_server.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/rollout/async_server.py)
- [`drkernel/kernel/workers/rollout/vllm_rollout/vllm_async_engine.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/rollout/vllm_rollout/vllm_async_engine.py)
- [`drkernel/kernel/workers/rollout/vllm_rollout/vllm_async_engine_multi_iter.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/rollout/vllm_rollout/vllm_async_engine_multi_iter.py)
- [`drkernel/kernel/workers/rollout/vllm_rollout/openai_async_engine_multi_iter.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/rollout/vllm_rollout/openai_async_engine_multi_iter.py)
- [`drkernel/verl_patch/workers/code/fsdp_workers.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/verl_patch/workers/code/fsdp_workers.py)
- [`drkernel/verl_patch/workers/code/actor/dp_actor.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/verl_patch/workers/code/actor/dp_actor.py)
- [`drkernel/verl_patch/workers/code/rollout/vllm_rollout/vllm_rollout_spmd.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/verl_patch/workers/code/rollout/vllm_rollout/vllm_rollout_spmd.py)
- [`drkernel/verl_patch/trainer/code/config/ref/dp_ref.yaml`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/verl_patch/trainer/code/config/ref/dp_ref.yaml)

### Commit 4: worker recycling / subprocess pool lifecycle

Suggested scope:
- [`kernelgym/worker/subprocess_pool.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/worker/subprocess_pool.py)
- [`kernelgym/worker/gpu_worker.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/kernelgym/worker/gpu_worker.py)

Do not submit this commit before fixing the PID-kill risk above.

### Commit 5: log routing and observability

Suggested scope:
- [`drkernel/kernel/scripts/rl/log_router.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/log_router.py)
- [`drkernel/kernel/event_logging.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/event_logging.py)
- [`drkernel/kernel/workers/reward_manager/kernel_async.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/reward_manager/kernel_async.py)
- [`drkernel/kernel/main_grading.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/main_grading.py)

`event_logging.py` can be split into its own commit if easier to review.

### Commit 6: docs and operator tooling

Suggested scope:
- [`INDEX.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/INDEX.md)
- [`SPEC.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/SPEC.md)
- repo-local skills under [`skills/`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/skills)
- plotting / monitoring / eval helper scripts under [`drkernel/kernel/scripts/rl/`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl)
- relevant handoff and report markdowns

### Commit 7: online quantization feature work

Suggested scope:
- [`blockwise_int8_kernels.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/rollout/vllm_rollout/blockwise_int8_kernels.py)
- [`blockwise_int8_quant.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/rollout/vllm_rollout/blockwise_int8_quant.py)
- [`online_quant_utils.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/workers/rollout/vllm_rollout/online_quant_utils.py)
- related launcher and helper scripts

## Suggested Order

1. Fix `subprocess_pool.py` high-risk kill path.
2. Gate or sample the new generated-code blob logging.
3. Gate the unconditional reward debug print.
4. Then split commits in the order above.

## Validation Already Performed

Static validation during the review:
- modified top-level Python files passed `python -m py_compile`
- modified shell scripts passed `bash -n`

This review did **not** include end-to-end runtime validation for the entire changeset.
