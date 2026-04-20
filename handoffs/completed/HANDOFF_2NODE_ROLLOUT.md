# 2-Node 16-GPU Rollout — Completed Investigation Handoff

## Status: COMPLETED — Historical 2026-03 A800 study

This document records the completed `14B` two-node `A800` rollout investigation that established the benefit of `bf16` gradient reduce and higher prompt oversampling for that topology.

It is **not** the current operational runbook for training in this repository. For current launches, use:

- [`drkernel/kernel/scripts/rl/start_training.sh`](../../drkernel/kernel/scripts/rl/start_training.sh)
- [`SPEC.md`](../../SPEC.md)

The historical reference run for this study was:

- `drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.bf16.2node.hybrid.20260329-042556/`

Historical configuration:

- `NNODES=2`
- `FSDP_SIZE=8` with `HYBRID_SHARD`
- `reduce_dtype=bf16`
- `PPO_MICRO_TOKEN=12288`
- `PROMPT_OVERSAMPLING_FACTOR=1.7`

## Results Summary

### Step time comparison

| Setup | gen | old_log_prob | update_actor | Total step | vs baseline |
|---|---|---|---|---|---|
| 1-node, no oversampling | 19.3m | 4.1m | 3.4m | 34.5m | 1.0x |
| 1-node, 1.7x oversampling | 18.9m | 4.1m | 4.2m | 27.2m | 1.27x |
| 2-node, fp32 reduce | 14.1m | 2.6m | 9.2m | 29.0m | 1.19x |
| **2-node, bf16 reduce** | **12.4m** | **2.5m** | **5.1m** | **20.1m** | **1.72x** |

### What each optimization contributed

| Change | Mechanism | Impact |
|---|---|---|
| 1.7x oversampling | Eliminate retry rollout | -7.3m/step |
| 16-GPU rollout | 2x generation parallelism with no cross-node rollout comm | -6.5m gen |
| 16-GPU logprob | 2x forward-only compute | -1.6m old_log_prob |
| bf16 gradient reduce | Halve cross-node all-reduce volume | -4.1m update_actor |

## Durable Findings

### vLLM engine wake-up was not the bottleneck in this study

Engine wake-up cost was only about `5s`. The long `servers=0/16` phase represented actual multi-turn generation plus reward evaluation, not idle wake-up overhead.

### NCCL socket-thread tuning hurt this ethernet setup

`NCCL_NSOCKS_PERTHREAD=4` and `NCCL_SOCKET_NTHREADS=4` made `update_actor` slower, increasing it from `9.2m` to `13.6m`. On this topology, extra socket-thread tuning increased contention rather than throughput.

### `reduce_dtype=bf16` was the biggest two-node training improvement

Changing gradient all-reduce from `fp32` to `bf16` reduced `update_actor` from `9.2m` to `5.1m`, a `45%` reduction. That was the most important improvement specific to the two-node setup studied here.

## Architecture Used In This Study

With `NNODES=2`, `FSDP_SIZE=8`, and `HYBRID_SHARD`:

- **Rollout** used 16 independent vLLM engines with `TP=1`, so rollout itself had no cross-node GPU communication.
- **old_log_prob** used FSDP forward-only work across 16 GPUs and showed relatively low cross-node overhead.
- **update_actor** kept FSDP shards intra-node but still required cross-node gradient reduce.
- **Sleep/wake** used engine sleep during training and wake-up remained a small cost relative to generation.

## Historical Changes That Landed During The Investigation

### Oversampling

- The `14B` coldstart launcher used in this study moved its prompt oversampling default from `1.0` to `1.7`.
- This finding was topology- and workload-specific. Current launchers may use different defaults.

### BF16 backward OOM mitigation

- Optimizer-state load was deferred to just before `optimizer.step()` in the actor/FSDP worker path.
- See [`HANDOFF_BF16_OOM.md`](HANDOFF_BF16_OOM.md) for the completed OOM write-up.

### Two-node networking and runtime-env plumbing

- Gloo/NCCL interface selection was wired into worker init and Ray runtime env propagation.
- `ray.init()` kwargs were merged from hydra config in `main_kernel.py`.
- Runtime-env env vars were coerced to strings to avoid Hydra parsing issues.

### Path handling

- Reward and prompt-config paths were made more robust by resolving them from repo-root context instead of relying on fragile relative paths.

### Gradient reduce dtype passthrough

- `FSDP_REDUCE_DTYPE` was added to the shared launch assembly so the actor FSDP reduce dtype could be controlled from launcher scripts.

## Historical Environment Notes

These notes are preserved only to explain the original measurements. They are **not** current run instructions:

- Two-node `A800` setup
- Ethernet path on `ens22f0`
- Socket NCCL path rather than IB
- Shared NFS-backed environment
- Legacy launcher: `drkernel/run_train_bf16_2node.sh`

## Why This Handoff Is Completed

This investigation answered its original question:

- two-node rollout itself improved generation and logprob throughput
- the real remaining two-node penalty was update-time cross-node reduction
- `bf16` reduce materially fixed that bottleneck for the studied setup

The document remains useful as a historical benchmark note, but its previous `RUNNING` status, live-run labels, and direct restart instructions are obsolete and should not be followed for current training.

## Key Historical Runs

| Run | Config | Result |
|---|---|---|
| `...bf16.spare2.refcache.20260328-063213` | 1-node, 1.7x oversampling | 27.2m/step baseline |
| `...bf16.2node.hybrid.20260328-113028` | 2-node, fp32 reduce | 29.0m/step |
| `...bf16.2node.hybrid.20260329-022222` | 2-node, NCCL tuned | 29.8m/step |
| `...bf16.2node.hybrid.20260329-042556` | **2-node, bf16 reduce** | **20.1m/step** |

## File References

- [`drkernel/run_train_bf16_2node.sh`](../../drkernel/run_train_bf16_2node.sh) — historical launcher used in this study
- [`drkernel/kernel/scripts/rl/train_rl_common.sh`](../../drkernel/kernel/scripts/rl/train_rl_common.sh) — shared RL launch assembly, including FSDP reduce-dtype passthrough
- [`drkernel/kernel/main_kernel.py`](../../drkernel/kernel/main_kernel.py) — training entrypoint with `ray.init()` kwargs merge
- [`drkernel/verl/verl/trainer/constants_ppo.py`](../../drkernel/verl/verl/trainer/constants_ppo.py) — Ray runtime-env passthrough
- [`drkernel/verl/verl/single_controller/ray/base.py`](../../drkernel/verl/verl/single_controller/ray/base.py) — worker env propagation
