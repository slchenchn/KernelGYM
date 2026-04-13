# 2-Node 16-GPU Rollout — Investigation Handoff

## Status: RUNNING — 2-node training with bf16 gradient reduce

Current live run: `drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.bf16.2node.hybrid.20260329-042556/`

Configuration: `NNODES=2`, `FSDP_SIZE=8` (HYBRID_SHARD), `reduce_dtype=bf16`, `PPO_MICRO_TOKEN=12288`, `PROMPT_OVERSAMPLING_FACTOR=1.7`

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
| 16-GPU rollout | 2x generation parallelism (no cross-node comm) | -6.5m gen |
| 16-GPU logprob | 2x forward-only compute | -1.6m old_log_prob |
| bf16 gradient reduce | Halve cross-node all-reduce data volume | -4.1m update_actor |

## Key Findings

### vLLM engine wake-up is NOT a bottleneck

Engine wake-up (weight reload from CPU to GPU) takes only ~5 seconds. The `servers=0/16` phase lasting ~5 minutes is actual multi-turn generation + reward evaluation time. Keeping engines warm would save negligible time.

### NCCL socket tuning hurt performance

`NCCL_NSOCKS_PERTHREAD=4` and `NCCL_SOCKET_NTHREADS=4` increased `update_actor` from 9.2m to 13.6m. More socket threads cause contention on the ethernet link. Reverted — default NCCL settings with bf16 reduce are optimal.

### `reduce_dtype=bf16` is the single most impactful change for 2-node training

Changing gradient all-reduce from fp32 to bf16 halves the cross-node traffic. `update_actor` dropped from 9.2m to 5.1m (45% reduction). This brought 2-node `update_actor` close to 1-node levels (3.4m), with the remaining ~1.7m overhead from HYBRID_SHARD's cross-node gradient sync at bf16 bandwidth.

## Architecture

With `NNODES=2, FSDP_SIZE=8, HYBRID_SHARD`:

- **Rollout**: 16 independent vLLM engines (TP=1), zero cross-node GPU communication
- **old_log_prob**: FSDP forward-only across 16 GPUs, minimal cross-node overhead
- **update_actor**: FSDP shards within each 8-GPU node (intra-node NVLink), gradient all-reduce across nodes (inter-node ethernet in bf16)
- **Sleep/wake**: engines sleep(level=1) during training, wake-up takes ~5s

## Fixes Landed

### Oversampling (1-node + 2-node)

- `PROMPT_OVERSAMPLING_FACTOR` default `1.0` → `1.7` in `14b_coldstart_trloo_mrs_pr_prs.sh`

### BF16 backward OOM (1-node + 2-node)

- Deferred optimizer-state load to just before `optimizer.step()` in `dp_actor.py` / `fsdp_workers.py`
- See `handoffs/completed/HANDOFF_BF16_OOM.md` for full details

### 2-node networking

- Auto-detect network interface for Gloo/NCCL before `init_process_group` in `fsdp_workers.py`
- `NCCL_IB_DISABLE=1`, `NCCL_NET=Socket`, `NCCL_SOCKET_IFNAME=ens22f0` for ethernet path
- Ray runtime-env passthrough for NCCL/Gloo env vars via `constants_ppo.py` and `base.py`
- `ray.init()` kwargs merged from hydra config in `main_kernel.py`
- Runtime-env env_vars coerced to str to avoid Hydra int-parsing bug

### 2-node paths

- `custom_reward_function.path` made absolute via `$DRKERNEL_ROOT`
- `prompt_config_path` resolved to absolute NFS path in async engine files
- `fsdp_size` field added to `dp_ref.yaml` (was missing, caused hydra error)

### Gradient reduce dtype

- Added `FSDP_REDUCE_DTYPE` variable to `train_rl_common.sh`, passed as `+actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype`
- Set to `bf16` in `run_train_bf16_2node.sh`

## Node Details

| Node | IP | SSH | Role |
|---|---|---|---|
| ai-16-18 | 192.168.16.18 | `ssh -p 10842 root@192.168.16.18` | Ray head |
| ai-16-24 | 192.168.16.24 | `ssh -p 14218 root@192.168.16.24` | Ray worker |

- Network interface: `ens22f0` on both nodes (`192.168.16.0/24`)
- No InfiniBand — ethernet only
- Shared NFS at `/nfs/FM/`
- venv: `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/.venv-vllm0180`
- tmux session: `train-2node` on head node

## Clean Restart Procedure

```bash
# 1. Stop Ray on both nodes
ssh -p 10842 root@192.168.16.18 "source .venv-vllm0180/bin/activate && ray stop --force"
ssh -p 14218 root@192.168.16.24 "source .venv-vllm0180/bin/activate && ray stop --force"

# 2. Start Ray cluster
ssh -p 10842 root@192.168.16.18 "source .venv-vllm0180/bin/activate && \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start --head --port=6379 --num-gpus=8"
ssh -p 14218 root@192.168.16.24 "source .venv-vllm0180/bin/activate && \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start --address=192.168.16.18:6379 --num-gpus=8"

# 3. Launch training on head node (in tmux)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export KERNELGYM_BF16_OOM_DEBUG=1
export PPO_MICRO_TOKEN=12288
bash drkernel/run_train_bf16_2node.sh
```

## Key Runs

| Run | Config | Result |
|---|---|---|
| `...bf16.spare2.refcache.20260328-063213` | 1-node, 1.7x oversampling | 27.2m/step baseline |
| `...bf16.2node.hybrid.20260328-113028` | 2-node, fp32 reduce | 29.0m/step (update_actor too slow) |
| `...bf16.2node.hybrid.20260329-022222` | 2-node, NCCL tuned | 29.8m/step (tuning hurt) |
| `...bf16.2node.hybrid.20260329-042556` | **2-node, bf16 reduce** | **20.1m/step** |

## File References

- `drkernel/run_train_bf16_2node.sh` — 2-node launch script
- `drkernel/kernel/scripts/rl/train_rl_common.sh` — FSDP_SIZE, FSDP_REDUCE_DTYPE passthrough
- `drkernel/verl_patch/workers/code/fsdp_workers.py` — FSDP init, device mesh, interface auto-detect
- `drkernel/verl_patch/workers/code/actor/dp_actor.py` — optimizer-state delay, BF16 debug snapshots
- `drkernel/verl/verl/trainer/constants_ppo.py` — Ray runtime-env passthrough
- `drkernel/verl/verl/single_controller/ray/base.py` — worker env propagation
- `drkernel/kernel/main_kernel.py` — ray.init kwargs merge
