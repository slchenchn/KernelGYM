# 2-Node Grading Deadlock — Investigation Handoff

## Status: Open — workaround in place (1-node), root cause identified but not fixed

## Summary

Multi-node (NNODES=2) grading evals deadlock after ~10 Env Results. Single-node (NNODES=1) works fine. The workaround is to run each temperature on a separate node with independent Ray clusters.

## Symptoms

- All 16 vLLM engines show `[BatchHeartbeat] completed=0/1 pending=1 elapsed=XXXs tokens_in_use=64/64`
- `RolloutProgress` stays at `servers=0/16 prompt_rows=0/800` indefinitely
- ~10 Env Results complete before the deadlock hits
- 1-node (8 GPUs) runs the same eval to completion in ~1 hour

## Root Cause Analysis (verified by code inspection)

The grading script uses `async_vllm` rollout mode, which creates `AsyncActorRolloutRefWorker` via `RayWorkerGroup` with FSDP. When `NNODES=2`:

### What the code does

1. `RayWorkerGroup` creates 16 colocated actor+rollout workers across 2 nodes
2. `wg.init_model()` calls `_build_model_optimizer()` which:
   - Calls `torch.distributed.init_process_group()` with NCCL+Gloo backend (cross-node)
   - Calls `torch.distributed.barrier()` (all 16 workers must sync)
   - Wraps model with `FSDP(sync_module_states=True)` (cross-node collective)
3. After init, `AsyncLLMEngineManager` creates independent vLLM engines per GPU
4. On `sleep()`, `FSDPVLLMShardingManager.__exit__()` is called per-worker (no collective)
5. On `wake_up()`, `FSDPVLLMShardingManager.__enter__()` calls `self.module.state_dict()` which may trigger cross-rank coordination in FSDP2/DTensor

### Key insight: FSDP is NOT used for inference

After `init_model()`, generation uses independent vLLM `AsyncLLM` engines — NOT FSDP's forward pass. The FSDP module is only a **weight source** during `wake_up()`. So the original hypothesis (FSDP all-gather during inference) is wrong.

### Where the deadlock actually occurs

Most likely in the **`wake_up()` → `sharding_manager.__enter__()`** path after the first batch completes:
- `__enter__()` calls `self.module.state_dict()`
- With FSDP2/DTensor, `state_dict()` may require cross-rank coordination
- If one worker's vLLM engine finishes and calls `wake_up()` while others are still generating, the cross-rank state_dict call deadlocks

Alternative: the deadlock could be in `wg.init_model()` itself if Ray worker scheduling is non-deterministic across nodes, causing one worker to miss the `torch.distributed.barrier()`.

### Why 1-node works

With 8 GPUs on one node, all FSDP shards are local. `state_dict()` and barriers complete without cross-node NCCL communication.

### Why training validation works with 2 nodes

Training validation is coordinated by the trainer loop:
- All workers `sleep()` simultaneously after the training step
- All workers `wake_up()` simultaneously before validation
- The barrier/state_dict calls complete because all ranks participate at the same time

In grading, there is no training coordinator — the `sleep()/wake_up()` cycle happens independently per engine, breaking FSDP's assumption of synchronized collective operations.

## What Was Tried

| Attempt | Result |
|---|---|
| FSDP_SIZE=-1 | Deadlock |
| FSDP_SIZE=8 | Deadlock |
| gpu_memory_util=0.75 | Deadlock |
| gpu_memory_util=0.5 | Deadlock |
| Added GLOO/NCCL env vars | Deadlock |
| `standalone_vllm` mode | Crash (vLLM `sleep()` error in `cumem.py`) |

## Potential Fixes (Not Implemented)

1. **Fix `standalone_vllm` mode** — This mode creates independent vLLM engines without FSDP. It crashed on `sleep()` in vLLM's memory allocator (`cumem.py`). Fixing the sleep/wake cycle in this mode would be the cleanest solution.

2. **Add FSDP-free inference mode to grading** — Modify `main_grading.py` to load model weights directly into each vLLM engine without FSDP wrapping when NNODES>1 and no training is needed.

3. **Synchronized batch dispatch** — Ensure all 16 engines receive and process their batches simultaneously, preventing any engine from going idle while others are still working. This matches how training validation works.

## Current Workaround

Run each temperature eval on a separate node with independent 1-node Ray clusters:

```bash
# Node 18: temp0.8
ssh -p 20629 root@192.168.16.18
RAY_ADDRESS=192.168.16.18:6379 bash kernel/scripts/eval/drkernel-14b-maxturns3-temp0.8.sh

# Node 24: temp0.6 (simultaneously)
ssh -p 14218 root@192.168.16.24
RAY_ADDRESS=192.168.16.24:6380 bash kernel/scripts/eval/drkernel-14b-maxturns3-temp0.6.sh
```

Each node runs 8 GPUs independently. Both share the same reward server (192.168.16.39:8111).

## Files Changed

- `drkernel/kernel/scripts/eval/drkernel-14b-maxturns3-temp0.8.sh` — Added 2-node network env vars, absolute CUSTOM_REWARD_PATH. Currently NNODES=1.
- `drkernel/kernel/scripts/eval/drkernel-14b-maxturns3-temp0.6.sh` — Same changes.

## Related

- `drkernel/kernel/main_grading.py:1132-1164` — The `standalone_vllm` vs `async_vllm` branch
- `drkernel/run_train_bf16_2node.sh` — Working 2-node training config for reference
- `handoffs/completed/HANDOFF_REF_TIMING_VARIANCE.md` — Timing variance investigation
