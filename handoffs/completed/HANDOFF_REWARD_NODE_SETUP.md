# Reward Node Setup — Handoff

## Status: Complete — locked GPU clocks working, timing variance investigation done

## Key Lessons (DO NOT REPEAT THESE MISTAKES)

### 1. Use the correct venv: `.venv-vllm0180`
- The project is `KernelGYM-vllm018`, NOT the primary `KernelGYM` repo
- The venv is at `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/.venv-vllm0180/`
- **NEVER** use `.venv-sglang059` — that's the old sglang setup
- The venv was created by `uv` and its python binary is a symlink to `/root/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12`
- This symlink **only works in containers where uv installed that Python** — it does NOT work in a fresh Docker container

### 2. uv venv symlink problem in new Docker containers
- uv venvs create symlinks to the uv-managed Python binary at `/root/.local/share/uv/python/...`
- In a new Docker container, this path doesn't exist → the venv's `bin/python` is broken
- **Fix**: either install the same uv Python version in the new container, OR use the container's system Python (3.12) with `PYTHONPATH` pointing to the venv's `site-packages`
- System Python approach: `PYTHONPATH=/nfs/FM/.../KernelGYM-vllm018/.venv-vllm0180/lib/python3.12/site-packages:/nfs/FM/.../KernelGYM python3 -m kernelgym.worker.gpu_worker`

### 3. Triton version matters
- drkernel-14b was trained against Triton 3.2.x API
- The Docker image `llmc:v1.1` system Python has Triton 3.2.0 (compatible)
- The `.venv-vllm0180` has Triton 3.6.0 (BREAKING — `tl.tanh` removed, `try` blocks unsupported)
- The `.venv-sglang059` has Triton 3.5.1 (compatible)
- **The Docker system Python (Triton 3.2.0) is the safest choice for reward workers**
- When using system Python + venv site-packages via PYTHONPATH, the system Triton takes priority → correct

### 4. Docker shared memory (`/dev/shm`) fills up
- With `--shm-size=64g`, shared memory fills completely from worker subprocess pools
- This causes `[Errno 28] No space left on device` → workers fall back to single-GPU emergency mode → 8x slowdown
- **Fix**: use `--shm-size=256g` when creating containers

### 5. Redis protected mode blocks external connections
- Redis 6.x starts with `protected-mode yes` by default
- Workers on node 40 connect to Redis on node 39 via IP (not localhost)
- Protected mode rejects these connections with "Connection lost"
- **Fix**: `redis-cli -p 8110 CONFIG SET protected-mode no` after starting Redis

### 6. GPU clock locking must be done on the HOST, not in the container
- `nvidia-smi -lgc 2700,2700` fails inside containers (Insufficient Permissions)
- Must run on the host machine before launching Docker
- Also enable persistence mode: `nvidia-smi -pm 1`
- Docker inherits the locked clock settings from the host

### 7. The `triton_profiler_matches` bug
- In `pipeline.py`, `metadata["triton_profiler_matches"]` raises KeyError on 4090 GPUs where Triton detection doesn't run
- **Fixed**: changed to `metadata.get("triton_profiler_matches", [])` in the primary KernelGYM repo
- Without this fix, all `kernel_runtime` values are -1.0 → all speedups are 0.0

### 8. The Pydantic dict-copy bug for kernel timing
- `KernelExecResult` (Pydantic BaseModel) copies the `metadata` dict on construction
- `_run_performance_step` writes to the original `metadata` dict, not `kernel_exec_result.metadata`
- **Fixed**: pass `kernel_exec_result.metadata` instead of `metadata` to `_run_performance_step`

## Current Setup

### Host machines
- `192.168.16.39` — SSH: `ssh chenshuailin@192.168.16.39` (port 22), sudo password: `csl`
- `192.168.16.40` — SSH: `ssh chenshuailin@192.168.16.40` (port 22), sudo password: `csl`
- Both have 8x RTX 4090 GPUs

### GPU clock locking (run on HOST before Docker)
```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 2700,2700
sudo nvidia-smi -pl 400
```

### NFS mount (run on HOST before Docker)
```bash
sudo mount -t nfs -o vers=3 eds.intellif:/FM /nfs/FM
```

### Docker container launch
```bash
sudo docker run -d \
  --name kernelgym-reward-39 \
  --gpus all \
  --network host \
  --shm-size=256g \
  --privileged \
  -v /nfs:/nfs \
  -w /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM \
  192.168.14.129:80/fm/llmc:v1.1 \
  sleep infinity
```

### Inside container: Redis
```bash
apt-get update -qq && apt-get install -y -qq redis-server
redis-server --port 8110 --save "" --appendonly no --maxmemory 50gb --protected-mode no --daemonize yes
```

### Inside container: Install missing deps
The Docker image (`llmc:v1.1`) has torch 2.6+cu126 and triton 3.2.0 but lacks some KernelGYM deps:
```bash
pip install redis pydantic-settings uvicorn fastapi aiohttp httpx
```
**Do NOT use the uv venvs** (`.venv-vllm0180`, `.venv-sglang059`) — their python symlinks point to uv-managed binaries that don't exist in fresh containers. Use the container's system Python which has the correct Triton 3.2.0.

### Inside container: API server
Use container system Python + KernelGYM on PYTHONPATH:
```bash
export PYTHONPATH=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM
REDIS_HOST=localhost REDIS_PORT=8110 REDIS_KEY_PREFIX=kernelgym_locked_clk API_HOST=0.0.0.0 API_PORT=8111 python3 -u -m kernelgym.server.api.server
```

### Inside container: Worker
Use the helper script which sets all env vars:
```bash
NODE_ID=reward-16-39 bash /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM/scripts/start_reward_worker.sh
```
Or manually:
```bash
export PYTHONPATH=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM
API_HOST=192.168.16.39 API_PORT=8111 REDIS_HOST=192.168.16.39 REDIS_PORT=8110 REDIS_KEY_PREFIX=kernelgym_locked_clk NODE_ID=reward-16-39 GPU_DEVICES="[0,1,2,3,4,5,6,7]" python3 -u -m kernelgym.worker.gpu_worker
```
For the second reward node (40), set `REDIS_HOST=192.168.16.39` (points to node 39's Redis).

The warmup and trim values are controlled by env vars `KERNELGYM_PERF_WARMUP` (default 30) and `KERNELGYM_PERF_TRIM` (default 5), read by `pipeline.py` at task execution time.

### Timing config (env vars read by primary KernelGYM repo pipeline.py)
- `KERNELGYM_PERF_WARMUP`: warmup iterations (default 30, recommended 5 with locked clocks)
- `KERNELGYM_PERF_TRIM`: trials to trim from each end (default 5, i.e. 20% trim with 50 trials)
- `num_perf_trials`: set via Hydra config / training scripts (default 50)

### Eval scripts
- `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/eval/drkernel-14b-maxturns3-temp1.0.sh`
- `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/eval/drkernel-14b-maxturns3-temp0.8.sh`
- `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/eval/drkernel-14b-maxturns3-temp0.6.sh`
