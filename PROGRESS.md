# Progress

#### 8B H20 run with prompt oversampling `2.0` failed again when the Windows-hosted reward relay dropped, then resumed from `global_step_40` after lowering the vLLM rollout memory budget to `0.6`; a later actor-update OOM was traced to external worker-node GPU contention, and after a user-requested stop on suspected reward instability the run has now been resumed again from `global_step_80` — ACTIVE

##### Problem & Impact

- The earlier `12xH20` 8B run at [`trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-033953`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-033953) was launched with `prompt_oversampling_factor=1.0` and repeatedly missed the required `256` selected samples at `step 2`, which forced buffered retry rollouts and repeatedly re-paid the expensive `old_log_prob` phase.
- The user therefore asked to raise prompt oversampling to `2.0` and restart training so the run could clear the early low-batch bottleneck.
- The new run at [`trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700) did clear the original early retry bottleneck and progressed through visible `step 48`, but it later failed again at `2026-04-15 06:18:46 UTC` during in-flight `step 49` with `RuntimeError: No valid samples were selected after filtering. Increase rollout number to ensure that there are valid examples for training.`
- The second failure window again aligned with the Windows-hosted reward relay disappearing. That broke the local head-node reward path and left the run unable to continue until the relay was restored and training was resumed from checkpoint.
- The first in-place resume after the relay was restored did reach `global_step_40`, but it then failed at `2026-04-15 07:09:46 UTC` during async vLLM wake-up with `RuntimeError: CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:139` while allocating `kv_cache`.
- The second in-place resume with `ROLLOUT_GPU_MEMORY_UTIL=0.6` did clear the earlier vLLM wake-up OOM and progressed through visible `step 45`, but it later failed again at `2026-04-15 09:22:58 UTC` during actor update on worker node `10.0.18.5` with `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.49 GiB. GPU 0 ... had only 1.19 GiB free.`
- By `2026-04-16 00:56 UTC`, the same resumed branch had progressed to visible `step 82` with in-flight `step 83`, but the latest durable checkpoint tracker still pointed at `global_step_80` because `trainer.save_freq=10`. The user then asked to stop training first due suspected reward-env instability and explicitly resume from `step80`.

##### Resolution

- Stopped the old `prompt_oversampling_factor=1.0` run with [`.agents/skills/stop_training/scripts/stop_ray_training.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/.agents/skills/stop_training/scripts/stop_ray_training.sh), changed [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp6_pytorch_eager.12xH20.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/8b_trloo_hfsdp6_pytorch_eager.12xH20.sh) so its default `PROMPT_OVERSAMPLING_FACTOR` is `2.0`, revalidated Python / model / data / reward-endpoint reachability on both training nodes, and relaunched through the canonical startup script.
- Verified from the full `main.log` rather than activity alone that the restarted run completed visible train steps through `step 38`, so `oversampling=2.0` did solve the original early retry loop rather than failing at the same `step 2` bottleneck.
- After the later crash, cross-checked `main.log`, `trainer.log`, and [`/tmp/train-8b-12xh20-ib.log`](/tmp/train-8b-12xh20-ib.log) and confirmed the terminal failure, then traced the concrete break to the local reward relay chain on the head node:
  - `CRITICAL: No samples selected! Batch is empty. Check filtering criteria.`
  - `RuntimeError: No valid samples were selected after filtering. Increase rollout number to ensure that there are valid examples for training.`
  - repeated `Server disconnected without sending a response.` in `reward.log` / `vllm.log`
  - `reward-relay-18112` still listening on `18112`, but repeated `connect(5, AF=2 127.0.0.1:18111, 16): Connection refused` because the Windows-hosted reverse relay behind `127.0.0.1:18111` was gone
- After the relay dropped again on `2026-04-15`, rechecked the concrete break from the training head before relaunch:
  - `curl http://127.0.0.1:18111/health` failed
  - `curl http://10.0.18.3:18112/health` returned `Empty reply from server`
  - `ss -ltnp` showed only `0.0.0.0:18112` from `socat`, not `127.0.0.1:18111`
  - the `reward-relay-18112` tmux pane again logged repeated `connect(5, AF=2 127.0.0.1:18111, 16): Connection refused`
- After the user restored the Windows relay again on `2026-04-15`, revalidated the full reward path from the training head before relaunch:
  - `http://127.0.0.1:18111/health` and `http://10.0.18.3:18112/health` both return `healthy`
  - `ss -ltnp` now shows both `127.0.0.1:18111` and `0.0.0.0:18112`
- Relaunched the same run directory through the canonical startup script with `RUN_LOG_DIR` pointed back at [`trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700), so the trainer reuses the existing checkpoint root instead of creating a new run.
- Revalidated prelaunch prerequisites on both training nodes before the resume relaunch:
  - `which python`: `/usr/bin/python`
  - `sys.executable`: `/usr/bin/python`
  - `VIRTUAL_ENV`: unset
  - model and train/val parquet paths reachable on the nodes that use them
  - reward endpoint healthy again from the training head
- Confirmed the resume source on disk:
  - [`checkpoints/latest_checkpointed_iteration.txt`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700/checkpoints/latest_checkpointed_iteration.txt) exists and contains `40`
  - `trainer.resume_mode=auto`
  - `trainer.default_local_dir` points at the same checkpoint root
  - [`drkernel/verl/verl/trainer/ppo/ray_trainer.py`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/verl/verl/trainer/ppo/ray_trainer.py:819) resolves auto-resume from that tracker file and then sets `self.global_steps` from the chosen `global_step_*` folder
- After the first resume hit the new wake-up OOM, relaunched the same run directory again through the canonical startup script with one runtime mitigation only:
  - `ROLLOUT_GPU_MEMORY_UTIL=0.6`
- Verified from the new trainer config dump that the mitigation landed:
  - `actor_rollout_ref.rollout.gpu_memory_utilization: 0.6`
- Confirmed the second resume again resolved the same checkpoint root and crossed the previous wake-up boundary:
  - `Found checkpoint: .../checkpoints/global_step_40`
  - `Load from checkpoint folder: .../checkpoints/global_step_40`
  - `Setting global step to 40`
  - `Resuming from .../checkpoints/global_step_40`
  - `Training Progress: 40/2249000`
  - `[RolloutProgress] step=41 ... phase=start`
- After the later stop, rechecked the full `main.log`, `trainer.log`, tee log, Ray state, and both nodes' live GPU process tables instead of assuming the new OOM came from the same rollout-side wake-up path:
  - `main.log` / tee log now end at `2026-04-15 09:22:58 UTC` with `ray.exceptions.RayTaskError(OutOfMemoryError)` from `WorkerDict.actor_rollout_update_actor()`
  - the failing stack is actor backward on worker `10.0.18.5`, not vLLM wake-up on the head node
  - the last successful visible train step is `45`, and the failed in-flight step is `46`
  - after the crash, `tmux` is gone and `ray status --address=10.0.18.3:6379` reports `0.0/12.0 GPU` in use
  - the head node `10.0.18.3` GPUs are now fully idle
  - the worker node `10.0.18.5` still has unrelated non-Ray Python jobs holding GPU memory, including one on training-slice `GPU 0`
  - the concrete external worker-node GPU users observed right after the crash were:
    - `GPU 0`: `/opt/conda/bin/python3 /data3/yhy/6/profile_ncu_wan_t2v.py ...` using about `23 GiB`
    - `GPU 6`: `python3 benchmark_14b_720p_fp8_sagesla.py --repeat 5 --warmup 1 --tag v_prep_overlap` using about `71.8 GiB`
    - `GPU 7`: `/opt/conda/bin/python /data3/yhy/project/IntellifTurboDiffusion/turbodiffusion/inference/wan2.1_t2v_infer_ncu.py ...` using about `26.4 GiB`
  - because this training topology intentionally uses GPUs `0-5` on each node, the external process on worker `GPU 0` directly overlaps the training slice
- After the user cleared the overlapping worker-node jobs, revalidated prerequisites before relaunch:
  - local and worker-node training-slice GPUs `0-5` were free again
  - reward health still passed on `http://127.0.0.1:18111/health` and `http://10.0.18.3:18112/health`
  - `which python` and `sys.executable` were still `/usr/bin/python`
  - `VIRTUAL_ENV` remained unset
  - the 8B model path and both train/val parquet paths were still reachable on the head and worker nodes
  - `checkpoints/latest_checkpointed_iteration.txt` still contained `40`
- Relaunched the same run directory again through the canonical startup script with the same runtime mitigation and topology:
  - `--profile h20`
  - `--skip-reward`
  - `--train-script drkernel/kernel/scripts/rl/8b_trloo_hfsdp6_pytorch_eager.12xH20.sh`
  - `--tmux-session train-8b-12xh20-ib`
  - `--local-log /tmp/train-8b-12xh20-ib.log`
  - `--env RUN_LOG_DIR=.../trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700`
  - `--env TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
  - `--env NCCL_NET=IB`
  - `--env NCCL_DEBUG=INFO`
  - `--env ROLLOUT_GPU_MEMORY_UTIL=0.6`
- Verified from the fresh `TaskRunner pid=3743147` config dump that the intended overrides landed again:
  - `nnodes: 2`
  - `n_gpus_per_node: 6`
  - `server_url: http://10.0.18.3:18112`
  - `NCCL_NET: IB`
  - `NCCL_DEBUG: INFO`
  - `actor_rollout_ref.rollout.gpu_memory_utilization: 0.6`
- Verified from the same resumed branch that checkpoint restore is real, not inferred from tmux activity alone:
  - `Found checkpoint: .../checkpoints/global_step_40`
  - `Load from checkpoint folder: .../checkpoints/global_step_40`
  - `Setting global step to 40`
  - `Resuming from .../checkpoints/global_step_40`
  - actor ranks then began loading `model_world_size_12_rank_*`, `optim_world_size_12_rank_*`, and `extra_state_world_size_12_rank_*` from `global_step_40`
- When the user later asked to stop first and resume from `step80`, stopped the live cluster through the canonical stop script instead of killing the local tmux pane:
  - pre-stop live status was visible `step 82` with in-flight `step 83`
  - `tmux has-session -t train-8b-12xh20-ib` then returned missing
  - `ray status --address=10.0.18.3:6379` stopped answering because GCS was down
  - both `main.log` and `trainer.log` stopped advancing after the stop window
- Revalidated resume prerequisites from the head node before relaunch:
  - `which python`: `/usr/bin/python`
  - `sys.executable`: `/usr/bin/python`
  - `VIRTUAL_ENV`: unset
  - reward endpoints `http://127.0.0.1:18111/health` and `http://10.0.18.3:18112/health` both returned healthy
  - model path, train parquet, val parquet, and `checkpoints/global_step_80` were all reachable
  - [`checkpoints/latest_checkpointed_iteration.txt`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700/checkpoints/latest_checkpointed_iteration.txt) still contained `80`
- Relaunched the same run directory again through the canonical startup script with the same sliced-node topology and runtime mitigation:
  - `--profile h20`
  - `--skip-reward`
  - `--train-script drkernel/kernel/scripts/rl/8b_trloo_hfsdp6_pytorch_eager.12xH20.sh`
  - `--tmux-session train-8b-12xh20-ib`
  - `--local-log /tmp/train-8b-12xh20-ib.log`
  - `--env RUN_LOG_DIR=.../trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700`
  - `--env TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
  - `--env NCCL_NET=IB`
  - `--env NCCL_DEBUG=INFO`
  - `--env ROLLOUT_GPU_MEMORY_UTIL=0.6`
- Verified from the fresh `TaskRunner pid=746600` logs that the requested checkpoint was honored:
  - `Found checkpoint: .../checkpoints/global_step_80`
  - `Load from checkpoint folder: .../checkpoints/global_step_80`
  - `Setting global step to 80`
  - `Resuming from .../checkpoints/global_step_80`
  - `Training Progress: 80/2249000`
  - `[RolloutProgress] step=81 ... phase=start`
- After the user asked why the most recent steps had become much slower, regenerated the canonical training dynamics plots and re-checked the full `main.log`, `trainer.log`, `reward.log`, and `vllm.log` instead of inferring from tmux or GPU activity:
  - ran [`drkernel/kernel/scripts/rl/plot_run_dynamics.py`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/plot_run_dynamics.py) on the active run directory and refreshed [`training_dynamics_plots_by_group`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700/training_dynamics_plots_by_group)
  - the completed recent steps split into two timing bands:
    - `step 102-103`: about `20-22 min/step`
    - `step 104-108`: about `56-64 min/step`
  - the slowdown is dominated by rollout generation rather than actor update or old-log-prob:
    - `step 108 timing_s/gen`: `3131.7s` (`52.2 min`)
    - `step 108 timing_s/old_log_prob`: `334.7s` (`5.6 min`)
    - `step 108 timing_s/update_actor`: `87.6s` (`1.5 min`)
  - the rollout progress traces for `step 104-109` show the same long-tail pattern:
    - long initial `servers=0/12` heartbeat windows
    - later only one or a few slow servers remain pending while the rest have finished
    - `step 108` did not reach `12/12` servers until `3117.5s`
    - current `step 109` is repeating the same pattern
  - the reward / vLLM side now shows heavy long-tail evidence during these slow steps, but the `30s` timeout needs a more precise interpretation:
    - `reward.log` contains large volumes of `timeout after 30s` and `Server disconnected without sending a response.`
    - the `timeout after 30s` message comes from the reward worker subprocess pool after a real per-task kernel evaluation timeout, not from queue wait timeout
    - launcher config still sets `REWARD_TASK_TIMEOUT=30`, while the reward client separately keeps a much larger client/network timeout window
    - `vllm.log` contains repeated `[BatchHeartbeat] completed=0/1 pending=1` lines persisting for `~840s`, `~960s`, `~1200s`, and in the previous step up to `~2880s`
    - repo-level structured heartbeat logs under [`drkernel/logs/structured`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/structured) show the exact pending task ids surviving for `1200-3060s` per ref rather than only a vague batch-level stall
    - those same heartbeat records show that the reward token bucket stays occupied by long-lived pending refs, which means the long tail is not "old_log_prob" and not actor update; it is reward request lifetime
    - the reward client uses synchronous `POST /evaluate` and holds its token until that HTTP call returns; `task_timeout=30` only limits the kernel worker execution, while the client still uses `timeout=1800`, `task_timeout_in_client=2400`, `acquire_timeout=2400`, and `max_retries=3`
    - the reward server's `/evaluate` path is synchronous and idempotent on `task_id`: retries with the same `task_id` can return a cached result immediately if the first request already completed server-side
    - concrete evidence of that mismatch exists in the live logs: some final reward payloads contain `completed_at` timestamps more than `20 min` earlier than the client-side `Task failed result` log line, which means the server had already finished while the client-side synchronous request was still unresolved
    - the repeated `Task failed after 2 retries. Last error: None` string is also now explained by code: it is the `kernelgym` subprocess pool path where `_get_idle_worker(timeout=30)` returns no idle worker across all retries, not a real kernel-level exception object named `None`
  - this changes the performance diagnosis boundary:
    - the recent slowdown is not primarily `old_log_prob`
    - the recent slowdown is not primarily actor backward/update
    - the recent slowdown is rollout-side long-tail waiting on reward-backed batches / tool tasks
    - more specifically, the active root cause is no longer "queue timeout"
    - the active root cause is a contract mismatch between the synchronous reward `/evaluate` API and the client-side token / retry model: a small number of long-lived or transport-stuck synchronous `/evaluate` calls keep reward refs pending for `20-50+ min`, pin rate-limit tokens, starve later reward submissions, and then eventually collapse into delayed cached results or delayed `failed after 2 retries` responses from an already-saturated worker pool

##### Result & Current State

- The active run directory remains [`trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700), and the run is active again after the latest resume relaunch.
- The reward-path diagnosis boundary is now stable:
  - the later `step 49` crash aligned with the Windows-hosted relay disappearing again
  - the relay has since been restored again
  - the training-head reward entrypoints `127.0.0.1:18111` and `10.0.18.3:18112` are healthy again
  - the stale `workers/status` heartbeats should still be treated as an observability inconsistency rather than a current blocker
- The latest relaunches did hit the intended checkpoint rather than starting from scratch:
  - the first post-relay resume at `2026-04-15 07:09 UTC` loaded `global_step_40` but then failed in async vLLM wake-up with the new `cumem_allocator` OOM
  - the second post-relay resume at `2026-04-15 07:19 UTC` also loaded `global_step_40`, but this time with `ROLLOUT_GPU_MEMORY_UTIL=0.6`
  - the latest post-contention resume at `2026-04-15 09:41 UTC` again loaded `global_step_40` with the same `ROLLOUT_GPU_MEMORY_UTIL=0.6` mitigation
  - the latest controlled stop-and-resume at `2026-04-16 01:01 UTC` loaded `global_step_80` as explicitly requested by the user
- The latest resumed branch made real new progress before the next stop:
  - it advanced beyond the previous resume point and completed visible steps through `45`
  - the failed in-flight step was `46`
- The current blocker is no longer reward reachability, and the earlier wake-up OOM is no longer the immediate failure mode:
  - the relay remains healthy on `127.0.0.1:18111` and `10.0.18.3:18112`
  - the previous OOM boundary was `TaskRunner.fit() -> async_rollout_manager.wake_up() -> MultiTurnAsyncvLLMEngine.wake_up() -> vllm/device_allocator/cumem.py`
  - lowering `ROLLOUT_GPU_MEMORY_UTIL` to `0.6` let the resumed run pass that boundary and continue training through visible `step 45`
  - the new stop instead came from actor-backward OOM on worker `10.0.18.5`
- The worker-node contention is no longer the current blocker:
  - the overlapping external job on worker `GPU 0` was cleared before relaunch
  - head-node tmux session `train-8b-12xh20-ib` exists again
  - `ray status --address=10.0.18.3:6379` again shows `12.0/12.0 GPU` in use
  - the newest resumed branch has already progressed well beyond the checkpoint it reloaded:
    - latest durable checkpoint tracker: `global_step_100`
    - latest successful visible train step: `108`
    - current in-flight step: `109`
  - retry / oversampling state:
    - no current low-batch retry or oversampling loop is visible
    - the active step is stuck inside rollout heartbeat / partial server completion rather than retry
  - the current performance blocker is rollout-side long-tail latency:
    - recent completed steps `104-108` all spent about `50-58 min` in `timing_s/gen`
    - `old_log_prob` remained about `4.4-5.6 min`
    - `update_actor` remained about `0.7-1.5 min`
    - current `step 109` is still in rollout with only partial server completion
    - the long-tail reward pattern is now explained well enough to act on:
      - reducing only the server-side `task_timeout=30` does not solve the stall
      - the critical path is the synchronous HTTP lifetime and retry behavior around `/evaluate`
      - until that contract is changed or the client times out / cancels much earlier, a few bad reward tasks can keep a rollout step alive for tens of minutes even after the underlying server work has already finished or failed
  - WandB state: this run still uses `trainer.logger=['console']`, so there is no live WandB run state to rely on

#### 8B H20 training relaunched as `12xH20` using GPUs `0-5` on both nodes — SUPERSEDED

##### Problem & Impact

- The user asked to stop the current training run and relaunch on both H20 nodes while using only the first `6` GPUs per node, for `12` training GPUs total.
- The repo could not express that topology as-is:
  - [`drkernel/kernel/scripts/rl/start_training.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_training.sh) hardcoded `ray start --num-gpus=8` on both nodes
  - the H20 8B launchers only covered `8xH20` and `16xH20`
  - the first relaunch attempt also exposed a tmux-session naming bug in `start_training.sh`, where `.` in the derived session name was not normalized but tmux silently converted it to `_`, causing duplicate-session failures on retry
- The requested `oversampling=1.0` also needed to be reflected explicitly in the new launcher rather than left implicit.

##### Resolution

- Stopped the active training cluster through [`.agents/skills/stop_training/scripts/stop_ray_training.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/.agents/skills/stop_training/scripts/stop_ray_training.sh) and verified that the old head and worker GPUs were free before relaunch.
- Updated [`drkernel/kernel/scripts/rl/start_training.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_training.sh) so launch-time env overrides can now control:
  - Ray `--num-gpus` per node
  - `CUDA_VISIBLE_DEVICES` pinning for Ray head/worker startup
  - the corresponding launcher-side `CUDA_VISIBLE_DEVICES`
- Fixed `sanitize_name()` in [`drkernel/kernel/scripts/rl/start_training.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_training.sh) so derived tmux session names normalize `.` to `_` and can be killed/reused safely.
- Parameterized [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh) so its topology defaults can be overridden cleanly by a thin top-level wrapper.
- Added the new top-level launcher [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp6_pytorch_eager.12xH20.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/8b_trloo_hfsdp6_pytorch_eager.12xH20.sh), which sets:
  - `NNODES=2`
  - `GPUS_PER_NODE=6`
  - `N_GPUS_PER_NODE=6`
  - `FSDP_SIZE=6`
  - `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
  - `prompt_oversampling_factor=1.0`
  - `sample_oversampling_factor=1.0`
  - cross-node reward URL `http://10.0.18.3:18112`
- Relaunched through the canonical startup script with:
  - `--profile h20`
  - `--skip-reward`
  - `--train-script drkernel/kernel/scripts/rl/8b_trloo_hfsdp6_pytorch_eager.12xH20.sh`
  - `--tmux-session train-8b-12xh20-ib`
  - `--local-log /tmp/train-8b-12xh20-ib.log`
  - `--env TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
  - `--env NCCL_NET=IB`
  - `--env NCCL_DEBUG=INFO`

##### Result & Current State

- The active run directory is [`trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-033953`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-033953).
- The active head-node tmux session is [`train-8b-12xh20-ib`](/tmp/train-8b-12xh20-ib.log), with tee log [`/tmp/train-8b-12xh20-ib.log`](/tmp/train-8b-12xh20-ib.log).
- `ray status --address=10.0.18.3:6379` now reports `2` active nodes and `12` total GPUs, with all `12` GPUs reserved by placement groups during worker initialization.
- `trainer.log` confirms:
  - `nnodes: 2`
  - `n_gpus_per_node: 6`
  - `fsdp_size: 6`
  - `prompt_oversampling_factor: 1.0`
  - `sample_oversampling_factor: 1.0`
  - `server_url: http://10.0.18.3:18112`
  - `NCCL_NET: IB`
  - `test_freq: 0`
  - `val_before_train: False`
- Both nodes currently show `6` `ray::WorkerDict.actor_rollout_init_model` GPU processes on GPUs `0-5`, each at about `2168 MiB`, while GPUs `6-7` are not used by the current Ray training workers.
- The latest logs show NCCL `Initialized NET plugin IB` and `Using network IB` on the active run.
- No new `Traceback`, `RuntimeError`, or `AssertionError` has appeared in the active tee log at the latest check.
- `Training Progress` is not yet visible; the run is still in distributed worker/model bring-up rather than failed.

#### Local training profile selection moved to `.infra_profile.local.sh` and shared harness paths — COMPLETED

##### Problem & Impact

- Switching between A800 and H20 still depended on editing tracked default profile values in launcher or infra scripts, which is fragile across upstream pulls and noisy in local commits.
- The first local-override implementation looked for `.infra_profile.local.sh` through `REWARD_REPO_PATH`, which can point at the old `/nfs/...` tree instead of the current worktree on H20, so the override was not reliably discovered.
- Some adjacent harness paths still bypassed the profile loader entirely, including the checkpoint-eval helper and the training/stop skills, which left the profile-selection behavior inconsistent.
- The first A800-side local-profile follow-up also exposed two tracked harness mismatches:
  - [`drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh) began inheriting the worktree default profile even though its checkpoint source still points at a fixed A800 run.
  - [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.sh) accepted `TRAIN_CLUSTER_PROFILE=a800` with `NNODES=1` or `--single-node` and still resolved to the `16xA800` launcher.

##### Resolution

- Changed [`drkernel/kernel/scripts/rl/infra_common.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/infra_common.sh) to resolve `.infra_profile.local.sh` from the current repo root and keep precedence ordered as:
  - explicit `--profile`
  - `TRAIN_CLUSTER_PROFILE`
  - `.infra_profile.local.sh`
  - loader fallback
- Added tracked example file [`.infra_profile.local.example.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/.infra_profile.local.example.sh) so each worktree can create an untracked local default without modifying tracked scripts.
- Kept the 8B compatibility launcher profile-driven instead of hardware-default-driven, and updated the start/stop skills so they direct operators to use `--profile` or `.infra_profile.local.sh` rather than editing tracked defaults.
- Updated [`drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh) to load the active infra profile and run through the shared train-node helpers instead of hardcoded A800 SSH endpoints.
- Restored an explicit A800 default in [`drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh) before `infra_common.sh` is loaded, and made the checkpoint root overrideable with `CKPT_BASE=...` so the helper keeps its historical default without requiring tracked-file edits for other runs.
- Added explicit guardrails in [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.sh) and [`drkernel/kernel/scripts/rl/start_training.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_training.sh) so unsupported `a800 + single-node` requests fail in validation instead of silently mapping to the `16xA800` launcher.

##### Result & Current State

- A worktree can now pin its default training hardware locally by creating `.infra_profile.local.sh`, while preserving explicit per-command overrides.
- The start, stop, and checkpoint-eval harness paths now share the same profile-selection mechanism instead of each carrying their own hardcoded device default.
- The checkpoint-eval helper once again preserves its historical A800 default behavior on a fresh worktree while still allowing explicit profile or path overrides.
- The generic 8B launcher no longer silently maps unsupported `a800 + single-node` requests onto the `16xA800` two-node launcher, and `start_training.sh --dry-run --profile a800 --single-node --train-script drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.sh` now exits with a clear validation error.
- Future upstream pulls no longer require local tracked-file edits just to switch between A800 and H20 in this repo.

#### 8B H20 training switched from `16xH20` two-node IB to `8xH20` single-node with oversampling `1.0` — SUPERSEDED

##### Problem & Impact

- The user asked to stop the active two-node `16xH20` 8B training run, relaunch on only the current node `gz01-h20-03`, and reduce oversampling to `1.0`.
- The repo's canonical startup path could not express that request as-is because [`drkernel/kernel/scripts/rl/start_training.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_training.sh) always started the worker Ray node, which would have silently rebuilt a two-node cluster again.
- The canonical stop path also had a repo-root bug: [`.agents/skills/stop_training/scripts/stop_ray_training.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/.agents/skills/stop_training/scripts/stop_ray_training.sh) walked up only three path levels from `.agents/skills/stop_training/scripts/`, so it resolved `.agents/drkernel/...` and failed before it could stop anything.

##### Resolution

- Fixed [`.agents/skills/stop_training/scripts/stop_ray_training.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/.agents/skills/stop_training/scripts/stop_ray_training.sh) to resolve `REPO_ROOT` correctly from the skill script path, then used it to stop the active H20 Ray cluster on both `gz01-h20-03` and `gz01-h20-05`.
- Added single-node support to [`drkernel/kernel/scripts/rl/start_training.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_training.sh) with `--single-node`, automatic `NNODES=1` handling, and worker-start suppression so the canonical orchestrator can now launch a true head-only run instead of requiring ad hoc tmux/SSH commands.
- Verified the launch-node environment and prerequisites before relaunch:
  - `which python`: `/usr/bin/python`
  - `sys.executable`: `/usr/bin/python`
  - `VIRTUAL_ENV`: unset
  - reward health: `http://127.0.0.1:18111/health`
  - model path: `/nfs/FM/chenshuailin/checkpoints/hkust-nlp/drkernel-8b-coldstart`
  - train/val parquet paths under `drkernel/data/`
- Removed only the stale repo-managed Ray tmux sessions left behind by the stopped `16xH20` run and confirmed its `main.log` / `trainer.log` timestamps stopped moving.
- Relaunched through the canonical startup script with:
  - launcher: [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh)
  - `--single-node`
  - `--skip-reward`
  - `--env PROMPT_OVERSAMPLING_FACTOR=1.0`
  - `--env SAMPLE_OVERSAMPLING_FACTOR=1.0`
  - tmux session `train-8b-8xh20-ovs1`
  - tee log `/tmp/train-8b-8xh20-ovs1.log`

##### Result & Current State

- The active run directory is [`trloo-8b-hfsdp8-pytorch-eager.train.8xH20.reward.16x4090.run.20260414-021605`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp8-pytorch-eager.train.8xH20.reward.16x4090.run.20260414-021605).
- The active head-node tmux session is [`train-8b-8xh20-ovs1`](/tmp/train-8b-8xh20-ovs1.log), with tee log [`/tmp/train-8b-8xh20-ovs1.log`](/tmp/train-8b-8xh20-ovs1.log).
- `ray status --address=10.0.18.3:6379` now reports `1` active node and `8` total GPUs.
- `trainer.log` confirms:
  - `nnodes: 1`
  - `prompt_oversampling_factor: 1.0`
  - `sample_oversampling_factor: 1.0`
  - `server_url: http://127.0.0.1:18111`
  - `test_freq: 0`
  - `val_before_train: False`
- The old `16xH20` IB run at [`trloo-8b-hfsdp8-pytorch-eager.train.16xH20.reward.16x4090.run.20260413-231743`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp8-pytorch-eager.train.16xH20.reward.16x4090.run.20260413-231743) is stopped; its `main.log` and `trainer.log` both stopped updating at `2026-04-14 02:14:04 UTC`, and the old repo-managed Ray tmux sessions were removed.
- The new run has already passed Ray bring-up, config load, dataset load, and actor/rollout worker initialization. At the latest check it had `8` local H20 GPUs reserved in Ray, `Training Progress: 0/4499000`, and `[RolloutProgress] step=1 ... phase=start` in `trainer.log`.

## 8B two-node H20 training relaunched on `gz01-h20-03` + `gz01-h20-05` with verified IB transport — SUPERSEDED

#### The initial two-node H20 launch used socket transport as a conservative fallback because the InfiniBand host view was mixed, but the user explicitly required IB, so the run was stopped, relaunched with `NCCL_NET=IB`, and then verified from Ray worker logs on both nodes instead of assuming that the env vars alone were enough

##### Problem & Impact

- The first two-node H20 launch was stable but intentionally used `NCCL_IB_DISABLE=1`, which did not satisfy the requirement to actually use InfiniBand.
- The host state was still mixed:
  - InfiniBand HCAs were `ACTIVE` / `LinkUp`
  - host-level `ibs*` netdevs remained `DOWN`
- `train_rl_common.sh` propagated `NCCL_NET` and `NCCL_IB_DISABLE` into Ray runtime env, but it did not yet propagate `NCCL_IB_HCA` or `NCCL_DEBUG_SUBSYS`, which made IB-targeted launch intent easier to lose across worker processes.

##### Resolution

- Stopped the socket-based two-node run cleanly by removing only the repo's current training tmux sessions and Ray processes on the head and worker containers.
- Relaunched the same `16xH20` training job through the canonical startup script with:
  - `TRAIN_NCCL_IB_DISABLE=0`
  - `TRAIN_NCCL_NET=IB`
  - `TRAIN_NCCL_DEBUG=INFO`
  - `KERNELGYM_SERVER_URL=http://10.0.18.3:18112`
  - `NNODES=2`
  - `NCCL_DEBUG_SUBSYS=INIT,NET`
- Verified the relaunch from worker-side Ray logs instead of only checking trainer config:
  - head-node worker logs showed `Initialized NET plugin IB` and `Using network IB`
  - `node5:/csl_verl` worker logs showed the same IB initialization on `mlx5_0:1` through `mlx5_3:1`
- Updated [`drkernel/kernel/scripts/rl/train_rl_common.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/train_rl_common.sh) so future Ray runtime envs also carry `NCCL_IB_HCA` and `NCCL_DEBUG_SUBSYS`.

##### Result & Current State

- The active IB run directory is [`trloo-8b-hfsdp8-pytorch-eager.train.16xH20.reward.16x4090.run.20260413-231743`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp8-pytorch-eager.train.16xH20.reward.16x4090.run.20260413-231743).
- The active head-node tmux session is [`train-8b-16xh20-ib`](/tmp/train-8b-16xh20-ib.log), with tee log [`/tmp/train-8b-16xh20-ib.log`](/tmp/train-8b-16xh20-ib.log).
- `trainer.log` confirms `nnodes: 2`, `fsdp_size: 8`, `server_url: http://10.0.18.3:18112`, `NCCL_IB_DISABLE: 0`, and `NCCL_NET: IB`.
- Ray worker logs on both nodes explicitly show:
  - `NET/IB`
  - `Initialized NET plugin IB`
  - `Assigned NET plugin IB to comm`
  - `Using network IB`
- Ray still reports `2` active nodes and `16` total GPUs, and the run remains in distributed worker/model bring-up with no new NCCL failure signatures at the latest check.

## Training infra split into shared helpers plus H20 and A800 profiles — COMPLETED

#### The orchestration layer had started to accumulate current H20-specific defaults inside one shared infra file, but the repo still needs to support the older A800 training and eval environment without cloning the same helper logic into a second copy

##### Problem & Impact

- The current H20 bring-up changed training-node access, Python activation, reward-repo wiring, and NCCL defaults enough that leaving all of that hardcoded in one shared infra file would make the A800 flows harder to preserve.
- Duplicating `infra_common.sh` into separate H20 and A800 versions would also duplicate the same command-routing, quoting, and Python-environment helper logic across startup and stop paths.
- Harness docs still described the training entrypoints as if there were one implicit infra environment, which no longer matched the code layout.

##### Resolution

- Split the shared infra logic into [`drkernel/kernel/scripts/rl/infra_lib.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/infra_lib.sh), which now holds the common execution helpers, Python environment prelude, and target-routing helpers.
- Moved cluster-specific defaults into:
  - [`drkernel/kernel/scripts/rl/infra_profiles/h20.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/infra_profiles/h20.sh)
  - [`drkernel/kernel/scripts/rl/infra_profiles/a800.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/infra_profiles/a800.sh)
- Reduced [`drkernel/kernel/scripts/rl/infra_common.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/infra_common.sh) to a loader layer that selects `TRAIN_CLUSTER_PROFILE`, defaults to `h20`, and then composes the shared infra helpers with the chosen profile.
- Updated [`drkernel/kernel/scripts/rl/start_training.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/start_training.sh) and [`.agents/skills/stop_training/scripts/stop_ray_training.sh`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/.agents/skills/stop_training/scripts/stop_ray_training.sh) so both accept `--profile h20|a800`.
- Decoupled reward-side repo wiring from training-side repo wiring so the reward startup path does not inherit the H20 training worktree path by accident.

##### Result & Current State

- The repo now supports both `h20` and `a800` training infra profiles without maintaining two copies of the orchestration helper layer.
- The active default remains `h20`, which matches the current two-node H20 training environment.
- The older A800 startup and checkpoint-eval paths can continue to load their own cluster defaults through the profile mechanism instead of relying on H20-specific values.
- `SPEC.md` and `INDEX.md` were updated alongside the code so the harness files now describe the profile-based infra layout instead of implying a single hardcoded training environment.

## 8B two-node H20 training launched on `gz01-h20-03` + `gz01-h20-05` with socket transport — SUPERSEDED

#### The requested second H20 node lived behind `ssh node5` plus an extra `docker exec csl_verl`, the worker repo was behind the current training worktree, the existing reward relay only listened on `127.0.0.1`, and the IB state was ambiguous for NCCL because the Mellanox ports were `ACTIVE` while the OS-level `ibs*` interfaces were `DOWN`; the launch therefore needed repo sync, system-Python validation, a shared cross-node reward entrypoint, and a conservative socket-based NCCL path before starting the 16xH20 run

##### Problem & Impact

- The requested training topology changed from the current single-node H20 run to `node3 + node5`, but `node5` is only reachable through the physical host plus `docker exec csl_verl`, not as a directly usable peer container.
- The `node5` worktree was on the correct branch name `vllm018` but behind the current head-node worktree, so relying on it as-is risked running a different startup path or missing the local vLLM 0.18.0 compatibility fixes.
- Cross-node reward access could not use the existing head-node-local relay at `http://127.0.0.1:18111`, because the worker container on `node5` could not reach that listener or `http://10.0.18.3:18111`.
- The H20 hosts exposed a mixed IB state:
  - Mellanox `/sys/class/infiniband/*/ports/1/{state,phys_state}` showed `ACTIVE` / `LinkUp`
  - the host-level `ibs*` network interfaces on both nodes were still `DOWN`
- The repo stop-training helper was not safe to use on these shared nodes because it assumes a repo venv path and force-kills all visible GPU compute PIDs, which could affect unrelated workloads.

##### Resolution

- Validated the worker path through `ssh node5` and `docker exec csl_verl`, then confirmed the container is running in `host` network mode so `bond0` is visible inside the worker container.
- Synced the training-critical worktree files from `gz01-h20-03` to `node5` with a tar-over-SSH copy, including:
  - `infra_common.sh`
  - `start_training.sh`
  - the H20 launchers
  - `train_rl_common.sh`
  - `setup_env.sh`
  - the local `vllm.lora.worker_manager.LoRAModel` fallback in `drkernel/verl/verl/utils/vllm/utils.py`
- Reinstalled the vendored `drkernel/verl` checkout editable into the system Python on `node5` and revalidated `which python`, `sys.executable`, `VIRTUAL_ENV`, `verl`, `vllm`, `ray`, and `torch` from the worker container without using `.venv`.
- Verified that the model path `/nfs/FM/chenshuailin/checkpoints/hkust-nlp/drkernel-8b-coldstart` and the repo-local train/val parquet files are present on both training nodes.
- Reused the existing reward stack but switched the two-node training endpoint to the cross-node relay at `http://10.0.18.3:18112`, which is exposed by local `socat` on `gz01-h20-03` and reachable from both the head container and `node5:/csl_verl`.
- Cleared only the repo's own old Ray/tmux sessions on the two training containers instead of using the unsafe shared-node kill-all helper.
- Launched the run through the repo orchestrator with:
  - `TRAIN_NCCL_IB_DISABLE=1`
  - `KERNELGYM_SERVER_URL=http://10.0.18.3:18112`
  - `NNODES=2`
  - launcher `8b_trloo_hfsdp8_pytorch_eager.16xH20.sh`
  - tmux session `train-8b-16xh20-socket`

##### Result & Current State

- The active two-node run directory is [`trloo-8b-hfsdp8-pytorch-eager.train.16xH20.reward.16x4090.run.20260413-230832`](/data3/csl/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp8-pytorch-eager.train.16xH20.reward.16x4090.run.20260413-230832).
- The active head-node tmux session is [`train-8b-16xh20-socket`](/tmp/train-8b-16xh20-socket.log), with tee log [`/tmp/train-8b-16xh20-socket.log`](/tmp/train-8b-16xh20-socket.log).
- `ray status --address=10.0.18.3:6379` now reports `2` active nodes and `16` total GPUs, with `16.0/16.0 GPU` reserved by placement groups during worker initialization.
- The trainer config for this run confirms:
  - `nnodes: 2`
  - `fsdp_size: 8`
  - `server_url: http://10.0.18.3:18112`
  - `NCCL_IB_DISABLE: 1`
  - `test_freq: 0`
  - `val_before_train: False`
- The worker container on `gz01-h20-05` is actively participating in bring-up and shows `8` `ray::WorkerDict.actor_rollout_init_model` GPU processes, each using about `2168 MiB`.
- The run has passed Ray cluster bring-up, dataset loading, and distributed worker/model initialization. At the latest check, `Training Progress` had not yet appeared, so the run is still in early distributed startup rather than failed.

## 8B single-node H20 training relaunched with relative data paths — SUPERSEDED

#### The 8B launcher still assumed old absolute dataset paths and the retired two-node remote startup path, so it was converted to the current single-node H20 environment, patched for the current local vLLM stack, and relaunched locally in tmux with the reward relay on `127.0.0.1:18111`; the active configuration now skips validation and uses a larger PPO micro-token budget

##### Problem & Impact

- The 8B eager launcher still pointed at absolute dataset paths under the old shared-tree location instead of the new local copies downloaded into the current worktree.
- The old startup path still assumed the retired two-node remote training cluster, which is not the current environment.
- Local startup on this node initially failed repeatedly for environment reasons:
  - `import verl` resolved to the outer namespace directory instead of the vendored package, so `DataProto` was missing.
  - `scipy` was not installed, so `refresh_moderate_sampler.py` failed during import.
  - The vendored `verl` code still imported `LoRAModel` from `vllm.lora.models`, but this node is running `vllm==0.18.0`, where that symbol is re-exported from `vllm.lora.worker_manager`.
  - The host Python environment was also missing runtime packages that the code path now reaches on this node, specifically `tenacity` and `sandbox-fusion`.
- After those startup issues were fixed, the active 8B configuration still performed validation, but the requested operating mode is to skip validation entirely and start training directly.

##### Resolution

- Updated [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh) so that:
  - train/val parquet paths use relative paths under `drkernel/data/`
  - the default reward URL is `http://127.0.0.1:18111`
  - the default training hardware label is `8xH20`
  - the default node count is `1`
  - the script `cd`s into `drkernel/` so those relative paths resolve consistently
- Updated [`drkernel/setup_env.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/setup_env.sh) to prepend `drkernel/verl` to `PYTHONPATH`, which fixes `import verl` for this repo checkout.
- Installed `scipy` into the current Python environment after the training import path advanced far enough to expose that missing dependency.
- Replaced the incompatible local `transformers 5.3.0` with `transformers==4.56.0`, which matches the repo setup expectation and restores `AutoModelForVision2Seq`.
- Patched [`drkernel/verl/verl/utils/vllm/utils.py`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/verl/verl/utils/vllm/utils.py) to fall back to `vllm.lora.worker_manager.LoRAModel` when `vllm.lora.models` is absent, which restores compatibility with the current local `vllm==0.18.0`.
- Installed the missing runtime packages `tenacity==8.2.3` and `sandbox-fusion`, both of which are already expected by the repo's requirements/setup flow.
- Installed the vendored [`drkernel/verl`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/verl) checkout as an editable package with `python -m pip install --no-deps -e .`, so the system Python now resolves `verl` directly to the repo worktree.
- Updated [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh) again so the default launch behavior now:
  - skips initial validation with `VAL_BEFORE_TRAIN=False`
  - disables periodic validation with `TEST_FREQ=0`
  - increases `PPO_MICRO_TOKEN` from `8192` to `16384`
- Split the 8B eager launcher into hardware-specific files:
  - [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh)
  - [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.16xA800.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.16xA800.sh)
- Kept [`drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.sh) as a compatibility wrapper to the H20 launcher so older references do not break immediately.
- Started the local Ray head and the 8B launcher inside tmux session `train-8b-h20-pytorch-eager` on `gz01-h20-03`, instead of using the old remote two-node launcher path.

##### Result & Current State

- The active 8B run directory is [`trloo-8b-hfsdp8-pytorch-eager.train.8xH20.reward.16x4090.run.20260413-115602`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/logs/trloo-8b-hfsdp8-pytorch-eager.train.8xH20.reward.16x4090.run.20260413-115602).
- The active local tee log is [`/tmp/train-8b-h20-pytorch-eager.log`](/tmp/train-8b-h20-pytorch-eager.log).
- The active run is using:
  - relative dataset paths `data/drkernel-rl-data/cuda_llm_rl_thinking_1025.parquet` and `data/drkernel-validation-data/validation_data_thinking.parquet`
  - reward endpoint `http://127.0.0.1:18111`
  - single-node `trainer.nnodes=1`
  - `trainer.val_before_train=False`
  - `trainer.test_freq=0`
  - `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384`
- The run has progressed past configuration loading and into trainer initialization:
  - Ray connected at `10.0.18.3:6379`
  - trainer config confirms `test_freq: 0` and `val_before_train: False`
  - the current relaunch has already progressed past the earlier `vllm.lora.models`, `tenacity`, and `sandbox_fusion` import failures and remains alive in tmux while the async vLLM stack continues initializing
- An execution adjustment was required because the old `start_training.sh` path still targets the retired remote A800 nodes and could not be used as-is on this host; the workaround was a local tmux launch on the current H20 node, and that gap still exists in the old remote launcher path.

## Single-node H20 cluster details and local training data refreshed — COMPLETED

#### The repo's run-specific operating facts were still pinned to the old two-node A800 cluster and to dataset paths that do not exist on the new cloud training host, so the current cluster and local data state were re-validated and rewritten

##### Problem & Impact

- `SPEC.md` still described the previous `192.168.16.18` / `192.168.16.24` two-node A800 training cluster as if it were current.
- The old dataset paths under `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM/drkernel/data/...` were not present on the new cloud training host.
- Without refreshing those run-specific facts, follow-up launch work would continue to assume the wrong training topology, stale reward endpoint assumptions, and missing local parquet paths.

##### Resolution

- Verified the current cloud training host as `gz01-h20-03` with observed IPs `10.0.18.3` and `172.17.0.1`.
- Verified the current training topology as a single node with `8 x NVIDIA H20` GPUs.
- Re-checked the cloud-side reward relay and confirmed that `http://127.0.0.1:18111` is healthy and exposes `8` workers from `reward-39` plus `8` workers from `reward-40`.
- Downloaded the current local training parquet from `hkust-nlp/drkernel-rl-data` and the validation parquet from `hkust-nlp/drkernel-validation-data` into the current worktree's `drkernel/data/` directory.
- Updated [`SPEC.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/SPEC.md) to reflect the new single-node H20 cluster, the relay-based reward endpoint, the absence of the old data paths on this node, and the new local dataset copies.
- Removed the stale old A800 training-node section from [`SPEC.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/SPEC.md) instead of preserving it as historical runbook material.

##### Result & Current State

- `SPEC.md` now contains only the current single-node H20 cloud environment facts and no longer keeps the retired two-node A800 training-node details.
- Local dataset copies are now present at:
  - [`drkernel/data/drkernel-rl-data/cuda_llm_rl_thinking_1025.parquet`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/data/drkernel-rl-data/cuda_llm_rl_thinking_1025.parquet)
  - [`drkernel/data/drkernel-validation-data/validation_data_thinking.parquet`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/data/drkernel-validation-data/validation_data_thinking.parquet)
- The train parquet is `399M` on disk and downloaded in about `27s`; the validation parquet is `96K` and downloaded in about `7s`, so download speed was acceptable on this node.
- A small repo helper gap was encountered: `drkernel/kernel/scripts/preprocess/pull_from_hub.py` fails with its default empty `ignore_patterns` handling, so direct `huggingface_hub.snapshot_download(...)` was used as the temporary workaround for this fetch.

## Cloud-side reward reachability validated through localhost relay — COMPLETED

#### The cloud host cannot reach A's reward API by direct LAN IP, but it can reach the reward service through the reverse-SSH relay on `127.0.0.1:18111`, and that relay exposes the 4090 worker pool

##### Problem & Impact

- The older remote-reward handoff still treated `http://192.168.16.39:8111` as the cloud-side reward URL, but the newer operating procedure had already shifted to a reverse-SSH relay entrypoint on the cloud host.
- Without validating the current cloud-side path directly on the host, training could still be pointed at a dead direct URL and fail to reach the reward service even though the relay path was healthy.

##### Resolution

- Verified the cloud host execution context on `gz01-h20-03` with `which python`, `sys.executable`, and `VIRTUAL_ENV`.
- Tested `http://127.0.0.1:18111/health` and `http://127.0.0.1:18111/workers/status` from the cloud host.
- Tested the older direct path `http://192.168.16.39:8111/{health,workers/status}` from the same host with an 8-second timeout.

##### Result & Current State

- `http://127.0.0.1:18111/health` returned `200 OK` on `2026-04-13T10:56:40Z` with reward service health data.
- `http://127.0.0.1:18111/workers/status` returned `200 OK` and included multiple `reward-40_gpu_*` entries, confirming that the 4090 worker host is reachable behind the relay.
- The direct cloud-to-A URL `http://192.168.16.39:8111` timed out for both `/health` and `/workers/status`, so it is not currently a valid cloud-side training endpoint.
- The current cloud-host reward entrypoint should therefore be treated as `http://127.0.0.1:18111` unless and until container-network validation requires the bridge-container alternatives documented in the relay runbook.

## Remote training with local reward topology guidance — COMPLETED

#### The current architecture already supports remote rollout/PPO with local reward via HTTP, so the immediate blocker is network reachability rather than a new standalone orchestrator

##### Problem & Impact

- There was a question about whether cloud-side rollout and PPO update can be combined with a local 4090 reward server, or whether this would require first splitting out a new standalone orchestrator.
- Without clarifying where reward requests originate and which side performs PPO updates, it was easy to over-scope the problem into a full training-control-plane redesign.

##### Resolution

- Reviewed the current trainer, rollout, and reward-client boundaries.
- Documented the conclusion in [`handoffs/completed/HANDOFF_REMOTE_TRAIN_LOCAL_REWARD.md`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/handoffs/completed/HANDOFF_REMOTE_TRAIN_LOCAL_REWARD.md).
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
