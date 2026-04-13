---
name: stop-training
description: Stop an active drkernel training or evaluation cluster for this repository on the two A800 training nodes. Use when the user asks to stop a live run, free the A800 nodes, or tear down the Ray cluster before restart or checkpoint evaluation.
---

# Stop Training

Use this skill for this repository's A800 training and eval jobs that run through the node and venv
definitions in [`drkernel/kernel/scripts/rl/infra_common.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/infra_common.sh).

## When To Use

- Stop a live RL training run.
- Stop an eval run that occupies the same A800 nodes.
- Free the two training nodes before checkpoint eval, restart, or relaunch.
- Verify that the cluster is really down instead of only assuming it from a tmux pane.

## Workflow

1. Confirm the target is active before stopping it.
   - Check target log timestamps such as `main.log` and `trainer.log`.
   - Check remote GPUs and Ray actors on both training nodes.
   - Do not infer run state from a local tmux pane alone.
2. Stop the remote training cluster.
   - Run `bash skills/stop_training/scripts/stop_ray_training.sh`.
3. Verify the stop succeeded.
   - Remote GPUs should show no compute apps.
   - Remote `ray stop --force` should have removed the live cluster.
   - Target logs should stop updating after a short wait.
4. If a local tmux orchestrator is still running, stop that separately after the remote cluster is down.

## Canonical Command

From the repo root:

```bash
bash skills/stop_training/scripts/stop_ray_training.sh
```

## What The Script Does

- Sources the canonical node and venv settings from `infra_common.sh`.
- Stops Ray on the head node and worker node.
- Kills any remaining GPU compute-app PIDs on both nodes.
- Prints post-stop GPU and compute-app state so the caller can confirm the nodes are free.

## Repo-Specific Notes

- Do not hardcode training node IPs in the workflow. Read `HEAD_NODE`, `HEAD_PORT`,
  `WORKER_NODE`, `WORKER_PORT`, and `VENV` from `infra_common.sh`.
- `ray` may not be on the default remote `PATH`. Use the repo venv path explicitly or activate the
  venv first.
- If an orchestrator wraps remote work as `ssh ... | tail -n 20`, the tmux pane is not a live
  progress source. Check files, remote processes, and GPU state directly.
- If the user asks whether the stop really happened, verify using both:
  - remote process and GPU state
  - target log timestamps after a short wait

## Manual Verification Snippets

Check remote GPUs after stopping:

```bash
source drkernel/kernel/scripts/rl/infra_common.sh
ssh -p "${HEAD_PORT}" "root@${HEAD_NODE}" \
  'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; \
   echo ---; \
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true'
ssh -p "${WORKER_PORT}" "root@${WORKER_NODE}" \
  'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; \
   echo ---; \
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true'
```

Check that target logs stopped moving:

```bash
stat -c '%y %n' /path/to/main.log /path/to/trainer.log
sleep 5
stat -c '%y %n' /path/to/main.log /path/to/trainer.log
```
