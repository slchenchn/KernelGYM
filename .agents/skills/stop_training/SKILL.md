---
name: stop-training
description: Stop an active drkernel training or evaluation cluster for this repository on the currently configured training nodes. Use when the user asks to stop a live run, free the training nodes, or tear down the Ray cluster before restart or checkpoint evaluation.
---

# Stop Training

Use this skill for this repository's training and eval jobs that run through the currently configured
targets in `drkernel/kernel/scripts/rl/infra_common.sh`.

## When To Use

- Stop a live RL training run.
- Stop an eval run that occupies the same training nodes.
- Free the two training nodes before checkpoint eval, restart, or relaunch.
- Verify that the cluster is really down instead of only assuming it from a tmux pane.

## Workflow

1. Confirm the target is active before stopping it.
   - Check target log timestamps such as `main.log` and `trainer.log`.
   - Check remote GPUs and Ray actors on both training nodes.
   - Do not infer run state from a local tmux pane alone.
2. Stop the remote training cluster.
   - Run `bash .agents/skills/stop_training/scripts/stop_ray_training.sh`.
   - Add `--profile` when the target differs from the worktree default selected by `.infra_profile.local.sh`.
3. Verify the stop succeeded.
   - Remote GPUs should show no compute apps.
   - Remote `ray stop --force` should have removed the live cluster.
   - Target logs should stop updating after a short wait.
4. If a local tmux orchestrator is still running, stop that separately after the remote cluster is down.

## Canonical Command

From the repo root:

```bash
bash .agents/skills/stop_training/scripts/stop_ray_training.sh
```

Explicit profile override:

```bash
bash .agents/skills/stop_training/scripts/stop_ray_training.sh --profile a800
```

## What The Script Does

- Sources the canonical training-target settings from `infra_common.sh`.
- Stops Ray on the head node and worker node.
- Prints post-stop GPU and compute-app state so the caller can confirm the nodes are free.

## Repo-Specific Notes

- Current node, container, and environment facts belong in `SPEC.md`, not in this skill.
- Do not hardcode launch hosts, ports, containers, or environment paths in the workflow. Read them from `infra_common.sh`.
- Prefer `.infra_profile.local.sh` or `--profile` over editing tracked default profile values in infra scripts.
- The stop script should work with the current Python environment selection instead of assuming a repo venv.
- If an orchestrator wraps remote work as `ssh ... | tail -n 20`, the tmux pane is not a live
  progress source. Check files, remote processes, and GPU state directly.
- If the user asks whether the stop really happened, verify using both:
  - remote process and GPU state
  - target log timestamps after a short wait

## Manual Verification Snippets

Check remote GPUs after stopping:

```bash
source drkernel/kernel/scripts/rl/infra_common.sh
run_on_train_head \
  'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; \
   echo ---; \
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true'
run_on_train_worker \
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
