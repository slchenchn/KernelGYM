---
name: start-training
description: Start a drkernel training run for this repository using the canonical launcher scripts and the repo's orchestrator script instead of hand-built SSH/tmux commands. Use when the user asks to launch or resume training on the currently configured training nodes, with or without targeted environment overrides.
---

# Start Training

Use this skill when the user asks to start or resume a training run in this repository.

The goal is to avoid hand-built orchestration commands. Prefer the repo's canonical startup script:
`drkernel/kernel/scripts/rl/start_training.sh`

and the existing launcher scripts under:
`drkernel/kernel/scripts/rl`

## When To Use

- Start a fresh 14B or 8B training run.
- Resume an existing run from a specific log/checkpoint directory.
- Launch training with a small number of explicit env overrides such as `RUN_LOG_DIR` or `REWARD_TASK_TIMEOUT`.
- Rebuild the Ray cluster and training tmux session without manually reconstructing SSH/tmux commands.

## Core Rule

Do not hand-build the full remote orchestration unless the repo script is broken in a way that blocks launch.
Use the canonical startup script and existing launcher scripts first.
If the startup script is too hardcoded for the requested launch, fix the script and then use it.

## Workflow

1. Pick the existing launcher script that matches the requested run.
   - Examples:
     - `14b_coldstart_trloo_hfsdp8_pytorch_eager.sh`
     - `8b_trloo_hfsdp8_pytorch_eager.sh`
2. Validate launch prerequisites from the head node before starting.
   - `which python`
   - `sys.executable`
   - `VIRTUAL_ENV`
   - reward health endpoint
   - model path
   - dataset path
   - if resuming, the target `RUN_LOG_DIR` and checkpoint root
3. Check whether the currently configured training nodes are already occupied.
   - Look for an active training or eval cluster before launching a new one.
   - Do not assume a fresh launch will clean up an existing run.
   - If the nodes are occupied and the user intends to switch or relaunch, first use the repo's stop-training procedure:
     - skill: `.agents/skills/stop_training/SKILL.md`
     - canonical command: `bash .agents/skills/stop_training/scripts/stop_ray_training.sh`
4. Use the canonical startup script.
   - Default form:
     ```bash
     bash drkernel/kernel/scripts/rl/start_training.sh \
       --train-script drkernel/kernel/scripts/rl/<launcher>.sh
     ```
5. For resume or targeted launch changes, pass env overrides through `--env`.
   - Example:
     ```bash
     bash drkernel/kernel/scripts/rl/start_training.sh \
       --skip-reward \
       --train-script drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh \
       --env RUN_LOG_DIR=<existing_run_log_dir> \
       --env REWARD_TASK_TIMEOUT=30
     ```
6. Keep the training under tmux.
   - The startup script launches a head-node tmux session automatically.
7. After launch, verify the run rather than assuming success from the tmux pane alone.
   - Confirm the head-node tmux session exists.
   - Confirm `ray status` on the head node.
   - Confirm the launcher env overrides landed if they were requested.
   - Confirm the run is writing to the expected log path.

## Key Options In `start_training.sh`

- `--train-script PATH`
  - Selects the existing launcher script.
- `--skip-reward`
  - Reuses the existing reward stack.
- `-f` / `--force-reward`
  - Force-restarts the reward stack with `start_reward.sh -f`.
- `--env KEY=VALUE`
  - Adds launcher env overrides without editing the launcher for one-off runs.
- `--tmux-session NAME`
  - Overrides the default session name derived from the launcher basename.
- `--local-log PATH`
  - Overrides the head-node `tee` log path.
- `--dry-run`
  - Prints the resolved plan without starting anything.

## Repo-Specific Notes

- Current node, container, reward-endpoint, and transport facts belong in `SPEC.md`, not in this skill.
- The orchestrator script reads the current training-target settings from `drkernel/kernel/scripts/rl/infra_common.sh`. Do not hardcode launch hosts, containers, or environment paths in this skill.
- If a user request cannot be expressed with the current startup script flags, modify `start_training.sh` rather than falling back immediately to ad hoc SSH orchestration.
- Resumed runs share the old run directory, so old log content and new log content will coexist. Use the new process pid and fresh config dump to distinguish the resumed run from historical log lines.
- For live progress checks after launch, use the separate training-status skill.

## Canonical Commands

Fresh 14B eager launch:

```bash
bash drkernel/kernel/scripts/rl/start_training.sh \
  --train-script drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh
```

Fresh 8B eager launch:

```bash
bash drkernel/kernel/scripts/rl/start_training.sh \
  --train-script drkernel/kernel/scripts/rl/8b_trloo_hfsdp8_pytorch_eager.sh
```

Resume an old run and shorten reward timeout:

```bash
bash drkernel/kernel/scripts/rl/start_training.sh \
  --skip-reward \
  --train-script drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh \
  --env RUN_LOG_DIR=<existing_run_log_dir> \
  --env REWARD_TASK_TIMEOUT=30
```
