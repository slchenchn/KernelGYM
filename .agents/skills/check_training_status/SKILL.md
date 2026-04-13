---
name: check-training-status
description: Check live drkernel training progress for this repository from a run directory or main.log. Use when the user asks to check training status, verify run phase, distinguish completed steps from retries or oversampling, or generate training and eval dynamics plots before making a status claim.
---

# Check Training Status

Use this skill for live or recent training runs in this repository when the user asks whether a run is progressing, what phase it is in, or whether a specific step really completed.

## When To Use

- Check current training progress.
- Verify whether a run is in init validation, rollout, actor update, retry, oversampling, or stalled.
- Distinguish actual new progress from worker activity, queue activity, or heartbeats.
- Generate training and eval dynamics plots before summarizing status.
- Re-check a run after a restart, resume, timeout, or suspected logging issue.

## Required Inputs

- A run directory or its `main.log`.
- Optionally the related `trainer.log`, `rollout.log`, `reward.log`, `vllm.log`, and WandB state when present.

## Workflow

1. Start from the full log, not a tail.
   - Do not infer run phase from a tmux pane or a short tail alone.
   - Search the full `main.log` for the latest `Initial validation metrics`, `Training Progress`, and `step:N` entries.
2. Generate plots first.
   - Use the canonical plotting entrypoint documented in `INDEX.md`.
   - Generate training plots alongside log inspection, and eval plots too if `eval_results` exists.
3. Verify completion state, not only activity.
   - Decide whether the current step truly completed or is only retrying, oversampling, or heartbeating.
   - Treat worker activity, queue activity, GPU utilization, and heartbeats as insufficient proof of completed progress.
4. Separate these counters in every status report.
   - successful visible train step
   - current in-flight step
   - retry / oversampling state
   - WandB run step or summary state
5. Handle resumed runs carefully.
   - Do not treat local `wandb-summary.json` as live truth unless its freshness was verified explicitly.
   - If WandB is stale, broken, or disabled, say that directly.
6. If evidence is partial, keep the uncertainty explicit.
   - State what is confirmed.
   - State what is still inference.
7. If correcting an earlier interpretation, preserve the delta.
   - the earlier mistaken assumption
   - the evidence that invalidated it
   - the corrected conclusion

## Output Requirements

When answering a training-status question, explicitly report:

- successful visible train step
- current in-flight step
- retry / oversampling state
- WandB state

Also state whether the run has made real new progress or is only active.

## Repo-Specific Notes

- The canonical plotting entrypoint is indexed in `INDEX.md`; do not hardcode superseded plotting scripts.
- If log routing is suspected to be noisy or incomplete, cross-check `trainer.log`, `rollout.log`, `reward.log`, and `vllm.log` before making a phase-level claim.
- If the user asks for progress on a live training run, combine log evidence and plot evidence in the same answer.
