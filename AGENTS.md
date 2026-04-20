# Agents Notes

## Scope

`AGENTS.md` is for repository-level, relatively stable working conventions and principles for Codex.

Do not put volatile deployment details here, including:

- machine IPs or hostnames
- one-off SSH endpoints
- temporary model paths
- run-specific dataset paths
- experiment-specific topology

Put those details in `SPEC.md` or a dedicated runbook instead.

## Runbooks

- For the `KernelGYM-vllm018` online-quantization investigation, use
  `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/handoffs/in_progress/HANDOFF_QUANTIZED_ROLLOUT.md`
  as the primary handoff / reproduction document.

## Harness Files

The repository's top-level harness files are the stable operational documents that guide work in this repo.

- `CLAUDE.md` records repository-level principles and stable collaboration rules for Claude Code.
- `AGENTS.md` records repository-level principles and stable collaboration rules for Codex.
- `PROGRESS.md` is the progress log for decision-level milestones, resolutions, and current state.
- `SPEC.md` records specific run details such as environments, service endpoints, model paths, dataset paths, hyperparameters, and other experiment-specific configuration.
- `INDEX.md` is the index for important documents, logs, scripts, and code entry points so key references are easy to locate.
- `CONFIRMATION_GATES.md` records additional confirmation gates that the user explicitly asked to write down. It is not an exhaustive list of every situation that may require confirmation; the agent may still ask the user to confirm other actions when judgment, risk, or ambiguity makes confirmation necessary. After confirmation, delete the item or mark the confirmed result as appropriate.
- `PROGRESS.md`, `SPEC.md`, and `INDEX.md` may and should be updated proactively by the working agent when diagnosis, configuration, run-specific facts, or important references materially change. `CONFIRMATION_GATES.md` should not be populated proactively; add items there only when the user explicitly asks to record an additional confirmation gate.
- `AGENTS.md` and `CLAUDE.md` are not routine scratchpads; do not update them proactively. Change them only when the user explicitly asks for an instruction or policy update.

## Working Conventions

1. If the user asks to try a shared uv/virtual environment across nodes, validate it explicitly before relying on it.
2. Any persistent `KernelGYM` or `drkernel` process must run under `tmux`.
3. Before launching distributed jobs, verify environment selection explicitly:
   - `which python`
   - `sys.executable`
   - `VIRTUAL_ENV`
4. Before launching training, verify that service endpoints, model paths, and dataset paths are reachable from the node that will use them.
5. If environment or process-state issues block startup, record the exact blocking point and the minimal remediation.
6. Do not proactively delete `__pycache__` directories during cleanup unless the user explicitly asks for that.
7. When creating commits, use a commit message in `xx:yy` form.
8. Keep local git history clean while work is still unpublished:
   - if commit scope needs to change, rewrite the local commits instead of adding follow-up "remove from commit" or "fix previous commit scope" commits
   - do not create a commit and then immediately compensate for it with another local cleanup commit when the history can still be rewritten safely

## Execution Policy

1. If user instructions need adjustment during execution, it is acceptable to adapt pragmatically instead of stalling.
2. Any such adjustment must be reported back at the end, including:
   - why the original instruction could not be executed as-is
   - what was changed
   - the result after the change
   - the current status and remaining gaps
3. If execution encounters a situation that the relevant skill did not anticipate, report that explicitly afterwards.
   - state what the skill did not cover
   - state what workaround, judgment call, or temporary procedure was used instead
   - state whether the gap still exists in the skill after the task
4. In progress documents and final summaries, prefer a combined "problem and resolution" structure instead of separating "problems" and "adjustments".
5. When recording execution deviations or milestone entries, write them in this order:
   - what problem was encountered
   - how it was resolved or mitigated
   - what result that produced
   - what the current state is after the change
6. In progress documents such as `PROGRESS.md`, organize each milestone entry as a level-4 heading.
7. Under each milestone entry, use exactly these sections:
   - `Problem & Impact`
   - `Resolution`
   - `Result & Current State`
8. Update `PROGRESS.md` when diagnosis, configuration, or externally visible run state materially changes; do not defer those changes until the end.
9. Treat `PROGRESS.md` as a decision log, not a chronological transcript.
10. Keep related restarts, checks, and monitor updates under the same milestone entry when they belong to the same root-cause thread.
11. Prefer fewer higher-signal `PROGRESS.md` entries over many tiny incremental ones when no decision boundary changed.
12. Omit routine restarts, repeated health checks, and step-by-step operator actions unless they changed the diagnosis, configuration, or decision boundary.
13. Do not record routine stop, resume, or relaunch actions in `PROGRESS.md` as milestones by themselves.
14. If a stop, resume, or relaunch matters, record only the resulting diagnosis change, configuration change, or experiment-boundary change, not the operator action itself.
15. Detailed live training-status checking procedures belong in a repo-local skill, not in `AGENTS.md`.
