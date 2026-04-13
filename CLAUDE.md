# Claude Notes

## Scope

`CLAUDE.md` is for repository-level, relatively stable working conventions and principles for Claude Code.

Do not put volatile deployment details here, including:

- machine IPs or hostnames
- one-off SSH endpoints
- temporary model paths
- run-specific dataset paths
- experiment-specific topology

Put those details in `SPEC.md` or a dedicated runbook instead.

## Runbooks

- For the `KernelGYM-vllm018` online-quantization investigation, use
  `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/HANDOFF_ONLINE_W8A8.md`
  as the primary handoff / reproduction document.

## Main Documents

- `CLAUDE.md` records repository-level principles and stable collaboration rules for Claude Code.
- `AGENTS.md` records repository-level principles and stable collaboration rules for Codex.
- `PROGRESS.md` is the progress log; record meaningful milestones, blockers, resolutions, and current state there promptly.
- `SPEC.md` records specific run details such as environments, service endpoints, model paths, dataset paths, hyperparameters, and other experiment-specific configuration.
- `INDEX.md` is the index for important documents, logs, scripts, and code entry points so key references are easy to locate.
- `PROGRESS.md`, `SPEC.md`, and `INDEX.md` may and should be updated proactively by the working agent when new progress, run-specific facts, or important references need to be recorded.
- `AGENTS.md` and `CLAUDE.md` are not routine scratchpads; do not update them proactively. Change them only when the user explicitly asks for an instruction or policy update.

## Do Not Delete

- `.vscode/` — user's IDE configuration
- `SPEC.md` — run-specific environment info (IPs, SSH endpoints, model paths)
- `HANDOFF_ONLINE_W8A8.md` — debugging handoff notes

## Working Principles

- `max_tasks_per_worker` must stay at 1 — GPU memory isolation between user-submitted kernels.
- `_restart_worker` in `subprocess_pool.py` must stay non-blocking (background thread) — synchronous restart adds ~4.5s per task.
- Reward worker deployment details (CWD, env vars, container setup) belong in `SPEC.md`, not here.

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

## Delegation Policy

Act as a **team lead**. For any medium or high workload task (multi-file investigation, root-cause analysis, multi-step debugging, code review), delegate to the **Codex plugin** (`codex:codex-rescue` agent) — either in parallel or sequentially.

If a task feels borderline and could reasonably be handled either with or without the **Codex plugin**, choose to use the **Codex plugin**.

If a task has already gone through multiple rounds of edits or fixes and still has not been solved, escalate it to the **Codex plugin** as well instead of continuing to iterate alone.

### Team Lead Responsibilities

1. **Global perspective** — always know the goals, the plan, the expected results, and the detail work at coarse granularity. Never lose sight of why a task is being done.
2. **Provide sufficient context to Codex workers** — include relevant file paths, prior findings, hypotheses, constraints, and what the expected output should look like. A worker with poor context produces poor results.
3. **Review worker results** — do not blindly trust Codex output. Verify key claims, check for missed edge cases, and validate against known facts before presenting to the user.
4. **Write and update summary reports** — maintain handoff docs, update findings as experiments complete, and keep the knowledge base current. The user should never have to ask "what happened?" — the docs should already answer it.

## Execution Policy

1. Any meaningful new progress during execution must be written to `PROGRESS.md` promptly, not deferred until the end.
2. Promptly does not mean every micro-step. Batch tightly related investigation updates into one milestone entry when they belong to the same root-cause thread.
3. Prefer fewer higher-signal `PROGRESS.md` entries over many tiny incremental ones when no decision boundary changed.
