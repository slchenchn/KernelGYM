# Confirmation Gates

## Purpose

`CONFIRMATION_GATES.md` records additional user-confirmation gates that the user explicitly asked to write down.

It is not a general reminder list, it is not populated proactively, and it is not an exhaustive list of every situation where the agent may need confirmation.

## Enforcement Rule

1. If an action matches an item in this file, ask the recorded confirmation question before executing that action.
2. If an action does not have a matching item in this file, the agent may still ask for confirmation when judgment, risk, or ambiguity makes confirmation necessary.
3. Add a new item only when the user explicitly asks to record that confirmation gate.
4. After the user confirms, either:
   - delete the item if the confirmation was one-off and already consumed, or
   - mark it as confirmed and record the confirmed choice if the result should remain visible for follow-up work.

## Gate Types

### 1. One-Off Gates That Can Be Canceled Once The Answer Is Known

- Rule:
  - If the answer is already available from the current conversation, harness files, or a previously confirmed result, do not ask again.
  - Move the item out of the active gate list, or mark it as consumed, once the answer is known.

#### Consumed Example: Model Testing Timeout

- Action: test a model
- Confirm first: whether model testing should use `REWARD_TASK_TIMEOUT=30` rather than the default timeout.
- Status: consumed for the 2026-04-13 14B checkpoint-eval request
- Consumed result:
  - use `REWARD_TASK_TIMEOUT=30`
- Active gate state:
  - canceled as an always-ask gate
  - ask again only if a future request does not already make the timeout choice clear

#### Consumed: Checkpoint Eval Merge Mode

- Action:
  - start checkpoint testing or checkpoint eval that runs the merge flow before evaluation
  - examples include running [`merge_and_eval_checkpoints.sh`](/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018/drkernel/kernel/scripts/rl/merge_and_eval_checkpoints.sh) or any equivalent checkpoint-eval workflow that merges FSDP shards to HF format first
- Confirm first:
  - whether checkpoint eval should use GPU merge or CPU merge
- Status:
  - consumed
- Consumed result:
  - always use GPU merge
  - remove CPU merge as an available path
- Active gate state:
  - canceled as an always-ask gate
  - ask again only if a future request explicitly asks to reintroduce CPU merge

### 2. Gates That Must Be Confirmed Every Time

#### 1. Reward Env Stop Or Restart

- Action:
  - stop, restart, force-recreate, or otherwise disrupt the current reward environment
  - examples include stopping or restarting reward API, Redis, reward workers, or running reward bring-up in a way that replaces the existing reward stack
- Confirm first:
  - whether the agent should proceed with closing or restarting the reward env
- Status:
  - always ask
