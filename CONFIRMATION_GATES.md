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

## Pending Confirmation Gates

### 1. Model Testing Timeout

- Action: test a model
- Confirm first: whether model testing should use `REWARD_TASK_TIMEOUT=30` rather than the default timeout.
- Status: confirmed for the current 14B checkpoint-eval request on 2026-04-13
- Confirmed result:
  - use `REWARD_TASK_TIMEOUT=30`
