# Reward Environment Robustness Handoff

**Date**: 2026-04-20  
**Last updated**: 2026-04-20  
**Status**: Root cause diagnosed; hardening plan drafted; no robustness fix implemented yet

## Summary

The reward environment on `192.168.16.39` and `192.168.16.40` is currently not robust enough for
continuous generated-kernel evaluation.

Two distinct failure layers were confirmed:

1. **service survivability is weak**
   - both reward containers were configured with `restart=no`
   - after host reboot, the reward API did not come back automatically
   - this turned a host-level recovery action into a prolonged reward outage
2. **the evaluated workload can destabilize the GPUs**
   - the reward workload repeatedly triggered NVIDIA `Xid 13` / `31` / `43`
   - multiple real `Xid 109` (`CTX SWITCH TIMEOUT`) events were recorded across several GPUs on
     both hosts
   - the faulting process was consistently `python3`, matching the reward-side evaluation path

The current evidence points more strongly to **workload-induced GPU faulting or workload plus
driver/runtime interaction** than to a single persistently bad GPU board.

## Impact

- reward became fully unavailable on both `39` and `40`
- fresh checkpoint eval on `.18` produced contaminated or invalid outputs
- `step_20` of run `20260419-131619` collapsed to zero because reward requests failed
- `step_30` should not be trusted until reward is restored and revalidated

## What Happened

### Immediate outage trigger

- both hosts accepted a `root` SSH session from `192.168.120.100`
- both hosts then entered a clean `systemd` reboot sequence around `2026-04-20 14:04 +08:00`
- reward containers exited during shutdown
- after reboot, neither reward container restarted automatically because both were configured with
  `restart=no`

### Deeper technical trigger

The reboot was a recovery response to GPU faults, not the origin of the problem.

Previous-boot NVIDIA kernel logs show repeated GPU fault chains from `python3`:

- `Xid 13`: Graphics Engine Exception
- `Xid 31`: MMU Fault
- `Xid 43`: engine/channel failure
- `Xid 109`: Context Switch Timeout

The detailed fault strings repeatedly include:

- `MMU Fault`
- `MMU NACK Errors`
- `Out Of Range Address`
- `Invalid Address Space`

This is the pattern expected when generated kernels or their execution context trigger illegal
memory access or invalid GPU execution state and the fault escalates.

## Evidence

### Reward service outage

- `kernelgym-reward-39`
  - `Exited (255)`
  - `finished=2026-04-20T06:10:43Z`
- `kernelgym-reward-40`
  - `Exited (255)`
  - `finished=2026-04-20T06:10:42Z`
- both hosts:
  - `curl http://127.0.0.1:8111/health` -> `Connection refused`
  - nothing listening on `8110` or `8111`
- restart policy on both containers:
  - `restart=no`

### Host reboot trigger

- `auth.log` and `journalctl` show:
  - `root` SSH login from `192.168.120.100`
  - immediate transition into `Stopping Docker Application Container Engine`
  - `Finished Reboot`
- `last` shows those `root@192.168.120.100` sessions ending with the machines going `down`

### Real `Xid 109` incidents

On `.39`:

- `2026-04-17 21:45:07 +08:00`
  - `PCI 0000:d6:00`
  - `GPU 7`
  - `Xid 109`, `python3`
- `2026-04-18 07:02:57 +08:00`
  - `PCI 0000:d5:00`
  - `GPU 6`
  - `Xid 109`, `python3`

On `.40`:

- `2026-04-18 09:57:55 +08:00`
  - `PCI 0000:52:00`
  - `GPU 1`
  - `Xid 109`, `python3`
- `2026-04-19 08:36:42 +08:00`
  - `PCI 0000:57:00`
  - `GPU 3`
  - `Xid 109`, `python3`
- `2026-04-19 12:21:58 +08:00`
  - `PCI 0000:d6:00`
  - `GPU 7`
  - `Xid 109`, `python3`
- `2026-04-19 21:58:59 +08:00`
  - `PCI 0000:d5:00`
  - `GPU 6`
  - `Xid 109`, `python3`
- `2026-04-20 08:20:06 +08:00`
  - `PCI 0000:ce:00`
  - `GPU 4`
  - `Xid 109`, `python3`

### Why this does not look like a single bad GPU

- `Xid 109` rotated across several GPUs on both hosts
- surrounding `Xid 13/31` faults also appeared on many different PCI bus IDs
- the same signature recurs across multiple days in `kern.log`
- the crashing process stays consistent as `python3`

That pattern is more consistent with the reward workload provoking unsafe GPU execution than with
one permanently defective board.

## Why The Current Reward Environment Is Fragile

### 1. No automatic recovery after reboot

The reward containers do not auto-restart on host boot.

Effect:
- a host reboot turns into a sustained reward outage until an operator intervenes

### 2. No health-aware routing or drain

Fresh evals can continue trying to use reward while the service is unavailable.

Effect:
- infrastructure failure leaks upward as misleading model metrics

### 3. No Xid-driven quarantine

The system does not automatically remove a GPU or node from rotation when severe NVIDIA faults
start appearing.

Effect:
- the same unhealthy machine can keep receiving tasks until humans notice

### 4. Infra failures are not separated cleanly from model failures

When reward requests fail, the final aggregated checkpoint metrics can collapse to zeros.

Effect:
- contaminated eval output can be mistaken for poor checkpoint quality

### 5. No staged safety gate before full generated-kernel execution

Generated kernels reach full evaluation without an earlier low-risk canary ladder.

Effect:
- unsafe kernels hit the expensive and dangerous execution path too often

## Hardening Plan

### Immediate

1. Change reward containers to auto-restart.
   - use `--restart unless-stopped` or a systemd-managed wrapper
2. Add a reward health gate before training or eval.
   - require `39/40` API health and worker-pool health before trusting results
3. Treat reward infra failure as invalid evaluation, not zero score.
   - if reward is unreachable, mark the sample or step as infra-failed
4. Add host-level Xid monitoring.
   - alert on `Xid 13/31/43/109`
   - drain the node immediately on repeated faults

### Near-Term

1. Add per-node circuit breakers.
   - if a reward host starts returning `Connection refused`, `No route to host`, or repeated task
     failures, stop sending traffic there
2. Add post-fault GPU or node quarantine.
   - after severe Xid bursts, remove that GPU or host from service until reset and revalidation
3. Add worker-pool degradation detection.
   - if worker count falls or worker init starts failing, mark the node unhealthy
4. Add end-to-end reward request deadlines.
   - queue wait plus execution plus retries should share one bounded timeout envelope

### Longer-Term

1. Add a preflight safety ladder before full benchmark execution.
   - syntax validation
   - import/build validation
   - tiny-shape correctness canary
   - only then full benchmark/profiling
2. Isolate generated-kernel execution more aggressively.
   - prefer shorter-lived workers or stricter recycling after suspicious failures
3. Separate service orchestration from ad hoc container lifecycle.
   - move reward into a managed service with explicit health, restart, and drain semantics

## Operational Guidance

When `Xid 109` appears, do not treat it as an isolated final error only.

In these reward hosts, `Xid 109` is usually the late-stage symptom of a broader chain:

1. `Xid 13` / `31` illegal-address and MMU faults start appearing
2. the reward workload continues running on the unhealthy GPU or host
3. the fault chain escalates into `Xid 109`
4. operators reboot the host to recover service
5. reward stays down if the container is not configured to auto-restart

The practical lesson is:

- trigger quarantine on the earlier `13/31` burst pattern
- do not wait for `109` before removing the host from service

## Current Conclusion

The current reward environment is fragile for two reasons:

- **it is operationally brittle**
  - no automatic restart, no automatic drain, no fail-closed eval semantics
- **it is not robust to hostile or malformed generated kernels**
  - the reward workload can drive repeated GPU MMU / graphics faults across multiple GPUs

Any serious robustness work should start with:

1. auto-restart and health-gated routing
2. explicit invalidation of infra-failed evals
3. Xid-driven node quarantine
4. staged preflight before full generated-kernel execution
