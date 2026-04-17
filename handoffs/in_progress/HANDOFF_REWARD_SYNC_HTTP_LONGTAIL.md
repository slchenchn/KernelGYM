# Reward Sync HTTP Long-Tail Bug Report

## Summary

This report covers one bug family with two layers:

1. a structural reward-path bug:
   the training side uses a synchronous `POST /evaluate` contract, and client-side rate-limit tokens
   are only released after that HTTP call returns
2. a concrete A800 incident trigger:
   reward host `192.168.16.40` entered a host-level NVIDIA/UVM bad state, reward worker capacity
   collapsed, and the synchronous `/evaluate` contract amplified that failure into `20-50+ min`
   rollout tails

The key correction from the later A800 incident is:

- relay is **not** required for this symptom family
- the same long-tail pattern reproduced on same-LAN training and reward nodes
- in that incident, the deeper root cause was reward host failure, not relay instability

## Impact

User-visible impact:

- recent training steps inflate from normal range into `50-60+ min`
- rollout stalls at partial progress for long periods
- reward requests accumulate behind a small number of long-lived tails
- training can continue with degraded or suspect reward behavior unless operators intervene

Training impact:

- this is not primarily an `old_log_prob` slowdown
- this is not primarily actor update slowdown
- the dominant impact is rollout-side tail latency
- if reward capacity is partially degraded, the training segment may become operationally or
  statistically suspect even when some steps still complete

## Affected Runs

### H20 investigation run

- active run during initial diagnosis:
  `drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700/`
- main log:
  `drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700/main.log`
- reward log:
  `drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700/reward.log`

### A800 reproduction and recovery run

- affected run:
  `drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/`
- main log:
  `drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/main.log`
- reward log:
  `drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/reward.log`
- timing summary:
  `drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/training_dynamics_plots_by_group/training_summary.txt`

## Final Diagnosis

### Structural bug

The reward client holds scarce client-side capacity across the full lifetime of a synchronous
`POST /evaluate` call. When that call does not return promptly, client-side tokens remain occupied
and later reward requests are starved behind a few stuck tails.

Plain-language version:

`client 没有及时收到第一次 /evaluate 的 HTTP response，所以它一直占着 token 等。`

### Incident-specific root cause on A800

The A800 incident was not caused primarily by relay behavior. The deeper root cause was:

- reward host `192.168.16.40` entered a host-level NVIDIA/UVM bad state
- reward subprocess workers stopped initializing reliably
- reward pool capacity collapsed on that host
- the synchronous `/evaluate` contract then exposed that degraded capacity as long-lived tails,
  token starvation, and `Task failed after 2 retries. Last error: None`

Plain-language version:

`A800 这次不是 relay 卡住，而是 reward-40 宿主机先坏了；同步 /evaluate 只是把这个底层故障放大成了训练侧长尾。`

## Scope Boundaries

This report does **not** support these explanations as the primary root cause:

- `old_log_prob` as the main reason recent steps ballooned
- actor update as the main reason recent steps ballooned
- “30s queue timeout”
- “client got the response but forgot to release token”
- relay as a necessary explanation for every instance of this symptom family

Relay or transport can still be an occasional contributing factor, but the A800 reproduction proves
that the same symptom family can happen with no relay dependency.

## Evidence

### H20 Evidence: synchronous `/evaluate` response path is too fragile

### 1. The slowdown is rollout-side

The initial H20 investigation showed rollout tails, not `old_log_prob` tails:

- `main.log:5163` shows `step=108 ... elapsed=3117.5s`
- `main.log:5164` shows visible completion of `step 108`
- `main.log:5188-5197` shows `step 109` stalling at `servers=8/12` and `prompt_rows=22/32`

This is rollout-side latency backed by reward behavior.

### 2. `REWARD_TASK_TIMEOUT=30` is execution timeout, not queue wait timeout

The timeout path in:

- `kernelgym/worker/subprocess_pool.py:504-557`

shows that:

- `timeout after 30s` refers to task execution timeout inside the worker path
- it does not mean the task waited in the queue for `30s`

Counterexample:

- `reward.log:561059-561063` shows a task that really times out at `30s`
- the same second, `_HybridHttpWorker` still logs `POST /evaluate resp=200`

So the system is capable of timing out and returning promptly in the normal path.

### 3. Token release happens only after synchronous `POST /evaluate` returns

In:

- `drkernel/kernel/rewards/reward_client.py:77-99`

the flow is:

1. acquire token
2. do synchronous `self._client.post(.../evaluate...)`
3. release token after the HTTP call returns

Therefore, long-lived token occupancy strongly implies the client is still waiting on the HTTP call.

### 4. The server contract is synchronous and cached-result reuse is possible

In:

- `kernelgym/server/api/server.py:337-370`

the server:

- returns an existing cached result immediately if the same `task_id` already has one
- otherwise executes synchronously before returning

This explains why a later retry can quickly return an already-finished result while the original
synchronous request may have stayed open for a long time.

### 5. Structured heartbeats show long-lived pending refs with occupied tokens

Evidence:

- `drkernel/logs/structured/batch_heartbeat.pid750009.jsonl:3801`
- `drkernel/logs/structured/batch_heartbeat.pid750009.jsonl:3840`

These show the same task staying pending while `tokens_in_use` remains nonzero.

### 6. Smoking-gun H20 case: server-side completion existed long before client saw `resp=200`

Evidence in `reward.log`:

- `561065` contains a payload for `parallel_task_001413_29499634`
- that payload says `completed_at='2026-04-16T05:00:43.710603'`
- `561068` later shows `_HybridHttpWorker POST /evaluate resp=200 task_id=parallel_task_001413_29499634`
- those client-visible lines happen at `2026-04-16T05:21:55+00:00`

This creates an approximately `21 min` gap between server-side completion and client-visible
successful response.

That rules out:

- the kernel still running for the full long-tail interval
- prompt client-side receipt of the first response

### A800 Evidence: relay is not necessary, reward host failure can be the deeper cause

### 1. Same-LAN A800 topology still showed the same tail pattern

On the A800 14B run, training and reward were on the same LAN, but reward log still showed
extremely long `POST /evaluate -> resp=200` lifetimes, for example:

- `reward.log:689533` to `reward.log:693193`
- `reward.log:689597` to `reward.log:693213`
- `reward.log:719781` to `reward.log:719783`

Those intervals correspond to multi-minute to multi-dozen-minute waits on the reward side.

### 2. Some of those long HTTP waits far exceeded the actual toolkit execution time

One representative case ended with a payload where toolkit runtime was only about `8.4s`, but the
client-visible HTTP return came much later:

- `reward.log:693209`

So the shape remains consistent with the structural bug above:

- the HTTP lifecycle can be much longer than the useful execution time

### 3. A800 training timing confirms rollout inflation

The run summary for the affected 14B run shows late-step timing inflation:

- `training_summary.txt` reports latest visible completed `step 170`
- its timing section shows:
  - `gen (rollout): 14.8 min`
  - `total step: 28.0 min`

That is the same family of rollout inflation diagnosed in the earlier H20 case.

### 4. The deeper failure boundary on A800 was reward host `16.40`

During live diagnosis, the decisive signals were:

- host-side `nvidia-smi -L` on `16.40` became unhealthy
- reward worker child processes piled up and stopped initializing reliably
- worker logs on `16.40` shifted from successful init into repeated worker-init failure behavior
- the system later surfaced `Task failed after 2 retries. Last error: None`

Relevant code path for the misleading final error string:

- `kernelgym/worker/subprocess_pool.py:456-569`

This error means:

- no idle worker across retries

It does **not** mean:

- a literal kernel exception named `None`

### 5. After physical reboot, the failure mode changed from GPU/UVM hang to recoverable infra faults

The recovery sequence proved the deeper A800 incident was host-related:

- after reboot, `16.40` host-side GPU enumeration worked again
- the old `kernelgym-reward-40` container then degraded into `Dead` /
  `Removal In Progress` Docker state instead of GPU-driver hang
- after clearing Docker metadata, the next blocker was a missing `/nfs/FM` mount on `16.40`
- once `/nfs/FM` was remounted, the canonical reward startup succeeded

That transition is exactly what a host-level incident looks like after reboot:

- the deepest hardware/driver hang is gone
- secondary infra cleanup is still needed before the service becomes healthy

## Root Cause Statement

This bug family should be recorded with two layers:

### Product / design root cause

The reward path uses a synchronous HTTP contract that is too fragile for long-running or partially
degraded reward execution. Client-side rate-limit tokens remain occupied until the synchronous
request returns, so any delay in the response path directly converts into downstream reward
starvation and rollout long tails.

### Incident root cause for the A800 outage

Reward host `192.168.16.40` suffered a host-level NVIDIA/UVM failure, which reduced or destroyed
effective reward worker capacity. The existing synchronous `/evaluate` design then amplified that
capacity loss into training-side long tails.

## Why This Became So Expensive

The active configurations tolerate very long client-side wait windows compared with the
`30s` server-side execution timeout. For example, the run config includes large values such as:

- `reward_model.acquire_timeout: 2400`
- `reward_model.max_retries: 3`
- `reward_model.task_timeout_in_client: 2400`
- `reward_model.timeout: 1800`

This mismatch allows a small number of bad requests to live much longer than the nominal kernel
execution timeout and to poison the effective throughput of the whole step.

## Recovery Performed For The A800 Incident

The incident was recovered with this sequence:

1. stop the live 14B run
2. delete `global_step_160` and above, leaving resume state at `global_step_150`
3. reboot reward hosts `16.39` and `16.40`
4. on `16.40`, clear dead Docker metadata for `kernelgym-reward-40`
5. remount `eds.intellif:/FM` to `/nfs/FM` on `16.40`
6. rerun canonical reward startup:
   `drkernel/kernel/scripts/rl/start_reward.sh -f`
7. recreate the training containers on `192.168.16.50/51`
8. reinstall `uv`, rerun `set_uv_python.sh`, and verify the shared
   `.venv-vllm0180`
9. relaunch the run on `50/51` with IB through the canonical launcher and no validation:
   - `VAL_BEFORE_TRAIN=False`
   - `TEST_FREQ=0`

## Current State After Recovery

As of the latest recovery and relaunch:

- reward health endpoint is back to `healthy`:
  `http://192.168.16.39:8111/health`
- reward worker logs on both `39/40` show fresh registration and heartbeat
- the 14B run is active again on `192.168.16.50/51`
- `ray status` reports:
  - `2` active nodes
  - `16.0/16.0 GPU` used/reserved
- the relaunch has advanced past launcher startup into distributed model initialization and
  checkpoint shard loading

## Verification Commands

### H20: confirm the smoking-gun delayed response case

```bash
run=drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700
nl -ba "$run/reward.log" | sed -n '561059,561068p'
```

### H20: confirm long-lived pending refs with occupied tokens

```bash
nl -ba drkernel/logs/structured/batch_heartbeat.pid750009.jsonl | sed -n '3801,3840p'
```

### Code: confirm token release placement

```bash
nl -ba drkernel/kernel/rewards/reward_client.py | sed -n '77,99p'
```

### Code: confirm synchronous server contract and cached-result reuse

```bash
nl -ba kernelgym/server/api/server.py | sed -n '337,370p'
```

### Code: confirm `Last error: None` means no-idle-worker path

```bash
nl -ba kernelgym/worker/subprocess_pool.py | sed -n '456,569p'
```

### A800: confirm late-step timing inflation

```bash
sed -n '1,120p' \
  drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519/training_dynamics_plots_by_group/training_summary.txt
```

### A800: confirm representative long-lived reward responses

```bash
run=drkernel/logs/trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260409-092519
nl -ba "$run/reward.log" | sed -n '689533,693222p'
nl -ba "$run/reward.log" | sed -n '719781,719786p'
```

## Recommended Fixes

### 1. Separate design fix from incident response

Do not treat every long-tail event as purely network or purely reward-worker. The next incident
response should explicitly split:

- reward host health
- reward worker capacity
- synchronous HTTP response behavior

### 2. Replace the synchronous submit-and-wait contract

Best fix:

- move to a true submit/poll contract

Minimum acceptable fix:

- do not hold scarce client-side rate-limit tokens across the full synchronous `/evaluate`
  lifetime

### 3. Split timeout domains

The system needs separate controls for:

- connect timeout
- request submission / ack timeout
- task execution timeout
- result polling timeout

### 4. Add fail-fast host health gates

Before blaming relay or training topology, gate reward hosts with:

- `nvidia-smi -L`
- a minimal CUDA init
- full worker-pool startup
- one real `/evaluate` warmup

Any failure should remove that host from service immediately.

### 5. Improve error semantics

Replace the misleading final error:

- `Task failed after 2 retries. Last error: None`

with an explicit structured failure such as:

- `NO_IDLE_WORKER`
- `WORKER_INIT_TIMEOUT`
- `HOST_HEALTH_FAILED`

### 6. Make recovery runbooks explicit

For this reward stack, the operational runbook should explicitly include:

- host reboot as a valid recovery step for NVIDIA/UVM incidents
- Docker dead-container cleanup
- `/nfs/FM` remount verification after host reboot

## One-Sentence Handoff

`同步 /evaluate 长尾是这个 bug family 的表层机制，但 A800 已证明更深的 root cause 也可能是 reward 节点宿主机故障；出现同类长尾时，先查 reward host 健康，再查 HTTP/relay return path。`
