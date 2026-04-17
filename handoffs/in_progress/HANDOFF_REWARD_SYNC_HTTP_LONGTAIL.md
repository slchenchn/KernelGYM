# Reward Sync HTTP Long-Tail Handoff

## Goal

Explain why recent steps in the active 8B 12xH20 run balloon from normal `20-22 min` to `50-60+ min`, and pin down whether the failure is in:
- reward kernel execution
- reward worker queueing
- token release
- or the `/evaluate` HTTP return path

## Fast Start

- active run:
  `drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700/`
- main rollout log:
  `drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700/main.log`
- reward client / engine log:
  `drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700/reward.log`
- structured long-tail heartbeats:
  `drkernel/logs/structured/batch_heartbeat.pid*.jsonl`
- client code:
  `drkernel/kernel/rewards/reward_client.py`
- reward server code:
  `kernelgym/server/api/server.py`
- reward worker pool code:
  `kernelgym/worker/subprocess_pool.py`
- token bucket:
  `drkernel/verl/verl/tools/sandbox_fusion_tools.py`

## Bottom Line

Plain-language conclusion:

`client 没有及时收到第一次 /evaluate 的 HTTP response，所以它一直占着 token 等。`

This is the most important thing to preserve when handing off:

- it is **not** “client already received the response but forgot to release the token”
- it is **not** “30s queue timeout”
- it is **not** primarily `old_log_prob`
- it is **not** primarily actor update

What is actually happening:

1. reward client sends a synchronous `POST /evaluate`
2. that request does not return to the client promptly
3. client keeps waiting
4. token stays occupied because token release happens only after `post()` returns
5. enough stuck requests starve the token bucket and then starve downstream reward requests
6. rollout gets stuck at tails like `8/12 servers, 22/32 rows` for tens of minutes

The only remaining uncertainty is **where** inside the HTTP return path it hangs:

- inside reward server response-return
- or in the return chain from server back to client, such as relay / long-lived HTTP connection handling

Current evidence is enough to say:

`问题在同步 /evaluate 的 response 回不来，而不是 kernel 还在跑，也不是 token release 逻辑漏掉了。`

## What Is Proven

### 1. Recent slow steps are rollout-side, not `old_log_prob`

`step 108` completed after `3117.5s` rollout, then `step 109` got stuck at `8/12` and `22/32` for over `1500s`:

- `main.log:5163` shows `step=108 ... elapsed=3117.5s`
- `main.log:5164` shows `Training Progress ... 108/2249000`
- `main.log:5188-5197` shows `step=109` stuck at `servers=8/12 prompt_rows=22/32`

This is reward-backed rollout tail latency, not `old_log_prob`.

### 2. `30s` is real kernel execution timeout, not queue timeout

`REWARD_TASK_TIMEOUT=30` is the server-side execution timeout. The worker-pool timeout path is explicit in:

- `kernelgym/worker/subprocess_pool.py:504-557`

The wording:

- `timeout after 30s. Not retried to avoid blocking worker queue.`

means:

- the task already hit execution timeout
- then the system chooses not to retry, to avoid further blocking the queue

It does **not** mean:

- “the task waited in queue for 30s”

Counterexample proving normal behavior exists:

- `reward.log:561059-561063` shows a task that truly timed out after `30s`
- the same second, `_HybridHttpWorker` logs `POST /evaluate resp=200`

So normal timeout handling can return promptly.

### 3. Token release happens only after synchronous `POST /evaluate` returns

In `drkernel/kernel/rewards/reward_client.py:77-99`:

- token is acquired first
- synchronous `self._client.post(.../evaluate...)` is executed
- token is released only after `post()` returns

That means:

- if token stays occupied for a long time, the most direct explanation is that `post()` has not returned yet
- this rules out “client got the response but forgot to release token” as the main theory

### 4. `/evaluate` is synchronous and reuses existing results for the same `task_id`

`kernelgym/server/api/server.py:337-370` is decisive:

- `_execute_workflow()` returns existing cached result immediately if `task_id` already has one
- otherwise it executes the workflow synchronously and then completes the task

This matters because a retried request with the same `task_id` can quickly return a cached result, even if the original synchronous request stayed hung for a long time.

### 5. There are long-lived pending refs with occupied tokens

Structured heartbeat evidence:

- `drkernel/logs/structured/batch_heartbeat.pid750009.jsonl:3801`
- `drkernel/logs/structured/batch_heartbeat.pid750009.jsonl:3840`

The same task `parallel_task_003529_580604ae` stays pending from about `60s` to `2401s+`.

Important fields in those heartbeats:

- `pending=1`
- `tokens_in_use` stays nonzero

This proves long-lived pending reward refs are holding shared client-side capacity.

### 6. Smoking gun: server-side result existed long before client saw `resp=200`

This is the hardest piece of evidence.

In `reward.log`:

- `561065` shows the client-visible failed result for `parallel_task_001413_29499634`
- that payload says `completed_at='2026-04-16T05:00:43.710603'`
- `561068` shows `_HybridHttpWorker POST /evaluate resp=200 task_id=parallel_task_001413_29499634`
- both of those client-visible lines happen at `2026-04-16T05:21:55+00:00`

So:

- server-side completion time was `05:00:43`
- client did not get a `200` response until `05:21:55`

That gap is about `21 min`.

This rules out:

- “kernel was still running the whole time”
- “client received the response promptly”

This strongly supports:

- the original synchronous `/evaluate` request did not return promptly to the client
- a later retry likely got the already-cached result for the same `task_id`

### 7. `Last error: None` is not a mysterious kernel exception

`kernelgym/worker/subprocess_pool.py:456-569` shows:

- if `_get_idle_worker()` returns `None`
- retry count increments
- `last_error` may remain unset
- final error becomes `Task failed after 2 retries. Last error: None`

So this message means:

- worker pool saturation / no idle worker across retries

It does **not** mean:

- a literal kernel exception named `None`

## Root Cause

### Primary Root Cause

The reward client and reward server are coupled through a synchronous `/evaluate` contract that is too fragile for this workload.

The pathological sequence is:

1. client acquires a token
2. client makes a synchronous `POST /evaluate`
3. server-side workflow may already finish or fail
4. but that response does not close back to the client promptly
5. client keeps the token occupied while waiting
6. many such stuck calls reduce effective token capacity and create long rollout tails

In plain language:

`真正坏的是第一次 /evaluate 的 response 回不来或者回得太晚。`

### Secondary Amplifier

Server-side reward worker pool saturation amplifies the problem after the first stuck wave begins.

Why:

- token starvation slows future submissions
- worker-pool retries then begin failing with `Last error: None`
- some requests then quickly get cached results on retry, while others stay hung

This creates the observed mixed symptom set:

- some tasks timeout and return normally at `30s`
- some tasks fail due to no idle worker
- some tasks appear to “finish” only tens of minutes later even though `completed_at` is much earlier

## What Is Ruled Out

- `old_log_prob` as the main cause of the `50-60+ min` steps
- “30s queue timeout”
- “client got response but forgot to release token”
- kernel execution still running for the entire long-tail interval
- reward relay/network instability as the primary repeated explanation for the whole pattern

Network or relay faults may still happen occasionally, but they do not explain the repeated multi-step pattern where:

- server-side completion exists
- client-visible `resp=200` arrives much later
- long-lived pending refs keep tokens occupied

## Why The Current Configuration Makes This Worse

The active run keeps very large client-side wait windows:

- `reward_model.acquire_timeout: 2400`
- `reward_model.max_retries: 3`
- `reward_model.task_timeout_in_client: 2400`
- `reward_model.timeout: 1800`

See `trainer.log` around:

- `13877-13981`
- `14500-14604`
- `16105-16209`
- `24658-24762`

This means:

- server-side kernel timeout is only `30s`
- but client-side HTTP / token / retry lifetime can extend into tens of minutes

That mismatch is large enough to turn a subset of bad requests into step-killing long tails.

## Verification Commands

### Confirm the smoking-gun case

```bash
run=drkernel/logs/trloo-8b-hfsdp6-pytorch-eager.train.12xH20.reward.16x4090.run.20260414-044700
nl -ba "$run/reward.log" | sed -n '561059,561068p'
```

Look for:

- a task whose payload `completed_at` is much earlier than the eventual `POST /evaluate resp=200`

### Confirm long-lived pending refs

```bash
nl -ba drkernel/logs/structured/batch_heartbeat.pid750009.jsonl | sed -n '3801,3840p'
```

Look for:

- same `task_id`
- `pending=1`
- `elapsed_s` climbing into thousands of seconds
- `tokens_in_use` staying nonzero

### Confirm token-release placement

```bash
nl -ba drkernel/kernel/rewards/reward_client.py | sed -n '77,99p'
```

### Confirm synchronous server contract and cached-result reuse

```bash
nl -ba kernelgym/server/api/server.py | sed -n '337,370p'
```

### Confirm `Last error: None` means no idle worker path

```bash
nl -ba kernelgym/worker/subprocess_pool.py | sed -n '456,569p'
```

## Recommended Next Fixes

### 1. Stop holding client token across the full synchronous `/evaluate` lifetime

Best fix:

- move to a true submit/poll contract

Minimum acceptable fix:

- do not keep the rate-limit token occupied while waiting for final synchronous completion

### 2. Separate submission timeout from final-result timeout

Current behavior conflates:

- request submission
- server execution
- final HTTP response delivery

The system needs separate controls for:

- connect timeout
- request/ack timeout
- result polling timeout

### 3. On timeout or retry, do not blindly reissue the same long synchronous `POST /evaluate`

Preferred behavior:

- first retry should check task status or results by `task_id`
- only resubmit if the task truly does not exist

### 4. Add explicit attempt-level logging around the synchronous HTTP call

Specifically log:

- attempt id
- request start timestamp
- timeout / connect / transport exception type
- request end timestamp
- whether the result came from cached `task_id`

Without this, the next investigation will again have to infer too much from side effects.

### 5. Replace `Last error: None` with a structured `NO_IDLE_WORKER` failure

Current string is misleading and wastes debugging time.

## What Not To Do First

- do not start by tuning `old_log_prob`
- do not assume `REWARD_TASK_TIMEOUT=30` is the main cause
- do not start by changing rollout oversampling
- do not treat this as a pure network problem without preserving the synchronous-response evidence

## Current Run State At Time Of Diagnosis

At the time this handoff was written:

- latest successful visible step: `108`
- current in-flight step at diagnosis boundary: `109`
- observed stall shape: `servers=8/12`, `prompt_rows=22/32`, `elapsed>1500s`

Relevant lines:

- `main.log:5164`
- `main.log:5166-5205`

## One-Sentence Handoff

`最近几个 step 变慢的根因，是 reward client 在同步 /evaluate 上长时间等不到 response，导致 token 长时间不释放，后续 reward 请求被饿住，rollout 被少数尾部请求拖到 50-60+ 分钟。`
