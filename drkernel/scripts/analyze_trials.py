#!/usr/bin/env python3
"""Compare trial count + trimming configurations using existing data.
Falls back to Redis if available, otherwise tries vllm.log.
Can be run from any node with NFS access."""
import json, numpy as np, sys

# Try Redis first
data = []
try:
    import redis
    for host in ["192.168.16.39", "192.168.16.18", "localhost"]:
        try:
            r = redis.Redis(host=host, port=8110, decode_responses=True, socket_timeout=3)
            r.ping()
            keys = r.keys("kernelgym_remote16_sglang_20260320:result:*_ref")
            for key in keys:
                completed = r.hget(key, "completed_at")
                if not completed or "2026-03-30" not in completed:
                    continue
                result_raw = r.hget(key, "result")
                if not result_raw:
                    continue
                d = json.loads(result_raw)
                meta = d.get("metadata", {})
                times = meta.get("kg_reference_perf_elapsed_times_ms")
                if times and len(times) >= 50:
                    warmup = meta.get("kg_reference_perf_num_warmup", 3)
                    data.append({"times": times, "warmup": warmup})
            if data:
                print(f"Loaded {len(data)} samples from Redis ({host})")
                break
        except Exception:
            continue
except ImportError:
    pass

if not data:
    print("Redis unavailable, no data loaded")
    sys.exit(1)

# Split by warmup
w3 = [d["times"] for d in data if d["warmup"] < 10]
w10 = [d["times"] for d in data if d["warmup"] >= 10]

def gt(t):
    """Ground truth: 100 trials, trimmed drop 5+5."""
    return np.mean(sorted(t)[5:-5])

def analyze_config(times_list, n_trials, trim_n):
    cvs = []
    drifts = []
    for t in times_list:
        subset = t[:n_trials]
        if trim_n > 0 and len(subset) > 2 * trim_n:
            trimmed = sorted(subset)[trim_n:-trim_n]
        else:
            trimmed = subset
        m = np.mean(trimmed)
        s = np.std(trimmed)
        cv = s / m * 100 if m > 0 else 0
        cvs.append(cv)
        drift = abs(m - gt(t)) / gt(t) * 100
        drifts.append(drift)
    return cvs, drifts

configs = [
    ("100 trials, raw",            100, 0),
    ("100 trials, trim 5+5",       100, 5),
    ("50 trials, raw",              50, 0),
    ("50 trials, trim 3+3",         50, 3),
    ("50 trials, trim 5+5",         50, 5),
    ("30 trials, raw",              30, 0),
    ("30 trials, trim 3+3",         30, 3),
]

for label, dataset in [("warmup=3", w3), ("warmup=10", w10)]:
    if not dataset:
        continue
    print(f"\n=== {label} (n={len(dataset)}) ===")
    print(f"{'Config':<28} {'Med CV':>8} {'Mean CV':>9} {'CV<=2%':>7} {'CV<=5%':>7} {'CV<=10%':>8} {'Drift med':>10} {'Drift>1%':>9}")
    print("-" * 95)
    for name, n_trials, trim_n in configs:
        cvs, drifts = analyze_config(dataset, n_trials, trim_n)
        n = len(cvs)
        le2 = sum(1 for c in cvs if c <= 2)
        le5 = sum(1 for c in cvs if c <= 5)
        le10 = sum(1 for c in cvs if c <= 10)
        dgt1 = sum(1 for d in drifts if d > 1)
        print(f"{name:<28} {np.median(cvs):>7.2f}% {np.mean(cvs):>8.2f}% {le2/n*100:>6.0f}% {le5/n*100:>6.0f}% {le10/n*100:>7.0f}% {np.median(drifts):>9.2f}% {dgt1/n*100:>8.1f}%")
