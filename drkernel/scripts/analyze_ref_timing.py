#!/usr/bin/env python3
"""Analyze reference timing variance from Redis results."""
import json, numpy as np, sys

# Connect to Redis on the KernelGYM API host
import redis
r = redis.Redis(host="192.168.16.18", port=8110, decode_responses=True)

# Get all ref results from today with elapsed_times
keys = r.keys("kernelgym_remote16_sglang_20260320:result:*_ref")
ref_data = []
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
    if times and len(times) > 10:
        ref_data.append({
            "task": key.split(":")[-1],
            "runtime": d.get("reference_runtime"),
            "times": times,
            "std": meta.get("kg_reference_perf_std_ms"),
            "mean": meta.get("kg_reference_perf_mean_ms"),
            "completed": completed,
        })

print(f"Samples with per-trial data: {len(ref_data)}")
if not ref_data:
    print("No data found. Exiting.")
    sys.exit()
print(f"Trials per sample: {len(ref_data[0]['times'])}")

means = [np.mean(d["times"]) for d in ref_data]
stds = [np.std(d["times"]) for d in ref_data]
cvs = [np.std(d["times"])/np.mean(d["times"])*100 for d in ref_data]
ranges_pct = [(np.max(d["times"])-np.min(d["times"]))/np.mean(d["times"])*100 for d in ref_data]

print()
print("=== Per-sample ref timing stats (ms) ===")
print(f"Mean runtime:  min={np.min(means):.3f}  med={np.median(means):.3f}  max={np.max(means):.3f}")
print(f"Std:           min={np.min(stds):.4f}  med={np.median(stds):.4f}  max={np.max(stds):.4f}")
print(f"CV%:           min={np.min(cvs):.2f}  med={np.median(cvs):.2f}  max={np.max(cvs):.2f}")
print(f"Range%:        min={np.min(ranges_pct):.1f}  med={np.median(ranges_pct):.1f}  max={np.max(ranges_pct):.1f}")

print()
print("=== CV% distribution ===")
for thr in [1, 2, 5, 10, 20]:
    n = sum(1 for c in cvs if c <= thr)
    print(f"  CV <= {thr}%: {n}/{len(cvs)} ({n/len(cvs)*100:.0f}%)")

print()
print("=== First trial warmup residual ===")
diffs = [(d["times"][0]-np.mean(d["times"][1:]))/np.mean(d["times"][1:])*100 for d in ref_data]
print(f"Overhead: med={np.median(diffs):.1f}%  mean={np.mean(diffs):.1f}%  max={np.max(diffs):.1f}%")

print()
print("=== Trial count reduction stability ===")
d30 = []; d10 = []; d5 = []
for d in ref_data:
    t = d["times"]
    m100 = np.mean(t); m30 = np.mean(t[:30]); m10 = np.mean(t[:10]); m5 = np.mean(t[:5])
    d30.append(abs(m30 - m100) / m100 * 100)
    d10.append(abs(m10 - m100) / m100 * 100)
    d5.append(abs(m5 - m100) / m100 * 100)
print(f"30 vs 100: med={np.median(d30):.2f}%  mean={np.mean(d30):.2f}%  max={np.max(d30):.2f}%")
print(f"10 vs 100: med={np.median(d10):.2f}%  mean={np.mean(d10):.2f}%  max={np.max(d10):.2f}%")
print(f" 5 vs 100: med={np.median(d5):.2f}%  mean={np.mean(d5):.2f}%  max={np.max(d5):.2f}%")

print()
print("=== Binary decision risk (ref mean drift > 1%) ===")
n30 = sum(1 for d in d30 if d > 1.0)
n10 = sum(1 for d in d10 if d > 1.0)
n5 = sum(1 for d in d5 if d > 1.0)
print(f"30 trials: {n30}/{len(d30)} ({n30/len(d30)*100:.1f}%) have >1% drift")
print(f"10 trials: {n10}/{len(d10)} ({n10/len(d10)*100:.1f}%) have >1% drift")
print(f" 5 trials: {n5}/{len(d5)} ({n5/len(d5)*100:.1f}%) have >1% drift")

print()
print("=== 5 example samples ===")
indices = np.linspace(0, len(ref_data)-1, 5, dtype=int)
for i in indices:
    d = ref_data[i]
    t = d["times"]
    cv = np.std(t)/np.mean(t)*100
    print(f"#{i} ({d['task'][:30]}): mean={np.mean(t):.3f}ms  std={np.std(t):.4f}ms  CV={cv:.1f}%")
    print(f"  first5: {[round(x,3) for x in t[:5]]}")
    print(f"  last5:  {[round(x,3) for x in t[-5:]]}")
