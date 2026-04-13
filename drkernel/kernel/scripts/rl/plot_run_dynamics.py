#!/usr/bin/env python3
"""
Plot training and checkpoint-eval dynamics from a run directory or log path.

Usage:
    python plot_run_dynamics.py /path/to/run_dir
    python plot_run_dynamics.py /path/to/run_dir/main.log
    python plot_run_dynamics.py /path/to/eval_results

Default behavior:
- If the input is a run directory, parse `main.log` into `training_dynamics_plots_by_group`
  and auto-discover `eval_results` subdirectories to generate eval plots under `plots/`.
- If the input is `main.log`, generate only the training plots unless eval discovery still finds
  `eval_results` under the run directory.
- If the input is an `eval_results` directory, generate only the eval plots.

Training metrics remain grouped by log namespace; `train/**` and `val/**` metrics are split into
  subdirectories to avoid oversized figures.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np


def parse_step_metrics(log_path: str) -> dict:
    """Parse step:N metrics from training log."""
    steps = {}
    step_pattern = re.compile(r"step:(\d+)\s+-\s+(.+)")

    with open(log_path, "r") as f:
        for line in f:
            # Strip ANSI codes and Ray prefixes
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            line = re.sub(r"^\(TaskRunner pid=\d+\)\s*", "", line)

            m = step_pattern.search(line)
            if not m:
                continue

            step_num = int(m.group(1))
            if step_num == 0:  # skip val-before-train
                continue

            metrics_str = m.group(2)
            metrics = {}
            for kv in re.finditer(r"(\S+):([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", metrics_str):
                key, val = kv.group(1), kv.group(2)
                try:
                    metrics[key] = float(val)
                except ValueError:
                    continue

            if metrics:
                # Low-batch retry lines also use the "step:N -" prefix, but they
                # only contain over_sampling hints and are not completed steps.
                if "actor/loss" not in metrics and "timing_s/step" not in metrics:
                    continue
                steps[step_num] = metrics

    return steps


def metric_group(key: str) -> str:
    """Return the image/group name for a metric key."""
    parts = key.split("/")
    if parts[0] in {"train", "val"}:
        if len(parts) >= 4:
            return "/".join(parts[:3])
        if len(parts) >= 2:
            return "/".join(parts[:2])
    return parts[0]


def metric_axis_title(key: str, group: str) -> str:
    """Return the subplot title for a metric key."""
    prefix = f"{group}/"
    if key.startswith(prefix):
        return key[len(prefix):]
    return key.split("/", 1)[1] if "/" in key else key


def sanitize_filename(name: str) -> str:
    """Create a stable filesystem-safe filename stem."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "metrics"


def group_output_dir(base_output_dir: Path, group: str) -> Path:
    """Return the destination directory for a metric group."""
    if group == "val" or group.startswith("val/"):
        return base_output_dir / "val"
    if group == "train" or group.startswith("train/"):
        return base_output_dir / "train"
    return base_output_dir


def group_output_filename(group: str) -> str:
    """Return the filename for a metric group, relative to its output directory."""
    parts = group.split("/")
    if parts[0] in {"train", "val"} and len(parts) > 1:
        stem = "_".join(parts[1:])
    else:
        stem = group
    return f"{sanitize_filename(stem)}.png"


def write_training_summary(steps: dict, step_nums: list[int], output_dir: Path):
    """Write a compact latest-step summary next to the plots."""
    last_step = step_nums[-1]
    last = steps[last_step]
    summary = f"""Training Dynamics Summary
========================
Steps completed: {last_step}
Total steps logged: {len(step_nums)}

Latest metrics (step {last_step}):
  actor/loss:          {last.get('actor/loss', 'N/A')}
  reward mean:         {last.get('critic/rewards/mean', 'N/A')}
  kl_divergence:       {last.get('actor/kl_divergence', 'N/A')}
  clip_fraction:       {last.get('actor/clip_fraction', 'N/A')}
  response_length:     {last.get('response_length/mean', 'N/A')}
  entropy:             {last.get('actor/avg_entropy', 'N/A')}

Kernel metrics (step {last_step}):
  turn 1 correct:      {last.get('train/kernel/turn_1/correctness_rate', 'N/A')}
  turn 2 correct:      {last.get('train/kernel/turn_2/correctness_rate', 'N/A')}
  turn 3 correct:      {last.get('train/kernel/turn_3/correctness_rate', 'N/A')}
  turn 1 compile:      {last.get('train/kernel/turn_1/compilation_rate', 'N/A')}
  speedup positive:    {last.get('critic/rewards_extra/is_speedup_positive/mean', 'N/A')}
  decoy rate:          {last.get('critic/rewards_extra/is_decoy_kernel/mean', 'N/A')}

Timing (step {last_step}):
  gen (rollout):       {last.get('timing_s/gen', 0)/60:.1f} min
  update_actor:        {last.get('timing_s/update_actor', 0)/60:.1f} min
  old_log_prob:        {last.get('timing_s/old_log_prob', 0)/60:.1f} min
  total step:          {last.get('timing_s/step', 0)/60:.1f} min

MRS / masking (step {last_step}):
  mrs token masked:    {last.get('mismatch/rollout_rs_masked_fraction', 'N/A')}
  mrs seq masked:      {last.get('mismatch/rollout_rs_seq_masked_fraction', 'N/A')}
  correct mask rate:   {last.get('mismatch_quality/correct/mask_rate', 'N/A')}
  incorrect mask rate: {last.get('mismatch_quality/incorrect/mask_rate', 'N/A')}
  surviving correct:   {last.get('mismatch_quality/non_masked/correct_count', 'N/A')}
  coverage correct RS: {last.get('coverage/coverage_rs_correct_only_masked_fraction', 'N/A')}
"""
    summary_path = output_dir / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(summary)
    print("Saved training_summary.txt")


def plot_grouped_metrics(steps: dict, step_nums: list[int], output_dir: Path, get_series_aligned, apply_integer_step_axis):
    """Plot one image per top-level log namespace."""
    metric_keys = sorted({key for metrics in steps.values() for key in metrics})
    grouped_keys: dict[str, list[str]] = {}
    for key in metric_keys:
        grouped_keys.setdefault(metric_group(key), []).append(key)

    for group, keys in sorted(grouped_keys.items()):
        n_metrics = len(keys)
        ncols = min(3, n_metrics)
        nrows = math.ceil(n_metrics / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, max(3.2 * nrows, 4)),
            squeeze=False,
        )
        fig.suptitle(group, fontsize=14, fontweight="bold")

        flat_axes = axes.ravel()
        for ax, key in zip(flat_axes, keys):
            xs, ys = get_series_aligned(key)
            if ys:
                ax.plot(xs, ys, alpha=0.8, linewidth=1)
            ax.set_title(metric_axis_title(key, group), fontsize=9)
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            apply_integer_step_axis(ax)
            ax.grid(True, alpha=0.3)

        for ax in flat_axes[n_metrics:]:
            ax.axis("off")

        plt.tight_layout(rect=(0, 0, 1, 0.98))
        filename = group_output_filename(group)
        destination_dir = group_output_dir(output_dir, group)
        destination_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination_dir / filename, dpi=150)
        plt.close(fig)
        print(f"Saved {destination_dir / filename}")


def plot_training_dynamics(steps: dict, output_dir: str, include_legacy_curated: bool = False):
    """Generate training dynamics plots."""
    if not steps:
        print("No training steps found in log.")
        return

    step_nums = sorted(steps.keys())

    def get_series(key):
        vals = [steps[s].get(key) for s in step_nums]
        return [v for v in vals if v is not None]

    def get_series_aligned(key):
        xs, ys = [], []
        for s in step_nums:
            v = steps[s].get(key)
            if v is not None:
                xs.append(s)
                ys.append(v)
        return xs, ys

    def apply_integer_step_axis(ax):
        """Training steps are discrete; keep x-axis ticks integral."""
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_grouped_metrics(steps, step_nums, output_dir, get_series_aligned, apply_integer_step_axis)

    if not include_legacy_curated:
        write_training_summary(steps, step_nums, output_dir)
        return

    # =========================================================================
    # Figure 1: Core Training Metrics (2x3)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Dynamics — Core Metrics", fontsize=14, fontweight="bold")

    # 1. Actor Loss
    ax = axes[0, 0]
    xs, ys = get_series_aligned("actor/loss")
    if ys:
        ax.plot(xs, ys, "b-", alpha=0.7, linewidth=0.8)
        # Smoothed
        if len(ys) > 5:
            window = min(5, len(ys) // 3)
            smoothed = np.convolve(ys, np.ones(window) / window, mode="valid")
            ax.plot(xs[window - 1:], smoothed, "b-", linewidth=2, label=f"smooth(w={window})")
            ax.legend(fontsize=8)
    ax.set_title("Actor Loss")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.grid(True, alpha=0.3)

    # 2. Reward Mean
    ax = axes[0, 1]
    xs, ys = get_series_aligned("critic/rewards/mean")
    if ys:
        ax.plot(xs, ys, "g-", alpha=0.7, linewidth=0.8)
        if len(ys) > 5:
            window = min(5, len(ys) // 3)
            smoothed = np.convolve(ys, np.ones(window) / window, mode="valid")
            ax.plot(xs[window - 1:], smoothed, "g-", linewidth=2)
    ax.set_title("Reward Mean")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.grid(True, alpha=0.3)

    # 3. KL Divergence
    ax = axes[0, 2]
    xs, ys = get_series_aligned("actor/kl_divergence")
    if ys:
        ax.plot(xs, ys, "r-", alpha=0.7, linewidth=0.8)
    ax.set_title("KL Divergence")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.grid(True, alpha=0.3)

    # 4. Clip Fraction
    ax = axes[1, 0]
    for key, color, label in [
        ("actor/clip_fraction", "purple", "total"),
        ("actor/clip_fraction_lower", "blue", "lower"),
        ("actor/clip_fraction_upper", "red", "upper"),
    ]:
        xs, ys = get_series_aligned(key)
        if ys:
            ax.plot(xs, ys, color=color, alpha=0.7, linewidth=0.8, label=label)
    ax.set_title("Clip Fraction")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Response Length
    ax = axes[1, 1]
    xs, ys = get_series_aligned("response_length/mean")
    if ys:
        ax.plot(xs, ys, "orange", alpha=0.7, linewidth=1)
    ax.set_title("Response Length (mean)")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.grid(True, alpha=0.3)

    # 6. Entropy
    ax = axes[1, 2]
    xs, ys = get_series_aligned("actor/avg_entropy")
    if ys:
        ax.plot(xs, ys, "teal", alpha=0.7, linewidth=1)
    ax.set_title("Actor Entropy")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "01_core_metrics.png", dpi=150)
    plt.close(fig)
    print(f"Saved 01_core_metrics.png")

    # =========================================================================
    # Figure 2: Kernel-Specific Metrics (2x3)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Dynamics — Kernel Metrics", fontsize=14, fontweight="bold")

    # 1. Per-turn correctness rate
    ax = axes[0, 0]
    for turn in [1, 2, 3]:
        xs, ys = get_series_aligned(f"train/kernel/turn_{turn}/correctness_rate")
        if ys:
            ax.plot(xs, ys, alpha=0.7, linewidth=0.8, label=f"turn {turn}")
    ax.set_title("Correctness Rate (per turn)")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Per-turn compilation rate
    ax = axes[0, 1]
    for turn in [1, 2, 3]:
        xs, ys = get_series_aligned(f"train/kernel/turn_{turn}/compilation_rate")
        if ys:
            ax.plot(xs, ys, alpha=0.7, linewidth=0.8, label=f"turn {turn}")
    ax.set_title("Compilation Rate (per turn)")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Speedup positive rate
    ax = axes[0, 2]
    xs, ys = get_series_aligned("critic/rewards_extra/is_speedup_positive/mean")
    if ys:
        ax.plot(xs, ys, "green", alpha=0.7, linewidth=0.8)
    ax.set_title("Speedup Positive Rate")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.grid(True, alpha=0.3)

    # 4. Performance (mean in all)
    ax = axes[1, 0]
    for turn in [1, 2, 3]:
        xs, ys = get_series_aligned(f"train/kernel/turn_{turn}/mean_performance_in_all")
        if ys:
            ax.plot(xs, ys, alpha=0.7, linewidth=0.8, label=f"turn {turn}")
    ax.set_title("Mean Performance (in all, per turn)")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Decoy kernel rate
    ax = axes[1, 1]
    xs, ys = get_series_aligned("critic/rewards_extra/is_decoy_kernel/mean")
    if ys:
        ax.plot(xs, ys, "red", alpha=0.7, linewidth=0.8)
    ax.set_title("Decoy Kernel Rate")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.grid(True, alpha=0.3)

    # 6. Coverage
    ax = axes[1, 2]
    xs, ys = get_series_aligned("coverage/coverage_rs_mean_coverage")
    if ys:
        ax.plot(xs, ys, "purple", alpha=0.7, linewidth=0.8)
    ax.set_title("Mean Coverage")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "02_kernel_metrics.png", dpi=150)
    plt.close(fig)
    print(f"Saved 02_kernel_metrics.png")

    # =========================================================================
    # Figure 3: Timing & Efficiency (1x3)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Dynamics — Timing & Efficiency", fontsize=14, fontweight="bold")

    # 1. Step timing breakdown
    ax = axes[0]
    for key, label, color in [
        ("timing_s/gen", "rollout", "blue"),
        ("timing_s/update_actor", "update_actor", "red"),
        ("timing_s/old_log_prob", "ref_logprob", "green"),
        ("timing_s/step", "total", "black"),
    ]:
        xs, ys = get_series_aligned(key)
        if ys:
            ys_min = [v / 60 for v in ys]  # convert to minutes
            ax.plot(xs, ys_min, color=color, alpha=0.7, linewidth=1, label=label)
    ax.set_title("Step Timing (minutes)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Minutes")
    apply_integer_step_axis(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Batch size after filtering
    ax = axes[1]
    xs, ys = get_series_aligned("batch/effective_batch_size")
    if ys:
        ax.plot(xs, ys, "orange", alpha=0.7, linewidth=1)
    ax.set_title("Effective Batch Size")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.grid(True, alpha=0.3)

    # 3. Mismatch KL (policy drift)
    ax = axes[2]
    xs, ys = get_series_aligned("mismatch/mismatch_kl")
    if ys:
        ax.plot(xs, ys, "red", alpha=0.7, linewidth=1)
    ax.set_title("Mismatch KL (policy drift)")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "03_timing_efficiency.png", dpi=150)
    plt.close(fig)
    print(f"Saved 03_timing_efficiency.png")

    # =========================================================================
    # Figure 4: Mismatch Rejection Sampling Metrics (2x3)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Dynamics — MRS Metrics", fontsize=14, fontweight="bold")

    # 1. Token / sequence masking from rollout RS
    ax = axes[0, 0]
    for key, color, label in [
        ("mismatch/rollout_rs_masked_fraction", "red", "token masked"),
        ("mismatch/rollout_rs_seq_masked_fraction", "orange", "seq masked"),
    ]:
        xs, ys = get_series_aligned(key)
        if ys:
            ax.plot(xs, ys, color=color, alpha=0.75, linewidth=1, label=label)
    ax.set_title("MRS Masked Fraction")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction")
    apply_integer_step_axis(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Final mask rate by correctness after MRS + coverage RS
    ax = axes[0, 1]
    for key, color, label in [
        ("mismatch_quality/correct/mask_rate", "red", "correct"),
        ("mismatch_quality/incorrect/mask_rate", "gray", "incorrect"),
        ("mismatch_quality/overall_mask_rate", "black", "overall"),
    ]:
        xs, ys = get_series_aligned(key)
        if ys:
            ax.plot(xs, ys, color=color, alpha=0.75, linewidth=1, label=label)
    ax.set_title("Final Mask Rate by Quality")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction")
    apply_integer_step_axis(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Correct samples that survive all masking
    ax = axes[0, 2]
    xs, ys = get_series_aligned("mismatch_quality/non_masked/correct_count")
    if ys:
        ax.plot(xs, ys, "green", alpha=0.75, linewidth=1)
    ax.set_title("Surviving Correct Samples")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    apply_integer_step_axis(ax)
    ax.grid(True, alpha=0.3)

    # 4. Coverage RS pressure on correct samples, shown alongside MRS
    ax = axes[1, 0]
    for key, color, label in [
        ("coverage/coverage_rs_correct_only_masked_fraction", "purple", "correct masked by coverage"),
        ("coverage/coverage_rs_masked_fraction", "blue", "overall coverage masked"),
    ]:
        xs, ys = get_series_aligned(key)
        if ys:
            ax.plot(xs, ys, color=color, alpha=0.75, linewidth=1, label=label)
    ax.set_title("Coverage RS Masking")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction")
    apply_integer_step_axis(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Mismatch diagnostics
    ax = axes[1, 1]
    for key, color, label in [
        ("mismatch/mismatch_kl", "red", "KL"),
        ("mismatch/mismatch_log_ppl_abs_diff", "blue", "abs log-ppl diff"),
    ]:
        xs, ys = get_series_aligned(key)
        if ys:
            ax.plot(xs, ys, color=color, alpha=0.75, linewidth=1, label=label)
    ax.set_title("Mismatch Diagnostics")
    ax.set_xlabel("Step")
    apply_integer_step_axis(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Batch selection pressure
    ax = axes[1, 2]
    for key, color, label in [
        ("batch/complete_groups_before_selection", "blue", "complete groups before"),
        ("batch/complete_groups_after_selection", "green", "complete groups after"),
        ("batch/low_variance_groups", "orange", "low variance groups"),
    ]:
        xs, ys = get_series_aligned(key)
        if ys:
            ax.plot(xs, ys, color=color, alpha=0.75, linewidth=1, label=label)
    ax.set_title("Batch Selection Pressure")
    ax.set_xlabel("Step")
    ax.set_ylabel("Groups")
    apply_integer_step_axis(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "04_mrs_metrics.png", dpi=150)
    plt.close(fig)
    print(f"Saved 04_mrs_metrics.png")

    # =========================================================================
    # Figure 5: Validation metrics (if available)
    # =========================================================================
    # Check if val metrics exist
    val_keys = [k for k in steps[step_nums[0]].keys() if k.startswith("val/")]
    if val_keys:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Validation Metrics", fontsize=14, fontweight="bold")

        ax = axes[0]
        for key in ["val/test_score/kernelbench_level2_validation_pass@1",
                     "val/test_score/kernelbench_level2_validation_pass@8"]:
            xs, ys = get_series_aligned(key)
            if ys:
                label = "pass@1" if "pass@1" in key else "pass@8"
                ax.plot(xs, ys, alpha=0.7, linewidth=1.5, label=label, marker="o", markersize=3)
        ax.set_title("Validation pass@k")
        ax.set_xlabel("Step")
        apply_integer_step_axis(ax)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        xs, ys = get_series_aligned("val/test_score/kernelbench_level2_validation")
        if ys:
            ax.plot(xs, ys, "green", alpha=0.7, linewidth=1.5, marker="o", markersize=3)
        ax.set_title("Validation Score")
        ax.set_xlabel("Step")
        apply_integer_step_axis(ax)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        xs, ys = get_series_aligned("val/test_score_extra/correctness_kernelbench_level2_validation")
        if ys:
            ax.plot(xs, ys, "blue", alpha=0.7, linewidth=1.5, marker="o", markersize=3)
        ax.set_title("Validation Correctness")
        ax.set_xlabel("Step")
        apply_integer_step_axis(ax)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / "05_validation.png", dpi=150)
        plt.close(fig)
        print(f"Saved 05_validation.png")

    # =========================================================================
    # Summary text
    # =========================================================================
    last_step = step_nums[-1]
    last = steps[last_step]
    summary = f"""Training Dynamics Summary
========================
Steps completed: {last_step}
Total steps logged: {len(step_nums)}

Latest metrics (step {last_step}):
  actor/loss:          {last.get('actor/loss', 'N/A')}
  reward mean:         {last.get('critic/rewards/mean', 'N/A')}
  kl_divergence:       {last.get('actor/kl_divergence', 'N/A')}
  clip_fraction:       {last.get('actor/clip_fraction', 'N/A')}
  response_length:     {last.get('response_length/mean', 'N/A')}
  entropy:             {last.get('actor/avg_entropy', 'N/A')}

Kernel metrics (step {last_step}):
  turn 1 correct:      {last.get('train/kernel/turn_1/correctness_rate', 'N/A')}
  turn 2 correct:      {last.get('train/kernel/turn_2/correctness_rate', 'N/A')}
  turn 3 correct:      {last.get('train/kernel/turn_3/correctness_rate', 'N/A')}
  turn 1 compile:      {last.get('train/kernel/turn_1/compilation_rate', 'N/A')}
  speedup positive:    {last.get('critic/rewards_extra/is_speedup_positive/mean', 'N/A')}
  decoy rate:          {last.get('critic/rewards_extra/is_decoy_kernel/mean', 'N/A')}

Timing (step {last_step}):
  gen (rollout):       {last.get('timing_s/gen', 0)/60:.1f} min
  update_actor:        {last.get('timing_s/update_actor', 0)/60:.1f} min
  old_log_prob:        {last.get('timing_s/old_log_prob', 0)/60:.1f} min
  total step:          {last.get('timing_s/step', 0)/60:.1f} min

MRS / masking (step {last_step}):
  mrs token masked:    {last.get('mismatch/rollout_rs_masked_fraction', 'N/A')}
  mrs seq masked:      {last.get('mismatch/rollout_rs_seq_masked_fraction', 'N/A')}
  correct mask rate:   {last.get('mismatch_quality/correct/mask_rate', 'N/A')}
  incorrect mask rate: {last.get('mismatch_quality/incorrect/mask_rate', 'N/A')}
  surviving correct:   {last.get('mismatch_quality/non_masked/correct_count', 'N/A')}
  coverage correct RS: {last.get('coverage/coverage_rs_correct_only_masked_fraction', 'N/A')}
"""
    summary_path = output_dir / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(summary)
    print(f"Saved training_summary.txt")




def discover_eval_steps(results_dir: Path) -> list[int]:
    steps = []
    for path in results_dir.glob("step_*"):
        if not path.is_dir():
            continue
        m = re.fullmatch(r"step_(\d+)", path.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def load_eval_metrics(results_dir: Path, requested_steps: list[int] | None) -> list[tuple[int, dict]]:
    steps = requested_steps if requested_steps else discover_eval_steps(results_dir)
    rows = []
    for step in steps:
        metrics_path = results_dir / f"step_{step}" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                rows.append((step, json.load(f)))
    return rows


def maybe_plot_eval(ax, xs, ys, title, ylabel, color, marker):
    valid = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if valid:
        vx = [x for x, _ in valid]
        vy = [y for _, y in valid]
        ax.plot(vx, vy, color=color, marker=marker, markersize=6, linewidth=2)
        best_x, best_y = max(valid, key=lambda pair: pair[1])
        ax.annotate(
            f"{best_y:.4f}\n(step {best_x})",
            xy=(best_x, best_y),
            xytext=(best_x, best_y + max(0.01, 0.05 * max(vy))),
            arrowprops=dict(arrowstyle="->", color=color),
            fontsize=9,
            color=color,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)


def save_eval_dynamics(output_dir: Path, rows: list[tuple[int, dict]]):
    steps = [step for step, _ in rows]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Checkpoint Eval Dynamics", fontsize=14, fontweight="bold")

    maybe_plot_eval(
        axes[0, 0],
        steps,
        [d.get("val/test_score/kernelbench_level2_validation_pass@1") for _, d in rows],
        "pass@1",
        "Rate",
        "tab:blue",
        "o",
    )
    maybe_plot_eval(
        axes[0, 1],
        steps,
        [d.get("val/kernel/best_by_turn_3/correctness_rate") for _, d in rows],
        "best_by_turn_3 correctness_rate",
        "Rate",
        "tab:green",
        "s",
    )
    maybe_plot_eval(
        axes[1, 0],
        steps,
        [d.get("val/kernel/best_by_turn_3/fast@1_in_all") for _, d in rows],
        "best_by_turn_3 fast@1_in_all",
        "Rate",
        "tab:orange",
        "^",
    )
    maybe_plot_eval(
        axes[1, 1],
        steps,
        [d.get("val/kernel/best_by_turn_3/fast@1.2_in_all") for _, d in rows],
        "best_by_turn_3 fast@1.2_in_all",
        "Rate",
        "tab:red",
        "D",
    )

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = output_dir / "eval_dynamics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_fast_at_k_curve(output_dir: Path, rows: list[tuple[int, dict]], ref_fast1: float | None, ref_fast12: float | None):
    steps = [step for step, _ in rows]
    fast1_vals = [d.get("val/kernel/turn_3/fast@1_in_all") for _, d in rows]
    fast12_vals = [d.get("val/kernel/turn_3/fast@1.2_in_all") for _, d in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Kernel Speedup Across Training Steps (turn_3, level 2)", fontsize=14, fontweight="bold")

    valid_f1 = [(s, v) for s, v in zip(steps, fast1_vals) if v is not None]
    valid_f12 = [(s, v) for s, v in zip(steps, fast12_vals) if v is not None]

    if valid_f1:
        ax1.plot([x[0] for x in valid_f1], [x[1] for x in valid_f1], "g-o", markersize=6, linewidth=2, label="our training")
        best_s, best_v = max(valid_f1, key=lambda x: x[1])
        ax1.annotate(
            f"{best_v:.4f}\n(step {best_s})",
            xy=(best_s, best_v),
            xytext=(best_s, best_v + 0.02),
            arrowprops=dict(arrowstyle="->", color="green"),
            fontsize=9,
            color="green",
        )
    if ref_fast1 is not None:
        ax1.axhline(y=ref_fast1, color="green", linestyle="--", alpha=0.5, label=f"drkernel-14b RL ({ref_fast1:.4f})")
    ax1.set_title("fast@1_in_all")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Rate")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    if valid_f12:
        ax2.plot([x[0] for x in valid_f12], [x[1] for x in valid_f12], "r-s", markersize=6, linewidth=2, label="our training")
        best_s, best_v = max(valid_f12, key=lambda x: x[1])
        ax2.annotate(
            f"{best_v:.4f}\n(step {best_s})",
            xy=(best_s, best_v),
            xytext=(best_s, best_v + 0.02),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=9,
            color="red",
        )
    if ref_fast12 is not None:
        ax2.axhline(y=ref_fast12, color="red", linestyle="--", alpha=0.5, label=f"drkernel-14b RL ({ref_fast12:.4f})")
    ax2.set_title("fast@1.2_in_all")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Rate")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = output_dir / "fast_at_k_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_eval_results(results_dir: Path, output_dir: Path | None, steps: list[int] | None, ref_fast1: float | None, ref_fast12: float | None) -> bool:
    rows = load_eval_metrics(results_dir, steps)
    if not rows:
        print(f"No metrics.json found under {results_dir}; skipping eval plots")
        return False
    actual_output_dir = output_dir or results_dir / "plots"
    actual_output_dir.mkdir(parents=True, exist_ok=True)
    save_eval_dynamics(actual_output_dir, rows)
    save_fast_at_k_curve(actual_output_dir, rows, ref_fast1, ref_fast12)
    return True


def discover_eval_results_dirs(run_dir: Path, max_depth: int = 4) -> list[Path]:
    canonical = run_dir / 'eval_results'
    if canonical.is_dir():
        return [canonical]

    results = []
    seen = set()
    skip_dirs = {
        "__pycache__",
        "plots",
        "structured",
        "torchinductor_cache",
        "training_dynamics",
        "training_dynamics_plots",
        "training_dynamics_plots_by_group",
    }

    for root, dirnames, _filenames in __import__("os").walk(run_dir, topdown=True):
        root_path = Path(root)
        depth = len(root_path.relative_to(run_dir).parts)

        next_dirnames = []
        for dirname in dirnames:
            if dirname in skip_dirs:
                continue
            if dirname.startswith("global_step_"):
                continue
            if dirname == "eval_results":
                candidate = root_path / dirname
                resolved = candidate.resolve()
                if resolved not in seen:
                    results.append(candidate)
                    seen.add(resolved)
                continue
            next_dirnames.append(dirname)

        dirnames[:] = next_dirnames
        if depth >= max_depth:
            dirnames[:] = []

    return sorted(results)


def infer_inputs(input_path: Path) -> tuple[Path | None, Path | None, list[Path]]:
    if input_path.is_file():
        if input_path.name != "main.log":
            raise SystemExit(f"Unsupported file input: {input_path}. Expected main.log")
        run_dir = input_path.parent
        return input_path, run_dir, discover_eval_results_dirs(run_dir)

    if not input_path.is_dir():
        raise SystemExit(f"Input path does not exist: {input_path}")

    if re.fullmatch(r"eval_results", input_path.name) or any(input_path.glob("step_*/metrics.json")):
        return None, None, [input_path]

    log_path = input_path / "main.log"
    eval_dirs = discover_eval_results_dirs(input_path)
    return (log_path if log_path.exists() else None), input_path, eval_dirs


def main():
    parser = argparse.ArgumentParser(description="Plot training and eval dynamics from a run directory or log path")
    parser.add_argument("input_path", help="Run directory, main.log, or eval_results directory")
    parser.add_argument("--output-dir", help="Training plot output directory (default: run_dir/training_dynamics_plots_by_group)")
    parser.add_argument("--eval-output-dir", help="Optional shared eval plot output directory; defaults to each eval_results/plots")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval_results discovery and plotting")
    parser.add_argument("--eval-steps", nargs="+", type=int, help="Optional explicit checkpoint-eval step list")
    parser.add_argument(
        "--legacy-curated",
        action="store_true",
        help="Also emit the older curated training multi-metric figures.",
    )
    parser.add_argument("--ref-fast1", type=float, default=0.4038, help="drkernel-14b (RL) turn_3 fast@1 reference")
    parser.add_argument("--ref-fast12", type=float, default=0.2400, help="drkernel-14b (RL) turn_3 fast@1.2 reference")
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    log_path, run_dir, eval_dirs = infer_inputs(input_path)

    plotted_training = False
    plotted_eval = False

    if log_path is not None:
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else log_path.parent / "training_dynamics_plots_by_group"
        print(f"Parsing training log {log_path}...")
        steps = parse_step_metrics(str(log_path))
        print(f"Found {len(steps)} training steps")
        plot_training_dynamics(steps, str(output_dir), include_legacy_curated=args.legacy_curated)
        print(f"Training plots saved to {output_dir}")
        plotted_training = True
    elif run_dir is not None:
        print(f"No main.log found under {run_dir}; skipping training plots")

    if not args.skip_eval:
        if not eval_dirs and run_dir is not None:
            print(f"No eval_results directories found under {run_dir}")
        for results_dir in eval_dirs:
            target_dir = Path(args.eval_output_dir).expanduser().resolve() if args.eval_output_dir else None
            print(f"Parsing eval results {results_dir}...")
            plotted_eval |= plot_eval_results(results_dir, target_dir, args.eval_steps, args.ref_fast1, args.ref_fast12)

    if not plotted_training and not plotted_eval:
        raise SystemExit(f"No training or eval plots were generated from {input_path}")


if __name__ == "__main__":
    main()
