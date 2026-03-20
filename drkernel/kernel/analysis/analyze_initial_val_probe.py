#!/usr/bin/env python3
"""Analyze fine-grained initial-validation probe timings from drkernel logs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
REPEATED_RE = re.compile(r"\[repeated (\d+)x across cluster\]")
TURN_MODEL_RE = re.compile(r"Turn \d+ \| Model time: ([0-9]+(?:\.[0-9]+)?)s")
TURN_ENV_RE = re.compile(r"Turn \d+ \| Env time: ([0-9]+(?:\.[0-9]+)?)s")

BOOL_FIELDS = (
    "correctness",
    "compilation",
    "success",
    "is_decoy_kernel",
)

NUMERIC_FIELDS = (
    "kg_reference_runtime_ms",
    "kg_kernel_runtime_ms",
    "num_custom_kernel",
    "num_total_kernels",
    "num_coverage",
    "custom_kernel_cuda_time_in_profiling_us",
    "total_kernel_run_time_in_profiling_us",
    "time_coverage",
    "kg_reference_load_original_src_s",
    "kg_reference_prepare_init_inputs_s",
    "kg_reference_build_model_s",
    "kg_reference_backend_compile_s",
    "kg_reference_perf_warmup_s",
    "kg_reference_perf_measure_wall_s",
    "kg_reference_perf_measure_cuda_event_s",
    "kg_reference_perf_total_s",
    "kg_reference_perf_num_trials",
    "kg_reference_perf_num_warmup",
    "kg_reference_total_s",
    "kg_kernel_load_original_src_s",
    "kg_kernel_prepare_init_inputs_s",
    "kg_kernel_build_reference_model_s",
    "kg_kernel_compile_and_load_s",
    "kg_kernel_build_custom_model_s",
    "kg_kernel_correctness_s",
    "kg_kernel_triton_detect_s",
    "kg_kernel_perf_warmup_s",
    "kg_kernel_perf_measure_wall_s",
    "kg_kernel_perf_measure_cuda_event_s",
    "kg_kernel_perf_profile_s",
    "kg_kernel_perf_total_s",
    "kg_kernel_perf_num_trials",
    "kg_kernel_perf_num_warmup",
    "kg_kernel_perf_num_profile_trials",
    "kg_kernel_total_s",
    "kg_processing_time_s",
    "kg_end_to_end_time_s",
    "kg_queue_wait_time_s",
    "wg_pool_idle_wait_s",
    "wg_pool_execute_s",
    "wg_pool_restart_s",
    "wg_pool_return_s",
    "wg_pool_total_s",
    "wg_pool_retry_count",
    "wg_run_toolkit_s",
    "wg_complete_task_s",
    "wg_total_s",
)


def _clean_line(line: str) -> str:
    return ANSI_RE.sub("", line.rstrip("\n"))


def _extract_weight(line: str) -> int:
    match = REPEATED_RE.search(line)
    if not match:
        return 1
    return int(match.group(1)) + 1


def _extract_bool(line: str, field: str) -> bool | None:
    match = re.search(rf"'{re.escape(field)}': (True|False)", line)
    if not match:
        return None
    return match.group(1) == "True"


def _extract_number(line: str, field: str) -> float | None:
    match = re.search(
        rf"'{re.escape(field)}': (-?(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?|\.\d+(?:e[+-]?\d+)?))",
        line,
        re.IGNORECASE,
    )
    if not match:
        return None
    return float(match.group(1))


def _extract_string(line: str, field: str) -> str | None:
    match = re.search(rf"'{re.escape(field)}': '([^']*)'", line)
    if not match:
        return None
    return match.group(1)


def _categorize(record: dict) -> str:
    error = (record.get("error") or "").lower()
    success = record.get("success")
    correctness = record.get("correctness")
    compilation = record.get("compilation")
    decoy = record.get("is_decoy_kernel")

    if "missing class modelnew" in error:
        return "preflight_missing_modelnew"
    if decoy:
        return "decoy_rejected"
    if success:
        return "success"
    if compilation is False:
        return "compile_failed"
    if compilation and correctness is False:
        return "correctness_failed"
    return "other_failed"


def _summarize(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    result = {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }
    if len(values) >= 2:
        result["std"] = statistics.stdev(values)
    if len(ordered) >= 2:
        result["p90"] = ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))]
    return result


def _init_bucket() -> dict:
    return {
        "count": 0,
        "metrics": defaultdict(list),
        "hardware": Counter(),
    }


def _update_bucket(bucket: dict, record: dict, weight: int) -> None:
    bucket["count"] += weight
    hardware = record.get("hardware")
    if hardware:
        bucket["hardware"][hardware] += weight
    for field in NUMERIC_FIELDS:
        value = record.get(field)
        if value is None:
            continue
        if value < 0 and field.endswith(("_runtime_ms", "_total_s", "_measure_wall_s", "_measure_cuda_event_s")):
            continue
        bucket["metrics"][field].extend([value] * weight)


def _finalize_bucket(bucket: dict) -> dict:
    return {
        "count": bucket["count"],
        "hardware": dict(bucket["hardware"]),
        "metrics": {
            field: _summarize(values)
            for field, values in sorted(bucket["metrics"].items())
            if values
        },
    }


def extract_turn_timings(log_path: Path) -> dict:
    model_times: list[float] = []
    env_times: list[float] = []

    for raw_line in log_path.read_text().splitlines():
        line = _clean_line(raw_line)
        model_match = TURN_MODEL_RE.search(line)
        if model_match:
            model_times.append(float(model_match.group(1)))
        env_match = TURN_ENV_RE.search(line)
        if env_match:
            env_times.append(float(env_match.group(1)))

    return {
        "model_time_s": _summarize(model_times),
        "env_time_s": _summarize(env_times),
    }


def _get_nested_mean(summary: dict, *keys: str) -> float | None:
    current = summary
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, dict):
        return current.get("mean")
    return None


def analyze_log(path: Path, max_weighted_results: int | None = None) -> dict:
    overall = _init_bucket()
    by_category = defaultdict(_init_bucket)
    hardware_counts = Counter()
    env_result_count = 0

    for raw_line in path.read_text().splitlines():
        line = _clean_line(raw_line)
        if "Env Result:" not in line:
            continue
        weight = _extract_weight(line)
        if max_weighted_results is not None and overall["count"] >= max_weighted_results:
            break
        if max_weighted_results is not None and overall["count"] + weight > max_weighted_results:
            weight = max_weighted_results - overall["count"]
        env_result_count += 1
        record = {field: _extract_bool(line, field) for field in BOOL_FIELDS}
        record["error"] = _extract_string(line, "error")
        record["hardware"] = _extract_string(line, "hardware") or _extract_string(line, "gpu_name")
        for field in NUMERIC_FIELDS:
            record[field] = _extract_number(line, field)

        category = _categorize(record)
        hardware = record.get("hardware")
        if hardware:
            hardware_counts[hardware] += weight
        _update_bucket(overall, record, weight)
        _update_bucket(by_category[category], record, weight)

    return {
        "log_path": str(path),
        "env_result_lines": env_result_count,
        "weighted_env_results": overall["count"],
        "hardware_counts": dict(hardware_counts),
        "turn_timings": extract_turn_timings(path),
        "categories": {
            category: _finalize_bucket(bucket)
            for category, bucket in sorted(by_category.items())
        },
        "overall": _finalize_bucket(overall),
    }


def _metric_mean(summary: dict, *path: str) -> float | None:
    current = summary
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, dict):
        return current.get("mean")
    return None


def _metric_std(summary: dict, *path: str) -> float | None:
    current = summary
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, dict):
        return current.get("std")
    return None


def _extract_completed_at(line: str) -> datetime | None:
    match = re.search(r"'completed_at': '([^']+)'", line)
    if not match:
        return None
    try:
        return datetime.fromisoformat(match.group(1))
    except ValueError:
        return None


def extract_timeline(
    log_path: Path,
    max_weighted_results: int | None = None,
) -> list[tuple[float, str, int]]:
    events: list[tuple[float, str, int]] = []
    first_completed_at: datetime | None = None
    total_weight = 0

    for raw_line in log_path.read_text().splitlines():
        line = _clean_line(raw_line)
        if "Env Result:" not in line:
            continue
        weight = _extract_weight(line)
        if max_weighted_results is not None and total_weight >= max_weighted_results:
            break
        if max_weighted_results is not None and total_weight + weight > max_weighted_results:
            weight = max_weighted_results - total_weight

        record = {field: _extract_bool(line, field) for field in BOOL_FIELDS}
        record["error"] = _extract_string(line, "error")
        category = _categorize(record)
        completed_at = _extract_completed_at(line)
        if completed_at is None:
            continue
        if first_completed_at is None:
            first_completed_at = completed_at
        elapsed_minutes = (completed_at - first_completed_at).total_seconds() / 60.0
        events.append((elapsed_minutes, category, weight))
        total_weight += weight

    return events


def summarize_timeline_by_category(
    timeline: list[tuple[float, str, int]],
    category_order: list[str],
) -> tuple[list[str], list[float], list[float]]:
    values_by_category: dict[str, list[float]] = {key: [] for key in category_order}
    for elapsed_minutes, category, weight in timeline:
        if category not in values_by_category:
            continue
        values_by_category[category].extend([elapsed_minutes] * weight)

    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    for key in category_order:
        values = values_by_category[key]
        if not values:
            continue
        labels.append(key)
        means.append(statistics.fmean(values))
        stds.append(statistics.stdev(values) if len(values) >= 2 else 0.0)
    return labels, means, stds


def write_summary(
    log_path: Path,
    summary_path: Path,
    max_weighted_results: int | None = None,
) -> dict:
    result = analyze_log(log_path, max_weighted_results=max_weighted_results)
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def write_summary_plot(
    summary: dict,
    log_path: Path,
    plot_path: Path,
    max_weighted_results: int | None = None,
) -> None:
    if plt is None:
        svg_path = plot_path if plot_path.suffix == ".svg" else plot_path.with_suffix(".svg")
        write_summary_plot_svg(summary, log_path, svg_path, max_weighted_results)
        return

    categories = summary.get("categories", {})
    timeline = extract_timeline(log_path, max_weighted_results=max_weighted_results)
    timeline_order = [
        "preflight_missing_modelnew",
        "compile_failed",
        "correctness_failed",
        "decoy_rejected",
        "success",
    ]
    stage_time_labels, stage_time_values, stage_time_errors = summarize_timeline_by_category(
        timeline,
        timeline_order,
    )

    decomposition_specs = [
        ("inference_model_time_s", _get_nested_mean(summary, "turn_timings", "model_time_s")),
        ("reward_env_time_s", _get_nested_mean(summary, "turn_timings", "env_time_s")),
        ("reward_worker_total_s", _metric_mean(summary, "overall", "metrics", "wg_pool_total_s")),
        ("reward_processing_s", _metric_mean(summary, "overall", "metrics", "kg_processing_time_s")),
        ("reward_queue_wait_s", _metric_mean(summary, "overall", "metrics", "kg_queue_wait_time_s")),
        ("reward_compile_and_load_s", _metric_mean(summary, "overall", "metrics", "kg_kernel_compile_and_load_s")),
        ("reward_end_to_end_s", _metric_mean(summary, "overall", "metrics", "kg_end_to_end_time_s")),
    ]
    decomposition_labels = [label for label, value in decomposition_specs if value is not None]
    decomposition_values = [value for _, value in decomposition_specs if value is not None]

    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    fig.suptitle(plot_path.stem, fontsize=12, y=0.995)

    if decomposition_labels:
        colors = [
            "#4C78A8" if label.startswith("inference") else "#E45756"
            for label in decomposition_labels
        ]
        axes[0].bar(decomposition_labels, decomposition_values, color=colors)
        axes[0].set_title("Timing Decomposition: Inference vs Reward")
        axes[0].set_ylabel("Seconds")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)
    else:
        axes[0].set_title("Timing Decomposition: Inference vs Reward")
        axes[0].text(0.5, 0.5, "No timing decomposition data", ha="center", va="center")
        axes[0].set_axis_off()

    if stage_time_labels:
        axes[1].bar(
            stage_time_labels,
            stage_time_values,
            yerr=stage_time_errors,
            capsize=5,
            color="#E45756",
            ecolor="#222222",
        )
        axes[1].set_title("Mean Wall-Clock Arrival Time by Outcome")
        axes[1].set_ylabel("Minutes")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
    else:
        axes[1].set_title("Mean Wall-Clock Arrival Time by Outcome")
        axes[1].text(0.5, 0.5, "No timeline data", ha="center", va="center")
        axes[1].set_axis_off()

    timeline_colors = {
        "preflight_missing_modelnew": "#4C78A8",
        "compile_failed": "#F58518",
        "correctness_failed": "#E45756",
        "decoy_rejected": "#72B7B2",
        "success": "#54A24B",
    }
    if timeline:
        cumulative = {key: 0 for key in timeline_order}
        xs: dict[str, list[float]] = {key: [] for key in timeline_order}
        ys: dict[str, list[int]] = {key: [] for key in timeline_order}
        for elapsed_minutes, category, weight in timeline:
            if category not in cumulative:
                continue
            cumulative[category] += weight
            for key in timeline_order:
                xs[key].append(elapsed_minutes)
                ys[key].append(cumulative[key])
        for key in timeline_order:
            if xs[key]:
                axes[2].plot(xs[key], ys[key], label=key, color=timeline_colors[key], linewidth=2)
        axes[2].set_title("Cumulative Outcomes vs Wall Clock")
        axes[2].set_xlabel("Elapsed minutes")
        axes[2].set_ylabel("Absolute count")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, linestyle="--", alpha=0.4)
    else:
        axes[2].set_title("Cumulative Outcomes vs Wall Clock")
        axes[2].text(0.5, 0.5, "No completed_at timeline", ha="center", va="center")
        axes[2].set_axis_off()

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_summary_plot_svg(
    summary: dict,
    log_path: Path,
    plot_path: Path,
    max_weighted_results: int | None = None,
) -> None:
    categories = summary.get("categories", {})
    timeline = extract_timeline(log_path, max_weighted_results=max_weighted_results)
    timeline_order = [
        "preflight_missing_modelnew",
        "compile_failed",
        "correctness_failed",
        "decoy_rejected",
        "success",
    ]
    stage_time_labels, stage_time_values, _ = summarize_timeline_by_category(
        timeline,
        timeline_order,
    )
    decomposition_specs = [
        ("inference_model_time_s", _get_nested_mean(summary, "turn_timings", "model_time_s")),
        ("reward_env_time_s", _get_nested_mean(summary, "turn_timings", "env_time_s")),
        ("reward_worker_total_s", _metric_mean(summary, "overall", "metrics", "wg_pool_total_s")),
        ("reward_processing_s", _metric_mean(summary, "overall", "metrics", "kg_processing_time_s")),
        ("reward_queue_wait_s", _metric_mean(summary, "overall", "metrics", "kg_queue_wait_time_s")),
        ("reward_compile_and_load_s", _metric_mean(summary, "overall", "metrics", "kg_kernel_compile_and_load_s")),
        ("reward_end_to_end_s", _metric_mean(summary, "overall", "metrics", "kg_end_to_end_time_s")),
    ]
    decomposition_labels = [label for label, value in decomposition_specs if value is not None]
    decomposition_values = [value for _, value in decomposition_specs if value is not None]

    timeline_colors = {
        "preflight_missing_modelnew": "#4C78A8",
        "compile_failed": "#F58518",
        "correctness_failed": "#E45756",
        "decoy_rejected": "#72B7B2",
        "success": "#54A24B",
    }

    width = 1200
    row_height = 270
    top_margin = 40
    bottom_margin = 30
    left_margin = 90
    right_margin = 40
    rows = 3
    height = top_margin + rows * row_height + bottom_margin
    plot_width = width - left_margin - right_margin
    row_gap = 30
    inner_height = row_height - row_gap

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>"
        "text{font-family:Arial,sans-serif;fill:#222}"
        ".title{font-size:20px;font-weight:bold}"
        ".subtitle{font-size:16px;font-weight:bold}"
        ".label{font-size:12px}"
        ".axis{stroke:#333;stroke-width:1.2}"
        ".grid{stroke:#ddd;stroke-width:1;stroke-dasharray:4 3}"
        "</style>",
        f'<text x="{width/2:.1f}" y="24" text-anchor="middle" class="title">{_svg_escape(plot_path.stem)}</text>',
    ]

    def draw_bar_row(idx: int, title: str, labels: list[str], values: list[float], colors: list[str], y_label: str) -> None:
        y0 = top_margin + idx * row_height
        parts.append(f'<text x="{left_margin}" y="{y0 + 18}" class="subtitle">{_svg_escape(title)}</text>')
        if not labels:
            parts.append(f'<text x="{width/2:.1f}" y="{y0 + inner_height/2:.1f}" text-anchor="middle">No data</text>')
            return
        ymax = max(values) if values else 1.0
        ymax = ymax * 1.15 if ymax > 0 else 1.0
        chart_top = y0 + 30
        chart_bottom = y0 + inner_height
        chart_height = chart_bottom - chart_top
        bar_w = plot_width / max(len(labels), 1) * 0.65
        step_w = plot_width / max(len(labels), 1)
        # grid
        for i in range(6):
            frac = i / 5
            y = chart_bottom - frac * chart_height
            val = ymax * frac
            parts.append(f'<line x1="{left_margin}" y1="{y:.1f}" x2="{width-right_margin}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{left_margin-10}" y="{y+4:.1f}" text-anchor="end" class="label">{val:.2f}</text>')
        parts.append(f'<line x1="{left_margin}" y1="{chart_bottom}" x2="{width-right_margin}" y2="{chart_bottom}" class="axis"/>')
        parts.append(f'<line x1="{left_margin}" y1="{chart_top}" x2="{left_margin}" y2="{chart_bottom}" class="axis"/>')
        parts.append(
            f'<text x="22" y="{(chart_top+chart_bottom)/2:.1f}" text-anchor="middle" class="label" '
            f'transform="rotate(-90 22 {(chart_top+chart_bottom)/2:.1f})">{_svg_escape(y_label)}</text>'
        )
        for i, (label, value, color) in enumerate(zip(labels, values, colors)):
            x = left_margin + i * step_w + (step_w - bar_w) / 2
            h = 0 if ymax <= 0 else (value / ymax) * chart_height
            y = chart_bottom - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="label">{value:.2f}</text>')
            parts.append(
                f'<text x="{x + bar_w/2:.1f}" y="{chart_bottom + 18:.1f}" text-anchor="middle" class="label">{_svg_escape(label)}</text>'
            )

    draw_bar_row(
        0,
        "Timing Decomposition: Inference vs Reward",
        decomposition_labels,
        decomposition_values,
        ["#4C78A8" if label.startswith("inference") else "#E45756" for label in decomposition_labels],
        "seconds",
    )
    draw_bar_row(
        1,
        "Mean Wall-Clock Arrival Time by Outcome",
        stage_time_labels,
        stage_time_values,
        ["#E45756"] * len(stage_time_labels),
        "minutes",
    )

    # Row 3: cumulative timeline
    y0 = top_margin + 2 * row_height
    parts.append(f'<text x="{left_margin}" y="{y0 + 18}" class="subtitle">Cumulative Outcomes vs Wall Clock</text>')
    chart_top = y0 + 30
    chart_bottom = y0 + inner_height
    chart_height = chart_bottom - chart_top
    if timeline:
        max_x = max(elapsed for elapsed, _, _ in timeline)
        max_y = summary.get("weighted_env_results", 0) or 1
        for i in range(6):
            frac = i / 5
            y = chart_bottom - frac * chart_height
            val = max_y * frac
            parts.append(f'<line x1="{left_margin}" y1="{y:.1f}" x2="{width-right_margin}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{left_margin-10}" y="{y+4:.1f}" text-anchor="end" class="label">{val:.0f}</text>')
        for i in range(6):
            frac = i / 5
            x = left_margin + frac * plot_width
            val = max_x * frac
            parts.append(f'<line x1="{x:.1f}" y1="{chart_top}" x2="{x:.1f}" y2="{chart_bottom}" class="grid"/>')
            parts.append(f'<text x="{x:.1f}" y="{chart_bottom + 18:.1f}" text-anchor="middle" class="label">{val:.1f}</text>')
        parts.append(f'<line x1="{left_margin}" y1="{chart_bottom}" x2="{width-right_margin}" y2="{chart_bottom}" class="axis"/>')
        parts.append(f'<line x1="{left_margin}" y1="{chart_top}" x2="{left_margin}" y2="{chart_bottom}" class="axis"/>')
        cumulative = {key: 0 for key in timeline_order}
        xs: dict[str, list[float]] = {key: [] for key in timeline_order}
        ys: dict[str, list[int]] = {key: [] for key in timeline_order}
        for elapsed_minutes, category, weight in timeline:
            if category not in cumulative:
                continue
            cumulative[category] += weight
            for key in timeline_order:
                xs[key].append(elapsed_minutes)
                ys[key].append(cumulative[key])
        for key in timeline_order:
            if not xs[key]:
                continue
            points = []
            for xval, yval in zip(xs[key], ys[key]):
                x = left_margin + (xval / max_x * plot_width if max_x > 0 else 0)
                y = chart_bottom - (yval / max_y * chart_height if max_y > 0 else 0)
                points.append(f"{x:.1f},{y:.1f}")
            parts.append(
                f'<polyline fill="none" stroke="{timeline_colors[key]}" stroke-width="2" points="{" ".join(points)}"/>'
            )
        legend_x = width - right_margin - 200
        legend_y = chart_top + 12
        for i, key in enumerate(timeline_order):
            ly = legend_y + i * 18
            parts.append(f'<line x1="{legend_x}" y1="{ly:.1f}" x2="{legend_x+16}" y2="{ly:.1f}" stroke="{timeline_colors[key]}" stroke-width="3"/>')
            parts.append(f'<text x="{legend_x+22}" y="{ly+4:.1f}" class="label">{_svg_escape(key)}</text>')
        parts.append(f'<text x="{width/2:.1f}" y="{chart_bottom + 36:.1f}" text-anchor="middle" class="label">Elapsed minutes</text>')
        parts.append(
            f'<text x="22" y="{(chart_top+chart_bottom)/2:.1f}" text-anchor="middle" class="label" '
            f'transform="rotate(-90 22 {(chart_top+chart_bottom)/2:.1f})">absolute count</text>'
        )
    else:
        parts.append(f'<text x="{width/2:.1f}" y="{y0 + inner_height/2:.1f}" text-anchor="middle">No completed_at timeline</text>')

    parts.append("</svg>")
    plot_path.write_text("".join(parts))


def batch_analyze_init_val_logs(
    log_dir: Path,
    max_weighted_results: int | None = None,
) -> int:
    if not log_dir.exists():
        print(f"No init_val log directory found: {log_dir}", file=sys.stderr)
        return 1

    log_paths = sorted(log_dir.rglob("*.log"))
    if not log_paths:
        print(f"No .log files found in: {log_dir}")
        return 0

    for log_path in log_paths:
        summary_path = log_path.with_suffix(".summary.json")
        plot_path = log_path.with_suffix(".summary.png" if plt is not None else ".summary.svg")
        if summary_path.exists() and plot_path.exists():
            print(
                f"Skipping {log_path.name}: summary already exists at "
                f"{summary_path.name} and {plot_path.name}"
            )
            continue
        print(
            f"Analyzing {log_path.name} -> "
            f"{summary_path.name}, {plot_path.name}"
        )
        result = write_summary(log_path, summary_path, max_weighted_results=max_weighted_results)
        write_summary_plot(result, log_path, plot_path, max_weighted_results=max_weighted_results)

    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze drkernel initial-validation probe logs."
    )
    parser.add_argument(
        "--max-weighted-results",
        type=int,
        default=None,
        help="Cap analysis at the first N weighted Env Result records.",
    )
    parser.add_argument(
        "log",
        nargs="?",
        help="Optional single log path. If omitted, batch-process drkernel/logs/init_val/**/*.log.",
    )
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.log is None:
        default_log_dir = Path(__file__).resolve().parents[2] / "logs" / "init_val"
        return batch_analyze_init_val_logs(
            default_log_dir,
            max_weighted_results=args.max_weighted_results,
        )

    path = Path(args.log)
    result = analyze_log(path, max_weighted_results=args.max_weighted_results)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
