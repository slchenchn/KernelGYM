#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\b")
TURN_MODEL_RE = re.compile(r"Turn\s+(\d+)\s+\|\s+Model time:\s+([0-9.]+)s")
TURN_ENV_RE = re.compile(r"Turn\s+(\d+)\s+\|\s+Env time:\s+([0-9.]+)s")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"nan", "+nan", "-nan"}:
        return math.nan
    if lowered in {"inf", "+inf"}:
        return math.inf
    if lowered == "-inf":
        return -math.inf
    try:
        return float(value)
    except ValueError:
        return None


def rolling_mean(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(1, window)
    out: list[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        out.append(running / min(idx + 1, window))
    return out


def parse_log(log_path: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    step_rows: list[dict[str, float]] = []
    turn_rows: list[dict[str, float]] = []

    with log_path.open("r", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = strip_ansi(raw_line).strip()
            if not line:
                continue

            if "timing_s/" in line and "step:" in line:
                match = STEP_RE.search(line)
                if match:
                    row: dict[str, float] = {"step": float(match.group(1)), "line_no": float(line_no)}
                    for segment in line.split(" - "):
                        if ":" not in segment:
                            continue
                        key, value = segment.split(":", 1)
                        key = key.strip()
                        if key == "step" or not key.startswith("timing_s/"):
                            continue
                        parsed = parse_float(value)
                        if parsed is not None:
                            row[key] = parsed
                    if any(key.startswith("timing_s/") for key in row):
                        step_rows.append(row)

            model_match = TURN_MODEL_RE.search(line)
            if model_match:
                turn_rows.append(
                    {
                        "line_no": float(line_no),
                        "turn": float(model_match.group(1)),
                        "phase": "rollout",
                        "seconds": float(model_match.group(2)),
                    }
                )
                continue

            env_match = TURN_ENV_RE.search(line)
            if env_match:
                turn_rows.append(
                    {
                        "line_no": float(line_no),
                        "turn": float(env_match.group(1)),
                        "phase": "reward",
                        "seconds": float(env_match.group(2)),
                    }
                )

    return step_rows, turn_rows


def write_step_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    timing_keys = sorted({key for row in rows for key in row if key.startswith("timing_s/")})
    fieldnames = ["step", "line_no", *timing_keys]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_turn_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    fieldnames = ["line_no", "turn", "phase", "seconds"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(step_rows: list[dict[str, float]], turn_rows: list[dict[str, float]], output_path: Path) -> None:
    summary_rows: list[dict[str, object]] = []

    for key in sorted({key for row in step_rows for key in row if key.startswith("timing_s/")}):
        values = [row[key] for row in step_rows if key in row and math.isfinite(row[key])]
        if not values:
            continue
        summary_rows.append(
            {
                "series": key,
                "count": len(values),
                "mean_s": sum(values) / len(values),
                "min_s": min(values),
                "max_s": max(values),
            }
        )

    by_phase: dict[str, list[float]] = defaultdict(list)
    for row in turn_rows:
        seconds = float(row["seconds"])
        if math.isfinite(seconds):
            by_phase[str(row["phase"])].append(seconds)

    for phase in sorted(by_phase):
        values = by_phase[phase]
        summary_rows.append(
            {
                "series": f"turn/{phase}",
                "count": len(values),
                "mean_s": sum(values) / len(values),
                "min_s": min(values),
                "max_s": max(values),
            }
        )

    fieldnames = ["series", "count", "mean_s", "min_s", "max_s"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError:
        return None


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _finite_series(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]


def _write_line_svg(
    series: list[tuple[str, str, list[tuple[float, float]]]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    width = 1100
    height = 700
    left = 90
    right = 30
    top = 60
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    finite_points = [point for _, _, pts in series for point in _finite_series(pts)]
    if not finite_points:
        return

    xs = [x for x, _ in finite_points]
    ys = [y for _, y in finite_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 1.0
    y_pad = (max_y - min_y) * 0.08
    min_y -= y_pad
    max_y += y_pad

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-family="sans-serif">{_svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
    ]

    for tick in range(6):
        ratio = tick / 5
        y_value = min_y + (max_y - min_y) * ratio
        y_pos = sy(y_value)
        parts.append(f'<line x1="{left}" y1="{y_pos}" x2="{left + plot_w}" y2="{y_pos}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left - 10}" y="{y_pos + 4}" text-anchor="end" font-size="12" font-family="sans-serif">{y_value:.1f}</text>')

    for tick in range(6):
        ratio = tick / 5
        x_value = min_x + (max_x - min_x) * ratio
        x_pos = sx(x_value)
        parts.append(f'<line x1="{x_pos}" y1="{top}" x2="{x_pos}" y2="{top + plot_h}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{x_pos}" y="{top + plot_h + 22}" text-anchor="middle" font-size="12" font-family="sans-serif">{x_value:.0f}</text>')

    for idx, (label, color, points) in enumerate(series):
        cleaned = _finite_series(points)
        if not cleaned:
            continue
        polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in cleaned)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline}"/>')
        legend_y = top + 20 + idx * 20
        parts.append(f'<line x1="{left + 10}" y1="{legend_y}" x2="{left + 30}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left + 38}" y="{legend_y + 4}" font-size="12" font-family="sans-serif">{_svg_escape(label)}</text>')

    parts.extend(
        [
            f'<text x="{width/2}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="sans-serif">{_svg_escape(x_label)}</text>',
            f'<text x="20" y="{height/2}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90,20,{height/2})">{_svg_escape(y_label)}</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(parts))


def _write_grouped_bar_svg(
    group_labels: list[str],
    bars: list[tuple[str, str, list[float]]],
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    width = 1100
    height = 700
    left = 90
    right = 30
    top = 60
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_values = [value for _, _, values in bars for value in values if math.isfinite(value)]
    if not all_values:
        return

    max_y = max(all_values)
    if max_y <= 0:
        max_y = 1.0
    max_y *= 1.1

    group_count = max(1, len(group_labels))
    series_count = max(1, len(bars))
    group_width = plot_w / group_count
    bar_width = group_width / (series_count + 1)

    def sy(y: float) -> float:
        return top + plot_h - y / max_y * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-family="sans-serif">{_svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
    ]

    for tick in range(6):
        value = max_y * tick / 5
        y_pos = sy(value)
        parts.append(f'<line x1="{left}" y1="{y_pos}" x2="{left + plot_w}" y2="{y_pos}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left - 10}" y="{y_pos + 4}" text-anchor="end" font-size="12" font-family="sans-serif">{value:.1f}</text>')

    for idx, group_label in enumerate(group_labels):
        group_x = left + idx * group_width
        center_x = group_x + group_width / 2
        parts.append(f'<text x="{center_x}" y="{top + plot_h + 22}" text-anchor="middle" font-size="12" font-family="sans-serif">{_svg_escape(group_label)}</text>')
        for bar_idx, (_, color, values) in enumerate(bars):
            if idx >= len(values):
                continue
            value = values[idx]
            bar_x = group_x + (bar_idx + 0.5) * bar_width
            bar_y = sy(value)
            bar_h = top + plot_h - bar_y
            parts.append(
                f'<rect x="{bar_x}" y="{bar_y}" width="{bar_width * 0.8}" height="{bar_h}" fill="{color}" opacity="0.85"/>'
            )

    for idx, (label, color, _) in enumerate(bars):
        legend_y = top + 20 + idx * 20
        parts.append(f'<rect x="{left + 10}" y="{legend_y - 9}" width="18" height="10" fill="{color}"/>')
        parts.append(f'<text x="{left + 38}" y="{legend_y}" font-size="12" font-family="sans-serif">{_svg_escape(label)}</text>')

    parts.extend(
        [
            f'<text x="20" y="{height/2}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90,20,{height/2})">{_svg_escape(y_label)}</text>',
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(parts))


def plot_step_timings(rows: list[dict[str, float]], output_path: Path) -> list[Path]:
    if not rows:
        return []

    preferred_keys = [
        "timing_s/gen",
        "timing_s/old_log_prob",
        "timing_s/update_actor",
        "timing_s/step",
    ]
    available_keys = [key for key in preferred_keys if any(key in row for row in rows)]
    if not available_keys:
        return []

    steps = [int(row["step"]) for row in rows]
    plt = try_import_matplotlib()
    if plt is not None and output_path.suffix.lower() == ".png":
        plt.figure(figsize=(10, 6))
        for key in available_keys:
            ys = [row.get(key, math.nan) for row in rows]
            plt.plot(steps, ys, marker="o", linewidth=1.8, label=key)

        plt.xlabel("Training Step")
        plt.ylabel("Seconds")
        plt.title("Step-Level Timing From Training Log")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return [output_path]

    series = []
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for idx, key in enumerate(available_keys):
        ys = [row.get(key, math.nan) for row in rows]
        series.append((key, colors[idx % len(colors)], list(zip(steps, ys))))
    _write_line_svg(series, output_path, "Step-Level Timing From Training Log", "Training Step", "Seconds")
    return [output_path]


def plot_turn_timings(rows: list[dict[str, float]], output_path: Path, window: int) -> list[Path]:
    if not rows:
        return []

    by_phase: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_phase[str(row["phase"])].append(row)
    grouped: dict[str, dict[int, list[float]]] = {
        "rollout": defaultdict(list),
        "reward": defaultdict(list),
    }
    for row in rows:
        grouped[str(row["phase"])][int(row["turn"])].append(float(row["seconds"]))

    turns = sorted({int(row["turn"]) for row in rows})
    rollout_means = [sum(grouped["rollout"][turn]) / len(grouped["rollout"][turn]) if grouped["rollout"][turn] else 0.0 for turn in turns]
    reward_means = [sum(grouped["reward"][turn]) / len(grouped["reward"][turn]) if grouped["reward"][turn] else 0.0 for turn in turns]

    plt = try_import_matplotlib()
    if plt is not None and output_path.suffix.lower() == ".png":
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

        for phase, color in [("rollout", "#1f77b4"), ("reward", "#d62728")]:
            phase_rows = by_phase.get(phase, [])
            if not phase_rows:
                continue
            xs = list(range(1, len(phase_rows) + 1))
            ys = [float(row["seconds"]) for row in phase_rows]
            axes[0].plot(xs, ys, alpha=0.25, linewidth=1.0, color=color, label=f"{phase} raw")
            axes[0].plot(xs, rolling_mean(ys, window), linewidth=2.0, color=color, label=f"{phase} rolling mean")

        axes[0].set_title("Per-Turn Rollout vs External Evaluation Time")
        axes[0].set_xlabel("Occurrence Index In Log")
        axes[0].set_ylabel("Seconds")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        width = 0.35
        x_positions = list(range(len(turns)))
        axes[1].bar([x - width / 2 for x in x_positions], rollout_means, width=width, color="#1f77b4", label="rollout mean")
        axes[1].bar([x + width / 2 for x in x_positions], reward_means, width=width, color="#d62728", label="reward mean")
        axes[1].set_xticks(x_positions, [str(turn) for turn in turns])
        axes[1].set_xlabel("Turn")
        axes[1].set_ylabel("Mean Seconds")
        axes[1].set_title("Mean Time By Turn")
        axes[1].grid(True, axis="y", alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close(fig)
        return [output_path]

    line_series = []
    for phase, color in [("rollout", "#1f77b4"), ("reward", "#d62728")]:
        phase_rows = by_phase.get(phase, [])
        if not phase_rows:
            continue
        xs = list(range(1, len(phase_rows) + 1))
        ys = [float(row["seconds"]) for row in phase_rows]
        line_series.append((f"{phase} rolling mean", color, list(zip(xs, rolling_mean(ys, window)))))
    line_output = output_path.with_name(f"{output_path.stem}_series{output_path.suffix}")
    bar_output = output_path.with_name(f"{output_path.stem}_by_turn{output_path.suffix}")
    _write_line_svg(line_series, line_output, "Per-Turn Rollout vs External Evaluation Time", "Occurrence Index In Log", "Seconds")
    _write_grouped_bar_svg(
        [str(turn) for turn in turns],
        [("rollout mean", "#1f77b4", rollout_means), ("reward mean", "#d62728", reward_means)],
        bar_output,
        "Mean Time By Turn",
        "Mean Seconds",
    )
    return [line_output, bar_output]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and plot training times from a drkernel training log.")
    parser.add_argument("log_path", type=Path, help="Path to the training log file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV and PNG outputs. Defaults to <log_dir>/<log_stem>_timing_analysis",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Rolling window for turn-level smoothing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.log_path.resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = log_path.parent / f"{log_path.stem}_timing_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    step_rows, turn_rows = parse_log(log_path)

    step_csv = output_dir / "step_timings.csv"
    turn_csv = output_dir / "turn_timings.csv"
    summary_csv = output_dir / "summary.csv"
    step_png = output_dir / "step_timings.png"
    turn_png = output_dir / "turn_timings.png"

    write_step_csv(step_rows, step_csv)
    write_turn_csv(turn_rows, turn_csv)
    write_summary_csv(step_rows, turn_rows, summary_csv)
    plot_suffix = ".png" if try_import_matplotlib() is not None else ".svg"
    step_plot = output_dir / f"step_timings{plot_suffix}"
    turn_plot = output_dir / f"turn_timings{plot_suffix}"
    step_plots = plot_step_timings(step_rows, step_plot)
    turn_plots = plot_turn_timings(turn_rows, turn_plot, args.rolling_window)

    print(f"log_path={log_path}")
    print(f"step_rows={len(step_rows)}")
    print(f"turn_rows={len(turn_rows)}")
    print(f"output_dir={output_dir}")
    print(f"step_csv={step_csv}")
    print(f"turn_csv={turn_csv}")
    print(f"summary_csv={summary_csv}")
    for path in step_plots:
        print(f"step_plot={path}")
    for path in turn_plots:
        print(f"turn_plot={path}")


if __name__ == "__main__":
    main()
