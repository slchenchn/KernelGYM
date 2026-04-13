#!/usr/bin/env python3
"""Compare checkpoint-eval results from two runs on a single figure.

Usage:
    python compare_checkpoint_eval_runs.py RUN_A RUN_B
    python compare_checkpoint_eval_runs.py RUN_A/eval_results RUN_B/eval_results --output /tmp/compare.png
"""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


METRIC_SPECS = [
    {
        "keys": ("val/test_score/kernelbench_level2_validation_pass@1", "val/test_score/kernelbench_level2_validation_pass@8"),
        "title": "test_score pass",
    },
    {
        "keys": ("val/kernel/turn_3/correctness_rate",),
        "title": "turn_3 correctness_rate (last turn only)",
    },
    {
        "keys": ("val/kernel/turn_3/fast@1_in_all",),
        "title": "turn_3 fast@1_in_all (last turn only)",
        "reference": 0.4038,
        "reference_label": "official(step300)",
    },
    {
        "keys": ("val/kernel/turn_3/fast@1.2_in_all",),
        "title": "turn_3 fast@1.2_in_all (last turn only)",
        "reference": 0.2400,
        "reference_label": "official(step300)",
    },
]


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "comparison"


def resolve_eval_results_dir(path: Path) -> tuple[Path, Path | None]:
    path = path.resolve()
    if path.is_dir() and path.name == "eval_results":
        return path, path.parent
    if (path / "eval_results").is_dir():
        return (path / "eval_results").resolve(), path
    if (path / "checkpoints" / "eval_results").is_dir():
        return (path / "checkpoints" / "eval_results").resolve(), path
    raise SystemExit(f"Could not find eval_results under {path}")


def detect_reference_backend(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    trainer_log = run_dir / "trainer.log"
    if not trainer_log.is_file():
        return None
    pattern = re.compile(r"reference_backend'?:\s*'([^']+)'")
    backend = None
    with trainer_log.open() as f:
        for line in f:
            m = pattern.search(line)
            if m:
                backend = m.group(1)
    return backend


def load_eval_rows(results_dir: Path) -> list[tuple[int, dict]]:
    rows = []
    for step_dir in sorted(results_dir.glob("step_*")):
        if not step_dir.is_dir():
            continue
        try:
            step = int(step_dir.name.split("_", 1)[1])
        except Exception:
            continue
        metrics_file = step_dir / "metrics.json"
        if not metrics_file.is_file():
            continue
        with metrics_file.open() as f:
            rows.append((step, json.load(f)))
    return sorted(rows)


def get_metric(metrics: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def build_label(run_dir: Path | None, backend: str | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    backend_map = {
        "torch_compile": "ref: compile",
        "pytorch": "ref: eager",
    }
    if backend in backend_map:
        return backend_map[backend]
    base = run_dir.name if run_dir is not None else "eval_results"
    if backend:
        return f"ref: {backend}"
    return base


def default_output(run_a: Path | None, run_b: Path | None) -> Path:
    if run_a is not None and run_b is not None and run_a.parent == run_b.parent:
        out_dir = run_a.parent / "comparison_plots"
    else:
        out_dir = Path.cwd() / "comparison_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    a = sanitize(run_a.name if run_a is not None else "run_a")
    b = sanitize(run_b.name if run_b is not None else "run_b")
    return out_dir / f"{a}__vs__{b}.png"


def write_summary(
    path: Path,
    source_a: Path,
    source_b: Path,
    label_a: str,
    label_b: str,
    backend_a: str | None,
    backend_b: str | None,
    rows_a,
    rows_b,
) -> None:
    text = [
        "Checkpoint Eval Comparison Summary",
        "================================",
        f"Run A: {source_a}",
        f"Run B: {source_b}",
        f"Label A: {label_a}",
        f"Label B: {label_b}",
        f"Reference backend A: {backend_a or 'unknown'}",
        f"Reference backend B: {backend_b or 'unknown'}",
        f"Steps A: {[s for s, _ in rows_a]}",
        f"Steps B: {[s for s, _ in rows_b]}",
    ]
    path.write_text("\n".join(text) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare checkpoint-eval metrics from two runs on one figure.")
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--label-a")
    parser.add_argument("--label-b")
    parser.add_argument("--output")
    args = parser.parse_args()

    eval_a, run_dir_a = resolve_eval_results_dir(Path(args.run_a))
    eval_b, run_dir_b = resolve_eval_results_dir(Path(args.run_b))
    rows_a = load_eval_rows(eval_a)
    rows_b = load_eval_rows(eval_b)
    if not rows_a:
        raise SystemExit(f"No metrics.json found under {eval_a}")
    if not rows_b:
        raise SystemExit(f"No metrics.json found under {eval_b}")

    backend_a = detect_reference_backend(run_dir_a)
    backend_b = detect_reference_backend(run_dir_b)
    label_a = build_label(run_dir_a, backend_a, args.label_a)
    label_b = build_label(run_dir_b, backend_b, args.label_b)

    output_path = Path(args.output) if args.output else default_output(run_dir_a, run_dir_b)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_metrics = len(METRIC_SPECS)
    ncols = 2
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8.4), squeeze=False)
    fig.suptitle("Checkpoint Eval Comparison", fontsize=14, fontweight="bold")

    flat_axes = axes.ravel()
    for ax, spec in zip(flat_axes, METRIC_SPECS):
        keys = spec["keys"]
        title = spec["title"]
        xs_a, ys_a = [], []
        for step, metrics in rows_a:
            value = get_metric(metrics, keys)
            if value is not None:
                xs_a.append(step)
                ys_a.append(value)
        xs_b, ys_b = [], []
        for step, metrics in rows_b:
            value = get_metric(metrics, keys)
            if value is not None:
                xs_b.append(step)
                ys_b.append(value)

        if ys_a:
            ax.plot(xs_a, ys_a, marker="o", linewidth=2, label=label_a)
        if ys_b:
            ax.plot(xs_b, ys_b, marker="o", linewidth=2, label=label_b)
        reference = spec.get("reference")
        if reference is not None:
            ref_label = f"{spec.get('reference_label', 'reference')} ({reference:.4f})"
            ax.axhline(y=reference, color="gray", linestyle="--", linewidth=1.5, alpha=0.8, label=ref_label)

        ax.set_title(title)
        ax.set_xlabel("Checkpoint Step")
        ax.set_ylabel("Value")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for ax in flat_axes[n_metrics:]:
        ax.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    summary_path = output_path.with_suffix(".txt")
    write_summary(summary_path, eval_a, eval_b, label_a, label_b, backend_a, backend_b, rows_a, rows_b)
    print(f"Saved {output_path}")
    print(f"Saved {summary_path}")
    print(f"Detected backend A: {backend_a or 'unknown'}")
    print(f"Detected backend B: {backend_b or 'unknown'}")


if __name__ == "__main__":
    main()
