#!/usr/bin/env python3
"""Extract initial-validation metrics from one or more drkernel logs."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def _find_metrics_block(text: str) -> dict:
    marker = "Initial validation metrics:"
    start = text.find(marker)
    if start < 0:
        raise ValueError("Initial validation metrics block not found")

    brace_start = text.find("{", start)
    if brace_start < 0:
        raise ValueError("Initial validation metrics dict start not found")

    depth = 0
    end = None
    for idx in range(brace_start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        raise ValueError("Initial validation metrics dict end not found")

    raw = text[brace_start:end]
    return ast.literal_eval(raw)


def _select_metrics(metrics: dict) -> dict:
    wanted_prefixes = (
        "val/test_score_extra/kg_",
        "val/test_score_extra/custom_kernel_cuda_time_in_profiling_us",
        "val/test_score_extra/total_kernel_run_time_in_profiling_us",
        "val/test_score_extra/time_coverage",
        "val/test_score_extra/num_coverage",
        "val/test_score/",
    )
    selected = {}
    for key, value in metrics.items():
        if key.startswith(wanted_prefixes):
            selected[key] = value
    return selected


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "Usage: extract_initial_val_metrics.py LOG [LOG ...]",
            file=sys.stderr,
        )
        return 1

    output = {}
    for raw_path in argv[1:]:
        path = Path(raw_path)
        metrics = _find_metrics_block(path.read_text())
        output[str(path)] = _select_metrics(metrics)

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
