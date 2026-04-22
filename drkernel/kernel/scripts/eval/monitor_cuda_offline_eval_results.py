#!/usr/bin/env python3
"""Monitor CUDA offline eval runs and restart Qwen3-14B if it exits incomplete."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


REQUIRED_METRIC_KEYS = (
    "val/test_score/kernelbench_level2_validation",
    "val/test_score/kernelbench_level2_validation_pass@1",
    "val/test_score_extra/compilation_kernelbench_level2_validation",
    "val/kernel/best_by_turn_3/correctness_rate",
    "val/multiturn/num_turns/mean",
)

REQUIRED_LOG_MARKERS = (
    "Key Eval Parameters",
    "KERNEL_BACKEND: cuda_agent",
    "REFERENCE_CACHE_ENABLE: true",
    "REWARD_TASK_TIMEOUT: 30",
    "NUM_WARMUP: 5",
    "NUM_PERF_TRIALS: 50",
    "PERF_TRIM_COUNT: 5",
    "VLLM_LANGUAGE_MODEL_ONLY: true",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_event(log_path: Path, event: str, **fields: object) -> None:
    record = {"time": now(), "event": event, **fields}
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_cmd(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def tmux_has_session(session: str) -> bool:
    return run_cmd(["tmux", "has-session", "-t", session]).returncode == 0


def matching_process_exists(pattern: str) -> bool:
    result = run_cmd(["pgrep", "-f", pattern])
    return result.returncode == 0 and bool(result.stdout.strip())


def latest_run(logs_dir: Path, prefix: str) -> Path | None:
    runs = sorted(logs_dir.glob(f"{prefix}.run.*"), key=lambda path: path.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def validate_and_fix_run(run_dir: Path, expected_model_name: str, log_path: Path) -> tuple[bool, dict[str, object]]:
    step_dir = run_dir / "eval_results" / "step_0"
    metrics_path = step_dir / "metrics.json"
    raw_path = step_dir / "raw_responses.jsonl"
    graded_path = step_dir / "graded_results.parquet"
    main_log_path = run_dir / "main.log"

    details: dict[str, object] = {"run_dir": str(run_dir), "expected_model_name": expected_model_name}
    missing = []
    for path in (metrics_path, raw_path, graded_path, main_log_path):
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path))
    if missing:
        details["missing_or_empty"] = missing
        return False, details

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        details["metrics_error"] = repr(exc)
        return False, details

    missing_metrics = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    if missing_metrics:
        details["missing_metrics"] = missing_metrics
        return False, details
    details["metrics"] = {key: metrics.get(key) for key in REQUIRED_METRIC_KEYS}

    try:
        dataframe = pd.read_parquet(graded_path)
    except Exception as parquet_exc:
        jsonl_backup = step_dir / "graded_results.jsonl"
        try:
            if not jsonl_backup.exists():
                shutil.copy2(graded_path, jsonl_backup)
            dataframe = pd.read_json(jsonl_backup, lines=True)
            dataframe.to_parquet(graded_path, index=False)
            log_event(log_path, "converted_mislabeled_jsonl_to_parquet", run_dir=str(run_dir), rows=len(dataframe))
            dataframe = pd.read_parquet(graded_path)
        except Exception as convert_exc:
            details["parquet_error"] = repr(parquet_exc)
            details["conversion_error"] = repr(convert_exc)
            return False, details

    details["graded_rows"] = int(len(dataframe))
    if len(dataframe) != 100:
        details["row_count_error"] = len(dataframe)
        return False, details

    main_log = main_log_path.read_text(encoding="utf-8", errors="replace")
    missing_markers = [marker for marker in REQUIRED_LOG_MARKERS if marker not in main_log]
    if expected_model_name not in main_log:
        missing_markers.append(expected_model_name)
    if missing_markers:
        details["missing_log_markers"] = missing_markers
        return False, details

    return True, details


def start_qwen3_14b(args: argparse.Namespace, log_path: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = args.logs_dir / f"{args.qwen3_14b_prefix}.run.{timestamp}"
    script = args.drkernel_root / "kernel" / "scripts" / "eval" / "cuda_qwen3_14b_maxturns3_temp1_0_w5t50trim.sh"
    run_dir.mkdir(parents=True, exist_ok=True)

    if tmux_has_session(args.qwen3_14b_session):
        run_cmd(["tmux", "kill-session", "-t", args.qwen3_14b_session])

    command = "\n".join(
        [
            "set -euo pipefail",
            f"source {args.venv}/bin/activate",
            f"cd {args.drkernel_root}",
            "ray stop --force >/dev/null 2>&1 || true",
            "export RAY_ADDRESS=local",
            "export GLOO_SOCKET_IFNAME=ens22f0",
            "export NCCL_SOCKET_IFNAME=ens22f0",
            "export NCCL_NET=Socket",
            "export NCCL_IB_DISABLE=1",
            "export NCCL_SOCKET_FAMILY=AF_INET",
            "export NCCL_DEBUG=WARN",
            "export VLLM_LOGGING_LEVEL=INFO",
            "export KERNELGYM_VLLM_STATS_LOG_INTERVAL=10",
            "export WANDB_MODE=disabled",
            f"export RUN_TIMESTAMP={timestamp}",
            f"export RUN_DIR={run_dir}",
            f"export LOG_PATH={run_dir / 'main.log'}",
            f"bash {script}",
        ]
    )
    run_cmd(["tmux", "new-session", "-d", "-s", args.qwen3_14b_session, "bash", "-lc", command], check=True)
    log_event(log_path, "started_qwen3_14b", run_dir=str(run_dir), session=args.qwen3_14b_session)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--qwen35-run-dir", type=Path, required=True)
    parser.add_argument("--qwen3-14b-prefix", default="cuda-qwen3-14b-maxturns3-temp1.0-w5t50trim")
    parser.add_argument("--qwen3-14b-session", default="cuda-qwen3-14b-eval-step0-18")
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--max-restarts", type=int, default=4)
    parser.add_argument("--log-path", type=Path, required=True)
    args = parser.parse_args()
    args.drkernel_root = args.repo_root / "drkernel"
    args.logs_dir = args.drkernel_root / "logs"
    return args


def main() -> int:
    args = parse_args()
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    restarts = 0

    log_event(args.log_path, "supervisor_started", interval_seconds=args.interval_seconds)
    while True:
        qwen35_ok, qwen35_details = validate_and_fix_run(
            args.qwen35_run_dir, "Qwen3.5-27B", args.log_path
        )
        log_event(args.log_path, "qwen35_validation", ok=qwen35_ok, details=qwen35_details)

        qwen3_run = latest_run(args.logs_dir, args.qwen3_14b_prefix)
        qwen3_ok = False
        qwen3_details: dict[str, object] = {"run_dir": None}
        if qwen3_run is not None:
            qwen3_ok, qwen3_details = validate_and_fix_run(qwen3_run, "Qwen3-14B", args.log_path)
        log_event(args.log_path, "qwen3_14b_validation", ok=qwen3_ok, details=qwen3_details)

        if qwen35_ok and qwen3_ok:
            log_event(args.log_path, "all_eval_results_valid")
            return 0

        active = tmux_has_session(args.qwen3_14b_session) or matching_process_exists(
            "python -m kernel.main_grading .*Qwen3-14B"
        )
        if not qwen3_ok and not active:
            if restarts >= args.max_restarts:
                log_event(args.log_path, "restart_budget_exhausted", restarts=restarts)
                return 2
            restarts += 1
            start_qwen3_14b(args, args.log_path)
        else:
            log_event(args.log_path, "waiting", qwen3_14b_active=active, restarts=restarts)

        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    sys.exit(main())
