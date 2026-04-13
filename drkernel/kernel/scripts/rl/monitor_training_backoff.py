#!/usr/bin/env python3

import argparse
import datetime as dt
import re
import subprocess
import time
from pathlib import Path


def run(cmd: str, cwd: Path) -> str:
    proc = subprocess.run(cmd, shell=True, cwd=str(cwd), text=True, capture_output=True)
    parts = []
    if proc.stdout.strip():
        parts.append(proc.stdout.strip())
    if proc.stderr.strip():
        parts.append(proc.stderr.strip())
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--head-host", required=True)
    parser.add_argument("--head-port", required=True)
    parser.add_argument("--venv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--intervals", default="0,60,120,240,480")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    run_dir = Path(args.run_dir)
    main_log = run_dir / "main.log"
    output_path = Path(args.output)
    latest_output_path = output_path.with_suffix(output_path.suffix + ".latest")
    alerts_output_path = output_path.with_suffix(output_path.suffix + ".alerts")
    intervals = [int(x) for x in args.intervals.split(",") if x.strip()]

    grep_pattern = (
        "Stale file handle|step=0|step=1|prompt_rows=|Initial validation metrics|"
        "Training Progress|Traceback|Error executing job|OutOfMemoryError|CUDA out of memory"
    )
    alert_pattern = (
        "Stale file handle|Traceback|Error executing job|OutOfMemoryError|CUDA out of memory|"
        "timeout after [0-9]+s|RayTaskError|torch\\.OutOfMemoryError"
    )
    ssh_prefix = f"ssh -p {args.head_port} root@{args.head_host}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"Backoff monitor started at {dt.datetime.now().isoformat()}\n"
        f"run_dir={run_dir}\n"
        f"intervals={intervals}\n\n"
    )
    for path in (output_path, latest_output_path, alerts_output_path):
        with path.open("w", encoding="utf-8") as f:
            f.write(header)

    start = time.time()
    for idx, delay in enumerate(intervals):
        if idx > 0:
            time.sleep(delay)

        elapsed = int(time.time() - start)
        main_matches = run(f"grep -nE \"{grep_pattern}\" \"{main_log}\" | tail -n 120 || true", repo_root)
        stdout_tail = run(f"{ssh_prefix} \"tail -n 120 /tmp/train-8b-hfsdp8-pytorch-eager.log\"", repo_root)
        ray_status = run(
            f"{ssh_prefix} \"source {args.venv}/bin/activate && ray status | sed -n '1,40p'\"",
            repo_root,
        )
        alert_sources = "\n".join(x for x in (main_matches, stdout_tail) if x)
        alert_lines = [
            line for line in alert_sources.splitlines() if re.search(alert_pattern, line, flags=re.IGNORECASE)
        ]
        summary_lines = [
            f"summary: alerts={len(alert_lines)} main_log_matches={len(main_matches.splitlines()) if main_matches else 0}",
        ]
        if alert_lines:
            summary_lines.append(f"latest_alert: {alert_lines[-1]}")
        sections = [
            f"===== check {idx} elapsed={elapsed}s at {dt.datetime.now().isoformat()} =====",
            "\n".join(summary_lines),
            main_matches,
            stdout_tail,
            ray_status,
        ]
        chunk = "\n".join(s for s in sections if s) + "\n\n"
        with output_path.open("a", encoding="utf-8") as f:
            f.write(chunk)
        with latest_output_path.open("w", encoding="utf-8") as f:
            f.write(header)
            f.write(chunk)
        with alerts_output_path.open("a", encoding="utf-8") as f:
            if alert_lines:
                f.write(f"===== check {idx} elapsed={elapsed}s at {dt.datetime.now().isoformat()} =====\n")
                f.write("\n".join(alert_lines))
                f.write("\n\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
