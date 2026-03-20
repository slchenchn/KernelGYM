from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def default_event_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "structured"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-round prefill/decode token counts from turn_token_stats jsonl sidecars."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional jsonl files or directories. Defaults to drkernel/logs/structured/turn_token_stats*.jsonl",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=3,
        help="Only include turn_index <= this value. Default: 3",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write one CSV summary next to each input jsonl.",
    )
    return parser.parse_args()


def discover_jsonl_files(paths: list[str]) -> list[Path]:
    if not paths:
        return sorted(default_event_dir().glob("turn_token_stats*.jsonl"))

    discovered: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            discovered.extend(sorted(path.glob("turn_token_stats*.jsonl")))
        elif path.is_file():
            discovered.append(path)
    return discovered


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize_jsonl(path: Path, max_round: int) -> list[dict[str, object]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            turn_index = int(record.get("turn_index", 0))
            if turn_index <= 0 or turn_index > max_round:
                continue
            grouped[turn_index]["prefill_tokens"].append(int(record["prefill_tokens"]))
            grouped[turn_index]["decode_tokens"].append(int(record["decode_tokens"]))
            grouped[turn_index]["model_time_s"].append(float(record.get("model_time_s", 0.0)))
            grouped[turn_index]["env_time_s"].append(float(record.get("env_time_s", 0.0)))

    rows: list[dict[str, object]] = []
    for turn_index in sorted(grouped):
        bucket = grouped[turn_index]
        prefill = bucket["prefill_tokens"]
        decode = bucket["decode_tokens"]
        model_time = bucket["model_time_s"]
        env_time = bucket["env_time_s"]
        rows.append(
            {
                "round": turn_index,
                "count": len(prefill),
                "prefill_mean": sum(prefill) / len(prefill),
                "prefill_median": float(median(prefill)),
                "prefill_p90": percentile(prefill, 0.9),
                "prefill_total": sum(prefill),
                "decode_mean": sum(decode) / len(decode),
                "decode_median": float(median(decode)),
                "decode_p90": percentile(decode, 0.9),
                "decode_total": sum(decode),
                "model_time_mean_s": sum(model_time) / len(model_time),
                "env_time_mean_s": sum(env_time) / len(env_time),
            }
        )
    return rows


def print_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    print(f"\n{path}")
    if not rows:
        print("No matching turn_token_stats rows.")
        return
    print(
        "| round | count | prefill_mean | prefill_median | prefill_p90 | prefill_total | "
        "decode_mean | decode_median | decode_p90 | decode_total | model_time_mean_s | env_time_mean_s |"
    )
    print(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in rows:
        print(
            f"| {row['round']} | {row['count']} | {row['prefill_mean']:.2f} | {row['prefill_median']:.2f} | "
            f"{row['prefill_p90']:.2f} | {row['prefill_total']} | {row['decode_mean']:.2f} | "
            f"{row['decode_median']:.2f} | {row['decode_p90']:.2f} | {row['decode_total']} | "
            f"{row['model_time_mean_s']:.2f} | {row['env_time_mean_s']:.2f} |"
        )


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    output_path = path.with_suffix(".round_token_summary.csv")
    fieldnames = [
        "round",
        "count",
        "prefill_mean",
        "prefill_median",
        "prefill_p90",
        "prefill_total",
        "decode_mean",
        "decode_median",
        "decode_p90",
        "decode_total",
        "model_time_mean_s",
        "env_time_mean_s",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> int:
    args = parse_args()
    files = discover_jsonl_files(args.paths)
    if not files:
        print("No turn_token_stats jsonl files found.")
        return 0

    for path in files:
        rows = summarize_jsonl(path, max_round=args.max_round)
        print_markdown(path, rows)
        if args.write_csv:
            output_path = write_csv(path, rows)
            print(f"csv={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
