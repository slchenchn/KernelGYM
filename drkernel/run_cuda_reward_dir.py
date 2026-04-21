"""Batch reward evaluation for stored CUDA-Agent result JSON files."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DRKERNEL_ROOT = REPO_ROOT / "drkernel"
for path in (REPO_ROOT, DRKERNEL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from kernel.rewards.reward_client import KernelRewardClient
from kernel.utils.kernel_code import (
    extract_cuda_agent_sections,
    extract_kernel_submission,
)
from kernelgym.toolkit.validation import precheck_cuda_agent_submission


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def _sample_sort_key(path: Path) -> tuple[int, str]:
    match = re.match(r"result_(\d+)\.json$", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**18, path.name


def _read_existing_results(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_file = record.get("sample_file")
            if sample_file:
                completed.add(sample_file)
    return completed


def _build_reward_config(args: argparse.Namespace) -> SimpleNamespace:
    penalties = AttrDict(
        {
            "penalty_score": args.penalty_score,
            "precheck_fail": args.precheck_fail,
            "compilation_fail": args.compilation_fail,
            "correctness_fail": args.correctness_fail,
            "perf_degrade": args.perf_degrade,
        }
    )
    return SimpleNamespace(
        server_url=args.server_url,
        timeout=args.task_timeout,
        task_timeout_in_client=args.task_timeout_in_client,
        max_retries=args.max_retries,
        rate_limit=args.max_concurrent,
        max_concurrent=args.max_concurrent,
        acquire_timeout=args.acquire_timeout,
        reward_func_name=args.reward_func_name,
        init_correct_weight=args.init_correct_weight,
        init_performance_weight=args.init_performance_weight,
        speedup_eps=args.speedup_eps,
        apply_compilation_fail_penalty=args.apply_compilation_fail_penalty,
        apply_precheck_fail_penalty=args.apply_precheck_fail_penalty,
        speedup_reward_upper_bound=args.speedup_reward_upper_bound,
        speedup_reward_lower_bound=args.speedup_reward_lower_bound,
        reward_policy=SimpleNamespace(penalties=penalties),
        coverage_reward=SimpleNamespace(
            enable=args.coverage_reward_enable,
            weight=args.coverage_reward_weight,
            reward_type=args.coverage_reward_type,
        ),
    )


def _normalize_result(
    sample_path: Path,
    result: dict[str, Any],
    *,
    local_error: str | None = None,
    local_precheck: bool = False,
) -> dict[str, Any]:
    metadata = result.get("metadata") or {}
    normalized = {
        "sample_file": sample_path.name,
        "sample_path": str(sample_path),
        "status": result.get("status"),
        "reward": result.get("reward"),
        "score": result.get("score", result.get("reward")),
        "success": result.get("success"),
        "compiled": result.get("compiled"),
        "correctness": result.get("correctness"),
        "speedup": result.get("speedup"),
        "error": result.get("error"),
        "decoy_kernel": result.get("decoy_kernel", result.get("is_decoy_kernel")),
        "num_custom_kernel": result.get("num_custom_kernel"),
        "num_total_kernels": result.get("num_total_kernels"),
        "custom_kernel_cuda_time_in_profiling_us": result.get(
            "custom_kernel_cuda_time_in_profiling_us"
        ),
        "total_kernel_run_time_in_profiling_us": result.get(
            "total_kernel_run_time_in_profiling_us"
        ),
        "local_error": local_error,
        "local_precheck": local_precheck,
        "metadata_backend": metadata.get("backend"),
        "metadata_device": metadata.get("device"),
        "metadata_gpu_name": metadata.get("gpu_name"),
    }
    return normalized


def _local_failed_reward(
    client: KernelRewardClient,
    sample_path: Path,
    error_message: str,
) -> dict[str, Any]:
    result = client.calculate_reward_weighted(
        {
            "status": "failed",
            "error_message": error_message,
        }
    )
    return _normalize_result(
        sample_path,
        result,
        local_error=error_message,
        local_precheck=error_message.lower().startswith("precheck failed:"),
    )


def _extract_assistant_response(messages: list[dict[str, Any]]) -> str | None:
    assistants = [
        message.get("content", "")
        for message in messages
        if message.get("role") == "assistant"
    ]
    if not assistants:
        return None
    return assistants[-1]


def _build_remote_task(sample_path: Path, sample: dict[str, Any]) -> dict[str, Any]:
    response = _extract_assistant_response(sample.get("messages") or [])
    if response is None:
        raise ValueError("missing assistant response")
    return {
        "reference_code": sample["original_python_code"],
        "kernel_code": extract_kernel_submission(response, kernel_backend="cuda_agent"),
        "kernel_backend": "cuda_agent",
        "entry_point": sample.get("entry_point", "Model"),
        "use_reference_cache": True,
        "uuid": str(sample.get("uuid", sample_path.stem)),
        "is_valid": False,
    }


def _classify_local_sample(
    client: KernelRewardClient, sample_path: Path
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    response = _extract_assistant_response(sample.get("messages") or [])
    if response is None:
        return _local_failed_reward(
            client,
            sample_path,
            "Precheck failed: assistant response is missing",
        ), None

    sections = extract_cuda_agent_sections(response)
    if not sections:
        return _local_failed_reward(
            client,
            sample_path,
            "Precheck failed: CUDA-Agent sections are missing from the assistant response",
        ), None

    model_new = sections.get("MODEL_NEW", "")
    cuda_sources: dict[str, str] = {}
    if sections.get("CUDA_KERNELS"):
        cuda_sources["kernels/generated.cu"] = sections["CUDA_KERNELS"]
    if sections.get("APPLY_BINDINGS"):
        cuda_sources["kernels/generated_binding.cpp"] = sections["APPLY_BINDINGS"]

    error_message, _, _ = precheck_cuda_agent_submission(model_new, cuda_sources)
    if error_message:
        return _local_failed_reward(client, sample_path, error_message), None

    task = _build_remote_task(sample_path, sample)
    return None, task


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": len(records),
        "reward_nonzero": 0,
        "compiled_true": 0,
        "correct_true": 0,
        "decoy_true": 0,
        "local_precheck_fail": 0,
        "server_fail": 0,
        "by_error": Counter(),
    }
    for record in records:
        if record.get("reward") not in (0, 0.0, None):
            summary["reward_nonzero"] += 1
        if record.get("compiled") is True:
            summary["compiled_true"] += 1
        if record.get("correctness") is True:
            summary["correct_true"] += 1
        if record.get("decoy_kernel") is True:
            summary["decoy_true"] += 1
        if record.get("local_precheck"):
            summary["local_precheck_fail"] += 1
        if record.get("status") != "completed" and not record.get("local_precheck"):
            summary["server_fail"] += 1
        if record.get("error"):
            summary["by_error"][record["error"]] += 1
    summary["by_error"] = dict(summary["by_error"].most_common())
    return summary


async def _process_remote_batches(
    client: KernelRewardClient,
    output_jsonl: Path,
    pending_batches: list[tuple[list[Path], list[dict[str, Any]]]],
    *,
    task_timeout: int,
    task_timeout_in_client: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with output_jsonl.open("a", encoding="utf-8") as handle:
        for batch_idx, (sample_paths, tasks) in enumerate(pending_batches, start=1):
            started = time.time()
            results = await client.compute_batch_rewards(
                tasks,
                task_timeout=task_timeout,
                task_timeout_in_client=task_timeout_in_client,
            )
            elapsed = time.time() - started
            for sample_path, result in zip(sample_paths, results):
                record = _normalize_result(sample_path, result)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[batch {batch_idx}] remote={len(tasks)} elapsed={elapsed:.1f}s "
                f"last_sample={sample_paths[-1].name}"
            )
    return records


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    reward_config = _build_reward_config(args)
    client = KernelRewardClient(reward_config=reward_config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "results.jsonl"
    summary_json = output_dir / "summary.json"

    completed = _read_existing_results(output_jsonl) if args.resume else set()
    all_sample_paths = sorted(Path(args.input_dir).glob("result_*.json"), key=_sample_sort_key)
    target_sample_paths = [path for path in all_sample_paths if path.name not in completed]

    print(
        f"[setup] input_dir={args.input_dir} total={len(all_sample_paths)} "
        f"skip_existing={len(completed)} remaining={len(target_sample_paths)}"
    )

    local_records: list[dict[str, Any]] = []
    remote_batches: list[tuple[list[Path], list[dict[str, Any]]]] = []
    current_batch_paths: list[Path] = []
    current_batch_tasks: list[dict[str, Any]] = []

    with output_jsonl.open("a", encoding="utf-8") as handle:
        for idx, sample_path in enumerate(target_sample_paths, start=1):
            local_record, remote_task = _classify_local_sample(client, sample_path)
            if local_record is not None:
                local_records.append(local_record)
                handle.write(json.dumps(local_record, ensure_ascii=False) + "\n")
                if len(local_records) % 50 == 0:
                    handle.flush()
            else:
                current_batch_paths.append(sample_path)
                current_batch_tasks.append(
                    {
                        **remote_task,
                        "task_timeout": args.task_timeout,
                        "task_timeout_in_client": args.task_timeout_in_client,
                        "num_correct_trials": args.num_correct_trials,
                        "num_perf_trials": args.num_perf_trials,
                        "num_warmup": args.num_warmup,
                        "perf_trim_count": args.perf_trim_count,
                        "enable_profiling": args.enable_profiling,
                        "verbose_errors": args.verbose_errors,
                        "detect_decoy_kernel": args.detect_decoy_kernel,
                        "reference_backend": args.reference_backend,
                        "use_reference_cache": args.use_reference_cache,
                    }
                )
                if len(current_batch_tasks) >= args.batch_size:
                    remote_batches.append((current_batch_paths, current_batch_tasks))
                    current_batch_paths = []
                    current_batch_tasks = []

            if idx % 200 == 0:
                print(
                    f"[scan] processed={idx}/{len(target_sample_paths)} "
                    f"local_fail={len(local_records)} queued_remote={sum(len(b[1]) for b in remote_batches) + len(current_batch_tasks)}"
                )
        handle.flush()

    if current_batch_tasks:
        remote_batches.append((current_batch_paths, current_batch_tasks))

    print(
        f"[dispatch] local_fail={len(local_records)} remote_batches={len(remote_batches)} "
        f"remote_tasks={sum(len(batch[1]) for batch in remote_batches)}"
    )

    remote_records = await _process_remote_batches(
        client,
        output_jsonl,
        remote_batches,
        task_timeout=args.task_timeout,
        task_timeout_in_client=args.task_timeout_in_client,
    )

    all_records = local_records + remote_records
    if args.resume:
        with output_jsonl.open("r", encoding="utf-8") as handle:
            all_records = [json.loads(line) for line in handle if line.strip()]

    summary = _summarize(all_records)
    summary.update(
        {
            "input_dir": str(args.input_dir),
            "output_jsonl": str(output_jsonl),
            "resume": args.resume,
            "server_url": args.server_url,
            "max_concurrent": args.max_concurrent,
            "batch_size": args.batch_size,
            "num_correct_trials": args.num_correct_trials,
            "num_perf_trials": args.num_perf_trials,
            "num_warmup": args.num_warmup,
            "perf_trim_count": args.perf_trim_count,
        }
    )
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing result_*.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for results.jsonl and summary.json",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing results.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--server-url", default="http://192.168.16.39:8111")
    parser.add_argument("--reward-func-name", default="calculate_reward_weighted")
    parser.add_argument("--reference-backend", default="torch_compile")
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--acquire-timeout", type=int, default=2400)
    parser.add_argument("--task-timeout", type=int, default=30)
    parser.add_argument("--task-timeout-in-client", type=int, default=1800)
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=50)
    parser.add_argument("--num-warmup", type=int, default=30)
    parser.add_argument("--perf-trim-count", type=int, default=5)
    parser.add_argument("--init-correct-weight", type=float, default=0.5)
    parser.add_argument("--init-performance-weight", type=float, default=0.5)
    parser.add_argument("--speedup-eps", type=float, default=0.01)
    parser.add_argument("--speedup-reward-upper-bound", type=float, default=3.0)
    parser.add_argument("--speedup-reward-lower-bound", type=float, default=0.0)
    parser.add_argument("--penalty-score", type=float, default=0.0)
    parser.add_argument("--precheck-fail", type=float, default=-0.5)
    parser.add_argument("--compilation-fail", type=float, default=-0.5)
    parser.add_argument("--correctness-fail", type=float, default=-0.3)
    parser.add_argument("--perf-degrade", type=float, default=-0.1)
    parser.add_argument("--coverage-reward-enable", action="store_true")
    parser.add_argument("--coverage-reward-weight", type=float, default=0.25)
    parser.add_argument("--coverage-reward-type", default="time_coverage")
    parser.add_argument("--apply-compilation-fail-penalty", action="store_true", default=True)
    parser.add_argument("--apply-precheck-fail-penalty", action="store_true", default=True)
    parser.add_argument("--detect-decoy-kernel", action="store_true", default=True)
    parser.add_argument("--enable-profiling", action="store_true", default=True)
    parser.add_argument("--verbose-errors", action="store_true", default=True)
    parser.add_argument(
        "--no-reference-cache",
        dest="use_reference_cache",
        action="store_false",
        help="Disable reference runtime cache for remote tasks",
    )
    parser.set_defaults(use_reference_cache=True)
    args = parser.parse_args()

    summary = asyncio.run(_run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
