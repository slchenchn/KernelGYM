from __future__ import annotations

import json
import os
import socket
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


def _default_event_log_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "logs" / "structured"


def _event_log_dir() -> Path:
    override = os.environ.get("DRKERNEL_EVENT_LOG_DIR")
    if override:
        return Path(override)
    return _default_event_log_dir()


def _event_log_path(event_type: str) -> Path:
    safe_type = event_type.replace("/", "_")
    return _event_log_dir() / f"{safe_type}.pid{os.getpid()}.jsonl"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_or_none(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[0]
    return value


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, defaultdict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_to_jsonable(v) for v in sorted(value, key=repr)]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(value.min()) if value.size else None,
            "max": float(value.max()) if value.size else None,
        }
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        detached = value.detach().cpu()
        return {
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "sum": float(detached.sum().item()),
            "mean": float(detached.float().mean().item()),
            "min": float(detached.min().item()),
            "max": float(detached.max().item()),
        }
    return repr(value)


def _truncate_text(text: Any, limit: int = 400) -> Any:
    if not isinstance(text, str):
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def append_jsonl_event(event_type: str, payload: Dict[str, Any]) -> None:
    path = _event_log_path(event_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event_type": event_type,
        "ts": _utc_timestamp(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }
    record.update(payload)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")


def _extract_scalar(mapping: Dict[str, Any], key: str) -> Any:
    if key not in mapping:
        return None
    return _first_or_none(mapping.get(key))


def build_env_result_record(env_result: Dict[str, Any]) -> Dict[str, Any]:
    reward_tensor = env_result.get("reward_tensor")
    reward_sum = None
    reward_shape = None
    if torch.is_tensor(reward_tensor):
        reward_sum = float(reward_tensor.sum().item())
        reward_shape = list(reward_tensor.shape)
    elif reward_tensor is not None:
        try:
            reward_sum = float(reward_tensor)
        except Exception:
            reward_sum = None

    reward_extra_info = env_result.get("reward_extra_info", {}) or {}
    env_state = env_result.get("env_state", {}) or {}
    metadata = env_state.get("metadata", {}) or {}

    record = {
        "task_id": env_state.get("task_id"),
        "status": env_state.get("status", _extract_scalar(reward_extra_info, "status")),
        "success": env_state.get("success", _extract_scalar(reward_extra_info, "success")),
        "compilation": env_state.get("compiled", _extract_scalar(reward_extra_info, "compilation")),
        "correctness": env_state.get("correctness", _extract_scalar(reward_extra_info, "correctness")),
        "is_decoy_kernel": env_state.get(
            "decoy_kernel", _extract_scalar(reward_extra_info, "is_decoy_kernel")
        ),
        "reward": env_state.get("reward"),
        "score": env_state.get("score"),
        "reward_sum": reward_sum,
        "reward_shape": reward_shape,
        "reference_runtime": env_state.get("reference_runtime"),
        "kernel_runtime": env_state.get("kernel_runtime"),
        "speedup": env_state.get("speedup"),
        "error": _truncate_text(
            env_state.get("error_message")
            or env_state.get("error")
            or _extract_scalar(reward_extra_info, "error")
        ),
        "runtime_error_name": metadata.get("runtime_error_name"),
        "runtime_error": _truncate_text(metadata.get("runtime_error")),
        "hardware": metadata.get("hardware") or metadata.get("gpu_name"),
        "gpu_name": metadata.get("gpu_name") or metadata.get("hardware"),
        "device": metadata.get("device"),
        "backend": metadata.get("backend"),
        "num_correct_trials": metadata.get("num_correct_trials"),
        "num_perf_trials": metadata.get("num_perf_trials"),
        "processing_time": env_state.get("processing_time"),
        "submitted_at": env_state.get("submitted_at"),
        "completed_at": env_state.get("completed_at"),
        "reward_extra_info": reward_extra_info,
        "env_state": env_state,
        "metadata": metadata,
    }
    record["compiled"] = record["compilation"]

    return record


def format_env_result_summary(env_result: Dict[str, Any]) -> str:
    record = build_env_result_record(env_result)
    reward_value = record.get("reward_sum")
    if reward_value is None:
        reward_value = record.get("reward")
    return (
        "Env Result: "
        f"task_id={record.get('task_id')} "
        f"status={record.get('status')} "
        f"success={record.get('success')} "
        f"compilation={record.get('compilation')} "
        f"correctness={record.get('correctness')} "
        f"decoy={record.get('is_decoy_kernel')} "
        f"reward={reward_value} "
        f"hardware={record.get('hardware')} "
        f"error={repr(record.get('error'))}"
    )


def build_batch_heartbeat_record(
    *,
    completed: int,
    total: int,
    pending: int,
    elapsed: float,
    tokens_in_use: int,
    rate_limit: int,
    pending_tasks_info: Iterable[str],
) -> Dict[str, Any]:
    return {
        "completed": completed,
        "total": total,
        "pending": pending,
        "elapsed_s": float(elapsed),
        "tokens_in_use": tokens_in_use,
        "rate_limit": rate_limit,
        "pending_tasks": list(pending_tasks_info),
    }


def format_batch_heartbeat_summary(record: Dict[str, Any]) -> str:
    return (
        "[BatchHeartbeat] "
        f"completed={record['completed']}/{record['total']} "
        f"pending={record['pending']} "
        f"elapsed={record['elapsed_s']:.1f}s "
        f"tokens_in_use={record['tokens_in_use']}/{record['rate_limit']}"
    )
