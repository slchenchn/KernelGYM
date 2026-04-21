from __future__ import annotations

import json
import os
import re
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


def _event_blob_dir(event_type: str) -> Path:
    safe_type = event_type.replace("/", "_")
    return _event_log_dir() / f"{safe_type}.pid{os.getpid()}.blobs"


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


def truncate_feedback_for_prompt(
    text: Any,
    max_chars: int | None,
    truncate_side: str = "middle",
) -> str:
    """Bound tool feedback before it is injected into the next rollout prompt."""
    if not isinstance(text, str):
        text = str(text)
    try:
        limit = int(max_chars or 0)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0 or len(text) <= limit:
        return text

    marker = f"\n...[truncated, total_chars={len(text)}]...\n"
    keep_chars = limit - len(marker)
    if keep_chars <= 0:
        return text[:limit]

    side = str(truncate_side or "middle").lower()
    if side in {"left", "start", "head"}:
        return marker + text[-keep_chars:]
    if side in {"right", "end", "tail"}:
        return text[:keep_chars] + marker

    head_chars = keep_chars // 2
    tail_chars = keep_chars - head_chars
    return text[:head_chars] + marker + text[-tail_chars:]


CODE_BLOCK_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<code>.*?)```", re.DOTALL)


def _safe_blob_component(value: str | None, default: str) -> str:
    text = (value or "").strip() or default
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _persist_generated_code_debug_blob(
    *,
    request_id: str,
    turn_index: int,
    prompt_token_ids: list[int] | None,
    model_response_token_ids: list[int] | None,
    model_logprobs: list[float] | None,
) -> str | None:
    prompt_ids = list(prompt_token_ids or [])
    token_ids = list(model_response_token_ids or [])
    token_logprobs = [float(v) for v in (model_logprobs or [])]
    if not prompt_ids and not token_ids and not token_logprobs:
        return None

    blob_dir = _event_blob_dir("generated_code")
    blob_dir.mkdir(parents=True, exist_ok=True)
    request_part = _safe_blob_component(request_id, "request")
    filename = f"{request_part}.turn{int(turn_index):02d}.npz"
    blob_path = blob_dir / filename
    np.savez_compressed(
        blob_path,
        prompt_token_ids=np.asarray(prompt_ids, dtype=np.int32),
        model_response_token_ids=np.asarray(token_ids, dtype=np.int32),
        model_logprobs=np.asarray(token_logprobs, dtype=np.float32),
    )
    return os.path.relpath(blob_path, _event_log_dir())


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


def extract_code_blocks(text: str | None) -> list[dict[str, str]]:
    if not text:
        return []
    blocks: list[dict[str, str]] = []
    for match in CODE_BLOCK_RE.finditer(text):
        blocks.append(
            {
                "language": match.group("lang").strip(),
                "code": match.group("code").strip(),
            }
        )
    return blocks


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


def build_turn_token_record(
    *,
    request_id: str,
    turn_index: int,
    prefill_tokens: int,
    decode_tokens: int,
    model_time_s: float,
    env_time_s: float,
    is_validate: bool,
    global_step: int,
    sample_uuid: str | None = None,
    entry_point: str | None = None,
) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "sample_uuid": sample_uuid,
        "entry_point": entry_point,
        "turn_index": int(turn_index),
        "prefill_tokens": int(prefill_tokens),
        "decode_tokens": int(decode_tokens),
        "model_time_s": float(model_time_s),
        "env_time_s": float(env_time_s),
        "is_validate": bool(is_validate),
        "global_step": int(global_step),
    }


def build_generated_code_record(
    *,
    request_id: str,
    turn_index: int,
    model_response: str,
    tool_response: str | None,
    prompt_token_ids: list[int] | None = None,
    model_response_token_ids: list[int] | None = None,
    model_logprobs: list[float] | None = None,
    prefill_tokens: int,
    decode_tokens: int,
    model_time_s: float,
    env_time_s: float,
    is_validate: bool,
    global_step: int,
    sample_uuid: str | None = None,
    entry_point: str | None = None,
) -> Dict[str, Any]:
    code_blocks = extract_code_blocks(model_response)
    prompt_count = len(prompt_token_ids or [])
    token_count = len(model_response_token_ids or [])
    logprob_count = len(model_logprobs or [])
    blob_relpath = _persist_generated_code_debug_blob(
        request_id=request_id,
        turn_index=turn_index,
        prompt_token_ids=prompt_token_ids,
        model_response_token_ids=model_response_token_ids,
        model_logprobs=model_logprobs,
    )
    token_logprobs = [float(v) for v in (model_logprobs or [])]
    token_logprob_sum = float(sum(token_logprobs)) if token_logprobs else None
    token_logprob_mean = float(token_logprob_sum / len(token_logprobs)) if token_logprobs else None
    return {
        "request_id": request_id,
        "sample_uuid": sample_uuid,
        "entry_point": entry_point,
        "turn_index": int(turn_index),
        "prefill_tokens": int(prefill_tokens),
        "decode_tokens": int(decode_tokens),
        "model_time_s": float(model_time_s),
        "env_time_s": float(env_time_s),
        "is_validate": bool(is_validate),
        "global_step": int(global_step),
        "model_response": model_response,
        "tool_response": tool_response,
        "model_trace_blob": blob_relpath,
        "prompt_token_count": prompt_count,
        "model_response_token_count": token_count,
        "model_logprob_count": logprob_count,
        "model_logprob_sum": token_logprob_sum,
        "model_logprob_mean": token_logprob_mean,
        "code_block_count": len(code_blocks),
        "code_blocks": code_blocks,
    }


def format_turn_model_summary(
    *,
    turn_index: int,
    model_time_s: float,
    prefill_tokens: int,
    decode_tokens: int,
    model_response: str,
) -> str:
    code_blocks = extract_code_blocks(model_response)
    return (
        f"Turn {turn_index} | Model time: {model_time_s:.2f}s | "
        "Model Response: "
        f"chars={len(model_response)} prefill_tokens={prefill_tokens} "
        f"decode_tokens={decode_tokens} code_blocks={len(code_blocks)}"
    )


def format_turn_env_summary(
    *,
    turn_index: int,
    env_time_s: float,
    tool_response: str,
) -> str:
    return (
        f"Turn {turn_index} | Env time: {env_time_s:.2f}s | "
        f"Tool Response: chars={len(tool_response)}"
    )
