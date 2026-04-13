import os
import shutil
import subprocess
from collections.abc import Mapping
from typing import Any

import torch


BF16_OOM_DEBUG_ENV = "KERNELGYM_BF16_OOM_DEBUG"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def bf16_oom_debug_enabled() -> bool:
    return os.getenv(BF16_OOM_DEBUG_ENV, "").strip().lower() in _TRUE_VALUES


def _to_gib(num_bytes: Any) -> float:
    return float(num_bytes) / (1024**3)


def _stat_gib(stats: Mapping[str, Any], key: str) -> float | None:
    value = stats.get(key)
    if value is None:
        return None
    return _to_gib(value)


def format_cuda_memory_snapshot() -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"

    stats = torch.cuda.memory_stats()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes

    fields: list[tuple[str, str]] = [
        ("device", str(torch.cuda.current_device())),
        ("alloc_gb", f"{_to_gib(torch.cuda.memory_allocated()):.2f}"),
        ("reserved_gb", f"{_to_gib(torch.cuda.memory_reserved()):.2f}"),
        ("active_gb", f"{_stat_gib(stats, 'active_bytes.all.current') or 0.0:.2f}"),
        ("requested_gb", f"{_stat_gib(stats, 'requested_bytes.all.current') or 0.0:.2f}"),
        ("inactive_split_gb", f"{_stat_gib(stats, 'inactive_split_bytes.all.current') or 0.0:.2f}"),
        ("max_alloc_gb", f"{_to_gib(torch.cuda.max_memory_allocated()):.2f}"),
        ("max_reserved_gb", f"{_to_gib(torch.cuda.max_memory_reserved()):.2f}"),
        ("device_used_gb", f"{_to_gib(used_bytes):.2f}"),
        ("device_free_gb", f"{_to_gib(free_bytes):.2f}"),
        ("device_total_gb", f"{_to_gib(total_bytes):.2f}"),
        ("alloc_retries", str(int(stats.get("num_alloc_retries", 0)))),
        ("ooms", str(int(stats.get("num_ooms", 0)))),
    ]

    optional_gib_fields = (
        ("reserved_bytes.private_pool.current", "private_pool_reserved_gb"),
        ("allocated_bytes.private_pool.current", "private_pool_alloc_gb"),
        ("reserved_bytes.large_pool.current", "large_pool_reserved_gb"),
        ("allocated_bytes.large_pool.current", "large_pool_alloc_gb"),
        ("reserved_bytes.small_pool.current", "small_pool_reserved_gb"),
        ("allocated_bytes.small_pool.current", "small_pool_alloc_gb"),
    )
    for stat_key, label in optional_gib_fields:
        value = _stat_gib(stats, stat_key)
        if value is not None:
            fields.append((label, f"{value:.2f}"))

    optional_counter_fields = (
        ("oversize_allocations.current", "oversize_allocs"),
        ("oversize_segments.current", "oversize_segments"),
        ("num_sync_all_streams", "sync_all_streams"),
    )
    for stat_key, label in optional_counter_fields:
        value = stats.get(stat_key)
        if value is not None:
            fields.append((label, str(int(value))))

    return ", ".join(f"{key}={value}" for key, value in fields)


def format_nvidia_smi_compute_apps_snapshot(*, limit: int = 12, current_device_only: bool = True) -> str:
    if shutil.which("nvidia-smi") is None:
        return "nvidia_smi=unavailable"

    try:
        current_device_index = torch.cuda.current_device() if current_device_only and torch.cuda.is_available() else None
    except Exception:
        current_device_index = None

    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        uuid_to_index: dict[str, str] = {}
        for raw_line in gpu_query.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",", maxsplit=1)]
            if len(parts) != 2:
                continue
            uuid_to_index[parts[1]] = parts[0]

        apps_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return f"nvidia_smi_error={type(exc).__name__}:{exc}"

    current_pid = str(os.getpid())
    entries: list[tuple[int, str, str, str]] = []
    for raw_line in apps_query.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=3)]
        if len(parts) != 4:
            continue
        gpu_uuid, pid, process_name, used_memory = parts
        gpu_index = uuid_to_index.get(gpu_uuid, "?")
        if current_device_index is not None and gpu_index != str(current_device_index):
            continue
        try:
            used_memory_mib = int(used_memory)
        except ValueError:
            continue
        marker = "*" if pid == current_pid else ""
        entry = f"gpu={gpu_index} pid={pid}{marker} name={process_name} used_mib={used_memory_mib}"
        entries.append((used_memory_mib, entry, gpu_index, pid))

    if not entries:
        scope = f"gpu={current_device_index}" if current_device_index is not None else "all_gpus"
        return f"nvidia_smi_apps=none scope={scope}"

    entries.sort(key=lambda item: (-item[0], item[2], item[3]))
    rendered = "; ".join(entry for _, entry, _, _ in entries[:limit])
    truncated = len(entries) - min(len(entries), limit)
    if truncated > 0:
        rendered += f"; truncated={truncated}"
    return f"nvidia_smi_apps={rendered}"
