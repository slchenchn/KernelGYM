#!/usr/bin/env python3

import errno
import os
import re
import sys
from datetime import datetime, timezone
from typing import TextIO


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


# Keep the summary log focused on step-level progress and batch-selection signals.
SUMMARY_KEYWORDS = (
    "Training Progress",
    "[RolloutProgress]",
    "Initial validation metrics",
    "[Oversampling]",
    "[Buffer]",
    "Low batch size",
    "suggested_min_oversample_factor",
    "current_sample_oversample_factor",
    "Using buffered batch",
    "Save to buffer",
)

# Keep trainer metric dumps such as `step:123 - val/...` without matching arbitrary
# substrings like `acc[step:step + UNROLL]` inside sample tracebacks.
SUMMARY_REGEXES = (
    re.compile(r"^(?:\([^)]*\)\s*)?step:\d+\b"),
    re.compile(r"^(?:\([^)]*\)\s*)?\[Oversampling\]\s+Selected\s+\d+\s+of\s+\d+\s+required\s+samples\b"),
)

# Fatal lines outside dedicated worker streams stay in the summary log.
FATAL_KEYWORDS = (
    "Traceback",
    "Error executing job",
    "AssertionError",
    "RuntimeError",
    "Exception",
    "ray.exceptions.",
    "CRITICAL",
)

# Reward/env workers are noisy but still useful for debugging correctness issues, so they
# get their own stream instead of flooding the trainer summary.
REWARD_KEYWORDS = (
    "[RewardManager]",
    "[HybridClient]",
    "[HybridWorker]",
    "[KernelEvalStatus]",
    "Env Result:",
    "tool_response:",
    "preflight failed",
    "batch submitted=",
    "entry point in reward manager",
    "num_custom_kernel",
    "num_total_kernels",
    "custom_kernel_cuda_time_in_profiling_us",
    "total_kernel_run_time_in_profiling_us",
    "Decoy kernel is not found",
)

# vLLM engine logs are useful on their own when debugging engine init, wake_up, CUDA graph
# capture, and distributed executor issues. Split them from the broader rollout worker log
# so engine/runtime failures are easier to inspect without unrelated worker chatter.
VLLM_KEYWORDS = (
    "override_generation_config:",
    "Unknown vLLM environment variable detected",
    "Async scheduling will be disabled",
    "Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'",
    "Loading safetensors checkpoint shards",
    "Loading checkpoint shards",
    "Capturing CUDA graphs",
    "Enforce eager set, disabling torch.compile and CUDAGraphs",
    "Inductor compilation was disabled by user settings",
    "Invocation of wake_up method failed",
    "[LMHeadActivationCheck]",
    "[ACTOR_VLLM_SYNC_DIAG]",
)

VLLM_PREFIXES = (
    "MultiTurnAsyncvLLMEngine pid=",
    "EngineCore pid=",
)

# Rollout workers emit a mix of initialization, model-load, and runtime heartbeats. Route
# these away from `main.log` unless they also match a summary/fatal signal.
ROLLOUT_KEYWORDS = (
    "initializes with external actors",
    "intializes finished",
    "initializing ray ...",
    "Connected to Ray cluster",
    "Connecting to existing Ray cluster",
    "Using address ",
    "Actor use_remove_padding",
    "Actor use_fused_kernels",
    "Actor Sum_pi_squared computation enabled",
    "Monkey patch _flash_attention_forward",
    "Skipping monkey patch",
    "Flash Attention 2 only supports",
    "`torch_dtype` is deprecated",
    "[Gloo]",
    "NCCL version ",
    "Total steps:",
    "No CUDA runtime is found",
    "Unknown vLLM environment variable detected",
    "Async scheduling will be disabled",
    "Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'",
)

ROLLOUT_PREFIXES = ("WorkerDict pid=",)

TRAINER_PREFIXES = (
    "TaskRunner pid=",
)


def get_env_path(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def extract_prefix(line: str) -> str | None:
    clean = strip_ansi(line)
    if clean.startswith("(") and ")" in clean:
        return clean[1 : clean.index(")")]
    return None


def has_any(line: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in line for pattern in patterns)


def matches_summary(line: str) -> bool:
    return has_any(line, SUMMARY_KEYWORDS) or any(regex.search(line) for regex in SUMMARY_REGEXES)


def should_keep_in_main(clean_line: str, *, is_rollout: bool, is_reward: bool) -> bool:
    # `main.log` is intentionally sparse: keep trainer summaries, step markers, and fatal
    # diagnostics, but suppress detailed rollout/reward chatter.
    if matches_summary(clean_line):
        return True
    if is_rollout or is_reward:
        return False
    if has_any(clean_line, FATAL_KEYWORDS):
        return True
    prefix = extract_prefix(clean_line)
    if prefix is None:
        return True
    return False


def route_line(clean_line: str) -> set[str]:
    destinations: set[str] = set()

    prefix = extract_prefix(clean_line) or ""
    is_reward = has_any(clean_line, REWARD_KEYWORDS)
    is_vllm = prefix.startswith(VLLM_PREFIXES) or has_any(clean_line, VLLM_KEYWORDS)
    is_rollout = prefix.startswith(ROLLOUT_PREFIXES) or has_any(clean_line, ROLLOUT_KEYWORDS)
    is_trainer = prefix.startswith(TRAINER_PREFIXES)

    if should_keep_in_main(clean_line, is_rollout=(is_rollout or is_vllm), is_reward=is_reward):
        destinations.add("main")

    # Detailed worker logs still get split into their dedicated files even when a subset of
    # lines is also mirrored into `main.log`.
    if is_reward:
        destinations.add("reward")
    elif is_vllm:
        destinations.add("vllm")
    elif is_rollout:
        destinations.add("rollout")
    elif is_trainer:
        destinations.add("trainer")
    else:
        # Keep non-prefixed launch and summary lines readable in the summary log.
        if "main" not in destinations:
            destinations.add("trainer")

    return destinations


def timestamp_mode() -> str:
    mode = os.environ.get("LOG_ROUTER_TIMESTAMP_MODE", "prefix").strip().lower()
    return mode if mode in {"prefix", "none"} else "prefix"


def current_timestamp_prefix() -> str:
    now = datetime.now(timezone.utc).astimezone()
    return now.isoformat(timespec="seconds")


def ensure_timestamped_line(clean_line: str) -> str:
    if timestamp_mode() == "none":
        return clean_line
    return f"[{current_timestamp_prefix()}] {clean_line}"


def open_log(path: str) -> TextIO:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "a", buffering=1, encoding="utf-8")


def safe_close(handle: TextIO) -> None:
    try:
        handle.close()
    except OSError:
        # NFS-backed files can transiently return ESTALE during close after the
        # underlying inode was recycled. Best effort close is sufficient here.
        pass


def should_reopen(exc: OSError) -> bool:
    return exc.errno in {errno.ESTALE, errno.EIO, errno.ENOENT}


def reopen_writer(path: str) -> TextIO:
    last_exc: OSError | None = None
    for _ in range(2):
        try:
            return open_log(path)
        except OSError as exc:
            last_exc = exc
            if not should_reopen(exc):
                raise
    assert last_exc is not None
    raise last_exc


def write_line(key: str, clean_line: str, *, writers: dict[str, TextIO], paths: dict[str, str]) -> None:
    handle = writers[key]
    try:
        handle.write(clean_line)
        return
    except OSError as exc:
        if not should_reopen(exc):
            raise

    safe_close(handle)
    refreshed = reopen_writer(paths[key])
    writers[key] = refreshed
    refreshed.write(clean_line)


def main() -> int:
    main_log = get_env_path("MAIN_LOG", "main.log")
    trainer_log = get_env_path("TRAINER_LOG", "trainer.log")
    rollout_log = get_env_path("ROLLOUT_LOG", "rollout.log")
    reward_log = get_env_path("REWARD_LOG", "reward.log")
    vllm_log = get_env_path("VLLM_LOG", "vllm.log")

    paths = {
        "main": main_log,
        "trainer": trainer_log,
        "rollout": rollout_log,
        "reward": reward_log,
        "vllm": vllm_log,
    }
    writers = {
        key: open_log(path) for key, path in paths.items()
    }

    try:
        for raw_line in sys.stdin:
            # Preserve original stdout behavior so tmux/console monitoring still works, while
            # also writing a cleaned copy into the routed log files.
            sys.stdout.write(raw_line)
            clean_line = strip_ansi(raw_line)
            if not clean_line.strip():
                continue
            timestamped_line = ensure_timestamped_line(clean_line)
            destinations = route_line(clean_line)
            for key in destinations:
                write_line(key, timestamped_line, writers=writers, paths=paths)
    finally:
        for handle in writers.values():
            safe_close(handle)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
