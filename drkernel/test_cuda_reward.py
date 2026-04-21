"""Single-sample CUDA reward smoke test for this repository."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from kernelgym.backend.kernelbench.cuda_agent_backend import KernelBenchCudaAgentBackend
from kernelgym.toolkit.kernelbench.pipeline import eval_kernel_against_ref, eval_reference_only


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_format_cuda_agent_submission():
    module_path = Path(__file__).resolve().parent / "kernel" / "utils" / "kernel_code.py"
    spec = importlib.util.spec_from_file_location("drkernel_kernel_code", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load kernel_code helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.format_cuda_agent_submission


def _build_submission(sample_dir: Path) -> tuple[str, str]:
    format_cuda_agent_submission = _load_format_cuda_agent_submission()
    original_model_code = _read_text(sample_dir / "model.py")
    model_new_code = _read_text(sample_dir / "model_new.py")
    cuda_kernel_code = _read_text(sample_dir / "kernels" / "axpby.cu")
    binding_code = _read_text(sample_dir / "kernels" / "axpby_binding.cpp")

    submission = format_cuda_agent_submission(
        {
            "CUDA_KERNELS": cuda_kernel_code,
            "APPLY_BINDINGS": binding_code,
            "MODEL_NEW": model_new_code,
        }
    )
    if submission is None:
        raise ValueError(f"Failed to build CUDA submission from {sample_dir}")

    return original_model_code, submission


def _to_result_dict(reference_result, kernel_result, *, backend: str, sample_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_dir": str(sample_dir),
        "backend": backend,
        "status": "failed",
        "compiled": False,
        "correctness": False,
        "decoy_kernel": False,
        "reference_runtime_ms": getattr(reference_result, "runtime", 0.0) or 0.0,
        "kernel_runtime_ms": 0.0,
        "speedup": 0.0,
        "metadata": {},
        "error_message": None,
    }

    if kernel_result is None:
        result["error_message"] = "Evaluation returned None"
        return result

    result["compiled"] = bool(kernel_result.compiled)
    result["metadata"] = dict(kernel_result.metadata or {})
    if not kernel_result.compiled:
        result["error_message"] = str(result["metadata"].get("compilation_error", "Unknown"))
        return result

    result["correctness"] = bool(kernel_result.correctness)
    if not kernel_result.correctness:
        result["error_message"] = str(
            result["metadata"].get("runtime_error")
            or result["metadata"].get("correctness_issue")
            or "Correctness check failed"
        )
        return result

    result["status"] = "completed"
    result["decoy_kernel"] = bool(getattr(kernel_result, "decoy_kernel", False))
    result["kernel_runtime_ms"] = float(getattr(kernel_result, "runtime", 0.0) or 0.0)
    if result["kernel_runtime_ms"] > 0:
        result["speedup"] = result["reference_runtime_ms"] / result["kernel_runtime_ms"]
    return result


def evaluate_sample(
    sample_dir: Path,
    *,
    device: int,
    seed: int,
    num_correct_trials: int,
    num_perf_trials: int,
    num_warmup: int,
    perf_trim_count: int,
    verbose: bool,
) -> dict[str, Any]:
    original_model_code, submission = _build_submission(sample_dir)

    reference_result = eval_reference_only(
        original_model_src=original_model_code,
        seed_num=seed,
        num_perf_trials=max(1, min(num_perf_trials, 10)),
        verbose=verbose,
        device=device,
        entry_point="Model",
        reference_backend=None,
    )
    if not reference_result or not reference_result.compiled:
        raise RuntimeError(
            f"Reference evaluation failed: {getattr(reference_result, 'metadata', {})}"
        )

    backend_adapter = KernelBenchCudaAgentBackend()
    kernel_result = eval_kernel_against_ref(
        original_model_src=original_model_code,
        custom_model_src=submission,
        seed_num=seed,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        num_warmup=num_warmup,
        perf_trim_count=perf_trim_count,
        verbose=verbose,
        measure_performance=True,
        build_dir=None,
        device=device,
        backend="cuda_agent",
        entry_point="Model",
        enable_profiling=True,
        enable_triton_detection=False,
        backend_adapter=backend_adapter,
    )
    return _to_result_dict(
        reference_result,
        kernel_result,
        backend="cuda_agent",
        sample_dir=sample_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path("/nfs/FM/gongoubo/cuda_kernel/datasets/0"),
        help="Directory containing model.py, model_new.py, and kernels/*",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=20)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--perf-trim-count", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = evaluate_sample(
        args.sample_dir,
        device=args.device,
        seed=args.seed,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        num_warmup=args.num_warmup,
        perf_trim_count=args.perf_trim_count,
        verbose=not args.quiet,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
