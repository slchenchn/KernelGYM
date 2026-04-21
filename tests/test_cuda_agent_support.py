from __future__ import annotations

import asyncio
import enum
import importlib.util
import numpy as np
import os
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "cuda_agent_support"
sys.path.insert(0, str(ROOT / "drkernel"))

if "verl" not in sys.modules:
    verl_stub = types.ModuleType("verl")

    class DataProto:
        pass

    verl_stub.DataProto = DataProto
    sys.modules["verl"] = verl_stub

from kernel.metrics.kernel_multi_turn_metrics import compute_kernel_multi_turn_metrics
from kernel.metrics.mismatch_quality_metrics import compute_mismatch_quality_metrics
from kernel.event_logging import truncate_feedback_for_prompt
from kernel.utils.kernel_code import extract_kernel_submission
from kernelgym.backend.kernelbench.cuda_agent_backend import KernelBenchCudaAgentBackend
from kernelgym.schema.task import EvaluationTask
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult
from kernelgym.toolkit.kernelbench.pipeline import _apply_coverage_metadata
from kernelgym.toolkit.kernelbench.profiling import compute_named_kernel_coverage
from kernelgym.toolkit.kernelbench.toolkit import KernelBenchToolkit
from kernelgym.toolkit.validation import precheck_cuda_agent_submission
from kernelgym.workflow.kernelbench import KernelBenchWorkflowController
from kernelgym.workflow.kernelbench_helpers import _create_paired_tasks, set_reference_cache


class DummyTokenizer:
    pad_token_id = -1

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, str):
            return token_ids
        return "".join(chr(token_id) for token_id in token_ids)


def _load_kernel_agent_class():
    base_agent_mod = types.ModuleType("verl_patch.workers.code.agent.base_agent")

    class BaseAgent:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    base_agent_mod.BaseAgent = BaseAgent
    sys.modules["verl_patch.workers.code.agent.base_agent"] = base_agent_mod

    env_mod = types.ModuleType("verl_patch.workers.code.agent_env.base_env")

    class FinishReasonTypeEnum(enum.Enum):
        ANSWER = "answer"
        NO_TOOL_CALL = "no_tool_call"

    env_mod.FinishReasonTypeEnum = FinishReasonTypeEnum
    sys.modules["verl_patch.workers.code.agent_env.base_env"] = env_mod

    agent_path = ROOT / "drkernel" / "kernel" / "workers" / "agent" / "kernel_agent.py"
    spec = importlib.util.spec_from_file_location("kernel_agent_direct", agent_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.KernelAgent


def _load_reward_client_class():
    sys.path.insert(0, str(ROOT / "drkernel"))

    ray_mod = types.ModuleType("ray")

    def _remote(obj=None, **_kwargs):
        if obj is None:
            return lambda cls: cls
        return obj

    ray_mod.remote = _remote
    ray_mod.get = lambda value: value
    ray_mod.wait = lambda refs, num_returns=1, timeout=None: (refs[:num_returns], refs[num_returns:])
    ray_mod.ObjectRef = object
    sys.modules["ray"] = ray_mod

    verl_tools_mod = types.ModuleType("verl.tools")
    sys.modules["verl.tools"] = verl_tools_mod

    sandbox_mod = types.ModuleType("verl.tools.sandbox_fusion_tools")

    class TokenBucketWorker:
        @classmethod
        def options(cls, **_kwargs):
            return cls

        @staticmethod
        def remote(*_args, **_kwargs):
            return object()

    sandbox_mod.TokenBucketWorker = TokenBucketWorker
    sys.modules["verl.tools.sandbox_fusion_tools"] = sandbox_mod

    reward_client_path = ROOT / "drkernel" / "kernel" / "rewards" / "reward_client.py"
    spec = importlib.util.spec_from_file_location("reward_client_direct", reward_client_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.KernelRewardClient


def _make_reward_client_dummy(
    *,
    penalty_score: float = -1.0,
    precheck_fail: float = -0.5,
    compilation_fail: float = -0.25,
    correctness_fail: float = -0.3,
    perf_degrade: float = -0.1,
    apply_compilation_fail_penalty: bool = False,
    apply_precheck_fail_penalty: bool = False,
    coverage_enable: bool = True,
    coverage_weight: float = 0.5,
    reward_type: str = "time_coverage",
    speedup_eps: float = 0.01,
    init_correct_weight: float = 1.0,
    init_performance_weight: float = 1.0,
):
    KernelRewardClient = _load_reward_client_class()
    dummy = SimpleNamespace(
        penalty_score=penalty_score,
        speedup_eps=speedup_eps,
        init_correct_weight=init_correct_weight,
        init_performance_weight=init_performance_weight,
        apply_compilation_fail_penalty=apply_compilation_fail_penalty,
        apply_precheck_fail_penalty=apply_precheck_fail_penalty,
        reward_config=SimpleNamespace(
            coverage_reward=SimpleNamespace(
                enable=coverage_enable,
                weight=coverage_weight,
                reward_type=reward_type,
            ),
            reward_policy=SimpleNamespace(
                penalties={
                    "penalty_score": penalty_score,
                    "precheck_fail": precheck_fail,
                    "compilation_fail": compilation_fail,
                    "correctness_fail": correctness_fail,
                    "perf_degrade": perf_degrade,
                }
            ),
        ),
    )
    dummy.compute_coverage_reward = (
        lambda result: KernelRewardClient.compute_coverage_reward(dummy, result)
    )
    dummy._normalize_failed_error = (
        lambda result: KernelRewardClient._normalize_failed_error(dummy, result)
    )
    dummy._get_compilation_fail_penalty = (
        lambda: KernelRewardClient._get_compilation_fail_penalty(dummy)
    )
    dummy._get_precheck_fail_penalty = (
        lambda: KernelRewardClient._get_precheck_fail_penalty(dummy)
    )
    dummy._resolve_failure_reward = (
        lambda error_message: KernelRewardClient._resolve_failure_reward(dummy, error_message)
    )
    return KernelRewardClient, dummy


def _load_evaluation_request_class():
    models_path = ROOT / "kernelgym" / "server" / "api" / "models.py"
    spec = importlib.util.spec_from_file_location("kernelgym_api_models_direct", models_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.EvaluationRequest


def _load_async_kernel_reward_manager_class():
    verl_mod = types.ModuleType("verl")
    verl_mod.DataProto = object
    sys.modules["verl"] = verl_mod

    reward_score_mod = types.ModuleType("verl.utils.reward_score")
    reward_score_mod.default_compute_score = lambda *args, **kwargs: {}
    sys.modules["verl.utils.reward_score"] = reward_score_mod

    reward_manager_mod = types.ModuleType("verl.workers.reward_manager")
    reward_manager_mod.register = lambda *_args, **_kwargs: (lambda cls: cls)
    sys.modules["verl.workers.reward_manager"] = reward_manager_mod

    kernel_async_path = ROOT / "drkernel" / "kernel" / "workers" / "reward_manager" / "kernel_async.py"
    spec = importlib.util.spec_from_file_location("kernel_async_direct", kernel_async_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.AsyncKernelRewardManager


def _load_coverage_helper_module():
    if "verl" not in sys.modules:
        verl_mod = types.ModuleType("verl")
        sys.modules["verl"] = verl_mod
    else:
        verl_mod = sys.modules["verl"]

    verl_utils_mod = types.ModuleType("verl.utils")
    sys.modules["verl.utils"] = verl_utils_mod
    verl_mod.utils = verl_utils_mod

    torch_functional_mod = types.ModuleType("verl.utils.torch_functional")

    def masked_mean(values, mask, axis=None):
        values = values.float()
        mask = mask.float()
        if axis is None:
            denom = mask.sum().clamp(min=1.0)
            return (values * mask).sum() / denom
        denom = mask.sum(dim=axis).clamp(min=1.0)
        return (values * mask).sum(dim=axis) / denom

    torch_functional_mod.masked_mean = masked_mean
    sys.modules["verl.utils.torch_functional"] = torch_functional_mod
    verl_utils_mod.torch_functional = torch_functional_mod

    coverage_helper_path = ROOT / "drkernel" / "kernel" / "rewards" / "coverage_helper.py"
    spec = importlib.util.spec_from_file_location("coverage_helper_direct", coverage_helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _fixture_cuda_sources(
    *,
    kernel_fixture: str = "kernels_generated.cu",
    binding_fixture: str = "kernels_generated_binding.cpp",
    include_kernel: bool = True,
) -> dict[str, str]:
    sources = {
        "kernels/generated_binding.cpp": _fixture_text(binding_fixture),
    }
    if include_kernel:
        sources["kernels/generated.cu"] = _fixture_text(kernel_fixture)
    return sources


def test_extract_kernel_submission_preserves_cuda_sections():
    extracted = extract_kernel_submission(
        _fixture_text("response_valid.md"),
        kernel_backend="cuda_agent",
    )
    assert extracted.startswith("### CUDA_KERNELS")
    assert "### APPLY_BINDINGS" in extracted
    assert "### MODEL_NEW" in extracted


def test_precheck_cuda_agent_submission_accepts_valid_bundle():
    error, error_code, info = precheck_cuda_agent_submission(
        model_code=_fixture_text("model_new_valid.py"),
        cuda_sources=_fixture_cuda_sources(),
    )
    assert error == ""
    assert error_code is None
    assert info["passed"] is True


def test_precheck_cuda_agent_submission_rejects_missing_cuda_extension_reference():
    error, error_code, info = precheck_cuda_agent_submission(
        model_code=_fixture_text("model_new_missing_cuda_extension.py"),
        cuda_sources=_fixture_cuda_sources(),
    )
    assert info["passed"] is False
    assert error_code is not None
    assert "cuda_extension" in error


def test_precheck_cuda_agent_submission_rejects_missing_cu_file():
    error, error_code, info = precheck_cuda_agent_submission(
        model_code=_fixture_text("model_new_valid.py"),
        cuda_sources=_fixture_cuda_sources(include_kernel=False),
    )
    assert info["passed"] is False
    assert error_code is not None
    assert ".cu file" in error


def test_cuda_agent_backend_extracts_custom_kernel_names():
    kernel_names = KernelBenchCudaAgentBackend._extract_custom_kernel_names(
        {"kernels/generated.cu": _fixture_text("kernels_custom_names.cu")}
    )
    assert kernel_names == ["noop_kernel", "noop_kernel_vec"]


def test_cuda_agent_backend_scaffold_keeps_binding_registry_header_only(tmp_path):
    backend = KernelBenchCudaAgentBackend()
    backend._write_runtime_scaffold(tmp_path, _fixture_text("model_new_valid.py"))

    source_names = {
        path.relative_to(tmp_path).as_posix()
        for path in map(Path, backend._collect_compile_sources(tmp_path))
    }

    assert "binding.cpp" in source_names
    assert "binding_registry.cpp" not in source_names
    assert "void applyBindings(pybind11::module& m)" in (
        tmp_path / "binding_registry.h"
    ).read_text(encoding="utf-8")


def test_cuda_agent_backend_uses_configured_tmpdir_parent(tmp_path):
    backend = KernelBenchCudaAgentBackend()
    env_name = "KERNELGYM_CUDA_AGENT_TMPDIR"
    previous = os.environ.get(env_name)
    work_dir = None
    os.environ[env_name] = str(tmp_path)
    try:
        work_dir = backend._create_work_dir()
        assert work_dir.parent == tmp_path
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)


def test_compute_named_kernel_coverage_matches_cuda_kernel_names():
    coverage = compute_named_kernel_coverage(
        ["noop_kernel", "noop_kernel_vec"],
        {
            "kernels": [
                {"name": "void noop_kernel_vec<float>(float*)", "cuda_time_us": 8.0, "cpu_time_us": 1.0, "count": 4},
                {"name": "cudaLaunchKernel", "cuda_time_us": 2.0, "cpu_time_us": 3.0, "count": 4},
            ]
        },
    )
    assert coverage["num_custom_kernels"] == 1
    assert coverage["num_total_kernels"] == 2
    assert coverage["custom_kernel_cuda_time_in_profiling_us"] == 8.0
    assert coverage["total_kernel_run_time_in_profiling_us"] == 10.0


def test_compute_named_kernel_coverage_returns_zero_when_no_names_match():
    coverage = compute_named_kernel_coverage(
        ["missing_kernel"],
        {
            "kernels": [
                {"name": "aten::abs", "cuda_time_us": 4.0, "cpu_time_us": 1.0, "count": 2},
                {"name": "cudaLaunchKernel", "cuda_time_us": 2.0, "cpu_time_us": 3.0, "count": 2},
            ]
        },
    )
    assert coverage["num_custom_kernels"] == 0
    assert coverage["num_total_kernels"] == 2
    assert coverage["custom_kernel_cuda_time_in_profiling_us"] == 0.0
    assert coverage["total_kernel_run_time_in_profiling_us"] == 6.0
    assert coverage["custom_kernels_not_in_profiling"] == ["missing_kernel"]


def test_precheck_cuda_agent_submission_rejects_missing_binding_registration():
    error, error_code, info = precheck_cuda_agent_submission(
        model_code=_fixture_text("model_new_valid.py"),
        cuda_sources=_fixture_cuda_sources(
            binding_fixture="kernels_generated_binding_missing_registration.cpp"
        ),
    )
    assert info["passed"] is False
    assert error_code is not None
    assert "REGISTER_BINDING" in error


def test_precheck_cuda_agent_submission_rejects_missing_binding_semicolon():
    error, error_code, info = precheck_cuda_agent_submission(
        model_code=_fixture_text("model_new_valid.py"),
        cuda_sources=_fixture_cuda_sources(
            binding_fixture="kernels_generated_binding_missing_semicolon.cpp"
        ),
    )
    assert info["passed"] is False
    assert error_code is not None
    assert "trailing ';'" in error


def test_evaluation_request_accepts_detect_decoy_kernel():
    EvaluationRequest = _load_evaluation_request_class()
    request = EvaluationRequest(
        task_id="task_1",
        reference_code=_fixture_text("reference_identity_model.py"),
        kernel_code=_fixture_text("kernel_identity_model_new.py"),
        backend="cuda_agent",
        detect_decoy_kernel=False,
    )
    payload = request.dict()
    assert payload["detect_decoy_kernel"] is False
    assert payload["backend"] == "cuda_agent"


def test_create_paired_tasks_preserves_detect_decoy_kernel_for_kernel_task():
    task = EvaluationTask(
        task_id="task_1",
        reference_code=_fixture_text("reference_minimal_model.py"),
        kernel_code=_fixture_text("kernel_minimal_model_new.py"),
        backend="cuda_agent",
        num_warmup=7,
        perf_trim_count=2,
        detect_decoy_kernel=False,
        use_reference_cache=False,
    )
    reference_task, kernel_task = _create_paired_tasks(task)
    assert reference_task is not None
    assert reference_task.num_warmup == 7
    assert reference_task.perf_trim_count == 2
    assert kernel_task.detect_decoy_kernel is False
    assert kernel_task.backend == "cuda_agent"
    assert kernel_task.num_warmup == 7
    assert kernel_task.perf_trim_count == 2


def test_workflow_reference_cache_fallback_preserves_timing_controls():
    class FlakyReferenceCache:
        def __init__(self):
            self.get_calls = 0
            self.put_values = []

        def get(self, uuid, reference_code, is_valid):
            self.get_calls += 1
            return 1.23 if self.get_calls == 1 else None

        def put(self, uuid, reference_code, is_valid, runtime):
            self.put_values.append((uuid, is_valid, runtime))

    class FakeScheduler:
        def __init__(self):
            self.submitted = []

        async def submit(self, spec):
            self.submitted.append(spec)
            return f"task_{len(self.submitted)}"

        async def wait(self, task_id):
            if task_id == "task_1":
                return {
                    "task_id": "eval_1_kernel",
                    "base_task_id": "eval_1",
                    "compiled": True,
                    "correctness": True,
                    "decoy_kernel": False,
                    "kernel_runtime": 1.0,
                    "metadata": {},
                    "status": "completed",
                }
            return {
                "task_id": "eval_1_ref",
                "base_task_id": "eval_1",
                "reference_runtime": 2.0,
                "metadata": {},
                "status": "completed",
            }

    cache = FlakyReferenceCache()
    scheduler = FakeScheduler()
    set_reference_cache(cache)
    try:
        result = asyncio.run(
            KernelBenchWorkflowController().handle_request(
                {
                    "task_id": "eval_1",
                    "reference_code": _fixture_text("reference_minimal_model.py"),
                    "kernel_code": _fixture_text("kernel_minimal_model_new.py"),
                    "backend": "cuda_agent",
                    "num_warmup": 7,
                    "perf_trim_count": 2,
                    "use_reference_cache": True,
                    "uuid": "sample_1",
                },
                scheduler,
            )
        )
    finally:
        set_reference_cache(None)

    assert result["status"] == "completed"
    assert len(scheduler.submitted) == 2
    reference_payload = scheduler.submitted[1].payload
    assert reference_payload["task_type"] == "reference_timing"
    assert reference_payload["num_warmup"] == 7
    assert reference_payload["perf_trim_count"] == 2
    assert cache.put_values == [("sample_1", False, 2.0)]


def test_truncate_feedback_for_prompt_respects_middle_and_side_modes():
    text = "0123456789" * 20

    unchanged = truncate_feedback_for_prompt(text, 0, "middle")
    assert unchanged == text

    middle = truncate_feedback_for_prompt(text, 80, "middle")
    assert len(middle) <= 80
    assert middle.startswith("012345")
    assert middle.endswith("456789")
    assert "total_chars=200" in middle

    left = truncate_feedback_for_prompt(text, 60, "left")
    assert len(left) <= 60
    assert left.endswith("456789")
    assert "total_chars=200" in left

    right = truncate_feedback_for_prompt(text, 60, "right")
    assert len(right) <= 60
    assert right.startswith("012345")
    assert "total_chars=200" in right


def test_kernelbench_toolkit_resolve_eval_flags_defaults_and_overrides():
    toolkit = KernelBenchToolkit()
    cuda_task = SimpleNamespace(
        run_correctness=None,
        run_triton_detection=None,
        enable_triton_detection=None,
        backend="cuda_agent",
        run_performance=None,
        measure_performance=None,
        detect_decoy_kernel=None,
    )
    assert toolkit._resolve_eval_flags(cuda_task) == (True, False, True, True)

    triton_task = SimpleNamespace(
        run_correctness=None,
        run_triton_detection=None,
        enable_triton_detection=None,
        backend="triton",
        run_performance=None,
        measure_performance=None,
        detect_decoy_kernel=False,
    )
    assert toolkit._resolve_eval_flags(triton_task) == (True, True, True, False)


def test_apply_coverage_metadata_sets_cuda_coverage_without_decoy():
    metadata = {}
    result = KernelExecResult(compiled=True, correctness=True, metadata={})
    _apply_coverage_metadata(
        metadata=metadata,
        kernel_exec_result=result,
        coverage_result_dict={
            "num_custom_kernels": 1,
            "num_total_kernels": 2,
            "custom_kernels_not_in_profiling": [],
            "custom_kernels_in_profiling": ["noop_kernel"],
            "total_kernel_run_time_in_profiling_us": 10.0,
            "total_kernel_cuda_time_in_profiling_us": 10.0,
            "total_kernel_run_time_in_profiling_us_cpu_cuda": 12.0,
            "custom_kernel_cuda_time_in_profiling_us": 8.0,
        },
        coverage_backend="cuda_agent",
        detect_decoy_kernel=True,
    )
    assert metadata["coverage_backend"] == "cuda_agent"
    assert metadata["num_custom_kernels"] == 1
    assert metadata["custom_kernel_cuda_time_in_profiling_us"] == 8.0
    assert result.decoy_kernel is False


def test_apply_coverage_metadata_marks_decoy_for_unmatched_cuda_kernels():
    metadata = {}
    result = KernelExecResult(compiled=True, correctness=True, metadata={})
    _apply_coverage_metadata(
        metadata=metadata,
        kernel_exec_result=result,
        coverage_result_dict={
            "num_custom_kernels": 0,
            "num_total_kernels": 3,
            "custom_kernels_not_in_profiling": ["noop_kernel"],
            "custom_kernels_in_profiling": [],
            "total_kernel_run_time_in_profiling_us": 9.0,
            "total_kernel_cuda_time_in_profiling_us": 9.0,
            "total_kernel_run_time_in_profiling_us_cpu_cuda": 11.0,
            "custom_kernel_cuda_time_in_profiling_us": 0.0,
        },
        coverage_backend="cuda_agent",
        detect_decoy_kernel=True,
    )
    assert result.decoy_kernel is True
    assert metadata["num_total_kernels"] == 3


def test_apply_coverage_metadata_can_disable_decoy_penalty():
    metadata = {}
    result = KernelExecResult(compiled=True, correctness=True, metadata={})
    _apply_coverage_metadata(
        metadata=metadata,
        kernel_exec_result=result,
        coverage_result_dict={
            "num_custom_kernels": 0,
            "num_total_kernels": 3,
            "custom_kernels_not_in_profiling": ["noop_kernel"],
            "custom_kernels_in_profiling": [],
            "total_kernel_run_time_in_profiling_us": 9.0,
            "total_kernel_cuda_time_in_profiling_us": 9.0,
            "total_kernel_run_time_in_profiling_us_cpu_cuda": 11.0,
            "custom_kernel_cuda_time_in_profiling_us": 0.0,
        },
        coverage_backend="cuda_agent",
        detect_decoy_kernel=False,
    )
    assert result.decoy_kernel is False


def test_apply_coverage_metadata_does_not_mark_decoy_when_profiler_captures_nothing():
    metadata = {}
    result = KernelExecResult(compiled=True, correctness=True, metadata={})
    _apply_coverage_metadata(
        metadata=metadata,
        kernel_exec_result=result,
        coverage_result_dict={
            "num_custom_kernels": 0,
            "num_total_kernels": 0,
            "custom_kernels_not_in_profiling": ["noop_kernel"],
            "custom_kernels_in_profiling": [],
            "total_kernel_run_time_in_profiling_us": 0.0,
            "total_kernel_cuda_time_in_profiling_us": 0.0,
            "total_kernel_run_time_in_profiling_us_cpu_cuda": 0.0,
            "custom_kernel_cuda_time_in_profiling_us": 0.0,
        },
        coverage_backend="cuda_agent",
        detect_decoy_kernel=True,
    )
    assert result.decoy_kernel is False
    assert result.metadata["num_total_kernels"] == 0
    assert metadata["custom_kernel_not_in_profiling"] == ["noop_kernel"]


def test_apply_coverage_metadata_sets_triton_legacy_fields():
    metadata = {}
    result = KernelExecResult(compiled=True, correctness=True, metadata={})
    _apply_coverage_metadata(
        metadata=metadata,
        kernel_exec_result=result,
        coverage_result_dict={
            "num_custom_kernels": 1,
            "num_total_kernels": 1,
            "custom_kernels_not_in_profiling": [],
            "custom_kernels_in_profiling": ["triton_kernel"],
            "total_kernel_run_time_in_profiling_us": 7.0,
            "total_kernel_cuda_time_in_profiling_us": 7.0,
            "total_kernel_run_time_in_profiling_us_cpu_cuda": 9.0,
            "custom_kernel_cuda_time_in_profiling_us": 7.0,
        },
        coverage_backend="triton",
        detect_decoy_kernel=True,
    )
    assert metadata["triton_kernel_coverage"].startswith("Run 1 custom kernels")
    assert metadata["triton_kernel_in_profiling"] == ["triton_kernel"]
    assert result.metadata["triton_kernel_coverage"] == metadata["triton_kernel_coverage"]


def test_reward_client_compute_coverage_reward_reads_cuda_metadata():
    KernelRewardClient = _load_reward_client_class()
    dummy = SimpleNamespace(
        reward_config=SimpleNamespace(
            coverage_reward=SimpleNamespace(reward_type="time_coverage")
        )
    )
    coverage = KernelRewardClient.compute_coverage_reward(
        dummy,
        {
            "metadata": {
                "num_custom_kernels": 1,
                "num_total_kernels": 2,
                "custom_kernel_cuda_time_in_profiling_us": 8.0,
                "total_kernel_cuda_time_in_profiling_us": 10.0,
                "total_kernel_run_time_in_profiling_us_cpu_cuda": 12.0,
            }
        },
    )
    assert coverage["coverage"] == 0.8
    assert coverage["num_custom_kernel"] == 1
    assert coverage["num_total_kernels"] == 2


def test_reward_client_compute_coverage_reward_falls_back_to_runtime_total_field():
    KernelRewardClient = _load_reward_client_class()
    dummy = SimpleNamespace(
        reward_config=SimpleNamespace(
            coverage_reward=SimpleNamespace(reward_type="time_coverage")
        )
    )
    coverage = KernelRewardClient.compute_coverage_reward(
        dummy,
        {
            "metadata": {
                "num_custom_kernels": 1,
                "num_total_kernels": 4,
                "custom_kernel_cuda_time_in_profiling_us": 3.0,
                "total_kernel_run_time_in_profiling_us": 6.0,
            }
        },
    )
    assert coverage["coverage"] == 0.5
    assert coverage["total_kernel_cuda_time_in_profiling_us"] == 6.0


def test_reward_client_compute_coverage_reward_supports_number_coverage():
    KernelRewardClient = _load_reward_client_class()
    dummy = SimpleNamespace(
        reward_config=SimpleNamespace(
            coverage_reward=SimpleNamespace(reward_type="number_coverage")
        )
    )
    coverage = KernelRewardClient.compute_coverage_reward(
        dummy,
        {
            "num_custom_kernel": 2,
            "num_total_kernels": 5,
            "custom_kernel_cuda_time_in_profiling_us": 4.0,
            "total_kernel_run_time_in_profiling_us": 10.0,
        },
    )
    assert coverage["coverage"] == 0.4


def test_reward_client_calculate_reward_weighted_penalizes_decoy_kernel():
    KernelRewardClient, dummy = _make_reward_client_dummy()

    reward = KernelRewardClient.calculate_reward_weighted(
        dummy,
        {
            "status": "completed",
            "compiled": True,
            "correctness": True,
            "speedup": 2.0,
            "decoy_kernel": True,
        },
    )
    assert reward["reward"] == -1.0
    assert reward["decoy_kernel"] is True
    assert reward["correctness"] is False
    assert reward["compiled"] is False


def test_reward_client_calculate_reward_weighted_gives_higher_reward_to_better_fusion_than_lazy_optimization():
    KernelRewardClient, dummy = _make_reward_client_dummy()

    lazy_result = KernelRewardClient.calculate_reward_weighted(
        dummy,
        {
            "status": "completed",
            "compiled": True,
            "correctness": True,
            "speedup": 1.02,
            "metadata": {
                "num_custom_kernels": 1,
                "num_total_kernels": 5,
                "custom_kernel_cuda_time_in_profiling_us": 0.014,
                "total_kernel_cuda_time_in_profiling_us": 100.0,
                "total_kernel_run_time_in_profiling_us_cpu_cuda": 120.0,
            },
        },
    )
    better_fusion_result = KernelRewardClient.calculate_reward_weighted(
        dummy,
        {
            "status": "completed",
            "compiled": True,
            "correctness": True,
            "speedup": 1.02,
            "metadata": {
                "num_custom_kernels": 3,
                "num_total_kernels": 5,
                "custom_kernel_cuda_time_in_profiling_us": 86.15,
                "total_kernel_cuda_time_in_profiling_us": 100.0,
                "total_kernel_run_time_in_profiling_us_cpu_cuda": 120.0,
            },
        },
    )

    assert lazy_result["reward"] < better_fusion_result["reward"]
    assert lazy_result["num_custom_kernel"] == 1
    assert better_fusion_result["num_custom_kernel"] == 3
    assert lazy_result["reward"] == 2.00007
    assert better_fusion_result["reward"] == 2.43075


def test_reward_client_calculate_reward_weighted_maps_precheck_fail_to_precheck_penalty():
    KernelRewardClient, dummy = _make_reward_client_dummy(
        apply_precheck_fail_penalty=True,
        precheck_fail=-0.75,
        compilation_fail=-0.25,
        penalty_score=-1.0,
    )

    reward = KernelRewardClient.calculate_reward_weighted(
        dummy,
        {
            "status": "failed",
            "error_message": "Precheck failed: model_new.py must import or reference cuda_extension",
        },
    )

    assert reward["reward"] == -0.75
    assert reward["error"] == "Task failed: code pre-check error"


def test_reward_client_calculate_reward_weighted_maps_compilation_fail_to_compilation_penalty():
    KernelRewardClient, dummy = _make_reward_client_dummy(
        apply_compilation_fail_penalty=True,
        precheck_fail=-0.75,
        compilation_fail=-0.25,
        penalty_score=-1.0,
    )

    reward = KernelRewardClient.calculate_reward_weighted(
        dummy,
        {
            "status": "failed",
            "error_message": "Kernel compilation failed: nvcc fatal   : Unsupported gpu architecture 'compute_999'",
        },
    )

    assert reward["reward"] == -0.25
    assert reward["error"] == "Task failed due to kernel compilation error"


def test_reward_client_calculate_reward_weighted_uses_generic_penalty_for_other_failures():
    KernelRewardClient, dummy = _make_reward_client_dummy(
        apply_precheck_fail_penalty=True,
        apply_compilation_fail_penalty=True,
        precheck_fail=-0.75,
        compilation_fail=-0.25,
        penalty_score=-1.0,
    )

    reward = KernelRewardClient.calculate_reward_weighted(
        dummy,
        {
            "status": "timeout",
            "error_message": "Task timeout after 600s (client-side)",
        },
    )

    assert reward["reward"] == -1.0
    assert reward["error"] == "Task timeout after 600s (client-side)"


def test_reward_client_calculate_reward_weighted_keeps_generic_penalty_when_fail_type_penalties_disabled():
    KernelRewardClient, dummy = _make_reward_client_dummy(
        apply_precheck_fail_penalty=False,
        apply_compilation_fail_penalty=False,
        precheck_fail=-0.75,
        compilation_fail=-0.25,
        penalty_score=-1.0,
    )

    reward = KernelRewardClient.calculate_reward_weighted(
        dummy,
        {
            "status": "failed",
            "error_message": "Precheck failed: CUDA-Agent sources must include at least one .cu file",
        },
    )

    assert reward["reward"] == -1.0
    assert reward["error"] == "Task failed: code pre-check error"


def test_coverage_rejection_mask_filters_lazy_correct_sample_but_keeps_better_fusion_and_incorrect_sample():
    coverage_helper = _load_coverage_helper_module()
    response_mask = torch.ones((3, 4), dtype=torch.float32)
    correctness = torch.tensor([True, True, False], dtype=torch.bool)

    modified_mask, metrics = coverage_helper.compute_coverage_rejection_mask(
        time_coverage=torch.tensor([0.00014, 0.8615, 0.00014], dtype=torch.float32),
        num_coverage=torch.tensor([0.2, 0.6, 0.2], dtype=torch.float32),
        response_mask=response_mask,
        correctness=correctness,
        max_turns=1,
        coverage_rs="turn",
        coverage_rs_key="time_coverage",
        coverage_rs_threshold=0.3,
        coverage_rs_factor=0.0,
    )

    assert torch.equal(modified_mask[0], torch.zeros(4))
    assert torch.equal(modified_mask[1], torch.ones(4))
    assert torch.equal(modified_mask[2], torch.ones(4))
    assert abs(metrics["coverage/coverage_rs_correct_only_masked_fraction"] - 0.5) < 1e-6
    assert abs(metrics["coverage/coverage_rs_masked_fraction"] - (1.0 / 3.0)) < 1e-6


def test_coverage_rejection_mask_speedup_escape_hatch_keeps_low_coverage_sample():
    coverage_helper = _load_coverage_helper_module()
    response_mask = torch.ones((1, 4), dtype=torch.float32)
    correctness = torch.tensor([True], dtype=torch.bool)

    modified_mask, metrics = coverage_helper.compute_coverage_rejection_mask(
        time_coverage=torch.tensor([0.00014], dtype=torch.float32),
        num_coverage=torch.tensor([0.2], dtype=torch.float32),
        response_mask=response_mask,
        correctness=correctness,
        max_turns=1,
        coverage_rs="turn",
        coverage_rs_key="time_coverage",
        coverage_rs_threshold=0.3,
        coverage_rs_factor=0.0,
        speedup=torch.tensor([1.8], dtype=torch.float32),
        speedup_threshold=1.5,
    )

    assert torch.equal(modified_mask[0], torch.ones(4))
    assert metrics["coverage/coverage_rs_masked_fraction"] == 0.0


def test_coverage_rejection_mask_geometric_mode_masks_entire_lazy_sequence():
    coverage_helper = _load_coverage_helper_module()
    response_mask = torch.ones((2, 4), dtype=torch.float32)
    correctness = torch.tensor([True, True], dtype=torch.bool)

    modified_mask, metrics = coverage_helper.compute_coverage_rejection_mask(
        time_coverage=torch.tensor([[0.00014, 0.8615]], dtype=torch.float32),
        num_coverage=torch.tensor([[0.2, 0.6]], dtype=torch.float32),
        response_mask=response_mask,
        correctness=correctness,
        max_turns=2,
        coverage_rs="geometric",
        coverage_rs_key="time_coverage",
        coverage_rs_threshold=0.3,
        coverage_rs_factor=0.0,
    )

    assert torch.equal(modified_mask, torch.zeros_like(response_mask))
    assert metrics["coverage/coverage_rs_seq_masked_fraction"] == 1.0
    assert metrics["coverage/coverage_rs_correct_only_masked_fraction"] == 1.0


def test_async_reward_manager_writes_both_decoy_keys():
    AsyncKernelRewardManager = _load_async_kernel_reward_manager_class()
    reward_config = SimpleNamespace(
        server_url="http://example",
        reward_policy=SimpleNamespace(
            penalties=SimpleNamespace(penalty_score=-1.0)
        ),
        reward_weights={},
        task_timeout=30,
        speedup_reward_upper_bound=10.0,
        speedup_eps=0.01,
    )
    manager = AsyncKernelRewardManager(
        tokenizer=None,
        reward_config=reward_config,
    )
    manager.execute_env = lambda *args, **kwargs: [
        {
            "score": 0.5,
            "reward": 0.5,
            "correctness": True,
            "success": True,
            "compiled": True,
            "speedup": 1.2,
            "status": "completed",
            "decoy_kernel": True,
            "num_custom_kernel": 0,
            "num_total_kernels": 3,
            "custom_kernel_cuda_time_in_profiling_us": 0.0,
            "total_kernel_run_time_in_profiling_us": 5.0,
        }
    ]

    result = manager(
        response_ids=[1, 2, 3],
        response_str="",
        ground_truth="",
        entry_point="Model",
        uuid="u1",
        return_dict=True,
    )

    info = result["reward_extra_info"]
    assert info["decoy_kernel"] is True
    assert info["is_decoy_kernel"] is True


def test_compute_kernel_multi_turn_metrics_respects_is_decoy_kernel_alias():
    batch = SimpleNamespace(
        batch={"turn_indices": np.array([1])},
        non_tensor_batch={
            "reward_extra_info": [
                {
                    "correctness": True,
                    "performance": 2.0,
                    "compilation": True,
                    "is_decoy_kernel": True,
                    "time_coverage": 1.0,
                    "num_coverage": 1.0,
                }
            ],
            "uid": ["sample_1"],
        },
    )

    metrics = compute_kernel_multi_turn_metrics(batch, prefix="kernel")
    assert metrics["kernel/turn_1/correctness_rate"] == 0.0
    assert metrics["kernel/turn_1/mean_performance"] == 0.0
    assert metrics["kernel/turn_1/mean_performance_in_all"] == 0.0


def test_compute_mismatch_quality_metrics_respects_is_decoy_kernel_alias():
    batch = SimpleNamespace(
        non_tensor_batch={
            "reward_extra_info": [
                {
                    "correctness": True,
                    "performance": 2.0,
                    "is_decoy_kernel": True,
                }
            ]
        }
    )

    metrics = compute_mismatch_quality_metrics(
        batch=batch,
        original_response_mask=torch.tensor([[1, 1]], dtype=torch.int64),
        modified_response_mask=torch.tensor([[0, 0]], dtype=torch.int64),
    )
    assert metrics["mismatch_quality/masked/correct_count"] == 0
    assert metrics["mismatch_quality/masked/correct_rate"] == 0.0


def test_kernel_agent_returns_cuda_bundle_as_answer():
    KernelAgent = _load_kernel_agent_class()
    tokenizer = DummyTokenizer()
    agent = KernelAgent(tokenizer)
    response_ids = [ord(char) for char in _fixture_text("response_valid.md")]

    _, _, action, _, agent_info = asyncio.run(
        agent.generate_thought_and_action(response_ids, response_truncation=None)
    )

    assert action.startswith("### CUDA_KERNELS")
    assert "### MODEL_NEW" in action
    assert agent_info["finish_type"].value == "answer"
