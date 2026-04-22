from __future__ import annotations

import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "drkernel"))

from verl_patch.workers.code.rollout.vllm_rollout.engine_kwargs import get_vllm_engine_kwargs
from kernel.workers.rollout.vllm_rollout import vllm_async_engine as vllm_async_engine_module
from kernel.workers.rollout.vllm_rollout.vllm_async_engine import (
    ExternalRayDistributedExecutor,
    _build_async_engine_args_kwargs,
)


def test_vllm_engine_kwargs_filter_none_and_keep_language_model_only():
    config = {
        "engine_kwargs": {
            "vllm": {
                "language_model_only": True,
                "runner": "generate",
                "hf_overrides": None,
            }
        }
    }

    assert get_vllm_engine_kwargs(config) == {
        "language_model_only": True,
        "runner": "generate",
    }


def test_vllm_engine_kwargs_add_limit_images_mapping():
    config = {
        "limit_images": 2,
        "engine_kwargs": {
            "vllm": {
                "language_model_only": True,
            }
        },
    }

    assert get_vllm_engine_kwargs(config) == {
        "language_model_only": True,
        "limit_mm_per_prompt": {"image": 2},
    }


def test_cuda_kernel_trainer_sets_vllm_language_model_only():
    config = yaml.safe_load((ROOT / "drkernel" / "kernel" / "config" / "cuda_kernel_trainer.yaml").read_text())

    assert config["actor_rollout_ref"]["rollout"]["engine_kwargs"] == {
        "vllm": {
            "language_model_only": True,
        }
    }


def test_async_engine_args_include_rollout_max_num_seqs():
    rollout = {
        "dtype": "bfloat16",
        "enforce_eager": False,
        "gpu_memory_utilization": 0.75,
        "enable_chunked_prefill": True,
        "seed": 0,
        "max_num_seqs": 32,
        "engine_kwargs": {"vllm": {"language_model_only": True}},
    }
    config = OmegaConf.create({"rollout": rollout})

    kwargs = _build_async_engine_args_kwargs(
        config=config,
        local_model_path="/model",
        tokenizer_path="/tokenizer",
        override_generation_config={},
        tensor_parallel_size=2,
        distributed_executor_backend=None,
        max_model_len=22240,
        max_num_batched_tokens=23240,
        trust_remote_code=False,
    )

    assert kwargs["max_num_seqs"] == 32
    assert kwargs["language_model_only"] is True


class _FakeSchedulerOutput:
    def __init__(self, total_num_scheduled_tokens):
        self.total_num_scheduled_tokens = total_num_scheduled_tokens


class _FakeRemoteMethod:
    def __init__(self, result, calls):
        self._result = result
        self._calls = calls

    def remote(self, method, *args, **kwargs):
        self._calls.append((method, args, kwargs))
        return self._result


class _FakeWorker:
    def __init__(self, result, calls):
        self.execute_method = _FakeRemoteMethod(result, calls)


def _make_executor(*, uses_sampler=True, worker_results=("rank0", "rank1")):
    executor = object.__new__(ExternalRayDistributedExecutor)
    executor.uses_sampler = uses_sampler
    executor.scheduler_output = None
    calls = []
    executor.workers = [_FakeWorker(result, calls) for result in worker_results]
    return executor, calls


def test_external_ray_executor_defers_sampled_batches():
    executor, _ = _make_executor()
    scheduler_output = _FakeSchedulerOutput(total_num_scheduled_tokens=128)

    future = executor.execute_model(scheduler_output, non_block=True)

    assert future.result() is None
    assert executor.scheduler_output is scheduler_output


def test_external_ray_executor_sample_tokens_executes_all_workers(monkeypatch):
    executor, calls = _make_executor(worker_results=("driver-output", "rank1-output"))
    scheduler_output = _FakeSchedulerOutput(total_num_scheduled_tokens=128)
    executor.scheduler_output = scheduler_output
    monkeypatch.setattr(vllm_async_engine_module.ray, "get", lambda refs, timeout=None: refs)

    future = executor.sample_tokens("grammar", non_block=True)

    assert future.result() == "driver-output"
    assert executor.scheduler_output is None
    assert calls == [
        ("execute_model_ray", ((scheduler_output, "grammar"),), {}),
        ("execute_model_ray", ((scheduler_output, "grammar"),), {}),
    ]


def test_external_ray_executor_runs_unsampled_batches_immediately(monkeypatch):
    executor, calls = _make_executor(worker_results=("empty-output", "rank1-output"))
    scheduler_output = _FakeSchedulerOutput(total_num_scheduled_tokens=0)
    monkeypatch.setattr(vllm_async_engine_module.ray, "get", lambda refs, timeout=None: refs)

    output = executor.execute_model(scheduler_output, non_block=False)

    assert output == "empty-output"
    assert executor.scheduler_output is None
    assert calls == [
        ("execute_model_ray", ((scheduler_output, None),), {}),
        ("execute_model_ray", ((scheduler_output, None),), {}),
    ]
