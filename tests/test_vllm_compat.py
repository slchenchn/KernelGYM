import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


pytest.importorskip("vllm")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "drkernel"))
sys.path.insert(0, str(ROOT / "drkernel" / "verl"))

_old_verl_modules = {
    name: module
    for name, module in sys.modules.items()
    if name == "verl" or name.startswith("verl.")
}
for name in _old_verl_modules:
    sys.modules.pop(name, None)
try:
    from verl.utils.vllm.utils import LoRAModel
    from verl_patch.workers.code.rollout.vllm_rollout.vllm_compat import (
        mark_vllm_config_for_external_launcher,
    )
finally:
    for name in list(sys.modules):
        if name == "verl" or name.startswith("verl."):
            sys.modules.pop(name, None)
    sys.modules.update(_old_verl_modules)


def test_vllm_lora_model_compat_import_exposes_required_constructors():
    assert hasattr(LoRAModel, "from_local_checkpoint")
    assert hasattr(LoRAModel, "from_lora_tensors")


def test_vllm_worker_config_is_marked_as_external_launcher():
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(distributed_executor_backend=object())
    )

    assert mark_vllm_config_for_external_launcher(vllm_config) is vllm_config
    assert vllm_config.parallel_config.distributed_executor_backend == "external_launcher"
