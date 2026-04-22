import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drkernel" / "verl"))

transformers = pytest.importorskip("transformers")

AutoModel = transformers.AutoModel
AutoModelForCausalLM = transformers.AutoModelForCausalLM

_old_verl_modules = {
    name: module
    for name, module in sys.modules.items()
    if name == "verl" or name.startswith("verl.")
}
for name in _old_verl_modules:
    sys.modules.pop(name, None)
try:
    from verl.utils.model import get_hf_auto_model_class
finally:
    for name in list(sys.modules):
        if name == "verl" or name.startswith("verl."):
            sys.modules.pop(name, None)
    sys.modules.update(_old_verl_modules)


try:
    AutoModelForVisionText = transformers.AutoModelForImageTextToText
except AttributeError:
    AutoModelForVisionText = transformers.AutoModelForVision2Seq


class _RemoteConfig:
    architectures = ["RemoteArch"]

    def __init__(self, auto_map):
        self.auto_map = auto_map


def test_get_hf_auto_model_class_supports_image_text_to_text_auto_map():
    config = _RemoteConfig({"AutoModelForImageTextToText": "remote_module.RemoteArch"})

    assert get_hf_auto_model_class(config) is AutoModelForVisionText


def test_get_hf_auto_model_class_supports_legacy_vision2seq_auto_map():
    config = _RemoteConfig({"AutoModelForVision2Seq": "remote_module.RemoteArch"})

    assert get_hf_auto_model_class(config) is AutoModelForVisionText


def test_get_hf_auto_model_class_preserves_causal_lm_auto_map():
    config = _RemoteConfig({"AutoModelForCausalLM": "remote_module.RemoteArch"})

    assert get_hf_auto_model_class(config) is AutoModelForCausalLM


def test_get_hf_auto_model_class_falls_back_for_unknown_remote_auto_map():
    config = _RemoteConfig({"AutoModel": "remote_module.RemoteArch"})

    assert get_hf_auto_model_class(config) is AutoModel
