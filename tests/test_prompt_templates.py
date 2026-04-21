from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "drkernel"))

from kernel.workers.rollout.prompt_templates import apply_turn_prompt_template

CUDA_PROMPT_CONFIG = ROOT / "drkernel" / "kernel" / "config" / "prompt_config" / "multi_turn_cuda_kernel.yaml"


def _apply(
    messages: list[dict],
    prompt_template: str | None,
    *,
    current_turn: int,
    tool_as_user: bool,
) -> list[dict]:
    result = apply_turn_prompt_template(
        messages,
        prompt_template,
        current_turn=current_turn,
        tool_as_user=tool_as_user,
    )

    assert result is messages
    return result


def _load_actual_cuda_prompt_template(name: str) -> str | None:
    config = yaml.safe_load(CUDA_PROMPT_CONFIG.read_text())
    for prompt_config in config["per_turn_prompts"]:
        if prompt_config["name"] == name:
            return prompt_config["template"]
    raise AssertionError(f"Prompt template {name!r} not found")


@pytest.mark.parametrize("tool_as_user", [True, False])
def test_first_turn_template_is_prepended_before_problem_for_both_tool_modes(tool_as_user):
    messages = [{"role": "user", "content": "PROBLEM"}]

    _apply(messages, "INSTRUCTION", current_turn=0, tool_as_user=tool_as_user)

    assert messages == [{"role": "user", "content": "INSTRUCTION\n\nPROBLEM"}]


def test_first_turn_template_is_prepended_with_triton_style_instruction_before_model_order():
    messages = [
        {
            "role": "user",
            "content": "You are given the following PyTorch model:\n```python\nclass Model: pass\n```",
        }
    ]

    _apply(
        messages,
        "Return exactly CUDA_KERNELS, APPLY_BINDINGS, and MODEL_NEW.",
        current_turn=0,
        tool_as_user=True,
    )

    assert messages[0]["content"] == (
        "Return exactly CUDA_KERNELS, APPLY_BINDINGS, and MODEL_NEW.\n"
        "\n"
        "You are given the following PyTorch model:\n"
        "```python\n"
        "class Model: pass\n"
        "```"
    )


def test_first_turn_template_problem_placeholder_controls_problem_position():
    messages = [{"role": "user", "content": "\nPROBLEM\n"}]

    _apply(
        messages,
        "INSTRUCTION\n\n{problem}\n\nLet's think step by step.",
        current_turn=0,
        tool_as_user=True,
    )

    assert messages == [{"role": "user", "content": "INSTRUCTION\n\nPROBLEM\n\nLet's think step by step."}]


def test_actual_cuda_first_turn_template_is_instruction_before_neutral_problem():
    template = _load_actual_cuda_prompt_template("first_turn")
    messages = [
        {
            "role": "user",
            "content": "You are given the following PyTorch model:\n```python\nclass Model: pass\n```",
        }
    ]

    _apply(messages, template, current_turn=0, tool_as_user=True)

    assert messages[0]["content"] == (
        "Optimize the PyTorch model below with a CUDA-Agent implementation.\n"
        "\n"
        "Requirements:\n"
        "1. Preserve the public interface: the optimized class must be named `ModelNew`, and `__init__` and `forward` signatures must match `Model`.\n"
        "2. Keep all submodule names and state-dict keys unchanged. Do not create extra trainable parameters during initialization.\n"
        "3. Do not use inline CUDA strings or `torch.utils.cpp_extension.load_inline` in `MODEL_NEW`; CUDA/C++ code must live only in the CUDA sections below.\n"
        "4. `MODEL_NEW` must import and call the compiled extension as `cuda_extension`.\n"
        "5. `APPLY_BINDINGS` must include `#include \"../binding_registry.h\"` and register exported functions with `REGISTER_BINDING(...)`.\n"
        "6. Implement CUDA operators yourself where possible. You may use cuBLAS/cuDNN only for GEMM/convolution-style primitives.\n"
        "7. Return real, compilable code, not pseudocode.\n"
        "\n"
        "Return exactly this format:\n"
        "### CUDA_KERNELS\n"
        "```cpp\n"
        "<CUDA .cu code here>\n"
        "```\n"
        "\n"
        "### APPLY_BINDINGS\n"
        "```cpp\n"
        "// Must include exactly: #include \"../binding_registry.h\"\n"
        "<apply_bindings.cpp code here>\n"
        "```\n"
        "\n"
        "### MODEL_NEW\n"
        "```python\n"
        "<model_new.py code here>\n"
        "```\n"
        "\n"
        "You are given the following PyTorch model:\n"
        "```python\n"
        "class Model: pass\n"
        "```\n"
        "\n"
        "Let's think step by step."
    )


def test_actual_cuda_tool_response_template_wraps_feedback_and_keeps_cuda_sections():
    template = _load_actual_cuda_prompt_template("tool_response")
    messages = [{"role": "user", "content": "compile error: missing binding"}]

    _apply(messages, template, current_turn=1, tool_as_user=True)

    assert messages[0]["content"] == (
        "Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1. If the feedback reports compilation, runtime, correctness, or other errors, fix those errors first.\n"
        "2. If the previous implementation was correct, do not output the same code; try a different CUDA optimization strategy for better performance.\n"
        "\n"
        "Here is the server feedback. Please refer to this feedback to improve the implementation:\n"
        "Server feedback (status/metrics/errors):\n"
        "compile error: missing binding\n"
        "\n"
        "Modify any section as needed.\n"
        "\n"
        "Return an improved CUDA implementation with the same output format:\n"
        "### CUDA_KERNELS\n"
        "```cpp\n"
        "<CUDA .cu code here>\n"
        "```\n"
        "\n"
        "### APPLY_BINDINGS\n"
        "```cpp\n"
        "// Must include exactly: #include \"../binding_registry.h\"\n"
        "<apply_bindings.cpp code here>\n"
        "```\n"
        "\n"
        "### MODEL_NEW\n"
        "```python\n"
        "<model_new.py code here>\n"
        "```\n"
        "Let's think step by step.\n"
    )


def test_first_turn_template_strips_outer_boundary_whitespace_only():
    messages = [{"role": "user", "content": "\n\nPROBLEM\n"}]

    _apply(messages, "INSTRUCTION\n\n", current_turn=0, tool_as_user=True)

    assert messages == [{"role": "user", "content": "INSTRUCTION\n\nPROBLEM\n"}]


def test_first_turn_template_preserves_internal_problem_whitespace():
    problem = "line 1\n\n    indented line\nline 3"
    messages = [{"role": "user", "content": problem}]

    _apply(messages, "INSTRUCTION", current_turn=0, tool_as_user=True)

    assert messages[0]["content"] == f"INSTRUCTION\n\n{problem}"


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "system", "content": "SYSTEM"}],
        [{"role": "assistant", "content": "PREVIOUS ANSWER"}],
    ],
)
def test_first_turn_template_is_not_silently_dropped_when_last_message_is_not_user(messages):
    _apply(messages, "INSTRUCTION", current_turn=0, tool_as_user=True)

    assert messages[-1] == {"role": "user", "content": "INSTRUCTION"}


def test_none_template_is_noop_and_preserves_object_identity():
    messages = [{"role": "user", "content": "PROBLEM"}]
    before = deepcopy(messages)

    result = _apply(messages, None, current_turn=0, tool_as_user=True)

    assert result is messages
    assert messages == before


def test_later_user_feedback_turn_is_wrapped_by_template():
    messages = [
        {"role": "user", "content": "PROBLEM"},
        {"role": "assistant", "content": "ANSWER"},
        {"role": "user", "content": "ERROR"},
    ]

    _apply(messages, "Feedback:\n{feedback}", current_turn=1, tool_as_user=True)

    assert messages[-1] == {"role": "user", "content": "Feedback:\nERROR"}


def test_later_user_feedback_turn_keeps_feedback_verbatim_inside_template():
    feedback = "line 1\n\n```text\ncompiler error {not_a_placeholder}\n```"
    messages = [{"role": "user", "content": feedback}]

    _apply(messages, "Server feedback:\n{feedback}\nFix it.", current_turn=2, tool_as_user=True)

    assert messages[-1]["content"] == f"Server feedback:\n{feedback}\nFix it."


def test_later_user_feedback_requires_user_message_when_tool_as_user_is_enabled():
    messages = [{"role": "assistant", "content": "ANSWER"}]

    with pytest.raises(AssertionError, match="last message should be a user turn"):
        _apply(messages, "Feedback:\n{feedback}", current_turn=1, tool_as_user=True)


def test_later_user_feedback_template_must_contain_feedback_placeholder_to_include_feedback():
    messages = [{"role": "user", "content": "ERROR"}]

    _apply(messages, "No placeholder here.", current_turn=1, tool_as_user=True)

    assert messages[-1] == {"role": "user", "content": "No placeholder here."}


def test_later_non_user_tool_mode_appends_template_as_user_message():
    messages = [{"role": "assistant", "content": "ANSWER"}]

    _apply(messages, "NEXT INSTRUCTION", current_turn=1, tool_as_user=False)

    assert messages == [
        {"role": "assistant", "content": "ANSWER"},
        {"role": "user", "content": "NEXT INSTRUCTION"},
    ]


def test_later_non_user_tool_mode_appends_even_when_last_message_is_user():
    messages = [{"role": "user", "content": "EXISTING"}]

    _apply(messages, "NEXT INSTRUCTION", current_turn=1, tool_as_user=False)

    assert messages == [
        {"role": "user", "content": "EXISTING"},
        {"role": "user", "content": "NEXT INSTRUCTION"},
    ]
