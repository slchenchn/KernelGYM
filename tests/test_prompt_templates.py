from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "drkernel"))

from kernel.workers.rollout import prompt_templates as prompt_template_module
from kernel.workers.rollout.prompt_templates import (
    apply_initial_user_prompt_template,
    apply_turn_prompt_template,
    iter_prompt_template_candidates,
    prompt_template_from_config,
)

CUDA_PROMPT_CONFIG = ROOT / "drkernel" / "kernel" / "config" / "prompt_config" / "multi_turn_cuda_kernel.yaml"
PROMPT_FIXTURES = ROOT / "tests" / "fixtures" / "prompt_templates"


def _load_expected_prompt_fixture(name: str) -> str:
    return (PROMPT_FIXTURES / name).read_text(encoding="utf-8")


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


def _load_actual_cuda_prompt_template(name: str):
    config = yaml.safe_load(CUDA_PROMPT_CONFIG.read_text())
    for prompt_config in config["per_turn_prompts"]:
        if prompt_config["name"] == name:
            return prompt_template_from_config(prompt_config, base_dir=CUDA_PROMPT_CONFIG.parent)
    raise AssertionError(f"Prompt template {name!r} not found")


@pytest.mark.parametrize("tool_as_user", [True, False])
def test_first_turn_template_is_prepended_before_problem_for_both_tool_modes(tool_as_user):
    messages = [{"role": "user", "content": "PROBLEM"}]

    _apply(messages, "INSTRUCTION", current_turn=0, tool_as_user=tool_as_user)

    assert messages == [{"role": "user", "content": "INSTRUCTION\n\nPROBLEM"}]


def test_first_turn_template_is_prepended_with_instruction_before_model_order():
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
        "INSTRUCTION\n\n{{ problem }}",
        current_turn=0,
        tool_as_user=True,
    )

    assert messages == [{"role": "user", "content": "INSTRUCTION\n\nPROBLEM"}]


def test_legacy_brace_problem_placeholder_is_not_rendered():
    messages = [{"role": "user", "content": "PROBLEM"}]

    _apply(
        messages,
        "INSTRUCTION\n\n{problem}",
        current_turn=0,
        tool_as_user=True,
    )

    assert messages == [{"role": "user", "content": "INSTRUCTION\n\n{problem}\n\nPROBLEM"}]


def test_actual_cuda_first_turn_template_directory_loads_multiple_templates():
    templates = _load_actual_cuda_prompt_template("first_turn")

    candidates = iter_prompt_template_candidates(templates)

    assert len(candidates) == 3
    assert any("KEY RULES" in candidate for candidate in candidates)
    assert any("You write custom CUDA implementations." in candidate for candidate in candidates)
    assert all("{{ problem }}" in candidate for candidate in candidates)


def test_actual_cuda_tool_response_template_directory_loads_multiple_templates():
    templates = _load_actual_cuda_prompt_template("tool_response")

    candidates = iter_prompt_template_candidates(templates)

    assert len(candidates) >= 2
    assert all("{{ feedback }}" in candidate for candidate in candidates)


def test_actual_cuda_first_turn_template_is_instruction_before_neutral_problem(monkeypatch):
    monkeypatch.setattr(prompt_template_module.random, "choice", lambda candidates: candidates[0])
    template = _load_actual_cuda_prompt_template("first_turn")
    messages = [
        {
            "role": "user",
            "content": "You are given the following PyTorch model:\n```python\nclass Model: pass\n```",
        }
    ]

    _apply(messages, template, current_turn=0, tool_as_user=True)

    assert messages[0]["content"] == _load_expected_prompt_fixture("cuda_first_turn_expected.txt")


def test_actual_cuda_first_turn_templates_do_not_duplicate_problem_intro(monkeypatch):
    problem = "You are given the following PyTorch model:\n```python\nclass Model: pass\n```"
    templates = iter_prompt_template_candidates(_load_actual_cuda_prompt_template("first_turn"))

    for template in templates:
        messages = [{"role": "user", "content": problem}]
        _apply(messages, template, current_turn=0, tool_as_user=True)

        rendered = messages[0]["content"]
        assert "reference pytorch code:" not in rendered.lower()
        assert "reference problem:" not in rendered.lower()
        assert rendered.count("You are given the following PyTorch model:") == 1


def test_actual_cuda_tool_response_template_wraps_feedback_and_keeps_cuda_sections(monkeypatch):
    monkeypatch.setattr(prompt_template_module.random, "choice", lambda candidates: candidates[0])
    template = _load_actual_cuda_prompt_template("tool_response")
    messages = [{"role": "user", "content": "compile error: missing binding"}]

    _apply(messages, template, current_turn=1, tool_as_user=True)

    assert messages[0]["content"] == _load_expected_prompt_fixture("cuda_tool_response_expected.txt")


def test_later_turn_can_render_initial_cuda_instruction_and_feedback_template(monkeypatch):
    monkeypatch.setattr(prompt_template_module.random, "choice", lambda candidates: candidates[0])
    first_turn_template = _load_actual_cuda_prompt_template("first_turn")
    tool_template = _load_actual_cuda_prompt_template("tool_response")
    messages = [
        {
            "role": "user",
            "content": "You are given the following PyTorch model:\n```python\nclass Model: pass\n```",
        },
        {"role": "assistant", "content": "BAD CODE"},
        {"role": "user", "content": "compile error: missing binding"},
    ]

    apply_initial_user_prompt_template(messages, first_turn_template)
    _apply(messages, tool_template, current_turn=1, tool_as_user=True)

    assert messages[0]["content"] == _load_expected_prompt_fixture("cuda_first_turn_expected.txt")
    assert messages[-1]["content"] == _load_expected_prompt_fixture("cuda_tool_response_expected.txt")


def test_first_turn_template_is_idempotent_when_initial_message_is_already_rendered(monkeypatch):
    monkeypatch.setattr(prompt_template_module.random, "choice", lambda candidates: candidates[0])
    template = _load_actual_cuda_prompt_template("first_turn")
    messages = [
        {
            "role": "user",
            "content": "You are given the following PyTorch model:\n```python\nclass Model: pass\n```",
        }
    ]

    apply_initial_user_prompt_template(messages, template)
    first_render = messages[0]["content"]
    _apply(messages, template, current_turn=0, tool_as_user=True)

    assert messages[0]["content"] == first_render


def test_multiple_prompt_templates_are_randomly_selected(monkeypatch):
    choices: list[str] = []

    def choose_last(candidates):
        choices.append(candidates[-1])
        return candidates[-1]

    monkeypatch.setattr(prompt_template_module.random, "choice", choose_last)
    messages = [{"role": "user", "content": "PROBLEM"}]

    _apply(messages, ["FIRST {{ problem }}", "SECOND {{ problem }}"], current_turn=0, tool_as_user=True)

    assert messages == [{"role": "user", "content": "SECOND PROBLEM"}]
    assert choices == ["SECOND {{ problem }}"]


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

    _apply(messages, "Feedback:\n{{ feedback }}", current_turn=1, tool_as_user=True)

    assert messages[-1] == {"role": "user", "content": "Feedback:\nERROR"}


def test_later_user_feedback_turn_keeps_feedback_verbatim_inside_template():
    feedback = "line 1\n\n```text\ncompiler error {not_a_placeholder}\n```"
    messages = [{"role": "user", "content": feedback}]

    _apply(messages, "Server feedback:\n{{ feedback }}\nFix it.", current_turn=2, tool_as_user=True)

    assert messages[-1]["content"] == f"Server feedback:\n{feedback}\nFix it."


def test_legacy_brace_feedback_placeholder_is_not_rendered():
    messages = [{"role": "user", "content": "ERROR"}]

    _apply(messages, "Feedback:\n{feedback}", current_turn=1, tool_as_user=True)

    assert messages[-1] == {"role": "user", "content": "Feedback:\n{feedback}"}


def test_later_user_feedback_requires_user_message_when_tool_as_user_is_enabled():
    messages = [{"role": "assistant", "content": "ANSWER"}]

    with pytest.raises(AssertionError, match="last message should be a user turn"):
        _apply(messages, "Feedback:\n{{ feedback }}", current_turn=1, tool_as_user=True)


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
