from __future__ import annotations

from typing import Any


def _format_first_turn_prompt(prompt_template: str, problem_prompt: str) -> str:
    template = prompt_template.rstrip()
    if "{problem}" in template:
        problem = problem_prompt.strip()
        return template.replace("{problem}", problem)
    problem = problem_prompt.lstrip()
    return f"{template}\n\n{problem}"


def apply_turn_prompt_template(
    messages: list[dict[str, Any]],
    prompt_template: str | None,
    *,
    current_turn: int,
    tool_as_user: bool,
) -> list[dict[str, Any]]:
    """Apply a per-turn prompt template to chat messages in-place.

    First-turn templates may include ``{problem}`` to control where the
    original problem prompt is inserted in the rendered user message.
    """
    if prompt_template is None:
        return messages

    if current_turn == 0:
        if messages and messages[-1].get("role") == "user":
            original_content = messages[-1].get("content", "")
            messages[-1]["content"] = _format_first_turn_prompt(prompt_template, original_content)
        else:
            messages.append({"role": "user", "content": prompt_template.replace("{problem}", "").rstrip()})
        return messages

    if not tool_as_user:
        messages.append({"role": "user", "content": prompt_template})
        return messages

    if current_turn > 0:
        assert messages[-1]["role"] == "user", "The last message should be a user turn"
        feedback = messages[-1]["content"]
        messages[-1]["content"] = prompt_template.format(feedback=feedback)

    return messages
