from __future__ import annotations

import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jinja2 import Environment


_JINJA_PROBLEM_RE = re.compile(r"{{\s*problem\s*}}")
_JINJA_ENV = Environment(keep_trailing_newline=True)


def iter_prompt_template_candidates(prompt_template: Any) -> list[str]:
    """Return all concrete template strings from a single template or a candidate list."""
    if prompt_template is None:
        return []
    if isinstance(prompt_template, str):
        return [prompt_template]
    if isinstance(prompt_template, Iterable):
        return [str(candidate) for candidate in prompt_template if candidate is not None]
    return [str(prompt_template)]


def _read_prompt_template_path(template_path: Any, base_dir: Path | None) -> str:
    path = Path(str(template_path)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.read_text(encoding="utf-8")


def _read_prompt_template_dir(template_dir: Any, base_dir: Path | None) -> list[str]:
    path = Path(str(template_dir)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.is_dir():
        raise ValueError(f"prompt template dir does not exist: {path}")
    template_paths = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix in {".jinja", ".j2", ".txt", ".md"}
    )
    if not template_paths:
        raise ValueError(f"prompt template dir contains no template files: {path}")
    return [template_path.read_text(encoding="utf-8") for template_path in template_paths]


def prompt_template_from_config(prompt_config: Any, *, base_dir: str | Path | None = None) -> str | list[str] | None:
    """Extract prompt text from a config object supporting inline text or template files."""
    base_path = Path(base_dir).expanduser() if base_dir is not None else None
    template_dirs = prompt_config.get("template_dirs", None)
    if template_dirs is not None:
        templates: list[str] = []
        for template_dir in iter_prompt_template_candidates(template_dirs):
            templates.extend(_read_prompt_template_dir(template_dir, base_path))
        return templates or None

    template_dir = prompt_config.get("template_dir", None)
    if template_dir is not None:
        return _read_prompt_template_dir(template_dir, base_path)

    template_paths = prompt_config.get("template_paths", None)
    if template_paths is not None:
        return [
            _read_prompt_template_path(template_path, base_path)
            for template_path in iter_prompt_template_candidates(template_paths)
        ]

    template_path = prompt_config.get("template_path", None)
    if template_path is not None:
        return _read_prompt_template_path(template_path, base_path)

    templates = prompt_config.get("templates", None)
    if templates is not None:
        candidates = iter_prompt_template_candidates(templates)
        return candidates or None
    return prompt_config.get("template", None)


def select_prompt_template(prompt_template: Any) -> str | None:
    """Randomly choose one template when multiple candidates are configured."""
    candidates = iter_prompt_template_candidates(prompt_template)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return random.choice(candidates)


def render_prompt_template(prompt_template: Any, **variables: Any) -> str:
    """Render a prompt template using Jinja syntax."""
    selected_template = select_prompt_template(prompt_template)
    if selected_template is None:
        return ""
    return _JINJA_ENV.from_string(selected_template).render(**variables)


def _template_has_problem_placeholder(prompt_template: str) -> bool:
    return _JINJA_PROBLEM_RE.search(prompt_template) is not None


def _format_first_turn_prompt(prompt_template: Any, problem_prompt: str) -> str:
    selected_template = select_prompt_template(prompt_template)
    if selected_template is None:
        return problem_prompt
    if _template_has_problem_placeholder(selected_template):
        problem = problem_prompt.strip()
        return render_prompt_template(selected_template, problem=problem)
    template = selected_template.rstrip()
    problem = problem_prompt.lstrip()
    return f"{template}\n\n{problem}"


def _template_render_prefix(prompt_template: str) -> str:
    template = prompt_template.rstrip()
    jinja_problem_match = _JINJA_PROBLEM_RE.search(template)
    if jinja_problem_match:
        return template[: jinja_problem_match.start()].strip()
    return template.strip()


def _template_render_prefixes(prompt_template: Any) -> list[str]:
    return [
        prefix
        for prefix in (_template_render_prefix(candidate) for candidate in iter_prompt_template_candidates(prompt_template))
        if prefix
    ]


def apply_initial_user_prompt_template(
    messages: list[dict[str, Any]],
    prompt_template: Any,
) -> list[dict[str, Any]]:
    """Ensure the initial problem message carries the first-turn instruction.

    Multi-turn rollout keeps dataset rows backend-neutral. The live request state
    must still carry the backend-specific first-turn instruction so later turns
    do not start from a neutral problem plus a previous bad answer.
    """
    if prompt_template is None:
        return messages

    rendered_prefixes = _template_render_prefixes(prompt_template)
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if any(str(content).lstrip().startswith(prefix) for prefix in rendered_prefixes):
            return messages
        message["content"] = _format_first_turn_prompt(prompt_template, str(content))
        return messages

    messages.append(
        {
            "role": "user",
            "content": render_prompt_template(prompt_template, problem="").rstrip(),
        }
    )
    return messages


def apply_turn_prompt_template(
    messages: list[dict[str, Any]],
    prompt_template: Any,
    *,
    current_turn: int,
    tool_as_user: bool,
) -> list[dict[str, Any]]:
    """Apply a per-turn prompt template to chat messages in-place.

    First-turn templates may include ``{{ problem }}`` to control where the
    original problem prompt is inserted in the rendered user message.
    """
    if prompt_template is None:
        return messages

    if current_turn == 0:
        return apply_initial_user_prompt_template(messages, prompt_template)

    if not tool_as_user:
        selected_template = select_prompt_template(prompt_template)
        if selected_template is not None:
            messages.append({"role": "user", "content": selected_template})
        return messages

    if current_turn > 0:
        assert messages[-1]["role"] == "user", "The last message should be a user turn"
        feedback = messages[-1]["content"]
        messages[-1]["content"] = render_prompt_template(prompt_template, feedback=feedback)

    return messages
