#!/usr/bin/env python3
"""Build backend-neutral KernelBench parquet data from prompt-heavy source data."""

from __future__ import annotations

import argparse
import random
import re
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from jinja2 import Environment


ARCHITECTURE_BLOCK_PATTERNS = (
    re.compile(
        r"You\s+are\s+given\s+the\s+following\s+architecture:\s*"
        r"```(?:python)?\s*\n(?P<code>.*?)\n\s*```",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"Now\s+you\s+are\s+given\s+the\s+below\s+PyTorch\s+model:\s*"
        r"```(?:python)?\s*\n(?P<code>.*?)\n\s*```",
        re.IGNORECASE | re.DOTALL,
    ),
)
_JINJA_ENV = Environment(keep_trailing_newline=True)


@dataclass(frozen=True)
class NeutralizationSummary:
    input_path: Path
    output_path: Path
    text_output_path: Path | None
    rows: int
    rows_with_triton: int
    rows_with_cuda_agent_sections: int


def extract_architecture_code(prompt_text: str) -> str:
    """Extract only the problem model code from a backend-specific source prompt."""

    for pattern in ARCHITECTURE_BLOCK_PATTERNS:
        match = pattern.search(prompt_text)
        if match is not None:
            return match.group("code").strip()
    raise ValueError("could not find a fenced PyTorch model architecture block")


def build_backend_neutral_prompt(architecture_code: str) -> str:
    return (
        "You are given the following PyTorch model:\n"
        "```python\n"
        f"{architecture_code.strip()}\n"
        "```"
    )


def neutralize_prompt_messages(messages: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Replace the first user message with a backend-neutral model-only prompt."""

    neutralized: list[dict[str, str]] = []
    replaced = False
    for message in messages:
        new_message = dict(message)
        if not replaced and new_message.get("role") == "user":
            architecture_code = extract_architecture_code(new_message.get("content", ""))
            new_message["content"] = build_backend_neutral_prompt(architecture_code)
            replaced = True
        neutralized.append(new_message)
    if not replaced:
        raise ValueError("prompt row does not contain a user message")
    return neutralized


def _has_cuda_agent_sections(text: str) -> bool:
    return any(section in text for section in ("CUDA_KERNELS", "APPLY_BINDINGS", "MODEL_NEW"))


def format_prompt_row_for_text(row_index: int, messages: Iterable[dict[str, str]]) -> str:
    lines = [f"==================== row {row_index} ===================="]
    for message_index, message in enumerate(messages):
        role = message.get("role", "")
        content = message.get("content", "")
        lines.append(f"--- message {message_index} role={role} ---")
        lines.append(content.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def write_backend_neutral_text(text_output_path: Path, neutral_prompts: Iterable[list[dict[str, str]]]) -> None:
    text_output_path.parent.mkdir(parents=True, exist_ok=True)
    with text_output_path.open("w", encoding="utf-8") as output_file:
        for row_index, messages in enumerate(neutral_prompts):
            if row_index:
                output_file.write("\n")
            output_file.write(format_prompt_row_for_text(row_index, messages))


def _prompt_template_candidates(prompt_template: object) -> list[str]:
    if prompt_template is None:
        return []
    if isinstance(prompt_template, str):
        return [prompt_template]
    if isinstance(prompt_template, IterableABC):
        return [str(candidate) for candidate in prompt_template if candidate is not None]
    return [str(prompt_template)]


def _read_prompt_template_path(template_path: object, base_dir: Path) -> str:
    path = Path(str(template_path)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.read_text(encoding="utf-8")


def _read_prompt_template_dir(template_dir: object, base_dir: Path) -> list[str]:
    path = Path(str(template_dir)).expanduser()
    if not path.is_absolute():
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


def _prompt_template_from_config(per_turn_prompt: dict[str, object], base_dir: Path) -> str | list[str] | None:
    template_dirs = per_turn_prompt.get("template_dirs")
    if template_dirs is not None:
        templates: list[str] = []
        for template_dir in _prompt_template_candidates(template_dirs):
            templates.extend(_read_prompt_template_dir(template_dir, base_dir))
        return templates or None

    template_dir = per_turn_prompt.get("template_dir")
    if template_dir is not None:
        return _read_prompt_template_dir(template_dir, base_dir)

    template_paths = per_turn_prompt.get("template_paths")
    if template_paths is not None:
        return [
            _read_prompt_template_path(template_path, base_dir)
            for template_path in _prompt_template_candidates(template_paths)
        ]

    template_path = per_turn_prompt.get("template_path")
    if template_path is not None:
        return _read_prompt_template_path(template_path, base_dir)

    templates = per_turn_prompt.get("templates")
    if templates is not None:
        candidates = _prompt_template_candidates(templates)
        return candidates or None

    return per_turn_prompt.get("template")  # type: ignore[return-value]


def load_per_turn_prompt_template(prompt_config_path: Path, name: str) -> str | list[str] | None:
    prompt_config = yaml.safe_load(prompt_config_path.read_text(encoding="utf-8"))
    for per_turn_prompt in prompt_config.get("per_turn_prompts", []):
        if per_turn_prompt.get("name") == name:
            return _prompt_template_from_config(per_turn_prompt, prompt_config_path.parent)
    raise ValueError(f"prompt template {name!r} not found in {prompt_config_path}")


def _render_prompt_template(template: str, **variables: str) -> str:
    return _JINJA_ENV.from_string(template).render(**variables)


def _template_has_problem_placeholder(template: str) -> bool:
    return re.search(r"{{\s*problem\s*}}", template) is not None


def _select_prompt_template(prompt_template: str | list[str] | None, rng: random.Random | None = None) -> str | None:
    candidates = _prompt_template_candidates(prompt_template)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return (rng or random).choice(candidates)


def format_prompt_messages_for_first_turn(
    messages: Iterable[dict[str, str]],
    first_turn_template: str | list[str] | None,
    *,
    rng: random.Random | None = None,
) -> list[dict[str, str]]:
    formatted = [dict(message) for message in messages]
    selected_template = _select_prompt_template(first_turn_template, rng)
    if selected_template is None:
        return formatted

    if formatted and formatted[-1].get("role") == "user":
        original_content = formatted[-1].get("content", "")
        if _template_has_problem_placeholder(selected_template):
            problem = original_content.strip()
            formatted[-1]["content"] = _render_prompt_template(selected_template, problem=problem)
        else:
            template = selected_template.rstrip()
            problem = original_content.lstrip()
            formatted[-1]["content"] = f"{template}\n\n{problem}"
    else:
        formatted.append(
            {
                "role": "user",
                "content": _render_prompt_template(selected_template, problem="").rstrip(),
            }
        )
    return formatted


def sample_prompt_indices(row_count: int, sample_size: int, seed: int) -> list[int]:
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")

    sample_count = min(row_count, sample_size)
    return sorted(random.Random(seed).sample(range(row_count), sample_count))


def format_sampled_prompt_row_for_text(
    sample_index: int,
    source_row_index: int,
    messages: Iterable[dict[str, str]],
) -> str:
    lines = [f"==================== sample {sample_index} source_row {source_row_index} ===================="]
    for message_index, message in enumerate(messages):
        role = message.get("role", "")
        content = message.get("content", "")
        lines.append(f"--- message {message_index} role={role} ---")
        lines.append(content.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def write_formatted_prompt_sample_text(
    neutral_parquet_path: Path,
    sample_output_path: Path,
    prompt_config_path: Path,
    *,
    sample_size: int = 100,
    seed: int = 0,
) -> list[int]:
    table = pq.read_table(neutral_parquet_path, columns=["prompt"])
    prompt_index = table.schema.get_field_index("prompt")
    if prompt_index < 0:
        raise ValueError(f"{neutral_parquet_path} does not contain a prompt column")

    prompts = table.column(prompt_index).to_pylist()
    sample_indices = sample_prompt_indices(len(prompts), sample_size, seed)
    first_turn_template = load_per_turn_prompt_template(prompt_config_path, "first_turn")
    rng = random.Random(seed)

    sample_output_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_output_path.open("w", encoding="utf-8") as output_file:
        for sample_index, source_row_index in enumerate(sample_indices):
            if sample_index:
                output_file.write("\n")
            formatted_messages = format_prompt_messages_for_first_turn(
                prompts[source_row_index],
                first_turn_template,
                rng=rng,
            )
            output_file.write(
                format_sampled_prompt_row_for_text(
                    sample_index,
                    source_row_index,
                    formatted_messages,
                )
            )
    return sample_indices


def materialize_backend_neutral_parquet(
    input_path: Path,
    output_path: Path,
    text_output_path: Path | None = None,
) -> NeutralizationSummary:
    table = pq.read_table(input_path)
    prompt_index = table.schema.get_field_index("prompt")
    if prompt_index < 0:
        raise ValueError(f"{input_path} does not contain a prompt column")

    prompt_field = table.schema.field(prompt_index)
    neutral_prompts = []
    rows_with_triton = 0
    rows_with_cuda_agent_sections = 0
    for row_index, messages in enumerate(table.column(prompt_index).to_pylist()):
        try:
            neutralized = neutralize_prompt_messages(messages)
        except Exception as exc:
            raise ValueError(f"failed to neutralize row {row_index} in {input_path}: {exc}") from exc

        combined = "\n".join(message.get("content", "") for message in neutralized)
        rows_with_triton += int("triton" in combined.lower())
        rows_with_cuda_agent_sections += int(_has_cuda_agent_sections(combined))
        neutral_prompts.append(neutralized)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    neutral_prompt_column = pa.array(neutral_prompts, type=prompt_field.type)
    neutral_table = table.set_column(prompt_index, prompt_field, neutral_prompt_column)
    pq.write_table(neutral_table, output_path)
    if text_output_path is not None:
        write_backend_neutral_text(text_output_path, neutral_prompts)

    return NeutralizationSummary(
        input_path=input_path,
        output_path=output_path,
        text_output_path=text_output_path,
        rows=neutral_table.num_rows,
        rows_with_triton=rows_with_triton,
        rows_with_cuda_agent_sections=rows_with_cuda_agent_sections,
    )


def _default_paths(repo_root: Path) -> tuple[Path, Path, Path, Path]:
    data_root = repo_root / "drkernel" / "data"
    return (
        data_root / "drkernel-rl-data" / "cuda_llm_rl_thinking_1025.parquet",
        data_root / "drkernel-validation-data" / "validation_data_thinking.parquet",
        data_root / "drkernel-rl-data-neutral" / "cuda_llm_rl_thinking_1025.parquet",
        data_root / "drkernel-validation-data-neutral" / "validation_data_thinking.parquet",
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    train_input, val_input, train_output, val_output = _default_paths(repo_root)
    prompt_config = repo_root / "drkernel" / "kernel" / "config" / "prompt_config" / "multi_turn_cuda_kernel.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=train_input)
    parser.add_argument("--val-input", type=Path, default=val_input)
    parser.add_argument("--train-output", type=Path, default=train_output)
    parser.add_argument("--val-output", type=Path, default=val_output)
    parser.add_argument("--prompt-config", type=Path, default=prompt_config)
    parser.add_argument(
        "--train-text-output",
        type=Path,
        default=None,
        help="Text copy of the neutralized train prompts. Defaults to train output with .txt suffix.",
    )
    parser.add_argument(
        "--val-text-output",
        type=Path,
        default=None,
        help="Text copy of the neutralized validation prompts. Defaults to val output with .txt suffix.",
    )
    parser.add_argument(
        "--no-text-output",
        action="store_true",
        help="Only write parquet outputs, without companion human-readable text files.",
    )
    parser.add_argument(
        "--formatted-sample-output",
        type=Path,
        default=None,
        help="Text file containing sampled train prompts after first-turn prompt formatting.",
    )
    parser.add_argument(
        "--formatted-sample-size",
        type=int,
        default=100,
        help="Number of formatted train prompts to sample for manual inspection.",
    )
    parser.add_argument(
        "--formatted-sample-seed",
        type=int,
        default=0,
        help="Random seed used when sampling formatted train prompts.",
    )
    parser.add_argument(
        "--no-formatted-sample",
        action="store_true",
        help="Do not write the sampled formatted prompt inspection file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_text_output = None if args.no_text_output else (args.train_text_output or args.train_output.with_suffix(".txt"))
    val_text_output = None if args.no_text_output else (args.val_text_output or args.val_output.with_suffix(".txt"))
    summaries = [
        materialize_backend_neutral_parquet(args.train_input, args.train_output, train_text_output),
        materialize_backend_neutral_parquet(args.val_input, args.val_output, val_text_output),
    ]
    for summary in summaries:
        text_suffix = f", text_output={summary.text_output_path}" if summary.text_output_path is not None else ""
        print(
            f"{summary.output_path}: rows={summary.rows}, "
            f"rows_with_triton={summary.rows_with_triton}, "
            f"rows_with_cuda_agent_sections={summary.rows_with_cuda_agent_sections}"
            f"{text_suffix}"
        )
    if not args.no_formatted_sample:
        formatted_sample_output = args.formatted_sample_output or args.train_output.with_name(
            f"{args.train_output.stem}.formatted_sample{args.formatted_sample_size}.txt"
        )
        sample_indices = write_formatted_prompt_sample_text(
            args.train_output,
            formatted_sample_output,
            args.prompt_config,
            sample_size=args.formatted_sample_size,
            seed=args.formatted_sample_seed,
        )
        print(
            f"{formatted_sample_output}: formatted_sample_rows={len(sample_indices)}, "
            f"seed={args.formatted_sample_seed}"
        )


if __name__ == "__main__":
    main()
