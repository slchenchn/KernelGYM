"""Helpers for extracting kernel submissions from model responses."""

from __future__ import annotations

import re


CUDA_AGENT_SECTION_ORDER = (
    ("CUDA_KERNELS", "cpp"),
    ("APPLY_BINDINGS", "cpp"),
    ("MODEL_NEW", "python"),
)

CUDA_AGENT_SECTION_NAMES = tuple(section_name for section_name, _ in CUDA_AGENT_SECTION_ORDER)
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_END_RE = re.compile(r"</think\s*>", re.IGNORECASE)
CUDA_AGENT_MARKER_RE = re.compile(
    r"###\s*(?:CUDA_KERNELS|APPLY_BINDINGS|MODEL_NEW)\b",
    re.IGNORECASE,
)


def strip_think_blocks(text: str) -> str:
    """Return the final answer region after model reasoning.

    Qwen-style thinking output is delimited most reliably by the closing
    ``</think>`` marker. If that marker exists, parse only the text after its
    last occurrence. Otherwise, fall back to removing paired think blocks.
    """
    text = text or ""
    think_end_matches = list(THINK_END_RE.finditer(text))
    if think_end_matches:
        return text[think_end_matches[-1].end() :]
    return THINK_BLOCK_RE.sub("", text)


def response_has_cuda_agent_markers(text: str) -> bool:
    return CUDA_AGENT_MARKER_RE.search(text or "") is not None


def _section_pattern(section_name: str, language: str) -> re.Pattern[str]:
    if language == "python":
        language_pattern = r"(?:python|py)"
    else:
        language_pattern = r"(?:cpp|c\+\+|cxx|cuda|cu)?"
    return re.compile(
        rf"###\s*{section_name}\s*```{language_pattern}\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )


def _find_last_complete_cuda_agent_group(text: str) -> dict[str, str]:
    matches = {
        section_name: list(_section_pattern(section_name, language).finditer(text))
        for section_name, language in CUDA_AGENT_SECTION_ORDER
    }
    best_group: dict[str, str] = {}

    for cuda_match in matches["CUDA_KERNELS"]:
        binding_match = next(
            (
                match
                for match in matches["APPLY_BINDINGS"]
                if match.start() > cuda_match.end()
            ),
            None,
        )
        if binding_match is None:
            continue
        model_match = next(
            (
                match
                for match in matches["MODEL_NEW"]
                if match.start() > binding_match.end()
            ),
            None,
        )
        if model_match is None:
            continue

        best_group = {
            "CUDA_KERNELS": cuda_match.group(1).strip(),
            "APPLY_BINDINGS": binding_match.group(1).strip(),
            "MODEL_NEW": model_match.group(1).strip(),
        }

    return best_group


def extract_cuda_agent_sections(text: str, *, require_complete: bool = False) -> dict[str, str]:
    text = strip_think_blocks(text)

    complete_group = _find_last_complete_cuda_agent_group(text)
    if complete_group:
        return complete_group
    if require_complete:
        return {}

    sections: dict[str, str] = {}
    for section_name, language in CUDA_AGENT_SECTION_ORDER:
        matches = list(_section_pattern(section_name, language).finditer(text))
        if matches:
            sections[section_name] = matches[-1].group(1).strip()
    return sections


def format_cuda_agent_submission(
    sections: dict[str, str],
    *,
    require_complete: bool = False,
) -> str | None:
    if require_complete and any(section_name not in sections for section_name in CUDA_AGENT_SECTION_NAMES):
        return None

    ordered_sections: list[str] = []
    for section_name, language in CUDA_AGENT_SECTION_ORDER:
        body = sections.get(section_name)
        if body:
            ordered_sections.append(f"### {section_name}\n```{language}\n{body}\n```")
    if not ordered_sections:
        return None
    return "\n\n".join(ordered_sections)


def extract_kernel_submission(text: str, kernel_backend: str = "triton") -> str:
    """Extract the executable kernel submission from a raw model response."""

    kernel_backend = (kernel_backend or "triton").strip().lower()

    if kernel_backend == "cuda_agent":
        complete_sections = extract_cuda_agent_sections(text, require_complete=True)
        formatted = format_cuda_agent_submission(complete_sections, require_complete=True)
        if formatted is not None:
            return formatted

        partial_sections = extract_cuda_agent_sections(text)
        formatted = format_cuda_agent_submission(partial_sections)
        if formatted is not None:
            return formatted

        if response_has_cuda_agent_markers(text):
            return strip_think_blocks(text).strip()

    if kernel_backend == "triton":
        patterns = [
            r"#\s*Kernel\s+Implementation\s*\n(.*?)(?=\#\s*End\b|$)",
            r"```python\s*#\s*Kernel\s*\n(.*?)```",
            r"#\s*Your\s+implementation:\s*\n(.*?)(?=\#\s*End\b|$)",
            r"#\s*Generated\s+kernel:\s*\n(.*?)(?=\#\s*End\b|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

    code_blocks = re.findall(r"```(?:[\w+-]+)?\s*\n?(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    return text
