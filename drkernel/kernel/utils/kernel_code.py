"""Helpers for extracting kernel submissions from model responses."""

from __future__ import annotations

import re


CUDA_AGENT_SECTION_ORDER = (
    ("CUDA_KERNELS", "cpp"),
    ("APPLY_BINDINGS", "cpp"),
    ("MODEL_NEW", "python"),
)


def extract_cuda_agent_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for section_name, language in CUDA_AGENT_SECTION_ORDER:
        if language == "python":
            pattern = rf"###\s*{section_name}\s*```python\s*\n(.*?)```"
        else:
            pattern = rf"###\s*{section_name}\s*```(?:cpp|c\+\+)?\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[section_name] = match.group(1).strip()
    return sections


def format_cuda_agent_submission(sections: dict[str, str]) -> str | None:
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
        formatted = format_cuda_agent_submission(extract_cuda_agent_sections(text))
        if formatted is not None:
            return formatted

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
