"""Validation helpers for KernelBench toolkit."""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from kernelgym.common import ErrorCode


def validate_code(code: str, entry_point: str = "Model") -> Tuple[bool, str]:
    """Basic validation of PyTorch code."""
    try:
        if not code:
            return False, "Code is required"
        if f"class {entry_point}" not in code:
            return False, f"Code must contain a '{entry_point}' class"
        return True, ""
    except Exception as exc:
        return False, f"Code validation error: {exc}"


def early_kernel_validation(
    kernel_code: str,
    backend: str = "triton",
    entry_point: str = "Model",
) -> Tuple[bool, str, Optional[ErrorCode]]:
    """Perform early kernel code validation without GPU resources."""
    try:
        kernel_entry_point = f"{entry_point}New"
        is_valid, error_msg = validate_code(kernel_code, kernel_entry_point)
        if not is_valid:
            return False, error_msg, ErrorCode.VALIDATION_ERROR

        try:
            compile(kernel_code, "<string>", "exec")
        except SyntaxError as e:
            return False, f"Syntax error in kernel code: {str(e)}", ErrorCode.SYNTAX_ERROR

        if backend == "triton":
            required_imports = ["import triton", "from triton import"]
            if not any(imp in kernel_code for imp in required_imports):
                return False, "Kernel code must import triton for triton backend", ErrorCode.IMPORT_ERROR
        elif backend == "cuda":
            cuda_indicators = [
                "torch.cuda",
                "cuda_kernel",
                "@cuda.jit",
                "from numba import cuda",
            ]
            if not any(indicator in kernel_code for indicator in cuda_indicators):
                return False, "Kernel code must contain CUDA kernel code for cuda backend", ErrorCode.IMPORT_ERROR

        kernel_patterns = [
            "@triton.jit",
            "def.*kernel.*\\(",
            "torch\\.cuda",
            "\\.cuda\\(",
        ]

        import re

        has_kernel_pattern = any(re.search(pattern, kernel_code) for pattern in kernel_patterns)
        if not has_kernel_pattern and backend == "triton":
            return True, "", None

        try:
            test_code = f"""
            import torch
            import torch.nn as nn
            {kernel_code}

            try:
                model = {kernel_entry_point}()
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate {kernel_entry_point}: {{e}}")
            """
            compile(test_code, "<test>", "exec")
        except Exception as e:
            error_str = str(e)
            if "Failed to instantiate" in error_str:
                return False, error_str, ErrorCode.INSTANTIATION_ERROR

        return True, "", None

    except Exception as e:
        return False, f"Early validation error: {str(e)}", ErrorCode.VALIDATION_ERROR


def _find_register_binding_semicolon_issue(source_map: dict[str, str]) -> tuple[str, int] | None:
    marker = "REGISTER_BINDING("
    for filename, content in source_map.items():
        search_start = 0
        while True:
            marker_index = content.find(marker, search_start)
            if marker_index == -1:
                break

            depth = 0
            closing_index: int | None = None
            for index in range(marker_index, len(content)):
                char = content[index]
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        closing_index = index
                        break

            if closing_index is None:
                line_no = content.count("\n", 0, marker_index) + 1
                return filename, line_no

            next_index = closing_index + 1
            while next_index < len(content) and content[next_index].isspace():
                next_index += 1

            if next_index >= len(content) or content[next_index] != ";":
                line_no = content.count("\n", 0, marker_index) + 1
                return filename, line_no

            search_start = closing_index + 1

    return None


def precheck_cuda_agent_submission(
    model_code: str,
    cuda_sources: dict[str, str],
    *,
    entry_point: str = "ModelNew",
) -> Tuple[str, Optional[ErrorCode], dict[str, Any]]:
    """Run a cheap structure precheck before CUDA-Agent compilation."""

    source_map = cuda_sources or {}
    precheck: dict[str, Any] = {
        "passed": False,
        "cuda_source_count": len(source_map),
        "binding_files": [],
    }

    def _fail(message: str, code: Optional[ErrorCode]) -> Tuple[str, Optional[ErrorCode], dict[str, Any]]:
        formatted = f"Precheck failed: {message}"
        precheck["error_message"] = formatted
        precheck["error_code"] = code.name if code is not None else None
        return formatted, code, precheck

    try:
        is_valid, error_msg = validate_code(model_code, entry_point)
        if not is_valid:
            return _fail(error_msg, ErrorCode.VALIDATION_ERROR)

        try:
            compile(model_code, "<string>", "exec")
        except SyntaxError as exc:
            return _fail(f"Syntax error in model code: {exc}", ErrorCode.SYNTAX_ERROR)

        if "cuda_extension" not in model_code:
            return _fail(
                "model_new.py must import or reference cuda_extension",
                ErrorCode.IMPORT_ERROR,
            )

        if not source_map:
            return _fail(
                "CUDA sources are required for CUDA-Agent compilation",
                ErrorCode.VALIDATION_ERROR,
            )

        cu_files = [name for name in source_map if name.lower().endswith(".cu")]
        cpp_files = [
            name for name in source_map if name.lower().endswith((".cpp", ".cc", ".cxx"))
        ]
        header_files = [
            name for name in source_map if name.lower().endswith((".h", ".hpp", ".hh", ".cuh"))
        ]
        precheck["cu_files"] = cu_files
        precheck["cpp_files"] = cpp_files
        precheck["header_files"] = header_files

        if not cu_files:
            return _fail(
                "CUDA-Agent sources must include at least one .cu file",
                ErrorCode.VALIDATION_ERROR,
            )

        binding_candidates = [
            name for name in cpp_files if "binding" in name.lower() or "bind" in name.lower()
        ]
        precheck["binding_files"] = binding_candidates
        if not binding_candidates:
            return _fail(
                "CUDA-Agent sources must include a binding .cpp file",
                ErrorCode.VALIDATION_ERROR,
            )

        combined_cpp = "\n".join(str(source_map[name]) for name in cpp_files)
        include_markers = (
            '#include "../binding_registry.h"',
            '#include "binding_registry.h"',
        )
        if not any(marker in combined_cpp for marker in include_markers):
            return _fail(
                "Binding source must include binding_registry.h",
                ErrorCode.VALIDATION_ERROR,
            )

        if "REGISTER_BINDING(" not in combined_cpp:
            return _fail(
                "Binding source must register functions with REGISTER_BINDING(...)",
                ErrorCode.VALIDATION_ERROR,
            )

        binding_issue = _find_register_binding_semicolon_issue(source_map)
        if binding_issue is not None:
            issue_file, issue_line = binding_issue
            return _fail(
                f"{issue_file}:{issue_line} has REGISTER_BINDING(...) without a trailing ';'",
                ErrorCode.SYNTAX_ERROR,
            )

        precheck["detected_extension_calls"] = sorted(
            set(
                re.findall(
                    r"(?:cuda_extension|torch\.ops\.cuda_extension)\.([A-Za-z_]\w*)\s*\(",
                    model_code,
                )
            )
        )
        precheck["passed"] = True
        precheck["error_message"] = ""
        precheck["error_code"] = None
        return "", None, precheck
    except Exception as exc:
        return _fail(f"CUDA-Agent precheck error: {exc}", ErrorCode.VALIDATION_ERROR)
