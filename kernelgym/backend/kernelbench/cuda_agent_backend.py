"""CUDA-Agent-specific KernelBench backend implementation."""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict

import torch
import torch.utils.cpp_extension as cpp_ext

from kernelgym.toolkit.validation import precheck_cuda_agent_submission

from .base import KernelBenchBackendBase


_CUDA_AGENT_TMPDIR_ENV = "KERNELGYM_CUDA_AGENT_TMPDIR"
_CUDA_AGENT_DEFAULT_TMPDIR = "/dev/shm"
_CUDA_AGENT_MIN_TMPDIR_FREE_BYTES = 512 * 1024 * 1024


class KernelBenchCudaAgentBackend(KernelBenchBackendBase):
    """Compile and load CUDA-Agent style submissions."""

    name = "kernelbench.cuda_agent"

    @staticmethod
    def _default_binding_cpp() -> str:
        return """#include <pybind11/pybind11.h>
#include "binding_registry.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BindingRegistry::getInstance().applyBindings(m);
}
"""

    @staticmethod
    def _default_binding_registry_h() -> str:
        return """#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class BindingRegistry {
public:
    using BindingFunction = std::function<void(pybind11::module&)>;

    static BindingRegistry& getInstance() {
        static BindingRegistry instance;
        return instance;
    }

    void registerBinding(const std::string& name, BindingFunction func) {
        bindings_.push_back({name, std::move(func)});
    }

    void applyBindings(pybind11::module& m) {
        for (auto& binding : bindings_) {
            binding.second(m);
        }
    }

private:
    std::vector<std::pair<std::string, BindingFunction>> bindings_;
    BindingRegistry() = default;
};

class BindingRegistrar {
public:
    BindingRegistrar(const std::string& name, BindingRegistry::BindingFunction func) {
        BindingRegistry::getInstance().registerBinding(name, std::move(func));
    }
};

#define REGISTER_BINDING(name, func) \\
    static BindingRegistrar _registrar_##name(#name, [](pybind11::module& m) { func(m); })
"""

    @staticmethod
    def _parse_embedded_sources(code: str) -> tuple[dict[str, str], str]:
        import re

        legacy_match = re.search(
            r"###\s*CUDA_SOURCES\s*###\s*(.*?)###\s*END_CUDA_SOURCES\s*###",
            code,
            re.DOTALL | re.IGNORECASE,
        )
        if legacy_match is not None:
            source_blob = legacy_match.group(1).strip()
            parsed = ast.literal_eval(source_blob)
            if not isinstance(parsed, dict):
                raise TypeError("Embedded CUDA_SOURCES must evaluate to a dict[str, str]")
            normalized = {str(name): str(content) for name, content in parsed.items()}
            python_code = f"{code[:legacy_match.start()]}{code[legacy_match.end():]}".strip()
            return normalized, python_code

        section_patterns = {
            "CUDA_KERNELS": re.compile(
                r"###\s*CUDA_KERNELS\s*```(?:cpp|c\+\+)?\s*\n(.*?)```",
                re.DOTALL | re.IGNORECASE,
            ),
            "APPLY_BINDINGS": re.compile(
                r"###\s*APPLY_BINDINGS\s*```(?:cpp|c\+\+)?\s*\n(.*?)```",
                re.DOTALL | re.IGNORECASE,
            ),
            "MODEL_NEW": re.compile(
                r"###\s*MODEL_NEW\s*```python\s*\n(.*?)```",
                re.DOTALL | re.IGNORECASE,
            ),
        }
        section_matches = {
            name: pattern.search(code) for name, pattern in section_patterns.items()
        }
        if not any(section_matches.values()):
            return {}, code.strip()

        cuda_sources: dict[str, str] = {}
        if section_matches["CUDA_KERNELS"] is not None:
            cuda_sources["kernels/generated.cu"] = section_matches["CUDA_KERNELS"].group(1).strip()
        if section_matches["APPLY_BINDINGS"] is not None:
            cuda_sources["kernels/generated_binding.cpp"] = section_matches["APPLY_BINDINGS"].group(1).strip()
        python_code = (
            section_matches["MODEL_NEW"].group(1).strip()
            if section_matches["MODEL_NEW"] is not None
            else ""
        )
        return cuda_sources, python_code

    @staticmethod
    def _normalize_cuda_sources_input(cuda_sources: Any) -> dict[str, str]:
        if not cuda_sources:
            return {}
        if not isinstance(cuda_sources, dict):
            raise TypeError("cuda_sources must be a dict[str, str]")
        return {str(name): str(content) for name, content in cuda_sources.items()}

    @staticmethod
    def _select_work_dir_parent() -> str | None:
        candidates = [
            os.environ.get(_CUDA_AGENT_TMPDIR_ENV),
            _CUDA_AGENT_DEFAULT_TMPDIR,
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            try:
                if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
                    continue
                if shutil.disk_usage(path).free < _CUDA_AGENT_MIN_TMPDIR_FREE_BYTES:
                    continue
            except OSError:
                continue
            return str(path)
        return None

    def _create_work_dir(self) -> Path:
        parent = self._select_work_dir_parent()
        return Path(tempfile.mkdtemp(prefix="kernelgym_cuda_agent_", dir=parent))

    def _write_runtime_scaffold(self, work_dir: Path, model_code: str) -> None:
        (work_dir / "__init__.py").write_text("", encoding="utf-8")
        (work_dir / "model_new.py").write_text(model_code, encoding="utf-8")
        (work_dir / "binding.cpp").write_text(
            self._default_binding_cpp(),
            encoding="utf-8",
        )
        (work_dir / "binding_registry.h").write_text(
            self._default_binding_registry_h(),
            encoding="utf-8",
        )

    @staticmethod
    def _materialize_sources(work_dir: Path, cuda_sources: dict[str, str]) -> None:
        for filename, content in cuda_sources.items():
            relative_path = Path(filename)
            if relative_path.parent == Path("."):
                file_path = work_dir / "kernels" / relative_path
            else:
                file_path = work_dir / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

    @staticmethod
    def _collect_compile_sources(work_dir: Path) -> list[str]:
        sources: list[str] = []
        for path in work_dir.rglob("*"):
            if not path.is_file():
                continue
            if "build" in path.parts:
                continue
            if path.suffix.lower() in {".cu", ".cpp", ".cc", ".cxx"}:
                sources.append(str(path))
        return sorted(set(sources))

    @staticmethod
    def _extract_custom_kernel_names(cuda_sources: dict[str, str]) -> list[str]:
        kernel_pattern = re.compile(
            r"__global__\s+"
            r"(?:__launch_bounds__\s*\([^)]*\)\s*)?"
            r"(?:(?:inline|static|constexpr|__forceinline__|__host__|__device__)\s+)*"
            r"(?:[\w:<>]+\s+)+"
            r"([A-Za-z_]\w*)\s*\(",
            re.MULTILINE,
        )
        kernel_names: set[str] = set()
        for filename, content in cuda_sources.items():
            if not filename.lower().endswith(".cu"):
                continue
            kernel_names.update(kernel_pattern.findall(content))
        return sorted(kernel_names)

    @staticmethod
    def _build_extension(work_dir: Path, sources: list[str]) -> Dict[str, Any]:
        if not sources:
            return {
                "compiled": False,
                "error": "No CUDA source files found for CUDA-Agent compilation",
            }

        build_dir = work_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        ext_name = work_dir.name.replace("-", "_")
        module = cpp_ext.load(
            name=ext_name,
            sources=sources,
            build_directory=str(build_dir),
            verbose=False,
            with_cuda=True,
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
        return {
            "compiled": True,
            "so_path": getattr(module, "__file__", ""),
            "module_name": ext_name,
        }

    def compile(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        device = self._normalize_device(kwargs.get("device"))
        entry_point = kwargs.get("entry_point", "ModelNew")
        explicit_sources = self._normalize_cuda_sources_input(kwargs.get("cuda_sources"))

        try:
            embedded_sources, python_code = self._parse_embedded_sources(code)
        except Exception as exc:
            return {
                "compiled": False,
                "error": f"Failed to parse CUDA-Agent submission: {exc}",
                "device": str(device),
                "entry_point": entry_point,
                "backend": "cuda_agent",
            }

        cuda_sources = {**explicit_sources, **embedded_sources}
        model_code = python_code.strip() or code.strip()

        precheck_error, _precheck_code, precheck_info = precheck_cuda_agent_submission(
            model_code,
            cuda_sources,
            entry_point=entry_point,
        )
        if not precheck_info.get("passed", False):
            return {
                "compiled": False,
                "error": precheck_error,
                "device": str(device),
                "entry_point": entry_point,
                "backend": "cuda_agent",
                "precheck": precheck_info,
            }

        work_dir = self._create_work_dir()
        work_dir.mkdir(parents=True, exist_ok=True)
        self._write_runtime_scaffold(work_dir, model_code)
        self._materialize_sources(work_dir, cuda_sources)

        try:
            result = self._build_extension(work_dir, self._collect_compile_sources(work_dir))
        except Exception as exc:
            result = {"compiled": False, "error": str(exc)}

        return {
            "compiled": bool(result.get("compiled")),
            "error": result.get("error"),
            "device": str(device),
            "entry_point": entry_point,
            "backend": "cuda_agent",
            "work_dir": str(work_dir),
            "so_path": result.get("so_path"),
            "module_name": result.get("module_name"),
            "code": model_code,
            "precheck": precheck_info,
            "profiling_hints": {
                "backend": "cuda_agent",
                "custom_kernel_names": self._extract_custom_kernel_names(cuda_sources),
                "detected_extension_calls": list(
                    precheck_info.get("detected_extension_calls", [])
                ),
                "source_files": sorted(cuda_sources.keys()),
            },
        }

    @staticmethod
    def _load_extension_module(module_name: str, so_path: Path) -> types.ModuleType:
        loader = importlib.machinery.ExtensionFileLoader(module_name, str(so_path))
        spec = importlib.util.spec_from_file_location(module_name, str(so_path), loader=loader)
        if spec is None:
            raise ImportError(f"Failed to create import spec for {so_path}")
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module

    def load(self, artifact: Dict[str, Any], **kwargs: Any) -> Any:
        code = artifact.get("code")
        entry_point = artifact.get("entry_point", "ModelNew")
        work_dir = artifact.get("work_dir")
        so_path = artifact.get("so_path")
        module_name = artifact.get("module_name")
        context = kwargs.get("context") or {}

        if not code:
            raise ValueError("KernelBenchCudaAgentBackend.load requires kernel code in artifact")
        if not work_dir:
            raise ValueError("KernelBenchCudaAgentBackend.load requires a work_dir in artifact")
        if not so_path or not Path(so_path).exists():
            raise ValueError(f"Compiled shared library not found: {so_path}")
        if not module_name:
            raise ValueError("KernelBenchCudaAgentBackend.load requires module_name in artifact")

        device = self._normalize_device(kwargs.get("device") or artifact.get("device"))
        self._maybe_set_cuda_device(device)
        os.environ["TORCH_USE_CUDA_DSA"] = "1"

        work_dir_path = Path(work_dir)
        so_path_path = Path(so_path)
        runtime_package_name = f"_kernelgym_cuda_agent_{work_dir_path.name.replace('-', '_')}"
        package_module = types.ModuleType(runtime_package_name)
        package_module.__path__ = [str(work_dir_path)]  # type: ignore[attr-defined]
        package_module.__package__ = runtime_package_name
        sys.modules[runtime_package_name] = package_module

        ext_module = sys.modules.get(module_name)
        if ext_module is None:
            ext_module = self._load_extension_module(module_name, so_path_path)

        module_aliases = [
            module_name,
            "cuda_extension",
            f"{runtime_package_name}.cuda_extension",
            runtime_package_name,
            f"{runtime_package_name}.model_new",
        ]
        sys.modules["cuda_extension"] = ext_module
        sys.modules[f"{runtime_package_name}.cuda_extension"] = ext_module

        model_name = f"{runtime_package_name}.model_new"
        model_file = work_dir_path / "model_new.py"
        spec = importlib.util.spec_from_file_location(model_name, str(model_file))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create import spec for {model_file}")
        model_module = importlib.util.module_from_spec(spec)
        sys.modules[model_name] = model_module
        spec.loader.exec_module(model_module)

        model_cls = getattr(model_module, entry_point, None)
        if model_cls is None:
            raise ValueError(f"Failed to load model class '{entry_point}' from code")

        return {
            "model_cls": model_cls,
            "context": context,
            "backend": "cuda_agent",
            "entry_point": entry_point,
            "device": device,
            "work_dir": str(work_dir_path),
            "so_path": str(so_path_path),
            "module_aliases": module_aliases,
            "tempfile_handle": None,
            "profiling_hints": artifact.get("profiling_hints", {}),
        }

    def cleanup(self, handle: Any, **kwargs: Any) -> None:
        super().cleanup(handle, **kwargs)
        if not isinstance(handle, dict):
            return

        for module_name in handle.get("module_aliases", []):
            sys.modules.pop(module_name, None)

        work_dir = handle.get("work_dir")
        if work_dir and Path(work_dir).exists():
            shutil.rmtree(work_dir, ignore_errors=True)
