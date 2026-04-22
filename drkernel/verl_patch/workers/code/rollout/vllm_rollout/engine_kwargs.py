from __future__ import annotations

from typing import Any


def get_vllm_engine_kwargs(config: Any) -> dict[str, Any]:
    """Return explicit vLLM engine kwargs from rollout config.

    Values set to None are intentionally omitted so vLLM can keep its version
    specific defaults.
    """

    engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}
    engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
    if config.get("limit_images", None):
        engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}
    return engine_kwargs
