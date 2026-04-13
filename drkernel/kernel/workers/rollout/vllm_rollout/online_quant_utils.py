import fnmatch
from typing import Any

import torch
from compressed_tensors.quantization import preset_name_to_scheme

DEFAULT_ONLINE_QUANT_IGNORE = ("lm_head",)
SUPPORTED_ONLINE_QUANT_PRESETS = {"W8A8", "W8A16", "W8A8_BLOCK"}


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def normalize_online_quantization_preset(preset: Any) -> str | None:
    if preset is None:
        return None
    preset = str(preset).strip()
    if not preset:
        return None
    preset = preset.upper()
    if preset not in SUPPORTED_ONLINE_QUANT_PRESETS:
        raise NotImplementedError(
            f"Unsupported rollout online quantization preset: {preset}. "
            f"Supported presets: {sorted(SUPPORTED_ONLINE_QUANT_PRESETS)}"
        )
    return preset


def get_online_quantization_preset(rollout_config: Any) -> str | None:
    return normalize_online_quantization_preset(
        _config_get(rollout_config, "online_quantization_preset", None)
    )


def get_online_quantization_ignore(rollout_config: Any) -> list[str]:
    ignore = _config_get(rollout_config, "online_quantization_ignore", None)
    if not ignore:
        return list(DEFAULT_ONLINE_QUANT_IGNORE)
    return list(ignore)


def build_compressed_tensors_ignore_list(ignore: list[str] | None) -> list[str]:
    resolved_ignore = list(ignore or DEFAULT_ONLINE_QUANT_IGNORE)
    ct_ignore: list[str] = []
    for pattern in resolved_ignore:
        if any(ch in pattern for ch in "*?["):
            ct_ignore.append(f"re:{fnmatch.translate(pattern)}")
        else:
            ct_ignore.append(pattern)
    return ct_ignore


def build_blockwise_int8_config_dict(
    ignore: list[str] | None = None,
    block_size: list[int] | None = None,
) -> dict[str, Any]:
    """Build a ``compression_config`` dict for the blockwise-int8 quant method."""
    return {
        "quant_method": "blockwise-int8",
        "weight_block_size": block_size or [128, 128],
        "ignored_layers": list(ignore or DEFAULT_ONLINE_QUANT_IGNORE),
    }


def build_online_quantization_config_dict(
    preset: str, ignore: list[str] | None = None
) -> dict[str, Any]:
    preset = normalize_online_quantization_preset(preset)
    if preset == "W8A8_BLOCK":
        return build_blockwise_int8_config_dict(ignore=ignore)
    scheme = preset_name_to_scheme(preset, ["Linear"])
    scheme_dict = scheme.model_dump()
    # vLLM scheme routing by format:
    # - "int-quantized" + input_activations → CompressedTensorsW8A8Int8 (param: "weight")
    # - "pack-quantized" + weight-only → CompressedTensorsWNA16 (param: "weight_packed")
    has_input_activations = scheme.input_activations is not None
    fmt = "int-quantized" if has_input_activations else "pack-quantized"
    return {
        "quant_method": "compressed-tensors",
        "format": fmt,
        "config_groups": {"group_0": scheme_dict},
        "ignore": build_compressed_tensors_ignore_list(ignore),
        "sparsity_config": {},
        "transform_config": {},
    }


def maybe_apply_online_quantization_engine_kwargs(
    rollout_config: Any,
    engine_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    resolved_engine_kwargs = dict(engine_kwargs or {})
    preset = get_online_quantization_preset(rollout_config)
    if preset is None:
        return resolved_engine_kwargs

    if preset == "W8A8_BLOCK":
        # Blockwise INT8 uses a custom vLLM quantization config
        # Ensure the registration module is imported so the decorator fires
        import kernel.workers.rollout.vllm_rollout.blockwise_int8_quant  # noqa: F401

        expected_quant = "blockwise-int8"
    else:
        expected_quant = "compressed-tensors"

    quantization = resolved_engine_kwargs.get("quantization")
    if quantization not in (None, expected_quant):
        raise ValueError(
            f"Rollout online quantization preset {preset} requires "
            f"engine_kwargs.vllm.quantization={expected_quant}"
        )
    resolved_engine_kwargs["quantization"] = expected_quant

    hf_overrides = dict(resolved_engine_kwargs.get("hf_overrides") or {})
    hf_overrides.setdefault(
        "compression_config",
        build_online_quantization_config_dict(
            preset, ignore=get_online_quantization_ignore(rollout_config)
        ),
    )
    resolved_engine_kwargs["hf_overrides"] = hf_overrides

    # WNA16 (W8A16) registers weight_packed instead of weight, so a bf16
    # checkpoint cannot satisfy the init-time load. Use dummy init so the
    # engine creates the model with random packed weights, then the
    # sharding manager syncs real quantized weights before first generation.
    #
    # Both Marlin and AllSpark repack ops in process_weights_after_loading
    # are CUDA-only, but the EngineCore subprocess creates the dummy model
    # with weights on CPU. Set KERNELGYM_SKIP_QUANT_PROCESS_WEIGHTS=1 so
    # the patched vLLM process_weights_after_loading skips the repack.
    # The repack runs later on CUDA via the sharding manager's
    # _process_online_quantized_weights_after_loading.
    if preset == "W8A16":
        resolved_engine_kwargs.setdefault("load_format", "dummy")
        import os
        os.environ["KERNELGYM_SKIP_QUANT_PROCESS_WEIGHTS"] = "1"

    return resolved_engine_kwargs


def pack_int8_to_packed_int32(int8_weight: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Pack an int8 weight tensor into GPTQ-style packed int32 format.

    For 8-bit: pack_factor = 4, so every 4 int8 values become one int32.
    Input shape:  [out_features, in_features]
    Output shape: [out_features, in_features // pack_factor]
    """
    pack_factor = 32 // num_bits
    out_features, in_features = int8_weight.shape
    assert in_features % pack_factor == 0, (
        f"in_features ({in_features}) must be divisible by pack_factor ({pack_factor})"
    )

    # Convert signed int8 to unsigned representation for bit packing
    uint8_weight = int8_weight.to(torch.int32) & 0xFF  # [out, in]

    # Reshape to [out, in // pack_factor, pack_factor]
    reshaped = uint8_weight.reshape(out_features, in_features // pack_factor, pack_factor)

    # Pack: shift each element by (i * num_bits) and OR together
    packed = torch.zeros(out_features, in_features // pack_factor, dtype=torch.int32,
                         device=int8_weight.device)
    for i in range(pack_factor):
        packed |= reshaped[:, :, i] << (i * num_bits)

    return packed.contiguous()


def quantize_linear_weight_rtn(
    weight: torch.Tensor,
    preset: str,
    target_scale_shape: tuple[int, ...] | torch.Size | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    preset = normalize_online_quantization_preset(preset)
    if preset not in {"W8A8", "W8A16"}:
        raise NotImplementedError(f"Unsupported online RTN preset: {preset}")

    weight_fp32 = weight.detach().to(dtype=torch.float32)
    
    # Determine quantization strategy from target scale shape
    is_per_channel = False
    reduce_dim = 1
    if target_scale_shape is not None:
        target_scale_shape = tuple(target_scale_shape)
        if target_scale_shape == (weight_fp32.shape[0], 1):
            is_per_channel = True
            reduce_dim = 1
        elif target_scale_shape == (weight_fp32.shape[1], 1):
            is_per_channel = True
            reduce_dim = 0
        elif target_scale_shape == (1,) or target_scale_shape == ():
            is_per_channel = False
        else:
            # If target shape is not per-channel but has elements, assume it wants 
            # a scalar per weight (e.g. for PerTensorScaleParameter in merged layers)
            is_per_channel = False
    else:
        # Default to per-channel if no target shape provided
        is_per_channel = True
        reduce_dim = 1

    if is_per_channel:
        abs_max = weight_fp32.abs().amax(dim=reduce_dim, keepdim=True)
    else:
        abs_max = weight_fp32.abs().max()

    scale_for_quant = torch.clamp(abs_max / 127.0, min=torch.finfo(torch.float32).eps)
    quantized = torch.round(weight_fp32 / scale_for_quant).clamp(-127, 127).to(torch.int8)
    
    if is_per_channel:
        scale = scale_for_quant if reduce_dim == 1 else scale_for_quant.transpose(0, 1)
    else:
        scale = scale_for_quant.reshape(1) if scale_for_quant.dim() == 0 else scale_for_quant

    return quantized.contiguous(), scale.to(torch.float32).contiguous()


def quantize_linear_weight_block_rtn(
    weight: torch.Tensor,
    block_size: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise RTN quantization of a 2-D weight tensor to int8.

    Args:
        weight: ``[out_features, in_features]`` float weight.
        block_size: ``[block_n, block_k]`` (default ``[128, 128]``).

    Returns:
        (quantized_weight_int8, block_scale_float32)
        where ``block_scale`` has shape
        ``[ceil(out/block_n), ceil(in/block_k)]``.
    """
    if block_size is None:
        block_size = [128, 128]
    block_n, block_k = block_size

    weight_fp32 = weight.detach().to(dtype=torch.float32)
    out_features, in_features = weight_fp32.shape

    # Pad to multiples of block size if necessary
    pad_n = (block_n - out_features % block_n) % block_n
    pad_k = (block_k - in_features % block_k) % block_k
    if pad_n > 0 or pad_k > 0:
        weight_padded = torch.nn.functional.pad(weight_fp32, (0, pad_k, 0, pad_n))
    else:
        weight_padded = weight_fp32

    padded_n, padded_k = weight_padded.shape
    n_blocks = padded_n // block_n
    k_blocks = padded_k // block_k

    # Reshape into blocks: [n_blocks, block_n, k_blocks, block_k]
    weight_blocks = weight_padded.reshape(n_blocks, block_n, k_blocks, block_k)
    # Compute per-block abs max -> [n_blocks, k_blocks]
    abs_max = weight_blocks.abs().amax(dim=(1, 3))
    scale = torch.clamp(abs_max / 127.0, min=torch.finfo(torch.float32).eps)

    # Quantize: broadcast scale back -> [n_blocks, 1, k_blocks, 1]
    quantized_blocks = torch.round(
        weight_blocks / scale[:, None, :, None]
    ).clamp(-127, 127).to(torch.int8)

    # Reshape back to padded shape then trim
    quantized = quantized_blocks.reshape(padded_n, padded_k)[:out_features, :in_features]

    # Scale shape: [ceil(out/block_n), ceil(in/block_k)]
    # We computed on padded dims so n_blocks/k_blocks already equal ceil values
    block_scale = scale  # already [n_blocks, k_blocks]

    return quantized.contiguous(), block_scale.to(torch.float32).contiguous()


def dequantize_linear_weight_rtn(
    quantized_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    dequant = quantized_weight.to(torch.float32) * weight_scale.to(torch.float32)
    return dequant.to(output_dtype).contiguous()
