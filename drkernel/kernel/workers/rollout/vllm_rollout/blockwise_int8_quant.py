"""
Blockwise INT8 W8A8 quantization config for vLLM.

Registers a custom ``"blockwise-int8"`` quantization method via vLLM's
``register_quantization_config`` decorator so that it can be selected from
engine kwargs without modifying vLLM source.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import ModelWeightParameter

logger = logging.getLogger(__name__)


@register_quantization_config("blockwise-int8")
class BlockwiseInt8Config(QuantizationConfig):
    """Block-wise W8A8 INT8 quantization with dynamic activation quantization."""

    def __init__(
        self,
        weight_block_size: List[int] | None = None,
        ignored_layers: List[str] | None = None,
    ) -> None:
        self.weight_block_size = weight_block_size or [128, 128]
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> str:
        return "blockwise-int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BlockwiseInt8Config":
        weight_block_size = config.get("weight_block_size", [128, 128])
        ignored_layers = config.get("ignored_layers", None)
        return cls(weight_block_size=weight_block_size, ignored_layers=ignored_layers)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return BlockwiseInt8LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class BlockwiseInt8LinearMethod(QuantizeMethodBase):
    """Linear method for block-wise W8A8 INT8 quantization.

    Weights are statically quantised to int8 with per-block scales.
    Activations are dynamically quantised per-token-group at inference time.
    """

    def __init__(self, quant_config: BlockwiseInt8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        block_n, block_k = self.quant_config.weight_block_size

        tp_size = get_tensor_model_parallel_world_size()

        # Validate block alignment for tensor parallelism
        if tp_size > 1 and input_size // input_size_per_partition == tp_size:
            if input_size_per_partition % block_k != 0:
                raise ValueError(
                    f"input_size_per_partition={input_size_per_partition} "
                    f"is not divisible by block_k={block_k}."
                )
        if (
            tp_size > 1 and output_size // output_size_per_partition == tp_size
        ) or len(output_partition_sizes) > 1:
            for ops in output_partition_sizes:
                if ops % block_n != 0:
                    raise ValueError(
                        f"output_partition_size={ops} "
                        f"is not divisible by block_n={block_n}."
                    )

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT -- load as the model's native dtype; quantize in process_weights_after_loading
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Dynamic activation quantization -- no static input scale
        layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: Module) -> None:
        """Quantize bf16/fp16 weights to int8 with per-block scales at load time."""
        from kernel.workers.rollout.vllm_rollout.online_quant_utils import (
            quantize_linear_weight_block_rtn,
        )

        weight_data = layer.weight.data
        block_size = self.quant_config.weight_block_size
        quantized_weight, block_scale = quantize_linear_weight_block_rtn(
            weight_data, block_size=block_size,
        )
        layer.weight = torch.nn.Parameter(
            quantized_weight.to(weight_data.device), requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            block_scale.to(weight_data.device), requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from kernel.workers.rollout.vllm_rollout.blockwise_int8_kernels import (
            apply_w8a8_block_int8_linear,
        )

        return apply_w8a8_block_int8_linear(
            input=x,
            weight=layer.weight,
            block_size=self.quant_config.weight_block_size,
            weight_scale=layer.weight_scale,
            input_scale=None,
            bias=bias,
        )
