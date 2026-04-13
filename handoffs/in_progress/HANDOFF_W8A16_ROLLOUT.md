# W8A16 Online Quantized Rollout — Investigation Handoff

## Status: BLOCKED — engine runs but produces gibberish output

W8A16 per-channel INT8 weight-only online quantization on `vllm==0.18.0` has been unblocked through init and device issues, but the final output is garbage — the Marlin kernel computes wrong results due to a weight packing format mismatch.

## Evidence of Gibberish Output

1-node run `drkernel/logs/trloo-14b.train.8XA800.reward.16x4090.bf16.spare2.refcache.20260330-024304/`:

```
model_response: _roles俪pe_album_characters stro_votereed-shopping_front...
model_logprob_mean: -7.12  (should be ~-1 to -2 for coherent text)
code_block_count: 0
```

The engine initializes, accepts weights, runs Marlin repack, and generates tokens — but the results are wrong.

## Why W8A16 Is Hard (and W8A8 Isn't)

vLLM 0.18.0 has two INT8 compressed-tensors schemes:

| | W8A8 (`CompressedTensorsW8A8Int8`) | W8A16 (`CompressedTensorsWNA16`) |
|---|---|---|
| Format | `int-quantized` | `pack-quantized` |
| Weight param name | `weight` (same as bf16) | `weight_packed` (int32, packed) |
| Weight dtype | int8 | int32 (4 int8 values per element) |
| Kernel | CUTLASS int8 GEMM | Marlin/AllSpark with repack |
| Init from bf16 | Works (name matches, missing scales tolerated) | Fails (`weight` vs `weight_packed` mismatch) |
| Weight sync | Direct: int8 tensor → `weight` | Complex: int8 → pack int32 → `weight_packed` + `weight_shape` → Marlin repack |

W8A8 works because the `weight` parameter name is unchanged and the int8 format is consumed directly by CUTLASS. W8A16 requires a multi-step packing pipeline through Marlin's proprietary format.

## Issues Fixed (in order)

### 1. Init-time checkpoint mismatch

WNA16 registers `weight_packed` but bf16 checkpoint has `weight`.

**Fix**: `load_format=dummy` in `online_quant_utils.py` for W8A16 preset. Engine creates with random dummy weights instead of loading from checkpoint.

### 2. `process_weights_after_loading` on CPU during dummy init

Both AllSpark (`rearrange_kn_weight_as_n32k16_order`) and Marlin (`gptq_marlin_repack`) require CUDA, but the EngineCore subprocess runs dummy init with weights on CPU.

**Fix**: Patched vLLM's `process_weights_after_loading` in `utils.py` to skip when `KERNELGYM_SKIP_QUANT_PROCESS_WEIGHTS=1`. The repack deferred to first weight sync.

### 3. Env var not reaching EngineCore subprocess

The standalone async engine path (`async_server.py`) creates Ray actors with an explicit `runtime_env` containing only 3 vars, overriding inherited env.

**Fix**: Added `KERNELGYM_SKIP_QUANT_PROCESS_WEIGHTS` and `VLLM_DISABLED_KERNELS` to the standalone runtime_env in `async_server.py`. Also added both to `PPO_RAY_PASSTHROUGH_ENV_VARS` in `constants_ppo.py`.

### 4. `process_weights_after_loading` on CPU during layerwise weight reload

After dummy init, the first weight sync triggers `_layerwise_process` → `quant_method.process_weights_after_loading(layer)` in `layerwise.py:238` without `device_loading_context`. Weights arrive on CPU, Marlin repack fails.

**Fix**: Patched vLLM's `layerwise.py:238` to wrap with `device_loading_context(layer, cuda)` before calling `process_weights_after_loading`.

### 5. AllSpark selected before Marlin on A800

vLLM's kernel priority: `CutlassW4A8 → Machete → AllSpark → Marlin → ...`. AllSpark has `min_capability=80` (A800 = Ampere cc=80), selected before Marlin.

**Fix**: `VLLM_DISABLED_KERNELS=AllSparkLinearKernel` in launch script + Ray passthrough. Forces Marlin.

## Current Blocker: Packing Format Mismatch

All init/device/env issues are resolved. The engine runs with Marlin kernel and generates tokens. But the output is gibberish.

### What we produce

1. `quantize_linear_weight_rtn()` → per-channel int8 tensor `[out_features, in_features]`
2. `pack_int8_to_packed_int32()` → int32 tensor `[out_features, in_features/4]` by ORing 4 unsigned int8 values at bit offsets `0, 8, 16, 24`
3. `weight_scale` → float32 `[out_features, 1]`
4. `weight_shape` → int64 `[2]` with original dimensions
5. Sent to EngineCore → `_layerwise_process` → `process_weights_after_loading` → Marlin's `gptq_marlin_repack`

### What Marlin expects

Marlin's `process_weights_after_loading` does:
1. `permute_param_layout_(weight_packed, input_dim=0, output_dim=1, packed_dim=0)` — transposes dims
2. `ops.gptq_marlin_repack(weight_packed, perm, size_k, size_n, num_bits)` — reshuffles to Marlin layout
3. `marlin_permute_scales(weight_scale, ...)` — reshuffles scales

The input to `gptq_marlin_repack` must be in standard GPTQ packed format with specific dimension ordering. Our `pack_int8_to_packed_int32` may differ in:
- **Pack dimension**: we pack along `input_dim` (dim=1), but Marlin may expect packing along `output_dim` (dim=0)
- **Byte order within int32**: GPTQ may use a different bit packing convention
- **Scale shape**: Marlin may expect `[1, out_features]` not `[out_features, 1]`
- **Weight layout**: the `PackedvLLMParameter` has `full_weight_shape`, `partition_weight_shape` metadata that affects `permute_param_layout_`

## Code Changes Made

### In this repo

| File | Change |
|---|---|
| `online_quant_utils.py` | `format: pack-quantized` for W8A16; `load_format=dummy`; `KERNELGYM_SKIP_QUANT_PROCESS_WEIGHTS=1`; `pack_int8_to_packed_int32` helper; disable AllSpark |
| `fsdp_vllm.py` | W8A16 weight sync: pack int8→int32, yield `weight_packed` + `weight_scale` + `weight_shape`; CUDA materialize forced |
| `async_server.py` | Propagate `KERNELGYM_SKIP_QUANT_PROCESS_WEIGHTS` and `VLLM_DISABLED_KERNELS` to standalone engine runtime_env |
| `constants_ppo.py` | Added `VLLM_DISABLED_KERNELS` and `KERNELGYM_SKIP_QUANT_PROCESS_WEIGHTS` to Ray passthrough |
| `run_train_bf16.sh` | `VAL_BEFORE_TRAIN` respects env override |
| `run_train_bf16_2node.sh` | `VAL_BEFORE_TRAIN`, `ENFORCE_EAGER` respect env overrides; `VLLM_DISABLED_KERNELS=AllSparkLinearKernel` |

### In vLLM installed package (`.venv-vllm0180`)

| File | Change |
|---|---|
| `model_executor/model_loader/utils.py` | Skip `process_weights_after_loading` when `KERNELGYM_SKIP_QUANT_PROCESS_WEIGHTS=1` |
| `model_executor/model_loader/reload/layerwise.py` | Added `device_loading_context(layer, cuda)` around `process_weights_after_loading` call at line 238 |

## Recommended Next Steps

### To fix W8A16 packing format

1. Study GPTQ's exact int8→int32 packing convention (check `auto_gptq` or `compressed-tensors` reference)
2. Check if `permute_param_layout_` expects `[in, out/pack_factor]` not `[out, in/pack_factor]`
3. Check `PackedvLLMParameter` metadata (`packed_dim`, `packed_factor`) and match our packing to it
4. Compare with a known-good GPTQ-quantized INT8 checkpoint to validate the format

### Alternative: use W8A8

W8A8 already works, produces coherent output, and the only difference is dynamic per-token activation quantization. For a 14B model this has minimal accuracy impact. If W8A16 is not strictly required, W8A8 is the pragmatic path.

## Key Runs

| Run | Config | Result |
|---|---|---|
| `...bf16.spare2.refcache.20260330-024304` | 1-node, W8A16, val | Engines run, **gibberish output** |
| `...bf16.2node.hybrid.20260330-021854` | 2-node, W8A16, standalone | EngineCore CPU error (layerwise path) |
| `...bf16.2node.hybrid.20260329-042556` | 2-node, bf16, reduce_dtype=bf16 | **Working baseline** — 20.1m/step |
