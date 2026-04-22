# Triton And CUDA Prompt Extracts

This file extracts the prompt texts requested for direct comparison.

- Triton first-turn prompt source: `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM/drkernel/data/drkernel-validation-data/validation_data_thinking.parquet`, row 0, `prompt[0].content`
- Triton multi-turn template source: `/nfs/FM/lihongbin/CODE/KernelGYM/drkernel/kernel/config/prompt_config/multi_turn_kernel.yaml`
- CUDA first-turn prompt source: `/nfs/FM/lihongbin/CODE/KernelGYM/drkernel/cuda_prompt.txt`
- CUDA multi-turn template source: `/nfs/FM/lihongbin/CODE/KernelGYM/drkernel/kernel/config/prompt_config/multi_turn_cuda_kernel.yaml`

## Triton First-Turn Prompt

````text
You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups. 

    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.


        Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is:

        ```

        import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []


        ```

        The example new arch with custom Triton kernels looks like this:

        ```
        import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise addition
    out = x + y
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # Instead of "return a + b", call our Triton-based addition
        return triton_add(a, b)
        ```
        
    You are given the following architecture:
    ```
    import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.clamp(x, min=self.min_value)
        x = x / self.divisor
        return x

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]
    ```
    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
````

## Triton Multi-Turn Template

````yaml
method: system
prompt: null

per_turn_prompts:
  - name: first_turn
    condition: "current_turn == 0"
    history_mode: all
    skip_env: 0.0
    response_truncation: python_code, answer_code
    template: null
  - name: tool_response
    condition: "make_up_tool_response == True"
    history_mode: all
    skip_env: 0.0
    response_truncation: python_code, answer_code
    update_memory: false
    template: |
      Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.

      Here is the server feedback. Please refer to this feedback to improve the implementation:
      Server feedback (status/metrics/errors):
      {feedback}

      Return an improved Triton implementation named `ModelNew` as a single ```python``` block. Let's think step by step.
````

## lihongbin CUDA First-Turn Prompt

````text
You are a PyTorch and CUDA expert. Accelerate the given PyTorch Model by creating a high-performance CUDA C++ extension, targeting the best possible performance faster than baseline.

========================
⚠️ STRICTLY FORBIDDEN
========================
NO torch operators in C++: NEVER use torch::* or torch::nn::functional::* in binding.cpp or .cu files
NO torch operations in model_new.py: Only tensor creation and your custom ops allowed
NO third-party libraries: Except cuBLAS (GEMM only) and cuDNN (Conv only)

========================
✅ ALLOWED ONLY
========================
C++: Raw CUDA kernels (for custom ops), cuBLAS (for GEMM), cuDNN (MANDATORY for Conv/ConvTranspose)
Python: torch.tensor creation, custom extension ops, tensor properties (.shape, .device)
Memory: torch::empty_like for allocation only
Focus: Implement kernels in kernels/ directory only

========================
WORKSPACE STRUCTURE
========================
Write only the following output sections:
- CUDA_KERNELS: CUDA kernels and helper/device code.
- APPLY_BINDINGS: C++/pybind bindings and launcher wrappers.
- MODEL_NEW: Python ModelNew implementation using custom extension ops.

========================
UNIFIED WORKFLOW
========================

write CUDA kernels with __global__ functions (custom implementations)
```c++
#include <cuda_runtime.h>

// Template kernel for performance tuning
template<int BLOCK_SIZE, int TILE_SIZE>
__global__ void my_kernel_impl(float* output, const float* input, int size) {
    // Shared memory for tiling
    extern __shared__ float smem[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for large data
    for (int i = tid; i < size; i += stride) {
        // Kernel logic with optimizations
        output[i] = /* computation */;
    }
}

// C-interface launcher (no PyTorch dependencies)
extern "C" void my_kernel_launcher(
    float* output,
    const float* input,
    int size,
    int config,
    cudaStream_t stream
) {
    // Default configuration. You may tune block size, tile size, and shared memory.
    (void)config;
    int blocks = (size + 255) / 256;
    int shared_mem_size = 256 * sizeof(float);
    my_kernel_impl<256, 16><<<blocks, 256, shared_mem_size, stream>>>(
        output, input, size);
}
```

write apply_bindings.cpp (PyTorch tensor handling and Python bindings)
```c++
// Use this two headers to replace torch/extension.h for faster compilation
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from .cu file
extern "C" void my_kernel_launcher(
    float* output,
    const float* input,
    int size,
    int config,
    cudaStream_t stream
);

// PyTorch wrapper with config parameter
torch::Tensor my_kernel_forward(torch::Tensor input, int config = 0) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    
    // Get current CUDA stream (correct way)
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call CUDA launcher with config
    my_kernel_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel(),
        config,
        stream
    );
    
    return output;
}

// Registration function
void register_my_kernel(pybind11::module& m) {
    m.def("my_kernel_forward", &my_kernel_forward, 
          "My kernel forward",
          py::arg("input"),
          py::arg("config") = 0);
}

The APPLY_BINDINGS section must include exactly `#include "../binding_registry.h"`

// Auto-register
REGISTER_BINDING(my_kernel, register_my_kernel);
```

create model_new.py
```
import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, ...):  # MUST match Model signature exactly
        super().__init__()
        # Initialize parameters - preserve original structure for state_dict compatibility
        self.weight = nn.Parameter(torch.randn(...))
        self.bias = nn.Parameter(torch.zeros(...))
        
    def forward(self, x):
        # Use custom ops only - NO torch operations
        x = cuda_extension.my_kernel_forward(x, config=0)
        x = cuda_extension.gemm_forward(x, self.weight, self.bias)
        return x
```


========================
📦 OUTPUT FORMAT (STRICT MARKDOWN)
========================

You MUST follow this EXACT format:

### CUDA_KERNELS
```cpp
<CUDA .cu code here>
```

### APPLY_BINDINGS
```cpp
<apply_bindings.cpp code here>
```

### MODEL_NEW
```python
<model_new.py code here>
```

Before output:
- Ensure all 3 sections exist
- Ensure each section has valid code
- Ensure no extra text outside sections



reference pytorch code:
```python
import torch
from torch import digamma, max
from torch.nn import BatchNorm3d, Parameter


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = BatchNorm3d(10)
        self.parameter = Parameter(torch.randn(10))

    def forward(self, x):
        x = self.batch_norm(x)
        x = digamma(x)
        x = max(x)
        x = x + self.parameter
        return x


batch_size = 512
channels = 10
depth = 15
height = 15
width = 15


def get_inputs():
    return [torch.randn(batch_size, channels, depth, height, width)]


def get_init_inputs():
    return []
```

Let's think step by step.
````

## lihongbin CUDA Multi-Turn Template

````yaml
method: system
prompt: null

per_turn_prompts:
  - name: first_turn
    condition: "current_turn == 0"
    history_mode: all
    skip_env: 0.0
    response_truncation: python_code, answer_code
    template: null
  - name: tool_response
    condition: "make_up_tool_response == True"
    history_mode: all
    skip_env: 0.0
    response_truncation: python_code, answer_code
    update_memory: false
    template: |
      Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.

      CRITICAL RULES:
      1. If the feedback reports compilation, runtime, correctness, or other errors, fix those errors first.
      2. If the previous implementation was correct, do not output the same code; try a different CUDA optimization strategy for better performance.
      3. Avoid decoy kernels. In the server feedback, you might see `"decoy_kernel": true`. A decoy kernel means the evaluation system detected that you bypassed writing actual custom CUDA kernels and instead used PyTorch's native operators (for example, `torch.matmul`, `F.softmax`, `torch.exp`) or their C++ equivalents (`aten::*`) as a shortcut.
         To avoid generating decoy kernels:
         - You MUST write genuine custom CUDA C++ kernels (`__global__` functions) for the core computations.
         - DO NOT use native PyTorch operators in `model_new.py` or C++ bindings to cheat the performance test.
         - Only `cuBLAS` (for GEMM) and `cuDNN` (for Conv) are allowed as third-party calls; everything else must be implemented by you.

      Here is the server feedback. Please refer to this feedback to improve the implementation:
      Server feedback (status/metrics/errors):
      {feedback}

      Modify any section as needed.

      Return an improved CUDA implementation with the same output format:
      ### CUDA_KERNELS
      ```cpp
      <CUDA .cu code here>
      ```

      ### APPLY_BINDINGS
      ```cpp
      // Must include exactly: #include "../binding_registry.h"
      <apply_bindings.cpp code here>
      ```
      
      ### MODEL_NEW
      ```python
      <model_new.py code here>
      ```
      Let's think step by step.
````
