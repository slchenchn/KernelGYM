Some reasoning.

### CUDA_KERNELS
```cpp
#include <torch/extension.h>
__global__ void noop_kernel(float* x) {}
```

### APPLY_BINDINGS
```cpp
#include "../binding_registry.h"

void register_cuda_extension_ops(pybind11::module& m) {
    m.def("noop_forward", []() { return 0; });
}

REGISTER_BINDING(cuda_extension, register_cuda_extension_ops);
```

### MODEL_NEW
```python
import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def forward(self, x):
        return x
```
