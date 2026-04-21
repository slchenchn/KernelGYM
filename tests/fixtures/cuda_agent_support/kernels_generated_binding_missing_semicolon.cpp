#include "../binding_registry.h"

void register_cuda_extension_ops(pybind11::module& m) {
    m.def("noop_forward", []() { return 0; });
}

REGISTER_BINDING(cuda_extension, register_cuda_extension_ops)
