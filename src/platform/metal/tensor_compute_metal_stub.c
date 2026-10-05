#include "tensor_compute_backend.h"

static int unavailable(void) {
    return 0;
}

const TensorComputeBackend *tensor_compute_metal_backend(void) {
    static const TensorComputeBackend backend = {
        .name = "metal", .is_available = unavailable
    };
    return &backend;
}
