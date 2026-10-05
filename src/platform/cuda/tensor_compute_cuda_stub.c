#include "tensor_compute_backend.h"

static int unavailable(void) {
    return 0;
}

const TensorComputeBackend *tensor_compute_cuda_backend(void) {
    static const TensorComputeBackend backend = {
        .name = "cuda", .is_available = unavailable
    };
    return &backend;
}
