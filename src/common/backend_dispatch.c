#include "tensor_compute_backend.h"

#include <stdlib.h>
#include <string.h>

#if defined(INFERENCE_SDK_QNN_HTP_ENABLED)
#include "qnn_tensor_backend.h"
#endif

const TensorComputeBackend *tensor_compute_preferred_backend(void) {
    const char *requested = getenv("TENSOR_BACKEND");
    if (requested != NULL && strcmp(requested, "qnn") == 0) {
#if defined(INFERENCE_SDK_QNN_HTP_ENABLED)
        const TensorComputeBackend *qnn = tensor_compute_qnn_htp_backend();
        return qnn != NULL && qnn->is_available != NULL && qnn->is_available() ? qnn : NULL;
#else
        return NULL;
#endif
    }
    if (requested == NULL || requested[0] == '\0' || strcmp(requested, "auto") == 0) {
        const TensorComputeBackend *cuda = tensor_compute_cuda_backend();
        if (cuda != NULL && cuda->is_available != NULL && cuda->is_available()) return cuda;
        const TensorComputeBackend *metal = tensor_compute_metal_backend();
        return metal != NULL && metal->is_available != NULL && metal->is_available() ? metal : NULL;
    }
    if (strcmp(requested, "cuda") == 0) {
        const TensorComputeBackend *cuda = tensor_compute_cuda_backend();
        return cuda != NULL && cuda->is_available != NULL && cuda->is_available() ? cuda : NULL;
    }
    if (strcmp(requested, "metal") == 0) {
        const TensorComputeBackend *metal = tensor_compute_metal_backend();
        return metal != NULL && metal->is_available != NULL && metal->is_available() ? metal : NULL;
    }
    return NULL;
}
