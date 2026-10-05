#ifndef INFERENCE_SDK_QNN_TENSOR_BACKEND_H
#define INFERENCE_SDK_QNN_TENSOR_BACKEND_H

#include "tensor_compute_backend.h"
#include "attention_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

int inference_sdk_qnn_htp_backend_initialize(const char *htp_directory,
                                             char *message, size_t capacity);
void inference_sdk_qnn_htp_backend_shutdown(void);
const TensorComputeBackend *tensor_compute_qnn_htp_backend(void);
const TensorAttentionBackendOps *tensor_attention_qnn_htp_backend(void);

#ifdef __cplusplus
}
#endif
#endif
