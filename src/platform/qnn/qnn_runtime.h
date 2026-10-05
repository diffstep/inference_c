#ifndef INFERENCE_SDK_QNN_RUNTIME_H
#define INFERENCE_SDK_QNN_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bring up QAIRT HTP and run a known-answer ReLU graph. Returns 1 on success. */
int inference_sdk_qnn_htp_smoke_test(const char *htp_directory,
                                    unsigned iterations,
                                    char *message,
                                    size_t message_capacity);
/* Read layer-0 q_proj.weight from Safetensors and validate an HTP FP16 MatMul. */
int inference_sdk_qnn_htp_safetensors_linear_test(const char *htp_directory,
                                                  const char *model_path,
                                                  unsigned iterations,
                                                  char *message,
                                                  size_t message_capacity);

#ifdef __cplusplus
}
#endif

#endif
