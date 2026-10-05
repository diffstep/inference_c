#ifndef TENSOR_GGML_H
#define TENSOR_GGML_H

#include "tensor_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(INFERENCE_SDK_GGML_ENABLED)
typedef struct TensorGGMLQ8Weight TensorGGMLQ8Weight;

/* Quantize a rank-2 F16/F32 weight matrix to ggml Q8_0 and upload it to the
 * selected GPU. The input shape is [output_features, input_features]. */
int tensor_ggml_q8_weight_create(const TensorBuffer *weight,
                                 TensorGGMLQ8Weight **result,
                                 char *error, size_t error_capacity);
void tensor_ggml_q8_weight_free(TensorGGMLQ8Weight *weight);

/* Run input @ weight^T using ggml's selected GPU backend. Input and output
 * may be F16 or F32 with arbitrary leading dimensions. ggml accumulates into
 * F32; output is converted to the requested output dtype. */
int tensor_ggml_linear_q8(const TensorBuffer *input,
                          const TensorGGMLQ8Weight *weight,
                          const TensorBuffer *bias, TensorBuffer *output,
                          char *error, size_t error_capacity);
#endif

#ifdef __cplusplus
}
#endif

#endif
