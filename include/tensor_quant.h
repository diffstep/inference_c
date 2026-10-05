#ifndef TENSOR_QUANT_H
#define TENSOR_QUANT_H

#include "tensor_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TENSOR_WEIGHT_Q8_0 = 1,
    TENSOR_WEIGHT_Q4_0 = 2
} TensorWeightQuantization;

/* Quantize an [output, input] FP16 matrix into GGML-compatible Q*_0 blocks.
 * The input width must be a multiple of 32. Returned U8 tensor has shape
 * [output, input/32, block_bytes] and remains host resident. */
int tensor_quantize_linear_weight_f16(const uint16_t *weight,
                                      size_t output_width,
                                      size_t input_width,
                                      TensorWeightQuantization format,
                                      TensorBuffer **result,
                                      char *error, size_t error_capacity);

/* Device-only quantized linear: input/output are FP16 and weight is the U8
 * packed tensor above. No CPU fallback is performed. */
int tensor_linear_q8_0_f16(const TensorBuffer *input,
                           const TensorBuffer *packed_weight,
                           TensorBuffer *output, char *error,
                           size_t error_capacity);
int tensor_linear_q4_0_f16(const TensorBuffer *input,
                           const TensorBuffer *packed_weight,
                           TensorBuffer *output, char *error,
                           size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
