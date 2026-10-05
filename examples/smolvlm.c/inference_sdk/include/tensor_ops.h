#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor_buffer.h"

#include <stddef.h>

int tensor_matmul_f32(const TensorBuffer *left, const TensorBuffer *right,
                      TensorBuffer *output, char *error,
                      size_t error_capacity);
int tensor_matmul_f16(const TensorBuffer *left, const TensorBuffer *right,
                      TensorBuffer *output, char *error,
                      size_t error_capacity);
int tensor_linear_f32(const TensorBuffer *input, const TensorBuffer *weight,
                      const TensorBuffer *bias, TensorBuffer *output,
                      char *error, size_t error_capacity);
int tensor_linear_f16(const TensorBuffer *input, const TensorBuffer *weight,
                      const TensorBuffer *bias, TensorBuffer *output,
                      char *error, size_t error_capacity);
int tensor_embedding_lookup_f32(const TensorBuffer *table,
                                const TensorBuffer *indices,
                                TensorBuffer *output, char *error,
                                size_t error_capacity);
int tensor_embedding_lookup_f16(const TensorBuffer *table,
                                const TensorBuffer *indices,
                                TensorBuffer *output, char *error,
                                size_t error_capacity);
int tensor_rms_norm_f32(const TensorBuffer *input, const TensorBuffer *weight,
                        float epsilon, TensorBuffer *output, char *error,
                        size_t error_capacity);
int tensor_rms_norm_f16(const TensorBuffer *input, const TensorBuffer *weight,
                        float epsilon, TensorBuffer *output, char *error,
                        size_t error_capacity);
int tensor_layer_norm_f32(const TensorBuffer *input, const TensorBuffer *scale,
                          const TensorBuffer *bias, float epsilon,
                          TensorBuffer *output, char *error,
                          size_t error_capacity);
int tensor_layer_norm_f16(const TensorBuffer *input, const TensorBuffer *scale,
                          const TensorBuffer *bias, float epsilon,
                          TensorBuffer *output, char *error,
                          size_t error_capacity);
int tensor_silu_f32(const TensorBuffer *input, TensorBuffer *output,
                    char *error, size_t error_capacity);
int tensor_silu_f16(const TensorBuffer *input, TensorBuffer *output,
                    char *error, size_t error_capacity);

#endif
