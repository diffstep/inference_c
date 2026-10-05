#ifndef TENSOR_COMPUTE_BACKEND_H
#define TENSOR_COMPUTE_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#include "tensor_quant.h"

typedef struct {
    const uint16_t *norm1_scale;
    const uint16_t *norm1_bias;
    const uint16_t *q_weight;
    const uint16_t *q_bias;
    const uint16_t *k_weight;
    const uint16_t *k_bias;
    const uint16_t *v_weight;
    const uint16_t *v_bias;
    const uint16_t *out_weight;
    const uint16_t *out_bias;
    const uint16_t *norm2_scale;
    const uint16_t *norm2_bias;
    const uint16_t *fc1_weight;
    const uint16_t *fc1_bias;
    const uint16_t *fc2_weight;
    const uint16_t *fc2_bias;
} TensorVisionLayerF16;

typedef struct {
    const uint16_t *norm1_scale, *norm1_bias;
    const uint8_t *q_weight, *k_weight, *v_weight;
    const uint16_t *q_bias, *k_bias, *v_bias;
    const uint8_t *out_weight;
    const uint16_t *out_bias;
    const uint16_t *norm2_scale, *norm2_bias;
    const uint8_t *fc1_weight;
    const uint16_t *fc1_bias;
    const uint8_t *fc2_weight;
    const uint16_t *fc2_bias;
} TensorVisionLayerQuantF16;

typedef struct {
    const float *norm1_scale, *norm1_bias;
    const float *q_weight, *q_bias, *k_weight, *k_bias, *v_weight, *v_bias;
    const float *out_weight, *out_bias;
    const float *norm2_scale, *norm2_bias;
    const float *fc1_weight, *fc1_bias, *fc2_weight, *fc2_bias;
} TensorVisionLayerF32;

typedef struct {
    const char *name;
    int (*is_available)(void);
    int (*linear_f32)(const float *, const float *, float *, size_t, size_t, size_t);
    int (*linear_f16)(const uint16_t *, const uint16_t *, uint16_t *, size_t, size_t, size_t);
    int (*embedding_f16)(const uint16_t *, const uint32_t *, uint16_t *, size_t, size_t, size_t);
    int (*rms_norm_f16)(const uint16_t *, const uint16_t *, uint16_t *, size_t, size_t, float);
    int (*layer_norm_f16)(const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *, size_t, size_t, float);
    int (*bias_add_f16)(const uint16_t *, const uint16_t *, uint16_t *, size_t, size_t);
    int (*gelu_f16)(const uint16_t *, uint16_t *, size_t);
    /* Exact ModernBERT-style GELU(input) * gate, evaluated on the GPU. */
    int (*gelu_multiply_f16)(const uint16_t *, const uint16_t *, uint16_t *, size_t);
    int (*relu_f16)(const uint16_t *, uint16_t *, size_t);
    int (*rope_f16)(uint16_t *, uint16_t *, size_t, size_t, size_t, size_t, double, size_t);
    int (*silu_multiply_f16)(const uint16_t *, const uint16_t *, uint16_t *, size_t);
    int (*residual_add_f16)(const uint16_t *, const uint16_t *, uint16_t *, size_t);
    int (*argmax_f16)(const uint16_t *, size_t, uint32_t *);
    int (*embedding_f32)(const float *, const uint32_t *, float *, size_t, size_t, size_t);
    int (*rms_norm_f32)(const float *, const float *, float *, size_t, size_t, float);
    int (*rms_norm_linear_f32)(const float *, const float *, float *,
                               const float *, float *, size_t, size_t, size_t,
                               float);
    int (*layer_norm_f32)(const float *, const float *, const float *, float *, size_t, size_t, float);
    int (*bias_add_f32)(const float *, const float *, float *, size_t, size_t);
    int (*gelu_f32)(const float *, float *, size_t);
    int (*rope_f32)(float *, float *, size_t, size_t, size_t, size_t, double);
    int (*silu_multiply_f32)(const float *, const float *, float *, size_t);
    int (*residual_add_f32)(const float *, const float *, float *, size_t);
    int (*argmax_f32)(const float *, size_t, uint32_t *);
    void (*release_cache)(void);
    int (*mlp_f32)(const float *, const float *, const float *, const float *,
                   const float *, float *, size_t, size_t, size_t, size_t);
    int (*mlp_f16)(const uint16_t *, const uint16_t *, const uint16_t *,
                   const uint16_t *, const uint16_t *, uint16_t *,
                   size_t, size_t, size_t, size_t);
    int (*qkv_f16)(const uint16_t *, const uint16_t *, const uint16_t *,
                   const uint16_t *, const uint16_t *, const uint16_t *,
                   const uint16_t *, uint16_t *, uint16_t *,
                   uint16_t *, size_t, size_t, size_t);
    int (*vision_layer_f16)(uint16_t *, const TensorVisionLayerF16 *, size_t,
                            size_t, size_t, float);
    int (*vision_layer_f32)(float *, const TensorVisionLayerF32 *, size_t,
                            size_t, size_t, float);
    void *(*vision_sequence_begin_f16)(const uint16_t *, size_t, size_t);
    int (*vision_sequence_layer_f16)(void *, const TensorVisionLayerF16 *,
                                    size_t, size_t, size_t, float);
    int (*vision_sequence_finish_f16)(void *, uint16_t *);
    void (*vision_sequence_release_f16)(void *);
    int (*vision_sequence_layer_quant_f16)(void *,
        const TensorVisionLayerQuantF16 *, TensorWeightQuantization,
        size_t, size_t, size_t, float);
    void *(*vision_sequence_begin_f32)(const float *, size_t, size_t);
    int (*vision_sequence_layer_f32)(void *, const TensorVisionLayerF32 *,
                                    size_t, size_t, size_t, float);
    int (*vision_sequence_finish_f32)(void *, float *);
    void (*vision_sequence_release_f32)(void *);
    int (*linear_f32_device)(const void *, const void *, void *, size_t,
                             size_t, size_t);
    int (*linear_f16_device)(const void *, const void *, void *, size_t,
                             size_t, size_t);
    int (*linear_q8_0_f16_device)(const void *, const void *, void *, size_t,
                                  size_t, size_t);
    int (*linear_q4_0_f16_device)(const void *, const void *, void *, size_t,
                                  size_t, size_t);
    int (*linear_q8_0_f16)(const uint16_t *, const uint8_t *, uint16_t *,
                           size_t, size_t, size_t);
    int (*linear_q4_0_f16)(const uint16_t *, const uint8_t *, uint16_t *,
                           size_t, size_t, size_t);
    int (*bias_add_f32_device)(const void *, const void *, void *, size_t,
                               size_t);
    int (*bias_add_f16_device)(const void *, const void *, void *, size_t,
                               size_t);
    int (*residual_add_f16_device)(const void *, const void *, void *, size_t);
} TensorComputeBackend;

#ifdef __cplusplus
extern "C" {
#endif

const TensorComputeBackend *tensor_compute_metal_backend(void);
const TensorComputeBackend *tensor_compute_cuda_backend(void);
const TensorComputeBackend *tensor_compute_preferred_backend(void);

#ifdef __cplusplus
}
#endif

#endif
