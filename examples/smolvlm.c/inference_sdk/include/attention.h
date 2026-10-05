#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor_buffer.h"
#include "tensor_workspace.h"

#include <stddef.h>
#include <stdint.h>

typedef enum {
    TENSOR_ATTENTION_BACKEND_CPU,
    TENSOR_ATTENTION_BACKEND_METAL,
    TENSOR_ATTENTION_BACKEND_CUDA
} TensorAttentionBackend;

typedef struct {
    const float *input_norm;
    const float *q_weight;
    const float *k_weight;
    const float *v_weight;
    const float *o_weight;
    const float *post_norm;
    const float *gate_weight;
    const float *up_weight;
    const float *down_weight;
} TensorAttentionDecoderLayer;

typedef struct {
    const uint16_t *input_norm, *q_weight, *k_weight, *v_weight, *o_weight;
    const uint16_t *post_norm, *gate_weight, *up_weight, *down_weight;
} TensorAttentionDecoderLayerF16;

int tensor_sdpa_f32(const TensorBuffer *query, const TensorBuffer *key,
                    const TensorBuffer *value,
                    const TensorBuffer *additive_mask, float scale,
                    int causal, TensorBuffer *output,
                    TensorWorkspace *workspace,
                    TensorAttentionBackend *backend_used, char *error,
                    size_t error_capacity);
int tensor_sdpa_f16(const TensorBuffer *query, const TensorBuffer *key,
                    const TensorBuffer *value,
                    const TensorBuffer *additive_mask, float scale,
                    int causal, TensorBuffer *output,
                    TensorWorkspace *workspace,
                    TensorAttentionBackend *backend_used, char *error,
                    size_t error_capacity);

int tensor_attention_decode_f32(const float *query, const float *key,
                                const float *value, const float *output_weight,
                                const float *residual, float *output,
                                size_t query_heads, size_t kv_heads,
                                size_t kv_length, size_t kv_capacity,
                                size_t head_dim);
int tensor_attention_decode_sequence_available(void);
int tensor_attention_decode_sequence_f32(
    const TensorAttentionDecoderLayer *layers, size_t layer_count,
    const float *embedding, uint32_t initial_token,
    size_t token_count, uint32_t *generated_tokens,
    const float *const *key_cache, const float *const *value_cache,
    size_t past_length, size_t cache_capacity, size_t hidden_size,
    size_t intermediate_size, size_t query_heads, size_t kv_heads,
    size_t head_dim, float rms_norm_epsilon, double rope_theta,
    const float *final_norm, const float *lm_head, size_t vocabulary_size);
int tensor_attention_decode_sequence_f16_available(void);
int tensor_attention_decode_sequence_f16(
    const TensorAttentionDecoderLayerF16 *layers, size_t layer_count,
    const uint16_t *embedding, uint32_t initial_token, size_t token_count,
    uint32_t *generated_tokens, const uint16_t *const *key_cache,
    const uint16_t *const *value_cache, size_t past_length,
    size_t cache_capacity, size_t hidden_size, size_t intermediate_size,
    size_t query_heads, size_t kv_heads, size_t head_dim,
    float rms_norm_epsilon, double rope_theta, const uint16_t *final_norm,
    const uint16_t *lm_head, size_t vocabulary_size);
void tensor_attention_decode_cache_clear(void);

#endif
