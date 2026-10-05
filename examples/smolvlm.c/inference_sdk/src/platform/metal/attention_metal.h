#ifndef ATTENTION_METAL_H
#define ATTENTION_METAL_H

#include <stddef.h>
#include "attention_backend.h"

int tensor_attention_metal_encode_tiled_f16(void *command_buffer,
                                             void *query_buffer,
                                             void *key_buffer,
                                             void *value_buffer,
                                             void *output_buffer,
                                             size_t heads, size_t sequence,
                                             float scale);
int tensor_attention_metal_encode_tiled_f32(void *command_buffer,
                                             void *query_buffer,
                                             void *key_buffer,
                                             void *value_buffer,
                                             void *output_buffer,
                                             size_t heads, size_t sequence,
                                             float scale);

int tensor_attention_metal_decode_f32(const float *query, const float *key,
                                      const float *value, const float *output_weight,
                                      const float *residual, float *output,
                                      size_t query_heads, size_t kv_heads,
                                      size_t kv_length, size_t kv_capacity,
                                      size_t head_dim);
void tensor_attention_metal_decode_cache_clear(void);
int tensor_attention_metal_decode_sequence_available(void);
int tensor_attention_metal_decode_sequence_f32(
    const TensorAttentionDecoderLayer *layers, size_t layer_count,
    const float *embedding, uint32_t initial_token, size_t token_count,
    uint32_t *generated_tokens,
    const float *const *key_cache, const float *const *value_cache,
    size_t past_length, size_t cache_capacity, size_t hidden_size,
    size_t intermediate_size, size_t query_heads, size_t kv_heads,
    size_t head_dim, float rms_norm_epsilon, double rope_theta,
    const float *final_norm, const float *lm_head, size_t vocabulary_size);
int tensor_attention_metal_decode_sequence_f16_available(void);
int tensor_attention_metal_decode_sequence_f16(
    const TensorAttentionDecoderLayerF16 *layers, size_t layer_count,
    const uint16_t *embedding, uint32_t initial_token, size_t token_count,
    uint32_t *generated_tokens, const uint16_t *const *key_cache,
    const uint16_t *const *value_cache, size_t past_length, size_t cache_capacity,
    size_t hidden_size, size_t intermediate_size, size_t query_heads,
    size_t kv_heads, size_t head_dim, float rms_norm_epsilon, double rope_theta,
    const uint16_t *final_norm, const uint16_t *lm_head, size_t vocabulary_size);

#endif
