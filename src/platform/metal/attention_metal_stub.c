#include "attention_metal.h"

int tensor_attention_metal_decode_f32(const float *query, const float *key,
                                      const float *value, const float *output_weight,
                                      const float *residual, float *output,
                                      size_t query_heads, size_t kv_heads,
                                      size_t kv_length, size_t kv_capacity,
                                      size_t head_dim) {
    (void)query; (void)key; (void)value; (void)output_weight; (void)residual; (void)output;
    (void)query_heads; (void)kv_heads; (void)kv_length;
    (void)kv_capacity; (void)head_dim;
    return 0;
}

void tensor_attention_metal_decode_cache_clear(void) {}

int tensor_attention_metal_decode_sequence_available(void) { return 0; }

int tensor_attention_metal_decode_sequence_f32(
    const TensorAttentionDecoderLayer *layers, size_t layer_count,
    const float *embedding, uint32_t initial_token, size_t token_count,
    uint32_t *generated_tokens,
    const float *const *key_cache, const float *const *value_cache,
    size_t past_length, size_t cache_capacity, size_t hidden_size,
    size_t intermediate_size, size_t query_heads, size_t kv_heads,
    size_t head_dim, float rms_norm_epsilon, double rope_theta,
    const float *final_norm, const float *lm_head, size_t vocabulary_size) {
    (void)layers; (void)layer_count; (void)embedding; (void)initial_token;
    (void)token_count; (void)generated_tokens;
    (void)key_cache; (void)value_cache; (void)past_length; (void)cache_capacity;
    (void)hidden_size; (void)intermediate_size; (void)query_heads; (void)kv_heads;
    (void)head_dim; (void)rms_norm_epsilon; (void)rope_theta;
    (void)final_norm; (void)lm_head; (void)vocabulary_size;
    return 0;
}

int tensor_attention_metal_decode_sequence_f16_available(void) { return 0; }
int tensor_attention_metal_decode_sequence_quant_f16_available(TensorWeightQuantization format) { (void)format; return 0; }
int tensor_attention_metal_decode_sequence_quant_f16(const TensorAttentionDecoderLayerQuantF16 *layers, size_t layer_count, const uint16_t *embedding, uint32_t initial_token, size_t token_count, uint32_t *generated_tokens, const uint16_t *const *key_cache, const uint16_t *const *value_cache, size_t past_length, size_t cache_capacity, size_t hidden_size, size_t intermediate_size, size_t query_heads, size_t kv_heads, size_t head_dim, float rms_norm_epsilon, double rope_theta, const uint16_t *final_norm, const uint8_t *lm_head, size_t vocabulary_size, TensorWeightQuantization format) { (void)layers; (void)layer_count; (void)embedding; (void)initial_token; (void)token_count; (void)generated_tokens; (void)key_cache; (void)value_cache; (void)past_length; (void)cache_capacity; (void)hidden_size; (void)intermediate_size; (void)query_heads; (void)kv_heads; (void)head_dim; (void)rms_norm_epsilon; (void)rope_theta; (void)final_norm; (void)lm_head; (void)vocabulary_size; (void)format; return 0; }
int tensor_attention_metal_decode_sequence_f16(
    const TensorAttentionDecoderLayerF16 *layers, size_t layer_count,
    const uint16_t *embedding, uint32_t initial_token, size_t token_count,
    uint32_t *generated_tokens, const uint16_t *const *key_cache,
    const uint16_t *const *value_cache, size_t past_length, size_t cache_capacity,
    size_t hidden_size, size_t intermediate_size, size_t query_heads,
    size_t kv_heads, size_t head_dim, float rms_norm_epsilon, double rope_theta,
    const uint16_t *final_norm, const uint16_t *lm_head, size_t vocabulary_size) {
    (void)layers; (void)layer_count; (void)embedding; (void)initial_token;
    (void)token_count; (void)generated_tokens; (void)key_cache; (void)value_cache;
    (void)past_length; (void)cache_capacity; (void)hidden_size;
    (void)intermediate_size; (void)query_heads; (void)kv_heads; (void)head_dim;
    (void)rms_norm_epsilon; (void)rope_theta; (void)final_norm; (void)lm_head;
    (void)vocabulary_size; return 0;
}

const TensorAttentionBackendOps *tensor_attention_metal_backend(void) {
    return NULL;
}
