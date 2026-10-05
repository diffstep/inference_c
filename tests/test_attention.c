#include "attention.h"
#include "tensor_f16.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static TensorBuffer *make_f32(const uint64_t *shape, size_t rank,
                              const float *values, size_t count) {
    TensorBuffer *buffer = NULL;
    char error[256] = {0};
    assert(tensor_buffer_create(TENSOR_DTYPE_F32, shape, rank, &buffer,
                                error, sizeof(error)));
    assert(tensor_buffer_element_count(buffer) == count);
    if (values != NULL)
        memcpy(tensor_buffer_data_mut(buffer), values, count * sizeof(float));
    return buffer;
}

static void assert_near(float actual, float expected) {
    assert(fabsf(actual - expected) < 1e-5f);
}

static void test_decode_cache_residency(void) {
    if (!tensor_attention_decode_sequence_f16_available()) {
        puts("decode cache residency test skipped: accelerated F16 decode unavailable");
        return;
    }

    enum { HIDDEN = 4, INTERMEDIATE = 8, VOCABULARY = 8, CAPACITY = 8,
           KV_WIDTH = 4, DECODE_TOKENS = 4 };
    uint16_t input_norm[HIDDEN], post_norm[HIDDEN];
    uint16_t q_weight[HIDDEN * HIDDEN], k_weight[HIDDEN * HIDDEN];
    uint16_t v_weight[HIDDEN * HIDDEN], o_weight[HIDDEN * HIDDEN];
    uint16_t gate_weight[INTERMEDIATE * HIDDEN];
    uint16_t up_weight[INTERMEDIATE * HIDDEN];
    uint16_t down_weight[HIDDEN * INTERMEDIATE];
    uint16_t embedding[VOCABULARY * HIDDEN], lm_head[VOCABULARY * HIDDEN];
    for (size_t i = 0; i < HIDDEN; ++i) {
        input_norm[i] = tensor_f32_to_f16(1.0f);
        post_norm[i] = tensor_f32_to_f16(1.0f);
    }
    for (size_t i = 0; i < HIDDEN * HIDDEN; ++i) {
        q_weight[i] = tensor_f32_to_f16((float)((int)(i % 7) - 3) * 0.025f);
        k_weight[i] = tensor_f32_to_f16((float)((int)(i % 5) - 2) * 0.03f);
        v_weight[i] = tensor_f32_to_f16((float)((int)(i % 9) - 4) * 0.02f);
        o_weight[i] = tensor_f32_to_f16((float)((int)(i % 11) - 5) * 0.015f);
    }
    for (size_t i = 0; i < INTERMEDIATE * HIDDEN; ++i) {
        gate_weight[i] = tensor_f32_to_f16((float)((int)(i % 13) - 6) * 0.01f);
        up_weight[i] = tensor_f32_to_f16((float)((int)(i % 7) - 3) * 0.02f);
    }
    for (size_t i = 0; i < HIDDEN * INTERMEDIATE; ++i)
        down_weight[i] = tensor_f32_to_f16((float)((int)(i % 9) - 4) * 0.01f);
    for (size_t i = 0; i < VOCABULARY * HIDDEN; ++i) {
        embedding[i] = tensor_f32_to_f16((float)((int)(i % 17) - 8) * 0.02f);
        lm_head[i] = tensor_f32_to_f16((float)((int)(i % 13) - 6) * 0.025f);
    }
    const TensorAttentionDecoderLayerF16 layer = {
        input_norm, q_weight, k_weight, v_weight, o_weight, post_norm,
        gate_weight, up_weight, down_weight
    };

    uint16_t full_key[CAPACITY * KV_WIDTH], full_value[CAPACITY * KV_WIDTH];
    uint16_t split_key[CAPACITY * KV_WIDTH], split_value[CAPACITY * KV_WIDTH];
    memset(full_key, 0, sizeof(full_key));
    memset(full_value, 0, sizeof(full_value));
    memset(split_key, 0, sizeof(split_key));
    memset(split_value, 0, sizeof(split_value));
    uint16_t full_key_before[CAPACITY * KV_WIDTH];
    uint16_t full_value_before[CAPACITY * KV_WIDTH];
    uint16_t split_key_before[CAPACITY * KV_WIDTH];
    uint16_t split_value_before[CAPACITY * KV_WIDTH];
    memcpy(full_key_before, full_key, sizeof(full_key));
    memcpy(full_value_before, full_value, sizeof(full_value));
    memcpy(split_key_before, split_key, sizeof(split_key));
    memcpy(split_value_before, split_value, sizeof(split_value));
    const uint16_t *full_keys[] = {full_key};
    const uint16_t *full_values[] = {full_value};
    const uint16_t *split_keys[] = {split_key};
    const uint16_t *split_values[] = {split_value};
    uint32_t full_tokens[DECODE_TOKENS] = {0};
    uint32_t split_tokens[DECODE_TOKENS] = {0};

    tensor_attention_decode_cache_clear();
    assert(tensor_attention_decode_sequence_f16(&layer, 1, embedding, 1,
        DECODE_TOKENS, full_tokens, full_keys, full_values, 0, CAPACITY,
        HIDDEN, INTERMEDIATE, 2, 2, 2, 1e-5f, 10000.0,
        post_norm, lm_head, VOCABULARY));
    assert(memcmp(full_key, full_key_before, sizeof(full_key)) == 0);
    assert(memcmp(full_value, full_value_before, sizeof(full_value)) == 0);

    tensor_attention_decode_cache_clear();
    assert(tensor_attention_decode_sequence_f16(&layer, 1, embedding, 1, 2,
        split_tokens, split_keys, split_values, 0, CAPACITY, HIDDEN,
        INTERMEDIATE, 2, 2, 2, 1e-5f, 10000.0, post_norm, lm_head,
        VOCABULARY));
    assert(tensor_attention_decode_sequence_f16(&layer, 1, embedding,
        split_tokens[1], 2, split_tokens + 2, split_keys, split_values, 2,
        CAPACITY, HIDDEN, INTERMEDIATE, 2, 2, 2, 1e-5f, 10000.0,
        post_norm, lm_head, VOCABULARY));
    assert(memcmp(full_tokens, split_tokens, sizeof(full_tokens)) == 0);
    assert(memcmp(split_key, split_key_before, sizeof(split_key)) == 0);
    assert(memcmp(split_value, split_value_before, sizeof(split_value)) == 0);

    TensorPagedKVCache *paged_cache = NULL;
    TensorPagedKVSequence paged_sequence = 0;
    assert(tensor_paged_kv_cache_create(2, CAPACITY / 2, 1, &paged_cache));
    assert(tensor_paged_kv_sequence_create(paged_cache, &paged_sequence));
    TensorPagedKVDecodeRequest paged_request = {0};
    paged_request.cache = paged_cache;
    paged_request.sequence = paged_sequence;
    paged_request.storage = TENSOR_PAGED_KV_F16;
    paged_request.layers = &layer;
    paged_request.layer_count = 1;
    paged_request.embedding = embedding;
    paged_request.initial_token = 1;
    paged_request.token_count = DECODE_TOKENS;
    uint32_t paged_tokens[DECODE_TOKENS] = {0};
    paged_request.generated_tokens = paged_tokens;
    paged_request.key_cache = (const void *const *)full_keys;
    paged_request.value_cache = (const void *const *)full_values;
    paged_request.hidden_size = HIDDEN;
    paged_request.intermediate_size = INTERMEDIATE;
    paged_request.query_heads = 2;
    paged_request.kv_heads = 2;
    paged_request.head_dim = 2;
    paged_request.rms_norm_epsilon = 1e-5f;
    paged_request.rope_theta = 10000.0;
    paged_request.final_norm = post_norm;
    paged_request.lm_head = lm_head;
    paged_request.vocabulary_size = VOCABULARY;
    assert(tensor_attention_decode_sequence_paged(&paged_request));
    assert(memcmp(full_tokens, paged_tokens, sizeof(full_tokens)) == 0);
    assert(tensor_paged_kv_cache_free_blocks(paged_cache) == 2);
    assert(tensor_paged_kv_sequence_release(paged_cache, paged_sequence));
    assert(tensor_paged_kv_cache_free_blocks(paged_cache) == CAPACITY / 2);
    tensor_paged_kv_cache_free(paged_cache);

    uint16_t prefix_key[CAPACITY * KV_WIDTH] = {0};
    uint16_t prefix_value[CAPACITY * KV_WIDTH] = {0};
    for (size_t i = 0; i < 3 * KV_WIDTH; ++i) {
        prefix_key[i] = tensor_f32_to_f16((float)((int)i - 5) * 0.01f);
        prefix_value[i] = tensor_f32_to_f16((float)((int)i - 3) * 0.02f);
    }
    const uint16_t *prefix_keys[] = {prefix_key};
    const uint16_t *prefix_values[] = {prefix_value};
    uint32_t contiguous_prefix_tokens[2] = {0};
    tensor_attention_decode_cache_clear();
    assert(tensor_attention_decode_sequence_f16(&layer, 1, embedding, 1, 2,
        contiguous_prefix_tokens, prefix_keys, prefix_values, 3, CAPACITY,
        HIDDEN, INTERMEDIATE, 2, 2, 2, 1e-5f, 10000.0,
        post_norm, lm_head, VOCABULARY));
    assert(memcmp(prefix_key + 3 * KV_WIDTH, (uint16_t[KV_WIDTH]){0},
                  KV_WIDTH * sizeof(uint16_t)) == 0);
    assert(memcmp(prefix_value + 3 * KV_WIDTH, (uint16_t[KV_WIDTH]){0},
                  KV_WIDTH * sizeof(uint16_t)) == 0);
    TensorPagedKVCache *prefix_paged_cache = NULL;
    TensorPagedKVSequence prefix_sequence = 0;
    assert(tensor_paged_kv_cache_create(2, CAPACITY / 2, 1,
                                        &prefix_paged_cache));
    assert(tensor_paged_kv_sequence_create(prefix_paged_cache, &prefix_sequence));
    TensorPagedKVDecodeRequest prefix_request = paged_request;
    prefix_request.cache = prefix_paged_cache;
    prefix_request.sequence = prefix_sequence;
    prefix_request.initial_token = 1;
    prefix_request.token_count = 2;
    uint32_t paged_prefix_tokens[2] = {0};
    prefix_request.generated_tokens = paged_prefix_tokens;
    prefix_request.key_cache = (const void *const *)prefix_keys;
    prefix_request.value_cache = (const void *const *)prefix_values;
    prefix_request.past_length = 3;
    prefix_request.head_dim = 4;
    assert(!tensor_attention_decode_sequence_paged(&prefix_request));
    assert(tensor_paged_kv_cache_free_blocks(prefix_paged_cache) == CAPACITY / 2);
    prefix_request.head_dim = 2;
    assert(tensor_attention_decode_sequence_paged(&prefix_request));
    assert(memcmp(contiguous_prefix_tokens, paged_prefix_tokens,
                  sizeof(contiguous_prefix_tokens)) == 0);
    assert(tensor_paged_kv_cache_free_blocks(prefix_paged_cache) == 1);
    assert(tensor_paged_kv_sequence_release(prefix_paged_cache, prefix_sequence));
    tensor_paged_kv_cache_free(prefix_paged_cache);

    enum { LONG_PAST = 1025, LONG_CAPACITY = 1040, LONG_BLOCK = 16 };
    uint16_t long_key[LONG_CAPACITY * KV_WIDTH] = {0};
    uint16_t long_value[LONG_CAPACITY * KV_WIDTH] = {0};
    for (size_t i = 0; i < LONG_PAST * KV_WIDTH; ++i) {
        long_key[i] = tensor_f32_to_f16((float)((int)(i % 19) - 9) * 0.002f);
        long_value[i] = tensor_f32_to_f16((float)((int)(i % 13) - 6) * 0.003f);
    }
    const uint16_t *long_keys[] = {long_key};
    const uint16_t *long_values[] = {long_value};
    uint32_t long_contiguous = 0, long_paged = 0;
    tensor_attention_decode_cache_clear();
    assert(tensor_attention_decode_sequence_f16(&layer, 1, embedding, 1, 1,
        &long_contiguous, long_keys, long_values, LONG_PAST, LONG_CAPACITY,
        HIDDEN, INTERMEDIATE, 2, 2, 2, 1e-5f, 10000.0,
        post_norm, lm_head, VOCABULARY));
    TensorPagedKVCache *long_cache = NULL;
    TensorPagedKVSequence long_sequence = 0;
    assert(tensor_paged_kv_cache_create(LONG_BLOCK,
        LONG_CAPACITY / LONG_BLOCK, 1, &long_cache));
    assert(tensor_paged_kv_sequence_create(long_cache, &long_sequence));
    TensorPagedKVDecodeRequest long_request = paged_request;
    long_request.cache = long_cache;
    long_request.sequence = long_sequence;
    long_request.token_count = 1;
    long_request.generated_tokens = &long_paged;
    long_request.key_cache = (const void *const *)long_keys;
    long_request.value_cache = (const void *const *)long_values;
    long_request.past_length = LONG_PAST;
    assert(tensor_attention_decode_sequence_paged(&long_request));
    assert(long_contiguous == long_paged);
    assert(tensor_paged_kv_sequence_release(long_cache, long_sequence));
    tensor_paged_kv_cache_free(long_cache);

    TensorPagedKVCache *interleaved_cache = NULL;
    TensorPagedKVSequence sequence_a = 0, sequence_b = 0;
    assert(tensor_paged_kv_cache_create(2, 4, 2, &interleaved_cache));
    assert(tensor_paged_kv_sequence_create(interleaved_cache, &sequence_a));
    assert(tensor_paged_kv_sequence_create(interleaved_cache, &sequence_b));
    TensorPagedKVDecodeRequest request_a = paged_request;
    TensorPagedKVDecodeRequest request_b = paged_request;
    uint32_t expected_a[2] = {0}, expected_b[2] = {0};
    uint32_t actual_a[2] = {0}, actual_b[2] = {0};
    assert(tensor_attention_decode_sequence_f16(&layer, 1, embedding, 1, 2,
        expected_a, full_keys, full_values, 0, CAPACITY, HIDDEN, INTERMEDIATE,
        2, 2, 2, 1e-5f, 10000.0, post_norm, lm_head, VOCABULARY));
    tensor_attention_decode_cache_clear();
    assert(tensor_attention_decode_sequence_f16(&layer, 1, embedding, 2, 2,
        expected_b, full_keys, full_values, 0, CAPACITY, HIDDEN, INTERMEDIATE,
        2, 2, 2, 1e-5f, 10000.0, post_norm, lm_head, VOCABULARY));
    request_a.cache = request_b.cache = interleaved_cache;
    request_a.sequence = sequence_a;
    request_b.sequence = sequence_b;
    request_b.initial_token = 2;
    request_a.token_count = request_b.token_count = 1;
    request_a.generated_tokens = actual_a;
    request_b.generated_tokens = actual_b;
    assert(tensor_attention_decode_sequence_paged(&request_a));
    assert(tensor_attention_decode_sequence_paged(&request_b));
    request_a.initial_token = actual_a[0];
    request_a.past_length = 1;
    request_a.generated_tokens = actual_a + 1;
    request_b.initial_token = actual_b[0];
    request_b.past_length = 1;
    request_b.generated_tokens = actual_b + 1;
    assert(tensor_attention_decode_sequence_paged(&request_a));
    assert(tensor_attention_decode_sequence_paged(&request_b));
    assert(memcmp(expected_a, actual_a, sizeof(actual_a)) == 0);
    assert(memcmp(expected_b, actual_b, sizeof(actual_b)) == 0);
    assert(tensor_paged_kv_sequence_release(interleaved_cache, sequence_a));
    assert(tensor_paged_kv_sequence_release(interleaved_cache, sequence_b));
    assert(tensor_paged_kv_cache_free_blocks(interleaved_cache) == 4);
    tensor_paged_kv_cache_free(interleaved_cache);

    tensor_attention_decode_cache_clear();
    puts("GPU consistency: paged/contiguous F16 decode, interleaving, and rollback matched");
}

static void test_paged_decode_f32(void) {
    if (!tensor_attention_decode_sequence_available()) {
        puts("paged F32 decode skipped: accelerated F32 decode unavailable");
        return;
    }
    enum { H = 8, I = 8, V = 8, CAP = 8, KW = 8, TOKENS = 2 };
    float input_norm[H], post_norm[H], q[H * H], k[H * H], v[H * H];
    float o[H * H], gate[I * H], up[I * H], down[H * I];
    float embedding[V * H], lm_head[V * H];
    for (size_t i = 0; i < H; ++i) input_norm[i] = post_norm[i] = 1.0f;
    for (size_t i = 0; i < H * H; ++i) {
        q[i] = (float)((int)(i % 7) - 3) * 0.02f;
        k[i] = (float)((int)(i % 5) - 2) * 0.03f;
        v[i] = (float)((int)(i % 9) - 4) * 0.02f;
        o[i] = (float)((int)(i % 11) - 5) * 0.015f;
    }
    for (size_t i = 0; i < I * H; ++i) {
        gate[i] = (float)((int)(i % 13) - 6) * 0.01f;
        up[i] = (float)((int)(i % 7) - 3) * 0.02f;
    }
    for (size_t i = 0; i < H * I; ++i) down[i] = (float)((int)(i % 9) - 4) * 0.01f;
    for (size_t i = 0; i < V * H; ++i) {
        embedding[i] = (float)((int)(i % 17) - 8) * 0.02f;
        lm_head[i] = (float)((int)(i % 13) - 6) * 0.025f;
    }
    TensorAttentionDecoderLayer layer = {
        input_norm, q, k, v, o, post_norm, gate, up, down
    };
    float key[CAP * KW] = {0}, value[CAP * KW] = {0};
    const float *keys[] = {key}, *values[] = {value};
    uint32_t contiguous_prefix[3] = {0}, contiguous[TOKENS] = {0};
    tensor_attention_decode_cache_clear();
    assert(tensor_attention_decode_sequence_f32(&layer, 1, embedding, 1,
        3, contiguous_prefix, keys, values, 0, CAP, H, I, 2, 2, 4,
        1e-5f, 10000.0, post_norm, lm_head, V));
    assert(tensor_attention_decode_sequence_f32(&layer, 1, embedding,
        contiguous_prefix[2], TOKENS, contiguous, keys, values, 3, CAP, H, I,
        2, 2, 4, 1e-5f, 10000.0, post_norm, lm_head, V));
    TensorPagedKVCache *cache = NULL;
    TensorPagedKVSequence sequence = 0;
    assert(tensor_paged_kv_cache_create(2, CAP / 2, 1, &cache));
    assert(tensor_paged_kv_sequence_create(cache, &sequence));
    TensorPagedKVDecodeRequest request = {0};
    request.cache = cache;
    request.sequence = sequence;
    request.storage = TENSOR_PAGED_KV_F32;
    request.layers = &layer;
    request.layer_count = 1;
    request.embedding = embedding;
    request.initial_token = 1;
    request.token_count = 3;
    uint32_t paged_prefix[3] = {0};
    uint32_t paged[TOKENS] = {0};
    request.generated_tokens = paged_prefix;
    request.key_cache = (const void *const *)keys;
    request.value_cache = (const void *const *)values;
    request.past_length = 0;
    request.hidden_size = H;
    request.intermediate_size = I;
    request.query_heads = 2;
    request.kv_heads = 2;
    request.head_dim = 4;
    request.rms_norm_epsilon = 1e-5f;
    request.rope_theta = 10000.0;
    request.final_norm = post_norm;
    request.lm_head = lm_head;
    request.vocabulary_size = V;
    assert(tensor_attention_decode_sequence_paged(&request));
    request.initial_token = paged_prefix[2];
    request.token_count = TOKENS;
    request.generated_tokens = paged;
    request.past_length = 3;
    assert(tensor_attention_decode_sequence_paged(&request));
    assert(memcmp(contiguous, paged, sizeof(contiguous)) == 0);
    assert(tensor_paged_kv_cache_free_blocks(cache) == 1);
    assert(tensor_paged_kv_sequence_release(cache, sequence));
    tensor_paged_kv_cache_free(cache);
    tensor_attention_decode_cache_clear();
}

static void test_quantized_decode_sequence(TensorWeightQuantization format) {
    if (!tensor_attention_decode_sequence_quant_f16_available(format)) {
        printf("quantized decode test skipped: format %d unavailable\n", (int)format);
        return;
    }
    enum { H = 32, I = 32, V = 32, CAPACITY = 4, TOKENS = 2 };
    uint16_t input_norm[H], post_norm[H], q[H * H], k[H * H], v[H * H];
    uint16_t o[H * H], gate[I * H], up[I * H], down[H * I];
    uint16_t embedding[V * H], final_norm[H];
    for (size_t i = 0; i < H; ++i) {
        input_norm[i] = tensor_f32_to_f16(1.0f);
        post_norm[i] = tensor_f32_to_f16(1.0f);
        final_norm[i] = tensor_f32_to_f16(1.0f);
    }
    for (size_t i = 0; i < H * H; ++i) {
        q[i] = tensor_f32_to_f16((float)((int)(i % 9) - 4) * 0.012f);
        k[i] = tensor_f32_to_f16((float)((int)(i % 7) - 3) * 0.015f);
        v[i] = tensor_f32_to_f16((float)((int)(i % 11) - 5) * 0.01f);
        o[i] = tensor_f32_to_f16((float)((int)(i % 13) - 6) * 0.008f);
        gate[i] = tensor_f32_to_f16((float)((int)(i % 15) - 7) * 0.009f);
        up[i] = tensor_f32_to_f16((float)((int)(i % 9) - 4) * 0.011f);
        down[i] = tensor_f32_to_f16((float)((int)(i % 11) - 5) * 0.007f);
        embedding[i] = tensor_f32_to_f16((float)((int)(i % 17) - 8) * 0.018f);
    }
    TensorBuffer *packed[7] = {0};
    char error[256] = {0};
    assert(tensor_quantize_linear_weight_f16(q, H, H, format, &packed[0], error, sizeof(error)));
    assert(tensor_quantize_linear_weight_f16(k, H, H, format, &packed[1], error, sizeof(error)));
    assert(tensor_quantize_linear_weight_f16(v, H, H, format, &packed[2], error, sizeof(error)));
    assert(tensor_quantize_linear_weight_f16(o, H, H, format, &packed[3], error, sizeof(error)));
    assert(tensor_quantize_linear_weight_f16(gate, I, H, format, &packed[4], error, sizeof(error)));
    assert(tensor_quantize_linear_weight_f16(up, I, H, format, &packed[5], error, sizeof(error)));
    assert(tensor_quantize_linear_weight_f16(down, H, I, format, &packed[6], error, sizeof(error)));
    const TensorAttentionDecoderLayerQuantF16 layer = {
        input_norm, post_norm,
        tensor_buffer_data(packed[0]), tensor_buffer_data(packed[1]),
        tensor_buffer_data(packed[2]), tensor_buffer_data(packed[3]),
        tensor_buffer_data(packed[4]), tensor_buffer_data(packed[5]),
        tensor_buffer_data(packed[6])
    };
    uint16_t full_key[CAPACITY * H] = {0}, full_value[CAPACITY * H] = {0};
    uint16_t split_key[CAPACITY * H] = {0}, split_value[CAPACITY * H] = {0};
    const uint16_t *full_keys[] = {full_key}, *full_values[] = {full_value};
    const uint16_t *split_keys[] = {split_key}, *split_values[] = {split_value};
    uint32_t full_tokens[TOKENS] = {0}, split_tokens[TOKENS] = {0};
    tensor_attention_decode_cache_clear();
    assert(tensor_attention_decode_sequence_quant_f16(&layer, 1, embedding, 1,
        TOKENS, full_tokens, full_keys, full_values, 0, CAPACITY, H, I, 2, 2,
        16, 1e-5f, 10000.0, final_norm,
        tensor_buffer_data(packed[0]), V, format));
    tensor_attention_decode_cache_clear();
    assert(tensor_attention_decode_sequence_quant_f16(&layer, 1, embedding, 1,
        1, split_tokens, split_keys, split_values, 0, CAPACITY, H, I, 2, 2,
        16, 1e-5f, 10000.0, final_norm,
        tensor_buffer_data(packed[0]), V, format));
    assert(tensor_attention_decode_sequence_quant_f16(&layer, 1, embedding,
        split_tokens[0], 1, split_tokens + 1, split_keys, split_values, 1,
        CAPACITY, H, I, 2, 2, 16, 1e-5f, 10000.0, final_norm,
        tensor_buffer_data(packed[0]), V, format));
    assert(memcmp(full_tokens, split_tokens, sizeof(full_tokens)) == 0);

    TensorPagedKVCache *paged_cache = NULL;
    TensorPagedKVSequence paged_sequence = 0;
    assert(tensor_paged_kv_cache_create(2, CAPACITY / 2, 1, &paged_cache));
    assert(tensor_paged_kv_sequence_create(paged_cache, &paged_sequence));
    TensorPagedKVDecodeRequest request = {0};
    request.cache = paged_cache;
    request.sequence = paged_sequence;
    request.storage = TENSOR_PAGED_KV_QUANT_F16;
    request.layers = &layer;
    request.layer_count = 1;
    request.embedding = embedding;
    request.initial_token = 1;
    request.token_count = TOKENS;
    uint32_t paged_tokens[TOKENS] = {0};
    request.generated_tokens = paged_tokens;
    request.key_cache = (const void *const *)full_keys;
    request.value_cache = (const void *const *)full_values;
    request.hidden_size = H;
    request.intermediate_size = I;
    request.query_heads = 2;
    request.kv_heads = 2;
    request.head_dim = 16;
    request.rms_norm_epsilon = 1e-5f;
    request.rope_theta = 10000.0;
    request.final_norm = final_norm;
    request.lm_head = tensor_buffer_data(packed[0]);
    request.vocabulary_size = V;
    request.weight_quantization = format;
    assert(tensor_attention_decode_sequence_paged(&request));
    assert(memcmp(full_tokens, paged_tokens, sizeof(full_tokens)) == 0);
    assert(tensor_paged_kv_sequence_release(paged_cache, paged_sequence));
    tensor_paged_kv_cache_free(paged_cache);
    tensor_attention_decode_cache_clear();
    for (size_t i = 0; i < 7; ++i) tensor_buffer_free(packed[i]);
    printf("GPU consistency: quantized decode chunk parity passed (format=%d)\n", (int)format);
}

int main(void) {
    const uint64_t query_shape[] = {1, 2, 2, 2};
    const uint64_t kv_shape[] = {1, 1, 2, 2};
    const uint64_t output_shape[] = {1, 2, 2, 2};
    const uint64_t mask_shape[] = {2, 2};
    const float query_values[] = {1, 0, 0, 1, 1, 0, 0, 1};
    const float key_values[] = {1, 0, 0, 1};
    const float value_values[] = {1, 0, 0, 2};
    const float mask_values[] = {-INFINITY, 0, 0, 0};
    TensorBuffer *query = make_f32(query_shape, 4, query_values, 8);
    TensorBuffer *key = make_f32(kv_shape, 4, key_values, 4);
    TensorBuffer *value = make_f32(kv_shape, 4, value_values, 4);
    TensorBuffer *output = make_f32(output_shape, 4, NULL, 8);
    TensorBuffer *mask = make_f32(mask_shape, 2, mask_values, 4);
    TensorWorkspace *workspace = NULL;
    TensorAttentionBackend backend = TENSOR_ATTENTION_BACKEND_CPU;
    char error[256] = {0};
    assert(tensor_workspace_create(64, &workspace, error, sizeof(error)));

    assert(tensor_sdpa_f32(query, key, value, NULL, 1.0f, 0, output,
                           workspace, &backend, error, sizeof(error)));
    const char *backend_name = backend == TENSOR_ATTENTION_BACKEND_CUDA ? "cuda" :
        backend == TENSOR_ATTENTION_BACKEND_METAL ? "metal" : "cpu";
    printf("attention backend=%s\n", backend_name);
    const float *output_values = tensor_buffer_data(output);
    assert_near(output_values[0], 0.73105858f);
    assert_near(output_values[1], 0.53788284f);
    assert_near(output_values[2], 0.26894143f);
    assert_near(output_values[3], 1.4621172f);
    assert_near(output_values[4], output_values[0]);
    assert_near(output_values[5], output_values[1]);
    assert_near(output_values[6], output_values[2]);
    assert_near(output_values[7], output_values[3]);
    assert(tensor_workspace_used_bytes(workspace) == 0);

    assert(tensor_sdpa_f32(query, key, value, NULL, 1.0f, 1, output,
                           workspace, &backend, error, sizeof(error)));
    output_values = tensor_buffer_data(output);
    assert_near(output_values[0], 1.0f);
    assert_near(output_values[1], 0.0f);
    assert_near(output_values[2], 0.26894143f);
    assert_near(output_values[3], 1.4621172f);

    assert(tensor_sdpa_f32(query, key, value, mask, 1.0f, 0, output,
                           workspace, &backend, error, sizeof(error)));
    output_values = tensor_buffer_data(output);
    assert_near(output_values[0], 0.0f);
    assert_near(output_values[1], 2.0f);

    const float all_masked_values[] = {-INFINITY, -INFINITY,
                                       -INFINITY, -INFINITY};
    TensorBuffer *all_masked = make_f32(mask_shape, 2, all_masked_values, 4);
    assert(tensor_sdpa_f32(query, key, value, all_masked, 1.0f, 0, output,
                           workspace, &backend, error, sizeof(error)));
    output_values = tensor_buffer_data(output);
    for (size_t index = 0; index < 8; index++) assert(output_values[index] == 0.0f);

    test_decode_cache_residency();
    test_paged_decode_f32();
    test_quantized_decode_sequence(TENSOR_WEIGHT_Q8_0);
    test_quantized_decode_sequence(TENSOR_WEIGHT_Q4_0);

    tensor_buffer_free(all_masked);
    tensor_workspace_free(workspace);
    tensor_buffer_free(mask);
    tensor_buffer_free(output);
    tensor_buffer_free(value);
    tensor_buffer_free(key);
    tensor_buffer_free(query);
    puts("attention tests passed");
    return 0;
}
