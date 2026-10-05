#include "tensor_compute_backend.h"
#include "tensor_f16.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void near(float actual, float expected) {
    assert(fabsf(actual - expected) < 2e-4f);
}

static float test_weight(size_t i, size_t salt, float scale) {
    return (float)((int)((i * 17 + salt * 13) % 29) - 14) * scale;
}

static void test_vision_sequence(const TensorComputeBackend *backend) {
    if (backend->vision_sequence_begin_f32 == NULL ||
        backend->vision_sequence_layer_f32 == NULL ||
        backend->vision_sequence_finish_f32 == NULL ||
        backend->vision_layer_f32 == NULL) return;
    const size_t rows = 32, hidden = 64, intermediate = 128, count = rows * hidden;
    float n1s[64], n1b[64], qw[4096], qb[64], kw[4096], kb[64];
    float vw[4096], vb[64], ow[4096], ob[64], n2s[64], n2b[64];
    float f1w[8192], f1b[128], f2w[8192], f2b[64];
    for (size_t i = 0; i < 64; ++i) {
        n1s[i] = 1.0f + test_weight(i, 1, .002f); n1b[i] = test_weight(i, 2, .002f);
        qb[i] = test_weight(i, 4, .002f); kb[i] = test_weight(i, 6, .002f);
        vb[i] = test_weight(i, 8, .002f); ob[i] = test_weight(i, 10, .002f);
        n2s[i] = 1.0f + test_weight(i, 11, .002f); n2b[i] = test_weight(i, 12, .002f);
        f2b[i] = test_weight(i, 16, .002f);
    }
    for (size_t i = 0; i < 4096; ++i) {
        qw[i] = test_weight(i, 3, .001f); kw[i] = test_weight(i, 5, .001f);
        vw[i] = test_weight(i, 7, .001f); ow[i] = test_weight(i, 9, .001f);
    }
    for (size_t i = 0; i < 8192; ++i) {
        f1w[i] = test_weight(i, 13, .001f); f2w[i] = test_weight(i, 15, .001f);
    }
    for (size_t i = 0; i < 128; ++i) f1b[i] = test_weight(i, 14, .002f);
    const TensorVisionLayerF32 layer = {n1s,n1b,qw,qb,kw,kb,vw,vb,ow,ob,
        n2s,n2b,f1w,f1b,f2w,f2b};
    float initial[count], sequence_output[count], fallback_output[count];
    for (size_t i = 0; i < count; ++i) initial[i] = test_weight(i, 71, .01f);
    void *context = backend->vision_sequence_begin_f32(initial, rows, hidden);
    assert(context != NULL);
    assert(backend->vision_sequence_layer_f32(context, &layer, rows, hidden, intermediate, 1e-5f));
    assert(backend->vision_sequence_layer_f32(context, &layer, rows, hidden, intermediate, 1e-5f));
    assert(backend->vision_sequence_finish_f32(context, sequence_output));
    backend->vision_sequence_release_f32(context);
    for (size_t i = 0; i < count; ++i) fallback_output[i] = initial[i];
    assert(backend->vision_layer_f32(fallback_output, &layer, rows, hidden, intermediate, 1e-5f));
    assert(backend->vision_layer_f32(fallback_output, &layer, rows, hidden, intermediate, 1e-5f));
    for (size_t i = 0; i < count; ++i) near(sequence_output[i], fallback_output[i]);
}

static void test_vision_sequence_f16(const TensorComputeBackend *backend) {
    if (backend->vision_sequence_begin_f16 == NULL ||
        backend->vision_sequence_layer_f16 == NULL ||
        backend->vision_sequence_finish_f16 == NULL ||
        backend->vision_layer_f16 == NULL) return;
    const size_t rows = 32, hidden = 64, intermediate = 128, count = rows * hidden;
    uint16_t n1s[64], n1b[64], qw[4096], qb[64], kw[4096], kb[64];
    uint16_t vw[4096], vb[64], ow[4096], ob[64], n2s[64], n2b[64];
    uint16_t f1w[8192], f1b[128], f2w[8192], f2b[64];
#define FILL_HALF(array, n, salt, scale) \
    for (size_t i = 0; i < (n); ++i) (array)[i] = tensor_f32_to_f16(test_weight(i, (salt), (scale)))
    FILL_HALF(n1s,64,1,.002f); FILL_HALF(n1b,64,2,.002f);
    for (size_t i = 0; i < 64; ++i) {
        n1s[i] = tensor_f32_to_f16(1.0f + test_weight(i, 1, .002f));
        n2s[i] = tensor_f32_to_f16(1.0f + test_weight(i, 11, .002f));
    }
    FILL_HALF(qw,4096,3,.001f); FILL_HALF(qb,64,4,.002f);
    FILL_HALF(kw,4096,5,.001f); FILL_HALF(kb,64,6,.002f);
    FILL_HALF(vw,4096,7,.001f); FILL_HALF(vb,64,8,.002f);
    FILL_HALF(ow,4096,9,.001f); FILL_HALF(ob,64,10,.002f);
    FILL_HALF(n2b,64,12,.002f); FILL_HALF(f1w,8192,13,.001f);
    FILL_HALF(f1b,128,14,.002f); FILL_HALF(f2w,8192,15,.001f);
    FILL_HALF(f2b,64,16,.002f);
#undef FILL_HALF
    const TensorVisionLayerF16 layer = {n1s,n1b,qw,qb,kw,kb,vw,vb,ow,ob,
        n2s,n2b,f1w,f1b,f2w,f2b};
    uint16_t initial[count], sequence_output[count], fallback_output[count];
    for (size_t i = 0; i < count; ++i) initial[i] = tensor_f32_to_f16(test_weight(i, 71, .01f));
    void *context = backend->vision_sequence_begin_f16(initial, rows, hidden);
    assert(context != NULL);
    assert(backend->vision_sequence_layer_f16(context, &layer, rows, hidden, intermediate, 1e-5f));
    assert(backend->vision_sequence_layer_f16(context, &layer, rows, hidden, intermediate, 1e-5f));
    assert(backend->vision_sequence_finish_f16(context, sequence_output));
    backend->vision_sequence_release_f16(context);
    for (size_t i = 0; i < count; ++i) fallback_output[i] = initial[i];
    assert(backend->vision_layer_f16(fallback_output, &layer, rows, hidden, intermediate, 1e-5f));
    assert(backend->vision_layer_f16(fallback_output, &layer, rows, hidden, intermediate, 1e-5f));
    for (size_t i = 0; i < count; ++i)
        near(tensor_f16_to_f32(sequence_output[i]), tensor_f16_to_f32(fallback_output[i]));
}

int main(void) {
    const TensorComputeBackend *backend = tensor_compute_metal_backend();
    if (backend == NULL || !backend->is_available()) {
        puts("Metal compute tests skipped");
        return 0;
    }
    const float input[] = {1, 2, 3, 4, 5, 6};
    const float weight[] = {1, 0, 0, 0, 1, 0};
    float output[6] = {0};
    assert(backend->linear_f32(input, weight, output, 2, 3, 2));
    near(output[0], 1);
    near(output[1], 2);
    near(output[2], 4);
    near(output[3], 5);

    const float table[] = {0, 1, 2, 3, 4, 5, 6, 7};
    const uint32_t ids[] = {2, 0};
    float embeddings[4] = {0};
    assert(backend->embedding_f32(table, ids, embeddings, 2, 2, 4));
    near(embeddings[0], 4);
    near(embeddings[1], 5);
    near(embeddings[2], 0);
    near(embeddings[3], 1);

    const float norm_input[] = {1, 2, 3, 4};
    const float norm_weight[] = {1, 1};
    float norm_output[4] = {0};
    assert(backend->rms_norm_f32(norm_input, norm_weight, norm_output, 2, 2, 1e-5f));
    near(norm_output[0], 0.63245f);
    near(norm_output[1], 1.2649f);
    near(norm_output[2], 0.84853f);
    near(norm_output[3], 1.13137f);

    const float layer_input[] = {1, 3, 2, 4};
    const float layer_scale[] = {1, 1};
    const float layer_bias[] = {0, 0};
    float layer_output[4] = {0};
    assert(backend->layer_norm_f32(layer_input, layer_scale, layer_bias,
                                   layer_output, 2, 2, 1e-5f));
    near(layer_output[0], -1.0f);
    near(layer_output[1], 1.0f);
    near(layer_output[2], -1.0f);
    near(layer_output[3], 1.0f);

    const float bias_input[] = {1, 2, 3, 4};
    const float row_bias[] = {10, 20};
    float biased[4] = {0};
    assert(backend->bias_add_f32(bias_input, row_bias, biased, 2, 2));
    near(biased[0], 11);
    near(biased[1], 22);
    near(biased[2], 13);
    near(biased[3], 24);

    const float gelu_input[] = {0, 1, 10.375f, -10.375f, 20, -20};
    float gelu_output[6] = {0};
    assert(backend->gelu_f32(gelu_input, gelu_output, 6));
    near(gelu_output[0], 0);
    near(gelu_output[1], 0.84119f);
    assert(isfinite(gelu_output[2]) && isfinite(gelu_output[3]));
    near(gelu_output[2], 10.375f);
    near(gelu_output[3], 0);
    assert(isfinite(gelu_output[4]) && isfinite(gelu_output[5]));

    float query[] = {1, 0, 1, 0};
    float key[] = {1, 0, 1, 0};
    assert(backend->rope_f32(query, key, 2, 1, 1, 2, 10000.0));
    near(query[0], 1);
    near(query[1], 0);
    near(query[2], cosf(1.0f));
    near(query[3], sinf(1.0f));
    near(key[2], cosf(1.0f));
    near(key[3], sinf(1.0f));

    const float gate[] = {0, 1};
    const float up[] = {2, 3};
    float gated[2] = {0};
    assert(backend->silu_multiply_f32(gate, up, gated, 2));
    near(gated[0], 0);
    near(gated[1], 2.19318f);

    const float left[] = {1, 2, 3};
    const float right[] = {4, 5, 6};
    float sum[3] = {0};
    assert(backend->residual_add_f32(left, right, sum, 3));
    near(sum[0], 5);
    near(sum[1], 7);
    near(sum[2], 9);

    const float logits[] = {-2, 5, 5, 1};
    uint32_t best = UINT32_MAX;
    assert(backend->argmax_f32(logits, 4, &best));
    assert(best == 1);

    const uint16_t input_f16[] = {0x3c00, 0x4000, 0x4200, 0x4400};
    const uint16_t q_weight_f16[] = {0x3c00, 0, 0, 0x3c00};
    const uint16_t k_weight_f16[] = {0, 0x4000, 0x4000, 0};
    const uint16_t v_weight_f16[] = {0x3c00, 0x3c00, 0x3c00, 0xbc00};
    uint16_t query_f16[4] = {0}, key_f16[4] = {0}, value_f16[4] = {0};
    assert(backend->qkv_f16 != NULL);
    assert(backend->qkv_f16(input_f16, q_weight_f16, k_weight_f16, v_weight_f16,
        NULL, NULL, NULL, query_f16, key_f16, value_f16, 2, 2, 2));
    const uint16_t expected_query_f16[] = {0x3c00, 0x4000, 0x4200, 0x4400};
    const uint16_t expected_key_f16[] = {0x4400, 0x4000, 0x4800, 0x4600};
    const uint16_t expected_value_f16[] = {0x4200, 0xbc00, 0x4700, 0xbc00};
    for (size_t index = 0; index < 4; ++index) {
        assert(query_f16[index] == expected_query_f16[index]);
        assert(key_f16[index] == expected_key_f16[index]);
        assert(value_f16[index] == expected_value_f16[index]);
    }
    test_vision_sequence(backend);
    test_vision_sequence_f16(backend);
    backend->release_cache();
    puts("Metal compute tests passed");
    return 0;
}
