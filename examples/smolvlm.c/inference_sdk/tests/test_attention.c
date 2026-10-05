#include "attention.h"

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
