#include "tensor_ops.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static TensorBuffer *make_buffer(const uint64_t *shape, size_t rank,
                                 const float *values, size_t count) {
    TensorBuffer *buffer = NULL;
    char error[256] = {0};
    assert(tensor_buffer_create(TENSOR_DTYPE_F32, shape, rank, &buffer,
                                error, sizeof(error)));
    assert(tensor_buffer_element_count(buffer) == count);
    if (values != NULL) {
        float *data = tensor_buffer_data_mut(buffer);
        for (size_t index = 0; index < count; index++) data[index] = values[index];
    }
    return buffer;
}

static void assert_near(float actual, float expected) {
    assert(fabsf(actual - expected) < 1e-5f);
}

int main(void) {
    char error[256] = {0};
    const uint64_t left_shape[] = {2, 3};
    const uint64_t right_shape[] = {3, 2};
    const uint64_t result_shape[] = {2, 2};
    const float left_values[] = {1, 2, 3, 4, 5, 6};
    const float right_values[] = {7, 8, 9, 10, 11, 12};
    TensorBuffer *left = make_buffer(left_shape, 2, left_values, 6);
    TensorBuffer *right = make_buffer(right_shape, 2, right_values, 6);
    TensorBuffer *product = make_buffer(result_shape, 2, NULL, 4);
    assert(tensor_matmul_f32(left, right, product, error, sizeof(error)));
    const float *product_data = tensor_buffer_data(product);
    assert_near(product_data[0], 58.0f);
    assert_near(product_data[1], 64.0f);
    assert_near(product_data[2], 139.0f);
    assert_near(product_data[3], 154.0f);

    const uint64_t linear_weight_shape[] = {2, 3};
    const uint64_t linear_bias_shape[] = {2};
    const uint64_t linear_output_shape[] = {2, 2};
    const float linear_weight_values[] = {1, 0, -1, 0.5f, 2, 1};
    const float linear_bias_values[] = {0.25f, -0.5f};
    TensorBuffer *linear_weight = make_buffer(linear_weight_shape, 2,
                                              linear_weight_values, 6);
    TensorBuffer *linear_bias = make_buffer(linear_bias_shape, 1,
                                             linear_bias_values, 2);
    TensorBuffer *linear_output = make_buffer(linear_output_shape, 2, NULL, 4);
    assert(tensor_linear_f32(left, linear_weight, linear_bias, linear_output,
                             error, sizeof(error)));
    const float *linear_data = tensor_buffer_data(linear_output);
    assert_near(linear_data[0], -1.75f);
    assert_near(linear_data[1], 7.0f);
    assert_near(linear_data[2], -1.75f);
    assert_near(linear_data[3], 17.5f);

    const uint64_t embedding_shape[] = {4, 3};
    const uint64_t id_shape[] = {3};
    const uint64_t gathered_shape[] = {3, 3};
    const float embedding_values[] = {0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32};
    const uint32_t id_values[] = {2, 0, 2};
    TensorBuffer *embedding_table = make_buffer(embedding_shape, 2,
                                                embedding_values, 12);
    TensorBuffer *ids = NULL;
    assert(tensor_buffer_create(TENSOR_DTYPE_U32, id_shape, 1, &ids,
                                error, sizeof(error)));
    memcpy(tensor_buffer_data_mut(ids), id_values, sizeof(id_values));
    TensorBuffer *gathered = make_buffer(gathered_shape, 2, NULL, 9);
    assert(tensor_embedding_lookup_f32(embedding_table, ids, gathered,
                                       error, sizeof(error)));
    const float *gathered_data = tensor_buffer_data(gathered);
    const float gathered_expected[] = {20, 21, 22, 0, 1, 2, 20, 21, 22};
    for (size_t index = 0; index < 9; index++)
        assert_near(gathered_data[index], gathered_expected[index]);
    ((uint32_t *)tensor_buffer_data_mut(ids))[1] = 4;
    assert(!tensor_embedding_lookup_f32(embedding_table, ids, gathered,
                                        error, sizeof(error)));

    const uint64_t norm_shape[] = {2, 2};
    const uint64_t weight_shape[] = {2};
    const float norm_values[] = {3, 4, 0, 2};
    const float weight_values[] = {2, 0.5f};
    TensorBuffer *norm_input = make_buffer(norm_shape, 2, norm_values, 4);
    TensorBuffer *weight = make_buffer(weight_shape, 1, weight_values, 2);
    TensorBuffer *norm_output = make_buffer(norm_shape, 2, NULL, 4);
    assert(tensor_rms_norm_f32(norm_input, weight, 0.0f, norm_output,
                               error, sizeof(error)));
    const float *norm_data = tensor_buffer_data(norm_output);
    assert_near(norm_data[0], 1.69705627f);
    assert_near(norm_data[1], 0.56568545f);
    assert_near(norm_data[2], 0.0f);
    assert_near(norm_data[3], 0.70710678f);

    const uint64_t layer_shape[] = {2, 4};
    const uint64_t affine_shape[] = {4};
    const float layer_values[] = {1, 2, 3, 4, 2, 2, 2, 2};
    const float scale_values[] = {1, 2, 1, 0.5f};
    const float bias_values[] = {0, 1, -1, 2};
    TensorBuffer *layer_input = make_buffer(layer_shape, 2, layer_values, 8);
    TensorBuffer *layer_scale = make_buffer(affine_shape, 1, scale_values, 4);
    TensorBuffer *layer_bias = make_buffer(affine_shape, 1, bias_values, 4);
    TensorBuffer *layer_output = make_buffer(layer_shape, 2, NULL, 8);
    assert(tensor_layer_norm_f32(layer_input, layer_scale, layer_bias, 1e-5f,
                                 layer_output, error, sizeof(error)));
    const float *layer_data = tensor_buffer_data(layer_output);
    assert_near(layer_data[0], -1.34164079f);
    assert_near(layer_data[1], 0.10557281f);
    assert_near(layer_data[2], -0.55278640f);
    assert_near(layer_data[3], 2.67082039f);
    assert_near(layer_data[4], 0.0f);
    assert_near(layer_data[5], 1.0f);
    assert_near(layer_data[6], -1.0f);
    assert_near(layer_data[7], 2.0f);

    const uint64_t activation_shape[] = {4};
    const float activation_values[] = {-2, -1, 0, 1};
    TensorBuffer *activation = make_buffer(activation_shape, 1, activation_values, 4);
    TensorBuffer *activated = make_buffer(activation_shape, 1, NULL, 4);
    assert(tensor_silu_f32(activation, activated, error, sizeof(error)));
    const float *activated_data = tensor_buffer_data(activated);
    assert_near(activated_data[0], -0.23840584f);
    assert_near(activated_data[1], -0.26894143f);
    assert_near(activated_data[2], 0.0f);
    assert_near(activated_data[3], 0.73105858f);

    tensor_buffer_free(activated);
    tensor_buffer_free(activation);
    tensor_buffer_free(gathered);
    tensor_buffer_free(ids);
    tensor_buffer_free(embedding_table);
    tensor_buffer_free(layer_output);
    tensor_buffer_free(layer_bias);
    tensor_buffer_free(layer_scale);
    tensor_buffer_free(layer_input);
    tensor_buffer_free(linear_output);
    tensor_buffer_free(linear_bias);
    tensor_buffer_free(linear_weight);
    tensor_buffer_free(norm_output);
    tensor_buffer_free(weight);
    tensor_buffer_free(norm_input);
    tensor_buffer_free(product);
    tensor_buffer_free(right);
    tensor_buffer_free(left);
    puts("tensor ops tests passed");
    return 0;
}
