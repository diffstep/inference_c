#include "tensor_graph.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static TensorBuffer *make_f32(const uint64_t *shape, size_t rank,
                             const float *values, size_t count) {
    TensorBuffer *tensor = NULL;
    char error[256] = {0};
    assert(tensor_buffer_create(TENSOR_DTYPE_F32, shape, rank, &tensor,
                                error, sizeof(error)));
    assert(tensor_buffer_element_count(tensor) == count);
    if (values != NULL) {
        float *data = tensor_buffer_data_mut(tensor);
        for (size_t i = 0; i < count; ++i) data[i] = values[i];
    }
    return tensor;
}

static void near(float actual, float expected) {
    assert(fabsf(actual - expected) < 1.0e-5f);
}

int main(void) {
    char error[256] = {0};
    TensorGraph *graph = NULL;
    assert(tensor_graph_create(&graph, error, sizeof(error)));

    const uint64_t x_shape[] = {2, 3}, w_shape[] = {2, 3}, b_shape[] = {2};
    TensorBuffer *weight = make_f32(w_shape, 2, (float[]){1, 0, -1, 0.5f, 2, 1}, 6);
    TensorBuffer *bias = make_f32(b_shape, 1, (float[]){0.25f, -0.5f}, 2);
    TensorGraphValue x, w, b, projected, activated;
    assert(tensor_graph_add_input(graph, "x", TENSOR_DTYPE_F32, x_shape, 2,
                                  &x, error, sizeof(error)));
    assert(tensor_graph_add_constant(graph, "weight", weight, &w, error, sizeof(error)));
    assert(tensor_graph_add_constant(graph, "bias", bias, &b, error, sizeof(error)));
    assert(tensor_graph_add_linear(graph, "projection", x, w, b, 1,
                                   &projected, error, sizeof(error)));
    assert(tensor_graph_add_silu(graph, "activation", projected, &activated,
                                 error, sizeof(error)));
    assert(tensor_graph_mark_output(graph, activated, error, sizeof(error)));
    assert(tensor_graph_compile(graph, error, sizeof(error)));
    TensorGraphValue late_node;
    assert(!tensor_graph_add_silu(graph, "late_node", activated, &late_node,
                                  error, sizeof(error)));

    const float input_values[] = {1, 2, 3, 4, 5, 6};
    TensorBuffer *input = make_f32(x_shape, 2, input_values, 6);
    TensorGraphBinding binding = {.name = "x", .value = input};
    TensorBuffer **outputs = NULL;
    size_t output_count = 0;
    assert(tensor_graph_run(graph, &binding, 1, &outputs, &output_count,
                            error, sizeof(error)));
    assert(output_count == 1);
    assert(tensor_buffer_rank(outputs[0]) == 2);
    assert(tensor_buffer_shape(outputs[0])[0] == 2 && tensor_buffer_shape(outputs[0])[1] == 2);
    const float *actual = tensor_buffer_data(outputs[0]);
    const float raw[] = {-1.75f, 7.0f, -1.75f, 17.5f};
    for (size_t i = 0; i < 4; ++i)
        near(actual[i], raw[i] / (1.0f + expf(-raw[i])));
    tensor_buffer_free(outputs[0]);
    free(outputs);

    const uint64_t bad_shape[] = {2, 4};
    TensorBuffer *bad_input = make_f32(bad_shape, 2, NULL, 8);
    binding.value = bad_input;
    outputs = NULL;
    assert(!tensor_graph_run(graph, &binding, 1, &outputs, &output_count,
                             error, sizeof(error)));
    assert(outputs == NULL && output_count == 0);

    tensor_buffer_free(bad_input);
    tensor_buffer_free(input);
    tensor_buffer_free(weight);
    tensor_buffer_free(bias);
    tensor_graph_free(graph);

    /* Returning an input value must produce independently owned output memory. */
    TensorGraph *identity = NULL;
    assert(tensor_graph_create(&identity, error, sizeof(error)));
    TensorGraphValue identity_input;
    assert(tensor_graph_add_input(identity, "input", TENSOR_DTYPE_F32,
                                  x_shape, 2, &identity_input,
                                  error, sizeof(error)));
    assert(tensor_graph_mark_output(identity, identity_input, error, sizeof(error)));
    assert(tensor_graph_compile(identity, error, sizeof(error)));
    input = make_f32(x_shape, 2, input_values, 6);
    binding = (TensorGraphBinding){.name = "input", .value = input};
    outputs = NULL;
    assert(tensor_graph_run(identity, &binding, 1, &outputs, &output_count,
                            error, sizeof(error)));
    assert(output_count == 1 && outputs[0] != input);
    assert(memcmp(tensor_buffer_data(input), tensor_buffer_data(outputs[0]),
                  tensor_buffer_byte_length(input)) == 0);
    tensor_buffer_free(outputs[0]);
    free(outputs);
    tensor_buffer_free(input);
    tensor_graph_free(identity);

    puts("tensor graph IR checks passed");
    return 0;
}
