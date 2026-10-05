#include "tensor_coreml.h"
#include "tensor_graph.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(void) {
    const char *path = getenv("INFERENCE_SDK_COREML_TEST_MODEL");
    if (path == NULL || path[0] == '\0') {
        puts("SKIP: set INFERENCE_SDK_COREML_TEST_MODEL to a compiled Core ML package");
        return 77;
    }
    if (!tensor_coreml_available()) {
        puts("SKIP: Core ML Neural Engine unavailable");
        return 77;
    }

    char error[512] = {0};
    TensorCoreMLModel *model = NULL;
    if (!tensor_coreml_model_open(path, &model, error, sizeof(error))) {
        fprintf(stderr, "Core ML model open failed: %s\n", error);
        return 1;
    }

    const uint64_t input_shape[] = {1, 3, 512, 512};
    const uint64_t output_shape[] = {1, 64, 576};
    TensorBuffer *input = NULL, *output = NULL;
    if (!tensor_buffer_create(TENSOR_DTYPE_F16, input_shape, 4, &input,
                              error, sizeof(error)) ||
        !tensor_buffer_create(TENSOR_DTYPE_F16, output_shape, 3, &output,
                              error, sizeof(error))) {
        fprintf(stderr, "Tensor allocation failed: %s\n", error);
        tensor_buffer_free(input);
        tensor_buffer_free(output);
        tensor_coreml_model_close(model);
        return 1;
    }
    TensorCoreMLBinding binding_in = {.name = "pixels", .value = input};
    TensorCoreMLBinding binding_out = {.name = "features", .value = output};
    int ok = tensor_coreml_model_predict(model, &binding_in, 1,
        &binding_out, 1, error, sizeof(error));
    if (!ok) fprintf(stderr, "Core ML prediction failed: %s\n", error);
    const uint16_t *values = tensor_buffer_data(output);
    if (ok) for (size_t i = 0; i < tensor_buffer_element_count(output); ++i) {
        const uint16_t bits = values[i];
        const unsigned exponent = (bits >> 10) & 31u;
        if (exponent == 31u) {
            fprintf(stderr, "Core ML output contains a non-finite value at %zu\n", i);
            ok = 0;
            break;
        }
    }
    tensor_buffer_free(input);
    tensor_buffer_free(output);

    /* Exercise one graph layer on ANE and an independent F16 linear on Metal.
       Graph scheduling places both ready nodes in the same parallel wave. */
    TensorGraph *graph = NULL;
    const uint64_t vector_shape[] = {1, 2};
    const uint64_t weight_shape[] = {1, 2};
    TensorGraphValue pixels_value, vector_value, weight_value, vision_value,
        vision_copy_value, linear_value;
    TensorBuffer *weight_host = NULL, *vector_host = NULL, *pixels_host = NULL;
    TensorBuffer *vector_device = NULL, *pixels_device = NULL;
    TensorBuffer **graph_outputs = NULL;
    size_t graph_output_count = 0;
    if (ok && (!tensor_graph_create(&graph, error, sizeof(error)) ||
        !tensor_graph_add_input(graph, "pixels", TENSOR_DTYPE_F16, input_shape, 4,
            &pixels_value, error, sizeof(error)) ||
        !tensor_graph_add_input(graph, "vector", TENSOR_DTYPE_F16, vector_shape, 2,
            &vector_value, error, sizeof(error)) ||
        !tensor_buffer_create(TENSOR_DTYPE_F16, weight_shape, 2, &weight_host,
            error, sizeof(error)) ||
        !tensor_buffer_create(TENSOR_DTYPE_F16, vector_shape, 2, &vector_host,
            error, sizeof(error)) ||
        !tensor_buffer_create(TENSOR_DTYPE_F16, input_shape, 4, &pixels_host,
            error, sizeof(error)))) ok = 0;
    if (ok) {
        uint16_t *weight = tensor_buffer_data_mut(weight_host);
        uint16_t *vector = tensor_buffer_data_mut(vector_host);
        weight[0] = 0x3c00; weight[1] = 0x4000; /* 1, 2 */
        vector[0] = 0x4200; vector[1] = 0x4400; /* 3, 4 */
        if (!tensor_graph_add_constant(graph, "weight", weight_host, &weight_value,
                error, sizeof(error)) ||
            !tensor_graph_add_coreml(graph, "ane_vision", model, "pixels",
                pixels_value, "features", TENSOR_DTYPE_F16, output_shape, 3,
                &vision_value, error, sizeof(error)) ||
            !tensor_graph_add_coreml(graph, "ane_vision_copy", model, "pixels",
                pixels_value, "features", TENSOR_DTYPE_F16, output_shape, 3,
                &vision_copy_value, error, sizeof(error)) ||
            !tensor_graph_add_linear(graph, "metal_linear", vector_value,
                weight_value, 0, 0, &linear_value, error, sizeof(error)) ||
            !tensor_graph_mark_output(graph, vision_value, error, sizeof(error)) ||
            !tensor_graph_mark_output(graph, vision_copy_value, error, sizeof(error)) ||
            !tensor_graph_mark_output(graph, linear_value, error, sizeof(error)) ||
            !tensor_graph_compile(graph, error, sizeof(error)) ||
            !tensor_buffer_create_device(TENSOR_DTYPE_F16, input_shape, 4,
                &pixels_device, error, sizeof(error)) ||
            !tensor_buffer_create_device(TENSOR_DTYPE_F16, vector_shape, 2,
                &vector_device, error, sizeof(error)) ||
            !tensor_buffer_upload(pixels_device, 0, tensor_buffer_data(pixels_host),
                tensor_buffer_byte_length(pixels_host), error, sizeof(error)) ||
            !tensor_buffer_upload(vector_device, 0, tensor_buffer_data(vector_host),
                tensor_buffer_byte_length(vector_host), error, sizeof(error))) ok = 0;
    }
    if (ok) {
        TensorGraphBinding graph_inputs[] = {
            {.name = "pixels", .value = pixels_device},
            {.name = "vector", .value = vector_device},
        };
        struct timespec start, finish;
        clock_gettime(CLOCK_MONOTONIC, &start);
        ok = tensor_graph_run(graph, graph_inputs, 2, &graph_outputs,
                              &graph_output_count, error, sizeof(error));
        clock_gettime(CLOCK_MONOTONIC, &finish);
        const double elapsed_ms = (finish.tv_sec - start.tv_sec) * 1000.0 +
            (finish.tv_nsec - start.tv_nsec) / 1000000.0;
        printf("hybrid_graph_elapsed_ms=%.3f (2 ANE nodes + 1 Metal node)\n", elapsed_ms);
        if (!ok) fprintf(stderr, "Hybrid graph run failed: %s\n", error);
    }
    if (ok && (graph_output_count != 3 ||
        tensor_buffer_storage(graph_outputs[0]) != TENSOR_BUFFER_STORAGE_DEVICE ||
        tensor_buffer_storage(graph_outputs[1]) != TENSOR_BUFFER_STORAGE_DEVICE)) {
        fprintf(stderr, "Hybrid graph returned invalid outputs\n");
        ok = 0;
    }
    if (ok) {
        const size_t vision_bytes = tensor_buffer_byte_length(graph_outputs[0]);
        uint8_t *first_vision = malloc(vision_bytes);
        uint8_t *second_vision = malloc(vision_bytes);
        if (first_vision == NULL || second_vision == NULL ||
            !tensor_buffer_download(graph_outputs[0], 0, first_vision,
                vision_bytes, error, sizeof(error)) ||
            !tensor_buffer_download(graph_outputs[1], 0, second_vision,
                vision_bytes, error, sizeof(error)) ||
            memcmp(first_vision, second_vision, vision_bytes) != 0) {
            fprintf(stderr, "Parallel ANE branch outputs do not match: %s\n", error);
            ok = 0;
        }
        free(first_vision);
        free(second_vision);
    }
    if (ok) {
        uint16_t linear_result = 0;
        ok = tensor_buffer_download(graph_outputs[2], 0, &linear_result,
                                    sizeof(linear_result), error, sizeof(error));
        if (!ok || linear_result != 0x4980) {
            fprintf(stderr, "Metal linear result mismatch: 0x%04x (%s)\n",
                    linear_result, error);
            ok = 0;
        }
    }
    if (!ok && error[0] != '\0') fprintf(stderr, "Hybrid graph error: %s\n", error);
    for (size_t i = 0; i < graph_output_count; ++i) tensor_buffer_free(graph_outputs[i]);
    free(graph_outputs);
    tensor_buffer_free(weight_host);
    tensor_buffer_free(vector_host);
    tensor_buffer_free(pixels_host);
    tensor_buffer_free(vector_device);
    tensor_buffer_free(pixels_device);
    tensor_graph_free(graph);
    tensor_coreml_model_close(model);
    if (ok) puts("Core ML ANE model load and prediction passed");
    return ok ? 0 : 1;
}
