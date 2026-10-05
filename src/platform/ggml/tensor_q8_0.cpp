#include "tensor_ggml.h"
#include "tensor_f16.h"

#include "ggml.h"
#include "ggml-backend.h"
#if defined(INFERENCE_SDK_BACKEND_CUDA)
#include "ggml-cuda.h"
#elif defined(INFERENCE_SDK_BACKEND_METAL)
#include "ggml-metal.h"
#endif

#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

struct TensorGGMLQ8Weight {
    std::vector<uint8_t> arena;
    ggml_context *context = nullptr;
    ggml_tensor *tensor = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    size_t input_width = 0;
    size_t output_width = 0;
};

namespace {

std::once_flag backend_once;
ggml_backend_t backend = nullptr;
std::mutex backend_mutex;

void set_error(char *error, size_t capacity, const char *message) {
    if (error != nullptr && capacity > 0) std::snprintf(error, capacity, "%s", message);
}

ggml_backend_t get_backend() {
    std::call_once(backend_once, [] {
#if defined(INFERENCE_SDK_BACKEND_CUDA)
        backend = ggml_backend_cuda_init(0);
#elif defined(INFERENCE_SDK_BACKEND_METAL)
        backend = ggml_backend_metal_init();
#endif
    });
    return backend;
}

bool is_float_tensor(const TensorBuffer *tensor) {
    return tensor != nullptr &&
           (tensor_buffer_dtype(tensor) == TENSOR_DTYPE_F16 ||
            tensor_buffer_dtype(tensor) == TENSOR_DTYPE_F32);
}

float read_value(const TensorBuffer *tensor, size_t index) {
    if (tensor_buffer_dtype(tensor) == TENSOR_DTYPE_F32) {
        return static_cast<const float *>(tensor_buffer_data(tensor))[index];
    }
    const uint16_t bits = static_cast<const uint16_t *>(tensor_buffer_data(tensor))[index];
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t fraction = bits & 0x03ffu;
    uint32_t value;
    if (exponent == 0) {
        if (fraction == 0) {
            value = sign;
        } else {
            int adjusted = -14;
            while ((fraction & 0x0400u) == 0) {
                fraction <<= 1;
                --adjusted;
            }
            fraction &= 0x03ffu;
            value = sign | (static_cast<uint32_t>(adjusted + 127) << 23) |
                    (fraction << 13);
        }
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | (fraction << 13);
    } else {
        exponent = exponent - 15 + 127;
        value = sign | (exponent << 23) | (fraction << 13);
    }
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

size_t context_size(size_t tensor_count) {
    return ggml_tensor_overhead() * tensor_count + ggml_graph_overhead() + 4096;
}

ggml_context *create_context(std::vector<uint8_t> &arena, size_t tensor_count) {
    arena.resize(context_size(tensor_count));
    ggml_init_params params{};
    params.mem_size = arena.size();
    params.mem_buffer = arena.data();
    params.no_alloc = true;
    return ggml_init(params);
}

} // namespace

extern "C" int tensor_ggml_q8_weight_create(const TensorBuffer *weight,
                                              TensorGGMLQ8Weight **result,
                                              char *error, size_t error_capacity) {
    if (result != nullptr) *result = nullptr;
    if (result == nullptr || !is_float_tensor(weight) || tensor_buffer_rank(weight) != 2) {
        set_error(error, error_capacity, "Q8_0 weight creation requires a rank-2 F16 or F32 tensor");
        return 0;
    }
    const size_t output_width = static_cast<size_t>(tensor_buffer_shape(weight)[0]);
    const size_t input_width = static_cast<size_t>(tensor_buffer_shape(weight)[1]);
    if (input_width == 0 || output_width == 0 ||
        input_width % static_cast<size_t>(ggml_blck_size(GGML_TYPE_Q8_0)) != 0) {
        set_error(error, error_capacity, "Q8_0 input width must be nonzero and block aligned");
        return 0;
    }
    ggml_backend_t selected_backend = get_backend();
    if (selected_backend == nullptr) {
        set_error(error, error_capacity, "ggml GPU backend initialization failed");
        return 0;
    }

    auto *quantized = new (std::nothrow) TensorGGMLQ8Weight;
    if (quantized == nullptr) {
        set_error(error, error_capacity, "out of memory creating Q8_0 weight");
        return 0;
    }
    quantized->context = create_context(quantized->arena, 1);
    if (quantized->context == nullptr) {
        delete quantized;
        set_error(error, error_capacity, "ggml context allocation failed");
        return 0;
    }
    quantized->tensor = ggml_new_tensor_2d(quantized->context, GGML_TYPE_Q8_0,
                                            static_cast<int64_t>(input_width),
                                            static_cast<int64_t>(output_width));
    quantized->buffer = ggml_backend_alloc_ctx_tensors(quantized->context, selected_backend);
    if (quantized->tensor == nullptr || quantized->buffer == nullptr) {
        tensor_ggml_q8_weight_free(quantized);
        set_error(error, error_capacity, "ggml GPU weight allocation failed");
        return 0;
    }

    std::vector<float> float_weights(tensor_buffer_element_count(weight));
    for (size_t i = 0; i < float_weights.size(); ++i) float_weights[i] = read_value(weight, i);
    std::vector<uint8_t> packed(ggml_nbytes(quantized->tensor));
    const size_t packed_bytes = ggml_quantize_chunk(
        GGML_TYPE_Q8_0, float_weights.data(), packed.data(), 0,
        static_cast<int64_t>(output_width), static_cast<int64_t>(input_width), nullptr);
    if (packed_bytes != packed.size()) {
        tensor_ggml_q8_weight_free(quantized);
        set_error(error, error_capacity, "ggml Q8_0 quantization returned an unexpected size");
        return 0;
    }
    ggml_backend_tensor_set(quantized->tensor, packed.data(), 0, packed.size());
    quantized->input_width = input_width;
    quantized->output_width = output_width;
    *result = quantized;
    return 1;
}

extern "C" void tensor_ggml_q8_weight_free(TensorGGMLQ8Weight *weight) {
    if (weight == nullptr) return;
    if (weight->buffer != nullptr) ggml_backend_buffer_free(weight->buffer);
    if (weight->context != nullptr) ggml_free(weight->context);
    delete weight;
}

extern "C" int tensor_ggml_linear_q8(const TensorBuffer *input,
                                       const TensorGGMLQ8Weight *weight,
                                       const TensorBuffer *bias, TensorBuffer *output,
                                       char *error, size_t error_capacity) {
    if (!is_float_tensor(input) || !is_float_tensor(output) || weight == nullptr ||
        tensor_buffer_rank(input) == 0 || tensor_buffer_rank(output) != tensor_buffer_rank(input) ||
        (bias != nullptr && (!is_float_tensor(bias) || tensor_buffer_rank(bias) != 1))) {
        set_error(error, error_capacity, "ggml Q8_0 linear requires matching F16/F32 tensors and a rank-2 quantized weight");
        return 0;
    }
    const size_t rank = tensor_buffer_rank(input);
    const size_t input_width = static_cast<size_t>(tensor_buffer_shape(input)[rank - 1]);
    const size_t output_width = weight->output_width;
    if (input_width != weight->input_width || input_width == 0 ||
        tensor_buffer_shape(output)[rank - 1] != output_width ||
        (bias != nullptr && tensor_buffer_shape(bias)[0] != output_width)) {
        set_error(error, error_capacity, "ggml Q8_0 linear dimensions do not match");
        return 0;
    }
    for (size_t i = 0; i + 1 < rank; ++i) {
        if (tensor_buffer_shape(input)[i] != tensor_buffer_shape(output)[i]) {
            set_error(error, error_capacity, "ggml Q8_0 linear leading dimensions do not match");
            return 0;
        }
    }
    const size_t rows = tensor_buffer_element_count(input) / input_width;
    if (rows == 0 || rows > static_cast<size_t>(INT64_MAX) ||
        input_width > static_cast<size_t>(INT64_MAX) ||
        output_width > static_cast<size_t>(INT64_MAX)) {
        set_error(error, error_capacity, "ggml Q8_0 linear dimensions exceed supported limits");
        return 0;
    }
    ggml_backend_t selected_backend = get_backend();
    if (selected_backend == nullptr) {
        set_error(error, error_capacity, "ggml GPU backend is unavailable");
        return 0;
    }

    std::lock_guard<std::mutex> lock(backend_mutex);
    std::vector<uint8_t> arena;
    ggml_context *ctx = create_context(arena, bias == nullptr ? 4 : 5);
    if (ctx == nullptr) {
        set_error(error, error_capacity, "ggml graph context allocation failed");
        return 0;
    }
    const enum ggml_type type = tensor_buffer_dtype(input) == TENSOR_DTYPE_F16 ?
                                GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor *x = ggml_new_tensor_2d(ctx, type, static_cast<int64_t>(input_width),
                                         static_cast<int64_t>(rows));
    ggml_tensor *result = ggml_mul_mat(ctx, weight->tensor, x);
    ggml_tensor *bias_tensor = nullptr;
    std::vector<float> float_bias;
    if (bias != nullptr) {
        bias_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                                         static_cast<int64_t>(output_width));
        result = ggml_add(ctx, result, bias_tensor);
    }
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, selected_backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        set_error(error, error_capacity, "ggml GPU graph allocation failed");
        return 0;
    }
    ggml_backend_tensor_set(x, tensor_buffer_data(input), 0, tensor_buffer_byte_length(input));
    if (bias_tensor != nullptr) {
        float_bias.resize(output_width);
        for (size_t i = 0; i < output_width; ++i) float_bias[i] = read_value(bias, i);
        ggml_backend_tensor_set(bias_tensor, float_bias.data(), 0,
                                float_bias.size() * sizeof(float));
    }
    const ggml_status status = ggml_backend_graph_compute(selected_backend, graph);
    if (status == GGML_STATUS_SUCCESS) {
        std::vector<float> float_output(tensor_buffer_element_count(output));
        ggml_backend_tensor_get(result, float_output.data(), 0,
                                float_output.size() * sizeof(float));
        if (tensor_buffer_dtype(output) == TENSOR_DTYPE_F32) {
            std::memcpy(tensor_buffer_data_mut(output), float_output.data(),
                        float_output.size() * sizeof(float));
        } else {
            auto *output_data = static_cast<uint16_t *>(tensor_buffer_data_mut(output));
            for (size_t i = 0; i < float_output.size(); ++i)
                output_data[i] = tensor_f32_to_f16(float_output[i]);
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    if (status != GGML_STATUS_SUCCESS) {
        set_error(error, error_capacity, "ggml GPU graph execution failed");
        return 0;
    }
    return 1;
}
