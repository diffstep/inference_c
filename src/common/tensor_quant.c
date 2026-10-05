#include "tensor_quant.h"
#include "tensor_compute_backend.h"
#include "tensor_buffer_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static void quant_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity != 0) snprintf(error, capacity, "%s", message);
}

static float half_to_float(uint16_t h) {
    const unsigned sign = (unsigned)(h & 0x8000u) << 16;
    unsigned exp = (h >> 10) & 31u, mantissa = h & 1023u, bits;
    if (exp == 0) {
        if (mantissa == 0) bits = sign;
        else {
            int e = -14;
            while ((mantissa & 1024u) == 0) { mantissa <<= 1; --e; }
            bits = sign | (unsigned)(e + 127) << 23 | (mantissa & 1023u) << 13;
        }
    } else if (exp == 31) bits = sign | 0x7f800000u | mantissa << 13;
    else bits = sign | (exp + 112u) << 23 | mantissa << 13;
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t float_to_half(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(bits));
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const int exponent = (int)((bits >> 23) & 255u) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        mantissa |= 0x800000u;
        return (uint16_t)(sign | ((mantissa + (1u << (13 + -exponent - 1))) >> (13 + -exponent)));
    }
    if (exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    mantissa += 0xfffu + ((mantissa >> 13) & 1u);
    if (mantissa & 0x800000u) return (uint16_t)(sign | (uint16_t)(exponent + 1) << 10);
    return (uint16_t)(sign | (uint16_t)exponent << 10 | (uint16_t)(mantissa >> 13));
}

int tensor_quantize_linear_weight_f16(const uint16_t *weight, size_t output_width,
                                      size_t input_width, TensorWeightQuantization format,
                                      TensorBuffer **result, char *error, size_t capacity) {
    if (result) *result = NULL;
    const size_t block_bytes = format == TENSOR_WEIGHT_Q8_0 ? 34 :
                               format == TENSOR_WEIGHT_Q4_0 ? 18 : 0;
    if (!weight || !result || !output_width || !input_width || input_width % 32 || !block_bytes ||
        output_width > SIZE_MAX / (input_width / 32) / block_bytes) {
        quant_error(error, capacity, "quantized linear weight requires a valid format and input width divisible by 32");
        return 0;
    }
    const uint64_t shape[3] = {output_width, input_width / 32, block_bytes};
    if (!tensor_buffer_create(TENSOR_DTYPE_U8, shape, 3, result, error, capacity)) return 0;
    uint8_t *dst = tensor_buffer_data_mut(*result);
    for (size_t row = 0; row < output_width; ++row) {
        for (size_t b = 0; b < input_width / 32; ++b) {
            float x[32], max_abs = 0.0f, max_signed = 0.0f;
            for (size_t j = 0; j < 32; ++j) {
                x[j] = half_to_float(weight[row * input_width + b * 32 + j]);
                if (!isfinite(x[j])) {
                    tensor_buffer_free(*result); *result=NULL;
                    quant_error(error,capacity,"quantized weights must contain only finite values");
                    return 0;
                }
                if (fabsf(x[j]) > max_abs) max_abs = fabsf(x[j]);
                if (fabsf(x[j]) > fabsf(max_signed)) max_signed = x[j];
            }
            uint8_t *block = dst + (row * (input_width / 32) + b) * block_bytes;
            if (format == TENSOR_WEIGHT_Q8_0) {
                const float scale = max_abs / 127.0f;
                const uint16_t hscale = float_to_half(scale);
                block[0] = (uint8_t)hscale; block[1] = (uint8_t)(hscale >> 8);
                const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
                for (size_t j = 0; j < 32; ++j) {
                    int q = (int)lrintf(x[j] * inverse);
                    if (q < -127) q = -127; if (q > 127) q = 127;
                    block[2 + j] = (uint8_t)(int8_t)q;
                }
            } else {
                const float scale = max_signed / -8.0f;
                const uint16_t hscale = float_to_half(scale);
                block[0] = (uint8_t)hscale; block[1] = (uint8_t)(hscale >> 8);
                const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
                for (size_t j = 0; j < 16; ++j) {
                    int q0 = (int)(x[j] * inverse + 8.5f);
                    int q1 = (int)(x[j + 16] * inverse + 8.5f);
                    if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
                    if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
                    block[2 + j] = (uint8_t)(q0 | (q1 << 4));
                }
            }
        }
    }
    return 1;
}

static int linear_quantized_f16(const TensorBuffer *input, const TensorBuffer *weight,
                                TensorBuffer *output, size_t block_bytes,
                                TensorWeightQuantization format, char *error, size_t capacity) {
    if (!input || !weight || !output || tensor_buffer_storage(input) != TENSOR_BUFFER_STORAGE_DEVICE ||
        tensor_buffer_storage(weight) != TENSOR_BUFFER_STORAGE_DEVICE || tensor_buffer_storage(output) != TENSOR_BUFFER_STORAGE_DEVICE ||
        tensor_buffer_dtype(input) != TENSOR_DTYPE_F16 || tensor_buffer_dtype(weight) != TENSOR_DTYPE_U8 ||
        tensor_buffer_dtype(output) != TENSOR_DTYPE_F16 || tensor_buffer_rank(input) == 0 ||
        tensor_buffer_rank(weight) != 3 || tensor_buffer_rank(output) != tensor_buffer_rank(input)) {
        quant_error(error, capacity, "quantized linear requires device-resident FP16 input/output and packed U8 weights"); return 0;
    }
    const size_t rank = tensor_buffer_rank(input);
    const uint64_t *is = tensor_buffer_shape(input), *ws = tensor_buffer_shape(weight), *os = tensor_buffer_shape(output);
    const size_t iw = (size_t)is[rank-1], ow = (size_t)ws[0];
    if (ws[1] != iw / 32 || iw % 32 || ws[2] != block_bytes || os[rank-1] != ow) {
        quant_error(error, capacity, "quantized linear shapes are incompatible"); return 0;
    }
    for (size_t d=0; d+1<rank; ++d) if (is[d] != os[d]) {
        quant_error(error, capacity, "quantized linear leading dimensions do not match"); return 0;
    }
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    int (*operation)(const void *, const void *, void *, size_t, size_t, size_t) = NULL;
    if (backend) operation = format == TENSOR_WEIGHT_Q8_0 ? backend->linear_q8_0_f16_device : backend->linear_q4_0_f16_device;
    const size_t rows = tensor_buffer_element_count(input) / iw;
    if (!operation || tensor_buffer_device_handle_internal(input) == NULL ||
        tensor_buffer_device_handle_internal(weight) == NULL || tensor_buffer_device_handle_internal(output) == NULL ||
        !operation(tensor_buffer_device_handle_internal(input),
        tensor_buffer_device_handle_internal(weight), tensor_buffer_device_handle_internal(output), rows, iw, ow)) {
        quant_error(error, capacity, "quantized linear GPU kernel is unavailable"); return 0;
    }
    return 1;
}

int tensor_linear_q8_0_f16(const TensorBuffer *i, const TensorBuffer *w, TensorBuffer *o, char *e, size_t c) {
    return linear_quantized_f16(i,w,o,34,TENSOR_WEIGHT_Q8_0,e,c);
}
int tensor_linear_q4_0_f16(const TensorBuffer *i, const TensorBuffer *w, TensorBuffer *o, char *e, size_t c) {
    return linear_quantized_f16(i,w,o,18,TENSOR_WEIGHT_Q4_0,e,c);
}
