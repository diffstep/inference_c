#include "tensor_buffer.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void assert_near(float actual, float expected, float tolerance) {
    const float difference = actual > expected ? actual - expected : expected - actual;
    assert(difference <= tolerance);
}

int main(void) {
    const uint64_t shape[] = {2, 5};
    TensorBuffer *f32 = NULL;
    char error[256];
    assert(tensor_buffer_create(TENSOR_DTYPE_F32, shape, 2, &f32,
                                error, sizeof(error)));
    assert(tensor_buffer_rank(f32) == 2);
    assert(tensor_buffer_shape(f32)[0] == 2 && tensor_buffer_shape(f32)[1] == 5);
    assert(tensor_buffer_element_count(f32) == 10);
    assert(tensor_buffer_byte_length(f32) == 10 * sizeof(float));
    float *values = tensor_buffer_data_mut(f32);
    const float inputs[10] = {
        0.0f, 1.0f, -2.0f, 0.33325f, 65504.0f,
        0.00006103515625f, 5.96046448e-8f, 1e-8f, 70000.0f, -0.5f
    };
    memcpy(values, inputs, sizeof(inputs));

    TensorBuffer *bf16 = NULL;
    TensorBuffer *bf16_roundtrip = NULL;
    assert(tensor_buffer_convert(f32, TENSOR_DTYPE_BF16, &bf16,
                                 error, sizeof(error)));
    assert(tensor_buffer_convert(bf16, TENSOR_DTYPE_F32, &bf16_roundtrip,
                                 error, sizeof(error)));
    const float *bf16_values = tensor_buffer_data(bf16_roundtrip);
    assert(bf16_values[0] == 0.0f && bf16_values[1] == 1.0f);
    assert(bf16_values[2] == -2.0f);
    assert_near(bf16_values[3], inputs[3], 0.002f);

    TensorBuffer *f16 = NULL;
    TensorBuffer *f16_roundtrip = NULL;
    assert(tensor_buffer_convert(f32, TENSOR_DTYPE_F16, &f16,
                                 error, sizeof(error)));
    assert(tensor_buffer_convert(f16, TENSOR_DTYPE_F32, &f16_roundtrip,
                                 error, sizeof(error)));
    const float *f16_values = tensor_buffer_data(f16_roundtrip);
    assert(f16_values[0] == 0.0f && f16_values[1] == 1.0f);
    assert(f16_values[2] == -2.0f);
    assert_near(f16_values[3], inputs[3], 0.001f);
    assert(f16_values[4] == 65504.0f);
    assert(f16_values[5] == inputs[5]);
    assert(f16_values[6] == inputs[6]);
    assert(f16_values[7] == 0.0f);
    assert(((float_bits(f16_values[8]) >> 23) & 0xffu) == 0xffu);

    TensorBuffer *same_dtype = NULL;
    assert(tensor_buffer_convert(f16, TENSOR_DTYPE_F16, &same_dtype,
                                 error, sizeof(error)));
    assert(memcmp(tensor_buffer_data(f16), tensor_buffer_data(same_dtype),
                  tensor_buffer_byte_length(f16)) == 0);

    tensor_buffer_free(same_dtype);
    tensor_buffer_free(f16_roundtrip);
    tensor_buffer_free(f16);
    tensor_buffer_free(bf16_roundtrip);
    tensor_buffer_free(bf16);
    tensor_buffer_free(f32);
    puts("tensor buffer tests passed");
    return 0;
}
