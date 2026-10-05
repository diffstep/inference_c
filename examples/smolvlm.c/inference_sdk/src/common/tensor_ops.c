#include "tensor_ops.h"
#include "tensor_f16.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static int same_shape(const TensorBuffer *left, const TensorBuffer *right) {
    if (tensor_buffer_rank(left) != tensor_buffer_rank(right)) return 0;
    for (size_t index = 0; index < tensor_buffer_rank(left); index++) {
        if (tensor_buffer_shape(left)[index] != tensor_buffer_shape(right)[index])
            return 0;
    }
    return 1;
}

static int valid_f32(const TensorBuffer *buffer) {
    return buffer != NULL && tensor_buffer_dtype(buffer) == TENSOR_DTYPE_F32;
}

static int valid_f16(const TensorBuffer *buffer) {
    return buffer != NULL && tensor_buffer_dtype(buffer) == TENSOR_DTYPE_F16;
}

static float read_f16(const TensorBuffer *buffer, size_t index) {
    return tensor_f16_to_f32(((const uint16_t *)tensor_buffer_data(buffer))[index]);
}

static void write_f16(TensorBuffer *buffer, size_t index, float value) {
    ((uint16_t *)tensor_buffer_data_mut(buffer))[index] = tensor_f32_to_f16(value);
}

static int read_embedding_index(const TensorBuffer *indices, size_t position,
                                uint64_t *result) {
    const unsigned char *data = tensor_buffer_data(indices);
    switch (tensor_buffer_dtype(indices)) {
        case TENSOR_DTYPE_I16: {
            int16_t value;
            memcpy(&value, data + position * sizeof(value), sizeof(value));
            if (value < 0) return 0;
            *result = (uint64_t)value;
            return 1;
        }
        case TENSOR_DTYPE_U16: {
            uint16_t value;
            memcpy(&value, data + position * sizeof(value), sizeof(value));
            *result = value;
            return 1;
        }
        case TENSOR_DTYPE_I32: {
            int32_t value;
            memcpy(&value, data + position * sizeof(value), sizeof(value));
            if (value < 0) return 0;
            *result = (uint64_t)value;
            return 1;
        }
        case TENSOR_DTYPE_U32: {
            uint32_t value;
            memcpy(&value, data + position * sizeof(value), sizeof(value));
            *result = value;
            return 1;
        }
        case TENSOR_DTYPE_I64: {
            int64_t value;
            memcpy(&value, data + position * sizeof(value), sizeof(value));
            if (value < 0) return 0;
            *result = (uint64_t)value;
            return 1;
        }
        case TENSOR_DTYPE_U64:
            memcpy(result, data + position * sizeof(*result), sizeof(*result));
            return 1;
        default:
            return 0;
    }
}

int tensor_embedding_lookup_f32(const TensorBuffer *table,
                                const TensorBuffer *indices,
                                TensorBuffer *output, char *error,
                                size_t error_capacity) {
    if (!valid_f32(table) || !valid_f32(output) ||
        tensor_buffer_rank(table) != 2 || tensor_buffer_rank(indices) != 1 ||
        tensor_buffer_rank(output) != 2 ||
        (tensor_buffer_dtype(indices) != TENSOR_DTYPE_I16 &&
         tensor_buffer_dtype(indices) != TENSOR_DTYPE_U16 &&
         tensor_buffer_dtype(indices) != TENSOR_DTYPE_I32 &&
         tensor_buffer_dtype(indices) != TENSOR_DTYPE_U32 &&
         tensor_buffer_dtype(indices) != TENSOR_DTYPE_I64 &&
         tensor_buffer_dtype(indices) != TENSOR_DTYPE_U64)) {
        set_error(error, error_capacity, "embedding lookup requires a rank-2 F32 table, integer rank-1 indices, and rank-2 F32 output");
        return 0;
    }
    const uint64_t *table_shape = tensor_buffer_shape(table);
    const uint64_t *output_shape = tensor_buffer_shape(output);
    const size_t index_count = tensor_buffer_element_count(indices);
    if (table_shape[0] == 0 || table_shape[1] == 0 ||
        output_shape[0] != index_count || output_shape[1] != table_shape[1]) {
        set_error(error, error_capacity, "embedding table and output dimensions do not match");
        return 0;
    }
    for (size_t position = 0; position < index_count; position++) {
        uint64_t token_id;
        if (!read_embedding_index(indices, position, &token_id) ||
            token_id >= table_shape[0]) {
            set_error(error, error_capacity, "embedding index is negative or outside the table");
            return 0;
        }
    }
    const size_t width = (size_t)table_shape[1];
    const float *table_data = tensor_buffer_data(table);
    float *output_data = tensor_buffer_data_mut(output);
    for (size_t position = 0; position < index_count; position++) {
        uint64_t token_id;
        read_embedding_index(indices, position, &token_id);
        memcpy(output_data + position * width, table_data + (size_t)token_id * width,
               width * sizeof(float));
    }
    return 1;
}

int tensor_embedding_lookup_f16(const TensorBuffer *table,
                                const TensorBuffer *indices,
                                TensorBuffer *output, char *error,
                                size_t error_capacity) {
    if (!valid_f16(table) || !valid_f16(output) || indices == NULL ||
        tensor_buffer_rank(table) != 2 || tensor_buffer_rank(indices) != 1 ||
        tensor_buffer_rank(output) != 2) {
        set_error(error, error_capacity, "embedding lookup requires a rank-2 F16 table, integer rank-1 indices, and rank-2 F16 output");
        return 0;
    }
    const uint64_t *ts = tensor_buffer_shape(table), *os = tensor_buffer_shape(output);
    const size_t count = tensor_buffer_element_count(indices), width = (size_t)ts[1];
    if (ts[0] == 0 || width == 0 || os[0] != count || os[1] != ts[1]) {
        set_error(error, error_capacity, "embedding table and output dimensions do not match");
        return 0;
    }
    for (size_t row = 0; row < count; row++) {
        uint64_t id;
        if (!read_embedding_index(indices, row, &id) || id >= ts[0]) {
            set_error(error, error_capacity, "embedding index is negative or outside the table");
            return 0;
        }
    }
    for (size_t row = 0; row < count; row++) {
        uint64_t id;
        read_embedding_index(indices, row, &id);
        memcpy((uint16_t *)tensor_buffer_data_mut(output) + row * width,
               (const uint16_t *)tensor_buffer_data(table) + (size_t)id * width,
               width * sizeof(uint16_t));
    }
    return 1;
}

int tensor_matmul_f32(const TensorBuffer *left, const TensorBuffer *right,
                      TensorBuffer *output, char *error,
                      size_t error_capacity) {
    if (!valid_f32(left) || !valid_f32(right) || !valid_f32(output) ||
        tensor_buffer_rank(left) != 2 || tensor_buffer_rank(right) != 2 ||
        tensor_buffer_rank(output) != 2) {
        set_error(error, error_capacity, "matmul requires rank-2 F32 tensors");
        return 0;
    }
    const uint64_t *left_shape = tensor_buffer_shape(left);
    const uint64_t *right_shape = tensor_buffer_shape(right);
    const uint64_t *output_shape = tensor_buffer_shape(output);
    if (left_shape[1] != right_shape[0] || output_shape[0] != left_shape[0] ||
        output_shape[1] != right_shape[1]) {
        set_error(error, error_capacity, "matmul tensor dimensions do not match");
        return 0;
    }
    const size_t rows = (size_t)left_shape[0];
    const size_t inner = (size_t)left_shape[1];
    const size_t columns = (size_t)right_shape[1];
    const float *left_data = tensor_buffer_data(left);
    const float *right_data = tensor_buffer_data(right);
    float *output_data = tensor_buffer_data_mut(output);
    for (size_t row = 0; row < rows; row++) {
        for (size_t column = 0; column < columns; column++) {
            float sum = 0.0f;
            for (size_t index = 0; index < inner; index++)
                sum += left_data[row * inner + index] *
                       right_data[index * columns + column];
            output_data[row * columns + column] = sum;
        }
    }
    return 1;
}

int tensor_matmul_f16(const TensorBuffer *left, const TensorBuffer *right,
                      TensorBuffer *output, char *error,
                      size_t error_capacity) {
    if (!valid_f16(left) || !valid_f16(right) || !valid_f16(output) ||
        tensor_buffer_rank(left) != 2 || tensor_buffer_rank(right) != 2 ||
        tensor_buffer_rank(output) != 2) {
        set_error(error, error_capacity, "matmul requires rank-2 F16 tensors");
        return 0;
    }
    const uint64_t *ls = tensor_buffer_shape(left), *rs = tensor_buffer_shape(right);
    const uint64_t *os = tensor_buffer_shape(output);
    if (ls[1] != rs[0] || os[0] != ls[0] || os[1] != rs[1]) {
        set_error(error, error_capacity, "matmul tensor dimensions do not match");
        return 0;
    }
    const size_t rows = (size_t)ls[0], inner = (size_t)ls[1], cols = (size_t)rs[1];
    for (size_t row = 0; row < rows; row++)
        for (size_t col = 0; col < cols; col++) {
            float sum = 0.0f;
            for (size_t k = 0; k < inner; k++)
                sum += read_f16(left, row * inner + k) * read_f16(right, k * cols + col);
            write_f16(output, row * cols + col, sum);
        }
    return 1;
}

int tensor_linear_f32(const TensorBuffer *input, const TensorBuffer *weight,
                      const TensorBuffer *bias, TensorBuffer *output,
                      char *error, size_t error_capacity) {
    if (!valid_f32(input) || !valid_f32(weight) || !valid_f32(output) ||
        tensor_buffer_rank(input) == 0 || tensor_buffer_rank(weight) != 2 ||
        tensor_buffer_rank(output) != tensor_buffer_rank(input) ||
        (bias != NULL && (!valid_f32(bias) || tensor_buffer_rank(bias) != 1))) {
        set_error(error, error_capacity, "linear requires F32 input, rank-2 weight, optional rank-1 bias, and matching output rank");
        return 0;
    }
    const size_t input_rank = tensor_buffer_rank(input);
    const uint64_t *input_shape = tensor_buffer_shape(input);
    const uint64_t *weight_shape = tensor_buffer_shape(weight);
    const uint64_t *output_shape = tensor_buffer_shape(output);
    const size_t input_width = (size_t)input_shape[input_rank - 1];
    const size_t output_width = (size_t)weight_shape[0];
    if (input_width == 0 || output_width == 0 || weight_shape[1] != input_width ||
        output_shape[input_rank - 1] != output_width ||
        (bias != NULL && tensor_buffer_shape(bias)[0] != output_width)) {
        set_error(error, error_capacity, "linear input, weight, bias, and output dimensions do not match");
        return 0;
    }
    for (size_t dimension = 0; dimension + 1 < input_rank; dimension++) {
        if (input_shape[dimension] != output_shape[dimension]) {
            set_error(error, error_capacity, "linear output leading dimensions must match input");
            return 0;
        }
    }
    const size_t rows = tensor_buffer_element_count(input) / input_width;
    const float *input_data = tensor_buffer_data(input);
    const float *weight_data = tensor_buffer_data(weight);
    const float *bias_data = bias == NULL ? NULL : tensor_buffer_data(bias);
    float *output_data = tensor_buffer_data_mut(output);
    for (size_t row = 0; row < rows; row++) {
        for (size_t output_index = 0; output_index < output_width; output_index++) {
            float sum = bias_data == NULL ? 0.0f : bias_data[output_index];
            const size_t weight_offset = output_index * input_width;
            const size_t input_offset = row * input_width;
            for (size_t input_index = 0; input_index < input_width; input_index++)
                sum += input_data[input_offset + input_index] *
                       weight_data[weight_offset + input_index];
            output_data[row * output_width + output_index] = sum;
        }
    }
    return 1;
}

int tensor_linear_f16(const TensorBuffer *input, const TensorBuffer *weight,
                      const TensorBuffer *bias, TensorBuffer *output,
                      char *error, size_t error_capacity) {
    if (!valid_f16(input) || !valid_f16(weight) || !valid_f16(output) ||
        tensor_buffer_rank(input) == 0 || tensor_buffer_rank(weight) != 2 ||
        tensor_buffer_rank(output) != tensor_buffer_rank(input) ||
        (bias != NULL && (!valid_f16(bias) || tensor_buffer_rank(bias) != 1))) {
        set_error(error, error_capacity, "linear requires F16 input, rank-2 weight, optional rank-1 bias, and matching output rank");
        return 0;
    }
    const size_t rank = tensor_buffer_rank(input);
    const uint64_t *is = tensor_buffer_shape(input), *ws = tensor_buffer_shape(weight);
    const uint64_t *os = tensor_buffer_shape(output);
    const size_t in_width = (size_t)is[rank - 1], out_width = (size_t)ws[0];
    if (in_width == 0 || out_width == 0 || ws[1] != in_width || os[rank - 1] != out_width ||
        (bias != NULL && tensor_buffer_shape(bias)[0] != out_width)) {
        set_error(error, error_capacity, "linear input, weight, bias, and output dimensions do not match");
        return 0;
    }
    for (size_t dim = 0; dim + 1 < rank; dim++)
        if (is[dim] != os[dim]) {
            set_error(error, error_capacity, "linear output leading dimensions must match input");
            return 0;
        }
    const size_t rows = tensor_buffer_element_count(input) / in_width;
    for (size_t row = 0; row < rows; row++)
        for (size_t col = 0; col < out_width; col++) {
            float sum = bias == NULL ? 0.0f : read_f16(bias, col);
            for (size_t k = 0; k < in_width; k++)
                sum += read_f16(input, row * in_width + k) * read_f16(weight, col * in_width + k);
            write_f16(output, row * out_width + col, sum);
        }
    return 1;
}

int tensor_rms_norm_f32(const TensorBuffer *input, const TensorBuffer *weight,
                        float epsilon, TensorBuffer *output, char *error,
                        size_t error_capacity) {
    if (!valid_f32(input) || !valid_f32(weight) || !valid_f32(output) ||
        tensor_buffer_rank(input) == 0 || tensor_buffer_rank(weight) != 1 ||
        !same_shape(input, output) || !isfinite(epsilon) || epsilon < 0.0f) {
        set_error(error, error_capacity, "RMSNorm requires compatible F32 tensors and nonnegative epsilon");
        return 0;
    }
    const size_t width = (size_t)tensor_buffer_shape(input)[tensor_buffer_rank(input) - 1];
    if (tensor_buffer_shape(weight)[0] != width || width == 0) {
        set_error(error, error_capacity, "RMSNorm weight width does not match input");
        return 0;
    }
    const size_t rows = tensor_buffer_element_count(input) / width;
    const float *input_data = tensor_buffer_data(input);
    const float *weight_data = tensor_buffer_data(weight);
    float *output_data = tensor_buffer_data_mut(output);
    for (size_t row = 0; row < rows; row++) {
        const size_t offset = row * width;
        float square_sum = 0.0f;
        for (size_t column = 0; column < width; column++) {
            const float value = input_data[offset + column];
            square_sum += value * value;
        }
        const float scale = 1.0f / sqrtf(square_sum / (float)width + epsilon);
        for (size_t column = 0; column < width; column++)
            output_data[offset + column] = input_data[offset + column] * scale * weight_data[column];
    }
    return 1;
}

int tensor_rms_norm_f16(const TensorBuffer *input, const TensorBuffer *weight,
                        float epsilon, TensorBuffer *output, char *error,
                        size_t error_capacity) {
    if (!valid_f16(input) || !valid_f16(weight) || !valid_f16(output) ||
        tensor_buffer_rank(input) == 0 || tensor_buffer_rank(weight) != 1 ||
        !same_shape(input, output) || !isfinite(epsilon) || epsilon < 0.0f) {
        set_error(error, error_capacity, "RMSNorm requires compatible F16 tensors and nonnegative epsilon");
        return 0;
    }
    const size_t width = (size_t)tensor_buffer_shape(input)[tensor_buffer_rank(input) - 1];
    if (width == 0 || tensor_buffer_shape(weight)[0] != width) {
        set_error(error, error_capacity, "RMSNorm weight width does not match input");
        return 0;
    }
    const size_t rows = tensor_buffer_element_count(input) / width;
    for (size_t row = 0; row < rows; row++) {
        float squares = 0.0f;
        for (size_t col = 0; col < width; col++) {
            const float x = read_f16(input, row * width + col);
            squares += x * x;
        }
        const float scale = 1.0f / sqrtf(squares / (float)width + epsilon);
        for (size_t col = 0; col < width; col++)
            write_f16(output, row * width + col,
                      read_f16(input, row * width + col) * scale * read_f16(weight, col));
    }
    return 1;
}

int tensor_layer_norm_f32(const TensorBuffer *input, const TensorBuffer *scale,
                          const TensorBuffer *bias, float epsilon,
                          TensorBuffer *output, char *error,
                          size_t error_capacity) {
    if (!valid_f32(input) || !valid_f32(output) ||
        tensor_buffer_rank(input) == 0 || !same_shape(input, output) ||
        !isfinite(epsilon) || epsilon < 0.0f ||
        (scale != NULL && (!valid_f32(scale) || tensor_buffer_rank(scale) != 1)) ||
        (bias != NULL && (!valid_f32(bias) || tensor_buffer_rank(bias) != 1))) {
        set_error(error, error_capacity, "LayerNorm requires compatible F32 tensors and nonnegative epsilon");
        return 0;
    }
    const size_t width = (size_t)tensor_buffer_shape(input)[tensor_buffer_rank(input) - 1];
    if (width == 0 ||
        (scale != NULL && tensor_buffer_shape(scale)[0] != width) ||
        (bias != NULL && tensor_buffer_shape(bias)[0] != width)) {
        set_error(error, error_capacity, "LayerNorm affine parameter width does not match input");
        return 0;
    }
    const size_t rows = tensor_buffer_element_count(input) / width;
    const float *input_data = tensor_buffer_data(input);
    const float *scale_data = scale == NULL ? NULL : tensor_buffer_data(scale);
    const float *bias_data = bias == NULL ? NULL : tensor_buffer_data(bias);
    float *output_data = tensor_buffer_data_mut(output);
    for (size_t row = 0; row < rows; row++) {
        const size_t offset = row * width;
        float mean = 0.0f;
        for (size_t column = 0; column < width; column++)
            mean += input_data[offset + column];
        mean /= (float)width;
        float variance = 0.0f;
        for (size_t column = 0; column < width; column++) {
            const float difference = input_data[offset + column] - mean;
            variance += difference * difference;
        }
        const float inverse_std = 1.0f / sqrtf(variance / (float)width + epsilon);
        for (size_t column = 0; column < width; column++) {
            float value = (input_data[offset + column] - mean) * inverse_std;
            if (scale_data != NULL) value *= scale_data[column];
            if (bias_data != NULL) value += bias_data[column];
            output_data[offset + column] = value;
        }
    }
    return 1;
}

int tensor_layer_norm_f16(const TensorBuffer *input, const TensorBuffer *scale,
                          const TensorBuffer *bias, float epsilon,
                          TensorBuffer *output, char *error,
                          size_t error_capacity) {
    if (!valid_f16(input) || !valid_f16(output) || tensor_buffer_rank(input) == 0 ||
        !same_shape(input, output) || !isfinite(epsilon) || epsilon < 0.0f ||
        (scale != NULL && (!valid_f16(scale) || tensor_buffer_rank(scale) != 1)) ||
        (bias != NULL && (!valid_f16(bias) || tensor_buffer_rank(bias) != 1))) {
        set_error(error, error_capacity, "LayerNorm requires compatible F16 tensors and nonnegative epsilon");
        return 0;
    }
    const size_t width = (size_t)tensor_buffer_shape(input)[tensor_buffer_rank(input) - 1];
    if (width == 0 || (scale != NULL && tensor_buffer_shape(scale)[0] != width) ||
        (bias != NULL && tensor_buffer_shape(bias)[0] != width)) {
        set_error(error, error_capacity, "LayerNorm affine parameter width does not match input");
        return 0;
    }
    const size_t rows = tensor_buffer_element_count(input) / width;
    for (size_t row = 0; row < rows; row++) {
        float mean = 0.0f;
        for (size_t col = 0; col < width; col++) mean += read_f16(input, row * width + col);
        mean /= (float)width;
        float variance = 0.0f;
        for (size_t col = 0; col < width; col++) {
            const float delta = read_f16(input, row * width + col) - mean;
            variance += delta * delta;
        }
        const float inverse_std = 1.0f / sqrtf(variance / (float)width + epsilon);
        for (size_t col = 0; col < width; col++) {
            float value = (read_f16(input, row * width + col) - mean) * inverse_std;
            if (scale != NULL) value *= read_f16(scale, col);
            if (bias != NULL) value += read_f16(bias, col);
            write_f16(output, row * width + col, value);
        }
    }
    return 1;
}

int tensor_silu_f32(const TensorBuffer *input, TensorBuffer *output,
                    char *error, size_t error_capacity) {
    if (!valid_f32(input) || !valid_f32(output) || !same_shape(input, output)) {
        set_error(error, error_capacity, "SiLU requires same-shaped F32 tensors");
        return 0;
    }
    const size_t count = tensor_buffer_element_count(input);
    const float *input_data = tensor_buffer_data(input);
    float *output_data = tensor_buffer_data_mut(output);
    for (size_t index = 0; index < count; index++) {
        const float value = input_data[index];
        output_data[index] = value / (1.0f + expf(-value));
    }
    return 1;
}

int tensor_silu_f16(const TensorBuffer *input, TensorBuffer *output,
                    char *error, size_t error_capacity) {
    if (!valid_f16(input) || !valid_f16(output) || !same_shape(input, output)) {
        set_error(error, error_capacity, "SiLU requires same-shaped F16 tensors");
        return 0;
    }
    const size_t count = tensor_buffer_element_count(input);
    for (size_t index = 0; index < count; index++) {
        const float value = read_f16(input, index);
        write_f16(output, index, value / (1.0f + expf(-value)));
    }
    return 1;
}
