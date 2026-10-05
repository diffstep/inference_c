#include "tensor_buffer.h"
#include "tensor_device.h"
#include "tensor_device_internal.h"
#include "tensor_workspace.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TENSOR_BUFFER_MAX_RANK 16u

struct TensorBuffer {
    TensorDType dtype;
    size_t rank;
    uint64_t *shape;
    size_t element_count;
    size_t byte_length;
    void *data;
    TensorDeviceBuffer *device_buffer;
    int owns_data;
    TensorBufferStorage storage;
};

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float bits_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float bf16_to_float(uint16_t value) {
    return bits_float((uint32_t)value << 16);
}

static uint16_t float_to_bf16(float value) {
    uint32_t bits = float_bits(value);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) {
        return (uint16_t)((bits >> 16) | 0x0040u);
    }
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float f16_to_float(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t fraction = value & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            int adjusted_exponent = -14;
            while ((fraction & 0x0400u) == 0) {
                fraction <<= 1;
                adjusted_exponent--;
            }
            fraction &= 0x03ffu;
            bits = sign | ((uint32_t)(adjusted_exponent + 127) << 23) |
                   (fraction << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (fraction << 13);
    } else {
        exponent = exponent - 15 + 127;
        bits = sign | (exponent << 23) | (fraction << 13);
    }
    return bits_float(bits);
}

static uint16_t float_to_f16(float value) {
    const uint32_t bits = float_bits(value);
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t exponent_bits = (bits >> 23) & 0xffu;
    const uint32_t fraction = bits & 0x007fffffu;
    if (exponent_bits == 0xffu) {
        if (fraction == 0) return (uint16_t)(sign | 0x7c00u);
        uint16_t payload = (uint16_t)(fraction >> 13);
        if (payload == 0) payload = 1;
        return (uint16_t)(sign | 0x7c00u | payload);
    }
    int exponent = (int)exponent_bits - 127 + 15;
    if (exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        const uint32_t significand = fraction | 0x00800000u;
        const unsigned shift = (unsigned)(14 - exponent);
        uint32_t rounded = significand >> shift;
        const uint32_t remainder = significand & ((1u << shift) - 1u);
        const uint32_t midpoint = 1u << (shift - 1u);
        if (remainder > midpoint || (remainder == midpoint && (rounded & 1u))) rounded++;
        return (uint16_t)(sign | (uint16_t)rounded);
    }
    uint32_t rounded_fraction = fraction >> 13;
    const uint32_t remainder = fraction & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded_fraction & 1u))) {
        rounded_fraction++;
        if (rounded_fraction == 0x400u) {
            rounded_fraction = 0;
            exponent++;
            if (exponent >= 31) return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint16_t)exponent << 10) |
                      (uint16_t)rounded_fraction);
}

static int supported_float_dtype(TensorDType dtype) {
    return dtype == TENSOR_DTYPE_F32 || dtype == TENSOR_DTYPE_F16 ||
           dtype == TENSOR_DTYPE_BF16;
}

static float read_float(const TensorBuffer *buffer, size_t index) {
    if (buffer->dtype == TENSOR_DTYPE_F32) {
        return ((const float *)buffer->data)[index];
    }
    if (buffer->dtype == TENSOR_DTYPE_F16) {
        return f16_to_float(((const uint16_t *)buffer->data)[index]);
    }
    return bf16_to_float(((const uint16_t *)buffer->data)[index]);
}

static void write_float(TensorBuffer *buffer, size_t index, float value) {
    if (buffer->dtype == TENSOR_DTYPE_F32) {
        ((float *)buffer->data)[index] = value;
    } else if (buffer->dtype == TENSOR_DTYPE_F16) {
        ((uint16_t *)buffer->data)[index] = float_to_f16(value);
    } else {
        ((uint16_t *)buffer->data)[index] = float_to_bf16(value);
    }
}

static int tensor_buffer_create_internal(TensorDType dtype,
                                         const uint64_t *shape, size_t rank,
                                         TensorWorkspace *workspace,
                                         int host_storage,
                                         TensorBuffer **result, char *error,
                                         size_t error_capacity) {
    if (result != NULL) *result = NULL;
    const size_t element_size = tensor_dtype_size(dtype);
    if (result == NULL || element_size == 0 || rank > TENSOR_BUFFER_MAX_RANK ||
        (rank > 0 && shape == NULL)) {
        set_error(error, error_capacity, "invalid tensor buffer dtype, rank, or shape");
        return 0;
    }
    size_t elements = 1;
    for (size_t index = 0; index < rank; index++) {
        if (shape[index] > SIZE_MAX ||
            (shape[index] != 0 && elements > SIZE_MAX / (size_t)shape[index])) {
            set_error(error, error_capacity, "tensor shape exceeds addressable size");
            return 0;
        }
        elements *= (size_t)shape[index];
    }
    if (elements > SIZE_MAX / element_size) {
        set_error(error, error_capacity, "tensor byte length exceeds addressable size");
        return 0;
    }
    TensorBuffer *buffer = calloc(1, sizeof(*buffer));
    if (buffer == NULL) {
        set_error(error, error_capacity, "out of memory creating tensor buffer");
        return 0;
    }
    if (rank > 0) {
        buffer->shape = malloc(rank * sizeof(*buffer->shape));
        if (buffer->shape == NULL) {
            free(buffer);
            set_error(error, error_capacity, "out of memory copying tensor shape");
            return 0;
        }
        memcpy(buffer->shape, shape, rank * sizeof(*shape));
    }
    const size_t bytes = elements * element_size;
    if (host_storage) {
        buffer->data = workspace == NULL ? calloc(bytes == 0 ? 1 : bytes, 1) :
            tensor_workspace_allocate(workspace, bytes, _Alignof(max_align_t),
                                      error, error_capacity);
    }
    if (host_storage && buffer->data == NULL) {
        free(buffer->shape);
        free(buffer);
        set_error(error, error_capacity, "out of memory allocating tensor data");
        return 0;
    }
    buffer->dtype = dtype;
    buffer->rank = rank;
    buffer->element_count = elements;
    buffer->byte_length = bytes;
    buffer->owns_data = workspace == NULL && host_storage;
    buffer->storage = host_storage ? TENSOR_BUFFER_STORAGE_HOST :
                                     TENSOR_BUFFER_STORAGE_DEVICE;
    *result = buffer;
    return 1;
}

int tensor_buffer_create(TensorDType dtype, const uint64_t *shape, size_t rank,
                         TensorBuffer **result, char *error,
                         size_t error_capacity) {
    return tensor_buffer_create_internal(dtype, shape, rank, NULL, 1, result,
                                         error, error_capacity);
}

int tensor_buffer_create_device(TensorDType dtype, const uint64_t *shape,
                                size_t rank, TensorBuffer **result,
                                char *error, size_t error_capacity) {
    if (result != NULL) *result = NULL;
    if (result == NULL) {
        set_error(error, error_capacity, "device tensor result pointer is required");
        return 0;
    }
    TensorBuffer *buffer = NULL;
    if (!tensor_buffer_create_internal(dtype, shape, rank, NULL, 0, &buffer,
                                       error, error_capacity)) return 0;
    TensorDeviceBuffer *device_buffer = NULL;
    if (buffer->byte_length > 0 &&
        !tensor_device_buffer_create(buffer->byte_length, &device_buffer,
                                     error, error_capacity)) {
        tensor_buffer_free(buffer);
        return 0;
    }
    buffer->device_buffer = device_buffer;
    *result = buffer;
    return 1;
}

int tensor_buffer_create_in_workspace(TensorDType dtype, const uint64_t *shape,
                                      size_t rank, TensorWorkspace *workspace,
                                      TensorBuffer **result, char *error,
                                      size_t error_capacity) {
    if (workspace == NULL) {
        if (result != NULL) *result = NULL;
        set_error(error, error_capacity, "workspace-backed tensor requires a workspace");
        return 0;
    }
    return tensor_buffer_create_internal(dtype, shape, rank, workspace, 1,
                                         result, error, error_capacity);
}

int tensor_buffer_load(const TensorStore *store, const char *tensor_name,
                       TensorBuffer **result, char *error,
                       size_t error_capacity) {
    if (result != NULL) *result = NULL;
    const SafeTensorInfo *info = tensor_store_find(store, tensor_name);
    if (info == NULL || result == NULL) {
        set_error(error, error_capacity, "tensor metadata was not found");
        return 0;
    }
    TensorBuffer *buffer = NULL;
    if (!tensor_buffer_create(info->dtype, info->shape, info->rank, &buffer,
                              error, error_capacity)) {
        return 0;
    }
    if (buffer->byte_length != info->byte_length ||
        !tensor_store_read_raw(store, tensor_name, buffer->data,
                               buffer->byte_length, error, error_capacity)) {
        tensor_buffer_free(buffer);
        if (error != NULL && error_capacity > 0 && error[0] == '\0')
            set_error(error, error_capacity, "tensor metadata byte length mismatch");
        return 0;
    }
    *result = buffer;
    return 1;
}

int tensor_buffer_convert(const TensorBuffer *source, TensorDType target_dtype,
                          TensorBuffer **result, char *error,
                          size_t error_capacity) {
    if (result != NULL) *result = NULL;
    if (source == NULL || result == NULL || !supported_float_dtype(source->dtype) ||
        source->storage != TENSOR_BUFFER_STORAGE_HOST ||
        !supported_float_dtype(target_dtype)) {
        set_error(error, error_capacity, "conversion supports F32, F16, and BF16 tensors");
        return 0;
    }
    TensorBuffer *destination = NULL;
    if (!tensor_buffer_create(target_dtype, source->shape, source->rank,
                              &destination, error, error_capacity)) {
        return 0;
    }
    if (source->dtype == target_dtype) {
        memcpy(destination->data, source->data, source->byte_length);
    } else {
        for (size_t index = 0; index < source->element_count; index++) {
            write_float(destination, index, read_float(source, index));
        }
    }
    *result = destination;
    return 1;
}

void tensor_buffer_free(TensorBuffer *buffer) {
    if (buffer == NULL) return;
    if (buffer->owns_data) free(buffer->data);
    tensor_device_buffer_free(buffer->device_buffer);
    free(buffer->shape);
    free(buffer);
}

TensorDType tensor_buffer_dtype(const TensorBuffer *buffer) {
    return buffer == NULL ? TENSOR_DTYPE_UNKNOWN : buffer->dtype;
}

size_t tensor_buffer_rank(const TensorBuffer *buffer) {
    return buffer == NULL ? 0 : buffer->rank;
}

const uint64_t *tensor_buffer_shape(const TensorBuffer *buffer) {
    return buffer == NULL ? NULL : buffer->shape;
}

size_t tensor_buffer_element_count(const TensorBuffer *buffer) {
    return buffer == NULL ? 0 : buffer->element_count;
}

size_t tensor_buffer_byte_length(const TensorBuffer *buffer) {
    return buffer == NULL ? 0 : buffer->byte_length;
}

void *tensor_buffer_device_handle_internal(const TensorBuffer *buffer) {
    if (buffer == NULL || buffer->storage != TENSOR_BUFFER_STORAGE_DEVICE)
        return NULL;
    return tensor_device_buffer_native_handle(buffer->device_buffer);
}

TensorBufferStorage tensor_buffer_storage(const TensorBuffer *buffer) {
    return buffer == NULL ? TENSOR_BUFFER_STORAGE_HOST : buffer->storage;
}

static int valid_buffer_range(const TensorBuffer *buffer, size_t offset,
                              size_t length) {
    return buffer != NULL && offset <= buffer->byte_length &&
           length <= buffer->byte_length - offset;
}

int tensor_buffer_upload(TensorBuffer *buffer, size_t byte_offset,
                         const void *source, size_t byte_length,
                         char *error, size_t error_capacity) {
    if (!valid_buffer_range(buffer, byte_offset, byte_length) ||
        (byte_length > 0 && source == NULL)) {
        set_error(error, error_capacity, "invalid tensor upload range or pointer");
        return 0;
    }
    if (byte_length == 0) return 1;
    if (buffer->storage == TENSOR_BUFFER_STORAGE_DEVICE)
        return tensor_device_buffer_upload(buffer->device_buffer, byte_offset,
                                          source, byte_length, error,
                                          error_capacity);
    if (byte_length > 0) memcpy((unsigned char *)buffer->data + byte_offset,
                                source, byte_length);
    return 1;
}

int tensor_buffer_download(const TensorBuffer *buffer, size_t byte_offset,
                           void *destination, size_t byte_length,
                           char *error, size_t error_capacity) {
    if (!valid_buffer_range(buffer, byte_offset, byte_length) ||
        (byte_length > 0 && destination == NULL)) {
        set_error(error, error_capacity, "invalid tensor download range or pointer");
        return 0;
    }
    if (byte_length == 0) return 1;
    if (buffer->storage == TENSOR_BUFFER_STORAGE_DEVICE)
        return tensor_device_buffer_download(buffer->device_buffer, byte_offset,
                                            destination, byte_length, error,
                                            error_capacity);
    if (byte_length > 0) memcpy(destination,
                                (const unsigned char *)buffer->data + byte_offset,
                                byte_length);
    return 1;
}

int tensor_buffer_copy_device(TensorBuffer *destination,
                              size_t destination_offset,
                              const TensorBuffer *source, size_t source_offset,
                              size_t byte_length, char *error,
                              size_t error_capacity) {
    if (!valid_buffer_range(destination, destination_offset, byte_length) ||
        !valid_buffer_range(source, source_offset, byte_length) ||
        destination->storage != TENSOR_BUFFER_STORAGE_DEVICE ||
        source->storage != TENSOR_BUFFER_STORAGE_DEVICE) {
        set_error(error, error_capacity,
                  "tensor device copy requires valid device-resident ranges");
        return 0;
    }
    if (byte_length == 0) return 1;
    return tensor_device_buffer_copy(destination->device_buffer,
                                     destination_offset, source->device_buffer,
                                     source_offset, byte_length, error,
                                     error_capacity);
}

const void *tensor_buffer_data(const TensorBuffer *buffer) {
    return buffer == NULL || buffer->storage != TENSOR_BUFFER_STORAGE_HOST ?
           NULL : buffer->data;
}

void *tensor_buffer_data_mut(TensorBuffer *buffer) {
    return buffer == NULL || buffer->storage != TENSOR_BUFFER_STORAGE_HOST ?
           NULL : buffer->data;
}
