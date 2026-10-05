#include "tensor_device.h"
#include "tensor_device_internal.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

struct TensorDeviceBuffer {
    void *data;
    size_t byte_length;
};

namespace {
void set_error(char *error, size_t capacity, const char *message) {
    if (error != nullptr && capacity > 0) std::snprintf(error, capacity, "%s", message);
}

bool valid_range(size_t total, size_t offset, size_t length) {
    return offset <= total && length <= total - offset;
}

int cuda_error(cudaError_t status, const char *operation,
               char *error, size_t capacity) {
    if (status == cudaSuccess) return 1;
    if (error != nullptr && capacity > 0)
        std::snprintf(error, capacity, "%s: %s", operation,
                      cudaGetErrorString(status));
    return 0;
}
}

extern "C" int tensor_device_available(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

extern "C" int tensor_device_buffer_create(size_t byte_length,
                                            TensorDeviceBuffer **result,
                                            char *error, size_t error_capacity) {
    if (result != nullptr) *result = nullptr;
    if (result == nullptr || byte_length == 0) {
        set_error(error, error_capacity, "device buffer size must be nonzero");
        return 0;
    }
    auto *buffer = static_cast<TensorDeviceBuffer *>(
        std::calloc(1, sizeof(TensorDeviceBuffer)));
    if (buffer == nullptr) {
        set_error(error, error_capacity, "out of memory creating device buffer");
        return 0;
    }
    const cudaError_t status = cudaMalloc(&buffer->data, byte_length);
    if (!cuda_error(status, "cudaMalloc", error, error_capacity)) {
        std::free(buffer);
        return 0;
    }
    buffer->byte_length = byte_length;
    *result = buffer;
    return 1;
}

extern "C" void tensor_device_buffer_free(TensorDeviceBuffer *buffer) {
    if (buffer == nullptr) return;
    if (buffer->data != nullptr) cudaFree(buffer->data);
    std::free(buffer);
}

extern "C" size_t tensor_device_buffer_size(const TensorDeviceBuffer *buffer) {
    return buffer == nullptr ? 0 : buffer->byte_length;
}

extern "C" void *tensor_device_buffer_native_handle(const TensorDeviceBuffer *buffer) {
    return buffer == nullptr ? nullptr : buffer->data;
}

extern "C" int tensor_device_buffer_upload(TensorDeviceBuffer *destination,
                                            size_t destination_offset,
                                            const void *source,
                                            size_t byte_length,
                                            char *error, size_t error_capacity) {
    if (destination == nullptr || (byte_length > 0 && source == nullptr) ||
        !valid_range(destination->byte_length, destination_offset, byte_length)) {
        set_error(error, error_capacity, "invalid device upload range or pointer");
        return 0;
    }
    return cuda_error(cudaMemcpy(static_cast<char *>(destination->data) + destination_offset,
                                 source, byte_length, cudaMemcpyHostToDevice),
                      "cudaMemcpy host-to-device", error, error_capacity);
}

extern "C" int tensor_device_buffer_download(const TensorDeviceBuffer *source,
                                              size_t source_offset,
                                              void *destination,
                                              size_t byte_length,
                                              char *error,
                                              size_t error_capacity) {
    if (source == nullptr || (byte_length > 0 && destination == nullptr) ||
        !valid_range(source->byte_length, source_offset, byte_length)) {
        set_error(error, error_capacity, "invalid device download range or pointer");
        return 0;
    }
    return cuda_error(cudaMemcpy(destination,
                                 static_cast<const char *>(source->data) + source_offset,
                                 byte_length, cudaMemcpyDeviceToHost),
                      "cudaMemcpy device-to-host", error, error_capacity);
}

extern "C" int tensor_device_buffer_copy(TensorDeviceBuffer *destination,
                                          size_t destination_offset,
                                          const TensorDeviceBuffer *source,
                                          size_t source_offset,
                                          size_t byte_length,
                                          char *error, size_t error_capacity) {
    if (destination == nullptr || source == nullptr ||
        !valid_range(destination->byte_length, destination_offset, byte_length) ||
        !valid_range(source->byte_length, source_offset, byte_length)) {
        set_error(error, error_capacity, "invalid device-to-device copy range");
        return 0;
    }
    return cuda_error(cudaMemcpy(static_cast<char *>(destination->data) + destination_offset,
                                 static_cast<const char *>(source->data) + source_offset,
                                 byte_length, cudaMemcpyDeviceToDevice),
                      "cudaMemcpy device-to-device", error, error_capacity);
}

extern "C" int tensor_device_synchronize(char *error, size_t error_capacity) {
    return cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize",
                      error, error_capacity);
}
