#include "tensor_device.h"

#include <stdio.h>
#include <stdlib.h>

struct TensorDeviceBuffer {
    size_t byte_length;
};

static void set_error(char *error, size_t capacity) {
    if (error != NULL && capacity > 0)
        snprintf(error, capacity, "no GPU device backend was configured");
}

int tensor_device_available(void) { return 0; }

int tensor_device_buffer_create(size_t byte_length, TensorDeviceBuffer **result,
                                char *error, size_t error_capacity) {
    if (result != NULL) *result = NULL;
    (void)byte_length;
    set_error(error, error_capacity);
    return 0;
}

void tensor_device_buffer_free(TensorDeviceBuffer *buffer) { free(buffer); }
size_t tensor_device_buffer_size(const TensorDeviceBuffer *buffer) {
    (void)buffer;
    return 0;
}
void *tensor_device_buffer_native_handle(const TensorDeviceBuffer *buffer) {
    (void)buffer;
    return NULL;
}
int tensor_device_buffer_upload(TensorDeviceBuffer *destination, size_t offset,
                                const void *source, size_t length,
                                char *error, size_t capacity) {
    (void)destination; (void)offset; (void)source; (void)length;
    set_error(error, capacity); return 0;
}
int tensor_device_buffer_download(const TensorDeviceBuffer *source, size_t offset,
                                  void *destination, size_t length,
                                  char *error, size_t capacity) {
    (void)source; (void)offset; (void)destination; (void)length;
    set_error(error, capacity); return 0;
}
int tensor_device_buffer_copy(TensorDeviceBuffer *destination, size_t destination_offset,
                              const TensorDeviceBuffer *source, size_t source_offset,
                              size_t length, char *error, size_t capacity) {
    (void)destination; (void)destination_offset; (void)source; (void)source_offset;
    (void)length; set_error(error, capacity); return 0;
}
int tensor_device_synchronize(char *error, size_t capacity) {
    set_error(error, capacity); return 0;
}
