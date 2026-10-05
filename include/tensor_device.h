#ifndef TENSOR_DEVICE_H
#define TENSOR_DEVICE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TensorDeviceBuffer TensorDeviceBuffer;

/* Device buffers are allocated on the configured GPU backend. */
int tensor_device_available(void);
int tensor_device_buffer_create(size_t byte_length,
                                TensorDeviceBuffer **result,
                                char *error, size_t error_capacity);
void tensor_device_buffer_free(TensorDeviceBuffer *buffer);
size_t tensor_device_buffer_size(const TensorDeviceBuffer *buffer);

/* Uploads consume the host source before returning but enqueue device work.
   Downloads wait for results. Device-to-device copies enqueue work. Call
   synchronize before requiring queued work to be complete or reading results
   on the host. */
int tensor_device_buffer_upload(TensorDeviceBuffer *destination,
                                size_t destination_offset,
                                const void *source, size_t byte_length,
                                char *error, size_t error_capacity);
int tensor_device_buffer_download(const TensorDeviceBuffer *source,
                                  size_t source_offset, void *destination,
                                  size_t byte_length, char *error,
                                  size_t error_capacity);
int tensor_device_buffer_copy(TensorDeviceBuffer *destination,
                              size_t destination_offset,
                              const TensorDeviceBuffer *source,
                              size_t source_offset, size_t byte_length,
                              char *error, size_t error_capacity);
int tensor_device_synchronize(char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
