#ifndef TENSOR_BUFFER_H
#define TENSOR_BUFFER_H

#include "safetensors.h"
#include "tensor_store.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TensorBuffer TensorBuffer;
typedef struct TensorWorkspace TensorWorkspace;

typedef enum {
    TENSOR_BUFFER_STORAGE_HOST,
    TENSOR_BUFFER_STORAGE_DEVICE
} TensorBufferStorage;

int tensor_buffer_create(TensorDType dtype, const uint64_t *shape, size_t rank,
                         TensorBuffer **result, char *error,
                         size_t error_capacity);
int tensor_buffer_create_device(TensorDType dtype, const uint64_t *shape,
                                size_t rank, TensorBuffer **result,
                                char *error, size_t error_capacity);
int tensor_buffer_create_in_workspace(TensorDType dtype, const uint64_t *shape,
                                      size_t rank, TensorWorkspace *workspace,
                                      TensorBuffer **result, char *error,
                                      size_t error_capacity);
int tensor_buffer_load(const TensorStore *store, const char *tensor_name,
                       TensorBuffer **result, char *error,
                       size_t error_capacity);
int tensor_buffer_convert(const TensorBuffer *source, TensorDType target_dtype,
                          TensorBuffer **result, char *error,
                          size_t error_capacity);
void tensor_buffer_free(TensorBuffer *buffer);
TensorDType tensor_buffer_dtype(const TensorBuffer *buffer);
size_t tensor_buffer_rank(const TensorBuffer *buffer);
const uint64_t *tensor_buffer_shape(const TensorBuffer *buffer);
size_t tensor_buffer_element_count(const TensorBuffer *buffer);
size_t tensor_buffer_byte_length(const TensorBuffer *buffer);
TensorBufferStorage tensor_buffer_storage(const TensorBuffer *buffer);
int tensor_buffer_upload(TensorBuffer *buffer, size_t byte_offset,
                         const void *source, size_t byte_length,
                         char *error, size_t error_capacity);
int tensor_buffer_download(const TensorBuffer *buffer, size_t byte_offset,
                           void *destination, size_t byte_length,
                           char *error, size_t error_capacity);
/* Enqueues a device-to-device copy; synchronize before dependent host access. */
int tensor_buffer_copy_device(TensorBuffer *destination,
                              size_t destination_offset,
                              const TensorBuffer *source, size_t source_offset,
                              size_t byte_length, char *error,
                              size_t error_capacity);
const void *tensor_buffer_data(const TensorBuffer *buffer);
void *tensor_buffer_data_mut(TensorBuffer *buffer);

#ifdef __cplusplus
}
#endif

#endif
