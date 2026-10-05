#ifndef TENSOR_BUFFER_H
#define TENSOR_BUFFER_H

#include "safetensors.h"
#include "tensor_store.h"

#include <stddef.h>
#include <stdint.h>

typedef struct TensorBuffer TensorBuffer;
typedef struct TensorWorkspace TensorWorkspace;

int tensor_buffer_create(TensorDType dtype, const uint64_t *shape, size_t rank,
                         TensorBuffer **result, char *error,
                         size_t error_capacity);
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
const void *tensor_buffer_data(const TensorBuffer *buffer);
void *tensor_buffer_data_mut(TensorBuffer *buffer);

#endif
