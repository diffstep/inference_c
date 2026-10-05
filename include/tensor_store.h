#ifndef TENSOR_STORE_H
#define TENSOR_STORE_H

#include "safetensors.h"

#include <stddef.h>

typedef struct TensorStore TensorStore;

int tensor_store_open_file(const char *safetensors_path, TensorStore **result,
                           char *error, size_t error_capacity);
int tensor_store_open_index(const char *index_path, const char *shard_directory,
                            TensorStore **result, char *error,
                            size_t error_capacity);
void tensor_store_close(TensorStore *store);
size_t tensor_store_count(const TensorStore *store);
const SafeTensorInfo *tensor_store_find(const TensorStore *store,
                                        const char *tensor_name);
int tensor_store_read_raw(const TensorStore *store, const char *tensor_name,
                          void *destination, size_t capacity,
                          char *error, size_t error_capacity);
int tensor_store_read_f16(const TensorStore *store, const char *tensor_name,
                          uint16_t *destination, size_t capacity,
                          char *error, size_t error_capacity);
int tensor_store_read_f32(const TensorStore *store, const char *tensor_name,
                          float *destination, size_t capacity,
                          char *error, size_t error_capacity);

#endif
