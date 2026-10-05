#ifndef TENSOR_PREFETCH_H
#define TENSOR_PREFETCH_H

#include "tensor_buffer.h"
#include "tensor_store.h"

#include <stddef.h>

typedef struct TensorPrefetch TensorPrefetch;

int tensor_prefetch_create(const TensorStore *store, TensorPrefetch **result,
                           char *error, size_t error_capacity);
int tensor_prefetch_request(TensorPrefetch *prefetch, const char *tensor_name,
                            char *error, size_t error_capacity);
int tensor_prefetch_wait(TensorPrefetch *prefetch, TensorBuffer **result,
                         char *error, size_t error_capacity);
void tensor_prefetch_free(TensorPrefetch *prefetch);

#endif
