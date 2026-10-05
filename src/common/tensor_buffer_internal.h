#ifndef TENSOR_BUFFER_INTERNAL_H
#define TENSOR_BUFFER_INTERNAL_H

#include "tensor_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif
void *tensor_buffer_device_handle_internal(const TensorBuffer *buffer);
#ifdef __cplusplus
}
#endif

#endif
