#ifndef TENSOR_DEVICE_INTERNAL_H
#define TENSOR_DEVICE_INTERNAL_H

#include "tensor_device.h"

#ifdef __cplusplus
extern "C" {
#endif
void *tensor_device_buffer_native_handle(const TensorDeviceBuffer *buffer);
#ifdef __cplusplus
}
#endif

#endif
