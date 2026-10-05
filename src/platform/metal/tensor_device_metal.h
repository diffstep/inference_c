#ifndef TENSOR_DEVICE_METAL_INTERNAL_H
#define TENSOR_DEVICE_METAL_INTERNAL_H

#import <Metal/Metal.h>

id<MTLDevice> tensor_device_metal_shared_device(void);
id<MTLCommandQueue> tensor_device_metal_shared_queue(void);

#endif
