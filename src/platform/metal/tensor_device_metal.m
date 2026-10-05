#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "tensor_device.h"
#include "tensor_device_metal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct TensorDeviceBuffer {
    void *native_buffer;
    size_t byte_length;
};

id<MTLDevice> tensor_device_metal_shared_device(void) {
    static id<MTLDevice> device;
    static dispatch_once_t once_token;
    dispatch_once(&once_token, ^{ device = MTLCreateSystemDefaultDevice(); });
    return device;
}

id<MTLCommandQueue> tensor_device_metal_shared_queue(void) {
    static id<MTLCommandQueue> queue;
    static dispatch_once_t once_token;
    dispatch_once(&once_token, ^{
        id<MTLDevice> device = tensor_device_metal_shared_device();
        if (device != nil) queue = [device newCommandQueue];
    });
    return queue;
}

static void tensor_device_set_error(char *error, size_t capacity,
                                    const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static int tensor_device_valid_range(size_t total, size_t offset, size_t length) {
    return offset <= total && length <= total - offset;
}

static id<MTLBuffer> tensor_device_native_buffer(const TensorDeviceBuffer *buffer) {
    return buffer == NULL ? nil : (__bridge id<MTLBuffer>)buffer->native_buffer;
}

static int tensor_device_metal_copy(id<MTLBuffer> source, size_t source_offset,
                                   id<MTLBuffer> destination,
                                   size_t destination_offset, size_t length,
                                   int wait_until_completed,
                                   char *error, size_t error_capacity) {
    if (length == 0) return 1;
    id<MTLCommandQueue> queue = tensor_device_metal_shared_queue();
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (command == nil || blit == nil) {
        tensor_device_set_error(error, error_capacity, "Metal blit command creation failed");
        return 0;
    }
    [blit copyFromBuffer:source sourceOffset:source_offset
                toBuffer:destination destinationOffset:destination_offset
                    size:length];
    [blit endEncoding];
    [command commit];
    if (wait_until_completed) [command waitUntilCompleted];
    if (wait_until_completed && command.status != MTLCommandBufferStatusCompleted) {
        tensor_device_set_error(error, error_capacity, "Metal buffer copy failed");
        return 0;
    }
    return 1;
}

int tensor_device_available(void) {
    return tensor_device_metal_shared_device() != nil && tensor_device_metal_shared_queue() != nil;
}

int tensor_device_buffer_create(size_t byte_length, TensorDeviceBuffer **result,
                                char *error, size_t error_capacity) {
    if (result != NULL) *result = NULL;
    if (result == NULL || byte_length == 0) {
        tensor_device_set_error(error, error_capacity, "device buffer size must be nonzero");
        return 0;
    }
    id<MTLDevice> device = tensor_device_metal_shared_device();
    id<MTLBuffer> native = [device newBufferWithLength:byte_length
                                               options:MTLResourceStorageModePrivate];
    if (native == nil) {
        tensor_device_set_error(error, error_capacity, "Metal device buffer allocation failed");
        return 0;
    }
    TensorDeviceBuffer *buffer = calloc(1, sizeof(*buffer));
    if (buffer == NULL) {
        tensor_device_set_error(error, error_capacity, "out of memory creating device buffer");
        return 0;
    }
    buffer->native_buffer = (__bridge_retained void *)native;
    buffer->byte_length = byte_length;
    *result = buffer;
    return 1;
}

void tensor_device_buffer_free(TensorDeviceBuffer *buffer) {
    if (buffer == NULL) return;
    if (buffer->native_buffer != NULL) CFBridgingRelease(buffer->native_buffer);
    free(buffer);
}

size_t tensor_device_buffer_size(const TensorDeviceBuffer *buffer) {
    return buffer == NULL ? 0 : buffer->byte_length;
}

void *tensor_device_buffer_native_handle(const TensorDeviceBuffer *buffer) {
    return buffer == NULL ? NULL : buffer->native_buffer;
}

int tensor_device_buffer_upload(TensorDeviceBuffer *destination,
                                size_t destination_offset, const void *source,
                                size_t byte_length, char *error,
                                size_t error_capacity) {
    if (destination == NULL || (byte_length > 0 && source == NULL) ||
        !tensor_device_valid_range(destination->byte_length, destination_offset,
                                   byte_length)) {
        tensor_device_set_error(error, error_capacity, "invalid device upload range or pointer");
        return 0;
    }
    if (byte_length == 0) return 1;
    id<MTLBuffer> staging = [tensor_device_metal_shared_device()
        newBufferWithLength:byte_length options:MTLResourceStorageModeShared];
    if (staging == nil) {
        tensor_device_set_error(error, error_capacity, "Metal upload staging allocation failed");
        return 0;
    }
    memcpy(staging.contents, source, byte_length);
    return tensor_device_metal_copy(staging, 0, tensor_device_native_buffer(destination),
                                    destination_offset, byte_length, 0, error,
                                    error_capacity);
}

int tensor_device_buffer_download(const TensorDeviceBuffer *source,
                                  size_t source_offset, void *destination,
                                  size_t byte_length, char *error,
                                  size_t error_capacity) {
    if (source == NULL || (byte_length > 0 && destination == NULL) ||
        !tensor_device_valid_range(source->byte_length, source_offset, byte_length)) {
        tensor_device_set_error(error, error_capacity, "invalid device download range or pointer");
        return 0;
    }
    if (byte_length == 0) return 1;
    id<MTLBuffer> staging = [tensor_device_metal_shared_device()
        newBufferWithLength:byte_length options:MTLResourceStorageModeShared];
    if (staging == nil) {
        tensor_device_set_error(error, error_capacity, "Metal download staging allocation failed");
        return 0;
    }
    if (!tensor_device_metal_copy(tensor_device_native_buffer(source), source_offset,
                                  staging, 0, byte_length, 1, error,
                                  error_capacity)) return 0;
    memcpy(destination, staging.contents, byte_length);
    return 1;
}

int tensor_device_buffer_copy(TensorDeviceBuffer *destination,
                              size_t destination_offset,
                              const TensorDeviceBuffer *source,
                              size_t source_offset, size_t byte_length,
                              char *error, size_t error_capacity) {
    if (destination == NULL || source == NULL ||
        !tensor_device_valid_range(destination->byte_length, destination_offset,
                                   byte_length) ||
        !tensor_device_valid_range(source->byte_length, source_offset, byte_length)) {
        tensor_device_set_error(error, error_capacity, "invalid device-to-device copy range");
        return 0;
    }
    return tensor_device_metal_copy(tensor_device_native_buffer(source), source_offset,
                                    tensor_device_native_buffer(destination),
                                    destination_offset, byte_length, 0, error,
                                    error_capacity);
}

int tensor_device_synchronize(char *error, size_t error_capacity) {
    id<MTLCommandBuffer> command = [tensor_device_metal_shared_queue() commandBuffer];
    if (command == nil) {
        tensor_device_set_error(error, error_capacity, "Metal synchronization command creation failed");
        return 0;
    }
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        tensor_device_set_error(error, error_capacity, "Metal device synchronization failed");
        return 0;
    }
    return 1;
}
