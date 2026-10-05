#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "tensor_compute_backend.h"
#include "metal_library.h"
#include "attention_metal.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    uint32_t rows;
    uint32_t width;
    float epsilon;
} NormParams;

typedef struct {
    uint32_t sequence;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    float theta;
    uint32_t position_offset;
} RopeParams;

typedef struct {
    uint32_t width;
} BiasParams;

typedef struct {
    uint32_t width;
} BiasGeluParams;

static id<MTLDevice> compute_device;
static id<MTLCommandQueue> compute_queue;
static id<MTLLibrary> compute_library;
static id<MTLComputePipelineState> embedding_pipeline;
static id<MTLComputePipelineState> rms_pipeline;
static id<MTLComputePipelineState> rms_f16_pipeline;
static id<MTLComputePipelineState> layer_norm_pipeline;
static id<MTLComputePipelineState> layer_norm_f16_pipeline;
static id<MTLComputePipelineState> embedding_f16_pipeline;
static id<MTLComputePipelineState> bias_add_pipeline;
static id<MTLComputePipelineState> bias_add_f16_pipeline;
static id<MTLComputePipelineState> gelu_pipeline;
static id<MTLComputePipelineState> gelu_f16_pipeline;
static id<MTLComputePipelineState> bias_gelu_pipeline;
static id<MTLComputePipelineState> bias_gelu_f16_pipeline;
static id<MTLComputePipelineState> rope_pipeline;
static id<MTLComputePipelineState> rope_f16_pipeline;
static id<MTLComputePipelineState> silu_pipeline;
static id<MTLComputePipelineState> silu_f16_pipeline;
static id<MTLComputePipelineState> residual_pipeline;
static id<MTLComputePipelineState> residual_f16_pipeline;
static id<MTLComputePipelineState> argmax_pipeline;
static id<MTLComputePipelineState> argmax_f16_pipeline;
static NSMutableDictionary<NSValue *, id<MTLBuffer>> *weight_buffers;
static NSMutableDictionary<NSValue *, id<MTLBuffer>> *vision_state_buffers;
static int compute_initialized;
static int compute_available;

static void dispatch_linear(id<MTLComputeCommandEncoder> encoder,
                            id<MTLComputePipelineState> pipeline,
                            NSUInteger count);

static int initialize_compute(void) {
    @synchronized([NSProcessInfo class]) {
        if (compute_initialized) return compute_available;
        compute_initialized = 1;
        compute_device = MTLCreateSystemDefaultDevice();
        if (compute_device == nil) {
            fprintf(stderr, "Metal tensor backend: no default Metal device\n");
            return 0;
        }
        compute_queue = [compute_device newCommandQueue];
        if (compute_queue == nil) {
            fprintf(stderr, "Metal tensor backend: command queue creation failed\n");
            return 0;
        }
        const char *configured_path = getenv("TENSOR_KERNELS_METALLIB_PATH");
        NSString *path = configured_path == NULL ?
            @"build/tensor_kernels.metallib" :
            [NSString stringWithUTF8String:configured_path];
        NSError *error = nil;
        compute_library = metal_load_library(
            compute_device, "tensor", path, &error);
        if (compute_library == nil) {
            const char *message = error.localizedDescription.UTF8String;
            fprintf(stderr, "Metal tensor backend library load failed: %s\n",
                    message == NULL ? "unknown error" : message);
            return 0;
        }
        id<MTLFunction> (^function)(NSString *) = ^id<MTLFunction>(NSString *name) {
            return [compute_library newFunctionWithName:name];
        };
        id<MTLFunction> embedding = function(@"tensor_embedding_f32");
        id<MTLFunction> embedding_f16 = function(@"tensor_embedding_f16");
        id<MTLFunction> rms = function(@"tensor_rms_norm_f32");
        id<MTLFunction> rms_f16 = function(@"tensor_rms_norm_f16");
        id<MTLFunction> layer_norm = function(@"tensor_layer_norm_f32");
        id<MTLFunction> layer_norm_f16 = function(@"tensor_layer_norm_f16");
        id<MTLFunction> bias_add = function(@"tensor_bias_add_f32");
        id<MTLFunction> bias_add_f16 = function(@"tensor_bias_add_f16");
        id<MTLFunction> gelu = function(@"tensor_gelu_f32");
        id<MTLFunction> gelu_f16 = function(@"tensor_gelu_f16");
        id<MTLFunction> bias_gelu = function(@"tensor_bias_gelu_f32");
        id<MTLFunction> bias_gelu_f16 = function(@"tensor_bias_gelu_f16");
        id<MTLFunction> rope = function(@"tensor_rope_f32");
        id<MTLFunction> rope_f16 = function(@"tensor_rope_f16");
        id<MTLFunction> silu = function(@"tensor_silu_multiply_f32");
        id<MTLFunction> silu_f16 = function(@"tensor_silu_multiply_f16");
        id<MTLFunction> residual = function(@"tensor_residual_add_f32");
        id<MTLFunction> residual_f16 = function(@"tensor_residual_add_f16");
        id<MTLFunction> argmax = function(@"tensor_argmax_f32");
        id<MTLFunction> argmax_f16 = function(@"tensor_argmax_f16");
        if (embedding == nil || embedding_f16 == nil || rms == nil || rms_f16 == nil ||
            layer_norm == nil || layer_norm_f16 == nil || bias_add == nil || bias_add_f16 == nil ||
            gelu == nil || gelu_f16 == nil || bias_gelu == nil || bias_gelu_f16 == nil || rope == nil || silu == nil ||
            rope_f16 == nil || silu_f16 == nil || residual == nil || residual_f16 == nil ||
            argmax == nil || argmax_f16 == nil) return 0;
        embedding_pipeline = [compute_device newComputePipelineStateWithFunction:embedding error:&error];
        embedding_f16_pipeline = [compute_device newComputePipelineStateWithFunction:embedding_f16 error:&error];
        rms_pipeline = [compute_device newComputePipelineStateWithFunction:rms error:&error];
        rms_f16_pipeline = [compute_device newComputePipelineStateWithFunction:rms_f16 error:&error];
        layer_norm_pipeline = [compute_device newComputePipelineStateWithFunction:layer_norm error:&error];
        layer_norm_f16_pipeline = [compute_device newComputePipelineStateWithFunction:layer_norm_f16 error:&error];
        bias_add_pipeline = [compute_device newComputePipelineStateWithFunction:bias_add error:&error];
        bias_add_f16_pipeline = [compute_device newComputePipelineStateWithFunction:bias_add_f16 error:&error];
        gelu_pipeline = [compute_device newComputePipelineStateWithFunction:gelu error:&error];
        gelu_f16_pipeline = [compute_device newComputePipelineStateWithFunction:gelu_f16 error:&error];
        bias_gelu_pipeline = [compute_device newComputePipelineStateWithFunction:bias_gelu error:&error];
        bias_gelu_f16_pipeline = [compute_device newComputePipelineStateWithFunction:bias_gelu_f16 error:&error];
        rope_pipeline = [compute_device newComputePipelineStateWithFunction:rope error:&error];
        rope_f16_pipeline = [compute_device newComputePipelineStateWithFunction:rope_f16 error:&error];
        silu_pipeline = [compute_device newComputePipelineStateWithFunction:silu error:&error];
        silu_f16_pipeline = [compute_device newComputePipelineStateWithFunction:silu_f16 error:&error];
        residual_pipeline = [compute_device newComputePipelineStateWithFunction:residual error:&error];
        residual_f16_pipeline = [compute_device newComputePipelineStateWithFunction:residual_f16 error:&error];
        argmax_pipeline = [compute_device newComputePipelineStateWithFunction:argmax error:&error];
        argmax_f16_pipeline = [compute_device newComputePipelineStateWithFunction:argmax_f16 error:&error];
        compute_available = embedding_pipeline != nil && rms_pipeline != nil &&
            embedding_f16_pipeline != nil && rms_f16_pipeline != nil && layer_norm_f16_pipeline != nil &&
            bias_add_f16_pipeline != nil && gelu_f16_pipeline != nil &&
            rope_f16_pipeline != nil && silu_f16_pipeline != nil && residual_f16_pipeline != nil &&
            argmax_f16_pipeline != nil &&
            layer_norm_pipeline != nil && bias_add_pipeline != nil && gelu_pipeline != nil &&
            bias_gelu_pipeline != nil &&
            bias_gelu_f16_pipeline != nil &&
            rope_pipeline != nil && silu_pipeline != nil &&
            residual_pipeline != nil && argmax_pipeline != nil;
        if (!compute_available) {
            const char *message = error.localizedDescription.UTF8String;
            fprintf(stderr, "Metal tensor backend initialization failed: %s\n",
                    message == NULL ? "pipeline unavailable" : message);
        }
        if (compute_available && weight_buffers == nil)
            weight_buffers = [NSMutableDictionary dictionary];
        return compute_available;
    }
}

static id<MTLBuffer> input_buffer(const void *data, size_t bytes) {
    if (data == NULL || bytes == 0) return nil;
    return [compute_device newBufferWithBytes:data length:bytes
                                      options:MTLResourceStorageModeShared];
}

static NSValue *vision_state_key(const void *host_key) {
    return [NSValue valueWithPointer:host_key];
}

static id<MTLBuffer> vision_state_input_buffer(const void *host_key, size_t bytes) {
    NSValue *key = vision_state_key(host_key);
    @synchronized(vision_state_buffers) {
        id<MTLBuffer> buffer = vision_state_buffers[key];
        if (buffer != nil && buffer.length >= bytes) return buffer;
    }
    return input_buffer(host_key, bytes);
}

static int vision_state_store_result(const void *host_key, id<MTLBuffer> buffer) {
    NSValue *key = vision_state_key(host_key);
    @synchronized(vision_state_buffers) {
        if (vision_state_buffers[key] == nil) return 0;
        vision_state_buffers[key] = buffer;
        return 1;
    }
}

static id<MTLBuffer> cached_weight_buffer(const void *data, size_t bytes) {
    if (data == NULL || bytes == 0) return nil;
    NSValue *key = [NSValue valueWithPointer:data];
    @synchronized(weight_buffers) {
        id<MTLBuffer> buffer = weight_buffers[key];
        if (buffer != nil && buffer.length >= bytes) return buffer;
        const long page_size_value = sysconf(_SC_PAGESIZE);
        id<MTLBuffer> (^wrap_aligned_memory)(void) = ^id<MTLBuffer> {
            if (page_size_value <= 0) return nil;
            const size_t page_size = (size_t)page_size_value;
            if ((uintptr_t)data % page_size != 0 || bytes > SIZE_MAX - (page_size - 1))
                return nil;
            const size_t length = (bytes + page_size - 1) / page_size * page_size;
            return [compute_device newBufferWithBytesNoCopy:(void *)data length:length
                options:MTLResourceStorageModeShared deallocator:nil];
        };
        buffer = wrap_aligned_memory();
        if (buffer == nil)
            buffer = [compute_device newBufferWithBytes:data length:bytes
                options:MTLResourceStorageModeShared];
        if (buffer != nil) weight_buffers[key] = buffer;
        return buffer;
    }
}

static id<MTLBuffer> output_buffer(size_t bytes) {
    return bytes == 0 ? nil : [compute_device newBufferWithLength:bytes
        options:MTLResourceStorageModeShared];
}

static int encode_linear_f32(id<MTLCommandBuffer> command, id<MTLBuffer> input,
                             const float *weight, id<MTLBuffer> output,
                             size_t rows, size_t input_width, size_t output_width) {
    if (command == nil || input == nil || output == nil || weight == NULL ||
        rows == 0 || input_width == 0 || output_width == 0) return 0;
    id<MTLBuffer> weight_gpu = cached_weight_buffer(weight,
        output_width * input_width * sizeof(float));
    if (weight_gpu == nil) return 0;
    MPSMatrixDescriptor *input_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
        columns:input_width rowBytes:input_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *weight_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:output_width
        columns:input_width rowBytes:input_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *output_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
        columns:output_width rowBytes:output_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrix *input_matrix = [[MPSMatrix alloc] initWithBuffer:input descriptor:input_desc];
    MPSMatrix *weight_matrix = [[MPSMatrix alloc] initWithBuffer:weight_gpu descriptor:weight_desc];
    MPSMatrix *output_matrix = [[MPSMatrix alloc] initWithBuffer:output descriptor:output_desc];
    MPSMatrixMultiplication *op = [[MPSMatrixMultiplication alloc]
        initWithDevice:compute_device transposeLeft:NO transposeRight:YES
        resultRows:rows resultColumns:output_width interiorColumns:input_width
        alpha:1.0 beta:0.0];
    if (input_matrix == nil || weight_matrix == nil || output_matrix == nil || op == nil) return 0;
    [op encodeToCommandBuffer:command leftMatrix:input_matrix
        rightMatrix:weight_matrix resultMatrix:output_matrix];
    return 1;
}

static int encode_norm_f32(id<MTLCommandBuffer> command, id<MTLBuffer> input,
                           const float *scale, const float *bias,
                           id<MTLBuffer> output, size_t rows, size_t width,
                           float epsilon) {
    id<MTLBuffer> scale_gpu = cached_weight_buffer(scale, width * sizeof(float));
    id<MTLBuffer> bias_gpu = cached_weight_buffer(bias, width * sizeof(float));
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil || scale_gpu == nil || bias_gpu == nil || rows * width > UINT32_MAX) return 0;
    NormParams params = {(uint32_t)rows, (uint32_t)width, epsilon};
    [encoder setComputePipelineState:layer_norm_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:scale_gpu offset:0 atIndex:1];
    [encoder setBuffer:bias_gpu offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    return 1;
}

static int encode_bias_f32(id<MTLCommandBuffer> command, id<MTLBuffer> input,
                           const float *bias, id<MTLBuffer> output,
                           size_t rows, size_t width) {
    id<MTLBuffer> bias_gpu = cached_weight_buffer(bias, width * sizeof(float));
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil || bias_gpu == nil || rows * width > UINT32_MAX) return 0;
    BiasParams params = {(uint32_t)width};
    [encoder setComputePipelineState:bias_add_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:bias_gpu offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    dispatch_linear(encoder, bias_add_pipeline, rows * width);
    [encoder endEncoding];
    return 1;
}

static int encode_residual_f32(id<MTLCommandBuffer> command, id<MTLBuffer> left,
                               id<MTLBuffer> right, id<MTLBuffer> output,
                               size_t count) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil || count > UINT32_MAX) return 0;
    [encoder setComputePipelineState:residual_pipeline];
    [encoder setBuffer:left offset:0 atIndex:0];
    [encoder setBuffer:right offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    const uint32_t count32 = (uint32_t)count;
    [encoder setBytes:&count32 length:sizeof(count32) atIndex:3];
    dispatch_linear(encoder, residual_pipeline, count);
    [encoder endEncoding];
    return 1;
}

static int encode_bias_gelu_f32(id<MTLCommandBuffer> command,
                                id<MTLBuffer> input, const float *bias,
                                id<MTLBuffer> output, size_t rows, size_t width) {
    id<MTLBuffer> bias_gpu = cached_weight_buffer(bias, width * sizeof(float));
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil || bias_gpu == nil || rows * width > UINT32_MAX) return 0;
    BiasGeluParams params = {(uint32_t)width};
    [encoder setComputePipelineState:bias_gelu_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:bias_gpu offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    dispatch_linear(encoder, bias_gelu_pipeline, rows * width);
    [encoder endEncoding];
    return 1;
}

static int encode_linear_f16(id<MTLCommandBuffer> command, id<MTLBuffer> input,
                             const uint16_t *weight, id<MTLBuffer> output,
                             size_t rows, size_t input_width, size_t output_width) {
    if (command == nil || input == nil || output == nil || weight == NULL ||
        rows == 0 || input_width == 0 || output_width == 0) return 0;
    id<MTLBuffer> weight_gpu = cached_weight_buffer(weight,
        output_width * input_width * sizeof(uint16_t));
    if (weight_gpu == nil) return 0;
    MPSMatrixDescriptor *input_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:input_width
        rowBytes:input_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *weight_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:output_width columns:input_width
        rowBytes:input_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *output_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:output_width
        rowBytes:output_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrix *input_matrix = [[MPSMatrix alloc] initWithBuffer:input descriptor:input_descriptor];
    MPSMatrix *weight_matrix = [[MPSMatrix alloc] initWithBuffer:weight_gpu descriptor:weight_descriptor];
    MPSMatrix *output_matrix = [[MPSMatrix alloc] initWithBuffer:output descriptor:output_descriptor];
    MPSMatrixMultiplication *operation = [[MPSMatrixMultiplication alloc]
        initWithDevice:compute_device transposeLeft:NO transposeRight:YES
        resultRows:rows resultColumns:output_width interiorColumns:input_width
        alpha:1.0 beta:0.0];
    if (input_matrix == nil || weight_matrix == nil || output_matrix == nil || operation == nil)
        return 0;
    [operation encodeToCommandBuffer:command leftMatrix:input_matrix
        rightMatrix:weight_matrix resultMatrix:output_matrix];
    return 1;
}

static int encode_norm_f16(id<MTLCommandBuffer> command, id<MTLBuffer> input,
                           const uint16_t *scale, const uint16_t *bias,
                           id<MTLBuffer> output, size_t rows, size_t width,
                           float epsilon) {
    id<MTLBuffer> scale_gpu = cached_weight_buffer(scale, width * sizeof(uint16_t));
    id<MTLBuffer> bias_gpu = cached_weight_buffer(bias, width * sizeof(uint16_t));
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil || scale_gpu == nil || bias_gpu == nil) return 0;
    NormParams params = {(uint32_t)rows, (uint32_t)width, epsilon};
    [encoder setComputePipelineState:layer_norm_f16_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:scale_gpu offset:0 atIndex:1];
    [encoder setBuffer:bias_gpu offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    return 1;
}

static int encode_bias_f16(id<MTLCommandBuffer> command, id<MTLBuffer> input,
                           const uint16_t *bias, id<MTLBuffer> output,
                           size_t rows, size_t width) {
    id<MTLBuffer> bias_gpu = cached_weight_buffer(bias, width * sizeof(uint16_t));
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil || bias_gpu == nil || rows * width > UINT32_MAX) return 0;
    BiasParams params = {(uint32_t)width};
    [encoder setComputePipelineState:bias_add_f16_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:bias_gpu offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    dispatch_linear(encoder, bias_add_f16_pipeline, rows * width);
    [encoder endEncoding];
    return 1;
}

static int encode_residual_f16(id<MTLCommandBuffer> command, id<MTLBuffer> left,
                               id<MTLBuffer> right, id<MTLBuffer> output,
                               size_t count) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil || count > UINT32_MAX) return 0;
    [encoder setComputePipelineState:residual_f16_pipeline];
    [encoder setBuffer:left offset:0 atIndex:0];
    [encoder setBuffer:right offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    dispatch_linear(encoder, residual_f16_pipeline, count);
    [encoder endEncoding];
    return 1;
}

static int encode_bias_gelu_f16(id<MTLCommandBuffer> command,
                                id<MTLBuffer> input, const uint16_t *bias,
                                id<MTLBuffer> output, size_t rows, size_t width) {
    id<MTLBuffer> bias_gpu = cached_weight_buffer(bias, width * sizeof(uint16_t));
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil || bias_gpu == nil || rows * width > UINT32_MAX) return 0;
    BiasGeluParams params = {(uint32_t)width};
    [encoder setComputePipelineState:bias_gelu_f16_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:bias_gpu offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    dispatch_linear(encoder, bias_gelu_f16_pipeline, rows * width);
    [encoder endEncoding];
    return 1;
}

static void dispatch_linear(id<MTLComputeCommandEncoder> encoder,
                            id<MTLComputePipelineState> pipeline,
                            NSUInteger count) {
    [encoder setComputePipelineState:pipeline];
    const MTLSize grid = MTLSizeMake(count, 1, 1);
    const MTLSize group = MTLSizeMake(MIN((NSUInteger)256,
        pipeline.maxTotalThreadsPerThreadgroup), 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
}

static int finish_command(id<MTLCommandBuffer> command, float *output,
                          id<MTLBuffer> output_buffer, size_t output_bytes) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    if (output_bytes != 0) memcpy(output, output_buffer.contents, output_bytes);
    return 1;
}

static int metal_available(void) {
    @autoreleasepool {
        return initialize_compute();
    }
}

static int metal_linear(const float *input, const float *weight, float *output,
                        size_t rows, size_t input_width, size_t output_width) {
    @autoreleasepool {
    if (!initialize_compute() ||
        input == NULL || weight == NULL || output == NULL ||
        rows == 0 || input_width == 0 || output_width == 0 ||
        rows > INT_MAX || input_width > INT_MAX || output_width > INT_MAX ||
        rows > SIZE_MAX / input_width / sizeof(float) ||
        output_width > SIZE_MAX / input_width / sizeof(float) ||
        rows > SIZE_MAX / output_width / sizeof(float)) return 0;
    id<MTLBuffer> input_gpu = input_buffer(input, rows * input_width * sizeof(float));
    id<MTLBuffer> weight_gpu = cached_weight_buffer(weight,
        output_width * input_width * sizeof(float));
    id<MTLBuffer> output_gpu = output_buffer(rows * output_width * sizeof(float));
    if (input_gpu == nil || weight_gpu == nil || output_gpu == nil) return 0;
    MPSMatrixDescriptor *input_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:input_width
        rowBytes:input_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *weight_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:output_width columns:input_width
        rowBytes:input_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *output_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:output_width
        rowBytes:output_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrix *input_matrix = [[MPSMatrix alloc] initWithBuffer:input_gpu
        descriptor:input_descriptor];
    MPSMatrix *weight_matrix = [[MPSMatrix alloc] initWithBuffer:weight_gpu
        descriptor:weight_descriptor];
    MPSMatrix *output_matrix = [[MPSMatrix alloc] initWithBuffer:output_gpu
        descriptor:output_descriptor];
    MPSMatrixMultiplication *operation = [[MPSMatrixMultiplication alloc]
        initWithDevice:compute_device transposeLeft:NO transposeRight:YES
        resultRows:rows resultColumns:output_width interiorColumns:input_width
        alpha:1.0 beta:0.0];
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    if (input_matrix == nil || weight_matrix == nil || output_matrix == nil ||
        operation == nil || command == nil) return 0;
    [operation encodeToCommandBuffer:command leftMatrix:input_matrix
        rightMatrix:weight_matrix resultMatrix:output_matrix];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, output_gpu.contents, rows * output_width * sizeof(float));
    return 1;
    }
}

static int metal_linear_f16(const uint16_t *input, const uint16_t *weight,
                            uint16_t *output, size_t rows,
                            size_t input_width, size_t output_width) {
    @autoreleasepool {
    if (!initialize_compute() || input == NULL || weight == NULL || output == NULL ||
        rows == 0 || input_width == 0 || output_width == 0 ||
        rows > UINT32_MAX || input_width > UINT32_MAX || output_width > UINT32_MAX ||
        rows > SIZE_MAX / input_width / sizeof(uint16_t) ||
        output_width > SIZE_MAX / input_width / sizeof(uint16_t) ||
        rows > SIZE_MAX / output_width / sizeof(uint16_t)) return 0;
    const size_t input_bytes = rows * input_width * sizeof(uint16_t);
    const size_t weight_bytes = output_width * input_width * sizeof(uint16_t);
    const size_t output_bytes = rows * output_width * sizeof(uint16_t);
    id<MTLBuffer> input_gpu = input_buffer(input, input_bytes);
    id<MTLBuffer> weight_gpu = cached_weight_buffer(weight, weight_bytes);
    id<MTLBuffer> output_gpu = output_buffer(output_bytes);
    if (input_gpu == nil || weight_gpu == nil || output_gpu == nil) return 0;
    MPSMatrixDescriptor *input_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:input_width
        rowBytes:input_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *weight_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:output_width columns:input_width
        rowBytes:input_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *output_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:output_width
        rowBytes:output_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrix *input_matrix = [[MPSMatrix alloc] initWithBuffer:input_gpu
        descriptor:input_descriptor];
    MPSMatrix *weight_matrix = [[MPSMatrix alloc] initWithBuffer:weight_gpu
        descriptor:weight_descriptor];
    MPSMatrix *output_matrix = [[MPSMatrix alloc] initWithBuffer:output_gpu
        descriptor:output_descriptor];
    MPSMatrixMultiplication *operation = [[MPSMatrixMultiplication alloc]
        initWithDevice:compute_device transposeLeft:NO transposeRight:YES
        resultRows:rows resultColumns:output_width interiorColumns:input_width
        alpha:1.0 beta:0.0];
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    if (input_matrix == nil || weight_matrix == nil || output_matrix == nil ||
        operation == nil || command == nil) return 0;
    [operation encodeToCommandBuffer:command leftMatrix:input_matrix
        rightMatrix:weight_matrix resultMatrix:output_matrix];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, output_gpu.contents, output_bytes);
    return 1;
    }
}

static int metal_mlp(const float *input, const float *fc1_weight,
                     const float *fc1_bias, const float *fc2_weight,
                     const float *fc2_bias, float *output, size_t rows,
                     size_t input_width, size_t intermediate_width,
                     size_t output_width) {
    @autoreleasepool {
    if (!initialize_compute() || input == NULL || fc1_weight == NULL ||
        fc1_bias == NULL || fc2_weight == NULL || fc2_bias == NULL ||
        output == NULL || rows == 0 || input_width == 0 ||
        intermediate_width == 0 || output_width == 0 || rows > INT_MAX ||
        input_width > INT_MAX || intermediate_width > INT_MAX ||
        output_width > INT_MAX || rows > SIZE_MAX / input_width / sizeof(float) ||
        rows > SIZE_MAX / intermediate_width / sizeof(float) ||
        rows > SIZE_MAX / output_width / sizeof(float) ||
        intermediate_width > SIZE_MAX / input_width / sizeof(float) ||
        output_width > SIZE_MAX / intermediate_width / sizeof(float) ||
        intermediate_width > UINT32_MAX || output_width > UINT32_MAX) return 0;

    const size_t input_bytes = rows * input_width * sizeof(float);
    const size_t intermediate_bytes = rows * intermediate_width * sizeof(float);
    const size_t output_bytes = rows * output_width * sizeof(float);
    id<MTLBuffer> input_gpu = input_buffer(input, input_bytes);
    id<MTLBuffer> fc1_weight_gpu = cached_weight_buffer(fc1_weight,
        intermediate_width * input_width * sizeof(float));
    id<MTLBuffer> fc1_bias_gpu = cached_weight_buffer(fc1_bias,
        intermediate_width * sizeof(float));
    id<MTLBuffer> fc1_output_gpu = output_buffer(intermediate_bytes);
    id<MTLBuffer> activated_gpu = output_buffer(intermediate_bytes);
    id<MTLBuffer> fc2_weight_gpu = cached_weight_buffer(fc2_weight,
        output_width * intermediate_width * sizeof(float));
    id<MTLBuffer> fc2_bias_gpu = cached_weight_buffer(fc2_bias,
        output_width * sizeof(float));
    id<MTLBuffer> fc2_output_gpu = output_buffer(output_bytes);
    id<MTLBuffer> output_gpu = output_buffer(output_bytes);
    if (input_gpu == nil || fc1_weight_gpu == nil || fc1_bias_gpu == nil ||
        fc1_output_gpu == nil || activated_gpu == nil || fc2_weight_gpu == nil ||
        fc2_bias_gpu == nil || fc2_output_gpu == nil || output_gpu == nil) return 0;

    MPSMatrixDescriptor *input_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:input_width
        rowBytes:input_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *fc1_weight_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:intermediate_width columns:input_width
        rowBytes:input_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *fc1_output_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:intermediate_width
        rowBytes:intermediate_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *fc2_weight_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:output_width columns:intermediate_width
        rowBytes:intermediate_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *output_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:output_width
        rowBytes:output_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrix *input_matrix = [[MPSMatrix alloc] initWithBuffer:input_gpu
        descriptor:input_descriptor];
    MPSMatrix *fc1_weight_matrix = [[MPSMatrix alloc] initWithBuffer:fc1_weight_gpu
        descriptor:fc1_weight_descriptor];
    MPSMatrix *fc1_output_matrix = [[MPSMatrix alloc] initWithBuffer:fc1_output_gpu
        descriptor:fc1_output_descriptor];
    MPSMatrix *fc2_weight_matrix = [[MPSMatrix alloc] initWithBuffer:fc2_weight_gpu
        descriptor:fc2_weight_descriptor];
    MPSMatrix *activated_matrix = [[MPSMatrix alloc] initWithBuffer:activated_gpu
        descriptor:fc1_output_descriptor];
    MPSMatrix *fc2_output_matrix = [[MPSMatrix alloc] initWithBuffer:fc2_output_gpu
        descriptor:output_descriptor];
    MPSMatrixMultiplication *fc1 = [[MPSMatrixMultiplication alloc]
        initWithDevice:compute_device transposeLeft:NO transposeRight:YES
        resultRows:rows resultColumns:intermediate_width interiorColumns:input_width
        alpha:1.0 beta:0.0];
    MPSMatrixMultiplication *fc2 = [[MPSMatrixMultiplication alloc]
        initWithDevice:compute_device transposeLeft:NO transposeRight:YES
        resultRows:rows resultColumns:output_width interiorColumns:intermediate_width
        alpha:1.0 beta:0.0];
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    if (input_matrix == nil || fc1_weight_matrix == nil || fc1_output_matrix == nil ||
        fc2_weight_matrix == nil || activated_matrix == nil || fc2_output_matrix == nil ||
        fc1 == nil || fc2 == nil || command == nil) return 0;

    [fc1 encodeToCommandBuffer:command leftMatrix:input_matrix
        rightMatrix:fc1_weight_matrix resultMatrix:fc1_output_matrix];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    BiasGeluParams first_params = {(uint32_t)intermediate_width};
    [encoder setComputePipelineState:bias_gelu_pipeline];
    [encoder setBuffer:fc1_output_gpu offset:0 atIndex:0];
    [encoder setBuffer:fc1_bias_gpu offset:0 atIndex:1];
    [encoder setBuffer:activated_gpu offset:0 atIndex:2];
    [encoder setBytes:&first_params length:sizeof(first_params) atIndex:3];
    dispatch_linear(encoder, bias_gelu_pipeline, rows * intermediate_width);
    [encoder endEncoding];
    [fc2 encodeToCommandBuffer:command leftMatrix:activated_matrix
        rightMatrix:fc2_weight_matrix resultMatrix:fc2_output_matrix];
    encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    BiasParams second_params = {(uint32_t)output_width};
    [encoder setComputePipelineState:bias_add_pipeline];
    [encoder setBuffer:fc2_output_gpu offset:0 atIndex:0];
    [encoder setBuffer:fc2_bias_gpu offset:0 atIndex:1];
    [encoder setBuffer:output_gpu offset:0 atIndex:2];
    [encoder setBytes:&second_params length:sizeof(second_params) atIndex:3];
    dispatch_linear(encoder, bias_add_pipeline, rows * output_width);
    [encoder endEncoding];
    return finish_command(command, output, output_gpu, output_bytes);
    }
}

static int metal_mlp_f16(const uint16_t *input, const uint16_t *fc1_weight,
                         const uint16_t *fc1_bias, const uint16_t *fc2_weight,
                         const uint16_t *fc2_bias, uint16_t *output, size_t rows,
                         size_t input_width, size_t intermediate_width,
                         size_t output_width) {
    @autoreleasepool {
    if (!initialize_compute() || input == NULL || fc1_weight == NULL ||
        fc1_bias == NULL || fc2_weight == NULL || fc2_bias == NULL || output == NULL ||
        rows == 0 || input_width == 0 || intermediate_width == 0 || output_width == 0 ||
        rows > INT_MAX || input_width > INT_MAX || intermediate_width > INT_MAX ||
        output_width > INT_MAX || rows > SIZE_MAX / input_width / sizeof(uint16_t) ||
        rows > SIZE_MAX / intermediate_width / sizeof(uint16_t) ||
        rows > SIZE_MAX / output_width / sizeof(uint16_t) ||
        rows > UINT32_MAX / intermediate_width ||
        intermediate_width > SIZE_MAX / input_width / sizeof(uint16_t) ||
        output_width > SIZE_MAX / intermediate_width / sizeof(uint16_t) ||
        intermediate_width > UINT32_MAX || output_width > UINT32_MAX) return 0;

    const size_t input_bytes = rows * input_width * sizeof(uint16_t);
    const size_t intermediate_bytes = rows * intermediate_width * sizeof(uint16_t);
    const size_t output_bytes = rows * output_width * sizeof(uint16_t);
    id<MTLBuffer> input_gpu = input_buffer(input, input_bytes);
    id<MTLBuffer> fc1_weight_gpu = cached_weight_buffer(fc1_weight,
        intermediate_width * input_width * sizeof(uint16_t));
    id<MTLBuffer> fc1_bias_gpu = cached_weight_buffer(fc1_bias,
        intermediate_width * sizeof(uint16_t));
    id<MTLBuffer> fc1_output_gpu = output_buffer(intermediate_bytes);
    id<MTLBuffer> activated_gpu = output_buffer(intermediate_bytes);
    id<MTLBuffer> fc2_weight_gpu = cached_weight_buffer(fc2_weight,
        output_width * intermediate_width * sizeof(uint16_t));
    id<MTLBuffer> fc2_bias_gpu = cached_weight_buffer(fc2_bias,
        output_width * sizeof(uint16_t));
    id<MTLBuffer> fc2_output_gpu = output_buffer(output_bytes);
    id<MTLBuffer> output_gpu = output_buffer(output_bytes);
    if (input_gpu == nil || fc1_weight_gpu == nil || fc1_bias_gpu == nil ||
        fc1_output_gpu == nil || activated_gpu == nil ||
        fc2_weight_gpu == nil || fc2_bias_gpu == nil || fc2_output_gpu == nil ||
        output_gpu == nil) return 0;

    MPSMatrixDescriptor *input_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:input_width
        rowBytes:input_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *fc1_weight_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:intermediate_width columns:input_width
        rowBytes:input_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *fc1_output_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:intermediate_width
        rowBytes:intermediate_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *fc2_weight_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:output_width columns:intermediate_width
        rowBytes:intermediate_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *output_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:output_width
        rowBytes:output_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrix *input_matrix = [[MPSMatrix alloc] initWithBuffer:input_gpu
        descriptor:input_descriptor];
    MPSMatrix *fc1_weight_matrix = [[MPSMatrix alloc] initWithBuffer:fc1_weight_gpu
        descriptor:fc1_weight_descriptor];
    MPSMatrix *fc1_output_matrix = [[MPSMatrix alloc] initWithBuffer:fc1_output_gpu
        descriptor:fc1_output_descriptor];
    MPSMatrix *activated_matrix = [[MPSMatrix alloc] initWithBuffer:activated_gpu
        descriptor:fc1_output_descriptor];
    MPSMatrix *fc2_weight_matrix = [[MPSMatrix alloc] initWithBuffer:fc2_weight_gpu
        descriptor:fc2_weight_descriptor];
    MPSMatrix *fc2_output_matrix = [[MPSMatrix alloc] initWithBuffer:fc2_output_gpu
        descriptor:output_descriptor];
    MPSMatrixMultiplication *fc1 = [[MPSMatrixMultiplication alloc]
        initWithDevice:compute_device transposeLeft:NO transposeRight:YES
        resultRows:rows resultColumns:intermediate_width interiorColumns:input_width
        alpha:1.0 beta:0.0];
    MPSMatrixMultiplication *fc2 = [[MPSMatrixMultiplication alloc]
        initWithDevice:compute_device transposeLeft:NO transposeRight:YES
        resultRows:rows resultColumns:output_width interiorColumns:intermediate_width
        alpha:1.0 beta:0.0];
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    if (input_matrix == nil || fc1_weight_matrix == nil || fc1_output_matrix == nil ||
        activated_matrix == nil || fc2_weight_matrix == nil ||
        fc2_output_matrix == nil || fc1 == nil || fc2 == nil ||
        command == nil) return 0;

    [fc1 encodeToCommandBuffer:command leftMatrix:input_matrix
        rightMatrix:fc1_weight_matrix resultMatrix:fc1_output_matrix];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    BiasGeluParams bias_gelu_params = {(uint32_t)intermediate_width};
    [encoder setComputePipelineState:bias_gelu_f16_pipeline];
    [encoder setBuffer:fc1_output_gpu offset:0 atIndex:0];
    [encoder setBuffer:fc1_bias_gpu offset:0 atIndex:1];
    [encoder setBuffer:activated_gpu offset:0 atIndex:2];
    [encoder setBytes:&bias_gelu_params length:sizeof(bias_gelu_params) atIndex:3];
    dispatch_linear(encoder, bias_gelu_f16_pipeline, rows * intermediate_width);
    [encoder endEncoding];
    [fc2 encodeToCommandBuffer:command leftMatrix:activated_matrix
        rightMatrix:fc2_weight_matrix resultMatrix:fc2_output_matrix];
    encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    BiasParams output_params = {(uint32_t)output_width};
    [encoder setComputePipelineState:bias_add_f16_pipeline];
    [encoder setBuffer:fc2_output_gpu offset:0 atIndex:0];
    [encoder setBuffer:fc2_bias_gpu offset:0 atIndex:1];
    [encoder setBuffer:output_gpu offset:0 atIndex:2];
    [encoder setBytes:&output_params length:sizeof(output_params) atIndex:3];
    dispatch_linear(encoder, bias_add_f16_pipeline, rows * output_width);
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, output_gpu.contents, output_bytes);
    return 1;
    }
}

static int metal_qkv_f16(const uint16_t *input, const uint16_t *q_weight,
                         const uint16_t *k_weight, const uint16_t *v_weight,
                         const uint16_t *q_bias, const uint16_t *k_bias,
                         const uint16_t *v_bias, uint16_t *query,
                         uint16_t *key, uint16_t *value, size_t rows,
                         size_t input_width, size_t output_width) {
    @autoreleasepool {
    if (!initialize_compute() || input == NULL || q_weight == NULL ||
        k_weight == NULL || v_weight == NULL || query == NULL || key == NULL ||
        value == NULL || rows == 0 || input_width == 0 || output_width == 0 ||
        rows > INT_MAX || input_width > INT_MAX || output_width > INT_MAX ||
        rows > SIZE_MAX / input_width / sizeof(uint16_t) ||
        output_width > SIZE_MAX / input_width / sizeof(uint16_t) ||
        rows > SIZE_MAX / output_width / sizeof(uint16_t) ||
        rows > UINT32_MAX || output_width > UINT32_MAX) return 0;

    const size_t input_bytes = rows * input_width * sizeof(uint16_t);
    const size_t output_bytes = rows * output_width * sizeof(uint16_t);
    const size_t weight_bytes = output_width * input_width * sizeof(uint16_t);
    id<MTLBuffer> input_gpu = input_buffer(input, input_bytes);
    id<MTLBuffer> q_weight_gpu = cached_weight_buffer(q_weight, weight_bytes);
    id<MTLBuffer> k_weight_gpu = cached_weight_buffer(k_weight, weight_bytes);
    id<MTLBuffer> v_weight_gpu = cached_weight_buffer(v_weight, weight_bytes);
    id<MTLBuffer> q_bias_gpu = q_bias == NULL ? nil : cached_weight_buffer(q_bias, output_width * sizeof(uint16_t));
    id<MTLBuffer> k_bias_gpu = k_bias == NULL ? nil : cached_weight_buffer(k_bias, output_width * sizeof(uint16_t));
    id<MTLBuffer> v_bias_gpu = v_bias == NULL ? nil : cached_weight_buffer(v_bias, output_width * sizeof(uint16_t));
    id<MTLBuffer> q_matmul_gpu = output_buffer(output_bytes);
    id<MTLBuffer> k_matmul_gpu = output_buffer(output_bytes);
    id<MTLBuffer> v_matmul_gpu = output_buffer(output_bytes);
    id<MTLBuffer> q_gpu = output_buffer(output_bytes);
    id<MTLBuffer> k_gpu = output_buffer(output_bytes);
    id<MTLBuffer> v_gpu = output_buffer(output_bytes);
    if (input_gpu == nil || q_weight_gpu == nil || k_weight_gpu == nil ||
        v_weight_gpu == nil || (q_bias != NULL && q_bias_gpu == nil) ||
        (k_bias != NULL && k_bias_gpu == nil) || (v_bias != NULL && v_bias_gpu == nil) ||
        q_matmul_gpu == nil || k_matmul_gpu == nil ||
        v_matmul_gpu == nil || q_gpu == nil || k_gpu == nil || v_gpu == nil) return 0;

    MPSMatrixDescriptor *input_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:input_width
        rowBytes:input_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *weight_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:output_width columns:input_width
        rowBytes:input_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *output_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:output_width
        rowBytes:output_width * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrix *input_matrix = [[MPSMatrix alloc] initWithBuffer:input_gpu
        descriptor:input_descriptor];
    MPSMatrix *q_weight_matrix = [[MPSMatrix alloc] initWithBuffer:q_weight_gpu
        descriptor:weight_descriptor];
    MPSMatrix *k_weight_matrix = [[MPSMatrix alloc] initWithBuffer:k_weight_gpu
        descriptor:weight_descriptor];
    MPSMatrix *v_weight_matrix = [[MPSMatrix alloc] initWithBuffer:v_weight_gpu
        descriptor:weight_descriptor];
    MPSMatrix *q_matrix = [[MPSMatrix alloc] initWithBuffer:q_matmul_gpu descriptor:output_descriptor];
    MPSMatrix *k_matrix = [[MPSMatrix alloc] initWithBuffer:k_matmul_gpu descriptor:output_descriptor];
    MPSMatrix *v_matrix = [[MPSMatrix alloc] initWithBuffer:v_matmul_gpu descriptor:output_descriptor];
    MPSMatrixMultiplication *operation = [[MPSMatrixMultiplication alloc]
        initWithDevice:compute_device transposeLeft:NO transposeRight:YES
        resultRows:rows resultColumns:output_width interiorColumns:input_width
        alpha:1.0 beta:0.0];
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    if (input_matrix == nil || q_weight_matrix == nil || k_weight_matrix == nil ||
        v_weight_matrix == nil || q_matrix == nil || k_matrix == nil || v_matrix == nil ||
        operation == nil || command == nil) return 0;
    [operation encodeToCommandBuffer:command leftMatrix:input_matrix
        rightMatrix:q_weight_matrix resultMatrix:q_matrix];
    [operation encodeToCommandBuffer:command leftMatrix:input_matrix
        rightMatrix:k_weight_matrix resultMatrix:k_matrix];
    [operation encodeToCommandBuffer:command leftMatrix:input_matrix
        rightMatrix:v_weight_matrix resultMatrix:v_matrix];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:bias_add_f16_pipeline];
    const BiasParams params = {(uint32_t)output_width};
    id<MTLBuffer> matrices[] = {q_matmul_gpu, k_matmul_gpu, v_matmul_gpu};
    id<MTLBuffer> biases[] = {q_bias_gpu, k_bias_gpu, v_bias_gpu};
    id<MTLBuffer> outputs[] = {q_gpu, k_gpu, v_gpu};
    for (size_t i = 0; i < 3; i++) {
        if (biases[i] == nil) continue;
        [encoder setBuffer:matrices[i] offset:0 atIndex:0];
        [encoder setBuffer:biases[i] offset:0 atIndex:1];
        [encoder setBuffer:outputs[i] offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_linear(encoder, bias_add_f16_pipeline, rows * output_width);
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(query, (q_bias_gpu == nil ? q_matmul_gpu : q_gpu).contents, output_bytes);
    memcpy(key, (k_bias_gpu == nil ? k_matmul_gpu : k_gpu).contents, output_bytes);
    memcpy(value, (v_bias_gpu == nil ? v_matmul_gpu : v_gpu).contents, output_bytes);
    return 1;
    }
}

static int metal_vision_layer_f16(uint16_t *state,
                                  const TensorVisionLayerF16 *layer,
                                  size_t rows, size_t hidden,
                                  size_t intermediate, float epsilon) {
    @autoreleasepool {
    if (!initialize_compute() || state == NULL || layer == NULL || rows == 0 ||
        hidden == 0 || intermediate == 0 || hidden % 64 != 0 ||
        rows > INT_MAX || hidden > INT_MAX || intermediate > INT_MAX ||
        rows > SIZE_MAX / hidden / sizeof(uint16_t) ||
        rows > SIZE_MAX / intermediate / sizeof(uint16_t) ||
        hidden > SIZE_MAX / hidden / sizeof(uint16_t) ||
        intermediate > SIZE_MAX / hidden / sizeof(uint16_t) ||
        intermediate > SIZE_MAX / intermediate / sizeof(uint16_t) ||
        rows > UINT32_MAX / intermediate) return 0;
    const size_t count = rows * hidden;
    if (count > UINT32_MAX) return 0;
    const size_t bytes = count * sizeof(uint16_t);
    const size_t intermediate_bytes = rows * intermediate * sizeof(uint16_t);
    id<MTLBuffer> state_input = vision_state_input_buffer(state, bytes);
    id<MTLBuffer> norm1 = output_buffer(bytes);
    id<MTLBuffer> q_raw = output_buffer(bytes), k_raw = output_buffer(bytes);
    id<MTLBuffer> v_raw = output_buffer(bytes), q = output_buffer(bytes);
    id<MTLBuffer> k = output_buffer(bytes), v = output_buffer(bytes);
    id<MTLBuffer> attention = output_buffer(bytes);
    id<MTLBuffer> out_raw = output_buffer(bytes), projected = output_buffer(bytes);
    id<MTLBuffer> after_attention = output_buffer(bytes), norm2 = output_buffer(bytes);
    id<MTLBuffer> fc1_raw = output_buffer(intermediate_bytes);
    id<MTLBuffer> activated = output_buffer(intermediate_bytes);
    id<MTLBuffer> fc2_raw = output_buffer(bytes), mlp = output_buffer(bytes);
    id<MTLBuffer> result = output_buffer(bytes);
    if (state_input == nil || norm1 == nil || q_raw == nil || k_raw == nil ||
        v_raw == nil || q == nil || k == nil || v == nil || attention == nil ||
        out_raw == nil || projected == nil || after_attention == nil || norm2 == nil ||
        fc1_raw == nil || activated == nil || fc2_raw == nil || mlp == nil || result == nil)
        return 0;
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    if (command == nil || !encode_norm_f16(command, state_input, layer->norm1_scale,
            layer->norm1_bias, norm1, rows, hidden, epsilon) ||
        !encode_linear_f16(command, norm1, layer->q_weight, q_raw, rows, hidden, hidden) ||
        !encode_linear_f16(command, norm1, layer->k_weight, k_raw, rows, hidden, hidden) ||
        !encode_linear_f16(command, norm1, layer->v_weight, v_raw, rows, hidden, hidden) ||
        !encode_bias_f16(command, q_raw, layer->q_bias, q, rows, hidden) ||
        !encode_bias_f16(command, k_raw, layer->k_bias, k, rows, hidden) ||
        !encode_bias_f16(command, v_raw, layer->v_bias, v, rows, hidden) ||
        !tensor_attention_metal_encode_tiled_f16((__bridge void *)command,
            (__bridge void *)q, (__bridge void *)k, (__bridge void *)v,
            (__bridge void *)attention, hidden / 64, rows, 1.0f / sqrtf(64.0f)) ||
        !encode_linear_f16(command, attention, layer->out_weight, out_raw,
                           rows, hidden, hidden) ||
        !encode_bias_f16(command, out_raw, layer->out_bias, projected, rows, hidden) ||
        !encode_residual_f16(command, state_input, projected, after_attention, count) ||
        !encode_norm_f16(command, after_attention, layer->norm2_scale,
                         layer->norm2_bias, norm2, rows, hidden, epsilon) ||
        !encode_linear_f16(command, norm2, layer->fc1_weight, fc1_raw,
                           rows, hidden, intermediate) ||
        !encode_bias_gelu_f16(command, fc1_raw, layer->fc1_bias, activated,
                              rows, intermediate) ||
        !encode_linear_f16(command, activated, layer->fc2_weight, fc2_raw,
                           rows, intermediate, hidden) ||
        !encode_bias_f16(command, fc2_raw, layer->fc2_bias, mlp, rows, hidden) ||
        !encode_residual_f16(command, after_attention, mlp, result, count)) return 0;
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    if (!vision_state_store_result(state, result)) memcpy(state, result.contents, bytes);
    return 1;
    }
}

static int metal_vision_layer_f32(float *state,
                                  const TensorVisionLayerF32 *layer,
                                  size_t rows, size_t hidden,
                                  size_t intermediate, float epsilon) {
    @autoreleasepool {
    if (!initialize_compute() || state == NULL || layer == NULL || rows == 0 ||
        hidden == 0 || intermediate == 0 || hidden % 64 != 0 ||
        rows > INT_MAX || hidden > INT_MAX || intermediate > INT_MAX ||
        rows > SIZE_MAX / hidden / sizeof(float) ||
        rows > SIZE_MAX / intermediate / sizeof(float) ||
        hidden > SIZE_MAX / hidden / sizeof(float) ||
        intermediate > SIZE_MAX / hidden / sizeof(float) ||
        intermediate > SIZE_MAX / intermediate / sizeof(float) ||
        rows > UINT32_MAX / intermediate) return 0;
    const size_t count = rows * hidden;
    if (count > UINT32_MAX) return 0;
    const size_t bytes = count * sizeof(float);
    const size_t intermediate_bytes = rows * intermediate * sizeof(float);
    id<MTLBuffer> state_input = vision_state_input_buffer(state, bytes);
    id<MTLBuffer> norm1 = output_buffer(bytes);
    id<MTLBuffer> q_raw = output_buffer(bytes), k_raw = output_buffer(bytes);
    id<MTLBuffer> v_raw = output_buffer(bytes), q = output_buffer(bytes);
    id<MTLBuffer> k = output_buffer(bytes), v = output_buffer(bytes);
    id<MTLBuffer> attention = output_buffer(bytes), out_raw = output_buffer(bytes);
    id<MTLBuffer> projected = output_buffer(bytes), after_attention = output_buffer(bytes);
    id<MTLBuffer> norm2 = output_buffer(bytes);
    id<MTLBuffer> fc1_raw = output_buffer(intermediate_bytes);
    id<MTLBuffer> activated = output_buffer(intermediate_bytes);
    id<MTLBuffer> fc2_raw = output_buffer(bytes), mlp = output_buffer(bytes);
    id<MTLBuffer> result = output_buffer(bytes);
    if (state_input == nil || norm1 == nil || q_raw == nil || k_raw == nil ||
        v_raw == nil || q == nil || k == nil || v == nil || attention == nil ||
        out_raw == nil || projected == nil || after_attention == nil || norm2 == nil ||
        fc1_raw == nil || activated == nil || fc2_raw == nil || mlp == nil || result == nil)
        return 0;
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    if (command == nil || !encode_norm_f32(command, state_input, layer->norm1_scale,
            layer->norm1_bias, norm1, rows, hidden, epsilon) ||
        !encode_linear_f32(command, norm1, layer->q_weight, q_raw, rows, hidden, hidden) ||
        !encode_linear_f32(command, norm1, layer->k_weight, k_raw, rows, hidden, hidden) ||
        !encode_linear_f32(command, norm1, layer->v_weight, v_raw, rows, hidden, hidden) ||
        !encode_bias_f32(command, q_raw, layer->q_bias, q, rows, hidden) ||
        !encode_bias_f32(command, k_raw, layer->k_bias, k, rows, hidden) ||
        !encode_bias_f32(command, v_raw, layer->v_bias, v, rows, hidden) ||
        !tensor_attention_metal_encode_tiled_f32((__bridge void *)command,
            (__bridge void *)q, (__bridge void *)k, (__bridge void *)v,
            (__bridge void *)attention, hidden / 64, rows, 1.0f / sqrtf(64.0f)) ||
        !encode_linear_f32(command, attention, layer->out_weight, out_raw,
                           rows, hidden, hidden) ||
        !encode_bias_f32(command, out_raw, layer->out_bias, projected, rows, hidden) ||
        !encode_residual_f32(command, state_input, projected, after_attention, count) ||
        !encode_norm_f32(command, after_attention, layer->norm2_scale,
                         layer->norm2_bias, norm2, rows, hidden, epsilon) ||
        !encode_linear_f32(command, norm2, layer->fc1_weight, fc1_raw,
                           rows, hidden, intermediate) ||
        !encode_bias_gelu_f32(command, fc1_raw, layer->fc1_bias, activated,
                              rows, intermediate) ||
        !encode_linear_f32(command, activated, layer->fc2_weight, fc2_raw,
                           rows, intermediate, hidden) ||
        !encode_bias_f32(command, fc2_raw, layer->fc2_bias, mlp, rows, hidden) ||
        !encode_residual_f32(command, after_attention, mlp, result, count)) return 0;
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    if (!vision_state_store_result(state, result)) memcpy(state, result.contents, bytes);
    return 1;
    }
}

typedef struct { const void *host_key; size_t bytes; } MetalVisionSequence;

static void *metal_vision_sequence_begin(const void *state, size_t rows,
                                         size_t hidden, size_t element_size) {
    @autoreleasepool {
    if (!initialize_compute() || state == NULL || rows == 0 || hidden == 0 ||
        rows > SIZE_MAX / hidden || rows * hidden > SIZE_MAX / element_size)
        return NULL;
    const size_t bytes = rows * hidden * element_size;
    id<MTLBuffer> buffer = input_buffer(state, bytes);
    MetalVisionSequence *sequence = malloc(sizeof(*sequence));
    if (buffer == nil || sequence == NULL) { free(sequence); return NULL; }
    if (vision_state_buffers == nil)
        vision_state_buffers = [NSMutableDictionary dictionary];
    NSValue *key = vision_state_key(state);
    @synchronized(vision_state_buffers) {
        if (vision_state_buffers[key] != nil) { free(sequence); return NULL; }
        vision_state_buffers[key] = buffer;
    }
    sequence->host_key = state;
    sequence->bytes = bytes;
    return sequence;
    }
}

static void *metal_vision_sequence_begin_f16(const uint16_t *state,
                                              size_t rows, size_t hidden) {
    return metal_vision_sequence_begin(state, rows, hidden, sizeof(uint16_t));
}

static void *metal_vision_sequence_begin_f32(const float *state,
                                              size_t rows, size_t hidden) {
    return metal_vision_sequence_begin(state, rows, hidden, sizeof(float));
}

static int metal_vision_sequence_layer_f16(void *context,
    const TensorVisionLayerF16 *layer, size_t rows, size_t hidden,
    size_t intermediate, float epsilon) {
    MetalVisionSequence *sequence = context;
    if (sequence == NULL || rows == 0 || hidden == 0 || rows > SIZE_MAX / hidden ||
        rows * hidden > SIZE_MAX / sizeof(uint16_t) ||
        rows * hidden * sizeof(uint16_t) != sequence->bytes) return 0;
    return metal_vision_layer_f16((uint16_t *)sequence->host_key, layer,
                                  rows, hidden, intermediate, epsilon);
}

static int metal_vision_sequence_layer_f32(void *context,
    const TensorVisionLayerF32 *layer, size_t rows, size_t hidden,
    size_t intermediate, float epsilon) {
    MetalVisionSequence *sequence = context;
    if (sequence == NULL || rows == 0 || hidden == 0 || rows > SIZE_MAX / hidden ||
        rows * hidden > SIZE_MAX / sizeof(float) ||
        rows * hidden * sizeof(float) != sequence->bytes) return 0;
    return metal_vision_layer_f32((float *)sequence->host_key, layer,
                                  rows, hidden, intermediate, epsilon);
}

static int metal_vision_sequence_finish(void *context, void *output,
                                        size_t element_size) {
    @autoreleasepool {
    MetalVisionSequence *sequence = context;
    if (sequence == NULL || output == NULL || sequence->bytes % element_size != 0)
        return 0;
    NSValue *key = vision_state_key(sequence->host_key);
    id<MTLBuffer> buffer = nil;
    @synchronized(vision_state_buffers) { buffer = vision_state_buffers[key]; }
    if (buffer == nil || buffer.length < sequence->bytes) return 0;
    memcpy(output, buffer.contents, sequence->bytes);
    return 1;
    }
}

static int metal_vision_sequence_finish_f16(void *context, uint16_t *output) {
    return metal_vision_sequence_finish(context, output, sizeof(uint16_t));
}

static int metal_vision_sequence_finish_f32(void *context, float *output) {
    return metal_vision_sequence_finish(context, output, sizeof(float));
}

static void metal_vision_sequence_release(void *context) {
    MetalVisionSequence *sequence = context;
    if (sequence == NULL) return;
    if (vision_state_buffers != nil) {
        NSValue *key = vision_state_key(sequence->host_key);
        @synchronized(vision_state_buffers) { [vision_state_buffers removeObjectForKey:key]; }
    }
    free(sequence);
}

static void metal_vision_sequence_release_f16(void *context) {
    metal_vision_sequence_release(context);
}

static void metal_vision_sequence_release_f32(void *context) {
    metal_vision_sequence_release(context);
}

static int metal_embedding(const float *table, const uint32_t *ids, float *output,
                           size_t rows, size_t width, size_t table_rows) {
    @autoreleasepool {
    if (!initialize_compute() ||
        table == NULL || ids == NULL || output == NULL ||
        rows == 0 || width == 0 || table_rows == 0 || rows > UINT32_MAX ||
        width > UINT32_MAX || table_rows > SIZE_MAX / width ||
        rows > SIZE_MAX / width / sizeof(float)) return 0;
    id<MTLBuffer> table_gpu = cached_weight_buffer(table, table_rows * width * sizeof(float));
    id<MTLBuffer> ids_gpu = input_buffer(ids, rows * sizeof(uint32_t));
    id<MTLBuffer> result_gpu = output_buffer(rows * width * sizeof(float));
    if (table_gpu == nil || ids_gpu == nil || result_gpu == nil) return 0;
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setBuffer:table_gpu offset:0 atIndex:0];
    [encoder setBuffer:ids_gpu offset:0 atIndex:1];
    [encoder setBuffer:result_gpu offset:0 atIndex:2];
    const uint32_t width32 = (uint32_t)width;
    [encoder setBytes:&width32 length:sizeof(width32) atIndex:3];
    dispatch_linear(encoder, embedding_pipeline, rows * width);
    [encoder endEncoding];
    return finish_command(command, output, result_gpu, rows * width * sizeof(float));
    }
}

static int metal_embedding_f16(const uint16_t *table, const uint32_t *ids,
                               uint16_t *output, size_t rows, size_t width,
                               size_t table_rows) {
    @autoreleasepool {
    if (!initialize_compute() || table == NULL || ids == NULL || output == NULL ||
        rows == 0 || width == 0 || table_rows == 0 || rows > UINT32_MAX ||
        width > UINT32_MAX || table_rows > SIZE_MAX / width ||
        rows > SIZE_MAX / width / sizeof(uint16_t)) return 0;
    id<MTLBuffer> table_gpu = cached_weight_buffer(table, table_rows * width * sizeof(uint16_t));
    id<MTLBuffer> ids_gpu = input_buffer(ids, rows * sizeof(uint32_t));
    id<MTLBuffer> output_gpu = output_buffer(rows * width * sizeof(uint16_t));
    if (table_gpu == nil || ids_gpu == nil || output_gpu == nil) return 0;
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    const uint32_t width32 = (uint32_t)width;
    [encoder setComputePipelineState:embedding_f16_pipeline];
    [encoder setBuffer:table_gpu offset:0 atIndex:0];
    [encoder setBuffer:ids_gpu offset:0 atIndex:1];
    [encoder setBuffer:output_gpu offset:0 atIndex:2];
    [encoder setBytes:&width32 length:sizeof(width32) atIndex:3];
    dispatch_linear(encoder, embedding_f16_pipeline, rows * width);
    [encoder endEncoding];
    [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, output_gpu.contents, rows * width * sizeof(uint16_t));
    return 1;
    }
}

static int metal_rms_norm(const float *input, const float *weight, float *output,
                          size_t rows, size_t width, float epsilon) {
    @autoreleasepool {
    if (!initialize_compute() ||
        input == NULL || weight == NULL || output == NULL ||
        rows == 0 || width == 0 || rows > UINT32_MAX || width > UINT32_MAX ||
        rows > SIZE_MAX / width / sizeof(float)) return 0;
    id<MTLBuffer> input_gpu = input_buffer(input, rows * width * sizeof(float));
    id<MTLBuffer> weight_gpu = cached_weight_buffer(weight, width * sizeof(float));
    id<MTLBuffer> result_gpu = output_buffer(rows * width * sizeof(float));
    if (input_gpu == nil || weight_gpu == nil || result_gpu == nil) return 0;
    NormParams params = {(uint32_t)rows, (uint32_t)width, epsilon};
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:rms_pipeline];
    [encoder setBuffer:input_gpu offset:0 atIndex:0];
    [encoder setBuffer:weight_gpu offset:0 atIndex:1];
    [encoder setBuffer:result_gpu offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    return finish_command(command, output, result_gpu, rows * width * sizeof(float));
    }
}

static int metal_norm_f16(id<MTLComputePipelineState> pipeline,
                          const uint16_t *input, const uint16_t *scale,
                          const uint16_t *bias, uint16_t *output,
                          size_t rows, size_t width, float epsilon, int layer_norm) {
    @autoreleasepool {
    if (!initialize_compute() || input == NULL || scale == NULL || output == NULL ||
        (layer_norm && bias == NULL) || rows == 0 || width == 0 ||
        rows > UINT32_MAX || width > UINT32_MAX ||
        rows > SIZE_MAX / width / sizeof(uint16_t)) return 0;
    const size_t bytes = rows * width * sizeof(uint16_t);
    id<MTLBuffer> input_gpu = input_buffer(input, bytes);
    id<MTLBuffer> scale_gpu = cached_weight_buffer(scale, width * sizeof(uint16_t));
    id<MTLBuffer> bias_gpu = layer_norm ? cached_weight_buffer(bias, width * sizeof(uint16_t)) : nil;
    id<MTLBuffer> output_gpu = output_buffer(bytes);
    if (input_gpu == nil || scale_gpu == nil || (layer_norm && bias_gpu == nil) || output_gpu == nil) return 0;
    NormParams params = {(uint32_t)rows, (uint32_t)width, epsilon};
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input_gpu offset:0 atIndex:0];
    [encoder setBuffer:scale_gpu offset:0 atIndex:1];
    if (layer_norm) [encoder setBuffer:bias_gpu offset:0 atIndex:2];
    [encoder setBuffer:output_gpu offset:0 atIndex:layer_norm ? 3 : 2];
    [encoder setBytes:&params length:sizeof(params) atIndex:layer_norm ? 4 : 3];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, output_gpu.contents, bytes);
    return 1;
    }
}

static int metal_rms_norm_f16(const uint16_t *input, const uint16_t *weight,
                              uint16_t *output, size_t rows, size_t width, float epsilon) {
    return metal_norm_f16(rms_f16_pipeline, input, weight, NULL, output, rows, width, epsilon, 0);
}

static int metal_layer_norm_f16(const uint16_t *input, const uint16_t *scale,
                                const uint16_t *bias, uint16_t *output,
                                size_t rows, size_t width, float epsilon) {
    return metal_norm_f16(layer_norm_f16_pipeline, input, scale, bias, output, rows, width, epsilon, 1);
}

static int metal_bias_add_f16(const uint16_t *input, const uint16_t *bias,
                              uint16_t *output, size_t rows, size_t width) {
    @autoreleasepool {
    if (!initialize_compute() || input == NULL || bias == NULL || output == NULL ||
        rows == 0 || width == 0 || rows > UINT32_MAX || width > UINT32_MAX ||
        rows > SIZE_MAX / width / sizeof(uint16_t)) return 0;
    const size_t bytes = rows * width * sizeof(uint16_t);
    id<MTLBuffer> in = input_buffer(input, bytes);
    id<MTLBuffer> b = cached_weight_buffer(bias, width * sizeof(uint16_t));
    id<MTLBuffer> out = output_buffer(bytes);
    if (in == nil || b == nil || out == nil) return 0;
    BiasParams params = {(uint32_t)width};
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:bias_add_f16_pipeline];
    [encoder setBuffer:in offset:0 atIndex:0]; [encoder setBuffer:b offset:0 atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2]; [encoder setBytes:&params length:sizeof(params) atIndex:3];
    dispatch_linear(encoder, bias_add_f16_pipeline, rows * width);
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, out.contents, bytes); return 1;
    }
}

static int metal_gelu_f16(const uint16_t *input, uint16_t *output, size_t count) {
    @autoreleasepool {
    if (!initialize_compute() || input == NULL || output == NULL || count == 0 ||
        count > SIZE_MAX / sizeof(uint16_t)) return 0;
    const size_t bytes = count * sizeof(uint16_t);
    id<MTLBuffer> in = input_buffer(input, bytes), out = output_buffer(bytes);
    if (in == nil || out == nil) return 0;
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:gelu_f16_pipeline];
    [encoder setBuffer:in offset:0 atIndex:0]; [encoder setBuffer:out offset:0 atIndex:1];
    dispatch_linear(encoder, gelu_f16_pipeline, count);
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, out.contents, bytes); return 1;
    }
}

static int metal_rms_norm_linear(const float *input, const float *norm_weight,
                                 float *normalized, const float *linear_weight,
                                 float *output, size_t rows, size_t input_width,
                                 size_t output_width, float epsilon) {
    @autoreleasepool {
    if (!initialize_compute() || input == NULL || norm_weight == NULL ||
        normalized == NULL || linear_weight == NULL || output == NULL ||
        rows == 0 || input_width == 0 || output_width == 0 ||
        rows > INT_MAX || input_width > INT_MAX || output_width > INT_MAX ||
        rows > SIZE_MAX / input_width / sizeof(float) ||
        output_width > SIZE_MAX / input_width / sizeof(float) ||
        rows > SIZE_MAX / output_width / sizeof(float) ||
        rows > UINT32_MAX || input_width > UINT32_MAX) return 0;

    const size_t input_bytes = rows * input_width * sizeof(float);
    const size_t output_bytes = rows * output_width * sizeof(float);
    id<MTLBuffer> input_gpu = input_buffer(input, input_bytes);
    id<MTLBuffer> norm_weight_gpu = cached_weight_buffer(
        norm_weight, input_width * sizeof(float));
    id<MTLBuffer> normalized_gpu = output_buffer(input_bytes);
    id<MTLBuffer> linear_weight_gpu = cached_weight_buffer(
        linear_weight, output_width * input_width * sizeof(float));
    id<MTLBuffer> output_gpu = output_buffer(output_bytes);
    if (input_gpu == nil || norm_weight_gpu == nil || normalized_gpu == nil ||
        linear_weight_gpu == nil || output_gpu == nil) return 0;

    MPSMatrixDescriptor *input_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:input_width
        rowBytes:input_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *weight_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:output_width columns:input_width
        rowBytes:input_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *output_descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows columns:output_width
        rowBytes:output_width * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrix *input_matrix = [[MPSMatrix alloc] initWithBuffer:normalized_gpu
        descriptor:input_descriptor];
    MPSMatrix *weight_matrix = [[MPSMatrix alloc] initWithBuffer:linear_weight_gpu
        descriptor:weight_descriptor];
    MPSMatrix *output_matrix = [[MPSMatrix alloc] initWithBuffer:output_gpu
        descriptor:output_descriptor];
    MPSMatrixMultiplication *operation = [[MPSMatrixMultiplication alloc]
        initWithDevice:compute_device transposeLeft:NO transposeRight:YES
        resultRows:rows resultColumns:output_width interiorColumns:input_width
        alpha:1.0 beta:0.0];
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = command == nil ? nil :
        [command computeCommandEncoder];
    if (input_matrix == nil || weight_matrix == nil || output_matrix == nil ||
        operation == nil || encoder == nil) return 0;

    NormParams norm_params = {(uint32_t)rows, (uint32_t)input_width, epsilon};
    [encoder setComputePipelineState:rms_pipeline];
    [encoder setBuffer:input_gpu offset:0 atIndex:0];
    [encoder setBuffer:norm_weight_gpu offset:0 atIndex:1];
    [encoder setBuffer:normalized_gpu offset:0 atIndex:2];
    [encoder setBytes:&norm_params length:sizeof(norm_params) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    [operation encodeToCommandBuffer:command leftMatrix:input_matrix
        rightMatrix:weight_matrix resultMatrix:output_matrix];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(normalized, normalized_gpu.contents, input_bytes);
    memcpy(output, output_gpu.contents, output_bytes);
    return 1;
    }
}

static int metal_layer_norm(const float *input, const float *scale, const float *bias,
                            float *output, size_t rows, size_t width, float epsilon) {
    @autoreleasepool {
    if (!initialize_compute() ||
        input == NULL || scale == NULL || bias == NULL ||
        output == NULL || rows == 0 || width == 0 || rows > UINT32_MAX ||
        width > UINT32_MAX || rows > SIZE_MAX / width / sizeof(float)) return 0;
    const size_t value_bytes = rows * width * sizeof(float);
    id<MTLBuffer> input_gpu = input_buffer(input, value_bytes);
    id<MTLBuffer> scale_gpu = cached_weight_buffer(scale, width * sizeof(float));
    id<MTLBuffer> bias_gpu = cached_weight_buffer(bias, width * sizeof(float));
    id<MTLBuffer> result_gpu = output_buffer(value_bytes);
    if (input_gpu == nil || scale_gpu == nil || bias_gpu == nil || result_gpu == nil) return 0;
    NormParams params = {(uint32_t)rows, (uint32_t)width, epsilon};
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:layer_norm_pipeline];
    [encoder setBuffer:input_gpu offset:0 atIndex:0];
    [encoder setBuffer:scale_gpu offset:0 atIndex:1];
    [encoder setBuffer:bias_gpu offset:0 atIndex:2];
    [encoder setBuffer:result_gpu offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    return finish_command(command, output, result_gpu, value_bytes);
    }
}

static int metal_bias_add(const float *input, const float *bias, float *output,
                          size_t rows, size_t width) {
    @autoreleasepool {
    if (!initialize_compute() ||
        input == NULL || bias == NULL || output == NULL ||
        rows == 0 || width == 0 || width > UINT32_MAX ||
        rows > SIZE_MAX / width / sizeof(float)) return 0;
    const size_t value_bytes = rows * width * sizeof(float);
    id<MTLBuffer> input_gpu = input_buffer(input, value_bytes);
    id<MTLBuffer> bias_gpu = cached_weight_buffer(bias, width * sizeof(float));
    id<MTLBuffer> result_gpu = output_buffer(value_bytes);
    if (input_gpu == nil || bias_gpu == nil || result_gpu == nil) return 0;
    BiasParams params = {(uint32_t)width};
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:bias_add_pipeline];
    [encoder setBuffer:input_gpu offset:0 atIndex:0];
    [encoder setBuffer:bias_gpu offset:0 atIndex:1];
    [encoder setBuffer:result_gpu offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    dispatch_linear(encoder, bias_add_pipeline, rows * width);
    [encoder endEncoding];
    return finish_command(command, output, result_gpu, value_bytes);
    }
}

static int metal_gelu(const float *input, float *output, size_t count) {
    @autoreleasepool {
    if (!initialize_compute() ||
        input == NULL || output == NULL || count == 0 ||
        count > SIZE_MAX / sizeof(float)) return 0;
    const size_t value_bytes = count * sizeof(float);
    id<MTLBuffer> input_gpu = input_buffer(input, value_bytes);
    id<MTLBuffer> result_gpu = output_buffer(value_bytes);
    if (input_gpu == nil || result_gpu == nil) return 0;
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:gelu_pipeline];
    [encoder setBuffer:input_gpu offset:0 atIndex:0];
    [encoder setBuffer:result_gpu offset:0 atIndex:1];
    dispatch_linear(encoder, gelu_pipeline, count);
    [encoder endEncoding];
    return finish_command(command, output, result_gpu, value_bytes);
    }
}

static int metal_rope(float *query, float *key, size_t sequence,
                      size_t query_heads, size_t kv_heads, size_t head_dim,
                      double theta) {
    @autoreleasepool {
    if (!initialize_compute() ||
        query == NULL || key == NULL || sequence == 0 ||
        query_heads == 0 || kv_heads == 0 || head_dim == 0 || head_dim % 2 != 0 ||
        sequence > UINT32_MAX || query_heads > UINT32_MAX || kv_heads > UINT32_MAX ||
        head_dim > UINT32_MAX) return 0;
    const size_t query_count = sequence * query_heads * head_dim;
    const size_t key_count = sequence * kv_heads * head_dim;
    id<MTLBuffer> query_gpu = input_buffer(query, query_count * sizeof(float));
    id<MTLBuffer> key_gpu = input_buffer(key, key_count * sizeof(float));
    id<MTLBuffer> query_result = output_buffer(query_count * sizeof(float));
    id<MTLBuffer> key_result = output_buffer(key_count * sizeof(float));
    if (query_gpu == nil || key_gpu == nil || query_result == nil || key_result == nil) return 0;
    RopeParams params = {(uint32_t)sequence, (uint32_t)query_heads,
        (uint32_t)kv_heads, (uint32_t)head_dim, (float)theta, 0};
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:rope_pipeline];
    [encoder setBuffer:query_gpu offset:0 atIndex:0];
    [encoder setBuffer:key_gpu offset:0 atIndex:1];
    [encoder setBuffer:query_result offset:0 atIndex:2];
    [encoder setBuffer:key_result offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    dispatch_linear(encoder, rope_pipeline, MAX(query_count, key_count));
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(query, query_result.contents, query_count * sizeof(float));
    memcpy(key, key_result.contents, key_count * sizeof(float));
    return 1;
    }
}

static int metal_rope_f16(uint16_t *query, uint16_t *key, size_t sequence,
                          size_t query_heads, size_t kv_heads, size_t head_dim,
                          double theta, size_t position_offset) {
    @autoreleasepool {
    if (!initialize_compute() || query == NULL || key == NULL || sequence == 0 ||
        query_heads == 0 || kv_heads == 0 || head_dim == 0 || head_dim % 2 != 0 ||
        sequence > UINT32_MAX || query_heads > UINT32_MAX || kv_heads > UINT32_MAX ||
        head_dim > UINT32_MAX || position_offset > UINT32_MAX ||
        sequence > SIZE_MAX / query_heads / head_dim ||
        sequence > SIZE_MAX / kv_heads / head_dim) return 0;
    const size_t qcount = sequence * query_heads * head_dim;
    const size_t kcount = sequence * kv_heads * head_dim;
    const size_t qbytes = qcount * sizeof(uint16_t), kbytes = kcount * sizeof(uint16_t);
    id<MTLBuffer> qin = input_buffer(query, qbytes), kin = input_buffer(key, kbytes);
    id<MTLBuffer> qout = output_buffer(qbytes), kout = output_buffer(kbytes);
    if (qin == nil || kin == nil || qout == nil || kout == nil) return 0;
    RopeParams params = {(uint32_t)sequence, (uint32_t)query_heads,
        (uint32_t)kv_heads, (uint32_t)head_dim, (float)theta, (uint32_t)position_offset};
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:rope_f16_pipeline];
    [encoder setBuffer:qin offset:0 atIndex:0]; [encoder setBuffer:kin offset:0 atIndex:1];
    [encoder setBuffer:qout offset:0 atIndex:2]; [encoder setBuffer:kout offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    dispatch_linear(encoder, rope_f16_pipeline, MAX(qcount, kcount));
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(query, qout.contents, qbytes); memcpy(key, kout.contents, kbytes);
    return 1;
    }
}

static int metal_binary_kernel(id<MTLComputePipelineState> pipeline,
                               const float *left, const float *right,
                               float *output, size_t count) {
    if (!initialize_compute() || left == NULL || right == NULL || output == NULL ||
        count == 0) return 0;
    id<MTLBuffer> left_gpu = input_buffer(left, count * sizeof(float));
    id<MTLBuffer> right_gpu = input_buffer(right, count * sizeof(float));
    id<MTLBuffer> output_gpu = output_buffer(count * sizeof(float));
    if (left_gpu == nil || right_gpu == nil || output_gpu == nil) return 0;
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setBuffer:left_gpu offset:0 atIndex:0];
    [encoder setBuffer:right_gpu offset:0 atIndex:1];
    [encoder setBuffer:output_gpu offset:0 atIndex:2];
    dispatch_linear(encoder, pipeline, count);
    [encoder endEncoding];
    return finish_command(command, output, output_gpu, count * sizeof(float));
}

static int metal_silu_multiply(const float *gate, const float *up,
                               float *output, size_t count) {
    @autoreleasepool {
        if (!initialize_compute()) return 0;
        return metal_binary_kernel(silu_pipeline, gate, up, output, count);
    }
}

static int metal_residual_add(const float *left, const float *right,
                              float *output, size_t count) {
    @autoreleasepool {
        if (!initialize_compute()) return 0;
        return metal_binary_kernel(residual_pipeline, left, right, output, count);
    }
}

static int metal_binary_f16(id<MTLComputePipelineState> pipeline,
                            const uint16_t *left, const uint16_t *right,
                            uint16_t *output, size_t count) {
    if (!initialize_compute() || left == NULL || right == NULL || output == NULL ||
        count == 0 || count > SIZE_MAX / sizeof(uint16_t)) return 0;
    const size_t bytes = count * sizeof(uint16_t);
    id<MTLBuffer> l = input_buffer(left, bytes), r = input_buffer(right, bytes), o = output_buffer(bytes);
    if (l == nil || r == nil || o == nil) return 0;
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:l offset:0 atIndex:0]; [encoder setBuffer:r offset:0 atIndex:1];
    [encoder setBuffer:o offset:0 atIndex:2];
    dispatch_linear(encoder, pipeline, count);
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, o.contents, bytes);
    return 1;
}

static int metal_silu_multiply_f16(const uint16_t *gate, const uint16_t *up,
                                  uint16_t *output, size_t count) {
    @autoreleasepool { return metal_binary_f16(silu_f16_pipeline, gate, up, output, count); }
}

static int metal_residual_add_f16(const uint16_t *left, const uint16_t *right,
                                  uint16_t *output, size_t count) {
    @autoreleasepool { return metal_binary_f16(residual_f16_pipeline, left, right, output, count); }
}

static int metal_argmax(const float *input, size_t count, uint32_t *result) {
    @autoreleasepool {
    if (!initialize_compute() ||
        input == NULL || result == NULL || count == 0 ||
        count > UINT32_MAX) return 0;
    id<MTLBuffer> input_gpu = input_buffer(input, count * sizeof(float));
    id<MTLBuffer> result_gpu = output_buffer(sizeof(uint32_t));
    if (input_gpu == nil || result_gpu == nil) return 0;
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:argmax_pipeline];
    [encoder setBuffer:input_gpu offset:0 atIndex:0];
    [encoder setBuffer:result_gpu offset:0 atIndex:1];
    const uint32_t count32 = (uint32_t)count;
    [encoder setBytes:&count32 length:sizeof(count32) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(result, result_gpu.contents, sizeof(*result));
    return 1;
    }
}

static int metal_argmax_f16(const uint16_t *input, size_t count, uint32_t *result) {
    @autoreleasepool {
    if (!initialize_compute() || input == NULL || result == NULL || count == 0 || count > UINT32_MAX) return 0;
    id<MTLBuffer> in = input_buffer(input, count * sizeof(uint16_t));
    id<MTLBuffer> out = output_buffer(sizeof(uint32_t));
    if (in == nil || out == nil) return 0;
    id<MTLCommandBuffer> command = [compute_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:argmax_f16_pipeline];
    [encoder setBuffer:in offset:0 atIndex:0]; [encoder setBuffer:out offset:0 atIndex:1];
    const uint32_t count32 = (uint32_t)count;
    [encoder setBytes:&count32 length:sizeof(count32) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(result, out.contents, sizeof(*result)); return 1;
    }
}

static void metal_release_cache(void) {
    @autoreleasepool {
    if (weight_buffers == nil) return;
    @synchronized(weight_buffers) {
        [weight_buffers removeAllObjects];
    }
    }
}

const TensorComputeBackend *tensor_compute_metal_backend(void) {
    static const TensorComputeBackend backend = {
        .name = "metal",
        .is_available = metal_available,
        .linear_f32 = metal_linear,
        .linear_f16 = metal_linear_f16,
        .embedding_f32 = metal_embedding,
        .embedding_f16 = metal_embedding_f16,
        .rms_norm_f32 = metal_rms_norm,
        .rms_norm_f16 = metal_rms_norm_f16,
        .rms_norm_linear_f32 = metal_rms_norm_linear,
        .layer_norm_f32 = metal_layer_norm,
        .layer_norm_f16 = metal_layer_norm_f16,
        .bias_add_f32 = metal_bias_add,
        .bias_add_f16 = metal_bias_add_f16,
        .gelu_f32 = metal_gelu,
        .gelu_f16 = metal_gelu_f16,
        .rope_f32 = metal_rope,
        .rope_f16 = metal_rope_f16,
        .silu_multiply_f32 = metal_silu_multiply,
        .silu_multiply_f16 = metal_silu_multiply_f16,
        .residual_add_f32 = metal_residual_add,
        .residual_add_f16 = metal_residual_add_f16,
        .argmax_f32 = metal_argmax,
        .argmax_f16 = metal_argmax_f16,
        .release_cache = metal_release_cache,
        .mlp_f32 = metal_mlp,
        .mlp_f16 = metal_mlp_f16,
        .qkv_f16 = metal_qkv_f16,
        .vision_layer_f16 = metal_vision_layer_f16,
        .vision_layer_f32 = metal_vision_layer_f32,
        .vision_sequence_begin_f16 = metal_vision_sequence_begin_f16,
        .vision_sequence_layer_f16 = metal_vision_sequence_layer_f16,
        .vision_sequence_finish_f16 = metal_vision_sequence_finish_f16,
        .vision_sequence_release_f16 = metal_vision_sequence_release_f16,
        .vision_sequence_begin_f32 = metal_vision_sequence_begin_f32,
        .vision_sequence_layer_f32 = metal_vision_sequence_layer_f32,
        .vision_sequence_finish_f32 = metal_vision_sequence_finish_f32,
        .vision_sequence_release_f32 = metal_vision_sequence_release_f32
    };
    return &backend;
}
