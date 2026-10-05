#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdio>
#include <cstring>
#include <mutex>

struct MpsNdArrayMatmulBench {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLBuffer> input;
  id<MTLBuffer> weight;
  id<MTLBuffer> output;
  MPSNDArrayMatrixMultiplication* kernel;
  NSUInteger rows;
  NSUInteger input_dim;
  NSUInteger output_dim;
};

static MPSNDArrayMatrixMultiplication* cached_mps_matmul_kernel(id<MTLDevice> device) {
  static std::mutex mutex;
  static MPSNDArrayMatrixMultiplication* kernel = nil;
  std::lock_guard<std::mutex> guard(mutex);
  if (kernel == nil) {
    kernel = [[MPSNDArrayMatrixMultiplication alloc] initWithDevice:device sourceCount:2];
  }
  return kernel;
}

static void write_error(char* error, size_t capacity, const char* message) {
  if (error != nullptr && capacity > 0) {
    std::snprintf(error, capacity, "%s", message);
  }
}

extern "C" void* fast_reasoning_mps_ndarray_matmul_create(
    const float* input,
    const float* weight,
    size_t rows,
    size_t input_dim,
    size_t output_dim,
    char* error,
    size_t error_capacity) {
  if (@available(macOS 15.0, *)) {
    @try {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) {
        write_error(error, error_capacity, "no default Metal device");
        return nullptr;
      }
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLBuffer> input_buffer = [device newBufferWithBytes:input
                                                       length:rows * input_dim * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
      id<MTLBuffer> weight_buffer = [device newBufferWithBytes:weight
                                                        length:output_dim * input_dim * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
      id<MTLBuffer> output_buffer = [device newBufferWithLength:rows * output_dim * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
      if (queue == nil || input_buffer == nil || weight_buffer == nil || output_buffer == nil) {
        write_error(error, error_capacity, "failed to allocate Metal queue or buffer");
        return nullptr;
      }
      auto* context = new MpsNdArrayMatmulBench{
          device,
          queue,
          input_buffer,
          weight_buffer,
          output_buffer,
          [[MPSNDArrayMatrixMultiplication alloc] initWithDevice:device sourceCount:2],
          rows,
          input_dim,
          output_dim};
      if (context->kernel == nil) {
        delete context;
        write_error(error, error_capacity, "failed to create MPSNDArray matmul kernel");
        return nullptr;
      }
      return context;
    } @catch (NSException* exception) {
      write_error(error, error_capacity, exception.reason.UTF8String ?: "MPS initialization exception");
      return nullptr;
    }
  }
  write_error(error, error_capacity, "MPSNDArray direct encoding requires macOS 15 or newer");
  return nullptr;
}

extern "C" int fast_reasoning_mps_ndarray_matmul_encode(
    void* encoder_handle,
    void* command_buffer_handle,
    void* input_handle,
    void* weight_handle,
    void* output_handle,
    size_t input_offset_bytes,
    size_t weight_offset_bytes,
    size_t rows,
    size_t input_dim,
    size_t output_dim,
    char* error,
    size_t error_capacity) {
  if (@available(macOS 15.0, *)) {
    if (encoder_handle == nullptr || command_buffer_handle == nullptr || input_handle == nullptr ||
        weight_handle == nullptr || output_handle == nullptr || rows == 0 || input_dim == 0 ||
        output_dim == 0) {
      write_error(error, error_capacity, "invalid Candle Metal handle or matmul shape");
      return 0;
    }
    @try {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            (__bridge id<MTLComputeCommandEncoder>)encoder_handle;
        id<MTLCommandBuffer> command_buffer =
            (__bridge id<MTLCommandBuffer>)command_buffer_handle;
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)input_handle;
        id<MTLBuffer> weight_buffer = (__bridge id<MTLBuffer>)weight_handle;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)output_handle;
        id<MTLDevice> device = command_buffer.commandQueue.device;
        if (device == nil) {
          write_error(error, error_capacity, "Candle command buffer has no Metal device");
          return 0;
        }
        MPSNDArrayMatrixMultiplication* kernel = cached_mps_matmul_kernel(device);
        if (kernel == nil) {
          write_error(error, error_capacity, "failed to create cached MPSNDArray matmul kernel");
          return 0;
        }

        MPSNDArrayDescriptor* input_descriptor =
            [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                    shape:@[ @(rows), @(input_dim) ]];
        MPSNDArrayDescriptor* weight_descriptor =
            [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                    shape:@[ @(output_dim), @(input_dim) ]];
        weight_descriptor.preferPackedRows = YES;
        [weight_descriptor transposeDimension:0 withDimension:1];
        MPSNDArrayDescriptor* output_descriptor =
            [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                    shape:@[ @(rows), @(output_dim) ]];
        MPSNDArray* input_array = [[MPSNDArray alloc] initWithBuffer:input_buffer
                                                              offset:input_offset_bytes
                                                          descriptor:input_descriptor];
        MPSNDArray* weight_array = [[MPSNDArray alloc] initWithBuffer:weight_buffer
                                                               offset:weight_offset_bytes
                                                           descriptor:weight_descriptor];
        MPSNDArray* output_array = [[MPSNDArray alloc] initWithBuffer:output_buffer
                                                               offset:0
                                                           descriptor:output_descriptor];
        if (input_array == nil || weight_array == nil || output_array == nil) {
          write_error(error, error_capacity, "failed to create MPSNDArray views over Candle buffers");
          return 0;
        }
        [kernel encodeToCommandEncoder:encoder
                         commandBuffer:command_buffer
                          sourceArrays:@[ input_array, weight_array ]
                      destinationArray:output_array];
        return 1;
      }
    } @catch (NSException* exception) {
      write_error(error, error_capacity, exception.reason.UTF8String ?: "MPSNDArray encode exception");
      return 0;
    }
  }
  write_error(error, error_capacity, "MPSNDArray direct encoding requires macOS 15 or newer");
  return 0;
}

extern "C" int fast_reasoning_mps_ndarray_matmul_run(
    void* opaque_context,
    size_t iterations,
    int synchronize_each,
    char* error,
    size_t error_capacity) {
  auto* context = static_cast<MpsNdArrayMatmulBench*>(opaque_context);
  if (context == nullptr || iterations == 0) {
    write_error(error, error_capacity, "invalid MPSNDArray benchmark context or iteration count");
    return 0;
  }
  @try {
    if (synchronize_each == 2) {
      // Model Candle custom-op behavior more closely: each operation obtains
      // its own compute encoder, but all encoders stay in one command buffer.
      id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
      if (command_buffer == nil) {
        write_error(error, error_capacity, "failed to create Metal command buffer");
        return 0;
      }
      for (size_t iteration = 0; iteration < iterations; ++iteration) {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
          write_error(error, error_capacity, "failed to create Metal compute encoder");
          return 0;
        }
        @autoreleasepool {
          MPSNDArrayDescriptor* input_descriptor =
              [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                      shape:@[ @(context->rows), @(context->input_dim) ]];
          MPSNDArrayDescriptor* weight_descriptor =
              [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                      shape:@[ @(context->output_dim), @(context->input_dim) ]];
          weight_descriptor.preferPackedRows = YES;
          [weight_descriptor transposeDimension:0 withDimension:1];
          MPSNDArrayDescriptor* output_descriptor =
              [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                      shape:@[ @(context->rows), @(context->output_dim) ]];
          MPSNDArray* input_array = [[MPSNDArray alloc] initWithBuffer:context->input
                                                                offset:0
                                                            descriptor:input_descriptor];
          MPSNDArray* weight_array = [[MPSNDArray alloc] initWithBuffer:context->weight
                                                                 offset:0
                                                             descriptor:weight_descriptor];
          MPSNDArray* output_array = [[MPSNDArray alloc] initWithBuffer:context->output
                                                                 offset:0
                                                             descriptor:output_descriptor];
          if (input_array == nil || weight_array == nil || output_array == nil) {
            [encoder endEncoding];
            write_error(error, error_capacity, "failed to create MPSNDArray buffer views");
            return 0;
          }
          [context->kernel encodeToCommandEncoder:encoder
                                    commandBuffer:command_buffer
                                     sourceArrays:@[ input_array, weight_array ]
                                 destinationArray:output_array];
        }
        [encoder endEncoding];
      }
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        write_error(error, error_capacity,
                    command_buffer.error.localizedDescription.UTF8String ?: "MPS command failed");
        return 0;
      }
      return 1;
    }

    if (synchronize_each == 1) {
      // Approximate the integration cost of a synchronous per-op bridge: every
      // matmul gets its own command buffer and the caller waits before returning.
      for (size_t iteration = 0; iteration < iterations; ++iteration) {
        id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
          write_error(error, error_capacity, "failed to create Metal command buffer or encoder");
          return 0;
        }
        @autoreleasepool {
          MPSNDArrayDescriptor* input_descriptor =
              [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                      shape:@[ @(context->rows), @(context->input_dim) ]];
          MPSNDArrayDescriptor* weight_descriptor =
              [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                      shape:@[ @(context->output_dim), @(context->input_dim) ]];
          weight_descriptor.preferPackedRows = YES;
          [weight_descriptor transposeDimension:0 withDimension:1];
          MPSNDArrayDescriptor* output_descriptor =
              [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                      shape:@[ @(context->rows), @(context->output_dim) ]];
          MPSNDArray* input_array = [[MPSNDArray alloc] initWithBuffer:context->input
                                                                offset:0
                                                            descriptor:input_descriptor];
          MPSNDArray* weight_array = [[MPSNDArray alloc] initWithBuffer:context->weight
                                                                 offset:0
                                                             descriptor:weight_descriptor];
          MPSNDArray* output_array = [[MPSNDArray alloc] initWithBuffer:context->output
                                                                 offset:0
                                                             descriptor:output_descriptor];
          if (input_array == nil || weight_array == nil || output_array == nil) {
            [encoder endEncoding];
            write_error(error, error_capacity, "failed to create MPSNDArray buffer views");
            return 0;
          }
          [context->kernel encodeToCommandEncoder:encoder
                                    commandBuffer:command_buffer
                                     sourceArrays:@[ input_array, weight_array ]
                                 destinationArray:output_array];
        }
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
          write_error(error, error_capacity,
                      command_buffer.error.localizedDescription.UTF8String ?: "MPS command failed");
          return 0;
        }
      }
      return 1;
    }

    id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
      write_error(error, error_capacity, "failed to create Metal command buffer or encoder");
      return 0;
    }
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
      @autoreleasepool {
        MPSNDArrayDescriptor* input_descriptor =
            [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                    shape:@[ @(context->rows), @(context->input_dim) ]];
        MPSNDArrayDescriptor* weight_descriptor =
            [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                    shape:@[ @(context->output_dim), @(context->input_dim) ]];
        weight_descriptor.preferPackedRows = YES;
        [weight_descriptor transposeDimension:0 withDimension:1];
        MPSNDArrayDescriptor* output_descriptor =
            [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                    shape:@[ @(context->rows), @(context->output_dim) ]];

        MPSNDArray* input_array = [[MPSNDArray alloc] initWithBuffer:context->input
                                                              offset:0
                                                          descriptor:input_descriptor];
        MPSNDArray* weight_array = [[MPSNDArray alloc] initWithBuffer:context->weight
                                                               offset:0
                                                           descriptor:weight_descriptor];
        MPSNDArray* output_array = [[MPSNDArray alloc] initWithBuffer:context->output
                                                               offset:0
                                                           descriptor:output_descriptor];
        if (input_array == nil || weight_array == nil || output_array == nil) {
          [encoder endEncoding];
          write_error(error, error_capacity, "failed to create MPSNDArray buffer views");
          return 0;
        }
        [context->kernel encodeToCommandEncoder:encoder
                                  commandBuffer:command_buffer
                                   sourceArrays:@[ input_array, weight_array ]
                               destinationArray:output_array];
      }
    }
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      write_error(error, error_capacity, command_buffer.error.localizedDescription.UTF8String ?: "MPS command failed");
      return 0;
    }
    return 1;
  } @catch (NSException* exception) {
    write_error(error, error_capacity, exception.reason.UTF8String ?: "MPS matmul exception");
    return 0;
  }
}

extern "C" int fast_reasoning_mps_ndarray_matmul_copy_output(
    void* opaque_context,
    float* output,
    size_t output_elements,
    char* error,
    size_t error_capacity) {
  auto* context = static_cast<MpsNdArrayMatmulBench*>(opaque_context);
  const size_t expected = context == nullptr ? 0 : context->rows * context->output_dim;
  if (context == nullptr || output == nullptr || output_elements != expected) {
    write_error(error, error_capacity, "invalid output buffer or element count");
    return 0;
  }
  std::memcpy(output, context->output.contents, expected * sizeof(float));
  return 1;
}

extern "C" void fast_reasoning_mps_ndarray_matmul_destroy(void* opaque_context) {
  delete static_cast<MpsNdArrayMatmulBench*>(opaque_context);
}
