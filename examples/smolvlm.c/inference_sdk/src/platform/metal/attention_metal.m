#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "attention_metal.h"
#include "metal_library.h"
#include "timing.h"

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    uint32_t batch;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t query_length;
    uint32_t kv_length;
    uint32_t head_dim;
    uint32_t value_dim;
    uint32_t causal_offset;
    uint32_t causal;
    uint32_t has_mask;
    float scale;
} SdpaParams;

typedef struct {
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t kv_length;
    uint32_t kv_capacity;
    uint32_t head_dim;
    float scale;
} DecodeSdpaParams;

typedef struct {
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t position;
    float theta;
} DecodeRopeParams;

@interface DecodeCacheEntry : NSObject
@property(nonatomic, strong) id<MTLBuffer> key_buffer;
@property(nonatomic, strong) id<MTLBuffer> value_buffer;
@property(nonatomic) const void *value_pointer;
@property(nonatomic) size_t valid_length;
@property(nonatomic) size_t capacity;
@property(nonatomic) size_t hidden_size;
@property(nonatomic) size_t intermediate_size;
@property(nonatomic, strong) id<MTLBuffer> norm_buffer;
@property(nonatomic, strong) id<MTLBuffer> qkv_weight_buffer;
@property(nonatomic, strong) id<MTLBuffer> qkv_buffer;
@property(nonatomic, strong) id<MTLBuffer> gate_up_weight_buffer;
@property(nonatomic) const void *q_weight_pointer;
@property(nonatomic) const void *k_weight_pointer;
@property(nonatomic) const void *v_weight_pointer;
@property(nonatomic) const void *gate_weight_pointer;
@property(nonatomic) const void *up_weight_pointer;
@property(nonatomic, strong) id<MTLBuffer> rotated_query_buffer;
@property(nonatomic, strong) id<MTLBuffer> attention_buffer;
@property(nonatomic, strong) id<MTLBuffer> attention_partials_buffer;
@property(nonatomic, strong) id<MTLBuffer> attention_sums_buffer;
@property(nonatomic, strong) id<MTLBuffer> attention_maxs_buffer;
@property(nonatomic, strong) id<MTLBuffer> residual_buffer;
@property(nonatomic, strong) id<MTLBuffer> post_norm_buffer;
@property(nonatomic, strong) id<MTLBuffer> activated_buffer;
@property(nonatomic, strong) id<MTLBuffer> final_norm_buffer;
@property(nonatomic, strong) id<MTLBuffer> logits_buffer;
@property(nonatomic, strong) id<MTLBuffer> token_buffer;
@property(nonatomic) double input_norm_encode_ms;
@property(nonatomic) double qkv_gemv_encode_ms;
@property(nonatomic) double rope_kvcache_encode_ms;
@property(nonatomic) double attention_encode_ms;
@property(nonatomic) double output_gemv_encode_ms;
@property(nonatomic) double post_norm_encode_ms;
@property(nonatomic) double mlp_gate_up_encode_ms;
@property(nonatomic) double mlp_down_encode_ms;
@end

@implementation DecodeCacheEntry
@end

static id<MTLDevice> cached_device;
static id<MTLCommandQueue> cached_queue;
static id<MTLComputePipelineState> cached_online_pipeline;
static id<MTLComputePipelineState> cached_online_f16_pipeline;
static id<MTLComputePipelineState> cached_tiled_vision_pipeline;
static id<MTLComputePipelineState> cached_tiled_vision_f16_pipeline;
static id<MTLComputePipelineState> cached_decode_pipeline;
static id<MTLComputePipelineState> cached_decode_partial_pipeline;
static id<MTLComputePipelineState> cached_decode_reduce_pipeline;
static id<MTLComputePipelineState> cached_residual_pipeline;
static id<MTLComputePipelineState> cached_rope_cache_pipeline;
static id<MTLComputePipelineState> cached_rms_pipeline;
static id<MTLComputePipelineState> cached_argmax_pipeline;
static id<MTLComputePipelineState> cached_embedding_pipeline;
static id<MTLComputePipelineState> cached_gemv_pipeline;
static id<MTLComputePipelineState> cached_gemv_residual_pipeline;
static id<MTLComputePipelineState> cached_gemv_silu_pipeline;
static id<MTLComputePipelineState> cached_gemv_4simd_pipeline;
static id<MTLComputePipelineState> cached_gemv_residual_4simd_pipeline;
static id<MTLComputePipelineState> cached_gemv_silu_4simd_pipeline;
static id<MTLComputePipelineState> cached_decode_f16_pipeline;
static id<MTLComputePipelineState> cached_decode_partial_f16_pipeline;
static id<MTLComputePipelineState> cached_decode_reduce_f16_pipeline;
static id<MTLComputePipelineState> cached_rope_cache_f16_pipeline;
static id<MTLComputePipelineState> cached_gemv_f16_pipeline;
static id<MTLComputePipelineState> cached_gemv_residual_f16_pipeline;
static id<MTLComputePipelineState> cached_gemv_silu_f16_pipeline;
static id<MTLComputePipelineState> cached_embedding_f16_pipeline;
static id<MTLComputePipelineState> cached_rms_f16_pipeline;
static id<MTLComputePipelineState> cached_argmax_f16_pipeline;
static id<MTLLibrary> cached_tensor_library;
static NSMutableDictionary<NSValue *, DecodeCacheEntry *> *decode_cache;
static NSMutableDictionary<NSValue *, id<MTLBuffer>> *decode_weight_buffers;
static id<MTLBuffer> decode_hidden_a;
static id<MTLBuffer> decode_hidden_b;

static id<MTLBuffer> decode_weight_buffer(const void *weight, size_t byte_count);
static id<MTLBuffer> decode_weight_buffer_f16(const uint16_t *weight, size_t byte_count);
static int initialization_attempted;
static int initialization_succeeded;
static int reported_vision_sdpa_path;
static int reported_vision_sdpa_f16_path;

static int checked_product(size_t left, size_t right, size_t *result) {
    if (right != 0 && left > SIZE_MAX / right) return 0;
    *result = left * right;
    return 1;
}

static int initialize_metal(void) {
    @synchronized([NSProcessInfo class]) {
        if (initialization_attempted) return initialization_succeeded;
        initialization_attempted = 1;
        cached_device = MTLCreateSystemDefaultDevice();
        if (cached_device == nil) return 0;
        cached_queue = [cached_device newCommandQueue];
        if (cached_queue == nil) return 0;
        const char *configured_path = getenv("TENSOR_METALLIB_PATH");
        NSString *library_path = configured_path == NULL ?
            @"build/attention.metallib" :
            [NSString stringWithUTF8String:configured_path];
        NSError *error = nil;
        id<MTLLibrary> library = metal_load_library(
            cached_device, "attention", library_path, &error);
        if (library == nil) return 0;
        id<MTLFunction> online_function = [library newFunctionWithName:@"sdpa_online_f32"];
        id<MTLFunction> online_f16_function = [library newFunctionWithName:@"sdpa_online_f16"];
        if (online_function == nil)
            return 0;
        cached_online_pipeline = [cached_device
            newComputePipelineStateWithFunction:online_function error:&error];
        if (online_f16_function == nil) return 0;
        cached_online_f16_pipeline = [cached_device
            newComputePipelineStateWithFunction:online_f16_function error:&error];
        MTLFunctionConstantValues *vision_constants = [MTLFunctionConstantValues new];
        BOOL align_query = YES, align_key = YES, has_mask = NO, causal = NO;
        [vision_constants setConstantValue:&align_query type:MTLDataTypeBool atIndex:200];
        [vision_constants setConstantValue:&align_key type:MTLDataTypeBool atIndex:201];
        [vision_constants setConstantValue:&has_mask type:MTLDataTypeBool atIndex:300];
        [vision_constants setConstantValue:&causal type:MTLDataTypeBool atIndex:301];
        id<MTLFunction> tiled_vision_function = [library
            newFunctionWithName:@"steel_attention_float32_bq32_bk32_bd64_wm4_wn1_maskfloat32"
            constantValues:vision_constants error:&error];
        if (tiled_vision_function != nil)
            cached_tiled_vision_pipeline = [cached_device
                newComputePipelineStateWithFunction:tiled_vision_function error:&error];
        id<MTLFunction> tiled_vision_f16_function = [library
            newFunctionWithName:@"steel_attention_float16_bq32_bk32_bd64_wm4_wn1_maskfloat16"
            constantValues:vision_constants error:&error];
        if (tiled_vision_f16_function != nil)
            cached_tiled_vision_f16_pipeline = [cached_device
                newComputePipelineStateWithFunction:tiled_vision_f16_function error:&error];
        id<MTLFunction> decode_function = [library newFunctionWithName:@"sdpa_decode_cache_f32"];
        if (decode_function != nil)
            cached_decode_pipeline = [cached_device
                newComputePipelineStateWithFunction:decode_function error:&error];
        id<MTLFunction> decode_partial_function =
            [library newFunctionWithName:@"sdpa_decode_cache_partial_f32"];
        if (decode_partial_function != nil)
            cached_decode_partial_pipeline = [cached_device
                newComputePipelineStateWithFunction:decode_partial_function error:&error];
        id<MTLFunction> decode_reduce_function =
            [library newFunctionWithName:@"sdpa_decode_cache_reduce_f32"];
        if (decode_reduce_function != nil)
            cached_decode_reduce_pipeline = [cached_device
                newComputePipelineStateWithFunction:decode_reduce_function error:&error];
        id<MTLFunction> residual_function = [library newFunctionWithName:@"sdpa_residual_add_f32"];
        if (residual_function != nil)
            cached_residual_pipeline = [cached_device
                newComputePipelineStateWithFunction:residual_function error:&error];
        id<MTLFunction> gemv_function = [library newFunctionWithName:@"tensor_gemv_f32"];
        if (gemv_function != nil)
            cached_gemv_pipeline = [cached_device
                newComputePipelineStateWithFunction:gemv_function error:&error];
        id<MTLFunction> gemv_residual_function = [library newFunctionWithName:@"tensor_gemv_residual_f32"];
        if (gemv_residual_function != nil)
            cached_gemv_residual_pipeline = [cached_device
                newComputePipelineStateWithFunction:gemv_residual_function error:&error];
        id<MTLFunction> gemv_silu_function = [library newFunctionWithName:@"tensor_gemv_silu_multiply_f32"];
        if (gemv_silu_function != nil)
            cached_gemv_silu_pipeline = [cached_device
                newComputePipelineStateWithFunction:gemv_silu_function error:&error];
        id<MTLFunction> gemv_f16 = [library newFunctionWithName:@"tensor_gemv_f16"];
        id<MTLFunction> gemv_residual_f16 = [library newFunctionWithName:@"tensor_gemv_residual_f16"];
        id<MTLFunction> gemv_silu_f16 = [library newFunctionWithName:@"tensor_gemv_silu_multiply_f16"];
        id<MTLFunction> decode_f16 = [library newFunctionWithName:@"sdpa_decode_cache_f16"];
        id<MTLFunction> decode_partial_f16 = [library newFunctionWithName:@"sdpa_decode_cache_partial_f16"];
        id<MTLFunction> decode_reduce_f16 = [library newFunctionWithName:@"sdpa_decode_cache_reduce_f16"];
        id<MTLFunction> rope_f16 = [library newFunctionWithName:@"sdpa_decode_rope_cache_f16"];
        if (gemv_f16 != nil) cached_gemv_f16_pipeline = [cached_device newComputePipelineStateWithFunction:gemv_f16 error:&error];
        if (gemv_residual_f16 != nil) cached_gemv_residual_f16_pipeline = [cached_device newComputePipelineStateWithFunction:gemv_residual_f16 error:&error];
        if (gemv_silu_f16 != nil) cached_gemv_silu_f16_pipeline = [cached_device newComputePipelineStateWithFunction:gemv_silu_f16 error:&error];
        if (decode_f16 != nil) cached_decode_f16_pipeline = [cached_device newComputePipelineStateWithFunction:decode_f16 error:&error];
        if (decode_partial_f16 != nil) cached_decode_partial_f16_pipeline = [cached_device newComputePipelineStateWithFunction:decode_partial_f16 error:&error];
        if (decode_reduce_f16 != nil) cached_decode_reduce_f16_pipeline = [cached_device newComputePipelineStateWithFunction:decode_reduce_f16 error:&error];
        if (rope_f16 != nil) cached_rope_cache_f16_pipeline = [cached_device newComputePipelineStateWithFunction:rope_f16 error:&error];
        id<MTLFunction> gemv_4simd_function =
            [library newFunctionWithName:@"tensor_gemv_4simd_f32"];
        if (gemv_4simd_function != nil)
            cached_gemv_4simd_pipeline = [cached_device
                newComputePipelineStateWithFunction:gemv_4simd_function error:&error];
        id<MTLFunction> gemv_residual_4simd_function =
            [library newFunctionWithName:@"tensor_gemv_residual_4simd_f32"];
        if (gemv_residual_4simd_function != nil)
            cached_gemv_residual_4simd_pipeline = [cached_device
                newComputePipelineStateWithFunction:gemv_residual_4simd_function error:&error];
        id<MTLFunction> gemv_silu_4simd_function =
            [library newFunctionWithName:@"tensor_gemv_silu_multiply_4simd_f32"];
        if (gemv_silu_4simd_function != nil)
            cached_gemv_silu_4simd_pipeline = [cached_device
                newComputePipelineStateWithFunction:gemv_silu_4simd_function error:&error];
        const char *tensor_path = getenv("TENSOR_KERNELS_METALLIB_PATH");
        NSString *tensor_library_path = tensor_path == NULL ?
            @"build/tensor_kernels.metallib" :
            [NSString stringWithUTF8String:tensor_path];
        cached_tensor_library = metal_load_library(
            cached_device, "tensor", tensor_library_path, &error);
        if (cached_tensor_library != nil) {
            id<MTLFunction> rope_cache = [library newFunctionWithName:@"sdpa_decode_rope_cache_f32"];
            id<MTLFunction> rms = [cached_tensor_library newFunctionWithName:@"tensor_rms_norm_f32"];
            id<MTLFunction> argmax = [cached_tensor_library newFunctionWithName:@"tensor_argmax_f32"];
            id<MTLFunction> embedding = [cached_tensor_library newFunctionWithName:@"tensor_embedding_f32"];
            id<MTLFunction> rms_f16 = [cached_tensor_library newFunctionWithName:@"tensor_rms_norm_f16"];
            id<MTLFunction> argmax_f16 = [cached_tensor_library newFunctionWithName:@"tensor_argmax_f16"];
            id<MTLFunction> embedding_f16 = [cached_tensor_library newFunctionWithName:@"tensor_embedding_f16"];
            if (rope_cache != nil)
                cached_rope_cache_pipeline = [cached_device
                    newComputePipelineStateWithFunction:rope_cache error:&error];
            if (rms != nil)
                cached_rms_pipeline = [cached_device
                    newComputePipelineStateWithFunction:rms error:&error];
            if (argmax != nil)
                cached_argmax_pipeline = [cached_device
                    newComputePipelineStateWithFunction:argmax error:&error];
            if (embedding != nil)
                cached_embedding_pipeline = [cached_device
                    newComputePipelineStateWithFunction:embedding error:&error];
            if (rms_f16 != nil) cached_rms_f16_pipeline = [cached_device newComputePipelineStateWithFunction:rms_f16 error:&error];
            if (argmax_f16 != nil) cached_argmax_f16_pipeline = [cached_device newComputePipelineStateWithFunction:argmax_f16 error:&error];
            if (embedding_f16 != nil) cached_embedding_f16_pipeline = [cached_device newComputePipelineStateWithFunction:embedding_f16 error:&error];
        }
        decode_cache = [NSMutableDictionary dictionary];
        decode_weight_buffers = [NSMutableDictionary dictionary];
        initialization_succeeded = cached_online_pipeline != nil && cached_online_f16_pipeline != nil;
        return initialization_succeeded;
    }
}

int tensor_attention_metal_decode_sequence_available(void) {
    return initialize_metal() && cached_decode_pipeline != nil &&
        cached_decode_partial_pipeline != nil && cached_decode_reduce_pipeline != nil &&
        cached_rope_cache_pipeline != nil && cached_rms_pipeline != nil &&
        cached_argmax_pipeline != nil && cached_embedding_pipeline != nil &&
        cached_gemv_pipeline != nil && cached_gemv_residual_pipeline != nil &&
        cached_gemv_silu_pipeline != nil;
}

static int decode_gemv_uses_4simd(void) {
    const char *variant = getenv("TENSOR_METAL_DECODE_GEMV");
    return variant != NULL && strcmp(variant, "4simd") == 0 &&
        cached_gemv_4simd_pipeline != nil &&
        cached_gemv_residual_4simd_pipeline != nil &&
        cached_gemv_silu_4simd_pipeline != nil;
}

static id<MTLBuffer> decode_scratch_buffer(size_t bytes) {
    return bytes == 0 ? nil : [cached_device newBufferWithLength:bytes
        options:MTLResourceStorageModeShared];
}

static int prepare_decode_entry(DecodeCacheEntry *entry, size_t hidden,
                                size_t intermediate, size_t capacity,
                                size_t kv_width, size_t vocabulary) {
    if (entry.hidden_size == hidden && entry.intermediate_size == intermediate &&
        entry.capacity == capacity && entry.norm_buffer != nil) return 1;
    entry.hidden_size = hidden;
    entry.intermediate_size = intermediate;
    entry.capacity = capacity;
    entry.valid_length = 0;
    size_t hidden_bytes = hidden * sizeof(float);
    size_t cache_bytes = capacity * kv_width * sizeof(float);
    entry.key_buffer = decode_scratch_buffer(cache_bytes);
    entry.value_buffer = decode_scratch_buffer(cache_bytes);
    entry.norm_buffer = decode_scratch_buffer(hidden_bytes);
    entry.rotated_query_buffer = decode_scratch_buffer(hidden_bytes);
    entry.attention_buffer = decode_scratch_buffer(hidden_bytes);
    entry.attention_partials_buffer = decode_scratch_buffer(32 * hidden_bytes);
    entry.attention_sums_buffer = decode_scratch_buffer(32 * hidden * sizeof(float));
    entry.attention_maxs_buffer = decode_scratch_buffer(32 * hidden * sizeof(float));
    entry.residual_buffer = decode_scratch_buffer(hidden_bytes);
    entry.post_norm_buffer = decode_scratch_buffer(hidden_bytes);
    entry.activated_buffer = decode_scratch_buffer(intermediate * sizeof(float));
    entry.final_norm_buffer = decode_scratch_buffer(hidden_bytes);
    entry.logits_buffer = decode_scratch_buffer(vocabulary * sizeof(float));
    entry.token_buffer = decode_scratch_buffer(sizeof(uint32_t));
    return entry.key_buffer != nil && entry.value_buffer != nil &&
        entry.norm_buffer != nil && entry.rotated_query_buffer != nil &&
        entry.attention_buffer != nil && entry.residual_buffer != nil &&
        entry.attention_partials_buffer != nil && entry.attention_sums_buffer != nil &&
        entry.attention_maxs_buffer != nil &&
        entry.post_norm_buffer != nil && entry.activated_buffer != nil &&
        entry.final_norm_buffer != nil &&
        entry.logits_buffer != nil && entry.token_buffer != nil;
}

static void encode_decode_rms(id<MTLCommandBuffer> command,
                              id<MTLBuffer> input, const float *weight,
                              id<MTLBuffer> output, size_t width, float epsilon) {
    id<MTLBuffer> weight_buffer = decode_weight_buffer(weight, width * sizeof(float));
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    struct { uint32_t rows; uint32_t width; float epsilon; } params =
        {1, (uint32_t)width, epsilon};
    [encoder setComputePipelineState:cached_rms_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:weight_buffer offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
}

static int prepare_decode_packed_weights(DecodeCacheEntry *entry,
                                         const TensorAttentionDecoderLayer *layer,
                                         size_t hidden_size, size_t kv_width,
                                         size_t intermediate_size) {
    if (entry.q_weight_pointer == layer->q_weight &&
        entry.k_weight_pointer == layer->k_weight &&
        entry.v_weight_pointer == layer->v_weight &&
        entry.gate_weight_pointer == layer->gate_weight &&
        entry.up_weight_pointer == layer->up_weight &&
        entry.qkv_weight_buffer != nil && entry.gate_up_weight_buffer != nil) return 1;
    const size_t qkv_width = hidden_size + 2 * kv_width;
    if (qkv_width > SIZE_MAX / hidden_size / sizeof(float) ||
        intermediate_size > SIZE_MAX / hidden_size / (2 * sizeof(float))) return 0;
    entry.qkv_weight_buffer = decode_scratch_buffer(qkv_width * hidden_size * sizeof(float));
    entry.qkv_buffer = decode_scratch_buffer(qkv_width * sizeof(float));
    entry.gate_up_weight_buffer = decode_scratch_buffer(
        2 * intermediate_size * hidden_size * sizeof(float));
    if (entry.qkv_weight_buffer == nil || entry.qkv_buffer == nil ||
        entry.gate_up_weight_buffer == nil) return 0;
    float *qkv = entry.qkv_weight_buffer.contents;
    memcpy(qkv, layer->q_weight, hidden_size * hidden_size * sizeof(float));
    memcpy(qkv + hidden_size * hidden_size, layer->k_weight,
           kv_width * hidden_size * sizeof(float));
    memcpy(qkv + (hidden_size + kv_width) * hidden_size, layer->v_weight,
           kv_width * hidden_size * sizeof(float));
    float *gate_up = entry.gate_up_weight_buffer.contents;
    memcpy(gate_up, layer->gate_weight,
           intermediate_size * hidden_size * sizeof(float));
    memcpy(gate_up + intermediate_size * hidden_size, layer->up_weight,
           intermediate_size * hidden_size * sizeof(float));
    entry.q_weight_pointer = layer->q_weight;
    entry.k_weight_pointer = layer->k_weight;
    entry.v_weight_pointer = layer->v_weight;
    entry.gate_weight_pointer = layer->gate_weight;
    entry.up_weight_pointer = layer->up_weight;
    return 1;
}

static int encode_decode_gemv(id<MTLCommandBuffer> command, id<MTLBuffer> input,
                              id<MTLBuffer> weight, id<MTLBuffer> output,
                              size_t input_width, size_t output_width) {
    const int use_4simd = decode_gemv_uses_4simd();
    id<MTLComputePipelineState> pipeline = use_4simd ?
        cached_gemv_4simd_pipeline : cached_gemv_pipeline;
    if (pipeline == nil || input_width > UINT32_MAX || output_width > UINT32_MAX) return 0;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    const uint32_t width = (uint32_t)input_width;
    const uint32_t output_size = (uint32_t)output_width;
    const uint64_t strides[2] = {1, input_width};
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:weight offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&width length:sizeof(width) atIndex:3];
    [encoder setBytes:strides length:sizeof(strides) atIndex:4];
    if (use_4simd) [encoder setBytes:&output_size length:sizeof(output_size) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(use_4simd ? (output_width + 3) / 4 : output_width, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    return 1;
}

static int encode_decode_gemv_residual(id<MTLCommandBuffer> command,
                                       id<MTLBuffer> input, id<MTLBuffer> weight,
                                       id<MTLBuffer> residual, id<MTLBuffer> output,
                                       size_t input_width, size_t output_width) {
    const int use_4simd = decode_gemv_uses_4simd();
    id<MTLComputePipelineState> pipeline = use_4simd ?
        cached_gemv_residual_4simd_pipeline : cached_gemv_residual_pipeline;
    if (pipeline == nil || input_width > UINT32_MAX || output_width > UINT32_MAX) return 0;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    const uint32_t width = (uint32_t)input_width;
    const uint32_t output_size = (uint32_t)output_width;
    const uint64_t strides[2] = {1, input_width};
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:weight offset:0 atIndex:1];
    [encoder setBuffer:residual offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&width length:sizeof(width) atIndex:4];
    [encoder setBytes:strides length:sizeof(strides) atIndex:5];
    if (use_4simd) [encoder setBytes:&output_size length:sizeof(output_size) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(use_4simd ? (output_width + 3) / 4 : output_width, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    return 1;
}

static int encode_decode_gemv_silu(id<MTLCommandBuffer> command,
                                   id<MTLBuffer> input, id<MTLBuffer> weight,
                                   id<MTLBuffer> output, size_t input_width,
                                   size_t output_width) {
    const int use_4simd = decode_gemv_uses_4simd();
    id<MTLComputePipelineState> pipeline = use_4simd ?
        cached_gemv_silu_4simd_pipeline : cached_gemv_silu_pipeline;
    if (pipeline == nil || input_width > UINT32_MAX ||
        output_width > UINT32_MAX) return 0;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    const uint32_t input_size = (uint32_t)input_width;
    const uint32_t output_size = (uint32_t)output_width;
    const uint64_t strides[2] = {1, input_width};
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:weight offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&input_size length:sizeof(input_size) atIndex:3];
    [encoder setBytes:strides length:sizeof(strides) atIndex:4];
    [encoder setBytes:&output_size length:sizeof(output_size) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(use_4simd ? (output_width + 3) / 4 : output_width, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    return 1;
}

static int prepare_decode_entry_f16(DecodeCacheEntry *entry, size_t hidden,
                                size_t intermediate, size_t capacity,
                                size_t kv_width, size_t vocabulary) {
    if (entry.hidden_size == hidden && entry.intermediate_size == intermediate &&
        entry.capacity == capacity && entry.norm_buffer != nil) return 1;
    entry.hidden_size = hidden;
    entry.intermediate_size = intermediate;
    entry.capacity = capacity;
    entry.valid_length = 0;
    size_t hidden_bytes = hidden * sizeof(uint16_t);
    size_t cache_bytes = capacity * kv_width * sizeof(uint16_t);
    entry.key_buffer = decode_scratch_buffer(cache_bytes);
    entry.value_buffer = decode_scratch_buffer(cache_bytes);
    entry.norm_buffer = decode_scratch_buffer(hidden_bytes);
    entry.rotated_query_buffer = decode_scratch_buffer(hidden_bytes);
    entry.attention_buffer = decode_scratch_buffer(hidden_bytes);
    entry.attention_partials_buffer = decode_scratch_buffer(32 * hidden * sizeof(float));
    entry.attention_sums_buffer = decode_scratch_buffer(32 * hidden * sizeof(float));
    entry.attention_maxs_buffer = decode_scratch_buffer(32 * hidden * sizeof(float));
    entry.residual_buffer = decode_scratch_buffer(hidden_bytes);
    entry.post_norm_buffer = decode_scratch_buffer(hidden_bytes);
    entry.activated_buffer = decode_scratch_buffer(intermediate * sizeof(uint16_t));
    entry.final_norm_buffer = decode_scratch_buffer(hidden_bytes);
    entry.logits_buffer = decode_scratch_buffer(vocabulary * sizeof(uint16_t));
    entry.token_buffer = decode_scratch_buffer(sizeof(uint32_t));
    return entry.key_buffer != nil && entry.value_buffer != nil &&
        entry.norm_buffer != nil && entry.rotated_query_buffer != nil &&
        entry.attention_buffer != nil && entry.residual_buffer != nil &&
        entry.attention_partials_buffer != nil && entry.attention_sums_buffer != nil &&
        entry.attention_maxs_buffer != nil &&
        entry.post_norm_buffer != nil && entry.activated_buffer != nil &&
        entry.final_norm_buffer != nil &&
        entry.logits_buffer != nil && entry.token_buffer != nil;
}

static void encode_decode_rms_f16(id<MTLCommandBuffer> command,
                              id<MTLBuffer> input, const uint16_t *weight,
                              id<MTLBuffer> output, size_t width, float epsilon) {
    id<MTLBuffer> weight_buffer = decode_weight_buffer_f16(weight, width * sizeof(uint16_t));
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    struct { uint32_t rows; uint32_t width; float epsilon; } params =
        {1, (uint32_t)width, epsilon};
    [encoder setComputePipelineState:cached_rms_f16_pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:weight_buffer offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
}

static int prepare_decode_packed_weights_f16(DecodeCacheEntry *entry,
                                         const TensorAttentionDecoderLayerF16 *layer,
                                         size_t hidden_size, size_t kv_width,
                                         size_t intermediate_size) {
    if (entry.q_weight_pointer == layer->q_weight &&
        entry.k_weight_pointer == layer->k_weight &&
        entry.v_weight_pointer == layer->v_weight &&
        entry.gate_weight_pointer == layer->gate_weight &&
        entry.up_weight_pointer == layer->up_weight &&
        entry.qkv_weight_buffer != nil && entry.gate_up_weight_buffer != nil) return 1;
    const size_t qkv_width = hidden_size + 2 * kv_width;
    if (qkv_width > SIZE_MAX / hidden_size / sizeof(uint16_t) ||
        intermediate_size > SIZE_MAX / hidden_size / (2 * sizeof(uint16_t))) return 0;
    entry.qkv_weight_buffer = decode_scratch_buffer(qkv_width * hidden_size * sizeof(uint16_t));
    entry.qkv_buffer = decode_scratch_buffer(qkv_width * sizeof(uint16_t));
    entry.gate_up_weight_buffer = decode_scratch_buffer(
        2 * intermediate_size * hidden_size * sizeof(uint16_t));
    if (entry.qkv_weight_buffer == nil || entry.qkv_buffer == nil ||
        entry.gate_up_weight_buffer == nil) return 0;
    uint16_t *qkv = entry.qkv_weight_buffer.contents;
    memcpy(qkv, layer->q_weight, hidden_size * hidden_size * sizeof(uint16_t));
    memcpy(qkv + hidden_size * hidden_size, layer->k_weight,
           kv_width * hidden_size * sizeof(uint16_t));
    memcpy(qkv + (hidden_size + kv_width) * hidden_size, layer->v_weight,
           kv_width * hidden_size * sizeof(uint16_t));
    uint16_t *gate_up = entry.gate_up_weight_buffer.contents;
    memcpy(gate_up, layer->gate_weight,
           intermediate_size * hidden_size * sizeof(uint16_t));
    memcpy(gate_up + intermediate_size * hidden_size, layer->up_weight,
           intermediate_size * hidden_size * sizeof(uint16_t));
    entry.q_weight_pointer = layer->q_weight;
    entry.k_weight_pointer = layer->k_weight;
    entry.v_weight_pointer = layer->v_weight;
    entry.gate_weight_pointer = layer->gate_weight;
    entry.up_weight_pointer = layer->up_weight;
    return 1;
}

static int encode_decode_gemv_f16(id<MTLCommandBuffer> command, id<MTLBuffer> input,
                              id<MTLBuffer> weight, id<MTLBuffer> output,
                              size_t input_width, size_t output_width) {
    const int use_4simd = 0;
    id<MTLComputePipelineState> pipeline = cached_gemv_f16_pipeline;
    if (pipeline == nil || input_width > UINT32_MAX || output_width > UINT32_MAX) return 0;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    const uint32_t width = (uint32_t)input_width;
    const uint32_t output_size = (uint32_t)output_width;
    const uint64_t strides[2] = {1, input_width};
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:weight offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&width length:sizeof(width) atIndex:3];
    [encoder setBytes:strides length:sizeof(strides) atIndex:4];
    if (use_4simd) [encoder setBytes:&output_size length:sizeof(output_size) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(use_4simd ? (output_width + 3) / 4 : output_width, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    return 1;
}

static int encode_decode_gemv_residual_f16(id<MTLCommandBuffer> command,
                                       id<MTLBuffer> input, id<MTLBuffer> weight,
                                       id<MTLBuffer> residual, id<MTLBuffer> output,
                                       size_t input_width, size_t output_width) {
    const int use_4simd = 0;
    id<MTLComputePipelineState> pipeline = cached_gemv_residual_f16_pipeline;
    if (pipeline == nil || input_width > UINT32_MAX || output_width > UINT32_MAX) return 0;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    const uint32_t width = (uint32_t)input_width;
    const uint32_t output_size = (uint32_t)output_width;
    const uint64_t strides[2] = {1, input_width};
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:weight offset:0 atIndex:1];
    [encoder setBuffer:residual offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&width length:sizeof(width) atIndex:4];
    [encoder setBytes:strides length:sizeof(strides) atIndex:5];
    if (use_4simd) [encoder setBytes:&output_size length:sizeof(output_size) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(use_4simd ? (output_width + 3) / 4 : output_width, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    return 1;
}

static int encode_decode_gemv_silu_f16(id<MTLCommandBuffer> command,
                                   id<MTLBuffer> input, id<MTLBuffer> weight,
                                   id<MTLBuffer> output, size_t input_width,
                                   size_t output_width) {
    const int use_4simd = 0;
    id<MTLComputePipelineState> pipeline = cached_gemv_silu_f16_pipeline;
    if (pipeline == nil || input_width > UINT32_MAX ||
        output_width > UINT32_MAX) return 0;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    const uint32_t input_size = (uint32_t)input_width;
    const uint32_t output_size = (uint32_t)output_width;
    const uint64_t strides[2] = {1, input_width};
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:weight offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&input_size length:sizeof(input_size) atIndex:3];
    [encoder setBytes:strides length:sizeof(strides) atIndex:4];
    [encoder setBytes:&output_size length:sizeof(output_size) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(use_4simd ? (output_width + 3) / 4 : output_width, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder endEncoding];
    return 1;
}

int tensor_attention_metal_decode_sequence_f16_available(void) { return initialize_metal() && cached_decode_f16_pipeline != nil && cached_decode_partial_f16_pipeline != nil && cached_decode_reduce_f16_pipeline != nil && cached_rope_cache_f16_pipeline != nil && cached_rms_f16_pipeline != nil && cached_argmax_f16_pipeline != nil && cached_embedding_f16_pipeline != nil && cached_gemv_f16_pipeline != nil && cached_gemv_residual_f16_pipeline != nil && cached_gemv_silu_f16_pipeline != nil; }

int tensor_attention_metal_decode_sequence_f32(
    const TensorAttentionDecoderLayer *layers, size_t layer_count,
    const float *embedding, uint32_t initial_token, size_t token_count,
    uint32_t *generated_tokens,
    const float *const *key_cache, const float *const *value_cache,
    size_t past_length, size_t cache_capacity, size_t hidden_size,
    size_t intermediate_size, size_t query_heads, size_t kv_heads,
    size_t head_dim, float rms_norm_epsilon, double rope_theta,
    const float *final_norm, const float *lm_head, size_t vocabulary_size) {
    @autoreleasepool {
        if (timing_enabled())
            fprintf(stderr, "timing_decode_gemv_variant=%s\n",
                    decode_gemv_uses_4simd() ? "4simd" : "baseline");
        double setup_started = timing_start();
        if (!tensor_attention_metal_decode_sequence_available() || layers == NULL ||
            layer_count == 0 || embedding == NULL || token_count == 0 ||
            generated_tokens == NULL || key_cache == NULL ||
            value_cache == NULL || final_norm == NULL || lm_head == NULL ||
            query_heads * head_dim != hidden_size ||
            query_heads % kv_heads != 0 || head_dim > 256 ||
            past_length >= cache_capacity || token_count > cache_capacity - past_length ||
            cache_capacity > UINT32_MAX || token_count > SIZE_MAX / sizeof(uint32_t) ||
            vocabulary_size > UINT32_MAX || intermediate_size > UINT32_MAX ||
            hidden_size > UINT32_MAX || head_dim > UINT32_MAX) return 0;
        const size_t kv_width = kv_heads * head_dim;
        const size_t hidden_bytes = hidden_size * sizeof(float);
        if (vocabulary_size > SIZE_MAX / hidden_size / sizeof(float)) return 0;
        id<MTLBuffer> embedding_buffer = decode_weight_buffer(embedding,
            vocabulary_size * hidden_size * sizeof(float));
        id<MTLBuffer> initial_token_buffer = [cached_device
            newBufferWithBytes:&initial_token length:sizeof(initial_token)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> token_history_buffer = decode_scratch_buffer(
            token_count * sizeof(uint32_t));
        if (embedding_buffer == nil || initial_token_buffer == nil ||
            token_history_buffer == nil) return 0;
        @synchronized(decode_cache) {
            if (decode_hidden_a == nil || decode_hidden_a.length != hidden_bytes) {
                decode_hidden_a = decode_scratch_buffer(hidden_bytes);
                decode_hidden_b = decode_scratch_buffer(hidden_bytes);
            }
            if (decode_hidden_a == nil || decode_hidden_b == nil) return 0;
            NSMutableArray<DecodeCacheEntry *> *entries =
                [NSMutableArray arrayWithCapacity:layer_count];
            for (size_t index = 0; index < layer_count; index++) {
                NSValue *cache_key = [NSValue valueWithPointer:key_cache[index]];
                DecodeCacheEntry *entry = decode_cache[cache_key];
                if (entry == nil || entry.value_pointer != value_cache[index] ||
                    entry.capacity != cache_capacity) {
                    entry = [DecodeCacheEntry new];
                    entry.value_pointer = value_cache[index];
                    entry.valid_length = 0;
                    if (!prepare_decode_entry(entry, hidden_size, intermediate_size,
                                              cache_capacity, kv_width, vocabulary_size)) return 0;
                    if (past_length > 0) {
                        const size_t prefill_bytes = past_length * kv_width * sizeof(float);
                        memcpy(entry.key_buffer.contents, key_cache[index], prefill_bytes);
                        memcpy(entry.value_buffer.contents, value_cache[index], prefill_bytes);
                        entry.valid_length = past_length;
                    }
                    decode_cache[cache_key] = entry;
                }
                if (entry.valid_length != past_length ||
                    !prepare_decode_entry(entry, hidden_size, intermediate_size,
                                          cache_capacity, kv_width, vocabulary_size)) return 0;
                entry.input_norm_encode_ms = 0.0;
                entry.qkv_gemv_encode_ms = 0.0;
                entry.rope_kvcache_encode_ms = 0.0;
                entry.attention_encode_ms = 0.0;
                entry.output_gemv_encode_ms = 0.0;
                entry.post_norm_encode_ms = 0.0;
                entry.mlp_gate_up_encode_ms = 0.0;
                entry.mlp_down_encode_ms = 0.0;
                if (!prepare_decode_packed_weights(entry, &layers[index], hidden_size,
                                                   kv_width, intermediate_size)) return 0;
                [entries addObject:entry];
            }
            timing_report("decode_sequence_setup", setup_started);
            const int collect_gpu_timing = timing_enabled();
            NSMutableArray<id<MTLCommandBuffer>> *timed_commands = collect_gpu_timing ?
                [NSMutableArray arrayWithCapacity:token_count] : nil;
            double submit_started = timing_start();
            id<MTLCommandBuffer> command = nil;
            for (size_t token_index = 0; token_index < token_count; token_index++) {
                command = [cached_queue commandBuffer];
                if (command == nil) return 0;
                id<MTLBuffer> current_hidden = decode_hidden_a;
                id<MTLBuffer> next_hidden = decode_hidden_b;
                id<MTLComputeCommandEncoder> embedding_encoder = [command computeCommandEncoder];
                if (embedding_encoder == nil) return 0;
                const uint32_t hidden_width = (uint32_t)hidden_size;
                id<MTLBuffer> input_token_buffer = token_index == 0 ?
                    initial_token_buffer : token_history_buffer;
                const NSUInteger input_token_offset = token_index == 0 ? 0 :
                    (token_index - 1) * sizeof(uint32_t);
                [embedding_encoder setComputePipelineState:cached_embedding_pipeline];
                [embedding_encoder setBuffer:embedding_buffer offset:0 atIndex:0];
                [embedding_encoder setBuffer:input_token_buffer offset:input_token_offset atIndex:1];
                [embedding_encoder setBuffer:current_hidden offset:0 atIndex:2];
                [embedding_encoder setBytes:&hidden_width length:sizeof(hidden_width) atIndex:3];
                [embedding_encoder dispatchThreads:MTLSizeMake(hidden_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                        cached_embedding_pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
                [embedding_encoder endEncoding];
                for (size_t index = 0; index < layer_count; index++) {
                const TensorAttentionDecoderLayer *layer = &layers[index];
                DecodeCacheEntry *entry = entries[index];
                double stage_started = timing_start();
                encode_decode_rms(command, current_hidden, layer->input_norm,
                                  entry.norm_buffer, hidden_size, rms_norm_epsilon);
                if (stage_started != 0.0)
                    entry.input_norm_encode_ms += timing_now_ms() - stage_started;
                stage_started = timing_start();
                if (!encode_decode_gemv(command, entry.norm_buffer,
                        entry.qkv_weight_buffer, entry.qkv_buffer, hidden_size,
                        hidden_size + 2 * kv_width)) return 0;
                if (stage_started != 0.0)
                    entry.qkv_gemv_encode_ms += timing_now_ms() - stage_started;
                stage_started = timing_start();
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (encoder == nil) return 0;
                DecodeRopeParams rope = {(uint32_t)query_heads, (uint32_t)kv_heads,
                    (uint32_t)head_dim, (uint32_t)(past_length + token_index), (float)rope_theta};
                [encoder setComputePipelineState:cached_rope_cache_pipeline];
                [encoder setBuffer:entry.qkv_buffer offset:0 atIndex:0];
                [encoder setBuffer:entry.qkv_buffer offset:hidden_bytes atIndex:1];
                [encoder setBuffer:entry.qkv_buffer offset:hidden_bytes + kv_width * sizeof(float) atIndex:2];
                [encoder setBuffer:entry.rotated_query_buffer offset:0 atIndex:3];
                [encoder setBuffer:entry.key_buffer offset:0 atIndex:4];
                [encoder setBuffer:entry.value_buffer offset:0 atIndex:5];
                [encoder setBytes:&rope length:sizeof(rope) atIndex:6];
                [encoder dispatchThreads:MTLSizeMake(MAX(hidden_size, kv_width), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                        cached_rope_cache_pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
                [encoder endEncoding];
                if (stage_started != 0.0)
                    entry.rope_kvcache_encode_ms += timing_now_ms() - stage_started;
                stage_started = timing_start();
                DecodeSdpaParams attention = {(uint32_t)query_heads, (uint32_t)kv_heads,
                    (uint32_t)(past_length + token_index + 1), (uint32_t)cache_capacity,
                    (uint32_t)head_dim, 1.0f / sqrtf((float)head_dim)};
                if (attention.kv_length >= 1024) {
                    encoder = [command computeCommandEncoder];
                    if (encoder == nil) return 0;
                    [encoder setComputePipelineState:cached_decode_partial_pipeline];
                    [encoder setBuffer:entry.rotated_query_buffer offset:0 atIndex:0];
                    [encoder setBuffer:entry.key_buffer offset:0 atIndex:1];
                    [encoder setBuffer:entry.value_buffer offset:0 atIndex:2];
                    [encoder setBuffer:entry.attention_partials_buffer offset:0 atIndex:3];
                    [encoder setBuffer:entry.attention_sums_buffer offset:0 atIndex:4];
                    [encoder setBuffer:entry.attention_maxs_buffer offset:0 atIndex:5];
                    [encoder setBytes:&attention length:sizeof(attention) atIndex:6];
                    [encoder dispatchThreadgroups:MTLSizeMake(query_heads, 1, 32)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [encoder endEncoding];
                    encoder = [command computeCommandEncoder];
                    if (encoder == nil) return 0;
                    [encoder setComputePipelineState:cached_decode_reduce_pipeline];
                    [encoder setBuffer:entry.attention_partials_buffer offset:0 atIndex:0];
                    [encoder setBuffer:entry.attention_sums_buffer offset:0 atIndex:1];
                    [encoder setBuffer:entry.attention_maxs_buffer offset:0 atIndex:2];
                    [encoder setBuffer:entry.attention_buffer offset:0 atIndex:3];
                    [encoder setBytes:&attention length:sizeof(attention) atIndex:4];
                    [encoder dispatchThreadgroups:MTLSizeMake(query_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [encoder endEncoding];
                } else {
                    encoder = [command computeCommandEncoder];
                    if (encoder == nil) return 0;
                    [encoder setComputePipelineState:cached_decode_pipeline];
                    [encoder setBuffer:entry.rotated_query_buffer offset:0 atIndex:0];
                    [encoder setBuffer:entry.key_buffer offset:0 atIndex:1];
                    [encoder setBuffer:entry.value_buffer offset:0 atIndex:2];
                    [encoder setBuffer:entry.attention_buffer offset:0 atIndex:3];
                    [encoder setBytes:&attention length:sizeof(attention) atIndex:4];
                    [encoder dispatchThreadgroups:MTLSizeMake(query_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [encoder endEncoding];
                }
                if (stage_started != 0.0)
                    entry.attention_encode_ms += timing_now_ms() - stage_started;
                id<MTLBuffer> o_weight = decode_weight_buffer(layer->o_weight,
                    hidden_size * hidden_size * sizeof(float));
                stage_started = timing_start();
                if (!encode_decode_gemv_residual(command, entry.attention_buffer,
                    o_weight, current_hidden, entry.residual_buffer,
                    hidden_size, hidden_size)) return 0;
                if (stage_started != 0.0)
                    entry.output_gemv_encode_ms += timing_now_ms() - stage_started;
                stage_started = timing_start();
                encode_decode_rms(command, entry.residual_buffer, layer->post_norm,
                    entry.post_norm_buffer, hidden_size, rms_norm_epsilon);
                if (stage_started != 0.0)
                    entry.post_norm_encode_ms += timing_now_ms() - stage_started;
                stage_started = timing_start();
                if (!encode_decode_gemv_silu(command, entry.post_norm_buffer,
                    entry.gate_up_weight_buffer, entry.activated_buffer,
                    hidden_size, intermediate_size)) return 0;
                if (stage_started != 0.0)
                    entry.mlp_gate_up_encode_ms += timing_now_ms() - stage_started;
                id<MTLBuffer> down_weight = decode_weight_buffer(layer->down_weight,
                    hidden_size * intermediate_size * sizeof(float));
                stage_started = timing_start();
                if (!encode_decode_gemv_residual(command, entry.activated_buffer,
                    down_weight, entry.residual_buffer, next_hidden,
                    intermediate_size, hidden_size)) return 0;
                if (stage_started != 0.0)
                    entry.mlp_down_encode_ms += timing_now_ms() - stage_started;
                id<MTLBuffer> swap = current_hidden;
                current_hidden = next_hidden;
                next_hidden = swap;
                }
                DecodeCacheEntry *last_entry = entries.lastObject;
                encode_decode_rms(command, current_hidden, final_norm,
                    last_entry.final_norm_buffer, hidden_size, rms_norm_epsilon);
                id<MTLBuffer> lm_head_buffer = decode_weight_buffer(lm_head,
                    vocabulary_size * hidden_size * sizeof(float));
                if (!encode_decode_gemv(command, last_entry.final_norm_buffer,
                    lm_head_buffer, last_entry.logits_buffer, hidden_size,
                    vocabulary_size)) return 0;
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (encoder == nil) return 0;
                [encoder setComputePipelineState:cached_argmax_pipeline];
                [encoder setBuffer:last_entry.logits_buffer offset:0 atIndex:0];
                [encoder setBuffer:token_history_buffer
                    offset:token_index * sizeof(uint32_t) atIndex:1];
                const uint32_t vocab_count = (uint32_t)vocabulary_size;
                [encoder setBytes:&vocab_count length:sizeof(vocab_count) atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];
                [command commit];
                if (timed_commands != nil) [timed_commands addObject:command];
                for (DecodeCacheEntry *entry in entries)
                    entry.valid_length = past_length + token_index + 1;
            }
            timing_report("decode_sequence_submit_loop", submit_started);
            double wait_started = timing_start();
            [command waitUntilCompleted];
            timing_report("decode_sequence_final_wait", wait_started);
            if (command.status != MTLCommandBufferStatusCompleted) return 0;
            if (timed_commands != nil) {
                double gpu_time_ms = 0.0;
                size_t sampled_commands = 0;
                for (id<MTLCommandBuffer> timed_command in timed_commands) {
                    if (timed_command.GPUStartTime > 0.0 &&
                        timed_command.GPUEndTime >= timed_command.GPUStartTime) {
                        gpu_time_ms += (timed_command.GPUEndTime -
                                        timed_command.GPUStartTime) * 1000.0;
                        sampled_commands++;
                    }
                }
                timing_report_elapsed("decode_sequence_gpu_sum", gpu_time_ms);
                if (sampled_commands > 0)
                    fprintf(stderr, "timing_decode_sequence_gpu_avg_per_token_ms=%.3f sampled_tokens=%zu\n",
                            gpu_time_ms / (double)sampled_commands, sampled_commands);
            }
            if (timing_enabled()) {
                for (size_t index = 0; index < layer_count; index++) {
                    DecodeCacheEntry *entry = entries[index];
                    fprintf(stderr, "timing_decode_layer_%02zu_encode_ms=input_norm:%.3f qkv_gemv:%.3f rope_kvcache:%.3f attention:%.3f output_gemv:%.3f post_norm:%.3f mlp_gate_up_fused:%.3f mlp_down_gemv:%.3f\n",
                            index, entry.input_norm_encode_ms,
                            entry.qkv_gemv_encode_ms,
                            entry.rope_kvcache_encode_ms,
                            entry.attention_encode_ms,
                            entry.output_gemv_encode_ms,
                            entry.post_norm_encode_ms,
                            entry.mlp_gate_up_encode_ms,
                            entry.mlp_down_encode_ms);
                }
            }
            memcpy(generated_tokens, token_history_buffer.contents,
                   token_count * sizeof(uint32_t));
            return 1;
        }
    }
}

int tensor_attention_metal_decode_sequence_f16(
    const TensorAttentionDecoderLayerF16 *layers, size_t layer_count,
    const uint16_t *embedding, uint32_t initial_token, size_t token_count,
    uint32_t *generated_tokens,
    const uint16_t *const *key_cache, const uint16_t *const *value_cache,
    size_t past_length, size_t cache_capacity, size_t hidden_size,
    size_t intermediate_size, size_t query_heads, size_t kv_heads,
    size_t head_dim, float rms_norm_epsilon, double rope_theta,
    const uint16_t *final_norm, const uint16_t *lm_head, size_t vocabulary_size) {
    @autoreleasepool {
        if (timing_enabled())
            fprintf(stderr, "timing_decode_gemv_variant=%s\n",
                    decode_gemv_uses_4simd() ? "4simd" : "baseline");
        double setup_started = timing_start();
        if (!tensor_attention_metal_decode_sequence_f16_available() || layers == NULL ||
            layer_count == 0 || embedding == NULL || token_count == 0 ||
            generated_tokens == NULL || key_cache == NULL ||
            value_cache == NULL || final_norm == NULL || lm_head == NULL ||
            query_heads * head_dim != hidden_size ||
            query_heads % kv_heads != 0 || head_dim > 256 ||
            past_length >= cache_capacity || token_count > cache_capacity - past_length ||
            cache_capacity > UINT32_MAX || token_count > SIZE_MAX / sizeof(uint32_t) ||
            vocabulary_size > UINT32_MAX || intermediate_size > UINT32_MAX ||
            hidden_size > UINT32_MAX || head_dim > UINT32_MAX) return 0;
        const size_t kv_width = kv_heads * head_dim;
        const size_t hidden_bytes = hidden_size * sizeof(uint16_t);
        if (vocabulary_size > SIZE_MAX / hidden_size / sizeof(uint16_t)) return 0;
        id<MTLBuffer> embedding_buffer = decode_weight_buffer_f16(embedding,
            vocabulary_size * hidden_size * sizeof(uint16_t));
        id<MTLBuffer> initial_token_buffer = [cached_device
            newBufferWithBytes:&initial_token length:sizeof(initial_token)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> token_history_buffer = decode_scratch_buffer(
            token_count * sizeof(uint32_t));
        if (embedding_buffer == nil || initial_token_buffer == nil ||
            token_history_buffer == nil) return 0;
        @synchronized(decode_cache) {
            if (decode_hidden_a == nil || decode_hidden_a.length != hidden_bytes) {
                decode_hidden_a = decode_scratch_buffer(hidden_bytes);
                decode_hidden_b = decode_scratch_buffer(hidden_bytes);
            }
            if (decode_hidden_a == nil || decode_hidden_b == nil) return 0;
            NSMutableArray<DecodeCacheEntry *> *entries =
                [NSMutableArray arrayWithCapacity:layer_count];
            for (size_t index = 0; index < layer_count; index++) {
                NSValue *cache_key = [NSValue valueWithPointer:key_cache[index]];
                DecodeCacheEntry *entry = decode_cache[cache_key];
                if (entry == nil || entry.value_pointer != value_cache[index] ||
                    entry.capacity != cache_capacity) {
                    entry = [DecodeCacheEntry new];
                    entry.value_pointer = value_cache[index];
                    entry.valid_length = 0;
                    if (!prepare_decode_entry_f16(entry, hidden_size, intermediate_size,
                                              cache_capacity, kv_width, vocabulary_size)) return 0;
                    if (past_length > 0) {
                        const size_t prefill_bytes = past_length * kv_width * sizeof(uint16_t);
                        memcpy(entry.key_buffer.contents, key_cache[index], prefill_bytes);
                        memcpy(entry.value_buffer.contents, value_cache[index], prefill_bytes);
                        entry.valid_length = past_length;
                    }
                    decode_cache[cache_key] = entry;
                }
                if (entry.valid_length != past_length ||
                    !prepare_decode_entry_f16(entry, hidden_size, intermediate_size,
                                          cache_capacity, kv_width, vocabulary_size)) return 0;
                entry.input_norm_encode_ms = 0.0;
                entry.qkv_gemv_encode_ms = 0.0;
                entry.rope_kvcache_encode_ms = 0.0;
                entry.attention_encode_ms = 0.0;
                entry.output_gemv_encode_ms = 0.0;
                entry.post_norm_encode_ms = 0.0;
                entry.mlp_gate_up_encode_ms = 0.0;
                entry.mlp_down_encode_ms = 0.0;
                if (!prepare_decode_packed_weights_f16(entry, &layers[index], hidden_size,
                                                   kv_width, intermediate_size)) return 0;
                [entries addObject:entry];
            }
            timing_report("decode_sequence_setup", setup_started);
            const int collect_gpu_timing = timing_enabled();
            NSMutableArray<id<MTLCommandBuffer>> *timed_commands = collect_gpu_timing ?
                [NSMutableArray arrayWithCapacity:token_count] : nil;
            double submit_started = timing_start();
            id<MTLCommandBuffer> command = nil;
            for (size_t token_index = 0; token_index < token_count; token_index++) {
                command = [cached_queue commandBuffer];
                if (command == nil) return 0;
                id<MTLBuffer> current_hidden = decode_hidden_a;
                id<MTLBuffer> next_hidden = decode_hidden_b;
                id<MTLComputeCommandEncoder> embedding_encoder = [command computeCommandEncoder];
                if (embedding_encoder == nil) return 0;
                const uint32_t hidden_width = (uint32_t)hidden_size;
                id<MTLBuffer> input_token_buffer = token_index == 0 ?
                    initial_token_buffer : token_history_buffer;
                const NSUInteger input_token_offset = token_index == 0 ? 0 :
                    (token_index - 1) * sizeof(uint32_t);
                [embedding_encoder setComputePipelineState:cached_embedding_f16_pipeline];
                [embedding_encoder setBuffer:embedding_buffer offset:0 atIndex:0];
                [embedding_encoder setBuffer:input_token_buffer offset:input_token_offset atIndex:1];
                [embedding_encoder setBuffer:current_hidden offset:0 atIndex:2];
                [embedding_encoder setBytes:&hidden_width length:sizeof(hidden_width) atIndex:3];
                [embedding_encoder dispatchThreads:MTLSizeMake(hidden_size, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                        cached_embedding_f16_pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
                [embedding_encoder endEncoding];
                for (size_t index = 0; index < layer_count; index++) {
                const TensorAttentionDecoderLayerF16 *layer = &layers[index];
                DecodeCacheEntry *entry = entries[index];
                double stage_started = timing_start();
                encode_decode_rms_f16(command, current_hidden, layer->input_norm,
                                  entry.norm_buffer, hidden_size, rms_norm_epsilon);
                if (stage_started != 0.0)
                    entry.input_norm_encode_ms += timing_now_ms() - stage_started;
                stage_started = timing_start();
                if (!encode_decode_gemv_f16(command, entry.norm_buffer,
                        entry.qkv_weight_buffer, entry.qkv_buffer, hidden_size,
                        hidden_size + 2 * kv_width)) return 0;
                if (stage_started != 0.0)
                    entry.qkv_gemv_encode_ms += timing_now_ms() - stage_started;
                stage_started = timing_start();
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (encoder == nil) return 0;
                DecodeRopeParams rope = {(uint32_t)query_heads, (uint32_t)kv_heads,
                    (uint32_t)head_dim, (uint32_t)(past_length + token_index), (float)rope_theta};
                [encoder setComputePipelineState:cached_rope_cache_f16_pipeline];
                [encoder setBuffer:entry.qkv_buffer offset:0 atIndex:0];
                [encoder setBuffer:entry.qkv_buffer offset:hidden_bytes atIndex:1];
                [encoder setBuffer:entry.qkv_buffer offset:hidden_bytes + kv_width * sizeof(uint16_t) atIndex:2];
                [encoder setBuffer:entry.rotated_query_buffer offset:0 atIndex:3];
                [encoder setBuffer:entry.key_buffer offset:0 atIndex:4];
                [encoder setBuffer:entry.value_buffer offset:0 atIndex:5];
                [encoder setBytes:&rope length:sizeof(rope) atIndex:6];
                [encoder dispatchThreads:MTLSizeMake(MAX(hidden_size, kv_width), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                        cached_rope_cache_f16_pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
                [encoder endEncoding];
                if (stage_started != 0.0)
                    entry.rope_kvcache_encode_ms += timing_now_ms() - stage_started;
                stage_started = timing_start();
                DecodeSdpaParams attention = {(uint32_t)query_heads, (uint32_t)kv_heads,
                    (uint32_t)(past_length + token_index + 1), (uint32_t)cache_capacity,
                    (uint32_t)head_dim, 1.0f / sqrtf((float)head_dim)};
                if (attention.kv_length >= 1024) {
                    encoder = [command computeCommandEncoder];
                    if (encoder == nil) return 0;
                    [encoder setComputePipelineState:cached_decode_partial_f16_pipeline];
                    [encoder setBuffer:entry.rotated_query_buffer offset:0 atIndex:0];
                    [encoder setBuffer:entry.key_buffer offset:0 atIndex:1];
                    [encoder setBuffer:entry.value_buffer offset:0 atIndex:2];
                    [encoder setBuffer:entry.attention_partials_buffer offset:0 atIndex:3];
                    [encoder setBuffer:entry.attention_sums_buffer offset:0 atIndex:4];
                    [encoder setBuffer:entry.attention_maxs_buffer offset:0 atIndex:5];
                    [encoder setBytes:&attention length:sizeof(attention) atIndex:6];
                    [encoder dispatchThreadgroups:MTLSizeMake(query_heads, 1, 32)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [encoder endEncoding];
                    encoder = [command computeCommandEncoder];
                    if (encoder == nil) return 0;
                    [encoder setComputePipelineState:cached_decode_reduce_f16_pipeline];
                    [encoder setBuffer:entry.attention_partials_buffer offset:0 atIndex:0];
                    [encoder setBuffer:entry.attention_sums_buffer offset:0 atIndex:1];
                    [encoder setBuffer:entry.attention_maxs_buffer offset:0 atIndex:2];
                    [encoder setBuffer:entry.attention_buffer offset:0 atIndex:3];
                    [encoder setBytes:&attention length:sizeof(attention) atIndex:4];
                    [encoder dispatchThreadgroups:MTLSizeMake(query_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [encoder endEncoding];
                } else {
                    encoder = [command computeCommandEncoder];
                    if (encoder == nil) return 0;
                    [encoder setComputePipelineState:cached_decode_f16_pipeline];
                    [encoder setBuffer:entry.rotated_query_buffer offset:0 atIndex:0];
                    [encoder setBuffer:entry.key_buffer offset:0 atIndex:1];
                    [encoder setBuffer:entry.value_buffer offset:0 atIndex:2];
                    [encoder setBuffer:entry.attention_buffer offset:0 atIndex:3];
                    [encoder setBytes:&attention length:sizeof(attention) atIndex:4];
                    [encoder dispatchThreadgroups:MTLSizeMake(query_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [encoder endEncoding];
                }
                if (stage_started != 0.0)
                    entry.attention_encode_ms += timing_now_ms() - stage_started;
                id<MTLBuffer> o_weight = decode_weight_buffer_f16(layer->o_weight,
                    hidden_size * hidden_size * sizeof(uint16_t));
                stage_started = timing_start();
                if (!encode_decode_gemv_residual_f16(command, entry.attention_buffer,
                    o_weight, current_hidden, entry.residual_buffer,
                    hidden_size, hidden_size)) return 0;
                if (stage_started != 0.0)
                    entry.output_gemv_encode_ms += timing_now_ms() - stage_started;
                stage_started = timing_start();
                encode_decode_rms_f16(command, entry.residual_buffer, layer->post_norm,
                    entry.post_norm_buffer, hidden_size, rms_norm_epsilon);
                if (stage_started != 0.0)
                    entry.post_norm_encode_ms += timing_now_ms() - stage_started;
                stage_started = timing_start();
                if (!encode_decode_gemv_silu_f16(command, entry.post_norm_buffer,
                    entry.gate_up_weight_buffer, entry.activated_buffer,
                    hidden_size, intermediate_size)) return 0;
                if (stage_started != 0.0)
                    entry.mlp_gate_up_encode_ms += timing_now_ms() - stage_started;
                id<MTLBuffer> down_weight = decode_weight_buffer_f16(layer->down_weight,
                    hidden_size * intermediate_size * sizeof(uint16_t));
                stage_started = timing_start();
                if (!encode_decode_gemv_residual_f16(command, entry.activated_buffer,
                    down_weight, entry.residual_buffer, next_hidden,
                    intermediate_size, hidden_size)) return 0;
                if (stage_started != 0.0)
                    entry.mlp_down_encode_ms += timing_now_ms() - stage_started;
                id<MTLBuffer> swap = current_hidden;
                current_hidden = next_hidden;
                next_hidden = swap;
                }
                DecodeCacheEntry *last_entry = entries.lastObject;
                encode_decode_rms_f16(command, current_hidden, final_norm,
                    last_entry.final_norm_buffer, hidden_size, rms_norm_epsilon);
                id<MTLBuffer> lm_head_buffer = decode_weight_buffer_f16(lm_head,
                    vocabulary_size * hidden_size * sizeof(uint16_t));
                if (!encode_decode_gemv_f16(command, last_entry.final_norm_buffer,
                    lm_head_buffer, last_entry.logits_buffer, hidden_size,
                    vocabulary_size)) return 0;
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (encoder == nil) return 0;
                [encoder setComputePipelineState:cached_argmax_f16_pipeline];
                [encoder setBuffer:last_entry.logits_buffer offset:0 atIndex:0];
                [encoder setBuffer:token_history_buffer
                    offset:token_index * sizeof(uint32_t) atIndex:1];
                const uint32_t vocab_count = (uint32_t)vocabulary_size;
                [encoder setBytes:&vocab_count length:sizeof(vocab_count) atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [encoder endEncoding];
                [command commit];
                if (timed_commands != nil) [timed_commands addObject:command];
                for (DecodeCacheEntry *entry in entries)
                    entry.valid_length = past_length + token_index + 1;
            }
            timing_report("decode_sequence_submit_loop", submit_started);
            double wait_started = timing_start();
            [command waitUntilCompleted];
            timing_report("decode_sequence_final_wait", wait_started);
            if (command.status != MTLCommandBufferStatusCompleted) return 0;
            if (timed_commands != nil) {
                double gpu_time_ms = 0.0;
                size_t sampled_commands = 0;
                for (id<MTLCommandBuffer> timed_command in timed_commands) {
                    if (timed_command.GPUStartTime > 0.0 &&
                        timed_command.GPUEndTime >= timed_command.GPUStartTime) {
                        gpu_time_ms += (timed_command.GPUEndTime -
                                        timed_command.GPUStartTime) * 1000.0;
                        sampled_commands++;
                    }
                }
                timing_report_elapsed("decode_sequence_gpu_sum", gpu_time_ms);
                if (sampled_commands > 0)
                    fprintf(stderr, "timing_decode_sequence_gpu_avg_per_token_ms=%.3f sampled_tokens=%zu\n",
                            gpu_time_ms / (double)sampled_commands, sampled_commands);
            }
            if (timing_enabled()) {
                for (size_t index = 0; index < layer_count; index++) {
                    DecodeCacheEntry *entry = entries[index];
                    fprintf(stderr, "timing_decode_layer_%02zu_encode_ms=input_norm:%.3f qkv_gemv:%.3f rope_kvcache:%.3f attention:%.3f output_gemv:%.3f post_norm:%.3f mlp_gate_up_fused:%.3f mlp_down_gemv:%.3f\n",
                            index, entry.input_norm_encode_ms,
                            entry.qkv_gemv_encode_ms,
                            entry.rope_kvcache_encode_ms,
                            entry.attention_encode_ms,
                            entry.output_gemv_encode_ms,
                            entry.post_norm_encode_ms,
                            entry.mlp_gate_up_encode_ms,
                            entry.mlp_down_encode_ms);
                }
            }
            memcpy(generated_tokens, token_history_buffer.contents,
                   token_count * sizeof(uint32_t));
            return 1;
        }
    }
}

static id<MTLBuffer> decode_weight_buffer(const void *weight, size_t byte_count) {
    NSValue *key = [NSValue valueWithPointer:weight];
    id<MTLBuffer> buffer = decode_weight_buffers[key];
    if (buffer != nil && buffer.length >= byte_count) return buffer;
    const long page_size_value = sysconf(_SC_PAGESIZE);
    if (page_size_value > 0 && (uintptr_t)weight % (size_t)page_size_value == 0) {
        const size_t page_size = (size_t)page_size_value;
        if (byte_count <= SIZE_MAX - page_size + 1) {
            const size_t aligned_length = (byte_count + page_size - 1) / page_size * page_size;
            buffer = [cached_device newBufferWithBytesNoCopy:(void *)weight
                length:aligned_length options:MTLResourceStorageModeShared
                deallocator:nil];
        }
    }
    if (buffer == nil)
        buffer = [cached_device newBufferWithBytes:weight length:byte_count
            options:MTLResourceStorageModeShared];
    if (buffer != nil) decode_weight_buffers[key] = buffer;
    return buffer;
}

static id<MTLBuffer> decode_weight_buffer_f16(const uint16_t *weight, size_t byte_count) {
    return decode_weight_buffer(weight, byte_count);
}

int tensor_attention_metal_decode_f32(const float *query, const float *key,
                                      const float *value, const float *output_weight,
                                      const float *residual, float *output,
                                      size_t query_heads, size_t kv_heads,
                                      size_t kv_length, size_t kv_capacity,
                                      size_t head_dim) {
    @autoreleasepool {
        if (query == NULL || key == NULL || value == NULL || output == NULL ||
            query_heads == 0 || kv_heads == 0 || query_heads % kv_heads != 0 ||
            kv_length == 0 || kv_length > kv_capacity || head_dim == 0 ||
            head_dim > 256 || kv_capacity > UINT32_MAX || kv_length > UINT32_MAX ||
            query_heads > UINT32_MAX || kv_heads > UINT32_MAX ||
            output_weight == NULL || residual == NULL || !initialize_metal() ||
            cached_decode_pipeline == nil || cached_residual_pipeline == nil) return 0;
        size_t kv_width, query_count, cache_bytes;
        if (!checked_product(kv_heads, head_dim, &kv_width) ||
            !checked_product(kv_capacity, kv_width, &cache_bytes) ||
            !checked_product(cache_bytes, sizeof(float), &cache_bytes) ||
            !checked_product(query_heads, head_dim, &query_count) ||
            query_count > SIZE_MAX / sizeof(float)) return 0;
        const size_t query_bytes = query_count * sizeof(float);
        @synchronized(decode_cache) {
            NSValue *cache_key = [NSValue valueWithPointer:key];
            DecodeCacheEntry *entry = decode_cache[cache_key];
            if (entry == nil || entry.capacity != kv_capacity ||
                entry.value_pointer != value) {
                entry = [DecodeCacheEntry new];
                entry.key_buffer = [cached_device newBufferWithLength:cache_bytes
                    options:MTLResourceStorageModeShared];
                entry.value_buffer = [cached_device newBufferWithLength:cache_bytes
                    options:MTLResourceStorageModeShared];
                if (entry.key_buffer == nil || entry.value_buffer == nil) return 0;
                entry.value_pointer = value;
                entry.capacity = kv_capacity;
                entry.valid_length = 0;
                decode_cache[cache_key] = entry;
            }
            const size_t first_position = entry.valid_length != 0 &&
                kv_length == entry.valid_length + 1 ? entry.valid_length : 0;
            const size_t copy_positions = first_position == 0 ? kv_length : 1;
            const size_t copy_offset = first_position * kv_width;
            const size_t copy_bytes = copy_positions * kv_width * sizeof(float);
            memcpy((float *)entry.key_buffer.contents + copy_offset,
                   key + copy_offset, copy_bytes);
            memcpy((float *)entry.value_buffer.contents + copy_offset,
                   value + copy_offset, copy_bytes);

            id<MTLBuffer> query_buffer = [cached_device newBufferWithBytes:query
                length:query_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> residual_buffer = [cached_device newBufferWithBytes:residual
                length:query_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> attention_buffer = [cached_device newBufferWithLength:query_bytes
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> projected_buffer = [cached_device newBufferWithLength:query_bytes
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buffer = [cached_device newBufferWithLength:query_bytes
                options:MTLResourceStorageModeShared];
            if (query_count != 0 && query_bytes > SIZE_MAX / query_count) return 0;
            id<MTLBuffer> weight_buffer = decode_weight_buffer(output_weight,
                query_bytes * query_count);
            id<MTLCommandBuffer> command = [cached_queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = command == nil ? nil :
                [command computeCommandEncoder];
            if (query_buffer == nil || residual_buffer == nil || attention_buffer == nil ||
                projected_buffer == nil || output_buffer == nil || weight_buffer == nil ||
                encoder == nil) return 0;
            DecodeSdpaParams params = {(uint32_t)query_heads, (uint32_t)kv_heads,
                (uint32_t)kv_length, (uint32_t)kv_capacity, (uint32_t)head_dim,
                1.0f / sqrtf((float)head_dim)};
                    [encoder setComputePipelineState:cached_decode_pipeline];
            [encoder setBuffer:query_buffer offset:0 atIndex:0];
            [encoder setBuffer:entry.key_buffer offset:0 atIndex:1];
            [encoder setBuffer:entry.value_buffer offset:0 atIndex:2];
            [encoder setBuffer:attention_buffer offset:0 atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(query_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encoder endEncoding];
            MPSMatrixDescriptor *input_descriptor = [MPSMatrixDescriptor
                matrixDescriptorWithRows:1 columns:query_heads * head_dim
                rowBytes:query_heads * head_dim * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *weight_descriptor = [MPSMatrixDescriptor
                matrixDescriptorWithRows:query_heads * head_dim columns:query_heads * head_dim
                rowBytes:query_heads * head_dim * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *output_descriptor = [MPSMatrixDescriptor
                matrixDescriptorWithRows:1 columns:query_heads * head_dim
                rowBytes:query_heads * head_dim * sizeof(float) dataType:MPSDataTypeFloat32];
            MPSMatrix *attention_matrix = [[MPSMatrix alloc]
                initWithBuffer:attention_buffer descriptor:input_descriptor];
            MPSMatrix *weight_matrix = [[MPSMatrix alloc]
                initWithBuffer:weight_buffer descriptor:weight_descriptor];
            MPSMatrix *projected_matrix = [[MPSMatrix alloc]
                initWithBuffer:projected_buffer descriptor:output_descriptor];
            MPSMatrixMultiplication *projection = [[MPSMatrixMultiplication alloc]
                initWithDevice:cached_device transposeLeft:NO transposeRight:YES
                resultRows:1 resultColumns:query_heads * head_dim
                interiorColumns:query_heads * head_dim alpha:1.0 beta:0.0];
            if (attention_matrix == nil || weight_matrix == nil ||
                projected_matrix == nil || projection == nil) return 0;
            [projection encodeToCommandBuffer:command leftMatrix:attention_matrix
                rightMatrix:weight_matrix resultMatrix:projected_matrix];
            encoder = [command computeCommandEncoder];
            if (encoder == nil) return 0;
            [encoder setComputePipelineState:cached_residual_pipeline];
            [encoder setBuffer:projected_buffer offset:0 atIndex:0];
            [encoder setBuffer:residual_buffer offset:0 atIndex:1];
            [encoder setBuffer:output_buffer offset:0 atIndex:2];
            const uint32_t count = (uint32_t)query_count;
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(query_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                    cached_residual_pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) return 0;
            memcpy(output, output_buffer.contents, query_bytes);
            entry.valid_length = kv_length;
            return 1;
        }
    }
}

void tensor_attention_metal_decode_cache_clear(void) {
    @autoreleasepool {
        if (!initialize_metal() || decode_cache == nil) return;
        @synchronized(decode_cache) {
            [decode_cache removeAllObjects];
            [decode_weight_buffers removeAllObjects];
        }
    }
}

static int metal_backend_available(void) {
    @autoreleasepool {
        return initialize_metal();
    }
}

typedef struct {
    int32_t batch;
    int32_t heads;
    int32_t head_dim;
    int32_t query_length;
    int32_t key_length;
    int32_t gqa_factor;
    float scale;
    float softcapping;
    int32_t query_blocks;
    int32_t key_blocks;
    int32_t query_blocks_aligned;
    int32_t key_blocks_aligned;
    int32_t query_remainder;
    int32_t key_remainder;
    int32_t query_offset;
    int32_t stride_padding;
    int64_t query_strides[3];
    int64_t key_strides[3];
    int64_t value_strides[3];
    int64_t output_strides[3];
} TiledVisionSdpaParams;

static int metal_backend_run_tiled_vision(const float *query, const float *key,
                                          const float *value, float *output,
                                          size_t batch, size_t query_heads,
                                          size_t kv_heads, size_t sequence,
                                          float scale) {
    if (cached_tiled_vision_pipeline == nil || batch != 1 || sequence == 0 ||
        sequence % 32 != 0 || sequence > INT32_MAX ||
        query_heads > INT32_MAX || kv_heads > INT32_MAX ||
        query_heads % kv_heads != 0 ||
        sequence > SIZE_MAX / query_heads / 64 / sizeof(float)) return 0;
    const size_t query_bytes = batch * query_heads * sequence * 64 * sizeof(float);
    const size_t key_bytes = batch * kv_heads * sequence * 64 * sizeof(float);
    id<MTLBuffer> query_buffer = [cached_device newBufferWithBytes:query
        length:query_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> key_buffer = [cached_device newBufferWithBytes:key
        length:key_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> value_buffer = [cached_device newBufferWithBytes:value
        length:key_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [cached_device newBufferWithLength:query_bytes
        options:MTLResourceStorageModeShared];
    if (query_buffer == nil || key_buffer == nil || value_buffer == nil ||
        output_buffer == nil) return 0;

    TiledVisionSdpaParams params = {0};
    params.batch = (int32_t)batch;
    params.heads = (int32_t)query_heads;
    params.head_dim = 64;
    params.query_length = (int32_t)sequence;
    params.key_length = (int32_t)sequence;
    params.gqa_factor = (int32_t)(query_heads / kv_heads);
    params.scale = scale;
    params.softcapping = 1.0f;
    params.query_blocks = (int32_t)(sequence / 32);
    params.key_blocks = (int32_t)(sequence / 32);
    params.query_blocks_aligned = params.query_blocks;
    params.key_blocks_aligned = params.key_blocks;
    params.query_strides[0] = (int64_t)(query_heads * sequence * 64);
    params.query_strides[1] = (int64_t)(sequence * 64);
    params.query_strides[2] = 64;
    params.key_strides[0] = (int64_t)(kv_heads * sequence * 64);
    params.key_strides[1] = (int64_t)(sequence * 64);
    params.key_strides[2] = 64;
    memcpy(params.value_strides, params.key_strides, sizeof(params.key_strides));
    memcpy(params.output_strides, params.query_strides, sizeof(params.query_strides));

    id<MTLCommandBuffer> command = [cached_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = command == nil ? nil :
        [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:cached_tiled_vision_pipeline];
    [encoder setBuffer:query_buffer offset:0 atIndex:0];
    [encoder setBuffer:key_buffer offset:0 atIndex:1];
    [encoder setBuffer:value_buffer offset:0 atIndex:2];
    [encoder setBuffer:output_buffer offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(params.query_blocks, query_heads, batch)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, output_buffer.contents, query_bytes);
    return 1;
}

static int metal_backend_run_tiled_vision_f16(const uint16_t *query,
                                              const uint16_t *key,
                                              const uint16_t *value,
                                              uint16_t *output, size_t batch,
                                              size_t query_heads, size_t kv_heads,
                                              size_t sequence, float scale) {
    if (cached_tiled_vision_f16_pipeline == nil || batch != 1 || sequence == 0 ||
        sequence % 32 != 0 || sequence > INT32_MAX ||
        query_heads > INT32_MAX || kv_heads > INT32_MAX ||
        query_heads % kv_heads != 0 ||
        sequence > SIZE_MAX / query_heads / 64 / sizeof(uint16_t)) return 0;
    const size_t query_bytes = batch * query_heads * sequence * 64 * sizeof(uint16_t);
    const size_t key_bytes = batch * kv_heads * sequence * 64 * sizeof(uint16_t);
    id<MTLBuffer> query_buffer = [cached_device newBufferWithBytes:query
        length:query_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> key_buffer = [cached_device newBufferWithBytes:key
        length:key_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> value_buffer = [cached_device newBufferWithBytes:value
        length:key_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [cached_device newBufferWithLength:query_bytes
        options:MTLResourceStorageModeShared];
    if (query_buffer == nil || key_buffer == nil || value_buffer == nil ||
        output_buffer == nil) return 0;

    TiledVisionSdpaParams params = {0};
    params.batch = (int32_t)batch;
    params.heads = (int32_t)query_heads;
    params.head_dim = 64;
    params.query_length = (int32_t)sequence;
    params.key_length = (int32_t)sequence;
    params.gqa_factor = (int32_t)(query_heads / kv_heads);
    params.scale = scale;
    params.softcapping = 1.0f;
    params.query_blocks = (int32_t)(sequence / 32);
    params.key_blocks = (int32_t)(sequence / 32);
    params.query_blocks_aligned = params.query_blocks;
    params.key_blocks_aligned = params.key_blocks;
    params.query_strides[0] = (int64_t)(query_heads * sequence * 64);
    params.query_strides[1] = (int64_t)(sequence * 64);
    params.query_strides[2] = 64;
    params.key_strides[0] = (int64_t)(kv_heads * sequence * 64);
    params.key_strides[1] = (int64_t)(sequence * 64);
    params.key_strides[2] = 64;
    memcpy(params.value_strides, params.key_strides, sizeof(params.key_strides));
    memcpy(params.output_strides, params.query_strides, sizeof(params.query_strides));

    id<MTLCommandBuffer> command = [cached_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = command == nil ? nil :
        [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:cached_tiled_vision_f16_pipeline];
    [encoder setBuffer:query_buffer offset:0 atIndex:0];
    [encoder setBuffer:key_buffer offset:0 atIndex:1];
    [encoder setBuffer:value_buffer offset:0 atIndex:2];
    [encoder setBuffer:output_buffer offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(params.query_blocks, query_heads, batch)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, output_buffer.contents, query_bytes);
    return 1;
}

int tensor_attention_metal_encode_tiled_f32(void *command_buffer_value,
                                             void *query_buffer_value,
                                             void *key_buffer_value,
                                             void *value_buffer_value,
                                             void *output_buffer_value,
                                             size_t heads, size_t sequence,
                                             float scale) {
    @autoreleasepool {
    if (!initialize_metal() || cached_tiled_vision_pipeline == nil ||
        command_buffer_value == NULL || query_buffer_value == NULL ||
        key_buffer_value == NULL || value_buffer_value == NULL ||
        output_buffer_value == NULL || heads == 0 || sequence == 0 ||
        heads > INT32_MAX || sequence > INT32_MAX || sequence % 32 != 0 ||
        sequence > SIZE_MAX / heads / 64 / sizeof(float)) return 0;
    TiledVisionSdpaParams params = {0};
    params.batch = 1;
    params.heads = (int32_t)heads;
    params.head_dim = 64;
    params.query_length = (int32_t)sequence;
    params.key_length = (int32_t)sequence;
    params.gqa_factor = 1;
    params.scale = scale;
    params.softcapping = 1.0f;
    params.query_blocks = (int32_t)(sequence / 32);
    params.key_blocks = params.query_blocks;
    params.query_blocks_aligned = params.query_blocks;
    params.key_blocks_aligned = params.key_blocks;
    params.query_strides[0] = (int64_t)(heads * sequence * 64);
    params.query_strides[1] = 64;
    params.query_strides[2] = (int64_t)(heads * 64);
    memcpy(params.key_strides, params.query_strides, sizeof(params.query_strides));
    memcpy(params.value_strides, params.query_strides, sizeof(params.query_strides));
    memcpy(params.output_strides, params.query_strides, sizeof(params.query_strides));
    id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)command_buffer_value;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:cached_tiled_vision_pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)query_buffer_value offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)key_buffer_value offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)value_buffer_value offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)output_buffer_value offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(params.query_blocks, heads, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [encoder endEncoding];
    if (!reported_vision_sdpa_path && timing_enabled()) {
        fprintf(stderr, "timing_vision_sdpa_kernel=candle_tiled_bq32_bk32\n");
        reported_vision_sdpa_path = 1;
    }
    return 1;
    }
}

int tensor_attention_metal_encode_tiled_f16(void *command_buffer_value,
                                             void *query_buffer_value,
                                             void *key_buffer_value,
                                             void *value_buffer_value,
                                             void *output_buffer_value,
                                             size_t heads, size_t sequence,
                                             float scale) {
    @autoreleasepool {
    if (!initialize_metal() || cached_tiled_vision_f16_pipeline == nil ||
        command_buffer_value == NULL || query_buffer_value == NULL ||
        key_buffer_value == NULL || value_buffer_value == NULL ||
        output_buffer_value == NULL || heads == 0 || sequence == 0 ||
        heads > INT32_MAX || sequence > INT32_MAX || sequence % 32 != 0 ||
        sequence > SIZE_MAX / heads / 64) return 0;
    TiledVisionSdpaParams params = {0};
    params.batch = 1;
    params.heads = (int32_t)heads;
    params.head_dim = 64;
    params.query_length = (int32_t)sequence;
    params.key_length = (int32_t)sequence;
    params.gqa_factor = 1;
    params.scale = scale;
    params.softcapping = 1.0f;
    params.query_blocks = (int32_t)(sequence / 32);
    params.key_blocks = params.query_blocks;
    params.query_blocks_aligned = params.query_blocks;
    params.key_blocks_aligned = params.key_blocks;
    /* Buffers are [sequence, heads, head_dim]; use non-contiguous head/seq strides. */
    params.query_strides[0] = (int64_t)(heads * sequence * 64);
    params.query_strides[1] = 64;
    params.query_strides[2] = (int64_t)(heads * 64);
    memcpy(params.key_strides, params.query_strides, sizeof(params.query_strides));
    memcpy(params.value_strides, params.query_strides, sizeof(params.query_strides));
    memcpy(params.output_strides, params.query_strides, sizeof(params.query_strides));
    id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)command_buffer_value;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:cached_tiled_vision_f16_pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)query_buffer_value offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)key_buffer_value offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)value_buffer_value offset:0 atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)output_buffer_value offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(params.query_blocks, heads, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [encoder endEncoding];
    if (!reported_vision_sdpa_f16_path && timing_enabled()) {
        fprintf(stderr, "timing_vision_sdpa_kernel_fp16=candle_tiled_bq32_bk32\n");
        reported_vision_sdpa_f16_path = 1;
    }
    return 1;
    }
}

static int metal_backend_run(const float *query, const float *key,
                             const float *value, const float *additive_mask,
                             float *output, size_t batch, size_t query_heads,
                             size_t kv_heads, size_t query_length,
                             size_t kv_length, size_t head_dim,
                             size_t value_dim, float scale, int causal) {
    @autoreleasepool {
    if (query == NULL || key == NULL || value == NULL || output == NULL ||
        batch == 0 || query_heads == 0 || kv_heads == 0 || query_length == 0 ||
        kv_length == 0 || head_dim == 0 || value_dim == 0 ||
        query_heads % kv_heads != 0 || batch > UINT32_MAX ||
        query_heads > UINT32_MAX || kv_heads > UINT32_MAX ||
        query_length > UINT32_MAX || kv_length > UINT32_MAX ||
        head_dim > UINT32_MAX || value_dim > UINT32_MAX ||
        value_dim > 256 ||
        !initialize_metal()) return 0;

    const uint32_t batch32 = (uint32_t)batch;
    const uint32_t query_heads32 = (uint32_t)query_heads;
    const uint32_t kv_heads32 = (uint32_t)kv_heads;
    const uint32_t query_length32 = (uint32_t)query_length;
    const uint32_t kv_length32 = (uint32_t)kv_length;
    const uint32_t head_dim32 = (uint32_t)head_dim;
    const uint32_t value_dim32 = (uint32_t)value_dim;

    size_t kv_rows, query_count, key_count, value_count;
    size_t output_count, mask_count = 0;
    if (!checked_product(batch, query_heads, &output_count) ||
        !checked_product(output_count, query_length, &output_count) ||
        !checked_product(batch, query_heads, &query_count) ||
        !checked_product(query_count, query_length, &query_count) ||
        !checked_product(query_count, head_dim, &query_count) ||
        !checked_product(batch, kv_heads, &kv_rows) ||
        !checked_product(kv_rows, kv_length, &key_count) ||
        !checked_product(key_count, head_dim, &key_count) ||
        !checked_product(kv_rows, kv_length, &value_count) ||
        !checked_product(value_count, value_dim, &value_count) ||
        !checked_product(output_count, value_dim, &output_count) ||
        (additive_mask != NULL &&
         !checked_product(query_length, kv_length, &mask_count))) return 0;
    size_t query_bytes, key_bytes, value_bytes, mask_bytes = 0;
    if (!checked_product(query_count, sizeof(float), &query_bytes) ||
        !checked_product(key_count, sizeof(float), &key_bytes) ||
        !checked_product(value_count, sizeof(float), &value_bytes) ||
        (additive_mask != NULL &&
         !checked_product(mask_count, sizeof(float), &mask_bytes))) return 0;

    if (additive_mask == NULL && causal == 0 && head_dim == 64 &&
        value_dim == 64 && query_length == kv_length &&
        query_length % 32 == 0) {
        const int tiled_ok = metal_backend_run_tiled_vision(query, key, value,
            output, batch, query_heads, kv_heads, query_length, scale);
        if (!reported_vision_sdpa_path && timing_enabled()) {
            fprintf(stderr, "timing_vision_sdpa_kernel=%s\n",
                    tiled_ok ? "candle_tiled_bq32_bk32" : "online_fallback");
            reported_vision_sdpa_path = 1;
        }
        if (tiled_ok) return 1;
    }

    SdpaParams params = {batch32, query_heads32, kv_heads32, query_length32,
                         kv_length32, head_dim32, value_dim32,
                         kv_length32 > query_length32 ? kv_length32 - query_length32 : 0,
                         causal != 0, additive_mask != NULL, scale};
    id<MTLBuffer> query_buffer = [cached_device
        newBufferWithBytes:query length:query_bytes
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> key_buffer = [cached_device
        newBufferWithBytes:key length:key_bytes
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> value_buffer = [cached_device
        newBufferWithBytes:value length:value_bytes
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> mask_buffer = additive_mask == NULL ? nil : [cached_device
        newBufferWithBytes:additive_mask length:mask_bytes
        options:MTLResourceStorageModeShared];
    if (query_buffer == nil || key_buffer == nil || value_buffer == nil ||
        (additive_mask != NULL && mask_buffer == nil)) return 0;

    size_t byte_count;
    if (!checked_product(output_count, sizeof(float), &byte_count)) return 0;
    id<MTLBuffer> output_buffer = [cached_device newBufferWithLength:byte_count
        options:MTLResourceStorageModeShared];
    if (output_buffer == nil) return 0;

    id<MTLCommandBuffer> command_buffer = [cached_queue commandBuffer];
    if (command_buffer == nil) return 0;
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:cached_online_pipeline];
    [encoder setBuffer:query_buffer offset:0 atIndex:0];
    [encoder setBuffer:key_buffer offset:0 atIndex:1];
    [encoder setBuffer:value_buffer offset:0 atIndex:2];
    [encoder setBuffer:mask_buffer offset:0 atIndex:3];
    [encoder setBuffer:output_buffer offset:0 atIndex:4];
    [encoder setBytes:&params length:sizeof(params) atIndex:5];
    const MTLSize grid = MTLSizeMake(query_length, query_heads, batch);
    const MTLSize group = MTLSizeMake(32, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, [output_buffer contents], output_count * sizeof(float));
    return 1;
    }
}

static int metal_backend_run_f16(const uint16_t *query, const uint16_t *key,
                                 const uint16_t *value, const uint16_t *mask,
                                 uint16_t *output, size_t batch, size_t q_heads,
                                 size_t kv_heads, size_t q_len, size_t kv_len,
                                 size_t head_dim, size_t value_dim,
                                 float scale, int causal) {
    @autoreleasepool {
    if (query == NULL || key == NULL || value == NULL || output == NULL ||
        batch == 0 || q_heads == 0 || kv_heads == 0 || q_len == 0 || kv_len == 0 ||
        head_dim == 0 || value_dim == 0 || value_dim > 256 || q_heads % kv_heads != 0 ||
        batch > UINT32_MAX || q_heads > UINT32_MAX || kv_heads > UINT32_MAX ||
        q_len > UINT32_MAX || kv_len > UINT32_MAX || head_dim > UINT32_MAX ||
        value_dim > UINT32_MAX || !initialize_metal()) return 0;
    if (mask == NULL && causal == 0 && head_dim == 64 && value_dim == 64 &&
        q_len == kv_len && q_len % 32 == 0) {
        const int tiled_ok = metal_backend_run_tiled_vision_f16(query, key, value,
            output, batch, q_heads, kv_heads, q_len, scale);
        if (!reported_vision_sdpa_f16_path && timing_enabled()) {
            fprintf(stderr, "timing_vision_sdpa_kernel_fp16=%s\n",
                    tiled_ok ? "candle_tiled_bq32_bk32" : "online_fallback");
            reported_vision_sdpa_f16_path = 1;
        }
        if (tiled_ok) return 1;
    }
    size_t qcount, kcount, vcount, ocount, mask_count = 0;
    if (!checked_product(batch, q_heads, &qcount) || !checked_product(qcount, q_len, &qcount) ||
        !checked_product(qcount, head_dim, &qcount) ||
        !checked_product(batch, kv_heads, &kcount) || !checked_product(kcount, kv_len, &kcount) ||
        !checked_product(kcount, head_dim, &kcount) ||
        !checked_product(batch, kv_heads, &vcount) || !checked_product(vcount, kv_len, &vcount) ||
        !checked_product(vcount, value_dim, &vcount) ||
        !checked_product(batch, q_heads, &ocount) || !checked_product(ocount, q_len, &ocount) ||
        !checked_product(ocount, value_dim, &ocount) ||
        (mask != NULL && !checked_product(q_len, kv_len, &mask_count))) return 0;
    size_t qb, kb, vb, ob, mb = 0;
    if (!checked_product(qcount, sizeof(uint16_t), &qb) ||
        !checked_product(kcount, sizeof(uint16_t), &kb) ||
        !checked_product(vcount, sizeof(uint16_t), &vb) ||
        !checked_product(ocount, sizeof(uint16_t), &ob) ||
        (mask != NULL && !checked_product(mask_count, sizeof(uint16_t), &mb))) return 0;
    SdpaParams params = {(uint32_t)batch, (uint32_t)q_heads, (uint32_t)kv_heads,
        (uint32_t)q_len, (uint32_t)kv_len, (uint32_t)head_dim, (uint32_t)value_dim,
        kv_len > q_len ? (uint32_t)(kv_len - q_len) : 0, causal != 0, mask != NULL, scale};
    id<MTLBuffer> qbfr = [cached_device newBufferWithBytes:query length:qb options:MTLResourceStorageModeShared];
    id<MTLBuffer> kbfr = [cached_device newBufferWithBytes:key length:kb options:MTLResourceStorageModeShared];
    id<MTLBuffer> vbfr = [cached_device newBufferWithBytes:value length:vb options:MTLResourceStorageModeShared];
    id<MTLBuffer> mbfr = mask == NULL ? nil : [cached_device newBufferWithBytes:mask length:mb options:MTLResourceStorageModeShared];
    id<MTLBuffer> obfr = [cached_device newBufferWithLength:ob options:MTLResourceStorageModeShared];
    if (qbfr == nil || kbfr == nil || vbfr == nil || obfr == nil || (mask != NULL && mbfr == nil)) return 0;
    id<MTLCommandBuffer> command = [cached_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = command == nil ? nil : [command computeCommandEncoder];
    if (encoder == nil) return 0;
    [encoder setComputePipelineState:cached_online_f16_pipeline];
    [encoder setBuffer:qbfr offset:0 atIndex:0]; [encoder setBuffer:kbfr offset:0 atIndex:1];
    [encoder setBuffer:vbfr offset:0 atIndex:2]; [encoder setBuffer:mbfr offset:0 atIndex:3];
    [encoder setBuffer:obfr offset:0 atIndex:4];
    [encoder setBytes:&params length:sizeof(params) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(q_len, q_heads, batch)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) return 0;
    memcpy(output, obfr.contents, ob);
    return 1;
    }
}

const TensorAttentionBackendOps *tensor_attention_metal_backend(void) {
    static const TensorAttentionBackendOps backend = {
        TENSOR_ATTENTION_BACKEND_METAL,
        "metal",
        metal_backend_available,
        metal_backend_run,
        metal_backend_run_f16,
        tensor_attention_metal_decode_f32,
        tensor_attention_metal_decode_sequence_available,
        tensor_attention_metal_decode_sequence_f32,
        tensor_attention_metal_decode_sequence_f16_available,
        tensor_attention_metal_decode_sequence_f16,
        tensor_attention_metal_decode_cache_clear
    };
    return &backend;
}
