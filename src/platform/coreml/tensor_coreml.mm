#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "tensor_coreml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#import <dispatch/dispatch.h>

@interface TensorCoreMLHandle : NSObject
@property(nonatomic, strong) MLModel *model;
@property(nonatomic, strong) NSURL *compiledURL;
@property(nonatomic) BOOL ownsCompiledURL;
@end

@implementation TensorCoreMLHandle
- (void)dealloc {
    if (self.ownsCompiledURL && self.compiledURL != nil)
        [[NSFileManager defaultManager] removeItemAtURL:self.compiledURL error:nil];
}
@end

struct TensorCoreMLModel {
    __strong TensorCoreMLHandle *handle;
};

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != nullptr && capacity != 0) std::snprintf(error, capacity, "%s", message);
}

static void set_ns_error(char *error, size_t capacity, NSString *prefix, NSError *cause) {
    NSString *message = cause.localizedDescription ?: @"unknown Core ML error";
    NSString *combined = [NSString stringWithFormat:@"%@: %@", prefix, message];
    set_error(error, capacity, combined.UTF8String);
}

static MLMultiArrayDataType coreml_dtype(TensorDType dtype) {
    switch (dtype) {
        case TENSOR_DTYPE_F16: return MLMultiArrayDataTypeFloat16;
        case TENSOR_DTYPE_F32: return MLMultiArrayDataTypeFloat32;
        default: return MLMultiArrayDataTypeInt32;
    }
}

static BOOL supported_dtype(TensorDType dtype) {
    return dtype == TENSOR_DTYPE_F16 || dtype == TENSOR_DTYPE_F32;
}

static NSArray<NSNumber *> *array_shape(const TensorBuffer *tensor) {
    NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:tensor_buffer_rank(tensor)];
    const uint64_t *dimensions = tensor_buffer_shape(tensor);
    for (size_t i = 0; i < tensor_buffer_rank(tensor); ++i)
        [shape addObject:@(dimensions[i])];
    return shape;
}

static NSArray<NSNumber *> *array_strides(const TensorBuffer *tensor) {
    const size_t rank = tensor_buffer_rank(tensor);
    NSMutableArray<NSNumber *> *strides = [NSMutableArray arrayWithCapacity:rank];
    for (size_t i = 0; i < rank; ++i) [strides addObject:@0];
    size_t stride = 1;
    const uint64_t *shape = tensor_buffer_shape(tensor);
    for (size_t i = rank; i-- > 0;) {
        strides[i] = @(stride);
        stride *= (size_t)shape[i];
    }
    return strides;
}

static BOOL array_matches(const MLMultiArray *array, const TensorBuffer *tensor) {
    if (array == nil || tensor == nullptr ||
        array.dataType != coreml_dtype(tensor_buffer_dtype(tensor)) ||
        array.shape.count != tensor_buffer_rank(tensor)) return NO;
    const uint64_t *shape = tensor_buffer_shape(tensor);
    for (size_t i = 0; i < tensor_buffer_rank(tensor); ++i)
        if (array.shape[i].unsignedLongLongValue != shape[i]) return NO;
    return YES;
}

static BOOL block_prefers_only_neural_engine(MLComputePlan *plan,
    MLModelStructureProgramBlock *block, char *error, size_t capacity) {
    for (MLModelStructureProgramOperation *operation in block.operations) {
        /* Constants are materialized by Core ML and have no compute device. */
        if ([operation.operatorName isEqualToString:@"const"]) continue;
        MLComputePlanDeviceUsage *usage =
            [plan computeDeviceUsageForMLProgramOperation:operation];
        if (usage == nil ||
            ![usage.preferredComputeDevice isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
            NSString *device = usage == nil ? @"unknown" :
                NSStringFromClass([usage.preferredComputeDevice class]);
            NSString *message = [NSString stringWithFormat:
                @"Core ML operation %@ maps to %@, not the Neural Engine",
                operation.operatorName, device];
            set_error(error, capacity, message.UTF8String);
            return NO;
        }
        for (MLModelStructureProgramBlock *nestedBlock in operation.blocks)
            if (!block_prefers_only_neural_engine(plan, nestedBlock, error, capacity))
                return NO;
    }
    return YES;
}

static BOOL plan_prefers_only_neural_engine(NSURL *modelURL,
                                             MLModelConfiguration *configuration,
                                             char *error, size_t capacity) {
    if (@available(macOS 14.4, *)) {
        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
        __block MLComputePlan *plan = nil;
        __block NSError *planError = nil;
        [MLComputePlan loadContentsOfURL:modelURL configuration:configuration
            completionHandler:^(MLComputePlan *value, NSError *cause) {
                plan = value;
                planError = cause;
                dispatch_semaphore_signal(semaphore);
            }];
        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
        if (plan == nil) {
            set_ns_error(error, capacity, @"Core ML compute plan unavailable", planError);
            return NO;
        }
        MLModelStructureProgram *program = plan.modelStructure.program;
        MLModelStructureProgramFunction *mainFunction = program.functions[@"main"];
        if (mainFunction == nil || mainFunction.block.operations.count == 0) {
            set_error(error, capacity, "Core ML model must contain a non-empty ML Program main function");
            return NO;
        }
        return block_prefers_only_neural_engine(plan, mainFunction.block,
                                                 error, capacity);
    }
    set_error(error, capacity, "Strict ANE validation requires macOS 14.4 or newer");
    return NO;
}

extern "C" int tensor_coreml_available(void) {
    if (@available(macOS 14.0, *)) {
        for (id<MLComputeDeviceProtocol> device in MLModel.availableComputeDevices)
            if ([device isKindOfClass:[MLNeuralEngineComputeDevice class]]) return 1;
    }
    return 0;
}

extern "C" int tensor_coreml_model_open(const char *model_path,
                                          TensorCoreMLModel **result,
                                          char *error, size_t capacity) {
    if (result != nullptr) *result = nullptr;
    if (model_path == nullptr || result == nullptr) {
        set_error(error, capacity, "invalid Core ML model path or result");
        return 0;
    }
    if (!tensor_coreml_available()) {
        set_error(error, capacity, "Core ML Neural Engine is unavailable (macOS 14 or newer is required)");
        return 0;
    }
    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:model_path];
        NSURL *sourceURL = [NSURL fileURLWithPath:path];
        NSURL *compiledURL = sourceURL;
        BOOL ownsCompiledURL = NO;
        if (![path.pathExtension.lowercaseString isEqualToString:@"mlmodelc"]) {
            NSError *compileError = nil;
            compiledURL = [MLModel compileModelAtURL:sourceURL error:&compileError];
            if (compiledURL == nil) {
                set_ns_error(error, capacity, @"Core ML model compilation failed", compileError);
                return 0;
            }
            ownsCompiledURL = YES;
        }
        MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        if (!plan_prefers_only_neural_engine(compiledURL, configuration, error, capacity)) {
            if (ownsCompiledURL) [[NSFileManager defaultManager] removeItemAtURL:compiledURL error:nil];
            return 0;
        }
        NSError *loadError = nil;
        MLModel *coreModel = [MLModel modelWithContentsOfURL:compiledURL
            configuration:configuration error:&loadError];
        if (coreModel == nil) {
            if (ownsCompiledURL) [[NSFileManager defaultManager] removeItemAtURL:compiledURL error:nil];
            set_ns_error(error, capacity, @"Core ML model load failed", loadError);
            return 0;
        }
        TensorCoreMLHandle *handle = [[TensorCoreMLHandle alloc] init];
        handle.model = coreModel;
        handle.compiledURL = compiledURL;
        handle.ownsCompiledURL = ownsCompiledURL;
        TensorCoreMLModel *model = new (std::nothrow) TensorCoreMLModel{handle};
        if (model == nullptr) {
            if (ownsCompiledURL) [[NSFileManager defaultManager] removeItemAtURL:compiledURL error:nil];
            set_error(error, capacity, "out of memory creating Core ML model handle");
            return 0;
        }
        *result = model;
        return 1;
    }
}

extern "C" void tensor_coreml_model_close(TensorCoreMLModel *model) {
    delete model;
}

extern "C" int tensor_coreml_model_predict(TensorCoreMLModel *model,
    const TensorCoreMLBinding *inputs, size_t input_count,
    const TensorCoreMLBinding *outputs, size_t output_count,
    char *error, size_t capacity) {
    if (model == nullptr || model->handle == nil || inputs == nullptr || input_count == 0 ||
        outputs == nullptr || output_count == 0) {
        set_error(error, capacity, "invalid Core ML prediction arguments");
        return 0;
    }
    @autoreleasepool {
        NSMutableDictionary<NSString *, MLFeatureValue *> *features =
            [NSMutableDictionary dictionaryWithCapacity:input_count];
        std::vector<void *> inputStorage;
        inputStorage.reserve(input_count);
        for (size_t i = 0; i < input_count; ++i) {
            if (inputs[i].name == nullptr || inputs[i].value == nullptr ||
                !supported_dtype(tensor_buffer_dtype(inputs[i].value))) {
                set_error(error, capacity, "Core ML inputs require named F16/F32 tensors");
                for (void *memory : inputStorage) std::free(memory);
                return 0;
            }
            const size_t bytes = tensor_buffer_byte_length(inputs[i].value);
            void *memory = std::malloc(bytes);
            if (memory == nullptr) {
                set_error(error, capacity, "out of memory preparing Core ML input");
                for (void *allocated : inputStorage) std::free(allocated);
                return 0;
            }
            inputStorage.push_back(memory);
            const int copied = tensor_buffer_storage(inputs[i].value) == TENSOR_BUFFER_STORAGE_DEVICE ?
                tensor_buffer_download(inputs[i].value, 0, memory, bytes, error, capacity) :
                (std::memcpy(memory, tensor_buffer_data(inputs[i].value), bytes), 1);
            if (!copied) {
                for (void *allocated : inputStorage) std::free(allocated);
                return 0;
            }
            NSError *arrayError = nil;
            MLMultiArray *array = [[MLMultiArray alloc] initWithDataPointer:memory
                shape:array_shape(inputs[i].value) dataType:coreml_dtype(tensor_buffer_dtype(inputs[i].value))
                strides:array_strides(inputs[i].value)
                deallocator:^(void *pointer) { std::free(pointer); } error:&arrayError];
            if (array == nil) {
                for (size_t j = i; j < inputStorage.size(); ++j) std::free(inputStorage[j]);
                set_ns_error(error, capacity, @"Core ML input array creation failed", arrayError);
                return 0;
            }
            inputStorage[i] = nullptr;
            NSString *key = [NSString stringWithUTF8String:inputs[i].name];
            features[key] = [MLFeatureValue featureValueWithMultiArray:array];
        }
        NSError *providerError = nil;
        MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:features error:&providerError];
        if (provider == nil) {
            set_ns_error(error, capacity, @"Core ML input provider creation failed", providerError);
            return 0;
        }
        NSError *predictionError = nil;
        id<MLFeatureProvider> prediction = [model->handle.model predictionFromFeatures:provider
            error:&predictionError];
        if (prediction == nil) {
            set_ns_error(error, capacity, @"Core ML prediction failed", predictionError);
            return 0;
        }
        for (size_t i = 0; i < output_count; ++i) {
            if (outputs[i].name == nullptr || outputs[i].value == nullptr ||
                !supported_dtype(tensor_buffer_dtype(outputs[i].value))) {
                set_error(error, capacity, "Core ML outputs require named F16/F32 tensors");
                return 0;
            }
            NSString *key = [NSString stringWithUTF8String:outputs[i].name];
            MLMultiArray *array = [prediction featureValueForName:key].multiArrayValue;
            if (!array_matches(array, outputs[i].value)) {
                set_error(error, capacity, "Core ML output shape or dtype does not match destination tensor");
                return 0;
            }
            const size_t bytes = tensor_buffer_byte_length(outputs[i].value);
            const size_t element_size = tensor_dtype_size(tensor_buffer_dtype(outputs[i].value));
            const uint64_t *shape = tensor_buffer_shape(outputs[i].value);
            std::vector<uint8_t> contiguous(bytes);
            const size_t rank = tensor_buffer_rank(outputs[i].value);
            std::vector<size_t> coordinates(rank, 0);
            const auto *source = static_cast<const uint8_t *>(array.dataPointer);
            for (size_t linear = 0; linear < tensor_buffer_element_count(outputs[i].value); ++linear) {
                size_t source_offset = 0;
                for (size_t axis = 0; axis < rank; ++axis)
                    source_offset += coordinates[axis] * array.strides[axis].unsignedLongLongValue;
                std::memcpy(contiguous.data() + linear * element_size,
                            source + source_offset * element_size, element_size);
                for (size_t axis = rank; axis-- > 0;) {
                    if (++coordinates[axis] < shape[axis]) break;
                    coordinates[axis] = 0;
                }
            }
            if (tensor_buffer_storage(outputs[i].value) == TENSOR_BUFFER_STORAGE_DEVICE) {
                if (!tensor_buffer_upload(outputs[i].value, 0, contiguous.data(), bytes, error, capacity)) return 0;
            } else {
                std::memcpy(tensor_buffer_data_mut(outputs[i].value), contiguous.data(), bytes);
            }
        }
        return 1;
    }
}
