#ifndef TENSOR_COREML_H
#define TENSOR_COREML_H

#include "tensor_buffer.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TensorCoreMLModel TensorCoreMLModel;
typedef struct {
    const char *name;
    TensorBuffer *value;
} TensorCoreMLBinding;

/* Loads a compiled .mlmodelc or compiles a .mlpackage/.mlmodel on demand.
   Core ML is restricted to CPU + Neural Engine, so it never takes Metal work. */
int tensor_coreml_available(void);
int tensor_coreml_model_open(const char *model_path, TensorCoreMLModel **result,
                             char *error, size_t error_capacity);
void tensor_coreml_model_close(TensorCoreMLModel *model);
int tensor_coreml_model_predict(TensorCoreMLModel *model,
                                const TensorCoreMLBinding *inputs,
                                size_t input_count,
                                const TensorCoreMLBinding *outputs,
                                size_t output_count,
                                char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
