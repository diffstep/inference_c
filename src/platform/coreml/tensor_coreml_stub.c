#include "tensor_coreml.h"

#include <stdio.h>

int tensor_coreml_available(void) { return 0; }

int tensor_coreml_model_open(const char *path, TensorCoreMLModel **result,
                             char *error, size_t capacity) {
    (void)path;
    if (result != NULL) *result = NULL;
    if (error != NULL && capacity != 0)
        snprintf(error, capacity, "Core ML support was not enabled when building the SDK");
    return 0;
}

void tensor_coreml_model_close(TensorCoreMLModel *model) { (void)model; }

int tensor_coreml_model_predict(TensorCoreMLModel *model,
                                const TensorCoreMLBinding *inputs,
                                size_t input_count,
                                const TensorCoreMLBinding *outputs,
                                size_t output_count,
                                char *error, size_t capacity) {
    (void)model; (void)inputs; (void)input_count;
    (void)outputs; (void)output_count;
    if (error != NULL && capacity != 0)
        snprintf(error, capacity, "Core ML support was not enabled when building the SDK");
    return 0;
}
