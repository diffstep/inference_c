#ifndef TENSOR_GRAPH_H
#define TENSOR_GRAPH_H

#include "tensor_buffer.h"
#include "tensor_coreml.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TensorGraph TensorGraph;
typedef uint32_t TensorGraphValue;

typedef struct {
    const char *name;
    TensorBuffer *value;
} TensorGraphBinding;

/* Graph IR v1 is a static-shape DAG over F16/F32 values. Host execution uses
   tensor_ops; direct device-buffer execution currently supports F16 linear/add. */
int tensor_graph_create(TensorGraph **result, char *error, size_t error_capacity);
void tensor_graph_free(TensorGraph *graph);
int tensor_graph_add_input(TensorGraph *graph, const char *name,
                           TensorDType dtype, const uint64_t *shape,
                           size_t rank, TensorGraphValue *result,
                           char *error, size_t error_capacity);
int tensor_graph_add_constant(TensorGraph *graph, const char *name,
                              const TensorBuffer *value,
                              TensorGraphValue *result, char *error,
                              size_t error_capacity);
int tensor_graph_add_linear(TensorGraph *graph, const char *name,
                            TensorGraphValue input, TensorGraphValue weight,
                            TensorGraphValue bias, int has_bias,
                            TensorGraphValue *result, char *error,
                            size_t error_capacity);
int tensor_graph_add_add(TensorGraph *graph, const char *name,
                         TensorGraphValue left, TensorGraphValue right,
                         TensorGraphValue *result, char *error,
                         size_t error_capacity);
int tensor_graph_add_silu(TensorGraph *graph, const char *name,
                          TensorGraphValue input, TensorGraphValue *result,
                          char *error, size_t error_capacity);
int tensor_graph_add_rms_norm(TensorGraph *graph, const char *name,
                              TensorGraphValue input, TensorGraphValue weight,
                              float epsilon, TensorGraphValue *result,
                              char *error, size_t error_capacity);
int tensor_graph_add_layer_norm(TensorGraph *graph, const char *name,
                                TensorGraphValue input, TensorGraphValue scale,
                                TensorGraphValue bias, float epsilon,
                                TensorGraphValue *result, char *error,
                                size_t error_capacity);
int tensor_graph_mark_output(TensorGraph *graph, TensorGraphValue value,
                                char *error, size_t error_capacity);
/* Adds a precompiled Core ML subgraph as a single graph node. The model must
   remain open until graph execution has completed; Core ML is ANE-checked.
   It may run in parallel with independent Metal graph nodes. */
int tensor_graph_add_coreml(TensorGraph *graph, const char *name,
                            TensorCoreMLModel *model, const char *input_name,
                            TensorGraphValue input, const char *output_name,
                            TensorDType output_dtype,
                            const uint64_t *output_shape, size_t output_rank,
                            TensorGraphValue *result, char *error,
                            size_t error_capacity);
int tensor_graph_compile(TensorGraph *graph, char *error,
                         size_t error_capacity);
int tensor_graph_run(const TensorGraph *graph,
                     const TensorGraphBinding *inputs, size_t input_count,
                     TensorBuffer ***outputs, size_t *output_count,
                     char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
