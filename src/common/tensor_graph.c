#include "tensor_graph.h"

#include "tensor_ops.h"
#include "tensor_device.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define GRAPH_MAX_RANK 16u

typedef enum {
    GRAPH_INPUT,
    GRAPH_CONSTANT,
    GRAPH_LINEAR,
    GRAPH_ADD,
    GRAPH_SILU,
    GRAPH_RMS_NORM,
    GRAPH_LAYER_NORM,
    GRAPH_COREML
} GraphOp;

typedef struct {
    GraphOp op;
    char *name;
    TensorGraphValue args[3];
    size_t arg_count;
    TensorDType dtype;
    size_t rank;
    uint64_t shape[GRAPH_MAX_RANK];
    float epsilon;
    TensorBuffer *constant;
    TensorCoreMLModel *coreml_model;
    char *coreml_input_name;
    char *coreml_output_name;
} GraphNode;

struct TensorGraph {
    GraphNode *nodes;
    size_t node_count;
    size_t node_capacity;
    TensorGraphValue *outputs;
    size_t output_count;
    size_t output_capacity;
    int compiled;
};

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity != 0) snprintf(error, capacity, "%s", message);
}

static char *copy_string(const char *text) {
    if (text == NULL) return NULL;
    const size_t size = strlen(text) + 1;
    char *copy = malloc(size);
    if (copy != NULL) memcpy(copy, text, size);
    return copy;
}

static int float_dtype(TensorDType dtype) {
    return dtype == TENSOR_DTYPE_F16 || dtype == TENSOR_DTYPE_F32;
}

static int valid_name(const char *name) {
    return name != NULL && name[0] != '\0';
}

static int append_node(TensorGraph *graph, GraphOp op, const char *name,
                       const TensorGraphValue *args, size_t arg_count,
                       TensorDType dtype, const uint64_t *shape, size_t rank,
                       float epsilon, TensorBuffer *constant,
                       TensorGraphValue *result, char *error, size_t capacity) {
    if (result != NULL) *result = UINT32_MAX;
    if (graph == NULL || result == NULL || !valid_name(name) ||
        !float_dtype(dtype) || rank == 0 || rank > GRAPH_MAX_RANK || shape == NULL ||
        graph->compiled || graph->node_count >= UINT32_MAX) {
        set_error(error, capacity, "invalid or immutable graph node arguments");
        return 0;
    }
    for (size_t i = 0; i < arg_count; ++i) {
        if ((size_t)args[i] >= graph->node_count) {
            set_error(error, capacity, "graph nodes may only reference existing values");
            return 0;
        }
    }
    for (size_t i = 0; i < graph->node_count; ++i) {
        if (strcmp(graph->nodes[i].name, name) == 0) {
            set_error(error, capacity, "graph value names must be unique");
            return 0;
        }
    }
    if (graph->node_count == graph->node_capacity) {
        size_t next = graph->node_capacity == 0 ? 16 : graph->node_capacity * 2;
        if (next < graph->node_capacity || next > SIZE_MAX / sizeof(*graph->nodes)) {
            set_error(error, capacity, "graph node capacity overflow");
            return 0;
        }
        GraphNode *grown = realloc(graph->nodes, next * sizeof(*grown));
        if (grown == NULL) {
            set_error(error, capacity, "out of memory adding graph node");
            return 0;
        }
        graph->nodes = grown;
        graph->node_capacity = next;
    }
    GraphNode node = {.op = op, .name = copy_string(name), .arg_count = arg_count,
                      .dtype = dtype, .rank = rank, .epsilon = epsilon,
                      .constant = constant};
    if (node.name == NULL) {
        set_error(error, capacity, "out of memory copying graph node name");
        return 0;
    }
    memcpy(node.shape, shape, rank * sizeof(*shape));
    if (arg_count != 0) memcpy(node.args, args, arg_count * sizeof(*args));
    graph->nodes[graph->node_count] = node;
    *result = (TensorGraphValue)graph->node_count++;
    return 1;
}

int tensor_graph_create(TensorGraph **result, char *error, size_t capacity) {
    if (result != NULL) *result = NULL;
    if (result == NULL) {
        set_error(error, capacity, "graph result is null");
        return 0;
    }
    TensorGraph *graph = calloc(1, sizeof(*graph));
    if (graph == NULL) {
        set_error(error, capacity, "out of memory creating graph");
        return 0;
    }
    *result = graph;
    return 1;
}

void tensor_graph_free(TensorGraph *graph) {
    if (graph == NULL) return;
    for (size_t i = 0; i < graph->node_count; ++i) {
        free(graph->nodes[i].name);
        free(graph->nodes[i].coreml_input_name);
        free(graph->nodes[i].coreml_output_name);
        tensor_buffer_free(graph->nodes[i].constant);
    }
    free(graph->nodes);
    free(graph->outputs);
    free(graph);
}

int tensor_graph_add_input(TensorGraph *graph, const char *name,
                           TensorDType dtype, const uint64_t *shape,
                           size_t rank, TensorGraphValue *result,
                           char *error, size_t capacity) {
    if (rank == 0 || rank > GRAPH_MAX_RANK || shape == NULL) {
        set_error(error, capacity, "graph inputs require a static, non-scalar shape");
        return 0;
    }
    for (size_t i = 0; i < rank; ++i) if (shape[i] == 0) {
        set_error(error, capacity, "graph inputs cannot have dynamic or empty dimensions");
        return 0;
    }
    return append_node(graph, GRAPH_INPUT, name, NULL, 0, dtype, shape, rank,
                       0.0f, NULL, result, error, capacity);
}

static int clone_host_buffer(const TensorBuffer *source, TensorBuffer **result,
                             char *error, size_t capacity) {
    if (source == NULL || tensor_buffer_storage(source) != TENSOR_BUFFER_STORAGE_HOST) {
        set_error(error, capacity, "graph constants must be host tensors");
        return 0;
    }
    if (!tensor_buffer_create(tensor_buffer_dtype(source), tensor_buffer_shape(source),
        tensor_buffer_rank(source), result, error, capacity)) return 0;
    memcpy(tensor_buffer_data_mut(*result), tensor_buffer_data(source),
           tensor_buffer_byte_length(source));
    return 1;
}

int tensor_graph_add_constant(TensorGraph *graph, const char *name,
                              const TensorBuffer *value,
                              TensorGraphValue *result, char *error,
                              size_t capacity) {
    if (value == NULL || tensor_buffer_rank(value) == 0 ||
        tensor_buffer_rank(value) > GRAPH_MAX_RANK || !float_dtype(tensor_buffer_dtype(value))) {
        set_error(error, capacity, "graph constants require rank-1 or higher F16/F32 host tensors");
        return 0;
    }
    TensorBuffer *copy = NULL;
    if (!clone_host_buffer(value, &copy, error, capacity)) return 0;
    if (!append_node(graph, GRAPH_CONSTANT, name, NULL, 0,
        tensor_buffer_dtype(copy), tensor_buffer_shape(copy), tensor_buffer_rank(copy),
        0.0f, copy, result, error, capacity)) {
        tensor_buffer_free(copy);
        return 0;
    }
    return 1;
}

static int value_valid(const TensorGraph *graph, TensorGraphValue value) {
    return graph != NULL && (size_t)value < graph->node_count;
}

static int same_shape(const GraphNode *a, const GraphNode *b) {
    return a->dtype == b->dtype && a->rank == b->rank &&
        memcmp(a->shape, b->shape, a->rank * sizeof(*a->shape)) == 0;
}

int tensor_graph_add_linear(TensorGraph *graph, const char *name,
                            TensorGraphValue input, TensorGraphValue weight,
                            TensorGraphValue bias, int has_bias,
                            TensorGraphValue *result, char *error,
                            size_t capacity) {
    if (!value_valid(graph, input) || !value_valid(graph, weight) ||
        (has_bias && !value_valid(graph, bias))) {
        set_error(error, capacity, "linear references an unknown graph value"); return 0;
    }
    const GraphNode *x = &graph->nodes[input], *w = &graph->nodes[weight];
    if (!float_dtype(x->dtype) || x->dtype != w->dtype || x->rank == 0 ||
        w->rank != 2 || x->shape[x->rank - 1] != w->shape[1]) {
        set_error(error, capacity, "linear requires matching F16/F32 input and rank-2 weight"); return 0;
    }
    if (has_bias) {
        const GraphNode *b = &graph->nodes[bias];
        if (b->dtype != x->dtype || b->rank != 1 || b->shape[0] != w->shape[0]) {
            set_error(error, capacity, "linear bias shape or dtype does not match weight"); return 0;
        }
    }
    uint64_t output_shape[GRAPH_MAX_RANK];
    memcpy(output_shape, x->shape, x->rank * sizeof(*output_shape));
    output_shape[x->rank - 1] = w->shape[0];
    TensorGraphValue args[3] = {input, weight, bias};
    return append_node(graph, GRAPH_LINEAR, name, args, has_bias ? 3 : 2,
        x->dtype, output_shape, x->rank, 0.0f, NULL, result, error, capacity);
}

int tensor_graph_add_add(TensorGraph *graph, const char *name,
                         TensorGraphValue left, TensorGraphValue right,
                         TensorGraphValue *result, char *error, size_t capacity) {
    if (!value_valid(graph, left) || !value_valid(graph, right) ||
        !same_shape(&graph->nodes[left], &graph->nodes[right])) {
        set_error(error, capacity, "add requires graph values with identical static shapes and dtypes");
        return 0;
    }
    const TensorGraphValue args[] = {left, right};
    const GraphNode *x = &graph->nodes[left];
    return append_node(graph, GRAPH_ADD, name, args, 2, x->dtype, x->shape,
        x->rank, 0.0f, NULL, result, error, capacity);
}

int tensor_graph_add_silu(TensorGraph *graph, const char *name,
                          TensorGraphValue input, TensorGraphValue *result,
                          char *error, size_t capacity) {
    if (!value_valid(graph, input)) { set_error(error, capacity, "SiLU input is unknown"); return 0; }
    const GraphNode *x = &graph->nodes[input];
    return append_node(graph, GRAPH_SILU, name, &input, 1, x->dtype, x->shape,
        x->rank, 0.0f, NULL, result, error, capacity);
}

int tensor_graph_add_rms_norm(TensorGraph *graph, const char *name,
                              TensorGraphValue input, TensorGraphValue weight,
                              float epsilon, TensorGraphValue *result,
                              char *error, size_t capacity) {
    if (!value_valid(graph, input) || !value_valid(graph, weight) ||
        !isfinite(epsilon) || epsilon <= 0.0f) {
        set_error(error, capacity, "invalid RMSNorm graph arguments"); return 0;
    }
    const GraphNode *x = &graph->nodes[input], *w = &graph->nodes[weight];
    if (x->dtype != w->dtype || w->rank != 1 || w->shape[0] != x->shape[x->rank - 1]) {
        set_error(error, capacity, "RMSNorm weight must match the input's final dimension"); return 0;
    }
    const TensorGraphValue args[] = {input, weight};
    return append_node(graph, GRAPH_RMS_NORM, name, args, 2, x->dtype, x->shape,
        x->rank, epsilon, NULL, result, error, capacity);
}

int tensor_graph_add_layer_norm(TensorGraph *graph, const char *name,
                                TensorGraphValue input, TensorGraphValue scale,
                                TensorGraphValue bias, float epsilon,
                                TensorGraphValue *result, char *error,
                                size_t capacity) {
    if (!value_valid(graph, input) || !value_valid(graph, scale) ||
        !value_valid(graph, bias) || !isfinite(epsilon) || epsilon <= 0.0f) {
        set_error(error, capacity, "invalid LayerNorm graph arguments"); return 0;
    }
    const GraphNode *x = &graph->nodes[input], *s = &graph->nodes[scale],
                    *b = &graph->nodes[bias];
    if (s->dtype != x->dtype || b->dtype != x->dtype || s->rank != 1 || b->rank != 1 ||
        s->shape[0] != x->shape[x->rank - 1] || b->shape[0] != s->shape[0]) {
        set_error(error, capacity, "LayerNorm scale and bias must match the input's final dimension");
        return 0;
    }
    const TensorGraphValue args[] = {input, scale, bias};
    return append_node(graph, GRAPH_LAYER_NORM, name, args, 3, x->dtype, x->shape,
        x->rank, epsilon, NULL, result, error, capacity);
}

int tensor_graph_add_coreml(TensorGraph *graph, const char *name,
                            TensorCoreMLModel *model, const char *input_name,
                            TensorGraphValue input, const char *output_name,
                            TensorDType output_dtype,
                            const uint64_t *output_shape, size_t output_rank,
                            TensorGraphValue *result, char *error,
                            size_t capacity) {
    if (graph == NULL || model == NULL || !valid_name(input_name) ||
        !valid_name(output_name) || !value_valid(graph, input) ||
        !float_dtype(output_dtype) || output_shape == NULL || output_rank == 0 ||
        output_rank > GRAPH_MAX_RANK) {
        set_error(error, capacity, "invalid Core ML graph node arguments");
        return 0;
    }
    for (size_t i = 0; i < output_rank; ++i) if (output_shape[i] == 0) {
        set_error(error, capacity, "Core ML graph output shape must be static and non-empty");
        return 0;
    }
    char *input_copy = copy_string(input_name);
    char *output_copy = copy_string(output_name);
    if (input_copy == NULL || output_copy == NULL) {
        free(input_copy);
        free(output_copy);
        set_error(error, capacity, "out of memory copying Core ML feature names");
        return 0;
    }
    if (!append_node(graph, GRAPH_COREML, name, &input, 1, output_dtype,
        output_shape, output_rank, 0.0f, NULL, result, error, capacity)) {
        free(input_copy);
        free(output_copy);
        return 0;
    }
    GraphNode *node = &graph->nodes[*result];
    node->coreml_model = model;
    node->coreml_input_name = input_copy;
    node->coreml_output_name = output_copy;
    return 1;
}

int tensor_graph_mark_output(TensorGraph *graph, TensorGraphValue value,
                             char *error, size_t capacity) {
    if (graph == NULL || graph->compiled || !value_valid(graph, value)) {
        set_error(error, capacity, "cannot mark invalid graph output"); return 0;
    }
    for (size_t i = 0; i < graph->output_count; ++i)
        if (graph->outputs[i] == value) return 1;
    if (graph->output_count == graph->output_capacity) {
        size_t next = graph->output_capacity == 0 ? 4 : graph->output_capacity * 2;
        TensorGraphValue *grown = realloc(graph->outputs, next * sizeof(*grown));
        if (grown == NULL) { set_error(error, capacity, "out of memory marking graph output"); return 0; }
        graph->outputs = grown; graph->output_capacity = next;
    }
    graph->outputs[graph->output_count++] = value;
    return 1;
}

int tensor_graph_compile(TensorGraph *graph, char *error, size_t capacity) {
    if (graph == NULL || graph->compiled || graph->node_count == 0 || graph->output_count == 0) {
        set_error(error, capacity, "graph must have nodes and outputs and may only compile once");
        return 0;
    }
    size_t input_count = 0;
    for (size_t i = 0; i < graph->node_count; ++i) {
        GraphNode *node = &graph->nodes[i];
        if (node->op == GRAPH_INPUT) input_count++;
        if ((node->op == GRAPH_INPUT || node->op == GRAPH_CONSTANT) && node->arg_count != 0) {
            set_error(error, capacity, "input and constant nodes cannot have dependencies"); return 0;
        }
        if (node->op == GRAPH_COREML &&
            (node->coreml_model == NULL || node->coreml_input_name == NULL ||
             node->coreml_output_name == NULL || node->arg_count != 1)) {
            set_error(error, capacity, "Core ML graph node is missing its model or feature names");
            return 0;
        }
        for (size_t a = 0; a < node->arg_count; ++a) if ((size_t)node->args[a] >= i) {
            set_error(error, capacity, "graph is not a valid topological DAG"); return 0;
        }
    }
    if (input_count == 0) { set_error(error, capacity, "graph has no inputs"); return 0; }
    graph->compiled = 1;
    return 1;
}

static int make_tensor(TensorDType dtype, const uint64_t *shape, size_t rank,
                       TensorBufferStorage storage, TensorBuffer **result,
                       char *error, size_t capacity) {
    if (storage == TENSOR_BUFFER_STORAGE_DEVICE)
        return tensor_buffer_create_device(dtype, shape, rank, result, error, capacity);
    return tensor_buffer_create(dtype, shape, rank, result, error, capacity);
}

static int copy_to_storage(const TensorBuffer *source, TensorBufferStorage storage,
                           TensorBuffer **result, char *error, size_t capacity) {
    if (!make_tensor(tensor_buffer_dtype(source), tensor_buffer_shape(source),
        tensor_buffer_rank(source), storage, result, error, capacity)) return 0;
    const size_t bytes = tensor_buffer_byte_length(source);
    if (tensor_buffer_storage(source) == storage) {
        if (storage == TENSOR_BUFFER_STORAGE_HOST)
            memcpy(tensor_buffer_data_mut(*result), tensor_buffer_data(source), bytes);
        else if (!tensor_buffer_copy_device(*result, 0, source, 0, bytes, error, capacity)) {
            tensor_buffer_free(*result); *result = NULL; return 0;
        }
    } else if (storage == TENSOR_BUFFER_STORAGE_DEVICE) {
        if (!tensor_buffer_upload(*result, 0, tensor_buffer_data(source), bytes, error, capacity)) {
            tensor_buffer_free(*result); *result = NULL; return 0;
        }
    } else {
        if (!tensor_buffer_download(source, 0, tensor_buffer_data_mut(*result), bytes,
                                    error, capacity)) {
            tensor_buffer_free(*result); *result = NULL; return 0;
        }
    }
    return 1;
}

static int device_op_supported(const GraphNode *node) {
    if (node->op == GRAPH_INPUT || node->op == GRAPH_CONSTANT) return 1;
    /* Only these tensor_ops currently enqueue work directly into device buffers. */
    if (node->op == GRAPH_COREML) return 1;
    return node->dtype == TENSOR_DTYPE_F16 &&
        (node->op == GRAPH_LINEAR || node->op == GRAPH_ADD);
}

typedef struct {
    const TensorGraph *graph;
    TensorBuffer **values;
    TensorBufferStorage storage;
    const size_t *ready_nodes;
    size_t ready_count;
    size_t next;
    int failed;
    char error[256];
    pthread_mutex_t mutex;
} GraphRunGroup;

static int execute_graph_node(GraphRunGroup *group, size_t index,
                              char *error, size_t capacity) {
    const GraphNode *node = &group->graph->nodes[index];
    TensorBuffer **values = group->values;
    if (!make_tensor(node->dtype, node->shape, node->rank, group->storage,
                     &values[index], error, capacity)) return 0;
    switch (node->op) {
    case GRAPH_LINEAR:
        return node->dtype == TENSOR_DTYPE_F16 ?
            (node->arg_count == 3 ? tensor_linear_f16(values[node->args[0]],
                values[node->args[1]], values[node->args[2]], values[index], error, capacity) :
                tensor_linear_f16(values[node->args[0]], values[node->args[1]], NULL,
                                  values[index], error, capacity)) :
            (node->arg_count == 3 ? tensor_linear_f32(values[node->args[0]],
                values[node->args[1]], values[node->args[2]], values[index], error, capacity) :
                tensor_linear_f32(values[node->args[0]], values[node->args[1]], NULL,
                                  values[index], error, capacity));
    case GRAPH_ADD:
        if (node->dtype == TENSOR_DTYPE_F16)
            return tensor_residual_add_f16(values[node->args[0]], values[node->args[1]],
                                           values[index], error, capacity);
        else {
            const float *a = tensor_buffer_data(values[node->args[0]]);
            const float *b = tensor_buffer_data(values[node->args[1]]);
            float *out = tensor_buffer_data_mut(values[index]);
            for (size_t j = 0; j < tensor_buffer_element_count(values[index]); ++j)
                out[j] = a[j] + b[j];
            return 1;
        }
    case GRAPH_SILU:
        return node->dtype == TENSOR_DTYPE_F16 ? tensor_silu_f16(values[node->args[0]],
            values[index], error, capacity) : tensor_silu_f32(values[node->args[0]],
            values[index], error, capacity);
    case GRAPH_RMS_NORM:
        return node->dtype == TENSOR_DTYPE_F16 ? tensor_rms_norm_f16(values[node->args[0]],
            values[node->args[1]], node->epsilon, values[index], error, capacity) :
            tensor_rms_norm_f32(values[node->args[0]], values[node->args[1]],
            node->epsilon, values[index], error, capacity);
    case GRAPH_LAYER_NORM:
        return node->dtype == TENSOR_DTYPE_F16 ? tensor_layer_norm_f16(values[node->args[0]],
            values[node->args[1]], values[node->args[2]], node->epsilon, values[index], error, capacity) :
            tensor_layer_norm_f32(values[node->args[0]], values[node->args[1]],
            values[node->args[2]], node->epsilon, values[index], error, capacity);
    case GRAPH_COREML: {
        TensorCoreMLBinding input = {.name = node->coreml_input_name,
                                     .value = values[node->args[0]]};
        TensorCoreMLBinding output = {.name = node->coreml_output_name,
                                      .value = values[index]};
        return tensor_coreml_model_predict(node->coreml_model, &input, 1,
            &output, 1, error, capacity);
    }
    case GRAPH_INPUT: case GRAPH_CONSTANT:
        break;
    }
    set_error(error, capacity, "invalid operation in compiled graph");
    return 0;
}

static void *graph_run_worker(void *opaque) {
    GraphRunGroup *group = opaque;
    for (;;) {
        pthread_mutex_lock(&group->mutex);
        if (group->failed || group->next == group->ready_count) {
            pthread_mutex_unlock(&group->mutex);
            return NULL;
        }
        const size_t index = group->ready_nodes[group->next++];
        pthread_mutex_unlock(&group->mutex);

        char local_error[256] = {0};
        if (!execute_graph_node(group, index, local_error, sizeof(local_error))) {
            pthread_mutex_lock(&group->mutex);
            if (!group->failed) {
                group->failed = 1;
                snprintf(group->error, sizeof(group->error), "%s",
                    local_error[0] == '\0' ? "graph operation failed" : local_error);
            }
            pthread_mutex_unlock(&group->mutex);
            return NULL;
        }
    }
}

static int run_ready_nodes(const TensorGraph *graph, TensorBuffer **values,
                           TensorBufferStorage storage, const size_t *ready,
                           size_t ready_count, char *error, size_t capacity) {
    GraphRunGroup group = {.graph = graph, .values = values, .storage = storage,
        .ready_nodes = ready, .ready_count = ready_count};
    pthread_mutex_init(&group.mutex, NULL);
    const size_t worker_count = ready_count < 4 ? ready_count : 4;
    pthread_t workers[3];
    size_t created = 0;
    while (created + 1 < worker_count &&
           pthread_create(&workers[created], NULL, graph_run_worker, &group) == 0)
        created++;
    graph_run_worker(&group);
    for (size_t i = 0; i < created; ++i) pthread_join(workers[i], NULL);
    pthread_mutex_destroy(&group.mutex);
    if (group.failed) {
        set_error(error, capacity, group.error);
        return 0;
    }
    return 1;
}

int tensor_graph_run(const TensorGraph *graph,
                     const TensorGraphBinding *inputs, size_t input_count,
                     TensorBuffer ***outputs, size_t *output_count,
                     char *error, size_t capacity) {
    if (outputs != NULL) *outputs = NULL;
    if (output_count != NULL) *output_count = 0;
    if (graph == NULL || !graph->compiled || outputs == NULL || output_count == NULL ||
        inputs == NULL || input_count == 0) {
        set_error(error, capacity, "invalid or uncompiled graph execution arguments"); return 0;
    }
    size_t expected_inputs = 0;
    TensorBufferStorage storage = TENSOR_BUFFER_STORAGE_HOST;
    int have_storage = 0;
    for (size_t i = 0; i < graph->node_count; ++i) if (graph->nodes[i].op == GRAPH_INPUT) {
        const GraphNode *node = &graph->nodes[i];
        const TensorGraphBinding *binding = NULL;
        for (size_t j = 0; j < input_count; ++j)
            if (inputs[j].name != NULL && strcmp(inputs[j].name, node->name) == 0) {
                binding = &inputs[j]; break;
            }
        if (binding == NULL || binding->value == NULL ||
            tensor_buffer_dtype(binding->value) != node->dtype ||
            tensor_buffer_rank(binding->value) != node->rank ||
            memcmp(tensor_buffer_shape(binding->value), node->shape,
                   node->rank * sizeof(*node->shape)) != 0) {
            set_error(error, capacity, "graph input binding is missing or has an incompatible shape/dtype");
            return 0;
        }
        if (!have_storage) { storage = tensor_buffer_storage(binding->value); have_storage = 1; }
        else if (storage != tensor_buffer_storage(binding->value)) {
            set_error(error, capacity, "graph inputs must use the same storage kind"); return 0;
        }
        expected_inputs++;
    }
    if (expected_inputs != input_count) {
        set_error(error, capacity, "graph input binding count does not match graph inputs"); return 0;
    }
    if (storage == TENSOR_BUFFER_STORAGE_DEVICE) {
        for (size_t i = 0; i < graph->node_count; ++i) {
            if (!device_op_supported(&graph->nodes[i])) {
                set_error(error, capacity,
                    "graph contains an operation without a device-buffer kernel");
                return 0;
            }
        }
    }
    TensorBuffer **values = calloc(graph->node_count, sizeof(*values));
    uint8_t *done = calloc(graph->node_count, sizeof(*done));
    if (values == NULL || done == NULL) {
        free(values);
        free(done);
        set_error(error, capacity, "out of memory preparing graph run");
        return 0;
    }
    int ok = 1;
    for (size_t i = 0; ok && i < graph->node_count; ++i) {
        const GraphNode *node = &graph->nodes[i];
        if (node->op == GRAPH_INPUT) {
            for (size_t j = 0; j < input_count; ++j)
                if (strcmp(inputs[j].name, node->name) == 0) { values[i] = inputs[j].value; break; }
            done[i] = 1;
            continue;
        }
        if (node->op == GRAPH_CONSTANT) {
            ok = copy_to_storage(node->constant, storage, &values[i], error, capacity);
            done[i] = ok;
            continue;
        }
    }
    size_t pending = 0;
    for (size_t i = 0; i < graph->node_count; ++i)
        if (!done[i]) pending++;
    while (ok && pending != 0) {
        size_t *ready = malloc(graph->node_count * sizeof(*ready));
        if (ready == NULL) {
            set_error(error, capacity, "out of memory scheduling graph nodes");
            ok = 0;
            break;
        }
        size_t ready_count = 0;
        for (size_t i = 0; i < graph->node_count; ++i) {
            const GraphNode *node = &graph->nodes[i];
            if (done[i] || node->op == GRAPH_INPUT || node->op == GRAPH_CONSTANT) continue;
            size_t arg = 0;
            for (; arg < node->arg_count; ++arg) if (!done[node->args[arg]]) break;
            if (arg == node->arg_count) ready[ready_count++] = i;
        }
        if (ready_count == 0) {
            set_error(error, capacity, "compiled graph has no schedulable node");
            free(ready);
            ok = 0;
            break;
        }
        ok = run_ready_nodes(graph, values, storage, ready, ready_count, error, capacity);
        if (ok) for (size_t i = 0; i < ready_count; ++i) {
            done[ready[i]] = 1;
            pending--;
        }
        free(ready);
    }
    if (ok) {
        TensorBuffer **result = calloc(graph->output_count, sizeof(*result));
        if (result == NULL) { set_error(error, capacity, "out of memory allocating graph outputs"); ok = 0; }
        else {
            for (size_t i = 0; i < graph->output_count; ++i) {
                if (!copy_to_storage(values[graph->outputs[i]], storage,
                                     &result[i], error, capacity)) {
                    for (size_t j = 0; j < i; ++j) tensor_buffer_free(result[j]);
                    free(result);
                    result = NULL;
                    ok = 0;
                    break;
                }
            }
            if (ok) { *outputs = result; *output_count = graph->output_count; }
        }
    }
    for (size_t i = 0; i < graph->node_count; ++i) {
        if (graph->nodes[i].op != GRAPH_INPUT) tensor_buffer_free(values[i]);
    }
    free(values);
    free(done);
    if (!ok && (error == NULL || capacity == 0 || error[0] == '\0'))
        set_error(error, capacity, "graph execution failed");
    return ok;
}
