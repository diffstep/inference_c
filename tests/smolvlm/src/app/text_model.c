#define _POSIX_C_SOURCE 200809L

#include "text_model.h"

#include "attention.h"
#include "tensor_compute_backend.h"
#include "tensor_buffer.h"
#include "tensor_store.h"
#include "tensor_workspace.h"
#include "allocator.h"
#include "tensor_memory.h"
#include "tensor_f16.h"
#include "tensor_quant.h"
#include "timing.h"

#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#endif

typedef struct {
    void *input_norm;
    void *post_norm;
    void *q_weight;
    void *k_weight;
    void *v_weight;
    void *o_weight;
    void *gate_weight;
    void *up_weight;
    void *down_weight;
} TextLayer;

struct TextModel {
    SmolVLMConfig config;
    TensorStore *weights;
    TensorStore *quantized_weights;
    void *embedding;
    void *lm_head;
    void *final_norm;
    TextLayer *layers;
    int fp16;
    TensorWeightQuantization weight_quant;
};

typedef struct {
    float *keys;
    float *values;
    size_t length;
    size_t capacity;
} TextKVCache;

static _Atomic size_t gpu_operation_count;
static _Atomic size_t cpu_fallback_count;
static _Atomic size_t gpu_attention_count;
static _Atomic size_t cpu_attention_count;

static double text_monotonic_ms(void) {
#if defined(__APPLE__)
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * (double)timebase.numer /
           (double)timebase.denom / 1000000.0;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
#endif
}

static void record_compute(int used_gpu) {
    if (used_gpu) atomic_fetch_add(&gpu_operation_count, 1);
    else atomic_fetch_add(&cpu_fallback_count, 1);
}

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static const char *resolve_weight_name(TensorStore *store, const char *name,
                                       char *alternate, size_t capacity) {
    if (tensor_store_find(store, name) != NULL) return name;
    static const char prefix[] = "model.";
    if (strncmp(name, prefix, sizeof(prefix) - 1) == 0) {
        const int length = snprintf(alternate, capacity, "encoder.%s",
                                    name + sizeof(prefix) - 1);
        if (length > 0 && (size_t)length < capacity &&
            tensor_store_find(store, alternate) != NULL) return alternate;
    }
    static const char head_prefix[] = "lm_head.";
    if (strncmp(name, head_prefix, sizeof(head_prefix) - 1) == 0) {
        const int length = snprintf(alternate, capacity, "encoder.%s", name);
        if (length > 0 && (size_t)length < capacity &&
            tensor_store_find(store, alternate) != NULL) return alternate;
    }
    return name;
}

static char *make_path(const char *directory, const char *filename) {
    const size_t directory_length = strlen(directory);
    const size_t filename_length = strlen(filename);
    const int separator = directory_length != 0 && directory[directory_length - 1] != '/';
    if (directory_length > SIZE_MAX - filename_length - (size_t)separator - 1) return NULL;
    char *path = malloc(directory_length + filename_length + (size_t)separator + 1);
    if (path == NULL) return NULL;
    memcpy(path, directory, directory_length);
    size_t offset = directory_length;
    if (separator) path[offset++] = '/';
    memcpy(path + offset, filename, filename_length + 1);
    return path;
}

static int load_weight(TextModel *model, const char *name, size_t expected,
                       void **result, char *error, size_t error_capacity) {
    char alternate[512];
    const char *resolved_name = resolve_weight_name(model->weights, name,
                                                     alternate, sizeof(alternate));
    const SafeTensorInfo *info = tensor_store_find(model->weights, resolved_name);
    const size_t element_size = model->fp16 ? sizeof(uint16_t) : sizeof(float);
    const int quantize = model->fp16 && model->weight_quant != 0 &&
        info != NULL && info->rank == 2 && strcmp(name, "model.text_model.embed_tokens.weight") != 0;
    if (info == NULL || (!model->fp16 && info->dtype != TENSOR_DTYPE_F32 &&
                         info->dtype != TENSOR_DTYPE_F16 && info->dtype != TENSOR_DTYPE_BF16) ||
        (model->fp16 && info->dtype != TENSOR_DTYPE_F16 && info->dtype != TENSOR_DTYPE_F32 &&
         info->dtype != TENSOR_DTYPE_BF16) ||
        expected > SIZE_MAX / element_size ||
        expected > SIZE_MAX / tensor_dtype_size(info->dtype) ||
        info->byte_length != expected * tensor_dtype_size(info->dtype)) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "missing or incompatible F32 weight: %s", name);
        return 0;
    }
    if (quantize) {
        if (info->shape[0] == 0 || info->shape[1] == 0 ||
            info->shape[0] > SIZE_MAX || info->shape[1] > SIZE_MAX ||
            (size_t)info->shape[0] > SIZE_MAX / (size_t)info->shape[1] ||
            (size_t)info->shape[0] * (size_t)info->shape[1] != expected) {
            set_error(error, error_capacity, "quantized linear tensor shape is invalid"); return 0;
        }
        char packed_alternate[512];
        const char *packed_name = model->quantized_weights == NULL ? name :
            resolve_weight_name(model->quantized_weights, name, packed_alternate,
                                sizeof(packed_alternate));
        const SafeTensorInfo *packed_info = model->quantized_weights == NULL ? NULL :
            tensor_store_find(model->quantized_weights, packed_name);
        const size_t block_bytes = model->weight_quant == TENSOR_WEIGHT_Q8_0 ? 34 : 18;
        if (packed_info != NULL) {
            if (packed_info->dtype != TENSOR_DTYPE_U8 || packed_info->rank != 3 ||
                packed_info->shape[0] != info->shape[0] ||
                packed_info->shape[1] != info->shape[1] / 32 ||
                packed_info->shape[2] != block_bytes || packed_info->byte_length == 0) {
                if (error && error_capacity) snprintf(error,error_capacity,"quantized sidecar tensor has incompatible layout: %s",name);
                return 0;
            }
            void *packed_data=tensor_aligned_allocate((size_t)packed_info->byte_length);
            if (!packed_data || !tensor_store_read_raw(model->quantized_weights,packed_name,
                    packed_data,(size_t)packed_info->byte_length,error,error_capacity)) {
                free(packed_data); return 0;
            }
            *result=packed_data; return 1;
        }
        uint16_t *source = tensor_aligned_allocate(expected * sizeof(uint16_t));
        TensorBuffer *packed = NULL;
        if (source == NULL || !tensor_store_read_f16(model->weights, resolved_name, source,
                expected * sizeof(uint16_t), error, error_capacity) ||
            !tensor_quantize_linear_weight_f16(source, (size_t)info->shape[0],
                (size_t)info->shape[1], model->weight_quant, &packed, error, error_capacity)) {
            free(source); tensor_buffer_free(packed); return 0;
        }
        const size_t packed_bytes = tensor_buffer_byte_length(packed);
        void *data = tensor_aligned_allocate(packed_bytes);
        if (data == NULL) {
            free(source); tensor_buffer_free(packed);
            set_error(error, error_capacity, "out of memory storing quantized model weight"); return 0;
        }
        memcpy(data, tensor_buffer_data(packed), packed_bytes);
        free(source); tensor_buffer_free(packed); *result = data; return 1;
    }
    void *data = tensor_aligned_allocate(expected * element_size);
    if (data == NULL) {
        set_error(error, error_capacity, "out of memory loading model weight");
        return 0;
    }
    const int loaded = model->fp16 ? tensor_store_read_f16(model->weights, resolved_name,
        data, expected * element_size, error, error_capacity) :
        tensor_store_read_f32(model->weights, resolved_name, data, expected * element_size,
                               error, error_capacity);
    if (!loaded) {
        free(data);
        return 0;
    }
    *result = data;
    return 1;
}

static int model_linear_f16(const TextModel *model, const uint16_t *input,
        const void *weight, uint16_t *output, size_t rows, size_t input_width,
        size_t output_width) {
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend == NULL) return 0;
    if (model->weight_quant == TENSOR_WEIGHT_Q8_0 && backend->linear_q8_0_f16)
        return backend->linear_q8_0_f16(input, weight, output, rows, input_width, output_width);
    if (model->weight_quant == TENSOR_WEIGHT_Q4_0 && backend->linear_q4_0_f16)
        return backend->linear_q4_0_f16(input, weight, output, rows, input_width, output_width);
    return model->weight_quant == 0 && backend->linear_f16 &&
        backend->linear_f16(input, weight, output, rows, input_width, output_width);
}

static int load_layer_weight(TextModel *model, size_t layer_index,
                             const char *suffix, size_t count, void **result,
                             char *error, size_t error_capacity) {
    char name[192];
    const int length = snprintf(name, sizeof(name), "model.text_model.layers.%zu.%s",
                                layer_index, suffix);
    return length > 0 && (size_t)length < sizeof(name) &&
        load_weight(model, name, count, result, error, error_capacity);
}

int text_model_open(const char *model_directory, TextModel **result,
                    char *error, size_t error_capacity) {
    if (result != NULL) *result = NULL;
    if (model_directory == NULL || result == NULL) {
        set_error(error, error_capacity, "invalid model path");
        return 0;
    }
    TextModel *model = calloc(1, sizeof(*model));
    const char *requested_dtype = getenv("TENSOR_DTYPE");
    if (requested_dtype != NULL && requested_dtype[0] != '\0' &&
        strcmp(requested_dtype, "fp32") != 0 && strcmp(requested_dtype, "fp16") != 0) {
        free(model);
        set_error(error, error_capacity, "TENSOR_DTYPE must be fp32 or fp16");
        return 0;
    }
    if (model != NULL) model->fp16 = requested_dtype != NULL && strcmp(requested_dtype, "fp16") == 0;
    const char *requested_quant = getenv("TENSOR_WEIGHT_QUANT");
    if (model != NULL) {
        if (requested_quant == NULL || requested_quant[0] == '\0' || strcmp(requested_quant, "fp16") == 0)
            model->weight_quant = 0;
        else if (strcmp(requested_quant, "q8_0") == 0) model->weight_quant = TENSOR_WEIGHT_Q8_0;
        else if (strcmp(requested_quant, "q4_0") == 0) model->weight_quant = TENSOR_WEIGHT_Q4_0;
        else { free(model); set_error(error,error_capacity,"TENSOR_WEIGHT_QUANT must be fp16, q8_0, or q4_0"); return 0; }
        if (model->weight_quant != 0 && !model->fp16) {
            free(model); set_error(error,error_capacity,"TENSOR_WEIGHT_QUANT=q8_0/q4_0 requires TENSOR_DTYPE=fp16"); return 0;
        }
    }
    if (model != NULL && model->fp16) {
        const TensorComputeBackend *fp16_backend = tensor_compute_preferred_backend();
        if (fp16_backend == NULL || fp16_backend->name == NULL ||
            fp16_backend->linear_f16 == NULL || fp16_backend->embedding_f16 == NULL ||
            fp16_backend->rms_norm_f16 == NULL || fp16_backend->rope_f16 == NULL ||
            fp16_backend->silu_multiply_f16 == NULL || fp16_backend->residual_add_f16 == NULL ||
            fp16_backend->argmax_f16 == NULL ||
            (model->weight_quant == 0 && fp16_backend->mlp_f16 == NULL) ||
            (model->weight_quant == TENSOR_WEIGHT_Q8_0 && fp16_backend->linear_q8_0_f16 == NULL) ||
            (model->weight_quant == TENSOR_WEIGHT_Q4_0 && fp16_backend->linear_q4_0_f16 == NULL)) {
            free(model);
            set_error(error, error_capacity,
                      "TENSOR_DTYPE=fp16 requires a complete FP16 compute backend");
            return 0;
        }
    }
    char *config_path = make_path(model_directory, "config.json");
    char *generation_path = make_path(model_directory, "generation_config.json");
    char *weight_path = make_path(model_directory, "model.safetensors");
    char *tokenizer_path = make_path(model_directory, "tokenizer.json");
    if (model == NULL || config_path == NULL || generation_path == NULL ||
        weight_path == NULL || tokenizer_path == NULL) {
        free(config_path);
        free(generation_path);
        free(weight_path);
        free(tokenizer_path);
        free(model);
        set_error(error, error_capacity, "out of memory preparing model paths");
        return 0;
    }
    const int config_ok = smolvlm_config_load(config_path, generation_path,
        &model->config, error, error_capacity);
    free(config_path);
    free(generation_path);
    free(tokenizer_path);
    if (!config_ok || !tensor_store_open_file(weight_path, &model->weights,
                                               error, error_capacity)) {
        free(weight_path);
        text_model_close(model);
        return 0;
    }
    if (model->weight_quant != 0) {
        char sidecar_name[64];
        snprintf(sidecar_name,sizeof(sidecar_name),"model.%s.safetensors",
            model->weight_quant == TENSOR_WEIGHT_Q8_0 ? "q8_0" : "q4_0");
        char *sidecar_path=make_path(model_directory,sidecar_name);
        FILE *sidecar_file=sidecar_path ? fopen(sidecar_path,"rb") : NULL;
        if (sidecar_file) {
            fclose(sidecar_file);
            if (!tensor_store_open_file(sidecar_path,&model->quantized_weights,error,error_capacity)) {
                free(sidecar_path); free(weight_path); text_model_close(model); return 0;
            }
        }
        free(sidecar_path);
    }
    free(weight_path);

    const size_t hidden = model->config.text.hidden_size;
    const size_t intermediate = model->config.text.intermediate_size;
    const size_t q_width = model->config.text.attention_heads * model->config.text.head_dim;
    const size_t kv_width = model->config.text.key_value_heads * model->config.text.head_dim;
    if (model->config.text.layers > SIZE_MAX / sizeof(*model->layers) ||
        hidden > SIZE_MAX / model->config.vocabulary_size ||
        hidden > SIZE_MAX / intermediate || q_width != hidden) {
        set_error(error, error_capacity, "text model dimensions overflow or are unsupported");
        text_model_close(model);
        return 0;
    }
    model->layers = calloc(model->config.text.layers, sizeof(*model->layers));
    int ok = model->layers != NULL &&
        load_weight(model, "model.text_model.embed_tokens.weight",
                    model->config.vocabulary_size * hidden, &model->embedding,
                    error, error_capacity);
    if (ok) {
        char head_alternate[512];
        const char *head_name = resolve_weight_name(model->weights,
            "lm_head.weight", head_alternate, sizeof(head_alternate));
        if (tensor_store_find(model->weights, head_name) != NULL) {
            ok = load_weight(model, "lm_head.weight",
                model->config.vocabulary_size * hidden, &model->lm_head,
                error, error_capacity);
        } else if (model->config.text.tie_word_embeddings) {
            model->lm_head = model->embedding;
        } else {
            set_error(error, error_capacity,
                      "lm_head.weight is missing and text_config.tie_word_embeddings is false");
            ok = 0;
        }
    }
    if (ok) ok = load_weight(model, "model.text_model.norm.weight", hidden,
                             &model->final_norm, error, error_capacity);
    for (size_t index = 0; ok && index < model->config.text.layers; index++) {
        TextLayer *layer = &model->layers[index];
        ok = load_layer_weight(model, index, "input_layernorm.weight", hidden,
                               &layer->input_norm, error, error_capacity) &&
             load_layer_weight(model, index, "post_attention_layernorm.weight", hidden,
                               &layer->post_norm, error, error_capacity) &&
             load_layer_weight(model, index, "self_attn.q_proj.weight", hidden * hidden,
                               &layer->q_weight, error, error_capacity) &&
             load_layer_weight(model, index, "self_attn.k_proj.weight", kv_width * hidden,
                               &layer->k_weight, error, error_capacity) &&
             load_layer_weight(model, index, "self_attn.v_proj.weight", kv_width * hidden,
                               &layer->v_weight, error, error_capacity) &&
             load_layer_weight(model, index, "self_attn.o_proj.weight", hidden * hidden,
                               &layer->o_weight, error, error_capacity) &&
             load_layer_weight(model, index, "mlp.gate_proj.weight", intermediate * hidden,
                               &layer->gate_weight, error, error_capacity) &&
             load_layer_weight(model, index, "mlp.up_proj.weight", intermediate * hidden,
                               &layer->up_weight, error, error_capacity) &&
             load_layer_weight(model, index, "mlp.down_proj.weight", hidden * intermediate,
                               &layer->down_weight, error, error_capacity);
    }
    if (!ok) {
        text_model_close(model);
        return 0;
    }
    *result = model;
    return 1;
}

void text_model_close(TextModel *model) {
    if (model == NULL) return;
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend != NULL && backend->release_cache != NULL) backend->release_cache();
    if (model->layers != NULL) {
        for (size_t index = 0; index < model->config.text.layers; index++) {
            TextLayer *layer = &model->layers[index];
            free(layer->input_norm);
            free(layer->post_norm);
            free(layer->q_weight);
            free(layer->k_weight);
            free(layer->v_weight);
            free(layer->o_weight);
            free(layer->gate_weight);
            free(layer->up_weight);
            free(layer->down_weight);
        }
    }
    free(model->layers);
    free(model->embedding);
    if (model->lm_head != model->embedding) free(model->lm_head);
    free(model->final_norm);
    tensor_store_close(model->weights);
    tensor_store_close(model->quantized_weights);
    smolvlm_config_free(&model->config);
    free(model);
}

static void linear(const float *input, const float *weight, float *output,
                   size_t rows, size_t input_width, size_t output_width) {
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend != NULL && backend->linear_f32 != NULL &&
        backend->linear_f32(input, weight, output, rows, input_width, output_width)) {
        record_compute(1);
        return;
    }
    record_compute(0);
#if defined(__APPLE__)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        (int)rows, (int)output_width, (int)input_width, 1.0f,
        input, (int)input_width, weight, (int)input_width, 0.0f,
        output, (int)output_width);
#else
    for (size_t row = 0; row < rows; row++) {
        for (size_t column = 0; column < output_width; column++) {
            float sum = 0.0f;
            for (size_t inner = 0; inner < input_width; inner++)
                sum += input[row * input_width + inner] *
                       weight[column * input_width + inner];
            output[row * output_width + column] = sum;
        }
    }
#endif
}

static void rms_norm(const float *input, const float *weight, float *output,
                     size_t rows, size_t width, float epsilon) {
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend != NULL && backend->rms_norm_f32 != NULL &&
        backend->rms_norm_f32(input, weight, output, rows, width, epsilon)) {
        record_compute(1);
        return;
    }
    record_compute(0);
    for (size_t row = 0; row < rows; row++) {
        const size_t offset = row * width;
        float square_sum = 0.0f;
        for (size_t column = 0; column < width; column++)
            square_sum += input[offset + column] * input[offset + column];
        const float inverse = 1.0f / sqrtf(square_sum / (float)width + epsilon);
        for (size_t column = 0; column < width; column++)
            output[offset + column] = input[offset + column] * inverse * weight[column];
    }
}

static void rms_norm_linear(const float *input, const float *norm_weight,
                            float *normalized, const float *linear_weight,
                            float *output, size_t rows, size_t input_width,
                            size_t output_width, float epsilon) {
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend != NULL && backend->rms_norm_linear_f32 != NULL &&
        backend->rms_norm_linear_f32(input, norm_weight, normalized,
            linear_weight, output, rows, input_width, output_width, epsilon)) {
        record_compute(1);
        return;
    }
    rms_norm(input, norm_weight, normalized, rows, input_width, epsilon);
    linear(normalized, linear_weight, output, rows, input_width, output_width);
}

static void apply_rope(float *query, float *key, size_t sequence,
                       size_t query_heads, size_t kv_heads, size_t head_dim,
                       double theta, size_t position_offset) {
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (position_offset == 0 && backend != NULL && backend->rope_f32 != NULL &&
        backend->rope_f32(query, key, sequence, query_heads, kv_heads,
                          head_dim, theta)) {
        record_compute(1);
        return;
    }
    record_compute(0);
    const size_t half = head_dim / 2;
    for (size_t position = 0; position < sequence; position++) {
        for (size_t dimension = 0; dimension < half; dimension++) {
            const float angle = (float)((double)(position_offset + position) /
                pow(theta, (2.0 * (double)dimension) / (double)head_dim));
            const float cosine = cosf(angle);
            const float sine = sinf(angle);
            for (size_t head = 0; head < query_heads; head++) {
                const size_t offset = (position * query_heads + head) * head_dim;
                const float left = query[offset + dimension];
                const float right = query[offset + dimension + half];
                query[offset + dimension] = left * cosine - right * sine;
                query[offset + dimension + half] = right * cosine + left * sine;
            }
            for (size_t head = 0; head < kv_heads; head++) {
                const size_t offset = (position * kv_heads + head) * head_dim;
                const float left = key[offset + dimension];
                const float right = key[offset + dimension + half];
                key[offset + dimension] = left * cosine - right * sine;
                key[offset + dimension + half] = right * cosine + left * sine;
            }
        }
    }
}

static int run_attention(const TextModel *model, const float *query,
                         const float *key, const float *value, size_t query_length,
                         size_t kv_length, size_t kv_capacity,
                         const float *output_weight, const float *residual_input,
                         float *output, int *fused_output,
                         TensorWorkspace *workspace,
                         char *error, size_t error_capacity) {
    const size_t q_heads = model->config.text.attention_heads;
    const size_t kv_heads = model->config.text.key_value_heads;
    const size_t dim = model->config.text.head_dim;
    if (fused_output != NULL) *fused_output = 0;
    const char *attention_backend = getenv("TENSOR_ATTENTION_BACKEND");
    if (attention_backend == NULL || attention_backend[0] == '\0')
        attention_backend = getenv("TENSOR_BACKEND");
    if (query_length == 1 &&
        (attention_backend == NULL || attention_backend[0] == '\0' ||
         strcmp(attention_backend, "auto") == 0 || strcmp(attention_backend, "metal") == 0) &&
        tensor_attention_decode_f32(query, key, value, output_weight,
                                          residual_input, output, q_heads,
                                          kv_heads, kv_length, kv_capacity, dim)) {
        atomic_fetch_add(&gpu_attention_count, 1);
        if (fused_output != NULL) *fused_output = 1;
        return 1;
    }
    const uint64_t q_shape[] = {1, q_heads, query_length, dim};
    const uint64_t kv_shape[] = {1, kv_heads, kv_length, dim};
    TensorBuffer *q = NULL, *k = NULL, *v = NULL, *out = NULL;
    int ok = tensor_buffer_create(TENSOR_DTYPE_F32, q_shape, 4, &q, error, error_capacity) &&
        tensor_buffer_create(TENSOR_DTYPE_F32, kv_shape, 4, &k, error, error_capacity) &&
        tensor_buffer_create(TENSOR_DTYPE_F32, kv_shape, 4, &v, error, error_capacity) &&
        tensor_buffer_create(TENSOR_DTYPE_F32, q_shape, 4, &out, error, error_capacity);
    if (!ok) goto cleanup;
    float *q_data = tensor_buffer_data_mut(q);
    float *k_data = tensor_buffer_data_mut(k);
    float *v_data = tensor_buffer_data_mut(v);
    for (size_t position = 0; position < query_length; position++) {
        for (size_t head = 0; head < q_heads; head++)
            memcpy(q_data + (head * query_length + position) * dim,
                   query + (position * q_heads + head) * dim, dim * sizeof(float));
    }
    for (size_t position = 0; position < kv_length; position++)
        for (size_t head = 0; head < kv_heads; head++) {
            memcpy(k_data + (head * kv_length + position) * dim,
                   key + (position * kv_heads + head) * dim, dim * sizeof(float));
            memcpy(v_data + (head * kv_length + position) * dim,
                   value + (position * kv_heads + head) * dim, dim * sizeof(float));
        }
    TensorAttentionBackend used;
    ok = tensor_sdpa_f32(q, k, v, NULL, 1.0f / sqrtf((float)dim),
                         query_length > 1, out,
                         workspace, &used, error, error_capacity);
    if (ok) {
        if (used == TENSOR_ATTENTION_BACKEND_CPU)
            atomic_fetch_add(&cpu_attention_count, 1);
        else
            atomic_fetch_add(&gpu_attention_count, 1);
    }
    if (ok) {
        const float *attn = tensor_buffer_data(out);
        for (size_t position = 0; position < query_length; position++)
            for (size_t head = 0; head < q_heads; head++)
                memcpy(output + (position * q_heads + head) * dim,
                       attn + (head * query_length + position) * dim, dim * sizeof(float));
    }
cleanup:
    tensor_buffer_free(out);
    tensor_buffer_free(v);
    tensor_buffer_free(k);
    tensor_buffer_free(q);
    return ok;
}

static int run_attention_f16(const TextModel *model, const uint16_t *query,
                             const uint16_t *key, const uint16_t *value,
                             size_t query_length, size_t kv_length,
                             const uint16_t *output_weight,
                             const uint16_t *residual, uint16_t *output,
                             TensorWorkspace *workspace,
                             char *error, size_t error_capacity) {
    const size_t q_heads = model->config.text.attention_heads;
    const size_t kv_heads = model->config.text.key_value_heads;
    const size_t dim = model->config.text.head_dim;
    const uint64_t qshape[] = {1, q_heads, query_length, dim};
    const uint64_t kvshape[] = {1, kv_heads, kv_length, dim};
    TensorBuffer *q = NULL, *k = NULL, *v = NULL, *o = NULL;
    int ok = tensor_buffer_create(TENSOR_DTYPE_F16, qshape, 4, &q, error, error_capacity) &&
        tensor_buffer_create(TENSOR_DTYPE_F16, kvshape, 4, &k, error, error_capacity) &&
        tensor_buffer_create(TENSOR_DTYPE_F16, kvshape, 4, &v, error, error_capacity) &&
        tensor_buffer_create(TENSOR_DTYPE_F16, qshape, 4, &o, error, error_capacity);
    if (!ok) goto done;
    uint16_t *qd = tensor_buffer_data_mut(q), *kd = tensor_buffer_data_mut(k);
    uint16_t *vd = tensor_buffer_data_mut(v);
    for (size_t pos = 0; pos < query_length; pos++)
        for (size_t head = 0; head < q_heads; head++)
            memcpy(qd + (head * query_length + pos) * dim,
                   query + (pos * q_heads + head) * dim, dim * sizeof(uint16_t));
    for (size_t pos = 0; pos < kv_length; pos++)
        for (size_t head = 0; head < kv_heads; head++) {
            memcpy(kd + (head * kv_length + pos) * dim,
                   key + (pos * kv_heads + head) * dim, dim * sizeof(uint16_t));
            memcpy(vd + (head * kv_length + pos) * dim,
                   value + (pos * kv_heads + head) * dim, dim * sizeof(uint16_t));
        }
    TensorAttentionBackend used;
    ok = tensor_sdpa_f16(q, k, v, NULL, 1.0f / sqrtf((float)dim),
                         query_length > 1, o, workspace, &used, error, error_capacity);
    if (ok) {
        const uint16_t *od = tensor_buffer_data(o);
        for (size_t pos = 0; pos < query_length; pos++)
            for (size_t head = 0; head < q_heads; head++)
                memcpy(output + (pos * q_heads + head) * dim,
                       od + (head * query_length + pos) * dim, dim * sizeof(uint16_t));
        const TensorComputeBackend *backend = tensor_compute_preferred_backend();
        if (backend == NULL ||
            !model_linear_f16(model, output, output_weight, output, query_length,
                                 model->config.text.hidden_size,
                                 model->config.text.hidden_size) ||
            backend->residual_add_f16 == NULL ||
            !backend->residual_add_f16(output, residual, output,
                                      query_length * model->config.text.hidden_size)) {
            set_error(error, error_capacity, "FP16 attention projection failed");
            ok = 0;
        }
    }
done:
    tensor_buffer_free(o); tensor_buffer_free(v); tensor_buffer_free(k); tensor_buffer_free(q);
    return ok;
}

static int forward_fp16_gpu(const TextModel *model, const uint32_t *tokens,
                            size_t sequence, uint16_t *last_logits, char *error,
                            size_t error_capacity, const float *image_embeddings,
                            size_t image_token_count, TextKVCache *cache) {
    const size_t hidden = model->config.text.hidden_size;
    const size_t q_heads = model->config.text.attention_heads;
    const size_t kv_heads = model->config.text.key_value_heads;
    const size_t dim = model->config.text.head_dim;
    const size_t kv_width = kv_heads * dim;
    const size_t intermediate = model->config.text.intermediate_size;
    const size_t count = sequence * hidden;
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend == NULL || backend->name == NULL ||
        backend->embedding_f16 == NULL || backend->linear_f16 == NULL ||
        backend->rms_norm_f16 == NULL || backend->rope_f16 == NULL ||
        backend->silu_multiply_f16 == NULL || backend->residual_add_f16 == NULL) {
        set_error(error, error_capacity, "FP16 inference requires complete FP16 compute support");
        return 0;
    }
    if (sequence == 0 || cache == NULL || cache->length > cache->capacity ||
        sequence > cache->capacity - cache->length || sequence > SIZE_MAX / hidden ||
        sequence > SIZE_MAX / intermediate || sequence * hidden > SIZE_MAX / sizeof(uint16_t) ||
        sequence * intermediate > SIZE_MAX / sizeof(uint16_t) ||
        q_heads * dim != hidden || dim % 2 != 0) {
        set_error(error, error_capacity, "prompt exceeds model context or has unsupported dimensions");
        return 0;
    }
    uint16_t *state = allocator_malloc(count * sizeof(uint16_t));
    uint16_t *norm = allocator_malloc(count * sizeof(uint16_t));
    uint16_t *query = allocator_malloc(count * sizeof(uint16_t));
    uint16_t *key = allocator_malloc(sequence * kv_width * sizeof(uint16_t));
    uint16_t *value = allocator_malloc(sequence * kv_width * sizeof(uint16_t));
    uint16_t *attention = allocator_malloc(count * sizeof(uint16_t));
    uint16_t *mlp = allocator_malloc(count * sizeof(uint16_t));
    uint16_t *gate = allocator_malloc(sequence * intermediate * sizeof(uint16_t));
    uint16_t *up = allocator_malloc(sequence * intermediate * sizeof(uint16_t));
    TensorWorkspace *workspace = NULL;
    int ok = state != NULL && norm != NULL && query != NULL && key != NULL && value != NULL &&
        attention != NULL && mlp != NULL && gate != NULL && up != NULL &&
        tensor_workspace_create(4096, &workspace, error, error_capacity);
    if (!ok) {
        set_error(error, error_capacity, "out of memory allocating FP16 GPU activations");
        goto done;
    }
    for (size_t row = 0; row < sequence; row++)
        if ((size_t)tokens[row] >= model->config.vocabulary_size) {
            set_error(error, error_capacity, "prompt token ID exceeds model vocabulary");
            ok = 0; goto done;
        }
    if (!backend->embedding_f16(model->embedding, tokens, state, sequence, hidden,
                               model->config.vocabulary_size)) {
        set_error(error, error_capacity, "FP16 embedding failed");
        ok = 0; goto done;
    }
    if (image_embeddings != NULL) {
        size_t image_index = 0;
        for (size_t row = 0; row < sequence; row++)
            if (tokens[row] == model->config.image_token_id) {
                if (image_index >= image_token_count) {
                    set_error(error, error_capacity, "image token count exceeds supplied image embeddings");
                    ok = 0; goto done;
                }
                for (size_t col = 0; col < hidden; col++)
                    state[row * hidden + col] = tensor_f32_to_f16(image_embeddings[image_index * hidden + col]);
                image_index++;
            }
        if (image_index != image_token_count) {
            set_error(error, error_capacity, "image token count does not match supplied image embeddings");
            ok = 0; goto done;
        }
    }
    for (size_t layer_index = 0; layer_index < model->config.text.layers; layer_index++) {
        const TextLayer *layer = &model->layers[layer_index];
        if (!backend->rms_norm_f16(state, layer->input_norm, norm, sequence, hidden,
                (float)model->config.text.rms_norm_eps) ||
            (model->weight_quant == 0 && backend->qkv_f16 != NULL && kv_width == hidden ?
                !backend->qkv_f16(norm, layer->q_weight, layer->k_weight,
                    layer->v_weight, NULL, NULL, NULL, query, key, value,
                    sequence, hidden, kv_width) :
                (!model_linear_f16(model,norm,layer->q_weight,query,sequence,hidden,hidden) ||
                 !model_linear_f16(model,norm,layer->k_weight,key,sequence,hidden,kv_width) ||
                 !model_linear_f16(model,norm,layer->v_weight,value,sequence,hidden,kv_width)))) {
        set_error(error, error_capacity, "FP16 QKV projection failed");
            ok = 0; goto done;
        }
        if (!backend->rope_f16(query, key, sequence, q_heads, kv_heads, dim,
                               model->config.text.rope_theta, cache->length)) {
        set_error(error, error_capacity, "FP16 rotary embedding failed");
            ok = 0; goto done;
        }
        const size_t past = cache->length;
        uint16_t *layer_keys = (uint16_t *)cache->keys + layer_index * cache->capacity * kv_width;
        uint16_t *layer_values = (uint16_t *)cache->values + layer_index * cache->capacity * kv_width;
        memcpy(layer_keys + past * kv_width, key, sequence * kv_width * sizeof(uint16_t));
        memcpy(layer_values + past * kv_width, value, sequence * kv_width * sizeof(uint16_t));
        if (!run_attention_f16(model, query, layer_keys, layer_values, sequence,
                past + sequence, layer->o_weight, state, attention, workspace,
                error, error_capacity)) {
            ok = 0; goto done;
        }
        memcpy(state, attention, count * sizeof(uint16_t));
        if (!backend->rms_norm_f16(state, layer->post_norm, norm, sequence, hidden,
                (float)model->config.text.rms_norm_eps) ||
            !model_linear_f16(model,norm,layer->gate_weight,gate,sequence,hidden,intermediate) ||
            !model_linear_f16(model,norm,layer->up_weight,up,sequence,hidden,intermediate) ||
            !backend->silu_multiply_f16(gate, up, gate, sequence * intermediate) ||
            !model_linear_f16(model,gate,layer->down_weight,mlp,sequence,intermediate,hidden) ||
            !backend->residual_add_f16(state, mlp, state, count)) {
            set_error(error, error_capacity, "FP16 MLP execution failed");
            ok = 0; goto done;
        }
    }
    cache->length += sequence;
    if (!backend->rms_norm_f16(state + (sequence - 1) * hidden, model->final_norm,
            norm, 1, hidden, (float)model->config.text.rms_norm_eps) ||
        !model_linear_f16(model,norm,model->lm_head,last_logits,1,hidden,
                             model->config.vocabulary_size)) {
            set_error(error, error_capacity, "FP16 output projection failed");
        ok = 0;
    }
done:
    tensor_workspace_free(workspace);
    allocator_free(up); allocator_free(gate); allocator_free(mlp); allocator_free(attention);
    allocator_free(value); allocator_free(key); allocator_free(query); allocator_free(norm);
    allocator_free(state);
    return ok;
}

static int forward(const TextModel *model, const uint32_t *tokens,
                   size_t sequence, void *last_logits, char *error,
                   size_t error_capacity, const float *image_embeddings,
                   size_t image_token_count, TextKVCache *cache) {
    if (model->fp16)
        return forward_fp16_gpu(model, tokens, sequence, last_logits, error,
            error_capacity, image_embeddings, image_token_count, cache);
    const size_t hidden = model->config.text.hidden_size;
    const size_t q_heads = model->config.text.attention_heads;
    const size_t kv_heads = model->config.text.key_value_heads;
    const size_t dim = model->config.text.head_dim;
    const size_t q_width = q_heads * dim;
    const size_t kv_width = kv_heads * dim;
    if (sequence == 0 || cache == NULL || cache->length > cache->capacity ||
        sequence > cache->capacity - cache->length ||
        cache->length + sequence > model->config.text.max_position_embeddings ||
        sequence > SIZE_MAX / hidden || q_width != hidden || dim % 2 != 0) {
        set_error(error, error_capacity, "prompt exceeds model context or has unsupported dimensions");
        return 0;
    }
    const size_t elements = sequence * hidden;
    float *hidden_state = allocator_malloc(elements * sizeof(float));
    float *normalized = allocator_malloc(elements * sizeof(float));
    float *residual = allocator_malloc(elements * sizeof(float));
    float *query = allocator_malloc(elements * sizeof(float));
    float *key = allocator_malloc(sequence * kv_width * sizeof(float));
    float *value = allocator_malloc(sequence * kv_width * sizeof(float));
    float *attention = allocator_malloc(elements * sizeof(float));
    float *intermediate = allocator_malloc(sequence * model->config.text.intermediate_size * sizeof(float));
    float *gate = allocator_malloc(sequence * model->config.text.intermediate_size * sizeof(float));
    float *mlp_output = allocator_malloc(elements * sizeof(float));
    TensorWorkspace *workspace = NULL;
    int ok = hidden_state != NULL && normalized != NULL && residual != NULL &&
        query != NULL && key != NULL && value != NULL && attention != NULL &&
        intermediate != NULL && gate != NULL && mlp_output != NULL &&
        tensor_workspace_create(4096, &workspace, error, error_capacity);
    if (!ok) {
        set_error(error, error_capacity, "out of memory allocating text activations");
        goto cleanup;
    }
    for (size_t index = 0; index < sequence; index++) {
        if ((size_t)tokens[index] >= model->config.vocabulary_size) {
            set_error(error, error_capacity, "prompt token ID exceeds model vocabulary");
            ok = 0;
            goto cleanup;
        }
    }
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    const int embedded_on_gpu = backend != NULL && backend->embedding_f32 != NULL &&
        backend->embedding_f32(model->embedding, tokens, hidden_state, sequence,
                               hidden, model->config.vocabulary_size);
    record_compute(embedded_on_gpu);
    if (!embedded_on_gpu) {
        for (size_t index = 0; index < sequence; index++)
            memcpy(hidden_state + index * hidden,
                   (const float *)model->embedding + (size_t)tokens[index] * hidden,
                   hidden * sizeof(float));
    }
    if (image_embeddings != NULL) {
        size_t image_index = 0;
        for (size_t position = 0; position < sequence; position++) {
            if (tokens[position] == model->config.image_token_id) {
                if (image_index >= image_token_count) {
                    set_error(error, error_capacity, "image token count exceeds supplied image embeddings");
                    ok = 0;
                    goto cleanup;
                }
                memcpy(hidden_state + position * hidden,
                       image_embeddings + image_index * hidden, hidden * sizeof(float));
                image_index++;
            }
        }
        if (image_index != image_token_count) {
            set_error(error, error_capacity, "image token count does not match supplied image embeddings");
            ok = 0;
            goto cleanup;
        }
    }
    for (size_t layer_index = 0; layer_index < model->config.text.layers; layer_index++) {
        const TextLayer *layer = &model->layers[layer_index];
        rms_norm_linear(hidden_state, layer->input_norm, normalized,
                        layer->q_weight, query, sequence, hidden, q_width,
                        (float)model->config.text.rms_norm_eps);
        linear(normalized, layer->k_weight, key, sequence, hidden, kv_width);
        linear(normalized, layer->v_weight, value, sequence, hidden, kv_width);
        apply_rope(query, key, sequence, q_heads, kv_heads, dim,
                   model->config.text.rope_theta, cache->length);
        float *layer_keys = cache->keys + layer_index * cache->capacity * kv_width;
        float *layer_values = cache->values + layer_index * cache->capacity * kv_width;
        memcpy(layer_keys + cache->length * kv_width, key,
               sequence * kv_width * sizeof(float));
        memcpy(layer_values + cache->length * kv_width, value,
               sequence * kv_width * sizeof(float));
        int attention_fused_output = 0;
        if (!run_attention(model, query, layer_keys, layer_values, sequence,
                           cache->length + sequence, cache->capacity, layer->o_weight,
                           hidden_state + (sequence - 1) * hidden, attention,
                           &attention_fused_output,
                           workspace, error, error_capacity)) {
            ok = 0;
            goto cleanup;
        }
        if (attention_fused_output) {
            memcpy(hidden_state + (sequence - 1) * hidden, attention,
                   hidden * sizeof(float));
        } else {
            linear(attention, layer->o_weight, residual, sequence, hidden, hidden);
            const int residual_on_gpu = backend != NULL && backend->residual_add_f32 != NULL &&
                backend->residual_add_f32(hidden_state, residual, hidden_state, elements);
            record_compute(residual_on_gpu);
            if (!residual_on_gpu)
                for (size_t index = 0; index < elements; index++)
                    hidden_state[index] += residual[index];
        }
        rms_norm_linear(hidden_state, layer->post_norm, normalized,
                        layer->gate_weight, gate, sequence, hidden,
                        model->config.text.intermediate_size,
                        (float)model->config.text.rms_norm_eps);
        linear(normalized, layer->up_weight, intermediate, sequence, hidden,
               model->config.text.intermediate_size);
        const size_t intermediate_count = sequence * model->config.text.intermediate_size;
        const int silu_on_gpu = backend != NULL && backend->silu_multiply_f32 != NULL &&
            backend->silu_multiply_f32(gate, intermediate, gate, intermediate_count);
        record_compute(silu_on_gpu);
        if (!silu_on_gpu)
            for (size_t index = 0; index < intermediate_count; index++) {
                const float value_at_index = gate[index];
                gate[index] = value_at_index / (1.0f + expf(-value_at_index)) * intermediate[index];
            }
        linear(gate, layer->down_weight, mlp_output, sequence,
               model->config.text.intermediate_size, hidden);
        const int mlp_residual_on_gpu = backend != NULL && backend->residual_add_f32 != NULL &&
            backend->residual_add_f32(hidden_state, mlp_output, hidden_state, elements);
        record_compute(mlp_residual_on_gpu);
        if (!mlp_residual_on_gpu)
            for (size_t index = 0; index < elements; index++)
                hidden_state[index] += mlp_output[index];
    }
    cache->length += sequence;
    rms_norm_linear(hidden_state + (sequence - 1) * hidden, model->final_norm,
                    normalized, model->lm_head, last_logits, 1, hidden,
                    model->config.vocabulary_size,
                    (float)model->config.text.rms_norm_eps);
cleanup:
    tensor_workspace_free(workspace);
    allocator_free(mlp_output);
    allocator_free(gate);
    allocator_free(intermediate);
    allocator_free(attention);
    allocator_free(value);
    allocator_free(key);
    allocator_free(query);
    allocator_free(residual);
    allocator_free(normalized);
    allocator_free(hidden_state);
    return ok;
}

static int append_token(uint32_t **tokens, size_t *count, size_t *capacity,
                        uint32_t token) {
    if (*count == *capacity) {
        if (*capacity >= SIZE_MAX / sizeof(**tokens)) return 0;
        size_t next_capacity = *capacity == 0 ? 8 : *capacity * 2;
        if (next_capacity < *capacity ||
            next_capacity > SIZE_MAX / sizeof(**tokens))
            next_capacity = SIZE_MAX / sizeof(**tokens);
        uint32_t *grown = allocator_realloc(*tokens,
                                            next_capacity * sizeof(**tokens));
        if (grown == NULL) return 0;
        *tokens = grown;
        *capacity = next_capacity;
    }
    (*tokens)[(*count)++] = token;
    return 1;
}

static int generate_encoded(TextModel *model, Tokenizer *tokenizer,
                            uint32_t *tokens, size_t token_count,
                            size_t max_new_tokens, const float *image_embeddings,
                            size_t image_token_count, char **result,
                            TextGenerationStats *stats,
                            char *error, size_t error_capacity) {
    if (stats != NULL) *stats = (TextGenerationStats){0};
    atomic_store(&gpu_operation_count, 0);
    atomic_store(&cpu_fallback_count, 0);
    atomic_store(&gpu_attention_count, 0);
    atomic_store(&cpu_attention_count, 0);
    int ok = token_count > 0 && token_count < model->config.text.max_position_embeddings;
    if (!ok) {
        free(tokens);
        set_error(error, error_capacity, "encoded prompt exceeds model context");
        return 0;
    }
    double setup_started = timing_start();
    const size_t vocab = model->config.vocabulary_size;
    void *logits = malloc(vocab * (model->fp16 ? sizeof(uint16_t) : sizeof(float)));
    const size_t kv_width = model->config.text.key_value_heads * model->config.text.head_dim;
    TextKVCache cache = {0};
    cache.capacity = token_count + max_new_tokens;
    if (cache.capacity < token_count) {
        free(tokens);
        free(logits);
        set_error(error, error_capacity, "KV cache dimensions overflow");
        return 0;
    }
    if (cache.capacity > model->config.text.max_position_embeddings)
        cache.capacity = model->config.text.max_position_embeddings;
    size_t cache_elements;
    if (model->config.text.layers > SIZE_MAX / cache.capacity ||
        model->config.text.layers * cache.capacity > SIZE_MAX / kv_width) {
        free(tokens);
        free(logits);
        set_error(error, error_capacity, "KV cache dimensions overflow");
        return 0;
    }
    cache_elements = model->config.text.layers * cache.capacity * kv_width;
    const size_t cache_element_size = model->fp16 ? sizeof(uint16_t) : sizeof(float);
    if (cache_elements > SIZE_MAX / cache_element_size / 2) {
        free(tokens);
        free(logits);
        set_error(error, error_capacity, "KV cache size overflows addressable memory");
        return 0;
    }
    cache.keys = malloc(cache_elements * cache_element_size);
    cache.values = malloc(cache_elements * cache_element_size);
    uint32_t *generated = NULL;
    size_t generated_count = 0;
    size_t generated_capacity = 0;
    if (logits == NULL || cache.keys == NULL || cache.values == NULL) {
        free(tokens);
        free(logits);
        free(cache.keys);
        free(cache.values);
        set_error(error, error_capacity, "out of memory allocating logits");
        return 0;
    }
    const size_t layer_count = model->config.text.layers;
    TensorAttentionDecoderLayer *decoder_layers = calloc(layer_count, sizeof(*decoder_layers));
    TensorAttentionDecoderLayerF16 *decoder_layers_f16 = calloc(layer_count, sizeof(*decoder_layers_f16));
    TensorAttentionDecoderLayerQuantF16 *decoder_layers_quant_f16 =
        calloc(layer_count, sizeof(*decoder_layers_quant_f16));
    const float **key_cache = calloc(layer_count, sizeof(*key_cache));
    const float **value_cache = calloc(layer_count, sizeof(*value_cache));
    const uint16_t **key_cache_f16 = calloc(layer_count, sizeof(*key_cache_f16));
    const uint16_t **value_cache_f16 = calloc(layer_count, sizeof(*value_cache_f16));
    if (decoder_layers == NULL || decoder_layers_f16 == NULL ||
        decoder_layers_quant_f16 == NULL || key_cache == NULL ||
        value_cache == NULL || key_cache_f16 == NULL || value_cache_f16 == NULL) {
        free(decoder_layers);
        free(decoder_layers_f16);
        free(decoder_layers_quant_f16);
        free(key_cache);
        free(value_cache);
        free(key_cache_f16);
        free(value_cache_f16);
        free(tokens);
        free(logits);
        free(cache.keys);
        free(cache.values);
        set_error(error, error_capacity, "out of memory preparing decoder state");
        return 0;
    }
    for (size_t index = 0; index < layer_count; index++) {
        const TextLayer *layer = &model->layers[index];
        decoder_layers[index] = (TensorAttentionDecoderLayer){
            layer->input_norm, layer->q_weight, layer->k_weight, layer->v_weight,
            layer->o_weight, layer->post_norm, layer->gate_weight,
            layer->up_weight, layer->down_weight
        };
        decoder_layers_f16[index] = (TensorAttentionDecoderLayerF16){
            layer->input_norm, layer->q_weight, layer->k_weight, layer->v_weight,
            layer->o_weight, layer->post_norm, layer->gate_weight,
            layer->up_weight, layer->down_weight
        };
        decoder_layers_quant_f16[index] = (TensorAttentionDecoderLayerQuantF16){
            .input_norm = layer->input_norm, .post_norm = layer->post_norm,
            .q_weight = layer->q_weight, .k_weight = layer->k_weight,
            .v_weight = layer->v_weight, .o_weight = layer->o_weight,
            .gate_weight = layer->gate_weight, .up_weight = layer->up_weight,
            .down_weight = layer->down_weight
        };
        key_cache[index] = (const float *)((const uint8_t *)cache.keys +
            index * cache.capacity * kv_width * cache_element_size);
        value_cache[index] = (const float *)((const uint8_t *)cache.values +
            index * cache.capacity * kv_width * cache_element_size);
        key_cache_f16[index] = (const uint16_t *)((const uint8_t *)cache.keys +
            index * cache.capacity * kv_width * cache_element_size);
        value_cache_f16[index] = (const uint16_t *)((const uint8_t *)cache.values +
            index * cache.capacity * kv_width * cache_element_size);
    }
    const char *requested_attention = getenv("TENSOR_ATTENTION_BACKEND");
    if (requested_attention == NULL || requested_attention[0] == '\0')
        requested_attention = getenv("TENSOR_BACKEND");
    const int attention_allows_accelerator = requested_attention == NULL ||
        requested_attention[0] == '\0' || strcmp(requested_attention, "auto") == 0 ||
        strcmp(requested_attention, "metal") == 0 || strcmp(requested_attention, "cuda") == 0;
    const char *disable_decode_sequence = getenv("TENSOR_DISABLE_DECODE_SEQUENCE");
    const int quantized_decode_available = model->weight_quant != 0 && model->fp16 &&
        tensor_attention_decode_sequence_quant_f16_available(model->weight_quant);
    const int use_accelerated_decode =
        (disable_decode_sequence == NULL || disable_decode_sequence[0] == '\0' ||
         strcmp(disable_decode_sequence, "0") == 0) &&
        tensor_compute_preferred_backend() != NULL &&
        attention_allows_accelerator && (model->weight_quant != 0 ?
            quantized_decode_available : model->fp16 ?
            tensor_attention_decode_sequence_f16_available() :
            tensor_attention_decode_sequence_available());
    if (stats != NULL) {
        stats->prompt_tokens = token_count;
        stats->used_accelerated_decode = use_accelerated_decode;
    }
    timing_report("text_generation_setup", setup_started);
    double phase_started = timing_start();
    if (!forward(model, tokens, token_count, logits, error, error_capacity,
                 image_embeddings, image_token_count, &cache)) ok = 0;
    timing_report("text_prefill_forward", phase_started);
    double fallback_decode_forward_ms = 0.0;
    const double decode_started_ms = text_monotonic_ms();
    for (size_t step = 0; ok && step < max_new_tokens; step++) {
        uint32_t best_id = 0;
        if (step > 0 && !use_accelerated_decode) {
            double forward_started = timing_start();
            const int forward_ok = forward(model, &generated[generated_count - 1], 1,
                logits, error, error_capacity, NULL, 0, &cache);
            if (forward_started != 0.0)
                fallback_decode_forward_ms += timing_now_ms() - forward_started;
            if (!forward_ok) {
                ok = 0;
                break;
            }
        }
        if (step == 0 || !use_accelerated_decode) {
            const TensorComputeBackend *backend = tensor_compute_preferred_backend();
            const int argmax_on_gpu = model->fp16 ?
                (backend != NULL && backend->argmax_f16 != NULL &&
                 backend->argmax_f16(logits, vocab, &best_id)) :
                (backend != NULL && backend->argmax_f32 != NULL &&
                 backend->argmax_f32(logits, vocab, &best_id));
            record_compute(argmax_on_gpu);
            if (!argmax_on_gpu) {
                if (model->fp16) {
                    set_error(error, error_capacity, "FP16 argmax failed");
                    ok = 0;
                    break;
                }
                float best_value = -INFINITY;
                for (size_t index = 0; index < vocab; index++) {
                    const float candidate = model->fp16 ?
                        tensor_f16_to_f32(((const uint16_t *)logits)[index]) :
                        ((const float *)logits)[index];
                    if (candidate > best_value) {
                        best_value = candidate;
                        best_id = (uint32_t)index;
                    }
                }
            }
        }
        if (!append_token(&generated, &generated_count, &generated_capacity,
                          best_id)) {
            set_error(error, error_capacity, "out of memory appending generated token");
            ok = 0;
            break;
        }
        if (best_id == model->config.eos_token_id) break;
        if (step == 0 && use_accelerated_decode && generated_count < max_new_tokens) {
            enum { DECODE_CHUNK_TOKENS = 16 };
            size_t remaining_tokens = max_new_tokens - generated_count;
            uint32_t current_token = generated[generated_count - 1];
            phase_started = timing_start();
            int decode_ok = 1;
            while (decode_ok && remaining_tokens > 0) {
                const size_t chunk_tokens = remaining_tokens < DECODE_CHUNK_TOKENS ?
                    remaining_tokens : DECODE_CHUNK_TOKENS;
                uint32_t *decoded_tokens = malloc(chunk_tokens * sizeof(*decoded_tokens));
                decode_ok = decoded_tokens != NULL &&
                    (model->weight_quant != 0 ? tensor_attention_decode_sequence_quant_f16(
                        decoder_layers_quant_f16, layer_count, model->embedding,
                        current_token, chunk_tokens, decoded_tokens,
                        key_cache_f16, value_cache_f16, cache.length, cache.capacity,
                        model->config.text.hidden_size, model->config.text.intermediate_size,
                        model->config.text.attention_heads, model->config.text.key_value_heads,
                        model->config.text.head_dim, (float)model->config.text.rms_norm_eps,
                        model->config.text.rope_theta, model->final_norm, model->lm_head,
                        vocab, model->weight_quant) : model->fp16 ? tensor_attention_decode_sequence_f16(
                        decoder_layers_f16, layer_count, model->embedding,
                        current_token, chunk_tokens, decoded_tokens,
                        key_cache_f16, value_cache_f16, cache.length, cache.capacity,
                        model->config.text.hidden_size, model->config.text.intermediate_size,
                        model->config.text.attention_heads, model->config.text.key_value_heads,
                        model->config.text.head_dim, (float)model->config.text.rms_norm_eps,
                        model->config.text.rope_theta, model->final_norm, model->lm_head, vocab) :
                    tensor_attention_decode_sequence_f32(decoder_layers, layer_count,
                        model->embedding, current_token, chunk_tokens, decoded_tokens,
                        key_cache, value_cache, cache.length, cache.capacity,
                        model->config.text.hidden_size, model->config.text.intermediate_size,
                        model->config.text.attention_heads, model->config.text.key_value_heads,
                        model->config.text.head_dim,
                        (float)model->config.text.rms_norm_eps,
                        model->config.text.rope_theta, model->final_norm,
                        model->lm_head, vocab));
                if (!decode_ok) {
                    free(decoded_tokens);
                    break;
                }
                cache.length += chunk_tokens;
                atomic_fetch_add(&gpu_operation_count,
                                 chunk_tokens * (14 * layer_count + 3));
                atomic_fetch_add(&gpu_attention_count, chunk_tokens * layer_count);
                size_t appended = 0;
                for (; appended < chunk_tokens; appended++) {
                    if (!append_token(&generated, &generated_count,
                                      &generated_capacity, decoded_tokens[appended])) {
                        set_error(error, error_capacity, "out of memory appending generated tokens");
                        decode_ok = 0;
                        break;
                    }
                    if (decoded_tokens[appended] == model->config.eos_token_id) {
                        remaining_tokens = 0;
                        break;
                    }
                }
                if (decode_ok && remaining_tokens > 0) {
                    remaining_tokens -= chunk_tokens;
                    current_token = decoded_tokens[chunk_tokens - 1];
                }
                free(decoded_tokens);
                if (generated_count >= max_new_tokens) remaining_tokens = 0;
            }
            timing_report("text_decode_sequence", phase_started);
            if (!decode_ok) {
                set_error(error, error_capacity, "accelerated decoder sequence failed");
                ok = 0;
                break;
            }
            break;
        }
    }
    const double decode_finished_ms = text_monotonic_ms();
    if (stats != NULL && decode_finished_ms >= decode_started_ms)
        stats->decode_latency_ms = decode_finished_ms - decode_started_ms;
    if (fallback_decode_forward_ms > 0.0)
        timing_report_elapsed("text_decode_fallback_forward", fallback_decode_forward_ms);
    phase_started = timing_start();
    if (ok) ok = tokenizer_decode(tokenizer, generated, generated_count, result,
                                  error, error_capacity);
    if (stats != NULL && ok) stats->generated_tokens = generated_count;
    timing_report("text_output_decode", phase_started);
    allocator_free(generated);
    free(tokens);
    free(logits);
    free(cache.values);
    free(cache.keys);
    free(value_cache);
    free(key_cache);
    free(value_cache_f16);
    free(key_cache_f16);
    free(decoder_layers_quant_f16);
    free(decoder_layers_f16);
    free(decoder_layers);
    tensor_attention_decode_cache_clear();
    return ok;
}

int text_model_generate(TextModel *model, Tokenizer *tokenizer,
                        const char *prompt, size_t max_new_tokens,
                        char **result, char *error, size_t error_capacity) {
    return text_model_generate_with_stats(model, tokenizer, prompt, max_new_tokens,
                                          result, NULL, error, error_capacity);
}

int text_model_generate_with_stats(TextModel *model, Tokenizer *tokenizer,
                                   const char *prompt, size_t max_new_tokens,
                                   char **result, TextGenerationStats *stats,
                                   char *error, size_t error_capacity) {
    if (stats != NULL) *stats = (TextGenerationStats){0};
    if (result != NULL) *result = NULL;
    if (model == NULL || tokenizer == NULL || prompt == NULL || result == NULL || max_new_tokens == 0) {
        set_error(error, error_capacity, "invalid text generation arguments");
        return 0;
    }
    const size_t prompt_length = strlen(prompt);
    if (prompt_length > SIZE_MAX - 80) {
        set_error(error, error_capacity, "prompt is too long");
        return 0;
    }
    char *formatted = malloc(prompt_length + 80);
    if (formatted == NULL) {
        set_error(error, error_capacity, "out of memory formatting prompt");
        return 0;
    }
    snprintf(formatted, prompt_length + 80,
        "<|im_start|>User: %s<end_of_utterance>\n<|im_start|>Assistant:", prompt);
    uint32_t *tokens = NULL;
    size_t token_count = 0;
    int ok = tokenizer_encode(tokenizer, formatted, &tokens, &token_count, error, error_capacity);
    free(formatted);
    if (!ok) return 0;
    return generate_encoded(model, tokenizer, tokens, token_count, max_new_tokens,
        NULL, 0, result, stats, error, error_capacity);
}

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} PromptBuilder;

static int prompt_append(PromptBuilder *builder, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    va_list copy;
    va_copy(copy, arguments);
    const int needed = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (needed < 0 || (size_t)needed > SIZE_MAX - builder->length - 1) {
        va_end(arguments);
        return 0;
    }
    const size_t required = builder->length + (size_t)needed + 1;
    if (required > builder->capacity) {
        size_t capacity = builder->capacity == 0 ? 256 : builder->capacity;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2) { capacity = required; break; }
            capacity *= 2;
        }
        char *grown = realloc(builder->data, capacity);
        if (grown == NULL) { va_end(arguments); return 0; }
        builder->data = grown;
        builder->capacity = capacity;
    }
    vsnprintf(builder->data + builder->length, builder->capacity - builder->length,
              format, arguments);
    va_end(arguments);
    builder->length += (size_t)needed;
    return 1;
}

int text_model_generate_image_with_stats(TextModel *model, Tokenizer *tokenizer,
                              const char *prompt, size_t max_new_tokens,
                              size_t image_rows, size_t image_columns,
                              size_t image_tokens_per_tile,
                              const float *image_embeddings,
                              size_t image_token_count, char **result,
                              TextGenerationStats *stats,
                              char *error, size_t error_capacity) {
    if (stats != NULL) *stats = (TextGenerationStats){0};
    if (result != NULL) *result = NULL;
    if (model == NULL || tokenizer == NULL || prompt == NULL || result == NULL ||
        image_embeddings == NULL || !model->config.has_image_token_id || max_new_tokens == 0 ||
        image_tokens_per_tile == 0 || ((image_rows == 0) != (image_columns == 0))) {
        set_error(error, error_capacity, "invalid image generation arguments");
        return 0;
    }
    PromptBuilder builder = {0};
    double phase_started = timing_start();
    int ok = prompt_append(&builder, "<|im_start|>User:");
    for (size_t row = 0; ok && row < image_rows; row++) {
        for (size_t column = 0; ok && column < image_columns; column++) {
            ok = prompt_append(&builder, "<fake_token_around_image><row_%zu_col_%zu>", row + 1, column + 1);
            for (size_t token = 0; ok && token < image_tokens_per_tile; token++)
                ok = prompt_append(&builder, "<image>");
        }
        if (ok) ok = prompt_append(&builder, "\n");
    }
    if (ok && image_rows > 0) ok = prompt_append(&builder, "\n");
    if (ok) ok = prompt_append(&builder, "<fake_token_around_image><global-img>");
    for (size_t token = 0; ok && token < image_tokens_per_tile; token++)
        ok = prompt_append(&builder, "<image>");
    if (ok) ok = prompt_append(&builder, "<fake_token_around_image>%s<end_of_utterance>\nAssistant:", prompt);
    if (!ok) {
        free(builder.data);
        set_error(error, error_capacity, "out of memory formatting image prompt");
        timing_report("image_prompt_build", phase_started);
        return 0;
    }
    timing_report("image_prompt_build", phase_started);
    uint32_t *tokens = NULL;
    size_t token_count = 0;
    phase_started = timing_start();
    ok = tokenizer_encode(tokenizer, builder.data, &tokens, &token_count, error, error_capacity);
    timing_report("image_prompt_encode", phase_started);
    free(builder.data);
    if (!ok) return 0;
    size_t occurrences = 0;
    for (size_t i = 0; i < token_count; i++)
        if (tokens[i] == model->config.image_token_id) occurrences++;
    if (occurrences != image_token_count) {
        free(tokens);
        set_error(error, error_capacity, "expanded image token count does not match vision features");
        return 0;
    }
    return generate_encoded(model, tokenizer, tokens, token_count, max_new_tokens,
        image_embeddings, image_token_count, result, stats, error, error_capacity);
}

int text_model_generate_image(TextModel *model, Tokenizer *tokenizer,
                              const char *prompt, size_t max_new_tokens,
                              size_t image_rows, size_t image_columns,
                              size_t image_tokens_per_tile,
                              const float *image_embeddings,
                              size_t image_token_count, char **result,
                              char *error, size_t error_capacity) {
    return text_model_generate_image_with_stats(model, tokenizer, prompt,
        max_new_tokens, image_rows, image_columns, image_tokens_per_tile,
        image_embeddings, image_token_count, result, NULL, error, error_capacity);
}
