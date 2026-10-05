#include "vision_model.h"

#include "attention.h"
#include "smolvlm_config.h"
#include "tensor_buffer.h"
#include "tensor_compute_backend.h"
#include "tensor_store.h"
#include "tensor_workspace.h"
#include "tensor_memory.h"
#include "tensor_f16.h"
#include "tensor_quant.h"
#include "timing.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

typedef struct {
    void *norm1_scale, *norm1_bias, *norm2_scale, *norm2_bias;
    void *q_weight, *q_bias, *k_weight, *k_bias, *v_weight, *v_bias;
    void *out_weight, *out_bias;
    void *fc1_weight, *fc1_bias, *fc2_weight, *fc2_bias;
} VisionLayer;

struct VisionModel {
    SmolVLMConfig config;
    TensorStore *weights;
    TensorStore *quantized_weights;
    size_t output_width;
    void *patch_weight, *patch_bias, *position_embedding;
    void *post_norm_scale, *post_norm_bias, *connector_weight;
    VisionLayer *layers;
    int fp16;
    TensorWeightQuantization weight_quant;
};

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static int multiply_size(size_t left, size_t right, size_t *result) {
    if (right != 0 && left > SIZE_MAX / right) return 0;
    *result = left * right;
    return 1;
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
    return name;
}

static char *make_path(const char *directory, const char *filename) {
    const size_t a = strlen(directory), b = strlen(filename);
    const int slash = a > 0 && directory[a - 1] != '/';
    if (a > SIZE_MAX - b - (size_t)slash - 1) return NULL;
    char *path = malloc(a + b + (size_t)slash + 1);
    if (path == NULL) return NULL;
    memcpy(path, directory, a);
    size_t offset = a;
    if (slash) path[offset++] = '/';
    memcpy(path + offset, filename, b + 1);
    return path;
}

static int load_weight(VisionModel *model, const char *name, size_t count,
                       void **result, char *error, size_t capacity) {
    char alternate[512];
    const char *resolved_name = resolve_weight_name(model->weights, name,
                                                     alternate, sizeof(alternate));
    const SafeTensorInfo *info = tensor_store_find(model->weights, resolved_name);
    const size_t element_size = model->fp16 ? sizeof(uint16_t) : sizeof(float);
    const int quantize=model->fp16 && model->weight_quant!=0 && info && info->rank==2 &&
        strstr(name,"position_embedding")==NULL;
    if (info == NULL || (!model->fp16 && info->dtype != TENSOR_DTYPE_F32 &&
                         info->dtype != TENSOR_DTYPE_F16 && info->dtype != TENSOR_DTYPE_BF16) ||
        (model->fp16 && info->dtype != TENSOR_DTYPE_F16 && info->dtype != TENSOR_DTYPE_F32 &&
         info->dtype != TENSOR_DTYPE_BF16) ||
        count > SIZE_MAX / element_size ||
        count > SIZE_MAX / tensor_dtype_size(info->dtype) ||
        info->byte_length != count * tensor_dtype_size(info->dtype)) {
        if (error != NULL && capacity > 0)
            snprintf(error, capacity, "missing or incompatible vision weight: %s", name);
        return 0;
    }
    if (quantize) {
        if (info->shape[0]==0 || info->shape[1]==0 || info->shape[0]>SIZE_MAX || info->shape[1]>SIZE_MAX ||
            (size_t)info->shape[0]>SIZE_MAX/(size_t)info->shape[1] ||
            (size_t)info->shape[0]*(size_t)info->shape[1]!=count || info->shape[1]%32!=0) {
            if (error && capacity)
                snprintf(error,capacity,"quantized vision weight shape is invalid: %s",name);
            return 0;
        }
        char packed_alternate[512];
        const char *packed_name = model->quantized_weights
            ? resolve_weight_name(model->quantized_weights, name, packed_alternate,
                                  sizeof(packed_alternate)) : name;
        const SafeTensorInfo *packed_info=model->quantized_weights
            ? tensor_store_find(model->quantized_weights,packed_name) : NULL;
        const size_t block_bytes=model->weight_quant==TENSOR_WEIGHT_Q8_0 ? 34 : 18;
        if (packed_info) {
            if (packed_info->dtype!=TENSOR_DTYPE_U8 || packed_info->rank!=3 ||
                packed_info->shape[0]!=info->shape[0] || packed_info->shape[1]!=info->shape[1]/32 ||
                packed_info->shape[2]!=block_bytes || packed_info->byte_length>SIZE_MAX) {
                if (error && capacity)
                    snprintf(error,capacity,"quantized vision sidecar layout is invalid: %s",name);
                return 0;
            }
            void *packed=tensor_aligned_allocate((size_t)packed_info->byte_length);
            if (!packed || !tensor_store_read_raw(model->quantized_weights,packed_name,packed,
                    (size_t)packed_info->byte_length,error,capacity)) { free(packed); return 0; }
            *result=packed; return 1;
        }
        uint16_t *source=tensor_aligned_allocate(count*sizeof(uint16_t)); TensorBuffer *packed=NULL;
        if (!source || !tensor_store_read_f16(model->weights,resolved_name,source,count*sizeof(uint16_t),error,capacity) ||
            !tensor_quantize_linear_weight_f16(source,(size_t)info->shape[0],(size_t)info->shape[1],
                model->weight_quant,&packed,error,capacity)) { free(source);tensor_buffer_free(packed);return 0; }
        const size_t packed_bytes=tensor_buffer_byte_length(packed); void *values=tensor_aligned_allocate(packed_bytes);
        if (!values) { free(source);tensor_buffer_free(packed);set_error(error,capacity,"out of memory storing quantized vision weight");return 0; }
        memcpy(values,tensor_buffer_data(packed),packed_bytes);free(source);tensor_buffer_free(packed);*result=values;return 1;
    }
    void *values = tensor_aligned_allocate(count * element_size);
    if (values == NULL) {
        set_error(error, capacity, "out of memory loading vision weight");
        return 0;
    }
    const int loaded = model->fp16 ? tensor_store_read_f16(model->weights, resolved_name, values,
        count * element_size, error, capacity) : tensor_store_read_f32(model->weights,
        resolved_name, values, count * element_size, error, capacity);
    if (!loaded) {
        free(values);
        return 0;
    }
    *result = values;
    return 1;
}

static int load_layer_weight(VisionModel *model, size_t layer, const char *suffix,
                             size_t count, void **result, char *error, size_t capacity) {
    char name[192];
    const int length = snprintf(name, sizeof(name),
        "model.vision_model.encoder.layers.%zu.%s", layer, suffix);
    return length > 0 && (size_t)length < sizeof(name) &&
        load_weight(model, name, count, result, error, capacity);
}

static void cpu_linear(const float *input, const float *weight, const float *bias,
                       float *output, size_t rows, size_t in_width, size_t out_width) {
#if defined(__APPLE__)
    if (rows <= INT_MAX && in_width <= INT_MAX && out_width <= INT_MAX) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)rows, (int)out_width,
            (int)in_width, 1.0f, input, (int)in_width, weight, (int)in_width,
            0.0f, output, (int)out_width);
    } else {
        for (size_t row = 0; row < rows; row++)
            for (size_t column = 0; column < out_width; column++) {
                float sum = 0.0f;
                for (size_t inner = 0; inner < in_width; inner++)
                    sum += input[row * in_width + inner] * weight[column * in_width + inner];
                output[row * out_width + column] = sum;
            }
    }
#else
    for (size_t row = 0; row < rows; row++)
        for (size_t output_index = 0; output_index < out_width; output_index++) {
            float sum = 0.0f;
            for (size_t input_index = 0; input_index < in_width; input_index++)
                sum += input[row * in_width + input_index] *
                       weight[output_index * in_width + input_index];
            output[row * out_width + output_index] = sum;
        }
#endif
    if (bias != NULL)
        for (size_t row = 0; row < rows; row++)
            for (size_t column = 0; column < out_width; column++)
                output[row * out_width + column] += bias[column];
}

static void linear(const float *input, const float *weight, const float *bias,
                   float *output, size_t rows, size_t in_width, size_t out_width) {
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend != NULL && backend->linear_f32 != NULL &&
        backend->linear_f32(input, weight, output, rows, in_width, out_width) &&
        (bias == NULL || (backend->bias_add_f32 != NULL &&
            backend->bias_add_f32(output, bias, output, rows, out_width)))) return;
    cpu_linear(input, weight, bias, output, rows, in_width, out_width);
}

static void layer_norm(const float *input, const float *scale, const float *bias,
                       float *output, size_t rows, size_t width, float epsilon) {
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend != NULL && backend->layer_norm_f32 != NULL &&
        backend->layer_norm_f32(input, scale, bias, output, rows, width, epsilon)) return;
    for (size_t row = 0; row < rows; row++) {
        const float *source = input + row * width;
        float mean = 0.0f;
        for (size_t column = 0; column < width; column++) mean += source[column];
        mean /= (float)width;
        float variance = 0.0f;
        for (size_t column = 0; column < width; column++) {
            const float centered = source[column] - mean;
            variance += centered * centered;
        }
        const float inverse = 1.0f / sqrtf(variance / (float)width + epsilon);
        for (size_t column = 0; column < width; column++)
            output[row * width + column] = (source[column] - mean) * inverse * scale[column] + bias[column];
    }
}

static void add(const float *left, const float *right, float *output, size_t count) {
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend != NULL && backend->residual_add_f32 != NULL &&
        backend->residual_add_f32(left, right, output, count)) return;
    for (size_t i = 0; i < count; i++) output[i] = left[i] + right[i];
}

static void gelu(float *values, size_t count) {
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend != NULL && backend->gelu_f32 != NULL) {
        float *output = malloc(count * sizeof(float));
        if (output != NULL && backend->gelu_f32(values, output, count)) {
            memcpy(values, output, count * sizeof(float));
            free(output);
            return;
        }
        free(output);
    }
    for (size_t i = 0; i < count; i++) {
        const float value = values[i];
        if (value >= 5.0f) {
            values[i] = value;
            continue;
        }
        if (value <= -5.0f) {
            values[i] = 0.0f;
            continue;
        }
        values[i] = 0.5f * value * (1.0f + tanhf(0.7978845608f *
            (value + 0.044715f * value * value * value)));
    }
}

int vision_model_open(const char *directory, VisionModel **result,
                      char *error, size_t capacity) {
    if (result != NULL) *result = NULL;
    if (directory == NULL || result == NULL) {
        set_error(error, capacity, "invalid vision model arguments");
        return 0;
    }
    VisionModel *model = calloc(1, sizeof(*model));
    const char *requested_dtype = getenv("TENSOR_DTYPE");
    if (requested_dtype != NULL && requested_dtype[0] != '\0' &&
        strcmp(requested_dtype, "fp32") != 0 && strcmp(requested_dtype, "fp16") != 0) {
        free(model);
        set_error(error, capacity, "TENSOR_DTYPE must be fp32 or fp16");
        return 0;
    }
    if (model != NULL) model->fp16 = requested_dtype != NULL && strcmp(requested_dtype, "fp16") == 0;
    const char *requested_quant=getenv("TENSOR_WEIGHT_QUANT");
    if (model) {
        if (!requested_quant || !requested_quant[0] || strcmp(requested_quant,"fp16")==0) model->weight_quant=0;
        else if (strcmp(requested_quant,"q8_0")==0) model->weight_quant=TENSOR_WEIGHT_Q8_0;
        else if (strcmp(requested_quant,"q4_0")==0) model->weight_quant=TENSOR_WEIGHT_Q4_0;
        else { free(model);set_error(error,capacity,"TENSOR_WEIGHT_QUANT must be fp16, q8_0, or q4_0");return 0; }
        if (model->weight_quant && !model->fp16) { free(model);set_error(error,capacity,"quantized vision weights require TENSOR_DTYPE=fp16");return 0; }
    }
    if (model != NULL && model->fp16) {
        const TensorComputeBackend *fp16_backend = tensor_compute_preferred_backend();
        if (fp16_backend == NULL || fp16_backend->name == NULL ||
            fp16_backend->linear_f16 == NULL || fp16_backend->layer_norm_f16 == NULL ||
            fp16_backend->bias_add_f16 == NULL ||
            fp16_backend->residual_add_f16 == NULL ||
            (model->weight_quant==0 && (fp16_backend->mlp_f16 == NULL || fp16_backend->qkv_f16 == NULL)) ||
            (model->weight_quant==TENSOR_WEIGHT_Q8_0 && fp16_backend->linear_q8_0_f16==NULL) ||
            (model->weight_quant==TENSOR_WEIGHT_Q4_0 && fp16_backend->linear_q4_0_f16==NULL) ||
            (model->weight_quant!=0 && fp16_backend->gelu_f16==NULL)) {
            free(model);
            set_error(error, capacity,
                      "TENSOR_DTYPE=fp16 requires a complete FP16 compute backend");
            return 0;
        }
    }
    char *config_path = make_path(directory, "config.json");
    char *generation_path = make_path(directory, "generation_config.json");
    char *weights_path = make_path(directory, "model.safetensors");
    if (model == NULL || config_path == NULL || generation_path == NULL || weights_path == NULL) {
        free(config_path); free(generation_path); free(weights_path); free(model);
        set_error(error, capacity, "out of memory preparing vision model");
        return 0;
    }
    int ok = smolvlm_config_load(config_path, generation_path, &model->config, error, capacity) &&
             tensor_store_open_file(weights_path, &model->weights, error, capacity);
    if (ok && model->weight_quant) {
        char filename[64]; snprintf(filename,sizeof(filename),"model.%s.safetensors",
            model->weight_quant==TENSOR_WEIGHT_Q8_0 ? "q8_0" : "q4_0");
        char *sidecar=make_path(directory,filename); FILE *f=sidecar?fopen(sidecar,"rb"):NULL;
        if (f) { fclose(f); if (!tensor_store_open_file(sidecar,&model->quantized_weights,error,capacity)) ok=0; }
        free(sidecar);
    }
    free(config_path); free(generation_path); free(weights_path);
    const size_t hidden = model->config.vision.hidden_size;
    const size_t intermediate = model->config.vision.intermediate_size;
    const size_t patch = model->config.vision.patch_size;
    const size_t tile = model->config.vision.image_size;
    const size_t scale = model->config.pixel_shuffle_factor;
    size_t patch_width = 0, sequence = 0, layer_matrix = 0, connector_width = 0;
    size_t connector_count = 0, projection_count = 0;
    if (ok && (hidden == 0 || intermediate == 0 || patch == 0 || tile % patch != 0 ||
        scale == 0 || tile / patch % scale != 0 ||
        !multiply_size(3, patch, &patch_width) ||
        !multiply_size(patch_width, patch, &patch_width) ||
        !multiply_size(tile / patch, tile / patch, &sequence) ||
        !multiply_size(hidden, hidden, &layer_matrix) ||
        !multiply_size(hidden, scale, &connector_width) ||
        !multiply_size(connector_width, scale, &connector_width) ||
        !multiply_size(connector_width, model->config.text.hidden_size, &connector_count) ||
        !multiply_size(intermediate, hidden, &projection_count))) {
        set_error(error, capacity, "unsupported vision model dimensions");
        ok = 0;
    }
    model->output_width = model->config.text.hidden_size;
    model->layers = ok ? calloc(model->config.vision.layers, sizeof(*model->layers)) : NULL;
    if (ok && model->layers == NULL) {
        set_error(error, capacity, "out of memory allocating vision layers");
        ok = 0;
    }
    if (ok) ok = load_weight(model, "model.vision_model.embeddings.patch_embedding.weight",
        hidden * patch_width, &model->patch_weight, error, capacity) &&
        load_weight(model, "model.vision_model.embeddings.patch_embedding.bias", hidden,
        &model->patch_bias, error, capacity) &&
        load_weight(model, "model.vision_model.embeddings.position_embedding.weight",
        sequence * hidden, &model->position_embedding, error, capacity) &&
        load_weight(model, "model.vision_model.post_layernorm.weight", hidden,
        &model->post_norm_scale, error, capacity) &&
        load_weight(model, "model.vision_model.post_layernorm.bias", hidden,
        &model->post_norm_bias, error, capacity) &&
        load_weight(model, "model.connector.modality_projection.proj.weight",
        connector_count, &model->connector_weight, error, capacity);
    for (size_t i = 0; ok && i < model->config.vision.layers; i++) {
        VisionLayer *layer = &model->layers[i];
        ok = load_layer_weight(model, i, "layer_norm1.weight", hidden, &layer->norm1_scale, error, capacity) &&
             load_layer_weight(model, i, "layer_norm1.bias", hidden, &layer->norm1_bias, error, capacity) &&
             load_layer_weight(model, i, "layer_norm2.weight", hidden, &layer->norm2_scale, error, capacity) &&
             load_layer_weight(model, i, "layer_norm2.bias", hidden, &layer->norm2_bias, error, capacity) &&
             load_layer_weight(model, i, "self_attn.q_proj.weight", layer_matrix, &layer->q_weight, error, capacity) &&
             load_layer_weight(model, i, "self_attn.q_proj.bias", hidden, &layer->q_bias, error, capacity) &&
             load_layer_weight(model, i, "self_attn.k_proj.weight", layer_matrix, &layer->k_weight, error, capacity) &&
             load_layer_weight(model, i, "self_attn.k_proj.bias", hidden, &layer->k_bias, error, capacity) &&
             load_layer_weight(model, i, "self_attn.v_proj.weight", layer_matrix, &layer->v_weight, error, capacity) &&
             load_layer_weight(model, i, "self_attn.v_proj.bias", hidden, &layer->v_bias, error, capacity) &&
             load_layer_weight(model, i, "self_attn.out_proj.weight", layer_matrix, &layer->out_weight, error, capacity) &&
             load_layer_weight(model, i, "self_attn.out_proj.bias", hidden, &layer->out_bias, error, capacity) &&
             load_layer_weight(model, i, "mlp.fc1.weight", projection_count, &layer->fc1_weight, error, capacity) &&
             load_layer_weight(model, i, "mlp.fc1.bias", intermediate, &layer->fc1_bias, error, capacity) &&
             load_layer_weight(model, i, "mlp.fc2.weight", projection_count, &layer->fc2_weight, error, capacity) &&
             load_layer_weight(model, i, "mlp.fc2.bias", hidden, &layer->fc2_bias, error, capacity);
    }
    if (!ok) {
        vision_model_close(model);
        return 0;
    }
    *result = model;
    return 1;
}

void vision_model_close(VisionModel *model) {
    if (model == NULL) return;
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend != NULL && backend->release_cache != NULL) backend->release_cache();
    const size_t layers = model->config.vision.layers;
    for (size_t i = 0; model->layers != NULL && i < layers; i++) {
        VisionLayer *layer = &model->layers[i];
        free(layer->norm1_scale); free(layer->norm1_bias);
        free(layer->norm2_scale); free(layer->norm2_bias);
        free(layer->q_weight); free(layer->q_bias); free(layer->k_weight); free(layer->k_bias);
        free(layer->v_weight); free(layer->v_bias); free(layer->out_weight); free(layer->out_bias);
        free(layer->fc1_weight); free(layer->fc1_bias); free(layer->fc2_weight); free(layer->fc2_bias);
    }
    free(model->layers);
    free(model->patch_weight); free(model->patch_bias); free(model->position_embedding);
    free(model->post_norm_scale); free(model->post_norm_bias); free(model->connector_weight);
    tensor_store_close(model->weights);
    tensor_store_close(model->quantized_weights);
    smolvlm_config_free(&model->config);
    free(model);
}

size_t vision_model_input_size(const VisionModel *model) {
    return model == NULL ? 0 : model->config.vision.image_size;
}

size_t vision_model_output_tokens(const VisionModel *model) {
    if (model == NULL) return 0;
    const size_t patches = model->config.vision.image_size / model->config.vision.patch_size;
    const size_t scale = model->config.pixel_shuffle_factor;
    return patches / scale * (patches / scale);
}

size_t vision_model_output_width(const VisionModel *model) {
    return model == NULL ? 0 : model->output_width;
}

int vision_model_uses_fused_sequence(const VisionModel *model) {
    if (model == NULL) return 0;
    const size_t hidden = model->config.vision.hidden_size;
    const size_t heads = model->config.vision.attention_heads;
    const size_t grid = model->config.vision.image_size / model->config.vision.patch_size;
    const size_t sequence = grid * grid;
    if (heads == 0 || hidden % heads != 0 || hidden / heads != 64 || sequence % 32 != 0)
        return 0;
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend == NULL) return 0;
    if (model->fp16 && model->weight_quant != 0)
        return backend->vision_sequence_layer_quant_f16 != NULL &&
            backend->vision_sequence_begin_f16 != NULL &&
            backend->vision_sequence_finish_f16 != NULL &&
            backend->vision_sequence_release_f16 != NULL;
    if (model->fp16)
        return backend->vision_sequence_begin_f16 != NULL &&
            backend->vision_sequence_layer_f16 != NULL &&
            backend->vision_sequence_finish_f16 != NULL &&
            backend->vision_sequence_release_f16 != NULL;
    return backend->vision_sequence_begin_f32 != NULL &&
        backend->vision_sequence_layer_f32 != NULL &&
        backend->vision_sequence_finish_f32 != NULL &&
        backend->vision_sequence_release_f32 != NULL;
}

static int vision_f16_linear(const TensorComputeBackend *backend,
                             const uint16_t *input, const void *weight,
                             const void *bias, uint16_t *output,
                             size_t rows, size_t input_width, size_t output_width) {
    return backend->linear_f16 != NULL && backend->linear_f16(input, weight, output,
        rows, input_width, output_width) &&
        (bias == NULL || (backend->bias_add_f16 != NULL &&
         backend->bias_add_f16(output, bias, output, rows, output_width)));
}

static int vision_model_f16_linear(const VisionModel *model,
        const TensorComputeBackend *backend,const uint16_t *input,const void *weight,
        const void *bias,uint16_t *output,size_t rows,size_t input_width,size_t output_width) {
    int ok=0;
    if (model->weight_quant==TENSOR_WEIGHT_Q8_0 && backend->linear_q8_0_f16)
        ok=backend->linear_q8_0_f16(input,weight,output,rows,input_width,output_width);
    else if (model->weight_quant==TENSOR_WEIGHT_Q4_0 && backend->linear_q4_0_f16)
        ok=backend->linear_q4_0_f16(input,weight,output,rows,input_width,output_width);
    else if (model->weight_quant==0 && backend->linear_f16)
        ok=backend->linear_f16(input,weight,output,rows,input_width,output_width);
    return ok && (bias==NULL || (backend->bias_add_f16 &&
        backend->bias_add_f16(output,bias,output,rows,output_width)));
}

static int vision_f16_attention(const uint16_t *q, const uint16_t *k,
                                const uint16_t *v, uint16_t *output,
                                size_t sequence, size_t heads, size_t head_dim,
                                TensorWorkspace *workspace, char *error, size_t capacity) {
    const uint64_t shape[] = {1, heads, sequence, head_dim};
    TensorBuffer *query = NULL, *key = NULL, *value = NULL, *attended = NULL;
    int ok = tensor_buffer_create(TENSOR_DTYPE_F16, shape, 4, &query, error, capacity) &&
        tensor_buffer_create(TENSOR_DTYPE_F16, shape, 4, &key, error, capacity) &&
        tensor_buffer_create(TENSOR_DTYPE_F16, shape, 4, &value, error, capacity) &&
        tensor_buffer_create(TENSOR_DTYPE_F16, shape, 4, &attended, error, capacity);
    if (ok) {
        uint16_t *qd = tensor_buffer_data_mut(query), *kd = tensor_buffer_data_mut(key);
        uint16_t *vd = tensor_buffer_data_mut(value);
        for (size_t token = 0; token < sequence; token++)
            for (size_t head = 0; head < heads; head++)
                for (size_t d = 0; d < head_dim; d++) {
                    const size_t src = token * heads * head_dim + head * head_dim + d;
                    const size_t dst = (head * sequence + token) * head_dim + d;
                    qd[dst] = q[src]; kd[dst] = k[src]; vd[dst] = v[src];
                }
        ok = tensor_sdpa_f16(query, key, value, NULL, 1.0f / sqrtf((float)head_dim),
            0, attended, workspace, NULL, error, capacity);
        if (ok) {
            const uint16_t *ad = tensor_buffer_data(attended);
            for (size_t token = 0; token < sequence; token++)
                for (size_t head = 0; head < heads; head++)
                    for (size_t d = 0; d < head_dim; d++)
                        output[token * heads * head_dim + head * head_dim + d] =
                            ad[(head * sequence + token) * head_dim + d];
        }
    }
    tensor_buffer_free(attended); tensor_buffer_free(value);
    tensor_buffer_free(key); tensor_buffer_free(query);
    return ok;
}

static int vision_encode_tile_fp16(VisionModel *model, const float *pixels,
                                   float **features, char *error, size_t capacity) {
    const TensorComputeBackend *backend = tensor_compute_preferred_backend();
    if (backend == NULL || backend->name == NULL ||
        backend->linear_f16 == NULL || backend->layer_norm_f16 == NULL ||
        backend->bias_add_f16 == NULL || backend->gelu_f16 == NULL ||
        backend->residual_add_f16 == NULL) {
        set_error(error, capacity, "FP16 vision inference requires complete FP16 compute support");
        return 0;
    }
    const size_t hidden = model->config.vision.hidden_size;
    const size_t intermediate = model->config.vision.intermediate_size;
    const size_t patch = model->config.vision.patch_size, tile = model->config.vision.image_size;
    const size_t grid = tile / patch, sequence = grid * grid;
    const size_t heads = model->config.vision.attention_heads, head_dim = hidden / heads;
    const size_t scale = model->config.pixel_shuffle_factor;
    const size_t output_tokens = vision_model_output_tokens(model), output_width = scale * scale * hidden;
    const int fused_layer_path = head_dim == 64 && sequence % 32 == 0 &&
        backend->vision_sequence_begin_f16 != NULL &&
        (model->weight_quant == 0 ? backend->vision_sequence_layer_f16 != NULL :
         backend->vision_sequence_layer_quant_f16 != NULL) &&
        backend->vision_sequence_finish_f16 != NULL &&
        backend->vision_sequence_release_f16 != NULL;
    const size_t patch_width = 3 * patch * patch;
    size_t patch_count, count, connector_count, feature_count;
    if (heads == 0 || !multiply_size(sequence, patch_width, &patch_count) ||
        !multiply_size(sequence, hidden, &count) ||
        !multiply_size(output_tokens, output_width, &connector_count) ||
        !multiply_size(output_tokens, model->output_width, &feature_count) ||
        patch_count > SIZE_MAX / sizeof(uint16_t) || count > SIZE_MAX / sizeof(uint16_t) ||
        connector_count > SIZE_MAX / sizeof(uint16_t) ||
        feature_count > SIZE_MAX / sizeof(uint16_t) || feature_count > SIZE_MAX / sizeof(float)) {
        set_error(error, capacity, "FP16 vision activation size overflows");
        return 0;
    }
    uint16_t *patches = malloc(patch_count * sizeof(uint16_t));
    uint16_t *state = malloc(count * sizeof(uint16_t));
    uint16_t *norm = fused_layer_path ? NULL : malloc(count * sizeof(uint16_t));
    uint16_t *q = fused_layer_path ? NULL : malloc(count * sizeof(uint16_t));
    uint16_t *k = fused_layer_path ? NULL : malloc(count * sizeof(uint16_t));
    uint16_t *v = fused_layer_path ? NULL : malloc(count * sizeof(uint16_t));
    uint16_t *attention = fused_layer_path ? NULL : malloc(count * sizeof(uint16_t));
    uint16_t *projected = fused_layer_path ? NULL : malloc(count * sizeof(uint16_t));
    uint16_t *mlp = fused_layer_path ? NULL : malloc(count * sizeof(uint16_t));
    uint16_t *mlp_hidden = !fused_layer_path && model->weight_quant ?
        malloc(sequence * intermediate * sizeof(uint16_t)) : NULL;
    uint16_t *shuffled = malloc(connector_count * sizeof(uint16_t));
    uint16_t *result = malloc(feature_count * sizeof(uint16_t));
    TensorWorkspace *workspace = NULL;
    double patch_embed_ms = 0.0, norm_ms = 0.0, qkv_ms = 0.0;
    double attention_ms = 0.0, output_ms = 0.0, connector_ms = 0.0;
    double layer_command_buffer_ms = 0.0;
    double vision_gpu_wait_ms = 0.0;
    void *vision_sequence = NULL;
    int ok = patches && state && shuffled && result &&
        (fused_layer_path || (norm && q && k && v && attention && projected && mlp &&
         (!model->weight_quant || mlp_hidden)));
    if (ok && !fused_layer_path)
        ok = tensor_workspace_create(4096, &workspace, error, capacity);
    if (!ok) { set_error(error, capacity, "out of memory allocating FP16 vision activations"); goto done; }
    double phase_started = timing_start();
    for (size_t py = 0; py < grid; py++) for (size_t px = 0; px < grid; px++) {
        const size_t token = py * grid + px;
        for (size_t channel = 0; channel < 3; channel++)
            for (size_t y = 0; y < patch; y++) for (size_t x = 0; x < patch; x++) {
                const size_t src = channel * tile * tile + (py * patch + y) * tile + px * patch + x;
                const size_t dst = token * patch_width + channel * patch * patch + y * patch + x;
                patches[dst] = tensor_f32_to_f16(pixels[src]);
            }
    }
    ok = vision_f16_linear(backend, patches, model->patch_weight, model->patch_bias,
                           state, sequence, patch_width, hidden) &&
         backend->residual_add_f16(state, model->position_embedding, state, count);
    if (phase_started != 0.0) patch_embed_ms += timing_now_ms() - phase_started;
    if (!ok) { set_error(error, capacity, "FP16 patch embedding failed"); goto done; }
    if (fused_layer_path) {
        vision_sequence = backend->vision_sequence_begin_f16(state, sequence, hidden);
        if (vision_sequence == NULL) {
            set_error(error, capacity, "FP16 vision sequence initialization failed");
            ok = 0; goto done;
        }
    }
    for (size_t li = 0; ok && li < model->config.vision.layers; li++) {
        VisionLayer *layer = &model->layers[li];
        if (fused_layer_path) {
            int layer_ok = 0;
            if (model->weight_quant != 0) {
                const TensorVisionLayerQuantF16 layer_weights = {
                    .norm1_scale = layer->norm1_scale, .norm1_bias = layer->norm1_bias,
                    .q_weight = layer->q_weight, .q_bias = layer->q_bias,
                    .k_weight = layer->k_weight, .k_bias = layer->k_bias,
                    .v_weight = layer->v_weight, .v_bias = layer->v_bias,
                    .out_weight = layer->out_weight, .out_bias = layer->out_bias,
                    .norm2_scale = layer->norm2_scale, .norm2_bias = layer->norm2_bias,
                    .fc1_weight = layer->fc1_weight, .fc1_bias = layer->fc1_bias,
                    .fc2_weight = layer->fc2_weight, .fc2_bias = layer->fc2_bias
                };
                phase_started = timing_start();
                layer_ok = backend->vision_sequence_layer_quant_f16(vision_sequence,
                    &layer_weights, model->weight_quant, sequence, hidden, intermediate,
                    (float)model->config.vision.layer_norm_eps);
            } else {
                const TensorVisionLayerF16 layer_weights = {
                .norm1_scale = layer->norm1_scale, .norm1_bias = layer->norm1_bias,
                .q_weight = layer->q_weight, .q_bias = layer->q_bias,
                .k_weight = layer->k_weight, .k_bias = layer->k_bias,
                .v_weight = layer->v_weight, .v_bias = layer->v_bias,
                .out_weight = layer->out_weight, .out_bias = layer->out_bias,
                .norm2_scale = layer->norm2_scale, .norm2_bias = layer->norm2_bias,
                .fc1_weight = layer->fc1_weight, .fc1_bias = layer->fc1_bias,
                .fc2_weight = layer->fc2_weight, .fc2_bias = layer->fc2_bias
                };
                phase_started = timing_start();
                layer_ok = backend->vision_sequence_layer_f16(vision_sequence,
                    &layer_weights, sequence, hidden, intermediate,
                    (float)model->config.vision.layer_norm_eps);
            }
            if (phase_started != 0.0)
                layer_command_buffer_ms += timing_now_ms() - phase_started;
            ok = layer_ok;
            if (!ok) set_error(error, capacity,
                "FP16 fused vision transformer layer failed");
            continue;
        }
        phase_started = timing_start();
        ok = backend->layer_norm_f16(state, layer->norm1_scale, layer->norm1_bias,
                norm, sequence, hidden, (float)model->config.vision.layer_norm_eps);
        if (phase_started != 0.0) norm_ms += timing_now_ms() - phase_started;
        phase_started = timing_start();
        if (model->weight_quant==0)
            ok = ok && backend->qkv_f16(norm, layer->q_weight, layer->k_weight,
                layer->v_weight, layer->q_bias, layer->k_bias, layer->v_bias,
                q, k, v, sequence, hidden, hidden);
        else
            ok = ok && vision_model_f16_linear(model,backend,norm,layer->q_weight,layer->q_bias,q,sequence,hidden,hidden) &&
                vision_model_f16_linear(model,backend,norm,layer->k_weight,layer->k_bias,k,sequence,hidden,hidden) &&
                vision_model_f16_linear(model,backend,norm,layer->v_weight,layer->v_bias,v,sequence,hidden,hidden);
        if (phase_started != 0.0) qkv_ms += timing_now_ms() - phase_started;
        phase_started = timing_start();
        ok = ok && vision_f16_attention(q, k, v, attention, sequence, heads, head_dim, workspace, error, capacity);
        if (phase_started != 0.0) attention_ms += timing_now_ms() - phase_started;
        phase_started = timing_start();
        ok = ok && vision_model_f16_linear(model,backend,attention,layer->out_weight,layer->out_bias,
                               projected, sequence, hidden, hidden) &&
             backend->residual_add_f16(state, projected, state, count) &&
             backend->layer_norm_f16(state, layer->norm2_scale, layer->norm2_bias,
                norm, sequence, hidden, (float)model->config.vision.layer_norm_eps) &&
             (model->weight_quant ?
              (vision_model_f16_linear(model,backend,norm,layer->fc1_weight,layer->fc1_bias,
                   mlp_hidden,sequence,hidden,intermediate) &&
               backend->gelu_f16(mlp_hidden,mlp_hidden,sequence*intermediate) &&
               vision_model_f16_linear(model,backend,mlp_hidden,layer->fc2_weight,layer->fc2_bias,
                   mlp,sequence,intermediate,hidden)) :
              backend->mlp_f16(norm, layer->fc1_weight, layer->fc1_bias,
                              layer->fc2_weight, layer->fc2_bias, mlp, sequence,
                              hidden, intermediate, hidden)) &&
             backend->residual_add_f16(state, mlp, state, count);
        if (phase_started != 0.0) output_ms += timing_now_ms() - phase_started;
        if (!ok && (error == NULL || error[0] == '\0'))
                set_error(error, capacity, "FP16 vision transformer execution failed");
    }
    if (ok && fused_layer_path) {
        phase_started = timing_start();
        ok = backend->vision_sequence_finish_f16(vision_sequence, state);
        if (phase_started != 0.0) vision_gpu_wait_ms += timing_now_ms() - phase_started;
    }
    if (ok) ok = backend->layer_norm_f16(state, model->post_norm_scale, model->post_norm_bias,
        state, sequence, hidden, (float)model->config.vision.layer_norm_eps);
    if (ok) {
        phase_started = timing_start();
        const size_t block = grid / scale;
        for (size_t oy = 0; oy < block; oy++) for (size_t ox = 0; ox < block; ox++) {
            const size_t ot = oy * block + ox;
            for (size_t sy = 0; sy < scale; sy++) for (size_t sx = 0; sx < scale; sx++)
                for (size_t c = 0; c < hidden; c++) {
                    const size_t it = (oy * scale + sy) * grid + ox * scale + sx;
                    shuffled[ot * output_width + (sy * scale + sx) * hidden + c] = state[it * hidden + c];
                }
        }
        ok = vision_model_f16_linear(model,backend,shuffled,model->connector_weight,NULL,
                                 result,output_tokens,output_width,model->output_width);
        if (phase_started != 0.0) connector_ms += timing_now_ms() - phase_started;
    }
    if (ok) {
        float *float_result = malloc(feature_count * sizeof(float));
        if (float_result == NULL) { set_error(error, capacity, "out of memory converting vision features"); ok = 0; }
        else {
            for (size_t i = 0; i < feature_count; i++)
                float_result[i] = tensor_f16_to_f32(result[i]);
            *features = float_result;
        }
    }
    if (patch_embed_ms != 0.0 || attention_ms != 0.0 || layer_command_buffer_ms != 0.0)
        fprintf(stderr, "timing_vision_fp16_stages_ms=patch_embed:%.3f norm:%.3f qkv:%.3f attention:%.3f output_mlp:%.3f layer_submit:%.3f gpu_wait:%.3f connector:%.3f\n",
            patch_embed_ms, norm_ms, qkv_ms, attention_ms, output_ms,
            layer_command_buffer_ms, vision_gpu_wait_ms, connector_ms);
done:
    if (vision_sequence != NULL)
        backend->vision_sequence_release_f16(vision_sequence);
    tensor_workspace_free(workspace);
    free(result); free(shuffled); free(mlp_hidden); free(mlp); free(projected); free(attention);
    free(v); free(k); free(q); free(norm); free(state); free(patches);
    return ok;
}

int vision_model_encode_tile(VisionModel *model, const float *pixels,
                             float **features, char *error, size_t capacity) {
    if (features != NULL) *features = NULL;
    if (model == NULL || pixels == NULL || features == NULL) {
        set_error(error, capacity, "invalid vision tile arguments");
        return 0;
    }
    if (model->fp16) return vision_encode_tile_fp16(model, pixels, features, error, capacity);
    const size_t hidden = model->config.vision.hidden_size;
    const size_t intermediate = model->config.vision.intermediate_size;
    const size_t patch = model->config.vision.patch_size;
    const size_t tile = model->config.vision.image_size;
    const size_t grid = tile / patch;
    const size_t sequence = grid * grid;
    const size_t heads = model->config.vision.attention_heads;
    const size_t head_dim = hidden / heads;
    const size_t scale = model->config.pixel_shuffle_factor;
    const size_t output_tokens = vision_model_output_tokens(model);
    const size_t patch_width = 3 * patch * patch;
    const TensorComputeBackend *vision_backend = tensor_compute_preferred_backend();
    const int fused_layer_path = head_dim == 64 && sequence % 32 == 0 &&
        vision_backend != NULL && vision_backend->vision_sequence_begin_f32 != NULL &&
        vision_backend->vision_sequence_layer_f32 != NULL &&
        vision_backend->vision_sequence_finish_f32 != NULL &&
        vision_backend->vision_sequence_release_f32 != NULL;
    size_t patch_count = 0, hidden_count = 0, intermediate_count = 0;
    size_t connector_input_count = 0, feature_count = 0;
    if (sequence > SIZE_MAX / hidden || sequence > SIZE_MAX / intermediate ||
        sequence > SIZE_MAX / heads / head_dim ||
        !multiply_size(sequence, patch_width, &patch_count) ||
        !multiply_size(sequence, hidden, &hidden_count) ||
        !multiply_size(sequence, intermediate, &intermediate_count) ||
        !multiply_size(output_tokens, scale * scale * hidden, &connector_input_count) ||
        !multiply_size(output_tokens, model->output_width, &feature_count) ||
        patch_count > SIZE_MAX / sizeof(float) || hidden_count > SIZE_MAX / sizeof(float) ||
        intermediate_count > SIZE_MAX / sizeof(float) ||
        connector_input_count > SIZE_MAX / sizeof(float) || feature_count > SIZE_MAX / sizeof(float)) {
        set_error(error, capacity, "vision activation size overflows");
        return 0;
    }
    float *patches = malloc(patch_count * sizeof(float));
    float *hidden_states = malloc(hidden_count * sizeof(float));
    float *normalized = fused_layer_path ? NULL : malloc(hidden_count * sizeof(float));
    float *q = fused_layer_path ? NULL : malloc(hidden_count * sizeof(float));
    float *k = fused_layer_path ? NULL : malloc(hidden_count * sizeof(float));
    float *v = fused_layer_path ? NULL : malloc(hidden_count * sizeof(float));
    float *attention = fused_layer_path ? NULL : malloc(hidden_count * sizeof(float));
    float *projected = fused_layer_path ? NULL : malloc(hidden_count * sizeof(float));
    float *intermediate_values = fused_layer_path ? NULL : malloc(intermediate_count * sizeof(float));
    float *mlp_values = fused_layer_path ? NULL : malloc(hidden_count * sizeof(float));
    float *head_q = fused_layer_path ? NULL : malloc(hidden_count * sizeof(float));
    float *head_k = fused_layer_path ? NULL : malloc(hidden_count * sizeof(float));
    float *head_v = fused_layer_path ? NULL : malloc(hidden_count * sizeof(float));
    float *head_out = fused_layer_path ? NULL : malloc(hidden_count * sizeof(float));
    const size_t output_width = scale * scale * hidden;
    float *shuffled = malloc(connector_input_count * sizeof(float));
    float *result = malloc(feature_count * sizeof(float));
    int ok = patches && hidden_states && shuffled && result &&
        (fused_layer_path || (normalized && q && k && v && attention && projected &&
        intermediate_values && mlp_values && head_q && head_k && head_v && head_out));
    if (!ok) set_error(error, capacity, "out of memory allocating vision activations");
    double setup_ms = 0.0, patch_embed_ms = 0.0, norm_ms = 0.0, qkv_ms = 0.0;
    double repack_ms = 0.0, attention_ms = 0.0, output_ms = 0.0, mlp_ms = 0.0;
    double mlp_norm_ms = 0.0, mlp_fc1_ms = 0.0, mlp_gelu_ms = 0.0;
    double mlp_fc2_ms = 0.0, mlp_residual_ms = 0.0, mlp_fused_ms = 0.0;
    double connector_ms = 0.0, layer_command_buffer_ms = 0.0;
    void *vision_sequence = NULL;
    if (ok) {
        double phase_started = timing_start();
        for (size_t py = 0; py < grid; py++) for (size_t px = 0; px < grid; px++) {
            const size_t token = py * grid + px;
            for (size_t channel = 0; channel < 3; channel++)
                for (size_t y = 0; y < patch; y++)
                    for (size_t x = 0; x < patch; x++) {
                        const size_t src = channel * tile * tile + (py * patch + y) * tile + px * patch + x;
                        const size_t dst = token * patch_width + channel * patch * patch + y * patch + x;
                        patches[dst] = pixels[src];
                    }
        }
        linear(patches, model->patch_weight, NULL, hidden_states,
               sequence, patch_width, hidden);
        add(hidden_states, model->position_embedding, hidden_states, sequence * hidden);
        if (phase_started != 0.0) patch_embed_ms += timing_now_ms() - phase_started;
        if (fused_layer_path) {
            vision_sequence = vision_backend->vision_sequence_begin_f32(
                hidden_states, sequence, hidden);
            if (vision_sequence == NULL) {
                set_error(error, capacity, "vision sequence initialization failed");
                ok = 0;
            }
        }
        phase_started = timing_start();
        TensorWorkspace *workspace = NULL;
        TensorBuffer *query = NULL, *key = NULL, *value = NULL, *attended = NULL;
        const uint64_t q_shape[] = {1, heads, sequence, head_dim};
        ok = ok && (fused_layer_path || (tensor_workspace_create(4096, &workspace, error, capacity) &&
             tensor_buffer_create(TENSOR_DTYPE_F32, q_shape, 4, &query, error, capacity) &&
             tensor_buffer_create(TENSOR_DTYPE_F32, q_shape, 4, &key, error, capacity) &&
             tensor_buffer_create(TENSOR_DTYPE_F32, q_shape, 4, &value, error, capacity) &&
             tensor_buffer_create(TENSOR_DTYPE_F32, q_shape, 4, &attended, error, capacity)));
        if (phase_started != 0.0) setup_ms += timing_now_ms() - phase_started;
        for (size_t layer_index = 0; ok && layer_index < model->config.vision.layers; layer_index++) {
            VisionLayer *layer = &model->layers[layer_index];
            if (fused_layer_path) {
                const TensorVisionLayerF32 layer_weights = {
                    .norm1_scale = layer->norm1_scale, .norm1_bias = layer->norm1_bias,
                    .q_weight = layer->q_weight, .q_bias = layer->q_bias,
                    .k_weight = layer->k_weight, .k_bias = layer->k_bias,
                    .v_weight = layer->v_weight, .v_bias = layer->v_bias,
                    .out_weight = layer->out_weight, .out_bias = layer->out_bias,
                    .norm2_scale = layer->norm2_scale, .norm2_bias = layer->norm2_bias,
                    .fc1_weight = layer->fc1_weight, .fc1_bias = layer->fc1_bias,
                    .fc2_weight = layer->fc2_weight, .fc2_bias = layer->fc2_bias
                };
                phase_started = timing_start();
                ok = vision_backend->vision_sequence_layer_f32(vision_sequence, &layer_weights,
                    sequence, hidden, intermediate,
                    (float)model->config.vision.layer_norm_eps);
                if (phase_started != 0.0)
                    layer_command_buffer_ms += timing_now_ms() - phase_started;
                if (!ok) set_error(error, capacity,
                    "fused FP32 vision transformer layer failed");
                continue;
            }
            phase_started = timing_start();
            layer_norm(hidden_states, layer->norm1_scale, layer->norm1_bias,
                       normalized, sequence, hidden, (float)model->config.vision.layer_norm_eps);
            if (phase_started != 0.0) norm_ms += timing_now_ms() - phase_started;
            phase_started = timing_start();
            linear(normalized, layer->q_weight, layer->q_bias, q, sequence, hidden, hidden);
            linear(normalized, layer->k_weight, layer->k_bias, k, sequence, hidden, hidden);
            linear(normalized, layer->v_weight, layer->v_bias, v, sequence, hidden, hidden);
            if (phase_started != 0.0) qkv_ms += timing_now_ms() - phase_started;
            phase_started = timing_start();
            for (size_t token = 0; token < sequence; token++) for (size_t head = 0; head < heads; head++)
                for (size_t dim = 0; dim < head_dim; dim++) {
                    const size_t src = token * hidden + head * head_dim + dim;
                    const size_t dst = (head * sequence + token) * head_dim + dim;
                    head_q[dst] = q[src]; head_k[dst] = k[src]; head_v[dst] = v[src];
                }
            memcpy(tensor_buffer_data_mut(query), head_q, sequence * hidden * sizeof(float));
            memcpy(tensor_buffer_data_mut(key), head_k, sequence * hidden * sizeof(float));
            memcpy(tensor_buffer_data_mut(value), head_v, sequence * hidden * sizeof(float));
            if (phase_started != 0.0) repack_ms += timing_now_ms() - phase_started;
            phase_started = timing_start();
            ok = tensor_sdpa_f32(query, key, value, NULL, 1.0f / sqrtf((float)head_dim),
                                 0, attended, workspace, NULL, error, capacity);
            if (phase_started != 0.0) attention_ms += timing_now_ms() - phase_started;
            if (!ok) break;
            phase_started = timing_start();
            memcpy(head_out, tensor_buffer_data(attended), sequence * hidden * sizeof(float));
            for (size_t token = 0; token < sequence; token++) for (size_t head = 0; head < heads; head++)
                for (size_t dim = 0; dim < head_dim; dim++)
                    attention[token * hidden + head * head_dim + dim] =
                        head_out[(head * sequence + token) * head_dim + dim];
            linear(attention, layer->out_weight, layer->out_bias, projected, sequence, hidden, hidden);
            add(hidden_states, projected, hidden_states, sequence * hidden);
            if (phase_started != 0.0) output_ms += timing_now_ms() - phase_started;
            phase_started = timing_start();
            double mlp_phase_started = timing_start();
            layer_norm(hidden_states, layer->norm2_scale, layer->norm2_bias,
                       normalized, sequence, hidden, (float)model->config.vision.layer_norm_eps);
            if (mlp_phase_started != 0.0) mlp_norm_ms += timing_now_ms() - mlp_phase_started;
            const TensorComputeBackend *mlp_backend = tensor_compute_preferred_backend();
            mlp_phase_started = timing_start();
            const int fused_mlp = mlp_backend != NULL && mlp_backend->mlp_f32 != NULL &&
                mlp_backend->mlp_f32(normalized, layer->fc1_weight, layer->fc1_bias,
                    layer->fc2_weight, layer->fc2_bias, mlp_values, sequence,
                    hidden, intermediate, hidden);
            if (fused_mlp) {
                if (mlp_phase_started != 0.0) mlp_fused_ms += timing_now_ms() - mlp_phase_started;
            } else {
                mlp_phase_started = timing_start();
                linear(normalized, layer->fc1_weight, layer->fc1_bias,
                       intermediate_values, sequence, hidden, intermediate);
                if (mlp_phase_started != 0.0) mlp_fc1_ms += timing_now_ms() - mlp_phase_started;
                mlp_phase_started = timing_start();
                gelu(intermediate_values, sequence * intermediate);
                if (mlp_phase_started != 0.0) mlp_gelu_ms += timing_now_ms() - mlp_phase_started;
                mlp_phase_started = timing_start();
                linear(intermediate_values, layer->fc2_weight, layer->fc2_bias,
                       mlp_values, sequence, intermediate, hidden);
                if (mlp_phase_started != 0.0) mlp_fc2_ms += timing_now_ms() - mlp_phase_started;
            }
            mlp_phase_started = timing_start();
            add(hidden_states, mlp_values, hidden_states, sequence * hidden);
            if (mlp_phase_started != 0.0) mlp_residual_ms += timing_now_ms() - mlp_phase_started;
            if (phase_started != 0.0) mlp_ms += timing_now_ms() - phase_started;
        }
        tensor_buffer_free(query); tensor_buffer_free(key); tensor_buffer_free(value);
        tensor_buffer_free(attended); tensor_workspace_free(workspace);
    }
    if (ok && fused_layer_path)
        ok = vision_backend->vision_sequence_finish_f32(vision_sequence, hidden_states);
    if (ok) {
        double phase_started = timing_start();
        layer_norm(hidden_states, model->post_norm_scale, model->post_norm_bias,
                   hidden_states, sequence, hidden, (float)model->config.vision.layer_norm_eps);
        const size_t block = grid / scale;
        for (size_t out_y = 0; out_y < block; out_y++) for (size_t out_x = 0; out_x < block; out_x++) {
            const size_t out_token = out_y * block + out_x;
            for (size_t sub_y = 0; sub_y < scale; sub_y++) for (size_t sub_x = 0; sub_x < scale; sub_x++)
                for (size_t channel = 0; channel < hidden; channel++) {
                    const size_t in_token = (out_y * scale + sub_y) * grid + out_x * scale + sub_x;
                    const size_t shuffled_channel = (sub_y * scale + sub_x) * hidden + channel;
                    shuffled[out_token * output_width + shuffled_channel] =
                        hidden_states[in_token * hidden + channel];
                }
        }
        const TensorComputeBackend *backend = tensor_compute_preferred_backend();
        if (backend != NULL && backend->linear_f32 != NULL &&
            backend->linear_f32(shuffled, model->connector_weight, result,
                                output_tokens, output_width, model->output_width)) {
        } else {
            cpu_linear(shuffled, model->connector_weight, NULL, result,
                       output_tokens, output_width, model->output_width);
        }
        *features = result;
        if (phase_started != 0.0) connector_ms += timing_now_ms() - phase_started;
    } else free(result);
    if (setup_ms != 0.0 || patch_embed_ms != 0.0)
        fprintf(stderr, "timing_vision_tile_stages_ms=setup:%.3f patch_embed:%.3f norm:%.3f qkv:%.3f repack:%.3f attention:%.3f output:%.3f mlp:%.3f mlp_norm:%.3f mlp_fused:%.3f mlp_fc1:%.3f mlp_gelu:%.3f mlp_fc2:%.3f mlp_residual:%.3f layer_command_buffer:%.3f connector:%.3f\n",
                setup_ms, patch_embed_ms, norm_ms, qkv_ms, repack_ms,
                attention_ms, output_ms, mlp_ms, mlp_norm_ms, mlp_fused_ms, mlp_fc1_ms,
                mlp_gelu_ms, mlp_fc2_ms, mlp_residual_ms, layer_command_buffer_ms, connector_ms);
    if (vision_sequence != NULL)
        vision_backend->vision_sequence_release_f32(vision_sequence);
    free(patches); free(hidden_states); free(normalized); free(q); free(k); free(v);
    free(attention); free(projected); free(intermediate_values); free(mlp_values);
    free(head_q); free(head_k); free(head_v); free(head_out); free(shuffled);
    return ok;
}
