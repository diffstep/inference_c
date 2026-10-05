#define _POSIX_C_SOURCE 200809L

#include "laya_model.h"

#include "attention.h"
#include "json.h"
#include "tensor_compute_backend.h"
#include "tensor_f16.h"
#include "tensor_store.h"
#include "tensor_buffer.h"
#include "tensor_workspace.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 768u
#define HEADS 12u
#define HEAD_DIM 64u
#define INTERMEDIATE 1152u
#define HEAD_INTERMEDIATE (4u * HIDDEN)
#define ENCODER_LAYERS 22u
#define DECISION_LAYERS 2u
#define EPSILON 1.0e-5f
#define ROPE_THETA 160000.0
#define LOCAL_RADIUS 64u
#define MAX_SEQUENCE 1024u
#define MAX_HEAD_TOKENS 256u

typedef struct {
    uint16_t *attn_norm;
    uint16_t *q, *k, *v, *out;
    uint16_t *mlp_norm;
    uint16_t *mlp_in, *mlp_gate, *mlp_out;
} EncoderLayer;

typedef struct {
    uint16_t *norm1_scale, *norm1_bias;
    uint16_t *q, *k, *v, *qb, *kb, *vb;
    uint16_t *out, *out_bias;
    uint16_t *norm2_scale, *norm2_bias;
    uint16_t *linear1, *linear1_bias, *linear2, *linear2_bias;
} DecisionLayer;

typedef struct {
    const char *name;
    const char *type;
    const char *instructions;
    size_t option_count;
    const char *labels[3];
    const char *descriptions[3];
    int qtype;
} Question;

typedef struct {
    uint32_t *tokens;
    size_t token_count;
    size_t marker_positions[3];
    size_t option_count;
} EncodedQuestion;

struct LayaModel {
    const TensorComputeBackend *backend;
    TensorStore *weights;
    Tokenizer *tokenizer;
    uint16_t *embedding;
    uint16_t *embedding_norm, *embedding_norm_bias;
    uint16_t *final_norm, *final_norm_bias;
    EncoderLayer encoder[ENCODER_LAYERS];
    DecisionLayer head[DECISION_LAYERS];
    uint16_t *type_embedding;
    uint16_t *scorer_norm, *scorer_norm_bias, *scorer_linear, *scorer_linear_bias;
    uint16_t *scorer_output, *scorer_output_bias;
    float temperature[3];
    EncodedQuestion encoded[3];
    uint64_t vocabulary_size;
    size_t max_len;
    size_t head_max_len;
};

static const Question questions[3] = {
    {"department", "choice", "选择最适合负责处理该问题的部门。", 3,
     {"maintenance", "software", "quality"},
     {"设备维修与产线维护", "视觉检测软件与算法", "质量管理与产品判定"}, 0},
    {"urgency", "score", "评估这个问题对当前生产的紧急程度。", 3,
     {"level 0", "level 1", "level 2"},
     {"不紧急", "需要尽快处理", "正在阻塞生产"}, 1},
    {"escalate", "noul", "是否需要立即升级给值班负责人？", 2,
     {"false", "true"}, {"no, the statement does not hold", "yes, the statement holds"}, 2}
};

static const char sample_state[] =
    "客户提交工单：产线视觉检测站从今天上午开始间歇性漏检，重启后暂时恢复。"
    "现场已经尝试清洁镜头，问题仍然出现；当前班次每小时约有两批产品需要人工复检。";

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static char *join_path(const char *directory, const char *name) {
    const size_t a = strlen(directory), b = strlen(name);
    const int slash = a != 0 && directory[a - 1] != '/';
    if (a > SIZE_MAX - b - (size_t)slash - 1) return NULL;
    char *path = malloc(a + b + (size_t)slash + 1);
    if (path == NULL) return NULL;
    memcpy(path, directory, a);
    size_t offset = a;
    if (slash) path[offset++] = '/';
    memcpy(path + offset, name, b + 1);
    return path;
}

static int json_u64(const JsonValue *object, const char *key, uint64_t expected) {
    uint64_t value = 0;
    return json_number_u64(json_object_get(object, key), &value) && value == expected;
}

static int verify_config(LayaModel *model, const char *directory,
                         char *error, size_t error_capacity) {
    char *agent_path = join_path(directory, "rl_agent_config.json");
    char *encoder_dir = join_path(directory, "encoder");
    char *encoder_path = encoder_dir == NULL ? NULL : join_path(encoder_dir, "config.json");
    JsonValue *agent = NULL, *encoder = NULL;
    int ok = agent_path != NULL && encoder_path != NULL &&
        json_parse_file(agent_path, &agent, error, error_capacity) &&
        json_parse_file(encoder_path, &encoder, error, error_capacity);
    if (ok) {
        const JsonValue *name = json_object_get(agent, "encoder");
        const JsonValue *head_layers = json_object_get(agent, "head_layers");
        const JsonValue *max_len = json_object_get(agent, "max_len");
        const JsonValue *head_max_len = json_object_get(agent, "head_max_len");
        uint64_t n = 0, configured_max_len = 0, configured_head_max_len = 0;
        ok = name != NULL && json_string(name) != NULL &&
            strcmp(json_string(name), "jhu-clsp/mmBERT-base") == 0 &&
            json_number_u64(head_layers, &n) && n == DECISION_LAYERS &&
            json_number_u64(max_len, &configured_max_len) && configured_max_len <= MAX_SEQUENCE &&
            json_number_u64(head_max_len, &configured_head_max_len) && configured_head_max_len <= MAX_HEAD_TOKENS &&
            json_u64(encoder, "hidden_size", HIDDEN) &&
            json_u64(encoder, "num_attention_heads", HEADS) &&
            json_u64(encoder, "num_hidden_layers", ENCODER_LAYERS) &&
            json_u64(encoder, "intermediate_size", INTERMEDIATE) &&
            json_u64(encoder, "local_attention", 128) &&
            json_u64(encoder, "global_attn_every_n_layers", 3);
        if (ok) {
            model->max_len = (size_t)configured_max_len;
            model->head_max_len = (size_t)configured_head_max_len;
        }
    }
    if (!ok && error != NULL && error_capacity > 0 && error[0] == '\0')
        set_error(error, error_capacity, "checkpoint config does not match supported Laya multilingual architecture");
    json_free(agent); json_free(encoder);
    free(encoder_path); free(encoder_dir); free(agent_path);
    return ok;
}

static uint16_t *load_f16(LayaModel *model, const char *name, size_t count,
                          char *error, size_t error_capacity) {
    const SafeTensorInfo *info = tensor_store_find(model->weights, name);
    if (info == NULL || info->dtype != TENSOR_DTYPE_F16 ||
        count > SIZE_MAX / sizeof(uint16_t) || info->byte_length != count * sizeof(uint16_t)) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "missing or incompatible F16 tensor: %s", name);
        return NULL;
    }
    uint16_t *data = malloc(count * sizeof(*data));
    if (data == NULL || !tensor_store_read_f16(model->weights, name, data,
                                               count * sizeof(*data), error, error_capacity)) {
        free(data);
        return NULL;
    }
    return data;
}

static int split_matrix(LayaModel *model, const char *name, size_t rows,
                        size_t columns, uint16_t **parts, size_t part_count,
                        char *error, size_t error_capacity) {
    uint16_t *fused = load_f16(model, name, rows * columns, error, error_capacity);
    if (fused == NULL) return 0;
    if (rows % part_count != 0) { free(fused); set_error(error,error_capacity,"fused tensor rows are not evenly divisible"); return 0; }
    const size_t part_bytes = (rows / part_count) * columns * sizeof(uint16_t);
    int ok = 1;
    for (size_t i = 0; i < part_count; ++i) {
        parts[i] = malloc(part_bytes);
        if (parts[i] == NULL) { ok = 0; break; }
        memcpy(parts[i], fused + i * (rows / part_count) * columns, part_bytes);
    }
    free(fused);
    if (!ok) set_error(error, error_capacity, "out of memory splitting fused projection weights");
    return ok;
}

static int split_vector(LayaModel *model, const char *name, size_t count,
                        uint16_t **parts, size_t part_count,
                        char *error, size_t error_capacity) {
    uint16_t *fused = load_f16(model, name, count, error, error_capacity);
    if (fused == NULL) return 0;
    if (count % part_count != 0) { free(fused); set_error(error,error_capacity,"fused bias is not evenly divisible"); return 0; }
    const size_t bytes = (count / part_count) * sizeof(uint16_t);
    int ok = 1;
    for (size_t i = 0; i < part_count; ++i) {
        parts[i] = malloc(bytes);
        if (parts[i] == NULL) { ok = 0; break; }
        memcpy(parts[i], fused + i * count / part_count, bytes);
    }
    free(fused);
    if (!ok) set_error(error, error_capacity, "out of memory splitting fused projection bias");
    return ok;
}

#define LOAD(member, name, count) do { \
    (member) = load_f16(model, (name), (count), error, error_capacity); \
    if ((member) == NULL) return 0; \
} while (0)

static int load_weights(LayaModel *model, char *error, size_t error_capacity) {
    char name[192];
    LOAD(model->embedding, "encoder.embeddings.tok_embeddings.weight", 256000u * HIDDEN);
    LOAD(model->embedding_norm, "encoder.embeddings.norm.weight", HIDDEN);
    model->embedding_norm_bias = calloc(HIDDEN, sizeof(uint16_t));
    LOAD(model->final_norm, "encoder.final_norm.weight", HIDDEN);
    model->final_norm_bias = calloc(HIDDEN, sizeof(uint16_t));
    model->type_embedding = load_f16(model, "type_emb.weight", 3u * HIDDEN, error, error_capacity);
    if (model->embedding_norm_bias == NULL || model->final_norm_bias == NULL || model->type_embedding == NULL) {
        set_error(error, error_capacity, "out of memory loading Laya normalization or type embeddings"); return 0;
    }
    for (size_t i = 0; i < ENCODER_LAYERS; ++i) {
        EncoderLayer *layer = &model->encoder[i];
        if (i != 0) {
            snprintf(name, sizeof(name), "encoder.layers.%zu.attn_norm.weight", i);
            LOAD(layer->attn_norm, name, HIDDEN);
        }
        snprintf(name, sizeof(name), "encoder.layers.%zu.attn.Wqkv.weight", i);
        uint16_t *qkv[3] = {0};
        if (!split_matrix(model, name, 3u * HIDDEN, HIDDEN, qkv, 3, error, error_capacity)) return 0;
        layer->q = qkv[0]; layer->k = qkv[1]; layer->v = qkv[2];
        snprintf(name, sizeof(name), "encoder.layers.%zu.attn.Wo.weight", i);
        LOAD(layer->out, name, HIDDEN * HIDDEN);
        snprintf(name, sizeof(name), "encoder.layers.%zu.mlp_norm.weight", i);
        LOAD(layer->mlp_norm, name, HIDDEN);
        snprintf(name, sizeof(name), "encoder.layers.%zu.mlp.Wi.weight", i);
        uint16_t *wi[2] = {0};
        if (!split_matrix(model, name, 2u * INTERMEDIATE, HIDDEN, wi, 2, error, error_capacity)) return 0;
        layer->mlp_in = wi[0]; layer->mlp_gate = wi[1];
        snprintf(name, sizeof(name), "encoder.layers.%zu.mlp.Wo.weight", i);
        LOAD(layer->mlp_out, name, HIDDEN * INTERMEDIATE);
    }
    for (size_t i = 0; i < DECISION_LAYERS; ++i) {
        DecisionLayer *layer = &model->head[i];
        snprintf(name, sizeof(name), "head.layers.%zu.norm1.weight", i); LOAD(layer->norm1_scale, name, HIDDEN);
        snprintf(name, sizeof(name), "head.layers.%zu.norm1.bias", i); LOAD(layer->norm1_bias, name, HIDDEN);
        snprintf(name, sizeof(name), "head.layers.%zu.self_attn.in_proj_weight", i);
        uint16_t *qkv[3] = {0};
        if (!split_matrix(model, name, 3u * HIDDEN, HIDDEN, qkv, 3, error, error_capacity)) return 0;
        layer->q = qkv[0]; layer->k = qkv[1]; layer->v = qkv[2];
        snprintf(name, sizeof(name), "head.layers.%zu.self_attn.in_proj_bias", i);
        uint16_t *qb[3] = {0};
        if (!split_vector(model, name, 3u * HIDDEN, qb, 3, error, error_capacity)) return 0;
        layer->qb = qb[0]; layer->kb = qb[1]; layer->vb = qb[2];
        snprintf(name, sizeof(name), "head.layers.%zu.self_attn.out_proj.weight", i); LOAD(layer->out, name, HIDDEN * HIDDEN);
        snprintf(name, sizeof(name), "head.layers.%zu.self_attn.out_proj.bias", i); LOAD(layer->out_bias, name, HIDDEN);
        snprintf(name, sizeof(name), "head.layers.%zu.norm2.weight", i); LOAD(layer->norm2_scale, name, HIDDEN);
        snprintf(name, sizeof(name), "head.layers.%zu.norm2.bias", i); LOAD(layer->norm2_bias, name, HIDDEN);
        snprintf(name, sizeof(name), "head.layers.%zu.linear1.weight", i); LOAD(layer->linear1, name, HEAD_INTERMEDIATE * HIDDEN);
        snprintf(name, sizeof(name), "head.layers.%zu.linear1.bias", i); LOAD(layer->linear1_bias, name, HEAD_INTERMEDIATE);
        snprintf(name, sizeof(name), "head.layers.%zu.linear2.weight", i); LOAD(layer->linear2, name, HIDDEN * HEAD_INTERMEDIATE);
        snprintf(name, sizeof(name), "head.layers.%zu.linear2.bias", i); LOAD(layer->linear2_bias, name, HIDDEN);
    }
    LOAD(model->scorer_norm, "scorer.0.weight", HIDDEN);
    LOAD(model->scorer_norm_bias, "scorer.0.bias", HIDDEN);
    LOAD(model->scorer_linear, "scorer.1.weight", HIDDEN * HIDDEN);
    LOAD(model->scorer_linear_bias, "scorer.1.bias", HIDDEN);
    LOAD(model->scorer_output, "scorer.3.weight", HIDDEN);
    LOAD(model->scorer_output_bias, "scorer.3.bias", 1);
    const SafeTensorInfo *temperature = tensor_store_find(model->weights, "temperature");
    if (temperature == NULL || temperature->byte_length != sizeof(model->temperature) ||
        !tensor_store_read_f32(model->weights, "temperature", model->temperature,
                               sizeof(model->temperature), error, error_capacity)) return 0;
    return 1;
}

static int backend_ready(const TensorComputeBackend *backend) {
    return backend != NULL && backend->is_available != NULL && backend->is_available() &&
        backend->name != NULL && backend->linear_f16 != NULL && backend->embedding_f16 != NULL &&
        backend->layer_norm_f16 != NULL && backend->rope_f16 != NULL &&
        backend->gelu_f16 != NULL && backend->gelu_multiply_f16 != NULL && backend->relu_f16 != NULL &&
        backend->residual_add_f16 != NULL && backend->bias_add_f16 != NULL &&
        backend->qkv_f16 != NULL;
}

static int linear(LayaModel *model, const uint16_t *input, const uint16_t *weight,
                  uint16_t *output, size_t rows, size_t in, size_t out,
                  char *error, size_t capacity) {
    if (!model->backend->linear_f16(input, weight, output, rows, in, out)) {
        set_error(error, capacity, "Metal/CUDA FP16 linear failed"); return 0;
    }
    return 1;
}

static int norm(LayaModel *model, const uint16_t *input, const uint16_t *scale,
                const uint16_t *bias, uint16_t *output, size_t rows,
                char *error, size_t capacity) {
    if (!model->backend->layer_norm_f16(input, scale, bias, output, rows, HIDDEN, EPSILON)) {
        set_error(error, capacity, "Metal/CUDA FP16 layer norm failed"); return 0;
    }
    return 1;
}

static int attention(LayaModel *model, const uint16_t *query, const uint16_t *key,
                     const uint16_t *value, size_t sequence, int local,
                     uint16_t *result, TensorWorkspace *workspace,
                     char *error, size_t capacity) {
    const uint64_t qshape[] = {1, HEADS, sequence, HEAD_DIM};
    const uint64_t mshape[] = {sequence, sequence};
    TensorBuffer *q = NULL, *k = NULL, *v = NULL, *o = NULL, *mask = NULL;
    int ok = tensor_buffer_create(TENSOR_DTYPE_F16, qshape, 4, &q, error, capacity) &&
        tensor_buffer_create(TENSOR_DTYPE_F16, qshape, 4, &k, error, capacity) &&
        tensor_buffer_create(TENSOR_DTYPE_F16, qshape, 4, &v, error, capacity) &&
        tensor_buffer_create(TENSOR_DTYPE_F16, qshape, 4, &o, error, capacity);
    if (!ok) goto done;
    if (local) {
        ok = tensor_buffer_create(TENSOR_DTYPE_F16, mshape, 2, &mask, error, capacity);
        if (!ok) goto done;
        uint16_t *m = tensor_buffer_data_mut(mask);
        for (size_t i = 0; i < sequence; ++i)
            for (size_t j = 0; j < sequence; ++j)
                m[i * sequence + j] = tensor_f32_to_f16(
                    (i > j ? i - j : j - i) <= LOCAL_RADIUS ? 0.0f : -65504.0f);
    }
    uint16_t *qd = tensor_buffer_data_mut(q), *kd = tensor_buffer_data_mut(k);
    uint16_t *vd = tensor_buffer_data_mut(v);
    for (size_t t = 0; t < sequence; ++t)
        for (size_t h = 0; h < HEADS; ++h) {
            const size_t src = (t * HEADS + h) * HEAD_DIM;
            const size_t dst = (h * sequence + t) * HEAD_DIM;
            memcpy(qd + dst, query + src, HEAD_DIM * sizeof(uint16_t));
            memcpy(kd + dst, key + src, HEAD_DIM * sizeof(uint16_t));
            memcpy(vd + dst, value + src, HEAD_DIM * sizeof(uint16_t));
        }
    TensorAttentionBackend used;
    ok = tensor_sdpa_f16(q, k, v, mask, 1.0f / sqrtf((float)HEAD_DIM), 0, o,
                         workspace, &used, error, capacity);
    if (ok && strcmp(model->backend->name, "metal") == 0 && used != TENSOR_ATTENTION_BACKEND_METAL) {
        set_error(error, capacity, "Metal attention backend was not used"); ok = 0;
    }
    if (ok && strcmp(model->backend->name, "cuda") == 0 && used != TENSOR_ATTENTION_BACKEND_CUDA) {
        set_error(error, capacity, "CUDA attention backend was not used"); ok = 0;
    }
    if (ok) {
        const uint16_t *od = tensor_buffer_data(o);
        for (size_t t = 0; t < sequence; ++t)
            for (size_t h = 0; h < HEADS; ++h) {
                const size_t src = (h * sequence + t) * HEAD_DIM;
                const size_t dst = (t * HEADS + h) * HEAD_DIM;
                memcpy(result + dst, od + src, HEAD_DIM * sizeof(uint16_t));
            }
    }
done:
    tensor_buffer_free(mask); tensor_buffer_free(o); tensor_buffer_free(v);
    tensor_buffer_free(k); tensor_buffer_free(q);
    return ok;
}

static int append_ids(uint32_t **buffer, size_t *length, size_t *allocated,
                      const uint32_t *ids, size_t count) {
    if (count > SIZE_MAX - *length) return 0;
    const size_t needed = *length + count;
    if (needed > *allocated) {
        size_t next = *allocated == 0 ? 32 : *allocated;
        while (next < needed) {
            if (next > SIZE_MAX / 2) return 0;
            next *= 2;
        }
        uint32_t *grown = realloc(*buffer, next * sizeof(*grown));
        if (grown == NULL) return 0;
        *buffer = grown;
        *allocated = next;
    }
    if (count != 0) memcpy(*buffer + *length, ids, count * sizeof(*ids));
    *length = needed;
    return 1;
}

static int encode_one_token(Tokenizer *tokenizer, const char *spelling,
                            uint32_t *id, char *error, size_t capacity) {
    uint32_t *ids = NULL;
    size_t count = 0;
    if (!tokenizer_encode(tokenizer, spelling, &ids, &count, error, capacity)) return 0;
    const int ok = count == 1;
    if (ok) *id = ids[0];
    else set_error(error, capacity, "checkpoint tokenizer special-token spelling is invalid");
    free(ids);
    return ok;
}

static int encode_question(LayaModel *model, size_t index, char *error, size_t capacity) {
    const Question *q = &questions[index];
    uint32_t cls_id, sep_id, mask_id;
    if (!encode_one_token(model->tokenizer, "<bos>", &cls_id, error, capacity) ||
        !encode_one_token(model->tokenizer, "<eos>", &sep_id, error, capacity) ||
        !encode_one_token(model->tokenizer, "<mask>", &mask_id, error, capacity)) return 0;

    char text[4096];
    int n = snprintf(text, sizeof(text), "%s question: %s", q->type, q->instructions);
    if (n < 0 || (size_t)n >= sizeof(text)) {
        set_error(error, capacity, "Laya question text is too long");
        return 0;
    }
    uint32_t *instruction_ids = NULL;
    size_t instruction_count = 0;
    if (!tokenizer_encode(model->tokenizer, text, &instruction_ids, &instruction_count,
                         error, capacity)) return 0;

    uint32_t *option_ids[3] = {0};
    size_t option_counts[3] = {0};
    size_t options_total = 0;
    int ok = 1;
    for (size_t i = 0; i < q->option_count && ok; ++i) {
        n = snprintf(text, sizeof(text), " %s%s%s", q->labels[i],
                     q->descriptions[i] == NULL ? "" : ": ",
                     q->descriptions[i] == NULL ? "" : q->descriptions[i]);
        if (n < 0 || (size_t)n >= sizeof(text) ||
            !tokenizer_encode(model->tokenizer, text, &option_ids[i], &option_counts[i],
                              error, capacity)) {
            ok = 0;
            break;
        }
        if (option_counts[i] > 48) option_counts[i] = 48;
        options_total += 1 + option_counts[i];
    }
    if (!ok) goto done;

    if (model->head_max_len < 16 || options_total > model->head_max_len - 16) {
        const size_t per_option = model->head_max_len > 16 ?
            (model->head_max_len - 16) / q->option_count : 4;
        options_total = 0;
        for (size_t i = 0; i < q->option_count; ++i) {
            const size_t cap = per_option < 4 ? 4 : per_option;
            if (option_counts[i] + 1 > cap) option_counts[i] = cap - 1;
            options_total += 1 + option_counts[i];
        }
    }
    size_t instruction_budget = model->head_max_len > options_total ?
        model->head_max_len - options_total : 0;
    if (instruction_budget < 8) instruction_budget = 8;
    if (instruction_count > instruction_budget) instruction_count = instruction_budget;

    uint32_t *tokens = NULL;
    size_t token_count = 0, token_capacity = 0;
    EncodedQuestion *encoded = &model->encoded[index];
    ok = append_ids(&tokens, &token_count, &token_capacity, &cls_id, 1) &&
         append_ids(&tokens, &token_count, &token_capacity, instruction_ids, instruction_count) &&
         append_ids(&tokens, &token_count, &token_capacity, &sep_id, 1);
    for (size_t i = 0; i < q->option_count && ok; ++i) {
        encoded->marker_positions[i] = token_count;
        ok = append_ids(&tokens, &token_count, &token_capacity, &mask_id, 1) &&
             append_ids(&tokens, &token_count, &token_capacity, option_ids[i], option_counts[i]);
    }
    ok = ok && append_ids(&tokens, &token_count, &token_capacity, &sep_id, 1);

    uint32_t *state_ids = NULL;
    size_t state_count = 0;
    if (ok) ok = tokenizer_encode(model->tokenizer, sample_state, &state_ids, &state_count,
                                  error, capacity);
    if (ok) {
        const size_t state_room = model->max_len > token_count + 1 ?
            model->max_len - token_count - 1 : 0;
        if (state_count > state_room) state_count = state_room;
        ok = append_ids(&tokens, &token_count, &token_capacity, state_ids, state_count) &&
             append_ids(&tokens, &token_count, &token_capacity, &sep_id, 1);
    }
    free(state_ids);
    if (!ok || token_count == 0 || token_count > model->max_len || token_count > MAX_SEQUENCE) {
        free(tokens);
        if (error != NULL && capacity > 0 && error[0] == '\0')
            set_error(error, capacity, "could not assemble Laya token sequence");
        ok = 0;
        goto done;
    }
    encoded->tokens = tokens;
    encoded->token_count = token_count;
    encoded->option_count = q->option_count;
    for (size_t i = 0; i < q->option_count; ++i) {
        if (encoded->marker_positions[i] >= model->head_max_len) {
            set_error(error, capacity, "Laya question head exceeds configured head_max_len");
            free(encoded->tokens);
            encoded->tokens = NULL;
            ok = 0;
            break;
        }
    }
done:
    free(instruction_ids);
    for (size_t i = 0; i < q->option_count; ++i) free(option_ids[i]);
    return ok;
}

static int encoder_forward(LayaModel *model, const uint32_t *tokens, size_t sequence,
                           uint16_t *state, TensorWorkspace *workspace,
                           char *error, size_t capacity) {
    const size_t count = sequence * HIDDEN;
    uint16_t *normal = malloc(count * sizeof(uint16_t));
    uint16_t *q = malloc(count * sizeof(uint16_t)), *k = malloc(count * sizeof(uint16_t));
    uint16_t *v = malloc(count * sizeof(uint16_t)), *attention_out = malloc(count * sizeof(uint16_t));
    uint16_t *mlp_in = malloc(sequence * INTERMEDIATE * sizeof(uint16_t));
    uint16_t *mlp_gate = malloc(sequence * INTERMEDIATE * sizeof(uint16_t));
    uint16_t *mlp_act = malloc(sequence * INTERMEDIATE * sizeof(uint16_t));
    uint16_t *projected = malloc(count * sizeof(uint16_t));
    uint16_t *bias_zero = calloc(HIDDEN, sizeof(uint16_t));
    int ok = normal && q && k && v && attention_out && mlp_in && mlp_gate && mlp_act && projected && bias_zero;
    if (!ok) { set_error(error,capacity,"out of memory allocating Laya encoder activations"); goto done; }
    if (!model->backend->embedding_f16(model->embedding, tokens, state, sequence,
                                      HIDDEN, (size_t)model->vocabulary_size)) {
        set_error(error,capacity,"GPU embedding lookup failed"); ok = 0; goto done;
    }
    if (!norm(model, state, model->embedding_norm, model->embedding_norm_bias,
              state, sequence, error, capacity)) { ok = 0; goto done; }
    for (size_t li = 0; li < ENCODER_LAYERS; ++li) {
        EncoderLayer *layer = &model->encoder[li];
        if (li == 0) memcpy(normal, state, count * sizeof(uint16_t));
        else if (!norm(model, state, layer->attn_norm, bias_zero, normal, sequence, error, capacity)) { ok = 0; goto done; }
        if (!model->backend->qkv_f16(normal, layer->q, layer->k, layer->v,
                NULL, NULL, NULL, q, k, v, sequence, HIDDEN, HIDDEN) ||
            !model->backend->rope_f16(q, k, sequence, HEADS, HEADS, HEAD_DIM, ROPE_THETA, 0) ||
            !attention(model, q, k, v, sequence, (li % 3) != 0, attention_out,
                       workspace, error, capacity) ||
            !linear(model, attention_out, layer->out, projected, sequence, HIDDEN, HIDDEN, error, capacity) ||
            !model->backend->residual_add_f16(state, projected, state, count)) {
            if (error != NULL && capacity > 0 && error[0] == '\0') set_error(error,capacity,"Laya encoder attention block failed");
            ok = 0; goto done;
        }
        if (!norm(model, state, layer->mlp_norm, bias_zero, normal, sequence, error, capacity) ||
            !linear(model, normal, layer->mlp_in, mlp_in, sequence, HIDDEN, INTERMEDIATE, error, capacity) ||
            !linear(model, normal, layer->mlp_gate, mlp_gate, sequence, HIDDEN, INTERMEDIATE, error, capacity) ||
            !model->backend->gelu_multiply_f16(mlp_in, mlp_gate, mlp_act, sequence * INTERMEDIATE) ||
            !linear(model, mlp_act, layer->mlp_out, projected, sequence, INTERMEDIATE, HIDDEN, error, capacity) ||
            !model->backend->residual_add_f16(state, projected, state, count)) {
            if (error != NULL && capacity > 0 && error[0] == '\0') set_error(error,capacity,"Laya encoder MLP block failed");
            ok = 0; goto done;
        }
    }
    ok = norm(model, state, model->final_norm, model->final_norm_bias, state, sequence, error, capacity);
done:
    free(bias_zero); free(projected); free(mlp_act); free(mlp_gate); free(mlp_in);
    free(attention_out); free(v); free(k); free(q); free(normal);
    return ok;
}

static int head_forward(LayaModel *model, uint16_t *state, size_t sequence,
                        size_t qtype, TensorWorkspace *workspace,
                        char *error, size_t capacity) {
    const size_t count = sequence * HIDDEN;
    uint16_t *normal = malloc(count * sizeof(uint16_t));
    uint16_t *q = malloc(count * sizeof(uint16_t)), *k = malloc(count * sizeof(uint16_t));
    uint16_t *v = malloc(count * sizeof(uint16_t)), *attn = malloc(count * sizeof(uint16_t));
    uint16_t *projected = malloc(count * sizeof(uint16_t));
    uint16_t *hidden = malloc(sequence * HEAD_INTERMEDIATE * sizeof(uint16_t));
    uint16_t *activated = malloc(sequence * HEAD_INTERMEDIATE * sizeof(uint16_t));
    int ok = normal && q && k && v && attn && projected && hidden && activated;
    if (!ok) { set_error(error,capacity,"out of memory allocating Laya decision-head activations"); goto done; }
    const uint16_t *type_vector = model->type_embedding + qtype * HIDDEN;
    if (!model->backend->bias_add_f16(state, type_vector, state, sequence, HIDDEN)) {
        set_error(error,capacity,"GPU type embedding add failed"); ok = 0; goto done;
    }
    for (size_t li = 0; li < DECISION_LAYERS; ++li) {
        DecisionLayer *layer = &model->head[li];
        if (!norm(model, state, layer->norm1_scale, layer->norm1_bias, normal, sequence, error, capacity) ||
            !model->backend->qkv_f16(normal, layer->q, layer->k, layer->v,
                layer->qb, layer->kb, layer->vb, q, k, v, sequence, HIDDEN, HIDDEN) ||
            !attention(model, q, k, v, sequence, 0, attn, workspace, error, capacity) ||
            !linear(model, attn, layer->out, projected, sequence, HIDDEN, HIDDEN, error, capacity) ||
            !model->backend->bias_add_f16(projected, layer->out_bias, projected, sequence, HIDDEN) ||
            !model->backend->residual_add_f16(state, projected, state, count) ||
            !norm(model, state, layer->norm2_scale, layer->norm2_bias, normal, sequence, error, capacity) ||
            !linear(model, normal, layer->linear1, hidden, sequence, HIDDEN, HEAD_INTERMEDIATE, error, capacity) ||
            !model->backend->bias_add_f16(hidden, layer->linear1_bias, hidden, sequence, HEAD_INTERMEDIATE) ||
            !model->backend->relu_f16(hidden, activated, sequence * HEAD_INTERMEDIATE) ||
            !linear(model, activated, layer->linear2, projected, sequence, HEAD_INTERMEDIATE, HIDDEN, error, capacity) ||
            !model->backend->bias_add_f16(projected, layer->linear2_bias, projected, sequence, HIDDEN) ||
            !model->backend->residual_add_f16(state, projected, state, count)) {
            if (error != NULL && capacity > 0 && error[0] == '\0') set_error(error,capacity,"Laya decision transformer block failed");
            ok = 0; goto done;
        }
    }
done:
    free(activated); free(hidden); free(projected); free(attn); free(v); free(k); free(q); free(normal);
    return ok;
}

static int run_question(LayaModel *model, size_t qi, float probabilities[3],
                        size_t *input_tokens, char *error, size_t capacity) {
    const EncodedQuestion *encoded = &model->encoded[qi];
    const size_t sequence = encoded->token_count;
    const size_t count = sequence * HIDDEN;
    uint16_t *state = malloc(count * sizeof(uint16_t));
    uint16_t *markers = malloc(encoded->option_count * HIDDEN * sizeof(uint16_t));
    uint16_t *normed = malloc(encoded->option_count * HIDDEN * sizeof(uint16_t));
    uint16_t *scored = malloc(encoded->option_count * HIDDEN * sizeof(uint16_t));
    uint16_t *logits = malloc(encoded->option_count * sizeof(uint16_t));
    TensorWorkspace *workspace = NULL;
    int ok = state && markers && normed && scored && logits &&
        tensor_workspace_create(4096, &workspace, error, capacity);
    if (!ok) { set_error(error,capacity,"out of memory preparing Laya input"); goto done; }
    if (!encoder_forward(model, encoded->tokens, sequence, state, workspace, error, capacity)) { ok = 0; goto done; }
    if (!head_forward(model, state, sequence, (size_t)questions[qi].qtype,
                      workspace, error, capacity)) { ok = 0; goto done; }
    for (size_t i = 0; i < encoded->option_count; ++i)
        memcpy(markers + i * HIDDEN, state + encoded->marker_positions[i] * HIDDEN,
               HIDDEN * sizeof(uint16_t));
    if (!norm(model, markers, model->scorer_norm, model->scorer_norm_bias,
              normed, encoded->option_count, error, capacity) ||
        !linear(model, normed, model->scorer_linear, scored, encoded->option_count,
                HIDDEN, HIDDEN, error, capacity) ||
        !model->backend->bias_add_f16(scored, model->scorer_linear_bias, scored,
                                      encoded->option_count, HIDDEN) ||
        !model->backend->gelu_f16(scored, scored, encoded->option_count * HIDDEN) ||
        !linear(model, scored, model->scorer_output, logits, encoded->option_count, HIDDEN, 1, error, capacity) ||
        !model->backend->bias_add_f16(logits, model->scorer_output_bias, logits,
                                      encoded->option_count, 1)) {
        set_error(error,capacity,"Laya scorer failed"); ok = 0; goto done;
    }
    float maximum = -INFINITY, sum = 0.0f;
    for (size_t i = 0; i < encoded->option_count; ++i) {
        const float temperature = model->temperature[questions[qi].qtype] > 0.0f ?
            model->temperature[questions[qi].qtype] : 1.0f;
        const float value = tensor_f16_to_f32(logits[i]) / temperature;
        probabilities[i] = value;
        if (value > maximum) maximum = value;
    }
    for (size_t i = 0; i < encoded->option_count; ++i) {
        probabilities[i] = expf(probabilities[i] - maximum);
        sum += probabilities[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) { set_error(error,capacity,"Laya softmax produced invalid values"); ok = 0; goto done; }
    for (size_t i = 0; i < encoded->option_count; ++i) probabilities[i] /= sum;
    *input_tokens = sequence;
done:
    tensor_workspace_free(workspace); free(logits); free(scored); free(normed); free(markers); free(state);
    return ok;
}

int laya_model_open(const char *directory, LayaModel **result,
                    char *error, size_t error_capacity) {
    if (result != NULL) *result = NULL;
    if (directory == NULL || result == NULL) { set_error(error,error_capacity,"invalid Laya model path"); return 0; }
    LayaModel *model = calloc(1, sizeof(*model));
    if (model == NULL) { set_error(error,error_capacity,"out of memory creating Laya model"); return 0; }
    model->backend = tensor_compute_preferred_backend();
    if (!backend_ready(model->backend)) { set_error(error,error_capacity,"Laya requires an available Metal or CUDA FP16 backend; CPU execution is disabled"); goto fail; }
    if (!verify_config(model, directory, error, error_capacity)) goto fail;
    char *weights_path = join_path(directory, "model.safetensors");
    char *tokenizer_path = join_path(directory, "tokenizer/tokenizer.json");
    if (weights_path == NULL || tokenizer_path == NULL ||
        !tensor_store_open_file(weights_path, &model->weights, error, error_capacity) ||
        !tokenizer_open(tokenizer_path, &model->tokenizer, error, error_capacity)) {
        free(weights_path); free(tokenizer_path); goto fail;
    }
    free(weights_path); free(tokenizer_path);
    const SafeTensorInfo *embedding = tensor_store_find(model->weights, "encoder.embeddings.tok_embeddings.weight");
    if (embedding == NULL || embedding->rank != 2 || embedding->shape[0] != 256000 || embedding->shape[1] != HIDDEN) {
        set_error(error,error_capacity,"multilingual token embedding tensor has unexpected shape"); goto fail;
    }
    model->vocabulary_size = embedding->shape[0];
    if (!load_weights(model, error, error_capacity)) goto fail;
    for (size_t i = 0; i < 3; ++i) if (!encode_question(model, i, error, error_capacity)) goto fail;
    *result = model;
    return 1;
fail:
    laya_model_close(model);
    return 0;
}

void laya_model_close(LayaModel *model) {
    if (model == NULL) return;
#define FREE_FIELD(x) free(model->x)
    FREE_FIELD(embedding); FREE_FIELD(embedding_norm); FREE_FIELD(embedding_norm_bias);
    FREE_FIELD(final_norm); FREE_FIELD(final_norm_bias); FREE_FIELD(type_embedding);
    FREE_FIELD(scorer_norm); FREE_FIELD(scorer_norm_bias); FREE_FIELD(scorer_linear);
    FREE_FIELD(scorer_linear_bias); FREE_FIELD(scorer_output); FREE_FIELD(scorer_output_bias);
#undef FREE_FIELD
    for (size_t i = 0; i < ENCODER_LAYERS; ++i) {
        EncoderLayer *l = &model->encoder[i];
        free(l->attn_norm); free(l->q); free(l->k); free(l->v); free(l->out);
        free(l->mlp_norm); free(l->mlp_in); free(l->mlp_gate); free(l->mlp_out);
    }
    for (size_t i = 0; i < DECISION_LAYERS; ++i) {
        DecisionLayer *l = &model->head[i];
        free(l->norm1_scale); free(l->norm1_bias); free(l->q); free(l->k); free(l->v);
        free(l->qb); free(l->kb); free(l->vb); free(l->out); free(l->out_bias);
        free(l->norm2_scale); free(l->norm2_bias); free(l->linear1); free(l->linear1_bias);
        free(l->linear2); free(l->linear2_bias);
    }
    for (size_t i = 0; i < 3; ++i) free(model->encoded[i].tokens);
    tokenizer_close(model->tokenizer); tensor_store_close(model->weights);
    if (model->backend != NULL && model->backend->release_cache != NULL) model->backend->release_cache();
    free(model);
}

int laya_model_predict(LayaModel *model, LayaPrediction *prediction,
                       char *error, size_t error_capacity) {
    if (model == NULL || prediction == NULL) { set_error(error,error_capacity,"invalid Laya prediction arguments"); return 0; }
    memset(prediction, 0, sizeof(*prediction));
    for (size_t i = 0; i < 3; ++i) {
        LayaDecision *decision = &prediction->decisions[i];
        decision->name = questions[i].name;
        decision->type = questions[i].type;
        decision->option_count = questions[i].option_count;
        decision->labels = questions[i].labels;
        if (!run_question(model, i, decision->probabilities, &decision->input_tokens,
                          error, error_capacity)) return 0;
    }
    return 1;
}

const char *laya_model_backend(const LayaModel *model) {
    return model != NULL && model->backend != NULL ? model->backend->name : "unknown";
}
