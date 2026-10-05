#include "attention.h"
#include "attention_backend.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const TensorAttentionBackendOps *selected_decode_backend(void) {
    const char *requested = getenv("TENSOR_ATTENTION_BACKEND");
    if (requested == NULL || requested[0] == '\0') requested = getenv("TENSOR_BACKEND");
    if (requested == NULL || requested[0] == '\0' || strcmp(requested, "auto") == 0) {
        const TensorAttentionBackendOps *cuda = tensor_attention_cuda_backend();
        if (cuda != NULL && cuda->is_available != NULL && cuda->is_available()) return cuda;
        const TensorAttentionBackendOps *metal = tensor_attention_metal_backend();
        return metal != NULL && metal->is_available != NULL && metal->is_available() ? metal : NULL;
    }
    if (strcmp(requested, "qnn") == 0) {
#if defined(INFERENCE_SDK_QNN_HTP_ENABLED)
        const TensorAttentionBackendOps *qnn = tensor_attention_qnn_htp_backend();
        return qnn != NULL && qnn->is_available != NULL && qnn->is_available() ? qnn : NULL;
#else
        return NULL;
#endif
    }
    if (strcmp(requested, "cuda") == 0) {
        const TensorAttentionBackendOps *cuda = tensor_attention_cuda_backend();
        return cuda != NULL && cuda->is_available != NULL && cuda->is_available() ? cuda : NULL;
    }
    if (strcmp(requested, "metal") == 0) {
        const TensorAttentionBackendOps *metal = tensor_attention_metal_backend();
        return metal != NULL && metal->is_available != NULL && metal->is_available() ? metal : NULL;
    }
    return NULL;
}

int tensor_attention_decode_f32(const float *query, const float *key,
                                const float *value, const float *output_weight,
                                const float *residual, float *output,
                                size_t query_heads, size_t kv_heads,
                                size_t kv_length, size_t kv_capacity,
                                size_t head_dim) {
    const TensorAttentionBackendOps *backend = selected_decode_backend();
    return backend != NULL && backend->decode != NULL &&
        backend->decode(query, key, value, output_weight, residual, output,
                        query_heads, kv_heads, kv_length, kv_capacity, head_dim);
}

int tensor_attention_decode_sequence_available(void) {
    const TensorAttentionBackendOps *backend = selected_decode_backend();
    return backend != NULL && backend->decode_sequence_available != NULL &&
        backend->decode_sequence_available();
}

int tensor_attention_decode_sequence_f32(
    const TensorAttentionDecoderLayer *layers, size_t layer_count,
    const float *embedding, uint32_t initial_token,
    size_t token_count, uint32_t *generated_tokens,
    const float *const *key_cache, const float *const *value_cache,
    size_t past_length, size_t cache_capacity, size_t hidden_size,
    size_t intermediate_size, size_t query_heads, size_t kv_heads,
    size_t head_dim, float rms_norm_epsilon, double rope_theta,
    const float *final_norm, const float *lm_head, size_t vocabulary_size) {
    const TensorAttentionBackendOps *backend = selected_decode_backend();
    return backend != NULL && backend->decode_sequence != NULL &&
        backend->decode_sequence(layers, layer_count, embedding, initial_token,
            token_count, generated_tokens,
            key_cache, value_cache, past_length, cache_capacity, hidden_size,
            intermediate_size, query_heads, kv_heads, head_dim,
            rms_norm_epsilon, rope_theta, final_norm, lm_head, vocabulary_size);
}

int tensor_attention_decode_sequence_f16_available(void) {
    const TensorAttentionBackendOps *backend = selected_decode_backend();
    return backend != NULL && backend->decode_sequence_f16_available != NULL &&
        backend->decode_sequence_f16_available();
}

int tensor_attention_decode_sequence_f16(
    const TensorAttentionDecoderLayerF16 *layers, size_t layer_count,
    const uint16_t *embedding, uint32_t initial_token, size_t token_count,
    uint32_t *generated_tokens, const uint16_t *const *key_cache,
    const uint16_t *const *value_cache, size_t past_length,
    size_t cache_capacity, size_t hidden_size, size_t intermediate_size,
    size_t query_heads, size_t kv_heads, size_t head_dim,
    float rms_norm_epsilon, double rope_theta, const uint16_t *final_norm,
    const uint16_t *lm_head, size_t vocabulary_size) {
    const TensorAttentionBackendOps *backend = selected_decode_backend();
    return backend != NULL && backend->decode_sequence_f16 != NULL &&
        backend->decode_sequence_f16(layers, layer_count, embedding, initial_token,
            token_count, generated_tokens, key_cache, value_cache, past_length,
            cache_capacity, hidden_size, intermediate_size, query_heads, kv_heads,
            head_dim, rms_norm_epsilon, rope_theta, final_norm, lm_head,
            vocabulary_size);
}

int tensor_attention_decode_sequence_quant_f16_available(
    TensorWeightQuantization format) {
    const TensorAttentionBackendOps *backend = selected_decode_backend();
    return backend != NULL && backend->decode_sequence_quant_f16_available != NULL &&
        backend->decode_sequence_quant_f16_available(format);
}

int tensor_attention_decode_sequence_quant_f16(
    const TensorAttentionDecoderLayerQuantF16 *layers, size_t layer_count,
    const uint16_t *embedding, uint32_t initial_token, size_t token_count,
    uint32_t *generated_tokens, const uint16_t *const *key_cache,
    const uint16_t *const *value_cache, size_t past_length,
    size_t cache_capacity, size_t hidden_size, size_t intermediate_size,
    size_t query_heads, size_t kv_heads, size_t head_dim,
    float rms_norm_epsilon, double rope_theta, const uint16_t *final_norm,
    const uint8_t *lm_head, size_t vocabulary_size,
    TensorWeightQuantization format) {
    const TensorAttentionBackendOps *backend = selected_decode_backend();
    return backend != NULL && backend->decode_sequence_quant_f16 != NULL &&
        backend->decode_sequence_quant_f16(layers, layer_count, embedding,
            initial_token, token_count, generated_tokens, key_cache, value_cache,
            past_length, cache_capacity, hidden_size, intermediate_size,
            query_heads, kv_heads, head_dim, rms_norm_epsilon, rope_theta,
            final_norm, lm_head, vocabulary_size, format);
}

int tensor_attention_decode_sequence_paged(
    const TensorPagedKVDecodeRequest *request) {
    if (request == NULL || request->cache == NULL || request->sequence == 0 ||
        request->layers == NULL || request->layer_count == 0 ||
        request->embedding == NULL || request->generated_tokens == NULL ||
        request->key_cache == NULL || request->value_cache == NULL ||
        request->final_norm == NULL || request->lm_head == NULL ||
        request->token_count == 0 || request->hidden_size == 0 ||
        request->intermediate_size == 0 || request->query_heads == 0 ||
        request->kv_heads == 0 || request->head_dim == 0 ||
        request->past_length > SIZE_MAX - request->token_count ||
        (request->storage != TENSOR_PAGED_KV_F32 &&
         request->storage != TENSOR_PAGED_KV_F16 &&
         request->storage != TENSOR_PAGED_KV_QUANT_F16) ||
        (request->storage == TENSOR_PAGED_KV_QUANT_F16 &&
         request->weight_quantization != TENSOR_WEIGHT_Q4_0 &&
         request->weight_quantization != TENSOR_WEIGHT_Q8_0)) return 0;
    if (!tensor_paged_kv_sequence_begin_decode(request->cache, request->sequence))
        return 0;
    size_t current_length = 0;
    if (!tensor_paged_kv_sequence_length(request->cache, request->sequence,
                                         &current_length)) {
        tensor_paged_kv_sequence_end_decode(request->cache, request->sequence);
        return 0;
    }
    if (current_length != 0 && current_length != request->past_length) {
        tensor_paged_kv_sequence_end_decode(request->cache, request->sequence);
        return 0;
    }
    const size_t target_length = request->past_length + request->token_count;
    if (!tensor_paged_kv_sequence_reserve(request->cache, request->sequence,
                                          target_length)) {
        tensor_paged_kv_sequence_end_decode(request->cache, request->sequence);
        return 0;
    }
    const TensorAttentionBackendOps *backend = selected_decode_backend();
    if (backend == NULL || backend->decode_sequence_paged == NULL ||
        !backend->decode_sequence_paged(request)) {
        (void)tensor_paged_kv_sequence_set_length(request->cache,
            request->sequence, current_length);
        tensor_paged_kv_sequence_end_decode(request->cache, request->sequence);
        return 0;
    }
    if (!tensor_paged_kv_sequence_set_length(request->cache,
            request->sequence, target_length)) {
        (void)tensor_paged_kv_sequence_set_length(request->cache,
            request->sequence, current_length);
        tensor_paged_kv_sequence_end_decode(request->cache, request->sequence);
        return 0;
    }
    tensor_paged_kv_sequence_end_decode(request->cache, request->sequence);
    return 1;
}

void tensor_attention_decode_cache_clear(void) {
    const TensorAttentionBackendOps *backend = selected_decode_backend();
    if (backend != NULL && backend->decode_cache_clear != NULL)
        backend->decode_cache_clear();
}

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static int valid_f32(const TensorBuffer *buffer) {
    return buffer != NULL && tensor_buffer_storage(buffer) == TENSOR_BUFFER_STORAGE_HOST &&
           tensor_buffer_dtype(buffer) == TENSOR_DTYPE_F32;
}

static int try_backend(const TensorAttentionBackendOps *backend,
                       const TensorBuffer *query, const TensorBuffer *key,
                       const TensorBuffer *value, const TensorBuffer *mask,
                       float *output, size_t batch, size_t query_heads,
                       size_t kv_heads, size_t query_length, size_t kv_length,
                       size_t head_dim, size_t value_dim, float scale,
                       int causal) {
    return backend != NULL && backend->is_available() &&
        backend->run(tensor_buffer_data(query), tensor_buffer_data(key),
                     tensor_buffer_data(value),
                     mask == NULL ? NULL : tensor_buffer_data(mask), output,
                     batch, query_heads, kv_heads, query_length, kv_length,
                     head_dim, value_dim, scale, causal);
}

int tensor_sdpa_f32(const TensorBuffer *query, const TensorBuffer *key,
                    const TensorBuffer *value,
                    const TensorBuffer *additive_mask, float scale,
                    int causal, TensorBuffer *output,
                    TensorWorkspace *workspace,
                    TensorAttentionBackend *backend_used, char *error,
                    size_t error_capacity) {
    if (backend_used != NULL) *backend_used = TENSOR_ATTENTION_BACKEND_CPU;
    if (!valid_f32(query) || !valid_f32(key) || !valid_f32(value) ||
        !valid_f32(output) || workspace == NULL ||
        tensor_buffer_rank(query) != 4 || tensor_buffer_rank(key) != 4 ||
        tensor_buffer_rank(value) != 4 || tensor_buffer_rank(output) != 4 ||
        (additive_mask != NULL &&
         (!valid_f32(additive_mask) || tensor_buffer_rank(additive_mask) != 2)) ||
        !isfinite(scale) || (causal != 0 && causal != 1)) {
        set_error(error, error_capacity, "SDPA requires rank-4 F32 tensors, workspace, finite scale, and optional rank-2 F32 mask");
        return 0;
    }
    const uint64_t *q_shape = tensor_buffer_shape(query);
    const uint64_t *k_shape = tensor_buffer_shape(key);
    const uint64_t *v_shape = tensor_buffer_shape(value);
    const uint64_t *out_shape = tensor_buffer_shape(output);
    const size_t batch = (size_t)q_shape[0];
    const size_t q_heads = (size_t)q_shape[1];
    const size_t q_length = (size_t)q_shape[2];
    const size_t head_dim = (size_t)q_shape[3];
    const size_t kv_heads = (size_t)k_shape[1];
    const size_t kv_length = (size_t)k_shape[2];
    const size_t value_dim = (size_t)v_shape[3];
    if (batch == 0 || q_heads == 0 || kv_heads == 0 || q_length == 0 ||
        kv_length == 0 || head_dim == 0 || value_dim == 0 ||
        q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] ||
        k_shape[1] != v_shape[1] || q_heads % kv_heads != 0 ||
        k_shape[2] != v_shape[2] || k_shape[3] != q_shape[3] ||
        out_shape[0] != q_shape[0] || out_shape[1] != q_shape[1] ||
        out_shape[2] != q_shape[2] || out_shape[3] != v_shape[3]) {
        set_error(error, error_capacity, "SDPA tensor dimensions do not match");
        return 0;
    }
    const size_t causal_offset = kv_length > q_length ? kv_length - q_length : 0;
    if (additive_mask != NULL) {
        const uint64_t *mask_shape = tensor_buffer_shape(additive_mask);
        if (mask_shape[0] != q_length || mask_shape[1] != kv_length) {
            set_error(error, error_capacity, "SDPA mask shape must be [query_length, key_length]");
            return 0;
        }
        const float *mask_data = tensor_buffer_data(additive_mask);
        for (size_t index = 0; index < tensor_buffer_element_count(additive_mask); index++) {
            if (isnan(mask_data[index]) || mask_data[index] == INFINITY) {
                set_error(error, error_capacity, "SDPA mask cannot contain NaN or positive infinity");
                return 0;
            }
        }
    }

    const char *requested_backend = getenv("TENSOR_ATTENTION_BACKEND");
    if (requested_backend == NULL || requested_backend[0] == '\0')
        requested_backend = getenv("TENSOR_BACKEND");
    if (requested_backend == NULL || requested_backend[0] == '\0')
        requested_backend = "auto";
    const int use_cuda = strcmp(requested_backend, "auto") == 0 ||
                         strcmp(requested_backend, "cuda") == 0;
    const int use_metal = strcmp(requested_backend, "auto") == 0 ||
                          strcmp(requested_backend, "metal") == 0;
    if (strcmp(requested_backend, "auto") != 0 &&
        strcmp(requested_backend, "cpu") != 0 &&
        strcmp(requested_backend, "cuda") != 0 &&
        strcmp(requested_backend, "metal") != 0) {
        set_error(error, error_capacity, "TENSOR_ATTENTION_BACKEND must be auto, cpu, metal, or cuda");
        return 0;
    }
    if (use_cuda && try_backend(tensor_attention_cuda_backend(), query, key,
                                value, additive_mask,
                                tensor_buffer_data_mut(output), batch, q_heads,
                                kv_heads, q_length, kv_length, head_dim,
                                value_dim, scale, causal)) {
        if (backend_used != NULL) *backend_used = TENSOR_ATTENTION_BACKEND_CUDA;
        return 1;
    }
    if (use_metal && try_backend(tensor_attention_metal_backend(), query, key,
                                 value, additive_mask,
                                 tensor_buffer_data_mut(output), batch, q_heads,
                                 kv_heads, q_length, kv_length, head_dim,
                                 value_dim, scale, causal)) {
        if (backend_used != NULL) *backend_used = TENSOR_ATTENTION_BACKEND_METAL;
        return 1;
    }

    const TensorWorkspaceMark mark = tensor_workspace_mark(workspace);
    if (q_length > SIZE_MAX / kv_length ||
        q_length * kv_length > SIZE_MAX / sizeof(float)) {
        set_error(error, error_capacity, "SDPA score buffer size overflows addressable memory");
        return 0;
    }
    const size_t score_count = q_length * kv_length;
    float *scores = tensor_workspace_allocate(workspace, score_count * sizeof(float),
                                              _Alignof(float), error,
                                              error_capacity);
    if (scores == NULL) return 0;
    const float *q_data = tensor_buffer_data(query);
    const float *k_data = tensor_buffer_data(key);
    const float *v_data = tensor_buffer_data(value);
    const float *mask_data = additive_mask == NULL ? NULL :
                             tensor_buffer_data(additive_mask);
    float *out_data = tensor_buffer_data_mut(output);
    const size_t heads_per_kv = q_heads / kv_heads;

    for (size_t batch_index = 0; batch_index < batch; batch_index++) {
        for (size_t query_head = 0; query_head < q_heads; query_head++) {
            const size_t kv_head = query_head / heads_per_kv;
            const size_t query_head_offset =
                (batch_index * q_heads + query_head) * q_length * head_dim;
            const size_t key_head_offset =
                (batch_index * kv_heads + kv_head) * kv_length * head_dim;
            const size_t value_head_offset =
                (batch_index * kv_heads + kv_head) * kv_length * value_dim;
#if defined(__APPLE__)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)q_length, (int)kv_length, (int)head_dim, scale,
                q_data + query_head_offset, (int)head_dim,
                k_data + key_head_offset, (int)head_dim, 0.0f,
                scores, (int)kv_length);
#endif
            for (size_t query_position = 0; query_position < q_length; query_position++) {
                float *score_row = scores + query_position * kv_length;
                float maximum = -INFINITY;
                size_t positive_infinities = 0;
                for (size_t key_position = 0; key_position < kv_length; key_position++) {
#if defined(__APPLE__)
                    float score = score_row[key_position];
#else
                    float score = 0.0f;
                    const size_t q_offset = query_head_offset + query_position * head_dim;
                    const size_t k_offset = key_head_offset + key_position * head_dim;
                    for (size_t dimension = 0; dimension < head_dim; dimension++)
                        score += q_data[q_offset + dimension] * k_data[k_offset + dimension];
                    score *= scale;
#endif
                    if (mask_data != NULL)
                        score += mask_data[query_position * kv_length + key_position];
                    if (causal && key_position > query_position + causal_offset)
                        score = -INFINITY;
                    score_row[key_position] = score;
                    if (score == INFINITY) positive_infinities++;
                    if (score > maximum) maximum = score;
                }
                const size_t output_offset = ((batch_index * q_heads + query_head) *
                                              q_length + query_position) * value_dim;
                if (maximum == -INFINITY || isnan(maximum)) {
                    for (size_t dimension = 0; dimension < value_dim; dimension++)
                        out_data[output_offset + dimension] = maximum == -INFINITY ? 0.0f : NAN;
                    if (maximum == -INFINITY)
                        memset(score_row, 0, kv_length * sizeof(*score_row));
                    continue;
                }
                float denominator = 0.0f;
                if (positive_infinities == 0) {
                    for (size_t key_position = 0; key_position < kv_length; key_position++) {
                        score_row[key_position] = expf(score_row[key_position] - maximum);
                        denominator += score_row[key_position];
                    }
                } else {
                    denominator = (float)positive_infinities;
                    for (size_t key_position = 0; key_position < kv_length; key_position++)
                        score_row[key_position] = score_row[key_position] == INFINITY ? 1.0f : 0.0f;
                }
                for (size_t key_position = 0; key_position < kv_length; key_position++)
                    score_row[key_position] /= denominator;
            }
#if defined(__APPLE__)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)q_length, (int)value_dim, (int)kv_length, 1.0f,
                scores, (int)kv_length, v_data + value_head_offset,
                (int)value_dim, 0.0f,
                out_data + ((batch_index * q_heads + query_head) * q_length * value_dim),
                (int)value_dim);
#else
            for (size_t query_position = 0; query_position < q_length; query_position++) {
                const size_t output_offset = ((batch_index * q_heads + query_head) *
                                              q_length + query_position) * value_dim;
                const float *score_row = scores + query_position * kv_length;
                for (size_t dimension = 0; dimension < value_dim; dimension++) {
                    float result = 0.0f;
                    for (size_t key_position = 0; key_position < kv_length; key_position++) {
                        const size_t v_offset = value_head_offset + key_position * value_dim;
                        result += score_row[key_position] * v_data[v_offset + dimension];
                    }
                    out_data[output_offset + dimension] = result;
                }
            }
#endif
        }
    }
    return tensor_workspace_release(workspace, mark, error, error_capacity);
}

int tensor_sdpa_f16(const TensorBuffer *query, const TensorBuffer *key,
                    const TensorBuffer *value,
                    const TensorBuffer *additive_mask, float scale,
                    int causal, TensorBuffer *output,
                    TensorWorkspace *workspace,
                    TensorAttentionBackend *backend_used, char *error,
                    size_t error_capacity) {
    if (backend_used != NULL) *backend_used = TENSOR_ATTENTION_BACKEND_CPU;
    if (query == NULL || key == NULL || value == NULL || output == NULL || workspace == NULL ||
        tensor_buffer_dtype(query) != TENSOR_DTYPE_F16 ||
        tensor_buffer_dtype(key) != TENSOR_DTYPE_F16 ||
        tensor_buffer_dtype(value) != TENSOR_DTYPE_F16 ||
        tensor_buffer_dtype(output) != TENSOR_DTYPE_F16 ||
        tensor_buffer_rank(query) != 4 || tensor_buffer_rank(key) != 4 ||
        tensor_buffer_rank(value) != 4 || tensor_buffer_rank(output) != 4 ||
        (additive_mask != NULL && (tensor_buffer_dtype(additive_mask) != TENSOR_DTYPE_F16 ||
                                   tensor_buffer_rank(additive_mask) != 2)) ||
        !isfinite(scale) || (causal != 0 && causal != 1)) {
        set_error(error, error_capacity, "SDPA requires rank-4 F16 tensors, workspace, finite scale, and optional rank-2 F16 mask");
        return 0;
    }
    const uint64_t *qs = tensor_buffer_shape(query), *ks = tensor_buffer_shape(key);
    const uint64_t *vs = tensor_buffer_shape(value), *os = tensor_buffer_shape(output);
    const size_t batches = (size_t)qs[0], q_heads = (size_t)qs[1];
    const size_t q_len = (size_t)qs[2], dim = (size_t)qs[3];
    const size_t kv_heads = (size_t)ks[1], kv_len = (size_t)ks[2], vdim = (size_t)vs[3];
    if (batches == 0 || q_heads == 0 || kv_heads == 0 || q_len == 0 || kv_len == 0 ||
        dim == 0 || vdim == 0 || qs[0] != ks[0] || qs[0] != vs[0] ||
        ks[1] != vs[1] || q_heads % kv_heads != 0 || ks[2] != vs[2] ||
        ks[3] != dim || os[0] != batches || os[1] != q_heads ||
        os[2] != q_len || os[3] != vdim) {
        set_error(error, error_capacity, "SDPA tensor dimensions do not match");
        return 0;
    }
    if (additive_mask != NULL) {
        const uint64_t *ms = tensor_buffer_shape(additive_mask);
        if (ms[0] != q_len || ms[1] != kv_len) {
            set_error(error, error_capacity, "SDPA mask shape must be [query_length, key_length]");
            return 0;
        }
    }
    const TensorAttentionBackendOps *backend = selected_decode_backend();
    if (backend == NULL || backend->run_f16 == NULL || !backend->is_available() ||
        !backend->run_f16(tensor_buffer_data(query), tensor_buffer_data(key),
            tensor_buffer_data(value), additive_mask == NULL ? NULL : tensor_buffer_data(additive_mask),
            tensor_buffer_data_mut(output), batches, q_heads, kv_heads, q_len, kv_len,
            dim, vdim, scale, causal)) {
        set_error(error, error_capacity, "FP16 SDPA requires a working accelerator attention kernel");
        return 0;
    }
    if (backend_used != NULL) *backend_used = backend->kind;
    return 1;


}
