#include "attention_backend.h"
#include <cuda_fp16.h>
#include <math_constants.h>

#include "cuda_common.cuh"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace smolvlm_cuda;

namespace {

template<typename T> __global__ void sdpa_score_kernel(const T *q, const T *k,
    const T *mask, float *scores, size_t batches, size_t qh, size_t kh,
    size_t qlen, size_t klen, size_t dim, float scale, bool causal) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t rows = batches * qh * qlen, total = rows * klen;
    if (index >= total) return;
    size_t keypos = index % klen, row = index / klen;
    size_t qpos = row % qlen, head = (row / qlen) % qh, batch = row / (qlen * qh);
    size_t kvhead = head / (qh / kh);
    const T *qp = q + ((batch * qh + head) * qlen + qpos) * dim;
    const T *kp = k + ((batch * kh + kvhead) * klen + keypos) * dim;
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) sum += as_float(qp[d]) * as_float(kp[d]);
    sum *= scale;
    if (mask != nullptr) sum += as_float(mask[qpos * klen + keypos]);
    if (causal && keypos > qpos + (klen > qlen ? klen - qlen : 0)) sum = -CUDART_INF_F;
    scores[index] = sum;
}

__global__ void softmax_rows_kernel(float *scores, size_t rows, size_t width) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float *p = scores + row * width, maximum = -CUDART_INF_F;
    for (size_t i = 0; i < width; ++i) if (p[i] > maximum) maximum = p[i];
    if (!isfinite(maximum)) { for (size_t i = 0; i < width; ++i) p[i] = 0.0f; return; }
    float sum = 0.0f;
    for (size_t i = 0; i < width; ++i) { p[i] = expf(p[i] - maximum); sum += p[i]; }
    float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
    for (size_t i = 0; i < width; ++i) p[i] *= inv;
}

template<typename T> __global__ void sdpa_output_kernel(const float *scores,
    const T *v, T *out, size_t batches, size_t qh, size_t kh,
    size_t qlen, size_t klen, size_t vdim) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batches * qh * qlen * vdim;
    if (index >= total) return;
    size_t d = index % vdim, qpos = (index / vdim) % qlen;
    size_t head = (index / (vdim * qlen)) % qh, batch = index / (vdim * qlen * qh);
    size_t kvhead = head / (qh / kh), row = ((batch * qh + head) * qlen + qpos);
    float sum = 0.0f;
    for (size_t j = 0; j < klen; ++j)
        sum += scores[row * klen + j] * as_float(v[((batch * kh + kvhead) * klen + j) * vdim + d]);
    out[index] = from_float<T>(sum);
}

template<typename T> bool sdpa(const T *qh, const T *kh, const T *vh,
    const T *mh, T *oh, size_t batches, size_t qheads, size_t kvheads,
    size_t qlen, size_t klen, size_t dim, size_t vdim, float scale, int causal) {
    if (!cuda_ready() || qh == nullptr || kh == nullptr || vh == nullptr || oh == nullptr ||
        batches == 0 || qheads == 0 || kvheads == 0 || qheads % kvheads != 0 ||
        qlen == 0 || klen == 0 || dim == 0 || vdim == 0 || !std::isfinite(scale) ||
        batches > SIZE_MAX / qheads / qlen || batches * qheads * qlen > SIZE_MAX / klen) return false;
    size_t qcount = batches * qheads * qlen * dim;
    size_t kvcount = batches * kvheads * klen * dim;
    size_t vcount = batches * kvheads * klen * vdim;
    size_t outcount = batches * qheads * qlen * vdim;
    size_t rows = batches * qheads * qlen, scorecount = rows * klen;
    T *dq = nullptr, *dk = nullptr, *dv = nullptr, *dm = nullptr, *dout = nullptr;
    float *ds = nullptr;
    bool ok = upload(qh, qcount, &dq) && upload(kh, kvcount, &dk) && upload(vh, vcount, &dv) &&
        alloc_device(scorecount, &ds) && alloc_device(outcount, &dout);
    if (ok && mh != nullptr) ok = upload(mh, qlen * klen, &dm);
    if (ok) {
        sdpa_score_kernel<<<(unsigned)((scorecount + 255) / 256), 256>>>(dq, dk, dm, ds,
            batches, qheads, kvheads, qlen, klen, dim, scale, causal != 0);
        softmax_rows_kernel<<<(unsigned)((rows + 127) / 128), 128>>>(ds, rows, klen);
        sdpa_output_kernel<<<(unsigned)((outcount + 255) / 256), 256>>>(ds, dv, dout,
            batches, qheads, kvheads, qlen, klen, vdim);
        ok = cudaGetLastError() == cudaSuccess && download(oh, dout, outcount);
    }
    if (dq) cudaFree(dq); if (dk) cudaFree(dk); if (dv) cudaFree(dv);
    if (dm) cudaFree(dm); if (dout) cudaFree(dout); if (ds) cudaFree(ds);
    return ok;
}

template<typename T> __global__ void decode_embedding_kernel(
    const T *embedding, const uint32_t *token, T *hidden, size_t width) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width) hidden[index] = embedding[(size_t)(*token) * width + index];
}

template<typename T> __global__ void decode_rms_kernel(
    const T *input, const T *weight, T *output, size_t width, float epsilon) {
    __shared__ float partials[128];
    const unsigned lane = threadIdx.x;
    float sum = 0.0f;
    for (size_t i = lane; i < width; i += 128) {
        const float value = as_float(input[i]);
        sum = fmaf(value, value, sum);
    }
    partials[lane] = sum;
    __syncthreads();
    for (unsigned stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) partials[lane] += partials[lane + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partials[0] / (float)width + epsilon);
    for (size_t i = lane; i < width; i += 128)
        output[i] = from_float<T>(as_float(input[i]) * scale * as_float(weight[i]));
}

template<typename T> __global__ void decode_gemv_kernel(
    const T *input, const T *weight, T *output,
    size_t input_width, size_t output_width) {
    __shared__ float partials[128];
    const unsigned lane = threadIdx.x;
    const size_t row = blockIdx.x;
    if (row >= output_width) return;
    float sum = 0.0f;
    const T *row_weight = weight + row * input_width;
    for (size_t i = lane; i < input_width; i += 128)
        sum = fmaf(as_float(input[i]), as_float(row_weight[i]), sum);
    partials[lane] = sum;
    __syncthreads();
    for (unsigned stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) partials[lane] += partials[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[row] = from_float<T>(partials[0]);
}

template<typename T> __global__ void decode_gemv_residual_kernel(
    const T *input, const T *weight, const T *residual, T *output,
    size_t input_width, size_t output_width) {
    __shared__ float partials[128];
    const unsigned lane = threadIdx.x;
    const size_t row = blockIdx.x;
    if (row >= output_width) return;
    float sum = 0.0f;
    const T *row_weight = weight + row * input_width;
    for (size_t i = lane; i < input_width; i += 128)
        sum = fmaf(as_float(input[i]), as_float(row_weight[i]), sum);
    partials[lane] = sum;
    __syncthreads();
    for (unsigned stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) partials[lane] += partials[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[row] = from_float<T>(partials[0] + as_float(residual[row]));
}

template<typename T> __global__ void decode_gemv_silu_kernel(
    const T *input, const T *gate_up_weight, T *output,
    size_t input_width, size_t output_width) {
    __shared__ float gate_partials[128];
    __shared__ float up_partials[128];
    const unsigned lane = threadIdx.x;
    const size_t row = blockIdx.x;
    if (row >= output_width) return;
    const T *gate = gate_up_weight + row * input_width;
    const T *up = gate_up_weight + (row + output_width) * input_width;
    float gate_sum = 0.0f, up_sum = 0.0f;
    for (size_t i = lane; i < input_width; i += 128) {
        const float value = as_float(input[i]);
        gate_sum = fmaf(value, as_float(gate[i]), gate_sum);
        up_sum = fmaf(value, as_float(up[i]), up_sum);
    }
    gate_partials[lane] = gate_sum;
    up_partials[lane] = up_sum;
    __syncthreads();
    for (unsigned stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) {
            gate_partials[lane] += gate_partials[lane + stride];
            up_partials[lane] += up_partials[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        const float gate_value = gate_partials[0];
        output[row] = from_float<T>(gate_value / (1.0f + expf(-gate_value)) * up_partials[0]);
    }
}

template<typename T> __global__ void decode_rope_cache_kernel(
    const T *qkv, T *rotated_query, T *key_cache, T *value_cache,
    size_t hidden_size, size_t kv_width, size_t kv_heads, size_t head_dim,
    size_t position, float theta) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t half_dim = head_dim / 2;
    if (index < hidden_size) {
        const size_t dimension = index % head_dim;
        const size_t base = (index / head_dim) * head_dim;
        const size_t paired = base + (dimension < half_dim ? dimension + half_dim : dimension - half_dim);
        const size_t frequency = dimension < half_dim ? dimension : dimension - half_dim;
        const float angle = (float)position / powf(theta, 2.0f * (float)frequency / (float)head_dim);
        const float sign = dimension < half_dim ? -1.0f : 1.0f;
        const float value = as_float(qkv[index]) * cosf(angle) +
            sign * as_float(qkv[base + (paired - base)]) * sinf(angle);
        rotated_query[index] = from_float<T>(value);
    }
    if (index < kv_width) {
        const T *key = qkv + hidden_size;
        const T *value = key + kv_width;
        const size_t dimension = index % head_dim;
        const size_t base = (index / head_dim) * head_dim;
        const size_t paired = base + (dimension < half_dim ? dimension + half_dim : dimension - half_dim);
        const size_t frequency = dimension < half_dim ? dimension : dimension - half_dim;
        const float angle = (float)position / powf(theta, 2.0f * (float)frequency / (float)head_dim);
        const float sign = dimension < half_dim ? -1.0f : 1.0f;
        const size_t cache_index = position * kv_width + index;
        key_cache[cache_index] = from_float<T>(as_float(key[index]) * cosf(angle) +
            sign * as_float(key[base + (paired - base)]) * sinf(angle));
        value_cache[cache_index] = value[index];
    }
    (void)kv_heads;
}

template<typename T> __global__ void decode_score_kernel(
    const T *query, const T *key_cache, float *scores,
    size_t query_heads, size_t kv_heads, size_t kv_length,
    size_t kv_width, size_t head_dim, float scale) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = query_heads * kv_length;
    if (index >= count) return;
    const size_t head = index / kv_length;
    const size_t position = index % kv_length;
    const size_t kv_head = head / (query_heads / kv_heads);
    float sum = 0.0f;
    for (size_t d = 0; d < head_dim; ++d)
        sum = fmaf(as_float(query[head * head_dim + d]),
                   as_float(key_cache[position * kv_width + kv_head * head_dim + d]), sum);
    scores[index] = sum * scale;
}

__global__ void decode_softmax_kernel(float *scores, size_t rows, size_t width) {
    const size_t row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    float *values = scores + row * width;
    float maximum = -CUDART_INF_F;
    for (size_t i = 0; i < width; ++i) maximum = fmaxf(maximum, values[i]);
    if (!isfinite(maximum)) {
        for (size_t i = 0; i < width; ++i) values[i] = 0.0f;
        return;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < width; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (size_t i = 0; i < width; ++i) values[i] *= inverse;
}

template<typename T> __global__ void decode_attention_output_kernel(
    const float *scores, const T *value_cache, T *output,
    size_t query_heads, size_t kv_heads, size_t kv_length,
    size_t kv_width, size_t head_dim) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = query_heads * head_dim;
    if (index >= count) return;
    const size_t head = index / head_dim;
    const size_t dimension = index % head_dim;
    const size_t kv_head = head / (query_heads / kv_heads);
    float sum = 0.0f;
    for (size_t position = 0; position < kv_length; ++position)
        sum = fmaf(scores[head * kv_length + position],
            as_float(value_cache[position * kv_width + kv_head * head_dim + dimension]), sum);
    output[index] = from_float<T>(sum);
}

template<typename T> __global__ void decode_argmax_kernel(
    const T *logits, uint32_t *token, size_t vocabulary_size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    size_t best = 0;
    float maximum = as_float(logits[0]);
    for (size_t i = 1; i < vocabulary_size; ++i) {
        const float value = as_float(logits[i]);
        if (value > maximum) { maximum = value; best = i; }
    }
    *token = (uint32_t)best;
}

struct DecodeWeight { void *device = nullptr; size_t bytes = 0; bool fp16 = false; };
std::unordered_map<const void *, DecodeWeight> decode_weights;
std::mutex decode_mutex;

void *decode_weight_buffer(const void *host, size_t bytes, bool fp16) {
    if (host == nullptr || bytes == 0) return nullptr;
    auto found = decode_weights.find(host);
    if (found != decode_weights.end() && found->second.bytes == bytes &&
        found->second.fp16 == fp16) return found->second.device;
    if (found != decode_weights.end()) {
        if (found->second.device != nullptr) cudaFree(found->second.device);
        decode_weights.erase(found);
    }
    void *device = nullptr;
    cudaError_t allocation_error = cudaMalloc(&device, bytes);
    if (allocation_error != cudaSuccess) {
        fprintf(stderr, "cuda decode: weight cudaMalloc(%zu) failed: %s\n",
                bytes, cudaGetErrorString(allocation_error));
        return nullptr;
    }
    cudaError_t copy_error = cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
    if (copy_error != cudaSuccess) {
        fprintf(stderr, "cuda decode: weight copy(%zu) failed: %s\n",
                bytes, cudaGetErrorString(copy_error));
        if (device != nullptr) cudaFree(device);
        return nullptr;
    }
    decode_weights.emplace(host, DecodeWeight{device, bytes, fp16});
    return device;
}

template<typename T> struct DecodeEntry {
    const void *value_host = nullptr;
    const void *q_host = nullptr, *k_host = nullptr, *v_host = nullptr;
    const void *gate_host = nullptr, *up_host = nullptr;
    size_t capacity = 0, hidden = 0, intermediate = 0, valid_length = 0;
    T *key = nullptr, *value = nullptr, *norm = nullptr, *qkv_weight = nullptr;
    T *qkv = nullptr, *query = nullptr, *attention = nullptr, *gate_up_weight = nullptr;
    T *residual = nullptr, *post_norm = nullptr, *activated = nullptr;
    float *scores = nullptr;

    void release() {
        void *buffers[] = {key, value, norm, qkv_weight, qkv, query, attention,
            gate_up_weight, residual, post_norm, activated, scores};
        for (void *buffer : buffers) if (buffer != nullptr) cudaFree(buffer);
        key = value = norm = qkv_weight = qkv = query = attention = nullptr;
        gate_up_weight = residual = post_norm = activated = nullptr;
        scores = nullptr;
        capacity = hidden = intermediate = valid_length = 0;
        value_host = q_host = k_host = v_host = gate_host = up_host = nullptr;
    }
    ~DecodeEntry() { release(); }
};

template<typename T> std::unordered_map<const void *, std::unique_ptr<DecodeEntry<T>>> &decode_entries() {
    static std::unordered_map<const void *, std::unique_ptr<DecodeEntry<T>>> entries;
    return entries;
}

template<typename T> struct DecodeScratch {
    T *hidden_a = nullptr, *hidden_b = nullptr, *norm = nullptr, *logits = nullptr;
    uint32_t *tokens = nullptr, *initial_token = nullptr;
    size_t hidden_size = 0, vocabulary_size = 0, token_count = 0;
    bool ensure(size_t hidden, size_t vocabulary, size_t count) {
        if (hidden == hidden_size && vocabulary == vocabulary_size && count == token_count &&
            hidden_a != nullptr && hidden_b != nullptr && norm != nullptr &&
            logits != nullptr && tokens != nullptr && initial_token != nullptr)
            return true;
        release();
        if (!alloc_device(hidden, &hidden_a) || !alloc_device(hidden, &hidden_b) ||
            !alloc_device(hidden, &norm) || !alloc_device(vocabulary, &logits) ||
            !alloc_device(count, &tokens) || !alloc_device((size_t)1, &initial_token)) {
            release(); return false;
        }
        hidden_size = hidden; vocabulary_size = vocabulary; token_count = count;
        return true;
    }
    void release() {
        if (hidden_a) cudaFree(hidden_a); if (hidden_b) cudaFree(hidden_b);
        if (norm) cudaFree(norm); if (logits) cudaFree(logits);
        if (tokens) cudaFree(tokens); if (initial_token) cudaFree(initial_token);
        hidden_a = hidden_b = norm = logits = nullptr;
        tokens = initial_token = nullptr;
        hidden_size = vocabulary_size = token_count = 0;
    }
    ~DecodeScratch() { release(); }
};

template<typename T> DecodeScratch<T> &decode_scratch() {
    static DecodeScratch<T> scratch;
    return scratch;
}

template<typename T> bool prepare_decode_entry(DecodeEntry<T> &entry,
    const void *value_host, size_t hidden, size_t intermediate,
    size_t capacity, size_t kv_width, size_t query_heads) {
    if (entry.key != nullptr && entry.value_host == value_host &&
        entry.hidden == hidden && entry.intermediate == intermediate &&
        entry.capacity == capacity) return true;
    entry.release();
    const size_t qkv_width = hidden + 2 * kv_width;
    if (capacity > SIZE_MAX / kv_width || qkv_width > SIZE_MAX / hidden ||
        intermediate > SIZE_MAX / hidden / 2 || query_heads > SIZE_MAX / capacity)
        return false;
    bool ok = alloc_device(capacity * kv_width, &entry.key) &&
        alloc_device(capacity * kv_width, &entry.value) &&
        alloc_device(hidden, &entry.norm) &&
        alloc_device(qkv_width * hidden, &entry.qkv_weight) &&
        alloc_device(qkv_width, &entry.qkv) &&
        alloc_device(hidden, &entry.query) &&
        alloc_device(hidden, &entry.attention) &&
        alloc_device(2 * intermediate * hidden, &entry.gate_up_weight) &&
        alloc_device(hidden, &entry.residual) &&
        alloc_device(hidden, &entry.post_norm) &&
        alloc_device(intermediate, &entry.activated) &&
        alloc_device(query_heads * capacity, &entry.scores);
    if (!ok) { entry.release(); return false; }
    entry.value_host = value_host;
    entry.hidden = hidden;
    entry.intermediate = intermediate;
    entry.capacity = capacity;
    entry.valid_length = 0;
    return true;
}

template<typename T, typename Layer> bool prepare_packed_weights(
    DecodeEntry<T> &entry, const Layer &layer, size_t hidden, size_t kv_width,
    size_t intermediate) {
    if (entry.q_host == layer.q_weight && entry.k_host == layer.k_weight &&
        entry.v_host == layer.v_weight && entry.gate_host == layer.gate_weight &&
        entry.up_host == layer.up_weight) return true;
    const size_t qkv_width = hidden + 2 * kv_width;
    std::vector<T> qkv, gate_up;
    try {
        qkv.resize(qkv_width * hidden);
        gate_up.resize(2 * intermediate * hidden);
    } catch (...) { return false; }
    const T *q = reinterpret_cast<const T *>(layer.q_weight);
    const T *k = reinterpret_cast<const T *>(layer.k_weight);
    const T *v = reinterpret_cast<const T *>(layer.v_weight);
    const T *gate = reinterpret_cast<const T *>(layer.gate_weight);
    const T *up = reinterpret_cast<const T *>(layer.up_weight);
    memcpy(qkv.data(), q, hidden * hidden * sizeof(T));
    memcpy(qkv.data() + hidden * hidden, k, kv_width * hidden * sizeof(T));
    memcpy(qkv.data() + (hidden + kv_width) * hidden, v,
           kv_width * hidden * sizeof(T));
    memcpy(gate_up.data(), gate, intermediate * hidden * sizeof(T));
    memcpy(gate_up.data() + intermediate * hidden, up,
           intermediate * hidden * sizeof(T));
    if (cudaMemcpy(entry.qkv_weight, qkv.data(), qkv.size() * sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(entry.gate_up_weight, gate_up.data(), gate_up.size() * sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) return false;
    entry.q_host = layer.q_weight; entry.k_host = layer.k_weight;
    entry.v_host = layer.v_weight; entry.gate_host = layer.gate_weight;
    entry.up_host = layer.up_weight;
    return true;
}

template<typename T, typename Layer> int decode_sequence_impl(
    const Layer *layers, size_t layer_count, const T *embedding,
    uint32_t initial_token, size_t token_count, uint32_t *generated_tokens,
    const T *const *key_cache, const T *const *value_cache,
    size_t past_length, size_t cache_capacity, size_t hidden_size,
    size_t intermediate_size, size_t query_heads, size_t kv_heads,
    size_t head_dim, float rms_norm_epsilon, double rope_theta,
    const T *final_norm, const T *lm_head, size_t vocabulary_size) {
    if (!cuda_ready() || layers == nullptr || layer_count == 0 || embedding == nullptr ||
        token_count == 0 || generated_tokens == nullptr || key_cache == nullptr ||
        value_cache == nullptr || final_norm == nullptr || lm_head == nullptr ||
        hidden_size == 0 || intermediate_size == 0 || vocabulary_size == 0 ||
        query_heads == 0 || kv_heads == 0 || head_dim == 0 ||
        query_heads > SIZE_MAX / head_dim || query_heads * head_dim != hidden_size ||
        query_heads % kv_heads != 0 ||
        head_dim % 2 != 0 || past_length >= cache_capacity ||
        token_count > cache_capacity - past_length || cache_capacity > UINT32_MAX ||
        vocabulary_size > UINT32_MAX || hidden_size > UINT32_MAX ||
        intermediate_size > UINT32_MAX || head_dim > UINT32_MAX ||
        token_count > SIZE_MAX / sizeof(uint32_t) ||
        vocabulary_size > SIZE_MAX / hidden_size / sizeof(T)) {
        fprintf(stderr, "cuda decode: invalid dimensions or arguments\n");
        return 0;
    }
    if (kv_heads > SIZE_MAX / head_dim) return 0;
    const size_t kv_width = kv_heads * head_dim;
    if (kv_width > (SIZE_MAX - hidden_size) / 2 ||
        cache_capacity > SIZE_MAX / kv_width ||
        hidden_size > SIZE_MAX / (hidden_size + 2 * kv_width) ||
        intermediate_size > SIZE_MAX / hidden_size / 2) {
        fprintf(stderr, "cuda decode: dimensions overflow\n");
        return 0;
    }

    std::lock_guard<std::mutex> lock(decode_mutex);
    auto &entries = decode_entries<T>();
    std::vector<DecodeEntry<T> *> layer_entries;
    try { layer_entries.reserve(layer_count); } catch (...) { return 0; }
    for (size_t index = 0; index < layer_count; ++index) {
        if (layers[index].input_norm == nullptr || layers[index].q_weight == nullptr ||
            layers[index].k_weight == nullptr || layers[index].v_weight == nullptr ||
            layers[index].o_weight == nullptr || layers[index].post_norm == nullptr ||
            layers[index].gate_weight == nullptr || layers[index].up_weight == nullptr ||
            layers[index].down_weight == nullptr || key_cache[index] == nullptr ||
            value_cache[index] == nullptr) {
            fprintf(stderr, "cuda decode: layer %zu has a null tensor pointer\n", index);
            return 0;
        }
        auto &slot = entries[key_cache[index]];
        if (!slot) {
            try { slot = std::make_unique<DecodeEntry<T>>(); }
            catch (...) { return 0; }
        }
        DecodeEntry<T> &entry = *slot;
        if (entry.key != nullptr && (entry.value_host != value_cache[index] ||
            entry.capacity != cache_capacity || entry.hidden != hidden_size ||
            entry.intermediate != intermediate_size)) entry.release();
        const bool new_buffers = entry.key == nullptr;
        if (!prepare_decode_entry(entry, value_cache[index], hidden_size,
                                  intermediate_size, cache_capacity, kv_width,
                                  query_heads)) {
            fprintf(stderr, "cuda decode: layer %zu buffer allocation failed\n", index);
            return 0;
        }
        if (new_buffers) {
            const size_t prefix_bytes = past_length * kv_width * sizeof(T);
            if (past_length > 0 &&
                (cudaMemcpy(entry.key, key_cache[index], prefix_bytes,
                            cudaMemcpyHostToDevice) != cudaSuccess ||
                 cudaMemcpy(entry.value, value_cache[index], prefix_bytes,
                            cudaMemcpyHostToDevice) != cudaSuccess)) {
                fprintf(stderr, "cuda decode: layer %zu KV upload failed\n", index);
                return 0;
            }
            entry.valid_length = past_length;
        } else if (entry.valid_length != past_length) {
            fprintf(stderr, "cuda decode: layer %zu cache length mismatch (%zu != %zu)\n",
                    index, entry.valid_length, past_length);
            return 0;
        }
        if (!prepare_packed_weights(entry, layers[index], hidden_size,
                                    kv_width, intermediate_size)) {
            fprintf(stderr, "cuda decode: layer %zu weight packing failed\n", index);
            return 0;
        }
        layer_entries.push_back(&entry);
    }

    DecodeScratch<T> &scratch = decode_scratch<T>();
    if (!scratch.ensure(hidden_size, vocabulary_size, token_count) ||
        cudaMemcpy(scratch.initial_token, &initial_token, sizeof(initial_token),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cuda decode: scratch allocation or input token upload failed\n");
        return 0;
    }
    const bool fp16 = sizeof(T) == sizeof(uint16_t);
    T *device_embedding = static_cast<T *>(decode_weight_buffer(
        embedding, vocabulary_size * hidden_size * sizeof(T), fp16));
    T *device_final_norm = static_cast<T *>(decode_weight_buffer(
        final_norm, hidden_size * sizeof(T), fp16));
    T *device_lm_head = static_cast<T *>(decode_weight_buffer(
        lm_head, vocabulary_size * hidden_size * sizeof(T), fp16));
    if (device_embedding == nullptr || device_final_norm == nullptr ||
        device_lm_head == nullptr) {
        fprintf(stderr, "cuda decode: embedding or output weight upload failed\n");
        return 0;
    }

    const unsigned hidden_blocks = (unsigned)((hidden_size + 255) / 256);
    for (size_t token_index = 0; token_index < token_count; ++token_index) {
        const uint32_t *input_token = token_index == 0 ? scratch.initial_token :
            scratch.tokens + token_index - 1;
        decode_embedding_kernel<<<hidden_blocks, 256>>>(device_embedding,
            input_token, scratch.hidden_a, hidden_size);
        T *current_hidden = scratch.hidden_a;
        T *next_hidden = scratch.hidden_b;
        const size_t position = past_length + token_index;
        for (size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
            DecodeEntry<T> &entry = *layer_entries[layer_index];
            const Layer &layer = layers[layer_index];
            const T *input_norm = static_cast<T *>(decode_weight_buffer(
                layer.input_norm, hidden_size * sizeof(T), fp16));
            if (input_norm == nullptr) {
                fprintf(stderr, "cuda decode: layer %zu input norm upload failed\n", layer_index);
                return 0;
            }
            decode_rms_kernel<<<1, 128>>>(current_hidden, input_norm, entry.norm,
                hidden_size, rms_norm_epsilon);
            const size_t qkv_width = hidden_size + 2 * kv_width;
            decode_gemv_kernel<<<(unsigned)qkv_width, 128>>>(entry.norm,
                entry.qkv_weight, entry.qkv, hidden_size, qkv_width);
            const unsigned rope_count = (unsigned)((std::max(hidden_size, kv_width) + 255) / 256);
            decode_rope_cache_kernel<<<rope_count, 256>>>(entry.qkv, entry.query,
                entry.key, entry.value, hidden_size, kv_width, kv_heads, head_dim,
                position, (float)rope_theta);
            const size_t kv_length = position + 1;
            decode_score_kernel<<<(unsigned)((query_heads * kv_length + 127) / 128), 128>>>(
                entry.query, entry.key, entry.scores, query_heads, kv_heads,
                kv_length, kv_width, head_dim, 1.0f / sqrtf((float)head_dim));
            decode_softmax_kernel<<<(unsigned)query_heads, 1>>>(
                entry.scores, query_heads, kv_length);
            decode_attention_output_kernel<<<(unsigned)((hidden_size + 255) / 256), 256>>>(
                entry.scores, entry.value, entry.attention, query_heads, kv_heads,
                kv_length, kv_width, head_dim);
            T *device_o_weight = static_cast<T *>(decode_weight_buffer(
                layer.o_weight, hidden_size * hidden_size * sizeof(T), fp16));
            if (device_o_weight == nullptr) {
                fprintf(stderr, "cuda decode: layer %zu output projection upload failed\n", layer_index);
                return 0;
            }
            decode_gemv_residual_kernel<<<(unsigned)hidden_size, 128>>>(
                entry.attention, device_o_weight, current_hidden, entry.residual,
                hidden_size, hidden_size);
            const T *post_norm = static_cast<T *>(decode_weight_buffer(
                layer.post_norm, hidden_size * sizeof(T), fp16));
            if (post_norm == nullptr) {
                fprintf(stderr, "cuda decode: layer %zu post norm upload failed\n", layer_index);
                return 0;
            }
            decode_rms_kernel<<<1, 128>>>(entry.residual, post_norm,
                entry.post_norm, hidden_size, rms_norm_epsilon);
            decode_gemv_silu_kernel<<<(unsigned)intermediate_size, 128>>>(
                entry.post_norm, entry.gate_up_weight, entry.activated,
                hidden_size, intermediate_size);
            T *device_down_weight = static_cast<T *>(decode_weight_buffer(
                layer.down_weight, hidden_size * intermediate_size * sizeof(T), fp16));
            if (device_down_weight == nullptr) {
                fprintf(stderr, "cuda decode: layer %zu down projection upload failed\n", layer_index);
                return 0;
            }
            decode_gemv_residual_kernel<<<(unsigned)hidden_size, 128>>>(
                entry.activated, device_down_weight, entry.residual, next_hidden,
                intermediate_size, hidden_size);
            T *swap = current_hidden; current_hidden = next_hidden; next_hidden = swap;
        }
        decode_rms_kernel<<<1, 128>>>(current_hidden, device_final_norm,
            scratch.norm, hidden_size, rms_norm_epsilon);
        decode_gemv_kernel<<<(unsigned)vocabulary_size, 128>>>(scratch.norm,
            device_lm_head, scratch.logits, hidden_size, vocabulary_size);
        decode_argmax_kernel<<<1, 1>>>(scratch.logits, scratch.tokens + token_index,
            vocabulary_size);
    }
    cudaError_t last_error = cudaGetLastError();
    cudaError_t sync_error = last_error == cudaSuccess ? cudaDeviceSynchronize() : last_error;
    if (sync_error != cudaSuccess) {
        fprintf(stderr, "cuda decode: kernel execution failed: %s\n", cudaGetErrorString(sync_error));
        return 0;
    }
    if (!download(generated_tokens, scratch.tokens, token_count)) {
        fprintf(stderr, "cuda decode: generated token download failed\n");
        return 0;
    }
    const size_t valid_bytes = (past_length + token_count) * kv_width * sizeof(T);
    for (size_t index = 0; index < layer_count; ++index) {
        DecodeEntry<T> &entry = *layer_entries[index];
        if (!download((T *)key_cache[index], entry.key, (past_length + token_count) * kv_width) ||
            !download((T *)value_cache[index], entry.value, (past_length + token_count) * kv_width))
        {
            fprintf(stderr, "cuda decode: layer %zu cache download failed\n", index);
            return 0;
        }
        entry.valid_length = past_length + token_count;
    }
    (void)valid_bytes;
    return 1;
}

void clear_decode_cache() {
    std::lock_guard<std::mutex> lock(decode_mutex);
    decode_entries<float>().clear();
    decode_entries<__half>().clear();
    for (auto &entry : decode_weights)
        if (entry.second.device != nullptr) cudaFree(entry.second.device);
    decode_weights.clear();
    decode_scratch<float>().release();
    decode_scratch<__half>().release();
}


} // namespace

extern "C" {

static int attention_available() { return cuda_ready(); }
static int attention_f32(const float*q,const float*k,const float*v,const float*m,float*o,size_t b,size_t qh,size_t kh,size_t ql,size_t kl,size_t d,size_t vd,float s,int c) { return sdpa(q,k,v,m,o,b,qh,kh,ql,kl,d,vd,s,c); }
static int attention_f16(const uint16_t*q,const uint16_t*k,const uint16_t*v,const uint16_t*m,uint16_t*o,size_t b,size_t qh,size_t kh,size_t ql,size_t kl,size_t d,size_t vd,float s,int c) { return sdpa((const __half *)q,(const __half *)k,(const __half *)v,(const __half *)m,(__half *)o,b,qh,kh,ql,kl,d,vd,s,c); }
static int decode_sequence_available() { return cuda_ready(); }
static int decode_sequence_f16_available() { return cuda_ready(); }
static int decode_sequence_f32(const TensorAttentionDecoderLayer *layers,
    size_t layer_count, const float *embedding, uint32_t initial_token,
    size_t token_count, uint32_t *generated_tokens,
    const float *const *key_cache, const float *const *value_cache,
    size_t past_length, size_t cache_capacity, size_t hidden_size,
    size_t intermediate_size, size_t query_heads, size_t kv_heads,
    size_t head_dim, float rms_norm_epsilon, double rope_theta,
    const float *final_norm, const float *lm_head, size_t vocabulary_size) {
    return decode_sequence_impl(layers, layer_count, embedding, initial_token,
        token_count, generated_tokens, key_cache, value_cache, past_length,
        cache_capacity, hidden_size, intermediate_size, query_heads, kv_heads,
        head_dim, rms_norm_epsilon, rope_theta, final_norm, lm_head,
        vocabulary_size);
}
static int decode_sequence_f16(const TensorAttentionDecoderLayerF16 *layers,
    size_t layer_count, const uint16_t *embedding, uint32_t initial_token,
    size_t token_count, uint32_t *generated_tokens,
    const uint16_t *const *key_cache, const uint16_t *const *value_cache,
    size_t past_length, size_t cache_capacity, size_t hidden_size,
    size_t intermediate_size, size_t query_heads, size_t kv_heads,
    size_t head_dim, float rms_norm_epsilon, double rope_theta,
    const uint16_t *final_norm, const uint16_t *lm_head, size_t vocabulary_size) {
    return decode_sequence_impl(layers, layer_count,
        reinterpret_cast<const __half *>(embedding), initial_token, token_count,
        generated_tokens, reinterpret_cast<const __half *const *>(key_cache),
        reinterpret_cast<const __half *const *>(value_cache), past_length,
        cache_capacity, hidden_size, intermediate_size, query_heads, kv_heads,
        head_dim, rms_norm_epsilon, rope_theta,
        reinterpret_cast<const __half *>(final_norm),
        reinterpret_cast<const __half *>(lm_head), vocabulary_size);
}
const TensorAttentionBackendOps *tensor_attention_cuda_backend(void) {
    static const TensorAttentionBackendOps backend = [] {
        TensorAttentionBackendOps value{};
        value.kind = TENSOR_ATTENTION_BACKEND_CUDA;
        value.name = "cuda"; value.is_available = attention_available;
        value.run = attention_f32; value.run_f16 = attention_f16;
        value.decode_sequence_available = decode_sequence_available;
        value.decode_sequence = decode_sequence_f32;
        value.decode_sequence_f16_available = decode_sequence_f16_available;
        value.decode_sequence_f16 = decode_sequence_f16;
        value.decode_cache_clear = clear_decode_cache;
        return value;
    }();
    return &backend;
}


} // extern C
