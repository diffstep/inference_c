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

template<bool Q8> __device__ __forceinline__ float decode_quant_value(
    const uint8_t *block, size_t k) {
    const uint16_t bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
    const float scale = __half2float(*reinterpret_cast<const __half *>(&bits));
    int q;
    if constexpr (Q8) q = (int)(int8_t)block[2 + k];
    else { const uint8_t packed = block[2 + (k & 15)]; q = (int)(k < 16 ? packed & 15 : packed >> 4) - 8; }
    return scale * (float)q;
}

template<bool Q8> __global__ void decode_quant_gemv_kernel(
    const __half *input, const uint8_t *weight, __half *output,
    size_t input_width, size_t output_width) {
    const size_t row = blockIdx.x;
    if (row >= output_width) return;
    const size_t blocks = input_width / 32, block_bytes = Q8 ? 34 : 18;
    const uint8_t *row_weight = weight + row * blocks * block_bytes;
    float sum = 0.0f;
    for (size_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint8_t *block = row_weight + b * block_bytes;
        for (size_t k = 0; k < 32; ++k)
            sum = fmaf(__half2float(input[b * 32 + k]), decode_quant_value<Q8>(block, k), sum);
    }
    __shared__ float partials[128];
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = __float2half(partials[0]);
}

template<bool Q8> __global__ void decode_quant_qkv_kernel(
    const __half *input, const uint8_t *q_weight, const uint8_t *k_weight,
    const uint8_t *v_weight, __half *output, size_t input_width,
    size_t q_width, size_t kv_width) {
    const size_t row = blockIdx.x;
    const size_t weight_row = row < q_width ? row :
        (row < q_width + kv_width ? row - q_width : row - q_width - kv_width);
    const uint8_t *matrix = row < q_width ? q_weight :
        (row < q_width + kv_width ? k_weight : v_weight);
    const size_t blocks = input_width / 32, block_bytes = Q8 ? 34 : 18;
    const uint8_t *row_weight = matrix + weight_row * blocks * block_bytes;
    float sum = 0.0f;
    for (size_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint8_t *block = row_weight + b * block_bytes;
        for (size_t k = 0; k < 32; ++k)
            sum = fmaf(__half2float(input[b * 32 + k]), decode_quant_value<Q8>(block, k), sum);
    }
    __shared__ float partials[128];
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = __float2half(partials[0]);
}

template<bool Q8> __global__ void decode_quant_swiglu_kernel(
    const __half *input, const uint8_t *gate_weight, const uint8_t *up_weight,
    __half *output, size_t input_width, size_t output_width) {
    const size_t row = blockIdx.x;
    if (row >= output_width) return;
    const size_t blocks = input_width / 32, block_bytes = Q8 ? 34 : 18;
    const uint8_t *gate_row = gate_weight + row * blocks * block_bytes;
    const uint8_t *up_row = up_weight + row * blocks * block_bytes;
    float gate_sum = 0.0f, up_sum = 0.0f;
    for (size_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint8_t *gb = gate_row + b * block_bytes;
        const uint8_t *ub = up_row + b * block_bytes;
        for (size_t k = 0; k < 32; ++k) {
            const float x = __half2float(input[b * 32 + k]);
            gate_sum = fmaf(x, decode_quant_value<Q8>(gb, k), gate_sum);
            up_sum = fmaf(x, decode_quant_value<Q8>(ub, k), up_sum);
        }
    }
    __shared__ float gate_partials[128];
    __shared__ float up_partials[128];
    gate_partials[threadIdx.x] = gate_sum;
    up_partials[threadIdx.x] = up_sum;
    __syncthreads();
    for (unsigned stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            gate_partials[threadIdx.x] += gate_partials[threadIdx.x + stride];
            up_partials[threadIdx.x] += up_partials[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float gate = gate_partials[0];
        output[row] = __float2half(gate / (1.0f + expf(-gate)) * up_partials[0]);
    }
}

__global__ void decode_add_residual_kernel(const __half *value,
    const __half *residual, __half *output, size_t width) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) output[i] = __float2half(__half2float(value[i]) + __half2float(residual[i]));
}

__device__ __forceinline__ size_t decode_kv_position(
    size_t position, const uint32_t *block_table, size_t block_size) {
    return block_table == nullptr ? position :
        (size_t)block_table[position / block_size] * block_size + position % block_size;
}

template<typename T> __global__ void decode_rope_cache_kernel(
    const T *qkv, T *rotated_query, T *key_cache, T *value_cache,
    size_t hidden_size, size_t kv_width, size_t kv_heads, size_t head_dim,
    size_t position, float theta, const uint32_t *block_table,
    size_t block_size) {
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
        const size_t cache_index = decode_kv_position(position, block_table,
                                                      block_size) * kv_width + index;
        key_cache[cache_index] = from_float<T>(as_float(key[index]) * cosf(angle) +
            sign * as_float(key[base + (paired - base)]) * sinf(angle));
        value_cache[cache_index] = value[index];
    }
    (void)kv_heads;
}

template<typename T> __global__ void decode_score_kernel(
    const T *query, const T *key_cache, float *scores,
    size_t query_heads, size_t kv_heads, size_t kv_length,
    size_t kv_width, size_t head_dim, float scale,
    const uint32_t *block_table, size_t block_size) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = query_heads * kv_length;
    if (index >= count) return;
    const size_t head = index / kv_length;
    const size_t position = index % kv_length;
    const size_t kv_head = head / (query_heads / kv_heads);
    float sum = 0.0f;
    for (size_t d = 0; d < head_dim; ++d)
        sum = fmaf(as_float(query[head * head_dim + d]),
                   as_float(key_cache[decode_kv_position(position, block_table,
                       block_size) * kv_width + kv_head * head_dim + d]), sum);
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
    size_t kv_width, size_t head_dim,
    const uint32_t *block_table, size_t block_size) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = query_heads * head_dim;
    if (index >= count) return;
    const size_t head = index / head_dim;
    const size_t dimension = index % head_dim;
    const size_t kv_head = head / (query_heads / kv_heads);
    float sum = 0.0f;
    for (size_t position = 0; position < kv_length; ++position)
        sum = fmaf(scores[head * kv_length + position],
            as_float(value_cache[decode_kv_position(position, block_table,
                block_size) * kv_width + kv_head * head_dim + dimension]), sum);
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

bool decode_quant_gemv(const __half *input, const uint8_t *host_weight,
    __half *output, size_t input_width, size_t output_width,
    TensorWeightQuantization format) {
    if (!input || !host_weight || !output || input_width == 0 || input_width % 32 ||
        output_width == 0 || (format != TENSOR_WEIGHT_Q4_0 && format != TENSOR_WEIGHT_Q8_0)) return false;
    const size_t block_bytes = format == TENSOR_WEIGHT_Q8_0 ? 34 : 18;
    if (output_width > SIZE_MAX / (input_width / 32) / block_bytes) return false;
    const size_t bytes = output_width * (input_width / 32) * block_bytes;
    auto *device_weight = static_cast<const uint8_t *>(decode_weight_buffer(host_weight, bytes, false));
    if (!device_weight) return false;
    if (format == TENSOR_WEIGHT_Q8_0)
        decode_quant_gemv_kernel<true><<<(unsigned)output_width, 128>>>(input, device_weight, output, input_width, output_width);
    else
        decode_quant_gemv_kernel<false><<<(unsigned)output_width, 128>>>(input, device_weight, output, input_width, output_width);
    return cudaGetLastError() == cudaSuccess;
}

bool decode_quant_qkv(const __half *input,
    const uint8_t *host_q, const uint8_t *host_k, const uint8_t *host_v,
    __half *output, size_t input_width, size_t q_width, size_t kv_width,
    TensorWeightQuantization format) {
    if (!input || !host_q || !host_k || !host_v || !output || input_width == 0 ||
        input_width % 32 || q_width == 0 || kv_width == 0 ||
        (format != TENSOR_WEIGHT_Q4_0 && format != TENSOR_WEIGHT_Q8_0)) return false;
    const size_t block_bytes = format == TENSOR_WEIGHT_Q8_0 ? 34 : 18;
    const size_t row_bytes = (input_width / 32) * block_bytes;
    if (q_width > SIZE_MAX / row_bytes || kv_width > SIZE_MAX / row_bytes) return false;
    const size_t q_bytes = q_width * row_bytes, kv_bytes = kv_width * row_bytes;
    const auto *q = static_cast<const uint8_t *>(decode_weight_buffer(host_q, q_bytes, false));
    const auto *k = static_cast<const uint8_t *>(decode_weight_buffer(host_k, kv_bytes, false));
    const auto *v = static_cast<const uint8_t *>(decode_weight_buffer(host_v, kv_bytes, false));
    if (!q || !k || !v || kv_width > (SIZE_MAX - q_width) / 2 ||
        q_width + 2 * kv_width > UINT_MAX) return false;
    const unsigned rows = (unsigned)(q_width + 2 * kv_width);
    if (format == TENSOR_WEIGHT_Q8_0)
        decode_quant_qkv_kernel<true><<<rows, 128>>>(input, q, k, v, output,
            input_width, q_width, kv_width);
    else
        decode_quant_qkv_kernel<false><<<rows, 128>>>(input, q, k, v, output,
            input_width, q_width, kv_width);
    return cudaGetLastError() == cudaSuccess;
}

bool decode_quant_swiglu(const __half *input,
    const uint8_t *host_gate, const uint8_t *host_up, __half *output,
    size_t input_width, size_t output_width, TensorWeightQuantization format) {
    if (!input || !host_gate || !host_up || !output || input_width == 0 ||
        input_width % 32 || output_width == 0 ||
        (format != TENSOR_WEIGHT_Q4_0 && format != TENSOR_WEIGHT_Q8_0)) return false;
    const size_t block_bytes = format == TENSOR_WEIGHT_Q8_0 ? 34 : 18;
    if (output_width > SIZE_MAX / (input_width / 32) / block_bytes ||
        output_width > UINT_MAX) return false;
    const size_t bytes = output_width * (input_width / 32) * block_bytes;
    const auto *gate = static_cast<const uint8_t *>(decode_weight_buffer(host_gate, bytes, false));
    const auto *up = static_cast<const uint8_t *>(decode_weight_buffer(host_up, bytes, false));
    if (!gate || !up) return false;
    if (format == TENSOR_WEIGHT_Q8_0)
        decode_quant_swiglu_kernel<true><<<(unsigned)output_width, 128>>>(input,
            gate, up, output, input_width, output_width);
    else
        decode_quant_swiglu_kernel<false><<<(unsigned)output_width, 128>>>(input,
            gate, up, output, input_width, output_width);
    return cudaGetLastError() == cudaSuccess;
}

template<typename T> struct DecodeEntry {
    const void *value_host = nullptr;
    const void *q_host = nullptr, *k_host = nullptr, *v_host = nullptr;
    const void *gate_host = nullptr, *up_host = nullptr;
    size_t capacity = 0, hidden = 0, intermediate = 0, valid_length = 0;
    T *key = nullptr, *value = nullptr, *norm = nullptr, *qkv_weight = nullptr;
    T *qkv = nullptr, *query = nullptr, *attention = nullptr, *gate_up_weight = nullptr;
    T *residual = nullptr, *post_norm = nullptr, *activated = nullptr, *quant_aux = nullptr;
    float *scores = nullptr;
    bool quantized = false;
    bool owns_kv = true;
    uint32_t *block_table = nullptr;
    size_t block_table_capacity = 0;
    const TensorPagedKVCache *paged_cache = nullptr;
    TensorPagedKVSequence paged_sequence = 0;
    size_t layer_index = SIZE_MAX;
    size_t score_capacity = 0;

    void release() {
        if (owns_kv) {
            if (key != nullptr) cudaFree(key);
            if (value != nullptr) cudaFree(value);
        }
        void *buffers[] = {norm, qkv_weight, qkv, query, attention,
            gate_up_weight, residual, post_norm, activated, quant_aux, scores};
        for (void *buffer : buffers) if (buffer != nullptr) cudaFree(buffer);
        if (block_table != nullptr) cudaFree(block_table);
        key = value = norm = qkv_weight = qkv = query = attention = nullptr;
        gate_up_weight = residual = post_norm = activated = quant_aux = nullptr;
        scores = nullptr; block_table = nullptr; block_table_capacity = 0;
        capacity = hidden = intermediate = valid_length = 0;
        value_host = q_host = k_host = v_host = gate_host = up_host = nullptr;
        owns_kv = true; paged_cache = nullptr; paged_sequence = 0;
        layer_index = SIZE_MAX; score_capacity = 0;
        quantized = false;
    }
    ~DecodeEntry() { release(); }
};

struct CudaPagedKVStateBase {
    virtual ~CudaPagedKVStateBase() = default;
    TensorPagedKVCache *owner = nullptr;
    int storage = -1;
    size_t block_size = 0, block_count = 0, layer_count = 0, kv_width = 0;
};

template<typename T> struct CudaPagedKVState final : CudaPagedKVStateBase {
    std::vector<T *> keys, values;
    ~CudaPagedKVState() override {
        for (T *pointer : keys) if (pointer != nullptr) cudaFree(pointer);
        for (T *pointer : values) if (pointer != nullptr) cudaFree(pointer);
    }
};

void clear_decode_cache();
void release_cuda_paged_kv_sequence(void *state,
                                    TensorPagedKVSequence sequence);
void destroy_cuda_paged_kv_state(void *state) {
    clear_decode_cache();
    delete static_cast<CudaPagedKVStateBase *>(state);
}

template<typename T> CudaPagedKVState<T> *get_cuda_paged_kv_state(
    TensorPagedKVCache *cache, int storage, size_t layer_count,
    size_t kv_width) {
    auto *base = static_cast<CudaPagedKVStateBase *>(
        tensor_paged_kv_cache_backend_state(cache, TENSOR_PAGED_KV_BACKEND_CUDA));
    const size_t block_size = tensor_paged_kv_cache_block_size(cache);
    const size_t block_count = tensor_paged_kv_cache_block_count(cache);
    if (base != nullptr) {
        if (base->storage != storage || base->block_size != block_size ||
            base->block_count != block_count || base->layer_count != layer_count ||
            base->kv_width != kv_width) return nullptr;
        return dynamic_cast<CudaPagedKVState<T> *>(base);
    }
    if (block_count == 0 || block_size == 0 ||
        block_count > SIZE_MAX / block_size ||
        block_count * block_size > SIZE_MAX / kv_width) return nullptr;
    std::unique_ptr<CudaPagedKVState<T>> created(new (std::nothrow) CudaPagedKVState<T>());
    if (!created) return nullptr;
    created->storage = storage;
    created->owner = cache;
    created->block_size = block_size;
    created->block_count = block_count;
    created->layer_count = layer_count;
    created->kv_width = kv_width;
    try { created->keys.resize(layer_count, nullptr); created->values.resize(layer_count, nullptr); }
    catch (...) { return nullptr; }
    const size_t elements = block_count * block_size * kv_width;
    for (size_t i = 0; i < layer_count; ++i)
        if (!alloc_device(elements, &created->keys[i]) ||
            !alloc_device(elements, &created->values[i])) return nullptr;
    CudaPagedKVState<T> *raw = created.get();
    if (!tensor_paged_kv_cache_set_backend_state(cache,
                                                  TENSOR_PAGED_KV_BACKEND_CUDA, raw,
                                                  destroy_cuda_paged_kv_state,
                                                  release_cuda_paged_kv_sequence)) {
        auto *existing = static_cast<CudaPagedKVStateBase *>(
            tensor_paged_kv_cache_backend_state(cache, TENSOR_PAGED_KV_BACKEND_CUDA));
        return existing != nullptr && existing->storage == storage ?
            dynamic_cast<CudaPagedKVState<T> *>(existing) : nullptr;
    }
    created.release();
    return raw;
}

template<typename T> std::unordered_map<const void *, std::unique_ptr<DecodeEntry<T>>> &decode_entries() {
    static std::unordered_map<const void *, std::unique_ptr<DecodeEntry<T>>> entries;
    return entries;
}

template<typename T> void erase_paged_sequence_entries(
    std::unordered_map<const void *, std::unique_ptr<DecodeEntry<T>>> &entries,
    const TensorPagedKVCache *cache, TensorPagedKVSequence sequence) {
    for (auto it = entries.begin(); it != entries.end();) {
        const DecodeEntry<T> &entry = *it->second;
        if (entry.paged_cache == cache && entry.paged_sequence == sequence)
            it = entries.erase(it);
        else
            ++it;
    }
}

void release_cuda_paged_kv_sequence(void *opaque,
                                    TensorPagedKVSequence sequence) {
    auto *state = static_cast<CudaPagedKVStateBase *>(opaque);
    if (state == nullptr) return;
    std::lock_guard<std::mutex> lock(decode_mutex);
    erase_paged_sequence_entries(decode_entries<float>(), state->owner, sequence);
    erase_paged_sequence_entries(decode_entries<__half>(), state->owner, sequence);
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
    size_t capacity, size_t kv_width, size_t query_heads, bool quantized,
    bool allocate_kv, size_t score_capacity) {
    if (entry.norm != nullptr && entry.value_host == value_host && entry.quantized == quantized &&
        entry.hidden == hidden && entry.intermediate == intermediate &&
        entry.capacity == capacity && entry.owns_kv == allocate_kv) {
        if (score_capacity <= entry.score_capacity) return true;
        if (query_heads > SIZE_MAX / score_capacity) return false;
        if (entry.scores != nullptr) cudaFree(entry.scores);
        entry.scores = nullptr;
        if (!alloc_device(query_heads * score_capacity, &entry.scores)) return false;
        entry.score_capacity = score_capacity;
        return true;
    }
    entry.release();
    const size_t qkv_width = hidden + 2 * kv_width;
    if (capacity > SIZE_MAX / kv_width || qkv_width > SIZE_MAX / hidden ||
        intermediate > SIZE_MAX / hidden / 2 || score_capacity == 0 ||
        query_heads > SIZE_MAX / score_capacity)
        return false;
    bool ok = (!allocate_kv || (alloc_device(capacity * kv_width, &entry.key) &&
        alloc_device(capacity * kv_width, &entry.value))) &&
        alloc_device(hidden, &entry.norm) &&
        (quantized || alloc_device(qkv_width * hidden, &entry.qkv_weight)) &&
        alloc_device(qkv_width, &entry.qkv) &&
        alloc_device(hidden, &entry.query) &&
        alloc_device(hidden, &entry.attention) &&
        (quantized || alloc_device(2 * intermediate * hidden, &entry.gate_up_weight)) &&
        alloc_device(hidden, &entry.residual) &&
        alloc_device(hidden, &entry.post_norm) &&
        alloc_device(intermediate, &entry.activated) &&
        alloc_device(std::max(hidden, intermediate), &entry.quant_aux) &&
        alloc_device(query_heads * score_capacity, &entry.scores);
    if (!ok) { entry.release(); return false; }
    entry.value_host = value_host;
    entry.quantized = quantized;
    entry.hidden = hidden;
    entry.intermediate = intermediate;
    entry.capacity = capacity;
    entry.valid_length = 0;
    entry.owns_kv = allocate_kv;
    entry.score_capacity = score_capacity;
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
    const T *final_norm, const void *lm_head, size_t vocabulary_size,
    TensorWeightQuantization quant_format = (TensorWeightQuantization)0,
    const TensorPagedKVDecodeRequest *paged = nullptr) {
    constexpr bool quantized = std::is_same<Layer, TensorAttentionDecoderLayerQuantF16>::value;
    const bool use_paged = paged != nullptr;
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
        vocabulary_size > SIZE_MAX / hidden_size / sizeof(T) ||
        (quantized && (hidden_size % 32 || intermediate_size % 32 ||
            (kv_heads * head_dim) % 32 ||
            (quant_format != TENSOR_WEIGHT_Q4_0 && quant_format != TENSOR_WEIGHT_Q8_0)))) {
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

    CudaPagedKVState<T> *paged_state = nullptr;
    std::vector<uint32_t> host_block_table;
    const size_t page_size = use_paged ? tensor_paged_kv_cache_block_size(paged->cache) : 0;
    const size_t pool_blocks = use_paged ? tensor_paged_kv_cache_block_count(paged->cache) : 0;
    const size_t target_length = past_length + token_count;
    if (use_paged) {
        const int expected_storage = quantized ? TENSOR_PAGED_KV_QUANT_F16 :
            (sizeof(T) == sizeof(float) ? TENSOR_PAGED_KV_F32 : TENSOR_PAGED_KV_F16);
        if (paged->cache == nullptr || paged->sequence == 0 ||
            paged->storage != expected_storage || page_size == 0 || pool_blocks == 0 ||
            pool_blocks > SIZE_MAX / page_size ||
            pool_blocks * page_size > UINT32_MAX ||
            pool_blocks * page_size > SIZE_MAX / kv_width ||
            cache_capacity != pool_blocks * page_size) return 0;
    }

    std::lock_guard<std::mutex> lock(decode_mutex);
    if (use_paged) {
        paged_state = get_cuda_paged_kv_state<T>(paged->cache,
            (int)paged->storage, layer_count, kv_width);
        if (paged_state == nullptr) return 0;
        size_t table_count = 0;
        if (!tensor_paged_kv_sequence_copy_block_table(paged->cache,
                paged->sequence, nullptr, 0, &table_count) ||
            table_count * page_size < target_length) return 0;
        try { host_block_table.resize(table_count); } catch (...) { return 0; }
        if (!tensor_paged_kv_sequence_copy_block_table(paged->cache,
                paged->sequence, host_block_table.data(), table_count,
                &table_count)) return 0;
    }
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
        DecodeEntry<T> *entry_pointer = nullptr;
        if (use_paged) {
            for (auto &candidate : entries) {
                if (candidate.second->paged_cache == paged->cache &&
                    candidate.second->paged_sequence == paged->sequence &&
                    candidate.second->layer_index == index) {
                    entry_pointer = candidate.second.get();
                    break;
                }
            }
            if (entry_pointer == nullptr) {
                try {
                    std::unique_ptr<DecodeEntry<T>> created(new DecodeEntry<T>());
                    entry_pointer = created.get();
                    entries.emplace(entry_pointer, std::move(created));
                } catch (...) { return 0; }
            }
        } else {
            auto &slot = entries[key_cache[index]];
            if (!slot) {
                try { slot = std::make_unique<DecodeEntry<T>>(); }
                catch (...) { return 0; }
            }
            entry_pointer = slot.get();
        }
        DecodeEntry<T> &entry = *entry_pointer;
        if (entry.norm != nullptr && (entry.value_host != value_cache[index] ||
            entry.capacity != cache_capacity || entry.hidden != hidden_size ||
            entry.intermediate != intermediate_size || entry.quantized != quantized ||
            entry.paged_cache != (use_paged ? paged->cache : nullptr) ||
            entry.paged_sequence != (use_paged ? paged->sequence : 0))) entry.release();
        const bool new_buffers = entry.norm == nullptr;
        if (!prepare_decode_entry(entry, value_cache[index], hidden_size,
                                  intermediate_size, cache_capacity, kv_width,
                                  query_heads, quantized, !use_paged,
                                  use_paged ? target_length : cache_capacity)) {
            fprintf(stderr, "cuda decode: layer %zu buffer allocation failed\n", index);
            return 0;
        }
        if (use_paged) {
            entry.key = paged_state->keys[index];
            entry.value = paged_state->values[index];
            entry.owns_kv = false;
            if (host_block_table.size() > entry.block_table_capacity) {
                if (entry.block_table != nullptr) cudaFree(entry.block_table);
                entry.block_table = nullptr;
                if (!alloc_device(host_block_table.size(), &entry.block_table)) return 0;
                entry.block_table_capacity = host_block_table.size();
            }
            if (!host_block_table.empty() && cudaMemcpy(entry.block_table,
                    host_block_table.data(), host_block_table.size() * sizeof(uint32_t),
                    cudaMemcpyHostToDevice) != cudaSuccess) return 0;
            entry.paged_cache = paged->cache;
            entry.paged_sequence = paged->sequence;
            entry.layer_index = index;
        }
        if (new_buffers || (use_paged && entry.valid_length == 0 && past_length > 0)) {
            if (past_length > 0) {
                if (use_paged) {
                    size_t position = 0;
                    while (position < past_length) {
                        const size_t logical_block = position / page_size;
                        const size_t page_offset = position % page_size;
                        const size_t count = std::min(page_size - page_offset,
                                                      past_length - position);
                        const size_t physical = host_block_table[logical_block];
                        const size_t destination =
                            (physical * page_size + page_offset) * kv_width;
                        const size_t source = position * kv_width;
                        const size_t bytes = count * kv_width * sizeof(T);
                        if (cudaMemcpy(entry.key + destination, key_cache[index] + source,
                                       bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
                            cudaMemcpy(entry.value + destination,
                                       value_cache[index] + source, bytes,
                                       cudaMemcpyHostToDevice) != cudaSuccess) return 0;
                        position += count;
                    }
                } else {
                    const size_t prefix_bytes = past_length * kv_width * sizeof(T);
                    if (cudaMemcpy(entry.key, key_cache[index], prefix_bytes,
                                   cudaMemcpyHostToDevice) != cudaSuccess ||
                        cudaMemcpy(entry.value, value_cache[index], prefix_bytes,
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
                        fprintf(stderr, "cuda decode: layer %zu KV upload failed\n", index);
                        return 0;
                    }
                }
            }
        } else if (entry.valid_length != past_length) {
            fprintf(stderr, "cuda decode: layer %zu cache length mismatch (%zu != %zu)\n",
                    index, entry.valid_length, past_length);
            return 0;
        }
        if constexpr (!quantized) if (!prepare_packed_weights(entry, layers[index], hidden_size,
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
    T *device_lm_head = nullptr;
    if constexpr (quantized) {
        const size_t block_bytes = quant_format == TENSOR_WEIGHT_Q8_0 ? 34 : 18;
        if (vocabulary_size > SIZE_MAX / (hidden_size / 32) / block_bytes) return 0;
        device_lm_head = static_cast<T *>(decode_weight_buffer(lm_head,
            vocabulary_size * (hidden_size / 32) * block_bytes, false));
    } else {
        device_lm_head = static_cast<T *>(decode_weight_buffer(
            lm_head, vocabulary_size * hidden_size * sizeof(T), fp16));
    }
    if (device_embedding == nullptr || device_final_norm == nullptr || device_lm_head == nullptr) {
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
            if constexpr (quantized) {
                const auto &ql = layers[layer_index];
                if (!decode_quant_qkv(entry.norm, ql.q_weight, ql.k_weight,
                    ql.v_weight, entry.qkv, hidden_size, hidden_size, kv_width,
                    quant_format)) return 0;
            } else {
                decode_gemv_kernel<<<(unsigned)qkv_width, 128>>>(entry.norm,
                    entry.qkv_weight, entry.qkv, hidden_size, qkv_width);
            }
            const unsigned rope_count = (unsigned)((std::max(hidden_size, kv_width) + 255) / 256);
            decode_rope_cache_kernel<<<rope_count, 256>>>(entry.qkv, entry.query,
                entry.key, entry.value, hidden_size, kv_width, kv_heads, head_dim,
                position, (float)rope_theta, use_paged ? entry.block_table : nullptr,
                use_paged ? page_size : 0);
            const size_t kv_length = position + 1;
            decode_score_kernel<<<(unsigned)((query_heads * kv_length + 127) / 128), 128>>>(
                entry.query, entry.key, entry.scores, query_heads, kv_heads,
                kv_length, kv_width, head_dim, 1.0f / sqrtf((float)head_dim),
                use_paged ? entry.block_table : nullptr, use_paged ? page_size : 0);
            decode_softmax_kernel<<<(unsigned)query_heads, 1>>>(
                entry.scores, query_heads, kv_length);
            decode_attention_output_kernel<<<(unsigned)((hidden_size + 255) / 256), 256>>>(
                entry.scores, entry.value, entry.attention, query_heads, kv_heads,
                kv_length, kv_width, head_dim,
                use_paged ? entry.block_table : nullptr, use_paged ? page_size : 0);
            if constexpr (quantized) {
                if (!decode_quant_gemv(entry.attention,
                    layers[layer_index].o_weight, entry.quant_aux,
                    hidden_size, hidden_size, quant_format)) return 0;
                decode_add_residual_kernel<<<(unsigned)((hidden_size + 255) / 256), 256>>>(
                    entry.quant_aux, current_hidden, entry.residual, hidden_size);
            } else {
                T *device_o_weight = static_cast<T *>(decode_weight_buffer(
                    layer.o_weight, hidden_size * hidden_size * sizeof(T), fp16));
                if (device_o_weight == nullptr) return 0;
                decode_gemv_residual_kernel<<<(unsigned)hidden_size, 128>>>(
                    entry.attention, device_o_weight, current_hidden, entry.residual,
                    hidden_size, hidden_size);
            }
            const T *post_norm = static_cast<T *>(decode_weight_buffer(
                layer.post_norm, hidden_size * sizeof(T), fp16));
            if (post_norm == nullptr) {
                fprintf(stderr, "cuda decode: layer %zu post norm upload failed\n", layer_index);
                return 0;
            }
            decode_rms_kernel<<<1, 128>>>(entry.residual, post_norm,
                entry.post_norm, hidden_size, rms_norm_epsilon);
            if constexpr (quantized) {
                if (!decode_quant_swiglu(entry.post_norm, layers[layer_index].gate_weight,
                        layers[layer_index].up_weight, entry.activated, hidden_size,
                        intermediate_size, quant_format)) return 0;
                if (!decode_quant_gemv(entry.activated, layers[layer_index].down_weight,
                        entry.quant_aux, intermediate_size, hidden_size, quant_format)) return 0;
                decode_add_residual_kernel<<<(unsigned)((hidden_size + 255) / 256), 256>>>(
                    entry.quant_aux, entry.residual, next_hidden, hidden_size);
            } else {
                decode_gemv_silu_kernel<<<(unsigned)intermediate_size, 128>>>(
                    entry.post_norm, entry.gate_up_weight, entry.activated,
                    hidden_size, intermediate_size);
                T *device_down_weight = static_cast<T *>(decode_weight_buffer(
                    layer.down_weight, hidden_size * intermediate_size * sizeof(T), fp16));
                if (device_down_weight == nullptr) return 0;
                decode_gemv_residual_kernel<<<(unsigned)hidden_size, 128>>>(
                    entry.activated, device_down_weight, entry.residual, next_hidden,
                    intermediate_size, hidden_size);
            }
            T *swap = current_hidden; current_hidden = next_hidden; next_hidden = swap;
        }
        decode_rms_kernel<<<1, 128>>>(current_hidden, device_final_norm,
            scratch.norm, hidden_size, rms_norm_epsilon);
        if constexpr (quantized) {
            if (!decode_quant_gemv(scratch.norm, static_cast<const uint8_t *>(lm_head),
                    scratch.logits, hidden_size, vocabulary_size, quant_format)) return 0;
        } else {
            decode_gemv_kernel<<<(unsigned)vocabulary_size, 128>>>(scratch.norm,
                device_lm_head, scratch.logits, hidden_size, vocabulary_size);
        }
        decode_argmax_kernel<<<1, 1>>>(scratch.logits, scratch.tokens + token_index,
            vocabulary_size);
    }
    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        fprintf(stderr, "cuda decode: kernel launch failed: %s\n", cudaGetErrorString(last_error));
        return 0;
    }
    /* The blocking D2H copy orders the default stream and reports execution
       errors, so a separate device-wide synchronization here is redundant. */
    const cudaError_t token_copy_status = cudaMemcpy(
        generated_tokens, scratch.tokens, token_count * sizeof(uint32_t),
        cudaMemcpyDeviceToHost);
    if (token_copy_status != cudaSuccess) {
        fprintf(stderr, "cuda decode: generated token readback failed: %s\n",
                cudaGetErrorString(token_copy_status));
        return 0;
    }
    for (size_t index = 0; index < layer_count; ++index) {
        DecodeEntry<T> &entry = *layer_entries[index];
        /* KV state remains in the persistent device entry. The public cache
           arguments are read-only inputs, so copying the full cache back to
           host here is both unnecessary and contrary to the API contract. */
        entry.valid_length = past_length + token_count;
    }
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
static int decode_sequence_quant_f16_available(TensorWeightQuantization format) {
    return cuda_ready() && (format == TENSOR_WEIGHT_Q4_0 || format == TENSOR_WEIGHT_Q8_0);
}
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
static int decode_sequence_quant_f16(const TensorAttentionDecoderLayerQuantF16 *layers,
    size_t layer_count, const uint16_t *embedding, uint32_t initial_token,
    size_t token_count, uint32_t *generated_tokens,
    const uint16_t *const *key_cache, const uint16_t *const *value_cache,
    size_t past_length, size_t cache_capacity, size_t hidden_size,
    size_t intermediate_size, size_t query_heads, size_t kv_heads,
    size_t head_dim, float rms_norm_epsilon, double rope_theta,
    const uint16_t *final_norm, const uint8_t *lm_head, size_t vocabulary_size,
    TensorWeightQuantization format) {
    return decode_sequence_impl(layers, layer_count,
        reinterpret_cast<const __half *>(embedding), initial_token, token_count,
        generated_tokens, reinterpret_cast<const __half *const *>(key_cache),
        reinterpret_cast<const __half *const *>(value_cache), past_length,
        cache_capacity, hidden_size, intermediate_size, query_heads, kv_heads,
        head_dim, rms_norm_epsilon, rope_theta,
        reinterpret_cast<const __half *>(final_norm), lm_head, vocabulary_size, format);
}
static int decode_sequence_paged(const TensorPagedKVDecodeRequest *request) {
    if (request == nullptr) return 0;
    const size_t block_size = tensor_paged_kv_cache_block_size(request->cache);
    const size_t block_count = tensor_paged_kv_cache_block_count(request->cache);
    if (block_size == 0 || block_count == 0 ||
        block_count > SIZE_MAX / block_size) return 0;
    const size_t capacity = block_size * block_count;
    if (request->storage == TENSOR_PAGED_KV_F32)
        return decode_sequence_impl(
            static_cast<const TensorAttentionDecoderLayer *>(request->layers),
            request->layer_count, static_cast<const float *>(request->embedding),
            request->initial_token, request->token_count, request->generated_tokens,
            reinterpret_cast<const float *const *>(request->key_cache),
            reinterpret_cast<const float *const *>(request->value_cache),
            request->past_length, capacity, request->hidden_size,
            request->intermediate_size, request->query_heads, request->kv_heads,
            request->head_dim, request->rms_norm_epsilon, request->rope_theta,
            static_cast<const float *>(request->final_norm), request->lm_head,
            request->vocabulary_size, (TensorWeightQuantization)0, request);
    if (request->storage == TENSOR_PAGED_KV_F16)
        return decode_sequence_impl(
            static_cast<const TensorAttentionDecoderLayerF16 *>(request->layers),
            request->layer_count,
            reinterpret_cast<const __half *>(request->embedding),
            request->initial_token, request->token_count, request->generated_tokens,
            reinterpret_cast<const __half *const *>(request->key_cache),
            reinterpret_cast<const __half *const *>(request->value_cache),
            request->past_length, capacity, request->hidden_size,
            request->intermediate_size, request->query_heads, request->kv_heads,
            request->head_dim, request->rms_norm_epsilon, request->rope_theta,
            reinterpret_cast<const __half *>(request->final_norm), request->lm_head,
            request->vocabulary_size, (TensorWeightQuantization)0, request);
    if (request->storage == TENSOR_PAGED_KV_QUANT_F16)
        return decode_sequence_impl(
            static_cast<const TensorAttentionDecoderLayerQuantF16 *>(request->layers),
            request->layer_count,
            reinterpret_cast<const __half *>(request->embedding),
            request->initial_token, request->token_count, request->generated_tokens,
            reinterpret_cast<const __half *const *>(request->key_cache),
            reinterpret_cast<const __half *const *>(request->value_cache),
            request->past_length, capacity, request->hidden_size,
            request->intermediate_size, request->query_heads, request->kv_heads,
            request->head_dim, request->rms_norm_epsilon, request->rope_theta,
            reinterpret_cast<const __half *>(request->final_norm), request->lm_head,
            request->vocabulary_size, request->weight_quantization, request);
    return 0;
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
        value.decode_sequence_quant_f16_available = decode_sequence_quant_f16_available;
        value.decode_sequence_quant_f16 = decode_sequence_quant_f16;
        value.decode_sequence_paged = decode_sequence_paged;
        value.decode_cache_clear = clear_decode_cache;
        return value;
    }();
    return &backend;
}


} // extern C
