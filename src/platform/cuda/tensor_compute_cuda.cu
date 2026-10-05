#include "tensor_compute_backend.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "cuda_common.cuh"

#include <cmath>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <type_traits>
#include <initializer_list>
#include <unordered_map>

using namespace smolvlm_cuda;

namespace {

struct CachedWeight { void *device; size_t bytes; };
std::unordered_map<const void *, CachedWeight> weight_cache;
std::mutex cache_mutex;
cublasHandle_t blas_handle = nullptr;
const float one_f = 1.0f, zero_f = 0.0f;


bool blas_ready() {
    if (blas_handle != nullptr) return true;
    return cublasCreate(&blas_handle) == CUBLAS_STATUS_SUCCESS;
}

void *weight_buffer(const void *host, size_t bytes) {
    if (host == nullptr || bytes == 0) return nullptr;
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = weight_cache.find(host);
    if (found != weight_cache.end())
        return found->second.bytes == bytes ? found->second.device : nullptr;
    void *device = nullptr;
    if (cudaMalloc(&device, bytes) != cudaSuccess ||
        cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        if (device != nullptr) cudaFree(device);
        return nullptr;
    }
    weight_cache.emplace(host, CachedWeight{device, bytes});
    return device;
}


__global__ void embedding_f32_kernel(const float *table, const uint32_t *ids,
                                     float *out, size_t rows, size_t width,
                                     size_t table_rows) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = rows * width;
    if (i < n) { size_t id = ids[i / width]; out[i] = id < table_rows ? table[id * width + i % width] : 0.0f; }
}

__global__ void embedding_f16_kernel(const __half *table, const uint32_t *ids,
                                     __half *out, size_t rows, size_t width,
                                     size_t table_rows) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = rows * width;
    if (i < n) { size_t id = ids[i / width]; out[i] = id < table_rows ? table[id * width + i % width] : __float2half(0.0f); }
}

__global__ void norm_f32_kernel(const float *x, const float *scale, const float *bias,
                                float *y, size_t rows, size_t width, float eps,
                                bool layer_norm) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float *src = x + row * width;
    float mean = 0.0f;
    if (layer_norm) { for (size_t i = 0; i < width; ++i) mean += src[i]; mean /= (float)width; }
    float sum = 0.0f;
    for (size_t i = 0; i < width; ++i) { float v = src[i] - mean; sum += v * v; }
    float factor = rsqrtf(sum / (float)width + eps);
    for (size_t i = 0; i < width; ++i)
        y[row * width + i] = (src[i] - mean) * factor * scale[i] + (bias == nullptr ? 0.0f : bias[i]);
}

__global__ void norm_f16_kernel(const __half *x, const __half *scale,
                                const __half *bias, __half *y, size_t rows,
                                size_t width, float eps, bool layer_norm) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const __half *src = x + row * width;
    float mean = 0.0f;
    if (layer_norm) { for (size_t i = 0; i < width; ++i) mean += __half2float(src[i]); mean /= (float)width; }
    float sum = 0.0f;
    for (size_t i = 0; i < width; ++i) { float v = __half2float(src[i]) - mean; sum += v * v; }
    float factor = rsqrtf(sum / (float)width + eps);
    for (size_t i = 0; i < width; ++i) {
        float v = (__half2float(src[i]) - mean) * factor * __half2float(scale[i]);
        if (bias != nullptr) v += __half2float(bias[i]);
        y[row * width + i] = __float2half_rn(v);
    }
}

template<typename T> __global__ void vision_score_kernel(
    const T *query, const T *key, float *scores, size_t rows,
    size_t heads, size_t head_dim) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = heads * rows * rows;
    if (index >= count) return;
    const size_t key_pos = index % rows;
    const size_t query_pos = (index / rows) % rows;
    const size_t head = index / (rows * rows);
    float sum = 0.0f;
    const size_t qbase = query_pos * heads * head_dim + head * head_dim;
    const size_t kbase = key_pos * heads * head_dim + head * head_dim;
    for (size_t d = 0; d < head_dim; d++)
        sum = fmaf(as_float(query[qbase + d]), as_float(key[kbase + d]), sum);
    scores[index] = sum * rsqrtf((float)head_dim);
}

__global__ void vision_softmax_kernel(float *scores, size_t rows, size_t heads) {
    __shared__ float partial[128];
    const size_t row = blockIdx.x;
    if (row >= rows * heads) return;
    const unsigned lane = threadIdx.x;
    float maximum = -3.402823466e+38F;
    for (size_t i = lane; i < rows; i += 128)
        maximum = fmaxf(maximum, scores[row * rows + i]);
    partial[lane] = maximum;
    __syncthreads();
    for (unsigned stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] = fmaxf(partial[lane], partial[lane + stride]);
        __syncthreads();
    }
    maximum = partial[0];
    float sum = 0.0f;
    for (size_t i = lane; i < rows; i += 128) {
        const float value = expf(scores[row * rows + i] - maximum);
        scores[row * rows + i] = value;
        sum += value;
    }
    partial[lane] = sum;
    __syncthreads();
    for (unsigned stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float inverse = partial[0] > 0.0f ? 1.0f / partial[0] : 0.0f;
    for (size_t i = lane; i < rows; i += 128)
        scores[row * rows + i] *= inverse;
}

template<typename T> __global__ void vision_attention_kernel(
    const float *scores, const T *value, T *output,
    size_t rows, size_t heads, size_t head_dim) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t hidden = heads * head_dim;
    if (index >= rows * hidden) return;
    const size_t dimension = index % head_dim;
    const size_t head = (index / head_dim) % heads;
    const size_t query_pos = index / hidden;
    float sum = 0.0f;
    const float *probabilities = scores + (head * rows + query_pos) * rows;
    for (size_t key_pos = 0; key_pos < rows; key_pos++)
        sum = fmaf(probabilities[key_pos],
            as_float(value[key_pos * hidden + head * head_dim + dimension]), sum);
    output[index] = from_float<T>(sum);
}

template<typename T> __global__ void residual_device_kernel(
    const T *left, const T *right, T *output, size_t count) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = from_float<T>(as_float(left[index]) + as_float(right[index]));
}

template<typename T> bool linear_device(const T *input, const void *weight,
    T *output, size_t rows, size_t input_width, size_t output_width) {
    if (!blas_ready() || input == nullptr || weight == nullptr || output == nullptr ||
        rows == 0 || input_width == 0 || output_width == 0 || rows > INT_MAX ||
        input_width > INT_MAX || output_width > INT_MAX) return false;
    void *device_weight = weight_buffer(weight, output_width * input_width * sizeof(T));
    if (device_weight == nullptr) return false;
    if constexpr (std::is_same<T, float>::value) {
        return cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)output_width, (int)rows, (int)input_width, &one_f,
            (const float *)device_weight, (int)input_width, input, (int)input_width,
            &zero_f, output, (int)output_width) == CUBLAS_STATUS_SUCCESS;
    } else {
        const float alpha = 1.0f, beta = 0.0f;
        return cublasGemmEx(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)output_width, (int)rows, (int)input_width, &alpha,
            device_weight, CUDA_R_16F, (int)input_width, input, CUDA_R_16F,
            (int)input_width, &beta, output, CUDA_R_16F, (int)output_width,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
    }
}

template<typename T> bool linear_resident(const T *input, const T *weight,
    T *output, size_t rows, size_t input_width, size_t output_width) {
    if (!cuda_ready() || !blas_ready() || input == nullptr || weight == nullptr ||
        output == nullptr || rows == 0 || input_width == 0 || output_width == 0 ||
        rows > INT_MAX || input_width > INT_MAX || output_width > INT_MAX) return false;
    if constexpr (std::is_same<T, float>::value) {
        return cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)output_width, (int)rows, (int)input_width, &one_f,
            weight, (int)input_width, input, (int)input_width,
            &zero_f, output, (int)output_width) == CUBLAS_STATUS_SUCCESS;
    } else {
        const float alpha = 1.0f, beta = 0.0f;
        return cublasGemmEx(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)output_width, (int)rows, (int)input_width, &alpha,
            weight, CUDA_R_16F, (int)input_width, input, CUDA_R_16F,
            (int)input_width, &beta, output, CUDA_R_16F, (int)output_width,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
    }
}

template<bool Q8> __global__ void quant_linear_f16_kernel(const __half *x,
        const uint8_t *w, __half *y, size_t rows, size_t input_width,
        size_t output_width) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * output_width) return;
    const size_t row = index / output_width, out = index % output_width;
    const size_t blocks = input_width / 32;
    const size_t block_bytes = Q8 ? 34 : 18;
    const uint8_t *base = w + out * blocks * block_bytes;
    float sum = 0.0f;
    for (size_t b = 0; b < blocks; ++b) {
        const uint8_t *block = base + b * block_bytes;
        const uint16_t bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
        const float scale = __half2float(*reinterpret_cast<const __half *>(&bits));
        for (size_t k = 0; k < 32; ++k) {
            int q;
            if constexpr (Q8) q = (int)(int8_t)block[2 + k];
            else { const uint8_t packed = block[2 + (k & 15)]; q = (int)(k < 16 ? packed & 15 : packed >> 4) - 8; }
            sum = fmaf(__half2float(x[row * input_width + b * 32 + k]), scale * (float)q, sum);
        }
    }
    y[index] = __float2half(sum);
}

static bool quant_linear_f16_device(const void *x, const void *w, void *y,
        size_t rows, size_t input_width, size_t output_width, bool q8) {
    if (!cuda_ready() || !x || !w || !y || !rows || !input_width || !output_width ||
        input_width % 32 || rows > SIZE_MAX / output_width) return false;
    const size_t count = rows * output_width;
    if (q8) quant_linear_f16_kernel<true><<<(unsigned)((count + 127) / 128),128>>>(
        (const __half *)x, (const uint8_t *)w, (__half *)y, rows, input_width, output_width);
    else quant_linear_f16_kernel<false><<<(unsigned)((count + 127) / 128),128>>>(
        (const __half *)x, (const uint8_t *)w, (__half *)y, rows, input_width, output_width);
    return cudaGetLastError() == cudaSuccess;
}

static bool quant_linear_f16_resident(const __half *x, const uint8_t *host_weight,
        __half *y, size_t rows, size_t input_width, size_t output_width,
        TensorWeightQuantization format) {
    if (format != TENSOR_WEIGHT_Q8_0 && format != TENSOR_WEIGHT_Q4_0) return false;
    if (input_width == 0 || input_width % 32 || output_width == 0 ||
        output_width > SIZE_MAX / (input_width / 32) /
            (format == TENSOR_WEIGHT_Q8_0 ? 34u : 18u)) return false;
    const size_t bytes = output_width * (input_width / 32) *
        (format == TENSOR_WEIGHT_Q8_0 ? 34u : 18u);
    void *device_weight = weight_buffer(host_weight, bytes);
    return device_weight != nullptr && quant_linear_f16_device(
        x, device_weight, y, rows, input_width, output_width,
        format == TENSOR_WEIGHT_Q8_0);
}

__global__ void bias_f32_kernel(const float *, const float *, float *, size_t, size_t, bool);
__global__ void bias_f16_kernel(const __half *, const __half *, __half *, size_t, size_t, bool);

template<typename T> bool bias_resident(const T *input, const T *bias,
    T *output, size_t rows, size_t width) {
    if (input == nullptr || bias == nullptr || output == nullptr || rows == 0 ||
        width == 0 || rows > SIZE_MAX / width) return false;
    const size_t count = rows * width;
    if constexpr (std::is_same<T, float>::value)
        bias_f32_kernel<<<(unsigned)((count + 255) / 256), 256>>>(input, bias,
            output, count, width, false);
    else
        bias_f16_kernel<<<(unsigned)((count + 255) / 256), 256>>>(input, bias,
            output, count, width, false);
    return cudaGetLastError() == cudaSuccess;
}

template<typename T> bool add_bias_device(const T *input, const void *bias,
    T *output, size_t rows, size_t width, bool gelu) {
    void *device_bias = weight_buffer(bias, width * sizeof(T));
    if (device_bias == nullptr) return false;
    const size_t count = rows * width;
    if constexpr (std::is_same<T, float>::value)
        bias_f32_kernel<<<(unsigned)((count + 255) / 256), 256>>>(input,
            (const float *)device_bias, output, count, width, gelu);
    else
        bias_f16_kernel<<<(unsigned)((count + 255) / 256), 256>>>(input,
            (const __half *)device_bias, output, count, width, gelu);
    return cudaGetLastError() == cudaSuccess;
}

__global__ void unary_f32_kernel(const float *x, float *y, size_t n, int op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (op == 0) {
        if (v >= 5.0f) y[i] = v;
        else if (v <= -5.0f) y[i] = 0.0f;
        else y[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
    else y[i] = v;
}

__global__ void unary_f16_kernel(const __half *x, __half *y, size_t n, int op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    if (op == 0) {
        if (v >= 5.0f) y[i] = x[i];
        else if (v <= -5.0f) y[i] = __float2half(0.0f);
        else y[i] = __float2half_rn(0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))));
    } else y[i] = x[i];
}

__global__ void gelu_multiply_f16_kernel(const __half *x, const __half *gate,
                                         __half *y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(x[i]);
    float activated;
    if (v >= 5.0f) activated = v;
    else if (v <= -5.0f) activated = 0.0f;
    else activated = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    y[i] = __float2half_rn(activated * __half2float(gate[i]));
}

__global__ void relu_f16_kernel(const __half *x, __half *y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(fmaxf(__half2float(x[i]), 0.0f));
}

__global__ void bias_f32_kernel(const float *x, const float *bias, float *y,
                                size_t n, size_t width, bool gelu) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] + bias[i % width];
        if (gelu && v >= 5.0f) y[i] = v;
        else if (gelu && v <= -5.0f) y[i] = 0.0f;
        else y[i] = gelu ? 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))) : v;
    }
}

__global__ void bias_f16_kernel(const __half *x, const __half *bias, __half *y,
                                size_t n, size_t width, bool gelu) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(x[i]) + __half2float(bias[i % width]);
        if (gelu && v >= 5.0f) { y[i] = __float2half_rn(v); return; }
        if (gelu && v <= -5.0f) { y[i] = __float2half(0.0f); return; }
        if (gelu) v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        y[i] = __float2half_rn(v);
    }
}

__global__ void binary_f32_kernel(const float *a, const float *b, float *y, size_t n, int op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = a[i]; if (op == 0) v += b[i]; else v = v / (1.0f + expf(-v)) * b[i]; y[i] = v; }
}

__global__ void binary_f16_kernel(const __half *a, const __half *b, __half *y, size_t n, int op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = __half2float(a[i]), rhs = __half2float(b[i]); if (op == 0) v += rhs; else v = v / (1.0f + expf(-v)) * rhs; y[i] = __float2half_rn(v); }
}

template<typename T> __global__ void rope_kernel(const T *src, T *dst, size_t seq,
    size_t qh, size_t kh, size_t dim, double theta, size_t offset, bool query) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t heads = query ? qh : kh, n = seq * heads * dim;
    if (i >= n) return;
    size_t d = i % dim, pos = i / (heads * dim), head = (i / dim) % heads, half = dim / 2;
    size_t pair_dim = d < half ? d : d - half;
    float angle = (float)((double)(offset + pos) / pow(theta, (2.0 * (double)pair_dim) / (double)dim));
    float c = cosf(angle), s = sinf(angle);
    const T *data = src;
    size_t base = (pos * heads + head) * dim;
    float left = as_float(data[base + pair_dim]), right = as_float(data[base + pair_dim + half]);
    dst[i] = from_float<T>(d < half ? left * c - right * s : right * c + left * s);
}

template<typename T> __global__ void argmax_kernel(const T *input, size_t count,
                                                   uint32_t *result) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        size_t best = 0;
        float maximum = as_float(input[0]);
        for (size_t i = 1; i < count; ++i) {
            float value = as_float(input[i]);
            if (value > maximum) { maximum = value; best = i; }
        }
        *result = (uint32_t)best;
    }
}

template<typename T> bool linear(const T *x, const T *w, T *y,
                                size_t rows, size_t in, size_t out) {
    if (!cuda_ready() || !blas_ready() || x == nullptr || w == nullptr || y == nullptr ||
        rows == 0 || in == 0 || out == 0 || rows > INT_MAX || in > INT_MAX || out > INT_MAX ||
        rows > SIZE_MAX / in || out > SIZE_MAX / in || rows > SIZE_MAX / out) return false;
    size_t xc = rows * in, yc = rows * out;
    T *dx = nullptr, *dy = nullptr;
    bool ok = upload(x, xc, &dx) && alloc_device(yc, &dy);
    void *dw = ok ? weight_buffer(w, out * in * sizeof(T)) : nullptr;
    ok = ok && dw != nullptr;
    if (ok) {
        cublasStatus_t status;
        if constexpr (std::is_same<T, float>::value) {
            status = cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                (int)out, (int)rows, (int)in, &one_f, (const float *)dw, (int)in,
                dx, (int)in, &zero_f, dy, (int)out);
        } else {
            const float alpha = 1.0f, beta = 0.0f;
            status = cublasGemmEx(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                (int)out, (int)rows, (int)in, &alpha, dw, CUDA_R_16F, (int)in,
                dx, CUDA_R_16F, (int)in, &beta, dy, CUDA_R_16F, (int)out,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        ok = status == CUBLAS_STATUS_SUCCESS && download(y, dy, yc);
    }
    if (dx) cudaFree(dx); if (dy) cudaFree(dy);
    return ok;
}

template<typename T> bool norm(const T *x, const T *scale, const T *bias,
    T *y, size_t rows, size_t width, float eps, bool layernorm) {
    if (!cuda_ready() || x == nullptr || scale == nullptr || y == nullptr || rows == 0 || width == 0 ||
        rows > SIZE_MAX / width) return false;
    size_t count = rows * width;
    T *dx = nullptr, *ds = nullptr, *db = nullptr, *dy = nullptr;
    bool ok = upload(x, count, &dx) && upload(scale, width, &ds) && alloc_device(count, &dy);
    if (ok && bias != nullptr) ok = upload(bias, width, &db);
    if (ok) {
        if constexpr (std::is_same<T, float>::value)
            norm_f32_kernel<<<(unsigned)((rows + 127) / 128), 128>>>(dx, ds, db, dy, rows, width, eps, layernorm);
        else
            norm_f16_kernel<<<(unsigned)((rows + 127) / 128), 128>>>(dx, ds, db, dy, rows, width, eps, layernorm);
        ok = cudaGetLastError() == cudaSuccess && download(y, dy, count);
    }
    if (dx) cudaFree(dx); if (ds) cudaFree(ds); if (db) cudaFree(db); if (dy) cudaFree(dy);
    return ok;
}

template<typename T> bool bias_add(const T *x, const T *bias, T *y, size_t rows, size_t width, bool gelu) {
    if (!cuda_ready() || x == nullptr || bias == nullptr || y == nullptr || rows == 0 || width == 0 || rows > SIZE_MAX / width) return false;
    size_t n = rows * width; T *dx = nullptr, *db = nullptr, *dy = nullptr;
    bool ok = upload(x, n, &dx) && upload(bias, width, &db) && alloc_device(n, &dy);
    if (ok) {
        if constexpr (std::is_same<T, float>::value) bias_f32_kernel<<<(unsigned)((n + 255) / 256), 256>>>(dx, db, dy, n, width, gelu);
        else bias_f16_kernel<<<(unsigned)((n + 255) / 256), 256>>>(dx, db, dy, n, width, gelu);
        ok = cudaGetLastError() == cudaSuccess && download(y, dy, n);
    }
    if (dx) cudaFree(dx); if (db) cudaFree(db); if (dy) cudaFree(dy); return ok;
}

template<typename T> bool binary(const T *a, const T *b, T *y, size_t n, int op) {
    if (!cuda_ready() || a == nullptr || b == nullptr || y == nullptr || n == 0) return false;
    T *da = nullptr, *db = nullptr, *dy = nullptr;
    bool ok = upload(a, n, &da) && upload(b, n, &db) && alloc_device(n, &dy);
    if (ok) {
        if constexpr (std::is_same<T, float>::value) binary_f32_kernel<<<(unsigned)((n + 255) / 256), 256>>>(da, db, dy, n, op);
        else binary_f16_kernel<<<(unsigned)((n + 255) / 256), 256>>>(da, db, dy, n, op);
        ok = cudaGetLastError() == cudaSuccess && download(y, dy, n);
    }
    if (da) cudaFree(da); if (db) cudaFree(db); if (dy) cudaFree(dy); return ok;
}

template<typename T> bool rope(T *q, T *k, size_t seq, size_t qh, size_t kh, size_t dim, double theta, size_t offset) {
    if (!cuda_ready() || q == nullptr || k == nullptr || seq == 0 || qh == 0 || kh == 0 || dim == 0 || dim % 2 ||
        seq > SIZE_MAX / qh / dim || seq > SIZE_MAX / kh / dim) return false;
    size_t qn = seq * qh * dim, kn = seq * kh * dim; T *dq = nullptr, *dk = nullptr, *qo = nullptr, *ko = nullptr;
    bool ok = upload(q, qn, &dq) && upload(k, kn, &dk) && alloc_device(qn, &qo) && alloc_device(kn, &ko);
    if (ok) {
        rope_kernel<<<(unsigned)((qn + 255) / 256), 256>>>(dq, qo, seq, qh, kh, dim, theta, offset, true);
        rope_kernel<<<(unsigned)((kn + 255) / 256), 256>>>(dk, ko, seq, qh, kh, dim, theta, offset, false);
        ok = cudaGetLastError() == cudaSuccess && download(q, qo, qn) && download(k, ko, kn);
    }
    if (dq) cudaFree(dq); if (dk) cudaFree(dk); if (qo) cudaFree(qo); if (ko) cudaFree(ko); return ok;
}

template<typename T> bool embedding(const T *table, const uint32_t *ids, T *out, size_t rows, size_t width, size_t table_rows) {
    if (!cuda_ready() || table == nullptr || ids == nullptr || out == nullptr || rows == 0 || width == 0 || table_rows == 0 || rows > SIZE_MAX / width || table_rows > SIZE_MAX / width) return false;
    size_t n = rows * width; T *dy = nullptr; uint32_t *di = nullptr;
    bool ok = upload(ids, rows, &di) && alloc_device(n, &dy);
    void *cached = ok ? weight_buffer(table, table_rows * width * sizeof(T)) : nullptr;
    ok = ok && cached != nullptr;
    if (ok) {
        if constexpr (std::is_same<T, float>::value) embedding_f32_kernel<<<(unsigned)((n + 255) / 256), 256>>>((const float *)cached, di, (float *)dy, rows, width, table_rows);
        else embedding_f16_kernel<<<(unsigned)((n + 255) / 256), 256>>>((const __half *)cached, di, (__half *)dy, rows, width, table_rows);
        ok = cudaGetLastError() == cudaSuccess && download(out, dy, n);
    }
    if (di) cudaFree(di); if (dy) cudaFree(dy); return ok;
}

template<typename T> bool argmax(const T *x, size_t n, uint32_t *result) {
    if (!cuda_ready() || x == nullptr || result == nullptr || n == 0 || n > UINT32_MAX) return false;
    T *dx = nullptr; uint32_t *dr = nullptr;
    bool ok = upload(x, n, &dx) && alloc_device((size_t)1, &dr);
    if (ok) { argmax_kernel<<<1, 1>>>(dx, n, dr); ok = cudaGetLastError() == cudaSuccess && download(result, dr, 1); }
    if (dx) cudaFree(dx); if (dr) cudaFree(dr); return ok;
}

template<typename T> struct VisionSequence {
    size_t rows, hidden, heads, count, intermediate_count;
    T *states[2]{};
    T *norm1{}, *q{}, *k{}, *v{}, *attention{}, *projected{}, *after_attention{};
    T *norm2{}, *fc1{}, *activated{}, *mlp{};
    float *scores{};
    int current = 0;
};

template<typename T> void free_vision_sequence(VisionSequence<T> *s) {
    if (!s) return;
    for (T *p : {s->states[0], s->states[1], s->norm1, s->q, s->k, s->v,
                 s->attention, s->projected, s->after_attention, s->norm2,
                 s->fc1, s->activated, s->mlp}) if (p) cudaFree(p);
    if (s->scores) cudaFree(s->scores);
    delete s;
}

template<typename T> VisionSequence<T> *begin_vision_sequence(
    const T *host, size_t rows, size_t hidden) {
    if (!cuda_ready() || !host || !rows || !hidden || hidden % 64 ||
        rows > SIZE_MAX / hidden || rows > INT_MAX || hidden > INT_MAX) return nullptr;
    auto *s = new VisionSequence<T>();
    s->rows = rows; s->hidden = hidden; s->heads = hidden / 64;
    s->count = rows * hidden;
    if (rows > SIZE_MAX / rows / s->heads) { delete s; return nullptr; }
    const size_t score_count = rows * rows * s->heads;
    if (score_count > SIZE_MAX / sizeof(float)) { delete s; return nullptr; }
    bool ok = upload(host, s->count, &s->states[0]) && alloc_device(s->count, &s->states[1]) &&
        alloc_device(s->count, &s->norm1) && alloc_device(s->count, &s->q) &&
        alloc_device(s->count, &s->k) && alloc_device(s->count, &s->v) &&
        alloc_device(s->count, &s->attention) && alloc_device(s->count, &s->projected) &&
        alloc_device(s->count, &s->after_attention) && alloc_device(s->count, &s->norm2) &&
        alloc_device(s->count, &s->mlp) &&
        cudaMalloc(reinterpret_cast<void **>(&s->scores), score_count * sizeof(float)) == cudaSuccess;
    if (!ok) { free_vision_sequence(s); return nullptr; }
    return s;
}

template<typename T, typename Layer> bool vision_sequence_layer(
    VisionSequence<T> *s, const Layer *w, size_t rows, size_t hidden,
    size_t intermediate, float epsilon) {
    if (!s || !w || rows != s->rows || hidden != s->hidden || !intermediate ||
        rows > SIZE_MAX / intermediate || rows * intermediate > INT_MAX) return false;
    if (s->fc1 == nullptr || s->activated == nullptr) {
        s->intermediate_count = rows * intermediate;
        if (!alloc_device(s->intermediate_count, &s->fc1) ||
            !alloc_device(s->intermediate_count, &s->activated)) return false;
    } else if (s->intermediate_count != rows * intermediate) return false;
    const T *scale1 = (const T *)weight_buffer(w->norm1_scale, hidden * sizeof(T));
    const T *bias1 = (const T *)weight_buffer(w->norm1_bias, hidden * sizeof(T));
    const T *scale2 = (const T *)weight_buffer(w->norm2_scale, hidden * sizeof(T));
    const T *bias2 = (const T *)weight_buffer(w->norm2_bias, hidden * sizeof(T));
    if (!scale1 || !bias1 || !scale2 || !bias2 || !blas_ready()) {
        std::fprintf(stderr, "CUDA vision normalization weights or cuBLAS unavailable\n");
        return false;
    }
    T *input = s->states[s->current], *output = s->states[1 - s->current];
    if constexpr (std::is_same<T, float>::value)
        norm_f32_kernel<<<(unsigned)((rows + 127) / 128),128>>>(input,scale1,bias1,s->norm1,rows,hidden,epsilon,true);
    else
        norm_f16_kernel<<<(unsigned)((rows + 127) / 128),128>>>(input,scale1,bias1,s->norm1,rows,hidden,epsilon,true);
    const cudaError_t norm1_error = cudaGetLastError();
    if (norm1_error != cudaSuccess) {
        std::fprintf(stderr, "CUDA vision norm1 kernel failed: %s\n", cudaGetErrorString(norm1_error));
        return false;
    }
    if (!linear_device(s->norm1,w->q_weight,s->q,rows,hidden,hidden) ||
        !add_bias_device(s->q,w->q_bias,s->q,rows,hidden,false)) {
        std::fprintf(stderr, "CUDA vision Q projection failed\n"); return false;
    }
    if (!linear_device(s->norm1,w->k_weight,s->k,rows,hidden,hidden) ||
        !add_bias_device(s->k,w->k_bias,s->k,rows,hidden,false)) {
        std::fprintf(stderr, "CUDA vision K projection failed\n"); return false;
    }
    if (!linear_device(s->norm1,w->v_weight,s->v,rows,hidden,hidden) ||
        !add_bias_device(s->v,w->v_bias,s->v,rows,hidden,false)) {
        std::fprintf(stderr, "CUDA vision V projection failed\n"); return false;
    }
    const size_t score_count = s->heads * rows * rows;
    vision_score_kernel<<<(unsigned)((score_count+255)/256),256>>>(s->q,s->k,s->scores,rows,s->heads,64);
    if (cudaGetLastError() != cudaSuccess) { std::fprintf(stderr,"CUDA vision score kernel failed\n"); return false; }
    vision_softmax_kernel<<<(unsigned)(rows*s->heads),128>>>(s->scores,rows,s->heads);
    if (cudaGetLastError() != cudaSuccess) { std::fprintf(stderr,"CUDA vision softmax kernel failed\n"); return false; }
    vision_attention_kernel<<<(unsigned)((s->count+255)/256),256>>>(s->scores,s->v,s->attention,rows,s->heads,64);
    if (cudaGetLastError() != cudaSuccess) { std::fprintf(stderr,"CUDA vision attention kernel failed\n"); return false; }
    if (!linear_device(s->attention,w->out_weight,s->projected,rows,hidden,hidden) ||
        !add_bias_device(s->projected,w->out_bias,s->projected,rows,hidden,false)) {
        std::fprintf(stderr, "CUDA vision output projection failed\n"); return false;
    }
    residual_device_kernel<<<(unsigned)((s->count+255)/256),256>>>(input,s->projected,s->after_attention,s->count);
    if (cudaGetLastError() != cudaSuccess) { std::fprintf(stderr,"CUDA vision attention residual failed\n"); return false; }
    if constexpr (std::is_same<T,float>::value)
        norm_f32_kernel<<<(unsigned)((rows+127)/128),128>>>(s->after_attention,scale2,bias2,s->norm2,rows,hidden,epsilon,true);
    else
        norm_f16_kernel<<<(unsigned)((rows+127)/128),128>>>(s->after_attention,scale2,bias2,s->norm2,rows,hidden,epsilon,true);
    if (cudaGetLastError() != cudaSuccess) { std::fprintf(stderr,"CUDA vision norm2 kernel failed\n"); return false; }
    if (!linear_device(s->norm2,w->fc1_weight,s->fc1,rows,hidden,intermediate) ||
        !add_bias_device(s->fc1,w->fc1_bias,s->activated,rows,intermediate,true)) {
        std::fprintf(stderr, "CUDA vision FC1 projection failed\n"); return false;
    }
    if (!linear_device(s->activated,w->fc2_weight,s->mlp,rows,intermediate,hidden) ||
        !add_bias_device(s->mlp,w->fc2_bias,s->mlp,rows,hidden,false)) {
        std::fprintf(stderr, "CUDA vision FC2 projection failed\n"); return false;
    }
    residual_device_kernel<<<(unsigned)((s->count+255)/256),256>>>(s->after_attention,s->mlp,output,s->count);
    if (cudaGetLastError() != cudaSuccess) return false;
    s->current = 1 - s->current;
    return true;
}

bool vision_sequence_layer_quant_f16(VisionSequence<__half> *s,
    const TensorVisionLayerQuantF16 *w, TensorWeightQuantization format,
    size_t rows, size_t hidden, size_t intermediate, float epsilon) {
    if (!s || !w || rows != s->rows || hidden != s->hidden || !intermediate ||
        (format != TENSOR_WEIGHT_Q8_0 && format != TENSOR_WEIGHT_Q4_0) ||
        hidden % 32 || intermediate % 32 || rows > SIZE_MAX / intermediate ||
        rows * intermediate > INT_MAX) return false;
    if (s->fc1 == nullptr || s->activated == nullptr) {
        s->intermediate_count = rows * intermediate;
        if (!alloc_device(s->intermediate_count, &s->fc1) ||
            !alloc_device(s->intermediate_count, &s->activated)) return false;
    } else if (s->intermediate_count != rows * intermediate) return false;
    const __half *scale1 = (const __half *)weight_buffer(w->norm1_scale, hidden * sizeof(__half));
    const __half *bias1 = (const __half *)weight_buffer(w->norm1_bias, hidden * sizeof(__half));
    const __half *scale2 = (const __half *)weight_buffer(w->norm2_scale, hidden * sizeof(__half));
    const __half *bias2 = (const __half *)weight_buffer(w->norm2_bias, hidden * sizeof(__half));
    if (!scale1 || !bias1 || !scale2 || !bias2) return false;
    __half *input = s->states[s->current], *output = s->states[1 - s->current];
    norm_f16_kernel<<<(unsigned)((rows + 127) / 128),128>>>(input,scale1,bias1,
        s->norm1,rows,hidden,epsilon,true);
    if (cudaGetLastError() != cudaSuccess ||
        !quant_linear_f16_resident(s->norm1,w->q_weight,s->q,rows,hidden,hidden,format) ||
        !add_bias_device(s->q,w->q_bias,s->q,rows,hidden,false) ||
        !quant_linear_f16_resident(s->norm1,w->k_weight,s->k,rows,hidden,hidden,format) ||
        !add_bias_device(s->k,w->k_bias,s->k,rows,hidden,false) ||
        !quant_linear_f16_resident(s->norm1,w->v_weight,s->v,rows,hidden,hidden,format) ||
        !add_bias_device(s->v,w->v_bias,s->v,rows,hidden,false)) return false;
    const size_t score_count = s->heads * rows * rows;
    vision_score_kernel<<<(unsigned)((score_count+255)/256),256>>>(s->q,s->k,
        s->scores,rows,s->heads,64);
    if (cudaGetLastError() != cudaSuccess) return false;
    vision_softmax_kernel<<<(unsigned)(rows*s->heads),128>>>(s->scores,rows,s->heads);
    if (cudaGetLastError() != cudaSuccess) return false;
    vision_attention_kernel<<<(unsigned)((s->count+255)/256),256>>>(s->scores,
        s->v,s->attention,rows,s->heads,64);
    if (cudaGetLastError() != cudaSuccess ||
        !quant_linear_f16_resident(s->attention,w->out_weight,s->projected,
            rows,hidden,hidden,format) ||
        !add_bias_device(s->projected,w->out_bias,s->projected,rows,hidden,false)) return false;
    residual_device_kernel<<<(unsigned)((s->count+255)/256),256>>>(input,
        s->projected,s->after_attention,s->count);
    if (cudaGetLastError() != cudaSuccess) return false;
    norm_f16_kernel<<<(unsigned)((rows+127)/128),128>>>(s->after_attention,
        scale2,bias2,s->norm2,rows,hidden,epsilon,true);
    if (cudaGetLastError() != cudaSuccess ||
        !quant_linear_f16_resident(s->norm2,w->fc1_weight,s->fc1,
            rows,hidden,intermediate,format) ||
        !add_bias_device(s->fc1,w->fc1_bias,s->activated,rows,intermediate,true) ||
        !quant_linear_f16_resident(s->activated,w->fc2_weight,s->mlp,
            rows,intermediate,hidden,format) ||
        !add_bias_device(s->mlp,w->fc2_bias,s->mlp,rows,hidden,false)) return false;
    residual_device_kernel<<<(unsigned)((s->count+255)/256),256>>>(
        s->after_attention,s->mlp,output,s->count);
    if (cudaGetLastError() != cudaSuccess) return false;
    s->current = 1 - s->current;
    return true;
}

template<typename T> bool finish_vision_sequence(VisionSequence<T> *s, T *host) {
    return s && host && download(host, s->states[s->current], s->count);
}

} // namespace

extern "C" {

static int cuda_compute_available() { return cuda_ready(); }

static int linear_f32(const float *x, const float *w, float *y, size_t r, size_t i, size_t o) { return linear(x,w,y,r,i,o); }
static int linear_f16(const uint16_t *x, const uint16_t *w, uint16_t *y, size_t r, size_t i, size_t o) { return linear((const __half *)x,(const __half *)w,(__half *)y,r,i,o); }
static int linear_f32_device(const void *x, const void *w, void *y,
                             size_t r, size_t i, size_t o) {
    return linear_resident((const float *)x, (const float *)w, (float *)y, r, i, o);
}
static int linear_f16_device(const void *x, const void *w, void *y,
                             size_t r, size_t i, size_t o) {
    return linear_resident((const __half *)x, (const __half *)w, (__half *)y, r, i, o);
}
static int linear_q8_f16_device(const void *x,const void *w,void *y,size_t r,size_t i,size_t o) { return quant_linear_f16_device(x,w,y,r,i,o,true); }
static int linear_q4_f16_device(const void *x,const void *w,void *y,size_t r,size_t i,size_t o) { return quant_linear_f16_device(x,w,y,r,i,o,false); }
static int quant_linear_f16_host(const uint16_t *x,const uint8_t *w,uint16_t *y,
        size_t rows,size_t input_width,size_t output_width,bool q8) {
    if (!cuda_ready() || !x || !w || !y || !rows || !input_width || !output_width || input_width % 32 ||
        rows > SIZE_MAX / input_width || rows > SIZE_MAX / output_width ||
        output_width > SIZE_MAX / (input_width/32) / (q8 ? 34u : 18u)) return 0;
    const size_t input_count=rows*input_width, output_count=rows*output_width;
    const size_t weight_bytes=output_width*(input_width/32)*(q8 ? 34u : 18u);
    __half *dx=nullptr,*dy=nullptr;
    void *dw=weight_buffer(w,weight_bytes);
    bool ok=dw && upload((const __half *)x,input_count,&dx) && alloc_device(output_count,&dy) &&
        quant_linear_f16_device(dx,dw,dy,rows,input_width,output_width,q8) &&
        download((__half *)y,dy,output_count);
    if (dx) cudaFree(dx); if (dy) cudaFree(dy);
    return ok;
}
static int linear_q8_f16_host(const uint16_t*x,const uint8_t*w,uint16_t*y,size_t r,size_t i,size_t o) { return quant_linear_f16_host(x,w,y,r,i,o,true); }
static int linear_q4_f16_host(const uint16_t*x,const uint8_t*w,uint16_t*y,size_t r,size_t i,size_t o) { return quant_linear_f16_host(x,w,y,r,i,o,false); }
static int bias_f32_device(const void *x, const void *b, void *y,
                           size_t r, size_t w) {
    return bias_resident((const float *)x, (const float *)b, (float *)y, r, w);
}
static int bias_f16_device(const void *x, const void *b, void *y,
                           size_t r, size_t w) {
    return bias_resident((const __half *)x, (const __half *)b, (__half *)y, r, w);
}
static int embedding_f32(const float *t,const uint32_t *ids,float *o,size_t r,size_t w,size_t tr) { return embedding(t,ids,o,r,w,tr); }
static int embedding_f16(const uint16_t *t,const uint32_t *ids,uint16_t *o,size_t r,size_t w,size_t tr) { return embedding((const __half *)t,ids,(__half *)o,r,w,tr); }
static int rms_f32(const float *x,const float *s,float *y,size_t r,size_t w,float e) { return norm(x,s,(const float *)nullptr,y,r,w,e,false); }
static int layer_f32(const float *x,const float *s,const float *b,float *y,size_t r,size_t w,float e) { return norm(x,s,b,y,r,w,e,true); }
static int rms_f16(const uint16_t *x,const uint16_t *s,uint16_t *y,size_t r,size_t w,float e) { return norm((const __half *)x,(const __half *)s,(const __half *)nullptr,(__half *)y,r,w,e,false); }
static int layer_f16(const uint16_t *x,const uint16_t *s,const uint16_t *b,uint16_t *y,size_t r,size_t w,float e) { return norm((const __half *)x,(const __half *)s,(const __half *)b,(__half *)y,r,w,e,true); }
static int bias32(const float *x,const float *b,float *y,size_t r,size_t w) { return bias_add(x,b,y,r,w,false); }
static int bias16(const uint16_t *x,const uint16_t *b,uint16_t *y,size_t r,size_t w) { return bias_add((const __half *)x,(const __half *)b,(__half *)y,r,w,false); }
static int gelu32(const float *x,float *y,size_t n) { if (!cuda_ready()||!x||!y||!n)return 0; float *dx=nullptr,*dy=nullptr; bool ok=upload(x,n,&dx)&&alloc_device(n,&dy); if(ok){unary_f32_kernel<<<(unsigned)((n+255)/256),256>>>(dx,dy,n,0);ok=cudaGetLastError()==cudaSuccess&&download(y,dy,n);} if(dx)cudaFree(dx);if(dy)cudaFree(dy);return ok; }
static int gelu16(const uint16_t *x,uint16_t *y,size_t n) { if (!cuda_ready()||!x||!y||!n)return 0; __half *dx=nullptr,*dy=nullptr; bool ok=upload((const __half *)x,n,&dx)&&alloc_device(n,&dy); if(ok){unary_f16_kernel<<<(unsigned)((n+255)/256),256>>>(dx,dy,n,0);ok=cudaGetLastError()==cudaSuccess&&download((__half *)y,dy,n);} if(dx)cudaFree(dx);if(dy)cudaFree(dy);return ok; }
static int gelu_multiply16(const uint16_t *x,const uint16_t *g,uint16_t *y,size_t n) { if(!cuda_ready()||!x||!g||!y||!n)return 0; __half *dx=nullptr,*dg=nullptr,*dy=nullptr; bool ok=upload((const __half *)x,n,&dx)&&upload((const __half *)g,n,&dg)&&alloc_device(n,&dy); if(ok){gelu_multiply_f16_kernel<<<(unsigned)((n+255)/256),256>>>(dx,dg,dy,n);ok=cudaGetLastError()==cudaSuccess&&download((__half *)y,dy,n);} if(dx)cudaFree(dx);if(dg)cudaFree(dg);if(dy)cudaFree(dy);return ok; }
static int relu16(const uint16_t *x,uint16_t *y,size_t n) { if(!cuda_ready()||!x||!y||!n)return 0; __half *dx=nullptr,*dy=nullptr; bool ok=upload((const __half *)x,n,&dx)&&alloc_device(n,&dy); if(ok){relu_f16_kernel<<<(unsigned)((n+255)/256),256>>>(dx,dy,n);ok=cudaGetLastError()==cudaSuccess&&download((__half *)y,dy,n);} if(dx)cudaFree(dx);if(dy)cudaFree(dy);return ok; }
static int rope32(float*q,float*k,size_t s,size_t qh,size_t kh,size_t d,double t) { return rope(q,k,s,qh,kh,d,t,0); }
static int rope16(uint16_t*q,uint16_t*k,size_t s,size_t qh,size_t kh,size_t d,double t,size_t p) { return rope((__half *)q,(__half *)k,s,qh,kh,d,t,p); }
static int silu32(const float*a,const float*b,float*y,size_t n) { return binary(a,b,y,n,1); }
static int residual32(const float*a,const float*b,float*y,size_t n) { return binary(a,b,y,n,0); }
static int silu16(const uint16_t*a,const uint16_t*b,uint16_t*y,size_t n) { return binary((const __half *)a,(const __half *)b,(__half *)y,n,1); }
static int residual16(const uint16_t*a,const uint16_t*b,uint16_t*y,size_t n) { return binary((const __half *)a,(const __half *)b,(__half *)y,n,0); }
static int residual16_device(const void *a,const void *b,void *y,size_t n) {
    if (!cuda_ready() || a == nullptr || b == nullptr || y == nullptr || n == 0 ||
        n > (size_t)UINT_MAX * 256u) return 0;
    binary_f16_kernel<<<(unsigned)((n + 255) / 256), 256>>>(
        (const __half *)a, (const __half *)b, (__half *)y, n, 0);
    return cudaGetLastError() == cudaSuccess;
}
static int argmax32(const float*x,size_t n,uint32_t*r) { return argmax(x,n,r); }
static int argmax16(const uint16_t*x,size_t n,uint32_t*r) { return argmax((const __half *)x,n,r); }
static int bias_gelu16(const uint16_t*x,const uint16_t*b,uint16_t*y,size_t r,size_t w) { return bias_add((const __half *)x,(const __half *)b,(__half *)y,r,w,true); }

static void *vision_begin32(const float *x,size_t r,size_t h) { return begin_vision_sequence(x,r,h); }
static void *vision_begin16(const uint16_t *x,size_t r,size_t h) { return begin_vision_sequence((const __half *)x,r,h); }
static int vision_layer_seq32(void *s,const TensorVisionLayerF32*w,size_t r,size_t h,size_t m,float e) { return vision_sequence_layer((VisionSequence<float>*)s,w,r,h,m,e); }
static int vision_layer_seq16(void *s,const TensorVisionLayerF16*w,size_t r,size_t h,size_t m,float e) { return vision_sequence_layer((VisionSequence<__half>*)s,w,r,h,m,e); }
static int vision_layer_seq_quant16(void *s,const TensorVisionLayerQuantF16*w,TensorWeightQuantization q,size_t r,size_t h,size_t m,float e) { return vision_sequence_layer_quant_f16((VisionSequence<__half>*)s,w,q,r,h,m,e); }
static int vision_finish32(void*s,float*x) { return finish_vision_sequence((VisionSequence<float>*)s,x); }
static int vision_finish16(void*s,uint16_t*x) { return finish_vision_sequence((VisionSequence<__half>*)s,( __half*)x); }
static void vision_release32(void*s) { free_vision_sequence((VisionSequence<float>*)s); }
static void vision_release16(void*s) { free_vision_sequence((VisionSequence<__half>*)s); }
static int vision_layer32(float*x,const TensorVisionLayerF32*w,size_t r,size_t h,size_t m,float e) {
    auto *s=begin_vision_sequence(x,r,h); if(!s)return 0;
    bool ok=vision_sequence_layer(s,w,r,h,m,e)&&finish_vision_sequence(s,x); free_vision_sequence(s); return ok;
}
static int vision_layer16(uint16_t*x,const TensorVisionLayerF16*w,size_t r,size_t h,size_t m,float e) {
    auto *s=begin_vision_sequence((const __half*)x,r,h); if(!s)return 0;
    bool ok=vision_sequence_layer(s,w,r,h,m,e)&&finish_vision_sequence(s,( __half*)x); free_vision_sequence(s); return ok;
}

static int rms_linear32(const float*x,const float*s,float*n,const float*w,float*y,size_t r,size_t iw,size_t ow,float e) {
    return rms_f32(x,s,n,r,iw,e) && linear_f32(n,w,y,r,iw,ow);
}

static int mlp16(const uint16_t*x,const uint16_t*w1,const uint16_t*b1,const uint16_t*w2,const uint16_t*b2,uint16_t*y,size_t r,size_t iw,size_t mid,size_t ow) {
    if (r > SIZE_MAX / mid) return 0;
    size_t n = r * mid; uint16_t *raw=(uint16_t *)malloc(n*sizeof(uint16_t)), *act=(uint16_t *)malloc(n*sizeof(uint16_t));
    if(!raw||!act){free(raw);free(act);return 0;}
    int ok=linear_f16(x,w1,raw,r,iw,mid)&&bias_gelu16(raw,b1,act,r,mid)&&linear_f16(act,w2,y,r,mid,ow)&&bias16(y,b2,y,r,ow);
    free(raw);free(act);return ok;
}

static int qkv16(const uint16_t*x,const uint16_t*wq,const uint16_t*wk,const uint16_t*wv,
 const uint16_t*bq,const uint16_t*bk,const uint16_t*bv,uint16_t*q,uint16_t*k,uint16_t*v,size_t r,size_t iw,size_t ow) {
    return linear_f16(x,wq,q,r,iw,ow)&&(!bq||bias16(q,bq,q,r,ow))&&
        linear_f16(x,wk,k,r,iw,ow)&&(!bk||bias16(k,bk,k,r,ow))&&
        linear_f16(x,wv,v,r,iw,ow)&&(!bv||bias16(v,bv,v,r,ow));
}

static void release_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    for (auto &entry : weight_cache) cudaFree(entry.second.device);
    weight_cache.clear();
}

const TensorComputeBackend *tensor_compute_cuda_backend(void) {
    static const TensorComputeBackend backend = [] {
        TensorComputeBackend value{};
        value.name = "cuda"; value.is_available = cuda_compute_available;
        value.linear_f32 = linear_f32; value.linear_f16 = linear_f16;
        value.linear_f32_device = linear_f32_device;
        value.linear_f16_device = linear_f16_device;
        value.linear_q8_0_f16_device = linear_q8_f16_device;
        value.linear_q4_0_f16_device = linear_q4_f16_device;
        value.linear_q8_0_f16 = linear_q8_f16_host;
        value.linear_q4_0_f16 = linear_q4_f16_host;
        value.bias_add_f32_device = bias_f32_device;
        value.bias_add_f16_device = bias_f16_device;
        value.residual_add_f16_device = residual16_device;
        value.embedding_f16 = embedding_f16; value.rms_norm_f16 = rms_f16;
        value.layer_norm_f16 = layer_f16; value.bias_add_f16 = bias16;
        value.gelu_f16 = gelu16; value.rope_f16 = rope16;
        value.gelu_multiply_f16 = gelu_multiply16;
        value.relu_f16 = relu16;
        value.silu_multiply_f16 = silu16; value.residual_add_f16 = residual16;
        value.argmax_f16 = argmax16; value.embedding_f32 = embedding_f32;
        value.rms_norm_f32 = rms_f32; value.rms_norm_linear_f32 = rms_linear32;
        value.layer_norm_f32 = layer_f32; value.bias_add_f32 = bias32;
        value.gelu_f32 = gelu32; value.rope_f32 = rope32;
        value.silu_multiply_f32 = silu32; value.residual_add_f32 = residual32;
        value.argmax_f32 = argmax32; value.release_cache = release_cache;
        value.mlp_f16 = mlp16; value.qkv_f16 = qkv16;
        value.vision_layer_f16 = vision_layer16; value.vision_layer_f32 = vision_layer32;
        value.vision_sequence_begin_f16 = vision_begin16;
        value.vision_sequence_layer_f16 = vision_layer_seq16;
        value.vision_sequence_layer_quant_f16 = vision_layer_seq_quant16;
        value.vision_sequence_finish_f16 = vision_finish16;
        value.vision_sequence_release_f16 = vision_release16;
        value.vision_sequence_begin_f32 = vision_begin32;
        value.vision_sequence_layer_f32 = vision_layer_seq32;
        value.vision_sequence_finish_f32 = vision_finish32;
        value.vision_sequence_release_f32 = vision_release32;
        return value;
    }();
    return &backend;
}


} // extern C
