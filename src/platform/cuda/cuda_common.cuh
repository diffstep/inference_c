#ifndef SMOLVLM_CUDA_COMMON_CUH
#define SMOLVLM_CUDA_COMMON_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace smolvlm_cuda {

inline bool cuda_ready() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template<typename T> inline bool upload(const T *host, size_t count, T **device) {
    if (host == nullptr || device == nullptr || count == 0 || count > SIZE_MAX / sizeof(T)) return false;
    *device = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(device), count * sizeof(T)) != cudaSuccess) return false;
    if (cudaMemcpy(*device, host, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*device); *device = nullptr; return false;
    }
    return true;
}

template<typename T> inline bool download(T *host, const T *device, size_t count) {
    return host != nullptr && device != nullptr && count <= SIZE_MAX / sizeof(T) &&
        cudaMemcpy(host, device, count * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
}

template<typename T> inline bool alloc_device(size_t count, T **device) {
    if (device == nullptr || count == 0 || count > SIZE_MAX / sizeof(T)) return false;
    *device = nullptr;
    return cudaMalloc(reinterpret_cast<void **>(device), count * sizeof(T)) == cudaSuccess;
}

template<typename T> __device__ float as_float(T value) { return (float)value; }
template<> __device__ inline float as_float<__half>(__half value) { return __half2float(value); }
template<typename T> __device__ T from_float(float value) { return (T)value; }
template<> __device__ inline __half from_float<__half>(float value) { return __float2half_rn(value); }

} // namespace smolvlm_cuda

#endif
