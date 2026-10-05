#include "tensor_compute_backend.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

template<typename T> T value(float x) { return (T)x; }
template<> uint16_t value(float x) { return __half_as_ushort(__float2half_rn(x)); }
template<typename T> float as_float(T x) { return (float)x; }
template<> float as_float(uint16_t x) { return __half2float(__ushort_as_half(x)); }

template<typename T> void fill(std::vector<T> &v, float scale, size_t seed) {
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = value<T>((float)((int)((i * 17 + seed * 13) % 29) - 14) * scale);
}

template<typename T> bool compare(const std::vector<T> &a, const std::vector<T> &b,
                                  float tolerance) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(as_float(a[i])) ||
            std::fabs(as_float(a[i]) - as_float(b[i])) > tolerance) return false;
    return true;
}

template<typename T, typename Layer> struct Weights {
    std::vector<T> n1s = std::vector<T>(64), n1b = std::vector<T>(64);
    std::vector<T> qw = std::vector<T>(4096), qb = std::vector<T>(64);
    std::vector<T> kw = std::vector<T>(4096), kb = std::vector<T>(64);
    std::vector<T> vw = std::vector<T>(4096), vb = std::vector<T>(64);
    std::vector<T> ow = std::vector<T>(4096), ob = std::vector<T>(64);
    std::vector<T> n2s = std::vector<T>(64), n2b = std::vector<T>(64);
    std::vector<T> f1w = std::vector<T>(8192), f1b = std::vector<T>(128);
    std::vector<T> f2w = std::vector<T>(8192), f2b = std::vector<T>(64);
    Layer layer;
    explicit Weights(size_t seed) {
        fill(n1s, .02f, seed + 1); fill(n1b, .002f, seed + 2);
        fill(qw, .001f, seed + 3); fill(qb, .002f, seed + 4);
        fill(kw, .001f, seed + 5); fill(kb, .002f, seed + 6);
        fill(vw, .001f, seed + 7); fill(vb, .002f, seed + 8);
        fill(ow, .001f, seed + 9); fill(ob, .002f, seed + 10);
        fill(n2s, .02f, seed + 11); fill(n2b, .002f, seed + 12);
        fill(f1w, .001f, seed + 13); fill(f1b, .002f, seed + 14);
        fill(f2w, .001f, seed + 15); fill(f2b, .002f, seed + 16);
        layer = {n1s.data(), n1b.data(), qw.data(), qb.data(), kw.data(), kb.data(),
            vw.data(), vb.data(), ow.data(), ob.data(), n2s.data(), n2b.data(),
            f1w.data(), f1b.data(), f2w.data(), f2b.data()};
    }
};

template<typename T, typename Layer, typename Begin, typename LayerFn,
         typename Finish, typename Release, typename Single>
bool check_path(const char *label, Begin begin, LayerFn run_layer, Finish finish,
                Release release, Single single, float tolerance) {
    constexpr size_t rows = 32, hidden = 64, intermediate = 128;
    Weights<T, Layer> first(1), second(41);
    std::vector<T> initial(rows * hidden), sequence_result(rows * hidden),
        fallback_result(rows * hidden);
    fill(initial, .01f, 71);
    auto *context = begin(initial.data(), rows, hidden);
    if (!context) { std::fprintf(stderr, "%s sequence begin failed\n", label); return false; }
    if (!run_layer(context, &first.layer, rows, hidden, intermediate, 1e-5f)) {
        release(context); std::fprintf(stderr, "%s sequence layer 1 failed\n", label); return false;
    }
    if (!run_layer(context, &second.layer, rows, hidden, intermediate, 1e-5f)) {
        release(context); std::fprintf(stderr, "%s sequence layer 2 failed\n", label); return false;
    }
    if (!finish(context, sequence_result.data())) {
        release(context); std::fprintf(stderr, "%s sequence finish failed\n", label); return false;
    }
    release(context);
    fallback_result = initial;
    if (!single(fallback_result.data(), &first.layer, rows, hidden, intermediate, 1e-5f) ||
        !single(fallback_result.data(), &second.layer, rows, hidden, intermediate, 1e-5f)) {
        std::fprintf(stderr, "%s layer fallback execution failed\n", label);
        return false;
    }
    if (!compare(sequence_result, fallback_result, tolerance)) {
        std::fprintf(stderr, "%s fused/fallback outputs differ\n", label);
        return false;
    }
    std::printf("%s fused/fallback outputs match\n", label);
    return true;
}

int main() {
    const TensorComputeBackend *backend = tensor_compute_cuda_backend();
    if (!backend || !backend->is_available || !backend->is_available()) {
        std::puts("CUDA device unavailable; test skipped");
        return 77;
    }
    const uint16_t gelu_values[] = {
        __half_as_ushort(__float2half_rn(0.0f)),
        __half_as_ushort(__float2half_rn(1.0f)),
        __half_as_ushort(__float2half_rn(-1.0f)),
        __half_as_ushort(__float2half_rn(10.375f))
    };
    const uint16_t gelu_gate[] = {
        __half_as_ushort(__float2half_rn(2.0f)),
        __half_as_ushort(__float2half_rn(3.0f)),
        __half_as_ushort(__float2half_rn(4.0f)),
        __half_as_ushort(__float2half_rn(-2.0f))
    };
    uint16_t gelu_result[4] = {};
    if (!backend->gelu_multiply_f16 ||
        !backend->gelu_multiply_f16(gelu_values, gelu_gate, gelu_result, 4)) {
        std::fprintf(stderr, "CUDA FP16 GELU multiply failed\n");
        return 1;
    }
    for (size_t i = 0; i < 4; ++i) {
        const float x = as_float(gelu_values[i]);
        const float expected = (x >= 5.0f ? x : x <= -5.0f ? 0.0f :
            0.5f * x * (1.0f + std::tanh(0.7978845608f *
                (x + 0.044715f * x * x * x)))) * as_float(gelu_gate[i]);
        if (std::fabs(as_float(gelu_result[i]) - expected) > 2e-3f) {
            std::fprintf(stderr, "CUDA FP16 GELU multiply mismatch at %zu\n", i);
            return 1;
        }
    }
    const uint16_t relu_values[] = {
        __half_as_ushort(__float2half_rn(-1.0f)),
        __half_as_ushort(__float2half_rn(0.0f)),
        __half_as_ushort(__float2half_rn(3.0f))
    };
    uint16_t relu_result[3] = {};
    if (!backend->relu_f16 || !backend->relu_f16(relu_values, relu_result, 3) ||
        as_float(relu_result[0]) != 0.0f || as_float(relu_result[1]) != 0.0f ||
        as_float(relu_result[2]) != 3.0f) {
        std::fprintf(stderr, "CUDA FP16 ReLU failed\n");
        return 1;
    }
    const bool fp32 = check_path<float, TensorVisionLayerF32>("FP32",
        backend->vision_sequence_begin_f32, backend->vision_sequence_layer_f32,
        backend->vision_sequence_finish_f32, backend->vision_sequence_release_f32,
        backend->vision_layer_f32, 2e-4f);
    if (backend->release_cache) backend->release_cache();
    const bool fp16 = check_path<uint16_t, TensorVisionLayerF16>("FP16",
        backend->vision_sequence_begin_f16, backend->vision_sequence_layer_f16,
        backend->vision_sequence_finish_f16, backend->vision_sequence_release_f16,
        backend->vision_layer_f16, 2e-3f);
    if (backend->release_cache) backend->release_cache();
    return fp32 && fp16 ? 0 : 1;
}
