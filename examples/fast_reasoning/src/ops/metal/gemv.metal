#include <metal_stdlib>
using namespace metal;

// Computes y = x * W for one input row. W is presented as a logical [K, N]
// tensor; strides preserve the underlying weight layout (usually [1, K]).
kernel void gemv_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& k_size [[buffer(3)]],
    constant ulong2& w_strides [[buffer(4)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint THREADS = 128;
    constexpr uint SIMDGROUPS = THREADS / 32;
    threadgroup float partials[SIMDGROUPS];

    const uint col = group_id.x;
    float acc = 0.0f;
    for (uint k = tid; k < k_size; k += THREADS) {
        acc = fma(x[k], w[ulong(k) * w_strides.x + ulong(col) * w_strides.y], acc);
    }
    const float reduced = simd_sum(acc);
    if (lane == 0) {
        partials[simdgroup] = reduced;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        const float group_sum = lane < SIMDGROUPS ? partials[lane] : 0.0f;
        const float total = simd_sum(group_sum);
        if (lane == 0) {
            y[col] = total;
        }
    }
}
