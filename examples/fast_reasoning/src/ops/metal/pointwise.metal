#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  const uint base = tid * 4;
  if (base + 3 < count) {
    const device float4* lhs4 = reinterpret_cast<const device float4*>(lhs);
    const device float4* rhs4 = reinterpret_cast<const device float4*>(rhs);
    device float4* output4 = reinterpret_cast<device float4*>(output);
    output4[tid] = lhs4[tid] + rhs4[tid];
    return;
  }
  for (uint lane = 0; lane < 4 && base + lane < count; ++lane) {
    output[base + lane] = lhs[base + lane] + rhs[base + lane];
  }
}

kernel void elementwise_mul_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  const uint base = tid * 4;
  if (base + 3 < count) {
    const device float4* lhs4 = reinterpret_cast<const device float4*>(lhs);
    const device float4* rhs4 = reinterpret_cast<const device float4*>(rhs);
    device float4* output4 = reinterpret_cast<device float4*>(output);
    output4[tid] = lhs4[tid] * rhs4[tid];
    return;
  }
  for (uint lane = 0; lane < 4 && base + lane < count; ++lane) {
    output[base + lane] = lhs[base + lane] * rhs[base + lane];
  }
}
