#include <metal_stdlib>
using namespace metal;

kernel void scalar_add_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& value [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  const uint base = tid * 4;
  if (base + 3 < count) {
    const device float4* input4 = reinterpret_cast<const device float4*>(input);
    device float4* output4 = reinterpret_cast<device float4*>(output);
    output4[tid] = input4[tid] + float4(value);
    return;
  }
  for (uint lane = 0; lane < 4 && base + lane < count; ++lane) {
    output[base + lane] = input[base + lane] + value;
  }
}
