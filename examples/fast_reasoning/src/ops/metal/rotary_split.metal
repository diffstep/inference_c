#include <metal_stdlib>
using namespace metal;

struct RotarySplitHalfParams {
  uint batch;
  uint heads;
  uint sequence;
  uint head_dim;
};

kernel void rotary_split_half_f32(
    device const float* input [[buffer(0)]],
    device const float* cos_values [[buffer(1)]],
    device const float* sin_values [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant RotarySplitHalfParams& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = params.batch * params.heads * params.sequence * params.head_dim;
  if (index >= count) {
    return;
  }

  const uint head_dim = params.head_dim;
  const uint half_dim = head_dim / 2;
  const uint dimension = index % head_dim;
  const uint token = (index / head_dim) % params.sequence;
  const uint paired_dimension = dimension < half_dim ? dimension + half_dim : dimension - half_dim;
  const uint frequency = dimension < half_dim ? dimension : dimension - half_dim;
  const float angle_cos = cos_values[token * half_dim + frequency];
  const float angle_sin = sin_values[token * half_dim + frequency];
  const float current = input[index];
  const float paired = input[index - dimension + paired_dimension];
  output[index] = dimension < half_dim
      ? current * angle_cos - paired * angle_sin
      : current * angle_cos + paired * angle_sin;
}
