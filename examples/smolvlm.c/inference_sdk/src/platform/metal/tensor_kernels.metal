#include <metal_stdlib>
using namespace metal;

struct NormParams {
  uint rows;
  uint width;
  float epsilon;
};

struct RopeParams {
  uint sequence;
  uint query_heads;
  uint kv_heads;
  uint head_dim;
  float theta;
  uint position_offset;
};

struct BiasParams {
  uint width;
};

struct BiasGeluParams {
  uint width;
};

kernel void tensor_embedding_f32(
    device const float* table [[buffer(0)]],
    device const uint* token_ids [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint row = index / width;
  output[index] = table[token_ids[row] * width + index % width];
}

kernel void tensor_embedding_f16(
    device const half* table [[buffer(0)]], device const uint* token_ids [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& width [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = table[token_ids[index / width] * width + index % width];
}

kernel void tensor_rms_norm_f16(
    device const half* input [[buffer(0)]], device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]], constant NormParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  float sum = 0.0f;
  for (uint col = lane; col < params.width; col += 32) {
    const float x = float(input[row * params.width + col]); sum = fma(x, x, sum);
  }
  const float inverse = rsqrt(simd_sum(sum) / float(params.width) + params.epsilon);
  for (uint col = lane; col < params.width; col += 32) {
    const uint i = row * params.width + col;
    output[i] = half(float(input[i]) * inverse * float(weight[col]));
  }
}

kernel void tensor_layer_norm_f16(
    device const half* input [[buffer(0)]], device const half* scale [[buffer(1)]],
    device const half* bias [[buffer(2)]], device half* output [[buffer(3)]],
    constant NormParams& params [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  float total = 0.0f;
  for (uint col = lane; col < params.width; col += 32) total += float(input[row * params.width + col]);
  const float mean = simd_sum(total) / float(params.width);
  float squares = 0.0f;
  for (uint col = lane; col < params.width; col += 32) {
    const float d = float(input[row * params.width + col]) - mean; squares = fma(d, d, squares);
  }
  const float inverse = rsqrt(simd_sum(squares) / float(params.width) + params.epsilon);
  for (uint col = lane; col < params.width; col += 32) {
    const uint i = row * params.width + col;
    output[i] = half((float(input[i]) - mean) * inverse * float(scale[col]) + float(bias[col]));
  }
}

kernel void tensor_rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant NormParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  float sum = 0.0f;
  for (uint column = lane; column < params.width; column += 32) {
    const float value = input[row * params.width + column];
    sum = fma(value, value, sum);
  }
  const float square_sum = simd_sum(sum);
  const float inverse = rsqrt(square_sum / float(params.width) + params.epsilon);
  for (uint column = lane; column < params.width; column += 32)
    output[row * params.width + column] =
        input[row * params.width + column] * inverse * weight[column];
}

kernel void tensor_layer_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant NormParams& params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  float sum = 0.0f;
  for (uint column = lane; column < params.width; column += 32)
    sum += input[row * params.width + column];
  const float mean = simd_sum(sum) / float(params.width);
  float squared_sum = 0.0f;
  for (uint column = lane; column < params.width; column += 32) {
    const float centered = input[row * params.width + column] - mean;
    squared_sum = fma(centered, centered, squared_sum);
  }
  const float inverse = rsqrt(simd_sum(squared_sum) / float(params.width) + params.epsilon);
  for (uint column = lane; column < params.width; column += 32) {
    const uint index = row * params.width + column;
    output[index] = (input[index] - mean) * inverse * scale[column] + bias[column];
  }
}

kernel void tensor_bias_add_f32(
    device const float* input [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant BiasParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = input[index] + bias[index % params.width];
}

kernel void tensor_bias_add_f16(device const half* input [[buffer(0)]],
    device const half* bias [[buffer(1)]], device half* output [[buffer(2)]],
    constant BiasParams& params [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  output[index] = half(float(input[index]) + float(bias[index % params.width]));
}

kernel void tensor_gelu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  const float value = input[index];
  if (value >= 5.0f) {
    output[index] = value;
    return;
  }
  if (value <= -5.0f) {
    output[index] = 0.0f;
    return;
  }
  const float cubic = 0.044715f * value * value * value;
  output[index] = 0.5f * value * (1.0f + tanh(0.7978845608f * (value + cubic)));
}

kernel void tensor_gelu_f16(device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]], uint index [[thread_position_in_grid]]) {
  const float x = float(input[index]);
  if (x >= 5.0f) { output[index] = half(x); return; }
  if (x <= -5.0f) { output[index] = half(0.0f); return; }
  const float cubic = 0.044715f * x * x * x;
  output[index] = half(0.5f * x * (1.0f + tanh(0.7978845608f * (x + cubic))));
}

kernel void tensor_bias_gelu_f32(
    device const float* input [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant BiasGeluParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const float value = input[index] + bias[index % params.width];
  if (value >= 5.0f) {
    output[index] = value;
    return;
  }
  if (value <= -5.0f) {
    output[index] = 0.0f;
    return;
  }
  const float cubic = 0.044715f * value * value * value;
  output[index] = 0.5f * value * (1.0f + tanh(0.7978845608f * (value + cubic)));
}

kernel void tensor_bias_gelu_f16(
    device const half* input [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant BiasGeluParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const float value = float(input[index]) + float(bias[index % params.width]);
  if (value >= 5.0f) { output[index] = half(value); return; }
  if (value <= -5.0f) { output[index] = half(0.0f); return; }
  const float cubic = 0.044715f * value * value * value;
  output[index] = half(0.5f * value * (1.0f + tanh(0.7978845608f * (value + cubic))));
}

kernel void tensor_rope_f32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* query_output [[buffer(2)]],
    device float* key_output [[buffer(3)]],
    constant RopeParams& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const uint half_dim = params.head_dim / 2;
  const uint query_count = params.sequence * params.query_heads * params.head_dim;
  if (index < query_count) {
    const uint dimension = index % params.head_dim;
    const uint position = index / (params.query_heads * params.head_dim);
    const uint frequency = dimension < half_dim ? dimension : dimension - half_dim;
    const uint paired = dimension < half_dim ? index + half_dim : index - half_dim;
    const float angle = float(params.position_offset + position) /
        pow(params.theta, 2.0f * float(frequency) / float(params.head_dim));
    const float sign = dimension < half_dim ? -1.0f : 1.0f;
    query_output[index] = query[index] * cos(angle) + sign * query[paired] * sin(angle);
  }
  const uint key_count = params.sequence * params.kv_heads * params.head_dim;
  if (index < key_count) {
    const uint dimension = index % params.head_dim;
    const uint position = index / (params.kv_heads * params.head_dim);
    const uint frequency = dimension < half_dim ? dimension : dimension - half_dim;
    const uint paired = dimension < half_dim ? index + half_dim : index - half_dim;
    const float angle = float(params.position_offset + position) /
        pow(params.theta, 2.0f * float(frequency) / float(params.head_dim));
    const float sign = dimension < half_dim ? -1.0f : 1.0f;
    key_output[index] = key[index] * cos(angle) + sign * key[paired] * sin(angle);
  }
}

kernel void tensor_rope_f16(
    device const half* query [[buffer(0)]], device const half* key [[buffer(1)]],
    device half* query_output [[buffer(2)]], device half* key_output [[buffer(3)]],
    constant RopeParams& params [[buffer(4)]], uint index [[thread_position_in_grid]]) {
  const uint half_dim = params.head_dim / 2;
  const uint q_count = params.sequence * params.query_heads * params.head_dim;
  if (index < q_count) {
    const uint d = index % params.head_dim, pos = index / (params.query_heads * params.head_dim);
    const uint freq = d < half_dim ? d : d - half_dim;
    const uint pair = d < half_dim ? index + half_dim : index - half_dim;
    const float angle = float(params.position_offset + pos) / pow(params.theta, 2.0f * float(freq) / float(params.head_dim));
    query_output[index] = half(float(query[index]) * cos(angle) +
        (d < half_dim ? -1.0f : 1.0f) * float(query[pair]) * sin(angle));
  }
  const uint k_count = params.sequence * params.kv_heads * params.head_dim;
  if (index < k_count) {
    const uint d = index % params.head_dim, pos = index / (params.kv_heads * params.head_dim);
    const uint freq = d < half_dim ? d : d - half_dim;
    const uint pair = d < half_dim ? index + half_dim : index - half_dim;
    const float angle = float(params.position_offset + pos) / pow(params.theta, 2.0f * float(freq) / float(params.head_dim));
    key_output[index] = half(float(key[index]) * cos(angle) +
        (d < half_dim ? -1.0f : 1.0f) * float(key[pair]) * sin(angle));
  }
}

kernel void tensor_silu_multiply_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const float value = gate[index];
  output[index] = value / (1.0f + exp(-value)) * up[index];
}

kernel void tensor_residual_add_f32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = left[index] + right[index];
}

kernel void tensor_silu_multiply_f16(device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]], device half* output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const float x = float(gate[index]); output[index] = half(x / (1.0f + exp(-x)) * float(up[index]));
}

kernel void tensor_residual_add_f16(device const half* left [[buffer(0)]],
    device const half* right [[buffer(1)]], device half* output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = half(float(left[index]) + float(right[index]));
}

kernel void tensor_argmax_f32(
    device const float* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]]) {
  float best_value = -INFINITY;
  uint best_index = 0;
  for (uint index = lane; index < count; index += 32) {
    const float value = input[index];
    if (value > best_value) {
      best_value = value;
      best_index = index;
    }
  }
  const float maximum = simd_max(best_value);
  const uint candidate = simd_min(best_value == maximum ? best_index : 0xffffffffu);
  if (lane == 0) output[0] = candidate;
}

kernel void tensor_argmax_f16(device const half* input [[buffer(0)]],
    device uint* output [[buffer(1)]], constant uint& count [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]]) {
  float best_value = -INFINITY; uint best_index = 0;
  for (uint i = lane; i < count; i += 32) {
    const float value = float(input[i]);
    if (value > best_value) { best_value = value; best_index = i; }
  }
  const float maximum = simd_max(best_value);
  const uint candidate = simd_min(best_value == maximum ? best_index : 0xffffffffu);
  if (lane == 0) output[0] = candidate;
}
