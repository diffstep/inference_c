#include <metal_stdlib>
using namespace metal;

struct SdpaParams {
  uint batch;
  uint query_heads;
  uint kv_heads;
  uint query_length;
  uint kv_length;
  uint head_dim;
  uint value_dim;
  uint causal_offset;
  uint causal;
  uint has_mask;
  float scale;
};

kernel void sdpa_qk_scores(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* mask [[buffer(2)]],
    device float* scores [[buffer(3)]],
    constant SdpaParams& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const uint row = index / params.kv_length;
  const uint key_position = index % params.kv_length;
  const uint query_position = row % params.query_length;
  const uint query_head = (row / params.query_length) % params.query_heads;
  const uint batch_index = row / (params.query_heads * params.query_length);
  const uint heads_per_kv = params.query_heads / params.kv_heads;
  const uint kv_head = query_head / heads_per_kv;
  const uint query_offset = row * params.head_dim;
  const uint key_offset = ((batch_index * params.kv_heads + kv_head) *
                           params.kv_length + key_position) * params.head_dim;
  float score = 0.0f;
  for (uint dimension = 0; dimension < params.head_dim; ++dimension) {
    score = fma(query[query_offset + dimension],
                key[key_offset + dimension], score);
  }
  score *= params.scale;
  if (params.has_mask != 0)
    score += mask[query_position * params.kv_length + key_position];
  if (params.causal != 0 &&
      key_position > query_position + params.causal_offset)
    score = -INFINITY;
  scores[index] = score;
}

kernel void sdpa_row_softmax(
    device const float* scores [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    constant SdpaParams& params [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
  const uint offset = row * params.kv_length;
  float maximum = -INFINITY;
  for (uint key_position = 0; key_position < params.kv_length; ++key_position)
    maximum = max(maximum, scores[offset + key_position]);
  if (!isfinite(maximum)) {
    for (uint key_position = 0; key_position < params.kv_length; ++key_position)
      probabilities[offset + key_position] = 0.0f;
    return;
  }
  float denominator = 0.0f;
  for (uint key_position = 0; key_position < params.kv_length; ++key_position) {
    const float probability = exp(scores[offset + key_position] - maximum);
    probabilities[offset + key_position] = probability;
    denominator += probability;
  }
  const float inverse_denominator = 1.0f / denominator;
  for (uint key_position = 0; key_position < params.kv_length; ++key_position)
    probabilities[offset + key_position] *= inverse_denominator;
}

kernel void sdpa_av_apply(
    device const float* probabilities [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant SdpaParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint row = index / params.value_dim;
  const uint output_dimension = index % params.value_dim;
  const uint query_head = row / params.query_length % params.query_heads;
  const uint batch_index = row / (params.query_heads * params.query_length);
  const uint heads_per_kv = params.query_heads / params.kv_heads;
  const uint kv_head = query_head / heads_per_kv;
  const uint probability_offset = row * params.kv_length;
  float result = 0.0f;
  for (uint key_position = 0; key_position < params.kv_length; ++key_position) {
    const uint value_offset = ((batch_index * params.kv_heads + kv_head) *
                               params.kv_length + key_position) * params.value_dim;
    result = fma(probabilities[probability_offset + key_position],
                 value[value_offset + output_dimension], result);
  }
  output[index] = result;
}

kernel void sdpa_online_f32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant SdpaParams& params [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint query_index = group.x;
  const uint query_head = group.y;
  const uint batch_index = group.z;
  const uint heads_per_kv = params.query_heads / params.kv_heads;
  const uint kv_head = query_head / heads_per_kv;
  const uint query_base = ((batch_index * params.query_heads + query_head) *
                           params.query_length + query_index) * params.head_dim;
  const uint key_head_base = (batch_index * params.kv_heads + kv_head) *
                             params.kv_length * params.head_dim;
  const uint value_head_base = (batch_index * params.kv_heads + kv_head) *
                               params.kv_length * params.value_dim;
  const uint output_base = ((batch_index * params.query_heads + query_head) *
                            params.query_length + query_index) * params.value_dim;
  float accumulator[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f};
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  for (uint key_index = 0; key_index < params.kv_length; ++key_index) {
    if (params.causal != 0 &&
        key_index > query_index + params.causal_offset) continue;
    float dot = 0.0f;
    for (uint dimension = lane; dimension < params.head_dim; dimension += 32) {
      dot = fma(query[query_base + dimension],
                key[key_head_base + key_index * params.head_dim + dimension], dot);
    }
    dot = simd_sum(dot) * params.scale;
    if (params.has_mask != 0)
      dot += mask[query_index * params.kv_length + key_index];
    const float next_max = max(running_max, dot);
    const float previous_scale = isfinite(running_max) ?
        fast::exp(running_max - next_max) : 0.0f;
    const float probability = isfinite(dot) ? fast::exp(dot - next_max) : 0.0f;
    running_sum = running_sum * previous_scale + probability;
    for (uint block = 0; block < (params.value_dim + 31) / 32; ++block) {
      const uint dimension = block * 32 + lane;
      if (dimension < params.value_dim)
        accumulator[block] = accumulator[block] * previous_scale + probability *
            value[value_head_base + key_index * params.value_dim + dimension];
    }
    running_max = next_max;
  }
  for (uint block = 0; block < (params.value_dim + 31) / 32; ++block) {
    const uint dimension = block * 32 + lane;
    if (dimension < params.value_dim)
      output[output_base + dimension] = running_sum > 0.0f ?
          accumulator[block] / running_sum : 0.0f;
  }
}

kernel void sdpa_online_f16(
    device const half* query [[buffer(0)]], device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]], device const half* mask [[buffer(3)]],
    device half* output [[buffer(4)]], constant SdpaParams& params [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint qi = group.x, qh = group.y, batch = group.z;
  const uint heads_per_kv = params.query_heads / params.kv_heads;
  const uint kvh = qh / heads_per_kv;
  const uint qbase = ((batch * params.query_heads + qh) * params.query_length + qi) * params.head_dim;
  const uint kbase = (batch * params.kv_heads + kvh) * params.kv_length * params.head_dim;
  const uint vbase = (batch * params.kv_heads + kvh) * params.kv_length * params.value_dim;
  const uint obase = ((batch * params.query_heads + qh) * params.query_length + qi) * params.value_dim;
  float accumulator[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float running_max = -INFINITY, running_sum = 0.0f;
  for (uint ki = 0; ki < params.kv_length; ++ki) {
    if (params.causal != 0 && ki > qi + params.causal_offset) continue;
    float dot = 0.0f;
    for (uint d = lane; d < params.head_dim; d += 32)
      dot = fma(float(query[qbase + d]), float(key[kbase + ki * params.head_dim + d]), dot);
    dot = simd_sum(dot) * params.scale;
    if (params.has_mask != 0) dot += float(mask[qi * params.kv_length + ki]);
    const float next_max = max(running_max, dot);
    const float old_scale = isfinite(running_max) ? fast::exp(running_max - next_max) : 0.0f;
    const float probability = isfinite(dot) ? fast::exp(dot - next_max) : 0.0f;
    running_sum = running_sum * old_scale + probability;
    for (uint block = 0; block < (params.value_dim + 31) / 32; ++block) {
      const uint d = block * 32 + lane;
      if (d < params.value_dim)
        accumulator[block] = accumulator[block] * old_scale + probability * float(value[vbase + ki * params.value_dim + d]);
    }
    running_max = next_max;
  }
  for (uint block = 0; block < (params.value_dim + 31) / 32; ++block) {
    const uint d = block * 32 + lane;
    if (d < params.value_dim)
      output[obase + d] = half(running_sum > 0.0f ? accumulator[block] / running_sum : 0.0f);
  }
}

struct DecodeSdpaParams {
  uint query_heads;
  uint kv_heads;
  uint kv_length;
  uint kv_capacity;
  uint head_dim;
  float scale;
  uint block_size;
  uint paged;
};

struct DecodeRopeParams {
  uint query_heads;
  uint kv_heads;
  uint head_dim;
  uint position;
  float theta;
  uint block_size;
  uint paged;
};

inline uint decode_physical_token(uint position, uint block_size, uint paged,
                                  device const uint* block_table) {
  return paged != 0 ? block_table[position / block_size] * block_size +
                      position % block_size : position;
}

kernel void tensor_gemv_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_width [[buffer(3)]],
    constant ulong2& weight_strides [[buffer(4)]],
    uint column [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr uint thread_count = 128;
  constexpr uint simdgroup_count = thread_count / 32;
  threadgroup float partials[simdgroup_count];
  float accumulator = 0.0f;
  for (uint index = thread_index; index < input_width; index += thread_count)
    accumulator = fma(input[index],
        weight[ulong(index) * weight_strides.x + ulong(column) * weight_strides.y],
        accumulator);
  const float reduced = simd_sum(accumulator);
  if (lane == 0) partials[simdgroup] = reduced;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup == 0) {
    const float value = lane < simdgroup_count ? partials[lane] : 0.0f;
    const float total = simd_sum(value);
    if (lane == 0) output[column] = total;
  }
}

kernel void tensor_gemv_residual_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& input_width [[buffer(4)]],
    constant ulong2& weight_strides [[buffer(5)]],
    uint column [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr uint thread_count = 128;
  constexpr uint simdgroup_count = thread_count / 32;
  threadgroup float partials[simdgroup_count];
  float accumulator = 0.0f;
  for (uint index = thread_index; index < input_width; index += thread_count)
    accumulator = fma(input[index],
        weight[ulong(index) * weight_strides.x + ulong(column) * weight_strides.y],
        accumulator);
  const float reduced = simd_sum(accumulator);
  if (lane == 0) partials[simdgroup] = reduced;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup == 0) {
    const float value = lane < simdgroup_count ? partials[lane] : 0.0f;
    const float total = simd_sum(value);
    if (lane == 0) output[column] = total + residual[column];
  }
}

kernel void tensor_gemv_silu_multiply_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_width [[buffer(3)]],
    constant ulong2& weight_strides [[buffer(4)]],
    constant uint& output_width [[buffer(5)]],
    uint column [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr uint thread_count = 128;
  constexpr uint simdgroup_count = thread_count / 32;
  threadgroup float gate_partials[simdgroup_count];
  threadgroup float up_partials[simdgroup_count];
  float gate_accumulator = 0.0f;
  float up_accumulator = 0.0f;
  for (uint index = thread_index; index < input_width; index += thread_count) {
    gate_accumulator = fma(input[index],
        weight[ulong(index) * weight_strides.x + ulong(column) * weight_strides.y],
        gate_accumulator);
    up_accumulator = fma(input[index],
        weight[ulong(index) * weight_strides.x +
               ulong(column + output_width) * weight_strides.y], up_accumulator);
  }
  const float gate_reduced = simd_sum(gate_accumulator);
  const float up_reduced = simd_sum(up_accumulator);
  if (lane == 0) {
    gate_partials[simdgroup] = gate_reduced;
    up_partials[simdgroup] = up_reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup == 0) {
    const float gate_value = lane < simdgroup_count ? gate_partials[lane] : 0.0f;
    const float up_value = lane < simdgroup_count ? up_partials[lane] : 0.0f;
    const float gate = simd_sum(gate_value);
    const float up = simd_sum(up_value);
    if (lane == 0) output[column] = gate / (1.0f + fast::exp(-gate)) * up;
  }
}

kernel void tensor_gemv_4simd_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_width [[buffer(3)]],
    constant ulong2& weight_strides [[buffer(4)]],
    constant uint& output_width [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  const uint column = group * 4 + simdgroup;
  if (column >= output_width) return;
  float accumulator = 0.0f;
  for (uint index = lane; index < input_width; index += 32)
    accumulator = fma(input[index],
        weight[ulong(index) * weight_strides.x + ulong(column) * weight_strides.y],
        accumulator);
  const float total = simd_sum(accumulator);
  if (lane == 0) output[column] = total;
}

kernel void tensor_gemv_residual_4simd_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& input_width [[buffer(4)]],
    constant ulong2& weight_strides [[buffer(5)]],
    constant uint& output_width [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  const uint column = group * 4 + simdgroup;
  if (column >= output_width) return;
  float accumulator = 0.0f;
  for (uint index = lane; index < input_width; index += 32)
    accumulator = fma(input[index],
        weight[ulong(index) * weight_strides.x + ulong(column) * weight_strides.y],
        accumulator);
  const float total = simd_sum(accumulator);
  if (lane == 0) output[column] = total + residual[column];
}

kernel void tensor_gemv_silu_multiply_4simd_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& input_width [[buffer(3)]],
    constant ulong2& weight_strides [[buffer(4)]],
    constant uint& output_width [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  const uint column = group * 4 + simdgroup;
  if (column >= output_width) return;
  float gate_accumulator = 0.0f;
  float up_accumulator = 0.0f;
  for (uint index = lane; index < input_width; index += 32) {
    gate_accumulator = fma(input[index],
        weight[ulong(index) * weight_strides.x + ulong(column) * weight_strides.y],
        gate_accumulator);
    up_accumulator = fma(input[index],
        weight[ulong(index) * weight_strides.x +
               ulong(column + output_width) * weight_strides.y], up_accumulator);
  }
  const float gate = simd_sum(gate_accumulator);
  const float up = simd_sum(up_accumulator);
  if (lane == 0) output[column] = gate / (1.0f + fast::exp(-gate)) * up;
}

kernel void sdpa_decode_rope_cache_f32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* rotated_query [[buffer(3)]],
    device float* key_cache [[buffer(4)]],
    device float* value_cache [[buffer(5)]],
    constant DecodeRopeParams& params [[buffer(6)]],
    device const uint* block_table [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  const uint query_count = params.query_heads * params.head_dim;
  const uint kv_count = params.kv_heads * params.head_dim;
  const uint half_dim = params.head_dim / 2;
  if (index < query_count) {
    const uint dimension = index % params.head_dim;
    const uint head_base = (index / params.head_dim) * params.head_dim;
    const uint frequency = dimension < half_dim ? dimension : dimension - half_dim;
    const uint paired = head_base + (dimension < half_dim ? dimension + half_dim : dimension - half_dim);
    const float angle = float(params.position) /
        pow(params.theta, 2.0f * float(frequency) / float(params.head_dim));
    const float sign = dimension < half_dim ? -1.0f : 1.0f;
    rotated_query[index] = query[index] * cos(angle) + sign * query[paired] * sin(angle);
  }
  if (index < kv_count) {
    const uint dimension = index % params.head_dim;
    const uint head_base = (index / params.head_dim) * params.head_dim;
    const uint frequency = dimension < half_dim ? dimension : dimension - half_dim;
    const uint paired = head_base + (dimension < half_dim ? dimension + half_dim : dimension - half_dim);
    const float angle = float(params.position) /
        pow(params.theta, 2.0f * float(frequency) / float(params.head_dim));
    const float sign = dimension < half_dim ? -1.0f : 1.0f;
    const uint cache_position = decode_physical_token(params.position,
        params.block_size, params.paged, block_table);
    key_cache[cache_position * kv_count + index] =
        key[index] * cos(angle) + sign * key[paired] * sin(angle);
    value_cache[cache_position * kv_count + index] = value[index];
  }
}

kernel void sdpa_decode_cache_f32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant DecodeSdpaParams& params [[buffer(4)]],
    device const uint* block_table [[buffer(5)]],
    uint query_head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint kv_head = query_head / (params.query_heads / params.kv_heads);
  float accumulator[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f};
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  for (uint key_index = 0; key_index < params.kv_length; ++key_index) {
    float dot = 0.0f;
    for (uint dimension = lane; dimension < params.head_dim; dimension += 32) {
      const uint query_index = query_head * params.head_dim + dimension;
      const uint cache_position = decode_physical_token(key_index,
          params.block_size, params.paged, block_table);
      const uint key_indexed = cache_position * params.kv_heads * params.head_dim +
                               kv_head * params.head_dim + dimension;
      dot = fma(query[query_index], key[key_indexed], dot);
    }
    dot = simd_sum(dot) * params.scale;
    const float next_max = max(running_max, dot);
    const float old_scale = isfinite(running_max) ? fast::exp(running_max - next_max) : 0.0f;
    const float probability = fast::exp(dot - next_max);
    running_sum = running_sum * old_scale + probability;
    for (uint block = 0; block < (params.head_dim + 31) / 32; ++block) {
      const uint dimension = block * 32 + lane;
      if (dimension < params.head_dim) {
        const uint cache_position = decode_physical_token(key_index,
            params.block_size, params.paged, block_table);
        const uint value_index = cache_position * params.kv_heads * params.head_dim +
                                 kv_head * params.head_dim + dimension;
        accumulator[block] = accumulator[block] * old_scale + probability * value[value_index];
      }
    }
    running_max = next_max;
  }
  for (uint block = 0; block < (params.head_dim + 31) / 32; ++block) {
    const uint dimension = block * 32 + lane;
    if (dimension < params.head_dim)
      output[query_head * params.head_dim + dimension] = accumulator[block] / running_sum;
  }
}

// FP16 storage with FP32 accumulation for autoregressive decode.
kernel void sdpa_decode_rope_cache_f16(
    device const half* query [[buffer(0)]], device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]], device half* rotated_query [[buffer(3)]],
    device half* key_cache [[buffer(4)]], device half* value_cache [[buffer(5)]],
    constant DecodeRopeParams& params [[buffer(6)]],
    device const uint* block_table [[buffer(7)]], uint index [[thread_position_in_grid]]) {
  const uint query_count = params.query_heads * params.head_dim;
  const uint kv_count = params.kv_heads * params.head_dim;
  const uint half_dim = params.head_dim / 2;
  if (index < query_count) {
    const uint d = index % params.head_dim, base = (index / params.head_dim) * params.head_dim;
    const uint frequency = d < half_dim ? d : d - half_dim;
    const uint paired = base + (d < half_dim ? d + half_dim : d - half_dim);
    const float angle = float(params.position) / pow(params.theta, 2.0f * float(frequency) / float(params.head_dim));
    const float sign = d < half_dim ? -1.0f : 1.0f;
    rotated_query[index] = half(float(query[index]) * cos(angle) + sign * float(query[paired]) * sin(angle));
  }
  if (index < kv_count) {
    const uint d = index % params.head_dim, base = (index / params.head_dim) * params.head_dim;
    const uint frequency = d < half_dim ? d : d - half_dim;
    const uint paired = base + (d < half_dim ? d + half_dim : d - half_dim);
    const float angle = float(params.position) / pow(params.theta, 2.0f * float(frequency) / float(params.head_dim));
    const float sign = d < half_dim ? -1.0f : 1.0f;
    const uint cache_position = decode_physical_token(params.position,
        params.block_size, params.paged, block_table);
    key_cache[cache_position * kv_count + index] = half(float(key[index]) * cos(angle) + sign * float(key[paired]) * sin(angle));
    value_cache[cache_position * kv_count + index] = value[index];
  }
}

kernel void sdpa_decode_cache_f16(
    device const half* query [[buffer(0)]], device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]], device half* output [[buffer(3)]],
    constant DecodeSdpaParams& params [[buffer(4)]],
    device const uint* block_table [[buffer(5)]], uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint kv_head = head / (params.query_heads / params.kv_heads);
  float acc[8] = {0}; float maximum = -INFINITY, sum = 0.0f;
  for (uint k = 0; k < params.kv_length; ++k) {
    float dot = 0.0f;
    const uint cache_position = decode_physical_token(k,
        params.block_size, params.paged, block_table);
    for (uint d = lane; d < params.head_dim; d += 32)
      dot = fma(float(query[head * params.head_dim + d]), float(key[cache_position * params.kv_heads * params.head_dim + kv_head * params.head_dim + d]), dot);
    dot = simd_sum(dot) * params.scale;
    const float next_max = max(maximum, dot);
    const float old_scale = isfinite(maximum) ? fast::exp(maximum - next_max) : 0.0f;
    const float probability = fast::exp(dot - next_max);
    sum = sum * old_scale + probability;
    for (uint b = 0; b < (params.head_dim + 31) / 32; ++b) {
      const uint d = b * 32 + lane;
      if (d < params.head_dim) acc[b] = acc[b] * old_scale + probability * float(value[cache_position * params.kv_heads * params.head_dim + kv_head * params.head_dim + d]);
    }
    maximum = next_max;
  }
  for (uint b = 0; b < (params.head_dim + 31) / 32; ++b) {
    const uint d = b * 32 + lane;
    if (d < params.head_dim) output[head * params.head_dim + d] = half(acc[b] / sum);
  }
}

kernel void tensor_gemv_f16(device const half* input [[buffer(0)]], device const half* weight [[buffer(1)]], device half* output [[buffer(2)]], constant uint& width [[buffer(3)]], constant ulong2& strides [[buffer(4)]], uint column [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[4]; float acc = 0.0f;
  for (uint i = lane + sg * 32; i < width; i += 128) acc = fma(float(input[i]), float(weight[ulong(i) * strides.x + ulong(column) * strides.y]), acc);
  const float s = simd_sum(acc); if (lane == 0) partial[sg] = s; threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) { float v = lane < 4 ? partial[lane] : 0.0f; v = simd_sum(v); if (lane == 0) output[column] = half(v); }
}
kernel void tensor_gemv_residual_f16(device const half* input [[buffer(0)]], device const half* weight [[buffer(1)]], device const half* residual [[buffer(2)]], device half* output [[buffer(3)]], constant uint& width [[buffer(4)]], constant ulong2& strides [[buffer(5)]], uint column [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[4]; float acc = 0.0f;
  for (uint i = lane + sg * 32; i < width; i += 128) acc = fma(float(input[i]), float(weight[ulong(i) * strides.x + ulong(column) * strides.y]), acc);
  const float s = simd_sum(acc); if (lane == 0) partial[sg] = s; threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) { float v = lane < 4 ? partial[lane] : 0.0f; v = simd_sum(v); if (lane == 0) output[column] = half(float(half(v)) + float(residual[column])); }
}
kernel void tensor_gemv_silu_multiply_f16(device const half* input [[buffer(0)]], device const half* weight [[buffer(1)]], device half* output [[buffer(2)]], constant uint& width [[buffer(3)]], constant ulong2& strides [[buffer(4)]], constant uint& out_width [[buffer(5)]], uint column [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float pg[4], pu[4]; float g = 0.0f, u = 0.0f;
  for (uint i = lane + sg * 32; i < width; i += 128) { g = fma(float(input[i]), float(weight[ulong(i)*strides.x + ulong(column)*strides.y]), g); u = fma(float(input[i]), float(weight[ulong(i)*strides.x + ulong(column+out_width)*strides.y]), u); }
  const float gs = simd_sum(g), us = simd_sum(u); if (lane == 0) { pg[sg] = gs; pu[sg] = us; } threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) { float gv = simd_sum(lane < 4 ? pg[lane] : 0.0f); float uv = simd_sum(lane < 4 ? pu[lane] : 0.0f); if (lane == 0) { const float gate = float(half(gv)); const float up = float(half(uv)); output[column] = half(gate / (1.0f + exp(-gate)) * up); } }
}

inline float quant_block_value(device const uchar* block, uint index, bool q8) {
  const ushort scale_bits = ushort(block[0]) | (ushort(block[1]) << 8);
  const float scale = float(as_type<half>(scale_bits));
  if (q8) return scale * float(as_type<char>(block[2 + index]));
  const uchar packed = block[2 + (index & 15)];
  const int q = int(index < 16 ? packed & 15 : packed >> 4) - 8;
  return scale * float(q);
}

inline float quant_dot_f16(device const half* input, device const uchar* weight,
                           uint width, uint column, bool q8) {
  const uint blocks = width / 32;
  const uint block_bytes = q8 ? 34 : 18;
  device const uchar* row = weight + ulong(column) * blocks * block_bytes;
  float sum = 0.0f;
  for (uint b = 0; b < blocks; ++b) {
    device const uchar* block = row + b * block_bytes;
    for (uint k = 0; k < 32; ++k)
      sum = fma(float(input[b * 32 + k]), quant_block_value(block, k, q8), sum);
  }
  return sum;
}

kernel void tensor_gemv_quant_f16(device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]], device half* output [[buffer(2)]],
    constant uint& width [[buffer(3)]], constant uint& q8 [[buffer(4)]],
    uint column [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[4];
  const uint blocks = width / 32;
  const uint block_bytes = q8 ? 34 : 18;
  device const uchar* row = weight + ulong(column) * blocks * block_bytes;
  float acc = 0.0f;
  for (uint b = 0; b < blocks; ++b) {
    device const uchar* block = row + b * block_bytes;
    for (uint k = lane + sg * 32; k < 32; k += 128)
      acc = fma(float(input[b * 32 + k]), quant_block_value(block, k, q8 != 0), acc);
  }
  const float sum = simd_sum(acc);
  if (lane == 0) partial[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float total = simd_sum(lane < 4 ? partial[lane] : 0.0f);
    if (lane == 0) output[column] = half(total);
  }
}

kernel void tensor_gemv_quant_qkv_f16(device const half* input [[buffer(0)]],
    device const uchar* q_weight [[buffer(1)]],
    device const uchar* k_weight [[buffer(2)]],
    device const uchar* v_weight [[buffer(3)]], device half* output [[buffer(4)]],
    constant uint& width [[buffer(5)]], constant uint& q8 [[buffer(6)]],
    constant uint2& output_widths [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint q_width = output_widths.x, kv_width = output_widths.y;
  const uint column = row < q_width ? row :
      (row < q_width + kv_width ? row - q_width : row - q_width - kv_width);
  device const uchar* weight = row < q_width ? q_weight :
      (row < q_width + kv_width ? k_weight : v_weight);
  threadgroup float partial[4];
  const uint blocks = width / 32;
  const uint block_bytes = q8 ? 34 : 18;
  device const uchar* packed_row = weight + ulong(column) * blocks * block_bytes;
  float acc = 0.0f;
  for (uint b = 0; b < blocks; ++b) {
    device const uchar* block = packed_row + b * block_bytes;
    for (uint k = lane + sg * 32; k < 32; k += 128)
      acc = fma(float(input[b * 32 + k]), quant_block_value(block, k, q8 != 0), acc);
  }
  const float sum = simd_sum(acc);
  if (lane == 0) partial[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float total = simd_sum(lane < 4 ? partial[lane] : 0.0f);
    if (lane == 0) output[row] = half(total);
  }
}

kernel void tensor_gemv_quant_residual_f16(device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]], device const half* residual [[buffer(2)]],
    device half* output [[buffer(3)]], constant uint& width [[buffer(4)]],
    constant uint& q8 [[buffer(5)]], uint column [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[4];
  const uint blocks = width / 32;
  const uint block_bytes = q8 ? 34 : 18;
  device const uchar* row = weight + ulong(column) * blocks * block_bytes;
  float acc = 0.0f;
  for (uint b = 0; b < blocks; ++b) {
    device const uchar* block = row + b * block_bytes;
    for (uint k = lane + sg * 32; k < 32; k += 128)
      acc = fma(float(input[b * 32 + k]), quant_block_value(block, k, q8 != 0), acc);
  }
  const float sum = simd_sum(acc);
  if (lane == 0) partial[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float total = simd_sum(lane < 4 ? partial[lane] : 0.0f);
    if (lane == 0) output[column] = half(float(half(total)) + float(residual[column]));
  }
}

kernel void tensor_gemv_quant_silu_multiply_f16(device const half* input [[buffer(0)]],
    device const uchar* gate_weight [[buffer(1)]],
    device const uchar* up_weight [[buffer(2)]], device half* output [[buffer(3)]],
    constant uint& width [[buffer(4)]], constant uint& q8 [[buffer(5)]],
    uint column [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float pg[4], pu[4];
  const uint blocks = width / 32;
  const uint block_bytes = q8 ? 34 : 18;
  device const uchar* grow = gate_weight + ulong(column) * blocks * block_bytes;
  device const uchar* urow = up_weight + ulong(column) * blocks * block_bytes;
  float gate = 0.0f, up = 0.0f;
  for (uint b = 0; b < blocks; ++b) {
    device const uchar* gb = grow + b * block_bytes;
    device const uchar* ub = urow + b * block_bytes;
    for (uint k = lane + sg * 32; k < 32; k += 128) {
      const float x = float(input[b * 32 + k]);
      gate = fma(x, quant_block_value(gb, k, q8 != 0), gate);
      up = fma(x, quant_block_value(ub, k, q8 != 0), up);
    }
  }
  if (lane == 0) { pg[sg] = simd_sum(gate); pu[sg] = simd_sum(up); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float gv = simd_sum(lane < 4 ? pg[lane] : 0.0f);
    const float uv = simd_sum(lane < 4 ? pu[lane] : 0.0f);
    if (lane == 0) {
      const float g = float(half(gv)), u = float(half(uv));
      output[column] = half(g / (1.0f + exp(-g)) * u);
    }
  }
}

kernel void sdpa_decode_cache_partial_f32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* partials [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant DecodeSdpaParams& params [[buffer(6)]],
    device const uint* block_table [[buffer(7)]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint block_count = 32;
  const uint query_head = group_position.x;
  const uint block_index = group_position.z;
  const uint kv_head = query_head / (params.query_heads / params.kv_heads);
  const uint begin = uint((ulong(params.kv_length) * block_index) / block_count);
  const uint end = uint((ulong(params.kv_length) * (block_index + 1)) / block_count);
  float accumulator[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f};
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  for (uint key_index = begin; key_index < end; ++key_index) {
    float dot = 0.0f;
    const uint cache_position = decode_physical_token(key_index,
        params.block_size, params.paged, block_table);
    for (uint dimension = lane; dimension < params.head_dim; dimension += 32) {
      const uint query_index = query_head * params.head_dim + dimension;
      const uint key_indexed = cache_position * params.kv_heads * params.head_dim +
                               kv_head * params.head_dim + dimension;
      dot = fma(query[query_index], key[key_indexed], dot);
    }
    dot = simd_sum(dot) * params.scale;
    const float next_max = max(running_max, dot);
    const float old_scale = isfinite(running_max) ? fast::exp(running_max - next_max) : 0.0f;
    const float probability = fast::exp(dot - next_max);
    running_sum = running_sum * old_scale + probability;
    for (uint value_block = 0; value_block < (params.head_dim + 31) / 32; ++value_block) {
      const uint dimension = value_block * 32 + lane;
      if (dimension < params.head_dim) {
        const uint value_index = cache_position * params.kv_heads * params.head_dim +
                                 kv_head * params.head_dim + dimension;
        accumulator[value_block] = accumulator[value_block] * old_scale +
                                   probability * value[value_index];
      }
    }
    running_max = next_max;
  }
  const uint stats_index = query_head * block_count + block_index;
  sums[stats_index] = running_sum;
  maxs[stats_index] = running_max;
  for (uint value_block = 0; value_block < (params.head_dim + 31) / 32; ++value_block) {
    const uint dimension = value_block * 32 + lane;
    if (dimension < params.head_dim) {
      const uint partial_index = (query_head * block_count + block_index) *
                                 params.head_dim + dimension;
      partials[partial_index] = accumulator[value_block];
    }
  }
}

kernel void sdpa_decode_cache_reduce_f32(
    device const float* partials [[buffer(0)]],
    device const float* sums [[buffer(1)]],
    device const float* maxs [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant DecodeSdpaParams& params [[buffer(4)]],
    uint query_head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint block_count = 32;
  float global_max = -INFINITY;
  for (uint block_index = 0; block_index < block_count; ++block_index)
    global_max = max(global_max, maxs[query_head * block_count + block_index]);
  float denominator = 0.0f;
  for (uint block_index = 0; block_index < block_count; ++block_index) {
    const uint stats_index = query_head * block_count + block_index;
    denominator += sums[stats_index] * fast::exp(maxs[stats_index] - global_max);
  }
  for (uint value_block = 0; value_block < (params.head_dim + 31) / 32; ++value_block) {
    const uint dimension = value_block * 32 + lane;
    if (dimension < params.head_dim) {
      float numerator = 0.0f;
      for (uint block_index = 0; block_index < block_count; ++block_index) {
        const uint partial_index = (query_head * block_count + block_index) *
                                   params.head_dim + dimension;
        const uint stats_index = query_head * block_count + block_index;
        numerator += partials[partial_index] * fast::exp(maxs[stats_index] - global_max);
      }
      output[query_head * params.head_dim + dimension] = numerator / denominator;
    }
  }
}

kernel void sdpa_decode_cache_partial_f16(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device float* partials [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant DecodeSdpaParams& params [[buffer(6)]],
    device const uint* block_table [[buffer(7)]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint block_count = 32;
  const uint query_head = group_position.x;
  const uint block_index = group_position.z;
  const uint kv_head = query_head / (params.query_heads / params.kv_heads);
  const uint begin = uint((ulong(params.kv_length) * block_index) / block_count);
  const uint end = uint((ulong(params.kv_length) * (block_index + 1)) / block_count);
  float accumulator[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f};
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  for (uint key_index = begin; key_index < end; ++key_index) {
    float dot = 0.0f;
    const uint cache_position = decode_physical_token(key_index,
        params.block_size, params.paged, block_table);
    for (uint dimension = lane; dimension < params.head_dim; dimension += 32) {
      const uint query_index = query_head * params.head_dim + dimension;
      const uint key_indexed = cache_position * params.kv_heads * params.head_dim +
                               kv_head * params.head_dim + dimension;
      dot = fma(float(query[query_index]), float(key[key_indexed]), dot);
    }
    dot = simd_sum(dot) * params.scale;
    const float next_max = max(running_max, dot);
    const float old_scale = isfinite(running_max) ? fast::exp(running_max - next_max) : 0.0f;
    const float probability = fast::exp(dot - next_max);
    running_sum = running_sum * old_scale + probability;
    for (uint value_block = 0; value_block < (params.head_dim + 31) / 32; ++value_block) {
      const uint dimension = value_block * 32 + lane;
      if (dimension < params.head_dim) {
        const uint value_index = cache_position * params.kv_heads * params.head_dim +
                                 kv_head * params.head_dim + dimension;
        accumulator[value_block] = accumulator[value_block] * old_scale +
                                   probability * float(value[value_index]);
      }
    }
    running_max = next_max;
  }
  const uint stats_index = query_head * block_count + block_index;
  sums[stats_index] = running_sum;
  maxs[stats_index] = running_max;
  for (uint value_block = 0; value_block < (params.head_dim + 31) / 32; ++value_block) {
    const uint dimension = value_block * 32 + lane;
    if (dimension < params.head_dim) {
      const uint partial_index = (query_head * block_count + block_index) *
                                 params.head_dim + dimension;
      partials[partial_index] = accumulator[value_block];
    }
  }
}

kernel void sdpa_decode_cache_reduce_f16(
    device const float* partials [[buffer(0)]],
    device const float* sums [[buffer(1)]],
    device const float* maxs [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant DecodeSdpaParams& params [[buffer(4)]],
    uint query_head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint block_count = 32;
  float global_max = -INFINITY;
  for (uint block_index = 0; block_index < block_count; ++block_index)
    global_max = max(global_max, maxs[query_head * block_count + block_index]);
  float denominator = 0.0f;
  for (uint block_index = 0; block_index < block_count; ++block_index) {
    const uint stats_index = query_head * block_count + block_index;
    denominator += sums[stats_index] * fast::exp(maxs[stats_index] - global_max);
  }
  for (uint value_block = 0; value_block < (params.head_dim + 31) / 32; ++value_block) {
    const uint dimension = value_block * 32 + lane;
    if (dimension < params.head_dim) {
      float numerator = 0.0f;
      for (uint block_index = 0; block_index < block_count; ++block_index) {
        const uint partial_index = (query_head * block_count + block_index) *
                                   params.head_dim + dimension;
        const uint stats_index = query_head * block_count + block_index;
        numerator += partials[partial_index] * fast::exp(maxs[stats_index] - global_max);
      }
      output[query_head * params.head_dim + dimension] = half(numerator / denominator);
    }
  }
}

kernel void sdpa_residual_add_f32(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) output[index] = input[index] + residual[index];
}

#include "../../../vendor/candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal"
