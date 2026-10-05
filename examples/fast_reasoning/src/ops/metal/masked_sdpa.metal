#include <metal_stdlib>
using namespace metal;

struct MaskedSdpaParams {
  uint query_heads;
  uint kv_heads;
  uint query_len;
  uint key_len;
  uint head_dim;
  uint window;
  float scale;
  uint causal;
};

// Short-sequence fused GQA attention. One SIMD group owns a single query/head:
// lanes represent keys for QK and then output dimensions for the AV reduction.
// The host limits key_len to 32 so all normalized probabilities fit in a SIMD
// group and are shared with simd_shuffle without a score/probability tensor.
kernel void masked_sdpa_f32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant MaskedSdpaParams& params [[buffer(4)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint query_index = group_id.x;
  const uint query_head = group_id.y;
  const uint kv_head = query_head / (params.query_heads / params.kv_heads);
  const uint query_base = (query_head * params.query_len + query_index) * params.head_dim;
  const uint key_head_base = kv_head * params.key_len * params.head_dim;

  float score = -INFINITY;
  if (lane < params.key_len) {
    const uint distance = query_index > lane ? query_index - lane : lane - query_index;
    const bool causal_allowed = params.causal == 0 || lane <= query_index;
    const bool window_allowed = params.window == 0 || distance < params.window;
    if (causal_allowed && window_allowed) {
      float dot = 0.0f;
      const uint key_base = key_head_base + lane * params.head_dim;
      for (uint d = 0; d < params.head_dim; ++d) {
        dot = fma(query[query_base + d], key[key_base + d], dot);
      }
      score = dot * params.scale;
    }
  }

  const float max_score = simd_max(score);
  const float exp_score = isfinite(score) && isfinite(max_score) ? fast::exp(score - max_score) : 0.0f;
  const float denominator = simd_sum(exp_score);
  const float probability = denominator > 0.0f ? exp_score / denominator : 0.0f;

  const uint value_head_base = kv_head * params.key_len * params.head_dim;
  for (uint d = 0; d < params.head_dim; d += 32) {
    const uint output_dim = d + lane;
    float result = 0.0f;
    for (uint key_index = 0; key_index < params.key_len; ++key_index) {
      const float key_probability = simd_shuffle(probability, ushort(key_index));
      result = fma(
          key_probability,
          value[value_head_base + key_index * params.head_dim + output_dim],
          result);
    }
    output[query_base + output_dim] = result;
  }
}
