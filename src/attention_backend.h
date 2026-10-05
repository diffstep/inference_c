#ifndef ATTENTION_BACKEND_H
#define ATTENTION_BACKEND_H

#include "attention.h"

typedef struct {
    TensorAttentionBackend kind;
    const char *name;
    int (*is_available)(void);
    int (*run)(const float *query, const float *key, const float *value,
               const float *additive_mask, float *output,
               size_t batch, size_t query_heads, size_t kv_heads,
               size_t query_length, size_t kv_length, size_t head_dim,
               size_t value_dim, float scale, int causal);
    int (*run_f16)(const uint16_t *query, const uint16_t *key,
                   const uint16_t *value, const uint16_t *additive_mask,
                   uint16_t *output, size_t batch, size_t query_heads,
                   size_t kv_heads, size_t query_length, size_t kv_length,
                   size_t head_dim, size_t value_dim, float scale, int causal);
    int (*decode)(const float *, const float *, const float *, const float *,
                  const float *, float *, size_t, size_t, size_t, size_t, size_t);
    int (*decode_sequence_available)(void);
    int (*decode_sequence)(const TensorAttentionDecoderLayer *, size_t,
                           const float *, uint32_t, size_t, uint32_t *,
                           const float *const *, const float *const *, size_t,
                           size_t, size_t, size_t, size_t, size_t, size_t,
                           float, double, const float *, const float *, size_t);
    int (*decode_sequence_f16_available)(void);
    int (*decode_sequence_f16)(const TensorAttentionDecoderLayerF16 *, size_t,
                           const uint16_t *, uint32_t, size_t, uint32_t *,
                           const uint16_t *const *, const uint16_t *const *, size_t,
                           size_t, size_t, size_t, size_t, size_t, size_t,
                           float, double, const uint16_t *, const uint16_t *, size_t);
    int (*decode_sequence_quant_f16_available)(TensorWeightQuantization);
    int (*decode_sequence_quant_f16)(const TensorAttentionDecoderLayerQuantF16 *, size_t,
                           const uint16_t *, uint32_t, size_t, uint32_t *,
                           const uint16_t *const *, const uint16_t *const *, size_t,
                           size_t, size_t, size_t, size_t, size_t, size_t,
                           float, double, const uint16_t *, const uint8_t *, size_t,
                           TensorWeightQuantization);
    int (*decode_sequence_paged)(const TensorPagedKVDecodeRequest *);
    void (*decode_cache_clear)(void);
} TensorAttentionBackendOps;

#ifdef __cplusplus
extern "C" {
#endif

const TensorAttentionBackendOps *tensor_attention_metal_backend(void);
const TensorAttentionBackendOps *tensor_attention_cuda_backend(void);
#if defined(INFERENCE_SDK_QNN_HTP_ENABLED)
const TensorAttentionBackendOps *tensor_attention_qnn_htp_backend(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
