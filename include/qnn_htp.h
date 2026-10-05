#ifndef INFERENCE_SDK_QNN_HTP_H
#define INFERENCE_SDK_QNN_HTP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "safetensors.h"

typedef struct InferenceSdkQnnHtpSession InferenceSdkQnnHtpSession;
typedef struct InferenceSdkQnnHtpLinear InferenceSdkQnnHtpLinear;
typedef struct InferenceSdkQnnHtpRmsNorm InferenceSdkQnnHtpRmsNorm;
typedef struct InferenceSdkQnnHtpEmbedding InferenceSdkQnnHtpEmbedding;
typedef struct InferenceSdkQnnHtpRope InferenceSdkQnnHtpRope;
typedef struct InferenceSdkQnnHtpMlp InferenceSdkQnnHtpMlp;
typedef struct InferenceSdkQnnHtpAttention InferenceSdkQnnHtpAttention;
typedef struct InferenceSdkQnnHtpLayerNorm InferenceSdkQnnHtpLayerNorm;
typedef struct InferenceSdkQnnHtpLinearBias InferenceSdkQnnHtpLinearBias;
typedef struct InferenceSdkQnnHtpElementwise InferenceSdkQnnHtpElementwise;
typedef struct InferenceSdkQnnHtpGelu InferenceSdkQnnHtpGelu;

/* Persistent HTP runtime. No CPU execution fallback is provided. */
int inference_sdk_qnn_htp_session_create(const char *htp_directory,
                                         InferenceSdkQnnHtpSession **session,
                                         char *message, size_t message_capacity);
void inference_sdk_qnn_htp_session_destroy(InferenceSdkQnnHtpSession *session);

/* Create a reusable fixed-batch FP16 [rows, input] x [output, input]^T graph.
 * Use one row for decode and multiple rows for a prefill projection. The session
 * must outlive every graph created from it; destroy graphs before the session.
 * QNN releases compiled graph resources when the owning session is destroyed. */
int inference_sdk_qnn_htp_linear_create(InferenceSdkQnnHtpSession *session,
                                        const char *name,
                                        const uint16_t *weights,
                                        uint32_t rows,
                                        uint32_t input_width,
                                        uint32_t output_width,
                                        InferenceSdkQnnHtpLinear **linear,
                                        char *message, size_t message_capacity);
/* Load an F16/F32/BF16 [output_width,input_width] projection directly from
 * Safetensors and compile it as a persistent HTP linear graph. */
int inference_sdk_qnn_htp_linear_create_from_safetensors(
                                        InferenceSdkQnnHtpSession *session,
                                        const char *graph_name,
                                        const SafeTensors *weights_file,
                                        const char *tensor_name,
                                        uint32_t rows,
                                        uint32_t input_width,
                                        uint32_t output_width,
                                        InferenceSdkQnnHtpLinear **linear,
                                        char *message, size_t message_capacity);
int inference_sdk_qnn_htp_linear_execute(InferenceSdkQnnHtpLinear *linear,
                                         const uint16_t *input,
                                         uint32_t rows,
                                         uint16_t *output,
                                         char *message, size_t message_capacity);
void inference_sdk_qnn_htp_linear_destroy(InferenceSdkQnnHtpLinear *linear);

/* Reusable FP16 RMSNorm over the last dimension of a fixed [rows,width] tensor. */
int inference_sdk_qnn_htp_rms_norm_create(InferenceSdkQnnHtpSession *session,
                                          const char *name,
                                          const uint16_t *scale,
                                          uint32_t rows, uint32_t width,
                                          float epsilon,
                                          InferenceSdkQnnHtpRmsNorm **norm,
                                          char *message, size_t message_capacity);
int inference_sdk_qnn_htp_rms_norm_execute(InferenceSdkQnnHtpRmsNorm *norm,
                                           const uint16_t *input,
                                           uint32_t rows,
                                           uint16_t *output,
                                           char *message, size_t message_capacity);
void inference_sdk_qnn_htp_rms_norm_destroy(InferenceSdkQnnHtpRmsNorm *norm);

/* Fixed-batch token embedding lookup: INT32 token IDs -> FP16 [rows,width]. */
int inference_sdk_qnn_htp_embedding_create(InferenceSdkQnnHtpSession *session,
                                           const char *name,
                                           const uint16_t *table,
                                           uint32_t vocabulary_size,
                                           uint32_t width,
                                           uint32_t rows,
                                           InferenceSdkQnnHtpEmbedding **embedding,
                                           char *message, size_t message_capacity);
int inference_sdk_qnn_htp_embedding_execute(InferenceSdkQnnHtpEmbedding *embedding,
                                            const int32_t *token_ids,
                                            uint32_t rows,
                                            uint16_t *output,
                                            char *message, size_t message_capacity);
void inference_sdk_qnn_htp_embedding_destroy(InferenceSdkQnnHtpEmbedding *embedding);

/* Split-half rotary embedding. Input/output layout is [1,heads,rows,width];
 * position IDs select rows from the static cos/sin cache. */
int inference_sdk_qnn_htp_rope_create(InferenceSdkQnnHtpSession *session,
                                      const char *name, uint32_t heads,
                                      uint32_t rows, uint32_t width,
                                      uint32_t max_positions, float theta,
                                      InferenceSdkQnnHtpRope **rope,
                                      char *message, size_t message_capacity);
int inference_sdk_qnn_htp_rope_execute(InferenceSdkQnnHtpRope *rope,
                                       const uint16_t *input,
                                       const int32_t *position_ids,
                                       uint16_t *output,
                                       char *message, size_t message_capacity);
void inference_sdk_qnn_htp_rope_destroy(InferenceSdkQnnHtpRope *rope);

/* One-dispatch gated MLP: down(SiLU(gate(x))*up(x)) + residual. */
int inference_sdk_qnn_htp_mlp_create(InferenceSdkQnnHtpSession *session,
                                     const char *name,
                                     const uint16_t *gate_weights,
                                     const uint16_t *up_weights,
                                     const uint16_t *down_weights,
                                     uint32_t rows, uint32_t hidden_width,
                                     uint32_t intermediate_width,
                                     InferenceSdkQnnHtpMlp **mlp,
                                     char *message, size_t message_capacity);
int inference_sdk_qnn_htp_mlp_execute(InferenceSdkQnnHtpMlp *mlp,
                                      const uint16_t *input,
                                      const uint16_t *residual,
                                      uint16_t *output,
                                      char *message, size_t message_capacity);
void inference_sdk_qnn_htp_mlp_destroy(InferenceSdkQnnHtpMlp *mlp);

/* Fixed-shape HTP self-attention. Inputs use [1,heads,rows,head_width].
 * position_offset + causal mask support prefill or a single decode query with
 * a caller-managed KV cache. GQA expands KV heads on HTP. */
int inference_sdk_qnn_htp_attention_create(InferenceSdkQnnHtpSession *session,
                                           const char *name,
                                           uint32_t query_heads,
                                           uint32_t kv_heads,
                                           uint32_t query_rows,
                                           uint32_t key_rows,
                                           uint32_t head_width,
                                           uint32_t position_offset,
                                           int causal,
                                           InferenceSdkQnnHtpAttention **attention,
                                           char *message, size_t message_capacity);
int inference_sdk_qnn_htp_attention_execute(InferenceSdkQnnHtpAttention *attention,
                                            const uint16_t *query,
                                            const uint16_t *key,
                                            const uint16_t *value,
                                            uint16_t *output,
                                            char *message, size_t message_capacity);
int inference_sdk_qnn_htp_attention_execute_masked(InferenceSdkQnnHtpAttention *attention,
                                                   const uint16_t *query,
                                                   const uint16_t *key,
                                                   const uint16_t *value,
                                                   const uint16_t *additive_mask,
                                                   uint16_t *output,
                                                   char *message, size_t message_capacity);
void inference_sdk_qnn_htp_attention_destroy(InferenceSdkQnnHtpAttention *attention);

int inference_sdk_qnn_htp_layer_norm_create(InferenceSdkQnnHtpSession *session,
                                            const char *name,
                                            const uint16_t *scale,
                                            const uint16_t *bias,
                                            uint32_t rows, uint32_t width,
                                            float epsilon,
                                            InferenceSdkQnnHtpLayerNorm **norm,
                                            char *message, size_t message_capacity);
int inference_sdk_qnn_htp_layer_norm_execute(InferenceSdkQnnHtpLayerNorm *norm,
                                             const uint16_t *input,
                                             uint16_t *output,
                                             char *message, size_t message_capacity);
void inference_sdk_qnn_htp_layer_norm_destroy(InferenceSdkQnnHtpLayerNorm *norm);

/* FP16 MatMul plus broadcast bias, compiled as one reusable graph. */
int inference_sdk_qnn_htp_linear_bias_create(InferenceSdkQnnHtpSession *session,
                                             const char *name,
                                             const uint16_t *weights,
                                             const uint16_t *bias,
                                             uint32_t rows, uint32_t input_width,
                                             uint32_t output_width,
                                             InferenceSdkQnnHtpLinearBias **linear,
                                             char *message, size_t message_capacity);
int inference_sdk_qnn_htp_linear_bias_execute(InferenceSdkQnnHtpLinearBias *linear,
                                              const uint16_t *input,
                                              uint16_t *output,
                                              char *message, size_t message_capacity);
void inference_sdk_qnn_htp_linear_bias_destroy(InferenceSdkQnnHtpLinearBias *linear);

typedef enum {
    INFERENCE_SDK_QNN_HTP_ADD = 0,
    INFERENCE_SDK_QNN_HTP_MULTIPLY = 1,
    INFERENCE_SDK_QNN_HTP_SILU_MULTIPLY = 2
} InferenceSdkQnnHtpElementwiseOp;
int inference_sdk_qnn_htp_elementwise_create(InferenceSdkQnnHtpSession *session,
                                             const char *name,
                                             uint32_t rows, uint32_t width,
                                             uint32_t rhs_rows,
                                             InferenceSdkQnnHtpElementwiseOp operation,
                                             InferenceSdkQnnHtpElementwise **op,
                                             char *message, size_t message_capacity);
int inference_sdk_qnn_htp_elementwise_execute(InferenceSdkQnnHtpElementwise *op,
                                              const uint16_t *lhs,
                                              const uint16_t *rhs,
                                              uint16_t *output,
                                              char *message, size_t message_capacity);
void inference_sdk_qnn_htp_elementwise_destroy(InferenceSdkQnnHtpElementwise *op);
int inference_sdk_qnn_htp_gelu_create(InferenceSdkQnnHtpSession *session,
                                      const char *name, uint32_t rows,
                                      uint32_t width, InferenceSdkQnnHtpGelu **gelu,
                                      char *message, size_t message_capacity);
int inference_sdk_qnn_htp_gelu_execute(InferenceSdkQnnHtpGelu *gelu,
                                       const uint16_t *input, uint16_t *output,
                                       char *message, size_t message_capacity);
void inference_sdk_qnn_htp_gelu_destroy(InferenceSdkQnnHtpGelu *gelu);

#ifdef __cplusplus
}
#endif
#endif
