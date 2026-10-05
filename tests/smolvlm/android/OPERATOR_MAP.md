# SmolVLM → QNN HTP operation and weight map

This is the stage-0 execution map derived from `tests/smolvlm/src/app/`,
`tests/smolvlm/sample-model/config.json`, the processor/tokenizer metadata, and
Safetensors headers. The dimensions below describe that checkpoint; runtime
graph creation must use validated config and Safetensors metadata, not these
constants. The local file has 471 F32 tensors (about 0.955 GiB of tensor data).

## Text path

| Stage | Checkpoint tensors / operation | Expected layout | HTP work |
| --- | --- | --- | --- |
| Token embedding | `model.text_model.embed_tokens.weight`, gather by token ID | `[49280, 576]` → `[S,576]` | Embedding/gather |
| Per decoder layer ×30: input norm | `input_layernorm.weight`; RMSNorm, epsilon from config | `[S,576]` | RMS + scale |
| Q projection | `self_attn.q_proj.weight` | `[576,576]` | MatMul |
| K/V projections | `self_attn.k_proj.weight`, `v_proj.weight` | `[192,576]` each (`3 KV heads ×64`) | MatMul |
| Position encoding | Q/K RoPE, theta from config, offset = prior KV length | Q `[S,9,64]`, K `[S,3,64]` | RoPE or fused Q/K op |
| KV state | per-layer K/V history | `[capacity,3,64]` each | Correct cache read/write and position; no host attention fallback |
| Self attention | 9 Q heads / 3 KV heads, causal mask during prefill | scores `[9,S,past+S]` | GQA attention / softmax / value product |
| Attention output | `self_attn.o_proj.weight` | `[576,576]` | MatMul + residual |
| Per decoder layer: post norm | `post_attention_layernorm.weight`; RMSNorm | `[S,576]` | RMS + scale |
| Gated MLP | `mlp.gate_proj.weight`, `up_proj.weight` | `[1536,576]` each | Two MatMuls + SiLU(gate) × up |
| MLP down | `mlp.down_proj.weight` | `[576,1536]` | MatMul + residual |
| Output | `model.text_model.norm.weight`, `lm_head.weight` | RMSNorm `[576]`; LM head `[49280,576]` | RMSNorm, MatMul; CPU sampling only |

Text layer weights have no projection biases (`attention_bias=false`,
`mlp_bias=false`). Top-level `tie_word_embeddings=false`, and a separate
`lm_head.weight` exists, so the runner must load both embedding and LM head.
RoPE is non-interleaved (`rope_interleaved=false`), theta is 100000, epsilon is
`1e-5`, and max context is 8192. The text model uses SiLU gated MLP.

## Vision and connector path

| Stage | Checkpoint tensors / operation | Expected layout | HTP work |
| --- | --- | --- | --- |
| Patch embedding | `embeddings.patch_embedding.weight`, `.bias` | flattened RGB patch `[768]` → `[768]`; tile `512×512`, patch 16, sequence 1024 | Batched patch projection + bias |
| Position embedding | `embeddings.position_embedding.weight` | `[1024,768]` | Add |
| Per vision layer ×12: pre-norm | `layer_norm1.{weight,bias}`, epsilon `1e-6` | `[1024,768]` | LayerNorm |
| Q/K/V | `self_attn.{q,k,v}_proj.{weight,bias}` | each weight `[768,768]` | Batched projections; preserve bias |
| Vision self-attention | 12 heads ×64, non-causal | sequence 1024 | Attention + softmax + value product |
| Attention output | `self_attn.out_proj.{weight,bias}` | `[768,768]` | Projection + residual |
| Post-norm MLP | `layer_norm2.{weight,bias}`, `mlp.fc1/2.{weight,bias}` | FC1 `[3072,768]`, FC2 `[768,3072]` | LayerNorm, MatMul, GELU, MatMul, residual |
| Final norm | `post_layernorm.{weight,bias}` | `[768]` | LayerNorm |
| Pixel shuffle | config `scale_factor=4` | `32×32×768` → `8×8×12288` | Exact permutation/reshape; keep data on HTP graph |
| Modality projection | `connector.modality_projection.proj.weight` | `[576,12288]` | MatMul to text hidden width |

Vision metadata confirms 12 layers, intermediate width 3072, tile size 512,
patch 16, scale factor 4, and 64 connector/image tokens per tile. The image
processor permits a 2048 longest edge and the current path adds a global
thumbnail; therefore the number of local tiles is input dependent and every
local tile plus the global thumbnail contributes 64 visual rows. Vision
attention is non-causal, uses LayerNorm epsilon `1e-6`, and its Q/K/V, output,
and MLP projections have biases. The checkpoint contains a distinct connector
weight of shape `[576,12288]` and no connector bias.

Image decode, resize/crop/tile selection, RGB normalization, text tokenization,
and generated-token decoding are CPU preprocessing/postprocessing. Vision
patch projection onward, including pixel shuffle and connector projection, is
model execution and must remain on HTP.

## Logic and known integration gaps

- Existing `forward_fp16_gpu` lays out Q/K/V as row-major packed head features,
  rotates Q/K at `cache->length`, appends K/V to each layer cache, then performs
  causal attention for multi-token prefill and non-causal single-row decode.
  The QNN path must preserve those semantics and prove prefill/decode parity.
- Image generation builds local tile row/column markers and a global-image
  marker; each image marker must consume exactly one connector feature row.
  Existing code validates image-token occurrence count, which the Android path
  must retain. The local tokenizer assigns `<image>` ID 49190,
  `<fake_token_around_image>` ID 49189, and `<global-img>` ID 49152; the
  configured vocabulary is 49280.
- Current `tensor_compute_preferred_backend()` only selects CPU/Metal/CUDA.
  Several existing reference paths can fall back to CPU operators. They must
  not be reused as the QNN inference dispatch path unless a strict mode turns
  every missing callback into a hard failure.
- `tensor_sdpa_f16()` is not a QNN attention implementation. The SDK needs an
  explicit QNN HTP attention graph or fused HTP op; relying on the host tensor
  buffer path would violate the no-model-op-fallback rule.
- Model load and execution ownership currently live in test-only SmolVLM C
  code. The reusable SDK runner must own QNN graphs, weight staging, cache, and
  teardown; JNI should only pass files/inputs and receive status/results.

## Stage-0 checks still required on hardware

1. Check the installed QNN HTP compiler supports each candidate FP16 op, rank,
   broadcast, and graph size. Compilation success alone is not numerical proof.
2. Compare one output for every primitive with the CPU reference, then compare
   decoder/vision layer boundaries and final token IDs.
3. Validate cache positions and image-token/feature-row mapping with tiny
   synthetic cases before running the full checkpoint.
