# Full SmolVLM Android HTP inference plan

## Goal and execution rules

Deliver camera-to-caption SmolVLM inference on Android using Qualcomm QNN HTP
for **every model operator**. CPU may perform image decode/resize/normalization,
tokenization, prompt formatting, sampling, and reference calculations used only
by validation. CPU must not be an execution fallback for embedding, vision,
connector, decoder, attention, MLP, or logits model operations. Unsupported
HTP operations and QNN errors must stop inference with a useful error.

The reusable runtime belongs under `src/platform/qnn/` and the public SDK
surface under `include/`. `tests/smolvlm/android/` is an integration example,
not the owner of QNN operators or model execution. Start with FP16, which
matches the existing Metal/CUDA paths and is represented by the local F32
checkpoint after deterministic conversion. Add lower precision only after the
FP16 graph is correct and the HTP operator supports it.

## Current state (2026-10-04)

- HTP v73 provider/device initialization and a ReLU graph were verified on the
  previously connected SM8735 phone.
- `InferenceSDK::QNNHTP` builds for Android. It owns a persistent HTP context
  and fixed-shape FP16 linear, RMSNorm, embedding, split-half RoPE, gated MLP,
  and causal GQA attention graphs. Android startup runs small known-answer
  comparisons before retaining the session for inference.
- The Android JNI entrypoint now calls the existing SmolVLM tokenizer, image
  preprocessing, vision encoder/connector, and text generation code. It links
  the selected Safetensors descriptor as `model.safetensors`, requires QNN for
  compute and attention, and fixes activation/weight dtype to FP16.
- The app packages model config and tokenizer metadata but not the checkpoint.
  The Android APK builds locally. The phone is currently unavailable, so full
  HTP graph execution, output consistency, repeated-run stability, and device
  performance are pending hardware verification.

## Phased implementation

## Progress ledger

- **Phase 0 — partial.** Added [`OPERATOR_MAP.md`](OPERATOR_MAP.md) and
  checked the local config, processor/tokenizer metadata, selected Safetensors
  shapes, and existing text/vision forward paths. The reference intermediate
  fixture and per-op HTP capability table are still outstanding.
- **Phase 1 — partial, primitive/layer foundation.** Persistent HTP session;
  fixed-shape linear, RMSNorm, embedding, RoPE, gated MLP, and attention graphs;
  plus direct F16/F32/BF16 Safetensors linear loading. Android startup contains
  tiny numerical consistency checks. They compile locally; HTP execution and
  numerical results still need device validation.
- **Phases 2–3 — implementation partial.** The existing SmolVLM text and vision
  forward paths now dispatch their required FP16 operators through QNN. The
  model graphs are still assembled as individual reusable SDK operator graphs;
  there is no fused whole-layer graph or HTP-resident KV cache yet.
- **Phase 4 — integration partial.** Android camera → image tiles → vision
  features → image prompt → greedy text generation → caption UI is connected.
  The tokenizer/config metadata is packaged and weights are opened from a
  selected seekable Safetensors descriptor. Cancellation, progress reporting,
  generic model-session API, and device execution checks remain outstanding.
- **Phase 5 — not started.** Numerical comparisons, repeated-run stability,
  logic instrumentation, memory profiling, and end-to-end device benchmarks
  remain required.

### Preliminary local reference capture

On 2026-10-04, the existing macOS Metal FP16 example ran the Statue of Liberty
image with prompt `Describe the image briefly.` and `max_new_tokens=1`. It
returned the first text piece ` In`; measured vision encode was 2034.9 ms and
text prefill was 1213.9 ms. The image is 1600×1067; the current preprocessing
snaps it to 4×3 local crops plus one global tile (13 vision passes). This is a
single-run path/performance reference only: it does not record intermediate
activations or logits and does not establish HTP correctness or Android speed.

### 0. Freeze the reference contract and operation map

1. Record the exact checkpoint config, tokenizer files/special tokens, prompt
   template, image resize/tile policy, and expected image-token expansion.
2. Use the existing SmolVLM implementation as the numerical reference. Make a
   deterministic small input fixture for vision and text, and record tensor
   shapes, dtype, logits, and greedy token IDs at model boundaries.
3. Trace every operation in `text_model.c`, `vision_model.c`, and the connector
   into an operator/weight-layout table. Include bias, normalization epsilon,
   RoPE indexing, GQA head mapping, causal mask, residual order, activation,
   pixel shuffle, and Safetensors tensor names.
4. Mark each operation as a native QNN op, a fused HTP graph sequence, or an
   unsupported blocker. Confirm operator dtype/rank/shape limits against the
   installed QAIRT headers and HTP compiler. Do not proceed with silent
   assumptions about layouts or broadcasting.

**Gate:** reference inputs and intermediate outputs are reproducible; every
model tensor has an explicit consumer and every model operation has an HTP
lowering or a tracked blocker.

### 1. Make the SDK graph/runtime layer model-capable

1. Extend the current linear-only API into reusable graph construction for
   static weights, runtime inputs/outputs, multiple nodes, and named graph I/O.
2. Add the required FP16 primitives and/or fused layer builders in dependency
   order: embedding/gather, linear with optional bias, RMSNorm/LayerNorm,
   RoPE, elementwise add/multiply/SiLU, GELU, reshape/transpose, softmax, and
   batched MatMul. Keep graph dimensions and buffer sizes checked.
3. Add explicit HTP graph/session lifecycle, ownership, execution status,
   profiling hooks, and error propagation. Avoid rebuilding a graph per token;
   compile model graphs once and reuse them.
4. Keep the SDK QNN target independently consumable and installable. Do not
   expose SmolVLM-only assumptions in the generic QNN API.

**Gate:** standalone operator graphs run known-answer inputs on HTP and match
host references within per-op tolerances; repeated execute does not rebuild or
leak graph resources; all failure paths return errors without CPU dispatch.

### 2. Implement the text decoder on HTP

Implement and validate in layer-sized increments:

1. Token embedding and final RMSNorm / LM head.
2. Per-layer input RMSNorm; Q/K/V projections and bias; RoPE; grouped-query
   causal attention; output projection and residual.
3. Post-attention RMSNorm; gate/up projections; SiLU gate multiplication;
   down projection and residual.
4. Prefill graph for a token block and decode graph for one token. Add HTP-backed
   KV-cache writes/reads with correct positions, capacity checks, and reset.
5. Resolve the exact weight layout and optional tied embedding/LM-head behavior
   from the checkpoint rather than assuming names or sharing.

**Gate per layer/path:** compare hidden states and logits with the reference;
verify causal isolation, RoPE positions, grouped KV-head reuse, residual
ordering, prefill/decode equivalence, and deterministic greedy token agreement.
Instrument dispatch so a missing HTP op is a hard error.

### 3. Implement vision encoder and multimodal connector on HTP

1. Lower patch embedding, position embeddings, pre-norm vision blocks,
   Q/K/V, bidirectional attention, output projection, post-norm MLP, and
   connector projection/pixel shuffle to HTP graphs.
2. Preserve the model's normalization epsilon, attention scale, tensor order,
   tile order, and exact image-token count.
3. Batch tiles where shapes permit. Avoid per-tile graph compilation and
   unnecessary host/device copies. Keep preprocessing on CPU, then stage the
   prepared tensor once for HTP execution.

**Gate:** compare patch embeddings, each selected vision block, connector
features, and final visual tokens against the reference. Confirm identical
feature shapes and image-token placement before text prefill.

### 4. Add the reusable model runner and connect camera capture

1. Add a generic SDK model/session entry point that owns config, tokenizer,
   Safetensors access, HTP graphs, weights, KV cache, and teardown. Keep model
   path and prompt outside the QNN platform layer.
2. Load the local Safetensors checkpoint without bundling the ~1 GB file in the
   APK. Convert/stage weights once; report missing, duplicate, wrong-shape, or
   unsupported-dtype tensors before graph execution.
3. Connect Android camera JPEG → CPU preprocessing → vision HTP → connector
   HTP → text prefill/decode HTP → CPU tokenizer decode → caption UI.
4. Add cancellation, background/foreground handling, progress/error state, and
   serialized inference ownership so UI callbacks cannot race session teardown.

**Gate:** a fixed image and prompt produce the reference caption/token IDs on
device, then a captured image produces a visible caption. QNN traces show all
model graph executions on HTP; no model-op fallback is present.

### 5. Run consistency, stability, logic, and performance qualification

**Consistency**

- Compare deterministic intermediate tensors at embedding, each decoder and
  vision block, connector, logits, and generated tokens.
- Report max absolute/relative error, cosine similarity where meaningful, and
  greedy token agreement. Use explicit FP16 tolerances and investigate drift by
  first failing layer; do not loosen tolerances to hide layout bugs.
- Check repeated execution gives stable outputs and that prefill followed by
  decode agrees with the reference full-sequence path.

**Stability**

- Exercise cold/warm start, repeated image inference, repeated model
  open/close, cancellation, app backgrounding, malformed/missing Safetensors,
  invalid tensor shapes/dtypes, HTP unavailable, graph compile failure,
  execution failure, and memory pressure.
- Check every partial-init failure releases files, buffers, graphs, context,
  device, and backend. Ensure no use-after-free between JNI/UI and native work.
- Run repeated inference under sanitizers or host-side lifecycle fakes where
  available; on-device repeat and memory checks are required before release.

**Logic**

- Assert config invariants: hidden/head dimensions, GQA divisibility, layer
  counts, vocabulary bounds, image-token IDs, patch/tile geometry, and tensor
  shapes against loaded weights.
- Check causal masks prevent future-token influence, RoPE uses absolute cache
  positions, cache reset isolates images, image tokens replace exactly the
  expected prompt span, and tokenizer special tokens round-trip as expected.
- Verify every model operation increments HTP execution/profiling counters;
  deliberate unsupported ops must fail visibly. Search and audit all model
  paths for CPU fallback calls.

**Performance**

- Record cold model load/weight staging/graph compilation separately from warm
  vision prefill, text prefill, and per-token decode.
- Measure HTP graph time, host submission time, synchronization/wait time,
  memory peak, and tokens/s. Compare fused vs unfused graph submission only
  after output equivalence passes.
- Set device-specific baselines after the first complete HTP run. Do not label
  tiny operator smoke timings as end-to-end inference performance.

**Release gate:** all logic and numerical gates pass; repeated lifecycle is
clean; there is no CPU model-op execution; full camera-to-caption inference
works on the target phone; and performance numbers include cold/warm splits.

## Work order and current next action

Proceed 0 → 1 → 2 → 3 → 4 → 5. The full camera-to-caption code path is now
connected and the APK compiles. Continue with tensor-name/shape auditing,
deterministic intermediate/logit fixtures, HTP capability and graph-memory
review, then run the startup known-answer checks and complete caption path on a
Qualcomm Android device. Do not mark numerical consistency, stability, or
performance gates complete based on host/APK builds alone.
