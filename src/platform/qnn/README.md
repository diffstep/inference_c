# QNN HTP platform layer

`qnn_runtime.cpp` owns the diagnostic QAIRT HTP smoke checks. `qnn_graph.cpp`
provides public reusable session and compiled fixed-shape FP16 linear,
Safetensors projection loader, RMSNorm, token embedding, split-half RoPE,
gated MLP, and GQA attention graph APIs through `include/qnn_htp.h`. A graph can
be compiled for multiple rows for prefill projections or one row for token
decode. A session retains the provider, backend, device, and context; graph
objects own copied static weights and a finalized graph across executions. The
Android app links the same CMake target exposed as
`InferenceSDK::QNNHTP`.

Current implemented checks:

- Resolve QAIRT's HTP provider, initialize the backend/device, and execute a
  known-answer ReLU graph repeatedly.
- Read `model.text_model.layers.0.self_attn.q_proj.weight` directly from a
  Safetensors file, convert F32 weights to FP16, execute a QNN MatMul graph,
  and compare outputs against a host reference.
- Build and reuse a single-token FP16 linear graph with static weights.
- Run a two-row FP16 RMSNorm graph and validate against a host FP32 reference.
- Run a fixed-batch INT32 token-ID to FP16 embedding Gather graph and check
  output ordering and out-of-range ID rejection.
- Run split-half RoPE with runtime position IDs against a host reference.
- Run a fused gated MLP graph: gate/up MatMul, SiLU, multiply, down MatMul, and
  residual add, checked against a host reference.
- Run causal GQA attention with KV-head expansion on HTP and verify a tiny
  deterministic output.
- Load the layer-0 `q_proj.weight` directly through the SDK Safetensors loader,
  then compare the HTP projection against a host reference.

Build the optional Android runtime with `INFERENCE_SDK_ENABLE_QNN_HTP=ON` and
`QAIRT_SDK_ROOT`, or consume the standalone `InferenceSDK::QNNHTP` target from
another Android CMake project. These are fixed-shape primitive/layer graphs,
not a complete runner, and are not wired into `TensorComputeBackend`. Full
checkpoint loading, vision layers, pixel shuffle/connector, end-to-end decoder
scheduling, cache management, and token generation remain to be implemented.
Graph compilation and numerical execution checks are part of the Android
startup probe and require a compatible HTP device; an NDK build alone does not
validate HTP operator support. These checks are correctness gates, not model
benchmarks.

Example root-SDK configuration with the Android NDK:

```sh
cmake -S . -B build/android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DINFERENCE_SDK_ENABLE_QNN_HTP=ON \
  -DQAIRT_SDK_ROOT="$HOME/qairt/2.50.0.260828"
cmake --build build/android --target inference_sdk_qnn_htp
```
