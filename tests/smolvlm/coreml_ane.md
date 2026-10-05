# SmolVLM Core ML and Metal hybrid check

This check builds the SmolVLM vision tower and connector as a fixed-shape Core
ML ML Program. The SDK rejects the model unless Core ML's compute plan assigns
every executable operation to the Neural Engine. Independent graph nodes run
in parallel, so this target checks two Core ML predictions alongside a Metal
F16 linear operation.

This first cut accepts an offline-converted Core ML partition as an explicit
Graph IR node. It does not yet extract Core ML partitions automatically from
arbitrary Graph IR operations or wire the hybrid node into the SmolVLM decoder
example.

The generated model package is intentionally ignored by source control. It is
about 180 MB and can be recreated from the local Hugging Face snapshot:

```sh
python3 tools/coreml/convert_smolvlm_vision.py \
  --model-dir /path/to/SmolVLM-256M-Instruct \
  --output tests/smolvlm/models/vision_ane.mlpackage
```

The converter requires PyTorch, Transformers, NumPy, and Core ML Tools. It
checks the manual vision graph against Hugging Face and the converted model
against the PyTorch reference before exiting successfully.

Build the macOS backend with Core ML enabled:

```sh
cmake -S . -B build-hybrid \
  -DINFERENCE_SDK_BACKEND=METAL \
  -DINFERENCE_SDK_ENABLE_COREML=ON \
  -DINFERENCE_SDK_BUILD_TESTS=ON
cmake --build build-hybrid --parallel
```

Run the hybrid runtime check:

```sh
INFERENCE_SDK_COREML_TEST_MODEL="$PWD/tests/smolvlm/models/vision_ane.mlpackage" \
TENSOR_KERNELS_METALLIB_PATH="$PWD/build-hybrid/metal/tensor_kernels.metallib" \
ctest --test-dir build-hybrid --output-on-failure -R '^test_tensor_coreml$'
```

`test_tensor_coreml` verifies model conversion/loading, ANE placement, finite
vision features, matching outputs from parallel ANE nodes, and the independent
Metal linear result. The Core ML model input is one normalized 512×512 RGB
tile in NCHW FP16; the output is 64×576 FP16 visual features.
