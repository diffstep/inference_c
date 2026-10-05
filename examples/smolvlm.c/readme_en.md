# SmolVLM-256M-Instruct Inference

This repository implements a lightweight inference application for
**HuggingFaceTB/SmolVLM-256M-Instruct**, following the Idefics3 architecture.
It supports image preprocessing, vision encoding, multimodal prompt assembly,
and text generation on macOS with Metal and Linux with CUDA.

`examples/sample-model` contains a chart-reading model fine-tuned from this
architecture. It is not the original Hugging Face base checkpoint.

## Platforms and backends

- **Target platforms:** macOS with Metal and Linux with CUDA.
- **Compute backend:** Set `TENSOR_BACKEND=metal` or `TENSOR_BACKEND=cuda` to
  select a GPU backend explicitly. `auto` selects an available backend that
  was built into the application.
- **Inference precision:** FP32 by default. Set `TENSOR_DTYPE=fp16` to convert
  FP32/BF16 safetensors weights to FP16 while loading. FP16 requires an
  available Metal or CUDA backend and does not fall back to CPU.
- **Image decoding:** The built-in `stb_image` decoder is used by default. Set
  `IMAGE_DECODER=rust` to use the Rust image decoder and Rust `tokenizers`.
- **Tokenizer:** With `IMAGE_DECODER=rust`, the C API delegates to Rust
  `tokenizers` for UTF-8 behavior consistent with Transformers. The default
  `IMAGE_DECODER=stb` setting keeps the pure C tokenizer.

## Requirements

- Metal builds require Xcode Command Line Tools and Apple's Metal Toolchain.
  Install the latter with `xcodebuild -downloadComponent MetalToolchain` if
  needed.
- Linux CUDA builds require the NVIDIA CUDA Toolkit (`nvcc`, CUDA Runtime, and
  cuBLAS). The build looks in `/usr/local/cuda` by default; set `CUDA_HOME` to
  use a different installation path.
- `IMAGE_DECODER=rust` requires Rust/Cargo. Cargo may need network access to
  download dependencies for `inference_sdk/rust_decoder`.

## Build the application

Run from the repository root:

```sh
make -B all ATTENTION_BACKEND=metal IMAGE_DECODER=rust
```

`-B` forces a rebuild of the C application, Rust static library, and Metal
kernels. Check the built executable with:

```sh
./build/smolvlm-c --version
```

For the pure C tokenizer and built-in image decoder:

```sh
make -B all ATTENTION_BACKEND=metal IMAGE_DECODER=stb
```

Build for Linux with CUDA:

```sh
make -B all ATTENTION_BACKEND=cuda IMAGE_DECODER=rust
```

For CUDA FP32 or FP16 inference, set `TENSOR_BACKEND=cuda`; add
`TENSOR_DTYPE=fp16` for FP16. The CUDA path executes supported operators on the
GPU. Metal-specific fused decode command-buffer execution is not used by the
CUDA path.

The root application can also select mimalloc:

```sh
make -B all ATTENTION_BACKEND=metal IMAGE_DECODER=rust ALLOCATOR=mimalloc
```

## Run

Run the chart-image example from the repository root:

```sh
TENSOR_BACKEND=metal ./build/smolvlm-c generate-image \
  ../examples/sample-model \
  ../examples/test-data/zxqh/sample5.jpg \
  600
```

The last argument is the maximum number of generated tokens. An optional
prompt can replace the default market-data extraction prompt:

```sh
TENSOR_BACKEND=metal ./build/smolvlm-c generate-image \
  ../examples/sample-model \
  ../examples/test-data/zxqh/sample5.jpg \
  600 "Read the market data in the image and return JSON only."
```

Text-only generation:

```sh
TENSOR_BACKEND=metal ./build/smolvlm-c generate \
  ../examples/sample-model "Say hello." 32
```

FP16 inference uses the same FP32 model files:

```sh
TENSOR_DTYPE=fp16 TENSOR_BACKEND=metal ./build/smolvlm-c generate \
  ../examples/sample-model "Say hello." 32
```

Set `TENSOR_BACKEND_VERBOSE=1` to inspect backend selection. Debug dumps and
per-operator timing are not needed for normal runs.

## Tests

Run the C tests from the repository root:

```sh
make test ATTENTION_BACKEND=metal IMAGE_DECODER=rust
```

The standalone SDK tokenizer test uses a bundled BPE fixture to check the C ABI
encode/decode round trip. To run it by itself:

```sh
make build/test-tokenizer ATTENTION_BACKEND=metal IMAGE_DECODER=rust
./build/test-tokenizer
```

## Project layout

- `inference_sdk/`: standalone, model-independent C inference SDK, including
  tensor operations, attention, Benchmark, image processing, tokenization, and
  Metal/CUDA backends.
- `inference_sdk/rust_decoder/`: optional Rust static library for image
  decoding, tokenizer FFI, and embedded Metal libraries.
- `inference_sdk/third_party/`: bundled `stb_image` and mimalloc sources.
- `src/app/`: SmolVLM model configuration, text/vision inference, and CLI.

The SmolVLM application layer targets this repository's SmolVLM/Idefics3 model
and is not a general Transformers replacement. Which operators execute on the
GPU depends on the selected backend and the implementation available for each
operator. The reusable runtime and platform interfaces live in `inference_sdk/`.
