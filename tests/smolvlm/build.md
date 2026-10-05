# SmolVLM build and benchmark

The Android camera / QNN bring-up project is documented separately in
[`android/README.md`](android/README.md). It currently verifies HTP runtime and
graph execution; SmolVLM inference on HTP is still under development.

This page documents the options accepted by [`build.sh`](build.sh). The script
configures CMake, builds the SDK and SmolVLM executable, and runs the image
benchmark by default.

## Quick start

On macOS with Metal:

```bash
./tests/smolvlm/build.sh
```

Build only, without loading the model or running the benchmark:

```bash
SMOLVLM_RUN_BENCHMARK=OFF ./tests/smolvlm/build.sh
```

Build and benchmark with 4-bit weights, a 256-token limit, one warmup, and three
measured runs:

```bash
SMOLVLM_WEIGHT_QUANT_CASES=q4_0 \
SMOLVLM_MAX_TOKEN_CASES=256 \
SMOLVLM_WARMUP_RUNS=1 \
SMOLVLM_BENCHMARK_RUNS=3 \
./tests/smolvlm/build.sh
```

On Linux, set `SMOLVLM_BACKEND=CUDA`; the script rejects a non-CUDA backend on
Linux. CUDA builds require a CUDA-capable toolchain and the CUDA libraries used
by the SDK.

## Build configuration variables

Unset variables use the defaults shown below. `SMOLVLM_CMAKE_ARGS` is appended
to the CMake configure command; its value is parsed as space-separated
arguments.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SMOLVLM_BACKEND` | `METAL` | Build backend. macOS requires `METAL`; Linux requires `CUDA`. |
| `SMOLVLM_IMAGE_DECODER` | `RUST` | Image/tokenizer implementation: `RUST` or `STB`. |
| `SMOLVLM_ALLOCATOR` | `SYSTEM` | Allocator: `SYSTEM` or `MIMALLOC`. |
| `SMOLVLM_ENABLE_GGML` | `OFF` | Enable the optional ggml GPU Q8_0 linear path. CPU backend is not supported with this option. |
| `SMOLVLM_BUILD_TYPE` | `Release` | CMake build type. |
| `SMOLVLM_TIMING` | `ON` | Add `TENSOR_TIMING_ENABLE` to the example and SDK build. Timing logs can affect measurements. |
| `SMOLVLM_CC` | `cc` | C compiler for the example and CMake SDK build. |
| `SMOLVLM_CXX` | `c++` | C++ compiler for CMake, used by optional C++ components such as ggml. |
| `SMOLVLM_CFLAGS` | `-O2` | Flags for the SmolVLM example C sources. |
| `SMOLVLM_CPPFLAGS` | empty | Preprocessor flags for the example C sources. |
| `SMOLVLM_LDFLAGS` | empty | Additional example linker flags. |
| `SMOLVLM_LDLIBS` | empty | Additional example libraries. |
| `SMOLVLM_SDK_CFLAGS` | empty | C compiler flags passed to CMake for SDK C sources. |
| `SMOLVLM_SDK_CXXFLAGS` | empty | C++ compiler flags passed to CMake. |
| `SMOLVLM_CMAKE_ARGS` | empty | Additional space-separated CMake arguments. |
| `SMOLVLM_RUSTFLAGS` | empty | Exported as Cargo's `RUSTFLAGS` when building the Rust decoder. |
| `SMOLVLM_BUILD_DIR` | `tests/smolvlm/build-metal-rust` | CMake build directory and default location for generated metallibs. |
| `SMOLVLM_EXECUTABLE` | `$SMOLVLM_BUILD_DIR/smolvlm` | Output executable path. |
| `SMOLVLM_RUN_BENCHMARK` | `ON` | Run the image benchmark after building; set to `OFF` to build only. |

## Benchmark variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `SMOLVLM_MODEL_DIR` | `tests/smolvlm/sample-model` | Directory containing `model.safetensors` and model configuration/tokenizer files. Required when running the benchmark. |
| `SMOLVLM_IMAGE_PATH` | `tests/smolvlm/test-data/statue-of-liberty.jpg` | Input image. Required when running the benchmark. |
| `SMOLVLM_PROMPT` | `Describe the image briefly.` | Prompt passed with the image. |
| `SMOLVLM_MAX_TOKEN_CASES` | `1024` | Whitespace-separated maximum new-token values; each value runs as a separate case. |
| `SMOLVLM_DTYPE_CASES` | `fp16` | Whitespace-separated activation/model compute dtypes: `fp16` or `fp32`. |
| `SMOLVLM_WEIGHT_QUANT_CASES` | `fp16` | Whitespace-separated weight formats: `fp16`, `q8_0`, or `q4_0`. Quantized weights require FP16 activations. Metal automatically uses cached GPU dequantization plus MPS FP16 GEMM for Q4/Q8 vision projections; CUDA uses its native quantized kernels. |
| `SMOLVLM_GEMV_VARIANTS` | `baseline` | Whitespace-separated Metal decode GEMV variants. `4simd` is currently available for the `fp32` decode path; `fp16` runs `baseline`. |
| `SMOLVLM_BENCHMARK_RUNS` | `1` | Number of measured runs per case, from 1 to 1000. |
| `SMOLVLM_WARMUP_RUNS` | `0` | Number of warmup runs per case, from 0 to 1000. |
| `SMOLVLM_BENCHMARK_FORMAT` | `text` | Report format: `text` or `json`. |
| `SMOLVLM_TENSOR_BACKEND` | Build platform backend | Runtime tensor backend: `metal` on macOS or `cuda` on Linux. |
| `SMOLVLM_ATTENTION_BACKEND` | `SMOLVLM_TENSOR_BACKEND` | Runtime attention backend. |

Example matrix:

```bash
SMOLVLM_DTYPE_CASES='fp16 fp32' \
SMOLVLM_WEIGHT_QUANT_CASES=fp16 \
SMOLVLM_GEMV_VARIANTS='baseline 4simd' \
SMOLVLM_MAX_TOKEN_CASES='128 512' \
SMOLVLM_BENCHMARK_RUNS=3 \
SMOLVLM_WARMUP_RUNS=1 \
./tests/smolvlm/build.sh
```

The script runs every valid dtype, quantization, GEMV, and token-limit
combination. Quantized weights with `fp32` are rejected. For `fp16`, the script
selects `baseline` GEMV because `4simd` is an FP32 decode variant. Q4_0 and
Q8_0 benchmark cases use fused vision and quantized accelerated decode when
the selected GPU backend exposes those kernels. For a short smoke run, set
`SMOLVLM_MAX_TOKEN_CASES=2` and select one quantization format, for example:

The Metal `gpu_wait` timing is the host wait for queued vision GPU work to
finish, so it includes execution time. About 1.3 seconds indicates the older
packed scalar GEMM path; the MPS FP16 GEMM path measured around 0.13 seconds per
tile on the sample model. No environment variable is needed to select MPS.

```bash
SMOLVLM_WEIGHT_QUANT_CASES=q8_0 SMOLVLM_MAX_TOKEN_CASES=2 \
  SMOLVLM_BENCHMARK_RUNS=1 SMOLVLM_WARMUP_RUNS=0 \
  ./tests/smolvlm/build.sh
```

## Compiler and CMake arguments

`build.sh` forwards the build variables to `tests/smolvlm/Makefile`. The
Makefile configures CMake with these SDK options:

| CMake option | Value supplied by the Makefile |
| --- | --- |
| `CMAKE_BUILD_TYPE` | `SMOLVLM_BUILD_TYPE` |
| `CMAKE_C_COMPILER` | `SMOLVLM_CC` |
| `CMAKE_CXX_COMPILER` | `SMOLVLM_CXX` |
| `CMAKE_C_FLAGS` | `SMOLVLM_SDK_CFLAGS` |
| `CMAKE_CXX_FLAGS` | `SMOLVLM_SDK_CXXFLAGS` |
| `INFERENCE_SDK_BACKEND` | `SMOLVLM_BACKEND` |
| `INFERENCE_SDK_METALLIB_DIR` | `$SMOLVLM_BUILD_DIR/metal` |
| `INFERENCE_SDK_BUILD_TESTS` | `OFF` |
| `INFERENCE_SDK_IMAGE_DECODER` | `SMOLVLM_IMAGE_DECODER` |
| `INFERENCE_SDK_ALLOCATOR` | `SMOLVLM_ALLOCATOR` |
| `INFERENCE_SDK_ENABLE_TIMING` | `SMOLVLM_TIMING` |
| `INFERENCE_SDK_ENABLE_GGML` | `SMOLVLM_ENABLE_GGML` |
| `EMBED_METALLIBS` compile definition | Automatically enabled for `METAL` + `RUST`; not a user-facing CMake option. |

For example, additional SDK optimization flags can be passed as:

```bash
SMOLVLM_SDK_CFLAGS='-O3 -ffast-math' \
SMOLVLM_CMAKE_ARGS='-DINFERENCE_SDK_METALLIB_DIR=/tmp/smolvlm-metal' \
SMOLVLM_RUN_BENCHMARK=OFF \
./tests/smolvlm/build.sh
```

The Makefile supplies `INFERENCE_SDK_METALLIB_DIR` explicitly as
`$SMOLVLM_BUILD_DIR/metal`. CMake writes `attention.metallib` and
`tensor_kernels.metallib` there. With `IMAGE_DECODER=RUST`, CMake passes this
directory to Cargo as the build-time `INFERENCE_SDK_METALLIB_DIR`; the Rust
decoder embeds both libraries into its static archive. The generated
metallibs are also Rust build dependencies, so changing a Metal kernel rebuilds
the archive that embeds it.

`EMBED_METALLIBS` is defined automatically on the SDK's Metal Objective-C
sources when both the Metal backend and Rust decoder are selected. In that
configuration, `metal_load_library()` reads the embedded bytes exported by the
Rust archive. Do not add `-DEMBED_METALLIBS` manually: defining the macro
without the Rust `embedded-metallibs` feature would leave those exported byte
functions unavailable at link time. With the `STB` decoder, the macro is not
defined and the Metal implementation loads libraries from runtime paths.

## Metallib paths and runtime environment

The build-time and runtime variables have different roles:

| Variable | Role |
| --- | --- |
| `INFERENCE_SDK_METALLIB_DIR` | CMake cache path used to generate Metal libraries and, for the Rust decoder, embed them into the Rust static library. Defaults to `${CMAKE_CURRENT_BINARY_DIR}/metal`; the SmolVLM Makefile sets it to `$SMOLVLM_BUILD_DIR/metal`. |
| `SMOLVLM_METALLIB_PATH` | Optional `build.sh` override for the runtime attention library path. |
| `SMOLVLM_TENSOR_METALLIB_PATH` | Optional `build.sh` override for the runtime tensor-kernel library path. |
| `TENSOR_METALLIB_PATH` | Runtime attention library path consumed by the Metal attention implementation. |
| `TENSOR_KERNELS_METALLIB_PATH` | Runtime tensor-kernel path consumed by the Metal compute implementation. |

The `SMOLVLM_*METALLIB_PATH` overrides are passed as runtime paths only for a
Metal build using the `STB` decoder. A Metal build using the default `RUST`
decoder loads the metallibs embedded during compilation; runtime metallib path
variables do not select those embedded files.

Other runtime values set by the benchmark script are `TENSOR_DTYPE`,
`TENSOR_WEIGHT_QUANT`, `TENSOR_BACKEND`, `TENSOR_ATTENTION_BACKEND`, and
`TENSOR_METAL_DECODE_GEMV`. Prefer the corresponding `SMOLVLM_*` variables
above to configure them through `build.sh`.

## Direct Makefile use

The Makefile can also be used without `build.sh`:

```bash
make -C tests/smolvlm \
  BACKEND=METAL IMAGE_DECODER=RUST TIMING=ON \
  SDK_BUILD_DIR="$PWD/tests/smolvlm/build-metal-rust" \
  TARGET="$PWD/tests/smolvlm/build-metal-rust/smolvlm"

make -C tests/smolvlm run \
  BACKEND=METAL IMAGE_DECODER=RUST \
  SDK_BUILD_DIR="$PWD/tests/smolvlm/build-metal-rust" \
  TARGET="$PWD/tests/smolvlm/build-metal-rust/smolvlm" \
  ARGS='--version'
```

Available Makefile settings are `BACKEND`, `IMAGE_DECODER`, `ALLOCATOR`,
`TIMING`, `ENABLE_GGML`, `BUILD_TYPE`, `CC`, `CXX`, `CFLAGS`, `CPPFLAGS`,
`LDFLAGS`, `LDLIBS`, `SDK_CFLAGS`, `SDK_CXXFLAGS`, `SDK_CMAKE_ARGS`,
`SDK_BUILD_DIR`, and `TARGET`. `make run` accepts the application command in
`ARGS`.
