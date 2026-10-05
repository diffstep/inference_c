# Running the Inference SDK

This guide lists the SDK's runtime environment variables, build-time options,
and the values an application should set for Metal or CUDA. For standalone
build commands and dependency details, see [`build.md`](build.md).

## Runtime environment variables

Set these in the environment of the application process before it calls the
SDK. Backend names are lowercase and case-sensitive.

| Variable | Values | Default | Purpose |
| --- | --- | --- | --- |
| `TENSOR_BACKEND` | `auto`, `metal`, `cuda` | `auto` | Selects the tensor compute backend. `auto` probes CUDA first, then Metal. The selected backend must have been included in the SDK build and be available on the machine. |
| `TENSOR_ATTENTION_BACKEND` | `auto`, `metal`, `cuda`, `cpu` | Follows `TENSOR_BACKEND`; otherwise `auto` | Selects the attention backend independently. `cpu` is available for regular attention operations; GPU decode-sequence APIs require a GPU attention backend. |
| `TENSOR_METALLIB_PATH` | File path | `build/attention.metallib` | Runtime path to `attention.metallib` for Metal builds that load shader files. |
| `TENSOR_KERNELS_METALLIB_PATH` | File path | `build/tensor_kernels.metallib` | Runtime path to `tensor_kernels.metallib` for Metal builds that load shader files. |
| `TENSOR_METAL_DECODE_GEMV` | `baseline`, `4simd` | `baseline` | Selects the Metal decode GEMV implementation when that kernel is available. `4simd` is currently intended for the FP32 decode path; unsupported combinations use the baseline kernel. |

### Metal library loading

For `INFERENCE_SDK_IMAGE_DECODER=STB`, the Metal backend loads metallib files
at runtime. Set both file paths when the files are outside the process working
directory's `build/` folder:

```sh
TENSOR_BACKEND=metal \
TENSOR_ATTENTION_BACKEND=metal \
TENSOR_METALLIB_PATH="$PWD/build-metal/metal/attention.metallib" \
TENSOR_KERNELS_METALLIB_PATH="$PWD/build-metal/metal/tensor_kernels.metallib" \
./your_application
```

For `METAL` + `RUST`, CMake embeds both metallibs into the Rust static library.
The two runtime path variables are ignored; build with the correct
`INFERENCE_SDK_METALLIB_DIR` instead. Do not define `EMBED_METALLIBS` yourself.

### CUDA

Build with `INFERENCE_SDK_BACKEND=CUDA`, then select CUDA at runtime:

```sh
TENSOR_BACKEND=cuda TENSOR_ATTENTION_BACKEND=cuda ./your_application
```

CUDA is currently supported on Linux. Metal library variables have no effect
on CUDA.

## Build-time SDK options

Pass these as CMake cache arguments when configuring the SDK. They are not
runtime environment variables.

| CMake option | Values | Default | Purpose |
| --- | --- | --- | --- |
| `INFERENCE_SDK_BACKEND` | `CPU`, `METAL`, `CUDA` | `CPU` | Chooses which platform implementation is compiled. `METAL` requires macOS; `CUDA` requires Linux and the CUDA Toolkit. |
| `INFERENCE_SDK_BUILD_TESTS` | `ON`, `OFF` | `ON` | Builds SDK test executables. |
| `INFERENCE_SDK_ENABLE_TIMING` | `ON`, `OFF` | `OFF` | Enables detailed timing instrumentation by defining `TENSOR_TIMING_ENABLE`. This is a compile-time setting, not an environment variable. |
| `INFERENCE_SDK_ENABLE_GGML` | `ON`, `OFF` | `OFF` | Builds the optional vendored ggml GPU adapter. Requires `METAL` or `CUDA`. Native SDK Q4_0/Q8_0 kernels do not require it. |
| `INFERENCE_SDK_IMAGE_DECODER` | `STB`, `RUST` | `STB` | Selects the bundled C decoder or the Rust decoder/tokenizer FFI. Metal + Rust embeds generated metallibs. |
| `INFERENCE_SDK_ALLOCATOR` | `SYSTEM`, `MIMALLOC` | `SYSTEM` | Selects the system allocator or vendored mimalloc. |
| `INFERENCE_SDK_METALLIB_DIR` | Directory path | `<build-dir>/metal` | Output directory for Metal libraries and input location used to embed them in Metal + Rust builds. |

Standard CMake toolchain options also apply, including `CMAKE_BUILD_TYPE`,
`CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER`, `CMAKE_C_FLAGS`,
`CMAKE_CXX_FLAGS`, `CMAKE_CUDA_COMPILER`, and `CMAKE_CUDA_ARCHITECTURES`.

Example Metal configuration:

```sh
cmake -S . -B build-metal \
  -DCMAKE_BUILD_TYPE=Release \
  -DINFERENCE_SDK_BACKEND=METAL \
  -DINFERENCE_SDK_IMAGE_DECODER=RUST \
  -DINFERENCE_SDK_ENABLE_TIMING=ON
cmake --build build-metal --parallel
```

## Application-specific settings

The SDK library itself does not read model-specific settings such as
`TENSOR_DTYPE` or `TENSOR_WEIGHT_QUANT`; those belong to the model runner.
SmolVLM's build and benchmark options are documented in
[`tests/smolvlm/build.md`](tests/smolvlm/build.md).
