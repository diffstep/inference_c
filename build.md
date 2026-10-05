# Inference SDK build guide

This document covers standalone builds of the model-independent SDK library.
It does not cover application-specific settings such as the SmolVLM model,
image prompt, token limits, or benchmark matrix; those belong with the example
in `tests/smolvlm/`.

## Requirements

All builds require CMake 3.18 or newer and a C11 compiler.

| Backend | Supported platform | Additional requirements |
| --- | --- | --- |
| `METAL` | macOS | Xcode Command Line Tools / `xcrun` with the Metal compiler and linker; Apple Metal frameworks. |
| `CUDA` | Linux | CUDA Toolkit with `nvcc`, CUDA runtime and cuBLAS libraries; CMake CUDAToolkit package. |
| `CPU` | macOS, Linux | Portable C build with GPU backend stubs. Use this for host-only SDK components and tests; it does not provide GPU inference. |

The SDK is a runtime library, not a model executable. Applications supply model
loading, architecture execution, and request handling.

## Quick builds

Build the default portable library and its standalone tests:

```bash
cmake -S . -B build-cpu \
  -DINFERENCE_SDK_BACKEND=CPU \
  -DINFERENCE_SDK_BUILD_TESTS=ON
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

Build the Metal backend on macOS:

```bash
cmake -S . -B build-metal \
  -DINFERENCE_SDK_BACKEND=METAL \
  -DINFERENCE_SDK_BUILD_TESTS=ON
cmake --build build-metal --parallel
ctest --test-dir build-metal --output-on-failure
```

Build the CUDA backend on Linux:

```bash
cmake -S . -B build-cuda \
  -DINFERENCE_SDK_BACKEND=CUDA \
  -DINFERENCE_SDK_BUILD_TESTS=ON
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```

Use a separate build directory for each backend/configuration. CMake caches the
backend and compiler settings in the build directory.

## CMake options

Pass options during `cmake -S . -B <build-dir>`. Set `-DNAME=value` to override
the listed default.

| Option | Default | Description |
| --- | --- | --- |
| `INFERENCE_SDK_BACKEND` | `CPU` | `CPU`, `METAL`, or `CUDA`. `METAL` is macOS-only; `CUDA` is Linux-only. |
| `INFERENCE_SDK_BUILD_TESTS` | `ON` | Build SDK test executables and register them with CTest. |
| `INFERENCE_SDK_ENABLE_TIMING` | `OFF` | Define `TENSOR_TIMING_ENABLE` for the SDK's detailed timing instrumentation. |
| `INFERENCE_SDK_ENABLE_GGML` | `OFF` | Build the optional vendored ggml GPU Q8_0 adapter. Requires `METAL` or `CUDA`; ggml's CPU backend is disabled by this configuration. |
| `INFERENCE_SDK_IMAGE_DECODER` | `STB` | `STB` uses the bundled C implementation. `RUST` builds the Rust image decoder and Hugging Face tokenizer FFI. |
| `INFERENCE_SDK_ALLOCATOR` | `SYSTEM` | `SYSTEM` uses the platform allocator. `MIMALLOC` compiles the vendored static mimalloc allocator into the SDK. |
| `INFERENCE_SDK_METALLIB_DIR` | `<build-dir>/metal` | Output directory for generated `attention.metallib` and `tensor_kernels.metallib`; also the source directory for Rust embedding in Metal + Rust builds. |

### Optional ggml adapter

Enable the ggml adapter with the same GPU backend used by the SDK:

```bash
# macOS
cmake -S . -B build-metal-ggml \
  -DINFERENCE_SDK_BACKEND=METAL \
  -DINFERENCE_SDK_ENABLE_GGML=ON
cmake --build build-metal-ggml --parallel

# Linux
cmake -S . -B build-cuda-ggml \
  -DINFERENCE_SDK_BACKEND=CUDA \
  -DINFERENCE_SDK_ENABLE_GGML=ON
cmake --build build-cuda-ggml --parallel
```

The optional ggml adapter is separate from the SDK's native Q4_0 and Q8_0
weight-only linear kernels. The native Metal and CUDA kernels are part of their
respective backend builds and use FP16 activations; they do not require
`INFERENCE_SDK_ENABLE_GGML=ON`. These are GPU linear-operation primitives;
their availability alone does not mean every model execution path has a fused
quantized implementation.

## Metal libraries and `EMBED_METALLIBS`

For a Metal build, CMake compiles the shader sources and writes:

```text
<build-dir>/metal/attention.metallib
<build-dir>/metal/tensor_kernels.metallib
```

`INFERENCE_SDK_METALLIB_DIR` controls that location. For example:

```bash
cmake -S . -B build-metal \
  -DINFERENCE_SDK_BACKEND=METAL \
  -DINFERENCE_SDK_METALLIB_DIR="$PWD/build-metal/shaders"
cmake --build build-metal --parallel
```

The Rust decoder changes how the Metal libraries are made available at runtime:

| Configuration | Build behavior | Runtime behavior |
| --- | --- | --- |
| `METAL` + `STB` | CMake builds metallibs in `INFERENCE_SDK_METALLIB_DIR`. | Metal loads libraries from files. Set `TENSOR_METALLIB_PATH` for the attention library and `TENSOR_KERNELS_METALLIB_PATH` for tensor kernels when using a non-default location. |
| `METAL` + `RUST` | CMake passes `INFERENCE_SDK_METALLIB_DIR` to Cargo and enables Cargo's `embedded-metallibs` feature. It also defines `EMBED_METALLIBS` for the Metal Objective-C sources. | Metal loads embedded bytes from the Rust static library; runtime metallib path variables are ignored. |

Do not set `EMBED_METALLIBS` manually. CMake defines it only for the Metal +
Rust combination, where the Rust static library exports the embedded shader
bytes. CMake also makes both generated metallibs dependencies of that Rust
archive, so changing a shader rebuilds the archive that embeds it.

The Rust option requires Rust/Cargo. Cargo uses the pinned dependencies in
`src/rust/decoder/Cargo.lock`; the first build may need to download dependencies
if they are not cached.

## Compiler and build flags

Use standard CMake compiler settings for toolchains and optimization:

```bash
cmake -S . -B build-metal \
  -DINFERENCE_SDK_BACKEND=METAL \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_C_FLAGS='-O3 -ffast-math'
cmake --build build-metal --parallel
```

Common CMake settings:

| Setting | Typical values | Purpose |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel` | Select standard single-configuration build flags. |
| `CMAKE_C_COMPILER` | `clang`, `gcc`, absolute compiler path | Select C compiler. |
| `CMAKE_CXX_COMPILER` | `clang++`, `g++`, absolute compiler path | Select C++ compiler; used by ggml. |
| `CMAKE_C_FLAGS` | e.g. `-O3 -ffast-math` | Extra C compiler flags. |
| `CMAKE_CXX_FLAGS` | e.g. `-O3` | Extra C++ compiler flags. |
| `CMAKE_CUDA_COMPILER` | `nvcc` or absolute path | Select CUDA compiler when configuring the CUDA backend. |
| `CMAKE_CUDA_ARCHITECTURES` | architecture numbers supported by the installed toolkit | Select CUDA GPU architectures to compile. |

## Runtime backend selection

The build selects which GPU implementation is compiled. Runtime dispatch can
also be controlled by environment variables:

| Variable | Values | Description |
| --- | --- | --- |
| `TENSOR_BACKEND` | `metal`, `cuda`, `auto` | Select tensor compute backend. If unset or `auto`, the SDK probes available backends. |
| `TENSOR_ATTENTION_BACKEND` | `metal`, `cuda`, `auto` | Select attention backend. If unset, attention dispatch follows the tensor backend selection. |
| `TENSOR_METALLIB_PATH` | filesystem path | Runtime path to `attention.metallib` for Metal builds that do not embed metallibs. |
| `TENSOR_KERNELS_METALLIB_PATH` | filesystem path | Runtime path to `tensor_kernels.metallib` for Metal builds that do not embed metallibs. |

The backend must be compiled into the SDK and available on the current device.
These runtime settings do not enable a backend that was omitted at build time.
Metal builds using Rust embedding use the compiled-in libraries and ignore the
two runtime path variables.

## Use from another CMake project

The simplest development integration is to add the SDK source tree and link
the exported in-tree target:

```cmake
add_subdirectory(path/to/inference_sdk)
target_link_libraries(my_model PRIVATE InferenceSDK::InferenceSDK)
```

The public C API headers are in `include/`. To create an install tree:

```bash
cmake --install build-metal --prefix "$PWD/stage"
```

The install includes the static SDK archive, public headers, and exported CMake
targets under `stage/lib/cmake/InferenceSDK/`.

## Build outputs

| Output | Location |
| --- | --- |
| Static SDK library | `<build-dir>/libinference_sdk.a` |
| Test executables and CTest metadata | `<build-dir>/` when `INFERENCE_SDK_BUILD_TESTS=ON` |
| Metal AIR and metallib files | `INFERENCE_SDK_METALLIB_DIR` |
| Rust decoder static library | `<build-dir>/cargo/release/libimage_decoder_ffi.a` when `INFERENCE_SDK_IMAGE_DECODER=RUST` |
