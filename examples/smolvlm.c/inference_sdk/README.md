# Inference SDK

A standalone C11 inference runtime for reusable tensor operations, attention,
weight loading, tokenization, image preprocessing, and benchmark workloads.
Model architecture and application code belong in the consuming project.

## Build

The default build uses portable C implementations and backend stubs:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Choose a GPU backend when its toolchain is installed:

```sh
cmake -S . -B build-metal -DINFERENCE_SDK_BACKEND=METAL
cmake --build build-metal

cmake -S . -B build-cuda -DINFERENCE_SDK_BACKEND=CUDA
cmake --build build-cuda
```

Metal requires macOS, Xcode's Metal toolchain, and the Metal frameworks. Set
`TENSOR_METALLIB_PATH` and `TENSOR_KERNELS_METALLIB_PATH` to the generated
libraries when running a client outside the corresponding build directory.
CUDA requires Linux, a CUDA toolkit, and CMake's CUDAToolkit package.

The default image decoder and tokenizer are implemented in C. To use the Rust
image decoder and Hugging Face tokenizer instead, install Rust/Cargo and build
with:

```sh
cmake -S . -B build-rust -DINFERENCE_SDK_IMAGE_DECODER=RUST
cmake --build build-rust
ctest --test-dir build-rust --output-on-failure
```

This Rust option also works with the Linux CUDA backend. On macOS with Metal,
the build embeds the generated Metal libraries into the Rust static library.
Cargo downloads the pinned dependencies in `rust_decoder/Cargo.lock` if they
are not already cached.

The default allocator is the system C allocator. To compile the vendored
mimalloc static allocator into the SDK, enable:

```sh
cmake -S . -B build-mimalloc -DINFERENCE_SDK_ALLOCATOR=MIMALLOC
cmake --build build-mimalloc
```

The allocator can be combined with either image/tokenizer implementation and
with CPU, Metal, or CUDA builds.

## Use as a library

Link `InferenceSDK::InferenceSDK` from CMake:

```cmake
add_subdirectory(path/to/inference_sdk)
target_link_libraries(my_model PRIVATE InferenceSDK::InferenceSDK)
```

The public headers are in `include/`. A client defines model-specific
`prepare`, `run`, and `cleanup` callbacks and passes them to the generic
Benchmark API. For example:

```c
#include "benchmark.h"

BenchmarkWorkload workload = {
    .name = "decoder_step",
    .unit = "token",
    .context = decoder_context,
    .prepare = prepare_decoder,
    .run = run_decoder,
    .cleanup = cleanup_decoder,
};

BenchmarkConfig config = {
    .warmup_iterations = 3,
    .measured_iterations = 20,
    .platform = "macOS",
    .backend = "metal",
    .device = "selected-device",
    .synchronize = synchronize_device,
};

BenchmarkSession *session = NULL;
char error[256] = {0};
if (benchmark_session_create(&config, &workload, &session,
                              error, sizeof(error)) &&
    benchmark_session_run(session, error, sizeof(error))) {
    const BenchmarkReport *report = benchmark_session_report(session);
    /* Consume latency_ms, work_units, and throughput_per_second here. */
}
benchmark_session_destroy(session);
```

`run` reports actual completed work units. The SDK measures with a monotonic
host clock by default and invokes the optional synchronization callback before
and after every measured iteration. Set `clock_ms` to supply a device clock or
a deterministic clock for tests. Prepare and cleanup are outside the timed
region. JSON reports are allocated with `malloc`; release them with `free`.

## Layout

- `include/`: model-independent public C API
- `src/common/`: tensor, attention, storage, tokenizer, image, and benchmark code
- `src/platform/`: CPU stubs, Metal, and CUDA backend implementations
- `tests/`: SDK-level validation
- `third_party/`: bundled stb image decoder used by the default image loader
- `third_party/mimalloc/`: vendored mimalloc source, headers, and license
- `vendor/`: Metal attention shader dependency
