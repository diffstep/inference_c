# Inference C SDK 特性

Inference C SDK 是一个面向可复用推理基础能力的 C11 库。它提供张量运算、模型输入处理和性能测量等通用组件，模型结构与应用逻辑由接入 SDK 的项目负责。

## 核心能力

- **C11 公共 API**：通过 `include/` 中的头文件访问张量、注意力、存储、分词、图像处理和基准测试接口。
- **多计算后端**：默认构建使用可移植 C 实现和后端桩；可按平台选择 CPU、Apple Metal 或 NVIDIA CUDA 后端。
- **张量与注意力运算**：提供通用张量操作和注意力相关实现，并可根据目标平台构建对应 GPU 内核。
- **量化线性层**：Metal 和 CUDA 后端支持 GGML 兼容的 Q4_0、Q8_0 权重量化线性运算，使用 FP16 激活；该路径要求张量驻留在设备端，且不提供 CPU 回退。
- **模型输入处理**：默认提供 C 实现的图像解码和分词器，也可选择 Rust 图像解码及 Hugging Face 分词实现。
- **基准测试 API**：可为自定义工作负载配置预热、测量迭代、设备同步与时钟回调，并输出 JSON 或文本报告，包含延迟和吞吐量指标。

## 构建与集成

- **CMake 集成**：可通过 `add_subdirectory()` 引入，并链接 `InferenceSDK::InferenceSDK`。
- **可选组件**：支持可选的 ggml 适配器，以及可选的 mimalloc 静态分配器。
- **Metal 库管理**：构建时生成 Metal 库；Rust 构建可以将库嵌入静态库，非 Rust 构建可通过环境变量指定库路径。
- **模型示例**：仓库包含 SmolVLM 等示例与工具，演示 SDK 组件在模型工作负载中的使用；这些示例不等同于通用模型架构 API。

## 平台要求

- **CPU**：使用默认可移植 C 实现。
- **Metal**：需要 macOS、Xcode Metal 工具链及 Metal 框架。
- **CUDA**：需要 Linux、CUDA Toolkit 和 CMake 的 CUDAToolkit 包。
- **Rust 可选路径**：需要 Rust/Cargo；Cargo 可能需要下载锁定的依赖。

## 文档

构建选项和示例请参阅 [README.md](README.md)、[build.md](build.md) 和 [run_sdk.md](run_sdk.md)。

---

# Inference C SDK Features

Inference C SDK is a C11 library for reusable inference building blocks. It provides common components for tensor operations, model input processing, and performance measurement. Model architectures and application logic remain the responsibility of the projects that use the SDK.

## Core capabilities

- **C11 public API**: Access tensor, attention, storage, tokenization, image processing, and benchmarking interfaces through the headers in `include/`.
- **Multiple compute backends**: The default build uses portable C implementations and backend stubs. Select the CPU, Apple Metal, or NVIDIA CUDA backend for the target platform.
- **Tensor and attention operations**: Includes general tensor operations and attention implementations, with GPU kernels available through the corresponding platform backends.
- **Quantized linear layers**: The Metal and CUDA backends support GGML-compatible Q4_0 and Q8_0 weight-only linear operations with FP16 activations. Tensors must reside on the device for these operations; there is no CPU fallback.
- **Model input processing**: The default image decoder and tokenizer are implemented in C. Optional Rust implementations provide image decoding and Hugging Face tokenization.
- **Benchmarking API**: Configure warmup and measured iterations, device synchronization, and clock callbacks for custom workloads. Generate JSON or text reports with latency and throughput metrics.

## Build and integration

- **CMake integration**: Include the SDK with `add_subdirectory()` and link against `InferenceSDK::InferenceSDK`.
- **Optional components**: Enable the ggml adapter or the mimalloc static allocator when needed.
- **Metal library handling**: Metal libraries are generated during the build. Rust builds can embed them in the static library; non-Rust builds can use environment variables to specify their paths.
- **Model examples**: The repository includes examples and tools, including SmolVLM, that demonstrate SDK components in model workloads. These examples are not a generic model architecture API.

## Platform requirements

- **CPU**: Uses the default portable C implementation.
- **Metal**: Requires macOS, the Xcode Metal toolchain, and the Metal frameworks.
- **CUDA**: Requires Linux, the CUDA Toolkit, and CMake's CUDAToolkit package.
- **Optional Rust path**: Requires Rust/Cargo; Cargo may need to download pinned dependencies.

## Documentation

See [README.md](README.md), [build.md](build.md), and [run_sdk.md](run_sdk.md) for build options and usage examples.
