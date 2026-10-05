# SmolVLM-256M-Instruct 本地推理实现

本项目面向 **HuggingFaceTB/SmolVLM-256M-Instruct**，实现 Idefics3 架构的轻量级视觉语言模型推理流程，支持 macOS Metal 和 Linux CUDA GPU。支持图像预处理、视觉编码、多模态提示词构建和文本生成。

仓库中的 `examples/sample-model` 是基于该架构微调的市场图表读取模型，不是 Hugging Face 发布的原始基座权重。

## 平台与后端

- **目标平台：** macOS Metal，Linux CUDA。
- **计算后端：** `TENSOR_BACKEND=metal` 或 `TENSOR_BACKEND=cuda` 显式选择 GPU；`auto` 自动选择可用的已构建 GPU 后端。
- **推理精度：** 默认 FP32。设置 `TENSOR_DTYPE=fp16` 可将 FP32/BF16 safetensors 权重在加载时流式转换为 FP16。FP16 要求可用的 Metal 或 CUDA 后端，不支持 CPU 回退。
- **图像解码：** 默认使用内置 `stb_image`；也可通过 `IMAGE_DECODER=rust` 使用 Rust 图像解码和 Rust `tokenizers`。
- **Tokenizer：** `IMAGE_DECODER=rust` 时，C 接口委托给 Rust `tokenizers`，以保持与 Transformers/Rust tokenizer 一致的 UTF-8 编码、解码行为；默认 `IMAGE_DECODER=stb` 保留纯 C tokenizer 实现。

## 环境要求

- macOS Metal 构建需要 Xcode Command Line Tools 和 Apple Metal Toolchain。若 Metal Toolchain 尚未安装，可执行 `xcodebuild -downloadComponent MetalToolchain`。
- Linux CUDA 构建需要 NVIDIA CUDA Toolkit（`nvcc`、CUDA Runtime 和 cuBLAS）；默认查找 `/usr/local/cuda`，也可通过 `CUDA_HOME` 指定安装目录。
- 使用 `IMAGE_DECODER=rust` 时，需要 Rust/Cargo，以及 Cargo 下载 `inference_sdk/rust_decoder` 依赖所需的网络访问。

## 构建

在 `smolvlm_c` 目录执行：

```sh
make -B all ATTENTION_BACKEND=metal IMAGE_DECODER=rust
```

`-B` 会强制重新构建 C 程序、Rust 静态库和 Metal kernel。构建完成后检查版本：

```sh
./build/smolvlm-c --version
```

如需使用纯 C tokenizer 和内置图像解码器：

```sh
make -B all ATTENTION_BACKEND=metal IMAGE_DECODER=stb
```

Linux CUDA 构建：

```sh
make -B all ATTENTION_BACKEND=cuda IMAGE_DECODER=rust
```

运行 CUDA FP32 或 FP16 推理时分别设置 `TENSOR_BACKEND=cuda`，FP16 另加 `TENSOR_DTYPE=fp16`。CUDA 路径使用 GPU 算子逐层/逐步执行；Metal 专用的融合解码 command buffer 尚未移植。

## 运行

从 `smolvlm_c` 目录运行市场图表图像示例：

```sh
TENSOR_BACKEND=metal ./build/smolvlm-c generate-image \
  ../examples/sample-model \
  ../examples/test-data/zxqh/sample5.jpg \
  600
```

最后一个参数是最大生成 token 数。也可以通过可选提示词覆盖默认的市场数据提取提示词：

```sh
TENSOR_BACKEND=metal ./build/smolvlm-c generate-image \
  ../examples/sample-model \
  ../examples/test-data/zxqh/sample5.jpg \
  600 "请读取图片中的市场数据，只返回 JSON。"
```

纯文本生成示例：

```sh
TENSOR_BACKEND=metal ./build/smolvlm-c generate \
  ../examples/sample-model "Say hello." 32
```

FP16 推理使用同一份 FP32 模型文件：

```sh
TENSOR_DTYPE=fp16 TENSOR_BACKEND=metal ./build/smolvlm-c generate \
  ../examples/sample-model "Say hello." 32
```

如果要检查后端选择信息，可设置 `TENSOR_BACKEND_VERBOSE=1`。常规运行不需要开启调试转储或逐算子计时。

## 测试

在 `smolvlm_c` 目录运行 C 测试：

```sh
make test ATTENTION_BACKEND=metal IMAGE_DECODER=rust
```

Rust tokenizer 的独立 SDK 测试会用内置 BPE fixture 检查 C ABI 编码/解码往返。若只想单独运行 tokenizer 测试：

```sh
make build/test-tokenizer ATTENTION_BACKEND=metal IMAGE_DECODER=rust
./build/test-tokenizer
```

## 项目结构

- `inference_sdk/`：可独立编译的通用 C 推理 SDK，包含张量、attention、Benchmark、图像处理、tokenizer、Metal/CUDA 后端，以及可选 Rust 和 mimalloc 依赖。
- `src/app/`：模型配置、视觉/文本推理流程和命令行入口。
- `inference_sdk/rust_decoder/`：供 SDK 调用的 Rust 静态库，包括可选图像解码、tokenizer FFI 和嵌入式 Metal library 接口。
- `inference_sdk/third_party/`：SDK 使用的 stb_image 与 mimalloc 源码。

通用 SDK 可脱离 SmolVLM 工程单独构建：

```sh
cmake -S inference_sdk -B inference_sdk/build
cmake --build inference_sdk/build
ctest --test-dir inference_sdk/build --output-on-failure
```

启用 Rust tokenizer/图像解码或 mimalloc 时，分别传入
`-DINFERENCE_SDK_IMAGE_DECODER=RUST` 和 `-DINFERENCE_SDK_ALLOCATOR=MIMALLOC`。

## 当前范围

SmolVLM 应用层面向本仓库的 SmolVLM/Idefics3 模型与测试流程，不是通用 Transformers 替代品。通用推理能力、Benchmark 和平台后端接口位于 `inference_sdk/`；具体算子是否使用 GPU，取决于所选后端和该算子的实现。
