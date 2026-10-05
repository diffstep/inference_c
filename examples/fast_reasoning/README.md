# fast-reasoning

Rust/Candle 本地多模态推理引擎，重点是推理路径和 Apple Metal 算子优化。模型适配与设备 kernel 尽量分开；专用 kernel 按设备、dtype、形状和布局选择，不满足条件时回退到 Candle。

## 当前范例

当前验证的是 Idefics3/SmolVLM 兼容架构。`examples/sample-model` 是以 [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) 为基座、用匹配的图像/JSON 数据对微调的市场图表读取模型，不是原始基座权重。当前测试图像是 `examples/test-data/zxqh/sample5.jpg`。

## 运行

```sh
cargo build --release
FAST_REASONING_DEVICE=metal ./examples/test_generate_image.sh
```

也可以自行指定模型、图片、token 上限和提示词：

```sh
FAST_REASONING_DEVICE=metal ./target/release/fast-reasoning generate-image \
  examples/sample-model examples/test-data/zxqh/sample5.jpg \
  600 "请读取图片中的市场数据。"
```

设备设为 `metal` 或 `cpu`。`generate-image` 的 token 上限和提示词均可省略；省略上限时默认 2048，省略提示词时使用内置市场数据提示词。

## EmbeddingGemma

支持本地 `google/embeddinggemma-300m`（`model_type=gemma3_text`）的文本向量推理。查询默认使用检索 query 指令；文档向量使用 `title: none` 前缀：

```sh
MODEL_DIR="/Users/dengtao2nd/.cache/huggingface/hub/models--google--embeddinggemma-300m/snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
FAST_REASONING_DEVICE=metal cargo run -- embed "$MODEL_DIR" --task query "要检索的问题"
FAST_REASONING_DEVICE=metal cargo run -- embed "$MODEL_DIR" --task document "要入库的文本"
```

首次接入时以 Python/Transformers CPU 结果做完整向量对照：

```sh
cargo build
FAST_REASONING_DEVICE=metal python3 examples/compare_embeddinggemma.py --offline --device mps --model \
  "/Users/dengtao2nd/.cache/huggingface/hub/models--google--embeddinggemma-300m/snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
```

## 增加范例模型

- **同一架构、另一组权重**：增加独立的模型目录和对应测试数据，使用现有模型适配器验证。
- **新架构**：在 `src/models/<架构>/` 增加配置、权重映射及模型适配，并在 `src/models/mod.rs` 注册架构名。入口 `src/main.rs`、公共权重运行时和通用算子目录不需要因接入新模型而改动；只有缺少可复用算子能力时，才扩展公共计算模块。不能仅放入权重就假定引擎能够运行。

推理命令接收模型目录路径，因此不同权重不需要写死到算子实现中。后续整理多个权重时，建议统一命名为 `models/<模型名>/`；当前 `examples/sample-model` 是现有脚本使用的范例路径。

## 代码与性能测试

- `src/decoder/`：模型无关的文本解码原语与 KV cache。
- `src/ops/`：模型共享的优化算子与后端 dispatch；`src/ops/metal/`：可复用的 Metal kernels（具体语义限制由算子能力检查保证）。
- `src/models/idefics3/`：Idefics3 配置、权重映射、CLI、视觉编码及多模态适配；`src/vision/`：可复用的图像张量与 Transformer 视觉算子。
- `examples/*.rs`、`examples/*.py`：可运行的 Rust/PyTorch 算子基准。
- `docs/`：优化记录、测量条件、误差与回退范围。

示例：

```sh
cargo run --example memory_ops_benchmark -- metal 88 576 50 10
python3 examples/benchmark_memory_ops.py --device mps --rows 88 --cols 576 --iterations 50 --warmup 10
```

微基准不是整模型耗时。每次优化都应保持设备、dtype、形状和迭代设置一致，并检查算子误差及完整模型输出。

当前 `sample5.jpg` 图像推理的端到端实测（FP32，Mac mini，约数）：

| 实现 | 耗时 |
|---|---:|
| Python 原版模型推理 | 约 35–40 秒 |
| 优化后的 Rust/Metal 推理引擎 | 约 20 秒 |

这次测量约为 1.8–2 倍提速。属于当前测试记录，不是跨模型或跨设备的性能保证；严格复测时应固定图片、提示词、生成 token 数和设备，并记录运行条件。

详细记录：[Metal attention](docs/metal_attention_optimization.md)、[Metal pointwise/copy](docs/metal_pointwise_optimization.md)、
[EmbeddingGemma MPS 线性层](docs/embeddinggemma_mps_linear_optimization.md)。

### EmbeddingGemma MPS 线性层实测

Mac mini、PyTorch 2.10.0/MPS 与 Rust/Metal、FP32、SDPA；20 次迭代。启用
`FAST_REASONING_MPS_NDARRAY_ALL_LINEAR=1` 后，Gemma3/EmbeddingGemma 的线性投影使用
MPSNDArray custom op：

| 输入 | Rust 默认 Candle | Rust 全线性 MPSNDArray | Python MPS | Rust 加速 | 最大绝对误差 | cosine similarity |
|---|---:|---:|---:|---:|---:|---:|
| query，19 tokens | 42.107 ms | 38.084 ms | 25.849 ms | 9.6% | `1.78813934e-07` | `1.000000000` |
| document，84 tokens | 56.199 ms | 52.360 ms | 37.757 ms | 6.8% | `9.406358e-08` | `1.000000000` |

两种输入均保持 embedding 数值一致性，Rust 分别减少约 4.023 ms 和 3.839 ms。此优化目前
默认关闭；只在 Apple Metal、FP32 下显式开启，且要求 macOS 15 或更新版本。较长输入、其他
设备和 dtype 尚未覆盖，不应将这组结果视为通用性能保证。测试方法及算子级数据见
[EmbeddingGemma MPS 线性层优化记录](docs/embeddinggemma_mps_linear_optimization.md)。

### EmbeddingGemma MPS 长序列 mask 实测

Mac mini、PyTorch 2.10.0/MPS 与 Rust/Metal、FP32；EmbeddingGemma document 输入 1129 tokens，warmup 2 次、计时 5 次。双向注意力配置中的序列化 `sliding_window=512` 会由 Transformers 转换为有效窗口 257；Rust 已同步该转换，并将 mask 预展开到 query heads，避免 Candle Metal 的通用 strided broadcast kernel。

| 路径 | Rust 耗时 | Python MPS 耗时 | 最大绝对误差 | cosine similarity |
|---|---:|---:|---:|---:|
| mask 广播 A/B | 527.840 ms | 约 441 ms | `9.35979187e-08` | `1.000000000` |
| mask 预展开 A/B | 447.937 ms | 约 442 ms | `9.35979187e-08` | `1.000000000` |
| 最终预展开实现复测 | 448.862 ms | 439.824 ms | `9.35979187e-08` | `1.000000000` |

A/B 中预展开使 Rust 耗时降低约 79.9 ms（15.1%）；最终实现与 Python MPS 耗时接近。广播版 A/B 使用了临时测试开关，当前代码默认并固定使用预展开实现。

复跑当前实现的端到端比较：

```sh
MODEL_DIR="/Users/dengtao2nd/.cache/huggingface/hub/models--google--embeddinggemma-300m/snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
TEXT="$(python3 -c 'print("The current price is near resistance at 4,250. " * 80, end="")')"
cargo build --release
python3 examples/compare_embeddinggemma_system.py \
  --model "$MODEL_DIR" --device mps --task document \
  --text "$TEXT" --warmup 2 --iterations 5
```

复跑 attention block 布局微基准（该命令比较广播与预展开加法，不代表整模型耗时）：

```sh
python3 examples/benchmark_embeddinggemma_attention_blocks.py --device mps --seq 122 --warmup 5 --iterations 20
cargo run --release --example embeddinggemma_attention_blocks -- metal 122 5 20
```
