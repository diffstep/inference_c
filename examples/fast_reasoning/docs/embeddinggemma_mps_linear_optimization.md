# EmbeddingGemma Metal 线性层优化记录

## 范围与现状

本记录覆盖 `google/embeddinggemma-300m`（Gemma3 text encoder）在 Apple Metal、FP32 下的
线性投影优化实验。它与 Idefics3 decoder 的 packed-RHS 实验是不同路径，结果不可互相外推。

PyTorch 2.10 的 MPS `F.linear` 在满足连续布局等条件时使用
`MPSNDArrayMatrixMultiplication`，并将转置权重描述为 packed rows。Candle Metal 的默认
矩阵乘则走 MLX GEMM。实验为 Candle 增加了一个使用同一 Metal command buffer 的 MPSNDArray
custom op，以避免每个投影单独提交并等待。

当前这条路径仍是**显式 opt-in**，默认推理保持 Candle MLX：

| 环境变量 | 作用范围 |
|---|---|
| `FAST_REASONING_MPS_NDARRAY_QKV=1` | 只替换 Gemma3/EmbeddingGemma 的 QKV 投影 |
| `FAST_REASONING_MPS_NDARRAY_ALL_LINEAR=1` | 替换该编码器中的 QKV、attention output、gate/up、down 投影 |

两条路径都只对 Metal FP32 启用；其他设备/dtype 走原 Candle 路径。custom op 还要求连续的
二维矩阵；显式开启但输入布局不符或 MPSNDArray 不可用时会报错，而不是静默切换后端。
MPSNDArray 编码要求 macOS 15 或更新版本。已验证 19-token query 和 84-token document；
其他设备、系统版本、dtype 与更长输入仍需验证，因此暂不默认开启。

## 算子基准

测试条件：Mac mini，Metal/MPS，FP32，`rows=19`，50 次迭代、10 次 warmup；确定性输入，
比较 `input @ weight.T`。MPS custom op 与 Candle 的最大绝对误差均为 0。

| 投影形状（K×N） | Candle MLX | MPSNDArray Candle custom op | 加速比 |
|---|---:|---:|---:|
| q `[768, 768]` | 0.169 ms | 0.086 ms | 1.96× |
| k `[768, 256]` | 0.139 ms | 0.073 ms | 1.90× |
| v `[768, 256]` | 0.132 ms | 0.073 ms | 1.81× |
| o `[768, 768]` | 0.167 ms | 0.087 ms | 1.93× |
| gate `[768, 1152]` | 0.219 ms | 0.095 ms | 2.30× |
| up `[768, 1152]` | 0.221 ms | 0.097 ms | 2.29× |
| down `[1152, 768]` | 0.213 ms | 0.101 ms | 2.12× |

独立的原生 MPSNDArray 基准约快 2.2–3.7×，但它复用输入、权重和输出 buffer，不能直接
代表模型接入收益。每次算子都单独提交 command buffer 并等待时只有 Candle 的约 0.45–0.77×；
每个算子独立 encoder、但共用 command buffer 时仍快约 2.1–3.4×。结论是同步/批次边界
会主导测量，不能用单算子强制同步的数字推断整模型效果。

custom op 早期版本每次调用都查询默认 Metal device，CPU 提交耗时升至约 0.22–0.32 ms，
反而比 Candle 慢。改为从 Candle 当前 command buffer 取得其队列 device 后，提交耗时降至
约 0.02–0.03 ms；分项测量中的输出 buffer 分配约 0–0.001 ms、encoder 获取约 0.002–0.003 ms、
MPS 编码约 0.005–0.006 ms。MPS 运算被编码到 Candle 当前 encoder/command buffer，不做
每次算子级 CPU wait。

## 整模型对照

使用 `examples/compare_embeddinggemma.py`，PyTorch 2.10.0/MPS 与 Rust/Metal，
`attn_implementation=sdpa`，误差由最终 768 维 embedding 计算：

| 输入 | Rust 路径 | Rust 耗时 | Python 耗时 | 最大绝对误差 | cosine similarity |
|---|---|---:|---:|---:|---:|
| query，19 tokens | 默认 Candle，20 iterations | 42.107 ms/iter | 26.019 ms/iter | `1.78813934e-07` | `1.000000000` |
| query，19 tokens | 全部线性投影走 MPSNDArray，20 iterations | 38.084 ms/iter | 25.849 ms/iter | `1.78813934e-07` | `1.000000000` |
| document，84 tokens | 默认 Candle，20 iterations | 56.199 ms/iter | 37.907 ms/iter | `9.406358e-08` | `1.000000000` |
| document，84 tokens | 全部线性投影走 MPSNDArray，20 iterations | 52.360 ms/iter | 37.757 ms/iter | `9.406358e-08` | `1.000000000` |

同输入 A/B 中，全线性路径在 19-token query 节省 4.023 ms/iter（约 9.6%），在 84-token
document 节省 3.839 ms/iter（约 6.8%）。Rust 仍慢于 Python；这些数据只代表上述模型、设备、
dtype 和输入长度，不是跨模型或跨设备性能保证。此前 5-iteration query 测量方向一致
（41.906 → 38.212 ms/iter）。QKV-only query 测量为 40.841 ms/iter，收益较小；完整向量误差
与默认路径相同。

## Profile 的解释限制

`examples/embeddinggemma_profile.rs` 在阶段边界显式同步 Metal，属于诊断工具，不是端到端
延迟。10 次迭代的两组 profile 中，全线性路径的 MLP 汇总从 28.143 降到 23.731 ms，但
attention 和 other 汇总同时变高；这些分项不与端到端 A/B 一致，说明阶段同步会改变异步
提交/调度，不能据此判断全线性路径整体变慢。需要决策时以相同设置下的端到端对照为准。

短输入已经满足 `masked_sdpa` 的 Metal/F32/连续布局能力条件（19 tokens、head_dim 256），
因此当前注意力路径使用 fused masked SDPA；不要把结果误判为回退到重复 KV 的通用 attention。

## 复测命令

设置本地模型路径后，使用相同 prompt、warmup 和 iteration 数做 A/B：

```sh
MODEL_DIR="/path/to/embeddinggemma-300m"

python3 examples/compare_embeddinggemma.py --offline --device mps \
  --attn-implementation sdpa --warmup 5 --iterations 20 --model "$MODEL_DIR"

FAST_REASONING_MPS_NDARRAY_ALL_LINEAR=1 \
python3 examples/compare_embeddinggemma.py --offline --device mps \
  --attn-implementation sdpa --warmup 5 --iterations 20 --model "$MODEL_DIR"
```

微基准（也会报告提交、同步分项及 custom op 误差）：

```sh
cargo run --example mps_ndarray_matmul_benchmark -- 19 50 10
```

若扩大默认启用范围，还需在更多 document 长度、其他提示词、macOS 版本和 dtype 上验证，
并检查 MPSNDArray 不可用或输入布局不兼容时的安全回退策略。

## 实现位置

- `src/ops/metal/mps_ndarray_linear.rs`：通用 FP32 MPSNDArray Candle custom op。
- `src/models/gemma3/encoder.rs`：Gemma3/EmbeddingGemma 的 opt-in 路由；环境变量未设置时走 Candle。
- `src/ops/metal/mps_ndarray_bench.mm`、`examples/mps_ndarray_matmul_benchmark.rs`：原生与
  Candle custom op 基准。
- `vendor/candle-metal-kernels/`：本地 Candle Metal kernels 0.9.2 小型补丁，给当前 compute
  encoder 暴露底层 encoder/command-buffer handle，使 MPS 能复用 Candle 提交批次。
