# Metal Attention Optimization

## Goal

Replace the decoder's separate `QKᵀ → scale/mask → softmax → AV` operations with a fused
scaled-dot-product-attention (SDPA) Metal kernel where the backend supports the input tensors.
The optimization is based on operator capabilities, not a specific model configuration.

The fused kernel keeps K/V in `[batch, kv_heads, context, head_dim]` form. It maps each query
head to its KV head inside the kernel, so it does not materialize repeated K/V tensors. QK,
stable softmax, and AV are fused, reducing intermediate tensors and command submissions.

## Kernel selection

`src/ops/metal/sdpa.rs` calls SDPA kernels bundled with `candle-metal-kernels`:

| Input shape | Metal kernel path | Causal behavior |
|---|---|---|
| One query token, context `< 1024` | Vector SDPA, one pass | No explicit mask needed: all cached keys precede the current token |
| One query token, context `≥ 1024` | Vector SDPA, two pass | Same decode property; pass 1 computes block partials and pass 2 combines them |
| Multiple query tokens | Tiled/full SDPA | Causal mask is applied by the fused kernel; the Q/K length offset handles cached prefixes |

The current dispatch threshold and kernel variants mirror the Candle SDPA API and the measured
PyTorch MPS path. The Metal two-pass path was measured at context length 1024. Other contexts and
prefill shapes should be benchmarked on the target Mac before making performance claims.

## Capability checks and fallback

`supports_fused_sdpa` checks tensor properties rather than `DecoderSpec`: rank/shape compatibility,
GQA head divisibility, contiguous layouts, matching dtype, and the head dimensions instantiated by
the bundled Metal kernels. The vector kernels support head dimensions 32, 64, 96, 128, and 256;
the tiled kernel additionally supports 72 and 80. Supported dtypes are F32, F16, and BF16.

Compatible Metal inputs use fused SDPA automatically. CPU inputs and Metal inputs outside the
kernel's supported shapes/layouts/dtypes keep the generic attention implementation as a fallback.
The fallback is intentional: the decoder remains model-agnostic and can still run shapes for which
the current fused kernels have no implementation.

## Benchmarking

The Rust microbenchmark compares the generic repeated-K/V reference with the same fused Metal
operator used by the decoder. It uses deterministic FP32 inputs and reports maximum absolute output
difference as well as average iteration time. Arguments are:

```text
cargo run --example gqa_benchmark -- <cpu|metal> <context> <query_len> <iterations> <warmup>
```

Examples:

```sh
# Two-pass single-token decode (the measured case)
cargo run --example gqa_benchmark -- metal 1024 1 50 10

# One-pass single-token decode
cargo run --example gqa_benchmark -- metal 512 1 50 10

# Causal prefill, query length 16 and context length 128
cargo run --example gqa_benchmark -- metal 128 16 50 10
```

The PyTorch MPS operator comparison is available separately:

```sh
python3 examples/benchmark_gqa.py --device mps --seq 1 --context 1024 --warmup 10 --iterations 50
```

Do not compare CPU timings with Metal timings. For framework comparisons, keep device, dtype,
query/KV head counts, head dimension, query length, context length, warm-up, and synchronization
consistent. The operator benchmark is not a full-model latency test.

## Recorded result

On the tested Mac mini, for FP32 `q_heads=9`, `kv_heads=3`, `head_dim=64`, `seq=1`, and
`context=1024`:

| Rust/Candle Metal implementation | Time per iteration | Max absolute difference vs. generic reference |
|---|---:|---:|
| Repeated K/V + separate attention ops | 2.271 ms | Reference |
| Fused SDPA, two pass | 0.175 ms | `0.00000060` |

This is about a 13× operator-level speedup for that shape. The earlier per-query-head no-repeat
variant took 2.716 ms and was removed because it was slower than the reference. Python/PyTorch MPS
on the same shape reported 0.606 ms for repeated-K/V separate ops and 0.559 ms for SDPA; these are
framework-specific measurements and should not be treated as a direct implementation comparison.

## Correctness and rollout

The fused-vs-reference difference above is small, but it is not proof that every greedy token is
unchanged: a small logit perturbation can change a token when candidate logits are nearly tied.
After changing a fused kernel or its dispatch, validate both:

1. Operator output error on each relevant shape (decode below/above 1024 and causal prefill).
2. Full-model generated tokens and completion on the same prompt/image, dtype, and token limit.

The 1024-token decode result is verified on Metal. The newly wired one-pass short-decode and tiled
prefill routes compile, but still need target-Mac timing and full-model output checks.

## Packed RHS for decoder linear prefill

PyTorch MPS uses `MPSNDArrayMatrixMultiplication` for contiguous linear layers and describes
the transposed weight with packed rows. Candle's Metal matmul uses its MLX GEMM kernel instead.
The decoder therefore retains the original row-major weights (used by single-row GEMV) and, for
FP32 Metal weights with output width 768 through 4096, makes a contiguous transposed copy for
prefill. The packed copy is selected only for inputs with at least 32 rows. Other widths keep the
original matmul route; this avoids the regressions measured for width 576.

For the sample decoder's FP32 `rows=88` shapes, the Mac mini benchmark reported:

| Projection | Original RHS | Contiguous packed RHS | Max absolute error |
|---|---:|---:|---:|
| QKV, `[88,576] x [576,960]` | 0.237 ms | 0.224 ms | 0 |
| Attention output, `[88,576] x [576,576]` | 0.198 ms | 0.211 ms | 0 |
| Gate/up, `[88,576] x [576,3072]` | 0.395 ms | 0.354 ms | 0 |
| Down, `[88,1536] x [1536,576]` | 0.245 ms | 0.279 ms | 0 |

The implementation packs only widths that improved in this sample (960 and 3072), leaving
attention output and down on the original path. The packed weights consume additional model memory;
recheck this policy against other model sizes and full-model prefill latency before widening it.
