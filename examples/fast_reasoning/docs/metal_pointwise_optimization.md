# Metal Pointwise and Copy Optimization

## Scope

This note records the operator-level comparison of copy/contiguous and pointwise
operations on Candle Metal and PyTorch MPS, plus the guarded F32 Metal kernels
implemented for contiguous elementwise operations.

These are microbenchmarks, not end-to-end model latency measurements. Each
measurement includes framework-side tensor allocation, command encoding and
submission, and synchronization around the iteration batch. It does not report
kernel-only time.

## Benchmark

The Rust/Candle benchmark is:

```sh
cargo run --example memory_ops_benchmark -- metal 88 576 50 10
```

The equivalent PyTorch MPS benchmark is:

```sh
python3 examples/benchmark_memory_ops.py --device mps --rows 88 --cols 576 --iterations 50 --warmup 10
```

Both use FP32 deterministic inputs of shape `[88, 576]`, 50 timed iterations,
and 10 warmups. The transposed-copy case materializes a contiguous copy from a
transpose. Results below were measured on the same Mac mini; small run-to-run
variation is expected.

## Recorded operator results

| Operation | Candle Metal | PyTorch MPS | Candle variant / error |
|---|---:|---:|---|
| Contiguous actual copy | 0.036 ms | 0.011 ms | Candle uses `force_contiguous()`; do not use `Tensor::copy()` for this comparison because Candle Metal's storage clone shares the buffer |
| Transpose then contiguous copy | 0.167 ms | 0.022 ms | — |
| Same-shape elementwise add | 0.080 ms | 0.012 ms | Custom F32 Metal path: 0.044 ms; max abs error 0 |
| Same-shape elementwise multiply | 0.057 ms | 0.012 ms | Custom F32 Metal path: 0.046 ms; max abs error 0 |
| Scalar broadcast add | 0.146 ms | 0.010 ms | `affine(1, scalar)`: 0.055 ms; custom F32 Metal path: 0.043 ms; both max abs error 0 |

The add/multiply fast paths improved the tested Candle operator times by about
1.8x and 1.2x, respectively. The scalar-add kernel improved the generic
broadcast path by about 3.4x in this run. They remained slower than PyTorch MPS
on this shape. These ratios apply only to this shape, dtype, device, and
measurement method; do not extrapolate them to all tensors or full-model speed.

## Implementation and dispatch guards

`src/ops/metal/pointwise.rs` and `pointwise.metal` contain contiguous F32
vectorized kernels. A thread processes four adjacent values using `float4`; a
tail path handles element counts not divisible by four.

The add/multiply wrappers use the custom kernel only when both inputs are on
Metal, have dtype F32, have equal shapes, and are contiguous. Other devices,
dtypes, shapes, and layouts use Candle's normal `add`/`mul` implementation.
Empty tensors also stay on the generic path. The custom Candle ops do not
implement gradients; they are intended for inference.

The decoder's two residual additions call these guarded wrappers. The SwiGLU
multiply also calls the wrapper, but it uses the custom path only if its
post-SiLU tensor and `up` tensor both satisfy the contiguity and shape guards;
otherwise it falls back without inserting a `.contiguous()` copy.

The scalar-add kernel remains a benchmarked hook and is not routed into model
inference. The current model has no identified rank-0 scalar broadcast-add
callsite that would benefit from it.

## Correctness and rollout status

For `[88, 576]` FP32 test inputs, custom add, multiply, and scalar-add outputs
matched their Candle references exactly in the reported run (`max_abs_error =
0`). Rust/Candle and PyTorch sums also matched closely; tiny FP32 reduction or
operation-order differences can occur between frameworks.

`cargo check` and CPU fallback smoke tests passed. CPU tests do not compile or
execute the Metal kernels. The kernels were exercised by the reported Mac MPS
benchmark, but the decoder integration still requires a full-model output and
latency regression check on the target Mac. If generated output changes or
latency regresses, disable the guarded routing before broadening the kernel's
shape/dtype coverage.

## Useful follow-up

- Measure several larger and smaller shapes to separate dispatch overhead from
  memory-throughput effects.
- Profile the full decoder with Instruments to see whether residual add is a
  material fraction of total time.
- Benchmark non-contiguous pointwise inputs separately before considering a
  strided kernel; avoid forcing contiguity unless its copy cost is included.
