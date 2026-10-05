"""Microbenchmark equivalent GQA attention paths on PyTorch MPS or CPU.

Run from the repository root, e.g.:
  python3 examples/benchmark_gqa.py --seq 1 --context 1024 --device mps

This is an operator-level comparison, not a full-model benchmark. It compares
materialized repeat_kv attention against PyTorch SDPA. Both paths use the same
tensors and causal mask.
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def make_mask(query_len: int, key_len: int, past: int, device: torch.device) -> torch.Tensor:
    q_pos = past + torch.arange(query_len, device=device)[:, None]
    k_pos = torch.arange(key_len, device=device)[None, :]
    return torch.zeros((1, 1, query_len, key_len), device=device).masked_fill(
        k_pos > q_pos, float("-inf")
    )


def make_input(shape: tuple[int, int, int, int], phase: float, device: torch.device) -> torch.Tensor:
    # Match examples/gqa_benchmark.rs; construct on CPU then explicitly move to
    # the selected device so all operator work is performed on that device.
    count = math.prod(shape)
    values = [
        math.sin(i * 0.013 + phase) * 0.5 + math.cos(i * 0.007 + phase) * 0.25
        for i in range(count)
    ]
    return torch.tensor(values, dtype=torch.float32).reshape(shape).to(device)


def repeat_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, groups: int,
                     mask: torch.Tensor) -> torch.Tensor:
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    return torch.matmul(torch.softmax(scores + mask, dim=-1), v)


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   groups: int, mask: torch.Tensor) -> torch.Tensor:
    # SDPA currently gets explicit, repeated K/V so this also measures its fused
    # attention implementation without conflating GQA support/dispatch behavior.
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)


def benchmark(fn, device: torch.device, warmup: int, iterations: int) -> tuple[torch.Tensor, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            result = fn()
        sync(device)
        start = time.perf_counter()
        for _ in range(iterations):
            result = fn()
        sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
    return result, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--q-heads", type=int, default=9)
    parser.add_argument("--kv-heads", type=int, default=3)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seq", type=int, default=1, help="query length (1 for decode)")
    parser.add_argument("--context", type=int, default=1024, help="total KV length")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    if args.q_heads % args.kv_heads:
        parser.error("--q-heads must be divisible by --kv-heads")
    if args.context < args.seq:
        parser.error("--context must be >= --seq")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this Python environment")

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    batch, groups = 1, args.q_heads // args.kv_heads
    past = args.context - args.seq
    q = make_input((batch, args.q_heads, args.seq, args.head_dim), 0.1, device).to(dtype=dtype)
    k = make_input((batch, args.kv_heads, args.context, args.head_dim), 0.2, device).to(dtype=dtype)
    v = make_input((batch, args.kv_heads, args.context, args.head_dim), 0.3, device).to(dtype=dtype)
    mask = make_mask(args.seq, args.context, past, device)

    funcs = {
        "repeat+matmul/softmax/matmul": lambda: repeat_attention(q, k, v, groups, mask),
        "SDPA + repeated KV": lambda: sdpa_attention(q, k, v, groups, mask),
    }
    results = {}
    print(f"device={device}, dtype={dtype}, q_heads={args.q_heads}, kv_heads={args.kv_heads}, "
          f"seq={args.seq}, context={args.context}, dim={args.head_dim}")
    for name, fn in funcs.items():
        output, elapsed_ms = benchmark(fn, device, args.warmup, args.iterations)
        results[name] = output.float()
        print(f"{name}: {elapsed_ms:.3f} ms/iteration")
    reference = results["repeat+matmul/softmax/matmul"]
    for name, output in results.items():
        error = (reference - output).abs()
        print(
            f"max_abs_error({name}): {error.max().item():.8g}; "
            f"sum={output.sum().item():.8g}; first8={output.flatten()[:8].tolist()}"
        )


if __name__ == "__main__":
    main()
