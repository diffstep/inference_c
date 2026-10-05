"""Benchmark copy/contiguous and pointwise operations on PyTorch MPS or CPU.

Run alongside the Candle benchmark with identical shape/iteration arguments:
  python3 examples/benchmark_memory_ops.py --device mps --rows 88 --cols 576
  cargo run --example memory_ops_benchmark -- metal 88 576 50 10
"""

import argparse
import time

import torch


def make_tensor(rows: int, cols: int, phase: float, device: torch.device) -> torch.Tensor:
    values = [
        ((i * 17) % 101 - 50) * 0.01 + phase
        for i in range(rows * cols)
    ]
    return torch.tensor(values, dtype=torch.float32).reshape(rows, cols).to(device)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def benchmark(name, fn, device, warmup, iterations):
    with torch.inference_mode():
        for _ in range(warmup):
            output = fn()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            output = fn()
        synchronize(device)
    ms = (time.perf_counter() - start) * 1000 / iterations
    print(f"{name}: {ms:.3f} ms/iter, sum={output.float().sum().item():.8f}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--rows", type=int, default=88)
    parser.add_argument("--cols", type=int, default=576)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this Python environment")
    device = torch.device(args.device)
    x = make_tensor(args.rows, args.cols, 0.125, device)
    y = make_tensor(args.rows, args.cols, 0.375, device)
    transposed = x.transpose(0, 1)
    scalar = torch.tensor(0.25, dtype=torch.float32, device=device)
    print(f"device={device}, dtype=torch.float32, rows={args.rows}, cols={args.cols}, "
          f"iterations={args.iterations}, warmup={args.warmup}")

    benchmark("contiguous_forced_copy", lambda: x.clone(), device, args.warmup, args.iterations)
    benchmark("transpose_contiguous_copy", lambda: transposed.contiguous(), device, args.warmup, args.iterations)
    benchmark("elementwise_add", lambda: x + y, device, args.warmup, args.iterations)
    benchmark("elementwise_mul", lambda: x * y, device, args.warmup, args.iterations)
    benchmark("broadcast_scalar_add", lambda: x + scalar, device, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
