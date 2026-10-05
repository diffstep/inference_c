"""Compare PyTorch MPS linear projections at model-relevant shapes.

Run from the repository root:
  python3 examples/benchmark_matmul.py --device mps --rows 1
  python3 examples/benchmark_matmul.py --device mps --rows 88
  python3 examples/benchmark_matmul.py --device mps --preset embeddinggemma --rows 19

Inputs are deterministic FP32 tensors constructed on CPU and explicitly moved
to the selected device. The embeddinggemma preset uses the same dimensions and
values as embeddinggemma_matmul_benchmark.rs. This measures operators, not
full-model latency.
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F


CASES = (
    ("qkv_projection", 576, 960, 0.125),
    ("attention_output", 576, 576, 0.25),
    ("mlp_gate_up", 576, 3072, 0.375),
    ("mlp_down", 1536, 576, 0.5),
)

EMBEDDINGGEMMA_CASES = (
    ("q_proj", 768, 768, 0.125),
    ("k_proj", 768, 256, 0.1875),
    ("v_proj", 768, 256, 0.25),
    ("o_proj", 768, 768, 0.3125),
    ("gate_proj", 768, 1152, 0.375),
    ("up_proj", 768, 1152, 0.4375),
    ("down_proj", 1152, 768, 0.5),
)


def make_tensor(rows: int, cols: int, phase: float, device: torch.device) -> torch.Tensor:
    values = [
        math.sin(i * 0.013 + phase) * 0.5 + math.cos(i * 0.007 + phase) * 0.25
        for i in range(rows * cols)
    ]
    return torch.tensor(values, dtype=torch.float32).reshape(rows, cols).to(device)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument(
        "--preset",
        choices=("decoder", "embeddinggemma"),
        default="decoder",
        help="projection shapes to benchmark",
    )
    parser.add_argument("--rows", type=int, default=1, help="1 for decode; prompt token count for prefill")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this Python environment")

    device = torch.device(args.device)
    print(f"device={device}, dtype=torch.float32, rows={args.rows}, iterations={args.iterations}, warmup={args.warmup}")
    with torch.inference_mode():
        cases = EMBEDDINGGEMMA_CASES if args.preset == "embeddinggemma" else CASES
        for name, in_features, out_features, phase in cases:
            input = make_tensor(args.rows, in_features, 0.0625, device)
            weight = make_tensor(out_features, in_features, phase, device)
            run = lambda: F.linear(input, weight)
            for _ in range(args.warmup):
                output = run()
            synchronize(device)
            start = time.perf_counter()
            for _ in range(args.iterations):
                output = run()
            synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000 / args.iterations
            flat = output.flatten()
            print(
                f"{name}: [{args.rows}, {in_features}] x [{out_features}, {in_features}] "
                f"-> [{args.rows}, {out_features}], {elapsed_ms:.3f} ms/iter, "
                f"sum={output.sum().item():.8f}, first8={flat[:8].tolist()}"
            )


if __name__ == "__main__":
    main()
