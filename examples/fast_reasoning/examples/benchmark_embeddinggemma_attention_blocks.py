"""Benchmark the unfused EmbeddingGemma attention blocks on PyTorch MPS/CPU.

Pair with:
  cargo run --release --example embeddinggemma_attention_blocks -- metal 122 5 20
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F


def make_tensor(shape: tuple[int, ...], phase: float, device: torch.device) -> torch.Tensor:
    count = math.prod(shape)
    values = [
        math.sin(index * 0.013 + phase) * 0.5 + math.cos(index * 0.007 + phase) * 0.25
        for index in range(count)
    ]
    return torch.tensor(values, dtype=torch.float32).reshape(shape).to(device)


def benchmark(name, operation, device, warmup: int, iterations: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            output = operation()
        if device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            output = operation()
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000 / iterations
        del output
    print(f"{name}: {elapsed_ms:.3f} ms/iter")
    return elapsed_ms


def rotate_half(input: torch.Tensor) -> torch.Tensor:
    first, second = input.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(input: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return input * cos + rotate_half(input) * sin


def rope_cache(sequence: int, head_dim: int, theta: float, device: torch.device):
    inverse_frequency = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(sequence, dtype=torch.float32)
    frequencies = torch.outer(positions, inverse_frequency)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos().to(device)[None, None], embedding.sin().to(device)[None, None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--seq", type=int, default=122)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.seq <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("seq and iterations must be positive; warmup must be non-negative")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this Python environment")

    device = torch.device(args.device)
    batch, query_heads, kv_heads, head_dim = 1, 3, 1, 256
    groups = query_heads // kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    query = make_tensor((batch, query_heads, args.seq, head_dim), 0.1, device)
    key = make_tensor((batch, kv_heads, args.seq, head_dim), 0.2, device)
    value = make_tensor((batch, kv_heads, args.seq, head_dim), 0.3, device)
    mask = torch.zeros((batch, 1, args.seq, args.seq), dtype=torch.float32, device=device)
    expanded_mask = torch.zeros(
        (batch, query_heads, args.seq, args.seq), dtype=torch.float32, device=device
    )

    key_repeated = key.repeat_interleave(groups, dim=1)
    value_repeated = value.repeat_interleave(groups, dim=1)

    def qk_matmul():
        return torch.matmul(query, key_repeated.transpose(-2, -1))

    scores = qk_matmul()
    scaled_scores = scores * scale
    masked_scores = scaled_scores + mask
    probabilities = torch.softmax(masked_scores, dim=-1)

    hidden_size = 768
    projection_input = make_tensor((args.seq, hidden_size), 0.4, device)
    q_weight = make_tensor((query_heads * head_dim, hidden_size), 0.5, device)
    k_weight = make_tensor((kv_heads * head_dim, hidden_size), 0.6, device)
    v_weight = make_tensor((kv_heads * head_dim, hidden_size), 0.7, device)
    packed_qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
    o_weight = make_tensor((hidden_size, hidden_size), 0.8, device)
    q_norm_weight = torch.ones((head_dim,), dtype=torch.float32, device=device)
    k_norm_weight = torch.ones((head_dim,), dtype=torch.float32, device=device)
    rope_cos, rope_sin = rope_cache(args.seq, head_dim, 1_000_000.0, device)

    def split_qkv_projection():
        projected_query = F.linear(projection_input, q_weight).view(1, args.seq, query_heads, head_dim).transpose(1, 2)
        projected_key = F.linear(projection_input, k_weight).view(1, args.seq, kv_heads, head_dim).transpose(1, 2)
        projected_value = F.linear(projection_input, v_weight).view(1, args.seq, kv_heads, head_dim).transpose(1, 2)
        return projected_query, projected_key, projected_value

    def packed_qkv_projection():
        projected = F.linear(projection_input, packed_qkv_weight)
        projected_query = projected[:, :query_heads * head_dim].view(1, args.seq, query_heads, head_dim).transpose(1, 2)
        key_offset = query_heads * head_dim
        projected_key = projected[:, key_offset:key_offset + kv_heads * head_dim].view(1, args.seq, kv_heads, head_dim).transpose(1, 2)
        value_offset = key_offset + kv_heads * head_dim
        projected_value = projected[:, value_offset:].view(1, args.seq, kv_heads, head_dim).transpose(1, 2)
        return projected_query, projected_key, projected_value

    projected_query, projected_key, _ = split_qkv_projection()

    def rms_norm(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        normalized = input.float() * torch.rsqrt(input.float().square().mean(-1, keepdim=True) + 1e-6)
        return (normalized * weight).to(input.dtype)

    def full_attention():
        repeated_key = key.repeat_interleave(groups, dim=1)
        repeated_value = value.repeat_interleave(groups, dim=1)
        logits = torch.matmul(query, repeated_key.transpose(-2, -1)) * scale
        weights = torch.softmax(logits + mask, dim=-1)
        return torch.matmul(weights, repeated_value)

    def full_attention_skip_all_valid_mask():
        repeated_key = key.repeat_interleave(groups, dim=1)
        repeated_value = value.repeat_interleave(groups, dim=1)
        logits = torch.matmul(query, repeated_key.transpose(-2, -1)) * scale
        weights = torch.softmax(logits, dim=-1)
        return torch.matmul(weights, repeated_value)

    print(
        f"device={device}, dtype=float32, batch={batch}, q_heads={query_heads}, "
        f"kv_heads={kv_heads}, seq={args.seq}, head_dim={head_dim}, "
        f"warmup={args.warmup}, iterations={args.iterations}"
    )
    print("blocks=KV repeat, QK matmul, scale, mask add, softmax, AV matmul")
    benchmark("repeat_K", lambda: key.repeat_interleave(groups, dim=1), device, args.warmup, args.iterations)
    benchmark("repeat_V", lambda: value.repeat_interleave(groups, dim=1), device, args.warmup, args.iterations)
    benchmark("QK_matmul", qk_matmul, device, args.warmup, args.iterations)
    benchmark("scale", lambda: scores * scale, device, args.warmup, args.iterations)
    benchmark("mask_add", lambda: scaled_scores + mask, device, args.warmup, args.iterations)
    benchmark(
        "mask_add_expanded_contiguous",
        lambda: scaled_scores + expanded_mask,
        device,
        args.warmup,
        args.iterations,
    )
    benchmark("softmax", lambda: torch.softmax(masked_scores, dim=-1), device, args.warmup, args.iterations)
    benchmark("AV_matmul", lambda: torch.matmul(probabilities, value_repeated), device, args.warmup, args.iterations)
    benchmark("full_manual_attention", full_attention, device, args.warmup, args.iterations)
    benchmark("full_bidirectional_skip_all_valid_mask", full_attention_skip_all_valid_mask, device, args.warmup, args.iterations)
    print("blocks=QKV projections, O projection, Q/K RMSNorm, Q/K RoPE")
    benchmark("QKV_split_projection_and_heads", split_qkv_projection, device, args.warmup, args.iterations)
    benchmark("QKV_packed_projection_and_heads", packed_qkv_projection, device, args.warmup, args.iterations)
    benchmark("O_projection", lambda: F.linear(projection_input, o_weight), device, args.warmup, args.iterations)
    benchmark("Q_RMSNorm", lambda: rms_norm(projected_query, q_norm_weight), device, args.warmup, args.iterations)
    benchmark("K_RMSNorm", lambda: rms_norm(projected_key, k_norm_weight), device, args.warmup, args.iterations)
    normalized_query = rms_norm(projected_query, q_norm_weight)
    normalized_key = rms_norm(projected_key, k_norm_weight)
    benchmark("Q_RoPE", lambda: apply_rope(normalized_query, rope_cos, rope_sin), device, args.warmup, args.iterations)
    benchmark("K_RoPE", lambda: apply_rope(normalized_key, rope_cos, rope_sin), device, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
