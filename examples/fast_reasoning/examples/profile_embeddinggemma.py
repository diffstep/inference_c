#!/usr/bin/env python3
"""Diagnostic per-layer profile for EmbeddingGemma on CPU or MPS."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from embed_embeddinggemma import QUERY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/embeddinggemma-300m")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--task", choices=("query", "document"), default="query")
    parser.add_argument("--text", default=QUERY)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be non-negative and iterations must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")

    model_path = args.model
    if not Path(model_path).is_dir():
        model_path = snapshot_download(model_path, local_files_only=args.offline)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.offline)
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.float32,
        local_files_only=args.offline,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()
    text = (
        f"task: search result | query: {args.text}"
        if args.task == "query"
        else f"title: none | text: {args.text}"
    )
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    batch = {key: value.to(args.device) for key, value in batch.items()}

    def synchronize() -> None:
        if args.device == "mps":
            torch.mps.synchronize()

    def forward() -> None:
        model(**batch, return_dict=True)

    with torch.inference_mode():
        for _ in range(args.warmup):
            forward()
        synchronize()

        blocks = [
            (name, module)
            for name, module in model.named_modules()
            if hasattr(module, "self_attn") and hasattr(module, "mlp")
        ]
        if not blocks:
            parser.error("could not locate Transformer blocks with self_attn and mlp modules")

        samples: dict[str, list[float]] = {}
        handles = []

        def hook(name: str):
            def before(_module, _inputs):
                synchronize()
                _module._profile_started = time.perf_counter()

            def after(_module, _inputs, _output):
                synchronize()
                elapsed = (time.perf_counter() - _module._profile_started) * 1000
                samples.setdefault(name, []).append(elapsed)

            return before, after

        for index, (name, block) in enumerate(blocks):
            before, after = hook(f"layer_{index:02d}.whole")
            handles.extend((block.register_forward_pre_hook(before), block.register_forward_hook(after)))
            for attr, label in (("self_attn", "attention"), ("mlp", "mlp")):
                module = getattr(block, attr)
                before, after = hook(f"layer_{index:02d}.{label}")
                handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))

        for _ in range(args.iterations):
            forward()
        synchronize()
        for handle in handles:
            handle.remove()

    print(f"device={args.device}, dtype=f32, attention={args.attn_implementation}, tokens={batch['input_ids'].shape[1]}, iterations={args.iterations}")
    print("profile_notice=hooks synchronize Metal at each module boundary; diagnostic only, not end-to-end latency")
    layer_whole = []
    attention_total = 0.0
    mlp_total = 0.0
    other_total = 0.0
    for index in range(len(blocks)):
        whole = sum(samples.get(f"layer_{index:02d}.whole", [])) / args.iterations
        attention = sum(samples.get(f"layer_{index:02d}.attention", [])) / args.iterations
        mlp = sum(samples.get(f"layer_{index:02d}.mlp", [])) / args.iterations
        layer_whole.append(whole)
        attention_total += attention
        mlp_total += mlp
        other = max(whole - attention - mlp, 0.0)
        other_total += other
        print(f"layer_{index:02d}: whole={whole:.3f} ms, self_attn={attention:.3f} ms, mlp={mlp:.3f} ms, other={other:.3f} ms")
    print(f"layer_whole_total: {sum(layer_whole):.3f} ms/iter")
    print(f"self_attn_modules_total: {attention_total:.3f} ms/iter")
    print(f"mlp_modules_total: {mlp_total:.3f} ms/iter")
    print(f"norm_and_residual_other_total: {other_total:.3f} ms/iter")


if __name__ == "__main__":
    main()
