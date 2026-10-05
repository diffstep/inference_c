#!/usr/bin/env python3
"""Compare installed Transformers Gemma3 inference with the Rust release binary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local EmbeddingGemma model directory")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--task", choices=("query", "document"), default="query")
    parser.add_argument("--text", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "target/release/fast-reasoning",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    if not args.model.is_dir():
        parser.error(f"model directory not found: {args.model}")
    if not args.binary.is_file():
        parser.error(f"Rust binary not found: {args.binary}; build it with cargo build --release")

    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this Python/PyTorch runtime")
    device = torch.device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    prompted_text = (
        f"task: search result | query: {args.text}"
        if args.task == "query"
        else f"title: none | text: {args.text}"
    )
    batch = tokenizer(prompted_text, return_tensors="pt", truncation=True, max_length=2048)
    batch = {name: tensor.to(device) for name, tensor in batch.items()}

    def synchronize() -> None:
        if device_name == "mps":
            torch.mps.synchronize()

    def infer() -> torch.Tensor:
        output = model(**batch, return_dict=True).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(output.dtype)
        pooled = (output * mask).sum(1) / mask.sum(1).clamp_min(1)
        return F.normalize(pooled.float(), p=2, dim=-1)[0]

    with torch.inference_mode():
        for _ in range(args.warmup):
            infer()
        synchronize()
        started = time.perf_counter()
        for _ in range(args.iterations):
            reference = infer()
        synchronize()
        python_ms = (time.perf_counter() - started) * 1000 / args.iterations
    reference = reference.cpu()

    environment = os.environ.copy()
    environment["FAST_REASONING_DEVICE"] = "metal" if device_name == "mps" else "cpu"
    result = subprocess.run(
        [
            str(args.binary),
            "embed",
            str(args.model),
            "--task",
            args.task,
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            args.text,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    lines = result.stdout.splitlines()
    rust_tokens = int(next(line.split(": ", 1)[1] for line in lines if line.startswith("Input tokens: ")))
    rust_ms = float(
        next(line.split(": ", 1)[1].split(" ms/iter", 1)[0]
             for line in lines if line.startswith("Embedding compute time: "))
    )
    rust = torch.tensor(
        json.loads(next(line.removeprefix("Embedding: ") for line in lines if line.startswith("Embedding: "))),
        dtype=torch.float32,
    )
    python_tokens = batch["input_ids"].shape[1]
    if rust_tokens != python_tokens:
        raise AssertionError(f"token count mismatch: Rust {rust_tokens}, Python {python_tokens}")
    if rust.shape != reference.shape:
        raise AssertionError(f"embedding shape mismatch: Rust {tuple(rust.shape)}, Python {tuple(reference.shape)}")

    max_abs_error = (rust - reference).abs().max().item()
    cosine = F.cosine_similarity(rust.unsqueeze(0), reference.unsqueeze(0)).item()
    print(f"python={sys.executable}")
    print(f"transformers={transformers.__version__} ({transformers.__file__})")
    print(f"gemma3_source={Gemma3TextModel.__module__}")
    print(f"torch={torch.__version__} ({torch.__file__})")
    print(f"device={device_name}, dtype=float32, attention=sdpa, task={args.task}")
    print(f"tokens={python_tokens}, iterations={args.iterations}, warmup={args.warmup}")
    print(f"transformers_compute_ms={python_ms:.3f}")
    print(f"rust_release_compute_ms={rust_ms:.3f}")
    print(f"max_abs_error={max_abs_error:.9g}")
    print(f"cosine_similarity={cosine:.9f}")


if __name__ == "__main__":
    main()
