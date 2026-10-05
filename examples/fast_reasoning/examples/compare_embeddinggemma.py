#!/usr/bin/env python3
"""Compare Gemma3 embeddings and encoder compute time against Transformers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from embed_embeddinggemma import QUERY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/embeddinggemma-300m")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "sdpa", "eager"),
        default="auto",
        help="Transformers attention path; auto preserves the model/library default",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--task", choices=("query", "document"), default="query")
    parser.add_argument("--text", default=QUERY)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "target/debug/fast-reasoning",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is not available in this PyTorch installation")

    model_path = args.model
    if not Path(model_path).is_dir():
        model_path = snapshot_download(model_path, local_files_only=args.offline)
    if not args.binary.is_file():
        parser.error(f"Rust binary not found: {args.binary}; build it first with cargo build")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.offline)
    model_kwargs = {"dtype": torch.float32, "local_files_only": args.offline}
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModel.from_pretrained(model_path, **model_kwargs).to(args.device)
    model.eval()
    config = model.config
    attention_implementation = getattr(config, "_attn_implementation", "unknown")
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        attention_implementation = getattr(
            text_config, "_attn_implementation", attention_implementation
        )
    prompted_text = (
        f"task: search result | query: {args.text}"
        if args.task == "query"
        else f"title: none | text: {args.text}"
    )
    batch = tokenizer(prompted_text, return_tensors="pt", truncation=True, max_length=2048)
    batch = {name: tensor.to(args.device) for name, tensor in batch.items()}

    def synchronize() -> None:
        if args.device == "mps":
            torch.mps.synchronize()

    def infer() -> torch.Tensor:
        output = model(**batch, return_dict=True).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(output.dtype)
        return F.normalize(
            (output * mask).sum(1) / mask.sum(1).clamp_min(1), p=2, dim=-1
        )[0]

    with torch.inference_mode():
        for _ in range(args.warmup):
            infer()
        synchronize()
        started = time.perf_counter()
        for _ in range(args.iterations):
            reference = infer()
        synchronize()
        python_compute_ms = (time.perf_counter() - started) * 1000 / args.iterations
    reference = reference.cpu()

    env = os.environ.copy()
    env["FAST_REASONING_DEVICE"] = "metal" if args.device == "mps" else "cpu"
    result = subprocess.run(
        [
            str(args.binary), "embed", model_path,
            "--task", args.task,
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            args.text,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    vector_line = next(
        (line.removeprefix("Embedding: ") for line in result.stdout.splitlines()
         if line.startswith("Embedding: ")),
        None,
    )
    if vector_line is None:
        raise RuntimeError("Rust output did not contain an embedding vector")
    rust_token_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("Input tokens: ")),
        None,
    )
    if rust_token_line is None:
        raise RuntimeError("Rust output did not contain its input token count")
    rust_time_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("Embedding compute time: ")),
        None,
    )
    if rust_time_line is None:
        raise RuntimeError("Rust output did not contain its embedding compute time")
    rust_compute_ms = float(
        rust_time_line.removeprefix("Embedding compute time: ").split(" ms/iter", 1)[0]
    )
    rust_tokens = int(rust_token_line.removeprefix("Input tokens: "))
    if rust_tokens != batch["input_ids"].shape[1]:
        raise AssertionError(
            f"token count mismatch: Rust {rust_tokens}, Python {batch['input_ids'].shape[1]}"
        )
    rust = torch.tensor(json.loads(vector_line), dtype=torch.float32)
    if rust.shape != reference.shape:
        raise AssertionError(f"shape mismatch: Rust {tuple(rust.shape)}, Python {tuple(reference.shape)}")

    max_abs_error = (rust - reference).abs().max().item()
    cosine = F.cosine_similarity(rust.unsqueeze(0), reference.unsqueeze(0)).item()
    print(f"task: {args.task}")
    print(f"device: {args.device}")
    print(f"pytorch: {torch.__version__} ({torch.__file__})")
    print(f"transformers_attention_implementation: {attention_implementation}")
    print(f"python_compute_time: {python_compute_ms:.3f} ms/iter ({args.iterations} iterations)")
    print(f"rust_compute_time: {rust_compute_ms:.3f} ms/iter ({args.iterations} iterations)")
    print(f"tokens: python={batch['input_ids'].shape[1]}")
    print(f"embedding_dim: {reference.numel()}")
    print(f"max_abs_error: {max_abs_error:.9g}")
    print(f"cosine_similarity: {cosine:.9f}")
    if max_abs_error > 5e-4 or cosine < 0.99999:
        raise AssertionError("Rust embedding diverges from the Transformers reference")


if __name__ == "__main__":
    main()
