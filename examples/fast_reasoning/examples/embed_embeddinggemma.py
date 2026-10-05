#!/usr/bin/env python3
"""Minimal PyTorch reference inference for google/embeddinggemma-300m.

Uses the official retrieval prefixes, attention-mask mean pooling, and L2
normalization so the eventual Rust implementation has a concrete baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer


QUERY = "Which market level is closest to the current price?"
DOCUMENTS = [
    "The current price is near resistance at 4,250.",
    "Support is located around 4,100, below the current price.",
    "Trading volume increased during the morning session.",
]


def embed(texts: list[str], tokenizer, model, device: torch.device,
          max_length: int) -> torch.Tensor:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {name: tensor.to(device) for name, tensor in batch.items()}

    output = model(**batch, return_dict=True)
    token_embeddings = output.last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
    pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return F.normalize(pooled.float(), p=2, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="google/embeddinggemma-300m",
        help="HF model ID or local model directory (default: %(default)s)",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--offline", action="store_true", help="Do not access the Hub")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is not available")
    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model_path = args.model
    if args.offline and not Path(model_path).is_dir():
        # Resolve the model ID to its cached snapshot so tokenizer loading never
        # needs Hub metadata/network access in offline mode.
        model_path = snapshot_download(model_path, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=args.offline
    )
    model = AutoModel.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=args.offline,
    ).to(device)
    model.eval()

    # EmbeddingGemma uses task-aware prefixes; query and document inputs are
    # intentionally encoded with their respective prompts.
    query_text = f"task: search result | query: {QUERY}"
    document_texts = [f"title: none | text: {text}" for text in DOCUMENTS]

    with torch.inference_mode():
        query_embedding = embed([query_text], tokenizer, model, device, args.max_length)
        document_embeddings = embed(
            document_texts, tokenizer, model, device, args.max_length
        )
        similarities = query_embedding @ document_embeddings.T

    print(f"model={args.model}, device={device}, dtype={dtype}")
    print(f"query_embedding_shape={tuple(query_embedding.shape)}")
    print(f"query_embedding_first8={query_embedding[0, :8].tolist()}")
    print(f"document_embeddings_shape={tuple(document_embeddings.shape)}")
    for index in similarities[0].argsort(descending=True).tolist():
        print(f"similarity={similarities[0, index].item():.6f}  {DOCUMENTS[index]}")


if __name__ == "__main__":
    main()
