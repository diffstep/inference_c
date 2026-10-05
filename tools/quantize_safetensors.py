#!/usr/bin/env python3
"""Write an auxiliary GGML Q8_0/Q4_0 SafeTensors file for linear weights.

Requires numpy. The original model.safetensors remains the source for norms,
embeddings, biases, and other non-quantized tensors. Output is named
model.<format>.safetensors by default so the SmolVLM loader can discover it.
"""
import argparse
import json
import mmap
import os
import struct
import sys

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("quantization requires numpy: python3 -m pip install numpy") from exc


def eligible(name, shape):
    if len(shape) != 2 or "embed_tokens" in name or "position_embedding" in name:
        return False
    return any(part in name for part in (
        ".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight", ".out_proj.weight",
        ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight",
        ".fc1.weight", ".fc2.weight", "modality_projection.proj.weight",
        "lm_head.weight"))


def as_fp16_values(raw, dtype):
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").astype(np.float16).astype(np.float32)
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return bits.view(np.float32).astype(np.float16).astype(np.float32)
    raise ValueError(f"unsupported source dtype {dtype}; expected F16, F32, or BF16")


def half_bits(value):
    return np.asarray(value, dtype="<f2").view("<u2").reshape(()).item()


def quantize(raw, dtype, shape, fmt):
    out, width = shape
    if width % 32:
        raise ValueError(f"input width {width} is not divisible by 32")
    x = as_fp16_values(raw, dtype).reshape(out, width // 32, 32)
    blocks = x.shape[1]
    if fmt == "q8_0":
        scale = np.max(np.abs(x), axis=-1) / 127.0
        inv = np.divide(1.0, scale, out=np.zeros_like(scale), where=scale != 0)
        q = np.clip(np.rint(x * inv[..., None]), -127, 127).astype(np.int8)
        packed = np.empty((out, blocks, 34), dtype=np.uint8)
        packed[:, :, :2] = np.asarray(scale, dtype="<f2").view("<u2").view(np.uint8).reshape(out, blocks, 2)
        packed[:, :, 2:] = q.view(np.uint8)
        block_bytes = 34
    else:
        max_at = np.argmax(np.abs(x), axis=-1)
        max_signed = np.take_along_axis(x, max_at[..., None], axis=-1)[..., 0]
        scale = max_signed / -8.0
        inv = np.divide(1.0, scale, out=np.zeros_like(scale), where=scale != 0)
        q = np.clip(np.trunc(x * inv[..., None] + 8.5), 0, 15).astype(np.uint8)
        packed = np.empty((out, blocks, 18), dtype=np.uint8)
        packed[:, :, :2] = np.asarray(scale, dtype="<f2").view("<u2").view(np.uint8).reshape(out, blocks, 2)
        packed[:, :, 2:] = q[:, :, :16] | (q[:, :, 16:] << 4)
        block_bytes = 18
    return packed.tobytes(), [out, blocks, block_bytes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="source model.safetensors")
    parser.add_argument("--format", choices=("q8_0", "q4_0"), required=True)
    parser.add_argument("--output", help="defaults to model.<format>.safetensors beside input")
    args = parser.parse_args()
    source = os.path.abspath(args.model)
    output = args.output or os.path.join(os.path.dirname(source), f"model.{args.format}.safetensors")
    with open(source, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            header_size = struct.unpack_from("<Q", mapped, 0)[0]
            header = json.loads(mapped[8:8 + header_size])
            data_start = 8 + header_size
            tensors, chunks = {}, []
            cursor = 0
            for name, meta in header.items():
                if name == "__metadata__" or not eligible(name, meta["shape"]):
                    continue
                begin, end = meta["data_offsets"]
                payload, packed_shape = quantize(mapped[data_start + begin:data_start + end],
                    meta["dtype"], meta["shape"], args.format)
                tensors[name] = {"dtype": "U8", "shape": packed_shape,
                                 "data_offsets": [cursor, cursor + len(payload)]}
                chunks.append(payload)
                cursor += len(payload)
            if not tensors:
                raise SystemExit("no eligible rank-2 linear weights found")
    tensors["__metadata__"] = {"format": f"dsinfra_{args.format}_v1"}
    encoded = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    temporary = output + ".tmp"
    with open(temporary, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for chunk in chunks:
            f.write(chunk)
    os.replace(temporary, output)
    print(f"wrote {len(tensors)-1} tensors to {output} ({cursor / (1024**2):.1f} MiB)")


if __name__ == "__main__":
    main()
