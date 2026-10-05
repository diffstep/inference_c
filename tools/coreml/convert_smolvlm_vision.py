#!/usr/bin/env python3
"""Convert the SmolVLM-256M vision tower and connector to a fixed-shape Core ML model."""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from transformers import Idefics3ForConditionalGeneration


class VisionTileEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision = model.model.vision_model
        self.connector = model.model.connector

    def forward(self, pixels):
        embeddings = self.vision.embeddings
        hidden = embeddings.patch_embedding(pixels)
        hidden = hidden.flatten(2).transpose(1, 2)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        hidden = hidden + embeddings.position_embedding(positions).unsqueeze(0)
        config = self.vision.config
        batch, sequence, width = hidden.shape
        heads = config.num_attention_heads
        head_width = width // heads
        for layer in self.vision.encoder.layers:
            residual = hidden
            norm = self._layer_norm(hidden, layer.layer_norm1)
            attention = layer.self_attn
            query = attention.q_proj(norm).view(batch, sequence, heads, head_width).transpose(1, 2)
            key = attention.k_proj(norm).view(batch, sequence, heads, head_width).transpose(1, 2)
            value = attention.v_proj(norm).view(batch, sequence, heads, head_width).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-2, -1)) * (head_width ** -0.5)
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, sequence, width)
            hidden = residual + attention.out_proj(attended)
            residual = hidden
            norm = self._layer_norm(hidden, layer.layer_norm2)
            mlp = layer.mlp
            norm = functional.gelu(mlp.fc1(norm), approximate="tanh")
            hidden = residual + mlp.fc2(norm)
        hidden = self._layer_norm(hidden, self.vision.post_layernorm)

        scale = self.connector.scale_factor
        grid = int(sequence ** 0.5)
        hidden = hidden.reshape(batch, grid, grid, width)
        hidden = hidden.reshape(batch, grid, grid // scale, width * scale).permute(0, 2, 1, 3)
        hidden = hidden.reshape(batch, grid // scale, grid // scale, width * scale * scale)
        hidden = hidden.permute(0, 2, 1, 3).reshape(batch, sequence // (scale * scale),
                                                    width * scale * scale)
        return self.connector.modality_projection(hidden)

    @staticmethod
    def _layer_norm(value, layer):
        output = functional.layer_norm(value.float(), (value.shape[-1],),
                                       layer.weight.float(), layer.bias.float(), layer.eps)
        return output.to(value.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path,
                        help="Local Hugging Face SmolVLM-256M-Instruct snapshot")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output .mlpackage path")
    args = parser.parse_args()

    model = Idefics3ForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.float16, local_files_only=True
    ).eval()
    model.config.vision_config._attn_implementation = "eager"
    model.model.vision_model.config._attn_implementation = "eager"
    wrapper = VisionTileEncoder(model).eval()
    pixels = torch.zeros((1, 3, 512, 512), dtype=torch.float16)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, pixels, strict=False, check_trace=False)
        output = traced(pixels).cpu().numpy().astype(np.float32)
        reference_hidden = model.model.vision_model(pixel_values=pixels).last_hidden_state
        reference_output = model.model.connector(reference_hidden).cpu().numpy().astype(np.float32)
    if tuple(output.shape) != (1, 64, 576):
        raise RuntimeError(f"unexpected vision output shape: {tuple(output.shape)}")
    wrapper_error = float(np.max(np.abs(output - reference_output)))
    if not np.allclose(output, reference_output, rtol=2.0e-2, atol=5.0e-2):
        raise RuntimeError(f"manual vision graph differs from Hugging Face (max abs error {wrapper_error:.6g})")

    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="pixels", shape=(1, 3, 512, 512),
                              dtype=np.float16)],
        outputs=[ct.TensorType(name="features", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
    )
    converted.short_description = "SmolVLM-256M vision tile encoder and connector"
    converted.input_description["pixels"] = "Normalized RGB tile, NCHW FP16, 512x512"
    converted.output_description["features"] = "FP16 visual tokens, shape 1x64x576"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    predicted = converted.predict({"pixels": pixels.cpu().numpy()})["features"].astype(np.float32)
    absolute_error = np.abs(predicted - output)
    max_error = float(np.max(absolute_error))
    rms_error = float(np.sqrt(np.mean(absolute_error * absolute_error)))
    if not np.allclose(predicted, output, rtol=2.0e-2, atol=7.5e-1):
        raise RuntimeError(f"Core ML output differs from PyTorch reference (max abs error {max_error:.6g}; "
                           f"reference max {float(np.max(np.abs(output))):.6g}; "
                           f"Core ML max {float(np.max(np.abs(predicted))):.6g})")
    print(f"saved {args.output} (input=1x3x512x512, output=1x64x576)")
    print(f"PyTorch/Core ML numerical check passed (max abs error {max_error:.6g}, "
          f"RMSE {rms_error:.6g})")
    print(f"Manual/Hugging Face numerical check passed (max abs error {wrapper_error:.6g})")


if __name__ == "__main__":
    main()
