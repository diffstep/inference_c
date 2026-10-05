import importlib.util
import json
import os
import struct
import sys
import tempfile

TOOL = os.path.join(os.path.dirname(__file__), "..", "tools", "quantize_safetensors.py")
spec = importlib.util.spec_from_file_location("quantize_safetensors", TOOL)
quantizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantizer)


def write_safetensors(path, metadata, payload):
    encoded = json.dumps(metadata, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.write(payload)


def run():
    values = [((i % 16) - 8) / 4 for i in range(32)]
    payload = struct.pack("<32e", *values)
    with tempfile.TemporaryDirectory() as directory:
        source = os.path.join(directory, "model.safetensors")
        metadata = {"model.text_model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F16", "shape": [1, 32], "data_offsets": [0, len(payload)]}}
        write_safetensors(source, metadata, payload)
        for fmt, block_size in (("q8_0", 34), ("q4_0", 18)):
            output = os.path.join(directory, f"model.{fmt}.safetensors")
            old_argv = sys.argv
            try:
                sys.argv = [TOOL, source, "--format", fmt, "--output", output]
                quantizer.main()
            finally:
                sys.argv = old_argv
            with open(output, "rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_size))
                raw = stream.read()
            entry = header["model.text_model.layers.0.self_attn.q_proj.weight"]
            assert entry["dtype"] == "U8" and entry["shape"] == [1, 1, block_size]
            assert len(raw) == block_size
            expected, _ = quantizer.quantize(payload, "F16", [1, 32], fmt)
            assert raw == expected
    print("offline Q4_0/Q8_0 SafeTensors packaging passed")


if __name__ == "__main__":
    run()
