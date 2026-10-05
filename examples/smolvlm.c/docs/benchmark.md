# Application Benchmark Workloads

The reusable Benchmark API and generic inference runtime live in the standalone
[Inference SDK](../inference_sdk/README.md). SmolVLM uses that SDK through two
application-specific workloads:

- `smolvlm-c benchmark MODEL_DIR PROMPT [MAX_TOKENS] [RUNS] [WARMUPS]` measures
  the accelerated text generation path. Its report includes end-to-end output
  tokens per second and a decode-loop-only token rate.
- `smolvlm-c benchmark-vision MODEL_DIR SQUARE_TILE_IMAGE` measures one tile
  through the fused vision sequence path and reports tile latency and throughput.

Both workloads require Metal or CUDA and fail if the selected optimized path is
unavailable. Model and image loading are outside the timed regions.

`./build.sh benchmark` builds the available platform backend and runs both
workloads with the fixed sample model, prompt, image, token limit, and iteration
counts configured in the script.
