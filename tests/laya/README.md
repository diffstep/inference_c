# Native Laya multilingual inference example

This example runs the local Laya multilingual checkpoint through the dsinfra C
execution path on Metal or CUDA. It uses the existing Hugging Face snapshot in
place; model weights are not copied into the repository. The workload builds one
industrial support ticket and returns three typed decisions: department routing
(`choice`), urgency (`score`), and escalation (`noul`).

## Build and run

From the SDK root:

```sh
./tests/laya/build.sh
```

The script builds the SDK and native example, then runs five measured iterations
after two warmups. It prints the report with `benchmark_report_to_text()` and the
last iteration's decision probabilities. The default model directory is the local
snapshot at
`/Users/dengtao2nd/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/multilingual`.

Useful overrides:

```sh
LAYA_MODEL_DIR=/path/to/multilingual \
LAYA_RUNS=10 LAYA_WARMUP_RUNS=3 LAYA_FORMAT=json \
LAYA_TIMING=ON ./tests/laya/build.sh
```

`LAYA_BACKEND` defaults to `METAL` on macOS and `CUDA` on Linux. The build is
GPU-only and rejects other backends. `LAYA_BUILD_TYPE`, `LAYA_BUILD_DIR`,
`LAYA_CC`, `LAYA_SDK_CFLAGS`, `LAYA_SDK_CXXFLAGS`, and `LAYA_CMAKE_ARGS` configure
the build. For direct invocation after building:

```sh
TENSOR_DTYPE=fp16 TENSOR_BACKEND=metal \
TENSOR_ATTENTION_BACKEND=metal \
TENSOR_METALLIB_PATH=tests/laya/build/sdk/metal/attention.metallib \
TENSOR_KERNELS_METALLIB_PATH=tests/laya/build/sdk/metal/tensor_kernels.metallib \
tests/laya/build/laya /path/to/multilingual 5 2 text
```

On CUDA, set `TENSOR_BACKEND=cuda` and `TENSOR_ATTENTION_BACKEND=cuda`; the Metal
library variables do not apply. The `text|json` argument selects report output.
`reference.py` remains available as a separate official Laya/PyTorch comparison
when its Python dependencies and a supported GPU are installed; it is not used by
the native build script.

## Implementation scope

The native path loads the checkpoint's F16 SafeTensors weights and tokenizer,
executes the ModernBERT encoder, two-layer typed-decision transformer, marker
scorer, and temperature-scaled option probabilities. Its supported model config
is validated at load time. Neural network layers use dsinfra Metal/CUDA FP16
compute; CPU is not an inference backend. Host code prepares token IDs and
formats final decision results, and performs the small temperature-scaled
softmax over each question's option logits.

The native scorer covers the three typed outputs used in this example. Laya's
separate action-selection head is not included. This example has been compile
checked and its tensor names/shapes were checked against the local checkpoint;
end-to-end probability parity and throughput have not yet been measured because
this environment reports no usable GPU. Compare fixed inputs against
`reference.py` on a Metal or CUDA host before treating its probabilities as a
validated reproduction of the official implementation.
