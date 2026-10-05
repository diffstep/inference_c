#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "benchmark" ]]; then
    if [[ "$#" -ne 1 ]]; then
        echo "Usage: $0 benchmark" >&2
        exit 2
    fi

    script_directory=$(cd "$(dirname "$0")" && pwd)
    model_directory="$script_directory/tests/sample-model"
    vision_image="$script_directory/tests/test-data/zxqh/sample5.jpg"
    prompt="Write a short sentence about the sky."
    max_new_tokens=32
    runs=10
    warmup_runs=2
    if [[ ! -f "$model_directory/model.safetensors" ]]; then
        echo "Benchmark model not found: $model_directory/model.safetensors" >&2
        exit 1
    fi

    case "$(uname -s)" in
        Darwin)
            attention_backend=${ATTENTION_BACKEND:-metal}
            if [[ "$attention_backend" == "metal" || "$attention_backend" == "auto" ]]; then
                default_tensor_backend=metal
            else
                default_tensor_backend=cpu
            fi
            tensor_backend=${TENSOR_BACKEND:-$default_tensor_backend}
            ;;
        Linux)
            if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
                attention_backend=$ATTENTION_BACKEND
            elif [[ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
                attention_backend=cuda
            else
                attention_backend=cpu
            fi
            if [[ "$attention_backend" == "cuda" ]]; then
                default_tensor_backend=cuda
            else
                default_tensor_backend=cpu
            fi
            tensor_backend=${TENSOR_BACKEND:-$default_tensor_backend}
            ;;
        *)
            echo "Unsupported build platform: $(uname -s)" >&2
            exit 2
            ;;
    esac

    cd "$script_directory"
    make -B all ATTENTION_BACKEND="$attention_backend" ALLOCATOR=system IMAGE_DECODER=rust TIMING=0
    TENSOR_BACKEND="$tensor_backend" ./build/smolvlm-c benchmark \
        "$model_directory" "$prompt" "$max_new_tokens" "$runs" "$warmup_runs"
    TENSOR_BACKEND="$tensor_backend" ./build/smolvlm-c benchmark-vision \
        "$model_directory" "$vision_image"
    exit 0
fi

if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [benchmark]" >&2
    exit 2
fi

make -B all ATTENTION_BACKEND=metal ALLOCATOR=system IMAGE_DECODER=rust TIMING=0

/usr/bin/time -l env TENSOR_DTYPE=fp16 TENSOR_BACKEND=metal \
    ./build/smolvlm-c generate-image \
    ./tests/sample-model ./tests/test-data/zxqh/sample5.jpg 600

/usr/bin/time -l env TENSOR_BACKEND=metal \
    ./build/smolvlm-c generate-image \
    ./tests/sample-model ./tests/test-data/zxqh/sample5.jpg 600


/usr/bin/time -l env TENSOR_DTYPE=fp16 TENSOR_BACKEND=metal \
    ./build/smolvlm-c generate-image \
    /Users/dengtao2nd/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots/7e3e67edbbed1bf9888184d9df282b700a323964 \
    ./tests/test-data/zxqh/sample5.jpg

/usr/bin/time -l env TENSOR_BACKEND=metal ./build/smolvlm-c generate-image \
  /Users/dengtao2nd/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots/7e3e67edbbed1bf9888184d9df282b700a323964 \
  ./tests/test-data/statue-of-liberty.jpg \
  "Can you describe this image?"


# CUDA_HOME=/usr/local/cuda make -B all ATTENTION_BACKEND=cuda IMAGE_DECODER=rust
