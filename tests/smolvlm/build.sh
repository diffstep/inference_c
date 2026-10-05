#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: tests/smolvlm/build.sh [--help]

Build configuration:
  SMOLVLM_BACKEND             METAL on macOS, CUDA on Linux
  SMOLVLM_IMAGE_DECODER        RUST or STB (default: RUST)
  SMOLVLM_ALLOCATOR            SYSTEM or MIMALLOC (default: SYSTEM)
  SMOLVLM_ENABLE_GGML          ON or OFF (default: OFF)
  SMOLVLM_BUILD_TYPE           CMake build type (default: Release)
  SMOLVLM_TIMING               ON or OFF (default: ON; affects measurements)
  SMOLVLM_CC / SMOLVLM_CXX     C/C++ compiler executables
  SMOLVLM_CFLAGS               flags for the example C sources
  SMOLVLM_CPPFLAGS             preprocessor flags for the example C sources
  SMOLVLM_LDFLAGS / SMOLVLM_LDLIBS
                               extra linker flags and libraries
  SMOLVLM_SDK_CFLAGS           flags for SDK C sources
  SMOLVLM_SDK_CXXFLAGS         flags for SDK C++ sources
  SMOLVLM_CMAKE_ARGS           additional space-separated CMake arguments
  SMOLVLM_RUSTFLAGS            flags passed to Cargo/Rust
  SMOLVLM_BUILD_DIR            build output directory
  SMOLVLM_EXECUTABLE           output executable path
  SMOLVLM_RUN_BENCHMARK        ON or OFF (default: ON)
  SMOLVLM_RUN_COREML_TEST      AUTO, ON, or OFF (default: AUTO; macOS only)
  SMOLVLM_COREML_MODEL         converted vision_ane.mlpackage path
  SMOLVLM_COREML_SOURCE_MODEL_DIR
                               local Hugging Face model directory for conversion
  SMOLVLM_COREML_PYTHON        Python with torch/transformers/coremltools installed
  SMOLVLM_COREML_BUILD_DIR     Core ML test build directory

Benchmark configuration:
  SMOLVLM_MODEL_DIR            model directory
  SMOLVLM_IMAGE_PATH           input image
  SMOLVLM_PROMPT               image prompt
  SMOLVLM_MAX_TOKEN_CASES      whitespace-separated token limits
  SMOLVLM_DTYPE_CASES          whitespace-separated dtypes, e.g. fp16 fp32
  SMOLVLM_WEIGHT_QUANT_CASES   fp16, q8_0, or q4_0 (default: fp16)
  SMOLVLM_GEMV_VARIANTS        whitespace-separated decode GEMV variants
  SMOLVLM_BENCHMARK_RUNS       measured runs (default: 1)
  SMOLVLM_WARMUP_RUNS          warmup runs (default: 0)
  SMOLVLM_BENCHMARK_FORMAT     text or json (default: text)
  SMOLVLM_TENSOR_BACKEND       runtime tensor backend (defaults to build backend)
  SMOLVLM_ATTENTION_BACKEND    runtime attention backend (same default)
  SMOLVLM_METALLIB_PATH        attention Metal library path
  SMOLVLM_TENSOR_METALLIB_PATH tensor Metal library path
HELP
    exit 0
fi
if [[ "$#" -ne 0 ]]; then
    echo "Usage: $0 [--help]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SMOLVLM_BUILD_DIR:-${SCRIPT_DIR}/build-metal-rust}"
MODEL_DIR="${SMOLVLM_MODEL_DIR:-${SCRIPT_DIR}/sample-model}"
IMAGE_PATH="${SMOLVLM_IMAGE_PATH:-${SCRIPT_DIR}/test-data/statue-of-liberty.jpg}"
PROMPT="${SMOLVLM_PROMPT:-Describe the image briefly.}"

# Configure the compiler, SDK, and Rust decoder build.
BACKEND="${SMOLVLM_BACKEND:-METAL}"
IMAGE_DECODER="${SMOLVLM_IMAGE_DECODER:-RUST}"
ALLOCATOR="${SMOLVLM_ALLOCATOR:-SYSTEM}"
ENABLE_GGML="${SMOLVLM_ENABLE_GGML:-OFF}"
BUILD_TYPE="${SMOLVLM_BUILD_TYPE:-Release}"
TIMING="${SMOLVLM_TIMING:-ON}"
CC="${SMOLVLM_CC:-cc}"
CXX="${SMOLVLM_CXX:-c++}"
CFLAGS="${SMOLVLM_CFLAGS:--O2}"
CPPFLAGS="${SMOLVLM_CPPFLAGS:-}"
LDFLAGS="${SMOLVLM_LDFLAGS:-}"
LDLIBS="${SMOLVLM_LDLIBS:-}"
SDK_CFLAGS="${SMOLVLM_SDK_CFLAGS:-}"
SDK_CXXFLAGS="${SMOLVLM_SDK_CXXFLAGS:-}"
SDK_CMAKE_ARGS="${SMOLVLM_CMAKE_ARGS:-}"
RUSTFLAGS="${SMOLVLM_RUSTFLAGS:-}"

# Configure the benchmark matrix and runtime backend settings.
MAX_TOKEN_CASES="${SMOLVLM_MAX_TOKEN_CASES:-1024}"
DTYPE_CASES="${SMOLVLM_DTYPE_CASES:-fp16}"
WEIGHT_QUANT_CASES="${SMOLVLM_WEIGHT_QUANT_CASES:-fp16 q8_0 q4_0}"
GEMV_VARIANTS="${SMOLVLM_GEMV_VARIANTS:-baseline}"
RUNS="${SMOLVLM_BENCHMARK_RUNS:-1}"
WARMUP_RUNS="${SMOLVLM_WARMUP_RUNS:-0}"
BENCHMARK_FORMAT="${SMOLVLM_BENCHMARK_FORMAT:-text}"
RUN_BENCHMARK="${SMOLVLM_RUN_BENCHMARK:-ON}"
RUN_COREML_TEST="${SMOLVLM_RUN_COREML_TEST:-AUTO}"
COREML_MODEL="${SMOLVLM_COREML_MODEL:-${SCRIPT_DIR}/models/vision_ane.mlpackage}"
COREML_SOURCE_MODEL_DIR="${SMOLVLM_COREML_SOURCE_MODEL_DIR:-}"
COREML_PYTHON="${SMOLVLM_COREML_PYTHON:-python3}"
COREML_BUILD_DIR="${SMOLVLM_COREML_BUILD_DIR:-${BUILD_DIR}/coreml-hybrid}"
TENSOR_BACKEND="${SMOLVLM_TENSOR_BACKEND:-}"
ATTENTION_BACKEND="${SMOLVLM_ATTENTION_BACKEND:-}"
METALLIB_PATH="${SMOLVLM_METALLIB_PATH:-${BUILD_DIR}/metal/attention.metallib}"
TENSOR_KERNELS_METALLIB_PATH="${SMOLVLM_TENSOR_METALLIB_PATH:-${BUILD_DIR}/metal/tensor_kernels.metallib}"
EXECUTABLE="${SMOLVLM_EXECUTABLE:-${BUILD_DIR}/smolvlm}"

to_upper() { printf '%s' "$1" | tr '[:lower:]' '[:upper:]'; }
BACKEND_UPPER="$(to_upper "${BACKEND}")"
IMAGE_DECODER_UPPER="$(to_upper "${IMAGE_DECODER}")"
ALLOCATOR_UPPER="$(to_upper "${ALLOCATOR}")"
ENABLE_GGML_UPPER="$(to_upper "${ENABLE_GGML}")"
TIMING_UPPER="$(to_upper "${TIMING}")"
RUN_BENCHMARK_UPPER="$(to_upper "${RUN_BENCHMARK}")"
RUN_COREML_TEST_UPPER="$(to_upper "${RUN_COREML_TEST}")"

case "$(uname -s)" in
    Darwin)
        if [[ "${BACKEND_UPPER}" != "METAL" ]]; then
            echo "macOS SmolVLM build requires SMOLVLM_BACKEND=METAL" >&2
            exit 2
        fi
        DEFAULT_TENSOR_BACKEND=metal
        ;;
    Linux)
        if [[ "${BACKEND_UPPER}" != "CUDA" ]]; then
            echo "Linux SmolVLM build requires SMOLVLM_BACKEND=CUDA" >&2
            exit 2
        fi
        DEFAULT_TENSOR_BACKEND=cuda
        ;;
    *)
        echo "Unsupported build platform: $(uname -s)" >&2
        exit 2
        ;;
esac

TENSOR_BACKEND="${TENSOR_BACKEND:-${DEFAULT_TENSOR_BACKEND}}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-${TENSOR_BACKEND}}"

case "${TIMING_UPPER}" in ON|OFF) ;; *) echo "SMOLVLM_TIMING must be ON or OFF" >&2; exit 2 ;; esac
case "${RUN_BENCHMARK_UPPER}" in ON|OFF) ;; *) echo "SMOLVLM_RUN_BENCHMARK must be ON or OFF" >&2; exit 2 ;; esac
case "${RUN_COREML_TEST_UPPER}" in AUTO|ON|OFF) ;; *) echo "SMOLVLM_RUN_COREML_TEST must be AUTO, ON, or OFF" >&2; exit 2 ;; esac
case "${IMAGE_DECODER_UPPER}" in RUST|STB) ;; *) echo "SMOLVLM_IMAGE_DECODER must be RUST or STB" >&2; exit 2 ;; esac
case "${ALLOCATOR_UPPER}" in SYSTEM|MIMALLOC) ;; *) echo "SMOLVLM_ALLOCATOR must be SYSTEM or MIMALLOC" >&2; exit 2 ;; esac
case "${ENABLE_GGML_UPPER}" in ON|OFF) ;; *) echo "SMOLVLM_ENABLE_GGML must be ON or OFF" >&2; exit 2 ;; esac
case "${BENCHMARK_FORMAT}" in text|json) ;; *) echo "SMOLVLM_BENCHMARK_FORMAT must be text or json" >&2; exit 2 ;; esac

if [[ "${RUN_BENCHMARK_UPPER}" == "ON" ]]; then
    if [[ ! -f "${MODEL_DIR}/model.safetensors" ]]; then
        echo "model weights not found: ${MODEL_DIR}/model.safetensors" >&2
        exit 1
    fi
    if [[ ! -f "${IMAGE_PATH}" ]]; then
        echo "benchmark image not found: ${IMAGE_PATH}" >&2
        exit 1
    fi
fi

echo "--- SmolVLM build ---"
echo "backend=${BACKEND_UPPER} decoder=${IMAGE_DECODER_UPPER} allocator=${ALLOCATOR_UPPER} ggml=${ENABLE_GGML_UPPER} timing=${TIMING_UPPER} build_type=${BUILD_TYPE}"
echo "CC=${CC} CFLAGS=${CFLAGS} CPPFLAGS=${CPPFLAGS} LDFLAGS=${LDFLAGS} LDLIBS=${LDLIBS}"
echo "SDK_CFLAGS=${SDK_CFLAGS} SDK_CXXFLAGS=${SDK_CXXFLAGS} SDK_CMAKE_ARGS=${SDK_CMAKE_ARGS}"

export RUSTFLAGS
make -C "${SCRIPT_DIR}" \
    BACKEND="${BACKEND_UPPER}" \
    IMAGE_DECODER="${IMAGE_DECODER_UPPER}" \
    ALLOCATOR="${ALLOCATOR_UPPER}" \
    ENABLE_GGML="${ENABLE_GGML_UPPER}" \
    BUILD_TYPE="${BUILD_TYPE}" \
    TIMING="${TIMING_UPPER}" \
    CC="${CC}" \
    CXX="${CXX}" \
    CFLAGS="${CFLAGS}" \
    CPPFLAGS="${CPPFLAGS}" \
    LDFLAGS="${LDFLAGS}" \
    LDLIBS="${LDLIBS}" \
    SDK_CFLAGS="${SDK_CFLAGS}" \
    SDK_CXXFLAGS="${SDK_CXXFLAGS}" \
    SDK_CMAKE_ARGS="${SDK_CMAKE_ARGS}" \
    SDK_BUILD_DIR="${BUILD_DIR}" \
    TARGET="${EXECUTABLE}"

run_coreml_test() {
    if [[ "${RUN_COREML_TEST_UPPER}" == "OFF" ]]; then
        echo "Skipping SmolVLM Core ML/ANE test (SMOLVLM_RUN_COREML_TEST=OFF)"
        return
    fi
    if [[ "$(uname -s)" != "Darwin" ]]; then
        if [[ "${RUN_COREML_TEST_UPPER}" == "ON" ]]; then
            echo "SmolVLM Core ML/ANE test requires macOS" >&2
            exit 2
        fi
        echo "Skipping SmolVLM Core ML/ANE test (macOS only)"
        return
    fi

    if [[ ! -d "${COREML_MODEL}" ]]; then
        if [[ -z "${COREML_SOURCE_MODEL_DIR}" ]]; then
            if [[ -f "${MODEL_DIR}/model.safetensors" ]]; then
                COREML_SOURCE_MODEL_DIR="${MODEL_DIR}"
            elif [[ -n "${HOME:-}" ]]; then
                snapshot_root="${HOME}/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots"
                if [[ -d "${snapshot_root}" ]]; then
                    for candidate in "${snapshot_root}"/*; do
                        [[ -d "${candidate}" ]] && COREML_SOURCE_MODEL_DIR="${candidate}"
                    done
                fi
            fi
        fi
        if [[ -z "${COREML_SOURCE_MODEL_DIR}" || ! -f "${COREML_SOURCE_MODEL_DIR}/model.safetensors" ]]; then
            if [[ "${RUN_COREML_TEST_UPPER}" == "ON" ]]; then
                echo "Core ML package is missing and no local SmolVLM weights were found; set SMOLVLM_COREML_SOURCE_MODEL_DIR" >&2
                exit 1
            fi
            echo "Skipping SmolVLM Core ML/ANE test (no converted package or local weights)"
            return
        fi
        if ! "${COREML_PYTHON}" -c 'import coremltools, numpy, torch, transformers' >/dev/null 2>&1; then
            if [[ "${RUN_COREML_TEST_UPPER}" == "ON" ]]; then
                echo "Core ML conversion dependencies are missing from ${COREML_PYTHON}" >&2
                exit 1
            fi
            echo "Skipping SmolVLM Core ML/ANE test (Python conversion dependencies unavailable)"
            return
        fi
        echo "Converting SmolVLM vision tower for Core ML: ${COREML_SOURCE_MODEL_DIR}"
        "${COREML_PYTHON}" "${SCRIPT_DIR}/../../tools/coreml/convert_smolvlm_vision.py" \
            --model-dir "${COREML_SOURCE_MODEL_DIR}" --output "${COREML_MODEL}"
    fi

    echo "--- SmolVLM Core ML/ANE consistency test ---"
    cmake -S "${SCRIPT_DIR}/../.." -B "${COREML_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DINFERENCE_SDK_BACKEND=METAL \
        -DINFERENCE_SDK_METALLIB_DIR="${COREML_BUILD_DIR}/metal" \
        -DINFERENCE_SDK_ENABLE_COREML=ON \
        -DINFERENCE_SDK_BUILD_TESTS=ON
    cmake --build "${COREML_BUILD_DIR}" --parallel --target test_tensor_graph test_tensor_coreml
    INFERENCE_SDK_COREML_TEST_MODEL="${COREML_MODEL}" \
        ctest --test-dir "${COREML_BUILD_DIR}" --output-on-failure \
            -R '^test_tensor_(graph|coreml)$'
}

run_coreml_test

if [[ "${RUN_BENCHMARK_UPPER}" == "OFF" ]]; then
    echo "Build complete: ${EXECUTABLE}"
    exit 0
fi

read -r -a max_token_cases <<< "${MAX_TOKEN_CASES}"
read -r -a dtype_cases <<< "${DTYPE_CASES}"
read -r -a weight_quant_cases <<< "${WEIGHT_QUANT_CASES}"
read -r -a gemv_variants <<< "${GEMV_VARIANTS}"
for dtype in "${dtype_cases[@]}"; do
    for weight_quant in "${weight_quant_cases[@]}"; do
    case "${weight_quant}" in fp16|q8_0|q4_0) ;; *) echo "invalid weight quantization: ${weight_quant}" >&2; exit 2 ;; esac
    if [[ "${dtype}" != "fp16" && "${weight_quant}" != "fp16" ]]; then
        echo "TENSOR_WEIGHT_QUANT=${weight_quant} requires TENSOR_DTYPE=fp16" >&2
        exit 2
    fi
    if [[ "${dtype}" == "fp32" ]]; then
        selected_gemv_variants=("${gemv_variants[@]}")
    else
        # The 4-SIMD kernel currently exists only for the FP32 decode path.
        selected_gemv_variants=(baseline)
    fi
    for gemv_variant in "${selected_gemv_variants[@]}"; do
        for max_tokens in "${max_token_cases[@]}"; do
            echo "--- SmolVLM image benchmark: backend=${TENSOR_BACKEND}, dtype=${dtype}, weight_quant=${weight_quant}, decode_gemv=${gemv_variant}, image=${IMAGE_PATH}, max_new_tokens=${max_tokens}, runs=${RUNS}, warmup_runs=${WARMUP_RUNS}, timing=${TIMING_UPPER} ---"
            runtime_env=(
                "TENSOR_DTYPE=${dtype}"
                "TENSOR_WEIGHT_QUANT=${weight_quant}"
                "TENSOR_BACKEND=${TENSOR_BACKEND}"
                "TENSOR_ATTENTION_BACKEND=${ATTENTION_BACKEND}"
                "TENSOR_METAL_DECODE_GEMV=${gemv_variant}"
            )
            if [[ "${TENSOR_BACKEND}" == "metal" && "${IMAGE_DECODER_UPPER}" != "RUST" ]]; then
                runtime_env+=(
                    "TENSOR_METALLIB_PATH=${METALLIB_PATH}"
                    "TENSOR_KERNELS_METALLIB_PATH=${TENSOR_KERNELS_METALLIB_PATH}"
                )
            fi
            env "${runtime_env[@]}" "${EXECUTABLE}" benchmark-image \
                "${MODEL_DIR}" "${IMAGE_PATH}" "${PROMPT}" \
                "${max_tokens}" "${RUNS}" "${WARMUP_RUNS}" "${BENCHMARK_FORMAT}"
        done
    done
    done
done
