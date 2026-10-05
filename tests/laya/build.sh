#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DEFAULT="/Users/dengtao2nd/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/multilingual"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: tests/laya/build.sh [--help]

Builds and runs the native dsinfra Laya multilingual FP16 example. Runtime inference
requires Metal on macOS or CUDA on Linux. CPU fallback is disabled.

Build settings:
  LAYA_BACKEND          METAL on macOS, CUDA on Linux
  LAYA_BUILD_TYPE       Release or Debug (default: Release)
  LAYA_BUILD_DIR        output directory (default: tests/laya/build)
  LAYA_TIMING           ON or OFF (default: OFF)
  LAYA_CC               C compiler
  LAYA_SDK_CFLAGS       extra SDK C flags
  LAYA_SDK_CXXFLAGS     extra SDK C++ flags
  LAYA_CMAKE_ARGS       extra space-separated CMake arguments

Run settings:
  LAYA_MODEL_DIR        multilingual model directory (default: local HF snapshot)
  LAYA_RUNS             measured runs (default: 5)
  LAYA_WARMUP_RUNS      warmups (default: 2)
  LAYA_FORMAT           text or json (default: text)

The checkpoint is read in place; weights are not copied into this repository.
HELP
    exit 0
fi
if [[ "$#" -ne 0 ]]; then
    echo "Usage: $0 [--help]" >&2
    exit 2
fi

case "$(uname -s)" in
    Darwin) DEFAULT_BACKEND=METAL; DEFAULT_RUNTIME_BACKEND=metal ;;
    Linux) DEFAULT_BACKEND=CUDA; DEFAULT_RUNTIME_BACKEND=cuda ;;
    *) echo "Laya native inference supports macOS/Metal and Linux/CUDA only" >&2; exit 2 ;;
esac

BACKEND="${LAYA_BACKEND:-${DEFAULT_BACKEND}}"
BACKEND_UPPER="$(printf '%s' "${BACKEND}" | tr '[:lower:]' '[:upper:]')"
case "${BACKEND_UPPER}" in
    METAL) RUNTIME_BACKEND=metal ;;
    CUDA) RUNTIME_BACKEND=cuda ;;
    *) echo "LAYA_BACKEND must be METAL or CUDA" >&2; exit 2 ;;
esac
if [[ "$(uname -s)" == "Darwin" && "${BACKEND_UPPER}" != "METAL" ]]; then
    echo "macOS requires LAYA_BACKEND=METAL" >&2; exit 2
fi
if [[ "$(uname -s)" == "Linux" && "${BACKEND_UPPER}" != "CUDA" ]]; then
    echo "Linux requires LAYA_BACKEND=CUDA" >&2; exit 2
fi

BUILD_DIR="${LAYA_BUILD_DIR:-${SCRIPT_DIR}/build}"
SDK_BUILD_DIR="${BUILD_DIR}/sdk"
MODEL_DIR="${LAYA_MODEL_DIR:-${MODEL_DEFAULT}}"
RUNS="${LAYA_RUNS:-5}"
WARMUP_RUNS="${LAYA_WARMUP_RUNS:-2}"
FORMAT="${LAYA_FORMAT:-text}"
BUILD_TYPE="${LAYA_BUILD_TYPE:-Release}"
TIMING="${LAYA_TIMING:-OFF}"
CC="${LAYA_CC:-cc}"
SDK_CFLAGS="${LAYA_SDK_CFLAGS:-}"
SDK_CXXFLAGS="${LAYA_SDK_CXXFLAGS:-}"
SDK_CMAKE_ARGS="${LAYA_CMAKE_ARGS:-}"

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Laya multilingual model directory not found: ${MODEL_DIR}" >&2; exit 1
fi
for file in model.safetensors rl_agent_config.json encoder/config.json tokenizer/tokenizer.json; do
    if [[ ! -f "${MODEL_DIR}/${file}" ]]; then
        echo "required model file not found: ${MODEL_DIR}/${file}" >&2; exit 1
    fi
done
if [[ ! "${RUNS}" =~ ^[1-9][0-9]*$ || ! "${WARMUP_RUNS}" =~ ^[0-9]+$ ]]; then
    echo "LAYA_RUNS must be positive and LAYA_WARMUP_RUNS non-negative" >&2; exit 2
fi
if [[ "${FORMAT}" != "text" && "${FORMAT}" != "json" ]]; then
    echo "LAYA_FORMAT must be text or json" >&2; exit 2
fi

echo "--- Native Laya build: backend=${BACKEND_UPPER}, model=${MODEL_DIR}, runs=${RUNS}, warmups=${WARMUP_RUNS} ---"
make -C "${SCRIPT_DIR}" all \
    BACKEND="${BACKEND_UPPER}" \
    BUILD_TYPE="${BUILD_TYPE}" \
    TIMING="${TIMING}" \
    CC="${CC}" \
    CFLAGS="${LAYA_CFLAGS:--O3}" \
    SDK_CFLAGS="${SDK_CFLAGS}" \
    SDK_CXXFLAGS="${SDK_CXXFLAGS}" \
    SDK_CMAKE_ARGS="${SDK_CMAKE_ARGS}" \
    SDK_BUILD_DIR="${SDK_BUILD_DIR}" \
    TARGET="${BUILD_DIR}/laya"

runtime_env=("TENSOR_DTYPE=fp16" "TENSOR_BACKEND=${RUNTIME_BACKEND}" "TENSOR_ATTENTION_BACKEND=${RUNTIME_BACKEND}")
if [[ "${RUNTIME_BACKEND}" == "metal" ]]; then
    runtime_env+=(
        "TENSOR_METALLIB_PATH=${SDK_BUILD_DIR}/metal/attention.metallib"
        "TENSOR_KERNELS_METALLIB_PATH=${SDK_BUILD_DIR}/metal/tensor_kernels.metallib"
    )
fi
env "${runtime_env[@]}" "${BUILD_DIR}/laya" "${MODEL_DIR}" "${RUNS}" "${WARMUP_RUNS}" "${FORMAT}"
