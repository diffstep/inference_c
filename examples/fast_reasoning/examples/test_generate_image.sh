#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/examples/sample-model"
IMAGE_PATH="${REPO_ROOT}/examples/test-data/zxqh/sample5.jpg"
BIN_PATH="${REPO_ROOT}/target/release/fast-reasoning"

if [[ ! -x "${BIN_PATH}" ]]; then
  printf 'Release binary not found: %s\nBuild it with: cargo build --release\n' "${BIN_PATH}" >&2
  exit 127
fi

export FAST_REASONING_DEVICE="${FAST_REASONING_DEVICE:-metal}"
exec "${BIN_PATH}" generate-image "${MODEL_DIR}" "${IMAGE_PATH}" "$@"
