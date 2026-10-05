#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Library/Android/sdk}}"
QAIRT_ROOT="${QAIRT_SDK_ROOT:-$HOME/qairt/2.50.0.260828}"
GRADLE_BIN="${GRADLE:-gradle}"
ADB="${ADB:-${SDK_ROOT}/platform-tools/adb}"
DEVICE_SERIAL="${ANDROID_SERIAL:-}"
ACTION="${1:-build}"

case "${ACTION}" in
    build|install|run) ;;
    --help|-h)
        cat <<'HELP'
Usage: tests/smolvlm/android/build.sh [build|install|run]

Environment:
  ANDROID_SDK_ROOT / ANDROID_HOME  Android SDK (default: ~/Library/Android/sdk)
  QAIRT_SDK_ROOT                  QAIRT SDK (default: ~/qairt/2.50.0.260828)
  SMOLVLM_MODEL_PATH              Safetensors checkpoint embedded in the APK
  GRADLE                          Gradle executable (default: gradle)
  ADB                             adb executable (default: SDK platform-tools/adb)
  ANDROID_SERIAL                  Optional target device serial

Actions:
  build    Assemble debug APK
  install  Build and install on the connected device
  run      Build, install, and launch camera-to-caption SmolVLM demo

Usage flow: build with the checkpoint path, install, capture a photo, then run
SmolVLM. The APK embeds the checkpoint. Check README.md for supported paths
and current device-validation status.
HELP
        exit 0
        ;;
    *) echo "Unknown action: ${ACTION}" >&2; exit 2 ;;
esac

if [[ ! -d "${SDK_ROOT}" ]]; then echo "Android SDK not found: ${SDK_ROOT}" >&2; exit 1; fi
if [[ ! -f "${QAIRT_ROOT}/include/QNN/QnnInterface.h" ]]; then echo "QAIRT SDK not found: ${QAIRT_ROOT}" >&2; exit 1; fi

export ANDROID_HOME="${SDK_ROOT}"
export ANDROID_SDK_ROOT="${SDK_ROOT}"
export QAIRT_SDK_ROOT="${QAIRT_ROOT}"
"${GRADLE_BIN}" --no-daemon -p "${SCRIPT_DIR}" assembleDebug

if [[ "${ACTION}" == "build" ]]; then
    echo "APK: ${SCRIPT_DIR}/app/build/outputs/apk/debug/app-debug.apk"
    exit 0
fi

APK="${SCRIPT_DIR}/app/build/outputs/apk/debug/app-debug.apk"
if [[ -n "${DEVICE_SERIAL}" ]]; then
    "${ADB}" -s "${DEVICE_SERIAL}" install -r --no-streaming "${APK}"
    if [[ "${ACTION}" == "run" ]]; then
        "${ADB}" -s "${DEVICE_SERIAL}" shell am force-stop ai.diffstep.smolvlm
        "${ADB}" -s "${DEVICE_SERIAL}" shell am start -n ai.diffstep.smolvlm/.MainActivity
    fi
else
    "${ADB}" install -r --no-streaming "${APK}"
    if [[ "${ACTION}" == "run" ]]; then
        "${ADB}" shell am force-stop ai.diffstep.smolvlm
        "${ADB}" shell am start -n ai.diffstep.smolvlm/.MainActivity
    fi
fi
