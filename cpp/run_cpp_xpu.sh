#!/usr/bin/env bash
set -euo pipefail

# Run from: Road_Vision/cpp
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: venv not active. Run: source <venv>/bin/activate"
  exit 1
fi

VENV="$VIRTUAL_ENV"
TORCH_LIB="$VENV/lib/python3.12/site-packages/torch/lib"

UR_DIR="${UR_LOADER_DIR:-$VENV/lib}"
SYCL_DIR="${SYCL_RUNTIME_DIR:-$VENV/lib}"

GOOD_UR="${UR_DIR}/libur_loader.so"
if [[ ! -f "$GOOD_UR" ]]; then
  echo "ERROR: libur_loader.so not found at: $GOOD_UR"
  exit 1
fi

export LD_PRELOAD="$GOOD_UR${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="$VENV/lib:$TORCH_LIB:$SYCL_DIR:$UR_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Optional: pin device selection
# export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
# export SYCL_DEVICE_FILTER="level_zero:gpu"

exec "$ROOT/build/road_vision_xpu" "$@"
