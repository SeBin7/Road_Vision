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
TORCH_DIR="${TORCH_DIR:-$VENV/lib/python3.12/site-packages/torch/share/cmake/Torch}"

# Pick a UR loader that actually exports the symbols our SYCL runtime expects.
# Torch-shipped libsycl.so.8 currently requires these UR experimental symbols.
need_syms=(
  urKernelSuggestMaxCooperativeGroupCountExp
  urEnqueueKernelLaunchCustomExp
  urEnqueueCooperativeKernelLaunchExp
)

has_all_syms() {
  local so="$1"
  [[ -f "$so" ]] || return 1
  for s in "${need_syms[@]}"; do
    nm -D "$so" 2>/dev/null | grep -q "$s" || return 1
  done
  return 0
}

UR_DIR="$VENV/lib"
if ! has_all_syms "$UR_DIR/libur_loader.so"; then
  for d in     /opt/intel/oneapi/compiler/2025.3/lib     /opt/intel/oneapi/2025.3/lib     /opt/intel/oneapi/compiler/latest/lib     /opt/intel/oneapi/latest/lib
  do
    if has_all_syms "$d/libur_loader.so"; then
      UR_DIR="$d"
      break
    fi
  done
fi

SYCL_DIR="$VENV/lib"
if [[ ! -f "$SYCL_DIR/libsycl.so.8" ]]; then
  for d in     /opt/intel/oneapi/compiler/2025.3/lib     /opt/intel/oneapi/compiler/latest/lib
  do
    if [[ -f "$d/libsycl.so" || -f "$d/libsycl.so.8" ]]; then
      SYCL_DIR="$d"
      break
    fi
  done
fi

echo "[build] VENV      = $VENV"
echo "[build] Torch_DIR  = $TORCH_DIR"
echo "[build] SYCL_DIR   = $SYCL_DIR"
echo "[build] UR_DIR     = $UR_DIR"

rm -rf build
cmake -S . -B build   -DCMAKE_BUILD_TYPE=Release   -DPython3_EXECUTABLE="$VENV/bin/python"   -DTorch_DIR="$TORCH_DIR"   -DRV_VENV_ROOT="$VENV"   -DSYCL_RUNTIME_DIR="$SYCL_DIR"   -DUR_LOADER_DIR="$UR_DIR"

cmake --build build -j
