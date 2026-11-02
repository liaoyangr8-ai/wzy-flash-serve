#!/usr/bin/env bash
# build.sh -- Compile wzy-flash-serve pure-CUDA/C++ dual-GPU SM75 inference engine
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p bin data

NVCC="${NVCC:-/home/liaoyang/ff75/third_party_env/cuda-12.8/bin/nvcc}"
if [ ! -x "$NVCC" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
  else
    echo "[build.sh] ERROR: nvcc not found. Set NVCC=/path/to/nvcc" >&2
    exit 1
  fi
fi

echo "[build.sh] Compiling src/wzy_engine_v2.cu with $NVCC (-O3 -std=c++17 -arch=sm_75 -mavx2 -mfma)..."
"$NVCC" -O3 -std=c++17 -arch=sm_75 -Wno-deprecated-gpu-targets -DWZY_GRP_MAX=${WZY_GRP_MAX:-8} -Xcompiler -mavx2 -Xcompiler -mfma -Xcompiler -pthread src/wzy_engine_v2.cu -lcublas -o bin/wzy_pp2_engine.new

if [ ! -f bin/wzy_pp2_engine ] || ! pgrep -x wzy_pp2_engine >/dev/null 2>&1; then
  mv -f bin/wzy_pp2_engine.new bin/wzy_pp2_engine
  echo "[build.sh] Installed bin/wzy_pp2_engine:"
  ls -lh bin/wzy_pp2_engine
else
  echo "[build.sh] Engine is currently running; compiled binary staged at bin/wzy_pp2_engine.new"
  echo "           Run: ./stop.sh && mv -f bin/wzy_pp2_engine.new bin/wzy_pp2_engine && ./start.sh"
  ls -lh bin/wzy_pp2_engine.new
fi
