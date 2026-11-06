#!/usr/bin/env bash
# M1 standalone run on one V620: builds the kernel as a JIT extension and runs
# m1_bench.py against this tree's vLLM ops.
#   GPU=6 OUT=~/w4a8_runs/monokernel-2a/m1 bash tools/rdna/mono/run_m1.sh [m1_bench args]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../../.." && pwd)
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final}
OUT=${OUT:-$HOME/w4a8_runs/monokernel-2a/m1}
GPU=${GPU:-6}
# HIP 6-9 under GPU69_LOCK; HIP 3-5 only while we hold GPU25_LOCK; never 0-2.
case ",$GPU," in *,0,* | *,1,* | *,2,*) echo "never HIP 0-2"; exit 2 ;; esac
case ",$GPU," in *,3,* | *,4,* | *,5,*)
    grep -qx monokernel-2a "$HOME/w4a8_runs/port-v031/GPU25_LOCK" 2>/dev/null \
        || { echo "HIP 3-5 need our GPU25_LOCK"; exit 2; } ;;
esac
SCRIPT=${SCRIPT:-$HERE/m1_bench.py}
mkdir -p "$OUT" "$OUT/ext_build" "$OUT/run"
SITE=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib:$SITE/torch/lib
export PYTHONPATH=$TREE ROCM_HOME=/opt/rocm/core-7.14 ROCM_PATH=/opt/rocm/core-7.14
export PATH=/opt/rocm/core-7.14/bin:$VENV/bin:$PATH
export HIP_VISIBLE_DEVICES=$GPU MONO_BUILD_DIR=$OUT/ext_build PYTORCH_ROCM_ARCH=gfx1030
export HSA_FORCE_FINE_GRAIN_PCIE=${HSA_FORCE_FINE_GRAIN_PCIE:-1} HSA_ENABLE_SDMA=0 GPU_MAX_HW_QUEUES=2 VLLM_RDNA_FORCE_FP16=1
cd "$OUT/run"
exec "$VENV/bin/python" "$SCRIPT" "$@"
