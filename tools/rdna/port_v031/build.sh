#!/usr/bin/env bash
# Build the rdna_extra/v0.31.0 tree into its own venv on the gfx1030 box.
#
#   bash tools/rdna/port_v031/build.sh            # clone venv (once) + build
#   CLEAN=1 bash tools/rdna/port_v031/build.sh    # wipe build artifacts first
#
# Clones venv-7.14.0_0.28.0 (same torch 2.12.0+rocm7.14.0) into
# venv-7.14.0_0.31.0, applies the v0.31 requirement deltas, then does the
# editable install from this tree. Logs go to ~/w4a8_runs/port-v031/.
set -euo pipefail

TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SRC_VENV=${SRC_VENV:-$HOME/Apps/vllm/venv-7.14.0_0.28.0}
VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0}
LOGDIR=${LOGDIR:-$HOME/w4a8_runs/port-v031}
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/build.log") 2>&1
echo "=== build $(date -Is) tree=$TREE venv=$VENV"

if [[ ! -x $VENV/bin/python ]]; then
    echo "--- cloning $SRC_VENV -> $VENV"
    cp -a "$SRC_VENV" "$VENV"
    # Re-point the venv: shebangs and activate scripts hardcode the old path.
    grep -rlI --exclude-dir=site-packages "$SRC_VENV" "$VENV/bin" | while read -r f; do
        sed -i "s#$SRC_VENV#$VENV#g" "$f"
    done
    sed -i "s#venv-7.14.0\$#$(basename "$VENV")#" "$VENV/pyvenv.cfg" || true
fi

PY=$VENV/bin/python
"$PY" -m pip install -q 'huggingface_hub>=1.31.0' 'xgrammar==0.2.7' 'oss-harmony>=0.0.11'

cd "$TREE"
if [[ ${CLEAN:-0} == 1 ]]; then
    rm -rf build/ .deps/ vllm/*.abi3.so
fi

export SETUPTOOLS_SCM_PRETEND_VERSION=0.31.0+rdna
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx1030
export VLLM_PYTHON_EXECUTABLE=$PY
export MAX_JOBS=${MAX_JOBS:-32}
export CMAKE_BUILD_TYPE=RelWithDebInfo
export CMAKE_HIP_COMPILER=/opt/rocm/core-7.14/bin/hipcc
export ROCM_HOME=/opt/rocm/core-7.14 ROCM_PATH=/opt/rocm/core-7.14
export HIP_PATH=/opt/rocm/core-7.14 HIP_ROOT_DIR=/opt/rocm/core-7.14
export CMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-7.14
export PATH=/opt/rocm/core-7.14/bin:$VENV/bin:$PATH

t0=$(date +%s)
"$PY" -m pip install -e . --no-build-isolation --no-deps
echo "--- build took $(( $(date +%s) - t0 ))s"

SITE=$VENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SITE/_rocm_sdk_libraries/lib:$SITE/_rocm_sdk_core/lib/host-math/lib:$SITE/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SITE/_rocm_sdk_core/lib/core/lib:$SITE/torch/lib
echo "--- unresolved libs of _rocm_C:"
ldd vllm/_rocm_C.abi3.so | grep "not found" || echo "none"
cd /  # avoid importing the tree's vllm/entrypoints/cli/openai.py as openai
HIP_VISIBLE_DEVICES=${CHECK_GPU:-9} "$PY" - <<'EOF'
import torch
import vllm._custom_ops  # noqa: F401  (loads _C / _rocm_C)
schemas = [str(s) for s in torch._C._jit_get_all_schemas()]
rocm = [s for s in schemas if s.startswith("_rocm_C::")]
rdna = [s for s in rocm if "rdna" in s]
assert len(rdna) > 10, f"Expected >10 RDNA ops, got {len(rdna)}"
assert any("all_reduce" in s for s in rocm), "all_reduce not registered"
print(f"REGISTRATION OK: {len(rocm)} _rocm_C ops ({len(rdna)} rdna)")
EOF
echo "=== build done $(date -Is)"
