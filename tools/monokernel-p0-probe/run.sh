#!/usr/bin/env bash
# Runs both variants, records environment + results to a timestamped log.
# Extra args are passed to the probe (e.g. --lds 32768 --replays 500).
set -uo pipefail
cd "$(dirname "$0")"
ROCM=${ROCM_PATH:-/opt/rocm}
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p logs; LOG=logs/p0-$TS.log
exec > >(tee "$LOG") 2>&1
echo "# MonoKernel P0 probe  $(date -Is)  host=$(hostname)"
echo "## versions"
cat "$ROCM/.info/version" 2>/dev/null || echo "rocm .info/version missing"
"$ROCM/bin/hipcc" --version 2>&1
"$ROCM/llvm/bin/clang" --version 2>&1 | head -2
echo "kernel: $(uname -r)"
modinfo amdgpu 2>/dev/null | grep -E '^(version|filename|srcversion)' || echo "amdgpu modinfo n/a (in-tree driver?)"
cat /sys/module/amdgpu/version 2>/dev/null || true
echo "## rocminfo (gfx agents)"
"$ROCM/bin/rocminfo" 2>&1 | awk '/^Agent /{a=$0} /Marketing Name|Name: +gfx|Compute Unit|Max Waves Per CU|Workgroup Max Size|Wavefront Size|Max Clock|GROUP segment|Size:.*KB|Segment:/{print a" | "$0}' | head -200
echo "## env"; env | grep -E '^(HIP|HSA|ROC|AMD|GPU_)' || true
for v in wgp cumode; do
  echo; echo "################ variant: $v  flags: $(cat build/$v/flags.txt 2>/dev/null)"
  echo "## code-object metadata"; cat build/$v/kernel_meta.txt 2>/dev/null
  echo "## .amdhsa directives (incl. float_denorm_mode, workgroup_processor_mode)"; cat build/$v/amdhsa_directives.txt 2>/dev/null
  echo "## run (residency + graph)"
  timeout --signal=KILL 300 build/$v/probe --only residency "$@"; echo "exit=$?"
  timeout --signal=KILL 300 build/$v/probe --only graph "$@"; echo "exit=$?"
  for o in busy-first flag-first; do
    echo "## run progress order=$o"
    timeout --signal=KILL 300 build/$v/probe --only progress --order $o "$@"; echo "exit=$?"
  done
done
echo; echo "log: $LOG"
