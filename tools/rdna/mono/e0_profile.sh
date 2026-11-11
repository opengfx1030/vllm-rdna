#!/usr/bin/env bash
# E0: eager torch-profiler serves (kernel durations do not depend on launch
# mode) of today's path, MTP2 then MTP0, for the attention-block breakdown.
#   GPUS=2,3,4,5 PORT=18791 bash tools/rdna/mono/e0_profile.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
P0=$HERE/../monokernel-p0-probe
export RUN=${RUN:-$HOME/w4a8_runs/monokernel-2a/e0}
mkdir -p "$RUN"
export MODEL=${MODEL:-/home/chenco_adm/hfcache/hub/models--wtdcode--Qwen3.8-Flash-Next-AWQ-W4A16/snapshots/0939125b929543a783ce700c90e36dd1a575c00c}
export PLE_QUANT_DIR=${PLE_QUANT_DIR:-/home/chenco_adm/hfcache/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/snapshots/4f861b63f69e61bfc2e22130ec91ec67f03ec43e/ples_int4}
: "${GPUS:?}"
for r in mtp2 mtp0; do
    echo "== $(date +%H:%M:%S) e0-$r"
    RECIPE=flashnext-$r TAG=e0-$r CG_MODE=NONE PROF=torch CELLS=none PPL_PROBE=0 \
        PREFIX_PROBE=0 CONC_PROBE=0 bash "$P0/serve_p0b.sh" > "$RUN/e0-$r.out" 2>&1
    grep -E "READY|FAILED|LOAD HOOK|PASS|FAIL" "$RUN/serve-e0-$r/summary.txt" | head -5
done
