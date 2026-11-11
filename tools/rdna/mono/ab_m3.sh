#!/usr/bin/env bash
# M3 serve A/B, MonoKernel switch off vs on, alternating:
#   MTP2: off on off on    MTP0: off on
# Four cells per serve at --temperature 0, plus the serve_validate gates.
#   GPUS=6,7,8,9 PORT=18790 bash tools/rdna/mono/ab_m3.sh
# Results: ~/w4a8_runs/monokernel-2a/serve-m3-<recipe>-<arm>-<rep>/; table via ab_table.py.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${GPUS:?}"
CELLS=${CELLS:-"1024:512:1 1024:512:8 16384:1024:1 16384:1024:8"}
ORDER=${ORDER:-"mtp2:off:a mtp2:on:a mtp2:off:b mtp2:on:b mtp0:off:a mtp0:on:a"}
wait_idle() {  # rocm-smi GPU[0-3] are HIP 6-9 on this box; the last server may still be exiting
    for _ in $(seq 1 120); do
        busy=$(rocm-smi --showmemuse 2>/dev/null | awk '/GPU\[[0-3]\].*VRAM%/{s+=$NF} END{print s+0}')
        [[ $busy -eq 0 ]] && return 0
        sleep 10
    done
    return 1
}
for spec in $ORDER; do
    IFS=: read -r mtp arm rep <<< "$spec"
    wait_idle || { echo "GPUs not idle"; exit 1; }
    tag=${TAGP:-m3}-$mtp-$arm-$rep
    echo "== $(date +%H:%M:%S) $tag"
    ARM=$arm RECIPE=flashnext-$mtp TAG=$tag CACHE=$HOME/w4a8_runs/monokernel-2a/cache-2a-$arm \
        CELLS="$CELLS" BENCH_EXTRA="--temperature 0" LOAD_HOOK=${LOAD_HOOK:-$HERE/parity_hook.sh} \
        bash "$HERE/serve_2a.sh" > "$HOME/w4a8_runs/monokernel-2a/$tag.out" 2>&1
    grep -E "READY|FAILED|DIED|bench-|PASS|FAIL" "$HOME/w4a8_runs/monokernel-2a/serve-$tag/summary.txt" \
        | grep -vE "^\s+(PASS|FAIL)  prompt" | cut -c1-200
done
python3 "$HERE/ab_table.py" "$HOME"/w4a8_runs/monokernel-2a/serve-m3-*
