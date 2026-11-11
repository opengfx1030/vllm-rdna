#!/usr/bin/env bash
# M2: switch-on then switch-off serve on the same GPUs (gates + logprob dump),
# then the logprob parity comparison. Waits for the GPUs to be empty first.
#   GPUS=6,7,8,9 PORT=18790 bash tools/rdna/mono/m2_pair.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=${RUN:-$HOME/w4a8_runs/monokernel-2a}
: "${GPUS:?}"
wait_idle() {  # rocm-smi GPU[0-3] are HIP 6-9 on this box
    for _ in $(seq 1 120); do
        busy=$(rocm-smi --showmemuse 2>/dev/null | awk '/GPU\[[0-3]\].*VRAM%/{s+=$NF} END{print s+0}')
        [[ $busy -eq 0 ]] && return 0
        sleep 10
    done
    echo "GPUs not idle"; return 1
}
for arm in on off; do
    wait_idle || exit 1
    tag=m2b-$arm
    echo "== $(date +%H:%M:%S) $tag"
    ARM=$arm TAG=$tag CELLS=none bash "$HERE/serve_2a.sh" > "$RUN/$tag.out" 2>&1
    grep -E "READY|FAILED|PROBE|PPL|PREFIX|cold TTFT|CONCURRENCY|RESULT|LOAD HOOK|mono k2_rdna2" \
        "$RUN/serve-$tag/summary.txt" | cut -c1-200
    sleep 20
done
python3 "$HERE/logprob_parity.py" compare "$RUN/serve-m2b-off/parity.json" "$RUN/serve-m2b-on/parity.json" \
    | tee "$RUN/m2b_parity.txt"
